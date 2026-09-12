"""Human-in-the-loop escalation: what happens when replay cannot proceed safely.

`src/replay/engine.py` is designed to run unattended, but two situations
should never be handled by an unattended heuristic: a step the artifact's
author flagged as `risky`/`irreversible` (submitting a payment, deleting a
record), and a hard failure the deterministic replay logic could not recover
from on its own. In both cases the right move is to stop touching the page
and put a human in front of it.

The control-transfer model this module implements is deliberately narrow:

1. The replay engine calls :func:`escalate`, which freezes the current
   Playwright page (no further navigation/actions), captures enough state
   for a human to orient themselves (screenshot, URL, step history, a CDP
   endpoint they can attach DevTools to), and blocks.
2. A human opens the operator UI (`src/handoff/operator_server.py`), reviews
   the intervention, and either clicks **Resume Automation** directly (a
   bare approval — the engine will still perform the pending step itself)
   or **Take Control** first, manually fixes/performs the action in the
   *same visible browser window* Playwright already has open, and then
   clicks **Resume Automation**.
3. :func:`escalate` returns an :class:`EscalationOutcome` whose
   `human_took_control` flag is the entire handoff contract: it tells the
   caller whether the pending action still needs to be performed by
   automation, or was already handled by the human and should not be
   repeated (double-submitting a payment because both the human and the
   engine tried to click "Confirm" would be worse than the original
   failure).

There is deliberately no timeout on the wait: an unattended timeout that
silently resumes or abandons a risky/irreversible action defeats the entire
point of asking a human first.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import time
import urllib.request
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ControlState",
    "EscalationReason",
    "InterventionRequest",
    "EscalationOutcome",
    "EscalationHub",
    "get_hub",
    "escalate",
    "DEFAULT_OPERATOR_PORT",
    "DEFAULT_CDP_PORT",
]

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ESCALATION_EVIDENCE_DIR = REPO_ROOT / "evidence" / "escalation"

DEFAULT_OPERATOR_PORT = 8080
DEFAULT_CDP_PORT = 9222

_CDP_HTTP_TIMEOUT_S = 2
_SERVER_START_POLL_INTERVAL_S = 0.05
_SERVER_START_POLL_ATTEMPTS = 100  # ~5s max wait for uvicorn to bind
_DIFF_SUMMARY_MAX_LINES = 10


# --------------------------------------------------------------------------
# Control-state / reason vocabulary
# --------------------------------------------------------------------------


class ControlState(str, Enum):
    """Who is driving the browser right now.

    AUTOMATION -> PAUSED -> (HUMAN_CONTROL, optional) -> AUTOMATION is the
    normal cycle, once per escalation. COMPLETED is terminal, set once the
    whole replay run ends (see `EscalationHub.mark_completed`).
    """

    AUTOMATION = "AUTOMATION"
    PAUSED = "PAUSED"
    HUMAN_CONTROL = "HUMAN_CONTROL"
    COMPLETED = "COMPLETED"


class EscalationReason(str, Enum):
    """Why the replay engine stopped and asked for a human."""

    STUCK = "stuck"
    RISKY_ACTION = "risky_action"
    HARD_FAILURE = "hard_failure"
    UNKNOWN_STATE = "unknown_state"


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------


class InterventionRequest(BaseModel):
    """Everything a human needs to understand and act on one pause.

    `control_state` is mutated in place on this same instance as the pause
    progresses (PAUSED -> HUMAN_CONTROL -> AUTOMATION) rather than replaced,
    because the operator UI polls `GET /api/intervention` repeatedly across
    a single pause and reads this field to render its status line.
    """

    model_config = ConfigDict(extra="forbid")

    request_id: UUID = Field(default_factory=uuid4)
    capability_name: str = Field(..., min_length=1)
    current_step_id: str = Field(..., min_length=1)
    reason: EscalationReason
    description: str = Field(..., min_length=1, description="Human-readable explanation of why replay stopped.")
    screenshot_path: str
    current_url: str
    timestamp: datetime
    control_state: ControlState = Field(default=ControlState.PAUSED)
    step_history: list[dict[str, Any]] = Field(
        default_factory=list, description="Execution log of every step attempted so far this run."
    )
    cdp_websocket_url: str | None = Field(
        default=None,
        description="Chrome DevTools Protocol websocket endpoint a human can attach DevTools or a "
        "separate Playwright session to, in addition to interacting with the visible window directly.",
    )
    window_info: dict[str, Any] | None = Field(
        default=None, description="Best-effort CDP window id/bounds for the automation's browser window."
    )


class EscalationOutcome(BaseModel):
    """What happened during one pause, once the human hands control back."""

    model_config = ConfigDict(extra="forbid")

    request_id: UUID
    human_took_control: bool = Field(
        ..., description="True if the human clicked Take Control before Resume — see module docstring."
    )
    duration_seconds: float = Field(..., ge=0)
    url_before: str
    url_after: str
    url_changed: bool
    content_diff_summary: str | None = Field(
        default=None, description="Truncated unified diff of visible page text before vs. after the pause."
    )


# --------------------------------------------------------------------------
# Coordination hub
#
# One instance per process, shared between the replay engine (this module's
# `escalate()`) and the operator server's HTTP handlers. Both sides run on
# the SAME asyncio event loop by construction (see `ensure_server_started`),
# which is what makes the plain (non-locked) attribute reads/writes below
# safe: asyncio.Event is not thread-safe, and FastAPI's `async def` handlers
# execute directly on the loop that received the request rather than in a
# worker thread, so as long as uvicorn's `Server.serve()` is scheduled as a
# task on this same loop instead of getting its own, there is only ever one
# thread touching this object.
# --------------------------------------------------------------------------


class EscalationHub:
    """Coordinates one replay run's escalations with the operator server."""

    def __init__(self) -> None:
        self.control_state: ControlState = ControlState.AUTOMATION
        self.current_intervention: InterventionRequest | None = None
        self.current_screenshot_path: Path | None = None
        self.history: list[InterventionRequest] = []
        self._resume_event = asyncio.Event()
        self._human_took_control = False
        self._server: Any | None = None
        self._server_task: asyncio.Task[None] | None = None

    @property
    def server_running(self) -> bool:
        return self._server_task is not None and not self._server_task.done()

    async def ensure_server_started(self, port: int = DEFAULT_OPERATOR_PORT) -> None:
        """Start the operator server once; a no-op if it's already up.

        Multiple escalations within one replay run share a single server
        instance instead of one per pause, so the operator's browser tab
        doesn't need to reconnect to a new port each time.
        """
        if self.server_running:
            return

        import uvicorn

        from .operator_server import app

        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        server = uvicorn.Server(config)
        self._server = server
        self._server_task = asyncio.create_task(server.serve())

        for _ in range(_SERVER_START_POLL_ATTEMPTS):
            if server.started:
                return
            await asyncio.sleep(_SERVER_START_POLL_INTERVAL_S)
        logger.warning("operator server on port %d did not confirm startup in time; continuing anyway", port)

    async def shutdown_server(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            try:
                await asyncio.wait_for(self._server_task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._server_task.cancel()
        self._server = None
        self._server_task = None

    def begin_pause(self, intervention: InterventionRequest) -> None:
        self.control_state = ControlState.PAUSED
        intervention.control_state = ControlState.PAUSED
        self.current_intervention = intervention
        self.current_screenshot_path = Path(intervention.screenshot_path)
        self._human_took_control = False
        self._resume_event = asyncio.Event()

    def take_control(self) -> None:
        self.control_state = ControlState.HUMAN_CONTROL
        self._human_took_control = True
        if self.current_intervention is not None:
            self.current_intervention.control_state = ControlState.HUMAN_CONTROL

    def signal_resume(self) -> None:
        self._resume_event.set()

    async def wait_for_resume(self) -> bool:
        """Block until the operator calls `POST /api/resume`.

        Returns whether the human took control at any point during this
        pause — the flag :func:`escalate`'s caller needs to decide whether
        to still perform the pending step itself.
        """
        await self._resume_event.wait()
        return self._human_took_control

    def end_pause(self) -> None:
        self.control_state = ControlState.AUTOMATION
        if self.current_intervention is not None:
            self.current_intervention.control_state = ControlState.AUTOMATION
            self.history.append(self.current_intervention)
        self.current_intervention = None

    def mark_completed(self) -> None:
        self.control_state = ControlState.COMPLETED
        self.current_intervention = None


_hub: EscalationHub | None = None


def get_hub() -> EscalationHub:
    """Process-wide singleton so the replay engine and operator server agree on state."""
    global _hub
    if _hub is None:
        _hub = EscalationHub()
    return _hub


# --------------------------------------------------------------------------
# Browser exposure: CDP websocket endpoint + window info
# --------------------------------------------------------------------------


def _fetch_json_sync(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=_CDP_HTTP_TIMEOUT_S) as resp:  # localhost-only CDP endpoint
        return json.loads(resp.read().decode("utf-8"))


async def _get_cdp_info(cdp_port: int, page_url: str) -> dict[str, Any]:
    """Best-effort lookup of the browser's Chrome DevTools Protocol endpoint.

    Chromium exposes this over plain HTTP on `--remote-debugging-port`
    independent of Playwright's own driver connection, so a human can attach
    a *separate* DevTools tab or Playwright session to inspect/drive the
    same browser without disturbing what's already open. This only works if
    the browser was launched with that flag (see `ReplayConfig` in
    `src/replay/engine.py`, which sets it whenever escalation is enabled);
    failure here is non-fatal since the human can always act on the visible
    window directly.
    """
    try:
        version = await asyncio.to_thread(_fetch_json_sync, f"http://127.0.0.1:{cdp_port}/json/version")
        targets = await asyncio.to_thread(_fetch_json_sync, f"http://127.0.0.1:{cdp_port}/json/list")
    except Exception:
        logger.warning(
            "could not reach chromium's CDP endpoint on port %d; the human will need to use the "
            "visible browser window directly rather than attaching DevTools remotely",
            cdp_port,
            exc_info=True,
        )
        return {}

    matching_target = next((t for t in targets if t.get("url") == page_url), None)
    return {
        "browser_websocket_url": version.get("webSocketDebuggerUrl"),
        "page_websocket_url": (matching_target or {}).get("webSocketDebuggerUrl"),
        "devtools_frontend_url": (matching_target or {}).get("devtoolsFrontendUrl"),
    }


async def _get_window_info(page: Page) -> dict[str, Any]:
    """Best-effort OS window id/bounds so a human knows which window to focus.

    `Browser.getWindowForTarget` is a Browser-domain CDP command; Chromium
    forwards it from a page-scoped session, but that isn't guaranteed across
    every Chromium build, so this degrades to an empty dict rather than
    raising. Either way the fallback instruction is the same: look for the
    visible, automation-controlled Chromium window on this machine.
    """
    try:
        cdp = await page.context.new_cdp_session(page)
        try:
            window = await cdp.send("Browser.getWindowForTarget")
        finally:
            await cdp.detach()
        logger.info(
            "automation browser window: windowId=%s bounds=%s", window.get("windowId"), window.get("bounds")
        )
        return {"window_id": window.get("windowId"), "bounds": window.get("bounds")}
    except Exception:
        logger.warning(
            "could not determine browser window bounds via CDP; look for the visible "
            "Chromium window this automation opened",
            exc_info=True,
        )
        return {}


# --------------------------------------------------------------------------
# State capture
# --------------------------------------------------------------------------


async def _capture_state(page: Page, evidence_dir: Path, capability_name: str, tag: str) -> tuple[str, str]:
    """Screenshot + visible page text, for both the pause moment and the resume moment.

    A full-page screenshot can time out on some pages (font loading, layout
    thrash, an in-flight navigation) — observed in practice right after a
    human resumes a run mid-page-transition. Retrying once without
    `full_page` covers that case; if even that fails, an empty
    `screenshot_path` degrades the operator UI to "no image" rather than
    crashing a run a human has already handed back to automation.
    """
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = evidence_dir / f"{capability_name}_{tag}_{timestamp}.png"
    screenshot_path = ""
    try:
        await page.screenshot(path=str(path), full_page=True)
        screenshot_path = str(path)
    except Exception:
        logger.warning("full-page screenshot failed for %s; retrying without full_page", tag, exc_info=True)
        try:
            await page.screenshot(path=str(path))
            screenshot_path = str(path)
        except Exception:
            logger.warning("screenshot capture failed entirely for %s", tag, exc_info=True)
    try:
        text = await page.inner_text("body")
    except Exception:
        text = ""
    return screenshot_path, text


def _summarize_diff(before: str, after: str, max_lines: int = _DIFF_SUMMARY_MAX_LINES) -> str | None:
    if before == after:
        return None
    diff = list(difflib.unified_diff(before.splitlines(), after.splitlines(), lineterm="", n=0))
    if not diff:
        return None
    truncated = diff[:max_lines]
    suffix = "\n... (truncated)" if len(diff) > max_lines else ""
    return "\n".join(truncated) + suffix


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


async def escalate(
    *,
    page: Page,
    capability_name: str,
    step_id: str,
    reason: EscalationReason,
    description: str,
    step_history: list[dict[str, Any]] | None = None,
    hub: EscalationHub | None = None,
    evidence_dir: Path = DEFAULT_ESCALATION_EVIDENCE_DIR,
    cdp_port: int = DEFAULT_CDP_PORT,
    operator_port: int = DEFAULT_OPERATOR_PORT,
) -> EscalationOutcome:
    """Pause automation, hand the browser to a human, and wait to be resumed.

    See the module docstring for the full control-transfer contract. In
    short: this coroutine never touches `page` itself (no navigation, no
    closing) so the human sees exactly the state the engine stopped in, and
    it does not return until `POST /api/resume` is called — there is no
    timeout, by design.
    """
    hub = hub or get_hub()
    started_at = datetime.now(timezone.utc)
    start_perf = time.monotonic()

    screenshot_path, content_before = await _capture_state(page, evidence_dir, capability_name, f"{step_id}_pause")
    cdp_info = await _get_cdp_info(cdp_port, page.url)
    window_info = await _get_window_info(page)

    intervention = InterventionRequest(
        capability_name=capability_name,
        current_step_id=step_id,
        reason=reason,
        description=description,
        screenshot_path=screenshot_path,
        current_url=page.url,
        timestamp=started_at,
        step_history=step_history or [],
        cdp_websocket_url=cdp_info.get("page_websocket_url") or cdp_info.get("browser_websocket_url"),
        window_info=window_info or None,
    )

    hub.begin_pause(intervention)
    await hub.ensure_server_started(operator_port)

    logger.warning(
        "ESCALATION: capability=%s step=%s reason=%s -- automation paused, waiting for a human. "
        "Open http://127.0.0.1:%d to review and resume. The automation's own Chromium window is "
        "visible on this machine%s.",
        capability_name,
        step_id,
        reason.value,
        operator_port,
        f"; DevTools can also attach at {intervention.cdp_websocket_url}" if intervention.cdp_websocket_url else "",
    )

    human_took_control = await hub.wait_for_resume()

    duration_seconds = time.monotonic() - start_perf
    url_after = page.url
    # The handoff contract is already satisfied at this point — a human has
    # resumed the run. Everything below is bookkeeping (logging what changed
    # for the run's audit trail), so a failure here must never fail the run
    # itself; it just means the log entry is missing its diff.
    diff_summary: str | None = None
    try:
        _screenshot_after_path, content_after = await _capture_state(
            page, evidence_dir, capability_name, f"{step_id}_resume"
        )
        diff_summary = _summarize_diff(content_before, content_after)
    except Exception:
        logger.warning("failed to capture post-resume state for the audit log; continuing without it", exc_info=True)

    outcome = EscalationOutcome(
        request_id=intervention.request_id,
        human_took_control=human_took_control,
        duration_seconds=duration_seconds,
        url_before=intervention.current_url,
        url_after=url_after,
        url_changed=intervention.current_url != url_after,
        content_diff_summary=diff_summary,
    )

    logger.info(
        "ESCALATION resolved: capability=%s step=%s duration=%.1fs human_took_control=%s url_changed=%s",
        capability_name,
        step_id,
        duration_seconds,
        human_took_control,
        outcome.url_changed,
    )

    hub.end_pause()
    return outcome
