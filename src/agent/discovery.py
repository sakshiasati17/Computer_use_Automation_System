"""Discovery agent loop: drive a real browser, observe with Claude, act, repeat.

This is the caller `observer.observe_and_decide` expects: it owns the
Playwright browser, executes whatever single next action Claude decides on,
logs everything, and repeats until the goal is declared complete (or the
agent gets stuck, or `max_steps` runs out). Every step is screenshotted and
recorded to `/evidence/discovery/<session_id>/` so a run can be inspected or
debugged after the fact; a successful run is handed to
`src.artifacts.emitter.emit_artifact` to produce a replayable Artifact.

Element resolution has no test IDs or CSS classes to lean on (see
`observer.py`'s docstring), so `resolve_element` tries, in order: label
proximity (for type/select, since these apps associate a label with its
field only by table-cell adjacency), exact visible text, ARIA role/name,
and finally a loose substring match on the free-form description Claude
gave us. Each candidate is tried against the main frame and then every
iframe in turn, since Playwright locators are frame-scoped and this app
embeds member data in an iframe.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from playwright.async_api import Frame, Page, async_playwright

from ..artifacts.emitter import emit_artifact
from ..observability import RunLogger, RunOutcome, RunType, save_evidence
from ..safety import check_action, check_url, classify_action, redact_dict
from .observer import AgentAction, AgentActionType, observe_and_decide

__all__ = ["run_discovery", "resolve_element", "ResolvedElement", "build_arg_parser", "main"]

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_ROOT = REPO_ROOT / "evidence" / "discovery"

DEFAULT_MAX_STEPS = 20
SETTLE_DELAY_MS = 1000
WAIT_FOR_ELEMENT_TIMEOUT_MS = 8000
DEFAULT_WAIT_SECONDS = 2.0
PAGE_TEXT_SNIPPET_LIMIT = 2000


# --------------------------------------------------------------------------
# Element resolution
# --------------------------------------------------------------------------


@dataclass
class ResolvedElement:
    """A Claude-described target, resolved to a concrete Playwright locator."""

    locator: Any
    frame: Frame
    strategy: str
    anchor_text: str | None
    metadata: dict[str, Any]


def _xpath_literal(value: str) -> str:
    """Quote `value` as an XPath string literal, handling embedded quotes."""
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    parts = value.split("'")
    return "concat(" + ", \"'\", ".join(f"'{p}'" for p in parts) + ")"


def _label_proximity_xpath(label_text: str) -> str:
    """XPath for the nearest input/select/textarea following a label-like element.

    Legacy table layouts (see demo_app/templates/*.html) pair a label in one
    `<td>` with the field in the next `<td>` and no name/id/aria association
    between them, so proximity in document order is the only reliable
    connection available.
    """
    literal = _xpath_literal(label_text)
    return (
        "//*[self::td or self::label or self::span or self::div or self::b or self::th]"
        f"[contains(normalize-space(string(.)), {literal})]"
        "/following::*[self::input or self::select or self::textarea][1]"
    )


_ROLE_NAME_RE = re.compile(
    r"""^\s*(?P<role>[a-zA-Z]+)\s*(?:["'“](?P<name1>.*?)["'”]|(?P<name2>.+))?\s*$"""
)


def _parse_role_name(role_str: str) -> tuple[str, str | None]:
    """Parse strings like `button "Submit"` or bare `textbox` into (role, name)."""
    match = _ROLE_NAME_RE.match(role_str or "")
    if not match:
        return (role_str or "").strip(), None
    role = match.group("role")
    name = match.group("name1") or match.group("name2")
    return role, (name.strip() if name else None)


_QUOTED_LABEL_RE = re.compile(r"['\"“]([^'\"”]{1,60})['\"”]")


def _extract_quoted_labels(text: str | None) -> list[str]:
    """Pull quoted substrings out of a free-form description.

    `observer.py`'s system prompt tells Claude to embed the actual label
    text in quotes inside `target_description` (e.g. "the textbox in the
    row labeled 'Password:'"), specifically so a locator can recover it even
    when `target_text` itself is left unset. Without this, the label-
    proximity candidate for `target_description` would search for the
    *whole sentence* as if it were the label, which never matches anything.
    """
    if not text:
        return []
    seen: list[str] = []
    for match in _QUOTED_LABEL_RE.findall(text):
        label = match.strip()
        if label and label not in seen:
            seen.append(label)
    return seen


def _iter_frames(page: Page) -> list[Frame]:
    """Main frame first, then any live iframes, so text/role matches prefer top-level content."""
    frames = [page.main_frame]
    frames.extend(f for f in page.frames if f is not page.main_frame and not f.is_detached())
    return frames


async def _try_candidate(ctx: Page | Frame, make_locator: Any, *, strict: bool) -> Any | None:
    """Build a locator from `make_locator(ctx)` and return a match, if any.

    A single match is always accepted. Multiple matches are only accepted
    when `strict` is False, on the assumption that the query itself already
    carries enough disambiguating context (a specific label/text/role+name)
    that "the first visible one" is a reasonable tiebreaker. `strict=True`
    is for queries with little to no disambiguating power - a bare ARIA
    role with no accessible name, or a loose substring match on free-form
    text - where picking *any* match out of several is as likely to be
    wrong as right (e.g. `get_by_role("textbox")` with no name matches
    every plain text input on a legacy form, including unrelated ones).
    """
    try:
        loc = make_locator(ctx)
        count = await loc.count()
    except Exception:
        return None
    if count == 0:
        return None
    if count == 1:
        return loc.first
    if strict:
        return None
    for i in range(count):
        nth = loc.nth(i)
        try:
            if await nth.is_visible():
                return nth
        except Exception:
            continue
    return None


async def _element_metadata(locator: Any) -> dict[str, Any]:
    """Capture tag/name/id/role/text and bounding box for locator-fallback construction."""
    try:
        meta = await locator.evaluate(
            """el => ({
                tagName: el.tagName ? el.tagName.toLowerCase() : null,
                type: el.getAttribute ? el.getAttribute('type') : null,
                name: el.getAttribute ? el.getAttribute('name') : null,
                id: el.getAttribute ? el.getAttribute('id') : null,
                role: el.getAttribute ? el.getAttribute('role') : null,
                text: ((el.innerText || el.value || '') + '').trim().slice(0, 200),
            })"""
        )
    except Exception:
        meta = {}
    try:
        box = await locator.bounding_box()
    except Exception:
        box = None
    meta["bounding_box"] = box
    return meta


async def resolve_element(page: Page, action: AgentAction) -> ResolvedElement | None:
    """Resolve an AgentAction's target to a concrete locator + the frame it lives in.

    Candidates are ordered from most to least disambiguating, and each
    carries a `strict` flag (see `_try_candidate`): a bare ARIA role with no
    accessible name, and the loose free-text fallback on the whole
    description, are marked strict so an ambiguous multi-match there is
    treated as a miss and falls through to a more specific candidate,
    instead of guessing "the first one" - which, on a form with two
    same-role text inputs (e.g. username/password), is exactly how a
    password would silently end up typed into the username field.

    ARIA role+name is tried *before* the plain visible-text match, not
    after: this legacy app's buttons are `<input type="submit">` elements,
    which have no text child nodes at all (their `value` is not a text
    node), so `get_by_text(...)` can never match them correctly - it can
    only ever accidentally match some unrelated heading/label that happens
    to contain the same word (e.g. a "Search" button's text query matching
    a "Member Search" page heading instead, and silently clicking that
    instead of the button). Role+name is scoped to elements that actually
    carry that role, which a plain heading does not, so it doesn't have
    this failure mode.
    """
    needs_label_first = action.action_type in (AgentActionType.TYPE, AgentActionType.SELECT)
    candidates: list[tuple[str, str | None, Any, bool]] = []

    if needs_label_first:
        # Label proximity first, and specifically before the bare-role fallback
        # below: a bare role match is only accepted when it happens to be
        # unique *today*, which is fragile if the vendor ever adds a second
        # same-role field to the page, whereas a locator anchored to the
        # field's own label remains meaningful regardless of what else is on
        # the page. Try target_text's label first, then recover a label Claude
        # may have only written inline in target_description (e.g. "the
        # textbox in the row labeled 'Password:'" when target_text is unset).
        label_candidates = list(_extract_quoted_labels(action.target_description))
        if action.target_text and action.target_text not in label_candidates:
            label_candidates.insert(0, action.target_text)
        for anchor in label_candidates:
            candidates.append(
                (
                    "text_near",
                    anchor,
                    lambda ctx, a=anchor: ctx.locator(f"xpath={_label_proximity_xpath(a)}"),
                    False,
                )
            )

    if action.target_role:
        role, name = _parse_role_name(action.target_role)
        if name:
            candidates.append(
                ("aria_role", name, lambda ctx, r=role, n=name: ctx.get_by_role(r, name=n), False)
            )
        else:
            # No accessible name to disambiguate with: a bare role like "textbox"
            # matches every plain text input on a legacy form, so only accept
            # this if it happens to be unique.
            candidates.append(("aria_role", None, lambda ctx, r=role: ctx.get_by_role(r), True))

    if action.target_text:
        text = action.target_text
        candidates.append(("text", text, lambda ctx, t=text: ctx.get_by_text(t, exact=False), False))

    if action.target_description:
        desc = action.target_description
        # Loosest possible match (substring of a whole free-form sentence): only
        # accept it if unique, for the same reason as the bare-role candidate above.
        candidates.append(("description_text", desc, lambda ctx, d=desc: ctx.get_by_text(d, exact=False), True))

    for frame in _iter_frames(page):
        for strategy, anchor_text, make_locator, strict in candidates:
            locator = await _try_candidate(frame, make_locator, strict=strict)
            if locator is not None:
                metadata = await _element_metadata(locator)
                return ResolvedElement(
                    locator=locator, frame=frame, strategy=strategy, anchor_text=anchor_text, metadata=metadata
                )
    return None


# --------------------------------------------------------------------------
# Action execution
# --------------------------------------------------------------------------


async def _execute_navigate(page: Page, action: AgentAction) -> None:
    await page.goto(action.value)


async def _execute_click(page: Page, action: AgentAction) -> ResolvedElement:
    resolved = await resolve_element(page, action)
    if resolved is None:
        raise RuntimeError(f"could not resolve click target: {action.target_description!r}")
    await resolved.locator.click()
    return resolved


async def _execute_type(page: Page, action: AgentAction) -> ResolvedElement:
    resolved = await resolve_element(page, action)
    if resolved is None:
        raise RuntimeError(f"could not resolve type target: {action.target_description!r}")
    await resolved.locator.click()
    await resolved.locator.fill("")
    value = action.value or ""
    try:
        await resolved.locator.press_sequentially(value, delay=15)
    except AttributeError:
        await resolved.locator.fill(value)
    return resolved


async def _execute_select(page: Page, action: AgentAction) -> ResolvedElement:
    resolved = await resolve_element(page, action)
    if resolved is None:
        raise RuntimeError(f"could not resolve select target: {action.target_description!r}")
    await resolved.locator.select_option(label=action.value)
    return resolved


async def _execute_wait(page: Page, action: AgentAction) -> ResolvedElement | None:
    if action.target_text or action.target_role:
        resolved = await resolve_element(page, action)
        if resolved is not None:
            await resolved.locator.wait_for(state="visible", timeout=WAIT_FOR_ELEMENT_TIMEOUT_MS)
            return resolved
    seconds = DEFAULT_WAIT_SECONDS
    if action.value:
        try:
            seconds = float(action.value)
        except ValueError:
            seconds = DEFAULT_WAIT_SECONDS
    await page.wait_for_timeout(seconds * 1000)
    return None


# --------------------------------------------------------------------------
# Logging helpers
# --------------------------------------------------------------------------


async def _capture_page_text(page: Page) -> str:
    try:
        text = await page.inner_text("body")
    except Exception:
        text = ""
    return text.strip()[:PAGE_TEXT_SNIPPET_LIMIT]


async def _screenshot(page: Page, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    await page.screenshot(path=str(path), full_page=True)


def _resolution_to_dict(resolved: ResolvedElement | None) -> dict[str, Any] | None:
    if resolved is None:
        return None
    return {
        "strategy": resolved.strategy,
        "anchor_text": resolved.anchor_text,
        "frame_url": resolved.frame.url,
        "metadata": resolved.metadata,
    }


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------


async def run_discovery(goal: str, target: str, max_steps: int = DEFAULT_MAX_STEPS) -> dict[str, Any]:
    """Run the observe -> decide -> act loop against `target` until the goal completes.

    Returns the full discovery log (also persisted under
    `evidence/discovery/<session_id>/`). On success, additionally emits a
    replayable Artifact via `src.artifacts.emitter.emit_artifact` and records
    its path in the returned log under `artifact_path`.
    """
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    session_dir = EVIDENCE_ROOT / session_id
    screenshots_dir = session_dir / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)

    log: dict[str, Any] = {
        "goal": goal,
        "target_url": target,
        "session_id": session_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "steps": [],
        "outcome": None,
        "final_extracted_data": None,
        "stuck_reason": None,
        "dialogs_seen": [],
    }

    run_logger = RunLogger(
        RunType.DISCOVERY,
        goal=goal,
        input_params={"target_url": target, "max_steps": max_steps},
    )

    try:
        await _run_session(goal, target, max_steps, log, screenshots_dir, run_logger)
    finally:
        # Redact before anything touches disk; `log` itself stays unredacted
        # in memory so emit_artifact (below) and the return value still see
        # real content.
        log_path = session_dir / "discovery_log.json"
        log_path.write_text(json.dumps(redact_dict(log), indent=2, default=str))
        log["log_path"] = str(log_path)

        if log["outcome"] == "goal_complete":
            try:
                artifact_path = emit_artifact(log)
                log["artifact_path"] = str(artifact_path)
                logger.info("discovery succeeded; artifact saved to %s", artifact_path)
            except Exception:
                logger.exception("emit_artifact failed after a successful discovery run")
        else:
            failure_path = session_dir / "failure_log.json"
            failure_summary = {
                "outcome": log["outcome"],
                "stuck_reason": log.get("stuck_reason"),
                "last_screenshot": log["steps"][-1]["screenshot_path"] if log["steps"] else None,
                "last_step": log["steps"][-1] if log["steps"] else None,
            }
            failure_path.write_text(json.dumps(redact_dict(failure_summary), indent=2, default=str))
            log["failure_log_path"] = str(failure_path)
            logger.warning("discovery ended without success: %s", log["outcome"])

        run_outcome = RunOutcome.SUCCESS if log["outcome"] == "goal_complete" else RunOutcome.HARD_FAILURE
        evidence = await run_logger.finish(run_outcome, error_details=log.get("failure_context"))
        evidence_path = save_evidence(evidence)
        log["evidence_path"] = str(evidence_path)

    return log


async def _run_session(
    goal: str, target: str, max_steps: int, log: dict[str, Any], screenshots_dir: Path, run_logger: RunLogger
) -> None:
    """Own the browser lifecycle and run the loop, mutating `log` in place.

    Split out from `run_discovery` so that *any* exception here - expected
    (a malformed LLM response) or not - still leaves `run_discovery`'s
    `finally` block to persist whatever was captured before failing, rather
    than losing the whole run's evidence to an uncaught traceback.
    """
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=False)
        page = await browser.new_page()
        run_logger.set_page(page)

        async def _on_dialog(dialog: Any) -> None:
            log["dialogs_seen"].append({"type": dialog.type, "message": dialog.message})
            await dialog.accept()

        page.on("dialog", _on_dialog)

        step_history: list[dict[str, Any]] = []

        try:
            await page.goto(target)

            initial_step_start = run_logger.start_timer()
            initial_screenshot = screenshots_dir / "step_0.png"
            await _screenshot(page, initial_screenshot)
            initial_page_title = await page.title()
            log["steps"].append(
                {
                    "step_number": 0,
                    "action_type": "navigate",
                    "target": {
                        "target_description": "Initial navigation to the target URL.",
                        "target_text": None,
                        "target_role": None,
                    },
                    "value": target,
                    "reasoning": "Navigate to the starting URL for this flow.",
                    "screenshot_path": _relative_path(initial_screenshot),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": True,
                    "error_message": None,
                    "resolution": None,
                    "page_url": page.url,
                    "page_title": initial_page_title,
                    "page_text_snippet": await _capture_page_text(page),
                    "extracted_data": None,
                }
            )
            run_logger.record_step(
                step_id="step_0_navigate",
                action="navigate",
                target="Initial navigation to the target URL.",
                value=target,
                result="passed",
                start_perf=initial_step_start,
                screenshot_path=_relative_path(initial_screenshot),
                reasoning="Navigate to the starting URL for this flow.",
                page_url=page.url,
                page_title=initial_page_title,
            )

            for step_number in range(1, max_steps + 1):
                step_start = run_logger.start_timer()
                try:
                    action = await observe_and_decide(page, goal, step_history)
                except Exception as exc:  # noqa: BLE001 - treat as stuck, don't crash the run
                    logger.warning("observe_and_decide failed at step %d: %s", step_number, exc)
                    failure_screenshot = screenshots_dir / f"step_{step_number}_observe_failed.png"
                    try:
                        await _screenshot(page, failure_screenshot)
                        failure_screenshot_rel: str | None = _relative_path(failure_screenshot)
                    except Exception:
                        failure_screenshot_rel = None
                    log["outcome"] = "stuck"
                    log["stuck_reason"] = f"observe_and_decide failed: {exc}"
                    log["steps"].append(
                        {
                            "step_number": step_number,
                            "action_type": "stuck",
                            "target": None,
                            "value": None,
                            "reasoning": f"observe_and_decide failed: {exc}",
                            "screenshot_path": failure_screenshot_rel,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "success": False,
                            "error_message": str(exc),
                            "resolution": None,
                            "page_url": page.url,
                            "page_title": await page.title(),
                            "page_text_snippet": await _capture_page_text(page),
                            "extracted_data": None,
                        }
                    )
                    run_logger.record_step(
                        step_id=f"step_{step_number}_stuck",
                        action="stuck",
                        result="failed",
                        start_perf=step_start,
                        screenshot_path=failure_screenshot_rel,
                        error_message=str(exc),
                        page_url=page.url,
                        page_title=await page.title(),
                    )
                    log["failure_context"] = await run_logger.capture_failure_context(
                        f"observe_and_decide_failed_step_{step_number}"
                    )
                    break

                screenshot_path = screenshots_dir / f"step_{step_number}.png"
                await _screenshot(page, screenshot_path)

                resolved: ResolvedElement | None = None
                error_message: str | None = None
                success = True

                risk_level = classify_action(action, url=page.url)
                blocked_reason: str | None = None
                if action.action_type is AgentActionType.NAVIGATE:
                    if not check_url(action.value or ""):
                        blocked_reason = f"navigation to {action.value!r} blocked by the domain/URL allowlist"
                elif action.action_type in (
                    AgentActionType.CLICK,
                    AgentActionType.TYPE,
                    AgentActionType.SELECT,
                ):
                    action_details = " ".join(
                        filter(None, [action.target_text, action.target_description, action.value])
                    )
                    if not check_action(action.action_type.value, action_details):
                        blocked_reason = (
                            f"{action.action_type.value} action blocked by the action-keyword allowlist: "
                            f"{action_details!r}"
                        )
                if blocked_reason is None and risk_level == "irreversible":
                    blocked_reason = (
                        f"{action.action_type.value} action classified as irreversible; discovery has no "
                        "human-in-the-loop escalation, so it cannot be performed unattended here"
                    )

                if blocked_reason is not None:
                    success = False
                    error_message = blocked_reason
                    logger.warning("step %d blocked by safety policy: %s", step_number, blocked_reason)
                    run_logger.record_safety_violation(
                        {
                            "step_number": step_number,
                            "action_type": action.action_type.value,
                            "risk_level": risk_level,
                            "reason": blocked_reason,
                        }
                    )
                else:
                    try:
                        if action.action_type is AgentActionType.NAVIGATE:
                            await _execute_navigate(page, action)
                        elif action.action_type is AgentActionType.CLICK:
                            resolved = await _execute_click(page, action)
                        elif action.action_type is AgentActionType.TYPE:
                            resolved = await _execute_type(page, action)
                        elif action.action_type is AgentActionType.SELECT:
                            resolved = await _execute_select(page, action)
                        elif action.action_type is AgentActionType.WAIT:
                            resolved = await _execute_wait(page, action)
                        elif action.action_type in (
                            AgentActionType.EXTRACT_DATA,
                            AgentActionType.GOAL_COMPLETE,
                            AgentActionType.STUCK,
                        ):
                            pass  # no page interaction: vision-only read or loop control
                    except Exception as exc:  # noqa: BLE001 - record and continue; don't crash the loop
                        success = False
                        error_message = str(exc)
                        logger.warning("step %d (%s) failed: %s", step_number, action.action_type.value, exc)

                if action.action_type not in (AgentActionType.GOAL_COMPLETE, AgentActionType.STUCK):
                    await page.wait_for_timeout(SETTLE_DELAY_MS)

                step_record = {
                    "step_number": step_number,
                    "action_type": action.action_type.value,
                    "target": {
                        "target_description": action.target_description,
                        "target_text": action.target_text,
                        "target_role": action.target_role,
                    },
                    "value": action.value,
                    "reasoning": action.reasoning,
                    "screenshot_path": _relative_path(screenshot_path),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "success": success,
                    "error_message": error_message,
                    "risk_level": risk_level,
                    "safety_blocked": blocked_reason is not None,
                    "resolution": _resolution_to_dict(resolved),
                    "page_url": page.url,
                    "page_title": await page.title(),
                    "page_text_snippet": await _capture_page_text(page),
                    "extracted_data": action.extracted_data,
                }
                log["steps"].append(step_record)
                step_history.append(
                    {
                        "step_number": step_number,
                        "action_type": action.action_type.value,
                        "target_description": action.target_description,
                        "value": action.value,
                        "reasoning": action.reasoning,
                        "success": success,
                    }
                )
                run_logger.record_step(
                    step_id=f"step_{step_number}_{action.action_type.value}",
                    action=action.action_type.value,
                    target=action.target_description,
                    value=action.value,
                    result="blocked" if blocked_reason is not None else ("passed" if success else "failed"),
                    start_perf=step_start,
                    screenshot_path=_relative_path(screenshot_path),
                    reasoning=action.reasoning,
                    error_message=error_message,
                    page_url=step_record["page_url"],
                    page_title=step_record["page_title"],
                    extra={
                        "risk_level": risk_level,
                        "safety_blocked": blocked_reason is not None,
                        "resolution": step_record["resolution"],
                        "extracted_data": action.extracted_data,
                    },
                )

                if action.action_type is AgentActionType.GOAL_COMPLETE:
                    log["outcome"] = "goal_complete"
                    log["final_extracted_data"] = action.extracted_data or {}
                    break
                if action.action_type is AgentActionType.STUCK:
                    log["outcome"] = "stuck"
                    log["stuck_reason"] = action.stuck_reason
                    log["failure_context"] = await run_logger.capture_failure_context(
                        f"agent_stuck_step_{step_number}"
                    )
                    break
            else:
                log["outcome"] = "max_steps_reached"
                log["failure_context"] = await run_logger.capture_failure_context("max_steps_reached")
        finally:
            log["completed_at"] = datetime.now(timezone.utc).isoformat()
            await browser.close()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the discovery agent loop against a target web app.")
    parser.add_argument("--goal", required=True, help="Natural-language goal for the agent to accomplish.")
    parser.add_argument("--target", required=True, help="URL to start the discovery session from.")
    parser.add_argument(
        "--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum steps before giving up."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_arg_parser().parse_args(argv)
    log = asyncio.run(run_discovery(goal=args.goal, target=args.target, max_steps=args.max_steps))
    print(
        json.dumps(
            {
                "outcome": log["outcome"],
                "log_path": log.get("log_path"),
                "artifact_path": log.get("artifact_path"),
                "failure_log_path": log.get("failure_log_path"),
            },
            indent=2,
        )
    )
    return 0 if log["outcome"] == "goal_complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
