"""Deterministic artifact replay engine: no LLM in the loop.

`src.agent.discovery` spends an LLM call per step to *record* an Artifact
once. This module *replays* one: it walks `artifact.steps` in order, resolves
each step's locator fallback chain with Playwright, executes the action, and
checks the checkpoint — all driven purely by the artifact's own data. This is
the path meant to run unattended, thousands of times a day, so every failure
is triaged through the same three-category taxonomy the schema was built
for (see `src/models/results.py`): a hit `business_outcomes` entry is a
legitimate answer, not an error; a hit `error_handlers` entry is a transient
condition worth recovering from; anything else is a hard failure worth
waking a human up for.

Locator resolution deliberately mirrors `src.agent.discovery.resolve_element`
where the two overlap (frame iteration order, xpath-based label proximity)
so that what discovery recorded and what replay executes stay consistent —
but replay trusts the artifact's own locator strategy per attempt (it was
chosen once, deliberately, by whoever recorded or reviewed the artifact)
rather than guessing at ambiguity the way discovery must.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse
from uuid import uuid4

from playwright.async_api import Frame, Locator, Page, async_playwright

from ..handoff import DEFAULT_CDP_PORT, DEFAULT_OPERATOR_PORT, EscalationReason, escalate, get_hub
from ..observability import RunEvidence, RunLogger, RunOutcome, RunType, save_evidence
from ..safety import check_action, check_url, classify_action
from ..models import (
    ActionType,
    Artifact,
    AriaRoleLocator,
    BusinessOutcome,
    BusinessOutcomeOutcome,
    Condition,
    CoordinatesLocator,
    CssLocator,
    ElementHasValueCondition,
    ElementTarget,
    ElementVisibleCondition,
    Escalation,
    EscalationResolution,
    ErrorHandler,
    ExecutionOutcome,
    ExecutionResult,
    HardFailureOutcome,
    LocatorAttempt,
    RecoveryAction,
    RiskLevel,
    Step,
    StepExecutionRecord,
    StepStatus,
    SuccessOutcome,
    TextLocator,
    TextNearLocator,
    TextPresentCondition,
    UrlCondition,
    UrlMatchType,
)
from ..models.actions import TEMPLATE_REF_PATTERN, XPathLocator

__all__ = ["ReplayConfig", "replay_artifact"]

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_DIR = REPO_ROOT / "evidence" / "failure"

MIN_LOCATOR_TIMEOUT_MS = 500
CONDITION_POLL_INTERVAL_MS = 200
CHECKPOINT_ELEMENT_TIMEOUT_MS = 1000
PAGE_TEXT_SNIPPET_LIMIT = 300
MAX_ESCALATION_RETRIES = 3
"""Cap on how many times a single step re-attempts after a human takes control
during a hard-failure escalation. A human-driven retry loop cannot spin
forever the way an unattended one could, but it still needs *some* bound in
case the human keeps clicking Take Control without resolving the underlying
problem."""


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class ReplayConfig:
    """Runtime knobs for a replay run, independent of any one artifact.

    `permitted_domains` is the navigation allowlist safety gate. Left as
    `None`, it defaults to the artifact's own recorded domain — the engine
    should never silently follow a `navigate` step somewhere the artifact
    wasn't recorded against, and an explicit list lets a caller widen or
    narrow that per-deployment.
    """

    permitted_domains: list[str] | None = None
    headless: bool = True
    evidence_dir: Path = field(default_factory=lambda: DEFAULT_EVIDENCE_DIR)
    enable_escalation: bool = False
    """When True, `risky`/`irreversible` steps and hard failures pause for a
    human via `src/handoff` instead of proceeding or failing unattended.
    Forces `headless=False` at launch time, since a human needs a visible
    window to intervene in (see `replay_artifact`)."""
    operator_port: int = DEFAULT_OPERATOR_PORT
    cdp_port: int = DEFAULT_CDP_PORT
    """Chrome DevTools Protocol port the browser is launched with when
    escalation is enabled, so a human can attach DevTools independently of
    Playwright's own connection."""


# --------------------------------------------------------------------------
# Internal failure types
#
# Every step failure - a locator chain that resolved nothing, a checkpoint
# that never became true, a navigation blocked by the safety allowlist -
# carries `expected`/`observed` strings so a HardFailureOutcome can be built
# uniformly regardless of which of these raised it.
# --------------------------------------------------------------------------


class StepFailure(RuntimeError):
    def __init__(self, message: str, *, expected: str, observed: str) -> None:
        super().__init__(message)
        self.expected = expected
        self.observed = observed


class ElementNotFoundError(StepFailure):
    pass


class CheckpointFailedError(StepFailure):
    pass


class SafetyViolationError(StepFailure):
    """A navigate step's target URL is outside `permitted_domains`.

    Deliberately not passed through the business-outcome/error-handler
    taxonomy: an allowlist violation is a safety gate, not a runtime
    condition the artifact's author anticipated, so it always hard-fails
    immediately rather than being retried or reinterpreted.
    """


# --------------------------------------------------------------------------
# Resolved element: either a real Playwright Locator, or a last-resort
# on-screen coordinate pair that has no Locator at all.
# --------------------------------------------------------------------------


@dataclass
class ResolvedTarget:
    kind: Literal["locator", "coordinates"]
    frame: Frame
    locator: Locator | None = None
    x: int | None = None
    y: int | None = None


# --------------------------------------------------------------------------
# Template substitution
# --------------------------------------------------------------------------


def _substitute(value: str, inputs: dict[str, Any]) -> str:
    """Replace every `{{input.x}}` reference in `value` with its input's value.

    Artifact validation already guarantees every reference in a step's
    `value` names a declared input; this is applied more broadly (checkpoint
    `expected_value`/`pattern`/`text` fields can carry the same syntax, as
    the saved artifacts under `saved_artifacts/` demonstrate) so any
    remaining reference at replay time is a genuinely missing input.
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in inputs:
            raise ValueError(f"template reference '{{{{input.{name}}}}}' has no matching input parameter")
        return str(inputs[name])

    return TEMPLATE_REF_PATTERN.sub(_replace, value)


def _resolve_inputs(artifact: Artifact, input_params: dict[str, Any]) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for param in artifact.inputs:
        if param.name in input_params:
            resolved[param.name] = input_params[param.name]
        elif not param.required:
            resolved[param.name] = param.default_value
        else:
            raise ValueError(f"missing required input parameter '{param.name}'")
    return resolved


# --------------------------------------------------------------------------
# Safety allowlist / risk classification
# --------------------------------------------------------------------------


def _domain_allowed(url: str, permitted_domains: list[str]) -> bool:
    netloc = urlparse(url).netloc
    return any(netloc == domain or netloc.endswith(f".{domain}") for domain in permitted_domains)


_RISK_ORDER = {"safe": 0, "risky": 1, "irreversible": 2}


def _step_risk_details(step: Step) -> str:
    """Identifying text for `step`, for use by `check_action`/`classify_action`.

    Pulls whatever text the step's locator fallback chain carries (label
    text, accessible name, anchor text, selector) alongside the step's own
    description and value, since a `Step`'s `target` is a structured
    `ElementTarget`, not a plain string, unlike the discovery agent's
    free-form `AgentAction`.
    """
    parts = [step.description]
    if step.target is not None:
        for locator in step.target.locators:
            for attr in ("text", "name", "anchor_text", "selector", "expression"):
                value = getattr(locator, attr, None)
                if value:
                    parts.append(str(value))
    if step.value:
        parts.append(step.value)
    return " ".join(parts)


def _effective_risk_level(declared: RiskLevel, classified: str) -> RiskLevel:
    """The more severe of the artifact's declared risk_level and a fresh classification.

    Defense in depth: an artifact's `risk_level` was set once, at recording
    time; `classify_action` re-derives it from the step's current text/URL
    against the live `config/risk_rules.json`, so a step that now matches a
    risky/irreversible pattern is never treated as safe just because it was
    recorded that way.
    """
    if _RISK_ORDER.get(classified, 0) > _RISK_ORDER.get(declared.value, 0):
        return RiskLevel(classified)
    return declared


# --------------------------------------------------------------------------
# Frame / locator helpers
# --------------------------------------------------------------------------


def _iter_frames(page: Page) -> list[Frame]:
    """Main frame first, then any live iframes - member data lives in one (see demo_app)."""
    frames = [page.main_frame]
    frames.extend(f for f in page.frames if f is not page.main_frame and not f.is_detached())
    return frames


def _xpath_literal(value: str) -> str:
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    parts = value.split("'")
    return "concat(" + ", \"'\", ".join(f"'{p}'" for p in parts) + ")"


def _text_near_xpath(spec: TextNearLocator) -> str:
    """XPath for the element positioned relative to an anchor text, per `spec.relation`."""
    literal = _xpath_literal(spec.anchor_text)
    tag_filter = (
        f"self::{spec.element_type}"
        if spec.element_type
        else "self::input or self::select or self::textarea or self::button or self::a"
    )
    anchor = (
        "//*[self::td or self::label or self::span or self::div or self::b or self::th]"
        f"[contains(normalize-space(string(.)), {literal})]"
    )
    if spec.relation == "below":
        return f"{anchor}/ancestor::tr[1]/following-sibling::tr[1]//*[{tag_filter}][1]"
    if spec.relation == "same_container":
        return f"{anchor}/ancestor::*[self::td or self::tr or self::div][1]//*[{tag_filter}][1]"
    # "right_of" and "same_row" both resolve to the next matching element in
    # document order, which is how this legacy table markup pairs a label
    # cell with its field (see demo_app/templates/*.html).
    return f"{anchor}/following::*[{tag_filter}][1]"


def _build_locator(ctx: Page | Frame, spec: Any, inputs: dict[str, Any]) -> Locator | None:
    if isinstance(spec, TextLocator):
        return ctx.get_by_text(_substitute(spec.text, inputs), exact=spec.exact)
    if isinstance(spec, AriaRoleLocator):
        if spec.name:
            return ctx.get_by_role(spec.role, name=_substitute(spec.name, inputs))
        return ctx.get_by_role(spec.role)
    if isinstance(spec, XPathLocator):
        return ctx.locator(f"xpath={spec.expression}")
    if isinstance(spec, CssLocator):
        return ctx.locator(spec.selector)
    if isinstance(spec, TextNearLocator):
        return ctx.locator(f"xpath={_text_near_xpath(spec)}")
    return None


async def _locate_across_frames(
    page: Page, spec: Any, inputs: dict[str, Any], timeout_ms: int
) -> tuple[Locator, Frame] | None:
    """Poll every frame for `spec` until one match appears, or `timeout_ms` elapses."""
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        for frame in _iter_frames(page):
            locator = _build_locator(frame, spec, inputs)
            if locator is None:
                continue
            try:
                count = await locator.count()
            except Exception:
                continue
            if count > 0:
                return locator.first, frame
        if time.monotonic() >= deadline:
            return None
        await page.wait_for_timeout(CONDITION_POLL_INTERVAL_MS)


async def _resolve_element_target(
    page: Page, element_target: ElementTarget, inputs: dict[str, Any], timeout_ms: int
) -> tuple[ResolvedTarget | None, list[LocatorAttempt]]:
    """Try each locator in `element_target`'s fallback chain, in order, across all frames.

    The chain's overall time budget (`timeout_ms`) is split evenly across
    strategies (floored at `MIN_LOCATOR_TIMEOUT_MS`) so an early, wrong
    strategy can't starve the more reliable ones later in the chain of any
    time to wait for a slow-loading page.
    """
    attempts: list[LocatorAttempt] = []
    locators = element_target.locators
    per_locator_timeout = max(timeout_ms // max(len(locators), 1), MIN_LOCATOR_TIMEOUT_MS)

    for spec in locators:
        if isinstance(spec, CoordinatesLocator):
            attempts.append(LocatorAttempt(strategy="coordinates", confidence=spec.confidence, succeeded=True))
            return ResolvedTarget(kind="coordinates", frame=page.main_frame, x=spec.x, y=spec.y), attempts
        try:
            found = await _locate_across_frames(page, spec, inputs, per_locator_timeout)
        except Exception as exc:  # noqa: BLE001 - record and try the next strategy in the chain
            attempts.append(
                LocatorAttempt(strategy=spec.strategy, confidence=spec.confidence, succeeded=False, error_message=str(exc))
            )
            continue
        if found is None:
            attempts.append(
                LocatorAttempt(
                    strategy=spec.strategy, confidence=spec.confidence, succeeded=False, error_message="no matching element found"
                )
            )
            continue
        locator, frame = found
        attempts.append(LocatorAttempt(strategy=spec.strategy, confidence=spec.confidence, succeeded=True))
        return ResolvedTarget(kind="locator", frame=frame, locator=locator), attempts

    return None, attempts


# --------------------------------------------------------------------------
# Action execution
# --------------------------------------------------------------------------


async def _click(resolved: ResolvedTarget) -> None:
    if resolved.kind == "locator":
        assert resolved.locator is not None
        await resolved.locator.click()
    else:
        await resolved.frame.page.mouse.click(resolved.x, resolved.y)


async def _type(resolved: ResolvedTarget, value: str) -> None:
    if resolved.kind == "locator":
        assert resolved.locator is not None
        try:
            await resolved.locator.clear()
        except Exception:
            await resolved.locator.fill("")
        await resolved.locator.fill(value)
    else:
        page = resolved.frame.page
        await page.mouse.click(resolved.x, resolved.y)
        await page.keyboard.type(value)


async def _select(resolved: ResolvedTarget, value: str) -> None:
    if resolved.kind != "locator" or resolved.locator is None:
        raise RuntimeError("cannot select an option using a coordinates-only fallback locator")
    await resolved.locator.select_option(label=value)


async def _extract_text(resolved: ResolvedTarget) -> str:
    if resolved.kind != "locator" or resolved.locator is None:
        raise RuntimeError("cannot extract text using a coordinates-only fallback locator")
    text = await resolved.locator.text_content()
    return (text or "").strip()


# --------------------------------------------------------------------------
# Condition evaluation (checkpoints, business outcomes, error handlers)
# --------------------------------------------------------------------------


async def _all_frames_text(page: Page) -> str:
    parts: list[str] = []
    for frame in _iter_frames(page):
        try:
            parts.append(await frame.inner_text("body"))
        except Exception:
            continue
    return "\n".join(parts)


async def _evaluate_condition_once(page: Page, condition: Condition, inputs: dict[str, Any]) -> bool:
    if isinstance(condition, ElementVisibleCondition):
        for frame in _iter_frames(page):
            if condition.role and condition.text:
                locator = frame.get_by_role(condition.role, name=_substitute(condition.text, inputs))
            elif condition.role:
                locator = frame.get_by_role(condition.role)
            else:
                assert condition.text is not None
                locator = frame.get_by_text(_substitute(condition.text, inputs), exact=False)
            try:
                if await locator.count() > 0 and await locator.first.is_visible():
                    return True
            except Exception:
                continue
        return False

    if isinstance(condition, TextPresentCondition):
        target_text = _substitute(condition.text, inputs)
        haystack = await _all_frames_text(page)
        if condition.case_sensitive:
            return target_text in haystack
        return target_text.lower() in haystack.lower()

    if isinstance(condition, UrlCondition):
        pattern = _substitute(condition.pattern, inputs)
        url = page.url
        if condition.match_type == UrlMatchType.EXACT:
            return url == pattern
        if condition.match_type == UrlMatchType.CONTAINS:
            return pattern in url
        return re.search(pattern, url) is not None

    if isinstance(condition, ElementHasValueCondition):
        resolved, _attempts = await _resolve_element_target(page, condition.target, inputs, CHECKPOINT_ELEMENT_TIMEOUT_MS)
        if resolved is None or resolved.kind != "locator" or resolved.locator is None:
            return False
        expected = _substitute(condition.expected_value, inputs)
        try:
            value = await resolved.locator.input_value()
        except Exception:
            try:
                value = (await resolved.locator.text_content()) or ""
            except Exception:
                return False
        return value == expected

    return False


async def _wait_for_condition(page: Page, condition: Condition, inputs: dict[str, Any], timeout_ms: int) -> bool:
    deadline = time.monotonic() + timeout_ms / 1000
    while True:
        if await _evaluate_condition_once(page, condition, inputs):
            return True
        if time.monotonic() >= deadline:
            return False
        await page.wait_for_timeout(CONDITION_POLL_INTERVAL_MS)


def _describe_condition(condition: Condition) -> str:
    if isinstance(condition, ElementVisibleCondition):
        return f"element visible (text={condition.text!r}, role={condition.role!r})"
    if isinstance(condition, TextPresentCondition):
        return f"text present: {condition.text!r}"
    if isinstance(condition, UrlCondition):
        return f"url {condition.match_type.value} {condition.pattern!r}"
    if isinstance(condition, ElementHasValueCondition):
        return f"element has value {condition.expected_value!r}"
    return "condition"


async def _describe_page_state(page: Page) -> str:
    snippet = (await _all_frames_text(page))[:PAGE_TEXT_SNIPPET_LIMIT]
    return f"url={page.url!r}, visible_text={snippet!r}"


async def _match_business_outcome(page: Page, artifact: Artifact, inputs: dict[str, Any]) -> BusinessOutcome | None:
    for outcome in artifact.business_outcomes:
        if await _evaluate_condition_once(page, outcome.detection, inputs):
            return outcome
    return None


async def _match_error_handler(page: Page, artifact: Artifact, inputs: dict[str, Any]) -> ErrorHandler | None:
    for handler in artifact.error_handlers:
        if await _evaluate_condition_once(page, handler.detection, inputs):
            return handler
    return None


def _extract_business_outcome_data(outcome: BusinessOutcome, inputs: dict[str, Any]) -> dict[str, Any]:
    """Best-effort fill of `outcome.data_to_extract`.

    The schema only carries a human-readable description per field, not a
    locator (unlike `OutputField`), so there is nothing to mechanically read
    off the page. The one thing replay *can* do without guessing is pass
    through an input parameter whose name matches a requested field exactly.
    """
    return {key: inputs[key] for key in (outcome.data_to_extract or {}) if key in inputs}


# --------------------------------------------------------------------------
# Per-step action dispatch
# --------------------------------------------------------------------------


async def _perform_step_action(
    page: Page,
    step: Step,
    inputs: dict[str, Any],
    permitted_domains: list[str],
    locator_attempts: list[LocatorAttempt],
    outputs: dict[str, Any],
) -> None:
    if step.action is ActionType.NAVIGATE:
        assert step.value is not None
        url = _substitute(step.value, inputs)
        if not _domain_allowed(url, permitted_domains):
            raise SafetyViolationError(
                f"navigation to {url!r} blocked by permitted_domains allowlist",
                expected=f"navigation target within permitted domains {permitted_domains}",
                observed=f"navigation to {url!r} ({urlparse(url).netloc!r})",
            )
        if not check_url(url):
            raise SafetyViolationError(
                f"navigation to {url!r} blocked by the safety domain/URL allowlist",
                expected="navigation target permitted by config/allowlist.json",
                observed=f"navigation to {url!r}",
            )
        await page.goto(url)
        return

    if step.action is ActionType.WAIT:
        if step.target is not None:
            resolved, attempts = await _resolve_element_target(page, step.target, inputs, step.timeout_ms)
            locator_attempts.extend(attempts)
            if resolved is None:
                raise ElementNotFoundError(
                    f"wait target not found for step {step.step_id!r}",
                    expected="the wait target element to appear",
                    observed=f"none of {[a.strategy for a in attempts]} resolved an element",
                )
        elif step.checkpoint is not None:
            ok = await _wait_for_condition(page, step.checkpoint, inputs, step.timeout_ms)
            if not ok:
                raise CheckpointFailedError(
                    f"wait checkpoint never became true for step {step.step_id!r}",
                    expected=_describe_condition(step.checkpoint),
                    observed=await _describe_page_state(page),
                )
        return

    assert step.target is not None  # guaranteed by Step's own validator for click/type/select/extract
    action_details = _step_risk_details(step)
    if not check_action(step.action.value, action_details):
        raise SafetyViolationError(
            f"{step.action.value} action blocked by the safety action-keyword allowlist",
            expected="action permitted by config/allowlist.json blocked_action_keywords",
            observed=f"{step.action.value}: {action_details!r}",
        )

    resolved, attempts = await _resolve_element_target(page, step.target, inputs, step.timeout_ms)
    locator_attempts.extend(attempts)
    if resolved is None:
        raise ElementNotFoundError(
            f"could not resolve target for step {step.step_id!r}",
            expected="one of the target's locator strategies to resolve an element",
            observed=f"none of {[a.strategy for a in attempts]} resolved an element (including iframes)",
        )

    if step.action is ActionType.CLICK:
        await _click(resolved)
    elif step.action is ActionType.TYPE:
        assert step.value is not None
        await _type(resolved, _substitute(step.value, inputs))
    elif step.action is ActionType.SELECT:
        assert step.value is not None
        await _select(resolved, _substitute(step.value, inputs))
    elif step.action is ActionType.EXTRACT:
        assert step.output_field is not None
        outputs[step.output_field] = await _extract_text(resolved)


# --------------------------------------------------------------------------
# Recovery actions
# --------------------------------------------------------------------------


async def _attempt_recovery(
    page: Page,
    artifact: Artifact,
    step_index: int,
    inputs: dict[str, Any],
    permitted_domains: list[str],
    recovery_action: RecoveryAction,
) -> None:
    """Best-effort remediation for a matched (non-escalate) ErrorHandler.

    Swallows its own failures: if recovery itself doesn't pan out, the
    caller's retry of the step will simply fail again and eventually exhaust
    `max_retries`, which is the correct outcome either way.
    """
    if recovery_action is RecoveryAction.DISMISS_DIALOG:
        try:
            await page.keyboard.press("Escape")
        except Exception:
            logger.warning("dismiss_dialog recovery: failed to press Escape", exc_info=True)
    elif recovery_action is RecoveryAction.WAIT_AND_RETRY:
        await page.wait_for_timeout(1000)
    elif recovery_action is RecoveryAction.RETRY_STEP:
        pass  # the caller's loop simply re-attempts the step
    elif recovery_action is RecoveryAction.RE_AUTHENTICATE:
        try:
            await page.goto(artifact.target.entry_url)
            for prior_step in artifact.steps[:step_index]:
                await _perform_step_action(page, prior_step, inputs, permitted_domains, [], {})
        except Exception:
            logger.warning("re_authenticate recovery: replaying prior steps failed", exc_info=True)


# --------------------------------------------------------------------------
# Per-step orchestration: locate/act, checkpoint, and the three-category
# failure taxonomy.
# --------------------------------------------------------------------------


async def _execute_step(
    page: Page,
    artifact: Artifact,
    step: Step,
    step_index: int,
    inputs: dict[str, Any],
    permitted_domains: list[str],
    outputs: dict[str, Any],
    escalations: list[Escalation],
    execution_log: list[StepExecutionRecord],
    config: ReplayConfig,
) -> tuple[StepExecutionRecord, ExecutionOutcome | None]:
    """Run one artifact step to completion. Returns (log record, stop-outcome-or-None).

    A `None` outcome means "record the step and continue to the next one".
    A non-`None` outcome means the whole run is over: either a legitimate
    BusinessOutcomeOutcome or a HardFailureOutcome, per the module docstring.
    """
    started_at = datetime.now(timezone.utc)
    start_perf = time.monotonic()
    locator_attempts: list[LocatorAttempt] = []
    retries = 0
    escalation_attempts = 0

    classified_risk = classify_action(
        {"action_type": step.action.value, "description": _step_risk_details(step)}, url=page.url
    )
    effective_risk_level = _effective_risk_level(step.risk_level, classified_risk)
    if effective_risk_level is not step.risk_level:
        logger.warning(
            "step %s: risk re-classified from %s to %s at replay time (config/risk_rules.json)",
            step.step_id,
            step.risk_level.value,
            effective_risk_level.value,
        )

    if effective_risk_level in (RiskLevel.RISKY, RiskLevel.IRREVERSIBLE) and page.is_closed():
        # Nothing to escalate: there's no page left for a human to review or
        # act on. Surface this plainly rather than pausing an escalation
        # that can never be resolved (see the matching guard in the
        # exception handler below).
        duration_ms = int((time.monotonic() - start_perf) * 1000)
        record = StepExecutionRecord(
            step_id=step.step_id,
            action=step.action,
            description=step.description,
            status=StepStatus.FAILED,
            started_at=started_at,
            duration_ms=duration_ms,
            retries=0,
            locator_attempts=[],
            error_message="browser page was closed unexpectedly before this risky/irreversible step could run",
        )
        stop_outcome = HardFailureOutcome(
            failed_step_id=step.step_id,
            action_attempted=step.action,
            expected="the browser page to still be open for this risky/irreversible step",
            observed="the browser page was closed",
            page_url=None,
            exception_type="PageClosed",
            message="browser page was closed unexpectedly; cannot perform or escalate a risky/irreversible step with no page",
        )
        return record, stop_outcome

    if effective_risk_level in (RiskLevel.RISKY, RiskLevel.IRREVERSIBLE):
        if not config.enable_escalation:
            logger.warning(
                "step %s has risk_level=%s but enable_escalation=False; proceeding unattended",
                step.step_id,
                effective_risk_level.value,
            )
            escalations.append(
                Escalation(
                    step_id=step.step_id,
                    reason=f"step risk_level={effective_risk_level.value}",
                    risk_level=effective_risk_level,
                    requested_at=datetime.now(timezone.utc),
                )
            )
        else:
            esc_outcome = await escalate(
                page=page,
                capability_name=artifact.name,
                step_id=step.step_id,
                reason=EscalationReason.RISKY_ACTION,
                description=(
                    f"Step {step.step_id!r} ({step.description}) is marked "
                    f"risk_level={effective_risk_level.value!r} and requires human approval before "
                    f"it runs: {step.action.value} on the step's target."
                ),
                step_history=[r.model_dump(mode="json") for r in execution_log],
                cdp_port=config.cdp_port,
                operator_port=config.operator_port,
            )
            escalations.append(
                Escalation(
                    step_id=step.step_id,
                    reason=f"step risk_level={effective_risk_level.value}",
                    risk_level=effective_risk_level,
                    requested_at=started_at,
                    resolution=EscalationResolution.APPROVED,
                    resolved_at=datetime.now(timezone.utc),
                    resolved_by="operator" if esc_outcome.human_took_control else None,
                    notes=(
                        f"human_took_control={esc_outcome.human_took_control}, "
                        f"duration={esc_outcome.duration_seconds:.1f}s, "
                        f"content_changed={esc_outcome.content_diff_summary is not None}"
                    ),
                )
            )
            if esc_outcome.human_took_control:
                # The human already performed (or deliberately avoided) the risky
                # action directly in the visible browser. Do NOT also perform it
                # here — re-running a risky/irreversible step because both the
                # human and the engine acted on it would be worse than the pause
                # itself (e.g. double-submitting a payment). Verify the
                # checkpoint instead and hand back to the normal step loop.
                duration_ms = int((time.monotonic() - start_perf) * 1000)
                checkpoint_ok = True
                if step.checkpoint is not None:
                    checkpoint_ok = await _wait_for_condition(page, step.checkpoint, inputs, step.timeout_ms)
                record = StepExecutionRecord(
                    step_id=step.step_id,
                    action=step.action,
                    description=step.description,
                    status=StepStatus.PASSED if checkpoint_ok else StepStatus.FAILED,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    retries=0,
                    locator_attempts=[],
                    error_message=None if checkpoint_ok else "checkpoint not met after human-performed action",
                )
                if checkpoint_ok:
                    return record, None
                screenshot_path = await _save_failure_screenshot(page, artifact, step)
                stop_outcome = HardFailureOutcome(
                    failed_step_id=step.step_id,
                    action_attempted=step.action,
                    expected=_describe_condition(step.checkpoint) if step.checkpoint else "step to complete successfully",
                    observed=await _describe_page_state(page),
                    screenshot_path=screenshot_path,
                    page_url=page.url,
                    exception_type="HumanInterventionCheckpointFailed",
                    message="human took control during a risky-action escalation, but the step's checkpoint still was not met",
                )
                return record, stop_outcome
            # else: a bare "Resume Automation" click is an approval, not a
            # substitute for the action — fall through and let the engine
            # perform the step itself, same as any other step.

    last_exc: StepFailure | Exception | None = None

    while True:
        try:
            await _perform_step_action(page, step, inputs, permitted_domains, locator_attempts, outputs)
            if step.checkpoint is not None and step.action is not ActionType.WAIT:
                ok = await _wait_for_condition(page, step.checkpoint, inputs, step.timeout_ms)
                if not ok:
                    raise CheckpointFailedError(
                        f"checkpoint never became true for step {step.step_id!r}",
                        expected=_describe_condition(step.checkpoint),
                        observed=await _describe_page_state(page),
                    )

            duration_ms = int((time.monotonic() - start_perf) * 1000)
            record = StepExecutionRecord(
                step_id=step.step_id,
                action=step.action,
                description=step.description,
                status=StepStatus.RETRIED if retries else StepStatus.PASSED,
                started_at=started_at,
                duration_ms=duration_ms,
                retries=retries,
                locator_attempts=locator_attempts,
                error_message=None,
            )
            return record, None

        except SafetyViolationError as exc:
            duration_ms = int((time.monotonic() - start_perf) * 1000)
            record = StepExecutionRecord(
                step_id=step.step_id,
                action=step.action,
                description=step.description,
                status=StepStatus.FAILED,
                started_at=started_at,
                duration_ms=duration_ms,
                retries=retries,
                locator_attempts=locator_attempts,
                error_message=str(exc),
            )
            screenshot_path = await _save_failure_screenshot(page, artifact, step)
            outcome = HardFailureOutcome(
                failed_step_id=step.step_id,
                action_attempted=step.action,
                expected=exc.expected,
                observed=exc.observed,
                screenshot_path=screenshot_path,
                page_url=page.url,
                exception_type=type(exc).__name__,
                message=str(exc),
                locator_attempts=locator_attempts,
            )
            return record, outcome

        except Exception as exc:  # noqa: BLE001 - triaged below via the 3-category taxonomy
            last_exc = exc

            business_outcome = await _match_business_outcome(page, artifact, inputs)
            if business_outcome is not None:
                duration_ms = int((time.monotonic() - start_perf) * 1000)
                record = StepExecutionRecord(
                    step_id=step.step_id,
                    action=step.action,
                    description=step.description,
                    status=StepStatus.FAILED,
                    started_at=started_at,
                    duration_ms=duration_ms,
                    retries=retries,
                    locator_attempts=locator_attempts,
                    error_message=f"checkpoint not met; matched business outcome {business_outcome.name!r} instead",
                )
                outcome = BusinessOutcomeOutcome(
                    outcome_name=business_outcome.name,
                    description=business_outcome.description,
                    severity=business_outcome.severity,
                    extracted_data=_extract_business_outcome_data(business_outcome, inputs),
                )
                return record, outcome

            handler = await _match_error_handler(page, artifact, inputs)
            give_up_reason: str | None = None
            if handler is not None and handler.recovery_action is RecoveryAction.ESCALATE:
                give_up_reason = f"error_handler {handler.name!r} requested escalation: {handler.description}"
            elif handler is not None and retries < handler.max_retries:
                retries += 1
                logger.info(
                    "recovering step %s via %s (attempt %d/%d)",
                    step.step_id,
                    handler.recovery_action.value,
                    retries,
                    handler.max_retries,
                )
                await _attempt_recovery(page, artifact, step_index, inputs, permitted_domains, handler.recovery_action)
                continue
            else:
                give_up_reason = f"step {step.step_id!r} failed: {exc}"

            if page.is_closed():
                # A closed page can't be shown to a human, and every locator
                # attempt against it fails silently (caught per-frame in
                # `_locate_across_frames`), which is indistinguishable from
                # "element genuinely not found" by the time we get here -
                # check explicitly rather than pausing an escalation no one
                # can ever resolve.
                give_up_reason = f"{give_up_reason} (the browser page was closed unexpectedly; nothing left to escalate)"
            elif config.enable_escalation and escalation_attempts < MAX_ESCALATION_RETRIES:
                escalation_attempts += 1
                esc_outcome = await escalate(
                    page=page,
                    capability_name=artifact.name,
                    step_id=step.step_id,
                    reason=EscalationReason.HARD_FAILURE,
                    description=(
                        f"{give_up_reason} (escalation {escalation_attempts}/{MAX_ESCALATION_RETRIES} "
                        "before this run gives up on the step)."
                    ),
                    step_history=[r.model_dump(mode="json") for r in execution_log],
                    cdp_port=config.cdp_port,
                    operator_port=config.operator_port,
                )
                # Bare "Resume" (no Take Control) means the human looked and
                # couldn't/wouldn't fix it either — REJECTED. Taking control
                # means they intervened, so the step deserves another try.
                escalations.append(
                    Escalation(
                        step_id=step.step_id,
                        reason=give_up_reason,
                        risk_level=effective_risk_level,
                        requested_at=datetime.now(timezone.utc),
                        resolution=EscalationResolution.APPROVED
                        if esc_outcome.human_took_control
                        else EscalationResolution.REJECTED,
                        resolved_at=datetime.now(timezone.utc),
                        resolved_by="operator" if esc_outcome.human_took_control else None,
                        notes=(
                            f"human_took_control={esc_outcome.human_took_control}, "
                            f"duration={esc_outcome.duration_seconds:.1f}s"
                        ),
                    )
                )
                if esc_outcome.human_took_control:
                    continue

            duration_ms = int((time.monotonic() - start_perf) * 1000)
            record = StepExecutionRecord(
                step_id=step.step_id,
                action=step.action,
                description=step.description,
                status=StepStatus.FAILED,
                started_at=started_at,
                duration_ms=duration_ms,
                retries=retries,
                locator_attempts=locator_attempts,
                error_message=str(exc),
            )
            screenshot_path = await _save_failure_screenshot(page, artifact, step)
            expected = getattr(exc, "expected", "step to complete successfully")
            observed = getattr(exc, "observed", str(exc))
            outcome = HardFailureOutcome(
                failed_step_id=step.step_id,
                action_attempted=step.action,
                expected=expected,
                observed=observed,
                screenshot_path=screenshot_path,
                page_url=page.url,
                exception_type=type(exc).__name__,
                message=str(exc),
                locator_attempts=locator_attempts,
            )
            return record, outcome


async def _save_failure_screenshot(page: Page, artifact: Artifact, step: Step) -> str | None:
    evidence_dir = DEFAULT_EVIDENCE_DIR
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = evidence_dir / f"{artifact.name}_{step.step_id}_{timestamp}.png"
    try:
        await page.screenshot(path=str(path), full_page=True)
        return str(path)
    except Exception:
        logger.warning("failed to capture failure screenshot", exc_info=True)
        return None


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def _load_artifact(artifact_path: str | Path) -> Artifact:
    path = Path(artifact_path)
    return Artifact.model_validate_json(path.read_text())


def _run_outcome_and_error_details(outcome: ExecutionOutcome) -> tuple[RunOutcome, dict[str, Any] | None]:
    """Map the three-way `ExecutionOutcome` to a `RunOutcome` plus debugging context.

    A `HardFailureOutcome`/`BusinessOutcomeOutcome` already carries everything
    worth keeping (see `src/models/results.py`); this just reshapes it into
    the run-level `error_details` bucket instead of re-deriving it.
    """
    if isinstance(outcome, SuccessOutcome):
        return RunOutcome.SUCCESS, None
    if isinstance(outcome, BusinessOutcomeOutcome):
        return RunOutcome.BUSINESS_OUTCOME, {
            "outcome_name": outcome.outcome_name,
            "description": outcome.description,
            "severity": outcome.severity.value,
            "extracted_data": outcome.extracted_data,
        }
    return RunOutcome.HARD_FAILURE, {
        "failed_step_id": outcome.failed_step_id,
        "action_attempted": outcome.action_attempted.value,
        "expected": outcome.expected,
        "observed": outcome.observed,
        "exception_type": outcome.exception_type,
        "message": outcome.message,
        "screenshot_path": outcome.screenshot_path,
        "page_url": outcome.page_url,
    }


async def replay_artifact(
    artifact_path: str | Path,
    input_params: dict[str, Any],
    config: ReplayConfig | None = None,
) -> ExecutionResult:
    """Replay a saved Artifact deterministically, with no LLM calls.

    Returns the full `ExecutionResult`, whose `outcome` is one of
    `SuccessOutcome`, `BusinessOutcomeOutcome`, or `HardFailureOutcome` - see
    this module's docstring and `src/models/results.py` for why that split
    is load-bearing rather than incidental.
    """
    config = config or ReplayConfig()
    artifact = _load_artifact(artifact_path)
    inputs = _resolve_inputs(artifact, input_params)
    permitted_domains = config.permitted_domains or [urlparse(artifact.target.entry_url).netloc]

    started_at = datetime.now(timezone.utc)
    execution_log: list[StepExecutionRecord] = []
    escalations: list[Escalation] = []
    outputs: dict[str, Any] = {}

    run_logger = RunLogger(RunType.REPLAY, artifact_name=artifact.name, input_params=inputs)
    evidence: RunEvidence | None = None

    # Human-in-the-loop escalation requires a browser window a human can
    # actually see and click on, and a Chrome DevTools Protocol port they can
    # attach to independently of Playwright's own connection - a headless
    # browser satisfies neither. `enable_escalation` therefore always wins
    # over `headless` rather than silently escalating into a browser no one
    # can reach.
    if config.enable_escalation and config.headless:
        logger.warning("enable_escalation=True requires a visible browser; overriding headless=True to False")
    launch_headless = False if config.enable_escalation else config.headless
    launch_args = [f"--remote-debugging-port={config.cdp_port}"] if config.enable_escalation else None

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=launch_headless, args=launch_args)
        page = await browser.new_page()
        run_logger.set_page(page)
        try:
            outcome: ExecutionOutcome | None = None
            for step_index, step in enumerate(artifact.steps):
                record, stop_outcome = await _execute_step(
                    page,
                    artifact,
                    step,
                    step_index,
                    inputs,
                    permitted_domains,
                    outputs,
                    escalations,
                    execution_log,
                    config,
                )
                execution_log.append(record)

                step_screenshot_path: str | None = None
                step_page_url: str | None = None
                if isinstance(stop_outcome, HardFailureOutcome):
                    step_screenshot_path = stop_outcome.screenshot_path
                    step_page_url = stop_outcome.page_url
                    if stop_outcome.exception_type == "SafetyViolationError":
                        run_logger.record_safety_violation(
                            {
                                "step_id": step.step_id,
                                "expected": stop_outcome.expected,
                                "observed": stop_outcome.observed,
                                "message": stop_outcome.message,
                            }
                        )
                run_logger.record_step(
                    step_id=record.step_id,
                    action=record.action.value,
                    target=step.description,
                    value=step.value,
                    result=record.status.value,
                    duration_ms=record.duration_ms,
                    started_at=record.started_at,
                    screenshot_path=step_screenshot_path,
                    error_message=record.error_message,
                    page_url=step_page_url,
                    retries=record.retries,
                    extra={"locator_attempts": [a.model_dump(mode="json") for a in record.locator_attempts]}
                    if record.locator_attempts
                    else {},
                )

                if stop_outcome is not None:
                    outcome = stop_outcome
                    break

            if outcome is None:
                # Artifact.steps has min_length=1, so there is always a last step to
                # attribute this check to.
                last_step = artifact.steps[-1]
                condition_start = run_logger.start_timer()
                if not await _wait_for_condition(page, artifact.success_condition, inputs, timeout_ms=5000):
                    screenshot_path = await _save_failure_screenshot(page, artifact, last_step)
                    outcome = HardFailureOutcome(
                        failed_step_id=last_step.step_id,
                        action_attempted=ActionType.WAIT,
                        expected=_describe_condition(artifact.success_condition),
                        observed=await _describe_page_state(page),
                        screenshot_path=screenshot_path,
                        page_url=page.url,
                        exception_type="SuccessConditionNotMet",
                        message="all steps passed their own checkpoints, but the artifact's overall success_condition was not met",
                    )
                    run_logger.record_step(
                        step_id="success_condition_check",
                        action="wait",
                        result="failed",
                        start_perf=condition_start,
                        screenshot_path=screenshot_path,
                        error_message=outcome.message,
                        page_url=page.url,
                    )
                else:
                    outcome = SuccessOutcome(outputs=outputs)
                    run_logger.record_step(
                        step_id="success_condition_check",
                        action="wait",
                        result="passed",
                        start_perf=condition_start,
                        page_url=page.url,
                    )

            for escalation in escalations:
                run_logger.record_escalation(escalation)
            run_outcome, error_details = _run_outcome_and_error_details(outcome)
            evidence = await run_logger.finish(run_outcome, error_details=error_details)
        finally:
            await browser.close()
            if config.enable_escalation:
                hub = get_hub()
                hub.mark_completed()
                await hub.shutdown_server()

    if evidence is not None:
        save_evidence(evidence)

    completed_at = datetime.now(timezone.utc)
    total_duration_ms = int((completed_at - started_at).total_seconds() * 1000)

    return ExecutionResult(
        run_id=uuid4(),
        artifact_name=artifact.name,
        artifact_version=artifact.version,
        input_params=inputs,
        started_at=started_at,
        completed_at=completed_at,
        total_duration_ms=total_duration_ms,
        outcome=outcome,
        execution_log=execution_log,
        escalations=escalations,
    )
