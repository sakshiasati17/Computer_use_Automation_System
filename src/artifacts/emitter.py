"""Artifact emitter: convert a raw discovery step log into a replayable Artifact.

Discovery (`src.agent.discovery.run_discovery`) produces a step-by-step
execution log driven by an LLM watching screenshots of one specific run.
This module turns that raw, one-off recording into the structured
`Artifact` schema (`src.models.artifact`):

- Concrete values typed into fields become templated `{{input.x}}`
  parameters, named after the field's label.
- Locator fallback chains are built from whatever was actually resolved on
  the page during discovery (label proximity, text, ARIA role, a CSS
  selector derived from the element's `name` attribute, and a last-resort
  screen position) rather than guessed.
- Checkpoints come from what was actually observed to happen next (a URL
  change after a click, the value just typed/selected), not invented.
- Business outcomes are only added when their trigger text was actually
  seen somewhere in the run; a handful of generic, defensive error handlers
  (session timeout, an unexpected native dialog) are always attached, since
  a replay engine runs unattended long after discovery and should watch for
  them regardless of whether this particular run happened to hit them.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..models import (
    ActionType,
    AriaRoleLocator,
    Artifact,
    BusinessOutcome,
    Condition,
    Confidence,
    CoordinatesLocator,
    CssLocator,
    ElementHasValueCondition,
    ElementTarget,
    ElementVisibleCondition,
    ErrorHandler,
    InputParameter,
    OutputField,
    ParamType,
    RecoveryAction,
    RiskLevel,
    Severity,
    Step,
    SurfaceType,
    Target,
    TextLocator,
    TextNearLocator,
    TextPresentCondition,
    UrlCondition,
    UrlMatchType,
)
from ..safety import redact_dict

__all__ = ["emit_artifact"]

REPO_ROOT = Path(__file__).resolve().parents[2]
SAVED_ARTIFACTS_DIR = REPO_ROOT / "saved_artifacts"

_NON_STEP_ACTIONS = {"goal_complete", "stuck"}
_RISKY_KEYWORDS = ("confirm", "create", "submit", "delete")
_SUCCESS_KEYWORDS = ("success", "successfully", "complete", "completed", "created", "confirmed")

_OUTCOME_PATTERNS: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"member not found", re.I), "member_not_found", "The searched member ID does not exist in the system."),
    (re.compile(r"invalid credentials", re.I), "invalid_credentials", "The submitted login credentials were rejected."),
    (
        re.compile(r"minimum[^.\n]*deposit[^.\n]*\$?[\d,.]+", re.I),
        "minimum_deposit_not_met",
        "The entered deposit amount is below the vendor-enforced minimum.",
    ),
    (re.compile(r"\bnot found\b", re.I), "record_not_found", "The searched record does not exist in the system."),
]

_ROLE_NAME_RE = re.compile(
    r"""^\s*(?P<role>[a-zA-Z]+)\s*(?:["'“](?P<name1>.*?)["'”]|(?P<name2>.+))?\s*$"""
)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _slugify(text: str | None, *, fallback: str = "value") -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not text:
        text = fallback
    if not text[0].isalpha():
        text = f"field_{text}"
    return text


def _unique_name(base: str, used: set[str]) -> str:
    name = base
    i = 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def _parse_role_name(role_str: str | None) -> tuple[str, str | None]:
    match = _ROLE_NAME_RE.match(role_str or "")
    if not match:
        return (role_str or "").strip(), None
    role = match.group("role")
    name = match.group("name1") or match.group("name2")
    return role, (name.strip() if name else None)


# --------------------------------------------------------------------------
# Locator fallback chains
# --------------------------------------------------------------------------


def _build_element_target(step: dict[str, Any]) -> ElementTarget:
    """Build a locator fallback chain from whatever was actually resolved during discovery."""
    resolution = step.get("resolution") or {}
    target = step.get("target") or {}
    locators: list[Any] = []

    strategy = resolution.get("strategy")
    metadata = resolution.get("metadata") or {}
    anchor_text = resolution.get("anchor_text")
    tag = metadata.get("tagName")
    name_attr = metadata.get("name")

    if strategy == "text_near" and anchor_text:
        locators.append(
            TextNearLocator(
                confidence=Confidence.HIGH,
                rationale=(
                    f"Discovered by proximity to the label text {anchor_text!r}; this legacy layout has no "
                    "name/id association between the label and its field."
                ),
                anchor_text=anchor_text,
                relation="right_of",
                element_type=tag,
            )
        )
    elif strategy == "text" and target.get("target_text"):
        locators.append(
            TextLocator(
                confidence=Confidence.HIGH,
                rationale="Matched directly on the element's visible text during discovery.",
                text=target["target_text"],
                exact=False,
            )
        )
    elif strategy == "aria_role" and target.get("target_role"):
        role, name = _parse_role_name(target["target_role"])
        locators.append(
            AriaRoleLocator(
                confidence=Confidence.HIGH,
                rationale="Matched by ARIA role and accessible name during discovery.",
                role=role,
                name=name,
            )
        )
    elif strategy == "description_text" and target.get("target_description"):
        locators.append(
            TextLocator(
                confidence=Confidence.LOW,
                rationale=(
                    "No direct text/role match was found during discovery; fell back to a loose match on "
                    "the free-form target description."
                ),
                text=target["target_description"],
                exact=False,
            )
        )

    if name_attr and tag:
        locators.append(
            CssLocator(
                confidence=Confidence.MEDIUM,
                rationale=(
                    f"The '{name_attr}' form field name is part of the HTML contract and typically more "
                    "stable than surrounding markup, even where CSS classes/ids are not."
                ),
                selector=f"{tag}[name='{name_attr}']",
            )
        )

    box = metadata.get("bounding_box")
    if box:
        locators.append(
            CoordinatesLocator(
                confidence=Confidence.LOW,
                rationale=(
                    "Last-resort fallback captured from the element's on-screen position during discovery; "
                    "breaks under any layout reflow or viewport change."
                ),
                x=round(box["x"] + box["width"] / 2),
                y=round(box["y"] + box["height"] / 2),
            )
        )

    if not locators:
        locators.append(
            TextLocator(
                confidence=Confidence.LOW,
                rationale="No locator signal was captured for this element during discovery.",
                text=target.get("target_description") or target.get("target_text") or "unknown element",
                exact=False,
            )
        )

    return ElementTarget(locators=locators)


# --------------------------------------------------------------------------
# Checkpoints, risk levels
# --------------------------------------------------------------------------


def _build_checkpoint(
    step: dict[str, Any],
    prev_step: dict[str, Any] | None,
    element_target: ElementTarget | None,
    value: str | None,
) -> Condition | None:
    action_type = step["action_type"]

    if action_type in ("type", "select") and element_target is not None and value is not None:
        return ElementHasValueCondition(target=element_target, expected_value=value)

    if action_type == "navigate":
        nav_url = step.get("value")
        path = urlparse(nav_url).path if nav_url else None
        if path:
            return UrlCondition(pattern=path, match_type=UrlMatchType.CONTAINS)
        return None

    if action_type == "click":
        # `page_url` on every step record is captured *after* that step's action
        # ran, so a click's own `page_url` already reflects any resulting
        # navigation - compare against the *previous* step's page_url (not the
        # next step's) to detect whether this click actually caused one.
        this_url = step.get("page_url")
        prev_url = prev_step.get("page_url") if prev_step is not None else None
        if this_url and this_url != prev_url:
            path = urlparse(this_url).path
            if path:
                return UrlCondition(pattern=path, match_type=UrlMatchType.CONTAINS)

    return None


def _fallback_wait_checkpoint(step: dict[str, Any]) -> Condition:
    """A checkpoint for `wait` steps that resolved no element: schema requires one."""
    snippet = (step.get("page_text_snippet") or "").strip()
    for line in snippet.splitlines():
        line = line.strip()
        if line:
            return TextPresentCondition(text=line[:120], case_sensitive=False)
    path = urlparse(step.get("page_url") or "").path
    if path:
        return UrlCondition(pattern=path, match_type=UrlMatchType.CONTAINS)
    return TextPresentCondition(text=(step.get("reasoning") or "page ready")[:120], case_sensitive=False)


def _risk_level(step: dict[str, Any]) -> RiskLevel:
    if step["action_type"] != "click":
        return RiskLevel.SAFE
    target = step.get("target") or {}
    haystack = " ".join(filter(None, [target.get("target_text"), target.get("target_description")])).lower()
    if any(keyword in haystack for keyword in _RISKY_KEYWORDS):
        return RiskLevel.RISKY
    return RiskLevel.SAFE


# --------------------------------------------------------------------------
# Steps / inputs / outputs
# --------------------------------------------------------------------------


def _build_steps(
    discovery_steps: list[dict[str, Any]],
) -> tuple[list[Step], list[InputParameter], list[OutputField]]:
    steps: list[Step] = []
    inputs: list[InputParameter] = []
    mid_flow_outputs: list[OutputField] = []

    used_input_names: set[str] = set()
    used_output_names: set[str] = set()
    used_step_ids: set[str] = set()

    real_steps = [s for s in discovery_steps if s["action_type"] not in _NON_STEP_ACTIONS]

    for idx, step in enumerate(real_steps):
        action_type_str = step["action_type"]
        target = step.get("target") or {}
        needs_target = action_type_str in ("click", "type", "select", "extract")
        element_target = _build_element_target(step) if needs_target else None

        if action_type_str == "type":
            anchor_label = (
                (step.get("resolution") or {}).get("anchor_text")
                or target.get("target_text")
                or target.get("target_description")
                or "value"
            )
            param_name = _unique_name(_slugify(anchor_label.rstrip(": ")), used_input_names)
            inputs.append(
                InputParameter(
                    name=param_name,
                    type=ParamType.STRING,
                    required=True,
                    description=f"Value entered for the field labeled '{anchor_label}'.",
                )
            )
            value: str | None = f"{{{{input.{param_name}}}}}"
        else:
            value = step.get("value")

        if action_type_str == "extract":
            raw_keys = list((step.get("extracted_data") or {}).keys())
            if not raw_keys:
                raw_keys = [_slugify(target.get("target_description"), fallback="extracted_value")]
            for key in raw_keys:
                out_name = _unique_name(_slugify(key), used_output_names)
                extractor = element_target or ElementTarget(
                    locators=[
                        TextLocator(
                            confidence=Confidence.LOW,
                            rationale="No structural locator was captured for this read-only extraction.",
                            text=target.get("target_description") or key,
                            exact=False,
                        )
                    ]
                )
                mid_flow_outputs.append(
                    OutputField(
                        name=out_name,
                        type=ParamType.STRING,
                        description=f"Value read from the page: {target.get('target_description') or key}.",
                        extractor=extractor,
                    )
                )
                step_id = _unique_name(f"extract_{_slugify(key)}", used_step_ids)
                steps.append(
                    Step(
                        step_id=step_id,
                        description=step.get("reasoning") or f"Extract '{key}' from the page.",
                        action=ActionType.EXTRACT,
                        target=extractor,
                        value=None,
                        output_field=out_name,
                        checkpoint=None,
                        risk_level=RiskLevel.SAFE,
                        timeout_ms=5000,
                        pre_conditions=None,
                    )
                )
            continue

        prev_step = real_steps[idx - 1] if idx > 0 else None
        checkpoint = _build_checkpoint(step, prev_step, element_target, value)
        if action_type_str == "wait" and checkpoint is None and element_target is None:
            checkpoint = _fallback_wait_checkpoint(step)

        step_id_source = target.get("target_text") or target.get("target_description") or action_type_str
        step_id = _unique_name(f"{action_type_str}_{_slugify(step_id_source)}", used_step_ids)

        steps.append(
            Step(
                step_id=step_id,
                description=step.get("reasoning") or step_id.replace("_", " "),
                action=ActionType(action_type_str),
                target=element_target,
                value=value,
                output_field=None,
                checkpoint=checkpoint,
                risk_level=_risk_level(step),
                timeout_ms=8000 if action_type_str in ("navigate", "click") else 5000,
                pre_conditions=None,
            )
        )

    return steps, inputs, mid_flow_outputs


def _outputs_from_final_extracted_data(
    final_data: dict[str, Any] | None, used_output_names: set[str]
) -> list[OutputField]:
    outputs: list[OutputField] = []
    for key in final_data or {}:
        name = _unique_name(_slugify(key), used_output_names)
        label = re.sub(r"_", " ", key).strip().title() + ":"
        outputs.append(
            OutputField(
                name=name,
                type=ParamType.STRING,
                description=f"'{key}', read from the final page once the goal was completed.",
                extractor=ElementTarget(
                    locators=[
                        TextNearLocator(
                            confidence=Confidence.LOW,
                            rationale=(
                                "Inferred from the output name, not directly observed: this value was read "
                                "visually by the discovery agent rather than located in the DOM, so this "
                                "locator is a best-effort guess at the label likely to precede it and should "
                                "be verified before relying on it."
                            ),
                            anchor_text=label,
                            relation="right_of",
                            element_type="td",
                        )
                    ]
                ),
            )
        )
    return outputs


# --------------------------------------------------------------------------
# Business outcomes, error handlers, success condition
# --------------------------------------------------------------------------


def _detect_business_outcomes(discovery_steps: list[dict[str, Any]]) -> list[BusinessOutcome]:
    found: dict[str, BusinessOutcome] = {}
    for step in discovery_steps:
        snippet = step.get("page_text_snippet") or ""
        for pattern, name, description in _OUTCOME_PATTERNS:
            if name in found:
                continue
            match = pattern.search(snippet)
            if match:
                found[name] = BusinessOutcome(
                    name=name,
                    description=description,
                    detection=TextPresentCondition(text=match.group(0), case_sensitive=False),
                    severity=Severity.EXPECTED,
                    data_to_extract=None,
                )
    return list(found.values())


def _default_error_handlers() -> list[ErrorHandler]:
    return [
        ErrorHandler(
            name="session_expired",
            detection=UrlCondition(pattern="/login", match_type=UrlMatchType.CONTAINS),
            recovery_action=RecoveryAction.RE_AUTHENTICATE,
            max_retries=1,
            description=(
                "An unexpected redirect back to the login page, indicating the session timed out between "
                "discovery and replay."
            ),
        ),
        ErrorHandler(
            name="unexpected_dialog",
            detection=ElementVisibleCondition(text=None, role="dialog"),
            recovery_action=RecoveryAction.DISMISS_DIALOG,
            max_retries=1,
            description=(
                "A native confirm()/alert() dialog appeared that wasn't part of the recorded flow and must "
                "be dismissed before continuing."
            ),
        ),
    ]


def _derive_success_condition(final_step: dict[str, Any]) -> Condition:
    snippet = final_step.get("page_text_snippet") or ""
    for line in snippet.splitlines():
        line = line.strip()
        if line and any(keyword in line.lower() for keyword in _SUCCESS_KEYWORDS):
            return TextPresentCondition(text=line[:150], case_sensitive=False)
    url = final_step.get("page_url")
    if url:
        path = urlparse(url).path
        if path and path != "/":
            return UrlCondition(pattern=path, match_type=UrlMatchType.CONTAINS)
    reasoning = (final_step.get("reasoning") or "goal complete")[:150]
    return TextPresentCondition(text=reasoning, case_sensitive=False)


def _derive_app_name(discovery_steps: list[dict[str, Any]]) -> str | None:
    for step in discovery_steps:
        title = step.get("page_title")
        if title:
            return title
    return None


def _artifact_name(goal: str) -> str:
    slug = _slugify(goal, fallback="discovered_flow")
    words = slug.split("_")
    if len(words) > 8:
        slug = "_".join(words[:8])
    return slug


def _next_version(output_dir: Path, name: str) -> int:
    existing_path = output_dir / f"{name}.json"
    if not existing_path.exists():
        return 1
    try:
        existing = json.loads(existing_path.read_text())
        return int(existing.get("version", 1)) + 1
    except (json.JSONDecodeError, OSError, ValueError, TypeError):
        return 1


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def emit_artifact(discovery_log: dict[str, Any], output_dir: Path | None = None) -> Path:
    """Convert a raw discovery log (as produced by `run_discovery`) into a saved Artifact.

    Returns the path the artifact was written to.
    """
    output_dir = output_dir or SAVED_ARTIFACTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    discovery_steps: list[dict[str, Any]] = discovery_log.get("steps") or []
    steps, inputs, mid_flow_outputs = _build_steps(discovery_steps)

    if not steps:
        raise ValueError("emit_artifact: discovery log has no replayable steps.")

    used_output_names = {o.name for o in mid_flow_outputs}
    final_outputs = _outputs_from_final_extracted_data(discovery_log.get("final_extracted_data"), used_output_names)
    outputs = mid_flow_outputs + final_outputs

    business_outcomes = _detect_business_outcomes(discovery_steps)
    error_handlers = _default_error_handlers()

    final_step = discovery_steps[-1] if discovery_steps else {}
    success_condition = _derive_success_condition(final_step)

    entry_url = (
        discovery_log.get("target_url")
        or (discovery_steps[0].get("page_url") if discovery_steps else None)
        or "http://localhost/"
    )
    app_name = _derive_app_name(discovery_steps) or "Discovered Application"

    name = _artifact_name(discovery_log.get("goal") or "discovered_flow")
    version = _next_version(output_dir, name)

    artifact = Artifact(
        name=name,
        version=version,
        description=discovery_log.get("goal") or "Discovered automation flow.",
        created_at=datetime.now(timezone.utc),
        created_by="discovery-agent-v1",
        target=Target(surface_type=SurfaceType.WEB, entry_url=entry_url, app_name=app_name),
        inputs=inputs,
        outputs=outputs,
        steps=steps,
        business_outcomes=business_outcomes,
        error_handlers=error_handlers,
        success_condition=success_condition,
    )

    # Redact before this hits disk: page-derived text (business-outcome
    # detection strings, descriptions) can carry account numbers, SSNs, or
    # other sensitive values lifted from the page during discovery.
    redacted = redact_dict(artifact.model_dump(mode="json"))
    path = output_dir / f"{name}.json"
    path.write_text(json.dumps(redacted, indent=2))
    return path
