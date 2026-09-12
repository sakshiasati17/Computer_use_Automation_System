"""Structured run evidence: what the agent did, why, and what to look at on failure.

Both `src.agent.discovery` (an LLM deciding one step at a time) and
`src.replay.engine` (a deterministic engine executing a saved `Artifact`)
need the same thing once a run ends: a clean, reviewable record of every
step taken, plus enough extra signal to debug a failure without re-running
it. `RunLogger` is the single place that record gets built up during a run;
`RunEvidence` is its final, saved shape; `save_evidence` is the only place
that shape touches disk.

Two things are deliberately centralized here rather than duplicated per
call site (discovery previously wrote its own ad-hoc screenshot/URL/title
capture on failure, and replay's `_save_failure_screenshot` did the same):

- **Failure enrichment.** `RunLogger.capture_failure_context` is the one
  place that takes a screenshot and reads `page.url`/`page.title()` for the
  evidence record, so "what did the page look like when this broke" always
  has the same shape regardless of which run type asked for it.
- **Redaction before disk.** `save_evidence` runs `redact_dict` over the
  entire evidence payload before writing, mirroring the rule already
  enforced for artifacts and discovery logs (see `src/safety/redactor.py`):
  nothing derived from the target application reaches disk unredacted.
  `RunLogger` also redacts `target`/`value` as each step is recorded, since
  those are the fields most likely to carry a typed password or account
  number and the cheapest to catch immediately rather than waiting for the
  final save.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..models import Escalation
from ..safety import redact, redact_dict

__all__ = [
    "RunType",
    "RunOutcome",
    "StepEvidence",
    "RunEvidence",
    "RunLogger",
    "save_evidence",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVIDENCE_ROOT = REPO_ROOT / "evidence"


class RunType(str, Enum):
    """Which of the two run kinds this evidence describes."""

    DISCOVERY = "discovery"
    REPLAY = "replay"


class RunOutcome(str, Enum):
    """How a run ended, for both filing (`save_evidence`) and triage.

    Mirrors the success/business-outcome/hard-failure split used elsewhere
    (`src/models/results.py`), plus `ESCALATED` for a run that ended with an
    unresolved human handoff rather than any of the other three.
    """

    SUCCESS = "SUCCESS"
    BUSINESS_OUTCOME = "BUSINESS_OUTCOME"
    HARD_FAILURE = "HARD_FAILURE"
    ESCALATED = "ESCALATED"


class StepEvidence(BaseModel):
    """One entry in a run's step-by-step log.

    `target`/`value` are redacted twice: once here, as soon as
    `RunLogger.record_step` builds this record (catches an obviously
    sensitive value immediately), and again over the whole evidence payload
    in `save_evidence` (catches anything that slipped in via `extra`, or
    that only becomes recognizable in context, e.g. a label in a sibling
    field). Redacting twice is safe: `redact()` is idempotent against its
    own `[REDACTED-*]` placeholders.
    """

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(..., min_length=1)
    action: str = Field(..., min_length=1, description="e.g. 'click', 'type', 'navigate'.")
    target: str | None = Field(default=None, description="What the action acted on, redacted.")
    value: str | None = Field(default=None, description="Value typed/selected/navigated to, redacted.")
    result: str = Field(..., min_length=1, description="e.g. 'passed', 'failed', 'blocked', 'retried'.")
    duration_ms: int = Field(..., ge=0)
    screenshot_path: str | None = Field(default=None)
    started_at: datetime | None = Field(default=None)
    reasoning: str | None = Field(default=None, description="Why this action was chosen, when known.")
    error_message: str | None = Field(default=None)
    page_url: str | None = Field(default=None)
    page_title: str | None = Field(default=None)
    retries: int = Field(default=0, ge=0)
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Anything else worth keeping for this step (locator attempts, risk_level, "
            "safety_blocked, extracted_data, ...) without forcing every caller into one rigid shape."
        ),
    )


class RunEvidence(BaseModel):
    """The complete, reviewable record of one discovery or replay run."""

    model_config = ConfigDict(extra="forbid")

    run_id: UUID
    run_type: RunType
    started_at: datetime
    completed_at: datetime
    goal: str | None = Field(default=None, description="Discovery runs: the natural-language goal.")
    artifact_name: str | None = Field(default=None, description="Replay runs: the artifact that was replayed.")
    input_params: dict[str, Any] = Field(default_factory=dict)
    outcome: RunOutcome
    steps: list[StepEvidence] = Field(default_factory=list)
    escalations: list[Escalation] = Field(default_factory=list, description="Human interventions during this run.")
    safety_violations: list[dict[str, Any]] = Field(
        default_factory=list, description="Allowlist/risk-classifier violations hit during this run."
    )
    total_duration_ms: int = Field(..., ge=0)
    error_details: dict[str, Any] | None = Field(
        default=None, description="Extra debugging context captured when outcome != SUCCESS."
    )

    @model_validator(mode="after")
    def _timestamps_consistent(self) -> "RunEvidence":
        if self.completed_at < self.started_at:
            raise ValueError("completed_at precedes started_at.")
        return self


class RunLogger:
    """Accumulates one run's evidence as it happens, for a caller to `finish()` at the end.

    Usage (either discovery or replay):

        logger = RunLogger(RunType.DISCOVERY, goal=goal, page=page)
        start = logger.start_timer()
        ...perform the action...
        logger.record_step(step_id=..., action=..., target=..., value=...,
                            result="passed", start_perf=start)
        ...
        evidence = await logger.finish(RunOutcome.SUCCESS)
        save_evidence(evidence)

    `page` is optional and can be set later via `set_page` (a replay/
    discovery session's Playwright `Page` is usually created after the
    logger itself); when set, `capture_failure_context` and the automatic
    enrichment in `finish()` can take a screenshot and read `page.url`/
    `page.title()`. Without a `page`, both degrade to a no-op rather than
    raising, since a logger must never be the reason a run fails.
    """

    def __init__(
        self,
        run_type: RunType | str,
        *,
        goal: str | None = None,
        artifact_name: str | None = None,
        input_params: dict[str, Any] | None = None,
        page: Any | None = None,
        evidence_root: Path | None = None,
    ) -> None:
        self.run_id = uuid4()
        self.run_type = RunType(run_type)
        self.started_at = datetime.now(timezone.utc)
        self.goal = goal
        self.artifact_name = artifact_name
        self.input_params: dict[str, Any] = input_params or {}
        self.page = page
        self.steps: list[StepEvidence] = []
        self.escalations: list[Escalation] = []
        self.safety_violations: list[dict[str, Any]] = []

        evidence_root = evidence_root or DEFAULT_EVIDENCE_ROOT
        self.screenshots_dir = evidence_root / self.run_type.value / "screenshots" / str(self.run_id)

    def set_page(self, page: Any) -> None:
        """Attach (or replace) the Playwright `Page` used for failure capture."""
        self.page = page

    @staticmethod
    def start_timer() -> float:
        """A `time.monotonic()` marker to pass to `record_step` as `start_perf`."""
        return time.monotonic()

    def record_step(
        self,
        *,
        step_id: str,
        action: str,
        result: str,
        target: str | None = None,
        value: str | None = None,
        start_perf: float | None = None,
        duration_ms: int | None = None,
        screenshot_path: str | None = None,
        started_at: datetime | None = None,
        reasoning: str | None = None,
        error_message: str | None = None,
        page_url: str | None = None,
        page_title: str | None = None,
        retries: int = 0,
        extra: dict[str, Any] | None = None,
    ) -> StepEvidence:
        """Record one step. Pass either `start_perf` (from `start_timer()`) or `duration_ms` directly."""
        if duration_ms is None:
            if start_perf is None:
                raise ValueError("record_step requires either 'start_perf' or 'duration_ms'.")
            duration_ms = int((time.monotonic() - start_perf) * 1000)

        step = StepEvidence(
            step_id=step_id,
            action=action,
            target=redact(target) if target else target,
            value=redact(value) if value else value,
            result=result,
            duration_ms=duration_ms,
            screenshot_path=screenshot_path,
            started_at=started_at or self.started_at,
            reasoning=reasoning,
            error_message=error_message,
            page_url=page_url,
            page_title=page_title,
            retries=retries,
            extra=extra or {},
        )
        self.steps.append(step)
        return step

    def record_safety_violation(self, violation: dict[str, Any]) -> None:
        self.safety_violations.append(violation)

    def record_escalation(self, escalation: Escalation) -> None:
        self.escalations.append(escalation)

    async def capture_failure_context(self, tag: str) -> dict[str, Any]:
        """Best-effort screenshot + page URL + page title, for a failing step or run.

        Never raises: a failure captured to help debugging must not itself
        cause a new failure. Returns `{}` (not an error) when there is no
        `page` attached or every capture attempt fails.
        """
        if self.page is None:
            return {}

        context: dict[str, Any] = {}
        try:
            context["page_url"] = self.page.url
        except Exception:
            pass
        try:
            context["page_title"] = await self.page.title()
        except Exception:
            pass
        try:
            self.screenshots_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            path = self.screenshots_dir / f"{tag}_{timestamp}.png"
            await self.page.screenshot(path=str(path), full_page=True)
            context["screenshot_path"] = str(path)
        except Exception:
            pass
        return context

    async def finish(
        self,
        outcome: RunOutcome | str,
        *,
        error_details: dict[str, Any] | None = None,
    ) -> RunEvidence:
        """Build the final `RunEvidence` for this run. Does not save it — see `save_evidence`.

        When the run didn't succeed and the caller hasn't already supplied
        `error_details` (e.g. because a step-level failure already captured
        one), this makes one last best-effort attempt to enrich it with the
        current page's screenshot/URL/title.
        """
        outcome = RunOutcome(outcome)
        if outcome is not RunOutcome.SUCCESS and error_details is None:
            captured = await self.capture_failure_context(f"run_{self.run_id}_final")
            error_details = captured or None

        completed_at = datetime.now(timezone.utc)
        return RunEvidence(
            run_id=self.run_id,
            run_type=self.run_type,
            started_at=self.started_at,
            completed_at=completed_at,
            goal=self.goal,
            artifact_name=self.artifact_name,
            input_params=self.input_params,
            outcome=outcome,
            steps=self.steps,
            escalations=self.escalations,
            safety_violations=self.safety_violations,
            total_duration_ms=int((completed_at - self.started_at).total_seconds() * 1000),
            error_details=error_details,
        )


def save_evidence(evidence: RunEvidence, evidence_root: Path | None = None) -> Path:
    """Redact and persist `evidence` to the directory its run type/outcome dictates.

    - `evidence/discovery/` for every discovery run, regardless of outcome.
    - `evidence/replay/` for a replay run that succeeded.
    - `evidence/failure/` for a replay run that didn't (business outcome,
      hard failure, or an unresolved escalation) — grouping these together
      is deliberate: none of them is "just" a clean success, and a human
      triaging failures should find all three in one place.
    """
    evidence_root = evidence_root or DEFAULT_EVIDENCE_ROOT

    if evidence.run_type is RunType.DISCOVERY:
        target_dir = evidence_root / "discovery"
    elif evidence.outcome is RunOutcome.SUCCESS:
        target_dir = evidence_root / "replay"
    else:
        target_dir = evidence_root / "failure"
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp = evidence.completed_at.strftime("%Y%m%dT%H%M%S%f")
    filename = f"{evidence.run_type.value}_{evidence.run_id}_{timestamp}.json"
    path = target_dir / filename

    redacted = redact_dict(evidence.model_dump(mode="json"))
    path.write_text(json.dumps(redacted, indent=2, default=str))
    return path
