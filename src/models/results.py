"""Execution results for a single artifact run.

The central design decision here is the three-way split of what a run can
produce. Interface.ai flags conflating business outcomes and failures as
the most common design mistake in this space, so the type system makes it
structurally impossible: `ExecutionResult.outcome` is a discriminated union
and callers must branch on `outcome.type` before they can reach any field.

- **SuccessOutcome** — the goal was achieved; here are the extracted outputs.
- **BusinessOutcomeOutcome** — a legitimate, expected, non-success result
  (e.g. "member not found"). Not an error. The caller handles it as a normal
  return value, exactly like `Artifact.business_outcomes` promised it would.
- **HardFailureOutcome** — something actually broke. Carries everything
  needed to debug without re-running: which step, what was expected vs.
  observed, and a screenshot.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated, Literal, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .actions import ActionType, Confidence
from .artifact import RiskLevel, Severity

__all__ = [
    "ExecutionOutcomeType",
    "StepStatus",
    "EscalationResolution",
    "LocatorAttempt",
    "StepExecutionRecord",
    "SuccessOutcome",
    "BusinessOutcomeOutcome",
    "HardFailureOutcome",
    "ExecutionOutcome",
    "Escalation",
    "ExecutionResult",
]


class ExecutionOutcomeType(str, Enum):
    """The three, mutually exclusive shapes an artifact run can end in."""

    SUCCESS = "success"
    BUSINESS_OUTCOME = "business_outcome"
    HARD_FAILURE = "hard_failure"


class StepStatus(str, Enum):
    """The result of attempting a single step during replay."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRIED = "retried"


class EscalationResolution(str, Enum):
    """How a human-in-the-loop escalation was resolved."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    TIMED_OUT = "timed_out"


class LocatorAttempt(BaseModel):
    """One try of one locator strategy from an ElementTarget's fallback chain."""

    model_config = ConfigDict(extra="forbid")

    strategy: str = Field(..., description="Locator strategy name, e.g. 'aria_role', 'css'.")
    confidence: Confidence
    succeeded: bool
    error_message: str | None = Field(default=None, description="Set when succeeded=False.")


class StepExecutionRecord(BaseModel):
    """A log entry for one step attempted during replay."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    action: ActionType
    description: str
    status: StepStatus
    started_at: datetime
    duration_ms: int = Field(..., ge=0)
    retries: int = Field(default=0, ge=0)
    locator_attempts: list[LocatorAttempt] = Field(default_factory=list)
    error_message: str | None = Field(default=None, description="Set when status='failed'.")


class SuccessOutcome(BaseModel):
    """The flow reached its declared success_condition."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["success"] = "success"
    outputs: dict[str, str | float | bool] = Field(
        ..., description="Values extracted per the artifact's declared OutputFields."
    )


class BusinessOutcomeOutcome(BaseModel):
    """The flow matched one of the artifact's declared BusinessOutcomes.

    Not an error: this is a valid, anticipated result and the caller is
    expected to branch on `outcome_name` as part of normal handling.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["business_outcome"] = "business_outcome"
    outcome_name: str = Field(..., description="Matches a BusinessOutcome.name on the artifact.")
    description: str
    severity: Severity
    extracted_data: dict[str, str | float | bool] = Field(default_factory=dict)


class HardFailureOutcome(BaseModel):
    """Something actually broke — this did not match success or any known outcome.

    Carries enough to debug without re-running the flow: the failing step,
    what was expected vs. what was actually observed, and a screenshot.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["hard_failure"] = "hard_failure"
    failed_step_id: str
    action_attempted: ActionType
    expected: str = Field(..., description="What the step/checkpoint expected to find or happen.")
    observed: str = Field(..., description="What was actually found/observed on the page instead.")
    screenshot_path: str | None = Field(default=None)
    page_url: str | None = Field(default=None, description="URL at the time of failure.")
    exception_type: str | None = Field(default=None, description="e.g. 'TimeoutError', 'ElementNotFound'.")
    message: str = Field(..., min_length=1, description="Human-readable summary of what went wrong.")
    locator_attempts: list[LocatorAttempt] = Field(
        default_factory=list, description="Every locator in the fallback chain that was tried and failed."
    )


ExecutionOutcome = Annotated[
    Union[SuccessOutcome, BusinessOutcomeOutcome, HardFailureOutcome],
    Field(discriminator="type"),
]
"""Discriminated union of the three run outcomes, tagged by `type`.

Callers must check `outcome.type` before accessing outcome-specific fields —
there is no field on ExecutionResult that is populated in all three cases,
by design, so success/business/failure handling cannot be silently conflated.
"""


class Escalation(BaseModel):
    """A point where the replay engine paused for human input.

    Triggered by a step's `risk_level` (risky/irreversible) or by an
    ErrorHandler with `recovery_action='escalate'`.
    """

    model_config = ConfigDict(extra="forbid")

    escalation_id: UUID = Field(default_factory=uuid4)
    step_id: str
    reason: str = Field(..., min_length=1)
    risk_level: RiskLevel
    requested_at: datetime
    resolution: EscalationResolution = Field(default=EscalationResolution.PENDING)
    resolved_at: datetime | None = Field(default=None)
    resolved_by: str | None = Field(default=None, description="Who/what resolved this, e.g. a user email.")
    notes: str | None = Field(default=None)

    @model_validator(mode="after")
    def _resolution_consistency(self) -> "Escalation":
        if self.resolution is EscalationResolution.PENDING and self.resolved_at is not None:
            raise ValueError("escalation is 'pending' but has a resolved_at timestamp.")
        if self.resolution is not EscalationResolution.PENDING and self.resolved_at is None:
            raise ValueError(f"escalation resolution={self.resolution.value!r} requires resolved_at.")
        return self


class ExecutionResult(BaseModel):
    """The full record of one attempt to replay an artifact."""

    model_config = ConfigDict(extra="forbid")

    run_id: UUID = Field(default_factory=uuid4)
    artifact_name: str
    artifact_version: int = Field(..., ge=1)
    input_params: dict[str, str | float | bool] = Field(default_factory=dict)
    started_at: datetime
    completed_at: datetime
    total_duration_ms: int = Field(..., ge=0)
    outcome: ExecutionOutcome
    execution_log: list[StepExecutionRecord] = Field(default_factory=list)
    escalations: list[Escalation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _timestamps_consistent(self) -> "ExecutionResult":
        if self.completed_at < self.started_at:
            raise ValueError("completed_at precedes started_at.")
        return self
