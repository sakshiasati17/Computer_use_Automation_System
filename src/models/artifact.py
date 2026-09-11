"""The UI automation artifact: a recorded, replayable "capability".

An artifact is produced once by an LLM driving a legacy web app (discovery),
and replayed many times afterward by a deterministic engine with no LLM in
the loop. It is designed to be called like a function: typed `inputs` in,
typed `outputs` out, plus everything the replay engine needs to survive
runtime reality — fallback locators, checkpoints after every step, known
non-error outcomes, and recoverable-error handling.

The schema deliberately separates three concerns that are easy to conflate:

1. **What the flow does** — `steps`, each a single UI action with a
   locator fallback chain and a checkpoint.
2. **What "done" can legitimately mean** — `success_condition` for the
   happy path, `business_outcomes` for expected-but-not-success results
   (e.g. "member not found" is not a bug).
3. **What can go wrong that ISN'T a business outcome** — `error_handlers`,
   for transient/recoverable runtime conditions (session timeouts, slow
   loads, unexpected dialogs) that should be retried or escalated rather
   than surfaced as a business result.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .actions import ActionType, Condition, ElementTarget, TEMPLATE_REF_PATTERN

__all__ = [
    "SurfaceType",
    "ParamType",
    "RiskLevel",
    "Severity",
    "RecoveryAction",
    "Target",
    "InputParameter",
    "OutputField",
    "Step",
    "BusinessOutcome",
    "ErrorHandler",
    "Artifact",
]


class SurfaceType(str, Enum):
    """The kind of UI surface the artifact was recorded against."""

    WEB = "web"
    DESKTOP = "desktop"
    NATIVE = "native"


class ParamType(str, Enum):
    """Primitive types supported for artifact inputs and outputs."""

    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"


class RiskLevel(str, Enum):
    """How consequential a step is if it turns out to be wrong.

    `risky` and `irreversible` steps (submitting a payment, opening an
    account, deleting a record) should require human approval before the
    replay engine executes them; `safe` steps (navigation, search,
    reading data) can run unattended.
    """

    SAFE = "safe"
    RISKY = "risky"
    IRREVERSIBLE = "irreversible"


class Severity(str, Enum):
    """How a hit business outcome should be treated by the caller."""

    EXPECTED = "expected"
    """A normal, anticipated result. The caller handles it like any other
    return value — no warning needed."""

    UNUSUAL = "unusual"
    """A valid, non-error result that is nonetheless worth flagging — log a
    warning so a human notices the pattern even though nothing broke."""


class RecoveryAction(str, Enum):
    """The remedial action an error handler takes when its condition fires."""

    RETRY_STEP = "retry_step"
    DISMISS_DIALOG = "dismiss_dialog"
    RE_AUTHENTICATE = "re_authenticate"
    WAIT_AND_RETRY = "wait_and_retry"
    ESCALATE = "escalate"


class Target(BaseModel):
    """Identifies the application surface an artifact was recorded against."""

    model_config = ConfigDict(extra="forbid")

    surface_type: SurfaceType
    entry_url: str = Field(
        ..., min_length=1, description="URL (web) or launch identifier (desktop/native) to start from."
    )
    app_name: str = Field(..., min_length=1, description="Human-readable name of the target application.")

    @model_validator(mode="after")
    def _web_entry_url_looks_like_a_url(self) -> "Target":
        if self.surface_type is SurfaceType.WEB and "://" not in self.entry_url:
            raise ValueError(
                f"entry_url {self.entry_url!r} does not look like a URL for surface_type='web'."
            )
        return self


class InputParameter(BaseModel):
    """A typed parameter the caller supplies when invoking the artifact."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$", description="Snake_case parameter name.")
    type: ParamType
    required: bool = True
    description: str = Field(..., min_length=1)
    default_value: str | float | bool | None = Field(
        default=None, description="Used when the caller omits this parameter and required=False."
    )

    @model_validator(mode="after")
    def _defaults_only_when_optional(self) -> "InputParameter":
        if self.required and self.default_value is not None:
            raise ValueError(f"input {self.name!r} is required but also declares a default_value.")
        return self


class OutputField(BaseModel):
    """A typed value the artifact extracts from the page and returns."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$", description="Snake_case output name.")
    type: ParamType
    description: str = Field(..., min_length=1)
    extractor: ElementTarget = Field(
        ..., description="Where on the page to read this value from (fallback locator chain)."
    )


class Step(BaseModel):
    """A single, ordered action in the recorded flow.

    Field requirements vary by `action` (e.g. `click` needs `target`,
    `type` needs `target` and `value`) and are enforced in
    :meth:`_validate_action_requirements` rather than left to convention.
    """

    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(..., min_length=1, description="Unique identifier, e.g. 'login_enter_username'.")
    description: str = Field(..., min_length=1, description="Human-readable summary of this step.")
    action: ActionType
    target: ElementTarget | None = Field(
        default=None, description="Element to act on. Required for click/type/select/extract."
    )
    value: str | None = Field(
        default=None,
        description=(
            "Value to type/select, or the URL to navigate to. Supports template syntax "
            "like '{{input.member_id}}' referencing a declared input parameter."
        ),
    )
    output_field: str | None = Field(
        default=None,
        description="Name of the OutputField this 'extract' step populates. Required for action=extract.",
    )
    checkpoint: Condition | None = Field(
        default=None, description="Condition that must hold after this step to consider it successful."
    )
    risk_level: RiskLevel = Field(
        default=RiskLevel.SAFE, description="Whether this step needs human approval before running."
    )
    timeout_ms: int = Field(default=5000, gt=0, description="How long to wait before failing this step.")
    pre_conditions: list[Condition] | None = Field(
        default=None, description="Conditions that must already hold before this step is attempted."
    )

    @model_validator(mode="after")
    def _validate_action_requirements(self) -> "Step":
        needs_target = {ActionType.CLICK, ActionType.TYPE, ActionType.SELECT, ActionType.EXTRACT}
        needs_value = {ActionType.TYPE, ActionType.SELECT}

        if self.action in needs_target and self.target is None:
            raise ValueError(f"step {self.step_id!r}: action={self.action.value!r} requires 'target'.")
        if self.action in needs_value and self.value is None:
            raise ValueError(f"step {self.step_id!r}: action={self.action.value!r} requires 'value'.")
        if self.action is ActionType.NAVIGATE and not self.value:
            raise ValueError(f"step {self.step_id!r}: action='navigate' requires 'value' (the URL).")
        if self.action is ActionType.EXTRACT and not self.output_field:
            raise ValueError(f"step {self.step_id!r}: action='extract' requires 'output_field'.")
        if self.action is ActionType.WAIT and self.checkpoint is None and self.target is None:
            raise ValueError(
                f"step {self.step_id!r}: action='wait' requires 'checkpoint' and/or 'target' "
                "to know what it's waiting for."
            )
        return self


class BusinessOutcome(BaseModel):
    """A legitimate, expected result of running the flow that is NOT success.

    Example: searching for a member who doesn't exist. The flow executed
    correctly and reached a known, valid state — it just isn't the happy
    path. Conflating this with a failure is exactly the design mistake this
    schema exists to prevent; see `results.py`.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$", description="e.g. 'member_not_found'.")
    description: str = Field(..., min_length=1)
    detection: Condition = Field(..., description="Condition that identifies this outcome occurred.")
    severity: Severity = Field(default=Severity.EXPECTED)
    data_to_extract: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional map of field name -> human-readable description of what to pull "
            "from the page when this outcome is detected, e.g. "
            "{'searched_id': 'the member ID that was searched for'}."
        ),
    )


class ErrorHandler(BaseModel):
    """A recoverable runtime condition the replay engine watches for on every step.

    Distinct from `BusinessOutcome`: these are not valid results of the
    flow, they are infrastructure/UI hiccups (a stale session, a slow
    server, a rogue confirm() dialog) that should be recovered from, not
    reported as the flow's outcome.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., pattern=r"^[a-z][a-z0-9_]*$", description="e.g. 'session_expired'.")
    detection: Condition = Field(..., description="Condition that identifies this error occurred.")
    recovery_action: RecoveryAction
    max_retries: int = Field(default=1, ge=0)
    description: str = Field(..., min_length=1)


class Artifact(BaseModel):
    """A recorded, replayable UI automation capability.

    Invariants enforced beyond field-level types:

    - `step_id`, input names, output names, business-outcome names, and
      error-handler names are each unique within the artifact.
    - Every `{{input.X}}` template reference used in a step's `value` names
      a declared input parameter (catches typos/renames at save time
      rather than at replay time, mid-flow, in production).
    - Every `extract` step's `output_field` names a declared output.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(
        default="1.0.0",
        pattern=r"^\d+\.\d+\.\d+$",
        description="Version of this artifact schema format, for forward compatibility.",
    )
    name: str = Field(
        ..., pattern=r"^[a-z][a-z0-9_]*$", description="Machine-readable identifier, e.g. 'lookup_member_balance'."
    )
    version: int = Field(default=1, ge=1, description="Increments each time this artifact is re-recorded.")
    description: str = Field(..., min_length=1)
    created_at: datetime
    created_by: str = Field(..., min_length=1, description="e.g. 'discovery-agent-v1'.")
    target: Target

    inputs: list[InputParameter] = Field(default_factory=list)
    outputs: list[OutputField] = Field(default_factory=list)
    steps: list[Step] = Field(..., min_length=1)
    business_outcomes: list[BusinessOutcome] = Field(default_factory=list)
    error_handlers: list[ErrorHandler] = Field(default_factory=list)
    success_condition: Condition = Field(
        ..., description="Condition checked at the end of the flow to confirm the goal was achieved."
    )

    @model_validator(mode="after")
    def _validate_cross_references(self) -> "Artifact":
        def _dupes(names: list[str]) -> set[str]:
            seen: set[str] = set()
            dupes: set[str] = set()
            for n in names:
                (dupes if n in seen else seen).add(n)
            return dupes

        if dupes := _dupes([s.step_id for s in self.steps]):
            raise ValueError(f"duplicate step_id(s): {sorted(dupes)}")

        input_names = {i.name for i in self.inputs}
        if dupes := _dupes([i.name for i in self.inputs]):
            raise ValueError(f"duplicate input parameter name(s): {sorted(dupes)}")

        output_names = {o.name for o in self.outputs}
        if dupes := _dupes([o.name for o in self.outputs]):
            raise ValueError(f"duplicate output field name(s): {sorted(dupes)}")

        if dupes := _dupes([b.name for b in self.business_outcomes]):
            raise ValueError(f"duplicate business_outcome name(s): {sorted(dupes)}")

        if dupes := _dupes([e.name for e in self.error_handlers]):
            raise ValueError(f"duplicate error_handler name(s): {sorted(dupes)}")

        for step in self.steps:
            if step.value:
                for ref in TEMPLATE_REF_PATTERN.findall(step.value):
                    if ref not in input_names:
                        raise ValueError(
                            f"step {step.step_id!r} references undeclared input "
                            f"'{{{{input.{ref}}}}}' (declared inputs: {sorted(input_names)})."
                        )
            if step.action is ActionType.EXTRACT and step.output_field not in output_names:
                raise ValueError(
                    f"step {step.step_id!r} extracts into undeclared output "
                    f"{step.output_field!r} (declared outputs: {sorted(output_names)})."
                )

        return self
