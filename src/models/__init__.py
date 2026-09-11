"""Pydantic data models for the UI automation artifact system.

- `actions`: locator strategies (how to find an element) and page-state
  conditions (how to tell what state the page is in).
- `artifact`: the recorded, replayable capability — metadata, typed
  inputs/outputs, steps, business outcomes, and error handlers.
- `results`: the outcome of replaying an artifact, with a strict
  success / business-outcome / hard-failure split.
"""

from .actions import (
    ActionType,
    AriaRoleLocator,
    Condition,
    Confidence,
    CoordinatesLocator,
    CssLocator,
    ElementHasValueCondition,
    ElementTarget,
    ElementVisibleCondition,
    Locator,
    TextLocator,
    TextNearLocator,
    TextPresentCondition,
    UrlCondition,
    UrlMatchType,
)
from .artifact import (
    Artifact,
    BusinessOutcome,
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
)
from .results import (
    BusinessOutcomeOutcome,
    Escalation,
    EscalationResolution,
    ExecutionOutcome,
    ExecutionOutcomeType,
    ExecutionResult,
    HardFailureOutcome,
    LocatorAttempt,
    StepExecutionRecord,
    StepStatus,
    SuccessOutcome,
)

__all__ = [
    # actions
    "ActionType",
    "AriaRoleLocator",
    "Condition",
    "Confidence",
    "CoordinatesLocator",
    "CssLocator",
    "ElementHasValueCondition",
    "ElementTarget",
    "ElementVisibleCondition",
    "Locator",
    "TextLocator",
    "TextNearLocator",
    "TextPresentCondition",
    "UrlCondition",
    "UrlMatchType",
    # artifact
    "Artifact",
    "BusinessOutcome",
    "ErrorHandler",
    "InputParameter",
    "OutputField",
    "ParamType",
    "RecoveryAction",
    "RiskLevel",
    "Severity",
    "Step",
    "SurfaceType",
    "Target",
    # results
    "BusinessOutcomeOutcome",
    "Escalation",
    "EscalationResolution",
    "ExecutionOutcome",
    "ExecutionOutcomeType",
    "ExecutionResult",
    "HardFailureOutcome",
    "LocatorAttempt",
    "StepExecutionRecord",
    "StepStatus",
    "SuccessOutcome",
]
