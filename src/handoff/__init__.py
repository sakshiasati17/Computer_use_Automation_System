"""Human-in-the-loop escalation: pausing replay for risky/stuck/failed steps
and handing control to a human via a small operator web UI.

See `escalation.py` for the control-transfer model and `operator_server.py`
for the FastAPI app a human interacts with at http://127.0.0.1:8080.
"""

from .escalation import (
    ControlState,
    DEFAULT_CDP_PORT,
    DEFAULT_OPERATOR_PORT,
    EscalationHub,
    EscalationOutcome,
    EscalationReason,
    InterventionRequest,
    escalate,
    get_hub,
)

__all__ = [
    "ControlState",
    "DEFAULT_CDP_PORT",
    "DEFAULT_OPERATOR_PORT",
    "EscalationHub",
    "EscalationOutcome",
    "EscalationReason",
    "InterventionRequest",
    "escalate",
    "get_hub",
]
