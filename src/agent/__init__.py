"""Agent loop primitives: observing page state and deciding the next action.

- `observer`: captures a screenshot + accessibility tree of the current page
  and asks Claude for the single next action to take toward a stated goal.
"""

from .observer import AgentAction, AgentActionType, observe_and_decide

__all__ = [
    "AgentAction",
    "AgentActionType",
    "observe_and_decide",
]
