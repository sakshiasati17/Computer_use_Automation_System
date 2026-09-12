"""Observation module: capture page state and ask Claude for the next action.

Legacy banking apps offer no test IDs, no semantic HTML, and often bury
controls in nested tables or iframes. An ARIA snapshot is sent to Claude
alongside the screenshot because it captures logical structure (role, name,
value) that survives poor markup quality, unlike raw DOM/CSS selectors —
the same tradeoff ``ElementTarget`` in ``src/models/actions.py`` makes at
the artifact-recording layer. ``page.aria_snapshot(mode="ai")`` is used
rather than the older ``page.accessibility.snapshot()`` (removed from
current Playwright): it also renders the content of same-origin iframes
inline, which matters here since this app embeds member details in one.

This module performs a single observe -> decide cycle. It does not run a
loop; the caller is responsible for executing the returned ``AgentAction``
and invoking :func:`observe_and_decide` again with updated history.
"""

from __future__ import annotations

import base64
import json
import logging
from enum import Enum

import anthropic
from playwright.async_api import Page
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..models import Confidence

__all__ = ["AgentActionType", "AgentAction", "observe_and_decide"]

logger = logging.getLogger(__name__)

MODEL_ID = "claude-sonnet-4-6"
MAX_TOKENS = 8192
MAX_ATTEMPTS = 2

_client: anthropic.AsyncAnthropic | None = None


def _get_client() -> anthropic.AsyncAnthropic:
    """Lazily construct the Anthropic client.

    Deferred past import time so importing this module doesn't require
    ANTHROPIC_API_KEY to already be set. The SDK resolves credentials from
    the environment (ANTHROPIC_API_KEY, then other fallbacks) on its own.
    """
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic()
    return _client


class AgentActionType(str, Enum):
    """The single next action Claude can choose to take."""

    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    NAVIGATE = "navigate"
    WAIT = "wait"
    EXTRACT_DATA = "extract_data"
    GOAL_COMPLETE = "goal_complete"
    STUCK = "stuck"


class AgentAction(BaseModel):
    """Claude's decision for the single next step to take on the page.

    Target fields are deliberately loose (visible text / ARIA role or
    label / a positional description) rather than a selector, since these
    apps have no stable HTML ids or CSS classes to key off of.
    """

    model_config = ConfigDict(extra="forbid")

    action_type: AgentActionType
    target_description: str = Field(
        ...,
        min_length=1,
        description=(
            "Required for every action. For click/type/select: how you identified the "
            "element — its nearby label text and/or position, e.g. 'the textbox in the "
            "row labeled Username:', not just a restatement of target_role. This field, "
            "not 'reasoning', is what a replay engine uses to find the element, so it must "
            "name enough context to be unique even when target_role is shared by several "
            "elements. For other action types: briefly state what the action targets or why."
        ),
    )
    target_text: str | None = Field(
        default=None, description="Visible text on or identifying the target element."
    )
    target_role: str | None = Field(
        default=None,
        description="ARIA role and/or accessible label of the target element, e.g. 'button \"Submit\"'.",
    )
    value: str | None = Field(
        default=None,
        description="Value to type/select, or the URL for a navigate action.",
    )
    extracted_data: dict[str, str] | None = Field(
        default=None,
        description="Data read from the page, for extract_data and goal_complete actions.",
    )
    reasoning: str = Field(..., min_length=1, description="Why this action was chosen.")
    is_goal_complete: bool = Field(
        default=False, description="True once the stated goal has been fully achieved."
    )
    confidence: Confidence = Field(..., description="How confident Claude is in this decision.")
    stuck_reason: str | None = Field(
        default=None, description="Why the agent is stuck, for action_type='stuck'."
    )

    @model_validator(mode="after")
    def _validate_action_requirements(self) -> "AgentAction":
        # target_description is required for every action_type (see its Field(...)
        # above), so click/type/select are already guaranteed some identifying text;
        # target_text/target_role remain optional supplements to it.
        if self.action_type in {AgentActionType.TYPE, AgentActionType.SELECT} and not self.value:
            raise ValueError(f"action_type={self.action_type.value!r} requires 'value'.")
        if self.action_type is AgentActionType.NAVIGATE and not self.value:
            raise ValueError("action_type='navigate' requires 'value' (the URL to navigate to).")
        if self.action_type is AgentActionType.GOAL_COMPLETE and not self.is_goal_complete:
            raise ValueError("action_type='goal_complete' requires is_goal_complete=True.")
        if self.action_type is AgentActionType.STUCK and not self.stuck_reason:
            raise ValueError("action_type='stuck' requires 'stuck_reason'.")
        return self


async def _capture_page_state(page: Page) -> tuple[bytes, str, str, str]:
    """Capture a screenshot, an ARIA snapshot, URL, and title of ``page``."""
    screenshot = await page.screenshot(full_page=True, type="png")
    aria_snapshot = await page.aria_snapshot(mode="ai")
    url = page.url
    title = await page.title()
    return screenshot, aria_snapshot, url, title


def _build_system_prompt(goal: str, step_history: list[dict]) -> str:
    history_text = (
        json.dumps(step_history, indent=2, default=str) if step_history else "(no steps taken yet)"
    )
    return f"""You are an automation agent operating a legacy banking application through a browser.

This application has hostile markup: no test IDs, no semantic HTML, table-based layouts, and iframes. Do not rely on HTML ids or CSS classes — they are unstable or nonexistent. Identify elements the way a human would: by their visible text, their ARIA role/accessible name, or their position relative to nearby text (e.g. "the input to the right of the label 'Account Number'").

Your goal is: {goal}

You have completed these steps so far:
{history_text}

You are given a screenshot of the current page and its ARIA accessibility snapshot (a more reliable structural representation than raw HTML for this kind of app, including the content of any iframes). Elements in the snapshot carry a reference like [ref=e21] — you may cite it in target_description to disambiguate, but always also describe the element by its visible text/role since refs are not stable across page states. Decide the SINGLE next action to take toward the goal.

Available actions:
- click: click an element. Identify it via target_text, target_role, and/or target_description.
- type: type text into a field. Identify the field the same way, and set 'value' to the text to enter.
- select: choose an option in a dropdown. Identify the dropdown the same way, and set 'value' to the option to select.
- navigate: go to a URL. Set 'value' to the URL.
- wait: wait for the page to reach a expected state before proceeding (e.g. a slow legacy page load).
- extract_data: read specific data from the page without interacting with it. Describe what to read in 'target_description' and put the result in 'extracted_data'.
- goal_complete: declare the goal achieved. Set is_goal_complete=true and put any requested output data in 'extracted_data'.
- stuck: you are unsure how to proceed or the page is in an unexpected state. Explain why in 'stuck_reason'.

'target_description' is required on every action. For click/type/select it must do real identifying work, not restate the role: in this kind of app, form fields are frequently bare `textbox`/`combobox` roles with NO accessible name, because the visible label is just an adjacent table cell rather than a programmatically associated <label>, so several elements can share the same target_role. Name the nearby label text and/or position instead (e.g. "the textbox in the row labeled 'Username:'", "the second textbox on the form", "the input to the right of the 'Password:' cell") — put this identifying context in target_description itself, not only in 'reasoning', since target_description is what a replay engine acts on.

Always fill in 'reasoning' with why you chose this action, and 'confidence' with how confident you are in it.

Respond with your decision as the single next action."""


async def observe_and_decide(page: Page, goal: str, step_history: list[dict]) -> AgentAction:
    """Capture the current page state and ask Claude for the single next action.

    Retries once if Claude's response is malformed (invalid JSON or fails
    AgentAction validation). Genuine API errors (auth, rate limits, server
    errors) are not retried here — the SDK already retries transient ones,
    and the rest should surface with their real type rather than being
    masked as a malformed-response failure.
    """
    screenshot, aria_snapshot, url, title = await _capture_page_state(page)
    screenshot_b64 = base64.standard_b64encode(screenshot).decode("utf-8")
    aria_text = aria_snapshot.strip() or "(empty — no accessible nodes found)"

    system_prompt = _build_system_prompt(goal, step_history)
    user_content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64},
        },
        {
            "type": "text",
            "text": (
                f"Current URL: {url}\n"
                f"Page title: {title}\n\n"
                f"ARIA accessibility snapshot:\n{aria_text}"
            ),
        },
    ]

    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = await client.messages.parse(
                model=MODEL_ID,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": user_content}],
                output_format=AgentAction,
            )
            return response.parsed_output
        except (ValidationError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning(
                "observe_and_decide: malformed response on attempt %d/%d (%s)",
                attempt,
                MAX_ATTEMPTS,
                exc,
            )

    raise RuntimeError(
        f"observe_and_decide: Claude did not return a valid action after {MAX_ATTEMPTS} attempts"
    ) from last_error
