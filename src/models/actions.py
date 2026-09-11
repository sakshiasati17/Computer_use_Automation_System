"""Action, locator, and page-condition primitives for UI automation artifacts.

Two vocabularies live here, and both are shared across the artifact schema
(``artifact.py``) rather than duplicated per use site:

- **Locators** (``TextLocator``, ``XPathLocator``, ...) answer "how do I find
  an element on the page?" A :class:`ElementTarget` bundles several of them
  into an ordered fallback chain, because legacy, table-based markup rarely
  offers one reliable selector.
- **Conditions** (``TextPresentCondition``, ``UrlCondition``, ...) answer "is
  the page in the state I expect?" The same condition vocabulary is reused
  for step checkpoints, step pre-conditions, business-outcome detection,
  error-handler detection, and the artifact's overall success condition —
  one vocabulary, four call sites, instead of four near-duplicate schemas.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "Confidence",
    "TextLocator",
    "AriaRoleLocator",
    "XPathLocator",
    "CssLocator",
    "TextNearLocator",
    "CoordinatesLocator",
    "Locator",
    "ElementTarget",
    "ActionType",
    "UrlMatchType",
    "ElementVisibleCondition",
    "TextPresentCondition",
    "UrlCondition",
    "ElementHasValueCondition",
    "Condition",
]


class Confidence(str, Enum):
    """How likely a locator is to keep working as the app drifts over time.

    Set deliberately by whoever records the artifact (an LLM during
    discovery, or a human reviewing it) — not inferred automatically — since
    it encodes *why* a locator was chosen, not just what it matches.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class _LocatorBase(BaseModel):
    """Common fields every locator strategy carries."""

    model_config = ConfigDict(extra="forbid")

    confidence: Confidence = Field(
        ...,
        description="How stable this locator is expected to be as the app changes.",
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description=(
            "Why this locator was chosen over the alternatives, e.g. "
            "'the label text is stable across tenants but the DOM id is "
            "auto-generated per session'."
        ),
    )


class TextLocator(_LocatorBase):
    """Find an element by its visible text content."""

    strategy: Literal["text"] = "text"
    text: str = Field(..., min_length=1, description="Visible text to match.")
    exact: bool = Field(
        default=False,
        description="If false, matches text as a substring rather than requiring equality.",
    )


class AriaRoleLocator(_LocatorBase):
    """Find an element by its ARIA role and accessible name."""

    strategy: Literal["aria_role"] = "aria_role"
    role: str = Field(..., min_length=1, description="ARIA role, e.g. 'button', 'textbox'.")
    name: str | None = Field(
        default=None, description="Accessible name to disambiguate elements sharing a role."
    )


class XPathLocator(_LocatorBase):
    """Find an element via an XPath expression."""

    strategy: Literal["xpath"] = "xpath"
    expression: str = Field(..., min_length=1, description="XPath expression.")


class CssLocator(_LocatorBase):
    """Find an element via a CSS selector.

    Least reliable of the structural locators against legacy, table-based
    markup: these apps rarely carry stable classes or ids, so a selector
    often encodes incidental layout structure that changes across tenants
    or app versions.
    """

    strategy: Literal["css"] = "css"
    selector: str = Field(..., min_length=1, description="CSS selector.")


class TextNearLocator(_LocatorBase):
    """Find an element positioned near a text anchor (e.g. a label).

    Useful for legacy table layouts where an input has no name/label
    association in the DOM, only visual proximity to a text cell such as
    "Member ID:".
    """

    strategy: Literal["text_near"] = "text_near"
    anchor_text: str = Field(..., min_length=1, description="The nearby text to anchor on.")
    relation: Literal["right_of", "below", "same_row", "same_container"] = Field(
        default="same_row",
        description="Spatial/structural relationship of the target element to the anchor text.",
    )
    element_type: str | None = Field(
        default=None,
        description="Expected tag/role of the target element, e.g. 'input', 'select'.",
    )


class CoordinatesLocator(_LocatorBase):
    """Find an element by fixed pixel coordinates.

    Last resort: breaks under any layout reflow, viewport resize, or
    responsive change. Confidence is pinned to LOW regardless of what is
    passed in, since a coordinate locator is never anything but fragile.
    """

    strategy: Literal["coordinates"] = "coordinates"
    x: int = Field(..., ge=0, description="X pixel coordinate.")
    y: int = Field(..., ge=0, description="Y pixel coordinate.")
    viewport_width: int | None = Field(
        default=None, description="Viewport width the coordinates were captured at."
    )
    viewport_height: int | None = Field(
        default=None, description="Viewport height the coordinates were captured at."
    )

    @model_validator(mode="after")
    def _force_low_confidence(self) -> "CoordinatesLocator":
        if self.confidence is not Confidence.LOW:
            self.confidence = Confidence.LOW
        return self


Locator = Annotated[
    Union[
        TextLocator,
        AriaRoleLocator,
        XPathLocator,
        CssLocator,
        TextNearLocator,
        CoordinatesLocator,
    ],
    Field(discriminator="strategy"),
]
"""Discriminated union of all locator strategies, tagged by `strategy`."""


class ElementTarget(BaseModel):
    """An element to interact with, described as an ordered fallback chain.

    ``locators`` is tried in order at replay time: the first strategy that
    resolves to exactly one element wins. Order locators from most to least
    reliable (e.g. `aria_role` before `css` before `coordinates`) since that
    order IS the fallback policy — there is no separate priority field.
    """

    model_config = ConfigDict(extra="forbid")

    locators: list[Locator] = Field(
        ...,
        min_length=1,
        description="Fallback chain of locator strategies, tried in list order.",
    )

    @property
    def primary(self) -> Locator:
        """The first (preferred) locator in the fallback chain."""
        return self.locators[0]


class ActionType(str, Enum):
    """The kind of interaction a step performs."""

    NAVIGATE = "navigate"
    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    EXTRACT = "extract"
    WAIT = "wait"


class UrlMatchType(str, Enum):
    """How a URL condition's pattern is compared against the current URL."""

    EXACT = "exact"
    CONTAINS = "contains"
    REGEX = "regex"


class _ConditionBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ElementVisibleCondition(_ConditionBase):
    """True when an element matching the given text and/or role is visible."""

    type: Literal["element_visible"] = "element_visible"
    text: str | None = Field(default=None, description="Visible text the element must contain.")
    role: str | None = Field(default=None, description="ARIA role the element must have.")

    @model_validator(mode="after")
    def _require_text_or_role(self) -> "ElementVisibleCondition":
        if not self.text and not self.role:
            raise ValueError("element_visible condition needs 'text' and/or 'role'.")
        return self


class TextPresentCondition(_ConditionBase):
    """True when the given text appears anywhere on the page."""

    type: Literal["text_present"] = "text_present"
    text: str = Field(..., min_length=1, description="Text expected to be present on the page.")
    case_sensitive: bool = Field(default=False)


class UrlCondition(_ConditionBase):
    """True when the current URL matches a pattern.

    Covers both the "url_matches" checkpoint case and the "url_contains"
    business-outcome-detection case from a single schema: pick
    ``match_type="contains"`` for the latter.
    """

    type: Literal["url_matches"] = "url_matches"
    pattern: str = Field(..., min_length=1, description="Pattern to compare the URL against.")
    match_type: UrlMatchType = Field(default=UrlMatchType.CONTAINS)


class ElementHasValueCondition(_ConditionBase):
    """True when a form field resolved by `target` holds `expected_value`."""

    type: Literal["element_has_value"] = "element_has_value"
    target: ElementTarget
    expected_value: str = Field(..., description="Value the field is expected to hold.")


Condition = Annotated[
    Union[
        ElementVisibleCondition,
        TextPresentCondition,
        UrlCondition,
        ElementHasValueCondition,
    ],
    Field(discriminator="type"),
]
"""Discriminated union of all page-state conditions, tagged by `type`.

Reused for step checkpoints, step pre-conditions, business-outcome
detection, error-handler detection, and the artifact success condition.
"""


TEMPLATE_REF_PATTERN = re.compile(r"\{\{\s*input\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
"""Matches `{{input.name}}` template references used in step `value` fields."""
