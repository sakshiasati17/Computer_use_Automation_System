"""Action risk classification: safe / risky / irreversible.

Both the discovery agent (`src/agent/discovery.py`, deciding actions from an
LLM one step at a time) and the replay engine (`src/replay/engine.py`,
executing a recorded `Step`) need to know how consequential an action is
before it runs. `classify_action` accepts either shape - a raw dict, an
`AgentAction`, or anything else exposing the same fields by attribute - so
both call sites can share one rule set (`config/risk_rules.json`) instead of
duplicating keyword lists.

Classification is keyword/URL-pattern based, not exhaustive NLP: irreversible
patterns (delete/transfer/close/...) always win over risky patterns, since
an action that looks both "risky" and "irreversible" (e.g. a "Confirm Wire
Transfer" button) must be treated as the more dangerous of the two. Anything
that matches neither list defaults to "safe" - matching the examples in the
schema this classifies for (`RiskLevel.SAFE` in `src/models/artifact.py`),
where safe is the default for ordinary navigation/read actions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

__all__ = ["classify_action", "load_rules"]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = REPO_ROOT / "config" / "risk_rules.json"

_rules_cache: dict[str, Any] | None = None
_rules_cache_path: Path | None = None


def load_rules(rules_path: str | Path = DEFAULT_RULES_PATH) -> dict[str, Any]:
    """Load (and cache) the risk rules config, re-reading if `rules_path` changes."""
    global _rules_cache, _rules_cache_path
    path = Path(rules_path)
    if _rules_cache is not None and _rules_cache_path == path:
        return _rules_cache
    with path.open() as f:
        rules = json.load(f)
    _rules_cache = rules
    _rules_cache_path = path
    return rules


def _field(step: Any, *names: str) -> str | None:
    """Duck-typed field lookup: works for a plain dict or an attribute-bearing object.

    Returns the first non-empty match among `names`, stringified (unwrapping
    an Enum's `.value` if that's what was stored).
    """
    for name in names:
        value = step.get(name) if isinstance(step, dict) else getattr(step, name, None)
        if value:
            return str(getattr(value, "value", value))
    return None


def classify_action(
    step: Any, url: str | None = None, rules_path: str | Path = DEFAULT_RULES_PATH
) -> str:
    """Classify one action as "safe", "risky", or "irreversible".

    `step` may be a raw discovery step dict, an `AgentAction`, a replay
    `Step`, or any object/dict exposing `action_type`/`action`,
    `target_text`, `target_description`, `description`, and/or `value`.
    `url` is the current or target page URL (falls back to a `url`/
    `page_url`/`current_url` field on `step` if omitted) and is checked
    against `risky_url_patterns`.
    """
    rules = load_rules(rules_path)
    risky_patterns = [p.lower() for p in rules.get("risky_patterns", [])]
    irreversible_patterns = [p.lower() for p in rules.get("irreversible_patterns", [])]
    risky_url_patterns = rules.get("risky_url_patterns", [])

    action_type = _field(step, "action_type", "action") or ""
    haystack = " ".join(
        [
            action_type,
            _field(step, "target_text") or "",
            _field(step, "target_description") or "",
            _field(step, "description") or "",
            _field(step, "value") or "",
        ]
    ).lower()

    effective_url = url or _field(step, "url", "page_url", "current_url") or ""
    path = urlparse(effective_url).path if effective_url else ""

    if any(pattern in haystack for pattern in irreversible_patterns):
        return "irreversible"

    if any(pattern in path for pattern in risky_url_patterns):
        return "risky"

    if any(pattern in haystack for pattern in risky_patterns):
        return "risky"

    return "safe"
