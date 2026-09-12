"""Sensitive-data redaction.

Discovery logs, replay evidence, and saved artifacts all pass through
whatever the target application put on the page - account numbers, SSNs,
card numbers, credentials typed during a recorded run. Since this system
handles regulated financial data, none of that may reach disk unredacted:
`redact`/`redact_dict`/`redact_file` must run on every artifact and every
log/evidence record before it is written (see `src/agent/discovery.py`,
`src/artifacts/emitter.py`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

__all__ = ["redact", "redact_dict", "redact_file"]

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_ACCOUNT_RE = re.compile(r"\bACT-\d{4,10}\b", re.IGNORECASE)
# Grouped card numbers (e.g. "4111-1111-1111-1111" / "4111 1111 1111 1111")
# are matched before the plain digit-run pattern below, so the separators
# don't stop a real card number from being recognized as one 13-19 digit
# sequence.
_CC_GROUPED_RE = re.compile(r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{1,7}\b")
_CC_PLAIN_RE = re.compile(r"\b\d{13,19}\b")
_PASSWORD_JSON_RE = re.compile(r'("password"\s*:\s*")([^"]*)(")', re.IGNORECASE)
_PASSWORD_TEXT_RE = re.compile(r"\b(password\s*(?:is|[:=])\s*)([^\s'\",]+)", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9.-]+\b")

_PASSWORD_KEY_NAMES = {"password", "pwd", "passwd"}
_LABEL_FIELDS = ("target_description", "target_text", "description", "reasoning", "anchor_text", "label")
_VALUE_FIELDS = ("value", "expected_value")
_TEMPLATE_REF_RE = re.compile(r"\{\{\s*input\.")


def redact(text: str) -> str:
    """Replace sensitive patterns in `text` with `[REDACTED-*]` placeholders."""
    if not text:
        return text

    text = _SSN_RE.sub("[REDACTED-SSN]", text)
    text = _ACCOUNT_RE.sub("[REDACTED-ACCT]", text)
    text = _PASSWORD_JSON_RE.sub(lambda m: f"{m.group(1)}[REDACTED-PWD]{m.group(3)}", text)
    text = _PASSWORD_TEXT_RE.sub(lambda m: f"{m.group(1)}[REDACTED-PWD]", text)
    text = _CC_GROUPED_RE.sub("[REDACTED-CC]", text)
    text = _CC_PLAIN_RE.sub("[REDACTED-CC]", text)
    text = _EMAIL_RE.sub("[REDACTED-EMAIL]", text)
    return text


def _redact_value(value: Any, key: str | None = None) -> Any:
    if isinstance(value, str):
        if key is not None and key.lower() in _PASSWORD_KEY_NAMES:
            return "[REDACTED-PWD]"
        return redact(value)
    if isinstance(value, dict):
        return redact_dict(value)
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value


def redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact all string values in `data` (nested dicts/lists included).

    Beyond pattern matching within a single string (`redact`), this also
    catches the common log/step shape where a password's *label* and its
    *value* live in separate fields (e.g. `target_description: "...labeled
    'Password:'"`, `value: "hunter2"`): if any label-like field in a dict
    mentions "password", sibling value fields are redacted outright even
    though the value itself carries no recognizable pattern.

    Exempt from that sibling-context override: a value that is (or embeds) a
    `{{input.x}}` template reference. Those appear verbatim in saved
    Artifacts (`src/models/actions.py`'s `TEMPLATE_REF_PATTERN`) and in
    `ElementHasValueCondition.expected_value`, never as the literal secret,
    so overwriting one with a placeholder doesn't protect anything - it
    corrupts the artifact, e.g. turning a password step's
    `value: "{{input.password}}"` into a literal string that gets typed into
    the field at replay time instead of the real templated input.
    """
    password_context = any(
        isinstance(data.get(field), str) and "password" in data[field].lower() for field in _LABEL_FIELDS
    )

    result: dict[str, Any] = {}
    for key, value in data.items():
        if (
            password_context
            and key in _VALUE_FIELDS
            and isinstance(value, str)
            and not _TEMPLATE_REF_RE.search(value)
        ):
            result[key] = "[REDACTED-PWD]"
        else:
            result[key] = _redact_value(value, key)
    return result


def redact_file(filepath: str) -> None:
    """Read a JSON file, redact its contents, and write it back in place."""
    path = Path(filepath)
    data = json.loads(path.read_text())

    if isinstance(data, dict):
        redacted: Any = redact_dict(data)
    elif isinstance(data, list):
        redacted = [_redact_value(item) for item in data]
    else:
        redacted = _redact_value(data)

    path.write_text(json.dumps(redacted, indent=2, default=str))
