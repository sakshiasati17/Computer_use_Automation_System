"""Safety gates for a system that drives a browser over regulated financial data.

- `allowlist`: domain/URL/action allowlisting, consulted before every
  navigate/click/type/select in both discovery and replay.
- `classifier`: risk classification (safe/risky/irreversible) for a single
  action, shared by discovery and replay via one rule set.
- `redactor`: sensitive-data redaction, run before any artifact or log
  reaches disk.
"""

from __future__ import annotations

from .allowlist import check_action, check_url
from .classifier import classify_action
from .redactor import redact, redact_dict, redact_file

__all__ = ["check_action", "check_url", "classify_action", "redact", "redact_dict", "redact_file"]
