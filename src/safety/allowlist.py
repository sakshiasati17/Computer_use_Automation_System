"""Domain/URL/action allowlist enforcement.

This system drives a browser against regulated financial data, so every
`navigate` target and every click/type/select action is checked against an
explicit allowlist (`config/allowlist.json`) before it is allowed to run,
rather than trusting whatever the LLM (discovery) or a recorded artifact
(replay) decided to do. Anything blocked is logged with full details -
both through the standard logger and as a persistent JSON-lines audit trail
under `evidence/safety/violations.jsonl` - since a silently-dropped action is
as dangerous as one that ran without a check.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

__all__ = ["check_url", "check_action", "load_config"]

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "allowlist.json"
VIOLATIONS_LOG_PATH = REPO_ROOT / "evidence" / "safety" / "violations.jsonl"

_config_cache: dict[str, Any] | None = None
_config_cache_path: Path | None = None


def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load (and cache) the allowlist config, re-reading if `config_path` changes."""
    global _config_cache, _config_cache_path
    path = Path(config_path)
    if _config_cache is not None and _config_cache_path == path:
        return _config_cache
    with path.open() as f:
        config = json.load(f)
    _config_cache = config
    _config_cache_path = path
    return config


def _log_violation(kind: str, details: dict[str, Any]) -> None:
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "violation_type": kind, **details}
    logger.warning("SAFETY VIOLATION [%s]: %s", kind, record)
    try:
        VIOLATIONS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with VIOLATIONS_LOG_PATH.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except OSError:
        logger.exception("failed to persist safety violation record to %s", VIOLATIONS_LOG_PATH)


def _domain_permitted(netloc: str, permitted_domains: list[str]) -> bool:
    host = netloc.split(":")[0].lower()
    return any(host == domain.lower() or host.endswith(f".{domain.lower()}") for domain in permitted_domains)


def _port_permitted(netloc: str, permitted_ports: list[int]) -> bool:
    if not permitted_ports or ":" not in netloc:
        # No port list configured, or the URL carries no explicit port (i.e.
        # the scheme's default): nothing to gate on.
        return True
    try:
        port = int(netloc.rsplit(":", 1)[1])
    except ValueError:
        return True
    return port in permitted_ports


def check_url(url: str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> bool:
    """Return True if `url`'s domain, port, and path are all permitted.

    Checked against `permitted_domains`/`permitted_ports` (allowlist) and
    `blocked_url_patterns` (denylist on path). Every rejection is logged via
    `_log_violation` before returning False.
    """
    config = load_config(config_path)
    parsed = urlparse(url)
    netloc = parsed.netloc

    permitted_domains = config.get("permitted_domains", [])
    permitted_ports = config.get("permitted_ports", [])
    blocked_url_patterns = config.get("blocked_url_patterns", [])

    if not _domain_permitted(netloc, permitted_domains):
        _log_violation(
            "domain_not_permitted",
            {"url": url, "netloc": netloc, "permitted_domains": permitted_domains},
        )
        return False

    if not _port_permitted(netloc, permitted_ports):
        _log_violation(
            "port_not_permitted",
            {"url": url, "netloc": netloc, "permitted_ports": permitted_ports},
        )
        return False

    path = parsed.path or ""
    for pattern in blocked_url_patterns:
        if pattern in path:
            _log_violation(
                "blocked_url_pattern",
                {"url": url, "path": path, "pattern": pattern},
            )
            return False

    return True


def check_action(action_type: str, action_details: str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> bool:
    """Return True unless `action_type`/`action_details` contains a blocked action keyword.

    `action_details` should carry whatever identifying text is available for
    the action - target text/description, the value being typed/selected -
    so keywords embedded there (e.g. a button labeled "Wire Transfer") are
    caught even when `action_type` itself is generic (e.g. "click").
    """
    config = load_config(config_path)
    blocked_keywords = config.get("blocked_action_keywords", [])
    haystack = f"{action_type} {action_details}".lower()

    for keyword in blocked_keywords:
        # Keywords are configured snake_case (e.g. "wire_transfer"), but real
        # UI text is usually space-separated ("Wire Transfer"): check both so
        # the keyword still matches human-readable button/label text.
        keyword_lower = keyword.lower()
        if keyword_lower in haystack or keyword_lower.replace("_", " ") in haystack:
            _log_violation(
                "blocked_action_keyword",
                {"action_type": action_type, "action_details": action_details, "keyword": keyword},
            )
            return False

    return True
