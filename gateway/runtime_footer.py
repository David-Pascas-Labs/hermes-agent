"""Display-only runtime-metadata footer for terminal and gateway surfaces.

Renders a compact footer showing runtime state and optional numeric counters
already present in ``state.db`` or the completed execution result. It is built
only after the model turn and is off by default, so it never adds a provider
call, tokens, or prompt/history/cache content.

Config (``~/.hermes/config.yaml``)::

    display:
      runtime_footer:
        enabled: true                       # off by default
        fields: [api_calls, input_tokens, output_tokens, cache_read_tokens, tool_calls]

Per-platform overrides live under ``display.platforms.<platform>.runtime_footer``.
Users can toggle the global setting with ``/footer on|off`` from both the CLI
and any gateway platform.

The gateway appends the footer only at the final transport boundary, after the
canonical response has already passed hooks and persistence. If streaming has
already delivered the response, the footer is suppressed rather than creating
a second Telegram/Discord delivery. The interactive CLI prints it as separate
terminal chrome after the response box.
"""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_DEFAULT_FIELDS: tuple[str, ...] = (
    "api_calls",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "tool_calls",
)
_SEP = " · "
_EXECUTION_FOOTER_KEYS: tuple[str, ...] = (
    "session_id",
    "api_calls",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "tool_calls",
)


@dataclass(frozen=True)
class RuntimeCounters:
    """Non-sensitive numeric counters already recorded for one session."""

    api_calls: int | None
    input_tokens: int | None
    output_tokens: int | None
    cache_read_tokens: int | None
    cache_write_tokens: int | None
    reasoning_tokens: int | None
    tool_calls: int | None


def project_execution_footer_state(execution: dict[str, Any] | None) -> dict[str, Any]:
    """Retain only fields needed by display chrome.

    In particular, response text, messages, tool arguments/results, and unknown
    metadata are dropped before the CLI keeps the completed turn for rendering.
    ``session_id`` is retained only as the parameterized local DB lookup key and
    is never formatted.
    """
    if not isinstance(execution, dict):
        return {}
    return {key: execution[key] for key in _EXECUTION_FOOTER_KEYS if key in execution}


def read_local_session_counters(
    db_path: str | os.PathLike[str] | None,
    session_id: str | None,
) -> RuntimeCounters | None:
    """Read one session's counters from an existing ``state.db``.

    The connection is URI ``mode=ro`` plus ``PRAGMA query_only``. Missing,
    locked, or older-schema databases fail soft because display chrome must
    never alter (or prevent) the agent response.
    """
    if not db_path or not session_id:
        return None
    path = Path(db_path)
    if not path.is_file():
        return None
    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True, timeout=0.05)) as conn:
            conn.execute("PRAGMA query_only = ON")
            row = conn.execute(
                """SELECT api_call_count, input_tokens, output_tokens,
                          cache_read_tokens, cache_write_tokens,
                          reasoning_tokens, tool_call_count
                     FROM sessions
                    WHERE id = ?""",
                (session_id,),
            ).fetchone()
    except (OSError, sqlite3.Error):
        return None
    if row is None:
        return None
    try:
        values = [max(0, int(value or 0)) for value in row]
    except (TypeError, ValueError, OverflowError):
        return None
    return RuntimeCounters(*values)


def _counters_from_execution(execution: dict[str, Any] | None) -> RuntimeCounters | None:
    """Project an existing agent result onto the numeric counter allow-list."""
    if not isinstance(execution, dict):
        return None
    keys = (
        "api_calls",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "tool_calls",
    )
    if not any(key in execution for key in keys):
        return None
    values: list[int | None] = []
    for key in keys:
        if key not in execution:
            values.append(None)
            continue
        try:
            values.append(max(0, int(execution.get(key, 0) or 0)))
        except (TypeError, ValueError, OverflowError):
            values.append(None)
    return RuntimeCounters(*values)


def _compact_count(value: int) -> str:
    """Format a non-negative counter without losing small exact values."""
    value = max(0, int(value))
    if value < 1000:
        return str(value)
    if value < 1_000_000:
        return f"{value / 1000:.1f}k"
    return f"{value / 1_000_000:.1f}M"


def resolve_footer_config(
    user_config: dict[str, Any] | None,
    platform_key: str | None = None,
) -> dict[str, Any]:
    """Resolve effective runtime-footer config for *platform_key*.

    Merge order (later wins):
        1. Built-in defaults (enabled=False)
        2. ``display.runtime_footer``
        3. ``display.platforms.<platform_key>.runtime_footer``
    """
    resolved = {"enabled": False, "fields": list(_DEFAULT_FIELDS)}
    cfg = (user_config or {}).get("display") or {}

    global_cfg = cfg.get("runtime_footer")
    if isinstance(global_cfg, dict):
        if "enabled" in global_cfg:
            resolved["enabled"] = bool(global_cfg.get("enabled"))
        if isinstance(global_cfg.get("fields"), list) and global_cfg["fields"]:
            resolved["fields"] = [str(f) for f in global_cfg["fields"]]

    if platform_key:
        platforms = cfg.get("platforms") or {}
        plat_cfg = platforms.get(platform_key)
        if isinstance(plat_cfg, dict):
            plat_footer = plat_cfg.get("runtime_footer")
            if isinstance(plat_footer, dict):
                if "enabled" in plat_footer:
                    resolved["enabled"] = bool(plat_footer.get("enabled"))
                if isinstance(plat_footer.get("fields"), list) and plat_footer["fields"]:
                    resolved["fields"] = [str(f) for f in plat_footer["fields"]]

    return resolved


def format_runtime_footer(
    *,
    fields: Iterable[str] = _DEFAULT_FIELDS,
    counters: RuntimeCounters | None = None,
) -> str:
    """Render the footer line, or return "" if no fields have data.

    Fields are skipped silently when their underlying data is missing — a
    partially-populated footer is better than a line with ``?%`` or empty slots.
    """
    parts: list[str] = []
    for field in fields:
        if counters is not None:
            counter_fields = {
                "api_calls": ("api", counters.api_calls),
                "input_tokens": ("in", counters.input_tokens),
                "output_tokens": ("out", counters.output_tokens),
                "cache_read_tokens": ("cache", counters.cache_read_tokens),
                "cache_write_tokens": ("cache+", counters.cache_write_tokens),
                "reasoning_tokens": ("reason", counters.reasoning_tokens),
                "tool_calls": ("tools", counters.tool_calls),
            }
            counter = counter_fields.get(field)
            if counter is not None:
                label, value = counter
                if value is not None:
                    parts.append(f"{label} {_compact_count(value)}")
        # Unknown field names are silently ignored.

    if not parts:
        return ""
    return _SEP.join(parts)


def build_footer_line(
    *,
    user_config: dict[str, Any] | None,
    platform_key: str | None,
    db_path: str | os.PathLike[str] | None = None,
    session_id: str | None = None,
    execution: dict[str, Any] | None = None,
) -> str:
    """Top-level entry point used by gateway/run.py.

    Returns the footer text (empty string when disabled or no data).  Callers
    append this to the final response themselves, preserving a single blank
    line of separation.
    """
    cfg = resolve_footer_config(user_config, platform_key)
    if not cfg.get("enabled"):
        return ""
    counters = read_local_session_counters(db_path, session_id)
    if counters is None:
        counters = _counters_from_execution(execution)
    return format_runtime_footer(
        fields=cfg.get("fields") or _DEFAULT_FIELDS,
        counters=counters,
    )


def append_footer_for_delivery(
    response: str,
    footer: str,
    *,
    already_sent: bool,
    intentional_silence: bool,
) -> str:
    """Append display chrome only when it can ride the existing final send.

    Streaming surfaces have already delivered their body; returning it
    unchanged deliberately suppresses the footer instead of creating a second
    Telegram/Discord delivery. The caller's canonical response and history are
    never accepted or mutated here.
    """
    if already_sent or intentional_silence or not response or not footer:
        return response
    return f"{response}\n\n{footer}"
