"""Bounded tool-loop guardrails for ordinary gateway chats.

These limits are deliberately opt-in at the ``AIAgent`` boundary. Gateway
standard chats resolve and pass them; CLI runs, cron runs, delegated agents,
and Kanban workers do not. This keeps long-running autonomous work separate
from a user-facing chat turn that should fail visibly instead of looping until
the global iteration budget is exhausted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


class GuardrailAction(str, Enum):
    CONTINUE = "continue"
    WARN = "warn"
    STOP = "stop"


@dataclass(frozen=True)
class StandardChatGuardrails:
    warn_tool_rounds: int = 8
    max_tool_rounds: int = 12
    warn_context_tokens: int = 60_000
    max_context_tokens: int = 80_000


@dataclass(frozen=True)
class StandardChatGuardrailDecision:
    action: GuardrailAction
    reasons: Tuple[str, ...] = ()
    tool_rounds: int = 0
    context_tokens: int = 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "kind": "standard_chat",
            "action": self.action.value,
            "reasons": list(self.reasons),
            "tool_rounds": self.tool_rounds,
            "context_tokens": self.context_tokens,
        }


def _positive_int(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def resolve_standard_chat_guardrails(
    user_config: Mapping[str, Any] | None,
    *,
    platform: str,
    is_worker: bool = False,
) -> Optional[StandardChatGuardrails]:
    """Resolve limits for a normal gateway chat, or ``None`` when exempt.

    ``platform`` is the normalized agent platform key (``cli`` for local).
    Workers are explicitly exempt even if they happen to use a gateway-like
    platform in a custom launcher.
    """
    normalized_platform = str(platform or "").strip().lower()
    if is_worker or normalized_platform in {"", "cli", "local"}:
        return None

    config = user_config if isinstance(user_config, Mapping) else {}
    agent_config = config.get("agent")
    if not isinstance(agent_config, Mapping):
        agent_config = {}
    raw = agent_config.get("standard_chat_guardrails")
    if raw is False:
        return None
    if not isinstance(raw, Mapping):
        raw = {}
    if raw.get("enabled") is False:
        return None

    defaults = StandardChatGuardrails()
    resolved = StandardChatGuardrails(
        warn_tool_rounds=_positive_int(raw.get("warn_tool_rounds"))
        or defaults.warn_tool_rounds,
        max_tool_rounds=_positive_int(raw.get("max_tool_rounds"))
        or defaults.max_tool_rounds,
        warn_context_tokens=_positive_int(raw.get("warn_context_tokens"))
        or defaults.warn_context_tokens,
        max_context_tokens=_positive_int(raw.get("max_context_tokens"))
        or defaults.max_context_tokens,
    )
    # A malformed order could remove the warning window or move the hard stop
    # beyond the operator's intended ceiling. Fail closed to known-safe defaults.
    if (
        resolved.warn_tool_rounds >= resolved.max_tool_rounds
        or resolved.warn_context_tokens >= resolved.max_context_tokens
    ):
        return defaults
    return resolved


def evaluate_standard_chat_guardrails(
    config: StandardChatGuardrails,
    *,
    tool_rounds: int,
    context_tokens: int,
) -> StandardChatGuardrailDecision:
    rounds = max(0, int(tool_rounds or 0))
    tokens = max(0, int(context_tokens or 0))

    stop_reasons = []
    if rounds >= config.max_tool_rounds:
        stop_reasons.append("tool_rounds")
    if tokens >= config.max_context_tokens:
        stop_reasons.append("context_tokens")
    if stop_reasons:
        return StandardChatGuardrailDecision(
            GuardrailAction.STOP,
            tuple(stop_reasons),
            rounds,
            tokens,
        )

    warning_reasons = []
    if rounds >= config.warn_tool_rounds:
        warning_reasons.append("tool_rounds")
    if tokens >= config.warn_context_tokens:
        warning_reasons.append("context_tokens")
    if warning_reasons:
        return StandardChatGuardrailDecision(
            GuardrailAction.WARN,
            tuple(warning_reasons),
            rounds,
            tokens,
        )

    return StandardChatGuardrailDecision(
        GuardrailAction.CONTINUE,
        (),
        rounds,
        tokens,
    )


def build_standard_chat_warning_message(
    decision: StandardChatGuardrailDecision,
    config: StandardChatGuardrails,
) -> str:
    details = []
    if "tool_rounds" in decision.reasons:
        details.append(f"{decision.tool_rounds}/{config.max_tool_rounds} tool rounds")
    if "context_tokens" in decision.reasons:
        details.append(
            f"about {decision.context_tokens:,}/{config.max_context_tokens:,} context tokens"
        )
    joined = " and ".join(details) or "the configured limits"
    return (
        f"⚠️ Standard chat is approaching its safety limit ({joined}). "
        "Hermes will stop after a completed tool batch rather than continue silently."
    )


def build_standard_chat_stop_message(
    *,
    reasons: Tuple[str, ...],
    tool_rounds: int,
    context_tokens: int,
) -> str:
    details = []
    if "tool_rounds" in reasons:
        details.append(f"{tool_rounds} tool rounds")
    if "context_tokens" in reasons:
        details.append(f"about {context_tokens:,} context tokens")
    joined = " and ".join(details) or "a configured safety threshold"
    return (
        f"⚠️ Safety stop: this standard chat reached {joined}. The current "
        "tool batch returned, but the requested task is not complete and must "
        "not be treated as a success. No further tool calls were started. "
        "Send “continue” to start a fresh bounded turn, or move long-running "
        "work to a worker."
    )


__all__ = [
    "GuardrailAction",
    "StandardChatGuardrails",
    "StandardChatGuardrailDecision",
    "build_standard_chat_stop_message",
    "build_standard_chat_warning_message",
    "evaluate_standard_chat_guardrails",
    "resolve_standard_chat_guardrails",
]
