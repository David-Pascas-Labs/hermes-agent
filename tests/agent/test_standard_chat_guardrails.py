"""Behavioral coverage for bounded gateway standard-chat tool loops."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent
from agent.standard_chat_guardrails import (
    GuardrailAction,
    StandardChatGuardrails,
    build_standard_chat_stop_message,
    evaluate_standard_chat_guardrails,
    resolve_standard_chat_guardrails,
)


def _tool_call(i: int):
    return SimpleNamespace(
        id=f"call_{i}",
        type="function",
        function=SimpleNamespace(
            name="web_search",
            arguments=json.dumps({"query": f"x{i}"}),
        ),
    )


def _tool_response(i: int, *, tool_count: int = 1):
    message = SimpleNamespace(
        content=None,
        reasoning_content=None,
        reasoning=None,
        tool_calls=[_tool_call((i * 10) + n) for n in range(tool_count)],
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        model="test/model",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5, total_tokens=105),
    )


def _stop_response(text: str = "done"):
    message = SimpleNamespace(
        content=text,
        reasoning_content=None,
        reasoning=None,
        tool_calls=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        model="test/model",
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=5, total_tokens=105),
    )


def _tool_defs():
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=_tool_defs()),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        instance = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            max_iterations=20,
            platform="telegram",
            standard_chat_guardrails=StandardChatGuardrails(
                warn_tool_rounds=1,
                max_tool_rounds=2,
                warn_context_tokens=60_000,
                max_context_tokens=80_000,
            ),
        )
    instance.client = MagicMock()
    instance._cached_system_prompt = "You are helpful."
    instance._use_prompt_caching = False
    instance._disable_streaming = True
    instance.save_trajectories = False
    instance.compression_enabled = False
    instance.context_compressor = MagicMock()
    instance.context_compressor.should_defer_preflight_to_real_usage.return_value = True
    instance.context_compressor._context_probe_persistable = False
    instance.context_compressor.last_prompt_tokens = 100
    return instance


def _run(agent, handled=None):
    statuses = []
    handled = handled if handled is not None else []
    agent.status_callback = lambda event_type, message: statuses.append((
        event_type,
        message,
    ))
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch(
            "run_agent.handle_function_call",
            lambda name, args, task_id=None, **kwargs: (
                handled.append(name) or json.dumps({"ok": True})
            ),
        ),
    ):
        result = agent.run_conversation("do tool work")
    return result, statuses


def test_default_gateway_threshold_boundaries_warn_then_stop():
    cfg = StandardChatGuardrails()

    assert (
        evaluate_standard_chat_guardrails(
            cfg, tool_rounds=7, context_tokens=59_999
        ).action
        is GuardrailAction.CONTINUE
    )
    warning = evaluate_standard_chat_guardrails(
        cfg, tool_rounds=8, context_tokens=60_000
    )
    assert warning.action is GuardrailAction.WARN
    assert set(warning.reasons) == {"tool_rounds", "context_tokens"}

    by_turns = evaluate_standard_chat_guardrails(cfg, tool_rounds=12, context_tokens=1)
    assert by_turns.action is GuardrailAction.STOP
    assert by_turns.reasons == ("tool_rounds",)

    by_context = evaluate_standard_chat_guardrails(
        cfg, tool_rounds=1, context_tokens=80_000
    )
    assert by_context.action is GuardrailAction.STOP
    assert by_context.reasons == ("context_tokens",)


def test_resolver_applies_only_to_standard_gateway_chats():
    config = {
        "agent": {
            "standard_chat_guardrails": {
                "warn_tool_rounds": 3,
                "max_tool_rounds": 5,
                "warn_context_tokens": 10_000,
                "max_context_tokens": 20_000,
            }
        }
    }

    resolved = resolve_standard_chat_guardrails(config, platform="telegram")
    assert resolved == StandardChatGuardrails(3, 5, 10_000, 20_000)
    assert resolve_standard_chat_guardrails(config, platform="cli") is None
    assert (
        resolve_standard_chat_guardrails(config, platform="telegram", is_worker=True)
        is None
    )


def test_invalid_threshold_order_falls_back_to_safe_defaults():
    resolved = resolve_standard_chat_guardrails(
        {
            "agent": {
                "standard_chat_guardrails": {
                    "warn_tool_rounds": 50,
                    "max_tool_rounds": 10,
                    "warn_context_tokens": 90_000,
                    "max_context_tokens": 80_000,
                }
            }
        },
        platform="telegram",
    )
    assert resolved == StandardChatGuardrails()


def test_stop_message_cannot_be_misread_as_success():
    text = build_standard_chat_stop_message(
        reasons=("tool_rounds", "context_tokens"),
        tool_rounds=12,
        context_tokens=80_500,
    )
    assert "not complete" in text.lower()
    assert "no further tool calls" in text.lower()
    assert "continue" in text.lower()


def test_loop_warns_once_then_hard_stops_without_extra_model_call(agent):
    agent.client.chat.completions.create.side_effect = [
        _tool_response(1),
        _tool_response(2),
        _stop_response("must not be reached"),
    ]

    result, statuses = _run(agent)

    assert agent.client.chat.completions.create.call_count == 2
    assert result["completed"] is False
    assert result["failed"] is True
    assert result["partial"] is True
    assert result["guardrail"]["kind"] == "standard_chat"
    assert result["guardrail"]["action"] == "stop"
    assert result["guardrail"]["tool_rounds"] == 2
    assert "not complete" in result["final_response"].lower()
    warning_statuses = [m for _kind, m in statuses if "approaching" in m.lower()]
    assert len(warning_statuses) == 1


def test_hard_stop_waits_for_every_call_in_the_current_tool_batch(agent):
    agent.standard_chat_guardrails = StandardChatGuardrails(
        warn_tool_rounds=1,
        max_tool_rounds=2,
        warn_context_tokens=60_000,
        max_context_tokens=80_000,
    )
    agent.client.chat.completions.create.side_effect = [
        _tool_response(1, tool_count=2),
        _tool_response(2, tool_count=2),
    ]
    handled = []

    result, _statuses = _run(agent, handled)

    assert agent.client.chat.completions.create.call_count == 2
    assert handled == ["web_search", "web_search", "web_search", "web_search"]
    assert result["failed"] is True
    assert result["partial"] is True
    assert result["guardrail"]["tool_rounds"] == 2
    assert "no further tool calls were started" in result["final_response"].lower()


def test_worker_agent_remains_unbounded_without_opt_in(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker")
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        worker = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            platform="cli",
        )
    assert worker.standard_chat_guardrails is None
