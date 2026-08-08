"""Regression tests for context-local Kanban tool availability.

A dispatcher worker can construct cron and delegated agents in-process. Their
ContextVars change Kanban ownership while the process environment stays fixed,
so process-global availability caches must not leak the first warmup verdict.
"""

from __future__ import annotations

from contextlib import AbstractContextManager

import pytest


KANBAN_LIFECYCLE_TOOLS = {
    "kanban_show",
    "kanban_complete",
    "kanban_block",
    "kanban_heartbeat",
    "kanban_comment",
    "kanban_attach",
    "kanban_attach_url",
    "kanban_attachments",
    "kanban_create",
    "kanban_link",
}


@pytest.fixture(autouse=True)
def _isolate_tool_caches(monkeypatch):
    import model_tools
    from tools import kanban_tools
    from tools.registry import invalidate_check_fn_cache

    monkeypatch.setattr(kanban_tools, "_profile_has_kanban_toolset", lambda: False)
    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()
    yield
    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()


@pytest.fixture
def worker_env(monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_cache_isolation")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "test-lock")


def _child_context(kind: str) -> AbstractContextManager[None]:
    from agent.delegation_context import (
        delegated_child_context,
        non_dispatcher_owned_context,
    )

    if kind == "cron":
        return non_dispatcher_owned_context()
    return delegated_child_context("delegate-cache-isolation")


def _registry_kanban_names() -> set[str]:
    """Resolve the real Kanban toolset directly through registry.get_definitions."""
    import model_tools  # noqa: F401 -- imports and registers the built-in tools
    from tools.registry import registry
    from toolsets import resolve_toolset

    definitions = registry.get_definitions(set(resolve_toolset("kanban")), quiet=True)
    return {definition["function"]["name"] for definition in definitions}


def _agent_kanban_names(*, platform: str) -> set[str]:
    """Resolve the production AIAgent -> model_tools -> registry assembly path."""
    from run_agent import AIAgent

    agent = AIAgent(
        base_url="https://stub.invalid/v1",
        api_key="stub",
        provider="openai",
        model="gpt-4o-mini",
        enabled_toolsets=["kanban"],
        quiet_mode=True,
        platform=platform,
        skip_context_files=True,
        skip_memory=True,
    )
    try:
        return set(getattr(agent, "valid_tool_names"))
    finally:
        agent.close()


def test_stable_check_fn_remains_cached():
    import tools.registry as reg

    state = {"available": True, "calls": 0}

    def stable_check():
        state["calls"] += 1
        return state["available"]

    assert reg._check_fn_cached(stable_check) is True
    state["available"] = False
    assert reg._check_fn_cached(stable_check) is True
    assert state["calls"] == 1


def test_context_dependent_check_bypasses_ttl_and_last_good():
    import tools.registry as reg

    state = {"available": True, "calls": 0}

    def context_dependent_check():
        state["calls"] += 1
        return state["available"]

    setattr(context_dependent_check, "_hermes_context_dependent", True)

    assert reg._check_fn_cached(context_dependent_check) is True
    state["available"] = False
    assert reg._check_fn_cached(context_dependent_check) is False
    assert state["calls"] == 2


@pytest.mark.parametrize("child_kind", ["cron", "delegate"])
def test_registry_parent_warmup_does_not_expose_lifecycle_tools_to_child(
    worker_env, child_kind
):
    parent_names = _registry_kanban_names()
    assert KANBAN_LIFECYCLE_TOOLS <= parent_names

    with _child_context(child_kind):
        child_names = _registry_kanban_names()

    assert KANBAN_LIFECYCLE_TOOLS.isdisjoint(child_names)


@pytest.mark.parametrize("child_kind", ["cron", "delegate"])
def test_registry_child_warmup_does_not_hide_lifecycle_tools_from_parent(
    worker_env, child_kind
):
    with _child_context(child_kind):
        child_names = _registry_kanban_names()
    assert KANBAN_LIFECYCLE_TOOLS.isdisjoint(child_names)

    parent_names = _registry_kanban_names()
    assert KANBAN_LIFECYCLE_TOOLS <= parent_names


@pytest.mark.parametrize("child_kind", ["cron", "delegate"])
def test_agent_parent_warmup_does_not_expose_lifecycle_tools_to_child(
    worker_env, child_kind
):
    parent_names = _agent_kanban_names(platform="cli")
    assert KANBAN_LIFECYCLE_TOOLS <= parent_names

    with _child_context(child_kind):
        child_names = _agent_kanban_names(platform=child_kind)

    assert KANBAN_LIFECYCLE_TOOLS.isdisjoint(child_names)


@pytest.mark.parametrize("child_kind", ["cron", "delegate"])
def test_agent_child_warmup_does_not_hide_lifecycle_tools_from_parent(
    worker_env, child_kind
):
    with _child_context(child_kind):
        child_names = _agent_kanban_names(platform=child_kind)
    assert KANBAN_LIFECYCLE_TOOLS.isdisjoint(child_names)

    parent_names = _agent_kanban_names(platform="cli")
    assert KANBAN_LIFECYCLE_TOOLS <= parent_names


@pytest.mark.parametrize("child_kind", ["cron", "delegate"])
def test_child_does_not_run_parent_heartbeat(worker_env, child_kind, monkeypatch):
    from tools import kanban_tools

    monkeypatch.setattr(kanban_tools, "_auto_heartbeat_last_attempt", 0.0)
    calls = {"connect": 0}

    def unexpected_connect(*_args, **_kwargs):
        calls["connect"] += 1
        raise RuntimeError("must not be reached")

    monkeypatch.setattr(kanban_tools, "_connect", unexpected_connect)

    with _child_context(child_kind):
        assert kanban_tools.heartbeat_current_worker_from_env() is False
    assert calls["connect"] == 0


@pytest.mark.parametrize("child_kind", ["cron", "delegate"])
def test_child_does_not_poll_parent_operator_comments(
    worker_env, child_kind, monkeypatch
):
    from tools import kanban_tools

    monkeypatch.setattr(kanban_tools, "_comment_poll_last_attempt", 0.0)
    calls = {"connect": 0}

    def unexpected_connect(*_args, **_kwargs):
        calls["connect"] += 1
        raise RuntimeError("must not be reached")

    monkeypatch.setattr(kanban_tools, "_connect", unexpected_connect)

    class Agent:
        def steer(self, _message):
            raise AssertionError("child must not receive parent operator comments")

    with _child_context(child_kind):
        assert kanban_tools.inject_new_comments_from_env(Agent()) is False
    assert calls["connect"] == 0


@pytest.mark.parametrize("warmup_order", ["parent-first", "child-first"])
@pytest.mark.parametrize("child_kind", ["cron", "delegate"])
def test_send_message_worker_gate_is_context_local(
    worker_env, child_kind, warmup_order, monkeypatch
):
    import gateway.session_context as session_context
    import gateway.status as gateway_status
    from tools import send_message_tool
    from tools.registry import _check_fn_cached

    monkeypatch.setattr(session_context, "get_session_env", lambda *_args: "")
    monkeypatch.setattr(gateway_status, "is_gateway_running", lambda: False)

    check = send_message_tool._check_send_message
    assert getattr(check, "_hermes_context_dependent", False) is True

    if warmup_order == "parent-first":
        assert _check_fn_cached(check) is True
        with _child_context(child_kind):
            assert _check_fn_cached(check) is False
    else:
        with _child_context(child_kind):
            assert _check_fn_cached(check) is False
        assert _check_fn_cached(check) is True
