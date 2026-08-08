"""Regression: first-turn ``session_meta`` row must be re-baselined into the
agent cache's message_count snapshot.

Bug
---
On a fresh gateway conversation the post-turn re-baseline
(``_refresh_agent_cache_message_count``) runs *before* the first-turn
``session_meta`` marker row is appended to the transcript:

    gateway/run.py:
        ... agent run completes ...
        self._refresh_agent_cache_message_count(...)   # snapshot taken HERE
        ...
        if not history:                                # session_meta written LATER
            append_to_transcript({"role": "session_meta", ...})

``append_to_transcript`` (no ``skip_db``) increments the session's
``message_count`` unconditionally (hermes_state.append_message), so the
snapshot ends up exactly +1 below the live on-disk count.

The cross-process coherence guard (#45966) compares the live count against
that snapshot on the *next* inbound message, sees ``live != snapshot``,
mistakes this process's own ``session_meta`` write for a foreign write, and
**rebuilds the cached agent on turn 2 of every fresh conversation** — silently
busting the per-conversation prompt cache the cache exists to protect.

The fix re-baselines AFTER all of this turn's transcript writes (including the
first-turn ``session_meta`` row), so the snapshot matches the live count and
the guard fires only on genuinely foreign writes.

This drives the REAL ``_handle_message_with_agent`` against a REAL SessionDB
(the ``session_meta`` write actually increments ``message_count``) and asserts
the invariant: after a first turn, snapshot == live count → next turn reuses.
"""

import sys
import threading
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionEntry, SessionSource


SESSION_KEY = "agent:main:telegram:group:-1001:12345"
SESSION_ID = "sess-first-turn"


def _bootstrap(monkeypatch, tmp_path, db):
    """GatewayRunner wired to a REAL SessionDB for count reads, mirroring the
    proven #42039 harness but with a live cache + real transcript counter."""
    fake_dotenv = types.ModuleType("dotenv")
    fake_dotenv.load_dotenv = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "dotenv", fake_dotenv)

    config = GatewayConfig()
    runner = gateway_run.GatewayRunner(config)
    runner.adapters = {}
    runner._running_agents = {}
    runner._running_agents_ts = {}
    runner._pending_messages = {}
    runner._pending_approvals = {}
    runner._is_user_authorized = lambda _source: True
    runner._set_session_env = lambda _context: None
    runner._handle_active_session_busy_message = AsyncMock(return_value=False)
    # REAL SessionDB behind the async facade the gateway holds — the
    # production re-baseline does ``await self._session_db.get_session(...)``,
    # so it must be the AsyncSessionDB wrapper, not the raw sync DB.
    from hermes_state import AsyncSessionDB

    runner._session_db = AsyncSessionDB(db)
    runner._recover_telegram_topic_thread_id = lambda _source: None
    runner._cache_session_source = lambda _key, _source: None
    runner._is_session_run_current = lambda _key, _gen: True
    runner._begin_session_run_generation = lambda _key: 1
    runner._reply_anchor_for_event = lambda _event: None
    runner._get_guild_id = lambda _event: None
    runner._should_send_voice_reply = lambda *_a, **_kw: False
    runner.hooks = MagicMock()
    runner.hooks.emit = AsyncMock()

    # Live agent cache (not a MagicMock) so the re-baseline actually rewrites
    # the snapshot tuple in place.
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.Lock()

    runner.session_store = MagicMock()
    runner.session_store.get_or_create_session.return_value = SessionEntry(
        session_key=SESSION_KEY,
        session_id=SESSION_ID,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        platform=Platform.TELEGRAM,
        chat_type="group",
    )
    # Empty history → triggers the first-turn ``session_meta`` write path.
    runner.session_store.load_transcript.return_value = []
    runner.session_store.has_platform_message_id.return_value = False
    runner.session_store.update_session = MagicMock()

    # Mirror the real SessionStore.append_to_transcript: forward non-skip_db
    # writes to the real DB so a ``session_meta`` row genuinely increments
    # message_count, exactly as in production. skip_db=True writes (the agent
    # already persisted them via _flush_messages_to_session_db) are no-ops here.
    def _append(session_id, message, skip_db=False):
        if not skip_db:
            db.append_message(
                session_id=session_id,
                role=message.get("role", "unknown"),
                content=message.get("content"),
            )

    runner.session_store.append_to_transcript = MagicMock(side_effect=_append)

    monkeypatch.setattr(gateway_run, "_hermes_home", tmp_path)
    monkeypatch.setattr(
        gateway_run, "_resolve_runtime_agent_kwargs", lambda: {"api_key": "fake"}
    )
    monkeypatch.setattr(
        "agent.model_metadata.get_model_context_length",
        lambda *_args, **_kwargs: 100_000,
    )
    return runner


def _event():
    return MessageEvent(
        text="hello world",
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="-1001",
            chat_type="group",
            user_id="12345",
        ),
        message_id="msg-1",
    )


def _source():
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="group",
        user_id="12345",
    )


def _live_count(db, session_id):
    row = db.get_session(session_id)
    return (row.get("message_count", 0) if row else 0)


@pytest.mark.asyncio
async def test_first_turn_session_meta_is_captured_by_rebaseline(
    monkeypatch, tmp_path
):
    """After a fresh first turn, the cache snapshot must equal the live
    message_count — including the first-turn ``session_meta`` row.

    WITHOUT the fix the re-baseline snapshots the count *before* the
    session_meta append, leaving the snapshot one short; the cross-process
    guard then rebuilds the cached agent on turn 2 (prompt-cache churn).
    """
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "sessions.db")
    db.create_session(SESSION_ID, source="telegram")

    runner = _bootstrap(monkeypatch, tmp_path, db)

    # Cache snapshot taken at agent-BUILD time = count before this turn's
    # writes (a fresh session → 0). This is what the #45966 guard stores.
    build_count = _live_count(db, SESSION_ID)
    agent_obj = object()
    with runner._agent_cache_lock:
        runner._agent_cache[SESSION_KEY] = (agent_obj, "sig", build_count)

    # Stubbed agent run: the gateway's own user/assistant rows are persisted
    # by the agent (skip_db=True downstream); only the session_meta marker is
    # written by the gateway with skip_db=False.
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Hi there!",
            "messages": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "tools": [{"name": "noop"}],
            "history_offset": 0,
            "last_prompt_tokens": 0,
        }
    )

    await runner._handle_message_with_agent(_event(), _source(), SESSION_KEY, 1)

    # The first-turn session_meta row was written → live count advanced.
    live = _live_count(db, SESSION_ID)
    assert live == build_count + 1, (
        "first-turn session_meta should increment message_count by exactly 1"
    )

    # THE INVARIANT: the cache snapshot must now equal the live count, so the
    # next turn's cross-process guard reuses the cached agent.
    with runner._agent_cache_lock:
        cached = runner._agent_cache[SESSION_KEY]
    snapshot = cached[2]
    assert snapshot == live, (
        f"cache snapshot {snapshot} != live count {live}: the first-turn "
        f"session_meta write was not re-baselined, so the #45966 guard will "
        f"rebuild the cached agent on turn 2 and bust the prompt cache."
    )
    # And the cached agent instance must be untouched (never rebuilt).
    assert cached[0] is agent_obj


@pytest.mark.asyncio
async def test_runtime_footer_is_added_only_after_hooks_and_persistence(
    monkeypatch, tmp_path
):
    """The real gateway handler decorates only its returned delivery text."""
    import yaml
    from hermes_state import SessionDB

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "display": {
                    "runtime_footer": {
                        "enabled": True,
                        "fields": ["api_calls", "input_tokens", "output_tokens"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    db = SessionDB(db_path=tmp_path / "sessions.db")
    db.create_session(SESSION_ID, source="telegram")
    db.update_token_counts(
        SESSION_ID,
        input_tokens=111,
        output_tokens=22,
        api_call_count=1,
        absolute=True,
    )
    db.flush_token_counts()
    runner = _bootstrap(monkeypatch, tmp_path, db)
    runner._should_send_voice_reply = lambda *_a, **_kw: True
    runner._send_voice_reply = AsyncMock()
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Hi there!",
            "messages": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "tools": [],
            "history_offset": 0,
            "session_id": SESSION_ID,
            "api_calls": 1,
            "input_tokens": 111,
            "output_tokens": 22,
        }
    )

    event = _event()
    response = await runner._handle_message_with_agent(
        event, _source(), SESSION_KEY, 1
    )

    assert response == "Hi there!\n\napi 1 · in 111 · out 22"
    assert getattr(event, "_runtime_footer_canonical_response", None) == "Hi there!"
    agent_end_events = [
        call.args[1]
        for call in runner.hooks.emit.await_args_list
        if call.args and call.args[0] == "agent:end"
    ]
    assert len(agent_end_events) == 1
    assert agent_end_events[0]["response"] == "Hi there!"
    assert "api 1" not in agent_end_events[0]["response"]
    assert runner._send_voice_reply.await_args.args[1] == "Hi there!"
    persisted_contents = [
        call.args[1].get("content")
        for call in runner.session_store.append_to_transcript.call_args_list
    ]
    assert all("api 1" not in str(content) for content in persisted_contents)
    db.close()


@pytest.mark.asyncio
async def test_streamed_gateway_turn_never_sends_a_trailing_footer(
    monkeypatch, tmp_path
):
    """Already-delivered streaming returns without any footer delivery."""
    import yaml
    from hermes_state import SessionDB

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "display": {
                    "runtime_footer": {"enabled": True, "fields": ["api_calls"]}
                }
            }
        ),
        encoding="utf-8",
    )
    db = SessionDB(db_path=tmp_path / "sessions.db")
    db.create_session(SESSION_ID, source="telegram")
    runner = _bootstrap(monkeypatch, tmp_path, db)
    adapter = MagicMock()
    adapter._streaming_tts_turn_completed.return_value = False
    adapter.send = AsyncMock()
    runner.adapters[Platform.TELEGRAM] = adapter
    runner._deliver_media_from_response = AsyncMock()
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "Hi there!",
            "messages": [],
            "tools": [],
            "history_offset": 0,
            "session_id": SESSION_ID,
            "api_calls": 1,
            "already_sent": True,
        }
    )

    response = await runner._handle_message_with_agent(
        _event(), _source(), SESSION_KEY, 1
    )

    assert response is None
    delivered_texts = [
        str(call.args[1] if len(call.args) > 1 else call.kwargs.get("content", ""))
        for call in adapter.send.await_args_list
    ]
    assert all("api 1" not in text for text in delivered_texts)
    db.close()


@pytest.mark.asyncio
async def test_active_goal_judge_receives_canonical_response_before_display_decoration(
    monkeypatch, tmp_path
):
    """Reasoning and telemetry may be delivered, but never enter the goal judge."""
    import hermes_cli.goals as goals
    import yaml
    from hermes_state import SessionDB

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "display": {
                    "show_reasoning": True,
                    "runtime_footer": {
                        "enabled": True,
                        "fields": ["api_calls", "input_tokens", "output_tokens"],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    db = SessionDB(db_path=tmp_path / "sessions.db")
    db.create_session(SESSION_ID, source="telegram")
    db.update_token_counts(
        SESSION_ID,
        input_tokens=111,
        output_tokens=22,
        api_call_count=1,
        absolute=True,
    )
    db.flush_token_counts()
    runner = _bootstrap(monkeypatch, tmp_path, db)
    runner._goal_max_turns_from_config = lambda: 20
    runner._run_agent = AsyncMock(
        return_value={
            "final_response": "canonical answer",
            "last_reasoning": "display-only reasoning",
            "messages": [
                {"role": "user", "content": "hello world"},
                {"role": "assistant", "content": "canonical answer"},
            ],
            "tools": [],
            "history_offset": 0,
            "session_id": SESSION_ID,
            "api_calls": 1,
            "input_tokens": 111,
            "output_tokens": 22,
        }
    )

    seen = {}

    class FakeGoalManager:
        def __init__(self, *, session_id, default_max_turns):
            seen["session_id"] = session_id
            seen["max_turns"] = default_max_turns

        def is_active(self):
            return True

        def evaluate_after_turn(self, last_response, **kwargs):
            seen["last_response"] = last_response
            return {
                "message": "",
                "should_continue": False,
                "continuation_prompt": None,
            }

    monkeypatch.setattr(goals, "GoalManager", FakeGoalManager)
    event = _event()
    delivered_response = await runner._handle_message_with_agent(
        event, _source(), SESSION_KEY, 1
    )
    final_response = runner._goal_response_for_result(event, delivered_response)
    await runner._post_turn_goal_continuation(
        session_entry=types.SimpleNamespace(session_id=SESSION_ID),
        source=None,
        final_response=final_response,
    )

    assert "display-only reasoning" in delivered_response
    assert "api 1 · in 111 · out 22" in delivered_response
    assert getattr(event, "_runtime_footer_canonical_response", None) == "canonical answer"
    assert seen["last_response"] == "canonical answer"
    db.close()


