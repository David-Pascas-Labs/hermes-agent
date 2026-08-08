"""Unit tests for gateway.runtime_footer — the opt-in runtime-metadata footer
appended to final gateway replies."""

from __future__ import annotations

import copy
import os
import sqlite3

import pytest

from gateway.runtime_footer import (
    RuntimeCounters,
    _home_relative_cwd,
    _model_short,
    append_footer_for_delivery,
    build_footer_line,
    format_runtime_footer,
    project_execution_footer_state,
    read_local_session_counters,
    resolve_footer_config,
)


# ---------------------------------------------------------------------------
# _model_short + _home_relative_cwd
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "model,expected",
    [
        ("openai/gpt-5.4", "gpt-5.4"),
        ("anthropic/claude-sonnet-4.6", "claude-sonnet-4.6"),
        ("gpt-5.4", "gpt-5.4"),
        ("", ""),
        (None, ""),
    ],
)
def test_model_short_drops_vendor_prefix(model, expected):
    assert _model_short(model) == expected


def test_home_relative_cwd_collapses_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    sub = tmp_path / "projects" / "hermes"
    sub.mkdir(parents=True)
    result = _home_relative_cwd(str(sub))
    assert result == "~/projects/hermes"


# ---------------------------------------------------------------------------
# format_runtime_footer
# ---------------------------------------------------------------------------

def test_format_footer_all_fields(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("TERMINAL_CWD", str(tmp_path / "projects" / "hermes"))
    (tmp_path / "projects" / "hermes").mkdir(parents=True)
    out = format_runtime_footer(
        model="openrouter/openai/gpt-5.4",
        context_tokens=68000,
        context_length=100000,
        cwd=None,  # falls back to TERMINAL_CWD env var
        fields=("model", "context_pct", "cwd"),
    )
    assert out == "gpt-5.4 · 68% · ~/projects/hermes"


def test_format_footer_skips_missing_context_length():
    out = format_runtime_footer(
        model="openai/gpt-5.4",
        context_tokens=500,
        context_length=None,
        cwd="/tmp/wd",
        fields=("model", "context_pct", "cwd"),
    )
    # context_pct dropped silently; no "?%" artifact
    assert "%" not in out
    assert "gpt-5.4" in out
    assert "/tmp/wd" in out


def test_format_footer_renders_compact_numeric_telemetry_without_identifiers():
    counters = RuntimeCounters(
        api_calls=7,
        input_tokens=12345,
        output_tokens=678,
        cache_read_tokens=9000,
        cache_write_tokens=321,
        reasoning_tokens=456,
        tool_calls=12,
    )
    out = format_runtime_footer(
        model="secret-provider/private-model",
        context_tokens=0,
        context_length=None,
        cwd="/home/alice/private-client",
        fields=(
            "api_calls",
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_write_tokens",
            "reasoning_tokens",
            "tool_calls",
        ),
        counters=counters,
    )

    assert out == "api 7 · in 12.3k · out 678 · cache 9.0k · cache+ 321 · reason 456 · tools 12"
    assert "private" not in out
    assert "alice" not in out


def test_execution_footer_state_drops_response_history_and_arbitrary_metadata():
    snapshot = project_execution_footer_state(
        {
            "session_id": "lookup-only-session-id",
            "api_calls": 2,
            "input_tokens": 5000,
            "final_response": "private answer",
            "messages": [{"role": "user", "content": "private prompt"}],
            "private_note": "client name",
        }
    )

    assert snapshot == {
        "session_id": "lookup-only-session-id",
        "api_calls": 2,
        "input_tokens": 5000,
    }


# ---------------------------------------------------------------------------
# resolve_footer_config
# ---------------------------------------------------------------------------


def test_resolve_platform_override_wins():
    user = {
        "display": {
            "runtime_footer": {"enabled": True, "fields": ["model"]},
            "platforms": {
                "slack": {"runtime_footer": {"enabled": False}},
            },
        },
    }
    # Telegram picks up the global enable
    assert resolve_footer_config(user, "telegram")["enabled"] is True
    # Slack overrides to off
    assert resolve_footer_config(user, "slack")["enabled"] is False


def test_resolve_platform_can_add_fields_only():
    user = {
        "display": {
            "runtime_footer": {"enabled": True},
            "platforms": {
                "discord": {"runtime_footer": {"fields": ["context_pct"]}},
            },
        },
    }
    tg = resolve_footer_config(user, "telegram")
    assert tg["enabled"] is True
    assert tg["fields"] == [
        "api_calls",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "tool_calls",
    ]
    dc = resolve_footer_config(user, "discord")
    assert dc["enabled"] is True
    assert dc["fields"] == ["context_pct"]


def test_default_fields_are_numeric_and_identifier_free():
    resolved = resolve_footer_config({}, "telegram")

    assert resolved["fields"] == [
        "api_calls",
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "tool_calls",
    ]
    assert "model" not in resolved["fields"]
    assert "cwd" not in resolved["fields"]


# ---------------------------------------------------------------------------
# build_footer_line — top-level entry point used by gateway/run.py
# ---------------------------------------------------------------------------


def test_build_footer_per_platform_off_suppresses():
    user = {
        "display": {
            "runtime_footer": {"enabled": True},
            "platforms": {"slack": {"runtime_footer": {"enabled": False}}},
        },
    }
    out = build_footer_line(
        user_config=user,
        platform_key="slack",
        model="openai/gpt-5.4",
        context_tokens=10, context_length=100,
        cwd="/tmp",
    )
    assert out == ""


def test_read_local_session_counters_uses_existing_state_db_read_only(tmp_path):
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                api_call_count INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cache_read_tokens INTEGER,
                cache_write_tokens INTEGER,
                reasoning_tokens INTEGER,
                tool_call_count INTEGER
            )"""
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("private-session-id", 7, 12345, 678, 9000, 321, 456, 12),
        )

    before = db_path.read_bytes()
    counters = read_local_session_counters(db_path, "private-session-id")

    assert counters == RuntimeCounters(
        api_calls=7,
        input_tokens=12345,
        output_tokens=678,
        cache_read_tokens=9000,
        cache_write_tokens=321,
        reasoning_tokens=456,
        tool_calls=12,
    )
    assert db_path.read_bytes() == before
    assert not list(tmp_path.glob("state.db-*"))


def test_read_local_session_counters_fails_soft_on_malformed_legacy_values(tmp_path):
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                api_call_count TEXT,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cache_read_tokens INTEGER,
                cache_write_tokens INTEGER,
                reasoning_tokens INTEGER,
                tool_call_count INTEGER
            )"""
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("sess", "not-a-number", 1, 2, 3, 4, 5, 6),
        )

    assert read_local_session_counters(db_path, "sess") is None


def test_build_footer_reads_telemetry_fields_from_state_db(tmp_path):
    db_path = tmp_path / "state.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                api_call_count INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cache_read_tokens INTEGER,
                cache_write_tokens INTEGER,
                reasoning_tokens INTEGER,
                tool_call_count INTEGER
            )"""
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("sess", 3, 2000, 400, 1500, 0, 25, 8),
        )

    out = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["api_calls", "input_tokens", "output_tokens", "tool_calls"],
                }
            }
        },
        platform_key="telegram",
        model=None,
        context_tokens=0,
        context_length=None,
        db_path=db_path,
        session_id="sess",
    )

    assert out == "api 3 · in 2.0k · out 400 · tools 8"


def test_build_footer_falls_back_to_existing_execution_counters_when_db_unavailable(tmp_path):
    out = build_footer_line(
        user_config={
            "display": {
                "runtime_footer": {
                    "enabled": True,
                    "fields": ["api_calls", "input_tokens", "output_tokens", "cache_read_tokens"],
                }
            }
        },
        platform_key=None,
        model=None,
        context_tokens=0,
        context_length=None,
        db_path=tmp_path / "missing-state.db",
        session_id="sess",
        execution={
            "api_calls": 2,
            "input_tokens": 5000,
            "output_tokens": 750,
            "cache_read_tokens": 4200,
            "private_note": "must never render",
        },
    )

    assert out == "api 2 · in 5.0k · out 750 · cache 4.2k"
    assert "private" not in out


def test_footer_transport_is_token_free_and_history_invariant(tmp_path):
    class InstrumentedProvider:
        def __init__(self):
            self.calls = 0
            self.payloads = []
            self.tokens = {"input": 0, "output": 0, "cache": 0}

        def complete(self, payload):
            self.calls += 1
            self.payloads.append(copy.deepcopy(payload))
            self.tokens = {"input": 111, "output": 22, "cache": 33}
            return "unchanged answer"

    def run(enabled):
        from hermes_state import SessionDB

        provider = InstrumentedProvider()
        request_payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        answer = provider.complete(request_payload)

        # Persist exactly the canonical provider exchange through the real
        # state.db API before display chrome is constructed.
        db_path = tmp_path / f"state-{'on' if enabled else 'off'}.db"
        db = SessionDB(db_path=db_path)
        session_id = "private-session-id"
        db.create_session(session_id, source="test")
        db.append_message(session_id, role="user", content="hello")
        db.append_message(session_id, role="assistant", content=answer)
        db.update_token_counts(
            session_id,
            input_tokens=provider.tokens["input"],
            output_tokens=provider.tokens["output"],
            cache_read_tokens=provider.tokens["cache"],
            api_call_count=provider.calls,
        )
        db.flush_token_counts()
        persisted_history = [
            {"role": message["role"], "content": message["content"]}
            for message in db.get_messages(session_id)
        ]

        footer = build_footer_line(
            user_config={
                "display": {
                    "runtime_footer": {
                        "enabled": enabled,
                        "fields": [
                            "api_calls",
                            "input_tokens",
                            "output_tokens",
                            "cache_read_tokens",
                        ],
                    }
                }
            },
            platform_key="terminal",
            model="test-model",
            context_tokens=0,
            context_length=None,
            db_path=db_path,
            session_id=session_id,
        )
        delivered = append_footer_for_delivery(
            answer,
            footer,
            already_sent=False,
            intentional_silence=False,
        )
        history_after_footer = [
            {"role": message["role"], "content": message["content"]}
            for message in db.get_messages(session_id)
        ]
        db.close()
        assert history_after_footer == persisted_history
        return provider, request_payload, persisted_history, delivered

    off = run(False)
    on = run(True)

    assert off[0].calls == on[0].calls == 1
    assert off[0].tokens == on[0].tokens == {"input": 111, "output": 22, "cache": 33}
    assert off[0].payloads == on[0].payloads
    assert off[1] == on[1]
    assert off[2] == on[2]
    assert off[3] == "unchanged answer"
    assert on[3] == "unchanged answer\n\napi 1 · in 111 · out 22 · cache 33"
    assert "private-session-id" not in on[3]


def test_already_streamed_gateway_response_never_creates_a_trailing_footer_delivery():
    sent = ["unchanged answer"]
    response = append_footer_for_delivery(
        "unchanged answer",
        "api 1 · in 111 · out 22",
        already_sent=True,
        intentional_silence=False,
    )

    assert response == "unchanged answer"
    assert sent == ["unchanged answer"]


