"""Default list/status tool outputs stay compact unless full data is requested."""

from __future__ import annotations

import json


def _job(index: int) -> dict:
    return {
        "id": f"job-{index}",
        "name": f"Job {index}",
        "prompt": "very detailed prompt " + ("x" * 200),
        "schedule_display": "every 1h",
        "next_run_at": "2026-08-05T12:00:00Z",
        "last_status": "success",
        "enabled": True,
        "state": "scheduled",
    }


def test_cron_list_is_compact_and_paginated_by_default(monkeypatch):
    import tools.cronjob_tools as cron_tools

    monkeypatch.setattr(
        cron_tools,
        "list_jobs",
        lambda include_disabled=False: [_job(i) for i in range(25)],
    )
    monkeypatch.setattr(cron_tools, "get_max_list_items", lambda: 20)

    result = json.loads(cron_tools.cronjob(action="list"))

    assert result["success"] is True
    assert result["count"] == 25
    assert result["returned"] == 20
    assert result["truncated"] is True
    assert result["next_offset"] == 20
    assert len(result["jobs"]) == 20
    assert "prompt" not in result["jobs"][0]
    assert "prompt_preview" not in result["jobs"][0]
    assert "full=true" in result["hint"]


def test_cron_list_full_data_requires_explicit_flag(monkeypatch):
    import tools.cronjob_tools as cron_tools

    monkeypatch.setattr(
        cron_tools, "list_jobs", lambda include_disabled=False: [_job(0), _job(1)]
    )

    result = json.loads(cron_tools.cronjob(action="list", full=True))

    assert result["returned"] == 2
    assert result["truncated"] is False
    assert result["jobs"][0]["prompt"].startswith("very detailed prompt")


def test_process_list_is_compact_and_paginated_by_default(monkeypatch):
    import tools.process_registry as process_tools

    processes = [
        {"session_id": str(i), "status": "running", "command": "cmd"} for i in range(25)
    ]
    monkeypatch.setattr(
        process_tools.process_registry, "list_sessions", lambda **kwargs: processes
    )
    monkeypatch.setattr(process_tools, "get_max_list_items", lambda: 20)

    result = json.loads(process_tools._handle_process({"action": "list"}, task_id="t"))

    assert result["count"] == 25
    assert result["returned"] == 20
    assert result["truncated"] is True
    assert result["next_offset"] == 20
    assert len(result["processes"]) == 20
    assert "full=true" in result["hint"]


def test_process_log_limit_is_capped_without_explicit_full(monkeypatch):
    import tools.process_registry as process_tools

    seen = []
    monkeypatch.setattr(process_tools, "get_max_log_lines", lambda: 200)
    monkeypatch.setattr(
        process_tools.process_registry,
        "read_log",
        lambda session_id, offset, limit: seen.append(limit) or {"output": "ok"},
    )

    json.loads(
        process_tools._handle_process({
            "action": "log",
            "session_id": "p1",
            "limit": 999,
        })
    )
    json.loads(
        process_tools._handle_process({
            "action": "log",
            "session_id": "p1",
            "limit": 999,
            "full": True,
        })
    )

    assert seen == [200, 999]


def test_process_log_reports_default_truncation_and_full_path(monkeypatch):
    import tools.process_registry as process_tools

    monkeypatch.setattr(process_tools, "get_max_log_lines", lambda: 200)
    monkeypatch.setattr(
        process_tools.process_registry,
        "read_log",
        lambda session_id, offset, limit: {
            "session_id": session_id,
            "output": "ok",
            "total_lines": 1_000,
            "showing": f"{limit} lines",
        },
    )

    compact = json.loads(
        process_tools._handle_process({
            "action": "log",
            "session_id": "p1",
            "limit": 999,
        })
    )
    explicit = json.loads(
        process_tools._handle_process({
            "action": "log",
            "session_id": "p1",
            "limit": 999,
            "full": True,
        })
    )

    assert compact["truncated"] is True
    assert "full=true" in compact["hint"]
    assert "truncated" not in explicit
    assert "hint" not in explicit
