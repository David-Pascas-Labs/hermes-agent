"""Behavioral coverage for TUI cron management RPCs."""

from tui_gateway import server
from tools import cronjob_tools


def test_cron_manage_list_returns_every_job_beyond_compact_page(monkeypatch):
    jobs = [{"id": f"job-{index}", "name": f"Job {index}"} for index in range(25)]
    monkeypatch.setattr(
        cronjob_tools,
        "list_jobs",
        lambda include_disabled=False: jobs,
    )

    response = server.handle_request({
        "id": "cron-list",
        "method": "cron.manage",
        "params": {"action": "list"},
    })

    assert response is not None
    result = response["result"]
    assert result["count"] == 25
    assert result["returned"] == 25
    assert result["truncated"] is False
    assert result["next_offset"] is None
    assert [job["id"] for job in result["jobs"]] == [
        f"job-{index}" for index in range(25)
    ]
