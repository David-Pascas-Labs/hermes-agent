"""Tests for terminal truncation spill + metadata (deferred retrieval)."""

import importlib
import json
import os
import stat
from pathlib import Path

import pytest

from tools.terminal_tool import terminal_tool


@pytest.fixture
def small_cap(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    import tools.tool_output_limits as lim
    monkeypatch.setattr(lim, "_cached_limits", {
        "max_bytes": 2000, "max_lines": 2000, "max_line_length": 2000,
    })
    return tmp_path


class TestTruncationSpill:
    def test_truncated_output_has_metadata_and_spill(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"print('marker_head'); [print(f'row_{i}', 'x'*80) for i in range(200)]; print('marker_tail')\"",
            task_id="t-spill-1"))
        assert r["exit_code"] == 0
        assert "OUTPUT TRUNCATED" in r["output"]
        assert r["output_total_chars"] > 2000
        p = Path(r["full_output_path"])
        assert p.exists()
        expected_dir = (small_cap / ".hermes" / "cache" / "terminal-output").resolve()
        assert p.parent.resolve() == expected_dir
        if os.name != "nt":
            assert stat.S_IMODE(p.stat().st_mode) == 0o600
            assert stat.S_IMODE(expected_dir.stat().st_mode) == 0o700
        full = p.read_text()
        assert "marker_head" in full and "marker_tail" in full
        # The spill contains rows that were cut from the visible window.
        assert "row_100 " in full
        assert "read_file" in r["truncation_note"]

    def test_small_output_has_no_metadata(self, small_cap):
        r = json.loads(terminal_tool("echo tiny", task_id="t-spill-2"))
        assert r["exit_code"] == 0
        assert "full_output_path" not in r
        assert "output_total_chars" not in r

    def test_spill_is_redacted(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"print('sk-proj-' + 'a1B2c3D4e5F6g7H8i9J0' * 3); [print('pad', 'y'*90) for i in range(200)]\"",
            task_id="t-spill-3"))
        p = Path(r["full_output_path"])
        full = p.read_text()
        assert "a1B2c3D4e5F6g7H8i9J0a1B2c3D4e5F6g7H8i9J0" not in full

    def test_old_spills_cleaned(self, small_cap, tmp_path):
        spill_dir = tmp_path / ".hermes" / "cache" / "terminal-output"
        spill_dir.mkdir(parents=True, exist_ok=True)
        stale = spill_dir / "out-1-2-dead.log"
        stale.write_text("old")
        os.utime(stale, (1, 1))
        json.loads(terminal_tool(
            "python3 -c \"[print('z'*90) for i in range(200)]\"", task_id="t-spill-4"))
        assert not stale.exists()

    def test_remote_style_whole_result_gets_host_spill(
        self, small_cap, monkeypatch
    ):
        """Backends that bypass BaseEnvironment streaming still stay recoverable."""
        terminal_module = importlib.import_module("tools.terminal_tool")
        captured = {}
        full_output = "remote_head\n" + ("r" * 10_000) + "\nremote_tail"

        class WholeResultEnv:
            cwd = str(small_cap)

            def execute(self, command, **kwargs):
                captured.update(kwargs)
                return {"output": full_output, "returncode": 0}

        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setitem(
            terminal_module._active_environments, "default", WholeResultEnv()
        )

        r = json.loads(terminal_tool("printf remote", task_id="t-spill-remote"))

        assert captured["bounded_capture"] is True
        assert len(r["output"]) <= 2100
        assert r["output_total_chars"] == len(full_output)
        full = Path(r["full_output_path"]).read_text()
        assert full == full_output

    def test_transform_hook_replacement_owns_spill_contract(
        self, small_cap, monkeypatch
    ):
        """A hook replacement, not the raw process stream, is recoverable."""
        transformed = "hook_head\n" + ("h" * 10_000) + "\nhook_tail"

        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: [transformed]
            if hook_name == "transform_terminal_output"
            else [],
        )

        r = json.loads(terminal_tool(
            "python3 -c \"print('raw_head'); "
            "[print('raw', 'x'*90) for i in range(200)]; print('raw_tail')\"",
            task_id="t-spill-hook"))

        assert r["output_total_chars"] == len(transformed)
        assert len(r["output"]) <= 2100
        assert Path(r["full_output_path"]).read_text() == transformed

    def test_small_transform_hook_replacement_discards_raw_spill(
        self, small_cap, monkeypatch
    ):
        monkeypatch.setattr(
            "hermes_cli.lifecycle.invoke_hook",
            lambda hook_name, **kwargs: ["small replacement"]
            if hook_name == "transform_terminal_output"
            else [],
        )

        r = json.loads(terminal_tool(
            "python3 -c \"[print('raw', 'x'*90) for i in range(200)]\"",
            task_id="t-spill-hook-small"))

        assert r["output"] == "small replacement"
        assert "full_output_path" not in r
        assert "output_total_chars" not in r
        spill_dir = small_cap / ".hermes" / "cache" / "terminal-output"
        assert list(spill_dir.glob("out-*.log")) == []

    def test_spill_remains_lossless_beyond_upstream_five_megabyte_cap(
        self, small_cap
    ):
        payload_chars = 5_100_000
        r = json.loads(terminal_tool(
            "python3 -c \"import sys; sys.stdout.write('marker_head' + "
            f"'q'*{payload_chars} + 'marker_tail')\"",
            task_id="t-spill-large"))

        p = Path(r["full_output_path"])
        full = p.read_text()
        assert full.startswith("marker_head")
        assert full.endswith("marker_tail")
        assert full.count("q") == payload_chars
        assert "spill capped" not in full

    def test_untrusted_backend_spill_path_is_never_followed(
        self, small_cap, monkeypatch, tmp_path
    ):
        terminal_module = importlib.import_module("tools.terminal_tool")
        victim = tmp_path / "outside-victim.log"
        victim.write_text("do-not-touch")

        class UntrustedPathEnv:
            cwd = str(small_cap)

            def execute(self, command, **kwargs):
                return {
                    "output": "tiny",
                    "returncode": 0,
                    "output_total_chars": 9999,
                    "full_output_path": str(victim),
                }

        monkeypatch.setenv("TERMINAL_ENV", "local")
        monkeypatch.setitem(
            terminal_module._active_environments, "default", UntrustedPathEnv()
        )

        r = json.loads(terminal_tool("printf safe", task_id="t-spill-untrusted"))

        assert victim.read_text() == "do-not-touch"
        assert "full_output_path" not in r
        assert "output_total_chars" not in r

    @pytest.mark.skipif(os.name == "nt", reason="symlink permissions vary on Windows")
    def test_profile_cache_symlink_escape_fails_closed(
        self, small_cap, tmp_path
    ):
        hermes_home = small_cap / ".hermes"
        hermes_home.mkdir()
        outside = tmp_path / "outside-cache"
        outside.mkdir()
        (hermes_home / "cache").symlink_to(outside, target_is_directory=True)

        r = json.loads(terminal_tool(
            "python3 -c \"[print('escape', 'x'*90) for i in range(200)]\"",
            task_id="t-spill-symlink"))

        assert "OUTPUT TRUNCATED" in r["output"]
        assert "full_output_path" not in r
        assert list(outside.iterdir()) == []

    def test_timeout_still_gets_spill(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"import sys,time; "
            "sys.stdout.write('timeout_head' + 't'*10000 + 'timeout_tail'); "
            "sys.stdout.flush(); time.sleep(5)\"",
            timeout=1,
            task_id="t-spill-timeout"))

        assert r["exit_code"] == 124
        assert "Command timed out" in r["output"]
        full = Path(r["full_output_path"]).read_text()
        assert full.startswith("timeout_head")
        assert "timeout_tail" in full

    def test_failed_command_still_gets_spill(self, small_cap):
        r = json.loads(terminal_tool(
            "python3 -c \"[print('e'*90) for i in range(200)]; import sys; sys.exit(3)\"",
            task_id="t-spill-5"))
        assert r["exit_code"] == 3
        assert Path(r["full_output_path"]).exists()
