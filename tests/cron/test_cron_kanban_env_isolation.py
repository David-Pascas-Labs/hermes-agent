"""Cron sessions must not inherit a kanban worker's dispatcher identity.

A cron job can be fired *in-process* from a kanban worker: the worker is a
normal ``hermes chat -q`` CLI agent (its default toolset includes ``cronjob``)
running with ``HERMES_KANBAN_TASK`` legitimately set in its own environment,
and ``cronjob(action="run")`` calls ``run_one_job()`` -> ``run_job()`` in that
same process.

Without isolation the cron ``AIAgent`` is misidentified as that worker: the
kanban toolset is force-added, the kanban-worker protocol is injected into its
system prompt, and ``kanban_complete`` defaults ``task_id`` to
``$HERMES_KANBAN_TASK`` — letting an unrelated cron job close the worker's task
and overwrite real results.

The isolation is a **ContextVar**, deliberately not an ``os.environ`` clear:
``os.environ`` is process-global and shared with

  * the worker's own claim heartbeat (``run_agent._touch_activity`` ->
    ``heartbeat_current_worker_from_env``), which would starve and let the
    dispatcher reclaim a task whose worker is still alive;
  * the gateway's kanban watchers, which do their own board save/restore;
  * concurrent cron jobs on the parallel pool, which take a *shared* read lock
    and can interleave one another's snapshot/restore.

So these tests assert both that the identity is hidden AND that the environment
is left completely untouched.
"""

from __future__ import annotations

import os
import threading

import pytest


@pytest.fixture(autouse=True)
def _clear_kanban_detect_cache():
    """Reset every process-wide cache touched by Kanban availability checks."""
    import agent.skill_utils as su
    import model_tools
    from tools.registry import invalidate_check_fn_cache

    su._ENV_DETECT_CACHE.pop("kanban", None)
    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()
    yield
    su._ENV_DETECT_CACHE.pop("kanban", None)
    invalidate_check_fn_cache()
    model_tools._clear_tool_defs_cache()


@pytest.fixture()
def worker_env(monkeypatch):
    """Simulate running inside a dispatcher-spawned kanban worker."""
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_worker_real_task")
    monkeypatch.setenv("HERMES_KANBAN_WORKSPACE", "/tmp/ws")
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "42")
    monkeypatch.setenv("HERMES_KANBAN_CLAIM_LOCK", "lock-abc")
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "team-alpha")


# ---------------------------------------------------------------------------
# The predicate itself
# ---------------------------------------------------------------------------

class TestDispatcherOwnedPredicate:
    def test_default_without_worker_identity_is_not_dispatcher_owned(
        self, monkeypatch
    ):
        from agent.delegation_context import is_dispatcher_owned_worker_context

        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        assert is_dispatcher_owned_worker_context() is False

    def test_worker_identity_is_dispatcher_owned(self, worker_env):
        from agent.delegation_context import is_dispatcher_owned_worker_context

        assert is_dispatcher_owned_worker_context() is True

    def test_false_inside_non_dispatcher_context(self, worker_env):
        from agent.delegation_context import (
            is_dispatcher_owned_worker_context,
            non_dispatcher_owned_context,
        )

        with non_dispatcher_owned_context():
            assert is_dispatcher_owned_worker_context() is False
        assert is_dispatcher_owned_worker_context() is True

    def test_token_form_restores(self, worker_env):
        from agent.delegation_context import (
            enter_non_dispatcher_owned_context,
            exit_non_dispatcher_owned_context,
            is_dispatcher_owned_worker_context,
        )

        token = enter_non_dispatcher_owned_context()
        assert is_dispatcher_owned_worker_context() is False
        exit_non_dispatcher_owned_context(token)
        assert is_dispatcher_owned_worker_context() is True

    def test_nesting_restores_outer_value(self, worker_env):
        from agent.delegation_context import (
            is_dispatcher_owned_worker_context,
            non_dispatcher_owned_context,
        )

        with non_dispatcher_owned_context():
            with non_dispatcher_owned_context():
                assert is_dispatcher_owned_worker_context() is False
            assert is_dispatcher_owned_worker_context() is False
        assert is_dispatcher_owned_worker_context() is True

    def test_delegated_child_still_not_dispatcher_owned(self, monkeypatch):
        """The pre-existing delegate_task flag keeps its meaning."""
        import agent.delegation_context as dc

        token = dc._DELEGATED_CHILD_CONTEXT.set(True)
        try:
            assert dc.is_dispatcher_owned_worker_context() is False
        finally:
            dc._DELEGATED_CHILD_CONTEXT.reset(token)

    def test_thread_isolation(self, worker_env):
        """A ContextVar set in one thread must not leak into a sibling thread.

        This is the property an os.environ clear cannot provide, and the reason
        concurrent cron jobs can't corrupt each other.
        """
        from agent.delegation_context import (
            is_dispatcher_owned_worker_context,
            non_dispatcher_owned_context,
        )

        seen = {}
        release = threading.Event()

        def sibling():
            seen["sibling"] = is_dispatcher_owned_worker_context()
            release.set()

        def job():
            with non_dispatcher_owned_context():
                seen["job"] = is_dispatcher_owned_worker_context()
                t = threading.Thread(target=sibling)
                t.start()
                release.wait(5)
                t.join(5)

        t = threading.Thread(target=job)
        t.start()
        t.join(5)

        assert seen["job"] is False, "job thread must be marked non-dispatcher"
        assert seen["sibling"] is True, "sibling thread must be unaffected"


# ---------------------------------------------------------------------------
# The gates that consume it
# ---------------------------------------------------------------------------

class TestKanbanGatesRespectContext:
    def test_task_tools_hidden_from_cron_agent(self, worker_env):
        from agent.delegation_context import non_dispatcher_owned_context
        from tools import kanban_tools

        assert kanban_tools._check_kanban_mode() is True
        with non_dispatcher_owned_context():
            assert kanban_tools._check_kanban_mode() is False

    def test_complete_does_not_default_to_worker_task(self, worker_env):
        """The damage path: kanban_complete must not inherit the task id."""
        from agent.delegation_context import non_dispatcher_owned_context
        from tools import kanban_tools

        assert kanban_tools._default_task_id(None) == "t_worker_real_task"
        with non_dispatcher_owned_context():
            assert kanban_tools._default_task_id(None) is None

    def test_explicit_task_id_still_honoured(self, worker_env):
        """Only the implicit default is suppressed, not an explicit argument."""
        from agent.delegation_context import non_dispatcher_owned_context
        from tools import kanban_tools

        with non_dispatcher_owned_context():
            assert kanban_tools._default_task_id("t_explicit") == "t_explicit"

    def test_worker_runtime_metadata_is_not_inherited(self, worker_env):
        """Explicit tool args must not reactivate ambient worker ownership."""
        from agent.delegation_context import non_dispatcher_owned_context
        from tools import kanban_tools

        metadata = {"source": "cron"}
        with non_dispatcher_owned_context():
            assert kanban_tools._worker_run_id("t_worker_real_task") is None
            assert (
                kanban_tools._stamp_worker_session_metadata(
                    "t_worker_real_task", metadata
                )
                is metadata
            )
            assert kanban_tools._enforce_worker_task_ownership("t_other") is None
            assert kanban_tools._require_orchestrator_tool("kanban_list") is None

    def test_skill_environment_gate(self, worker_env):
        from agent.delegation_context import non_dispatcher_owned_context
        import agent.skill_utils as su

        su._ENV_DETECT_CACHE.pop("kanban", None)
        assert su._detect_environment("kanban") is True
        with non_dispatcher_owned_context():
            su._ENV_DETECT_CACHE.pop("kanban", None)
            assert su._detect_environment("kanban") is False

    def test_kanban_env_verdict_is_not_memoized(self, worker_env):
        """`kanban` must bypass _ENV_DETECT_CACHE: caching it process-wide would
        freeze whichever context asked first and leak it to the others."""
        from agent.delegation_context import non_dispatcher_owned_context
        import agent.skill_utils as su

        su._ENV_DETECT_CACHE.pop("kanban", None)
        assert su._detect_environment("kanban") is True
        with non_dispatcher_owned_context():
            # No manual cache clear here — the production code must not have
            # cached the previous True.
            assert su._detect_environment("kanban") is False
        assert su._detect_environment("kanban") is True

    def test_toolset_force_add_suppressed(self, worker_env):
        from agent.delegation_context import non_dispatcher_owned_context
        import model_tools

        assert model_tools._is_dispatcher_owned_worker() is True
        with non_dispatcher_owned_context():
            assert model_tools._is_dispatcher_owned_worker() is False

    def test_tool_search_policy_does_not_treat_cron_as_worker(
        self, monkeypatch, worker_env
    ):
        from agent.delegation_context import non_dispatcher_owned_context
        import model_tools
        import toolsets

        observed = []

        def policy(_enabled, *, platform, is_kanban_worker):
            observed.append((platform, is_kanban_worker))
            return frozenset()

        monkeypatch.setattr(toolsets, "progressive_disclosure_builtin_tools", policy)

        model_tools._tool_search_additional_deferrable_names([], platform="cron")
        with non_dispatcher_owned_context():
            model_tools._tool_search_additional_deferrable_names([], platform="cron")

        assert observed == [("cron", True), ("cron", False)]


# ---------------------------------------------------------------------------
# Real registry / AIAgent tool assembly
# ---------------------------------------------------------------------------

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


def _assemble_real_agent_tool_names(*, platform: str) -> set[str]:
    """Run the production AIAgent -> model_tools -> registry assembly path."""
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


def _child_context(kind: str):
    from agent.delegation_context import (
        delegated_child_context,
        non_dispatcher_owned_context,
    )

    return (
        non_dispatcher_owned_context()
        if kind == "cron"
        else delegated_child_context()
    )


class TestKanbanRegistryCacheContextIsolation:
    @pytest.fixture(autouse=True)
    def _disable_profile_opt_in(self, monkeypatch):
        """Only dispatcher ownership may expose Kanban in these contracts."""
        from tools import kanban_tools

        monkeypatch.setattr(kanban_tools, "_profile_has_kanban_toolset", lambda: False)

    def test_dispatcher_parent_receives_every_lifecycle_tool(self, worker_env):
        from tools import kanban_tools

        assert kanban_tools._check_kanban_mode() is True
        names = _assemble_real_agent_tool_names(platform="cli")
        assert KANBAN_LIFECYCLE_TOOLS <= names
        assert {"kanban_list", "kanban_unblock"}.isdisjoint(names)

    @pytest.mark.parametrize("child_kind", ["cron", "delegate"])
    def test_parent_warmup_does_not_expose_lifecycle_tools_to_child(
        self, worker_env, child_kind
    ):
        from tools import kanban_tools

        parent_names = _assemble_real_agent_tool_names(platform="cli")
        assert KANBAN_LIFECYCLE_TOOLS <= parent_names

        with _child_context(child_kind):
            assert kanban_tools._check_kanban_mode() is False
            child_names = _assemble_real_agent_tool_names(platform=child_kind)

        assert KANBAN_LIFECYCLE_TOOLS.isdisjoint(child_names)

    @pytest.mark.parametrize("child_kind", ["cron", "delegate"])
    def test_child_warmup_does_not_hide_lifecycle_tools_from_parent(
        self, worker_env, child_kind
    ):
        from tools import kanban_tools

        with _child_context(child_kind):
            assert kanban_tools._check_kanban_mode() is False
            child_names = _assemble_real_agent_tool_names(platform=child_kind)
        assert KANBAN_LIFECYCLE_TOOLS.isdisjoint(child_names)

        assert kanban_tools._check_kanban_mode() is True
        parent_names = _assemble_real_agent_tool_names(platform="cli")
        assert KANBAN_LIFECYCLE_TOOLS <= parent_names


# ---------------------------------------------------------------------------
# run_job wiring
# ---------------------------------------------------------------------------

class TestRunJobKanbanIsolation:
    @staticmethod
    def _install_stubs(monkeypatch, observed: dict, agent_cls=None):
        import sys

        import cron.scheduler as sched
        from agent.delegation_context import is_dispatcher_owned_worker_context

        class FakeAgent:
            def __init__(self, **kwargs):
                observed["dispatcher_owned_during_init"] = (
                    is_dispatcher_owned_worker_context()
                )
                observed["kanban_env_during_init"] = {
                    k: v for k, v in os.environ.items()
                    if k.startswith("HERMES_KANBAN_")
                }

            def run_conversation(self, *_a, **_kw):
                observed["dispatcher_owned_during_run"] = (
                    is_dispatcher_owned_worker_context()
                )
                return {"final_response": "done", "messages": []}

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

            def close(self):
                pass

        fake_mod = type(sys)("run_agent")
        fake_mod.AIAgent = agent_cls or FakeAgent
        monkeypatch.setitem(sys.modules, "run_agent", fake_mod)

        from hermes_cli import runtime_provider as _rtp

        monkeypatch.setattr(
            _rtp, "resolve_runtime_provider",
            lambda **_kw: {
                "provider": "test", "api_key": "k",
                "base_url": "http://test.local",
                "api_mode": "chat_completions",
            },
        )
        monkeypatch.setattr(
            sched, "_build_job_prompt", lambda job, prerun_script=None: "hi"
        )
        monkeypatch.setattr(sched, "_resolve_origin", lambda job: None)
        monkeypatch.setattr(sched, "_resolve_delivery_target", lambda job: None)
        monkeypatch.setattr(
            sched, "_resolve_cron_enabled_toolsets", lambda job, cfg: None
        )
        monkeypatch.setenv("HERMES_CRON_TIMEOUT", "0")

        import dotenv

        monkeypatch.setattr(dotenv, "load_dotenv", lambda *_a, **_kw: True)

    @staticmethod
    def _job(job_id="kanban-iso"):
        return {
            "id": job_id, "name": "kanban-iso-job",
            "workdir": None, "schedule_display": "manual",
        }

    def test_agent_runs_as_non_dispatcher(self, monkeypatch, worker_env):
        import cron.scheduler as sched

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        success, *_ = sched.run_job(self._job())
        assert success is True
        assert observed["dispatcher_owned_during_init"] is False
        assert observed["dispatcher_owned_during_run"] is False

    def test_shared_scheduler_fire_path_keeps_cron_agent_isolated(
        self, monkeypatch, worker_env
    ):
        """The ticker/provider path delegates to run_one_job -> run_job."""
        import cron.scheduler as sched

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)
        monkeypatch.setattr(sched, "claim_dispatch", lambda _job_id: True)
        monkeypatch.setattr(sched, "mark_execution_running", lambda _execution_id: None)
        monkeypatch.setattr(sched, "save_job_output", lambda *_args: "/tmp/output")
        monkeypatch.setattr(sched, "_deliver_result", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(sched, "mark_job_run", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(sched, "finish_execution", lambda *_args, **_kwargs: None)

        job = self._job("kanban-iso-scheduler")
        job["execution_id"] = "exec-scheduler"
        job["deliver"] = "local"

        assert sched.run_one_job(job) is True
        assert observed["dispatcher_owned_during_init"] is False
        assert observed["dispatcher_owned_during_run"] is False
        assert os.environ["HERMES_KANBAN_TASK"] == "t_worker_real_task"

    def test_environment_is_left_untouched(self, monkeypatch, worker_env):
        """The whole point of the ContextVar: os.environ must not be mutated, so
        the worker's claim heartbeat and the gateway watchers keep working."""
        import cron.scheduler as sched

        before = {
            k: v for k, v in os.environ.items() if k.startswith("HERMES_KANBAN_")
        }
        assert before, "fixture should have populated kanban env"

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        success, *_ = sched.run_job(self._job())
        assert success is True

        # Untouched DURING the job (the heartbeat thread reads it concurrently)...
        assert observed["kanban_env_during_init"] == before
        # ...and after.
        after = {
            k: v for k, v in os.environ.items() if k.startswith("HERMES_KANBAN_")
        }
        assert after == before

    def test_cron_child_subprocess_env_strips_all_worker_vars(
        self, monkeypatch, worker_env
    ):
        """Terminal/CLI children of the cron agent must not regain identity."""
        from agent.delegation_context import (
            delegated_child_subprocess_env,
            non_dispatcher_owned_context,
        )

        monkeypatch.setenv("HERMES_KANBAN_BRANCH", "worker/branch")
        monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
        monkeypatch.setenv("HERMES_KANBAN_FUTURE_IDENTITY", "must-not-leak")
        monkeypatch.setenv("HERMES_HOME", "/tmp/profile-home")
        monkeypatch.setenv("HERMES_PROFILE", "cron-profile")
        monkeypatch.setenv("HERMES_CRON_SESSION", "1")
        monkeypatch.setenv("TERMINAL_CWD", "/tmp/cron-workdir")

        with non_dispatcher_owned_context():
            child_env = delegated_child_subprocess_env(dict(os.environ))

        assert child_env is not None
        assert not any(key.startswith("HERMES_KANBAN_") for key in child_env)
        assert "HERMES_DELEGATED_CHILD_CONTEXT" not in child_env
        assert child_env["HERMES_HOME"] == "/tmp/profile-home"
        assert child_env["HERMES_PROFILE"] == "cron-profile"
        assert child_env["HERMES_CRON_SESSION"] == "1"
        assert child_env["TERMINAL_CWD"] == "/tmp/cron-workdir"
        assert os.environ["HERMES_KANBAN_TASK"] == "t_worker_real_task"
        assert os.environ["HERMES_KANBAN_BOARD"] == "team-alpha"

    def test_cron_terminal_env_strips_all_worker_vars(self, worker_env):
        from agent.delegation_context import non_dispatcher_owned_context
        from tools.environments.local import _scrub_delegated_child_kanban_env

        env = dict(os.environ)
        with non_dispatcher_owned_context():
            child_env = _scrub_delegated_child_kanban_env(env)

        assert not any(key.startswith("HERMES_KANBAN_") for key in child_env)

    def test_normal_session_subprocess_keeps_explicit_board(self, monkeypatch):
        """Non-worker is not synonymous with child: preserve normal routing."""
        from agent.delegation_context import delegated_child_subprocess_env
        from tools.environments.local import _scrub_delegated_child_kanban_env

        monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
        env = {"PATH": "/usr/bin", "HERMES_KANBAN_BOARD": "selected-board"}

        assert delegated_child_subprocess_env(env) == env
        assert _scrub_delegated_child_kanban_env(env) == env

    def test_context_reset_after_job(self, monkeypatch, worker_env):
        import cron.scheduler as sched
        from agent.delegation_context import is_dispatcher_owned_worker_context

        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        sched.run_job(self._job("kanban-iso-reset"))
        assert is_dispatcher_owned_worker_context() is True

    def test_context_reset_even_when_job_raises(self, monkeypatch, worker_env):
        import cron.scheduler as sched
        from agent.delegation_context import is_dispatcher_owned_worker_context

        class ExplodingAgent:
            def __init__(self, **kwargs):
                pass

            def run_conversation(self, *_a, **_kw):
                raise RuntimeError("boom")

            def get_activity_summary(self):
                return {"seconds_since_activity": 0.0}

        observed: dict = {}
        self._install_stubs(monkeypatch, observed, agent_cls=ExplodingAgent)

        success, *_ = sched.run_job(self._job("kanban-iso-fail"))
        assert success is False
        assert is_dispatcher_owned_worker_context() is True
        # And the env survived the failure too.
        assert os.environ.get("HERMES_KANBAN_BOARD") == "team-alpha"

    def test_concurrent_jobs_do_not_corrupt_worker_identity(
        self, monkeypatch, worker_env
    ):
        """Two workdir-less jobs run concurrently on the parallel pool and take a
        SHARED read lock, so they interleave. With an os.environ snapshot/clear/
        restore this permanently destroyed the worker's identity; a ContextVar is
        per-thread and cannot."""
        import cron.scheduler as sched

        before = {
            k: v for k, v in os.environ.items() if k.startswith("HERMES_KANBAN_")
        }
        observed: dict = {}
        self._install_stubs(monkeypatch, observed)

        results = {}

        def run(name):
            ok, *_ = sched.run_job(self._job(f"kanban-iso-{name}"))
            results[name] = ok

        threads = [threading.Thread(target=run, args=(n,)) for n in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(60)

        assert results == {"a": True, "b": True}
        after = {
            k: v for k, v in os.environ.items() if k.startswith("HERMES_KANBAN_")
        }
        assert after == before, "worker identity must survive concurrent cron jobs"
