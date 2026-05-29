"""Unit tests for orchestrator/stages.execute_stage routing.

After PR#7a there is no docker-spawn path left in the dispatcher: every
stage routes to a native Python executor. This file pins the routes for
DEPLOY / READY_WAIT / CAPABILITY / CLEANUP / SHOWCASE.

Test IDs map to docs/PR4_TEST_PLAN.md §Dispatcher.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from cc_agent import showcase_runner as sc_mod
from orchestrator import capability as cap_mod
from orchestrator import stages, stages_py
from orchestrator.config import OrchestratorConfig
from orchestrator.state_machine import Run, StageName


def _make_cfg(tmp: Path) -> OrchestratorConfig:
    cfg = OrchestratorConfig(data_root=tmp / "data", repo_root=tmp / "repo")
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _make_run() -> Run:
    return Run(run_id="dispatcher_run", hf_id="Qwen/Qwen2.5-0.5B-Instruct")


def _ok_result(**extra) -> stages_py.StageResult:
    return stages_py.StageResult(
        ok=True, duration_s=0.01, artifacts=["x.json"], rc=0, **extra,
    )


def _ok_cap_result(**extra) -> cap_mod.StageResult:
    return cap_mod.StageResult(
        ok=True, duration_s=0.01, artifacts=["capability.json"], rc=0, **extra,
    )


class DispatcherTests(unittest.TestCase):

    def test_d1_deploy_routes_to_stages_py(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            store = MagicMock()
            with patch.object(stages_py, "execute_deploy",
                              return_value=_ok_result()) as native:
                r = stages.execute_stage(run, StageName.DEPLOY, cfg, store)
            self.assertTrue(r.ok)
            native.assert_called_once_with(run, cfg)

    def test_d2_ready_wait_routes_to_stages_py(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            store = MagicMock()
            with patch.object(stages_py, "execute_ready_wait",
                              return_value=_ok_result()) as native:
                stages.execute_stage(run, StageName.READY_WAIT, cfg, store)
            native.assert_called_once_with(run, cfg)

    def test_d3_capability_routes_to_capability_module(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            store = MagicMock()
            with patch.object(cap_mod, "execute_capability",
                              return_value=_ok_cap_result()) as native:
                stages.execute_stage(run, StageName.CAPABILITY, cfg, store)
            # CAPABILITY now passes the multimodal judges so image_gen /
            # video_gen are actually scored (INV-14).
            from orchestrator import llm_judge as _judge
            native.assert_called_once_with(
                run, cfg,
                judge_image=_judge.judge_image,
                judge_video_first_frame=_judge.judge_video_first_frame,
            )

    def test_d4_cleanup_routes_to_stages_py(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            store = MagicMock()
            with patch.object(stages_py, "execute_cleanup",
                              return_value=_ok_result()) as native:
                stages.execute_stage(run, StageName.CLEANUP, cfg, store)
            native.assert_called_once_with(run, cfg)

    def test_d5_showcase_routes_to_showcase_runner(self):
        """SHOWCASE routes to the in-process cc_agent.showcase_runner.
        After PR#7a the dispatcher has no docker-spawn fallback to assert
        against — that surface is gone entirely."""
        sc_ok = sc_mod.ShowcaseResult(
            ok=True, duration_s=0.01,
            artifacts=["showcase.json"], rc=0,
        )
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            store = MagicMock()
            with patch.object(sc_mod, "execute_showcase",
                              return_value=sc_ok) as native:
                r = stages.execute_stage(run, StageName.SHOWCASE, cfg, store)
            self.assertTrue(r.ok)
            native.assert_called_once_with(run, cfg)

    def test_d6_dispatcher_has_no_docker_spawn_helper(self):
        """INV-11 belt-and-suspenders: the dispatcher module must not
        expose any cc-agent docker spawn helper after PR#7a. Anyone
        re-adding a docker-spawn path will trip this guard."""
        forbidden = (
            "_docker_run_cc_agent",
            "_execute_cc_stage",
            "_CC_STAGES",
            "_cc_agent_container_name",
        )
        for name in forbidden:
            self.assertFalse(
                hasattr(stages, name),
                f"orchestrator/stages.py grew back {name!r} — that path was "
                f"deleted in PR#7a; the dispatcher must stay docker-spawn-free.",
            )


if __name__ == "__main__":
    unittest.main()
