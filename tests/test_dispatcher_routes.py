"""Unit tests for orchestrator/stages.execute_stage routing.

Pins that DEPLOY / READY_WAIT / CAPABILITY / CLEANUP route to the v10
native Python executors, and SHOWCASE still routes to cc-agent until
PR#5 restricts the latter.

Test IDs map to docs/PR4_TEST_PLAN.md §Dispatcher.
"""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

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
            with (
                patch.object(stages_py, "execute_deploy", return_value=_ok_result()) as native,
                patch.object(stages, "_docker_run_cc_agent") as cc,
            ):
                r = stages.execute_stage(run, StageName.DEPLOY, cfg, store)
            self.assertTrue(r.ok)
            native.assert_called_once_with(run, cfg)
            cc.assert_not_called()

    def test_d2_ready_wait_routes_to_stages_py(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            store = MagicMock()
            with (
                patch.object(stages_py, "execute_ready_wait",
                             return_value=_ok_result()) as native,
                patch.object(stages, "_docker_run_cc_agent") as cc,
            ):
                stages.execute_stage(run, StageName.READY_WAIT, cfg, store)
            native.assert_called_once_with(run, cfg)
            cc.assert_not_called()

    def test_d3_capability_routes_to_capability_module(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            store = MagicMock()
            with (
                patch.object(cap_mod, "execute_capability",
                             return_value=_ok_cap_result()) as native,
                patch.object(stages, "_docker_run_cc_agent") as cc,
            ):
                stages.execute_stage(run, StageName.CAPABILITY, cfg, store)
            native.assert_called_once_with(run, cfg)
            cc.assert_not_called()

    def test_d4_cleanup_routes_to_stages_py(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            store = MagicMock()
            with (
                patch.object(stages_py, "execute_cleanup",
                             return_value=_ok_result()) as native,
                patch.object(stages, "_docker_run_cc_agent") as cc,
            ):
                stages.execute_stage(run, StageName.CLEANUP, cfg, store)
            native.assert_called_once_with(run, cfg)
            cc.assert_not_called()

    def test_d5_showcase_still_routes_to_cc_until_pr5(self):
        """v10-Until-PR5 contract: SHOWCASE is the last cc-agent stage.
        Once PR#5 lands the restricted runner, this test flips to assert
        the new restricted-runner module is called instead."""
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            store = MagicMock()
            with (
                patch.object(stages, "_docker_run_cc_agent",
                             return_value=(0, "e8-cc-showcase-xyz")) as cc,
                patch.object(stages, "validate_showcase",
                             return_value={"items": [], "summary": "ok"}),
            ):
                stages.execute_stage(run, StageName.SHOWCASE, cfg, store)
            cc.assert_called_once()
            # Confirm the stage argument is SHOWCASE so the spawn isn't
            # accidentally happening for some other stage by mistake.
            call_args = cc.call_args
            # _docker_run_cc_agent(run, stage, cfg) signature
            self.assertEqual(call_args.args[1], StageName.SHOWCASE)


if __name__ == "__main__":
    unittest.main()
