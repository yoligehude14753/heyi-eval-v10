"""PR#35: extend PR#33 deploy-repair to catch late-crash vLLM failures.

Background
----------
PR#33 wired auto-repair into DEPLOY's _try_once, but the probe was a
single 0.5 s sleep + container.status check. Real vLLM 0.11.0 crashes
on freshly-released model architectures take 5-15 s to surface (the
container imports torch, loads pydantic, validates the model config,
THEN raises). With a 0.5 s probe the orchestrator declared DEPLOY OK
and the crash leaked into READY_WAIT, where no repair loop exists.

We observed this live on nv8 with:
  - zai-org/GLM-OCR: vLLM rejected ``model_type=glm_ocr`` (~5 s)
  - unsloth/Qwen3.6-27B-GGUF: vLLM rejected ``--model=<dir>`` for
    GGUF, demanding a single .gguf file path (~8 s)

Both crashes are exactly the kind PR#33's strategies SHOULD have
fixed (swap_vllm_image_latest + new strategy_use_gguf_file_path)
if they'd been visible inside the repair loop.

This file tests:
  1. _try_once polls container.status across an early-crash window
     of n_steps * poll_step_s before declaring success.
  2. A container that stays "running" through all polls is reported
     ok (regression — no over-tightening).
  3. A container that goes "exited" mid-poll is reported as
     early_exit with tail logs (the repair-visible failure shape).
  4. DeployFailure.classify() recognises the GGUF "needs file path"
     log signature.
  5. strategy_use_gguf_file_path adds model_path_override to the
     vllm_args, suppressed from CLI translation, used as --model.
  6. propose_attempts() places use_gguf_file_path right after
     trust_remote_code (cheap; no rebuild).
  7. The full ``execute_deploy`` repair loop, given a GGUF failure
     followed by a healthy retry, picks the use_gguf_file_path
     strategy and records it as the winner.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestrator import deploy_repair, stages_py  # noqa: E402

# Reuse the helpers from the existing PR#33 deploy tests.
from tests.test_stages_py_deploy import (  # noqa: E402
    _NOP_SLEEP,
    _fake_docker_client,
    _FakeContainer,
    _make_cfg,
    _make_model_cache,
    _make_run,
    _patch_docker,
    _write_engine_plan,
)

# Sample real log lines captured from nv8 (truncated for compactness).
GGUF_LOG_TAIL = (
    "INFO 05-23 09:19:41 [api_server.py:1839] vLLM API server version 0.11.0\n"
    "ValidationError: 1 validation error for ModelConfig\n"
    "  Value error, Invalid repository ID or local directory specified: '/model'.\n"
    "Please verify the following requirements:\n"
    "1. Provide a valid Hugging Face repository ID.\n"
    "2. Specify a local directory that contains a recognized configuration file.\n"
    "   - For Hugging Face models: ensure the presence of a 'config.json'.\n"
    "3. For GGUF: pass the local path of the GGUF checkpoint.\n"
    "   Loading GGUF from a remote repo directly is not yet supported.\n"
)


# ── (1)+(2)+(3) early-crash polling ─────────────────────────────────


class EarlyCrashPollingTests(unittest.TestCase):
    """The polling loop must catch crashes that surface a few seconds
    after docker.run, while still letting healthy containers through."""

    def test_container_stays_running_returns_ok(self):
        """Regression: nothing changes for the happy path."""
        with TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td))
            cfg.deploy_early_crash_window_s = 1.0
            run = _make_run()
            _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {}})
            _make_model_cache(cfg, run)
            cname = stages_py.container_name_for(run.run_id, "vllm")
            client = _fake_docker_client()
            client.containers.run.return_value = _FakeContainer(
                cname, status="running"
            )
            with _patch_docker(client):
                r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP,
                                             enable_repair=False)
            self.assertTrue(r.ok, msg=r.error)

    def test_container_dies_during_poll_triggers_repair(self):
        """The crash MUST become a repair-visible early_exit."""

        class _DyingContainer(_FakeContainer):
            """Reports 'running' for the first 2 reloads, then 'exited'."""
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._reload_count = 0

            def reload(self) -> None:
                self._reload_count += 1
                if self._reload_count >= 2:
                    self.status = "exited"

        with TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td))
            cfg.deploy_early_crash_window_s = 2.0
            run = _make_run()
            _write_engine_plan(cfg, run, {
                "engine": "vllm",
                "vllm_args": {},
                "engine_image": "vllm/vllm-openai:v0.11.0",
            })
            _make_model_cache(cfg, run)
            cname = stages_py.container_name_for(run.run_id, "vllm")
            client = _fake_docker_client()
            dying = _DyingContainer(
                cname, status="running",
                logs_text=GGUF_LOG_TAIL.encode("utf-8"),
            )
            client.containers.run.return_value = dying

            with _patch_docker(client):
                r = stages_py.execute_deploy(
                    run, cfg, sleep=_NOP_SLEEP, enable_repair=False,
                )

            # With enable_repair=False the failure is surfaced as-is.
            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "early_exit")
            self.assertIn("exited within early-crash window", r.error or "")


# ── (4) GGUF classification ─────────────────────────────────────────


class GGUFClassifyTests(unittest.TestCase):

    def test_real_log_classified_as_gguf_needs_file_path(self):
        f = deploy_repair.DeployFailure(
            engine="vllm", image="vllm/vllm-openai:v0.11.0",
            error_kind="early_exit", logs=GGUF_LOG_TAIL,
        )
        self.assertEqual(f.classify(), "gguf_needs_file_path")

    def test_gguf_check_runs_before_model_type_unknown(self):
        """The GGUF log ALSO contains "Invalid repository ID or local
        directory" which is similar to the unknown-model-type pattern.
        Our classifier must hit gguf_needs_file_path first."""
        f = deploy_repair.DeployFailure(
            engine="vllm", image="vllm/vllm-openai:v0.11.0",
            error_kind="early_exit",
            logs=(
                "Unrecognized configuration class foo\n"
                "3. For GGUF: pass the local path of the GGUF checkpoint.\n"
            ),
        )
        # gguf check is first → wins over model_type_unknown.
        self.assertEqual(f.classify(), "gguf_needs_file_path")

    def test_non_gguf_unknown_model_unaffected(self):
        f = deploy_repair.DeployFailure(
            engine="vllm", image="vllm/vllm-openai:v0.11.0",
            error_kind="early_exit",
            logs=(
                "ValueError: model type `glm_ocr` but Transformers does not "
                "recognize this architecture"
            ),
        )
        self.assertEqual(f.classify(), "model_type_unknown")


# ── (5) strategy_use_gguf_file_path ─────────────────────────────────


class GgufStrategyTests(unittest.TestCase):

    def test_only_applies_to_gguf_failures(self):
        f = deploy_repair.DeployFailure(
            engine="vllm", image="v",
            error_kind="early_exit", logs="something else entirely",
        )
        r = deploy_repair.strategy_use_gguf_file_path({"engine": "vllm"}, f)
        self.assertIsNone(r.new_plan)
        self.assertIn("does not match", r.notes)

    def test_only_applies_to_vllm(self):
        f = deploy_repair.DeployFailure(
            engine="sglang", image="v",
            error_kind="early_exit", logs=GGUF_LOG_TAIL,
        )
        r = deploy_repair.strategy_use_gguf_file_path({"engine": "sglang"}, f)
        self.assertIsNone(r.new_plan)
        self.assertIn("not a vllm", r.notes)

    def test_uses_hint_if_provided(self):
        f = deploy_repair.DeployFailure(
            engine="vllm", image="v",
            error_kind="early_exit", logs=GGUF_LOG_TAIL,
        )
        plan = {
            "engine": "vllm",
            "vllm_args": {"max_model_len": 4096},
            "_gguf_filename_hint": "Qwen3.6-27B-Q4_K_M.gguf",
        }
        r = deploy_repair.strategy_use_gguf_file_path(plan, f)
        self.assertIsNotNone(r.new_plan)
        assert r.new_plan is not None
        self.assertEqual(
            r.new_plan["vllm_args"]["model_path_override"],
            "/model/Qwen3.6-27B-Q4_K_M.gguf",
        )
        # Original plan untouched
        self.assertNotIn("model_path_override", plan["vllm_args"])

    def test_falls_back_to_q4_k_m_glob_without_hint(self):
        f = deploy_repair.DeployFailure(
            engine="vllm", image="v",
            error_kind="early_exit", logs=GGUF_LOG_TAIL,
        )
        plan = {"engine": "vllm", "vllm_args": {}}
        r = deploy_repair.strategy_use_gguf_file_path(plan, f)
        assert r.new_plan is not None
        self.assertIn(
            "Q4_K_M",
            r.new_plan["vllm_args"]["model_path_override"],
        )


# ── (5b) _vllm_command respects model_path_override ─────────────────


class VllmCommandOverrideTests(unittest.TestCase):

    def test_override_replaces_model_flag(self):
        cmd = stages_py._vllm_command(
            "/model",
            {"max_model_len": 4096,
             "model_path_override": "/model/foo.Q4_K_M.gguf"},
            port=18200,
        )
        self.assertIn("/model/foo.Q4_K_M.gguf", cmd)
        # the default '--model /model' should be gone
        idx = cmd.index("--model")
        self.assertEqual(cmd[idx + 1], "/model/foo.Q4_K_M.gguf")
        # override key must NOT leak as a CLI flag
        self.assertNotIn("--model-path-override", cmd)

    def test_without_override_unchanged(self):
        cmd = stages_py._vllm_command(
            "/model", {"max_model_len": 4096}, port=18200,
        )
        idx = cmd.index("--model")
        self.assertEqual(cmd[idx + 1], "/model")
        self.assertIn("--max-model-len", cmd)


# ── (6) ordering in BUILTIN_STRATEGIES ──────────────────────────────


class StrategyOrderingTests(unittest.TestCase):

    def test_gguf_strategy_runs_before_image_swap(self):
        """For a GGUF failure, use_gguf_file_path should appear in
        propose_attempts() output BEFORE strategy_swap_engine_*."""
        f = deploy_repair.DeployFailure(
            engine="vllm", image="vllm/vllm-openai:v0.11.0",
            error_kind="early_exit", logs=GGUF_LOG_TAIL,
        )
        plan = {
            "engine": "vllm",
            "vllm_args": {},
            "_gguf_filename_hint": "x.Q4_K_M.gguf",
        }
        attempts = deploy_repair.propose_attempts(plan, f)
        names = [a.name for a in attempts]
        self.assertIn("use_gguf_file_path", names,
                      f"GGUF strategy must be applicable, got {names}")
        # use_gguf_file_path comes before swap_engine_transformers.
        self.assertLess(
            names.index("use_gguf_file_path"),
            names.index("swap_engine_transformers")
            if "swap_engine_transformers" in names else 9999,
        )


# ── (7) end-to-end execute_deploy with repair ───────────────────────


class GgufEndToEndRepairTests(unittest.TestCase):
    """Boot a real-shape repair loop: first attempt crashes with the
    GGUF log, second attempt (after strategy mutation) stays running.
    The winning strategy must be use_gguf_file_path."""

    def test_gguf_failure_then_fixed_attempt_records_winner(self):

        class _Dying(_FakeContainer):
            def __init__(self, *a, **kw):
                super().__init__(*a, **kw)
                self._n = 0

            def reload(self) -> None:
                self._n += 1
                if self._n >= 2:
                    self.status = "exited"

        with TemporaryDirectory() as td:
            cfg = _make_cfg(Path(td))
            cfg.deploy_early_crash_window_s = 2.0
            cfg.deploy_repair_agent_enabled = False  # rule-based only
            run = _make_run()
            _write_engine_plan(cfg, run, {
                "engine": "vllm",
                "engine_image": "vllm/vllm-openai:v0.11.0",
                "vllm_args": {"max_model_len": 8192},
            })
            cache = _make_model_cache(cfg, run)
            (cache / "Qwen3.6-27B-Q4_K_M.gguf").write_bytes(b"\x00" * 256)

            cname = stages_py.container_name_for(run.run_id, "vllm")
            dying = _Dying(
                cname, status="running",
                logs_text=GGUF_LOG_TAIL.encode("utf-8"),
            )
            healthy = _FakeContainer(cname, status="running")
            client = _fake_docker_client()
            # First call -> dies; subsequent calls -> healthy.
            client.containers.run.side_effect = [dying, healthy, healthy]

            with _patch_docker(client):
                r = stages_py.execute_deploy(
                    run, cfg, sleep=_NOP_SLEEP, enable_repair=True,
                )

            self.assertTrue(r.ok,
                            msg=f"repair should succeed, got {r.error}")

            repair_log = json.loads(
                (cfg.run_dir(run.run_id) / "_meta" / "deploy_repair.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(
                repair_log["winning_strategy"], "use_gguf_file_path"
            )
            self.assertEqual(
                repair_log["failure_class"], "gguf_needs_file_path"
            )

            # The actual command that finally succeeded should reference
            # the .gguf FILE, not the directory.
            self.assertGreaterEqual(client.containers.run.call_count, 2)
            second_call_kwargs = client.containers.run.call_args_list[1].kwargs
            cmd = second_call_kwargs["command"]
            model_arg = cmd[cmd.index("--model") + 1]
            self.assertTrue(model_arg.endswith(".gguf"),
                            f"second attempt should --model a gguf file, got {model_arg}")
            self.assertIn("Q4_K_M", model_arg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
