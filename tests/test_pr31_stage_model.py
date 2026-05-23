"""PR#31: STAGE_MODEL pipeline stage + model_stager unit tests.

Closes the "every fresh discovered run aborts at DEPLOY with
'model not in eval-cache'" loop that PR#30's auto-discovery surfaced
on nv8. STAGE_MODEL now runs immediately after ENGINE_SELECT and
synchronously downloads the weights (snapshot_download via the HF Hub
mirror) so DEPLOY's bind-mount is guaranteed to find them.

What we have to test:

  1. Pure ``model_stager.ensure_model_staged`` contract:
       - already-staged dir → skipped, no downloader call
       - oversize from engine plan → INV-23 inheritance, no downloader
       - disk-headroom gate denies → graceful skip, no downloader
       - downloader exception → hard fail (not graceful skip)
       - downloader returns ok but no weight files → hard fail
       - happy path → downloader called once, provenance written
  2. State machine wiring:
       - STAGE_MODEL is in STAGES_IN_ORDER, between ENGINE_SELECT and
         DEPLOY (positional invariant)
       - STAGE_MODEL is in RUN_LEVEL_STAGES (full restart eligible)
  3. ``stages.execute_stage(STAGE_MODEL, ...)`` integration:
       - reads engine.json + metadata.json
       - delegates to model_stager with the right target_dir
       - on graceful skip, returns extra.aborted=True so main.run_pipeline
         marks the run ABORTED, not FAILED (the PR#11 contract)
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from orchestrator import model_stager, stages
from orchestrator.config import OrchestratorConfig
from orchestrator.state_machine import (
    RUN_LEVEL_STAGES,
    STAGES_IN_ORDER,
    Run,
    StageName,
)


# ── helpers ─────────────────────────────────────────────────────────────────


def _stage_complete(target_dir: Path, weight_name: str = "model.safetensors",
                    weight_bytes: bytes = b"\x00" * 32) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "config.json").write_text("{}", encoding="utf-8")
    (target_dir / weight_name).write_bytes(weight_bytes)


# ── 1. state machine invariants ─────────────────────────────────────────────


class TestStateMachineWiring(unittest.TestCase):

    def test_stage_model_present(self) -> None:
        self.assertIn(StageName.STAGE_MODEL, STAGES_IN_ORDER)

    def test_stage_model_between_engine_select_and_deploy(self) -> None:
        # Positional invariant: STAGE_MODEL MUST run after we've decided
        # the engine (so we know if it's oversize) and before DEPLOY
        # tries to bind-mount the weights.
        i_es = STAGES_IN_ORDER.index(StageName.ENGINE_SELECT)
        i_sm = STAGES_IN_ORDER.index(StageName.STAGE_MODEL)
        i_dp = STAGES_IN_ORDER.index(StageName.DEPLOY)
        self.assertLess(i_es, i_sm)
        self.assertLess(i_sm, i_dp)
        self.assertEqual(i_sm, i_es + 1, "STAGE_MODEL must directly follow ENGINE_SELECT")
        self.assertEqual(i_dp, i_sm + 1, "DEPLOY must directly follow STAGE_MODEL")

    def test_stage_model_is_run_level(self) -> None:
        # Full restartable: idempotent on already-staged dirs.
        self.assertIn(StageName.STAGE_MODEL, RUN_LEVEL_STAGES)


# ── 2. model_stager.ensure_model_staged ────────────────────────────────────


class TestEnsureModelStaged(unittest.TestCase):
    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.root = Path(self._td.name)
        self.target = self.root / "Qwen2.5-0.5B-Instruct"

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_already_staged_skips_download(self) -> None:
        _stage_complete(self.target)
        dl = mock.MagicMock()
        r = model_stager.ensure_model_staged(
            hf_id="Qwen/Qwen2.5-0.5B-Instruct",
            target_dir=self.target,
            metadata={"param_count": "0.5B"},
            downloader=dl,
        )
        self.assertTrue(r.ok)
        self.assertFalse(r.skipped)
        self.assertTrue((r.extra or {}).get("already_staged"))
        dl.assert_not_called()
        self.assertGreater(r.bytes_on_disk, 0)

    def test_oversize_engine_plan_inherits_skip(self) -> None:
        # ENGINE_SELECT already marked the run oversize → no download.
        dl = mock.MagicMock()
        r = model_stager.ensure_model_staged(
            hf_id="meta-llama/Llama-3.1-405B-Instruct",
            target_dir=self.target,
            metadata={"param_count": "405B"},
            engine_plan={"engine": "metadata_only", "oversize": True},
            downloader=dl,
        )
        self.assertFalse(r.ok)
        self.assertTrue(r.skipped)
        self.assertEqual(r.error_kind, model_stager.GRACEFUL_OVERSIZE_INHERITED)
        dl.assert_not_called()

    def test_disk_headroom_gate_denies_huge_download(self) -> None:
        # Pretend the model needs 100 TB; harness has GB-scale free.
        dl = mock.MagicMock()
        r = model_stager.ensure_model_staged(
            hf_id="some/huge",
            target_dir=self.target,
            metadata={"hf_info": {"usedStorage": 100 * 1024**4}},  # 100 TiB
            downloader=dl,
        )
        self.assertFalse(r.ok)
        self.assertTrue(r.skipped)
        self.assertEqual(r.error_kind, model_stager.GRACEFUL_DISK_FULL)
        dl.assert_not_called()
        self.assertIn("free=", r.skipped_reason or "")
        self.assertIn("need", r.skipped_reason or "")

    def test_download_exception_hard_fails_not_graceful(self) -> None:
        # Hard fail because the orchestrator's normal retry path is the
        # right home for transient network failures, AND the Panel
        # should surface this run as red (failed), not silently
        # aborted (the PR#30 symptom we are eradicating).
        def boom(*, repo_id: str, local_dir: str,
                 max_workers: int, allow_patterns: list[str] | None) -> str:
            raise ConnectionError("hub unreachable")

        r = model_stager.ensure_model_staged(
            hf_id="some/repo",
            target_dir=self.target,
            metadata={"param_count": "0.5B"},
            downloader=boom,
        )
        self.assertFalse(r.ok)
        self.assertFalse(r.skipped, "transient network failures are NOT skips")
        self.assertEqual(r.error_kind, "download_failed")
        self.assertIn("ConnectionError", r.error or "")

    def test_download_returns_ok_but_no_weights_hard_fails(self) -> None:
        # snapshot_download "succeeds" but the dir lacks config.json or
        # any weight file (e.g. someone passes allow_patterns="*.txt").
        def empty(*, repo_id: str, local_dir: str,
                  max_workers: int, allow_patterns: list[str] | None) -> str:
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            return local_dir

        r = model_stager.ensure_model_staged(
            hf_id="some/repo",
            target_dir=self.target,
            metadata={"param_count": "0.5B"},
            downloader=empty,
        )
        self.assertFalse(r.ok)
        self.assertFalse(r.skipped)
        self.assertEqual(r.error_kind, "incomplete_after_download")

    def test_happy_path_invokes_downloader_with_correct_args(self) -> None:
        captured: dict[str, object] = {}

        def fake(*, repo_id: str, local_dir: str,
                 max_workers: int, allow_patterns: list[str] | None) -> str:
            captured["repo_id"] = repo_id
            captured["local_dir"] = local_dir
            captured["max_workers"] = max_workers
            captured["allow_patterns"] = allow_patterns
            # Simulate snapshot_download writing weights.
            _stage_complete(Path(local_dir))
            return local_dir

        r = model_stager.ensure_model_staged(
            hf_id="Qwen/Qwen2.5-0.5B-Instruct",
            target_dir=self.target,
            metadata={"param_count": "0.5B",
                      "hf_info": {"usedStorage": 1 * 1024 * 1024 * 1024}},  # 1GB
            hf_endpoint="https://hf-mirror.com",
            downloader=fake,
        )
        self.assertTrue(r.ok, f"unexpected: {r}")
        self.assertFalse(r.skipped)
        self.assertEqual(captured["repo_id"], "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertEqual(captured["local_dir"], str(self.target))
        # Default value path (we didn't override).
        self.assertEqual(captured["max_workers"], 8)
        self.assertGreater(r.bytes_on_disk, 0)
        self.assertGreaterEqual(r.files, 2)  # config + weight

    def test_hf_endpoint_env_var_set_for_child(self) -> None:
        # Setting HF_ENDPOINT into the process env so child threads in
        # huggingface_hub see the mirror.
        seen_env = []

        def fake(*, repo_id: str, local_dir: str,
                 max_workers: int, allow_patterns: list[str] | None) -> str:
            import os
            seen_env.append(os.environ.get("HF_ENDPOINT"))
            _stage_complete(Path(local_dir))
            return local_dir

        model_stager.ensure_model_staged(
            hf_id="x/y", target_dir=self.target,
            metadata={"param_count": "0.5B"},
            hf_endpoint="https://my.mirror.test",
            downloader=fake,
        )
        self.assertEqual(seen_env, ["https://my.mirror.test"])


class TestSizeEstimator(unittest.TestCase):
    def test_used_storage_wins(self) -> None:
        n = model_stager._estimate_size_bytes(
            {"hf_info": {"usedStorage": 12345}, "param_count": "7B"})
        self.assertEqual(n, 12345)

    def test_safetensors_total_fallback(self) -> None:
        n = model_stager._estimate_size_bytes(
            {"hf_info": {"safetensors": {"total": 999}}})
        self.assertEqual(n, 999)

    def test_param_count_fallback_fp16(self) -> None:
        # 7B params * 2 bytes ≈ 14 GB.
        n = model_stager._estimate_size_bytes({"param_count": "7B"})
        self.assertIsNotNone(n)
        assert n is not None
        self.assertEqual(n, int(7e9 * 2))

    def test_hf_id_fallback_when_param_count_missing(self) -> None:
        n = model_stager._estimate_size_bytes(
            {"hf_id": "meta-llama/Llama-3.1-405B-Instruct"})
        self.assertEqual(n, int(405e9 * 2))

    def test_unknown_returns_none(self) -> None:
        n = model_stager._estimate_size_bytes({"hf_id": "anon/no-name"})
        self.assertIsNone(n)


# ── 3. stages.execute_stage(STAGE_MODEL) integration ───────────────────────


class TestStageModelDispatcher(unittest.TestCase):
    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        root = Path(self._td.name)
        self.cfg = OrchestratorConfig(
            data_root=root / "data",
            model_cache_root=root / "models",
        )
        self.run = Run(run_id="ut-pr31-001", hf_id="Qwen/Qwen2.5-0.5B-Instruct")
        rd = self.cfg.run_dir(self.run.run_id)
        meta = rd / "_meta"
        meta.mkdir(parents=True, exist_ok=True)
        # Plausible engine + metadata for the executor to read.
        (meta / "engine.json").write_text(json.dumps({
            "stage": "ENGINE_SELECT", "engine": "vllm", "oversize": False,
            "vllm_args": {"tensor_parallel_size": 1},
        }), encoding="utf-8")
        (meta / "metadata.json").write_text(json.dumps({
            "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "param_count": "0.5B",
        }), encoding="utf-8")

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_happy_path_writes_provenance_and_returns_ok(self) -> None:
        # Patch model_stager so we never touch the real Hub.
        from orchestrator import model_stager as mod
        fake = mock.patch.object(
            mod, "ensure_model_staged",
            return_value=mod.StageModelResult(
                ok=True, bytes_on_disk=999, files=2,
                target_dir=str(self.cfg.model_cache_root / "Qwen2.5-0.5B-Instruct"),
                extra={"already_staged": False},
            ),
        )
        with fake:
            r = stages.execute_stage(
                self.run, StageName.STAGE_MODEL, self.cfg, store=None,  # type: ignore[arg-type]
            )
        self.assertTrue(r.ok, f"got: {r}")
        rd = self.cfg.run_dir(self.run.run_id)
        self.assertTrue((rd / "_meta" / "stage_model.json").exists())
        prov = json.loads((rd / "_meta" / "stage_model.json").read_text())
        self.assertTrue(prov["ok"])
        self.assertEqual(prov["bytes_on_disk"], 999)

    def test_graceful_skip_returns_extra_aborted_true(self) -> None:
        # main.run_pipeline expects extra.aborted=True to convert a
        # not-ok StageResult into ABORTED-not-FAILED (PR#11 contract).
        from orchestrator import model_stager as mod
        fake = mock.patch.object(
            mod, "ensure_model_staged",
            return_value=mod.StageModelResult(
                ok=False, skipped=True,
                skipped_reason="disk too full",
                error_kind=mod.GRACEFUL_DISK_FULL,
                target_dir="/whatever",
            ),
        )
        with fake:
            r = stages.execute_stage(
                self.run, StageName.STAGE_MODEL, self.cfg, store=None,  # type: ignore[arg-type]
            )
        self.assertFalse(r.ok)
        self.assertEqual(r.error_kind, mod.GRACEFUL_DISK_FULL)
        self.assertIsNotNone(r.extra)
        assert r.extra is not None
        self.assertTrue(r.extra.get("aborted"))
        self.assertIn("disk too full", r.extra.get("reason") or "")

    def test_hard_failure_does_NOT_mark_aborted(self) -> None:
        # main.run_pipeline expects a missing extra.aborted to convert
        # the not-ok StageResult into FAILED (retry-worthy) — that's
        # the right home for a transient download exception.
        from orchestrator import model_stager as mod
        fake = mock.patch.object(
            mod, "ensure_model_staged",
            return_value=mod.StageModelResult(
                ok=False, skipped=False,
                error="ConnectionError: hub down",
                error_kind="download_failed",
                target_dir="/whatever",
            ),
        )
        with fake:
            r = stages.execute_stage(
                self.run, StageName.STAGE_MODEL, self.cfg, store=None,  # type: ignore[arg-type]
            )
        self.assertFalse(r.ok)
        self.assertEqual(r.error_kind, "download_failed")
        # extra is empty or missing 'aborted' — main.run_pipeline routes
        # to FAILED branch (retry).
        self.assertFalse((r.extra or {}).get("aborted", False))


if __name__ == "__main__":
    unittest.main()
