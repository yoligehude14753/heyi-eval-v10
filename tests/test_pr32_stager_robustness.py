"""PR#32 robustness fixes for STAGE_MODEL exposed by the first live
batch on nv8 (5 fresh 2026 enqueues, three of which silently broke
the cache to the tune of 357 GB):

  1. unsloth/Qwen3.6-27B-GGUF: GGUF repo with 8 quantization variants.
     PR#31 estimated 54 GB (param_count × 2), actual download was
     328 GB before a Hub-403 truncated it. PR#32 picks one quantization
     (Q4_K_M) via allow_patterns + uses siblings size estimate, and
     cleans up the partial dir on hard fail.

  2. argmaxinc/whisperkit-coreml: CoreML format (.mlpackage / .mlmodelc
     directories, no .safetensors). PR#31's `_has_weight_file` only
     recognised file-suffix weights, so a fully-downloaded 29 GB
     repo got marked `incomplete_after_download` and dropped on the
     floor. PR#32 also recognises weight DIRECTORIES (mlpackage etc.).

  3. Failed downloads leak 357 GB of partial weights into the cache
     with no automatic recovery. PR#32 calls ``_cleanup_partial`` on
     every hard-fail path AND surfaces ``bytes_freed`` in provenance.
"""
from __future__ import annotations

import unittest
import unittest.mock
from pathlib import Path
from tempfile import TemporaryDirectory

from orchestrator import model_stager


def _make_dir(d: Path, files: dict[str, bytes] | None = None,
              dirs: list[str] | None = None) -> None:
    d.mkdir(parents=True, exist_ok=True)
    for name, data in (files or {}).items():
        (d / name).write_bytes(data)
    for name in (dirs or []):
        (d / name).mkdir(parents=True, exist_ok=True)


# ── 1. _has_weight_file: CoreML / mlpackage / mlmodelc ─────────────────────


class TestHasWeightFile(unittest.TestCase):

    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.d = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_safetensors_plus_config_is_complete(self) -> None:
        _make_dir(self.d, files={"config.json": b"{}",
                                  "model.safetensors": b"\x00" * 16})
        self.assertTrue(model_stager._has_weight_file(self.d))

    def test_coreml_mlpackage_dirs_count_as_weights(self) -> None:
        # PR#32 motivating case: argmaxinc/whisperkit-coreml ships
        # only .mlpackage directories + a config.json. No .safetensors
        # file exists in the repo at all.
        _make_dir(self.d, files={"config.json": b"{}"},
                  dirs=["openai_whisper-tiny.mlpackage",
                        "openai_whisper-base.mlmodelc"])
        # Put a manifest INSIDE the mlpackage to look realistic.
        (self.d / "openai_whisper-tiny.mlpackage" / "Manifest.json"
         ).write_text("{}", encoding="utf-8")
        self.assertTrue(
            model_stager._has_weight_file(self.d),
            "CoreML repos with only .mlpackage/.mlmodelc directories "
            "must be recognised as complete",
        )

    def test_nested_mlmodelc_two_levels_deep(self) -> None:
        # Exact whisperkit-coreml layout from nv8 inspection:
        #   whisperkit-coreml/
        #     config.json
        #     openai_whisper-base/
        #       AudioEncoder.mlmodelc/
        #         analytics/...
        #         weights/...
        # Old _has_weight_file only iterated 1 level → missed the
        # nested .mlmodelc → falsely flagged the 29 GB snapshot
        # as incomplete.
        _make_dir(self.d, files={"config.json": b"{}"})
        nested = self.d / "openai_whisper-base" / "AudioEncoder.mlmodelc"
        nested.mkdir(parents=True)
        (nested / "weights").mkdir()
        (nested / "weights" / "0.bin").write_bytes(b"\x00" * 64)
        self.assertTrue(
            model_stager._has_weight_file(self.d),
            "nested .mlmodelc dirs must be discoverable at depth 2-3"
            " (this is the actual on-nv8 whisperkit-coreml layout)",
        )

    def test_diffusion_pipeline_model_index_counts_as_manifest(self) -> None:
        # diffusers pipelines use model_index.json instead of config.json.
        _make_dir(self.d, files={"model_index.json": b"{}",
                                  "model.safetensors": b"\x00" * 16})
        self.assertTrue(model_stager._has_weight_file(self.d))

    def test_gguf_only_repo_with_no_config_json(self) -> None:
        # Many GGUF repos ship only the .gguf shards + a README,
        # no config.json. The post-download ledger (no .incomplete
        # files) should signal "done" even without a manifest.
        _make_dir(self.d, files={"README.md": b"Q4_K_M ggml",
                                  "model-Q4_K_M.gguf": b"\x00" * 32})
        self.assertTrue(
            model_stager._has_weight_file(self.d),
            "GGUF-only repos without config.json must be recognised "
            "as complete after download",
        )

    def test_incomplete_ledger_disqualifies_even_if_weights_present(self) -> None:
        # Mid-flight download → MUST NOT be considered complete.
        _make_dir(self.d, files={"config.json": b"{}",
                                  "model.safetensors": b"\x00" * 16})
        ledger = self.d / ".cache" / "huggingface" / "download"
        ledger.mkdir(parents=True)
        (ledger / "shard.incomplete").write_bytes(b"\x00")
        self.assertFalse(model_stager._has_weight_file(self.d))

    def test_empty_dir_is_not_complete(self) -> None:
        self.d.mkdir(parents=True, exist_ok=True)
        self.assertFalse(model_stager._has_weight_file(self.d))

    def test_config_only_no_weights_is_not_complete(self) -> None:
        # PR#31's bug: the test e2e harness used to do exactly this
        # (config.json + nothing else). The PR#32 detector still
        # rejects it — the harness was patched separately to also
        # write a fake weight file.
        _make_dir(self.d, files={"config.json": b"{}"})
        self.assertFalse(model_stager._has_weight_file(self.d))


# ── 2. GGUF allow_patterns heuristic ───────────────────────────────────────


class TestGgufAllowPatterns(unittest.TestCase):

    def test_dash_gguf_suffix_triggers_pattern(self) -> None:
        pats = model_stager._compute_allow_patterns(
            "unsloth/Qwen3.6-27B-GGUF", {})
        self.assertIsNotNone(pats)
        assert pats is not None
        # Must include one quantization filter — NOT all .gguf.
        self.assertTrue(any("Q4_K_M" in p for p in pats),
                        f"expected a Q4_K_M filter, got {pats!r}")
        self.assertTrue(any("*.json" in p for p in pats),
                        "should still allow tokenizer / config")

    def test_library_name_gguf_triggers_pattern(self) -> None:
        pats = model_stager._compute_allow_patterns(
            "someone/model-name", {"hf_info": {"library_name": "gguf"}})
        self.assertIsNotNone(pats)

    def test_majority_gguf_siblings_triggers(self) -> None:
        sib = [
            {"rfilename": "model.Q4_K_M.gguf"},
            {"rfilename": "model.Q5_K_M.gguf"},
            {"rfilename": "config.json"},  # 1/3 not gguf, still majority
        ]
        pats = model_stager._compute_allow_patterns(
            "someone/strange",
            {"hf_info": {"siblings": sib}})
        self.assertIsNotNone(pats)

    def test_normal_repo_returns_none(self) -> None:
        # Qwen2.5-0.5B-Instruct etc. must NOT get a pattern, otherwise
        # we'd accidentally drop their .safetensors shards.
        self.assertIsNone(model_stager._compute_allow_patterns(
            "Qwen/Qwen2.5-0.5B-Instruct", {"param_count": "0.5B"}))


# ── 3. Cleanup on hard fail ────────────────────────────────────────────────


class TestCleanupOnFail(unittest.TestCase):

    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.root = Path(self._td.name)
        self.target = self.root / "the-model"

    def tearDown(self) -> None:
        self._td.cleanup()

    def test_download_exception_cleans_partial_and_reports_freed(self) -> None:
        # Simulate a partial download (1 MB of stale shard data) then
        # the downloader raising.
        def boom(*, repo_id, local_dir, max_workers, allow_patterns):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "model-partial.safetensors"
             ).write_bytes(b"\x00" * 1_000_000)
            raise ConnectionError("simulated 403")

        r = model_stager.ensure_model_staged(
            hf_id="x/y", target_dir=self.target,
            metadata={"param_count": "0.5B"},
            downloader=boom,
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.error_kind, "download_failed")
        # Cleanup must have removed the partial 1 MB.
        self.assertGreaterEqual(r.bytes_freed, 1_000_000)
        self.assertFalse(self.target.exists(),
                         "target_dir must be removed after hard fail")

    def test_incomplete_after_download_cleans_too(self) -> None:
        # snapshot_download "succeeds" but the dir lacks any weights
        # (e.g. allow_patterns excluded everything).
        def empty(*, repo_id, local_dir, max_workers, allow_patterns):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "README.md").write_bytes(b"hello")
            return local_dir

        r = model_stager.ensure_model_staged(
            hf_id="x/y", target_dir=self.target,
            metadata={"param_count": "0.5B"},
            downloader=empty,
        )
        self.assertFalse(r.ok)
        self.assertEqual(r.error_kind, "incomplete_after_download")
        self.assertGreater(r.bytes_freed, 0)
        self.assertFalse(self.target.exists())

    def test_happy_path_does_not_cleanup(self) -> None:
        def ok(*, repo_id, local_dir, max_workers, allow_patterns):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "config.json").write_bytes(b"{}")
            (Path(local_dir) / "model.safetensors"
             ).write_bytes(b"\x00" * 32)
            return local_dir

        r = model_stager.ensure_model_staged(
            hf_id="x/y", target_dir=self.target,
            metadata={"param_count": "0.5B"},
            downloader=ok,
        )
        self.assertTrue(r.ok)
        self.assertEqual(r.bytes_freed, 0)
        self.assertTrue(self.target.exists())


# ── 4. size estimator with allow_patterns + siblings ───────────────────────


class TestSizeEstimatorWithSiblings(unittest.TestCase):

    def _qwen_gguf_siblings(self) -> list[dict]:
        # Real-shape siblings list for a GGUF repo with 8 quants.
        return [
            {"rfilename": "Qwen3.6-27B-Q4_K_M.gguf", "size": 16 * 10**9},
            {"rfilename": "Qwen3.6-27B-Q4_0.gguf", "size": 15 * 10**9},
            {"rfilename": "Qwen3.6-27B-Q5_K_M.gguf", "size": 20 * 10**9},
            {"rfilename": "Qwen3.6-27B-Q8_0.gguf", "size": 28 * 10**9},
            {"rfilename": "Qwen3.6-27B-BF16.gguf", "size": 54 * 10**9},
            {"rfilename": "README.md", "size": 1024},
            {"rfilename": "config.json", "size": 512},
        ]

    def test_unfiltered_sums_all_siblings(self) -> None:
        md = {"hf_info": {"siblings": self._qwen_gguf_siblings()}}
        est = model_stager._estimate_size_bytes(md)
        self.assertGreater(est or 0, 100 * 10**9,
                           "unfiltered estimate must reflect all 5 GGUF shards")

    def test_q4km_pattern_only_counts_one_shard(self) -> None:
        md = {"hf_info": {"siblings": self._qwen_gguf_siblings()}}
        est = model_stager._estimate_size_bytes(
            md, allow_patterns=["*Q4_K_M*.gguf", "*.json"])
        # Should be ~16 GB + a few hundred bytes of json.
        self.assertGreater(est or 0, 15 * 10**9)
        self.assertLess(est or 0, 17 * 10**9)

    def test_usedStorage_ignored_when_allow_patterns_active(self) -> None:
        # usedStorage = 328 GB (whole repo) but allow_patterns will
        # narrow it. Trusting usedStorage would over-estimate by 20×
        # and block the download via disk-headroom-gate.
        md = {"hf_info": {
            "usedStorage": 328 * 10**9,
            "siblings": self._qwen_gguf_siblings(),
        }}
        est = model_stager._estimate_size_bytes(
            md, allow_patterns=["*Q4_K_M*.gguf"])
        self.assertLess(est or 0, 20 * 10**9,
                        "usedStorage must be ignored when allow_patterns filters")


# ── 5. End-to-end: GGUF download is narrow + sized correctly ───────────────


class TestEnsureStagedGgufNarrowsDownload(unittest.TestCase):

    def test_gguf_repo_auto_narrows_to_Q4_K_M(self) -> None:
        captured: dict = {}

        def fake(*, repo_id, local_dir, max_workers, allow_patterns):
            captured["allow_patterns"] = allow_patterns
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "model-Q4_K_M.gguf").write_bytes(b"\x00" * 64)
            (Path(local_dir) / "README.md").write_bytes(b"")
            return local_dir

        with TemporaryDirectory() as td, \
             unittest.mock.patch.object(
                 model_stager, "_free_bytes", return_value=10**12):
            target = Path(td) / "Qwen3.6-27B-GGUF"
            r = model_stager.ensure_model_staged(
                hf_id="unsloth/Qwen3.6-27B-GGUF",
                target_dir=target,
                metadata={"param_count": "27B"},
                downloader=fake,
            )
        self.assertTrue(r.ok, f"unexpected: {r}")
        pats = captured["allow_patterns"]
        self.assertIsNotNone(pats,
                             "GGUF repo MUST get narrowed allow_patterns")
        assert pats is not None
        self.assertTrue(any("Q4_K_M" in p for p in pats),
                        f"expected Q4_K_M filter, got {pats!r}")


# ── 6. cleanup_failed_cache --narrow-gguf ──────────────────────────────────


class TestNarrowGgufCleanup(unittest.TestCase):
    """The cleanup script's --narrow-gguf path: for any cache that
    contains multiple GGUF quant variants, keep ONLY Q4_K_M, delete
    the rest. Mirrors the unsloth/Qwen3.6-27B-GGUF 351 GB recovery
    scenario."""

    def setUp(self) -> None:
        self._td = TemporaryDirectory()
        self.cache_root = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def _build_qwen_gguf_cache(self) -> Path:
        d = self.cache_root / "Qwen3.6-27B-GGUF"
        d.mkdir(parents=True)
        # 8 quants, sizes loosely matching real ones (scaled down 1e6×).
        variants = {
            "Qwen3.6-27B-Q4_K_M.gguf": 16_000,  # kept
            "Qwen3.6-27B-Q4_0.gguf": 15_000,
            "Qwen3.6-27B-Q5_K_M.gguf": 20_000,
            "Qwen3.6-27B-Q8_0.gguf": 28_000,
            "Qwen3.6-27B-IQ4_NL.gguf": 15_000,
            "Qwen3.6-27B-IQ4_XS.gguf": 15_000,
            "Qwen3.6-27B-Q3_K_M.gguf": 13_000,
            "Qwen3.6-27B-Q3_K_S.gguf": 12_000,
        }
        for name, sz in variants.items():
            (d / name).write_bytes(b"\x00" * sz)
        # Realistic ledger.
        ledger = d / ".cache" / "huggingface" / "download"
        ledger.mkdir(parents=True)
        for name in variants:
            (ledger / f"{name}.metadata").write_bytes(b"x")
        return d

    def test_narrow_keeps_only_q4_k_m_dry_run(self) -> None:
        from scripts.cleanup_failed_cache import _narrow_gguf_caches
        self._build_qwen_gguf_cache()
        freed = _narrow_gguf_caches(self.cache_root,
                                    preferred_quant="Q4_K_M",
                                    dry_run=True)
        # 7 variants total ~118 KB raw — exact bytes depend on ledger,
        # but the order of magnitude is right.
        self.assertGreater(freed, 100_000)
        # Nothing actually deleted in dry-run.
        d = self.cache_root / "Qwen3.6-27B-GGUF"
        self.assertTrue((d / "Qwen3.6-27B-Q4_0.gguf").exists())
        self.assertTrue((d / "Qwen3.6-27B-Q4_K_M.gguf").exists())

    def test_narrow_actually_deletes(self) -> None:
        from scripts.cleanup_failed_cache import _narrow_gguf_caches
        d = self._build_qwen_gguf_cache()
        freed = _narrow_gguf_caches(self.cache_root,
                                    preferred_quant="Q4_K_M",
                                    dry_run=False)
        self.assertGreater(freed, 100_000)
        # Only Q4_K_M survives.
        remaining = sorted(p.name for p in d.iterdir()
                           if p.suffix == ".gguf")
        self.assertEqual(remaining, ["Qwen3.6-27B-Q4_K_M.gguf"])

    def test_narrow_skips_single_quant_cache(self) -> None:
        # A cache with one .gguf file should be untouched.
        from scripts.cleanup_failed_cache import _narrow_gguf_caches
        d = self.cache_root / "single-quant-model"
        d.mkdir()
        (d / "model-Q4_K_M.gguf").write_bytes(b"\x00" * 1024)
        freed = _narrow_gguf_caches(self.cache_root,
                                    preferred_quant="Q4_K_M",
                                    dry_run=False)
        self.assertEqual(freed, 0)
        self.assertTrue((d / "model-Q4_K_M.gguf").exists())

    def test_narrow_preserves_when_no_match(self) -> None:
        # If no variant matches Q4_K_M, leave everything alone (safer
        # than deleting all weights and ending up with zero usable
        # quantizations).
        from scripts.cleanup_failed_cache import _narrow_gguf_caches
        d = self.cache_root / "no-q4km-here"
        d.mkdir()
        (d / "model.Q8_0.gguf").write_bytes(b"\x00" * 1024)
        (d / "model.Q5_K_M.gguf").write_bytes(b"\x00" * 1024)
        freed = _narrow_gguf_caches(self.cache_root,
                                    preferred_quant="Q4_K_M",
                                    dry_run=False)
        self.assertEqual(freed, 0)
        self.assertTrue((d / "model.Q8_0.gguf").exists())
        self.assertTrue((d / "model.Q5_K_M.gguf").exists())


if __name__ == "__main__":
    unittest.main()
