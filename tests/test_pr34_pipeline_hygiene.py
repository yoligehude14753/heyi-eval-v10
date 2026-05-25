"""PR#34 pipeline hygiene — stop the bleeding before the GLM-OCR
auto-repair experiment can fairly finish.

PR#33 ran a live experiment on nv8 to validate the new deploy-repair
loop. The orchestrator queue immediately collapsed under three
unrelated structural issues:

  a) discover/enqueue admitted ``Kijai/WanVideo_comfy`` — a Kijai
     convenience bundle (``library_name=diffusion-single-file``,
     ``pipeline_tag=None``) that contains 14 unrelated text-to-video
     model variants. 142 GB of unevaluable data poured into the cache.

  b) The hourly auto-enqueue timer pushed the same hf_id into
     queue.jsonl four times because there was no dedup against either
     the queue or the in_progress set. We ended up with 16 queue
     entries pointing at five distinct models.

  c) ``snapshot_download`` against hf-mirror occasionally enters a
     CLOSE-WAIT retry storm and makes zero progress for an hour+. We
     observed gemma-4-26B stuck at 6.5 GB for 56 minutes with no
     orchestrator log activity. The HF library has no native total-time
     budget, so we wrap the call with a daemon-thread + join(timeout).

  d) When systemd restarts the orchestrator mid-run, the active row
     stays at ``in_progress`` forever, the cache leaks, and dedup (b)
     then thinks the model is still being worked on. Sweep stale rows
     on startup.

This file tests all four fixes.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from discover.main import _enqueue_policy_passes  # noqa: E402
from discover.tracker import Candidate  # noqa: E402
from orchestrator import model_stager  # noqa: E402
from orchestrator.main import (  # noqa: E402
    _hf_ids_in_progress,
    _hf_ids_in_queue,
    _sweep_orphan_in_progress,
    enqueue,
)
from orchestrator.state_machine import Run, RunStatus  # noqa: E402
from orchestrator.store import Store  # noqa: E402


def _args(**overrides):
    a = argparse.Namespace(limit=5, min_downloads=1000, min_likes=20)
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _cand(hf_id: str, **kw) -> Candidate:
    return Candidate(
        hf_id=hf_id,
        discovered_at="2026-05-23T00:00:00+00:00",
        reason=kw.get("reason", "trending"),
        source_org=hf_id.split("/", 1)[0] if "/" in hf_id else None,
        pipeline_tag=kw.get("pipeline_tag", "text-generation"),
        library_name=kw.get("library_name"),
        downloads=kw.get("downloads", 100_000),
        likes=kw.get("likes", 50),
        private=kw.get("private", False),
        gated=kw.get("gated", False),
    )


# ── PR#34a: library_name rejection ──────────────────────────────────


class LibraryNameRejectionTests(unittest.TestCase):
    """The WanVideo_comfy regression: convenience bundles must not enter
    the queue."""

    def test_diffusion_single_file_rejected(self):
        c = _cand("Kijai/WanVideo_comfy", pipeline_tag=None,
                  library_name="diffusion-single-file")
        allow, reason = _enqueue_policy_passes(c, _args())
        self.assertFalse(allow)
        self.assertIn("library_name=diffusion-single-file", reason)

    def test_comfyui_rejected(self):
        c = _cand("X/Y", pipeline_tag=None, library_name="ComfyUI")
        allow, reason = _enqueue_policy_passes(c, _args())
        self.assertFalse(allow)
        self.assertIn("ComfyUI", reason)

    def test_diffusers_single_file_rejected(self):
        c = _cand("X/Y", library_name="diffusers-single-file")
        allow, _reason = _enqueue_policy_passes(c, _args())
        self.assertFalse(allow)

    def test_transformers_library_passes(self):
        c = _cand("X/Y", library_name="transformers")
        allow, _ = _enqueue_policy_passes(c, _args())
        self.assertTrue(allow)

    def test_unknown_library_passes(self):
        """We don't whitelist library_name — only blocklist obvious bundle
        markers. A library_name we've never seen should still pass."""
        c = _cand("X/Y", library_name="some-new-lib-2027")
        allow, _ = _enqueue_policy_passes(c, _args())
        self.assertTrue(allow)


# ── PR#34b: queue dedup ─────────────────────────────────────────────


class QueueDedupTests(unittest.TestCase):

    def test_first_enqueue_succeeds(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            run_id = enqueue(store, "owner/model")
            self.assertIsNotNone(run_id)

    def test_duplicate_queue_entry_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            a = enqueue(store, "owner/model")
            b = enqueue(store, "owner/model")
            self.assertIsNotNone(a)
            self.assertIsNone(b, "second enqueue should dedup")

    def test_in_progress_blocks_enqueue(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            r = Run(run_id="r-old", hf_id="owner/model")
            r.status = RunStatus.IN_PROGRESS
            r.created_at = time.time()
            store.save_run(r)
            result = enqueue(store, "owner/model")
            self.assertIsNone(result)

    def test_pending_status_blocks_enqueue(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            r = Run(run_id="r-old", hf_id="owner/model")
            r.status = RunStatus.PENDING
            r.created_at = time.time()
            store.save_run(r)
            result = enqueue(store, "owner/model")
            self.assertIsNone(result)

    def test_distinct_hf_ids_both_enqueue(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            self.assertIsNotNone(enqueue(store, "a/x"))
            self.assertIsNotNone(enqueue(store, "b/y"))

    def test_failed_run_does_not_block(self):
        """If the previous run FAILED, re-enqueue is legitimate (operator
        wants to retry after a fix)."""
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            r = Run(run_id="r-old", hf_id="owner/model")
            r.status = RunStatus.FAILED
            r.created_at = time.time()
            r.ended_at = time.time()
            store.save_run(r)
            result = enqueue(store, "owner/model")
            self.assertIsNotNone(result)

    def test_helpers_return_expected_sets(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            enqueue(store, "a/x")
            r = Run(run_id="r-old", hf_id="b/y")
            r.status = RunStatus.IN_PROGRESS
            r.created_at = time.time()
            store.save_run(r)
            self.assertEqual(_hf_ids_in_queue(store), {"a/x"})
            self.assertEqual(_hf_ids_in_progress(store), {"b/y"})


# ── PR#34c: download wall-clock budget ──────────────────────────────


class DownloadTimeoutTests(unittest.TestCase):

    def _slow_downloader(self, delay: float):
        def _dl(*, repo_id, local_dir, max_workers, allow_patterns):
            time.sleep(delay)
            # touch a dummy weight so post-download check would pass IF
            # we ever got that far.
            (Path(local_dir) / "model.safetensors").write_bytes(b"x" * 256)
            return local_dir
        return _dl

    def _hanging_downloader(self):
        def _dl(*, repo_id, local_dir, max_workers, allow_patterns):
            time.sleep(60)  # longer than any test timeout
            raise RuntimeError("would never get here")
        return _dl

    def test_no_timeout_means_no_thread_wrapper(self):
        """When timeout_s is None / 0, run inline (preserves the prior
        behaviour for stub tests)."""
        with tempfile.TemporaryDirectory() as td:
            r = model_stager.ensure_model_staged(
                hf_id="x/y",
                target_dir=Path(td),
                metadata={"hf_info": {"safetensors": {"total": 1024}}},
                downloader=self._slow_downloader(0.0),
                download_timeout_s=None,
            )
            self.assertTrue(r.ok, msg=r.error)

    def test_timeout_fast_call_succeeds(self):
        with tempfile.TemporaryDirectory() as td:
            r = model_stager.ensure_model_staged(
                hf_id="x/y",
                target_dir=Path(td),
                metadata={"hf_info": {"safetensors": {"total": 1024}}},
                downloader=self._slow_downloader(0.0),
                download_timeout_s=5.0,
            )
            self.assertTrue(r.ok, msg=r.error)

    def test_timeout_slow_call_hard_fails(self):
        with tempfile.TemporaryDirectory() as td:
            r = model_stager.ensure_model_staged(
                hf_id="x/y",
                target_dir=Path(td),
                metadata={"hf_info": {"safetensors": {"total": 1024}}},
                downloader=self._hanging_downloader(),
                download_timeout_s=0.3,
            )
            self.assertFalse(r.ok)
            self.assertFalse(r.skipped, "timeout is hard-fail, not graceful")
            self.assertEqual(r.error_kind, "download_timeout")
            self.assertIn("timeout", (r.error or "").lower())

    def test_timeout_failure_cleans_up_partial(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "partial-model"

            def _dl_that_leaves_partial(
                *, repo_id, local_dir, max_workers, allow_patterns
            ):
                # write some half-finished bytes to disk before "hanging"
                Path(local_dir).mkdir(parents=True, exist_ok=True)
                (Path(local_dir) / ".incomplete").write_bytes(b"x" * 4096)
                time.sleep(60)

            r = model_stager.ensure_model_staged(
                hf_id="x/y",
                target_dir=target,
                metadata={"hf_info": {"safetensors": {"total": 1024}}},
                downloader=_dl_that_leaves_partial,
                download_timeout_s=0.3,
            )
            self.assertEqual(r.error_kind, "download_timeout")
            self.assertGreater(r.bytes_freed or 0, 0,
                               "partial bytes must be cleaned up")
            self.assertFalse(target.exists(), "cleanup must delete dir")


# ── PR#34d: orphan in_progress sweeper ──────────────────────────────


class OrphanSweepTests(unittest.TestCase):

    def _make_run(self, store: Store, run_id: str, hf_id: str, *,
                  status: RunStatus, age_s: float) -> None:
        r = Run(run_id=run_id, hf_id=hf_id)
        r.status = status
        r.created_at = time.time() - age_s
        r.updated_at = r.created_at
        store.save_run(r)

    def test_fresh_in_progress_is_left_alone(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            self._make_run(store, "r-1", "x/y",
                           status=RunStatus.IN_PROGRESS, age_s=60.0)
            n = _sweep_orphan_in_progress(store, stale_after_s=1800.0)
            self.assertEqual(n, 0)
            self.assertEqual(
                store.get_run("r-1").status, RunStatus.IN_PROGRESS
            )

    def test_stale_in_progress_becomes_aborted(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            self._make_run(store, "r-1", "x/y",
                           status=RunStatus.IN_PROGRESS, age_s=3600.0)
            n = _sweep_orphan_in_progress(store, stale_after_s=1800.0)
            self.assertEqual(n, 1)
            r = store.get_run("r-1")
            self.assertEqual(r.status, RunStatus.ABORTED)
            self.assertIn("orphan", (r.failure_reason or "").lower())

    def test_sweep_does_not_touch_terminal_states(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            self._make_run(store, "r-ok", "x/y",
                           status=RunStatus.OK, age_s=10_000.0)
            self._make_run(store, "r-fail", "x/y",
                           status=RunStatus.FAILED, age_s=10_000.0)
            n = _sweep_orphan_in_progress(store, stale_after_s=1.0)
            self.assertEqual(n, 0)

    def test_dedup_unblocks_after_sweep(self):
        """After sweeping a dead orphan, the dedup gate should release
        the hf_id so re-enqueue works."""
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            self._make_run(store, "r-1", "owner/model",
                           status=RunStatus.IN_PROGRESS, age_s=3600.0)
            blocked = enqueue(store, "owner/model")
            self.assertIsNone(blocked, "must dedup against stale row")
            _sweep_orphan_in_progress(store, stale_after_s=1800.0)
            unblocked = enqueue(store, "owner/model")
            self.assertIsNotNone(unblocked,
                                 "must work once orphan is aborted")

    # PR#44: force_all kwarg for the startup-time path
    def test_force_all_sweeps_even_when_fresh(self):
        """PR#44: at orchestrator startup, anything still in_progress
        is necessarily orphaned because the orchestrator is the only
        process that writes that column. ``force_all=True`` bypasses
        the time check.

        Live nv8 motivation: restarting the orchestrator while
        STAGE_MODEL was downloading mistralai/Voxtral-Mini-3B-2507
        left the row in_progress with last_touch=7min ago — below
        the 30-min threshold, so the run stayed orphaned forever.
        """
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            # Fresh — 1-minute-old, well under stale_after_s=1800
            self._make_run(store, "r-fresh", "x/y",
                           status=RunStatus.IN_PROGRESS, age_s=60.0)
            # Time-based path would NOT sweep this
            n_time = _sweep_orphan_in_progress(store, stale_after_s=1800.0)
            self.assertEqual(n_time, 0)
            # force_all=True must sweep it regardless
            n_force = _sweep_orphan_in_progress(
                store, stale_after_s=1800.0, force_all=True,
            )
            self.assertEqual(n_force, 1)
            self.assertEqual(
                store.get_run("r-fresh").status, RunStatus.ABORTED,
            )

    def test_force_all_still_skips_terminal_states(self):
        """Regression: force_all only affects the time-check; OK /
        FAILED / ABORTED rows must NOT be touched."""
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            self._make_run(store, "r-ok", "x/y",
                           status=RunStatus.OK, age_s=10.0)
            self._make_run(store, "r-fail", "x/y",
                           status=RunStatus.FAILED, age_s=10.0)
            n = _sweep_orphan_in_progress(
                store, stale_after_s=1800.0, force_all=True,
            )
            self.assertEqual(n, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
