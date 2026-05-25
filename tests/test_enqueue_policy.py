"""Tests for discover.main.cmd_enqueue + orchestrator dedup.

Covers the filtering rules (pipeline_tag, downloads/likes threshold,
private/gated) and the "skip if already evaluated successfully" path.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from discover.main import _enqueue_policy_passes, cmd_enqueue  # noqa: E402
from discover.tracker import Candidate, append_candidates  # noqa: E402
from orchestrator.main import _has_recent_successful_run, enqueue  # noqa: E402
from orchestrator.state_machine import Run, RunStatus  # noqa: E402
from orchestrator.store import Store  # noqa: E402


def _args(**overrides):
    a = argparse.Namespace(
        limit=5, min_downloads=1000, min_likes=20,
    )
    for k, v in overrides.items():
        setattr(a, k, v)
    return a


def _cand(hf_id: str, **kw) -> Candidate:
    return Candidate(
        hf_id=hf_id,
        discovered_at="2026-05-21T00:00:00+00:00",
        reason=kw.get("reason", "whitelist"),
        source_org=hf_id.split("/", 1)[0] if "/" in hf_id else None,
        pipeline_tag=kw.get("pipeline_tag", "text-generation"),
        downloads=kw.get("downloads", 100_000),
        likes=kw.get("likes", 50),
        private=kw.get("private", False),
        gated=kw.get("gated", False),
    )


class PolicyTests(unittest.TestCase):

    def test_text_gen_passes(self):
        allow, reason = _enqueue_policy_passes(_cand("X/Y"), _args())
        self.assertTrue(allow, reason)

    def test_private_rejected(self):
        # PR#57: reason text is localized to Chinese.
        allow, reason = _enqueue_policy_passes(_cand("X/Y", private=True), _args())
        self.assertFalse(allow)
        self.assertTrue("私有" in reason or "受限" in reason, reason)

    def test_gated_rejected(self):
        allow, _reason = _enqueue_policy_passes(_cand("X/Y", gated=True), _args())
        self.assertFalse(allow)

    def test_unsupported_modality_rejected(self):
        c = _cand("X/Y", pipeline_tag="text-to-image")
        allow, reason = _enqueue_policy_passes(c, _args())
        self.assertFalse(allow)
        # PR#57: Chinese rendering keeps the raw pipeline tag inline.
        self.assertIn("text-to-image", reason)
        self.assertTrue("不支持" in reason or "pipeline_tag" in reason, reason)

    def test_asr_supported(self):
        c = _cand("X/Y", pipeline_tag="automatic-speech-recognition")
        allow, _ = _enqueue_policy_passes(c, _args())
        self.assertTrue(allow)

    def test_image_text_to_text_supported(self):
        c = _cand("X/Y", pipeline_tag="image-text-to-text")
        allow, _ = _enqueue_policy_passes(c, _args())
        self.assertTrue(allow)

    def test_low_signal_rejected(self):
        # PR#56: whitelist candidates bypass the dl/likes gate (we trust
        # the vendor). Use ``reason="trending"`` so the threshold applies.
        c = _cand("X/Y", downloads=10, likes=2, reason="trending")
        allow, reason = _enqueue_policy_passes(c, _args())
        self.assertFalse(allow)
        # PR#57: reason text is localized to Chinese.
        self.assertIn("信号过低", reason)

    def test_pr56_whitelist_bypasses_low_signal(self):
        """A whitelisted vendor's fresh release with zero downloads must
        STILL be admitted — PR#56 trust-the-vendor policy."""
        c = _cand("Qwen/Brand-New", downloads=0, likes=0, reason="whitelist")
        allow, reason = _enqueue_policy_passes(c, _args())
        self.assertTrue(allow, reason)

    def test_high_likes_alone_passes(self):
        """Downloads low but likes high — OR logic should admit."""
        c = _cand("X/Y", downloads=10, likes=200, reason="trending")
        allow, _ = _enqueue_policy_passes(c, _args())
        self.assertTrue(allow)

    def test_unknown_tag_passes_through(self):
        """If pipeline_tag is None (e.g. HF didn't classify), we let it
        through and engine_select decides later."""
        c = _cand("X/Y", pipeline_tag=None)
        allow, _ = _enqueue_policy_passes(c, _args())
        self.assertTrue(allow)


class DedupTests(unittest.TestCase):

    def test_no_recent_run_returns_false(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            self.assertFalse(_has_recent_successful_run(store, "X/Y"))

    def test_recent_ok_run_returns_true(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            r = Run(run_id="r-1", hf_id="X/Y")
            r.status = RunStatus.OK
            r.created_at = time.time()
            r.ended_at = time.time()
            store.save_run(r)
            self.assertTrue(_has_recent_successful_run(store, "X/Y"))

    def test_failed_run_does_not_block(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            r = Run(run_id="r-1", hf_id="X/Y")
            r.status = RunStatus.FAILED
            r.created_at = time.time()
            store.save_run(r)
            self.assertFalse(_has_recent_successful_run(store, "X/Y"))

    def test_old_ok_run_does_not_block(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            r = Run(run_id="r-1", hf_id="X/Y")
            r.status = RunStatus.OK
            # 60 days ago
            r.created_at = time.time() - 60 * 86400
            r.ended_at = r.created_at + 1000
            store.save_run(r)
            self.assertFalse(_has_recent_successful_run(store, "X/Y", within_days=30))

    def test_enqueue_skip_if_recent_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            r = Run(run_id="r-1", hf_id="X/Y")
            r.status = RunStatus.OK
            r.created_at = time.time()
            r.ended_at = r.created_at + 1000
            store.save_run(r)

            result = enqueue(store, "X/Y", skip_if_recent=True)
            self.assertIsNone(result)

    def test_enqueue_without_skip_always_adds(self):
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            r = Run(run_id="r-1", hf_id="X/Y")
            r.status = RunStatus.OK
            r.created_at = time.time()
            store.save_run(r)

            new_id = enqueue(store, "X/Y", skip_if_recent=False)
            self.assertIsNotNone(new_id)

    # PR#38: recent-failure dampening on the cron path
    def test_enqueue_skip_if_recent_blocks_on_recent_failure(self):
        """PR#38: cron auto-enqueue (skip_if_recent=True) must NOT
        re-enqueue a model that failed in the last 12 h. Live nv8
        observed 4-5× repeats of the same fast-fail models per day."""
        from orchestrator.main import _has_recent_failed_run
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            r = Run(run_id="r-1", hf_id="X/Y")
            r.status = RunStatus.FAILED
            r.created_at = time.time() - 3600  # 1h ago
            r.ended_at = r.created_at + 30
            store.save_run(r)

            self.assertTrue(_has_recent_failed_run(store, "X/Y"))
            result = enqueue(store, "X/Y", skip_if_recent=True)
            self.assertIsNone(result, "cron should skip recently-failed")

    def test_enqueue_skip_if_recent_blocks_on_recent_aborted(self):
        """Aborted is also a 'recent fail' signal — typically oversize
        gating; no point in re-trying within 12 h."""
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            r = Run(run_id="r-1", hf_id="X/Y")
            r.status = RunStatus.ABORTED
            r.created_at = time.time() - 1800  # 30min ago
            r.ended_at = r.created_at + 5
            store.save_run(r)

            result = enqueue(store, "X/Y", skip_if_recent=True)
            self.assertIsNone(result)

    def test_enqueue_manual_path_still_adds_failed(self):
        """Manual CLI (`orchestrator.main enqueue`) passes
        skip_if_recent=False and MUST still allow re-running failed
        models — that's the only way to retest after a fix."""
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            r = Run(run_id="r-1", hf_id="X/Y")
            r.status = RunStatus.FAILED
            r.created_at = time.time() - 600
            store.save_run(r)

            new_id = enqueue(store, "X/Y", skip_if_recent=False)
            self.assertIsNotNone(new_id, "manual enqueue must bypass "
                                         "the recent-fail block")

    def test_enqueue_skip_old_failure_does_not_block(self):
        """A failure from 15 h ago should NOT block; the 12 h window
        expires."""
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            r = Run(run_id="r-1", hf_id="X/Y")
            r.status = RunStatus.FAILED
            r.created_at = time.time() - 15 * 3600
            r.ended_at = r.created_at + 30
            store.save_run(r)

            new_id = enqueue(store, "X/Y", skip_if_recent=True)
            self.assertIsNotNone(new_id, "cron must retry after 12h "
                                         "backoff window")

    def test_has_recent_failed_run_window_param(self):
        """Custom window parameter works."""
        from orchestrator.main import _has_recent_failed_run
        with tempfile.TemporaryDirectory() as td:
            store = Store(Path(td))
            r = Run(run_id="r-1", hf_id="X/Y")
            r.status = RunStatus.FAILED
            r.created_at = time.time() - 6 * 3600  # 6h ago
            store.save_run(r)
            self.assertTrue(_has_recent_failed_run(
                store, "X/Y", within_hours=12))
            self.assertFalse(_has_recent_failed_run(
                store, "X/Y", within_hours=4))


class CmdEnqueueIntegrationTests(unittest.TestCase):

    def test_full_flow_filters_and_enqueues(self):
        """Three candidates: one good (whitelist+high signal), one
        private (always rejected), one low-signal trending (rejected by
        PR#56 threshold). Should enqueue only the first.

        PR#56: the original ``bad/lowsig`` fixture had reason="whitelist"
        which now bypasses the dl/likes gate (trust-the-vendor). Force
        reason="trending" so the threshold still applies."""
        with tempfile.TemporaryDirectory() as td:
            data_root = Path(td)
            cands_path = data_root / "discover" / "candidates.jsonl"
            append_candidates(cands_path, [
                _cand("good/model", downloads=200_000, likes=100),
                _cand("bad/private", private=True, downloads=200_000),
                _cand("bad/lowsig", downloads=10, likes=2,
                      reason="trending"),
            ])
            args = _args(limit=10)
            with mock.patch("discover.main._default_data_root",
                            return_value=data_root):
                rc = cmd_enqueue(args)
            self.assertEqual(rc, 0)

            # Queue file should have 1 line
            qp = data_root / "store" / "queue.jsonl"
            self.assertTrue(qp.exists())
            lines = qp.read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertIn("good/model", lines[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
