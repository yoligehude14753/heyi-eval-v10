"""
Smoke tests for state machine + store + notify (T1 deliverable).

Run from heyi-eval-v9/:
    python -m unittest discover tests
"""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from orchestrator import notify
from orchestrator.main import enqueue, pop_one, run_pipeline
from orchestrator.state_machine import (
    STAGES_IN_ORDER,
    Run,
    RunStatus,
    StageName,
    StageStatus,
    load_state,
    save_state,
)
from orchestrator.store import Store


class StateMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="heyi-eval-test-"))
        self.store = Store(self.tmp)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ── basic state machine ───────────────────────────────────────────────

    def test_new_run_has_all_stages_pending(self) -> None:
        run = Run(run_id="r-test-1", hf_id="X/Y")
        for s in STAGES_IN_ORDER:
            self.assertEqual(run.get_stage(s).status, StageStatus.PENDING)

    def test_first_pending_after_some_ok(self) -> None:
        run = Run(run_id="r-test-2", hf_id="X/Y")
        run.get_stage(StageName.DISCOVER).status = StageStatus.OK
        run.get_stage(StageName.CURATE).status = StageStatus.OK
        self.assertEqual(run.first_pending_stage(), StageName.METADATA)

    def test_needs_full_restart_before_deploy(self) -> None:
        run = Run(run_id="r-test-3", hf_id="X/Y")
        run.get_stage(StageName.CURATE).status = StageStatus.FAILED
        self.assertTrue(run.needs_full_restart())

    def test_no_full_restart_after_deploy(self) -> None:
        run = Run(run_id="r-test-4", hf_id="X/Y")
        run.get_stage(StageName.DEPLOY).status = StageStatus.OK
        run.get_stage(StageName.CAPABILITY).status = StageStatus.FAILED
        self.assertFalse(run.needs_full_restart())

    def test_save_and_load_state_roundtrip(self) -> None:
        run = Run(run_id="r-test-5", hf_id="org/model")
        run.get_stage(StageName.DISCOVER).mark_started()
        run.get_stage(StageName.DISCOVER).mark_ok(artifacts=["candidates.jsonl"])
        save_state(self.store.runs_dir, run)

        roundtrip = load_state(self.store.runs_dir, run.run_id)
        self.assertIsNotNone(roundtrip)
        assert roundtrip is not None  # for type checker
        self.assertEqual(roundtrip.hf_id, "org/model")
        info = roundtrip.get_stage(StageName.DISCOVER)
        self.assertEqual(info.status, StageStatus.OK)
        self.assertIn("candidates.jsonl", info.artifacts)
        self.assertIsNotNone(info.duration_s)

    # ── full pipeline + recovery ──────────────────────────────────────────

    def test_run_pipeline_all_stages_ok_with_stubs(self) -> None:
        run = Run(run_id="r-pipe-1", hf_id="org/m")
        self.store.save_run(run)
        run_pipeline(run, self.store, stub_only=True)
        self.assertEqual(run.status, RunStatus.OK)
        for s in STAGES_IN_ORDER:
            self.assertEqual(run.get_stage(s).status, StageStatus.OK)

    def test_recovery_picks_up_in_progress(self) -> None:
        # Simulate crash: mark DEPLOY ok but CAPABILITY in_progress, leave run IN_PROGRESS
        run = Run(run_id="r-recov-1", hf_id="org/m", status=RunStatus.IN_PROGRESS)
        for s in (StageName.DISCOVER, StageName.CURATE, StageName.METADATA,
                  StageName.ENGINE_SELECT, StageName.DEPLOY, StageName.READY_WAIT):
            run.get_stage(s).status = StageStatus.OK
        run.get_stage(StageName.CAPABILITY).status = StageStatus.IN_PROGRESS
        self.store.save_run(run)

        recovered = self.store.recover_in_progress()
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].run_id, "r-recov-1")
        # Resume should NOT do full restart (DEPLOY was ok)
        self.assertFalse(recovered[0].needs_full_restart())

    def test_queue_enqueue_pop(self) -> None:
        rid = enqueue(self.store, "Qwen/Qwen2.5-0.5B-Instruct")
        self.assertTrue(rid.startswith("r-"))
        item = pop_one(self.store)
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item["hf_id"], "Qwen/Qwen2.5-0.5B-Instruct")
        # queue now empty
        self.assertIsNone(pop_one(self.store))


class NotifyOutboxTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="heyi-eval-notify-"))
        self.outbox = self.tmp / "store" / "notify_outbox.jsonl"

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_started_writes_one_jsonl_line(self) -> None:
        notify.run_started(self.outbox, run_id="r-1", hf_id="A/B", engine="vllm", eta_s=200)
        lines = self.outbox.read_text().splitlines()
        self.assertEqual(len(lines), 1)
        evt = json.loads(lines[0])
        self.assertEqual(evt["event_type"], "run_started")
        self.assertEqual(evt["hf_id"], "A/B")
        self.assertIn("wechat", evt["channels"])

    def test_run_failed_is_p1_and_persists_to_disk(self) -> None:
        notify.run_failed(
            self.outbox, run_id="r-2", hf_id="A/B", stage="DEPLOY", error="ConnectionReset",
        )
        evt = json.loads(self.outbox.read_text().splitlines()[-1])
        self.assertEqual(evt["level"], "error")
        self.assertEqual(evt["priority"], "P1")

    def test_heartbeat_skips_wechat_channel(self) -> None:
        notify.heartbeat(
            self.outbox, in_flight=1, completed_today=3, failed_today=0, free_disk_gb=423,
        )
        evt = json.loads(self.outbox.read_text().splitlines()[-1])
        self.assertNotIn("wechat", evt["channels"])
        self.assertIn("desktop", evt["channels"])


if __name__ == "__main__":
    unittest.main()
