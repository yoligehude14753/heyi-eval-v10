"""
heyi-eval v9 orchestrator entry point.

Usage:
    python -m orchestrator.main enqueue <hf_id>
    python -m orchestrator.main run [--stub-only]    # consume one job
    python -m orchestrator.main loop [--stub-only]   # consume forever
    python -m orchestrator.main status               # list recent runs
    python -m orchestrator.main resume [--stub-only] # scan in-progress runs

--stub-only: skip the CC-agent docker invocations (DEPLOY/CAPABILITY/
SHOWCASE/CLEANUP all become no-ops that mark OK). Useful for testing the
state machine / queue / outbox without GPU. Default off — on nv8 we want
real CC-agent invocations.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from . import notify
from .config import OrchestratorConfig
from .stages import StageResult, execute_stage
from .state_machine import (
    STAGES_IN_ORDER,
    Run,
    RunStatus,
    StageName,
    StageStatus,
)
from .store import Store
from .validator import ValidationError


def _default_data_root() -> Path:
    return Path(os.environ.get("HEYI_EVAL_DATA", "~/heyi-eval-data")).expanduser()


def _new_run_id(hf_id: str) -> str:
    short = hf_id.replace("/", "_").lower()[:30]
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"r-{ts}-{short}-{uuid.uuid4().hex[:4]}"


def _free_disk_gb(path: Path) -> float:
    """Return free space on the filesystem holding `path`, in GiB.
    Returns -1.0 if the call fails (e.g. path doesn't exist yet)."""
    try:
        import shutil
        usage = shutil.disk_usage(str(path) if path.exists() else str(path.parent))
        return round(usage.free / (1024 ** 3), 1)
    except Exception:
        return -1.0


@dataclass
class LoopState:
    """Mutable state carried across iterations of the orchestrator main loop.

    Extracted so cmd_loop has zero local state and the gates can be unit
    tested in isolation.
    """
    last_heartbeat: float = 0.0
    engine_unhealthy_since: float | None = None
    engine_last_notify: float = 0.0
    completed_today: int = 0
    failed_today: int = 0
    today_date: object = None  # datetime.date

    def __post_init__(self):
        from datetime import date as _date
        if self.today_date is None:
            self.today_date = _date.today()


def _engine_preflight_gate(
    state: LoopState, cfg: OrchestratorConfig, store: Store,
    *, remind_interval_s: float = 3600.0, now: float | None = None,
    probe=None,
) -> tuple[bool, str]:
    """Returns (allow_intake, reason).

    `allow_intake=True` means heyi_engine is healthy, the loop should
    proceed to pop a job. `allow_intake=False` means we must back off
    this iteration. Emits incident events on state transitions.
    """
    if probe is None:
        from curator.health import probe_engine as probe
    now = now or time.time()
    h = probe(cfg.engine_url, api_key=cfg.engine_api_key, timeout_s=10.0)

    if h.ok:
        if state.engine_unhealthy_since is not None:
            down_s = int(now - state.engine_unhealthy_since)
            try:
                notify.incident(
                    store.outbox_path,
                    what="orchestrator-resumed-engine-recovered",
                    detail=f"heyi_engine healthy after {down_s}s. Resuming job intake.",
                )
            except Exception:
                pass
            state.engine_unhealthy_since = None
            state.engine_last_notify = 0.0
        return True, "healthy"

    # unhealthy
    if state.engine_unhealthy_since is None:
        state.engine_unhealthy_since = now
        try:
            notify.incident(
                store.outbox_path,
                what="orchestrator-paused-engine-down",
                detail=(f"heyi_engine unhealthy: {h.detail}. "
                        f"Orchestrator will pause job intake until upstream recovers. "
                        f"Queue is preserved."),
            )
            state.engine_last_notify = now
        except Exception:
            pass
    elif now - state.engine_last_notify > remind_interval_s:
        down_s = int(now - state.engine_unhealthy_since)
        try:
            notify.incident(
                store.outbox_path,
                what="orchestrator-still-paused",
                detail=f"Still paused: heyi_engine down for {down_s}s. Last probe: {h.detail}",
            )
            state.engine_last_notify = now
        except Exception:
            pass
    return False, h.detail


def _maybe_emit_heartbeat(
    state: LoopState, cfg: OrchestratorConfig, store: Store,
    *, interval_s: float, now: float | None = None,
) -> bool:
    """If due, emit a heartbeat event. Returns True iff we emitted."""
    now = now or time.time()
    if now - state.last_heartbeat < interval_s:
        return False
    try:
        in_flight = len([r for r in store.list_runs(limit=200)
                         if r["status"] == "in_progress"])
        free_disk_gb = _free_disk_gb(cfg.data_root)
        notify.heartbeat(
            store.outbox_path,
            in_flight=in_flight,
            completed_today=state.completed_today,
            failed_today=state.failed_today,
            free_disk_gb=free_disk_gb,
        )
        state.last_heartbeat = now
        return True
    except Exception as e:
        print(f"[loop] heartbeat failed: {type(e).__name__}: {e}")
        return False


# ── Stage execution shim ───────────────────────────────────────────────────


def execute_stage_stub(run: Run, stage: StageName, store: Store) -> None:
    """Legacy stub kept for tests that import this symbol directly."""
    info = run.get_stage(stage)
    info.mark_started()
    store.save_run(run)
    print(f"  [stub] {stage.value} starting (attempt {info.attempt})...")
    time.sleep(0.05)
    info.mark_ok(artifacts=[])
    print(f"  [stub] {stage.value} OK in {info.duration_s}s")
    store.save_run(run)


class GracefulSkip(Exception):
    """The current stage decided in advance it can't run (insufficient
    GPU, eval pool empty, prod transient overlap). Caller marks the run
    ABORTED rather than FAILED; not a retry-worthy condition."""

    def __init__(self, stage: StageName, reason: str) -> None:
        super().__init__(reason)
        self.stage = stage
        self.reason = reason


def _execute_stage_real(
    run: Run, stage: StageName, store: Store, cfg: OrchestratorConfig
) -> None:
    """Wrap stages.execute_stage with state-machine bookkeeping."""
    info = run.get_stage(stage)
    info.mark_started()
    store.save_run(run)
    print(f"  [{stage.value}] starting (attempt {info.attempt})...")
    try:
        result: StageResult = execute_stage(run, stage, cfg, store)
    except Exception as e:
        info.mark_failed(f"executor crashed: {type(e).__name__}: {e}")
        store.save_run(run)
        raise
    if not result.ok:
        # graceful-skip path: the executor returned aborted=True in extra
        # (PR#11). The stage is marked SKIPPED (not FAILED) and the
        # pipeline-level handler should turn the run into ABORTED.
        extra = getattr(result, "extra", None) or {}
        if extra.get("aborted"):
            reason = extra.get("reason") or result.error or "aborted"
            info.mark_skipped(reason)
            store.save_run(run)
            print(f"  [{stage.value}] SKIPPED (graceful): {reason}")
            raise GracefulSkip(stage, reason)
        info.mark_failed(result.error or "executor returned ok=False")
        store.save_run(run)
        raise ValidationError(result.error or "stage failed")
    info.mark_ok(artifacts=result.artifacts)
    print(f"  [{stage.value}] OK in {info.duration_s:.1f}s")
    store.save_run(run)


# ── Pipeline runner ─────────────────────────────────────────────────────────


def run_pipeline(
    run: Run, store: Store, *, cfg: OrchestratorConfig | None = None, stub_only: bool = False,
) -> None:
    """
    Execute all 9 stages with mixed checkpoint semantics.

    Resume rules (when called on an existing IN_PROGRESS run):
      - if needs_full_restart() True: reset all run-level stages back to PENDING
        (DEPLOY hasn't happened yet, so we cheaply replay everything)
      - else: skip every stage whose status is already OK; pick up from first
        PENDING/FAILED/IN_PROGRESS stage

    Args:
      stub_only: if True, every stage runs the in-process stub (no docker / no
                 GPU). Used by tests and by `python -m orchestrator.main run
                 --stub-only` for state-machine smoke without a CC agent.
    """
    cfg = cfg or OrchestratorConfig()

    run.status = RunStatus.IN_PROGRESS
    store.save_run(run)
    notify.run_started(
        store.outbox_path,
        run_id=run.run_id,
        hf_id=run.hf_id,
        engine="vllm",
    )

    if run.needs_full_restart():
        print(f"[run] {run.run_id} restarting all run-level stages (DEPLOY not yet OK)")
        for s in (StageName.DISCOVER, StageName.CURATE, StageName.METADATA, StageName.ENGINE_SELECT):
            info = run.get_stage(s)
            if info.status != StageStatus.OK:
                info.status = StageStatus.PENDING
                info.started_at = None
                info.ended_at = None
                info.duration_s = None
                info.error = None
        store.save_run(run)

    failed_stage: StageName | None = None
    fail_reason: str | None = None

    def _exec(stage: StageName) -> None:
        if stub_only:
            execute_stage_stub(run, stage, store)
        else:
            _execute_stage_real(run, stage, store, cfg)

    for stage in STAGES_IN_ORDER:
        if run.abort_flag:
            print(f"[run] {run.run_id} aborted by flag at {stage.value}")
            run.status = RunStatus.ABORTED
            run.failure_reason = "abort_flag"
            run.ended_at = time.time()
            store.save_run(run)
            return

        info = run.get_stage(stage)
        if info.status == StageStatus.OK:
            print(f"[run] {run.run_id} skipping {stage.value} (already ok, resume)")
            continue

        try:
            _exec(stage)
        except GracefulSkip as gs:
            # Aborted, not failed: the run cannot proceed under current
            # conditions but the situation doesn't warrant retry. Mark
            # the run ABORTED, best-effort CLEANUP, return without
            # touching failure counters or run_failed notifications.
            run.status = RunStatus.ABORTED
            run.failure_reason = f"aborted at {gs.stage.value}: {gs.reason}"
            run.ended_at = time.time()
            store.save_run(run)
            try:
                cinfo = run.get_stage(StageName.CLEANUP)
                if cinfo.status not in (StageStatus.OK, StageStatus.SKIPPED) \
                        and gs.stage != StageName.CLEANUP:
                    _exec(StageName.CLEANUP)
            except Exception as ce:
                print(f"[run] cleanup-on-abort also failed: {ce}")
            try:
                notify.run_aborted(
                    store.outbox_path,
                    run_id=run.run_id,
                    hf_id=run.hf_id,
                    stage=gs.stage.value,
                    reason=gs.reason,
                )
            except Exception as ne:
                print(f"[run] notify.run_aborted failed: {ne}")
            return
        except ValidationError as ve:
            # mark_failed already done inside _execute_stage_real
            if info.status != StageStatus.FAILED:
                info.mark_failed(f"validation: {ve}")
                store.save_run(run)
            failed_stage = stage
            fail_reason = str(ve)
            break
        except Exception as e:
            if info.status != StageStatus.FAILED:
                info.mark_failed(f"{type(e).__name__}: {e}")
                store.save_run(run)
            failed_stage = stage
            fail_reason = f"{type(e).__name__}: {e}"
            break

    # Pull capability_pass_rate / showcase_impression for the completion notify
    cap_pass = None
    showcase_imp = None
    cap_info = run.get_stage(StageName.CAPABILITY)
    sc_info = run.get_stage(StageName.SHOWCASE)
    if cap_info.status == StageStatus.OK:
        try:
            import json
            cp = cfg.run_dir(run.run_id) / "capability.json"
            if cp.exists():
                cap_pass = float(json.loads(cp.read_text()).get("pass_rate") or 0.0)
        except Exception:
            cap_pass = None
    if sc_info.status == StageStatus.OK:
        try:
            import json
            sp = cfg.run_dir(run.run_id) / "showcase.json"
            if sp.exists():
                showcase_imp = json.loads(sp.read_text()).get("model_first_impression")
        except Exception:
            showcase_imp = None

    if failed_stage:
        # CLEANUP best-effort even on failure
        if failed_stage != StageName.CLEANUP:
            try:
                cinfo = run.get_stage(StageName.CLEANUP)
                if cinfo.status != StageStatus.OK:
                    _exec(StageName.CLEANUP)
            except Exception as ce:
                print(f"[run] cleanup-on-failure also failed: {ce}")
        run.status = RunStatus.FAILED
        run.failure_reason = fail_reason
        run.ended_at = time.time()
        store.save_run(run)
        notify.run_failed(
            store.outbox_path,
            run_id=run.run_id,
            hf_id=run.hf_id,
            stage=failed_stage.value,
            error=fail_reason or "unknown",
        )
        return

    run.status = RunStatus.OK
    run.ended_at = time.time()
    store.save_run(run)
    total_dur = (run.ended_at or 0) - run.created_at
    notify.run_completed(
        store.outbox_path,
        run_id=run.run_id,
        hf_id=run.hf_id,
        capability_pass_rate=cap_pass,
        showcase_impression=showcase_imp,
        duration_s=total_dur,
    )


# ── Queue (MVP: file-based jsonl) ───────────────────────────────────────────


def queue_path(store: Store) -> Path:
    return store.store_dir / "queue.jsonl"


def _has_recent_successful_run(store: Store, hf_id: str, *, within_days: int = 30) -> bool:
    """Has this hf_id been run successfully recently? Used by enqueue() to
    skip duplicate work on the cron path."""
    cutoff = time.time() - within_days * 86400
    for r in store.list_runs(limit=500):
        if r["hf_id"] == hf_id and r["status"] == "ok" and r["created_at"] >= cutoff:
            return True
    return False


def enqueue(store: Store, hf_id: str, *, skip_if_recent: bool = False) -> str | None:
    """Add an hf_id to the queue. Returns the new run_id, or None if
    skip_if_recent was True and we already have a recent OK run."""
    if skip_if_recent and _has_recent_successful_run(store, hf_id):
        return None
    qp = queue_path(store)
    qp.parent.mkdir(parents=True, exist_ok=True)
    run_id = _new_run_id(hf_id)
    line = f'{{"run_id": "{run_id}", "hf_id": "{hf_id}"}}\n'
    with qp.open("a", encoding="utf-8") as f:
        f.write(line)
    print(f"enqueued: {run_id} ({hf_id})")
    return run_id


def pop_one(store: Store) -> dict | None:
    qp = queue_path(store)
    if not qp.exists():
        return None
    lines = qp.read_text(encoding="utf-8").splitlines()
    if not lines:
        return None
    head, rest = lines[0], lines[1:]
    qp.write_text("\n".join(rest) + ("\n" if rest else ""), encoding="utf-8")
    import json
    try:
        return json.loads(head)
    except Exception:
        return None


# ── CLI ────────────────────────────────────────────────────────────────────


def cmd_enqueue(args: argparse.Namespace) -> int:
    store = Store(_default_data_root())
    enqueue(store, args.hf_id)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = OrchestratorConfig()
    store = Store(_default_data_root())
    item = pop_one(store)
    if not item:
        print("queue empty")
        return 0
    run = Run(run_id=item["run_id"], hf_id=item["hf_id"])
    store.save_run(run)
    run_pipeline(run, store, cfg=cfg, stub_only=args.stub_only)
    return 0 if run.status == RunStatus.OK else 1


def cmd_loop(args: argparse.Namespace) -> int:
    from datetime import date as _date
    cfg = OrchestratorConfig()
    store = Store(_default_data_root())
    state = LoopState()
    heartbeat_interval = float(os.environ.get("HEYI_HEARTBEAT_INTERVAL", "14400"))  # 4h
    engine_remind_interval = float(os.environ.get("HEYI_ENGINE_REMIND_INTERVAL", "3600"))  # 1h
    print(f"loop mode (stub_only={args.stub_only}), ctrl-c to stop")

    while True:
        # Daily counter rollover
        today = _date.today()
        if today != state.today_date:
            state.completed_today = 0
            state.failed_today = 0
            state.today_date = today

        if not args.stub_only:
            _maybe_emit_heartbeat(state, cfg, store, interval_s=heartbeat_interval)

            # Pre-flight: don't pop a job from the queue if heyi_engine is
            # dead. The curator + showcase stages need it; better to leave
            # the job queued and re-probe in `idle_sleep` seconds than to
            # pull it, fail it, and have it disappear. (INV-5)
            allow, reason = _engine_preflight_gate(
                state, cfg, store, remind_interval_s=engine_remind_interval,
            )
            if not allow:
                if state.engine_unhealthy_since is not None and (
                    time.time() - state.engine_unhealthy_since < args.idle_sleep + 1
                ):
                    print(f"[loop] heyi_engine UNHEALTHY: {reason}. Pausing job intake.")
                time.sleep(args.idle_sleep)
                continue
            if state.engine_unhealthy_since is None and reason == "healthy" and \
               state.engine_last_notify == 0.0:
                pass  # no-op; just a probe-success branch

        item = pop_one(store)
        if not item:
            time.sleep(args.idle_sleep)
            continue
        run = Run(run_id=item["run_id"], hf_id=item["hf_id"])
        store.save_run(run)
        try:
            run_pipeline(run, store, cfg=cfg, stub_only=args.stub_only)
            if run.status == RunStatus.OK:
                state.completed_today += 1
            elif run.status in (RunStatus.FAILED, RunStatus.ABORTED):
                state.failed_today += 1
        except KeyboardInterrupt:
            print("interrupted")
            return 130
        except Exception as e:
            # Catastrophic crash inside run_pipeline. Make sure the run is
            # marked FAILED and a run_failed event lands in the outbox so the
            # operator hears about it (the previous behaviour was to leave
            # the run dangling at in_progress, which is the worst possible
            # outcome — silent stuck queue consumer).
            print(f"pipeline raised unexpectedly: {type(e).__name__}: {e}")
            state.failed_today += 1
            try:
                run.status = RunStatus.FAILED
                run.failure_reason = f"orchestrator crash: {type(e).__name__}: {e}"
                run.ended_at = time.time()
                store.save_run(run)
                notify.run_failed(
                    store.outbox_path,
                    run_id=run.run_id,
                    hf_id=run.hf_id,
                    stage="orchestrator",
                    error=f"{type(e).__name__}: {e}",
                )
            except Exception as e2:
                print(f"  also failed to record FAILED state: {e2}")


def cmd_status(args: argparse.Namespace) -> int:
    store = Store(_default_data_root())
    rows = store.list_runs(limit=args.limit)
    if not rows:
        print("no runs yet")
        return 0
    print(f"{'run_id':<40}{'hf_id':<40}{'status':<12}{'duration':<10}")
    for r in rows:
        dur = ""
        if r["ended_at"]:
            dur = f"{(r['ended_at'] - r['created_at']):.1f}s"
        print(f"{r['run_id']:<40}{r['hf_id']:<40}{r['status']:<12}{dur:<10}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    cfg = OrchestratorConfig()
    store = Store(_default_data_root())
    recovered = store.recover_in_progress()
    if not recovered:
        print("no IN_PROGRESS runs to resume")
        return 0
    for run in recovered:
        print(f"resuming: {run.run_id} ({run.hf_id})")
        run_pipeline(run, store, cfg=cfg, stub_only=args.stub_only)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="heyi-eval-orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("enqueue")
    s1.add_argument("hf_id")
    s1.set_defaults(fn=cmd_enqueue)

    s2 = sub.add_parser("run")
    s2.add_argument("--stub-only", action="store_true",
                    help="Use in-process stubs instead of CC agent docker invocations")
    s2.set_defaults(fn=cmd_run)

    s3 = sub.add_parser("loop")
    s3.add_argument("--idle-sleep", type=float, default=5.0)
    s3.add_argument("--stub-only", action="store_true",
                    help="Use in-process stubs instead of CC agent docker invocations")
    s3.set_defaults(fn=cmd_loop)

    s4 = sub.add_parser("status")
    s4.add_argument("--limit", type=int, default=20)
    s4.set_defaults(fn=cmd_status)

    s5 = sub.add_parser("resume")
    s5.add_argument("--stub-only", action="store_true",
                    help="Use in-process stubs instead of CC agent docker invocations")
    s5.set_defaults(fn=cmd_resume)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
