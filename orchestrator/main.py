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
from typing import Any

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
    from .config import _DEFAULT_LLM_PROVIDER
    provider = (
        os.environ.get("HEYI_EVAL_JUDGE_PROVIDER") or _DEFAULT_LLM_PROVIDER
    ).strip().lower()
    if provider in ("zhipu", "yunwu") and probe is None:
        # Cloud LLM is a managed service; the ``/v1/models`` liveness
        # probe isn't a reliable readiness signal (Zhipu's v4 endpoint
        # may not expose ``/models``), and each stage already degrades
        # gracefully on a transient upstream error. Don't gate intake on
        # it — otherwise a quirky ``/models`` response would silently
        # freeze the whole queue.
        if state.engine_unhealthy_since is not None:
            state.engine_unhealthy_since = None
            state.engine_last_notify = 0.0
        return True, "cloud-provider"
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


def _has_recent_failed_run(
    store: Store, hf_id: str, *, within_hours: float = 12.0,
) -> bool:
    """PR#38 (post-PR#36 nv8 observation): has this hf_id recently
    failed or been aborted? Used by enqueue() to dampen hourly cron
    re-tries of models that always fail fast.

    Live nv8 data: discover/auto-enqueue brings ``gemma-4-26B-it-GGUF``,
    ``Andycurrent/Gemma-3-1B-...-GGUF``, ``Qwen3-Coder-Next-GGUF`` back
    to the queue every hour. Each gets re-staged (sometimes 13 GB of
    download, then deleted on failure) and re-deployed for ~30s before
    failing identically. Five hours, ~70 GB disk consumed for the same
    three failures over and over.

    We only block on the *last* status to allow PR#33-style auto-repair
    to retry once after a manual fix lands. Default window is 12 h —
    long enough for a fix-PR turnaround, short enough that abandoned
    models eventually re-enter the queue.
    """
    cutoff = time.time() - within_hours * 3600.0
    for r in store.list_runs(limit=500):
        if r["hf_id"] != hf_id:
            continue
        if r["created_at"] < cutoff:
            continue
        # Most recent run wins (list_runs is ordered DESC by created_at)
        return r["status"] in ("failed", "aborted")
    return False


def _hf_ids_in_queue(store: Store) -> set[str]:
    """Read queue.jsonl and return the set of hf_ids currently pending."""
    qp = queue_path(store)
    if not qp.exists():
        return set()
    out: set[str] = set()
    import json as _json
    for raw in qp.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.add(_json.loads(raw).get("hf_id", ""))
        except Exception:
            continue
    out.discard("")
    return out


def _hf_ids_in_progress(store: Store) -> set[str]:
    """Return the set of hf_ids currently active in the DB.

    Includes both ``pending`` and ``in_progress`` since both consume a
    slot and would cause duplicate work if re-enqueued. We deliberately
    exclude terminal states (``ok``, ``failed``, ``aborted``) — a
    failed run is allowed to be retried by re-enqueue.
    """
    rows = store.list_runs(limit=500)
    return {
        r["hf_id"]
        for r in rows
        if r["status"] in ("in_progress", "pending")
    }


def enqueue(store: Store, hf_id: str, *, skip_if_recent: bool = False) -> str | None:
    """Add an hf_id to the queue. Returns the new run_id, or None if:
    - skip_if_recent was True and we already have a recent OK run, OR
    - the same hf_id is already pending in the queue or currently
      in_progress / queued (PR#34b dedup).

    The dedup is unconditional (independent of skip_if_recent) because
    enqueueing the same model twice always wastes a download and a GPU
    slot — we observed this when the hourly auto-enqueue timer pushed
    the same hf_id 4× before the orchestrator finished the first run.
    """
    pending = _hf_ids_in_queue(store) | _hf_ids_in_progress(store)
    if hf_id in pending:
        print(f"skip (dedup): {hf_id} already pending/in_progress")
        return None
    if skip_if_recent and _has_recent_successful_run(store, hf_id):
        return None
    # PR#38: dampen hourly cron retries of models that failed within
    # the last 12h. Honor the same `skip_if_recent` flag (only applies
    # to the cron/auto-discover path; manual `enqueue` CLI calls pass
    # skip_if_recent=False and bypass this).
    if skip_if_recent and _has_recent_failed_run(store, hf_id):
        print(f"skip (recent-fail): {hf_id} failed/aborted in last 12h")
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


def _sweep_orphan_in_progress(
    store: Store, *, stale_after_s: float = 1800.0,
    force_all: bool = False,
) -> int:
    """Mark stale 'in_progress' runs as aborted on orchestrator startup.

    PR#34d: when the orchestrator process is killed mid-run (systemd
    restart, OOM, manual stop), the active run's row in runs.sqlite
    stays 'in_progress' forever, the cache directory leaks (we saw
    142 GB of WanVideo_comfy left behind), and worse — duplicate
    enqueue dedup thinks the model is still being worked on. Sweep
    any in_progress run that hasn't seen a heartbeat update for
    longer than `stale_after_s` and re-mark it ABORTED with reason
    'orchestrator_restart_orphan'.

    PR#44 (2026-05-24 nv8): the 30-min stale_after_s default was too
    conservative — restarting the orchestrator within 7 min of an
    in-flight STAGE_MODEL left the row in_progress, which blocked
    PR#38 dedup from re-enqueuing the model. ``force_all=True`` (the
    startup-time path) skips the time check entirely: by definition,
    anything still ``in_progress`` when the orchestrator process has
    just started cannot be making progress, because the orchestrator
    is the only writer.

    Returns the number of runs swept.
    """
    now = time.time()
    swept = 0
    for r in store.list_runs(status=RunStatus.IN_PROGRESS, limit=200):
        # Heartbeat = most recent of created_at / updated_at / ended_at.
        # ended_at should be NULL for in_progress, so use the max of
        # the others. We also defensively fall back to started_at.
        candidates = [
            r.get("updated_at") or 0.0,
            r.get("created_at") or 0.0,
            r.get("started_at") or 0.0,
        ]
        last_touch = max(float(x) for x in candidates if x)
        if not force_all and last_touch and (now - last_touch) < stale_after_s:
            continue  # still fresh, leave alone
        run = store.get_run(r["run_id"])
        if run is None:
            continue
        run.status = RunStatus.ABORTED
        run.failure_reason = "orchestrator_restart_orphan"
        run.ended_at = now
        store.save_run(run)
        swept += 1
        age = f"stale_for={now - last_touch:.0f}s" if last_touch else "no-touch"
        print(
            f"[sweep] orphan in_progress -> aborted: {r['run_id']} "
            f"({r['hf_id']}) {age}"
            + (" [force_all]" if force_all else "")
        )
    return swept


def cmd_loop(args: argparse.Namespace) -> int:
    from datetime import date as _date
    cfg = OrchestratorConfig()
    store = Store(_default_data_root())
    state = LoopState()
    heartbeat_interval = float(os.environ.get("HEYI_HEARTBEAT_INTERVAL", "14400"))  # 4h
    engine_remind_interval = float(os.environ.get("HEYI_ENGINE_REMIND_INTERVAL", "3600"))  # 1h

    # PR#44: at process startup nothing else can be writing in_progress,
    # so unconditionally sweep all of them. The `stale_after_s` knob is
    # kept for any future in-loop sweep call (none today).
    stale_after = float(os.environ.get("HEYI_ORPHAN_STALE_S", "1800"))
    n_swept = _sweep_orphan_in_progress(
        store, stale_after_s=stale_after, force_all=True,
    )
    if n_swept:
        print(f"[startup] swept {n_swept} orphan in_progress run(s)")

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


# ── project_lane CLI handlers (M4) ───────────────────────────────────────
#
# These are intentionally thin: real logic lives in
# ``orchestrator.project_lane``. The wrappers do three things:
#
#   1. Resolve a real docker client if we're actually going to dispatch
#      ("run" subcommand). For "enqueue" and "status" we don't need docker.
#   2. Construct a PoolManager + StreamExec from the wired-up M2d helpers.
#   3. Translate the lane's status enum to a stdout summary.
#
# Why no daemon subcommand here: the systemd unit file is the right shape
# for that; we deliberately don't bake one into Python to keep the model
# of "one process, one round" cleaner for operators reading logs.


def _project_store():
    from .project_lane import ProjectStore
    return ProjectStore(_default_data_root())


def cmd_project_enqueue(args: argparse.Namespace) -> int:
    """Manually enqueue one github project by ``owner/repo``.

    Bypasses radar_ingest — useful for drills and for one-off operator
    requests ("evaluate this specific repo now"). Uses a synthetic
    ProjectCandidate with reason='manual' so the audit trail records
    that the request didn't come from the daily ingest.
    """
    from datetime import datetime as _dt

    from discover.radar_ingest import ProjectCandidate
    store = _project_store()
    cand = ProjectCandidate(
        full_id=args.full_id,
        source_url=f"https://github.com/{args.full_id}",
        source_report="manual-cli",
        source_date=_dt.now(UTC).strftime("%Y-%m-%d"),
        discovered_at=_dt.now(UTC).isoformat(),
        reason="manual",
        short_desc="(manually enqueued via CLI)",
    )
    run_id = store.enqueue(cand, skip_if_recent_hours=0.0)
    if run_id is None:
        print(f"project enqueue {args.full_id}: dedup-skipped (recent run found)")
        return 0
    print(f"project enqueued {args.full_id} -> {run_id}")
    return 0


def _resolve_m2b_host_workspace(
    client: Any, container_name: str, env_override: str | None,
) -> Path:
    """Return the host-side directory that is mounted into ``container_name``
    at ``/home/agent/workspace``.

    When the operator sets ``$HEYI_PROJECT_WORKSPACE`` / ``$HEYI_SKILL_WORKSPACE``
    explicitly, we trust that value (used by tests / sandbox drills).
    Otherwise we inspect the running container's mounts and pick the
    one whose Destination matches the agent workspace inside the
    container — that's the path PoolManager has to ``mkdir`` so the
    container's ``cd /home/agent/workspace/<run_id>`` resolves.

    Falling back to a hard-coded ``/tmp/heyi-*-ws`` (the previous
    default) silently broke ``orchestrator project run`` on heyi
    because the m2b container's actual mount is
    ``/home/ai/nanchang-demos/m2b-claude-code/workspace``; OCI then
    rejected every exec with ``chdir to cwd … no such file or
    directory`` and every run died in <1s.
    """
    if env_override:
        return Path(env_override)
    try:
        info = client.api.inspect_container(container_name)
    except Exception as e:
        raise RuntimeError(
            f"docker inspect {container_name!r} failed: {e}"
        ) from e
    mounts = info.get("Mounts") or []
    for m in mounts:
        if m.get("Destination") == "/home/agent/workspace":
            src = m.get("Source")
            if src:
                return Path(src)
    raise RuntimeError(
        f"container {container_name!r} has no mount with destination "
        "/home/agent/workspace; either fix the m2b compose file or set "
        "HEYI_PROJECT_WORKSPACE / HEYI_SKILL_WORKSPACE explicitly"
    )


def cmd_project_run(args: argparse.Namespace) -> int:
    """Execute up to ``--limit`` pending project_lane runs in series.

    Requires docker daemon + the m2b container named ``--container``.
    Fails cleanly with rc=2 if docker isn't reachable so systemd notices.
    """
    try:
        import docker
    except ImportError:
        print("project run: 'docker' python package not installed", file=sys.stderr)
        return 2
    try:
        client = docker.from_env()
        client.ping()
    except Exception as e:
        print(f"project run: docker not reachable: {e}", file=sys.stderr)
        return 2

    from agent_driver.exec_runner import make_docker_stream_exec
    from agent_driver.pool_manager import PoolManager

    from .project_lane import run_pending as project_run_pending
    store = _project_store()
    try:
        workspace = _resolve_m2b_host_workspace(
            client, args.container,
            os.environ.get("HEYI_PROJECT_WORKSPACE"),
        )
    except RuntimeError as e:
        print(f"project run: {e}", file=sys.stderr)
        return 2
    workspace.mkdir(parents=True, exist_ok=True)
    pool = PoolManager(
        container_names=[args.container],
        host_workspace_root=workspace,
        docker_client=client,
    )
    n = project_run_pending(
        store, pool=pool, host_workspace_root=workspace,
        limit=args.limit,
        stream_exec=make_docker_stream_exec(client),
    )
    print(f"project run: processed {n} pending runs")
    return 0


def cmd_project_status(args: argparse.Namespace) -> int:
    """Tail of recent project_lane runs, panel-shaped (status / outcome /
    full_id / age). Useful for `watch` during a drill."""
    from .project_lane import list_recent as project_list_recent
    store = _project_store()
    rows = project_list_recent(store, limit=args.limit)
    if not rows:
        print("project status: no runs yet")
        return 0
    print(f"{'run_id':<28} {'status':<22} {'outcome':<8} {'full_id':<40} enqueued_at")
    for r in rows:
        print(
            f"{r.run_id:<28} {r.status.value:<22} "
            f"{(r.summary_outcome or '-'):<8} {r.full_id:<40} {r.enqueued_at}"
        )
    return 0


# ── skill_lane CLI handlers (M4) ─────────────────────────────────────────


def _skill_store():
    from .skill_lane import SkillStore
    return SkillStore(_default_data_root())


def cmd_skill_enqueue(args: argparse.Namespace) -> int:
    """Manually enqueue one skill by full_id. If ``--source-path`` is
    not given, we resolve from the candidate list (which discover/
    skill_local_scan emits to data/discover/skill_candidates.jsonl)."""
    from discover.skill_local_scan import (
        SkillCandidate,
        default_skill_candidates_path,
    )
    src_path = args.source_path
    if src_path is None:
        cand_file = default_skill_candidates_path()
        if not cand_file.exists():
            print(
                "skill enqueue: --source-path not given and no "
                f"{cand_file} on disk; run skill scan first or "
                "pass --source-path explicitly", file=sys.stderr,
            )
            return 2
        for line in cand_file.read_text().splitlines():
            try:
                c = SkillCandidate.from_jsonl(line)
            except Exception:
                continue
            if c.full_id == args.full_id:
                src_path = c.source_path
                break
    if not src_path:
        print(f"skill enqueue: {args.full_id} not in candidates", file=sys.stderr)
        return 2

    cand = SkillCandidate(
        full_id=args.full_id,
        source_path=src_path,
        source_root=args.full_id.split("/", 1)[0],
        discovered_at=datetime.now(UTC).isoformat(),
        name=args.full_id.split("/", 1)[-1],
        reason="manual",
    )
    store = _skill_store()
    run_id = store.enqueue(cand, skip_if_recent_hours=0.0)
    if run_id is None:
        print(f"skill enqueue {args.full_id}: dedup-skipped")
        return 0
    print(f"skill enqueued {args.full_id} -> {run_id}")
    return 0


def cmd_skill_run(args: argparse.Namespace) -> int:
    """Execute pending skill_lane runs. Same docker prereqs as
    cmd_project_run."""
    try:
        import docker
    except ImportError:
        print("skill run: 'docker' python package not installed", file=sys.stderr)
        return 2
    try:
        client = docker.from_env()
        client.ping()
    except Exception as e:
        print(f"skill run: docker not reachable: {e}", file=sys.stderr)
        return 2

    from agent_driver.exec_runner import make_docker_stream_exec
    from agent_driver.pool_manager import PoolManager

    from .skill_lane import run_pending as skill_run_pending
    store = _skill_store()
    try:
        workspace = _resolve_m2b_host_workspace(
            client, args.container,
            os.environ.get("HEYI_SKILL_WORKSPACE"),
        )
    except RuntimeError as e:
        print(f"skill run: {e}", file=sys.stderr)
        return 2
    workspace.mkdir(parents=True, exist_ok=True)
    pool = PoolManager(
        container_names=[args.container],
        host_workspace_root=workspace,
        docker_client=client,
    )
    n = skill_run_pending(
        store, pool=pool, host_workspace_root=workspace,
        limit=args.limit,
        stream_exec=make_docker_stream_exec(client),
    )
    print(f"skill run: processed {n} pending runs")
    return 0


def cmd_skill_status(args: argparse.Namespace) -> int:
    from .skill_lane import list_recent as skill_list_recent
    store = _skill_store()
    rows = skill_list_recent(store, limit=args.limit)
    if not rows:
        print("skill status: no runs yet")
        return 0
    print(f"{'run_id':<30} {'status':<22} {'outcome':<8} {'full_id':<50} demos")
    for r in rows:
        print(
            f"{r.run_id:<30} {r.status.value:<22} "
            f"{(r.summary_outcome or '-'):<8} {r.full_id:<50} {r.summary_demos or 0}"
        )
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

    # ── project_lane subcommands (M4) ───────────────────────────────────
    # Namespaced under ``project`` so model_lane CLI surface is untouched
    # (INV-L0) and lane intent is obvious in operator logs.
    proj = sub.add_parser(
        "project", help="project_lane operations (github project evals via M2.7)",
    )
    proj_sub = proj.add_subparsers(dest="project_cmd", required=True)

    pp_enq = proj_sub.add_parser(
        "enqueue", help="enqueue one github repo by owner/name",
    )
    pp_enq.add_argument("full_id", help="github full id, e.g. simonw/llm")
    pp_enq.set_defaults(fn=cmd_project_enqueue)

    pp_run = proj_sub.add_parser(
        "run", help="execute one round of pending project runs",
    )
    pp_run.add_argument("--limit", type=int, default=5)
    pp_run.add_argument(
        "--container", default=os.environ.get("HEYI_AGENT_CONTAINER", "m2b-1"),
        help="m2b container to dispatch through (default $HEYI_AGENT_CONTAINER or m2b-1)",
    )
    pp_run.set_defaults(fn=cmd_project_run)

    pp_status = proj_sub.add_parser(
        "status", help="recent project_lane runs (panel-shaped output)",
    )
    pp_status.add_argument("--limit", type=int, default=20)
    pp_status.set_defaults(fn=cmd_project_status)

    # ── skill_lane subcommands (M4) ─────────────────────────────────────
    sk = sub.add_parser(
        "skill", help="skill_lane operations (SKILL.md evals via M2.7)",
    )
    sk_sub = sk.add_subparsers(dest="skill_cmd", required=True)

    sk_enq = sk_sub.add_parser(
        "enqueue", help="enqueue one skill by full_id (sourceLabel/slug)",
    )
    sk_enq.add_argument("full_id", help="e.g. claude-user/agent-development")
    sk_enq.add_argument(
        "--source-path",
        help="absolute path to SKILL.md (default: derived from full_id under "
             "$HEYI_CLAUDE_SKILLS_ROOT or $HEYI_CURSOR_SKILLS_ROOT)",
    )
    sk_enq.set_defaults(fn=cmd_skill_enqueue)

    sk_run = sk_sub.add_parser(
        "run", help="execute one round of pending skill runs",
    )
    sk_run.add_argument("--limit", type=int, default=5)
    sk_run.add_argument(
        "--container", default=os.environ.get("HEYI_AGENT_CONTAINER", "m2b-1"),
    )
    sk_run.set_defaults(fn=cmd_skill_run)

    sk_status = sk_sub.add_parser(
        "status", help="recent skill_lane runs",
    )
    sk_status.add_argument("--limit", type=int, default=20)
    sk_status.set_defaults(fn=cmd_skill_status)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
