"""
10-stage state machine with mixed checkpoint policy.

Stage list (in order):
    DISCOVER  CURATE  METADATA  ENGINE_SELECT               <- run-level checkpoint
    DEPLOY  READY_WAIT  CAPABILITY  PERF_BENCH  SHOWCASE  CLEANUP  <- stage-level checkpoint

Checkpoint semantics:
- run-level stages: if any fails, restart the whole run from DISCOVER on retry
- stage-level stages: if any fails, retry from the last OK stage (DEPLOY result is preserved)
  This avoids wasting 60-120s of vllm boot if CAPABILITY/SHOWCASE fail.
- CLEANUP is idempotent and always runs (best-effort) on terminal transitions.

State is persisted to runs/<run_id>/state.json after every transition.
Process can crash and recover by scanning state.json files.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class StageName(str, Enum):
    DISCOVER = "DISCOVER"
    CURATE = "CURATE"
    METADATA = "METADATA"
    ENGINE_SELECT = "ENGINE_SELECT"
    DEPLOY = "DEPLOY"
    READY_WAIT = "READY_WAIT"
    CAPABILITY = "CAPABILITY"
    PERF_BENCH = "PERF_BENCH"
    SHOWCASE = "SHOWCASE"
    CLEANUP = "CLEANUP"


STAGES_IN_ORDER: list[StageName] = [
    StageName.DISCOVER,
    StageName.CURATE,
    StageName.METADATA,
    StageName.ENGINE_SELECT,
    StageName.DEPLOY,
    StageName.READY_WAIT,
    StageName.CAPABILITY,
    StageName.PERF_BENCH,
    StageName.SHOWCASE,
    StageName.CLEANUP,
]

RUN_LEVEL_STAGES: set[StageName] = {
    StageName.DISCOVER,
    StageName.CURATE,
    StageName.METADATA,
    StageName.ENGINE_SELECT,
}

STAGE_LEVEL_STAGES: set[StageName] = {
    StageName.DEPLOY,
    StageName.READY_WAIT,
    StageName.CAPABILITY,
    StageName.PERF_BENCH,
    StageName.SHOWCASE,
    StageName.CLEANUP,
}


class StageStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    TIMED_OUT = "timed_out"


class RunStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    OK = "ok"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class StageInfo:
    name: StageName
    status: StageStatus = StageStatus.PENDING
    started_at: float | None = None
    ended_at: float | None = None
    duration_s: float | None = None
    attempt: int = 0
    error: str | None = None
    artifacts: list[str] = field(default_factory=list)

    def mark_started(self) -> None:
        self.status = StageStatus.IN_PROGRESS
        self.started_at = time.time()
        self.attempt += 1

    def mark_ok(self, artifacts: list[str] | None = None) -> None:
        self.status = StageStatus.OK
        self.ended_at = time.time()
        if self.started_at is not None:
            self.duration_s = round(self.ended_at - self.started_at, 2)
        if artifacts:
            self.artifacts = list(artifacts)

    def mark_failed(self, error: str, *, timed_out: bool = False) -> None:
        self.status = StageStatus.TIMED_OUT if timed_out else StageStatus.FAILED
        self.ended_at = time.time()
        if self.started_at is not None:
            self.duration_s = round(self.ended_at - self.started_at, 2)
        self.error = error[:500]

    def mark_skipped(self, reason: str) -> None:
        """Graceful-skip: the stage decided in advance that it can't run
        in the current environment (insufficient GPU, eval pool overlaps
        production, etc.) and there's no point retrying. Distinct from
        mark_failed (which signals retry-worthy failure)."""
        self.status = StageStatus.SKIPPED
        self.ended_at = time.time()
        if self.started_at is not None:
            self.duration_s = round(self.ended_at - self.started_at, 2)
        self.error = reason[:500]


@dataclass
class Run:
    run_id: str
    hf_id: str
    status: RunStatus = RunStatus.PENDING
    created_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    stages: dict[str, StageInfo] = field(default_factory=dict)
    failure_reason: str | None = None
    abort_flag: bool = False

    def __post_init__(self) -> None:
        if not self.stages:
            self.stages = {s.value: StageInfo(name=s) for s in STAGES_IN_ORDER}

    def get_stage(self, stage: StageName) -> StageInfo:
        return self.stages[stage.value]

    def first_pending_stage(self) -> StageName | None:
        """The first stage that is not OK (used for resume after crash)."""
        for stage in STAGES_IN_ORDER:
            info = self.get_stage(stage)
            if info.status != StageStatus.OK:
                return stage
        return None

    def needs_full_restart(self) -> bool:
        """
        If any RUN_LEVEL stage failed and we have not yet entered DEPLOY,
        full restart is allowed (cheap). After DEPLOY, we keep partial progress.
        """
        deploy_ok = self.get_stage(StageName.DEPLOY).status == StageStatus.OK
        if deploy_ok:
            return False
        for stage in RUN_LEVEL_STAGES:
            if self.get_stage(stage).status in (StageStatus.FAILED, StageStatus.TIMED_OUT):
                return True
        return False

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["stages"] = {
            name: {
                **{k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(info).items()},
            }
            for name, info in self.stages.items()
        }
        d["status"] = self.status.value
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Run:
        stages_raw = d.pop("stages", {})
        run = cls(
            run_id=d["run_id"],
            hf_id=d["hf_id"],
            status=RunStatus(d.get("status", RunStatus.PENDING.value)),
            created_at=d.get("created_at", time.time()),
            ended_at=d.get("ended_at"),
            failure_reason=d.get("failure_reason"),
            abort_flag=d.get("abort_flag", False),
        )
        for name, info_d in stages_raw.items():
            try:
                stage_enum = StageName(name)
            except ValueError:
                continue
            run.stages[name] = StageInfo(
                name=stage_enum,
                status=StageStatus(info_d.get("status", StageStatus.PENDING.value)),
                started_at=info_d.get("started_at"),
                ended_at=info_d.get("ended_at"),
                duration_s=info_d.get("duration_s"),
                attempt=info_d.get("attempt", 0),
                error=info_d.get("error"),
                artifacts=list(info_d.get("artifacts", [])),
            )
        return run


def state_file_for(runs_root: Path, run_id: str) -> Path:
    return runs_root / run_id / "state.json"


def save_state(runs_root: Path, run: Run) -> None:
    """Atomic write of state.json (write to tmp + rename)."""
    target = state_file_for(runs_root, run.run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(run.to_json(), indent=2, ensure_ascii=False))
    tmp.replace(target)


def load_state(runs_root: Path, run_id: str) -> Run | None:
    p = state_file_for(runs_root, run_id)
    if not p.exists():
        return None
    return Run.from_json(json.loads(p.read_text()))


def scan_recoverable_runs(runs_root: Path) -> list[Run]:
    """Find all runs in IN_PROGRESS state (recover after crash)."""
    out: list[Run] = []
    if not runs_root.exists():
        return out
    for state_p in runs_root.glob("*/state.json"):
        try:
            run = Run.from_json(json.loads(state_p.read_text()))
            if run.status == RunStatus.IN_PROGRESS:
                out.append(run)
        except Exception:
            continue
    return out
