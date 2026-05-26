"""skill_lane orchestrator (M3): store + stages + main loop.

Mirrors ``orchestrator.project_lane`` structurally so the panel /
queue manager can treat the two lanes the same way at the surface,
but with three substantive differences in the pipeline shape:

  1. **No PREFLIGHT**: skills are local files we already inspected
     during discover; there's no remote metadata to check. We do a
     lightweight LOCAL_GATE instead (file still exists, body length
     reasonable, no shebang of doom).
  2. **No git clone permitted (INV-S1)**: the agent must demonstrate
     the skill on a stock sandbox using only the SKILL.md content. We
     don't let the agent ``git clone <some-repo>`` because that turns a
     skill eval into a project eval. Enforcement is twofold:
       - the task prompt forbids it explicitly
       - the report post-processor scans agent.log for
         ``git clone`` / ``gh repo clone`` invocations and downgrades
         OUTCOME=PASS → OUTCOME=SKILL_CLONE_ATTEMPT
  3. **Task prompt** is skill-shaped: read SKILL.md, identify the
     skill's "I help with X" intent, generate ≥2 plausible user
     requests that should trigger the skill, walk through how the
     skill would respond, and emit fenced JSON.

INV-L0 holds: this module imports from ``agent_driver`` and from
``discover.skill_local_scan``, never from ``orchestrator.project_lane``
or ``orchestrator.store`` (model_lane). Both lanes are siblings.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from agent_driver.budget_guard import BudgetGuard
from agent_driver.exec_runner import AgentRunResult, run_agent
from agent_driver.pool_manager import PoolManager
from agent_driver.schema import FENCE_CLOSE, FENCE_OPEN, Outcome, StepStatus
from discover.skill_local_scan import SkillCandidate

log = logging.getLogger(__name__)


# ── status enum ───────────────────────────────────────────────────────────


class SkillStatus(str, Enum):
    PENDING = "pending"
    LOCAL_GATE = "local_gate"
    AGENT_RUN = "agent_run"
    JUDGE = "judge"
    DONE = "done"

    # Local gate terminal
    SKILL_FILE_MISSING = "skill_file_missing"
    SKILL_FILE_EMPTY = "skill_file_empty"

    # Post-execution terminal
    SKILL_CLONE_ATTEMPT = "skill_clone_attempt"  # INV-S1 violation
    REPORT_PARSE_ERROR = "report_parse_error"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    SANDBOX_DEAD = "sandbox_dead"


_TERMINAL = {
    SkillStatus.DONE,
    SkillStatus.SKILL_FILE_MISSING,
    SkillStatus.SKILL_FILE_EMPTY,
    SkillStatus.SKILL_CLONE_ATTEMPT,
    SkillStatus.REPORT_PARSE_ERROR,
    SkillStatus.BUDGET_EXCEEDED,
    SkillStatus.TIMEOUT,
    SkillStatus.SANDBOX_DEAD,
}


# ── data model ───────────────────────────────────────────────────────────


@dataclass
class SkillRun:
    """One skill-lane execution attempt.

    ``run_id`` format: ``skill-YYYYMMDD-<6char-uuid>`` for grep-friendly
    co-existence with project_lane's ``proj-...`` ids and model_lane's
    own scheme.
    """
    run_id: str
    full_id: str  # e.g. "claude-user/agent-development"
    skill_path: str
    enqueued_at: str
    candidate: dict[str, Any]
    status: SkillStatus = SkillStatus.PENDING
    started_at: str | None = None
    ended_at: str | None = None
    failure_reason_zh: str = ""

    # Artifacts
    agent_log: str | None = None
    report_json: str | None = None
    bad_report_txt: str | None = None
    clone_evidence: str | None = None

    # Summary
    summary_outcome: str | None = None
    summary_demos: int | None = None  # how many demo scenarios the agent ran

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> SkillRun:
        kwargs = dict(obj)
        kwargs["status"] = SkillStatus(obj.get("status", "pending"))
        return cls(**kwargs)


# ── store ─────────────────────────────────────────────────────────────────


_SCHEMA = """
CREATE TABLE IF NOT EXISTS skill_runs (
    run_id          TEXT PRIMARY KEY,
    full_id         TEXT NOT NULL,
    status          TEXT NOT NULL,
    enqueued_at     TEXT NOT NULL,
    ended_at        TEXT,
    summary_outcome TEXT,
    state_path      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skill_runs_status   ON skill_runs(status);
CREATE INDEX IF NOT EXISTS idx_skill_runs_full_id  ON skill_runs(full_id);
CREATE INDEX IF NOT EXISTS idx_skill_runs_enqueued ON skill_runs(enqueued_at);
"""


class SkillStore:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.expanduser().resolve()
        self.lane_root = self.data_root / "skill_lane"
        self.lane_root.mkdir(parents=True, exist_ok=True)
        (self.lane_root / "runs").mkdir(exist_ok=True)
        self._db = self.lane_root / "runs.sqlite"
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Any:
        conn = sqlite3.connect(str(self._db), timeout=15.0)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        finally:
            conn.close()

    def enqueue(
        self, cand: SkillCandidate,
        *,
        skip_if_recent_hours: float = 24.0,
    ) -> str | None:
        if self._recently_completed(cand.full_id, hours=skip_if_recent_hours):
            return None
        run_id = self._new_run_id()
        now_iso = datetime.now(UTC).isoformat()
        run = SkillRun(
            run_id=run_id,
            full_id=cand.full_id,
            skill_path=cand.source_path,
            enqueued_at=now_iso,
            candidate=asdict(cand),
        )
        self._write_state(run)
        with self._conn() as c:
            c.execute(
                "INSERT INTO skill_runs "
                "(run_id, full_id, status, enqueued_at, state_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, cand.full_id, run.status.value,
                 now_iso, str(self._state_path(run_id))),
            )
        return run_id

    def _recently_completed(self, full_id: str, *, hours: float) -> bool:
        cutoff_ts = time.time() - hours * 3600
        cutoff_iso = datetime.fromtimestamp(cutoff_ts, UTC).isoformat()
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM skill_runs WHERE full_id = ? "
                "AND enqueued_at > ? AND status = 'done' LIMIT 1",
                (full_id, cutoff_iso),
            ).fetchone()
        return row is not None

    def list_pending(self, *, limit: int = 100) -> list[SkillRun]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT run_id FROM skill_runs WHERE status = 'pending' "
                "ORDER BY enqueued_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self.load(r["run_id"]) for r in rows]

    def load(self, run_id: str) -> SkillRun:
        path = self._state_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"no state.json for run_id={run_id}")
        return SkillRun.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def update_status(
        self, run: SkillRun, status: SkillStatus,
        *, failure_reason_zh: str = "",
    ) -> None:
        run.status = status
        if failure_reason_zh:
            run.failure_reason_zh = failure_reason_zh
        if status == SkillStatus.AGENT_RUN and run.started_at is None:
            run.started_at = datetime.now(UTC).isoformat()
        if status in _TERMINAL:
            run.ended_at = datetime.now(UTC).isoformat()
        self._write_state(run)
        with self._conn() as c:
            c.execute(
                "UPDATE skill_runs SET status = ?, ended_at = ?, "
                "summary_outcome = ? WHERE run_id = ?",
                (status.value, run.ended_at, run.summary_outcome, run.run_id),
            )

    def write_report_artifacts(
        self, run: SkillRun, result: AgentRunResult,
    ) -> None:
        result.write_to(self._run_dir(run.run_id))
        run.agent_log = "agent.log"
        run.report_json = "report.json"
        if result.raw_bad_report:
            run.bad_report_txt = "bad_report.txt"
        run.summary_outcome = result.report.outcome.value
        run.summary_demos = len(result.report.verdict.core_features_demonstrated)
        self._write_state(run)

    def _run_dir(self, run_id: str) -> Path:
        return self.lane_root / "runs" / run_id

    def _state_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "state.json"

    def _write_state(self, run: SkillRun) -> None:
        d = self._run_dir(run.run_id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "state.json.tmp"
        tmp.write_text(
            json.dumps(run.to_jsonable(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(d / "state.json")

    @staticmethod
    def _new_run_id() -> str:
        now = datetime.now(UTC).strftime("%Y%m%d")
        return f"skill-{now}-{uuid.uuid4().hex[:6]}"


# ── local gate ───────────────────────────────────────────────────────────


@dataclass
class LocalGateResult:
    allow: bool
    status_if_blocked: SkillStatus | None
    reason_zh: str
    skill_md_text: str = ""


def local_gate(run: SkillRun, *, min_body_chars: int = 50) -> LocalGateResult:
    """Verify the SKILL.md still exists on disk and has non-trivial
    content. Discover happened earlier; the file could have been
    edited or deleted in between.
    """
    p = Path(run.skill_path)
    if not p.exists() or not p.is_file():
        return LocalGateResult(
            allow=False, status_if_blocked=SkillStatus.SKILL_FILE_MISSING,
            reason_zh=f"SKILL.md 不存在或已被移除：{run.skill_path}",
        )
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return LocalGateResult(
            allow=False, status_if_blocked=SkillStatus.SKILL_FILE_MISSING,
            reason_zh=f"无法读取 SKILL.md：{e}",
        )
    if len(text.strip()) < min_body_chars:
        return LocalGateResult(
            allow=False, status_if_blocked=SkillStatus.SKILL_FILE_EMPTY,
            reason_zh=f"SKILL.md 正文过短 ({len(text.strip())} <{min_body_chars} 字符)",
        )
    return LocalGateResult(
        allow=True, status_if_blocked=None, reason_zh="", skill_md_text=text,
    )


# ── task prompt ──────────────────────────────────────────────────────────


def build_task_prompt(run: SkillRun, skill_md_text: str) -> str:
    """Compose the M2.7 prompt for skill evaluation.

    Key INV-S1 enforcement clauses are inline in the prompt:
      - DO NOT git clone any repo
      - DO NOT download / install external skills
      - the only material you have is the SKILL.md text below

    Why include the full SKILL.md in the prompt rather than letting
    the agent read it from disk: cleaner audit trail. Whatever the
    agent reasoned about, it's verbatim what we pass; no race with
    disk edits between LOCAL_GATE and AGENT_RUN.
    """
    name = run.candidate.get("name", run.full_id)
    desc = run.candidate.get("description", "")
    return f"""# 任务：评测 Skill "{name}"

## Skill 元信息
- 全名：{run.full_id}
- 描述：{desc}
- 版本：{run.candidate.get('version') or '(未声明)'}
- 来源：{run.candidate.get('source_root', '?')}

## SKILL.md 全文

```
{skill_md_text}
```

## 你需要做的

按下列步骤评测这个 Skill 的实用性，每步在 stdout 输出 "[STEP X DONE]" 行：

1. **理解 Skill 意图**：读完上面的 SKILL.md，用 1 句话总结这个 Skill 在什么场景下应被触发，它声称能解决什么问题。
2. **生成触发场景**：列出 2-3 个具体、可观察的用户请求示例，每个都应该真实会触发这个 Skill。
3. **演练应答**：对每个触发场景，写一段你作为这个 Skill 的化身**应当**给出的回答（200-400 字），尽可能贴近 SKILL.md 描述的工作流。
4. **缺口分析**：列出这个 Skill 在 SKILL.md 描述之外的盲区（场景没覆盖、依赖外部工具、与其他 Skill 冲突等），至少 1 条。
5. **打分**：给这个 Skill 的"实际可用性"打 1-10 分并简单说明。

## 报告契约（INV-S3）

完成后在 stdout 末尾输出**且仅输出一次**：

```
{FENCE_OPEN}
{{
  "schema_version": "1.0",
  "lane": "skill",
  "target_id": "{run.full_id}",
  "outcome": "pass" | "partial" | "fail",
  "steps": [
    {{"name": "understand_intent", "status": "ok|fail|skip", "duration_s": 0.0}},
    {{"name": "trigger_scenarios", "status": "..."}},
    {{"name": "demo_responses", "status": "..."}},
    {{"name": "gap_analysis", "status": "..."}},
    {{"name": "scoring", "status": "..."}}
  ],
  "verdict": {{
    "deploys": true,
    "quickstart_works": true,
    "core_features_demonstrated": ["scenario_1", "scenario_2"],
    "blockers": []
  }},
  "self_assessment_zh": "1-2 句中文总结：这个 skill 是否真的有用、最大的缺口在哪",
  "follow_ups": ["改进建议（可选）"]
}}
{FENCE_CLOSE}
```

## 硬约束（INV-S1 — 不可违反）

- **绝对禁止**：``git clone`` / ``gh repo clone`` / ``svn checkout`` 任何外部仓库
- **绝对禁止**：``curl|bash`` / ``wget -O- | sh`` 下载并执行外部脚本
- **绝对禁止**：``pip install`` / ``npm install`` 额外依赖
- **绝对禁止**：从 ``/home/agent/.claude/skills`` 之外的路径读取其他 SKILL.md
- 评测材料**只有**上面方框里的 SKILL.md，不要去网上找补充资料
- ``outcome=pass`` 要求 ``len(core_features_demonstrated) >= 2``
- 如果 SKILL.md 信息不足以演练，老实写 ``outcome=fail`` + ``blockers``

预算：单 run 上限 30,000 tokens、墙钟 10 分钟。
"""


# ── INV-S1 post-processor ────────────────────────────────────────────────


# Regexes for clone-style commands in agent.log. Conservative + literal:
# we want few false positives (downgrading a legit run is annoying) but
# any of these constitutes a hard INV-S1 violation.
_CLONE_PATTERNS = (
    re.compile(r"\bgit\s+clone\b"),
    re.compile(r"\bgh\s+repo\s+clone\b"),
    re.compile(r"\bsvn\s+(?:checkout|co)\b"),
    re.compile(r"curl\s+[^|]+\s*\|\s*(?:bash|sh)\b"),
    re.compile(r"wget\s+[^|]+\s*\|\s*(?:bash|sh)\b"),
)


def detect_clone_attempt(agent_log_text: str) -> tuple[bool, str]:
    """Scan agent.log for clone/install signatures. Returns
    ``(violated, evidence_line)``. Evidence is the first matched line
    (capped to 240 chars) so the operator can spot-check before trusting
    the verdict downgrade.
    """
    for line in agent_log_text.splitlines():
        for pat in _CLONE_PATTERNS:
            if pat.search(line):
                return True, line[:240]
    return False, ""


# ── stage runner ─────────────────────────────────────────────────────────


def execute_run(
    store: SkillStore,
    run: SkillRun,
    *,
    pool: PoolManager,
    host_workspace_root: Path,
    stream_exec: Any | None = None,
    budget_guard: BudgetGuard | None = None,
) -> SkillRun:
    """LOCAL_GATE → AGENT_RUN → INV-S1 check → JUDGE → DONE."""
    log.info("skill_lane.execute_run %s (%s) start", run.run_id, run.full_id)

    # ── LOCAL_GATE ─────────────────────────────────────────────────────
    store.update_status(run, SkillStatus.LOCAL_GATE)
    gate = local_gate(run)
    if not gate.allow:
        assert gate.status_if_blocked is not None
        store.update_status(run, gate.status_if_blocked,
                            failure_reason_zh=gate.reason_zh)
        return run

    # ── AGENT_RUN ──────────────────────────────────────────────────────
    store.update_status(run, SkillStatus.AGENT_RUN)
    prompt = build_task_prompt(run, gate.skill_md_text)
    result = run_agent(
        pool=pool, lane="skill", target_id=run.full_id,
        run_id=run.run_id, task_prompt=prompt,
        budget_guard=budget_guard,
        stream_exec=stream_exec,
        host_workspace_root=host_workspace_root,
    )
    store.write_report_artifacts(run, result)

    # ── INV-S1 check on the raw agent log ───────────────────────────────
    # We run this AFTER report-extract so we have access to raw stdout
    # but still take action before declaring DONE.
    run_dir = store._run_dir(run.run_id)
    log_path = run_dir / "agent.log"
    if log_path.exists():
        violated, evidence = detect_clone_attempt(log_path.read_text(encoding="utf-8"))
        if violated:
            (run_dir / "clone_evidence.txt").write_text(evidence, encoding="utf-8")
            run.clone_evidence = "clone_evidence.txt"
            store.update_status(
                run, SkillStatus.SKILL_CLONE_ATTEMPT,
                failure_reason_zh=(
                    "Agent 在沙箱内尝试 clone/install 外部资源 (INV-S1 违规)"
                ),
            )
            log.warning("skill_lane.execute_run %s INV-S1 violation: %s",
                        run.run_id, evidence)
            return run

    # ── status mapping ────────────────────────────────────────────────
    terminal = _agent_outcome_to_status(result.report.outcome)
    if terminal != SkillStatus.DONE:
        store.update_status(
            run, terminal,
            failure_reason_zh=result.report.self_assessment_zh,
        )
        return run

    store.update_status(run, SkillStatus.JUDGE)
    store.update_status(run, SkillStatus.DONE)
    log.info("skill_lane.execute_run %s DONE outcome=%s",
             run.run_id, run.summary_outcome)
    return run


def _agent_outcome_to_status(outcome: Outcome) -> SkillStatus:
    return {
        Outcome.PASS: SkillStatus.DONE,
        Outcome.PARTIAL: SkillStatus.DONE,
        Outcome.FAIL: SkillStatus.DONE,
        Outcome.REPORT_PARSE_ERROR: SkillStatus.REPORT_PARSE_ERROR,
        Outcome.BUDGET_EXCEEDED: SkillStatus.BUDGET_EXCEEDED,
        Outcome.TIMEOUT: SkillStatus.TIMEOUT,
        Outcome.SANDBOX_DEAD: SkillStatus.SANDBOX_DEAD,
        # Project-only outcomes that "shouldn't" appear here — map
        # to DONE to avoid future enum drift crashes.
        Outcome.OVERSIZE: SkillStatus.DONE,
        Outcome.NO_README: SkillStatus.DONE,
        Outcome.NEEDS_GPU: SkillStatus.DONE,
        Outcome.JUDGE_UNAVAILABLE: SkillStatus.DONE,
        Outcome.SKILL_CLONE_ATTEMPT: SkillStatus.SKILL_CLONE_ATTEMPT,
    }.get(outcome, SkillStatus.DONE)


def run_pending(
    store: SkillStore,
    *, pool: PoolManager,
    host_workspace_root: Path,
    limit: int = 5,
    stream_exec: Any | None = None,
) -> int:
    completed = 0
    for run in store.list_pending(limit=limit):
        try:
            execute_run(
                store, run,
                pool=pool, host_workspace_root=host_workspace_root,
                stream_exec=stream_exec,
            )
            completed += 1
        except Exception as e:  # log + continue
            log.exception("skill_lane.run_pending fatal for %s: %s",
                          run.run_id, e)
            store.update_status(
                run, SkillStatus.SANDBOX_DEAD,
                failure_reason_zh=f"未预期异常：{type(e).__name__}: {e}",
            )
    return completed


def list_recent(store: SkillStore, *, limit: int = 50) -> list[SkillRun]:
    with store._conn() as c:
        rows = c.execute(
            "SELECT run_id FROM skill_runs ORDER BY enqueued_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [store.load(r["run_id"]) for r in rows]


# unused-import suppressor
_ = StepStatus
