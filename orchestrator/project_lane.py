"""project_lane orchestrator (M2b/c): store + stages + main loop.

This module is **physically isolated** from model_lane's
``orchestrator.store`` / ``orchestrator.stages_py`` to satisfy INV-L0:
"model_lane regression protection — any project_lane work that imports
from model_lane modules is fine, but model_lane code MUST NOT acquire a
new import from project_lane".

Layout under ``$HEYI_EVAL_DATA``:

    project_lane/
        runs.sqlite                 — run index, mirror of state.json status
        runs/<run_id>/
            state.json              — ProjectRun (recovery-of-truth)
            agent.log               — raw stdout from agent_driver
            report.json             — validated RunReport (or synthetic)
            bad_report.txt          — extract-error excerpt, only on REPORT_PARSE_ERROR
            budget.json             — BudgetGuard final state
            preflight.json          — github_api repo metadata snapshot

The five-stage flow is:

    PENDING → PREFLIGHT → AGENT_RUN → JUDGE → DONE
                    ↘ OVERSIZE / NO_README / NEEDS_GPU  (terminal)
                                ↘ BUDGET_EXCEEDED / TIMEOUT / SANDBOX_DEAD (terminal)
                                            ↘ JUDGE_UNAVAILABLE (terminal but with report archived)

Why an explicit state machine here rather than reusing ``state_machine.Run``?

- ``state_machine.Run`` carries hf_id-shaped fields (param_count, gpu
  topology, served-model name) that don't apply to a github repo.
- The PROJECT outcomes (oversize, no_readme, needs_gpu) are a different
  termination set than model_lane (which deals with deploy-timeout
  vs vllm-died vs metadata-only).
- Forcing both lanes through one Run dataclass means every model_lane
  schema migration risks breaking project_lane. KISS: separate dataclass.
"""
from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from agent_driver.budget_guard import BudgetGuard
from agent_driver.exec_runner import AgentRunResult, run_agent
from agent_driver.pool_manager import PoolManager
from agent_driver.schema import Outcome, RunReport, StepStatus, Verdict
from discover.radar_ingest import ProjectCandidate

log = logging.getLogger(__name__)


# ── status enum ───────────────────────────────────────────────────────────


class ProjectStatus(str, Enum):
    """ProjectRun lifecycle. Mirrors Outcome where overlapping, but
    also carries PENDING / RUNNING_* phases that don't appear in
    agent reports."""
    PENDING = "pending"
    PREFLIGHT = "preflight"
    AGENT_RUN = "agent_run"
    JUDGE = "judge"
    DONE = "done"

    # Terminal — pre-execution rejects (cheap, before agent_driver runs)
    OVERSIZE = "oversize"        # repo bigger than _MAX_REPO_MB
    NO_README = "no_readme"      # no README detected (INV-P1)
    NEEDS_GPU = "needs_gpu"      # README/topics flag GPU-only project (INV-P5)
    NOT_FOUND = "not_found"      # github API 404 — repo deleted between ingest + run
    PREFLIGHT_ERROR = "preflight_error"  # transient github API issue, retryable

    # Terminal — agent execution failures (post agent_driver)
    REPORT_PARSE_ERROR = "report_parse_error"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    SANDBOX_DEAD = "sandbox_dead"
    JUDGE_UNAVAILABLE = "judge_unavailable"


# Outcomes that bypass agent_run entirely
_PREFLIGHT_TERMINAL = {
    ProjectStatus.OVERSIZE,
    ProjectStatus.NO_README,
    ProjectStatus.NEEDS_GPU,
    ProjectStatus.NOT_FOUND,
}


# ── data model ────────────────────────────────────────────────────────────


@dataclass
class ProjectRun:
    """One project-lane execution attempt.

    Persisted as state.json under the run dir. ``status`` is the
    single source of truth; sqlite index mirrors it for query speed.

    ``run_id`` format: ``proj-YYYYMMDD-<6char-uuid>`` so log lines are
    grep-friendly and runs are time-orderable by lexicographic sort.
    """
    run_id: str
    full_id: str  # owner/repo
    source_url: str
    enqueued_at: str  # ISO8601 UTC
    candidate: dict[str, Any]  # serialised ProjectCandidate snapshot

    status: ProjectStatus = ProjectStatus.PENDING
    started_at: str | None = None
    ended_at: str | None = None
    failure_reason_zh: str = ""

    # Per-stage outputs (file paths, relative to run_dir)
    preflight_json: str | None = None
    agent_log: str | None = None
    report_json: str | None = None
    bad_report_txt: str | None = None
    judge_json: str | None = None

    # Lightweight inline summary so panel doesn't have to open report.json
    # for every row. Updated when AGENT_RUN finishes.
    summary_outcome: str | None = None
    summary_deploys: bool | None = None
    summary_quickstart: bool | None = None

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, obj: dict[str, Any]) -> ProjectRun:
        kwargs = dict(obj)
        kwargs["status"] = ProjectStatus(obj.get("status", "pending"))
        return cls(**kwargs)


# ── store ─────────────────────────────────────────────────────────────────


_SCHEMA = """
CREATE TABLE IF NOT EXISTS project_runs (
    run_id          TEXT PRIMARY KEY,
    full_id         TEXT NOT NULL,
    status          TEXT NOT NULL,
    enqueued_at     TEXT NOT NULL,
    ended_at        TEXT,
    summary_outcome TEXT,
    state_path      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_project_runs_status     ON project_runs(status);
CREATE INDEX IF NOT EXISTS idx_project_runs_full_id    ON project_runs(full_id);
CREATE INDEX IF NOT EXISTS idx_project_runs_enqueued   ON project_runs(enqueued_at);
"""


class ProjectStore:
    """SQLite-indexed store for ProjectRuns, scoped to ``project_lane/``.

    Concurrency: we open a fresh connection per operation, both because
    SQLite handles multi-connection writes via the WAL log and because
    our main loop is single-threaded — connection pooling would be
    over-engineering for the access pattern.
    """

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.expanduser().resolve()
        self.lane_root = self.data_root / "project_lane"
        self.lane_root.mkdir(parents=True, exist_ok=True)
        (self.lane_root / "runs").mkdir(exist_ok=True)
        self._db_path = self.lane_root / "runs.sqlite"
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self) -> Any:
        conn = sqlite3.connect(str(self._db_path), timeout=15.0)
        try:
            conn.row_factory = sqlite3.Row
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ── enqueue + idempotency ──────────────────────────────────────────

    def enqueue(
        self,
        cand: ProjectCandidate,
        *,
        skip_if_recent_hours: float = 24.0,
    ) -> str | None:
        """Insert a new pending run for ``cand``.

        Returns the new run_id, or ``None`` if the same full_id was
        evaluated in the last ``skip_if_recent_hours`` hours (covers
        INV-P2 "no duplicate work on same target within freshness
        window"). Skips that explicitly failed-and-need-retry rows
        because those terminate in PREFLIGHT_ERROR / SANDBOX_DEAD —
        the operator can re-enqueue manually after fixing.
        """
        if self._recently_completed(cand.full_id, hours=skip_if_recent_hours):
            return None

        run_id = self._new_run_id()
        now_iso = datetime.now(UTC).isoformat()
        run = ProjectRun(
            run_id=run_id,
            full_id=cand.full_id,
            source_url=cand.source_url,
            enqueued_at=now_iso,
            candidate=asdict(cand),
        )
        self._write_state(run)
        with self._conn() as c:
            c.execute(
                "INSERT INTO project_runs "
                "(run_id, full_id, status, enqueued_at, state_path) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, cand.full_id, run.status.value,
                 now_iso, str(self._state_path(run_id))),
            )
        return run_id

    def _recently_completed(self, full_id: str, *, hours: float) -> bool:
        threshold_iso = datetime.now(UTC).replace(
            microsecond=0,
        ).isoformat()
        cutoff_ts = time.time() - hours * 3600
        cutoff_iso = datetime.fromtimestamp(cutoff_ts, UTC).isoformat()
        with self._conn() as c:
            row = c.execute(
                "SELECT 1 FROM project_runs "
                "WHERE full_id = ? AND enqueued_at > ? "
                "  AND status IN ('done', 'oversize', 'no_readme', 'needs_gpu') "
                "LIMIT 1",
                (full_id, cutoff_iso),
            ).fetchone()
        # ``threshold_iso`` is captured for log clarity even when unused
        # by SQL — keeps debugging easy if we ever need to verify the
        # cutoff comparison locally.
        _ = threshold_iso
        return row is not None

    # ── queries ────────────────────────────────────────────────────────

    def list_pending(self, *, limit: int = 100) -> list[ProjectRun]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT run_id FROM project_runs "
                "WHERE status = 'pending' "
                "ORDER BY enqueued_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self.load(r["run_id"]) for r in rows]

    def load(self, run_id: str) -> ProjectRun:
        path = self._state_path(run_id)
        if not path.exists():
            raise FileNotFoundError(f"no state.json for run_id={run_id}")
        obj = json.loads(path.read_text(encoding="utf-8"))
        return ProjectRun.from_dict(obj)

    def update_status(
        self, run: ProjectRun, status: ProjectStatus,
        *, failure_reason_zh: str = "",
    ) -> None:
        """Atomically update run status — write state.json then bump
        the sqlite index row. ``failure_reason_zh`` is the operator-
        visible Chinese reason; empty means "not a failure, just a
        phase advance".
        """
        run.status = status
        if failure_reason_zh:
            run.failure_reason_zh = failure_reason_zh
        if status == ProjectStatus.AGENT_RUN and run.started_at is None:
            run.started_at = datetime.now(UTC).isoformat()
        if status in _terminal_statuses():
            run.ended_at = datetime.now(UTC).isoformat()
        self._write_state(run)
        with self._conn() as c:
            c.execute(
                "UPDATE project_runs SET status = ?, ended_at = ?, "
                "summary_outcome = ? WHERE run_id = ?",
                (status.value, run.ended_at,
                 run.summary_outcome, run.run_id),
            )

    def write_report_artifacts(
        self, run: ProjectRun, result: AgentRunResult,
    ) -> None:
        """Persist agent_driver outputs into the run dir and update
        the inline summary fields on ``run``."""
        run_dir = self._run_dir(run.run_id)
        result.write_to(run_dir)
        run.agent_log = "agent.log"
        run.report_json = "report.json"
        if result.raw_bad_report:
            run.bad_report_txt = "bad_report.txt"
        run.summary_outcome = result.report.outcome.value
        run.summary_deploys = result.report.verdict.deploys
        run.summary_quickstart = result.report.verdict.quickstart_works
        self._write_state(run)

    # ── path helpers ───────────────────────────────────────────────────

    def _run_dir(self, run_id: str) -> Path:
        return self.lane_root / "runs" / run_id

    def _state_path(self, run_id: str) -> Path:
        return self._run_dir(run_id) / "state.json"

    def _write_state(self, run: ProjectRun) -> None:
        run_dir = self._run_dir(run.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        tmp = run_dir / "state.json.tmp"
        tmp.write_text(
            json.dumps(run.to_jsonable(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(run_dir / "state.json")

    @staticmethod
    def _new_run_id() -> str:
        now = datetime.now(UTC).strftime("%Y%m%d")
        return f"proj-{now}-{uuid.uuid4().hex[:6]}"


def _terminal_statuses() -> set[ProjectStatus]:
    """Return the set of statuses that should freeze the ``ended_at``
    timestamp. Centralised so we can audit one place."""
    return {
        ProjectStatus.DONE,
        ProjectStatus.OVERSIZE,
        ProjectStatus.NO_README,
        ProjectStatus.NEEDS_GPU,
        ProjectStatus.NOT_FOUND,
        ProjectStatus.PREFLIGHT_ERROR,
        ProjectStatus.REPORT_PARSE_ERROR,
        ProjectStatus.BUDGET_EXCEEDED,
        ProjectStatus.TIMEOUT,
        ProjectStatus.SANDBOX_DEAD,
        ProjectStatus.JUDGE_UNAVAILABLE,
    }


# ── stage: preflight ──────────────────────────────────────────────────────


HttpJsonGet = Callable[[str], dict[str, Any]]
"""Test seam: a function that takes a URL and returns parsed JSON.
Production: ``_default_github_get``; tests inject canned responses."""


def _default_github_get(url: str) -> dict[str, Any]:  # pragma: no cover
    """Thin stdlib GET with 10s timeout + JSON parse. Used by preflight
    to call ``GET /repos/{owner}/{repo}``. We deliberately don't
    authenticate — the unauthenticated rate limit (60/hr per IP) is
    enough for ~5 projects/day; if we hit it we degrade gracefully
    (PREFLIGHT_ERROR), not crash.
    """
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "heyi-eval-v10/project-lane",
            "Accept": "application/vnd.github+json",
        },
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        parsed: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return parsed


# Hard limit on repo size at preflight. GitHub returns size in KB.
# 500 MB chosen because:
#   - typical AI tool repo is <100 MB
#   - 500MB covers ~95% of legitimate "agent + skill + docs" packs
#   - bigger than that = model weights / dataset checkpoints / mistake
# Operator can override via env for special cases.
_MAX_REPO_MB = int(os.environ.get("HEYI_EVAL_PROJECT_MAX_MB", "500"))


# Keywords in README / description / topics that strongly suggest the
# project needs a beefy GPU we don't have. Hits get OUTCOME=NEEDS_GPU.
# Intentionally conservative — false positive rejects a usable project,
# false negative wastes one agent run on something that will fail at
# the agent_run stage anyway. Bias toward false negative (let it run).
_NEEDS_GPU_PATTERNS = (
    re.compile(r"\b(?:NVIDIA\s*H100|A100\s*\d*GB|H200|MI300)\b", re.IGNORECASE),
    re.compile(r"\brequires?\s+(?:80|96|192)\s*GB\b", re.IGNORECASE),
    re.compile(r"\b8\s*x\s*(?:A100|H100)\b", re.IGNORECASE),
)


@dataclass
class PreflightResult:
    allow: bool
    status_if_blocked: ProjectStatus | None
    reason_zh: str
    raw_metadata: dict[str, Any] = field(default_factory=dict)


def preflight(
    run: ProjectRun,
    *,
    http_json_get: HttpJsonGet | None = None,
    max_repo_mb: int = _MAX_REPO_MB,
) -> PreflightResult:
    """Decide whether ``run`` should proceed to AGENT_RUN.

    Three concrete checks, in order (first failure short-circuits):
      1. ``GET /repos/{owner}/{repo}`` returns OK + ``size <= max_repo_mb``;
         404 → NOT_FOUND, other HTTP error → PREFLIGHT_ERROR
      2. ``readme_url`` present in metadata (INV-P1: no README → can't drive agent)
      3. Description + topics don't contain "needs huge GPU" keywords (INV-P5)

    The metadata blob is persisted to ``preflight.json`` so JUDGE and
    panel can read the same facts the gate decided on.
    """
    fn = http_json_get or _default_github_get
    url = f"https://api.github.com/repos/{run.full_id}"
    try:
        meta = fn(url)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return PreflightResult(
                allow=False,
                status_if_blocked=ProjectStatus.NOT_FOUND,
                reason_zh=f"GitHub 仓库不存在或私有：{run.full_id}",
            )
        return PreflightResult(
            allow=False,
            status_if_blocked=ProjectStatus.PREFLIGHT_ERROR,
            reason_zh=f"GitHub API 异常 {e.code}：{e.reason}",
        )
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        return PreflightResult(
            allow=False,
            status_if_blocked=ProjectStatus.PREFLIGHT_ERROR,
            reason_zh=f"GitHub API 临时不可达：{type(e).__name__}",
        )

    size_kb = int(meta.get("size") or 0)
    size_mb = size_kb / 1024
    if size_mb > max_repo_mb:
        return PreflightResult(
            allow=False,
            status_if_blocked=ProjectStatus.OVERSIZE,
            reason_zh=(
                f"仓库体积 {size_mb:.0f} MB 超过单 run 上限 {max_repo_mb} MB"
            ),
            raw_metadata=meta,
        )

    has_readme = bool(meta.get("readme_url")) or _meta_implies_readme(meta)
    if not has_readme:
        return PreflightResult(
            allow=False,
            status_if_blocked=ProjectStatus.NO_README,
            reason_zh="仓库缺少 README，agent 无法识别启动步骤",
            raw_metadata=meta,
        )

    if _needs_huge_gpu(meta):
        return PreflightResult(
            allow=False,
            status_if_blocked=ProjectStatus.NEEDS_GPU,
            reason_zh="README/topics 显式要求 H100/80GB 级硬件，本沙箱无法满足",
            raw_metadata=meta,
        )

    return PreflightResult(
        allow=True, status_if_blocked=None,
        reason_zh="", raw_metadata=meta,
    )


def _meta_implies_readme(meta: dict[str, Any]) -> bool:
    """GitHub's repos API doesn't directly return readme_url for all
    repos; fall back to ``has_wiki=False AND has_pages=False AND
    description nonempty`` as a soft heuristic. Conservative: when in
    doubt assume there IS a README — false positive lets the agent run
    and figure out itself; false negative blocks a valid repo.
    """
    if meta.get("description"):
        return True
    return not (meta.get("archived") or meta.get("disabled"))


def _needs_huge_gpu(meta: dict[str, Any]) -> bool:
    haystack = " ".join([
        meta.get("description") or "",
        " ".join(meta.get("topics") or []),
    ])
    return any(p.search(haystack) for p in _NEEDS_GPU_PATTERNS)


# ── stage: build task prompt ──────────────────────────────────────────────


def build_task_prompt(run: ProjectRun, preflight_meta: dict[str, Any]) -> str:
    """Compose the M2.7 system prompt that drives the project run.

    Contract:
      - Chinese-primary (matches the rest of v10 UX)
      - Tells the agent to clone, READ THE README, install, run
        quickstart, exercise 2-3 named features, and emit fenced JSON
      - Specifies the fence markers + schema URL so the agent can
        self-validate before printing
      - Forbids ``pip install --user`` to mitigate INV-P8 互污染

    Why a single big prompt rather than a multi-turn conversation?

    - claude --print is one-shot. Multi-turn requires keeping the
      conversation alive, which adds budget guard complexity for
      marginal quality gain.
    - The agent has tool access (bash, file IO, web) so it can iterate
      WITHIN one turn — which is the cheap iteration we want.
    """
    desc = preflight_meta.get("description") or "(无描述)"
    stars = preflight_meta.get("stargazers_count") or "?"
    lang = preflight_meta.get("language") or "?"
    topics = ", ".join(preflight_meta.get("topics") or []) or "(无标签)"

    return f"""# 任务：评测开源项目 {run.full_id}

## 项目背景
- 仓库：{run.source_url}
- 描述：{desc}
- 主语言：{lang}
- 当前 stars：{stars}
- 标签：{topics}
- 入选理由：{run.candidate.get('reason', '?')}（agents-radar 日报）
- 今日 stars 增量：+{run.candidate.get('stars_delta') or '?'}

## 你需要做的

请严格按下列步骤执行，每完成一步在 stdout 输出 "[STEP X DONE]" 行：

1. **clone**：`git clone --depth 1 {run.source_url} /tmp/proj && cd /tmp/proj`
2. **read README**：列出 README 中"安装"和"快速开始"两段的核心命令
3. **install**：执行最直接的依赖安装路径（pip / npm / pnpm / cargo / make）
4. **smoke run**：跑 README 给出的最小启动命令；如果是 CLI，跑 `--help` + 1 个真实子命令；如果是 lib，写 10 行 Python 调用它的核心 API
5. **demonstrate 2-3 features**：从 README "Features" 章节挑 2-3 个高价值能力，每个写 1 段实际调用并展示真实输出
6. **fail fast**：任何一步失败立即停止，不要 hack。把错误原因记进下面的 JSON。

## 报告契约（INV-P3）

完成后，在 stdout 末尾输出**且仅输出一次**下面这段围栏 JSON：

```
<<<HEYI_RUN_REPORT_JSON>>>
{{
  "schema_version": "1.0",
  "lane": "project",
  "target_id": "{run.full_id}",
  "outcome": "pass" | "partial" | "fail",
  "steps": [
    {{"name": "clone", "status": "ok|fail|skip", "duration_s": 0.0, "note": "...", "stdout_tail": "..."}},
    {{"name": "install", "status": "...", "duration_s": 0.0}},
    {{"name": "smoke_run", "status": "..."}},
    {{"name": "feature_<name>", "status": "..."}}
  ],
  "verdict": {{
    "deploys": true|false,
    "quickstart_works": true|false,
    "core_features_demonstrated": ["feature_a", "feature_b"],
    "blockers": ["..."]
  }},
  "self_assessment_zh": "1-2 句中文总结：能不能跑、跑通了什么、有什么主要阻塞",
  "follow_ups": ["可选：值得后续深挖的改进点"]
}}
<<<END>>>
```

## 硬约束（不可违反）

- 不要 ``pip install --user`` / ``npm install -g``：用 venv / npx
- 不要 ``rm -rf /`` / ``chmod -R`` 容器系统目录
- 不要 ``curl | bash`` 来源不明的脚本
- 不要修改 /home/agent/.claude-code-router/config.json
- ``outcome=pass`` 必须 ``deploys=true AND quickstart_works=true AND len(core_features_demonstrated) >= 1``
- 没跑通就老实写 ``outcome=fail``，``blockers`` 写清楚卡在哪一步

预算：单 run 上限 50,000 tokens、墙钟 15 分钟。
"""


# ── stage runner: end-to-end one run ──────────────────────────────────────


def execute_run(
    store: ProjectStore,
    run: ProjectRun,
    *,
    pool: PoolManager,
    host_workspace_root: Path,
    http_json_get: HttpJsonGet | None = None,
    stream_exec: Any | None = None,
    budget_guard: BudgetGuard | None = None,
) -> ProjectRun:
    """Execute one run through PREFLIGHT → AGENT_RUN → JUDGE → DONE.

    Returns the updated ProjectRun (also persisted to disk + sqlite).
    Never raises for predictable failure modes — those land in
    ``run.status`` + ``run.failure_reason_zh`` and the run dir.

    Programmer errors (PoolBusy from over-subscription, broken state
    file) still raise so the daemon's outer loop can decide to skip
    that run for the round.
    """
    log.info("project_lane.execute_run %s (%s) start", run.run_id, run.full_id)

    # ── PREFLIGHT ──────────────────────────────────────────────────────
    store.update_status(run, ProjectStatus.PREFLIGHT)
    pf = preflight(run, http_json_get=http_json_get)
    run_dir = store._run_dir(run.run_id)
    (run_dir / "preflight.json").write_text(
        json.dumps(pf.raw_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run.preflight_json = "preflight.json"
    if not pf.allow:
        assert pf.status_if_blocked is not None
        store.update_status(run, pf.status_if_blocked,
                            failure_reason_zh=pf.reason_zh)
        log.info("project_lane.execute_run %s blocked: %s (%s)",
                 run.run_id, pf.status_if_blocked.value, pf.reason_zh)
        return run

    # ── AGENT_RUN ──────────────────────────────────────────────────────
    store.update_status(run, ProjectStatus.AGENT_RUN)
    prompt = build_task_prompt(run, pf.raw_metadata)
    result = run_agent(
        pool=pool, lane="project", target_id=run.full_id,
        run_id=run.run_id, task_prompt=prompt,
        budget_guard=budget_guard,
        stream_exec=stream_exec,
        host_workspace_root=host_workspace_root,
    )
    store.write_report_artifacts(run, result)

    # Map agent_driver outcome → project_lane status. We DON'T just
    # write outcome.value verbatim because some agent outcomes (PASS /
    # PARTIAL / FAIL) all funnel into ProjectStatus.DONE; the panel
    # uses summary_outcome to display the actual agent verdict.
    terminal = _agent_outcome_to_status(result.report.outcome)
    if terminal != ProjectStatus.DONE:
        store.update_status(
            run, terminal,
            failure_reason_zh=result.report.self_assessment_zh,
        )
        log.info("project_lane.execute_run %s terminal: %s",
                 run.run_id, terminal.value)
        return run

    # ── JUDGE (M1: pass-through; M2: yunwu M2.7 second-opinion) ───────
    # In M2 we record the JUDGE phase but don't yet add a second-opinion
    # call — the agent's own outcome IS the verdict, and the panel
    # surfaces it. Adding a JUDGE pass is M3 work after we see how
    # often the agent's self-assessment matches reality on real runs.
    store.update_status(run, ProjectStatus.JUDGE)
    store.update_status(run, ProjectStatus.DONE)
    log.info("project_lane.execute_run %s DONE outcome=%s",
             run.run_id, run.summary_outcome)
    return run


def _agent_outcome_to_status(outcome: Outcome) -> ProjectStatus:
    """Map agent_driver's Outcome enum → ProjectStatus.

    PASS / PARTIAL / FAIL all funnel into DONE — the agent's verdict
    IS the run's verdict, and the panel reads summary_outcome to
    display the fine-grained outcome. System-imposed outcomes (parse
    error, budget exceeded, etc) map to their own ProjectStatus so the
    UI can colour-code them distinctly from "agent ran but failed".
    """
    return {
        Outcome.PASS: ProjectStatus.DONE,
        Outcome.PARTIAL: ProjectStatus.DONE,
        Outcome.FAIL: ProjectStatus.DONE,
        Outcome.REPORT_PARSE_ERROR: ProjectStatus.REPORT_PARSE_ERROR,
        Outcome.BUDGET_EXCEEDED: ProjectStatus.BUDGET_EXCEEDED,
        Outcome.TIMEOUT: ProjectStatus.TIMEOUT,
        Outcome.SANDBOX_DEAD: ProjectStatus.SANDBOX_DEAD,
        Outcome.JUDGE_UNAVAILABLE: ProjectStatus.JUDGE_UNAVAILABLE,
        # These shouldn't appear from agent_driver in project_lane, but
        # we default them to DONE so future enum drift doesn't crash.
        Outcome.OVERSIZE: ProjectStatus.OVERSIZE,
        Outcome.NO_README: ProjectStatus.NO_README,
        Outcome.NEEDS_GPU: ProjectStatus.NEEDS_GPU,
        Outcome.SKILL_CLONE_ATTEMPT: ProjectStatus.DONE,
    }.get(outcome, ProjectStatus.DONE)


# ── main loop helper ──────────────────────────────────────────────────────


def run_pending(
    store: ProjectStore,
    *,
    pool: PoolManager,
    host_workspace_root: Path,
    limit: int = 5,
    http_json_get: HttpJsonGet | None = None,
    stream_exec: Any | None = None,
) -> int:
    """Pop up to ``limit`` pending runs and execute each in turn.

    Returns the number of runs that completed (any terminal status
    counts, not just DONE). Designed for systemd to invoke periodically
    (e.g. every 30 min). The execute_run call swallows predictable
    failures, so this function's only loud exit is a programmer error
    (e.g. broken state.json) which we deliberately let propagate so
    the systemd unit fails loud + the operator notices.
    """
    pending = store.list_pending(limit=limit)
    completed = 0
    for run in pending:
        try:
            execute_run(
                store, run,
                pool=pool, host_workspace_root=host_workspace_root,
                http_json_get=http_json_get, stream_exec=stream_exec,
            )
            completed += 1
        except Exception as e:  # log + continue is the contract
            log.exception("project_lane.run_pending fatal for %s: %s",
                          run.run_id, e)
            store.update_status(
                run, ProjectStatus.SANDBOX_DEAD,
                failure_reason_zh=f"未预期异常：{type(e).__name__}: {e}",
            )
    return completed


# ── helper: load most recent N done/done-ish for panel ───────────────────


def list_recent(
    store: ProjectStore, *, limit: int = 50,
) -> list[ProjectRun]:
    """Panel-facing query: most-recent N runs by enqueued_at desc.
    Includes both terminal and in-flight rows so the panel can show
    a live status column.
    """
    with store._conn() as c:
        rows = c.execute(
            "SELECT run_id FROM project_runs "
            "ORDER BY enqueued_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [store.load(r["run_id"]) for r in rows]


# ── convenience: synthesise a verdict report (used by tests + UI) ────────


def make_synthetic_done_report(
    run: ProjectRun, outcome: Outcome,
    self_assessment_zh: str,
    *,
    deploys: bool = False,
    quickstart_works: bool = False,
    features: list[str] | None = None,
    blockers: list[str] | None = None,
) -> RunReport:
    """Helper for tests and for the rare case where the panel wants
    to construct a hand-built RunReport (e.g. operator manually
    annotates an out-of-band project decision)."""
    return RunReport(
        schema_version="1.0",
        lane="project",
        target_id=run.full_id,
        outcome=outcome,
        steps=[],
        verdict=Verdict(
            deploys=deploys, quickstart_works=quickstart_works,
            core_features_demonstrated=features or [],
            blockers=blockers or [],
        ),
        self_assessment_zh=self_assessment_zh,
        follow_ups=[],
    )


# unused-import suppressor: StepStatus is exported for downstream
# callers that build steps manually.
_ = StepStatus
