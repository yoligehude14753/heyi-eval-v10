"""Orchestrate: acquire container → write TASK.md → docker exec claude → harvest.

This is the file the rest of v10 calls. project_lane / skill_lane each
build a task_prompt (e.g. "clone X, install deps, run quickstart, emit
fenced JSON") and hand it here; the run result (logs + report) lands on
disk under ``runs/<lane>/<target_id>/<run_id>/``.

What this module does NOT do:

- Decide what task the agent should run — that's the lane-specific
  Stage logic.
- Talk to yunwu directly — ccr_bridge owns the routing config; we
  just exec ``claude`` inside a container that has it loaded.
- Render the panel — that's panel/server.py.

The function ``run_agent`` is the single public entry point. The rest
of the module is private helpers and stays close to the function so
reading top-to-bottom tells a story.

Determinism + testability:

- Every external interaction is parametrised: ``docker_client``,
  ``stream_exec`` (defaults to ``_default_stream_exec`` which uses
  docker-py's exec_run), ``budget_guard``, ``now`` (clock).
- The unit tests construct a fake pool + fake stream_exec that yields
  pre-canned stdout chunks → we exercise the full path without docker.
"""
from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agent_driver.budget_guard import BudgetCheck, BudgetDecision, BudgetGuard
from agent_driver.pool_manager import (
    ContainerHandle,
    PoolBusy,
    PoolManager,
    safe_run_workspace,
)
from agent_driver.report_extractor import (
    ExtractError,
    extract_report,
    synthetic_parse_error_report,
)
from agent_driver.schema import Outcome, RunReport

log = logging.getLogger(__name__)


# ── public result type ────────────────────────────────────────────────────


@dataclass
class AgentRunResult:
    """Everything one ``run_agent`` call produced — written to disk by
    the caller (project_lane / skill_lane Stage code), which decides
    which directory tree to use under ``runs/``.

    Fields:
      report: the validated RunReport (synthetic if extraction failed)
      raw_stdout: agent.log content (always populated even on failure)
      raw_bad_report: when ``extract_error`` is not None, the offending
          string that failed to parse; otherwise empty
      extract_error: None if extraction succeeded
      budget_final: BudgetGuard's final state snapshot
      elapsed_s: wall-clock from acquire to release (includes
          container exec setup)
      container_name: which pool member ran this
    """
    report: RunReport
    raw_stdout: str
    raw_bad_report: str = ""
    extract_error: ExtractError | None = None
    budget_final: BudgetCheck | None = None
    elapsed_s: float = 0.0
    container_name: str = ""

    def write_to(self, run_dir: Path) -> None:
        """Persist this result under ``run_dir/``. Caller controls the
        run_dir path (lane-specific), this helper just lays out files.

        Files written:
          - ``report.json``      — RunReport.to_dict()
          - ``agent.log``        — raw_stdout, utf-8
          - ``bad_report.txt``   — only when extraction failed
          - ``budget.json``      — BudgetCheck details
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "report.json").write_text(
            json.dumps(self.report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (run_dir / "agent.log").write_text(self.raw_stdout, encoding="utf-8")
        if self.raw_bad_report:
            (run_dir / "bad_report.txt").write_text(
                self.raw_bad_report, encoding="utf-8",
            )
        if self.budget_final is not None:
            (run_dir / "budget.json").write_text(
                json.dumps({
                    "decision": self.budget_final.decision.value,
                    "elapsed_s": self.budget_final.elapsed_s,
                    "total_tokens": self.budget_final.total_tokens,
                    "detail": self.budget_final.detail,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )


# ── stream type alias ─────────────────────────────────────────────────────
#
# stream_exec is the seam where we plug docker-exec in production and
# canned chunks in tests. It takes (container, command, workdir) and
# returns an iterable yielding bytes chunks (stdout). We use bytes
# because docker-py's exec stream yields bytes; we decode at the
# boundary.
StreamExec = Callable[
    [ContainerHandle, list[str], str],
    Iterable[bytes],
]


# ── public entry point ────────────────────────────────────────────────────


def run_agent(
    *,
    pool: PoolManager,
    lane: Literal["project", "skill"],
    target_id: str,
    run_id: str,
    task_prompt: str,
    extra_files: dict[str, str] | None = None,
    budget_guard: BudgetGuard | None = None,
    stream_exec: StreamExec | None = None,
    host_workspace_root: Path,
    now: Callable[[], float] = time.monotonic,
) -> AgentRunResult:
    """Run one agent task end-to-end and return its result.

    Sequence (each step on its own line so failures are easy to map):
      1. ``pool.acquire(run_id)`` → ContainerHandle
      2. write ``TASK.md`` (+ any extra_files) to
         ``host_workspace_root/$run_id/`` so the in-container mount sees it
      3. ``stream_exec(container, ["claude", "--print", ...], workdir)``
         streaming bytes chunks of stdout
      4. for each chunk: decode, append to buffer, peek for ccr usage
         lines, ``budget_guard.note_tokens`` + ``budget_guard.check()``;
         break if EXCEEDED_*
      5. extract_report on full stdout; synth REPORT_PARSE_ERROR if needed
      6. ``pool.release(run_id)`` — happens in ``finally``, so even
         exec_exception paths clean up the workspace

    Returns:
      AgentRunResult — always; never raises for predictable failure
      modes (extraction error, budget exceeded, sandbox dead). Only
      raises for programmer errors (PoolBusy, ValueError on bad
      run_id, etc.) which the orchestrator main loop catches.
    """
    if stream_exec is None:
        stream_exec = _default_stream_exec
    if budget_guard is None:
        budget_guard = BudgetGuard()

    # Validate extra_files BEFORE acquire so caller programmer errors
    # surface as ValueError (not as a synthetic SANDBOX_DEAD report).
    # The runtime-failure block below explicitly catches everything,
    # so without this pre-check a malformed extra_files dict would get
    # swallowed and a misleading report would be produced.
    for rel_path in (extra_files or {}):
        if "/" in rel_path or ".." in rel_path:
            raise ValueError(
                f"extra_files keys must be flat filenames, got {rel_path!r}"
            )

    t0 = now()
    handle: ContainerHandle | None = None
    final_budget: BudgetCheck | None = None
    raw_stdout = ""

    try:
        handle = pool.acquire(run_id)
        _write_task_files(
            host_workspace_root, run_id, task_prompt, extra_files or {},
        )
        raw_stdout, final_budget = _stream_and_track(
            handle, run_id, stream_exec, budget_guard,
        )

        # If budget tripped, synthesize a system report and don't even
        # try to extract — the agent was cut off, anything in its
        # stdout is likely truncated mid-JSON.
        if final_budget.decision is BudgetDecision.EXCEEDED_TOKENS:
            report = _synth_budget_report(
                lane=lane, target_id=target_id,
                outcome=Outcome.BUDGET_EXCEEDED,
                detail_zh=final_budget.detail,
            )
            return AgentRunResult(
                report=report, raw_stdout=raw_stdout,
                budget_final=final_budget,
                elapsed_s=now() - t0,
                container_name=handle.name,
            )
        if final_budget.decision is BudgetDecision.EXCEEDED_WALL_CLOCK:
            report = _synth_budget_report(
                lane=lane, target_id=target_id,
                outcome=Outcome.TIMEOUT,
                detail_zh=final_budget.detail,
            )
            return AgentRunResult(
                report=report, raw_stdout=raw_stdout,
                budget_final=final_budget,
                elapsed_s=now() - t0,
                container_name=handle.name,
            )

        # Budget OK → try to extract agent's report
        extracted = extract_report(
            raw_stdout, lane=lane, target_id=target_id,
        )
        if isinstance(extracted, ExtractError):
            return AgentRunResult(
                report=synthetic_parse_error_report(
                    extracted, lane=lane, target_id=target_id,
                ),
                raw_stdout=raw_stdout,
                raw_bad_report=raw_stdout[-4096:],  # keep last 4KB for inspection
                extract_error=extracted,
                budget_final=final_budget,
                elapsed_s=now() - t0,
                container_name=handle.name,
            )

        return AgentRunResult(
            report=extracted,
            raw_stdout=raw_stdout,
            budget_final=final_budget,
            elapsed_s=now() - t0,
            container_name=handle.name,
        )

    except PoolBusy:
        # Programmer error in single-threaded M1; let it propagate.
        raise
    except Exception as e:  # pragma: no cover — sandbox-death path
        # Container exec blew up mid-stream. Build a SANDBOX_DEAD
        # report so the run still lands on disk and panel can show it.
        log.exception("run_agent sandbox_dead lane=%s target=%s run_id=%s",
                      lane, target_id, run_id)
        return AgentRunResult(
            report=_synth_sandbox_dead_report(
                lane=lane, target_id=target_id, exc=e,
            ),
            raw_stdout=raw_stdout,
            extract_error=None,
            budget_final=final_budget,
            elapsed_s=now() - t0,
            container_name=handle.name if handle else "",
        )
    finally:
        if handle is not None:
            try:
                pool.release(run_id)
            except Exception:  # pragma: no cover — pool already logs
                log.exception("pool.release failed in finally; "
                              "container=%s run_id=%s",
                              handle.name, run_id)


# ── helpers ───────────────────────────────────────────────────────────────


def _write_task_files(
    host_workspace_root: Path,
    run_id: str,
    task_prompt: str,
    extra_files: dict[str, str],
) -> None:
    """Drop TASK.md + extras into the per-run workspace subdir.

    pool.acquire already created the subdir + chmod 0o700, so we don't
    re-create. We DO double-check the path via ``safe_run_workspace``
    to fail fast if someone mutates run_id between acquire and write.
    """
    ws = safe_run_workspace(host_workspace_root, run_id)
    if not ws.exists():
        raise FileNotFoundError(
            f"workspace dir missing: {ws}; pool.acquire should have made it"
        )
    (ws / "TASK.md").write_text(task_prompt, encoding="utf-8")
    # Pre-validated at run_agent entry; assert here as a tripwire so a
    # future caller that bypasses run_agent doesn't sneak nested paths in.
    for rel_path, content in extra_files.items():
        assert "/" not in rel_path and ".." not in rel_path, (
            f"extra_files keys must be flat (pre-validated upstream): {rel_path!r}"
        )
        (ws / rel_path).write_text(content, encoding="utf-8")


def _stream_and_track(
    handle: ContainerHandle,
    run_id: str,
    stream_exec: StreamExec,
    budget_guard: BudgetGuard,
) -> tuple[str, BudgetCheck]:
    """Run ``claude --print TASK.md`` inside the container, streaming
    stdout into a buffer. Returns (full_stdout, final_budget_check).

    Token accounting: we look for ccr usage lines mid-stream of the
    form ``"usage": {"prompt_tokens": N, "completion_tokens": M}``.
    They appear in ccr's log, NOT in the agent's stdout — but ccr also
    logs them to its stdout when LOG=true. docker exec returns the
    combined container-process stdout, which for a ``claude`` exec
    doesn't include ccr's log. So the M1 implementation here uses a
    heuristic: count chars / 3 as a rough token estimate ON STDOUT
    only; M2 will switch to reading ccr.log directly.

    The heuristic is intentional + documented + matches what the
    budget tests assert. The point of M1 is wiring; M2 swaps in the
    real meter without changing this function's contract.
    """
    cmd_in_container = [
        "claude", "--print",
        # Anthropic CLI reads the prompt from the file argument when
        # given as positional; --print suppresses interactive UI.
        f"/home/agent/workspace/{run_id}/TASK.md",
    ]
    workdir = f"/home/agent/workspace/{run_id}"

    chunks: list[bytes] = []
    final = budget_guard.check()
    for chunk in stream_exec(handle, cmd_in_container, workdir):
        if not chunk:
            continue
        chunks.append(chunk)
        # M1 heuristic: 1 token ≈ 3 chars of utf-8. Cheap, monotone.
        # M2 replaces this with a ccr-log tailer reading the true usage.
        approx_out_tokens = max(1, len(chunk) // 3)
        budget_guard.note_tokens(input_tokens=0, output_tokens=approx_out_tokens)
        final = budget_guard.check()
        if final.decision is not BudgetDecision.OK:
            break

    return b"".join(chunks).decode("utf-8", errors="replace"), final


def _default_stream_exec(  # pragma: no cover — exercised in M2 drill
    handle: ContainerHandle, command: list[str], workdir: str,
) -> Iterable[bytes]:
    """Production stream: docker-py's exec_run with stream=True.

    Intentionally kept thin so the integration risk lives in M2's drill
    rather than M1's unit suite. M1 ships ``run_agent`` with mock
    stream_exec; M2 wires this in + e2e on heyi.
    """
    raise NotImplementedError(
        "agent_driver.exec_runner._default_stream_exec is M2 work — "
        "M1 tests inject stream_exec explicitly."
    )


def _synth_budget_report(
    *, lane: str, target_id: str, outcome: Outcome, detail_zh: str,
) -> RunReport:
    """Construct a system-imposed RunReport for the budget-exceeded path.

    Used when budget_guard tripped — the agent's own report (if any) is
    untrusted because the agent was interrupted mid-thought.
    """
    from agent_driver.schema import Verdict  # local import to avoid cycle in stubs
    return RunReport(
        schema_version="1.0",
        lane=lane,  # type: ignore[arg-type]
        target_id=target_id,
        outcome=outcome,
        steps=[],
        verdict=Verdict(deploys=False, quickstart_works=False),
        self_assessment_zh=detail_zh,
        follow_ups=[],
    )


def _synth_sandbox_dead_report(  # pragma: no cover
    *, lane: str, target_id: str, exc: BaseException,
) -> RunReport:
    """Construct a system-imposed RunReport for the sandbox-dead path."""
    from agent_driver.schema import Verdict
    return RunReport(
        schema_version="1.0",
        lane=lane,  # type: ignore[arg-type]
        target_id=target_id,
        outcome=Outcome.SANDBOX_DEAD,
        steps=[],
        verdict=Verdict(deploys=False, quickstart_works=False),
        self_assessment_zh=f"沙箱执行异常：{type(exc).__name__}: {exc}",
        follow_ups=[],
    )
