"""Extract + validate the fenced JSON report from agent stdout (INV-P3).

Contract (`docs/ARCHITECTURE_LANES.md` §6, `docs/INVARIANTS_LANES.md` INV-P3):

- agent's stdout must contain ``<<<HEYI_RUN_REPORT_JSON>>>...<<<END>>>``
- multiple blocks → last one wins
- block missing OR JSON invalid OR schema mismatch → ``REPORT_PARSE_ERROR``
- ``outcome=PASS`` with empty ``core_features_demonstrated`` is downgraded
  to ``PARTIAL`` (soft business rule, see ``_enforce_pass_requires_feature``)

This module is **pure** — no I/O, no docker, no logging side-effects.
That's intentional: extractor is on the hot path of every run, and any
hidden side-effects would make budget_guard accounting unreliable.

Typical call site (exec_runner)::

    raw_stdout: str = collect_agent_stdout(container, run_id)
    result = extract_report(raw_stdout, lane="project", target_id="owner/repo")
    if isinstance(result, ExtractError):
        outcome = Outcome.REPORT_PARSE_ERROR
        # raw_stdout still gets archived to runs/.../bad_report.txt
    else:
        report = result   # RunReport, ready for jsonschema-validated
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import jsonschema

from agent_driver.schema import (
    FENCE_CLOSE,
    FENCE_OPEN,
    REPORT_JSON_SCHEMA,
    Outcome,
    RunReport,
    Step,
    StepStatus,
    Verdict,
)

# ── public types ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractError:
    """Why extraction failed — drives both ``outcome=REPORT_PARSE_ERROR``
    setting AND the human-readable Chinese ``self_assessment_zh`` we
    surface on the panel when the agent didn't produce one.

    ``kind`` is one of:
      - ``fence_missing``   — no opening or closing marker found
      - ``fence_empty``     — both markers found but nothing between
      - ``json_invalid``    — content between markers wasn't parseable JSON
      - ``schema_invalid``  — JSON parsed but failed REPORT_JSON_SCHEMA
      - ``lane_mismatch``   — JSON ``lane`` field disagrees with caller
      - ``target_mismatch`` — JSON ``target_id`` field disagrees with caller

    ``detail`` is short English (panel-rendered alongside the
    operator-facing zh message); ``raw_excerpt`` is up to 500 chars of
    the offending text so the user can see what the agent actually wrote.
    """
    kind: Literal[
        "fence_missing", "fence_empty", "json_invalid",
        "schema_invalid", "lane_mismatch", "target_mismatch",
    ]
    detail: str
    raw_excerpt: str = ""

    def to_zh_assessment(self) -> str:
        """Render a Chinese ``self_assessment_zh`` suitable for stuffing
        into a synthetic RunReport when the agent's own version is unusable.

        Kept short (~80 chars each) so the panel preview column reads
        like a normal agent self-assessment, not like a stack trace.
        """
        return {
            "fence_missing":   f"Agent 未输出 <<<HEYI_RUN_REPORT_JSON>>> 围栏：{self.detail}",
            "fence_empty":     f"Agent 围栏内为空：{self.detail}",
            "json_invalid":    f"Agent 围栏内 JSON 无法解析：{self.detail}",
            "schema_invalid":  f"Agent 报告字段不合 schema：{self.detail}",
            "lane_mismatch":   f"Agent 报告 lane 与系统期望不一致：{self.detail}",
            "target_mismatch": f"Agent 报告 target_id 与系统期望不一致：{self.detail}",
        }.get(self.kind, f"未知报告错误：{self.detail}")


# ── extraction ────────────────────────────────────────────────────────────


def extract_report(
    raw_stdout: str,
    *,
    lane: Literal["project", "skill"],
    target_id: str,
) -> RunReport | ExtractError:
    """Return a validated RunReport, or ExtractError describing failure.

    The function never raises on parse / schema problems — it returns
    the error so callers can decide whether to fail the run or fall back
    to a synthetic ``REPORT_PARSE_ERROR`` report. Truly exceptional
    cases (raw_stdout is not a string, schema definition itself is
    broken) are still allowed to raise, because those are bugs not data.
    """
    fenced = _last_fence_payload(raw_stdout)
    if fenced is None:
        return ExtractError(
            kind="fence_missing",
            detail=f"neither '{FENCE_OPEN}' nor '{FENCE_CLOSE}' found "
                   f"in {len(raw_stdout)} chars of stdout",
            raw_excerpt=raw_stdout[-500:],
        )

    if not fenced.strip():
        return ExtractError(
            kind="fence_empty",
            detail="markers present but body is whitespace-only",
        )

    try:
        obj = json.loads(fenced)
    except json.JSONDecodeError as e:
        return ExtractError(
            kind="json_invalid",
            detail=f"{type(e).__name__}: {e.msg} at line {e.lineno} col {e.colno}",
            raw_excerpt=fenced[:500],
        )

    try:
        jsonschema.validate(obj, REPORT_JSON_SCHEMA)
    except jsonschema.ValidationError as e:
        return ExtractError(
            kind="schema_invalid",
            # e.message is short + actionable, e.json_path points at offender
            detail=f"{e.json_path}: {e.message}",
            raw_excerpt=fenced[:500],
        )

    if obj["lane"] != lane:
        return ExtractError(
            kind="lane_mismatch",
            detail=f"expected lane={lane!r}, agent said {obj['lane']!r}",
        )

    if obj["target_id"] != target_id:
        return ExtractError(
            kind="target_mismatch",
            detail=f"expected target_id={target_id!r}, "
                   f"agent said {obj['target_id']!r}",
        )

    report = _to_report_dataclass(obj)
    return _enforce_pass_requires_feature(report)


# ── helpers ───────────────────────────────────────────────────────────────


def _last_fence_payload(raw_stdout: str) -> str | None:
    """Return the body of the LAST ``<<<...>>> ... <<<END>>>`` block.

    Edge cases:
    - opener present, closer missing → returns None (treated as missing)
    - multiple openers + closers → matches by position of the LAST
      ``FENCE_OPEN`` followed by the FIRST ``FENCE_CLOSE`` after it.
      Agent draft-then-revise pattern is handled by the agent itself
      writing the second block AFTER the first close; the last open
      after the last close wins.
    """
    last_open = raw_stdout.rfind(FENCE_OPEN)
    if last_open < 0:
        return None
    body_start = last_open + len(FENCE_OPEN)
    close_at = raw_stdout.find(FENCE_CLOSE, body_start)
    if close_at < 0:
        return None
    return raw_stdout[body_start:close_at]


def _to_report_dataclass(obj: dict[str, Any]) -> RunReport:
    """Schema-validated dict → typed RunReport. Safe because by the time
    we're here, jsonschema has confirmed every required field is present
    with correct types — so the ``Outcome(...)`` / ``StepStatus(...)``
    enum coercions cannot fail.
    """
    verdict = Verdict(**obj["verdict"])
    steps = [
        Step(
            name=s["name"],
            status=StepStatus(s["status"]),
            duration_s=float(s["duration_s"]),
            note=s.get("note", ""),
            stdout_tail=s.get("stdout_tail", ""),
            artifacts=list(s.get("artifacts", [])),
        )
        for s in obj["steps"]
    ]
    return RunReport(
        schema_version=obj["schema_version"],
        lane=obj["lane"],
        target_id=obj["target_id"],
        outcome=Outcome(obj["outcome"]),
        steps=steps,
        verdict=verdict,
        self_assessment_zh=obj["self_assessment_zh"],
        follow_ups=list(obj.get("follow_ups", [])),
    )


def _enforce_pass_requires_feature(report: RunReport) -> RunReport:
    """INV-P3 soft constraint: ``outcome=PASS`` requires at least one
    entry in ``verdict.core_features_demonstrated``. Empty list with
    PASS is downgraded (not rejected) to PARTIAL — we'd rather record
    "agent completed but didn't name a feature" than discard the run.

    This is intentionally NOT a jsonschema constraint because the
    panel still wants the rest of the report intact + visible.
    """
    if report.outcome is Outcome.PASS and not report.verdict.core_features_demonstrated:
        return RunReport(
            schema_version=report.schema_version,
            lane=report.lane,
            target_id=report.target_id,
            outcome=Outcome.PARTIAL,
            steps=report.steps,
            verdict=report.verdict,
            self_assessment_zh=(
                report.self_assessment_zh
                + "\n[系统降级] outcome 从 pass 降为 partial：未列出 "
                  "verdict.core_features_demonstrated"
            )[:1024],
            follow_ups=report.follow_ups,
        )
    return report


# ── synthetic-report helper (used when extraction fails) ──────────────────


def synthetic_parse_error_report(
    err: ExtractError,
    *,
    lane: Literal["project", "skill"],
    target_id: str,
) -> RunReport:
    """Build a RunReport that fills the same shape as the agent's would,
    so panel + downstream JUDGE stage can still render uniformly when
    the agent failed to deliver. The ``self_assessment_zh`` carries the
    Chinese-translated failure reason.

    Note: this report has ``outcome=REPORT_PARSE_ERROR`` which is itself
    a valid REPORT_JSON_SCHEMA value (we deliberately allowed all
    Outcome values in the schema's ``outcome.enum``).
    """
    return RunReport(
        schema_version="1.0",
        lane=lane,
        target_id=target_id,
        outcome=Outcome.REPORT_PARSE_ERROR,
        steps=[],
        verdict=Verdict(deploys=False, quickstart_works=False),
        self_assessment_zh=err.to_zh_assessment(),
        follow_ups=[],
    )
