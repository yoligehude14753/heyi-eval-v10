"""RunReport dataclass + JSON Schema v1.0.

This is the **single source of truth** for the report contract enforced by
INV-P3: the agent's last block of stdout MUST be a fenced JSON document
matching ``REPORT_JSON_SCHEMA`` exactly; non-conformance maps to
``Outcome.REPORT_PARSE_ERROR`` (handled by ``report_extractor``).

Why two parallel representations (dataclass + jsonschema dict)?

- The dataclass lets call-sites construct + manipulate reports in typed
  Python (panel renderer, test fixtures, judge stage).
- The jsonschema dict is what we feed to ``jsonschema.validate`` on agent
  output, which we cannot dataclass-load directly because agent output
  may have extra keys / wrong types and we want crisp error messages
  rather than a TypeError from ``RunReport(**raw)``.

Both representations are kept in sync by ``test_schema_dataclass_parity``
which constructs a RunReport, dumps it, validates against the schema,
and asserts round-trip fidelity. See also
[`docs/TEST_PLAN_LANES.md`](../docs/TEST_PLAN_LANES.md) §H1/S3/S4.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Literal


class Outcome(str, Enum):
    """Top-level run outcome — every run lands in exactly one of these.

    The first 3 (PASS / PARTIAL / FAIL) are emitted by the agent itself
    inside the fenced JSON. The remaining 7 are **system-imposed** when
    the agent is unable to deliver a usable report:

    - ``REPORT_PARSE_ERROR``: agent stdout missing fence OR JSON invalid
      OR schema mismatch (INV-P3 violation)
    - ``BUDGET_EXCEEDED`` / ``TIMEOUT``: budget_guard tripped (INV-P4)
    - ``SANDBOX_DEAD``: pool_manager detected container died mid-run
    - ``OVERSIZE`` / ``NO_README`` / ``NEEDS_GPU``: project_lane FETCH-stage
      pre-filter rejected (INV-P1 / INV-P5)
    - ``JUDGE_UNAVAILABLE``: yunwu 429 storms exhausted retry budget;
      the run itself may have produced a valid report but the JUDGE
      stage couldn't validate it
    - ``SKILL_CLONE_ATTEMPT``: skill_lane only — SKILL.md tried to
      ``git clone`` (INV-S1 violation)

    Adding a new outcome:
    1. Add the literal here
    2. Update ``test_schema_outcomes_covered`` to cover panel rendering
    3. Update ``RUNBOOK_NV8.md`` operator playbook
    """
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"
    REPORT_PARSE_ERROR = "report_parse_error"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"
    SANDBOX_DEAD = "sandbox_dead"
    OVERSIZE = "oversize"
    NO_README = "no_readme"
    NEEDS_GPU = "needs_gpu"
    JUDGE_UNAVAILABLE = "judge_unavailable"
    SKILL_CLONE_ATTEMPT = "skill_clone_attempt"


class StepStatus(str, Enum):
    """Per-step status emitted by the agent.

    ``PARTIAL`` was added 2026-05-26 after the heyi production loop
    observed M2.7 routinely emit ``status="partial"`` for steps that
    half-succeeded (e.g. CLI built but one demo command failed).
    Forcing the agent to round-trip those through ``ok``/``fail``
    erases real signal — a partial step matters for the panel
    ("install OK, smoke partial") and for the verdict heuristic
    (``outcome=partial`` was already a top-level Outcome literal, but
    the step grain didn't mirror it).  Run-level outcome is still
    decided by ``Verdict.core_features_demonstrated`` + ``blockers``,
    so adding PARTIAL here doesn't widen the pass-bar.
    """
    OK = "ok"
    PARTIAL = "partial"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class Step:
    """One stage of agent-driven execution (clone / install / smoke / ...).

    ``stdout_tail`` is capped to ~2 KB by the agent contract. ``artifacts``
    must be workspace-relative paths (INV-L2 path-escape guard rejects
    absolute paths or ``..`` traversal).
    """
    name: str
    status: StepStatus
    duration_s: float
    note: str = ""
    stdout_tail: str = ""
    artifacts: list[str] = field(default_factory=list)


@dataclass
class Verdict:
    """Agent's structured verdict — drives panel's at-a-glance columns.

    ``core_features_demonstrated`` carries the contract that
    ``outcome == PASS`` requires at least one entry; an empty list with
    ``outcome=PASS`` is downgraded to ``PARTIAL`` by ``report_extractor``
    (see ``_enforce_pass_requires_feature``).
    """
    deploys: bool
    quickstart_works: bool
    core_features_demonstrated: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)


@dataclass
class RunReport:
    """Top-level report — matches ``REPORT_JSON_SCHEMA`` exactly."""
    schema_version: Literal["1.0"]
    lane: Literal["project", "skill"]
    target_id: str
    outcome: Outcome
    steps: list[Step]
    verdict: Verdict
    self_assessment_zh: str
    follow_ups: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Dump to plain dict for JSON serialisation + jsonschema validation."""
        d = asdict(self)
        # asdict() leaves enums as Enum objects, not their values; the
        # schema validator expects string literals so we coerce here.
        d["outcome"] = self.outcome.value
        d["steps"] = [
            {**s, "status": s["status"].value if isinstance(s["status"], StepStatus) else s["status"]}
            for s in d["steps"]
        ]
        return d


# ── JSON Schema v1.0 ──────────────────────────────────────────────────────
#
# This is what jsonschema.validate(...) is called with. The contract here
# is the SAME as the RunReport dataclass — when they drift, the parity
# test in test_schema.py::test_schema_dataclass_parity fails loudly.
#
# A few decisions worth calling out:
#
# - ``additionalProperties: false`` is on every object. Agent free-form
#   keys leak into the panel as untrusted strings; better to reject and
#   force the agent to put extra context in ``self_assessment_zh``.
# - ``self_assessment_zh`` is required + max 1024 chars. Panel renders
#   this as a popover-tooltip-style preview; longer text is truncated
#   visually but the JSON layer is hard-capped for storage hygiene.
# - ``steps[*].stdout_tail`` max 2048 chars: we want a quick glance at
#   "what was the last thing on screen?" not a full log dump. Full
#   logs live in agent.log.
# - We deliberately do NOT enforce the "PASS requires non-empty
#   core_features_demonstrated" constraint at the schema level — it's
#   a soft business rule that the extractor downgrades rather than
#   rejects. See ``report_extractor._enforce_pass_requires_feature``.
#
REPORT_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://heyi-eval-v10/schemas/run_report_v1.json",
    "title": "RunReport v1.0",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema_version", "lane", "target_id", "outcome",
        "steps", "verdict", "self_assessment_zh",
    ],
    "properties": {
        "schema_version": {"const": "1.0"},
        "lane": {"enum": ["project", "skill"]},
        "target_id": {"type": "string", "minLength": 1, "maxLength": 256},
        "outcome": {"enum": [o.value for o in Outcome]},
        "steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "status"],
                "properties": {
                    "name": {"type": "string", "minLength": 1, "maxLength": 64},
                    "status": {"enum": [s.value for s in StepStatus]},
                    "duration_s": {"type": "number", "minimum": 0},
                    "note": {"type": "string", "maxLength": 500},
                    "stdout_tail": {"type": "string", "maxLength": 2048},
                    "artifacts": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 256},
                        "maxItems": 32,
                    },
                },
            },
            "maxItems": 32,
        },
        "verdict": {
            "type": "object",
            "additionalProperties": False,
            "required": ["deploys", "quickstart_works"],
            "properties": {
                "deploys": {"type": "boolean"},
                "quickstart_works": {"type": "boolean"},
                "core_features_demonstrated": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 128},
                    "maxItems": 16,
                },
                "blockers": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 500},
                    "maxItems": 16,
                },
            },
        },
        "self_assessment_zh": {
            "type": "string",
            "minLength": 1,
            "maxLength": 1024,
        },
        "follow_ups": {
            "type": "array",
            "items": {"type": "string", "minLength": 1, "maxLength": 200},
            "maxItems": 16,
        },
    },
}


# ── fence markers ─────────────────────────────────────────────────────────
#
# The agent contract: somewhere in stdout there is a block of the form:
#
#     <<<HEYI_RUN_REPORT_JSON>>>
#     {"schema_version": "1.0", ...}
#     <<<END>>>
#
# Multiple blocks → the LAST one wins (agent may produce a draft mid-run
# then revise; the trailing block is the authoritative report). Empty
# block / missing closing fence → REPORT_PARSE_ERROR.
#
# These constants are imported by report_extractor (the only legitimate
# parser) and by tests that construct synthetic agent stdout. They are
# also referenced verbatim in ``docs/ARCHITECTURE_LANES.md §6`` so the
# agent system prompt can quote them.
FENCE_OPEN = "<<<HEYI_RUN_REPORT_JSON>>>"
FENCE_CLOSE = "<<<END>>>"
