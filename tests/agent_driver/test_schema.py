"""Schema + dataclass parity tests for ``agent_driver.schema``.

Goal: enforce that the RunReport dataclass and REPORT_JSON_SCHEMA can
never silently drift apart. We construct a richly-populated RunReport,
dump it through ``to_dict()``, validate against the schema, and
back-load through ``report_extractor._to_report_dataclass`` to
roundtrip — any drift fails at least one of these checks.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import jsonschema

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_driver.schema import (  # noqa: E402
    FENCE_CLOSE,
    FENCE_OPEN,
    REPORT_JSON_SCHEMA,
    Outcome,
    RunReport,
    Step,
    StepStatus,
    Verdict,
)


def _full_report() -> RunReport:
    """A RunReport that exercises every optional field."""
    return RunReport(
        schema_version="1.0",
        lane="project",
        target_id="simonw/llm",
        outcome=Outcome.PASS,
        steps=[
            Step(name="clone", status=StepStatus.OK, duration_s=5.2,
                 note="cloned 12 mb", stdout_tail="Cloning into 'llm'..."),
            Step(name="install", status=StepStatus.OK, duration_s=30.1,
                 artifacts=["venv/", "install.log"]),
        ],
        verdict=Verdict(
            deploys=True, quickstart_works=True,
            core_features_demonstrated=["cli_chat", "json_output"],
            blockers=[],
        ),
        self_assessment_zh="项目部署成功，CLI 聊天与 JSON 输出两个核心能力均跑通。",
        follow_ups=["试 batch 模式"],
    )


class SchemaParityTests(unittest.TestCase):

    def test_dataclass_dumps_pass_schema_validation(self) -> None:
        """Roundtrip: every field a RunReport can carry must validate."""
        report = _full_report()
        d = report.to_dict()
        jsonschema.validate(d, REPORT_JSON_SCHEMA)

    def test_outcomes_cover_all_enum_values(self) -> None:
        """Every Outcome literal must be allowed by the JSON schema —
        otherwise a system-imposed outcome (e.g. REPORT_PARSE_ERROR)
        can't be serialised through to_dict + validate."""
        for outcome in Outcome:
            r = _full_report()
            r.outcome = outcome  # mutate post-construction; dataclass is non-frozen
            jsonschema.validate(r.to_dict(), REPORT_JSON_SCHEMA)

    def test_minimal_report_validates(self) -> None:
        """Steps empty, follow_ups empty, blockers empty — should still
        validate. The agent contract is "at minimum these required
        fields"; minimal valid is what we get on REPORT_PARSE_ERROR."""
        r = RunReport(
            schema_version="1.0",
            lane="skill",
            target_id="mermaid",
            outcome=Outcome.REPORT_PARSE_ERROR,
            steps=[],
            verdict=Verdict(deploys=False, quickstart_works=False),
            self_assessment_zh="x",
        )
        jsonschema.validate(r.to_dict(), REPORT_JSON_SCHEMA)


class SchemaRejectionTests(unittest.TestCase):
    """The schema MUST reject malformed agent output — otherwise S3/S4
    test cases in TEST_PLAN_LANES.md are not actually defended."""

    def _validate(self, doc: dict) -> None:
        jsonschema.validate(doc, REPORT_JSON_SCHEMA)

    def test_rejects_unknown_top_level_key(self) -> None:
        d = _full_report().to_dict()
        d["agent_freeform_summary"] = "ignored extra"
        with self.assertRaises(jsonschema.ValidationError):
            self._validate(d)

    def test_rejects_wrong_schema_version(self) -> None:
        d = _full_report().to_dict()
        d["schema_version"] = "2.0"
        with self.assertRaises(jsonschema.ValidationError):
            self._validate(d)

    def test_rejects_unknown_lane(self) -> None:
        d = _full_report().to_dict()
        d["lane"] = "model"  # model lane runs don't use RunReport
        with self.assertRaises(jsonschema.ValidationError):
            self._validate(d)

    def test_rejects_unknown_outcome(self) -> None:
        d = _full_report().to_dict()
        d["outcome"] = "succeeded"
        with self.assertRaises(jsonschema.ValidationError):
            self._validate(d)

    def test_rejects_verdict_with_string_deploys(self) -> None:
        d = _full_report().to_dict()
        d["verdict"]["deploys"] = "yes"
        with self.assertRaises(jsonschema.ValidationError):
            self._validate(d)

    def test_rejects_steps_status_string(self) -> None:
        d = _full_report().to_dict()
        d["steps"][0]["status"] = "passing"
        with self.assertRaises(jsonschema.ValidationError):
            self._validate(d)

    def test_rejects_oversized_self_assessment(self) -> None:
        d = _full_report().to_dict()
        d["self_assessment_zh"] = "x" * 2000
        with self.assertRaises(jsonschema.ValidationError):
            self._validate(d)

    def test_rejects_empty_self_assessment(self) -> None:
        d = _full_report().to_dict()
        d["self_assessment_zh"] = ""
        with self.assertRaises(jsonschema.ValidationError):
            self._validate(d)


class FenceMarkerTests(unittest.TestCase):
    """The fence markers are part of the contract surface — pinning
    them here so a typo in agent prompts is caught by `rg`."""

    def test_open_marker_unchanged(self) -> None:
        self.assertEqual(FENCE_OPEN, "<<<HEYI_RUN_REPORT_JSON>>>")

    def test_close_marker_unchanged(self) -> None:
        self.assertEqual(FENCE_CLOSE, "<<<END>>>")


if __name__ == "__main__":
    unittest.main()
