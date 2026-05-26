"""Tests for ``agent_driver.report_extractor``.

Covers TEST_PLAN_LANES.md cases:
  S3 — fence missing
  S4 — fence present but JSON invalid / schema mismatch
  H1 fragment — happy fence extraction
  + many edge cases around multi-fence / lane mismatch / soft downgrade
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_driver.report_extractor import (  # noqa: E402
    ExtractError,
    extract_report,
    synthetic_parse_error_report,
)
from agent_driver.schema import (  # noqa: E402
    FENCE_CLOSE,
    FENCE_OPEN,
    Outcome,
    RunReport,
)


def _fence(payload: dict) -> str:
    return f"\n{FENCE_OPEN}\n{json.dumps(payload, ensure_ascii=False)}\n{FENCE_CLOSE}\n"


def _good_payload(**overrides: object) -> dict:
    base = {
        "schema_version": "1.0",
        "lane": "project",
        "target_id": "simonw/llm",
        "outcome": "pass",
        "steps": [{
            "name": "clone", "status": "ok", "duration_s": 1.0,
        }],
        "verdict": {
            "deploys": True, "quickstart_works": True,
            "core_features_demonstrated": ["cli_chat"],
            "blockers": [],
        },
        "self_assessment_zh": "跑通了",
        "follow_ups": [],
    }
    base.update(overrides)  # type: ignore[arg-type]
    return base


class HappyPathTests(unittest.TestCase):

    def test_clean_fence_extracts(self) -> None:
        agent_stdout = "前面的过程日志\n" + _fence(_good_payload())
        r = extract_report(
            agent_stdout, lane="project", target_id="simonw/llm",
        )
        self.assertIsInstance(r, RunReport)
        assert isinstance(r, RunReport)
        self.assertEqual(r.outcome, Outcome.PASS)
        self.assertEqual(r.target_id, "simonw/llm")
        self.assertEqual(r.verdict.core_features_demonstrated, ["cli_chat"])

    def test_extra_log_lines_after_close_are_tolerated(self) -> None:
        agent_stdout = _fence(_good_payload()) + "\nccr log tail noise\n"
        r = extract_report(agent_stdout, lane="project", target_id="simonw/llm")
        self.assertIsInstance(r, RunReport)


class MultipleFenceTests(unittest.TestCase):
    """Agent draft-then-revise: write a tentative report, then a final
    block. Last fence wins (per docstring contract)."""

    def test_last_fence_wins(self) -> None:
        draft = _good_payload(outcome="partial")
        final = _good_payload(outcome="pass")
        stdout = _fence(draft) + "...更多思考...\n" + _fence(final)
        r = extract_report(stdout, lane="project", target_id="simonw/llm")
        assert isinstance(r, RunReport)
        self.assertEqual(r.outcome, Outcome.PASS)

    def test_open_without_close_after_a_complete_fence_is_ignored(self) -> None:
        """If the final open marker has no matching close, we treat the
        whole sequence as malformed even though earlier fences are
        complete. This is conservative: the agent's last word was
        unfinished, so we don't trust its earlier draft either."""
        first = _fence(_good_payload(outcome="partial"))
        stdout = first + f"\n{FENCE_OPEN}\n{{\"...truncated"  # no close
        r = extract_report(stdout, lane="project", target_id="simonw/llm")
        self.assertIsInstance(r, ExtractError)
        assert isinstance(r, ExtractError)
        self.assertEqual(r.kind, "fence_missing")


class FailureModeTests(unittest.TestCase):
    """S3 / S4 from TEST_PLAN_LANES.md."""

    def test_no_fence_at_all(self) -> None:
        r = extract_report(
            "完全自由发挥的 markdown 总结\n没有围栏",
            lane="project", target_id="simonw/llm",
        )
        self.assertIsInstance(r, ExtractError)
        assert isinstance(r, ExtractError)
        self.assertEqual(r.kind, "fence_missing")

    def test_empty_fence(self) -> None:
        r = extract_report(
            f"{FENCE_OPEN}\n   \n{FENCE_CLOSE}",
            lane="project", target_id="simonw/llm",
        )
        self.assertIsInstance(r, ExtractError)
        assert isinstance(r, ExtractError)
        self.assertEqual(r.kind, "fence_empty")

    def test_invalid_json(self) -> None:
        r = extract_report(
            f"{FENCE_OPEN}\n{{not json{FENCE_CLOSE}",
            lane="project", target_id="simonw/llm",
        )
        self.assertIsInstance(r, ExtractError)
        assert isinstance(r, ExtractError)
        self.assertEqual(r.kind, "json_invalid")

    def test_schema_violation_missing_required(self) -> None:
        bad = _good_payload()
        del bad["self_assessment_zh"]
        r = extract_report(_fence(bad), lane="project", target_id="simonw/llm")
        self.assertIsInstance(r, ExtractError)
        assert isinstance(r, ExtractError)
        self.assertEqual(r.kind, "schema_invalid")

    def test_schema_violation_wrong_type(self) -> None:
        bad = _good_payload()
        bad["verdict"]["deploys"] = "yes"  # type: ignore[index]
        r = extract_report(_fence(bad), lane="project", target_id="simonw/llm")
        self.assertIsInstance(r, ExtractError)
        assert isinstance(r, ExtractError)
        self.assertEqual(r.kind, "schema_invalid")

    def test_lane_mismatch(self) -> None:
        r = extract_report(
            _fence(_good_payload(lane="skill")),
            lane="project", target_id="simonw/llm",
        )
        self.assertIsInstance(r, ExtractError)
        assert isinstance(r, ExtractError)
        self.assertEqual(r.kind, "lane_mismatch")

    def test_target_mismatch(self) -> None:
        r = extract_report(
            _fence(_good_payload(target_id="evil/repo")),
            lane="project", target_id="simonw/llm",
        )
        self.assertIsInstance(r, ExtractError)
        assert isinstance(r, ExtractError)
        self.assertEqual(r.kind, "target_mismatch")


class SoftDowngradeTests(unittest.TestCase):
    """``outcome=PASS`` with empty core_features_demonstrated is
    downgraded to PARTIAL — not rejected, but the panel must see a
    truthful state."""

    def test_pass_without_features_downgrades_to_partial(self) -> None:
        bad = _good_payload()
        bad["verdict"]["core_features_demonstrated"] = []  # type: ignore[index]
        r = extract_report(_fence(bad), lane="project", target_id="simonw/llm")
        assert isinstance(r, RunReport)
        self.assertEqual(r.outcome, Outcome.PARTIAL)
        self.assertIn("[系统降级]", r.self_assessment_zh)

    def test_partial_without_features_stays_partial(self) -> None:
        """No downgrade if agent already said partial."""
        bad = _good_payload(outcome="partial")
        bad["verdict"]["core_features_demonstrated"] = []  # type: ignore[index]
        r = extract_report(_fence(bad), lane="project", target_id="simonw/llm")
        assert isinstance(r, RunReport)
        self.assertEqual(r.outcome, Outcome.PARTIAL)
        self.assertNotIn("[系统降级]", r.self_assessment_zh)


class SyntheticReportTests(unittest.TestCase):

    def test_zh_assessment_for_each_kind(self) -> None:
        kinds = [
            "fence_missing", "fence_empty", "json_invalid",
            "schema_invalid", "lane_mismatch", "target_mismatch",
        ]
        for kind in kinds:
            err = ExtractError(kind=kind, detail="x")  # type: ignore[arg-type]
            zh = err.to_zh_assessment()
            self.assertTrue(zh, msg=f"empty zh for kind={kind}")
            # All Chinese assessments must be non-trivial; this is what
            # the panel surface, so empty would be a regression.
            self.assertGreater(len(zh), 8, msg=f"zh too short for kind={kind}")

    def test_synthetic_report_has_parse_error_outcome(self) -> None:
        err = ExtractError(kind="fence_missing", detail="x")
        r = synthetic_parse_error_report(
            err, lane="project", target_id="x/y",
        )
        self.assertEqual(r.outcome, Outcome.REPORT_PARSE_ERROR)
        self.assertEqual(r.target_id, "x/y")
        self.assertIn("围栏", r.self_assessment_zh)


if __name__ == "__main__":
    unittest.main()
