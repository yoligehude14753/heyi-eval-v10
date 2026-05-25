"""PR#25: two surgical cleanups exposed by the PR#24 real-nv8 run.

1. ``orchestrator/llm_text_utils.strip_think_blocks`` strips the
   ``<think>...</think>`` chain-of-thought wrapper that reasoning
   models (MiniMax-M2.7, R1, Qwen3-thinking) emit before the real
   answer. Wired into the judge JSON parser AND the showcase
   planner JSON-array extractor, so neither path falls through to
   default when the model is "thinky".

2. ``orchestrator.stages_py._graceful_skip`` ``error_kind`` is now
   parameterised. The model-missing path now correctly reports
   ``error_kind="model_missing"`` instead of the legacy
   ``insufficient_gpu`` catch-all.
"""
from __future__ import annotations

import unittest

from orchestrator.llm_text_utils import strip_think_blocks


class TestStripThinkBlocks(unittest.TestCase):
    def test_empty_string_passthrough(self) -> None:
        self.assertEqual(strip_think_blocks(""), "")

    def test_no_think_tag_unchanged(self) -> None:
        s = '{"pass": true, "reason": "ok"}'
        self.assertEqual(strip_think_blocks(s), s)

    def test_basic_block_removed(self) -> None:
        s = '<think>I should output JSON.</think>{"pass": true}'
        self.assertEqual(strip_think_blocks(s), '{"pass": true}')

    def test_multiline_block_removed(self) -> None:
        s = (
            '<think>\nThe user wants 5 items. Let me brainstorm:\n'
            '- item one\n- item two\n</think>\n'
            '[{"id": "a", "prompt": "p"}]'
        )
        self.assertEqual(strip_think_blocks(s).strip(),
                         '[{"id": "a", "prompt": "p"}]')

    def test_case_insensitive_tag(self) -> None:
        s = '<THINK>x</THINK>{"pass": false}'
        self.assertEqual(strip_think_blocks(s), '{"pass": false}')

    def test_whitespace_in_tag_tolerated(self) -> None:
        s = '< think >x< / think >{"pass": false}'
        self.assertEqual(strip_think_blocks(s), '{"pass": false}')

    def test_multiple_think_blocks_all_removed(self) -> None:
        s = '<think>a</think>{"x":1}<think>b</think>{"y":2}'
        # both blocks stripped, JSON snippets preserved
        out = strip_think_blocks(s)
        self.assertNotIn("<think", out.lower())
        self.assertIn('{"x":1}', out)
        self.assertIn('{"y":2}', out)

    def test_unclosed_think_tail_truncated(self) -> None:
        # Model cut off mid-CoT; we drop from the tag to EOF (better
        # to lose half a sentence than confuse the parser).
        s = '{"pass": true}<think>I was about to say'
        self.assertEqual(strip_think_blocks(s), '{"pass": true}')

    def test_idempotent(self) -> None:
        s = '<think>x</think>{"pass": true}'
        once = strip_think_blocks(s)
        twice = strip_think_blocks(once)
        self.assertEqual(once, twice)

    def test_think_inside_string_left_alone_when_no_tag(self) -> None:
        # The function is tag-based, not word-based; the literal
        # word "think" in JSON values must survive.
        s = '{"pass": true, "reason": "I think yes"}'
        self.assertEqual(strip_think_blocks(s), s)


class TestJudgeParserHandlesThink(unittest.TestCase):
    def test_m27_style_response_parses(self) -> None:
        from orchestrator.llm_judge import _parse_judge_json
        # Real M2.7 shape: CoT block before the JSON, with braces
        # inside the CoT to test that the find/rfind fallback was
        # actually defeated by stripping rather than coincidence.
        text = (
            '<think>The user asked me to return {"pass": false}.\n'
            'But the actual artifact matches, so I should return '
            'pass: true with a brief reason.</think>\n'
            '{"pass": true, "reason": "artifact matches expected"}'
        )
        p, reason = _parse_judge_json(text)
        self.assertTrue(p)
        self.assertEqual(reason, "artifact matches expected")

    def test_no_think_still_parses_baseline(self) -> None:
        from orchestrator.llm_judge import _parse_judge_json
        p, reason = _parse_judge_json('{"pass": false, "reason": "no"}')
        self.assertFalse(p)
        self.assertEqual(reason, "no")


class TestShowcasePlannerHandlesThink(unittest.TestCase):
    def test_planner_extractor_strips_think(self) -> None:
        from cc_agent.showcase_runner import _extract_json_array
        text = (
            '<think>I will draft 2 items.\nFor instance: '
            '[{"id": "draft", "prompt": "TBD"}] is not real.</think>\n'
            '[{"id": "real1", "prompt": "p1"}, {"id": "real2", "prompt": "p2"}]'
        )
        parsed = _extract_json_array(text)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]["id"], "real1")
        self.assertEqual(parsed[1]["id"], "real2")

    def test_planner_baseline_without_think(self) -> None:
        from cc_agent.showcase_runner import _extract_json_array
        parsed = _extract_json_array('[{"id":"x","prompt":"y"}]')
        self.assertIsNotNone(parsed)


class TestPlannerFallbackDiagnostic(unittest.TestCase):
    """PR#25: when the planner falls back, the log must name the
    cause so operators know whether they're hit by reasoning-model
    truncation (M2.7 / R1), malformed JSON, or a totally unrelated
    response.
    """

    def _run_with(self, text: str) -> str:
        import io
        import logging
        from unittest import mock

        from cc_agent import showcase_runner

        # Capture the WARNING log line.
        buf = io.StringIO()
        h = logging.StreamHandler(buf)
        h.setLevel(logging.WARNING)
        showcase_runner.log.addHandler(h)
        showcase_runner.log.setLevel(logging.WARNING)
        try:
            mock_client = mock.MagicMock()
            mock_client.call.return_value = mock.MagicMock(text=text)
            out = showcase_runner._plan_items(
                mock_client, n_items=5, hf_id="x", curated={},
                modelcard="", timeout_s=10.0,
            )
            self.assertEqual(out, [])
            return buf.getvalue()
        finally:
            showcase_runner.log.removeHandler(h)

    def test_thinking_truncated_cause(self) -> None:
        # Open <think> never closes — M2.7 ran out of max_tokens.
        log = self._run_with("<think>Let me plan items...")
        self.assertIn("cause=thinking_truncated", log)

    def test_thinking_only_cause(self) -> None:
        # Closed <think> block but no [ array marker after it.
        log = self._run_with("<think>done thinking</think>I refuse to comply.")
        self.assertIn("cause=thinking_only", log)

    def test_no_array_marker_cause(self) -> None:
        log = self._run_with("plain prose with no JSON brackets at all")
        self.assertIn("cause=no_array_marker", log)

    def test_malformed_json_cause(self) -> None:
        log = self._run_with("[{this is not valid JSON]")
        self.assertIn("cause=malformed_json", log)


class TestGracefulSkipErrorKindParametrized(unittest.TestCase):
    def test_default_keeps_insufficient_gpu(self) -> None:
        # Old callers stay backward-compatible.
        from orchestrator.stages_py import _graceful_skip
        r = _graceful_skip(0.0, "test reason")
        self.assertEqual(r.error_kind, "insufficient_gpu")
        self.assertTrue(r.error.startswith("insufficient_gpu:"))

    def test_explicit_kind_propagates(self) -> None:
        from orchestrator.stages_py import _graceful_skip
        r = _graceful_skip(0.0, "no model", error_kind="model_missing")
        self.assertEqual(r.error_kind, "model_missing")
        self.assertTrue(r.error.startswith("model_missing:"))
        self.assertEqual(r.extra.get("reason"), "no model")
        self.assertTrue(r.extra.get("aborted"))


if __name__ == "__main__":
    unittest.main()
