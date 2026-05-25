"""PR#17: SHOWCASE _strip_think — reasoning-model <think> chain filter.

Reasoning models (DeepSeek-R1, Qwen3-Thinking, MiniMax-M2.7, etc.)
emit an internal reasoning chain inside ``<think>...</think>`` tags
before the actual answer. Prior to PR#17, the grader LLM saw the
raw output truncated at 600 chars — which usually only contained
reasoning and never the final answer.

PR#17 strips <think>/<thinking> blocks before grading.
"""
from __future__ import annotations

import unittest

from cc_agent.showcase_runner import _render_items_for_grading, _strip_think

# ── T series: pure unit tests for _strip_think ────────────────────────────


class StripThinkUnits(unittest.TestCase):

    def test_t1_simple_block_removed(self):
        s = "<think>let me compute 2+2</think>The answer is 4."
        self.assertEqual(_strip_think(s), "The answer is 4.")

    def test_t2_thinking_synonym(self):
        s = "<thinking>hmm</thinking>Final: 7"
        self.assertEqual(_strip_think(s), "Final: 7")

    def test_t3_case_insensitive(self):
        s = "<THINK>reasoning</THINK>OK"
        self.assertEqual(_strip_think(s), "OK")
        s = "<Thinking>r</Thinking>OK"
        self.assertEqual(_strip_think(s), "OK")

    def test_t4_multiline_block(self):
        s = ("<think>\n  step 1: try X\n  step 2: try Y\n"
             "  step 3: aha!\n</think>\nThe answer is therefore 42.")
        self.assertEqual(_strip_think(s), "The answer is therefore 42.")

    def test_t5_multiple_blocks_all_removed(self):
        s = ("<think>first</think>partial<thinking>second</thinking> "
             "more text <think>third</think>final.")
        self.assertEqual(_strip_think(s), "partial more text final.")

    def test_t6_unmatched_opening_truncates_to_empty_answer(self):
        # Model was cut off mid-reasoning — no final answer exists.
        s = "<think>I will compute this step by step. First,"
        self.assertEqual(_strip_think(s), "")

    def test_t7_unmatched_opening_with_prefix_keeps_prefix(self):
        s = "Preamble before reasoning. <think>now I reason..."
        self.assertEqual(_strip_think(s), "Preamble before reasoning.")

    def test_t8_no_think_tags_returns_input_stripped(self):
        s = "  Simply 18.  "
        self.assertEqual(_strip_think(s), "Simply 18.")

    def test_t9_empty_string_returns_empty(self):
        self.assertEqual(_strip_think(""), "")
        # Treat None-derived empty consistently — code in showcase_runner
        # passes ``(it.get("actual") or "")`` so we never receive None,
        # but be defensive about the empty-string contract.

    def test_t10_whitespace_only_inside_block(self):
        s = "<think>   \n   </think>Just the answer."
        self.assertEqual(_strip_think(s), "Just the answer.")

    def test_t11_tag_with_extra_spaces(self):
        s = "< think >reasoning< / think >answer here"
        self.assertEqual(_strip_think(s), "answer here")

    def test_t12_nested_text_with_angle_brackets_inside_block(self):
        # The DOTALL+non-greedy regex stops at the first </think>, but
        # angle brackets in the reasoning shouldn't break it.
        s = ("<think>I'd compare a < b and 5 > 3 first.</think>"
             "Answer: yes.")
        self.assertEqual(_strip_think(s), "Answer: yes.")

    def test_t13_block_at_end_of_string(self):
        s = "Some text <think>and reasoning</think>"
        self.assertEqual(_strip_think(s), "Some text")


# ── I series: integration with _render_items_for_grading ─────────────────


class RenderItemsForGradingIntegration(unittest.TestCase):

    def test_i1_actual_field_is_stripped_before_truncate(self):
        # 700-char reasoning chain + a clear short answer at the end.
        long_think = "<think>" + ("x" * 700) + "</think>The answer is 42."
        items = [{
            "id": "demo-001",
            "prompt": "What is the answer?",
            "rationale": "demo",
            "actual": long_think,
        }]
        rendered = _render_items_for_grading(items)
        # The 600-char truncation must NOT cut the answer because the
        # whole <think> block is stripped before truncation.
        self.assertIn("The answer is 42.", rendered)
        self.assertNotIn("<think>", rendered)
        self.assertNotIn("xxx", rendered)

    def test_i2_missing_actual_safe(self):
        items = [{"id": "x", "prompt": "p", "rationale": "r"}]
        rendered = _render_items_for_grading(items)
        self.assertIn("Model output: ", rendered)

    def test_i3_actual_explicit_none_safe(self):
        items = [{"id": "x", "prompt": "p", "rationale": "r", "actual": None}]
        rendered = _render_items_for_grading(items)
        self.assertIn("Model output: ", rendered)

    def test_i4_no_think_chain_behaves_as_before(self):
        items = [{
            "id": "x", "prompt": "p", "rationale": "r",
            "actual": "Direct answer: 42",
        }]
        rendered = _render_items_for_grading(items)
        self.assertIn("Direct answer: 42", rendered)


if __name__ == "__main__":
    unittest.main()
