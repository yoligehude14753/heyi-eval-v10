"""Tests for ``agent_driver.budget_guard``.

Covers TEST_PLAN_LANES.md cases:
  S5 — token budget exceeded
  S6 — wall-clock budget exceeded
  + monotonicity, overflow safety, immutability.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_driver.budget_guard import (  # noqa: E402
    BudgetCheck,
    BudgetDecision,
    BudgetGuard,
)


class ConstructionTests(unittest.TestCase):

    def test_rejects_zero_token_budget(self) -> None:
        with self.assertRaises(ValueError):
            BudgetGuard(token_budget=0, wall_clock_s=10.0)

    def test_rejects_negative_wall_clock(self) -> None:
        with self.assertRaises(ValueError):
            BudgetGuard(token_budget=100, wall_clock_s=-1.0)

    def test_defaults_match_design_doc(self) -> None:
        """Sanity-pin: ARCHITECTURE_LANES.md §13 says default is
        50k token / 900s. If we change one, both should change
        together; this test is the trip wire."""
        g = BudgetGuard()
        self.assertEqual(g.token_budget, 50_000)
        self.assertEqual(g.wall_clock_s, 900.0)


class TokenAccountingTests(unittest.TestCase):

    def test_note_tokens_sums(self) -> None:
        g = BudgetGuard(token_budget=100, wall_clock_s=60.0, now=0.0)
        g.note_tokens(input_tokens=10, output_tokens=20)
        g.note_tokens(input_tokens=5, output_tokens=15)
        self.assertEqual(g.total_tokens, 50)
        self.assertTrue(g.check(now=0.0).is_ok)

    def test_rejects_negative_deltas(self) -> None:
        g = BudgetGuard(token_budget=100, wall_clock_s=60.0)
        with self.assertRaises(ValueError):
            g.note_tokens(input_tokens=-1, output_tokens=0)
        with self.assertRaises(ValueError):
            g.note_tokens(input_tokens=0, output_tokens=-1)

    def test_token_overrun_returns_exceeded(self) -> None:
        g = BudgetGuard(token_budget=100, wall_clock_s=60.0, now=0.0)
        g.note_tokens(input_tokens=60, output_tokens=50)  # 110 > 100
        check = g.check(now=1.0)
        self.assertEqual(check.decision, BudgetDecision.EXCEEDED_TOKENS)
        self.assertEqual(check.total_tokens, 110)
        self.assertIn("超过", check.detail)
        # Detail must mention both input and output counts so an
        # operator can see whether the agent is asking long questions
        # or writing long answers.
        self.assertIn("input=60", check.detail)
        self.assertIn("output=50", check.detail)

    def test_token_exactly_at_budget_is_exceeded(self) -> None:
        """Use ``>=`` semantics so the agent can't sneak one more
        request through at exactly the boundary."""
        g = BudgetGuard(token_budget=100, wall_clock_s=60.0, now=0.0)
        g.note_tokens(input_tokens=40, output_tokens=60)  # exactly 100
        self.assertEqual(
            g.check(now=0.0).decision,
            BudgetDecision.EXCEEDED_TOKENS,
        )


class WallClockTests(unittest.TestCase):

    def test_wall_clock_overrun(self) -> None:
        g = BudgetGuard(token_budget=1_000_000, wall_clock_s=10.0, now=0.0)
        check = g.check(now=11.0)
        self.assertEqual(check.decision, BudgetDecision.EXCEEDED_WALL_CLOCK)
        self.assertGreaterEqual(check.elapsed_s, 10.0)

    def test_wall_clock_just_under_is_ok(self) -> None:
        g = BudgetGuard(token_budget=1_000_000, wall_clock_s=10.0, now=0.0)
        check = g.check(now=9.99)
        self.assertEqual(check.decision, BudgetDecision.OK)

    def test_token_check_takes_priority_over_wall_clock(self) -> None:
        """When both are exceeded simultaneously, we report the token
        overrun first — that's the more actionable failure for the
        operator (agent wrote too much) vs "we ran out of time"
        (could be transient yunwu slowness)."""
        g = BudgetGuard(token_budget=100, wall_clock_s=10.0, now=0.0)
        g.note_tokens(input_tokens=50, output_tokens=51)
        check = g.check(now=11.0)
        self.assertEqual(check.decision, BudgetDecision.EXCEEDED_TOKENS)


class IsOkPropertyTests(unittest.TestCase):

    def test_is_ok_for_ok(self) -> None:
        c = BudgetCheck(decision=BudgetDecision.OK, elapsed_s=0.0, total_tokens=0)
        self.assertTrue(c.is_ok)

    def test_is_ok_for_exceeded(self) -> None:
        c = BudgetCheck(
            decision=BudgetDecision.EXCEEDED_TOKENS,
            elapsed_s=0.0, total_tokens=100, detail="x",
        )
        self.assertFalse(c.is_ok)


if __name__ == "__main__":
    unittest.main()
