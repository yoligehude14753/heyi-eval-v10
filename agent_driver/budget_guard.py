"""Token + wall-clock dual upper bound on agent runs (INV-P4).

The guard is **passive**: it owns a clock + accumulator, and exposes
``check()`` which callers (exec_runner's stream loop) invoke between
ccr-log chunks to decide whether to keep going. There's no async
thread, no signal handler — the guard never interrupts anything on its
own. This keeps it deterministic in unit tests and predictable in
production.

Why is this safer than e.g. a daemon thread firing SIGTERM?

- The agent process inside m2b is a child of ccr which is a child of
  PID 1 (tini). SIGTERM-from-outside semantics through that chain are
  fragile (zombies, half-killed shells).
- exec_runner already wraps the docker exec stream in a loop; the loop
  is a natural and observable interrupt point.
- Token accounting is a side channel (we read ccr's per-request log
  lines), so even if we wanted to interrupt asynchronously we'd need
  to re-check token totals against the budget anyway.

Contract:

- ``BudgetGuard(token_budget=50_000, wall_clock_s=900.0)`` constructor
- ``guard.note_tokens(input_tokens=, output_tokens=)`` called each time
  ccr's log emits a usage line
- ``guard.check()`` returns ``BudgetCheck`` — caller decides what to do
- when ``decision`` is ``EXCEEDED_TOKENS`` or ``EXCEEDED_WALL_CLOCK``,
  exec_runner stops streaming + writes ``outcome=BUDGET_EXCEEDED`` or
  ``TIMEOUT`` into a synthetic RunReport
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum


class BudgetDecision(str, Enum):
    OK = "ok"
    EXCEEDED_TOKENS = "exceeded_tokens"
    EXCEEDED_WALL_CLOCK = "exceeded_wall_clock"


@dataclass(frozen=True)
class BudgetCheck:
    """One snapshot of guard state, returned by ``BudgetGuard.check()``.

    ``elapsed_s`` and ``total_tokens`` are always present (cheap to
    compute). ``detail`` is empty unless ``decision`` is one of the
    EXCEEDED_* states, in which case it is a short human-readable
    explanation suitable for the panel's failure-reason column.
    """
    decision: BudgetDecision
    elapsed_s: float
    total_tokens: int
    detail: str = ""

    @property
    def is_ok(self) -> bool:
        return self.decision is BudgetDecision.OK


class BudgetGuard:
    """Track token + wall-clock usage; surface a verdict per ``check()``.

    Defaults are pulled from ``docs/ARCHITECTURE_LANES.md §13``:
        token_budget = 50_000  (input + output combined)
        wall_clock_s = 900     (15 minutes)
    """

    def __init__(
        self,
        *,
        token_budget: int = 50_000,
        wall_clock_s: float = 900.0,
        now: float | None = None,
    ) -> None:
        if token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {token_budget}")
        if wall_clock_s <= 0:
            raise ValueError(f"wall_clock_s must be positive, got {wall_clock_s}")
        self._token_budget = token_budget
        self._wall_clock_s = wall_clock_s
        # ``now`` is overridable for deterministic tests; production
        # always uses ``time.monotonic()`` so wall-clock can't go
        # backwards on NTP slew.
        self._t0 = now if now is not None else time.monotonic()
        self._input_tokens = 0
        self._output_tokens = 0

    # ── accounting ──────────────────────────────────────────────────────

    def note_tokens(self, *, input_tokens: int, output_tokens: int) -> None:
        """Add to the running totals. Both args must be non-negative.

        ccr's log lines look like ``"usage": {"prompt_tokens": 1234,
        "completion_tokens": 56}`` — exec_runner's ccr-log tailer should
        map prompt→input, completion→output, then call this.
        """
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError(
                f"token deltas must be non-negative, "
                f"got input={input_tokens} output={output_tokens}"
            )
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens

    # ── verdict ─────────────────────────────────────────────────────────

    def check(self, *, now: float | None = None) -> BudgetCheck:
        """Return current state. Pure — call as often as you want."""
        elapsed = (now if now is not None else time.monotonic()) - self._t0
        total = self._input_tokens + self._output_tokens

        if total >= self._token_budget:
            return BudgetCheck(
                decision=BudgetDecision.EXCEEDED_TOKENS,
                elapsed_s=elapsed,
                total_tokens=total,
                detail=(
                    f"agent 累计消耗 {total} tokens "
                    f"(input={self._input_tokens}, output={self._output_tokens})，"
                    f"超过单 run 上限 {self._token_budget}"
                ),
            )
        if elapsed >= self._wall_clock_s:
            return BudgetCheck(
                decision=BudgetDecision.EXCEEDED_WALL_CLOCK,
                elapsed_s=elapsed,
                total_tokens=total,
                detail=(
                    f"agent 已运行 {elapsed:.1f}s，"
                    f"超过单 run 墙钟上限 {self._wall_clock_s:.0f}s"
                ),
            )
        return BudgetCheck(
            decision=BudgetDecision.OK,
            elapsed_s=elapsed,
            total_tokens=total,
        )

    # ── inspection (no setters; immutable budgets) ──────────────────────

    @property
    def token_budget(self) -> int:
        return self._token_budget

    @property
    def wall_clock_s(self) -> float:
        return self._wall_clock_s

    @property
    def total_tokens(self) -> int:
        return self._input_tokens + self._output_tokens
