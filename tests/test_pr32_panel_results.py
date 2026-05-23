"""PR#32 follow-up: panel /results page rendering was hiding all
successful evaluation data behind one failed run with a 4 KB vLLM
ANSI traceback dumped raw into a single cell.

User reported: '/results' shows '总评测 RUNS 41 / 完成 OK 13' in the
top stat tiles but the table below shows only ONE row (failed
GLM-OCR). All 41 rows ARE in the HTML, but the GLM-OCR row's
failure_reason cell expanded to ~4 KB of text with no height/overflow
control, pushing every other row below the fold.

These tests pin two fixes:
  1. Rows are sorted so status=ok WITH capability/pass_rate data
     come first, then ok-without-data, then in_progress, then
     failed, then aborted. The top of the page is always the
     useful data.
  2. failure_reason cells are truncated to 200 chars + a tooltip
     with the full content (via title= attribute) + max-height + 
     overflow:hidden inline style.
"""
from __future__ import annotations

import unittest
from unittest import mock

from panel import server


class TestRowPrioritySort(unittest.TestCase):
    """The internal _row_priority used to sort the results table."""

    def _rows(self):
        return [
            {"hf_id": "fail/foo", "status": "failed",
             "failure_reason": "X" * 5000},
            {"hf_id": "abort/foo", "status": "aborted",
             "failure_reason": "oversize"},
            {"hf_id": "ok-cap/foo", "status": "ok", "pass_rate": 0.71,
             "capability": "25/35"},
            {"hf_id": "ok-cap-low/foo", "status": "ok", "pass_rate": 0.48,
             "capability": "12/25"},
            {"hf_id": "ok-nodata/foo", "status": "ok"},
            {"hf_id": "in_prog/foo", "status": "in_progress"},
            {"hf_id": "ok-cap-high/foo", "status": "ok", "pass_rate": 1.0,
             "capability": "5/5"},
        ]

    def test_sort_priority_buckets(self) -> None:
        rows = self._rows()
        sorted_ids = sorted(rows, key=server.render_results_page.__globals__["_row_priority"]
                            if "_row_priority" in server.render_results_page.__globals__
                            else lambda x: 0)
        # The function is defined inside render_results_page, so we
        # exercise the full render path and verify the table-body
        # ordering instead.
        with mock.patch.object(server, "results_leaderboard",
                               return_value={
                                   "total": len(rows),
                                   "completed_ok": 3,
                                   "failed": 1,
                                   "in_progress": 1,
                                   "avg_pass_rate": 0.6,
                                   "rows": rows,
                               }):
            html = server.render_results_page()

        # ok-cap-high (1.0) must appear before ok-cap (0.71) must
        # appear before ok-cap-low (0.48) must appear before
        # ok-nodata, in_progress, failed, aborted.
        def pos(needle: str) -> int:
            i = html.find(needle)
            self.assertGreaterEqual(i, 0, f"{needle!r} not in page")
            return i

        order = [
            "ok-cap-high/foo",
            "ok-cap/foo",
            "ok-cap-low/foo",
            "ok-nodata/foo",
            "in_prog/foo",
            "fail/foo",
            "abort/foo",
        ]
        positions = [pos(name) for name in order]
        for prev, curr, name in zip(positions, positions[1:], order[1:]):
            self.assertLess(prev, curr,
                            f"{name!r} should come AFTER previous group")


class TestFailureReasonTruncated(unittest.TestCase):
    """The vLLM error log (real-world ~4 KB) must NOT be rendered raw."""

    def test_long_failure_truncated_to_200(self) -> None:
        long_err = ("(APIServer pid=1) " * 200)  # ~4 KB
        rows = [{
            "hf_id": "z/glm-ocr",
            "status": "failed",
            "failure_reason": long_err,
        }]
        with mock.patch.object(server, "results_leaderboard",
                               return_value={
                                   "total": 1, "completed_ok": 0,
                                   "failed": 1, "in_progress": 0,
                                   "avg_pass_rate": None,
                                   "rows": rows,
                               }):
            html = server.render_results_page()

        # 200-char truncation + ellipsis is present.
        self.assertIn("…", html)
        # max-height inline style is present (overflow control).
        self.assertIn("max-height:80px", html)
        # Full text is preserved in title= attribute for tooltip.
        # (look for a substring unique to the error)
        self.assertIn("APIServer pid=1", html)
        # And the cell does NOT contain the raw 4 KB inline (after
        # the 200-char truncation we should see far less than 4 KB
        # in the visible cell body — the title attr is searchable
        # but inside an attribute).
        body_after_title = html.split('title="')[-1].split('">', 1)[-1]
        # The visible-cell portion (after `title="..."` close + ">") 
        # must contain only the truncated `fail` (200 chars + …).
        # The full input had 200 copies — visible cell must have far
        # fewer (≤ 15 fits in 200-char truncation budget; raw render
        # would have 200).
        first_visible_chunk = body_after_title[:400]
        self.assertLess(first_visible_chunk.count("APIServer pid=1"), 30,
                        "visible cell contains too many copies of error "
                        "(truncation didn't happen?)")

    def test_short_failure_unchanged(self) -> None:
        short_err = "oversize: model needs tp=8, eval pool=3"
        rows = [{
            "hf_id": "meta/llama-3.1-405B",
            "status": "aborted",
            "failure_reason": short_err,
        }]
        with mock.patch.object(server, "results_leaderboard",
                               return_value={
                                   "total": 1, "completed_ok": 0,
                                   "failed": 0, "in_progress": 0,
                                   "avg_pass_rate": None,
                                   "rows": rows,
                               }):
            html = server.render_results_page()
        # Short message must NOT be truncated.
        self.assertNotIn("…", html)
        # Must still be rendered.
        self.assertIn(short_err, html)


class TestEmptyResults(unittest.TestCase):
    """Render must not crash on zero rows."""

    def test_empty_rows(self) -> None:
        with mock.patch.object(server, "results_leaderboard",
                               return_value={
                                   "total": 0, "completed_ok": 0,
                                   "failed": 0, "in_progress": 0,
                                   "avg_pass_rate": None,
                                   "rows": [],
                               }):
            html = server.render_results_page()
        self.assertIn("无评测结果", html)


if __name__ == "__main__":
    unittest.main()
