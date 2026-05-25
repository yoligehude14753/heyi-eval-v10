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
    """PR#58 superseded the original PR#32 status-priority sort. The
    user's new contract is purely time-DESC ("测试结果时间排序，最新的
    放前面") regardless of status. This test pins that behavior."""

    def _rows(self):
        # Each row tagged with a distinct created_at; ordering by status
        # would scramble these but time-DESC keeps them strictly oldest→
        # newest reversed.
        return [
            {"hf_id": "t1-oldest/foo", "status": "failed",
             "failure_reason": "X" * 5000, "created_at": 1000},
            {"hf_id": "t2-old/foo", "status": "aborted",
             "failure_reason": "oversize", "created_at": 2000},
            {"hf_id": "t3-mid/foo", "status": "ok", "pass_rate": 0.48,
             "capability": "12/25", "created_at": 3000},
            {"hf_id": "t4-mid/foo", "status": "ok", "pass_rate": 0.71,
             "capability": "25/35", "created_at": 4000},
            {"hf_id": "t5-new/foo", "status": "ok", "created_at": 5000},
            {"hf_id": "t6-newer/foo", "status": "in_progress",
             "created_at": 6000},
            {"hf_id": "t7-newest/foo", "status": "ok", "pass_rate": 1.0,
             "capability": "5/5", "created_at": 7000},
        ]

    def test_sort_time_desc(self) -> None:
        """PR#58: newest created_at first, regardless of status."""
        rows = self._rows()
        with mock.patch.object(server, "results_leaderboard",
                               return_value={
                                   "total": len(rows),
                                   "completed_ok": 3,
                                   "failed": 1,
                                   "in_progress": 1,
                                   "avg_pass_rate": 0.6,
                                   "rows": rows,
                               }):
            html_str = server.render_results_page()

        def pos(needle: str) -> int:
            i = html_str.find(needle)
            self.assertGreaterEqual(i, 0, f"{needle!r} not in page")
            return i

        order = [
            "t7-newest/foo", "t6-newer/foo", "t5-new/foo",
            "t4-mid/foo", "t3-mid/foo",
            "t2-old/foo", "t1-oldest/foo",
        ]
        positions = [pos(name) for name in order]
        for prev, curr, name in zip(positions, positions[1:], order[1:]):
            self.assertLess(prev, curr,
                            f"{name!r} should come AFTER older row")


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
        # Short message must NOT be truncated. Check only the row body
        # (the toolkit's placeholder text "搜索 ... …" lives in JS and
        # is not part of the data).
        tbody_start = html.find("<tbody>")
        tbody_end = html.find("</tbody>")
        body = html[tbody_start:tbody_end] if tbody_start >= 0 else html
        self.assertNotIn("…", body)
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
