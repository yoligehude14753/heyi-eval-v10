"""Tests for ``discover.radar_ingest`` — agents-radar manifest ingest.

Strategy: every HTTP call is mocked via the ``http_get`` injection
seam. We pin a known-good manifest + digest pair under
``tests/data/radar_*`` (small inline fixtures rather than committed
files since the upstream changes daily and we want stable tests).
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from discover.radar_ingest import (  # noqa: E402
    _PROJECT_REPORT_KINDS,
    ProjectCandidate,
    fetch_manifest,
    ingest_today,
    latest_date_entry,
    manifest_staleness_days,
    parse_digest_markdown,
)

# ── fixtures ──────────────────────────────────────────────────────────────


def _sample_manifest_bytes(latest_date: str = "2026-05-26") -> bytes:
    """Synthetic manifest with three dates; the latest one has all
    project-kind reports available."""
    return json.dumps({
        "generated": "2026-05-26T00:38:06.556Z",
        "dates": [
            {
                "date": latest_date,
                "reports": [
                    "ai-cli", "ai-agents", "ai-trending", "ai-web",
                    "ai-hn", "ai-ph", "ai-hf", "ai-arxiv", "ai-community",
                ],
            },
            {
                "date": "2026-05-25",
                "reports": ["ai-cli", "ai-agents", "ai-trending"],
            },
            {
                "date": "2026-05-24",
                "reports": ["ai-cli"],
            },
        ],
    }).encode("utf-8")


_AI_TRENDING_FIXTURE = """\
# AI 开源趋势日报 2026-05-26

## 今日 GitHub Trending 项目

| 项目 | Stars | 今日新增 | 说明 |
|:---|:---|:---|:---|
| [Understand-Anything](https://github.com/Lum1104/Understand-Anything) | — | +5,604 | 将任意代码转为可交互知识图谱 |
| [codegraph](https://github.com/colbymchenry/codegraph) | — | +3,161 | 预索引代码知识图谱，主打 100% 本地 |
| [ECC](https://github.com/affaan-m/ECC) | 192,314 | +2,025 | Agent Harness 性能优化系统 |
| [awesome-agents](https://github.com/foo/awesome-agents) | 50,000 | +1,500 | 应被 blocklist 过滤的链接合集 |
| [learn-claude-code](https://github.com/bar/learn-claude-code) | 60,000 | +900 | 应被 blocklist 过滤的教程 |
| [ollama](https://github.com/ollama/ollama) | 172,304 | — | 无 today's delta 的稳态项目 |

## 名人推荐

- [andrej-karpathy-skills](https://github.com/multica-ai/andrej-karpathy-skills) (+2,749) — 名人效应+实用价值
"""


_AI_CLI_FIXTURE = """\
# AI CLI 工具日报

- [Claude Code](https://github.com/anthropics/claude-code)
- [OpenAI Codex](https://github.com/openai/codex)
- [Gemini CLI](https://github.com/google-gemini/gemini-cli)
"""


_AI_AGENTS_FIXTURE = """\
# AI Agents 日报

| 项目 | Stars | 今日新增 | 说明 |
|:---|:---|:---|:---|
| [browser-use](https://github.com/browser-use/browser-use) | 95,508 | +500 | 浏览器自动化 |
| [Understand-Anything](https://github.com/Lum1104/Understand-Anything) | — | +5,604 | (overlap with ai-trending — should dedup) |
"""


# ── parse_digest_markdown tests ───────────────────────────────────────────


class ParseTests(unittest.TestCase):

    def test_extracts_table_rows(self) -> None:
        out = parse_digest_markdown(
            _AI_TRENDING_FIXTURE,
            source_report="ai-trending",
            source_date="2026-05-26",
        )
        ids = [c.full_id for c in out]
        # 6 visible rows + 1 bullet, minus 2 blocklist (awesome-/learn-)
        self.assertIn("Lum1104/Understand-Anything", ids)
        self.assertIn("colbymchenry/codegraph", ids)
        self.assertIn("affaan-m/ECC", ids)
        self.assertIn("ollama/ollama", ids)
        self.assertIn("multica-ai/andrej-karpathy-skills", ids)
        # Blocklisted
        self.assertNotIn("foo/awesome-agents", ids)
        self.assertNotIn("bar/learn-claude-code", ids)

    def test_extracts_stars_delta(self) -> None:
        out = parse_digest_markdown(
            _AI_TRENDING_FIXTURE,
            source_report="ai-trending",
            source_date="2026-05-26",
        )
        by_id = {c.full_id: c for c in out}
        self.assertEqual(by_id["Lum1104/Understand-Anything"].stars_delta, 5604)
        self.assertEqual(by_id["colbymchenry/codegraph"].stars_delta, 3161)
        # "ollama — —" has no +N delta
        self.assertIsNone(by_id["ollama/ollama"].stars_delta)

    def test_extracts_stars_total_when_present(self) -> None:
        out = parse_digest_markdown(
            _AI_TRENDING_FIXTURE,
            source_report="ai-trending",
            source_date="2026-05-26",
        )
        by_id = {c.full_id: c for c in out}
        self.assertEqual(by_id["affaan-m/ECC"].stars_total, 192314)
        self.assertEqual(by_id["ollama/ollama"].stars_total, 172304)
        self.assertIsNone(by_id["Lum1104/Understand-Anything"].stars_total)

    def test_extracts_short_desc(self) -> None:
        out = parse_digest_markdown(
            _AI_TRENDING_FIXTURE,
            source_report="ai-trending",
            source_date="2026-05-26",
        )
        by_id = {c.full_id: c for c in out}
        self.assertIn("知识图谱", by_id["Lum1104/Understand-Anything"].short_desc)

    def test_handles_bullet_list_format(self) -> None:
        out = parse_digest_markdown(
            _AI_CLI_FIXTURE,
            source_report="ai-cli",
            source_date="2026-05-26",
        )
        ids = {c.full_id for c in out}
        self.assertEqual(ids, {
            "anthropics/claude-code",
            "openai/codex",
            "google-gemini/gemini-cli",
        })

    def test_page_local_dedup(self) -> None:
        """If the same repo appears twice in one report, we only emit
        it once. Cross-report dedup happens at a higher layer."""
        md = (
            "[X](https://github.com/owner/repo)\n"
            "...intervening prose...\n"
            "[Y](https://github.com/owner/repo)\n"
        )
        out = parse_digest_markdown(md, source_report="ai-trending", source_date="2026-05-26")
        self.assertEqual(len(out), 1)


# ── manifest helpers ──────────────────────────────────────────────────────


class ManifestTests(unittest.TestCase):

    def _http_get_returning(self, payload: bytes):
        def fn(url: str) -> bytes:
            return payload
        return fn

    def test_fetch_manifest_parses_dates(self) -> None:
        m = fetch_manifest(http_get=self._http_get_returning(_sample_manifest_bytes()))
        self.assertEqual(len(m["dates"]), 3)

    def test_fetch_manifest_rejects_missing_dates(self) -> None:
        with self.assertRaisesRegex(ValueError, "'dates' array"):
            fetch_manifest(http_get=self._http_get_returning(b'{"foo": 1}'))

    def test_latest_date_picks_newest(self) -> None:
        m = fetch_manifest(http_get=self._http_get_returning(_sample_manifest_bytes()))
        self.assertEqual(latest_date_entry(m)["date"], "2026-05-26")

    def test_staleness_days(self) -> None:
        # Construct a manifest whose latest date is N days behind today
        # (relative-to-today so the test isn't time-bombed in 2027).
        from datetime import UTC as _UTC
        from datetime import datetime as _dt
        from datetime import timedelta
        target = (_dt.now(_UTC) - timedelta(days=4)).strftime("%Y-%m-%d")
        m = {
            "generated": "x",
            "dates": [{"date": target, "reports": []}],
        }
        self.assertEqual(manifest_staleness_days(m), 4)


# ── ingest_today end-to-end ───────────────────────────────────────────────


class IngestTodayTests(unittest.TestCase):

    def _make_fake_http(self) -> tuple:
        """Returns (http_get, urls_called) tuple. The fake routes URLs
        to the inline fixtures defined at module top."""
        urls_called: list[str] = []

        def fake_get(url: str) -> bytes:
            urls_called.append(url)
            if url.endswith("/manifest.json"):
                return _sample_manifest_bytes()
            if url.endswith("/ai-trending.md"):
                return _AI_TRENDING_FIXTURE.encode("utf-8")
            if url.endswith("/ai-cli.md"):
                return _AI_CLI_FIXTURE.encode("utf-8")
            if url.endswith("/ai-agents.md"):
                return _AI_AGENTS_FIXTURE.encode("utf-8")
            if url.endswith("/ai-web.md"):
                # Empty digest — should be tolerated, contributes 0 candidates.
                return b"# AI Web\n\nno entries today\n"
            raise FileNotFoundError(url)

        return fake_get, urls_called

    def test_h_happy_path_writes_dedup_jsonl(self) -> None:
        fake_get, urls = self._make_fake_http()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "project_candidates.jsonl"
            stats = ingest_today(out_path=out, http_get=fake_get)

            # All four project-kind reports should have been fetched
            kinds_fetched = [u for u in urls if any(k in u for k in _PROJECT_REPORT_KINDS)]
            self.assertEqual(len(kinds_fetched), 4)
            self.assertEqual(stats.reports_seen, 4)
            self.assertEqual(stats.network_errors, 0)
            self.assertGreater(stats.candidates_new, 0)

            # Cross-report dedup: ``Understand-Anything`` appears in both
            # ai-trending and ai-agents — should only land once.
            lines = out.read_text().splitlines()
            ids = [json.loads(line)["full_id"] for line in lines]
            self.assertEqual(ids.count("Lum1104/Understand-Anything"), 1)

    def test_idempotent_second_run_appends_nothing(self) -> None:
        fake_get, _ = self._make_fake_http()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "project_candidates.jsonl"
            ingest_today(out_path=out, http_get=fake_get)
            first_len = len(out.read_text().splitlines())

            stats2 = ingest_today(out_path=out, http_get=fake_get)
            second_len = len(out.read_text().splitlines())

            self.assertEqual(first_len, second_len)
            self.assertEqual(stats2.candidates_new, 0)
            self.assertGreater(stats2.candidates_dedup_skipped, 0)

    def test_per_kind_cap_applied(self) -> None:
        """max_per_kind=1 → ai-trending contributes only the top hot
        repo (Understand-Anything, +5604)."""
        fake_get, _ = self._make_fake_http()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "project_candidates.jsonl"
            ingest_today(out_path=out, http_get=fake_get, max_per_kind=1)
            lines = out.read_text().splitlines()
            # 4 kinds × 1 each (minus ai-web which is empty) → at most 3
            self.assertLessEqual(len(lines), 4)
            ids = [json.loads(line)["full_id"] for line in lines]
            self.assertIn("Lum1104/Understand-Anything", ids)

    def test_network_error_returns_counted_not_raised(self) -> None:
        def bad_get(url: str) -> bytes:
            import urllib.error
            raise urllib.error.URLError("DNS fail")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "project_candidates.jsonl"
            stats = ingest_today(out_path=out, http_get=bad_get)
            self.assertEqual(stats.network_errors, 1)
            self.assertEqual(stats.candidates_new, 0)

    def test_partial_failure_continues(self) -> None:
        """If one digest 404s, the others still complete."""
        def flaky_get(url: str) -> bytes:
            if url.endswith("/manifest.json"):
                return _sample_manifest_bytes()
            if url.endswith("/ai-trending.md"):
                import urllib.error
                raise urllib.error.URLError("digest gone")
            if url.endswith("/ai-cli.md"):
                return _AI_CLI_FIXTURE.encode("utf-8")
            return b""  # empty body for other kinds
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "project_candidates.jsonl"
            stats = ingest_today(out_path=out, http_get=flaky_get)
            self.assertGreater(stats.network_errors, 0)
            self.assertGreater(stats.candidates_new, 0)  # cli still ingested

    def test_blocklist_respected(self) -> None:
        fake_get, _ = self._make_fake_http()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "project_candidates.jsonl"
            ingest_today(out_path=out, http_get=fake_get)
            ids = [json.loads(line)["full_id"] for line in out.read_text().splitlines()]
            self.assertFalse(any("awesome-" in i for i in ids))
            self.assertFalse(any("learn-" in i for i in ids))

    def test_corrupt_existing_file_does_not_abort(self) -> None:
        """An interrupted append could leave a partial last line; we
        skip-and-continue rather than refuse to write today's run."""
        fake_get, _ = self._make_fake_http()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "project_candidates.jsonl"
            out.write_text(
                '{"full_id": "good/one", "source_url": "x", "source_report": "x", '
                '"source_date": "2026-01-01", "discovered_at": "x"}\n'
                '{partial-corruption{\n'
            )
            stats = ingest_today(out_path=out, http_get=fake_get)
            self.assertGreater(stats.candidates_new, 0)
            # The corrupt line gets carried forward (we only append),
            # but new appends are valid:
            valid_lines = 0
            for line in out.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    valid_lines += 1
                except json.JSONDecodeError:
                    pass
            self.assertGreater(valid_lines, 1)


class CandidateRoundtripTests(unittest.TestCase):

    def test_to_from_jsonl_roundtrip(self) -> None:
        c = ProjectCandidate(
            full_id="x/y", source_url="https://github.com/x/y",
            source_report="ai-trending", source_date="2026-05-26",
            discovered_at="2026-05-26T10:00:00+00:00",
            stars_delta=100, stars_total=5000, short_desc="d", reason="ai-trending",
        )
        line = c.to_jsonl()
        c2 = ProjectCandidate.from_jsonl(line)
        self.assertEqual(c, c2)


if __name__ == "__main__":
    unittest.main()
