"""agents-radar manifest.json ingest — project_lane discover (M2a).

Pulls https://raw.githubusercontent.com/duanyytop/agents-radar/master/manifest.json
and the daily digest markdown files, extracts GitHub repo URLs, and lands
the dedup'd list as ``data/discover/project_candidates.jsonl`` for the
project_lane queue to consume.

Why subscribe to manifest.json instead of forking agents-radar?

- agents-radar is a TypeScript project (pnpm, vitest, husky). To run it
  in v10 we'd need a Node.js runtime + a maintained fork — both are
  ongoing costs we don't want to take on. The upstream already pushes
  a structured ``manifest.json`` daily via GitHub Actions, so subscribing
  is the lower-cost path.
- The contract surface is the JSON manifest + raw markdown digests. Both
  are mirrored on raw.githubusercontent.com which is reachable from
  Mac dev boxes and from NV8 via the same mirror trick we use for HF.
- If agents-radar upstream goes silent (no commits for >30 days) the
  ``staleness_warning_days`` knob below trips a healthcheck so we know.

Contract:

- Daily run pulls manifest, picks the latest date's reports.
- Filters reports to project-shaped kinds (``ai-trending``, ``ai-agents``,
  ``ai-cli``, ``ai-web``). Excluded: ``ai-arxiv`` (papers, no repo to run),
  ``ai-hn`` / ``ai-ph`` (mixed news, low signal-to-noise for executable
  projects), ``ai-community`` (Dev.to posts), ``ai-hf`` (covered by the
  model_lane HF discover).
- For each report, extracts ``[name](https://github.com/owner/repo)``
  markdown links + nearby "stars / today's delta" context.
- De-dupes against prior project_candidates.jsonl (full_id = owner/repo).
- Output: append-only jsonl, one ``ProjectCandidate`` per line.

The CLI surface lives in ``discover/main.py`` as new sub-commands
(``radar-once`` / ``radar-loop``); this module is pure data extraction
so unit tests can mock the HTTP layer with ``http_get`` injection.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── public types ──────────────────────────────────────────────────────────


@dataclass
class ProjectCandidate:
    """One GitHub repo that survived radar ingest + light pre-filtering.

    ``full_id`` is ``owner/repo`` (matches GitHub's canonical id form).
    Both ``stars_delta`` and ``stars_total`` are optional because the
    upstream markdown is human-written and not all rows include both —
    ``stars_delta`` is the primary "hotness" signal so we sort by it
    when picking the day's top-N.

    ``reason`` is a free-form bucket ("ai-trending" / "ai-agents" /
    "ai-cli" / "ai-web") plus optional "manual" override. The orchestrator
    uses this to apportion the per-day budget across kinds (so the day's
    pick isn't 100% ``ai-trending``).
    """
    full_id: str
    source_url: str
    source_report: str
    source_date: str  # ISO date e.g. "2026-05-26"
    discovered_at: str  # ISO8601 UTC
    stars_delta: int | None = None
    stars_total: int | None = None
    short_desc: str = ""
    reason: str = "ai-trending"
    # Free-form provenance for audit — what raw markdown row produced
    # this candidate. Capped to 500 chars so a malformed row can't blow
    # up disk usage when ingested.
    raw_excerpt: str = ""

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> ProjectCandidate:
        return cls(**json.loads(line))


@dataclass
class IngestStats:
    manifest_date: str
    reports_seen: int = 0
    reports_skipped: int = 0
    rows_parsed: int = 0
    candidates_new: int = 0
    candidates_dedup_skipped: int = 0
    network_errors: int = 0
    parse_warnings: list[str] = field(default_factory=list)


# ── network IO seam ───────────────────────────────────────────────────────


HttpGet = Callable[[str], bytes]
"""Type alias for a function that returns raw response bytes given a URL.

Tests inject a fake here so we don't hit the real upstream. Production
uses ``_default_http_get`` below which is plain stdlib urllib + 10s
timeout (radar manifests are <50KB so timeouts shouldn't matter).
"""


def _default_http_get(url: str) -> bytes:  # pragma: no cover — production seam
    """Production HTTP fetch. Plain stdlib, 10s timeout, no retries —
    the call site in ``ingest_today`` wraps this in try/except and
    records a single ``network_errors`` increment, then continues with
    whatever reports it did get. Better to ship a half-day's candidates
    than nothing.
    """
    req = urllib.request.Request(
        url, headers={"User-Agent": "heyi-eval-v10/radar-ingest"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return bytes(resp.read())


# ── constants ─────────────────────────────────────────────────────────────


_MANIFEST_URL = (
    "https://raw.githubusercontent.com/duanyytop/agents-radar/master/manifest.json"
)
_DIGEST_BASE = (
    "https://raw.githubusercontent.com/duanyytop/agents-radar/master/digests"
)

# Reports we treat as "GitHub-shaped projects worth running":
# - ai-trending / ai-agents / ai-cli / ai-web all yield repo links with
#   actionable context (stars delta, short description, sometimes README
#   pointers). These match the user request: "github 开源项目清单".
# - Excluded kinds (rationale in module docstring):
#   ai-arxiv, ai-hn, ai-ph, ai-community, ai-hf, ai-weekly
_PROJECT_REPORT_KINDS = ("ai-trending", "ai-agents", "ai-cli", "ai-web")

# Markdown row patterns we recognise. Both ai-trending (markdown table)
# and ai-cli (markdown list) embed repo links the same way:
#     [name](https://github.com/owner/repo)
# Capture name + owner + repo in named groups.
_GH_LINK_RE = re.compile(
    r"\[(?P<name>[^\]]+?)\]"
    r"\("
    r"https?://github\.com/"
    r"(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repo>[A-Za-z0-9_.-]+?)"
    r"(?:/[^)]*)?"  # tolerate /tree/main or trailing slashes
    r"\)",
)

# "today's delta" column in ai-trending tables: looks like "+5,604".
# We use a permissive match because some rows have "—" or absent.
_STARS_DELTA_RE = re.compile(r"\+([\d,]+)")

# Some rows include both stars_total and stars_delta; we pull whichever
# we see. "192,314" → integer. Numbers in the description column also
# get captured by accident, so we anchor to row context in caller.
_STARS_TOTAL_RE = re.compile(r"\b([\d,]{3,})\b")

# Soft blocklist — repos that match these patterns are dropped at ingest.
# Mostly meta/aggregator repos that have no "deploy + run + observe"
# semantics for the agent. Concrete examples:
#   awesome-* — link lists, no runnable code
#   *-papers — paper-link collections
#   roadmap-* — text-only roadmaps
#   <single-letter>/* — bot accounts that publish placeholder repos
_BLOCKLIST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^awesome-", re.IGNORECASE),
    re.compile(r"-papers$", re.IGNORECASE),
    re.compile(r"^roadmap", re.IGNORECASE),
    re.compile(r"^learn-", re.IGNORECASE),  # learn-claude-code, learn-X, ...
    re.compile(r"^([a-z])/", re.IGNORECASE),
)


# ── parsing helpers ───────────────────────────────────────────────────────


def _is_blocked(owner: str, repo: str) -> bool:
    """Return True for repos we drop at ingest. See _BLOCKLIST_PATTERNS
    above for the rationale list."""
    full = f"{owner}/{repo}"
    return any(pat.search(repo) or pat.match(full) for pat in _BLOCKLIST_PATTERNS)


def _parse_stars_delta(line: str) -> int | None:
    """Pull the first ``+N,NNN`` token from a markdown row, or None."""
    m = _STARS_DELTA_RE.search(line)
    if not m:
        return None
    try:
        return int(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _parse_stars_total(line: str) -> int | None:
    """Pull a "stars total" hint, deliberately conservative: only match
    numbers ≥1000 (anything smaller is likely "+317" delta or row
    indices) AND NOT prefixed by '+'. If multiple match, take the first.
    """
    # Strip the +delta token first so we don't accidentally match its digits.
    cleaned = _STARS_DELTA_RE.sub("", line)
    m = _STARS_TOTAL_RE.search(cleaned)
    if not m:
        return None
    try:
        n = int(m.group(1).replace(",", ""))
    except ValueError:
        return None
    return n if n >= 1000 else None


def _extract_short_desc(line: str, owner: str, repo: str) -> str:
    """Pull the per-row description text. For markdown tables the
    description is the last ``|``-separated cell; for bullet lists
    there isn't one (return "").

    Capped to 240 chars at this layer; the panel column has its own
    further visual truncation.
    """
    if "|" not in line:
        return ""
    cells = [c.strip() for c in line.split("|") if c.strip()]
    if not cells:
        return ""
    # Last cell is the description column for ai-trending; for tables
    # without a description column this still returns some short text.
    last = cells[-1]
    # The repo link itself is usually in the first/second cell. If the
    # last cell still looks link-y, give up.
    if f"github.com/{owner}/{repo}" in last:
        return ""
    return last[:240]


def parse_digest_markdown(
    md_text: str,
    *,
    source_report: str,
    source_date: str,
) -> list[ProjectCandidate]:
    """Extract candidates from one digest .md file.

    Parsing is line-based: every line with at least one
    ``[name](https://github.com/owner/repo)`` link contributes one or
    more candidates. We deliberately don't try to "understand" the
    document structure (heading hierarchy, table boundaries) — the
    upstream format is human-written and brittle. Line-based extraction
    + light dedup is sufficient.
    """
    now_iso = datetime.now(UTC).isoformat()
    out: list[ProjectCandidate] = []
    seen_on_page: set[str] = set()

    for raw_line in md_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        for m in _GH_LINK_RE.finditer(line):
            owner = m.group("owner")
            repo = m.group("repo")
            full_id = f"{owner}/{repo}"
            # Page-local dedup so a multi-section digest doesn't insert
            # the same repo twice. Cross-page dedup happens in the caller.
            if full_id in seen_on_page:
                continue
            seen_on_page.add(full_id)
            if _is_blocked(owner, repo):
                continue
            out.append(ProjectCandidate(
                full_id=full_id,
                source_url=f"https://github.com/{owner}/{repo}",
                source_report=source_report,
                source_date=source_date,
                discovered_at=now_iso,
                stars_delta=_parse_stars_delta(line),
                stars_total=_parse_stars_total(line),
                short_desc=_extract_short_desc(line, owner, repo),
                reason=source_report,
                raw_excerpt=line[:500],
            ))
    return out


# ── manifest layer ────────────────────────────────────────────────────────


def fetch_manifest(*, http_get: HttpGet | None = None) -> dict[str, Any]:
    """Fetch + parse the upstream manifest.json. Raises on network or
    JSON failure — the daemon caller catches and continues.
    """
    fn = http_get or _default_http_get
    raw = fn(_MANIFEST_URL)
    obj: dict[str, Any] = json.loads(raw.decode("utf-8"))
    if "dates" not in obj or not isinstance(obj["dates"], list):
        raise ValueError(
            "manifest.json has no 'dates' array — schema drift; "
            "upstream contract change in agents-radar"
        )
    return obj


def latest_date_entry(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return the most-recent date entry. The upstream sorts descending,
    but we re-sort to be robust against accidental upstream ordering
    changes (cheap, list is <100 items)."""
    dates = sorted(
        manifest["dates"],
        key=lambda d: d.get("date", ""),
        reverse=True,
    )
    if not dates:
        raise ValueError("manifest.json has zero date entries")
    entry: dict[str, Any] = dates[0]
    return entry


def manifest_staleness_days(manifest: dict[str, Any]) -> int:
    """How old is the freshest date in the manifest, in whole days?

    Used by the daemon to alert: if staleness > 3d the upstream is
    likely broken (GHA disabled, the maintainer stopped), and we want
    to surface that to the operator rather than silently degrading.
    """
    latest = latest_date_entry(manifest)["date"]
    try:
        d = datetime.strptime(latest, "%Y-%m-%d").replace(tzinfo=UTC)
    except ValueError as e:
        raise ValueError(f"manifest date not YYYY-MM-DD: {latest!r}") from e
    return (datetime.now(UTC) - d).days


# ── ingest orchestration ──────────────────────────────────────────────────


def ingest_today(
    *,
    out_path: Path,
    http_get: HttpGet | None = None,
    report_kinds: tuple[str, ...] = _PROJECT_REPORT_KINDS,
    max_per_kind: int = 10,
    staleness_warning_days: int = 3,
) -> IngestStats:
    """End-to-end: fetch manifest → fetch each project-kind digest →
    parse → dedup against ``out_path`` → append new rows.

    Args:
      out_path: path to ``project_candidates.jsonl``; created with
          parents if missing. Read-modify-append is a single pass so a
          concurrent reader sees a consistent state at any line boundary.
      http_get: HTTP fetcher; default uses stdlib urllib.
      report_kinds: which digest kinds to ingest from. Defaults to
          ``_PROJECT_REPORT_KINDS`` (4 kinds).
      max_per_kind: cap candidates per kind per day. Default 10 is
          conservative — even at 4 kinds × 10 we're far under the
          ``5 projects/day`` agent_lane execution budget.
      staleness_warning_days: log WARNING when upstream is older
          than this; orchestrator's healthcheck reads the log.

    Returns IngestStats — caller decides what to do with errors /
    counts (typically just log them).
    """
    fn = http_get or _default_http_get
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats = IngestStats(manifest_date="")
    try:
        manifest = fetch_manifest(http_get=fn)
    except (urllib.error.URLError, json.JSONDecodeError, ValueError) as e:
        log.error("radar.ingest manifest fetch failed: %s", e)
        stats.network_errors += 1
        return stats

    latest = latest_date_entry(manifest)
    stats.manifest_date = latest["date"]
    stale = manifest_staleness_days(manifest)
    if stale > staleness_warning_days:
        log.warning(
            "radar.ingest upstream manifest is %dd old (>%dd warn threshold); "
            "agents-radar may be unmaintained or GHA broken",
            stale, staleness_warning_days,
        )

    seen_full_ids = _load_seen_ids(out_path)
    new_rows: list[ProjectCandidate] = []

    for kind in report_kinds:
        if kind not in latest.get("reports", []):
            stats.reports_skipped += 1
            continue
        stats.reports_seen += 1
        url = f"{_DIGEST_BASE}/{latest['date']}/{kind}.md"
        try:
            md = fn(url).decode("utf-8")
        except (urllib.error.URLError, UnicodeDecodeError) as e:
            log.warning("radar.ingest digest %s fetch failed: %s", url, e)
            stats.network_errors += 1
            continue
        parsed = parse_digest_markdown(
            md, source_report=kind, source_date=latest["date"],
        )
        stats.rows_parsed += len(parsed)
        # Sort within-kind by stars_delta desc (None last), then take top N.
        parsed.sort(key=lambda c: (c.stars_delta or 0), reverse=True)
        admitted = 0
        for cand in parsed:
            if admitted >= max_per_kind:
                break
            if cand.full_id in seen_full_ids:
                stats.candidates_dedup_skipped += 1
                continue
            seen_full_ids.add(cand.full_id)
            new_rows.append(cand)
            admitted += 1

    if new_rows:
        _append_candidates(out_path, new_rows)
    stats.candidates_new = len(new_rows)
    log.info(
        "radar.ingest manifest_date=%s reports_seen=%d skipped=%d "
        "parsed=%d new=%d dedup=%d net_err=%d",
        stats.manifest_date, stats.reports_seen, stats.reports_skipped,
        stats.rows_parsed, stats.candidates_new,
        stats.candidates_dedup_skipped, stats.network_errors,
    )
    return stats


# ── persistence ──────────────────────────────────────────────────────────


def _load_seen_ids(path: Path) -> set[str]:
    """Read existing ``project_candidates.jsonl`` and build the seen set.

    Each line we silently skip on JSON parse error is logged but doesn't
    abort the load — a single corrupt line at the tail of an interrupted
    write shouldn't prevent today's run.
    """
    if not path.exists():
        return set()
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                log.warning("radar.ingest %s:%d skipping unparseable line", path, lineno)
                continue
            full_id = obj.get("full_id")
            if isinstance(full_id, str):
                seen.add(full_id)
    return seen


def _append_candidates(path: Path, candidates: list[ProjectCandidate]) -> None:
    """Append new candidates as JSON lines. Each line is fsync-safe by
    virtue of being a single write() call; partial-line tail recovery
    is handled by ``_load_seen_ids`` skipping unparseable rows.
    """
    with path.open("a", encoding="utf-8") as f:
        for c in candidates:
            f.write(c.to_jsonl() + "\n")


# ── convenience: default out path mirrors model_lane convention ──────────


def default_project_candidates_path() -> Path:
    """``$HEYI_EVAL_DATA/discover/project_candidates.jsonl`` — same
    layout as model_lane's ``candidates.jsonl`` so panel can iterate
    both with a single pattern."""
    root = os.environ.get("HEYI_EVAL_DATA", "/home/ai/heyi-eval-data")
    return Path(root) / "discover" / "project_candidates.jsonl"
