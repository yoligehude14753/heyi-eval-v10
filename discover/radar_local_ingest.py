"""Local radar ingest — project_lane discovery from our own radar.

The original ``radar_ingest`` subscribes to a third-party
``duanyytop/agents-radar`` manifest. This module instead consumes the
**local** radar service (``~/Desktop/all/radar``, co-located on heyi)
via its ``GET /api/items`` endpoint, which serves AI hotspots that radar
discovered from GitHub Search + Reddit and scored with the QAG rubric.

Why prefer the local radar over the third-party manifest:

- Richer signal: every item carries a QAG composite score + per-dimension
  breakdown (pain / market / feasibility / velocity) and a domain tag, so
  enqueue can prioritise genuinely high-demand projects rather than just
  "trending today".
- One brain: radar now scores with the same Zhipu GLM-5.1 that heyi-eval
  evaluates with, so discovery and evaluation share a provider/key.
- No external dependency drift: the contract is our own REST endpoint.

The third-party ``radar_ingest`` stays available as a fallback
(``discover.main radar-once``); this module adds ``radar-local-once`` /
``radar-local-loop``.

Contract:

- Pull ``GET {base_url}/api/items?source=github&min_score=..&limit=..``.
- Map each GitHub item to a ``ProjectCandidate`` (full_id from the
  repo's ``external_id`` which radar stores as ``owner/repo``; falls
  back to parsing the URL). Non-GitHub items (Reddit discussions) are
  skipped — they have no repo to deploy + run.
- De-dupe against the existing ``project_candidates.jsonl`` (full_id).
- Append-only jsonl, carrying ``qag_score`` for downstream prioritisation.
"""
from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .radar_ingest import (
    ProjectCandidate,
    _append_candidates,
    _load_seen_ids,
    default_project_candidates_path,
)

log = logging.getLogger(__name__)


# ── config ────────────────────────────────────────────────────────────────


def default_radar_base_url() -> str:
    """Local radar service base URL. Override with ``RADAR_BASE_URL``.

    Default targets the co-located radar on its documented port 7090.
    """
    return os.environ.get("RADAR_BASE_URL", "http://127.0.0.1:7090").rstrip("/")


# ── network seam (injectable for tests) ─────────────────────────────────────


JsonGet = Callable[[str], Any]
"""Returns parsed JSON given a URL. Tests inject a fake; production uses
``_default_json_get`` (stdlib urllib + 15s timeout)."""


def _default_json_get(url: str) -> Any:  # pragma: no cover — production seam
    req = urllib.request.Request(
        url, headers={"User-Agent": "heyi-eval-v10/radar-local-ingest"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── stats ───────────────────────────────────────────────────────────────────


@dataclass
class LocalIngestStats:
    base_url: str
    items_seen: int = 0
    non_github_skipped: int = 0
    unparseable_skipped: int = 0
    candidates_new: int = 0
    candidates_dedup_skipped: int = 0
    network_errors: int = 0
    warnings: list[str] = field(default_factory=list)


# ── mapping ─────────────────────────────────────────────────────────────────


_GH_URL_RE = re.compile(
    r"github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)"
)


def _full_id_from_item(item: dict[str, Any]) -> str | None:
    """Derive ``owner/repo`` from a radar item.

    radar's GitHub crawler stores ``external_id`` as the repo full name
    (``owner/repo``), so we trust that first and fall back to parsing
    the URL for robustness.
    """
    ext = (item.get("external_id") or "").strip().strip("/")
    if ext.count("/") == 1 and all(ext.split("/")):
        return ext
    url = item.get("url") or ""
    m = _GH_URL_RE.search(url)
    if not m:
        return None
    repo = m.group("repo")
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]
    return f"{m.group('owner')}/{repo}"


def item_to_candidate(item: dict[str, Any], *, source_date: str) -> ProjectCandidate | None:
    """Map one radar ``/api/items`` row to a ProjectCandidate.

    Returns ``None`` for non-GitHub items or rows we can't resolve to a
    repo id.
    """
    if item.get("source") != "github":
        return None
    full_id = _full_id_from_item(item)
    if not full_id:
        return None
    platform = item.get("platform_data") or {}
    stars = platform.get("stars")
    domain = item.get("domain") or "radar"
    desc = (item.get("title") or item.get("content") or "")[:240]
    return ProjectCandidate(
        full_id=full_id,
        source_url=item.get("url") or f"https://github.com/{full_id}",
        source_report=f"radar-{domain}",
        source_date=source_date,
        discovered_at=datetime.now(UTC).isoformat(),
        stars_delta=None,
        stars_total=int(stars) if isinstance(stars, (int, float)) else None,
        short_desc=desc,
        reason=f"radar:{domain}",
        raw_excerpt=json.dumps(
            {k: item.get(k) for k in ("score", "dimensions", "domain")},
            ensure_ascii=False,
        )[:500],
        qag_score=(
            float(item["score"])
            if isinstance(item.get("score"), (int, float))
            else None
        ),
    )


# ── ingest orchestration ────────────────────────────────────────────────────


def ingest_from_radar(
    *,
    out_path: Path,
    base_url: str | None = None,
    source: str = "github",
    min_score: float = 0.6,
    limit: int = 50,
    json_get: JsonGet | None = None,
) -> LocalIngestStats:
    """Fetch scored hotspots from the local radar and append new project
    candidates to ``out_path``.

    Args:
      out_path: ``project_candidates.jsonl`` (shared with radar_ingest).
      base_url: radar service base (default ``default_radar_base_url()``).
      source: radar source filter (``github`` — only repos are runnable).
      min_score: QAG composite-score floor; radar applies it server-side.
      limit: max items to pull from radar this round.
      json_get: HTTP seam; default stdlib urllib.
    """
    fn = json_get or _default_json_get
    base = (base_url or default_radar_base_url()).rstrip("/")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = LocalIngestStats(base_url=base)

    query = urllib.parse.urlencode({
        "source": source,
        "min_score": min_score,
        "limit": limit,
        "scored_only": "true",
    })
    url = f"{base}/api/items?{query}"
    try:
        items = fn(url)
    except (urllib.error.URLError, json.JSONDecodeError, ValueError, OSError) as e:
        log.error("radar_local.ingest fetch failed: %s", e)
        stats.network_errors += 1
        return stats
    if not isinstance(items, list):
        stats.warnings.append("radar /api/items did not return a list")
        return stats

    source_date = datetime.now(UTC).date().isoformat()
    seen = _load_seen_ids(out_path)
    new_rows: list[ProjectCandidate] = []
    for item in items:
        stats.items_seen += 1
        if not isinstance(item, dict):
            stats.unparseable_skipped += 1
            continue
        cand = item_to_candidate(item, source_date=source_date)
        if cand is None:
            if item.get("source") != "github":
                stats.non_github_skipped += 1
            else:
                stats.unparseable_skipped += 1
            continue
        if cand.full_id in seen:
            stats.candidates_dedup_skipped += 1
            continue
        seen.add(cand.full_id)
        new_rows.append(cand)

    if new_rows:
        _append_candidates(out_path, new_rows)
    stats.candidates_new = len(new_rows)
    log.info(
        "radar_local.ingest base=%s seen=%d new=%d dedup=%d non_github=%d "
        "unparseable=%d net_err=%d",
        stats.base_url, stats.items_seen, stats.candidates_new,
        stats.candidates_dedup_skipped, stats.non_github_skipped,
        stats.unparseable_skipped, stats.network_errors,
    )
    return stats


__all__ = [
    "LocalIngestStats",
    "default_project_candidates_path",
    "default_radar_base_url",
    "ingest_from_radar",
    "item_to_candidate",
]
