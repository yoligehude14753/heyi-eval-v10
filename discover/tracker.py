"""HF discovery core.

Pure functions where possible — the HfApi is injected so we can unit-test
against a fake.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# We import lazily inside callers so unit tests don't require huggingface_hub.


# ── data model ─────────────────────────────────────────────────────────────


@dataclass
class Candidate:
    """A single model id we think is worth feeding into the eval pipeline.

    Source-of-truth is the JSON-line representation in candidates.jsonl;
    this dataclass is the in-memory mirror.
    """
    hf_id: str
    discovered_at: str                # ISO8601 UTC
    reason: str                       # "whitelist" | "trending" | "both"
    source_org: str | None = None     # parsed prefix
    last_modified: str | None = None
    downloads: int | None = None
    likes: int | None = None
    pipeline_tag: str | None = None
    library_name: str | None = None
    private: bool = False
    gated: bool = False

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_jsonl(cls, line: str) -> Candidate:
        return cls(**json.loads(line))


@dataclass
class TrackerConfig:
    whitelist_orgs: list[str] = field(default_factory=list)
    min_downloads_30d: int = 100_000
    min_likes: int = 100
    from_date: str = "2026-01-01T00:00:00Z"     # ignore models last_modified before this
    modality_pipeline_tags: list[str] = field(default_factory=lambda: [
        "text-generation",
        "image-text-to-text",
        "text-to-image",
        "text-to-video",
        "image-to-text",
        "automatic-speech-recognition",
        "text-to-speech",
        "any-to-any",
    ])
    per_org_limit: int = 50
    trending_sweep_limit: int = 200

    @classmethod
    def from_yaml(cls, path: Path) -> TrackerConfig:
        # We keep yaml-loading lightweight — only the keys we expect.
        # If pyyaml isn't installed, fall back to a tiny parser.
        try:
            import yaml  # type: ignore[import-not-found]
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except ImportError:
            data = _parse_simple_yaml(path.read_text(encoding="utf-8"))

        # Accept either top-level or nested under `tracker:`
        if "tracker" in data:
            data = data["tracker"]
        return cls(
            whitelist_orgs=list(data.get("whitelist_orgs", []) or []),
            min_downloads_30d=int(data.get("min_downloads_30d", 100_000)),
            min_likes=int(data.get("min_likes", 100)),
            from_date=str(data.get("from_date", "2026-01-01T00:00:00Z")),
            modality_pipeline_tags=list(data.get("modality_pipeline_tags") or [
                "text-generation", "image-text-to-text", "text-to-image",
                "text-to-video", "image-to-text",
                "automatic-speech-recognition", "text-to-speech", "any-to-any",
            ]),
            per_org_limit=int(data.get("per_org_limit", 50)),
            trending_sweep_limit=int(data.get("trending_sweep_limit", 200)),
        )


@dataclass
class Cursor:
    """Persistent state of one tracker. `seen` is the set of hf_ids we have
    already emitted as a candidate (so repeat scans don't dup them).

    PR#49: extends with backfill tracking. ``backfill_complete`` flips to
    True once the daemon has paginated every 2026 model whose
    ``last_modified >= config.from_date`` into the candidates file. After
    that, daily incremental scans (``scan_incremental``) only fetch the
    delta since ``last_run_ts``.
    """
    seen: set[str] = field(default_factory=set)
    last_run_ts: str = ""
    backfill_complete: bool = False
    # The oldest last_modified we've already paged through during backfill.
    # Used to resume an interrupted backfill — next page starts strictly
    # before this timestamp.
    backfill_high_water: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "seen": sorted(self.seen),
            "last_run_ts": self.last_run_ts,
            "backfill_complete": self.backfill_complete,
            "backfill_high_water": self.backfill_high_water,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Cursor:
        return cls(
            seen=set(d.get("seen") or []),
            last_run_ts=str(d.get("last_run_ts", "")),
            backfill_complete=bool(d.get("backfill_complete", False)),
            backfill_high_water=str(d.get("backfill_high_water", "")),
        )


@dataclass
class RoundStats:
    new_candidates: int = 0
    seen_skipped: int = 0
    excluded_old: int = 0
    excluded_modality: int = 0
    excluded_trending_threshold: int = 0
    api_errors: int = 0


# ── helpers ────────────────────────────────────────────────────────────────


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    """Tiny stdlib-only YAML subset parser.

    Supports:
      key: value
      key:
        - item1
        - item2
    Nested dicts not supported. Used only as fallback when pyyaml isn't
    installed; production paths should have pyyaml.
    """
    result: dict[str, Any] = {}
    cur_key: str | None = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("  - "):
            if cur_key is None:
                continue
            result.setdefault(cur_key, [])
            v = line[4:].strip().strip('"\'')
            result[cur_key].append(v)
        elif ":" in line and not line.startswith(" "):
            k, _, v = line.partition(":")
            k = k.strip()
            v = v.strip()
            if v == "":
                cur_key = k
                result.setdefault(k, [])
            else:
                cur_key = None
                v = v.strip('"\'')
                if v.isdigit():
                    result[k] = int(v)
                elif v.lower() in ("true", "false"):
                    result[k] = v.lower() == "true"
                else:
                    result[k] = v
    return result


def _norm_ts(ts: Any) -> str | None:
    """Normalize HF's timestamp variants → ISO8601 UTC string."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        return ts.astimezone(UTC).isoformat(timespec="seconds")
    s = str(ts).strip()
    if not s:
        return None
    # HF sometimes returns "2026-04-12T03:21:00.000Z" — normalize
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s).astimezone(UTC).isoformat(timespec="seconds")
    except ValueError:
        return s  # fall back to raw


def _passes_window(model_ts: str | None, from_date: str) -> bool:
    """Did this model see action since `from_date`? If we have no timestamp,
    we err on the side of including it (HF lets a lot of models through with
    last_modified=None when we don't request `expand=lastModified`)."""
    if not model_ts:
        return True
    try:
        a = datetime.fromisoformat(model_ts.replace("Z", "+00:00"))
        b = datetime.fromisoformat(from_date.replace("Z", "+00:00"))
        return a >= b
    except ValueError:
        return True


def _passes_modality(pipeline_tag: str | None, allowed: list[str]) -> bool:
    if not allowed:
        return True
    if not pipeline_tag:
        return True  # unknown — let it through, ENGINE_SELECT can bail later
    return pipeline_tag in allowed


def _model_to_candidate(m: Any, reason: str) -> Candidate:
    """Convert HfApi ModelInfo (or dict from fake / raw mirror JSON) to
    our Candidate.

    PR#52: the mirror's /api/models JSON uses camelCase (``lastModified``,
    ``modelId``) while huggingface_hub's ModelInfo exposes snake_case
    attributes (``last_modified``). Read both so the same converter
    works for either input shape.
    """
    def _read(key_snake: str, key_camel: str | None = None) -> Any:
        v = getattr(m, key_snake, None)
        if v is not None:
            return v
        if isinstance(m, dict):
            return m.get(key_snake) or (m.get(key_camel) if key_camel else None)
        if key_camel:
            return getattr(m, key_camel, None)
        return None

    hf_id = (
        getattr(m, "id", None)
        or (m.get("id") if isinstance(m, dict) else None)
        or (m.get("modelId") if isinstance(m, dict) else None)
        or ""
    )
    pipeline_tag = _read("pipeline_tag", "pipelineTag")
    last_modified = _read("last_modified", "lastModified")
    library_name = _read("library_name", "libraryName")
    downloads = _read("downloads")
    likes = _read("likes")
    private = _read("private")
    gated = _read("gated")
    return Candidate(
        hf_id=hf_id,
        discovered_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        reason=reason,
        source_org=hf_id.split("/", 1)[0] if "/" in hf_id else None,
        last_modified=_norm_ts(last_modified),
        downloads=int(downloads or 0) or None,
        likes=int(likes or 0) or None,
        pipeline_tag=pipeline_tag,
        library_name=library_name,
        private=bool(private or False),
        gated=bool(gated or False),
    )


# ── core ───────────────────────────────────────────────────────────────────


_EXPAND_FIELDS = [
    "lastModified",
    "createdAt",
    "downloads",
    "likes",
    "pipeline_tag",
    "library_name",
    "private",
    "gated",
]


# ── PR#52: cursor-aware mirror paginator ──────────────────────────────────


def _parse_next_link(link_header: str) -> str | None:
    """Extract ``<URL>; rel="next"`` from a Link header."""
    if not link_header:
        return None
    m = re.search(r'<([^>]+)>;\s*rel=["\']?next["\']?', link_header)
    return m.group(1) if m else None


def _rewrite_to_mirror(url: str, mirror_endpoint: str) -> str:
    """HF mirror serves page 1 but its Link header points at huggingface.co
    for ``next``. NV8 / many CN networks can't reach huggingface.co, so
    we have to swap the host back to the mirror to keep paginating.

    Idempotent: if ``url`` already targets the mirror, returns unchanged.
    """
    mirror = mirror_endpoint.rstrip("/")
    # Replace huggingface.co (any scheme) with the mirror's scheme+host.
    return re.sub(r"https?://huggingface\.co", mirror, url)


def mirror_paginate_models(
    *,
    endpoint: str,
    token: str | None = None,
    sort: str = "lastModified",
    direction: int = -1,
    limit_per_page: int = 1000,
    expand: list[str] | None = None,
    stop_when_older_than: str | None = None,
    max_pages: int = 2000,
    page_sleep_s: float = 0.0,
    max_429_retries: int = 8,
    retry_after_default_s: float = 60.0,
    log: Any = print,
    opener: Any = None,
) -> Iterator[dict]:
    """Generator that yields raw model JSON dicts from the HF mirror's
    ``/api/models`` endpoint, transparently following cursor-based
    pagination via the Link header (rewriting next-URLs back to the
    mirror so we don't get redirected to the unreachable huggingface.co).

    Parameters
    ----------
    endpoint : str
        e.g. ``https://hf-mirror.com``.
    token : str | None
        Sent as ``Authorization: Bearer …`` if provided.
    sort, direction, limit_per_page, expand
        Forwarded as query params to ``/api/models``.
    stop_when_older_than : ISO-8601 string | None
        Short-circuit: as soon as we yield a model whose
        ``lastModified < stop_when_older_than``, raise StopIteration
        on the next loop iteration. Skips wasted pages for backfill.
    max_pages : int
        Safety cap so a runaway cursor doesn't page forever (default 2000
        pages × 1000 items = 2M models, well past the 2026 cohort).
    page_sleep_s : float
        Politeness delay between pages.
    opener : callable | None
        Pluggable opener for unit tests. Defaults to
        ``urllib.request.urlopen``. Must accept (Request, *, timeout) and
        return an object with ``.read()`` + ``.headers.get("Link", "")``.
    """
    if expand is None:
        expand = _EXPAND_FIELDS
    params: list[tuple[str, str]] = [
        ("sort", sort),
        ("direction", str(direction)),
        ("limit", str(limit_per_page)),
    ]
    for f in expand:
        params.append(("expand", f))
    url = f"{endpoint.rstrip('/')}/api/models?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": "heyi-eval-v10/discover (mirror_paginate)"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Normalise the cutoff once so we don't pay the parse cost per row.
    # The mirror returns "...000Z" with millis but config.from_date is
    # "...Z" with none; raw string compare ranks `.` (46) < `Z` (90)
    # and a model exactly at the boundary is wrongly judged "older".
    cutoff_norm = _norm_ts(stop_when_older_than) if stop_when_older_than else None

    _opener = opener or urllib.request.urlopen

    def _fetch_with_429_retry(req_url: str) -> tuple[bytes, str] | None:
        """Returns (body, link_hdr) or None on hard failure. The HF
        mirror throttles at ~150 consecutive page fetches with HTTP
        429; honour Retry-After and try again so we can drain the
        whole 2026 cohort in one round."""
        attempt = 0
        while True:
            try:
                req = urllib.request.Request(req_url, headers=headers)
                with _opener(req, timeout=60) as r:
                    body = r.read()
                    if hasattr(r, "headers"):
                        link = r.headers.get("Link", "") or ""
                    else:
                        link = r.getheader("Link", "") or ""
                return body, link
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < max_429_retries:
                    retry_after = e.headers.get("Retry-After") if hasattr(e, "headers") else None
                    try:
                        wait_s = float(retry_after) if retry_after else retry_after_default_s
                    except (TypeError, ValueError):
                        wait_s = retry_after_default_s
                    attempt += 1
                    log(f"[mirror_paginate] HTTP 429 throttle "
                        f"(attempt {attempt}/{max_429_retries}); "
                        f"sleeping {wait_s:.0f}s")
                    time.sleep(wait_s)
                    continue
                log(f"[mirror_paginate] HTTPError {e.code} on "
                    f"{req_url[:120]}: {e}")
                return None
            except Exception as e:
                log(f"[mirror_paginate] request failed on "
                    f"{req_url[:120]}: {type(e).__name__}: {e}")
                return None

    for page_idx in range(1, max_pages + 1):
        result = _fetch_with_429_retry(url)
        if result is None:
            return
        body, link_hdr = result
        try:
            models = json.loads(body)
        except json.JSONDecodeError as e:
            log(f"[mirror_paginate] page {page_idx} bad JSON: {e}; "
                f"body[:200]={body[:200]!r}")
            return
        if not isinstance(models, list):
            log(f"[mirror_paginate] page {page_idx}: expected list, got "
                f"{type(models).__name__}; body[:200]={body[:200]!r}")
            return
        if not models:
            log(f"[mirror_paginate] page {page_idx}: empty body, end of "
                f"results (link={link_hdr[:80]!r})")
            return

        # Per-page heartbeat: oldest lastModified in the page is the
        # interesting datum (tells us where we are in time).
        oldest_in_page = min(
            (m.get("lastModified") or m.get("last_modified") or "ZZZ")
            for m in models
        )
        if page_idx == 1 or page_idx % 50 == 0:
            log(f"[mirror_paginate] page {page_idx}: n={len(models)} "
                f"oldest_in_page={oldest_in_page}")

        for m in models:
            yield m
            if cutoff_norm:
                lm = m.get("lastModified") or m.get("last_modified")
                if lm:
                    lm_norm = _norm_ts(lm)
                    if lm_norm and lm_norm < cutoff_norm:
                        log(f"[mirror_paginate] hit cutoff "
                            f"{cutoff_norm} at page {page_idx}, stop")
                        return

        next_url = _parse_next_link(link_hdr)
        if not next_url:
            log(f"[mirror_paginate] page {page_idx}: no Link rel=next; "
                f"reached end of available pages "
                f"(oldest_in_page={oldest_in_page})")
            return
        url = _rewrite_to_mirror(next_url, endpoint)
        if page_sleep_s:
            time.sleep(page_sleep_s)
    log(f"[mirror_paginate] max_pages={max_pages} reached, stopping")


def scan_round(
    api: Any,
    config: TrackerConfig,
    cursor: Cursor,
    *,
    log: Any = print,
) -> tuple[list[Candidate], RoundStats]:
    """One pass: whitelist orgs + trending sweep. Returns new candidates.

    `api` must duck-type as huggingface_hub.HfApi.list_models(...).
    """
    stats = RoundStats()
    new: list[Candidate] = []

    # 1) whitelisted orgs — high signal
    for org in config.whitelist_orgs:
        try:
            models = api.list_models(
                author=org,
                limit=config.per_org_limit,
                expand=_EXPAND_FIELDS,
                sort="lastModified",
            )
        except Exception as e:
            log(f"[discover] org={org} list_models failed: {type(e).__name__}: {e}")
            stats.api_errors += 1
            continue

        for m in models:
            cand = _model_to_candidate(m, reason="whitelist")
            if cand.hf_id in cursor.seen:
                stats.seen_skipped += 1
                continue
            if not _passes_window(cand.last_modified, config.from_date):
                stats.excluded_old += 1
                continue
            if not _passes_modality(cand.pipeline_tag, config.modality_pipeline_tags):
                stats.excluded_modality += 1
                continue
            new.append(cand)
            cursor.seen.add(cand.hf_id)
            stats.new_candidates += 1

    # 2) trending sweep — wide net
    try:
        models = api.list_models(
            limit=config.trending_sweep_limit,
            sort="downloads",
            expand=_EXPAND_FIELDS,
        )
    except Exception as e:
        log(f"[discover] trending list_models failed: {type(e).__name__}: {e}")
        stats.api_errors += 1
        models = []

    for m in models:
        cand = _model_to_candidate(m, reason="trending")
        if cand.hf_id in cursor.seen:
            # If we discovered it via whitelist already, upgrade reason to both
            for n in new:
                if n.hf_id == cand.hf_id and n.reason == "whitelist":
                    n.reason = "both"
            stats.seen_skipped += 1
            continue
        if not _passes_window(cand.last_modified, config.from_date):
            stats.excluded_old += 1
            continue
        if not _passes_modality(cand.pipeline_tag, config.modality_pipeline_tags):
            stats.excluded_modality += 1
            continue
        downloads = cand.downloads or 0
        likes = cand.likes or 0
        if downloads < config.min_downloads_30d and likes < config.min_likes:
            stats.excluded_trending_threshold += 1
            continue
        new.append(cand)
        cursor.seen.add(cand.hf_id)
        stats.new_candidates += 1

    cursor.last_run_ts = datetime.now(tz=UTC).isoformat(timespec="seconds")
    return new, stats


# ── backfill (PR#49) ───────────────────────────────────────────────────────


def scan_backfill(
    paginator: Iterable[Any],
    config: TrackerConfig,
    cursor: Cursor,
    *,
    log: Any = print,
) -> tuple[list[Candidate], RoundStats, bool]:
    """One round of historical backfill: drain ``paginator`` (yielding
    models sorted by lastModified DESC) until we cross config.from_date.

    Returns (new_candidates, stats, finished).

    ``finished=True`` means we observed at least one model older than
    from_date — i.e. we paged through every 2026 model and have proof
    we hit the bottom. Caller should flip ``cursor.backfill_complete``.

    PR#52 fix: the previous PR#49 implementation tried to resume across
    rounds by tracking ``backfill_high_water`` and skipping the rolling
    "top 5000 newest" window. That didn't work because HF Hub returns
    the same window every call and the high_water marched forward as
    new models were uploaded. The new design pages cursor-style via the
    HF Link header (see ``mirror_paginate_models``), so a single round
    drains everything from "now" back to from_date in one shot.

    The trending threshold (downloads/likes) is NOT applied in backfill
    mode — we want EVERY 2026 model, not just popular ones, exactly as
    the user requested: "把26年历史的抓完后，就持续抓最新的就好，日频抓取".
    Modality and privacy filters still apply.
    """
    stats = RoundStats()
    new: list[Candidate] = []
    finished = False
    oldest_seen: str | None = None
    saw_any = False
    # PR#52: normalise the boundary once so we don't mis-compare due to
    # "Z" vs "+00:00" suffix mismatches (see mirror_paginate_models for
    # the gory details).
    from_date_norm = _norm_ts(config.from_date) or config.from_date

    try:
        for m in paginator:
            saw_any = True
            cand = _model_to_candidate(m, reason="backfill")
            if cand.last_modified and (
                oldest_seen is None or cand.last_modified < oldest_seen
            ):
                oldest_seen = cand.last_modified
            # Stop iterating once we cross the from_date boundary.
            if (cand.last_modified and
                    cand.last_modified < from_date_norm):
                finished = True
                stats.excluded_old += 1
                break
            if cand.hf_id in cursor.seen:
                stats.seen_skipped += 1
                continue
            if not _passes_window(cand.last_modified, config.from_date):
                stats.excluded_old += 1
                continue
            if not _passes_modality(cand.pipeline_tag,
                                    config.modality_pipeline_tags):
                stats.excluded_modality += 1
                continue
            if cand.private or cand.gated:
                # gated repos can be promoted by other paths (manual
                # enqueue) but backfill skips them to avoid the 403
                # GatedRepoError storm during model_stager downloads.
                stats.excluded_modality += 1
                continue
            new.append(cand)
            cursor.seen.add(cand.hf_id)
            stats.new_candidates += 1
    except Exception as e:
        log(f"[backfill] paginator failed mid-stream: "
            f"{type(e).__name__}: {e}")
        stats.api_errors += 1

    if not saw_any:
        log("[backfill] paginator yielded zero models — treating as "
            "finished so the daemon stops looping on an empty source")
        finished = True

    if oldest_seen:
        cursor.backfill_high_water = oldest_seen
    cursor.last_run_ts = datetime.now(tz=UTC).isoformat(timespec="seconds")
    if finished:
        cursor.backfill_complete = True
    return new, stats, finished


def scan_incremental(
    paginator: Iterable[Any],
    config: TrackerConfig,
    cursor: Cursor,
    *,
    log: Any = print,
) -> tuple[list[Candidate], RoundStats]:
    """Daily-incremental pass after backfill is complete.

    Drains ``paginator`` (newest-first) and stops as soon as it yields
    a model older than ``cursor.last_run_ts``. Same filters as
    backfill — no trending-downloads gate, so a new-but-unknown model
    is still captured.
    """
    stats = RoundStats()
    new: list[Candidate] = []
    cutoff_raw = cursor.last_run_ts or config.from_date
    cutoff = _norm_ts(cutoff_raw) or cutoff_raw

    try:
        for m in paginator:
            cand = _model_to_candidate(m, reason="incremental")
            if cand.last_modified and cand.last_modified < cutoff:
                break
            if cand.hf_id in cursor.seen:
                stats.seen_skipped += 1
                continue
            if not _passes_modality(cand.pipeline_tag,
                                    config.modality_pipeline_tags):
                stats.excluded_modality += 1
                continue
            if cand.private or cand.gated:
                stats.excluded_modality += 1
                continue
            new.append(cand)
            cursor.seen.add(cand.hf_id)
            stats.new_candidates += 1
    except Exception as e:
        log(f"[incremental] paginator failed mid-stream: "
            f"{type(e).__name__}: {e}")
        stats.api_errors += 1

    cursor.last_run_ts = datetime.now(tz=UTC).isoformat(timespec="seconds")
    return new, stats


# ── persistence ────────────────────────────────────────────────────────────


def append_candidates(path: Path, candidates: Iterable[Candidate]) -> int:
    """Append-only write. Returns number written. fsync to survive crash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("a", encoding="utf-8") as f:
        for c in candidates:
            f.write(c.to_jsonl() + "\n")
            n += 1
        f.flush()
        try:
            import os
            os.fsync(f.fileno())
        except (OSError, AttributeError):
            pass
    return n


def load_cursor(path: Path) -> Cursor:
    if not path.exists():
        return Cursor()
    try:
        return Cursor.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return Cursor()


def save_cursor(path: Path, cursor: Cursor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cursor.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)


def load_candidates(path: Path) -> list[Candidate]:
    """Load full candidates.jsonl — used by curator + status views."""
    if not path.exists():
        return []
    out: list[Candidate] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(Candidate.from_jsonl(line))
            except (json.JSONDecodeError, TypeError):
                continue
    return out
