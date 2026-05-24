"""HF discovery core.

Pure functions where possible — the HfApi is injected so we can unit-test
against a fake.
"""
from __future__ import annotations

import json
from collections.abc import Iterable
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
    """Convert HfApi ModelInfo (or dict from fake) to our Candidate."""
    hf_id = getattr(m, "id", None) or m.get("id")
    pipeline_tag = getattr(m, "pipeline_tag", None)
    if pipeline_tag is None and isinstance(m, dict):
        pipeline_tag = m.get("pipeline_tag")
    last_modified = (
        getattr(m, "last_modified", None)
        or getattr(m, "lastModified", None)
        or (m.get("last_modified") if isinstance(m, dict) else None)
    )
    library_name = getattr(m, "library_name", None)
    if library_name is None and isinstance(m, dict):
        library_name = m.get("library_name")
    return Candidate(
        hf_id=hf_id,
        discovered_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
        reason=reason,
        source_org=hf_id.split("/", 1)[0] if "/" in hf_id else None,
        last_modified=_norm_ts(last_modified),
        downloads=int(getattr(m, "downloads", 0) or 0) or None,
        likes=int(getattr(m, "likes", 0) or 0) or None,
        pipeline_tag=pipeline_tag,
        library_name=library_name,
        private=bool(getattr(m, "private", False) or False),
        gated=bool(getattr(m, "gated", False) or False),
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
    api: Any,
    config: TrackerConfig,
    cursor: Cursor,
    *,
    max_per_round: int = 5000,
    log: Any = print,
) -> tuple[list[Candidate], RoundStats, bool]:
    """One round of historical backfill: page through HF Hub sorted by
    lastModified DESC, capturing every model whose last_modified
    falls in [from_date, backfill_high_water or now]. Returns
    (new_candidates, stats, finished).

    ``finished=True`` means we walked off the from_date boundary AND
    nothing newer than backfill_high_water remains to discover — the
    cursor's ``backfill_complete`` should be flipped to True and the
    daemon can switch to incremental mode.

    HF Hub doesn't expose a paging cursor; we ask for ``limit=max_per_round``
    sorted by lastModified, then advance ``backfill_high_water`` to the
    oldest last_modified we saw. Next round asks for the next page.

    The trending threshold (downloads/likes) is NOT applied in backfill
    mode — we want EVERY 2026 model, not just popular ones, exactly as
    the user requested: "把26年历史的抓完后，就持续抓最新的就好，日频抓取".
    Modality and privacy filters still apply.
    """
    stats = RoundStats()
    new: list[Candidate] = []
    finished = False

    # Anchor the page: if we have a previous high-water, ask for models
    # strictly older than that. Otherwise start from "now".
    page_end = cursor.backfill_high_water or ""
    try:
        # `sort="lastModified", direction=-1` gives newest-first on HfApi.
        models_iter = api.list_models(
            limit=max_per_round,
            sort="lastModified",
            direction=-1,
            expand=_EXPAND_FIELDS,
        )
    except TypeError:
        # Older fakes / mocks may not accept `direction=`.
        models_iter = api.list_models(
            limit=max_per_round,
            sort="lastModified",
            expand=_EXPAND_FIELDS,
        )
    except Exception as e:
        log(f"[backfill] list_models failed: {type(e).__name__}: {e}")
        stats.api_errors += 1
        return new, stats, False

    oldest_seen: str | None = None
    saw_any = False

    for m in models_iter:
        saw_any = True
        cand = _model_to_candidate(m, reason="backfill")
        # Track high-water even for filtered-out items.
        if cand.last_modified and (
            oldest_seen is None or cand.last_modified < oldest_seen
        ):
            oldest_seen = cand.last_modified

        # Stop iterating once we cross the from_date boundary.
        if (cand.last_modified and
                cand.last_modified < config.from_date):
            finished = True
            break

        # PR#49: when resuming a partial backfill, skip everything
        # newer-or-equal to the last high-water (already paged through).
        if page_end and cand.last_modified and cand.last_modified >= page_end:
            stats.seen_skipped += 1
            continue

        if cand.hf_id in cursor.seen:
            stats.seen_skipped += 1
            continue
        if not _passes_window(cand.last_modified, config.from_date):
            stats.excluded_old += 1
            continue
        if not _passes_modality(cand.pipeline_tag, config.modality_pipeline_tags):
            stats.excluded_modality += 1
            continue
        if cand.private or cand.gated:
            # gated repos can be promoted by other paths (manual enqueue)
            # but backfill skips them to avoid the 403 GatedRepoError storm.
            stats.excluded_modality += 1
            continue
        new.append(cand)
        cursor.seen.add(cand.hf_id)
        stats.new_candidates += 1

    if not saw_any:
        # API returned nothing — assume nothing to page; mark finished
        # so the daemon doesn't loop forever on an empty API.
        finished = True

    if oldest_seen:
        cursor.backfill_high_water = oldest_seen
    cursor.last_run_ts = datetime.now(tz=UTC).isoformat(timespec="seconds")
    if finished:
        cursor.backfill_complete = True
    return new, stats, finished


def scan_incremental(
    api: Any,
    config: TrackerConfig,
    cursor: Cursor,
    *,
    log: Any = print,
) -> tuple[list[Candidate], RoundStats]:
    """Daily-incremental pass after backfill is complete.

    Asks for the newest ``trending_sweep_limit`` models sorted by
    lastModified DESC and stops as soon as we hit a last_modified that's
    older than ``cursor.last_run_ts``. Apply the SAME filters as
    backfill (no trending-downloads gate) so we don't miss models that
    are new but not yet popular.
    """
    stats = RoundStats()
    new: list[Candidate] = []
    cutoff = cursor.last_run_ts or config.from_date

    try:
        models_iter = api.list_models(
            limit=config.trending_sweep_limit,
            sort="lastModified",
            direction=-1,
            expand=_EXPAND_FIELDS,
        )
    except TypeError:
        models_iter = api.list_models(
            limit=config.trending_sweep_limit,
            sort="lastModified",
            expand=_EXPAND_FIELDS,
        )
    except Exception as e:
        log(f"[incremental] list_models failed: {type(e).__name__}: {e}")
        stats.api_errors += 1
        return new, stats

    for m in models_iter:
        cand = _model_to_candidate(m, reason="incremental")
        # Hit the boundary — stop iterating.
        if cand.last_modified and cand.last_modified < cutoff:
            break
        if cand.hf_id in cursor.seen:
            stats.seen_skipped += 1
            continue
        if not _passes_modality(cand.pipeline_tag, config.modality_pipeline_tags):
            stats.excluded_modality += 1
            continue
        if cand.private or cand.gated:
            stats.excluded_modality += 1
            continue
        new.append(cand)
        cursor.seen.add(cand.hf_id)
        stats.new_candidates += 1

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
