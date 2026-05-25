"""PR#65 — eval-cache LRU eviction tests.

Covers the four classification states (safe / failed_only / in_progress
/ orphan), the eviction priority order, the grace window guard, and
the dry-run vs apply semantics. All tests use a synthetic on-disk
layout in tmp_path so no real ML weights are touched."""

from __future__ import annotations

import json
import time
from pathlib import Path

from orchestrator.cache_evictor import (
    DEFAULT_GRACE_S,
    enforce_quota,
    find_evictable,
)


def _mk_cache_dir(root: Path, name: str, size_mb: int,
                  mtime_offset_s: float = 0) -> Path:
    """Create eval-cache/<name>/dummy.bin of ``size_mb`` MB and set
    its mtime to now + mtime_offset_s. Returns the dir."""
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    f = d / "dummy.bin"
    f.write_bytes(b"\0" * (size_mb * 1024 * 1024))
    t = time.time() + mtime_offset_s
    import os
    os.utime(d, (t, t))
    os.utime(f, (t, t))
    return d


def _mk_run(runs_root: Path, run_id: str, hf_id: str,
            status: str, ended_at: float | None = None) -> Path:
    """Create a runs/<run_id>/state.json that the evictor will pick
    up. Mirrors the live schema from nv8 — see check above of
    /home/ai/heyi-eval-data/runs/.../state.json."""
    rd = runs_root / run_id
    rd.mkdir(parents=True, exist_ok=True)
    state = {
        "run_id": run_id, "hf_id": hf_id,
        "status": status,
        "created_at": ended_at or time.time() - 3600,
        "ended_at": ended_at,
        "stages": {},
        "failure_reason": None, "abort_flag": False,
    }
    (rd / "state.json").write_text(json.dumps(state), encoding="utf-8")
    return rd


# ── classification ─────────────────────────────────────────────────────


def test_evictor_classifies_safe_when_run_status_ok(tmp_path):
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    cache.mkdir()
    _mk_cache_dir(cache, "Qwen2.5-7B", size_mb=5,
                  mtime_offset_s=-3 * 3600)  # 3h old, outside grace
    _mk_run(runs, "r-001", "Qwen/Qwen2.5-7B", status="ok",
            ended_at=time.time() - 3600)
    plan = find_evictable(cache, runs, quota_bytes=1024**3)
    assert len(plan.entries) == 1
    e = plan.entries[0]
    assert e.status == "safe"
    assert e.name == "Qwen2.5-7B"
    assert "Qwen/Qwen2.5-7B" in e.hf_ids


def test_evictor_classifies_in_progress_when_within_grace(tmp_path):
    """A dir whose mtime is inside the grace window is in_progress
    regardless of run state — protects active downloads / mmap'd
    weights."""
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    cache.mkdir()
    _mk_cache_dir(cache, "Recent-Model", size_mb=5,
                  mtime_offset_s=-60)  # 1 min old
    _mk_run(runs, "r-002", "org/Recent-Model", status="ok")
    plan = find_evictable(cache, runs, quota_bytes=1024**3,
                          grace_s=DEFAULT_GRACE_S)
    assert plan.entries[0].status == "in_progress"
    # And in_progress entries never appear in evictable
    assert plan.evictable == []


def test_evictor_classifies_in_progress_when_run_active(tmp_path):
    """Even if the dir is old, a live in_progress run referencing it
    protects the weights — vllm could still be serving from them."""
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    cache.mkdir()
    _mk_cache_dir(cache, "ActiveEval", size_mb=5,
                  mtime_offset_s=-3 * 3600)
    _mk_run(runs, "r-003", "org/ActiveEval", status="in_progress")
    plan = find_evictable(cache, runs, quota_bytes=1024**3)
    assert plan.entries[0].status == "in_progress"


def test_evictor_classifies_failed_only(tmp_path):
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    cache.mkdir()
    _mk_cache_dir(cache, "Broken-Model", size_mb=5, mtime_offset_s=-3 * 3600)
    _mk_run(runs, "r-004", "org/Broken-Model", status="failed",
            ended_at=time.time() - 3600)
    _mk_run(runs, "r-005", "org/Broken-Model", status="aborted",
            ended_at=time.time() - 1800)
    plan = find_evictable(cache, runs, quota_bytes=1024**3)
    assert plan.entries[0].status == "failed_only"


def test_evictor_classifies_orphan_when_no_runs_match(tmp_path):
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    cache.mkdir()
    runs.mkdir()
    _mk_cache_dir(cache, "Manually-Uploaded", size_mb=5,
                  mtime_offset_s=-3 * 3600)
    # No matching run dir
    plan = find_evictable(cache, runs, quota_bytes=1024**3)
    assert plan.entries[0].status == "orphan"


# ── enforce_quota ──────────────────────────────────────────────────────


def test_evictor_enforce_quota_picks_orphan_first(tmp_path):
    """When over quota, orphan should be deleted before failed_only
    which should be deleted before safe (assuming sizes don't matter
    for tie-breaking; mtime is the second key)."""
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    cache.mkdir()
    # All 100 MB, all 3h old (outside grace). Quota=250 MB → need to
    # drop ~150 MB. Order should be orphan → failed → safe.
    _mk_cache_dir(cache, "SafeModel", 100, mtime_offset_s=-3 * 3600)
    _mk_run(runs, "r-safe", "org/SafeModel", status="ok",
            ended_at=time.time() - 3600)
    _mk_cache_dir(cache, "FailedModel", 100, mtime_offset_s=-3 * 3600)
    _mk_run(runs, "r-fail", "org/FailedModel", status="failed",
            ended_at=time.time() - 3600)
    _mk_cache_dir(cache, "OrphanModel", 100, mtime_offset_s=-3 * 3600)
    # Quota 50 MB → need to drop 250 MB. Algo: evicts in priority
    # order until freed >= need. Orphan(100) + Failed(100) = 200 MB,
    # still short. Then Safe(100) = 300 MB total freed. All three out.
    quota = 50 * 1024 * 1024
    plan = enforce_quota(cache, runs, quota_bytes=quota, dry_run=False)
    evicted_names = [e.name for e in plan.evicted]
    # Priority order: orphan first, then failed_only, then safe.
    assert evicted_names == ["OrphanModel", "FailedModel", "SafeModel"]
    assert not (cache / "OrphanModel").exists()
    assert not (cache / "FailedModel").exists()
    assert not (cache / "SafeModel").exists()


def test_evictor_enforce_quota_stops_at_quota(tmp_path):
    """If freeing just orphan brings us under quota, don't keep going
    into failed/safe — minimum eviction."""
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    cache.mkdir()
    _mk_cache_dir(cache, "SafeModel", 100, mtime_offset_s=-3 * 3600)
    _mk_run(runs, "r-safe", "org/SafeModel", status="ok",
            ended_at=time.time() - 3600)
    _mk_cache_dir(cache, "OrphanModel", 100, mtime_offset_s=-3 * 3600)
    # 200 MB total, quota 150 MB, need to drop 50 MB → one eviction
    # suffices (orphan first by priority).
    quota = 150 * 1024 * 1024
    plan = enforce_quota(cache, runs, quota_bytes=quota, dry_run=False)
    evicted_names = [e.name for e in plan.evicted]
    assert evicted_names == ["OrphanModel"]
    assert (cache / "SafeModel").is_dir()


def test_evictor_enforce_quota_dry_run_does_not_delete(tmp_path):
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    cache.mkdir()
    _mk_cache_dir(cache, "ToEvict", 100, mtime_offset_s=-3 * 3600)
    quota = 1 * 1024 * 1024  # 1 MB, way under
    plan = enforce_quota(cache, runs, quota_bytes=quota, dry_run=True)
    assert plan.evicted, "dry-run should still report what would go"
    assert (cache / "ToEvict").is_dir(), "dry-run must NOT delete"
    # dry-run reports projected freed bytes (useful for the CLI preview)
    assert plan.bytes_freed > 0


def test_evictor_does_not_touch_in_progress_under_quota_pressure(tmp_path):
    """Quota pressure must NOT override the in_progress guard. If
    only in_progress dirs exist, nothing gets evicted (we degrade to
    over-quota rather than corrupt an active eval)."""
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    cache.mkdir()
    _mk_cache_dir(cache, "ActiveDownload", 500,
                  mtime_offset_s=-60)  # 1 min old
    quota = 100 * 1024 * 1024
    plan = enforce_quota(cache, runs, quota_bytes=quota, dry_run=False)
    assert plan.evicted == []
    assert (cache / "ActiveDownload").is_dir()


def test_evictor_under_quota_is_noop(tmp_path):
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    cache.mkdir()
    _mk_cache_dir(cache, "Small", 5, mtime_offset_s=-3 * 3600)
    quota = 1024 * 1024 * 1024  # 1 GB, way over what's there
    plan = enforce_quota(cache, runs, quota_bytes=quota, dry_run=False)
    assert plan.evicted == []
    assert plan.bytes_freed == 0
    assert (cache / "Small").is_dir()


def test_evictor_picks_oldest_within_same_priority(tmp_path):
    """Within 'safe', older mtime should evict before newer."""
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    cache.mkdir()
    _mk_cache_dir(cache, "OldSafe", 100, mtime_offset_s=-7 * 24 * 3600)
    _mk_run(runs, "r-old", "org/OldSafe", status="ok",
            ended_at=time.time() - 7 * 24 * 3600)
    _mk_cache_dir(cache, "NewSafe", 100, mtime_offset_s=-3 * 3600)
    _mk_run(runs, "r-new", "org/NewSafe", status="ok",
            ended_at=time.time() - 3600)
    quota = 150 * 1024 * 1024  # need to drop ~50 MB → drop one
    plan = enforce_quota(cache, runs, quota_bytes=quota, dry_run=False)
    evicted_names = [e.name for e in plan.evicted]
    assert evicted_names == ["OldSafe"], evicted_names


def test_evictor_handles_missing_runs_dir(tmp_path):
    """Robustness: if runs_root doesn't exist (fresh box), evictor
    treats every cache dir as orphan and proceeds without crashing."""
    cache = tmp_path / "cache"
    cache.mkdir()
    runs = tmp_path / "nonexistent"
    _mk_cache_dir(cache, "X", 5, mtime_offset_s=-3 * 3600)
    plan = find_evictable(cache, runs, quota_bytes=1024**3)
    assert plan.entries[0].status == "orphan"


def test_evictor_to_dict_is_json_serializable(tmp_path):
    """Logging contract: plan.to_dict() must roundtrip through JSON
    so the orchestrator can write it as an artifact."""
    cache = tmp_path / "cache"
    runs = tmp_path / "runs"
    cache.mkdir()
    _mk_cache_dir(cache, "Sample", 5, mtime_offset_s=-3 * 3600)
    plan = find_evictable(cache, runs, quota_bytes=1024**3)
    s = json.dumps(plan.to_dict())
    parsed = json.loads(s)
    assert parsed["n_entries"] == 1
    assert parsed["entries"][0]["name"] == "Sample"


# ── stager integration ────────────────────────────────────────────────


def test_stager_lru_eviction_runs_before_download(tmp_path, monkeypatch):
    """When ensure_model_staged is called with cache_quota_bytes and
    the cache is over quota, LRU should fire before the download
    starts. We verify by setting a quota so tight that one existing
    dir gets evicted; the stager should report the eviction message
    on stderr (best-effort logging) and proceed to the download
    attempt."""
    from orchestrator import model_stager

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    # Existing cache: one 100 MB 'safe' dir.
    _mk_cache_dir(cache_root, "OldSafe", 100, mtime_offset_s=-7 * 3600)
    _mk_run(runs_root, "r-old", "org/OldSafe", status="ok",
            ended_at=time.time() - 3600)

    target = cache_root / "NewModel"

    # Stub downloader so we don't actually fetch anything from HF.
    called = {"n": 0}

    def _fake_downloader(*args, **kwargs):
        called["n"] += 1
        # Simulate creating a couple weight files so _has_weight_file()
        # returns True on the post-check.
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model.safetensors").write_bytes(b"\x00" * 1024)
        return None

    quota = 50 * 1024 * 1024  # 50 MB → OldSafe (100 MB) is over quota
    result = model_stager.ensure_model_staged(
        hf_id="org/NewModel",
        target_dir=target,
        metadata={"hf_info": {"usedStorage": 1024}},  # tiny estimate
        engine_plan={"engine": "vllm"},
        downloader=_fake_downloader,
        cache_quota_bytes=quota,
        runs_root=runs_root,
        headroom_floor_bytes=1024,  # tiny so pre-flight passes
    )
    # OldSafe should have been evicted
    assert not (cache_root / "OldSafe").exists()
    # The downloader fired (eviction did not block staging)
    assert called["n"] == 1
    assert result.ok is True
