"""Tests for backup/retention.py — see docs/PR6_TEST_PLAN.md §3.2."""
from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from backup.retention import prune_old_snapshots

# ─── fixtures ──────────────────────────────────────────────────────────────

def _make_snapshot(backups_root: Path, ts: str, *, mtime_days_ago: int = 0) -> Path:
    """Create a fake snapshot dir with backup_meta.json + adjusted mtime."""
    d = backups_root / ts
    d.mkdir(parents=True, exist_ok=False)
    (d / "backup_meta.json").write_text("{}")
    (d / "store").mkdir()
    (d / "store" / "runs.sqlite").write_bytes(b"\x00")
    if mtime_days_ago > 0:
        t = (datetime.now(UTC) - timedelta(days=mtime_days_ago)).timestamp()
        os.utime(d, (t, t))
    return d


def _now() -> datetime:
    return datetime(2026, 5, 21, 18, 0, 0, tzinfo=UTC)


# ─── R: retention cases ────────────────────────────────────────────────────

def test_r1_all_within_window(tmp_path: Path):
    root = tmp_path / "backups"
    root.mkdir()
    for i, ts in enumerate(["20260520_120000", "20260520_180000", "20260521_120000"]):
        _make_snapshot(root, ts, mtime_days_ago=i)

    result = prune_old_snapshots(root, keep_days=7, now_fn=_now)

    assert result.removed == []
    assert sorted(result.kept) == ["20260520_120000", "20260520_180000", "20260521_120000"]
    assert result.failed == []


def test_r2_partial_aged(tmp_path: Path):
    root = tmp_path / "backups"
    root.mkdir()
    _make_snapshot(root, "20260101_120000", mtime_days_ago=140)
    _make_snapshot(root, "20260301_120000", mtime_days_ago=80)
    _make_snapshot(root, "20260520_120000", mtime_days_ago=1)
    _make_snapshot(root, "20260521_120000", mtime_days_ago=0)

    result = prune_old_snapshots(root, keep_days=7, now_fn=_now)

    assert sorted(result.removed) == ["20260101_120000", "20260301_120000"]
    assert sorted(result.kept) == ["20260520_120000", "20260521_120000"]
    # The removed ones really are gone:
    for ts in result.removed:
        assert not (root / ts).exists()


def test_r3_all_aged_keep_latest(tmp_path: Path):
    root = tmp_path / "backups"
    root.mkdir()
    _make_snapshot(root, "20260101_120000", mtime_days_ago=140)

    result = prune_old_snapshots(root, keep_days=7, now_fn=_now)

    # Lone snapshot stays even though it's months old.
    assert result.removed == []
    assert result.kept == ["20260101_120000"]
    assert (root / "20260101_120000").exists()


def test_r4_all_aged_n_keep_newest(tmp_path: Path):
    root = tmp_path / "backups"
    root.mkdir()
    _make_snapshot(root, "20260101_120000", mtime_days_ago=140)
    _make_snapshot(root, "20260201_120000", mtime_days_ago=110)
    _make_snapshot(root, "20260301_120000", mtime_days_ago=80)
    _make_snapshot(root, "20260401_120000", mtime_days_ago=50)
    _make_snapshot(root, "20260501_120000", mtime_days_ago=20)

    result = prune_old_snapshots(root, keep_days=7, now_fn=_now)

    assert "20260501_120000" in result.kept
    assert len(result.kept) == 1
    assert len(result.removed) == 4


def test_r5_non_snapshot_dirs_untouched(tmp_path: Path):
    root = tmp_path / "backups"
    root.mkdir()
    _make_snapshot(root, "20260101_120000", mtime_days_ago=140)
    _make_snapshot(root, "20260520_120000", mtime_days_ago=1)

    # Decoys
    (root / "latest").symlink_to("20260520_120000")
    (root / "last_backup.txt").write_text("2026-05-21T18:00:00+00:00")
    (root / "user-handcrafted").mkdir()
    (root / "user-handcrafted" / "file.txt").write_text("dont touch me")
    (root / "20260520.bak").mkdir()  # bad name pattern

    result = prune_old_snapshots(root, keep_days=7, now_fn=_now)

    assert result.removed == ["20260101_120000"]
    assert (root / "latest").is_symlink()
    assert (root / "last_backup.txt").exists()
    assert (root / "user-handcrafted").exists()
    assert (root / "user-handcrafted" / "file.txt").exists()
    assert (root / "20260520.bak").exists()


def test_r6_rmtree_failure_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path / "backups"
    root.mkdir()
    _make_snapshot(root, "20260101_120000", mtime_days_ago=140)
    _make_snapshot(root, "20260102_120000", mtime_days_ago=140)
    _make_snapshot(root, "20260520_120000", mtime_days_ago=1)

    import shutil

    import backup.retention as ret
    calls = {"n": 0}
    real_rmtree = shutil.rmtree

    def flaky_rmtree(*a: object, **kw: object) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("simulated permission denied")
        real_rmtree(*a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(ret.shutil, "rmtree", flaky_rmtree)

    result = ret.prune_old_snapshots(root, keep_days=7, now_fn=_now)

    assert len(result.failed) == 1
    assert len(result.removed) == 1
    assert "20260520_120000" in result.kept


def test_r7_backups_root_missing(tmp_path: Path):
    result = prune_old_snapshots(tmp_path / "absent", keep_days=7, now_fn=_now)
    assert result.removed == []
    assert result.failed == []
    assert result.kept == []


def test_empty_backups_root(tmp_path: Path):
    root = tmp_path / "backups"
    root.mkdir()
    result = prune_old_snapshots(root, keep_days=7, now_fn=_now)
    assert result.removed == []
    assert result.kept == []
    assert result.failed == []


def test_safe_mtime_returns_none_for_missing_path(tmp_path: Path):
    """Direct unit-test of _safe_mtime so its OSError → None branch is real."""
    from backup.retention import _safe_mtime
    assert _safe_mtime(tmp_path / "does-not-exist") is None
    real = tmp_path / "exists"
    real.mkdir()
    val = _safe_mtime(real)
    assert val is not None
    assert val > 0


def test_stat_failure_listed_as_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """If retention can't read mtime on a single snapshot we mark it failed
    and keep going. We patch the dedicated `_safe_mtime` seam so this test
    doesn't accidentally break Path.is_dir() on the prefilter pass — which
    is what happens on Python 3.11 where is_dir() internally calls stat()."""
    root = tmp_path / "backups"
    root.mkdir()
    _make_snapshot(root, "20260101_120000", mtime_days_ago=140)
    _make_snapshot(root, "20260520_120000", mtime_days_ago=1)

    import backup.retention as ret

    real_safe_mtime = ret._safe_mtime

    def flaky(path: Path) -> float | None:
        if path.name == "20260101_120000":
            return None
        return real_safe_mtime(path)

    monkeypatch.setattr(ret, "_safe_mtime", flaky)

    result = prune_old_snapshots(root, keep_days=7, now_fn=_now)

    assert result.failed == ["20260101_120000"]
    assert "20260520_120000" in result.kept
