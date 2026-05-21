"""Tests for backup/snapshot.py — see docs/PR6_TEST_PLAN.md §3.1."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backup.snapshot import (
    BackupState,
    RsyncOutcome,
    SnapshotResult,
    _atomic_symlink,
    _is_snapshot_name,
    _latest_snapshot,
    _overlaps,
    read_backup_state,
    take_snapshot,
)
from orchestrator.config import OrchestratorConfig

# ─── fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def cfg(tmp_path: Path) -> OrchestratorConfig:
    data = tmp_path / "data"
    data.mkdir()
    (data / "store").mkdir()
    (data / "runs").mkdir()
    (data / "store" / "runs.sqlite").write_bytes(b"\x00\x01\x02")
    (data / "runs" / "abc").mkdir()
    (data / "runs" / "abc" / "state.json").write_text('{"hello":1}')
    return OrchestratorConfig(
        data_root=data,
        backups_root=tmp_path / "backups",
    )


@dataclass
class _RsyncCall:
    """Captures one call to the rsync seam so tests can assert on args."""
    src: Path
    dst: Path
    link_dest: Path | None


def _make_rsync_seam(outcome: RsyncOutcome | None = None):
    calls: list[_RsyncCall] = []
    fixed_outcome = outcome or RsyncOutcome(
        returncode=0, seconds=0.1,
        files_total=4, files_transferred=4, size_bytes=12345,
    )

    def _seam(src: Path, dst: Path, link_dest: Path | None) -> RsyncOutcome:
        # Simulate rsync filling the destination so meta/health tests see
        # a non-empty tree (rsync was mocked, so we copy ourselves).
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "marker").write_text("ok")
        calls.append(_RsyncCall(src=src, dst=dst, link_dest=link_dest))
        return fixed_outcome

    return _seam, calls


def _at(ts: str = "20260521_180000") -> datetime:
    return datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=UTC)


# ─── helpers tests ─────────────────────────────────────────────────────────

def test_is_snapshot_name_accepts_valid():
    assert _is_snapshot_name("20260521_180000")
    assert not _is_snapshot_name("latest")
    assert not _is_snapshot_name("20260521-180000")
    assert not _is_snapshot_name("2026052_1800000")
    assert not _is_snapshot_name("YYYYMMDD_HHMMSS")
    assert not _is_snapshot_name("")


def test_overlaps_detects_parent_child(tmp_path: Path):
    a = tmp_path / "data"
    b = tmp_path / "data" / "backups"
    a.mkdir()
    b.mkdir()
    assert _overlaps(a, b)
    assert _overlaps(b, a)
    assert _overlaps(a, a)
    c = tmp_path / "other"
    c.mkdir()
    assert not _overlaps(a, c)


# ─── H: happy path ─────────────────────────────────────────────────────────

def test_h1_first_snapshot_no_link_dest(cfg: OrchestratorConfig):
    seam, calls = _make_rsync_seam()
    result = take_snapshot(cfg, now_fn=lambda: _at(), run_rsync=seam)

    assert isinstance(result, SnapshotResult)
    assert result.ok is True
    assert result.snapshot_ts == "20260521_180000"
    assert result.snapshot_dir is not None
    assert result.snapshot_dir.exists()
    assert len(calls) == 1
    assert calls[0].link_dest is None
    assert calls[0].src == cfg.data_root.resolve()

    # last_backup.txt + symlink
    assert (cfg.backups_root / "last_backup.txt").exists()
    last = (cfg.backups_root / "last_backup.txt").read_text()
    assert last == _at().isoformat()
    latest = cfg.backups_root / "latest"
    assert latest.is_symlink()
    assert os.readlink(latest) == "20260521_180000"


def test_h2_incremental_uses_prev_as_link_dest(cfg: OrchestratorConfig):
    seam, calls = _make_rsync_seam()

    first = take_snapshot(cfg, now_fn=lambda: _at("20260521_180000"), run_rsync=seam)
    second = take_snapshot(cfg, now_fn=lambda: _at("20260521_183000"), run_rsync=seam)

    assert first.ok and second.ok
    assert len(calls) == 2
    assert calls[0].link_dest is None
    assert calls[1].link_dest == cfg.backups_root.resolve() / "20260521_180000"
    assert os.readlink(cfg.backups_root / "latest") == "20260521_183000"


def test_h3_backup_meta_is_complete(cfg: OrchestratorConfig):
    outcome = RsyncOutcome(
        returncode=0, seconds=12.5,
        files_total=999, files_transferred=42, size_bytes=98765432,
    )
    seam, _ = _make_rsync_seam(outcome=outcome)

    take_snapshot(cfg, now_fn=lambda: _at("20260521_180000"), run_rsync=seam)
    take_snapshot(cfg, now_fn=lambda: _at("20260521_183000"), run_rsync=seam)

    meta_path = cfg.backups_root / "20260521_183000" / "backup_meta.json"
    meta = json.loads(meta_path.read_text())

    assert meta["snapshot_ts"] == "20260521_183000"
    assert meta["snapshot_ts_iso"] == _at("20260521_183000").isoformat()
    assert meta["src"] == str(cfg.data_root.resolve())
    assert meta["rsync_seconds"] == 12.5
    assert meta["rsync_files_total"] == 999
    assert meta["rsync_files_transferred"] == 42
    assert meta["size_bytes"] == 98765432
    assert meta["link_dest_from"] == "20260521_180000"


def test_h4_last_backup_txt_is_iso8601_with_tz(cfg: OrchestratorConfig):
    seam, _ = _make_rsync_seam()
    take_snapshot(cfg, now_fn=lambda: _at(), run_rsync=seam)

    raw = (cfg.backups_root / "last_backup.txt").read_text(encoding="utf-8").strip()
    parsed = datetime.fromisoformat(raw)
    assert parsed.tzinfo is not None
    assert parsed == _at()


# ─── S: sad path ───────────────────────────────────────────────────────────

def test_s1_rsync_nonzero_cleans_partial(cfg: OrchestratorConfig):
    bad_outcome = RsyncOutcome(returncode=23, seconds=0.1, stderr_tail="boom")

    def bad_seam(src: Path, dst: Path, link_dest: Path | None) -> RsyncOutcome:
        dst.mkdir(parents=True, exist_ok=True)
        (dst / "partial").write_text("garbage")
        return bad_outcome

    events: list[dict] = []
    result = take_snapshot(
        cfg, now_fn=lambda: _at(),
        run_rsync=bad_seam,
        outbox_writer=events.append,
    )

    assert result.ok is False
    assert result.error_kind == "rsync_nonzero"
    assert not (cfg.backups_root / "20260521_180000").exists()
    assert not (cfg.backups_root / "last_backup.txt").exists()
    assert not (cfg.backups_root / "latest").exists()
    assert events and events[0]["event_type"] == "backup_failed"
    assert events[0]["level"] == "error"


def test_s2_meta_write_failure_rolls_back(
    cfg: OrchestratorConfig, monkeypatch: pytest.MonkeyPatch,
):
    seam, _ = _make_rsync_seam()
    # Force write_text to raise OSError just for backup_meta.json.
    real_write_text = Path.write_text

    def failing_write_text(self: Path, *a: object, **kw: object) -> int:
        if self.name == "backup_meta.json":
            raise OSError("ENOSPC")
        return real_write_text(self, *a, **kw)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "write_text", failing_write_text)
    events: list[dict] = []
    result = take_snapshot(
        cfg, now_fn=lambda: _at(),
        run_rsync=seam,
        outbox_writer=events.append,
    )

    assert result.ok is False
    assert result.error_kind == "meta_write_failed"
    assert not (cfg.backups_root / "20260521_180000").exists()
    assert events


def test_s3_overlap_aborts_before_rsync(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    cfg = OrchestratorConfig(
        data_root=data,
        backups_root=data / "nested-backups",  # NOTE: under data_root
    )
    called: list[object] = []

    def must_not_be_called(*a: object, **kw: object) -> RsyncOutcome:
        called.append(1)
        return RsyncOutcome(returncode=0, seconds=0)

    result = take_snapshot(cfg, run_rsync=must_not_be_called)

    assert result.ok is False
    assert result.error_kind == "overlap"
    assert not called


def test_s4_rsync_missing_binary(cfg: OrchestratorConfig):
    def missing(src: Path, dst: Path, link_dest: Path | None) -> RsyncOutcome:
        raise FileNotFoundError("rsync")

    events: list[dict] = []
    result = take_snapshot(
        cfg, now_fn=lambda: _at(),
        run_rsync=missing,
        outbox_writer=events.append,
    )

    assert result.ok is False
    assert result.error_kind == "rsync_missing"
    assert not (cfg.backups_root / "20260521_180000").exists()


def test_src_missing_returns_failure(tmp_path: Path):
    cfg = OrchestratorConfig(
        data_root=tmp_path / "nonexistent",
        backups_root=tmp_path / "backups",
    )
    seam, calls = _make_rsync_seam()

    result = take_snapshot(cfg, run_rsync=seam)

    assert result.ok is False
    assert result.error_kind == "src_missing"
    assert not calls


# ─── E: edge ───────────────────────────────────────────────────────────────

def test_e1_empty_source(tmp_path: Path):
    data = tmp_path / "data"
    data.mkdir()
    cfg = OrchestratorConfig(data_root=data, backups_root=tmp_path / "backups")
    seam, calls = _make_rsync_seam(
        outcome=RsyncOutcome(returncode=0, seconds=0, files_total=0, size_bytes=0),
    )

    result = take_snapshot(cfg, now_fn=lambda: _at(), run_rsync=seam)

    assert result.ok is True
    assert len(calls) == 1
    meta = json.loads((cfg.backups_root / "20260521_180000" / "backup_meta.json").read_text())
    assert meta["size_bytes"] == 0


def test_e3_partial_from_prior_crash_is_wiped(cfg: OrchestratorConfig):
    # Pre-create a partial directory at the same ts.
    partial = cfg.backups_root / "20260521_180000"
    partial.mkdir(parents=True)
    (partial / "leftover.txt").write_text("garbage from crashed run")

    seam, _calls = _make_rsync_seam()
    result = take_snapshot(cfg, now_fn=lambda: _at(), run_rsync=seam)

    assert result.ok is True
    assert not (partial / "leftover.txt").exists()
    assert (partial / "marker").exists()  # written by our seam, proves it's fresh
    assert (partial / "backup_meta.json").exists()


def test_e4_stale_latest_symlink_recovers(cfg: OrchestratorConfig):
    # Create a `latest` pointing to a directory that no longer exists.
    cfg.backups_root.mkdir(parents=True, exist_ok=True)
    os.symlink("does-not-exist", cfg.backups_root / "latest")

    seam, _ = _make_rsync_seam()
    result = take_snapshot(cfg, now_fn=lambda: _at(), run_rsync=seam)

    assert result.ok is True
    assert os.readlink(cfg.backups_root / "latest") == "20260521_180000"


def test_latest_snapshot_excludes_in_progress(cfg: OrchestratorConfig):
    cfg.backups_root.mkdir()
    (cfg.backups_root / "20260520_120000").mkdir()
    (cfg.backups_root / "20260521_120000").mkdir()
    (cfg.backups_root / "20260522_120000").mkdir()

    assert _latest_snapshot(cfg.backups_root) == cfg.backups_root / "20260522_120000"
    assert _latest_snapshot(cfg.backups_root, exclude="20260522_120000") == \
        cfg.backups_root / "20260521_120000"


def test_atomic_symlink_replaces_existing(tmp_path: Path):
    link = tmp_path / "latest"
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    _atomic_symlink(link, "a")
    assert os.readlink(link) == "a"

    _atomic_symlink(link, "b")
    assert os.readlink(link) == "b"


def test_atomic_symlink_cleans_stale_tmp(tmp_path: Path):
    """If a previous run died between symlink-create and rename, the .tmp
    file might linger. The next invocation must clean it up first."""
    link = tmp_path / "latest"
    tmp = tmp_path / "latest.tmp"
    (tmp_path / "a").mkdir()
    # leave a stale .tmp behind
    os.symlink("does-not-exist", tmp)
    _atomic_symlink(link, "a")
    assert os.readlink(link) == "a"
    assert not tmp.exists() and not tmp.is_symlink()


def test_list_snapshot_dirs_absent_root_returns_empty(tmp_path: Path):
    from backup.snapshot import _list_snapshot_dirs
    out = _list_snapshot_dirs(tmp_path / "no-such-dir")
    assert out == []


def test_list_snapshot_dirs_skips_non_directory_entries(tmp_path: Path):
    from backup.snapshot import _list_snapshot_dirs
    (tmp_path / "20260521_120000").mkdir()
    # A *file* named like a snapshot ts must NOT show up
    (tmp_path / "20260521_180000").write_text("decoy file, not a dir")
    out = _list_snapshot_dirs(tmp_path)
    assert [p.name for p in out] == ["20260521_120000"]


# ─── INV-6 / INV-9 invariants ──────────────────────────────────────────────

def test_inv6_last_backup_txt_lives_in_backups_root_not_data_root(
    cfg: OrchestratorConfig,
):
    seam, _ = _make_rsync_seam()
    take_snapshot(cfg, now_fn=lambda: _at(), run_rsync=seam)

    assert (cfg.backups_root / "last_backup.txt").exists()
    assert not (cfg.data_root / "store" / "last_backup.txt").exists()
    assert not (cfg.data_root / "last_backup.txt").exists()


def test_inv9_source_dir_is_unchanged(cfg: OrchestratorConfig):
    """The backup pass must not mutate any file inside data_root."""
    seam, _ = _make_rsync_seam()
    before: dict[str, tuple[int, int]] = {}
    for p in cfg.data_root.rglob("*"):
        if p.is_file():
            st = p.stat()
            before[str(p)] = (st.st_size, st.st_mtime_ns)

    take_snapshot(cfg, now_fn=lambda: _at(), run_rsync=seam)

    after: dict[str, tuple[int, int]] = {}
    for p in cfg.data_root.rglob("*"):
        if p.is_file():
            st = p.stat()
            after[str(p)] = (st.st_size, st.st_mtime_ns)

    assert before == after


def test_inv10_backup_module_no_forbidden_imports():
    """Static check: backup/ must not import docker / heyi_engine / cc_agent / anthropic."""
    import importlib
    import sys

    forbidden = {"docker", "anthropic", "cc_agent", "heyi_engine"}
    # Ensure the module is loaded so transitive imports show up.
    importlib.import_module("backup")
    importlib.import_module("backup.snapshot")
    importlib.import_module("backup.retention")

    leaked: list[str] = []
    for name in list(sys.modules):
        if name in forbidden:
            # Allow these if loaded by other tests; only fail if backup.* pulled them in.
            # Conservative check: backup/__main__ imports orchestrator.config but that
            # module itself does not import any of the forbidden names — verified below.
            pass

    # Direct AST check is cheaper and more reliable: walk the backup/ source.
    import ast
    from pathlib import Path as P
    backup_dir = P(__file__).resolve().parent.parent / "backup"
    for src_path in backup_dir.rglob("*.py"):
        tree = ast.parse(src_path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in forbidden:
                        leaked.append(f"{src_path.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".", 1)[0]
                if root in forbidden:
                    leaked.append(f"{src_path.name}: from {node.module}")

    assert not leaked, f"backup/ leaked forbidden imports: {leaked}"


# ─── read_backup_state (panel data layer) ──────────────────────────────────

def test_read_backup_state_after_one_snapshot(cfg: OrchestratorConfig):
    seam, _ = _make_rsync_seam(
        outcome=RsyncOutcome(returncode=0, seconds=0.1, size_bytes=12345),
    )
    take_snapshot(cfg, now_fn=lambda: _at("20260521_180000"), run_rsync=seam)

    state = read_backup_state(
        cfg.backups_root, now_fn=lambda: _at("20260521_181000"),
    )
    assert isinstance(state, BackupState)
    assert state.health == "ok"
    assert state.snapshot_count == 1
    assert state.total_size_bytes == 12345
    assert state.last_backup_age_s == 600
    assert state.last_backup_ts is not None


def test_read_backup_state_no_backups_root(tmp_path: Path):
    state = read_backup_state(tmp_path / "absent")
    assert state.health == "down"
    assert state.snapshot_count == 0
    assert state.last_backup_ts is None


def test_read_backup_state_old_backup_is_warn(cfg: OrchestratorConfig):
    seam, _ = _make_rsync_seam()
    take_snapshot(cfg, now_fn=lambda: _at("20260521_000000"), run_rsync=seam)

    # 2h later → warn
    state = read_backup_state(
        cfg.backups_root, now_fn=lambda: _at("20260521_020000"),
    )
    assert state.health == "warn"


def test_read_backup_state_very_old_backup_is_down(cfg: OrchestratorConfig):
    seam, _ = _make_rsync_seam()
    take_snapshot(cfg, now_fn=lambda: _at("20260520_000000"), run_rsync=seam)

    # 36h later → down
    state = read_backup_state(
        cfg.backups_root, now_fn=lambda: _at("20260521_120000"),
    )
    assert state.health == "down"


def test_read_backup_state_corrupted_last_backup_txt(cfg: OrchestratorConfig):
    cfg.backups_root.mkdir()
    (cfg.backups_root / "last_backup.txt").write_text("not a date")
    state = read_backup_state(cfg.backups_root)
    assert state.last_backup_ts is None
    assert state.health == "down"


def test_read_backup_state_dir_without_meta(cfg: OrchestratorConfig):
    cfg.backups_root.mkdir()
    (cfg.backups_root / "20260521_180000").mkdir()
    (cfg.backups_root / "last_backup.txt").write_text(_at("20260521_180000").isoformat())
    state = read_backup_state(
        cfg.backups_root, now_fn=lambda: _at("20260521_181000"),
    )
    # Missing meta is tolerated; size_bytes=0 for that snapshot, health still ok.
    assert state.snapshot_count == 1
    assert state.total_size_bytes == 0
    assert state.health == "ok"


def test_read_backup_state_meta_with_bad_json(cfg: OrchestratorConfig):
    cfg.backups_root.mkdir()
    snap = cfg.backups_root / "20260521_180000"
    snap.mkdir()
    (snap / "backup_meta.json").write_text("not-json{{")
    (cfg.backups_root / "last_backup.txt").write_text(_at("20260521_180000").isoformat())
    state = read_backup_state(
        cfg.backups_root, now_fn=lambda: _at("20260521_181000"),
    )
    assert state.snapshot_count == 1
    assert state.total_size_bytes == 0
    snap_summary = state.snapshots[0]
    assert snap_summary["meta_ok"] is False
    assert snap_summary["size_bytes"] == 0


def test_read_backup_state_no_snapshots_but_pointer_says_recent(cfg: OrchestratorConfig):
    """A `last_backup.txt` without any backing snapshot dir is degraded."""
    cfg.backups_root.mkdir()
    (cfg.backups_root / "last_backup.txt").write_text(_at("20260521_180000").isoformat())
    state = read_backup_state(
        cfg.backups_root, now_fn=lambda: _at("20260521_180500"),
    )
    assert state.health == "down"
    assert state.snapshot_count == 0


# ─── _first_int / _parse_rsync_stats helpers ───────────────────────────────


def test_first_int_basic():
    from backup.snapshot import _first_int
    assert _first_int("Number of files: 1,234 (reg: 1,200)") == 1234
    assert _first_int("Total file size: 567,890,123 bytes") == 567890123
    assert _first_int("Number transferred: 0") == 0
    assert _first_int("no numbers here at all!") == 0
    assert _first_int("") == 0


def test_first_int_handles_trailing_separators():
    from backup.snapshot import _first_int
    # Trailing comma after the last digit shouldn't make us slurp into next word.
    assert _first_int("1,234,foo") == 1234


def test_parse_rsync_stats_full_block():
    from backup.snapshot import _parse_rsync_stats
    stdout = (
        "Number of files: 1,842 (reg: 1,820, dir: 22)\n"
        "Number of regular files transferred: 23\n"
        "Total file size: 12,345,678 bytes\n"
    )
    total, transferred, size = _parse_rsync_stats(stdout)
    assert total == 1842
    assert transferred == 23
    assert size == 12345678


def test_parse_rsync_stats_missing_lines():
    from backup.snapshot import _parse_rsync_stats
    total, transferred, size = _parse_rsync_stats("")
    assert (total, transferred, size) == (0, 0, 0)


# ─── _default_run_rsync via subprocess seam ────────────────────────────────


def test_default_run_rsync_success(monkeypatch, tmp_path: Path):
    from backup import snapshot as snap_mod

    captured = {}

    class FakeProc:
        def __init__(self):
            self.returncode = 0
            self.stdout = (
                "Number of files: 42\n"
                "Number of regular files transferred: 5\n"
                "Total file size: 9999 bytes\n"
            )
            self.stderr = ""

    def fake_run(cmd, **kw):
        captured["cmd"] = cmd
        captured["kw"] = kw
        return FakeProc()

    monkeypatch.setattr(snap_mod.subprocess, "run", fake_run)

    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    link = tmp_path / "prev"
    link.mkdir()

    result = snap_mod._default_run_rsync(src, dst, link)
    assert result.returncode == 0
    assert result.files_total == 42
    assert result.files_transferred == 5
    assert result.size_bytes == 9999
    # arg shape: rsync -a --delete --stats … --link-dest=<abs> src/ dst
    cmd = captured["cmd"]
    assert cmd[0] == "rsync"
    assert "-a" in cmd
    assert "--delete" in cmd
    assert any(c.startswith("--link-dest=") for c in cmd)
    assert cmd[-2].rstrip("/") == str(src).rstrip("/")
    assert cmd[-1] == str(dst)


def test_default_run_rsync_missing_binary(monkeypatch, tmp_path: Path):
    from backup import snapshot as snap_mod

    def boom(*a, **kw):
        raise FileNotFoundError("rsync")

    monkeypatch.setattr(snap_mod.subprocess, "run", boom)

    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    dst.mkdir()
    result = snap_mod._default_run_rsync(src, dst, None)
    assert result.returncode == 127
    assert "FileNotFoundError" in result.stderr_tail


def test_default_run_rsync_no_link_dest_omits_flag(monkeypatch, tmp_path: Path):
    from backup import snapshot as snap_mod

    captured = {}

    class FakeProc:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(
        snap_mod.subprocess, "run",
        lambda cmd, **kw: (captured.__setitem__("cmd", cmd), FakeProc())[1],
    )

    src = tmp_path / "src"
    src.mkdir()
    dst = tmp_path / "dst"
    dst.mkdir()
    snap_mod._default_run_rsync(src, dst, None)
    assert not any(c.startswith("--link-dest=") for c in captured["cmd"])


# ─── rsync seam exception path ─────────────────────────────────────────────


def test_rsync_seam_crash_is_translated_to_failure(cfg: OrchestratorConfig):
    def crashy(src: Path, dst: Path, link_dest: Path | None) -> RsyncOutcome:
        raise RuntimeError("seam exploded")

    events: list[dict] = []
    result = take_snapshot(
        cfg, now_fn=lambda: _at(),
        run_rsync=crashy,
        outbox_writer=events.append,
    )
    assert result.ok is False
    assert result.error_kind == "rsync_crash"
    assert "seam exploded" in result.error
    assert events and events[0]["event_type"] == "backup_failed"


# ─── outbox writer that itself fails must not propagate ────────────────────


def test_failing_outbox_writer_does_not_propagate(cfg: OrchestratorConfig):
    def crashy(src: Path, dst: Path, link_dest: Path | None) -> RsyncOutcome:
        return RsyncOutcome(returncode=23, seconds=0.1, stderr_tail="x")

    def bad_writer(payload):
        raise RuntimeError("disk full")

    # Should NOT raise, even though outbox writer blows up.
    result = take_snapshot(
        cfg, now_fn=lambda: _at(),
        run_rsync=crashy,
        outbox_writer=bad_writer,
    )
    assert result.ok is False
    assert result.error_kind == "rsync_nonzero"


# ─── pointer update failure leaves snapshot intact ─────────────────────────


def test_pointer_update_failure_does_not_wipe_snapshot(
    cfg: OrchestratorConfig, monkeypatch,
):
    seam, _ = _make_rsync_seam()
    import backup.snapshot as snap_mod

    # Make _atomic_symlink raise after rsync + meta succeeded.
    def boom(*a, **kw):
        raise OSError("symlink failed")

    monkeypatch.setattr(snap_mod, "_atomic_symlink", boom)

    events: list[dict] = []
    result = take_snapshot(
        cfg, now_fn=lambda: _at(),
        run_rsync=seam,
        outbox_writer=events.append,
    )
    assert result.ok is False
    assert result.error_kind == "pointer_update_failed"
    # Snapshot dir + meta survive so next run can use it as link-dest base.
    assert (cfg.backups_root / "20260521_180000").exists()
    assert (cfg.backups_root / "20260521_180000" / "backup_meta.json").exists()
    assert events
