"""
backup/snapshot.py — point-in-time rsync snapshot with hard-link sharing.

A `take_snapshot(cfg)` call performs one full pass:

  1. validate src/dst don't overlap (INV-9).
  2. resolve previous successful snapshot (for --link-dest space sharing).
  3. ensure target dir exists; if it already exists (partial from a prior
     crashed run) wipe it first.
  4. shell out to rsync via the injected `run_rsync` seam (default uses
     subprocess.run).
  5. on success: write backup_meta.json, atomically flip `latest` symlink,
     update `last_backup.txt`.
  6. on failure: wipe the partial target and emit an outbox event.

Trust boundary (INV-9 / INV-10):

  - Source directory is **read-only**. We never write to data_root.
  - All metadata (last_backup.txt, latest symlink, per-snapshot meta json)
    lives under backups_root, which must be disjoint from data_root.
  - No docker / heyi_engine / anthropic / claude imports — see __init__.py.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from orchestrator.config import OrchestratorConfig


# ─── result types ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RsyncOutcome:
    """Whatever the `run_rsync` seam returns. Tests inject this directly."""
    returncode: int
    seconds: float
    files_total: int = 0
    files_transferred: int = 0
    size_bytes: int = 0
    stderr_tail: str = ""


@dataclass(frozen=True)
class SnapshotResult:
    ok: bool
    snapshot_ts: str | None = None
    snapshot_dir: Path | None = None
    rsync_seconds: float = 0.0
    size_bytes: int = 0
    error: str = ""
    error_kind: str = ""


@dataclass(frozen=True)
class BackupState:
    """What the panel renders."""
    last_backup_ts: str | None  # ISO-8601 string (read from last_backup.txt)
    last_backup_age_s: int | None
    snapshot_count: int
    total_size_bytes: int
    health: str  # "ok" | "warn" | "down"
    backups_root: str
    snapshots: list[dict[str, Any]] = field(default_factory=list)


# ─── seams (overridable for tests) ─────────────────────────────────────────

RsyncRunner = Callable[[Path, Path, Path | None], RsyncOutcome]
NowFn = Callable[[], datetime]


def _default_now() -> datetime:
    return datetime.now(UTC)


def _default_run_rsync(src: Path, dst: Path, link_dest: Path | None) -> RsyncOutcome:
    """The real subprocess call. Single seam — tests patch this whole function."""
    # Trailing slash on src is intentional rsync semantics: copy contents,
    # not the directory itself, into dst.
    src_arg = str(src).rstrip("/") + "/"
    cmd = [
        "rsync",
        "-a",
        "--delete",
        "--stats",
        "--info=stats1",
        "--out-format=",
    ]
    if link_dest is not None:
        # rsync requires --link-dest to be either absolute or relative to dst.
        # We always pass it absolute to remove ambiguity.
        cmd.append(f"--link-dest={link_dest.resolve()}")
    cmd.extend([src_arg, str(dst)])

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,  # generous; rsync 1.5GB tree on local disk is ~30s
        )
    except FileNotFoundError as e:
        # rsync not installed on host
        return RsyncOutcome(
            returncode=127, seconds=time.time() - t0,
            stderr_tail=f"FileNotFoundError: {e}",
        )
    elapsed = time.time() - t0

    files_total, files_transferred, size_bytes = _parse_rsync_stats(proc.stdout)
    stderr_tail = (proc.stderr or "")[-2048:]
    return RsyncOutcome(
        returncode=proc.returncode,
        seconds=elapsed,
        files_total=files_total,
        files_transferred=files_transferred,
        size_bytes=size_bytes,
        stderr_tail=stderr_tail,
    )


def _parse_rsync_stats(stdout: str) -> tuple[int, int, int]:
    """Best-effort parse of rsync --info=stats1 output. Returns zeros if missing.

    Tolerates GNU rsync 3.x English locale stats blocks like:
      Number of files: 1,234 (reg: 1,200, dir: 34)
      Number of regular files transferred: 12
      Total file size: 567,890,123 bytes
    """
    files_total = files_transferred = size_bytes = 0
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("Number of files:"):
            files_total = _first_int(line)
        elif line.startswith("Number of regular files transferred:"):
            files_transferred = _first_int(line)
        elif line.startswith("Total file size:"):
            size_bytes = _first_int(line)
    return files_total, files_transferred, size_bytes


def _first_int(s: str) -> int:
    """Pull the first integer (allowing thousand separators) out of `s`."""
    cur: list[str] = []
    started = False
    for ch in s:
        if ch.isdigit():
            cur.append(ch)
            started = True
        elif ch == "," and started:
            continue
        elif started:
            break
    try:
        return int("".join(cur)) if cur else 0
    except ValueError:
        return 0


# ─── snapshot core ─────────────────────────────────────────────────────────

_TS_FORMAT = "%Y%m%d_%H%M%S"


def take_snapshot(
    cfg: OrchestratorConfig,
    *,
    now_fn: NowFn = _default_now,
    run_rsync: RsyncRunner = _default_run_rsync,
    outbox_writer: Callable[[dict[str, Any]], None] | None = None,
) -> SnapshotResult:
    """One full snapshot pass. Pure function modulo filesystem + rsync seams.

    `outbox_writer` (if provided) is called on failure with a notify event
    payload; the orchestrator wires this to its existing notify_outbox.jsonl
    in PR#7 via `python -m backup` invocation. Unit tests inject a stub.
    """
    src = cfg.data_root.expanduser().resolve()
    backups_root = cfg.backups_root.expanduser().resolve()

    # INV-9: src and backups_root must be disjoint.
    if _overlaps(src, backups_root):
        return _fail(
            error=f"backups_root {backups_root} overlaps data_root {src}",
            error_kind="overlap",
            outbox_writer=outbox_writer,
        )

    if not src.exists():
        return _fail(
            error=f"data_root does not exist: {src}",
            error_kind="src_missing",
            outbox_writer=outbox_writer,
        )

    backups_root.mkdir(parents=True, exist_ok=True)

    now = now_fn()
    snapshot_ts = now.strftime(_TS_FORMAT)
    snapshot_dir = backups_root / snapshot_ts

    # If a directory with this name already exists, treat as a partial from
    # a crashed prior run and wipe it (E3).
    if snapshot_dir.exists():
        shutil.rmtree(snapshot_dir, ignore_errors=True)

    # Resolve previous successful snapshot for --link-dest (H2).
    prev = _latest_snapshot(backups_root, exclude=snapshot_dir.name)
    link_dest = prev if prev is not None else None

    snapshot_dir.mkdir(parents=True, exist_ok=False)

    try:
        outcome = run_rsync(src, snapshot_dir, link_dest)
    except FileNotFoundError as e:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        return _fail(
            error=f"rsync binary missing: {e}",
            error_kind="rsync_missing",
            outbox_writer=outbox_writer,
        )
    except Exception as e:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        return _fail(
            error=f"rsync seam crashed: {type(e).__name__}: {e}",
            error_kind="rsync_crash",
            outbox_writer=outbox_writer,
        )

    if outcome.returncode != 0:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        return _fail(
            error=f"rsync exit {outcome.returncode}: {outcome.stderr_tail[:200]}",
            error_kind="rsync_nonzero",
            outbox_writer=outbox_writer,
        )

    # Write per-snapshot meta.
    try:
        meta = {
            "snapshot_ts": snapshot_ts,
            "snapshot_ts_iso": now.isoformat(),
            "src": str(src),
            "rsync_seconds": round(outcome.seconds, 3),
            "rsync_files_total": outcome.files_total,
            "rsync_files_transferred": outcome.files_transferred,
            "size_bytes": outcome.size_bytes,
            "link_dest_from": prev.name if prev is not None else None,
        }
        (snapshot_dir / "backup_meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as e:
        # Partial succeeded but we can't write meta — bail and clean up so
        # callers don't see a half-baked snapshot in `latest`.
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        return _fail(
            error=f"meta write failed: {e}",
            error_kind="meta_write_failed",
            outbox_writer=outbox_writer,
        )

    # Atomically update `latest` symlink + `last_backup.txt`.
    try:
        _atomic_symlink(backups_root / "latest", snapshot_dir.name)
        (backups_root / "last_backup.txt").write_text(
            now.isoformat(), encoding="utf-8",
        )
    except OSError as e:
        # The snapshot itself is fine — just couldn't update pointers.
        # Don't wipe the snapshot; next run will pick it up via `_latest_snapshot`.
        return _fail(
            error=f"pointer update failed: {e}",
            error_kind="pointer_update_failed",
            snapshot_ts=snapshot_ts,
            snapshot_dir=snapshot_dir,
            outbox_writer=outbox_writer,
        )

    return SnapshotResult(
        ok=True,
        snapshot_ts=snapshot_ts,
        snapshot_dir=snapshot_dir,
        rsync_seconds=outcome.seconds,
        size_bytes=outcome.size_bytes,
    )


def read_backup_state(
    backups_root: Path,
    *,
    now_fn: NowFn = _default_now,
) -> BackupState:
    """What the panel renders. Tolerant of partial/missing state."""
    backups_root = backups_root.expanduser()
    if not backups_root.exists():
        return BackupState(
            last_backup_ts=None, last_backup_age_s=None,
            snapshot_count=0, total_size_bytes=0,
            health="down", backups_root=str(backups_root),
        )

    last_txt = backups_root / "last_backup.txt"
    last_ts_iso: str | None = None
    last_age_s: int | None = None
    if last_txt.exists():
        try:
            last_ts_iso = last_txt.read_text(encoding="utf-8").strip()
            last_dt = datetime.fromisoformat(last_ts_iso)
            last_age_s = int((now_fn() - last_dt).total_seconds())
        except (OSError, ValueError):
            last_ts_iso = None
            last_age_s = None

    snapshots = _list_snapshot_dirs(backups_root)
    snap_summaries: list[dict[str, Any]] = []
    total_size = 0
    for snap in snapshots:
        meta_path = snap / "backup_meta.json"
        size = 0
        meta_ok = False
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                size = int(meta.get("size_bytes") or 0)
                meta_ok = True
            except (OSError, json.JSONDecodeError, ValueError):
                pass
        total_size += size
        snap_summaries.append({
            "ts": snap.name,
            "size_bytes": size,
            "meta_ok": meta_ok,
        })

    health = _health_from_age(last_age_s, has_snapshots=bool(snap_summaries))

    return BackupState(
        last_backup_ts=last_ts_iso,
        last_backup_age_s=last_age_s,
        snapshot_count=len(snap_summaries),
        total_size_bytes=total_size,
        health=health,
        backups_root=str(backups_root),
        snapshots=snap_summaries,
    )


# ─── helpers ───────────────────────────────────────────────────────────────

def _fail(
    *,
    error: str,
    error_kind: str,
    outbox_writer: Callable[[dict[str, Any]], None] | None = None,
    snapshot_ts: str | None = None,
    snapshot_dir: Path | None = None,
) -> SnapshotResult:
    if outbox_writer is not None:
        try:
            outbox_writer({
                "event_type": "backup_failed",
                "level": "error",
                "title": f"backup_failed:{error_kind}",
                "detail": error[:512],
                "ts": time.time(),
            })
        except Exception:
            pass
    return SnapshotResult(
        ok=False,
        snapshot_ts=snapshot_ts,
        snapshot_dir=snapshot_dir,
        error=error,
        error_kind=error_kind,
    )


def _overlaps(a: Path, b: Path) -> bool:
    """True if either is a parent of the other (or they're equal)."""
    a = a.resolve()
    b = b.resolve()
    if a == b:
        return True
    try:
        a.relative_to(b)
        return True
    except ValueError:
        pass
    try:
        b.relative_to(a)
        return True
    except ValueError:
        pass
    return False


def _list_snapshot_dirs(backups_root: Path) -> list[Path]:
    """All children matching the snapshot timestamp pattern. Sorted ascending."""
    out: list[Path] = []
    if not backups_root.exists():
        return out
    for child in backups_root.iterdir():
        if not child.is_dir():
            continue
        if _is_snapshot_name(child.name):
            out.append(child)
    out.sort(key=lambda p: p.name)
    return out


def _is_snapshot_name(name: str) -> bool:
    # 8 digits + underscore + 6 digits
    if len(name) != 15 or name[8] != "_":
        return False
    return name[:8].isdigit() and name[9:].isdigit()


def _latest_snapshot(backups_root: Path, *, exclude: str | None = None) -> Path | None:
    snaps = [s for s in _list_snapshot_dirs(backups_root) if s.name != exclude]
    return snaps[-1] if snaps else None


def _atomic_symlink(link_path: Path, target_name: str) -> None:
    """Atomic symlink replacement via tmp + rename. target_name is relative."""
    tmp = link_path.with_name(link_path.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(target_name, tmp)
    os.replace(tmp, link_path)


_HEALTH_OK_S = 60 * 60       # ≤ 60 min
_HEALTH_WARN_S = 24 * 60 * 60  # ≤ 24 h


def _health_from_age(age_s: int | None, *, has_snapshots: bool) -> str:
    if age_s is None:
        # No last_backup.txt: down regardless of whether stray dirs exist.
        return "down"
    if not has_snapshots:
        # Pointer says we ran, but no snapshot dirs found → degraded.
        return "down"
    if age_s <= _HEALTH_OK_S:
        return "ok"
    if age_s <= _HEALTH_WARN_S:
        return "warn"
    return "down"
