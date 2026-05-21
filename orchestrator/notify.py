"""
Notify outbox: append-only JSONL persisted on nv8.

mac-side sync_agent.py reads this file (via ssh tail / poll-with-offset)
and POSTs to alld :7070/notify, which broadcasts to:
  - zero wechat bot  -> 微信 yoligehude
  - desktop channel   -> macOS notification

We deliberately do NOT speak any IM protocol here. Every notify is a single
JSONL line; if any downstream layer fails, the line stays on disk and the
mac sync_agent will pick it up next round.

Format (one event per line):
  {
    "ts": ISO8601,
    "event_type": "run_started" | "run_completed" | "run_failed" | "incident" | "heartbeat",
    "level": "info" | "warn" | "error",
    "title": "...",
    "body":  "...",
    "channels": ["wechat", "desktop"],
    "priority": "normal" | "P0" | "P1",
    "run_id":   "...",         # optional
    "hf_id":    "...",         # optional
    "stage":    "...",         # optional
    "extras":   { ... }        # arbitrary structured payload
  }
"""
from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_OUTBOX_LOCK = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


def write_event(
    outbox_path: Path,
    *,
    event_type: str,
    title: str,
    body: str,
    level: str = "info",
    channels: list[str] | None = None,
    priority: str = "normal",
    run_id: str | None = None,
    hf_id: str | None = None,
    stage: str | None = None,
    extras: dict[str, Any] | None = None,
) -> None:
    """
    Append one event to the outbox JSONL.

    Atomic with respect to concurrent writers in the same process (file lock + flush).
    Across processes on the same host, we rely on append-only line semantics:
    OS-level write() of a single short line (<4KB) is atomic on POSIX.
    """
    event = {
        "ts": _utc_now_iso(),
        "event_type": event_type,
        "level": level,
        "title": title[:200],
        "body": body[:2000],
        "channels": channels if channels is not None else ["wechat", "desktop"],
        "priority": priority,
    }
    if run_id:
        event["run_id"] = run_id
    if hf_id:
        event["hf_id"] = hf_id
    if stage:
        event["stage"] = stage
    if extras:
        event["extras"] = extras

    line = json.dumps(event, ensure_ascii=False) + "\n"

    with _OUTBOX_LOCK:
        outbox_path.parent.mkdir(parents=True, exist_ok=True)
        # Open with line buffering + sync to disk for durability of P0/P1 events.
        with outbox_path.open("a", encoding="utf-8") as f:
            f.write(line)
            if priority in ("P0", "P1") or level == "error":
                f.flush()
                os.fsync(f.fileno())


# ── Convenience helpers ─────────────────────────────────────────────────────


def run_started(outbox_path: Path, *, run_id: str, hf_id: str, engine: str, eta_s: int | None = None) -> None:
    body = f"hf_id={hf_id}\nengine={engine}"
    if eta_s:
        body += f"\neta≈{eta_s}s"
    write_event(
        outbox_path,
        event_type="run_started",
        title=f"🟢 eval run started · {hf_id}",
        body=body,
        run_id=run_id,
        hf_id=hf_id,
    )


def run_completed(
    outbox_path: Path,
    *,
    run_id: str,
    hf_id: str,
    capability_pass_rate: float | None = None,
    showcase_impression: str | None = None,
    duration_s: float,
) -> None:
    lines = [f"hf_id={hf_id}", f"duration={duration_s:.0f}s"]
    if capability_pass_rate is not None:
        lines.append(f"capability={capability_pass_rate * 100:.0f}%")
    if showcase_impression:
        lines.append(f"first_impression: {showcase_impression}")
    write_event(
        outbox_path,
        event_type="run_completed",
        title=f"✅ eval run done · {hf_id}",
        body="\n".join(lines),
        run_id=run_id,
        hf_id=hf_id,
    )


def run_failed(
    outbox_path: Path,
    *,
    run_id: str,
    hf_id: str,
    stage: str,
    error: str,
) -> None:
    write_event(
        outbox_path,
        event_type="run_failed",
        title=f"❌ eval run FAILED · {hf_id}",
        body=f"stage={stage}\nerror={error[:1000]}",
        level="error",
        priority="P1",
        run_id=run_id,
        hf_id=hf_id,
        stage=stage,
    )


def incident(outbox_path: Path, *, what: str, detail: str, run_id: str | None = None) -> None:
    write_event(
        outbox_path,
        event_type="incident",
        title=f"⚠️ incident · {what}",
        body=detail[:1500],
        level="error",
        priority="P1",
        run_id=run_id,
    )


def heartbeat(
    outbox_path: Path,
    *,
    in_flight: int,
    completed_today: int,
    failed_today: int,
    free_disk_gb: float,
) -> None:
    body = (
        f"in_flight={in_flight}\n"
        f"completed_today={completed_today}\n"
        f"failed_today={failed_today}\n"
        f"free_disk={free_disk_gb:.0f}GB"
    )
    write_event(
        outbox_path,
        event_type="heartbeat",
        title="🫀 heyi-eval heartbeat",
        body=body,
        # heartbeats stay desktop-only to avoid 微信 spam
        channels=["desktop"],
    )
