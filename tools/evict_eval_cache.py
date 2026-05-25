#!/usr/bin/env python3
"""PR#65 CLI: inspect / evict the eval-cache.

Usage:

    # Show full classification (no deletes). Default.
    python3 tools/evict_eval_cache.py

    # Show what would be deleted to satisfy a 200 GB quota.
    python3 tools/evict_eval_cache.py --quota-gb 200

    # Actually delete to meet the quota.
    python3 tools/evict_eval_cache.py --quota-gb 200 --apply

    # Drop everything that's already 'safe' (= eval ok) regardless of
    # quota. Frees the maximum amount of space safely.
    python3 tools/evict_eval_cache.py --drop-all-safe --apply

Defaults (override with --eval-cache / --runs-root or env vars):
    --eval-cache : HEYI_EVAL_MODEL_CACHE or /DATA/Model/_eval-cache
    --runs-root  : HEYI_EVAL_DATA_ROOT/runs or /home/ai/heyi-eval-data/runs
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Allow running from anywhere — add repo root to path.
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from orchestrator.cache_evictor import (  # noqa: E402
    DEFAULT_GRACE_S,
    DEFAULT_QUOTA_BYTES,
    enforce_quota,
    find_evictable,
    humansize,
)


def _print_plan(plan, *, verbose: bool = False) -> None:
    print(f"Eval cache total: {humansize(plan.total_bytes)} "
          f"({plan.total_bytes / 1e9:.1f} GB) "
          f"in {len(plan.entries)} dirs")
    print(f"Quota:            {humansize(plan.quota_bytes)} "
          f"({plan.quota_bytes / 1e9:.1f} GB)")
    if plan.total_bytes > plan.quota_bytes:
        over = plan.total_bytes - plan.quota_bytes
        print(f"Over quota by:    {humansize(over)} ({over / 1e9:.1f} GB)")
    else:
        print("Under quota.")

    by_status = {}
    for e in plan.entries:
        by_status.setdefault(e.status, []).append(e)
    print()
    print("By status:")
    for st in ("in_progress", "safe", "failed_only", "orphan"):
        items = by_status.get(st, [])
        if not items:
            continue
        total = sum(e.size_bytes for e in items)
        print(f"  {st:12s}: {len(items):3d} dirs, "
              f"{total / 1e9:7.1f} GB")

    if verbose:
        print()
        print(f"{'STATUS':<12} {'SIZE':>10} {'MTIME':<20} {'NAME':<40} HF_IDS")
        for e in sorted(plan.entries, key=lambda e: -e.size_bytes):
            from datetime import datetime
            mt = (datetime.fromtimestamp(e.mtime).strftime("%Y-%m-%d %H:%M")
                  if e.mtime else "-")
            hfs = ",".join(e.hf_ids[:2])
            if len(e.hf_ids) > 2:
                hfs += f" (+{len(e.hf_ids) - 2})"
            print(f"  {e.status:<12} {humansize(e.size_bytes):>10} "
                  f"{mt:<20} {e.name[:40]:<40} {hfs}")


def _drop_all_safe(eval_cache: Path, runs_root: Path, *, apply: bool) -> None:
    """Aggressive sweep mode: delete every 'safe' entry regardless of
    quota. Used for the one-time post-incident cleanup."""
    plan = find_evictable(eval_cache, runs_root,
                          quota_bytes=DEFAULT_QUOTA_BYTES,
                          grace_s=DEFAULT_GRACE_S)
    safe = [e for e in plan.entries if e.status == "safe"]
    safe_orphan = [e for e in plan.entries if e.status in ("safe", "orphan")]
    total_safe = sum(e.size_bytes for e in safe_orphan)
    print(f"Would drop {len(safe_orphan)} dirs "
          f"({len(safe)} safe + {len(safe_orphan) - len(safe)} orphan), "
          f"freeing {total_safe / 1e9:.1f} GB")
    if not apply:
        for e in sorted(safe_orphan, key=lambda x: -x.size_bytes):
            print(f"  - {e.status:<6} {e.name} ({e.size_bytes / 1e9:.1f} GB)")
        return
    freed = 0
    failed = 0
    for e in safe_orphan:
        try:
            shutil.rmtree(e.path)
            freed += e.size_bytes
            print(f"  ✓ {e.name} (-{e.size_bytes / 1e9:.1f} GB)")
        except OSError as exc:
            failed += 1
            print(f"  ✗ {e.name}: {exc}", file=sys.stderr)
    print(f"\nFreed: {freed / 1e9:.1f} GB; failed: {failed}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-cache",
                   default=os.environ.get("HEYI_EVAL_MODEL_CACHE",
                                          "/DATA/Model/_eval-cache"),
                   help="Path to eval-cache root.")
    p.add_argument("--runs-root",
                   default=os.environ.get("HEYI_EVAL_RUNS_ROOT",
                                          "/home/ai/heyi-eval-data/runs"),
                   help="Path to data/runs directory.")
    p.add_argument("--quota-gb", type=float, default=200.0,
                   help="Cache quota in GB (default 200).")
    p.add_argument("--grace-min", type=float, default=DEFAULT_GRACE_S / 60,
                   help="Grace window in minutes: never touch dirs "
                        "with mtime in this window (default 30).")
    p.add_argument("--apply", action="store_true",
                   help="Actually delete. Default is dry-run.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-dir table.")
    p.add_argument("--drop-all-safe", action="store_true",
                   help="Bypass quota: drop ALL entries with "
                        "status=safe|orphan. Use for emergency cleanup.")
    args = p.parse_args()

    eval_cache = Path(args.eval_cache)
    runs_root = Path(args.runs_root)

    if not eval_cache.is_dir():
        print(f"error: --eval-cache {eval_cache} is not a directory",
              file=sys.stderr)
        return 2

    if args.drop_all_safe:
        _drop_all_safe(eval_cache, runs_root, apply=args.apply)
        return 0

    quota = int(args.quota_gb * 1024 * 1024 * 1024)
    grace = args.grace_min * 60

    if args.apply:
        plan = enforce_quota(eval_cache, runs_root,
                             quota_bytes=quota, grace_s=grace,
                             dry_run=False)
    else:
        plan = enforce_quota(eval_cache, runs_root,
                             quota_bytes=quota, grace_s=grace,
                             dry_run=True)

    _print_plan(plan, verbose=args.verbose)

    if plan.evicted:
        print()
        verb = "Evicted" if args.apply else "Would evict"
        print(f"{verb} {len(plan.evicted)} dirs "
              f"({plan.bytes_freed / 1e9:.1f} GB):")
        for e in plan.evicted:
            print(f"  - {e.status:<12} {e.name} "
                  f"({e.size_bytes / 1e9:.1f} GB)")
    if plan.skipped:
        print(f"\nSkipped {len(plan.skipped)} (errors):")
        for s in plan.skipped:
            print(f"  - {s['name']}: {s['error']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
