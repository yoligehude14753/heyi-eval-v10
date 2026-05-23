#!/usr/bin/env python3
"""PR#32e: walk runs DB + eval-cache, delete cached weights for runs
that failed at STAGE_MODEL (or were stranded by the PR#30 / PR#31
transitions). Idempotent. Reports bytes freed.

Design:

  - Single source of truth: `runs.sqlite::runs.status='failed' AND
    failure_reason LIKE '%stage_model%'` plus
    `failure_reason LIKE '%model not in eval-cache%'` (PR#30-era stub
    of the symptom).
  - For each such row, derive `target_dir = cfg.model_cache_root /
    cfg.hf_local_dir(hf_id)`, check the dir is incomplete (uses
    model_stager._has_weight_file), and rmtree if so.
  - Never delete dirs that pass the completeness check — those are
    healthy snapshots, even if a different run id still appears
    failed in the DB (race / re-enqueue case).
  - Dry-run flag for safety.

Run manually:
    python -m scripts.cleanup_failed_cache --dry-run
    python -m scripts.cleanup_failed_cache       # actually deletes

Or wire via systemd oneshot (heyi-eval-cache-gc.service).
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from orchestrator.config import OrchestratorConfig  # noqa: E402
from orchestrator.model_stager import (  # noqa: E402
    GGUF_PREFERRED_QUANT,
    _cleanup_partial,
    _has_weight_file,
    _looks_like_gguf_repo,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be deleted, don't actually delete")
    ap.add_argument("--data-root",
                    default=os.environ.get("HEYI_EVAL_DATA",
                                           "/home/ai/heyi-eval-data"))
    ap.add_argument("--cache-root",
                    default=os.environ.get("HEYI_EVAL_MODEL_CACHE",
                                           "/DATA/Model/_eval-cache"))
    ap.add_argument("--narrow-gguf", action="store_true",
                    help=(
                        "for GGUF caches that contain multiple "
                        "quantization variants (e.g. unsloth/Qwen3.6-27B-GGUF "
                        "with 8 quants = 351 GB), delete all variants "
                        f"except {GGUF_PREFERRED_QUANT}. Targets ANY GGUF "
                        "cache, not just failed ones."
                    ))
    args = ap.parse_args()

    cfg = OrchestratorConfig(
        data_root=Path(args.data_root),
        model_cache_root=Path(args.cache_root),
    )

    db_path = Path(args.data_root) / "store" / "runs.sqlite"
    if not db_path.exists():
        print(f"runs.sqlite not found at {db_path}", file=sys.stderr)
        return 2

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()
    rows = cur.execute("""
        SELECT run_id, hf_id, status, failure_reason
        FROM runs
        WHERE status IN ('failed', 'aborted')
          AND (
            failure_reason LIKE '%stage_model%'
            OR failure_reason LIKE '%model not in eval-cache%'
            OR failure_reason LIKE '%snapshot_download%'
            OR failure_reason LIKE '%incomplete_after_download%'
          )
    """).fetchall()
    # PR#32 race guard: any hf_id currently being processed by the
    # orchestrator (in_progress run) MUST NOT have its cache touched.
    # Without this, a cleanup invocation during an active download
    # would rmtree the dir under the orchestrator's feet.
    in_flight_hfids = {
        r[0] for r in cur.execute(
            "SELECT DISTINCT hf_id FROM runs WHERE status IN ('in_progress','queued')"
        ).fetchall()
    }
    conn.close()

    # Group by hf_id since many runs of the same model failed together;
    # we only want to look at each cache dir once.
    by_hfid: dict[str, list[tuple[str, str, str]]] = {}
    for run_id, hf_id, status, reason in rows:
        if hf_id in in_flight_hfids:
            # Skip; will report below.
            continue
        by_hfid.setdefault(hf_id, []).append((run_id, status, reason or ""))

    for hf_id in sorted(in_flight_hfids):
        print(f"[skip-in-flight] {hf_id}  has an in_progress/queued run — "
              f"refusing to touch cache")

    total_freed = 0
    inspected = 0
    deleted = 0
    skipped_healthy = 0
    missing = 0

    for hf_id, runs in sorted(by_hfid.items()):
        target = cfg.model_cache_root / cfg.hf_local_dir(hf_id)
        inspected += 1
        if not target.exists():
            missing += 1
            print(f"[skip-absent] {hf_id}  ({len(runs)} failed runs)  "
                  f"{target} does not exist")
            continue

        if _has_weight_file(target):
            skipped_healthy += 1
            try:
                sz = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
            except OSError:
                sz = 0
            print(f"[skip-healthy] {hf_id}  {target}  "
                  f"size={sz / 1e9:.1f} GB — preserving")
            continue

        # incomplete → clean
        try:
            current_size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        except OSError:
            current_size = 0
        if args.dry_run:
            print(f"[DRY-DELETE] {hf_id}  {target}  "
                  f"would free {current_size / 1e9:.1f} GB  "
                  f"({len(runs)} failed runs)")
            total_freed += current_size
        else:
            freed = _cleanup_partial(target)
            deleted += 1
            total_freed += freed
            print(f"[DELETED] {hf_id}  {target}  "
                  f"freed {freed / 1e9:.1f} GB")

    print()
    print(f"summary: inspected={inspected} healthy={skipped_healthy} "
          f"missing={missing} {'would-delete' if args.dry_run else 'deleted'}={deleted} "
          f"total={total_freed / 1e9:.1f} GB")

    if args.narrow_gguf:
        print()
        print("=== --narrow-gguf: trim multi-quant GGUF caches ===")
        # Pass in-flight hf_ids → narrow-gguf will skip caches whose
        # hf_id is being downloaded right now (race guard).
        in_flight_dir_names = {cfg.hf_local_dir(h) for h in in_flight_hfids}
        gguf_freed = _narrow_gguf_caches(
            cfg.model_cache_root,
            preferred_quant=GGUF_PREFERRED_QUANT,
            dry_run=args.dry_run,
            skip_dir_names=in_flight_dir_names,
        )
        print(f"narrow-gguf {'would-free' if args.dry_run else 'freed'} "
              f"= {gguf_freed / 1e9:.1f} GB")
    return 0


def _narrow_gguf_caches(cache_root: Path,
                        *,
                        preferred_quant: str,
                        dry_run: bool,
                        skip_dir_names: set[str] | None = None) -> int:
    """Find any cache dir whose content is predominantly ``.gguf`` and
    that contains multiple quantization variants. Keep only files
    matching ``*preferred_quant*.gguf`` (plus tokenizer/config/etc.);
    delete the rest. Returns bytes freed.

    Triggered by the PR#32 review: ``unsloth/Qwen3.6-27B-GGUF`` shipped
    18 quantization variants totalling 351 GB; the orchestrator never
    needed more than one. The PR#32 stager now picks ``Q4_K_M`` going
    forward, but existing fat caches need this oneshot to recover the
    335 GB they accidentally consumed.
    """
    if not cache_root.exists():
        return 0
    total_freed = 0
    skip = skip_dir_names or set()
    for entry in sorted(cache_root.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in skip:
            print(f"[gguf-skip-in-flight] {entry.name}: in_progress/queued "
                  f"run touches this cache — refusing to narrow")
            continue
        # Quick GGUF-ish probe: count .gguf files in this top-level dir.
        try:
            ggufs = [f for f in entry.iterdir()
                     if f.is_file() and f.suffix.lower() == ".gguf"]
        except OSError:
            continue
        if len(ggufs) <= 1:
            continue
        # We have multiple .gguf files → multi-quant cache.
        keep = [f for f in ggufs if preferred_quant.lower() in f.name.lower()]
        if not keep:
            # No preferred quant available — leave it alone (we don't
            # want to delete everything and end up with zero weights).
            print(f"[gguf-skip] {entry.name}: {len(ggufs)} variants, "
                  f"none match {preferred_quant} — preserving all")
            continue
        delete = [f for f in ggufs if f not in keep]
        # Also clean up the .cache/huggingface/download/.metadata files
        # belonging to the variants we're deleting.
        ledger = entry / ".cache" / "huggingface" / "download"
        ledger_kills: list[Path] = []
        if ledger.exists():
            for f in delete:
                m = ledger / f"{f.name}.metadata"
                if m.exists():
                    ledger_kills.append(m)

        bytes_to_free = sum(f.stat().st_size for f in delete) + \
                        sum(m.stat().st_size for m in ledger_kills)

        kept_names = ", ".join(f.name for f in keep)
        deleted_names = ", ".join(f.name for f in delete[:3]) + \
                        (f" + {len(delete) - 3} more" if len(delete) > 3 else "")
        if dry_run:
            print(f"[DRY-NARROW] {entry.name}: keep [{kept_names}], "
                  f"delete {len(delete)} variants ({deleted_names}) → "
                  f"would free {bytes_to_free / 1e9:.1f} GB")
            total_freed += bytes_to_free
        else:
            for f in delete + ledger_kills:
                try:
                    f.unlink()
                except OSError:
                    pass
            print(f"[NARROW] {entry.name}: kept [{kept_names}], "
                  f"deleted {len(delete)} variants → "
                  f"freed {bytes_to_free / 1e9:.1f} GB")
            total_freed += bytes_to_free
    return total_freed


if __name__ == "__main__":
    raise SystemExit(main())
