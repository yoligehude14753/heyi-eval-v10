"""CLI for curator.

  # enrich a single model (talks to local CCR; writes to data_root/curated/)
  python -m curator.main enrich Qwen/Qwen2.5-0.5B-Instruct

  # enrich the N newest discovered candidates
  python -m curator.main batch --limit 10

  # show what we have curated so far
  python -m curator.main status

  # show one curated json file
  python -m curator.main show Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .enricher import CuratorConfig, enrich_one, write_curated


def _default_data_root() -> Path:
    env = os.environ.get("HEYI_EVAL_DATA")
    if env:
        return Path(env)
    return Path("/home/ai/heyi-eval-data")


def _curated_dir(data_root: Path) -> Path:
    return data_root / "curated"


def _candidates_path(data_root: Path) -> Path:
    return data_root / "discover" / "candidates.jsonl"


def cmd_enrich(args: argparse.Namespace) -> int:
    data_root = _default_data_root()
    cfg = CuratorConfig.from_env()
    curated = enrich_one(args.hf_id, cfg)
    out = write_curated(_curated_dir(data_root), curated)
    print(f"[curator.enrich] wrote {out}")
    print(f"  publisher          : {curated.get('publisher', {}).get('name')}")
    print(f"  contributors       : {curated.get('contributors')[:3]}")
    print(f"  first_impression   : {curated.get('first_impression_tag')}")
    print(f"  strengths (top 3)  : {curated.get('claimed_strengths', [])[:3]}")
    print(f"  innovations (top 3): {curated.get('innovations', [])[:3]}")
    print(f"  interesting (top 4): {curated.get('interesting_points', [])[:4]}")
    print(f"  modalities         : {curated.get('modalities')}")
    print(f"  context_length     : {curated.get('context_length')}")
    print(f"  param_count        : {curated.get('param_count')}")
    lm = curated.get("_llm_meta", {})
    print(f"  llm_meta           : tokens_in={lm.get('input_tokens')} "
          f"tokens_out={lm.get('output_tokens')} elapsed={lm.get('elapsed_s'):.1f}s "
          f"parse_error={lm.get('parse_error')} fetch_error={lm.get('card_fetch_error')}")
    return 0 if not lm.get("card_fetch_error") and not lm.get("parse_error") else 2


def cmd_batch(args: argparse.Namespace) -> int:
    from discover.tracker import load_candidates  # local import — sibling pkg

    data_root = _default_data_root()
    cands = load_candidates(_candidates_path(data_root))
    if not cands:
        print("[curator.batch] no candidates — run `python -m discover.main once` first")
        return 1

    cands = sorted(cands, key=lambda c: c.discovered_at, reverse=True)
    cfg = CuratorConfig.from_env()
    curated_dir = _curated_dir(data_root)

    n_total = min(args.limit, len(cands))
    n_ok, n_err = 0, 0
    for i, c in enumerate(cands[: n_total], 1):
        safe = c.hf_id.replace("/", "__")
        out_path = curated_dir / f"{safe}.json"
        if out_path.exists() and not args.refresh:
            print(f"  [{i}/{n_total}] {c.hf_id} — already curated, skip")
            continue
        try:
            curated = enrich_one(c.hf_id, cfg)
            write_curated(curated_dir, curated)
            lm = curated.get("_llm_meta", {})
            err = lm.get("parse_error") or lm.get("card_fetch_error")
            if err:
                n_err += 1
                print(f"  [{i}/{n_total}] {c.hf_id} — DEGRADED ({err[:60]})")
            else:
                n_ok += 1
                tag = curated.get("first_impression_tag") or "-"
                print(f"  [{i}/{n_total}] {c.hf_id} → {tag}")
        except Exception as e:
            n_err += 1
            print(f"  [{i}/{n_total}] {c.hf_id} — EXCEPTION {type(e).__name__}: {e}",
                  file=sys.stderr)

    print(f"[curator.batch] ok={n_ok} degraded_or_err={n_err}")
    return 0 if n_err == 0 else 2


def cmd_status(args: argparse.Namespace) -> int:
    data_root = _default_data_root()
    cdir = _curated_dir(data_root)
    if not cdir.exists():
        print(f"no curated/ dir yet at {cdir}")
        return 0
    files = sorted(cdir.glob("*.json"))
    print(f"data_root: {data_root}")
    print(f"curated count: {len(files)}")
    # sample 5 most-recently-modified
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in files[:5]:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            print(f"  {data.get('hf_id'):<50} {data.get('first_impression_tag') or '-':<30} "
                  f"strengths={len(data.get('claimed_strengths') or [])} "
                  f"innovations={len(data.get('innovations') or [])}")
        except Exception:
            print(f"  {p.name} — unreadable")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    data_root = _default_data_root()
    safe = args.hf_id.replace("/", "__")
    p = _curated_dir(data_root) / f"{safe}.json"
    if not p.exists():
        print(f"not found: {p}", file=sys.stderr)
        return 1
    print(p.read_text(encoding="utf-8"))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="curator", description="HF modelcard → metadata enricher (CCR-backed)")
    sp = p.add_subparsers(dest="cmd")

    p_e = sp.add_parser("enrich", help="enrich a single hf_id")
    p_e.add_argument("hf_id")
    p_e.set_defaults(func=cmd_enrich)

    p_b = sp.add_parser("batch", help="enrich newest N candidates")
    p_b.add_argument("--limit", type=int, default=10)
    p_b.add_argument("--refresh", action="store_true",
                     help="re-curate even if curated/ has a file already")
    p_b.set_defaults(func=cmd_batch)

    p_s = sp.add_parser("status", help="count + sample existing curated files")
    p_s.set_defaults(func=cmd_status)

    p_sh = sp.add_parser("show", help="cat one curated json")
    p_sh.add_argument("hf_id")
    p_sh.set_defaults(func=cmd_show)

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
