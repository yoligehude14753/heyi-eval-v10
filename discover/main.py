"""CLI for discover/tracker.

    python -m discover.main once                 # one scan
    python -m discover.main loop --interval 14400 # daemon, 4h period
    python -m discover.main status               # show cursor + counts
    python -m discover.main list --limit 20      # tail candidates
    python -m discover.main enqueue --limit 5    # auto-enqueue newest into orchestrator queue
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .tracker import (
    TrackerConfig,
    append_candidates,
    load_candidates,
    load_cursor,
    save_cursor,
    scan_round,
)


def _default_data_root() -> Path:
    env = os.environ.get("HEYI_EVAL_DATA")
    if env:
        return Path(env)
    return Path("/home/ai/heyi-eval-data")


def _candidates_path(data_root: Path) -> Path:
    return data_root / "discover" / "candidates.jsonl"


def _cursor_path(data_root: Path) -> Path:
    return data_root / "discover" / "cursor.json"


def _whitelist_path() -> Path:
    """Whitelist lives in the repo (versioned) at discover/whitelist.yaml."""
    return Path(__file__).resolve().parent / "whitelist.yaml"


def _make_api(endpoint: str, token: str | None = None):
    """PR#40: forward HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) so list_models
    can hit gated repos and benefit from higher per-account rate limits
    on hf-mirror. ``token=None`` is the anonymous default and matches
    the pre-PR#40 behaviour."""
    from huggingface_hub import HfApi  # type: ignore[import-not-found]
    if token:
        return HfApi(endpoint=endpoint, token=token)
    return HfApi(endpoint=endpoint)


def _resolve_hf_token() -> str | None:
    """Read HF token from the standard env var pair, returning ``None``
    when neither is set or both are empty. Centralised here so every
    discover.* entry point uses the same resolution policy."""
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    ) or None


def cmd_once(args: argparse.Namespace) -> int:
    data_root = _default_data_root()
    config = TrackerConfig.from_yaml(args.whitelist or _whitelist_path())
    cursor = load_cursor(_cursor_path(data_root))
    api = _make_api(args.hf_endpoint, token=_resolve_hf_token())

    new, stats = scan_round(api, config, cursor)
    n_written = append_candidates(_candidates_path(data_root), new)
    save_cursor(_cursor_path(data_root), cursor)

    print(
        f"[discover.once] new={n_written} seen_skipped={stats.seen_skipped} "
        f"excluded_old={stats.excluded_old} excluded_modality={stats.excluded_modality} "
        f"excluded_trending_threshold={stats.excluded_trending_threshold} "
        f"api_errors={stats.api_errors}"
    )
    if new:
        sample = new[: min(5, len(new))]
        print("[discover.once] sample new candidates:")
        for c in sample:
            print(
                f"  + {c.hf_id}  reason={c.reason}  pipe={c.pipeline_tag} "
                f"dl={c.downloads} likes={c.likes} mod={c.last_modified}"
            )
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    print(f"[discover.loop] interval={args.interval}s, ctrl-c to stop", file=sys.stderr)
    while True:
        try:
            rc = cmd_once(args)
            if rc != 0:
                print(f"[discover.loop] once returned rc={rc}", file=sys.stderr)
        except KeyboardInterrupt:
            return 130
        except Exception as e:
            print(f"[discover.loop] UNHANDLED: {type(e).__name__}: {e}", file=sys.stderr)
        time.sleep(args.interval)


def cmd_status(args: argparse.Namespace) -> int:
    data_root = _default_data_root()
    cursor = load_cursor(_cursor_path(data_root))
    cands = load_candidates(_candidates_path(data_root))
    print(f"data_root         : {data_root}")
    print(f"candidates total  : {len(cands)}")
    print(f"cursor.last_run_ts: {cursor.last_run_ts or '(never)'}")
    print(f"cursor.seen size  : {len(cursor.seen)}")

    by_reason: dict[str, int] = {}
    by_modality: dict[str, int] = {}
    for c in cands:
        by_reason[c.reason] = by_reason.get(c.reason, 0) + 1
        if c.pipeline_tag:
            by_modality[c.pipeline_tag] = by_modality.get(c.pipeline_tag, 0) + 1
    print(f"by_reason         : {by_reason}")
    print(f"by_modality (top 6): {sorted(by_modality.items(), key=lambda x: -x[1])[:6]}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    data_root = _default_data_root()
    cands = load_candidates(_candidates_path(data_root))
    cands = sorted(cands, key=lambda c: c.discovered_at, reverse=True)
    limit = args.limit
    print(f"{'hf_id':<50} {'reason':<10} {'pipe':<22} {'dl':<10} {'likes':<6} {'discovered_at'}")
    for c in cands[:limit]:
        print(
            f"{c.hf_id:<50} {c.reason:<10} {(c.pipeline_tag or '-'):<22} "
            f"{(c.downloads or 0):<10} {(c.likes or 0):<6} {c.discovered_at}"
        )
    return 0


_SUPPORTED_PIPELINE_TAGS = {
    "text-generation",
    "text2text-generation",
    "image-text-to-text",
    "any-to-any",
    "automatic-speech-recognition",
    "text-to-speech",
    # Image / video diffusion supported via transformers runner; enable
    # later when we ship that engine.
    # "text-to-image", "text-to-video",
}

# Library names that indicate the repo is a convenience bundle of unrelated
# models or a non-LLM ecosystem (ComfyUI nodes, diffusion-single-file packs,
# raw .safetensors weights without a tokenizer, etc.). PR#34a: discovered
# during the WanVideo_comfy incident where one repo dragged 142 GB of
# unrelated text-to-video model variants into the cache.
_REJECTED_LIBRARIES = {
    "diffusion-single-file",
    "ComfyUI",
    "comfyui",
    "diffusers-single-file",
}


def _enqueue_policy_passes(c, args) -> tuple[bool, str]:
    """Return (allow, reason) for a candidate. reason is human-readable."""
    if c.private or c.gated:
        return False, "private/gated"
    if c.pipeline_tag and c.pipeline_tag not in _SUPPORTED_PIPELINE_TAGS:
        return False, f"pipeline_tag={c.pipeline_tag}"
    if c.library_name and c.library_name in _REJECTED_LIBRARIES:
        return False, f"library_name={c.library_name}"
    # need at least *some* recency signal
    if (c.downloads or 0) < args.min_downloads and (c.likes or 0) < args.min_likes:
        return False, f"low signal (dl={c.downloads} likes={c.likes})"
    return True, "ok"


def cmd_enqueue(args: argparse.Namespace) -> int:
    """Pick newest N candidates and enqueue into the orchestrator's queue.

    Skips:
      - private / gated
      - pipeline_tag we don't support yet (e.g. text-to-image until we ship
        the transformers+diffusers runner image)
      - candidates with no recent activity (downloads/likes both below
        threshold)
      - hf_ids already successfully evaluated in last 30 days

    Safe to call from a cron.
    """
    from orchestrator.main import enqueue as enqueue_one
    from orchestrator.store import Store

    data_root = _default_data_root()
    cands = load_candidates(_candidates_path(data_root))
    if not cands:
        print("[discover.enqueue] no candidates yet — run `once` first")
        return 1

    cands = sorted(cands, key=lambda c: c.discovered_at, reverse=True)
    store = Store(data_root)

    enqueued = 0
    rejected = 0
    skipped_recent = 0
    chosen_examined = 0
    rejection_reasons: dict[str, int] = {}

    for c in cands:
        if enqueued >= args.limit:
            break
        chosen_examined += 1
        allow, reason = _enqueue_policy_passes(c, args)
        if not allow:
            rejected += 1
            rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            continue
        try:
            run_id = enqueue_one(store, c.hf_id, skip_if_recent=True)
            if run_id is None:
                skipped_recent += 1
                print(f"  skip {c.hf_id} (already evaluated recently)")
                continue
            print(f"  enqueued {c.hf_id} → {run_id}  ({c.pipeline_tag}, dl={c.downloads})")
            enqueued += 1
        except Exception as e:
            print(f"  FAILED {c.hf_id}: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"[discover.enqueue] examined={chosen_examined} enqueued={enqueued} "
          f"skipped_recent={skipped_recent} rejected={rejected}")
    if rejection_reasons:
        print(f"[discover.enqueue] rejections by reason: {rejection_reasons}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="discover", description="HF model discovery tracker")
    p.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    p.add_argument("--whitelist", type=Path, default=None,
                   help=f"path to whitelist.yaml (default: {_whitelist_path()})")
    sp = p.add_subparsers(dest="cmd")

    p_once = sp.add_parser("once", help="single scan round")
    p_once.set_defaults(func=cmd_once)

    p_loop = sp.add_parser("loop", help="periodic daemon")
    p_loop.add_argument("--interval", type=int, default=14400, help="seconds between scans")
    p_loop.set_defaults(func=cmd_loop)

    p_status = sp.add_parser("status", help="show cursor + candidate counts")
    p_status.set_defaults(func=cmd_status)

    p_list = sp.add_parser("list", help="tail recent candidates")
    p_list.add_argument("--limit", type=int, default=20)
    p_list.set_defaults(func=cmd_list)

    p_enq = sp.add_parser("enqueue",
                          help="auto-enqueue newest N candidates into orchestrator queue")
    p_enq.add_argument("--limit", type=int, default=5,
                       help="max number of new runs to enqueue this call")
    p_enq.add_argument("--min-downloads", type=int, default=1000,
                       help="OR condition vs min-likes — at least one must pass")
    p_enq.add_argument("--min-likes", type=int, default=20)
    p_enq.set_defaults(func=cmd_enqueue)

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
