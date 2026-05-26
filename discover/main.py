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
    mirror_paginate_models,
    save_cursor,
    scan_backfill,
    scan_curated,
    scan_incremental,
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
    """One discover pass.

    PR#53: default mode is ``curated`` (= whitelist vendors + a few
    daily-hot models, deduped, the user's stated scope: "基模厂商发的 +
    每天热门的几个模型"). The PR#49 ``auto``/``backfill``/``incremental``
    paths are still selectable for backfill operators, but no longer
    the default — backfilling 557k 2026 LoRA spam repos was the wrong
    architecture.

    ``--reset`` purges discover/candidates.jsonl and discover/cursor.json
    before running so a single command can recover from a polluted
    candidates file (e.g. after a runaway backfill). Historical runs/
    are preserved.
    """
    data_root = _default_data_root()
    config = TrackerConfig.from_yaml(args.whitelist or _whitelist_path())

    if getattr(args, "reset", False):
        cand_path = _candidates_path(data_root)
        cur_path = _cursor_path(data_root)
        cand_n = (
            sum(1 for _ in cand_path.open("r", encoding="utf-8"))
            if cand_path.exists() else 0
        )
        if cand_path.exists():
            cand_path.unlink()
        if cur_path.exists():
            cur_path.unlink()
        print(f"[discover.reset] purged candidates.jsonl ({cand_n} rows) "
              f"and cursor.json — starting from scratch")

    cursor = load_cursor(_cursor_path(data_root))
    api = _make_api(args.hf_endpoint, token=_resolve_hf_token())

    mode = getattr(args, "mode", "curated") or "curated"

    if mode in ("curated", "legacy"):
        new, stats = scan_curated(api, config, cursor)
        kind = "curated"
    elif mode == "backfill" or (mode == "auto" and not cursor.backfill_complete):
        # PR#52: cursor-aware mirror paginator drains the entire 2026
        # cohort in one round (~30min) by following Link-header `next`
        # URLs rewritten to point back to the mirror (huggingface.co
        # itself is unreachable from NV8 / most CN networks).
        paginator = mirror_paginate_models(
            endpoint=args.hf_endpoint,
            token=_resolve_hf_token(),
            sort="lastModified",
            direction=-1,
            limit_per_page=getattr(args, "backfill_page_size", 1000),
            stop_when_older_than=config.from_date,
            max_pages=getattr(args, "backfill_max_pages", 2000),
            page_sleep_s=getattr(args, "backfill_page_sleep_s", 0.2),
        )
        new, stats, _finished = scan_backfill(paginator, config, cursor)
        kind = "backfill"
    else:
        # Incremental: 1-3 pages of newest models is plenty for a daily
        # delta; cap pages low and stop at cursor.last_run_ts.
        paginator = mirror_paginate_models(
            endpoint=args.hf_endpoint,
            token=_resolve_hf_token(),
            sort="lastModified",
            direction=-1,
            limit_per_page=1000,
            stop_when_older_than=cursor.last_run_ts or config.from_date,
            max_pages=10,
            page_sleep_s=0.2,
        )
        new, stats = scan_incremental(paginator, config, cursor)
        kind = "incremental"

    n_written = append_candidates(_candidates_path(data_root), new)
    save_cursor(_cursor_path(data_root), cursor)

    extra = ""
    if kind == "backfill":
        extra = (
            f" backfill_complete={cursor.backfill_complete} "
            f"high_water={cursor.backfill_high_water[:19] or '-'}"
        )
    print(
        f"[discover.{kind}] new={n_written} seen_skipped={stats.seen_skipped} "
        f"excluded_old={stats.excluded_old} excluded_modality={stats.excluded_modality} "
        f"excluded_trending_threshold={stats.excluded_trending_threshold} "
        f"excluded_oversize={stats.excluded_oversize} "
        f"api_errors={stats.api_errors}{extra}"
    )
    if new:
        sample = new[: min(5, len(new))]
        print(f"[discover.{kind}] sample new candidates:")
        for c in sample:
            pb = "-" if c.param_billion is None else f"{c.param_billion:.1f}B"
            print(
                f"  + {c.hf_id}  reason={c.reason}  pipe={c.pipeline_tag} "
                f"dl={c.downloads} likes={c.likes} size={pb} mod={c.last_modified}"
            )
    return 0


def cmd_loop(args: argparse.Namespace) -> int:
    """Periodic discover daemon.

    PR#53: ``--mode curated`` (the new default) loops once per
    ``--interval`` seconds (default 86400 = daily). ``--reset`` only
    fires on the very first iteration so a restart never wipes a
    healthy candidates file.

    PR#49 fallback: ``--mode auto`` keeps the backfill-then-incremental
    behavior — backfill_interval applies while cursor.backfill_complete
    is False, then it switches to ``--interval``.
    """
    backfill_interval = getattr(args, "backfill_interval", 300)
    incr_interval = args.interval
    mode = getattr(args, "mode", "curated") or "curated"
    print(
        f"[discover.loop] mode={mode} backfill_interval={backfill_interval}s "
        f"incremental_interval={incr_interval}s, ctrl-c to stop",
        file=sys.stderr,
    )
    first_iteration = True
    while True:
        try:
            rc = cmd_once(args)
            if rc != 0:
                print(f"[discover.loop] once returned rc={rc}", file=sys.stderr)
        except KeyboardInterrupt:
            return 130
        except Exception as e:
            print(f"[discover.loop] UNHANDLED: {type(e).__name__}: {e}", file=sys.stderr)
        # Reset only on the very first round; subsequent rounds must
        # never wipe the candidates file we just populated.
        if first_iteration and getattr(args, "reset", False):
            args.reset = False
        first_iteration = False
        # Decide sleep:
        #  - curated / legacy: always incremental interval (daily).
        #  - auto: backfill interval while backfill not yet complete.
        if mode in ("curated", "legacy", "incremental"):
            sleep_s = incr_interval
        else:
            data_root = _default_data_root()
            cursor = load_cursor(_cursor_path(data_root))
            sleep_s = incr_interval if cursor.backfill_complete else backfill_interval
        time.sleep(sleep_s)


def cmd_status(args: argparse.Namespace) -> int:
    data_root = _default_data_root()
    cursor = load_cursor(_cursor_path(data_root))
    cands = load_candidates(_candidates_path(data_root))
    print(f"data_root              : {data_root}")
    print(f"candidates total       : {len(cands)}")
    print(f"cursor.last_run_ts     : {cursor.last_run_ts or '(never)'}")
    print(f"cursor.seen size       : {len(cursor.seen)}")
    print(f"cursor.backfill_complete: {cursor.backfill_complete}")
    print(f"cursor.backfill_high_water: {cursor.backfill_high_water or '(none)'}")

    by_reason: dict[str, int] = {}
    by_modality: dict[str, int] = {}
    for c in cands:
        by_reason[c.reason] = by_reason.get(c.reason, 0) + 1
        if c.pipeline_tag:
            by_modality[c.pipeline_tag] = by_modality.get(c.pipeline_tag, 0) + 1
    print(f"by_reason              : {by_reason}")
    print(f"by_modality (top 6)    : {sorted(by_modality.items(), key=lambda x: -x[1])[:6]}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    data_root = _default_data_root()
    cands = load_candidates(_candidates_path(data_root))
    cands = sorted(cands, key=lambda c: c.discovered_at, reverse=True)
    limit = args.limit
    # PR#68: surface ``param_billion`` so operators can sanity-check the
    # size-gate output (no more "wait, why is Kimi-K2.6 in the queue?").
    print(
        f"{'hf_id':<50} {'reason':<10} {'pipe':<22} "
        f"{'dl':<10} {'likes':<6} {'B':<6} {'discovered_at'}"
    )
    for c in cands[:limit]:
        pb = "-" if c.param_billion is None else f"{c.param_billion:.1f}"
        print(
            f"{c.hf_id:<50} {c.reason:<10} {(c.pipeline_tag or '-'):<22} "
            f"{(c.downloads or 0):<10} {(c.likes or 0):<6} {pb:<6} "
            f"{c.discovered_at}"
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


_TRUSTED_REASONS = {"whitelist", "both", "manual"}


def _enqueue_policy_passes(c, args) -> tuple[bool, str]:
    """Return (allow, reason_zh) for a candidate. reason_zh is the
    human-readable Chinese reason shown in the enqueue audit log.

    PR#56: whitelist/both/manual candidates bypass the low-signal
    threshold. The old behavior was rejecting fresh vendor checkpoints
    (e.g. Qwen's research SAE-Res-* repos right after release, when
    downloads are still in the dozens) — exactly the models we WANT
    to evaluate first. We already curated the vendor list explicitly,
    so trust their releases regardless of HF velocity counters.
    """
    if c.private or c.gated:
        return False, "私有/受限仓库"
    if c.pipeline_tag and c.pipeline_tag not in _SUPPORTED_PIPELINE_TAGS:
        return False, f"暂不支持的 pipeline_tag={c.pipeline_tag}"
    if c.library_name and c.library_name in _REJECTED_LIBRARIES:
        return False, f"不可运行的 library_name={c.library_name}"
    if c.reason in _TRUSTED_REASONS:
        return True, "ok（白名单/手动，跳过 dl/likes 阈值）"
    if (c.downloads or 0) < args.min_downloads and (c.likes or 0) < args.min_likes:
        return False, f"信号过低（dl={c.downloads} likes={c.likes}）"
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

    # PR#56: sort newest-first by last_modified (model freshness) with
    # discovered_at as the tiebreaker. The previous purely-discovered_at
    # ordering meant the daily curated batch all sorted by ~same second,
    # so the model age signal was lost.
    cands = sorted(
        cands,
        key=lambda c: (c.last_modified or "", c.discovered_at or ""),
        reverse=True,
    )
    store = Store(data_root)

    enqueued = 0
    rejected = 0
    skipped_recent = 0
    chosen_examined = 0
    rejection_reasons: dict[str, int] = {}

    # PR#56: bucket the bulk "信号过低" rejections into a single key so
    # the systemd journal isn't 50KB per round; preserve per-pipeline_tag
    # and per-library rejection breakdowns since those are diagnostic.
    def _bucket_reason(r: str) -> str:
        return "信号过低" if r.startswith("信号过低") else r

    for c in cands:
        if enqueued >= args.limit:
            break
        chosen_examined += 1
        allow, reason = _enqueue_policy_passes(c, args)
        if not allow:
            rejected += 1
            key = _bucket_reason(reason)
            rejection_reasons[key] = rejection_reasons.get(key, 0) + 1
            continue
        try:
            run_id = enqueue_one(store, c.hf_id, skip_if_recent=True)
            if run_id is None:
                skipped_recent += 1
                print(f"  跳过 {c.hf_id}（最近已评测过）")
                continue
            print(f"  已入队 {c.hf_id} → {run_id} "
                  f"(reason={c.reason} pipe={c.pipeline_tag} "
                  f"dl={c.downloads} mod={c.last_modified})")
            enqueued += 1
        except Exception as e:
            print(f"  失败 {c.hf_id}: {type(e).__name__}: {e}", file=sys.stderr)

    print(f"[discover.enqueue] 检查={chosen_examined} 入队={enqueued} "
          f"近期已评测跳过={skipped_recent} 拒绝={rejected}")
    if rejection_reasons:
        top = sorted(rejection_reasons.items(), key=lambda kv: -kv[1])[:8]
        print("[discover.enqueue] 拒绝原因 Top:")
        for k, v in top:
            print(f"    {v:>5}  {k}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="discover", description="HF model discovery tracker")
    p.add_argument("--hf-endpoint", default=os.environ.get("HF_ENDPOINT", "https://hf-mirror.com"))
    p.add_argument("--whitelist", type=Path, default=None,
                   help=f"path to whitelist.yaml (default: {_whitelist_path()})")
    sp = p.add_subparsers(dest="cmd")

    p_once = sp.add_parser("once", help="single scan round")
    p_once.add_argument(
        "--mode",
        choices=["curated", "auto", "backfill", "incremental", "legacy"],
        default="curated",
        help=("PR#53 default: curated = whitelist vendors + daily "
              "trending top-N (deduped). PR#49 modes: auto = cursor-driven; "
              "backfill = drain all 2026 history (566k+ rows); incremental "
              "= delta since last_run_ts. ``legacy`` is an alias for "
              "``curated`` (kept for back-compat scripts)."),
    )
    p_once.add_argument(
        "--reset", action="store_true",
        help="purge discover/candidates.jsonl + discover/cursor.json before "
             "scanning (one-shot recovery from a polluted candidates file). "
             "Historical runs/ are preserved.",
    )
    p_once.add_argument(
        "--backfill-page-size", type=int, default=1000,
        help="models per cursor page (HF mirror caps at ~1000)",
    )
    p_once.add_argument(
        "--backfill-max-pages", type=int, default=2000,
        help="safety cap to prevent runaway cursor pagination",
    )
    p_once.add_argument(
        "--backfill-page-sleep-s", type=float, default=0.2,
        help="politeness delay between mirror pages",
    )
    p_once.set_defaults(func=cmd_once)

    p_loop = sp.add_parser("loop", help="periodic daemon")
    p_loop.add_argument(
        "--interval", type=int, default=86400,
        help="seconds between scans once backfill is complete (default 24h)",
    )
    p_loop.add_argument(
        "--backfill-interval", type=int, default=300,
        help="seconds between scans while backfilling (default 5min)",
    )
    p_loop.add_argument(
        "--mode",
        choices=["curated", "auto", "backfill", "incremental", "legacy"],
        default="curated",
        help="see `once --mode`",
    )
    p_loop.add_argument(
        "--backfill-page-size", type=int, default=1000,
        help="models per cursor page (HF mirror caps at ~1000)",
    )
    p_loop.add_argument(
        "--backfill-max-pages", type=int, default=2000,
        help="safety cap to prevent runaway cursor pagination",
    )
    p_loop.add_argument(
        "--backfill-page-sleep-s", type=float, default=0.2,
        help="politeness delay between mirror pages",
    )
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
    # PR#56: relaxed thresholds — whitelist candidates bypass these
    # anyway; the threshold only kicks in for trending candidates that
    # aren't in any vendor whitelist, where 200 downloads OR 5 likes
    # filters out the bulk of low-effort uploads without rejecting
    # legitimate new releases.
    p_enq.add_argument("--min-downloads", type=int, default=200,
                       help="OR condition vs min-likes — at least one must pass")
    p_enq.add_argument("--min-likes", type=int, default=5)
    p_enq.set_defaults(func=cmd_enqueue)

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
