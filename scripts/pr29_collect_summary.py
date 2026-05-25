#!/usr/bin/env python3
"""PR#29 collector: walk heyi-eval-data/runs and emit a single
machine-readable summary for the PR#29 cross-modality report."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RUN_IDS = [
    # PR#26 batch
    "r-20260523T030158Z-qwen_qwen2.5-0.5b-instruct-767d",
    "r-20260523T030159Z-deepseek-ai_deepseek-v3-4db7",
    "r-20260523T030806Z-meta-llama_llama-3.1-405b-inst-b203",
    # PR#27 fixes
    "r-20260523T064003Z-openai_whisper-tiny-ef9a",
    "r-20260523T071148Z-qwen_qwen2.5-0.5b-instruct-5bb9",
]


def _load(p: Path) -> dict | None:
    try:
        return json.loads(p.read_text("utf-8"))
    except Exception:
        return None


def collect(runs_root: Path, run_id: str) -> dict:
    rd = runs_root / run_id
    md = _load(rd / "_meta" / "curated.json") or {}
    mdt = _load(rd / "_meta" / "metadata.json") or {}
    en = _load(rd / "_meta" / "engine.json") or _load(rd / "engine.json") or {}
    dp = _load(rd / "deploy.json") or {}
    rdy = _load(rd / "ready.json") or {}
    ca = _load(rd / "capability.json") or {}
    pf = _load(rd / "perf_bench.json") or {}
    sc = _load(rd / "showcase.json") or {}

    pt = (mdt.get("hf_info") or {}).get("pipeline_tag") or ""
    cats = ca.get("categories") or {}
    applicable = []
    for n, c in cats.items():
        if c.get("applicable"):
            applicable.append({
                "name": n,
                "score": c.get("score"),
                "pass_rate": c.get("pass_rate"),
                "items": len(c.get("items") or []),
            })
    skipped = [
        {"name": n, "reason": c.get("reason")}
        for n, c in cats.items() if not c.get("applicable")
    ]

    return {
        "run_id": run_id,
        "hf_id": md.get("hf_id"),
        "pipeline_tag": pt,
        "modalities_curated": md.get("modalities"),
        "capability_tags_curated": md.get("capability_tags"),
        "engine": en.get("engine") or "-",
        "oversize": bool(en.get("oversize")),
        "tp_size": (en.get("vllm_args") or {}).get("tensor_parallel_size"),
        "deploy_base_url": dp.get("base_url"),
        "deploy_container": dp.get("container_name"),
        "ready_elapsed_s": rdy.get("elapsed_s"),
        "ready_probe_count": rdy.get("probe_count"),
        "cap_overall_score": ca.get("score"),
        "cap_overall_pass_rate": ca.get("pass_rate"),
        "cap_applicable": applicable,
        "cap_skipped_count": len(skipped),
        "cap_skipped_sample": [s["name"] for s in skipped[:5]],
        "perf_ttft_p50_ms": ((pf.get("ttft_ms") or {}).get("p50")),
        "perf_tps_single_p50": ((pf.get("tps_single") or {}).get("p50")),
        "perf_concurrent_agg_tps": ((pf.get("concurrent") or {}).get("aggregate_tps")),
        "perf_concurrent_n": ((pf.get("concurrent") or {}).get("n")),
        "showcase_items": len(sc.get("items") or []),
        "showcase_highlights": len(sc.get("highlights") or []),
    }


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/heyi-eval-data/runs"))
    out = [collect(root, rid) for rid in RUN_IDS]
    print(json.dumps({"runs": out}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
