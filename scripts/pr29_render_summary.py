#!/usr/bin/env python3
"""Pretty-render the PR#29 summary JSON for operator review."""

from __future__ import annotations

import json
import sys


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x:.3f}"


def fmt_num(x: float | None, decimals: int = 1) -> str:
    if x is None:
        return "-"
    return f"{x:.{decimals}f}"


def render(summary: dict) -> None:
    runs = summary.get("runs") or []
    print(f"=== PR#29 Cross-Modality Summary ({len(runs)} runs) ===")
    for r in runs:
        print()
        hf = r.get("hf_id")
        print(f"-- {hf}")
        pt = r.get("pipeline_tag") or "-"
        eng = r.get("engine")
        tp = r.get("tp_size")
        ovs = r.get("oversize")
        print(f"   pipeline_tag={pt!r}  engine={eng}  tp={tp}  oversize={ovs}")

        base = r.get("deploy_base_url")
        if base:
            cont = r.get("deploy_container")
            ready_s = r.get("ready_elapsed_s")
            probes = r.get("ready_probe_count")
            print(f"   deploy: {cont} @ {base}  ready_s={ready_s}  probes={probes}")

        score = r.get("cap_overall_score")
        pr = r.get("cap_overall_pass_rate")
        n_appl = len(r.get("cap_applicable") or [])
        n_skip = r.get("cap_skipped_count") or 0
        if score:
            print(f"   capability: {score}  pass_rate={fmt_pct(pr)}  (applicable={n_appl}, skipped={n_skip})")
            for ap in (r.get("cap_applicable") or []):
                print(f"      + {ap.get('name')}: {ap.get('score')}  pass_rate={fmt_pct(ap.get('pass_rate'))}  items={ap.get('items')}")

        ttft = r.get("perf_ttft_p50_ms")
        tps = r.get("perf_tps_single_p50")
        agg = r.get("perf_concurrent_agg_tps")
        if ttft is not None:
            print(f"   perf: ttft_p50={fmt_num(ttft, 1)}ms  tps_single_p50={fmt_num(tps, 1)}  concurrent_agg_tps={fmt_num(agg, 0)}")

        showc = r.get("showcase_items") or 0
        print(f"   showcase items={showc}")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "-"
    if path == "-":
        data = json.load(sys.stdin)
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    render(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
