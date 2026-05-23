"""Inspect capability.json for a recent batch of runs."""
import json
import os
import sys

DATA = "/home/ai/heyi-eval-data/runs"


def main(run_ids: list[str]) -> int:
    for r in run_ids:
        print(f"\n=== {r} ===")
        rd = os.path.join(DATA, r)
        p = os.path.join(rd, "capability.json")
        if not os.path.exists(p):
            print("  no capability.json")
            if os.path.isdir(rd):
                print("  files:", sorted(os.listdir(rd))[:10])
            continue
        d = json.load(open(p))
        score = d.get("score")
        rate = d.get("pass_rate")
        cats = list((d.get("categories") or {}).keys())
        print(f"  score: {score}   pass_rate: {rate}")
        print(f"  cats : {cats}")
        broken = d.get("broken_endpoint")
        if broken:
            print(f"  BROKEN: {broken}")
        for it in (d.get("results") or [])[:4]:
            pid = (it.get("id") or "")[:24]
            ok = it.get("pass")
            err = (it.get("error") or "")[:50]
            act = (it.get("actual") or "")[:70]
            print(f"    {pid:24s} pass={ok} err={err!r} actual={act!r}")
        # tally errors
        errs: dict[str, int] = {}
        for it in d.get("results") or []:
            e = (it.get("error") or "").strip()
            if e:
                key = e[:30]
                errs[key] = errs.get(key, 0) + 1
        if errs:
            top = sorted(errs.items(), key=lambda kv: -kv[1])[:3]
            print(f"  err-sigs: {top}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
