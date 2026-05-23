"""Dump deploy_repair.json + run status for a given HF id."""
import json
import sqlite3
import sys
from pathlib import Path


def main(hf_id: str) -> int:
    db = "file:/home/ai/heyi-eval-data/store/runs.sqlite?mode=ro"
    c = sqlite3.connect(db, uri=True)
    cols = [r[1] for r in c.execute("PRAGMA table_info(runs)")]
    print("runs columns:", cols)
    row = c.execute(
        "SELECT * FROM runs WHERE hf_id=? ORDER BY created_at DESC LIMIT 1",
        (hf_id,),
    ).fetchone()
    if not row:
        print("no runs for", hf_id)
        return 1
    d = dict(zip(cols, row))
    print("\nlatest run:")
    for k, v in d.items():
        if isinstance(v, str) and len(v) > 80:
            v = v[:80] + "..."
        print(f"  {k:20s} = {v!r}")
    rid = d["run_id"]
    rd = Path(f"/home/ai/heyi-eval-data/runs/{rid}")
    print(f"\n_meta/ files:")
    for p in sorted((rd / "_meta").glob("*.json")):
        print(f"  {p.name}: {p.stat().st_size}B")

    drp = rd / "_meta" / "deploy_repair.json"
    if drp.exists():
        rep = json.loads(drp.read_text())
        print(f"\n=== deploy_repair.json ===")
        print(f"  ok                  : {rep.get('ok')}")
        print(f"  winning_strategy    : {rep.get('winning_strategy')!r}")
        print(f"  failure_class       : {rep.get('failure_class')!r}")
        print(f"  strategies_proposed : {rep.get('strategies_proposed')}")
        attempts = rep.get("attempts", [])
        print(f"\n  attempts (n={len(attempts)}):")
        for i, a in enumerate(attempts):
            strat = (a.get("strategy") or "")
            engine = (a.get("engine") or "")
            ok = a.get("ok")
            ekind = a.get("error_kind") or ""
            dur = a.get("duration_s")
            print(f"    {i+1:>2}. {strat:30s} engine={engine:12s} "
                  f"ok={ok} ekind={ekind:30s} dur={dur}s")
            err = a.get("error") or ""
            if err and not ok:
                print(f"        err: {err[:100]}")
        agent = rep.get("agent_escalation") or {}
        if agent:
            print(f"\n  agent_escalation:")
            print(f"    ok    : {agent.get('ok')}")
            print(f"    phase : {agent.get('phase')!r}")
            for p in agent.get("proposals", []):
                print(f"    - {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
