#!/usr/bin/env python3
"""project_lane heyi drill — single end-to-end run on a real m2b-claude-code
container, talking to real Yunwu M2.7, against a real GitHub repo.

Pre-requisites (the script auto-checks and refuses to run if missing):
  1. docker daemon reachable (``docker info`` exits 0)
  2. an m2b-claude-code container is running and named per
     ``$HEYI_DRILL_CONTAINER`` (default ``m2b-1``)
  3. ``YUNWU_GENERAL_KEY`` (or ``YUNWU_KEY_2``) in env — needed for ccr config
  4. python ``docker`` package importable (``pip install docker``)

Usage:

    # smallest happy-path target — a tiny CLI with a clear README
    python scripts/drill_project_lane.py simonw/llm

    # sad path — known-oversize repo
    python scripts/drill_project_lane.py "huggingface/transformers" --expect-oversize

Output:

    - drill log to stdout (live)
    - artifacts under ``$HEYI_EVAL_DATA/project_lane/runs/<run_id>/``

Why a separate drill script rather than just ``orchestrator __main__``?

The drill answers questions like "does inject_into_container actually
land the config + does ccr really SIGHUP on it" that the unit suite
mocks away. It's deliberately a single-run, single-container narrow
script so a failure is unambiguous; orchestrator's main loop adds
queue scheduling + retry which obscure what broke when something does.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _die(msg: str, code: int = 2) -> None:
    print(f"DRILL ABORT: {msg}", file=sys.stderr)
    sys.exit(code)


def _check_prereqs(container_name: str) -> tuple[object, Path]:
    """Return (docker_client, host_workspace_root) or die with a clear message."""
    try:
        import docker
    except ImportError:
        _die("python 'docker' package not installed; pip install docker")
        raise  # for type-checker

    try:
        client = docker.from_env()
        client.ping()
    except Exception as e:
        _die(f"docker daemon unreachable: {e}")
        raise

    try:
        container = client.containers.get(container_name)
    except Exception as e:
        _die(
            f"container '{container_name}' not found; "
            f"start m2b-claude-code first (see docs/m2b-claude-code-readme.md): {e}"
        )
        raise
    if container.status != "running":
        _die(f"container '{container_name}' status={container.status}, expected 'running'")

    yunwu_key = (
        os.environ.get("YUNWU_GENERAL_KEY")
        or os.environ.get("YUNWU_KEY_2")
        or os.environ.get("YUNWU_GPT_KEY")
    )
    if not yunwu_key:
        _die("no YUNWU_GENERAL_KEY/YUNWU_KEY_2/YUNWU_GPT_KEY in env; needed for ccr config")

    workspace = Path(os.environ.get("HEYI_DRILL_WORKSPACE", "/tmp/heyi-drill-ws"))
    workspace.mkdir(parents=True, exist_ok=True)

    return client, workspace


def _ts() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")


def main() -> int:
    ap = argparse.ArgumentParser(description="project_lane heyi drill")
    ap.add_argument("full_id", help="GitHub repo owner/name (e.g. simonw/llm)")
    ap.add_argument(
        "--container", default=os.environ.get("HEYI_DRILL_CONTAINER", "m2b-1"),
        help="m2b container name (default: m2b-1)",
    )
    ap.add_argument(
        "--expect-oversize", action="store_true",
        help="assert preflight rejects this as OVERSIZE",
    )
    ap.add_argument(
        "--skip-inject", action="store_true",
        help="skip ccr config injection (assume container already configured)",
    )
    ap.add_argument(
        "--token-budget", type=int, default=50_000,
        help="agent token budget (default 50000)",
    )
    ap.add_argument(
        "--wall-clock-s", type=float, default=900.0,
        help="agent wall-clock budget in seconds (default 900 = 15 min)",
    )
    args = ap.parse_args()

    print(f"[{_ts()}] drill begin: target={args.full_id} container={args.container}")
    client, workspace = _check_prereqs(args.container)

    from agent_driver.budget_guard import BudgetGuard
    from agent_driver.ccr_bridge import build_ccr_config, inject_into_container
    from agent_driver.exec_runner import make_docker_stream_exec
    from agent_driver.pool_manager import PoolManager
    from discover.radar_ingest import ProjectCandidate
    from orchestrator.project_lane import ProjectStore, execute_run

    # ── 1. inject ccr config (unless told to skip) ─────────────────────
    if not args.skip_inject:
        print(f"[{_ts()}] inject ccr config (route to yunwu M2.7)...")
        cfg = build_ccr_config()  # picks up YUNWU_GENERAL_KEY from env
        inject_into_container(args.container, docker_client=client, config=cfg)
        print(f"[{_ts()}] ccr config injected")
    else:
        print(f"[{_ts()}] skipping ccr inject (--skip-inject)")

    # ── 2. enqueue a synthetic candidate ────────────────────────────────
    data_root = Path(os.environ.get("HEYI_EVAL_DATA", "/tmp/heyi-drill-data"))
    store = ProjectStore(data_root)
    cand = ProjectCandidate(
        full_id=args.full_id,
        source_url=f"https://github.com/{args.full_id}",
        source_report="drill-manual",
        source_date=datetime.now(UTC).strftime("%Y-%m-%d"),
        discovered_at=datetime.now(UTC).isoformat(),
        stars_delta=None, stars_total=None,
        short_desc="(drill target — manually injected)",
        reason="manual",
    )
    run_id = store.enqueue(cand, skip_if_recent_hours=0.0)  # always fresh
    if run_id is None:
        _die("enqueue returned None — dedup hit? Use skip_if_recent_hours=0 expected; bug?")
        return 2
    print(f"[{_ts()}] enqueued run_id={run_id}")

    # ── 3. execute end-to-end ────────────────────────────────────────────
    pool = PoolManager(
        container_names=[args.container],
        host_workspace_root=workspace,
        docker_client=client,
    )
    stream_exec = make_docker_stream_exec(client)
    guard = BudgetGuard(
        token_budget=args.token_budget,
        wall_clock_s=args.wall_clock_s,
    )
    run = store.load(run_id)
    print(f"[{_ts()}] execute_run start")
    result = execute_run(
        store, run,
        pool=pool, host_workspace_root=workspace,
        stream_exec=stream_exec, budget_guard=guard,
    )
    print(f"[{_ts()}] execute_run done  status={result.status.value} "
          f"outcome={result.summary_outcome}")

    # ── 4. verdict ───────────────────────────────────────────────────────
    if args.expect_oversize:
        if result.status.value == "oversize":
            print(f"[{_ts()}] DRILL PASS — preflight blocked OVERSIZE as expected")
            return 0
        print(
            f"[{_ts()}] DRILL FAIL — expected OVERSIZE, got {result.status.value} "
            f"({result.failure_reason_zh})", file=sys.stderr,
        )
        return 1

    if result.status.value == "done" and result.summary_outcome == "pass":
        print(f"[{_ts()}] DRILL PASS — agent self-reported PASS on real run")
        return 0
    # PARTIAL / FAIL are acceptable "happy-path" outcomes for the drill —
    # we're testing whether the pipeline plumbing works, not whether
    # this specific repo deploys cleanly.
    if result.status.value == "done":
        print(
            f"[{_ts()}] DRILL PARTIAL — pipeline ran end-to-end, "
            f"agent verdict was {result.summary_outcome}; check report.json"
        )
        return 0
    print(
        f"[{_ts()}] DRILL FAIL — pipeline did not reach DONE: "
        f"status={result.status.value} reason={result.failure_reason_zh}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
