"""E2E: Qwen2.5-0.5B full pipeline on nv8.

Triggered manually on nv8 with::

    HEYI_EVAL_E2E_ALLOW=1 pytest tests/e2e/ -m e2e -v

CI never runs these — see pyproject.toml ``-m "not e2e and not slow"``.

Test ID map (see docs/PR8_TEST_PLAN.md):

  E-1 full pipeline succeeds (9/9 stages OK)
  E-2 capability_score in [0, 1]
  E-3 showcase artifacts present (>= 5 prompts, each >= 200 chars)
  E-4 cleanup leaves no e9-* containers and preserves artifacts
  E-5 production containers untouched (INV-1 evidence)
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

QWEN_HF_ID = "Qwen/Qwen2.5-0.5B-Instruct"


pytestmark = [pytest.mark.e2e, pytest.mark.slow]


# ── helpers ────────────────────────────────────────────────────────────────


def _orchestrator_cmd(repo: Path, *args: str) -> list[str]:
    py = repo / ".venv" / "bin" / "python"
    if not py.exists():
        py = Path("python3.11")
    return [str(py), "-m", "orchestrator", *args]


def _enqueue(repo: Path, hf_id: str) -> str:
    out = subprocess.check_output(
        _orchestrator_cmd(repo, "enqueue", hf_id),
        text=True, timeout=60,
    )
    last_line = [ln for ln in out.splitlines() if ln.strip()][-1]
    run_id = last_line.split()[-1].strip()
    assert run_id, f"could not parse run_id from enqueue stdout: {out!r}"
    return run_id


def _run_one(repo: Path, *, timeout_s: int = 60 * 60) -> int:
    """Run a single queued run synchronously."""
    proc = subprocess.run(
        _orchestrator_cmd(repo, "run"),
        timeout=timeout_s,
        capture_output=True, text=True, check=False,
    )
    print(proc.stdout[-2000:])
    print(proc.stderr[-2000:])
    return proc.returncode


def _load_state(run_dir: Path) -> dict:
    sp = run_dir / "state.json"
    assert sp.exists(), f"state.json missing for run {run_dir}"
    return json.loads(sp.read_text(encoding="utf-8"))


# ── tests ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fresh_qwen_run(heyi_eval_repo: Path, heyi_eval_data: Path,
                   prod_container_snapshot) -> dict:
    """Single fixture that runs the whole pipeline once and returns a
    dict of references the per-test assertions need. We do NOT run the
    pipeline once per test — it's ~20 min on Qwen-0.5B."""
    snap_now, assert_unchanged = prod_container_snapshot
    before = snap_now()

    run_id = _enqueue(heyi_eval_repo, QWEN_HF_ID)
    t0 = time.time()
    rc = _run_one(heyi_eval_repo, timeout_s=45 * 60)
    elapsed = time.time() - t0

    return {
        "run_id": run_id,
        "rc": rc,
        "elapsed_s": elapsed,
        "run_dir": heyi_eval_data / "runs" / run_id,
        "before_prod_snap": before,
        "assert_unchanged": assert_unchanged,
    }


def test_e1_full_pipeline_succeeds(fresh_qwen_run: dict) -> None:
    """All 9 stages must finish ok."""
    assert fresh_qwen_run["rc"] == 0, (
        f"orchestrator run exited non-zero ({fresh_qwen_run['rc']})"
    )
    state = _load_state(fresh_qwen_run["run_dir"])
    failed = [
        s for s, info in state.get("stages", {}).items()
        if info.get("status") != "ok"
    ]
    assert not failed, f"stages not ok: {failed}; full state: {state}"


def test_e2_capability_score_in_range(fresh_qwen_run: dict) -> None:
    cap_path = fresh_qwen_run["run_dir"] / "capability.json"
    assert cap_path.exists(), "capability.json missing"
    cap = json.loads(cap_path.read_text(encoding="utf-8"))
    score = cap.get("overall_score") or cap.get("pass_rate")
    assert score is not None, f"capability score missing: {cap}"
    assert 0.0 <= float(score) <= 1.0, f"capability score out of range: {score}"


def test_e3_showcase_artifacts_present(fresh_qwen_run: dict) -> None:
    sp = fresh_qwen_run["run_dir"] / "showcase.json"
    assert sp.exists(), "showcase.json missing"
    showcase = json.loads(sp.read_text(encoding="utf-8"))
    prompts = showcase.get("prompts") or showcase.get("examples") or []
    assert len(prompts) >= 5, (
        f"expected >= 5 showcase prompts, got {len(prompts)}: {showcase}"
    )
    for i, p in enumerate(prompts):
        body = p.get("prompt") or p.get("body") or str(p)
        assert len(body) >= 200, (
            f"showcase prompt #{i} too short ({len(body)} chars) — "
            f"heyi_engine may have returned a stub: {body!r}"
        )


def test_e4_cleanup_removed_only_e9(fresh_qwen_run: dict) -> None:
    out = subprocess.check_output(
        ["docker", "ps", "-a", "--filter", "name=e9-",
         "--format", "{{.Names}}"],
        text=True, timeout=15,
    )
    remaining = [n.strip() for n in out.splitlines() if n.strip()]
    assert not remaining, (
        f"e9-* containers still present after run: {remaining}"
    )
    # Artifacts must have survived: INV-2
    assert fresh_qwen_run["run_dir"].exists(), (
        f"run dir wiped — INV-2 violation: {fresh_qwen_run['run_dir']}"
    )
    assert (fresh_qwen_run["run_dir"] / "capability.json").exists()
    assert (fresh_qwen_run["run_dir"] / "showcase.json").exists()


def test_e5_no_prod_container_touched(fresh_qwen_run: dict) -> None:
    """The hard guarantee: minimax / xrouter / glm-* / kimi-* / voipmonitor
    must be byte-for-byte identical (name, id, state, created) before
    and after the eval run."""
    fresh_qwen_run["assert_unchanged"](fresh_qwen_run["before_prod_snap"])
