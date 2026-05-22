"""
runner_validator: deterministic post-stage check.

CC's self-reported "success" is NOT trustworthy (proven by E8 attempt 1: CC
exited result=success but capability.json/manifest.json were never written and
the v8-era prefix was left dangling).

After every stage, validator runs the appropriate check. Failure overrides
whatever CC said; the stage is marked FAILED and the run goes to error path.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover  (jsonschema is a hard dep, but soft-fail import)
    Draft202012Validator = None


class ValidationError(Exception):
    """Raised when runner_validator finds a violation."""


def default_schema_root() -> Path:
    """Resolve the bundled SOPs schema directory.

    Layout (relative to this file):
      heyi-eval-v9/
        orchestrator/validator.py   <-- this file
        sops/schemas/*.schema.json
    """
    return Path(__file__).resolve().parent.parent / "sops" / "schemas"


# ── Generic helpers ────────────────────────────────────────────────────────


def _require_file(path: Path, hint: str) -> dict[str, Any]:
    if not path.exists():
        raise ValidationError(f"required artifact missing: {hint} ({path})")
    try:
        return json.loads(path.read_text())
    except Exception as e:
        raise ValidationError(f"{hint} is not valid JSON: {e}") from e


def _validate_schema(payload: dict[str, Any], schema_path: Path, label: str) -> None:
    if not schema_path.exists() or Draft202012Validator is None:
        # MVP soft-skip if schema or library not yet available
        return
    schema = json.loads(schema_path.read_text())
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
    if errors:
        first = errors[0]
        raise ValidationError(
            f"{label} schema violation: {first.message} (at {list(first.path)})"
        )


# ── Per-stage validators ───────────────────────────────────────────────────


def validate_deploy(
    run_dir: Path,
    *,
    schema_root: Path | None = None,
    check_container_live: bool = True,
) -> dict[str, Any]:
    """
    DEPLOY OK = ready.json exists & schema-valid & container_name has e9- prefix
                & (optionally) docker inspect reports container is running.

    Returns the parsed ready.json so the orchestrator can read e.g.
    container_name / endpoint for downstream stages.

    v10 changes vs v9:
      - prefix is e9- (not e8-) so we can run alongside any v9 leftovers
        during the migration window;
      - gpu_index range check removed — production GPU isolation moved
        to label-based filtering in stages_py.execute_cleanup (see INV-1
        rationale in docs/PLAN.md). gpu_index in ready.json is now
        informational only.
    """
    payload = _require_file(run_dir / "ready.json", "ready.json")
    if schema_root is not None:
        _validate_schema(payload, schema_root / "ready.schema.json", "ready.json")

    container_name = payload.get("container_name", "")
    if not container_name.startswith("e9-"):
        raise ValidationError(
            f"INV-1 violation: container '{container_name}' lacks e9- prefix"
        )

    if check_container_live:
        try:
            out = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Status}}", container_name],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except Exception as e:
            raise ValidationError(f"docker inspect failed: {e}") from e
        status = out.stdout.strip()
        if status != "running":
            raise ValidationError(
                f"DEPLOY: container '{container_name}' status={status!r}, expected 'running'"
            )

    return payload


def validate_capability(
    run_dir: Path,
    *,
    schema_root: Path | None = None,
    min_results: int = 1,
) -> dict[str, Any]:
    """
    CAPABILITY OK = capability.json exists, schema-valid, results non-empty.

    Returns parsed payload.
    """
    payload = _require_file(run_dir / "capability.json", "capability.json")
    if schema_root is not None:
        _validate_schema(payload, schema_root / "capability.schema.json", "capability.json")
    results = payload.get("results", [])
    if not isinstance(results, list) or len(results) < min_results:
        raise ValidationError(
            f"CAPABILITY: 'results' has {len(results) if isinstance(results, list) else 0} "
            f"items, need >= {min_results}"
        )
    for i, r in enumerate(results):
        if "pass" not in r or not isinstance(r["pass"], bool):
            raise ValidationError(f"CAPABILITY: results[{i}].pass missing or not bool")
    return payload


def validate_showcase(
    run_dir: Path,
    *,
    schema_root: Path | None = None,
    require_summary: bool = True,
) -> dict[str, Any]:
    """
    SHOWCASE OK = showcase.json exists, has at least 1 item, items have required fields.

    Schema is permissive (allow partial / degraded) — even 1 item counts.

    `require_summary=False` lets partial/aborted runs pass schema as degraded
    success when the orchestrator decides to.
    """
    payload = _require_file(run_dir / "showcase.json", "showcase.json")
    if schema_root is not None:
        _validate_schema(payload, schema_root / "showcase.schema.json", "showcase.json")
    items = payload.get("items", [])
    if not isinstance(items, list) or len(items) == 0:
        raise ValidationError("SHOWCASE: 'items' missing or empty")
    required_per_item = {"id", "prompt", "actual", "rationale"}
    for i, it in enumerate(items):
        missing = required_per_item - set(it.keys())
        if missing:
            raise ValidationError(f"SHOWCASE: items[{i}] missing fields: {missing}")
    if require_summary and not payload.get("summary"):
        raise ValidationError("SHOWCASE: 'summary' field missing")
    return payload


def validate_cleanup(run_dir: Path, *, ephemeral_container: str) -> None:
    """CLEANUP OK = the ephemeral container is gone (docker rm done)."""
    try:
        out = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"name={ephemeral_container}"],
            capture_output=True, text=True, timeout=10, check=False,
        )
    except Exception as e:
        raise ValidationError(f"docker ps failed: {e}") from e
    if out.stdout.strip():
        raise ValidationError(
            f"CLEANUP: ephemeral container '{ephemeral_container}' still exists "
            f"(INV-4 violation, CC should have docker rm)"
        )


# ── Invariant snapshots (call between stages) ──────────────────────────────


def assert_invariants(
    *,
    prod_engine_container: str = "minimax",
    prod_engine_gpus: tuple[int, ...] = (0, 1, 2, 3),
    expected_prod_gpu_min_mib: int = 80_000,
    # Deprecated since PR#10. Kept for backward compatibility with any
    # caller that hasn't migrated yet — emits DeprecationWarning.
    minimax_gpus: tuple[int, ...] | None = None,
    expected_minimax_gpu_min_mib: int | None = None,
) -> None:
    """
    Snapshot check between eval pipeline stages.

    - INV-1: production LLM still owns ``prod_engine_gpus`` (KV occupancy
             ≥ ``expected_prod_gpu_min_mib`` per GPU).
    - INV-2: ``prod_engine_container`` still running.
    - INV-3 is enforced by docker-socket-proxy (separate).

    The "production LLM" is whatever vLLM container heyi_engine talks to on
    :10814 — defaults to ``minimax`` (MiniMax-M2.7 TP=4 on GPU 0-3), but the
    user transiently switches to Kimi-K2.6 TP=8 etc.; callers should pass
    the live values from ``OrchestratorConfig.prod_engine_container`` /
    ``OrchestratorConfig.prod_engine_gpus``.
    """
    if minimax_gpus is not None:
        import warnings as _w
        _w.warn(
            "assert_invariants(minimax_gpus=...) is deprecated since PR#10; "
            "use prod_engine_gpus=... instead",
            DeprecationWarning,
            stacklevel=2,
        )
        prod_engine_gpus = minimax_gpus
    if expected_minimax_gpu_min_mib is not None:
        import warnings as _w
        _w.warn(
            "assert_invariants(expected_minimax_gpu_min_mib=...) is deprecated "
            "since PR#10; use expected_prod_gpu_min_mib=... instead",
            DeprecationWarning,
            stacklevel=2,
        )
        expected_prod_gpu_min_mib = expected_minimax_gpu_min_mib

    out = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Status}}", prod_engine_container],
        capture_output=True, text=True, timeout=5, check=False,
    )
    status = out.stdout.strip()
    if status != "running":
        raise ValidationError(
            f"INV-2 violation: {prod_engine_container!r} container "
            f"status={status!r} (expected 'running')"
        )

    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5, check=False,
    )
    for line in out.stdout.strip().splitlines():
        try:
            idx_s, mem_s = line.split(",")
            idx = int(idx_s.strip())
            mem = int(mem_s.strip())
        except ValueError:
            continue
        if idx in prod_engine_gpus and mem < expected_prod_gpu_min_mib:
            raise ValidationError(
                f"INV-1 violation: GPU {idx} mem={mem} MiB "
                f"< {expected_prod_gpu_min_mib} MiB; "
                f"production container {prod_engine_container!r} may have been "
                f"killed or swapped out"
            )
