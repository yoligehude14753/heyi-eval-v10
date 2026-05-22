"""Python-native DEPLOY / READY_WAIT / CLEANUP stage executors (v10).

Replaces the v9 cc-agent path. The single user-facing change is: cc-agent
no longer touches Docker for these three stages. The orchestrator owns
the docker socket and spawns / waits-on / removes containers itself.

Container naming convention (INV-1):
    e9-<engine>-<short>
where engine ∈ {vllm, sglang, tf}, short is the last 8 alphanumeric chars
of the run_id (lower-cased, '_' stripped). Every spawned container is
labelled:
    heyi_eval_run=<run_id>
    heyi_eval_stage=DEPLOY (or SHOWCASE for cc-agent)
    heyi_eval_engine=<engine>
so CLEANUP can find them by label without name-parsing, and a misnamed
container (e.g. someone named "minimax" but happened to match a label)
is double-checked against the e9-* prefix before removal.

Public surface:
    execute_deploy(run, cfg) -> StageResult
    execute_ready_wait(run, cfg) -> StageResult
    execute_cleanup(run, cfg, *, dry_run=False) -> StageResult
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import docker
from docker.errors import APIError, DockerException, ImageNotFound, NotFound

# docker-py is a hard runtime dep (see pyproject.toml). Tests that mock
# docker.from_env patch this module's `docker` symbol directly; they
# never need the package un-installable.
from .config import OrchestratorConfig
from .state_machine import Run, StageName

log = logging.getLogger(__name__)

# Engines we support and the stock images we deploy from. Pinned versions
# come from PLAN.md §3; bumping requires an ADR (rules § 15).
_ENGINE_IMAGES: dict[str, str] = {
    "vllm": "vllm/vllm-openai:v0.11.0",
    "sglang": "lmsysorg/sglang:v0.4.6.post1",
    "transformers": "heyi-eval/transformers-runner:v10",
}
# Short tag used in the container name; "transformers" → "tf" to stay
# under Docker's 63-char name limit even for long run_ids.
_ENGINE_SHORT: dict[str, str] = {
    "vllm": "vllm",
    "sglang": "sglang",
    "transformers": "tf",
}

# Labels every e9-* container must carry. Cleanup filters on heyi_eval_run.
LABEL_RUN = "heyi_eval_run"
LABEL_STAGE = "heyi_eval_stage"
LABEL_ENGINE = "heyi_eval_engine"

# Container name prefix INV-1 enforces.
E9_PREFIX = "e9-"


# ── shared dataclass ──────────────────────────────────────────────────────


@dataclass
class StageResult:
    """Outcome of one stage executor.

    Mirrors orchestrator.stages.StageResult so the dispatcher can return
    either type interchangeably until PR#4 retires the v9 module.
    """
    ok: bool
    duration_s: float
    artifacts: list[str]
    error: str | None = None
    payload: dict[str, Any] | None = None
    rc: int | None = None
    container_name: str | None = None
    # Free-form structured error details for the panel. Optional.
    error_kind: str | None = None  # "docker_down" | "image_pull" | "timeout" | ...
    extra: dict[str, Any] = field(default_factory=dict)


class StagePyError(RuntimeError):
    """Internal failure with structured kind+detail."""
    def __init__(self, message: str, *, kind: str, **extra: Any) -> None:
        super().__init__(message)
        self.kind = kind
        self.extra = extra


# ── helpers ───────────────────────────────────────────────────────────────


_ALNUM_RE = re.compile(r"[^a-z0-9]")


def _short_run_id(run_id: str) -> str:
    """8-char lowercase alphanumeric suffix for container naming.

    Stable: same run_id always produces the same suffix, so cleanup can
    locate by name even if the labels DB was wiped.
    """
    cleaned = _ALNUM_RE.sub("", run_id.lower())
    if len(cleaned) >= 8:
        return cleaned[-8:]
    # Pad with run_id hash if too short (test fixtures with run_id="a" etc).
    return cleaned.ljust(8, "0")


def container_name_for(run_id: str, engine: str) -> str:
    """Compose the deterministic e9-* container name."""
    short = _short_run_id(run_id)
    engine_short = _ENGINE_SHORT.get(engine, engine[:6])
    name = f"e9-{engine_short}-{short}"
    # Docker container name max length is 63 ([a-zA-Z0-9][a-zA-Z0-9_.-]+).
    return name[:63]


def _docker_client() -> Any:
    """Return a configured docker.DockerClient. Raises StagePyError on failure."""
    try:
        client = docker.from_env()
        client.ping()  # forces a roundtrip; fast fail if daemon is down
    except DockerException as e:
        raise StagePyError(
            f"docker daemon unreachable: {e}", kind="docker_down",
        ) from e
    return client


def _read_engine_plan(run_dir: Path) -> dict[str, Any]:
    """Load _meta/engine.json written by ENGINE_SELECT.

    Raises StagePyError if the file is missing or unreadable — DEPLOY
    cannot proceed without an explicit engine choice.
    """
    p = run_dir / "_meta" / "engine.json"
    if not p.exists():
        raise StagePyError(
            "engine.json not present (run ENGINE_SELECT first)",
            kind="missing_artifact",
            path=str(p),
        )
    try:
        loaded = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise StagePyError(
            f"engine.json invalid: {e}", kind="bad_artifact",
        ) from e
    if not isinstance(loaded, dict):
        raise StagePyError(
            f"engine.json must be a JSON object, got {type(loaded).__name__}",
            kind="bad_artifact",
        )
    return loaded


def _model_path_on_host(cfg: OrchestratorConfig, hf_id: str) -> Path:
    """Where the model weights live on the host filesystem.

    Local convention: `<cfg.model_cache_root>/<basename(hf_id)>/`.
    """
    return cfg.model_cache_root / cfg.hf_local_dir(hf_id)


def _vllm_command(model_in_container: str, args: Mapping[str, Any], port: int) -> list[str]:
    """Translate engine_select's vllm_args dict to vllm CLI flags.

    Accepts both snake_case and kebab-case keys. We don't validate
    every possible flag — vllm itself does that at startup; bad flags
    surface as STARTED-then-EXITED containers and S7 catches them.
    """
    cmd = [
        "--model", model_in_container,
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    seen: set[str] = set()
    for k, v in args.items():
        flag = "--" + k.replace("_", "-")
        if flag in seen:
            continue
        seen.add(flag)
        cmd.extend([flag, str(v)])
    return cmd


def _sglang_command(model_in_container: str, args: Mapping[str, Any], port: int) -> list[str]:
    """SGLang launch CLI. Similar structure to vllm."""
    cmd = [
        "python", "-m", "sglang.launch_server",
        "--model-path", model_in_container,
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    seen: set[str] = set()
    for k, v in args.items():
        flag = "--" + k.replace("_", "-")
        if flag in seen:
            continue
        seen.add(flag)
        cmd.extend([flag, str(v)])
    return cmd


def _http_get_json(url: str, *, timeout: float = 5.0) -> tuple[int, dict[str, Any] | None]:
    """Tiny GET helper. Returns (status_code, parsed_body_or_None).

    Lives in this module (not heyi_engine) because the eval target's
    OpenAI-shape API is a separate concern from the production engine
    we call through heyi_engine.client. Keeping them separate makes
    INV-2 trivially satisfiable.
    """
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8", errors="replace")
            try:
                return (r.getcode(), json.loads(raw))
            except json.JSONDecodeError:
                return (r.getcode(), None)
    except urllib.error.HTTPError as e:
        return (e.code, None)
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
        return (0, None)


# ── GPU isolation (PR#11) ─────────────────────────────────────────────────

# nvidia-smi memory.used threshold below which we consider a GPU "idle
# enough" for the eval pipeline to grab it. 1 GiB tolerates driver-reserved
# memory and small monitoring tools, but flags any real workload.
_EVAL_GPU_IDLE_THRESHOLD_MIB = 1024


def _nvidia_smi_used_mib() -> dict[int, int]:
    """Query each GPU's memory.used in MiB via nvidia-smi.

    Returns {gpu_index: mib}. Raises FileNotFoundError if nvidia-smi is
    missing (caller treats as graceful skip), TimeoutError on hang, or
    RuntimeError on parser failure.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5, check=False,
    )
    if out.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi exit {out.returncode}: {out.stderr.strip()[:200]}"
        )
    result: dict[int, int] = {}
    for line in out.stdout.strip().splitlines():
        try:
            idx_s, mem_s = line.split(",")
            result[int(idx_s.strip())] = int(mem_s.strip())
        except ValueError as e:
            raise RuntimeError(f"nvidia-smi parse failure on {line!r}: {e}") from e
    return result


SmiQuery = Callable[[], dict[int, int]]


def _select_eval_gpus(
    cfg: OrchestratorConfig,
    tp_size: int,
    *,
    smi_query: SmiQuery | None = None,
) -> tuple[list[int] | None, str | None]:
    """Decide which physical GPUs the eval container should bind to.

    Returns either:
        (selected_gpus, None)  → safe to spawn the container
        (None, reason)         → graceful skip, do not spawn

    Skip reasons cover:
      - tp_size > len(cfg.eval_gpus)             (model too wide)
      - cfg.eval_gpus is ()                       (operator drained the pool)
      - cfg.eval_gpus overlaps prod_engine_gpus   (transient prod takeover)
      - nvidia-smi shows an eval GPU non-idle    (operator-run squatter)
      - nvidia-smi missing/timed-out             (can't verify → skip)
    """
    if not cfg.eval_gpus:
        return None, "eval pool is empty (cfg.eval_gpus is ()); operator must drain prod first"

    overlap = sorted(set(cfg.eval_gpus) & set(cfg.prod_engine_gpus))
    if overlap:
        return None, (
            f"eval pool {list(cfg.eval_gpus)} overlaps prod_engine_gpus "
            f"{list(cfg.prod_engine_gpus)} on {overlap}; production is "
            f"transiently using eval GPUs (e.g. TP=8 mode)"
        )

    if tp_size > len(cfg.eval_gpus):
        return None, (
            f"model needs tensor_parallel_size={tp_size} but eval "
            f"pool has {len(cfg.eval_gpus)} GPUs ({list(cfg.eval_gpus)}); "
            f"too large for this machine"
        )

    smi_call = smi_query or _nvidia_smi_used_mib
    try:
        smi = smi_call()
    except (FileNotFoundError, TimeoutError, OSError, RuntimeError) as e:
        return None, f"nvidia-smi unavailable: {type(e).__name__}: {e}"

    busy: list[tuple[int, int]] = []
    for gpu in cfg.eval_gpus:
        mem = smi.get(gpu, 0)
        if mem > _EVAL_GPU_IDLE_THRESHOLD_MIB:
            busy.append((gpu, mem))
    if busy:
        details = ", ".join(f"GPU {g}={m}MiB" for g, m in busy)
        return None, (
            f"eval pool not idle (threshold {_EVAL_GPU_IDLE_THRESHOLD_MIB} MiB): {details}; "
            f"another workload is using the eval GPUs"
        )

    return list(cfg.eval_gpus[:tp_size]), None


def _graceful_skip(t0: float, reason: str) -> StageResult:
    """Wrap a graceful-skip reason into the canonical StageResult shape."""
    return StageResult(
        ok=False,
        duration_s=time.time() - t0,
        artifacts=[],
        error=f"insufficient_gpu: {reason}",
        error_kind="insufficient_gpu",
        extra={"aborted": True, "reason": reason},
    )


# ── DEPLOY ────────────────────────────────────────────────────────────────


def execute_deploy(
    run: Run,
    cfg: OrchestratorConfig,
    *,
    sleep: Any = time.sleep,
) -> StageResult:
    """Spawn an e9-<engine>-<short> container serving the run's model.

    Side effects (only on ok=True):
        runs/<run_id>/deploy.json  with container_name + base_url + image
        runs/<run_id>/_meta/deploy.json (legacy alias for validators)
    """
    t0 = time.time()
    rd = cfg.run_dir(run.run_id)
    rd.mkdir(parents=True, exist_ok=True)

    try:
        plan = _read_engine_plan(rd)
        engine = (plan.get("engine") or "").lower()
        if engine not in _ENGINE_IMAGES:
            raise StagePyError(
                f"unknown engine={engine!r}; valid: {sorted(_ENGINE_IMAGES)}",
                kind="unknown_engine",
            )

        model_host_path = _model_path_on_host(cfg, run.hf_id)
        if not model_host_path.exists():
            raise StagePyError(
                f"model path not found: {model_host_path}",
                kind="model_missing",
            )

        cname = container_name_for(run.run_id, engine)
        image = _ENGINE_IMAGES[engine]
        port = cfg.vllm_port  # one port for now; future PRs may multiplex
        model_in_container = "/model"

        vllm_args = plan.get("vllm_args") or {}
        tp_size = int(vllm_args.get("tensor_parallel_size", 1) or 1)

        # GPU isolation gate (PR#11): pick the eval-pool GPUs we'll bind
        # to, or graceful-skip if the eval pool is unavailable / too small
        # / overlapping production. Done BEFORE any docker call so we
        # don't even ping the daemon if we know we can't run.
        if engine in ("vllm", "sglang", "transformers"):
            selected_gpus, skip_reason = _select_eval_gpus(cfg, tp_size)
            if selected_gpus is None:
                assert skip_reason is not None
                return _graceful_skip(t0, skip_reason)
        else:
            selected_gpus = []  # non-GPU engines (none today, but keep shape)

        client = _docker_client()

        _reuse_or_recreate(client, cname, run.run_id, engine)

        if engine == "vllm":
            command = _vllm_command(model_in_container, vllm_args, port)
        elif engine == "sglang":
            command = _sglang_command(model_in_container, vllm_args, port)
        else:  # transformers
            command = ["serve", "--model-path", model_in_container, "--port", str(port)]

        labels = {
            LABEL_RUN: run.run_id,
            LABEL_STAGE: StageName.DEPLOY.value,
            LABEL_ENGINE: engine,
        }
        device_requests = (
            [
                docker.types.DeviceRequest(
                    device_ids=[str(i) for i in selected_gpus],
                    capabilities=[["gpu"]],
                )
            ]
            if selected_gpus
            else []
        )
        try:
            container = client.containers.run(
                image,
                command=command,
                name=cname,
                detach=True,
                remove=False,
                labels=labels,
                network_mode="host",
                ipc_mode="host",
                shm_size="16g",
                volumes={
                    str(model_host_path): {"bind": model_in_container, "mode": "ro"},
                },
                device_requests=device_requests,
            )
        except ImageNotFound as e:
            raise StagePyError(f"image not found: {image}: {e}",
                               kind="image_pull") from e
        except APIError as e:
            msg = str(e)
            if "address already in use" in msg or "port is already allocated" in msg:
                raise StagePyError(
                    f"port {port} already in use: {e}", kind="port_in_use",
                ) from e
            raise StagePyError(f"docker API error: {e}", kind="docker_api") from e

        sleep(0.5)
        container.reload()
        if container.status in ("exited", "dead"):
            tail = _tail_logs(container, 50)
            try:
                container.remove(force=True)
            except DockerException:
                pass
            raise StagePyError(
                f"container exited immediately; tail logs: {tail!r}",
                kind="early_exit",
                logs=tail,
            )

        base_url = f"http://127.0.0.1:{port}"
        deploy_payload = {
            "stage": "DEPLOY",
            "engine": engine,
            "engine_image": image,
            "container_name": cname,
            "base_url": base_url,
            "model_path_host": str(model_host_path),
            "model_path_container": model_in_container,
            "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        _write_artifact(rd, "deploy.json", deploy_payload)
        _write_artifact(rd / "_meta", "deploy.json", deploy_payload)

        return StageResult(
            ok=True,
            duration_s=time.time() - t0,
            artifacts=["deploy.json", "_meta/deploy.json"],
            payload=deploy_payload,
            rc=0,
            container_name=cname,
        )

    except StagePyError as e:
        return StageResult(
            ok=False,
            duration_s=time.time() - t0,
            artifacts=[],
            error=str(e),
            error_kind=e.kind,
            extra=e.extra,
        )


def _reuse_or_recreate(
    client: Any, cname: str, run_id: str, engine: str,
) -> None:
    """If a container named cname already exists, force-remove it unless
    it's the same run and still healthy (in which case caller will reuse).

    We don't bother with the "reuse healthy" path beyond logging — DEPLOY
    is idempotent and respawning is cheap relative to vllm startup. The
    next refresh of this code can add reuse if it shows up as a wall-time
    pain in PR#8 E2E.
    """
    try:
        existing = client.containers.get(cname)
    except NotFound:
        return

    labels = existing.attrs.get("Config", {}).get("Labels") or {}
    same_run = labels.get(LABEL_RUN) == run_id
    log.info(
        "deploy: found existing container %s (status=%s, same_run=%s); "
        "force-removing to redeploy", cname, existing.status, same_run,
    )
    try:
        existing.remove(force=True)
    except DockerException as e:  # pragma: no cover — best-effort
        log.warning("could not remove stale container %s: %s", cname, e)


def _tail_logs(container: Any, n: int) -> str:
    try:
        raw = container.logs(tail=n)
    except DockerException:
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _write_artifact(directory: Path, filename: str, payload: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ── READY_WAIT ────────────────────────────────────────────────────────────


def execute_ready_wait(
    run: Run,
    cfg: OrchestratorConfig,
    *,
    poll_interval_s: float = 5.0,
    early_probes: int = 5,
    early_interval_s: float = 2.0,
    sleep: Any = time.sleep,
) -> StageResult:
    """Poll /v1/models on the deployed engine until it returns 200 with a model.

    Strategy: 5 quick probes (2s apart) to catch fast-start engines like
    sglang on small models, then exponential back-off to 5s per probe.
    Total budget = cfg.timeout_for("READY_WAIT") or fall back to deploy_timeout_s.
    """
    t0 = time.time()
    rd = cfg.run_dir(run.run_id)

    deploy_path = rd / "deploy.json"
    if not deploy_path.exists():
        return StageResult(
            ok=False, duration_s=time.time() - t0, artifacts=[],
            error="deploy.json not present (run DEPLOY first)",
            error_kind="missing_artifact",
        )
    deploy = json.loads(deploy_path.read_text(encoding="utf-8"))
    base_url = deploy["base_url"]
    cname = deploy["container_name"]

    # Wall-clock budget. READY_WAIT shares the DEPLOY timeout pool because
    # they're operationally tied — if DEPLOY only had 600s and used 400s
    # spinning up vllm, we'd rather fail than budget another 600s here.
    timeout_s = cfg.deploy_timeout_s
    deadline = t0 + timeout_s

    client = None
    try:
        client = _docker_client()
    except StagePyError as e:
        return StageResult(
            ok=False, duration_s=time.time() - t0, artifacts=[],
            error=str(e), error_kind=e.kind,
        )

    probe_count = 0
    last_status = None
    last_error: str | None = None

    while time.time() < deadline:
        if client is not None:
            try:
                ctr = client.containers.get(cname)
                if ctr.status in ("exited", "dead"):
                    tail = _tail_logs(ctr, 50)
                    return StageResult(
                        ok=False, duration_s=time.time() - t0, artifacts=[],
                        error=f"container exited: {tail!r}",
                        error_kind="container_died",
                        container_name=cname,
                        extra={"logs": tail},
                    )
            except NotFound:
                return StageResult(
                    ok=False, duration_s=time.time() - t0, artifacts=[],
                    error=f"container {cname} disappeared",
                    error_kind="container_lost",
                    container_name=cname,
                )

        status, body = _http_get_json(f"{base_url}/v1/models", timeout=5.0)
        probe_count += 1
        last_status = status

        if status == 200 and isinstance(body, dict):
            data = body.get("data") or []
            if data:
                model_id = data[0].get("id") or "unknown"
                ready_payload = {
                    "stage": "READY_WAIT",
                    "container_name": cname,
                    "base_url": base_url,
                    "model_id": model_id,
                    "probe_count": probe_count,
                    "elapsed_s": round(time.time() - t0, 3),
                    "ready_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
                }
                _write_artifact(rd, "ready.json", ready_payload)
                _write_artifact(rd / "_meta", "ready.json", ready_payload)
                return StageResult(
                    ok=True, duration_s=time.time() - t0,
                    artifacts=["ready.json", "_meta/ready.json"],
                    payload=ready_payload, rc=0, container_name=cname,
                )
            last_error = "engine started but /v1/models data=[]"

        elif status not in (0, 503, 502, 504):
            last_error = f"unexpected status {status}"

        gap = early_interval_s if probe_count < early_probes else poll_interval_s
        sleep(gap)

    return StageResult(
        ok=False, duration_s=time.time() - t0, artifacts=[],
        error=(
            f"timeout after {round(time.time() - t0)}s "
            f"({probe_count} probes, last_status={last_status}, "
            f"last_error={last_error})"
        ),
        error_kind="timeout",
        container_name=cname,
        extra={"probe_count": probe_count, "last_status": last_status},
    )


# ── CLEANUP ───────────────────────────────────────────────────────────────


def execute_cleanup(
    run: Run,
    cfg: OrchestratorConfig,
    *,
    dry_run: bool = False,
) -> StageResult:
    """Remove all e9-* containers labelled for this run.

    INV-1 defense in depth:
      1. We filter by label heyi_eval_run=<run_id> so only containers
         we ourselves spawned can match.
      2. We additionally require name.startswith("e9-"). A container that
         passes (1) but fails (2) is a label collision attack: skip and
         log loudly.
      3. We never touch /home/ai/heyi-eval-data on disk — model weights
         under cfg.model_cache_root are not in scope for CLEANUP; that's
         a separate operator decision.
    """
    t0 = time.time()
    rd = cfg.run_dir(run.run_id)

    try:
        client = _docker_client()
    except StagePyError as e:
        return StageResult(
            ok=False, duration_s=time.time() - t0, artifacts=[],
            error=str(e), error_kind=e.kind,
        )

    try:
        candidates = client.containers.list(
            all=True,
            filters={"label": f"{LABEL_RUN}={run.run_id}"},
        )
    except DockerException as e:
        return StageResult(
            ok=False, duration_s=time.time() - t0, artifacts=[],
            error=f"list containers failed: {e}", error_kind="docker_api",
        )

    removed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for c in candidates:
        cname = getattr(c, "name", "") or c.attrs.get("Name", "").lstrip("/")
        if not cname.startswith(E9_PREFIX):
            skipped.append({"name": cname, "reason": "INV-1: not e9-* prefix"})
            log.error("INV-1 violation candidate skipped: %s", cname)
            continue

        if dry_run:
            removed.append({"name": cname, "dry_run": True})
            continue

        try:
            c.remove(force=True)
            removed.append({"name": cname, "stopped": True, "removed": True})
        except NotFound:
            removed.append({"name": cname, "vanished_before_remove": True})
        except APIError as e:
            try:
                c.remove(force=True, v=True)
                removed.append({"name": cname, "removed_after_retry": True})
            except DockerException as e2:
                failed.append({"name": cname, "error": str(e2)})
                log.warning("cleanup: failed to remove %s: %s", cname, e2)
                _ = e  # keep silenced

    # Also sweep any orphan e9-* containers from previous failed runs that
    # *don't* have the run label — log but never delete. Operator decides.
    orphans = []
    try:
        all_e9 = client.containers.list(all=True)
        for c in all_e9:
            n = getattr(c, "name", "") or c.attrs.get("Name", "").lstrip("/")
            if not n.startswith(E9_PREFIX):
                continue
            labels = c.attrs.get("Config", {}).get("Labels") or {}
            if labels.get(LABEL_RUN) != run.run_id and labels.get(LABEL_RUN) is None:
                orphans.append({"name": n})
    except DockerException:
        pass

    payload = {
        "stage": "CLEANUP",
        "run_id": run.run_id,
        "removed": removed,
        "skipped": skipped,
        "failed": failed,
        "orphans": orphans,
        "dry_run": dry_run,
        "cleaned_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
    }
    _write_artifact(rd, "cleanup.json", payload)
    _write_artifact(rd / "_meta", "cleanup.json", payload)

    ok = len(failed) == 0
    return StageResult(
        ok=ok,
        duration_s=time.time() - t0,
        artifacts=["cleanup.json", "_meta/cleanup.json"],
        payload=payload,
        rc=0 if ok else 1,
        error=(f"{len(failed)} container(s) failed to remove" if not ok else None),
        error_kind="remove_failed" if not ok else None,
    )


# Suppress unused-import warning for os when run from CI without the
# host-uid path active. Kept for future hooks (mount uid mapping).
_ = os
