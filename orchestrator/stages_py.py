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


def _graceful_skip(
    t0: float, reason: str, *, error_kind: str = "insufficient_gpu",
) -> StageResult:
    """Wrap a graceful-skip reason into the canonical StageResult shape.

    PR#25: ``error_kind`` is now parameterised. Historically every
    abort path emitted ``error_kind="insufficient_gpu"`` regardless
    of actual cause (model_missing, eval_pool_busy, port_collision,
    etc.), which (a) made the test_s3_model_path_missing test
    misleading and (b) made Panel grouping useless. New callers
    pass a specific kind; the default keeps the old behaviour so
    no caller needs an immediate update.
    """
    return StageResult(
        ok=False,
        duration_s=time.time() - t0,
        artifacts=[],
        error=f"{error_kind}: {reason}",
        error_kind=error_kind,
        extra={"aborted": True, "reason": reason},
    )


# ── DEPLOY ────────────────────────────────────────────────────────────────


def execute_deploy(
    run: Run,
    cfg: OrchestratorConfig,
    *,
    sleep: Any = time.sleep,
    enable_repair: bool = True,
) -> StageResult:
    """Spawn an e9-<engine>-<short> container serving the run's model.

    Side effects (only on ok=True):
        runs/<run_id>/deploy.json  with container_name + base_url + image
        runs/<run_id>/_meta/deploy.json (legacy alias for validators)
        runs/<run_id>/_meta/deploy_repair.json (PR#33; only if repair
            was triggered — absence means first-attempt success)

    PR#33: when the first attempt fails with an engine-side error
    (early_exit / docker_api / image_pull), the function walks a small
    ordered list of deterministic repair strategies — adding
    --trust-remote-code, lowering max_model_len, swapping the vLLM
    image to :latest, swapping the engine to SGLang, falling back to
    transformers — re-invoking the docker spawn after each plan
    mutation. The whole attempt history (winning strategy, every
    rejected strategy, full container log tails) is recorded into
    _meta/deploy_repair.json so the Panel can show 'auto-fixed via
    swap_engine_sglang' and the operator can trust the run despite
    the divergence from the original plan.

    Repair is skipped (enable_repair=False) by tests that want to
    exercise the failure path directly.
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
            # PR#31: STAGE_MODEL runs immediately before DEPLOY in
            # STAGES_IN_ORDER and is responsible for ensuring the
            # weights are on disk (snapshot_download into model_cache_root).
            # If DEPLOY still sees a missing path here, the pipeline
            # ordering was bypassed (resume from old state, tests
            # constructing runs manually, etc.) — fail hard with a
            # diagnostic rather than the old `model_missing` graceful
            # skip, which silently masked the bug PR#30 exposed.
            raise StagePyError(
                f"DEPLOY found no model at {model_host_path}; expected "
                f"STAGE_MODEL stage to have staged it. Either the stage "
                f"was skipped or the cache layout changed.",
                kind="model_missing_after_stage",
            )

        cname = container_name_for(run.run_id, engine)
        image = _ENGINE_IMAGES[engine]
        port = cfg.vllm_port  # one port for now; future PRs may multiplex
        model_in_container = "/model"

        vllm_args = dict(plan.get("vllm_args") or {})
        # Always serve under a fixed name so capability.py can hardcode "evaluated".
        vllm_args.setdefault("served_model_name", "evaluated")
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

        # PR#33: extract the docker-spawn + early-exit check into a
        # reusable helper so the repair loop can call it multiple
        # times with mutated plans/images without copy-pasting all of
        # the device_requests / volumes / labels boilerplate.
        def _try_once(active_plan: dict[str, Any], active_image: str,
                      active_engine: str) -> tuple[bool, dict[str, Any]]:
            """Spawn one deploy attempt. Returns (ok, info_dict).

            info_dict on success:
              {"container": <Container>, "engine": engine, "image": image,
               "args": vllm_args, "command": command}
            info_dict on failure (raised exceptions are NOT propagated
            here, they're translated into the failure dict so the
            repair loop can inspect them):
              {"error_kind": str, "error": str, "logs": str,
               "engine": str, "image": str}
            """
            cur_args = dict(active_plan.get("vllm_args") or {})
            cur_args.setdefault("served_model_name", "evaluated")
            _reuse_or_recreate(client, cname, run.run_id, active_engine)
            if active_engine == "vllm":
                cur_cmd = _vllm_command(model_in_container, cur_args, port)
            elif active_engine == "sglang":
                cur_cmd = _sglang_command(model_in_container, cur_args, port)
            else:  # transformers
                cur_cmd = ["serve", "--model-path", model_in_container,
                           "--port", str(port)]
            try:
                container = client.containers.run(
                    active_image,
                    command=cur_cmd,
                    name=cname,
                    detach=True, remove=False,
                    labels={
                        LABEL_RUN: run.run_id,
                        LABEL_STAGE: StageName.DEPLOY.value,
                        LABEL_ENGINE: active_engine,
                    },
                    network_mode="host",
                    ipc_mode="host",
                    shm_size="16g",
                    volumes={
                        str(model_host_path): {
                            "bind": model_in_container, "mode": "ro",
                        },
                    },
                    device_requests=(
                        [
                            docker.types.DeviceRequest(
                                device_ids=[str(i) for i in selected_gpus],
                                capabilities=[["gpu"]],
                            )
                        ] if selected_gpus else []
                    ),
                )
            except ImageNotFound as e:
                return False, {
                    "error_kind": "image_pull",
                    "error": f"image not found: {active_image}: {e}",
                    "logs": "",
                    "engine": active_engine, "image": active_image,
                }
            except APIError as e:
                msg = str(e)
                if "address already in use" in msg or "port is already allocated" in msg:
                    return False, {
                        "error_kind": "port_in_use",
                        "error": f"port {port} already in use: {e}",
                        "logs": "", "engine": active_engine,
                        "image": active_image,
                    }
                return False, {
                    "error_kind": "docker_api",
                    "error": f"docker API error: {e}",
                    "logs": "", "engine": active_engine,
                    "image": active_image,
                }

            sleep(0.5)
            container.reload()
            if container.status in ("exited", "dead"):
                tail = _tail_logs(container, 100)
                try:
                    container.remove(force=True)
                except DockerException:
                    pass
                return False, {
                    "error_kind": "early_exit",
                    "error": f"container exited immediately; "
                             f"tail logs: {tail[:200]!r}",
                    "logs": tail,
                    "engine": active_engine, "image": active_image,
                }
            return True, {
                "container": container, "engine": active_engine,
                "image": active_image, "args": cur_args, "command": cur_cmd,
            }

        # ── first attempt ──
        ok, info = _try_once(plan, image, engine)
        repair_log: dict[str, Any] | None = None

        # ── PR#33 repair loop ──
        if not ok and enable_repair and info["error_kind"] in (
                "early_exit", "image_pull", "docker_api"):
            from . import deploy_repair as dr

            failure = dr.DeployFailure(
                engine=info["engine"], image=info["image"],
                error_kind=info["error_kind"], logs=info["logs"],
            )
            fcls = failure.classify()
            print(f"  [DEPLOY] first attempt failed: {info['error_kind']} "
                  f"({fcls}); engaging deploy_repair")

            attempts_record: list[dict[str, Any]] = [{
                "strategy": "(initial)",
                "engine": info["engine"], "image": info["image"],
                "ok": False, "error_kind": info["error_kind"],
                "error": info["error"][:500],
                "logs_tail": info["logs"][-500:],
            }]
            proposals = dr.propose_attempts(plan, failure)
            print(f"  [DEPLOY] repair proposed {len(proposals)} strategies: "
                  f"{[p.name for p in proposals]}")

            winning: dr.StrategyResult | None = None
            for proposal in proposals:
                cand_plan = proposal.new_plan or plan
                cand_engine = (cand_plan.get("engine") or engine).lower()
                cand_image = proposal.new_image or _ENGINE_IMAGES.get(
                    cand_engine, image)
                t_strat = time.time()
                ok2, info2 = _try_once(cand_plan, cand_image, cand_engine)
                attempts_record.append({
                    "strategy": proposal.name,
                    "engine": cand_engine, "image": cand_image,
                    "notes": proposal.notes,
                    "duration_s": round(time.time() - t_strat, 1),
                    "ok": ok2,
                    "error_kind": info2.get("error_kind") if not ok2 else None,
                    "error": (info2.get("error") or "")[:500] if not ok2 else None,
                    "logs_tail": (info2.get("logs") or "")[-500:] if not ok2 else None,
                })
                if ok2:
                    winning = proposal
                    info = info2
                    plan = cand_plan
                    engine = cand_engine
                    image = cand_image
                    ok = True
                    break

            # ── PR#33 LLM-agent escalation ──
            # If rule-based strategies all failed, ask the MiniMax-M2.7
            # judge for a free-form proposal. The agent gets the
            # failure logs + curated metadata + every previous attempt
            # and is REQUIRED to propose a non-no-op change. We try
            # up to `cfg.deploy_repair_agent_attempts` agent proposals
            # in series.
            agent_summary: dict[str, Any] | None = None
            if not ok and cfg.deploy_repair_agent_enabled:
                agent_summary = _attempt_agent_repair(
                    rd=rd,
                    cfg=cfg,
                    hf_id=run.hf_id,
                    failure=failure,
                    engine=engine,
                    image=image,
                    plan=plan,
                    attempts_record=attempts_record,
                    try_once=_try_once,
                )
                if agent_summary and agent_summary.get("ok"):
                    info = agent_summary["info"]
                    plan = agent_summary["plan"]
                    engine = agent_summary["engine"]
                    image = agent_summary["image"]
                    ok = True
                    winning = dr.StrategyResult(
                        name=f"agent:{agent_summary['strategy']}",
                        new_plan=plan, new_image=image,
                        notes=agent_summary.get("diagnosis", "")[:200],
                    )

            repair_log = {
                "stage": "DEPLOY_REPAIR",
                "failure_class": fcls,
                "ok": ok,
                "winning_strategy": winning.name if winning else None,
                "attempts": attempts_record,
                "strategies_proposed": [p.name for p in proposals],
                "agent_escalation": agent_summary,
            }
            _write_artifact(rd / "_meta", "deploy_repair.json", repair_log)

        # ── translate the final outcome ──
        if not ok:
            # Re-raise as StagePyError so the outer try/except records
            # the same shape as pre-PR#33.
            raise StagePyError(
                info["error"], kind=info["error_kind"],
                logs=info.get("logs", ""),
            )

        container = info["container"]
        base_url = f"http://127.0.0.1:{port}"
        deploy_payload = {
            "stage": "DEPLOY",
            "engine": info["engine"],
            "engine_image": info["image"],
            "container_name": cname,
            "base_url": base_url,
            "model_path_host": str(model_host_path),
            "model_path_container": model_in_container,
            "started_at": datetime.now(tz=UTC).isoformat(timespec="seconds"),
        }
        if repair_log is not None:
            deploy_payload["repaired_via"] = repair_log["winning_strategy"]
            deploy_payload["repair_attempts"] = len(repair_log["attempts"])
        _write_artifact(rd, "deploy.json", deploy_payload)
        _write_artifact(rd / "_meta", "deploy.json", deploy_payload)

        artifacts = ["deploy.json", "_meta/deploy.json"]
        if repair_log is not None:
            artifacts.append("_meta/deploy_repair.json")
        return StageResult(
            ok=True,
            duration_s=time.time() - t0,
            artifacts=artifacts,
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


def _attempt_agent_repair(
    *,
    rd: Path,
    cfg: OrchestratorConfig,
    hf_id: str,
    failure: Any,           # deploy_repair.DeployFailure
    engine: str,
    image: str,
    plan: dict[str, Any],
    attempts_record: list[dict[str, Any]],
    try_once: Callable[..., tuple[bool, dict[str, Any]]],
) -> dict[str, Any] | None:
    """PR#33 LLM-agent escalation. Returns a summary dict that includes
    ``ok`` and (on success) the winning info/plan/engine/image so the
    caller can install it as if a rule-based strategy had won. Always
    returns a dict so the repair_log can record the escalation
    transparently (proposals_seen, why_each_failed, etc.) — even when
    the agent itself failed to propose anything usable.
    """
    try:
        from heyi_engine import HeyiEngineClient
        from cc_agent import deploy_repair_agent as agent_mod
    except Exception as e:  # pragma: no cover — env misconfig only
        return {
            "ok": False,
            "phase": "import",
            "error": f"could not import LLM-agent dependencies: {e}",
            "proposals": [],
        }

    # Load curated context (best effort — agent works with thinner data).
    curated = {}
    modelcard = ""
    try:
        cpath = rd / "_meta" / "curated.json"
        if cpath.exists():
            curated = json.loads(cpath.read_text(encoding="utf-8"))
        mpath = rd / "_meta" / "modelcard.md"
        if mpath.exists():
            modelcard = mpath.read_text(encoding="utf-8")
    except OSError:
        pass

    client = HeyiEngineClient(
        base_url=cfg.engine_url, api_key=cfg.engine_api_key,
    )

    proposals_log: list[dict[str, Any]] = []
    max_attempts = int(getattr(cfg, "deploy_repair_agent_attempts", 3))
    cur_engine, cur_image, cur_plan = engine, image, plan
    for attempt_i in range(max_attempts):
        t0 = time.time()
        # Build previous_attempts for the agent: rule-based attempts +
        # any prior agent proposals that actually got EXECUTED
        # (parse-error / engine-error rows are skipped because they
        # never proposed a concrete strategy to mark as 'tried').
        agent_priors = []
        for p in proposals_log:
            proposal = p.get("proposal")
            if not isinstance(proposal, dict):
                continue
            agent_priors.append({
                "strategy": proposal.get("strategy") or "agent_freeform",
                "engine": proposal.get("engine") or cur_engine,
                "image": proposal.get("image") or cur_image,
                "ok": p.get("ok", False),
                "error_kind": p.get("error_kind"),
            })
        try:
            prop = agent_mod.propose_repair(
                hf_id=hf_id, curated=curated, modelcard=modelcard,
                failure_class=failure.classify(),
                log_tail=failure.logs,
                engine_plan=cur_plan,
                previous_attempts=attempts_record + agent_priors,
                client=client,
                judge_model_name=getattr(cfg, "judge_model_name", "MiniMax-M2.7"),
                timeout_s=float(getattr(cfg, "deploy_repair_agent_timeout_s", 90.0)),
            )
        except Exception as e:
            proposals_log.append({
                "attempt": attempt_i + 1,
                "ok": False,
                "phase": "propose",
                "error": f"{type(e).__name__}: {e}",
                "duration_s": round(time.time() - t0, 1),
            })
            continue

        if not prop.ok:
            proposals_log.append({
                "attempt": attempt_i + 1,
                "ok": False,
                "phase": "propose",
                "error": prop.error,
                "raw_response_head": (prop.raw_response or "")[:300],
                "duration_s": round(time.time() - t0, 1),
            })
            continue

        # Build candidate plan + image + engine from proposal.
        cand_plan = dict(cur_plan)
        if prop.vllm_args:
            new_args = dict(cur_plan.get("vllm_args") or {})
            new_args.update(prop.vllm_args)
            cand_plan["vllm_args"] = new_args
        if prop.engine:
            cand_plan["engine"] = prop.engine
        cand_engine = (cand_plan.get("engine") or cur_engine).lower()
        cand_image = prop.image or _ENGINE_IMAGES.get(cand_engine, cur_image)

        # Run the candidate.
        t_run = time.time()
        ok2, info2 = try_once(cand_plan, cand_image, cand_engine)
        prop_summary = {
            "strategy": prop.strategy,
            "engine": prop.engine,
            "image": prop.image,
            "vllm_args": prop.vllm_args,
            "diagnosis": prop.diagnosis[:300],
            "rationale": prop.rationale[:300],
        }
        proposals_log.append({
            "attempt": attempt_i + 1,
            "ok": ok2,
            "phase": "execute",
            "proposal": prop_summary,
            "propose_duration_s": round(time.time() - t0 - (time.time() - t_run), 1),
            "execute_duration_s": round(time.time() - t_run, 1),
            "error_kind": info2.get("error_kind") if not ok2 else None,
            "error": (info2.get("error") or "")[:500] if not ok2 else None,
            "logs_tail": (info2.get("logs") or "")[-500:] if not ok2 else None,
        })
        # Also append to the outer attempts_record so any further
        # agent calls see this as already-tried.
        attempts_record.append({
            "strategy": f"agent:{prop.strategy}",
            "engine": cand_engine, "image": cand_image,
            "notes": prop.rationale[:200],
            "duration_s": round(time.time() - t_run, 1),
            "ok": ok2,
            "error_kind": info2.get("error_kind") if not ok2 else None,
            "error": (info2.get("error") or "")[:500] if not ok2 else None,
            "logs_tail": (info2.get("logs") or "")[-500:] if not ok2 else None,
        })

        if ok2:
            return {
                "ok": True,
                "strategy": prop.strategy,
                "diagnosis": prop.diagnosis,
                "rationale": prop.rationale,
                "info": info2,
                "plan": cand_plan,
                "engine": cand_engine,
                "image": cand_image,
                "proposals": proposals_log,
                "winning_attempt": attempt_i + 1,
            }
        # Update "current" to the just-tried so next agent proposal
        # sees fresh failure context.
        cur_plan, cur_engine, cur_image = cand_plan, cand_engine, cand_image
        # Update failure object's logs for next iteration so the agent
        # diagnoses the NEW error, not the original.
        # (failure is a frozen dataclass; rebuild it.)
        import dataclasses
        failure = dataclasses.replace(
            failure,
            engine=cand_engine, image=cand_image,
            error_kind=info2.get("error_kind") or failure.error_kind,
            logs=info2.get("logs") or failure.logs,
        )

    return {
        "ok": False,
        "phase": "exhausted",
        "proposals": proposals_log,
        "max_attempts": max_attempts,
    }


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
