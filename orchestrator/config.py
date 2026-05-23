"""
Orchestrator configuration — values that change per environment go here.

Defaults match the nv8 production layout. Local dev / smoke tests override via
env vars (HEYI_EVAL_*) or by constructing OrchestratorConfig directly.

After PR#7a: all v9 back-compat fields are gone. The single LLM endpoint
is ``engine_url`` (heyi_engine on :10814) and every stage talks to docker
via ``stages_py`` using the docker-py SDK — no more cc-agent docker spawn
path, no more LLM translation proxy. See ``docs/PR7a_TEST_PLAN.md``.

After PR#10: the production LLM container name and the GPUs it occupies
are configurable instead of hard-coded. The user's production model
changes every few months (default MiniMax-M2.7 on GPU 0-3; transient
Kimi-K2.6 TP=8 occupies 0-7). Eval-side pipelines must adapt without
code changes. See ``docs/PR10_TEST_PLAN.md``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


def _parse_gpu_tuple(env_name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """Parse a comma-separated GPU index list from an env var.

    Behaviour:
    - env var unset → returns ``default``.
    - env var set to empty / only whitespace / only commas → returns ``()``
      (meaningful: "this side has no GPU to offer"; PR#11 graceful skip
      consumes this).
    - any non-empty token that isn't a non-negative int → ``ValueError``
      naming the bad token AND the env var so the operator can fix it
      without grepping source.
    """
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    tokens = [t.strip() for t in raw.split(",")]
    out: list[int] = []
    for tok in tokens:
        if tok == "":
            continue
        try:
            value = int(tok)
        except ValueError as e:
            raise ValueError(
                f"{env_name}: cannot parse {tok!r} as a GPU index "
                f"(value was {raw!r}, expected comma-separated non-negative ints)"
            ) from e
        if value < 0:
            raise ValueError(
                f"{env_name}: GPU index {value} is negative "
                f"(value was {raw!r}, expected non-negative ints)"
            )
        out.append(value)
    return tuple(out)


@dataclass
class OrchestratorConfig:
    # data layout
    data_root: Path = field(default_factory=lambda: _env_path("HEYI_EVAL_DATA", "~/heyi-eval-data"))
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    # PR#6: out-of-tree snapshot root. Must NOT live under data_root (INV-9).
    # Override via HEYI_EVAL_BACKUPS for test/dev environments.
    backups_root: Path = field(
        default_factory=lambda: _env_path("HEYI_EVAL_BACKUPS", "~/heyi-eval-backups")
    )

    # heyi_engine (local LLM, v10). On nv8 this is the MiniMax-M2.7
    # vLLM container `minimax` serving on port 10814 (TP=4, GPU 0-3).
    # Three reachable paths per `rules/42-heyi-m27-api.md`:
    #   (A) nv8-local       http://127.0.0.1:10814    (this default)
    #   (B) Tailscale       http://100.127.173.85:10814
    #   (C) trycloudflare   read /home/ai/cf-m27-url.txt (URL is dynamic)
    # When you're on the mac dev box, export HEYI_ENGINE_URL to (B) or (C).
    engine_url: str = os.environ.get("HEYI_ENGINE_URL", "http://127.0.0.1:10814")
    engine_api_key: str | None = os.environ.get("HEYI_ENGINE_API_KEY")
    # PR#23: pin the model name. M2.7 vLLM serves model
    # "MiniMax-M2.7" and previously llm_judge.py hard-coded "auto"
    # (which vLLM accepts as "first registered model" but breaks the
    # day someone serves two models on the same endpoint). Pinned to
    # match the rules/42-heyi-m27-api.md contract.
    judge_model_name: str = os.environ.get("HEYI_EVAL_JUDGE_MODEL", "MiniMax-M2.7")

    # PR#33: DEPLOY auto-repair (rule-based + LLM-agent escalation).
    # When enabled, a DEPLOY failure with one of
    # {early_exit, image_pull, docker_api} triggers `deploy_repair`
    # which walks rule-based strategies first, then (if exhausted)
    # asks the judge LLM for free-form proposals. Each proposal is
    # validated + sandboxed (no docker access; agent only emits JSON).
    # Disable via env HEYI_EVAL_DEPLOY_REPAIR_AGENT=0.
    deploy_repair_agent_enabled: bool = (
        os.environ.get("HEYI_EVAL_DEPLOY_REPAIR_AGENT", "1") != "0"
    )
    deploy_repair_agent_attempts: int = int(
        os.environ.get("HEYI_EVAL_DEPLOY_REPAIR_AGENT_ATTEMPTS", "3")
    )
    deploy_repair_agent_timeout_s: float = float(
        os.environ.get("HEYI_EVAL_DEPLOY_REPAIR_AGENT_TIMEOUT_S", "90")
    )

    # PR#34c: wall-clock budget for snapshot_download inside STAGE_MODEL.
    # Default 30 min — long enough for legitimate ~50 GB downloads on the
    # hf-mirror, short enough to surface a CLOSE-WAIT hang as a real
    # failure instead of pinning the queue. Set 0 to disable the budget.
    stage_model_download_timeout_s: float = float(
        os.environ.get("HEYI_EVAL_STAGE_MODEL_TIMEOUT_S", "1800")
    )

    # HF mirror endpoint
    hf_endpoint: str = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

    # vllm provider
    model_cache_root: Path = field(
        default_factory=lambda: _env_path("HEYI_EVAL_MODEL_CACHE", "/DATA/Model/_eval-cache")
    )
    vllm_port: int = int(os.environ.get("HEYI_EVAL_VLLM_PORT", "18200"))

    # production LLM (the vLLM container that heyi_engine talks to on :10814).
    # The default container name 'minimax' matches the nv8 steady state
    # (MiniMax-M2.7, TP=4, GPU 0-3). When ops transiently switches to a larger
    # model (e.g. Kimi-K2.6 TP=8) the operator overrides these env vars; no
    # code change is required.
    prod_engine_container: str = field(
        default_factory=lambda: os.environ.get(
            "HEYI_EVAL_PROD_ENGINE_CONTAINER", "minimax"
        )
    )
    prod_engine_gpus: tuple[int, ...] = field(
        default_factory=lambda: _parse_gpu_tuple(
            "HEYI_EVAL_PROD_ENGINE_GPUS", (0, 1, 2, 3)
        )
    )
    prod_engine_min_gpu_mib: int = field(
        default_factory=lambda: int(
            os.environ.get("HEYI_EVAL_PROD_ENGINE_MIN_GPU_MIB", "80000")
        )
    )

    # evaluation-side GPU pool (the eval pipeline spawns e9-* containers
    # restricted to these GPU indices; PR#11 wires the actual injection).
    #
    # PR#23 (2026-05) shrinks default from (4,5,6,7) → (5,6,7) because
    # GPU 4 on nv8 is occupied by the ComfyUI host process
    # (`python main.py --port 8188`, ~93 GB). Production (M2.7) holds
    # (0,1,2,3) and the eval pipeline must not touch GPU 4. Setting
    # HEYI_EVAL_EVAL_GPUS="" still activates PR#11 graceful-skip.
    # Any model whose tensor_parallel_size > len(eval_gpus)=3 is now
    # gated at ENGINE_SELECT (PR#23 oversize gate) and aborted with
    # metadata-only — see orchestrator/stages.py + INV-23.
    eval_gpus: tuple[int, ...] = field(
        default_factory=lambda: _parse_gpu_tuple(
            "HEYI_EVAL_EVAL_GPUS", (5, 6, 7)
        )
    )

    # docker
    docker_socket: str = os.environ.get("HEYI_EVAL_DOCKER_SOCK", "/var/run/docker.sock")

    # stage timeouts (wall-clock seconds)
    deploy_timeout_s: int = int(os.environ.get("HEYI_EVAL_DEPLOY_TIMEOUT", "600"))              # 10 min
    capability_timeout_s: int = int(os.environ.get("HEYI_EVAL_CAPABILITY_TIMEOUT", "900"))      # 15 min
    perf_bench_timeout_s: int = int(os.environ.get("HEYI_EVAL_PERF_BENCH_TIMEOUT", "300"))      #  5 min
    showcase_timeout_s: int = int(os.environ.get("HEYI_EVAL_SHOWCASE_TIMEOUT", "3600"))         # 60 min
    cleanup_timeout_s: int = int(os.environ.get("HEYI_EVAL_CLEANUP_TIMEOUT", "300"))            #  5 min

    @property
    def runs_dir(self) -> Path:
        return self.data_root / "runs"

    @property
    def schema_root(self) -> Path:
        return self.repo_root / "sops" / "schemas"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    def hf_local_dir(self, hf_id: str) -> str:
        """
        Translate "Org/Model-Name" → "Model-Name". The local convention is to
        store models under <model_cache_root>/<basename(hf_id)>/.
        """
        return hf_id.split("/", 1)[-1] if "/" in hf_id else hf_id

    def timeout_for(self, stage_name: str) -> int:
        return {
            "DEPLOY": self.deploy_timeout_s,
            "CAPABILITY": self.capability_timeout_s,
            "PERF_BENCH": self.perf_bench_timeout_s,
            "SHOWCASE": self.showcase_timeout_s,
            "CLEANUP": self.cleanup_timeout_s,
        }.get(stage_name, 600)
