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

    # heyi_engine (local LLM, v10). The model name is auto-discovered from
    # /v1/models — there is no model-name field here.
    engine_url: str = os.environ.get("HEYI_ENGINE_URL", "http://127.0.0.1:10814")
    engine_api_key: str | None = os.environ.get("HEYI_ENGINE_API_KEY")

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
    # Default (4,5,6,7) is the steady-state complement of prod_engine_gpus.
    # If empty (HEYI_EVAL_EVAL_GPUS=""), PR#11 graceful-skip path activates.
    eval_gpus: tuple[int, ...] = field(
        default_factory=lambda: _parse_gpu_tuple(
            "HEYI_EVAL_EVAL_GPUS", (4, 5, 6, 7)
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
