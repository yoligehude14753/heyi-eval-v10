"""
Orchestrator configuration — values that change per environment go here.

Defaults match the nv8 production layout. Local dev / smoke tests override via
env vars (HEYI_EVAL_*) or by constructing OrchestratorConfig directly.

After PR#7a: all v9 back-compat fields are gone. The single LLM endpoint
is ``engine_url`` (heyi_engine on :10814) and every stage talks to docker
via ``stages_py`` using the docker-py SDK — no more cc-agent docker spawn
path, no more LLM translation proxy. See ``docs/PR7a_TEST_PLAN.md``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


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

    # docker
    docker_socket: str = os.environ.get("HEYI_EVAL_DOCKER_SOCK", "/var/run/docker.sock")

    # stage timeouts (wall-clock seconds)
    deploy_timeout_s: int = int(os.environ.get("HEYI_EVAL_DEPLOY_TIMEOUT", "600"))           # 10 min
    capability_timeout_s: int = int(os.environ.get("HEYI_EVAL_CAPABILITY_TIMEOUT", "900"))   # 15 min
    showcase_timeout_s: int = int(os.environ.get("HEYI_EVAL_SHOWCASE_TIMEOUT", "3600"))      # 60 min
    cleanup_timeout_s: int = int(os.environ.get("HEYI_EVAL_CLEANUP_TIMEOUT", "300"))         #  5 min

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
            "SHOWCASE": self.showcase_timeout_s,
            "CLEANUP": self.cleanup_timeout_s,
        }.get(stage_name, 600)
