"""
Orchestrator configuration — values that change per environment go here.

Defaults match the nv8 production layout. Local dev / smoke tests override via
env vars (HEYI_EVAL_*) or by constructing OrchestratorConfig directly.
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

    # heyi_engine (local LLM, v10). Replaces v9's CCR layer. The model
    # name is auto-discovered from /v1/models — no hardcoded ccr_model.
    engine_url: str = os.environ.get("HEYI_ENGINE_URL", "http://127.0.0.1:10814")
    engine_api_key: str | None = os.environ.get("HEYI_ENGINE_API_KEY")

    # v9 back-compat (only read by legacy code paths; v10 prefers engine_url).
    # ccr_url defaults to engine_url so any stale `cfg.ccr_url` accesses still
    # land on the working endpoint.
    ccr_url: str = os.environ.get(
        "HEYI_EVAL_CCR_URL",
        os.environ.get("HEYI_ENGINE_URL", "http://127.0.0.1:10814"),
    )
    ccr_apikey: str = os.environ.get("HEYI_EVAL_CCR_APIKEY", "")
    ccr_model: str = os.environ.get(
        "HEYI_EVAL_CCR_MODEL", "claude-sonnet-4-20250514",
    )
    # Curator's LLM choice — v10 ignores this (auto-discovered) but the
    # field is kept so v9 callers don't fail with AttributeError. Default
    # changed to empty string to make it obvious nobody should be reading it.
    curator_llm_model: str = os.environ.get("HEYI_EVAL_CURATOR_MODEL", "")

    # HF mirror endpoint
    hf_endpoint: str = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")

    # cc-agent image
    cc_agent_image: str = os.environ.get("HEYI_EVAL_CC_AGENT_IMAGE", "heyi-eval/cc-agent:v9")
    # Cap claude --print turns. 100 is too generous — caught during T11 dry
    # run on nv8 where the OOM-and-retry loop kept claude running for 6+ min
    # past task.md's stated 180s deadline. 40 is enough for a healthy
    # 5-question CAPABILITY stage and the typical DEPLOY plan-deploy-poll
    # sequence; SHOWCASE may want more (overridable via env).
    cc_agent_max_turns: int = int(os.environ.get("HEYI_EVAL_CC_AGENT_MAX_TURNS", "40"))

    # vllm provider
    model_cache_root: Path = field(
        default_factory=lambda: _env_path("HEYI_EVAL_MODEL_CACHE", "/DATA/Model/_eval-cache")
    )
    vllm_port: int = int(os.environ.get("HEYI_EVAL_VLLM_PORT", "18200"))
    vllm_container: str = os.environ.get("HEYI_EVAL_VLLM_CONTAINER", "e8-vllm")

    # docker
    docker_socket: str = os.environ.get("HEYI_EVAL_DOCKER_SOCK", "/var/run/docker.sock")
    # If set, mount the docker CLI binary from host to avoid installing in image.
    host_docker_bin: str | None = os.environ.get("HEYI_EVAL_HOST_DOCKER_BIN", "/usr/bin/docker")

    # stage timeouts (wall-clock seconds; CC turn limits enforce earlier internally)
    deploy_timeout_s: int = int(os.environ.get("HEYI_EVAL_DEPLOY_TIMEOUT", "600"))           # 10 min
    capability_timeout_s: int = int(os.environ.get("HEYI_EVAL_CAPABILITY_TIMEOUT", "900"))   # 15 min
    showcase_timeout_s: int = int(os.environ.get("HEYI_EVAL_SHOWCASE_TIMEOUT", "3600"))      # 60 min
    cleanup_timeout_s: int = int(os.environ.get("HEYI_EVAL_CLEANUP_TIMEOUT", "300"))         #  5 min

    @property
    def runs_dir(self) -> Path:
        return self.data_root / "runs"

    @property
    def handbook_path(self) -> Path:
        return self.repo_root / "cc-agent" / "handbooks" / "handbook.md"

    @property
    def tasks_dir(self) -> Path:
        return self.repo_root / "cc-agent" / "tasks"

    @property
    def schema_root(self) -> Path:
        return self.repo_root / "sops" / "schemas"

    def task_md(self, stage_name: str) -> Path:
        # stage names are uppercased; task files are lowercased
        return self.tasks_dir / f"{stage_name.lower()}.md"

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
