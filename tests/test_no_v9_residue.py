"""INV-11 static check: no v9 CCR / cc-agent docker symbols left in code.

After PR#7a the orchestrator talks to heyi_engine over plain HTTP and
every stage runs in-process or via docker-py. The remaining test surface
guards against accidental re-introduction of any of:

  - CCR proxy config: ccr_url / ccr_apikey / ccr_model / curator_llm_model
  - cc-agent docker spawn: _docker_run_cc_agent / _execute_cc_stage / _CC_STAGES
  - cc-agent docker image: cc_agent_image / cc_agent_max_turns / host_docker_bin
  - panel CCR probe: ccr_probe / call_ccr_messages
  - panel v8 container watch: vllm_container = "e8-*"
  - env vars: HEYI_EVAL_CCR_URL / HEYI_EVAL_CCR_APIKEY / HEYI_EVAL_CCR_MODEL /
              HEYI_EVAL_CC_AGENT_IMAGE / HEYI_EVAL_CC_AGENT_MAX_TURNS /
              HEYI_EVAL_HOST_DOCKER_BIN / HEYI_EVAL_CURATOR_MODEL

`docs/` is allowed to mention these as historical context; the check is
limited to .py / .yaml / .sh / systemd unit files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN_SYMBOLS = [
    "ccr_url",
    "ccr_apikey",
    "ccr_model",
    "curator_llm_model",
    "_docker_run_cc_agent",
    "_execute_cc_stage",
    "_CC_STAGES",
    "cc_agent_image",
    "cc_agent_max_turns",
    "host_docker_bin",
    "ccr_probe",
    "call_ccr_messages",
    "HEYI_EVAL_CCR_URL",
    "HEYI_EVAL_CCR_APIKEY",
    "HEYI_EVAL_CCR_MODEL",
    "HEYI_EVAL_CURATOR_MODEL",
    "HEYI_EVAL_CC_AGENT_IMAGE",
    "HEYI_EVAL_CC_AGENT_MAX_TURNS",
    "HEYI_EVAL_HOST_DOCKER_BIN",
    "HEYI_EVAL_VLLM_CONTAINER",
    "e8-vllm",
    "e8-cc-",
    "CcrHealthReport",
]

# Files we know about and accept as legitimate references (docs, the test
# itself, schemas where the old name is documented in a `description`).
ALLOWED_PATHS = {
    "tests/test_no_v9_residue.py",
    # INV-11 guard test asserts the dispatcher module no longer exposes
    # the v9 docker-spawn helpers — it must reference their names by string.
    "tests/test_dispatcher_routes.py",
    # Deploy-side static lint also asserts env.example does NOT contain
    # certain v9 keys — same reason as above.
    "tests/test_systemd_units.py",
    "docs/PLAN.md",
    "docs/_archive/PR3_TEST_PLAN.md",
    "docs/_archive/PR4_TEST_PLAN.md",
    "docs/_archive/PR5_TEST_PLAN.md",
    "docs/_archive/PR6_TEST_PLAN.md",
    "docs/_archive/PR7a_TEST_PLAN.md",
    "sops/known_quirks.md",
    "sops/schemas/ready.schema.json",
    "README.md",
    "AGENTS.md",
}

SCAN_EXTENSIONS = {".py", ".yaml", ".yml", ".sh", ".timer", ".service", ".plist"}


def _scan_targets() -> list[Path]:
    out: list[Path] = []
    for p in REPO_ROOT.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in SCAN_EXTENSIONS:
            continue
        # skip well-known noisy dirs
        rel = p.relative_to(REPO_ROOT)
        parts = rel.parts
        if any(
            part
            in (
                ".venv",
                ".git",
                ".pytest_cache",
                ".mypy_cache",
                ".ruff_cache",
                "__pycache__",
                "heyi_eval_v10.egg-info",
            )
            for part in parts
        ):
            continue
        if str(rel) in ALLOWED_PATHS:
            continue
        out.append(p)
    return out


@pytest.mark.parametrize("symbol", FORBIDDEN_SYMBOLS)
def test_no_v9_residue_for_symbol(symbol: str):
    """Per-symbol parametrization so failure reports tell you exactly
    which v9 ghost is still haunting the repo."""
    hits: list[str] = []
    for p in _scan_targets():
        try:
            text = p.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if symbol in text:
            for line_no, line in enumerate(text.splitlines(), start=1):
                if symbol in line:
                    hits.append(f"  {p.relative_to(REPO_ROOT)}:{line_no}: {line.strip()}")
    assert not hits, f"v9 residue for {symbol!r}:\n" + "\n".join(hits)


def test_cc_agent_dir_is_python_package_not_docker_buildroot():
    """v9 used to ship a top-level `cc-agent/` directory containing the
    Dockerfile + bash entrypoints. v10 replaces it with the Python package
    `cc_agent/` (underscore). Make sure the dash-flavored one stays gone."""
    assert not (REPO_ROOT / "cc-agent").exists(), (
        "v9 cc-agent/ build root resurrected — only cc_agent/ (Python pkg) is allowed"
    )
    pkg = REPO_ROOT / "cc_agent"
    assert pkg.is_dir(), "cc_agent/ package missing"
    assert (pkg / "__init__.py").is_file()
    assert (pkg / "showcase_runner.py").is_file()


def test_config_has_engine_fields_not_ccr():
    """OrchestratorConfig surface check: instantiation works without any
    v9 ccr_* kwargs, and the engine_url / engine_api_key fields exist."""
    from orchestrator.config import OrchestratorConfig

    cfg = OrchestratorConfig()
    assert hasattr(cfg, "engine_url")
    assert hasattr(cfg, "engine_api_key")
    assert not hasattr(cfg, "ccr_url")
    assert not hasattr(cfg, "ccr_apikey")
    assert not hasattr(cfg, "ccr_model")
    assert not hasattr(cfg, "cc_agent_image")
    assert not hasattr(cfg, "host_docker_bin")
    assert not hasattr(cfg, "vllm_container")
    assert not hasattr(cfg, "curator_llm_model")
    # also no leftover cc-agent paths
    assert not hasattr(cfg, "handbook_path")
    assert not hasattr(cfg, "tasks_dir")
    assert not hasattr(cfg, "task_md")
