"""PR#10 concept split — production LLM vs evaluation LLM.

v10 prior to PR#10 conflated two distinct things:
  - "production LLM"     : the vLLM container that heyi_engine talks to on
                            :10814; default model MiniMax-M2.7, occasionally
                            switched (every few months) to Kimi-K2.6, etc.
                            Owns GPU 0-3 (or 0-7 transiently when ops loads
                            a larger TP=8 model). User-managed.
  - "evaluation LLM"     : the `e9-*` container that the eval pipeline spawns
                            during DEPLOY for a single run, runs CAPABILITY /
                            SHOWCASE on, then tears down. Owns GPU 4-7. Pipeline-
                            managed.

PR#10 separates the two so that switching production from MiniMax-M2.7 to
Kimi-K2.6 (and back) does not require any code change — only env var
overrides.

See docs/PR10_TEST_PLAN.md §3 for the full case matrix.
"""
from __future__ import annotations

import subprocess
import warnings
from typing import Any

import pytest

from orchestrator.config import OrchestratorConfig
from orchestrator.validator import ValidationError, assert_invariants

# ── §A. Config surface (C1-C5) ─────────────────────────────────────────────


class _EnvScope:
    """Helper: snapshot of HEYI_EVAL_* env keys, restored on __exit__.

    We don't use pytest's monkeypatch.setenv on every key because the
    config dataclass reads os.environ at field-default-time, not at
    instantiation time. Tests therefore patch os.environ + reinstantiate.
    """

    _KEYS = (
        "HEYI_EVAL_PROD_ENGINE_CONTAINER",
        "HEYI_EVAL_PROD_ENGINE_GPUS",
        "HEYI_EVAL_PROD_ENGINE_MIN_GPU_MIB",
        "HEYI_EVAL_EVAL_GPUS",
    )


def _make_cfg_with_env(monkeypatch: pytest.MonkeyPatch, env: dict[str, str]) -> OrchestratorConfig:
    """Construct an OrchestratorConfig with the given env vars taking effect.

    The dataclass uses `field(default_factory=...)` for env-derived values,
    so env changes BEFORE instantiation are picked up. We patch the env then
    instantiate.
    """
    for key in _EnvScope._KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return OrchestratorConfig()


def test_c1_defaults_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """C1: with no env vars set, defaults match the steady state on nv8.

    PR#23 (2026-05) shrank the eval pool default from (4,5,6,7) to
    (5,6,7) because GPU 4 on nv8 holds the ComfyUI host process; see
    rules/42-heyi-m27-api.md + orchestrator/config.py::eval_gpus.
    Production (MiniMax-M2.7 TP=4) still owns (0,1,2,3).
    """
    cfg = _make_cfg_with_env(monkeypatch, {})
    assert cfg.prod_engine_container == "minimax"
    assert cfg.prod_engine_gpus == (0, 1, 2, 3)
    assert cfg.prod_engine_min_gpu_mib == 80_000
    assert cfg.eval_gpus == (5, 6, 7)


def test_c2_switch_to_kimi_k26_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    """C2: production temporarily switched to Kimi-K2.6 TP=8 across all 8
    GPUs. Operator should only need two env overrides — no code change."""
    cfg = _make_cfg_with_env(
        monkeypatch,
        {
            "HEYI_EVAL_PROD_ENGINE_CONTAINER": "kimi-k26",
            "HEYI_EVAL_PROD_ENGINE_GPUS": "0,1,2,3,4,5,6,7",
        },
    )
    assert cfg.prod_engine_container == "kimi-k26"
    assert cfg.prod_engine_gpus == (0, 1, 2, 3, 4, 5, 6, 7)
    # defaults for the eval pool are (5,6,7) (PR#23); PR#11 graceful-
    # skip logic will see prod_engine_gpus overlapping eval_gpus and
    # abort.
    assert cfg.eval_gpus == (5, 6, 7)


def test_c3_eval_gpu_pool_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """C3: separate machine with only GPU 2-3 free for eval."""
    cfg = _make_cfg_with_env(monkeypatch, {"HEYI_EVAL_EVAL_GPUS": "2,3"})
    assert cfg.eval_gpus == (2, 3)


def test_c4_invalid_gpu_token_is_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """C4: non-numeric token in the GPU list → ValueError that names the
    bad token AND the env var, so the operator can fix it immediately."""
    with pytest.raises(ValueError) as exc_info:
        _make_cfg_with_env(monkeypatch, {"HEYI_EVAL_PROD_ENGINE_GPUS": "0,abc,3"})
    msg = str(exc_info.value)
    assert "abc" in msg, f"error must name the bad token, got: {msg!r}"
    assert "HEYI_EVAL_PROD_ENGINE_GPUS" in msg, (
        f"error must name the env var so operator can fix it, got: {msg!r}"
    )


def test_c5_empty_gpu_list_yields_empty_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    """C5: ",," and "" both parse to (). Represents "this side has no GPU
    to offer right now"; PR#11 graceful-skip path will consume this."""
    cfg = _make_cfg_with_env(monkeypatch, {"HEYI_EVAL_EVAL_GPUS": ",,"})
    assert cfg.eval_gpus == ()

    cfg2 = _make_cfg_with_env(monkeypatch, {"HEYI_EVAL_EVAL_GPUS": ""})
    assert cfg2.eval_gpus == ()


# ── §B. Validator parametrization (V1-V5) ───────────────────────────────────


class _FakeRun:
    """Records each subprocess.run call so tests can assert arg lists,
    and returns canned stdout/stderr per command pattern."""

    def __init__(self, responses: dict[str, tuple[int, str, str]]) -> None:
        # key = first non-flag token in argv (e.g. "docker", "nvidia-smi")
        #   OR the literal docker subcommand for finer-grained dispatch
        #   (e.g. "docker:inspect:minimax").
        # value = (returncode, stdout, stderr)
        self.responses = responses
        self.calls: list[list[str]] = []

    def __call__(
        self,
        argv: list[str],
        capture_output: bool = False,
        text: bool = False,
        timeout: float | None = None,
        check: bool = False,
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(argv))
        rc, out, err = self._dispatch(argv)
        return subprocess.CompletedProcess(
            args=argv, returncode=rc, stdout=out, stderr=err
        )

    def _dispatch(self, argv: list[str]) -> tuple[int, str, str]:
        if not argv:
            return (0, "", "")
        if argv[0] == "docker" and len(argv) >= 2 and argv[1] == "inspect":
            container = argv[-1]
            key = f"docker:inspect:{container}"
            if key in self.responses:
                return self.responses[key]
        if argv[0] in self.responses:
            return self.responses[argv[0]]
        return (1, "", f"unmocked argv: {argv!r}")


def _nvidia_smi_csv(gpu_mem: dict[int, int]) -> str:
    return "\n".join(f"{idx}, {mib}" for idx, mib in sorted(gpu_mem.items()))


def _patch_subprocess_run(monkeypatch: pytest.MonkeyPatch, fake: _FakeRun) -> None:
    """Patch subprocess.run on the validator module (the only path that
    calls it for these checks)."""
    monkeypatch.setattr("orchestrator.validator.subprocess.run", fake)


def test_v1_default_prod_container_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """V1: default invocation — production is minimax on GPU 0-3, each
    holding ≥80 GB. Should pass silently."""
    fake = _FakeRun(
        responses={
            "docker:inspect:minimax": (0, "running\n", ""),
            "nvidia-smi": (0, _nvidia_smi_csv({0: 89_000, 1: 89_000, 2: 89_000, 3: 89_000}), ""),
        }
    )
    _patch_subprocess_run(monkeypatch, fake)
    assert_invariants()  # no exception
    # Asserts that the inspect call really used "minimax" (default).
    assert any("minimax" in argv for argv in fake.calls), (
        "expected docker inspect minimax to be called with default config"
    )


def test_v2_kimi_k26_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """V2: production was switched to Kimi-K2.6 TP=8. Caller passes the new
    container name + GPU set; validator must inspect the right container
    and check the right GPUs, NOT 'minimax' or 0-3."""
    fake = _FakeRun(
        responses={
            "docker:inspect:kimi-k26": (0, "running\n", ""),
            "nvidia-smi": (
                0,
                _nvidia_smi_csv({i: 89_000 for i in range(8)}),
                "",
            ),
        }
    )
    _patch_subprocess_run(monkeypatch, fake)
    assert_invariants(
        prod_engine_container="kimi-k26",
        prod_engine_gpus=(0, 1, 2, 3, 4, 5, 6, 7),
    )
    # The single docker inspect call must reference kimi-k26 — never minimax.
    inspect_calls = [a for a in fake.calls if a[:2] == ["docker", "inspect"]]
    assert len(inspect_calls) == 1
    assert "kimi-k26" in inspect_calls[0]
    assert not any("minimax" in tok for tok in inspect_calls[0]), (
        "validator must not fall back to inspecting 'minimax' when caller "
        "specified a different prod_engine_container"
    )


def test_v3_container_not_running_sad_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """V3: the configured production container is exited → error message
    must name the container that was checked (kimi-k26), so the operator
    knows which env var to inspect. Must NOT mention 'minimax' (that would
    suggest the default was used)."""
    fake = _FakeRun(
        responses={
            "docker:inspect:kimi-k26": (0, "exited\n", ""),
            "nvidia-smi": (0, _nvidia_smi_csv({i: 89_000 for i in range(8)}), ""),
        }
    )
    _patch_subprocess_run(monkeypatch, fake)
    with pytest.raises(ValidationError) as exc_info:
        assert_invariants(prod_engine_container="kimi-k26")
    msg = str(exc_info.value)
    assert "kimi-k26" in msg, f"error must name the configured container, got: {msg!r}"
    assert "minimax" not in msg, (
        f"error must not reference the default container when a different "
        f"one was configured, got: {msg!r}"
    )


def test_v4_gpu_memory_insufficient_checks_only_configured_gpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V4: caller restricts the check to GPU (4,5). If GPU 4 has too little
    memory the error must name GPU 4. GPU 0-3 having too little memory must
    be IGNORED in this configuration."""
    fake = _FakeRun(
        responses={
            "docker:inspect:custom-prod": (0, "running\n", ""),
            "nvidia-smi": (
                0,
                _nvidia_smi_csv({
                    0: 1_000, 1: 1_000, 2: 1_000, 3: 1_000,  # < threshold; ignored
                    4: 1_000,                                # < threshold; reported
                    5: 89_000,                               # ok
                    6: 89_000, 7: 89_000,                    # not checked
                }),
                "",
            ),
        }
    )
    _patch_subprocess_run(monkeypatch, fake)
    with pytest.raises(ValidationError) as exc_info:
        assert_invariants(
            prod_engine_container="custom-prod",
            prod_engine_gpus=(4, 5),
        )
    msg = str(exc_info.value)
    assert "GPU 4" in msg, f"error must point at GPU 4, got: {msg!r}"


def test_v5_legacy_kwargs_still_work_with_deprecation_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """V5: backward-compat — code that still calls
    ``assert_invariants(minimax_gpus=...)`` must keep working but emit a
    DeprecationWarning so the caller knows to migrate."""
    fake = _FakeRun(
        responses={
            "docker:inspect:minimax": (0, "running\n", ""),
            "nvidia-smi": (0, _nvidia_smi_csv({0: 89_000, 1: 89_000, 2: 89_000, 3: 89_000}), ""),
        }
    )
    _patch_subprocess_run(monkeypatch, fake)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert_invariants(minimax_gpus=(0, 1, 2, 3))
    deprecation_msgs = [
        str(w.message) for w in caught if issubclass(w.category, DeprecationWarning)
    ]
    assert any("minimax_gpus" in m for m in deprecation_msgs), (
        f"expected a DeprecationWarning mentioning 'minimax_gpus', got: {deprecation_msgs!r}"
    )


# ── §C. Docs/data compliance (D1-D2) ────────────────────────────────────────


def test_d1_client_docstring_abstract_not_model_enumeration() -> None:
    """D1: heyi_engine/client.py module docstring must NOT hard-code the
    'minimax / glm-51 / kimi-k26' enumeration (which made readers think
    those are the only possible production models). It MUST mention the
    abstract concept (prod_engine_container or "production LLM")."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "heyi_engine" / "client.py"
    text = path.read_text(encoding="utf-8")

    forbidden = ("minimax / glm-51 / kimi-k26", "minimax/glm-51/kimi-k26")
    for substr in forbidden:
        assert substr not in text, (
            f"client.py docstring must not enumerate specific production models "
            f"({substr!r}) — use prod_engine_container abstraction instead"
        )
    required_any = ("prod_engine_container", "production LLM")
    assert any(s in text for s in required_any), (
        "client.py docstring must reference the abstraction "
        "(prod_engine_container or 'production LLM')"
    )


def test_d2_invariants_doc_separates_two_layers() -> None:
    """D2: docs/INVARIANTS.md must contain a section that distinguishes
    'production LLM' from 'evaluation LLM'. INV-2's description must
    reference the prod_engine_container config rather than a bare
    'minimax-*' literal as the sole identifier."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "docs" / "INVARIANTS.md"
    text = path.read_text(encoding="utf-8")

    assert "产线 LLM" in text and "评估 LLM" in text, (
        "docs/INVARIANTS.md must have both '产线 LLM' and '评估 LLM' "
        "headings — the two-layer concept split must be documented"
    )
    # INV-2 row no longer relies solely on the literal 'minimax-*' as the
    # canonical container name. The config knob must be referenced.
    assert "prod_engine_container" in text, (
        "docs/INVARIANTS.md INV-2 description must reference "
        "OrchestratorConfig.prod_engine_container"
    )


# ── §D. Regression hook (R5) ───────────────────────────────────────────────


def test_r5_imports_still_clean() -> None:
    """R5: the refactored modules expose the new surface and still expose
    the legacy surface where required. Read-only check — no module
    reloads, which would invalidate ValidationError class identity for
    other tests."""
    import orchestrator.config as config_mod
    import orchestrator.validator as validator_mod

    cfg = config_mod.OrchestratorConfig()
    assert hasattr(cfg, "prod_engine_container")
    assert hasattr(cfg, "prod_engine_gpus")
    assert hasattr(cfg, "prod_engine_min_gpu_mib")
    assert hasattr(cfg, "eval_gpus")
    assert callable(validator_mod.assert_invariants)
    assert issubclass(validator_mod.ValidationError, Exception)
