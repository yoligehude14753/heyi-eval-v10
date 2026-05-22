"""PR#11 GPU isolation tests.

Production LLM (heyi_engine's vLLM) occupies cfg.prod_engine_gpus (default
0-3). The eval pipeline must spawn its e9-* containers restricted to
cfg.eval_gpus (default 4-7) — never accidentally grab a production GPU.

When the eval pool can't satisfy a request (model TP > pool size, prod
transiently occupies eval pool, or nvidia-smi reports the eval cards
non-idle), DEPLOY exits as a *graceful skip*: StageResult(ok=False,
error_kind="insufficient_gpu", extra={"aborted": True, "reason": ...}).
The run is marked ABORTED, not FAILED, so it doesn't get retried.

See docs/PR11_TEST_PLAN.md §4 for the full matrix.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from orchestrator import notify, stages_py
from orchestrator.config import OrchestratorConfig
from orchestrator.state_machine import Run, RunStatus, StageInfo, StageName, StageStatus

# ── shared fixtures (mirror tests/test_stages_py_deploy.py) ────────────────


def _make_cfg(
    tmp: Path,
    *,
    eval_gpus: tuple[int, ...] = (4, 5, 6, 7),
    prod_engine_gpus: tuple[int, ...] = (0, 1, 2, 3),
) -> OrchestratorConfig:
    cfg = OrchestratorConfig(
        data_root=tmp / "data",
        repo_root=tmp / "repo",
        model_cache_root=tmp / "cache",
        vllm_port=18200,
        eval_gpus=eval_gpus,
        prod_engine_gpus=prod_engine_gpus,
    )
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _make_run(run_id: str = "2026-05-22_001_qwenshort", hf_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Run:
    return Run(run_id=run_id, hf_id=hf_id)


_NOP_SLEEP = lambda _: None  # noqa: E731


def _write_engine_plan(cfg: OrchestratorConfig, run: Run, plan: dict[str, Any]) -> None:
    rd = cfg.run_dir(run.run_id) / "_meta"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "engine.json").write_text(json.dumps(plan), encoding="utf-8")


def _make_model_cache(cfg: OrchestratorConfig, run: Run) -> Path:
    p = cfg.model_cache_root / cfg.hf_local_dir(run.hf_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _FakeContainer:
    def __init__(self, name: str, status: str = "running") -> None:
        self.name = name
        self.status = status
        self.attrs: dict[str, Any] = {"Config": {"Labels": {}}, "Name": f"/{name}"}

    def reload(self) -> None:
        pass

    def logs(self, tail: int = 50) -> bytes:
        return b""

    def remove(self, force: bool = False, v: bool = False) -> None:
        pass


def _fake_docker_client() -> MagicMock:
    client = MagicMock()
    client.ping.return_value = True

    def _get(name: str) -> Any:
        from docker.errors import NotFound
        raise NotFound(f"no such container {name}")

    client.containers.get.side_effect = _get
    client.containers.list.return_value = []
    return client


def _patch_docker(client_mock: MagicMock) -> Any:
    docker_mod = MagicMock()
    docker_mod.from_env.return_value = client_mock
    docker_mod.types.DeviceRequest = MagicMock()
    try:
        from docker import errors as _e
        docker_mod.errors = _e
    except ImportError:
        docker_mod.errors = MagicMock()
    return patch.object(stages_py, "docker", docker_mod)


# ── §A. _select_eval_gpus (G1-G8) ──────────────────────────────────────────


def test_g1_happy_default_pool_all_idle() -> None:
    """G1: cfg eval=(4,5,6,7), prod=(0-3), tp=4, nvidia-smi shows all
    eval cards near-empty → select all four."""
    with TemporaryDirectory() as td:
        cfg = _make_cfg(Path(td))
    selected, reason = stages_py._select_eval_gpus(
        cfg,
        tp_size=4,
        smi_query=lambda: {i: 0 for i in range(8)},
    )
    assert selected == [4, 5, 6, 7]
    assert reason is None


def test_g2_tp_less_than_pool_selects_prefix() -> None:
    """G2: tp=2 picks the first two eval GPUs."""
    with TemporaryDirectory() as td:
        cfg = _make_cfg(Path(td))
    selected, reason = stages_py._select_eval_gpus(
        cfg,
        tp_size=2,
        smi_query=lambda: {i: 0 for i in range(8)},
    )
    assert selected == [4, 5]
    assert reason is None


def test_g3_tp_exceeds_pool_graceful_skip() -> None:
    """G3: tp=8 but pool has only 4 → graceful skip with clear reason."""
    with TemporaryDirectory() as td:
        cfg = _make_cfg(Path(td))
    selected, reason = stages_py._select_eval_gpus(
        cfg,
        tp_size=8,
        smi_query=lambda: {i: 0 for i in range(8)},
    )
    assert selected is None
    assert reason is not None
    assert "tensor_parallel_size=8" in reason
    assert "4" in reason  # pool size


def test_g4_empty_eval_pool() -> None:
    """G4: HEYI_EVAL_EVAL_GPUS='' (or () via constructor) → graceful skip."""
    with TemporaryDirectory() as td:
        cfg = _make_cfg(Path(td), eval_gpus=())
    selected, reason = stages_py._select_eval_gpus(
        cfg,
        tp_size=1,
        smi_query=lambda: {i: 0 for i in range(8)},
    )
    assert selected is None
    assert reason is not None
    assert "eval pool is empty" in reason.lower() or "no eval gpu" in reason.lower()


def test_g5_eval_pool_overlaps_prod_pool() -> None:
    """G5: production temporarily switched to TP=8 (K2.6 occupies 0-7).
    eval=(4,5,6,7), prod=(0,1,2,3,4,5,6,7) → overlap → graceful skip."""
    with TemporaryDirectory() as td:
        cfg = _make_cfg(
            Path(td),
            eval_gpus=(4, 5, 6, 7),
            prod_engine_gpus=(0, 1, 2, 3, 4, 5, 6, 7),
        )
    selected, reason = stages_py._select_eval_gpus(
        cfg,
        tp_size=4,
        smi_query=lambda: {i: 0 for i in range(8)},
    )
    assert selected is None
    assert reason is not None
    assert "overlap" in reason.lower()
    for gpu in (4, 5, 6, 7):
        assert str(gpu) in reason


def test_g6_smi_reports_eval_gpu_busy() -> None:
    """G6: nvidia-smi shows GPU 4 already using 50 GiB (ops manually
    started something) → graceful skip naming the busy GPU."""
    with TemporaryDirectory() as td:
        cfg = _make_cfg(Path(td))
    selected, reason = stages_py._select_eval_gpus(
        cfg,
        tp_size=4,
        smi_query=lambda: {0: 0, 1: 0, 2: 0, 3: 0, 4: 50_000, 5: 0, 6: 0, 7: 0},
    )
    assert selected is None
    assert reason is not None
    assert "GPU 4" in reason
    assert "50" in reason  # MiB value visible somewhere


def test_g7_smi_unavailable_filenotfound() -> None:
    """G7: nvidia-smi binary missing → conservative graceful skip."""
    with TemporaryDirectory() as td:
        cfg = _make_cfg(Path(td))

    def _raise_missing() -> dict[int, int]:
        raise FileNotFoundError("nvidia-smi: command not found")

    selected, reason = stages_py._select_eval_gpus(
        cfg,
        tp_size=4,
        smi_query=_raise_missing,
    )
    assert selected is None
    assert reason is not None
    assert "nvidia-smi" in reason.lower()


def test_g8_smi_timeout() -> None:
    """G8: nvidia-smi hangs → conservative graceful skip."""
    with TemporaryDirectory() as td:
        cfg = _make_cfg(Path(td))

    def _raise_timeout() -> dict[int, int]:
        raise TimeoutError("smi timed out")

    selected, reason = stages_py._select_eval_gpus(
        cfg,
        tp_size=4,
        smi_query=_raise_timeout,
    )
    assert selected is None
    assert reason is not None
    assert "nvidia-smi" in reason.lower()


# ── §B. execute_deploy end-to-end (D1-D4) ──────────────────────────────────


def _all_idle_smi() -> dict[int, int]:
    return {i: 0 for i in range(8)}


def test_d1_happy_tp4_uses_device_ids_4_to_7(monkeypatch: pytest.MonkeyPatch) -> None:
    """D1: TP=4 + eval pool (4,5,6,7) all idle → DeviceRequest gets
    device_ids=['4','5','6','7'] (mapped from cfg.eval_gpus, NOT physical
    GPUs the docker daemon picks at random)."""
    with TemporaryDirectory() as td:
        tmp = Path(td)
        cfg = _make_cfg(tmp)
        run = _make_run()
        _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {"tensor_parallel_size": 4}})
        _make_model_cache(cfg, run)
        expected_name = stages_py.container_name_for(run.run_id, "vllm")
        client = _fake_docker_client()
        client.containers.run.return_value = _FakeContainer(expected_name)

        monkeypatch.setattr(stages_py, "_nvidia_smi_used_mib", _all_idle_smi)
        with _patch_docker(client):
            r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)
            dr_call = stages_py.docker.types.DeviceRequest.call_args

        assert r.ok, r.error
        kwargs = client.containers.run.call_args.kwargs
        dev_reqs = kwargs["device_requests"]
        assert len(dev_reqs) == 1
        assert dr_call.kwargs["device_ids"] == ["4", "5", "6", "7"]
        assert dr_call.kwargs["capabilities"] == [["gpu"]]


def test_d2_tp2_uses_only_first_two_eval_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """D2: tensor_parallel_size=2 → device_ids=['4','5'] only."""
    with TemporaryDirectory() as td:
        tmp = Path(td)
        cfg = _make_cfg(tmp)
        run = _make_run()
        _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {"tensor_parallel_size": 2}})
        _make_model_cache(cfg, run)
        client = _fake_docker_client()
        client.containers.run.return_value = _FakeContainer("e9-vllm-x")

        monkeypatch.setattr(stages_py, "_nvidia_smi_used_mib", _all_idle_smi)
        with _patch_docker(client):
            r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)
            dr_call = stages_py.docker.types.DeviceRequest.call_args

        assert r.ok, r.error
        assert dr_call.kwargs["device_ids"] == ["4", "5"]


def test_d3_graceful_skip_on_tp_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """D3: tp=8 requires more cards than the 4-GPU eval pool → graceful
    skip without ever calling docker.containers.run."""
    with TemporaryDirectory() as td:
        tmp = Path(td)
        cfg = _make_cfg(tmp)
        run = _make_run()
        _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {"tensor_parallel_size": 8}})
        _make_model_cache(cfg, run)
        client = _fake_docker_client()

        monkeypatch.setattr(stages_py, "_nvidia_smi_used_mib", _all_idle_smi)
        with _patch_docker(client):
            r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

    assert r.ok is False
    assert r.error_kind == "insufficient_gpu"
    assert r.extra.get("aborted") is True
    assert "tensor_parallel_size" in (r.extra.get("reason") or "")
    assert client.containers.run.call_count == 0


def test_d4_graceful_skip_on_prod_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    """D4: production transient state — TP=8 occupies 0-7, overlapping the
    default eval pool. DEPLOY must back off without touching docker."""
    with TemporaryDirectory() as td:
        tmp = Path(td)
        cfg = _make_cfg(
            tmp,
            eval_gpus=(4, 5, 6, 7),
            prod_engine_gpus=(0, 1, 2, 3, 4, 5, 6, 7),
        )
        run = _make_run()
        _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {}})
        _make_model_cache(cfg, run)
        client = _fake_docker_client()

        monkeypatch.setattr(stages_py, "_nvidia_smi_used_mib", _all_idle_smi)
        with _patch_docker(client):
            r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

    assert r.ok is False
    assert r.error_kind == "insufficient_gpu"
    assert r.extra.get("aborted") is True
    assert "overlap" in (r.extra.get("reason") or "").lower()
    assert client.containers.run.call_count == 0


# ── §C. state machine integration (S1-S2) ──────────────────────────────────


def test_s1_stage_info_mark_skipped() -> None:
    """S1: StageInfo gains mark_skipped(reason) that sets status=SKIPPED,
    records the reason in .error, and computes duration_s."""
    info = StageInfo(name=StageName.DEPLOY)
    info.mark_started()
    info.mark_skipped("insufficient_gpu: tp=8 but pool=4")
    assert info.status == StageStatus.SKIPPED
    assert info.error is not None
    assert "insufficient_gpu" in info.error
    assert info.duration_s is not None and info.duration_s >= 0


def test_s2_run_pipeline_aborts_on_insufficient_gpu_deploy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """S2: when DEPLOY returns a graceful-skip StageResult, the orchestrator
    must mark the stage SKIPPED, the run ABORTED (not FAILED), record the
    reason, and NOT advance into downstream stages."""
    from orchestrator import main as orch_main
    from orchestrator import stages as stages_dispatch
    from orchestrator.store import Store

    store = Store(tmp_path)
    cfg = OrchestratorConfig(
        data_root=tmp_path / "data",
        repo_root=tmp_path / "repo",
        model_cache_root=tmp_path / "cache",
        vllm_port=18200,
        eval_gpus=(),  # forces graceful skip on DEPLOY
    )
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    run = _make_run()

    # Patch the `execute_stage` symbol that main.py imported into its own
    # namespace (`from .stages import execute_stage`). DEPLOY returns a
    # graceful-skip StageResult; every other stage returns a no-op ok.
    def _execute_stage_intercept(run: Run, stage: StageName,
                                 cfg_: OrchestratorConfig, store_: Store) -> Any:
        if stage == StageName.DEPLOY:
            return stages_dispatch.StageResult(
                ok=False,
                duration_s=0.1,
                artifacts=[],
                error="insufficient_gpu: eval pool is empty",
                error_kind="insufficient_gpu",
                extra={"aborted": True, "reason": "eval pool is empty"},
            )
        return stages_dispatch.StageResult(
            ok=True, duration_s=0.01, artifacts=[],
        )

    monkeypatch.setattr(orch_main, "execute_stage", _execute_stage_intercept)

    orch_main.run_pipeline(run, store, cfg=cfg, stub_only=False)

    assert run.status == RunStatus.ABORTED, f"expected ABORTED, got {run.status}"
    deploy_info = run.get_stage(StageName.DEPLOY)
    assert deploy_info.status == StageStatus.SKIPPED, (
        f"expected DEPLOY SKIPPED, got {deploy_info.status}"
    )
    assert run.failure_reason is not None
    assert "aborted" in run.failure_reason.lower()
    assert "eval pool" in run.failure_reason.lower()
    # downstream non-cleanup stages were not advanced
    for s in (StageName.READY_WAIT, StageName.CAPABILITY, StageName.SHOWCASE):
        assert run.get_stage(s).status == StageStatus.PENDING, (
            f"{s.value} unexpectedly advanced: {run.get_stage(s).status}"
        )


# ── §D. notify.run_aborted (single case) ───────────────────────────────────


def test_n1_notify_run_aborted_writes_jsonl(tmp_path: Path) -> None:
    """N1: notify.run_aborted emits one outbox JSONL event distinguishable
    from run_failed (event_type=run_aborted, level=warn not error)."""
    outbox = tmp_path / "outbox.jsonl"
    notify.run_aborted(
        outbox,
        run_id="r-test",
        hf_id="Qwen/Qwen2.5-0.5B",
        stage="DEPLOY",
        reason="insufficient_gpu: eval pool overlaps prod",
    )
    assert outbox.exists()
    lines = outbox.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event_type"] == "run_aborted"
    assert event["level"] == "warn"
    assert event["run_id"] == "r-test"
    assert event["stage"] == "DEPLOY"
    assert "insufficient_gpu" in event["body"]


# ── §E. real nvidia-smi parser (single case) ───────────────────────────────


@pytest.mark.real_nvidia_smi
def test_e1_nvidia_smi_parser_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """E1: _nvidia_smi_used_mib parses the canonical csv output. Patches
    subprocess.run to return a fixture string and asserts the dict shape."""
    fixture = "0, 1000\n1, 2000\n2, 89000\n3, 89000\n"

    def _fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert argv[0] == "nvidia-smi"
        return subprocess.CompletedProcess(
            args=argv, returncode=0, stdout=fixture, stderr="",
        )

    monkeypatch.setattr(stages_py.subprocess, "run", _fake_run)
    out = stages_py._nvidia_smi_used_mib()
    assert out == {0: 1000, 1: 2000, 2: 89000, 3: 89000}
