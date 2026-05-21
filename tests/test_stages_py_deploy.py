"""Unit tests for orchestrator/stages_py.execute_deploy.

Mocks docker.from_env and asserts the exact docker.containers.run kwargs
and the deploy.json payload. No real Docker required.

Test IDs match docs/PR3_TEST_PLAN.md §DEPLOY.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from orchestrator import stages_py
from orchestrator.config import OrchestratorConfig
from orchestrator.state_machine import Run


def _make_cfg(tmp: Path) -> OrchestratorConfig:
    cfg = OrchestratorConfig(
        data_root=tmp / "data",
        repo_root=tmp / "repo",
        model_cache_root=tmp / "cache",
        vllm_port=18200,
    )
    cfg.runs_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _make_run(run_id: str = "2026-05-21_001_qwenshort", hf_id: str = "Qwen/Qwen2.5-0.5B-Instruct") -> Run:
    return Run(run_id=run_id, hf_id=hf_id)


# Tests don't want the post-run 0.5s sanity sleep to run for real.
_NOP_SLEEP = lambda _: None  # noqa: E731


def _write_engine_plan(cfg: OrchestratorConfig, run: Run, plan: dict) -> None:
    rd = cfg.run_dir(run.run_id) / "_meta"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "engine.json").write_text(json.dumps(plan), encoding="utf-8")


def _make_model_cache(cfg: OrchestratorConfig, run: Run) -> Path:
    p = cfg.model_cache_root / cfg.hf_local_dir(run.hf_id)
    p.mkdir(parents=True, exist_ok=True)
    return p


class _FakeContainer:
    def __init__(self, name: str, status: str = "running",
                 labels: dict | None = None, logs_text: bytes = b""):
        self.name = name
        self.status = status
        self.attrs = {"Config": {"Labels": labels or {}}, "Name": f"/{name}"}
        self._logs = logs_text

    def reload(self) -> None:
        pass

    def logs(self, tail: int = 50) -> bytes:
        return self._logs

    def remove(self, force: bool = False, v: bool = False) -> None:
        pass


def _fake_docker_client(running_containers: dict | None = None) -> MagicMock:
    """Build a MagicMock that behaves enough like a docker.DockerClient."""
    rc = running_containers or {}
    client = MagicMock()
    client.ping.return_value = True

    def _get(name: str):
        if name in rc:
            return rc[name]
        from docker.errors import NotFound
        raise NotFound(f"no such container {name}")

    client.containers.get.side_effect = _get
    client.containers.list.return_value = list(rc.values())
    return client


def _patch_docker(client_mock: MagicMock):
    """Patch the docker module symbol *inside* stages_py."""
    # docker.from_env returns our mock
    docker_mod = MagicMock()
    docker_mod.from_env.return_value = client_mock
    docker_mod.types.DeviceRequest = MagicMock()
    # Errors are sentinel classes; reuse real ones if present, else fakes.
    try:
        from docker import errors as _e
        docker_mod.errors = _e
    except ImportError:  # pragma: no cover
        docker_mod.errors = MagicMock()
    return patch.object(stages_py, "docker", docker_mod)


# ── H series ──────────────────────────────────────────────────────────────


class DeployHappyTests(unittest.TestCase):

    def test_h1_vllm_text_model(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {}})
            _make_model_cache(cfg, run)
            expected_name = stages_py.container_name_for(run.run_id, "vllm")
            client = _fake_docker_client()
            fake_container = _FakeContainer(expected_name, status="running")
            client.containers.run.return_value = fake_container

            with _patch_docker(client):
                r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

            self.assertTrue(r.ok, msg=r.error)
            self.assertEqual(r.container_name, expected_name)
            self.assertIn("deploy.json", r.artifacts)

            deploy = json.loads(
                (cfg.run_dir(run.run_id) / "deploy.json").read_text(encoding="utf-8")
            )
            self.assertEqual(deploy["engine"], "vllm")
            self.assertEqual(deploy["base_url"], "http://127.0.0.1:18200")
            self.assertTrue(deploy["container_name"].startswith("e9-vllm-"))
            self.assertIn("started_at", deploy)

            kwargs = client.containers.run.call_args.kwargs
            self.assertEqual(kwargs["labels"][stages_py.LABEL_RUN], run.run_id)
            self.assertEqual(kwargs["labels"][stages_py.LABEL_STAGE], "DEPLOY")
            self.assertEqual(kwargs["labels"][stages_py.LABEL_ENGINE], "vllm")
            self.assertTrue(kwargs["detach"])

    def test_h2_sglang_engine(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_engine_plan(cfg, run, {"engine": "sglang", "vllm_args": {}})
            _make_model_cache(cfg, run)
            client = _fake_docker_client()
            client.containers.run.return_value = _FakeContainer("e9-sglang-qwenshort")

            with _patch_docker(client):
                r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

            self.assertTrue(r.ok, msg=r.error)
            self.assertTrue(r.container_name.startswith("e9-sglang-"))

    def test_h3_transformers_engine(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_engine_plan(cfg, run, {"engine": "transformers", "vllm_args": {}})
            _make_model_cache(cfg, run)
            client = _fake_docker_client()
            client.containers.run.return_value = _FakeContainer("e9-tf-qwenshort")

            with _patch_docker(client):
                r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

            self.assertTrue(r.ok, msg=r.error)
            self.assertTrue(r.container_name.startswith("e9-tf-"))

    def test_h4_vllm_max_model_len_arg_passed(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_engine_plan(cfg, run, {
                "engine": "vllm",
                "vllm_args": {"max_model_len": 8192},
            })
            _make_model_cache(cfg, run)
            client = _fake_docker_client()
            client.containers.run.return_value = _FakeContainer("e9-vllm-qwenshort")

            with _patch_docker(client):
                stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

            cmd = client.containers.run.call_args.kwargs["command"]
            self.assertIn("--max-model-len", cmd)
            self.assertIn("8192", cmd)

    def test_h5_tp_size_propagates_to_device_requests(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_engine_plan(cfg, run, {
                "engine": "vllm",
                "vllm_args": {"tensor_parallel_size": 2},
            })
            _make_model_cache(cfg, run)
            client = _fake_docker_client()
            client.containers.run.return_value = _FakeContainer("e9-vllm-qwenshort")

            with _patch_docker(client):
                stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

            kwargs = client.containers.run.call_args.kwargs
            self.assertIn("--tensor-parallel-size", kwargs["command"])
            self.assertEqual(len(kwargs["device_requests"]), 1)

    def test_h6_labels_present(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run("custom_run_2026")
            _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {}})
            _make_model_cache(cfg, run)
            client = _fake_docker_client()
            client.containers.run.return_value = _FakeContainer("e9-vllm-run2026")

            with _patch_docker(client):
                stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

            labels = client.containers.run.call_args.kwargs["labels"]
            self.assertEqual(labels[stages_py.LABEL_RUN], "custom_run_2026")
            self.assertEqual(labels[stages_py.LABEL_STAGE], "DEPLOY")
            self.assertEqual(labels[stages_py.LABEL_ENGINE], "vllm")


# ── S series ──────────────────────────────────────────────────────────────


class DeploySadTests(unittest.TestCase):

    def test_s1_docker_daemon_down(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {}})
            _make_model_cache(cfg, run)

            from docker.errors import DockerException
            client = MagicMock()
            client.ping.side_effect = DockerException("no daemon")
            with _patch_docker(client):
                r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "docker_down")

    def test_s2_image_not_found(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {}})
            _make_model_cache(cfg, run)
            from docker.errors import ImageNotFound
            client = _fake_docker_client()
            client.containers.run.side_effect = ImageNotFound("not pulled")
            with _patch_docker(client):
                r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "image_pull")

    def test_s3_model_path_missing(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {}})
            client = _fake_docker_client()
            with _patch_docker(client):
                r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "model_missing")
            self.assertEqual(client.containers.run.call_count, 0)

    def test_s4_engine_json_missing(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _make_model_cache(cfg, run)
            client = _fake_docker_client()
            with _patch_docker(client):
                r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "missing_artifact")

    def test_s5_unknown_engine(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_engine_plan(cfg, run, {"engine": "foobar"})
            _make_model_cache(cfg, run)
            client = _fake_docker_client()
            with _patch_docker(client):
                r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "unknown_engine")

    def test_s6_port_in_use(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {}})
            _make_model_cache(cfg, run)
            from docker.errors import APIError
            client = _fake_docker_client()
            client.containers.run.side_effect = APIError("Bind for 0.0.0.0:18200 failed: port is already allocated")
            with _patch_docker(client):
                r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "port_in_use")

    def test_s7_container_exits_immediately(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {}})
            _make_model_cache(cfg, run)
            client = _fake_docker_client()
            fake = _FakeContainer("e9-vllm-qwenshort", status="exited",
                                  logs_text=b"OOM killed\nfoo")
            client.containers.run.return_value = fake
            with _patch_docker(client):
                r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)

            self.assertFalse(r.ok)
            self.assertEqual(r.error_kind, "early_exit")
            self.assertIn("OOM killed", r.extra.get("logs", ""))


# ── E series ──────────────────────────────────────────────────────────────


class DeployEdgeTests(unittest.TestCase):

    def test_e1_existing_container_replaced(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {}})
            _make_model_cache(cfg, run)

            stale = _FakeContainer("e9-vllm-qwenshort", status="exited",
                                   labels={stages_py.LABEL_RUN: "old_run"})
            client = _fake_docker_client({"e9-vllm-qwenshort": stale})
            client.containers.run.return_value = _FakeContainer("e9-vllm-qwenshort")

            with _patch_docker(client):
                r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)
            self.assertTrue(r.ok, msg=r.error)

    def test_e2_kebab_and_snake_dedup(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run()
            _write_engine_plan(cfg, run, {
                "engine": "vllm",
                "vllm_args": {"max_model_len": 4096, "max-model-len": 8192},
            })
            _make_model_cache(cfg, run)
            client = _fake_docker_client()
            client.containers.run.return_value = _FakeContainer("e9-vllm-qwenshort")
            with _patch_docker(client):
                stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)
            cmd = client.containers.run.call_args.kwargs["command"]
            # de-duped: only one --max-model-len flag, with snake_case
            # value (since it was first).
            self.assertEqual(cmd.count("--max-model-len"), 1)
            idx = cmd.index("--max-model-len")
            self.assertEqual(cmd[idx + 1], "4096")

    def test_e3_short_runid_padded(self):
        with TemporaryDirectory() as td:
            tmp = Path(td)
            cfg = _make_cfg(tmp)
            run = _make_run(run_id="a")
            _write_engine_plan(cfg, run, {"engine": "vllm", "vllm_args": {}})
            _make_model_cache(cfg, run)
            client = _fake_docker_client()
            client.containers.run.return_value = _FakeContainer("e9-vllm-a0000000")
            with _patch_docker(client):
                r = stages_py.execute_deploy(run, cfg, sleep=_NOP_SLEEP)
            self.assertTrue(r.ok, msg=r.error)
            self.assertEqual(r.container_name, "e9-vllm-a0000000")
            self.assertLessEqual(len(r.container_name), 63)


if __name__ == "__main__":
    unittest.main()
