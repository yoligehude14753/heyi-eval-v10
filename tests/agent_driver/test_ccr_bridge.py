"""Tests for ``agent_driver.ccr_bridge``.

Focus: generated ccr-config.json
  - routes default → zhipu provider (glm-5.1) after the 2026-05 migration
  - single cloud provider (heyi-glm local fallback removed)
  - api_base_url avoids version doubling for /v1 (yunwu) and /v4 (zhipu)
  - secrets land in chmod 600 files
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from agent_driver.ccr_bridge import (  # noqa: E402
    build_ccr_config,
    write_ccr_config_to_path,
)


class _EnvIsolation(unittest.TestCase):
    """Common helper: clear the env keys ccr_bridge reads so test
    cases see a deterministic starting state."""

    _ENV_KEYS = (
        "HEYI_EVAL_JUDGE_PROVIDER",
        "HEYI_ENGINE_URL",
        "HEYI_ENGINE_API_KEY",
        "ZHIPU_API_KEY",
        "ZHIPU_BASE_URL",
        "ZHIPU_MODEL",
        "GLM_API_KEY",
        "YUNWU_BASE_URL",
        "YUNWU_GENERAL_KEY",
        "YUNWU_KEY_2",
        "YUNWU_GPT_KEY",
        "HEYI_EVAL_AGENT_MODEL",
    )

    def setUp(self) -> None:
        super().setUp()
        self._saved = {k: os.environ.pop(k, None) for k in self._ENV_KEYS}

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        super().tearDown()


class BuildCcrConfigTests(_EnvIsolation):

    def test_explicit_args_bypass_env(self) -> None:
        cfg = build_ccr_config(
            yunwu_url="https://yunwu.ai/v1",
            yunwu_key="sk-test",
        )
        # default model spec is yunwu-m27 / MiniMax-M2.7
        self.assertEqual(cfg["Router"]["default"], "yunwu-m27,MiniMax-M2.7")
        providers = {p["name"]: p for p in cfg["Providers"]}
        self.assertIn("yunwu-m27", providers)
        self.assertEqual(providers["yunwu-m27"]["api_key"], "sk-test")
        self.assertEqual(
            providers["yunwu-m27"]["api_base_url"],
            "https://yunwu.ai/v1/chat/completions",
        )

    def test_env_path_rejects_local_provider(self) -> None:
        """When called WITHOUT explicit args, the provider must be a
        cloud one (yunwu/zhipu) — the local :10814 path has no
        Anthropic-compatible bridge for the agent route."""
        os.environ["HEYI_EVAL_JUDGE_PROVIDER"] = "local"
        with self.assertRaisesRegex(RuntimeError, "cloud LLM provider"):
            build_ccr_config()

    def test_env_path_default_yunwu(self) -> None:
        """Default provider (env unset) is yunwu; needs a YUNWU_* key."""
        os.environ["YUNWU_GENERAL_KEY"] = "sk-yunwu"
        cfg = build_ccr_config()
        providers = {p["name"]: p for p in cfg["Providers"]}
        self.assertIn("yunwu-m27", providers)
        self.assertEqual(providers["yunwu-m27"]["api_key"], "sk-yunwu")
        self.assertEqual(
            providers["yunwu-m27"]["api_base_url"],
            "https://yunwu.ai/v1/chat/completions",
        )

    def test_env_path_zhipu_opt_in(self) -> None:
        """Opt into zhipu via env: provider/model flip to glm-5.1 and the
        /v4 base is not doubled to /v4/v1/chat/completions."""
        os.environ["HEYI_EVAL_JUDGE_PROVIDER"] = "zhipu"
        os.environ["ZHIPU_API_KEY"] = "sk-zhipu"
        os.environ["HEYI_EVAL_AGENT_MODEL"] = "zhipu-glm,glm-5.1"
        cfg = build_ccr_config()
        providers = {p["name"]: p for p in cfg["Providers"]}
        self.assertIn("zhipu-glm", providers)
        self.assertEqual(providers["zhipu-glm"]["api_key"], "sk-zhipu")
        self.assertEqual(
            providers["zhipu-glm"]["api_base_url"],
            "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        )

    def test_env_path_yunwu_provider(self) -> None:
        os.environ["HEYI_EVAL_JUDGE_PROVIDER"] = "yunwu"
        os.environ["YUNWU_BASE_URL"] = "https://yunwu.ai/v1"
        os.environ["YUNWU_GENERAL_KEY"] = "sk-env"
        cfg = build_ccr_config()
        # Single provider; its key resolves from the yunwu branch.
        self.assertEqual(cfg["Providers"][0]["api_key"], "sk-env")
        self.assertEqual(
            cfg["Providers"][0]["api_base_url"],
            "https://yunwu.ai/v1/chat/completions",
        )

    def test_missing_yunwu_key_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "api_key resolved to empty"):
            build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="")

    def test_alternate_agent_model_spec(self) -> None:
        cfg = build_ccr_config(
            yunwu_url="https://yunwu.ai/v1", yunwu_key="sk",
            agent_model_spec="yunwu-k26,Kimi-K2.6",
        )
        self.assertEqual(cfg["Router"]["default"], "yunwu-k26,Kimi-K2.6")
        self.assertEqual(cfg["Router"]["background"], "yunwu-k26,Kimi-K2.6")
        providers = {p["name"]: p for p in cfg["Providers"]}
        self.assertIn("yunwu-k26", providers)
        self.assertEqual(providers["yunwu-k26"]["models"], ["Kimi-K2.6"])

    def test_bad_model_spec_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "provider,model_id"):
            build_ccr_config(
                yunwu_url="https://yunwu.ai/v1", yunwu_key="sk",
                agent_model_spec="malformed-without-comma",
            )

    def test_url_without_v1_suffix_gets_completions_appended(self) -> None:
        """Defensive: base_url that lacks /v1 still produces a valid
        chat/completions URL."""
        cfg = build_ccr_config(
            yunwu_url="https://yunwu.ai", yunwu_key="sk",
        )
        self.assertEqual(
            cfg["Providers"][0]["api_base_url"],
            "https://yunwu.ai/v1/chat/completions",
        )

    def test_url_with_trailing_slash_normalises(self) -> None:
        """``rstrip("/")`` should kill trailing slash so we don't end
        up with ``//chat/completions``."""
        cfg = build_ccr_config(
            yunwu_url="https://yunwu.ai/v1/", yunwu_key="sk",
        )
        self.assertEqual(
            cfg["Providers"][0]["api_base_url"],
            "https://yunwu.ai/v1/chat/completions",
        )

    def test_local_glm_fallback_removed(self) -> None:
        """The local heyi-glm @ :10817 provider was removed in the
        2026-05 Zhipu migration — there should be exactly one (cloud)
        provider and no local-IP endpoint."""
        cfg = build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="sk")
        provider_names = {p["name"] for p in cfg["Providers"]}
        self.assertNotIn("heyi-glm", provider_names)
        self.assertEqual(len(cfg["Providers"]), 1)
        # No local IP leaked into any api_base_url
        for p in cfg["Providers"]:
            self.assertNotIn("10.10.11.198", p["api_base_url"])

    def test_required_transformers_present_in_order(self) -> None:
        """The cloud provider needs three transformers:

          - ``openai``: ccr's built-in Anthropic→OpenAI protocol shim,
            without which the OpenAI-compatible endpoint rejects every
            request (cache_control / content blocks / reasoning are not
            accepted by the OpenAI chat/completions API).
          - ``maxtoken``: caps response tokens so a runaway agent
            can't blow the budget in one call.
          - ``strip-thinking``: removes the model's <think> blocks before
            handing back to claude CLI, whose parser chokes on them.

        Order matters: ``openai`` must run first so downstream
        transformers operate on the already-converted shape.
        """
        cfg = build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="sk")
        providers = {p["name"]: p for p in cfg["Providers"]}
        transformer_use = providers["yunwu-m27"]["transformer"]["use"]
        flat = [t if isinstance(t, str) else t[0] for t in transformer_use]
        self.assertIn("openai", flat)
        self.assertIn("maxtoken", flat)
        self.assertIn("strip-thinking", flat)
        self.assertEqual(
            flat.index("openai"), 0,
            "openai must come first so other transformers see OpenAI shape",
        )


class WriteCcrConfigTests(_EnvIsolation):

    def test_atomic_write_chmod_600(self) -> None:
        cfg = build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="sk")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            write_ccr_config_to_path(cfg, path)
            self.assertTrue(path.exists())
            # mode bits — only owner read+write
            mode = stat.S_IMODE(path.stat().st_mode)
            self.assertEqual(mode, 0o600)
            # roundtrip
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded["Router"]["default"], cfg["Router"]["default"])

    def test_atomic_write_no_tmp_leftover(self) -> None:
        cfg = build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="sk")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            write_ccr_config_to_path(cfg, path)
            tmpfile = path.with_suffix(path.suffix + ".tmp")
            self.assertFalse(tmpfile.exists(),
                             "atomic write left .tmp behind")

    def test_missing_parent_dir_raises(self) -> None:
        cfg = build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="sk")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonexistent" / "config.json"
            with self.assertRaisesRegex(FileNotFoundError, "parent directory"):
                write_ccr_config_to_path(cfg, path)


class ResolverIntegrationTests(_EnvIsolation):
    """Wire-up check: ccr_bridge consults the central
    ``_resolve_engine_endpoint`` rather than its own env-reading code."""

    def test_uses_central_resolver(self) -> None:
        os.environ["HEYI_EVAL_JUDGE_PROVIDER"] = "yunwu"
        os.environ["YUNWU_GENERAL_KEY"] = "sk-from-resolver"
        # Patch the resolver to confirm ccr_bridge calls it, not its
        # own copy of env-reading code.
        with mock.patch(
            "agent_driver.ccr_bridge._resolve_engine_endpoint",
            return_value=("https://example.com/v1", "sk-from-resolver"),
        ) as m:
            cfg = build_ccr_config()
            m.assert_called_once_with()
        # Single provider; url/key come from the (mocked) resolver.
        self.assertEqual(
            cfg["Providers"][0]["api_base_url"],
            "https://example.com/v1/chat/completions",
        )
        self.assertEqual(cfg["Providers"][0]["api_key"], "sk-from-resolver")


class InjectIntoContainerTests(_EnvIsolation):
    """M2d: docker put_archive + pkill -HUP path.

    We use a hand-built fake docker client because docker-py's behaviour
    around put_archive + tar is the bit we want to validate (the tar
    must be parseable + content must be ours)."""

    def _fake_docker(self) -> tuple:
        """Returns (docker_client, captured) — ``captured`` is a dict
        with keys ``tar_bytes`` / ``exec_cmds`` / ``put_archive_calls``."""
        captured: dict = {
            "tar_bytes": b"",
            "exec_cmds": [],
            "put_archive_dest": None,
            "put_archive_calls": 0,
        }

        class _C:
            name = "m2b-1"

            def put_archive(self, path: str, data: bytes) -> bool:
                captured["put_archive_dest"] = path
                captured["tar_bytes"] = data
                captured["put_archive_calls"] += 1
                return True

            def exec_run(self, **kw):
                captured["exec_cmds"].append(kw.get("cmd"))
                class _R:
                    exit_code = 0
                    output = b""
                return _R()

        class _Containers:
            def get(self, name: str) -> _C:
                assert name == "m2b-1"
                return _C()

        class _Docker:
            containers = _Containers()

        return _Docker(), captured

    def test_put_archive_called_with_parent_dir(self) -> None:
        from agent_driver.ccr_bridge import inject_into_container
        docker, cap = self._fake_docker()
        cfg = build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="sk")
        inject_into_container(
            "m2b-1", docker_client=docker, config=cfg,
            settle_seconds=0,  # no sleep in tests
        )
        self.assertEqual(
            cap["put_archive_dest"],
            "/home/agent/.claude-code-router",
        )
        self.assertEqual(cap["put_archive_calls"], 1)

    def test_tar_contains_config_json_with_correct_body(self) -> None:
        import io
        import tarfile

        from agent_driver.ccr_bridge import inject_into_container
        docker, cap = self._fake_docker()
        cfg = build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="sk-injection-test")
        inject_into_container(
            "m2b-1", docker_client=docker, config=cfg, settle_seconds=0,
        )
        # The captured bytes should be a tar with one entry: ``config.json``
        with tarfile.open(fileobj=io.BytesIO(cap["tar_bytes"]), mode="r") as tar:
            names = tar.getnames()
            self.assertEqual(names, ["config.json"])
            member = tar.extractfile("config.json")
            assert member is not None
            body = member.read().decode("utf-8")
            loaded = json.loads(body)
            self.assertEqual(loaded["Router"]["default"], cfg["Router"]["default"])
            info = tar.getmember("config.json")
            self.assertEqual(info.mode, 0o600)
            # uid/gid must match the agent user inside m2b (uid 1100);
            # otherwise put_archive extracts as root and ccr (running as
            # agent) hits "permission denied" on read.  Confirmed by the
            # heyi real-machine drill on 2026-05-26.
            self.assertEqual(info.uid, 1100)
            self.assertEqual(info.gid, 1100)
            self.assertEqual(info.uname, "agent")
            self.assertEqual(info.gname, "agent")

    def test_ccr_restart_failure_raises(self) -> None:
        # If the in-container readiness loop times out (exit code 1),
        # inject must raise so the caller sees the failure rather than
        # silently moving on to a run_agent that will hit ConnectionRefused.
        from agent_driver.ccr_bridge import inject_into_container

        class _CFailing:
            name = "m2b-1"
            def put_archive(self, *a, **kw):
                return True
            def exec_run(self, **kw):
                class _R:
                    exit_code = 1
                    output = b"ccr never came up"
                return _R()

        class _Containers:
            def get(self, name):
                return _CFailing()

        class _Docker:
            containers = _Containers()

        cfg = build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="sk")
        with self.assertRaisesRegex(RuntimeError, "ccr restart"):
            inject_into_container(
                "m2b-1", docker_client=_Docker(),
                config=cfg, settle_seconds=0,
            )

    def test_owner_uid_gid_overridable(self) -> None:
        """Operators with a non-default m2b image should be able to point
        at a different uid via kwargs without forking the module."""
        import io
        import tarfile

        from agent_driver.ccr_bridge import inject_into_container
        docker, cap = self._fake_docker()
        cfg = build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="sk")
        inject_into_container(
            "m2b-1", docker_client=docker, config=cfg, settle_seconds=0,
            owner_uid=2000, owner_gid=2000,
        )
        with tarfile.open(fileobj=io.BytesIO(cap["tar_bytes"]), mode="r") as tar:
            info = tar.getmember("config.json")
            self.assertEqual(info.uid, 2000)
            self.assertEqual(info.gid, 2000)

    def test_stop_and_start_in_single_synchronous_script(self) -> None:
        # inject runs everything in ONE sh -c block so the call blocks
        # until ccr is actually answering :3456.
        #
        # Iteration history (all confirmed on heyi 2026-05-26 drills):
        #   - SIGHUP: ccr's node entry-point doesn't trap it → terminate.
        #   - detach=True + nohup start: docker-py 7.x lost the child.
        #   - pkill -f 'ccr start': sh -c's own argv contains "ccr start"
        #     literally, so pkill matched sh and we SIGTERM'd ourselves
        #     (exit 143).
        # Current shape: use ccr's own ``ccr stop`` subcommand (which
        # signals via its pidfile, no string-matching footgun), then
        # nohup ccr start, then a curl readiness loop on :3456.
        from agent_driver.ccr_bridge import inject_into_container
        docker, cap = self._fake_docker()
        cfg = build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="sk")
        inject_into_container(
            "m2b-1", docker_client=docker, config=cfg, settle_seconds=0,
        )
        self.assertEqual(len(cap["exec_cmds"]), 1)
        cmd = " ".join(cap["exec_cmds"][0])
        self.assertIn("ccr stop", cmd)
        self.assertNotIn("pkill", cmd, "must not use pkill -f (self-kill footgun)")
        self.assertIn(".claude-code-router.pid", cmd)
        self.assertIn("nohup ccr start", cmd)
        self.assertIn("127.0.0.1:3456", cmd)

    def test_put_archive_false_return_raises(self) -> None:
        """docker-py legacy paths return False on failure; we translate
        to RuntimeError so caller (pool_manager) sees a clear error."""
        from agent_driver.ccr_bridge import inject_into_container

        class _C:
            def put_archive(self, p, d):
                return False

            def exec_run(self, **kw):
                class _R:
                    exit_code = 0
                return _R()

        class _Containers:
            def get(self, n):
                return _C()

        class _Docker:
            containers = _Containers()

        cfg = build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="sk")
        with self.assertRaisesRegex(RuntimeError, "put_archive returned False"):
            inject_into_container(
                "m2b-1", docker_client=_Docker(), config=cfg, settle_seconds=0,
            )


class StreamExecFactoryTests(unittest.TestCase):
    """make_docker_stream_exec wires docker-py's exec_run; ensure
    the contract we pass into it matches what docker-py expects."""

    def test_uses_container_exec_run_with_stream(self) -> None:
        from agent_driver.exec_runner import make_docker_stream_exec
        from agent_driver.pool_manager import ContainerHandle

        captured = {}

        class _C:
            def exec_run(self, **kw):
                captured.update(kw)

                class _R:
                    output = (b"chunk-1", b"chunk-2")
                return _R()

        class _Containers:
            def get(self, name):
                return _C()

        class _Docker:
            containers = _Containers()

        stream = make_docker_stream_exec(_Docker())
        handle = ContainerHandle(name="m2b-1",
                                 workspace_root_in_container="/home/agent/workspace")
        chunks = list(stream(handle, ["claude", "--print", "/x"], "/home/agent/workspace/run-1"))
        self.assertEqual(chunks, [b"chunk-1", b"chunk-2"])
        # The factory must pass through stream=True, stdout=True
        self.assertTrue(captured["stream"])
        self.assertTrue(captured["stdout"])
        self.assertFalse(captured["stderr"])
        self.assertEqual(captured["workdir"], "/home/agent/workspace/run-1")
        self.assertEqual(captured["user"], "agent")

    def test_tuple_result_unpacks(self) -> None:
        """Some docker-py versions return ``(exit_code, generator)`` instead
        of ExecResult; the factory must handle both."""
        from agent_driver.exec_runner import make_docker_stream_exec
        from agent_driver.pool_manager import ContainerHandle

        class _C:
            def exec_run(self, **kw):
                return (0, iter([b"a", b"b"]))

        class _Containers:
            def get(self, name):
                return _C()

        class _Docker:
            containers = _Containers()

        stream = make_docker_stream_exec(_Docker())
        handle = ContainerHandle(name="m2b-1",
                                 workspace_root_in_container="/")
        chunks = list(stream(handle, ["x"], "/"))
        self.assertEqual(chunks, [b"a", b"b"])


if __name__ == "__main__":
    unittest.main()
