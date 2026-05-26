"""Tests for ``agent_driver.ccr_bridge``.

Focus: generated ccr-config.json
  - routes default → yunwu provider (M2.7)
  - keeps heyi-glm as fallback
  - api_base_url avoids /v1 doubling (the PR#18 bug pattern)
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

    def test_env_path_requires_provider_flag(self) -> None:
        """When called WITHOUT explicit args, HEYI_EVAL_JUDGE_PROVIDER
        must be yunwu — otherwise we refuse (M1 doesn't support
        falling back to local minimax)."""
        os.environ["YUNWU_GENERAL_KEY"] = "sk-y"
        with self.assertRaisesRegex(RuntimeError, "HEYI_EVAL_JUDGE_PROVIDER"):
            build_ccr_config()

    def test_env_path_yunwu_provider(self) -> None:
        os.environ["HEYI_EVAL_JUDGE_PROVIDER"] = "yunwu"
        os.environ["YUNWU_BASE_URL"] = "https://yunwu.ai/v1"
        os.environ["YUNWU_GENERAL_KEY"] = "sk-env"
        cfg = build_ccr_config()
        providers = {p["name"]: p for p in cfg["Providers"]}
        self.assertEqual(providers["yunwu-m27"]["api_key"], "sk-env")

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
        providers = {p["name"]: p for p in cfg["Providers"]}
        self.assertEqual(
            providers["yunwu-m27"]["api_base_url"],
            "https://yunwu.ai/v1/chat/completions",
        )

    def test_url_with_trailing_slash_normalises(self) -> None:
        """``rstrip("/")`` should kill trailing slash so we don't end
        up with ``//chat/completions``."""
        cfg = build_ccr_config(
            yunwu_url="https://yunwu.ai/v1/", yunwu_key="sk",
        )
        providers = {p["name"]: p for p in cfg["Providers"]}
        self.assertEqual(
            providers["yunwu-m27"]["api_base_url"],
            "https://yunwu.ai/v1/chat/completions",
        )

    def test_heyi_glm_fallback_provider_present(self) -> None:
        """Operator must be able to roll back to local GLM without
        re-deploying ccr config — the heyi-glm provider entry stays."""
        cfg = build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="sk")
        provider_names = {p["name"] for p in cfg["Providers"]}
        self.assertIn("heyi-glm", provider_names)
        self.assertIn("yunwu-m27", provider_names)

    def test_strip_thinking_transformer_present(self) -> None:
        """The strip-thinking ccr plugin is what keeps M2.7's <think>
        blocks from leaking through to claude CLI — without it, the
        CLI's parser sometimes chokes."""
        cfg = build_ccr_config(yunwu_url="https://yunwu.ai/v1", yunwu_key="sk")
        providers = {p["name"]: p for p in cfg["Providers"]}
        transformer_use = providers["yunwu-m27"]["transformer"]["use"]
        # transformer_use is a heterogeneous list of strings + [name, opts] lists
        flat = []
        for t in transformer_use:
            flat.append(t if isinstance(t, str) else t[0])
        self.assertIn("strip-thinking", flat)


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
    """Wire-up check: when HEYI_EVAL_JUDGE_PROVIDER=yunwu is set, the
    PR#18 ``_resolve_engine_endpoint`` is what ccr_bridge consults."""

    def test_uses_pr18_resolver(self) -> None:
        os.environ["HEYI_EVAL_JUDGE_PROVIDER"] = "yunwu"
        os.environ["YUNWU_BASE_URL"] = "https://example.com/v1"
        os.environ["YUNWU_GENERAL_KEY"] = "sk-from-resolver"
        # Patch the resolver to confirm ccr_bridge calls it, not its
        # own copy of env-reading code.
        with mock.patch(
            "agent_driver.ccr_bridge._resolve_engine_endpoint",
            return_value=("https://example.com/v1", "sk-from-resolver"),
        ) as m:
            cfg = build_ccr_config()
            m.assert_called_once_with()
        providers = {p["name"]: p for p in cfg["Providers"]}
        self.assertEqual(
            providers["yunwu-m27"]["api_base_url"],
            "https://example.com/v1/chat/completions",
        )
        self.assertEqual(providers["yunwu-m27"]["api_key"], "sk-from-resolver")


if __name__ == "__main__":
    unittest.main()
