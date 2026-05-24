"""PR#40: HF_TOKEN plumbing through the staging + discovery layers.

NV8 production-mode observation (2026-05-24 05-14 CST): the
auto-discover service surfaced multiple high-signal 2026 models that
all failed at STAGE_MODEL with ``snapshot_download failed:
RepositoryNotFoundError: 401 Client Error`` (google/gemma-3-9b-it,
meta-llama/Llama-4-Scout-17B, mistralai/Voxtral-Mini-4B-Realtime-2602).

These repos are gated, not missing — the orchestrator never sent a
token. PR#40 adds:

  1. ``OrchestratorConfig.hf_token`` reading both ``HF_TOKEN`` and the
     legacy ``HUGGING_FACE_HUB_TOKEN`` env var (HF_TOKEN wins).
  2. ``ensure_model_staged(hf_token=...)`` keyword that forwards to the
     downloader.
  3. ``_default_downloader`` passes ``token=`` to
     ``huggingface_hub.snapshot_download`` only when set.
  4. ``discover._make_api`` forwards ``token=`` to ``HfApi`` for the
     listing path (also benefits from higher per-account rate limits).

Backward compatibility contract: legacy downloader test fakes whose
signatures predate ``token=`` keep working. ``token=`` is only passed
to ``dl(...)`` when actually set.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from orchestrator import model_stager
from orchestrator.config import OrchestratorConfig


# ── helpers ───────────────────────────────────────────────────────────────


def _stage_complete(target: Path) -> None:
    """Mimic a successful snapshot_download writing weights so the
    post-download verifier (PR#32) accepts the dir."""
    target.mkdir(parents=True, exist_ok=True)
    (target / "config.json").write_text(
        '{"model_type":"qwen2","architectures":["Qwen2ForCausalLM"]}',
        encoding="utf-8",
    )
    (target / "model.safetensors").write_bytes(b"\x00" * 4096)


# ── config layer ──────────────────────────────────────────────────────────


class ConfigReadsHfToken(unittest.TestCase):

    def setUp(self) -> None:
        # Snapshot env so other tests aren't affected.
        self._saved = {
            k: os.environ.pop(k, None)
            for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")
        }

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_unset_token_is_none(self) -> None:
        cfg = OrchestratorConfig()
        self.assertIsNone(cfg.hf_token)

    def test_reads_hf_token(self) -> None:
        os.environ["HF_TOKEN"] = "hf_abc123"
        cfg = OrchestratorConfig()
        self.assertEqual(cfg.hf_token, "hf_abc123")

    def test_reads_legacy_env_var(self) -> None:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = "hf_legacy"
        cfg = OrchestratorConfig()
        self.assertEqual(cfg.hf_token, "hf_legacy")

    def test_hf_token_wins_over_legacy_when_both_set(self) -> None:
        os.environ["HF_TOKEN"] = "hf_new"
        os.environ["HUGGING_FACE_HUB_TOKEN"] = "hf_old"
        cfg = OrchestratorConfig()
        self.assertEqual(cfg.hf_token, "hf_new")

    def test_empty_string_treated_as_none(self) -> None:
        """``export HF_TOKEN=`` should not poison the call with empty
        string; huggingface_hub treats "" differently from missing."""
        os.environ["HF_TOKEN"] = ""
        cfg = OrchestratorConfig()
        self.assertIsNone(cfg.hf_token)


# ── stager forwards token only when set ───────────────────────────────────


class StagerForwardsToken(unittest.TestCase):

    def test_no_token_means_no_token_kwarg(self) -> None:
        """Legacy fake without ``token=`` keeps working when token is
        unset — proves backward compatibility on the production path."""
        seen: dict[str, object] = {}

        def fake(*, repo_id: str, local_dir: str,
                 max_workers: int, allow_patterns: list[str] | None) -> str:
            seen["called"] = True
            seen["repo_id"] = repo_id
            _stage_complete(Path(local_dir))
            return local_dir

        with TemporaryDirectory() as td:
            target = Path(td) / "M"
            r = model_stager.ensure_model_staged(
                hf_id="org/M", target_dir=target,
                metadata={"param_count": "0.5B"},
                downloader=fake,
                # PR#40: no hf_token kwarg, no token= passed to dl()
            )
        self.assertTrue(r.ok, f"unexpected: {r}")
        self.assertEqual(seen["repo_id"], "org/M")

    def test_token_forwarded_when_set(self) -> None:
        """When hf_token is set, downloader receives ``token=`` kwarg.
        New fakes opting into the token contract must accept it."""
        seen: dict[str, object] = {}

        def fake(*, repo_id: str, local_dir: str,
                 max_workers: int, allow_patterns: list[str] | None,
                 token: str | None = None) -> str:
            seen["token"] = token
            _stage_complete(Path(local_dir))
            return local_dir

        with TemporaryDirectory() as td:
            target = Path(td) / "M"
            r = model_stager.ensure_model_staged(
                hf_id="google/gemma-3-9b-it",
                target_dir=target,
                metadata={"param_count": "9B"},
                hf_token="hf_xxx",
                downloader=fake,
            )
        self.assertTrue(r.ok, f"unexpected: {r}")
        self.assertEqual(seen["token"], "hf_xxx")

    def test_explicit_none_token_is_not_forwarded(self) -> None:
        """Explicit ``hf_token=None`` must behave identically to omitting
        the kwarg — otherwise the legacy downloader signature breaks."""
        called_with: dict[str, object] = {}

        def fake(*, repo_id: str, local_dir: str,
                 max_workers: int, allow_patterns: list[str] | None) -> str:
            called_with["max_workers"] = max_workers
            _stage_complete(Path(local_dir))
            return local_dir

        with TemporaryDirectory() as td:
            target = Path(td) / "M"
            r = model_stager.ensure_model_staged(
                hf_id="org/M", target_dir=target,
                metadata={"param_count": "0.5B"},
                hf_token=None,  # explicit none
                downloader=fake,
            )
        self.assertTrue(r.ok)
        self.assertEqual(called_with["max_workers"], 8)

    def test_empty_string_token_is_not_forwarded(self) -> None:
        """Defensive: empty-string token (sloppy env) treated like None."""

        def fake(*, repo_id: str, local_dir: str,
                 max_workers: int, allow_patterns: list[str] | None) -> str:
            _stage_complete(Path(local_dir))
            return local_dir

        with TemporaryDirectory() as td:
            target = Path(td) / "M"
            r = model_stager.ensure_model_staged(
                hf_id="org/M", target_dir=target,
                metadata={"param_count": "0.5B"},
                hf_token="",  # falsy
                downloader=fake,
            )
        self.assertTrue(r.ok)


# ── discover layer ────────────────────────────────────────────────────────


class DiscoverMakeApi(unittest.TestCase):

    def _patch_hfapi(self):
        from discover import main as discover_main
        captured: dict[str, object] = {}

        class _FakeHfApi:
            def __init__(self, **kw):
                captured.update(kw)

        return patch.object(
            discover_main, "_make_api",
            wraps=discover_main._make_api,
        ), captured

    def test_resolve_hf_token_falls_back_through_envs(self) -> None:
        from discover.main import _resolve_hf_token
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
            self.assertIsNone(_resolve_hf_token())
            os.environ["HUGGING_FACE_HUB_TOKEN"] = "hf_legacy"
            self.assertEqual(_resolve_hf_token(), "hf_legacy")
            os.environ["HF_TOKEN"] = "hf_modern"
            self.assertEqual(_resolve_hf_token(), "hf_modern")
            os.environ.pop("HF_TOKEN", None)
            os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)

    def test_make_api_no_token_uses_anonymous(self) -> None:
        """When token=None, ``HfApi(endpoint=...)`` is called WITHOUT
        a token kwarg — important because the hf-mirror anonymous path
        works for the bulk of public 2025 models."""
        from discover import main as discover_main
        seen: dict[str, object] = {}

        class _FakeHfApi:
            def __init__(self, **kw):
                seen.update(kw)

        with patch("huggingface_hub.HfApi", _FakeHfApi):
            discover_main._make_api("https://hf-mirror.com")
        self.assertEqual(seen, {"endpoint": "https://hf-mirror.com"})
        self.assertNotIn("token", seen)

    def test_make_api_with_token_forwards(self) -> None:
        from discover import main as discover_main
        seen: dict[str, object] = {}

        class _FakeHfApi:
            def __init__(self, **kw):
                seen.update(kw)

        with patch("huggingface_hub.HfApi", _FakeHfApi):
            discover_main._make_api(
                "https://hf-mirror.com", token="hf_abc",
            )
        self.assertEqual(seen, {
            "endpoint": "https://hf-mirror.com", "token": "hf_abc",
        })


if __name__ == "__main__":
    unittest.main()
