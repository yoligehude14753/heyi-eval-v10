"""PR#68: discover-time parameter-count size gate.

Background
==========
Before PR#68 the discover layer filtered candidates by org + downloads/likes
+ modality + recency only. There was no parameter-count gate, so giant
models the eval-side GPU pool can never host (Kimi-K2.6 ≥1T, DeepSeek-V3
671B, Llama-3.1-405B, ...) made it into candidates.jsonl, sat in the
orchestrator queue, then aborted with ``oversize_skip`` at ENGINE_SELECT
after wasting one stager download attempt each.

The user's exact complaint that triggered this PR:

    “为什么 kimi2.6 明显超出参数量预期的还在测试的 list 里，它能在清单里，
     但是它参数量太大了，超出测试范围了，不会实际跑测试。”

These tests pin down the new behaviour:

1.  Param-count is extracted from HF's ``safetensors.total`` when present,
    else from the hf_id (``Kimi-K2.6-1T`` / ``DeepSeek-V3-671B`` / etc.),
    else left ``None`` (fail-open).
2.  ``TrackerConfig.max_param_billion`` (yaml: ``max_param_billion: 70``)
    drops candidates whose parsed size exceeds the ceiling, *at discover
    time*, with ``stats.excluded_oversize`` accounting.
3.  Existing fail-open invariants are preserved: param=None (unknown)
    still passes through, and max=None (gate disabled) restores legacy
    behaviour bit-for-bit.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from discover.tracker import (  # noqa: E402
    Candidate,
    Cursor,
    TrackerConfig,
    _coerce_max_param_billion,
    _model_to_candidate,
    _parse_param_billion,
    _parse_param_billion_from_id,
    _passes_size,
    scan_round,
)

# ── fake model shapes ──────────────────────────────────────────────────────


@dataclass
class FakeModel:
    """Attribute-shape (huggingface_hub.ModelInfo) for the converter."""

    id: str
    last_modified: str | None = None
    downloads: int = 0
    likes: int = 0
    pipeline_tag: str | None = None
    library_name: str | None = None
    private: bool = False
    gated: bool = False
    safetensors: object | None = None


@dataclass
class FakeSafetensors:
    """``safetensors`` block as exposed by HF expand=['safetensors']."""

    total: int


class FakeApi:
    """Minimal HfApi double — matches the FakeApi in tests/test_discover.py."""

    def __init__(self, by_author=None, trending=None):
        self.by_author = by_author or {}
        self.trending = trending or []

    def list_models(self, author=None, limit=None, sort=None, expand=None, **kw):
        if author:
            return list(self.by_author.get(author, []))[: (limit or 1000)]
        return list(self.trending)[: (limit or 1000)]


# ── (1) hf_id regex extractor ──────────────────────────────────────────────


class ParseParamBillionFromIdTests(unittest.TestCase):
    """The regex tail extractor must understand real production HF ids:

    *   Kimi-K2.6-1T              → 1000   (trillion-scale unit)
    *   DeepSeek-V3-671B          → 671    (post-fix B)
    *   Qwen3-72B-Instruct        → 72     (size + variant tail)
    *   Llama-3.1-405B-Instruct   → 405    (period inside version)
    *   Qwen3-Embedding-0.6B      → 0.6    (sub-1B, fractional)
    *   MoE-236.5B-A21B           → 236.5  (first token wins, not A21B)
    *   stable-diffusion-3        → None   (no size token at all)
    """

    def test_kimi_k26_1t(self):
        self.assertEqual(_parse_param_billion_from_id("moonshotai/Kimi-K2.6-1T"), 1000.0)
        self.assertEqual(
            _parse_param_billion_from_id("moonshotai/Kimi-K2.6-1T-Instruct"),
            1000.0,
        )

    def test_deepseek_v3_671b(self):
        self.assertEqual(
            _parse_param_billion_from_id("deepseek-ai/DeepSeek-V3-671B"), 671.0
        )

    def test_qwen3_72b(self):
        self.assertEqual(_parse_param_billion_from_id("Qwen/Qwen3-72B-Instruct"), 72.0)

    def test_llama_31_405b(self):
        # the "3.1" version token must not be confused with "405B"
        self.assertEqual(
            _parse_param_billion_from_id("meta-llama/Llama-3.1-405B-Instruct"),
            405.0,
        )

    def test_sub_1b_fraction(self):
        self.assertEqual(
            _parse_param_billion_from_id("Qwen/Qwen3-Embedding-0.6B"), 0.6
        )

    def test_moe_total_active_experts_ignored(self):
        # The "first size token wins" rule prevents misreporting the
        # active-experts count (21B) as the model size for a 236.5B MoE.
        self.assertEqual(
            _parse_param_billion_from_id("OrgX/MoE-236.5B-A21B"), 236.5
        )

    def test_no_size_token_returns_none(self):
        self.assertIsNone(_parse_param_billion_from_id("stabilityai/stable-diffusion-3"))
        self.assertIsNone(_parse_param_billion_from_id("apple/openelm-instruct"))

    def test_empty_string_returns_none(self):
        self.assertIsNone(_parse_param_billion_from_id(""))

    def test_no_slash_just_a_name(self):
        # Some FakeModel ids in tests are unqualified — still parse fine.
        self.assertEqual(_parse_param_billion_from_id("Llama-3-70B"), 70.0)


# ── (2) authoritative source: safetensors.total ────────────────────────────


class ParseParamBillionTests(unittest.TestCase):
    """Resolution order must be: safetensors.total → hf_id regex → None.

    safetensors.total counts parameters (not bytes), and is authoritative
    when HF returns it. Falling back to hf_id regex is only correct when
    safetensors is missing — otherwise a fine-tune named "merged-Llama-3-70B"
    that's actually 8B would be misclassified.
    """

    def test_safetensors_total_authoritative(self):
        m = FakeModel(id="org/Whatever-Misleading-405B",
                      safetensors=FakeSafetensors(total=7_700_000_000))
        # Even though the id says "405B", the on-disk count says 7.7B.
        self.assertAlmostEqual(_parse_param_billion(m, m.id), 7.7, places=2)

    def test_safetensors_total_dict_shape(self):
        # Mirror JSON returns dicts, not dataclass-like objects.
        m = {"id": "raw/T-model", "safetensors": {"total": 671_000_000_000}}
        self.assertAlmostEqual(_parse_param_billion(m, "raw/T-model"), 671.0, places=1)

    def test_fallback_to_hf_id_when_safetensors_missing(self):
        m = FakeModel(id="moonshotai/Kimi-K2.6-1T", safetensors=None)
        self.assertEqual(_parse_param_billion(m, m.id), 1000.0)

    def test_returns_none_when_no_signal(self):
        m = FakeModel(id="apple/openelm-instruct", safetensors=None)
        self.assertIsNone(_parse_param_billion(m, m.id))

    def test_garbage_safetensors_total_falls_through_to_id_regex(self):
        # A None/string total shouldn't kill the parse; we should fall
        # back to the hf_id regex (here: "70B" → 70.0).
        m = FakeModel(id="org/Some-70B", safetensors=FakeSafetensors(total=None))  # type: ignore[arg-type]
        self.assertEqual(_parse_param_billion(m, m.id), 70.0)


# ── (3) decision matrix ────────────────────────────────────────────────────


class PassesSizeTests(unittest.TestCase):
    """The four cells of the size gate."""

    def test_gate_disabled_lets_everything_through(self):
        self.assertTrue(_passes_size(1000.0, None))
        self.assertTrue(_passes_size(None, None))

    def test_unknown_param_passes_through_fail_open(self):
        # Same policy as _passes_window / _passes_modality on missing data:
        # don't false-positive the whole panel on an HF outage.
        self.assertTrue(_passes_size(None, 70.0))

    def test_under_or_equal_to_ceiling_passes(self):
        self.assertTrue(_passes_size(70.0, 70.0))
        self.assertTrue(_passes_size(7.7, 70.0))
        self.assertTrue(_passes_size(0.6, 70.0))

    def test_over_ceiling_drops(self):
        self.assertFalse(_passes_size(72.0, 70.0))    # Qwen3-72B at default
        self.assertFalse(_passes_size(671.0, 70.0))   # DeepSeek-V3
        self.assertFalse(_passes_size(1000.0, 70.0))  # Kimi-K2.6-1T


# ── (4) yaml coercion is generous: bad/missing input never mass-skips ──────


class CoerceMaxParamBillionTests(unittest.TestCase):
    """A typo'd yaml value must NOT silently turn the gate into "drop
    everything" — better to disable the gate and surface candidates than
    blackhole every model because someone wrote ``max_param_billion: 70 b``.
    """

    def test_explicit_none(self):
        self.assertIsNone(_coerce_max_param_billion(None))

    def test_yaml_null_string(self):
        for v in ("null", "None", "  ", "off", "false", "disabled", ""):
            self.assertIsNone(_coerce_max_param_billion(v), msg=v)

    def test_integer(self):
        self.assertEqual(_coerce_max_param_billion(70), 70.0)
        self.assertEqual(_coerce_max_param_billion("70"), 70.0)

    def test_float(self):
        self.assertEqual(_coerce_max_param_billion(70.5), 70.5)
        self.assertEqual(_coerce_max_param_billion("70.5"), 70.5)

    def test_junk_returns_none(self):
        # Garbage value disables the gate (safe default), not mass-skip.
        self.assertIsNone(_coerce_max_param_billion("garbage"))
        self.assertIsNone(_coerce_max_param_billion("70b"))

    def test_zero_or_negative_disables(self):
        # A 0-billion ceiling would drop everything — interpret as "off".
        self.assertIsNone(_coerce_max_param_billion(0))
        self.assertIsNone(_coerce_max_param_billion(-1))


# ── (5) yaml → TrackerConfig wiring ────────────────────────────────────────


class TrackerConfigFromYamlTests(unittest.TestCase):
    """The yaml-derived config must respect the new key (default 70.0,
    explicit null disables the gate)."""

    def _config_from_yaml(self, body: str) -> TrackerConfig:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "wl.yaml"
            p.write_text(body, encoding="utf-8")
            return TrackerConfig.from_yaml(p)

    def test_default_is_70(self):
        cfg = self._config_from_yaml(
            "tracker:\n  whitelist_orgs:\n    - Qwen\n"
        )
        self.assertEqual(cfg.max_param_billion, 70.0)

    def test_explicit_value(self):
        cfg = self._config_from_yaml(
            "tracker:\n  whitelist_orgs:\n    - Qwen\n"
            "  max_param_billion: 250\n"
        )
        self.assertEqual(cfg.max_param_billion, 250.0)

    def test_repo_whitelist_yaml_has_default(self):
        """Sanity check: the committed whitelist.yaml itself is honoured —
        nv8 ops are not running with the gate accidentally disabled."""
        wl = REPO_ROOT / "discover" / "whitelist.yaml"
        cfg = TrackerConfig.from_yaml(wl)
        self.assertIsNotNone(
            cfg.max_param_billion,
            "discover/whitelist.yaml lost its max_param_billion gate",
        )
        self.assertLessEqual(cfg.max_param_billion or 0, 250.0,
                             "committed gate is suspiciously high")


# ── (6) end-to-end through scan_round (the user-visible fix) ───────────────


class ScanRoundOversizeGateTests(unittest.TestCase):
    """The exact scenario from the user's complaint: moonshotai publishes
    Kimi-K2.6 (≥1T params), nv8 cannot host it, discover must drop it
    instead of forwarding it to the orchestrator queue."""

    def _cfg(self, max_b: float | None = 70.0):
        return TrackerConfig(
            whitelist_orgs=["moonshotai", "Qwen", "deepseek-ai"],
            min_downloads_30d=100_000,
            min_likes=200,
            from_date="2026-01-01T00:00:00Z",
            modality_pipeline_tags=["text-generation"],
            per_org_limit=50,
            trending_sweep_limit=100,
            max_param_billion=max_b,
        )

    def test_kimi_k26_1t_is_dropped_from_whitelist_bucket(self):
        """The reproducer for the user-reported bug."""
        api = FakeApi(by_author={
            "moonshotai": [
                # Authoritative: safetensors says 1T.
                FakeModel(
                    id="moonshotai/Kimi-K2.6-1T-Instruct",
                    last_modified="2026-05-20T00:00:00Z",
                    pipeline_tag="text-generation",
                    safetensors=FakeSafetensors(total=1_000_000_000_000),
                ),
                # A normal-size sibling must still come through.
                FakeModel(
                    id="moonshotai/Moonlight-7B",
                    last_modified="2026-05-20T00:00:00Z",
                    pipeline_tag="text-generation",
                    safetensors=FakeSafetensors(total=7_000_000_000),
                ),
            ]
        })
        new, stats = scan_round(api, self._cfg(70.0), Cursor())
        ids = {c.hf_id for c in new}
        self.assertEqual(ids, {"moonshotai/Moonlight-7B"})
        self.assertEqual(stats.excluded_oversize, 1)

    def test_id_only_fallback_also_drops_kimi(self):
        """If HF skipped the safetensors block (rare but happens during
        rolling card updates), the id regex must still catch ``1T``."""
        api = FakeApi(by_author={
            "moonshotai": [
                FakeModel(
                    id="moonshotai/Kimi-K2.6-1T",
                    last_modified="2026-05-20T00:00:00Z",
                    pipeline_tag="text-generation",
                    safetensors=None,
                )
            ]
        })
        new, stats = scan_round(api, self._cfg(70.0), Cursor())
        self.assertEqual(new, [])
        self.assertEqual(stats.excluded_oversize, 1)

    def test_threshold_72b_just_over_default(self):
        """Boundary: Qwen3-72B (just above the 70B default ceiling) is
        dropped; bumping the ceiling to 80 lets it through."""
        api = FakeApi(by_author={
            "Qwen": [
                FakeModel(
                    id="Qwen/Qwen3-72B-Instruct",
                    last_modified="2026-05-20T00:00:00Z",
                    pipeline_tag="text-generation",
                    safetensors=FakeSafetensors(total=72_000_000_000),
                ),
            ]
        })
        new70, stats70 = scan_round(api, self._cfg(70.0), Cursor())
        self.assertEqual(new70, [])
        self.assertEqual(stats70.excluded_oversize, 1)

        new80, stats80 = scan_round(api, self._cfg(80.0), Cursor())
        self.assertEqual({c.hf_id for c in new80}, {"Qwen/Qwen3-72B-Instruct"})
        self.assertEqual(stats80.excluded_oversize, 0)

    def test_gate_disabled_restores_legacy_behaviour(self):
        """``max_param_billion=None`` is the legacy escape hatch — every
        candidate that passed pre-PR#68 must still pass."""
        api = FakeApi(by_author={
            "moonshotai": [
                FakeModel(
                    id="moonshotai/Kimi-K2.6-1T-Instruct",
                    last_modified="2026-05-20T00:00:00Z",
                    pipeline_tag="text-generation",
                    safetensors=FakeSafetensors(total=1_000_000_000_000),
                )
            ]
        })
        new, stats = scan_round(api, self._cfg(None), Cursor())
        self.assertEqual({c.hf_id for c in new}, {"moonshotai/Kimi-K2.6-1T-Instruct"})
        self.assertEqual(stats.excluded_oversize, 0)

    def test_unknown_param_count_still_passes_fail_open(self):
        """A model with NO size signal (no safetensors block, no size
        token in the id) must still be admitted — otherwise an HF outage
        that strips the expand=safetensors response would zero out the
        candidates file, which is worse than letting ENGINE_SELECT bail
        downstream on the rare misclassification."""
        api = FakeApi(by_author={
            "apple": [
                FakeModel(
                    id="apple/openelm-instruct",
                    last_modified="2026-05-20T00:00:00Z",
                    pipeline_tag="text-generation",
                    safetensors=None,
                )
            ]
        })
        cfg = TrackerConfig(
            whitelist_orgs=["apple"],
            modality_pipeline_tags=["text-generation"],
            max_param_billion=70.0,
        )
        new, stats = scan_round(api, cfg, Cursor())
        self.assertEqual({c.hf_id for c in new}, {"apple/openelm-instruct"})
        self.assertEqual(stats.excluded_oversize, 0)

    def test_trending_bucket_also_gated(self):
        """A Kimi-K2.6 hitting the trending list (e.g. someone forks it
        into a random user repo and it goes viral) must be dropped too —
        the gate sits on every admit path, not just whitelist."""
        api = FakeApi(trending=[
            FakeModel(
                id="random-user/Kimi-K2.6-MoE-1T-finetune",
                last_modified="2026-05-20T00:00:00Z",
                downloads=500_000, likes=500,
                pipeline_tag="text-generation",
                safetensors=FakeSafetensors(total=1_000_000_000_000),
            ),
            FakeModel(
                id="random-user/Tiny-3B",
                last_modified="2026-05-20T00:00:00Z",
                downloads=500_000, likes=500,
                pipeline_tag="text-generation",
                safetensors=FakeSafetensors(total=3_000_000_000),
            ),
        ])
        new, stats = scan_round(
            api,
            TrackerConfig(
                whitelist_orgs=[],
                min_downloads_30d=100_000,
                min_likes=200,
                from_date="2026-01-01T00:00:00Z",
                modality_pipeline_tags=["text-generation"],
                max_param_billion=70.0,
            ),
            Cursor(),
        )
        self.assertEqual({c.hf_id for c in new}, {"random-user/Tiny-3B"})
        self.assertEqual(stats.excluded_oversize, 1)


# ── (7) Candidate serialization is forward + backward compatible ───────────


class CandidateSerializationTests(unittest.TestCase):
    """Older candidates.jsonl files (pre-PR#68) don't have the new field —
    load_candidates must still parse them."""

    def test_roundtrip_with_param_billion(self):
        c = Candidate(
            hf_id="moonshotai/Moonlight-7B",
            discovered_at="2026-05-20T00:00:00+00:00",
            reason="whitelist",
            source_org="moonshotai",
            param_billion=7.0,
        )
        line = c.to_jsonl()
        rt = Candidate.from_jsonl(line)
        self.assertEqual(rt.param_billion, 7.0)

    def test_legacy_jsonl_row_without_field_loads(self):
        legacy = json.dumps({
            "hf_id": "org/Legacy-7B",
            "discovered_at": "2026-04-01T00:00:00+00:00",
            "reason": "whitelist",
            "source_org": "org",
            "last_modified": None,
            "downloads": None,
            "likes": None,
            "pipeline_tag": "text-generation",
            "library_name": None,
            "private": False,
            "gated": False,
        })
        c = Candidate.from_jsonl(legacy)
        self.assertEqual(c.hf_id, "org/Legacy-7B")
        self.assertIsNone(c.param_billion)


# ── (8) _model_to_candidate populates the field on both shapes ─────────────


class ModelToCandidateParamBillionTests(unittest.TestCase):
    def test_attribute_shape_from_safetensors(self):
        m = FakeModel(
            id="Qwen/Qwen3-72B",
            pipeline_tag="text-generation",
            safetensors=FakeSafetensors(total=72_000_000_000),
        )
        c = _model_to_candidate(m, reason="whitelist")
        self.assertAlmostEqual(c.param_billion or 0.0, 72.0, places=1)

    def test_dict_shape_from_safetensors(self):
        m = {
            "id": "moonshotai/Kimi-K2.6-1T",
            "pipelineTag": "text-generation",
            "safetensors": {"total": 1_000_000_000_000},
        }
        c = _model_to_candidate(m, reason="whitelist")
        self.assertAlmostEqual(c.param_billion or 0.0, 1000.0, places=1)

    def test_id_regex_fallback(self):
        m = FakeModel(id="moonshotai/Kimi-K2.6-1T")  # no safetensors block
        c = _model_to_candidate(m, reason="whitelist")
        self.assertEqual(c.param_billion, 1000.0)


if __name__ == "__main__":
    unittest.main()
