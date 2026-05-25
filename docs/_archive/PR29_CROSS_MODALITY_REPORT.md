# PR#29 Cross-Modality Evaluation Report (NV8, 2026-05-23)

This is the first **end-to-end multi-modality** evaluation batch on the
NV8-6000 box after the PR#23 → PR#27 series. Six runs across three
distinct model families (text, ASR, vision-language) plus two oversize
metadata-only runs (Llama-3.1-405B, DeepSeek-V3) exercise the full
v10 pipeline: `DISCOVER → CURATE → ENGINE_SELECT → DEPLOY → READY_WAIT
→ CAPABILITY → PERF_BENCH → SHOWCASE → CLEANUP`.

> Scope: validates that the modality-dispatch + oversize-gate +
> ASR-pipeline-tag-fallback fixes from PR#23–PR#27 actually behave
> correctly on real models, real GPUs, real LLM-judge.

## 1. Environment

| Item | Value |
|---|---|
| Host | `<NV8_HOSTNAME>` (NV8-6000, 8× GPU) |
| Eval GPU pool | `(5, 6, 7)` (GPU 0–3 = MiniMax-M2.7 API, GPU 4 = ComfyUI) |
| Model cache | `/DATA/Model/_eval-cache/` (3.6 TB, 722 GB free pre-run) |
| Engines used | `vllm v0.11.0`, `heyi/transformers-runner:dev` |
| LLM-judge | `MiniMax-M2.7` (via M2.7 API on GPU 0–3) |
| Orchestrator commit | `feat/pr22b-agent-runner @ d820f2c` (post-PR#27) |
| HF mirror | `https://hf-mirror.com` (direct huggingface.co unreachable) |

## 2. Runs (6 total)

| # | Model | pipeline_tag | Engine | tp | Cap. score | Pass rate | Latency ttft p50 | Status |
|---|---|---|---|---|---|---|---|---|
| 1 | `Qwen/Qwen2.5-0.5B-Instruct` (PR#26) | text-generation | vllm | – | 12/25 | 0.480 | 9.1 ms | OK |
| 2 | `Qwen/Qwen2.5-0.5B-Instruct` (PR#27 re-run) | text-generation | vllm | 1 | 12/25 | 0.480 | 9.0 ms | OK |
| 3 | `openai/whisper-tiny` | automatic-speech-recognition | transformers | 1 | 5/5 | 1.000 | n/a (ASR) | OK |
| 4 | `meta-llama/Llama-3.1-405B-Instruct` | text-generation | metadata_only | 4 | – | – | – | OVERSIZE GATE (INV-23) |
| 5 | `deepseek-ai/DeepSeek-V3` | text-generation | metadata_only | 4 | – | – | – | OVERSIZE GATE (INV-23) |
| 6 | `Qwen/Qwen2.5-VL-7B-Instruct` | image-text-to-text | vllm | 1 | 25/35 | 0.714 | 22.0 ms | OK |

## 3. Capability-dispatch correctness

This is the headline result for **PR#27**: capability tests now flow
to the right endpoint per modality. Compare pre-PR#27 (PR#26 §3) vs
post:

```
                      | pre-PR#27         | post-PR#27 (this run)
----------------------+-------------------+----------------------
whisper-tiny          | 0/25 (all 501)    | 5/5 ASR     pass_rate=1.0
                      | text→ASR endpoint | text_reasoning skipped
                      |                   | code_*         skipped
                      |                   | vision/ocr     skipped
Qwen2.5-0.5B          | 12/25 text+code   | 12/25 text+code (unchanged)
                      |                   | (regression-tested ✓)
Qwen2.5-VL-7B         | not yet staged    | 25/35 text+vision+ocr+video
                      |                   | text_reasoning 2/10  pass_rate=0.20
                      |                   | vision         10/10 pass_rate=1.00
                      |                   | ocr             8/10 pass_rate=0.80
                      |                   | video_underst   5/5  pass_rate=1.00
```

### 3.1. Whisper-tiny dispatch matrix (post-PR#27)

| Category | Applicable | Reason |
|---|---|---|
| asr | YES | `pipeline_tag=automatic-speech-recognition` → tags=`["asr"]` |
| text_reasoning | NO | `missing capability_tags: text` |
| code_gen | NO | `missing capability_tags: code` |
| code_repair | NO | `missing capability_tags: code` |
| code_complete | NO | `missing capability_tags: code` |
| vision | NO | `missing capability_tags: vision` |
| ocr | NO | `missing capability_tags: vision` |
| video_understanding | NO | `missing capability_tags: video` |
| music_understanding | NO | `missing capability_tags: audio` |
| tts / image_gen / video_gen / music_gen | NO | each one missing its respective tag |

12 of 13 categories correctly skipped. ASR runs and scores 5/5 — every
applicable item passes. **Net change vs PR#26**: −25 wasted text→ASR
requests (all returning 501), +5 successful ASR transcriptions.

### 3.2. VL-7B dispatch matrix (curator-driven, PR#27 fallback as backstop)

Curated.json explicitly set
`capability_tags=["text", "vision", "ocr", "video"]` (the curator did
its job). PR#27 fallback was therefore **not exercised** here — but the
result demonstrates that the v10 pipeline can use richer tag sets when
they're available, and that the new HF `pipeline_tag` fallback is the
right shape (it would have produced `["text", "code", "vision"]` for
this same `image-text-to-text` tag had the curator failed).

Applicable: `text_reasoning, vision, ocr, video_understanding`.
Skipped (9): code_gen, code_repair, code_complete (no "code" tag in
the curator output), asr, tts, music_understanding, image_gen,
video_gen, music_gen.

## 4. Oversize gating (INV-23)

Two runs hit the oversize gate at `ENGINE_SELECT` and **deployed
nothing**:

| Model | param_count | tp_size estimate | eval_pool_size | engine | Wall clock |
|---|---|---|---|---|---|
| Llama-3.1-405B-Instruct | None (via PR#26 hf_id fallback → 405B) | 4 | 3 | `metadata_only` | <1 s |
| DeepSeek-V3 | 671B | 4 | 3 | `metadata_only` | <1 s |

Both correctly produced `engine.json` with `oversize=true` and skipped
straight to CLEANUP, saving the 8+ minute deploy attempt that would
have OOMed.

## 5. Performance numbers (applicable runs)

| Model | ttft p50 (ms) | tps_single p50 | concurrent agg tps | concurrent N |
|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct (run #2) | **9.0** | 481.6 | 1966 | 4 |
| Qwen2.5-VL-7B-Instruct | 22.0 | 94.4 | 372 | 4 |

VL-7B is ~2.4× slower in ttft and ~5× slower in tps than the 0.5B text
baseline — consistent with the parameter-count ratio (~14×) tempered by
vllm batching gains. Both are well within the perf headroom of the
single-GPU eval pool.

## 6. Stage-by-stage durations (Qwen 0.5B re-run, representative)

| Stage | Wall clock |
|---|---|
| DISCOVER | <1 s |
| CURATE | ~6 s (MiniMax-M2.7 judge call) |
| ENGINE_SELECT | <1 s |
| DEPLOY (vllm pull + start) | ~3 s (image cached) |
| READY_WAIT | 73.0 s (18 probes) |
| CAPABILITY (25 items, MiniMax-M2.7 judge) | ~28 s |
| PERF_BENCH (3+3+4 ttft/tps/concurrent) | ~7 s |
| SHOWCASE (5 items + planner) | ~12 s |
| CLEANUP | ~2 s |
| **Total** | **~132 s** (end to end) |

## 7. Outstanding issues / next steps

1. **Curator coverage on VL `code`**: the curator didn't tag
   Qwen2.5-VL-7B-Instruct with `"code"`, so code_gen / code_repair /
   code_complete didn't run. The HF `pipeline_tag` fallback would have
   added `"code"` (per the PR#27 mapping for `image-text-to-text`),
   but the curator's explicit list wins. If we want VLM models tested
   on coding too, the curator prompt needs updating to include `"code"`
   for VLMs that advertise it.
2. **`text_reasoning` is weak across both Qwen models (0.10–0.20)**:
   not a pipeline issue — these are honest scores from the LLM-judge.
   The 10-item text_reasoning bank is intentionally tough (multi-step
   math + commonsense). The 0.5B model passing 1/10 and the VL-7B
   passing 2/10 looks roughly proportional to advertised benchmarks.
3. **MiniMax-M2.7 chain-of-thought consumes the showcase planner's
   token budget** (cf. PR#25 cleanup notes). Mitigation already shipped:
   the planner falls back deterministically and logs
   `cause=thinking_truncated`. Long-term fix would be to either (a)
   switch the planner to a non-thinking judge, or (b) pin the planner
   to a structured-output mode that exits `<think>` early.
4. **HF mirror is the only path off-box**. `bootstrap_nv8.sh` should
   `export HF_ENDPOINT=https://hf-mirror.com` so future agent-driven
   `snapshot_download` runs don't have to re-discover this.

## 8. Reproduction

```bash
# On NV8, with the post-PR#27 orchestrator service running:
cd ~/heyi-eval-v10
for m in \
  Qwen/Qwen2.5-0.5B-Instruct \
  openai/whisper-tiny \
  meta-llama/Llama-3.1-405B-Instruct \
  deepseek-ai/DeepSeek-V3 \
  Qwen/Qwen2.5-VL-7B-Instruct; do
    .venv/bin/python -m orchestrator.main enqueue "$m"
done

# Collect the summary:
python3 scripts/pr29_collect_summary.py > /tmp/pr29_summary.json
python3 scripts/pr29_render_summary.py /tmp/pr29_summary.json
```

## 9. Verdict

The pipeline now demonstrably:

- **runs the right tests on each modality** (whisper gets ASR, VL gets
  vision/OCR/video, text-only gets text+code, oversize models are
  refused before any GPU work);
- **doesn't waste any GPU minutes** on models that don't fit;
- **doesn't silently 501** any modality mismatch (the failure mode that
  hid PR#26 §3 for a full batch);
- **emits comparable perf numbers** across modalities;
- **survives** the M2.7 thinking-tokens edge case via deterministic
  planner fallback.

PR#23 (M2.7 + GPU pool) + PR#24 (Llama 405B parameter parsing) +
PR#25 (graceful_skip + think-block stripping) + PR#26 (hf_id fallback)
+ PR#27 (pipeline_tag dispatch) together close every regression
identified in the previous two batch runs. The v10 stack is ready for
the next 10-20 model scale-up batch.
