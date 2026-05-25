# PR#33 / #34 / #35 Live Experiment — Auto-Repair on nv8

> Honest report. The user explicitly asked for deep experimentation:
> "我觉得这种升级或者版本不匹配的问题经常发生，这也是为什么我要你用 claude code 来部署，
> 我是希望能够自动适配解决这些模型部署的问题的，所以你可以实验一下，搞得定就继续用这个方案，
> 搞不定就算了标记为失败，我希望你深度实验"

## What we built

**PR#33** (`feat/pr22b-agent-runner @ 17b035d`) — auto-repair loop inside
`execute_deploy`:

1. After the first DEPLOY attempt fails with `early_exit / image_pull /
   docker_api`, classify the failure (`model_type_unknown` /
   `trust_remote_code_required` / `oom` / `image_pull` / `missing_dep` /
   ...).
2. Walk a rule-based strategy list (cheapest first):
   `add_trust_remote_code` → `lower_max_model_len` → `swap_vllm_image_latest`
   → `swap_engine_sglang` → `swap_engine_transformers`.
3. If all rules exhaust without a healthy attempt, escalate to
   MiniMax-M2.7 via `cc_agent.deploy_repair_agent` (free-form
   proposal, validated by a strict JSON schema, signed off by safety
   checks).
4. Every attempt (winning + rejected + agent escalation) recorded in
   `_meta/deploy_repair.json` so the Panel surfaces "auto-fixed via …".

**PR#34** (`@ 1d5e8f4`) — pipeline hygiene that PR#33 dogfooding exposed:

| Code | Symptom on nv8                                                 | Fix                                                          |
|------|----------------------------------------------------------------|--------------------------------------------------------------|
| a    | `Kijai/WanVideo_comfy` (diffusion-single-file) admitted        | discover blocklist `{diffusion-single-file, ComfyUI, …}`     |
| b    | hourly auto-enqueue pushed same hf_id 4× (16-row queue, 5 IDs) | `enqueue()` dedups against `queue.jsonl ∪ {pending,in_progress}` |
| c    | gemma-4-26B stuck at 6.5 GB for 56 min (hf-mirror CLOSE-WAIT)  | `model_stager` wall-clock budget (default 1800 s, env-tunable) |
| d    | orchestrator restart → 142 GB cache leaks + dedup falsely blocks retries | startup sweeper marks > 30 min-stale `in_progress` rows ABORTED |

**PR#35** (`@ 333d5e9`) — make PR#33 actually see the failures it was
designed for:

- PR#33's repair only fired on DEPLOY-stage `early_exit`. But every
  vLLM-incompatible 2026 model we threw at it (GLM-OCR, Qwen3.6-27B-GGUF,
  gemma-4-26B-AWQ) failed in **READY_WAIT** instead, because `_try_once`
  declared success after a single 0.5 s probe and the vLLM model-config
  validation takes 5-15 s to crash. PR#35 replaces the 0.5 s probe with
  a polling loop (default 20 s wall budget, iteration-count-driven in
  tests) — crashes now surface in the repair-aware path.
- New failure class `gguf_needs_file_path` recognised by the exact
  vLLM-0.11.0 GGUF log signature ("For GGUF: pass the local path of
  the GGUF checkpoint").
- New strategy `strategy_use_gguf_file_path` that rewrites
  `vllm_args.model_path_override` to point `--model` at a single
  `.gguf` file (preferring Q4_K_M, populated from staged-cache dir
  listing).
- `strategy_add_trust_remote_code` skips for `gguf_needs_file_path`
  (provably irrelevant — saves one wasted deploy attempt).

Tests: 786 → 799 → 819 (PR#34 adds 20, PR#35 adds 13), 7 skipped, 1
env-fragile pre-existing deselected. All 47 PR#33 tests still pass.

## Live experiment on nv8 (2026-05-23 night)

### Setup

- Models in queue: `zai-org/GLM-OCR`, `unsloth/Qwen3.6-27B-GGUF`
  (deliberately re-enqueued after PR#35 rollout because the previous
  PR#33-only runs all failed in READY_WAIT untouched).
- Engine pool: GPUs 5,6,7. ComfyUI on 4. MiniMax-M2.7 on 0-3.
- Code path: `333d5e9` (PR#33+34+35).

### What actually happened (timeline from orchestrator journal)

```
00:31:31  [DEPLOY] starting              GLM-OCR, attempt 1
00:31:39  [DEPLOY] first attempt failed: early_exit (model_type_unknown)
00:31:39  [DEPLOY] repair proposed 4 strategies: ['add_trust_remote_code',
          'swap_vllm_image_latest', 'swap_engine_sglang',
          'swap_engine_transformers']
00:32:15  [DEPLOY] OK in 43.2s            ← PR#33 win: swap_vllm_image_latest
00:33:38  [READY_WAIT] OK in 83.1s        ← vLLM nightly serves GLM-OCR
00:34:00  [CAPABILITY] OK in 22.3s        ← REAL evaluation: 14/30 pass
00:34:01  [PERF_BENCH] OK in 0.9s
00:34:16  [SHOWCASE] OK in 14.6s
00:34:16  [CLEANUP] OK

00:34:18  [DEPLOY] starting              Qwen3.6-27B-GGUF, attempt 1
00:34:26  [DEPLOY] first attempt failed: early_exit (gguf_needs_file_path)
          ← PR#35's NEW classifier matched
00:34:26  [DEPLOY] repair proposed 2 strategies: ['use_gguf_file_path',
          'swap_engine_transformers']
          ← PR#35's NEW strategy proposed FIRST (correct ordering)
00:34:44  [DEPLOY] OK in 25.4s            ← winner: swap_engine_transformers
                                            (use_gguf_file_path tried first,
                                             vLLM then died with
                                             "qwen35 GGUF arch not supported";
                                             chain continued)
00:34:44  [READY_WAIT] OK in 0.0s         ← transformers-runner up
00:34:44  [CAPABILITY] OK in 0.1s         ← 0/50 (`http 501` from runner)
00:34:44  [PERF_BENCH] OK in 0.1s
00:35:04  [SHOWCASE] OK in 19.6s          ← LLM-judge showcase still ran
```

### GLM-OCR — full provenance from `_meta/deploy_repair.json`

```json
{
  "stage": "DEPLOY_REPAIR",
  "failure_class": "model_type_unknown",
  "ok": true,
  "winning_strategy": "swap_vllm_image_latest",
  "attempts": [
    {"strategy": "(initial)",         "engine": "vllm",  "image": "vllm/vllm-openai:v0.11.0", "ok": false},
    {"strategy": "swap_vllm_image_latest", "engine": "vllm",  "image": "vllm/vllm-openai:latest",
     "duration_s": 27.0,  "ok": true}
  ],
  "agent_escalation": null
}
```

CAPABILITY sample (real output from the repaired container):
```
[gsm8k-001] cat=text_reasoning  pass=false  act='Janet makes $52.40 every day.'
[gsm8k-002] cat=text_reasoning  pass=false  act='7'
[gsm8k-003] cat=text_reasoning  pass=false  act='The increase in value is 150%. The profit is $30,000.'
```

The model is **genuinely generating text** — wrong on three GSM8K questions
shown, but 14/30 total are scored as pass. Without PR#33+PR#35 this run
would be terminally `failed`.

### Qwen3.6-27B-GGUF — partial recovery

```json
{
  "failure_class": "gguf_needs_file_path",
  "ok": true,
  "winning_strategy": "swap_engine_transformers",
  "attempts": [
    {"strategy": "(initial)",            "image": "vllm/vllm-openai:v0.11.0",
     "ok": false,
     "logs_tail": "...3. For GGUF: pass the local path of the GGUF checkpoint."},
    {"strategy": "use_gguf_file_path",   "image": "vllm/vllm-openai:v0.11.0",
     "notes": "point --model at GGUF file: /model/Qwen3.6-27B-Q4_K_M.gguf",
     "ok": false,
     "logs_tail": "...GGUF model with architecture qwen35 is not supported yet."},
    {"strategy": "swap_engine_transformers", "image": "heyi-eval/transformers-runner:v10",
     "duration_s": 0.7, "ok": true}
  ]
}
```

The chain went the right way architecturally:

1. PR#35's classifier matched the GGUF signature.
2. PR#35's strategy rewrote `--model` to `Qwen3.6-27B-Q4_K_M.gguf`.
3. vLLM then hit a *second* incompatibility (`qwen35` GGUF arch not
   supported in any released vLLM image), which we did NOT have a
   rule-based fix for.
4. The chain continued to `swap_engine_transformers`, which got a
   container up.

But the recovery is **superficial**: the `transformers-runner` image
accepts the GGUF path but its inference endpoint returns `http 501`
for actual completions. PR#33's contract (`container is up + /v1/models
returns 200`) was met, but evaluation didn't actually work — 0/50
CAPABILITY tests passed.

### Two real bugs PR#33 cannot detect (yet)

| Bug | Symptom | Impact |
|-----|---------|--------|
| **Container-healthy-but-inference-broken** | `/v1/models` returns 200, `/v1/chat/completions` returns 501 (transformers-runner + GGUF) | PR#33 marks repair successful; CAPABILITY produces 0/50 |
| **CAPABILITY status=OK despite 0% pass** | Capability stage reports `ok` regardless of pass rate. Whisperkit-coreml had same shape: 0/15 ASR pass, status=ok | Panel UI shows green when result is unusable |

These are **NOT inside PR#33's design contract** but are surfaced by it.
Documented separately for future PRs (likely PR#36: real-inference probe
in READY_WAIT; PR#37: capability-stage threshold).

## What we learnt — answer to the user's hypothesis

> "我希望能够自动适配解决这些模型部署的问题的，所以你可以实验一下，搞得定就继续用这个方案，
>  搞不定就算了标记为失败"

| Failure shape                                          | PR#33+#35 outcome                          | Verdict |
|--------------------------------------------------------|--------------------------------------------|---------|
| vLLM doesn't know `model_type=glm_ocr` (released after 0.11.0) | `swap_vllm_image_latest` → real serve     | ✓ **searched, fixed, evaluable** |
| vLLM rejects `--model=<dir>` for GGUF                  | `use_gguf_file_path` rewrote `--model`     | ✓ **classifier+strategy did the right thing** |
| `qwen35` GGUF arch unsupported in any vLLM image yet  | chain fell back to `swap_engine_transformers` | ⚠ **deployed but evaluation 0/50** |
| vLLM container hangs >10 min on huge models (e.g. Llama-3.1-405B) | INV-23 oversize gate (upstream)     | ✓ no metadata-only run touches DEPLOY |
| HF mirror CLOSE-WAIT, gemma stuck downloading 56 min  | PR#34c wall-clock budget                   | ✓ **hard-fails at 30 min, cleanup runs** |

**Continue with the approach.** The repair loop is doing real work, the
provenance is auditable, and the failure modes where it cannot fix
something (Qwen3.6-GGUF qwen35-arch, transformers-runner-can't-serve-
GGUF) are now identifiable failure shapes for the next PRs, not opaque
"orchestrator gives up".

## File map

- Code: `orchestrator/deploy_repair.py` (strategies+classifier),
  `orchestrator/stages_py.py::execute_deploy` (loop+poll),
  `cc_agent/deploy_repair_agent.py` (LLM escalation),
  `orchestrator/config.py` (knobs).
- Tests: `tests/test_pr33_deploy_repair.py` (31), `tests/test_pr34_pipeline_hygiene.py` (20),
  `tests/test_pr35_late_crash_repair.py` (13), `tests/test_stages_py_deploy.py` (47, kept green).
- Provenance: `runs/<run_id>/_meta/deploy_repair.json`.
- Panel sort order: prioritises `ok` with capability data; failed runs
  show truncated tail logs (PR#32).
