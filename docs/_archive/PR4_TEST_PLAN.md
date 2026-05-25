# PR#4 Test Plan — Python CAPABILITY stage + dispatcher wiring

> rules § 阶段 2: "先写功能完整性清单（Happy Path + 所有 Sad Path + 边界场景）"

## Scope

Replaces v9's CC-driven CAPABILITY stage with a pure-Python runner that:

1. reads `runs/<run_id>/deploy.json` → discovers `base_url`
2. iterates over bundled micro-benchmark slices (gsm8k / mmlu / humaneval)
3. POSTs to `<base_url>/v1/chat/completions` per item
4. writes `runs/<run_id>/capability.json` matching
   `sops/schemas/capability.schema.json`

Also wires `orchestrator/stages_py.py` (PR#3) and the new
`orchestrator/capability.py` (this PR) into `orchestrator/stages.py`
dispatcher so DEPLOY / READY_WAIT / CAPABILITY / CLEANUP all go through
Python. SHOWCASE stays on the CC path until PR#5 restricts it.

| Stage | Source of truth |
|---|---|
| DISCOVER / CURATE / METADATA / ENGINE_SELECT | v9 stages.py (unchanged) |
| DEPLOY / READY_WAIT / CLEANUP | `stages_py.py` (PR#3) |
| **CAPABILITY** | **`capability.py` (this PR)** |
| SHOWCASE | CC-agent (gets restricted in PR#5) |

## New module: `orchestrator/capability.py`

Public surface:

```python
execute_capability(run, cfg) -> StageResult
```

Datasets bundled under `orchestrator/capability_data/`:
- `gsm8k_mini.jsonl` — 10 items, format `{id, prompt, expected_substring}`
- `mmlu_mini.jsonl` — 10 items, multi-choice with `expected_substring`
- `humaneval_mini.jsonl` — 5 items, code completion with `expected_substring`

Total = 25 prompts. ~30-90s wall-clock on Qwen2.5-0.5B-Instruct.

## Invariants under test

- **INV-2**: CAPABILITY only ever talks to `deploy.json::base_url`
  (the e9-* eval container). Never to heyi_engine. We assert this by
  patching the HTTP boundary and checking the URL.
- **INV-7**: every stage writes its artifact and validator can read it.

## CAPABILITY (Happy)

| ID | Scenario | Pre | Action | Expect |
|---|---|---|---|---|
| H1 | All items pass | engine returns expected_substring for every prompt | execute_capability | ok=True; pass_rate=1.0; len(results)==25; capability.json validates |
| H2 | All gsm8k pass | only gsm8k slice loaded (overrides) | same | results length matches slice size |
| H3 | Latency captured | engine responds in 100ms each | same | each result has latency_ms field > 0 |
| H4 | Token counts captured | response includes usage.prompt_tokens, completion_tokens | same | tokens_in / tokens_out present per item |

## CAPABILITY (Sad)

| ID | Scenario | Pre | Action | Expect |
|---|---|---|---|---|
| S1 | deploy.json missing | no DEPLOY artifact | execute_capability | ok=False; error="deploy.json not present"; capability.json *not* written |
| S2 | All items fail | engine echoes "no idea" for every prompt | run | ok=True (stage completes); pass_rate=0.0; len(results)==25; all `pass=False` |
| S3 | Engine HTTP 500 mid-run | first 5 ok, 6th-25th return 500 | run | ok=True; degraded; first 5 results pass; remaining have error field |
| S4 | Engine connection refused | all probes get status=0 | run | ok=True (does not crash); pass_rate=0; results[].error populated |
| S5 | Slice files missing | capability_data/*.jsonl removed | run | ok=False; error="no capability slices available"; capability.json not written |
| S6 | Wall-clock timeout | per-item slow + capability_timeout_s=2 | run | aborts mid-stream; ok=True; aborted_due_to="timeout"; partial results written |

## CAPABILITY (Edge)

| ID | Scenario | Expect |
|---|---|---|
| E1 | Empty model output | actual="" → pass=False, no exception |
| E2 | `expected_substring=""` | passes for any non-error response (degenerate case) |
| E3 | Case-insensitive match | `expected_substring="HELLO"`, actual=" hello world " → pass=True |
| E4 | Per-item timeout independent of wall-clock | one item hits HTTP timeout; rest continue |

## Dispatcher (Happy)

| ID | Scenario | Action | Expect |
|---|---|---|---|
| D1 | DEPLOY routed to stages_py | execute_stage(run, DEPLOY, cfg, store) | stages_py.execute_deploy called; not _docker_run_cc_agent |
| D2 | READY_WAIT routed to stages_py | same with READY_WAIT | stages_py.execute_ready_wait called |
| D3 | CAPABILITY routed to capability.py | same with CAPABILITY | capability.execute_capability called |
| D4 | CLEANUP routed to stages_py | same with CLEANUP | stages_py.execute_cleanup called |
| D5 | SHOWCASE still routed to CC | same with SHOWCASE | _docker_run_cc_agent called (until PR#5) |

## Test files

- `tests/test_capability.py` — H1-H4, S1-S6, E1-E4
- `tests/test_dispatcher_routes.py` — D1-D5

## Mocking strategy

CAPABILITY tests patch `capability._http_post_chat` (small wrapper around
urllib) with a `Mock(side_effect=[...])`. Each side-effect entry is a
`(status, body_dict)` tuple, matching `stages_py._http_get_json` shape.

Dispatcher tests patch `stages_py.execute_deploy` etc. with a sentinel
`Mock(return_value=StageResult(ok=True, …))` and assert it's the one
called.

## Coverage target

- `capability.py` ≥ 85%
- combined project coverage stays ≥ 80%

## Out of scope

- Real dataset sourcing (gsm8k uses 10 hand-curated items from the public
  test split; mmlu mini is 10 items across 3 subjects; humaneval mini is
  5 items from the canonical 164). Full eval is PR#8 (E2E).
- Concurrent request batching: serial-only for now. Multi-concurrency
  TTFT/TPS sweeps belong to a future "perf" stage.
