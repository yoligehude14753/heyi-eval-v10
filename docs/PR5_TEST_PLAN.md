# PR#5 Test Plan — Restricted SHOWCASE runner

> rules § 阶段 2: "先写功能完整性清单（Happy Path + 所有 Sad Path + 边界场景）"

## Scope

Replace v9's cc-agent SHOWCASE path with a pure-Python in-process runner
that uses `heyi_engine.HeyiEngineClient` to:

1. Plan a small set of "interesting" prompts based on the model's
   claimed strengths (from `curated.json` + `modelcard.md`).
2. Send each prompt to the deployed engine (`deploy.json::base_url`).
3. Have heyi_engine score / comment on the response.
4. Write `runs/<run_id>/showcase.json` matching
   `sops/schemas/showcase.schema.json`.

After this PR there are **zero cc-agent docker spawns** anywhere in the
orchestrator. The `cc_agent/` package becomes a regular Python module
with the same trust profile as `curator/` — it never touches docker,
never gets a shell, only reads the run's metadata files and writes the
single `showcase.json` artifact.

## Trust boundary (the security thing)

- **Read** access (no write): `curated.json`, `metadata.json`,
  `modelcard.md`, `deploy.json`.
- **Write** access (no read of anything else): `showcase.json` only,
  under the run's directory.
- **No** access to docker socket, host filesystem outside the run dir,
  or other runs' artifacts.

There is no longer a "restricted container" because there is no
container — the runner is in-process Python with no shell/docker
imports.

## New module: `cc_agent/showcase_runner.py`

Public surface:

```python
execute_showcase(run, cfg, *, n_items=5, per_call_timeout_s=120.0,
                 plan_client=None, eval_http=None,
                 grade_client=None) -> StageResult
```

Why three injectable clients (plan, eval, grade):

- `plan_client` and `grade_client` default to a single
  `HeyiEngineClient(base_url=cfg.engine_url)`. They're distinct
  arguments so tests can verify INV-2: planning + grading hit the
  engine port, NOT `deploy.json::base_url`.
- `eval_http` defaults to `capability._http_post_chat` — same
  contract as CAPABILITY uses. Tests pin that it's the only function
  pointed at the eval container.

## Invariants under test

- **INV-2 (planning side)**: `plan_client.call(...)` and
  `grade_client.call(...)` only talk to heyi_engine
  (`http://127.0.0.1:10814` by default), never to deploy.json's base_url.
- **INV-2 (eval side)**: `eval_http(base_url=...)` only points at
  `deploy.json::base_url`, never at the engine.
- **INV-5 (showcase)**: every showcase item carries non-empty `id`,
  `rationale`, `prompt`, `actual` — the four schema-required fields.
  If grading fails we still emit the item with `actual` populated
  and `comment="grading skipped: ..."`; we do NOT drop the item.

## SHOWCASE (Happy)

| ID | Scenario | Pre | Action | Expect |
|---|---|---|---|---|
| H1 | full pipeline | curated + deploy + engine + eval all happy | execute_showcase(n_items=3) | ok=True; 3 items with all four required fields populated; summary present; pass schema |
| H2 | planning produces JSON array | engine returns clean JSON | same | items[i].rationale and items[i].prompt match plan output |
| H3 | per-item params honored | plan returns max_tokens / temperature in item | same | item.params persisted; eval_http receives same max_tokens |
| H4 | latency captured | eval responds in 100ms | same | item.latency_ms > 0 |
| H5 | summary from grade pass | grade_client returns "good at X" | same | showcase.summary == "good at X" |

## SHOWCASE (Sad)

| ID | Scenario | Pre | Action | Expect |
|---|---|---|---|---|
| S1 | deploy.json missing | DEPLOY not run | execute_showcase | ok=False; error="deploy.json not present"; showcase.json NOT written |
| S2 | curated.json missing | CURATE not run / cache miss | same | ok=True (degraded); planner falls back to a generic prompt; items still produced |
| S3 | planning LLM returns garbage | grade_client returns "no idea" not JSON | same | ok=True; falls back to one default item from modelcard sniff |
| S4 | planning HeyiEngineError | heyi_engine 503 | same | ok=False; error_kind="planning_failed"; showcase.json NOT written |
| S5 | eval engine 500 mid-stream | first 1 item works, rest 500 | same | ok=True; items with `comment="eval http 500"` still written |
| S6 | grade_client fails | grading HeyiEngineError | same | ok=True; items written with `comment="grading skipped: ..."`; summary falls back to count |

## SHOWCASE (Edge)

| ID | Scenario | Expect |
|---|---|---|
| E1 | n_items=0 explicit | ok=False; error="n_items must be >= 1" (schema requires minItems=1) |
| E2 | model card unreadable | modelcard.md missing → planner gets empty card; still produces 1 default item |
| E3 | duplicate ids in plan | runner deduplicates by id, keeps first |
| E4 | plan returns more items than n_items | runner truncates to n_items |
| E5 | plan returns fewer items than n_items | runner accepts what it gets, does not pad |

## Dispatcher (Routing)

| ID | Scenario | Action | Expect |
|---|---|---|---|
| D1 | SHOWCASE routes to showcase_runner | execute_stage(run, SHOWCASE, cfg, store) | cc_agent.showcase_runner.execute_showcase called; _docker_run_cc_agent NOT called |

After this PR, `_CC_STAGES` is empty. Removing it would touch
`execute_stage` again so we leave it as `set()` with a comment to make
PR#7's "remove cc-agent docker image refs" diff tighter.

## Test files

- `tests/test_showcase_runner.py` — H1-H5, S1-S6, E1-E5
- `tests/test_dispatcher_routes.py` — extends with D1 (SHOWCASE routing flip)

## Mocking strategy

`HeyiEngineClient` is replaced with a `MagicMock` whose `.call(...)`
returns sequenced `CallResult` instances. `eval_http` is mocked the same
way `capability._http_post_chat` is (see test_capability.py).

## Coverage target

- `cc_agent/showcase_runner.py` ≥ 85%
- Project total stays ≥ 80%.

## Out of scope

- Anthropic SDK / Claude integration: deliberately removed per the
  user's "都走本地的模型" directive. Local heyi_engine is the only LLM.
- Real model live-fire: PR#8 (E2E).
