# PR#3 Test Plan — Python stages: DEPLOY + READY_WAIT + CLEANUP

> rules § 阶段 2: "先写功能完整性清单（Happy Path + 所有 Sad Path + 边界场景）"

## Scope

This PR replaces three v9 cc-agent stages with pure-Python implementations
that talk to the Docker daemon via `docker-py` SDK. CAPABILITY and SHOWCASE
remain on the v9 CC path until PR#4 and PR#5 respectively.

| Stage | v10 module | Entry point |
|---|---|---|
| DEPLOY | `orchestrator/stages_py.py` | `execute_deploy(run, cfg) -> StageResult` |
| READY_WAIT | `orchestrator/stages_py.py` | `execute_ready_wait(run, cfg) -> StageResult` |
| CLEANUP | `orchestrator/stages_py.py` | `execute_cleanup(run, cfg) -> StageResult` |

## Invariants under test

- **INV-1**: container layer touches only `e9-*` prefix. Cleanup refuses to remove containers without the `heyi_eval_run=<run_id>` label AND `e9-*` name prefix.
- **INV-4**: orchestrator owns the docker socket; cc-agent does not appear here.
- **INV-7**: every stage writes its artifact and validator can read it.

## DEPLOY (Happy)

| ID | Scenario | Pre | Action | Expect |
|---|---|---|---|---|
| H1 | vllm engine, text model | `engine.json` says vllm | call execute_deploy | container `e9-vllm-<short>` running, `deploy.json` has container_name + base_url + started_at + image |
| H2 | sglang engine | engine=sglang | same | container `e9-sglang-<short>` running |
| H3 | transformers engine | engine=transformers | same | container `e9-tf-<short>` running with HTTP wrapper port |
| H4 | vllm with custom `max_model_len` | engine.json `vllm_args.max_model_len=8192` | same | command includes `--max-model-len 8192` |
| H5 | vllm with tp>1 | `vllm_args.tensor_parallel_size=2` | same | command includes `--tensor-parallel-size 2`; container gets `--gpus all` |
| H6 | Label invariant | any happy path | same | container labels include `heyi_eval_run=<run_id>`, `heyi_eval_stage=DEPLOY`, `heyi_eval_engine=vllm` |

## DEPLOY (Sad)

| ID | Scenario | Pre | Action | Expect |
|---|---|---|---|---|
| S1 | docker daemon down | `docker.from_env` raises | call execute_deploy | StageResult.ok=False; error contains "docker daemon"; no deploy.json |
| S2 | image pull fails | `client.containers.run` raises `ImageNotFound` | same | ok=False; error mentions image name |
| S3 | model_path missing | `<cache>/<hf_local_dir>` doesn't exist | same | ok=False; error="model path not found: ..."; no docker call |
| S4 | engine.json missing | no `_meta/engine.json` | same | ok=False; error="engine.json not present (run ENGINE_SELECT first)" |
| S5 | unknown engine | engine.json `engine="foobar"` | same | ok=False; error mentions valid engines |
| S6 | port already in use | docker raises `APIError 409` | same | ok=False; error="port 18200 already in use" + suggests cleanup |
| S7 | container exits immediately | container.status='exited' within 5s | same | ok=False; error includes last 50 log lines; container is force-removed |

## DEPLOY (Edge)

| ID | Scenario | Expect |
|---|---|---|
| E1 | stale container w/ same name | If name matches `e9-vllm-<short>` and labels match same run_id and status=running, *reuse*; else force-remove + respawn |
| E2 | `vllm_args` with hyphens vs underscores | normalize both `max_model_len` and `max-model-len` to `--max-model-len` flag |
| E3 | very long run_id | short suffix truncated to 8 alphanumeric chars, container name stays ≤ 63 chars |

## READY_WAIT (Happy)

| ID | Scenario | Action | Expect |
|---|---|---|---|
| H1 | engine ready within first probe | container.status=running, `/v1/models` returns 200 with model | ok=True; `ready.json` has model_id + elapsed_s + base_url |
| H2 | engine ready after 30s | first 3 probes get 503, 4th returns 200 | ok=True; elapsed_s > 30 |
| H3 | model_id matches engine choice | vllm → /v1/models data[0].id used | ok=True |

## READY_WAIT (Sad)

| ID | Scenario | Action | Expect |
|---|---|---|---|
| S1 | timeout | probe never returns 200 | ok=False; error="timeout after <N>s"; ready.json *not* written |
| S2 | container died mid-wait | container.status='exited' | ok=False; error="container exited: <last_logs>"; no further probes |
| S3 | deploy.json missing | no DEPLOY artifact | ok=False; error="deploy.json not present (run DEPLOY first)" |
| S4 | base_url unreachable (ConnectionRefused) | retry for full timeout | same as S1 |

## READY_WAIT (Edge)

| ID | Scenario | Expect |
|---|---|---|
| E1 | /v1/models returns 200 but `data=[]` | keep probing; treat as not-ready (engine started, model not loaded yet) |
| E2 | probe interval honors backoff | first 5 probes within 10s, then 1 per 5s |

## CLEANUP (Happy)

| ID | Scenario | Pre | Action | Expect |
|---|---|---|---|---|
| H1 | one engine container | `e9-vllm-<short>` running w/ matching label | execute_cleanup | container removed; `cleanup.json` has `removed: [name]`; ok=True |
| H2 | no containers | nothing matching | same | ok=True; cleanup.json has `removed: []` |
| H3 | multiple e9 containers | `e9-vllm-<short>` + `e9-cc-showcase-<short>` both labeled | same | both removed; cleanup.json lists both |
| H4 | container already stopped | status='exited' | same | rm without stop call; removed=[name] |

## CLEANUP (Sad / INV-1 protection)

| ID | Scenario | Pre | Action | Expect |
|---|---|---|---|---|
| S1 | container labeled but name doesn't start with `e9-` | label match but name="minimax" | execute_cleanup | **skip + raise** INV-1 violation in error log; refuse to remove |
| S2 | container e9-* but missing label | name match, no label | same | skip; logged as "orphan e9 container, not cleaned" — operator decides |
| S3 | docker daemon down | client raises | same | ok=False; error="docker daemon"; cleanup.json not written |
| S4 | `remove()` raises NotFound mid-loop | race: container vanished | same | treat as already-removed, continue, list as removed |
| S5 | `remove()` raises APIError | docker rm hung | retry once with force=True; if still failing, ok=False with container name |

## CLEANUP (Edge / INV-1 belt-and-suspenders)

| ID | Scenario | Expect |
|---|---|---|
| E1 | label has different run_id | not matched by `--filter label=heyi_eval_run=<run>`, never seen |
| E2 | xrouter/minimax/kimi-k26 production containers | confirm filter never returns them (no `heyi_eval_run` label exists on them by design) — tested via mock that lists all containers and asserts these names are never in the operation set |
| E3 | dry_run flag | `execute_cleanup(dry_run=True)` lists what would be removed without calling .remove() |

## Test files

- `tests/test_stages_py_deploy.py` — H1-H6, S1-S7, E1-E3
- `tests/test_stages_py_ready_wait.py` — H1-H3, S1-S4, E1-E2
- `tests/test_stages_py_cleanup.py` — H1-H4, S1-S5, E1-E3

## Mocking strategy

`docker.from_env()` is replaced with a `MagicMock` that:
- `.containers.run(...)` returns a `Mock(name=..., status=..., labels=..., logs=...)`
- `.containers.list(filters=...)` filters by the passed label kwargs
- `.containers.get(name)` raises `NotFound` if not in fixture set

HTTP probes (`urllib.request.urlopen`) are mocked at the `stages_py._http_get_json` boundary so test code controls the timeline (use a list of (status, body) tuples).

## Coverage target

`stages_py.py` ≥ 85% (logic-heavy module, well-mockable I/O).
