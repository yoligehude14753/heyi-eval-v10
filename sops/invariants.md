# Invariants (INV-1 ~ INV-5)

These are the non-negotiable rules that every pipeline component — CC agents,
orchestrator, watchdog, runner_validator — enforces. A violation is a stage
failure regardless of what any single component says.

| ID | Rule | Enforced by |
|---|---|---|
| INV-1 | Heyi-eval may only use GPU 4-7. GPUs 0-3 are owned by production `minimax` (vllm TP=4). | runner_validator (post-stage GPU memory snapshot); CC agent handbook |
| INV-2 | The following containers must never be stopped/removed/restarted/killed: `minimax`, `xrouter`, `xrouter-postgres`, `Xinf`, `Open-Webui`, `dify-*`. | docker-socket-proxy (deny-list); CC agent handbook |
| INV-3 | Every ephemeral container created by heyi-eval must have the `e8-` name prefix. | docker-socket-proxy (allow-list regex `^e8-`); runner_validator (post-stage scan); ready.schema.json |
| INV-4 | DEPLOY/CAPABILITY/SHOWCASE share one `e8-vllm` instance. Only CLEANUP stage performs `docker rm`. Premature cleanup of `e8-vllm` between stages is a violation. | runner_validator (between-stage container existence check); CC agent task.md |
| INV-5 | No `apt install` / `pip install` on host. CC agents may `docker pull` but must prefer host-already-present images. | CC agent handbook; eyeball at PR review |

## How runner_validator translates these

```
After stage S finishes (regardless of CC self-reported success):
  if S == DEPLOY:
    validate ready.json against ready.schema.json
    assert container_name starts with "e8-"        # INV-3
    assert docker inspect <container_name>.State.Status == "running"   # INV-4 entry
  if S == CAPABILITY or SHOWCASE:
    validate <stage>.json against schema
    assert e8-vllm container still running          # INV-4
  if S == CLEANUP:
    assert no container with name prefix "e8-" remains   # INV-4 exit

Between every stage:
  assert minimax container status == running                            # INV-2
  assert GPU 0-3 memory usage > 80GB each                               # INV-1

Any assertion failure → mark stage FAILED, notify outbox.
```

See `orchestrator/validator.py` for the actual implementation.
