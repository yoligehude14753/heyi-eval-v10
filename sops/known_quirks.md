# Known quirks (running log from Phase 0 + later debugging)

Each entry: (1) what fails, (2) why, (3) the workaround. Append new entries as
discovered. CC agent handbook references this file.

## Q-001 · vllm `--model /DATA/Model/...` is interpreted as HF repo id

**Symptom**: `--model /DATA/Model/_eval-cache/Qwen2.5-0.5B-Instruct` → vllm tries
to fetch from huggingface.co and fails (timeout / 404).

**Root cause**: vllm only accepts container-internal paths or HF repo ids.
Absolute host paths that look like nonexistent HF org/name get treated as
repo ids.

**Workaround**: bind-mount cache root, pass container path:
```bash
docker run ... -v /DATA/Model/_eval-cache:/mnt vllm/vllm-openai:... \
  --model /mnt/Qwen2.5-0.5B-Instruct
```

Discovered in: E5 (initial deploy spike).

## Q-002 · `--gpus '"device=7"'` shell quoting trap

**Symptom**: `docker: Error response from daemon: invalid device request: "device=7"`.

**Root cause**: The double-quotes shown in some docker docs are meant to be
literal in the docker daemon protocol but bash strips them before docker sees
them. Wrapping in single quotes preserves the bash-level quotes, which then
become part of the device string vibe — docker doesn't like it.

**Workaround**: Just pass `--gpus device=7` (no inner quotes). Works.

Discovered in: E5.

## Q-003 · vllm port binds before it accepts

**Symptom**: `docker run -d ... vllm` returns immediately, `curl /v1/models`
within the next ~3-5s gets `ConnectionReset by peer`.

**Root cause**: vllm opens the listen socket early in startup, but the HTTP
handler isn't wired in until after model loading completes (60-120s on
sm_120 with CUDA graph capture).

**Workaround**:
```bash
sleep 3                           # let initial bind settle
for i in $(seq 1 60); do          # poll up to 180s
  if curl -fsS --max-time 2 http://localhost:18200/v1/models >/dev/null; then
    echo ready
    break
  fi
  sleep 3
done
```

Discovered in: E5, reconfirmed in E8.

## Q-004 · sm_120 + vllm v0.20.x silent crashes on some ops

**Symptom**: vllm container exits 5-10s after `docker run` with no obvious
error in logs; or first inference call returns `CUDA error: unsupported`.

**Root cause**: Blackwell sm_120 needs vllm v0.21.0+. Older builds were
compiled before sm_120 was widely tested.

**Workaround**: Use `vllm/vllm-openai:v0.21.0` exactly. ENGINE_SELECT pinned.

Discovered in: E1 (hardware survey).

## Q-005 · MiniMax `<think>` blocks burn the entire token budget

**Symptom**: CC agent receives an `end_turn` SSE chunk with empty `content`
after a few seconds of "running". CC interprets this as task completion ("I'm
done") and exits successfully — but produces no artifacts (false-success).

**Root cause**: CCR default `max_tokens=4096`. MiniMax's
`minimax_m2_append_think` behavior generates `<think>...</think>` reasoning
that consumes the full budget; nothing left for the actual answer.

**Workaround**:
1. CCR `max_tokens` raised to 16384 (in `ccr/config.json`)
2. `strip-think` transformer removes the `<think>...</think>` content entirely
   so Claude Code doesn't see it (in `ccr/transformers/strip-think.js`)

Discovered in: E8 attempt 1 (the canonical false-success case).

## Q-006 · `claude --dangerously-skip-permissions` refuses to run as root

**Symptom**: Claude Code prints "running as root is forbidden" and exits.

**Root cause**: Claude Code security policy: `--dangerously-skip-permissions`
is only allowed for non-root users (defense in depth).

**Workaround**: `run_cc.sh` creates a non-root `agent` user (uid 1100), adds
them to the host docker gid (matched at runtime via `stat -c %g
/var/run/docker.sock`), and `su - agent -c "claude --print ..."`. Settings
file written to `/home/agent/.claude/settings.json` to skip onboarding.

Discovered in: E8 attempt 2.

## Q-007 · node:20-bullseye-slim GLIBC too old for bind-mounted host docker

**Symptom**: Bind-mounting `/usr/bin/docker` from host into a
`node:20-bullseye-slim` container, then `docker --version` inside gives
"version `GLIBC_2.36' not found".

**Root cause**: Host (Debian bookworm or Ubuntu 24.04) docker CLI is
dynamically linked against newer GLIBC than bullseye ships.

**Workaround**: Use `node:20-bookworm-slim` base. GLIBC matches host.

Discovered in: E8 attempt 2.

## Q-008 · Debian bookworm apt sources live in `.sources`, not `sources.list`

**Symptom**: Set `/etc/apt/sources.list` to Aliyun, `apt update` still
hits debian.org (and times out from nv8 network).

**Root cause**: Bookworm shipped a new deb822-style format under
`/etc/apt/sources.list.d/debian.sources`, which takes precedence over the
legacy `/etc/apt/sources.list`.

**Workaround**: Either `rm -f /etc/apt/sources.list.d/*` before writing the
legacy file, or write the new format directly:
```
Types: deb
URIs: http://mirrors.aliyun.com/debian
Suites: bookworm bookworm-updates
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
```

Discovered in: E8 attempt 2.

## Q-009 · MiniMax container can disappear without anyone calling stop

**Symptom**: Between two E8 runs, `minimax` was no longer in `docker ps`.
No heyi-eval component touched it.

**Root cause**: Unknown (likely external scheduler / another team's tooling).
heyi-eval can't prevent it.

**Workaround**: 
- watchdog probes minimax every 30s; if down → mark current run FAILED,
  pause queue, send incident notify
- v9 does NOT auto-restart minimax — that container is owned by an external
  party and restarting it could conflict with their lifecycle
- if user wants restart automation, add later as an opt-in policy

Discovered in: E8 attempt 2/3.

## Q-010 · npm-installed CLI packages ship as ESM and don't expose `package.json`

**Symptom**: `RUN npm install -g <pkg> && node -e "require('<pkg>/package.json').version"`
build step fails with `Cannot find module '<pkg>/package.json'`.

**Root cause**: Modern Anthropic / CCR packages ship as ESM and their
`exports` field in package.json doesn't include `./package.json`. Node 20's
ESM-strict resolver refuses to load it.

**Workaround**: Don't use `require('<pkg>/package.json')` for sanity-checking
the install. Use the binary the package put on PATH:
```dockerfile
RUN npm install -g <pkg> && which <bin> && <bin> --version 2>&1 | head -3 || true
```

Discovered in: v9 local CCR / cc-agent build (2026-05).

## Q-011 · `chown -R agent:agent /workspace` crashes on :ro bind mounts

**Symptom**: cc-agent entrypoint `set -e` aborts at stage 3 with
`chown: changing ownership of '/workspace/handbook.md': Read-only file system`.

**Root cause**: The orchestrator bind-mounts handbook.md and task.md as
`:ro` (intentionally — the agent reads them, must never mutate them). But
`chown -R /workspace` traverses into those :ro mounts too and the kernel
refuses (correctly), which trips `set -e`.

**Workaround**: Only chown the writable subpath:
```bash
mkdir -p /workspace/runs
chown -R agent:agent /workspace/runs 2>/dev/null || true
```
Don't chown /workspace recursively.

Discovered in: v9 local cc-agent entrypoint smoke (2026-05).

## Q-012 · Apostrophes in inline prompts break `su -c "...'$PROMPT'..."`

**Symptom**: cc-agent stage 5 launches but the trace shows:
```
-bash: -c: line 22: syntax error near unexpected token `)'
```
Claude never runs.

**Root cause**: When passing a multi-line PROMPT containing a literal `'`
(e.g. "this stage's task") through `su - agent -c "claude --print '$PROMPT'"`,
the inner apostrophe terminates bash's single-quote run and what follows
is parsed as shell syntax.

This is a textbook false-success surface: claude returns rc != 0 and the
orchestrator sees a "failed" stage, but the root cause is shell quoting,
not the model.

**Workaround**: Write the prompt to a tmp file via heredoc (no shell
expansion of single quotes inside `<<EOF`), then have claude read it from
stdin:
```bash
cat > /tmp/cc_prompt.txt <<PROMPT_EOF
You are an evaluation engineer agent...
(this stage's task is to ...)
PROMPT_EOF
chown agent:agent /tmp/cc_prompt.txt
su - agent -c "... claude --print < /tmp/cc_prompt.txt"
```

Robust to any prompt content (apostrophes, double quotes, backticks, $).

Discovered in: v9 local cc-agent entrypoint smoke (2026-05).

## Q-013 · Mock SSE upstream must `Connection: close` after `[DONE]`

**Symptom**: CCR + strip-think correctly strips `<think>` blocks but the
stream never terminates — client hangs to its read timeout (15s default),
and the Anthropic shape's `message_stop` event never arrives.

**Root cause**: CCR converts upstream OpenAI SSE → Anthropic SSE. Its
upstream-reading loop only emits `message_stop` after the underlying
`fetch` body resolves to done. With HTTP/1.1 keep-alive (the BaseHTTPServer
default in Python), the upstream socket doesn't close after `[DONE]\n\n`,
so CCR's reader keeps polling.

**Workaround (for mock providers used in tests)**: Force `Connection: close`
on the SSE response headers. Real vllm and real MiniMax already do this.

**Note**: This was discovered while writing a Python `BaseHTTPServer`-based
mock; production providers don't have this quirk.

Discovered in: v9 local CCR smoke (2026-05).

---

## Q-014 · cc-agent must run as host-uid, not root

**Symptom**: After cc-agent completes (or is killed), the orchestrator on
the host tries to `save_run()` → write `state.json.tmp` and gets
`PermissionError: [Errno 13] Permission denied`. The run stays at
`in_progress` forever; no `run_failed` event is emitted.

**Root cause**: `cc-agent`'s container default user is `root` (uid 0). It
writes `trace_<STAGE>.jsonl`, `ready.json`, etc. into `/workspace/runs/<run_id>/`
which is a bind-mount of `/home/ai/heyi-eval-data/runs/<run_id>/`. Files
land on the host as `root:root`. The orchestrator (running as `ai`, uid 1000)
then can't write into the same dir.

**Wrong fix (tried first, broken)**: pass `--user $(id -u):$(id -g)` to
`docker run`. This breaks `apt-get install`, `useradd`, `groupadd`,
`chown` in run_cc.sh's stage 1-3 (all need root).

**Correct fix (implemented)**:
1. Orchestrator passes `HOST_UID` / `HOST_GID` env vars to cc-agent
   (`stages.py::_docker_run_cc_agent`).
2. Container still runs as root, so apt + useradd still work.
3. `run_cc.sh` creates the `agent` user with `useradd -u $HOST_UID -g $HOST_GID`
   instead of hardcoded 1100.
4. Existing `chown -R agent:agent /workspace/runs` (already in run_cc.sh)
   then leaves files owned by the host uid, satisfying the orchestrator's
   later writes.

Discovered in: v9 first real T11 dry-run on nv8 (2026-05).

---

## Q-015 · Orchestrator must catch pipeline exceptions and emit run_failed

**Symptom**: When `run_pipeline()` raises an uncaught Python exception
(e.g. cascading from Q-014), the daemon's loop printed `pipeline raised
unexpectedly: ...` and continued. The run stayed at `in_progress` in
SQLite forever. The outbox got `run_started` but never `run_failed`. The
operator had no way to know the run was dead — silent broken queue
consumer is the worst possible outcome.

**Fix**: `main.py::cmd_loop` now catches the exception, marks the run
`FAILED`, calls `store.save_run()`, and writes a `run_failed` event via
`notify.run_failed()` to the outbox. The mac-side launchd sync_agent
then picks it up on its next poll cycle and propagates to
alld → WeChat + desktop notification with `priority="high"`.

Discovered in: v9 first real T11 dry-run on nv8 (2026-05).

---

## Q-016 · `cc_agent_max_turns=100` enables retry-hell

**Symptom**: When the underlying environment fails (e.g. CUDA OOM
because some other tenant owns all GPUs), claude doesn't just give up —
it keeps observing the failure, hypothesizing fixes, and retrying. With
`--max-turns 100` it can burn 5-10+ minutes way past `task.md`'s stated
180s readiness deadline.

**Workaround**: Reduced default to 40 (in `config.py::cc_agent_max_turns`).
40 is enough for the healthy paths (DEPLOY = plan-1, deploy-1, poll-1,
ready-write-1 = ~4 turns; CAPABILITY = 5 questions = ~10 turns;
SHOWCASE = ~15 turns). Override per-stage via
`HEYI_EVAL_CC_AGENT_MAX_TURNS` env if a SHOWCASE plan needs more headroom.

**Real fix (later)**: per-stage turn caps, and a strict wall-clock kill
inside `task.md` itself so claude self-aborts at the deadline instead of
retrying.

Discovered in: v9 first real T11 dry-run on nv8 (2026-05).

---

## Q-017 · `> $TRACE` redirect opens the file as the outer-shell user

**Symptom**: After Q-014 (HOST_UID passthrough) was fixed, `state.json`,
`ready.json`, and `_meta/` were all correctly owned by `ai:ai` on the
host — BUT `trace_DEPLOY.jsonl` was still `root:root` (67 KB,
post-write). When inspecting from the host, you can read it (mode 644)
but can't delete it without sudo.

**Root cause**: In `run_cc.sh`, claude is launched as:

```bash
su - "$AGENT_USER" -c "claude --print ..." > "$TRACE" 2>&1
```

The `> "$TRACE"` redirect is handled by the OUTER bash (running as
container root), not by the `su` subshell. So root opens TRACE for
writing → file is created root-owned. The subsequent `su` just streams
its stdout into that already-open root-owned fd.

**Fix**: Pre-create the trace file owned by `$AGENT_USER` BEFORE the
redirect. `>` on an existing file only truncates content; it does NOT
chown:

```bash
touch "$TRACE" && chown "$AGENT_USER:$AGENT_GID" "$TRACE"
su - "$AGENT_USER" -c "..." > "$TRACE" 2>&1
```

Now stdout from the su subshell writes into the pre-existing ai-owned
file, and ownership stays ai:ai through truncation.

Discovered in: v9 second real T11 dry-run on nv8 (2026-05).

---

## Q-018 · sync_agent must alarm on prolonged nv8 unreachability

**Symptom**: 2026-05-21 nv8 went silent for 3+ hours after T11 DEPLOY
succeeded (likely OOM cascade from concurrent glm-51 + e8-vllm + capability
stage). Mac launchd `com.heyi-eval.sync-agent` kept logging
`ssh fetch failed rc=255 stderr=ssh: connect timeout` to its stderr file
every 10s, but the user never knew because:

1. The whole point of sync_agent was to deliver outbox events to alld
   for WeChat/desktop notification.
2. When nv8 is down, the outbox is unreachable — so "the messenger
   itself is down" cannot be conveyed by reading the outbox.
3. The fix everyone reaches for ("just monitor sync_agent's log")
   defeats the purpose: the operator should not need to babysit the
   babysitter.

**Fix**: `sync_agent.main.run_loop` now tracks consecutive ssh-fetch
failures. When `UNREACHABLE_AFTER` (default 6 ≈ 1 minute at default
10s interval) failures stack up, it POSTs a `nv8_unreachable` event
directly to alld (which IS local to the mac, so the post works even
when nv8 is offline). The body explains the likely cause (nv8 down,
tailscaled stuck, sshd not accepting) and suggested action (try ssh
from another machine, reboot nv8).

When nv8 comes back, sync_agent POSTs a `nv8_recovered` event with the
total downtime. While still down, it re-reminds every `REMIND_EVERY`
(360 = roughly 1 hour) rounds so the alert doesn't get forgotten.

Both thresholds are configurable via env (`HEYI_SYNC_UNREACHABLE_AFTER`,
`HEYI_SYNC_REMIND_EVERY`) for future tuning without code changes.

Discovered in: v9 nv8 outage during T11 (2026-05).

## Q-019 · CCR returns 500 silently when upstream LLM container is Exited

**Symptom**: After a prod LLM container (minimax, glm-51, …) has Exited
or its process has crashed, CCR (claude-code-router) keeps reporting
`/health` 200 OK but every `/v1/messages` POST returns HTTP 500 with
body:
```json
{"error":{"message":"fetch failedTypeError: fetch failed\n   at node:internal/deps/undici/...", "type":"api_error"}}
```

CCR returns 500 in **~10 ms** — fast enough that it looks like a transient
network issue, not a configured-upstream-is-dead. There's nothing in the
CCR container logs to indicate which upstream (the relevant request never
even reaches CCR's outbound socket — it bails at undici dial time).

**Root cause**: CCR reads `Providers[].api_base_url` from config.json at
startup; if the target (e.g. `http://127.0.0.1:10814` for minimax) is
not listening, every forwarded request fails at `fetch()` in undici.
CCR's own `/health` only checks **its own** HTTP server, not the
configured upstream — by design.

**Workaround (in code, not infra)**:
1. **Pre-flight probe** before stages that talk to CCR.
   `curator/health.py::probe_ccr()` does a tiny POST `/v1/messages` and
   classifies the response. Used by `_execute_curate_stage` so we fail
   fast (~1s) instead of attempting 30s of enrichment work.
2. **Incident on detect**: failure path POSTs an `incident` event to the
   outbox via `notify.incident(what="curator-ccr-upstream", …)`. From
   there the sync_agent → alld → wechat chain alerts the operator.
3. **Degraded, not failed**: the run continues. CURATE writes a degraded
   `curated.json` with `_llm_meta.parse_error="ccr-preflight-fail: …"`.
   METADATA still produces useful output by joining the degraded
   `curated.json` with HF Hub's structured fields (license,
   pipeline_tag, downloads, library_name). ENGINE_SELECT continues
   normally on the HF-derived modality.

**Why not restart the LLM?**: per the heyi-fleet operating rules (see
`_workspace/40-heyi-fleet.md`), the eval pipeline must never touch
production containers. Restarting `minimax` or `glm-51` is the responsibility
of whoever owns those services. Our role is to (a) detect quickly, (b) tell
the operator, (c) keep delivering whatever value we still can.

Discovered in: 2026-05-21, both `minimax` (vllm) and `glm-51` (sglang)
went down within the same day on nv8 — curator stage started returning
gibberish-degraded output instead of failing loudly. Added pre-flight +
incident before this manifested as silent corruption of the data
pipeline.

## Q-020 · Upstream :10814 model name drifts (MiniMax → Kimi → …)

**Symptom**: CCR returns HTTP 404 with body
`The model 'MiniMax-M2.7' does not exist` even though `:10814` is listening
and serving requests fine.

**Root cause**: `:10814` is shared between several manually-launched vllm
instances on nv8 (whoever starts last wins the port). The eval pipeline's
CCR config hard-codes `MiniMax-M2.7` as the upstream model name. When
someone (a human operator from ssh, another agent, etc.) replaces the
process with `vllm ... --served-model-name Kimi-K2.6`, CCR still asks for
`MiniMax-M2.7` and the upstream rejects it. Q-019's probe correctly
classifies this as upstream-unhealthy because the response body contains
`HTTP 404`, but the actual fix is to update the config.

**Workaround**:
1. The CCR config (`ccr/config.json`) ships with **two providers** —
   `nv8-minimax` and `nv8-kimi` — sharing the same `api_base_url`. Only
   the `Router.default` picks which one is "live". Switching is a 1-line
   change + container recreate.
2. The orchestrator's `HEYI_EVAL_CURATOR_MODEL` env (set via a systemd
   drop-in at `/etc/systemd/system/heyi-eval-orchestrator.service.d/`)
   must match the active model name.
3. To switch:
   ```bash
   # 1) point Router.default in ccr/config.json to nv8-kimi,Kimi-K2.6
   # 2) recreate ccr (see Q-021 for why "docker restart" isn't enough)
   # 3) sudo systemctl edit heyi-eval-orchestrator  (set HEYI_EVAL_CURATOR_MODEL)
   # 4) sudo systemctl restart heyi-eval-orchestrator
   ```

**Better long-term fix**: have the probe discover the actual served model
name via `GET /v1/models` and configure CCR accordingly. Backlog item;
not blocking.

Discovered in: 2026-05-21 — both minimax and glm-51 had been replaced on
:10814 by a Kimi-K2.6 vllm instance launched directly from ssh. Pre-flight
probe correctly flagged it as upstream-unhealthy.

## Q-021 · `docker restart heyi-eval-ccr` lands in restart-loop (stale daemon PID)

**Symptom**: `docker restart heyi-eval-ccr` returns 0 but the container
oscillates "Restarting (0)" forever. `docker logs` only shows
`Loaded JSON config ... claude-code-router server is running` repeated
once per restart cycle. Port :3457 never reaches LISTEN.

**Root cause**: `ccr start` daemonizes (writes a PID file to
`/root/.claude-code-router/.daemon.pid`, then the parent process exits).
The container's PID 1 is `tini`, which sees its only child exit, so it
exits too, so docker restarts the container. On the second start, ccr
finds an existing (now stale) PID file, exits with "already running",
loop continues. Eventually the file might get reaped by ccr itself, but
the loop usually keeps it stale.

**Workaround**:
1. **Do not use `docker restart`** on the ccr container — `docker stop`
   then `docker run` a fresh container instead.
2. The bootstrap wrapper (`deploy/bootstrap_nv8.sh::ccr_recreate()`) now
   does the right thing: `docker rm -f` then `docker run -d` with a
   `bash -c "rm -f .pid && ccr start && tail -f /dev/null"` wrapper so
   PID 1 survives.

**Better long-term fix**: patch `ccr start` to support `--foreground`
or add a `node ccr.js` entrypoint that doesn't daemonize. Upstream PR
worth filing.

Discovered in: 2026-05-21 — after running `docker cp` to update
config.json followed by `docker restart`. Took 15 minutes to debug
because every log line said "server is running".
