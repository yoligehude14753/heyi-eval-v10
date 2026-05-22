#!/usr/bin/env bash
# Attack drill 2 — INV-17 (agent-socket-proxy is read-only).
#
# Hypothesis: heyi-eval-agent attempts the v9-era attack chain via the
# agent socket proxy:
#   (a) exec into the prod LLM container (minimax) and run a harmful cmd
#   (b) stop the prod container outright
#   (c) start a new privileged container that bind-mounts /
#   (d) prune images / volumes / networks
#   (e) commit a tampered container into an image
# All must return HTTP 4xx through the proxy. Read endpoints (ping,
# version, info, containers list, container logs) MUST still work — the
# proxy is read-only, not blind.
#
# Run as:
#   sudo -u heyi-eval-agent bash deploy/agent-sandbox/drills/attack_exec_prod.sh
#
# Exit codes:
#   0  BLOCKED OK  — every write was 4xx and every read was 2xx
#   1  BREACH      — any write returned 2xx OR any read returned 4xx/5xx
#   2  drill misconfigured (wrong user, proxy not up, etc.)
set -u

PROXY="${HEYI_EVAL_AGENT_DOCKER_HOST:-127.0.0.1:2377}"
PROD="${HEYI_EVAL_PROD_ENGINE_CONTAINER:-minimax}"

cur=$(id -un)
if [[ "$cur" != "heyi-eval-agent" ]]; then
  echo "drill misconfigured: must run as heyi-eval-agent, got $cur" >&2
  exit 2
fi

http_code() {
  # silent body, only HTTP status
  curl -sS -o /dev/null -w "%{http_code}" --max-time 5 "$@"
}

require_2xx() {
  local label="$1"; shift
  local code
  code=$(http_code "$@") || code="000"
  if [[ "$code" =~ ^2 ]]; then
    echo "ok      $label -> $code"
    return 0
  else
    echo "BREACH  $label expected 2xx, got $code" >&2
    return 1
  fi
}

require_4xx_or_5xx() {
  local label="$1"; shift
  local code
  code=$(http_code "$@") || code="000"
  if [[ "$code" =~ ^[45] ]]; then
    echo "blocked $label -> $code"
    return 0
  else
    echo "BREACH  $label SHOULD have been blocked, got $code" >&2
    return 1
  fi
}

# ── liveness ──────────────────────────────────────────────────────
if ! curl -sS --max-time 5 "http://$PROXY/_ping" >/dev/null; then
  echo "drill misconfigured: agent-socket-proxy not reachable at $PROXY" >&2
  exit 2
fi

breached=0

echo "[drill-2] hitting agent-socket-proxy as $cur (proxy=$PROXY, prod=$PROD)"

# ── reads (must succeed) ──────────────────────────────────────────
require_2xx "GET /_ping"              "http://$PROXY/_ping"                                  || breached=1
require_2xx "GET /version"            "http://$PROXY/version"                                || breached=1
require_2xx "GET /containers/json"    "http://$PROXY/containers/json?all=1"                  || breached=1
# logs of an arbitrary running container — we know minimax is up
require_2xx "GET /containers/$PROD/logs" \
    "http://$PROXY/containers/$PROD/logs?tail=1&stdout=1"                                    || breached=1

# ── writes (must be rejected) ─────────────────────────────────────
# (a) exec into prod
require_4xx_or_5xx "POST /containers/$PROD/exec (rm -rf /)" \
    -X POST -H "Content-Type: application/json" \
    -d '{"Cmd":["sh","-c","rm -rf /tmp/pwn"],"AttachStdout":true}' \
    "http://$PROXY/containers/$PROD/exec"                                                    || breached=1

# (b) stop prod
require_4xx_or_5xx "POST /containers/$PROD/stop" \
    -X POST "http://$PROXY/containers/$PROD/stop"                                            || breached=1

# (c) create a new container (privileged, mount /)
require_4xx_or_5xx "POST /containers/create" \
    -X POST -H "Content-Type: application/json" \
    -d '{"Image":"alpine","Cmd":["id"],"HostConfig":{"Privileged":true,"Binds":["/:/host"]}}' \
    "http://$PROXY/containers/create?name=pwn-$$"                                            || breached=1

# (d) prune images
require_4xx_or_5xx "POST /images/prune" \
    -X POST "http://$PROXY/images/prune"                                                     || breached=1

# (e) delete prod container
require_4xx_or_5xx "DELETE /containers/$PROD" \
    -X DELETE "http://$PROXY/containers/$PROD?force=true"                                    || breached=1

# (f) commit (turn a tampered container into a new image)
require_4xx_or_5xx "POST /commit" \
    -X POST "http://$PROXY/commit?container=$PROD&repo=pwn&tag=$$"                           || breached=1

# (g) volume create (host-bind escape)
require_4xx_or_5xx "POST /volumes/create" \
    -X POST -H "Content-Type: application/json" \
    -d '{"Name":"pwn-vol-'$$'","Driver":"local","DriverOpts":{"type":"none","device":"/","o":"bind"}}' \
    "http://$PROXY/volumes/create"                                                           || breached=1

if [[ $breached -eq 0 ]]; then
  echo "BLOCKED OK — INV-17 holds"
  exit 0
else
  echo "BREACH — INV-17 broken" >&2
  exit 1
fi
