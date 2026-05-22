#!/usr/bin/env bash
# scripts/start_eval_loop.sh — start the eval orchestrator loop without systemd.
#
# Use this when:
#   - systemd units are not yet installed / updated (run bootstrap_nv8.sh later)
#   - Quick restart is needed without root access
#
# Usage:
#   bash scripts/start_eval_loop.sh            # start with default M2.7 config
#   bash scripts/start_eval_loop.sh --dry-run  # print env and exit
#   bash scripts/start_eval_loop.sh --status   # show if loop is running
#   bash scripts/start_eval_loop.sh --stop     # graceful stop via pid file
#
# Logs: ~/eval-loop.log
# PID:  ~/eval-loop.pid
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${REPO_ROOT}/.venv"
LOG="${HOME}/eval-loop.log"
PID_FILE="${HOME}/eval-loop.pid"

# ── default env (M2.7 stable state) ────────────────────────────────────────
# Override by exporting vars before calling this script, or by editing
# /etc/heyi-eval-v10/env (needs sudo; bootstrap_nv8.sh installs it).
export HEYI_EVAL_DATA="${HEYI_EVAL_DATA:-/home/ai/heyi-eval-data}"
export HEYI_EVAL_PROD_ENGINE_CONTAINER="${HEYI_EVAL_PROD_ENGINE_CONTAINER:-minimax}"
export HEYI_EVAL_PROD_ENGINE_GPUS="${HEYI_EVAL_PROD_ENGINE_GPUS:-0,1,2,3}"
export HEYI_EVAL_EVAL_GPUS="${HEYI_EVAL_EVAL_GPUS:-4,5,6,7}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# ── helpers ─────────────────────────────────────────────────────────────────
log() { printf '[start_eval_loop] %s\n' "$*"; }

show_env() {
  log "HEYI_EVAL_DATA              = ${HEYI_EVAL_DATA}"
  log "HEYI_EVAL_PROD_ENGINE_CONTAINER = ${HEYI_EVAL_PROD_ENGINE_CONTAINER}"
  log "HEYI_EVAL_PROD_ENGINE_GPUS  = ${HEYI_EVAL_PROD_ENGINE_GPUS}"
  log "HEYI_EVAL_EVAL_GPUS         = ${HEYI_EVAL_EVAL_GPUS}"
  log "HF_ENDPOINT                 = ${HF_ENDPOINT}"
  log "log  -> ${LOG}"
  log "pid  -> ${PID_FILE}"
}

# ── arg parsing ─────────────────────────────────────────────────────────────
DRY_RUN=0
ACTION="start"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --status)  ACTION="status" ;;
    --stop)    ACTION="stop" ;;
    -h|--help)
      sed -n '1,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# ── status ───────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "status" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      log "running (pid=${pid})"
      tail -5 "$LOG" 2>/dev/null || true
    else
      log "stale pid file (pid=${pid} not found); loop is NOT running"
    fi
  else
    log "not running (no pid file)"
  fi
  exit 0
fi

# ── stop ─────────────────────────────────────────────────────────────────────
if [[ "$ACTION" == "stop" ]]; then
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    if kill -0 "$pid" 2>/dev/null; then
      log "sending SIGTERM to pid=${pid}"
      kill "$pid"
      sleep 2
      kill -0 "$pid" 2>/dev/null && log "still running; send SIGKILL" && kill -9 "$pid" || true
      rm -f "$PID_FILE"
      log "stopped"
    else
      log "pid=${pid} not running; removing stale pid file"
      rm -f "$PID_FILE"
    fi
  else
    log "not running"
  fi
  exit 0
fi

# ── start ────────────────────────────────────────────────────────────────────
show_env

if [[ ! -x "${VENV}/bin/python" ]]; then
  log "ERROR: venv not found at ${VENV}; run bootstrap_nv8.sh first" >&2
  exit 1
fi

# Refuse to start if another loop is already running.
if [[ -f "$PID_FILE" ]]; then
  pid="$(cat "$PID_FILE")"
  if kill -0 "$pid" 2>/dev/null; then
    log "already running (pid=${pid}); use --stop first"
    exit 1
  else
    log "stale pid file removed"
    rm -f "$PID_FILE"
  fi
fi

if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY] would start: ${VENV}/bin/python -m orchestrator loop"
  exit 0
fi

mkdir -p "${HEYI_EVAL_DATA}/runs" "${HEYI_EVAL_DATA}/queue"

log "starting loop → ${LOG}"
nohup "${VENV}/bin/python" -m orchestrator loop \
  >> "${LOG}" 2>&1 &
LOOP_PID=$!
echo "$LOOP_PID" > "$PID_FILE"
log "started (pid=${LOOP_PID})"
log "tail log: tail -f ${LOG}"
