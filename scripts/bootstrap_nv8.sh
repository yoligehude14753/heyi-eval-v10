#!/usr/bin/env bash
# scripts/bootstrap_nv8.sh — idempotent nv8 deployment.
#
# Designed to be safe to re-run: each step short-circuits when the desired
# state is already in place. Honors DRY_RUN=1 for a no-op preview.
#
# Usage:
#   sudo -u ai DRY_RUN=1 bash scripts/bootstrap_nv8.sh            # preview
#   sudo -u ai bash scripts/bootstrap_nv8.sh                       # real run
#   sudo -u ai bash scripts/bootstrap_nv8.sh --force               # skip hostname guard
set -euo pipefail

# ── config ──────────────────────────────────────────────────────────────────
REPO_ROOT="/home/ai/heyi-eval-v10"
DATA_ROOT="/home/ai/heyi-eval-data"
BACKUPS_ROOT="/home/ai/heyi-eval-backups"
ENV_DIR="/etc/heyi-eval-v10"
ENV_FILE="${ENV_DIR}/env"
VENV="${REPO_ROOT}/.venv"
SYSTEMD_DIR="/etc/systemd/system"

UNITS_SERVICES=(
  heyi-eval-orchestrator.service
  heyi-eval-discover.service
  heyi-eval-panel.service
  heyi-eval-notify-sync.service
  heyi-eval-backup.service
)
UNITS_TIMERS=(heyi-eval-backup.timer)
UNITS_TO_ENABLE=(
  heyi-eval-orchestrator.service
  heyi-eval-discover.service
  heyi-eval-panel.service
  heyi-eval-backup.timer
)

DRY_RUN="${DRY_RUN:-0}"
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    -h|--help)
      sed -n '1,/^set -euo/p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# ── helpers ─────────────────────────────────────────────────────────────────
step() { printf '\n== %s ==\n' "$*"; }
log()  { printf '   %s\n' "$*"; }
run()  {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '   [DRY] %s\n' "$*"
  else
    eval "$@"
  fi
}

# ── 1. preflight ────────────────────────────────────────────────────────────
step "preflight"
hn="$(hostname)"
log "hostname=${hn}"
if [[ "$FORCE" != "1" ]] && [[ "$hn" != *nv8* ]] && [[ "$hn" != *heyi-sh-nv8* ]]; then
  echo "refuse: hostname '${hn}' is not nv8. Re-run with --force to override." >&2
  exit 1
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "missing command: $1" >&2; exit 1;
  fi
  log "$1: $(command -v "$1")"
}
require_cmd docker
require_cmd nvidia-smi
require_cmd systemctl

# Pick the highest available python3 binary whose version is >= 3.11.
# Tried in descending preference; first to satisfy the version gate wins.
# Populates the global PYTHON_BIN consumed by the venv step.
find_python_311_plus() {
  local cand ver
  for cand in python3.14 python3.13 python3.12 python3.11 python3; do
    if ! command -v "$cand" >/dev/null 2>&1; then continue; fi
    ver="$("$cand" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    case "$ver" in
      3.11|3.12|3.13|3.14|3.15)
        PYTHON_BIN="$cand"
        log "python: $cand (version $ver) at $(command -v "$cand")"
        return 0
        ;;
    esac
  done
  echo "no python3.11+ found; tried python3.14/13/12/11/3. Install python3.11 or newer." >&2
  exit 1
}
PYTHON_BIN=""
find_python_311_plus

# user must be 'ai' (services run as ai)
who="$(id -un)"
log "running as user=${who}"
if [[ "$who" != "ai" ]] && [[ "$FORCE" != "1" ]]; then
  echo "refuse: bootstrap must run as user 'ai'. Use 'sudo -u ai' or --force." >&2
  exit 1
fi

# ── 2. directories ──────────────────────────────────────────────────────────
step "directories"
ensure_dir() {
  local d="$1"
  if [[ -d "$d" ]]; then
    log "${d} (exists)"
  else
    log "${d} (creating)"
    run "mkdir -p '${d}'"
    run "chmod 700 '${d}'"
  fi
}
# Safety: only ensure dirs that live under the heyi-eval-v10 deploy prefix.
for d in "$DATA_ROOT" "$BACKUPS_ROOT"; do
  case "$d" in
    /home/ai/heyi-eval-*) ensure_dir "$d" ;;
    *) echo "refuse: refusing to create non-heyi path '$d'" >&2; exit 1 ;;
  esac
done

# /etc/heyi-eval-v10/env needs sudo, so only attempt if writable.
if [[ -w "$(dirname "$ENV_DIR")" ]] || [[ "$DRY_RUN" == "1" ]]; then
  if [[ ! -d "$ENV_DIR" ]]; then
    run "sudo mkdir -p '${ENV_DIR}'"
  fi
  if [[ ! -f "$ENV_FILE" ]]; then
    log "installing default ${ENV_FILE}"
    run "sudo cp '${REPO_ROOT}/deploy/env.example' '${ENV_FILE}'"
    run "sudo chmod 0640 '${ENV_FILE}'"
  else
    log "${ENV_FILE} (exists, not overwriting)"
  fi
else
  log "skip ${ENV_DIR} setup (need sudo)"
fi

# ── 3. python venv ──────────────────────────────────────────────────────────
step "python venv"
if [[ ! -x "${VENV}/bin/python" ]]; then
  log "creating venv at ${VENV} (using ${PYTHON_BIN})"
  run "${PYTHON_BIN} -m venv '${VENV}'"
fi
run "'${VENV}/bin/pip' install --upgrade pip"
run "'${VENV}/bin/pip' install -e '${REPO_ROOT}'"

# ── 4. systemd units ────────────────────────────────────────────────────────
step "systemd units"
for u in "${UNITS_SERVICES[@]}" "${UNITS_TIMERS[@]}"; do
  src="${REPO_ROOT}/deploy/systemd/${u}"
  dst="${SYSTEMD_DIR}/${u}"
  if [[ ! -f "$src" ]]; then
    echo "missing unit source: $src" >&2; exit 1
  fi
  if [[ -f "$dst" ]] && cmp -s "$src" "$dst" 2>/dev/null; then
    log "${u} (unchanged)"
  else
    log "${u} (installing)"
    run "sudo install -m 0644 '${src}' '${dst}'"
  fi
done
# PR#22a agent sandbox slice + template unit. Installed but NOT enabled
# — `heyi-eval-agent.slice` activates lazily, `heyi-eval-agent@%i` is
# template-only (started by the orchestrator per run). See
# docs/RUNBOOK_NV8.md §13.
for u in heyi-eval-agent.slice "heyi-eval-agent@.service"; do
  src="${REPO_ROOT}/deploy/systemd/${u}"
  dst="${SYSTEMD_DIR}/${u}"
  if [[ ! -f "$src" ]]; then
    log "skip ${u} (source missing — sandbox PR not yet applied)"
    continue
  fi
  if [[ -f "$dst" ]] && cmp -s "$src" "$dst" 2>/dev/null; then
    log "${u} (unchanged)"
  else
    log "${u} (installing — sandbox)"
    run "sudo install -m 0644 '${src}' '${dst}'"
  fi
done
run "sudo systemctl daemon-reload"

for u in "${UNITS_TO_ENABLE[@]}"; do
  log "enable+start ${u}"
  run "sudo systemctl enable --now '${u}'"
done

# ── 4b. agent sandbox (PR#22a) ──────────────────────────────────────────────
step "agent sandbox (PR#22a)"
SANDBOX_DIR="${REPO_ROOT}/deploy/agent-sandbox"
if [[ -d "$SANDBOX_DIR" ]]; then
  # setup_agent_user.sh + acl_install.sh are idempotent; sudoers install
  # is a no-op when the file is byte-identical.
  run "sudo bash '${SANDBOX_DIR}/setup_agent_user.sh'"
  run "sudo bash '${SANDBOX_DIR}/acl_install.sh'"
  # Audit daemon + CLI client (PR#22b-M2). The setuid wrapper from M1
  # was removed because NoNewPrivileges=true in the agent unit blocks
  # `sudo`'s setuid — the daemon takes its place via a unix socket.
  if [[ -f "${SANDBOX_DIR}/heyi-eval-agent-audit-client.py" ]]; then
    run "sudo install -m 0755 -o root -g root '${SANDBOX_DIR}/heyi-eval-agent-audit-client.py' /usr/local/bin/heyi-eval-agent-audit-client"
  fi
  if [[ -f "${SANDBOX_DIR}/heyi-eval-agent-prepare" ]]; then
    run "sudo install -m 0755 -o root -g root '${SANDBOX_DIR}/heyi-eval-agent-prepare' /usr/local/sbin/heyi-eval-agent-prepare"
  fi
  if [[ -f "${SANDBOX_DIR}/heyi-eval-agent-run" ]]; then
    run "sudo install -m 0755 -o root -g root '${SANDBOX_DIR}/heyi-eval-agent-run' /usr/local/bin/heyi-eval-agent-run"
  fi
  # Install + enable the audit daemon. It must be listening BEFORE any
  # agent unit is started; ordering is also encoded as `Before=` in the
  # agent template unit.
  if [[ -f "${REPO_ROOT}/deploy/systemd/heyi-eval-audit.service" ]]; then
    if ! cmp -s "${REPO_ROOT}/deploy/systemd/heyi-eval-audit.service" \
                /etc/systemd/system/heyi-eval-audit.service 2>/dev/null; then
      log "installing heyi-eval-audit.service"
      run "sudo install -m 0644 '${REPO_ROOT}/deploy/systemd/heyi-eval-audit.service' /etc/systemd/system/heyi-eval-audit.service"
      run "sudo systemctl daemon-reload"
    fi
    run "sudo systemctl enable --now heyi-eval-audit.service"
    # Quick health probe: socket must exist and respond.
    if ! sudo test -S /run/heyi-eval-agent-audit.sock; then
      die "heyi-eval-audit.service did not produce /run/heyi-eval-agent-audit.sock"
    fi
  fi
  if ! cmp -s "${SANDBOX_DIR}/sudoers.d/heyi-eval-agent" \
              /etc/sudoers.d/heyi-eval-agent 2>/dev/null; then
    log "installing sudoers.d/heyi-eval-agent"
    run "sudo install -m 0440 '${SANDBOX_DIR}/sudoers.d/heyi-eval-agent' /etc/sudoers.d/heyi-eval-agent"
    run "sudo visudo -c -f /etc/sudoers.d/heyi-eval-agent"
  else
    log "sudoers.d/heyi-eval-agent (unchanged)"
  fi
  # PR#22b-M3: harvest helper + orchestrator sudoers. The helper is a
  # tiny root script the `ai` user can invoke via NOPASSWD sudo to copy
  # the agent's per-run outbox out of the 0750 HOME and chown it to ai:ai.
  if [[ -f "${SANDBOX_DIR}/heyi-eval-agent-harvest" ]]; then
    run "sudo install -m 0755 -o root -g root '${SANDBOX_DIR}/heyi-eval-agent-harvest' /usr/local/sbin/heyi-eval-agent-harvest"
  fi
  ORCH_SUDOERS_SRC="${REPO_ROOT}/deploy/sudoers.d/heyi-eval-orchestrator"
  if [[ -f "$ORCH_SUDOERS_SRC" ]]; then
    if ! cmp -s "$ORCH_SUDOERS_SRC" /etc/sudoers.d/heyi-eval-orchestrator 2>/dev/null; then
      log "installing sudoers.d/heyi-eval-orchestrator"
      run "sudo install -m 0440 '${ORCH_SUDOERS_SRC}' /etc/sudoers.d/heyi-eval-orchestrator"
      run "sudo visudo -c -f /etc/sudoers.d/heyi-eval-orchestrator"
    else
      log "sudoers.d/heyi-eval-orchestrator (unchanged)"
    fi
  fi
  # Agent-side docker-socket-proxy. We use the orchestrator's docker
  # daemon (the user is in `docker`), but the agent will reach it via
  # 127.0.0.1:2377 read-only — see INV-17.
  run "cd '${SANDBOX_DIR}' && sudo docker compose -f compose.agent-socket-proxy.yml up -d"
  if [[ "$DRY_RUN" != "1" ]]; then
    sleep 4
    if ! curl -sf --max-time 3 http://127.0.0.1:2377/_ping >/dev/null; then
      echo "bootstrap exit 1: heyi-eval-agent-socket-proxy did not become healthy" >&2
      exit 1
    fi
    log "agent-socket-proxy: 127.0.0.1:2377 healthy"
  fi
else
  log "skip agent sandbox (deploy/agent-sandbox missing — not yet on this branch)"
fi

# ── 5. health check ─────────────────────────────────────────────────────────
step "health check"
if [[ "$DRY_RUN" == "1" ]]; then
  log "[DRY] skipping live health probe"
else
  sleep 10
  ok=1
  for u in heyi-eval-orchestrator.service heyi-eval-panel.service; do
    if systemctl is-active --quiet "$u"; then
      log "${u}: active"
    else
      log "${u}: NOT ACTIVE — see 'journalctl -u ${u} -n 80'"
      ok=0
    fi
  done
  if curl -sf "http://127.0.0.1:8090/api/health" >/dev/null; then
    log "panel /api/health: 200"
  else
    log "panel /api/health: FAILED"
    ok=0
  fi
  if [[ "$ok" != "1" ]]; then
    echo "bootstrap exit 1: one or more services unhealthy" >&2
    exit 1
  fi
fi

# ── 6. summary ──────────────────────────────────────────────────────────────
step "summary"
log "repo:    ${REPO_ROOT}"
log "data:    ${DATA_ROOT}"
log "backups: ${BACKUPS_ROOT}"
log "env:     ${ENV_FILE}"
log "python:  ${PYTHON_BIN}"
log "panel:   http://$(hostname -I | awk '{print $1}'):8090"
log "units:   $(IFS=,; echo "${UNITS_TO_ENABLE[*]}")"
log "sandbox: $(systemctl list-unit-files heyi-eval-agent@.service 2>/dev/null | tail -1 || echo 'not installed')"
log "next:    journalctl -u heyi-eval-orchestrator.service -f"
log "verify:  sudo bash deploy/agent-sandbox/drills/run_all.sh   # all 5 sandbox drills"
echo "bootstrap_nv8 done."
