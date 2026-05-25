#!/usr/bin/env bash
# Create the dedicated heyi-eval-agent UNIX user used by the sandboxed Claude
# Code agent (PR#22a). Idempotent: re-running yields the same end state.
#
# What this script enforces (these are the negative invariants that protect
# us from v9-era incidents):
#   - user EXISTS as a system account, login shell /usr/sbin/nologin
#     (the agent is invoked via `sudo -u`, not interactive ssh)
#   - user is NOT in `docker`     (else INV-3 / INV-17 are bypassable)
#   - user is NOT in `sudo`/`wheel` (else INV-19 is meaningless)
#   - $HOME = /var/lib/heyi-eval-agent, 0750, owned by user
#
# Run as root on nv8: `sudo ./setup_agent_user.sh`
set -euo pipefail

AGENT_USER="${AGENT_USER:-heyi-eval-agent}"
AGENT_HOME="${AGENT_HOME:-/var/lib/heyi-eval-agent}"
AGENT_SHELL="${AGENT_SHELL:-/usr/sbin/nologin}"

log() { printf '[setup_agent_user] %s\n' "$*"; }
die() { printf '[setup_agent_user] ERROR: %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "must run as root (use sudo)"

# 1) user exists ──────────────────────────────────────────────────────────
if id -u "$AGENT_USER" >/dev/null 2>&1; then
  log "user $AGENT_USER already exists, will normalise"
else
  log "creating system user $AGENT_USER"
  useradd \
    --system \
    --home-dir "$AGENT_HOME" \
    --shell "$AGENT_SHELL" \
    --user-group \
    "$AGENT_USER"
fi

# 2) home dir owned + 0750 ───────────────────────────────────────────────
install -d -o "$AGENT_USER" -g "$AGENT_USER" -m 0750 "$AGENT_HOME"

# 3) forbidden groups ────────────────────────────────────────────────────
FORBIDDEN=(docker sudo wheel adm)
for g in "${FORBIDDEN[@]}"; do
  if id -nG "$AGENT_USER" | tr ' ' '\n' | grep -qx "$g"; then
    log "removing $AGENT_USER from group $g"
    gpasswd -d "$AGENT_USER" "$g" >/dev/null
  fi
done

# 4) lock direct interactive login (no password ever) ────────────────────
passwd -l "$AGENT_USER" >/dev/null
usermod -s "$AGENT_SHELL" "$AGENT_USER"

# 5) verify post-conditions (fail loud if any drift) ─────────────────────
groups_now=$(id -nG "$AGENT_USER")
for g in "${FORBIDDEN[@]}"; do
  if echo " $groups_now " | grep -q " $g "; then
    die "post-check failed: $AGENT_USER is still in group $g"
  fi
done

login_shell=$(getent passwd "$AGENT_USER" | awk -F: '{print $7}')
[[ "$login_shell" == "$AGENT_SHELL" ]] \
  || die "post-check failed: login shell is $login_shell, expected $AGENT_SHELL"

log "OK — $AGENT_USER ready: home=$AGENT_HOME shell=$AGENT_SHELL groups=($groups_now)"
