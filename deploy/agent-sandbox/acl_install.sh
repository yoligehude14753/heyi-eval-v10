#!/usr/bin/env bash
# Install POSIX ACLs that constrain heyi-eval-agent's filesystem reach
# (INV-16).
#
# End state we want to verify:
#   READ-ONLY:
#     - /home/ai/heyi-eval-data/store/      (run index sqlite + outbox)
#     - /home/ai/heyi-eval-data/runs/       (parent dir; non-recursive)
#     - /home/ai/heyi-eval-v10/             (source tree)
#   PER-RUN WRITABLE (granted at run-start by orchestrator, not here):
#     - /home/ai/heyi-eval-data/runs/<run_id>/
#   PRIVATE WRITABLE:
#     - /var/lib/heyi-eval-agent/           (HOME)
#     - /tmp                                (system default, fine)
#
# Run as root on nv8: `sudo ./acl_install.sh`
set -euo pipefail

AGENT_USER="${AGENT_USER:-heyi-eval-agent}"
DATA_ROOT="${HEYI_EVAL_DATA_ROOT:-/home/ai/heyi-eval-data}"
SRC_ROOT="${HEYI_EVAL_SRC_ROOT:-/home/ai/heyi-eval-v10}"

log() { printf '[acl_install] %s\n' "$*"; }
die() { printf '[acl_install] ERROR: %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "must run as root"
id -u "$AGENT_USER" >/dev/null 2>&1 || die "user $AGENT_USER missing (run setup_agent_user.sh first)"
command -v setfacl >/dev/null || die "setfacl missing — apt install acl"
command -v getfacl >/dev/null || die "getfacl missing — apt install acl"

# Source tree must exist; data root we tolerate creating since some installs
# stage it lazily.
[[ -d "$SRC_ROOT" ]]  || die "src root missing: $SRC_ROOT"
install -d -o ai -g ai -m 0755 "$DATA_ROOT" "$DATA_ROOT/store" "$DATA_ROOT/runs"

# ── 0. traversal: /home/ai is typically 0750, so the agent (not in ai's
#       group) cannot even cd through it. Grant `x`-only (no read) — this
#       lets it reach $SRC_ROOT / $DATA_ROOT under ai's home without ever
#       being able to `ls /home/ai` and discover other private dirs.
log "granting --x on /home/ai (traversal only) to $AGENT_USER"
setfacl -m "u:$AGENT_USER:x" /home/ai

# ── 1. source tree: read+execute ────────────────────────────────────────
log "granting r-x on $SRC_ROOT (recursive) to $AGENT_USER"
setfacl -R -m "u:$AGENT_USER:rx" "$SRC_ROOT"
setfacl -R -d -m "u:$AGENT_USER:rx" "$SRC_ROOT"

# ── 2. store/: READ-ONLY, default ACL also read-only ───────────────────
log "granting r-x on $DATA_ROOT/store (recursive, default) to $AGENT_USER"
setfacl -R -m "u:$AGENT_USER:rx" "$DATA_ROOT/store"
setfacl -R -d -m "u:$AGENT_USER:rx" "$DATA_ROOT/store"

# Critical: store/ itself MUST NOT grant w to the agent — strip any stale
# entry that might have leaked from a previous install.
if getfacl -p "$DATA_ROOT/store" 2>/dev/null | grep -qE "^user:$AGENT_USER:.*w"; then
  die "post-check: $AGENT_USER still has write on $DATA_ROOT/store (refusing)"
fi

# ── 3. runs/ parent: read+execute (traverse), per-run write granted later
log "granting r-x on $DATA_ROOT/runs (non-recursive) to $AGENT_USER"
setfacl -m "u:$AGENT_USER:rx" "$DATA_ROOT/runs"
# Default ACL on runs/ so any NEW child <run_id>/ inherits r-x by default;
# write is opt-in via grant_run_write.sh (granted by orchestrator, see M4).
setfacl -d -m "u:$AGENT_USER:rx" "$DATA_ROOT/runs"

# ── 4. heyi-engine container model dir: READ-ONLY ───────────────────────
# Some installs colocate the prod model weights under /DATA/Model/. Forbid
# the agent from even reading them by default (no ACL grant). We do NOT add
# any rule here on purpose — absence of grant = no access.

# ── 5. verify post-state ───────────────────────────────────────────────
acl_dump() { getfacl -p "$1" 2>/dev/null | grep -E "^(user|default:user):$AGENT_USER:" || echo "(none)"; }
log "ACL on $DATA_ROOT/store:"
acl_dump "$DATA_ROOT/store" | sed 's/^/  /'
log "ACL on $DATA_ROOT/runs:"
acl_dump "$DATA_ROOT/runs" | sed 's/^/  /'

# negative test: ensure no `w` mask in store
acl_str=$(getfacl -p "$DATA_ROOT/store" 2>/dev/null \
  | grep -E "^user:$AGENT_USER:" | awk -F: '{print $3}' | head -1)
if [[ "$acl_str" == *w* ]]; then
  die "FATAL: store still writable to $AGENT_USER (acl=$acl_str)"
fi

log "OK"
