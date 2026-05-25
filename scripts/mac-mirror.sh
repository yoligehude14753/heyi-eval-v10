#!/usr/bin/env bash
# scripts/mac-mirror.sh — nightly Mac-side pull of nv8 backups.
#
# Pull the latest snapshot tree from nv8 to a local Mac directory so we
# have an offsite (well, off-host) copy. Designed to be invoked by the
# launchd plist deploy/launchd/com.heyi.eval.mac-mirror.plist, but works
# fine as a cron one-liner or a manual `bash scripts/mac-mirror.sh`.
#
# What it does:
#   1. rsync over ssh from nv8:<HEYI_EVAL_BACKUPS>/latest/ → <MAC_MIRROR_ROOT>/<utc-date>/
#   2. update <MAC_MIRROR_ROOT>/latest symlink atomically
#   3. delete mirrors older than $MAC_MIRROR_KEEP_DAYS (default 30)
#   4. append a JSON line to <MAC_MIRROR_ROOT>/mirror.log
#
# What it does NOT do:
#   - touch nv8's data_root
#   - write to nv8 (read-only ssh pull)
#   - require Python (pure bash + ssh + rsync)
#
# Environment overrides:
#   NV8_SSH_TARGET    — REQUIRED, e.g. "ai@my-host" or "ai@10.0.0.5"
#   NV8_BACKUPS_PATH  — remote backups dir (default ~/heyi-eval-backups)
#   MAC_MIRROR_ROOT   — local mirror dir   (default ~/heyi-eval-mirror)
#   MAC_MIRROR_KEEP_DAYS — int, default 30
#
# Exit codes:
#   0 — success
#   1 — rsync failed
#   2 — pre-flight check failed (ssh unreachable, latest/ missing)

set -euo pipefail

NV8_SSH_TARGET="${NV8_SSH_TARGET:?NV8_SSH_TARGET is required, e.g. ai@my-nv8-host or ai@10.0.0.5}"
NV8_BACKUPS_PATH="${NV8_BACKUPS_PATH:-/home/ai/heyi-eval-backups}"
MAC_MIRROR_ROOT="${MAC_MIRROR_ROOT:-$HOME/heyi-eval-mirror}"
MAC_MIRROR_KEEP_DAYS="${MAC_MIRROR_KEEP_DAYS:-30}"

mkdir -p "$MAC_MIRROR_ROOT"
TS="$(date -u +%Y%m%d_%H%M%S)"
DST="$MAC_MIRROR_ROOT/$TS"
LOG="$MAC_MIRROR_ROOT/mirror.log"

log_event() {
    local status="$1"
    local detail="$2"
    printf '{"ts":"%s","status":"%s","detail":"%s","host":"%s","ts_local":"%s"}\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$status" "$detail" "$NV8_SSH_TARGET" "$TS" \
        >> "$LOG"
}

# Pre-flight: can we even reach the host?
if ! ssh -o ConnectTimeout=10 -o BatchMode=yes "$NV8_SSH_TARGET" "test -d $NV8_BACKUPS_PATH/latest"; then
    log_event "failed" "preflight: latest/ not reachable on $NV8_SSH_TARGET"
    echo "ERROR: $NV8_SSH_TARGET:$NV8_BACKUPS_PATH/latest is not reachable" >&2
    exit 2
fi

# Pull with hard-link optimization against the previous mirror if present.
LINK_DEST=""
if [[ -L "$MAC_MIRROR_ROOT/latest" ]]; then
    PREV="$(readlink "$MAC_MIRROR_ROOT/latest")"
    if [[ -d "$MAC_MIRROR_ROOT/$PREV" ]]; then
        LINK_DEST="--link-dest=$MAC_MIRROR_ROOT/$PREV"
    fi
fi

mkdir -p "$DST"
if ! rsync -az --delete --stats \
        ${LINK_DEST:+$LINK_DEST} \
        "$NV8_SSH_TARGET:$NV8_BACKUPS_PATH/latest/" \
        "$DST/"; then
    rm -rf "$DST"
    log_event "failed" "rsync nonzero"
    exit 1
fi

# Atomic latest symlink swap.
ln -sfn "$TS" "$MAC_MIRROR_ROOT/latest.tmp"
mv -fT "$MAC_MIRROR_ROOT/latest.tmp" "$MAC_MIRROR_ROOT/latest" 2>/dev/null \
    || mv -f "$MAC_MIRROR_ROOT/latest.tmp" "$MAC_MIRROR_ROOT/latest"

# Retention: drop directories older than MAC_MIRROR_KEEP_DAYS, but always
# leave the newest one (mirrors the nv8 side INV-9 safety net).
find "$MAC_MIRROR_ROOT" -mindepth 1 -maxdepth 1 -type d \
    -name '[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9]' \
    -mtime "+$MAC_MIRROR_KEEP_DAYS" \
    -not -newer "$DST" \
    -exec rm -rf {} +

log_event "ok" "size=$(du -sh "$DST" 2>/dev/null | cut -f1)"
echo "mirror ok: $DST"
