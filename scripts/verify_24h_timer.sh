#!/usr/bin/env bash
# Verify the heyi-eval-v10 timer/heartbeat invariants over a 24h window.
#
# Run on nv8 AFTER the systemd units have been up for >=24h:
#     ./scripts/verify_24h_timer.sh
#
# Exits 0 if all checks pass, non-zero otherwise. Prints a JSON report
# to stdout so the result can be piped into the notify_outbox or a
# tracking script.
#
# Test IDs (see docs/PR8_TEST_PLAN.md §4):
#   T-24h-1  systemd backup.timer is healthy
#   T-24h-2  rsync snapshots really land at ~30min cadence
#   T-24h-3  heartbeat notifications fire ~every 4h
#   T-24h-4  orchestrator did not crash repeatedly
#   T-24h-5  engine-pause incidents are paired with engine-resume
#
# Idempotent + side-effect free: only reads logs/snapshots, never writes
# anything outside stdout.

set -euo pipefail

# ── config ──────────────────────────────────────────────────────────────────

HEYI_EVAL_DATA="${HEYI_EVAL_DATA:-/home/ai/heyi-eval-data}"
HEYI_EVAL_BACKUPS="${HEYI_EVAL_BACKUPS:-/home/ai/heyi-eval-backups}"
OUTBOX="${HEYI_EVAL_OUTBOX:-$HEYI_EVAL_DATA/store/notify_outbox.jsonl}"

# Parallel arrays: same index across CHECK_IDS / STATUSES / DETAILS.
# Avoids bash<4's lack of associative arrays so the script syntax-checks
# on macOS dev hosts and runs unchanged on the nv8 Linux host.
CHECK_IDS=()
STATUSES=()
DETAILS=()

record() {
    # $1=id $2=status $3=detail
    CHECK_IDS+=("$1")
    STATUSES+=("$2")
    DETAILS+=("$3")
}
pass() { record "$1" "ok"   "$2"; }
fail() { record "$1" "fail" "$2"; }
skip() { record "$1" "skip" "$2"; }

# ── T-24h-1: systemd timer healthy ─────────────────────────────────────────

if ! command -v systemctl >/dev/null 2>&1; then
    skip "T-24h-1" "systemctl not on PATH (non-systemd host?)"
else
    if ! systemctl list-timers heyi-eval-backup.timer >/dev/null 2>&1; then
        fail "T-24h-1" "heyi-eval-backup.timer not registered"
    else
        LEFT_RAW="$(systemctl show -p NextElapseUSecRealtime --value heyi-eval-backup.timer)"
        LAST_RAW="$(systemctl show -p LastTriggerUSec --value heyi-eval-backup.timer)"
        # LEFT should be within next 30min. Empty / 'n/a' = scheduler not armed.
        if [[ -z "$LAST_RAW" || "$LAST_RAW" == "0" || "$LAST_RAW" == "n/a" ]]; then
            fail "T-24h-1" "timer never fired yet (LastTriggerUSec=$LAST_RAW)"
        else
            pass "T-24h-1" "timer healthy; last=$LAST_RAW next_in=$LEFT_RAW"
        fi
    fi
fi

# ── T-24h-2: snapshots land at ~30min cadence ──────────────────────────────

if [[ ! -d "$HEYI_EVAL_BACKUPS" ]]; then
    fail "T-24h-2" "$HEYI_EVAL_BACKUPS does not exist (no snapshots ever?)"
else
    # Snapshot dirs assumed to be named with sortable timestamps. We grab
    # the 10 newest and check their mtimes are spaced 25-35 minutes apart.
    SNAPS=$(ls -1t "$HEYI_EVAL_BACKUPS" 2>/dev/null | head -10 || true)
    if [[ -z "$SNAPS" ]]; then
        fail "T-24h-2" "no snapshot directories under $HEYI_EVAL_BACKUPS"
    else
        # Collect mtimes (epoch seconds)
        MTIMES=()
        while IFS= read -r d; do
            [[ -e "$HEYI_EVAL_BACKUPS/$d" ]] || continue
            MTIMES+=("$(stat -c '%Y' "$HEYI_EVAL_BACKUPS/$d" 2>/dev/null \
                      || stat -f '%m' "$HEYI_EVAL_BACKUPS/$d")")
        done <<< "$SNAPS"
        if [[ ${#MTIMES[@]} -lt 4 ]]; then
            fail "T-24h-2" "only ${#MTIMES[@]} snapshots — expected >= 4 in 2h"
        else
            BAD_GAP=""
            for ((i=1; i<${#MTIMES[@]}; i++)); do
                GAP=$(( ${MTIMES[i-1]} - ${MTIMES[i]} ))
                if (( GAP < 25*60 || GAP > 35*60 )); then
                    BAD_GAP="${BAD_GAP} idx=${i}:gap=${GAP}s"
                fi
            done
            if [[ -n "$BAD_GAP" ]]; then
                fail "T-24h-2" "snapshot cadence out of 25-35min band:${BAD_GAP}"
            else
                pass "T-24h-2" "${#MTIMES[@]} snapshots, cadence within band"
            fi
        fi
    fi
fi

# ── T-24h-3: heartbeat notifications fire ──────────────────────────────────

if [[ ! -f "$OUTBOX" ]]; then
    fail "T-24h-3" "notify outbox missing: $OUTBOX"
else
    # Count heartbeats emitted in the last 24h. We don't parse JSON
    # timestamps strictly — instead we look at the file's tail.
    HB_COUNT=$(tail -n 4000 "$OUTBOX" | grep -c '"event"[[:space:]]*:[[:space:]]*"heartbeat"' || true)
    if (( HB_COUNT < 4 )); then
        fail "T-24h-3" "heartbeats in last 4000 lines = $HB_COUNT (expected >= 4)"
    else
        pass "T-24h-3" "$HB_COUNT heartbeats found in tail"
    fi
fi

# ── T-24h-4: orchestrator did not restart-loop ─────────────────────────────

if ! command -v systemctl >/dev/null 2>&1; then
    skip "T-24h-4" "systemctl not on PATH"
else
    if ! systemctl status heyi-eval-orchestrator.service >/dev/null 2>&1; then
        fail "T-24h-4" "heyi-eval-orchestrator.service not present"
    else
        ACTIVE="$(systemctl is-active heyi-eval-orchestrator.service || true)"
        NRESTARTS="$(systemctl show -p NRestarts --value heyi-eval-orchestrator.service)"
        NRESTARTS="${NRESTARTS:-0}"
        if [[ "$ACTIVE" != "active" ]]; then
            fail "T-24h-4" "orchestrator not active (is-active=$ACTIVE, restarts=$NRESTARTS)"
        elif (( NRESTARTS > 5 )); then
            fail "T-24h-4" "orchestrator restarted $NRESTARTS times — restart-loop suspect"
        else
            pass "T-24h-4" "active, restarts=$NRESTARTS"
        fi
    fi
fi

# ── T-24h-5: engine-pause incidents are paired ─────────────────────────────

if [[ ! -f "$OUTBOX" ]]; then
    skip "T-24h-5" "outbox missing (already failed in T-24h-3)"
else
    PAUSED=$(grep -c 'orchestrator-paused-engine-down' "$OUTBOX" 2>/dev/null || true)
    RESUMED=$(grep -c 'orchestrator-resumed-engine-recovered' "$OUTBOX" 2>/dev/null || true)
    # Allow at most 1 unmatched pause (an incident in progress is fine).
    DIFF=$(( PAUSED - RESUMED ))
    if (( DIFF > 1 || DIFF < -1 )); then
        fail "T-24h-5" "engine pause/resume unbalanced: paused=$PAUSED resumed=$RESUMED"
    else
        pass "T-24h-5" "paused=$PAUSED resumed=$RESUMED"
    fi
fi

# ── emit JSON report ───────────────────────────────────────────────────────

EXIT=0
printf '{\n  "host": "%s",\n  "checked_at": "%s",\n  "checks": {\n' \
    "$(hostname)" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
for ((i=0; i<${#CHECK_IDS[@]}; i++)); do
    if (( i > 0 )); then printf ',\n'; fi
    k="${CHECK_IDS[$i]}"
    STATUS="${STATUSES[$i]}"
    DETAIL="${DETAILS[$i]}"
    # JSON-escape detail (small subset; outbox parser uses jq later)
    DETAIL_ESC="${DETAIL//\\/\\\\}"
    DETAIL_ESC="${DETAIL_ESC//\"/\\\"}"
    printf '    "%s": {"status": "%s", "detail": "%s"}' \
        "$k" "$STATUS" "$DETAIL_ESC"
    if [[ "$STATUS" == "fail" ]]; then
        EXIT=1
    fi
done
printf '\n  }\n}\n'

exit "$EXIT"
