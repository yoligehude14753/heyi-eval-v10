#!/usr/bin/env bash
# Attack drill 1 — INV-16 (filesystem isolation).
#
# Hypothesis: heyi-eval-agent attempts to delete the run index sqlite DB and
# the notify outbox. ACL must block it with EACCES.
#
# This drill MUST be invoked AS heyi-eval-agent:
#   sudo -u heyi-eval-agent bash deploy/agent-sandbox/drills/attack_delete_store.sh
#
# Exit codes:
#   0  attack BLOCKED (good — protection holds)
#   1  attack SUCCEEDED (bad — INV-16 breached)
#   2  drill misconfigured (wrong user, missing files, etc.)
set -u  # not -e: we WANT to see errors from rm

DATA_ROOT="${HEYI_EVAL_DATA_ROOT:-/home/ai/heyi-eval-data}"

cur=$(id -un)
if [[ "$cur" != "heyi-eval-agent" ]]; then
  echo "drill misconfigured: must run as heyi-eval-agent, got $cur" >&2
  exit 2
fi

target_db="$DATA_ROOT/store/runs.sqlite"
target_outbox="$DATA_ROOT/store/notify_outbox.jsonl"

if [[ ! -e "$target_db" ]] && [[ ! -e "$target_outbox" ]]; then
  echo "drill misconfigured: neither $target_db nor $target_outbox exists" >&2
  echo "(start the orchestrator at least once so store/ has files)" >&2
  exit 2
fi

breached=0

probe() {
  local action="$1" path="$2"
  if [[ ! -e "$path" ]]; then
    echo "  - $action $path: (skip, missing)"
    return
  fi
  local out rc
  out=$($action "$path" 2>&1)
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "BREACH  $action $path succeeded (rc=0)" >&2
    breached=1
  else
    echo "blocked $action $path (rc=$rc, msg='$out')"
  fi
}

echo "[drill-1] attacking store/ as $cur"
probe "rm -f"             "$target_db"
probe "rm -f"             "$target_outbox"
probe "truncate -s 0"     "$target_db"
probe "truncate -s 0"     "$target_outbox"
# also try to drop a hostile file into store/ (write-anywhere check)
hostile="$DATA_ROOT/store/pwn-$$"
if : >"$hostile" 2>/dev/null; then
  echo "BREACH  create $hostile succeeded" >&2
  breached=1
  rm -f "$hostile" 2>/dev/null || true
else
  echo "blocked create $hostile (EACCES)"
fi

if [[ $breached -eq 0 ]]; then
  echo "BLOCKED OK — INV-16 holds"
  exit 0
else
  echo "BREACH — INV-16 broken" >&2
  exit 1
fi
