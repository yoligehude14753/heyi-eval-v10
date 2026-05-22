#!/usr/bin/env bash
# PR#22a final acceptance: run all 5 sandbox attack drills in order,
# stop at first BREACH. This is the single command release engineers
# / future agents will run after touching anything in deploy/agent-sandbox/.
#
# Run as root: `sudo bash drills/run_all.sh`
# The script itself sudo -u heyi-eval-agent for the agent-side drills
# (1, 2, 3, 4) and stays root for the systemd drill (5).
set -u

if [[ $EUID -ne 0 ]]; then
  echo "must run as root (use sudo)" >&2
  exit 2
fi

cd "$(dirname "$0")"
drills_dir="$(pwd)"
agent_user="${AGENT_USER:-heyi-eval-agent}"

# preflight: ensure user exists, proxy is up, units installed
ok=1
if ! id -u "$agent_user" >/dev/null 2>&1; then
  echo "preflight: user $agent_user missing (run setup_agent_user.sh)" >&2
  ok=0
fi
if ! curl -sS --max-time 3 http://127.0.0.1:2377/_ping >/dev/null 2>&1; then
  echo "preflight: agent socket proxy not reachable at 127.0.0.1:2377 (run compose up)" >&2
  ok=0
fi
if ! systemctl list-unit-files heyi-eval-agent@.service >/dev/null 2>&1; then
  echo "preflight: heyi-eval-agent@.service template not installed (cp + daemon-reload)" >&2
  ok=0
fi
[[ $ok -eq 1 ]] || { echo "drill run_all: preflight failed" >&2; exit 2; }

results=()
run_drill() {
  local id="$1" path="$2" runas="$3"
  echo "=================================================================="
  echo "drill $id  ($path  as=$runas)"
  echo "=================================================================="
  local rc
  if [[ "$runas" == "root" ]]; then
    bash "$path"; rc=$?
  else
    sudo -u "$runas" bash "$path"; rc=$?
  fi
  results+=("drill-$id rc=$rc")
  if [[ $rc -ne 0 ]]; then
    echo "BREACH at drill-$id (rc=$rc) — stopping" >&2
    return 1
  fi
  return 0
}

run_drill 1 "$drills_dir/attack_delete_store.sh"        "$agent_user" || exit 1
run_drill 2 "$drills_dir/attack_exec_prod.sh"           "$agent_user" || exit 1
run_drill 3 "$drills_dir/attack_sudo_escalate.sh"       "$agent_user" || exit 1
run_drill 4 "$drills_dir/attack_evade_audit.sh"         "$agent_user" || exit 1
run_drill 5 "$drills_dir/attack_resource_budget.sh"     "root"        || exit 1
# Drill 6 is conditional — only run when the audit wrapper is installed
# (PR#22b-M1). Older sandbox deployments without the wrapper should
# still pass 1..5 cleanly.
if [[ -x /usr/local/sbin/heyi-eval-agent-audit-record ]]; then
  run_drill 6 "$drills_dir/attack_evade_audit_writes.sh" "$agent_user" || exit 1
else
  echo "skip drill-6 (audit wrapper not installed — PR#22b-M1 not deployed)"
  results+=("drill-6 SKIPPED (wrapper missing)")
fi

echo "=================================================================="
total_drills=$(( ${#results[@]} ))
echo "ALL $total_drills DRILLS PASSED — INV-16/17/18/19/20/21 hold end-to-end"
printf '  %s\n' "${results[@]}"
echo "=================================================================="
exit 0
