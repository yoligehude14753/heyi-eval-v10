#!/usr/bin/env bash
# Attack drill 5 — INV-19 (cgroup + RuntimeMaxSec budget).
#
# Two attacks, both expecting systemd's cgroup to step in:
#   (a) fork-bomb (~32 forks) inside the agent slice → TasksMax cap
#       must cause fork() to return EAGAIN; the host MUST remain
#       responsive.
#   (b) wall-clock budget — RuntimeMaxSec=6 then `sleep 300` → systemd
#       MUST terminate within ~10s, not allow the run to wedge for 5
#       minutes.
#
# Unlike drill 1/2/3/4, this drill MUST run as root (via systemd-run
# we need to start units owned by heyi-eval-agent). Invocation:
#   sudo bash deploy/agent-sandbox/drills/attack_resource_budget.sh
#
# Exit:
#   0  BLOCKED OK  — both budgets enforced
#   1  BREACH      — either limit failed to apply
#   2  drill misconfigured
set -u

if [[ $EUID -ne 0 ]]; then
  echo "drill misconfigured: must run as root (use sudo)" >&2
  exit 2
fi
command -v systemd-run >/dev/null || { echo "systemd-run missing" >&2; exit 2; }

breached=0

# ──── (a) fork limit ───────────────────────────────────────────────
# Start a transient unit with TasksMax=8 then try to fork beyond it.
# We do NOT `wait` for the children — that's the hang trap that bit
# us in v1: when fork() returns EAGAIN inside a loop and the child
# vanishes asynchronously, bash's wait can stall. Instead we just
# spawn-and-detach and let systemd reap the whole unit at
# RuntimeMaxSec=4. The KERNEL trace is the real signal: cgroup_pids
# logs "fork rejected" to dmesg/journal when it enforces the cap.
echo "[drill-5a] fork limit (TasksMax=8) — expecting kernel pids.max events"
# clear pid-tracker; mark start so we only count events from this run
start_marker=$(date +%s)
systemd-run \
  --quiet \
  --collect \
  --uid=heyi-eval-agent \
  --slice=heyi-eval-agent.slice \
  -p TasksMax=8 \
  -p MemoryMax=128M \
  -p RuntimeMaxSec=4 \
  -p TimeoutStopSec=2 \
  --wait \
  --service-type=oneshot \
  /bin/bash -c '
    # Run inside the cgroup, attempt aggressive fork. No wait, no
    # backgrounded waiting — just trigger the cap.
    for i in $(seq 1 32); do
      ( exec /bin/sleep 30 ) &
    done
    # Block here until the cgroup kills us at RuntimeMaxSec.
    /bin/sleep 30
  ' >/dev/null 2>&1 || true   # nonzero exit is expected (TERMed)

# Read the kernel journal AFTER the unit exited.
journal_5a=$(journalctl --since "@$start_marker" -k --no-pager 2>/dev/null \
  | grep -ciE "fork rejected by pids controller" || true)
echo "  kernel cgroup-pids events: $journal_5a"
if [[ "$journal_5a" -ge 1 ]]; then
  echo "blocked TasksMax=8 enforced — kernel logged 'fork rejected by pids controller'"
else
  echo "BREACH  TasksMax cap did not fire — no kernel pids.max event in this run" >&2
  breached=1
fi

# ──── (b) wall-clock budget ────────────────────────────────────────
# IMPORTANT: do NOT use `systemd-run --wait` over an ssh+sudo pipe —
# it inherits ttys/fds in ways that keep the parent ssh channel open
# even after the unit exits, causing the drill to appear to hang
# from the client side. Instead spawn detached (no --wait) and poll.
echo "[drill-5b] RuntimeMaxSec=6 — expecting kill within ~10s on sleep 300"
start_ts=$(date +%s)
unit_name="heyi-eval-drill-5b-$$"
systemd-run \
  --quiet \
  --collect \
  --unit="$unit_name" \
  --uid=heyi-eval-agent \
  --slice=heyi-eval-agent.slice \
  -p RuntimeMaxSec=6 \
  -p TimeoutStopSec=2 \
  /bin/sleep 300 >/dev/null 2>&1 || true
# Poll the unit; we tolerate 20s to cover the unit's TimeoutStopSec.
for _ in $(seq 1 25); do
  state=$(systemctl is-active "$unit_name.service" 2>/dev/null || true)
  if [[ "$state" != "active" && "$state" != "activating" ]]; then
    break
  fi
  sleep 1
done
systemctl reset-failed "$unit_name.service" 2>/dev/null || true
end_ts=$(date +%s)
elapsed=$(( end_ts - start_ts ))
echo "  elapsed=${elapsed}s"
if [[ $elapsed -le 15 ]]; then
  echo "blocked RuntimeMaxSec watchdog killed sleep-300 in ${elapsed}s"
else
  echo "BREACH  sleep-300 ran for ${elapsed}s — RuntimeMaxSec did NOT enforce" >&2
  breached=1
fi

if [[ $breached -eq 0 ]]; then
  echo "BLOCKED OK — INV-19 holds (cgroup + RuntimeMaxSec budgets enforced)"
  exit 0
else
  echo "BREACH — INV-19 budget enforcement failed" >&2
  exit 1
fi
