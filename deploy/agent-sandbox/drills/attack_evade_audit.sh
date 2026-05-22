#!/usr/bin/env bash
# Attack drill 4 — INV-18 (audit log is unreachable to the agent).
#
# Hypothesis: heyi-eval-agent attempts to read, list, delete, truncate,
# or overwrite the audit log dir. ACL must block all of it with EACCES.
#
# This drill stays meaningful even before the audit *daemon* lands in
# PR#22b — the empty placeholder audit.sqlite is created by
# acl_install.sh so this attack has a concrete target.
#
# Run as: sudo -u heyi-eval-agent bash <this>
# Exit:   0 BLOCKED OK / 1 BREACH / 2 misconfigured
set -u

AUDIT_DIR="${HEYI_EVAL_AUDIT_DIR:-/var/log/heyi-eval-agent}"

cur=$(id -un)
if [[ "$cur" != "heyi-eval-agent" ]]; then
  echo "drill misconfigured: must run as heyi-eval-agent, got $cur" >&2
  exit 2
fi
[[ -d "$AUDIT_DIR" ]] || { echo "drill misconfigured: $AUDIT_DIR missing (run acl_install.sh)" >&2; exit 2; }

breached=0

probe_blocked() {
  local label="$1"; shift
  local out rc
  out=$("$@" 2>&1)
  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "BREACH  $label succeeded (rc=0, output: $out)" >&2
    breached=1
  else
    echo "blocked $label (rc=$rc)"
  fi
}

echo "[drill-4] hitting audit dir as $cur (dir=$AUDIT_DIR)"

# (a) list the dir
probe_blocked "ls $AUDIT_DIR"                       ls "$AUDIT_DIR"
# (b) cat the audit DB
probe_blocked "cat $AUDIT_DIR/audit.sqlite"         cat "$AUDIT_DIR/audit.sqlite"
# (c) rm the audit DB
probe_blocked "rm -f $AUDIT_DIR/audit.sqlite"       rm -f "$AUDIT_DIR/audit.sqlite"
# (d) truncate the audit DB
probe_blocked "truncate -s 0 audit.sqlite"          truncate -s 0 "$AUDIT_DIR/audit.sqlite"
# (e) plant a hostile file
probe_blocked "echo > $AUDIT_DIR/pwn-$$"            bash -c ": >$AUDIT_DIR/pwn-$$"
# (f) overwrite via redirection (subtle: shell evals path before exec)
probe_blocked "redirect into audit.sqlite"          bash -c ": >$AUDIT_DIR/audit.sqlite"

if [[ $breached -eq 0 ]]; then
  echo "BLOCKED OK — INV-18 holds (audit log unreachable to agent)"
  exit 0
else
  echo "BREACH — INV-18 broken" >&2
  exit 1
fi
