#!/usr/bin/env bash
# Attack drill 6 — INV-21 (audit write path is append-only).
#
# Hypothesis: heyi-eval-agent tries to manipulate the audit log by:
#   (a) calling the wrapper without sudo (must fail — wrapper refuses
#       non-root EUID)
#   (b) calling the wrapper with a non-whitelisted subcommand like
#       `update` or `delete` (must fail — wrapper rejects)
#   (c) bypassing the wrapper and writing the DB directly via sqlite3
#       (must fail — INV-18 ACL denies access)
#   (d) calling the wrapper LEGITIMATELY with begin/end (must SUCCEED —
#       this is the happy path, and drill 6 fails if the happy path
#       breaks too).
#
# Run as: sudo -u heyi-eval-agent bash <this>
# Exit:   0 BLOCKED OK / 1 BREACH / 2 misconfigured
set -u

WRAPPER="${HEYI_EVAL_AUDIT_WRAPPER:-/usr/local/sbin/heyi-eval-agent-audit-record}"
DB="${HEYI_EVAL_AUDIT_DB:-/var/log/heyi-eval-agent/audit.sqlite}"

cur=$(id -un)
if [[ "$cur" != "heyi-eval-agent" ]]; then
  echo "drill misconfigured: must run as heyi-eval-agent, got $cur" >&2
  exit 2
fi
[[ -x "$WRAPPER" ]] || { echo "drill misconfigured: $WRAPPER not executable (run bootstrap §4b)" >&2; exit 2; }

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

probe_ok() {
  # IMPORTANT: status message goes to stderr so stdout is the raw
  # command output only (callers like `audit_id=$(probe_ok ...)`
  # depend on this).
  local label="$1"; shift
  local out rc
  out=$("$@" 2>&1)
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "BREACH  happy path $label failed (rc=$rc, output: $out)" >&2
    breached=1
    return
  fi
  echo "ok      $label (rc=0)" >&2
  printf '%s' "$out"
}

echo "[drill-6] audit-write hardening as $cur"

# (a) call the wrapper directly (no sudo) — must refuse EUID-non-zero
probe_blocked "direct call to wrapper (no sudo)" \
    "$WRAPPER" begin --run-id drill6 -- /bin/true

# (b) call sudo with a forbidden subcommand
probe_blocked "sudo wrapper 'update' subcommand" \
    sudo -n "$WRAPPER" update --run-id drill6
probe_blocked "sudo wrapper 'delete' subcommand" \
    sudo -n "$WRAPPER" delete --audit-id 1
probe_blocked "sudo wrapper 'drop' subcommand" \
    sudo -n "$WRAPPER" drop

# (c) bypass the wrapper and touch the DB file directly — INV-18 ACL
# (deny-all on /var/log/heyi-eval-agent) must EACCES every path:
#   - read     (cat / dd)
#   - append   (>> via tee)
#   - truncate (: > via bash redirect)
#   - rm       (covered in drill 4 already, included here for completeness)
# We deliberately use only POSIX coreutils so the drill works on
# any minimal sandbox image (sqlite3 binary is not assumed present).
probe_blocked "cat audit DB"                cat "$DB"
probe_blocked "dd if=audit DB"              dd if="$DB" of=/dev/null count=1
probe_blocked "tee -a >> audit DB"          bash -c "echo X | tee -a '$DB' >/dev/null"
probe_blocked "truncate audit DB to 0"      bash -c "exec 3>'$DB'"
probe_blocked "rm audit DB"                 rm -f "$DB"

# (d) HAPPY PATH — drill 6 must verify the legitimate write still works
audit_id=$(probe_ok "sudo wrapper begin" \
    sudo -n "$WRAPPER" begin --run-id drill6 --cwd /tmp -- echo drill6-probe)
if [[ -z "$audit_id" || ! "$audit_id" =~ ^[0-9]+$ ]]; then
  echo "BREACH  happy-path begin did not return numeric audit_id (got: ${audit_id@Q})" >&2
  breached=1
else
  echo "  begin returned audit_id=$audit_id"
  probe_ok "sudo wrapper end" \
      sudo -n "$WRAPPER" end --audit-id "$audit_id" --exit-code 0 --duration-ms 5 >/dev/null
fi

if [[ $breached -eq 0 ]]; then
  echo "BLOCKED OK — INV-21 holds (append-only audit write path)"
  exit 0
else
  echo "BREACH — INV-21 broken" >&2
  exit 1
fi
