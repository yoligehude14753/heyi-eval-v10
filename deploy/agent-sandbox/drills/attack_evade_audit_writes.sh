#!/usr/bin/env bash
# Attack drill 6 — INV-21 (audit write path is append-only, daemon-fronted).
#
# History: M1 fronted the audit write path with a setuid wrapper +
# sudoers NOPASSWD. M2 had to drop that design because the agent
# unit's NoNewPrivileges=true is incompatible with sudo's setuid.
# This drill now exercises the M2 architecture (unix-socket daemon).
#
# Hypothesis: heyi-eval-agent tries to manipulate the audit log by:
#   (a) talking to the daemon socket with malformed payloads (must
#       return {"error":...} not crash)
#   (b) bypassing the daemon and writing the DB file directly via
#       cat/dd/tee/truncate/rm (must EACCES — INV-18 deny-all ACL)
#   (c) calling the daemon with a legitimate begin/end pair (HAPPY
#       PATH; drill fails if this breaks too)
#
# Run as: sudo -u heyi-eval-agent bash <this>
# Exit:   0 BLOCKED OK / 1 BREACH / 2 misconfigured
set -u

CLIENT="${HEYI_EVAL_AUDIT_CLIENT:-/usr/local/bin/heyi-eval-agent-audit-client}"
SOCKET="${HEYI_EVAL_AUDIT_SOCKET:-/run/heyi-eval-agent-audit.sock}"
DB="${HEYI_EVAL_AUDIT_DB:-/var/log/heyi-eval-agent/audit.sqlite}"

cur=$(id -un)
if [[ "$cur" != "heyi-eval-agent" ]]; then
  echo "drill misconfigured: must run as heyi-eval-agent, got $cur" >&2
  exit 2
fi
[[ -x "$CLIENT" ]] || { echo "drill misconfigured: $CLIENT not executable (bootstrap §4b)" >&2; exit 2; }
[[ -S "$SOCKET" ]] || { echo "drill misconfigured: $SOCKET socket missing (heyi-eval-audit.service running?)" >&2; exit 2; }

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
  # status to stderr, raw stdout returned (so caller can pipe)
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

# (a) socket-level malformed input — daemon must respond with error,
#     not crash. We hand-craft requests with python to bypass the CLI.
probe_via_python() {
  local label="$1"
  local payload="$2"
  local expected_re="$3"
  local out
  out=$(python3 - "$SOCKET" <<PY 2>&1
import json, socket, sys
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(3.0)
s.connect(sys.argv[1])
s.sendall((${payload@Q} + "\n").encode())
data = b""
while b"\n" not in data and len(data) < 8192:
    chunk = s.recv(4096)
    if not chunk: break
    data += chunk
print(data.decode("utf-8", "replace").strip())
PY
  )
  if [[ "$out" =~ $expected_re ]]; then
    echo "blocked $label (daemon returned: $out)"
  else
    echo "BREACH  $label expected /$expected_re/, got: $out" >&2
    breached=1
  fi
}

probe_via_python "unknown op 'delete'"    '{"op":"delete"}'             '"error".*unknown op'
probe_via_python "missing run_id"         '{"op":"begin","argv":["x"]}' '"error"'
probe_via_python "non-JSON payload"       'not-json-at-all'             '"error".*JSON'
probe_via_python "end without begin"      '{"op":"end","audit_id":99999,"exit_code":0,"duration_ms":1}' '"error"'

# (b) direct DB file attacks — INV-18 deny-all ACL must EACCES every path
probe_blocked "cat audit DB"           cat "$DB"
probe_blocked "dd if=audit DB"         dd if="$DB" of=/dev/null count=1
probe_blocked "tee -a >> audit DB"     bash -c "echo X | tee -a '$DB' >/dev/null"
probe_blocked "truncate audit DB to 0" bash -c "exec 3>'$DB'"
probe_blocked "rm audit DB"            rm -f "$DB"

# (c) HAPPY PATH — drill fails if legitimate begin/end stops working
audit_id=$(probe_ok "client begin" \
    "$CLIENT" --socket "$SOCKET" begin \
        --run-id drill6 \
        --cwd /tmp \
        -- echo drill6-probe)
if [[ -z "$audit_id" || ! "$audit_id" =~ ^[0-9]+$ ]]; then
  echo "BREACH  happy-path begin did not return numeric audit_id (got: ${audit_id@Q})" >&2
  breached=1
else
  echo "  begin returned audit_id=$audit_id"
  probe_ok "client end" \
      "$CLIENT" --socket "$SOCKET" end \
          --audit-id "$audit_id" \
          --exit-code 0 \
          --duration-ms 5 >/dev/null
fi

if [[ $breached -eq 0 ]]; then
  echo "BLOCKED OK — INV-21 holds (append-only audit write path, daemon-fronted)"
  exit 0
else
  echo "BREACH — INV-21 broken" >&2
  exit 1
fi
