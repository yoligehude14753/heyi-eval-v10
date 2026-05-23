# Agent Sandbox (PR#22a + PR#22b)

Purpose: re-introduce a Claude Code agent into the evaluation pipeline (v10 was
fully Python-hardcoded as a reaction to v9 incidents) **without** giving the
agent the v9-era privileges (`docker` group + full `sudo`) that previously led
to store/* deletion and prod-engine restarts.

The agent runs as a dedicated Linux user `heyi-eval-agent` whose effective
privileges are constrained by six mechanisms; INV-16~21 each pins one of them
and `drills/` contains the adversarial verification scripts.

| Layer                       | Pinned by  | Drill                                       |
|-----------------------------|-----------|---------------------------------------------|
| filesystem ACL              | INV-16    | `drills/attack_delete_store.sh`             |
| docker access (proxy)       | INV-17    | `drills/attack_exec_prod.sh`                |
| audit log no-access         | INV-18    | `drills/attack_evade_audit.sh`              |
| cgroup + watchdog           | INV-19    | `drills/attack_resource_budget.sh`          |
| sudoers + identity          | INV-20    | `drills/attack_sudo_escalate.sh`            |
| audit write append-only     | INV-21    | `drills/attack_evade_audit_writes.sh` (daemon socket; M1 setuid wrapper was removed — see PR#22b-M2 note in INVARIANTS.md) |

All `drills/*.sh` MUST run as `heyi-eval-agent` (use `sudo -u heyi-eval-agent`)
and MUST exit non-zero when the protection works (i.e. the attack is blocked).
Running them as `ai` is meaningless and should be guarded against (`id -un`
check inside each script).

## Install order on nv8

```bash
cd /home/ai/heyi-eval-v10/deploy/agent-sandbox

# M1
sudo ./setup_agent_user.sh
sudo ./acl_install.sh
sudo install -m 0440 sudoers.d/heyi-eval-agent /etc/sudoers.d/heyi-eval-agent
sudo visudo -c -f /etc/sudoers.d/heyi-eval-agent   # syntax gate

# M2
sudo docker compose -f compose.agent-socket-proxy.yml up -d
# expect: container "heyi-eval-agent-socket-proxy" Up (healthy)
# bind: 127.0.0.1:2377 → container 2375 (loopback only)
sudo -u heyi-eval-agent bash drills/attack_exec_prod.sh
# expect: "BLOCKED OK — INV-17 holds" (4 reads → 200, 7 writes → 403)

# M3a — audit dir is part of acl_install.sh §5; re-running it is fine
sudo -u heyi-eval-agent bash drills/attack_evade_audit.sh
# expect: "BLOCKED OK — INV-18 holds" (6/6 access attempts EACCES)

# PR#22b-M2 — append-only audit write path via daemon socket (INV-21)
# M1 setuid wrapper was REMOVED — `NoNewPrivileges=true` in the agent
# unit blocks sudo's setuid, so the wrapper could never have worked from
# inside heyi-eval-agent@%i.service. M2 replaces it with a unix-socket
# daemon (heyi-eval-audit.service) that the agent connects to directly,
# no privilege change needed.
sudo install -m 0755 -o root -g root heyi-eval-agent-audit-client.py /usr/local/bin/heyi-eval-agent-audit-client
sudo install -m 0755 -o root -g root heyi-eval-agent-prepare         /usr/local/sbin/heyi-eval-agent-prepare
sudo install -m 0755 -o root -g root heyi-eval-agent-run             /usr/local/bin/heyi-eval-agent-run
sudo install -m 0644 ../systemd/heyi-eval-audit.service              /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now heyi-eval-audit.service
# expect: socket /run/heyi-eval-agent-audit.sock with mode 0660 root:heyi-eval-agent
sudo -u heyi-eval-agent bash drills/attack_evade_audit_writes.sh
# expect: "BLOCKED OK — INV-21 holds (append-only audit write path, daemon-fronted)"
#   4 socket-protocol attacks (unknown op / missing fields / non-JSON / end without begin) blocked
#   5 INV-18 ACL attacks (cat / dd / tee / truncate / rm) blocked
#   2 happy-path writes (client begin / client end via socket) succeed

# Verify the end-to-end agent unit:
sudo systemctl start heyi-eval-agent@m2demo.service
# expect: completes in <1s, journal shows prepare→audit-begin→smoke→audit-end
sudo cat /var/lib/heyi-eval-agent/runs/m2demo/outbox/run_meta.json
# expect: {"run_id":"m2demo","mode":"smoke","audit_id":N,"exit_code":0,...}

# M4
sudo install -m 0644 ../systemd/heyi-eval-agent.slice    /etc/systemd/system/
sudo install -m 0644 ../systemd/heyi-eval-agent@.service /etc/systemd/system/
sudo systemd-analyze verify /etc/systemd/system/heyi-eval-agent@.service /etc/systemd/system/heyi-eval-agent.slice
sudo systemctl daemon-reload
sudo bash drills/attack_resource_budget.sh
# expect: "BLOCKED OK — INV-19 holds"
#   5a: kernel cgroup-pids events ≥ 1
#   5b: RuntimeMaxSec watchdog kills sleep-300 in ≤ 15s

# M3, M4 — handled by their respective install scripts
```

After each milestone, run the corresponding `drills/` script as
`heyi-eval-agent` — it must report `BLOCKED OK` and exit 0 when the protection
holds, and `BREACH` exit 1 when it does not.

## What this directory does NOT contain

- `claude_code/` agent prompt or system message — that lives in `cc_agent/`
  (separate PR). PR#22a is purely the OS-level sandbox; the agent binary that
  runs inside it is plugged in by PR#22b.
- Any modification to existing pipeline stages (`orchestrator/stages.py` etc.)
  — INV-1~15 keep their contract; this PR only adds INV-16~20.
