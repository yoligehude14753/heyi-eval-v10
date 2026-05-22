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
| audit write append-only     | INV-21    | `drills/attack_evade_audit_writes.sh`       |

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

# PR#22b-M1 — append-only audit write path (INV-21)
# Wrapper is owned root:root 0755 so the agent can exec it and hit
# the EUID check, but the actual append is gated by sudo NOPASSWD.
sudo install -m 0755 -o root -g root heyi-eval-agent-audit-record /usr/local/sbin/
sudo /usr/local/sbin/heyi-eval-agent-audit-record init
# expect: "audit schema initialised at /var/log/heyi-eval-agent/audit.sqlite"
sudo -u heyi-eval-agent bash drills/attack_evade_audit_writes.sh
# expect: "BLOCKED OK — INV-21 holds"
#   4 wrapper-policy attacks (no-sudo / update / delete / drop) blocked
#   5 INV-18 ACL attacks (cat / dd / tee / truncate / rm) blocked
#   2 happy-path writes (begin/end via sudo wrapper) succeed

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
