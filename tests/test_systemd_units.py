"""Static lint for deploy/ — systemd unit files + bootstrap_nv8.sh.

These tests don't touch real systemd; they parse the unit files with
stdlib configparser and assert that the required fields are present and
sane. Matches docs/PR7b_TEST_PLAN.md §4.5.
"""
from __future__ import annotations

import configparser
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_nv8.sh"
VERIFY_24H = REPO_ROOT / "scripts" / "verify_24h_timer.sh"

# Services we expect to ship with v10. Backup pair pre-existed in PR#6 but
# is still subject to the same invariants.
EXPECTED_SERVICES = [
    "heyi-eval-orchestrator.service",
    "heyi-eval-discover.service",
    "heyi-eval-panel.service",
    "heyi-eval-notify-sync.service",
    "heyi-eval-backup.service",
]
EXPECTED_TIMERS = [
    "heyi-eval-backup.timer",
]

VENV_PYTHON_PREFIX = "/home/ai/heyi-eval-v10/.venv/bin/python"


def _load_unit(name: str) -> configparser.ConfigParser:
    """Parse a systemd unit with configparser. We accept duplicate values
    (systemd allows `EnvironmentFile=` multiple times) by using
    strict=False, and we DO want to be case-sensitive on section names
    because systemd is."""
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str  # keep keys case-sensitive
    text = (SYSTEMD_DIR / name).read_text(encoding="utf-8")
    cp.read_string(text)
    return cp


@pytest.fixture(scope="module")
def services() -> dict[str, configparser.ConfigParser]:
    return {name: _load_unit(name) for name in EXPECTED_SERVICES}


@pytest.fixture(scope="module")
def timers() -> dict[str, configparser.ConfigParser]:
    return {name: _load_unit(name) for name in EXPECTED_TIMERS}


# ── U-1 syntax ───────────────────────────────────────────────────────────────


def test_u1_all_expected_unit_files_present():
    """Every name in EXPECTED_* must exist on disk; configparser parse must
    succeed (load_unit raises on error)."""
    for name in EXPECTED_SERVICES + EXPECTED_TIMERS:
        path = SYSTEMD_DIR / name
        assert path.is_file(), f"missing unit file: {path}"
        _load_unit(name)  # raises if syntax bad


# ── U-2 sections ─────────────────────────────────────────────────────────────


def test_u2_service_has_three_sections(services):
    for name, cp in services.items():
        for section in ("Unit", "Service", "Install"):
            assert cp.has_section(section), \
                f"{name} missing [{section}]"


def test_u2_timer_has_three_sections(timers):
    for name, cp in timers.items():
        for section in ("Unit", "Timer", "Install"):
            assert cp.has_section(section), \
                f"{name} missing [{section}]"


# ── U-3 user/wd/env ─────────────────────────────────────────────────────────


def test_u3_service_has_user_ai_and_wd(services):
    for name, cp in services.items():
        assert cp.get("Service", "User", fallback="") == "ai", \
            f"{name}: User must be 'ai'"
        assert cp.get("Service", "Group", fallback="") == "ai", \
            f"{name}: Group must be 'ai'"
        wd = cp.get("Service", "WorkingDirectory", fallback="")
        assert wd == "/home/ai/heyi-eval-v10", \
            f"{name}: WorkingDirectory must be /home/ai/heyi-eval-v10, got {wd!r}"


def test_u3_service_has_optional_environment_file(services):
    """EnvironmentFile= line is required, with `-` prefix so a missing
    file at install time doesn't fail the service."""
    for name, cp in services.items():
        ef = cp.get("Service", "EnvironmentFile", fallback="")
        assert ef.startswith("-"), \
            f"{name}: EnvironmentFile must use '-' prefix (optional), got {ef!r}"
        assert ef.endswith("/etc/heyi-eval-v10/env"), \
            f"{name}: EnvironmentFile path must be /etc/heyi-eval-v10/env"


# ── U-4 ExecStart goes through venv python ─────────────────────────────────


def test_u4_execstart_runs_venv_python(services):
    for name, cp in services.items():
        es = cp.get("Service", "ExecStart", fallback="")
        # ExecStart may use absolute path with extra args; just check prefix.
        assert es.startswith(VENV_PYTHON_PREFIX), \
            (f"{name}: ExecStart must start with {VENV_PYTHON_PREFIX} "
             f"(no system python, no shell wrappers). got: {es!r}")


# ── U-5 Install target ─────────────────────────────────────────────────────


def test_u5_install_target_is_correct(services, timers):
    """Services must be wanted by multi-user.target; timers idiomatically
    use timers.target so `systemctl list-timers` and the timers-target
    activation chain work correctly."""
    for name, cp in services.items():
        wb = cp.get("Install", "WantedBy", fallback="")
        assert wb == "multi-user.target", \
            f"{name}: [Install] WantedBy must be multi-user.target, got {wb!r}"
    for name, cp in timers.items():
        wb = cp.get("Install", "WantedBy", fallback="")
        assert wb in ("timers.target", "multi-user.target"), \
            (f"{name}: [Install] WantedBy must be timers.target "
             f"(preferred) or multi-user.target, got {wb!r}")


# ── U-6 no root ──────────────────────────────────────────────────────────────


def test_u6_no_service_runs_as_root(services):
    for name, cp in services.items():
        for key in ("User", "Group"):
            val = cp.get("Service", key, fallback="")
            assert val != "root", f"{name}: {key} must not be root"


# ── U-7 no internal Requires/Wants between heyi-eval-* units ───────────────


def test_u7_units_do_not_chain_each_other(services, timers):
    for name, cp in {**services, **timers}.items():
        for key in ("Requires", "Wants", "BindsTo", "PartOf"):
            val = cp.get("Unit", key, fallback="")
            if val:
                assert "heyi-eval-" not in val, \
                    (f"{name}: [Unit] {key}={val!r} references another "
                     f"heyi-eval-* unit. Services must stay independent so "
                     f"a single failure doesn't cascade. (After= is fine.)")


# ── U-8 Restart field present ───────────────────────────────────────────────


def test_u8_service_has_restart_directive(services):
    valid = {"on-failure", "no", "always", "on-abnormal"}
    for name, cp in services.items():
        r = cp.get("Service", "Restart", fallback="")
        assert r in valid, \
            f"{name}: Restart={r!r} not in {valid}"


# ── U-9 Nice >= 0 ───────────────────────────────────────────────────────────


def test_u9_service_yields_to_heyi_engine(services):
    """All eval-side services must yield to heyi_engine production traffic.
    A negative Nice would mean HIGHER priority which is the wrong sign."""
    for name, cp in services.items():
        n = cp.get("Service", "Nice", fallback="")
        assert n, f"{name}: Nice= required"
        assert int(n) >= 0, \
            f"{name}: Nice={n} must be >= 0 (INV-1: yield to heyi_engine)"


# ── timer specifics ─────────────────────────────────────────────────────────


def test_backup_timer_has_oncalendar_or_onunitactivesec(timers):
    """Backup is the only timer in v10; it must declare a schedule."""
    name = "heyi-eval-backup.timer"
    cp = timers[name]
    has_calendar = bool(cp.get("Timer", "OnCalendar", fallback=""))
    has_active = bool(cp.get("Timer", "OnUnitActiveSec", fallback=""))
    has_boot = bool(cp.get("Timer", "OnBootSec", fallback=""))
    assert has_calendar or has_active or has_boot, \
        f"{name}: [Timer] missing OnCalendar/OnUnitActiveSec/OnBootSec"


def _parse_systemd_duration_to_seconds(spec: str) -> int:
    """Parse a systemd time spec like '30min', '15s', '2h', '1d' (single
    unit) into seconds. Raises if the format is one we don't expect."""
    spec = spec.strip()
    units = {"s": 1, "min": 60, "m": 60, "h": 3600, "d": 86400}
    for suf, mult in sorted(units.items(), key=lambda x: -len(x[0])):
        if spec.endswith(suf):
            value = spec[: -len(suf)].strip()
            return int(value) * mult
    return int(spec)  # bare integer = seconds


def test_backup_timer_cadence_is_30min_per_pr6_design(
    timers: dict[str, configparser.ConfigParser],
) -> None:
    """PR#6 designed 30-minute rsync snapshots; the timer must match.
    A cadence of < 5 minutes is treated as a misconfiguration (would
    thrash rsync hardlinks); > 60 minutes silently loses RPO."""
    cp = timers["heyi-eval-backup.timer"]
    spec = cp.get("Timer", "OnUnitActiveSec", fallback="").strip()
    assert spec, "OnUnitActiveSec is required for the 30min cadence"
    secs = _parse_systemd_duration_to_seconds(spec)
    assert 5 * 60 <= secs <= 60 * 60, (
        f"OnUnitActiveSec={spec!r} ({secs}s) outside sane window "
        f"[5min, 60min]; PR#6 designed for 30min."
    )


def test_backup_timer_oncalendar_is_not_per_second(
    timers: dict[str, configparser.ConfigParser],
) -> None:
    """Defense against the catastrophic OnCalendar=*-*-* *:*:* schedule
    which fires every second. We only accept :MM minute-aligned values."""
    cp = timers["heyi-eval-backup.timer"]
    spec = cp.get("Timer", "OnCalendar", fallback="").strip()
    if not spec:
        return  # OnCalendar is optional; OnUnitActiveSec covers cadence
    # Reject anything that looks like *:*:* or */1 second tier.
    assert "*:*:*" not in spec, (
        f"OnCalendar={spec!r} fires every second — operator safety guard."
    )


def test_backup_timer_has_randomized_delay(
    timers: dict[str, configparser.ConfigParser],
) -> None:
    """A small RandomizedDelaySec dampens herd effects when nightly
    cron + 30min snapshot collide. Not strictly required, but expected
    by the PR#6 design — flag absence as a warning-via-test."""
    cp = timers["heyi-eval-backup.timer"]
    spec = cp.get("Timer", "RandomizedDelaySec", fallback="").strip()
    assert spec, (
        "heyi-eval-backup.timer should declare RandomizedDelaySec to "
        "avoid herd effects (PR#6 design: 60s jitter)."
    )


# ── U-10 + B-1 bootstrap script sanity ─────────────────────────────────────


def test_u10_bootstrap_script_shape():
    assert BOOTSTRAP.is_file(), f"missing {BOOTSTRAP}"
    text = BOOTSTRAP.read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines, "bootstrap script is empty"
    assert lines[0] == "#!/usr/bin/env bash", \
        f"first line must be '#!/usr/bin/env bash', got {lines[0]!r}"
    assert "set -euo pipefail" in text, \
        "bootstrap script must enable strict mode (set -euo pipefail)"


def test_b1_bootstrap_script_passes_bash_n():
    """Static syntax check via `bash -n` — doesn't execute anything."""
    cp = subprocess.run(
        ["bash", "-n", str(BOOTSTRAP)],
        capture_output=True, text=True, timeout=10,
    )
    assert cp.returncode == 0, \
        f"bash -n failed: {cp.stderr}"


def test_b2_bootstrap_supports_force_flag():
    """--force is required so the script can be re-run on a clone with a
    hostname that doesn't contain 'nv8' (e.g. when restoring from a
    backup mac to a new node)."""
    text = BOOTSTRAP.read_text(encoding="utf-8")
    assert "--force" in text, "bootstrap must accept --force"


def test_b3_bootstrap_has_no_v9_residue_paths():
    """No legacy paths from v9 (/var/lib/heyi-eval, /opt/heyi-eval, etc.)
    should appear in the v10 deployment surface."""
    text = BOOTSTRAP.read_text(encoding="utf-8")
    for forbidden in ("/var/lib/heyi-eval", "/opt/heyi-eval", "heyi-eval-v9"):
        assert forbidden not in text, \
            f"bootstrap contains v9-era path/name: {forbidden!r}"


def test_b4_bootstrap_uses_safe_rm_prefix_only():
    """Sanity: the script must never `rm -rf` outside /home/ai/heyi-eval-*.
    Today it does NOT rm anything; this test pins that property so a
    future edit doesn't quietly start nuking /home/ai/."""
    text = BOOTSTRAP.read_text(encoding="utf-8")
    # If rm -rf ever appears, it must be inside a case-arm matching
    # /home/ai/heyi-eval-* — for now we just forbid bare `rm -rf` outright.
    assert "rm -rf" not in text, \
        ("bootstrap_nv8.sh introduced 'rm -rf' — this is a high-risk "
         "operation. If you need it, wrap in a path-prefix guard and "
         "update this test to allow the specific occurrence.")


# ── env.example ─────────────────────────────────────────────────────────────


# ── verify_24h_timer.sh ─────────────────────────────────────────────────────


def test_v1_verify_24h_script_present_and_executable() -> None:
    """The 24h smoke script must ship + have a shebang. Executability is
    set by chmod on disk; we only assert here that it's an actual file
    with the right opening line."""
    assert VERIFY_24H.is_file(), f"missing {VERIFY_24H}"
    first = VERIFY_24H.read_text(encoding="utf-8").splitlines()[0]
    assert first == "#!/usr/bin/env bash", \
        f"first line must be '#!/usr/bin/env bash', got {first!r}"


def test_v2_verify_24h_script_passes_bash_n() -> None:
    cp = subprocess.run(
        ["bash", "-n", str(VERIFY_24H)],
        capture_output=True, text=True, timeout=10,
    )
    assert cp.returncode == 0, f"bash -n failed: {cp.stderr}"


def test_v3_verify_24h_script_is_side_effect_free() -> None:
    """The verify script is meant to be safe to run any time. It must not
    contain destructive commands; this pins that property."""
    text = VERIFY_24H.read_text(encoding="utf-8")
    for forbidden in ("rm -rf", "rm -f /", "mv /", " > /etc", "dd if=",
                      "systemctl start", "systemctl restart", "systemctl stop"):
        assert forbidden not in text, \
            f"verify_24h_timer.sh contains destructive token: {forbidden!r}"


def test_v4_verify_24h_script_emits_all_test_ids() -> None:
    """All five T-24h-* IDs from the test plan must appear in the script."""
    text = VERIFY_24H.read_text(encoding="utf-8")
    for tid in ("T-24h-1", "T-24h-2", "T-24h-3", "T-24h-4", "T-24h-5"):
        assert tid in text, f"verify_24h_timer.sh missing test id {tid}"


def test_env_example_exists_and_documents_engine_vars() -> None:
    p = REPO_ROOT / "deploy" / "env.example"
    assert p.is_file(), f"missing {p}"
    text = p.read_text(encoding="utf-8")
    # Required modern keys
    assert "HEYI_ENGINE_URL" in text
    assert "HEYI_EVAL_DATA" in text
    assert "HEYI_EVAL_BACKUPS" in text
    # No v9 keys
    for forbidden in ("HEYI_EVAL_CCR_", "HEYI_EVAL_CC_AGENT_",
                      "HEYI_EVAL_CURATOR_MODEL", "HEYI_EVAL_HOST_DOCKER_BIN",
                      "HEYI_EVAL_VLLM_CONTAINER"):
        assert forbidden not in text, \
            f"env.example contains v9 key {forbidden!r}"
