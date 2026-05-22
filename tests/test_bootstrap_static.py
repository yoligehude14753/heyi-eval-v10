"""Static guards for ``scripts/bootstrap_nv8.sh`` (PR#12).

We don't run bootstrap in CI — it needs docker / nvidia-smi / systemctl,
none of which exist on mac or CI runners. Instead we scan the script's
source text for properties that PR#12 promised:

  B1. There is a ``find_python_311_plus`` function (or equivalent) that
      tries multiple candidate binaries instead of hard-coding one.
  B2. ``require_cmd python3.11`` (a hard single-version require) is gone.
  B3. The venv creation step does *not* hard-code ``python3.11`` — it
      uses a variable populated by the detection helper.
  B4. The hostname guard at preflight still substring-matches ``*nv8*``
      and still honors ``--force`` (the original safety contract).

These guards exist so a future "let me just hardcode python3.11 back"
patch fails fast.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_nv8.sh"


@pytest.fixture(scope="module")
def bootstrap_text() -> str:
    assert BOOTSTRAP.is_file(), f"missing {BOOTSTRAP}"
    return BOOTSTRAP.read_text(encoding="utf-8")


# ── B1: detection helper exists with multi-version fallback ───────────────


def test_b1_find_python_helper_exists(bootstrap_text: str) -> None:
    """The bootstrap must define a helper that searches for >=3.11 across
    multiple candidate binaries, not hard-code one."""
    assert "find_python_311_plus" in bootstrap_text, (
        "bootstrap_nv8.sh missing find_python_311_plus helper (PR#12 §1)"
    )
    # The fallback chain must mention at least python3.11 and one higher
    # version, otherwise it's not really a chain.
    for candidate in ("python3.11", "python3.12", "python3.13"):
        assert candidate in bootstrap_text, (
            f"bootstrap_nv8.sh fallback chain missing {candidate!r} "
            f"(PR#12 requires ≥3 candidates so SLES/Ubuntu/macos all work)"
        )


# ── B2: no hard single-version require_cmd ────────────────────────────────


def test_b2_no_hardcoded_python311_require_cmd(bootstrap_text: str) -> None:
    """The original line ``require_cmd python3.11`` (which would explode on
    a host with only python3.12 installed) must not reappear."""
    for line in bootstrap_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):  # comments are fine
            continue
        assert stripped != "require_cmd python3.11", (
            f"bootstrap_nv8.sh still hard-codes `require_cmd python3.11` "
            f"in line {line!r} — PR#12 says use find_python_311_plus"
        )


# ── B3: venv creation uses variable, not hard-coded version ───────────────


def test_b3_venv_creation_uses_detected_python(bootstrap_text: str) -> None:
    """The venv creation line must NOT contain ``python3.11 -m venv``
    (hard-coded). It must reference a variable (``${PYTHON_BIN}`` or
    similar) populated by the detection helper."""
    offenders: list[str] = []
    for line_no, line in enumerate(bootstrap_text.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "python3.11 -m venv" in stripped:
            offenders.append(f"line {line_no}: {stripped}")
    assert not offenders, (
        "venv creation must use the detected python binary, not "
        "hard-coded python3.11:\n  " + "\n  ".join(offenders)
    )

    # And we expect a positive signal: some form of ``${PYTHON_BIN}`` or
    # ``$PYTHON_BIN`` appears in the venv block.
    assert "PYTHON_BIN" in bootstrap_text, (
        "bootstrap_nv8.sh: expected a PYTHON_BIN variable set by "
        "find_python_311_plus and consumed by the venv step"
    )


# ── B4: hostname guard and --force still present (safety contract) ────────


def test_b4_hostname_guard_intact(bootstrap_text: str) -> None:
    """PR#12 must not weaken preflight: the substring nv8 hostname check
    and the ``--force`` escape hatch must both still be present."""
    assert "*nv8*" in bootstrap_text or "nv8" in bootstrap_text, (
        "hostname guard removed — refuse on non-nv8 hosts is a safety "
        "contract, not negotiable in PR#12"
    )
    assert "FORCE" in bootstrap_text, (
        "--force escape hatch removed — required for ops to override "
        "when running on staging clones"
    )
    assert "--force" in bootstrap_text, (
        "--force cli flag handler missing"
    )
