"""Panel write-surface auth guard (POST).

When ``HEYI_PANEL_TOKEN`` is set, mutating requests must carry a matching
``Authorization: Bearer <token>``; when unset the panel stays open
(historical Tailnet-only posture).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from panel import server  # noqa: E402


def _stub(auth: str | None):
    headers = {"Authorization": auth} if auth is not None else {}
    return types.SimpleNamespace(headers=headers)


def test_open_when_token_unset(monkeypatch) -> None:
    monkeypatch.setattr(server, "PANEL_TOKEN", None)
    assert server.Handler._post_authorized(_stub(None)) is True
    assert server.Handler._post_authorized(_stub("Bearer anything")) is True


def test_requires_matching_bearer_when_token_set(monkeypatch) -> None:
    monkeypatch.setattr(server, "PANEL_TOKEN", "s3cret")
    assert server.Handler._post_authorized(_stub("Bearer s3cret")) is True


def test_rejects_missing_or_wrong_token(monkeypatch) -> None:
    monkeypatch.setattr(server, "PANEL_TOKEN", "s3cret")
    assert server.Handler._post_authorized(_stub(None)) is False
    assert server.Handler._post_authorized(_stub("")) is False
    assert server.Handler._post_authorized(_stub("Bearer wrong")) is False
    assert server.Handler._post_authorized(_stub("s3cret")) is False  # no Bearer prefix
