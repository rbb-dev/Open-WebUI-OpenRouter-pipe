"""Tests for the pipe_dashboard authorization chokepoint (authz)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard import authz


def _pipe(pid="function_openrouter"):
    return SimpleNamespace(id=pid)


def _seam(*, model=None, read_ok=True, write_ok=False, verified=True, user=None):
    if model is None:
        model = Mock(user_id="admin1")

    async def check_model_access(u, m, bypass_filter=False):
        if not read_ok:
            raise HTTPException(status_code=403)

    def get_verified_user(u):
        if not verified:
            raise HTTPException(status_code=401)
        return u

    users = Mock()
    users.get_user_by_id = AsyncMock(return_value=user)
    models = Mock()
    models.get_model_by_id = AsyncMock(return_value=model)
    grants = Mock()
    grants.has_access = AsyncMock(return_value=write_ok)
    return SimpleNamespace(
        Users=users, Models=models, AccessGrants=grants,
        get_verified_user=get_verified_user, check_model_access=check_model_access,
        BYPASS_ADMIN=True, BYPASS_MODEL=False,
    )


def test_model_id():
    assert authz.model_id(_pipe("function_x")) == "function_x.pipe-dashboard"
    assert authz.model_id(SimpleNamespace(id=None)) is None


@pytest.mark.asyncio
async def test_can_view_granted(monkeypatch):
    user = SimpleNamespace(id="u1", role="user")
    monkeypatch.setattr(authz, "_owui", lambda: _seam(read_ok=True, user=user))
    assert await authz.can_view(user, _pipe()) is True


@pytest.mark.asyncio
async def test_can_view_denied_no_grant(monkeypatch):
    user = SimpleNamespace(id="u1", role="user")
    monkeypatch.setattr(authz, "_owui", lambda: _seam(read_ok=False, user=user))
    assert await authz.can_view(user, _pipe()) is False


@pytest.mark.asyncio
async def test_pending_denied(monkeypatch):
    user = SimpleNamespace(id="u1", role="pending")
    monkeypatch.setattr(authz, "_owui", lambda: _seam(verified=False, user=user))
    assert await authz.can_view(user, _pipe()) is False
    assert await authz.can_act(user, _pipe()) is False


@pytest.mark.asyncio
async def test_can_act_write_grant(monkeypatch):
    user = SimpleNamespace(id="u1", role="user")
    monkeypatch.setattr(authz, "_owui", lambda: _seam(write_ok=True, user=user))
    assert await authz.can_act(user, _pipe()) is True


@pytest.mark.asyncio
async def test_can_act_read_only_user_denied(monkeypatch):
    user = SimpleNamespace(id="u1", role="user")
    monkeypatch.setattr(
        authz, "_owui",
        lambda: _seam(read_ok=True, write_ok=False, model=Mock(user_id="someone_else"), user=user),
    )
    assert await authz.can_act(user, _pipe()) is False


@pytest.mark.asyncio
async def test_owner_can_act(monkeypatch):
    user = SimpleNamespace(id="owner1", role="user")
    monkeypatch.setattr(
        authz, "_owui",
        lambda: _seam(write_ok=False, model=Mock(user_id="owner1"), user=user),
    )
    assert await authz.can_act(user, _pipe()) is True


@pytest.mark.asyncio
async def test_admin_bypass_write(monkeypatch):
    user = SimpleNamespace(id="a1", role="admin")
    monkeypatch.setattr(
        authz, "_owui",
        lambda: _seam(write_ok=False, model=Mock(user_id="someone_else"), user=user),
    )
    assert await authz.can_act(user, _pipe()) is True


@pytest.mark.asyncio
async def test_none_user_denied(monkeypatch):
    monkeypatch.setattr(authz, "_owui", lambda: _seam())
    assert await authz.can_view(None, _pipe()) is False


@pytest.mark.asyncio
async def test_missing_pipe_id_denied(monkeypatch):
    user = SimpleNamespace(id="u1", role="user")
    monkeypatch.setattr(authz, "_owui", lambda: _seam(user=user))
    assert await authz.can_view(user, SimpleNamespace(id=None)) is False


@pytest.mark.asyncio
async def test_owui_import_failure_denies(monkeypatch):
    def boom():
        raise ImportError("no owui")

    monkeypatch.setattr(authz, "_owui", boom)
    assert await authz.can_view(SimpleNamespace(id="u1", role="user"), _pipe()) is False


@pytest.mark.asyncio
async def test_write_missing_model_denied(monkeypatch):
    user = SimpleNamespace(id="u1", role="user")
    monkeypatch.setattr(authz, "_owui", lambda: _seam(model=None, user=user))
    # get_model_by_id returns None → write path fails closed
    seam = _seam(user=user)
    seam.Models.get_model_by_id = AsyncMock(return_value=None)
    monkeypatch.setattr(authz, "_owui", lambda: seam)
    assert await authz.can_act(user, _pipe()) is False
