"""Tests for the pipe_dashboard HTTP action route (header-only bearer + APIRoute)."""

from __future__ import annotations

import json
import sys
import types
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.core.config import Valves
from open_webui_openrouter_pipe.plugins.pipe_dashboard import http_routes


def _on_valves() -> Any:
    """A pipe whose master plugin switch reads on: the route 404s without it."""
    return SimpleNamespace(id="openrouter", valves=Valves(ENABLE_PLUGIN_SYSTEM=True))


class _PersistedSwitchOn:
    """Open WebUI's `Functions` stand-in holding the row an operator's save leaves.

    The route guard reads the PERSISTED row, and that field's declared default is off,
    so a deployment whose action route answers carries `ENABLE_PLUGIN_SYSTEM` in its
    row -- writing it on stores the key. A readable row without it is a route the
    master switch has closed, which is not what any test in this file is about.
    """

    def __init__(self) -> None:
        self.valves = {"ENABLE_PLUGIN_SYSTEM": True}

    async def get_function_by_id(self, id, db=None):
        return SimpleNamespace(updated_at=1000)

    async def get_function_valves_by_id(self, id, db=None):
        return dict(self.valves)

    async def update_function_valves_by_id(self, id, valves, db=None):
        self.valves = dict(valves)
        return SimpleNamespace(updated_at=1001)


@pytest.fixture(autouse=True)
def _persisted_master_switch_on(monkeypatch):
    import open_webui.models.functions as functions_mod

    monkeypatch.setattr(functions_mod, "Functions", _PersistedSwitchOn())


def _install_auth_stub(monkeypatch, *, decode=lambda t: {"id": "u1"}, valid=True,
                       user=SimpleNamespace(id="u1", role="user")):
    auth: Any = types.ModuleType("open_webui.utils.auth")
    auth.decode_token = decode

    async def is_valid_token(data, redis):
        return valid

    auth.is_valid_token = is_valid_token
    users_mod: Any = types.ModuleType("open_webui.models.users")
    users = Mock()
    users.get_user_by_id = AsyncMock(return_value=user)
    users_mod.Users = users
    monkeypatch.setitem(sys.modules, "open_webui.utils.auth", auth)
    monkeypatch.setitem(sys.modules, "open_webui.models.users", users_mod)


def _req(headers=None, app_redis=None):
    r = Mock()
    r.headers = headers or {}
    r.app = SimpleNamespace(state=SimpleNamespace(redis=app_redis))
    r.client = SimpleNamespace(host="1.2.3.4")
    return r


@pytest.mark.asyncio
async def test_bearer_missing_header_401(monkeypatch):
    _install_auth_stub(monkeypatch)
    with pytest.raises(HTTPException) as e:
        await http_routes.bearer_user(_req(headers={}))
    assert e.value.status_code == 401


@pytest.mark.asyncio
async def test_bearer_cookie_ignored_401(monkeypatch):
    _install_auth_stub(monkeypatch)
    with pytest.raises(HTTPException):
        await http_routes.bearer_user(_req(headers={"Cookie": "token=abc"}))


@pytest.mark.asyncio
async def test_bearer_valid_header(monkeypatch):
    _install_auth_stub(monkeypatch)
    user = await http_routes.bearer_user(_req(headers={"Authorization": "Bearer good"}))
    assert user.id == "u1"


@pytest.mark.asyncio
async def test_bearer_revoked_401(monkeypatch):
    _install_auth_stub(monkeypatch, valid=False)
    with pytest.raises(HTTPException):
        await http_routes.bearer_user(_req(headers={"Authorization": "Bearer x"}))


@pytest.mark.asyncio
async def test_bearer_bad_token_401_not_500(monkeypatch):
    def boom(t):
        raise ValueError("bad")

    _install_auth_stub(monkeypatch, decode=boom)
    with pytest.raises(HTTPException):
        await http_routes.bearer_user(_req(headers={"Authorization": "Bearer x"}))


@pytest.mark.asyncio
async def test_bearer_pending_role_401(monkeypatch):
    _install_auth_stub(monkeypatch, user=SimpleNamespace(id="u1", role="pending"))
    with pytest.raises(HTTPException):
        await http_routes.bearer_user(_req(headers={"Authorization": "Bearer x"}))


def test_ensure_route_before_spa_moves_spa_last():
    spa = SimpleNamespace(name="spa-static-files")
    other = SimpleNamespace(name="api")
    app = SimpleNamespace(routes=[spa, other])
    http_routes.ensure_route_before_spa(app)
    assert app.routes[-1] is spa


def test_client_ip_prefers_forwarded():
    r = _req(headers={"x-forwarded-for": "5.6.7.8, 9.9.9.9"})
    assert http_routes._client_ip(r) == "5.6.7.8"
    assert http_routes._client_ip(_req(headers={})) == "1.2.3.4"


def test_route_binds_body_200_not_422(monkeypatch):
    """Regression guard: request: Request (not Any) → FastAPI injects Request
    and validates ActionBody from JSON, returning 200 not 422."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(http_routes, "bearer_user", AsyncMock(return_value=SimpleNamespace(id="u1", role="user")))
    monkeypatch.setattr(http_routes, "_live_dispatch", lambda: AsyncMock(return_value=(200, {"ok": True, "result": {"x": 1}})))
    http_routes.set_pipe_getter(lambda: SimpleNamespace(id="p", valves=Valves(ENABLE_PLUGIN_SYSTEM=True)))
    http_routes._coarse_state.clear()

    app = FastAPI()
    app.add_api_route(http_routes._ACTION_PATH, http_routes._action_route, methods=["POST"])
    client = TestClient(app)

    ok = client.post(http_routes._ACTION_PATH, json={"action": "whoami", "args": {}})
    assert ok.status_code == 200 and ok.json()["ok"] is True

    missing = client.post(http_routes._ACTION_PATH, json={"args": {}})
    assert missing.status_code == 422  # body validation ran → proves Request injection

    non_json = client.post(http_routes._ACTION_PATH, content="not json",
                           headers={"Content-Type": "application/json"})
    assert non_json.status_code == 422


def test_route_forbidden_flows_through(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    monkeypatch.setattr(http_routes, "bearer_user", AsyncMock(return_value=SimpleNamespace(id="u2", role="user")))
    monkeypatch.setattr(http_routes, "_live_dispatch", lambda: AsyncMock(return_value=(403, {"error": "forbidden"})))
    http_routes.set_pipe_getter(lambda: SimpleNamespace(id="p", valves=Valves(ENABLE_PLUGIN_SYSTEM=True)))
    http_routes._coarse_state.clear()

    app = FastAPI()
    app.add_api_route(http_routes._ACTION_PATH, http_routes._action_route, methods=["POST"])
    client = TestClient(app)
    r = client.post(http_routes._ACTION_PATH, json={"action": "echo", "args": {"message": "x"}})
    assert r.status_code == 403


def test_register_action_route_degrades_closed(monkeypatch):
    monkeypatch.setattr(http_routes, "get_owui_app", lambda: None)
    http_routes._registered_paths.clear()
    assert http_routes.register_action_route() is False


def test_registered_route_resolves_real_config_get(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from open_webui_openrouter_pipe.core.config import Valves
    from open_webui_openrouter_pipe.plugins.pipe_dashboard import actions

    app = FastAPI()
    monkeypatch.setattr(http_routes, "get_owui_app", lambda: app)
    monkeypatch.setattr(http_routes, "bearer_user",
                        AsyncMock(return_value=SimpleNamespace(id="u1", role="admin")))
    monkeypatch.setattr(actions, "can_view", AsyncMock(return_value=True))
    monkeypatch.setattr(http_routes, "can_view", AsyncMock(return_value=True))
    monkeypatch.setattr(http_routes, "_resolve_fresh", AsyncMock(return_value=None))
    monkeypatch.setattr(http_routes, "_fresh_dispatch", None)
    monkeypatch.setattr(http_routes, "_reconcile_retry_until", 0.0)
    monkeypatch.setattr(actions, "_current_config_rev", AsyncMock(return_value=1000))
    http_routes.set_pipe_getter(lambda: SimpleNamespace(id="openrouter", valves=Valves(ENABLE_PLUGIN_SYSTEM=True)))
    http_routes._registered_paths.clear()
    actions._rate_state.clear()

    assert http_routes.register_action_route() is True
    client = TestClient(app)

    http_routes._coarse_state.clear()
    ok = client.post(http_routes._ACTION_PATH, json={"action": "config_get", "args": {}})
    assert ok.status_code == 200, ok.json()
    assert "valves" in ok.json()["result"]

    http_routes._coarse_state.clear()
    missing = client.post(http_routes._ACTION_PATH, json={"action": "does_not_exist", "args": {}})
    assert missing.status_code == 404
    http_routes._registered_paths.clear()


def test_route_self_heals_unknown_action(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    served: dict = {}

    async def _fresh(pipe, user, name, args, client_ip=None):
        served["pipe"] = pipe
        return 200, {"ok": True, "result": {"healed": name}}

    fresh_pipe = SimpleNamespace(id="openrouter", marker="fresh")
    serving_pipe = SimpleNamespace(
        id="openrouter", marker="serving", valves=Valves(ENABLE_PLUGIN_SYSTEM=True)
    )
    app = FastAPI()
    monkeypatch.setattr(http_routes, "get_owui_app", lambda: app)
    monkeypatch.setattr(http_routes, "bearer_user",
                        AsyncMock(return_value=SimpleNamespace(id="u1", role="user")))
    monkeypatch.setattr(http_routes, "can_view", AsyncMock(return_value=True))
    monkeypatch.setattr(http_routes, "_resolve_fresh", AsyncMock(return_value=(_fresh, fresh_pipe)))
    monkeypatch.setattr(http_routes, "_fresh_dispatch", None)
    monkeypatch.setattr(http_routes, "_reconcile_retry_until", 0.0)
    http_routes.set_pipe_getter(lambda: serving_pipe)
    http_routes._registered_paths.clear()

    assert http_routes.register_action_route() is True
    client = TestClient(app)
    http_routes._coarse_state.clear()
    r = client.post(http_routes._ACTION_PATH, json={"action": "config_get_v99", "args": {}})
    assert r.status_code == 200
    # The reconcile's freshly exec'd Pipe has `_plugin_registry = None` -- only
    # `pipes()` initialises it, and an action route never reaches `pipes()`. So the
    # reconcile's dispatcher runs against the pipe the plugin system initialised.
    assert getattr(served["pipe"], "marker", "") == "serving"
    assert r.json()["result"]["healed"] == "config_get_v99"
    http_routes._registered_paths.clear()


def test_route_reconcile_requires_can_view(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from open_webui_openrouter_pipe.plugins.pipe_dashboard import actions

    app = FastAPI()
    resolve = AsyncMock(return_value=None)
    monkeypatch.setattr(http_routes, "get_owui_app", lambda: app)
    monkeypatch.setattr(http_routes, "bearer_user",
                        AsyncMock(return_value=SimpleNamespace(id="u1", role="user")))
    monkeypatch.setattr(http_routes, "can_view", AsyncMock(return_value=False))
    monkeypatch.setattr(actions, "can_view", AsyncMock(return_value=False))
    monkeypatch.setattr(http_routes, "_resolve_fresh", resolve)
    monkeypatch.setattr(http_routes, "_fresh_dispatch", None)
    monkeypatch.setattr(http_routes, "_reconcile_retry_until", 0.0)
    http_routes.set_pipe_getter(_on_valves)
    http_routes._registered_paths.clear()
    actions._rate_state.clear()

    assert http_routes.register_action_route() is True
    client = TestClient(app)
    http_routes._coarse_state.clear()
    r = client.post(http_routes._ACTION_PATH, json={"action": "unknown_x", "args": {}})
    assert r.status_code == 403
    resolve.assert_not_awaited()
    http_routes._registered_paths.clear()


def _post(client, action):
    from open_webui_openrouter_pipe.plugins.pipe_dashboard import actions

    http_routes._coarse_state.clear()
    actions._rate_state.clear()
    return client.post(http_routes._ACTION_PATH, json={"action": action, "args": {}})
