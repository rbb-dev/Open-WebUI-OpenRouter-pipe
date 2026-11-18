from __future__ import annotations

import asyncio
import contextlib
import importlib
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi import FastAPI

pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.plugins.pipe_dashboard import (
    dashboard_publisher,
    dashboard_socket,
    http_routes,
)
from tests._owui_auth_stub import install_open_webui_auth, token_for
from conftest import arm_dashboard_pipe

_LIFECYCLE_KEY = "_openrouter_pipe_lifecycle"
_ADMIN = SimpleNamespace(id="admin-1", role="admin", email="admin-1@example.com")
_REAL_SLEEP = asyncio.sleep


class _StoredSettings:
    async def get_function_by_id(self, id, db=None):
        return SimpleNamespace(updated_at=1000)

    async def get_function_valves_by_id(self, id, db=None):
        return {"ENABLE_PLUGIN_SYSTEM": True, "PIPE_DASHBOARD_ENABLE": True}


@pytest.fixture(autouse=True)
def _plugin_system_saved_on(monkeypatch):
    import open_webui.models.functions as functions_mod

    monkeypatch.setattr(functions_mod, "Functions", _StoredSettings())


@pytest.fixture(autouse=True)
def _own_lifecycle_registry():
    saved = sys.modules.pop(_LIFECYCLE_KEY, None)
    yield
    sys.modules.pop(_LIFECYCLE_KEY, None)
    if saved is not None:
        sys.modules[_LIFECYCLE_KEY] = saved


class _OpenWebUIRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}

    async def set(self, key, value, nx=False, ex=None):
        if nx and key in self.values:
            return None
        self.values[key] = value
        return True

    async def eval(self, script, numkeys, key, token, *args):
        if self.values.get(key) != token:
            return 0
        if "'del'" in script:
            del self.values[key]
        return 1

    async def hset(self, name, key, value):
        self.hashes.setdefault(name, {})[key] = value
        return 1

    async def hgetall(self, name):
        return dict(self.hashes.get(name, {}))

    async def hkeys(self, name):
        return list(self.hashes.get(name, {}))

    async def hdel(self, name, *keys):
        bucket = self.hashes.get(name, {})
        return sum(1 for key in keys if bucket.pop(key, None) is not None)


def _socket_layer(monkeypatch, **attrs):
    sio = Mock()
    sio.on = Mock()
    sio.enter_room = AsyncMock()
    sio.leave_room = AsyncMock()
    sio.emit = AsyncMock()
    sio.get_session = AsyncMock(return_value={"user": {"id": "admin-1"}})
    sio.save_session = AsyncMock()
    main = types.ModuleType("open_webui.socket.main")
    for key, value in {"sio": sio, "get_session_ids_from_room": lambda room: [], **attrs}.items():
        setattr(main, key, value)
    monkeypatch.setitem(sys.modules, "open_webui.socket", types.ModuleType("open_webui.socket"))
    monkeypatch.setitem(sys.modules, "open_webui.socket.main", main)
    return sio


def _running_pipe() -> Pipe:
    pipe = Pipe()
    pipe.valves.ENABLE_PLUGIN_SYSTEM = True
    pipe._ensure_plugin_registry()
    return pipe


async def _closed(pipe: Pipe) -> None:
    for _ in range(300):
        if pipe._closed:
            return
        await _REAL_SLEEP(0.01)
    raise AssertionError("the replaced copy never closed")


def _panel_body(pipe: Pipe, request: dict) -> dict:
    return {
        "model": f"{pipe.id}.pipe-dashboard",
        "stream": False,
        "messages": [{"role": "user", "content": json.dumps({"pipe_dashboard_panel": request})}],
    }


async def _panel_reply(pipe: Pipe, request: dict, metadata: dict | None = None) -> dict:
    reply = await pipe.pipe(
        _panel_body(pipe, request), {"id": "admin-1", "role": "admin"}, None, None, None, metadata or {}, None
    )
    assert isinstance(reply, dict), f"the panel's request was not answered by the dashboard: {reply!r}"
    return json.loads(reply["choices"][0]["message"]["content"])


def _open_webui_hands_out(monkeypatch, pipe: Pipe) -> None:
    loader = types.ModuleType("open_webui.functions")

    async def get_function_module_by_id(request, pipe_id):
        return pipe

    setattr(loader, "get_function_module_by_id", get_function_module_by_id)
    monkeypatch.setitem(sys.modules, "open_webui.functions", loader)
    monkeypatch.setattr(http_routes, "get_owui_app", lambda: SimpleNamespace(state=SimpleNamespace()))


def _dispatcher_reporting_who_it_served(monkeypatch, served_copy: Pipe | None = None) -> list:
    ran: list = []

    async def dispatch(pipe, user, name, args, **kwargs):
        ran.append(name)
        result = {"action": name, "user": user.id}
        if served_copy is not None:
            result["served_by_the_new_copy"] = pipe is served_copy
        return 200, {"ok": True, "result": result}

    monkeypatch.setattr(http_routes, "_live_dispatch", lambda: dispatch)
    http_routes._coarse_state.clear()
    return ran


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["sign-up", "tab click"])
async def test_after_a_pipe_update_the_worker_that_loaded_the_new_copy_still_answers_the_dashboard(
    entry, monkeypatch
):
    sio = _socket_layer(monkeypatch, get_user_id_from_session_pool=lambda sid: "admin-1")
    monkeypatch.setattr(dashboard_socket, "resolve_user", AsyncMock(return_value=_ADMIN))
    monkeypatch.setattr(dashboard_socket, "can_view", AsyncMock(return_value=True))
    old = _running_pipe()
    new = Pipe()
    new.valves.ENABLE_PLUGIN_SYSTEM = True
    try:
        await _closed(old)
        _open_webui_hands_out(monkeypatch, new)
        if entry == "sign-up":
            await dashboard_socket._pipe_dashboard_sub("sid-1", {"pipe": new.id})
            sio.enter_room.assert_any_await("sid-1", dashboard_socket.viewers_room(new.id))
            assert not [
                call for call in sio.emit.await_args_list
                if call.args[:1] == (dashboard_socket.DENIED_EVENT,)
            ]
        else:
            install_open_webui_auth(monkeypatch, user=_ADMIN)
            _dispatcher_reporting_who_it_served(monkeypatch, served_copy=new)
            app = FastAPI()
            app.add_api_route(
                http_routes._ACTION_PATH, http_routes._action_route, methods=["POST"],
                dependencies=http_routes._action_dependencies(),
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://owui.example.com"
            ) as client:
                response = await client.post(
                    http_routes._ACTION_PATH,
                    json={"action": "whoami", "args": {}, "pipe": new.id},
                    headers={"Authorization": f"Bearer {token_for('admin-1')}"},
                )
            assert (response.status_code, response.json()) == (
                200,
                {"ok": True, "result": {"action": "whoami", "user": "admin-1", "served_by_the_new_copy": True}},
            )
    finally:
        await new.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kind", ["sign-up", "tab click"])
async def test_a_panel_on_any_worker_is_answered_through_open_webuis_chat_route_to_the_dashboard_model(
    request_kind, monkeypatch
):
    sio = _socket_layer(
        monkeypatch, get_user_id_from_session_pool=lambda sid: "admin-1" if sid == "sid-elsewhere" else None
    )
    monkeypatch.setattr(dashboard_socket, "resolve_user", AsyncMock(return_value=_ADMIN))
    monkeypatch.setattr(dashboard_socket, "can_view", AsyncMock(return_value=True))
    cluster = importlib.import_module("open_webui_openrouter_pipe.plugins.pipe_dashboard.cluster")
    monkeypatch.setattr(cluster, "resolve_user", AsyncMock(return_value=_ADMIN))
    pipe = _running_pipe()
    try:
        if request_kind == "sign-up":
            reply = await _panel_reply(pipe, {"op": "sub", "sid": "sid-elsewhere"})
            assert reply == {"ok": True}
            sio.enter_room.assert_any_await("sid-elsewhere", dashboard_socket.viewers_room(pipe.id))
        else:
            _dispatcher_reporting_who_it_served(monkeypatch)
            reply = await _panel_reply(pipe, {"op": "act", "sid": "sid-elsewhere", "action": "whoami", "args": {}})
            assert reply == {"status": 200, "body": {"ok": True, "result": {"action": "whoami", "user": "admin-1"}}}
    finally:
        await pipe.close()


@pytest.mark.asyncio
async def test_a_panel_request_is_answered_while_the_callers_chats_are_refused_after_repeated_errors(monkeypatch):
    _socket_layer(monkeypatch, get_user_id_from_session_pool=lambda sid: "admin-1" if sid == "sid-1" else None)
    cluster = importlib.import_module("open_webui_openrouter_pipe.plugins.pipe_dashboard.cluster")
    monkeypatch.setattr(cluster, "resolve_user", AsyncMock(return_value=_ADMIN))
    _dispatcher_reporting_who_it_served(monkeypatch)
    pipe = _running_pipe()
    try:
        for _ in range(pipe.valves.BREAKER_MAX_FAILURES + 1):
            pipe._circuit_breaker.record_failure("admin-1")
        assert not pipe._circuit_breaker.allows("admin-1")
        reply = await _panel_reply(pipe, {"op": "act", "sid": "sid-1", "action": "whoami", "args": {}})
        assert reply == {"status": 200, "body": {"ok": True, "result": {"action": "whoami", "user": "admin-1"}}}
    finally:
        await pipe.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("sid", [None, "sid-of-someone-else"])
async def test_a_panel_action_runs_only_with_the_callers_own_live_socket(sid, monkeypatch):
    _socket_layer(monkeypatch, get_user_id_from_session_pool=lambda s: "someone-else" if s else None)
    cluster = importlib.import_module("open_webui_openrouter_pipe.plugins.pipe_dashboard.cluster")
    monkeypatch.setattr(cluster, "resolve_user", AsyncMock(return_value=_ADMIN))
    ran = _dispatcher_reporting_who_it_served(monkeypatch)
    pipe = _running_pipe()
    try:
        request = {"op": "act", "action": "config_get", "args": {}}
        if sid is not None:
            request["sid"] = sid
        reply = await _panel_reply(pipe, request)
        assert reply["status"] == 403, reply
        assert ran == [], f"an action ran without the caller's live dashboard socket: {ran}"
    finally:
        await pipe.close()


@pytest.mark.asyncio
async def test_a_panel_request_sent_within_a_chat_runs_nothing(monkeypatch):
    _socket_layer(monkeypatch, get_user_id_from_session_pool=lambda sid: "admin-1")
    cluster = importlib.import_module("open_webui_openrouter_pipe.plugins.pipe_dashboard.cluster")
    monkeypatch.setattr(cluster, "resolve_user", AsyncMock(return_value=_ADMIN))
    ran = _dispatcher_reporting_who_it_served(monkeypatch)
    pipe = _running_pipe()
    try:
        await pipe.pipe(
            _panel_body(pipe, {"op": "act", "sid": "sid-1", "action": "config_get", "args": {}}),
            {"id": "admin-1", "role": "admin"}, None, None, None, {"chat_id": "chat-1", "message_id": "msg-1"}, None,
        )
        assert ran == [], f"a panel request inside a chat ran an action: {ran}"
    finally:
        await pipe.close()


@pytest.mark.asyncio
async def test_a_panel_sign_up_naming_another_users_socket_is_refused(monkeypatch):
    sio = _socket_layer(monkeypatch, get_user_id_from_session_pool=lambda sid: "someone-else")
    monkeypatch.setattr(dashboard_socket, "resolve_user", AsyncMock(return_value=_ADMIN))
    monkeypatch.setattr(dashboard_socket, "can_view", AsyncMock(return_value=True))
    pipe = _running_pipe()
    try:
        reply = await _panel_reply(pipe, {"op": "sub", "sid": "sid-of-someone-else"})
        assert reply == {"ok": False}
        sio.enter_room.assert_not_awaited()
    finally:
        await pipe.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lock", ["free", "held by another worker"])
async def test_with_open_webuis_redis_only_the_worker_holding_the_dashboard_lock_sends_live_numbers_to_viewers_on_any_worker(
    lock, monkeypatch
):
    redis = _OpenWebUIRedis()
    sio = _socket_layer(
        monkeypatch,
        REDIS=redis,
        get_user_id_from_session_pool=lambda sid: "admin-1" if sid == "sid-elsewhere" else None,
    )
    monkeypatch.setattr(dashboard_socket, "resolve_user", AsyncMock(return_value=_ADMIN))
    monkeypatch.setattr(dashboard_socket, "can_view", AsyncMock(return_value=True))
    monkeypatch.setattr(dashboard_socket, "can_view_known", AsyncMock(return_value=True))
    pipe = SimpleNamespace(id="test-pipe", valves=SimpleNamespace(ENABLE_PLUGIN_SYSTEM=True))
    arm_dashboard_pipe(pipe)
    cluster = importlib.import_module("open_webui_openrouter_pipe.plugins.pipe_dashboard.cluster")
    assert await dashboard_socket.admit_viewer("sid-elsewhere", pipe.id, "admin-1")
    if lock == "held by another worker":
        assert await cluster.PublisherLock(pipe.id).hold(redis)

    async def instant(delay):
        await _REAL_SLEEP(0)

    monkeypatch.setattr(dashboard_publisher.asyncio, "sleep", instant)
    monkeypatch.setattr(dashboard_publisher, "_build_emit_payload", AsyncMock(return_value={"tick": 1}))
    sio.emit.reset_mock()
    task = asyncio.get_running_loop().create_task(
        dashboard_publisher.run_dashboard_publisher(lambda: pipe, lambda: (None, False), "testns")
    )
    try:
        for _ in range(50):
            await _REAL_SLEEP(0.01)
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    live = [call for call in sio.emit.await_args_list if call.args[:1] == (dashboard_socket.DASHBOARD_EVENT,)]
    if lock == "free":
        assert live, "the viewer on the other worker got no live numbers"
        assert all(
            call.kwargs.get("room") == dashboard_socket.viewers_room(pipe.id)
            and not call.kwargs.get("ignore_queue", False)
            for call in live
        ), live
    else:
        assert live == []
