from __future__ import annotations

import asyncio
import contextlib
import importlib
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
from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import handle_dashboard
from open_webui_openrouter_pipe.plugins.pipe_dashboard.context import CommandContext
from tests._owui_auth_stub import install_open_webui_auth, token_for
from conftest import arm_dashboard_pipe

_LIFECYCLE_KEY = "_openrouter_pipe_lifecycle"
_ADMIN = SimpleNamespace(id="admin-1", role="admin", email="admin-1@example.com")
_REPLY_EVENT = "openrouter:pipe_dashboard:reply"
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


async def _replied(sio: Mock, rid: str) -> None:
    for _ in range(300):
        if any(
            call.args[:1] == (_REPLY_EVENT,) and (call.args[1] or {}).get("rid") == rid
            for call in sio.emit.await_args_list
        ):
            return
        await _REAL_SLEEP(0.01)
    raise AssertionError(f"no answer reached the viewer's socket for request {rid}")


def _open_webui_hands_out(monkeypatch, pipe: Pipe) -> None:
    loader = types.ModuleType("open_webui.functions")

    async def get_function_module_by_id(request, pipe_id):
        return pipe

    setattr(loader, "get_function_module_by_id", get_function_module_by_id)
    monkeypatch.setitem(sys.modules, "open_webui.functions", loader)
    monkeypatch.setattr(http_routes, "get_owui_app", lambda: SimpleNamespace(state=SimpleNamespace()))


def _dispatcher_reporting_who_it_served(monkeypatch, served_copy: Pipe | None = None) -> None:
    async def dispatch(pipe, user, name, args, **kwargs):
        result = {"action": name, "user": user.id}
        if served_copy is not None:
            result["served_by_the_new_copy"] = pipe is served_copy
        return 200, {"ok": True, "result": result}

    monkeypatch.setattr(http_routes, "_live_dispatch", lambda: dispatch)
    http_routes._coarse_state.clear()


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
async def test_a_viewer_whose_socket_is_on_a_worker_without_the_pipe_is_answered_through_open_webuis_relay(
    request_kind, monkeypatch
):
    queues: dict = {}
    sio = _socket_layer(
        monkeypatch,
        EVENT_QUEUES=queues,
        REDIS=None,
        get_user_id_from_session_pool=lambda sid: "admin-1" if sid == "sid-elsewhere" else None,
    )
    monkeypatch.setattr(dashboard_socket, "resolve_user", AsyncMock(return_value=_ADMIN))
    monkeypatch.setattr(dashboard_socket, "can_view", AsyncMock(return_value=True))
    pipe = _running_pipe()
    try:
        await handle_dashboard(
            CommandContext(
                pipe=pipe, args="", user={"id": "admin-1", "role": "admin"},
                metadata={}, event_emitter=AsyncMock(),
            )
        )
        channel = f"admin-1:pipe_dashboard:{pipe.id}"
        assert channel in queues, "Open WebUI's relay drops an event nobody registered a queue for"
        cluster = importlib.import_module("open_webui_openrouter_pipe.plugins.pipe_dashboard.cluster")
        monkeypatch.setattr(cluster, "resolve_user", AsyncMock(return_value=_ADMIN))
        if request_kind == "sign-up":
            await queues[channel].put({"op": "sub", "rid": "r1", "sid": "sid-elsewhere"})
            await _replied(sio, "r1")
            sio.enter_room.assert_any_await("sid-elsewhere", dashboard_socket.viewers_room(pipe.id))
            sio.emit.assert_any_await(_REPLY_EVENT, {"rid": "r1", "ok": True}, to="sid-elsewhere")
        else:
            _dispatcher_reporting_who_it_served(monkeypatch)
            await queues[channel].put(
                {"op": "act", "rid": "r2", "sid": "sid-elsewhere", "action": "whoami", "args": {}}
            )
            await _replied(sio, "r2")
            sio.emit.assert_any_await(
                _REPLY_EVENT,
                {"rid": "r2", "status": 200, "body": {"ok": True, "result": {"action": "whoami", "user": "admin-1"}}},
                to="sid-elsewhere",
            )
    finally:
        await pipe.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("lock", ["free", "held by another worker"])
async def test_with_open_webuis_redis_only_the_worker_holding_the_dashboard_lock_sends_live_numbers_to_viewers_on_any_worker(
    lock, monkeypatch
):
    redis = _OpenWebUIRedis()
    queues: dict = {}
    sio = _socket_layer(
        monkeypatch,
        EVENT_QUEUES=queues,
        REDIS=redis,
        get_user_id_from_session_pool=lambda sid: "admin-1" if sid == "sid-elsewhere" else None,
    )
    monkeypatch.setattr(dashboard_socket, "resolve_user", AsyncMock(return_value=_ADMIN))
    monkeypatch.setattr(dashboard_socket, "can_view", AsyncMock(return_value=True))
    monkeypatch.setattr(dashboard_socket, "can_view_known", AsyncMock(return_value=True))
    pipe = SimpleNamespace(id="test-pipe", valves=SimpleNamespace(ENABLE_PLUGIN_SYSTEM=True))
    arm_dashboard_pipe(pipe)
    cluster = importlib.import_module("open_webui_openrouter_pipe.plugins.pipe_dashboard.cluster")
    await cluster.remember_dashboard_user(pipe.id, "admin-1")
    await queues[f"admin-1:pipe_dashboard:{pipe.id}"].put({"op": "sub", "rid": "r1", "sid": "sid-elsewhere"})
    await _replied(sio, "r1")
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
