"""Tests for the pipe_dashboard OWUI socket.io integration and publisher emit loop."""

from __future__ import annotations

import ast
import asyncio
import concurrent.futures
import contextlib
import logging
import os
import pathlib
import sys
import time
import types
from unittest.mock import AsyncMock, Mock, patch

import pytest
from open_webui.models import models as models_module

pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

_READ_CONFIG_REV_HARNESS = """
from __future__ import annotations

import asyncio, contextlib, os, sys, types
from typing import Any

os.environ.setdefault("WEBUI_SECRET_KEY", "probe")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))

import owui_stubs  # noqa: F401

import pydantic
from sqlalchemy import BigInteger, Boolean, Column, String, Text, event, select
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import declarative_base

import open_webui.models.functions as functions_module
from open_webui_openrouter_pipe.plugins.pipe_dashboard import dashboard_socket as ds

emitted: list[tuple[str, Any]] = []

# `owui_stubs` stops at `open_webui.models.functions`; `open_webui.internal.db` is a
# real module of a real install and absent here, so the harness supplies the one name
# the read closes over. `config_service._raw_valve_column` binds the same pair and is
# covered the same way, so this is the suite's established substitute, not a new one.
_internal = types.ModuleType("open_webui.internal")
_internal.__path__ = []
_db_module = types.ModuleType("open_webui.internal.db")
sys.modules["open_webui.internal"] = _internal
sys.modules["open_webui.internal.db"] = _db_module

Base = declarative_base()


class Function(Base):
    __tablename__ = "function"
    id = Column(String, primary_key=True, unique=True)
    user_id = Column(String, index=True)
    name = Column(Text, nullable=False)
    type = Column(Text, nullable=False)
    content = Column(Text, nullable=True)
    meta = Column(Text, nullable=True)
    valves = Column(Text, nullable=True)
    is_active = Column(Boolean, default=False)
    is_global = Column(Boolean)
    updated_at = Column(BigInteger)
    created_at = Column(BigInteger)


class _FunctionModel(pydantic.BaseModel):
    # Open WebUI's own `FunctionModel`, field for field. `model_validate(row)` reads
    # every column of the row it is handed, which is the cost this item is about: a
    # plain `select(Function)` cannot answer "one integer" without copying the source.
    id: str
    user_id: str | None = None
    name: str
    type: str
    content: str
    is_active: bool = False
    is_global: bool = False
    updated_at: int
    created_at: int

    model_config = pydantic.ConfigDict(from_attributes=True)


async def _build(rows, *, content="# source\\n" + "y" * 200_000):
    path = os.path.join(os.environ["ORPIPE_PROBE_TMPDIR"], "function.db")
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with engine.begin() as conn:
        await conn.execute(
            Function.__table__.insert(),
            [
                {
                    "id": row_id, "user_id": "u", "name": row_id, "type": "pipe",
                    "content": content, "is_active": True, "is_global": False,
                    "updated_at": rev, "created_at": 1,
                }
                for row_id, rev in rows
            ],
        )

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _record(conn, cursor, statement, parameters, context, executemany):
        emitted.append((statement, parameters))

    @contextlib.asynccontextmanager
    async def _ctx(*_a, **_k):
        async with engine.connect() as conn:
            yield conn

    class _Functions:
        # Open WebUI's own accessor, for whichever seam the read under test reaches.
        # Without it the pre-fix body would find the stub no-op returning None, and
        # every arm would go red on the same absent row instead of on the read's shape.
        # It answers exactly what Open WebUI's returns: a mapped instance, validated
        # through FunctionModel -- which reads every column, content included.

        @staticmethod
        async def get_function_by_id(id, db=None):
            async with engine.connect() as conn:
                result = await conn.execute(select(Function).filter_by(id=id))
                return _FunctionModel.model_validate(result.first())

    _db_module.get_async_db_context = _ctx
    functions_module.Function = Function
    functions_module.Functions = _Functions
    return engine


@contextlib.asynccontextmanager
async def _raising():
    @contextlib.asynccontextmanager
    async def _ctx(*_a, **_k):
        raise RuntimeError("(sqlite3.OperationalError) database is locked")
        yield

    _db_module.get_async_db_context = _ctx
    yield


def _projection(statement: str) -> str:
    # The column list of a SELECT, lowercased and whitespace-collapsed. Only the part
    # before FROM counts: a `WHERE function.id = ?` legitimately names the id, and
    # reading that as "the id was selected" would pass a wide read.
    head = statement.lower().split("from")[0]
    return " ".join(head.replace("select", "").split()).strip()


DEFAULT_ROWS = [("openrouter", 4242), ("other", 9999)]
"""

from open_webui_openrouter_pipe.plugins.pipe_dashboard import authz, dashboard_publisher, dashboard_socket
from open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_publisher import (
    _build_emit_payload,
    run_dashboard_publisher,
)
from open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_socket import (
    CONFIG_EVENT,
    DASHBOARD_EVENT,
    DENIED_EVENT,
    SUB_EVENT,
    VIEWERS_ROOM,
    _pipe_dashboard_sub,
    emit_config_changed,
    register_socket_handler,
)
from tests.pipe_limits import set_slot, slot

_REAL_SLEEP = asyncio.sleep

_PID = os.getpid()
_FOREIGN_PIDS = (_PID + 1_000_003, _PID + 1_000_033)
_FOREIGN_PIDS_C = (_PID + 2_000_003, _PID + 2_000_033)
# Below the local pid on purpose. `workers` is sorted ascending, so a remote that
# sorts below is `workers[0]`: "the first worker the scan returned" and "the worker
# holding the viewer's socket" are then different numbers, which is the only shape
# in which a build that answers with the former is caught.
_FOREIGN_PIDS_BELOW = (_PID - 3_000_017, _PID - 3_000_029)
_A, _B = _FOREIGN_PIDS
_C, _D = _FOREIGN_PIDS_C
_E, _F = _FOREIGN_PIDS_BELOW


def _install_socket_stub(monkeypatch, **attrs):
    """Install a fake open_webui.socket.main module for the lazy imports."""
    socket_pkg = types.ModuleType("open_webui.socket")
    main_mod = types.ModuleType("open_webui.socket.main")
    for key, value in attrs.items():
        setattr(main_mod, key, value)
    monkeypatch.setitem(sys.modules, "open_webui.socket", socket_pkg)
    monkeypatch.setitem(sys.modules, "open_webui.socket.main", main_mod)
    return main_mod


def _make_mock_pipe():
    pipe = Mock()
    pipe.id = "test-pipe"
    set_slot(pipe, "request_semaphore", Mock())
    slot(pipe, "request_semaphore")._value = 45
    set_slot(pipe, "request_limit", 50)
    set_slot(pipe, "tool_semaphore", Mock())
    slot(pipe, "tool_semaphore")._value = 8
    set_slot(pipe, "tool_limit", 10)
    pipe._request_queue = Mock()
    pipe._request_queue.qsize.return_value = 3
    pipe._QUEUE_MAXSIZE = 1000
    pipe._log_queue = Mock()
    pipe._log_queue.qsize.return_value = 7
    pipe._session_log_manager = Mock()
    pipe._session_log_manager._worker_thread = Mock()
    pipe._session_log_manager._worker_thread.is_alive.return_value = True
    pipe._session_log_manager._queue = Mock()
    pipe._session_log_manager._queue.qsize.return_value = 2
    pipe._session_log_manager._retention_days = 30
    pipe._circuit_breaker = Mock()
    pipe._circuit_breaker._threshold = 5
    pipe._circuit_breaker._window_seconds = 60.0
    pipe._circuit_breaker._breaker_records = {}
    pipe._circuit_breaker._tool_breakers = {}
    pipe._initialized = True
    pipe._startup_checks_complete = True
    pipe._warmup_failed = False
    pipe._http_session = Mock()
    pipe._http_session.closed = False
    pipe._redis_enabled = False
    pipe._redis_client = None
    pipe.valves = Mock()
    pipe.valves.SESSION_LOG_STORE_ENABLED = True
    pipe.valves.DEFAULT_LLM_ENDPOINT = "https://openrouter.ai/api/v1"
    pipe.valves.BREAKER_MAX_FAILURES = 5
    pipe.valves.BREAKER_WINDOW_SECONDS = 60
    pipe.valves.ENABLE_TIMING_LOG = False
    pipe.valves.ARTIFACT_CLEANUP_INTERVAL_HOURS = 24
    pipe.valves.ARTIFACT_CLEANUP_DAYS = 30
    pipe.valves.SESSION_LOG_RETENTION_DAYS = 14
    pipe.valves.REDIS_CACHE_TTL_SECONDS = 300
    pipe.valves.STREAMING_IDLE_FLUSH_MS = 100
    pipe._artifact_store = None
    pipe._plugin_registry = None
    pipe._active_pipes_calls = 0
    set_slot(pipe, "video_semaphore", None)
    set_slot(pipe, "video_limit", 0)
    pipe._video_active_tasks = {}
    return pipe


# ── Subscribe handler ──


class TestPipeDashboardSub:
    @pytest.mark.asyncio
    async def test_subscribe_no_user_denied(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.enter_room = AsyncMock()
        mock_sio.leave_room = AsyncMock()
        mock_sio.emit = AsyncMock()
        mock_sio.get_session = AsyncMock(side_effect=KeyError("Session not found"))
        dashboard_socket._resync = False
        _install_socket_stub(
            monkeypatch, sio=mock_sio, get_user_id_from_session_pool=lambda sid: None,
        )
        dashboard_socket._get_pipe = _plugin_on_pipe
        await _pipe_dashboard_sub("sid-anon")
        mock_sio.enter_room.assert_not_awaited()
        assert dashboard_socket._resync is False

    @pytest.mark.asyncio
    async def test_subscribe_denied_emits_denied(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.enter_room = AsyncMock()
        mock_sio.leave_room = AsyncMock()
        mock_sio.emit = AsyncMock()
        mock_sio.get_session = AsyncMock(return_value={"user": {"id": "user-1"}})
        dashboard_socket._resync = False
        _install_socket_stub(
            monkeypatch, sio=mock_sio, get_user_id_from_session_pool=lambda sid: "user-1",
        )
        monkeypatch.setattr(dashboard_socket,"resolve_user", AsyncMock(return_value=object()))
        monkeypatch.setattr(dashboard_socket,"can_view", AsyncMock(return_value=False))
        dashboard_socket._get_pipe = _plugin_on_pipe
        await _pipe_dashboard_sub("sid-denied")
        mock_sio.enter_room.assert_not_awaited()
        mock_sio.leave_room.assert_awaited_with("sid-denied", VIEWERS_ROOM)
        mock_sio.emit.assert_awaited_once_with(dashboard_socket.DENIED_EVENT, {}, room="sid-denied")
        assert dashboard_socket._resync is False

    @pytest.mark.asyncio
    async def test_subscribe_granted_joins_room(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.enter_room = AsyncMock()
        mock_sio.emit = AsyncMock()
        mock_sio.get_session = AsyncMock(return_value={"user": {"id": "user-1"}})
        dashboard_socket._resync = False
        _install_socket_stub(
            monkeypatch, sio=mock_sio, get_user_id_from_session_pool=lambda sid: "user-1",
        )
        fake_resolve_user = AsyncMock(return_value=object())
        monkeypatch.setattr(dashboard_socket, "resolve_user", fake_resolve_user)
        monkeypatch.setattr(dashboard_socket,"can_view", AsyncMock(return_value=True))
        dashboard_socket._get_pipe = _plugin_on_pipe
        await _pipe_dashboard_sub("sid-authed")
        mock_sio.enter_room.assert_awaited_once_with("sid-authed", VIEWERS_ROOM)
        assert dashboard_socket._resync is True
        # The resolver is async; a missing await hands resolve_user a coroutine, which
        # denies every subscriber in production while leaving argument-blind stubs green.
        fake_resolve_user.assert_awaited_once_with("user-1")

    @pytest.mark.asyncio
    async def test_enter_room_failure_no_resync(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.enter_room = AsyncMock(side_effect=RuntimeError("boom"))
        mock_sio.emit = AsyncMock()
        mock_sio.get_session = AsyncMock(return_value={"user": {"id": "user-1"}})
        dashboard_socket._resync = False
        _install_socket_stub(
            monkeypatch, sio=mock_sio, get_user_id_from_session_pool=lambda sid: "user-1",
        )
        monkeypatch.setattr(dashboard_socket,"resolve_user", AsyncMock(return_value=object()))
        monkeypatch.setattr(dashboard_socket,"can_view", AsyncMock(return_value=True))
        dashboard_socket._get_pipe = _plugin_on_pipe
        await _pipe_dashboard_sub("sid-err")
        assert dashboard_socket._resync is False

class _ValveStore:
    def __init__(self, row):
        self.row = row

    async def get_function_by_id(self, id, db=None):
        return types.SimpleNamespace(updated_at=1000)

    async def get_function_valves_by_id(self, id, db=None):
        return dict(self.row)

    async def update_function_valves_by_id(self, id, valves, db=None):
        self.row = dict(valves)
        return types.SimpleNamespace(updated_at=1001)


@pytest.fixture(autouse=True)
def _persisted_master_switch_on(monkeypatch):
    """The master switch as the operator's own save left it in the persisted row.

    A deployment whose dashboard answers has `ENABLE_PLUGIN_SYSTEM` in that row: the
    field's declared default is off, so writing it on stores the key. A readable row
    that omits it therefore means off, which is the switch the Config-tab writer
    leaves behind. Without this row every 'the dashboard is on' arm below would be
    refused by the master switch rather than by the thing it is testing, and the
    tests that install their own store afterwards (monkeypatch runs after this
    fixture) still say which row they mean.
    """
    import open_webui.models.functions as functions_mod

    monkeypatch.setattr(
        functions_mod, "Functions", _ValveStore({"ENABLE_PLUGIN_SYSTEM": True})
    )


def _plugin_on_pipe():
    return types.SimpleNamespace(
        id="test-pipe",
        valves=types.SimpleNamespace(ENABLE_PLUGIN_SYSTEM=True),
    )


class TestReauthorizeLocalViewers:
    @pytest.mark.asyncio
    async def test_evicts_ungranted(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.leave_room = AsyncMock()
        mock_sio.emit = AsyncMock()
        mock_sio.get_session = AsyncMock(return_value={"user": {"id": "user-1"}})
        _install_socket_stub(
            monkeypatch, sio=mock_sio,
            get_session_ids_from_room=lambda room: ["s1"],
            get_user_id_from_session_pool=lambda sid: "user-1",
        )
        monkeypatch.setattr(dashboard_socket,"resolve_user", AsyncMock(return_value=object()))
        monkeypatch.setattr(dashboard_socket,"can_view_known", AsyncMock(return_value=False))
        dashboard_socket._get_pipe = _plugin_on_pipe
        await dashboard_socket.reauthorize_local_viewers()
        mock_sio.leave_room.assert_awaited_once_with("s1", VIEWERS_ROOM)
        mock_sio.emit.assert_awaited_once_with(dashboard_socket.DENIED_EVENT, {}, room="s1")

    @pytest.mark.asyncio
    async def test_keeps_granted(self, monkeypatch, caplog):
        mock_sio = Mock()
        mock_sio.leave_room = AsyncMock()
        mock_sio.emit = AsyncMock()
        mock_sio.get_session = AsyncMock(side_effect=KeyError("Session not found"))
        _install_socket_stub(
            monkeypatch, sio=mock_sio,
            get_session_ids_from_room=lambda room: ["s1"],
            get_user_id_from_session_pool=lambda sid: "user-1",
        )
        fake_resolve_user = AsyncMock(return_value=object())
        monkeypatch.setattr(dashboard_socket, "resolve_user", fake_resolve_user)
        monkeypatch.setattr(dashboard_socket,"can_view_known", AsyncMock(return_value=True))
        dashboard_socket._get_pipe = _plugin_on_pipe
        with caplog.at_level(logging.WARNING, logger=dashboard_socket.__name__):
            await dashboard_socket.reauthorize_local_viewers()
        mock_sio.leave_room.assert_not_awaited()
        fake_resolve_user.assert_awaited_once_with("user-1")
        # Defect 5's other half: an eviction warning that also fires for authorized
        # viewers is worse than none, so pin the negative case beside the positive.
        assert not [r for r in caplog.records if r.levelno >= logging.WARNING]

    @pytest.mark.asyncio
    async def test_import_failure_safe(self, monkeypatch):
        _install_socket_stub(monkeypatch)
        await dashboard_socket.reauthorize_local_viewers()

    @pytest.mark.asyncio
    async def test_eviction_is_logged(self, monkeypatch, caplog):
        mock_sio = Mock()
        mock_sio.leave_room = AsyncMock()
        mock_sio.emit = AsyncMock()
        mock_sio.get_session = AsyncMock(return_value={"user": {"id": "user-1"}})
        _install_socket_stub(
            monkeypatch, sio=mock_sio,
            get_session_ids_from_room=lambda room: ["s1"],
            get_user_id_from_session_pool=lambda sid: "user-1",
        )
        monkeypatch.setattr(dashboard_socket, "resolve_user", AsyncMock(return_value=object()))
        monkeypatch.setattr(dashboard_socket, "can_view_known", AsyncMock(return_value=False))
        dashboard_socket._get_pipe = _plugin_on_pipe
        with caplog.at_level(logging.WARNING, logger=dashboard_socket.__name__):
            await dashboard_socket.reauthorize_local_viewers()
        warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("evicting viewer" in m and "s1" in m for m in warnings), warnings

class TestViewerIdentityOutlivesTheSessionPool:
    """The identity a viewer is judged by must outlive OWUI's heartbeat bookkeeping.

    OWUI identifies a socket through ``SESSION_POOL``, which its reaper deletes after
    ``SESSION_POOL_TIMEOUT`` seconds without a heartbeat -- while the socket is still
    connected and still in the viewers room. Reading only that store made a live admin
    indistinguishable from an anonymous socket, and the sweep evicted them for it.

    The durable store is the socket.io session, which belongs to python-socketio rather
    than to Open WebUI. That matters: Open WebUI 0.10.2 never writes it, so a fix that
    only *read* it would be dead code on that deployment. This plugin writes its own
    namespaced key at admission, so the read has a paired writer on every version.
    """

    @staticmethod
    def _stub(monkeypatch, *, session, pool):
        mock_sio = Mock()
        mock_sio.get_session = AsyncMock(**session)
        mock_sio.save_session = AsyncMock()
        _install_socket_stub(monkeypatch, sio=mock_sio, get_user_id_from_session_pool=pool)
        return mock_sio

    @pytest.mark.asyncio
    async def test_pinned_id_wins_over_the_pool(self, monkeypatch):
        self._stub(monkeypatch,
                   session={"return_value": {authz.VIEWER_ID_KEY: "u-sess"}},
                   pool=lambda sid: "u-pool")
        assert await authz.resolve_socket_user_id("sid-1") == "u-sess"

    @pytest.mark.asyncio
    async def test_reaped_pool_still_resolves(self, monkeypatch):
        """The reported bug: pool reaped at 120s while the socket is still connected."""
        self._stub(monkeypatch,
                   session={"return_value": {authz.VIEWER_ID_KEY: "u-sess"}},
                   pool=lambda sid: None)
        assert await authz.resolve_socket_user_id("sid-1") == "u-sess"

    @pytest.mark.asyncio
    async def test_pool_answers_before_anything_is_pinned(self, monkeypatch):
        """Pins the two lookups in SEPARATE try blocks.

        ``get_session`` raises ``KeyError`` for an unknown sid. Sharing one try block
        with the pool read swallows the session failure past the fallback.
        """
        self._stub(monkeypatch,
                   session={"side_effect": KeyError("Session not found")},
                   pool=lambda sid: "u-pool")
        assert await authz.resolve_socket_user_id("sid-1") == "u-pool"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("payload", [{}, {"user": {"id": "owui-only"}}, None, "not-a-dict",
                                         {authz.VIEWER_ID_KEY: ""}, {authz.VIEWER_ID_KEY: 42}])
    async def test_unusable_session_falls_through(self, monkeypatch, payload):
        """Including a session holding only OWUI's own key: that is not ours to read."""
        self._stub(monkeypatch, session={"return_value": payload}, pool=lambda sid: "u-pool")
        assert await authz.resolve_socket_user_id("sid-1") == "u-pool"

    @pytest.mark.asyncio
    async def test_both_sources_dead_returns_none(self, monkeypatch):
        def _boom(sid):
            raise RuntimeError("pool down")

        self._stub(monkeypatch, session={"side_effect": KeyError("gone")}, pool=_boom)
        assert await authz.resolve_socket_user_id("sid-1") is None

    @pytest.mark.asyncio
    async def test_redis_outage_on_pool_does_not_lose_identity(self, monkeypatch):
        """In Redis mode ``SESSION_POOL.get`` propagates ConnectionError; the pinned id
        is worker-local memory and is unaffected."""
        def _boom(sid):
            raise ConnectionError("redis down")

        self._stub(monkeypatch,
                   session={"return_value": {authz.VIEWER_ID_KEY: "u-sess"}},
                   pool=_boom)
        assert await authz.resolve_socket_user_id("sid-1") == "u-sess"

    @pytest.mark.asyncio
    async def test_pinning_preserves_whatever_owui_put_there(self, monkeypatch):
        """save_session REPLACES the whole dict, so the write must merge, not clobber."""
        mock_sio = self._stub(monkeypatch,
                              session={"return_value": {"user": {"id": "owui"}}},
                              pool=lambda sid: None)
        assert await authz.remember_socket_user_id("sid-1", "u-1") is True
        mock_sio.save_session.assert_awaited_once_with(
            "sid-1", {"user": {"id": "owui"}, authz.VIEWER_ID_KEY: "u-1"})

    @pytest.mark.asyncio
    async def test_pinning_survives_an_absent_session(self, monkeypatch):
        mock_sio = self._stub(monkeypatch, session={"side_effect": KeyError("new sid")},
                              pool=lambda sid: None)
        assert await authz.remember_socket_user_id("sid-1", "u-1") is True
        mock_sio.save_session.assert_awaited_once_with("sid-1", {authz.VIEWER_ID_KEY: "u-1"})

    @pytest.mark.asyncio
    async def test_pinning_reports_failure_rather_than_raising(self, monkeypatch):
        mock_sio = self._stub(monkeypatch, session={"return_value": {}}, pool=lambda sid: None)
        mock_sio.save_session = AsyncMock(side_effect=RuntimeError("socket gone"))
        assert await authz.remember_socket_user_id("sid-1", "u-1") is False

    @pytest.mark.asyncio
    async def test_admission_pins_the_id_it_authorized(self, monkeypatch):
        """End-to-end through the real subscribe handler: the write must actually happen,
        or the read added above is dead code."""
        mock_sio = Mock()
        mock_sio.enter_room = AsyncMock()
        mock_sio.emit = AsyncMock()
        mock_sio.get_session = AsyncMock(side_effect=KeyError("not yet"))
        mock_sio.save_session = AsyncMock()
        _install_socket_stub(monkeypatch, sio=mock_sio,
                             get_user_id_from_session_pool=lambda sid: "user-1")
        monkeypatch.setattr(dashboard_socket, "resolve_user", AsyncMock(return_value=object()))
        monkeypatch.setattr(dashboard_socket, "can_view", AsyncMock(return_value=True))
        dashboard_socket._get_pipe = _plugin_on_pipe
        await _pipe_dashboard_sub("sid-authed")
        mock_sio.save_session.assert_awaited_once_with(
            "sid-authed", {authz.VIEWER_ID_KEY: "user-1"})

    @pytest.mark.asyncio
    async def test_a_denied_subscriber_is_never_pinned(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.enter_room = AsyncMock()
        mock_sio.leave_room = AsyncMock()
        mock_sio.emit = AsyncMock()
        mock_sio.get_session = AsyncMock(side_effect=KeyError("not yet"))
        mock_sio.save_session = AsyncMock()
        _install_socket_stub(monkeypatch, sio=mock_sio,
                             get_user_id_from_session_pool=lambda sid: "user-1")
        monkeypatch.setattr(dashboard_socket, "resolve_user", AsyncMock(return_value=object()))
        monkeypatch.setattr(dashboard_socket, "can_view", AsyncMock(return_value=False))
        dashboard_socket._get_pipe = _plugin_on_pipe
        await _pipe_dashboard_sub("sid-denied")
        mock_sio.save_session.assert_not_awaited()


class TestRegisterSocketHandler:
    def test_registers_once(self, monkeypatch):
        mock_sio = Mock()
        dashboard_socket._registered = False
        _install_socket_stub(monkeypatch, sio=mock_sio)
        assert register_socket_handler() is True
        assert register_socket_handler() is True
        mock_sio.on.assert_called_once_with(SUB_EVENT, _pipe_dashboard_sub)
        assert dashboard_socket._registered is True

    def test_import_failure_returns_false(self, monkeypatch):
        dashboard_socket._registered = False
        _install_socket_stub(monkeypatch)
        assert register_socket_handler() is False
        assert dashboard_socket._registered is False

    def test_sio_on_failure_returns_false(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.on = Mock(side_effect=RuntimeError("no"))
        dashboard_socket._registered = False
        _install_socket_stub(monkeypatch, sio=mock_sio)
        assert register_socket_handler() is False
        assert dashboard_socket._registered is False


class TestSocketHelpers:
    def test_consume_resync(self):
        dashboard_socket._resync = True
        assert dashboard_socket.consume_resync() is True
        assert dashboard_socket.consume_resync() is False

    def test_local_viewer_sids(self, monkeypatch):
        _install_socket_stub(
            monkeypatch, get_session_ids_from_room=lambda room: ["s1", "s2"],
        )
        assert dashboard_socket.local_viewer_sids() == ["s1", "s2"]

    def test_local_viewer_sids_import_failure_empty(self, monkeypatch):
        _install_socket_stub(monkeypatch)
        assert dashboard_socket.local_viewer_sids() == []

    def test_local_viewer_sids_error_empty(self, monkeypatch):
        def _boom(room):
            raise RuntimeError("x")

        _install_socket_stub(monkeypatch, get_session_ids_from_room=_boom)
        assert dashboard_socket.local_viewer_sids() == []

    @pytest.mark.asyncio
    async def test_emit_dashboard(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.emit = AsyncMock()
        _install_socket_stub(monkeypatch, sio=mock_sio)
        dashboard_socket._get_pipe = _plugin_on_pipe
        ok = await dashboard_socket.emit_dashboard({"tick": 0})
        assert ok is True
        mock_sio.emit.assert_awaited_once_with(
            DASHBOARD_EVENT, {"tick": 0}, room=VIEWERS_ROOM, ignore_queue=True,
        )

    @pytest.mark.asyncio
    async def test_emit_dashboard_failure_returns_false(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.emit = AsyncMock(side_effect=RuntimeError("down"))
        _install_socket_stub(monkeypatch, sio=mock_sio)
        assert await dashboard_socket.emit_dashboard({"tick": 0}) is False

    @pytest.mark.asyncio
    async def test_unavailable_socket_warns_once_not_per_tick(self, monkeypatch, caplog):
        """emit_dashboard runs on every publish tick; a permanent failure must warn once."""
        import logging as _logging

        _install_socket_stub(monkeypatch)
        dashboard_socket._get_pipe = _plugin_on_pipe
        with caplog.at_level(_logging.WARNING, logger=dashboard_socket.__name__):
            assert await dashboard_socket.emit_dashboard({"tick": 0}) is False
            assert await dashboard_socket.emit_dashboard({"tick": 1}) is False
            assert await dashboard_socket.emit_dashboard({"tick": 2}) is False
        dropped = [r for r in caplog.records if "payloads are being dropped" in r.getMessage()]
        assert len(dropped) == 1
        assert dropped[0].exc_info is not None


# ── Emit payload assembly ──


def _redis_with_slices(slices):
    client = Mock()
    client.set = AsyncMock()

    async def scan_iter(match=None, count=None):
        for i in range(len(slices)):
            yield f"ns:dashboard:worker:{i}"

    client.scan_iter = scan_iter
    import json as _json
    client.mget = AsyncMock(return_value=[_json.dumps(s) for s in slices])
    return client


_FOREIGN_HOST = "ffff0000ffff"


def _compact_slice(pid, active=1, host=_FOREIGN_HOST):
    """A slice as a *peer* writes it: another host's tag beside its pid.

    The self-heal recognises its own row by host **and** pid, so a row that stands in
    for this worker's own has to pass the live tag -- `test_a_healthy_read_containing_
    this_pid_is_not_double_counted` is the row that does.
    """
    return {
        "pid": pid,
        "host": host,
        "up": 100.0,
        "c": {"ar": active, "mr": 50, "at": 0, "mt": 10},
        "q": {"rq": 0, "rm": 1000, "lq": 0, "aq": 0},
        "rl": {"tu": 0, "fu": 0, "tr": 0, "th": 5, "ws": 60.0, "tt": 0, "tp": 0, "aa": 0},
        "s": 0,
    }


class TestBuildEmitPayload:
    @pytest.mark.asyncio
    async def test_no_redis_shape(self):
        pipe = _make_mock_pipe()
        payload = await _build_emit_payload(pipe, None, "ns", "wk", 0, {})
        assert payload["tick"] == 0
        assert payload["worker_count"] == 1
        assert payload["concurrency"]["active_requests"] == 5
        assert len(payload["workers"]) == 1
        assert payload["workers"][0]["pid"] == payload["pid"]
        assert "identity" in payload
        assert "models" in payload
        assert "health" in payload
        assert "storage" in payload
        assert "config" in payload
        assert "plugins" in payload

    @pytest.mark.asyncio
    async def test_wall_clock_cadence(self):
        """Medium/slow tiers fire on elapsed wall-clock, not tick modulo."""
        pipe = _make_mock_pipe()
        state: dict = {}
        p0 = await _build_emit_payload(pipe, None, "ns", "wk", 0, state)
        assert "identity" in p0 and "storage" in p0

        p1 = await _build_emit_payload(pipe, None, "ns", "wk", 1, state)
        assert "identity" not in p1 and "storage" not in p1

        state["medium_sent_at"] -= 100.0
        p2 = await _build_emit_payload(pipe, None, "ns", "wk", 2, state)
        assert "identity" in p2 and "models" in p2 and "storage" not in p2

        state["slow_sent_at"] -= 100.0
        p3 = await _build_emit_payload(pipe, None, "ns", "wk", 3, state)
        assert "storage" in p3

    @pytest.mark.asyncio
    async def test_resync_storm_does_not_starve_slow_tier(self):
        """Regression guard: frequent tick resets (viewer resyncs/reconnects)
        must not prevent the slow tier from ever refreshing. Under the old
        tick-modulo scheduling, a non-multiple tick never sent storage."""
        pipe = _make_mock_pipe()
        state: dict = {}
        await _build_emit_payload(pipe, None, "ns", "wk", 0, state)
        state["slow_sent_at"] -= 100.0
        state["at"] -= 100.0
        p = await _build_emit_payload(pipe, None, "ns", "wk", 7, state)
        assert "storage" in p

    @pytest.mark.asyncio
    async def test_cadence_constants_pinned(self):
        """Boundary guard: medium fires at 16s, slow at 60s — just-under does
        not fire, just-over does. Pins the tuning constants."""
        from open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_publisher import (
            _PD_MEDIUM_EVERY,
            _PD_PUBLISH_INTERVAL,
            _PD_SLOW_EVERY,
        )
        medium_interval = _PD_MEDIUM_EVERY * _PD_PUBLISH_INTERVAL
        slow_interval = _PD_SLOW_EVERY * _PD_PUBLISH_INTERVAL
        assert medium_interval == 16.0
        assert slow_interval == 60.0

        pipe = _make_mock_pipe()
        state: dict = {}
        await _build_emit_payload(pipe, None, "ns", "wk", 0, state)

        state["medium_sent_at"] -= medium_interval - 1
        p = await _build_emit_payload(pipe, None, "ns", "wk", 1, state)
        assert "identity" not in p
        state["medium_sent_at"] -= 2
        p = await _build_emit_payload(pipe, None, "ns", "wk", 2, state)
        assert "identity" in p

        state["slow_sent_at"] -= slow_interval - 1
        p = await _build_emit_payload(pipe, None, "ns", "wk", 3, state)
        assert "storage" not in p
        state["slow_sent_at"] -= 2
        p = await _build_emit_payload(pipe, None, "ns", "wk", 4, state)
        assert "storage" in p

    @pytest.mark.asyncio
    async def test_resync_resends_cached_slow_tier(self):
        pipe = _make_mock_pipe()
        state: dict = {}
        await _build_emit_payload(pipe, None, "ns", "wk", 0, state)
        p = await _build_emit_payload(pipe, None, "ns", "wk", 0, state)
        assert "storage" in p and "identity" in p

    @pytest.mark.asyncio
    async def test_redis_aggregation_with_self_inclusion(self):
        import os
        pipe = _make_mock_pipe()
        client = _redis_with_slices([_compact_slice(_C, active=2), _compact_slice(_D, active=3)])
        payload = await _build_emit_payload(pipe, client, "ns", "wk", 1, {})
        assert payload["worker_count"] == 3
        pids = {w["pid"] for w in payload["workers"]}
        assert {_C, _D, os.getpid()} == pids
        assert payload["concurrency"]["active_requests"] == 2 + 3 + 5
        client.set.assert_awaited()

    @pytest.mark.asyncio
    async def test_redis_empty_falls_back_to_local(self):
        pipe = _make_mock_pipe()
        client = Mock()
        client.set = AsyncMock()

        async def scan_iter(match=None, count=None):
            return
            yield

        client.scan_iter = scan_iter
        payload = await _build_emit_payload(pipe, client, "ns", "wk", 1, {})
        assert payload["worker_count"] == 1
        assert payload["concurrency"]["active_requests"] == 5

    @pytest.mark.asyncio
    async def test_redis_blip_uses_cached_workers_and_degrades(self, monkeypatch):
        pipe = _make_mock_pipe()
        client = Mock()
        client.set = AsyncMock()
        clock = {"t": 1000.0}
        monkeypatch.setattr(dashboard_publisher.time, "monotonic", lambda: clock["t"], raising=False)

        async def scan_boom(match=None, count=None):
            raise RuntimeError("redis down")
            yield

        client.scan_iter = scan_boom
        cached = [
            {"pid": _A, "uptime_s": 50.0,
             "concurrency": {"active_requests": 1, "max_requests": 50, "active_tools": 0, "max_tools": 10},
             "queues": {}, "rate_limits": {}, "sessions": {"in_flight": 0}},
            {"pid": _B, "uptime_s": 60.0,
             "concurrency": {"active_requests": 2, "max_requests": 50, "active_tools": 0, "max_tools": 10},
             "queues": {}, "rate_limits": {}, "sessions": {"in_flight": 0}},
        ]
        agg_state = {"workers": list(cached), "misses": 0, "set_at": clock["t"]}
        payload = await _build_emit_payload(pipe, client, "ns", "wk", 1, {}, agg_state)
        assert payload["degraded"] is True
        assert payload["worker_count"] == 3
        pids = {w["pid"] for w in payload["workers"]}
        assert {_A, _B}.issubset(pids)

        agg_state = {"workers": list(cached), "misses": 2, "set_at": clock["t"]}
        payload = await _build_emit_payload(pipe, client, "ns", "wk", 2, {}, agg_state)
        # The replay is a fallback too, so it is a partial result and the banner must
        # say so. The set on screen is the last known one, not a live read.
        assert payload["degraded"] is True
        # Three, not two: the cached set is still inside the 3 x _PD_KEY_TTL age
        # bound, so it is replayed and the local worker joins it.
        assert payload["worker_count"] == 3
        # Two, not three: the cached set is the two-entry `cached` above and the fix
        # skips the cache write on a fallback tick, so the good set survives the outage.
        assert len(agg_state["workers"]) == 2

    @pytest.mark.asyncio
    async def test_medium_tick_redis_ping_sets_health(self):
        pipe = _make_mock_pipe()
        pipe._redis_client = Mock()
        pipe._redis_client.ping = Mock(return_value=True)
        payload = await _build_emit_payload(pipe, None, "ns", "wk", 8, {})
        assert payload["health"]["redis_connected"] is True

        pipe._redis_client.ping = Mock(side_effect=RuntimeError("down"))
        payload = await _build_emit_payload(pipe, None, "ns", "wk", 8, {})
        assert payload["health"]["redis_connected"] is False

        pipe._redis_client = None
        payload = await _build_emit_payload(pipe, None, "ns", "wk", 8, {})
        assert payload["health"]["redis_connected"] is False

    @pytest.mark.asyncio
    async def test_slow_floor_uses_cache_within_interval(self, monkeypatch):
        pipe = _make_mock_pipe()
        clock = {"t": 1000.0}
        monkeypatch.setattr(dashboard_publisher.time, "monotonic", lambda: clock["t"], raising=False)
        cached = {"storage": {"connected": False}, "config": {"endpoint": "cached"}, "plugins": []}
        slow_state = {"cache": cached, "at": clock["t"]}
        with patch.object(dashboard_publisher, "collect_slow_stats") as mock_slow:
            payload = await _build_emit_payload(pipe, None, "ns", "wk", 0, slow_state)
        mock_slow.assert_not_called()
        assert payload["config"]["endpoint"] == "cached"

    @pytest.mark.asyncio
    async def test_slow_floor_recomputes_after_interval(self):
        pipe = _make_mock_pipe()
        cached = {"storage": {"connected": False}, "config": {"endpoint": "stale"}, "plugins": []}
        slow_state = {"cache": cached, "at": time.monotonic() - 31.0}
        payload = await _build_emit_payload(pipe, None, "ns", "wk", 0, slow_state)
        assert payload["config"]["endpoint"] == "https://openrouter.ai/api/v1"
        assert slow_state["cache"]["config"]["endpoint"] == "https://openrouter.ai/api/v1"


# ── Publisher loop ──


class TestPublisherLoop:
    @pytest.fixture(autouse=True)
    def _fast_sleep(self, monkeypatch):
        real_sleep = asyncio.sleep

        async def fast(_delay):
            await real_sleep(0)

        monkeypatch.setattr(dashboard_publisher.asyncio, "sleep", fast)

    async def _run_briefly(self, get_pipe, get_redis):
        task = asyncio.get_running_loop().create_task(
            run_dashboard_publisher(get_pipe, get_redis, "testns"),
        )
        await _REAL_SLEEP(0.05)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_no_viewers_no_emit(self, monkeypatch):
        monkeypatch.setattr(dashboard_publisher, "local_viewer_sids", lambda: [])
        emit = AsyncMock()
        monkeypatch.setattr(dashboard_publisher, "emit_dashboard", emit)
        await self._run_briefly(_make_mock_pipe, lambda: (None, False))
        emit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_periodic_reauth_called(self, monkeypatch):
        monkeypatch.setattr(dashboard_publisher, "local_viewer_sids", lambda: ["s1"])
        monkeypatch.setattr(dashboard_publisher, "consume_resync", lambda: False)
        monkeypatch.setattr(dashboard_publisher, "emit_dashboard", AsyncMock())
        reauth = AsyncMock()
        monkeypatch.setattr(dashboard_publisher, "reauthorize_local_viewers", reauth)
        await self._run_briefly(_make_mock_pipe, lambda: (None, False))
        reauth.assert_awaited()

    @pytest.mark.asyncio
    async def test_local_viewers_emit_with_tick_sequence(self, monkeypatch):
        monkeypatch.setattr(dashboard_publisher, "local_viewer_sids", lambda: ["s1"])
        monkeypatch.setattr(dashboard_publisher, "consume_resync", lambda: False)
        emit = AsyncMock()
        monkeypatch.setattr(dashboard_publisher, "emit_dashboard", emit)
        await self._run_briefly(_make_mock_pipe, lambda: (None, False))
        assert emit.await_count >= 2
        first = emit.await_args_list[0].args[0]
        second = emit.await_args_list[1].args[0]
        assert first["tick"] == 0
        assert second["tick"] == 1
        assert "identity" in first
        assert "identity" not in second

    @pytest.mark.asyncio
    async def test_resync_resets_tick(self, monkeypatch):
        monkeypatch.setattr(dashboard_publisher, "local_viewer_sids", lambda: ["s1"])
        resyncs = iter([False, True])
        monkeypatch.setattr(dashboard_publisher, "consume_resync", lambda: next(resyncs, False))
        emit = AsyncMock()
        monkeypatch.setattr(dashboard_publisher, "emit_dashboard", emit)
        await self._run_briefly(_make_mock_pipe, lambda: (None, False))
        ticks = [c.args[0]["tick"] for c in emit.await_args_list[:3]]
        assert ticks[0] == 0
        assert ticks[1] == 0
        assert ticks[2] == 1

    @pytest.mark.asyncio
    async def test_other_worker_active_writes_slice_only(self, monkeypatch):
        monkeypatch.setattr(dashboard_publisher, "local_viewer_sids", lambda: [])
        emit = AsyncMock()
        monkeypatch.setattr(dashboard_publisher, "emit_dashboard", emit)
        client = Mock()
        client.set = AsyncMock()
        client.exists = AsyncMock(return_value=1)
        client.pubsub = Mock(side_effect=RuntimeError("no pubsub"))
        client.delete = AsyncMock()
        await self._run_briefly(_make_mock_pipe, lambda: (client, True))
        emit.assert_not_awaited()
        assert client.set.await_count >= 1
        set_key = client.set.await_args_list[0].args[0]
        assert set_key.startswith("testns:dashboard:worker:")

    @pytest.mark.asyncio
    async def test_viewer_activation_sets_flag_and_wakes(self, monkeypatch):
        monkeypatch.setattr(dashboard_publisher, "local_viewer_sids", lambda: ["s1"])
        monkeypatch.setattr(dashboard_publisher, "consume_resync", lambda: False)
        emit = AsyncMock()
        monkeypatch.setattr(dashboard_publisher, "emit_dashboard", emit)
        client = Mock()
        client.set = AsyncMock()
        client.publish = AsyncMock()
        client.pubsub = Mock(side_effect=RuntimeError("no pubsub"))
        client.delete = AsyncMock()

        async def scan_iter(match=None, count=None):
            return
            yield

        client.scan_iter = scan_iter
        await self._run_briefly(_make_mock_pipe, lambda: (client, True))
        set_keys = [c.args[0] for c in client.set.await_args_list]
        assert "testns:dashboard:active" in set_keys
        client.publish.assert_awaited()
        assert client.publish.await_args_list[0].args[0] == "testns:dashboard:wake"
        assert emit.await_count >= 1

    @pytest.mark.asyncio
    async def test_pipe_none_idles(self, monkeypatch):
        monkeypatch.setattr(dashboard_publisher, "local_viewer_sids", lambda: ["s1"])
        emit = AsyncMock()
        monkeypatch.setattr(dashboard_publisher, "emit_dashboard", emit)
        await self._run_briefly(lambda: None, lambda: (None, False))
        emit.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_emit_iteration_failure_does_not_kill_loop(self, monkeypatch):
        monkeypatch.setattr(dashboard_publisher, "local_viewer_sids", lambda: ["s1"])
        monkeypatch.setattr(dashboard_publisher, "consume_resync", lambda: False)
        emit = AsyncMock(side_effect=RuntimeError("emit boom"))
        monkeypatch.setattr(dashboard_publisher, "emit_dashboard", emit)
        await self._run_briefly(_make_mock_pipe, lambda: (None, False))
        assert emit.await_count >= 2

    @pytest.mark.asyncio
    async def test_dead_pubsub_is_reset_and_does_not_spin(self, monkeypatch):
        monkeypatch.setattr(dashboard_publisher, "local_viewer_sids", lambda: [])
        emit = AsyncMock()
        monkeypatch.setattr(dashboard_publisher, "emit_dashboard", emit)
        pubsub = Mock()
        pubsub.subscribe = AsyncMock()
        pubsub.get_message = AsyncMock(side_effect=RuntimeError("dead socket"))
        pubsub.unsubscribe = AsyncMock()
        pubsub.close = AsyncMock()
        client = Mock()
        client.exists = AsyncMock(side_effect=[0, 0, asyncio.CancelledError()])
        client.pubsub = Mock(return_value=pubsub)
        client.delete = AsyncMock()
        await self._run_briefly(_make_mock_pipe, lambda: (client, True))
        assert client.pubsub.call_count >= 2
        emit.assert_not_awaited()


# ── Blob module drift guard ──


class TestSocketIoClientModule:
    _EXPECTED_UMD_SHA384 = "sha384-Yf4YAvFvKwWn8OWlmrC4uKlmukLHHhGW+vZBC+IjvU7JiJYGJI5Z7ea0xLGpQjnE"

    def test_embedded_client_matches_pinned_digest(self):
        import base64
        import hashlib

        from open_webui_openrouter_pipe.plugins.pipe_dashboard._socketio_client import (
            SOCKETIO_UMD,
            SOCKETIO_UMD_SHA384,
        )
        digest = "sha384-" + base64.b64encode(
            hashlib.sha384(SOCKETIO_UMD.encode("utf-8")).digest()
        ).decode("ascii")
        assert digest == self._EXPECTED_UMD_SHA384
        assert SOCKETIO_UMD_SHA384 == self._EXPECTED_UMD_SHA384
        assert len(SOCKETIO_UMD) > 40000
        assert "</script" not in SOCKETIO_UMD


class TestConfigChangeNotification:
    @pytest.mark.asyncio
    async def test_emit_config_changed_no_ignore_queue(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.emit = AsyncMock()
        _install_socket_stub(monkeypatch, sio=mock_sio)
        assert await emit_config_changed(1234) is True
        mock_sio.emit.assert_awaited_once_with(CONFIG_EVENT, {"rev": 1234}, room=VIEWERS_ROOM)

    @pytest.mark.asyncio
    async def test_sink_emits_for_matching_valve_event(self, monkeypatch):
        monkeypatch.setattr(dashboard_socket, "read_config_rev", AsyncMock(return_value=999))
        spy = AsyncMock()
        monkeypatch.setattr(dashboard_socket, "emit_config_changed", spy)
        pipe = Mock()
        pipe.id = "test-pipe"
        dashboard_socket._get_pipe = lambda: pipe
        event = types.SimpleNamespace(event="function.valves_updated", subject={"id": "test-pipe"})
        await dashboard_socket._ValveEventSink().handle_event({}, event)
        await asyncio.sleep(0)
        spy.assert_awaited_once_with(999)

    @pytest.mark.asyncio
    async def test_sink_ignores_other_event(self, monkeypatch):
        spy = AsyncMock()
        monkeypatch.setattr(dashboard_socket, "emit_config_changed", spy)
        pipe = Mock()
        pipe.id = "test-pipe"
        dashboard_socket._get_pipe = lambda: pipe
        event = types.SimpleNamespace(event="function.updated", subject={"id": "test-pipe"})
        await dashboard_socket._ValveEventSink().handle_event({}, event)
        await asyncio.sleep(0)
        spy.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_sink_ignores_other_pipe_id(self, monkeypatch):
        spy = AsyncMock()
        monkeypatch.setattr(dashboard_socket, "emit_config_changed", spy)
        pipe = Mock()
        pipe.id = "test-pipe"
        dashboard_socket._get_pipe = lambda: pipe
        event = types.SimpleNamespace(event="function.valves_updated", subject={"id": "other-pipe"})
        await dashboard_socket._ValveEventSink().handle_event({}, event)
        await asyncio.sleep(0)
        spy.assert_not_awaited()

    def test_register_sink_is_deduped_across_reload(self, monkeypatch):
        other = Mock()
        sinks = [other, dashboard_socket._ValveEventSink()]
        events_mod = types.ModuleType("open_webui.events")
        setattr(events_mod, "EVENT_SINKS", sinks)
        monkeypatch.setitem(sys.modules, "open_webui.events", events_mod)
        assert dashboard_socket.register_valve_event_sink() is True
        assert [type(s).__name__ for s in sinks].count("_ValveEventSink") == 1
        assert other in sinks

    @pytest.mark.asyncio
    async def test_publisher_payload_carries_cfg_rev(self, monkeypatch):
        monkeypatch.setattr(dashboard_publisher, "read_config_rev", AsyncMock(return_value=555))
        pipe = _make_mock_pipe()
        payload = await _build_emit_payload(pipe, None, "ns", "wk", 0, {})
        assert payload["cfgRev"] == 555


class TestReadConfigRev:
    """The config revision the whole config tab notices changes by.

    Its body was never executed: every test that touches it replaced the function with
    an `AsyncMock`, so replacing the two lines that matter with `return None` left the
    suite green. Both consumers then go dark -- `_emit_config_rev` pushes
    `CONFIG_EVENT {"rev": None}` to every viewer, and the slow-tick publisher writes
    `cfg_rev = None` -- which are the two mechanisms the tab uses to notice a settings
    change at all.

    Driven over a real SQLAlchemy `Function` mapping and a real `aiosqlite` database --
    the same engine Open WebUI's own `get_async_db_context` builds -- with only the
    session helper substituted, which is the one boundary that is legitimately a seam.
    A narrow `select` does not go through `Functions.get_function_by_id`, so stubbing
    that method no longer reaches the code under test: the three arms below used to
    patch it, and each of them passed without executing the subject at all.

    The work runs in a subprocess because it re-executes Open WebUI's source and swaps
    `sys.modules` entries, and neither is undone by restoring the mapping.
    """

    @staticmethod
    def _run(tmp_path, body: str) -> None:
        import os
        import subprocess
        import textwrap
        from pathlib import Path

        project_root = Path(__file__).resolve().parents[1]
        script = textwrap.dedent(_READ_CONFIG_REV_HARNESS) + "\n" + textwrap.dedent(body)
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(project_root),
            env={**os.environ, "PYTHONPATH": str(project_root), "WEBUI_SECRET_KEY": "probe",
                 "ORPIPE_PROBE_TMPDIR": str(tmp_path)},
            timeout=300,
            check=False,
        )
        assert proc.returncode == 0, (
            f"probe failed (rc={proc.returncode})\n--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )

    @pytest.mark.parametrize("rev", [1717171717, 1828282828])
    def test_it_returns_the_rows_updated_at(self, tmp_path, rev):
        """Two revisions, because one is satisfied by returning that constant.

        Verified: with a single case, replacing the body with `return 1717171717`
        passed. A second distinct value makes any hardcoded answer fail one of them.
        The single-statement assertion is what makes this the read's own: Open WebUI's
        `get_function_by_id` issues a whole-row select, so the wide path fails here on
        the column list rather than on the value, which is the point.
        """
        self._run(
            tmp_path,
            f"""
            async def main():
                engine = await _build([("openrouter", {rev}), ("other", 9999)])
                try:
                    assert await ds.read_config_rev("openrouter") == {rev}, (
                        "the row's own updated_at is not what the tab was told"
                    )
                finally:
                    await engine.dispose()
                assert len(emitted) == 1, f"expected one statement, got {{emitted!r}}"
                statement, parameters = emitted[0]
                assert "openrouter" in str(parameters), (
                    f"the revision read for the wrong pipe id: {{statement!r}} "
                    f"with {{parameters!r}}"
                )
                assert "9999" not in str(parameters), (
                    f"another function's revision was read: {{parameters!r}}"
                )
                assert _projection(statement) == "function.updated_at", (
                    f"the revision read selected {{_projection(statement)!r}} rather "
                    f"than the one column it needs: {{statement!r}}"
                )

            asyncio.run(main())
            """
        )

    def test_it_is_none_when_the_row_is_missing(self, tmp_path):
        """Needed alongside the case above: alone, either is satisfied by a constant."""
        self._run(
            tmp_path,
            """
            async def main():
                engine = await _build([("other", 9999)])
                try:
                    assert await ds.read_config_rev("openrouter") is None
                finally:
                    await engine.dispose()

            asyncio.run(main())
            """
        )

    def test_it_is_none_when_the_lookup_raises(self, tmp_path):
        self._run(
            tmp_path,
            """
            async def main():
                async with _raising():
                    assert await ds.read_config_rev("openrouter") is None

            asyncio.run(main())
            """
        )
