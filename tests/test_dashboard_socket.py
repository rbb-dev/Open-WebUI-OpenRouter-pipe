"""Tests for the pipe_dashboard OWUI socket.io integration and publisher emit loop."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import time
import types
from unittest.mock import AsyncMock, Mock, patch

import pytest

pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard import dashboard_publisher, dashboard_socket
from open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_publisher import (
    _build_emit_payload,
    run_dashboard_publisher,
)
from open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_socket import (
    CONFIG_EVENT,
    DASHBOARD_EVENT,
    SUB_EVENT,
    VIEWERS_ROOM,
    _pipe_dashboard_sub,
    emit_config_changed,
    register_socket_handler,
)

_REAL_SLEEP = asyncio.sleep


def _install_socket_stub(monkeypatch, **attrs):
    """Install a fake open_webui.socket.main module for the lazy imports."""
    socket_pkg = types.ModuleType("open_webui.socket")
    main_mod = types.ModuleType("open_webui.socket.main")
    for key, value in attrs.items():
        setattr(main_mod, key, value)
    monkeypatch.setitem(sys.modules, "open_webui.socket", socket_pkg)
    monkeypatch.setitem(sys.modules, "open_webui.socket.main", main_mod)
    return main_mod


@pytest.fixture(autouse=True)
def _reset_socket_state():
    registered = dashboard_socket._registered
    resync = dashboard_socket._resync
    get_pipe = dashboard_socket._get_pipe
    yield
    dashboard_socket._registered = registered
    dashboard_socket._resync = resync
    dashboard_socket._get_pipe = get_pipe


def _make_mock_pipe():
    pipe = Mock()
    pipe.id = "test-pipe"
    pipe._global_semaphore = Mock()
    pipe._global_semaphore._value = 45
    pipe._semaphore_limit = 50
    pipe._tool_global_semaphore = Mock()
    pipe._tool_global_semaphore._value = 8
    pipe._tool_global_limit = 10
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
    pipe._video_global_semaphore = None
    pipe._video_global_limit = 0
    pipe._video_active_tasks = {}
    return pipe


# ── Subscribe handler ──


class TestPipeDashboardSub:
    @pytest.mark.asyncio
    async def test_subscribe_no_user_denied(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.enter_room = AsyncMock()
        mock_sio.emit = AsyncMock()
        dashboard_socket._resync = False
        _install_socket_stub(
            monkeypatch, sio=mock_sio, get_user_id_from_session_pool=lambda sid: None,
        )
        dashboard_socket._get_pipe = lambda: object()
        await _pipe_dashboard_sub("sid-anon")
        mock_sio.enter_room.assert_not_awaited()
        assert dashboard_socket._resync is False

    @pytest.mark.asyncio
    async def test_subscribe_denied_emits_denied(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.enter_room = AsyncMock()
        mock_sio.emit = AsyncMock()
        dashboard_socket._resync = False
        _install_socket_stub(
            monkeypatch, sio=mock_sio, get_user_id_from_session_pool=lambda sid: "user-1",
        )
        monkeypatch.setattr(dashboard_socket,"resolve_user", AsyncMock(return_value=object()))
        monkeypatch.setattr(dashboard_socket,"can_view", AsyncMock(return_value=False))
        dashboard_socket._get_pipe = lambda: object()
        await _pipe_dashboard_sub("sid-denied")
        mock_sio.enter_room.assert_not_awaited()
        mock_sio.emit.assert_awaited_once_with(dashboard_socket.DENIED_EVENT, {}, room="sid-denied")
        assert dashboard_socket._resync is False

    @pytest.mark.asyncio
    async def test_subscribe_granted_joins_room(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.enter_room = AsyncMock()
        mock_sio.emit = AsyncMock()
        dashboard_socket._resync = False
        _install_socket_stub(
            monkeypatch, sio=mock_sio, get_user_id_from_session_pool=lambda sid: "user-1",
        )
        monkeypatch.setattr(dashboard_socket,"resolve_user", AsyncMock(return_value=object()))
        monkeypatch.setattr(dashboard_socket,"can_view", AsyncMock(return_value=True))
        dashboard_socket._get_pipe = lambda: object()
        await _pipe_dashboard_sub("sid-authed")
        mock_sio.enter_room.assert_awaited_once_with("sid-authed", VIEWERS_ROOM)
        assert dashboard_socket._resync is True

    @pytest.mark.asyncio
    async def test_enter_room_failure_no_resync(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.enter_room = AsyncMock(side_effect=RuntimeError("boom"))
        mock_sio.emit = AsyncMock()
        dashboard_socket._resync = False
        _install_socket_stub(
            monkeypatch, sio=mock_sio, get_user_id_from_session_pool=lambda sid: "user-1",
        )
        monkeypatch.setattr(dashboard_socket,"resolve_user", AsyncMock(return_value=object()))
        monkeypatch.setattr(dashboard_socket,"can_view", AsyncMock(return_value=True))
        dashboard_socket._get_pipe = lambda: object()
        await _pipe_dashboard_sub("sid-err")
        assert dashboard_socket._resync is False


class TestReauthorizeLocalViewers:
    @pytest.mark.asyncio
    async def test_evicts_ungranted(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.leave_room = AsyncMock()
        mock_sio.emit = AsyncMock()
        _install_socket_stub(
            monkeypatch, sio=mock_sio,
            get_session_ids_from_room=lambda room: ["s1"],
            get_user_id_from_session_pool=lambda sid: "user-1",
        )
        monkeypatch.setattr(dashboard_socket,"resolve_user", AsyncMock(return_value=object()))
        monkeypatch.setattr(dashboard_socket,"can_view", AsyncMock(return_value=False))
        dashboard_socket._get_pipe = lambda: object()
        await dashboard_socket.reauthorize_local_viewers()
        mock_sio.leave_room.assert_awaited_once_with("s1", VIEWERS_ROOM)
        mock_sio.emit.assert_awaited_once_with(dashboard_socket.DENIED_EVENT, {}, room="s1")

    @pytest.mark.asyncio
    async def test_keeps_granted(self, monkeypatch):
        mock_sio = Mock()
        mock_sio.leave_room = AsyncMock()
        mock_sio.emit = AsyncMock()
        _install_socket_stub(
            monkeypatch, sio=mock_sio,
            get_session_ids_from_room=lambda room: ["s1"],
            get_user_id_from_session_pool=lambda sid: "user-1",
        )
        monkeypatch.setattr(dashboard_socket,"resolve_user", AsyncMock(return_value=object()))
        monkeypatch.setattr(dashboard_socket,"can_view", AsyncMock(return_value=True))
        dashboard_socket._get_pipe = lambda: object()
        await dashboard_socket.reauthorize_local_viewers()
        mock_sio.leave_room.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_import_failure_safe(self, monkeypatch):
        _install_socket_stub(monkeypatch)
        await dashboard_socket.reauthorize_local_viewers()


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


def _compact_slice(pid, active=1):
    return {
        "pid": pid,
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
        client = _redis_with_slices([_compact_slice(11111, active=2), _compact_slice(22222, active=3)])
        payload = await _build_emit_payload(pipe, client, "ns", "wk", 1, {})
        assert payload["worker_count"] == 3
        pids = {w["pid"] for w in payload["workers"]}
        assert {11111, 22222, os.getpid()} == pids
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
    async def test_redis_blip_uses_cached_workers_and_degrades(self):
        pipe = _make_mock_pipe()
        client = Mock()
        client.set = AsyncMock()

        async def scan_boom(match=None, count=None):
            raise RuntimeError("redis down")
            yield

        client.scan_iter = scan_boom
        cached = [
            {"pid": 111, "uptime_s": 50.0,
             "concurrency": {"active_requests": 1, "max_requests": 50, "active_tools": 0, "max_tools": 10},
             "queues": {}, "rate_limits": {}, "sessions": {"in_flight": 0}},
            {"pid": 222, "uptime_s": 60.0,
             "concurrency": {"active_requests": 2, "max_requests": 50, "active_tools": 0, "max_tools": 10},
             "queues": {}, "rate_limits": {}, "sessions": {"in_flight": 0}},
        ]
        agg_state = {"workers": list(cached), "misses": 0}
        payload = await _build_emit_payload(pipe, client, "ns", "wk", 1, {}, agg_state)
        assert payload["degraded"] is True
        assert payload["worker_count"] == 3
        pids = {w["pid"] for w in payload["workers"]}
        assert {111, 222}.issubset(pids)

        agg_state = {"workers": list(cached), "misses": 2}
        payload = await _build_emit_payload(pipe, client, "ns", "wk", 2, {}, agg_state)
        assert "degraded" not in payload
        assert payload["worker_count"] == 1

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
    async def test_slow_floor_uses_cache_within_interval(self):
        pipe = _make_mock_pipe()
        cached = {"storage": {"connected": False}, "config": {"endpoint": "cached"}, "plugins": []}
        slow_state = {"cache": cached, "at": time.monotonic()}
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
        monkeypatch.setattr(dashboard_publisher, "_PD_REAUTH_EVERY", 1)
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
