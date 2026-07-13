"""Lifecycle/DDL hardening tests: DDL race tolerance, awaited plugin teardown,
awaited redis-task teardown, sync-path cleanup cancel, and no-plugins safety
of every dispatch surface."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.plugins.registry import PluginRegistry
from open_webui_openrouter_pipe.storage.persistence import ArtifactStore


# ── DDL race guard ──


def _guard_host(heal: bool = False) -> Any:
    return SimpleNamespace(
        logger=Mock(),
        _is_table_exists_error=ArtifactStore._is_table_exists_error,
        _maybe_heal_index_conflict=lambda *a, **k: heal,
    )


def test_table_exists_error_detection():
    assert ArtifactStore._is_table_exists_error(Exception('relation "stats_x" already exists'))
    assert ArtifactStore._is_table_exists_error(Exception("table response_items already exists"))
    assert not ArtifactStore._is_table_exists_error(Exception("permission denied for schema public"))


def test_race_guard_concurrent_create_is_success():
    host = _guard_host()
    table = Mock()
    table.create.side_effect = Exception('relation "stats_x" already exists')
    assert ArtifactStore._create_table_with_race_guard(host, table, Mock(), "stats_x") is True


def test_race_guard_real_failure_disables():
    host = _guard_host()
    table = Mock()
    table.create.side_effect = Exception("permission denied")
    assert ArtifactStore._create_table_with_race_guard(host, table, Mock(), "t") is False
    host.logger.warning.assert_called()


def test_race_guard_clean_create_succeeds():
    host = _guard_host()
    table = Mock()
    assert ArtifactStore._create_table_with_race_guard(host, table, Mock(), "t") is True
    table.create.assert_called_once()


def test_race_guard_heals_index_conflict_before_exists_shortcircuit():
    """A duplicate-INDEX error also says 'already exists' — the healer must
    run first, then the retry succeeds. Guards against the short-circuit
    swallowing index conflicts and declaring a rolled-back create a success."""
    heal_calls = []
    host = _guard_host()
    host._maybe_heal_index_conflict = lambda *a, **k: heal_calls.append(1) or True
    table = Mock()
    table.create.side_effect = [Exception('index "ix_response_items_chat_id" already exists'), None]
    assert ArtifactStore._create_table_with_race_guard(host, table, Mock(), "t") is True
    assert heal_calls == [1]
    assert table.create.call_count == 2


def test_race_guard_heal_then_concurrent_create_is_success():
    host = _guard_host(heal=True)
    table = Mock()
    table.create.side_effect = [
        Exception('index "ix_t_chat_id" already exists'),
        Exception('relation "t" already exists'),
    ]
    assert ArtifactStore._create_table_with_race_guard(host, table, Mock(), "t") is True
    assert table.create.call_count == 2


# ── dispatch_on_shutdown awaitable collection ──


@pytest.mark.asyncio
async def test_dispatch_on_shutdown_collects_awaitables():
    task = asyncio.get_running_loop().create_task(asyncio.sleep(60))
    good = Mock()
    good.plugin_id = "good"
    good.on_shutdown = Mock(return_value=task)
    bad = Mock()
    bad.plugin_id = "bad"
    bad.on_shutdown = Mock(side_effect=RuntimeError("boom"))
    quiet = Mock()
    quiet.plugin_id = "quiet"
    quiet.on_shutdown = Mock(return_value=None)

    reg = PluginRegistry()
    reg._plugins = [good, bad, quiet]
    pending = reg.dispatch_on_shutdown()
    assert pending == [task]
    assert bad.on_shutdown.called and quiet.on_shutdown.called

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_dispatch_on_shutdown_empty_registry_returns_empty():
    assert PluginRegistry().dispatch_on_shutdown() == []


# ── no-plugins safety: every dispatcher no-ops on an empty registry ──


@pytest.mark.asyncio
async def test_empty_registry_dispatchers_are_noops():
    reg = PluginRegistry()
    models = [{"id": "m"}]
    await reg.dispatch_on_models(models)
    assert models == [{"id": "m"}]

    assert await reg.dispatch_on_request({}, {}, {}, None, None) is None

    body = {"model": "x", "k": 1}
    await reg.dispatch_on_request_transform(body, "x", None)
    assert body == {"model": "x", "k": 1}

    sentinel = object()
    assert await reg.dispatch_on_emitter_wrap(sentinel) is None

    assert reg.dispatch_on_shutdown() == []


@pytest.mark.asyncio
async def test_pipe_without_registry_shutdown_and_close_safe():
    pipe = Pipe()
    assert pipe._plugin_registry is None
    assert pipe.shutdown() == []
    await pipe.close()


# ── awaited plugin teardown through Pipe.close() ──


@pytest.mark.asyncio
async def test_pipe_close_awaits_plugin_shutdown_task():
    pipe = Pipe()
    flag = {"cleaned": False}

    async def _worker():
        try:
            await asyncio.sleep(60)
        finally:
            flag["cleaned"] = True

    task = asyncio.get_running_loop().create_task(_worker())

    def _on_shutdown(**kwargs):
        task.cancel()
        return task

    plugin = Mock()
    plugin.plugin_id = "p"
    plugin.on_shutdown = _on_shutdown
    reg = PluginRegistry()
    reg._plugins = [plugin]
    pipe._plugin_registry = reg

    await pipe.close()
    assert flag["cleaned"] is True


@pytest.mark.asyncio
async def test_pipe_dashboard_on_shutdown_returns_cancelled_task():
    pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.plugin import PipeDashboardPlugin

    plugin = PipeDashboardPlugin()
    task = asyncio.get_running_loop().create_task(asyncio.sleep(60))
    plugin._publisher_task = task

    returned = plugin.on_shutdown()
    assert returned is task
    with pytest.raises(asyncio.CancelledError):
        await task
    assert plugin.on_shutdown() is None


# ── awaited redis-task teardown ──


@pytest.mark.asyncio
async def test_stop_redis_awaits_cancelled_tasks():
    pipe = Pipe()
    flags = {"listener": False, "flush": False}

    async def _worker(key):
        try:
            await asyncio.sleep(60)
        finally:
            flags[key] = True

    loop = asyncio.get_running_loop()
    pipe._redis_listener_task = loop.create_task(_worker("listener"))
    pipe._redis_flush_task = loop.create_task(_worker("flush"))
    pipe._redis_ready_task = None
    pipe._redis_client = None
    await asyncio.sleep(0)

    await pipe._stop_redis()
    assert flags == {"listener": True, "flush": True}
    assert pipe._redis_listener_task is None
    assert pipe._redis_flush_task is None
    await pipe.close()


# ── sync shutdown() cancels the cleanup task ──


@pytest.mark.asyncio
async def test_sync_shutdown_cancels_cleanup_task():
    pipe = Pipe()
    task = asyncio.get_running_loop().create_task(asyncio.sleep(60))
    pipe._cleanup_task = task

    pipe.shutdown()
    with pytest.raises(asyncio.CancelledError):
        await task
    pipe._cleanup_task = None
    await pipe.close()
