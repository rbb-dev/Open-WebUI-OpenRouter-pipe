from __future__ import annotations

import asyncio
import contextlib
import gc
import importlib
import logging
import os
import sys
import threading
import time
import types
import weakref
from pathlib import Path
from typing import Any, cast

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
READABLE_BUNDLE = REPO_ROOT / "open_webui_openrouter_pipe_bundled.py"
COMPRESSED_BUNDLE = REPO_ROOT / "open_webui_openrouter_pipe_bundled_compressed.py"
LIFECYCLE_REGISTRY_KEY = "_openrouter_pipe_lifecycle"


@pytest.fixture(autouse=True)
def _enable_test_mode(monkeypatch):
    monkeypatch.setenv("OWUI_PIPE_TEST_MODE", "1")
    yield


@pytest.fixture
def fresh_registry():
    saved = sys.modules.pop(LIFECYCLE_REGISTRY_KEY, None)
    try:
        yield
    finally:
        sys.modules.pop(LIFECYCLE_REGISTRY_KEY, None)
        if saved is not None:
            sys.modules[LIFECYCLE_REGISTRY_KEY] = saved


@pytest.fixture
def isolated_sys_modules():
    modules_snapshot = dict(sys.modules)
    meta_path_snapshot = list(sys.meta_path)
    yield
    for k in list(sys.modules):
        if k not in modules_snapshot:
            del sys.modules[k]
    for k, v in modules_snapshot.items():
        sys.modules[k] = v
    sys.meta_path[:] = meta_path_snapshot


def _import_pipe_module():
    return importlib.import_module("open_webui_openrouter_pipe.pipe")


def _exec_bundle_as_module(bundle_path: Path, module_name: str) -> types.ModuleType:
    source = bundle_path.read_text(encoding="utf-8")
    module = types.ModuleType(module_name)
    module.__file__ = str(bundle_path)
    sys.modules[module_name] = module
    exec(compile(source, str(bundle_path), "exec"), module.__dict__)
    return module


def test_registry_swap_in_empty_returns_none(fresh_registry):
    pipe_mod = _import_pipe_module()
    pipe = pipe_mod.Pipe()
    pipe.id = "test_id_empty"
    registry = pipe_mod._get_lifecycle_registry()
    assert registry.swap_in(pipe) is None


def test_registry_swap_in_returns_predecessor(fresh_registry):
    pipe_mod = _import_pipe_module()
    a = pipe_mod.Pipe()
    a.id = "test_id_swap"
    b = pipe_mod.Pipe()
    b.id = "test_id_swap"
    registry = pipe_mod._get_lifecycle_registry()
    assert registry.swap_in(a) is None
    assert registry.swap_in(b) is a


def test_registry_weakref_decay(fresh_registry):
    pipe_mod = _import_pipe_module()
    a = pipe_mod.Pipe()
    a.id = "test_id_decay"
    registry = pipe_mod._get_lifecycle_registry()
    registry.swap_in(a)
    del a
    gc.collect()
    b = pipe_mod.Pipe()
    b.id = "test_id_decay"
    assert registry.swap_in(b) is None


@pytest.mark.asyncio
async def test_close_when_idle_immediate_close_when_no_active_calls(fresh_registry):
    pipe_mod = _import_pipe_module()
    pipe = pipe_mod.Pipe()
    pipe.id = "test_id_idle"
    assert pipe._active_pipes_calls == 0
    pipe.close_when_idle()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert pipe._draining is True
    assert pipe._closing is True or pipe._closed is True


@pytest.mark.asyncio
async def test_close_when_idle_defers_until_active_completes(fresh_registry):
    pipe_mod = _import_pipe_module()
    pipe = pipe_mod.Pipe()
    pipe.id = "test_id_defer"
    pipe._active_pipes_calls = 1
    pipe.close_when_idle()
    assert pipe._draining is True
    assert pipe._closing is False
    assert pipe._closed is False
    pipe._active_pipes_calls -= 1
    pipe._maybe_trigger_drain_close()
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    assert pipe._closing is True or pipe._closed is True


def test_close_when_idle_idempotent(fresh_registry):
    pipe_mod = _import_pipe_module()
    pipe = pipe_mod.Pipe()
    pipe.id = "test_id_idem"
    pipe._active_pipes_calls = 1
    pipe.close_when_idle()
    pipe.close_when_idle()
    pipe.close_when_idle()
    assert pipe._draining is True
    assert pipe._closing is False


@pytest.mark.asyncio
async def test_close_idempotent_concurrent_callers(fresh_registry):
    pipe_mod = _import_pipe_module()
    pipe = pipe_mod.Pipe()
    pipe.id = "test_id_close_idem"
    results = await asyncio.gather(
        pipe.close(), pipe.close(), pipe.close(), return_exceptions=True
    )
    assert all(r is None for r in results)
    assert pipe._closing is True
    assert pipe._closed is True


def test_replacement_predecessor_enters_drain_when_busy(fresh_registry):
    pipe_mod = _import_pipe_module()
    g1 = pipe_mod.Pipe()
    g1.id = "test_id_replace_busy"
    g1._active_pipes_calls = 1
    registry = pipe_mod._get_lifecycle_registry()
    registry.swap_in(g1)
    g2 = pipe_mod.Pipe()
    g2.id = "test_id_replace_busy"
    predecessor = registry.swap_in(g2)
    assert predecessor is g1
    predecessor.close_when_idle()
    assert g1._draining is True
    assert g1._closing is False
    assert g2._draining is False


@pytest.mark.asyncio
async def test_old_close_does_not_clobber_new_generation_log_queue(fresh_registry):
    pipe_mod = _import_pipe_module()
    from open_webui_openrouter_pipe.core.logging_system import SessionLogger

    g1 = pipe_mod.Pipe()
    g1.id = "test_id_static"
    g1_queue: asyncio.Queue = asyncio.Queue()
    g1._log_queue = g1_queue
    g1._log_queue_loop = asyncio.get_running_loop()

    g2 = pipe_mod.Pipe()
    g2.id = "test_id_static"
    g2_queue: asyncio.Queue = asyncio.Queue()

    SessionLogger.set_log_queue(g2_queue)
    SessionLogger.set_main_loop(asyncio.get_running_loop())

    await g1._stop_log_worker()
    assert SessionLogger.log_queue is g2_queue


def test_cleanup_loop_stop_latency_under_500ms():
    from open_webui_openrouter_pipe.logging.session_log_manager import SessionLogManager

    class _FakeValves:
        SESSION_LOG_CLEANUP_INTERVAL_SECONDS = 3600
        SESSION_LOG_RETENTION_DAYS = 7
        SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS = 60
        SESSION_LOG_ASSEMBLER_JITTER_SECONDS = 5
        SESSION_LOG_STORE_ENABLED = False
        SESSION_LOG_DIR = ""
        SESSION_LOG_PASSWORD = None
        SESSION_LOG_PASSWORD_KEY_VERSION = 0

    class _FakePipe:
        valves = _FakeValves()

    import logging as stdlib_logging
    mgr = SessionLogManager(logger=stdlib_logging.getLogger("test"), pipe=_FakePipe())  # type: ignore[arg-type]
    mgr.start_workers()

    started = time.monotonic()
    mgr.stop_workers()
    elapsed = time.monotonic() - started
    assert elapsed < 0.5, f"stop_workers took {elapsed:.3f}s (>0.5s)"


def test_pipe_does_not_pin_self_via_log_worker_closure(fresh_registry):
    pipe_mod = _import_pipe_module()
    pipe = pipe_mod.Pipe()
    pipe.id = "test_id_weakref"
    ref = weakref.ref(pipe)
    del pipe
    gc.collect()
    assert ref() is None


def test_multi_update_chain_drains_each_generation(fresh_registry):
    pipe_mod = _import_pipe_module()
    registry = pipe_mod._get_lifecycle_registry()

    g1 = pipe_mod.Pipe()
    g1.id = "test_id_chain"
    g1._active_pipes_calls = 1
    assert registry.swap_in(g1) is None

    g2 = pipe_mod.Pipe()
    g2.id = "test_id_chain"
    pred = registry.swap_in(g2)
    assert pred is g1
    pred.close_when_idle()
    assert g1._draining is True

    g3 = pipe_mod.Pipe()
    g3.id = "test_id_chain"
    pred = registry.swap_in(g3)
    assert pred is g2
    pred.close_when_idle()
    assert g2._draining is True
    assert g3._draining is False


@pytest.mark.skipif(not READABLE_BUNDLE.exists(), reason="readable bundle not built")
def test_readable_bundle_hot_reload_uses_new_code(isolated_sys_modules):
    module_name = "function_test_pipe_readable"
    for k in list(sys.modules):
        if k == module_name or k.startswith("open_webui_openrouter_pipe"):
            sys.modules.pop(k, None)
    sys.modules.pop(LIFECYCLE_REGISTRY_KEY, None)

    mod1 = _exec_bundle_as_module(READABLE_BUNDLE, module_name)
    pipe1_class = mod1.Pipe

    mod2 = _exec_bundle_as_module(READABLE_BUNDLE, module_name)
    pipe2_class = mod2.Pipe

    assert pipe1_class is not pipe2_class
    assert sys.modules.get("open_webui_openrouter_pipe") is mod2


def test_get_lifecycle_registry_preserves_cross_version_instance(fresh_registry):
    pipe_mod = _import_pipe_module()
    simulated_old_class = type(
        "_LifecycleRegistry",
        (),
        dict(pipe_mod._LifecycleRegistry.__dict__),
    )
    old_reg = simulated_old_class()
    sys.modules[LIFECYCLE_REGISTRY_KEY] = old_reg  # type: ignore[assignment]
    result = pipe_mod._get_lifecycle_registry()
    assert result is old_reg


def test_auto_register_chain_drains_predecessor(fresh_registry):
    pipe_mod = _import_pipe_module()
    g1 = pipe_mod.Pipe()
    g1.id = "auto_register_chain_test"
    g1._active_pipes_calls = 1
    g1._attach_to_lifecycle_registry()
    g2 = pipe_mod.Pipe()
    g2.id = "auto_register_chain_test"
    g2._attach_to_lifecycle_registry()
    assert g1._draining is True
    assert g2._draining is False


def test_attach_to_lifecycle_registry_no_predecessor_is_noop(fresh_registry):
    pipe_mod = _import_pipe_module()
    g1 = pipe_mod.Pipe()
    g1.id = "no_predecessor_test"
    g1._attach_to_lifecycle_registry()
    assert g1._draining is False
    assert g1._closing is False


@pytest.mark.asyncio
async def test_concurrent_close_all_callers_see_completion(fresh_registry):
    pipe_mod = _import_pipe_module()
    pipe = pipe_mod.Pipe()
    pipe.id = "concurrent_close_test"
    results = await asyncio.gather(
        pipe.close(), pipe.close(), pipe.close(), return_exceptions=True
    )
    assert all(r is None for r in results)
    assert pipe._closing is True
    assert pipe._closed is True
    assert pipe._close_done is not None
    assert pipe._close_done.done() is True


def test_counter_underflow_guard():
    pipe_mod = _import_pipe_module()
    pipe = pipe_mod.Pipe()
    pipe.id = "underflow_guard_test"
    pipe._active_pipes_calls = 0
    state = {"released": False}
    pipe_mod.Pipe._release_stream_counter(pipe, state)
    assert pipe._active_pipes_calls == 0


@pytest.mark.asyncio
async def test_supersession_emits_visible_lifecycle_messages(fresh_registry, capsys, caplog):
    """Operators must see three messages on hot-reload supersession:
    (1) new instance logs 'sent close' via stdlib logger,
    (2) old instance prints 'close received' to stderr,
    (3) old instance prints 'closing' to stderr.

    stderr is used for old-instance messages because by the time _do_close()
    tears down the SessionLogger queue, the logger pipeline is unreliable.
    """
    import logging as stdlib_logging

    pipe_mod = _import_pipe_module()

    g1 = pipe_mod.Pipe()
    g1.id = "supersession_visibility_test"
    g1._attach_to_lifecycle_registry()

    g2 = pipe_mod.Pipe()
    g2.id = "supersession_visibility_test"

    with caplog.at_level(stdlib_logging.INFO, logger=g2.logger.name):
        g2._attach_to_lifecycle_registry()
        await asyncio.sleep(0)
        await asyncio.sleep(0)

    captured = capsys.readouterr()

    # 1. New instance logged "sent close" via stdlib logger.
    assert any(
        "hot-reload: sent close to predecessor" in r.message
        for r in caplog.records
    ), f"caplog records: {[r.message for r in caplog.records]}"

    # 2. Old instance printed "close received" to stderr (bypassing logger).
    assert "hot-reload: close received" in captured.err, (
        f"stderr did not contain 'close received'. stderr was:\n{captured.err}"
    )

    # 3. Old instance printed "closing" to stderr.
    assert "hot-reload: closing" in captured.err, (
        f"stderr did not contain 'closing'. stderr was:\n{captured.err}"
    )

    # Identifying info must be present so multi-pipe deployments can disambiguate.
    assert "id=supersession_visibility_test" in captured.err
    assert f"pid={os.getpid()}" in captured.err


@pytest.mark.skipif(not COMPRESSED_BUNDLE.exists(), reason="compressed bundle not built")
def test_compressed_bundle_hot_reload_uses_new_code(isolated_sys_modules):
    module_name = "function_test_pipe_compressed"
    for k in list(sys.modules):
        if k == module_name or k.startswith("open_webui_openrouter_pipe"):
            sys.modules.pop(k, None)
    import sys as _sys
    _sys.meta_path[:] = [
        f for f in _sys.meta_path if type(f).__name__ != "_BundledModuleFinder"
    ]
    sys.modules.pop(LIFECYCLE_REGISTRY_KEY, None)

    mod1 = _exec_bundle_as_module(COMPRESSED_BUNDLE, module_name)
    pipe1_class = mod1.Pipe

    mod2 = _exec_bundle_as_module(COMPRESSED_BUNDLE, module_name)
    pipe2_class = mod2.Pipe

    assert pipe1_class is not pipe2_class
    pkg = sys.modules.get("open_webui_openrouter_pipe")
    assert pkg is not None
    assert pipe1_class not in pipe2_class.__mro__
