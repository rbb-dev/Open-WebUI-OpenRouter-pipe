from __future__ import annotations

import asyncio
import contextlib
import ctypes
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

from tests.pipe_limits import slot, set_slot, set_slots, REQUEST_SEMAPHORE, REQUEST_LIMIT
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
READABLE_BUNDLE = REPO_ROOT / "open_webui_openrouter_pipe_bundled.py"
COMPRESSED_BUNDLE = REPO_ROOT / "open_webui_openrouter_pipe_bundled_compressed.py"
LIFECYCLE_REGISTRY_KEY = "_openrouter_pipe_lifecycle"

stall_alarm = importlib.import_module("open_webui_openrouter_pipe.core.stall_alarm")

STALL_TICK = 0.02
STALL_VALVE = 0.2
STALL_HOLD = 0.35


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


@pytest.fixture(scope="module")
def compiled_bundles() -> dict[Path, types.CodeType]:
    return {
        path: compile(path.read_text(encoding="utf-8"), str(path), "exec")
        for path in (READABLE_BUNDLE, COMPRESSED_BUNDLE)
        if path.exists()
    }


def _exec_bundle_as_module(
    bundle_path: Path, module_name: str, code: types.CodeType
) -> types.ModuleType:
    module = types.ModuleType(module_name)
    module.__file__ = str(bundle_path)
    sys.modules[module_name] = module
    exec(code, module.__dict__)
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
    """A retired generation drains its own queue and leaves the live one's alone.

    Two halves of one property, and either alone is a false pass. Draining g1's queue
    through `SessionLogger.log_queue` instead of the local reference it captured would
    take g2's records and write them into g1's request, and the queue-identity assert
    below is what catches it -- but a close that drained nothing at all would satisfy
    that assert on its own, so g1's own records are read back out of the session log as
    well. g1 has no worker task, which is the ordinary close: a generation retired before
    its log worker ever started.
    """
    pipe_mod = _import_pipe_module()
    import logging

    from open_webui_openrouter_pipe.core.logging_system import SessionLogger

    rid = "req-retired-generation"
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

    for index in range(3):
        record = logging.LogRecord(
            name="open_webui_openrouter_pipe",
            level=logging.WARNING,
            pathname="pipe.py",
            lineno=1,
            msg="retired-generation-marker-%d",
            args=(index,),
            exc_info=None,
        )
        record.request_id = rid
        record.max_lines = 500
        g1_queue.put_nowait(record)

    try:
        await g1._stop_log_worker()
        delivered = [
            str(event.get("message", ""))
            for event in (SessionLogger.logs.get(rid) or [])
            if "retired-generation-marker" in str(event.get("message", ""))
        ]
        assert SessionLogger.log_queue is g2_queue, (
            f"g1's close released the global queue, which now holds {SessionLogger.log_queue!r} "
            f"rather than g2's; a hot reload is the shape where this fires on the same event"
        )
    finally:
        SessionLogger.logs.pop(rid, None)
        SessionLogger.set_log_queue(None)
        SessionLogger.set_main_loop(None)

    assert len(delivered) == 3, (
        f"g1 held 3 queued log records when its close ran and {len(delivered)} reached the "
        "session log; the retired generation's records are the tail of the archive"
    )


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
def test_readable_bundle_hot_reload_uses_new_code(isolated_sys_modules, compiled_bundles):
    module_name = "function_test_pipe_readable"
    for k in list(sys.modules):
        if k == module_name or k.startswith("open_webui_openrouter_pipe"):
            sys.modules.pop(k, None)
    sys.modules.pop(LIFECYCLE_REGISTRY_KEY, None)

    code = compiled_bundles[READABLE_BUNDLE]
    mod1 = _exec_bundle_as_module(READABLE_BUNDLE, module_name, code)
    pipe1_class = mod1.Pipe

    mod2 = _exec_bundle_as_module(READABLE_BUNDLE, module_name, code)
    pipe2_class = mod2.Pipe

    assert pipe1_class is not pipe2_class
    assert sys.modules.get("open_webui_openrouter_pipe") is mod2

    stub = _StubPipe()
    assert mod1.attach_stall_alarm(stub).state is mod2.attach_stall_alarm(stub).state


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


def test_attach_drains_instances_a_012_load_left_under_a_per_load_id(fresh_registry):
    pipe_mod = _import_pipe_module()
    left_behind = pipe_mod.Pipe()
    left_behind.id = "per_load_reload_test_0123456789abcdef0123456789abcdef"
    left_behind._active_pipes_calls = 1
    left_behind._attach_to_lifecycle_registry()
    unrelated = pipe_mod.Pipe()
    unrelated.id = "per_load_reload_test_other_0123456789abcdef0123456789abcdef"
    unrelated._active_pipes_calls = 1
    unrelated._attach_to_lifecycle_registry()
    fixed = pipe_mod.Pipe()
    fixed.id = "per_load_reload_test"
    fixed._attach_to_lifecycle_registry()
    assert left_behind._draining is True
    assert unrelated._draining is False
    assert fixed._draining is False


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
def test_compressed_bundle_hot_reload_uses_new_code(isolated_sys_modules, compiled_bundles):
    module_name = "function_test_pipe_compressed"
    for k in list(sys.modules):
        if k == module_name or k.startswith("open_webui_openrouter_pipe"):
            sys.modules.pop(k, None)
    import sys as _sys
    _sys.meta_path[:] = [
        f for f in _sys.meta_path if type(f).__name__ != "_BundledModuleFinder"
    ]
    sys.modules.pop(LIFECYCLE_REGISTRY_KEY, None)

    code = compiled_bundles[COMPRESSED_BUNDLE]
    mod1 = _exec_bundle_as_module(COMPRESSED_BUNDLE, module_name, code)
    pipe1_class = mod1.Pipe

    mod2 = _exec_bundle_as_module(COMPRESSED_BUNDLE, module_name, code)
    pipe2_class = mod2.Pipe

    assert pipe1_class is not pipe2_class
    pkg = sys.modules.get("open_webui_openrouter_pipe")
    assert pkg is not None
    assert pipe1_class not in pipe2_class.__mro__


class _StubPipe:
    def __init__(
        self,
        logger: logging.Logger | None = None,
        valve: float = 3,
        loop_seconds: float = 5,
        dump: bool = False,
    ) -> None:
        self.logger = logger or logging.getLogger("stall_alarm_stub")
        self.valves = types.SimpleNamespace(
            STALL_ALARM_ENABLED=True,
            STALL_ALARM_LOOP_SECONDS=loop_seconds,
            STALL_ALARM_GIL_SECONDS=valve,
            STALL_ALARM_REPEAT_SECONDS=60,
            STALL_ALARM_DUMP_ENABLED=dump,
        )


def _hold_the_interpreter_lock(seconds: float) -> None:
    ctypes.PyDLL(None).usleep(int(seconds * 1_000_000))


def _stall_watchers() -> list[str]:
    return [t.name for t in threading.enumerate() if t.name == stall_alarm._STALL_THREAD_NAME]


def _wait_until(condition, seconds: float = 2.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        time.sleep(STALL_TICK)
    assert condition()


def test_two_loaded_copies_of_the_pipe_share_one_stall_watcher_and_a_gil_hold_is_dumped_while_it_runs(
    monkeypatch,
):
    monkeypatch.setattr(stall_alarm, "_STALL_TICK_SECONDS", STALL_TICK)
    next_generation = type("NextGenerationHolder", (stall_alarm._StallAlarmHolder,), {})
    pipes = [_StubPipe(valve=STALL_VALVE, loop_seconds=30, dump=True) for _ in range(2)]
    first = stall_alarm.attach_stall_alarm(pipes[0])
    monkeypatch.setattr(stall_alarm, "_StallAlarmHolder", next_generation)
    second = stall_alarm.attach_stall_alarm(pipes[1])
    state = first.state
    assert type(sys.modules[stall_alarm._STALL_ALARM_HOLDER_KEY]) is next_generation
    assert second.state is state
    assert _stall_watchers() == []

    stall_alarm.ensure_stall_alarm(first)
    stall_alarm.ensure_stall_alarm(second)
    assert _stall_watchers() == [stall_alarm._STALL_THREAD_NAME]

    _wait_until(lambda: state.dump_path is not None)
    dump_path = state.dump_path
    watcher = state.watcher
    try:
        time.sleep(STALL_VALVE + 2 * STALL_TICK + 0.05)
        assert os.path.getsize(dump_path) == 0

        began = time.time()
        _hold_the_interpreter_lock(STALL_HOLD)
        ended = time.time()
        written = os.stat(dump_path)
        assert written.st_size > 0 and began < written.st_mtime < ended
        assert "_hold_the_interpreter_lock" in Path(dump_path).read_text(errors="replace")
    finally:
        stall_alarm.detach_stall_alarm(first)
        stall_alarm.detach_stall_alarm(second)
        if watcher is not None:
            watcher.join(2)
        with contextlib.suppress(OSError):
            os.unlink(dump_path)


def test_nothing_is_dumped_into_a_recycled_descriptor_after_the_last_pipe_closes(monkeypatch):
    monkeypatch.setattr(stall_alarm, "_STALL_TICK_SECONDS", STALL_TICK)
    pipe = _StubPipe(valve=STALL_VALVE, loop_seconds=30, dump=True)
    attachment = stall_alarm.attach_stall_alarm(pipe)
    stall_alarm.ensure_stall_alarm(attachment)
    state = attachment.state
    _wait_until(lambda: state.armed)
    dump_path, dump_fd, watcher = state.dump_path, state.dump_fd, state.watcher
    stall_alarm.detach_stall_alarm(attachment)
    watcher.join(2)
    assert _stall_watchers() == []

    spare: list[int] = []
    stray = None
    try:
        for _ in range(64):
            fd = os.open(dump_path + ".stray", os.O_RDWR | os.O_CREAT, 0o600)
            if fd == dump_fd:
                stray = fd
                break
            spare.append(fd)
            if fd > dump_fd:
                break
        assert stray is not None
        _hold_the_interpreter_lock(STALL_VALVE + 2 * STALL_TICK)
        assert os.fstat(stray).st_size == 0
    finally:
        for fd in spare + ([stray] if stray is not None else []):
            os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(dump_path + ".stray")


@pytest.mark.asyncio
async def test_a_stalled_event_loop_is_warned_about_with_its_stack_and_its_recovery_is_logged_once(
    monkeypatch, caplog
):
    monkeypatch.setattr(stall_alarm, "_STALL_BEAT_SECONDS", 0.01)
    logger = logging.getLogger("stall_alarm_loop_half")
    pipe = _StubPipe(logger)
    state = stall_alarm._StallAlarmState()
    attachment = stall_alarm._StallAttachment(pipe, state)
    state.attachments.append(attachment)
    stall_alarm._stall_ensure_beat(attachment)

    first_beat = attachment.beat_at
    await asyncio.sleep(0.1)
    assert attachment.beat_at > first_beat
    assert attachment.loop_thread == threading.get_ident()
    attachment.beat_task.cancel()

    def warnings() -> list[str]:
        return [r.getMessage() for r in caplog.records if r.name == logger.name]

    def evaluate(now: float) -> None:
        with state.lock:
            live = stall_alarm._stall_live_locked(state)
        stall_alarm._stall_check_loops(state, live, live[0][1], now)

    attachment.beat_at = 100.0
    with caplog.at_level(logging.WARNING, logger=logger.name):
        evaluate(101.0)
        assert warnings() == []

        evaluate(106.0)
        assert len(warnings()) == 1
        assert "test_a_stalled_event_loop_is_warned_about" in warnings()[0]

        evaluate(116.0)
        assert len(warnings()) == 1

        attachment.beat_at = 117.0
        evaluate(117.2)
        assert len(warnings()) == 2
        assert state.stalls == {}

        attachment.beat_at = 118.0
        evaluate(125.0)
        assert len(warnings()) == 2

        evaluate(170.0)
        assert len(warnings()) == 3
