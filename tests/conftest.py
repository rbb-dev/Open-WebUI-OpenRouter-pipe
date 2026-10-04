"""Test configuration helpers for unit tests."""

from __future__ import annotations

# First, so the environment defaults land before anything reads them. They live in
# owui_stubs alone: a second setdefault here would win in this process and lose in
# every probe subprocess, so retargeting DATA_DIR would move only half the harness.
import owui_stubs  # noqa: F401 - Open WebUI/sqlalchemy/tenacity stand-ins + env defaults

import contextlib
import os
import tempfile
import threading
from typing import Any

import asyncio
import base64
import json
import sys
import time
from collections import OrderedDict
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import pytest_asyncio


_PIPES_TO_CLOSE: list["Pipe"] = []


@pytest.fixture(autouse=True)
def _reset_site_default_memo():
    """Clear the site-default memo around every test, and never the user's own choice.

    The memo is process-wide, so without this it outlives the
    `monkeypatch.setattr(pipe_module, "_OwuiConfig", ...)` that the terminal parity
    matrix does per case: the second case would read a site default captured while the
    first case's stub was installed, and go red for the wrong reason in the very tests
    meant to certify the memo. Before *and* after, so a test cannot leave a value behind
    for a suite that never asked for it.
    """
    import open_webui_openrouter_pipe.pipe as pipe_module

    memo = getattr(pipe_module, "_TERMINAL_SITE_DEFAULT_MEMO", None)
    if isinstance(memo, dict):
        memo.clear()
    yield
    if isinstance(memo, dict):
        memo.clear()


@pytest.fixture(autouse=True, scope="session")
def _pipes_start_warm():
    """Every pipe the tree builds starts warm-up-complete, and the original is restored.

    The warm-up gate refuses a request while a warm-up has failed and none is in flight, and
    on a test that never stubs `_ping_openrouter` the warm-up fails for real -- a refused
    connection to a provider the test never wanted to reach. The gate is doing its job; the
    tests below it are simply not about warm-up, so 85 of them were red for a reason that had
    nothing to do with what they assert.

    The CANCEL is the load-bearing half, and it is here because of where the task is armed:
    `Pipe.__init__` ends by scheduling `_run_startup_checks` (`pipe.py:1009`), so setting the
    latch alone left a live `openrouter-warmup` task on the loop. Nothing awaits between that
    `__init__` arm and the statement below, so cancelling here always beats the task's first
    step -- and the task can only reach its `except` (`pipe.py:3158`) by awaiting the ping,
    which a cancelled task never does. Without the cancel the task ran on the first `await`
    inside whichever test held the pipe, and a failing ping set `_warmup_failed` on a pipe
    this guard had already declared warm: turn one was served, turn two came back 503, and
    which tests saw it depended on how the loop happened to schedule. `_startup_task` is
    cleared with it so no reader is left holding a reference to a task that is not running.

    `Pipe.__init__` is wrapped rather than each fixture patched, so a pipe built directly in
    a test body gets the same state as one from `pipe_instance`/`pipe_instance_async`, and the
    wrap is undone when the session ends. The wrapper keeps no reference to the pipes it saw:
    `test_hot_reload_lifecycle.py` asserts that a finished pipe is collectable, so a list of
    them would pin all of them and turn six arms that are not about warm-up red. A test that
    genuinely needs a cold pipe opts out by setting `_warmup_tests_may_refuse` on itself,
    which this guard reads; there are eight such tests.
    """
    import open_webui_openrouter_pipe.pipe as pipe_mod

    original = pipe_mod.Pipe.__init__

    def _warm_init(self, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        if getattr(self, "_warmup_tests_may_refuse", False):
            return
        task = getattr(self, "_startup_task", None)
        if task is not None:
            task.cancel()
        self._startup_task = None
        self._startup_checks_complete = True

    pipe_mod.Pipe.__init__ = _warm_init
    try:
        yield
    finally:
        pipe_mod.Pipe.__init__ = original


def _schedule_pipe_cleanup(pipe: "Pipe") -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(pipe.close())
    else:
        _PIPES_TO_CLOSE.append(pipe)


@pytest_asyncio.fixture(autouse=True)
async def _cleanup_pending_pipes():
    yield
    while _PIPES_TO_CLOSE:
        pipe = _PIPES_TO_CLOSE.pop()
        await pipe.close()


#: The key the two fixtures below hand a fresh `Pipe`. Every reader of `API_KEY` goes
#: through `Pipe._resolve_openrouter_api_key`, which refuses an unset one, so a keyless
#: fixture pipe refuses the request before it reaches the transport the test stubs --
#: and the assertions below that stub then read an empty turn rather than the turn under
#: test. Set on the valves rather than through `OPENROUTER_API_KEY`, which is the field's
#: `default_factory` and therefore a thing some tests read to assert the keyless state.
FIXTURE_API_KEY = "sk-test-key"


@pytest.fixture
def pipe_instance(request):
    """Return a fresh Pipe instance for tests."""
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr(FIXTURE_API_KEY)

    def _finalize() -> None:
        _schedule_pipe_cleanup(pipe)

    request.addfinalizer(_finalize)
    return pipe


@pytest_asyncio.fixture
async def pipe_instance_async():
    """Return a fresh Pipe instance for async tests with proper cleanup."""
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr(FIXTURE_API_KEY)
    yield pipe
    await pipe.close()


@pytest.fixture
def mock_request():
    """Mock FastAPI request used for storage uploads."""
    request = Mock()
    request.app = Mock()
    request.app.url_path_for = Mock(return_value="/api/v1/files/test123")
    return request


@pytest.fixture
def mock_user():
    """Mock user object used for uploads and storage context."""
    user = Mock()
    user.id = "user123"
    user.email = "test@example.com"
    user.name = "Test User"
    return user


@pytest.fixture
def sample_image_base64() -> str:
    """Return a 1x1 transparent PNG encoded as base64."""
    return (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
    )


@pytest.fixture
def sample_audio_base64() -> str:
    """Return sample base64-encoded audio data."""
    return base64.b64encode(b"FAKE_AUDIO_DATA").decode("utf-8")


@pytest.fixture
def config_rev_reader(monkeypatch):
    """Answer the Config tab's revision read at its own seam, from a revision the caller names.

    `actions._current_config_rev` reads one integer -- the stored `updated_at` -- and it
    reads it through `read_config_rev`, whose SELECT projects `function.updated_at` and
    the stored `function.valves` row (the digest of the latter is what the Config tab's
    live-update guard consults when an announcement carries no change identity) and
    nothing wider. `read_config_rev` closes over `open_webui.internal.db`, which no
    in-process test installs, so an unpatched Config-tab test sees `None`, every save
    looks stale, and every save takes the conflict arm.

    The revision is CALLER-SUPPLIED rather than derived from the caller's own
    `Functions.get_function_by_id` double. Deriving it would re-couple the two readers,
    and a test could no longer tell which one the code under test used -- which is the
    whole subject here, and the reason a suite already covering `read_config_rev` did
    not notice a second, wider reader of the same column. A test that wants the two to
    DISAGREE (and one that wants to count the narrow reads) writes its own reader; the
    two arms in `test_config_tab_secret_clear.py` do.

    Not autouse: an unasked-for revision is a revision no test chose, and the tests
    that are about an unreadable row need the opposite of this.
    """
    def _install(rev: Any) -> None:
        from open_webui_openrouter_pipe.plugins.pipe_dashboard import actions

        async def _read(pipe_id: str) -> tuple[Any, str | None]:
            return (rev() if callable(rev) else rev), None

        monkeypatch.setattr(actions, "read_config_rev", _read, raising=False)

    return _install



def _maybe_install_bundled_pipe() -> None:
    """Optionally preload a generated monolith bundle for testing.

    Set `OWUI_PIPE_BUNDLE_PATH` to a bundled .py file (e.g.
    open_webui_openrouter_pipe_bundled.py or open_webui_openrouter_pipe_bundled_compressed.py)
    to run the test suite against the single-file package import-hook implementation.
    """
    bundle_path = os.environ.get("OWUI_PIPE_BUNDLE_PATH")
    if not bundle_path:
        return

    prefix = "open_webui_openrouter_pipe"
    for name in list(sys.modules):
        if name == prefix or name.startswith(prefix + "."):
            sys.modules.pop(name, None)

    path = Path(bundle_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"OWUI_PIPE_BUNDLE_PATH does not exist: {path}")

    import importlib.util

    spec = importlib.util.spec_from_file_location("owui_pipe_bundle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for bundle: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["owui_pipe_bundle"] = module
    spec.loader.exec_module(module)


_maybe_install_bundled_pipe()

from open_webui_openrouter_pipe import EncryptedStr, Pipe
from open_webui_openrouter_pipe.core.circuit_breaker import CircuitBreaker
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry, ModelFamily
from tests.pipe_limits import reset_all_slots


#: Real paths the session was asked to collect, when every argument is a plain file and
#: no selection filter is in play. Empty means the guard stands down. Resolved once in
#: `pytest_configure` because the decision depends only on the command line.
_WANTED_FILES: set[str] = set()


def _decide_fast_path(config) -> bool:
    """Whether narrowing collection to the named files is safe for this invocation.

    Stand down unless ALL of: at least one argument, every argument's path part is an
    existing file, and neither `-k` nor `-m` is in play. A directory argument means "all
    of this", and a selection filter changes which tests RUN -- declining files there
    would drop tests the caller asked to keep. `::` is stripped because a node id is a
    file plus a selector.
    """
    paths = []
    for arg in config.args:
        path = arg.split("::", 1)[0]
        if path not in paths:
            paths.append(path)
    if not paths:
        return False
    if any(not os.path.isfile(path) for path in paths):
        return False
    if config.getoption("-k", default=None) or config.getoption("-m", default=None):
        return False
    return True


@pytest.hookimpl(trylast=True)
def pytest_configure(config):
    if _decide_fast_path(config):
        _WANTED_FILES.update(
            os.path.realpath(arg.split("::", 1)[0]) for arg in config.args
        )


@pytest.hookimpl(tryfirst=True)
def pytest_ignore_collect(collection_path, config):
    """Skip files the session did not ask for, before a collector is built for them.

    `_pytest.main.Session._collect_one_node` consults `self._collection_cache` only when
    `handle_dupes` is true, and pytest computes that as `not (len(matchparts) == 1 and
    isinstance(matchparts[0], Path) and matchparts[0].is_file())` -- False for a bare file
    argument. So each file named on the command line re-walks the whole `tests/` tree,
    building one `Module` node per file and keeping one. Declining the rest here is free:
    this hook is consulted before a collector is constructed, where `pytest_collect_file`
    would build the node first and decline it afterwards (measured slower than no guard).
    """
    if _WANTED_FILES and os.path.isfile(str(collection_path)):
        return os.path.realpath(str(collection_path)) not in _WANTED_FILES
    return None


_WARN_LATCH_PREFIX = "_warned"
_PIPE_MARKER_NAME = "OWUI_OPENROUTER_PIPE_MARKER"


def _warn_latches() -> dict[str, set | dict | list]:
    """Every module-level warn-once latch the pipe owns, by package name OR by marker.

    These suppress a warning for the life of the process, so the first test to trip
    one silently disarms every later assertion that the warning is emitted. Resolved
    fresh each call because bundled modes alias submodule names onto one module.

    Two owners, one sweep. The package-name branch is the one this file started with.
    The marker branch is the one the routing filter needs: a RENDERED filter is a module
    Open WebUI loaded from a Functions row, and it names it `function_<id>`, so the
    package-name branch never saw its latches at all -- whatever their spelling. Keying
    on the module NAME could not fix that, because the name is Open WebUI's, not ours.
    `OWUI_OPENROUTER_PIPE_MARKER` is the codebase's own ownership predicate (every
    renderer emits it, and `_installed_by` and `_row_owner` already read it off stored
    content), it names exactly this pipe's own generated modules and nothing else, and it
    travels with the module rather than depending on how the host happened to import it.
    Keying on `MODEL_SLUG` instead was measured and cannot work: a bare model slug is not
    a module identity, and many rows share one.
    """
    seen: dict[str, set | dict | list] = {}
    # Deduped by module OBJECT: this runs autouse before all ~5950 tests, and in a flat
    # bundle 107 submodule names alias 2 module objects, so scanning per name re-walked
    # the same namespace 107 times -- 6.3 ms a call, ~37 s per bundled CI run, for an
    # identical set of latches.
    scanned: set[int] = set()
    for name, module in list(sys.modules.items()):
        if id(module) in scanned:
            continue
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        owned_by_name = name == "open_webui_openrouter_pipe" or name.startswith(
            "open_webui_openrouter_pipe."
        )
        if not owned_by_name and not namespace.get(_PIPE_MARKER_NAME):
            continue
        scanned.add(id(module))
        for attr, value in list(namespace.items()):
            if attr.startswith(_WARN_LATCH_PREFIX) and isinstance(value, (set, dict, list)):
                seen.setdefault(f"{name}.{attr}", value)
    return seen


_CACHE_OWNER_PREFIX = "open_webui_openrouter_pipe"


def _package_modules() -> list:
    """Every package module object currently in sys.modules, once each.

    Deduped by OBJECT, not by name, for the reason `_warn_latches` records: a flat
    bundle aliases 107 dotted names onto 2 module objects, so scanning per name walked
    the same namespace 107 times for an identical set. Nothing is imported by name
    here -- conftest's module scope sits at line 128, where the package is only partly
    imported, and a name import would not resolve in a bundle anyway.
    """
    out, scanned = [], set()
    for name, module in list(sys.modules.items()):
        if name != _CACHE_OWNER_PREFIX and not name.startswith(_CACHE_OWNER_PREFIX + "."):
            continue
        if not isinstance(module, ModuleType):
            continue
        if id(module) in scanned:
            continue
        scanned.add(id(module))
        out.append(module)
    return out


def _package_cached_functions() -> list:
    """Every cache the package owns, as the wrapped functions the sweep will clear.

    The predicate lives HERE and nowhere else, because the census in
    test_unlinkable_chat_prefixes_isolation.py asserts against this same list. Written
    twice, the census would test a copy: narrowing the sweep to a name list would leave
    the copy intact and the census green over a cache nothing resets.

    Ownership is by `__globals__` IDENTITY rather than by name. A wrapped function
    defined inside one of the package's own module namespaces belongs to the package
    however it is spelled; the other spelling, a `__module__` prefix test, is inert in
    the flat bundle, where every function in the body carries the host module's name,
    and clears zero caches in two of the five CI modes. The identity form holds in all
    five. It is also what leaves `urllib.parse.urlsplit` alone: that IS an lru_cache,
    bound into two package namespaces, but it is interpreter state that every library
    in the process shares.
    """
    modules = _package_modules()
    found = []
    for module in modules:
        for value in list(vars(module).values()):
            clear = getattr(value, "cache_clear", None)
            if not (callable(clear) and callable(getattr(value, "cache_info", None))):
                continue
            wrapped = getattr(value, "__wrapped__", None)
            namespace = getattr(wrapped, "__globals__", None)
            if namespace is None or not any(namespace is vars(m) for m in modules):
                continue
            found.append((value, wrapped, module))
    return found


def _clear_package_caches() -> None:
    for value, _wrapped, _owner in _package_cached_functions():
        value.cache_clear()


def _package_logger():
    """The logger every open_webui_openrouter_pipe.* record ends at.

    Read from sys.modules rather than named, so the bundled runs -- where the package
    is imported under a different top-level name -- guard the logger they actually use
    instead of an empty stand-in that is always healthy.
    """
    import logging as _logging

    pkg = sys.modules.get("open_webui_openrouter_pipe")
    root_name = (getattr(pkg, "__name__", "") or "open_webui_openrouter_pipe").split(".")[0]
    return _logging.getLogger(root_name)


def _package_logger_emits(logger) -> bool:
    """Whether a record reaching this logger can still get out of it.

    A NullHandler is not a sink. Counting one would satisfy "has a handler" while
    emitting nothing, which is the exact shape that made the original failure silent;
    test_a_test_cannot_leave_the_package_logger_deaf asserts the same predicate.
    """
    import logging as _logging

    return any(
        not isinstance(handler, _logging.NullHandler) for handler in logger.handlers
    )


def _repair_package_logger() -> bool:
    """Make the package logger able to emit again. True if it could not.

    With no handler that can emit, the logger has to propagate or the records die on
    it -- so propagation is the repair, and adding a handler is not.
    """
    logger = _package_logger()
    if _package_logger_emits(logger) or logger.propagate:
        return False
    logger.propagate = True
    return True


#: Node ids whose test left a non-daemon thread running, collected during the run and
#: reported once at session finish. See `pytest_runtest_protocol` for why a per-test
#: timeout cannot see this at all.
_LEAKED_THREADS: list[str] = []


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Name the test that left a non-daemon thread running for the interpreter to join.

    The timeout the suite now arms bounds a hung TEST. This is the hang that follows the
    last one: `concurrent.futures.thread._python_exit` is registered through
    `threading._register_atexit` and joins every worker with `t.join()`, by which point
    `SIGALRM` has been reset to `SIG_DFL` and the plugin's timer thread is gone. A
    `ThreadPoolExecutor` worker is non-daemon, so a test that starts one and never shuts
    it down hands the run a process that cannot exit -- the symptom is a wall, and the
    cause is a test that passed.

    The post-yield body runs after every finalizer for the item, which is the first moment
    a thread the test started can be told apart from one it joined. Threads that existed
    before are recorded by identity, so an idle pool carried over from an earlier test is
    not attributed to this one. The names accumulate and are reported once, at session
    finish: failing each item here would attribute the damage to whichever test happened
    to run next, which is the failure mode this exists to remove.
    """
    before = {t.ident for t in threading.enumerate()}
    try:
        result = yield
    except BaseException:
        _record_leaked_threads(item, before)
        raise
    _record_leaked_threads(item, before)
    return result


def _record_leaked_threads(item, before: set) -> None:
    for thread in threading.enumerate():
        if thread.ident not in before and not thread.daemon:
            _LEAKED_THREADS.append(
                f"{item.nodeid} left non-daemon thread {thread.name!r} running"
            )


def pytest_sessionfinish(session, exitstatus):
    """Fail the session, naming every test that left a non-daemon thread running.

    Reported here rather than per item because the thread outlives the test that made it:
    the interpreter reaches `threading._shutdown()` long after the last protocol, and a
    failure raised inside an item would be repaired or cancelled by that item's own
    teardown long before anything joined it.
    """
    if not _LEAKED_THREADS:
        return
    report = session.config.get_terminal_writer()
    report.line("")
    report.line(
        f"{len(_LEAKED_THREADS)} test(s) left a non-daemon thread running:", red=True
    )
    for name in _LEAKED_THREADS:
        report.line(f"  {name}", red=True)
    report.line(
        "The interpreter joins non-daemon threads at exit (concurrent.futures.thread."
        "_python_exit -> t.join()), after the timeout's alarm has been reset, so a "
        "leaked thread hangs the run rather than failing a test. Shut the executor down, "
        "or join the thread, in the test named above.",
        red=True,
    )
    session.exitstatus = pytest.ExitCode.TESTS_FAILED


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item):
    """Name the test that left the package logger deaf, at the only point that can see it.

    A fixture finalizer cannot: `monkeypatch`'s undo is ordered after this fixture's,
    so a value it restores lands last and the guard never observes it. This wrapper's
    post-yield body runs after EVERY finalizer for the item, which is the first moment
    the logger's final state for that test is knowable.

    Without it the damage surfaces on some later test, which then fails for a reason
    that has nothing to do with it -- or, when the deafness only makes an
    `assert not caplog.records` vacuous, surfaces on nothing at all.
    """
    try:
        result = yield
    except BaseException:
        _repair_package_logger()
        raise
    if _repair_package_logger():
        pytest.fail(
            f"{item.nodeid} left the package logger with no handler that can emit and "
            "propagate=False, so every open_webui_openrouter_pipe record after it is "
            "silently discarded and any later `assert not caplog.records` passes for "
            "the wrong reason. It has been repaired for the tests that follow. The "
            "usual cause is a `monkeypatch.setattr(..., 'propagate', False)` taken "
            "when propagate was ALREADY False: the snapshot is the value being "
            "written, and the undo -- ordered after the restore fixture -- writes it "
            "back last.",
            pytrace=False,
        )
    return result


@pytest.fixture(autouse=True)
def _restore_package_logger():
    """Leave the package logger able to emit, whatever a test did to it.

    get_logger sets propagate=False, so a test that clears the handlers leaves the
    package root with no handlers AND no propagation: every later
    open_webui_openrouter_pipe.* record dies there, and any later
    `assert not caplog.records` passes for the wrong reason. Under the old
    propagate=True the same teardown was harmless.

    REPAIRS rather than replays, at BOTH ends. Teardown alone cannot hold the
    invariant: this fixture's finalizer is not the last write, because a `monkeypatch`
    undo is ordered after it and puts back whatever the test snapshotted. Repairing
    again at setup is what no finalizer ordering can defeat -- whatever the previous
    test left behind, the next one starts able to emit. The snapshot is taken after
    that repair, so the teardown cannot reinstate a state the setup rejected.

    Neither end adds a handler. A NullHandler is not a sink -- adding one is what
    makes this failure silent -- which is the same predicate
    test_a_test_cannot_leave_the_package_logger_deaf asserts.
    """
    logger = _package_logger()
    _repair_package_logger()
    saved_handlers = list(logger.handlers)
    saved_filters = list(logger.filters)
    saved_propagate = logger.propagate
    saved_level = logger.level
    try:
        yield
    finally:
        logger.handlers[:] = saved_handlers
        logger.filters[:] = saved_filters
        logger.setLevel(saved_level)
        logger.propagate = saved_propagate if _package_logger_emits(logger) else True


@pytest.fixture(autouse=True)
def _session_log_level_debug():
    """Let tests observe the pipe's DEBUG records.

    The package logger does not propagate; a forwarding handler re-emits to the host
    under SessionLogger.log_level, which defaults to INFO so an operator's LOG_LEVEL
    actually suppresses debug output in production. caplog attaches to the root
    logger, so without raising the threshold here every DEBUG assertion in the suite
    would be asserting on an empty capture. This is the test-side equivalent of an
    operator setting LOG_LEVEL=DEBUG -- the gate itself is guarded by
    test_the_host_forwarder_honours_the_session_log_level.

    Any future level-sensitive test hits this wall: it has to set
    `SessionLogger.log_level` in its own fixture and reset it in a `finally`, or
    the `effective_log_level()` it reads is DEBUG whatever the process floor says.
    """
    import logging as _logging

    from open_webui_openrouter_pipe.core.logging_system import SessionLogger

    token = SessionLogger.log_level.set(_logging.DEBUG)
    try:
        yield
    finally:
        SessionLogger.log_level.reset(token)


REBINDABLE_FILE_ACCESSOR_MODULES = (
    "open_webui_openrouter_pipe.integrations.video",
    "open_webui_openrouter_pipe.requests.orchestrator",
    "open_webui_openrouter_pipe.storage.owui_files",
)
FILE_ACCESSOR_NAMES = ("get_file_by_id", "infer_file_mime_type")


@pytest.fixture(autouse=True)
def _restore_rebound_file_accessors():
    """Put back the file accessors a test swapped on a module, however it swapped them.

    Several media-relay tests rebind `get_file_by_id` and `infer_file_mime_type` directly
    on a module rather than through monkeypatch, to a stub that raises for any id it does
    not know. Under the package layout that only affects the one module and the damage
    stays local. Under the flat bundle every submodule is the same object, so the
    replacement stands for the rest of the run and the next production caller of
    `get_file_by_id` trips over it -- which is exactly what happened when the stored-file
    budget added one: two of its tests passed alone and failed after those files loaded.
    """
    import open_webui_openrouter_pipe.storage.owui_files as _files

    import importlib

    modules = [importlib.import_module(name) for name in REBINDABLE_FILE_ACCESSOR_MODULES]
    names = FILE_ACCESSOR_NAMES
    saved = [
        (module, name, getattr(module, name))
        for module in modules
        for name in names
        if hasattr(module, name)
    ]
    assert len(saved) == len(modules) * len(names), (
        "_restore_rebound_file_accessors captured "
        f"{len(saved)} of {len(modules) * len(names)} accessors; a module that does not "
        "carry one is silently skipped and never restored"
    )
    yield
    for module, name, original in saved:
        if getattr(module, name, None) is not original:
            setattr(module, name, original)
    del _files


@pytest.fixture(autouse=True)
def _reset_stub_chat_files():
    """The chat_file row store is class state; without this it leaks across tests."""
    chats = getattr(sys.modules.get("open_webui.models.chats"), "Chats", None)
    store = getattr(chats, "_chat_files", None)
    if store is not None:
        store.clear()
    yield


@pytest.fixture(autouse=True)
def _reset_auth_failure_state():
    """Clear the auth-failure pause, which is class state with a 60-second life.

    Rendering a sign-in-failure card records the pause for every pipe instance in the
    process, and while it holds, task-model calls are skipped. Without this reset a
    test that renders such a card silently changes what a later test's background
    task returns -- a title test reads the fallback title and fails, far from the
    test that caused it. The reap stamp goes with it: it says when the map was last
    swept, so leaving it at one test's clock would suppress the next test's sweep --
    and a test that walks its own clock backwards to expire a scope would then keep
    every entry it meant to have dropped.
    """
    CircuitBreaker._AUTH_FAILURE_UNTIL.clear()
    CircuitBreaker._AUTH_FAILURE_SWEPT_AT = 0.0
    yield


@pytest.fixture(autouse=True)
def _reset_process_semaphores():
    """Drop the process-wide concurrency semaphores, which every pipe in the worker shares.

    A permit taken on one test's event loop is returned only when that loop runs the
    generation's `async with` exit. A later test on a new loop that reuses the same
    semaphore can wait for a permit that never comes back
    (`tests/test_video_generation.py` then `tests/test_api_call_video.py` hung that way).

    Every pool needs it, not just the video one: the slots live in a holder keyed by
    pipe id that outlives the test that filled it, so a request, tool or panel permit leaked
    here hangs the next test the same way a video permit did.
    """
    reset_all_slots()
    yield


@pytest.fixture(autouse=True)
def _reset_package_caches():
    """Clear every `lru_cache` the package owns, at BOTH ends of every test.

    A process-wide cache is one worker's answer, not this test's. `_unlinkable_chat_prefixes`
    is keyed on nothing and lives for the life of the process, so a test that published a
    longer prefix list decided `is_temporary_chat` for every test that ran after it in the
    same worker -- and `temporary_chat_prefixes()` drops only `channel:`, so a real
    conversation became a temporary chat, its uploads were handled as though the chat did
    not exist, and `core/costs.py` wrote its snapshot naming neither the chat nor the
    message. One test module reset that cache for its own tests and nothing else.

    Cleared by OWNERSHIP rather than by name, so the strict-schema memo is covered too and
    `urllib.parse.urlsplit` is not: see `_package_cached_functions`.

    BOTH ends. The teardown end is what stops an arming from poisoning the next test, and
    the setup end is what discards anything a finalizer ordering put back afterwards.
    Measured, single-end variants pass every test in that file -- with the whole fixture
    removed they do too -- so the second end is the established precedent rather than a
    shape a test distinguishes. Kept separate from `_reset_warn_latches` rather than
    folded into it: the two guard unrelated invariants, and folding them would couple a
    cache clear to a latch clear for ~2 ms a test.

    Every published failure count for the arms that test this fixture is SERIAL. CI
    (`.github/workflows/verify.yml`) runs `pytest tests -q` with no `-n`; a `-n 4` run
    is a weaker witness, not a green one, because arms 1-3 arm in one test and read in
    a later one.
    """
    _clear_package_caches()
    yield
    _clear_package_caches()


def _package_process_state_containers() -> tuple[tuple[str, str], ...]:
    """The package-owned containers whose lifetime is the process, as (module, name).

    Written as one predicate rather than two resets, so the census in
    test_unlinkable_chat_prefixes_isolation.py asserts against this same list.
    Written twice, the census would test a copy and a name added to one half would
    go unseen.

    Two containers, and the census is scoped to exactly these on purpose. A module
    name gives no signal about lifetime -- `_cache` and `ALLOWED_OPENROUTER_FIELDS`
    are both module-level dicts and only one of them is per-test state -- so the
    wider sweep a name-based list would need is not mechanically decidable from the
    source. A container that is really process-lifetime and really per-test state
    gets its own item and its own reach analysis.
    """
    return (
        ("open_webui_openrouter_pipe", "_cache"),
        (
            "open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_socket",
            "_pending_emits",
        ),
    )


def _clear_package_process_state() -> None:
    """Return both process-lifetime containers to what a freshly-imported tree has.

    `_cache` and the module namespace are cleared TOGETHER, and the second half is
    not optional. `__getattr__` writes `_cache[name]` and then `globals()[name]`, so
    emptying the dict alone leaves a stand-in bound in the module namespace --
    where `__getattr__` is never reached for it, because module-level lookup finds
    it first. Every such write in `__init__.py` is preceded by its `_cache` write
    (`__init__.py:385-386, 392-393, 412-413, 434-435`), so `keys(_cache)` is a sound
    upper bound on what `__getattr__` installed, and popping both is safe.

    Unbinding a SUCCESSFUL lazy import does not produce a new object on the next
    read: `importlib.import_module` finds the entry in `sys.modules` and re-binds
    the same one, so `Valves` is still the same class after a reset.

    Resolved FRESH on every call, never captured at module scope, because a test may
    swap `sys.modules[PKG]` for a stand-in and assert identity on what it put back.

    Guard-shaped, never assert-shaped, because four of the five CI modes load a flat
    bundle that synthesises `__all__ = ["Pipe"]` and has neither `_cache` nor a
    `__getattr__` (scripts/bundle_v2.py:1140-1152). There is nothing to clear in
    those modes; the guards are what make that a no-op rather than an error.
    """
    from open_webui_openrouter_pipe.logging.session_log_manager import (
        _LIVE_TURNS_HOLDER_KEY,
    )

    sys.modules.pop(_LIVE_TURNS_HOLDER_KEY, None)

    for module_name, attr in _package_process_state_containers():
        module = sys.modules.get(module_name)
        container = getattr(module, attr, None) if module is not None else None
        if isinstance(container, dict):
            for key in list(container):
                container.pop(key, None)
                vars(module).pop(key, None)
        elif isinstance(container, set):
            # Cancel before clearing. asyncio holds only a WEAK reference to a
            # running task, so dropping the last strong one would let the collector
            # take a fire-and-forget emit out mid-flight -- the set exists precisely
            # to prevent that. Cancelling is safe here because this runs at the
            # TEARDOWN of the test that armed the emit, while its loop is still open
            # (measured: pytest-asyncio closes a loop after the autouse fixtures
            # tear down). It is not safe in general -- cancelling a task already
            # suspended on a future, or one whose first step never ran, schedules
            # the next step on the loop and raises `RuntimeError: Event loop is
            # closed` if that loop has closed. Hence cancel-first rather than a bare
            # clear, and hence the arm in
            # test_a_package_owned_container_is_empty_at_the_start_of_every_test.py
            # builds the one shape that leaks without making this raise.
            for task in list(container):
                if not task.done():
                    task.cancel()
            container.clear()


@pytest.fixture(autouse=True)
def _reset_package_process_state():
    """Reset the two package containers that live for the life of the process.

    Both ends, for the reason `_reset_package_caches` gives: the teardown end stops an
    arming from poisoning the next test, the setup end discards whatever a finalizer
    ordering put back afterwards. Autouse fixtures are set up before `monkeypatch` and
    torn down after it, so a test that does `monkeypatch.setattr(pkg, "UserValves", ...)`
    is restored by the time the teardown end runs and cannot leave the fake behind.

    Kept separate from `_reset_warn_latches` rather than folded into it: the two
    guard unrelated invariants, and folding them would couple a state clear to a
    latch clear for ~2 ms a test. `_warned` is also the exact set equality
    test_warn_latch_isolation.py:155 asserts, so widening that predicate to reach a
    container with no such prefix would fail it.
    """
    _clear_package_process_state()
    yield
    _clear_package_process_state()


@pytest.fixture(autouse=True)
def _reset_warn_latches():
    """Reset every warn-once latch that is currently loaded.

    A latch suppresses its warning for the life of the process, so without this the
    first test to trip one silently disarms every later assertion that it is emitted.
    Which latches exist is asserted in test_warn_latch_isolation.py rather than here:
    a module that no test imported has no latch to reset, so a per-test assertion on
    the full set fails for reasons that have nothing to do with isolation.

    The two memos beside them are the same kind of state for the same reason. One holds
    downloaded image bytes so a picture reused across turns is fetched once; the other
    holds a payload's base64 decode verdict so a stored picture is not re-validated every
    turn. Both answer for the life of the worker by design, and across tests they must not:
    the first test to reach a payload would satisfy every later assertion that it was
    reached, which is a decoder mock that stops being called.
    """
    latches = _warn_latches()
    for latch in latches.values():
        latch.clear()
    from open_webui_openrouter_pipe.requests import transformer as _transformer

    memo = getattr(_transformer, "_reuse_download_memo", None)
    if memo is not None:
        memo.clear()
    verdicts = getattr(_transformer, "_validate_inline_memo", None)
    if verdicts is not None:
        verdicts.clear()
    _clear_stub_task_models()
    yield


#: Module-level accumulators the package owns that no other fixture resets. Resolved by
#: module OBJECT through `_package_modules()`, never by dotted name: a flat bundle aliases
#: 107 submodule names onto two module objects, and a name import does not resolve in a
#: bundle at all -- the same reasoning `_warn_latches` records.
_ACCUMULATOR_NAMES = ("_PIPE_OFF_LANDED_AT", "_REFUSED_FILTER_WRITES", "_PIPE_OFF_SETTLING")


def _package_accumulators() -> list:
    """Every package-global accumulator currently loaded, as the live containers.

    These are state that accumulates FOR the life of a worker by design -- `_PIPE_OFF_LANDED_AT`
    is when the pipe's own switch-off write landed, `_REFUSED_FILTER_WRITES` is the rows
    Open WebUI refused -- so inside a worker they must survive. Across tests they must not:
    the first test to write one silently decides what every later test in that worker reads.
    `_PIPE_OFF_LANDED_AT` is read on the live path by `_pipe_owns_the_off`, which uses it to
    tell the pipe's own switch-off from an administrator's later edit, so a stale entry can
    make a later test read a row as retired when it was not.
    """
    out = []
    for module in _package_modules():
        for name in _ACCUMULATOR_NAMES:
            value = getattr(module, name, None)
            if isinstance(value, set | dict | list):
                out.append(value)
    return out


@pytest.fixture(autouse=True)
def _reset_filter_pass_accumulators():
    """Clear the filter retire pass's module state around every test.

    A SEPARATE fixture from `_reset_warn_latches`, not folded into it: the two guard
    unrelated invariants, and folding them would couple a cache clear to a latch clear.
    About twenty test files already clear these two by hand, which is the coupling this
    removes -- each of those is a place the next file had to remember. What is swept is
    asserted in test_a_retired_filter_pass_state_is_reset_between_tests.py, which also
    carries the static census, so the next accumulator is caught where it is added rather
    than by a reader rediscovering it.
    """
    accumulators = _package_accumulators()
    for accumulator in accumulators:
        accumulator.clear()
    yield
    for accumulator in accumulators:
        accumulator.clear()


_DASHBOARD_SOCKET_MODULES = (
    ("open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_socket",
     ("_pipe_getters", "_registered", "_resync")),
    ("open_webui_openrouter_pipe.plugins.pipe_dashboard.http_routes",
     ("_routes_get_pipes",)),
    ("open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_publisher",
     ("_pd_snapshot_getters",)),
)
# The three dicts are the per-pipe REGISTRIES: `Pipe.id -> getter`. They are cleared
# in place at both ends of every test and never saved and restored -- a saved dict is
# the same object the test mutated, so restoring it would put the arming straight back.
_DASHBOARD_SOCKET_CONTAINERS = ("_pipe_getters", "_routes_get_pipes", "_pd_snapshot_getters")
_DASHBOARD_SOCKET_UNSET = {"_registered": False, "_resync": False}


def arm_dashboard_pipe(pipe: Any, *, get_pipe: Any = None, snapshot_getter: Any = None) -> str:
    """Register `pipe` on all three dashboard bindings, keyed by its own `Pipe.id`.

    The bindings are per-pipe REGISTRIES, so a test arms them by REGISTERING rather than
    by assigning: `dashboard_socket._pipe_getters[pid] = getter`. Two calls with the same
    id and different instances model a hot reload, two calls with different ids model two
    installed copies -- which is how the reload tests and
    `tests/test_two_installed_copies_do_not_share_a_dashboard.py` end up running ONE code
    path, so a bug in the fix shows up in both rather than in one of them.

    `get_pipe` and `snapshot_getter` default to the pipe itself, which is right for a test
    that only needs the id resolved. A test with a getter of its own passes it, because
    the getters are what the identity guards compare by identity.
    """
    pipe_id = str(getattr(pipe, "id", "") or "")
    getter = get_pipe if get_pipe is not None else (lambda: pipe)
    for module_name in (
        "open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_socket",
        "open_webui_openrouter_pipe.plugins.pipe_dashboard.http_routes",
        "open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_publisher",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if module_name.endswith("dashboard_socket"):
            module._pipe_getters[pipe_id] = getter
        elif module_name.endswith("http_routes"):
            module._routes_get_pipes[pipe_id] = getter
        elif snapshot_getter is not None:
            module._pd_snapshot_getters[pipe_id] = snapshot_getter
    return pipe_id


def _clear_dashboard_socket_containers() -> None:
    """Empty every per-pipe dashboard registry the tree has loaded.

    `.clear()`, never `setattr(module, name, {})`: replacing the dict would swap the
    object the production code and any binding installed through it hold, so a value
    written before the swap would be lost from one side and kept on the other. And never
    `setattr(module, name, None)` -- the unkeyed shape put `None` there, and a module
    whose registry is `None` raises `AttributeError` mid-test, which reads like a bug in
    the code under test rather than like a broken fixture.
    """
    for name, attrs in _DASHBOARD_SOCKET_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        for attr in attrs:
            if attr not in _DASHBOARD_SOCKET_CONTAINERS:
                continue
            container = getattr(module, attr, None)
            if isinstance(container, dict):
                container.clear()


@pytest.fixture(autouse=True)
def _reset_dashboard_socket_state():
    """Repair the dashboard's per-pipe registries and process-wide flags around every test.

    The dashboard binds three registries keyed by `Pipe.id` -- `_pipe_getters` in
    `dashboard_socket`, `_routes_get_pipes` in `http_routes`, `_pd_snapshot_getters` in
    `dashboard_publisher` -- and holds two process-wide flags beside them, `_registered`
    and `_resync` in `dashboard_socket`. The bindings are keyed so two installed copies
    of the pipe each reach their own admin; the flags stay process-wide because
    `_registered` guards one `sio.on(SUB_EVENT, ...)` for the process and `_resync` is a
    tick signal carrying no viewer state or data.

    Nothing in the package clears any of them, which is right for a live worker -- a
    socket gate must stay bound to the pipe it was bound to -- and wrong for a test
    process, where the bound pipe is drained and closed by the time the next test runs.
    This fixture repairs all five at BOTH ends of every test, and no test module restores
    any of them itself: a test that writes one of these is relying on the repair below,
    not on a fixture of its own. `tests/test_dashboard_socket_isolation.py` arms and
    reads; it is the witness, and
    `tests/test_a_test_module_does_not_restore_what_conftest_resets.py` is the census that
    keeps a per-file save/restore from being added back.

    The three registries are in scope together because ONE call
    (`PipeDashboardPlugin._re_register_registrations`) writes all three, and every
    `on_shutdown` reads back all three. Repairing one or two leaves the rest armed for
    the same test, which is the shape a two-installed-copies test would read as a leak.

    `_teardown_epoch` is deliberately NOT reset. `set_pipe_getter` and
    `clear_fresh_dispatch` both increment it, and its whole job is to be monotonic ACROSS
    a teardown: zeroing it per test makes a reconcile arm computed under one test read as
    un-armed under the next, and
    `tests/test_a_reconcile_that_overlaps_a_teardown_publishes_nothing.py` depends on the
    real semantics. Resetting it to `True`-ish is the other trap: `_registered` is
    restored, never re-armed.

    The two flags are saved and restored at teardown; the three registries are only
    cleared. A bundle may have set `_registered = True` at import and clobbering that
    breaks the bundle tier, while a restored registry would be the very object the test
    just wrote to. BOTH ends, as `_reset_package_caches` records: the setup end discards
    whatever a finalizer ordering put back, the teardown end stops this test's arm
    poisoning the next.

    Resolved through `sys.modules` and skipped when absent, the shape
    `_reset_stub_chat_files` already uses: a no-plugins bundle omits
    `plugins/pipe_dashboard` entirely, and a bare import here would be a collection
    error in the one mode that has no dashboard to reset.

    The three registries are containers, so the census in
    `tests/test_module_state_census.py` can see them; `_teardown_epoch`, `_fresh_dispatch`
    and the two flags are not, and that gap is the split this docstring records. The
    census reaches them because their names appear literally below, not because they are
    listed in `_MODULE_STATE_CONTAINERS` -- a name in both lists is the duplicate that
    census exists to name.
    """
    saved = [
        (sys.modules[name], attr, getattr(sys.modules[name], attr))
        for name, attrs in _DASHBOARD_SOCKET_MODULES
        if name in sys.modules
        for attr in attrs
        if attr not in _DASHBOARD_SOCKET_CONTAINERS
    ]
    for name, attrs in _DASHBOARD_SOCKET_MODULES:
        module = sys.modules.get(name)
        if module is None:
            continue
        for attr in attrs:
            if attr in _DASHBOARD_SOCKET_CONTAINERS:
                continue
            setattr(module, attr, _DASHBOARD_SOCKET_UNSET[attr])
    _clear_dashboard_socket_containers()
    yield
    for module, attr, original in saved:
        setattr(module, attr, original)
    _clear_dashboard_socket_containers()


_MODULE_STATE_CONTAINERS = (
    ("open_webui_openrouter_pipe.core.logging_system", "_ARCHIVE_CLAIMS"),
    ("open_webui_openrouter_pipe.core.valve_salvage", "_VALVE_SCHEMA_CACHE"),
    ("open_webui_openrouter_pipe.filters.filter_manager", "_PIPE_OFF_LANDED_AT"),
    ("open_webui_openrouter_pipe.filters.filter_manager", "_REFUSED_FILTER_WRITES"),
    ("open_webui_openrouter_pipe.filters.filter_manager", "_PIPE_OFF_SETTLING"),
    ("open_webui_openrouter_pipe.plugins.pipe_dashboard.actions", "_rate_state"),
    ("open_webui_openrouter_pipe.plugins.pipe_dashboard.actions", "_config_write_locks"),
    ("open_webui_openrouter_pipe.plugins.pipe_dashboard.http_routes", "_coarse_state"),
    ("open_webui_openrouter_pipe.plugins.pipe_dashboard.usage_queries", "_UQ_MEMO"),
)


@pytest.fixture(autouse=True)
def _reset_module_state_containers():
    """Empty every module-level container the package writes, at BOTH ends of every test.

    A container is one worker's answer, not this test's, and the package never clears
    them -- right for a live worker, wrong for a test process. The eight here are the
    rows `tests/test_module_state_census.py` lists as reset rather than exempt: the
    archive claim table, the stored-valve schema cache, the two filter-manager
    latches, both dashboard rate limiters, the per-pipe config write
    locks, and the usage query memo. The timing buffer was the ninth until the
    timing profiler's per-request in-memory copy was deleted with its two readers
    (B680, T967); the timing logger keeps no module-level container at all now, so
    the row went with the thing it named.

    `_coarse_state` and `_rate_state` are the pair the census exists for. Both key on
    `time.monotonic()`, and the suite's stub clocks run at 1000.0 -- hours behind the
    real clock -- so an entry armed under the real clock and read under a stub makes
    `now - last` NEGATIVE, the "too soon" branch is taken, and the FIRST call for a
    user this worker has never served is refused. Nothing goes red: the limiter simply
    declines work for a reason its caller cannot see.
    `tests/test_a_module_state_does_not_outlive_its_test.py` drives that arm.

    BOTH ends, for the reason `_reset_package_caches` records: the teardown stops this
    test's arm poisoning the next one, and the setup discards anything a finalizer
    ordering put back afterwards.

    Deliberately NOT cleared by THIS fixture: the import-time action registry, the
    named-latch registry, the monotone registration-path set, and the cooldown latches
    their own readers age out. A blanket "clear every module-level mutable" sweep would
    take all of those with the eight, plus the ~82 read-only constant tables, which is
    why each exclusion and its reason is written down IN THE CENSUS rather than here:
    naming a container in this docstring would make the census read it as a name
    conftest resets, and a row that says "nobody looks after this" while something else
    silently clears it is exactly the duplicate the census exists to name.

    The package's lazy-import memo and the self-draining pending-emit set were on that
    list once and are not any more: `_reset_package_process_state` above clears both at
    both ends of every test, for the reach analysis it gives in
    `_clear_package_process_state`. Two fixtures, one reset each, is the split that file
    and this one already keep for the socket singles below.

    Resolved through `sys.modules` and skipped when absent, the shape
    `_reset_warn_latches` uses: a module no test imported has no container to clear,
    and a no-plugins bundle has no `plugins/pipe_dashboard` at all. The census, not
    this fixture, is what covers a container in a module nothing imported.

    It cannot reach the two SINGLES in `_reset_dashboard_socket_state` above, and that
    is the split between the two fixtures: `dashboard_socket._registered` and
    `._resync` are bools, and this census admits only containers. The three per-pipe
    registries that fixture also repairs (`_pipe_getters`, `_routes_get_pipes`,
    `_pd_snapshot_getters`) appear literally in its text, so the census derives them into
    its covered side on its own; adding them here as well would be the duplicate this
    census exists to name. Reading one list for the other is how a fourth global beside
    those would arrive unaccounted for.
    """
    def _clear() -> None:
        for module_name, attr in _MODULE_STATE_CONTAINERS:
            module = sys.modules.get(module_name)
            if module is None:
                continue
            container = getattr(module, attr, None)
            if isinstance(container, (set, dict, list)):
                container.clear()

    _clear()
    yield
    _clear()


def _clear_stub_task_models() -> None:
    """Empty the Open WebUI Config stub's Task Model rows.

    The rows stand in for a database read, so they are module-global mutable state
    with no other reset: a test that configures a model id would otherwise leak it
    into every later test's stub, and the next test would silently get a candidate
    it never set.
    """
    from open_webui.models.config import Config as _OwuiConfig

    _OwuiConfig._rows.clear()


@pytest.fixture(autouse=True)
def _reset_model_registry():
    """Reset OpenRouterModelRegistry class-level state before each test.

    The registry uses class-level attributes for catalog caching. Without this reset,
    tests that run earlier can pollute the catalog state, causing later tests that
    mock HTTP responses to fail because the mock is never hit (cache is still valid).

    The lock reset is bookkeeping like every other line here, not a repair: each lock is
    rebound at its point of use onto the running loop, so a fresh one per test is not
    what keeps the loaders usable across a session that runs more than one loop.
    """
    reg = OpenRouterModelRegistry
    reg._models = []
    reg._specs = {}
    reg._id_map = {}
    reg._zdr_model_ids = None
    reg._zdr_rosters = {}
    reg._zdr_attempted_key = None
    reg._zdr_settle = {}
    reg._zdr_stamped_specs = None
    reg._enriched_cache = None
    reg._zdr_touched = OrderedDict()
    reg._last_fetch = 0.0
    reg._last_video_fetch = 0.0
    reg._last_video_attempt = 0.0
    reg._last_video_account = ""
    reg._last_video_modality_attempt = 0.0
    reg._last_image_fetch = 0.0
    reg._image_endpoints = {}
    reg._image_endpoints_by_target = {}
    reg._last_image_contract_attempt = {}
    reg._last_image_attempt = 0.0
    reg._last_image_account = {}
    reg._last_image_contract_account = {}
    reg._image_contract_retry_after = 0.0
    reg._image_contract_target = None
    reg._image_contract_owed = {}
    reg._image_catalog_norms = frozenset()
    reg._video_catalog_norms = frozenset()
    reg._chat_catalog_norms = frozenset()
    reg._chat_content_digest = ""
    reg._video_content_digest = ""
    reg._image_content_digest = ""
    reg._name_map = None
    reg._lock = asyncio.Lock()
    reg._next_refresh_after = 0.0
    reg._failure_counts = {}
    reg._last_errors = {}
    reg._last_error = None
    reg._last_error_time = 0.0
    ModelFamily.set_dynamic_specs(None)
    yield


@pytest.fixture(autouse=True)
def _isolate_webui_secret_key(monkeypatch):
    """Keep WEBUI_SECRET_KEY unset by default so the SEND_CACHE_SESSION_ID cache pin is
    deterministic regardless of ambient env or test order; tests that exercise the pin set
    it explicitly via monkeypatch.setenv. WEBUI_JWT_SECRET_KEY goes with it: it is a live
    fallback for Open WebUI's own key derivation, not a historical name, so an ambient one
    decides every reader that does not look at the primary name at all. OPENROUTER_API_KEY
    goes with both: `Valves.API_KEY` defaults to it, so a developer's exported key is the
    one input the suite must not inherit -- it decides whether a warm-up arms at all, and
    the tests that care about that set the valve themselves.
    """
    monkeypatch.delenv("WEBUI_SECRET_KEY", raising=False)
    monkeypatch.delenv("WEBUI_JWT_SECRET_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


_LOOPBACK = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}
_UNRESOLVABLE_MARKERS = ("nonexistent", "does-not-exist", "no-such-host")
# RFC 2606 reserves these TLDs precisely so they never resolve. Tests reach for them to
# mean "this name does not exist", and answering them turns an instant NXDOMAIN into a
# refused connection the retry policy then backs off over -- 35s per test.
_UNRESOLVABLE_SUFFIXES = (".test", ".invalid")
_STUB_PUBLIC_IP = "93.184.216.34"


def _stub_getaddrinfo(host, port, *args, **kwargs):
    """Answer every name from a rule, so no test depends on the network.

    23 test/host pairs were reaching live DNS -- `openrouter.ai`, `example.com`,
    `localhost` -- because the SSRF gate resolves a hostname before fetching it. A DNS
    hiccup therefore turned into five red tests with a failure message about video
    blocks, and the same outage in CI produces the same misleading build. Nothing here
    is testing name resolution; they are testing what the code does with the answer.

    A rule rather than a per-test allow-list: names carrying an unresolvable marker fail
    the way a real NXDOMAIN does, loopback stays loopback so the SSRF gate still sees a
    private address, and everything else is one fixed public IP.
    """
    import socket as _socket

    name = str(host or "").strip().strip("[]").lower()
    if any(marker in name for marker in _UNRESOLVABLE_MARKERS) or name.endswith(
        _UNRESOLVABLE_SUFFIXES
    ):
        raise _socket.gaierror(
            _socket.EAI_NONAME, f"stubbed NXDOMAIN for {host!r} (tests never use real DNS)"
        )
    ip = "127.0.0.1" if name in _LOOPBACK else _STUB_PUBLIC_IP
    try:
        resolved_port = int(port) if port is not None else 0
    except (TypeError, ValueError):
        resolved_port = 0
    return [(_socket.AF_INET, _socket.SOCK_STREAM, 6, "", (ip, resolved_port))]


def _is_loopback(address) -> bool:
    import ipaddress

    if not isinstance(address, tuple) or not address:
        return True  # AF_UNIX and friends: not outbound
    try:
        return ipaddress.ip_address(str(address[0])).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True, scope="session")
def _no_real_network():
    """Hermetic by default: a network outage must not read as a code failure.

    Two halves, and both are needed. Stubbing name resolution alone made the suite
    slower, not safer: a name that used to fail fast now answered with an address, and
    the connection attempt hung until its timeout. Two tests went from milliseconds to
    66 seconds, past CI's 60-second per-test ceiling. No choice of address fixes that --
    every candidate hangs behind a NAT, and the TEST-NET ranges read as private, which
    would disarm the SSRF tests.

    So outbound connections are refused instantly instead. An unmocked request becomes a
    fast, named failure rather than a slow one, and loopback still works for the tests
    that bind a local server or a Redis stub.

    The stub only answers names a caller asks the OS for, and with the production
    dependency set installed aiohttp's default resolver is c-ares-backed and never asks:
    a connector a test builds resolves for real and the rule above is bypassed. Both
    `DefaultResolver` names are therefore pinned to the threaded resolver, which reaches
    `socket.getaddrinfo` through the loop's executor -- the same thing Open WebUI itself
    does in `backend/open_webui/env.py` ("c-ares breaks name resolution in some
    environments"), off by default, so the test resolver is also the shipped one.
    """
    import socket

    import aiohttp.connector as _connector
    import aiohttp.resolver as _resolver

    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_resolver = _resolver.DefaultResolver
    real_connector_resolver = getattr(_connector, "DefaultResolver", None)

    def _blocked_connect(self, address):
        if _is_loopback(address):
            return real_connect(self, address)
        raise ConnectionRefusedError(
            f"outbound connection to {address!r} blocked: this test made a network "
            "request that is not mocked. Mock it (aioresponses) rather than relying on "
            "the request failing."
        )

    def _blocked_connect_ex(self, address):
        if _is_loopback(address):
            return real_connect_ex(self, address)
        import errno

        return errno.ECONNREFUSED

    socket.getaddrinfo = _stub_getaddrinfo
    socket.socket.connect = _blocked_connect
    socket.socket.connect_ex = _blocked_connect_ex
    _resolver.DefaultResolver = _resolver.ThreadedResolver
    # `aiohttp.connector` binds its own reference at import, so pinning the public module
    # alone leaves the connector building a c-ares resolver. Reached via setattr because
    # it is a private re-export pyright does not see as an attribute.
    setattr(_connector, "DefaultResolver", _resolver.ThreadedResolver)  # noqa: B010
    try:
        yield
    finally:
        socket.getaddrinfo = real_getaddrinfo
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex
        _resolver.DefaultResolver = real_resolver
        if real_connector_resolver is not None:
            setattr(_connector, "DefaultResolver", real_connector_resolver)  # noqa: B010


@pytest.fixture(autouse=True)
def retry_backoff():
    """Record what a retry would have waited instead of waiting it out.

    The pipe retries a failed call three times behind `tenacity`, backing off 0.5s then 1.0s.
    Nine call sites use the default, so every test that drives a failing request pays 1.5 seconds
    of real sleep -- about 40 seconds across the suite -- to prove things that have nothing to do
    with how long a retry pauses: which error card is shown, whether a failure was counted.

    `AsyncRetrying` binds its sleep as a default argument at class-definition time, so replacing
    the module function does not reach it; the constructor is the seam. A test that cares about
    the pause can ask for this fixture and read the delays, so nothing becomes unobservable --
    it just stops being lived through.
    """
    from tenacity import AsyncRetrying

    waited: list[float] = []
    original = AsyncRetrying.__init__

    async def _record(delay: float) -> None:
        waited.append(delay)
        await asyncio.sleep(0)

    def _patched(self, *args, **kwargs):
        if not args:
            kwargs.setdefault("sleep", _record)
        original(self, *args, **kwargs)

    AsyncRetrying.__init__ = _patched
    try:
        yield waited
    finally:
        AsyncRetrying.__init__ = original


def pytest_collection_modifyitems(session, config, items):
    """Refuse a whole-suite run against a bundle unless somebody said out loud that they meant it.

    Running every test against a flattened artifact costs minutes and answers one question: does the
    code still work once it is a single file. Nothing about the answer can change until the work is
    finished, so a mid-work run is time spent to learn something that has to be learned again later.
    The gate says this too, but the gate is easy to step around -- this is the same policy where the
    cost actually is, so stepping around the gate does not step around the decision.

    Targeted runs are untouched: the point is to make the cheap path the easy one.
    """
    if not os.environ.get("OWUI_PIPE_BUNDLE_PATH"):
        return
    # Collecting every test under a bundle is the cheap structural check the batch tier does on purpose: it
    # catches what only breaks once the package is one file, in seconds, without running anything.
    if getattr(config.option, "collectonly", False):
        return
    if os.environ.get("GATE_BUNDLE_RUN_APPROVED"):
        return
    # CI exists to run exactly this, on every bundle, every push. It is the one place where a whole
    # bundled suite is the point rather than a detour.
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return
    if len(items) < 500:
        return
    raise pytest.UsageError(
        f"\n\n  Refusing to run {len(items)} tests against a bundled artifact mid-work.\n\n"
        "  A bundled suite tells you whether the flattened file still behaves. That cannot\n"
        "  change until the work is done, so this is an answer you will need again anyway.\n"
        "  The package suite is what says whether the change is correct:\n\n"
        "      scripts/gate.sh batch\n\n"
        "  If the work IS finished and you mean to do the final check, the gate knows how to\n"
        "  ask for it. Run a smaller selection if you only need a few tests under the bundle.\n"
    )


class _TimeTravelLoop(asyncio.SelectorEventLoop):
    """An event loop that advances to the next timer rather than waiting for it.

    `_scheduled` is the loop's own heap of pending wake-ups. When nothing is runnable, the earliest of
    those is the only thing that can happen next, so moving the clock there changes what the loop does
    next by exactly nothing -- except that it costs no real time. Tests that wait out a limit expressed in
    minutes reach it for free, and a retry that honours a `Retry-After` of 30s costs no wall-clock either.
    """

    # CPython internals typeshed does not declare. Naming them states what this clock rests on;
    # `test_the_event_loop_internals_the_jumping_clock_rests_on_still_exist` in tests/test_tool_timeouts.py
    # fails if one goes away.
    _ready: Any
    _scheduled: Any

    def __init__(self) -> None:
        super().__init__()
        self._skew = 0.0

    def time(self) -> float:
        return super().time() + self._skew

    def _run_once(self) -> None:
        if not self._ready and self._scheduled:
            gap = self._scheduled[0]._when - self.time()
            if gap > 0:
                self._skew += gap
        super()._run_once()  # pyright: ignore[reportAttributeAccessIssue]


class _TimeTravelPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        return _TimeTravelLoop()
