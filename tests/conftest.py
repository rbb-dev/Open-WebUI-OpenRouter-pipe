"""Test configuration helpers for unit tests."""

from __future__ import annotations

# First, so the environment defaults land before anything reads them. They live in
# owui_stubs alone: a second setdefault here would win in this process and lose in
# every probe subprocess, so retargeting DATA_DIR would move only half the harness.
import owui_stubs  # noqa: F401 - Open WebUI/sqlalchemy/tenacity stand-ins + env defaults

import os

import asyncio
import base64
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
import pytest_asyncio


_PIPES_TO_CLOSE: list["Pipe"] = []

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


@pytest.fixture
def pipe_instance(request):
    """Return a fresh Pipe instance for tests."""
    pipe = Pipe()

    def _finalize() -> None:
        _schedule_pipe_cleanup(pipe)

    request.addfinalizer(_finalize)
    return pipe


@pytest_asyncio.fixture
async def pipe_instance_async():
    """Return a fresh Pipe instance for async tests with proper cleanup."""
    pipe = Pipe()
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

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry, ModelFamily


_WARN_LATCH_PREFIX = "_warned"


def _warn_latches() -> dict[str, set | dict | list]:
    """Every module-level warn-once latch in the package.

    These suppress a warning for the life of the process, so the first test to trip
    one silently disarms every later assertion that the warning is emitted. Resolved
    fresh each call because bundled modes alias submodule names onto one module.
    """
    seen: dict[str, set | dict | list] = {}
    # Deduped by module OBJECT: this runs autouse before all ~5950 tests, and in a flat
    # bundle 107 submodule names alias 2 module objects, so scanning per name re-walked
    # the same namespace 107 times -- 6.3 ms a call, ~37 s per bundled CI run, for an
    # identical set of latches.
    scanned: set[int] = set()
    for name, module in list(sys.modules.items()):
        if name != "open_webui_openrouter_pipe" and not name.startswith(
            "open_webui_openrouter_pipe."
        ):
            continue
        if id(module) in scanned:
            continue
        scanned.add(id(module))
        for attr, value in list(vars(module).items()):
            if attr.startswith(_WARN_LATCH_PREFIX) and isinstance(value, (set, dict, list)):
                seen.setdefault(f"{name}.{attr}", value)
    return seen


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
    """
    import logging as _logging

    from open_webui_openrouter_pipe.core.logging_system import SessionLogger

    token = SessionLogger.log_level.set(_logging.DEBUG)
    try:
        yield
    finally:
        SessionLogger.log_level.reset(token)


@pytest.fixture(autouse=True)
def _reset_stub_chat_files():
    """The chat_file row store is class state; without this it leaks across tests."""
    chats = getattr(sys.modules.get("open_webui.models.chats"), "Chats", None)
    store = getattr(chats, "_chat_files", None)
    if store is not None:
        store.clear()
    yield


@pytest.fixture(autouse=True)
def _reset_warn_latches():
    """Reset every warn-once latch that is currently loaded.

    A latch suppresses its warning for the life of the process, so without this the
    first test to trip one silently disarms every later assertion that it is emitted.
    Which latches exist is asserted in test_warn_latch_isolation.py rather than here:
    a module that no test imported has no latch to reset, so a per-test assertion on
    the full set fails for reasons that have nothing to do with isolation.
    """
    latches = _warn_latches()
    for latch in latches.values():
        latch.clear()
    yield


@pytest.fixture(autouse=True)
def _reset_model_registry():
    """Reset OpenRouterModelRegistry class-level state before each test.

    The registry uses class-level attributes for catalog caching. Without this reset,
    tests that run earlier can pollute the catalog state, causing later tests that
    mock HTTP responses to fail because the mock is never hit (cache is still valid).
    """
    reg = OpenRouterModelRegistry
    reg._models = []
    reg._specs = {}
    reg._id_map = {}
    reg._zdr_model_ids = None
    reg._last_fetch = 0.0
    reg._last_video_fetch = 0.0
    reg._last_video_attempt = 0.0
    reg._last_image_fetch = 0.0
    reg._image_endpoints = {}
    reg._last_image_contract_attempt = 0.0
    reg._last_image_attempt = 0.0
    reg._lock = asyncio.Lock()
    reg._next_refresh_after = 0.0
    reg._consecutive_failures = 0
    reg._last_error = None
    reg._last_error_time = 0.0
    ModelFamily.set_dynamic_specs(None)
    yield


@pytest.fixture(autouse=True)
def _isolate_webui_secret_key(monkeypatch):
    """Keep WEBUI_SECRET_KEY unset by default so the SEND_CACHE_SESSION_ID cache pin is
    deterministic regardless of ambient env or test order; tests that exercise the pin set
    it explicitly via monkeypatch.setenv.
    """
    monkeypatch.delenv("WEBUI_SECRET_KEY", raising=False)


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
    """
    import socket

    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

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
    try:
        yield
    finally:
        socket.getaddrinfo = real_getaddrinfo
        socket.socket.connect = real_connect
        socket.socket.connect_ex = real_connect_ex
