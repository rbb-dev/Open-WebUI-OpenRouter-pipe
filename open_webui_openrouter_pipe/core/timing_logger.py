from __future__ import annotations

import asyncio
import datetime
import functools
import inspect
import json
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

_timing_file_lock = threading.RLock()
_timing_file_path: Path | None = None
_timing_file_handle: Any | None = None

_timing_enabled: ContextVar[bool] = ContextVar("timing_enabled", default=False)
_timing_request_id: ContextVar[str | None] = ContextVar(
    "timing_request_id", default=None
)


@dataclass(slots=True)
class TimingEvent:
    ts: float
    wall_ts: float
    event: str
    label: str
    elapsed_ms: float | None = None


def _format_iso_utc(wall_ts: float) -> str:
    try:
        dt = datetime.datetime.fromtimestamp(wall_ts, tz=datetime.UTC)
        return dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")
    except (OSError, OverflowError, TypeError, ValueError):
        return datetime.datetime.now(datetime.UTC).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")


def _record_event(event: TimingEvent, request_id: str | None = None) -> None:
    if request_id is None:
        if not _timing_enabled.get():
            return
        request_id = _timing_request_id.get()
        if not request_id:
            return

    record: dict[str, Any] = {
        "ts": _format_iso_utc(event.wall_ts),
        "perf_ts": round(event.ts, 6),
        "event": event.event,
        "label": event.label,
        "request_id": request_id,
    }
    if event.elapsed_ms is not None:
        record["elapsed_ms"] = round(event.elapsed_ms, 3)

    with _timing_file_lock:
        if _timing_file_handle is not None:
            try:
                line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                _timing_file_handle.write(line + "\n")
                _timing_file_handle.flush()
            except (OSError, RecursionError, TypeError, ValueError):
                pass


def configure_timing_file(file_path: str) -> bool:
    global _timing_file_path, _timing_file_handle

    with _timing_file_lock:
        if _timing_file_handle is not None:
            try:
                _timing_file_handle.close()
            except OSError:
                pass
            _timing_file_handle = None

        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            _timing_file_handle = open(path, "a", encoding="utf-8")  # noqa: SIM115
            _timing_file_path = path
            return True
        except (OSError, TypeError, ValueError):
            _timing_file_path = None
            _timing_file_handle = None
            return False


def close_timing_file() -> None:
    global _timing_file_handle, _timing_file_path

    with _timing_file_lock:
        if _timing_file_handle is not None:
            try:
                _timing_file_handle.close()
            except OSError:
                pass
            _timing_file_handle = None
            _timing_file_path = None


def ensure_timing_file_configured(file_path: str) -> bool:
    global _timing_file_handle

    with _timing_file_lock:
        if _timing_file_handle is not None and _timing_file_path is not None:
            if str(_timing_file_path) == str(Path(file_path)):
                return True
            try:
                _timing_file_handle.close()
            except OSError:
                pass
            _timing_file_handle = None

    return configure_timing_file(file_path)


def set_timing_context(
    request_id: str, enabled: bool
) -> list[tuple[ContextVar[Any], Token[Any]]]:
    return [
        (_timing_request_id, _timing_request_id.set(request_id)),
        (_timing_enabled, _timing_enabled.set(enabled)),
    ]


def clear_timing_context() -> None:
    _timing_request_id.set(None)
    _timing_enabled.set(False)


def timing_mark(label: str) -> None:
    if not _timing_enabled.get():
        return
    now_perf = time.perf_counter()
    now_wall = time.time()
    _record_event(TimingEvent(
        ts=now_perf,
        wall_ts=now_wall,
        event="mark",
        label=label,
    ))


@contextmanager
def timing_scope(label: str):
    if not _timing_enabled.get():
        yield
        return
    start_perf = time.perf_counter()
    start_wall = time.time()
    _record_event(
        TimingEvent(
            ts=start_perf,
            wall_ts=start_wall,
            event="enter",
            label=label,
        )
    )
    try:
        yield
    finally:
        end_perf = time.perf_counter()
        elapsed_ms = (end_perf - start_perf) * 1000
        _record_event(
            TimingEvent(
                ts=end_perf,
                wall_ts=time.time(),
                event="exit",
                label=label,
                elapsed_ms=elapsed_ms,
            )
        )


F = TypeVar("F", bound=Callable[..., Any])


async def _bracket_generator(
    agen: Any, label: str, rid: str, start: float
) -> Any:
    inner, send, exc = agen, None, None
    try:
        while True:
            try:
                item = await (inner.asend(send) if exc is None else inner.athrow(exc))
            except StopAsyncIteration:
                return
            try:
                send = yield item
                exc = None
            except GeneratorExit:
                await inner.aclose()
                raise
            except BaseException as thrown:  # noqa: BLE001
                exc, send = thrown, None
    finally:
        end = time.perf_counter()
        _record_event(
            TimingEvent(
                ts=end,
                wall_ts=time.time(),
                event="exit",
                label=label,
                elapsed_ms=(end - start) * 1000,
            ),
            request_id=rid,
        )


def timed(func: F) -> F:
    module = getattr(func, "__module__", "") or ""
    qualname = getattr(func, "__qualname__", "") or getattr(func, "__name__", "unknown")

    module = module.removeprefix("open_webui_openrouter_pipe.")

    label = f"{module}.{qualname}" if module else qualname

    if inspect.isasyncgenfunction(func):

        @functools.wraps(func)
        def call_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _timing_enabled.get():
                return func(*args, **kwargs)
            rid = _timing_request_id.get()
            if not rid:
                return func(*args, **kwargs)
            start_perf = time.perf_counter()
            _record_event(
                TimingEvent(
                    ts=start_perf,
                    wall_ts=time.time(),
                    event="enter",
                    label=label,
                ),
                request_id=rid,
            )
            return _bracket_generator(
                func(*args, **kwargs), label, rid, start_perf
            )

        return call_wrapper  # type: ignore[return-value]
    if asyncio.iscoroutinefunction(func):

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _timing_enabled.get():
                return await func(*args, **kwargs)
            start_perf = time.perf_counter()
            start_wall = time.time()
            _record_event(
                TimingEvent(
                    ts=start_perf,
                    wall_ts=start_wall,
                    event="enter",
                    label=label,
                )
            )
            try:
                return await func(*args, **kwargs)
            finally:
                end_perf = time.perf_counter()
                elapsed_ms = (end_perf - start_perf) * 1000
                _record_event(
                    TimingEvent(
                        ts=end_perf,
                        wall_ts=time.time(),
                        event="exit",
                        label=label,
                        elapsed_ms=elapsed_ms,
                    )
                )

        return async_wrapper  # type: ignore[return-value]
    else:

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not _timing_enabled.get():
                return func(*args, **kwargs)
            start_perf = time.perf_counter()
            start_wall = time.time()
            _record_event(
                TimingEvent(
                    ts=start_perf,
                    wall_ts=start_wall,
                    event="enter",
                    label=label,
                )
            )
            try:
                return func(*args, **kwargs)
            finally:
                end_perf = time.perf_counter()
                elapsed_ms = (end_perf - start_perf) * 1000
                _record_event(
                    TimingEvent(
                        ts=end_perf,
                        wall_ts=time.time(),
                        event="exit",
                        label=label,
                        elapsed_ms=elapsed_ms,
                    )
                )

        return sync_wrapper  # type: ignore[return-value]
