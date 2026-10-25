from __future__ import annotations

import asyncio
import contextlib
import logging
import math
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar, Self

from .utils import _await_if_needed
from .warn_latch import warn_level

logger = logging.getLogger(__name__)

_SHARED_WAIT_SECONDS = 1.0

_SHARED_RECORD_MIN_TTL_SECONDS = 1800

_warned_shared_breaker: set[str] = set()


def counted_tool_failures(failures: deque[float], now: float, window_seconds: float) -> int:
    if failures and now - failures[-1] > window_seconds:
        return 0
    return len(failures)


class _SharedStamp(float):
    __slots__ = ("member",)
    member: str

    def __new__(cls, value: float, member: str) -> Self:
        stamp = super().__new__(cls, value)
        stamp.member = member
        return stamp


async def _shared_write_failure(
    client: Any, key: str, member: str, stamp: float, window_seconds: float
) -> None:
    pipeline = client.pipeline()
    pipeline.zadd(key, {member: stamp})
    pipeline.expire(key, max(_SHARED_RECORD_MIN_TTL_SECONDS, math.ceil(window_seconds)))
    await _await_if_needed(pipeline.execute())


async def _shared_take_back(client: Any, key: str, member: str) -> None:
    await _await_if_needed(client.zrem(key, member))


async def _shared_clear(client: Any, key: str) -> None:
    await _await_if_needed(client.delete(key))


async def _shared_read(client: Any, key: str, oldest: float) -> list[float]:
    pipeline = client.pipeline()
    pipeline.zremrangebyscore(key, "-inf", f"({oldest!r}")
    pipeline.zrange(key, 0, -1, withscores=True)
    replies = await _await_if_needed(pipeline.execute())
    return [float(score) for _member, score in replies[1]]


class CircuitBreaker:
    _AUTH_FAILURE_TTL_SECONDS = 60
    _AUTH_FAILURE_UNTIL: ClassVar[dict[str, float]] = {}
    _AUTH_FAILURE_SWEPT_AT: ClassVar[float] = 0.0
    _AUTH_FAILURE_LOCK = threading.Lock()

    def __init__(
        self,
        *,
        threshold: int,
        window_seconds: float,
        shared: Callable[[], tuple[Any, str] | None] | None = None,
    ):
        self._threshold = max(1, int(threshold))
        self._window_seconds = window_seconds

        self._shared = shared
        self._shared_tail: asyncio.Task[None] | None = None

        self._breaker_records: dict[str, deque[float]] = defaultdict(deque)

        self._tool_breakers: dict[str, dict[tuple[str, str], deque[float]]] = defaultdict(
            lambda: defaultdict(deque)
        )

        self._sweep_after: float = 0.0
        self._read_swept_at: float = 0.0

    def _sweep_expired(
        self, now: float, *, force: bool = False, which: str = "both"
    ) -> None:
        if not force and now < self._sweep_after:
            return
        window = self._window_seconds
        if which in ("both", "requests"):
            expired: list[str] = []
            for key, failures in self._breaker_records.items():
                if failures and now - failures[-1] <= window:
                    break
                expired.append(key)
            for key in expired:
                self._breaker_records.pop(key, None)
        if which in ("both", "tools"):
            emptied: list[str] = []
            for user_id, tools in self._tool_breakers.items():
                dead: list[tuple[str, str]] = []
                for tool_key in tools.keys():  # noqa: SIM118
                    failures = tools[tool_key]
                    if failures and now - failures[-1] <= window:
                        break
                    dead.append(tool_key)
                for tool_key in dead:
                    tools.pop(tool_key, None)
                if tools:
                    break
                emptied.append(user_id)
            for user_id in emptied:
                self._tool_breakers.pop(user_id, None)
        if which == "both":
            self._sweep_after = now + window

    @property
    def threshold(self) -> int:
        return self._threshold

    @threshold.setter
    def threshold(self, value: int) -> None:
        self._threshold = max(1, int(value))

    @property
    def window_seconds(self) -> float:
        return self._window_seconds

    @window_seconds.setter
    def window_seconds(self, value: float) -> None:
        self._window_seconds = max(0.1, float(value))
        self._sweep_after = min(self._sweep_after, time.time() + self._window_seconds)

    def allows(self, user_id: str) -> bool:
        if not user_id:
            return True

        now = time.time()
        if now - self._read_swept_at >= self._window_seconds:
            self._read_swept_at = now
            self._sweep_expired(now)

        window = self._breaker_records.get(user_id)
        if window is None:
            return True

        while window and now - window[0] > self._window_seconds:
            window.popleft()

        if not window:
            self._breaker_records.pop(user_id, None)

        return len(window) < self._threshold

    def _send(self, user_id: str, operation: Callable[[Any, str], Awaitable[None]]) -> bool:
        target = self._shared() if self._shared is not None else None
        if target is None:
            return False
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        client, prefix = target
        previous = self._shared_tail
        if previous is not None and (previous.done() or previous.get_loop() is not loop):
            previous = None
        self._shared_tail = loop.create_task(
            self._send_after(previous, operation, client, f"{prefix}:{user_id}")
        )
        return True

    async def _send_after(
        self,
        previous: asyncio.Task[None] | None,
        operation: Callable[[Any, str], Awaitable[None]],
        client: Any,
        key: str,
    ) -> None:
        if previous is not None:
            await asyncio.wait((previous,))
        try:
            async with asyncio.timeout(_SHARED_WAIT_SECONDS):
                await operation(client, key)
        except Exception as exc:
            logger.log(
                warn_level(_warned_shared_breaker, "write"),
                "Sharing a user's request failures through Redis failed, so the other workers do "
                "not see this change: %r",
                exc,
                exc_info=True,
            )

    async def settle(self) -> None:
        tail = self._shared_tail
        if tail is not None and not tail.done() and tail.get_loop() is asyncio.get_running_loop():
            await asyncio.wait((tail,), timeout=_SHARED_WAIT_SECONDS)

    async def refresh(self, user_id: str) -> None:
        target = self._shared() if self._shared is not None and user_id else None
        if target is None:
            return
        client, prefix = target
        try:
            async with asyncio.timeout(_SHARED_WAIT_SECONDS):
                await self.settle()
                stamps = await _shared_read(
                    client, f"{prefix}:{user_id}", time.time() - self._window_seconds
                )
        except Exception as exc:
            logger.log(
                warn_level(_warned_shared_breaker, "read"),
                "Reading a user's shared request failures from Redis failed, so this worker "
                "checked the user against the copy it holds in its own memory: %r",
                exc,
                exc_info=True,
            )
            return
        self._breaker_records.pop(user_id, None)
        if stamps:
            self._breaker_records[user_id] = deque(stamps)

    def record_failure(self, user_id: str) -> float | None:
        if not user_id:
            return None
        now = time.time()
        window = self._breaker_records[user_id]
        window.append(now)
        del self._breaker_records[user_id]
        self._breaker_records[user_id] = window
        self._sweep_expired(now, force=True, which="requests")
        member = f"{now!r}:{uuid.uuid4().hex}"
        window_seconds = self._window_seconds
        if not self._send(
            user_id,
            lambda client, key: _shared_write_failure(client, key, member, now, window_seconds),
        ):
            return now
        return _SharedStamp(now, member)

    def retract_failure(self, user_id: str, recorded_at: float | None) -> None:
        if not user_id or recorded_at is None:
            return
        member = getattr(recorded_at, "member", None)
        if member is not None:
            self._send(user_id, lambda client, key: _shared_take_back(client, key, member))
        window = self._breaker_records.get(user_id)
        if not window:
            return
        with contextlib.suppress(ValueError):
            window.remove(recorded_at)
        if not window:
            self._breaker_records.pop(user_id, None)

    def reset(self, user_id: str) -> None:
        if not user_id:
            return
        self._breaker_records.pop(user_id, None)
        self._send(user_id, _shared_clear)

    def tool_allows(self, user_id: str, tool_type: str, tool_name: str = "") -> bool:
        if not user_id or not tool_type:
            return True

        now = time.time()
        if now - self._read_swept_at >= self._window_seconds:
            self._read_swept_at = now
            self._sweep_expired(now)

        tools = self._tool_breakers.get(user_id)
        window = tools.get((tool_type, tool_name)) if tools else None
        if tools is None or window is None:
            return True
        live = counted_tool_failures(window, now, self._window_seconds)
        if not live:
            tools.pop((tool_type, tool_name), None)
            if not tools:
                self._tool_breakers.pop(user_id, None)
        return live < self._threshold

    def record_tool_failure(self, user_id: str, tool_type: str, tool_name: str = "") -> None:
        if not user_id or not tool_type:
            return
        now = time.time()
        tools = self._tool_breakers[user_id]
        tool_key = (tool_type, tool_name)
        window = tools[tool_key]
        window.append(now)
        del tools[tool_key]
        tools[tool_key] = window
        del self._tool_breakers[user_id]
        self._tool_breakers[user_id] = tools
        self._sweep_expired(now, force=True, which="tools")

    def reset_tool(self, user_id: str, tool_type: str, tool_name: str = "") -> None:
        if not user_id or not tool_type:
            return
        tool_key = (tool_type, tool_name)
        tools = self._tool_breakers.get(user_id)
        if not tools or tool_key not in tools:
            return
        tools.pop(tool_key, None)
        if not tools:
            self._tool_breakers.pop(user_id, None)

    @classmethod
    def note_auth_failure(cls, scope_key: str, *, ttl_seconds: int | None = None) -> None:
        if not scope_key:
            return

        ttl = cls._AUTH_FAILURE_TTL_SECONDS if ttl_seconds is None else int(ttl_seconds)
        if ttl <= 0:
            return

        until = time.time() + ttl
        with cls._AUTH_FAILURE_LOCK:
            now = time.time()
            if now - cls._AUTH_FAILURE_SWEPT_AT >= cls._AUTH_FAILURE_TTL_SECONDS:
                cls._AUTH_FAILURE_SWEPT_AT = now
                expired = [key for key, expires in cls._AUTH_FAILURE_UNTIL.items() if expires <= now]
                for key in expired:
                    cls._AUTH_FAILURE_UNTIL.pop(key, None)
            cls._AUTH_FAILURE_UNTIL[scope_key] = until

    @classmethod
    def auth_failure_active(cls, scope_key: str) -> bool:
        if not scope_key:
            return False

        now = time.time()
        with cls._AUTH_FAILURE_LOCK:
            until = cls._AUTH_FAILURE_UNTIL.get(scope_key)
            if until is None:
                return False
            if now >= until:
                cls._AUTH_FAILURE_UNTIL.pop(scope_key, None)
                return False
            return True
