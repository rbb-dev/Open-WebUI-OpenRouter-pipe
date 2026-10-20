from __future__ import annotations

import contextlib
import threading
import time
from collections import defaultdict, deque
from typing import ClassVar


def counted_tool_failures(failures: deque[float], now: float, window_seconds: float) -> int:
    if failures and now - failures[-1] > window_seconds:
        return 0
    return len(failures)


class CircuitBreaker:
    _AUTH_FAILURE_TTL_SECONDS = 60
    _AUTH_FAILURE_UNTIL: ClassVar[dict[str, float]] = {}
    _AUTH_FAILURE_SWEPT_AT: ClassVar[float] = 0.0
    _AUTH_FAILURE_LOCK = threading.Lock()

    def __init__(self, *, threshold: int, window_seconds: float):
        self._threshold = max(1, int(threshold))
        self._window_seconds = window_seconds

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

    def record_failure(self, user_id: str) -> float | None:
        if not user_id:
            return None
        now = time.time()
        window = self._breaker_records[user_id]
        window.append(now)
        del self._breaker_records[user_id]
        self._breaker_records[user_id] = window
        self._sweep_expired(now, force=True, which="requests")
        return now

    def retract_failure(self, user_id: str, recorded_at: float | None) -> None:
        if not user_id or recorded_at is None:
            return
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
