"""Circuit breaker system for request and tool failure tracking.

This module provides circuit breaker functionality to protect against:
- Excessive request failures per user
- Excessive tool execution failures per user/tool-type
- Authentication failures across all pipe instances

The circuit breaker automatically blocks requests when failure thresholds are exceeded,
preventing cascading failures and providing graceful degradation.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import ClassVar


def counted_tool_failures(failures: deque[float], now: float, window_seconds: float) -> int:
    if failures and now - failures[-1] > window_seconds:
        return 0
    return len(failures)


class CircuitBreaker:
    """Circuit breaker for request and tool failure tracking.

    This class manages two types of circuit breakers:
    1. Instance-level: Per-user request and tool failures
    2. Class-level: Authentication failures shared across all instances

    Instance-level breakers track failures per user to prevent individual users
    from overwhelming the system. Class-level auth breakers prevent repeated
    auth failures across all pipe instances.
    """

    # Class-level auth failure tracking (shared across all instances)
    _AUTH_FAILURE_TTL_SECONDS = 60
    _AUTH_FAILURE_UNTIL: ClassVar[dict[str, float]] = {}
    _AUTH_FAILURE_SWEPT_AT: ClassVar[float] = 0.0
    _AUTH_FAILURE_LOCK = threading.Lock()

    def __init__(self, *, threshold: int, window_seconds: float):
        self._threshold = max(1, int(threshold))
        self._window_seconds = window_seconds

        # Per-user request failure tracking
        self._breaker_records: dict[str, deque[float]] = defaultdict(deque)

        # Per-user per-tool-type failure tracking
        self._tool_breakers: dict[str, dict[tuple[str, str], deque[float]]] = defaultdict(
            lambda: defaultdict(deque)
        )

        self._sweep_after: float = 0.0
        self._read_swept_at: float = 0.0

    def _sweep_expired(self, now: float) -> None:
        if now < self._sweep_after:
            return
        self._sweep_after = now + self._window_seconds
        window = self._window_seconds
        for key, failures in list(self._breaker_records.items()):
            if not failures or now - failures[-1] > window:
                self._breaker_records.pop(key, None)
        for user_id, tools in list(self._tool_breakers.items()):
            for tool_key in list(tools.keys()):
                failures = tools[tool_key]
                if not failures or now - failures[-1] > window:
                    tools.pop(tool_key, None)
            if not tools:
                self._tool_breakers.pop(user_id, None)

    @property
    def threshold(self) -> int:
        """Get the failure threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: int) -> None:
        self._threshold = max(1, int(value))

    @property
    def window_seconds(self) -> float:
        """Get the time window in seconds."""
        return self._window_seconds

    @window_seconds.setter
    def window_seconds(self, value: float) -> None:
        self._window_seconds = max(0.1, float(value))
        self._sweep_after = min(self._sweep_after, time.time() + self._window_seconds)

    # --------------------------------------------------------------------------
    # Request Circuit Breaker (per-user)
    # --------------------------------------------------------------------------

    def allows(self, user_id: str) -> bool:
        """Check if requests are allowed for a user.

        Per-user breaker governed by threshold and window_seconds.

        Args:
            user_id: User identifier

        Returns:
            True if requests are allowed, False if breaker is open
        """
        if not user_id:
            return True

        now = time.time()
        if now - self._read_swept_at >= self._window_seconds:
            self._read_swept_at = now
            self._sweep_expired(now)

        window = self._breaker_records.get(user_id)
        if window is None:
            return True

        # Evict old failures outside the time window
        while window and now - window[0] > self._window_seconds:
            window.popleft()

        if not window:
            self._breaker_records.pop(user_id, None)

        return len(window) < self._threshold

    def record_failure(self, user_id: str) -> None:
        """Record a request failure for a user.

        Args:
            user_id: User identifier
        """
        if not user_id:
            return
        now = time.time()
        self._breaker_records[user_id].append(now)
        self._sweep_expired(now)

    def reset(self, user_id: str) -> None:
        """Clear all failure records for a user, allowing requests again.

        Args:
            user_id: User identifier
        """
        if not user_id:
            return
        self._breaker_records.pop(user_id, None)

    # --------------------------------------------------------------------------
    # Tool Circuit Breaker (per-user per-tool-type)
    # --------------------------------------------------------------------------

    def tool_allows(self, user_id: str, tool_type: str, tool_name: str = "") -> bool:
        """Check if a specific tool type is allowed for a user.

        Args:
            user_id: User identifier
            tool_type: Tool type identifier

        Returns:
            True if tool execution is allowed, False if breaker is open
        """
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
        """Record a tool execution failure for a specific tool type.

        Args:
            user_id: User identifier
            tool_type: Tool type identifier
        """
        if not user_id or not tool_type:
            return
        now = time.time()
        self._tool_breakers[user_id][(tool_type, tool_name)].append(now)
        self._sweep_expired(now)

    def reset_tool(self, user_id: str, tool_type: str, tool_name: str = "") -> None:
        """Clear failure records for a specific tool type.

        Args:
            user_id: User identifier
            tool_type: Tool type identifier
        """
        if not user_id or not tool_type:
            return
        tool_key = (tool_type, tool_name)
        tools = self._tool_breakers.get(user_id)
        if not tools or tool_key not in tools:
            return
        tools.pop(tool_key, None)
        if not tools:
            self._tool_breakers.pop(user_id, None)

    # --------------------------------------------------------------------------
    # Auth Failure Tracking (class-level, shared across all instances)
    # --------------------------------------------------------------------------

    @classmethod
    def note_auth_failure(cls, scope_key: str, *, ttl_seconds: int | None = None) -> None:
        """Record an authentication failure.

        This is a class-level method that tracks auth failures across all
        pipe instances to prevent repeated auth attempts.

        Args:
            scope_key: Scope identifier (e.g., pipe identifier or global scope)
            ttl_seconds: Time to live in seconds (default: 60)
        """
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
        """Check if an authentication failure is currently active.

        Args:
            scope_key: Scope identifier

        Returns:
            True if auth failure is active, False otherwise
        """
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
