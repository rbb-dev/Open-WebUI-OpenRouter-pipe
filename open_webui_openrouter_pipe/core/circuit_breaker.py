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
from typing import TYPE_CHECKING, Optional
from ..core.timing_logger import timed

if TYPE_CHECKING:
    pass


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
    _AUTH_FAILURE_UNTIL: dict[str, float] = {}
    _AUTH_FAILURE_LOCK = threading.Lock()

    @timed
    def __init__(self, *, threshold: int, window_seconds: float):
        """Initialize the circuit breaker.

        Args:
            threshold: Maximum failures allowed within the time window
            window_seconds: Time window in seconds for counting failures
        """
        self._threshold = threshold
        self._window_seconds = window_seconds

        # Per-user request failure tracking
        self._breaker_records: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=threshold)
        )

        # Per-user per-tool-type failure tracking
        self._tool_breakers: dict[str, dict[str, deque[float]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=threshold))
        )

    @property
    @timed
    def threshold(self) -> int:
        """Get the failure threshold."""
        return self._threshold

    @threshold.setter
    @timed
    def threshold(self, value: int) -> None:
        """Set the failure threshold."""
        self._threshold = max(1, int(value))

    @property
    @timed
    def window_seconds(self) -> float:
        """Get the time window in seconds."""
        return self._window_seconds

    @window_seconds.setter
    @timed
    def window_seconds(self, value: float) -> None:
        """Set the time window in seconds."""
        self._window_seconds = max(0.1, float(value))

    # --------------------------------------------------------------------------
    # Request Circuit Breaker (per-user)
    # --------------------------------------------------------------------------

    @timed
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

        window = self._breaker_records[user_id]
        now = time.time()

        # Evict old failures outside the time window
        while window and now - window[0] > self._window_seconds:
            window.popleft()

        return len(window) < self._threshold

    @timed
    def record_failure(self, user_id: str) -> None:
        """Record a request failure for a user.

        Args:
            user_id: User identifier
        """
        if not user_id:
            return
        self._breaker_records[user_id].append(time.time())

    @timed
    def reset(self, user_id: str) -> None:
        """Clear all failure records for a user, allowing requests again.

        Args:
            user_id: User identifier
        """
        if not user_id:
            return
        if user_id in self._breaker_records:
            self._breaker_records[user_id].clear()

    # --------------------------------------------------------------------------
    # Tool Circuit Breaker (per-user per-tool-type)
    # --------------------------------------------------------------------------

    @timed
    def tool_allows(self, user_id: str, tool_type: str) -> bool:
        """Check if a specific tool type is allowed for a user.

        Similar to allows() but tracks failures per tool type.
        Returns False if too many failures of this tool type have occurred
        within the time window.

        Args:
            user_id: User identifier
            tool_type: Tool type identifier

        Returns:
            True if tool execution is allowed, False if breaker is open
        """
        if not user_id or not tool_type:
            return True

        window = self._tool_breakers[user_id][tool_type]
        now = time.time()

        # Evict old failures outside the window
        while window and now - window[0] > self._window_seconds:
            window.popleft()

        return len(window) < self._threshold

    @timed
    def record_tool_failure(self, user_id: str, tool_type: str) -> None:
        """Record a tool execution failure for a specific tool type.

        Args:
            user_id: User identifier
            tool_type: Tool type identifier
        """
        if not user_id or not tool_type:
            return
        self._tool_breakers[user_id][tool_type].append(time.time())

    @timed
    def reset_tool(self, user_id: str, tool_type: str) -> None:
        """Clear failure records for a specific tool type.

        Args:
            user_id: User identifier
            tool_type: Tool type identifier
        """
        if not user_id or not tool_type:
            return
        if user_id in self._tool_breakers and tool_type in self._tool_breakers[user_id]:
            self._tool_breakers[user_id][tool_type].clear()

    # --------------------------------------------------------------------------
    # Auth Failure Tracking (class-level, shared across all instances)
    # --------------------------------------------------------------------------

    @classmethod
    @timed
    def note_auth_failure(cls, scope_key: str, *, ttl_seconds: Optional[int] = None) -> None:
        """Record an authentication failure.

        This is a class-level method that tracks auth failures across all
        pipe instances to prevent repeated auth attempts.

        Args:
            scope_key: Scope identifier (e.g., pipe identifier or global scope)
            ttl_seconds: Time to live in seconds (default: 60)
        """
        if not scope_key:
            return

        ttl = int(ttl_seconds or cls._AUTH_FAILURE_TTL_SECONDS)
        if ttl <= 0:
            return

        until = time.time() + ttl
        with cls._AUTH_FAILURE_LOCK:
            cls._AUTH_FAILURE_UNTIL[scope_key] = until

    @classmethod
    @timed
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
