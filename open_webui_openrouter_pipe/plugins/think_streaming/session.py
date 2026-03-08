"""Per-request session management for Think Streaming.

Each streaming request with Think Streaming enabled gets a ``ThinkSession``
containing an ``asyncio.Queue`` that bridges the emitter wrapper (producer)
and the SSE endpoint (consumer).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

_ts_session_log = logging.getLogger(__name__)

_TS_DEFAULT_TTL = 600.0  # 10 minutes
_TS_DEFAULT_MAX_SESSIONS = 100
_TS_DEFAULT_QUEUE_SIZE = 500


@dataclass
class ThinkSession:
    """A single think-streaming session tied to an ephemeral key."""

    key: str
    queue: asyncio.Queue[str | None] = field(default_factory=lambda: asyncio.Queue(maxsize=_TS_DEFAULT_QUEUE_SIZE))
    created_at: float = field(default_factory=time.monotonic)
    consumer_alive: bool = True
    user_id: str = ""


class SessionRegistry:
    """Process-local registry of active think-streaming sessions.

    Sessions are keyed by the ephemeral key used for the SSE endpoint.
    Cleanup is layered: explicit removal on completion, TTL expiry for
    stragglers, and capacity eviction as a hard cap.
    """

    __slots__ = ("_max_sessions", "_sessions", "_ttl")

    def __init__(
        self,
        max_sessions: int = _TS_DEFAULT_MAX_SESSIONS,
        ttl: float = _TS_DEFAULT_TTL,
    ) -> None:
        self._sessions: dict[str, ThinkSession] = {}
        self._max_sessions = max(1, max_sessions)
        self._ttl = max(1.0, ttl)

    def create(self, key: str, *, user_id: str = "") -> ThinkSession:
        """Create and register a new session for *key*."""
        self.cleanup_expired()
        # Evict oldest if at capacity
        while len(self._sessions) >= self._max_sessions:
            oldest_key = min(self._sessions, key=lambda k: self._sessions[k].created_at)
            removed = self._sessions.pop(oldest_key)
            _ts_session_log.debug("Evicted oldest think session %s (user=%s)", oldest_key[:8], removed.user_id)
        session = ThinkSession(key=key, user_id=user_id)
        self._sessions[key] = session
        return session

    def get(self, key: str) -> ThinkSession | None:
        """Look up a session by key, or ``None`` if not found."""
        return self._sessions.get(key)

    def remove(self, key: str) -> None:
        """Remove a session by key (no-op if absent)."""
        self._sessions.pop(key, None)

    def cleanup_expired(self) -> None:
        """Remove sessions older than TTL."""
        cutoff = time.monotonic() - self._ttl
        expired = [k for k, s in self._sessions.items() if s.created_at < cutoff]
        for k in expired:
            del self._sessions[k]

    @property
    def active_count(self) -> int:
        """Number of currently tracked sessions."""
        return len(self._sessions)
