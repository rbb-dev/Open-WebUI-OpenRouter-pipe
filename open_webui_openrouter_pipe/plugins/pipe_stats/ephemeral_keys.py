"""Process-local ephemeral API key store with auto-expiry."""

from __future__ import annotations

import secrets
import time

_PS_EK_MAX_KEYS = 10
_PS_EK_DEFAULT_TTL = 300.0  # 5 minutes


class EphemeralKeyStore:
    """In-memory key store for dashboard sessions.

    Keys are 64-char hex tokens generated via :func:`secrets.token_hex`.
    Each key tracks its last-access timestamp; keys idle longer than
    *ttl* seconds are purged automatically on the next :meth:`validate`
    call.  A hard cap of :data:`_PS_EK_MAX_KEYS` prevents unbounded growth —
    when exceeded, the oldest key is evicted.
    """

    __slots__ = ("_keys", "_max_keys", "_ttl")

    def __init__(self, ttl: float = _PS_EK_DEFAULT_TTL, max_keys: int = _PS_EK_MAX_KEYS) -> None:
        self._keys: dict[str, float] = {}
        self._ttl = max(1.0, ttl)
        self._max_keys = max(1, max_keys)

    # ── Public API ──

    def generate(self) -> str:
        """Create a new ephemeral key and return it."""
        self._cleanup()
        # Evict oldest if at capacity
        while len(self._keys) >= self._max_keys:
            oldest = min(self._keys, key=self._keys.get)  # type: ignore[arg-type]
            del self._keys[oldest]
        key = secrets.token_hex(32)
        self._keys[key] = time.monotonic()
        return key

    def validate(self, key: str) -> bool:
        """Check if *key* is valid (exists and not expired).

        On success, the key's last-access timestamp is refreshed
        (keep-alive).  Expired keys are purged as a side effect.
        """
        self._cleanup()
        if key not in self._keys:
            return False
        self._keys[key] = time.monotonic()
        return True

    def revoke(self, key: str) -> None:
        """Manually delete a key."""
        self._keys.pop(key, None)

    @property
    def active_count(self) -> int:
        """Number of currently active (non-expired) keys."""
        self._cleanup()
        return len(self._keys)

    # ── Internal ──

    def _cleanup(self) -> None:
        """Purge keys that have been idle longer than TTL."""
        cutoff = time.monotonic() - self._ttl
        expired = [k for k, ts in self._keys.items() if ts < cutoff]
        for k in expired:
            del self._keys[k]
