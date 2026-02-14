"""Shared streaming constants and helpers to avoid duplication across modules."""

from __future__ import annotations

from time import perf_counter
from typing import Optional

# Default reasoning status emission thresholds.
REASONING_STATUS_PUNCTUATION = (".", "!", "?", ":", "\n")
REASONING_STATUS_MAX_CHARS = 160
REASONING_STATUS_MIN_CHARS = 12
REASONING_STATUS_IDLE_SECONDS = 0.75


class ReasoningStatusThrottle:
    """Buffers reasoning text deltas and decides when a status update should fire.

    Callers feed deltas via :meth:`feed` and receive back the text to emit
    (or ``None`` when the update should be throttled).  The actual emit
    mechanism is left to the caller — this class is pure decision logic.
    """

    __slots__ = ("_buffer", "_last_emit")

    def __init__(self) -> None:
        self._buffer: str = ""
        self._last_emit: float | None = None

    def feed(self, delta: str, *, force: bool = False) -> Optional[str]:
        """Append *delta* to the buffer and return text to emit, or ``None``."""
        if not isinstance(delta, str):
            return None
        self._buffer += delta
        text = self._buffer.strip()
        if not text:
            return None
        should_emit = force
        now = perf_counter()
        if not should_emit:
            if delta.rstrip().endswith(REASONING_STATUS_PUNCTUATION):
                should_emit = True
            elif len(text) >= REASONING_STATUS_MAX_CHARS:
                should_emit = True
            else:
                elapsed = None if self._last_emit is None else (now - self._last_emit)
                if len(text) >= REASONING_STATUS_MIN_CHARS:
                    if elapsed is None or elapsed >= REASONING_STATUS_IDLE_SECONDS:
                        should_emit = True
        if not should_emit:
            return None
        self._buffer = ""
        self._last_emit = now
        return text

    @property
    def pending(self) -> str:
        """Return buffered text that hasn't been emitted yet."""
        return self._buffer
