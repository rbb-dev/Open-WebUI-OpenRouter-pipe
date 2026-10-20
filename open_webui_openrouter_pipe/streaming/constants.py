from __future__ import annotations

from time import perf_counter

REASONING_STATUS_PUNCTUATION = (".", "!", "?", ":", "\n")
REASONING_STATUS_MAX_CHARS = 160
REASONING_STATUS_MIN_CHARS = 12
REASONING_STATUS_IDLE_SECONDS = 0.75

DEFERRED_REASONING_FLUSH = "deferred_reasoning_flush"
FUSION_EMBED_ATTEMPTS = "fusion_embed_attempts"

_REPLAY_DROPPED_OPENING = (
    "Earlier tool results no longer fit this model's context, so the model "
    "did not receive:"
)


class ReasoningStatusThrottle:
    __slots__ = ("_buffer", "_last_emit")

    def __init__(self) -> None:
        self._buffer: str = ""
        self._last_emit: float | None = None

    def feed(self, delta: str, *, force: bool = False) -> str | None:
        if not isinstance(delta, str):
            return None
        self._buffer += delta
        text = self._buffer.strip()
        if not text:
            return None
        should_emit = force
        now = perf_counter()
        if not should_emit:
            if delta.rstrip(" \t").endswith(REASONING_STATUS_PUNCTUATION) or len(text) >= REASONING_STATUS_MAX_CHARS:
                should_emit = True
            else:
                elapsed = None if self._last_emit is None else (now - self._last_emit)
                if len(text) >= REASONING_STATUS_MIN_CHARS and (
                    elapsed is None or elapsed >= REASONING_STATUS_IDLE_SECONDS
                ):
                    should_emit = True
        if not should_emit:
            return None
        self._buffer = ""
        self._last_emit = now
        return text

    @property
    def pending(self) -> str:
        return self._buffer
