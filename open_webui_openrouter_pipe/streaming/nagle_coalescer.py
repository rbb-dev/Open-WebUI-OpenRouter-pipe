"""Nagle-inspired multi-buffer delta coalescer (RFC 896).

Provides adaptive backpressure-driven batching for SSE delta events.
When the downstream consumer (OWUI) is fast, events pass through with
minimal latency. When the consumer is slow, events accumulate and are
delivered in larger batches --- reducing the number of expensive
``serialize_output()`` calls on the OWUI side.

Two independent buffers (text and reasoning) prevent reasoning floods
from blocking text output, tool cards, or other structural events.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batchable event type sets
# ---------------------------------------------------------------------------

_BATCHABLE_TEXT_TYPES = frozenset({"response.output_text.delta"})
_BATCHABLE_REASONING_TYPES = frozenset({
    "response.reasoning_text.delta",
    "response.reasoning.delta",
})

_MAX_DRAIN_PER_CYCLE = 32

# ---------------------------------------------------------------------------
# NagleCoalescer — multi-buffer state machine
# ---------------------------------------------------------------------------


class NagleCoalescer:
    """Multi-buffer state machine for Nagle-style delta coalescing.

    Separates text and reasoning into independent buffers.  Flushes on
    type boundaries, ``item_id`` boundaries, non-batchable events, and
    end-of-drain-cycle (gated by *min_flush_chars*).
    """

    __slots__ = (
        "text_buffer", "text_template", "text_length",
        "reasoning_buffer", "reasoning_template", "reasoning_item_id",
        "reasoning_length", "min_flush_chars",
    )

    def __init__(self, min_flush_chars: int = 1) -> None:
        self.text_buffer: list[str] = []
        self.text_template: Optional[dict[str, Any]] = None
        self.text_length: int = 0
        self.reasoning_buffer: list[str] = []
        self.reasoning_template: Optional[dict[str, Any]] = None
        self.reasoning_item_id: Optional[str] = None
        self.reasoning_length: int = 0
        self.min_flush_chars: int = max(1, min_flush_chars)

    # -- properties ---------------------------------------------------------

    @property
    def has_buffered(self) -> bool:
        return bool(self.text_buffer or self.reasoning_buffer)

    # -- flush helpers ------------------------------------------------------

    def flush_text(self, force: bool = True) -> Optional[dict[str, Any]]:
        if not self.text_buffer:
            return None
        if not force and self.text_length < self.min_flush_chars:
            return None
        combined = "".join(self.text_buffer)
        base = dict(self.text_template or {"type": "response.output_text.delta"})
        base["delta"] = combined
        self.text_buffer = []
        self.text_template = None
        self.text_length = 0
        return base

    def flush_reasoning(self, force: bool = True) -> Optional[dict[str, Any]]:
        if not self.reasoning_buffer:
            return None
        if not force and self.reasoning_length < self.min_flush_chars:
            return None
        combined = "".join(self.reasoning_buffer)
        base = dict(self.reasoning_template or {"type": "response.reasoning_text.delta"})
        base["delta"] = combined
        self.reasoning_buffer = []
        self.reasoning_template = None
        self.reasoning_length = 0
        return base

    def flush_all_to(
        self,
        target: list[dict[str, Any]],
        *,
        force: bool = True,
    ) -> None:
        """Flush both buffers into *target*, preserving text-before-reasoning order."""
        for flush_fn in (self.flush_text, self.flush_reasoning):
            flushed = flush_fn(force=force)
            if flushed:
                target.append(flushed)

    # -- event routing ------------------------------------------------------

    def process_event(
        self,
        event: dict[str, Any],
        yield_queue: list[dict[str, Any]],
        *,
        passthrough: bool = False,
    ) -> None:
        """Route one event: buffer batchable deltas, pass through the rest."""
        etype = event.get("type")

        # --- Batchable text delta ---
        if etype in _BATCHABLE_TEXT_TYPES:
            delta_chunk = event.get("delta", "")
            if passthrough:
                if delta_chunk:
                    yield_queue.append(event)
                return
            # Type boundary: flush reasoning before buffering text
            flushed = self.flush_reasoning(force=True)
            if flushed:
                yield_queue.append(flushed)
            if delta_chunk:
                self.text_buffer.append(delta_chunk)
                self.text_length += len(delta_chunk)
                if self.text_template is None:
                    self.text_template = {
                        k: v for k, v in event.items() if k != "delta"
                    }
            return

        # --- Batchable reasoning delta ---
        if etype in _BATCHABLE_REASONING_TYPES:
            delta_chunk = event.get("delta", "")
            current_item_id = event.get("item_id")
            if passthrough:
                if delta_chunk:
                    yield_queue.append(event)
                return
            # Type boundary: flush text before buffering reasoning
            flushed = self.flush_text(force=True)
            if flushed:
                yield_queue.append(flushed)
            # Item_id boundary: flush reasoning if switching items
            if self.reasoning_buffer and self.reasoning_item_id != current_item_id:
                flushed = self.flush_reasoning(force=True)
                if flushed:
                    yield_queue.append(flushed)
            if delta_chunk:
                self.reasoning_buffer.append(delta_chunk)
                self.reasoning_length += len(delta_chunk)
                self.reasoning_item_id = current_item_id
                if self.reasoning_template is None:
                    self.reasoning_template = {
                        k: v for k, v in event.items() if k != "delta"
                    }
            return

        # --- Non-batchable: flush everything, pass through ---
        self.flush_all_to(yield_queue)
        yield_queue.append(event)


# ---------------------------------------------------------------------------
# nagle_coalesce_stream — async generator wrapper
# ---------------------------------------------------------------------------


async def nagle_coalesce_stream(
    source: AsyncGenerator[dict[str, Any], None],
    *,
    idle_flush_seconds: float | None = 0.03,
    passthrough: bool = False,
    min_flush_chars: int = 1,
) -> AsyncGenerator[dict[str, Any], None]:
    """Wrap an async generator with Nagle-style multi-buffer coalescing.

    Used for streams that lack their own internal queue (e.g. the Chat
    Completions SSE path).  Creates a lightweight pump task to drain
    *source* into a :class:`asyncio.Queue`, then applies the standard
    Nagle drain + :class:`NagleCoalescer` processing on the consumer
    side.
    """
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    coalescer = NagleCoalescer(min_flush_chars=min_flush_chars)

    async def _pump() -> None:
        try:
            async for event in source:
                await queue.put(event)
        except BaseException:
            logger.debug("nagle_coalesce_stream pump error", exc_info=True)
            raise
        finally:
            await queue.put(None)  # sentinel

    pump_task = asyncio.create_task(_pump(), name="nagle-pump")
    try:
        while True:
            # Idle timeout when buffers have content
            timeout = (
                idle_flush_seconds
                if (idle_flush_seconds and coalescer.has_buffered)
                else None
            )
            timed_out = False

            if timeout is not None:
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    event = None
            else:
                event = await queue.get()

            # -- idle flush --
            if timed_out:
                yield_queue: list[dict[str, Any]] = []
                coalescer.flush_all_to(yield_queue)  # force=True (default)
                for item in yield_queue:
                    yield item
                continue

            # -- sentinel: source exhausted --
            if event is None:
                break

            yield_queue: list[dict[str, Any]] = []
            source_done = False
            coalescer.process_event(
                event, yield_queue, passthrough=passthrough,
            )

            drained = 0
            while not yield_queue and drained < _MAX_DRAIN_PER_CYCLE:
                try:
                    extra = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if extra is None:
                    source_done = True
                    break
                drained += 1
                coalescer.process_event(
                    extra, yield_queue, passthrough=passthrough,
                )

            coalescer.flush_all_to(yield_queue, force=False)

            for item in yield_queue:
                yield item

            if source_done:
                break

        # -- final flush (unconditional) --
        final: list[dict[str, Any]] = []
        coalescer.flush_all_to(final)  # force=True
        for item in final:
            yield item

        # -- propagate source exceptions --
        # The pump sends a None sentinel even on error (via finally), so the
        # consumer loop exits normally.  We must check the pump task for a
        # stored exception and re-raise it so callers see the original error.
        if pump_task.done() and not pump_task.cancelled():
            pump_exc = pump_task.exception()
            if pump_exc is not None:
                raise pump_exc

    finally:
        if not pump_task.done():
            pump_task.cancel()
        await asyncio.gather(pump_task, return_exceptions=True)
