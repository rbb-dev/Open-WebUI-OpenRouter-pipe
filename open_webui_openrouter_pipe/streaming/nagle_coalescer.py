from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

logger = logging.getLogger(__name__)

_BATCHABLE_TEXT_TYPES = frozenset({"response.output_text.delta"})
_BATCHABLE_REASONING_TYPES = frozenset({
    "response.reasoning_text.delta",
    "response.reasoning.delta",
})
_TEXT_ROUTING_KEYS = ("item_id", "output_index", "content_index")

_MAX_DRAIN_PER_CYCLE = 32


class NagleCoalescer:
    __slots__ = (
        "min_flush_chars",
        "reasoning_buffer",
        "reasoning_item_id",
        "reasoning_length",
        "reasoning_template",
        "text_buffer",
        "text_length",
        "text_route",
        "text_template",
    )

    def __init__(self, min_flush_chars: int = 1) -> None:
        self.text_buffer: list[str] = []
        self.text_template: dict[str, Any] | None = None
        self.text_route: tuple[Any, ...] | None = None
        self.text_length: int = 0
        self.reasoning_buffer: list[str] = []
        self.reasoning_template: dict[str, Any] | None = None
        self.reasoning_item_id: str | None = None
        self.reasoning_length: int = 0
        self.min_flush_chars: int = max(1, min_flush_chars)

    @property
    def has_buffered(self) -> bool:
        return bool(self.text_buffer or self.reasoning_buffer)

    def flush_text(self, force: bool = True) -> dict[str, Any] | None:
        if not self.text_buffer:
            return None
        if not force and self.text_length < self.min_flush_chars:
            return None
        combined = "".join(self.text_buffer)
        base = dict(self.text_template or {"type": "response.output_text.delta"})
        base["delta"] = combined
        self.text_buffer = []
        self.text_template = None
        self.text_route = None
        self.text_length = 0
        return base

    def flush_reasoning(self, force: bool = True) -> dict[str, Any] | None:
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
        for flush_fn in (self.flush_text, self.flush_reasoning):
            flushed = flush_fn(force=force)
            if flushed:
                target.append(flushed)

    def process_event(
        self,
        event: dict[str, Any],
        yield_queue: list[dict[str, Any]],
        *,
        passthrough: bool = False,
    ) -> None:
        etype = event.get("type")

        if etype in _BATCHABLE_TEXT_TYPES:
            delta_chunk = event.get("delta", "")
            if passthrough:
                if delta_chunk:
                    yield_queue.append(event)
                return
            flushed = self.flush_reasoning(force=True)
            if flushed:
                yield_queue.append(flushed)
            route = tuple(event.get(key) for key in _TEXT_ROUTING_KEYS)
            if self.text_buffer and self.text_route != route:
                flushed = self.flush_text(force=True)
                if flushed:
                    yield_queue.append(flushed)
            if delta_chunk:
                self.text_route = route
                self.text_buffer.append(delta_chunk)
                self.text_length += len(delta_chunk)
                if self.text_template is None:
                    self.text_template = {
                        k: v for k, v in event.items() if k != "delta"
                    }
            return

        if etype in _BATCHABLE_REASONING_TYPES:
            delta_chunk = event.get("delta", "")
            current_item_id = event.get("item_id")
            if passthrough:
                if delta_chunk:
                    yield_queue.append(event)
                return
            flushed = self.flush_text(force=True)
            if flushed:
                yield_queue.append(flushed)
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

        self.flush_all_to(yield_queue)
        yield_queue.append(event)


def idle_flush_timeout(coalescer: Any, idle_flush_seconds: float | None) -> float | None:
    if idle_flush_seconds and coalescer.has_buffered:
        return idle_flush_seconds
    return None


async def nagle_coalesce_stream(
    source: AsyncGenerator[dict[str, Any], None],
    *,
    idle_flush_seconds: float | None = 0.03,
    passthrough: bool = False,
    min_flush_chars: int = 1,
    warn_size: int = 0,
    warn_sink: Callable[[int], None] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
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
            await queue.put(None)

    pump_task = asyncio.create_task(_pump(), name="nagle-pump")
    try:
        while True:
            timeout = idle_flush_timeout(coalescer, idle_flush_seconds)
            timed_out = False

            if timeout is not None:
                try:
                    event = await asyncio.wait_for(
                        queue.get(), timeout=timeout,
                    )
                except TimeoutError:
                    timed_out = True
                    event = None
            else:
                event = await queue.get()

            if timed_out:
                yield_queue: list[dict[str, Any]] = []
                coalescer.flush_all_to(yield_queue)
                for item in yield_queue:
                    yield item
                continue

            if event is None:
                break

            if warn_size > 0 and warn_sink is not None and queue.qsize() >= warn_size:
                warn_sink(queue.qsize())

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

        final: list[dict[str, Any]] = []
        coalescer.flush_all_to(final)
        for item in final:
            yield item

        if pump_task.done() and not pump_task.cancelled():
            pump_exc = pump_task.exception()
            if pump_exc is not None:
                raise pump_exc

    finally:
        if not pump_task.done():
            pump_task.cancel()
        await asyncio.gather(pump_task, return_exceptions=True)
