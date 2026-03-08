"""Event emitter wrapper that copies thinking and tool events for Think Streaming.

The wrapper is a callable class that sits between ``streaming_core`` and the
real middleware stream emitter.  It inspects each event, copies thinking
events and tool card events to the per-session ``asyncio.Queue`` for the
live iframe, AND passes them through to the original emitter so OWUI can
build its ``output`` list for DB persistence.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .session import ThinkSession


class ThinkStreamingEmitterWrapper:
    """Wraps a stream emitter to intercept and redirect thinking + tool events.

    Attributes:
        _think_streaming_active: Class-level flag checked by
            ``ThinkStreamingPlugin.on_emitter_wrap`` to prevent double-wrapping.
    """

    _think_streaming_active: bool = True

    __slots__ = ("_iframe_emitted", "_iframe_html", "_original", "_raw_emitter", "_session")

    def __init__(
        self,
        original: Any,
        session: ThinkSession,
        *,
        raw_emitter: Any | None = None,
        iframe_html: str = "",
    ) -> None:
        self._original = original
        self._session = session
        self._raw_emitter = raw_emitter
        self._iframe_html = iframe_html
        self._iframe_emitted = False

    async def __call__(self, event: dict[str, Any] | Any) -> None:
        if not isinstance(event, dict):
            await self._original(event)
            return

        etype = event.get("type")

        # Thinking events → emit iframe on first event, copy to SSE queue,
        # AND pass through to OWUI for DB persistence.
        if etype in ("reasoning:delta", "reasoning:completed"):
            await self._maybe_emit_iframe()
            self._route(event)
            await self._original(event)
            return

        # Thinking status events → copy + pass through
        if etype == "status":
            data = event.get("data", {})
            desc = data.get("description", "") if isinstance(data, dict) else ""
            if self._is_thinking_status(desc):
                self._route(event)
                await self._original(event)
                return

        # Tool card events → emit iframe on first event, copy to SSE queue,
        # AND pass through to OWUI.
        if etype == "response.output_item.added":
            item = event.get("item", {})
            if isinstance(item, dict) and item.get("type") in (
                "function_call",
                "function_call_output",
            ):
                await self._maybe_emit_iframe()
                self._route(event)
                await self._original(event)
                return

        # Everything else passes through unchanged
        await self._original(event)

    async def _maybe_emit_iframe(self) -> None:
        """Emit the iframe embed on the first thinking or tool event.

        Deferred emission means the iframe only appears when there is
        actual content to show.  The SSE queue buffers events until the
        iframe connects.
        """
        if self._iframe_emitted:
            return
        self._iframe_emitted = True
        if self._raw_emitter is not None and self._iframe_html:
            try:
                await self._raw_emitter({
                    "type": "embeds",
                    "data": {"embeds": [self._iframe_html]},
                })
            except Exception:
                pass  # Don't break the stream — iframe is optional

    def _route(self, event: dict[str, Any]) -> None:
        """Non-blocking push to session queue for the SSE iframe consumer.

        Events are always passed through to the original emitter by the
        caller, so no fallback forwarding is needed here.  If the consumer
        has disconnected we simply skip the queue push.
        """
        if not self._session.consumer_alive:
            return

        try:
            simplified = json.dumps(self._simplify(event))
            self._session.queue.put_nowait(simplified)
        except asyncio.QueueFull:
            pass  # Bounded queue protects memory — drop oldest-style

    @staticmethod
    def _simplify(event: dict[str, Any]) -> dict[str, Any]:
        """Convert internal event to compact SSE-friendly format."""
        etype = event.get("type")
        if etype == "reasoning:delta":
            data = event.get("data", {})
            return {
                "type": "thinking_delta",
                "delta": data.get("delta", ""),
                "content": data.get("content", ""),
            }
        if etype == "reasoning:completed":
            data = event.get("data", {})
            return {"type": "thinking_done", "content": data.get("content", "")}
        if etype == "status":
            data = event.get("data", {})
            return {"type": "thinking_status", "text": data.get("description", "")}
        if etype == "response.output_item.added":
            item = event.get("item", {})
            if not isinstance(item, dict):
                return {"type": "unknown"}
            item_type = item.get("type", "")
            if item_type == "function_call":
                simplified: dict[str, Any] = {
                    "type": "tool_start",
                    "name": item.get("name", ""),
                    "call_id": item.get("call_id", ""),
                }
                args = item.get("arguments")
                if isinstance(args, str) and args:
                    simplified["arguments"] = args
                return simplified
            if item_type == "function_call_output":
                simplified = {
                    "type": "tool_done",
                    "call_id": item.get("call_id", ""),
                    "status": item.get("status", "completed"),
                }
                output = item.get("output")
                if isinstance(output, str) and output:
                    simplified["output"] = output[:2000]  # Cap to avoid huge payloads
                return simplified
        return {"type": "unknown"}

    @staticmethod
    def _is_thinking_status(desc: str) -> bool:
        """Check if a status description is thinking-related.

        Matches the output patterns of ``ReasoningStatusThrottle`` from
        ``streaming_core.py``.  These status messages are emitted via
        ``_maybe_emit_reasoning_status`` and always describe reasoning
        activity.

        Conservative default: only suppress events that we are certain
        are thinking status.  This can be refined once we see the exact
        output patterns in production.
        """
        if not desc:
            return False
        return False

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the original emitter.

        This preserves ``flush_reasoning_status`` and any other attributes
        that downstream code may access via ``getattr(emitter, ...)``.
        """
        return getattr(self._original, name)
