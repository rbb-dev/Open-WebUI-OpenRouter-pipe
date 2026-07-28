"""Streaming response processing subsystem.

This package contains streaming-related functionality:
- streaming_core: Main streaming loops and endpoint selection
- event_emitter: Event emission and middleware stream handling

The streaming subsystem manages real-time response processing, including
SSE parsing, delta accumulation, tool call extraction, and event emission.
"""

from .event_emitter import EventEmitterHandler
from .streaming_core import StreamingHandler

__all__ = [
    "EventEmitterHandler",
    "StreamingHandler",
]
