"""Streaming response processing subsystem.

This package contains streaming-related functionality:
- streaming_core: Main streaming loops and endpoint selection
- sse_parser: Server-Sent Events parsing
- event_emitter: Event emission and middleware stream handling

The streaming subsystem manages real-time response processing, including
SSE parsing, delta accumulation, tool call extraction, and event emission.
"""

from .streaming_core import StreamingHandler
from .sse_parser import SSEParser
from .event_emitter import EventEmitterHandler

__all__ = [
    "StreamingHandler",
    "SSEParser",
    "EventEmitterHandler",
]
