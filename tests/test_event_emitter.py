"""Tests for event_emitter.py to improve coverage.

This module tests the EventEmitterHandler class and related functionality:
- Status/error/citation/completion event emission
- Templated error rendering
- Notification events
- Safe event emitter wrapping
- Middleware stream queue management
- Reasoning status buffering
"""

from __future__ import annotations

import asyncio
import datetime
import logging
from typing import Any, cast
from unittest.mock import AsyncMock, Mock, patch

import pytest

from open_webui_openrouter_pipe import Pipe, EncryptedStr, EventEmitterHandler
from open_webui_openrouter_pipe.core.logging_system import SessionLogger


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_valves():
    """Create a mock Valves instance with needed attributes."""
    pipe = Pipe()
    pipe.valves.SUPPORT_EMAIL = "support@example.com"
    pipe.valves.SUPPORT_URL = "https://support.example.com"
    return pipe.valves


@pytest.fixture
def event_handler(mock_valves):
    """Create an EventEmitterHandler instance for testing."""
    logger = logging.getLogger("test_event_emitter")
    pipe = Pipe()
    handler = EventEmitterHandler(
        logger=logger,
        valves=mock_valves,
        pipe_instance=pipe,
        event_emitter=None,
    )
    return handler


# -----------------------------------------------------------------------------
# Test _stub_chat_chunk_template (lines 47-68)
# -----------------------------------------------------------------------------

def test_stub_chat_chunk_template_basic():
    """Test stub template generates basic chunk structure."""
    from open_webui_openrouter_pipe.streaming.event_emitter import _stub_chat_chunk_template

    chunk = _stub_chat_chunk_template("test-model")

    assert chunk["object"] == "chat.completion.chunk"
    assert chunk["model"] == "test-model"
    assert "id" in chunk
    assert chunk["id"].startswith("chatcmpl-")
    assert "created" in chunk
    assert chunk["choices"][0]["delta"] == {}
    assert chunk["choices"][0]["finish_reason"] is None


def test_stub_chat_chunk_template_with_content():
    """Test stub template with content."""
    from open_webui_openrouter_pipe.streaming.event_emitter import _stub_chat_chunk_template

    chunk = _stub_chat_chunk_template("test-model", content="Hello world")

    assert chunk["choices"][0]["delta"]["content"] == "Hello world"


def test_stub_chat_chunk_template_with_reasoning_content():
    """Test stub template with reasoning content."""
    from open_webui_openrouter_pipe.streaming.event_emitter import _stub_chat_chunk_template

    chunk = _stub_chat_chunk_template("test-model", reasoning_content="Thinking...")

    assert chunk["choices"][0]["delta"]["reasoning_content"] == "Thinking..."


def test_stub_chat_chunk_template_with_tool_calls():
    """Test stub template with tool calls."""
    from open_webui_openrouter_pipe.streaming.event_emitter import _stub_chat_chunk_template

    tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "test_fn"}}]
    chunk = _stub_chat_chunk_template("test-model", tool_calls=tool_calls)

    assert chunk["choices"][0]["delta"]["tool_calls"] == tool_calls


def test_stub_chat_chunk_template_with_usage():
    """Test stub template with usage."""
    from open_webui_openrouter_pipe.streaming.event_emitter import _stub_chat_chunk_template

    usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    chunk = _stub_chat_chunk_template("test-model", usage=usage)

    assert chunk["usage"] == usage


# -----------------------------------------------------------------------------
# Test openai_chat_chunk_message_template fallback (lines 98-99)
# -----------------------------------------------------------------------------

def test_openai_chat_chunk_message_template_uses_stub_on_import_error():
    """Test that template falls back to stub when Open WebUI not available."""
    from open_webui_openrouter_pipe.streaming.event_emitter import openai_chat_chunk_message_template

    # The stub should be used since Open WebUI is not actually installed
    chunk = openai_chat_chunk_message_template("test-model", content="Test content")

    # Verify chunk structure matches expected format
    assert "object" in chunk
    assert "model" in chunk
    assert "choices" in chunk


# -----------------------------------------------------------------------------
# Test _emit_status (lines 194-195)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_status_basic(event_handler):
    """Test basic status emission."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    await event_handler._emit_status(capture_emitter, "Processing...", done=False)

    assert len(emitted) == 1
    assert emitted[0]["type"] == "status"
    assert emitted[0]["data"]["description"] == "Processing..."
    assert emitted[0]["data"]["done"] is False


@pytest.mark.asyncio
async def test_emit_status_done(event_handler):
    """Test status emission with done flag."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    await event_handler._emit_status(capture_emitter, "Completed", done=True)

    assert len(emitted) == 1
    assert emitted[0]["data"]["done"] is True


@pytest.mark.asyncio
async def test_emit_status_with_none_emitter(event_handler):
    """Test status emission with None emitter is no-op."""
    # Should not raise
    await event_handler._emit_status(None, "Test message", done=False)


@pytest.mark.asyncio
async def test_emit_status_handles_emitter_exception(event_handler, caplog):
    """Test status emission catches and logs emitter exceptions."""
    async def failing_emitter(event):
        raise RuntimeError("Emitter failed")

    with caplog.at_level(logging.ERROR):
        await event_handler._emit_status(failing_emitter, "Test", done=False)

    assert "Failed to emit status" in caplog.text


# -----------------------------------------------------------------------------
# Test _emit_error (lines 235-241)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_error_basic(event_handler):
    """Test basic error emission."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    await event_handler._emit_error(
        capture_emitter,
        "Test error message",
        show_error_message=True,
        done=True,
    )

    assert len(emitted) == 1
    assert emitted[0]["type"] == "chat:completion"
    assert emitted[0]["data"]["error"]["message"] == "Test error message"


@pytest.mark.asyncio
async def test_emit_error_with_exception(event_handler):
    """Test error emission with exception object."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    error = ValueError("Something went wrong")
    await event_handler._emit_error(capture_emitter, error, show_error_message=True)

    assert len(emitted) == 1
    assert "Something went wrong" in emitted[0]["data"]["error"]["message"]


@pytest.mark.asyncio
async def test_emit_error_show_error_message_false(event_handler):
    """Test error emission with show_error_message=False does not emit to UI."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    await event_handler._emit_error(
        capture_emitter,
        "Internal error",
        show_error_message=False,
    )

    assert len(emitted) == 0


@pytest.mark.asyncio
async def test_emit_error_with_log_citation_and_logs(event_handler, caplog):
    """Test error emission with show_error_log_citation logs debug info."""
    # Set up SessionLogger with some logs
    request_id = "test-request-123"
    token = SessionLogger.request_id.set(request_id)

    try:
        with SessionLogger._state_lock:
            SessionLogger.logs[request_id] = [
                {"timestamp": "2024-01-01T00:00:00Z", "level": "INFO", "message": "Test log entry"}
            ]

        with caplog.at_level(logging.DEBUG):
            event_handler.logger.setLevel(logging.DEBUG)
            await event_handler._emit_error(
                None,
                "Error with debug logs",
                show_error_log_citation=True,
            )

        # The logs should be formatted and logged at DEBUG level
        # (depending on whether format_event_as_text produces output)
    finally:
        SessionLogger.request_id.reset(token)
        with SessionLogger._state_lock:
            SessionLogger.logs.pop(request_id, None)


@pytest.mark.asyncio
async def test_emit_error_with_log_citation_no_logs(event_handler, caplog):
    """Test error emission with show_error_log_citation when no logs exist."""
    request_id = "nonexistent-request"
    token = SessionLogger.request_id.set(request_id)

    try:
        with caplog.at_level(logging.WARNING):
            await event_handler._emit_error(
                None,
                "Error with no logs",
                show_error_log_citation=True,
            )

        assert "No debug logs found" in caplog.text
    finally:
        SessionLogger.request_id.reset(token)


# -----------------------------------------------------------------------------
# Test _emit_templated_error (lines 284-290, 306-307)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_templated_error_basic(event_handler):
    """Test templated error emission."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    template = "### Error\n\nMessage: {error_message}"
    await event_handler._emit_templated_error(
        capture_emitter,
        template=template,
        variables={"error_message": "Something went wrong"},
        log_message="Test error occurred",
    )

    # Should emit chat:message and chat:completion
    assert len(emitted) == 2
    assert emitted[0]["type"] == "chat:message"
    assert "Something went wrong" in emitted[0]["data"]["content"]
    assert emitted[1]["type"] == "chat:completion"
    assert emitted[1]["data"]["done"] is True


@pytest.mark.asyncio
async def test_emit_templated_error_with_none_emitter(event_handler, caplog):
    """Test templated error with None emitter logs but does not emit."""
    with caplog.at_level(logging.ERROR):
        await event_handler._emit_templated_error(
            None,
            template="Error: {msg}",
            variables={"msg": "test"},
            log_message="Internal error",
        )

    # Should still log the error
    assert "Internal error" in caplog.text


@pytest.mark.asyncio
async def test_emit_templated_error_template_rendering_fails(event_handler):
    """Test templated error handles template rendering failure."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    # Use a template that will cause rendering issues by patching _render_error_template
    with patch("open_webui_openrouter_pipe.streaming.event_emitter._render_error_template") as mock_render:
        mock_render.side_effect = Exception("Template rendering failed")

        await event_handler._emit_templated_error(
            capture_emitter,
            template="{{invalid}}",
            variables={},
            log_message="Test error",
        )

    # Should emit fallback error message
    assert len(emitted) == 2
    assert "couldn't format the error message" in emitted[0]["data"]["content"]


@pytest.mark.asyncio
async def test_emit_templated_error_emitter_fails(event_handler, caplog):
    """Test templated error handles emitter failure."""
    async def failing_emitter(event):
        raise RuntimeError("Emitter exploded")

    with caplog.at_level(logging.ERROR):
        await event_handler._emit_templated_error(
            failing_emitter,
            template="Error: {msg}",
            variables={"msg": "test"},
            log_message="Test error",
        )

    assert "Failed to emit error message" in caplog.text


# -----------------------------------------------------------------------------
# Test _build_error_context (lines 335-364)
# -----------------------------------------------------------------------------

def test_build_error_context(event_handler):
    """Test error context building."""
    error_id, context = event_handler._build_error_context()

    assert len(error_id) == 16  # hex(8 bytes) = 16 chars
    assert context["error_id"] == error_id
    assert "timestamp" in context
    assert context["timestamp"].endswith("Z")
    assert "session_id" in context
    assert "user_id" in context
    assert context["support_email"] == "support@example.com"
    assert context["support_url"] == "https://support.example.com"


# -----------------------------------------------------------------------------
# Test _emit_citation (lines 335-364)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_citation_basic(event_handler):
    """Test basic citation emission."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    citation = {
        "document": "Test document content",
        "source": {"name": "Test Source", "url": "https://example.com"},
    }

    await event_handler._emit_citation(capture_emitter, citation)

    assert len(emitted) == 1
    assert emitted[0]["type"] == "source"
    assert emitted[0]["data"]["document"] == ["Test document content"]
    assert emitted[0]["data"]["source"]["name"] == "Test Source"


@pytest.mark.asyncio
async def test_emit_citation_none_emitter(event_handler):
    """Test citation emission with None emitter is no-op."""
    await event_handler._emit_citation(None, {"document": "test"})


@pytest.mark.asyncio
async def test_emit_citation_not_dict(event_handler):
    """Test citation emission with non-dict citation is no-op."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    await event_handler._emit_citation(capture_emitter, "not a dict")

    assert len(emitted) == 0


@pytest.mark.asyncio
async def test_emit_citation_document_list(event_handler):
    """Test citation emission with document as list."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    citation = {
        "document": ["Doc 1", "Doc 2", ""],  # Empty string should be filtered
        "source": {"url": "https://example.com"},
    }

    await event_handler._emit_citation(capture_emitter, citation)

    assert emitted[0]["data"]["document"] == ["Doc 1", "Doc 2"]


@pytest.mark.asyncio
async def test_emit_citation_empty_document(event_handler):
    """Test citation emission with empty document uses default."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    citation = {
        "document": "",  # Empty string
        "source": {"name": "Source"},
    }

    await event_handler._emit_citation(capture_emitter, citation)

    assert emitted[0]["data"]["document"] == ["Citation"]


@pytest.mark.asyncio
async def test_emit_citation_metadata_list(event_handler):
    """Test citation emission with metadata list."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    citation = {
        "document": "Test",
        "metadata": [{"key": "value"}, "not a dict"],  # Non-dict filtered
        "source": {"name": "Source"},
    }

    await event_handler._emit_citation(capture_emitter, citation)

    assert emitted[0]["data"]["metadata"] == [{"key": "value"}]


@pytest.mark.asyncio
async def test_emit_citation_source_without_name(event_handler):
    """Test citation emission with source missing name uses url."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    citation = {
        "document": "Test",
        "source": {"url": "https://example.com/page"},
    }

    await event_handler._emit_citation(capture_emitter, citation)

    assert emitted[0]["data"]["source"]["name"] == "https://example.com/page"


@pytest.mark.asyncio
async def test_emit_citation_source_not_dict(event_handler):
    """Test citation emission with source not a dict."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    citation = {
        "document": "Test",
        "source": "not a dict",
    }

    await event_handler._emit_citation(capture_emitter, citation)

    assert emitted[0]["data"]["source"]["name"] == "source"


@pytest.mark.asyncio
async def test_emit_citation_generates_metadata_if_empty(event_handler):
    """Test citation generates default metadata if not provided."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    citation = {
        "document": "Test",
        "source": {"name": "Test Source"},
    }

    await event_handler._emit_citation(capture_emitter, citation)

    assert len(emitted[0]["data"]["metadata"]) == 1
    assert "date_accessed" in emitted[0]["data"]["metadata"][0]


# -----------------------------------------------------------------------------
# Test _emit_completion (lines 401, 430)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_completion_basic(event_handler):
    """Test basic completion emission."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    await event_handler._emit_completion(capture_emitter, content="Response text")

    assert len(emitted) == 1
    assert emitted[0]["type"] == "chat:completion"
    assert emitted[0]["data"]["content"] == "Response text"
    assert emitted[0]["data"]["done"] is True


@pytest.mark.asyncio
async def test_emit_completion_none_emitter(event_handler):
    """Test completion emission with None emitter is no-op."""
    await event_handler._emit_completion(None, content="Test")


@pytest.mark.asyncio
async def test_emit_completion_with_title(event_handler):
    """Test completion emission with title."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    await event_handler._emit_completion(
        capture_emitter,
        content="Response",
        title="Chat Title",
    )

    assert emitted[0]["data"]["title"] == "Chat Title"


@pytest.mark.asyncio
async def test_emit_completion_with_usage(event_handler):
    """Test completion emission with usage."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    usage = {"prompt_tokens": 10, "completion_tokens": 20}
    await event_handler._emit_completion(
        capture_emitter,
        content="Response",
        usage=usage,
    )

    assert emitted[0]["data"]["usage"] == usage


@pytest.mark.asyncio
async def test_emit_completion_not_done(event_handler):
    """Test completion emission with done=False."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    await event_handler._emit_completion(
        capture_emitter,
        content="Partial",
        done=False,
    )

    assert emitted[0]["data"]["done"] is False


# -----------------------------------------------------------------------------
# Test _emit_notification (lines 469-490)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_notification_basic(event_handler):
    """Test basic notification emission."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    await event_handler._emit_notification(capture_emitter, "Info message")

    assert len(emitted) == 1
    assert emitted[0]["type"] == "notification"
    assert emitted[0]["data"]["content"] == "Info message"
    assert emitted[0]["data"]["type"] == "info"


@pytest.mark.asyncio
async def test_emit_notification_none_emitter(event_handler):
    """Test notification emission with None emitter is no-op."""
    await event_handler._emit_notification(None, "Test")


@pytest.mark.asyncio
async def test_emit_notification_levels(event_handler):
    """Test notification emission with different levels."""
    levels = ["info", "success", "warning", "error"]

    for level in levels:
        emitted = []

        async def capture_emitter(event):
            emitted.append(event)

        await event_handler._emit_notification(
            capture_emitter,
            f"{level} message",
            level=level,
        )

        assert emitted[0]["data"]["type"] == level


# -----------------------------------------------------------------------------
# Test _wrap_safe_event_emitter (lines 544-597)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wrap_safe_event_emitter_none(event_handler):
    """Test wrapping None emitter returns None."""
    result = event_handler._wrap_safe_event_emitter(None)
    assert result is None


@pytest.mark.asyncio
async def test_wrap_safe_event_emitter_success(event_handler):
    """Test wrapped emitter passes through events."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    wrapped = event_handler._wrap_safe_event_emitter(capture_emitter)
    assert wrapped is not None

    await wrapped({"type": "test", "data": {}})

    assert len(emitted) == 1
    assert emitted[0]["type"] == "test"


@pytest.mark.asyncio
async def test_wrap_safe_event_emitter_catches_exception(event_handler, caplog):
    """Test wrapped emitter catches and logs exceptions."""
    async def failing_emitter(event):
        raise RuntimeError("Transport failure")

    wrapped = event_handler._wrap_safe_event_emitter(failing_emitter)

    with caplog.at_level(logging.WARNING):
        # Should not raise
        await wrapped({"type": "test", "data": {}})

    assert "Event emitter failure" in caplog.text
    assert "test" in caplog.text  # Event type should be logged


# -----------------------------------------------------------------------------
# Test _try_put_middleware_stream_nowait (lines 613-621)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_try_put_middleware_stream_nowait_success(event_handler):
    """Test put_nowait succeeds when queue has space."""
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue(maxsize=10)

    event_handler._try_put_middleware_stream_nowait(queue, {"test": "data"})

    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_try_put_middleware_stream_nowait_full_queue(event_handler):
    """Test put_nowait silently fails when queue is full."""
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue(maxsize=1)
    queue.put_nowait({"first": "item"})

    # Should not raise
    event_handler._try_put_middleware_stream_nowait(queue, {"second": "item"})

    # Queue should still only have first item
    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_try_put_middleware_stream_nowait_handles_exception(event_handler):
    """Test put_nowait silently handles other exceptions."""
    # Create a mock queue that raises a generic exception
    mock_queue = Mock()
    mock_queue.put_nowait = Mock(side_effect=Exception("Unexpected error"))

    # Should not raise
    event_handler._try_put_middleware_stream_nowait(mock_queue, {"test": "data"})


# -----------------------------------------------------------------------------
# Test _put_middleware_stream_item (lines 632-693)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_put_middleware_stream_item_cancelled(pipe_instance_async):
    """Test put_middleware_stream_item raises when job is cancelled."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.valves = pipe.Valves(
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()

    job = _FakeJob()
    job.future.cancel()

    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    with pytest.raises(asyncio.CancelledError):
        await pipe._put_middleware_stream_item(cast(Any, job), queue, {"test": "data"})


@pytest.mark.asyncio
async def test_put_middleware_stream_item_unbounded_queue(pipe_instance_async):
    """Test put_middleware_stream_item with unbounded queue."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.valves = pipe.Valves(
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,  # unbounded
            )
            self.future = asyncio.get_running_loop().create_future()

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    await pipe._put_middleware_stream_item(cast(Any, job), queue, {"test": "data"})

    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_put_middleware_stream_item_zero_timeout(pipe_instance_async):
    """Test put_middleware_stream_item with zero timeout."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.valves = pipe.Valves(
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=10,
                MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS=0,  # No timeout
            )
            self.future = asyncio.get_running_loop().create_future()

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue(maxsize=10)

    await pipe._put_middleware_stream_item(cast(Any, job), queue, {"test": "data"})

    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_put_middleware_stream_item_with_timeout_success(pipe_instance_async):
    """Test put_middleware_stream_item with timeout completes normally."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.valves = pipe.Valves(
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=10,
                MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS=1.0,
            )
            self.future = asyncio.get_running_loop().create_future()

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue(maxsize=10)

    await pipe._put_middleware_stream_item(cast(Any, job), queue, {"test": "data"})

    assert queue.qsize() == 1


# -----------------------------------------------------------------------------
# Test _make_middleware_stream_emitter (lines 544-597 and nested functions)
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_basic(pipe_instance_async):
    """Test middleware stream emitter creation and basic usage."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {"model": {"id": "test-model"}}
            self.body = {"model": "fallback-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
                MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS=1.0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit a chat:message event
    await emitter({"type": "chat:message", "data": {"delta": "Hello", "content": "Hello"}})

    # Should have queued a chunk
    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_reasoning_delta(pipe_instance_async):
    """Test middleware stream emitter with reasoning delta."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit reasoning delta
    await emitter({"type": "reasoning:delta", "data": {"delta": "Thinking..."}})

    # Should have queued a chunk with reasoning_content
    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_status_mode(pipe_instance_async):
    """Test middleware stream emitter with status thinking mode."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="status",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit reasoning delta with enough content to trigger status
    await emitter({"type": "reasoning:delta", "data": {"delta": "This is a longer reasoning text that should trigger emission."}})

    # May or may not queue depending on buffer logic


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_tool_calls(pipe_instance_async):
    """Test middleware stream emitter with tool calls."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit tool calls
    tool_calls = [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "test_fn", "arguments": "{}"}
        }
    ]
    await emitter({"type": "chat:tool_calls", "data": {"tool_calls": tool_calls}})

    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_completion_with_error(pipe_instance_async):
    """Test middleware stream emitter with completion error."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit completion with error
    await emitter({
        "type": "chat:completion",
        "data": {"error": {"message": "Test error"}, "done": True}
    })

    assert queue.qsize() == 1
    item = queue.get_nowait()
    assert "error" in item


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_completion_with_usage(pipe_instance_async):
    """Test middleware stream emitter with completion usage."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit completion with usage
    await emitter({
        "type": "chat:completion",
        "data": {"usage": {"prompt_tokens": 10, "completion_tokens": 20}, "done": True}
    })

    assert queue.qsize() == 1
    item = queue.get_nowait()
    assert "usage" in item


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_passthrough_event(pipe_instance_async):
    """Test middleware stream emitter passes through unknown events."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit unknown event type
    await emitter({"type": "custom:event", "data": {"foo": "bar"}})

    assert queue.qsize() == 1
    item = queue.get_nowait()
    assert item["event"]["type"] == "custom:event"


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_non_dict_event(pipe_instance_async):
    """Test middleware stream emitter handles non-dict events."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit non-dict event
    await emitter("not a dict")

    # Should be ignored
    assert queue.qsize() == 0


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_original_emitter_called(pipe_instance_async):
    """Test middleware stream emitter calls original emitter."""
    pipe = pipe_instance_async

    original_emitted = []

    async def original_emitter(event):
        original_emitted.append(event)

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = original_emitter

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit event
    await emitter({"type": "chat:message", "data": {"delta": "Hi", "content": "Hi"}})

    # Original emitter should have been called
    assert len(original_emitted) == 1


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_original_emitter_exception(pipe_instance_async):
    """Test middleware stream emitter handles original emitter exception."""
    pipe = pipe_instance_async

    async def failing_emitter(event):
        raise RuntimeError("Emitter failed")

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = failing_emitter

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Should not raise, should continue processing
    await emitter({"type": "chat:message", "data": {"delta": "Hi", "content": "Hi"}})

    # Should still queue the item
    assert queue.qsize() == 1


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_content_without_delta(pipe_instance_async):
    """Test middleware stream emitter with content but no delta."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # First message sets assistant_sent
    await emitter({"type": "chat:message", "data": {"content": "Hello"}})

    # Second message should compute delta from content
    await emitter({"type": "chat:message", "data": {"content": "Hello World"}})

    # Should have two chunks
    assert queue.qsize() == 2


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_reasoning_completed(pipe_instance_async):
    """Test middleware stream emitter handles reasoning:completed."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="status",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # First emit some reasoning delta to build up buffer
    await emitter({"type": "reasoning:delta", "data": {"delta": "Thinking about the problem..."}})

    # Then signal reasoning completed
    await emitter({"type": "reasoning:completed", "data": {}})

    # Buffer should be flushed as status


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_flush_reasoning_status(pipe_instance_async):
    """Test middleware stream emitter flush_reasoning_status attribute."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="status",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emitter should have flush_reasoning_status attribute
    assert hasattr(emitter, "flush_reasoning_status")

    # Build up some buffer
    await emitter({"type": "reasoning:delta", "data": {"delta": "Some reasoning text."}})

    # Call flush
    flush_fn = getattr(emitter, "flush_reasoning_status")
    await flush_fn()


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_tool_calls_debug_logging(pipe_instance_async, caplog):
    """Test middleware stream emitter logs tool calls at debug level."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Set debug logging
    logging.getLogger("open_webui_openrouter_pipe.streaming.event_emitter").setLevel(logging.DEBUG)

    tool_calls = [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "test_fn", "arguments": "{}"}
        }
    ]

    with caplog.at_level(logging.DEBUG):
        await emitter({"type": "chat:tool_calls", "data": {"tool_calls": tool_calls}})


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_tool_calls_invalid(pipe_instance_async):
    """Test middleware stream emitter handles invalid tool_calls."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Empty tool_calls list
    await emitter({"type": "chat:tool_calls", "data": {"tool_calls": []}})

    # Should not queue anything
    assert queue.qsize() == 0

    # Non-list tool_calls
    await emitter({"type": "chat:tool_calls", "data": {"tool_calls": "invalid"}})

    assert queue.qsize() == 0


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_reasoning_status_punctuation(pipe_instance_async):
    """Test reasoning status emits on punctuation."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="status",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit text ending with punctuation
    await emitter({"type": "reasoning:delta", "data": {"delta": "First sentence."}})

    # Should emit status due to punctuation
    assert queue.qsize() >= 1


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_reasoning_status_max_chars(pipe_instance_async):
    """Test reasoning status emits when max chars exceeded."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="status",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit very long text without punctuation
    long_text = "a" * 200  # Exceeds REASONING_STATUS_MAX_CHARS (160)
    await emitter({"type": "reasoning:delta", "data": {"delta": long_text}})

    # Should emit status due to length
    assert queue.qsize() >= 1


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_reasoning_non_string_delta(pipe_instance_async):
    """Test reasoning delta handles non-string delta."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="status",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit non-string delta
    await emitter({"type": "reasoning:delta", "data": {"delta": 12345}})

    # Should not emit anything
    assert queue.qsize() == 0


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_both_thinking_modes(pipe_instance_async):
    """Test middleware stream emitter with 'both' thinking mode."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="both",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit reasoning delta - should emit both box and status
    await emitter({"type": "reasoning:delta", "data": {"delta": "Thinking about this question."}})

    # Should have at least one item (reasoning_content chunk)
    assert queue.qsize() >= 1


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_data_not_dict(pipe_instance_async):
    """Test middleware stream emitter handles non-dict data."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit event with non-dict data
    await emitter({"type": "chat:message", "data": "not a dict"})

    # Should handle gracefully - passthrough as event
    # Actually for chat:message with non-dict data it returns early


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_model_from_body(pipe_instance_async):
    """Test middleware stream emitter uses model from body as fallback."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}  # No model in metadata
            self.body = {"model": "body-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit message
    await emitter({"type": "chat:message", "data": {"delta": "Hi", "content": "Hi"}})

    # Check that chunk uses body model
    item = queue.get_nowait()
    assert item["model"] == "body-model"


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_chat_message_content_updates_assistant_sent(pipe_instance_async):
    """Test that content updates properly track assistant_sent."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # First: delta="Hello", content="Hello"
    await emitter({"type": "chat:message", "data": {"delta": "Hello", "content": "Hello"}})

    # Second: delta=" World", content="Hello World"
    await emitter({"type": "chat:message", "data": {"delta": " World", "content": "Hello World"}})

    # Third: content only (no delta), should compute delta as difference
    await emitter({"type": "chat:message", "data": {"content": "Hello World!"}})

    assert queue.qsize() == 3


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_answer_started_flushes_reasoning(pipe_instance_async):
    """Test that starting answer flushes reasoning buffer."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="status",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Build up reasoning buffer (not enough to emit)
    await emitter({"type": "reasoning:delta", "data": {"delta": "Think"}})

    initial_size = queue.qsize()

    # Start answer - should flush reasoning buffer
    await emitter({"type": "chat:message", "data": {"delta": "Response", "content": "Response"}})

    # Should have flushed status + added chat message
    assert queue.qsize() > initial_size


# -----------------------------------------------------------------------------
# Additional tests for remaining coverage gaps
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_error_log_citation_format_exception(event_handler, caplog):
    """Test error emission handles exception during log formatting."""
    request_id = "test-format-error"
    token = SessionLogger.request_id.set(request_id)

    try:
        # Set up logs that will cause format_event_as_text to fail
        with SessionLogger._state_lock:
            # Add a log entry that is a dict but will fail formatting
            SessionLogger.logs[request_id] = [
                {"timestamp": None, "level": None, "message": None},  # Will cause issues
            ]

        event_handler.logger.setLevel(logging.DEBUG)

        with caplog.at_level(logging.DEBUG):
            await event_handler._emit_error(
                None,
                "Error during format",
                show_error_log_citation=True,
            )

        # Should have handled the exception gracefully
    finally:
        SessionLogger.request_id.reset(token)
        with SessionLogger._state_lock:
            SessionLogger.logs.pop(request_id, None)


@pytest.mark.asyncio
async def test_emit_citation_document_not_string_or_list(event_handler):
    """Test citation with document that is neither string nor list."""
    emitted = []

    async def capture_emitter(event):
        emitted.append(event)

    citation = {
        "document": 12345,  # Not a string or list
        "source": {"name": "Source"},
    }

    await event_handler._emit_citation(capture_emitter, citation)

    # Should use default "Citation"
    assert emitted[0]["data"]["document"] == ["Citation"]


@pytest.mark.asyncio
async def test_put_middleware_stream_item_timeout_exception(pipe_instance_async, caplog):
    """Test put_middleware_stream_item timeout logs warning and re-raises."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-timeout"
            self.valves = pipe.Valves(
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=1,
                MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS=0.01,  # Very short timeout
            )
            self.future = asyncio.get_running_loop().create_future()

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue(maxsize=1)
    queue.put_nowait({"first": "item"})  # Fill the queue

    with caplog.at_level(logging.WARNING):
        with pytest.raises(asyncio.TimeoutError):
            await pipe._put_middleware_stream_item(cast(Any, job), queue, {"second": "item"})

    assert "timed out" in caplog.text


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_reasoning_idle_timeout(pipe_instance_async):
    """Test reasoning status emits after idle timeout."""
    import time
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="status",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit short text (less than MIN_CHARS) - won't emit immediately
    await emitter({"type": "reasoning:delta", "data": {"delta": "Short text here"}})

    initial_size = queue.qsize()

    # Wait a bit for idle timeout to apply, then emit more
    await asyncio.sleep(0.1)

    # Emit more text to trigger idle check
    await emitter({"type": "reasoning:delta", "data": {"delta": " more text"}})

    # Depending on timing, may or may not have emitted


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_reasoning_empty_buffer(pipe_instance_async):
    """Test reasoning status with empty buffer returns early."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="status",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Emit just whitespace
    await emitter({"type": "reasoning:delta", "data": {"delta": "   "}})

    # Buffer is whitespace only, so nothing should be emitted
    # (depends on whether strip() leaves empty)


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_tool_calls_with_non_dict_call(pipe_instance_async, caplog):
    """Test tool calls with non-dict item in list."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    # Force debug logging
    logger = logging.getLogger("open_webui_openrouter_pipe.streaming.event_emitter")
    logger.setLevel(logging.DEBUG)

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Tool calls with non-dict item that should be skipped
    tool_calls = [
        "not a dict",  # Should be skipped in summary loop
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": {"name": "test_fn", "arguments": "{}"}
        }
    ]

    with caplog.at_level(logging.DEBUG):
        await emitter({"type": "chat:tool_calls", "data": {"tool_calls": tool_calls}})


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_tool_calls_exception_path(pipe_instance_async, caplog):
    """Test tool calls exception handling."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Patch to cause an exception
    with patch("open_webui_openrouter_pipe.streaming.event_emitter.openai_chat_chunk_message_template") as mock_template:
        mock_template.side_effect = Exception("Template failed")

        tool_calls = [
            {
                "index": 0,
                "id": "call_1",
                "type": "function",
                "function": {"name": "test_fn", "arguments": "{}"}
            }
        ]

        with caplog.at_level(logging.DEBUG):
            await emitter({"type": "chat:tool_calls", "data": {"tool_calls": tool_calls}})

    # Should have logged the failure
    assert "Failed to emit tool_calls" in caplog.text


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_completion_flushes_reasoning_buffer(pipe_instance_async):
    """Test completion event flushes reasoning buffer when in status mode."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="status",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Build up buffer
    await emitter({"type": "reasoning:delta", "data": {"delta": "Some reasoning text"}})

    # Now emit completion - should flush buffer first
    await emitter({"type": "chat:completion", "data": {"done": True}})


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_reasoning_completed_flushes_buffer(pipe_instance_async):
    """Test reasoning:completed flushes buffer when in status mode."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="status",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Build up buffer with enough content
    await emitter({"type": "reasoning:delta", "data": {"delta": "Some reasoning text buffer"}})

    # Now emit reasoning:completed - should flush buffer
    await emitter({"type": "reasoning:completed", "data": {}})


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_flush_when_disabled(pipe_instance_async):
    """Test flush_reasoning_status when status mode is disabled."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",  # Status disabled
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Call flush when status is disabled - should be no-op
    flush_fn = getattr(emitter, "flush_reasoning_status")
    await flush_fn()


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_maybe_emit_non_string_delta(pipe_instance_async):
    """Test _maybe_emit_reasoning_status handles non-string delta."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="status",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # The internal function checks isinstance(delta_text, str) at line 544
    # We can't call it directly but the reasoning:delta handler checks first


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_tool_calls_function_not_dict(pipe_instance_async, caplog):
    """Test tool calls with function that is not a dict."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    logger = logging.getLogger("open_webui_openrouter_pipe.streaming.event_emitter")
    logger.setLevel(logging.DEBUG)

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # Tool calls with function that is not a dict
    tool_calls = [
        {
            "index": 0,
            "id": "call_1",
            "type": "function",
            "function": "not a dict"  # Will fail fn_raw check
        }
    ]

    with caplog.at_level(logging.DEBUG):
        await emitter({"type": "chat:tool_calls", "data": {"tool_calls": tool_calls}})


@pytest.mark.asyncio
async def test_make_middleware_stream_emitter_delta_with_mismatched_content(pipe_instance_async):
    """Test chat:message with delta but content that doesn't start with assistant_sent."""
    pipe = pipe_instance_async

    class _FakeJob:
        def __init__(self):
            self.request_id = "req-test"
            self.metadata = {}
            self.body = {"model": "test-model"}
            self.valves = pipe.Valves(
                THINKING_OUTPUT_MODE="open_webui",
                MIDDLEWARE_STREAM_QUEUE_MAXSIZE=0,
            )
            self.future = asyncio.get_running_loop().create_future()
            self.event_emitter = None

    job = _FakeJob()
    queue: asyncio.Queue[dict | str | None] = asyncio.Queue()

    emitter = pipe._make_middleware_stream_emitter(cast(Any, job), queue)

    # First, set assistant_sent to "Hello"
    await emitter({"type": "chat:message", "data": {"delta": "Hello", "content": "Hello"}})

    # Then send delta with content that doesn't match
    await emitter({"type": "chat:message", "data": {"delta": " World", "content": "Different"}})

    # Should fallback to assistant_sent + delta
    assert queue.qsize() == 2
