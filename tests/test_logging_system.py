"""Tests for logging_system.py to achieve 90%+ coverage.

This test module covers:
- SessionLogger class methods (get_logger, cleanup, process_record, etc.)
- Event classification and building
- Log queue processing and enqueuing
- Session log archive writing (write_session_log_archive)
- Format helpers (format_event_as_text, etc.)
- Edge cases and error handling
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import asyncio
import datetime
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import pytest

from open_webui_openrouter_pipe.core.logging_system import (
    SessionLogger,
    _SessionLogArchiveJob,
    write_session_log_archive,
)


class TestPyzipperImportFallback:
    """Test behavior when pyzipper is not available."""

    def test_write_session_log_archive_no_pyzipper(self, tmp_path: Path) -> None:
        """write_session_log_archive returns early when pyzipper is None."""
        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"test",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user1",
            session_id="sess1",
            chat_id="chat1",
            message_id="msg1",
            request_id="req1",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[{"message": "test"}],
        )
        # Patch pyzipper to None
        with patch("open_webui_openrouter_pipe.core.logging_system.pyzipper", None):
            write_session_log_archive(job)
        # No zip file should be created
        assert not list(tmp_path.rglob("*.zip"))


class TestClassifyEventType:
    """Test event type classification based on message content."""

    def test_classify_openrouter_request_headers(self) -> None:
        result = SessionLogger._classify_event_type("OpenRouter request headers: ...")
        assert result == "openrouter.request.headers"

    def test_classify_openrouter_request_payload(self) -> None:
        result = SessionLogger._classify_event_type("OpenRouter request payload: {...}")
        assert result == "openrouter.request.payload"

    def test_classify_openrouter_sse_event(self) -> None:
        result = SessionLogger._classify_event_type("OpenRouter payload: data: {...}")
        assert result == "openrouter.sse.event"

    def test_classify_tool_message(self) -> None:
        result = SessionLogger._classify_event_type("Tool execution completed")
        assert result == "pipe.tools"

    def test_classify_tool_emoji_message(self) -> None:
        # Test with wrench emoji prefix
        result = SessionLogger._classify_event_type("\U0001f527 Running tool...")
        assert result == "pipe.tools"

    def test_classify_skipping_message(self) -> None:
        result = SessionLogger._classify_event_type("Skipping duplicate tool call")
        assert result == "pipe.tools"

    def test_classify_generic_pipe_message(self) -> None:
        result = SessionLogger._classify_event_type("Some generic message")
        assert result == "pipe"

    def test_classify_empty_message(self) -> None:
        result = SessionLogger._classify_event_type("")
        assert result == "pipe"

    def test_classify_none_message(self) -> None:
        result = SessionLogger._classify_event_type(None)  # type: ignore[arg-type]
        assert result == "pipe"

    def test_classify_message_with_leading_whitespace(self) -> None:
        result = SessionLogger._classify_event_type("   OpenRouter request headers: ...")
        assert result == "openrouter.request.headers"


class TestBuildEvent:
    """Test building structured events from LogRecord."""

    def test_build_event_basic(self) -> None:
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.request_id = "req-123"
        record.session_id = "sess-456"
        record.user_id = "user-789"

        event = SessionLogger._build_event(record)

        assert event["level"] == "INFO"
        assert event["logger"] == "test.logger"
        assert event["request_id"] == "req-123"
        assert event["session_id"] == "sess-456"
        assert event["user_id"] == "user-789"
        assert event["message"] == "Test message"
        assert event["lineno"] == 42
        assert "created" in event
        assert event["event_type"] == "pipe"

    def test_build_event_with_exc_text(self) -> None:
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=None,
        )
        record.exc_text = "Traceback (most recent call last):\n  File ..."
        record.request_id = "req-1"
        record.session_id = None
        record.user_id = None

        event = SessionLogger._build_event(record)

        assert "exception" in event
        assert event["exception"]["text"] == record.exc_text

    def test_build_event_with_exc_info(self) -> None:
        try:
            raise ValueError("Test error")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=20,
            msg="Error with exc_info",
            args=(),
            exc_info=exc_info,
        )
        record.request_id = "req-2"
        record.session_id = None
        record.user_id = None

        event = SessionLogger._build_event(record)

        assert "exception" in event
        assert "ValueError" in event["exception"]["text"]
        assert "Test error" in event["exception"]["text"]

    def test_build_event_getMessage_fails(self) -> None:
        """Test fallback when getMessage() raises."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test %s",
            args=("arg",),
            exc_info=None,
        )
        record.request_id = "req-3"
        record.session_id = None
        record.user_id = None

        # Make getMessage raise
        def raise_error():
            raise RuntimeError("getMessage failed")

        record.getMessage = raise_error  # type: ignore[assignment]

        event = SessionLogger._build_event(record)
        assert event["message"] == "test %s"

    def test_build_event_missing_attributes(self) -> None:
        """Test with minimal LogRecord without custom attributes."""
        record = logging.LogRecord(
            name="",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="",
            args=(),
            exc_info=None,
        )

        event = SessionLogger._build_event(record)

        assert event["request_id"] is None
        assert event["session_id"] is None
        assert event["user_id"] is None
        assert event["logger"] == ""
        assert event["lineno"] == 0


class TestFormatEventAsText:
    """Test text formatting of log events."""

    def test_format_event_basic(self) -> None:
        event = {
            "created": time.time(),
            "level": "INFO",
            "user_id": "user-1",
            "message": "Test message",
        }
        result = SessionLogger.format_event_as_text(event)

        assert "[INFO]" in result
        assert "[user=user-1]" in result
        assert "Test message" in result

    def test_format_event_missing_created(self) -> None:
        event = {
            "level": "WARNING",
            "user_id": "u1",
            "message": "No created field",
        }
        result = SessionLogger.format_event_as_text(event)
        assert "[WARNING]" in result
        assert "No created field" in result

    def test_format_event_invalid_created(self) -> None:
        event = {
            "created": "not-a-number",
            "level": "ERROR",
            "user_id": None,
            "message": "Invalid created",
        }
        result = SessionLogger.format_event_as_text(event)
        assert "[ERROR]" in result
        assert "[user=-]" in result

    def test_format_event_none_message(self) -> None:
        event = {
            "created": time.time(),
            "level": "DEBUG",
            "user_id": "x",
            "message": None,
        }
        result = SessionLogger.format_event_as_text(event)
        assert "[DEBUG]" in result
        # Message should be empty string

    def test_format_event_exception_in_strftime(self) -> None:
        """Test fallback when time.localtime fails."""
        event = {
            "created": time.time(),
            "level": "INFO",
            "user_id": "user",
            "message": "msg",
        }
        with patch("time.localtime", side_effect=ValueError("localtime failed")):
            result = SessionLogger.format_event_as_text(event)
        assert "[INFO]" in result

    def test_format_event_message_str_fails(self) -> None:
        """Test fallback when str(message) fails."""

        class BadStr:
            def __str__(self):
                raise RuntimeError("str failed")

        event = {
            "created": time.time(),
            "level": "INFO",
            "user_id": "u",
            "message": BadStr(),
        }
        result = SessionLogger.format_event_as_text(event)
        assert "[INFO]" in result


class TestGetLogger:
    """Test logger creation and configuration."""

    def test_get_logger_basic(self) -> None:
        logger = SessionLogger.get_logger("test.get_logger")
        assert logger.name == "test.get_logger"
        assert logger.level == logging.DEBUG
        assert logger.propagate is False
        assert any(
            type(h).__name__ == "_HostForwardHandler" for h in logger.handlers
        ), (
            "propagation is off and nothing forwards to the host, so the pipe's own "
            "records never reach Open WebUI's log at any level"
        )

    def test_get_logger_adds_null_handler_to_root(self) -> None:
        """Ensure NullHandler is added to root logger if not present."""
        root = logging.getLogger()
        # Clear all NullHandlers
        root.handlers = [h for h in root.handlers if not isinstance(h, logging.NullHandler)]

        SessionLogger.get_logger("test.null_handler")

        has_null = any(isinstance(h, logging.NullHandler) for h in root.handlers)
        assert has_null

    def test_get_logger_filter_attaches_metadata(self) -> None:
        """Test that the filter attaches session metadata to records."""
        # Set up context
        token_sid = SessionLogger.session_id.set("sid-filter-test")
        token_rid = SessionLogger.request_id.set("rid-filter-test")
        token_uid = SessionLogger.user_id.set("uid-filter-test")

        try:
            logger = SessionLogger.get_logger("test.filter_metadata")
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="test",
                args=(),
                exc_info=None,
            )
            # Apply the filter
            for handler in logger.handlers:
                for f in handler.filters:
                    f(record)

            assert getattr(record, "session_id", None) == "sid-filter-test"
            assert getattr(record, "request_id", None) == "rid-filter-test"
            assert getattr(record, "user_id", None) == "uid-filter-test"
        finally:
            SessionLogger.session_id.reset(token_sid)
            SessionLogger.request_id.reset(token_rid)
            SessionLogger.user_id.reset(token_uid)

    def test_get_logger_filter_handles_exception(self) -> None:
        """Test filter doesn't break when contextvar access fails."""
        logger = SessionLogger.get_logger("test.filter_exception")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="test",
            args=(),
            exc_info=None,
        )

        for f in logger.filters:
            result = f(record)
            assert result is True


class TestEnqueue:
    """Test log record enqueuing."""

    def test_enqueue_no_queue(self) -> None:
        """When queue is None, process_record is called directly."""
        original_queue = SessionLogger.log_queue
        try:
            SessionLogger.log_queue = None
            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="test",
                args=(),
                exc_info=None,
            )
            with patch.object(SessionLogger, "process_record") as mock_process:
                SessionLogger._enqueue(record)
                mock_process.assert_called_once_with(record)
        finally:
            SessionLogger.log_queue = original_queue

    @pytest.mark.asyncio
    async def test_enqueue_same_loop(self) -> None:
        """When in the same event loop, use _safe_put directly."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[logging.LogRecord] = asyncio.Queue()
        original_queue = SessionLogger.log_queue
        original_loop = SessionLogger._main_loop

        try:
            SessionLogger.log_queue = queue
            SessionLogger._main_loop = loop

            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="test enqueue same loop",
                args=(),
                exc_info=None,
            )
            SessionLogger._enqueue(record)

            # Record should be in queue
            assert not queue.empty()
            queued_record = await queue.get()
            assert queued_record.msg == "test enqueue same loop"
        finally:
            SessionLogger.log_queue = original_queue
            SessionLogger._main_loop = original_loop

    def test_enqueue_different_loop_threadsafe(self) -> None:
        """When in different thread, use call_soon_threadsafe."""
        original_queue = SessionLogger.log_queue
        original_loop = SessionLogger._main_loop

        try:
            mock_loop = MagicMock()
            mock_loop.is_closed.return_value = False
            queue: asyncio.Queue[logging.LogRecord] = asyncio.Queue()

            SessionLogger.log_queue = queue
            SessionLogger._main_loop = mock_loop

            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="threadsafe test",
                args=(),
                exc_info=None,
            )
            SessionLogger._enqueue(record)

            mock_loop.call_soon_threadsafe.assert_called_once()
        finally:
            SessionLogger.log_queue = original_queue
            SessionLogger._main_loop = original_loop

    def test_enqueue_closed_loop_fallback(self) -> None:
        """When main loop is closed, fall back to process_record."""
        original_queue = SessionLogger.log_queue
        original_loop = SessionLogger._main_loop

        try:
            mock_loop = MagicMock()
            mock_loop.is_closed.return_value = True
            queue: asyncio.Queue[logging.LogRecord] = asyncio.Queue()

            SessionLogger.log_queue = queue
            SessionLogger._main_loop = mock_loop

            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="closed loop test",
                args=(),
                exc_info=None,
            )

            with patch.object(SessionLogger, "process_record") as mock_process:
                SessionLogger._enqueue(record)
                mock_process.assert_called_once_with(record)
        finally:
            SessionLogger.log_queue = original_queue
            SessionLogger._main_loop = original_loop


class TestSafePut:
    """Test safe queue put operation."""

    @pytest.mark.asyncio
    async def test_safe_put_success(self) -> None:
        queue: asyncio.Queue[logging.LogRecord] = asyncio.Queue()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="safe put test",
            args=(),
            exc_info=None,
        )
        SessionLogger._safe_put(queue, record)

        assert not queue.empty()
        result = await queue.get()
        assert result.msg == "safe put test"

    @pytest.mark.asyncio
    async def test_safe_put_queue_full(self) -> None:
        """When queue is full, fall back to process_record."""
        queue: asyncio.Queue[logging.LogRecord] = asyncio.Queue(maxsize=1)
        # Fill the queue
        queue.put_nowait(
            logging.LogRecord("x", logging.INFO, "", 0, "filler", (), None)
        )

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="overflow test",
            args=(),
            exc_info=None,
        )

        with patch.object(SessionLogger, "process_record") as mock_process:
            SessionLogger._safe_put(queue, record)
            mock_process.assert_called_once_with(record)


class TestProcessRecord:
    """Test log record processing."""

    def test_process_record_does_not_write_its_own_console_line(self, capsys) -> None:
        record = logging.LogRecord(
            name="test.stdout",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="stdout test message",
            args=(),
            exc_info=None,
        )
        record.request_id = None

        SessionLogger.process_record(record)

        captured = capsys.readouterr()
        assert "stdout test message" not in captured.out, (
            "process_record wrote its own console line. It is the session-log buffer "
            "path only; the console belongs to the host's handler chain, and writing "
            "here as well is what produced two lines per record -- and a non-JSON line "
            "in a LOG_FORMAT=json stream. See "
            "test_a_record_reaches_the_console_exactly_once."
        )

    def test_process_record_stores_in_buffer(self) -> None:
        """Records with request_id are stored in logs buffer."""
        original_logs = SessionLogger.logs.copy()
        original_last_seen = SessionLogger._session_last_seen.copy()

        try:
            # Clear
            SessionLogger.logs.clear()
            SessionLogger._session_last_seen.clear()

            record = logging.LogRecord(
                name="test.buffer",
                level=logging.INFO,
                pathname="test.py",
                lineno=20,
                msg="buffer test message",
                args=(),
                exc_info=None,
            )
            record.request_id = "req-buffer-test"
            record.session_id = "sess-1"
            record.user_id = "user-1"

            SessionLogger.process_record(record)

            assert "req-buffer-test" in SessionLogger.logs
            buffer = SessionLogger.logs["req-buffer-test"]
            assert len(buffer) == 1
            assert buffer[0]["message"] == "buffer test message"
        finally:
            SessionLogger.logs.clear()
            SessionLogger.logs.update(original_logs)
            SessionLogger._session_last_seen.clear()
            SessionLogger._session_last_seen.update(original_last_seen)

    def test_process_record_creates_new_buffer(self) -> None:
        """New buffer is created when request_id is new."""
        original_logs = SessionLogger.logs.copy()
        original_last_seen = SessionLogger._session_last_seen.copy()

        try:
            SessionLogger.logs.clear()
            SessionLogger._session_last_seen.clear()

            record = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="new buffer",
                args=(),
                exc_info=None,
            )
            record.request_id = "new-request-id"
            record.session_id = None
            record.user_id = None

            SessionLogger.process_record(record)

            assert "new-request-id" in SessionLogger.logs
            assert isinstance(SessionLogger.logs["new-request-id"], deque)
        finally:
            SessionLogger.logs.clear()
            SessionLogger.logs.update(original_logs)
            SessionLogger._session_last_seen.clear()
            SessionLogger._session_last_seen.update(original_last_seen)

    def test_process_record_build_event_fails(self, capsys) -> None:
        """Test fallback when _build_event raises."""
        original_logs = SessionLogger.logs.copy()
        original_last_seen = SessionLogger._session_last_seen.copy()

        try:
            SessionLogger.logs.clear()
            SessionLogger._session_last_seen.clear()

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=99,
                msg="fallback test",
                args=(),
                exc_info=None,
            )
            record.request_id = "req-fallback"
            record.session_id = "sess-fallback"
            record.user_id = "user-fallback"

            # Mock _build_event to raise
            with patch.object(SessionLogger, "_build_event", side_effect=RuntimeError("build failed")):
                SessionLogger.process_record(record)

            # Should still store a fallback event
            assert "req-fallback" in SessionLogger.logs
            event = SessionLogger.logs["req-fallback"][0]
            assert event["message"] == "fallback test"
            assert event["level"] == "ERROR"
        finally:
            SessionLogger.logs.clear()
            SessionLogger.logs.update(original_logs)
            SessionLogger._session_last_seen.clear()
            SessionLogger._session_last_seen.update(original_last_seen)

    def test_process_record_resizes_buffer_on_maxlen_change(self) -> None:
        """Buffer is recreated if SESSION_LOG_MAX_LINES changes."""
        original_logs = SessionLogger.logs.copy()
        original_last_seen = SessionLogger._session_last_seen.copy()
        original_max = SessionLogger.SESSION_LOG_MAX_LINES

        try:
            SessionLogger.logs.clear()
            SessionLogger._session_last_seen.clear()

            SessionLogger.SESSION_LOG_MAX_LINES = 100
            record1 = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="first",
                args=(),
                exc_info=None,
            )
            record1.request_id = "req-resize"
            record1.session_id = None
            record1.user_id = None

            SessionLogger.process_record(record1)
            assert SessionLogger.logs["req-resize"].maxlen == 100

            SessionLogger.SESSION_LOG_MAX_LINES = 200
            record2 = logging.LogRecord(
                name="test",
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg="second",
                args=(),
                exc_info=None,
            )
            record2.request_id = "req-resize"
            record2.session_id = None
            record2.user_id = None

            SessionLogger.process_record(record2)
            assert SessionLogger.logs["req-resize"].maxlen == 200
        finally:
            SessionLogger.logs.clear()
            SessionLogger.logs.update(original_logs)
            SessionLogger._session_last_seen.clear()
            SessionLogger._session_last_seen.update(original_last_seen)
            SessionLogger.SESSION_LOG_MAX_LINES = original_max


class TestCleanup:
    """Test session cleanup functionality."""

    def test_cleanup_removes_stale_sessions(self) -> None:
        original_logs = SessionLogger.logs.copy()
        original_last_seen = SessionLogger._session_last_seen.copy()

        try:
            SessionLogger.logs.clear()
            SessionLogger._session_last_seen.clear()

            stale_time = time.time() - 7200
            SessionLogger.logs["stale-session"] = deque([{"message": "old"}])
            SessionLogger._session_last_seen["stale-session"] = stale_time

            # Add a fresh session
            SessionLogger.logs["fresh-session"] = deque([{"message": "new"}])
            SessionLogger._session_last_seen["fresh-session"] = time.time()

            # Cleanup with 1 hour max age
            SessionLogger.cleanup(max_age_seconds=3600)

            assert "stale-session" not in SessionLogger.logs
            assert "stale-session" not in SessionLogger._session_last_seen
            assert "fresh-session" in SessionLogger.logs
        finally:
            SessionLogger.logs.clear()
            SessionLogger.logs.update(original_logs)
            SessionLogger._session_last_seen.clear()
            SessionLogger._session_last_seen.update(original_last_seen)

    def test_cleanup_no_stale_sessions(self) -> None:
        original_logs = SessionLogger.logs.copy()
        original_last_seen = SessionLogger._session_last_seen.copy()

        try:
            SessionLogger.logs.clear()
            SessionLogger._session_last_seen.clear()

            # Add only fresh sessions
            SessionLogger.logs["fresh-1"] = deque([{"message": "msg1"}])
            SessionLogger._session_last_seen["fresh-1"] = time.time()
            SessionLogger.logs["fresh-2"] = deque([{"message": "msg2"}])
            SessionLogger._session_last_seen["fresh-2"] = time.time()

            SessionLogger.cleanup(max_age_seconds=3600)

            assert "fresh-1" in SessionLogger.logs
            assert "fresh-2" in SessionLogger.logs
        finally:
            SessionLogger.logs.clear()
            SessionLogger.logs.update(original_logs)
            SessionLogger._session_last_seen.clear()
            SessionLogger._session_last_seen.update(original_last_seen)


class TestWriteSessionLogArchive:
    """Test session log archive writing functionality."""

    def test_archive_empty_base_dir(self) -> None:
        """Returns early when base_dir is empty."""
        job = _SessionLogArchiveJob(
            base_dir="",
            zip_password=b"test",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="u1",
            session_id="s1",
            chat_id="c1",
            message_id="m1",
            request_id="r1",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[],
        )
        # Should return early without error
        write_session_log_archive(job)

    def test_archive_whitespace_base_dir(self) -> None:
        """Returns early when base_dir is whitespace only."""
        job = _SessionLogArchiveJob(
            base_dir="   ",
            zip_password=b"test",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="u1",
            session_id="s1",
            chat_id="c1",
            message_id="m1",
            request_id="r1",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[],
        )
        write_session_log_archive(job)

    def test_archive_creates_zip_jsonl_format(self, tmp_path: Path) -> None:
        """Creates zip archive with JSONL format."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"testpassword",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-archive",
            session_id="sess-archive",
            chat_id="chat-archive",
            message_id="msg-archive",
            request_id="req-archive",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[
                {"created": time.time(), "level": "INFO", "message": "Test log 1", "request_id": "req-archive"},
                {"created": time.time(), "level": "DEBUG", "message": "Test log 2", "request_id": "req-archive"},
            ],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-archive" / "chat-archive" / "msg-archive.zip"
        assert zip_path.exists()

        # Verify contents
        with pyzipper.AESZipFile(zip_path, "r") as zf:
            zf.setpassword(b"testpassword")
            names = zf.namelist()
            assert "meta.json" in names
            assert "logs.jsonl" in names
            assert "logs.txt" not in names

    def test_archive_creates_zip_text_format(self, tmp_path: Path) -> None:
        """Creates zip archive with text format."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"testpw",
            zip_compression="deflated",
            zip_compresslevel=6,
            user_id="user-text",
            session_id="sess-text",
            chat_id="chat-text",
            message_id="msg-text",
            request_id="req-text",
            created_at=time.time(),
            log_format="text",
            log_events=[
                {"created": time.time(), "level": "WARNING", "message": "Warning message"},
            ],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-text" / "chat-text" / "msg-text.zip"
        assert zip_path.exists()

        with pyzipper.AESZipFile(zip_path, "r") as zf:
            zf.setpassword(b"testpw")
            names = zf.namelist()
            assert "meta.json" in names
            assert "logs.txt" in names
            assert "logs.jsonl" in names

    def test_archive_creates_zip_both_format(self, tmp_path: Path) -> None:
        """Creates zip archive with both text and JSONL formats."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="bzip2",
            zip_compresslevel=9,
            user_id="user-both",
            session_id="sess-both",
            chat_id="chat-both",
            message_id="msg-both",
            request_id="req-both",
            created_at=time.time(),
            log_format="both",
            log_events=[
                {"created": time.time(), "level": "ERROR", "message": "Error msg"},
            ],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-both" / "chat-both" / "msg-both.zip"
        assert zip_path.exists()

        with pyzipper.AESZipFile(zip_path, "r") as zf:
            zf.setpassword(b"pw")
            names = zf.namelist()
            assert "meta.json" in names
            assert "logs.txt" in names
            assert "logs.jsonl" in names

    def test_archive_invalid_log_format_defaults_jsonl(self, tmp_path: Path) -> None:
        """Invalid log_format defaults to jsonl."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-invalid",
            session_id="sess-invalid",
            chat_id="chat-invalid",
            message_id="msg-invalid",
            request_id="req-invalid",
            created_at=time.time(),
            log_format="invalid-format",
            log_events=[{"message": "test"}],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-invalid" / "chat-invalid" / "msg-invalid.zip"
        assert zip_path.exists()

        with pyzipper.AESZipFile(zip_path, "r") as zf:
            zf.setpassword(b"pw")
            assert "logs.jsonl" in zf.namelist()

    def test_archive_stored_compression(self, tmp_path: Path) -> None:
        """Test stored (no compression) mode."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="stored",
            zip_compresslevel=None,
            user_id="user-stored",
            session_id="sess-stored",
            chat_id="chat-stored",
            message_id="msg-stored",
            request_id="req-stored",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[{"message": "stored test"}],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-stored" / "chat-stored" / "msg-stored.zip"
        assert zip_path.exists()

    def test_archive_handles_non_dict_events(self, tmp_path: Path) -> None:
        """Test coercion of non-dict log events."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-coerce",
            session_id="sess-coerce",
            chat_id="chat-coerce",
            message_id="msg-coerce",
            request_id="req-coerce",
            created_at=time.time(),
            log_format="both",
            log_events=[
                "string event",
                123,
                {"message": "dict event"},
            ],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-coerce" / "chat-coerce" / "msg-coerce.zip"
        assert zip_path.exists()

    def test_archive_handles_exception_in_event(self, tmp_path: Path) -> None:
        """Test events with exception block."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-exc",
            session_id="sess-exc",
            chat_id="chat-exc",
            message_id="msg-exc",
            request_id="req-exc",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[
                {
                    "created": time.time(),
                    "level": "ERROR",
                    "message": "Error with exception",
                    "exception": {"text": "Traceback..."},
                },
            ],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-exc" / "chat-exc" / "msg-exc.zip"
        assert zip_path.exists()

        with pyzipper.AESZipFile(zip_path, "r") as zf:
            zf.setpassword(b"pw")
            content = zf.read("logs.jsonl").decode("utf-8")
            data = json.loads(content.strip())
            assert "exception" in data

    def test_archive_extracts_request_ids(self, tmp_path: Path) -> None:
        """Test that unique request_ids are extracted from events."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-ids",
            session_id="sess-ids",
            chat_id="chat-ids",
            message_id="msg-ids",
            request_id="req-main",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[
                {"message": "1", "request_id": "req-a"},
                {"message": "2", "request_id": "req-b"},
                {"message": "3", "request_id": "req-a"},
            ],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-ids" / "chat-ids" / "msg-ids.zip"
        with pyzipper.AESZipFile(zip_path, "r") as zf:
            zf.setpassword(b"pw")
            meta = json.loads(zf.read("meta.json").decode("utf-8"))
            assert "request_ids" in meta
            assert sorted(meta["request_ids"]) == ["req-a", "req-b"]

    def test_archive_mkdir_fails(self, tmp_path: Path) -> None:
        """Test when mkdir fails."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user",
            session_id="sess",
            chat_id="chat",
            message_id="msg",
            request_id="req",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[],
        )

        with patch("pathlib.Path.mkdir", side_effect=PermissionError("denied")):
            # Should return without error
            write_session_log_archive(job)

    def test_archive_zip_write_fails(self, tmp_path: Path) -> None:
        """Test when zip file creation fails."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-fail",
            session_id="sess-fail",
            chat_id="chat-fail",
            message_id="msg-fail",
            request_id="req-fail",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[{"message": "test"}],
        )

        with patch("pyzipper.AESZipFile", side_effect=IOError("zip creation failed")):
            write_session_log_archive(job)

        # No zip file should exist
        zip_path = tmp_path / "user-fail" / "chat-fail" / "msg-fail.zip"
        assert not zip_path.exists()

    def test_archive_replace_fails(self, tmp_path: Path) -> None:
        """Test when os.replace fails."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-replace",
            session_id="sess-replace",
            chat_id="chat-replace",
            message_id="msg-replace",
            request_id="req-replace",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[{"message": "test"}],
        )

        with patch("os.replace", side_effect=OSError("replace failed")):
            write_session_log_archive(job)

        zip_path = tmp_path / "user-replace" / "chat-replace" / "msg-replace.zip"
        assert not zip_path.exists()

    def test_archive_invalid_created_timestamp(self, tmp_path: Path) -> None:
        """Test events with invalid created timestamp."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-ts",
            session_id="sess-ts",
            chat_id="chat-ts",
            message_id="msg-ts",
            request_id="req-ts",
            created_at=time.time(),
            log_format="both",
            log_events=[
                {"created": "not-a-number", "level": "INFO", "message": "invalid ts"},
                {"created": None, "level": "INFO", "message": "no ts"},
            ],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-ts" / "chat-ts" / "msg-ts.zip"
        assert zip_path.exists()

    def test_archive_json_encode_fails(self, tmp_path: Path) -> None:
        """Test fallback when JSON encoding of event fails."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        class Unencodable:
            def __str__(self):
                return "unencodable"

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-enc",
            session_id="sess-enc",
            chat_id="chat-enc",
            message_id="msg-enc",
            request_id="req-enc",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[
                {"created": time.time(), "level": "INFO", "message": Unencodable()},
            ],
        )

        original_dumps = json.dumps
        call_count = [0]

        def failing_dumps(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise TypeError("cannot serialize")
            return original_dumps(*args, **kwargs)

        with patch("json.dumps", side_effect=failing_dumps):
            write_session_log_archive(job)

        zip_path = tmp_path / "user-enc" / "chat-enc" / "msg-enc.zip"
        assert zip_path.exists()

    def test_archive_empty_events(self, tmp_path: Path) -> None:
        """Test with empty log events."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-empty",
            session_id="sess-empty",
            chat_id="chat-empty",
            message_id="msg-empty",
            request_id="req-empty",
            created_at=time.time(),
            log_format="both",
            log_events=[],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-empty" / "chat-empty" / "msg-empty.zip"
        assert zip_path.exists()


class TestQueueAndLoopSetters:
    """Test queue and loop setters."""

    def test_set_log_queue(self) -> None:
        original = SessionLogger.log_queue
        try:
            queue: asyncio.Queue[logging.LogRecord] = asyncio.Queue()
            SessionLogger.set_log_queue(queue)
            assert SessionLogger.log_queue is queue

            SessionLogger.set_log_queue(None)
            assert SessionLogger.log_queue is None
        finally:
            SessionLogger.log_queue = original

    @pytest.mark.asyncio
    async def test_set_main_loop(self) -> None:
        original = SessionLogger._main_loop
        try:
            loop = asyncio.get_running_loop()
            SessionLogger.set_main_loop(loop)
            assert SessionLogger._main_loop is loop

            SessionLogger.set_main_loop(None)
            assert SessionLogger._main_loop is None
        finally:
            SessionLogger._main_loop = original


class TestSessionLogArchiveJob:
    """Test the archive job dataclass."""

    def test_job_creation(self) -> None:
        job = _SessionLogArchiveJob(
            base_dir="/tmp/logs",
            zip_password=b"secret",
            zip_compression="deflated",
            zip_compresslevel=6,
            user_id="user1",
            session_id="sess1",
            chat_id="chat1",
            message_id="msg1",
            request_id="req1",
            created_at=1234567890.123,
            log_format="both",
            log_events=[{"a": 1}, {"b": 2}],
        )
        assert job.base_dir == "/tmp/logs"
        assert job.zip_password == b"secret"
        assert job.zip_compression == "deflated"
        assert job.zip_compresslevel == 6
        assert job.user_id == "user1"
        assert job.session_id == "sess1"
        assert job.chat_id == "chat1"
        assert job.message_id == "msg1"
        assert job.request_id == "req1"
        assert job.created_at == 1234567890.123
        assert job.log_format == "both"
        assert len(job.log_events) == 2


class TestConsoleFormatterException:
    """Test console formatter exception handling."""

    def test_formatter_exception_caught(self, capsys) -> None:
        """Test that exceptions in console formatting are caught."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="test message",
            args=(),
            exc_info=None,
        )
        record.request_id = None

        # Mock formatter to raise
        original_formatter = SessionLogger._console_formatter

        class FailingFormatter:
            def format(self, record):
                raise RuntimeError("format failed")

        try:
            SessionLogger._console_formatter = FailingFormatter()  # type: ignore
            # Should not raise
            SessionLogger.process_record(record)
        finally:
            SessionLogger._console_formatter = original_formatter


class TestFilterContextUpdates:
    """Test filter updates _session_last_seen."""

    def test_filter_updates_last_seen(self) -> None:
        original_last_seen = SessionLogger._session_last_seen.copy()

        try:
            SessionLogger._session_last_seen.clear()

            token_rid = SessionLogger.request_id.set("rid-last-seen-test")
            token_sid = SessionLogger.session_id.set("sid-test")
            token_uid = SessionLogger.user_id.set("uid-test")

            try:
                logger = SessionLogger.get_logger("test.last_seen")
                record = logging.LogRecord(
                    name="test",
                    level=logging.INFO,
                    pathname="",
                    lineno=0,
                    msg="test",
                    args=(),
                    exc_info=None,
                )

                # Apply filters
                for handler in logger.handlers:
                    for f in handler.filters:
                        f(record)

                # Check that last_seen was updated
                assert "rid-last-seen-test" in SessionLogger._session_last_seen
            finally:
                SessionLogger.request_id.reset(token_rid)
                SessionLogger.session_id.reset(token_sid)
                SessionLogger.user_id.reset(token_uid)
        finally:
            SessionLogger._session_last_seen.clear()
            SessionLogger._session_last_seen.update(original_last_seen)


class TestBuildEventExceptionInTraceback:
    """Test exception handling in _build_event traceback formatting."""

    def test_build_event_traceback_format_fails(self) -> None:
        """Test when traceback.format_exception fails."""
        try:
            raise ValueError("Test")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=exc_info,
        )
        record.request_id = "req-tb"
        record.session_id = None
        record.user_id = None

        with patch("traceback.format_exception", side_effect=RuntimeError("format failed")):
            event = SessionLogger._build_event(record)

        assert event.get("exception") == {"text": "<<failed to format exception>>"}


class TestArchiveEdgeCases:
    """Additional edge case tests for write_session_log_archive."""

    def test_archive_coerce_event_str_fails(self, tmp_path: Path) -> None:
        """Test _coerce_event when str() fails on non-dict."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        class BadStr:
            def __str__(self):
                raise RuntimeError("str failed")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-badstr",
            session_id="sess-badstr",
            chat_id="chat-badstr",
            message_id="msg-badstr",
            request_id="req-badstr",
            created_at=time.time(),
            log_format="both",
            log_events=[BadStr()],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-badstr" / "chat-badstr" / "msg-badstr.zip"
        assert zip_path.exists()

    def test_archive_message_str_fails_in_jsonl(self, tmp_path: Path) -> None:
        """Test when str(message) fails in _build_jsonl_record."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        class BadMessage:
            def __str__(self):
                raise RuntimeError("str failed")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-badmsg",
            session_id="sess-badmsg",
            chat_id="chat-badmsg",
            message_id="msg-badmsg",
            request_id="req-badmsg",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[
                {"created": time.time(), "level": "INFO", "message": BadMessage()},
            ],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-badmsg" / "chat-badmsg" / "msg-badmsg.zip"
        assert zip_path.exists()

    def test_archive_text_format_with_none_values(self, tmp_path: Path) -> None:
        """Test text format with various None/empty values."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-none",
            session_id="sess-none",
            chat_id="chat-none",
            message_id="msg-none",
            request_id="req-none",
            created_at=time.time(),
            log_format="text",
            log_events=[
                {"message": None, "level": None, "user_id": None, "created": None},
                {"message": "", "level": "", "user_id": "", "created": ""},
            ],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-none" / "chat-none" / "msg-none.zip"
        assert zip_path.exists()

    def test_archive_format_iso_utc_fails(self, tmp_path: Path) -> None:
        """Test when _format_iso_utc fails."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-iso",
            session_id="sess-iso",
            chat_id="chat-iso",
            message_id="msg-iso",
            request_id="req-iso",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[
                {"created": float("inf"), "level": "INFO", "message": "infinity"},
            ],
        )

        write_session_log_archive(job)

        zip_path = tmp_path / "user-iso" / "chat-iso" / "msg-iso.zip"
        assert zip_path.exists()

    def test_archive_request_ids_extraction_exception(self, tmp_path: Path) -> None:
        """Test when request_ids extraction fails."""
        try:
            import pyzipper
        except ImportError:
            pytest.skip("pyzipper not available")

        class BadIterable:
            def __iter__(self):
                raise RuntimeError("iteration failed")

        job = _SessionLogArchiveJob(
            base_dir=str(tmp_path),
            zip_password=b"pw",
            zip_compression="lzma",
            zip_compresslevel=None,
            user_id="user-reqids",
            session_id="sess-reqids",
            chat_id="chat-reqids",
            message_id="msg-reqids",
            request_id="req-reqids",
            created_at=time.time(),
            log_format="jsonl",
            log_events=[{"message": "test"}],
        )

        class FailingList(list):
            _count = 0

            def __iter__(self):
                FailingList._count += 1
                if FailingList._count == 1:
                    raise RuntimeError("iteration failed")
                return super().__iter__()

        write_session_log_archive(job)

        zip_path = tmp_path / "user-reqids" / "chat-reqids" / "msg-reqids.zip"
        assert zip_path.exists()


class TestProcessRecordOuterException:
    """Test outer exception handling in process_record."""

    def test_process_record_outer_exception_caught(self) -> None:
        """Test that outer exception in process_record is caught."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="test",
            args=(),
            exc_info=None,
        )
        record.request_id = "req-outer"
        record.session_id = None
        record.user_id = None

        original_lock = SessionLogger._state_lock
        try:
            class FailingLock:
                def __enter__(self):
                    raise RuntimeError("lock failed")

                def __exit__(self, *args):
                    pass

            SessionLogger._state_lock = FailingLock()  # type: ignore
            # Should not raise
            SessionLogger.process_record(record)
        finally:
            SessionLogger._state_lock = original_lock


class TestEmitHandler:
    """Test the emit handler in get_logger."""

    def test_emit_handler_calls_enqueue(self) -> None:
        """Test that emit handler properly calls _enqueue."""
        logger = SessionLogger.get_logger("test.emit_handler")

        handler = next(
            (h for h in logger.handlers if type(h).__name__ != "_HostForwardHandler"),
            None,
        )

        assert handler is not None

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="emit test",
            args=(),
            exc_info=None,
        )

        with patch.object(SessionLogger, "_enqueue") as mock_enqueue:
            handler.emit(record)
            mock_enqueue.assert_called_once_with(record)


def test_process_record_survives_stdout_write_failure(monkeypatch):
    """Console echo failure must not break capture: the event still lands in
    the buffer (panel condition for the silent stdout swallow)."""
    import sys as _sys

    from open_webui_openrouter_pipe.core.logging_system import SessionLogger

    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="hello capture", args=(), exc_info=None,
    )
    record.session_id = "sess-stdout-fail"
    record.request_id = "req-stdout-fail"
    record.user_id = "u1"

    def _boom(*_a, **_k):
        raise OSError("stdout closed")

    monkeypatch.setattr(_sys.stdout, "write", _boom, raising=True)
    SessionLogger.process_record(record)
    monkeypatch.undo()

    events = list(SessionLogger.logs.get("req-stdout-fail") or [])
    SessionLogger.logs.pop("req-stdout-fail", None)
    SessionLogger._session_last_seen.pop("req-stdout-fail", None)
    assert any(e.get("message") == "hello capture" for e in events)


def test_archive_renders_sentinel_for_unrenderable_message(tmp_path):
    """A message whose str() raises must degrade to a visible sentinel in the
    archive, never a silently blank field (panel condition covering the
    str-coercion guards)."""
    from open_webui_openrouter_pipe.core.logging_system import write_session_log_archive

    class _Unrenderable:
        def __str__(self):
            raise RuntimeError("nope")

    from open_webui_openrouter_pipe.core.logging_system import _SessionLogArchiveJob

    job = _SessionLogArchiveJob(
        base_dir=str(tmp_path),
        zip_password=b"pw",
        zip_compression="stored",
        zip_compresslevel=None,
        user_id="u-sent",
        session_id="s-sent",
        chat_id="c-sent",
        message_id="m-sent",
        request_id="r-sent",
        created_at=1700000000.0,
        log_format="text",
        log_events=[
            {"created": 1700000000.0, "level": "INFO", "message": _Unrenderable()},
            {"created": 1700000001.0, "level": "INFO", "message": "fine"},
        ],
    )

    write_session_log_archive(job)

    import pyzipper

    zpath = tmp_path / "u-sent" / "c-sent" / "m-sent.zip"
    assert zpath.exists()
    with pyzipper.AESZipFile(zpath) as zf:
        zf.setpassword(b"pw")
        jsonl = zf.read("logs.jsonl").decode()
    assert "<<unrenderable message>>" in jsonl
    assert "fine" in jsonl


class TestGetLoggerRootCapture:
    """Capture wired at the package root must receive sibling-module records."""

    def test_child_module_record_reaches_capture_buffers(self) -> None:
        root_name = "open_webui_openrouter_pipe"
        rid = "req-root-capture-pin"
        saved_queue = SessionLogger.log_queue
        SessionLogger.set_log_queue(None)
        token = SessionLogger.request_id.set(rid)
        try:
            SessionLogger.get_logger(root_name)
            child = logging.getLogger(root_name + ".plugins.pipe_dashboard.update_service")
            child.warning("root-capture-pin marker")
            events = SessionLogger.logs.get(rid)
            assert events, "child-module record was not captured under the active request"
            assert any("root-capture-pin marker" in str(e.get("message", "")) for e in events)
        finally:
            SessionLogger.request_id.reset(token)
            SessionLogger.set_log_queue(saved_queue)
            SessionLogger.logs.pop(rid, None)
            wired = logging.getLogger(root_name)
            wired.handlers.clear()
            wired.filters.clear()


def test_text_rendering_keeps_the_error_detail():
    """logger.exception() moves the detail out of the message; text output must keep it.

    Both text renderers emit only ``message``. Converting
    ``logger.error(f"failed: {exc}")`` to ``logger.exception("failed")`` therefore
    deletes the cause from the human-readable logs.txt archive and from the error-log
    citation dump, which are exactly the artifacts a user sends when reporting a bug.
    """
    from open_webui_openrouter_pipe.core.logging_system import SessionLogger

    event = {
        "created": 1_700_000_000.0,
        "level": "ERROR",
        "message": "Failed to emit completion",
        "exception": {"text": "Traceback...\nRuntimeError: upstream socket died"},
    }

    rendered = SessionLogger.format_event_as_text(event)

    assert "Failed to emit completion" in rendered
    assert "upstream socket died" in rendered, (
        "the traceback was captured but not rendered; logs.txt shows the headline "
        "and silently drops the cause"
    )


def test_text_rendering_without_an_exception_is_unchanged():
    from open_webui_openrouter_pipe.core.logging_system import SessionLogger

    rendered = SessionLogger.format_event_as_text(
        {"created": 1_700_000_000.0, "level": "INFO", "message": "hello"}
    )

    assert rendered.endswith("hello"), f"unexpected trailing content: {rendered!r}"


def test_archive_text_rendering_keeps_the_error_detail(tmp_path):
    """Round-trips a real encrypted archive; logs.txt must carry the traceback.

    write_session_log_archive builds its own _format_event_as_text as a closure, so
    the sibling test (which covers SessionLogger.format_event_as_text) does not reach
    it. An earlier version of this asserted that the source contained the call, which
    any edit keeping the call but neutralising its value would pass.
    """
    pyzipper = pytest.importorskip("pyzipper")

    from open_webui_openrouter_pipe.core.logging_system import (
        _SessionLogArchiveJob,
        write_session_log_archive,
    )

    event = {
        "created": 1_700_000_000.0,
        "level": "ERROR",
        "message": "Failed to emit completion",
        "exception": {"text": "Traceback (most recent call last):\nRuntimeError: upstream socket died"},
    }
    job = _SessionLogArchiveJob(
        base_dir=str(tmp_path),
        zip_password=b"pw",
        zip_compression="stored",
        zip_compresslevel=None,
        user_id="u1",
        session_id="s1",
        chat_id="c1",
        message_id="m1",
        request_id="r1",
        created_at=1_700_000_000.0,
        log_format="both",
        log_events=[dict(event)],
    )

    write_session_log_archive(job)

    archive = tmp_path / "u1" / "c1" / "m1.zip"
    assert archive.exists(), "archive was not written; the round-trip proves nothing"
    with pyzipper.AESZipFile(archive) as zf:
        zf.setpassword(b"pw")
        logs_txt = zf.read("logs.txt").decode()

    assert "Failed to emit completion" in logs_txt
    assert "upstream socket died" in logs_txt, (
        "logs.txt -- the human-readable log a user attaches to a bug report -- shows "
        "the headline and silently drops the cause"
    )


def test_the_host_forwarder_honours_the_session_log_level():
    """LOG_LEVEL must actually gate what reaches Open WebUI's handlers.

    The package logger sits at DEBUG so the session archive captures everything, and
    Open WebUI's root StreamHandler has no level of its own -- so plain propagation
    printed every package DEBUG record to the container log and neither GLOBAL_LOG_LEVEL
    nor this pipe's LOG_LEVEL could stop it. The autouse fixture raises the threshold
    to DEBUG for the rest of the suite, so this is the only place the production
    default is exercised.
    """
    import logging as _logging

    from open_webui_openrouter_pipe.core.logging_system import SessionLogger

    import sys as _sys

    _pkg = _sys.modules.get("open_webui_openrouter_pipe")
    _root = (getattr(_pkg, "__name__", "") or "open_webui_openrouter_pipe").split(".")[0]
    wired = SessionLogger.get_logger(_root)
    child = _logging.getLogger(f"{_root}.forwarder_probe")

    seen: list[tuple[str, str]] = []

    class _Host(_logging.Handler):
        def emit(self, record):
            seen.append((record.levelname, record.getMessage()))

    host = _Host()
    root = _logging.getLogger()
    root.addHandler(host)
    try:
        token = SessionLogger.log_level.set(_logging.INFO)
        try:
            child.debug("debug-at-info-threshold")
            child.warning("warning-at-info-threshold")
        finally:
            SessionLogger.log_level.reset(token)

        levels = [lvl for lvl, _ in seen]
        assert "DEBUG" not in levels, (
            f"a DEBUG record reached the host at an INFO threshold: {seen}. Every "
            "logger.debug in the package would print to the container log with no "
            "way for an operator to suppress it."
        )
        assert "WARNING" in levels, (
            f"a WARNING did not reach the host at all: {seen}. Suppressing debug "
            "output must not also silence the records operators need."
        )

        seen.clear()
        token = SessionLogger.log_level.set(_logging.DEBUG)
        try:
            child.debug("debug-at-debug-threshold")
        finally:
            SessionLogger.log_level.reset(token)
        assert [lvl for lvl, _ in seen] == ["DEBUG"], (
            f"raising the threshold to DEBUG did not let a DEBUG record through: {seen}"
        )
    finally:
        root.removeHandler(host)
        wired.handlers.clear()


def test_a_test_cannot_leave_the_package_logger_deaf():
    """The autouse restore fixture must actually undo a hostile teardown.

    Two tests in this file end by clearing the package logger's handlers. With
    propagate=False that leaves it with nowhere to send anything, and every later
    open_webui_openrouter_pipe.* record is dropped -- 33 tests ran that way before
    the fixture existed. An absence-assertion landing in that window would pass for
    the wrong reason, which is the failure mode with no symptom.
    """
    import logging as _logging

    import sys as _sys

    pkg = _sys.modules.get("open_webui_openrouter_pipe")
    root_name = (getattr(pkg, "__name__", "") or "open_webui_openrouter_pipe").split(".")[0]
    logger = _logging.getLogger(root_name)
    import logging as _lg

    real = [h for h in logger.handlers if not isinstance(h, _lg.NullHandler)]
    assert real or logger.propagate, (
        f"the package logger {logger.name!r} has no handler that can emit "
        f"({[type(h).__name__ for h in logger.handlers]}) and does not propagate, so "
        "every record under it is silently discarded. A previous test cleared it and "
        "the restore fixture did not put it back. A NullHandler does not count -- it "
        "is what makes this failure silent."
    )


def test_out_of_request_records_are_judged_against_the_configured_level():
    """No request in scope must not mean a hardcoded threshold.

    Everything pipes() logs -- filter auto-install failures, plugin dispatch failures,
    startup pruning -- runs with the log-level ContextVar unset. Reading that var
    directly pinned those records to its default, so the diagnostics added expressly
    to stop swallowing failures became unreachable at any setting.
    """
    import logging as _logging

    from open_webui_openrouter_pipe.core.logging_system import SessionLogger

    saved = SessionLogger.process_log_level
    token = SessionLogger.log_level.set(None)
    try:
        SessionLogger.process_log_level = _logging.DEBUG
        assert SessionLogger.effective_log_level() == _logging.DEBUG, (
            "with no request in scope the operator's configured level is ignored"
        )
        SessionLogger.process_log_level = _logging.WARNING
        assert SessionLogger.effective_log_level() == _logging.WARNING

        inner = SessionLogger.log_level.set(_logging.DEBUG)
        try:
            assert SessionLogger.effective_log_level() == _logging.DEBUG, (
                "the per-request level no longer overrides the process floor"
            )
        finally:
            SessionLogger.log_level.reset(inner)
    finally:
        SessionLogger.log_level.reset(token)
        SessionLogger.process_log_level = saved


def test_host_handlers_receive_enriched_records():
    """The forwarder must run after the filter that stamps the record.

    session_id / request_id / user_id are attached by a filter on the capture handler,
    which mutates the record in place. callHandlers runs handlers in insertion order,
    so registering the forwarder first handed host formatters bare records -- a
    regression against the old propagate=True shape, where the logger's own handlers
    always ran before ancestors.
    """
    import logging as _logging
    import sys as _sys

    from open_webui_openrouter_pipe.core.logging_system import SessionLogger

    seen: list[tuple[str, object, object]] = []

    class _Host(_logging.Handler):
        def emit(self, record):
            seen.append((
                record.getMessage(),
                getattr(record, "user_id", "<absent>"),
                getattr(record, "request_id", "<absent>"),
            ))

    host = _Host()
    root = _logging.getLogger()
    root.addHandler(host)
    uid = SessionLogger.user_id.set("u-enrich")
    rid = SessionLogger.request_id.set("r-enrich")
    try:
        pkg = _sys.modules.get("open_webui_openrouter_pipe")
        root_name = (getattr(pkg, "__name__", "") or "open_webui_openrouter_pipe").split(".")[0]
        SessionLogger.get_logger(root_name)
        _logging.getLogger(f"{root_name}.enrich_probe").warning("enrich-probe-marker")
    finally:
        SessionLogger.user_id.reset(uid)
        SessionLogger.request_id.reset(rid)
        root.removeHandler(host)

    hits = [h for h in seen if h[0] == "enrich-probe-marker"]
    assert hits, "the record never reached a host handler at all"
    assert hits[0][1] == "u-enrich" and hits[0][2] == "r-enrich", (
        f"the host handler saw {hits[0]!r}: the forwarder ran before the enriching "
        "filter, so any host formatter referencing %(user_id)s or %(request_id)s "
        "raises instead of formatting."
    )


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("DEBUG", 10), ("warning", 30), ("BASIC_FORMAT", 20), ("nonsense", 20), ("", 20), (None, 20)],
)
def test_a_misconfigured_global_log_level_cannot_break_logging(configured, expected):
    """`getattr(logging, name, INFO)` returns any module attribute, not just a level.

    GLOBAL_LOG_LEVEL=BASIC_FORMAT resolved to a format string, and the forwarder's
    int() coercion sat outside its own try -- so an unrelated logger.warning() raised
    ValueError into request handling, breaking the module's own thrice-stated rule
    that logging must never do that.
    """
    from open_webui_openrouter_pipe.core.logging_system import resolve_level

    assert resolve_level(configured, 20) == expected


def test_a_bad_threshold_never_escapes_the_forward_handler():
    """Even if the floor is corrupted at runtime, emitting must not raise."""
    import logging as _logging
    import sys as _sys

    from open_webui_openrouter_pipe.core.logging_system import SessionLogger

    saved = SessionLogger.process_log_level
    pkg = _sys.modules.get("open_webui_openrouter_pipe")
    root_name = (getattr(pkg, "__name__", "") or "open_webui_openrouter_pipe").split(".")[0]
    token = SessionLogger.log_level.set(None)
    try:
        SessionLogger.process_log_level = "%(levelname)s"  # type: ignore[assignment]
        SessionLogger.get_logger(root_name)
        _logging.getLogger(f"{root_name}.bad_threshold_probe").warning("must not raise")
    finally:
        SessionLogger.log_level.reset(token)
        SessionLogger.process_log_level = saved


class TestLevelResolution:
    """Every level name in the pipe goes through one resolver, and it is total.

    `getattr(logging, name, fallback)` returns any attribute of the module, so a name
    that happens to match one yields a non-level -- GLOBAL_LOG_LEVEL=BASIC_FORMAT gives
    a format string where an int belongs. Bare `getattr(logging, name)` raises instead,
    and the site that used it runs on every request.
    """

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("DEBUG", logging.DEBUG),
            ("info", logging.INFO),
            ("  WARNING  ", logging.WARNING),
            ("ERROR", logging.ERROR),
            ("CRITICAL", logging.CRITICAL),
        ],
    )
    def test_real_level_names_resolve(self, name, expected):
        from open_webui_openrouter_pipe.core.logging_system import resolve_level

        assert resolve_level(name, logging.INFO) == expected

    @pytest.mark.parametrize(
        "name",
        [
            "BASIC_FORMAT",
            "Formatter",
            "NOTSET",
            "TRACE",
            "",
            None,
        ],
    )
    def test_anything_that_is_not_a_usable_threshold_falls_back(self, name):
        from open_webui_openrouter_pipe.core.logging_system import resolve_level

        assert resolve_level(name, logging.WARNING) == logging.WARNING

    def test_a_bad_valve_does_not_break_request_handling(self, pipe_instance):
        """The failure this replaced: an AttributeError raised per request.

        `_apply_logging_context` runs for every request, so a LOG_LEVEL that is not an
        attribute of `logging` took the whole request down rather than falling back.
        """
        from open_webui_openrouter_pipe.core.logging_system import SessionLogger, resolve_level

        assert resolve_level("NOT_A_LEVEL", SessionLogger.process_log_level) == (
            SessionLogger.process_log_level
        )

    def test_a_request_does_not_redefine_the_process_wide_floor(self, pipe_instance):
        """`process_log_level` has one writer, and a request is not it.

        Every other value `_apply_logging_context` sets is a ContextVar captured in a
        token and reset when the request ends. `process_log_level` is a plain class
        attribute on a process-wide object, so a request that wrote it would set the
        floor for every out-of-request logger in the worker from then on -- the
        dashboard publisher, the Redis listener, video lifecycle tasks, and `pipes()`
        until its next call.

        The per-request level rides the `log_level` ContextVar, which
        `effective_log_level` prefers while a request is in scope. So this asserts both
        halves: the request's level is in force during the request, and the process
        floor is untouched by it.
        """
        import asyncio
        import logging as _logging

        from open_webui_openrouter_pipe import _PipeJob
        from open_webui_openrouter_pipe.core.logging_system import SessionLogger

        pipe = pipe_instance
        pipe.valves.LOG_LEVEL = "WARNING"
        pipe._refresh_process_log_level()
        floor_before = SessionLogger.process_log_level
        assert floor_before == _logging.WARNING, (
            "the sole writer did not install the operator's LOG_LEVEL, so nothing below "
            "is testing what it claims to"
        )

        # pins the no-request baseline: reset() restores what the var held before the
        # request, and a sibling test in this process may have left one set
        outer = SessionLogger.log_level.set(None)
        loop = asyncio.new_event_loop()
        try:
            job = _PipeJob(
                pipe=pipe,
                body={"model": "test", "messages": []},
                user={"id": "u1"},
                request=None,
                event_emitter=None,
                event_call=None,
                metadata={"session_id": "s1"},
                tools=None,
                task=None,
                task_body=None,
                valves=pipe.valves.model_copy(update={"LOG_LEVEL": "DEBUG"}),
                future=loop.create_future(),
            )
            tokens = pipe._apply_logging_context(job)
            try:
                assert SessionLogger.effective_log_level() == _logging.DEBUG, (
                    "the request's own level is not in force during the request"
                )
                assert SessionLogger.process_log_level == floor_before, (
                    f"the request moved the process-wide floor to "
                    f"{SessionLogger.process_log_level}. It has no token, so nothing "
                    "puts it back: every out-of-request logger in this worker is now "
                    "judged against one request's valve."
                )
            finally:
                for var, token in reversed(tokens):
                    var.reset(token)

            assert SessionLogger.effective_log_level() == floor_before, (
                "with the request over and no level in scope, the operator's configured "
                "floor is not what governs"
            )
        finally:
            SessionLogger.log_level.reset(outer)
            loop.close()

    def test_constructing_the_pipe_points_the_floor_at_the_operators_valve(self):
        """Covers the CALL SITE, which the test above does not.

        That one invokes `_refresh_process_log_level()` by hand and asserts on the
        result, so the function's logic is covered while both places production calls
        it are not: deleting either `self._refresh_process_log_level()` from
        `Pipe.__init__` or from `pipes()` left the whole suite green.

        With them gone the floor stays at whatever GLOBAL_LOG_LEVEL set at import and
        the LOG_LEVEL valve never reaches it. Everything that logs outside a request --
        filter auto-install failures, plugin dispatch failures, startup pruning, the
        Redis listener, the dashboard publisher, video lifecycle tasks -- is then judged
        against that stale floor at every valve setting.

        The floor is poisoned first so a pass cannot come from agreeing with the
        default, and no spy is used: a spy proves the call happened, not that the level
        arrived.
        """
        import logging as _logging

        from open_webui_openrouter_pipe import Pipe
        from open_webui_openrouter_pipe.core.logging_system import (
            SessionLogger,
            resolve_level,
        )

        saved = SessionLogger.process_log_level
        try:
            SessionLogger.process_log_level = _logging.CRITICAL
            pipe = Pipe()
            pipe.valves.LOG_LEVEL = "DEBUG"
            pipe._refresh_process_log_level()
            assert SessionLogger.process_log_level == _logging.DEBUG

            SessionLogger.process_log_level = _logging.CRITICAL
            fresh = Pipe()
            assert SessionLogger.process_log_level != _logging.CRITICAL, (
                "constructing a Pipe left the out-of-request floor at the poisoned "
                "value, so Pipe.__init__ never applied the LOG_LEVEL valve and nothing "
                "logged outside a request can be seen at any setting"
            )
            assert SessionLogger.process_log_level == resolve_level(
                str(fresh.valves.LOG_LEVEL), _logging.INFO
            ), (
                f"the floor is {SessionLogger.process_log_level} but the pipe's "
                f"LOG_LEVEL valve says {fresh.valves.LOG_LEVEL!r}"
            )
        finally:
            SessionLogger.process_log_level = saved

    @pytest.mark.asyncio
    async def test_listing_models_reapplies_the_operators_valve(self):
        """The second call site. `pipes()` runs on every model list, which is where an
        operator's LOG_LEVEL change first takes effect for out-of-request logging --
        the pipe object is long-lived, so `__init__` alone never sees the new value.
        """
        import logging as _logging

        from aioresponses import aioresponses

        from open_webui_openrouter_pipe import Pipe
        from open_webui_openrouter_pipe.core.config import EncryptedStr
        from open_webui_openrouter_pipe.core.logging_system import SessionLogger

        saved = SessionLogger.process_log_level
        pipe = Pipe()
        try:
            pipe.valves.API_KEY = EncryptedStr("sk-test-key")
            pipe.valves.LOG_LEVEL = "WARNING"
            SessionLogger.process_log_level = _logging.CRITICAL

            with aioresponses() as mock_http:
                mock_http.get(
                    "https://openrouter.ai/api/v1/models",
                    payload={"data": []},
                    repeat=True,
                )
                await pipe.pipes()

            assert SessionLogger.process_log_level == _logging.WARNING, (
                f"after pipes() the out-of-request floor is "
                f"{SessionLogger.process_log_level}, not the configured WARNING. An "
                "operator changing LOG_LEVEL on a running pipe never affects anything "
                "logged outside a request."
            )
        finally:
            SessionLogger.process_log_level = saved
            await pipe.close()

    def test_the_host_forwarder_reproduces_callHandlers_over_a_real_ancestor_chain(self):
        """The two halves that make the walk faithful, neither of which was asserted.

        `_HostForwardHandler` reimplements `logging.Logger.callHandlers` with the
        pipe's own threshold applied first. Its session-threshold half was covered; the
        per-handler level check and the `propagate=False` stop were not -- replacing
        `if record.levelno >= handler.level` with `if True`, or deleting the
        `if not node.propagate: break`, left the whole suite green.

        The first sends every INFO record to a host handler an operator configured at
        WARNING. The second pushes records across a boundary the host deliberately cut,
        into root handlers meant never to see them.

        A real three-deep chain is what makes both reachable: the package logger's
        parent is the root logger, which has no parent and never consults propagate, so
        against the live tree that branch cannot be exercised at all.
        """
        import logging as _logging

        from open_webui_openrouter_pipe.core.logging_system import SessionLogger

        class _Recorder(_logging.Handler):
            def __init__(self, level=_logging.NOTSET):
                super().__init__(level=level)
                self.seen: list[str] = []

            def emit(self, record):
                self.seen.append(record.getMessage())

        root = _logging.getLogger("hostfwd_probe")
        mid = _logging.getLogger("hostfwd_probe.mid")
        leaf = _logging.getLogger("hostfwd_probe.mid.leaf")
        root_rec, mid_rec = _Recorder(), _Recorder(level=_logging.ERROR)
        root.addHandler(root_rec)
        mid.addHandler(mid_rec)
        forwarder = SessionLogger._HostForwardHandler(leaf)

        saved = SessionLogger.process_log_level
        token = SessionLogger.log_level.set(None)
        try:
            SessionLogger.process_log_level = _logging.DEBUG
            record = leaf.makeRecord(
                leaf.name, _logging.WARNING, __file__, 0, "probe-one", None, None
            )
            forwarder.emit(record)

            assert mid_rec.seen == [], (
                "a host handler set to ERROR received a WARNING. The per-handler level "
                "check is not being applied, so operator handler thresholds are ignored."
            )
            assert root_rec.seen == ["probe-one"], (
                f"the record did not reach the ancestor's handler: {root_rec.seen}"
            )

            mid.propagate = False
            record2 = leaf.makeRecord(
                leaf.name, _logging.WARNING, __file__, 0, "probe-two", None, None
            )
            forwarder.emit(record2)
            assert root_rec.seen == ["probe-one"], (
                f"the walk crossed a propagate=False boundary: {root_rec.seen}. The "
                "host cut that link deliberately and these records were not meant to "
                "reach the root handlers."
            )
        finally:
            SessionLogger.log_level.reset(token)
            SessionLogger.process_log_level = saved
            mid.propagate = True
            root.removeHandler(root_rec)
            mid.removeHandler(mid_rec)


class TestSessionBufferResize:
    """Changing SESSION_LOG_MAX_LINES mid-request must not empty the buffer."""

    @staticmethod
    def _emit(request_id: str, message: str) -> None:
        import logging as _logging

        from open_webui_openrouter_pipe.core.logging_system import SessionLogger

        record = _logging.LogRecord(
            "probe", _logging.WARNING, __file__, 0, message, None, None
        )
        record.request_id = request_id
        record.session_id = "s"
        record.user_id = "u"
        SessionLogger.process_record(record)

    def _messages(self, request_id: str) -> list[str]:
        from open_webui_openrouter_pipe.core.logging_system import SessionLogger

        return [e.get("message", "") for e in SessionLogger.logs.get(request_id, [])]

    def test_growing_the_cap_keeps_every_line_already_captured(self):
        """The buffer was rebuilt empty on any change, so the archive silently began
        mid-request. Nothing warned, and no gap marker was written."""
        from open_webui_openrouter_pipe.core.logging_system import SessionLogger

        saved = SessionLogger.SESSION_LOG_MAX_LINES
        SessionLogger.logs.pop("resize-grow", None)
        try:
            SessionLogger.SESSION_LOG_MAX_LINES = 10
            self._emit("resize-grow", "before-one")
            self._emit("resize-grow", "before-two")
            assert self._messages("resize-grow") == ["before-one", "before-two"]

            SessionLogger.SESSION_LOG_MAX_LINES = 20
            self._emit("resize-grow", "after")

            assert self._messages("resize-grow") == ["before-one", "before-two", "after"], (
                "raising the cap discarded lines already captured. The archive the user "
                "downloads starts partway through the request with nothing saying so."
            )
        finally:
            SessionLogger.SESSION_LOG_MAX_LINES = saved
            SessionLogger.logs.pop("resize-grow", None)

    def test_shrinking_the_cap_drops_the_oldest_lines_not_all_of_them(self):
        """Shrinking must mean what the valve says: keep the most recent N."""
        from open_webui_openrouter_pipe.core.logging_system import SessionLogger

        saved = SessionLogger.SESSION_LOG_MAX_LINES
        SessionLogger.logs.pop("resize-shrink", None)
        try:
            SessionLogger.SESSION_LOG_MAX_LINES = 10
            for i in range(4):
                self._emit("resize-shrink", f"line-{i}")

            SessionLogger.SESSION_LOG_MAX_LINES = 2
            self._emit("resize-shrink", "newest")

            assert self._messages("resize-shrink") == ["line-3", "newest"], (
                f"got {self._messages('resize-shrink')}; shrinking the cap must keep the "
                "most recent lines up to the new limit, not empty the buffer"
            )
        finally:
            SessionLogger.SESSION_LOG_MAX_LINES = saved
            SessionLogger.logs.pop("resize-shrink", None)

    def test_a_change_does_not_wipe_every_other_session_in_flight(self):
        """The cap is process-wide and the buffers are per-request, so one admin edit
        reached every request in flight at that instant -- not just one."""
        from open_webui_openrouter_pipe.core.logging_system import SessionLogger

        saved = SessionLogger.SESSION_LOG_MAX_LINES
        ids = ["sess-a", "sess-b", "sess-c"]
        for rid in ids:
            SessionLogger.logs.pop(rid, None)
        try:
            SessionLogger.SESSION_LOG_MAX_LINES = 10
            for rid in ids:
                self._emit(rid, f"{rid}-early")

            SessionLogger.SESSION_LOG_MAX_LINES = 11
            for rid in ids:
                self._emit(rid, f"{rid}-late")

            for rid in ids:
                assert self._messages(rid) == [f"{rid}-early", f"{rid}-late"], (
                    f"{rid} lost its earlier lines: {self._messages(rid)}. One valve "
                    "change reached every concurrent session, not only the one whose "
                    "request carried the new value."
                )
        finally:
            SessionLogger.SESSION_LOG_MAX_LINES = saved
            for rid in ids:
                SessionLogger.logs.pop(rid, None)


class TestGlobalLogLevelHasOneResolver:
    """The valve default and the process floor must agree for every accepted spelling."""

    @pytest.mark.parametrize(
        ("env", "expected_name", "expected_level"),
        [
            ("WARN", "WARNING", 30),
            ("FATAL", "CRITICAL", 50),
            ("WARNING", "WARNING", 30),
            ("debug", "DEBUG", 10),
            ("NOTSET", "INFO", 20),
            ("BOGUS", "INFO", 20),
            ("", "INFO", 20),
        ],
    )
    def test_the_valve_default_and_resolve_level_denote_the_same_threshold(
        self, env, expected_name, expected_level, monkeypatch
    ):
        """WARN and FATAL are the cases that mattered.

        Open WebUI gates GLOBAL_LOG_LEVEL on `logging.getLevelNamesMapping()`, which
        contains both. The valve default used a membership test against the five
        canonical names, so it mapped them to INFO while `resolve_level` mapped them to
        WARNING and CRITICAL. An operator who set WARN got a WARNING floor at import and
        an INFO floor once the pipe applied its own valve -- in one process, for one
        configuration.

        Parametrised over the canonical names too, deliberately: they always agreed,
        which is exactly why the divergence survived. Only the alias spellings and the
        rejected ones can fail here.
        """
        import logging as _logging

        from open_webui_openrouter_pipe.core.config import _resolve_log_level_default
        from open_webui_openrouter_pipe.core.logging_system import resolve_level

        monkeypatch.setenv("GLOBAL_LOG_LEVEL", env)
        valve_name = _resolve_log_level_default()
        assert valve_name == expected_name, (
            f"GLOBAL_LOG_LEVEL={env!r} gives the valve {valve_name!r}, expected "
            f"{expected_name!r}"
        )
        floor = resolve_level(env, _logging.INFO)
        assert floor == expected_level, (
            f"GLOBAL_LOG_LEVEL={env!r} gives the process floor {floor}, expected "
            f"{expected_level}"
        )
        assert resolve_level(valve_name, _logging.INFO) == floor, (
            f"GLOBAL_LOG_LEVEL={env!r}: the valve says {valve_name!r} "
            f"({resolve_level(valve_name, _logging.INFO)}) and the floor says {floor}. "
            "One process, one configuration, two thresholds."
        )

    def test_the_valve_default_stays_inside_its_literal(self, monkeypatch):
        """A level registered by a third party must not leak into the UI dropdown."""
        import logging as _logging

        from open_webui_openrouter_pipe.core.config import (
            _ALLOWED_LOG_LEVELS,
            _resolve_log_level_default,
        )

        _logging.addLevelName(25, "NOTICE")
        try:
            monkeypatch.setenv("GLOBAL_LOG_LEVEL", "NOTICE")
            value = _resolve_log_level_default()
            assert value in _ALLOWED_LOG_LEVELS, (
                f"{value!r} is not one of the five the valve's Literal allows, so the "
                "config tab would render a value it cannot round-trip"
            )
        finally:
            _logging.addLevelName(25, "Level 25")


def test_both_session_log_text_renderers_produce_the_same_line(tmp_path):
    """The two renderers must agree, driven rather than counted.

    `SessionLogger.format_event_as_text` and the `_format_event_as_text` closure inside
    `write_session_log_archive` render the same line for the same event. A third copy
    once existed, had already drifted -- both live renderers gained the exception suffix
    and the dead one did not -- and the next person to need a formatter would have found
    the one that silently drops tracebacks.

    Counting `"[user="` in the source was the previous shape of this test and failed in
    both directions: consolidating the two onto one function, which is the fix the
    duplication calls for, made the count 1 and turned it red, while a third copy
    spelling the line as `"[user" + "="` left the count at 2. Driving both paths and
    comparing the output fails when they drift, passes through the consolidation, and
    cannot be satisfied by how the format string happens to be written.
    """
    pyzipper = pytest.importorskip("pyzipper")

    from open_webui_openrouter_pipe.core.logging_system import (
        SessionLogger,
        _SessionLogArchiveJob,
        write_session_log_archive,
    )

    event = {
        "created": 1_700_000_000.5,
        "level": "ERROR",
        "user_id": "u1",
        "message": "Failed to emit completion",
        "exception": {
            "text": "Traceback (most recent call last):\nRuntimeError: upstream socket died"
        },
    }

    direct = SessionLogger.format_event_as_text(dict(event))

    job = _SessionLogArchiveJob(
        base_dir=str(tmp_path),
        zip_password=b"pw",
        zip_compression="stored",
        zip_compresslevel=None,
        user_id="u1",
        session_id="s1",
        chat_id="c1",
        message_id="m1",
        request_id="r1",
        created_at=1_700_000_000.5,
        log_format="both",
        log_events=[dict(event)],
    )
    write_session_log_archive(job)

    archive = tmp_path / "u1" / "c1" / "m1.zip"
    assert archive.exists(), "no archive was written; the comparison would prove nothing"
    with pyzipper.AESZipFile(archive) as zf:
        zf.setpassword(b"pw")
        archived = zf.read("logs.txt").decode()

    assert direct.strip(), "the direct renderer produced nothing to compare"
    assert archived.strip() == direct.strip(), (
        "the two session-log text renderers disagree, so the archive a user downloads "
        "does not match what the live log showed.\n"
        f"direct  : {direct.strip()!r}\n"
        f"archived: {archived.strip()!r}"
    )


_CONSOLE_CASE_SCRIPT = """
import io, json, logging, os, sys
sys.path.insert(0, __ROOT__)
sys.path.insert(0, os.path.join(__ROOT__, "tests"))
import owui_stubs  # noqa: F401
from open_webui_openrouter_pipe.core.logging_system import SessionLogger

def run_case(host_handler, message, emit_level, threshold):
    buf = io.StringIO()
    sys.stdout = buf
    if host_handler == "plain":
        logging.basicConfig(level="INFO", stream=buf, force=True)
    elif host_handler == "json":
        class _JF(logging.Formatter):
            def format(self, r):
                return json.dumps({"level": r.levelname, "msg": r.getMessage()})
        h = logging.StreamHandler(buf)
        h.setFormatter(_JF())
        logging.basicConfig(level="INFO", handlers=[h], force=True)
    else:
        for h in list(logging.root.handlers):
            logging.root.removeHandler(h)
    SessionLogger.get_logger("open_webui_openrouter_pipe")
    SessionLogger.process_log_level = getattr(logging, threshold)
    _tok = SessionLogger.log_level.set(getattr(logging, threshold))
    getattr(logging.getLogger("open_webui_openrouter_pipe.core.config"), emit_level)(message)
    SessionLogger.log_level.reset(_tok)
    sys.stdout = sys.__stdout__
    return buf.getvalue()

results = {}
for key, spec in __CASES__:
    sys.stdout.flush()
    sys.stderr.flush()
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        try:
            payload = {"ok": True, "out": run_case(**spec)}
        except BaseException as exc:
            payload = {"ok": False, "out": repr(exc)}
        try:
            with os.fdopen(write_fd, "w") as handle:
                handle.write(json.dumps(payload))
        finally:
            os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd) as handle:
        raw = handle.read()
    _, status = os.waitpid(pid, 0)
    if not raw:
        raw = json.dumps({"ok": False, "out": "child produced nothing, status=%d" % status})
    results[key] = json.loads(raw)
print("CONSOLE_JSON:" + json.dumps(results))
"""


_CONSOLE_CASES = [
    ("plain|PROBE-MESSAGE|warning|DEBUG", None),
    ("none|PROBE-MESSAGE|warning|DEBUG", None),
    ("json|PROBE-MESSAGE|warning|DEBUG", None),
    ("plain|PROBE-MESSAGE|debug|WARNING", None),
    ("plain|PROBE-MESSAGE|info|WARNING", None),
    ("plain|PROBE-MESSAGE|warning|WARNING", None),
    ("none|PROBE-MESSAGE|debug|WARNING", None),
    ("none|PROBE-MESSAGE|info|WARNING", None),
    ("none|PROBE-MESSAGE|warning|WARNING", None),
]

_CONSOLE_RESULTS: dict[str, dict] = {}


def _run_console_script(cases: list) -> dict:
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = (
        _CONSOLE_CASE_SCRIPT.replace("__ROOT__", repr(str(root)))
        .replace("__CASES__", json.dumps(cases))
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        env={"PATH": "/usr/bin:/bin"},
    )
    line = next(
        (ln for ln in result.stdout.splitlines() if ln.startswith("CONSOLE_JSON:")), None
    )
    assert line, (
        f"console probe produced no output (rc={result.returncode}) "
        f"stderr={result.stderr[-400:]!r}"
    )
    return json.loads(line[len("CONSOLE_JSON:") :])


def _console_probe(
    *,
    host_handler: str,
    message: str = "PROBE-MESSAGE",
    emit_level: str = "warning",
    threshold: str = "DEBUG",
) -> str:
    """Run the console wiring in a FRESH process and return everything written to stdout.

    A fresh process because `get_logger` mutates the root logger and installs handlers
    that outlive the test, and because a StreamHandler binds its stream at construction --
    swapping `sys.stdout` afterwards measures nothing, which is how the first version of
    this probe reported zero lines from working code.

    Every case pays the same import (`open_webui.env` alone costs ~1s), so one interpreter
    imports once and `os.fork()`s per case: the child starts from the pristine
    post-import state and its handler mutations die with it, which is the same isolation
    a separate interpreter gives at a ninth of the wall clock. A case whose arguments are
    not in the precomputed batch still gets its own run rather than a wrong cached answer.
    """
    key = f"{host_handler}|{message}|{emit_level}|{threshold}"
    spec = {
        "host_handler": host_handler,
        "message": message,
        "emit_level": emit_level,
        "threshold": threshold,
    }
    if key not in _CONSOLE_RESULTS:
        batch = [(k, dict(zip(("host_handler", "message", "emit_level", "threshold"), k.split("|"))))
                 for k, _ in _CONSOLE_CASES]
        if key not in {k for k, _ in _CONSOLE_CASES}:
            batch = [(key, spec)]
        _CONSOLE_RESULTS.update(_run_console_script(batch))
    outcome = _CONSOLE_RESULTS[key]
    assert outcome["ok"], f"console probe case {key!r} raised: {outcome['out']}"
    return outcome["out"]


@pytest.mark.timeout(600)
@pytest.mark.parametrize("host_handler", ["plain", "none"], ids=["host-prints", "host-silent"])
def test_a_record_reaches_the_console_exactly_once(host_handler):
    """One line per record, whether or not the host configured a sink.

    The pipe used to write its own formatted line to stdout AND forward the record to the
    host's handlers, so every record appeared twice in two different formats wherever
    Open WebUI had installed a root handler -- which it does whenever GLOBAL_LOG_LEVEL
    names a real level.

    Both arms matter and they fail in opposite directions: deleting the private write
    alone makes the `host-silent` arm print nothing, and deleting the forwarder alone
    makes the pipe invisible to an operator's configured sinks. Only one of the two can
    own the console, and which one depends on whether an ancestor handler exists.
    """
    output = _console_probe(host_handler=host_handler)
    assert output.count("PROBE-MESSAGE") == 1, (
        f"with host_handler={host_handler!r} the record appeared "
        f"{output.count('PROBE-MESSAGE')} times:\n{output}"
    )


@pytest.mark.timeout(600)
def test_the_console_does_not_corrupt_a_json_log_stream():
    """LOG_FORMAT=json is a supported Open WebUI setting, and it must stay parseable.

    The private stdout write emitted the pipe's own text format regardless of how the
    operator had configured logging, so every pipe record put a non-JSON line into a JSON
    stream and broke structured log parsing for the whole deployment.
    """
    output = _console_probe(host_handler="json")
    lines = [ln for ln in output.splitlines() if "PROBE-MESSAGE" in ln]
    assert len(lines) == 1, f"expected one line, got {len(lines)}:\n{output}"
    import json as _json

    _json.loads(lines[0])  # raises if the pipe wrote its own text format instead


@pytest.mark.timeout(600)
@pytest.mark.parametrize("host_handler", ["plain", "none"], ids=["host-prints", "host-silent"])
@pytest.mark.parametrize(
    ("emit_level", "expected"),
    [("debug", 0), ("info", 0), ("warning", 1)],
    ids=["below", "below-2", "at"],
)
def test_the_console_honours_the_configured_threshold(host_handler, emit_level, expected):
    """LOG_LEVEL governs BOTH console sinks, not just the forwarder.

    The fallback handler was attached with no level and no filter while the package
    logger is pinned at DEBUG, so on a host with no root handler -- Open WebUI's DEFAULT,
    since env.py installs one only when GLOBAL_LOG_LEVEL names a real level -- every
    `logger.debug` in the package printed regardless of the valve. That path dumps the
    full outbound request payload and every non-delta SSE event: conversation text,
    system prompts and tool arguments.

    Parametrised over levels BELOW and AT the threshold, in BOTH host arms, because the
    guard that missed this asserted "exactly one line" against a record that was above
    the threshold either way. A count alone cannot distinguish a threshold that works
    from one that is never consulted, and a single level cannot either -- a sink hardwired
    to print everything and one hardwired to print nothing each satisfy one arm.
    """
    output = _console_probe(
        host_handler=host_handler, emit_level=emit_level, threshold="WARNING"
    )
    assert output.count("PROBE-MESSAGE") == expected, (
        f"with host_handler={host_handler!r}, LOG_LEVEL=WARNING and a {emit_level.upper()} "
        f"record, the console printed it {output.count('PROBE-MESSAGE')} times "
        f"(expected {expected}). A record below the configured threshold reaching stdout "
        "means the operator cannot turn off payload logging.\n"
        f"{output}"
    )


def test_a_hostile_message_object_still_lands_in_the_session_buffer():
    """`_safe_message` promises it never invokes user `__str__`/`__repr__`. Assert it.

    Nothing checked the promise: replacing `object.__repr__(msg)` with `str(msg)` left
    the whole suite green while a record whose message raises from both dunders was
    dropped entirely -- `_build_event` raises, `process_record`'s fallback calls
    `_safe_message`, that raises again, and the outer handler discards the record. The
    operator loses the one line describing the failure they are chasing.

    Asserted on the buffer, not on the returned text: the text is `object.__repr__`'s
    output today and pinning it would be a proxy for "the record survived".
    """
    import logging as _logging

    from open_webui_openrouter_pipe.core.logging_system import SessionLogger

    class _Hostile:
        def __str__(self):
            raise RuntimeError("hostile __str__")

        def __repr__(self):
            raise RuntimeError("hostile __repr__")

    record = _logging.LogRecord(
        name="test.hostile",
        level=_logging.ERROR,
        pathname="test.py",
        lineno=1,
        msg=_Hostile(),
        args=(),
        exc_info=None,
    )
    record.request_id = "req-hostile"
    record.session_id = "sess-hostile"
    record.user_id = "u"

    token = SessionLogger.logs.pop("req-hostile", None)
    try:
        SessionLogger.process_record(record)
        buffered = list(SessionLogger.logs.get("req-hostile") or [])
    finally:
        SessionLogger.logs.pop("req-hostile", None)
        if token is not None:
            SessionLogger.logs["req-hostile"] = token

    assert len(buffered) == 1, (
        f"a record whose message raises from both __str__ and __repr__ produced "
        f"{len(buffered)} buffered events, not 1. It was discarded entirely, so the "
        "failure it described reaches no sink at all."
    )
