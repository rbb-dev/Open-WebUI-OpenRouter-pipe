"""Coverage-focused tests for timing_logger.py.

Targets missing lines: 86-87, 100, 103, 122-123, 154-158, 168-171, 185-186,
219-220, 272-273, 287, 292-294, 428-443.

Tests:
- _format_iso_utc exception path (lines 86-87)
- _record_event early returns when disabled/no request_id (lines 100, 103)
- _record_event write exception handling (lines 122-123)
- configure_timing_file closing existing handle (lines 154-158)
- configure_timing_file failure to open (lines 168-171)
- close_timing_file exception when closing (lines 185-186)
- ensure_timing_file_configured close exception on path change (lines 219-220)
- clear_timing_events (lines 272-273)
- format_timing_jsonl empty events (line 287)
- format_timing_jsonl malformed event exception (lines 292-294)
- @timed async function with timing enabled (lines 428-443)
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import asyncio
import datetime
import json
from io import StringIO
from typing import Any
from unittest import mock

import pytest

from open_webui_openrouter_pipe.core import timing_logger as tl


@pytest.fixture(autouse=True)
def _reset_timing_state():
    """Reset all timing logger global state before and after each test."""
    tl.close_timing_file()
    tl.clear_timing_context()
    with tl._timing_lock:
        tl._timing_events.clear()
    # Reset global file state
    with tl._timing_file_lock:
        tl._timing_file_handle = None
        tl._timing_file_path = None
    yield
    tl.close_timing_file()
    tl.clear_timing_context()
    with tl._timing_lock:
        tl._timing_events.clear()
    with tl._timing_file_lock:
        tl._timing_file_handle = None
        tl._timing_file_path = None


class TestFormatIsoUtcExceptionPath:
    """Test _format_iso_utc exception handling (lines 86-87)."""

    def test_format_iso_utc_with_invalid_timestamp(self):
        """When fromtimestamp raises, fallback to datetime.now()."""
        # Use an extremely large timestamp that causes OverflowError/OSError
        # Values beyond year 9999 cause overflow on most systems
        invalid_timestamp = 1e20  # Far beyond valid range

        result = tl._format_iso_utc(invalid_timestamp)
        # Should return a valid ISO timestamp (fallback path)
        assert result.endswith("Z")
        assert "T" in result  # ISO format contains T separator

    def test_format_iso_utc_with_negative_overflow(self):
        """Test with negative timestamp causing exception."""
        # Very negative timestamps can also cause issues
        invalid_timestamp = -1e20

        result = tl._format_iso_utc(invalid_timestamp)
        # Should return a valid ISO timestamp (fallback path)
        assert result.endswith("Z")
        assert "T" in result


class TestRecordEventEarlyReturns:
    """Test _record_event early return paths (lines 100, 103)."""

    def test_record_event_when_timing_disabled(self, tmp_path):
        """_record_event returns early when timing is disabled (line 100)."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        # Timing disabled (default)
        tl.set_timing_context("req-disabled", enabled=False)
        event = tl.TimingEvent(
            ts=1.0, wall_ts=1234567890.0, event="mark", label="test"
        )
        tl._record_event(event)

        # No events should be recorded
        assert tl.get_timing_events("req-disabled") == []
        assert path.read_text() == ""

    def test_record_event_when_no_request_id(self, tmp_path):
        """_record_event returns early when no request_id set (line 103)."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        # Enable timing but don't set request_id
        tl._timing_enabled.set(True)
        tl._timing_request_id.set(None)

        event = tl.TimingEvent(
            ts=1.0, wall_ts=1234567890.0, event="mark", label="test"
        )
        tl._record_event(event)

        # No events should be recorded
        assert path.read_text() == ""


class TestRecordEventWriteException:
    """Test _record_event write exception handling (lines 122-123)."""

    def test_record_event_handles_write_exception(self, tmp_path):
        """_record_event silently ignores write errors."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        tl.set_timing_context("req-write-error", enabled=True)

        # Create a mock file handle that raises on write
        mock_handle = mock.MagicMock()
        mock_handle.write.side_effect = IOError("Disk full")

        with tl._timing_file_lock:
            original_handle = tl._timing_file_handle
            tl._timing_file_handle = mock_handle

        try:
            # This should not raise - exception is caught silently
            tl.timing_mark("test_mark")

            # Event should still be added to in-memory buffer
            events = tl.get_timing_events("req-write-error")
            mark_events = [
                event
                for event in events
                if event["event"] == "mark" and event["label"] == "test_mark"
            ]
            assert len(mark_events) == 1
        finally:
            with tl._timing_file_lock:
                tl._timing_file_handle = original_handle


class TestConfigureTimingFileCloseExisting:
    """Test configure_timing_file closing existing handle (lines 154-158)."""

    def test_configure_timing_file_closes_existing_handle(self, tmp_path):
        """configure_timing_file closes existing file before opening new one."""
        path1 = tmp_path / "first.jsonl"
        path2 = tmp_path / "second.jsonl"

        # Configure first file
        assert tl.configure_timing_file(str(path1)) is True
        assert tl._timing_file_handle is not None

        # Configure second file - should close first
        assert tl.configure_timing_file(str(path2)) is True
        assert tl._timing_file_path is not None
        assert tl._timing_file_path.name == "second.jsonl"

    def test_configure_timing_file_handles_close_exception(self, tmp_path):
        """configure_timing_file handles exception when closing existing file."""
        path1 = tmp_path / "first.jsonl"
        path2 = tmp_path / "second.jsonl"

        # Configure first file
        assert tl.configure_timing_file(str(path1)) is True

        # Replace handle with mock that raises on close
        mock_handle = mock.MagicMock()
        mock_handle.close.side_effect = IOError("Close failed")

        with tl._timing_file_lock:
            tl._timing_file_handle = mock_handle

        # Should still succeed - exception is caught
        assert tl.configure_timing_file(str(path2)) is True
        assert tl._timing_file_path.name == "second.jsonl"


class TestConfigureTimingFileOpenFailure:
    """Test configure_timing_file failure to open (lines 168-171)."""

    def test_configure_timing_file_returns_false_on_open_failure(self):
        """configure_timing_file returns False when open() fails."""
        # Try to open a file in a path that doesn't exist and can't be created
        # On Unix, /proc is read-only
        invalid_path = "/proc/nonexistent/subdir/timing.jsonl"

        result = tl.configure_timing_file(invalid_path)

        assert result is False
        assert tl._timing_file_path is None
        assert tl._timing_file_handle is None

    def test_configure_timing_file_with_permission_error(self, tmp_path):
        """configure_timing_file handles permission errors gracefully."""
        # Mock open to raise PermissionError
        with mock.patch("builtins.open", side_effect=PermissionError("Access denied")):
            result = tl.configure_timing_file(str(tmp_path / "test.jsonl"))

        assert result is False
        assert tl._timing_file_path is None
        assert tl._timing_file_handle is None


class TestCloseTimingFileException:
    """Test close_timing_file exception handling (lines 185-186)."""

    def test_close_timing_file_handles_close_exception(self, tmp_path):
        """close_timing_file handles exception when closing file."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        # Replace handle with mock that raises on close
        mock_handle = mock.MagicMock()
        mock_handle.close.side_effect = IOError("Close failed")

        with tl._timing_file_lock:
            tl._timing_file_handle = mock_handle

        # Should not raise - exception is caught
        tl.close_timing_file()

        # State should be cleaned up
        assert tl._timing_file_handle is None
        assert tl._timing_file_path is None


class TestEnsureTimingFileConfiguredCloseException:
    """Test ensure_timing_file_configured close exception (lines 219-220)."""

    def test_ensure_timing_file_configured_handles_close_exception_on_path_change(
        self, tmp_path
    ):
        """ensure_timing_file_configured handles close exception when path changes."""
        path1 = tmp_path / "first.jsonl"
        path2 = tmp_path / "second.jsonl"

        # Configure first file
        assert tl.configure_timing_file(str(path1)) is True

        # Replace handle with mock that raises on close
        mock_handle = mock.MagicMock()
        mock_handle.close.side_effect = IOError("Close failed")

        with tl._timing_file_lock:
            tl._timing_file_handle = mock_handle

        # Should still succeed when path changes - exception is caught
        result = tl.ensure_timing_file_configured(str(path2))

        assert result is True
        assert tl._timing_file_path.name == "second.jsonl"


class TestClearTimingEvents:
    """Test clear_timing_events function (lines 272-273)."""

    def test_clear_timing_events_removes_request_events(self, tmp_path):
        """clear_timing_events removes events for specified request."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        # Add events for a request
        tl.set_timing_context("req-clear", enabled=True)
        tl.timing_mark("mark1")
        tl.timing_mark("mark2")

        # Verify events exist
        events = tl.get_timing_events("req-clear")
        assert len(events) == 2

        # Clear events
        tl.clear_timing_events("req-clear")

        # Verify events are gone
        events = tl.get_timing_events("req-clear")
        assert events == []

    def test_clear_timing_events_nonexistent_request(self):
        """clear_timing_events handles non-existent request gracefully."""
        # Should not raise
        tl.clear_timing_events("nonexistent-request")

        # Verify no events
        events = tl.get_timing_events("nonexistent-request")
        assert events == []


class TestFormatTimingJsonl:
    """Test format_timing_jsonl function (lines 287, 292-294)."""

    def test_format_timing_jsonl_empty_events(self):
        """format_timing_jsonl returns empty string when no events (line 287)."""
        result = tl.format_timing_jsonl("nonexistent-request")
        assert result == ""

    def test_format_timing_jsonl_with_malformed_event(self, tmp_path):
        """format_timing_jsonl skips malformed events (lines 292-294)."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        # Add normal events
        tl.set_timing_context("req-malformed", enabled=True)
        tl.timing_mark("normal_mark")

        # Manually inject a malformed event into the buffer
        with tl._timing_lock:
            buffer = tl._timing_events.get("req-malformed")
            if buffer is not None:
                # Add an object that can't be JSON serialized
                class NonSerializable:
                    pass

                buffer.append({"label": "bad", "data": NonSerializable()})

        # format_timing_jsonl should skip the malformed event
        result = tl.format_timing_jsonl("req-malformed")

        # Should contain the normal mark but not crash
        assert "normal_mark" in result
        assert result.endswith("\n")

        # Parse the valid lines
        lines = [l for l in result.strip().split("\n") if l]
        # At least one valid line (the normal_mark)
        assert len(lines) >= 1
        # Verify we can parse the valid event
        valid_event = json.loads(lines[0])
        assert valid_event["label"] == "normal_mark"


class TestTimedAsyncFunction:
    """Test @timed decorator with async functions when enabled (lines 428-443)."""

    def test_timed_async_function_with_timing_enabled(self, tmp_path):
        """@timed records enter/exit for async functions when enabled."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        tl.set_timing_context("req-async", enabled=True)

        @tl.timed
        async def async_operation():
            await asyncio.sleep(0.01)
            return "async_result"

        # Run the async function
        result = asyncio.run(async_operation())

        assert result == "async_result"

        # Check events
        events = tl.get_timing_events("req-async")
        assert len(events) == 2

        # First event is enter
        assert events[0]["event"] == "enter"
        assert "async_operation" in events[0]["label"]

        # Second event is exit with elapsed_ms
        assert events[1]["event"] == "exit"
        assert "async_operation" in events[1]["label"]
        assert "elapsed_ms" in events[1]
        assert events[1]["elapsed_ms"] >= 10  # At least 10ms from sleep

    def test_timed_async_function_with_timing_disabled(self, tmp_path):
        """@timed bypasses timing for async functions when disabled."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        tl.set_timing_context("req-async-disabled", enabled=False)

        @tl.timed
        async def async_operation():
            return "async_result"

        # Run the async function
        result = asyncio.run(async_operation())

        assert result == "async_result"

        # No events recorded
        events = tl.get_timing_events("req-async-disabled")
        assert events == []

    def test_timed_async_function_with_exception(self, tmp_path):
        """@timed records exit even when async function raises."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        tl.set_timing_context("req-async-exc", enabled=True)

        @tl.timed
        async def async_operation_with_error():
            await asyncio.sleep(0.005)
            raise ValueError("Async error")

        # Run the async function and expect exception
        with pytest.raises(ValueError, match="Async error"):
            asyncio.run(async_operation_with_error())

        # Check events - should still have enter AND exit
        events = tl.get_timing_events("req-async-exc")
        assert len(events) == 2
        assert events[0]["event"] == "enter"
        assert events[1]["event"] == "exit"
        assert "elapsed_ms" in events[1]


class TestTimedSyncFunctionWithException:
    """Test @timed decorator with sync functions that raise exceptions."""

    def test_timed_sync_function_with_exception(self, tmp_path):
        """@timed records exit even when sync function raises."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        tl.set_timing_context("req-sync-exc", enabled=True)

        @tl.timed
        def sync_operation_with_error():
            raise ValueError("Sync error")

        # Run the function and expect exception
        with pytest.raises(ValueError, match="Sync error"):
            sync_operation_with_error()

        # Check events - should still have enter AND exit
        events = tl.get_timing_events("req-sync-exc")
        sync_events = [
            event
            for event in events
            if "sync_operation_with_error" in event["label"]
        ]
        assert len(sync_events) == 2
        assert sync_events[0]["event"] == "enter"
        assert sync_events[1]["event"] == "exit"


class TestTimingScopeWithException:
    """Test timing_scope context manager with exceptions."""

    def test_timing_scope_with_exception(self, tmp_path):
        """timing_scope records exit even when block raises."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        tl.set_timing_context("req-scope-exc", enabled=True)

        with pytest.raises(ValueError, match="Scope error"):
            with tl.timing_scope("error_scope"):
                raise ValueError("Scope error")

        # Check events - should still have enter AND exit
        events = tl.get_timing_events("req-scope-exc")
        assert len(events) == 2
        assert events[0]["event"] == "enter"
        assert events[0]["label"] == "error_scope"
        assert events[1]["event"] == "exit"
        assert events[1]["label"] == "error_scope"
        assert "elapsed_ms" in events[1]


class TestTimingMarkIntegration:
    """Integration tests for timing_mark."""

    def test_timing_mark_records_event_with_correct_format(self, tmp_path):
        """timing_mark creates event with correct structure."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        tl.set_timing_context("req-mark", enabled=True)
        tl.timing_mark("checkpoint_1")

        events = tl.get_timing_events("req-mark")
        assert len(events) == 1

        event = events[0]
        assert event["event"] == "mark"
        assert event["label"] == "checkpoint_1"
        assert event["request_id"] == "req-mark"
        assert "ts" in event  # ISO timestamp
        assert "perf_ts" in event  # Performance counter


class TestEdgeCases:
    """Edge case tests for additional coverage."""

    def test_get_timing_events_nonexistent_request(self):
        """get_timing_events returns empty list for unknown request."""
        events = tl.get_timing_events("unknown-request-id")
        assert events == []

    def test_multiple_requests_isolated(self, tmp_path):
        """Events from different requests are properly isolated."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        # Request 1
        tl.set_timing_context("req-1", enabled=True)
        tl.timing_mark("mark_req1")

        # Request 2
        tl.set_timing_context("req-2", enabled=True)
        tl.timing_mark("mark_req2")

        # Verify isolation
        events_1 = tl.get_timing_events("req-1")
        events_2 = tl.get_timing_events("req-2")

        assert len(events_1) == 1
        assert events_1[0]["label"] == "mark_req1"

        assert len(events_2) == 1
        assert events_2[0]["label"] == "mark_req2"

    def test_timed_decorator_label_format(self, tmp_path):
        """@timed decorator creates appropriate label from function name."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        tl.set_timing_context("req-label", enabled=True)

        @tl.timed
        def my_test_function():
            return 42

        result = my_test_function()
        assert result == 42

        events = tl.get_timing_events("req-label")
        assert len(events) == 2
        # Label should contain function name
        assert "my_test_function" in events[0]["label"]

    def test_close_timing_file_when_not_open(self):
        """close_timing_file is safe to call when no file is open."""
        # Should not raise
        tl.close_timing_file()
        tl.close_timing_file()  # Call twice to verify idempotency

        assert tl._timing_file_handle is None
        assert tl._timing_file_path is None

    def test_ensure_timing_file_configured_same_path(self, tmp_path):
        """ensure_timing_file_configured returns True for same path."""
        path = tmp_path / "timing.jsonl"

        # Configure initially
        assert tl.configure_timing_file(str(path)) is True

        # Ensure with same path should return True without reopening
        assert tl.ensure_timing_file_configured(str(path)) is True
        assert tl._timing_file_path.name == "timing.jsonl"


class TestTimingScopeContextManager:
    """Tests for the timing_scope context manager."""

    def test_timing_scope_disabled_early_return(self, tmp_path):
        """timing_scope yields immediately when timing disabled (lines 355-356)."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        # Set context with timing DISABLED
        tl.set_timing_context("req-scope-disabled", enabled=False)

        # Use the context manager - should do nothing
        executed = False
        with tl.timing_scope("should_not_record"):
            executed = True

        assert executed is True

        # No events should be recorded
        events = tl.get_timing_events("req-scope-disabled")
        assert events == []

        # File should be empty
        assert path.read_text() == ""

    def test_timing_scope_enabled_records_events(self, tmp_path):
        """timing_scope records enter/exit events when timing enabled."""
        path = tmp_path / "timing.jsonl"
        assert tl.configure_timing_file(str(path)) is True

        # Set context with timing ENABLED
        tl.set_timing_context("req-scope-enabled", enabled=True)

        with tl.timing_scope("my_scope"):
            pass  # Do something

        events = tl.get_timing_events("req-scope-enabled")
        # Should have enter and exit events
        assert len(events) == 2
        assert events[0]["event"] == "enter"
        assert events[0]["label"] == "my_scope"
        assert events[1]["event"] == "exit"
        assert events[1]["label"] == "my_scope"
        # Exit event should have elapsed time (in ms)
        assert "elapsed_ms" in events[1]

    def test_timing_scope_disabled_no_overhead(self):
        """timing_scope with disabled timing has minimal overhead."""
        # No file configured, timing disabled
        tl.set_timing_context("req-no-overhead", enabled=False)

        # Should execute without any errors or side effects
        result = None
        with tl.timing_scope("zero_overhead"):
            result = 42

        assert result == 42
