"""Tests for open_webui_openrouter_pipe/requests/debug.py to achieve high coverage.

Target uncovered lines:
- Lines 41-43: Exception handling in _debug_print_request
- Line 52: Early return in _debug_print_response when DEBUG not enabled
- Lines 56-57: Exception handling in _debug_print_response
- Lines 71-74: _debug_print_error_response when DEBUG not enabled (incl. exception on resp.text())
- Lines 79-80: Exception reading response text in DEBUG-enabled path
- Lines 89-94: Exception in outer try block of _debug_print_error_response
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from open_webui_openrouter_pipe.requests.debug import (
    _debug_print_request,
    _debug_print_response,
    _debug_print_error_response,
)


# -----------------------------------------------------------------------------
# Helper classes
# -----------------------------------------------------------------------------


class _NoDebugLogger(logging.Logger):
    """Logger that reports DEBUG is NOT enabled."""

    def __init__(self, name: str = "test.nodebug"):
        super().__init__(name, level=logging.WARNING)

    def isEnabledFor(self, level: int) -> bool:
        return level >= logging.WARNING

    def debug(self, *args, **kwargs):
        raise AssertionError("debug() should not be called when DEBUG is disabled")


class _DebugEnabledLogger(logging.Logger):
    """Logger that reports DEBUG is enabled and captures messages."""

    def __init__(self, name: str = "test.debug"):
        super().__init__(name, level=logging.DEBUG)
        self.messages: list[str] = []

    def isEnabledFor(self, level: int) -> bool:
        return True

    def debug(self, msg, *args, **kwargs):
        formatted = msg % args if args else msg
        self.messages.append(formatted)


class _DebugLoggerThatFails(logging.Logger):
    """Logger that reports DEBUG is enabled but fails when debug() is called."""

    def __init__(self, name: str = "test.fail"):
        super().__init__(name, level=logging.DEBUG)
        self.debug_call_count = 0

    def isEnabledFor(self, level: int) -> bool:
        return True

    def debug(self, msg, *args, **kwargs):
        self.debug_call_count += 1
        # Fail on first few calls to trigger exception paths
        if self.debug_call_count <= 2:
            raise RuntimeError("Simulated logging failure")


# -----------------------------------------------------------------------------
# Tests for _debug_print_request
# -----------------------------------------------------------------------------


def test_debug_print_request_success_with_payload(caplog):
    """Test normal success path with headers and payload."""
    logger = logging.getLogger("test.debug.request.success")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="test.debug.request.success")

    headers = {"Authorization": "Bearer sk-12345678901234567890", "Content-Type": "application/json"}
    payload = {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}

    _debug_print_request(headers, payload, logger=logger)

    joined = "\n".join(record.message for record in caplog.records)
    assert "OpenRouter request headers" in joined
    assert "OpenRouter request payload" in joined
    # Authorization should be redacted
    assert "sk-12345678901234567890" not in joined


def test_debug_print_request_no_payload(caplog):
    """Test with None payload."""
    logger = logging.getLogger("test.debug.request.nopayload")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="test.debug.request.nopayload")

    headers = {"Authorization": "Bearer abc"}
    _debug_print_request(headers, None, logger=logger)

    joined = "\n".join(record.message for record in caplog.records)
    assert "OpenRouter request headers" in joined
    # No payload log expected since payload is None
    # The function should still complete without error


def test_debug_print_request_empty_headers(caplog):
    """Test with empty/None headers."""
    logger = logging.getLogger("test.debug.request.emptyheaders")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="test.debug.request.emptyheaders")

    _debug_print_request({}, {"key": "value"}, logger=logger)

    joined = "\n".join(record.message for record in caplog.records)
    assert "OpenRouter request headers" in joined


def test_debug_print_request_short_authorization(caplog):
    """Test with short Authorization token (<=10 chars)."""
    logger = logging.getLogger("test.debug.request.shortauth")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="test.debug.request.shortauth")

    headers = {"Authorization": "short"}
    _debug_print_request(headers, None, logger=logger)

    joined = "\n".join(record.message for record in caplog.records)
    # Short tokens are replaced with "***"
    assert "***" in joined


def test_debug_print_request_exception_handling():
    """Test exception handling in _debug_print_request (lines 41-43).

    When an exception occurs during logging, it should be caught and
    logged with exc_info=True rather than propagating.
    """
    logger = _DebugEnabledLogger("test.debug.request.exception")

    # Create a payload that will cause json.dumps to fail
    class UnserializableObject:
        pass

    # This should trigger the exception path
    payload = {"bad": UnserializableObject()}

    # Should not raise - exception is caught internally
    _debug_print_request({"Authorization": "Bearer test1234567890"}, payload, logger=logger)

    # The exception handler should have logged the failure message
    assert any("failed" in msg.lower() for msg in logger.messages)


def test_debug_print_request_redact_exception():
    """Test exception when _redact_payload_blobs raises (lines 41-43)."""
    logger = _DebugEnabledLogger("test.debug.request.redact_exc")

    # Patch at the source module since it's a late import
    with patch(
        "open_webui_openrouter_pipe.core.utils._redact_payload_blobs",
        side_effect=ValueError("Simulated redaction failure"),
    ):
        # Should not raise
        _debug_print_request({"Authorization": "Bearer test1234567890"}, {"key": "value"}, logger=logger)

    # The exception should be caught
    assert any("failed" in msg.lower() for msg in logger.messages)


# -----------------------------------------------------------------------------
# Tests for _debug_print_response
# -----------------------------------------------------------------------------


def test_debug_print_response_debug_disabled():
    """Test early return when DEBUG is not enabled (line 52)."""
    logger = _NoDebugLogger("test.nodebug.response")

    # This should return early without calling debug()
    _debug_print_response({"result": "test"}, logger=logger)
    # If debug() was called, it would raise AssertionError


def test_debug_print_response_success(caplog):
    """Test normal success path for response logging."""
    logger = logging.getLogger("test.debug.response.success")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="test.debug.response.success")

    _debug_print_response({"id": "123", "result": "ok"}, logger=logger)

    joined = "\n".join(record.message for record in caplog.records)
    assert "OpenRouter response payload" in joined


def test_debug_print_response_non_dict_payload(caplog):
    """Test with non-dict payload (bypasses redaction)."""
    logger = logging.getLogger("test.debug.response.nondict")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="test.debug.response.nondict")

    # Non-dict payloads are passed through directly
    _debug_print_response(["item1", "item2"], logger=logger)

    joined = "\n".join(record.message for record in caplog.records)
    assert "OpenRouter response payload" in joined


def test_debug_print_response_exception_handling():
    """Test exception handling in _debug_print_response (lines 56-57).

    When json.dumps fails, the exception should be caught and logged.
    """
    logger = _DebugEnabledLogger("test.debug.response.exception")

    # Create an object that can't be serialized
    class Unserializable:
        pass

    payload = {"bad": Unserializable()}

    # Should not raise
    _debug_print_response(payload, logger=logger)

    # The exception handler should have logged
    assert any("failed" in msg.lower() for msg in logger.messages)


def test_debug_print_response_redact_exception():
    """Test exception when redaction fails (lines 56-57)."""
    logger = _DebugEnabledLogger("test.debug.response.redact_exc")

    # Patch at the source module since it's a late import
    with patch(
        "open_webui_openrouter_pipe.core.utils._redact_payload_blobs",
        side_effect=RuntimeError("Redaction failed"),
    ):
        _debug_print_response({"key": "value"}, logger=logger)

    # Exception should be caught
    assert any("failed" in msg.lower() for msg in logger.messages)


# -----------------------------------------------------------------------------
# Tests for _debug_print_error_response
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_debug_print_error_response_debug_disabled_success():
    """Test early return path when DEBUG is not enabled (lines 71-72).

    Should read and return response text without debug logging.
    """
    logger = _NoDebugLogger("test.nodebug.error")

    mock_resp = MagicMock()
    mock_resp.text = AsyncMock(return_value="Error body text")

    result = await _debug_print_error_response(mock_resp, logger=logger)

    assert result == "Error body text"
    mock_resp.text.assert_awaited_once()


@pytest.mark.asyncio
async def test_debug_print_error_response_debug_disabled_text_fails():
    """Test exception reading text when DEBUG disabled (lines 73-74).

    When resp.text() fails and DEBUG is not enabled, should return error message.
    """
    logger = _NoDebugLogger("test.nodebug.error.fail")

    mock_resp = MagicMock()
    mock_resp.text = AsyncMock(side_effect=RuntimeError("Connection closed"))

    result = await _debug_print_error_response(mock_resp, logger=logger)

    assert "failed to read body" in result
    assert "Connection closed" in result


@pytest.mark.asyncio
async def test_debug_print_error_response_debug_enabled_success():
    """Test success path when DEBUG is enabled."""
    logger = _DebugEnabledLogger("test.debug.error.success")

    mock_resp = MagicMock()
    mock_resp.status = 400
    mock_resp.reason = "Bad Request"
    mock_resp.url = "https://openrouter.ai/api/v1/chat/completions"
    mock_resp.text = AsyncMock(return_value='{"error": "invalid request"}')

    result = await _debug_print_error_response(mock_resp, logger=logger)

    assert result == '{"error": "invalid request"}'
    assert any("OpenRouter error response" in msg for msg in logger.messages)


@pytest.mark.asyncio
async def test_debug_print_error_response_debug_enabled_text_fails():
    """Test exception reading text when DEBUG enabled (lines 79-80).

    When resp.text() fails inside the debug block, should capture the error.
    """
    logger = _DebugEnabledLogger("test.debug.error.textfail")

    mock_resp = MagicMock()
    mock_resp.status = 500
    mock_resp.reason = "Internal Server Error"
    mock_resp.url = "https://openrouter.ai/api/v1/chat/completions"
    mock_resp.text = AsyncMock(side_effect=RuntimeError("Stream exhausted"))

    result = await _debug_print_error_response(mock_resp, logger=logger)

    # Should contain the error message
    assert "failed to read body" in result
    assert "Stream exhausted" in result


@pytest.mark.asyncio
async def test_debug_print_error_response_outer_exception_text_success():
    """Test outer exception handler with successful text retry (lines 89-92).

    When the outer try block fails but resp.text() succeeds in the handler.
    """
    logger = _DebugEnabledLogger("test.debug.error.outer")

    mock_resp = MagicMock()
    mock_resp.status = 400
    mock_resp.reason = "Bad Request"
    mock_resp.url = "https://openrouter.ai/api/v1/chat/completions"

    # First text() call succeeds, but we need json.dumps to fail
    text_call_count = [0]

    async def text_side_effect():
        text_call_count[0] += 1
        return "Error body"

    mock_resp.text = AsyncMock(side_effect=text_side_effect)

    # Make json.dumps fail after the text is read
    with patch("json.dumps", side_effect=ValueError("JSON encoding failed")):
        result = await _debug_print_error_response(mock_resp, logger=logger)

    # Should fall through to exception handler and retry text()
    assert result == "Error body"
    # Exception should have been logged
    assert any("failed" in msg.lower() for msg in logger.messages)


@pytest.mark.asyncio
async def test_debug_print_error_response_outer_exception_text_fails():
    """Test outer exception handler when text retry also fails (lines 93-94).

    When both the outer try block and the fallback text() call fail.
    """
    logger = _DebugEnabledLogger("test.debug.error.outer.fail")

    mock_resp = MagicMock()
    mock_resp.status = 400
    mock_resp.reason = "Bad Request"
    mock_resp.url = "https://openrouter.ai/api/v1/chat/completions"

    call_count = [0]

    async def text_side_effect():
        call_count[0] += 1
        if call_count[0] == 1:
            # First call succeeds
            return "First body"
        # Subsequent calls fail
        raise ConnectionError("Connection lost")

    mock_resp.text = AsyncMock(side_effect=text_side_effect)

    # Make json.dumps fail to trigger the exception handler
    with patch("json.dumps", side_effect=ValueError("JSON failed")):
        result = await _debug_print_error_response(mock_resp, logger=logger)

    # Should return the error message from the failed second text() call
    assert "failed to read body" in result
    assert "Connection lost" in result


@pytest.mark.asyncio
async def test_debug_print_error_response_missing_attributes():
    """Test with response object missing some attributes."""
    logger = _DebugEnabledLogger("test.debug.error.missing")

    mock_resp = MagicMock()
    # Remove attributes to test getattr fallbacks
    del mock_resp.status
    del mock_resp.reason
    mock_resp.url = None
    mock_resp.text = AsyncMock(return_value="Error body")

    result = await _debug_print_error_response(mock_resp, logger=logger)

    assert result == "Error body"
    # Should still log (with None values for missing attributes)
    assert any("OpenRouter error response" in msg for msg in logger.messages)


@pytest.mark.asyncio
async def test_debug_print_error_response_getattr_fallbacks():
    """Test getattr fallbacks for status, reason, url attributes."""
    logger = _DebugEnabledLogger("test.debug.error.getattr")

    class MinimalResponse:
        async def text(self):
            return "Minimal body"

    mock_resp = MinimalResponse()

    result = await _debug_print_error_response(mock_resp, logger=logger)

    assert result == "Minimal body"
    # Verify the debug message was logged with None/empty values
    logged = "\n".join(logger.messages)
    assert "OpenRouter error response" in logged

# ===== From test_debug_logging_redaction.py =====


import logging

from open_webui_openrouter_pipe.core.utils import _redact_payload_blobs
from open_webui_openrouter_pipe.requests.debug import (
    _debug_print_request,
    _debug_print_response,
)


def test_redact_payload_blobs_truncates_data_urls():
    payload = {"image": "data:image/png;base64," + ("a" * 300)}
    redacted = _redact_payload_blobs(payload, max_chars=80)

    assert redacted["image"].startswith("data:image/png;base64,")
    assert "[REDACTED]" in redacted["image"]
    assert len(redacted["image"]) < len(payload["image"])


def test_redact_payload_blobs_handles_nested_structures():
    payload = {"items": [{"inner": "data:image/png;base64," + ("a" * 300)}]}
    redacted = _redact_payload_blobs(payload, max_chars=80)

    assert "[REDACTED]" in redacted["items"][0]["inner"]


def test_debug_print_request_redacts_authorization_and_payload(caplog):
    logger = logging.getLogger("tests.debug.request")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="tests.debug.request")

    token = "Bearer 1234567890abcdef"
    payload = {"image": "data:image/png;base64," + ("a" * 300)}

    _debug_print_request({"Authorization": token}, payload, logger=logger)

    joined = "\n".join(record.message for record in caplog.records)
    assert "OpenRouter request headers" in joined
    assert "OpenRouter request payload" in joined
    assert "abcdef" not in joined
    assert "[REDACTED]" in joined


def test_debug_print_request_no_output_when_not_debug():
    class _NoDebugLogger(logging.Logger):
        def isEnabledFor(self, level):
            return False

        def debug(self, *_args, **_kwargs):
            raise AssertionError("debug should not be called when DEBUG is disabled")

    _debug_print_request(
        {"Authorization": "Bearer 1234567890abcdef"},
        {"k": "v"},
        logger=_NoDebugLogger("tests.debug.nooutput"),
    )


def test_debug_print_response_redacts_payload(caplog):
    logger = logging.getLogger("tests.debug.response")
    logger.setLevel(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="tests.debug.response")

    _debug_print_response({"image": "data:image/png;base64," + ("a" * 300)}, logger=logger)

    joined = "\n".join(record.message for record in caplog.records)
    assert "OpenRouter response payload" in joined
    assert "[REDACTED]" in joined
