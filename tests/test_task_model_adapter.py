"""Coverage tests for TaskModelAdapter.

These tests use real Pipe() instances with HTTP mocked at the boundary.
Target: 90%+ coverage on open_webui_openrouter_pipe/requests/task_model_adapter.py

Missing lines to cover:
- Line 48: response not a dict
- Line 55: item not a message
- Line 58: content not a dict
- Line 65: fallback output_text field
- Lines 111-114: USE_MODEL_MAX_OUTPUT_TOKENS path
- Line 141: session is None
- Lines 176-206: error handling and retry paths
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aioresponses import aioresponses, CallbackResult

from open_webui_openrouter_pipe import Pipe, EncryptedStr, OpenRouterAPIError
from open_webui_openrouter_pipe.requests.task_model_adapter import TaskModelAdapter
from open_webui_openrouter_pipe.models.registry import ModelFamily


# ============================================================================
# Helper Functions
# ============================================================================


def _make_responses_json(content: str = "Generated Title") -> dict:
    """Create a Responses API JSON response."""
    return {
        "id": "resp_123",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _make_callback(
    captured_payloads: list[dict],
    response_content: str = "Generated Title",
    status: int = 200,
    error_response: dict | None = None,
):
    """Create a callback for aioresponses that captures payloads."""

    def callback(url, **kwargs):
        payload = kwargs.get("json", {})
        captured_payloads.append(payload)

        if error_response is not None or status >= 400:
            return CallbackResult(
                body=json.dumps(error_response or {"error": {"message": "Error"}}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                status=status,
            )

        return CallbackResult(
            body=json.dumps(_make_responses_json(response_content)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            status=200,
        )

    return callback


# ============================================================================
# Test _extract_task_output_text - Line 48 (response not dict)
# ============================================================================


def test_extract_task_output_text_non_dict_response(pipe_instance):
    """Test _extract_task_output_text with non-dict response (line 48)."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    # Test with non-dict responses - should return empty string
    assert adapter._extract_task_output_text(None) == ""
    assert adapter._extract_task_output_text("string response") == ""
    assert adapter._extract_task_output_text([1, 2, 3]) == ""
    assert adapter._extract_task_output_text(123) == ""


# ============================================================================
# Test _extract_task_output_text - Line 55 (item not message type)
# ============================================================================


def test_extract_task_output_text_non_message_item(pipe_instance):
    """Test _extract_task_output_text with non-message items in output (line 55)."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    # Output with items that are not message type
    response = {
        "output": [
            {"type": "function_call", "call_id": "call_1", "name": "test"},
            "not_a_dict",
            {"type": "thinking", "content": "thinking text"},
            {"type": "message", "content": [{"type": "output_text", "text": "real text"}]},
        ]
    }
    result = adapter._extract_task_output_text(response)
    assert result == "real text"


def test_extract_task_output_text_item_missing_type(pipe_instance):
    """Test _extract_task_output_text with items missing type key (line 55)."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    # Items without type key should be skipped
    response = {
        "output": [
            {"content": [{"type": "output_text", "text": "no type key"}]},
        ]
    }
    result = adapter._extract_task_output_text(response)
    assert result == ""


# ============================================================================
# Test _extract_task_output_text - Line 58 (content not dict)
# ============================================================================


def test_extract_task_output_text_non_dict_content(pipe_instance):
    """Test _extract_task_output_text with non-dict content items (line 58)."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    # Message with non-dict content items
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    "string_content",
                    123,
                    None,
                    {"type": "output_text", "text": "valid text"},
                ],
            }
        ]
    }
    result = adapter._extract_task_output_text(response)
    assert result == "valid text"


# ============================================================================
# Test _extract_task_output_text - Line 65 (fallback output_text)
# ============================================================================


def test_extract_task_output_text_fallback_output_text(pipe_instance):
    """Test _extract_task_output_text with fallback output_text field (line 65)."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    # Response with collapsed output_text field (some providers return this)
    response = {
        "output": [],
        "output_text": "Fallback text content",
    }
    result = adapter._extract_task_output_text(response)
    assert result == "Fallback text content"


def test_extract_task_output_text_combined_output_and_fallback(pipe_instance):
    """Test _extract_task_output_text combines output items and fallback (line 65)."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    # Both output items and fallback field
    response = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "From output"}],
            }
        ],
        "output_text": "From fallback",
    }
    result = adapter._extract_task_output_text(response)
    assert "From output" in result
    assert "From fallback" in result


def test_extract_task_output_text_fallback_not_string(pipe_instance):
    """Test _extract_task_output_text ignores non-string fallback (line 64)."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    # Non-string output_text should be ignored
    response = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "valid"}],
            }
        ],
        "output_text": {"not": "a string"},
    }
    result = adapter._extract_task_output_text(response)
    assert result == "valid"


# ============================================================================
# Test _task_name helper
# ============================================================================


def test_task_name_with_string(pipe_instance):
    """Test _task_name with string input."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    assert adapter._task_name("title_generation") == "title_generation"
    assert adapter._task_name("  tags_generation  ") == "tags_generation"
    assert adapter._task_name("") == ""


def test_task_name_with_dict(pipe_instance):
    """Test _task_name with dict input."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    assert adapter._task_name({"type": "title_generation"}) == "title_generation"
    assert adapter._task_name({"task": "tags_generation"}) == "tags_generation"
    assert adapter._task_name({"name": "summary"}) == "summary"
    assert adapter._task_name({"other": "key"}) == ""
    assert adapter._task_name({}) == ""


def test_task_name_with_other_types(pipe_instance):
    """Test _task_name with other input types."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    assert adapter._task_name(None) == ""
    assert adapter._task_name(123) == ""
    assert adapter._task_name(["list"]) == ""


# ============================================================================
# Test _run_task_model_request - Line 141 (session is None)
# ============================================================================


@pytest.mark.asyncio
async def test_task_model_request_session_none(pipe_instance_async):
    """Test _run_task_model_request raises when session is None (line 141)."""
    pipe = pipe_instance_async
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    body = {"model": "openai/gpt-4o-mini", "input": "test"}

    with pytest.raises(RuntimeError, match="HTTP session is required"):
        await adapter._run_task_model_request(
            body,
            pipe.valves,
            session=None,
        )


# ============================================================================
# Test _run_task_model_request - Lines 111-114 (USE_MODEL_MAX_OUTPUT_TOKENS)
# ============================================================================


@pytest.mark.asyncio
async def test_task_model_request_use_model_max_output_tokens(pipe_instance_async):
    """Test _run_task_model_request applies max_output_tokens from model spec (lines 111-114)."""
    pipe = pipe_instance_async
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.USE_MODEL_MAX_OUTPUT_TOKENS = True

    # Set up model spec with max_completion_tokens
    ModelFamily.set_dynamic_specs({
        "openai.gpt-4o-mini": {
            "features": set(),
            "capabilities": {},
            "max_completion_tokens": 4096,
            "supported_parameters": frozenset(),
        }
    })

    captured_payloads: list[dict] = []
    callback = _make_callback(captured_payloads, "Generated Title")

    session = pipe._create_http_session(pipe.valves)

    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                callback=callback,
                repeat=True,
            )

            adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))
            result = await adapter._run_task_model_request(
                {"model": "openai/gpt-4o-mini", "input": "test"},
                pipe.valves,
                session=session,
            )

            assert result == "Generated Title"

            # Check that max_output_tokens was set
            assert len(captured_payloads) >= 1
            assert captured_payloads[0].get("max_output_tokens") == 4096
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_task_model_request_use_model_max_output_tokens_already_set(pipe_instance_async):
    """Test _run_task_model_request doesn't override explicit max_output_tokens (lines 111-114)."""
    pipe = pipe_instance_async
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.USE_MODEL_MAX_OUTPUT_TOKENS = True

    # Set up model spec with max_completion_tokens
    ModelFamily.set_dynamic_specs({
        "openai.gpt-4o-mini": {
            "features": set(),
            "capabilities": {},
            "max_completion_tokens": 4096,
            "supported_parameters": frozenset(),
        }
    })

    captured_payloads: list[dict] = []
    callback = _make_callback(captured_payloads, "Generated Title")

    session = pipe._create_http_session(pipe.valves)

    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                callback=callback,
                repeat=True,
            )

            adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))
            result = await adapter._run_task_model_request(
                {"model": "openai/gpt-4o-mini", "input": "test", "max_output_tokens": 1000},
                pipe.valves,
                session=session,
            )

            assert result == "Generated Title"

            # Check that explicit max_output_tokens was preserved
            assert len(captured_payloads) >= 1
            assert captured_payloads[0].get("max_output_tokens") == 1000
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_task_model_request_use_model_max_output_tokens_disabled(pipe_instance_async):
    """Test _run_task_model_request removes max_output_tokens when disabled (lines 116)."""
    pipe = pipe_instance_async
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    pipe.valves.USE_MODEL_MAX_OUTPUT_TOKENS = False

    captured_payloads: list[dict] = []
    callback = _make_callback(captured_payloads, "Generated Title")

    session = pipe._create_http_session(pipe.valves)

    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                callback=callback,
                repeat=True,
            )

            adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))
            # Include max_output_tokens in body - it should be removed
            result = await adapter._run_task_model_request(
                {"model": "openai/gpt-4o-mini", "input": "test", "max_output_tokens": 1000},
                pipe.valves,
                session=session,
            )

            assert result == "Generated Title"

            # max_output_tokens should be removed when valve is disabled
            assert len(captured_payloads) >= 1
            assert "max_output_tokens" not in captured_payloads[0]
    finally:
        await session.close()


# ============================================================================
# Test _run_task_model_request - Lines 176-178 (empty output_text)
# ============================================================================


@pytest.mark.asyncio
async def test_task_model_request_empty_output_retries(pipe_instance_async):
    """Test _run_task_model_request retries when output is empty (lines 176-206)."""
    pipe = pipe_instance_async
    pipe.valves.API_KEY = EncryptedStr("test-api-key")

    call_count = [0]

    def callback(url, **kwargs):
        call_count[0] += 1
        # Return empty output
        return CallbackResult(
            body=json.dumps({
                "id": "resp_123",
                "output": [],
                "usage": {"input_tokens": 10, "output_tokens": 0},
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            status=200,
        )

    session = pipe._create_http_session(pipe.valves)

    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                callback=callback,
                repeat=True,
            )

            adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))
            result = await adapter._run_task_model_request(
                {"model": "openai/gpt-4o-mini", "input": "test"},
                pipe.valves,
                session=session,
                task_context="title_generation",
            )

            # Should return error message after retries
            assert "[Task error]" in result
            assert "title_generation" in result
            # Should have retried (2 attempts)
            assert call_count[0] == 2
    finally:
        await session.close()


# ============================================================================
# Test _run_task_model_request - Lines 180-199 (exception handling)
# ============================================================================


@pytest.mark.asyncio
async def test_task_model_request_generic_exception_retry(pipe_instance_async):
    """Test _run_task_model_request retries on generic exception (lines 180-199)."""
    pipe = pipe_instance_async
    pipe.valves.API_KEY = EncryptedStr("test-api-key")

    call_count = [0]

    def callback(url, **kwargs):
        call_count[0] += 1
        # Return 500 error
        return CallbackResult(
            body=json.dumps({"error": {"message": "Internal server error"}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            status=500,
        )

    session = pipe._create_http_session(pipe.valves)

    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                callback=callback,
                repeat=True,
            )

            adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))
            result = await adapter._run_task_model_request(
                {"model": "openai/gpt-4o-mini", "input": "test"},
                pipe.valves,
                session=session,
                task_context="tags_generation",
            )

            # Should return error message after retries
            assert "[Task error]" in result
            assert "tags_generation" in result
            # Should have retried (2 attempts)
            assert call_count[0] == 2
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_task_model_request_auth_failure_no_retry(pipe_instance_async):
    """Test _run_task_model_request does not retry on 401/403 (lines 182-196)."""
    pipe = pipe_instance_async
    pipe.valves.API_KEY = EncryptedStr("test-api-key")

    call_count = [0]

    def callback(url, **kwargs):
        call_count[0] += 1
        return CallbackResult(
            body=json.dumps({"error": {"message": "Invalid API key"}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            status=401,
        )

    session = pipe._create_http_session(pipe.valves)

    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                callback=callback,
                repeat=True,
            )

            adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))
            result = await adapter._run_task_model_request(
                {"model": "openai/gpt-4o-mini", "input": "test"},
                pipe.valves,
                session=session,
                task_context="title_generation",
            )

            # Should return error message
            assert "[Task error]" in result
            # Should NOT retry on auth failure (only 1 attempt)
            assert call_count[0] == 1
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_task_model_request_auth_failure_403(pipe_instance_async):
    """Test _run_task_model_request does not retry on 403 (lines 182-196)."""
    pipe = pipe_instance_async
    pipe.valves.API_KEY = EncryptedStr("test-api-key")

    call_count = [0]

    def callback(url, **kwargs):
        call_count[0] += 1
        return CallbackResult(
            body=json.dumps({"error": {"message": "Forbidden"}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            status=403,
        )

    session = pipe._create_http_session(pipe.valves)

    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                callback=callback,
                repeat=True,
            )

            adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))
            result = await adapter._run_task_model_request(
                {"model": "openai/gpt-4o-mini", "input": "test"},
                pipe.valves,
                session=session,
                task_context="summary",
            )

            # Should return error message
            assert "[Task error]" in result
            # Should NOT retry on auth failure (only 1 attempt)
            assert call_count[0] == 1
    finally:
        await session.close()


# ============================================================================
# Test _run_task_model_request - Line 201 (task_name from context)
# ============================================================================


@pytest.mark.asyncio
async def test_task_model_request_task_name_in_error(pipe_instance_async):
    """Test _run_task_model_request uses task_context in error message (line 201)."""
    pipe = pipe_instance_async
    pipe.valves.API_KEY = EncryptedStr("test-api-key")

    def callback(url, **kwargs):
        return CallbackResult(
            body=json.dumps({"error": {"message": "Server error"}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            status=500,
        )

    session = pipe._create_http_session(pipe.valves)

    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                callback=callback,
                repeat=True,
            )

            adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

            # Test with dict task_context
            result = await adapter._run_task_model_request(
                {"model": "openai/gpt-4o-mini", "input": "test"},
                pipe.valves,
                session=session,
                task_context={"type": "custom_task"},
            )

            assert "custom_task" in result
    finally:
        await session.close()


@pytest.mark.asyncio
async def test_task_model_request_fallback_task_name(pipe_instance_async):
    """Test _run_task_model_request uses 'task' as fallback name (line 201)."""
    pipe = pipe_instance_async
    pipe.valves.API_KEY = EncryptedStr("test-api-key")

    def callback(url, **kwargs):
        return CallbackResult(
            body=json.dumps({"error": {"message": "Server error"}}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            status=500,
        )

    session = pipe._create_http_session(pipe.valves)

    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                callback=callback,
                repeat=True,
            )

            adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

            # Test with None task_context - should use "task" as fallback
            result = await adapter._run_task_model_request(
                {"model": "openai/gpt-4o-mini", "input": "test"},
                pipe.valves,
                session=session,
                task_context=None,
            )

            assert "[Task error]" in result
            assert "task" in result.lower()
    finally:
        await session.close()


# ============================================================================
# Test _run_task_model_request - Success with usage tracking
# ============================================================================


@pytest.mark.asyncio
async def test_task_model_request_success_with_usage(pipe_instance_async):
    """Test _run_task_model_request tracks usage on success (lines 153-170)."""
    pipe = pipe_instance_async
    pipe.valves.API_KEY = EncryptedStr("test-api-key")

    captured_payloads: list[dict] = []
    callback = _make_callback(captured_payloads, "Generated Title")

    session = pipe._create_http_session(pipe.valves)

    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                callback=callback,
                repeat=True,
            )

            adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))
            result = await adapter._run_task_model_request(
                {"model": "openai/gpt-4o-mini", "input": "test"},
                pipe.valves,
                session=session,
                user_id="user_123",
                pipe_id="test_pipe",
            )

            assert result == "Generated Title"
    finally:
        await session.close()


# ============================================================================
# Test _run_task_model_request - Successful retry after failure
# ============================================================================


@pytest.mark.asyncio
async def test_task_model_request_success_after_retry(pipe_instance_async):
    """Test _run_task_model_request succeeds after initial failure (lines 197-199)."""
    pipe = pipe_instance_async
    pipe.valves.API_KEY = EncryptedStr("test-api-key")

    call_count = [0]

    def callback(url, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call fails
            return CallbackResult(
                body=json.dumps({"error": {"message": "Temporary error"}}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                status=500,
            )
        # Second call succeeds
        return CallbackResult(
            body=json.dumps(_make_responses_json("Success on retry")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            status=200,
        )

    session = pipe._create_http_session(pipe.valves)

    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                callback=callback,
                repeat=True,
            )

            adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))
            result = await adapter._run_task_model_request(
                {"model": "openai/gpt-4o-mini", "input": "test"},
                pipe.valves,
                session=session,
            )

            # Should succeed on retry
            assert result == "Success on retry"
            assert call_count[0] == 2
    finally:
        await session.close()


# ============================================================================
# Test edge cases in _extract_task_output_text
# ============================================================================


def test_extract_task_output_text_empty_output_list(pipe_instance):
    """Test _extract_task_output_text with empty output list."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    response = {"output": []}
    result = adapter._extract_task_output_text(response)
    assert result == ""


def test_extract_task_output_text_empty_content_list(pipe_instance):
    """Test _extract_task_output_text with empty content list."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    response = {
        "output": [
            {"type": "message", "content": []}
        ]
    }
    result = adapter._extract_task_output_text(response)
    assert result == ""


def test_extract_task_output_text_output_not_list(pipe_instance):
    """Test _extract_task_output_text when output is not a list."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    response = {"output": "string_output"}
    result = adapter._extract_task_output_text(response)
    assert result == ""


def test_extract_task_output_text_content_wrong_type(pipe_instance):
    """Test _extract_task_output_text with content items of wrong type."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "image", "url": "http://example.com/img.png"},
                    {"type": "output_text", "text": "actual text"},
                ],
            }
        ]
    }
    result = adapter._extract_task_output_text(response)
    assert result == "actual text"


def test_extract_task_output_text_empty_text(pipe_instance):
    """Test _extract_task_output_text with empty text field."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": ""},
                    {"type": "output_text", "text": None},
                    {"type": "output_text", "text": "valid"},
                ],
            }
        ]
    }
    result = adapter._extract_task_output_text(response)
    assert result == "valid"


def test_extract_task_output_text_multiple_messages(pipe_instance):
    """Test _extract_task_output_text with multiple message items."""
    pipe = pipe_instance
    adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))

    response = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "First"}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "Second"}],
            },
        ]
    }
    result = adapter._extract_task_output_text(response)
    assert "First" in result
    assert "Second" in result


# ============================================================================
# Test model ID normalization in _run_task_model_request
# ============================================================================


@pytest.mark.asyncio
async def test_task_model_request_model_id_normalization(pipe_instance_async):
    """Test _run_task_model_request normalizes model ID (line 107)."""
    pipe = pipe_instance_async
    pipe.valves.API_KEY = EncryptedStr("test-api-key")

    captured_payloads: list[dict] = []
    callback = _make_callback(captured_payloads, "Generated Title")

    session = pipe._create_http_session(pipe.valves)

    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                callback=callback,
                repeat=True,
            )

            adapter = TaskModelAdapter(pipe, logging.getLogger(__name__))
            result = await adapter._run_task_model_request(
                {"model": "openai.gpt-4o-mini", "input": "test"},  # dotted format
                pipe.valves,
                session=session,
            )

            assert result == "Generated Title"
    finally:
        await session.close()


# ============================================================================
# Test adapter initialization
# ============================================================================


def test_task_model_adapter_initialization(pipe_instance):
    """Test TaskModelAdapter initialization."""
    pipe = pipe_instance
    logger = logging.getLogger(__name__)

    adapter = TaskModelAdapter(pipe, logger)

    assert adapter._pipe is pipe
    assert adapter.logger is logger
