"""Integration tests for NonStreamingAdapter using real Pipe() instances.

These tests use real Pipe() instances and call actual adapter methods.
HTTP calls are mocked at the boundary using aioresponses.

Coverage target: Missing lines from nonstreaming_adapter.py:
- 62, 66-74: _extract_chat_message_text edge cases
- 91, 101: _run_responses output item handling
- 138, 141, 154-158, 165, 167, 173-174, 176, 179, 184: _run_chat reasoning/annotation paths
- 195, 198, 201, 204, 207-208: _run_chat tool_calls edge cases
- 276-285: fallback from responses to chat_completions
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false
from __future__ import annotations

import json
from typing import Any

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import Pipe, OpenRouterAPIError
from open_webui_openrouter_pipe.requests.nonstreaming_adapter import NonStreamingAdapter


# ============================================================================
# Adapter Initialization Tests
# ============================================================================


def test_pipe_creates_nonstreaming_adapter(pipe_instance):
    """Test that Pipe lazily creates NonStreamingAdapter."""
    pipe = pipe_instance

    # Adapter should not exist yet
    assert pipe._nonstreaming_adapter is None

    # Ensure adapter is created
    adapter = pipe._ensure_nonstreaming_adapter()

    assert adapter is not None
    assert isinstance(adapter, NonStreamingAdapter)
    assert pipe._nonstreaming_adapter is adapter

    # Second call should return same instance
    adapter2 = pipe._ensure_nonstreaming_adapter()
    assert adapter2 is adapter


# ============================================================================
# _extract_chat_message_text Tests (lines 62, 66-74)
# ============================================================================


@pytest.mark.asyncio
async def test_nonstreaming_chat_message_not_dict(pipe_instance_async):
    """Test _extract_chat_message_text with non-dict message (line 62)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Response with message that is not a dict (will be handled as empty)
    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": "not a dict",  # Invalid message type
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [{"role": "user", "content": "Hi"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have a completed event
    completed = [e for e in events if e.get("type") == "response.completed"]
    assert len(completed) == 1


@pytest.mark.asyncio
async def test_nonstreaming_chat_message_content_list(pipe_instance_async):
    """Test _extract_chat_message_text with list content blocks (lines 66-74)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Response with content as a list of text blocks
    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "World"},
                    {"type": "image", "url": "http://example.com/image.png"},  # Non-text block
                    {"type": "text"},  # Missing text field
                    {"type": "text", "text": 123},  # Non-string text
                ],
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [{"role": "user", "content": "Hi"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have text delta with "Hello World"
    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    combined_text = "".join(e.get("delta", "") for e in text_deltas)
    assert "Hello" in combined_text
    assert "World" in combined_text


# ============================================================================
# _run_responses Tests (lines 91, 101)
# ============================================================================


@pytest.mark.asyncio
async def test_nonstreaming_responses_non_dict_output_item(pipe_instance_async):
    """Test _run_responses skips non-dict items in output (line 91)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Response with non-dict items in output array
    response_json = {
        "id": "resp_123",
        "output": [
            "not a dict",  # Invalid - should be skipped
            None,  # Invalid - should be skipped
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hello!"}],
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 2},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [{"role": "user", "content": "Hi"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="responses",
        ):
            events.append(event)

        await session.close()

    # Should have text delta from the valid output item
    text_deltas = [e for e in events if e.get("type") == "response.output_text.delta"]
    assert len(text_deltas) >= 1
    assert "Hello!" in text_deltas[0].get("delta", "")


@pytest.mark.asyncio
async def test_nonstreaming_responses_special_item_types(pipe_instance_async):
    """Test _run_responses yields special item types (line 101)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Response with various special item types
    response_json = {
        "id": "resp_123",
        "output": [
            {"type": "reasoning", "id": "reasoning-1", "content": [{"type": "reasoning_text", "text": "Thinking..."}]},
            {"type": "web_search_call", "id": "search-1", "query": "weather"},
            {"type": "file_search_call", "id": "file-1", "query": "document"},
            {"type": "image_generation_call", "id": "img-1", "prompt": "cat"},
            {"type": "local_shell_call", "id": "shell-1", "command": "ls"},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done!"}],
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [{"role": "user", "content": "Search and generate"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="responses",
        ):
            events.append(event)

        await session.close()

    # Should have output_item.done events for special types
    item_done_events = [e for e in events if e.get("type") == "response.output_item.done"]
    item_types = {e.get("item", {}).get("type") for e in item_done_events}
    assert "reasoning" in item_types
    assert "web_search_call" in item_types


@pytest.mark.asyncio
async def test_nonstreaming_responses_function_call(pipe_instance_async):
    """Test _run_responses handles function_call items."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Response with function_call in output
    response_json = {
        "id": "resp_123",
        "output": [
            {
                "type": "function_call",
                "id": "call_123",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
                "status": "completed",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Called function"}],
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 5},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [], "tools": [{"type": "function", "name": "get_weather"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="responses",
        ):
            events.append(event)

        await session.close()

    # Should have function call added and done events
    item_added = [e for e in events if e.get("type") == "response.output_item.added"]
    assert any(e.get("item", {}).get("type") == "function_call" for e in item_added)


# ============================================================================
# _run_chat Reasoning Tests (lines 138, 141, 154-158)
# ============================================================================


@pytest.mark.asyncio
async def test_nonstreaming_chat_reasoning_details_non_dict_entry(pipe_instance_async):
    """Test _run_chat skips non-dict reasoning_details entries (line 138)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Answer",
                "reasoning_details": [
                    "not a dict",  # Should be skipped
                    None,  # Should be skipped
                    {"type": "reasoning.text", "text": "Thinking..."},
                ],
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/o1", "input": [{"role": "user", "content": "Think"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have reasoning delta from valid entry
    reasoning_deltas = [e for e in events if e.get("type") == "response.reasoning_text.delta"]
    assert len(reasoning_deltas) >= 1


@pytest.mark.asyncio
async def test_nonstreaming_chat_reasoning_invalid_type(pipe_instance_async):
    """Test _run_chat skips reasoning entries with invalid/empty type (line 141)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Answer",
                "reasoning_details": [
                    {"type": None, "text": "Invalid"},  # None type
                    {"type": "", "text": "Empty type"},  # Empty string type
                    {"type": 123, "text": "Not string"},  # Non-string type
                    {"type": "reasoning.text", "text": "Valid"},
                ],
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/o1", "input": [{"role": "user", "content": "Think"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should only have one reasoning delta from the valid entry
    reasoning_deltas = [e for e in events if e.get("type") == "response.reasoning_text.delta"]
    assert len(reasoning_deltas) == 1
    assert "Valid" in reasoning_deltas[0].get("delta", "")


@pytest.mark.asyncio
async def test_nonstreaming_chat_reasoning_summary(pipe_instance_async):
    """Test _run_chat handles reasoning.summary type (lines 154-158)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "42",
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "Let me calculate...", "id": "r1"},
                    {"type": "reasoning.summary", "summary": "  The answer is derived from deep thought.  "},
                ],
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/o1", "input": [{"role": "user", "content": "What is 6 * 7?"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have reasoning summary done event
    summary_events = [e for e in events if e.get("type") == "response.reasoning_summary_text.done"]
    assert len(summary_events) == 1
    assert "deep thought" in summary_events[0].get("text", "")


@pytest.mark.asyncio
async def test_nonstreaming_chat_reasoning_field(pipe_instance_async):
    """Test _run_chat emits reasoning events for plain reasoning field."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Answer",
                "reasoning": "Thinking through the steps.",
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/o1", "input": [{"role": "user", "content": "Explain"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    reasoning_deltas = [e for e in events if e.get("type") == "response.reasoning_text.delta"]
    assert len(reasoning_deltas) == 1
    assert "Thinking through" in reasoning_deltas[0].get("delta", "")

    reasoning_done = [
        e for e in events
        if e.get("type") == "response.output_item.done" and e.get("item", {}).get("type") == "reasoning"
    ]
    assert len(reasoning_done) == 1


@pytest.mark.asyncio
async def test_nonstreaming_chat_message_images(pipe_instance_async):
    """Test _run_chat emits image output items for message images."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Image attached",
                "images": [{"image_url": {"url": "https://example.com/image.png"}}],
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [{"role": "user", "content": "Show image"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    image_done = [
        e for e in events
        if e.get("type") == "response.output_item.done" and e.get("item", {}).get("type") == "image_generation_call"
    ]
    assert len(image_done) == 1
    result = image_done[0].get("item", {}).get("result", [])
    assert isinstance(result, list) and result
    assert result[0].get("image_url", {}).get("url") == "https://example.com/image.png"


# ============================================================================
# _run_chat Annotations Tests (lines 165, 167, 173-174, 176, 179, 184)
# ============================================================================


@pytest.mark.asyncio
async def test_nonstreaming_chat_annotations_non_dict_skipped(pipe_instance_async):
    """Test _run_chat skips non-dict annotations (line 165)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Based on sources",
                "annotations": [
                    "not a dict",  # Should be skipped
                    {"type": "url_citation", "url_citation": {"url": "https://example.com", "title": "Example"}},
                ],
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [{"role": "user", "content": "Search"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have one annotation event from valid entry
    annotation_events = [e for e in events if e.get("type") == "response.output_text.annotation.added"]
    assert len(annotation_events) == 1


@pytest.mark.asyncio
async def test_nonstreaming_chat_annotations_non_url_citation_skipped(pipe_instance_async):
    """Test _run_chat skips non url_citation annotations (line 167)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Based on sources",
                "annotations": [
                    {"type": "file_citation", "file_id": "file-123"},  # Not url_citation
                    {"type": "url_citation", "url_citation": {"url": "https://example.com", "title": "Valid"}},
                ],
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [{"role": "user", "content": "Search"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have only one annotation event (the url_citation)
    annotation_events = [e for e in events if e.get("type") == "response.output_text.annotation.added"]
    assert len(annotation_events) == 1
    assert annotation_events[0]["annotation"]["url"] == "https://example.com"


@pytest.mark.asyncio
async def test_nonstreaming_chat_annotations_flat_format(pipe_instance_async):
    """Test _run_chat handles flat annotation format (lines 173-174)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Some providers may return annotations in flat format (url/title at top level)
    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Based on sources",
                "annotations": [
                    {"type": "url_citation", "url": "https://flat.example.com", "title": "Flat Format"},
                ],
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [{"role": "user", "content": "Search"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have annotation from flat format
    annotation_events = [e for e in events if e.get("type") == "response.output_text.annotation.added"]
    assert len(annotation_events) == 1
    assert annotation_events[0]["annotation"]["url"] == "https://flat.example.com"


@pytest.mark.asyncio
async def test_nonstreaming_chat_annotations_empty_url_skipped(pipe_instance_async):
    """Test _run_chat skips annotations with empty/whitespace URL (line 176)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Based on sources",
                "annotations": [
                    {"type": "url_citation", "url_citation": {"url": "", "title": "Empty URL"}},
                    {"type": "url_citation", "url_citation": {"url": "   ", "title": "Whitespace URL"}},
                    {"type": "url_citation", "url_citation": {"url": None, "title": "None URL"}},
                    {"type": "url_citation", "url_citation": {"url": "https://valid.com", "title": "Valid"}},
                ],
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [{"role": "user", "content": "Search"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have only one valid annotation
    annotation_events = [e for e in events if e.get("type") == "response.output_text.annotation.added"]
    assert len(annotation_events) == 1
    assert annotation_events[0]["annotation"]["url"] == "https://valid.com"


@pytest.mark.asyncio
async def test_nonstreaming_chat_annotations_duplicate_urls_skipped(pipe_instance_async):
    """Test _run_chat skips duplicate annotation URLs (line 179)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Based on sources",
                "annotations": [
                    {"type": "url_citation", "url_citation": {"url": "https://dupe.com", "title": "First"}},
                    {"type": "url_citation", "url_citation": {"url": "https://dupe.com", "title": "Second"}},
                    {"type": "url_citation", "url_citation": {"url": "https://unique.com", "title": "Unique"}},
                ],
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [{"role": "user", "content": "Search"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have only two annotations (duplicate URL deduped)
    annotation_events = [e for e in events if e.get("type") == "response.output_text.annotation.added"]
    assert len(annotation_events) == 2
    urls = {e["annotation"]["url"] for e in annotation_events}
    assert "https://dupe.com" in urls
    assert "https://unique.com" in urls


@pytest.mark.asyncio
async def test_nonstreaming_chat_annotations_non_string_title(pipe_instance_async):
    """Test _run_chat handles non-string title (falls back to URL) (line 184)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Based on sources",
                "annotations": [
                    {"type": "url_citation", "url_citation": {"url": "https://notitle.com", "title": 12345}},  # Non-string title
                    {"type": "url_citation", "url_citation": {"url": "https://empty.com", "title": "   "}},  # Whitespace title
                ],
            },
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [{"role": "user", "content": "Search"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have annotations with URL as fallback title
    annotation_events = [e for e in events if e.get("type") == "response.output_text.annotation.added"]
    assert len(annotation_events) == 2
    # First should use URL as title since title was non-string
    first = [e for e in annotation_events if e["annotation"]["url"] == "https://notitle.com"][0]
    assert first["annotation"]["title"] == "https://notitle.com"


# ============================================================================
# _run_chat Tool Calls Tests (lines 195, 198, 201, 204, 207-208)
# ============================================================================


@pytest.mark.asyncio
async def test_nonstreaming_chat_tool_calls_non_dict_skipped(pipe_instance_async):
    """Test _run_chat skips non-dict tool calls (line 195)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    "not a dict",  # Should be skipped
                    {
                        "id": "call_valid",
                        "type": "function",
                        "function": {"name": "valid_tool", "arguments": "{}"},
                    },
                ],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [], "tools": [{"type": "function", "name": "valid_tool"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have only one function call from valid entry
    func_done = [e for e in events if e.get("type") == "response.output_item.done" and e.get("item", {}).get("type") == "function_call"]
    assert len(func_done) == 1
    assert func_done[0]["item"]["name"] == "valid_tool"


@pytest.mark.asyncio
async def test_nonstreaming_chat_tool_calls_non_dict_function_skipped(pipe_instance_async):
    """Test _run_chat skips tool calls with non-dict function (line 198)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_bad", "type": "function", "function": "not a dict"},
                    {"id": "call_good", "type": "function", "function": {"name": "good_tool", "arguments": "{}"}},
                ],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [], "tools": [{"type": "function", "name": "good_tool"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have only one function call from valid entry
    func_done = [e for e in events if e.get("type") == "response.output_item.done" and e.get("item", {}).get("type") == "function_call"]
    assert len(func_done) == 1
    assert func_done[0]["item"]["name"] == "good_tool"


@pytest.mark.asyncio
async def test_nonstreaming_chat_tool_calls_invalid_name_skipped(pipe_instance_async):
    """Test _run_chat skips tool calls with invalid/empty name (line 201)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "", "arguments": "{}"}},
                    {"id": "call_2", "type": "function", "function": {"name": None, "arguments": "{}"}},
                    {"id": "call_3", "type": "function", "function": {"name": 123, "arguments": "{}"}},
                    {"id": "call_4", "type": "function", "function": {"name": "valid", "arguments": "{}"}},
                ],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [], "tools": [{"type": "function", "name": "valid"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have only one function call from valid entry
    func_done = [e for e in events if e.get("type") == "response.output_item.done" and e.get("item", {}).get("type") == "function_call"]
    assert len(func_done) == 1
    assert func_done[0]["item"]["name"] == "valid"


@pytest.mark.asyncio
async def test_nonstreaming_chat_tool_calls_non_string_arguments(pipe_instance_async):
    """Test _run_chat serializes non-string arguments (line 204)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_dict", "type": "function", "function": {"name": "tool_a", "arguments": {"city": "NYC"}}},  # Dict instead of string
                    {"id": "call_none", "type": "function", "function": {"name": "tool_b", "arguments": None}},  # None arguments
                ],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [], "tools": [{"type": "function", "name": "tool_a"}, {"type": "function", "name": "tool_b"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have two function calls with serialized arguments
    func_done = [e for e in events if e.get("type") == "response.output_item.done" and e.get("item", {}).get("type") == "function_call"]
    assert len(func_done) == 2

    tool_a = [f for f in func_done if f["item"]["name"] == "tool_a"][0]
    assert json.loads(tool_a["item"]["arguments"]) == {"city": "NYC"}

    tool_b = [f for f in func_done if f["item"]["name"] == "tool_b"][0]
    assert tool_b["item"]["arguments"] == "{}"


@pytest.mark.asyncio
async def test_nonstreaming_chat_tool_calls_missing_id(pipe_instance_async):
    """Test _run_chat generates call_id when missing (lines 207-208)."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-123",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"type": "function", "function": {"name": "tool_no_id", "arguments": "{}"}},  # Missing id
                    {"id": "", "type": "function", "function": {"name": "tool_empty_id", "arguments": "{}"}},  # Empty id
                    {"id": "   ", "type": "function", "function": {"name": "tool_whitespace_id", "arguments": "{}"}},  # Whitespace id
                ],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [], "tools": [{"type": "function", "name": "tool_no_id"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # All three should have generated call_ids
    func_done = [e for e in events if e.get("type") == "response.output_item.done" and e.get("item", {}).get("type") == "function_call"]
    assert len(func_done) == 3

    for fd in func_done:
        call_id = fd["item"]["call_id"]
        assert call_id is not None
        assert len(call_id) > 0
        assert "toolcall-" in call_id


# ============================================================================
# Fallback Tests (lines 276-285)
# ============================================================================


@pytest.mark.asyncio
async def test_nonstreaming_fallback_from_responses_to_chat(pipe_instance_async):
    """Test fallback from /responses to /chat/completions (lines 276-285)."""
    pipe = pipe_instance_async
    # Enable auto fallback
    valves = pipe.valves.model_copy(update={"AUTO_FALLBACK_CHAT_COMPLETIONS": True})
    session = pipe._create_http_session(valves)

    # Error response from /responses that triggers fallback
    responses_error = {
        "error": {
            "message": "The model `some/model` does not support the /responses endpoint",
            "type": "invalid_request_error",
            "code": "unsupported_model",
        }
    }

    # Success response from /chat/completions
    chat_success = {
        "id": "chatcmpl-fallback",
        "choices": [{
            "message": {"role": "assistant", "content": "Fallback worked!"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
    }

    with aioresponses() as mock_http:
        # First call to /responses fails
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=responses_error,
            status=400,
        )
        # Fallback to /chat/completions succeeds
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=chat_success,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "some/model", "input": [{"role": "user", "content": "Test"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            # Don't override endpoint - let it try responses first
        ):
            events.append(event)

        await session.close()

    # Should have completed event from fallback
    completed = [e for e in events if e.get("type") == "response.completed"]
    assert len(completed) == 1


@pytest.mark.asyncio
async def test_nonstreaming_no_fallback_when_disabled(pipe_instance_async):
    """Test no fallback when AUTO_FALLBACK_CHAT_COMPLETIONS is disabled."""
    pipe = pipe_instance_async
    # Disable auto fallback
    valves = pipe.valves.model_copy(update={"AUTO_FALLBACK_CHAT_COMPLETIONS": False})
    session = pipe._create_http_session(valves)

    # Error response from /responses
    responses_error = {
        "error": {
            "message": "The model does not support /responses",
            "type": "invalid_request_error",
            "code": "unsupported_model",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=responses_error,
            status=400,
        )

        with pytest.raises(OpenRouterAPIError):
            async for _ in pipe.send_openrouter_nonstreaming_request_as_events(
                session,
                {"model": "some/model", "input": [{"role": "user", "content": "Test"}]},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            ):
                pass

        await session.close()


@pytest.mark.asyncio
async def test_nonstreaming_no_fallback_for_non_responses_error(pipe_instance_async):
    """Test no fallback for errors that don't indicate unsupported responses."""
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(update={"AUTO_FALLBACK_CHAT_COMPLETIONS": True})
    session = pipe._create_http_session(valves)

    # Rate limit error (should not trigger fallback)
    responses_error = {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
        }
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=responses_error,
            status=429,
        )

        with pytest.raises(OpenRouterAPIError) as exc_info:
            async for _ in pipe.send_openrouter_nonstreaming_request_as_events(
                session,
                {"model": "some/model", "input": [{"role": "user", "content": "Test"}]},
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            ):
                pass

        await session.close()

    assert exc_info.value.status == 429


# ============================================================================
# Integration Tests
# ============================================================================


@pytest.mark.asyncio
async def test_nonstreaming_complete_chat_workflow(pipe_instance_async):
    """Test complete non-streaming workflow with all features."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "chatcmpl-complete",
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Here is your answer with sources.",
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "Let me think about this...", "id": "r1"},
                    {"type": "reasoning.summary", "summary": "Deep analysis completed."},
                ],
                "annotations": [
                    {"type": "url_citation", "url_citation": {"url": "https://source1.com", "title": "Source 1"}},
                    {"type": "url_citation", "url_citation": {"url": "https://source2.com", "title": "Source 2"}},
                ],
                "tool_calls": [
                    {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": '{"query": "test"}'}},
                ],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
            "prompt_tokens_details": {"cached_tokens": 20},
            "completion_tokens_details": {"reasoning_tokens": 10},
        },
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/chat/completions",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/o1", "input": [{"role": "user", "content": "Think and search"}], "tools": [{"type": "function", "name": "search"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="chat_completions",
        ):
            events.append(event)

        await session.close()

    # Should have all event types
    event_types = {e.get("type") for e in events}
    assert "response.output_item.added" in event_types  # Reasoning added
    assert "response.reasoning_text.delta" in event_types
    assert "response.reasoning_summary_text.done" in event_types
    assert "response.output_text.annotation.added" in event_types
    assert "response.output_item.done" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.completed" in event_types

    # Check completed event has proper usage
    completed = [e for e in events if e.get("type") == "response.completed"][0]
    usage = completed["response"]["usage"]
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50


@pytest.mark.asyncio
async def test_nonstreaming_responses_complete_workflow(pipe_instance_async):
    """Test complete non-streaming workflow via /responses endpoint."""
    pipe = pipe_instance_async
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    response_json = {
        "id": "resp_complete",
        "output": [
            {
                "type": "reasoning",
                "id": "reasoning-1",
                "status": "completed",
                "content": [{"type": "reasoning_text", "text": "Reasoning content"}],
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Final answer!"}],
            },
            {
                "type": "function_call",
                "id": "call_resp",
                "call_id": "call_resp",
                "name": "get_data",
                "arguments": '{"key": "value"}',
                "status": "completed",
            },
        ],
        "usage": {"input_tokens": 50, "output_tokens": 25},
    }

    with aioresponses() as mock_http:
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload=response_json,
        )

        events = []
        async for event in pipe.send_openrouter_nonstreaming_request_as_events(
            session,
            {"model": "openai/gpt-4o", "input": [{"role": "user", "content": "Process this"}]},
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            valves=valves,
            endpoint_override="responses",
        ):
            events.append(event)

        await session.close()

    # Should have all expected events
    event_types = {e.get("type") for e in events}
    assert "response.output_item.done" in event_types
    assert "response.output_text.delta" in event_types
    assert "response.completed" in event_types

    # Check for reasoning and function call in output_item.done events
    item_done = [e for e in events if e.get("type") == "response.output_item.done"]
    item_types = {e.get("item", {}).get("type") for e in item_done}
    assert "reasoning" in item_types
    assert "function_call" in item_types
