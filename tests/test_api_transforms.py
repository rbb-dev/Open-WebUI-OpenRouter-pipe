"""Comprehensive tests for open_webui_openrouter_pipe/api/transforms.py

This consolidated test module provides complete coverage for the transforms
module, including both unit tests for isolated functions and integration tests
through the Pipe class.
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false
from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock, AsyncMock

import pytest
from pydantic import ValidationError
from aioresponses import aioresponses, CallbackResult

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.api.transforms import (
    CompletionsBody,
    ResponsesBody,
    _filter_openrouter_request,
    _filter_openrouter_chat_request,
    _responses_payload_to_chat_completions_payload,
    _responses_tools_to_chat_tools,
    _chat_tools_to_responses_tools,
    _responses_tool_choice_to_chat_tool_choice,
    _responses_input_to_chat_messages,
    _chat_response_format_to_responses_text_format,
    _responses_text_format_to_chat_response_format,
    _normalise_openrouter_responses_text_format,
    _model_params_to_dict,
    _get_disable_param,
    _sanitize_openrouter_metadata,
    _apply_model_fallback_to_payload,
    _apply_disable_native_websearch_to_payload,
    _apply_identifier_valves_to_payload,
    _strip_disable_model_settings_params,
    _filter_replayable_input_items,
)


def _sse(obj: dict[str, Any]) -> str:
    """Format object as SSE data line."""
    return f"data: {json.dumps(obj)}\n\n"


# ============================================================================
# _filter_openrouter_request Tests
# ============================================================================


class TestFilterOpenrouterRequest:
    """Tests for _filter_openrouter_request()."""

    def test_removes_unknown_keys(self):
        """Test that unknown keys are filtered from request."""
        payload = {
            "model": "openai/gpt-4o",
            "input": [],
            "unknown_key": "should be removed",
            "another_unknown": 123,
        }
        result = _filter_openrouter_request(payload)

        assert "model" in result
        assert "input" in result
        # Unknown keys should be filtered
        assert "unknown_key" not in result
        assert "another_unknown" not in result

    def test_preserves_valid_keys(self):
        """Test that valid Responses API keys are preserved."""
        payload = {
            "model": "openai/gpt-4o",
            "input": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "max_output_tokens": 100,
            "stream": True,
        }
        result = _filter_openrouter_request(payload)

        assert result["model"] == "openai/gpt-4o"
        assert result["temperature"] == 0.7
        assert result["max_output_tokens"] == 100
        assert result["stream"] is True

    def test_handles_empty_payload(self):
        """Test filtering with empty payload."""
        result = _filter_openrouter_request({})
        assert result == {}

    def test_handles_non_dict(self):
        """Test filtering with non-dict input raises ValueError."""
        # The function tries to call dict() on input, which fails for strings
        with pytest.raises(ValueError):
            _filter_openrouter_request("not a dict")  # type: ignore[arg-type]

    def test_null_values_dropped(self):
        """Test explicit null values are dropped."""
        payload = {"model": "gpt-4", "temperature": None, "input": []}
        result = _filter_openrouter_request(payload)
        assert "temperature" not in result

    def test_top_k_int_converted_to_float(self):
        """Test top_k integer is converted to float."""
        payload = {"model": "gpt-4", "top_k": 10, "input": []}
        result = _filter_openrouter_request(payload)
        assert result["top_k"] == 10.0
        assert isinstance(result["top_k"], float)

    def test_top_k_string_converted(self):
        """Test top_k string is converted to float."""
        payload = {"model": "gpt-4", "top_k": "5", "input": []}
        result = _filter_openrouter_request(payload)
        assert result["top_k"] == 5.0

    def test_top_k_empty_string_dropped(self):
        """Test top_k empty string is dropped."""
        payload = {"model": "gpt-4", "top_k": "  ", "input": []}
        result = _filter_openrouter_request(payload)
        assert "top_k" not in result

    def test_top_k_invalid_string_dropped(self):
        """Test top_k invalid string is dropped."""
        payload = {"model": "gpt-4", "top_k": "invalid", "input": []}
        result = _filter_openrouter_request(payload)
        assert "top_k" not in result

    def test_metadata_sanitized(self):
        """Test metadata is sanitized."""
        payload = {
            "model": "gpt-4",
            "input": [],
            "metadata": {"valid": "value", "key[0]": "invalid"},
        }
        result = _filter_openrouter_request(payload)
        assert result["metadata"] == {"valid": "value"}

    def test_metadata_invalid_dropped(self):
        """Test invalid metadata is dropped entirely."""
        payload = {"model": "gpt-4", "input": [], "metadata": "not a dict"}
        result = _filter_openrouter_request(payload)
        assert "metadata" not in result

    def test_reasoning_filtered(self):
        """Test reasoning dict is filtered to allowed fields."""
        payload = {
            "model": "gpt-4",
            "input": [],
            "reasoning": {
                "effort": "high",
                "max_tokens": 1000,
                "exclude": True,
                "enabled": True,
                "summary": "thinking",
                "unknown": "dropped",
            },
        }
        result = _filter_openrouter_request(payload)
        assert "effort" in result["reasoning"]
        assert "max_tokens" in result["reasoning"]
        assert "unknown" not in result["reasoning"]

    def test_reasoning_non_dict_dropped(self):
        """Test non-dict reasoning is dropped."""
        payload = {"model": "gpt-4", "input": [], "reasoning": "not a dict"}
        result = _filter_openrouter_request(payload)
        assert "reasoning" not in result

    def test_reasoning_empty_dropped(self):
        """Test empty reasoning dict is dropped."""
        payload = {"model": "gpt-4", "input": [], "reasoning": {"unknown": "only"}}
        result = _filter_openrouter_request(payload)
        assert "reasoning" not in result

    def test_text_non_dict_dropped(self):
        """Test non-dict text is dropped."""
        payload = {"model": "gpt-4", "input": [], "text": "not a dict"}
        result = _filter_openrouter_request(payload)
        assert "text" not in result

    def test_text_empty_dropped(self):
        """Test empty text dict is dropped."""
        payload = {"model": "gpt-4", "input": [], "text": {}}
        result = _filter_openrouter_request(payload)
        assert "text" not in result


# ============================================================================
# _filter_openrouter_chat_request Tests
# ============================================================================


class TestFilterOpenrouterChatRequest:
    """Tests for _filter_openrouter_chat_request()."""

    def test_removes_responses_only_keys(self):
        """Test that Responses-only keys are filtered from chat requests."""
        payload = {
            "model": "openai/gpt-4o",
            "messages": [],
            "input": [],  # Responses-only
            "instructions": "System prompt",  # Responses-only
        }
        result = _filter_openrouter_chat_request(payload)

        assert "model" in result
        assert "messages" in result
        # Responses-only keys should be removed
        assert "input" not in result
        assert "instructions" not in result

    def test_preserves_chat_keys(self):
        """Test that Chat Completions keys are preserved."""
        payload = {
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": True,
        }
        result = _filter_openrouter_chat_request(payload)

        assert result["model"] == "openai/gpt-4o"
        assert len(result["messages"]) == 1
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100

    def test_non_dict_returns_empty(self):
        """Test non-dict input returns empty dict."""
        assert _filter_openrouter_chat_request("not a dict") == {}  # type: ignore[arg-type]
        assert _filter_openrouter_chat_request(None) == {}  # type: ignore[arg-type]
        assert _filter_openrouter_chat_request(123) == {}  # type: ignore[arg-type]

    def test_preserves_allowed_fields(self):
        """Test allowed fields are preserved."""
        payload = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": True,
            "tools": [],
            "tool_choice": "auto",
        }
        result = _filter_openrouter_chat_request(payload)
        assert result["model"] == "gpt-4"
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 100

    def test_removes_unknown_fields(self):
        """Test unknown fields are removed."""
        payload = {
            "model": "gpt-4",
            "unknown_field": "value",
            "another_unknown": 123,
        }
        result = _filter_openrouter_chat_request(payload)
        assert "unknown_field" not in result
        assert "another_unknown" not in result


# ============================================================================
# _responses_payload_to_chat_completions_payload Tests
# ============================================================================


class TestResponsesPayloadToChatCompletionsPayload:
    """Tests for _responses_payload_to_chat_completions_payload()."""

    def test_converts_input_to_messages(self):
        """Test that input array is converted to messages."""
        payload = {
            "model": "openai/gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
            ],
        }
        result = _responses_payload_to_chat_completions_payload(payload)

        assert "messages" in result
        assert result["messages"][0]["role"] == "user"
        # Content blocks (input_text) become structured blocks (type: text)
        content = result["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hello"

    def test_converts_instructions_to_system(self):
        """Test that instructions become system message."""
        payload = {
            "model": "openai/gpt-4o",
            "instructions": "You are a helpful assistant",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
            ],
        }
        result = _responses_payload_to_chat_completions_payload(payload)

        # Instructions should be first message as system
        assert result["messages"][0]["role"] == "system"
        assert "helpful assistant" in str(result["messages"][0]["content"])

    def test_converts_max_output_tokens(self):
        """Test that max_output_tokens becomes max_tokens."""
        payload = {
            "model": "openai/gpt-4o",
            "max_output_tokens": 500,
            "input": [],
        }
        result = _responses_payload_to_chat_completions_payload(payload)

        assert result["max_tokens"] == 500
        assert "max_output_tokens" not in result

    def test_converts_tools(self):
        """Test that Responses tools are converted to Chat tools format."""
        payload = {
            "model": "openai/gpt-4o",
            "input": [],
            "tools": [
                {
                    "type": "function",
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
        result = _responses_payload_to_chat_completions_payload(payload)

        assert "tools" in result
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "search"

    def test_stream_options_not_injected(self):
        """Test that stream_options are not injected when absent."""
        payload = {
            "model": "openai/gpt-4o",
            "stream": True,
            "input": [],
        }
        result = _responses_payload_to_chat_completions_payload(payload)

        assert "stream_options" not in result

    def test_rounds_top_k(self):
        """Test that top_k is rounded to integer."""
        payload = {
            "model": "openai/gpt-4o",
            "top_k": 2.7,
            "input": [],
        }
        result = _responses_payload_to_chat_completions_payload(payload)

        # Python round(2.7) = 3 (standard rounding)
        assert result["top_k"] == 3

    def test_preserves_cache_control(self):
        """Test that cache_control is preserved in content blocks."""
        payload = {
            "model": "anthropic/claude-sonnet-4.5",
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Large text",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        }
        result = _responses_payload_to_chat_completions_payload(payload)

        # Find the text content block
        content = result["messages"][0]["content"]
        if isinstance(content, list):
            text_block = next((c for c in content if c.get("type") == "text"), None)
            assert text_block is not None
            assert text_block.get("cache_control") == {"type": "ephemeral"}

    def test_non_dict_returns_empty(self):
        """Test non-dict input returns empty dict."""
        assert _responses_payload_to_chat_completions_payload("not a dict") == {}  # type: ignore[arg-type]
        assert _responses_payload_to_chat_completions_payload(None) == {}  # type: ignore[arg-type]

    def test_non_streaming_no_stream_options(self):
        """Test non-streaming request doesn't get stream_options."""
        payload = {"model": "gpt-4", "input": [], "stream": False}
        result = _responses_payload_to_chat_completions_payload(payload)
        assert result["stream"] is False
        assert "stream_options" not in result

    def test_existing_stream_options_preserved(self):
        """Test existing stream_options are preserved."""
        payload = {
            "model": "gpt-4",
            "input": [],
            "stream": True,
            "stream_options": {"custom": True},
        }
        result = _responses_payload_to_chat_completions_payload(payload)
        assert result["stream_options"]["custom"] is True

    def test_usage_not_forwarded(self):
        """Test usage is not forwarded to chat payload."""
        payload = {"model": "gpt-4", "input": [], "usage": {"custom": "value"}}
        result = _responses_payload_to_chat_completions_payload(payload)
        assert "usage" not in result

    def test_int_params_rounded(self):
        """Test integer parameters are rounded."""
        payload = {"model": "gpt-4", "input": [], "top_k": 2.7, "seed": 10.3}
        result = _responses_payload_to_chat_completions_payload(payload)
        assert result["top_k"] == 3
        assert result["seed"] == 10

    def test_int_param_string_converted(self):
        """Test string integer params are converted."""
        payload = {"model": "gpt-4", "input": [], "top_k": "5"}
        result = _responses_payload_to_chat_completions_payload(payload)
        assert result["top_k"] == 5

    def test_int_param_empty_string_removed(self):
        """Test empty string integer params are removed."""
        payload = {"model": "gpt-4", "input": [], "top_k": "  "}
        result = _responses_payload_to_chat_completions_payload(payload)
        assert "top_k" not in result

    def test_invalid_response_format_removed(self, caplog):
        """Test invalid response_format is removed with warning."""
        payload = {"model": "gpt-4", "input": [], "response_format": {"type": "invalid"}}
        with caplog.at_level(logging.WARNING):
            result = _responses_payload_to_chat_completions_payload(payload)
        assert "response_format" not in result

    def test_text_format_mapped_to_response_format(self):
        """Test text.format is mapped to response_format."""
        payload = {
            "model": "gpt-4",
            "input": [],
            "text": {"format": {"type": "json_object"}},
        }
        result = _responses_payload_to_chat_completions_payload(payload)
        assert result["response_format"] == {"type": "json_object"}

    def test_text_verbosity_preserved(self):
        """Test text.verbosity is mapped to verbosity."""
        payload = {
            "model": "gpt-4",
            "input": [],
            "text": {"verbosity": "verbose"},
        }
        result = _responses_payload_to_chat_completions_payload(payload)
        assert result["verbosity"] == "verbose"

    def test_instructions_prepended_to_existing_system(self):
        """Test instructions prepended to existing system message."""
        payload = {
            "model": "gpt-4",
            "instructions": "Be helpful",
            "input": [{"type": "message", "role": "system", "content": "Existing prompt"}],
        }
        result = _responses_payload_to_chat_completions_payload(payload)
        # Instructions should be prepended
        assert "Be helpful" in result["messages"][0]["content"]
        assert "Existing prompt" in result["messages"][0]["content"]

    def test_instructions_prepended_to_list_content(self):
        """Test instructions prepended when system has list content."""
        payload = {
            "model": "gpt-4",
            "instructions": "Be helpful",
            "input": [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Existing"}],
                }
            ],
        }
        result = _responses_payload_to_chat_completions_payload(payload)
        # Instructions should be prepended as text block
        content = result["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["text"] == "Be helpful"

    def test_instructions_with_no_messages(self):
        """Test instructions create system message when no messages."""
        payload = {"model": "gpt-4", "instructions": "Be helpful", "input": []}
        result = _responses_payload_to_chat_completions_payload(payload)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "Be helpful"

    def test_instructions_with_non_system_first_message(self):
        """Test instructions create system message when first message is not system."""
        payload = {
            "model": "gpt-4",
            "instructions": "Be helpful",
            "input": [{"type": "message", "role": "user", "content": "Hi"}],
        }
        result = _responses_payload_to_chat_completions_payload(payload)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "Be helpful"
        assert result["messages"][1]["role"] == "user"


# ============================================================================
# Tool Conversion Tests
# ============================================================================


class TestResponsesToolsToChatTools:
    """Tests for _responses_tools_to_chat_tools()."""

    def test_converts_format(self):
        """Test Responses tools are wrapped in function format."""
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        result = _responses_tools_to_chat_tools(tools)

        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "get_weather"
        assert result[0]["function"]["description"] == "Get weather"

    def test_skips_wrapped_format(self):
        """Test tools in Chat wrapped format are skipped (need name at top level)."""
        # This is Chat Completions format (wrapped with 'function' key)
        # The function expects Responses format (flat with name at top level)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {},
                },
            }
        ]
        result = _responses_tools_to_chat_tools(tools)

        # Wrapped format lacks top-level 'name', so it's skipped
        assert result == []

    def test_handles_empty(self):
        """Test empty tools list."""
        assert _responses_tools_to_chat_tools([]) == []
        assert _responses_tools_to_chat_tools(None) == []

    def test_non_list_returns_empty(self):
        """Test non-list input returns empty list."""
        assert _responses_tools_to_chat_tools("not a list") == []
        assert _responses_tools_to_chat_tools(123) == []
        assert _responses_tools_to_chat_tools({"type": "function"}) == []

    def test_non_dict_items_skipped(self):
        """Test non-dict items in list are skipped."""
        tools = [
            "not a dict",
            {"type": "function", "name": "valid_tool"},
            123,
        ]
        result = _responses_tools_to_chat_tools(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "valid_tool"

    def test_non_function_type_skipped(self):
        """Test items without type='function' are skipped."""
        tools = [
            {"type": "other", "name": "tool1"},
            {"type": "function", "name": "tool2"},
            {"name": "tool3"},  # no type
        ]
        result = _responses_tools_to_chat_tools(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "tool2"

    def test_whitespace_name_skipped(self):
        """Test tools with whitespace-only names are skipped."""
        tools = [
            {"type": "function", "name": "   "},
            {"type": "function", "name": "valid"},
        ]
        result = _responses_tools_to_chat_tools(tools)
        assert len(result) == 1
        assert result[0]["function"]["name"] == "valid"

    def test_description_preserved(self):
        """Test description is preserved."""
        tools = [{"type": "function", "name": "tool", "description": "A tool"}]
        result = _responses_tools_to_chat_tools(tools)
        assert result[0]["function"]["description"] == "A tool"

    def test_parameters_preserved(self):
        """Test parameters are preserved."""
        tools = [
            {"type": "function", "name": "tool", "parameters": {"type": "object"}}
        ]
        result = _responses_tools_to_chat_tools(tools)
        assert result[0]["function"]["parameters"] == {"type": "object"}


class TestChatToolsToResponsesTools:
    """Tests for _chat_tools_to_responses_tools()."""

    def test_unwraps(self):
        """Test Chat tools are unwrapped to Responses format."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {},
                },
            }
        ]
        result = _chat_tools_to_responses_tools(tools)

        assert result[0]["type"] == "function"
        assert result[0]["name"] == "get_weather"
        assert "function" not in result[0]

    def test_handles_already_flat(self):
        """Test tools already flat are preserved."""
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {},
            }
        ]
        result = _chat_tools_to_responses_tools(tools)

        assert result[0]["name"] == "get_weather"

    def test_non_list_returns_empty(self):
        """Test non-list input returns empty list."""
        assert _chat_tools_to_responses_tools("not a list") == []
        assert _chat_tools_to_responses_tools(123) == []

    def test_non_dict_items_skipped(self):
        """Test non-dict items are skipped."""
        tools = ["not a dict", 123]
        result = _chat_tools_to_responses_tools(tools)
        assert result == []

    def test_non_function_type_skipped(self):
        """Test items without type='function' are skipped."""
        tools = [{"type": "other", "function": {"name": "tool"}}]
        result = _chat_tools_to_responses_tools(tools)
        assert result == []

    def test_name_from_function_block(self):
        """Test name extracted from function block."""
        tools = [{"type": "function", "function": {"name": "my_tool"}}]
        result = _chat_tools_to_responses_tools(tools)
        assert result[0]["name"] == "my_tool"

    def test_description_from_function_block(self):
        """Test description extracted from function block when not at top level."""
        tools = [
            {
                "type": "function",
                "function": {"name": "tool", "description": "From function"},
            }
        ]
        result = _chat_tools_to_responses_tools(tools)
        assert result[0]["description"] == "From function"

    def test_top_level_description_preferred(self):
        """Test top-level description is preferred."""
        tools = [
            {
                "type": "function",
                "name": "tool",
                "description": "Top level",
                "function": {"name": "tool", "description": "From function"},
            }
        ]
        result = _chat_tools_to_responses_tools(tools)
        assert result[0]["description"] == "Top level"

    def test_parameters_from_function_block(self):
        """Test parameters extracted from function block."""
        tools = [
            {
                "type": "function",
                "function": {"name": "tool", "parameters": {"type": "object"}},
            }
        ]
        result = _chat_tools_to_responses_tools(tools)
        assert result[0]["parameters"] == {"type": "object"}

    def test_empty_description_not_included(self):
        """Test empty description is not included."""
        tools = [{"type": "function", "name": "tool", "description": "   "}]
        result = _chat_tools_to_responses_tools(tools)
        assert "description" not in result[0]


# ============================================================================
# Tool Choice Conversion Tests
# ============================================================================


class TestResponsesToolChoiceToChatToolChoice:
    """Tests for _responses_tool_choice_to_chat_tool_choice()."""

    def test_string_values(self):
        """Test string tool_choice values are preserved."""
        assert _responses_tool_choice_to_chat_tool_choice("auto") == "auto"
        assert _responses_tool_choice_to_chat_tool_choice("none") == "none"
        assert _responses_tool_choice_to_chat_tool_choice("required") == "required"

    def test_dict_converted(self):
        """Test dict tool_choice is converted to Chat format."""
        value = {"type": "function", "name": "get_weather"}
        result = _responses_tool_choice_to_chat_tool_choice(value)

        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"

    def test_none_returns_none(self):
        """Test None tool_choice returns None."""
        assert _responses_tool_choice_to_chat_tool_choice(None) is None

    def test_non_dict_non_string_returned_as_is(self):
        """Test non-dict, non-string values are returned as-is."""
        assert _responses_tool_choice_to_chat_tool_choice(123) == 123
        assert _responses_tool_choice_to_chat_tool_choice([1, 2]) == [1, 2]

    def test_dict_without_function_type(self):
        """Test dict without type='function' is returned as-is."""
        value = {"type": "other", "name": "tool"}
        assert _responses_tool_choice_to_chat_tool_choice(value) == value

    def test_dict_with_function_block_name(self):
        """Test dict extracts name from function block."""
        value = {"type": "function", "function": {"name": "my_tool"}}
        result = _responses_tool_choice_to_chat_tool_choice(value)
        assert result == {"type": "function", "function": {"name": "my_tool"}}

    def test_dict_with_empty_function_name(self):
        """Test dict with empty function name returns as-is."""
        value = {"type": "function", "function": {"name": ""}}
        result = _responses_tool_choice_to_chat_tool_choice(value)
        assert result == value


# ============================================================================
# Response Format Conversion Tests
# ============================================================================


class TestChatResponseFormatToResponsesTextFormat:
    """Tests for _chat_response_format_to_responses_text_format()."""

    def test_text_converts(self):
        """Test text response format conversion."""
        result = _chat_response_format_to_responses_text_format({"type": "text"})
        assert result == {"type": "text"}

    def test_json_object_converts(self):
        """Test json_object response format conversion."""
        result = _chat_response_format_to_responses_text_format({"type": "json_object"})
        assert result == {"type": "json_object"}

    def test_json_schema_converts(self):
        """Test json_schema response format conversion."""
        value = {
            "type": "json_schema",
            "json_schema": {
                "name": "person",
                "schema": {"type": "object"},
            },
        }
        result = _chat_response_format_to_responses_text_format(value)
        assert result is not None
        assert result["type"] == "json_schema"
        assert "schema" in result

    def test_non_dict_returns_none(self):
        """Test non-dict input returns None."""
        assert _chat_response_format_to_responses_text_format("not a dict") is None
        assert _chat_response_format_to_responses_text_format(123) is None
        assert _chat_response_format_to_responses_text_format(None) is None

    def test_unknown_type_returns_none(self):
        """Test unknown format type returns None."""
        assert _chat_response_format_to_responses_text_format({"type": "unknown"}) is None

    def test_json_schema_without_json_schema_key(self):
        """Test json_schema type without json_schema key returns None."""
        assert _chat_response_format_to_responses_text_format({"type": "json_schema"}) is None

    def test_json_schema_with_non_dict_json_schema(self):
        """Test json_schema with non-dict json_schema returns None."""
        value = {"type": "json_schema", "json_schema": "not a dict"}
        assert _chat_response_format_to_responses_text_format(value) is None

    def test_json_schema_without_name(self):
        """Test json_schema without name returns None."""
        value = {"type": "json_schema", "json_schema": {"schema": {}}}
        assert _chat_response_format_to_responses_text_format(value) is None

    def test_json_schema_with_empty_name(self):
        """Test json_schema with empty name returns None."""
        value = {"type": "json_schema", "json_schema": {"name": "  ", "schema": {}}}
        assert _chat_response_format_to_responses_text_format(value) is None

    def test_json_schema_without_schema(self):
        """Test json_schema without schema returns None."""
        value = {"type": "json_schema", "json_schema": {"name": "test"}}
        assert _chat_response_format_to_responses_text_format(value) is None

    def test_json_schema_with_description(self):
        """Test json_schema with description is preserved."""
        value = {
            "type": "json_schema",
            "json_schema": {
                "name": "test",
                "schema": {"type": "object"},
                "description": "A test schema",
            },
        }
        result = _chat_response_format_to_responses_text_format(value)
        assert result["description"] == "A test schema"

    def test_json_schema_with_strict(self):
        """Test json_schema with strict flag is preserved."""
        value = {
            "type": "json_schema",
            "json_schema": {"name": "test", "schema": {"type": "object"}, "strict": True},
        }
        result = _chat_response_format_to_responses_text_format(value)
        assert result["strict"] is True


class TestResponsesTextFormatToChatResponseFormat:
    """Tests for _responses_text_format_to_chat_response_format()."""

    def test_reverses(self):
        """Test responses text format converts back to chat format."""
        value = {"type": "json_schema", "name": "person", "schema": {"type": "object"}}
        result = _responses_text_format_to_chat_response_format(value)

        assert result["type"] == "json_schema"
        assert "json_schema" in result

    def test_non_dict_returns_none(self):
        """Test non-dict input returns None."""
        assert _responses_text_format_to_chat_response_format("not a dict") is None
        assert _responses_text_format_to_chat_response_format(None) is None

    def test_unknown_type_returns_none(self):
        """Test unknown format type returns None."""
        assert _responses_text_format_to_chat_response_format({"type": "unknown"}) is None

    def test_json_schema_without_name(self):
        """Test json_schema without name returns None."""
        value = {"type": "json_schema", "schema": {"type": "object"}}
        assert _responses_text_format_to_chat_response_format(value) is None

    def test_json_schema_without_schema(self):
        """Test json_schema without schema returns None."""
        value = {"type": "json_schema", "name": "test"}
        assert _responses_text_format_to_chat_response_format(value) is None

    def test_json_schema_with_description(self):
        """Test json_schema with description is preserved."""
        value = {
            "type": "json_schema",
            "name": "test",
            "schema": {"type": "object"},
            "description": "A description",
        }
        result = _responses_text_format_to_chat_response_format(value)
        assert result["json_schema"]["description"] == "A description"

    def test_json_schema_with_strict(self):
        """Test json_schema with strict is preserved."""
        value = {
            "type": "json_schema",
            "name": "test",
            "schema": {"type": "object"},
            "strict": False,
        }
        result = _responses_text_format_to_chat_response_format(value)
        assert result["json_schema"]["strict"] is False


# ============================================================================
# _normalise_openrouter_responses_text_format Tests
# ============================================================================


class TestNormaliseOpenrouterResponsesTextFormat:
    """Tests for _normalise_openrouter_responses_text_format()."""

    def test_non_dict_is_noop(self):
        """Test non-dict input is no-op."""
        _normalise_openrouter_responses_text_format("not a dict")
        _normalise_openrouter_responses_text_format(None)

    def test_removes_response_format_when_no_text(self):
        """Test response_format is removed and not migrated when invalid."""
        payload = {"response_format": {"type": "unknown"}}
        _normalise_openrouter_responses_text_format(payload)
        assert "response_format" not in payload
        assert "text" not in payload

    def test_migrates_response_format_to_text(self):
        """Test valid response_format is migrated to text.format."""
        payload = {"response_format": {"type": "json_object"}}
        _normalise_openrouter_responses_text_format(payload)
        assert "response_format" not in payload
        assert payload["text"]["format"] == {"type": "json_object"}

    def test_existing_text_format_preferred(self):
        """Test existing text.format is preferred over response_format."""
        payload = {
            "response_format": {"type": "json_object"},
            "text": {"format": {"type": "text"}},
        }
        _normalise_openrouter_responses_text_format(payload)
        assert payload["text"]["format"] == {"type": "text"}

    def test_invalid_text_format_dropped(self, caplog):
        """Test invalid text.format is dropped with warning."""
        payload = {"text": {"format": {"type": "invalid"}}}
        with caplog.at_level(logging.WARNING):
            _normalise_openrouter_responses_text_format(payload)
        assert "format" not in payload.get("text", {})

    def test_empty_text_removed_when_no_format(self):
        """Test empty text dict is removed when no format."""
        payload = {"text": {}}
        _normalise_openrouter_responses_text_format(payload)
        assert "text" not in payload

    def test_non_dict_text_with_response_format(self):
        """Test non-dict text is replaced when response_format present."""
        payload = {"text": "not a dict", "response_format": {"type": "text"}}
        _normalise_openrouter_responses_text_format(payload)
        assert payload["text"]["format"] == {"type": "text"}

    def test_non_dict_text_without_response_format(self):
        """Test non-dict text is preserved when no valid response_format."""
        payload = {"text": "not a dict"}
        _normalise_openrouter_responses_text_format(payload)
        # Should early return without changing text


# ============================================================================
# _responses_input_to_chat_messages Tests
# ============================================================================


class TestResponsesInputToChatMessages:
    """Tests for _responses_input_to_chat_messages()."""

    def test_handles_user_message(self):
        """Test user message conversion."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        # Content blocks (input_text) become structured (type: text)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hello"

    def test_handles_assistant_message(self):
        """Test assistant message conversion."""
        input_value = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there!"}],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        # Content blocks (output_text) become structured (type: text)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hi there!"

    def test_handles_system_message(self):
        """Test system message conversion."""
        input_value = [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "System prompt"}],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)

        assert result[0]["role"] == "system"
        # Content blocks (input_text) become structured (type: text)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "System prompt"

    def test_handles_image_content(self):
        """Test image content block conversion."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "What's in this image?"},
                    {
                        "type": "input_image",
                        "image_url": "https://example.com/image.png",
                    },
                ],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)

        content = result[0]["content"]
        assert isinstance(content, list)
        image_block = next((c for c in content if c.get("type") == "image_url"), None)
        assert image_block is not None

    def test_handles_function_call_output(self):
        """Test function call output (tool result) conversion."""
        input_value = [
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": '{"result": "sunny"}',
            }
        ]
        result = _responses_input_to_chat_messages(input_value)

        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_123"

    def test_handles_function_call(self):
        """Test function call conversion."""
        input_value = [
            {
                "type": "function_call",
                "id": "call_123",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            }
        ]
        result = _responses_input_to_chat_messages(input_value)

        assert result[0]["role"] == "assistant"
        tool_calls = result[0].get("tool_calls", [])
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"

    def test_handles_empty(self):
        """Test empty input."""
        assert _responses_input_to_chat_messages([]) == []
        assert _responses_input_to_chat_messages(None) == []

    def test_string_input(self):
        """Test plain string input is converted to user message."""
        result = _responses_input_to_chat_messages("Hello, world!")
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, world!"

    def test_empty_string_input(self):
        """Test empty string input returns empty list."""
        result = _responses_input_to_chat_messages("   ")
        assert result == []

    def test_non_list_non_string_returns_empty(self):
        """Test non-list, non-string input returns empty list."""
        assert _responses_input_to_chat_messages(123) == []
        assert _responses_input_to_chat_messages({"type": "message"}) == []

    def test_non_dict_items_skipped(self):
        """Test non-dict items in list are skipped."""
        input_value = [
            "not a dict",
            {"type": "message", "role": "user", "content": "Hi"},
        ]
        result = _responses_input_to_chat_messages(input_value)
        assert len(result) == 1
        assert result[0]["content"] == "Hi"

    def test_message_without_role_skipped(self):
        """Test messages without role are skipped."""
        input_value = [{"type": "message", "content": "No role"}]
        result = _responses_input_to_chat_messages(input_value)
        assert result == []

    def test_message_with_string_content(self):
        """Test message with string content."""
        input_value = [{"type": "message", "role": "user", "content": "Plain text"}]
        result = _responses_input_to_chat_messages(input_value)
        assert result[0]["content"] == "Plain text"

    def test_message_with_annotations(self):
        """Test message annotations are preserved."""
        input_value = [
            {
                "type": "message",
                "role": "assistant",
                "content": "Hi",
                "annotations": [{"type": "url_citation", "url": "http://example.com"}],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        assert "annotations" in result[0]
        assert len(result[0]["annotations"]) == 1

    def test_message_with_reasoning_details(self):
        """Test message reasoning_details are preserved."""
        input_value = [
            {
                "type": "message",
                "role": "assistant",
                "content": "Hi",
                "reasoning_details": [{"type": "thinking"}],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        assert "reasoning_details" in result[0]

    def test_image_url_block_dict(self):
        """Test image_url block with dict value."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},
                ],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        content = result[0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "http://example.com/img.png"

    def test_image_url_block_string(self):
        """Test image_url block with string value."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "http://example.com/img.png"},
                ],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        content = result[0]["content"]
        assert content[0]["image_url"]["url"] == "http://example.com/img.png"

    def test_input_image_with_detail(self):
        """Test input_image block with detail setting."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "http://example.com/img.png",
                        "detail": "high",
                    },
                ],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        content = result[0]["content"]
        assert content[0]["image_url"]["detail"] == "high"

    def test_input_audio_block(self):
        """Test input_audio block conversion."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": "base64data", "format": "wav"}},
                ],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        content = result[0]["content"]
        assert content[0]["type"] == "input_audio"

    def test_video_url_block_dict(self):
        """Test video_url block with dict value."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "http://example.com/vid.mp4"}},
                ],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        content = result[0]["content"]
        assert content[0]["type"] == "video_url"

    def test_video_url_block_string(self):
        """Test video_url block with string value."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": "http://example.com/vid.mp4"},
                ],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        content = result[0]["content"]
        assert content[0]["video_url"]["url"] == "http://example.com/vid.mp4"

    def test_input_file_block_with_data(self):
        """Test input_file block with file_data."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": "test.txt",
                        "file_data": "base64content",
                    },
                ],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        content = result[0]["content"]
        assert content[0]["type"] == "file"
        assert content[0]["file"]["filename"] == "test.txt"
        assert content[0]["file"]["file_data"] == "base64content"

    def test_input_file_block_with_url(self):
        """Test input_file block with file_url."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_url": "http://example.com/file.txt",
                    },
                ],
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        content = result[0]["content"]
        assert content[0]["file"]["file_data"] == "http://example.com/file.txt"

    def test_empty_content_blocks(self):
        """Test message with empty content blocks list."""
        input_value = [{"type": "message", "role": "user", "content": []}]
        result = _responses_input_to_chat_messages(input_value)
        assert result[0]["content"] == ""

    def test_function_call_output_non_string_output(self):
        """Test function_call_output with non-string output is JSON-serialized."""
        input_value = [
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": {"result": "data"},
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        assert result[0]["content"] == '{"result": "data"}'

    def test_function_call_output_none_output(self):
        """Test function_call_output with None output."""
        input_value = [
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": None,
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        assert result[0]["content"] == ""

    def test_function_call_with_call_id(self):
        """Test function_call using call_id field."""
        input_value = [
            {
                "type": "function_call",
                "call_id": "call_456",
                "name": "my_func",
                "arguments": '{"a": 1}',
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        assert result[0]["tool_calls"][0]["id"] == "call_456"

    def test_function_call_non_string_arguments(self):
        """Test function_call with non-string arguments is JSON-serialized."""
        input_value = [
            {
                "type": "function_call",
                "id": "call_789",
                "name": "my_func",
                "arguments": {"key": "value"},
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == '{"key": "value"}'

    def test_function_call_none_arguments(self):
        """Test function_call with None arguments defaults to empty object."""
        input_value = [
            {
                "type": "function_call",
                "id": "call_abc",
                "name": "my_func",
                "arguments": None,
            }
        ]
        result = _responses_input_to_chat_messages(input_value)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    def test_function_call_missing_required_fields(self):
        """Test function_call missing call_id or name is skipped."""
        input_value = [
            {"type": "function_call", "name": "func"},  # No id/call_id
            {"type": "function_call", "id": "123"},  # No name
            {"type": "function_call", "id": "456", "name": ""},  # Empty name
        ]
        result = _responses_input_to_chat_messages(input_value)
        assert result == []


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestModelParamsToDict:
    """Tests for _model_params_to_dict()."""

    def test_with_dict(self):
        """Test _model_params_to_dict with dict input."""
        params = {"temperature": 0.7, "max_tokens": 100}
        result = _model_params_to_dict(params)
        assert result == params

    def test_with_non_dict(self):
        """Test _model_params_to_dict with non-dict returns empty."""
        assert _model_params_to_dict("not a dict") == {}
        assert _model_params_to_dict(None) == {}
        assert _model_params_to_dict([1, 2, 3]) == {}


class TestGetDisableParam:
    """Tests for _get_disable_param()."""

    def test_returns_bool(self):
        """Test _get_disable_param extracts disable flags."""
        params = {"disable_feature": True, "other": False}

        assert _get_disable_param(params, "disable_feature") is True
        assert _get_disable_param(params, "other") is False
        assert _get_disable_param(params, "missing") is False

    def test_non_dict(self):
        """Test _get_disable_param with non-dict returns False."""
        assert _get_disable_param(None, "key") is False
        assert _get_disable_param("string", "key") is False

    def test_from_custom_params(self):
        """Test getting param from custom_params."""
        params = {"custom_params": {"disable_feature": True}}
        assert _get_disable_param(params, "disable_feature") is True

    def test_from_openrouter_pipe_container(self):
        """Test getting param from openrouter_pipe container."""
        params = {"openrouter_pipe": {"disable_feature": True}}
        assert _get_disable_param(params, "disable_feature") is True

    def test_from_openrouter_container(self):
        """Test getting param from openrouter container."""
        params = {"openrouter": {"disable_feature": True}}
        assert _get_disable_param(params, "disable_feature") is True

    def test_from_pipe_container(self):
        """Test getting param from pipe container."""
        params = {"pipe": {"disable_feature": True}}
        assert _get_disable_param(params, "disable_feature") is True

    def test_from_nested_custom_params_container(self):
        """Test getting param from nested custom_params.openrouter_pipe."""
        params = {"custom_params": {"openrouter_pipe": {"disable_feature": True}}}
        assert _get_disable_param(params, "disable_feature") is True

    def test_object_with_model_dump(self):
        """Test getting param from object with model_dump()."""

        class ParamsObj:
            def model_dump(self):
                return {"disable_feature": True}

        assert _get_disable_param(ParamsObj(), "disable_feature") is True

    def test_object_model_dump_exception(self):
        """Test handling when model_dump() raises exception."""

        class BadParamsObj:
            def model_dump(self):
                raise RuntimeError("Error")

        assert _get_disable_param(BadParamsObj(), "disable_feature") is False


class TestSanitizeOpenrouterMetadata:
    """Tests for _sanitize_openrouter_metadata()."""

    def test_with_valid_dict(self):
        """Test _sanitize_openrouter_metadata with valid dict."""
        # Function only accepts string keys AND string values
        raw = {"key": "value", "number": 123, "another": "text"}
        result = _sanitize_openrouter_metadata(raw)

        # Only string -> string pairs are preserved
        assert result["key"] == "value"
        assert result["another"] == "text"
        # Integer values are skipped (not stringified)
        assert "number" not in result

    def test_filters_none_values(self):
        """Test None values are filtered."""
        raw = {"key": "value", "empty": None}
        result = _sanitize_openrouter_metadata(raw)

        assert "key" in result
        assert "empty" not in result

    def test_non_dict(self):
        """Test non-dict returns None."""
        assert _sanitize_openrouter_metadata("string") is None
        assert _sanitize_openrouter_metadata(None) is None

    def test_max_pairs_limit(self):
        """Test max 16 pairs are allowed."""
        raw = {f"key{i}": f"value{i}" for i in range(20)}
        result = _sanitize_openrouter_metadata(raw)
        assert len(result) == 16

    def test_long_key_filtered(self):
        """Test keys longer than 64 chars are filtered."""
        raw = {"a" * 65: "value", "short": "value"}
        result = _sanitize_openrouter_metadata(raw)
        assert "short" in result
        assert "a" * 65 not in result

    def test_long_value_filtered(self):
        """Test values longer than 512 chars are filtered."""
        raw = {"key": "a" * 513, "short": "value"}
        result = _sanitize_openrouter_metadata(raw)
        assert "short" in result
        assert "key" not in result

    def test_brackets_in_key_filtered(self):
        """Test keys with brackets are filtered."""
        raw = {"key[0]": "value", "normal": "value"}
        result = _sanitize_openrouter_metadata(raw)
        assert "normal" in result
        assert "key[0]" not in result

    def test_all_invalid_returns_none(self):
        """Test all invalid entries returns None."""
        raw = {"key[0]": "value", "long": "a" * 600}
        result = _sanitize_openrouter_metadata(raw)
        assert result is None


# ============================================================================
# _strip_disable_model_settings_params Tests
# ============================================================================


class TestStripDisableModelSettingsParams:
    """Tests for _strip_disable_model_settings_params()."""

    def test_removes_disable_params(self):
        """Test that disable_* params are removed."""
        payload = {
            "model": "gpt-4",
            "disable_model_metadata_sync": True,
            "disable_capability_updates": False,
            "disable_image_updates": True,
            "disable_openrouter_search_auto_attach": True,
            "disable_openrouter_search_default_on": True,
            "disable_direct_uploads_auto_attach": True,
            "disable_description_updates": True,
            "disable_native_websearch": True,
            "disable_native_web_search": True,
        }
        _strip_disable_model_settings_params(payload)
        assert "model" in payload
        assert "disable_model_metadata_sync" not in payload
        assert "disable_capability_updates" not in payload
        assert "disable_image_updates" not in payload

    def test_non_dict_is_noop(self):
        """Test non-dict input is no-op."""
        _strip_disable_model_settings_params("not a dict")
        _strip_disable_model_settings_params(None)
        _strip_disable_model_settings_params(123)


# ============================================================================
# _apply_model_fallback_to_payload Tests
# ============================================================================


class TestApplyModelFallbackToPayload:
    """Tests for _apply_model_fallback_to_payload()."""

    def test_non_dict_is_noop(self):
        """Test non-dict input is no-op."""
        _apply_model_fallback_to_payload("not a dict")
        _apply_model_fallback_to_payload(None)

    def test_no_fallback_no_models(self):
        """Test no changes when neither model_fallback nor models present."""
        payload = {"model": "gpt-4"}
        _apply_model_fallback_to_payload(payload)
        assert "models" not in payload

    def test_csv_fallback_parsed(self):
        """Test model_fallback CSV is parsed."""
        payload = {"model": "gpt-4", "model_fallback": "model1, model2, model3"}
        _apply_model_fallback_to_payload(payload)
        assert payload["models"] == ["model1", "model2", "model3"]
        assert "model_fallback" not in payload

    def test_existing_models_preserved(self):
        """Test existing models are preserved and come first."""
        payload = {
            "model": "gpt-4",
            "models": ["existing1", "existing2"],
            "model_fallback": "new1, new2",
        }
        _apply_model_fallback_to_payload(payload)
        assert payload["models"][0] == "existing1"
        assert payload["models"][1] == "existing2"
        assert "new1" in payload["models"]

    def test_duplicates_removed(self):
        """Test duplicate models are removed."""
        payload = {
            "model": "gpt-4",
            "models": ["model1", "model2"],
            "model_fallback": "model2, model3",
        }
        _apply_model_fallback_to_payload(payload)
        assert payload["models"].count("model2") == 1

    def test_empty_fallback(self):
        """Test empty model_fallback is handled."""
        payload = {"model": "gpt-4", "model_fallback": ""}
        _apply_model_fallback_to_payload(payload)
        assert "models" not in payload


# ============================================================================
# _apply_disable_native_websearch_to_payload Tests
# ============================================================================


class TestApplyDisableNativeWebsearchToPayload:
    """Tests for _apply_disable_native_websearch_to_payload()."""

    def test_non_dict_is_noop(self):
        """Test non-dict input is no-op."""
        _apply_disable_native_websearch_to_payload("not a dict")
        _apply_disable_native_websearch_to_payload(None)

    def test_flag_not_set_is_noop(self):
        """Test no changes when flag is not set."""
        payload = {"model": "gpt-4", "web_search_options": {}}
        _apply_disable_native_websearch_to_payload(payload)
        assert "web_search_options" in payload

    def test_flag_false_is_noop(self):
        """Test no changes when flag is False."""
        payload = {
            "model": "gpt-4",
            "disable_native_websearch": False,
            "web_search_options": {},
        }
        _apply_disable_native_websearch_to_payload(payload)
        assert "web_search_options" in payload

    def test_removes_web_search_options(self):
        """Test web_search_options is removed when flag is True."""
        payload = {
            "model": "gpt-4",
            "disable_native_websearch": True,
            "web_search_options": {"enabled": True},
        }
        _apply_disable_native_websearch_to_payload(payload)
        assert "web_search_options" not in payload

    def test_removes_web_plugin(self):
        """Test web plugin is removed from plugins."""
        payload = {
            "model": "gpt-4",
            "disable_native_websearch": True,
            "plugins": [{"id": "web"}, {"id": "other"}],
        }
        _apply_disable_native_websearch_to_payload(payload)
        assert len(payload["plugins"]) == 1
        assert payload["plugins"][0]["id"] == "other"

    def test_removes_plugins_when_only_web(self):
        """Test plugins is removed when only web plugin."""
        payload = {
            "model": "gpt-4",
            "disable_native_websearch": True,
            "plugins": [{"id": "web"}],
        }
        _apply_disable_native_websearch_to_payload(payload)
        assert "plugins" not in payload

    def test_alternative_flag_name(self):
        """Test disable_native_web_search (with underscore) also works."""
        payload = {
            "model": "gpt-4",
            "disable_native_web_search": True,
            "web_search_options": {},
        }
        _apply_disable_native_websearch_to_payload(payload)
        assert "web_search_options" not in payload


# ============================================================================
# _filter_replayable_input_items Tests
# ============================================================================


class TestFilterReplayableInputItems:
    """Tests for _filter_replayable_input_items()."""

    def test_non_list_returns_as_is(self):
        """Test non-list input is returned as-is."""
        assert _filter_replayable_input_items("not a list") == "not a list"
        assert _filter_replayable_input_items(123) == 123

    def test_non_dict_items_preserved(self):
        """Test non-dict items are preserved."""
        items = ["string", 123, {"type": "message"}]
        result = _filter_replayable_input_items(items)
        assert len(result) == 3

    def test_non_replayable_filtered(self):
        """Test non-replayable tool artifacts are filtered."""
        # The actual non-replayable artifacts are: local_shell_call, image_generation_call,
        # file_search_call, web_search_call
        items = [
            {"type": "message", "content": "Hi"},
            {"type": "web_search_call", "id": "123"},
            {"type": "file_search_call", "id": "456"},
        ]
        result = _filter_replayable_input_items(items)
        assert len(result) == 1
        assert result[0]["type"] == "message"

    def test_no_changes_returns_original(self):
        """Test when no filtering needed, original list is returned."""
        items = [{"type": "message"}, {"type": "function_call"}]
        result = _filter_replayable_input_items(items)
        assert result is items  # Same object


# ============================================================================
# _apply_identifier_valves_to_payload Tests
# ============================================================================


class TestApplyIdentifierValvesToPayload:
    """Tests for _apply_identifier_valves_to_payload()."""

    def test_non_dict_payload_is_noop(self):
        """Test non-dict payload is no-op."""
        pipe = Pipe()
        _apply_identifier_valves_to_payload(
            "not a dict",
            valves=pipe.valves,
            owui_metadata={},
            owui_user_id="user123",
        )

    def test_user_id_added_when_valve_enabled(self, pipe_instance):
        """Test user is added when SEND_END_USER_ID is enabled."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(update={"SEND_END_USER_ID": True})
        payload = {}
        _apply_identifier_valves_to_payload(
            payload,
            valves=valves,
            owui_metadata={},
            owui_user_id="user123",
        )
        assert payload["user"] == "user123"
        assert payload["metadata"]["user_id"] == "user123"

    def test_user_removed_when_valve_disabled(self, pipe_instance):
        """Test user is removed when SEND_END_USER_ID is disabled."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(update={"SEND_END_USER_ID": False})
        payload = {"user": "existing"}
        _apply_identifier_valves_to_payload(
            payload,
            valves=valves,
            owui_metadata={},
            owui_user_id="user123",
        )
        assert "user" not in payload

    def test_session_id_added_when_valve_enabled(self, pipe_instance):
        """Test session_id is added when SEND_SESSION_ID is enabled."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(update={"SEND_SESSION_ID": True})
        payload = {}
        _apply_identifier_valves_to_payload(
            payload,
            valves=valves,
            owui_metadata={"session_id": "session123"},
            owui_user_id="",
        )
        assert payload["session_id"] == "session123"
        assert payload["metadata"]["session_id"] == "session123"

    def test_session_id_removed_when_valve_disabled(self, pipe_instance):
        """Test session_id is removed when SEND_SESSION_ID is disabled."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(update={"SEND_SESSION_ID": False})
        payload = {"session_id": "existing"}
        _apply_identifier_valves_to_payload(
            payload,
            valves=valves,
            owui_metadata={"session_id": "session123"},
            owui_user_id="",
        )
        assert "session_id" not in payload

    def test_chat_id_added_to_metadata(self, pipe_instance):
        """Test chat_id is added to metadata when SEND_CHAT_ID is enabled."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(update={"SEND_CHAT_ID": True})
        payload = {}
        _apply_identifier_valves_to_payload(
            payload,
            valves=valves,
            owui_metadata={"chat_id": "chat123"},
            owui_user_id="",
        )
        assert payload["metadata"]["chat_id"] == "chat123"

    def test_message_id_added_to_metadata(self, pipe_instance):
        """Test message_id is added to metadata when SEND_MESSAGE_ID is enabled."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(update={"SEND_MESSAGE_ID": True})
        payload = {}
        _apply_identifier_valves_to_payload(
            payload,
            valves=valves,
            owui_metadata={"message_id": "msg123"},
            owui_user_id="",
        )
        assert payload["metadata"]["message_id"] == "msg123"

    def test_metadata_removed_when_empty(self, pipe_instance):
        """Test metadata is removed when no identifiers enabled."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(
            update={
                "SEND_END_USER_ID": False,
                "SEND_SESSION_ID": False,
                "SEND_CHAT_ID": False,
                "SEND_MESSAGE_ID": False,
            }
        )
        payload = {"metadata": {"existing": "value"}}
        _apply_identifier_valves_to_payload(
            payload,
            valves=valves,
            owui_metadata={},
            owui_user_id="",
        )
        assert "metadata" not in payload

    def test_invalid_user_id_omitted(self, pipe_instance, caplog):
        """Test invalid user_id is omitted with debug log."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(update={"SEND_END_USER_ID": True})
        payload = {}
        with caplog.at_level(logging.DEBUG):
            _apply_identifier_valves_to_payload(
                payload,
                valves=valves,
                owui_metadata={},
                owui_user_id="",  # Empty user ID
            )
        assert "user" not in payload

    def test_invalid_session_id_omitted(self, pipe_instance, caplog):
        """Test invalid session_id is omitted with debug log."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(update={"SEND_SESSION_ID": True})
        payload = {}
        with caplog.at_level(logging.DEBUG):
            _apply_identifier_valves_to_payload(
                payload,
                valves=valves,
                owui_metadata={"session_id": "  "},  # Whitespace only
                owui_user_id="",
            )
        assert "session_id" not in payload


# ============================================================================
# ResponsesBody Field Validator Tests
# ============================================================================


class TestResponsesBodyValidators:
    """Tests for ResponsesBody field validators."""

    def test_strip_blank_string_whitespace_only(self):
        """Test that whitespace-only strings become None."""
        result = ResponsesBody._strip_blank_string("   ")
        assert result is None

    def test_strip_blank_string_empty(self):
        """Test that empty strings become None."""
        result = ResponsesBody._strip_blank_string("")
        assert result is None

    def test_strip_blank_string_valid(self):
        """Test that valid strings are trimmed."""
        result = ResponsesBody._strip_blank_string("  hello  ")
        assert result == "hello"

    def test_strip_blank_string_non_string(self):
        """Test that non-strings are returned as-is."""
        assert ResponsesBody._strip_blank_string(123) == 123
        assert ResponsesBody._strip_blank_string(None) is None
        assert ResponsesBody._strip_blank_string([1, 2]) == [1, 2]

    def test_coerce_int_fields_bool_raises(self):
        """Test that boolean values raise ValidationError for int fields."""
        with pytest.raises(ValidationError):
            ResponsesBody(model="test", input="hi", max_tokens=True)

    def test_coerce_int_fields_float_rounds(self):
        """Test that float values are rounded for int fields."""
        body = ResponsesBody(model="test", input="hi", max_tokens=100.7)
        assert body.max_tokens == 101

    def test_coerce_int_fields_string_numeric(self):
        """Test that numeric strings are converted for int fields."""
        body = ResponsesBody(model="test", input="hi", max_tokens="200")
        assert body.max_tokens == 200

    def test_coerce_int_fields_string_float(self):
        """Test that float strings are converted and rounded for int fields."""
        body = ResponsesBody(model="test", input="hi", max_tokens="150.9")
        assert body.max_tokens == 151

    def test_coerce_int_fields_invalid_string(self):
        """Test that invalid string values raise ValidationError."""
        with pytest.raises(ValidationError):
            ResponsesBody(model="test", input="hi", max_tokens="not-a-number")

    def test_coerce_int_fields_invalid_type(self):
        """Test that invalid types raise ValidationError."""
        with pytest.raises(ValidationError):
            ResponsesBody(model="test", input="hi", max_tokens=[100])

    def test_coerce_int_fields_whitespace_string(self):
        """Test that whitespace-only string becomes None."""
        body = ResponsesBody(model="test", input="hi", max_tokens="  ")
        assert body.max_tokens is None

    def test_coerce_float_fields_whitespace(self):
        """Test that whitespace-only string becomes None for float fields."""
        body = ResponsesBody(model="test", input="hi", temperature="  ")
        assert body.temperature is None

    def test_coerce_models_list_csv_string(self):
        """Test that CSV strings are parsed to list."""
        body = ResponsesBody(model="test", input="hi", models="model1, model2, model3")
        assert body.models == ["model1", "model2", "model3"]

    def test_coerce_models_list_csv_with_empty(self):
        """Test that empty entries in CSV are filtered."""
        body = ResponsesBody(model="test", input="hi", models="model1, , model3")
        assert body.models == ["model1", "model3"]

    def test_coerce_models_list_all_empty(self):
        """Test that all-empty CSV becomes None."""
        body = ResponsesBody(model="test", input="hi", models=", , ")
        assert body.models is None

    def test_coerce_models_list_array_with_non_strings(self):
        """Test that non-string entries in array are filtered."""
        body = ResponsesBody(model="test", input="hi", models=["model1", 123, "model2", None])
        assert body.models == ["model1", "model2"]

    def test_coerce_models_list_array_with_whitespace(self):
        """Test that whitespace entries in array are filtered."""
        body = ResponsesBody(model="test", input="hi", models=["model1", "  ", "model2"])
        assert body.models == ["model1", "model2"]

    def test_coerce_models_list_invalid_type(self):
        """Test that invalid type raises ValidationError."""
        with pytest.raises(ValidationError):
            ResponsesBody(model="test", input="hi", models=123)


# ============================================================================
# transform_owui_tools Tests
# ============================================================================


class TestTransformOwuiTools:
    """Tests for ResponsesBody.transform_owui_tools()."""

    def test_empty_tools(self):
        """Test with empty or None tools."""
        assert ResponsesBody.transform_owui_tools(None) == []
        assert ResponsesBody.transform_owui_tools({}) == []

    def test_malformed_entries_skipped(self):
        """Test that entries without spec or name are skipped."""
        tools = {
            "tool1": {},  # No spec
            "tool2": {"spec": {}},  # No name in spec
            "tool3": {"spec": {"name": ""}},  # Empty name
        }
        result = ResponsesBody.transform_owui_tools(tools)
        assert result == []

    def test_valid_tool_converted(self):
        """Test that valid tools are converted."""
        tools = {
            "my_tool": {
                "spec": {
                    "name": "my_tool",
                    "description": "A test tool",
                    "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}},
                }
            }
        }
        result = ResponsesBody.transform_owui_tools(tools)
        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["name"] == "my_tool"
        assert result[0]["description"] == "A test tool"

    def test_tool_without_description(self):
        """Test tool without description uses name as description."""
        tools = {
            "my_tool": {
                "spec": {
                    "name": "my_tool",
                    "parameters": {"type": "object"},
                }
            }
        }
        result = ResponsesBody.transform_owui_tools(tools)
        assert result[0]["description"] == "my_tool"

    def test_tool_without_parameters(self):
        """Test tool without parameters gets default."""
        tools = {
            "my_tool": {
                "spec": {
                    "name": "my_tool",
                }
            }
        }
        result = ResponsesBody.transform_owui_tools(tools)
        assert result[0]["parameters"] == {"type": "object", "properties": {}}

    def test_strict_mode(self):
        """Test strict mode adds strict flag."""
        tools = {
            "my_tool": {
                "spec": {
                    "name": "my_tool",
                    "parameters": {"type": "object", "properties": {}},
                }
            }
        }
        result = ResponsesBody.transform_owui_tools(tools, strict=True)
        assert result[0].get("strict") is True


# ============================================================================
# _convert_function_call_to_tool_choice Tests
# ============================================================================


class TestConvertFunctionCallToToolChoice:
    """Tests for ResponsesBody._convert_function_call_to_tool_choice()."""

    def test_none_returns_none(self):
        """Test None input returns None."""
        assert ResponsesBody._convert_function_call_to_tool_choice(None) is None

    def test_string_auto(self):
        """Test 'auto' string."""
        assert ResponsesBody._convert_function_call_to_tool_choice("auto") == "auto"
        assert ResponsesBody._convert_function_call_to_tool_choice("  AUTO  ") == "auto"

    def test_string_none(self):
        """Test 'none' string."""
        assert ResponsesBody._convert_function_call_to_tool_choice("none") == "none"
        assert ResponsesBody._convert_function_call_to_tool_choice("  NONE  ") == "none"

    def test_string_other(self):
        """Test other string values return None."""
        assert ResponsesBody._convert_function_call_to_tool_choice("required") is None
        assert ResponsesBody._convert_function_call_to_tool_choice("unknown") is None

    def test_dict_with_name(self):
        """Test dict with name field."""
        result = ResponsesBody._convert_function_call_to_tool_choice({"name": "my_func"})
        assert result == {"type": "function", "name": "my_func"}

    def test_dict_with_function_name(self):
        """Test dict with function.name field."""
        result = ResponsesBody._convert_function_call_to_tool_choice(
            {"function": {"name": "my_func"}}
        )
        assert result == {"type": "function", "name": "my_func"}

    def test_dict_name_stripped(self):
        """Test that name is stripped of whitespace."""
        result = ResponsesBody._convert_function_call_to_tool_choice({"name": "  my_func  "})
        assert result == {"type": "function", "name": "my_func"}

    def test_dict_empty_name(self):
        """Test dict with empty name returns None."""
        assert ResponsesBody._convert_function_call_to_tool_choice({"name": ""}) is None
        assert ResponsesBody._convert_function_call_to_tool_choice({"name": "  "}) is None

    def test_dict_no_name(self):
        """Test dict without name returns None."""
        assert ResponsesBody._convert_function_call_to_tool_choice({}) is None
        assert ResponsesBody._convert_function_call_to_tool_choice({"other": "value"}) is None


# ============================================================================
# CompletionsBody Tests
# ============================================================================


class TestCompletionsBody:
    """Tests for CompletionsBody model."""

    def test_basic_creation(self):
        """Test basic CompletionsBody creation."""
        body = CompletionsBody(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
        )
        assert body.model == "gpt-4"
        assert len(body.messages) == 1

    def test_stream_default_false(self):
        """Test stream defaults to False."""
        body = CompletionsBody(model="gpt-4", messages=[])
        assert body.stream is False

    def test_extra_fields_allowed(self):
        """Test extra fields are allowed."""
        body = CompletionsBody(
            model="gpt-4",
            messages=[],
            custom_field="value",
        )
        assert body.custom_field == "value"


# ============================================================================
# ResponsesBody Additional Tests
# ============================================================================


class TestResponsesBody:
    """Tests for ResponsesBody model."""

    def test_basic_creation(self):
        """Test basic ResponsesBody creation."""
        body = ResponsesBody(model="gpt-4", input="Hello")
        assert body.model == "gpt-4"
        assert body.input == "Hello"

    def test_input_as_list(self):
        """Test input as list of messages."""
        body = ResponsesBody(
            model="gpt-4",
            input=[{"type": "message", "role": "user", "content": "Hi"}],
        )
        assert isinstance(body.input, list)

    def test_extra_fields_allowed(self):
        """Test extra fields are allowed."""
        body = ResponsesBody(
            model="gpt-4",
            input="Hi",
            custom_field="value",
        )
        assert body.custom_field == "value"

    def test_model_normalized(self):
        """Test model ID is normalized."""
        # This test depends on ModelFamily.base_model implementation
        body = ResponsesBody(model="openai/gpt-4o", input="Hi")
        # The model should be preserved or normalized based on ModelFamily
        assert body.model is not None


# ============================================================================
# ResponsesBody.from_completions Tests
# ============================================================================


class TestResponsesBodyFromCompletions:
    """Tests for ResponsesBody.from_completions() class method."""

    @pytest.mark.asyncio
    async def test_requires_transformer_context(self):
        """Test that from_completions raises without transformer_context."""
        completions = CompletionsBody(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
        )
        with pytest.raises(RuntimeError, match="transformer_context"):
            await ResponsesBody.from_completions(completions, transformer_context=None)

    @pytest.mark.asyncio
    async def test_basic_conversion(self, pipe_instance_async):
        """Test basic conversion from completions to responses."""
        pipe = pipe_instance_async
        completions = CompletionsBody(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
            temperature=0.7,
        )
        result = await ResponsesBody.from_completions(
            completions,
            transformer_context=pipe,
        )
        assert result.model == "gpt-4"
        assert result.stream is True
        assert result.temperature == 0.7
        assert result.input is not None

    @pytest.mark.asyncio
    async def test_max_tokens_converted(self, pipe_instance_async):
        """Test max_tokens is converted to max_output_tokens."""
        pipe = pipe_instance_async
        completions = CompletionsBody(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=500,
        )
        result = await ResponsesBody.from_completions(
            completions,
            transformer_context=pipe,
        )
        assert result.max_output_tokens == 500

    @pytest.mark.asyncio
    async def test_reasoning_effort_converted(self, pipe_instance_async):
        """Test reasoning_effort is converted to reasoning.effort."""
        pipe = pipe_instance_async
        completions = CompletionsBody(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            reasoning_effort="high",
        )
        result = await ResponsesBody.from_completions(
            completions,
            transformer_context=pipe,
        )
        assert result.reasoning is not None
        assert result.reasoning.get("effort") == "high"

    @pytest.mark.asyncio
    async def test_function_call_converted(self, pipe_instance_async):
        """Test function_call is converted to tool_choice."""
        pipe = pipe_instance_async
        completions = CompletionsBody(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            function_call={"name": "my_func"},
        )
        result = await ResponsesBody.from_completions(
            completions,
            transformer_context=pipe,
        )
        assert result.tool_choice == {"type": "function", "name": "my_func"}

    @pytest.mark.asyncio
    async def test_unsupported_fields_dropped(self, pipe_instance_async, caplog):
        """Test unsupported fields are dropped with warning."""
        pipe = pipe_instance_async
        completions = CompletionsBody(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            n=5,  # Unsupported
            suffix="test",  # Unsupported
        )
        with caplog.at_level(logging.WARNING):
            result = await ResponsesBody.from_completions(
                completions,
                transformer_context=pipe,
            )
        # n and suffix should be dropped
        assert not hasattr(result, "n") or result.n is None
        assert not hasattr(result, "suffix") or result.suffix is None

    @pytest.mark.asyncio
    async def test_filters_invalid_messages(self, pipe_instance_async):
        """Test that messages without role are filtered."""
        pipe = pipe_instance_async
        completions = CompletionsBody(
            model="gpt-4",
            messages=[
                {"content": "no role"},  # Should be filtered (no role)
                {"role": "", "content": "empty role"},  # Should be filtered (empty role)
                {"role": "user", "content": "valid"},
            ],
        )
        result = await ResponsesBody.from_completions(
            completions,
            transformer_context=pipe,
        )
        # The result should have input, even if filtering happened
        assert result.input is not None

    @pytest.mark.asyncio
    async def test_tool_choice_normalized(self, pipe_instance_async):
        """Test tool_choice is normalized from chat format."""
        pipe = pipe_instance_async
        completions = CompletionsBody(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            tool_choice={"type": "function", "function": {"name": "my_tool"}},
        )
        result = await ResponsesBody.from_completions(
            completions,
            transformer_context=pipe,
        )
        # Should be normalized to responses format
        assert result.tool_choice == {"type": "function", "name": "my_tool"}

    @pytest.mark.asyncio
    async def test_tools_normalized(self, pipe_instance_async):
        """Test tools are normalized from chat format."""
        pipe = pipe_instance_async
        completions = CompletionsBody(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "my_tool",
                        "description": "A tool",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        )
        result = await ResponsesBody.from_completions(
            completions,
            transformer_context=pipe,
        )
        # Tools should be normalized to responses format
        assert result.tools is not None
        assert len(result.tools) == 1
        assert result.tools[0]["name"] == "my_tool"

    @pytest.mark.asyncio
    async def test_extra_params_merged(self, pipe_instance_async):
        """Test extra_params are merged into result."""
        pipe = pipe_instance_async
        completions = CompletionsBody(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
        )
        result = await ResponsesBody.from_completions(
            completions,
            transformer_context=pipe,
            custom_param="custom_value",
        )
        assert result.custom_param == "custom_value"

    @pytest.mark.asyncio
    async def test_with_response_format(self, pipe_instance_async):
        """Test response_format is handled correctly."""
        pipe = pipe_instance_async
        completions = CompletionsBody(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            response_format={"type": "json_object"},
        )
        result = await ResponsesBody.from_completions(
            completions,
            transformer_context=pipe,
        )
        # response_format should be normalized to text.format
        assert result.text is not None
        assert result.text.get("format", {}).get("type") == "json_object"


# ============================================================================
# Full Integration Tests - Transforms through Pipe
# ============================================================================


class TestPipeIntegration:
    """Integration tests for transforms applied through Pipe."""

    @pytest.mark.asyncio
    async def test_applies_request_transforms(self, pipe_instance_async):
        """Test that Pipe applies request transforms correctly."""
        pipe = pipe_instance_async
        valves = pipe.valves.model_copy(update={"DEFAULT_LLM_ENDPOINT": "chat_completions"})
        session = pipe._create_http_session(valves)

        sse_response = (
            _sse({"choices": [{"delta": {"content": "Hello"}, "finish_reason": None}]})
            + _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            + "data: [DONE]\n\n"
        )

        received_payload = None

        def capture_payload(url, **kwargs):
            nonlocal received_payload
            received_payload = kwargs.get("json", {})
            return CallbackResult(
                body=sse_response.encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
                status=200,
            )

        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                callback=capture_payload,
            )

            events = []
            async for event in pipe.send_openai_chat_completions_streaming_request(
                session,
                {
                    "model": "openai/gpt-4o",
                    "stream": True,
                    "input": [
                        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
                    ],
                    "max_output_tokens": 100,
                    "unknown_param": "should be filtered",
                },
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            ):
                events.append(event)

            await session.close()

        # Verify transforms were applied
        assert received_payload is not None
        # max_output_tokens -> max_tokens
        assert "max_tokens" in received_payload
        assert received_payload["max_tokens"] == 100
        # input -> messages
        assert "messages" in received_payload
        # Unknown params should be filtered
        assert "unknown_param" not in received_payload

    @pytest.mark.asyncio
    async def test_converts_responses_tools_to_chat(self, pipe_instance_async):
        """Test that Pipe converts Responses tools to Chat format."""
        pipe = pipe_instance_async
        valves = pipe.valves.model_copy(update={"DEFAULT_LLM_ENDPOINT": "chat_completions"})
        session = pipe._create_http_session(valves)

        sse_response = (
            _sse({"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]})
            + _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            + "data: [DONE]\n\n"
        )

        received_payload = None

        def capture_payload(url, **kwargs):
            nonlocal received_payload
            received_payload = kwargs.get("json", {})
            return CallbackResult(
                body=sse_response.encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
                status=200,
            )

        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                callback=capture_payload,
            )

            events = []
            async for event in pipe.send_openai_chat_completions_streaming_request(
                session,
                {
                    "model": "openai/gpt-4o",
                    "stream": True,
                    "input": [],
                    "tools": [
                        {
                            "type": "function",
                            "name": "get_weather",
                            "description": "Get weather",
                            "parameters": {"type": "object", "properties": {}},
                        }
                    ],
                },
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            ):
                events.append(event)

            await session.close()

        # Verify tools were converted to Chat format
        assert received_payload is not None
        tools = received_payload.get("tools", [])
        assert len(tools) == 1
        # Should be wrapped in function structure
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_converts_instructions_to_system(self, pipe_instance_async):
        """Test that Pipe converts instructions to system message."""
        pipe = pipe_instance_async
        valves = pipe.valves.model_copy(update={"DEFAULT_LLM_ENDPOINT": "chat_completions"})
        session = pipe._create_http_session(valves)

        sse_response = (
            _sse({"choices": [{"delta": {"content": "Hi"}, "finish_reason": None}]})
            + _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
            + "data: [DONE]\n\n"
        )

        received_payload = None

        def capture_payload(url, **kwargs):
            nonlocal received_payload
            received_payload = kwargs.get("json", {})
            return CallbackResult(
                body=sse_response.encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
                status=200,
            )

        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                callback=capture_payload,
            )

            events = []
            async for event in pipe.send_openai_chat_completions_streaming_request(
                session,
                {
                    "model": "openai/gpt-4o",
                    "stream": True,
                    "instructions": "You are a helpful assistant",
                    "input": [
                        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
                    ],
                },
                api_key="test-key",
                base_url="https://openrouter.ai/api/v1",
                valves=valves,
            ):
                events.append(event)

            await session.close()

        # Verify instructions became system message
        assert received_payload is not None
        messages = received_payload.get("messages", [])
        system_msg = next((m for m in messages if m["role"] == "system"), None)
        assert system_msg is not None
        assert "helpful assistant" in str(system_msg["content"])

# ===== From test_strip_disable_model_settings_params.py =====


from open_webui_openrouter_pipe import _strip_disable_model_settings_params


def test_strip_disable_model_settings_params_removes_pipe_control_flags() -> None:
    payload = {
        "model": "openai/gpt-5",
        "disable_model_metadata_sync": True,
        "disable_capability_updates": True,
        "disable_image_updates": True,
        "disable_openrouter_search_auto_attach": True,
        "disable_openrouter_search_default_on": True,
        "disable_direct_uploads_auto_attach": True,
        "disable_description_updates": True,
        "disable_native_websearch": True,
        "disable_native_web_search": True,
    }

    _strip_disable_model_settings_params(payload)

    assert payload == {"model": "openai/gpt-5"}


# ===== From test_top_k_passthrough.py =====


from open_webui_openrouter_pipe import _filter_openrouter_request


def test_filter_openrouter_request_forwards_numeric_top_k() -> None:
    payload = {"model": "openai/gpt-5", "input": [], "top_k": 50}
    filtered = _filter_openrouter_request(payload)
    assert filtered["top_k"] == 50.0


def test_filter_openrouter_request_parses_string_top_k() -> None:
    payload = {"model": "openai/gpt-5", "input": [], "top_k": " 50 "}
    filtered = _filter_openrouter_request(payload)
    assert filtered["top_k"] == 50.0


def test_filter_openrouter_request_drops_invalid_top_k() -> None:
    payload = {"model": "openai/gpt-5", "input": [], "top_k": "nope"}
    filtered = _filter_openrouter_request(payload)
    assert "top_k" not in filtered
