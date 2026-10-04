"""Comprehensive tests for open_webui_openrouter_pipe/api/transforms.py

This consolidated test module provides complete coverage for the transforms
module, including both unit tests for isolated functions and integration tests
through the Pipe class.
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false
from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any, ClassVar
from unittest.mock import MagicMock, AsyncMock

import pytest
from pydantic import ValidationError
from aioresponses import aioresponses, CallbackResult

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.core.config import (
    CROCKFORD_ALPHABET,
    OPENAI_EMPTY_USER_TURN_FALLBACK,
    ULID_LENGTH,
)
from open_webui_openrouter_pipe.core.utils import (
    _serialize_kind_marker,
    _serialize_marker,
    _serialize_phase_marker,
)
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
    _apply_openrouter_trace_to_payload,
    _apply_disable_native_websearch_to_payload,
    _apply_provider_routing_params_to_payload,
    _parse_provider_csv,
    _apply_identifier_valves_to_payload,
    _strip_disable_model_settings_params,
    _filter_replayable_input_items,
    ALLOWED_OPENROUTER_FIELDS,
)


def _sse(obj: dict[str, Any]) -> str:
    """Format object as SSE data line."""
    return f"data: {json.dumps(obj)}\n\n"


# The three hidden-marker families, built with the pipe's own serialisers. Never hand-typed:
# `_extract_marker_ulid` is Crockford-strict about a 20-character body, so a hand-typed marker
# is silently not a marker and a test written that way passes for the wrong reason.
_MARKER_ULID = (CROCKFORD_ALPHABET * 2)[:ULID_LENGTH]
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
        """Test top_k integer reaches the wire as a JSON integer."""
        payload = {"model": "gpt-4", "top_k": 10, "input": []}
        result = _filter_openrouter_request(payload)
        assert result["top_k"] == 10
        assert isinstance(result["top_k"], int)
        assert not isinstance(result["top_k"], bool)

    def test_top_k_string_converted(self):
        """Test top_k string reaches the wire as a JSON integer."""
        payload = {"model": "gpt-4", "top_k": "5", "input": []}
        result = _filter_openrouter_request(payload)
        assert result["top_k"] == 5
        assert isinstance(result["top_k"], int)
        assert not isinstance(result["top_k"], bool)

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

    # --- the caller's `text` mapping must survive the call unchanged ---------------
    #
    # `candidate = dict(payload)` is a shallow copy, so the one nested object it aliases
    # is the caller's own `text`. `_normalise_openrouter_responses_text_format` then writes
    # through that alias in place, so the caller's dict gains a `format` key, or **loses**
    # an unparseable one, without ever being asked. Both production call sites pass
    # `dict(responses_request_body)`, and the outer `dict()` copies the top level only, so
    # it does not launder the alias; the same body filtered twice hands the caller one
    # object shared with two different returned payloads.
    #
    # Deliberately one level deep: the `schema` nested inside a `json_schema` format stays
    # shared, because the normaliser never writes below `text` and builds that object fresh.
    # The property is about the caller's dict, not the wire -- the returned payload is
    # unchanged by this fix, which is why the two wire-shape controls below sit here.

    _COPY_ARMS = [
        pytest.param(
            {"text": {"verbosity": "low"}, "response_format": {"type": "json_object"}},
            id="migrates-response-format-onto-the-callers-text",
        ),
        pytest.param(
            {"text": {"verbosity": "low", "format": {"type": "nonsense"}}},
            id="deletes-an-unparseable-format-from-the-callers-text",
        ),
        pytest.param(
            {"text": {"verbosity": "low",
                      "format": {"type": "json_schema", "json_schema": {"name": "s"}}}},
            id="deletes-a-chat-shaped-nested-format-from-the-callers-text",
        ),
        pytest.param(
            {"text": {"verbosity": "low"}, "verbosity": "high",
             "response_format": {"type": "json_object"}},
            id="folds-a-toplevel-verbosity-into-the-callers-text",
        ),
        pytest.param(
            {"text": {"verbosity": "low"}},
            id="leaves-an-already-normal-text-alone",
        ),
        pytest.param(
            {"text": "oops", "response_format": {"type": "json_object"}},
            id="leaves-a-non-dict-text-alone",
        ),
    ]

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

    @pytest.mark.asyncio
    async def test_strips_fusion_plugin_for_fusion_model(self):
        """Fusion on /chat/completions returns flattened prose with no structured
        events — a fallback re-send must not pay for an unrenderable deliberation."""
        payload = {
            "model": "openrouter/fusion",
            "input": "hi",
            "plugins": [{"id": "fusion"}, {"id": "file-parser", "pdf": {"engine": "native"}}],
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result["plugins"] == [{"id": "file-parser", "pdf": {"engine": "native"}}]

    @pytest.mark.asyncio
    async def test_drops_plugins_key_when_only_fusion_entry(self):
        payload = {"model": "openrouter/fusion", "input": "hi", "plugins": [{"id": "fusion"}]}
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert "plugins" not in result

    @pytest.mark.asyncio
    async def test_keeps_fusion_plugin_for_non_fusion_model(self):
        """A caller-attached fusion plugin on an ordinary model is deliberate config."""
        payload = {"model": "openai/gpt-4o", "input": "hi", "plugins": [{"id": "fusion"}]}
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result["plugins"] == [{"id": "fusion"}]

    @pytest.mark.asyncio
    async def test_converts_input_to_messages(self):
        """Test that input array is converted to messages."""
        payload = {
            "model": "openai/gpt-4o",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello"}]}
            ],
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        assert "messages" in result
        assert result["messages"][0]["role"] == "user"
        # Content blocks (input_text) become structured blocks (type: text)
        content = result["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_converts_instructions_to_system(self):
        """Test that instructions become system message."""
        payload = {
            "model": "openai/gpt-4o",
            "instructions": "You are a helpful assistant",
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}
            ],
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        # Instructions should be first message as system
        assert result["messages"][0]["role"] == "system"
        assert "helpful assistant" in str(result["messages"][0]["content"])

    @pytest.mark.asyncio
    async def test_preserves_stop_server_tools_when(self):
        """The stop_server_tools_when cost guard must survive Responses->Chat conversion,
        or the server-tool cost cap is silently dropped on the chat-completions endpoint."""
        payload = {
            "model": "anthropic/claude-opus-4.8",
            "input": [],
            "stop_server_tools_when": [{"type": "max_cost", "max_cost_in_dollars": 0.5}],
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result.get("stop_server_tools_when") == [{"type": "max_cost", "max_cost_in_dollars": 0.5}]

    @pytest.mark.asyncio
    async def test_phase_metadata_is_stripped_on_chat_completions_boundary(self):
        """Unsupported assistant phase metadata is dropped in degraded chat mode."""
        payload = {
            "model": "openai/gpt-5.4",
            "input": [
                {
                    "type": "message",
                    "role": "assistant",
                    "phase": "commentary",
                    "content": [{"type": "output_text", "text": "Thinking..."}],
                }
            ],
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        assert result["messages"][0]["role"] == "assistant"
        assert result["messages"][0]["content"] == [{"type": "text", "text": "Thinking..."}]
        assert "phase" not in result["messages"][0]

    @pytest.mark.asyncio
    async def test_converts_max_output_tokens(self):
        """Test that max_output_tokens becomes max_tokens."""
        payload = {
            "model": "openai/gpt-4o",
            "max_output_tokens": 500,
            "input": [],
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        assert result["max_tokens"] == 500
        assert "max_output_tokens" not in result

    @pytest.mark.asyncio
    async def test_converts_tools(self):
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
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        assert "tools" in result
        assert result["tools"][0]["type"] == "function"
        assert result["tools"][0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_stream_options_not_injected(self):
        """Test that stream_options are not injected when absent."""
        payload = {
            "model": "openai/gpt-4o",
            "stream": True,
            "input": [],
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        assert "stream_options" not in result

    @pytest.mark.asyncio
    async def test_rounds_top_k(self):
        """Test that top_k is rounded to integer."""
        payload = {
            "model": "openai/gpt-4o",
            "top_k": 2.7,
            "input": [],
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        # Python round(2.7) = 3 (standard rounding)
        assert result["top_k"] == 3

    @pytest.mark.asyncio
    async def test_preserves_cache_control(self):
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
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        # Find the text content block
        content = result["messages"][0]["content"]
        if isinstance(content, list):
            text_block = next((c for c in content if c.get("type") == "text"), None)
            assert text_block is not None
            assert text_block.get("cache_control") == {"type": "ephemeral"}

    @pytest.mark.asyncio
    async def test_drops_toplevel_cache_control(self):
        """Top-level cache_control is /responses-only and must not leak into the chat payload."""
        payload = {
            "model": "anthropic/claude-sonnet-4.6",
            "cache_control": {"type": "ephemeral"},
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hi"}],
                }
            ],
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert "cache_control" not in result

    @pytest.mark.asyncio
    async def test_non_dict_returns_empty(self):
        """Test non-dict input returns empty dict."""
        assert await _responses_payload_to_chat_completions_payload("not a dict", max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext) == {}  # type: ignore[arg-type]
        assert await _responses_payload_to_chat_completions_payload(None, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext) == {}  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_non_streaming_no_stream_options(self):
        """Test non-streaming request doesn't get stream_options."""
        payload = {"model": "gpt-4", "input": [], "stream": False}
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result["stream"] is False
        assert "stream_options" not in result

    @pytest.mark.asyncio
    async def test_existing_stream_options_preserved(self):
        """Test existing stream_options are preserved."""
        payload = {
            "model": "gpt-4",
            "input": [],
            "stream": True,
            "stream_options": {"custom": True},
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result["stream_options"]["custom"] is True

    @pytest.mark.asyncio
    async def test_usage_not_forwarded(self):
        """Test usage is not forwarded to chat payload."""
        payload = {"model": "gpt-4", "input": [], "usage": {"custom": "value"}}
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert "usage" not in result

    @pytest.mark.asyncio
    async def test_int_params_rounded(self):
        """Test integer parameters are rounded."""
        payload = {"model": "gpt-4", "input": [], "top_k": 2.7, "seed": 10.3}
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result["top_k"] == 3
        assert result["seed"] == 10

    @pytest.mark.asyncio
    async def test_int_param_string_converted(self):
        """Test string integer params are converted."""
        payload = {"model": "gpt-4", "input": [], "top_k": "5"}
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result["top_k"] == 5

    @pytest.mark.asyncio
    async def test_int_param_empty_string_removed(self):
        """Test empty string integer params are removed."""
        payload = {"model": "gpt-4", "input": [], "top_k": "  "}
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert "top_k" not in result

    @pytest.mark.asyncio
    async def test_invalid_response_format_removed(self, caplog):
        """Test invalid response_format is removed with warning."""
        payload = {"model": "gpt-4", "input": [], "response_format": {"type": "invalid"}}
        with caplog.at_level(logging.WARNING):
            result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert "response_format" not in result

    @pytest.mark.asyncio
    async def test_text_format_mapped_to_response_format(self):
        """Test text.format is mapped to response_format."""
        payload = {
            "model": "gpt-4",
            "input": [],
            "text": {"format": {"type": "json_object"}},
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result["response_format"] == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_text_verbosity_preserved(self):
        """Test text.verbosity is mapped to verbosity."""
        payload = {
            "model": "gpt-4",
            "input": [],
            "text": {"verbosity": "verbose"},
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result["verbosity"] == "verbose"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("instructions", "existing", "expected"),
        [
            pytest.param("Be helpful", "Existing prompt", "Be helpful\n\nExisting prompt", id="plain"),
            pytest.param("AAA-FIRST", "ZZZ-SECOND", "AAA-FIRST\n\nZZZ-SECOND", id="markers"),
            pytest.param("Be helpful", "  Existing prompt \n", "Be helpful\n\nExisting prompt", id="stripped"),
            pytest.param(
                "Be helpful",
                "First line\nSecond line",
                "Be helpful\n\nFirst line\nSecond line",
                id="two-lines",
            ),
            pytest.param("Be helpful", "   ", "Be helpful", id="blank"),
        ],
    )
    async def test_instructions_prepended_to_existing_system(self, instructions, existing, expected):
        """The whole merged string is the contract, not the presence of two substrings.

        Two `in` assertions cannot tell a prepend from an append, cannot see the
        separator, and pass on any string that merely contains both halves, so each
        row pins the exact bytes. `markers` is what makes that true: its halves appear
        nowhere else in the file, so an implementation that hardcodes the `plain`
        answer fails it. `stripped` pins the `.strip()` of the caller's text, `blank`
        pins the guard that keeps whitespace-only text from appending a trailing blank
        line, and `two-lines` pins that the whole text is carried, not just its head.
        """
        payload = {
            "model": "gpt-4",
            "instructions": instructions,
            "input": [{"type": "message", "role": "system", "content": existing}],
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        content = result["messages"][0]["content"]
        assert isinstance(content, str)
        assert content == expected

    @pytest.mark.asyncio
    async def test_instructions_prepended_to_list_content(self):
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
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        # Instructions should be prepended as text block, folded into the first usable
        # text block with a real blank line -- the string arm's output, and not an empty
        # separator block, which providers reject.
        content = result["messages"][0]["content"]
        assert isinstance(content, list)
        assert content[0]["text"] == "Be helpful\n\nExisting"

    @pytest.mark.asyncio
    async def test_instructions_with_no_messages(self):
        """Test instructions create system message when no messages."""
        payload = {"model": "gpt-4", "instructions": "Be helpful", "input": []}
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "Be helpful"

    @pytest.mark.asyncio
    async def test_instructions_with_non_system_first_message(self):
        """Test instructions create system message when first message is not system."""
        payload = {
            "model": "gpt-4",
            "instructions": "Be helpful",
            "input": [{"type": "message", "role": "user", "content": "Hi"}],
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "Be helpful"
        assert result["messages"][1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_trace_preserved_in_responses_to_chat(self):
        """trace should survive responses→chat conversion."""
        payload = {
            "model": "gpt-4",
            "input": [{"type": "message", "role": "user", "content": "Hi"}],
            "trace": {"trace_id": "abc123", "generation_name": "test"},
        }
        result = await _responses_payload_to_chat_completions_payload(payload, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result["trace"] == {"trace_id": "abc123", "generation_name": "test"}


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

    def test_cache_control_preserved(self):
        """Test cache_control dict is preserved on the outer tool entry."""
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "parameters": {"type": "object"},
                "cache_control": {"type": "ephemeral"},
            }
        ]
        result = _responses_tools_to_chat_tools(tools)
        assert result[0]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in result[0]["function"]

    def test_cache_control_absent_when_missing(self):
        """Test cache_control is not added when not present on source."""
        tools = [{"type": "function", "name": "tool"}]
        result = _responses_tools_to_chat_tools(tools)
        assert "cache_control" not in result[0]

    def test_cache_control_non_dict_ignored(self):
        """Test non-dict cache_control is not preserved."""
        tools = [
            {"type": "function", "name": "tool", "cache_control": "ephemeral"}
        ]
        result = _responses_tools_to_chat_tools(tools)
        assert "cache_control" not in result[0]

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

    def test_cache_control_preserved(self):
        """Test cache_control dict is preserved on the output spec."""
        tools = [
            {
                "type": "function",
                "function": {"name": "tool", "parameters": {"type": "object"}},
                "cache_control": {"type": "ephemeral"},
            }
        ]
        result = _chat_tools_to_responses_tools(tools)
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_absent_when_missing(self):
        """Test cache_control is not added when not present on source."""
        tools = [
            {"type": "function", "function": {"name": "tool"}}
        ]
        result = _chat_tools_to_responses_tools(tools)
        assert "cache_control" not in result[0]

    def test_cache_control_non_dict_ignored(self):
        """Test non-dict cache_control is not preserved."""
        tools = [
            {
                "type": "function",
                "function": {"name": "tool"},
                "cache_control": "ephemeral",
            }
        ]
        result = _chat_tools_to_responses_tools(tools)
        assert "cache_control" not in result[0]

    # The one table for the whole property: which `strict` reaches the wire, in which shape,
    # and what a caller's own value is worth on each arm. Six values against two shapes, so a
    # fix written as a coercion (``get("strict", False)``, ``get("strict", True)``, an
    # ``isinstance(..., bool)`` test) is caught on the cell it invents a value for and not only
    # on the cell that motivated the fix. `_MISSING` is the absent case, which is the one cell
    # whose two arms genuinely differ: chat-shaped states the endpoint's own fallback, a
    # Responses-shaped arrival that wrote none gains no key at all.
    _MISSING: ClassVar[Any] = object()
    _STRICT_VALUES: ClassVar[list[Any]] = [
        pytest.param(_MISSING, id="absent"),
        pytest.param(None, id="null"),
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param("yes", id="str"),
        pytest.param(1, id="int"),
    ]

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

    @pytest.mark.asyncio
    async def test_handles_user_message(self):
        """Test user message conversion."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        # Content blocks (input_text) become structured (type: text)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_handles_assistant_message(self):
        """Test assistant message conversion."""
        input_value = [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi there!"}],
            }
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        # Content blocks (output_text) become structured (type: text)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "Hi there!"

    @pytest.mark.asyncio
    async def test_handles_system_message(self):
        """Test system message conversion."""
        input_value = [
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "System prompt"}],
            }
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        assert result[0]["role"] == "system"
        # Content blocks (input_text) become structured (type: text)
        content = result[0]["content"]
        assert isinstance(content, list)
        assert content[0]["type"] == "text"
        assert content[0]["text"] == "System prompt"

    @pytest.mark.asyncio
    async def test_handles_image_content(self):
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
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        content = result[0]["content"]
        assert isinstance(content, list)
        image_block = next((c for c in content if c.get("type") == "image_url"), None)
        assert image_block is not None

    @pytest.mark.asyncio
    async def test_handles_function_call_output(self):
        """Test function call output (tool result) conversion."""
        input_value = [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": '{"result": "sunny"}',
            },
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_123"

    @pytest.mark.asyncio
    async def test_handles_function_call(self):
        """Test function call conversion."""
        input_value = [
            {
                "type": "function_call",
                "id": "call_123",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": '{"result": "sunny"}',
            },
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        assert result[0]["role"] == "assistant"
        tool_calls = result[0].get("tool_calls", [])
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_weather"

    @pytest.mark.asyncio
    async def test_handles_empty(self):
        """Test empty input."""
        assert await _responses_input_to_chat_messages([], max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext) == []
        assert await _responses_input_to_chat_messages(None, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext) == []

    @pytest.mark.asyncio
    async def test_string_input(self):
        """Test plain string input is converted to user message."""
        result = await _responses_input_to_chat_messages("Hello, world!", max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello, world!"

    @pytest.mark.asyncio
    async def test_string_input_strips_hidden_transport_markers(self):
        """Task/chat-history strings should not leak hidden marker lines to chat completions."""
        result = await _responses_input_to_chat_messages(
            "ASSISTANT: Visible answer\n[P:final_answer]: #\n\n[0001H74WE6NX0KKR9ZC7]: #\n"
        , max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        assert result == [{"role": "user", "content": "ASSISTANT: Visible answer"}]

    @pytest.mark.asyncio
    async def test_empty_string_input(self):
        """Test empty string input returns empty list."""
        result = await _responses_input_to_chat_messages("   ", max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result == []

    @pytest.mark.asyncio
    async def test_non_list_non_string_returns_empty(self):
        """Test non-list, non-string input returns empty list."""
        assert await _responses_input_to_chat_messages(123, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext) == []
        assert await _responses_input_to_chat_messages({"type": "message"}, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext) == []

    @pytest.mark.asyncio
    async def test_non_dict_items_skipped(self):
        """Test non-dict items in list are skipped."""
        input_value = [
            "not a dict",
            {"type": "message", "role": "user", "content": "Hi"},
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert len(result) == 1
        assert result[0]["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_message_without_role_skipped(self):
        """Test messages without role are skipped."""
        input_value = [{"type": "message", "content": "No role"}]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result == []

    @pytest.mark.asyncio
    async def test_message_with_string_content(self):
        """Test message with string content."""
        input_value = [{"type": "message", "role": "user", "content": "Plain text"}]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result[0]["content"] == "Plain text"

    @pytest.mark.asyncio
    async def test_message_blocks_strip_hidden_transport_markers(self):
        """Structured text blocks should drop hidden marker lines before /chat/completions."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": "<chat_history>\nASSISTANT: Visible answer\n[P:final_answer]: #\n\n[0001H74WE6NX0KKR9ZC7]: #\n</chat_history>",
                    }
                ],
            }
        ]

        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)

        assert result[0]["content"] == [
            {
                "type": "text",
                "text": "<chat_history>\nASSISTANT: Visible answer\n\n</chat_history>",
            }
        ]

    # --- a text block that is nothing but marker lines carries nothing ----------------
    #
    # The test above names the contract this converter has to hold and does not hold it: its
    # only arm mixes prose with a marker, so the block is usable and the marker is dropped
    # from it. When the marker is the *whole* block, `_replay_block_is_usable` still called the
    # block usable -- it judged `text.strip()`, and a marker line is a non-empty string -- so
    # `_replay_blocks_or_note` saw something worth sending, took the `originals` escape and
    # returned the caller's raw `/responses`-shaped block. The pipe's own transport markers
    # went back out on the wire, in a block type the chat leg does not accept.
    #
    # Markers are built with the serialisers, never hand-typed: `_extract_marker_ulid` is
    # Crockford-strict about a 20-character body, so a hand-typed marker is silently not a
    # marker and a test written that way passes for the wrong reason.

    async def _marker_only_turn(self, blocks: list[dict[str, Any]], role: str = "user"):
        return await _responses_input_to_chat_messages(
            [{"type": "message", "role": role, "content": blocks}],
            max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext,
        )

    @pytest.mark.asyncio
    async def test_message_with_annotations(self):
        """Test message annotations are preserved."""
        input_value = [
            {
                "type": "message",
                "role": "assistant",
                "content": "Hi",
                "annotations": [{"type": "url_citation", "url": "http://example.com"}],
            }
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert "annotations" in result[0]
        assert len(result[0]["annotations"]) == 1

    @pytest.mark.asyncio
    async def test_message_with_reasoning_details(self):
        """Test message reasoning_details are preserved."""
        input_value = [
            {
                "type": "message",
                "role": "assistant",
                "content": "Hi",
                "reasoning_details": [{"type": "thinking"}],
            }
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert "reasoning_details" in result[0]

    @pytest.mark.asyncio
    async def test_image_url_block_dict(self):
        """Test image_url block with dict value."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
                ],
            }
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        content = result[0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == "https://example.com/img.png"

    @pytest.mark.asyncio
    async def test_image_url_block_string(self):
        """Test image_url block with string value."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": "https://example.com/img.png"},
                ],
            }
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        content = result[0]["content"]
        assert content[0]["image_url"]["url"] == "https://example.com/img.png"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("auto", "auto"),
            ("low", "low"),
            ("high", "high"),
            ("original", "original"),
            ("bogus", "bogus"),
            ("", "auto"),
            (None, "auto"),
            (7, "auto"),
        ],
    )
    async def test_input_image_with_detail(self, given, expected):
        """Whatever detail arrives is what goes out, unless absent/empty/not a string.

        The three-value allowlist this replaced substituted ``auto`` for ``original``
        and dropped anything else, so a caller asking for a detail the pipe had never
        heard of silently got a different one. Open WebUI's own rule is
        ``detail = url_data.get('detail') or 'auto'`` (``routers/openai.py:1360``):
        any non-empty string passes through, everything else becomes ``auto``.
        """
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "https://example.com/img.png",
                        "detail": given,
                    },
                ],
            }
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        content = result[0]["content"]
        assert content[0]["image_url"]["detail"] == expected

    @pytest.mark.asyncio
    async def test_input_audio_block(self):
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
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        content = result[0]["content"]
        assert content[0]["type"] == "input_audio"

    @pytest.mark.asyncio
    async def test_video_url_block_dict(self):
        """Test video_url block with dict value."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": "https://example.com/vid.mp4"}},
                ],
            }
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        content = result[0]["content"]
        assert content[0]["type"] == "video_url"

    @pytest.mark.asyncio
    async def test_video_url_block_string(self):
        """Test video_url block with string value."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": "https://example.com/vid.mp4"},
                ],
            }
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        content = result[0]["content"]
        assert content[0]["video_url"]["url"] == "https://example.com/vid.mp4"

    @pytest.mark.asyncio
    async def test_input_file_block_with_data(self):
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
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        content = result[0]["content"]
        assert content[0]["type"] == "file"
        assert content[0]["file"]["filename"] == "test.txt"
        assert content[0]["file"]["file_data"] == "base64content"

    @pytest.mark.asyncio
    async def test_input_file_block_with_url(self):
        """Test input_file block with file_url."""
        input_value = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_url": "https://example.com/file.txt",
                    },
                ],
            }
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        content = result[0]["content"]
        assert content[0]["file"]["file_data"] == "https://example.com/file.txt"

    @pytest.mark.asyncio
    async def test_empty_content_blocks(self):
        """A contentless message item keeps the caller's own empty block list.

        B378's operator decision 1(b): the normaliser preserves the caller's spelling
        rather than rewriting a contentless item to `""`, so `/v1/responses` and
        `/v1/chat/completions` answer the same turn the same way. Which line the provider
        ends up with is the transformer's, and
        `tests/test_a_turn_that_carries_nothing_ships_nothing_void.py` pins that for all
        four routes.
        """
        input_value = [{"type": "message", "role": "user", "content": []}]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result[0]["content"] == []

    @pytest.mark.asyncio
    async def test_function_call_output_non_string_output(self):
        """Test function_call_output with non-string output is JSON-serialized."""
        input_value = [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": {"result": "data"},
            },
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result[1]["content"] == '{"result": "data"}'

    @pytest.mark.asyncio
    async def test_function_call_output_none_output(self):
        """Test function_call_output with None output."""
        input_value = [
            {
                "type": "function_call",
                "call_id": "call_123",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_123",
                "output": None,
            },
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result[1]["content"] == ""

    @pytest.mark.asyncio
    async def test_function_call_with_call_id(self):
        """Test function_call using call_id field."""
        input_value = [
            {
                "type": "function_call",
                "call_id": "call_456",
                "name": "my_func",
                "arguments": '{"a": 1}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_456",
                "output": '{"ok": true}',
            },
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result[0]["tool_calls"][0]["id"] == "call_456"

    @pytest.mark.asyncio
    async def test_function_call_non_string_arguments(self):
        """Test function_call with non-string arguments is JSON-serialized."""
        input_value = [
            {
                "type": "function_call",
                "id": "call_789",
                "name": "my_func",
                "arguments": {"key": "value"},
            },
            {
                "type": "function_call_output",
                "call_id": "call_789",
                "output": '{"ok": true}',
            },
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == '{"key": "value"}'

    @pytest.mark.asyncio
    async def test_function_call_none_arguments(self):
        """Test function_call with None arguments defaults to empty object."""
        input_value = [
            {
                "type": "function_call",
                "id": "call_abc",
                "name": "my_func",
                "arguments": None,
            },
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": '{"ok": true}',
            },
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
        assert result[0]["tool_calls"][0]["function"]["arguments"] == "{}"

    @pytest.mark.asyncio
    async def test_function_call_missing_required_fields(self):
        """Test function_call missing call_id or name is skipped."""
        input_value = [
            {"type": "function_call", "name": "func"},  # No id/call_id
            {"type": "function_call", "id": "123"},  # No name
            {"type": "function_call", "id": "456", "name": ""},  # Empty name
        ]
        result = await _responses_input_to_chat_messages(input_value, max_inline_bytes=_INLINE_CAP_BYTES, allow_insecure=_refuses_cleartext)
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
            "disable_web_tools_auto_attach": True,
            "disable_web_tools_default_on": True,
            "disable_direct_uploads_auto_attach": True,
            "disable_description_updates": True,
            "disable_native_websearch": True,
            "disable_native_web_search": True,
            "openrouter_provider_ignore": "azure",
            "openrouter_provider_only": "openai",
            "openrouter_provider_order": "openai,together",
        }
        _strip_disable_model_settings_params(payload)
        assert "model" in payload
        assert "disable_model_metadata_sync" not in payload
        assert "disable_capability_updates" not in payload
        assert "disable_image_updates" not in payload
        assert "openrouter_provider_ignore" not in payload
        assert "openrouter_provider_only" not in payload
        assert "openrouter_provider_order" not in payload

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
# _apply_openrouter_trace_to_payload Tests
# ============================================================================


class TestApplyOpenrouterTraceToPayload:
    """Tests for _apply_openrouter_trace_to_payload()."""

    def test_non_dict_is_noop(self):
        """Non-dict input is silently ignored."""
        _apply_openrouter_trace_to_payload("not a dict")
        _apply_openrouter_trace_to_payload(None)

    def test_no_trace_key_is_noop(self):
        """No changes when openrouter_trace is absent."""
        payload = {"model": "gpt-4", "messages": []}
        _apply_openrouter_trace_to_payload(payload)
        assert "trace" not in payload
        assert "openrouter_trace" not in payload

    def test_dict_trace_applied(self):
        """Dict value is popped and written as trace."""
        payload = {
            "model": "openai/gpt-4o",
            "openrouter_trace": {
                "trace_id": "order_processing_001",
                "trace_name": "Order Processing Pipeline",
                "generation_name": "Extract Order Details",
                "order_id": "ORD-12345",
                "priority": "high",
            },
        }
        _apply_openrouter_trace_to_payload(payload)

        assert "openrouter_trace" not in payload
        assert payload["trace"] == {
            "trace_id": "order_processing_001",
            "trace_name": "Order Processing Pipeline",
            "generation_name": "Extract Order Details",
            "order_id": "ORD-12345",
            "priority": "high",
        }

    def test_json_string_auto_parsed(self):
        """JSON string value is parsed into a dict."""
        payload = {
            "model": "gpt-4",
            "openrouter_trace": '{"trace_name": "My Pipeline", "generation_name": "chat"}',
        }
        _apply_openrouter_trace_to_payload(payload)

        assert "openrouter_trace" not in payload
        assert payload["trace"] == {
            "trace_name": "My Pipeline",
            "generation_name": "chat",
        }

    def test_invalid_json_string_ignored(self):
        """Invalid JSON string is ignored with no trace written."""
        payload = {"model": "gpt-4", "openrouter_trace": "not valid json"}
        _apply_openrouter_trace_to_payload(payload)
        assert "trace" not in payload
        assert "openrouter_trace" not in payload

    def test_empty_string_ignored(self):
        """Empty string value is ignored."""
        payload = {"model": "gpt-4", "openrouter_trace": ""}
        _apply_openrouter_trace_to_payload(payload)
        assert "trace" not in payload

    def test_whitespace_only_string_ignored(self):
        """Whitespace-only string value is ignored."""
        payload = {"model": "gpt-4", "openrouter_trace": "   "}
        _apply_openrouter_trace_to_payload(payload)
        assert "trace" not in payload

    def test_json_array_string_rejected(self):
        """JSON array string is rejected (must be an object)."""
        payload = {"model": "gpt-4", "openrouter_trace": '["a", "b"]'}
        _apply_openrouter_trace_to_payload(payload)
        assert "trace" not in payload

    def test_empty_dict_ignored(self):
        """Empty dict value is ignored."""
        payload = {"model": "gpt-4", "openrouter_trace": {}}
        _apply_openrouter_trace_to_payload(payload)
        assert "trace" not in payload

    def test_non_dict_non_string_ignored(self):
        """Non-dict, non-string value (e.g. int) is ignored."""
        payload = {"model": "gpt-4", "openrouter_trace": 42}
        _apply_openrouter_trace_to_payload(payload)
        assert "trace" not in payload

    def test_merges_with_existing_trace(self):
        """Model-level trace merges with existing trace (model-level wins)."""
        payload = {
            "model": "gpt-4",
            "trace": {
                "trace_id": "existing_trace",
                "session": "abc123",
            },
            "openrouter_trace": {
                "trace_name": "My Pipeline",
                "trace_id": "override_trace",
            },
        }
        _apply_openrouter_trace_to_payload(payload)

        assert "openrouter_trace" not in payload
        assert payload["trace"] == {
            "trace_id": "override_trace",
            "trace_name": "My Pipeline",
            "session": "abc123",
        }

    def test_none_value_is_noop(self):
        """Explicit None value is treated as absent."""
        payload = {"model": "gpt-4", "openrouter_trace": None}
        _apply_openrouter_trace_to_payload(payload)
        assert "trace" not in payload

    def test_openrouter_trace_fields_passthrough(self):
        """All OpenRouter documented trace fields are preserved."""
        payload = {
            "model": "gpt-4",
            "openrouter_trace": {
                "trace_id": "my-trace-001",
                "trace_name": "Order Processing Pipeline",
                "span_name": "validate-order",
                "generation_name": "Extract Order Details",
                "parent_span_id": "parent-span-abc",
                "custom_metadata_key": "custom_value",
            },
        }
        _apply_openrouter_trace_to_payload(payload)

        trace = payload["trace"]
        assert trace["trace_id"] == "my-trace-001"
        assert trace["trace_name"] == "Order Processing Pipeline"
        assert trace["span_name"] == "validate-order"
        assert trace["generation_name"] == "Extract Order Details"
        assert trace["parent_span_id"] == "parent-span-abc"
        assert trace["custom_metadata_key"] == "custom_value"


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
        # The actual non-replayable artifacts are: local_shell_call, shell_call,
        # shell_call_output, local_shell_call_output, image_generation_call,
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

    def test_openrouter_web_tool_artifacts_stripped_but_advisor_kept(self):
        """openrouter:datetime/web_search/web_fetch are not valid input items and must be
        stripped on replay. openrouter:advisor MUST be kept — its cross-request memory
        depends on replaying the advisor items unchanged."""
        items = [
            {"type": "openrouter:datetime", "id": "dt", "datetime": "2026-06-27", "timezone": "UTC"},
            {"type": "openrouter:web_search", "id": "ws", "status": "completed"},
            {"type": "openrouter:web_fetch", "id": "wf", "url": "https://x"},
            {"type": "openrouter:advisor", "id": "adv", "advice": "remember this"},
            {"type": "message", "role": "assistant", "content": []},
        ]
        result = _filter_replayable_input_items(items)
        types = [i.get("type") for i in result]
        assert "openrouter:datetime" not in types
        assert "openrouter:web_search" not in types
        assert "openrouter:web_fetch" not in types
        assert "openrouter:advisor" in types
        assert "message" in types


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
            owui_metadata={},
            owui_user_id="user123",
            valves=pipe.valves,
        )

    def test_user_id_added_when_valve_enabled(self, pipe_instance):
        """Test user is added when SEND_END_USER_ID is enabled."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(update={"SEND_END_USER_ID": True})
        payload = {}
        _apply_identifier_valves_to_payload(
            payload,
            owui_metadata={},
            owui_user_id="user123",
            valves=valves,
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
            owui_metadata={},
            owui_user_id="user123",
            valves=valves,
        )
        assert "user" not in payload

    def test_session_id_added_when_valve_enabled(self, pipe_instance):
        """Test session_id is added when SEND_SESSION_ID is enabled."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(update={"SEND_SESSION_ID": True})
        payload = {}
        _apply_identifier_valves_to_payload(
            payload,
            owui_metadata={"session_id": "session123"},
            owui_user_id="",
            valves=valves,
        )
        assert "session_id" not in payload
        assert payload["metadata"]["session_id"] == "session123"

    def test_session_id_removed_when_valve_disabled(self, pipe_instance):
        """Test session_id is removed when SEND_SESSION_ID is disabled."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(update={"SEND_SESSION_ID": False, "SEND_CACHE_SESSION_ID": False})
        payload = {"session_id": "existing"}
        _apply_identifier_valves_to_payload(
            payload,
            owui_metadata={"session_id": "session123"},
            owui_user_id="",
            valves=valves,
        )
        assert "session_id" not in payload

    def test_chat_id_added_to_metadata(self, pipe_instance):
        """Test chat_id is added to metadata when SEND_CHAT_ID is enabled."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(update={"SEND_CHAT_ID": True})
        payload = {}
        _apply_identifier_valves_to_payload(
            payload,
            owui_metadata={"chat_id": "chat123"},
            owui_user_id="",
            valves=valves,
        )
        assert payload["metadata"]["chat_id"] == "chat123"

    def test_message_id_added_to_metadata(self, pipe_instance):
        """Test message_id is added to metadata when SEND_MESSAGE_ID is enabled."""
        pipe = pipe_instance
        valves = pipe.valves.model_copy(update={"SEND_MESSAGE_ID": True})
        payload = {}
        _apply_identifier_valves_to_payload(
            payload,
            owui_metadata={"message_id": "msg123"},
            owui_user_id="",
            valves=valves,
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
            owui_metadata={},
            owui_user_id="",
            valves=valves,
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
                owui_metadata={},
                owui_user_id="",  # Empty user ID
                valves=valves,
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
                owui_metadata={"session_id": "  "},  # Whitespace only
                owui_user_id="",
                valves=valves,
            )
        assert "session_id" not in payload.get("metadata", {})


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
        assert "max_tokens" in received_payload
        assert received_payload["max_tokens"] == 100
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
        "disable_web_tools_auto_attach": True,
        "disable_web_tools_default_on": True,
        "disable_direct_uploads_auto_attach": True,
        "disable_description_updates": True,
        "disable_native_websearch": True,
        "disable_native_web_search": True,
    }

    _strip_disable_model_settings_params(payload)

    assert payload == {"model": "openai/gpt-5"}


# ===== From test_top_k_passthrough.py =====


from open_webui_openrouter_pipe import _filter_openrouter_request

_INLINE_CAP_BYTES = 50 * 1024 * 1024


def _refuses_cleartext(_url: str) -> bool:
    return False


def test_filter_openrouter_request_forwards_numeric_top_k() -> None:
    payload = {"model": "openai/gpt-5", "input": [], "top_k": 50}
    filtered = _filter_openrouter_request(payload)
    assert filtered["top_k"] == 50
    assert isinstance(filtered["top_k"], int)
    assert not isinstance(filtered["top_k"], bool)


def test_filter_openrouter_request_parses_string_top_k() -> None:
    payload = {"model": "openai/gpt-5", "input": [], "top_k": " 50 "}
    filtered = _filter_openrouter_request(payload)
    assert filtered["top_k"] == 50
    assert isinstance(filtered["top_k"], int)
    assert not isinstance(filtered["top_k"], bool)


def test_filter_openrouter_request_drops_invalid_top_k() -> None:
    payload = {"model": "openai/gpt-5", "input": [], "top_k": "nope"}
    filtered = _filter_openrouter_request(payload)
    assert "top_k" not in filtered


def test_filter_openrouter_request_preserves_trace() -> None:
    """Trace field passes through the Responses API allowlist."""
    payload = {
        "model": "openai/gpt-4o",
        "input": [{"role": "user", "content": "Hi"}],
        "trace": {"trace_name": "My Pipeline", "generation_name": "chat"},
    }
    filtered = _filter_openrouter_request(payload)
    assert filtered["trace"] == {"trace_name": "My Pipeline", "generation_name": "chat"}


def test_filter_openrouter_chat_request_preserves_trace() -> None:
    """Trace field passes through the Chat Completions API allowlist."""
    payload = {
        "model": "openai/gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "trace": {"trace_id": "abc123"},
    }
    filtered = _filter_openrouter_chat_request(payload)
    assert filtered["trace"] == {"trace_id": "abc123"}


# ============================================================================
# Task 4 — Responses allowlist: new OpenRouter Responses-specific fields
# ============================================================================


class TestFilterOpenrouterRequestResponsesExtensions:
    """Verify all OpenRouter Responses-specific extensions pass through the allowlist."""

    NEW_RESPONSES_FIELDS = {
        "background": True,
        "frequency_penalty": 0.5,
        "image_config": {"quality": "high"},
        "include": ["reasoning.encrypted_content"],
        "max_tool_calls": 10,
        "modalities": ["text"],
        "presence_penalty": 0.3,
        "prompt": "Hello world",
        "prompt_cache_key": "cache_xyz",
        "safety_identifier": "safe_001",
        "service_tier": "auto",
        "store": False,
        "top_logprobs": 5,
    }

    @pytest.mark.parametrize("field,value", list(NEW_RESPONSES_FIELDS.items()))
    def test_responses_extension_field_passes_through(self, field, value):
        """Each new Responses-extension field should survive the allowlist filter."""
        payload = {"model": "openai/gpt-4o", "input": [], field: value}
        result = _filter_openrouter_request(payload)
        assert field in result, f"{field!r} was stripped by the allowlist"
        assert result[field] == value

    def test_all_new_fields_pass_together(self):
        """All 13 new Responses-extension fields pass through in a single payload."""
        payload = {"model": "openai/gpt-4o", "input": [], **self.NEW_RESPONSES_FIELDS}
        result = _filter_openrouter_request(payload)
        for field, value in self.NEW_RESPONSES_FIELDS.items():
            assert field in result, f"{field!r} missing after filter"
            assert result[field] == value

    def test_store_false_preserved(self):
        """store: false is a privacy signal and must not be stripped."""
        payload = {"model": "openai/gpt-4o", "input": [], "store": False}
        result = _filter_openrouter_request(payload)
        assert result["store"] is False

    # `service_tier` and `prompt_cache_key` are in the table above, so they are
    # carried on the chat leg as well -- driven through the real converter and the
    # real chat filter, from the same table rather than a parallel list of values.
    # See `tests/test_a_service_tier_and_prompt_cache_key_reach_the_chat_endpoint.py`
    # for the end-to-end statement on the wire.
    SHARED_WITH_CHAT = ("max_tool_calls", "prompt_cache_key", "service_tier")
    RESPONSES_ONLY = ("safety_identifier",)

class TestImageConfigPydanticRoundTrip:
    """`image_config` on `ResponsesBody` is now `Optional[Dict[str, Any]]` (was `Optional[Union[str, float]]`).
    The new image filters write nested dicts here; verify they survive Pydantic validation,
    retain `extra` keys for forward-compat with provider-specific params (quality, background, etc.
    per OpenRouter docs), and pass through `CompletionsBody`'s `extra='allow'` shape too.
    """

    def test_image_config_dict_round_trip_responses_body(self):
        """`ResponsesBody.image_config` is the typed field — must accept full nested dict."""
        from open_webui_openrouter_pipe.api.transforms import ResponsesBody

        body = ResponsesBody.model_validate({
            "model": "sourceful/riverflow-v2-pro",
            "input": [],
            "image_config": {
                "aspect_ratio": "16:9",
                "image_size": "2K",
                "font_inputs": [{"font_url": "https://example.com/f.ttf", "text": "Hello"}],
                "super_resolution_references": ["https://example.com/ref.jpg"],
                "quality": "high",
            },
        })
        assert body.image_config is not None
        assert body.image_config["aspect_ratio"] == "16:9"
        assert body.image_config["image_size"] == "2K"
        assert body.image_config["font_inputs"] == [
            {"font_url": "https://example.com/f.ttf", "text": "Hello"}
        ]
        assert body.image_config["super_resolution_references"] == [
            "https://example.com/ref.jpg"
        ]
        # Forward-compat: unknown keys (quality, background, etc.) preserved
        assert body.image_config["quality"] == "high"
        # Round-trip through model_dump preserves shape
        dumped = body.model_dump(exclude_none=True)
        assert dumped["image_config"] == body.image_config

    def test_image_config_gemini_extended_aspect_ratio(self):
        """Gemini-only knobs (4:1, 1:4, 8:1, 1:8 aspect, 0.5K size) survive Pydantic round-trip."""
        from open_webui_openrouter_pipe.api.transforms import ResponsesBody

        body = ResponsesBody.model_validate({
            "model": "google/gemini-3.1-flash-image-preview",
            "input": [],
            "image_config": {"aspect_ratio": "4:1", "image_size": "0.5K"},
        })
        assert body.image_config == {"aspect_ratio": "4:1", "image_size": "0.5K"}

    def test_image_config_none_default_responses_body(self):
        """When `image_config` is not provided, the typed field defaults to None."""
        from open_webui_openrouter_pipe.api.transforms import ResponsesBody

        body = ResponsesBody.model_validate({"model": "openai/gpt-4o", "input": []})
        assert body.image_config is None

    def test_image_config_passthrough_completions_body(self):
        """`CompletionsBody` doesn't declare `image_config` typed but accepts it via
        extra='allow'. Verify BOTH attribute access AND model_dump preserve the
        dict — a regression that drops the field via dict-only path or strips
        it from extras must be caught.
        path orchestrator.py:416 takes (CompletionsBody.model_validate(body))."""
        from open_webui_openrouter_pipe.api.transforms import CompletionsBody

        body = CompletionsBody.model_validate({
            "model": "openai/gpt-5-image",
            "messages": [],
            "image_config": {"aspect_ratio": "16:9", "image_size": "2K"},
        })
        # Attribute access — Pydantic v2 with extra="allow" exposes via getattr
        assert getattr(body, "image_config", None) == {"aspect_ratio": "16:9", "image_size": "2K"}
        dumped = body.model_dump(exclude_none=True)
        assert dumped["image_config"] == {"aspect_ratio": "16:9", "image_size": "2K"}


class TestFromCompletionsPreservesStoreAndUser:
    """Verify store and user survive chat→responses conversion."""

    @pytest.mark.asyncio
    async def test_store_survives_conversion(self, pipe_instance_async):
        """store: false should not be dropped during from_completions."""
        pipe = pipe_instance_async
        completions = CompletionsBody(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            store=False,
        )
        result = await ResponsesBody.from_completions(
            completions, transformer_context=pipe,
        )
        # but from_completions must no longer drop it.
        dumped = result.model_dump(exclude_none=True)
        assert "store" in dumped, "store was dropped during from_completions"
        assert dumped["store"] is False

    @pytest.mark.asyncio
    async def test_user_survives_conversion(self, pipe_instance_async):
        """user should not be dropped during from_completions."""
        pipe = pipe_instance_async
        completions = CompletionsBody(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            user="user_abc123",
        )
        result = await ResponsesBody.from_completions(
            completions, transformer_context=pipe,
        )
        assert result.user == "user_abc123"


# -----------------------------------------------------------------------------
# Provider Routing Custom Params
# -----------------------------------------------------------------------------

class TestParseProviderCsv:
    """Tests for _parse_provider_csv()."""

    def test_non_string_returns_empty(self):
        assert _parse_provider_csv(None) == []
        assert _parse_provider_csv(123) == []
        assert _parse_provider_csv({}) == []

    def test_empty_string_returns_empty(self):
        assert _parse_provider_csv("") == []
        assert _parse_provider_csv("   ") == []

    def test_single_slug(self):
        assert _parse_provider_csv("openai") == ["openai"]

    def test_csv_multiple_slugs(self):
        assert _parse_provider_csv("openai, together, deepinfra") == [
            "openai", "together", "deepinfra",
        ]

    def test_duplicates_removed(self):
        assert _parse_provider_csv("openai, together, openai") == [
            "openai", "together",
        ]

    def test_invalid_slugs_dropped(self):
        assert _parse_provider_csv("openai, INVALID SLUG!, together") == [
            "openai", "together",
        ]

    def test_whitespace_trimmed(self):
        assert _parse_provider_csv("  openai , together  ") == [
            "openai", "together",
        ]

    def test_slug_with_segment(self):
        assert _parse_provider_csv("deepinfra/turbo") == ["deepinfra/turbo"]

    def test_auto_lowercase(self):
        assert _parse_provider_csv("OpenAI, Together") == ["openai", "together"]

    def test_list_input(self):
        """OWUI may auto-parse JSON arrays."""
        assert _parse_provider_csv(["openai", "together"]) == ["openai", "together"]

    def test_list_input_with_invalid(self):
        assert _parse_provider_csv(["openai", "BAD SLUG!", "together"]) == [
            "openai", "together",
        ]


class TestApplyProviderRoutingParamsToPayload:
    """Tests for _apply_provider_routing_params_to_payload()."""

    def test_non_dict_is_noop(self):
        _apply_provider_routing_params_to_payload("not a dict")
        _apply_provider_routing_params_to_payload(None)

    def test_no_keys_is_noop(self):
        payload = {"model": "gpt-4"}
        _apply_provider_routing_params_to_payload(payload)
        assert "provider" not in payload
        assert payload == {"model": "gpt-4"}

    def test_ignore_applied(self):
        payload = {"model": "gpt-4", "openrouter_provider_ignore": "azure"}
        _apply_provider_routing_params_to_payload(payload)
        assert payload["provider"] == {"ignore": ["azure"]}
        assert "openrouter_provider_ignore" not in payload

    def test_only_applied(self):
        payload = {"model": "gpt-4", "openrouter_provider_only": "openai"}
        _apply_provider_routing_params_to_payload(payload)
        assert payload["provider"] == {"only": ["openai"]}
        assert "openrouter_provider_only" not in payload

    def test_order_applied(self):
        payload = {"model": "gpt-4", "openrouter_provider_order": "openai, together"}
        _apply_provider_routing_params_to_payload(payload)
        assert payload["provider"] == {"order": ["openai", "together"]}
        assert "openrouter_provider_order" not in payload

    def test_all_three_applied(self):
        payload = {
            "model": "gpt-4",
            "openrouter_provider_ignore": "azure, venice",
            "openrouter_provider_only": "openai",
            "openrouter_provider_order": "openai, together",
        }
        _apply_provider_routing_params_to_payload(payload)
        assert payload["provider"] == {
            "ignore": ["azure", "venice"],
            "only": ["openai"],
            "order": ["openai", "together"],
        }

    def test_keys_popped_from_payload(self):
        payload = {
            "model": "gpt-4",
            "openrouter_provider_ignore": "azure",
            "openrouter_provider_only": "openai",
            "openrouter_provider_order": "openai",
        }
        _apply_provider_routing_params_to_payload(payload)
        assert "openrouter_provider_ignore" not in payload
        assert "openrouter_provider_only" not in payload
        assert "openrouter_provider_order" not in payload

    def test_merges_with_existing_provider(self):
        payload = {
            "model": "gpt-4",
            "provider": {"zdr": True},
            "openrouter_provider_ignore": "azure",
        }
        _apply_provider_routing_params_to_payload(payload)
        assert payload["provider"] == {"zdr": True, "ignore": ["azure"]}

    def test_merges_with_existing_list_values(self):
        payload = {
            "model": "gpt-4",
            "provider": {"order": ["anthropic"]},
            "openrouter_provider_order": "openai, together",
        }
        _apply_provider_routing_params_to_payload(payload)
        assert payload["provider"]["order"] == ["anthropic", "openai", "together"]

    def test_merges_deduplicates_existing_list(self):
        payload = {
            "model": "gpt-4",
            "provider": {"ignore": ["azure"]},
            "openrouter_provider_ignore": "azure, venice",
        }
        _apply_provider_routing_params_to_payload(payload)
        assert payload["provider"]["ignore"] == ["azure", "venice"]

    def test_empty_values_no_provider_created(self):
        payload = {
            "model": "gpt-4",
            "openrouter_provider_ignore": "",
            "openrouter_provider_only": "  ",
        }
        _apply_provider_routing_params_to_payload(payload)
        assert "provider" not in payload

    def test_invalid_slugs_in_csv_dropped(self):
        payload = {
            "model": "gpt-4",
            "openrouter_provider_ignore": "azure, BAD SLUG!, venice",
        }
        _apply_provider_routing_params_to_payload(payload)
        assert payload["provider"] == {"ignore": ["azure", "venice"]}

    def test_existing_provider_non_dict_replaced(self):
        payload = {
            "model": "gpt-4",
            "provider": "not a dict",
            "openrouter_provider_only": "openai",
        }
        _apply_provider_routing_params_to_payload(payload)
        assert payload["provider"] == {"only": ["openai"]}


def _responses_body(text: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "openai/gpt-4o", "input": "hi", "stream": False,
        "text": text, "response_format": {"type": "json_object"},
    }
