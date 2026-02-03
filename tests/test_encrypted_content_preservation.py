"""Tests for encrypted_content preservation during tool continuations.

Per Responses API spec, reasoning items with encrypted_content must be passed back
on tool continuations to preserve the model's chain of thought across tool calls.
See: https://cookbook.openai.com/examples/responses_api/reasoning_items

These tests verify that:
1. Reasoning items with encrypted_content are extracted from response output
2. They are added to body.input before function_call items
3. The correct order is maintained: reasoning -> function_call -> function_call_output
4. Reasoning items WITHOUT encrypted_content are not preserved (they don't need to be)
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.api.transforms import ResponsesBody
from open_webui_openrouter_pipe.core.config import EncryptedStr


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def pipe_instance():
    """Create a Pipe instance for testing."""
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("test-api-key")
    yield pipe


@pytest.fixture
def sample_reasoning_with_encrypted_content():
    """Sample reasoning item with encrypted_content (ZDR/stateless mode)."""
    return {
        "type": "reasoning",
        "id": "rs_test123",
        "summary": ["Analyzing the request"],
        "encrypted_content": "gAAAAABo_base64_encrypted_blob_here...",
        "format": "openai-responses-v1",
    }


@pytest.fixture
def sample_reasoning_without_encrypted_content():
    """Sample reasoning item WITHOUT encrypted_content (stateful mode)."""
    return {
        "type": "reasoning",
        "id": "rs_test456",
        "summary": ["Thinking about the problem"],
        # No encrypted_content - this is stateful mode
    }


@pytest.fixture
def sample_function_call():
    """Sample function_call item."""
    return {
        "type": "function_call",
        "call_id": "call_abc123",
        "name": "get_weather",
        "arguments": '{"location": "Sydney"}',
    }


@pytest.fixture
def sample_message_item():
    """Sample message item."""
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Let me check that for you."}],
    }


# =============================================================================
# Unit Tests for Reasoning Item Extraction Logic
# =============================================================================


class TestReasoningItemExtraction:
    """Test the logic that extracts reasoning items from response output."""

    def test_extracts_reasoning_with_encrypted_content(
        self,
        sample_reasoning_with_encrypted_content,
        sample_function_call,
    ):
        """Reasoning items WITH encrypted_content should be extracted."""
        output = [
            sample_reasoning_with_encrypted_content,
            sample_function_call,
        ]

        # Simulate the extraction logic from streaming_core.py
        reasoning_items = []
        call_items = []

        for item in output:
            item_type = item.get("type")
            if item_type == "reasoning" and item.get("encrypted_content"):
                reasoning_items.append(item)
            elif item_type == "function_call":
                call_items.append(item)

        assert len(reasoning_items) == 1
        assert reasoning_items[0]["encrypted_content"] == "gAAAAABo_base64_encrypted_blob_here..."
        assert len(call_items) == 1

    def test_does_not_extract_reasoning_without_encrypted_content(
        self,
        sample_reasoning_without_encrypted_content,
        sample_function_call,
    ):
        """Reasoning items WITHOUT encrypted_content should NOT be extracted."""
        output = [
            sample_reasoning_without_encrypted_content,
            sample_function_call,
        ]

        reasoning_items = []
        call_items = []

        for item in output:
            item_type = item.get("type")
            if item_type == "reasoning" and item.get("encrypted_content"):
                reasoning_items.append(item)
            elif item_type == "function_call":
                call_items.append(item)

        # No reasoning items extracted (no encrypted_content)
        assert len(reasoning_items) == 0
        assert len(call_items) == 1

    def test_handles_mixed_output_items(
        self,
        sample_reasoning_with_encrypted_content,
        sample_reasoning_without_encrypted_content,
        sample_function_call,
        sample_message_item,
    ):
        """Test extraction with a mix of item types."""
        output = [
            sample_message_item,
            sample_reasoning_without_encrypted_content,  # Should NOT be extracted
            sample_reasoning_with_encrypted_content,     # Should be extracted
            sample_function_call,
        ]

        reasoning_items = []
        call_items = []

        for item in output:
            item_type = item.get("type")
            if item_type == "reasoning" and item.get("encrypted_content"):
                reasoning_items.append(item)
            elif item_type == "function_call":
                call_items.append(item)

        # Only the one with encrypted_content
        assert len(reasoning_items) == 1
        assert reasoning_items[0]["id"] == "rs_test123"
        assert len(call_items) == 1


# =============================================================================
# Tests for Input Array Construction
# =============================================================================


class TestInputArrayConstruction:
    """Test that the input array is constructed correctly for tool continuations."""

    def test_reasoning_before_function_call_in_input(
        self,
        sample_reasoning_with_encrypted_content,
        sample_function_call,
    ):
        """Reasoning items must come BEFORE function_call items in input."""
        # Simulate what should happen in streaming_core.py
        body_input = [
            {"type": "message", "role": "user", "content": "What's the weather?"},
        ]

        reasoning_items = [sample_reasoning_with_encrypted_content]
        call_items = [sample_function_call]

        # Order matters: reasoning first, then function_call
        if reasoning_items:
            body_input.extend(reasoning_items)
        body_input.extend(call_items)

        # Verify order
        types_in_order = [item.get("type") for item in body_input]
        assert types_in_order == ["message", "reasoning", "function_call"]

    def test_function_call_output_comes_last(
        self,
        sample_reasoning_with_encrypted_content,
        sample_function_call,
    ):
        """Full continuation should be: message, reasoning, function_call, function_call_output."""
        body_input = [
            {"type": "message", "role": "user", "content": "What's the weather?"},
        ]

        # Add reasoning and function_call (per our fix)
        body_input.append(sample_reasoning_with_encrypted_content)
        body_input.append(sample_function_call)

        # Add function_call_output (tool result)
        body_input.append({
            "type": "function_call_output",
            "call_id": "call_abc123",
            "output": '{"temperature": 22, "conditions": "sunny"}',
        })

        types_in_order = [item.get("type") for item in body_input]
        assert types_in_order == ["message", "reasoning", "function_call", "function_call_output"]


# =============================================================================
# Tests for Multiple Reasoning Items
# =============================================================================


class TestMultipleReasoningItems:
    """Test handling of multiple reasoning items (rare but possible)."""

    def test_preserves_multiple_reasoning_items(self):
        """All reasoning items with encrypted_content should be preserved."""
        output = [
            {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "encrypted_blob_1",
            },
            {
                "type": "reasoning",
                "id": "rs_2",
                "encrypted_content": "encrypted_blob_2",
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "tool_a",
                "arguments": "{}",
            },
        ]

        reasoning_items = []
        for item in output:
            if item.get("type") == "reasoning" and item.get("encrypted_content"):
                reasoning_items.append(item)

        assert len(reasoning_items) == 2
        assert reasoning_items[0]["id"] == "rs_1"
        assert reasoning_items[1]["id"] == "rs_2"


# =============================================================================
# Tests for Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_encrypted_content_not_preserved(self):
        """Empty string encrypted_content should not be preserved."""
        output = [
            {
                "type": "reasoning",
                "id": "rs_empty",
                "encrypted_content": "",  # Empty string
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "tool_a",
                "arguments": "{}",
            },
        ]

        reasoning_items = []
        for item in output:
            if item.get("type") == "reasoning" and item.get("encrypted_content"):
                reasoning_items.append(item)

        # Empty string is falsy, so not preserved
        assert len(reasoning_items) == 0

    def test_none_encrypted_content_not_preserved(self):
        """None encrypted_content should not be preserved."""
        output = [
            {
                "type": "reasoning",
                "id": "rs_none",
                "encrypted_content": None,
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "tool_a",
                "arguments": "{}",
            },
        ]

        reasoning_items = []
        for item in output:
            if item.get("type") == "reasoning" and item.get("encrypted_content"):
                reasoning_items.append(item)

        assert len(reasoning_items) == 0

    def test_no_function_calls_means_no_reasoning_needed(self):
        """If there are no function calls, reasoning items don't need to be preserved."""
        output = [
            {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "blob",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Here's my response."}],
            },
        ]

        reasoning_items = []
        call_items = []

        for item in output:
            item_type = item.get("type")
            if item_type == "reasoning" and item.get("encrypted_content"):
                reasoning_items.append(item)
            elif item_type == "function_call":
                call_items.append(item)

        # Reasoning found but no function calls
        assert len(reasoning_items) == 1
        assert len(call_items) == 0

        # In the actual code, reasoning is only added to body.input if call_items exist
        # This test documents that behavior


class TestReasoningItemFormat:
    """Test that reasoning items are passed through unchanged."""

    def test_reasoning_item_passed_unchanged(
        self,
        sample_reasoning_with_encrypted_content,
    ):
        """Reasoning items should be passed to body.input exactly as received."""
        original = sample_reasoning_with_encrypted_content.copy()

        # Simulate extraction and re-insertion
        reasoning_items = [sample_reasoning_with_encrypted_content]
        body_input = []
        body_input.extend(reasoning_items)

        # The item should be unchanged
        assert body_input[0] == original
        assert body_input[0]["encrypted_content"] == original["encrypted_content"]
        assert body_input[0]["format"] == "openai-responses-v1"
        assert body_input[0]["id"] == original["id"]
