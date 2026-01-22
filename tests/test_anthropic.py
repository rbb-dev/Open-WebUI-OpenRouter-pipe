"""Tests for open_webui_openrouter_pipe/integrations/anthropic.py

This test file covers the missing lines in anthropic.py (79% coverage, 12 missing lines):
- Line 36: return (when ENABLE_ANTHROPIC_PROMPT_CACHING is False)
- Line 49: continue (when item is not a dict)
- Line 51: continue (when type != "message")
- Line 66: target_indices.append(user_message_indices[-3]) (when there are > 2 user messages)
- Line 71: continue (when msg_idx already in seen)
- Line 76: continue (when content is not a list or empty)
- Line 79: continue (when block is not a dict)
- Line 81: continue (when block type != "input_text")
- Line 84: continue (when text is not a string or empty)
- Lines 88-90: handling existing cache_control with ttl merging

Uses real Pipe() instances and tests directly against the
_maybe_apply_anthropic_prompt_caching function.
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import pytest

from open_webui_openrouter_pipe import Pipe


def _get_cache_control_from_block(block: dict) -> dict | None:
    """Extract cache_control from a content block."""
    return block.get("cache_control") if isinstance(block, dict) else None


def _get_last_input_text_cache_control(message: dict) -> dict | None:
    """Get cache_control from the last input_text block in a message."""
    content = message.get("content")
    if not isinstance(content, list):
        return None
    for block in reversed(content):
        if not isinstance(block, dict):
            continue
        if block.get("type") != "input_text":
            continue
        return block.get("cache_control")
    return None


# ============================================================================
# Test: Line 36 - Early return when caching is disabled
# ============================================================================


def test_prompt_caching_disabled_returns_early(pipe_instance):
    """Test that when ENABLE_ANTHROPIC_PROMPT_CACHING is False, the function returns early (line 36)."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": False,
        }
    )

    # Create input items that would normally get cache_control added
    input_items = [
        {
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": "System prompt"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        },
    ]

    # Call the function - should return early without modifying
    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    # Verify no cache_control was added
    assert not Pipe._input_contains_cache_control(input_items)


# ============================================================================
# Test: Line 49 - Skip non-dict items in input_items
# ============================================================================


def test_skips_non_dict_items_in_input(pipe_instance):
    """Test that non-dict items in input_items are skipped (line 49)."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    # Include non-dict items mixed with valid messages
    input_items = [
        "not a dict",  # Should be skipped (line 49)
        123,  # Should be skipped (line 49)
        None,  # Should be skipped (line 49)
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        },
    ]

    # Should not raise and should process the valid message
    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    # The valid user message should have cache_control
    user_msg = input_items[3]
    assert _get_last_input_text_cache_control(user_msg) == {"type": "ephemeral", "ttl": "5m"}


# ============================================================================
# Test: Line 51 - Skip items where type != "message"
# ============================================================================


def test_skips_non_message_type_items(pipe_instance):
    """Test that items with type != 'message' are skipped (line 51)."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "tool_call",  # Not "message", should be skipped (line 51)
            "role": "user",
            "content": [{"type": "input_text", "text": "Tool call"}],
        },
        {
            "type": "function_call",  # Not "message", should be skipped (line 51)
            "content": [{"type": "input_text", "text": "Function"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Valid user message"}],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    # Only the valid message should have cache_control
    assert input_items[0].get("content", [{}])[0].get("cache_control") is None
    assert input_items[1].get("content", [{}])[0].get("cache_control") is None
    assert _get_last_input_text_cache_control(input_items[2]) == {"type": "ephemeral", "ttl": "5m"}


# ============================================================================
# Test: Line 66 - Add third user message when > 2 user messages exist
# ============================================================================


def test_caches_third_user_message_when_more_than_two(pipe_instance):
    """Test that when there are > 2 user messages, the third-to-last also gets cached (line 66)."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "system",
            "content": [{"type": "input_text", "text": "System"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "User 1 - third to last"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "input_text", "text": "Assistant 1"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "User 2 - second to last"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "input_text", "text": "Assistant 2"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "User 3 - last"}],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    expected = {"type": "ephemeral", "ttl": "5m"}

    # System message should be cached
    assert _get_last_input_text_cache_control(input_items[0]) == expected

    # All three user messages should be cached (line 66 covers the third)
    assert _get_last_input_text_cache_control(input_items[1]) == expected  # User 1 (third to last) - line 66
    assert _get_last_input_text_cache_control(input_items[3]) == expected  # User 2 (second to last)
    assert _get_last_input_text_cache_control(input_items[5]) == expected  # User 3 (last)

    # Assistant messages should NOT be cached
    assert _get_last_input_text_cache_control(input_items[2]) is None
    assert _get_last_input_text_cache_control(input_items[4]) is None


# ============================================================================
# Test: Line 71 - Skip duplicate indices in seen set
# ============================================================================


def test_seen_set_prevents_duplicate_processing(pipe_instance):
    """Test that the seen set prevents duplicate processing (line 71 is defensive code).

    Line 71 (`continue` when msg_idx in seen) is defensive/unreachable code.
    The target_indices list is built from:
    - system_message_indices[-1] (last system message)
    - user_message_indices[-1] (last user message)
    - user_message_indices[-2] (second-to-last user message)
    - user_message_indices[-3] (third-to-last user message)

    Since a message cannot be both system and user, and user indices are unique
    positions, duplicates cannot occur in target_indices through normal code paths.

    This test documents that behavior and shows the function works correctly
    even with the defensive check in place.
    """
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    # With only one user message, we verify correct behavior
    # The seen set is still useful as defensive code, but duplicates won't occur
    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Only user message"}],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    # Should have cache_control (added only once)
    cc = _get_last_input_text_cache_control(input_items[0])
    assert cc == {"type": "ephemeral", "ttl": "5m"}


# ============================================================================
# Test: Line 76 - Skip messages where content is not a list or is empty
# ============================================================================


def test_skips_messages_with_invalid_content(pipe_instance):
    """Test that messages with non-list or empty content are skipped (line 76)."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": "string content, not a list",  # Line 76: not a list
        },
        {
            "type": "message",
            "role": "user",
            "content": [],  # Line 76: empty list
        },
        {
            "type": "message",
            "role": "user",
            "content": None,  # Line 76: not a list
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Valid"}],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    # Only the last valid message should have cache_control
    assert input_items[0]["content"] == "string content, not a list"  # Unchanged
    assert input_items[1]["content"] == []  # Unchanged
    assert input_items[2]["content"] is None  # Unchanged
    assert _get_last_input_text_cache_control(input_items[3]) == {"type": "ephemeral", "ttl": "5m"}


# ============================================================================
# Test: Line 79 - Skip non-dict blocks in content
# ============================================================================


def test_skips_non_dict_blocks_in_content(pipe_instance):
    """Test that non-dict blocks in content are skipped (line 79).

    The loop iterates in reversed order and breaks after finding a valid block.
    To hit line 79, we need non-dict blocks at the END of the content list
    (which reversed() sees first), then a valid block earlier.
    """
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Valid block at start"},  # Found after skipping invalid
                "string block at end",  # Line 79: not a dict - seen first by reversed()
            ],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    # The valid input_text block should have cache_control
    content = input_items[0]["content"]
    assert content[0].get("cache_control") == {"type": "ephemeral", "ttl": "5m"}
    assert content[1] == "string block at end"  # Unchanged


# ============================================================================
# Test: Line 81 - Skip blocks where type != "input_text"
# ============================================================================


def test_skips_blocks_with_wrong_type(pipe_instance):
    """Test that blocks with type != 'input_text' are skipped (line 81).

    The loop iterates in reversed order. To hit line 81, we need blocks with
    wrong type at the END (seen first), then a valid input_text earlier.
    """
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Valid - found after skipping"},
                {"type": "image", "url": "http://example.com/img.png"},  # Line 81: seen first
                {"type": "file", "file_id": "abc123"},  # Line 81: seen second
            ],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    content = input_items[0]["content"]
    # Only the valid input_text block gets cache_control
    assert content[0].get("cache_control") == {"type": "ephemeral", "ttl": "5m"}
    # Non-input_text blocks should not have cache_control
    assert "cache_control" not in content[1]
    assert "cache_control" not in content[2]


# ============================================================================
# Test: Line 84 - Skip blocks where text is not a string or is empty
# ============================================================================


def test_skips_blocks_with_invalid_text(pipe_instance):
    """Test that blocks with non-string or empty text are skipped (line 84).

    The loop iterates in reversed order. To hit line 84, we need input_text blocks
    with invalid text at the END (seen first), then a valid input_text earlier.
    """
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Valid text - found after skipping"},
                {"type": "input_text", "text": None},  # Line 84: not a string - seen first
                {"type": "input_text", "text": ""},  # Line 84: empty string - seen second
                {"type": "input_text", "text": 123},  # Line 84: not a string - seen third
            ],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    content = input_items[0]["content"]
    # Only block with valid text gets cache_control
    assert content[0].get("cache_control") == {"type": "ephemeral", "ttl": "5m"}
    # Blocks with invalid text should not have cache_control
    assert "cache_control" not in content[1]
    assert "cache_control" not in content[2]
    assert "cache_control" not in content[3]


# ============================================================================
# Tests: Lines 88-90 - Existing cache_control handling with TTL merging
# ============================================================================


def test_existing_cache_control_with_ttl_not_overwritten(pipe_instance):
    """Test that existing cache_control with ttl is preserved (line 88-90)."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Has existing cache_control",
                    "cache_control": {"type": "ephemeral", "ttl": "10m"},  # Existing with ttl
                },
            ],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    # Existing cache_control with ttl should be preserved
    cc = input_items[0]["content"][0].get("cache_control")
    assert cc == {"type": "ephemeral", "ttl": "10m"}  # Original ttl preserved


def test_existing_cache_control_without_ttl_gets_ttl_added(pipe_instance):
    """Test that existing cache_control without ttl gets ttl added (lines 88-90)."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Has cache_control without ttl",
                    "cache_control": {"type": "ephemeral"},  # No ttl
                },
            ],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    # ttl should be added to existing cache_control
    cc = input_items[0]["content"][0].get("cache_control")
    assert cc == {"type": "ephemeral", "ttl": "5m"}  # ttl was added


def test_existing_cache_control_non_dict_is_not_modified(pipe_instance):
    """Test that non-dict existing cache_control is not modified (lines 88-90 branch)."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "Has non-dict cache_control",
                    "cache_control": "not a dict",  # Invalid but present
                },
            ],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    # Non-dict cache_control should be preserved as-is (not overwritten)
    cc = input_items[0]["content"][0].get("cache_control")
    assert cc == "not a dict"


# ============================================================================
# Test: Non-Anthropic models are skipped
# ============================================================================


def test_non_anthropic_model_skipped(pipe_instance):
    """Test that non-Anthropic models don't get cache_control added."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="openai/gpt-4o",  # Not an Anthropic model
        valves=valves,
    )

    # No cache_control should be added for non-Anthropic models
    assert not Pipe._input_contains_cache_control(input_items)


# ============================================================================
# Test: Developer role treated as system role
# ============================================================================


def test_developer_role_treated_as_system(pipe_instance):
    """Test that developer role messages are cached like system messages."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "developer",  # Should be treated like system
            "content": [{"type": "input_text", "text": "Developer instructions"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    expected = {"type": "ephemeral", "ttl": "5m"}
    # Both developer (as system) and user messages should be cached
    assert _get_last_input_text_cache_control(input_items[0]) == expected
    assert _get_last_input_text_cache_control(input_items[1]) == expected


# ============================================================================
# Test: TTL configuration variations
# ============================================================================


def test_empty_ttl_omits_ttl_field(pipe_instance):
    """Test that empty TTL string results in no ttl field in cache_control."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "",  # Empty string
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    # cache_control should only have type, no ttl
    cc = _get_last_input_text_cache_control(input_items[0])
    assert cc == {"type": "ephemeral"}
    assert "ttl" not in cc


def test_non_string_ttl_omits_ttl_field(pipe_instance):
    """Test that non-string TTL results in no ttl field."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": 300,  # Integer, not string
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    # cache_control should only have type, no ttl (since TTL wasn't a string)
    cc = _get_last_input_text_cache_control(input_items[0])
    assert cc == {"type": "ephemeral"}


# ============================================================================
# Test: The reversed content iteration finds last input_text
# ============================================================================


def test_caches_last_input_text_block_in_content(pipe_instance):
    """Test that only the last input_text block in content gets cached."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "First text block"},
                {"type": "image", "url": "http://example.com/img.png"},
                {"type": "input_text", "text": "Second text block"},
                {"type": "file", "file_id": "abc"},
                {"type": "input_text", "text": "Last text block"},  # Should get cached
            ],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    content = input_items[0]["content"]
    # Only the last input_text block should have cache_control
    assert "cache_control" not in content[0]  # First text block
    assert "cache_control" not in content[2]  # Second text block
    assert content[4].get("cache_control") == {"type": "ephemeral", "ttl": "5m"}  # Last text block


# ============================================================================
# Test: Anthropic model ID detection variations
# ============================================================================


def test_anthropic_model_with_dot_prefix(pipe_instance):
    """Test that models with 'anthropic.' prefix are recognized."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        },
    ]

    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic.claude-3-opus",  # Dot prefix
        valves=valves,
    )

    # Should be recognized as Anthropic model
    assert Pipe._input_contains_cache_control(input_items)


def test_is_anthropic_model_id_with_non_string(pipe_instance):
    """Test _is_anthropic_model_id returns False for non-string input."""
    assert Pipe._is_anthropic_model_id(None) is False
    assert Pipe._is_anthropic_model_id(123) is False
    assert Pipe._is_anthropic_model_id(["anthropic/claude"]) is False
    assert Pipe._is_anthropic_model_id({"model": "anthropic/claude"}) is False


def test_is_anthropic_model_id_with_whitespace(pipe_instance):
    """Test _is_anthropic_model_id handles whitespace correctly."""
    assert Pipe._is_anthropic_model_id("  anthropic/claude-sonnet-4.5  ") is True
    assert Pipe._is_anthropic_model_id("  anthropic.claude-3  ") is True


# ============================================================================
# Test: Message with missing or None role
# ============================================================================


def test_message_with_none_role(pipe_instance):
    """Test that messages with None role are handled gracefully."""
    pipe = pipe_instance
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    input_items = [
        {
            "type": "message",
            "role": None,  # None role
            "content": [{"type": "input_text", "text": "No role"}],
        },
        {
            "type": "message",
            # Missing role key entirely
            "content": [{"type": "input_text", "text": "Missing role"}],
        },
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Valid user"}],
        },
    ]

    # Should not raise and should only cache the valid user message
    pipe._maybe_apply_anthropic_prompt_caching(
        input_items,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    # Messages with None/missing role should not be cached (they're not system/developer/user)
    assert "cache_control" not in input_items[0]["content"][0]
    assert "cache_control" not in input_items[1]["content"][0]
    assert _get_last_input_text_cache_control(input_items[2]) == {"type": "ephemeral", "ttl": "5m"}


# ============================================================================
# Test: Integration with transform_messages_to_input
# ============================================================================


@pytest.mark.asyncio
async def test_prompt_caching_via_transform_messages(pipe_instance_async):
    """Test that prompt caching works through the full transform_messages_to_input path."""
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
        {"role": "assistant", "content": "Second answer"},
        {"role": "user", "content": "Third question"},
    ]

    input_items = await pipe.transform_messages_to_input(
        messages,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    # Should have cache_control on appropriate messages
    assert Pipe._input_contains_cache_control(input_items)

    # Count cached messages
    cached_count = 0
    for item in input_items:
        if isinstance(item, dict) and item.get("type") == "message":
            if _get_last_input_text_cache_control(item):
                cached_count += 1

    # Should have: 1 system + 3 user messages (with > 2 user messages, all 3 get cached)
    assert cached_count == 4


# ===== From test_anthropic_prompt_caching.py =====

import pytest

from typing import Any, cast

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe import ResponsesBody


def _last_input_text_cache_control(message: dict) -> dict | None:
    content = message.get("content")
    if not isinstance(content, list):
        return None
    for block in reversed(content):
        if not isinstance(block, dict):
            continue
        if block.get("type") != "input_text":
            continue
        return block.get("cache_control")
    return None


@pytest.mark.asyncio
async def test_anthropic_prompt_caching_inserts_breakpoints(pipe_instance_async):
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "user-1"},
        {"role": "assistant", "content": "assistant-1"},
        {"role": "user", "content": "user-2"},
    ]

    input_items = await pipe.transform_messages_to_input(
        messages,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    system_messages = [
        item for item in input_items
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "system"
    ]
    user_messages = [
        item for item in input_items
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "user"
    ]
    assistant_messages = [
        item for item in input_items
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "assistant"
    ]

    assert system_messages
    assert len(user_messages) == 2
    assert assistant_messages

    expected = {"type": "ephemeral", "ttl": "5m"}
    assert _last_input_text_cache_control(system_messages[-1]) == expected
    assert _last_input_text_cache_control(user_messages[-1]) == expected
    assert _last_input_text_cache_control(user_messages[-2]) == expected
    assert _last_input_text_cache_control(assistant_messages[-1]) is None


@pytest.mark.asyncio
async def test_non_anthropic_models_do_not_insert_cache_control(pipe_instance_async):
    pipe = pipe_instance_async
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "user-1"},
        {"role": "assistant", "content": "assistant-1"},
        {"role": "user", "content": "user-2"},
    ]

    input_items = await pipe.transform_messages_to_input(
        messages,
        model_id="openai/gpt-5.1",
    )

    assert not Pipe._input_contains_cache_control(input_items)


def test_strip_cache_control_from_input_removes_markers():
    payload = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "hello",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                }
            ],
        }
    ]

    assert Pipe._input_contains_cache_control(payload)
    Pipe._strip_cache_control_from_input(payload)
    assert not Pipe._input_contains_cache_control(payload)


@pytest.mark.asyncio
async def test_anthropic_prompt_caching_applied_to_existing_input(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    body = ResponsesBody(
        model="anthropic/claude-sonnet-4.5",
        input=[
            {
                "type": "message",
                "role": "system",
                "content": [{"type": "input_text", "text": "SYSTEM"}],
            },
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            },
        ],
        stream=True,
    )

    captured: dict[str, Any] = {}

    async def fake_stream(self, session, request_body, **_kwargs):
        captured["request_body"] = request_body
        yield {"type": "response.output_text.delta", "delta": "ok"}
        yield {
            "type": "response.completed",
            "response": {"output": [], "usage": {"input_tokens": 1, "output_tokens": 1}},
        }

    monkeypatch.setattr(Pipe, "send_openai_responses_streaming_request", fake_stream)

    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    result = await pipe._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "sandbox"}},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )

    assert result == "ok"
    request_body = captured.get("request_body") or {}
    assert Pipe._input_contains_cache_control(request_body.get("input"))
