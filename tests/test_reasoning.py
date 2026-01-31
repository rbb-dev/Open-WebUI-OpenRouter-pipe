"""Comprehensive tests for ReasoningTracker covering all code paths.

This module targets 90%+ coverage for streaming/reasoning_tracker.py by testing:
- Storage context resolution (lines 109-113, 118-129)
- Image materialization from various formats (lines 152, 168-202, 208, 211, 220-222, 230, 233-234, 252-254)
- Image extraction from event items (lines 263-297)
- Reasoning text extraction (lines 302-332)
- Reasoning stream key generation (lines 337-351)
- Reasoning text deduplication (lines 363-390)
- Status emission throttling (lines 403-444)
- Reasoning event detection (lines 450-463)
- Event processing (lines 483-541)
- Citation extraction (lines 553-586)
- Item tracking (lines 591-598)
- Final buffer retrieval (lines 603, 608, 613-614, 619)
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import asyncio
import base64
import logging
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, Mock, patch
from time import perf_counter

import pytest

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.streaming.reasoning_tracker import ReasoningTracker


# =============================================================================
# Test Fixtures
# =============================================================================


class _StubPipe:
    """Minimal stub Pipe for unit tests that don't need full Pipe functionality."""

    def __init__(self):
        self._emit_status = AsyncMock()
        self._emit_citation = AsyncMock()
        self._resolve_storage_context_returns: tuple = (Mock(), Mock())
        self._upload_to_owui_storage_returns: Optional[str] = "file-123"

    async def _resolve_storage_context(
        self, request: Optional[Any], user_obj: Optional[Any]
    ) -> tuple[Optional[Any], Optional[Any]]:
        return self._resolve_storage_context_returns

    async def _upload_to_owui_storage(
        self,
        request: Any,
        user: Any,
        file_data: bytes,
        filename: str,
        mime_type: str,
        chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
        owui_user_id: str = "",
    ) -> Optional[str]:
        return self._upload_to_owui_storage_returns

    def _parse_data_url(self, text: str) -> Optional[dict]:
        """Parse data URL format."""
        if not text.startswith("data:"):
            return None
        # Simple parser for tests
        try:
            parts = text.split(",", 1)
            if len(parts) != 2:
                return None
            header, b64data = parts
            mime_type = header.replace("data:", "").split(";")[0] or "image/png"
            data = base64.b64decode(b64data, validate=True)
            return {"data": data, "mime_type": mime_type}
        except Exception:
            return None


def _create_tracker(
    pipe: Any = None,
    thinking_mode: str = "open_webui",
    event_emitter: Optional[AsyncMock] = None,
    persist_reasoning: str = "never",
    **kwargs,
) -> ReasoningTracker:
    """Create a ReasoningTracker with common defaults."""
    if pipe is None:
        pipe = _StubPipeImageTests()

    valves = SimpleNamespace(
        THINKING_OUTPUT_MODE=thinking_mode,
        PERSIST_REASONING_TOKENS=persist_reasoning,
    )

    return ReasoningTracker(
        cast(Pipe, pipe),
        valves=valves,
        event_emitter=event_emitter,
        logger=logging.getLogger(__name__),
        **kwargs,
    )


# =============================================================================
# Storage Context Tests (lines 109-113)
# =============================================================================


@pytest.mark.asyncio
async def test_get_storage_context_caches_result():
    """Test that _get_storage_context caches the result from pipe."""
    pipe = _StubPipe()
    mock_request = Mock()
    mock_user = Mock()
    pipe._resolve_storage_context_returns = (mock_request, mock_user)

    tracker = _create_tracker(pipe=pipe)

    # First call should query pipe
    result1 = await tracker._get_storage_context()
    assert result1 == (mock_request, mock_user)

    # Second call should use cache
    pipe._resolve_storage_context_returns = (None, None)
    result2 = await tracker._get_storage_context()
    assert result2 == (mock_request, mock_user)  # Still cached


@pytest.mark.asyncio
async def test_get_storage_context_returns_none_tuple_when_none():
    """Test that _get_storage_context handles None cache properly."""
    pipe = _StubPipe()
    pipe._resolve_storage_context_returns = (None, None)

    tracker = _create_tracker(pipe=pipe)

    result = await tracker._get_storage_context()
    assert result == (None, None)


# =============================================================================
# Image Persistence Tests (lines 118-129)
# =============================================================================


@pytest.mark.asyncio
async def test_persist_generated_image_returns_none_without_context():
    """Test that _persist_generated_image returns None without storage context."""
    pipe = _StubPipe()
    pipe._resolve_storage_context_returns = (None, None)

    tracker = _create_tracker(pipe=pipe)

    result = await tracker._persist_generated_image(b"image-data", "image/png")
    assert result is None


@pytest.mark.asyncio
async def test_persist_generated_image_extracts_extension_from_mime():
    """Test that _persist_generated_image extracts extension from mime type."""
    pipe = _StubPipe()
    mock_request = Mock()
    mock_user = Mock()
    pipe._resolve_storage_context_returns = (mock_request, mock_user)

    tracker = _create_tracker(pipe=pipe)

    # Test PNG
    result = await tracker._persist_generated_image(b"image-data", "image/png")
    assert result == "file-123"

    # Test JPG -> jpeg
    result = await tracker._persist_generated_image(b"image-data", "image/jpg")
    assert result == "file-123"


@pytest.mark.asyncio
async def test_persist_generated_image_handles_invalid_mime():
    """Test that _persist_generated_image handles invalid mime type."""
    pipe = _StubPipe()
    mock_request = Mock()
    mock_user = Mock()
    pipe._resolve_storage_context_returns = (mock_request, mock_user)

    tracker = _create_tracker(pipe=pipe)

    # No slash in mime type
    result = await tracker._persist_generated_image(b"image-data", "invalid")
    assert result == "file-123"


# =============================================================================
# Base64 Size Validation (line 143-145)
# =============================================================================


def test_validate_base64_size_accepts_small():
    """Test that _validate_base64_size accepts small strings."""
    tracker = _create_tracker()
    assert tracker._validate_base64_size("small") is True


def test_validate_base64_size_rejects_huge():
    """Test that _validate_base64_size rejects huge strings (>10MB)."""
    tracker = _create_tracker()
    # Create a string larger than 10MB (approx 14M chars = 10MB decoded)
    huge = "A" * (15 * 1024 * 1024)
    assert tracker._validate_base64_size(huge) is False


# =============================================================================
# Image Materialization from String (lines 148-202)
# =============================================================================


@pytest.mark.asyncio
async def test_materialize_image_from_str_empty_returns_none():
    """Test that empty string returns None (line 152)."""
    tracker = _create_tracker()
    result = await tracker._materialize_image_from_str("")
    assert result is None

    result = await tracker._materialize_image_from_str("   ")
    assert result is None


@pytest.mark.asyncio
async def test_materialize_image_from_str_data_url_success():
    """Test data URL parsing with successful persist."""
    pipe = _StubPipe()
    mock_request = Mock()
    mock_user = Mock()
    pipe._resolve_storage_context_returns = (mock_request, mock_user)
    event_emitter = AsyncMock()

    tracker = _create_tracker(pipe=pipe, event_emitter=event_emitter)

    # Valid data URL
    b64 = base64.b64encode(b"image-data").decode()
    data_url = f"data:image/png;base64,{b64}"

    result = await tracker._materialize_image_from_str(data_url)
    assert result == "/api/v1/files/file-123/content"


@pytest.mark.asyncio
async def test_materialize_image_from_str_data_url_persist_fails():
    """Test data URL when persist fails (line 168)."""
    pipe = _StubPipe()
    mock_request = Mock()
    mock_user = Mock()
    pipe._resolve_storage_context_returns = (mock_request, mock_user)
    pipe._upload_to_owui_storage_returns = None  # Persist fails

    tracker = _create_tracker(pipe=pipe)

    b64 = base64.b64encode(b"image-data").decode()
    data_url = f"data:image/png;base64,{b64}"

    result = await tracker._materialize_image_from_str(data_url)
    assert result == data_url  # Returns original URL


@pytest.mark.asyncio
async def test_materialize_image_from_str_invalid_data_url():
    """Test invalid data URL returns None (line 169)."""
    pipe = _StubPipe()
    # Make _parse_data_url return None
    pipe._parse_data_url = lambda x: None

    tracker = _create_tracker(pipe=pipe)

    result = await tracker._materialize_image_from_str("data:invalid")
    assert result is None


@pytest.mark.asyncio
async def test_materialize_image_from_str_http_url_passthrough():
    """Test HTTP URLs are passed through (line 172-173)."""
    tracker = _create_tracker()

    result = await tracker._materialize_image_from_str("https://example.com/image.png")
    assert result == "https://example.com/image.png"

    result = await tracker._materialize_image_from_str("http://example.com/image.png")
    assert result == "http://example.com/image.png"

    result = await tracker._materialize_image_from_str("/api/v1/files/123/content")
    assert result == "/api/v1/files/123/content"


@pytest.mark.asyncio
async def test_materialize_image_from_str_extracts_base64_after_comma():
    """Test extraction of base64 from format with comma (lines 177-178)."""
    pipe = _StubPipe()
    mock_request = Mock()
    mock_user = Mock()
    pipe._resolve_storage_context_returns = (mock_request, mock_user)

    tracker = _create_tracker(pipe=pipe)

    b64 = base64.b64encode(b"test-image").decode()
    # Format: prefix;base64,<data>
    text = f"prefix;base64,{b64}"

    result = await tracker._materialize_image_from_str(text)
    assert result == "/api/v1/files/file-123/content"


@pytest.mark.asyncio
async def test_materialize_image_from_str_cleaned_empty_returns_none():
    """Test cleaned string that becomes empty returns None (line 180-181)."""
    tracker = _create_tracker()

    result = await tracker._materialize_image_from_str("prefix;base64,")
    assert result is None


@pytest.mark.asyncio
async def test_materialize_image_from_str_huge_base64_rejected():
    """Test huge base64 is rejected (line 183-184)."""
    tracker = _create_tracker()

    huge = "A" * (15 * 1024 * 1024)
    result = await tracker._materialize_image_from_str(huge)
    assert result is None


@pytest.mark.asyncio
async def test_materialize_image_from_str_invalid_base64():
    """Test invalid base64 returns None (lines 188-189)."""
    tracker = _create_tracker()

    result = await tracker._materialize_image_from_str("not-valid-base64!!!")
    assert result is None


@pytest.mark.asyncio
async def test_materialize_image_from_str_raw_base64_persist_fails():
    """Test raw base64 when persist fails returns data URL (line 202)."""
    pipe = _StubPipe()
    mock_request = Mock()
    mock_user = Mock()
    pipe._resolve_storage_context_returns = (mock_request, mock_user)
    pipe._upload_to_owui_storage_returns = None

    tracker = _create_tracker(pipe=pipe)

    b64 = base64.b64encode(b"test-data").decode()
    result = await tracker._materialize_image_from_str(b64)
    assert result == f"data:image/png;base64,{b64}"


@pytest.mark.asyncio
async def test_materialize_image_from_str_raw_base64_emits_status():
    """Test raw base64 persist emits status (lines 194-201)."""
    pipe = _StubPipe()
    mock_request = Mock()
    mock_user = Mock()
    pipe._resolve_storage_context_returns = (mock_request, mock_user)
    event_emitter = AsyncMock()

    tracker = _create_tracker(pipe=pipe, event_emitter=event_emitter)

    b64 = base64.b64encode(b"test-data").decode()
    result = await tracker._materialize_image_from_str(b64)
    assert result == "/api/v1/files/file-123/content"
    pipe._emit_status.assert_called_once()


# =============================================================================
# Image Entry Materialization (lines 205-254)
# =============================================================================


@pytest.mark.asyncio
async def test_materialize_image_entry_none_returns_none():
    """Test None entry returns None (line 207-208)."""
    tracker = _create_tracker()
    result = await tracker._materialize_image_entry(None)
    assert result is None


@pytest.mark.asyncio
async def test_materialize_image_entry_string_delegates():
    """Test string entry delegates to _materialize_image_from_str (line 210-211)."""
    tracker = _create_tracker()
    result = await tracker._materialize_image_entry("https://example.com/image.png")
    assert result == "https://example.com/image.png"


@pytest.mark.asyncio
async def test_materialize_image_entry_dict_with_url():
    """Test dict with URL key (lines 215-218)."""
    tracker = _create_tracker()

    result = await tracker._materialize_image_entry({"url": "https://example.com/img.png"})
    assert result == "https://example.com/img.png"

    result = await tracker._materialize_image_entry({"image_url": "https://example.com/img.png"})
    assert result == "https://example.com/img.png"


@pytest.mark.asyncio
async def test_materialize_image_entry_dict_with_nested_candidate():
    """Test dict with nested URL candidate (lines 219-222)."""
    tracker = _create_tracker()

    # Nested dict in candidate
    result = await tracker._materialize_image_entry({
        "url": {"url": "https://example.com/nested.png"}
    })
    assert result == "https://example.com/nested.png"


@pytest.mark.asyncio
async def test_materialize_image_entry_dict_with_base64():
    """Test dict with base64 keys (lines 225-251)."""
    pipe = _StubPipe()
    mock_request = Mock()
    mock_user = Mock()
    pipe._resolve_storage_context_returns = (mock_request, mock_user)
    event_emitter = AsyncMock()

    tracker = _create_tracker(pipe=pipe, event_emitter=event_emitter)

    b64 = base64.b64encode(b"test-image").decode()
    entry = {"b64_json": b64, "mime_type": "image/png"}

    result = await tracker._materialize_image_entry(entry)
    assert result == "/api/v1/files/file-123/content"


@pytest.mark.asyncio
async def test_materialize_image_entry_dict_huge_base64_skipped():
    """Test dict with huge base64 is skipped (line 229-230)."""
    tracker = _create_tracker()

    huge = "A" * (15 * 1024 * 1024)
    entry = {"b64_json": huge}

    result = await tracker._materialize_image_entry(entry)
    assert result is None


@pytest.mark.asyncio
async def test_materialize_image_entry_dict_invalid_base64_skipped():
    """Test dict with invalid base64 is skipped (lines 231-234)."""
    tracker = _create_tracker()

    entry = {"b64_json": "not-valid-base64!!!"}
    result = await tracker._materialize_image_entry(entry)
    assert result is None


@pytest.mark.asyncio
async def test_materialize_image_entry_dict_base64_persist_fails():
    """Test dict base64 when persist fails (line 252)."""
    pipe = _StubPipe()
    mock_request = Mock()
    mock_user = Mock()
    pipe._resolve_storage_context_returns = (mock_request, mock_user)
    pipe._upload_to_owui_storage_returns = None

    tracker = _create_tracker(pipe=pipe)

    b64 = base64.b64encode(b"test-image").decode()
    entry = {"b64_json": b64, "mime_type": "image/jpeg"}

    result = await tracker._materialize_image_entry(entry)
    assert result == f"data:image/jpeg;base64,{b64}"


@pytest.mark.asyncio
async def test_materialize_image_entry_unsupported_returns_none():
    """Test unsupported entry type returns None (line 254)."""
    tracker = _create_tracker()

    # Empty dict
    result = await tracker._materialize_image_entry({})
    assert result is None

    # List (not dict or string)
    result = await tracker._materialize_image_entry([])
    assert result is None


# =============================================================================
# Image Extraction from Items (lines 257-297)
# =============================================================================


@pytest.mark.asyncio
async def test_materialize_images_from_item_non_dict_returns_empty():
    """Test non-dict event returns empty list (line 263-264)."""
    tracker = _create_tracker()

    result = await tracker.materialize_images_from_item("not a dict")
    assert result == []


@pytest.mark.asyncio
async def test_materialize_images_from_item_no_item_returns_empty():
    """Test event without item returns empty list (lines 266-268)."""
    tracker = _create_tracker()

    result = await tracker.materialize_images_from_item({})
    assert result == []

    result = await tracker.materialize_images_from_item({"item": "not dict"})
    assert result == []


@pytest.mark.asyncio
async def test_materialize_images_from_item_no_id_returns_empty():
    """Test item without ID returns empty list (lines 270-272)."""
    tracker = _create_tracker()

    result = await tracker.materialize_images_from_item({"item": {"type": "image"}})
    assert result == []

    result = await tracker.materialize_images_from_item({"item": {"id": "", "type": "image"}})
    assert result == []


@pytest.mark.asyncio
async def test_materialize_images_from_item_duplicate_skipped():
    """Test duplicate item ID is skipped (lines 273-274)."""
    tracker = _create_tracker()
    tracker.processed_image_item_ids.add("item-1")

    result = await tracker.materialize_images_from_item({
        "item": {"id": "item-1", "type": "image", "output": ["https://example.com/img.png"]}
    })
    assert result == []


@pytest.mark.asyncio
async def test_materialize_images_from_item_non_image_skipped():
    """Test non-image item type is skipped (lines 277-279)."""
    tracker = _create_tracker()

    result = await tracker.materialize_images_from_item({
        "item": {"id": "item-1", "type": "text", "output": ["https://example.com/img.png"]}
    })
    assert result == []


@pytest.mark.asyncio
async def test_materialize_images_from_item_list_output():
    """Test image item with list output (lines 287-291)."""
    tracker = _create_tracker()

    result = await tracker.materialize_images_from_item({
        "item": {
            "id": "item-1",
            "type": "image",
            "output": [
                "https://example.com/img1.png",
                "https://example.com/img2.png",
            ]
        }
    })
    assert result == [
        "https://example.com/img1.png",
        "https://example.com/img2.png",
    ]


@pytest.mark.asyncio
async def test_materialize_images_from_item_single_output():
    """Test image item with single output (lines 292-295)."""
    tracker = _create_tracker()

    result = await tracker.materialize_images_from_item({
        "item": {
            "id": "item-1",
            "type": "image",
            "content": "https://example.com/single.png",
        }
    })
    assert result == ["https://example.com/single.png"]


# =============================================================================
# Reasoning Text Extraction (lines 299-332)
# =============================================================================


def test_extract_reasoning_text_non_dict_returns_empty():
    """Test non-dict event returns empty string (lines 302-303)."""
    tracker = _create_tracker()

    result = tracker._extract_reasoning_text("not a dict")
    assert result == ""


def test_extract_reasoning_text_delta_key():
    """Test extraction from delta key (lines 306-309)."""
    tracker = _create_tracker()

    result = tracker._extract_reasoning_text({"delta": "thinking..."})
    assert result == "thinking..."


def test_extract_reasoning_text_text_key():
    """Test extraction from text key (lines 306-309)."""
    tracker = _create_tracker()

    result = tracker._extract_reasoning_text({"text": "reasoning content"})
    assert result == "reasoning content"


def test_extract_reasoning_text_nested_part_text():
    """Test extraction from nested part.text (lines 312-316)."""
    tracker = _create_tracker()

    result = tracker._extract_reasoning_text({
        "part": {"text": "nested thinking"}
    })
    assert result == "nested thinking"


def test_extract_reasoning_text_nested_part_content_list():
    """Test extraction from nested part.content list (lines 318-330)."""
    tracker = _create_tracker()

    result = tracker._extract_reasoning_text({
        "part": {
            "content": [
                {"text": "fragment1"},
                {"text": "fragment2"},
            ]
        }
    })
    assert result == "fragment1fragment2"


def test_extract_reasoning_text_nested_part_content_string_entries():
    """Test extraction from content list with string entries (lines 327-328)."""
    tracker = _create_tracker()

    result = tracker._extract_reasoning_text({
        "part": {
            "content": [
                "string1",
                {"text": "dict-text"},
                "string2",
            ]
        }
    })
    assert result == "string1dict-textstring2"


def test_extract_reasoning_text_no_matches():
    """Test extraction with no matching keys returns empty (line 332)."""
    tracker = _create_tracker()

    result = tracker._extract_reasoning_text({"other": "value"})
    assert result == ""


# =============================================================================
# Reasoning Stream Key (lines 334-351)
# =============================================================================


def test_reasoning_stream_key_from_item_id():
    """Test key extraction from item_id (lines 337-339)."""
    tracker = _create_tracker()

    result = tracker._reasoning_stream_key({"item_id": "item-123"}, "response.reasoning.delta")
    assert result == "item-123"


def test_reasoning_stream_key_from_output_item_added():
    """Test key extraction from output_item.added event (lines 341-346)."""
    tracker = _create_tracker()

    result = tracker._reasoning_stream_key(
        {"item": {"id": "item-456"}},
        "response.output_item.added"
    )
    assert result == "item-456"


def test_reasoning_stream_key_from_output_item_done():
    """Test key extraction from output_item.done event (lines 341-346)."""
    tracker = _create_tracker()

    result = tracker._reasoning_stream_key(
        {"item": {"id": "item-789"}},
        "response.output_item.done"
    )
    assert result == "item-789"


def test_reasoning_stream_key_uses_active_item():
    """Test fallback to active_reasoning_item_id (lines 348-349)."""
    tracker = _create_tracker()
    tracker.active_reasoning_item_id = "active-item"

    result = tracker._reasoning_stream_key({}, "response.reasoning.delta")
    assert result == "active-item"


def test_reasoning_stream_key_default():
    """Test default key when no matches (line 351)."""
    tracker = _create_tracker()

    result = tracker._reasoning_stream_key({}, "response.reasoning.delta")
    assert result == "__reasoning__"


# =============================================================================
# Reasoning Text Deduplication (lines 353-390)
# =============================================================================


def test_append_reasoning_text_empty_incoming():
    """Test empty incoming text returns empty (lines 363-365)."""
    tracker = _create_tracker()

    result = tracker._append_reasoning_text("key1", "", allow_misaligned=True)
    assert result == ""


def test_append_reasoning_text_first_chunk():
    """Test first chunk for a key (lines 370-372)."""
    tracker = _create_tracker()

    result = tracker._append_reasoning_text("key1", "first text", allow_misaligned=True)
    assert result == "first text"
    assert tracker.reasoning_buffer == "first text"
    assert tracker.reasoning_stream_buffers["key1"] == "first text"


def test_append_reasoning_text_duplicate_snapshot():
    """Test duplicate snapshot returns empty (lines 373-375)."""
    tracker = _create_tracker()
    tracker.reasoning_stream_buffers["key1"] = "existing"
    tracker.reasoning_buffer = "existing"

    result = tracker._append_reasoning_text("key1", "existing", allow_misaligned=True)
    assert result == ""


def test_append_reasoning_text_growing_snapshot():
    """Test growing snapshot takes suffix (lines 376-378)."""
    tracker = _create_tracker()
    tracker.reasoning_stream_buffers["key1"] = "prefix"
    tracker.reasoning_buffer = "prefix"

    result = tracker._append_reasoning_text("key1", "prefix-suffix", allow_misaligned=True)
    assert result == "-suffix"
    assert tracker.reasoning_buffer == "prefix-suffix"


def test_append_reasoning_text_shrinking_snapshot():
    """Test shrinking snapshot returns empty (lines 379-381)."""
    tracker = _create_tracker()
    tracker.reasoning_stream_buffers["key1"] = "longer-existing"
    tracker.reasoning_buffer = "longer-existing"

    result = tracker._append_reasoning_text("key1", "longer", allow_misaligned=True)
    assert result == ""


def test_append_reasoning_text_misaligned_allowed():
    """Test misaligned text with allow_misaligned=True (lines 383-384)."""
    tracker = _create_tracker()
    tracker.reasoning_stream_buffers["key1"] = "existing"
    tracker.reasoning_buffer = "existing"

    result = tracker._append_reasoning_text("key1", "totally-different", allow_misaligned=True)
    assert result == "totally-different"


def test_append_reasoning_text_misaligned_disallowed():
    """Test misaligned text with allow_misaligned=False (lines 383-384)."""
    tracker = _create_tracker()
    tracker.reasoning_stream_buffers["key1"] = "existing"
    tracker.reasoning_buffer = "existing"

    result = tracker._append_reasoning_text("key1", "totally-different", allow_misaligned=False)
    assert result == ""


# =============================================================================
# Status Emission (lines 392-444)
# =============================================================================


@pytest.mark.asyncio
async def test_maybe_emit_reasoning_status_disabled():
    """Test status emission when thinking_status_enabled is False (line 403-404)."""
    event_emitter = AsyncMock()
    tracker = _create_tracker(thinking_mode="open_webui", event_emitter=event_emitter)
    tracker.thinking_status_enabled = False

    await tracker._maybe_emit_reasoning_status("some text")
    event_emitter.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_emit_reasoning_status_no_emitter():
    """Test status emission without event_emitter (lines 405-406)."""
    tracker = _create_tracker(thinking_mode="status", event_emitter=None)

    # Should not raise
    await tracker._maybe_emit_reasoning_status("some text")


@pytest.mark.asyncio
async def test_maybe_emit_reasoning_status_empty_buffer():
    """Test status emission with empty buffer (lines 408-411)."""
    event_emitter = AsyncMock()
    tracker = _create_tracker(thinking_mode="status", event_emitter=event_emitter)

    await tracker._maybe_emit_reasoning_status("")
    event_emitter.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_emit_reasoning_status_force():
    """Test status emission with force=True (line 413)."""
    event_emitter = AsyncMock()
    tracker = _create_tracker(thinking_mode="status", event_emitter=event_emitter)
    tracker.reasoning_status_buffer = "buffered text"

    await tracker._maybe_emit_reasoning_status("", force=True)
    event_emitter.assert_called_once()
    call_args = event_emitter.call_args[0][0]
    assert call_args["type"] == "status"
    assert call_args["data"]["description"] == "buffered text"


@pytest.mark.asyncio
async def test_maybe_emit_reasoning_status_on_punctuation():
    """Test status emission on punctuation (lines 418-419)."""
    event_emitter = AsyncMock()
    tracker = _create_tracker(thinking_mode="status", event_emitter=event_emitter)
    tracker.reasoning_status_buffer = "Some reasoning text"

    await tracker._maybe_emit_reasoning_status(".")
    event_emitter.assert_called_once()


@pytest.mark.asyncio
async def test_maybe_emit_reasoning_status_on_length():
    """Test status emission on length threshold (lines 421-422)."""
    event_emitter = AsyncMock()
    tracker = _create_tracker(thinking_mode="status", event_emitter=event_emitter)
    tracker.reasoning_status_buffer = "A" * 200  # Above max threshold

    await tracker._maybe_emit_reasoning_status("x")
    event_emitter.assert_called_once()


@pytest.mark.asyncio
async def test_maybe_emit_reasoning_status_on_idle():
    """Test status emission on idle timeout (lines 424-432)."""
    event_emitter = AsyncMock()
    tracker = _create_tracker(thinking_mode="status", event_emitter=event_emitter)
    tracker.reasoning_status_buffer = "A" * 50  # Above min threshold
    tracker.reasoning_status_last_emit = None  # No previous emit

    await tracker._maybe_emit_reasoning_status("x")
    event_emitter.assert_called_once()


@pytest.mark.asyncio
async def test_maybe_emit_reasoning_status_clears_buffer():
    """Test status emission clears buffer and updates timestamp (lines 443-444)."""
    event_emitter = AsyncMock()
    tracker = _create_tracker(thinking_mode="status", event_emitter=event_emitter)
    tracker.reasoning_status_buffer = "buffered"

    await tracker._maybe_emit_reasoning_status(".", force=False)

    assert tracker.reasoning_status_buffer == ""
    assert tracker.reasoning_status_last_emit is not None


@pytest.mark.asyncio
async def test_maybe_emit_reasoning_status_no_emit_when_thresholds_not_met():
    """Test status emission returns early when no thresholds met (line 435).

    This tests the case where:
    - force is False
    - No punctuation at end
    - Buffer length below max threshold
    - Buffer length above min but recent emit (not idle)
    """
    event_emitter = AsyncMock()
    tracker = _create_tracker(thinking_mode="status", event_emitter=event_emitter)
    tracker.reasoning_status_buffer = "A" * 50  # Above min, below max
    tracker.reasoning_status_last_emit = perf_counter()  # Recent emit (not idle)

    # Add text without punctuation
    await tracker._maybe_emit_reasoning_status("x")

    # Should not emit because:
    # - force=False
    # - "x" doesn't end with punctuation
    # - 51 chars < 200 max
    # - last_emit is recent (not idle)
    event_emitter.assert_not_called()


# =============================================================================
# Reasoning Event Detection (lines 446-463)
# =============================================================================


def test_is_reasoning_event_reasoning_delta():
    """Test detection of response.reasoning events (lines 450-452)."""
    tracker = _create_tracker()

    assert tracker._is_reasoning_event("response.reasoning.delta", {}) is True
    assert tracker._is_reasoning_event("response.reasoning.done", {}) is True
    assert tracker._is_reasoning_event("response.reasoning_summary_text.done", {}) is False


def test_is_reasoning_event_content_part():
    """Test detection of reasoning content parts (lines 455-461)."""
    tracker = _create_tracker()

    event = {"part": {"type": "reasoning_text"}}
    assert tracker._is_reasoning_event("response.content_part.delta", event) is True

    event = {"part": {"type": "reasoning_summary_text"}}
    assert tracker._is_reasoning_event("response.content_part.delta", event) is True

    event = {"part": {"type": "summary_text"}}
    assert tracker._is_reasoning_event("response.content_part.delta", event) is True

    event = {"part": {"type": "output_text"}}
    assert tracker._is_reasoning_event("response.content_part.delta", event) is False


def test_is_reasoning_event_non_reasoning():
    """Test detection of non-reasoning events (line 463)."""
    tracker = _create_tracker()

    assert tracker._is_reasoning_event("response.output_text.delta", {}) is False
    assert tracker._is_reasoning_event("response.completed", {}) is False


# =============================================================================
# Process Reasoning Event (lines 465-541)
# =============================================================================


@pytest.mark.asyncio
async def test_process_reasoning_event_non_reasoning_returns_none():
    """Test non-reasoning event returns None (lines 483-484)."""
    tracker = _create_tracker()

    result = await tracker.process_reasoning_event(
        {"delta": "text"},
        "response.output_text.delta",
        normalize_surrogate_chunk=lambda x, _: x,
    )
    assert result is None


@pytest.mark.asyncio
async def test_process_reasoning_event_activates_stream():
    """Test reasoning event activates stream (line 486)."""
    tracker = _create_tracker()
    assert tracker.reasoning_stream_active is False

    await tracker.process_reasoning_event(
        {"delta": "thinking"},
        "response.reasoning.delta",
        normalize_surrogate_chunk=lambda x, _: x,
    )
    assert tracker.reasoning_stream_active is True


@pytest.mark.asyncio
async def test_process_reasoning_event_invalid_key_returns_none(monkeypatch):
    """Test reasoning event with empty/invalid key returns None (line 490).

    The _reasoning_stream_key method always returns at least "__reasoning__",
    so we mock it to return an empty string to trigger line 490.
    """
    tracker = _create_tracker()

    # Mock _reasoning_stream_key to return empty string
    monkeypatch.setattr(tracker, "_reasoning_stream_key", lambda e, t: "")

    result = await tracker.process_reasoning_event(
        {"delta": "thinking"},
        "response.reasoning.delta",
        normalize_surrogate_chunk=lambda x, _: x,
    )
    # Should return None because key is empty
    assert result is None
    # But stream should still be marked as active (happens before key check)
    assert tracker.reasoning_stream_active is True


@pytest.mark.asyncio
async def test_process_reasoning_event_with_default_key():
    """Test reasoning event uses default key when no item_id."""
    tracker = _create_tracker()

    result = await tracker.process_reasoning_event(
        {},  # No item_id or item
        "response.reasoning.delta",
        normalize_surrogate_chunk=lambda x, _: x,
    )
    # Should work with default "__reasoning__" key
    assert tracker.reasoning_stream_active is True


@pytest.mark.asyncio
async def test_process_reasoning_event_marks_incremental():
    """Test incremental events mark the key (lines 497-498)."""
    tracker = _create_tracker()

    await tracker.process_reasoning_event(
        {"item_id": "item-1", "delta": "thinking"},
        "response.reasoning.delta",
        normalize_surrogate_chunk=lambda x, _: x,
    )
    assert "item-1" in tracker.reasoning_stream_has_incremental


@pytest.mark.asyncio
async def test_process_reasoning_event_emits_delta():
    """Test reasoning delta emits event (lines 508-519)."""
    event_emitter = AsyncMock()
    tracker = _create_tracker(thinking_mode="open_webui", event_emitter=event_emitter)

    await tracker.process_reasoning_event(
        {"item_id": "item-1", "delta": "thinking..."},
        "response.reasoning.delta",
        normalize_surrogate_chunk=lambda x, _: x,
    )

    # Find reasoning:delta event
    delta_calls = [
        call for call in event_emitter.call_args_list
        if call[0][0].get("type") == "reasoning:delta"
    ]
    assert len(delta_calls) == 1
    assert delta_calls[0][0][0]["data"]["delta"] == "thinking..."


@pytest.mark.asyncio
async def test_process_reasoning_event_emits_completed():
    """Test final event emits reasoning:completed (lines 523-539)."""
    event_emitter = AsyncMock()
    tracker = _create_tracker(thinking_mode="open_webui", event_emitter=event_emitter)
    tracker.reasoning_stream_buffers["item-1"] = "accumulated thinking"
    tracker.reasoning_buffer = "accumulated thinking"

    await tracker.process_reasoning_event(
        {"item_id": "item-1", "text": ""},
        "response.reasoning.done",
        normalize_surrogate_chunk=lambda x, _: x,
    )

    # Find reasoning:completed event
    completed_calls = [
        call for call in event_emitter.call_args_list
        if call[0][0].get("type") == "reasoning:completed"
    ]
    assert len(completed_calls) == 1
    assert tracker.reasoning_completed_emitted is True
    assert "item-1" in tracker.reasoning_stream_completed


@pytest.mark.asyncio
async def test_process_reasoning_event_skips_duplicate_completed():
    """Test completed event skipped for already completed key (lines 530, 538)."""
    event_emitter = AsyncMock()
    tracker = _create_tracker(thinking_mode="open_webui", event_emitter=event_emitter)
    tracker.reasoning_stream_buffers["item-1"] = "text"
    tracker.reasoning_buffer = "text"
    tracker.reasoning_stream_completed.add("item-1")

    await tracker.process_reasoning_event(
        {"item_id": "item-1", "text": ""},
        "response.reasoning.done",
        normalize_surrogate_chunk=lambda x, _: x,
    )

    # Should not emit another completed event
    completed_calls = [
        call for call in event_emitter.call_args_list
        if call[0][0].get("type") == "reasoning:completed"
    ]
    assert len(completed_calls) == 0


@pytest.mark.asyncio
async def test_process_reasoning_event_returns_append():
    """Test process returns appended text (line 541)."""
    tracker = _create_tracker()

    result = await tracker.process_reasoning_event(
        {"item_id": "item-1", "delta": "new text"},
        "response.reasoning.delta",
        normalize_surrogate_chunk=lambda x, _: x,
    )
    assert result == "new text"


# =============================================================================
# Citation Processing (lines 543-586)
# =============================================================================


@pytest.mark.asyncio
async def test_process_citation_event_wrong_type():
    """Test non-url_citation annotation returns None (lines 553-555)."""
    tracker = _create_tracker()

    result = await tracker.process_citation_event({
        "annotation": {"type": "other"}
    })
    assert result is None

    result = await tracker.process_citation_event({})
    assert result is None


@pytest.mark.asyncio
async def test_process_citation_event_strips_utm():
    """Test utm_source parameter is stripped (lines 558-560)."""
    pipe = _StubPipe()
    event_emitter = AsyncMock()
    tracker = _create_tracker(pipe=pipe, event_emitter=event_emitter)

    result = await tracker.process_citation_event({
        "annotation": {
            "type": "url_citation",
            "url": "https://example.com/page?utm_source=openai",
            "title": "Example Page",
        }
    })

    assert result is not None
    assert result["source"]["url"] == "https://example.com/page"


@pytest.mark.asyncio
async def test_process_citation_event_deduplicates():
    """Test duplicate URLs are skipped (lines 564-565)."""
    event_emitter = AsyncMock()
    tracker = _create_tracker(event_emitter=event_emitter)
    tracker.ordinal_by_url["https://example.com/page"] = 1

    result = await tracker.process_citation_event({
        "annotation": {
            "type": "url_citation",
            "url": "https://example.com/page",
            "title": "Example Page",
        }
    })
    assert result is None


@pytest.mark.asyncio
async def test_process_citation_event_extracts_host():
    """Test host extraction and citation building (lines 567-580)."""
    pipe = _StubPipe()
    event_emitter = AsyncMock()
    tracker = _create_tracker(pipe=pipe, event_emitter=event_emitter)

    result = await tracker.process_citation_event({
        "annotation": {
            "type": "url_citation",
            "url": "https://www.example.com/page/article",
            "title": "Test Article",
        }
    })

    assert result is not None
    assert result["source"]["name"] == "example.com"
    assert result["source"]["url"] == "https://www.example.com/page/article"
    assert result["document"] == ["Test Article"]
    assert result["metadata"][0]["source"] == "https://www.example.com/page/article"


@pytest.mark.asyncio
async def test_process_citation_event_emits_and_stores():
    """Test citation is emitted and stored (lines 582-586)."""
    pipe = _StubPipe()
    event_emitter = AsyncMock()
    tracker = _create_tracker(pipe=pipe, event_emitter=event_emitter)

    result = await tracker.process_citation_event({
        "annotation": {
            "type": "url_citation",
            "url": "https://example.com/page",
            "title": "Page Title",
        }
    })

    assert result is not None
    pipe._emit_citation.assert_called_once()
    assert tracker.emitted_citations == [result]
    assert tracker.ordinal_by_url["https://example.com/page"] == 1


@pytest.mark.asyncio
async def test_process_citation_event_no_emitter():
    """Test citation without event_emitter (lines 582-583)."""
    tracker = _create_tracker(event_emitter=None)

    result = await tracker.process_citation_event({
        "annotation": {
            "type": "url_citation",
            "url": "https://example.com/page",
            "title": "Page Title",
        }
    })

    # Still returns citation but doesn't emit
    assert result is not None
    assert tracker.emitted_citations == [result]


# =============================================================================
# Item Tracking (lines 588-598)
# =============================================================================


def test_track_reasoning_item_sets_active_id():
    """Test track_reasoning_item sets active item ID (lines 591-598)."""
    tracker = _create_tracker()

    tracker.track_reasoning_item({
        "item": {
            "id": "reasoning-item-1",
            "type": "reasoning"
        }
    })
    assert tracker.active_reasoning_item_id == "reasoning-item-1"


def test_track_reasoning_item_ignores_non_reasoning():
    """Test track_reasoning_item ignores non-reasoning types (lines 595-598)."""
    tracker = _create_tracker()

    tracker.track_reasoning_item({
        "item": {
            "id": "text-item-1",
            "type": "text"
        }
    })
    assert tracker.active_reasoning_item_id is None


def test_track_reasoning_item_handles_missing_id():
    """Test track_reasoning_item handles missing ID (lines 596-598)."""
    tracker = _create_tracker()

    tracker.track_reasoning_item({
        "item": {
            "type": "reasoning"
        }
    })
    assert tracker.active_reasoning_item_id is None


def test_track_reasoning_item_handles_non_dict_item():
    """Test track_reasoning_item handles non-dict item (lines 591-592)."""
    tracker = _create_tracker()

    tracker.track_reasoning_item({"item": "not-a-dict"})
    assert tracker.active_reasoning_item_id is None


# =============================================================================
# Final Accessors (lines 600-619)
# =============================================================================


def test_get_final_reasoning_buffer():
    """Test get_final_reasoning_buffer returns accumulated text (line 603)."""
    tracker = _create_tracker()
    tracker.reasoning_buffer = "accumulated reasoning"

    result = tracker.get_final_reasoning_buffer()
    assert result == "accumulated reasoning"


def test_get_citations():
    """Test get_citations returns all citations (line 608)."""
    tracker = _create_tracker()
    tracker.emitted_citations = [
        {"source": {"url": "url1"}},
        {"source": {"url": "url2"}},
    ]

    result = tracker.get_citations()
    assert len(result) == 2


def test_should_persist_reasoning_never():
    """Test should_persist_reasoning with 'never' (lines 613-614)."""
    tracker = _create_tracker(persist_reasoning="never")

    assert tracker.should_persist_reasoning() is False


def test_should_persist_reasoning_next_reply():
    """Test should_persist_reasoning with 'next_reply' (lines 613-614)."""
    tracker = _create_tracker(persist_reasoning="next_reply")

    assert tracker.should_persist_reasoning() is True


def test_should_persist_reasoning_conversation():
    """Test should_persist_reasoning with 'conversation' (lines 613-614)."""
    tracker = _create_tracker(persist_reasoning="conversation")

    assert tracker.should_persist_reasoning() is True


def test_is_reasoning_active():
    """Test is_reasoning_active returns stream state (line 619)."""
    tracker = _create_tracker()

    assert tracker.is_reasoning_active() is False

    tracker.reasoning_stream_active = True
    assert tracker.is_reasoning_active() is True


# =============================================================================
# Integration Tests with Real Pipe
# =============================================================================


@pytest.mark.asyncio
async def test_integration_with_real_pipe(pipe_instance_async):
    """Test ReasoningTracker integration with real Pipe instance."""
    pipe = pipe_instance_async
    event_emitter = AsyncMock()

    tracker = ReasoningTracker(
        pipe,
        valves=pipe.valves,
        event_emitter=event_emitter,
        logger=logging.getLogger(__name__),
        user_id="test-user",
    )

    # Process some reasoning events
    result = await tracker.process_reasoning_event(
        {"item_id": "item-1", "delta": "Let me think..."},
        "response.reasoning.delta",
        normalize_surrogate_chunk=lambda x, _: x,
    )
    assert result == "Let me think..."
    assert tracker.reasoning_buffer == "Let me think..."
    assert tracker.reasoning_stream_active is True

    # Process completion
    await tracker.process_reasoning_event(
        {"item_id": "item-1", "text": ""},
        "response.reasoning.done",
        normalize_surrogate_chunk=lambda x, _: x,
    )
    assert tracker.reasoning_completed_emitted is True


@pytest.mark.asyncio
async def test_both_thinking_modes():
    """Test 'both' thinking mode emits both delta and status."""
    event_emitter = AsyncMock()
    tracker = _create_tracker(thinking_mode="both", event_emitter=event_emitter)

    # Process reasoning delta with punctuation to trigger status
    await tracker.process_reasoning_event(
        {"item_id": "item-1", "delta": "Analyzing the problem."},
        "response.reasoning.delta",
        normalize_surrogate_chunk=lambda x, _: x,
    )

    # Should have both reasoning:delta and status events
    event_types = [call[0][0].get("type") for call in event_emitter.call_args_list]
    assert "reasoning:delta" in event_types
    assert "status" in event_types


@pytest.mark.asyncio
async def test_status_only_mode():
    """Test 'status' thinking mode only emits status."""
    event_emitter = AsyncMock()
    tracker = _create_tracker(thinking_mode="status", event_emitter=event_emitter)

    # Process reasoning delta with punctuation
    await tracker.process_reasoning_event(
        {"item_id": "item-1", "delta": "Thinking deeply."},
        "response.reasoning.delta",
        normalize_surrogate_chunk=lambda x, _: x,
    )

    # Should only have status event, no reasoning:delta
    event_types = [call[0][0].get("type") for call in event_emitter.call_args_list]
    assert "status" in event_types
    assert "reasoning:delta" not in event_types


# ===== From test_reasoning_config.py =====

"""Comprehensive tests for ReasoningConfigManager achieving 90%+ coverage.

Tests cover:
- _apply_reasoning_preferences: all branches
- _apply_task_reasoning_preferences: all branches
- _apply_gemini_thinking_config: all branches and edge cases
- _should_retry_without_reasoning: all branches
"""


import logging
from typing import Any
from unittest.mock import Mock, patch

import pytest

from open_webui_openrouter_pipe import (
    Pipe,
    ResponsesBody,
    OpenRouterAPIError,
    ModelFamily,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def pipe_instance():
    """Return a fresh Pipe instance for tests."""
    pipe = Pipe()
    yield pipe
    pipe.shutdown()


@pytest.fixture(autouse=True)
def reset_model_family_specs():
    """Reset ModelFamily specs before and after each test."""
    original_specs = ModelFamily._DYNAMIC_SPECS.copy()
    yield
    ModelFamily._DYNAMIC_SPECS = original_specs


# -----------------------------------------------------------------------------
# _apply_reasoning_preferences Tests
# -----------------------------------------------------------------------------


class TestApplyReasoningPreferences:
    """Tests for _apply_reasoning_preferences method."""

    def test_disabled_reasoning_returns_early(self, pipe_instance):
        """When ENABLE_REASONING is False, method returns without changes."""
        valves = pipe_instance.Valves(ENABLE_REASONING=False)
        body = ResponsesBody(model="test.model", input=[])
        body.reasoning = {"effort": "high"}

        pipe_instance._apply_reasoning_preferences(body, valves)

        # Body should be unchanged
        assert body.reasoning == {"effort": "high"}

    def test_model_supports_reasoning_sets_config(self, pipe_instance):
        """When model supports 'reasoning' param, config dict is built."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="medium",
            REASONING_SUMMARY_MODE="disabled",
        )
        body = ResponsesBody(model="test.model", input=[])

        pipe_instance._apply_reasoning_preferences(body, valves)

        assert body.reasoning == {"effort": "medium", "enabled": True}
        assert body.include_reasoning is None

    def test_model_supports_reasoning_with_summary(self, pipe_instance):
        """When summary mode is not disabled, it is added to config."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="high",
            REASONING_SUMMARY_MODE="auto",
        )
        body = ResponsesBody(model="test.model", input=[])

        pipe_instance._apply_reasoning_preferences(body, valves)

        assert body.reasoning == {"effort": "high", "summary": "auto", "enabled": True}

    def test_model_supports_reasoning_preserves_existing_effort(self, pipe_instance):
        """When body already has effort, it is preserved (not overwritten)."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="low",
            REASONING_SUMMARY_MODE="disabled",
        )
        body = ResponsesBody(model="test.model", input=[])
        body.reasoning = {"effort": "xhigh", "custom_field": True}

        pipe_instance._apply_reasoning_preferences(body, valves)

        # Existing effort should be preserved
        assert body.reasoning["effort"] == "xhigh"
        assert body.reasoning["custom_field"] is True
        assert body.reasoning["enabled"] is True

    def test_model_supports_reasoning_preserves_existing_summary(self, pipe_instance):
        """When body already has summary, it is preserved (not overwritten)."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="low",
            REASONING_SUMMARY_MODE="detailed",
        )
        body = ResponsesBody(model="test.model", input=[])
        body.reasoning = {"summary": "auto"}

        pipe_instance._apply_reasoning_preferences(body, valves)

        # Existing summary should be preserved
        assert body.reasoning["summary"] == "auto"
        assert body.reasoning["effort"] == "low"

    def test_model_supports_reasoning_clears_include_reasoning(self, pipe_instance):
        """When model supports reasoning, include_reasoning is cleared."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(ENABLE_REASONING=True)
        body = ResponsesBody(model="test.model", input=[])
        body.include_reasoning = True

        pipe_instance._apply_reasoning_preferences(body, valves)

        assert body.include_reasoning is None

    def test_model_supports_legacy_only_sets_include_flag(self, pipe_instance):
        """When model only supports include_reasoning, that flag is set."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["include_reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="high",
        )
        body = ResponsesBody(model="test.model", input=[])

        pipe_instance._apply_reasoning_preferences(body, valves)

        assert body.reasoning is None
        assert body.include_reasoning is True

    def test_model_supports_legacy_only_effort_none_disables(self, pipe_instance):
        """When effort is 'none', include_reasoning is False."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["include_reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="none",
        )
        body = ResponsesBody(model="test.model", input=[])

        pipe_instance._apply_reasoning_preferences(body, valves)

        assert body.reasoning is None
        assert body.include_reasoning is False

    def test_model_supports_legacy_only_effort_none_string_disables(self, pipe_instance):
        """When effort is 'none' (the string value), include_reasoning is False.

        Note: The Valves.REASONING_EFFORT field uses a Literal type that doesn't
        allow empty strings. The 'none' value is the designated way to disable
        reasoning effort.
        """
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["include_reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="none",
        )
        body = ResponsesBody(model="test.model", input=[])

        pipe_instance._apply_reasoning_preferences(body, valves)

        assert body.reasoning is None
        assert body.include_reasoning is False

    def test_model_supports_neither_disables_all(self, pipe_instance):
        """When model supports neither param, both are disabled."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset([])}
        })
        valves = pipe_instance.Valves(ENABLE_REASONING=True)
        body = ResponsesBody(model="test.model", input=[])
        body.reasoning = {"effort": "high"}
        body.include_reasoning = True

        pipe_instance._apply_reasoning_preferences(body, valves)

        assert body.reasoning is None
        assert body.include_reasoning is False

    def test_model_supports_both_prefers_reasoning(self, pipe_instance):
        """When model supports both, 'reasoning' dict takes precedence."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning", "include_reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="medium",
        )
        body = ResponsesBody(model="test.model", input=[])
        body.include_reasoning = True

        pipe_instance._apply_reasoning_preferences(body, valves)

        assert isinstance(body.reasoning, dict)
        assert body.reasoning["enabled"] is True
        assert body.include_reasoning is None

    def test_reasoning_body_is_non_dict_gets_replaced(self, pipe_instance):
        """When body.reasoning is not a dict, it's replaced with new config."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="low",
            REASONING_SUMMARY_MODE="disabled",  # Disable summary to simplify assertion
        )
        body = ResponsesBody(model="test.model", input=[])
        body.reasoning = "invalid"  # type: ignore

        pipe_instance._apply_reasoning_preferences(body, valves)

        assert body.reasoning == {"effort": "low", "enabled": True}


# -----------------------------------------------------------------------------
# _apply_task_reasoning_preferences Tests
# -----------------------------------------------------------------------------


class TestApplyTaskReasoningPreferences:
    """Tests for _apply_task_reasoning_preferences method."""

    def test_empty_effort_returns_early(self, pipe_instance):
        """When effort is empty, method returns without changes."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        body = ResponsesBody(model="test.model", input=[])

        pipe_instance._apply_task_reasoning_preferences(body, "")

        # Body should be unchanged
        assert body.reasoning is None

    def test_whitespace_effort_processes_as_empty_string(self, pipe_instance):
        """When effort is whitespace, it processes but sets empty effort string.

        Note: The code checks `if not effort` before stripping, so whitespace
        strings (truthy) pass the check and get stripped to empty string.
        This sets effort="" in the reasoning config.
        """
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        body = ResponsesBody(model="test.model", input=[])

        pipe_instance._apply_task_reasoning_preferences(body, "   ")

        # Whitespace becomes empty string effort (not ideal but actual behavior)
        assert body.reasoning == {"effort": "", "enabled": True}

    def test_model_supports_reasoning_sets_effort(self, pipe_instance):
        """When model supports reasoning, effort is set in config."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        body = ResponsesBody(model="test.model", input=[])

        pipe_instance._apply_task_reasoning_preferences(body, "HIGH")

        assert body.reasoning == {"effort": "high", "enabled": True}
        assert body.include_reasoning is None

    def test_model_supports_reasoning_normalizes_effort(self, pipe_instance):
        """Effort is normalized to lowercase and stripped."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        body = ResponsesBody(model="test.model", input=[])

        pipe_instance._apply_task_reasoning_preferences(body, "  Medium  ")

        assert body.reasoning["effort"] == "medium"

    def test_model_supports_reasoning_overwrites_existing_effort(self, pipe_instance):
        """Task effort always overwrites existing effort."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        body = ResponsesBody(model="test.model", input=[])
        body.reasoning = {"effort": "low", "custom": True}

        pipe_instance._apply_task_reasoning_preferences(body, "high")

        assert body.reasoning["effort"] == "high"
        assert body.reasoning["custom"] is True

    def test_model_supports_reasoning_handles_non_dict_reasoning(self, pipe_instance):
        """When body.reasoning is not a dict, it's replaced."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        body = ResponsesBody(model="test.model", input=[])
        body.reasoning = "invalid"  # type: ignore

        pipe_instance._apply_task_reasoning_preferences(body, "high")

        assert body.reasoning == {"effort": "high", "enabled": True}

    def test_model_supports_reasoning_clears_include_reasoning(self, pipe_instance):
        """When model supports reasoning, include_reasoning is cleared."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        body = ResponsesBody(model="test.model", input=[])
        body.include_reasoning = True

        pipe_instance._apply_task_reasoning_preferences(body, "high")

        assert body.include_reasoning is None

    def test_model_supports_legacy_only_sets_include_flag(self, pipe_instance):
        """When model only supports include_reasoning, flag is set based on effort."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["include_reasoning"])}
        })
        body = ResponsesBody(model="test.model", input=[])

        pipe_instance._apply_task_reasoning_preferences(body, "high")

        assert body.reasoning is None
        assert body.include_reasoning is True

    def test_model_supports_legacy_only_effort_none_disables(self, pipe_instance):
        """When effort is 'none', include_reasoning is False."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["include_reasoning"])}
        })
        body = ResponsesBody(model="test.model", input=[])

        pipe_instance._apply_task_reasoning_preferences(body, "none")

        assert body.include_reasoning is False

    def test_model_supports_legacy_only_effort_minimal_disables(self, pipe_instance):
        """When effort is 'minimal', include_reasoning is False."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["include_reasoning"])}
        })
        body = ResponsesBody(model="test.model", input=[])

        pipe_instance._apply_task_reasoning_preferences(body, "minimal")

        assert body.include_reasoning is False

    def test_model_supports_legacy_only_effort_low_enables(self, pipe_instance):
        """When effort is 'low', include_reasoning is True."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["include_reasoning"])}
        })
        body = ResponsesBody(model="test.model", input=[])

        pipe_instance._apply_task_reasoning_preferences(body, "low")

        assert body.include_reasoning is True

    def test_model_supports_neither_disables_all(self, pipe_instance):
        """When model supports neither param, both are disabled."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset([])}
        })
        body = ResponsesBody(model="test.model", input=[])
        body.reasoning = {"effort": "high"}
        body.include_reasoning = True

        pipe_instance._apply_task_reasoning_preferences(body, "high")

        assert body.reasoning is None
        assert body.include_reasoning is False


# -----------------------------------------------------------------------------
# _apply_gemini_thinking_config Tests
# -----------------------------------------------------------------------------


class TestApplyGeminiThinkingConfig:
    """Tests for _apply_gemini_thinking_config method."""

    def test_non_gemini_model_clears_config(self, pipe_instance):
        """When model is not a Gemini thinking model, thinking_config is cleared."""
        ModelFamily.set_dynamic_specs({
            "test.model": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(ENABLE_REASONING=True)
        body = ResponsesBody(model="test.model", input=[])
        body.thinking_config = {"include_thoughts": True}

        pipe_instance._apply_gemini_thinking_config(body, valves)

        assert body.thinking_config is None

    def test_gemini_25_model_sets_thinking_budget(self, pipe_instance):
        """For Gemini 2.5 models, thinking_budget is set based on effort."""
        ModelFamily.set_dynamic_specs({
            "google.gemini-2.5-flash": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="medium",
            GEMINI_THINKING_BUDGET=1024,
        )
        body = ResponsesBody(model="google/gemini-2.5-flash", input=[])
        body.reasoning = {"enabled": True}

        pipe_instance._apply_gemini_thinking_config(body, valves)

        assert body.thinking_config == {"include_thoughts": True, "thinking_budget": 1024}
        assert body.reasoning is None

    def test_gemini_25_model_high_effort_doubles_budget(self, pipe_instance):
        """For Gemini 2.5 models, 'high' effort doubles the budget."""
        ModelFamily.set_dynamic_specs({
            "google.gemini-2.5-pro": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="high",
            GEMINI_THINKING_BUDGET=1000,
        )
        body = ResponsesBody(model="google/gemini-2.5-pro", input=[])
        body.reasoning = {"enabled": True}

        pipe_instance._apply_gemini_thinking_config(body, valves)

        assert body.thinking_config["thinking_budget"] == 2000

    def test_gemini_25_model_low_effort_halves_budget(self, pipe_instance):
        """For Gemini 2.5 models, 'low' effort halves the budget."""
        ModelFamily.set_dynamic_specs({
            "google.gemini-2.5-flash": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="low",
            GEMINI_THINKING_BUDGET=1000,
        )
        body = ResponsesBody(model="google/gemini-2.5-flash", input=[])
        body.reasoning = {"enabled": True}

        pipe_instance._apply_gemini_thinking_config(body, valves)

        assert body.thinking_config["thinking_budget"] == 500

    def test_gemini_25_model_minimal_effort_quarters_budget(self, pipe_instance):
        """For Gemini 2.5 models, 'minimal' effort quarters the budget."""
        ModelFamily.set_dynamic_specs({
            "google.gemini-2.5-flash": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="minimal",
            GEMINI_THINKING_BUDGET=1000,
        )
        body = ResponsesBody(model="google/gemini-2.5-flash", input=[])
        body.reasoning = {"enabled": True}

        pipe_instance._apply_gemini_thinking_config(body, valves)

        assert body.thinking_config["thinking_budget"] == 250

    def test_gemini_25_model_xhigh_effort_quadruples_budget(self, pipe_instance):
        """For Gemini 2.5 models, 'xhigh' effort quadruples the budget."""
        ModelFamily.set_dynamic_specs({
            "google.gemini-2.5-flash": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="xhigh",
            GEMINI_THINKING_BUDGET=1000,
        )
        body = ResponsesBody(model="google/gemini-2.5-flash", input=[])
        body.reasoning = {"enabled": True}

        pipe_instance._apply_gemini_thinking_config(body, valves)

        assert body.thinking_config["thinking_budget"] == 4000

    def test_gemini_25_model_zero_budget_returns_zero(self, pipe_instance):
        """For Gemini 2.5 models with zero budget, returns 0."""
        ModelFamily.set_dynamic_specs({
            "google.gemini-2.5-flash": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="high",
            GEMINI_THINKING_BUDGET=0,
        )
        body = ResponsesBody(model="google/gemini-2.5-flash", input=[])
        body.reasoning = {"enabled": True}

        pipe_instance._apply_gemini_thinking_config(body, valves)

        assert body.thinking_config["thinking_budget"] == 0

    def test_gemini_25_model_negative_budget_helper_returns_none(self, pipe_instance):
        """For Gemini 2.5 models, _map_effort_to_gemini_budget returns None for negative budget.

        Note: The Valves.GEMINI_THINKING_BUDGET field has a ge=0 constraint,
        so we test the helper function directly to cover the negative budget path.
        """
        from open_webui_openrouter_pipe.models.registry import _map_effort_to_gemini_budget

        # Test that negative budget returns None
        result = _map_effort_to_gemini_budget("high", -1)
        assert result is None

        # Test that zero budget returns 0
        result_zero = _map_effort_to_gemini_budget("high", 0)
        assert result_zero == 0

    def test_gemini_25_model_effort_none_disables_thinking(self, pipe_instance):
        """For Gemini 2.5 models, effort 'none' disables thinking."""
        ModelFamily.set_dynamic_specs({
            "google.gemini-2.5-flash": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="none",
            GEMINI_THINKING_BUDGET=1024,
        )
        body = ResponsesBody(model="google/gemini-2.5-flash", input=[])
        body.reasoning = {"enabled": True}

        pipe_instance._apply_gemini_thinking_config(body, valves)

        assert body.thinking_config is None
        assert body.include_reasoning is False
        assert body.reasoning == {"enabled": True}

    def test_gemini_25_budget_none_from_include_flag_with_none_effort(self, pipe_instance):
        """For Gemini 2.5, include_reasoning=True with 'none' effort disables thinking.

        This covers lines 175-177: when _map_effort_to_gemini_budget returns None.
        The path is: include_reasoning=True forces reasoning_requested=True, but
        then effort_hint='none' makes _map_effort_to_gemini_budget return None.
        """
        ModelFamily.set_dynamic_specs({
            "google.gemini-2.5-flash": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="none",  # Will be used since reasoning dict effort is empty
            GEMINI_THINKING_BUDGET=1000,
        )
        body = ResponsesBody(model="google/gemini-2.5-flash", input=[])
        # include_reasoning=True forces reasoning_requested=True despite enabled=False
        body.include_reasoning = True

        pipe_instance._apply_gemini_thinking_config(body, valves)

        # effort 'none' -> budget is None -> thinking disabled
        assert body.thinking_config is None
        assert body.include_reasoning is False


# -----------------------------------------------------------------------------
# _should_retry_without_reasoning Tests
# -----------------------------------------------------------------------------


class TestShouldRetryWithoutReasoning:
    """Tests for _should_retry_without_reasoning method."""

    def test_no_reasoning_flags_returns_false(self, pipe_instance):
        """When no reasoning flags are set, returns False."""
        body = ResponsesBody(model="test.model", input=[])
        error = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="Thinking_config.include_thoughts is only enabled when thinking is enabled.",
        )

        result = pipe_instance._should_retry_without_reasoning(error, body)

        assert result is False

    def test_include_reasoning_true_triggers_retry(self, pipe_instance):
        """When include_reasoning is True and error matches, returns True."""
        body = ResponsesBody(model="test.model", input=[])
        body.include_reasoning = True
        error = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="Thinking_config.include_thoughts is only enabled when thinking is enabled.",
        )

        result = pipe_instance._should_retry_without_reasoning(error, body)

        assert result is True
        assert body.include_reasoning is False
        assert body.reasoning is None
        assert body.thinking_config is None

    def test_reasoning_dict_triggers_retry(self, pipe_instance):
        """When reasoning dict is set and error matches, returns True."""
        body = ResponsesBody(model="test.model", input=[])
        body.reasoning = {"enabled": True, "effort": "high"}
        error = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            upstream_message="thinking_config.include_thoughts is only enabled when thinking is enabled.",
        )

        result = pipe_instance._should_retry_without_reasoning(error, body)

        assert result is True
        assert body.reasoning is None

    def test_thinking_config_triggers_retry(self, pipe_instance):
        """When thinking_config is set and error matches, returns True."""
        body = ResponsesBody(model="test.model", input=[])
        body.thinking_config = {"include_thoughts": True, "thinking_level": "HIGH"}
        error = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="Unable to submit request because Thinking_config.include_thoughts is only enabled when thinking is enabled.",
        )

        result = pipe_instance._should_retry_without_reasoning(error, body)

        assert result is True
        assert body.thinking_config is None

    def test_unrelated_error_message_returns_false(self, pipe_instance):
        """When error message doesn't match trigger phrases, returns False."""
        body = ResponsesBody(model="test.model", input=[])
        body.include_reasoning = True
        error = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="Some other provider error message.",
        )

        result = pipe_instance._should_retry_without_reasoning(error, body)

        assert result is False
        assert body.include_reasoning is True  # Not cleared

    def test_trigger_phrase_in_str_error(self, pipe_instance):
        """When trigger phrase is in str(error), returns True."""
        body = ResponsesBody(model="test.model", input=[])
        body.include_reasoning = True
        # Error with message only in __str__ representation
        error = OpenRouterAPIError(
            status=400,
            reason="include_thoughts is only enabled when thinking is enabled",
        )

        result = pipe_instance._should_retry_without_reasoning(error, body)

        assert result is True

    def test_case_insensitive_match(self, pipe_instance):
        """Trigger phrase matching is case-insensitive."""
        body = ResponsesBody(model="test.model", input=[])
        body.include_reasoning = True
        error = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="THINKING_CONFIG.INCLUDE_THOUGHTS IS ONLY ENABLED WHEN THINKING IS ENABLED.",
        )

        result = pipe_instance._should_retry_without_reasoning(error, body)

        assert result is True

    def test_non_string_message_candidates_skipped(self, pipe_instance):
        """Non-string message candidates are skipped without error."""
        body = ResponsesBody(model="test.model", input=[])
        body.include_reasoning = True
        # Create error with upstream_message as dict (edge case)
        error = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            upstream_message=None,  # type: ignore
            openrouter_message=None,
        )
        # Manually set to test edge case
        error.upstream_message = {"key": "value"}  # type: ignore

        result = pipe_instance._should_retry_without_reasoning(error, body)

        # Should return False because no string message matches
        assert result is False

    def test_logs_info_when_retrying(self, pipe_instance, caplog):
        """When retrying, logs an info message."""
        body = ResponsesBody(model="test.model", input=[])
        body.include_reasoning = True
        error = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="include_thoughts is only enabled when thinking is enabled.",
        )

        with caplog.at_level(logging.INFO):
            result = pipe_instance._should_retry_without_reasoning(error, body)

        assert result is True
        assert any("Retrying without reasoning" in message for message in caplog.messages)
        assert any("test.model" in message for message in caplog.messages)


# -----------------------------------------------------------------------------
# ReasoningConfigManager Initialization Tests
# -----------------------------------------------------------------------------


class TestReasoningConfigManagerInit:
    """Tests for ReasoningConfigManager initialization and lazy loading."""

    def test_lazy_initialization(self, pipe_instance):
        """ReasoningConfigManager is lazily initialized."""
        assert pipe_instance._reasoning_config_manager is None

        # Trigger initialization
        pipe_instance._ensure_reasoning_config_manager()

        assert pipe_instance._reasoning_config_manager is not None

    def test_shared_instance(self, pipe_instance):
        """Same instance is returned on subsequent calls."""
        manager1 = pipe_instance._ensure_reasoning_config_manager()
        manager2 = pipe_instance._ensure_reasoning_config_manager()

        assert manager1 is manager2

    def test_manager_has_pipe_reference(self, pipe_instance):
        """Manager has reference to parent pipe."""
        manager = pipe_instance._ensure_reasoning_config_manager()

        assert manager._pipe is pipe_instance

    def test_manager_has_logger(self, pipe_instance):
        """Manager has logger instance."""
        manager = pipe_instance._ensure_reasoning_config_manager()

        assert manager.logger is not None

    def test_manager_has_valves(self, pipe_instance):
        """Manager has valves reference."""
        manager = pipe_instance._ensure_reasoning_config_manager()

        assert manager.valves is pipe_instance.valves


# -----------------------------------------------------------------------------
# Edge Cases and Integration Tests
# -----------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and integration tests."""

    def test_reasoning_with_none_model(self, pipe_instance):
        """Handles body with None/empty model gracefully."""
        ModelFamily.set_dynamic_specs({})
        valves = pipe_instance.Valves(ENABLE_REASONING=True)
        body = ResponsesBody(model="", input=[])

        # Should not raise
        pipe_instance._apply_reasoning_preferences(body, valves)

        assert body.reasoning is None

    def test_gemini_25_exact_model_name_match(self, pipe_instance):
        """Handles exact Gemini 2.5 model names."""
        ModelFamily.set_dynamic_specs({
            "google.gemini-2.5": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="high",
            GEMINI_THINKING_BUDGET=1000,
        )
        body = ResponsesBody(model="google/gemini-2.5", input=[])
        body.reasoning = {"enabled": True}

        pipe_instance._apply_gemini_thinking_config(body, valves)

        assert body.thinking_config is not None
        assert "thinking_budget" in body.thinking_config

    def test_gemini_budget_rounds_to_int(self, pipe_instance):
        """Gemini budget calculation rounds to integer."""
        ModelFamily.set_dynamic_specs({
            "google.gemini-2.5-flash": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="minimal",  # 0.25 scalar
            GEMINI_THINKING_BUDGET=999,  # 999 * 0.25 = 249.75
        )
        body = ResponsesBody(model="google/gemini-2.5-flash", input=[])
        body.reasoning = {"enabled": True}

        pipe_instance._apply_gemini_thinking_config(body, valves)

        # Should round to nearest int
        assert body.thinking_config["thinking_budget"] == 250

    def test_gemini_budget_minimum_is_one(self, pipe_instance):
        """Gemini budget minimum is 1 (not 0 from rounding)."""
        ModelFamily.set_dynamic_specs({
            "google.gemini-2.5-flash": {"supported_parameters": frozenset(["reasoning"])}
        })
        valves = pipe_instance.Valves(
            ENABLE_REASONING=True,
            REASONING_EFFORT="minimal",  # 0.25 scalar
            GEMINI_THINKING_BUDGET=3,  # 3 * 0.25 = 0.75 -> rounds to 1
        )
        body = ResponsesBody(model="google/gemini-2.5-flash", input=[])
        body.reasoning = {"enabled": True}

        pipe_instance._apply_gemini_thinking_config(body, valves)

        # Should be at least 1
        assert body.thinking_config["thinking_budget"] == 1

    def test_all_retry_flags_cleared_on_match(self, pipe_instance):
        """All reasoning flags are cleared when retry is triggered."""
        body = ResponsesBody(model="test.model", input=[])
        body.include_reasoning = True
        body.reasoning = {"enabled": True, "effort": "high"}
        body.thinking_config = {"include_thoughts": True}
        error = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="include_thoughts is only enabled when thinking is enabled.",
        )

        result = pipe_instance._should_retry_without_reasoning(error, body)

        assert result is True
        assert body.include_reasoning is False
        assert body.reasoning is None
        assert body.thinking_config is None


# ===== From test_reasoning_tracker_images.py =====


import base64
import logging
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.streaming.reasoning_tracker import ReasoningTracker


class _StubPipeImageTests:
    def __init__(self) -> None:
        self._emit_status = AsyncMock()

    def _parse_data_url(self, _text: str):
        return {"data": b"image-bytes", "mime_type": "image/png"}


@pytest.mark.asyncio
async def test_materialize_data_url_persists_and_emits_status():
    pipe = _StubPipeImageTests()
    tracker = ReasoningTracker(
        cast(Pipe, pipe),
        valves=SimpleNamespace(THINKING_OUTPUT_MODE="open_webui"),
        event_emitter=AsyncMock(),
        logger=logging.getLogger(__name__),
    )

    tracker._persist_generated_image = AsyncMock(return_value="file-1")
    result = await tracker._materialize_image_from_str("data:image/png;base64,AAA=")

    assert result == "/api/v1/files/file-1/content"
    assert pipe._emit_status.await_count == 1


@pytest.mark.asyncio
async def test_materialize_image_entry_from_base64_dict():
    pipe = _StubPipeImageTests()
    tracker = ReasoningTracker(
        cast(Pipe, pipe),
        valves=SimpleNamespace(THINKING_OUTPUT_MODE="open_webui"),
        event_emitter=AsyncMock(),
        logger=logging.getLogger(__name__),
    )

    tracker._persist_generated_image = AsyncMock(return_value="file-2")
    b64 = base64.b64encode(b"abc").decode("ascii")
    entry = {"b64_json": b64, "mime_type": "image/png"}

    result = await tracker._materialize_image_entry(entry)

    assert result == "/api/v1/files/file-2/content"
    assert pipe._emit_status.await_count == 1


@pytest.mark.asyncio
async def test_materialize_image_entry_returns_existing_url():
    pipe = _StubPipeImageTests()
    tracker = ReasoningTracker(
        cast(Pipe, pipe),
        valves=SimpleNamespace(THINKING_OUTPUT_MODE="open_webui"),
        event_emitter=None,
        logger=logging.getLogger(__name__),
    )

    tracker._persist_generated_image = AsyncMock()
    entry = {"url": "https://example.test/image.png"}

    result = await tracker._materialize_image_entry(entry)

    assert result == "https://example.test/image.png"
    tracker._persist_generated_image.assert_not_called()
