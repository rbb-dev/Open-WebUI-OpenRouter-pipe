"""Core streaming loop and response handling.

This module provides StreamingHandler for SSE streaming, non-streaming responses,
and endpoint selection.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import uuid
import time
import datetime
import base64
import binascii
import contextlib
import random
import aiohttp
from time import perf_counter
from typing import Any, Optional, Dict, Literal, no_type_check, TYPE_CHECKING
from fastapi import Request

# Import parent module classes that are needed at runtime
if TYPE_CHECKING:
    from ..pipe import Pipe
    from ..api.transforms import ResponsesBody
else:
    # At runtime, avoid circular imports
    Pipe = Any
    ResponsesBody = Any

# Import error classes
from ..core.errors import OpenRouterAPIError, StatusMessages

# Import costs helper
from ..core.costs import maybe_dump_costs_snapshot

# Import request sanitizer
from ..requests.sanitizer import _sanitize_request_input

# Import Anthropic integration
from ..integrations.anthropic import _maybe_apply_anthropic_prompt_caching

# Import SessionLogger
from ..core.logging_system import SessionLogger

# Import timing instrumentation
from ..core.timing_logger import timed, timing_scope, timing_mark
from .constants import (
    REASONING_STATUS_PUNCTUATION as _REASONING_STATUS_PUNCTUATION,
    REASONING_STATUS_MAX_CHARS as _REASONING_STATUS_MAX_CHARS,
    REASONING_STATUS_MIN_CHARS as _REASONING_STATUS_MIN_CHARS,
    REASONING_STATUS_IDLE_SECONDS as _REASONING_STATUS_IDLE_SECONDS,
)

# Imports from storage.persistence
from ..storage.persistence import (
    normalize_persisted_item as _normalize_persisted_item,
)
# Imports from core.utils
from ..core.utils import (
    wrap_code_block,
    _redact_payload_blobs,
    merge_usage_stats,
    _render_error_template,
    _serialize_marker,
    _safe_json_loads,
)
# Imports from models.registry
from ..models.registry import (
    _parse_model_patterns,
    _matches_any_model_pattern,
    ModelFamily,
)

# Import transform functions
from ..api.transforms import (
    _apply_identifier_valves_to_payload,
    _apply_model_fallback_to_payload,
    _apply_disable_native_websearch_to_payload,
    _strip_disable_model_settings_params,
)

# Import config classes
from ..core.config import EncryptedStr, DEFAULT_MAX_FUNCTION_CALL_LOOPS_REACHED_TEMPLATE

# Import Open WebUI models
try:
    from open_webui.models.chats import Chats  # type: ignore[import-not-found]
except ImportError:
    Chats = None  # type: ignore

# Import citation extraction function (OpenWebUI >= 0.7.0)
try:
    from open_webui.utils.middleware import get_citation_source_from_tool_result  # type: ignore[import-not-found]
except ImportError:
    get_citation_source_from_tool_result = None  # type: ignore

# Import OWUI's apply_source_context_to_messages for source context injection
# This is the authoritative implementation - we use adapter transforms to support Responses API
try:
    from open_webui.utils.middleware import apply_source_context_to_messages as _owui_apply_source_context  # type: ignore[import-not-found]
except ImportError:
    _owui_apply_source_context = None  # type: ignore

# Import our transform function for Responses API → Chat Completions conversion
from ..api.transforms import _responses_input_to_chat_messages


def _chat_messages_to_responses_input(messages: list) -> list:
    """
    Convert Chat Completions messages back to Responses API input format.

    TRUE ADAPTER PATTERN: Start with original, transform only what we know,
    pass through everything else unchanged. Never filter/drop unknown fields.

    Transforms applied:
    - Block type: "text" → "input_text"
    - Block type: "image_url" → "input_image" (also flattens nested url)
    - Message: adds "type": "message" wrapper

    Everything else passes through unchanged (unknown fields, future additions, etc.)
    """
    result = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue

        content = msg.get("content")

        # Transform content blocks
        content_blocks: list[dict] = []
        if isinstance(content, str):
            content_blocks.append({"type": "input_text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type", "")

                if btype == "text":
                    # Transform text → input_text, preserving ALL other fields
                    transformed = dict(block)  # Copy everything
                    transformed["type"] = "input_text"  # Only change type
                    content_blocks.append(transformed)

                elif btype == "image_url":
                    # Transform image_url → input_image, flatten nested url
                    transformed = dict(block)  # Copy everything
                    transformed["type"] = "input_image"
                    # Flatten nested {"url": "..."} to just the url string
                    img_url = transformed.pop("image_url", {})
                    if isinstance(img_url, dict):
                        transformed["image_url"] = img_url.get("url", "")
                        # Preserve detail if nested
                        if "detail" in img_url and "detail" not in transformed:
                            transformed["detail"] = img_url["detail"]
                    else:
                        transformed["image_url"] = str(img_url) if img_url else ""
                    content_blocks.append(transformed)

                elif btype == "file":
                    # Transform file → input_file (reverse of _responses_input_to_chat_messages)
                    # This ensures round-trip integrity when source context injection modifies messages
                    transformed = dict(block)  # Copy everything
                    transformed["type"] = "input_file"
                    file_payload = transformed.pop("file", {})
                    if isinstance(file_payload, dict):
                        if file_payload.get("filename"):
                            transformed["filename"] = file_payload["filename"]
                        if file_payload.get("file_data"):
                            transformed["file_data"] = file_payload["file_data"]
                    content_blocks.append(transformed)

                else:
                    # Pass through ALL other block types unchanged
                    content_blocks.append(block)

        if content_blocks:
            # Start with original message, add/update only what we need
            out_msg = dict(msg)  # Copy ALL fields from original
            out_msg["type"] = "message"  # Ensure type is set
            out_msg["content"] = content_blocks  # Replace with transformed content
            result.append(out_msg)

    return result


def _apply_source_context_responses_api(
    input_items: list,
    sources: list,
    user_message: str,
    request_context: Optional[Request] = None,
) -> list:
    """
    Apply source context to messages in Responses API format.

    Uses adapter pattern:
    1. Separate messages from non-message items (function_call, function_call_output)
    2. Convert messages: Responses API input → Chat Completions messages
    3. Apply OWUI's apply_source_context_to_messages()
    4. Convert back: Chat Completions → Responses API input
    5. Reassemble: modified messages first, then non-message items appended

    Note: Non-message items (function_call, function_call_output) are appended
    after transformed messages. For typical input (messages only), order is preserved.

    This delegates all citation logic to OWUI's implementation while preserving
    function_call and function_call_output items that must stay in the input.
    """
    _logger = logging.getLogger("open_webui_openrouter_pipe.streaming.source_context")

    if not sources or not user_message:
        return input_items

    if _owui_apply_source_context is None:
        _logger.warning("OWUI apply_source_context_to_messages not available - skipping source context")
        return input_items

    if request_context is None:
        _logger.warning("request_context is None - cannot apply source context")
        return input_items

    # Step 1: Separate messages from non-message items (function_call, function_call_output, etc.)
    # We need to preserve non-message items exactly as they were
    messages_only: list[dict] = []
    non_message_items: list[tuple[int, dict]] = []  # (original_index, item)

    for idx, item in enumerate(input_items):
        if isinstance(item, dict) and item.get("type") == "message":
            messages_only.append(item)
        elif isinstance(item, dict):
            non_message_items.append((idx, item))

    _logger.debug(
        "Separated input: %d messages, %d non-message items (function_call/output)",
        len(messages_only),
        len(non_message_items),
    )

    if not messages_only:
        # No messages to process - return original
        return input_items

    # Step 2: Convert Responses API messages → Chat Completions messages
    chat_messages = _responses_input_to_chat_messages(messages_only, allow_unknown_fields=True)
    _logger.debug("Converted %d Responses API messages → %d Chat Completions messages", len(messages_only), len(chat_messages))

    # Step 3: Apply OWUI's source context injection (handles RAG template, citation formatting)
    modified_chat_messages = _owui_apply_source_context(
        request_context,
        chat_messages,
        sources,
        user_message,
    )
    _logger.debug("Applied source context via OWUI adapter")

    # Step 4: Convert back Chat Completions → Responses API input
    modified_messages = _chat_messages_to_responses_input(modified_chat_messages)
    _logger.debug("Converted %d Chat Completions messages → %d Responses API messages", len(modified_chat_messages), len(modified_messages))

    # Step 5: Reassemble - messages first, then non-message items in their relative order
    # This preserves the expected structure: messages, then function_call items, then function_call_output items
    result = list(modified_messages)
    for _, item in non_message_items:
        result.append(item)

    _logger.debug(
        "Reassembled result: %d total items (%d messages + %d non-message)",
        len(result),
        len(modified_messages),
        len(non_message_items),
    )

    return result

# Type hints for Open WebUI components
EventEmitter = Any  # Callable[[dict[str, Any]], Awaitable[None]]

LOGGER = logging.getLogger(__name__)


class StreamingHandler:
    """Manages streaming response processing.

    This class encapsulates the core streaming logic including:
    - SSE event parsing and delta accumulation
    - Tool call extraction
    - Reasoning status tracking
    - Usage metrics aggregation
    - Endpoint selection (/responses vs /chat/completions)

    Dependencies:
        - logger: Logger instance for diagnostic output
        - valves: Configuration valves (Pipe.Valves instance)
        - model_registry: Model capability registry
        - Various Pipe methods and utilities
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        valves: Any,
        model_registry: Any,
        pipe_instance: Any,  # Reference to Pipe instance for helper methods
    ):
        """Initialize StreamingHandler with dependencies.

        Args:
            logger: Logger instance for diagnostic output
            valves: Configuration valves (Pipe.Valves instance)
            model_registry: Model capability registry
            pipe_instance: Reference to parent Pipe instance for helper methods
        """
        self.logger = logger
        self.valves = valves
        self._model_registry = model_registry
        self._pipe = pipe_instance

    @timed
    async def _run_streaming_loop(  # pyright: ignore[reportGeneralTypeIssues]
        self,
        body: ResponsesBody,
        valves: Pipe.Valves,
        event_emitter: EventEmitter | None,
        metadata: dict[str, Any] = {},
        tools: dict[str, Dict[str, Any]] | list[dict[str, Any]] | None = None,
        session: aiohttp.ClientSession | None = None,
        user_id: str = "",
        *,
        endpoint_override: Literal["responses", "chat_completions"] | None = None,
        request_context: Optional[Request] = None,
        user_obj: Optional[Any] = None,
        pipe_identifier: Optional[str] = None,
    ):
        """
        Stream assistant responses incrementally, handling function calls, status updates, and tool usage.
        """
        if session is None:
            raise RuntimeError("HTTP session is required for streaming")

        if not isinstance(metadata, dict):
            metadata = {}

        owui_tool_passthrough = getattr(valves, "TOOL_EXECUTION_MODE", "Pipeline") == "Open-WebUI"
        persist_tools_enabled = bool(valves.PERSIST_TOOL_RESULTS) and (not owui_tool_passthrough)
        self.logger.debug(
            "🔧 TOOL_EXECUTION_MODE=%s owui_passthrough=%s PERSIST_TOOL_RESULTS=%s effective_persist_tools=%s",
            getattr(valves, "TOOL_EXECUTION_MODE", "Pipeline"),
            owui_tool_passthrough,
            bool(valves.PERSIST_TOOL_RESULTS),
            persist_tools_enabled,
        )
        self.logger.debug("Streaming config: direct pass-through (no server batching)")
        streamed_tool_call_ids: set[str] = set()
        streamed_tool_call_indices: dict[str, int] = {}
        tool_call_names: dict[str, str] = {}
        streamed_tool_call_args: dict[str, str] = {}
        streamed_tool_call_name_sent: set[str] = set()
        session_log_reason: str = ""

        raw_tools = tools or {}
        tool_registry: dict[str, dict[str, Any]] = {}
        if isinstance(raw_tools, dict):
            tool_registry = raw_tools
        elif isinstance(raw_tools, list):
            skipped_no_callable = False
            for entry in raw_tools:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not name and isinstance(entry.get("spec"), dict):
                    name = entry["spec"].get("name")
                callable_obj = entry.get("callable")
                if callable_obj is None:
                    skipped_no_callable = True
                    continue
                if name:
                    tool_registry[name] = entry
            if skipped_no_callable and not tool_registry:
                self.logger.warning("Received list-based tools without callables; tool execution will be disabled for this request.")
        model_block = metadata.get("model")
        openwebui_model = model_block.get("id", "") if isinstance(model_block, dict) else ""
        assistant_message = ""
        pending_ulids: list[str] = []
        pending_items: list[dict[str, Any]] = []
        total_usage: dict[str, Any] = {}
        reasoning_buffer = ""
        reasoning_completed_emitted = False
        reasoning_stream_active = False
        active_reasoning_item_id: str | None = None
        reasoning_stream_buffers: dict[str, str] = {}
        reasoning_stream_has_incremental: set[str] = set()
        reasoning_stream_completed: set[str] = set()
        ordinal_by_url: dict[str, int] = {}
        emitted_citations: list[dict] = []
        annotation_citations_emitted = False
        chat_id = metadata.get("chat_id")
        message_id = metadata.get("message_id")
        model_started = asyncio.Event()
        responding_status_sent = False
        provider_status_seen = False
        generation_started_at: float | None = None
        generation_last_event_at: float | None = None
        response_completed_at: float | None = None
        stream_started_at: float | None = None
        surrogate_carry: dict[str, str] = {"assistant": "", "reasoning": ""}
        storage_context_cache: Optional[tuple[Optional[Request], Optional[Any]]] = None
        processed_image_item_ids: set[str] = set()
        generated_image_count = 0
        reasoning_status_buffer = ""
        reasoning_status_last_emit: float | None = None
        thinking_mode = valves.THINKING_OUTPUT_MODE
        thinking_box_enabled = thinking_mode in {"open_webui", "both"}
        thinking_status_enabled = thinking_mode in {"status", "both"}

        async def _maybe_emit_reasoning_status(delta_text: str, *, force: bool = False) -> None:
            """Emit readable status updates for reasoning text without flooding."""
            nonlocal reasoning_status_buffer, reasoning_status_last_emit
            if not thinking_status_enabled:
                return
            if not event_emitter:
                return
            reasoning_status_buffer += delta_text
            text = reasoning_status_buffer.strip()
            if not text:
                return
            should_emit = force
            now = perf_counter()
            if not should_emit:
                if delta_text.rstrip().endswith(_REASONING_STATUS_PUNCTUATION):
                    should_emit = True
                elif len(text) >= _REASONING_STATUS_MAX_CHARS:
                    should_emit = True
                else:
                    elapsed = None if reasoning_status_last_emit is None else (now - reasoning_status_last_emit)
                    if len(text) >= _REASONING_STATUS_MIN_CHARS:
                        if elapsed is None or elapsed >= _REASONING_STATUS_IDLE_SECONDS:
                            should_emit = True
            if not should_emit:
                return
            await event_emitter(
                {
                    "type": "status",
                    "data": {"description": text, "done": False},
                }
            )
            reasoning_status_buffer = ""
            reasoning_status_last_emit = now

        async def _get_storage_context() -> tuple[Optional[Request], Optional[Any]]:
            nonlocal storage_context_cache
            if storage_context_cache is None:
                storage_context_cache = await self._pipe._multimodal_handler._resolve_storage_context(request_context, user_obj)
            return storage_context_cache or (None, None)

        @timed
        async def _persist_generated_image(data: bytes, mime_type: str) -> Optional[str]:
            upload_request, upload_user = await _get_storage_context()
            if not upload_request or not upload_user:
                return None
            ext = "png"
            if isinstance(mime_type, str) and "/" in mime_type:
                ext = mime_type.split("/")[-1] or "png"
            if ext == "jpg":
                ext = "jpeg"
            filename = f"generated-image-{uuid.uuid4().hex}.{ext}"
            return await self._pipe._multimodal_handler._upload_to_owui_storage(
                request=upload_request,
                user=upload_user,
                file_data=data,
                filename=filename,
                mime_type=mime_type,
                chat_id=chat_id if isinstance(chat_id, str) else None,
                message_id=message_id if isinstance(message_id, str) else None,
                owui_user_id=user_id,
            )

        @timed
        async def _materialize_image_from_str(data_str: str) -> Optional[str]:
            text = (data_str or "").strip()
            if not text:
                return None
            if text.startswith("data:"):
                parsed = self._pipe._multimodal_handler._parse_data_url(text)
                if parsed:
                    stored = await _persist_generated_image(parsed["data"], parsed["mime_type"])
                    if stored:
                        await self._pipe._event_emitter_handler._emit_status(event_emitter, StatusMessages.IMAGE_BASE64_SAVED, done=False)
                        return f"/api/v1/files/{stored}/content"
                    return text
                return None
            if text.startswith(("http://", "https://", "/")):
                return text
            cleaned = text
            if "," in cleaned and ";base64" in cleaned.split(",", 1)[0]:
                cleaned = cleaned.split(",", 1)[1]
            cleaned = cleaned.strip()
            if not cleaned:
                return None
            if not self._pipe._multimodal_handler._validate_base64_size(cleaned):
                return None
            try:
                decoded = base64.b64decode(cleaned, validate=True)
            except (binascii.Error, ValueError):
                return None
            mime_type = "image/png"
            stored = await _persist_generated_image(decoded, mime_type)
            if stored:
                await self._pipe._event_emitter_handler._emit_status(event_emitter, StatusMessages.IMAGE_BASE64_SAVED, done=False)
                return f"/api/v1/files/{stored}/content"
            return f"data:{mime_type};base64,{cleaned}"

        @timed
        async def _materialize_image_entry(entry: Any) -> Optional[str]:
            if entry is None:
                return None
            if isinstance(entry, str):
                return await _materialize_image_from_str(entry)
            if isinstance(entry, dict):
                for key in ("url", "image_url", "imageUrl", "content_url"):
                    candidate = entry.get(key)
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()
                    if isinstance(candidate, dict):
                        nested = await _materialize_image_entry(candidate)
                        if nested:
                            return nested
                for key in ("b64_json", "b64", "base64", "data", "image_base64"):
                    b64_val = entry.get(key)
                    if isinstance(b64_val, str) and b64_val.strip():
                        cleaned = b64_val.strip()
                        if not self._pipe._multimodal_handler._validate_base64_size(cleaned):
                            continue
                        try:
                            decoded = base64.b64decode(cleaned, validate=True)
                        except (binascii.Error, ValueError):
                            continue
                        mime_type = (
                            entry.get("mime_type")
                            or entry.get("mimeType")
                            or "image/png"
                        ).lower()
                        if mime_type == "image/jpg":
                            mime_type = "image/jpeg"
                        stored = await _persist_generated_image(decoded, mime_type)
                        if stored:
                            await self._pipe._event_emitter_handler._emit_status(event_emitter, StatusMessages.IMAGE_BASE64_SAVED, done=False)
                            return f"/api/v1/files/{stored}/content"
                        return f"data:{mime_type};base64,{cleaned}"
                nested_result = entry.get("result")
                if nested_result is not None:
                    return await _materialize_image_entry(nested_result)
            return None

        async def _collect_image_output_urls(item: dict[str, Any]) -> list[str]:
            payload = item.get("result")
            urls: list[str] = []
            if isinstance(payload, list):
                for entry in payload:
                    resolved = await _materialize_image_entry(entry)
                    if resolved:
                        urls.append(resolved)
            else:
                resolved = await _materialize_image_entry(payload)
                if resolved:
                    urls.append(resolved)
            return urls

        async def _render_image_markdown(item: dict[str, Any]) -> list[str]:
            nonlocal generated_image_count
            urls = await _collect_image_output_urls(item)
            markdowns: list[str] = []
            for url in urls:
                generated_image_count += 1
                label = item.get("label") or f"Generated image {generated_image_count}"
                alt_text = re.sub(r"[\r\n]+", " ", str(label)).strip() or f"Generated image {generated_image_count}"
                markdowns.append(f"![{alt_text}]({url})")
            return markdowns

        def _append_output_block(current: str, block: str) -> str:
            snippet = (block or "").strip()
            if not snippet:
                return current
            if current:
                if not current.endswith("\n"):
                    current += "\n"
                if not current.endswith("\n\n"):
                    current += "\n"
            return f"{current}{snippet}\n"

        def _normalize_surrogate_chunk(text: str, bucket: str) -> str:
            """Coalesce surrogate pairs in streaming chunks to keep UTF-8 happy."""
            prev = surrogate_carry.get(bucket, "")
            combined = f"{prev}{text or ''}"
            if not combined:
                surrogate_carry[bucket] = ""
                return ""
            new_carry = ""
            try:
                normalized = combined.encode("utf-16", "surrogatepass").decode("utf-16")
            except UnicodeDecodeError:
                if combined:
                    last_char = combined[-1]
                    if 0xD800 <= ord(last_char) <= 0xDBFF:
                        new_carry = last_char
                        combined = combined[:-1]
                normalized = combined.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
            surrogate_carry[bucket] = new_carry
            return normalized

        def _extract_reasoning_text(event: dict[str, Any]) -> str:
            """Return best-effort reasoning text from assorted event payloads."""
            if not isinstance(event, dict):
                return ""
            for key in ("delta", "text"):
                value = event.get(key)
                if isinstance(value, str) and value:
                    return value
            part = event.get("part")
            if isinstance(part, dict):
                part_text = part.get("text")
                if isinstance(part_text, str) and part_text:
                    return part_text
                content = part.get("content")
                if isinstance(content, list):
                    fragments: list[str] = []
                    for entry in content:
                        if isinstance(entry, dict):
                            text_val = entry.get("text")
                            if isinstance(text_val, str):
                                fragments.append(text_val)
                        elif isinstance(entry, str):
                            fragments.append(entry)
                    if fragments:
                        return "".join(fragments)
            return ""

        def _reasoning_stream_key(event: dict[str, Any], etype: Optional[str]) -> str:
            """Associate reasoning deltas/snapshots with a stable upstream item id when possible."""
            item_id = event.get("item_id")
            if isinstance(item_id, str) and item_id:
                return item_id
            if etype in {"response.output_item.added", "response.output_item.done"}:
                item_raw = event.get("item")
                item = item_raw if isinstance(item_raw, dict) else {}
                iid = item.get("id")
                if isinstance(iid, str) and iid:
                    return iid
            if active_reasoning_item_id:
                return active_reasoning_item_id
            return "__reasoning__"

        def _extract_reasoning_text_from_item(item: dict[str, Any]) -> str:
            """Extract reasoning content/summary from a completed output item."""
            if not isinstance(item, dict):
                return ""
            fragments: list[str] = []
            content = item.get("content")
            if isinstance(content, list):
                for entry in content:
                    if isinstance(entry, dict):
                        text_val = entry.get("text")
                        if isinstance(text_val, str) and text_val:
                            fragments.append(text_val)
            if fragments:
                return "".join(fragments)
            summary = item.get("summary")
            if isinstance(summary, list):
                for entry in summary:
                    if isinstance(entry, dict):
                        text_val = entry.get("text")
                        if isinstance(text_val, str) and text_val:
                            fragments.append(text_val)
            return "".join(fragments)

        def _append_reasoning_text(key: str, incoming: str, *, allow_misaligned: bool) -> str:
            """Coalesce cumulative/snapshot reasoning payloads into a single stream without replay."""
            nonlocal reasoning_buffer
            candidate = (incoming or "")
            if not candidate:
                return ""
            current = reasoning_stream_buffers.get(key, "")
            append = ""
            if not current:
                append = candidate
            elif candidate == current:
                append = ""
            elif candidate.startswith(current):
                append = candidate[len(current) :]
            elif current.startswith(candidate):
                append = ""
            else:
                append = candidate if allow_misaligned else ""
            if append:
                reasoning_stream_buffers[key] = f"{current}{append}"
                reasoning_buffer += append
            return append

        @timed
        async def _flush_pending(reason: str) -> None:
            """Persist buffered artifacts and emit a warning when the DB fails."""
            if not pending_items:
                return
            rows = pending_items[:]
            # Clear immediately so a failure here does not cause the same rows
            # to be re-attempted during later flushes. Callers report the error
            # to the UI instead of silently retrying.
            pending_items.clear()
            try:
                ulids = await self._pipe._artifact_store._db_persist(rows)
            except Exception as exc:  # pragma: no cover - DB errors handled later
                self.logger.error("Failed to persist response artifacts (%s): %s", reason, exc, exc_info=self.logger.isEnabledFor(logging.DEBUG))
                if event_emitter:
                    await event_emitter(
                        {
                            "type": "status",
                            "data": {"description": "⚠️ Tool storage unavailable", "done": False},
                        }
                    )
                return
            pending_ulids.extend(ulids)

        thinking_tasks: list[asyncio.Task] = []
        thinking_cancelled = False
        if event_emitter:
            async def _later(delay: float, msg: str) -> None:
                """Emit a delayed status update to reassure the user during long thoughts."""
                try:
                    await asyncio.wait_for(model_started.wait(), timeout=delay)
                    return
                except asyncio.TimeoutError:
                    if model_started.is_set():
                        return
                await event_emitter({"type": "status", "data": {"description": msg}})

            thinking_tasks = []
            for delay, msg in [
                (0, "Thinking…"),
                (1.5, "Reading the user's question…"),
                (4.0, "Gathering my thoughts…"),
                (6.0, "Exploring possible responses…"),
                (7.0, "Building a plan…"),
            ]:
                if delay == 0:
                    await event_emitter({"type": "status", "data": {"description": msg}})
                    continue
                thinking_tasks.append(
                    asyncio.create_task(_later(delay + random.uniform(0, 0.5), msg))
                )

        def cancel_thinking() -> None:
            """Cancel any scheduled reasoning status updates once the loop completes."""
            nonlocal thinking_cancelled
            if thinking_cancelled:
                return
            thinking_cancelled = True
            for t in thinking_tasks:
                t.cancel()

        def note_model_activity() -> None:
            """Mark the stream as active and stop any pending thinking statuses."""
            if not model_started.is_set():
                model_started.set()
                cancel_thinking()

        def note_generation_activity() -> None:
            """Record when output tokens start/continue streaming."""
            nonlocal generation_started_at, generation_last_event_at
            now = perf_counter()
            generation_last_event_at = now
            if generation_started_at is None:
                generation_started_at = now

        async def _emit_annotation_citations(raw_annotations: Any) -> None:
            """Emit citations from a final annotation dump if no stream citations were seen."""
            nonlocal annotation_citations_emitted
            if annotation_citations_emitted:
                return
            if not isinstance(raw_annotations, list) or not raw_annotations:
                return
            emitted_any = False
            for raw_ann in raw_annotations:
                if not isinstance(raw_ann, dict):
                    continue
                if raw_ann.get("type") != "url_citation":
                    continue
                payload = raw_ann.get("url_citation")
                if isinstance(payload, dict):
                    url = payload.get("url")
                    title = payload.get("title") or url
                else:
                    url = raw_ann.get("url")
                    title = raw_ann.get("title") or url
                if not isinstance(url, str) or not url.strip():
                    continue
                url = url.strip()
                if url.endswith("?utm_source=openai"):
                    url = url[: -len("?utm_source=openai")]
                if isinstance(title, str):
                    title = title.strip() or url
                else:
                    title = url
                host = url.split("//", 1)[-1].split("/", 1)[0].lower().lstrip("www.")
                citation = {
                    "source": {"name": host or "source", "url": url},
                    "document": [title],
                    "metadata": [{
                        "source": url,
                        "date_accessed": datetime.date.today().isoformat(),
                    }],
                }
                try:
                    await self._pipe._event_emitter_handler._emit_citation(event_emitter, citation)
                except Exception as exc:
                    self.logger.debug("Failed to emit annotation citation (final): %s", exc)
                emitted_citations.append(citation)
                emitted_any = True
            if emitted_any:
                annotation_citations_emitted = True

        request_started_at = perf_counter()

        # Send OpenAI Responses API request, parse and emit response
        error_occurred = False
        was_cancelled = False
        loop_limit_reached = False
        try:
            for loop_index in range(valves.MAX_FUNCTION_CALL_LOOPS):
                final_response: dict[str, Any] | None = None
                _sanitize_request_input(self._pipe, body)
                api_model_override = getattr(body, "api_model", None)
                model_for_cache = api_model_override if isinstance(api_model_override, str) else body.model
                items = getattr(body, "input", None)
                if isinstance(items, list):
                    _maybe_apply_anthropic_prompt_caching(
                        items,
                        model_id=model_for_cache,
                        valves=valves,
                    )
                request_payload = body.model_dump(exclude_none=True)
                if api_model_override:
                    request_payload["model"] = api_model_override
                    request_payload.pop("api_model", None)
                _apply_identifier_valves_to_payload(
                    request_payload,
                    valves=valves,
                    owui_metadata=metadata,
                    owui_user_id=user_id,
                    logger=self.logger,
                )
                _apply_model_fallback_to_payload(request_payload, logger=self.logger)
                _apply_disable_native_websearch_to_payload(request_payload, logger=self.logger)
                _strip_disable_model_settings_params(request_payload)

                api_key_value = EncryptedStr.decrypt(valves.API_KEY)
                is_streaming = bool(request_payload.get("stream"))
                if is_streaming:
                    event_iter = self._pipe.send_openrouter_streaming_request(
                        session,
                        request_payload,
                        api_key=api_key_value,
                        base_url=valves.BASE_URL,
                        valves=valves,
                        endpoint_override=endpoint_override,
                        workers=valves.SSE_WORKERS_PER_REQUEST,
                        breaker_key=user_id or None,
                        delta_char_limit=0,
                        idle_flush_ms=0,
                        chunk_queue_maxsize=valves.STREAMING_CHUNK_QUEUE_MAXSIZE,
                        event_queue_maxsize=valves.STREAMING_EVENT_QUEUE_MAXSIZE,
                        event_queue_warn_size=valves.STREAMING_EVENT_QUEUE_WARN_SIZE,
                    )
                else:
                    event_iter = self._pipe.send_openrouter_nonstreaming_request_as_events(
                        session,
                        request_payload,
                        api_key=api_key_value,
                        base_url=valves.BASE_URL,
                        valves=valves,
                        endpoint_override=endpoint_override,
                        breaker_key=user_id or None,
                    )
                timing_mark("event_iteration_start")
                first_event_logged = False
                async for event in event_iter:
                    if stream_started_at is None:
                        stream_started_at = perf_counter()
                    if not first_event_logged:
                        first_event_logged = True
                        timing_mark("first_event_received")
                    etype = event.get("type")
                    # Note: Don't call note_model_activity() here for ALL events.
                    # We only want to cancel thinking tasks when actual output or action starts,
                    # not during reasoning phases. Moved to specific event handlers below.

                    is_delta_event = bool(etype and etype.endswith(".delta"))
                    if not is_delta_event and self.logger.isEnabledFor(logging.DEBUG):
                        redacted_event = _redact_payload_blobs(event)
                        self.logger.debug(
                            "OpenRouter payload: %s",
                            json.dumps(redacted_event, indent=2, ensure_ascii=False),
                        )

                    if etype:
                        # Track the most-recent reasoning item id so unkeyed reasoning deltas can be associated.
                        if etype == "response.output_item.added":
                            item_raw = event.get("item")
                            item = item_raw if isinstance(item_raw, dict) else {}
                            if item.get("type") == "reasoning":
                                iid = item.get("id")
                                if isinstance(iid, str) and iid:
                                    active_reasoning_item_id = iid

                        is_reasoning_event = (
                            etype.startswith("response.reasoning")
                            and etype != "response.reasoning_summary_text.done"
                        )
                        part = event.get("part") if isinstance(event, dict) else None
                        reasoning_part_types = {"reasoning_text", "reasoning_summary_text", "summary_text"}
                        is_reasoning_part_event = (
                            etype.startswith("response.content_part")
                            and isinstance(part, dict)
                            and part.get("type") in reasoning_part_types
                        )
                        if is_reasoning_event or is_reasoning_part_event:
                            reasoning_stream_active = True
                            # Stop "Thinking..." placeholders when reasoning starts
                            note_model_activity()

                            key = _reasoning_stream_key(event, etype)
                            is_incremental = etype.endswith(".delta") or etype.endswith(".added")
                            is_final = etype.endswith(".done") or etype.endswith(".completed")

                            delta_text = _extract_reasoning_text(event)
                            normalized_delta = _normalize_surrogate_chunk(delta_text, "reasoning") if delta_text else ""
                            if normalized_delta and is_incremental:
                                reasoning_stream_has_incremental.add(key)

                            append = ""
                            if normalized_delta:
                                append = _append_reasoning_text(
                                    key,
                                    normalized_delta,
                                    allow_misaligned=is_incremental,
                                )
                            if append:
                                note_generation_activity()

                                if event_emitter and thinking_box_enabled:
                                    await event_emitter(
                                        {
                                            "type": "reasoning:delta",
                                            "data": {
                                                "content": reasoning_buffer,
                                                "delta": append,
                                                "event": etype,
                                            },
                                        }
                                    )

                                await _maybe_emit_reasoning_status(append)
                            if is_final:
                                if event_emitter:
                                    await _maybe_emit_reasoning_status("", force=True)

                                    if (
                                        thinking_box_enabled
                                        and reasoning_stream_buffers.get(key)
                                        and key not in reasoning_stream_completed
                                    ):
                                        await event_emitter(
                                            {
                                                "type": "reasoning:completed",
                                                "data": {"content": reasoning_buffer},
                                            }
                                        )
                                        reasoning_stream_completed.add(key)
                                        reasoning_completed_emitted = True
                            continue

                    # --- Emit partial delta assistant message
                    if etype == "response.output_text.delta":
                        note_model_activity()  # Cancel thinking tasks when actual output starts
                        delta = event.get("delta") or ""
                        normalized_delta = _normalize_surrogate_chunk(delta, "assistant") if delta else ""
                        if normalized_delta:
                            note_generation_activity()
                            if (
                                not provider_status_seen
                                and not responding_status_sent
                                and not reasoning_stream_active
                                and event_emitter
                            ):
                                provider_status_seen = True
                                responding_status_sent = True
                                await event_emitter(
                                    {
                                        "type": "status",
                                        "data": {"description": "Responding to the user…"},
                                    }
                                )
                            assistant_message += normalized_delta
                            await event_emitter(
                                {
                                    "type": "chat:message:delta",
                                    "data": {
                                        "content": normalized_delta,
                                    },
                                }
                            )
                        continue

                    if etype == "response.content_part.done":
                        part = event.get("part") if isinstance(event, dict) else None
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            await _emit_annotation_citations(part.get("annotations"))
                        continue

                    if owui_tool_passthrough and event_emitter and etype in {
                        "response.function_call_arguments.delta",
                        "response.function_call_arguments.done",
                    }:
                        try:
                            raw_call_id = event.get("item_id") or event.get("call_id") or event.get("id")
                            call_id = raw_call_id.strip() if isinstance(raw_call_id, str) else ""
                            if not call_id:
                                continue
                            index = streamed_tool_call_indices.setdefault(
                                call_id, len(streamed_tool_call_indices)
                            )
                            raw_name = event.get("name") or tool_call_names.get(call_id)
                            tool_name = raw_name.strip() if isinstance(raw_name, str) else ""
                            if not tool_name:
                                continue

                            if etype == "response.function_call_arguments.delta":
                                raw_delta = (
                                    event.get("delta")
                                    or event.get("arguments_delta")
                                    or event.get("arguments")
                                )
                                delta_text = raw_delta if isinstance(raw_delta, str) else ""
                                if not delta_text:
                                    continue
                                streamed_tool_call_ids.add(call_id)
                                prev_args = streamed_tool_call_args.get(call_id, "")
                                streamed_tool_call_args[call_id] = f"{prev_args}{delta_text}"
                                include_name = call_id not in streamed_tool_call_name_sent
                                if include_name:
                                    streamed_tool_call_name_sent.add(call_id)
                                await event_emitter(
                                    {
                                        "type": "chat:tool_calls",
                                        "data": {
                                            "tool_calls": [
                                                {
                                                    "index": index,
                                                    "id": call_id,
                                                    "type": "function",
                                                    "function": {
                                                        **({"name": tool_name} if include_name else {}),
                                                        "arguments": delta_text,
                                                    },
                                                }
                                            ]
                                        },
                                    }
                                )
                                continue

                            raw_args = event.get("arguments")
                            args_text = raw_args.strip() if isinstance(raw_args, str) else ""
                            if args_text:
                                prev_args = streamed_tool_call_args.get(call_id, "")
                                suffix = ""
                                if not prev_args:
                                    suffix = args_text
                                elif args_text.startswith(prev_args):
                                    suffix = args_text[len(prev_args) :]
                                if not suffix:
                                    # If we cannot safely compute a suffix, fall back to completion payload later.
                                    continue

                                streamed_tool_call_ids.add(call_id)
                                streamed_tool_call_args[call_id] = f"{prev_args}{suffix}"
                                include_name = call_id not in streamed_tool_call_name_sent
                                if include_name:
                                    streamed_tool_call_name_sent.add(call_id)
                                await event_emitter(
                                    {
                                        "type": "chat:tool_calls",
                                        "data": {
                                            "tool_calls": [
                                                {
                                                    "index": index,
                                                    "id": call_id,
                                                    "type": "function",
                                                    "function": {
                                                        **({"name": tool_name} if include_name else {}),
                                                        "arguments": suffix,
                                                    },
                                                }
                                            ]
                                        },
                                    }
                                )
                        except Exception as exc:
                            self.logger.warning(
                                "Failed to stream tool-call arguments: %s",
                                exc,
                                exc_info=self.logger.isEnabledFor(logging.DEBUG),
                            )
                        continue

                    # --- Emit reasoning summary once done -----------------------
                    if etype == "response.reasoning_summary_text.done":
                        text = (event.get("text") or "").strip()
                        if text:
                            title_match = re.findall(r"\*\*(.+?)\*\*", text)
                            title = title_match[-1].strip() if title_match else "Thinking…"
                            content = re.sub(r"\*\*(.+?)\*\*", "", text).strip()
                            summary = title if not content else f"{title}\n{content}"
                            if event_emitter:
                                note_model_activity()
                                key = _reasoning_stream_key(event, etype)
                                if thinking_box_enabled and not reasoning_stream_buffers.get(key):
                                    normalized_summary = (
                                        _normalize_surrogate_chunk(summary, "reasoning") if summary else ""
                                    )
                                    append = ""
                                    if normalized_summary:
                                        append = _append_reasoning_text(
                                            key,
                                            normalized_summary,
                                            allow_misaligned=False,
                                        )
                                    if append:
                                        note_generation_activity()
                                        reasoning_stream_active = True
                                        await event_emitter(
                                            {
                                                "type": "reasoning:delta",
                                                "data": {
                                                    "content": reasoning_buffer,
                                                    "delta": append,
                                                    "event": etype,
                                                },
                                            }
                                        )
                                    if (
                                        key not in reasoning_stream_completed
                                        and thinking_box_enabled
                                        and reasoning_stream_buffers.get(key)
                                    ):
                                        await event_emitter(
                                            {
                                                "type": "reasoning:completed",
                                                "data": {"content": reasoning_buffer},
                                            }
                                        )
                                        reasoning_stream_completed.add(key)
                                        reasoning_completed_emitted = True
                                if thinking_status_enabled:
                                    cancel_thinking()
                                    await event_emitter(
                                        {
                                            "type": "status",
                                            "data": {"description": summary},
                                        }
                                    )
                        continue

                    # --- Citations from inline annotations (emit metadata only) ---------------
                    if etype == "response.output_text.annotation.added":
                        ann = event.get("annotation") or {}
                        if ann.get("type") == "url_citation":
                            # Basic fields
                            url = (ann.get("url") or "").strip()
                            if url.endswith("?utm_source=openai"):
                                url = url[: -len("?utm_source=openai")]
                            title = (ann.get("title") or url).strip()

                            if url in ordinal_by_url:
                                continue

                            ordinal_by_url[url] = len(ordinal_by_url) + 1

                            # Emit a citation event so Open WebUI can render
                            # references in the footer instead of mutating the
                            # streaming text (which was causing inline [n] markers).
                            host = url.split("//", 1)[-1].split("/", 1)[0].lower().lstrip("www.")
                            citation = {
                                "source": {"name": host or "source", "url": url},
                                "document": [title],
                                "metadata": [{
                                    "source": url,
                                    "date_accessed": datetime.date.today().isoformat(),
                                }],
                            }
                            try:
                                await self._pipe._event_emitter_handler._emit_citation(event_emitter, citation)
                            except Exception as exc:
                                self.logger.debug("Failed to emit annotation citation: %s", exc)
                            emitted_citations.append(citation)
                            annotation_citations_emitted = True

                        continue


                    # --- Emit status updates for in-progress items ----------------------
                    if etype == "response.output_item.added":
                        item_raw = event.get("item")
                        item = item_raw if isinstance(item_raw, dict) else {}
                        item_type = item.get("type", "")
                        item_status = item.get("status", "")

                        if item_type == "reasoning":
                            iid = item.get("id")
                            if isinstance(iid, str) and iid:
                                active_reasoning_item_id = iid
                            continue

                        if item_type == "message" and item_status == "in_progress":
                            provider_status_seen = True
                            if (not responding_status_sent) and (not reasoning_stream_active) and event_emitter:
                                responding_status_sent = True
                                await event_emitter(
                                    {
                                        "type": "status",
                                        "data": {"description": "Responding to the user…"},
                                    }
                                )
                            continue

                        if owui_tool_passthrough and item_type == "function_call":
                            raw_call_id = item.get("call_id") or item.get("id")
                            call_id = raw_call_id.strip() if isinstance(raw_call_id, str) else ""
                            raw_name = item.get("name")
                            tool_name = raw_name.strip() if isinstance(raw_name, str) else ""
                            if call_id:
                                streamed_tool_call_indices.setdefault(
                                    call_id, len(streamed_tool_call_indices)
                                )
                                if tool_name:
                                    tool_call_names[call_id] = tool_name
                            continue

                    # --- Emit detailed tool status upon completion ------------------------
                    if etype == "response.output_item.done":
                        item_raw = event.get("item")
                        item = item_raw if isinstance(item_raw, dict) else {}
                        item_type = item.get("type", "")
                        item_name = item.get("name", "unnamed_tool")

                        if item_type == "message":
                            content = item.get("content")
                            if isinstance(content, list):
                                for content_part in content:
                                    if not isinstance(content_part, dict):
                                        continue
                                    if content_part.get("type") != "output_text":
                                        continue
                                    await _emit_annotation_citations(content_part.get("annotations"))
                                    if annotation_citations_emitted:
                                        break
                            if not annotation_citations_emitted:
                                await _emit_annotation_citations(item.get("annotations"))
                            continue

                        should_persist = False
                        if item_type == "reasoning":
                            should_persist = valves.PERSIST_REASONING_TOKENS in {"next_reply", "conversation"}

                        elif item_type == "web_search_call":
                            # Never persist assistant/user messages or ephemeral search calls
                            should_persist = False

                        else:
                            should_persist = persist_tools_enabled

                        if item_type == "function_call":
                            # Defer persistence until the corresponding tool result is stored
                            should_persist = False

                        if should_persist:
                            normalized_item = _normalize_persisted_item(item)
                            if normalized_item:
                                row = self._pipe._artifact_store._make_db_row(
                                    chat_id, message_id, openwebui_model, normalized_item
                                )
                                if row:
                                    pending_items.append(row)


                        title = f"Running `{item_name}`"
                        content = ""
                        image_markdowns: list[str] = []

                        # Prepare detailed content per item_type
                        if item_type == "function_call":
                            title = f"Running the {item_name} tool…"
                            raw_arguments = item.get("arguments")
                            # OpenRouter `/responses` quirk: tool call items may be marked done with
                            # `arguments: ""` (or arguments only appearing later in `response.completed`).
                            # Avoid logging a misleading invocation like `tool_name()` in that case.
                            if isinstance(raw_arguments, str) and not raw_arguments.strip():
                                title = f"Tool call requested: {item_name} (arguments pending)"
                                content = ""
                            else:
                                parsed_arguments: dict[str, Any] | None = None
                                if isinstance(raw_arguments, dict):
                                    parsed_arguments = raw_arguments
                                elif isinstance(raw_arguments, str):
                                    parsed = _safe_json_loads(raw_arguments)
                                    if isinstance(parsed, dict):
                                        parsed_arguments = parsed

                                if parsed_arguments is not None:
                                    args_formatted = ", ".join(
                                        f"{k}={json.dumps(v, ensure_ascii=False)}"
                                        for k, v in parsed_arguments.items()
                                    )
                                    invocation = (
                                        f"{item_name}({args_formatted})"
                                        if args_formatted
                                        else f"{item_name}()"
                                    )
                                else:
                                    raw_text = ""
                                    if isinstance(raw_arguments, str):
                                        raw_text = raw_arguments.strip()
                                    elif raw_arguments is not None:
                                        with contextlib.suppress(TypeError, ValueError):
                                            raw_text = json.dumps(raw_arguments, ensure_ascii=False)
                                        if not raw_text:
                                            raw_text = str(raw_arguments)

                                    if not raw_text or raw_text == "{}":
                                        invocation = f"{item_name}()"
                                    else:
                                        truncated = (
                                            raw_text
                                            if len(raw_text) <= 1000
                                            else (raw_text[:1000] + "…")
                                        )
                                        invocation = (
                                            "# Unparsed tool arguments "
                                            "(provider sent invalid/non-object JSON)\n"
                                            f"{item_name}(raw_arguments={json.dumps(truncated, ensure_ascii=False)})"
                                        )

                                content = wrap_code_block(invocation, "python")

                        elif item_type == "web_search_call":
                            action = item.get("action", {}) or {}

                            if action.get("type") == "search":
                                query = action.get("query")
                                sources = action.get("sources") or []
                                urls = [s.get("url") for s in sources if s.get("url")]

                                if event_emitter:
                                    if query:
                                        await event_emitter({
                                            "type": "status",
                                            "data": {
                                                "action": "web_search_queries_generated",
                                                "description": "Searching",
                                                "queries": [query],
                                                "done": False,
                                            },
                                        })

                                    if urls:
                                        await event_emitter({
                                            "type": "status",
                                            "data": {
                                                "action": "web_search",
                                                "description": "Reading through {{count}} sites",
                                                "query": query,
                                                "urls": urls,
                                                "done": False,
                                            },
                                        })

                            elif action.get("type") == "open_page":
                                if event_emitter:
                                    raw_title = action.get("title")
                                    raw_host = action.get("host")
                                    raw_url = action.get("url")
                                    title = raw_title.strip() if isinstance(raw_title, str) else ""
                                    host = raw_host.strip() if isinstance(raw_host, str) else ""
                                    url = raw_url.strip() if isinstance(raw_url, str) else ""
                                    description = "Opening page"
                                    if host:
                                        description = f"Opening {host}"
                                    elif title:
                                        description = f"Opening {title}"
                                    elif url:
                                        description = f"Opening {url}"
                                    await event_emitter({
                                        "type": "status",
                                        "data": {
                                            "action": "open_page",
                                            "description": description,
                                            "title": raw_title or (title or None),
                                            "host": raw_host or (host or None),
                                            "url": raw_url or (url or None),
                                            "done": False,
                                        },
                                    })
                                continue
                            elif action.get("type") == "find_in_page":
                                if event_emitter:
                                    raw_query = next(
                                        (
                                            value
                                            for value in (
                                                action.get("needle"),
                                                action.get("query"),
                                                action.get("text"),
                                            )
                                            if isinstance(value, str) and value.strip()
                                        ),
                                        "",
                                    )
                                    query = raw_query.strip() if isinstance(raw_query, str) else ""
                                    raw_page_url = action.get("url")
                                    page_url = raw_page_url.strip() if isinstance(raw_page_url, str) else ""
                                    description = "Searching within page"
                                    if query:
                                        description = f"Searching for {query}"
                                    await event_emitter({
                                        "type": "status",
                                        "data": {
                                            "action": "find_in_page",
                                            "description": description,
                                            "needle": raw_query or (query or None),
                                            "url": raw_page_url or (page_url or None),
                                            "done": False,
                                        },
                                    })
                                continue
                                    
                            continue

                        elif item_type == "file_search_call":
                            title = "Let me skim those files…"
                        elif item_type == "image_generation_call":
                            title = "Let me create that image…"
                            item_id = item.get("id")
                            if item_id and item_id in processed_image_item_ids:
                                self.logger.debug("Skipping duplicate image item '%s'", item_id)
                            else:
                                if item_id:
                                    processed_image_item_ids.add(item_id)
                                try:
                                    image_markdowns = await _render_image_markdown(item)
                                except Exception as exc:
                                    self.logger.error(
                                        "Failed to process generated image for item '%s': %s",
                                        item_id or "<unknown>",
                                        exc,
                                        exc_info=self.logger.isEnabledFor(logging.DEBUG),
                                    )
                                    await self._pipe._event_emitter_handler._emit_status(
                                        event_emitter,
                                        "⚠️ Unable to process generated image output",
                                        done=False,
                                    )
                                    image_markdowns = []
                        elif item_type == "local_shell_call":
                            title = "Let me run that command…"
                        elif item_type == "reasoning":
                            title = None # Don't emit a title for reasoning items
                            key = _reasoning_stream_key(event, etype)
                            snapshot = _extract_reasoning_text_from_item(item)
                            normalized_snapshot = (
                                _normalize_surrogate_chunk(snapshot, "reasoning") if snapshot else ""
                            )
                            append = ""
                            if normalized_snapshot:
                                append = _append_reasoning_text(
                                    key,
                                    normalized_snapshot,
                                    allow_misaligned=False,
                                )
                            if append and event_emitter and thinking_box_enabled:
                                reasoning_stream_active = True
                                note_model_activity()
                                note_generation_activity()
                                await event_emitter(
                                    {
                                        "type": "reasoning:delta",
                                        "data": {
                                            "content": reasoning_buffer,
                                            "delta": append,
                                            "event": etype,
                                        },
                                    }
                                )
                            if (
                                event_emitter
                                and thinking_box_enabled
                                and reasoning_stream_buffers.get(key)
                                and key not in reasoning_stream_completed
                            ):
                                await event_emitter(
                                    {
                                        "type": "reasoning:completed",
                                        "data": {"content": reasoning_buffer},
                                    }
                                )
                                reasoning_stream_completed.add(key)
                                reasoning_completed_emitted = True

                        # Log the status with prepared title and detailed content instead of emitting it
                        if title:
                            desc = title if not content else f"{title}\n{content}"
                            if thinking_tasks:
                                cancel_thinking()
                            self.logger.debug("Tool status update: %s", desc)

                        if image_markdowns:
                            note_model_activity()
                            note_generation_activity()
                            msg_before = len(assistant_message)
                            for snippet in image_markdowns:
                                assistant_message = _append_output_block(assistant_message, snippet)
                            if event_emitter:
                                image_delta = assistant_message[msg_before:]
                                await event_emitter({"type": "chat:message:delta", "data": {"content": image_delta}})

                        continue

                    # --- Capture final response payload for this loop
                    if etype == "response.completed":
                        note_model_activity()  # Ensure thinking tasks are cancelled when response completes
                        final_response = event.get("response", {})
                        response_completed_at = perf_counter()
                        if generation_started_at is not None:
                            generation_last_event_at = response_completed_at
                        break

                if final_response is None:
                    raise ValueError("No final response received from OpenAI Responses API.")

                # Extract usage information from OpenAI response and pass-through to Open WebUI
                raw_usage = final_response.get("usage") or {}
                usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}

                if usage:
                    usage["turn_count"] = 1
                    usage["function_call_count"] = sum(
                        1 for i in final_response["output"] if i["type"] == "function_call"
                    )
                    total_usage = merge_usage_stats(total_usage, usage)
                    await self._pipe._event_emitter_handler._emit_completion(
                        event_emitter,
                        content=assistant_message,
                        usage=total_usage,
                        done=False,
                    )

                metadata_model = None
                if isinstance(metadata, dict):
                    model_block = metadata.get("model")
                    if isinstance(model_block, dict):
                        metadata_model = model_block.get("id")
                snapshot_model_id = self._pipe._qualify_model_for_pipe(
                    pipe_identifier,
                    metadata_model or body.model,
                )

                await maybe_dump_costs_snapshot(
                    self._pipe,
                    valves,
                    user_id=user_id or "",
                    model_id=snapshot_model_id,
                    usage=usage if usage else {},
                    user_obj=user_obj,
                    pipe_id=pipe_identifier,
                )

                # Execute tool calls (if any), persist results (if valve enabled), and append to body.input.
                # Per Responses API spec: reasoning items with encrypted_content must be passed back
                # on tool continuations to preserve the model's chain of thought across tool calls.
                # See: https://cookbook.openai.com/examples/responses_api/reasoning_items
                reasoning_items: list[dict[str, Any]] = []
                call_items: list[dict[str, Any]] = []
                if not owui_tool_passthrough:
                    for item in final_response.get("output", []):
                        item_type = item.get("type")
                        if item_type == "reasoning" and item.get("encrypted_content"):
                            # Preserve reasoning with encrypted_content for tool continuation
                            reasoning_items.append(item)
                        elif item_type == "function_call":
                            normalized_call = _normalize_persisted_item(item)
                            if normalized_call:
                                call_items.append(normalized_call)
                    if call_items:
                        note_model_activity()  # Cancel thinking tasks when function calls begin
                        # Order matters: reasoning first, then function_call (per spec)
                        if reasoning_items:
                            body.input.extend(reasoning_items)
                            self.logger.debug("🧠 Preserving %d reasoning item(s) with encrypted_content for tool continuation", len(reasoning_items))
                        body.input.extend(call_items)
                        _sanitize_request_input(self._pipe, body)

                calls = [i for i in final_response.get("output", []) if i.get("type") == "function_call"]
                self.logger.debug("📞 Found %d function_call items in response", len(calls))
                function_outputs: list[dict[str, Any]] = []
                if calls:
                    if loop_index >= (valves.MAX_FUNCTION_CALL_LOOPS - 1):
                        loop_limit_reached = True

                    if owui_tool_passthrough:
                        tool_calls_payload: list[dict[str, Any]] = []
                        exposed_to_origin: dict[str, str] = {}
                        if isinstance(metadata, dict):
                            raw_map = metadata.get("_pipe_exposed_to_origin")
                            if isinstance(raw_map, dict):
                                exposed_to_origin = {str(k): str(v) for k, v in raw_map.items() if k and v}
                        try:
                            for call in calls:
                                raw_call_id = call.get("call_id") or call.get("id")
                                call_id = raw_call_id.strip() if isinstance(raw_call_id, str) else ""
                                raw_name = call.get("name")
                                exposed_name = raw_name.strip() if isinstance(raw_name, str) else ""
                                tool_name = exposed_to_origin.get(exposed_name, exposed_name)
                                raw_args = call.get("arguments")
                                if isinstance(raw_args, str):
                                    args_text = raw_args.strip() or "{}"
                                else:
                                    args_text = json.dumps(raw_args or {}, ensure_ascii=False)
                                if not call_id or not tool_name:
                                    continue

                                tool_calls_payload.append(
                                    {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {"name": tool_name, "arguments": args_text},
                                    }
                                )

                                if body.stream and event_emitter:
                                    idx = streamed_tool_call_indices.setdefault(
                                        call_id, len(streamed_tool_call_indices)
                                    )
                                    prev_args = streamed_tool_call_args.get(call_id, "")
                                    suffix = ""
                                    if not prev_args:
                                        suffix = args_text
                                    elif args_text.startswith(prev_args):
                                        suffix = args_text[len(prev_args) :]

                                    include_name = call_id not in streamed_tool_call_name_sent
                                    if include_name:
                                        streamed_tool_call_name_sent.add(call_id)

                                    if suffix or include_name:
                                        function_obj: dict[str, Any] = {}
                                        if include_name:
                                            function_obj["name"] = tool_name
                                        if suffix:
                                            function_obj["arguments"] = suffix
                                        streamed_tool_call_ids.add(call_id)
                                        if suffix:
                                            streamed_tool_call_args[call_id] = f"{prev_args}{suffix}"
                                        await event_emitter(
                                            {
                                                "type": "chat:tool_calls",
                                                "data": {
                                                    "tool_calls": [
                                                        {
                                                            "index": idx,
                                                            "id": call_id,
                                                            "type": "function",
                                                            "function": function_obj,
                                                        }
                                                    ]
                                                },
                                            }
                                        )
                        except Exception as exc:
                            self.logger.warning(
                                "Tool pass-through failed while building tool_calls payload: %s",
                                exc,
                                exc_info=self.logger.isEnabledFor(logging.DEBUG),
                            )

                        if not body.stream:
                            try:
                                model_for_response = ""
                                metadata_model = metadata.get("model") if isinstance(metadata, dict) else None
                                if isinstance(metadata_model, dict):
                                    model_for_response = str(metadata_model.get("id") or "")
                                if not model_for_response:
                                    model_for_response = str(body.model or "pipe")
                                response = {
                                    "id": f"{model_for_response}-{uuid.uuid4()}",
                                    "object": "chat.completion",
                                    "created": int(time.time()),
                                    "model": model_for_response,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "message": {
                                                "role": "assistant",
                                                "content": assistant_message or None,
                                                **({"tool_calls": tool_calls_payload} if tool_calls_payload else {}),
                                            },
                                            "finish_reason": "tool_calls",
                                            "logprobs": None,
                                        }
                                    ],
                                    **({"usage": total_usage} if total_usage else {}),
                                }
                                if self.logger.isEnabledFor(logging.DEBUG) and tool_calls_payload:
                                    summaries: list[dict[str, Any]] = []
                                    for call in tool_calls_payload:
                                        if not isinstance(call, dict):
                                            continue
                                        fn_raw = call.get("function")
                                        fn = fn_raw if isinstance(fn_raw, dict) else {}
                                        args = fn.get("arguments")
                                        summaries.append(
                                            {
                                                "id": call.get("id"),
                                                "type": call.get("type"),
                                                "name": fn.get("name"),
                                                "args_len": len(args) if isinstance(args, str) else None,
                                                "args_empty": (isinstance(args, str) and not args.strip()),
                                            }
                                        )
                                    self.logger.debug(
                                        "Returning non-streaming tool_calls response (request_id=%s): %s",
                                        (SessionLogger.request_id.get() or ""),
                                        json.dumps(summaries, ensure_ascii=False),
                                    )
                                return response
                            except Exception as exc:
                                self.logger.warning(
                                    "Failed to build non-streaming tool_calls response: %s",
                                    exc,
                                    exc_info=self.logger.isEnabledFor(logging.DEBUG),
                                )
                                return assistant_message

                        break

                    # Build and emit in-progress (spinning) tool cards
                    # In-progress cards are ephemeral: emit via chat:completion so they do NOT
                    # enter content_blocks or persistence. They will be replaced once completed.
                    in_progress_cards_html = ""
                    show_tool_cards = getattr(valves, "SHOW_TOOL_CARDS", False)
                    if show_tool_cards and event_emitter and body.stream and calls:
                        try:
                            for call in calls:
                                call_id = call.get("call_id") or call.get("id") or ""
                                tool_name = (call.get("name") or "").strip()
                                raw_args = call.get("arguments") or "{}"
                                args_json = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                                in_progress_cards_html += (
                                    f'<details type="tool_calls" done="false" id="{html.escape(call_id)}" '
                                    f'name="{html.escape(tool_name)}" arguments="{html.escape(args_json)}">\n'
                                    f'<summary>Executing...</summary>\n</details>\n'
                                )
                            if in_progress_cards_html:
                                # Emit via chat:completion (immediate display, NOT tracked)
                                # Do NOT add to assistant_message - spinners are ephemeral
                                await event_emitter(
                                    {
                                        "type": "chat:completion",
                                        "data": {"content": assistant_message + in_progress_cards_html},
                                    }
                                )
                        except Exception as exc:
                            self.logger.debug("Failed to emit in-progress tool cards: %s", exc)

                    function_outputs = await self._pipe._ensure_tool_executor()._execute_function_calls(calls, tool_registry)

                    # Emit completed tool cards
                    # 1) chat:message:delta to persist completed cards
                    # 2) chat:completion to immediately replace spinners in the UI
                    if show_tool_cards and event_emitter and body.stream and calls and function_outputs:
                        try:
                            completed_cards_html = ""
                            for call, output in zip(calls, function_outputs):
                                call_id = call.get("call_id") or call.get("id") or ""
                                tool_name = (call.get("name") or "").strip()
                                raw_args = call.get("arguments") or "{}"
                                args_json = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
                                result_str = output.get("output") or ""
                                # Extract files/embeds from tool output (populated by process_tool_result)
                                tool_files = output.get("files") or []
                                tool_embeds = output.get("embeds") or []
                                files_attr = html.escape(json.dumps(tool_files if tool_files else []))
                                embeds_attr = html.escape(json.dumps(tool_embeds if tool_embeds else []))
                                completed_cards_html += (
                                    f'<details type="tool_calls" done="true" id="{html.escape(call_id)}" '
                                    f'name="{html.escape(tool_name)}" arguments="{html.escape(args_json)}" '
                                    f'result="{html.escape(json.dumps(result_str, ensure_ascii=False))}" '
                                    f'files="{files_attr}" embeds="{embeds_attr}">\n'
                                    f'<summary>Tool Executed</summary>\n</details>\n'
                                )
                            if completed_cards_html:
                                # Persist completed cards in the streaming content
                                assistant_message = assistant_message + completed_cards_html
                                await event_emitter(
                                    {
                                        "type": "chat:message:delta",
                                        "data": {"content": completed_cards_html},
                                    }
                                )
                                # Immediately replace spinners in the UI
                                await event_emitter(
                                    {
                                        "type": "chat:completion",
                                        "data": {"content": assistant_message},
                                    }
                                )
                        except Exception as exc:
                            self.logger.debug("Failed to emit completed tool cards: %s", exc)

                    # Extract citations from tool results (OpenWebUI parity)
                    # Collection happens regardless of event_emitter (needed for source context)
                    collected_sources: list[dict[str, Any]] = []
                    if get_citation_source_from_tool_result is not None:
                        for call, output in zip(calls, function_outputs):
                            tool_name = (call.get("name") or "").strip()
                            if output.get("status") != "completed":
                                continue
                            try:
                                tool_params = _safe_json_loads(call.get("arguments") or "{}")
                                tool_result = output.get("output") or ""
                                call_id = call.get("call_id") or call.get("id") or ""
                                citations = get_citation_source_from_tool_result(
                                    tool_name=tool_name,
                                    tool_params=tool_params if isinstance(tool_params, dict) else {},
                                    tool_result=tool_result,
                                    tool_id=call_id,
                                )
                                for source in citations:
                                    collected_sources.append(source)
                                    # Emit citation only if event_emitter available
                                    if event_emitter:
                                        await self._pipe._event_emitter_handler._emit_citation(event_emitter, source)
                                        self.logger.debug(
                                            "Emitted citation from tool=%s: %s",
                                            tool_name,
                                            source.get("source", {}).get("name", "unknown"),
                                        )
                            except Exception as exc:
                                self.logger.warning(
                                    "Failed to extract citations from tool=%s: %s",
                                    tool_name,
                                    exc,
                                    exc_info=self.logger.isEnabledFor(logging.DEBUG),
                                )

                    # RAG-style source context injection for non-native function calling.
                    # When function_calling is "native", tool results go via function_call_output
                    # and RAG injection would duplicate tokens. Only inject when NOT native.
                    is_native_fc = metadata.get("params", {}).get("function_calling") == "native"
                    self.logger.debug(
                        "Source context check: collected_sources=%d, native_fc=%s",
                        len(collected_sources),
                        is_native_fc,
                    )
                    if collected_sources and not is_native_fc:
                        try:
                            # Extract the last user message for context
                            user_message = ""
                            for item in reversed(body.input):
                                if isinstance(item, dict):
                                    if item.get("role") == "user":
                                        content = item.get("content")
                                        if isinstance(content, str):
                                            user_message = content
                                            break
                                        elif isinstance(content, list):
                                            for block in content:
                                                # Handle both Chat Completions (type=text) and Responses API (type=input_text)
                                                if isinstance(block, dict) and block.get("type") in ("text", "input_text"):
                                                    user_message = block.get("text") or ""
                                                    break
                                            if user_message:
                                                break
                            self.logger.debug(
                                "User message extraction: found=%s, input_items=%d",
                                bool(user_message),
                                len(body.input) if hasattr(body.input, '__len__') else -1,
                            )
                            if user_message:
                                # Use adapter function that transforms to/from Chat Completions
                                # This delegates to OWUI's apply_source_context_to_messages
                                input_before = len(body.input)
                                body.input = _apply_source_context_responses_api(
                                    list(body.input),
                                    collected_sources,
                                    user_message,
                                    request_context=request_context,
                                )
                                # Verify source context was injected
                                has_source_tags = any(
                                    '<source' in str(item.get('content', ''))
                                    for item in body.input if isinstance(item, dict)
                                )
                                self.logger.debug(
                                    "Applied source context: input=%d->%d items, sources=%d, has_source_tags=%s",
                                    input_before,
                                    len(body.input),
                                    len(collected_sources),
                                    has_source_tags,
                                )
                        except Exception as exc:
                            self.logger.debug(
                                "Failed to apply source context: %s",
                                exc,
                                exc_info=self.logger.isEnabledFor(logging.DEBUG),
                            )

                    if persist_tools_enabled:
                        self.logger.debug("💾 Persisting %d tool results", len(function_outputs))
                        persist_payloads: list[dict] = []
                        for idx, output in enumerate(function_outputs):
                            if idx < len(call_items):
                                persist_payloads.append(call_items[idx])
                            persist_payloads.append(output)

                        if persist_payloads:
                            self.logger.debug("🔍 Processing %d persist_payloads", len(persist_payloads))
                            for idx, payload in enumerate(persist_payloads, start=1):
                                payload_type = payload.get("type")
                                self.logger.debug(
                                    "🔍 [%d/%d] Payload type=%s",
                                    idx,
                                    len(persist_payloads),
                                    payload_type,
                                )
                                normalized_payload = _normalize_persisted_item(payload)
                                if not normalized_payload:
                                    self.logger.warning(
                                        "❌ [%d/%d] Normalization returned None for type=%s",
                                        idx,
                                        len(persist_payloads),
                                        payload_type,
                                    )
                                    continue
                                self.logger.debug(
                                    "✅ [%d/%d] Normalized successfully (type=%s)",
                                    idx,
                                    len(persist_payloads),
                                    payload_type,
                                )
                                row = self._pipe._artifact_store._make_db_row(
                                    chat_id, message_id, openwebui_model, normalized_payload
                                )
                                if not row:
                                    self.logger.warning(
                                        "❌ [%d/%d] _make_db_row returned None (chat_id=%s, message_id=%s, type=%s)",
                                        idx,
                                        len(persist_payloads),
                                        chat_id,
                                        message_id,
                                        payload_type,
                                    )
                                    continue
                                self.logger.debug(
                                    "✅ [%d/%d] Row created; enqueueing for persistence",
                                    idx,
                                    len(persist_payloads),
                                )
                                pending_items.append(row)
                            self.logger.debug(
                                "📦 Total pending_items after loop: %d", len(pending_items)
                            )
                            if thinking_tasks:
                                cancel_thinking()
                            await _flush_pending("function_outputs")

                    for output in function_outputs:
                        result_text = wrap_code_block(output.get("output", ""))
                        if thinking_tasks:
                            cancel_thinking()
                        self.logger.debug("Received tool result\n%s", result_text)
                    body.input.extend(function_outputs)
                    _sanitize_request_input(self._pipe, body)
                else:
                    break

            if loop_limit_reached:
                limit_value = valves.MAX_FUNCTION_CALL_LOOPS
                await self._pipe._event_emitter_handler._emit_notification(
                    event_emitter,
                    f"Tool step limit reached (MAX_FUNCTION_CALL_LOOPS={limit_value}). "
                    "Increase the limit or simplify the request to continue.",
                    level="warning",
                )
                try:
                    notice = _render_error_template(
                        valves.MAX_FUNCTION_CALL_LOOPS_REACHED_TEMPLATE,
                        {"max_function_call_loops": str(limit_value)},
                    )
                except Exception:
                    notice = _render_error_template(
                        DEFAULT_MAX_FUNCTION_CALL_LOOPS_REACHED_TEMPLATE,
                        {"max_function_call_loops": str(limit_value)},
                    )
                if notice:
                    delta = f"\n\n{notice}" if assistant_message else notice
                    assistant_message = f"{assistant_message}{delta}" if assistant_message else notice
                    if event_emitter:
                        await event_emitter(
                            {
                                "type": "chat:message:delta",
                                "data": {"content": delta},
                            }
                        )

        # Catch any exceptions during the streaming loop and emit an error
        except asyncio.CancelledError:
            was_cancelled = True
            session_log_reason = "cancelled"
            raise
        except OpenRouterAPIError as exc:
            error_occurred = True
            session_log_reason = str(exc)
            assistant_message = ""
            cancel_thinking()
            await self._pipe._ensure_error_formatter()._report_openrouter_error(
                exc,
                event_emitter=event_emitter,
                normalized_model_id=body.model,
                api_model_id=getattr(body, "api_model", None),
                usage=total_usage,
            )
        except Exception as e:  # pragma: no cover - network errors
            error_occurred = True
            session_log_reason = str(e)
            await self._pipe._ensure_error_formatter()._emit_error(event_emitter, f"Error: {str(e)}", show_error_message=True, show_error_log_citation=True, done=True)

        finally:
            cancel_thinking()
            for t in thinking_tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
            if (
                reasoning_buffer
                and reasoning_stream_active
                and not reasoning_completed_emitted
                and event_emitter
                and thinking_box_enabled
            ):
                await event_emitter(
                    {
                        "type": "reasoning:completed",
                        "data": {"content": reasoning_buffer},
                    }
                )
                reasoning_completed_emitted = True
            surrogate_carry["assistant"] = ""
            surrogate_carry["reasoning"] = ""
            if (not error_occurred) and (not was_cancelled):
                await self._cleanup_replayed_reasoning(body, valves)
            if (not error_occurred) and (not was_cancelled) and event_emitter:
                effective_start = stream_started_at or request_started_at
                elapsed = max(0.0, perf_counter() - effective_start)
                stream_window = None
                last_generation_stamp = generation_last_event_at or response_completed_at
                if last_generation_stamp is not None:
                    duration = max(0.0, last_generation_stamp - effective_start)
                    if duration > 0:
                        stream_window = duration
                description = self._pipe._ensure_error_formatter()._format_final_status_description(
                    elapsed=elapsed,
                    total_usage=total_usage,
                    valves=valves,
                    stream_duration=stream_window,
                )
                await event_emitter(
                    {
                        "type": "status",
                        "data": {
                            "description": description,
                            "done": True,
                        },
                    }
                )

            request_id = SessionLogger.request_id.get() or ""
            if request_id:
                with SessionLogger._state_lock:
                    log_events = list(SessionLogger.logs.get(request_id, []))
                if log_events and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("Collected %d session log entries for request %s.", len(log_events), request_id)
                resolved_user_id = str(user_id or metadata.get("user_id") or "")
                resolved_session_id = str(metadata.get("session_id") or "")
                resolved_chat_id = str(metadata.get("chat_id") or "")
                resolved_message_id = str(metadata.get("message_id") or "")
                has_function_calls = False
                if owui_tool_passthrough and final_response and isinstance(final_response.get("output"), list):
                    with contextlib.suppress(Exception):
                        has_function_calls = any(
                            isinstance(item, dict) and item.get("type") == "function_call"
                            for item in final_response.get("output", [])
                        )
                terminal = bool(
                    was_cancelled
                    or error_occurred
                    or (not owui_tool_passthrough)
                    or (owui_tool_passthrough and (not has_function_calls))
                )
                segment_status = "complete"
                if was_cancelled:
                    segment_status = "cancelled"
                elif error_occurred:
                    segment_status = "error"
                elif owui_tool_passthrough and has_function_calls:
                    segment_status = "needs_tool"
                try:
                    # Persist synchronously (shielded) so session logs aren't dropped on
                    # cancellation, generator close, or event loop timing quirks.
                    await asyncio.shield(
                        self._pipe._session_log_manager.persist_segment_to_db(
                            valves,
                            user_id=resolved_user_id,
                            session_id=resolved_session_id,
                            chat_id=resolved_chat_id,
                            message_id=resolved_message_id,
                            request_id=request_id,
                            log_events=log_events,
                            terminal=terminal,
                            status=segment_status,
                            reason=session_log_reason,
                            pipe_identifier=pipe_identifier,
                        )
                    )
                except Exception:
                    self.logger.debug(
                        "Failed to persist session log segment (chat_id=%s message_id=%s request_id=%s terminal=%s)",
                        resolved_chat_id,
                        resolved_message_id,
                        request_id,
                        terminal,
                        exc_info=True,
                    )

            if (not error_occurred) and (not was_cancelled):
                # Emit completion (middleware.py also does this so this just covers if there is a downstream error)
                # Emit the final completion frame with the last assistant snapshot so
                # late-arriving emitters (middleware, other workers) cannot wipe the UI.
                await self._pipe._event_emitter_handler._emit_completion(
                    event_emitter,
                    content=assistant_message,
                    usage=total_usage,
                    done=True,
                )

            # Clear logs
            if request_id:
                with SessionLogger._state_lock:
                    SessionLogger.logs.pop(request_id, None)
            SessionLogger.cleanup()

            chat_id = metadata.get("chat_id")
            message_id = metadata.get("message_id")
            if (not was_cancelled) and chat_id and message_id and emitted_citations and Chats is not None:
                try:
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        chat_id, message_id, {"sources": emitted_citations}
                    )
                except Exception as exc:
                    self.logger.warning("Failed to persist citations for chat_id=%s message_id=%s: %s", chat_id, message_id, exc)
                    await self._pipe._event_emitter_handler._emit_notification(
                        event_emitter,
                        "Unable to save citations for this response. Output was delivered successfully.",
                        level="warning",
                    )
            assistant_annotations: list[Any] = []
            if final_response and isinstance(final_response.get("output"), list):
                for item in final_response.get("output", []):
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "message":
                        continue
                    if item.get("role") != "assistant":
                        continue
                    raw_annotations = item.get("annotations")
                    if isinstance(raw_annotations, list) and raw_annotations:
                        assistant_annotations = list(raw_annotations)
                    break

            if (not was_cancelled) and chat_id and message_id and assistant_annotations and Chats is not None:
                try:
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        chat_id, message_id, {"annotations": assistant_annotations}
                    )
                except Exception as exc:
                    self.logger.warning("Failed to persist annotations for chat_id=%s message_id=%s: %s", chat_id, message_id, exc)
                    await self._pipe._event_emitter_handler._emit_notification(
                        event_emitter,
                        "Unable to save file annotations for this response. Output was delivered successfully.",
                        level="warning",
                    )

            assistant_reasoning_details: list[Any] = []
            if final_response and isinstance(final_response.get("output"), list):
                for item in final_response.get("output", []):
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "message":
                        continue
                    if item.get("role") != "assistant":
                        continue
                    raw_reasoning_details = item.get("reasoning_details")
                    if isinstance(raw_reasoning_details, list) and raw_reasoning_details:
                        assistant_reasoning_details = list(raw_reasoning_details)
                    break

            if (not was_cancelled) and chat_id and message_id and assistant_reasoning_details and Chats is not None:
                try:
                    Chats.upsert_message_to_chat_by_id_and_message_id(
                        chat_id, message_id, {"reasoning_details": assistant_reasoning_details}
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Failed to persist reasoning_details for chat_id=%s message_id=%s: %s",
                        chat_id,
                        message_id,
                        exc,
                    )
                    await self._pipe._event_emitter_handler._emit_notification(
                        event_emitter,
                        "Unable to save reasoning details for this response. Output was delivered successfully.",
                        level="warning",
                    )

            if not was_cancelled:
                await _flush_pending("finalize")
                if pending_ulids:
                    marker_lines = [_serialize_marker(ulid) for ulid in pending_ulids]
                    ulid_block = "\n".join(marker_lines)
                    if assistant_message:
                        if not assistant_message.endswith("\n"):
                            assistant_message += "\n"
                        if not assistant_message.endswith("\n\n"):
                            assistant_message += "\n"
                    assistant_message += ulid_block
                    if not assistant_message.endswith("\n"):
                        assistant_message += "\n"

        # Return the final output to ensure persistence (unless cancelled, in which case the exception propagates).
        return assistant_message



    @timed
    async def _run_nonstreaming_loop(
        self,
        body: ResponsesBody,
        valves: Pipe.Valves,
        event_emitter: EventEmitter | None,
        metadata: Dict[str, Any] = {},
        tools: dict[str, Dict[str, Any]] | list[dict[str, Any]] | None = None,
        session: aiohttp.ClientSession | None = None,
        user_id: str = "",
        *,
        endpoint_override: Literal["responses", "chat_completions"] | None = None,
        request_context: Optional[Request] = None,
        user_obj: Optional[Any] = None,
        pipe_identifier: Optional[str] = None,
    ) -> str | dict[str, Any]:
        """Reuse the streaming loop logic, but honour `stream=False` at the HTTP layer.

        This delegates to `_run_streaming_loop` with a wrapped emitter so incremental
        `chat:message` frames are suppressed while still running all value-add logic
        (tools, citations, usage snapshots, persistence).
        """

        # Pass through status / citations / usage, but do NOT emit partial text
        wrapped_emitter = _wrap_event_emitter(
            event_emitter,
            suppress_chat_messages=True,
            suppress_completion=False,
        )

        if session is None:
            raise RuntimeError("HTTP session is required for non-streaming")

        return await self._run_streaming_loop(
            body,
            valves,
            wrapped_emitter,
            metadata,
            tools or {},
            session=session,
            user_id=user_id,
            endpoint_override=endpoint_override,
            request_context=request_context,
            user_obj=user_obj,
            pipe_identifier=pipe_identifier,
        )



    @timed
    async def _cleanup_replayed_reasoning(self, body: ResponsesBody, valves: Pipe.Valves) -> None:
        """Delete once-used reasoning artifacts when retention is limited to the next reply."""
        if valves.PERSIST_REASONING_TOKENS != "next_reply":
            return
        refs = getattr(body, "_replayed_reasoning_refs", None)
        if not refs:
            return
        setattr(body, "_replayed_reasoning_refs", [])
        await self._pipe._artifact_store._delete_artifacts(refs)




    def _select_llm_endpoint(
        self,
        model_id: str,
        *,
        valves: "Pipe.Valves",
    ) -> Literal["responses", "chat_completions"]:
        """Choose which OpenRouter endpoint to use for a given model id."""
        base_id = ModelFamily.base_model(model_id or "") or (model_id or "")
        force_chat = _parse_model_patterns(getattr(valves, "FORCE_CHAT_COMPLETIONS_MODELS", ""))
        force_responses = _parse_model_patterns(getattr(valves, "FORCE_RESPONSES_MODELS", ""))
        if _matches_any_model_pattern(base_id, force_responses):
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "LLM endpoint selection: model_id=%s base_id=%s -> responses (FORCE_RESPONSES_MODELS=%s)",
                    model_id,
                    base_id,
                    force_responses,
                )
            return "responses"
        if _matches_any_model_pattern(base_id, force_chat):
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "LLM endpoint selection: model_id=%s base_id=%s -> chat_completions (FORCE_CHAT_COMPLETIONS_MODELS=%s)",
                    model_id,
                    base_id,
                    force_chat,
                )
            return "chat_completions"
        default_endpoint = getattr(valves, "DEFAULT_LLM_ENDPOINT", "responses")
        selected = "chat_completions" if default_endpoint == "chat_completions" else "responses"
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "LLM endpoint selection: model_id=%s base_id=%s default=%s -> %s",
                model_id,
                base_id,
                default_endpoint,
                selected,
            )
        return selected


    def _select_llm_endpoint_with_forced(
        self,
        model_id: str,
        *,
        valves: "Pipe.Valves",
    ) -> tuple[Literal["responses", "chat_completions"], bool]:
        """Return (endpoint, forced) where forced=True when a FORCE_* valve matched the model id."""
        base_id = ModelFamily.base_model(model_id or "") or (model_id or "")
        force_chat = _parse_model_patterns(getattr(valves, "FORCE_CHAT_COMPLETIONS_MODELS", ""))
        force_responses = _parse_model_patterns(getattr(valves, "FORCE_RESPONSES_MODELS", ""))
        if _matches_any_model_pattern(base_id, force_responses):
            return "responses", True
        if _matches_any_model_pattern(base_id, force_chat):
            return "chat_completions", True
        return self._select_llm_endpoint(model_id, valves=valves), False



    @staticmethod
    def _looks_like_responses_unsupported(exc: BaseException) -> bool:
        """Heuristic: detect 'model doesn't support /responses' so we can retry via /chat/completions."""
        if isinstance(exc, OpenRouterAPIError):
            code = exc.openrouter_code
            code_lower = code.strip().lower() if isinstance(code, str) else ""
            if code_lower in {
                "unsupported_endpoint",
                "unsupported_feature",
                "endpoint_not_supported",
                "responses_not_supported",
            }:
                return True

            message_parts = [
                exc.openrouter_message or "",
                exc.upstream_message or "",
                exc.raw_body or "",
                str(exc),
            ]
            haystack = " ".join(part for part in message_parts if part).lower()
            if "response" not in haystack and "responses" not in haystack:
                return False
            if any(token in haystack for token in ("not supported", "unsupported", "does not support")):
                return True
            if any(token in haystack for token in ("chat/completions", "chat completions")):
                return True
            if any(token in haystack for token in ("openai-responses-v1", "xai-responses-v1")):
                return True
            return False

        haystack_parts: list[str] = [str(exc)]
        for attr in ("openrouter_message", "upstream_message", "raw_body"):
            value = getattr(exc, attr, None)
            if isinstance(value, str) and value:
                haystack_parts.append(value)
        haystack = " ".join(haystack_parts).lower()
        if "response" not in haystack and "responses" not in haystack:
            return False
        if any(token in haystack for token in ("not supported", "unsupported", "does not support")):
            return True
        if any(token in haystack for token in ("chat/completions", "chat completions")):
            return True
        if any(token in haystack for token in ("openai-responses-v1", "xai-responses-v1")):
            return True
        return False



def _wrap_event_emitter(
    emitter: EventEmitter | None,
    *,
    suppress_chat_messages: bool = False,
    suppress_completion: bool = False,
):
    """
    Wrap the given event emitter and optionally suppress specific event types.

    Use-case: reuse the streaming loop for non-stream requests by swallowing
    incremental 'chat:message' frames while allowing status/citation/usage
    events through.
    """
    if emitter is None:
        async def _noop(_event: Dict[str, Any]) -> None:
            """Swallow events when no emitter is provided."""
            return

        return _noop

    async def _wrapped(event: Dict[str, Any]) -> None:
        """Proxy emitter that suppresses selected event types."""
        etype = (event or {}).get("type")
        # Suppress BOTH chat:message AND chat:message:delta for non-streaming
        # Tool cards use chat:message:delta which would leak through otherwise
        if suppress_chat_messages and etype in ("chat:message", "chat:message:delta"):
            return  # swallow incremental deltas
        if suppress_completion and etype == "chat:completion":
            return  # optionally swallow completion frames
        await emitter(event)

    return _wrapped
