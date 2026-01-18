"""Reasoning status tracking and image materialization.

This module handles reasoning-specific streaming features:
- Reasoning status accumulation and formatting
- Image URL detection and persistence
- Citation extraction from reasoning output
- Usage metrics tracking for reasoning models
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import datetime
import logging
import uuid
from time import perf_counter
from typing import TYPE_CHECKING, Any, Optional, Callable, Awaitable
from ..core.timing_logger import timed

if TYPE_CHECKING:
    from ..pipe import Pipe

LOGGER = logging.getLogger(__name__)

# Reasoning status emission constants
_REASONING_STATUS_PUNCTUATION = (".", "!", "?", ":", ";", ",")
_REASONING_STATUS_MAX_CHARS = 200
_REASONING_STATUS_MIN_CHARS = 40
_REASONING_STATUS_IDLE_SECONDS = 0.5


class ReasoningTracker:
    """Tracks reasoning model outputs including status, images, citations.

    This class manages all reasoning-related state during streaming:
    - Accumulates reasoning text from multiple event types
    - Emits throttled status updates for user feedback
    - Materializes image URLs to persistent storage
    - Extracts and formats citation metadata
    - Tracks reasoning completion state

    The tracker is designed to work with OpenAI Responses API and similar
    streaming protocols that emit reasoning events.
    """

    @timed
    def __init__(
        self,
        pipe: "Pipe",
        *,
        valves: Any,
        event_emitter: Optional[Callable[[dict[str, Any]], Awaitable[None]]],
        logger: Optional[logging.Logger] = None,
        request_context: Optional[Any] = None,
        user_obj: Optional[Any] = None,
        chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
        user_id: str = "",
    ):
        """Initialize the ReasoningTracker.

        Args:
            pipe: Reference to parent Pipe instance
            valves: Valve configuration object
            event_emitter: Optional callback to emit events to Open WebUI
            logger: Logger instance for diagnostic output
            request_context: Optional request context for storage operations
            user_obj: Optional user object for storage operations
            chat_id: Optional chat ID for associating uploaded images
            message_id: Optional message ID for associating uploaded images
            user_id: User ID for storage permissions
        """
        self._pipe = pipe
        self.valves = valves
        self._event_emitter = event_emitter
        self.logger = logger or LOGGER
        self._request_context = request_context
        self._user_obj = user_obj
        self._chat_id = chat_id
        self._message_id = message_id
        self._user_id = user_id

        # State variables
        self.reasoning_buffer = ""
        self.reasoning_completed_emitted = False
        self.reasoning_stream_active = False
        self.active_reasoning_item_id: str | None = None
        self.reasoning_stream_buffers: dict[str, str] = {}
        self.reasoning_stream_has_incremental: set[str] = set()
        self.reasoning_stream_completed: set[str] = set()
        self.emitted_citations: list[dict] = []
        self.reasoning_status_buffer = ""
        self.reasoning_status_last_emit: float | None = None
        self.processed_image_item_ids: set[str] = set()
        self.ordinal_by_url: dict[str, int] = {}

        # Thinking mode configuration
        self.thinking_mode = getattr(valves, "THINKING_OUTPUT_MODE", "open_webui")
        self.thinking_box_enabled = self.thinking_mode in {"open_webui", "both"}
        self.thinking_status_enabled = self.thinking_mode in {"status", "both"}

        # Storage context cache
        self._storage_context_cache: Optional[tuple[Optional[Any], Optional[Any]]] = None

    @timed
    async def _get_storage_context(self) -> tuple[Optional[Any], Optional[Any]]:
        """Get request and user context for storage operations (cached)."""
        if self._storage_context_cache is None:
            self._storage_context_cache = await self._pipe._resolve_storage_context(
                self._request_context, self._user_obj
            )
        return self._storage_context_cache or (None, None)

    @timed
    async def _persist_generated_image(self, data: bytes, mime_type: str) -> Optional[str]:
        """Upload image data to Open WebUI storage and return the file ID."""
        upload_request, upload_user = await self._get_storage_context()
        if not upload_request or not upload_user:
            return None

        ext = "png"
        if isinstance(mime_type, str) and "/" in mime_type:
            ext = mime_type.split("/")[-1] or "png"
        if ext == "jpg":
            ext = "jpeg"

        filename = f"generated-image-{uuid.uuid4().hex}.{ext}"
        return await self._pipe._upload_to_owui_storage(
            request=upload_request,
            user=upload_user,
            file_data=data,
            filename=filename,
            mime_type=mime_type,
            chat_id=self._chat_id if isinstance(self._chat_id, str) else None,
            message_id=self._message_id if isinstance(self._message_id, str) else None,
            owui_user_id=self._user_id,
        )

    @timed
    def _validate_base64_size(self, b64_string: str) -> bool:
        """Check if base64 string is within reasonable size limits."""
        # Rough estimate: 4 base64 chars = 3 bytes, limit to ~10MB
        max_chars = (10 * 1024 * 1024 * 4) // 3
        return len(b64_string) <= max_chars

    @timed
    async def _materialize_image_from_str(self, data_str: str) -> Optional[str]:
        """Convert image string (data URL or base64) to persistent storage URL."""
        text = (data_str or "").strip()
        if not text:
            return None

        # Handle data URLs
        if text.startswith("data:"):
            parsed = self._pipe._parse_data_url(text)
            if parsed:
                stored = await self._persist_generated_image(parsed["data"], parsed["mime_type"])
                if stored:
                    if self._event_emitter:
                        from ..core.errors import StatusMessages
                        await self._pipe._emit_status(
                            self._event_emitter,
                            StatusMessages.IMAGE_BASE64_SAVED,
                            done=False,
                        )
                    return f"/api/v1/files/{stored}/content"
                return text
            return None

        # Already a URL
        if text.startswith(("http://", "https://", "/")):
            return text

        # Try to extract base64 from various formats
        cleaned = text
        if "," in cleaned and ";base64" in cleaned.split(",", 1)[0]:
            cleaned = cleaned.split(",", 1)[1]
        cleaned = cleaned.strip()
        if not cleaned:
            return None

        if not self._validate_base64_size(cleaned):
            return None

        try:
            decoded = base64.b64decode(cleaned, validate=True)
        except (binascii.Error, ValueError):
            return None

        mime_type = "image/png"
        stored = await self._persist_generated_image(decoded, mime_type)
        if stored:
            if self._event_emitter:
                from ..core.errors import StatusMessages
                await self._pipe._emit_status(
                    self._event_emitter,
                    StatusMessages.IMAGE_BASE64_SAVED,
                    done=False,
                )
            return f"/api/v1/files/{stored}/content"
        return f"data:{mime_type};base64,{cleaned}"

    @timed
    async def _materialize_image_entry(self, entry: Any) -> Optional[str]:
        """Extract and materialize image URL from various entry formats."""
        if entry is None:
            return None

        if isinstance(entry, str):
            return await self._materialize_image_from_str(entry)

        if isinstance(entry, dict):
            # Try URL-like keys
            for key in ("url", "image_url", "imageUrl", "content_url"):
                candidate = entry.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    return candidate.strip()
                if isinstance(candidate, dict):
                    nested = await self._materialize_image_entry(candidate)
                    if nested:
                        return nested

            # Try base64 keys
            for key in ("b64_json", "b64", "base64", "data", "image_base64"):
                b64_val = entry.get(key)
                if isinstance(b64_val, str) and b64_val.strip():
                    cleaned = b64_val.strip()
                    if not self._validate_base64_size(cleaned):
                        continue
                    try:
                        decoded = base64.b64decode(cleaned, validate=True)
                    except (binascii.Error, ValueError):
                        continue

                    mime_type = (
                        entry.get("mime_type")
                        or entry.get("mimeType")
                        or entry.get("content_type")
                        or "image/png"
                    )
                    stored = await self._persist_generated_image(decoded, mime_type)
                    if stored:
                        if self._event_emitter:
                            from ..core.errors import StatusMessages
                            await self._pipe._emit_status(
                                self._event_emitter,
                                StatusMessages.IMAGE_BASE64_SAVED,
                                done=False,
                            )
                        return f"/api/v1/files/{stored}/content"
                    return f"data:{mime_type};base64,{cleaned}"

        return None

    @timed
    async def materialize_images_from_item(self, event: dict[str, Any]) -> list[str]:
        """Detect and persist image URLs from a completed output item.

        Returns:
            List of materialized image URLs
        """
        if not isinstance(event, dict):
            return []

        item = event.get("item")
        if not isinstance(item, dict):
            return []

        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            return []
        if item_id in self.processed_image_item_ids:
            return []

        # Only process image items once
        item_type = item.get("type")
        if item_type != "image":
            return []

        self.processed_image_item_ids.add(item_id)

        # Extract image URLs from various fields
        materialized = []
        for field in ("output", "content", "image"):
            value = item.get(field)
            if isinstance(value, list):
                for entry in value:
                    url = await self._materialize_image_entry(entry)
                    if url:
                        materialized.append(url)
            elif value is not None:
                url = await self._materialize_image_entry(value)
                if url:
                    materialized.append(url)

        return materialized

    @timed
    def _extract_reasoning_text(self, event: dict[str, Any]) -> str:
        """Extract reasoning text from various event payload structures."""
        if not isinstance(event, dict):
            return ""

        # Try direct text fields
        for key in ("delta", "text"):
            value = event.get(key)
            if isinstance(value, str) and value:
                return value

        # Try nested part structure
        part = event.get("part")
        if isinstance(part, dict):
            part_text = part.get("text")
            if isinstance(part_text, str) and part_text:
                return part_text

            # Try content array
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

    @timed
    def _reasoning_stream_key(self, event: dict[str, Any], etype: Optional[str]) -> str:
        """Get a stable key to associate reasoning deltas with upstream items."""
        item_id = event.get("item_id")
        if isinstance(item_id, str) and item_id:
            return item_id

        if etype in {"response.output_item.added", "response.output_item.done"}:
            item_raw = event.get("item")
            item = item_raw if isinstance(item_raw, dict) else {}
            iid = item.get("id")
            if isinstance(iid, str) and iid:
                return iid

        if self.active_reasoning_item_id:
            return self.active_reasoning_item_id

        return "__reasoning__"

    @timed
    def _append_reasoning_text(self, key: str, incoming: str, *, allow_misaligned: bool) -> str:
        """Deduplicate and append reasoning text to buffer.

        Handles both delta (incremental) and snapshot (cumulative) payloads
        without replaying content.

        Returns:
            The actual new text that was appended
        """
        candidate = (incoming or "")
        if not candidate:
            return ""

        current = self.reasoning_stream_buffers.get(key, "")
        append = ""

        if not current:
            # First chunk for this key
            append = candidate
        elif candidate == current:
            # Duplicate snapshot
            append = ""
        elif candidate.startswith(current):
            # Growing snapshot: take the new suffix
            append = candidate[len(current) :]
        elif current.startswith(candidate):
            # Shrinking or duplicate (shouldn't happen)
            append = ""
        else:
            # Misaligned: either delta or corruption
            append = candidate if allow_misaligned else ""

        if append:
            self.reasoning_stream_buffers[key] = f"{current}{append}"
            self.reasoning_buffer += append

        return append

    @timed
    async def _maybe_emit_reasoning_status(self, delta_text: str, *, force: bool = False) -> None:
        """Emit throttled status updates for reasoning progress.

        Batches reasoning text into readable status updates to avoid
        flooding the UI with rapid delta events.

        Args:
            delta_text: New reasoning text to buffer
            force: If True, emit immediately regardless of thresholds
        """
        if not self.thinking_status_enabled:
            return
        if not self._event_emitter:
            return

        self.reasoning_status_buffer += delta_text
        text = self.reasoning_status_buffer.strip()
        if not text:
            return

        should_emit = force
        now = perf_counter()

        if not should_emit:
            # Emit on punctuation
            if delta_text.rstrip().endswith(_REASONING_STATUS_PUNCTUATION):
                should_emit = True
            # Emit on length threshold
            elif len(text) >= _REASONING_STATUS_MAX_CHARS:
                should_emit = True
            # Emit on idle timeout
            else:
                elapsed = (
                    None
                    if self.reasoning_status_last_emit is None
                    else (now - self.reasoning_status_last_emit)
                )
                if len(text) >= _REASONING_STATUS_MIN_CHARS:
                    if elapsed is None or elapsed >= _REASONING_STATUS_IDLE_SECONDS:
                        should_emit = True

        if not should_emit:
            return

        await self._event_emitter(
            {
                "type": "status",
                "data": {"description": text, "done": False},
            }
        )
        self.reasoning_status_buffer = ""
        self.reasoning_status_last_emit = now

    @timed
    def _is_reasoning_event(self, etype: str, event: dict[str, Any]) -> bool:
        """Check if event contains reasoning content."""
        # Direct reasoning events
        is_reasoning_event = (
            etype.startswith("response.reasoning") and etype != "response.reasoning_summary_text.done"
        )

        # Reasoning content parts
        part = event.get("part") if isinstance(event, dict) else None
        reasoning_part_types = {"reasoning_text", "reasoning_summary_text", "summary_text"}
        is_reasoning_part_event = (
            etype.startswith("response.content_part")
            and isinstance(part, dict)
            and part.get("type") in reasoning_part_types
        )

        return is_reasoning_event or is_reasoning_part_event

    @timed
    async def process_reasoning_event(
        self,
        event: dict[str, Any],
        etype: str,
        *,
        normalize_surrogate_chunk: Callable[[str, str], str],
    ) -> Optional[str]:
        """Process a reasoning event and emit appropriate deltas/status.

        Args:
            event: The event dict
            etype: Event type string
            normalize_surrogate_chunk: Callback to normalize text with surrogate pairs

        Returns:
            The appended reasoning text delta, or None if nothing was added
        """
        if not self._is_reasoning_event(etype, event):
            return None

        self.reasoning_stream_active = True

        key = self._reasoning_stream_key(event, etype)
        if not isinstance(key, str) or not key:
            return None
        is_incremental = etype.endswith(".delta") or etype.endswith(".added")
        is_final = etype.endswith(".done") or etype.endswith(".completed")

        delta_text = self._extract_reasoning_text(event)
        normalized_delta = normalize_surrogate_chunk(delta_text, "reasoning") if delta_text else ""

        if normalized_delta and is_incremental:
            self.reasoning_stream_has_incremental.add(key)

        append = ""
        if normalized_delta:
            append = self._append_reasoning_text(
                key,
                normalized_delta,
                allow_misaligned=is_incremental,
            )

        if append:
            if self._event_emitter and self.thinking_box_enabled:
                await self._event_emitter(
                    {
                        "type": "reasoning:delta",
                        "data": {
                            "content": self.reasoning_buffer,
                            "delta": append,
                            "event": etype,
                        },
                    }
                )

            await self._maybe_emit_reasoning_status(append)

        if is_final:
            if self._event_emitter:
                await self._maybe_emit_reasoning_status("", force=True)

                if (
                    self.thinking_box_enabled
                    and self.reasoning_stream_buffers.get(key)
                    and key not in self.reasoning_stream_completed
                ):
                    await self._event_emitter(
                        {
                            "type": "reasoning:completed",
                            "data": {"content": self.reasoning_buffer},
                        }
                    )
                    self.reasoning_stream_completed.add(key)
                    self.reasoning_completed_emitted = True

        return append

    @timed
    async def process_citation_event(self, event: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Extract and emit citation from URL annotation event.

        Args:
            event: Event dict with annotation data

        Returns:
            The citation dict if emitted, None otherwise
        """
        ann = event.get("annotation") or {}
        if ann.get("type") != "url_citation":
            return None

        # Extract URL and title
        url = (ann.get("url") or "").strip()
        if url.endswith("?utm_source=openai"):
            url = url[: -len("?utm_source=openai")]
        title = (ann.get("title") or url).strip()

        # Deduplicate
        if url in self.ordinal_by_url:
            return None

        self.ordinal_by_url[url] = len(self.ordinal_by_url) + 1

        # Build citation metadata
        host = url.split("//", 1)[-1].split("/", 1)[0].lower().lstrip("www.")
        citation = {
            "source": {"name": host or "source", "url": url},
            "document": [title],
            "metadata": [
                {
                    "source": url,
                    "date_accessed": datetime.date.today().isoformat(),
                }
            ],
        }

        if self._event_emitter:
            await self._pipe._emit_citation(self._event_emitter, citation)

        self.emitted_citations.append(citation)
        return citation

    @timed
    def track_reasoning_item(self, event: dict[str, Any]) -> None:
        """Track active reasoning item ID from output_item.added events."""
        item_raw = event.get("item")
        item = item_raw if isinstance(item_raw, dict) else {}
        item_type = item.get("type", "")

        if item_type == "reasoning":
            iid = item.get("id")
            if isinstance(iid, str) and iid:
                self.active_reasoning_item_id = iid

    @timed
    def get_final_reasoning_buffer(self) -> str:
        """Get accumulated reasoning content."""
        return self.reasoning_buffer

    @timed
    def get_citations(self) -> list[dict]:
        """Get all extracted citations."""
        return self.emitted_citations

    @timed
    def should_persist_reasoning(self) -> bool:
        """Check if reasoning tokens should be persisted to conversation history."""
        persist_mode = getattr(self.valves, "PERSIST_REASONING_TOKENS", "never")
        return persist_mode in {"next_reply", "conversation"}

    @timed
    def is_reasoning_active(self) -> bool:
        """Check if reasoning stream is currently active."""
        return self.reasoning_stream_active
