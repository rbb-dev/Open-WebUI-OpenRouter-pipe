"""Message transformation to provider input format.

This module handles the transformation of Open WebUI message format
to OpenRouter/OpenAI Responses API input format, including:
- Multimodal content handling (images, files, audio)
- Tool call artifact management
- Persistence layer integration
- Content pruning and filtering
"""

import asyncio
import base64
import binascii
import contextlib
import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Callable, Awaitable, Tuple, TYPE_CHECKING
from urllib.parse import urlparse
from starlette.requests import Request

# Import helper functions from domain modules
from ..storage.persistence import (
    normalize_persisted_item as _normalize_persisted_item,
)
from ..tools.tool_schema import (
    _classify_function_call_artifacts,
)

# Import utility functions
from ..core.utils import (
    _extract_plain_text_content,
    contains_marker,
    split_text_by_markers,
)

# Import from storage
from ..storage.multimodal import (
    _extract_internal_file_id,
    _is_internal_file_url,
)

# Import from persistence
from ..storage.persistence import generate_item_id

# Import from config
from ..core.config import (
    _MARKDOWN_IMAGE_RE,
    _NON_REPLAYABLE_TOOL_ARTIFACTS,
)

# Import from registry
from ..models.registry import ModelFamily

# Import status messages
from ..core.errors import StatusMessages

if TYPE_CHECKING:
    from ..pipe import Pipe

LOGGER = logging.getLogger(__name__)

# Tool output pruning constants
_TOOL_OUTPUT_PRUNE_MIN_LENGTH = 800  # Minimum length before pruning is applied
_TOOL_OUTPUT_PRUNE_HEAD_CHARS = 256  # Characters to keep from the beginning
_TOOL_OUTPUT_PRUNE_TAIL_CHARS = 128  # Characters to keep from the end

async def transform_messages_to_input(
    self: "Pipe",
    messages: List[Dict[str, Any]],
    chat_id: Optional[str] = None,
    openwebui_model_id: Optional[str] = None,
    artifact_loader: Optional[
        Callable[[Optional[str], Optional[str], List[str]], Awaitable[Dict[str, Dict[str, Any]]]]
    ] = None,
    pruning_turns: int = 0,
    replayed_reasoning_refs: Optional[List[Tuple[str, str]]] = None,
    __request__: Optional[Request] = None,
    user_obj: Optional[Any] = None,
    event_emitter: Optional[Callable] = None,
    *,
    model_id: Optional[str] = None,
    valves: Optional["Pipe.Valves"] = None,
) -> List[Dict[str, Any]]:
    """
    Build an OpenAI Responses-API `input` array from Open WebUI-style messages.

    Parameters `chat_id` and `openwebui_model_id` are optional. When both are
    supplied and the messages contain empty-link encoded item references, the
    function fetches persisted items from the database and injects them in the
    correct order. When either parameter is missing, the messages are simply
    converted without attempting to fetch persisted items.

    When provided, `artifact_loader` is awaited with `(chat_id, message_id, ulids)`
    for each assistant message so database-backed artifacts can be replayed.
    When `replayed_reasoning_refs` is supplied, the function appends each
    `(chat_id, artifact_id)` pair for reasoning items so the caller can clean
    them up after they have been replayed once.

    Returns
    -------
    List[dict] : The fully-formed `input` list for the OpenAI Responses API.
    """

    logger = LOGGER
    active_valves = valves or self.valves
    image_limit = getattr(
        active_valves,
        "MAX_INPUT_IMAGES_PER_REQUEST",
        self.valves.MAX_INPUT_IMAGES_PER_REQUEST,
    )
    selection_mode = getattr(
        active_valves,
        "IMAGE_INPUT_SELECTION",
        self.valves.IMAGE_INPUT_SELECTION,
    )
    chunk_size = getattr(
        active_valves,
        "IMAGE_UPLOAD_CHUNK_BYTES",
        self.valves.IMAGE_UPLOAD_CHUNK_BYTES,
    )
    max_inline_bytes = (
        getattr(
            active_valves,
            "BASE64_MAX_SIZE_MB",
            self.valves.BASE64_MAX_SIZE_MB,
        )
        * 1024
        * 1024
    )
    target_model_id = model_id or openwebui_model_id or ""
    if target_model_id:
        vision_supported = ModelFamily.supports("vision", target_model_id)
    else:
        vision_supported = True

    openai_input: list[dict] = []
    last_assistant_images: list[dict[str, Any]] = []

    def _message_identifier(entry: dict[str, Any]) -> Optional[str]:
        """Return the most specific identifier available on ``entry``."""
        for key in ("id", "_id", "message_id"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _compute_turn_indices() -> tuple[list[Optional[int]], int]:
        """Label each message with a turn index and return the total count."""
        indices: list[Optional[int]] = []
        current_turn = -1
        max_turn = -1
        last_dialog_role: Optional[str] = None

        for msg in messages:
            role = (msg.get("role") or "").lower()
            turn_idx: Optional[int] = None

            if role == "user":
                if last_dialog_role != "user":
                    current_turn += 1
                turn_idx = current_turn
                last_dialog_role = "user"
            elif role == "assistant":
                if current_turn < 0:
                    current_turn = 0
                turn_idx = current_turn
                last_dialog_role = "assistant"
            else:
                turn_idx = current_turn if current_turn >= 0 else None

            if turn_idx is not None and turn_idx > max_turn:
                max_turn = turn_idx
            indices.append(turn_idx)

        total_turns = max_turn + 1 if max_turn >= 0 else 0
        return indices, total_turns

    def _markdown_images_from_text(text: str) -> list[str]:
        """Extract inline Markdown image URLs from a text block."""
        if not isinstance(text, str):
            return []
        return [
            match.group("url").strip()
            for match in _MARKDOWN_IMAGE_RE.finditer(text)
            if match.group("url").strip()
        ]

    def _is_old_turn(turn_index: Optional[int], *, threshold: Optional[int]) -> bool:
        """Return True when a message turn falls outside the retention window."""
        return (
            threshold is not None
            and turn_index is not None
            and turn_index < threshold
        )

    def _prune_tool_output(
        item: Dict[str, Any],
        *,
        marker: Optional[str],
        turn_index: Optional[int],
        retention_turns: int,
    ) -> bool:
        """Shorten oversized tool output strings while leaving markers intact."""
        if item.get("type") != "function_call_output":
            return False

        output_value = item.get("output")
        if output_value is None:
            return False
        if not isinstance(output_value, str):
            try:
                output_text = json.dumps(output_value, ensure_ascii=False)
            except (TypeError, ValueError):
                # Fallback to str() if object isn't JSON serializable
                output_text = str(output_value)
        else:
            output_text = output_value

        if len(output_text) < _TOOL_OUTPUT_PRUNE_MIN_LENGTH:
            return False

        head = output_text[:_TOOL_OUTPUT_PRUNE_HEAD_CHARS].rstrip()
        tail = output_text[-_TOOL_OUTPUT_PRUNE_TAIL_CHARS:].lstrip()
        removed_chars = len(output_text) - len(head) - len(tail)
        if removed_chars <= 0:
            return False

        ellipsis = "..."
        turn_label = (
            f"{retention_turns} turn" if retention_turns == 1 else f"{retention_turns} turns"
        )
        note = (
            f"{ellipsis}\n"
            f"[tool output pruned: removed {removed_chars} char"
            f"{'' if removed_chars == 1 else 's'}, older than {turn_label}]\n"
            f"{ellipsis}"
        )
        item["output"] = "\n".join(
            part for part in (head, note, tail) if part
        )
        LOGGER.debug("Pruned tool output (marker=%s, call_id=%s, turn=%s, removed_chars=%d, retention=%d)", marker, item.get("call_id"), turn_index, removed_chars, retention_turns)
        return True

    turn_indices, total_turns = _compute_turn_indices()
    prune_before_turn: Optional[int] = None
    if pruning_turns > 0 and total_turns > pruning_turns:
        prune_before_turn = total_turns - pruning_turns

    for idx, msg in enumerate(messages):
        raw_role = msg.get("role")
        role = (raw_role or "").lower()
        raw_content = msg.get("content", "")
        msg_id = msg.get("message_id") or _message_identifier(msg)
        msg_turn_index = turn_indices[idx]
        raw_tool_calls = msg.get("tool_calls")
        msg_tool_calls: list[dict[str, Any]] = (
            list(raw_tool_calls)
            if isinstance(raw_tool_calls, list) and raw_tool_calls
            else []
        )

        # -------- system / developer messages --------------------------- #
        if role in {"system", "developer"}:
            blocks: list[dict[str, Any]] = []

            if isinstance(raw_content, str):
                blocks.append({"type": "input_text", "text": raw_content})
            elif isinstance(raw_content, list):
                for entry in raw_content:
                    if isinstance(entry, str):
                        blocks.append({"type": "input_text", "text": entry})
                        continue
                    if isinstance(entry, dict):
                        text_val = entry.get("text")
                        if not isinstance(text_val, str):
                            text_val = entry.get("content")
                        if isinstance(text_val, str):
                            blocks.append({"type": "input_text", "text": text_val})
            elif isinstance(raw_content, dict):
                text_val = raw_content.get("text")
                if not isinstance(text_val, str):
                    text_val = raw_content.get("content")
                if isinstance(text_val, str):
                    blocks.append({"type": "input_text", "text": text_val})

            if blocks:
                openai_input.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": blocks,
                    }
                )
            continue

        # -------- tool response ---------------------------------------- #
        if role == "tool":
            call_id = msg.get("tool_call_id") or msg.get("id") or msg.get("call_id")
            call_id = call_id.strip() if isinstance(call_id, str) else ""
            if not call_id:
                continue

            tool_content = raw_content
            if tool_content is None:
                tool_content_text = ""
            elif isinstance(tool_content, str):
                tool_content_text = tool_content
            else:
                try:
                    tool_content_text = json.dumps(tool_content, ensure_ascii=False)
                except (TypeError, ValueError):
                    tool_content_text = str(tool_content)

            openai_input.append(
                {
                    "type": "function_call_output",
                    "id": f"fc_output_{generate_item_id()}",
                    "call_id": call_id,
                    "output": tool_content_text,
                }
            )
            continue

        # -------- user message ---------------------------------------- #
        if role == "user":
            content_blocks = msg.get("content") or []
            if isinstance(content_blocks, str):
                content_blocks = [{"type": "text", "text": content_blocks}]

            async def _to_input_image(block: dict) -> Optional[dict[str, Any]]:
                """Convert Open WebUI image block into Responses format.

                Handles image URLs and base64 data URLs, downloading remote images and
                saving all images to OWUI storage to prevent data loss.

                Supported Image Formats (per OpenRouter docs):
                    - image/png
                    - image/jpeg
                    - image/webp
                    - image/gif

                Args:
                    block: Content block from Open WebUI message

                Returns:
                    Responses API input_image block with internal OWUI storage URL

                Processing Flow:
                    1. Extract image URL from block (nested or flat structure)
                    2. If data URL: Parse, upload to OWUI storage
                    3. If remote URL: Download, upload to OWUI storage
                    4. If OWUI file reference: Keep as-is
                    5. Return Responses API format with detail level

                Note:
                    Image data URLs and remote URLs are ALWAYS saved to storage
                    to prevent chat history bloat, similar to the default behavior of
                    SAVE_FILE_DATA_CONTENT for file inputs. This cannot be disabled
                    via valve configuration as inline image payloads significantly
                    degrade UI performance and storage efficiency.
                    All errors are caught and logged with status emissions.
                    Failed processing returns empty image_url rather than crashing.
                """
                try:
                    image_payload = block.get("image_url")
                    detail: Optional[str] = None
                    url: str = ""
                    owui_file_id: Optional[str] = None

                    if isinstance(image_payload, dict):
                        url = image_payload.get("url", "")
                        detail = image_payload.get("detail")
                    elif isinstance(image_payload, str):
                        url = image_payload
                        block_detail = block.get("detail")
                        if isinstance(block_detail, str):
                            detail = block_detail

                    if not url:
                        return None

                    storage_context: Optional[Tuple[Optional[Request], Optional[Any]]] = None

                    async def _get_storage_context() -> tuple[Optional[Request], Optional[Any]]:
                        """Resolve (request,user) tuple only once for storage uploads."""
                        nonlocal storage_context
                        if storage_context is None:
                            storage_context = await self._resolve_storage_context(__request__, user_obj)
                        return storage_context

                    async def _save_image_bytes(
                        payload: bytes,
                        mime_type: str,
                        preferred_name: str,
                        status_message: str,
                    ) -> Optional[str]:
                        """Upload image bytes to Open WebUI storage and emit status."""
                        upload_request, upload_user = await _get_storage_context()
                        if not (upload_request and upload_user):
                            return None
                        stored_id = await self._upload_to_owui_storage(
                            request=upload_request,
                            user=upload_user,
                            file_data=payload,
                            filename=preferred_name,
                            mime_type=mime_type,
                            chat_id=chat_id,
                            message_id=msg_id,
                            owui_user_id=getattr(user_obj, "id", None),
                        )
                        if stored_id:
                            await self._emit_status(event_emitter, status_message, done=False)
                        return stored_id

                    if url.startswith("data:"):
                        try:
                            parsed = self._parse_data_url(url)
                            if parsed:
                                ext = parsed["mime_type"].split("/")[-1]
                                stored_id = await _save_image_bytes(
                                    parsed["data"],
                                    parsed["mime_type"],
                                    f"image-{uuid.uuid4().hex}.{ext}",
                                    StatusMessages.IMAGE_BASE64_SAVED,
                                )
                                if stored_id:
                                    owui_file_id = stored_id
                        except Exception as exc:
                            self.logger.error(f"Failed to process base64 image: {exc}")
                            await self._emit_error(
                                event_emitter,
                                f"Failed to save base64 image: {exc}",
                                show_error_message=False
                            )

                    elif url.startswith(("http://", "https://")) and not _is_internal_file_url(url):
                        try:
                            downloaded = await self._download_remote_url(url)
                            if downloaded:
                                filename = url.split("/")[-1].split("?")[0] or f"image-{uuid.uuid4().hex}"
                                if "." not in filename:
                                    ext = downloaded["mime_type"].split("/")[-1]
                                    filename = f"{filename}.{ext}"

                                stored_id = await _save_image_bytes(
                                    downloaded["data"],
                                    downloaded["mime_type"],
                                    filename,
                                    StatusMessages.IMAGE_REMOTE_SAVED,
                                )
                                if stored_id:
                                    owui_file_id = stored_id
                        except Exception as exc:
                            self.logger.error(f"Failed to download remote image {url}: {exc}")
                            await self._emit_error(
                                event_emitter,
                                f"Failed to download image: {exc}",
                                show_error_message=False
                            )
                    if owui_file_id is None and _is_internal_file_url(url):
                        owui_file_id = _extract_internal_file_id(url)

                    if owui_file_id:
                        inlined = await self._inline_owui_file_id(
                            owui_file_id,
                            chunk_size=chunk_size,
                            max_bytes=max_inline_bytes,
                        )
                        if not inlined:
                            await self._emit_status(
                                event_emitter,
                                f"Skipping image {owui_file_id}: Open WebUI file unavailable.",
                                done=False,
                            )
                            return None
                        url = inlined

                    result: dict[str, Any] = {"type": "input_image", "image_url": url}
                    if isinstance(detail, str) and detail in {"auto", "low", "high"}:
                        result["detail"] = detail
                    elif url:
                        result["detail"] = "auto"

                    return result

                except Exception as exc:
                    self.logger.error(f"Error in _to_input_image: {exc}")
                    await self._emit_error(
                        event_emitter,
                        f"Image processing error: {exc}",
                        show_error_message=False
                    )
                    return None

            async def _to_input_file(block: dict) -> dict:
                """Convert Open WebUI file blocks into Responses API format.

                Handles file content blocks from multiple sources, downloading remote files
                and saving base64 data to OWUI storage for persistence.

                Responses API File Input Fields (per OpenAPI spec):
                    - type: "input_file" (required)
                    - file_id: string | null (optional)
                    - file_data: string (optional) - base64 or data URL
                    - filename: string (optional) - for model context
                    - file_url: string (optional) - URL to file

                Args:
                    block: Content block from Open WebUI message

                Returns:
                    Responses API input_file block with all available fields

                Processing Flow:
                    1. Extract fields from nested or flat block structure
                    2. If file_data provided AND SAVE_FILE_DATA_CONTENT enabled:
                       - If data URL: Parse, upload to OWUI storage, set file_url
                       - If remote URL: Download, upload to OWUI storage, set file_url
                    3. If file_id: Keep as-is (already in OWUI storage)
                    4. If file_url provided AND SAVE_REMOTE_FILE_URLS enabled:
                       - If data URL: Parse, upload to OWUI storage
                       - If remote URL: Download, upload to OWUI storage
                    5. Return all available fields to Responses API

                Note:
                    All errors are caught and logged with status emissions.
                    Failed processing returns minimal valid block rather than crashing.
                    Size limits follow BASE64_MAX_SIZE_MB (default 50MB) for inline payloads.
                """
                try:
                    result = {"type": "input_file"}

                    nested_file = block.get("file")
                    source = nested_file if isinstance(nested_file, dict) else block

                    file_id = source.get("file_id")
                    file_data = source.get("file_data")
                    filename = source.get("filename")
                    file_url = source.get("file_url")
                    file_url_set_from_file_data = False

                    def _is_internal_storage(url: str) -> bool:
                        """Return True when a URL already references internal storage."""
                        return isinstance(url, str) and ("/api/v1/files/" in url or "/files/" in url)

                    storage_context: Optional[Tuple[Optional[Request], Optional[Any]]] = None

                    async def _get_storage_context() -> tuple[Optional[Request], Optional[Any]]:
                        """Lazy-load the request/user pair used for uploads."""
                        nonlocal storage_context
                        if storage_context is None:
                            storage_context = await self._resolve_storage_context(__request__, user_obj)
                        return storage_context

                    async def _save_bytes_to_storage(
                        payload: bytes,
                        mime_type: str,
                        *,
                        preferred_name: Optional[str],
                        status_message: str,
                    ) -> Optional[str]:
                        """Persist arbitrary bytes to Open WebUI storage and emit status."""
                        upload_request, upload_user = await _get_storage_context()
                        if not (upload_request and upload_user):
                            return None
                        safe_mime = mime_type or "application/octet-stream"
                        fname = preferred_name or filename or f"file-{uuid.uuid4().hex}"
                        if "." not in fname and safe_mime:
                            ext = safe_mime.split("/")[-1]
                            fname = f"{fname}.{ext}"

                        stored_id = await self._upload_to_owui_storage(
                            request=upload_request,
                            user=upload_user,
                            file_data=payload,
                            filename=fname,
                            mime_type=safe_mime,
                            chat_id=chat_id,
                            message_id=msg_id,
                            owui_user_id=getattr(user_obj, "id", None),
                        )
                        if stored_id:
                            await self._emit_status(
                                event_emitter,
                                status_message,
                                done=False,
                            )
                        return stored_id

                    async def _download_and_store(
                        remote_url: str,
                        *,
                        name_hint: Optional[str] = None,
                    ) -> Optional[str]:
                        """Download a remote file and persist it via `_save_bytes_to_storage`."""
                        downloaded = await self._download_remote_url(remote_url)
                        if not downloaded:
                            return None
                        derived_name = (
                            name_hint
                            or remote_url.split("/")[-1].split("?")[0]
                            or f"file-{uuid.uuid4().hex}"
                        )
                        return await _save_bytes_to_storage(
                            downloaded["data"],
                            downloaded.get("mime_type") or "application/octet-stream",
                            preferred_name=derived_name,
                            status_message=StatusMessages.FILE_REMOTE_SAVED,
                        )

                    # Normalize internal OWUI URLs into a file_id (preferred).
                    if isinstance(file_url, str) and file_url.strip() and _is_internal_file_url(file_url.strip()):
                        extracted = _extract_internal_file_id(file_url.strip())
                        if extracted:
                            file_id = extracted
                            file_url = None
                    if isinstance(file_data, str) and file_data.strip() and _is_internal_file_url(file_data.strip()):
                        extracted = _extract_internal_file_id(file_data.strip())
                        if extracted:
                            file_id = extracted
                            file_data = None

                    if (
                        file_data
                        and isinstance(file_data, str)
                        and self.valves.SAVE_FILE_DATA_CONTENT
                    ):
                        if file_data.startswith("data:"):
                            try:
                                parsed = self._parse_data_url(file_data)
                                if parsed:
                                    fname = filename or f"file-{uuid.uuid4().hex}"
                                    stored_id = await _save_bytes_to_storage(
                                        parsed["data"],
                                        parsed["mime_type"],
                                        preferred_name=fname,
                                        status_message=StatusMessages.FILE_BASE64_SAVED,
                                    )
                                    if stored_id:
                                        file_id = stored_id
                                        file_url = None
                                        file_data = None  # Clear base64; store via OWUI id instead.
                            except Exception as exc:
                                self.logger.error(f"Failed to process base64 file: {exc}")
                                await self._emit_error(
                                    event_emitter,
                                    f"Failed to save base64 file: {exc}",
                                    show_error_message=False
                                )

                        elif file_data.startswith(("http://", "https://")) and not _is_internal_storage(file_data):
                            try:
                                remote_url = file_data
                                fname = filename or remote_url.split("/")[-1].split("?")[0]
                                stored_id = await _download_and_store(remote_url, name_hint=fname)
                                if stored_id:
                                    file_id = stored_id
                                    file_url = None
                                    file_url_set_from_file_data = True
                                else:
                                    if not file_url:
                                        file_url = remote_url
                                        file_url_set_from_file_data = True
                                    if event_emitter:
                                        label = fname or "remote file"
                                        if not fname:
                                            with contextlib.suppress(Exception):
                                                host = urlparse(remote_url).netloc
                                                if host:
                                                    label = host
                                        await self._emit_notification(
                                            event_emitter,
                                            f"Unable to download/re-host file '{label}'. Using the remote URL as-is.",
                                            level="warning",
                                        )
                                file_data = None  # Clear, use URL instead
                            except Exception as exc:
                                self.logger.error(f"Failed to download remote file: {exc}")
                                await self._emit_error(
                                    event_emitter,
                                    f"Failed to download file: {exc}",
                                    show_error_message=False
                                )

                    if (
                        file_url
                        and isinstance(file_url, str)
                        and self.valves.SAVE_REMOTE_FILE_URLS
                        and not file_url_set_from_file_data
                    ):
                        if file_url.startswith("data:"):
                            try:
                                parsed = self._parse_data_url(file_url)
                                if parsed:
                                    fname = filename or f"file-{uuid.uuid4().hex}"
                                    stored_id = await _save_bytes_to_storage(
                                        parsed["data"],
                                        parsed["mime_type"],
                                        preferred_name=fname,
                                        status_message=StatusMessages.FILE_BASE64_SAVED,
                                    )
                                    if stored_id:
                                        file_id = stored_id
                                        file_url = None
                            except Exception as exc:
                                self.logger.error(f"Failed to process base64 file_url: {exc}")
                                await self._emit_error(
                                    event_emitter,
                                    f"Failed to save base64 file URL: {exc}",
                                    show_error_message=False
                                )
                        elif file_url.startswith(("http://", "https://")) and not _is_internal_storage(file_url):
                            try:
                                name_hint = filename or file_url.split("/")[-1].split("?")[0]
                                stored_id = await _download_and_store(file_url, name_hint=name_hint)
                                if stored_id:
                                    file_id = stored_id
                                    file_url = None
                                else:
                                    if event_emitter:
                                        label = name_hint or "remote file"
                                        if not name_hint:
                                            with contextlib.suppress(Exception):
                                                host = urlparse(file_url).netloc
                                                if host:
                                                    label = host
                                        await self._emit_notification(
                                            event_emitter,
                                            f"Unable to download/re-host file '{label}'. Using the remote URL as-is.",
                                            level="warning",
                                        )
                            except Exception as exc:
                                self.logger.error(f"Failed to download remote file_url: {exc}")
                                await self._emit_error(
                                    event_emitter,
                                    f"Failed to download file URL: {exc}",
                                    show_error_message=False
                                )

                    if file_id:
                        result["file_id"] = file_id
                    if file_data:
                        result["file_data"] = file_data
                    if filename:
                        result["filename"] = filename
                    if file_url:
                        result["file_url"] = file_url

                    return result

                except Exception as exc:
                    self.logger.error(f"Error in _to_input_file: {exc}")
                    await self._emit_error(
                        event_emitter,
                        f"File processing error: {exc}",
                        show_error_message=False
                    )
                    return {"type": "input_file"}

            async def _to_input_audio(block: dict) -> dict:
                """Convert Open WebUI audio blocks into Responses API format.

                Handles audio content blocks, transforming various input formats into
                the Responses API audio input format.

                Responses API Audio Input Format (per OpenAPI spec):
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "<base64_audio_data>",
                            "format": "mp3" | "wav"
                        }
                    }

                OpenRouter Audio Requirements (per documentation):
                    - Audio must be base64-encoded (URLs NOT supported)
                    - Supported formats: wav, mp3 only
                    - See: https://openrouter.ai/docs/guides/overview/multimodal/audio

                Input Formats Handled:
                    1. Chat Completions: {"type": "input_audio", "input_audio": "<base64>"}
                    2. Tool output: {"type": "audio", "mimeType": "audio/mp3", "data": "<base64>"}
                    3. Already correct: {"type": "input_audio", "input_audio": {"data": "...", "format": "..."}}

                Args:
                    block: Content block from Open WebUI message

                Returns:
                    Responses API input_audio block with data and format

                MIME Type to Format Mapping:
                    - audio/mpeg, audio/mp3 -> "mp3"
                    - audio/wav, audio/wave, audio/x-wav -> "wav"
                    - Unknown types default to "mp3"

                Note:
                    All errors are caught and logged with status emissions.
                    Failed processing returns minimal valid block rather than crashing.
                """
                format_map = {
                    "audio/mpeg": "mp3",
                    "audio/mp3": "mp3",
                    "audio/wav": "wav",
                    "audio/wave": "wav",
                    "audio/x-wav": "wav",
                }
                supported_formats = {"mp3", "wav"}

                def _map_format(mime: Optional[str]) -> str:
                    if not isinstance(mime, str):
                        return "mp3"
                    return format_map.get(mime.lower(), "mp3")

                def _empty_audio_block() -> dict[str, Any]:
                    return {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "",
                            "format": "mp3",
                        },
                    }

                def _normalize_base64(data: str) -> Optional[str]:
                    if not data:
                        return None
                    cleaned = "".join(data.split())
                    if not cleaned:
                        return None
                    if not self._validate_base64_size(cleaned):
                        return None
                    try:
                        base64.b64decode(cleaned, validate=True)
                    except (binascii.Error, ValueError):
                        return None
                    return cleaned

                def _resolved_mime_hint(payload: Optional[dict[str, Any]] = None) -> Optional[str]:
                    candidates: list[Any] = []
                    if isinstance(payload, dict):
                        candidates.extend(
                            [
                                payload.get("mimeType"),
                                payload.get("mime_type"),
                            ]
                        )
                    candidates.extend(
                        [
                            block.get("mimeType"),
                            block.get("mime_type"),
                            block.get("contentType"),
                            block.get("content_type"),
                        ]
                    )
                    for value in candidates:
                        if isinstance(value, str):
                            stripped = value.strip()
                            if stripped:
                                return stripped
                    return None

                def _normalize_format(
                    explicit_format: Optional[str],
                    mime_hint: Optional[str],
                ) -> str:
                    if isinstance(explicit_format, str):
                        normalized = explicit_format.strip().lower()
                        if normalized in supported_formats:
                            return normalized
                    return _map_format(mime_hint)

                def _build_audio_block(data: str, audio_format: str) -> dict[str, Any]:
                    return {
                        "type": "input_audio",
                        "input_audio": {
                            "data": data,
                            "format": audio_format,
                        },
                    }

                try:
                    audio_payload = block.get("input_audio") or block.get("data") or block.get("blob")

                    if isinstance(audio_payload, dict) and "data" in audio_payload and "format" in audio_payload:
                        cleaned = _normalize_base64(audio_payload.get("data", ""))
                        if not cleaned:
                            self.logger.warning("Audio payload rejected: invalid base64 data.")
                            await self._emit_error(
                                event_emitter,
                                "Audio input was not valid base64.",
                                show_error_message=False,
                            )
                            return _empty_audio_block()
                        audio_format = _normalize_format(audio_payload.get("format"), _resolved_mime_hint(audio_payload))
                        return _build_audio_block(cleaned, audio_format)

                    if isinstance(audio_payload, dict):
                        raw_data = audio_payload.get("data")
                        if isinstance(raw_data, str):
                            cleaned = _normalize_base64(raw_data)
                            if not cleaned:
                                self.logger.warning("Audio payload rejected: invalid base64 data.")
                                await self._emit_error(
                                    event_emitter,
                                    "Audio input was not valid base64.",
                                    show_error_message=False,
                                )
                                return _empty_audio_block()
                            mime_hint = _resolved_mime_hint(audio_payload)
                            audio_format = _normalize_format(audio_payload.get("format"), mime_hint)
                            return _build_audio_block(cleaned, audio_format)

                    if isinstance(audio_payload, str):
                        sanitized = audio_payload.strip()
                        lowercase = sanitized.lower()
                        if lowercase.startswith(("http://", "https://")):
                            self.logger.warning("Audio payload rejected: remote URLs are not supported.")
                            await self._emit_error(
                                event_emitter,
                                "Audio input must be base64-encoded. URLs are not supported.",
                                show_error_message=False,
                            )
                            return _empty_audio_block()

                        if lowercase.startswith("data:"):
                            parsed = self._parse_data_url(sanitized if sanitized.startswith("data:") else f"data:{sanitized.split(':', 1)[1]}")
                            if not parsed or not parsed.get("mime_type", "").startswith("audio/"):
                                self.logger.warning("Audio payload rejected: invalid data URL.")
                                await self._emit_error(
                                    event_emitter,
                                    "Audio input must be base64-encoded audio data.",
                                    show_error_message=False,
                                )
                                return _empty_audio_block()
                            if not self._validate_base64_size(parsed.get("b64", "")):
                                return _empty_audio_block()
                            audio_format = _map_format(parsed.get("mime_type"))
                            return _build_audio_block(parsed.get("b64", ""), audio_format)

                        cleaned = _normalize_base64(sanitized)
                        if not cleaned:
                            self.logger.warning("Audio payload rejected: invalid base64 data.")
                            await self._emit_error(
                                event_emitter,
                                "Audio input was not valid base64.",
                                show_error_message=False,
                            )
                            return _empty_audio_block()

                        mime_type = _resolved_mime_hint()
                        audio_format = _map_format(mime_type)
                        return _build_audio_block(cleaned, audio_format)

                    # Invalid/empty
                    self.logger.warning("Invalid audio payload format, returning empty audio block")
                    return _empty_audio_block()

                except Exception as exc:
                    self.logger.error(f"Error in _to_input_audio: {exc}")
                    await self._emit_error(
                        event_emitter,
                        f"Audio processing error: {exc}",
                        show_error_message=False,
                    )
                    return _empty_audio_block()

            async def _to_input_video(block: dict) -> dict:
                """Convert Open WebUI video blocks into Chat Completions video format.

                Note: The Responses API doesn't have explicit `input_video` type.
                Videos use the Chat Completions `video_url` format, which OpenRouter
                handles internally.

                Video Support by Provider (per OpenRouter docs):
                    - Gemini AI Studio: YouTube links only
                    - Most providers: Limited or no video support
                    - Check model's input_modalities for "video" capability

                Supported Video Formats:
                    - Remote URLs: YouTube links, direct video URLs
                    - Data URLs: data:video/mp4;base64,... (rarely used due to size)
                    - OWUI file references: /api/v1/files/...

                Chat Completions Video Format:
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": "https://youtube.com/..." or "data:video/mp4;base64,..."
                        }
                    }

                Input Formats Handled:
                    1. {"type": "video_url", "video_url": {"url": "..."}}
                    2. {"type": "video_url", "video_url": "..."}
                    3. {"type": "video", "url": "...", "mimeType": "video/mp4"}

                Args:
                    block: Content block from Open WebUI message

                Returns:
                    Chat Completions video_url block

                Note:
                    Videos are NOT downloaded/stored due to large size.
                    URL validation is minimal - OpenRouter will validate provider support.
                """
                try:
                    video_payload = block.get("video_url")
                    url: str = ""

                    if isinstance(video_payload, dict):
                        url = video_payload.get("url", "")
                    elif isinstance(video_payload, str):
                        url = video_payload

                    if not url:
                        url = block.get("url", "")

                    if not url:
                        self.logger.warning("Video block has no URL")
                        return {"type": "video_url", "video_url": {"url": ""}}

                    if url.startswith("data:"):
                        if "," in url:
                            b64_data = url.split(",", 1)[1]
                            # Estimate decoded size (base64 is ~33% larger than raw)
                            estimated_size_bytes = (len(b64_data) * 3) // 4
                            max_size_bytes = self.valves.VIDEO_MAX_SIZE_MB * 1024 * 1024
                            if estimated_size_bytes > max_size_bytes:
                                estimated_size_mb = estimated_size_bytes / (1024 * 1024)
                                self.logger.warning(
                                    f"Base64 video size (~{estimated_size_mb:.1f}MB) exceeds configured limit "
                                    f"({self.valves.VIDEO_MAX_SIZE_MB}MB), rejecting to prevent memory issues"
                                )
                                await self._emit_error(
                                    event_emitter,
                                    f"Video too large (~{estimated_size_mb:.1f}MB, max: {self.valves.VIDEO_MAX_SIZE_MB}MB)",
                                    show_error_message=True
                                )
                                return {"type": "video_url", "video_url": {"url": ""}}

                        await self._emit_status(
                            event_emitter,
                            StatusMessages.VIDEO_BASE64,
                            done=False
                        )
                    elif self._is_youtube_url(url):
                        # Note: YouTube videos only work with Gemini models (per OpenRouter docs)
                        await self._emit_status(
                            event_emitter,
                            StatusMessages.VIDEO_YOUTUBE,
                            done=False
                        )
                    elif url.startswith(("http://", "https://")) and not ("/api/v1/files/" in url or "/files/" in url):
                        # Apply SSRF protection for non-OWUI URLs
                        if not await self._is_safe_url(url):
                            self.logger.error(f"SSRF protection blocked video URL: {url}")
                            await self._emit_error(
                                event_emitter,
                                "Video URL blocked by security policy (private network)",
                                show_error_message=True
                            )
                            return {"type": "video_url", "video_url": {"url": ""}}

                        await self._emit_status(
                            event_emitter,
                            StatusMessages.VIDEO_REMOTE,
                            done=False
                        )
                    else:
                        await self._emit_status(
                            event_emitter,
                            StatusMessages.VIDEO_REMOTE,
                            done=False
                        )

                    return {
                        "type": "video_url",
                        "video_url": {
                            "url": url
                        }
                    }

                except Exception as exc:
                    self.logger.error(f"Error in _to_input_video: {exc}")
                    await self._emit_error(
                        event_emitter,
                        f"Video processing error: {exc}",
                        show_error_message=False
                    )
                    return {"type": "video_url", "video_url": {"url": ""}}

            def _identity_block(b: dict[str, Any]) -> dict[str, Any]:
                return b

            block_transform = {
                "text":       lambda b: {"type": "input_text",  "text": b.get("text", "")},
                "image_url":  _to_input_image,
                "input_image": _to_input_image,
                "input_file": _to_input_file,
                "file":       _to_input_file,  # Chat Completions format
                "input_audio": _to_input_audio,  # Responses API audio format
                "audio":      _to_input_audio,   # Open WebUI tool audio output format
                "video_url":  _to_input_video,   # Chat Completions video format
                "video":      _to_input_video,   # Alternative video format
            }

            converted_blocks: list[dict[str, Any]] = []
            user_images_used = 0
            dropped_images = 0
            encountered_user_images = False
            vision_warning_sent = False
            latest_user_message = role == "user" and idx == len(messages) - 1
            include_user_images = (
                latest_user_message and vision_supported and image_limit > 0
            )

            for block in content_blocks:
                if not block:
                    continue
                raw_block_type = block.get("type")
                block_type = raw_block_type if isinstance(raw_block_type, str) else ""
                transformer = block_transform.get(block_type, _identity_block)
                is_image_block = block_type in {"image_url", "input_image"}

                if is_image_block:
                    encountered_user_images = True
                    if not include_user_images:
                        if latest_user_message and not vision_supported and not vision_warning_sent:
                            await self._emit_status(
                                event_emitter,
                                "Model does not accept image inputs; skipping user attachments.",
                                done=False,
                            )
                            vision_warning_sent = True
                        continue
                    if user_images_used >= image_limit:
                        dropped_images += 1
                        continue

                try:
                    if asyncio.iscoroutinefunction(transformer):
                        result = await transformer(block)
                    else:
                        result = transformer(block)
                    if result is None:
                        continue
                    if is_image_block and result:
                        user_images_used += 1
                    converted_blocks.append(result)
                except Exception as exc:
                    self.logger.error(f"Failed to transform block type '{block_type}': {exc}")
                    await self._emit_error(
                        event_emitter,
                        f"Block transformation error for '{block_type}': {exc}",
                        show_error_message=False
                    )
                    if not is_image_block:
                        converted_blocks.append(block)

            if (
                latest_user_message
                and selection_mode == "user_then_assistant"
                and include_user_images
                and user_images_used == 0
                and last_assistant_images
            ):
                fallback_slots = min(image_limit, len(last_assistant_images))
                fallback_blocks: list[dict[str, Any]] = []
                for source_block in last_assistant_images[:fallback_slots]:
                    try:
                        transformed = await _to_input_image(source_block)
                        if transformed is not None:
                            fallback_blocks.append(transformed)
                    except Exception as exc:
                        self.logger.error("Failed to reuse assistant image: %s", exc)
                if fallback_blocks:
                    self.logger.debug(
                        "Rehydrating %d assistant-generated image(s) due to empty user attachments (selection_mode=%s, limit=%d).",
                        len(fallback_blocks),
                        selection_mode,
                        image_limit,
                    )
                    converted_blocks = fallback_blocks + converted_blocks
                    user_images_used = len(fallback_blocks)

            if dropped_images and latest_user_message:
                await self._emit_status(
                    event_emitter,
                    f"Dropped {dropped_images} extra image{'s' if dropped_images != 1 else ''}; limit is {image_limit}.",
                    done=False,
                )
            if (
                latest_user_message
                and encountered_user_images
                and not vision_supported
                and not vision_warning_sent
            ):
                await self._emit_status(
                    event_emitter,
                    "Model does not accept image inputs; skipping user attachments.",
                    done=False,
                )

            openai_input.append({
                "type": "message",
                "role": "user",
                "content": converted_blocks,
            })
            continue

        # -------- assistant message ----------------------------------- #
        raw_msg_annotations = msg.get("annotations")
        msg_annotations: list[Any] = (
            list(raw_msg_annotations)
            if isinstance(raw_msg_annotations, list) and raw_msg_annotations
            else []
        )
        raw_msg_reasoning_details = msg.get("reasoning_details")
        msg_reasoning_details: list[Any] = (
            list(raw_msg_reasoning_details)
            if isinstance(raw_msg_reasoning_details, list) and raw_msg_reasoning_details
            else []
        )
        assistant_text = (
            raw_content
            if isinstance(raw_content, str)
            else _extract_plain_text_content(raw_content)
        )
        is_old_message = _is_old_turn(msg_turn_index, threshold=prune_before_turn)
        assistant_image_urls = _markdown_images_from_text(assistant_text)
        if assistant_image_urls:
            last_assistant_images = [
                {"type": "image_url", "image_url": url, "detail": "auto"}
                for url in assistant_image_urls
            ]
        else:
            last_assistant_images = []

        # If tool_calls are provided explicitly (native OWUI tool execution flow),
        # do not attempt DB artifact replay for this message to avoid duplicate injection.
        if (not msg_tool_calls) and contains_marker(assistant_text):
            segments = split_text_by_markers(assistant_text)
            markers = [seg["marker"] for seg in segments if seg.get("type") == "marker"]

            db_artifacts: dict[str, dict] = {}
            orphaned_call_ids: set[str] = set()
            orphaned_output_ids: set[str] = set()
            if artifact_loader and chat_id and openwebui_model_id and markers:
                try:
                    db_artifacts = await artifact_loader(chat_id, msg_id, markers)
                    (
                        _,
                        orphaned_call_ids,
                        orphaned_output_ids,
                    ) = _classify_function_call_artifacts(db_artifacts)
                    if orphaned_call_ids:
                        logger.warning(
                            "Dropping %d persisted function_call artifact(s) missing outputs (chat_id=%s message_id=%s call_ids=%s)",
                            len(orphaned_call_ids),
                            chat_id,
                            msg_id,
                            sorted(orphaned_call_ids),
                        )
                    if orphaned_output_ids:
                        logger.warning(
                            "Dropping %d persisted function_call_output artifact(s) missing calls (chat_id=%s message_id=%s call_ids=%s)",
                            len(orphaned_output_ids),
                            chat_id,
                            msg_id,
                            sorted(orphaned_output_ids),
                        )
                except Exception:
                    # Catch all loader errors (DB, network, etc.) - pipe must continue
                    logger.warning("Artifact loader failed for chat_id=%s message_id=%s", chat_id, msg_id, exc_info=True)
                    db_artifacts = {}

            for segment in segments:
                if segment["type"] == "marker":
                    payload = db_artifacts.get(segment["marker"])
                    if payload is None:
                        logger.warning("Missing artifact %s for chat_id=%s message_id=%s", segment["marker"], chat_id, msg_id)
                        continue
                    if (
                        payload.get("type") == "reasoning"
                        and replayed_reasoning_refs is not None
                        and chat_id
                    ):
                        replayed_reasoning_refs.append((chat_id, segment["marker"]))
                    item = _normalize_persisted_item(payload)
                    if item is not None:
                        item_type = ((item.get("type") or "").lower())
                        if item_type in _NON_REPLAYABLE_TOOL_ARTIFACTS:
                            self.logger.debug(
                                "Skipping %s artifact when rebuilding provider context (not replayable).",
                                item_type,
                            )
                            continue
                        if (
                            item_type == "function_call"
                            and item.get("call_id") in orphaned_call_ids
                        ):
                            logger.debug(
                                "Skipping orphaned function_call artifact (call_id=%s chat_id=%s message_id=%s)",
                                item.get("call_id"),
                                chat_id,
                                msg_id,
                            )
                            continue
                        if (
                            item_type == "function_call_output"
                            and item.get("call_id") in orphaned_output_ids
                        ):
                            logger.debug(
                                "Skipping orphaned function_call_output artifact (call_id=%s chat_id=%s message_id=%s)",
                                item.get("call_id"),
                                chat_id,
                                msg_id,
                            )
                            continue
                        if (
                            is_old_message
                            and pruning_turns > 0
                            and prune_before_turn is not None
                        ):
                            _prune_tool_output(
                                item,
                                marker=segment["marker"],
                                turn_index=msg_turn_index,
                                retention_turns=pruning_turns,
                            )
                        openai_input.append(item)
                elif segment["type"] == "text" and segment["text"].strip():
                    item_out = {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": segment["text"].strip()}]
                    }
                    if msg_annotations:
                        item_out["annotations"] = msg_annotations
                    if msg_reasoning_details:
                        item_out["reasoning_details"] = msg_reasoning_details
                    openai_input.append(item_out)
        else:
            # Plain assistant text (no markers detected)
            if assistant_text:
                item_out = {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": assistant_text}],
                }
                if msg_annotations:
                    item_out["annotations"] = msg_annotations
                if msg_reasoning_details:
                    item_out["reasoning_details"] = msg_reasoning_details
                openai_input.append(item_out)

        # Native OWUI tool calling: assistant.tool_calls -> Responses function_call items.
        if msg_tool_calls:
            for index, tool_call in enumerate(msg_tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                if tool_call.get("type") not in (None, "function"):
                    continue

                tool_call_id = tool_call.get("id") or tool_call.get("call_id")
                tool_call_id = tool_call_id.strip() if isinstance(tool_call_id, str) else ""
                if not tool_call_id:
                    tool_call_id = f"call_{generate_item_id()}_{index}"

                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                name = name.strip() if isinstance(name, str) else ""
                if not name:
                    continue

                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    args_text = arguments.strip() or "{}"
                else:
                    try:
                        args_text = json.dumps(arguments or {}, ensure_ascii=False)
                    except (TypeError, ValueError):
                        args_text = "{}"

                openai_input.append(
                    {
                        "type": "function_call",
                        "id": tool_call_id,
                        "call_id": tool_call_id,
                        "name": name,
                        "arguments": args_text,
                    }
                )

    self._maybe_apply_anthropic_prompt_caching(
        openai_input,
        model_id=target_model_id,
        valves=active_valves,
    )
    return openai_input
