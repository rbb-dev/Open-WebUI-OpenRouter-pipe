"""Multimodal content handling (files, images, audio, video).

This module provides:
- File retrieval from Open WebUI storage
- Remote URL downloads with SSRF protection
- File uploads to Open WebUI storage
- Image processing and data URL handling
- Chat file tracking
"""

from __future__ import annotations

import asyncio
import base64
import datetime
import email.utils
import hashlib
import inspect
import io
import logging
import mimetypes
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional
from urllib.parse import quote, urlparse, parse_qs

# External dependencies
import aiohttp
import httpx
from fastapi import BackgroundTasks, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential
from ..core.timing_logger import timed

# Open WebUI integration
try:
    from open_webui.models.files import Files  # type: ignore[import-not-found]
except ImportError:
    Files = None  # type: ignore
try:
    from open_webui.models.users import Users  # type: ignore[import-not-found]
except ImportError:
    Users = None  # type: ignore
try:
    from open_webui.routers.files import upload_file_handler  # type: ignore[import-not-found]
except ImportError:
    upload_file_handler = None  # type: ignore

# Internal imports
from ..core.utils import _retry_after_seconds, _coerce_bool, _coerce_positive_int

# Constants
_OPENROUTER_SITE_URL = "https://openrouter.ai"
_MAX_MODEL_PROFILE_IMAGE_BYTES = 2 * 1024 * 1024
_REMOTE_FILE_MAX_SIZE_DEFAULT_MB = 50
_REMOTE_FILE_MAX_SIZE_MAX_MB = 500
_INTERNAL_FILE_ID_PATTERN = re.compile(r"/files/([A-Za-z0-9-]+)(?:/|\\?|$)")

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Standalone Utility Functions
# -----------------------------------------------------------------------------

@timed
def _guess_image_mime_type(url: str, content_type: str | None, data: bytes) -> str | None:
    """Guess MIME type for image data by inspecting magic bytes and URL extension.

    Args:
        url: Source URL (for extension fallback)
        content_type: HTTP Content-Type header value
        data: Raw image bytes

    Returns:
        Detected MIME type or None if unrecognized
    """
    content_type = (content_type or "").split(";", 1)[0].strip().lower()
    if content_type.startswith("image/"):
        return content_type
    allow_extension_fallback = not content_type or content_type in {
        "application/octet-stream",
        "binary/octet-stream",
    }

    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith((b"\x00\x00\x01\x00", b"\x00\x00\x02\x00")):
        return "image/x-icon"

    head = data[:512].lstrip().lower()
    if head.startswith((b"<svg", b"<?xml")) and b"<svg" in head:
        return "image/svg+xml"

    if allow_extension_fallback:
        path = (urlparse(url).path or "").lower()
        if path.endswith(".svg"):
            return "image/svg+xml"
        if path.endswith(".png"):
            return "image/png"
        if path.endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        if path.endswith(".webp"):
            return "image/webp"
        if path.endswith(".gif"):
            return "image/gif"
        if path.endswith((".ico", ".cur")):
            return "image/x-icon"

    return None


@timed
def _extract_openrouter_og_image(html: str) -> str | None:
    """Extract OpenGraph or Twitter image URL from HTML meta tags.

    Args:
        html: HTML content to parse

    Returns:
        Image URL or None if not found
    """
    if not isinstance(html, str) or not html:
        return None
    patterns = (
        r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image["\']',
        r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image["\']',
    )
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


@timed
def _extract_internal_file_id(url: str) -> Optional[str]:
    """Return the Open WebUI file identifier embedded in a storage URL.

    Args:
        url: URL to extract file ID from

    Returns:
        File ID or None if not found
    """
    if not isinstance(url, str):
        return None
    match = _INTERNAL_FILE_ID_PATTERN.search(url)
    if match:
        return match.group(1)
    return None


@timed
def _classify_retryable_http_error(
    exc: httpx.HTTPStatusError,
) -> tuple[bool, Optional[float]]:
    """Return (is_retryable, retry_after_seconds) for an HTTPStatusError.

    Args:
        exc: HTTP status error to classify

    Returns:
        Tuple of (is_retryable, retry_after_seconds)
    """
    response = exc.response
    if response is None:
        return False, None
    status_code = response.status_code
    if status_code >= 500 or status_code in {408, 425, 429}:
        return True, _retry_after_seconds(response.headers.get("retry-after"))
    return False, None


@timed
def _read_rag_file_constraints() -> tuple[bool, Optional[int]]:
    """Return (rag_enabled, rag_file_size_mb) gleaned from Open WebUI config.

    Returns:
        Tuple of (RAG enabled, max file size in MB)
    """
    module = _get_open_webui_config_module()
    if module is None:
        return False, None

    bypass_raw = _unwrap_config_value(getattr(module, "BYPASS_EMBEDDING_AND_RETRIEVAL", None))
    bypass_bool = _coerce_bool(bypass_raw)
    rag_enabled = True if bypass_bool is None else not bypass_bool

    limit_mb: Optional[int] = None
    for attr_name in ("RAG_FILE_MAX_SIZE", "FILE_MAX_SIZE"):
        attr_value = _unwrap_config_value(getattr(module, attr_name, None))
        limit_mb = _coerce_positive_int(attr_value)
        if limit_mb is not None:
            break

    if limit_mb is not None:
        limit_mb = min(limit_mb, _REMOTE_FILE_MAX_SIZE_MAX_MB)

    return rag_enabled, limit_mb


_OPEN_WEBUI_CONFIG_MODULE: Any | None = None


@timed
def _get_open_webui_config_module() -> Any | None:
    """Return the cached open_webui.config module if available."""
    global _OPEN_WEBUI_CONFIG_MODULE
    if _OPEN_WEBUI_CONFIG_MODULE is not None:
        return _OPEN_WEBUI_CONFIG_MODULE
    try:
        import open_webui.config  # type: ignore[import-not-found]
        _OPEN_WEBUI_CONFIG_MODULE = open_webui.config
        return _OPEN_WEBUI_CONFIG_MODULE
    except ImportError:
        return None


@timed
def _unwrap_config_value(value: Any) -> Any:
    """Unwrap ConfigVar wrapper from Open WebUI config values."""
    if value is None:
        return None
    if hasattr(value, "value"):
        return getattr(value, "value", None)
    return value


class _RetryableHTTPStatusError(Exception):
    """Wrapper that marks an HTTPStatusError as retryable."""

    @timed
    def __init__(self, original: httpx.HTTPStatusError, retry_after: Optional[float] = None):
        """Capture the original HTTP error plus optional Retry-After hint.

        Args:
            original: The original HTTPStatusError
            retry_after: Optional retry delay in seconds from Retry-After header
        """
        self.original = original
        self.retry_after = retry_after
        status_code = getattr(original.response, "status_code", "unknown")
        super().__init__(f"Retryable HTTP error ({status_code})")


class _RetryWait:
    """Custom Tenacity wait strategy honoring Retry-After headers."""

    @timed
    def __init__(self, base_wait):
        """Store the wrapped Tenacity wait callable used as a baseline.

        Args:
            base_wait: Base wait strategy from tenacity
        """
        self._base_wait = base_wait

    @timed
    def __call__(self, retry_state):
        """Return the greater of base delay or Retry-After header guidance.

        Args:
            retry_state: Tenacity retry state

        Returns:
            Delay in seconds
        """
        base_delay = self._base_wait(retry_state) if self._base_wait else 0
        exc = None
        if retry_state.outcome is not None:
            try:
                exc = retry_state.outcome.exception()
            except Exception:
                exc = None
        if isinstance(exc, _RetryableHTTPStatusError):
            retry_after = exc.retry_after
            if isinstance(retry_after, (int, float)) and retry_after > 0:
                return max(base_delay, retry_after)
        return base_delay


# -----------------------------------------------------------------------------
# MultimodalHandler Class
# -----------------------------------------------------------------------------

class MultimodalHandler:
    """Manages multimodal content operations.

    This class encapsulates all file, image, audio, and video operations including:
    - File retrieval from Open WebUI storage
    - Remote URL downloads with SSRF protection
    - File uploads to Open WebUI storage
    - Image processing and data URL handling
    - Chat file tracking

    Architecture:
    - aiohttp for HTTP downloads
    - Open WebUI database integration for file records
    - Base64 encoding for data URLs
    - MIME type detection
    - YouTube URL handling
    - SSRF protection with IP address validation
    """

    @timed
    def __init__(
        self,
        logger: logging.Logger,
        valves: Any,  # Pipe.Valves
        http_session: Optional[aiohttp.ClientSession] = None,
        artifact_store: Optional[Any] = None,  # ArtifactStore
        emit_status_callback: Optional[Callable] = None,
    ):
        """Initialize the MultimodalHandler with dependencies from Pipe.

        Args:
            logger: Logger instance for diagnostics
            valves: Pipe.Valves instance with configuration
            http_session: Optional aiohttp session for remote downloads (can be set later)
            artifact_store: Optional ArtifactStore for file persistence
            emit_status_callback: Optional callback for status updates
        """
        self.logger = logger
        self.valves = valves
        self._http_session = http_session
        self._artifact_store = artifact_store
        self._emit_status_callback = emit_status_callback

        # Cache for storage user lookup
        self._storage_user_cache: Optional[Any] = None
        self._storage_user_lock: Optional[asyncio.Lock] = None
        self._storage_role_warning_emitted = False
        self._user_insert_param_names: Optional[tuple] = None

    # -----------------------------------------------------------------
    # 1. FILE OPERATIONS (7 methods)
    # -----------------------------------------------------------------

    @timed
    async def _get_file_by_id(self, file_id: str):
        """Look up file metadata from Open WebUI's file storage.

        Args:
            file_id: The unique identifier for the file

        Returns:
            FileModel object if found, None otherwise

        Note:
            Uses run_in_threadpool to avoid blocking async operations.
            Failures are logged but do not raise exceptions.
        """
        if Files is None or run_in_threadpool is None:
            self.logger.debug(f"Cannot load file {file_id}: Open WebUI integration not available")
            return None
        try:
            return await run_in_threadpool(Files.get_file_by_id, file_id)
        except Exception as exc:
            self.logger.error(f"Failed to load file {file_id}: {exc}")
            return None

    @timed
    def _infer_file_mime_type(self, file_obj: Any) -> str:
        """Return the best-known MIME type for a stored Open WebUI file.

        Args:
            file_obj: File object from Open WebUI storage

        Returns:
            MIME type string (defaults to application/octet-stream)
        """
        candidates = [
            getattr(file_obj, "mime_type", None),
            getattr(file_obj, "content_type", None),
        ]
        meta = getattr(file_obj, "meta", None) or {}
        if isinstance(meta, dict):
            candidates.extend(
                [
                    meta.get("content_type"),
                    meta.get("mimeType"),
                    meta.get("mime_type"),
                ]
            )
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                normalized = candidate.strip().lower()
                if normalized == "image/jpg":
                    return "image/jpeg"
                return normalized
        return "application/octet-stream"

    @timed
    async def _inline_owui_file_id(
        self,
        file_id: str,
        *,
        chunk_size: int,
        max_bytes: int,
    ) -> Optional[str]:
        """Convert an Open WebUI file id into a data URL for providers.

        Args:
            file_id: Open WebUI file identifier
            chunk_size: Chunk size for reading file
            max_bytes: Maximum file size in bytes

        Returns:
            Data URL string or None if conversion fails
        """
        normalized = (file_id or "").strip()
        if not normalized:
            return None
        file_obj = await self._get_file_by_id(normalized)
        if not file_obj:
            return None
        mime_type = self._infer_file_mime_type(file_obj)
        try:
            b64 = await self._read_file_record_base64(file_obj, chunk_size, max_bytes)
        except ValueError as exc:
            self.logger.warning("Failed to inline file %s: %s", normalized, exc)
            return None
        if not b64:
            return None
        return f"data:{mime_type};base64,{b64}"

    @timed
    async def _inline_internal_file_url(
        self,
        url: str,
        *,
        chunk_size: int,
        max_bytes: int,
    ) -> Optional[str]:
        """Convert an Open WebUI file URL into a data URL for providers.

        Args:
            url: Open WebUI file URL
            chunk_size: Chunk size for reading file
            max_bytes: Maximum file size in bytes

        Returns:
            Data URL string or None if conversion fails
        """
        file_id = _extract_internal_file_id(url)
        if not file_id:
            return None
        return await self._inline_owui_file_id(file_id, chunk_size=chunk_size, max_bytes=max_bytes)

    @timed
    async def _inline_internal_responses_input_files_inplace(
        self,
        request_body: dict[str, Any],
        *,
        chunk_size: int,
        max_bytes: int,
    ) -> None:
        """Inline any Open WebUI internal file URLs referenced by /responses input_file blocks.

        OpenRouter providers cannot fetch Open WebUI internal URLs. This converts internal
        `file_url` (or internal `file_data` URL) values into `file_data` data URLs.

        Args:
            request_body: Request body dict containing input items
            chunk_size: Chunk size for reading files
            max_bytes: Maximum file size in bytes

        Raises:
            ValueError: If inlining fails for a required file
        """
        input_items = request_body.get("input")
        if not isinstance(input_items, list) or not input_items:
            return

        for item in input_items:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list) or not content:
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "input_file":
                    continue

                file_url = block.get("file_url")
                file_data = block.get("file_data")
                file_id = block.get("file_id")

                internal_file_id: str | None = None

                if isinstance(file_id, str) and file_id.strip():
                    candidate_id = file_id.strip()
                    # Avoid clobbering legitimate OpenAI/OpenRouter file ids (commonly `file-...`).
                    if candidate_id.startswith("file-"):
                        continue
                    internal_file_id = candidate_id
                elif isinstance(file_data, str) and file_data.strip() and _is_internal_file_url(file_data.strip()):
                    internal_file_id = _extract_internal_file_id(file_data.strip())
                elif isinstance(file_url, str) and file_url.strip() and _is_internal_file_url(file_url.strip()):
                    internal_file_id = _extract_internal_file_id(file_url.strip())

                if not internal_file_id:
                    continue

                inlined = await self._inline_owui_file_id(
                    internal_file_id,
                    chunk_size=chunk_size,
                    max_bytes=max_bytes,
                )
                if not inlined:
                    raise ValueError(
                        f"Failed to inline Open WebUI file id for /responses: {internal_file_id}"
                    )

                block["file_data"] = inlined
                block.pop("file_id", None)
                if isinstance(file_url, str) and file_url.strip() and _is_internal_file_url(file_url.strip()):
                    block.pop("file_url", None)

    @timed
    async def _read_file_record_base64(
        self,
        file_obj: Any,
        chunk_size: int,
        max_bytes: int,
    ) -> Optional[str]:
        """Return a base64 string for a stored Open WebUI file.

        Args:
            file_obj: File object from Open WebUI storage
            chunk_size: Chunk size for reading file
            max_bytes: Maximum file size in bytes

        Returns:
            Base64-encoded file content or None

        Raises:
            ValueError: If file exceeds size limit
        """
        if max_bytes <= 0:
            raise ValueError("BASE64_MAX_SIZE_MB must be greater than zero")

        @timed
        def _from_bytes(raw: bytes) -> str:
            if len(raw) > max_bytes:
                raise ValueError("File exceeds BASE64_MAX_SIZE_MB limit")
            return base64.b64encode(raw).decode("ascii")

        data_field = getattr(file_obj, "data", None)
        if isinstance(data_field, dict):
            for key in ("b64", "base64", "data"):
                inline_value = data_field.get(key)
                if isinstance(inline_value, str) and inline_value.strip():
                    if not self._validate_base64_size(inline_value):
                        raise ValueError("Stored base64 payload exceeds configured limit")
                    return inline_value.strip()
            blob_value = data_field.get("bytes")
            if isinstance(blob_value, (bytes, bytearray)):
                return _from_bytes(bytes(blob_value))

        prefer_paths = [
            getattr(file_obj, attr, None)
            for attr in (
                "path",
                "file_path",
                "absolute_path",
            )
        ]
        for candidate in prefer_paths:
            if not isinstance(candidate, str):
                continue
            path = Path(candidate)
            if not path.exists():
                continue
            return await self._encode_file_path_base64(path, chunk_size, max_bytes)

        raw_bytes = None
        for attr in ("content", "blob", "data"):
            value = getattr(file_obj, attr, None)
            if isinstance(value, (bytes, bytearray)):
                raw_bytes = bytes(value)
                break
        if raw_bytes is not None:
            return _from_bytes(raw_bytes)
        return None

    @timed
    async def _encode_file_path_base64(
        self,
        path: Path,
        chunk_size: int,
        max_bytes: int,
    ) -> str:
        """Read ``path`` in chunks and return a base64 string.

        Args:
            path: Path to file
            chunk_size: Chunk size for reading
            max_bytes: Maximum file size in bytes

        Returns:
            Base64-encoded file content

        Raises:
            ValueError: If file exceeds size limit
        """
        chunk_size = max(64 * 1024, min(chunk_size, max_bytes))

        @timed
        def _encode_stream() -> str:
            total = 0
            buffer = io.StringIO()
            leftover = b""
            with path.open("rb") as source:
                while True:
                    chunk = source.read(chunk_size)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("File exceeds BASE64_MAX_SIZE_MB limit")
                    chunk = leftover + chunk
                    whole_bytes = (len(chunk) // 3) * 3
                    if whole_bytes:
                        buffer.write(base64.b64encode(chunk[:whole_bytes]).decode("ascii"))
                    leftover = chunk[whole_bytes:]
            if leftover:
                buffer.write(base64.b64encode(leftover).decode("ascii"))
            return buffer.getvalue()

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _encode_stream)

    # -----------------------------------------------------------------
    # 2. UPLOAD TO OWUI STORAGE (4 methods)
    # -----------------------------------------------------------------

    @timed
    async def _upload_to_owui_storage(
        self,
        request: Request,
        user,
        file_data: bytes,
        filename: str,
        mime_type: str,
        chat_id: Optional[str] = None,
        message_id: Optional[str] = None,
        owui_user_id: Optional[str] = None,
    ) -> Optional[str]:
        """Upload file or image to Open WebUI storage and return the OWUI file id.

        This method ensures that files and images are persistently stored in Open WebUI's
        file system, preventing data loss from:
        - Remote URLs that may become inaccessible
        - Base64 data that is temporary
        - External image hosts that delete images after a period

        Args:
            request: FastAPI Request object for URL generation
            user: UserModel object representing the file owner
            file_data: Raw bytes of the file content
            filename: Desired filename (will be prefixed with UUID)
            mime_type: MIME type of the file (e.g., 'image/jpeg', 'application/pdf')
            chat_id: Optional chat ID to associate file with
            message_id: Optional message ID to associate file with
            owui_user_id: Optional user ID override

        Returns:
            Open WebUI file id (UUID string), or None if upload fails.

        Note:
            - File processing is disabled (process=False) to avoid unnecessary overhead
            - Uses run_in_threadpool to prevent blocking async event loop
            - Failures are logged but return None rather than raising exceptions

        Example:
            >>> file_id = await self._upload_to_owui_storage(
            ...     request, user, image_bytes, "photo.jpg", "image/jpeg"
            ... )
            >>> # file_id = 'abc123...'
        """
        if run_in_threadpool is None or upload_file_handler is None:
            self.logger.error("Open WebUI file upload helpers are unavailable; skipping OWUI storage upload.")
            return None
        try:
            upload_metadata: dict[str, Any] = {"mime_type": mime_type}
            if isinstance(chat_id, str):
                normalized_chat_id = chat_id.strip()
                if normalized_chat_id and not normalized_chat_id.startswith("local:"):
                    upload_metadata["chat_id"] = normalized_chat_id
            if isinstance(message_id, str):
                normalized_message_id = message_id.strip()
                if normalized_message_id:
                    upload_metadata["message_id"] = normalized_message_id

            file_item = await run_in_threadpool(
                upload_file_handler,
                request=request,
                file=UploadFile(
                    file=io.BytesIO(file_data),
                    filename=filename,
                    headers=Headers({"content-type": mime_type}),
                ),
                metadata=upload_metadata,
                process=False,  # Disable processing to avoid overhead
                process_in_background=False,
                user=user,
                background_tasks=BackgroundTasks(),
            )
            # Generate internal URL path
            file_id: Optional[str] = None
            if hasattr(file_item, "id"):
                candidate = getattr(file_item, "id", None)
                if isinstance(candidate, str) and candidate.strip():
                    file_id = candidate.strip()
            elif isinstance(file_item, dict):
                raw_id = file_item.get("id")
                if isinstance(raw_id, str) and raw_id.strip():
                    file_id = raw_id.strip()
            if not file_id:
                self.logger.error("Upload handler returned an object without an id; aborting OWUI storage write.")
                return None

            effective_user_id: Optional[str] = None
            if isinstance(owui_user_id, str) and owui_user_id.strip():
                effective_user_id = owui_user_id.strip()
            else:
                candidate = getattr(user, "id", None)
                if isinstance(candidate, str) and candidate.strip():
                    effective_user_id = candidate.strip()

            try:
                await run_in_threadpool(
                    self._try_link_file_to_chat,
                    chat_id=chat_id,
                    message_id=message_id,
                    file_id=file_id,
                    user_id=effective_user_id,
                )
            except Exception:
                pass

            self.logger.info(
                f"Uploaded {filename} ({len(file_data):,} bytes) to OWUI storage: /api/v1/files/{file_id}"
            )
            return file_id
        except Exception as exc:
            self.logger.error(f"Failed to upload {filename} to OWUI storage: {exc}")
            return None

    @timed
    def _try_link_file_to_chat(
        self,
        *,
        chat_id: Optional[str],
        message_id: Optional[str],
        file_id: str,
        user_id: Optional[str],
    ) -> bool:
        """Link uploaded file to chat and message in Open WebUI database.

        Args:
            chat_id: Chat identifier
            message_id: Message identifier
            file_id: File identifier
            user_id: User identifier

        Returns:
            True if linking succeeded, False otherwise
        """
        if not isinstance(chat_id, str):
            return False
        normalized_chat_id = chat_id.strip()
        if not normalized_chat_id or normalized_chat_id.startswith("local:"):
            return False
        if not isinstance(file_id, str) or not file_id.strip():
            return False
        if not isinstance(user_id, str):
            return False
        normalized_user_id = user_id.strip()
        if not normalized_user_id:
            return False

        normalized_message_id: Optional[str] = None
        if isinstance(message_id, str):
            candidate = message_id.strip()
            if candidate:
                normalized_message_id = candidate

        try:
            from open_webui.models.chats import Chats  # type: ignore[import-not-found]
        except Exception:
            return False

        insert_fn = getattr(Chats, "insert_chat_files", None)
        if not callable(insert_fn):
            return False

        try:
            insert_fn(
                chat_id=normalized_chat_id,
                message_id=normalized_message_id,
                file_ids=[file_id.strip()],
                user_id=normalized_user_id,
            )
            return True
        except TypeError:
            try:
                insert_fn(
                    normalized_chat_id,
                    normalized_message_id,
                    [file_id.strip()],
                    normalized_user_id,
                )
                return True
            except Exception:
                return False
        except Exception:
            return False

    @timed
    async def _resolve_storage_context(
        self,
        request: Optional[Request],
        user_obj: Optional[Any],
    ) -> tuple[Optional[Request], Optional[Any]]:
        """Return a `(request, user)` tuple suitable for OWUI uploads.

        Args:
            request: FastAPI Request object
            user_obj: User object

        Returns:
            Tuple of (request, user) or (None, None) if context unavailable
        """
        if request is None:
            if user_obj:
                self.logger.debug("Storage upload skipped: request context missing.")
            return None, None
        if user_obj is not None:
            return request, user_obj

        fallback_user = await self._ensure_storage_user()
        if fallback_user is None:
            return None, None
        self.logger.debug("Using fallback storage user '%s' for upload.", fallback_user.email)
        return request, fallback_user

    @timed
    async def _ensure_storage_user(self) -> Optional[Any]:
        """Ensure the fallback storage user exists (lazy creation).

        Returns:
            User object or None if creation fails
        """
        if Users is None or run_in_threadpool is None:
            self.logger.debug("Cannot create storage user: Open WebUI integration not available")
            return None

        if self._storage_user_cache is not None:
            return self._storage_user_cache

        if self._storage_user_lock is None:
            self._storage_user_lock = asyncio.Lock()

        async with self._storage_user_lock:
            if self._storage_user_cache is not None:
                return self._storage_user_cache

            fallback_email = (self.valves.FALLBACK_STORAGE_EMAIL or "").strip() or "openrouter-pipe@system.local"
            fallback_name = (self.valves.FALLBACK_STORAGE_NAME or "").strip() or "OpenRouter Pipe Storage"
            fallback_role = (self.valves.FALLBACK_STORAGE_ROLE or "").strip() or "pending"

            if (
                fallback_role.lower() in {"admin", "system", "owner"}
                and not self._storage_role_warning_emitted
            ):
                self.logger.warning(
                    "Fallback storage role '%s' is highly privileged. Configure FALLBACK_STORAGE_ROLE to a least-privilege service role if possible.",
                    fallback_role,
                )
                self._storage_role_warning_emitted = True

            try:
                fallback_user = await run_in_threadpool(
                    Users.get_user_by_email,
                    fallback_email,
                )
            except Exception as exc:  # pragma: no cover - defensive guard
                self.logger.error("Failed to load fallback storage user: %s", exc)
                return None

            if fallback_user is None:
                user_id = f"openrouter-pipe-{uuid.uuid4().hex}"
                try:
                    oauth_marker = f"openrouter-pipe-storage:{uuid.uuid4().hex}"
                    insert_fn = Users.insert_new_user
                    insert_kwargs: dict[str, Any] = {}
                    try:
                        if self._user_insert_param_names is None:
                            sig = inspect.signature(insert_fn)
                            self._user_insert_param_names = tuple(sig.parameters.keys())
                    except (TypeError, ValueError):
                        self._user_insert_param_names = ()

                    param_names = self._user_insert_param_names or ()
                    if "oauth" in param_names:
                        insert_kwargs["oauth"] = {"sub": oauth_marker}
                    elif "oauth_sub" in param_names:
                        insert_kwargs["oauth_sub"] = oauth_marker

                    fallback_user = await run_in_threadpool(
                        insert_fn,
                        user_id,
                        fallback_name,
                        fallback_email,
                        "/user.png",
                        fallback_role or "pending",
                        **insert_kwargs,
                    )
                    self.logger.info(
                        "Created fallback storage user '%s' (%s) for multimodal uploads.",
                        fallback_name,
                        fallback_email,
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    self.logger.error("Failed to create fallback storage user: %s", exc)
                    return None

            self._storage_user_cache = fallback_user
            return fallback_user

    # -----------------------------------------------------------------
    # 3. REMOTE DOWNLOADS (5 methods)
    # -----------------------------------------------------------------

    @timed
    async def _download_remote_url(
        self,
        url: str,
        timeout_seconds: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """Download file or image from remote URL with exponential backoff retry logic.

        This method fetches content from HTTP/HTTPS URLs with automatic retry on transient
        failures using exponential backoff. Retry behavior is configurable via valves.

        Args:
            url: The HTTP or HTTPS URL to download
            timeout_seconds: Optional timeout in seconds per attempt (defaults to valve setting, max 60s)

        Returns:
            Dictionary containing:
                - 'data': Raw bytes of the downloaded content
                - 'mime_type': Normalized MIME type from Content-Type header
                - 'url': Original URL for reference
            Returns None if download fails or URL is invalid

        Retry Behavior (configurable via valves):
            - REMOTE_DOWNLOAD_MAX_RETRIES: Maximum retry attempts (default: 3)
            - REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS: Initial delay before first retry (default: 5s)
            - REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS: Maximum total retry time (default: 45s)
            - Uses exponential backoff: delay * 2^attempt

        Retryable Errors:
            - Network errors (connection timeout, DNS failure, etc.)
            - HTTP 5xx server errors
            - HTTP 429 rate limit errors

        Non-Retryable Errors:
            - HTTP 4xx client errors (except 429)
            - Invalid URLs
            - Files exceeding configured size limit (REMOTE_FILE_MAX_SIZE_MB)

        Size Limits:
            - Maximum size configurable via REMOTE_FILE_MAX_SIZE_MB valve (default: 50MB)
            - When Open WebUI RAG uploads enforce FILE_MAX_SIZE, the limit auto-aligns (never exceeding 500MB)
            - Files exceeding the effective limit are rejected with a warning

        Supported Protocols:
            - http:// and https:// only
            - Other protocols return None

        MIME Type Normalization:
            - 'image/jpg' is normalized to 'image/jpeg'
            - MIME type extracted from Content-Type header (charset ignored)

        Note:
            - Timeout is capped at 60 seconds per attempt to prevent hanging requests
            - All exceptions are caught and logged, returning None
            - Empty or non-HTTP URLs return None immediately
            - Retry delays use exponential backoff to be respectful of remote servers

        Example:
            >>> result = await self._download_remote_url(
            ...     "https://example.com/image.jpg"
            ... )
            >>> if result:
            ...     print(f"Downloaded {len(result['data'])} bytes")
            ...     print(f"MIME type: {result['mime_type']}")
        """
        url = (url or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            return None

        # SSRF protection: Validate URL is not targeting private networks
        if not await self._is_safe_url(url):
            self.logger.error(f"SSRF protection blocked download from: {url}")
            return None

        max_retries = self.valves.REMOTE_DOWNLOAD_MAX_RETRIES
        initial_delay = self.valves.REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS
        max_retry_time = self.valves.REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS

        if timeout_seconds is None:
            timeout_seconds = self.valves.HTTP_CONNECT_TIMEOUT_SECONDS
        if timeout_seconds is None:
            timeout_seconds = 60
        timeout_seconds = min(timeout_seconds, 60)

        attempt = 0
        start_time = time.perf_counter()

        try:
            async for attempt_info in AsyncRetrying(
                retry=retry_if_exception_type((_RetryableHTTPStatusError, httpx.NetworkError, httpx.TimeoutException)),
                stop=stop_after_attempt(max_retries + 1),  # +1 because first attempt doesn't count as retry
                wait=_RetryWait(wait_exponential(multiplier=initial_delay, min=initial_delay, max=max_retry_time)),
                reraise=True
            ):
                with attempt_info:
                    attempt += 1

                    elapsed = time.perf_counter() - start_time
                    if attempt > 1 and elapsed > max_retry_time:
                        self.logger.warning(
                            f"Download retry timeout exceeded for {url} after {elapsed:.1f}s"
                        )
                        return None

                    if attempt > 1:
                        self.logger.info(
                            f"Retry attempt {attempt - 1}/{max_retries} for {url} after {elapsed:.1f}s"
                        )

                    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
                        async with client.stream("GET", url) as response:
                            try:
                                response.raise_for_status()
                            except httpx.HTTPStatusError as exc:
                                retryable, retry_after = _classify_retryable_http_error(exc)
                                if retryable:
                                    raise _RetryableHTTPStatusError(exc, retry_after=retry_after) from exc
                                raise

                            mime_type = response.headers.get("content-type", "").split(";")[0].lower().strip()
                            if mime_type == "image/jpg":
                                mime_type = "image/jpeg"

                            # Enforce configurable size limit (valve + optional RAG cap)
                            effective_limit_mb = self._get_effective_remote_file_limit_mb()
                            max_size_bytes = effective_limit_mb * 1024 * 1024

                            content_length = response.headers.get("content-length")
                            if content_length:
                                try:
                                    if int(content_length) > max_size_bytes:
                                        self.logger.warning(
                                            "Remote file %s exceeds configured limit based on Content-Length header "
                                            "(%s bytes > %s bytes); aborting download.",
                                            url,
                                            content_length,
                                            max_size_bytes,
                                        )
                                        return None
                                except ValueError:
                                    pass

                            payload = bytearray()
                            async for chunk in response.aiter_bytes():
                                if not chunk:
                                    continue
                                payload.extend(chunk)
                                if len(payload) > max_size_bytes:
                                    size_mb = len(payload) / (1024 * 1024)
                                    self.logger.warning(
                                        f"Remote file {url} exceeds configured limit "
                                        f"({size_mb:.1f}MB > {effective_limit_mb}MB), aborting download."
                                    )
                                    return None

                    # Success
                    if attempt > 1:
                        elapsed = time.perf_counter() - start_time
                        self.logger.info(
                            f"Successfully downloaded {url} after {attempt} attempt(s) in {elapsed:.1f}s"
                        )

                    return {
                        "data": bytes(payload),
                        "mime_type": mime_type,
                        "url": url
                    }

        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            self.logger.error(
                f"Failed to download {url} after {attempt} attempt(s) in {elapsed:.1f}s: {exc}"
            )
            return None

    @timed
    async def _is_safe_url(self, url: str) -> bool:
        """Async wrapper to validate URLs without blocking the event loop.

        Args:
            url: URL to validate

        Returns:
            True if URL is safe (not targeting private networks)
        """
        if not self.valves.ENABLE_SSRF_PROTECTION:
            return True
        return await asyncio.to_thread(self._is_safe_url_blocking, url)

    @timed
    def _is_safe_url_blocking(self, url: str) -> bool:
        """Blocking implementation of the SSRF guard (runs in a thread).

        Args:
            url: URL to validate

        Returns:
            True if URL is safe, False if it targets private/reserved networks
        """
        try:
            import ipaddress
            import socket
            from ipaddress import IPv4Address, IPv6Address

            parsed = urlparse(url)
            host = parsed.hostname

            if not host:
                self.logger.warning(f"URL has no hostname: {url}")
                return False

            ip_objects: list[IPv4Address | IPv6Address] = []
            seen_ips: set[str] = set()

            @timed
            def _record_ip(candidate: IPv4Address | IPv6Address) -> None:
                comp = candidate.compressed
                if comp not in seen_ips:
                    seen_ips.add(comp)
                    ip_objects.append(candidate)

            # Fast-path literal IPv4/IPv6 hosts
            try:
                literal_ip = ipaddress.ip_address(host)
            except ValueError:
                literal_ip = None
            else:
                _record_ip(literal_ip)

            # Resolve hostname to all available IPs (IPv4 + IPv6) when not a literal
            if literal_ip is None:
                try:
                    addrinfo = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
                except (socket.gaierror, UnicodeError):
                    self.logger.warning(f"DNS resolution failed for: {host}")
                    return False
                except Exception as exc:  # pragma: no cover - defensive guard
                    self.logger.error(f"Unexpected DNS error for {host}: {exc}")
                    return False

                for _, _, _, _, sockaddr in addrinfo:
                    if not sockaddr:
                        continue
                    ip_str = sockaddr[0]
                    try:
                        resolved_ip = ipaddress.ip_address(ip_str)
                    except ValueError:
                        self.logger.warning(f"Invalid IP address format: {ip_str}")
                        return False
                    _record_ip(resolved_ip)

            if not ip_objects:
                self.logger.warning(f"No IP addresses resolved for: {host}")
                return False

            for ip in ip_objects:
                if ip.is_private:
                    reason = "private"
                elif ip.is_loopback:
                    reason = "loopback"
                elif ip.is_link_local:
                    reason = "link-local"
                elif ip.is_multicast:
                    reason = "multicast"
                elif ip.is_reserved:
                    reason = "reserved"
                elif ip.is_unspecified:
                    reason = "unspecified"
                else:
                    continue

                self.logger.warning(f"Blocked SSRF attempt to {reason} IP: {url} ({ip})")
                return False

            return True

        except Exception as exc:
            # Defensive: treat validation errors as unsafe
            self.logger.error(f"URL safety validation failed for {url}: {exc}")
            return False

    @timed
    def _is_youtube_url(self, url: Optional[str]) -> bool:
        """Check if URL is a valid YouTube video URL.

        Supports both standard and short YouTube URL formats:
            - https://www.youtube.com/watch?v=VIDEO_ID
            - https://youtu.be/VIDEO_ID
            - http://youtube.com/watch?v=VIDEO_ID (http variant)

        Args:
            url: URL to validate

        Returns:
            True if URL matches YouTube video pattern, False otherwise

        Note:
            - Does not validate that the video ID exists or is accessible
            - Only checks URL format, not video availability
            - Query parameters (like &t=30s) are allowed

        Example:
            >>> self._is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
            True
            >>> self._is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
            True
            >>> self._is_youtube_url("https://vimeo.com/123456")
            False
        """
        if not url:
            return False

        patterns = [
            r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
            r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
        ]

        return any(re.match(pattern, url, re.IGNORECASE) for pattern in patterns)

    @timed
    def _get_effective_remote_file_limit_mb(self) -> int:
        """Return the active remote download limit, honoring RAG constraints.

        Returns:
            Effective file size limit in MB
        """
        base_limit_mb = self.valves.REMOTE_FILE_MAX_SIZE_MB
        rag_enabled, rag_limit_mb = _read_rag_file_constraints()
        if not rag_enabled or rag_limit_mb is None:
            return base_limit_mb

        # Never exceed Open WebUI's configured FILE_MAX_SIZE when RAG is active.
        if base_limit_mb > rag_limit_mb:
            return rag_limit_mb

        # If the valve is still using the default, upgrade to the RAG cap for consistency.
        if (
            base_limit_mb == _REMOTE_FILE_MAX_SIZE_DEFAULT_MB
            and rag_limit_mb > base_limit_mb
        ):
            return rag_limit_mb
        return base_limit_mb

    # -----------------------------------------------------------------
    # 4. IMAGE PROCESSING (2 methods)
    # -----------------------------------------------------------------

    @timed
    async def _fetch_image_as_data_url(
        self,
        session: aiohttp.ClientSession,
        url: str,
    ) -> str | None:
        """Fetch image from URL and convert to data URL.

        Args:
            session: aiohttp session for HTTP requests
            url: Image URL (supports relative URLs)

        Returns:
            Data URL string or None if fetch/conversion fails
        """
        url = (url or "").strip()
        if not url:
            return None
        if url.startswith("data:image"):
            return url
        if url.startswith("//"):
            url = f"https:{url}"
        elif url.startswith("/"):
            url = f"{_OPENROUTER_SITE_URL}{url}"
        elif not url.startswith(("http://", "https://")):
            url = f"{_OPENROUTER_SITE_URL}/{url.lstrip('/')}"

        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.read()
                if len(data) > _MAX_MODEL_PROFILE_IMAGE_BYTES:
                    self.logger.debug(
                        "Skipping oversized model icon (%d bytes, url=%s)",
                        len(data),
                        url,
                    )
                    return None
                content_type = resp.headers.get("Content-Type")
        except Exception as exc:
            self.logger.debug("Failed to download model icon (url=%s): %s", url, exc)
            return None

        mime = _guess_image_mime_type(url, content_type, data)
        if not mime:
            self.logger.debug(
                "Skipping model icon with unsupported content-type (%s, url=%s)",
                content_type,
                url,
            )
            return None
        if mime == "image/svg+xml":
            try:
                import cairosvg  # type: ignore[import-not-found]
            except Exception as exc:
                self.logger.debug("CairoSVG unavailable; skipping SVG model icon (url=%s): %s", url, exc)
                return None

            try:
                png_bytes = cairosvg.svg2png(
                    bytestring=data,
                    output_width=250,
                    output_height=250,
                )
            except Exception as exc:
                self.logger.debug("Failed to rasterize SVG model icon (url=%s): %s", url, exc)
                return None

            if not isinstance(png_bytes, (bytes, bytearray)):
                self.logger.debug(
                    "Unexpected SVG raster output type '%s' (url=%s)",
                    type(png_bytes).__name__,
                    url,
                )
                return None
            if isinstance(png_bytes, bytearray):
                png_bytes = bytes(png_bytes)

            if len(png_bytes) > _MAX_MODEL_PROFILE_IMAGE_BYTES:
                self.logger.debug(
                    "Skipping oversized rasterized SVG model icon (%d bytes, url=%s)",
                    len(png_bytes),
                    url,
                )
                return None

            encoded = base64.b64encode(png_bytes).decode("ascii")
            return f"data:image/png;base64,{encoded}"

        try:
            from PIL import Image
        except Exception as exc:
            self.logger.debug("Pillow unavailable; skipping model icon conversion (url=%s): %s", url, exc)
            return None

        try:
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                if image.mode not in ("RGB", "RGBA"):
                    image = image.convert("RGBA")
                output = io.BytesIO()
                image.save(output, format="PNG")
                png_bytes = output.getvalue()
        except Exception as exc:
            self.logger.debug("Failed to convert model icon to PNG (url=%s): %s", url, exc)
            return None

        if not isinstance(png_bytes, (bytes, bytearray)):
            self.logger.debug(
                "Unexpected PNG conversion output type '%s' (url=%s)",
                type(png_bytes).__name__,
                url,
            )
            return None
        if isinstance(png_bytes, bytearray):
            png_bytes = bytes(png_bytes)

        if len(png_bytes) > _MAX_MODEL_PROFILE_IMAGE_BYTES:
            self.logger.debug(
                "Skipping oversized converted model icon (%d bytes, url=%s)",
                len(png_bytes),
                url,
            )
            return None

        encoded = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @timed
    async def _fetch_maker_profile_image_url(
        self,
        session: aiohttp.ClientSession,
        maker_id: str,
    ) -> str | None:
        """Fetch OpenRouter maker profile image URL from their page.

        Args:
            session: aiohttp session for HTTP requests
            maker_id: Maker identifier

        Returns:
            Profile image URL or None if not found
        """
        maker_id = (maker_id or "").strip()
        if not maker_id:
            return None
        url = f"{_OPENROUTER_SITE_URL}/{quote(maker_id)}"
        try:
            async with session.get(url) as resp:
                resp.raise_for_status()
                html = await resp.text()
        except Exception as exc:
            self.logger.debug("OpenRouter maker page fetch failed (maker=%s): %s", maker_id, exc)
            return None

        # Protect against bad HTML responses that might crash downstream parsing.
        if not isinstance(html, str):
            self.logger.warning(
                "OpenRouter maker page returned non-string content type '%s' (maker=%s); treating as empty.",
                type(html).__name__,
                maker_id,
            )
            return None
        return _extract_openrouter_og_image(html)

    # -----------------------------------------------------------------
    # 5. DATA URL HANDLING (2 methods)
    # -----------------------------------------------------------------

    @timed
    def _validate_base64_size(self, b64_data: str) -> bool:
        """Validate base64 data size is within configured limits.

        Estimates the decoded size of base64 data and compares it against the
        configured BASE64_MAX_SIZE_MB valve to prevent memory issues from huge payloads.

        Args:
            b64_data: Base64-encoded string to validate

        Returns:
            True if within limits, False if too large

        Note:
            Base64 encoding increases size by approximately 33% (4/3 ratio).
            This method estimates the original size before validation to avoid
            decoding potentially huge strings just to reject them.

        Example:
            >>> if self._validate_base64_size(huge_base64_string):
            ...     decoded = base64.b64decode(huge_base64_string)
            ... else:
            ...     # Reject without decoding
            ...     return None
        """
        if not b64_data:
            return True  # Empty string is valid

        # Base64 is ~1.33x the original size (4/3 ratio)
        # Estimate original size: (base64_length * 3) / 4
        estimated_size_bytes = (len(b64_data) * 3) / 4
        max_size_bytes = self.valves.BASE64_MAX_SIZE_MB * 1024 * 1024

        if estimated_size_bytes > max_size_bytes:
            estimated_size_mb = estimated_size_bytes / (1024 * 1024)
            self.logger.warning(
                f"Base64 data size (~{estimated_size_mb:.1f}MB) exceeds configured limit "
                f"({self.valves.BASE64_MAX_SIZE_MB}MB), rejecting to prevent memory issues"
            )
            return False

        return True

    @timed
    def _parse_data_url(self, data_url: str) -> Optional[Dict[str, Any]]:
        """Extract base64 data from data URL.

        Parses data URLs in the format: data:<mime_type>;base64,<base64_data>

        Args:
            data_url: Data URL string to parse

        Returns:
            Dictionary containing:
                - 'data': Decoded bytes from base64
                - 'mime_type': Normalized MIME type
                - 'b64': Original base64 string (without prefix)
            Returns None if parsing fails or format is invalid

        Format Requirements:
            - Must start with 'data:'
            - Must contain ';base64,' separator
            - Base64 data must be valid
            - Size must not exceed BASE64_MAX_SIZE_MB valve (default: 50MB)

        MIME Type Normalization:
            - 'image/jpg' is normalized to 'image/jpeg'
            - MIME type extracted from prefix (e.g., 'data:image/png;base64,...')

        Size Validation:
            - Validates size before decoding to prevent memory issues
            - Uses BASE64_MAX_SIZE_MB valve for limit
            - Returns None if size exceeds limit

        Note:
            - Invalid base64 data results in None return
            - Oversized data results in None return
            - All exceptions are caught and logged
            - Non-data URLs return None immediately

        Example:
            >>> result = self._parse_data_url(
            ...     "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
            ... )
            >>> if result:
            ...     print(f"MIME: {result['mime_type']}")
            ...     print(f"Size: {len(result['data'])} bytes")
        """
        try:
            if not data_url or not data_url.startswith("data:"):
                return None

            parts = data_url.split(";base64,", 1)
            if len(parts) != 2:
                return None

            # Extract and normalize MIME type
            mime_type = parts[0].replace("data:", "", 1).lower().strip()
            if mime_type == "image/jpg":
                mime_type = "image/jpeg"

            b64_data = parts[1]

            # Validate base64 size before decoding to prevent memory issues
            if not self._validate_base64_size(b64_data):
                return None  # Size validation failed, already logged

            file_data = base64.b64decode(b64_data)

            return {
                "data": file_data,
                "mime_type": mime_type,
                "b64": b64_data
            }
        except Exception as exc:
            self.logger.error(f"Failed to parse data URL: {exc}")
            return None


@timed
def _is_internal_file_url(url: str) -> bool:
    """Heuristic to detect when a URL references Open WebUI file storage."""
    if not isinstance(url, str):
        return False
    return "/api/v1/files/" in url or "/files/" in url
