"""Open WebUI file handler adapter.

Integration with Open WebUI's file storage system:
- File upload to Open WebUI storage
- Storage user management with lazy creation
- MIME type inference and normalization
- Base64 encoding for file content
- SSRF protection for remote URLs
- YouTube URL detection

Layer: adapters (Open WebUI storage integration)

Dependencies:
- Open WebUI: Files, Users, upload_file_handler, run_in_threadpool
- httpx: HTTP client for remote downloads
- core.markers: generate_item_id (for filenames)
- domain.multimodal: File limit configuration

Security:
- SSRF protection validates DNS resolution against private IP ranges
- File size limits prevent memory exhaustion
- Storage user runs with least-privilege role
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import logging
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from fastapi import BackgroundTasks, Request
    from fastapi.datastructures import Headers, UploadFile

# Conditional imports for Open WebUI integration
try:
    from open_webui.apps.webui.models.files import Files
    from open_webui.apps.webui.models.users import Users
    from open_webui.storage import upload_file_handler
    from open_webui.utils import run_in_threadpool
except (ImportError, ModuleNotFoundError):
    Files = None  # type: ignore[assignment,misc]
    Users = None  # type: ignore[assignment,misc]
    upload_file_handler = None  # type: ignore[assignment]
    run_in_threadpool = None  # type: ignore[assignment]

# Module logger
LOGGER = logging.getLogger(__name__)


async def get_user_by_id(user_id: str) -> Optional[Any]:
    """Fetch user record from database for file upload operations.

    Args:
        user_id: The unique identifier for the user

    Returns:
        UserModel object if found, None otherwise

    Note:
        Uses run_in_threadpool to avoid blocking async operations.
        Failures are logged but do not raise exceptions.

    Example:
        >>> user = await get_user_by_id("user123")
        >>> if user:
        ...     print(user.email)
    """
    if Users is None or run_in_threadpool is None:
        return None
    try:
        return await run_in_threadpool(Users.get_user_by_id, user_id)
    except Exception as exc:
        LOGGER.error(f"Failed to load user {user_id}: {exc}")
        return None


async def get_file_by_id(file_id: str) -> Optional[Any]:
    """Look up file metadata from Open WebUI's file storage.

    Args:
        file_id: The unique identifier for the file

    Returns:
        FileModel object if found, None otherwise

    Note:
        Uses run_in_threadpool to avoid blocking async operations.
        Failures are logged but do not raise exceptions.

    Example:
        >>> file_obj = await get_file_by_id("file123")
        >>> if file_obj:
        ...     print(file_obj.path)
    """
    if Files is None or run_in_threadpool is None:
        return None
    try:
        return await run_in_threadpool(Files.get_file_by_id, file_id)
    except Exception as exc:
        LOGGER.error(f"Failed to load file {file_id}: {exc}")
        return None


def infer_file_mime_type(file_obj: Any) -> str:
    """Return the best-known MIME type for a stored Open WebUI file.

    Args:
        file_obj: File model object from Open WebUI

    Returns:
        MIME type string (defaults to 'application/octet-stream')

    Note:
        - Checks mime_type, content_type attributes
        - Checks meta dict for content_type/mimeType/mime_type
        - Normalizes 'image/jpg' to 'image/jpeg'

    Example:
        >>> mime = infer_file_mime_type(file_obj)
        >>> mime
        'image/jpeg'
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


async def inline_internal_file_url(
    file_obj: Any,
    *,
    chunk_size: int,
    max_bytes: int,
) -> Optional[str]:
    """Convert an Open WebUI file object into a data URL for providers.

    Args:
        file_obj: File model object from Open WebUI
        chunk_size: Size of chunks for streaming reads
        max_bytes: Maximum file size in bytes

    Returns:
        Data URL string (e.g., "data:image/jpeg;base64,...") or None if failed

    Example:
        >>> data_url = await inline_internal_file_url(file_obj, chunk_size=65536, max_bytes=10485760)
        >>> if data_url:
        ...     print(data_url[:50])
        'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABA...'
    """
    if not file_obj:
        return None
    mime_type = infer_file_mime_type(file_obj)
    try:
        b64 = await read_file_record_base64(file_obj, chunk_size, max_bytes)
    except ValueError as exc:
        LOGGER.warning("Failed to inline file: %s", exc)
        return None
    if not b64:
        return None
    return f"data:{mime_type};base64,{b64}"


async def read_file_record_base64(
    file_obj: Any,
    chunk_size: int,
    max_bytes: int,
) -> Optional[str]:
    """Return a base64 string for a stored Open WebUI file.

    Args:
        file_obj: File model object from Open WebUI
        chunk_size: Size of chunks for streaming reads
        max_bytes: Maximum file size in bytes

    Returns:
        Base64-encoded file content or None

    Raises:
        ValueError: If file exceeds size limit

    Note:
        Tries multiple sources in order:
        1. file_obj.data dict (b64/base64/data/bytes keys)
        2. file_obj.path/file_path/absolute_path
        3. file_obj.content/blob/data attributes

    Example:
        >>> b64 = await read_file_record_base64(file_obj, 65536, 10485760)
        >>> len(b64)
        14256
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")

    def _from_bytes(raw: bytes) -> str:
        if len(raw) > max_bytes:
            raise ValueError("File exceeds size limit")
        return base64.b64encode(raw).decode("ascii")

    data_field = getattr(file_obj, "data", None)
    if isinstance(data_field, dict):
        for key in ("b64", "base64", "data"):
            inline_value = data_field.get(key)
            if isinstance(inline_value, str) and inline_value.strip():
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
        return await encode_file_path_base64(path, chunk_size, max_bytes)

    raw_bytes = None
    for attr in ("content", "blob", "data"):
        value = getattr(file_obj, attr, None)
        if isinstance(value, (bytes, bytearray)):
            raw_bytes = bytes(value)
            break
    if raw_bytes is not None:
        return _from_bytes(raw_bytes)
    return None


async def encode_file_path_base64(
    path: Path,
    chunk_size: int,
    max_bytes: int,
) -> str:
    """Read ``path`` in chunks and return a base64 string.

    Args:
        path: Path to file on disk
        chunk_size: Size of chunks for streaming reads
        max_bytes: Maximum file size in bytes

    Returns:
        Base64-encoded file content

    Raises:
        ValueError: If file exceeds size limit

    Note:
        Chunks are padded to 3-byte boundaries for clean base64 encoding.

    Example:
        >>> b64 = await encode_file_path_base64(Path("/tmp/image.jpg"), 65536, 10485760)
        >>> isinstance(b64, str)
        True
    """
    chunk_size = max(64 * 1024, min(chunk_size, max_bytes))

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
                    raise ValueError("File exceeds size limit")
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


async def upload_to_owui_storage(
    request: Any,  # Request type
    user: Any,
    file_data: bytes,
    filename: str,
    mime_type: str
) -> Optional[str]:
    """Upload file or image to Open WebUI storage and return internal URL.

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

    Returns:
        Internal URL path to the uploaded file (e.g., '/api/v1/files/{id}'),
        or None if upload fails

    Note:
        - File processing is disabled (process=False) to avoid unnecessary overhead
        - Uses run_in_threadpool to prevent blocking async event loop
        - Failures are logged but return None rather than raising exceptions

    Example:
        >>> url = await upload_to_owui_storage(
        ...     request, user, image_bytes, "photo.jpg", "image/jpeg"
        ... )
        >>> # url = '/api/v1/files/abc123...'
    """
    if upload_file_handler is None or run_in_threadpool is None:
        return None

    try:
        # Dynamic import to avoid circular dependencies
        from fastapi import BackgroundTasks
        from fastapi.datastructures import Headers
        from starlette.datastructures import UploadFile as StarletteUploadFile

        file_item = await run_in_threadpool(
            upload_file_handler,
            request=request,
            file=StarletteUploadFile(
                file=io.BytesIO(file_data),
                filename=filename,
                headers=Headers({"content-type": mime_type}),
            ),
            metadata={"mime_type": mime_type},
            process=False,  # Disable processing to avoid overhead
            process_in_background=False,
            user=user,
            background_tasks=BackgroundTasks(),
        )
        # Generate internal URL path
        internal_url = request.app.url_path_for("get_file_content_by_id", id=file_item.id)
        LOGGER.info(
            f"Uploaded {filename} ({len(file_data):,} bytes) to OWUI storage: {internal_url}"
        )
        return internal_url
    except Exception as exc:
        LOGGER.error(f"Failed to upload {filename} to OWUI storage: {exc}")
        return None


def is_youtube_url(url: str) -> bool:
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
        >>> is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        True
        >>> is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        True
        >>> is_youtube_url("https://vimeo.com/123456")
        False
    """
    if not url:
        return False

    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(?:https?://)?(?:www\.)?youtu\.be/[\w-]+',
    ]

    return any(re.match(pattern, url, re.IGNORECASE) for pattern in patterns)


async def is_safe_url(url: str, *, enable_ssrf_protection: bool = True) -> bool:
    """Async wrapper to validate URLs without blocking the event loop.

    Args:
        url: URL to validate
        enable_ssrf_protection: Whether to enforce SSRF checks

    Returns:
        True if URL is safe, False otherwise

    Example:
        >>> await is_safe_url("https://example.com/image.jpg")
        True
        >>> await is_safe_url("http://192.168.1.1/admin")
        False
    """
    if not enable_ssrf_protection:
        return True
    return await asyncio.to_thread(is_safe_url_blocking, url)


def is_safe_url_blocking(url: str) -> bool:
    """Blocking implementation of the SSRF guard (runs in a thread).

    Args:
        url: URL to validate

    Returns:
        True if URL is safe, False if targeting private networks

    Note:
        Validates against:
        - Private IP ranges (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
        - Loopback addresses (127.x.x.x, ::1)
        - Link-local addresses
        - Multicast addresses
        - Reserved IP ranges

    Example:
        >>> is_safe_url_blocking("https://example.com")
        True
        >>> is_safe_url_blocking("http://localhost/admin")
        False
    """
    try:
        import ipaddress
        import socket

        parsed = urlparse(url)
        host = parsed.hostname

        if not host:
            LOGGER.warning(f"URL has no hostname: {url}")
            return False

        ip_objects: list = []
        seen_ips: set = set()

        def _record_ip(candidate) -> None:
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
                LOGGER.warning(f"DNS resolution failed for: {host}")
                return False
            except Exception as exc:
                LOGGER.error(f"Unexpected DNS error for {host}: {exc}")
                return False

            for _, _, _, _, sockaddr in addrinfo:
                if not sockaddr:
                    continue
                ip_str = sockaddr[0]
                try:
                    resolved_ip = ipaddress.ip_address(ip_str)
                except ValueError:
                    LOGGER.warning(f"Invalid IP address format: {ip_str}")
                    return False
                _record_ip(resolved_ip)

        if not ip_objects:
            LOGGER.warning(f"No IP addresses resolved for: {host}")
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

            LOGGER.warning(f"Blocked SSRF attempt to {reason} IP: {url} ({ip})")
            return False

        return True

    except Exception as exc:
        # Defensive: treat validation errors as unsafe
        LOGGER.error(f"URL safety validation failed for {url}: {exc}")
        return False


__all__ = [
    "get_user_by_id",
    "get_file_by_id",
    "infer_file_mime_type",
    "inline_internal_file_url",
    "read_file_record_base64",
    "encode_file_path_base64",
    "upload_to_owui_storage",
    "is_youtube_url",
    "is_safe_url",
    "is_safe_url_blocking",
]
