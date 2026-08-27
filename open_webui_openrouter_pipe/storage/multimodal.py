"""Multimodal content handling (files, images, audio, video).

This module provides:
- File retrieval from Open WebUI storage
- Remote URL downloads with SSRF protection (HTTPS-only by default; HTTP allowlist via valves)
- File uploads to Open WebUI storage
- Image processing and data URL handling
- Chat file tracking
"""

from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
import logging
import re
import socket
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urljoin, urlparse

# External dependencies
import aiohttp
import httpx
from aiohttp.abc import AbstractResolver, ResolveResult
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from yarl import URL as _DialledURL

# Internal imports
from ..core.config import (
    _MAX_MODEL_PROFILE_IMAGE_BYTES,
    _MAX_MODEL_PROFILE_IMAGE_PIXELS,
    _OPENROUTER_SITE_URL,
    _REMOTE_FILE_MAX_SIZE_DEFAULT_MB,
)
from ..core.errors import (
    _classify_retryable_http_error,
    _read_rag_file_constraints,
    _RetryableHTTPStatusError,
    _RetryWait,
)
from ..core.timing_logger import timed
from ..core.url_scheme import is_http_or_https_url
from ..core.warn_latch import warn_level

if TYPE_CHECKING:
    from .owui_files import OwuiFileGateway


# Standalone Utility Functions

ADDRESS_CHECK_SECONDS = 5.0

ADDRESS_CHECK_BUDGET_SECONDS = 20.0

_ICON_DOWNLOAD_CHUNK_BYTES = 64 * 1024

_ICON_FETCH_TIMEOUT_SECONDS = 15.0

_MAKER_PAGE_FETCH_TIMEOUT_SECONDS = 15.0

_MAKER_PAGE_MAX_BYTES = 4 * 1024 * 1024

_MAX_VETTED_REDIRECTS = 5

_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})

_DNS_WARN_LATCH_MAX_HOSTS = 256

_ADDRESS_WARN_COOLDOWN_SECONDS = 300.0

_ICON_DECODE_WORKERS = 2

_VETTED_CONNECTION_LIMIT = 20

_VETTED_CONNECTION_LIMIT_PER_HOST = 10

_VETTED_DNS_CACHE_SECONDS = 300


async def _capped_body(resp: Any, cap: int) -> bytes | None:
    declared = (resp.headers.get("Content-Length") or "").strip()
    if declared.isdigit() and int(declared) > cap:
        return None
    buffer = bytearray()
    async for chunk in resp.content.iter_chunked(_ICON_DOWNLOAD_CHUNK_BYTES):
        if len(buffer) + len(chunk) > cap:
            return None
        buffer.extend(chunk)
    return bytes(buffer)


class UnfetchableAddress(Exception):
    def __init__(self, url: str) -> None:
        super().__init__(f"address is not fetchable: {url}")
        self.url = url


class _AddressRefused(OSError):
    pass


class _IconPixelBudgetExceeded(Exception):
    pass


class _VettedResolver(AbstractResolver):
    def __init__(self, handler: MultimodalHandler, protection: bool) -> None:
        self._handler = handler
        self._protection = protection
        self._fallback = aiohttp.ThreadedResolver()

    async def resolve(
        self,
        host: str,
        port: int = 0,
        family: socket.AddressFamily = socket.AF_INET,
    ) -> list[ResolveResult]:
        if not self._protection:
            return await self._fallback.resolve(host, port, family=family)
        ips = await asyncio.wait_for(
            asyncio.to_thread(self._handler._validated_ips_for_host, host, port),
            timeout=ADDRESS_CHECK_SECONDS,
        )
        if not ips:
            raise _AddressRefused(f"address is not fetchable: {host}")
        return [
            {
                "hostname": host,
                "host": ip,
                "port": port,
                "family": socket.AF_INET6 if ":" in ip else socket.AF_INET,
                "proto": 0,
                "flags": socket.AI_NUMERICHOST,
            }
            for ip in ips
        ]

    async def close(self) -> None:
        await self._fallback.close()


_IMAGE_EXTENSIONS = frozenset(
    {
        "png", "jpeg", "gif", "webp", "svg", "bmp", "tiff", "avif", "heic", "heif",
        "apng", "ico", "x-icon", "jxl", "pjpeg",
    }
)

_IMAGE_EXTENSION_ALIASES = {"jpg": "jpeg", "svg+xml": "svg", "tif": "tiff"}


def canonical_image_mime(declared: str) -> str | None:
    """Return the media type the pipe will store, or None when the declaration is unknown.

    Takes the declaration as the reply gave it -- casing and parameters included -- because
    media types are case-insensitive and a caller that had to normalise first would be one
    more place for the two spellings to diverge.

    The declaration comes from the upstream reply, so it reaches a stored filename and a
    content-type header. Answering from the same closed set the extension comes from is
    what keeps those two from drifting, and what keeps an arbitrary string out of both.
    """
    if not isinstance(declared, str):
        return None
    normalised = declared.split(";", 1)[0].strip().lower()
    if not normalised.startswith("image/"):
        return None
    ext = image_extension_for_mime(normalised)
    if ext == "png" and not normalised.startswith("image/png"):
        return None
    return "image/svg+xml" if ext == "svg" else f"image/{ext}"


def image_extension_for_mime(mime_type: str | None) -> str:
    """Return the filename extension every persistence path uses for an image mime type."""
    ext = "png"
    if isinstance(mime_type, str) and "/" in mime_type:
        ext = (mime_type.split("/")[-1] or "png").split("+")[0]
    ext = _IMAGE_EXTENSION_ALIASES.get(ext, ext)
    return ext if ext in _IMAGE_EXTENSIONS else "png"


def _icon_png_bytes(data: bytes) -> bytes:
    from PIL import Image

    measured = image_pixel_size(data)
    if measured is None:
        if _decodes_during_open(data):
            raise _IconPixelBudgetExceeded
    elif measured[0] * measured[1] > _MAX_MODEL_PROFILE_IMAGE_PIXELS:
        raise _IconPixelBudgetExceeded

    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > _MAX_MODEL_PROFILE_IMAGE_PIXELS:
                raise _IconPixelBudgetExceeded
            image.load()
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA")
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise _IconPixelBudgetExceeded from exc


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


_ICO_PREFIXES = (b"\x00\x00\x01\x00", b"\x00\x00\x02\x00")

_ICO_MAX_FRAMES = 64

_BMP_INFO_HEADER_BYTES = 40


def _decodes_during_open(data: bytes) -> bool:
    return isinstance(data, (bytes, bytearray)) and bytes(data[:4]) in _ICO_PREFIXES


def _skips_the_resolver(host: str) -> bool:
    return bool(host) and (":" in host or host.replace(".", "").isdigit())


_FETCHABLE_SCHEMES = frozenset({"http", "https"})


_V6_SITE_LOCAL = ipaddress.IPv6Network("fec0::/10")

_V6_6TO4 = ipaddress.IPv6Network("2002::/16")

_V6_TEREDO = ipaddress.IPv6Network("2001::/32")

_V6_NAT64 = ipaddress.IPv6Network("64:ff9b::/96")


def _embedded_ipv4(ip: ipaddress.IPv6Address) -> tuple[ipaddress.IPv4Address, ...]:
    found: tuple[ipaddress.IPv4Address | None, ...] = ()
    mapped = ip.ipv4_mapped
    if mapped is not None:
        found = (mapped,)
    elif ip in _V6_6TO4:
        found = (ip.sixtofour,)
    elif ip in _V6_NAT64:
        found = (ipaddress.IPv4Address(int(ip) & 0xFFFFFFFF),)
    elif ip in _V6_TEREDO:
        found = ip.teredo or ()
    return tuple(carried for carried in found if carried is not None)


def _address_refusal(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> str | None:
    if isinstance(ip, ipaddress.IPv6Address):
        if ip in _V6_SITE_LOCAL:
            return "site-local"
        embedded = _embedded_ipv4(ip)
        if embedded:
            for carried in embedded:
                reason = _address_refusal(carried)
                if reason is not None:
                    return reason
            return None
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link-local"
    if ip.is_multicast:
        return "multicast"
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_private:
        return "private"
    if ip.is_reserved:
        return "reserved"
    if not ip.is_global:
        return "not globally routable"
    return None


def _ico_pixel_size(raw: bytes) -> tuple[int, int] | None:
    if len(raw) < 22:
        return None
    count = int.from_bytes(raw[4:6], "little")
    if count < 1 or count > _ICO_MAX_FRAMES:
        return None
    best: tuple[int, int] | None = None
    for index in range(count):
        entry = 6 + index * 16
        if entry + 16 > len(raw):
            return None
        offset = int.from_bytes(raw[entry + 12 : entry + 16], "little")
        frame = raw[offset : offset + _BMP_INFO_HEADER_BYTES]
        if frame.startswith(b"\x89PNG\r\n\x1a\n"):
            embedded = image_pixel_size(raw[offset : offset + 32])
            if embedded is None:
                return None
            side = embedded
        elif len(frame) >= 16 and int.from_bytes(frame[0:4], "little") == 12:
            width = int.from_bytes(frame[4:6], "little")
            height = int.from_bytes(frame[6:8], "little")
            side = (width, height // 2)
        elif len(frame) >= 16 and int.from_bytes(frame[0:4], "little") > 12:
            width = int.from_bytes(frame[4:8], "little")
            height = int.from_bytes(frame[8:12], "little")
            if frame[11] == 0xFF:
                height = 2**32 - height
            side = (width, height // 2)
        else:
            return None
        if best is None or side[0] * side[1] > best[0] * best[1]:
            best = side
    return best


def image_pixel_size(data: bytes) -> tuple[int, int] | None:
    if not isinstance(data, (bytes, bytearray)) or len(data) < 16:
        return None
    raw = bytes(data)

    if raw.startswith(b"\x89PNG\r\n\x1a\n") and raw[12:16] == b"IHDR":
        return (int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big"))

    if _decodes_during_open(raw):
        return _ico_pixel_size(raw)

    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        chunk = raw[12:16]
        if chunk == b"VP8X" and len(raw) >= 30:
            return (
                int.from_bytes(raw[24:27], "little") + 1,
                int.from_bytes(raw[27:30], "little") + 1,
            )
        if chunk == b"VP8 " and len(raw) >= 30:
            return (
                int.from_bytes(raw[26:28], "little") & 0x3FFF,
                int.from_bytes(raw[28:30], "little") & 0x3FFF,
            )
        if chunk == b"VP8L" and len(raw) >= 25:
            bits = int.from_bytes(raw[21:25], "little")
            return ((bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1)

    if raw.startswith(b"\xff\xd8\xff"):
        index = 2
        while index + 9 < len(raw):
            if raw[index] != 0xFF:
                index += 1
                continue
            marker = raw[index + 1]
            if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                index += 2
                continue
            length = int.from_bytes(raw[index + 2 : index + 4], "big")
            if length < 2:
                return None
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                          0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                return (
                    int.from_bytes(raw[index + 7 : index + 9], "big"),
                    int.from_bytes(raw[index + 5 : index + 7], "big"),
                )
            index += 2 + length
    return None


def _sniff_mime_from_prefix(data: bytes) -> str | None:
    if not isinstance(data, (bytes, bytearray)) or not data:
        return None
    raw = bytes(data)

    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    if raw.startswith((b"\x00\x00\x01\x00", b"\x00\x00\x02\x00")):
        return "image/x-icon"

    if len(raw) >= 12 and raw[4:8] == b"ftyp":
        return "video/mp4"
    if raw.startswith(b"\x1aE\xdf\xa3"):
        return "video/webm"
    if raw.startswith(b"OggS"):
        return "video/ogg"
    if raw.startswith(b"RIFF") and raw[8:12] == b"AVI ":
        return "video/x-msvideo"

    return None


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


# MultimodalHandler Class

class MultimodalHandler:
    """Manages multimodal content operations.

    This class encapsulates all file, image, audio, and video operations including:
    - File retrieval from Open WebUI storage
    - Remote URL downloads with SSRF protection (HTTPS-only by default)
    - File uploads to Open WebUI storage
    - Image processing and data URL handling
    - Chat file tracking

    Architecture:
    - aiohttp for HTTP downloads
    - Open WebUI database integration for file records
    - Base64 encoding for data URLs
    - MIME type detection
    - YouTube URL handling
    - SSRF protection with IP address validation (HTTP disabled by default)
    """

    def __init__(
        self,
        logger: logging.Logger,
        valves: Any,
        http_session: aiohttp.ClientSession | None = None,
        artifact_store: Any | None = None,
        emit_status_callback: Callable | None = None,
        file_gateway: OwuiFileGateway | None = None,
        valves_owner: Any | None = None,
    ):
        """Initialize the MultimodalHandler with dependencies from Pipe.

        Args:
            logger: Logger instance for diagnostics
            valves: Pipe.Valves instance with configuration
            http_session: Optional aiohttp session for remote downloads (can be set later)
            artifact_store: Optional ArtifactStore for file persistence
            emit_status_callback: Optional callback for status updates
            file_gateway: Optional OwuiFileGateway for authorized OWUI file I/O
        """
        self.logger = logger
        self._valves = valves
        self._valves_owner = valves_owner
        self._http_session = http_session
        self._artifact_store = artifact_store
        self._emit_status_callback = emit_status_callback
        self._file_gateway: OwuiFileGateway | None = file_gateway
        self._warned_missing_imaging: set[str] = set()
        self._warned_dns_failures: dict[str, float] = {}
        self._warned_blocked_hosts: dict[str, float] = {}
        self._vetted_http_session: aiohttp.ClientSession | None = None
        self._decode_pool: ThreadPoolExecutor | None = None
        self._transport_closed = False
        self._vetted_protection: bool | None = None
        self._vetted_loop: asyncio.AbstractEventLoop | None = None
        self._vetted_lock: asyncio.Lock | None = None
        self._vetted_lock_loop: asyncio.AbstractEventLoop | None = None

    @property
    def valves(self) -> Any:
        owner = self._valves_owner
        if owner is None:
            return self._valves
        live = getattr(owner, "valves", None)
        return self._valves if live is None else live

    def set_http_session(self, session: aiohttp.ClientSession | None) -> None:
        """Set or clear the HTTP session for remote downloads."""
        self._http_session = session

    def set_artifact_store(self, store: Any | None) -> None:
        """Set or clear the artifact store reference."""
        self._artifact_store = store


    @timed
    async def _download_remote_url(
        self,
        url: str,
        timeout_seconds: int | None = None
    ) -> dict[str, Any] | None:
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
            - https:// only by default
            - http:// allowed only when ALLOW_INSECURE_HTTP=True and host is allowlisted
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
        if not is_http_or_https_url(url):
            return None

        pinned = await self._prepare_pinned_request(url)
        if pinned is None:
            self.logger.error(
                "Remote download blocked by security policy (SSRF or HTTP disabled by default): %s",
                url,
            )
            return None
        request_url, pin_headers, pin_extensions = pinned

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
                stop=stop_after_attempt(max_retries + 1),
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

                    async with (
                        httpx.AsyncClient(timeout=timeout_seconds) as client,
                        client.stream(
                            "GET",
                            request_url,
                            headers=pin_headers or None,
                            extensions=pin_extensions or None,
                        ) as response,
                    ):
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
                            projected_size = len(payload) + len(chunk)
                            if projected_size > max_size_bytes:
                                size_mb = projected_size / (1024 * 1024)
                                self.logger.warning(
                                    f"Remote file {url} exceeds configured limit "
                                    f"({size_mb:.1f}MB > {effective_limit_mb}MB), aborting download."
                                )
                                return None
                            payload.extend(chunk)

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

        except Exception:
            elapsed = time.perf_counter() - start_time
            self.logger.exception(
                "Failed to download %s after %d attempt(s) in %.1fs", url, attempt, elapsed
            )
            return None

    async def _download_remote_url_streaming(
        self,
        url: str,
        dest_path: Path,
        *,
        chunk_size: int = 1024 * 1024,
        max_size_bytes: int | None = None,
        timeout_seconds: int | None = None,
        mime_allowlist: set[str] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, Any] | None:
        url = (url or "").strip()
        if not is_http_or_https_url(url):
            return None
        pinned = await self._prepare_pinned_request(url)
        if pinned is None:
            self.logger.error(
                "Remote streaming download blocked by security policy (SSRF or HTTP disabled by default): %s",
                url,
            )
            return None
        request_url, pin_headers, pin_extensions = pinned

        max_retries = self.valves.REMOTE_DOWNLOAD_MAX_RETRIES
        initial_delay = self.valves.REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS
        max_retry_time = self.valves.REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS

        if timeout_seconds is None:
            timeout_seconds = self.valves.HTTP_CONNECT_TIMEOUT_SECONDS
        if timeout_seconds is None:
            timeout_seconds = 60
        timeout_seconds = max(timeout_seconds, 60)

        effective_max = (
            max_size_bytes
            if max_size_bytes is not None
            else self._get_effective_remote_file_limit_mb() * 1024 * 1024
        )

        attempt = 0
        start_time = time.perf_counter()

        try:
            async for attempt_info in AsyncRetrying(
                retry=retry_if_exception_type(
                    (_RetryableHTTPStatusError, httpx.NetworkError, httpx.TimeoutException)
                ),
                stop=stop_after_attempt(max_retries + 1),
                wait=_RetryWait(wait_exponential(multiplier=initial_delay, min=initial_delay, max=max_retry_time)),
                reraise=True,
            ):
                with attempt_info:
                    attempt += 1
                    elapsed = time.perf_counter() - start_time
                    if attempt > 1 and elapsed > max_retry_time:
                        self.logger.warning(
                            f"Streaming download retry timeout exceeded for {url} after {elapsed:.1f}s"
                        )
                        return None
                    if attempt > 1:
                        self.logger.info(
                            f"Streaming retry attempt {attempt - 1}/{max_retries} for {url} after {elapsed:.1f}s"
                        )

                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    if dest_path.exists():
                        dest_path.unlink()

                    request_headers = dict(extra_headers) if extra_headers else {}
                    request_headers.update(pin_headers)
                    async with (
                        httpx.AsyncClient(timeout=timeout_seconds) as client,
                        client.stream(
                            "GET",
                            request_url,
                            headers=request_headers or None,
                            extensions=pin_extensions or None,
                        ) as response,
                    ):
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

                        content_length = response.headers.get("content-length")
                        if content_length:
                            try:
                                if int(content_length) > effective_max:
                                    self.logger.warning(
                                        "Remote streaming target %s exceeds configured limit per Content-Length "
                                        "(%s bytes > %s bytes); aborting.",
                                        url, content_length, effective_max,
                                    )
                                    return None
                            except ValueError:
                                pass

                        written = 0
                        sniff_buffer = bytearray()
                        sniffed_mime: str | None = mime_type
                        with dest_path.open("wb") as fh:
                            async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                                if not chunk:
                                    continue
                                projected = written + len(chunk)
                                if projected > effective_max:
                                    size_mb = projected / (1024 * 1024)
                                    limit_mb = effective_max / (1024 * 1024)
                                    self.logger.warning(
                                        f"Streaming download {url} exceeds limit "
                                        f"({size_mb:.1f}MB > {limit_mb:.1f}MB); aborting."
                                    )
                                    return None
                                if len(sniff_buffer) < 32:
                                    sniff_buffer.extend(chunk[: 32 - len(sniff_buffer)])
                                fh.write(chunk)
                                written = projected

                        if mime_allowlist is not None:
                            if not sniffed_mime or sniffed_mime in {"application/octet-stream", ""}:
                                sniffed_mime = _sniff_mime_from_prefix(bytes(sniff_buffer)) or sniffed_mime
                            if sniffed_mime not in mime_allowlist:
                                self.logger.warning(
                                    "Streaming download MIME %r not in allowlist %r; aborting.",
                                    sniffed_mime, sorted(mime_allowlist),
                                )
                                return None

                    if attempt > 1:
                        self.logger.info(
                            f"Successfully streamed {url} ({written:,} bytes) after {attempt} attempt(s)"
                        )

                    return {
                        "path": dest_path,
                        "mime_type": sniffed_mime or mime_type,
                        "url": url,
                        "size_bytes": written,
                    }
        except Exception:
            elapsed = time.perf_counter() - start_time
            self.logger.exception(
                "Failed streaming download of %s after %d attempt(s) in %.1fs",
                url,
                attempt,
                elapsed,
            )
            return None

    async def _is_safe_url(self, url: str, *, seconds: float = ADDRESS_CHECK_SECONDS) -> bool:
        """Whether this address may be fetched, decided inside a wall-clock budget.

        Args:
            url: URL to validate
            seconds: how long the check may take before the address counts as unsafe

        Returns:
            True if URL is safe (not targeting private networks) and allowed by HTTP policy

        The host is chosen by whoever wrote the request, so the nameserver it points at
        decides how long ``getaddrinfo`` blocks -- and callers run this while holding a
        deployment-wide slot. A budget that expires is a check that did not pass, so it
        answers False exactly as a failed resolution does. The worker thread cannot be
        interrupted and runs to completion; the budget frees the caller, not the thread.
        """
        try:
            resolved = await asyncio.wait_for(
                asyncio.to_thread(self._request_ips_blocking, url),
                timeout=max(0.0, seconds),
            )
        except TimeoutError:
            self.logger.warning(
                "Address check for %s did not finish within %.1fs; treating it as unsafe",
                url, seconds,
            )
            return False
        return resolved is not None

    def _parse_insecure_http_allowlist(self, raw: str) -> set[tuple[str, int | None]]:
        """Parse ALLOW_INSECURE_HTTP_HOSTS into host/port pairs (case-insensitive)."""
        if not isinstance(raw, str):
            return set()
        raw = raw.strip()
        if not raw:
            return set()
        allowed: set[tuple[str, int | None]] = set()
        for entry in raw.split(","):
            candidate = entry.strip()
            if not candidate:
                continue

            host = candidate
            port: int | None = None

            if candidate.startswith("[") and "]" in candidate:
                host = candidate[1:candidate.index("]")]
                remainder = candidate[candidate.index("]") + 1:]
                if remainder.startswith(":") and remainder[1:]:
                    if remainder[1:].isdigit():
                        port = int(remainder[1:])
                    else:
                        continue
            elif ":" in candidate:
                if candidate.count(":") == 1:
                    host_part, port_str = candidate.split(":", 1)
                    if port_str.isdigit():
                        port = int(port_str)
                    else:
                        continue
                    host = host_part
                else:
                    host = candidate

            host = host.strip().lower().rstrip(".")
            if not host:
                continue
            if port is not None and (port <= 0 or port > 65535):
                continue
            allowed.add((host, port))
        return allowed

    def _is_insecure_http_allowed(self, url: str) -> bool:
        """Return True when an http:// URL is explicitly allowed by valves."""
        try:
            parsed = _DialledURL(url)
        except ValueError:
            self.logger.warning("URL cannot be parsed: %s", url)
            return False
        scheme = (parsed.scheme or "").lower()
        if scheme not in _FETCHABLE_SCHEMES:
            self.logger.warning(
                "Blocked URL whose scheme is neither http nor https: %s", url
            )
            return False
        if scheme == "https":
            return True
        if not self.valves.ALLOW_INSECURE_HTTP:
            self.logger.warning(
                "Blocked insecure HTTP URL by default (HTTP disabled by default; "
                "set ALLOW_INSECURE_HTTP and ALLOW_INSECURE_HTTP_HOSTS to allow): %s",
                url,
            )
            return False
        allowlist = self._parse_insecure_http_allowlist(self.valves.ALLOW_INSECURE_HTTP_HOSTS)
        if not allowlist:
            self.logger.warning(
                "Blocked insecure HTTP URL; allowlist empty (HTTP disabled by default): %s",
                url,
            )
            return False
        host = (parsed.raw_host or "").lower().rstrip(".")
        if not host:
            self.logger.warning("HTTP URL has no hostname: %s", url)
            return False
        port = parsed.explicit_port or 80
        for allowed_host, allowed_port in allowlist:
            if host == allowed_host and (allowed_port is None or allowed_port == port):
                return True
        self.logger.warning(
            "Blocked insecure HTTP URL (host not allowlisted): %s (host=%s, port=%s)",
            url,
            host,
            port,
        )
        return False

    def _request_ips_blocking(self, url: str) -> list[str] | None:
        """Single SSRF gate (blocking): sequences the insecure-HTTP policy,
        the ENABLE_SSRF_PROTECTION valve, and address validation in one place
        so the pre-flight checks and the pinned download path cannot drift.

        Returns:
            None      — blocked (HTTP policy or address validation failed)
            []        — allowed WITHOUT an IP pin (SSRF protection disabled)
            non-empty — allowed; pin the connection to one of these IPs
        """
        if not self._is_insecure_http_allowed(url):
            return None
        if not self.valves.ENABLE_SSRF_PROTECTION:
            return []
        return self._resolve_validated_ips(url)

    def _latched_warn_level(
        self, latch: set[str] | dict[str, float], cause: str
    ) -> int:
        if len(latch) >= _DNS_WARN_LATCH_MAX_HOSTS:
            latch.clear()
        return warn_level(latch, cause, cooldown_s=_ADDRESS_WARN_COOLDOWN_SECONDS)

    def _parsed_target(self, url: str) -> tuple[str, int | None] | None:
        try:
            parsed = _DialledURL(url)
            port = parsed.port
        except ValueError:
            self.logger.warning("URL cannot be parsed: %s", url)
            return None
        host = parsed.raw_host
        if not host:
            self.logger.warning("URL has no hostname: %s", url)
            return None
        return (host, port)

    def _resolve_validated_ips(self, url: str) -> list[str] | None:
        """Resolve the URL's host and return every resolved IP (as strings) iff
        ALL of them are public addresses; return None if resolution fails or ANY
        address targets a private/reserved range.
        """
        target = self._parsed_target(url)
        if target is None:
            return None
        return self._validated_ips_for_host(target[0], target[1])

    def _validated_ips_for_host(
        self, host: str, port: int | None = None
    ) -> list[str] | None:
        try:
            ip_objects: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
            seen_ips: set[str] = set()

            def _record_ip(
                candidate: ipaddress.IPv4Address | ipaddress.IPv6Address,
            ) -> None:
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

            if literal_ip is None:
                try:
                    addrinfo = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
                except (socket.gaierror, UnicodeError):
                    self.logger.log(
                        self._latched_warn_level(self._warned_dns_failures, host),
                        "DNS resolution failed for: %s",
                        host,
                    )
                    return None
                except (OSError, TypeError, ValueError):  # pragma: no cover - defensive guard
                    self.logger.exception("Unexpected DNS error for %s", host)
                    return None

                for _, _, _, _, sockaddr in addrinfo:
                    if not sockaddr:
                        continue
                    ip_str = sockaddr[0]
                    try:
                        resolved_ip = ipaddress.ip_address(ip_str)
                    except ValueError:
                        self.logger.warning(f"Invalid IP address format: {ip_str}")
                        return None
                    _record_ip(resolved_ip)

            if not ip_objects:
                self.logger.warning(f"No IP addresses resolved for: {host}")
                return None

            for ip in ip_objects:
                reason = _address_refusal(ip)
                if reason is None:
                    continue

                self.logger.log(
                    self._latched_warn_level(
                        self._warned_blocked_hosts, f"{host}:{port}"
                    ),
                    "Blocked SSRF attempt to %s IP: %s port %s (%s)",
                    reason,
                    host,
                    port,
                    ip,
                )
                return None

            return [ip.compressed for ip in ip_objects]

        except Exception:
            self.logger.exception("Address validation failed for %s", host)
            return None

    def _build_pinned_request(
        self, url: str, ip: str
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        """Rewrite `url` to connect to the already-validated `ip` while keeping
        the original Host header and TLS SNI/cert verification for the hostname.

        Returns (request_url, headers, extensions) for an httpx call that cannot
        be DNS-rebound: the URL host is the IP literal (so httpx connects there
        with no further resolution), the Host header carries the original
        hostname (so virtual-hosted servers route correctly), and for https the
        sni_hostname extension makes the TLS handshake present/verify the
        original hostname (httpcore: server_hostname = sni_hostname or host).
        """
        parsed = _DialledURL(url)
        host = parsed.raw_host or ""
        port = parsed.explicit_port
        request_url = str(_DialledURL(url, encoded=True).with_host(ip))
        literal = f"[{host}]" if ":" in host else host
        host_header = f"{literal}:{port}" if port else literal
        headers = {"Host": host_header}
        extensions: dict[str, Any] = {}
        if (parsed.scheme or "").lower() == "https":
            extensions = {"sni_hostname": host}
        return (request_url, headers, extensions)

    async def _prepare_pinned_request(
        self, url: str
    ) -> tuple[str, dict[str, str], dict[str, Any]] | None:
        """Validate `url` against the SSRF guard and return (request_url,
        headers, extensions) for an IP-pinned httpx request, or None if blocked.

        When SSRF protection is disabled, the URL is returned unchanged with no
        pin (matching the legacy _is_safe_url fast-path). Otherwise the host is
        resolved+validated exactly once and the connection is pinned to a
        validated IP, so httpx cannot re-resolve to a rebound private address.
        """
        try:
            ips = await asyncio.wait_for(
                asyncio.to_thread(self._request_ips_blocking, url),
                timeout=ADDRESS_CHECK_SECONDS,
            )
        except TimeoutError:
            self.logger.warning(
                "Address check for %s did not finish within %.1fs; treating it as unsafe",
                url, ADDRESS_CHECK_SECONDS,
            )
            return None
        if ips is None:
            return None
        if not ips:
            return (url, {}, {})
        return self._build_pinned_request(url, ips[0])

    def _vetted_loop_is_stale(self) -> bool:
        bound = self._vetted_loop
        if bound is None:
            return False
        if bound.is_closed():
            return True
        try:
            return asyncio.get_running_loop() is not bound
        except RuntimeError:
            return False

    async def _retire_vetted_session(self) -> None:
        session = self._vetted_http_session
        bound = self._vetted_loop
        self._vetted_http_session = None
        self._vetted_loop = None
        if session is None or session.closed:
            return
        if bound is not None and not bound.is_closed():
            try:
                if bound is not asyncio.get_running_loop():
                    return
            except RuntimeError:
                return
        await session.close()

    def _vetted_transport_lock(self) -> asyncio.Lock:
        running = asyncio.get_running_loop()
        if self._vetted_lock is None or self._vetted_lock_loop is not running:
            self._vetted_lock = asyncio.Lock()
            self._vetted_lock_loop = running
        return self._vetted_lock

    async def _vetted_session(self, url: str = "") -> aiohttp.ClientSession:
        async with self._vetted_transport_lock():
            if self._transport_closed:
                raise UnfetchableAddress(url)
            protection = bool(self.valves.ENABLE_SSRF_PROTECTION)
            session = self._vetted_http_session
            if (
                session is not None
                and not session.closed
                and not self._vetted_loop_is_stale()
                and (self._vetted_protection is None or protection == self._vetted_protection)
            ):
                self._vetted_protection = protection
                return session
            await self._retire_vetted_session()
            connector = aiohttp.TCPConnector(
                resolver=_VettedResolver(self, protection),
                limit=_VETTED_CONNECTION_LIMIT,
                limit_per_host=_VETTED_CONNECTION_LIMIT_PER_HOST,
                ttl_dns_cache=_VETTED_DNS_CACHE_SECONDS,
            )
            session = aiohttp.ClientSession(connector=connector)
            self._vetted_http_session = session
            self._vetted_protection = protection
            self._vetted_loop = asyncio.get_running_loop()
            return session

    def _hop_is_refused(self, url: str) -> bool:
        target = self._parsed_target(url)
        if target is None:
            return True
        if not self._is_insecure_http_allowed(url):
            return True
        if not self.valves.ENABLE_SSRF_PROTECTION:
            return False
        host = target[0]
        if not _skips_the_resolver(host):
            return False
        try:
            ipaddress.ip_address(host)
        except ValueError:
            self.logger.warning("Address-shaped host is not an address: %s", url)
            return True
        return self._validated_ips_for_host(host, target[1]) is None

    def _joined_hop(self, target: str, location: str) -> str | None:
        try:
            return urljoin(target, location)
        except ValueError:
            self.logger.warning("Redirect target cannot be parsed: %s", location)
            return None

    @asynccontextmanager
    async def _vetted_get(
        self,
        url: str,
        *,
        total_seconds: float,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        caller_headers = dict(kwargs.pop("headers", None) or {})
        request_timeout = kwargs.pop("timeout", None) or aiohttp.ClientTimeout(
            total=total_seconds
        )
        target = url
        async with asyncio.timeout(total_seconds):
            for _hop in range(_MAX_VETTED_REDIRECTS + 1):
                if self._hop_is_refused(target):
                    raise UnfetchableAddress(target)
                session = await self._vetted_session(target)
                connected = False
                try:
                    async with session.get(
                        target,
                        headers=caller_headers,
                        allow_redirects=False,
                        timeout=request_timeout,
                        **kwargs,
                    ) as response:
                        connected = True
                        location = response.headers.get("Location")
                        if response.status in _REDIRECT_STATUSES and location:
                            joined = self._joined_hop(target, location)
                            if joined is None:
                                raise UnfetchableAddress(location)
                            target = joined
                            continue
                        yield response
                        return
                except aiohttp.ClientConnectorError as exc:
                    if connected or not isinstance(exc.os_error, _AddressRefused):
                        raise
                    raise UnfetchableAddress(target) from exc
        raise UnfetchableAddress(target)

    async def _run_decode(self, func: Callable[..., Any], *args: Any) -> Any:
        if self._transport_closed:
            raise RuntimeError("the model-icon decode pool is closed")
        pool = self._decode_pool
        if pool is None:
            pool = ThreadPoolExecutor(
                max_workers=_ICON_DECODE_WORKERS, thread_name_prefix="or-icon-decode"
            )
            self._decode_pool = pool
        return await asyncio.get_running_loop().run_in_executor(pool, func, *args)

    def transport_session_state(self) -> str:
        if self._transport_closed:
            return "closed"
        session = self._vetted_http_session
        if session is None or self._vetted_loop_is_stale():
            return "none"
        return "closed" if session.closed else "active"

    async def aclose(self) -> None:
        async with self._vetted_transport_lock():
            self._transport_closed = True
            await self._retire_vetted_session()
        pool = self._decode_pool
        self._decode_pool = None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)

    def _is_youtube_url(self, url: str | None) -> bool:
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
            - HTTP is disabled by default; http:// URLs require ALLOW_INSECURE_HTTP allowlisting

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

    def _get_effective_remote_file_limit_mb(self) -> int:
        """Return the active remote download limit, honoring RAG constraints.

        Returns:
            Effective file size limit in MB
        """
        base_limit_mb = self.valves.REMOTE_FILE_MAX_SIZE_MB
        rag_enabled, rag_limit_mb = _read_rag_file_constraints()
        if not rag_enabled or rag_limit_mb is None:
            return base_limit_mb

        if base_limit_mb > rag_limit_mb:
            return rag_limit_mb

        if (
            base_limit_mb == _REMOTE_FILE_MAX_SIZE_DEFAULT_MB
            and rag_limit_mb > base_limit_mb
        ):
            return rag_limit_mb
        return base_limit_mb


    @timed
    async def _fetch_image_as_data_url(self, url: str) -> str | None:
        """Fetch image from URL and convert to data URL.

        Args:
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
            async with self._vetted_get(
                url, total_seconds=_ICON_FETCH_TIMEOUT_SECONDS
            ) as resp:
                resp.raise_for_status()
                capped = await _capped_body(resp, _MAX_MODEL_PROFILE_IMAGE_BYTES)
                if capped is None:
                    self.logger.debug(
                        "Skipping model icon over %d bytes (url=%s)",
                        _MAX_MODEL_PROFILE_IMAGE_BYTES,
                        url,
                    )
                    return None
                data = capped
                content_type = resp.headers.get("Content-Type")
        except UnfetchableAddress as exc:
            self.logger.debug("Refusing model icon address: %s", exc)
            return None
        except aiohttp.ClientResponseError as exc:
            self.logger.debug("Failed to download model icon (url=%s): %s", url, exc)
            return None
        except Exception as exc:
            self.logger.debug(
                "Failed to download model icon (url=%s): %s", url, exc, exc_info=True
            )
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
                _level = warn_level(self._warned_missing_imaging, 'cairosvg')
                self.logger.log(
                    _level,
                    "CairoSVG is not installed, so no SVG model icon can be converted "
                    "while UPDATE_MODEL_IMAGES is enabled: %s",
                    exc,
                    exc_info=True,
                )
                return None

            try:
                png_bytes = await self._run_decode(
                    lambda: cairosvg.svg2png(
                        bytestring=data,
                        output_width=250,
                        output_height=250,
                    )
                )
            except Exception as exc:
                self.logger.debug(
                    "Failed to rasterize SVG model icon (url=%s): %s", url, exc, exc_info=True
                )
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
            png_bytes = await self._run_decode(_icon_png_bytes, data)
        except ImportError as exc:
            _level = warn_level(self._warned_missing_imaging, 'pillow')
            self.logger.log(
                _level,
                "Pillow is not installed, so no model icon can be converted while "
                "UPDATE_MODEL_IMAGES is enabled: %s",
                exc,
                exc_info=True,
            )
            return None
        except _IconPixelBudgetExceeded:
            self.logger.debug(
                "Skipping model icon not provably within the %d pixel budget "
                "(%d bytes, url=%s)",
                _MAX_MODEL_PROFILE_IMAGE_PIXELS,
                len(data),
                url,
            )
            return None
        except Exception as exc:
            self.logger.debug(
                "Failed to convert model icon to PNG (url=%s): %s", url, exc, exc_info=True
            )
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
    async def _fetch_maker_profile_image_url(self, maker_id: str) -> str | None:
        """Fetch OpenRouter maker profile image URL from their page.

        Args:
            maker_id: Maker identifier

        Returns:
            Profile image URL or None if not found
        """
        maker_id = (maker_id or "").strip()
        if not maker_id:
            return None
        url = f"{_OPENROUTER_SITE_URL}/{quote(maker_id)}"
        try:
            async with self._vetted_get(
                url, total_seconds=_MAKER_PAGE_FETCH_TIMEOUT_SECONDS
            ) as resp:
                resp.raise_for_status()
                capped = await _capped_body(resp, _MAKER_PAGE_MAX_BYTES)
        except Exception as exc:
            self.logger.debug(
                "OpenRouter maker page fetch failed (maker=%s): %s", maker_id, exc, exc_info=True
            )
            return None

        if capped is None:
            self.logger.warning(
                "OpenRouter maker page exceeds %d bytes (maker=%s); treating as empty.",
                _MAKER_PAGE_MAX_BYTES,
                maker_id,
            )
            return None
        return _extract_openrouter_og_image(capped.decode("utf-8", errors="replace"))


    def _parse_data_url(self, data_url: str) -> dict[str, Any] | None:
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
            - Parsing failures are caught and logged; an unconfigured file gateway raises
            - Non-data URLs return None immediately

        Example:
            >>> result = self._parse_data_url(
            ...     "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
            ... )
            >>> if result:
            ...     print(f"MIME: {result['mime_type']}")
            ...     print(f"Size: {len(result['data'])} bytes")
        """
        if self._file_gateway is None:
            raise RuntimeError("File gateway is not configured for data URL validation")
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

            if not self._file_gateway.validate_base64_size(b64_data):
                return None

            file_data = base64.b64decode(b64_data)

            return {
                "data": file_data,
                "mime_type": mime_type,
                "b64": b64_data
            }
        except (AttributeError, TypeError, ValueError):
            self.logger.exception("Failed to parse data URL")
            return None
