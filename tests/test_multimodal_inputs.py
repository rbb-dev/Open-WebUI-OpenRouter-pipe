"""
Comprehensive tests for file, image, and audio input handling.

Tests cover:
- Image URL and base64 transformations
- File URL and base64 transformations
- Audio format conversions
- Error handling and fallback scenarios
- OWUI storage integration
- Remote URL downloading
- Data URL parsing
"""
from __future__ import annotations

import asyncio
import base64
import datetime
import os
import socket
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

import open_webui_openrouter_pipe.open_webui_openrouter_pipe as pipe_module
from open_webui_openrouter_pipe.open_webui_openrouter_pipe import (
    ModelFamily,
    Pipe,
    StatusMessages,
)


async def _transform_single_block(
    pipe_instance: Pipe,
    block: dict,
    mock_request,
    mock_user,
) -> dict | None:
    """Helper to transform a single user message block."""
    messages = [
        {
            "role": "user",
            "content": [block],
        }
    ]
    transformed = await pipe_instance.transform_messages_to_input(
        messages,
        __request__=mock_request,
        user_obj=mock_user,
        event_emitter=None,
    )
    if not transformed:
        return None
    content = transformed[0].get("content")
    if not isinstance(content, list) or not content:
        return None
    first = content[0]
    return first if isinstance(first, dict) else None


def _make_stream_context(response=None, error=None):
    class _StreamContext:
        async def __aenter__(self):
            if error:
                raise error
            return response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    return _StreamContext()


def _set_aiter_bytes(mock_response, chunks):
    async def _iterator():
        for chunk in chunks:
            yield chunk

    mock_response.aiter_bytes = _iterator


# ─────────────────────────────────────────────────────────────────────────────
# Helper Method Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestDataURLParsing:
    """Tests for _parse_data_url helper method."""

    def test_parse_valid_image_data_url(self, pipe_instance, sample_image_base64):
        """Should parse valid image data URL correctly."""
        data_url = f"data:image/png;base64,{sample_image_base64}"
        result = pipe_instance._parse_data_url(data_url)

        assert result is not None
        assert result["mime_type"] == "image/png"
        assert result["b64"] == sample_image_base64
        assert isinstance(result["data"], bytes)

    def test_parse_normalizes_image_jpg_to_jpeg(self, pipe_instance):
        """Should normalize image/jpg to image/jpeg."""
        data_url = "data:image/jpg;base64,AAAA"
        result = pipe_instance._parse_data_url(data_url)

        assert result is not None
        assert result["mime_type"] == "image/jpeg"

    def test_parse_audio_data_url(self, pipe_instance, sample_audio_base64):
        """Should parse audio data URL correctly."""
        data_url = f"data:audio/mp3;base64,{sample_audio_base64}"
        result = pipe_instance._parse_data_url(data_url)

        assert result is not None
        assert result["mime_type"] == "audio/mp3"
        assert result["b64"] == sample_audio_base64

    def test_parse_invalid_data_url_returns_none(self, pipe_instance):
        """Should return None for invalid data URLs."""
        assert pipe_instance._parse_data_url("not a data url") is None
        assert pipe_instance._parse_data_url("data:image/png,missing_base64") is None
        assert pipe_instance._parse_data_url("") is None
        assert pipe_instance._parse_data_url(None) is None

    def test_parse_invalid_base64_returns_none(self, pipe_instance):
        """Should return None for invalid base64 data."""
        data_url = "data:image/png;base64,INVALID!!!BASE64"
        result = pipe_instance._parse_data_url(data_url)
        assert result is None


class TestImageTransformations:
    """Tests focused on user image block transformations."""

    @pytest.mark.asyncio
    async def test_remote_images_rehosted_and_inlined(
        self, pipe_instance, mock_request, mock_user, sample_image_base64
    ):
        """Remote images should be re-hosted and then inlined for provider delivery."""

        pipe_instance._download_remote_url = AsyncMock(
            return_value={
                "data": base64.b64decode(sample_image_base64),
                "mime_type": "image/png",
                "url": "https://example.com/cat.png",
            }
        )
        pipe_instance._upload_to_owui_storage = AsyncMock(
            return_value="cat123"
        )
        pipe_instance._inline_owui_file_id = AsyncMock(
            return_value="data:image/png;base64,INLINED=="
        )

        block = {
            "type": "image_url",
            "image_url": "https://example.com/cat.png",
        }

        transformed = await _transform_single_block(
            pipe_instance,
            block,
            mock_request,
            mock_user,
        )

        assert transformed is not None
        assert transformed["type"] == "input_image"
        assert transformed["image_url"] == "data:image/png;base64,INLINED=="
        pipe_instance._download_remote_url.assert_awaited_once()
        pipe_instance._upload_to_owui_storage.assert_awaited_once()
        pipe_instance._inline_owui_file_id.assert_awaited_once()
        inline_args = pipe_instance._inline_owui_file_id.await_args
        assert inline_args is not None
        assert inline_args.args[0] == "cat123"


class TestFileEncoding:
    """Tests covering file path base64 encoding helpers."""

    @pytest.mark.asyncio
    async def test_encode_file_path_base64_matches_standard_encoder(
        self, pipe_instance, tmp_path
    ):
        """Chunked base64 encoding should match the standard encoder output."""

        data = os.urandom(100_000)  # ensure multiple read iterations with remainder bytes
        file_path = tmp_path / "blob.bin"
        file_path.write_bytes(data)

        expected = base64.b64encode(data).decode("ascii")
        result = await pipe_instance._encode_file_path_base64(
            file_path,
            chunk_size=64 * 1024,
            max_bytes=len(data) + 1024,
        )

        assert result == expected


class TestRemoteURLDownloading:
    """Tests for _download_remote_url helper method."""

    @pytest.mark.asyncio
    async def test_download_successful(self, pipe_instance):
        """Should download remote file successfully."""
        pipe_instance._is_safe_url = AsyncMock(return_value=True)
        test_content = b"fake image data"

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_response.raise_for_status = Mock()
            _set_aiter_bytes(mock_response, [test_content])
            client_ctx = mock_client.return_value.__aenter__.return_value
            client_ctx.stream = Mock(return_value=_make_stream_context(response=mock_response))

            result = await pipe_instance._download_remote_url(
                "https://example.com/image.jpg"
            )

            assert result is not None
            assert result["data"] == test_content
            assert result["mime_type"] == "image/jpeg"
            assert result["url"] == "https://example.com/image.jpg"

    @pytest.mark.asyncio
    async def test_download_normalizes_mime_type(self, pipe_instance):
        """Should normalize image/jpg to image/jpeg."""
        pipe_instance._is_safe_url = AsyncMock(return_value=True)
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.headers = {"content-type": "image/jpg; charset=utf-8"}
            mock_response.raise_for_status = Mock()
            _set_aiter_bytes(mock_response, [b"data"])
            client_ctx = mock_client.return_value.__aenter__.return_value
            client_ctx.stream = Mock(return_value=_make_stream_context(response=mock_response))

            result = await pipe_instance._download_remote_url(
                "https://example.com/image.jpg"
            )

            assert result["mime_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_download_rejects_files_over_default_limit(self, pipe_instance):
        """Should reject files larger than the configured limit (default 50MB)."""
        pipe_instance._is_safe_url = AsyncMock(return_value=True)
        pipe_instance.valves.REMOTE_FILE_MAX_SIZE_MB = 1
        limit_bytes = pipe_instance._get_effective_remote_file_limit_mb() * 1024 * 1024

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.headers = {
                "content-type": "image/jpeg",
                "content-length": str(limit_bytes + 1),
            }
            mock_response.raise_for_status = Mock()
            _set_aiter_bytes(mock_response, [b"x"])
            client_ctx = mock_client.return_value.__aenter__.return_value
            client_ctx.stream = Mock(return_value=_make_stream_context(response=mock_response))

            result = await pipe_instance._download_remote_url(
                "https://example.com/huge.jpg"
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_download_invalid_url_returns_none(self, pipe_instance):
        """Should return None for non-HTTP URLs."""
        assert await pipe_instance._download_remote_url("file:///local/path") is None
        assert await pipe_instance._download_remote_url("ftp://example.com/file") is None
        assert await pipe_instance._download_remote_url("") is None

    @pytest.mark.asyncio
    async def test_download_network_error_returns_none(self, pipe_instance):
        """Should retry on network errors and return None when exhausted."""
        pipe_instance._is_safe_url = AsyncMock(return_value=True)
        pipe_instance.valves.REMOTE_DOWNLOAD_MAX_RETRIES = 1
        pipe_instance.valves.REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS = 0
        pipe_instance.valves.REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS = 5
        with patch("httpx.AsyncClient") as mock_client:
            client_ctx = mock_client.return_value.__aenter__.return_value
            client_ctx.stream = Mock(side_effect=httpx.NetworkError("Network error"))

            result = await pipe_instance._download_remote_url(
                "https://example.com/image.jpg"
            )

            assert result is None
            assert client_ctx.stream.call_count == 2

    @pytest.mark.asyncio
    async def test_download_does_not_retry_on_client_errors(self, pipe_instance):
        """Should not retry on non-429 HTTP 4xx errors."""
        pipe_instance._is_safe_url = AsyncMock(return_value=True)
        url = "https://example.com/forbidden.png"
        request = httpx.Request("GET", url)
        response = httpx.Response(status_code=403, request=request)
        error = httpx.HTTPStatusError("Forbidden", request=request, response=response)

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_response.raise_for_status = Mock(side_effect=error)
            _set_aiter_bytes(mock_response, [b"x"])
            client_ctx = mock_client.return_value.__aenter__.return_value
            client_ctx.stream = Mock(return_value=_make_stream_context(response=mock_response))

            result = await pipe_instance._download_remote_url(url)

            assert result is None
            assert client_ctx.stream.call_count == 1

    @pytest.mark.asyncio
    async def test_download_blocks_unsafe_url(self, pipe_instance):
        """SSRF guard should abort before making any HTTP request."""
        pipe_instance._is_safe_url = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient") as mock_client:
            result = await pipe_instance._download_remote_url("https://example.com/image.jpg")
            assert result is None
            mock_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_download_retries_on_429_then_succeeds(self, pipe_instance):
        """HTTP 429 should trigger a retry and succeed on a later attempt."""
        pipe_instance._is_safe_url = AsyncMock(return_value=True)
        pipe_instance.valves.REMOTE_DOWNLOAD_MAX_RETRIES = 1
        pipe_instance.valves.REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS = 0
        pipe_instance.valves.REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS = 5

        url = "https://example.com/limited.png"
        request = httpx.Request("GET", url)
        limited_response = httpx.Response(status_code=429, headers={"Retry-After": "0"}, request=request)
        limited_error = httpx.HTTPStatusError("Too Many Requests", request=request, response=limited_response)

        with patch("httpx.AsyncClient") as mock_client:
            first = Mock()
            first.headers = {"content-type": "image/png"}
            first.raise_for_status = Mock(side_effect=limited_error)
            _set_aiter_bytes(first, [b"ignored"])

            second = Mock()
            second.headers = {"content-type": "image/png"}
            second.raise_for_status = Mock()
            _set_aiter_bytes(second, [b"ok"])

            client_ctx = mock_client.return_value.__aenter__.return_value
            client_ctx.stream = Mock(
                side_effect=[
                    _make_stream_context(response=first),
                    _make_stream_context(response=second),
                ]
            )

            result = await pipe_instance._download_remote_url(url)

            assert result is not None
            assert result["data"] == b"ok"
            assert result["mime_type"] == "image/png"
            assert client_ctx.stream.call_count == 2


class TestSSRFIPv6Validation:
    """Ensure _is_safe_url handles IPv6 and mixed DNS responses."""

    pytestmark = pytest.mark.asyncio

    async def test_blocks_private_ipv6_literal(self, pipe_instance):
        """IPv6 literals in unique-local ranges should be rejected."""
        assert await pipe_instance._is_safe_url("http://[fd00::1]/") is False

    async def test_allows_global_ipv6_literal(self, pipe_instance):
        """Public IPv6 literals should be considered safe."""
        assert await pipe_instance._is_safe_url("https://[2001:4860:4860::8888]/foo")

    async def test_blocks_domain_with_private_ipv6_record(self, pipe_instance, monkeypatch):
        """Hosts resolving to any private IPv6 addresses are rejected."""

        def fake_getaddrinfo(host, *args, **kwargs):
            return [
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("fd00::abcd", 0, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0)),
            ]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        assert await pipe_instance._is_safe_url("https://example.com/resource") is False

    async def test_allows_domain_with_public_ips_only(self, pipe_instance, monkeypatch):
        """Hosts resolving exclusively to public IPv4/IPv6 addresses pass the guard."""

        def fake_getaddrinfo(host, *args, **kwargs):
            return [
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("2001:4860:4860::8888", 0, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0)),
            ]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        assert await pipe_instance._is_safe_url("https://example.com/resource")


class TestRemoteFileLimitResolution:
    """Tests for resolving the effective remote download size limit."""

    def _prepare_config(self, monkeypatch):
        config = sys.modules["open_webui.config"]
        monkeypatch.setattr(
            pipe_module,
            "_OPEN_WEBUI_CONFIG_MODULE",
            config,
            raising=False,
        )
        return config

    def test_uses_valve_when_rag_disabled(self, pipe_instance, monkeypatch):
        config = self._prepare_config(monkeypatch)
        monkeypatch.setattr(config.BYPASS_EMBEDDING_AND_RETRIEVAL, "value", True, raising=False)
        monkeypatch.setattr(config.RAG_FILE_MAX_SIZE, "value", 200, raising=False)
        pipe_instance.valves.REMOTE_FILE_MAX_SIZE_MB = 60

        assert pipe_instance._get_effective_remote_file_limit_mb() == 60

    def test_caps_to_rag_when_smaller(self, pipe_instance, monkeypatch):
        config = self._prepare_config(monkeypatch)
        monkeypatch.setattr(config.BYPASS_EMBEDDING_AND_RETRIEVAL, "value", False, raising=False)
        monkeypatch.setattr(config.RAG_FILE_MAX_SIZE, "value", 25, raising=False)
        pipe_instance.valves.REMOTE_FILE_MAX_SIZE_MB = 50

        assert pipe_instance._get_effective_remote_file_limit_mb() == 25

    def test_adopts_rag_when_default_and_larger(self, pipe_instance, monkeypatch):
        config = self._prepare_config(monkeypatch)
        monkeypatch.setattr(config.BYPASS_EMBEDDING_AND_RETRIEVAL, "value", False, raising=False)
        monkeypatch.setattr(config.RAG_FILE_MAX_SIZE, "value", 120, raising=False)
        pipe_instance.valves.REMOTE_FILE_MAX_SIZE_MB = 50

        assert pipe_instance._get_effective_remote_file_limit_mb() == 120

    def test_respects_custom_limit_when_lower_than_rag(self, pipe_instance, monkeypatch):
        config = self._prepare_config(monkeypatch)
        monkeypatch.setattr(config.BYPASS_EMBEDDING_AND_RETRIEVAL, "value", False, raising=False)
        monkeypatch.setattr(config.RAG_FILE_MAX_SIZE, "value", 150, raising=False)
        pipe_instance.valves.REMOTE_FILE_MAX_SIZE_MB = 80

        assert pipe_instance._get_effective_remote_file_limit_mb() == 80

    def test_falls_back_to_file_max_size_when_rag_missing(self, pipe_instance, monkeypatch):
        config = self._prepare_config(monkeypatch)
        monkeypatch.setattr(config.BYPASS_EMBEDDING_AND_RETRIEVAL, "value", False, raising=False)
        monkeypatch.setattr(config.RAG_FILE_MAX_SIZE, "value", None, raising=False)
        monkeypatch.setattr(config.FILE_MAX_SIZE, "value", 180, raising=False)
        pipe_instance.valves.REMOTE_FILE_MAX_SIZE_MB = 50

        assert pipe_instance._get_effective_remote_file_limit_mb() == 180


class TestRetryHelpers:
    """Unit tests for retry helper utilities."""

    def test_retry_after_seconds_parses_numeric(self):
        assert pipe_module._retry_after_seconds("5") == 5.0
        assert pipe_module._retry_after_seconds("0") == 0.0
        assert pipe_module._retry_after_seconds("") is None

    def test_retry_after_seconds_parses_http_date(self):
        future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=3)
        header = future.strftime("%a, %d %b %Y %H:%M:%S GMT")
        delay = pipe_module._retry_after_seconds(header)
        assert delay is not None and delay <= 4.0

    def test_retry_wait_honors_retry_after(self):
        base_wait = lambda state: 1.0
        wait = pipe_module._RetryWait(base_wait)

        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(status_code=429, request=request)
        error = httpx.HTTPStatusError("Too many requests", request=request, response=response)
        retry_exc = pipe_module._RetryableHTTPStatusError(error, retry_after=5.0)

        state = SimpleNamespace(outcome=Mock())
        state.outcome.exception = Mock(return_value=retry_exc)

        assert wait(state) == 5.0

    def test_classify_retryable_http_error_identifies_425(self):
        url = "https://example.com/file"
        request = httpx.Request("GET", url)
        response = httpx.Response(status_code=425, headers={"Retry-After": "2"}, request=request)
        error = httpx.HTTPStatusError("Too Early", request=request, response=response)
        retryable, retry_after = pipe_module._classify_retryable_http_error(error)
        assert retryable is True
        assert retry_after == 2.0

    def test_classify_retryable_http_error_rejects_403(self):
        url = "https://example.com/file"
        request = httpx.Request("GET", url)
        response = httpx.Response(status_code=403, request=request)
        error = httpx.HTTPStatusError("Forbidden", request=request, response=response)
        retryable, retry_after = pipe_module._classify_retryable_http_error(error)
        assert retryable is False
        assert retry_after is None


class TestStorageContext:
    """Tests for storage context resolution."""

    @pytest.mark.asyncio
    async def test_resolve_storage_context_prefers_existing_user(
        self,
        pipe_instance,
        mock_request,
        mock_user,
    ):
        request, user = await pipe_instance._resolve_storage_context(
            mock_request,
            mock_user,
        )
        assert request is mock_request
        assert user is mock_user

    @pytest.mark.asyncio
    async def test_resolve_storage_context_uses_fallback_user(
        self,
        pipe_instance,
        mock_request,
    ):
        fallback_user = Mock()
        fallback_user.email = "fallback@example.com"
        pipe_instance._ensure_storage_user = AsyncMock(return_value=fallback_user)

        request, user = await pipe_instance._resolve_storage_context(mock_request, None)
        assert request is mock_request
        assert user is fallback_user
        pipe_instance._ensure_storage_user.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_resolve_storage_context_without_request(
        self,
        pipe_instance,
    ):
        pipe_instance._ensure_storage_user = AsyncMock()
        request, user = await pipe_instance._resolve_storage_context(None, None)
        assert request is None
        assert user is None
        pipe_instance._ensure_storage_user.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Image Transformer Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestImageTransformer:
    """Tests for _to_input_image transformer function."""

    @pytest.mark.asyncio
    async def test_image_data_url_saved_to_storage(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        sample_image_base64,
        monkeypatch,
    ):
        """Base64 images should be re-hosted and emit status updates."""
        stored_id = "img123"
        upload_mock = AsyncMock(return_value=stored_id)
        inline_mock = AsyncMock(return_value="data:image/png;base64,INLINE")
        status_mock = AsyncMock()
        monkeypatch.setattr(pipe_instance, "_upload_to_owui_storage", upload_mock)
        monkeypatch.setattr(pipe_instance, "_inline_owui_file_id", inline_mock)
        monkeypatch.setattr(pipe_instance, "_emit_status", status_mock)

        block = {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{sample_image_base64}", "detail": "high"},
        }
        image_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert image_block is not None
        assert image_block["image_url"] == "data:image/png;base64,INLINE"
        assert image_block["detail"] == "high"
        upload_mock.assert_awaited()
        inline_mock.assert_awaited()
        status_mock.assert_awaited_with(
            None,
            StatusMessages.IMAGE_BASE64_SAVED,
            done=False,
        )

    @pytest.mark.asyncio
    async def test_image_remote_url_downloaded_and_saved(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        monkeypatch,
    ):
        """Remote URLs are downloaded, uploaded, and statuses emitted."""
        remote_url = "https://example.com/photo.png"
        stored_id = "remote-img"
        download_mock = AsyncMock(
            return_value={"data": b"img", "mime_type": "image/png", "url": remote_url}
        )
        upload_mock = AsyncMock(return_value=stored_id)
        inline_mock = AsyncMock(return_value="data:image/png;base64,INLINE")
        status_mock = AsyncMock()
        monkeypatch.setattr(pipe_instance, "_download_remote_url", download_mock)
        monkeypatch.setattr(pipe_instance, "_upload_to_owui_storage", upload_mock)
        monkeypatch.setattr(pipe_instance, "_inline_owui_file_id", inline_mock)
        monkeypatch.setattr(pipe_instance, "_emit_status", status_mock)

        block = {"type": "image_url", "image_url": {"url": remote_url, "detail": "auto"}}
        image_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert image_block is not None
        assert image_block["image_url"] == "data:image/png;base64,INLINE"
        assert image_block["detail"] == "auto"
        download_mock.assert_awaited_once_with(remote_url)
        upload_mock.assert_awaited()
        inline_mock.assert_awaited()
        status_mock.assert_any_await(
            None,
            StatusMessages.IMAGE_REMOTE_SAVED,
            done=False,
        )

    @pytest.mark.asyncio
    async def test_image_detail_level_preserved(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        monkeypatch,
    ):
        """Explicit detail selection should survive transformation."""
        async def fake_inline(file_id, chunk_size, max_bytes):  # type: ignore[no-untyped-def]
            assert file_id == "abc"
            return "data:image/png;base64,abc"

        monkeypatch.setattr(pipe_instance, "_inline_owui_file_id", fake_inline)
        block = {"type": "image_url", "image_url": {"url": "/api/v1/files/abc", "detail": "low"}}
        image_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert image_block is not None
        assert image_block["detail"] == "low"
        assert image_block["image_url"] == "data:image/png;base64,abc"

    @pytest.mark.asyncio
    async def test_internal_file_url_inlined_to_data_url(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        tmp_path,
        monkeypatch,
    ):
        """Internal OWUI URLs should be converted to data URLs to satisfy providers."""
        file_path = tmp_path / "inline.bin"
        file_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        class _FileRecord:
            path = str(file_path)
            meta = {"content_type": "image/png"}

        async def fake_get_file(_file_id):
            return _FileRecord()

        monkeypatch.setattr(pipe_instance, "_get_file_by_id", fake_get_file)
        block = {"type": "image_url", "image_url": {"url": "/api/v1/files/inline/content"}}
        image_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert image_block is not None
        assert image_block["image_url"].startswith("data:image/png;base64,")

    @pytest.mark.asyncio
    async def test_internal_file_url_missing_is_dropped(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        monkeypatch,
    ):
        """Missing OWUI file ids should not be sent upstream as internal URLs."""

        async def fake_inline(_file_id, chunk_size, max_bytes):  # type: ignore[no-untyped-def]
            return None

        monkeypatch.setattr(pipe_instance, "_inline_owui_file_id", fake_inline)
        block = {"type": "image_url", "image_url": {"url": "/api/v1/files/missing/content"}}
        image_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert image_block is None

    @pytest.mark.asyncio
    async def test_image_error_returns_empty_block(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        monkeypatch,
    ):
        """Errors while processing images should not leak exceptions."""
        boom = RuntimeError("boom")
        monkeypatch.setattr(pipe_instance, "_upload_to_owui_storage", AsyncMock(side_effect=boom))
        block = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        image_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert image_block is not None
        assert image_block["image_url"] == "data:image/png;base64,AAAA"
        assert image_block["detail"] == "auto"


# ─────────────────────────────────────────────────────────────────────────────
# File Transformer Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestFileTransformer:
    """Tests for _to_input_file transformer function."""

    @pytest.mark.asyncio
    async def test_file_remote_url_downloaded_and_saved(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        monkeypatch,
    ):
        """Remote file_url inputs should be downloaded and re-hosted in OWUI."""
        remote_url = "https://example.com/manual.pdf"
        stored_id = "remote123"
        pipe_instance.valves.SAVE_REMOTE_FILE_URLS = True

        download_mock = AsyncMock(
            return_value={
                "data": b"%PDF-1.7",
                "mime_type": "application/pdf",
                "url": remote_url,
            }
        )
        upload_mock = AsyncMock(return_value=stored_id)
        status_mock = AsyncMock()

        monkeypatch.setattr(pipe_instance, "_download_remote_url", download_mock)
        monkeypatch.setattr(pipe_instance, "_upload_to_owui_storage", upload_mock)
        monkeypatch.setattr(pipe_instance, "_emit_status", status_mock)

        events: list[dict] = []

        async def event_emitter(event: dict):
            events.append(event)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_url": remote_url,
                        "filename": "manual.pdf",
                    }
                ],
            }
        ]

        transformed = await pipe_instance.transform_messages_to_input(
            messages,
            __request__=mock_request,
            user_obj=mock_user,
            event_emitter=event_emitter,
        )

        assert transformed
        user_message = transformed[0]
        assert user_message["role"] == "user"
        file_block = user_message["content"][0]
        assert file_block["type"] == "input_file"
        assert file_block["file_id"] == stored_id
        assert "file_url" not in file_block

        download_mock.assert_awaited_once_with(remote_url)
        upload_mock.assert_awaited_once()
        status_mock.assert_awaited_with(
            event_emitter,
            StatusMessages.FILE_REMOTE_SAVED,
            done=False,
        )

    @pytest.mark.asyncio
    async def test_file_remote_url_passthrough_when_disabled(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        monkeypatch,
    ):
        """Remote file_url should pass through when valve disabled."""
        remote_url = "https://example.com/manual.pdf"
        pipe_instance.valves.SAVE_REMOTE_FILE_URLS = False

        download_mock = AsyncMock()
        upload_mock = AsyncMock()
        status_mock = AsyncMock()

        monkeypatch.setattr(pipe_instance, "_download_remote_url", download_mock)
        monkeypatch.setattr(pipe_instance, "_upload_to_owui_storage", upload_mock)
        monkeypatch.setattr(pipe_instance, "_emit_status", status_mock)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_url": remote_url,
                    }
                ],
            }
        ]

        transformed = await pipe_instance.transform_messages_to_input(
            messages,
            __request__=mock_request,
            user_obj=mock_user,
            event_emitter=None,
        )

        file_block = transformed[0]["content"][0]
        assert file_block["file_url"] == remote_url
        download_mock.assert_not_called()
        upload_mock.assert_not_called()
        status_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_file_remote_url_warns_when_download_returns_none(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        monkeypatch,
    ):
        remote_url = "https://example.com/manual.pdf"
        pipe_instance.valves.SAVE_REMOTE_FILE_URLS = True

        download_mock = AsyncMock(return_value=None)
        upload_mock = AsyncMock()
        notification_mock = AsyncMock()

        monkeypatch.setattr(pipe_instance, "_download_remote_url", download_mock)
        monkeypatch.setattr(pipe_instance, "_upload_to_owui_storage", upload_mock)
        monkeypatch.setattr(pipe_instance, "_emit_notification", notification_mock)

        async def event_emitter(_event: dict):
            return

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_url": remote_url,
                        "filename": "manual.pdf",
                    }
                ],
            }
        ]

        transformed = await pipe_instance.transform_messages_to_input(
            messages,
            __request__=mock_request,
            user_obj=mock_user,
            event_emitter=event_emitter,
        )

        file_block = transformed[0]["content"][0]
        assert file_block["file_url"] == remote_url
        assert "file_data" not in file_block

        download_mock.assert_awaited_once_with(remote_url)
        upload_mock.assert_not_called()
        notification_mock.assert_awaited()

    @pytest.mark.asyncio
    async def test_file_data_remote_url_moves_to_file_url_when_download_returns_none(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        monkeypatch,
    ):
        remote_url = "https://example.com/manual.pdf"
        pipe_instance.valves.SAVE_FILE_DATA_CONTENT = True

        download_mock = AsyncMock(return_value=None)
        upload_mock = AsyncMock()
        notification_mock = AsyncMock()

        monkeypatch.setattr(pipe_instance, "_download_remote_url", download_mock)
        monkeypatch.setattr(pipe_instance, "_upload_to_owui_storage", upload_mock)
        monkeypatch.setattr(pipe_instance, "_emit_notification", notification_mock)

        async def event_emitter(_event: dict):
            return

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "file_data": remote_url,
                        "filename": "manual.pdf",
                    }
                ],
            }
        ]

        transformed = await pipe_instance.transform_messages_to_input(
            messages,
            __request__=mock_request,
            user_obj=mock_user,
            event_emitter=event_emitter,
        )

        file_block = transformed[0]["content"][0]
        assert file_block["file_url"] == remote_url
        assert "file_data" not in file_block

        download_mock.assert_awaited_once_with(remote_url)
        upload_mock.assert_not_called()
        notification_mock.assert_awaited()


# ─────────────────────────────────────────────────────────────────────────────
# Audio Transformer Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAudioTransformer:
    """Tests for _to_input_audio transformer function."""

    @pytest.mark.asyncio
    async def test_audio_already_correct_format_passthrough(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        sample_audio_base64,
    ):
        """Should pass through audio already in Responses API format after validation."""
        block = {
            "type": "input_audio",
            "input_audio": {"data": sample_audio_base64, "format": "wav"},
        }
        audio_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert audio_block is not None
        assert audio_block["input_audio"]["data"] == sample_audio_base64
        assert audio_block["input_audio"]["format"] == "wav"

    @pytest.mark.asyncio
    async def test_audio_chat_completions_format_converted(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        sample_audio_base64,
    ):
        """Should convert Chat Completions-style audio payloads."""
        block = {
            "type": "input_audio",
            "mime_type": "audio/wave",
            "input_audio": sample_audio_base64,
        }
        audio_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert audio_block is not None
        assert audio_block["type"] == "input_audio"
        assert audio_block["input_audio"]["data"] == sample_audio_base64
        assert audio_block["input_audio"]["format"] == "wav"

    @pytest.mark.asyncio
    async def test_audio_tool_output_format_converted(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        sample_audio_base64,
    ):
        """Should convert Open WebUI tool audio output format using mimeType."""
        block = {
            "type": "audio",
            "mimeType": "audio/wav",
            "data": sample_audio_base64,
        }
        audio_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert audio_block is not None
        assert audio_block["input_audio"]["format"] == "wav"
        assert audio_block["input_audio"]["data"] == sample_audio_base64

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "mime_type,expected_format",
        [
            ("audio/mpeg", "mp3"),
            ("audio/mp3", "mp3"),
            ("audio/wav", "wav"),
            ("audio/wave", "wav"),
            ("audio/x-wav", "wav"),
            ("audio/unknown", "mp3"),
            (None, "mp3"),
        ],
    )
    async def test_audio_mime_type_to_format_mapping(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        sample_audio_base64,
        mime_type,
        expected_format,
    ):
        """Should correctly map MIME types to supported formats."""
        block = {
            "type": "audio",
            "mimeType": mime_type,
            "data": sample_audio_base64,
        }
        audio_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert audio_block is not None
        assert audio_block["input_audio"]["format"] == expected_format

    @pytest.mark.asyncio
    async def test_audio_invalid_payload_returns_empty_block(
        self,
        pipe_instance,
        mock_request,
        mock_user,
    ):
        """Should return empty audio block for malformed payloads."""
        block = {"type": "input_audio", "input_audio": 12345}
        audio_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert audio_block is not None
        assert audio_block["input_audio"]["data"] == ""
        assert audio_block["input_audio"]["format"] == "mp3"

    @pytest.mark.asyncio
    async def test_audio_error_returns_minimal_block(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        sample_audio_base64,
        monkeypatch,
    ):
        """Should swallow exceptions and return minimal block."""
        boom = RuntimeError("boom")
        monkeypatch.setattr(pipe_instance, "_parse_data_url", Mock(side_effect=boom))
        block = {
            "type": "input_audio",
            "input_audio": f"DATA:audio/mp3;base64,{sample_audio_base64}",
        }
        audio_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert audio_block is not None
        assert audio_block["input_audio"]["data"] == ""
        assert audio_block["input_audio"]["format"] == "mp3"

    @pytest.mark.asyncio
    async def test_audio_data_url_supported(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        sample_audio_base64,
    ):
        """Should parse and accept audio data URLs."""
        block = {
            "type": "input_audio",
            "input_audio": f"data:audio/mp3;base64,{sample_audio_base64}",
        }
        audio_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert audio_block is not None
        assert audio_block["input_audio"]["data"] == sample_audio_base64
        assert audio_block["input_audio"]["format"] == "mp3"

    @pytest.mark.asyncio
    async def test_audio_rejects_remote_urls(
        self,
        pipe_instance,
        mock_request,
        mock_user,
    ):
        """Should reject remote URLs to match OpenRouter requirements."""
        block = {
            "type": "input_audio",
            "input_audio": "https://example.com/audio.mp3",
        }
        audio_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert audio_block is not None
        assert audio_block["input_audio"]["data"] == ""
        assert audio_block["input_audio"]["format"] == "mp3"

    @pytest.mark.asyncio
    async def test_audio_partial_dict_without_format(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        sample_audio_base64,
    ):
        """Should derive format for dict payloads missing explicit format."""
        block = {
            "type": "audio",
            "mime_type": "audio/wav",
            "data": sample_audio_base64,
        }
        audio_block = await _transform_single_block(pipe_instance, block, mock_request, mock_user)
        assert audio_block is not None
        assert audio_block["input_audio"]["format"] == "wav"
        assert audio_block["input_audio"]["data"] == sample_audio_base64


# ─────────────────────────────────────────────────────────────────────────────
# Higher-level Conversation Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConversationRebuild:
    """Higher-level tests for transform_messages_to_input conversation assembly."""

    @pytest.mark.asyncio
    async def test_transform_messages_prunes_artifacts_and_reuses_images(
        self,
        pipe_instance,
        sample_image_base64,
        monkeypatch,
    ):
        call_marker_id = pipe_module.generate_item_id()
        output_marker_id = pipe_module.generate_item_id()
        marker_block = f"[{call_marker_id}]: #\n[{output_marker_id}]: #"
        long_output = "X" * (pipe_module._TOOL_OUTPUT_PRUNE_MIN_LENGTH + 50)

        messages = [
            {"role": "system", "content": "Stay on task."},
            {"role": "developer", "content": "Use CSV outputs."},
            {
                "role": "user",
                "message_id": "u-1",
                "content": [{"type": "text", "text": "First prompt"}],
            },
            {
                "role": "assistant",
                "message_id": "a-1",
                "content": [{"type": "text", "text": f"See cached data\n{marker_block}\n"}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"![chart](data:image/png;base64,{sample_image_base64})",
                    }
                ],
            },
            {
                "role": "user",
                "message_id": "u-2",
                "content": [{"type": "text", "text": "Latest question"}],
            },
        ]

        async def artifact_loader(chat_id, message_id, markers):
            assert chat_id == "chat-1"
            assert message_id == "a-1"
            assert markers == [call_marker_id, output_marker_id]
            return {
                call_marker_id: {
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": "lookup",
                    "arguments": {"foo": 1},
                },
                output_marker_id: {
                    "type": "function_call_output",
                    "call_id": "call-1",
                    "output": long_output,
                },
            }

        original_supports = ModelFamily.supports.__func__

        def fake_supports(cls, feature, model_id):
            if feature == "vision":
                return True
            return original_supports(cls, feature, model_id)

        monkeypatch.setattr(ModelFamily, "supports", classmethod(fake_supports))

        transformed = await pipe_instance.transform_messages_to_input(
            messages,
            chat_id="chat-1",
            openwebui_model_id="demo-model",
            artifact_loader=artifact_loader,
            pruning_turns=1,
            replayed_reasoning_refs=[],
        )

        system_msg = next(item for item in transformed if item.get("role") == "system")
        assert system_msg["content"][0]["text"] == "Stay on task."
        developer_msg = next(item for item in transformed if item.get("role") == "developer")
        assert "Use CSV outputs." in developer_msg["content"][0]["text"]

        artifact = next(
            (item for item in transformed if item.get("type") == "function_call_output"),
            None,
        )
        assert artifact is not None, f"transformed conversation missing artifact: {transformed}"
        assert "[tool output pruned" in artifact["output"]

        final_user = [item for item in transformed if item.get("role") == "user"][-1]
        image_blocks = [block for block in final_user["content"] if block["type"] == "input_image"]
        expected_image = f"data:image/png;base64,{sample_image_base64}"
        assert image_blocks and image_blocks[0]["image_url"] == expected_image


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultimodalIntegration:
    """Integration tests for combined multimodal inputs."""

    @pytest.mark.asyncio
    async def test_combined_text_image_file(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        sample_image_base64,
        monkeypatch,
    ) -> None:
        """Should handle message with text, image, and file."""
        pipe_instance.valves.SAVE_REMOTE_FILE_URLS = True
        pipe_instance._is_safe_url = AsyncMock(return_value=True)

        remote_file_url = "https://example.com/manual.pdf"
        download_mock = AsyncMock(
            return_value={"data": b"%PDF-1.7", "mime_type": "application/pdf", "url": remote_file_url}
        )
        upload_mock = AsyncMock(side_effect=["img123", "file123"])
        inline_mock = AsyncMock(return_value="data:image/png;base64,INLINE")
        monkeypatch.setattr(pipe_instance, "_download_remote_url", download_mock)
        monkeypatch.setattr(pipe_instance, "_upload_to_owui_storage", upload_mock)
        monkeypatch.setattr(pipe_instance, "_inline_owui_file_id", inline_mock)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{sample_image_base64}", "detail": "low"},
                    },
                    {"type": "input_file", "file_url": remote_file_url, "filename": "manual.pdf"},
                ],
            }
        ]
        transformed = await pipe_instance.transform_messages_to_input(
            messages,
            __request__=mock_request,
            user_obj=mock_user,
            event_emitter=None,
        )

        assert transformed and transformed[0]["role"] == "user"
        blocks = transformed[0]["content"]
        assert [b["type"] for b in blocks] == ["input_text", "input_image", "input_file"]
        assert blocks[0]["text"] == "hello"
        assert blocks[1]["image_url"] == "data:image/png;base64,INLINE"
        assert blocks[1]["detail"] == "low"
        assert blocks[2]["file_id"] == "file123"
        assert "file_url" not in blocks[2]

        download_mock.assert_awaited_once_with(remote_file_url)
        assert upload_mock.await_count == 2
        inline_mock.assert_awaited()

    @pytest.mark.asyncio
    async def test_combined_text_audio_image(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        sample_audio_base64,
        monkeypatch,
    ) -> None:
        """Should handle message with text, audio, and image."""
        upload_mock = AsyncMock(return_value="img999")
        inline_mock = AsyncMock(return_value="data:image/png;base64,INLINE")
        monkeypatch.setattr(pipe_instance, "_upload_to_owui_storage", upload_mock)
        monkeypatch.setattr(pipe_instance, "_inline_owui_file_id", inline_mock)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "listen"},
                    {"type": "input_audio", "input_audio": f"data:audio/mp3;base64,{sample_audio_base64}"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA", "detail": "auto"}},
                ],
            }
        ]
        transformed = await pipe_instance.transform_messages_to_input(
            messages,
            __request__=mock_request,
            user_obj=mock_user,
            event_emitter=None,
        )

        blocks = transformed[0]["content"]
        assert [b["type"] for b in blocks] == ["input_text", "input_audio", "input_image"]
        assert blocks[0]["text"] == "listen"
        assert blocks[1]["input_audio"]["data"] == sample_audio_base64
        assert blocks[1]["input_audio"]["format"] == "mp3"
        assert blocks[2]["image_url"] == "data:image/png;base64,INLINE"
        inline_mock.assert_awaited()

    @pytest.mark.asyncio
    async def test_multiple_images_in_message(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        sample_image_base64,
        monkeypatch,
    ) -> None:
        """Should handle multiple images in single message."""
        upload_mock = AsyncMock(side_effect=["img1", "img2"])
        inline_mock = AsyncMock(side_effect=["data:image/png;base64,img1", "data:image/png;base64,img2"])
        monkeypatch.setattr(pipe_instance, "_upload_to_owui_storage", upload_mock)
        monkeypatch.setattr(pipe_instance, "_inline_owui_file_id", inline_mock)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "two images"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{sample_image_base64}"}},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }
        ]
        transformed = await pipe_instance.transform_messages_to_input(
            messages,
            __request__=mock_request,
            user_obj=mock_user,
            event_emitter=None,
        )

        blocks = transformed[0]["content"]
        types = [b["type"] for b in blocks]
        assert types == ["input_text", "input_image", "input_image"]
        assert [b["image_url"] for b in blocks[1:]] == [
            "data:image/png;base64,img1",
            "data:image/png;base64,img2",
        ]

    @pytest.mark.asyncio
    async def test_error_in_one_block_does_not_crash_others(
        self,
        pipe_instance,
        mock_request,
        mock_user,
        monkeypatch,
    ) -> None:
        """Should process other blocks even if one fails."""
        pipe_instance.valves.SAVE_REMOTE_FILE_URLS = True
        pipe_instance._is_safe_url = AsyncMock(return_value=True)

        remote_file_url = "https://example.com/manual.pdf"
        download_mock = AsyncMock(
            return_value={"data": b"%PDF-1.7", "mime_type": "application/pdf", "url": remote_file_url}
        )
        upload_mock = AsyncMock(side_effect=[RuntimeError("boom"), "file-ok"])
        monkeypatch.setattr(pipe_instance, "_download_remote_url", download_mock)
        monkeypatch.setattr(pipe_instance, "_upload_to_owui_storage", upload_mock)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    {"type": "input_file", "file_url": remote_file_url, "filename": "manual.pdf"},
                ],
            }
        ]
        transformed = await pipe_instance.transform_messages_to_input(
            messages,
            __request__=mock_request,
            user_obj=mock_user,
            event_emitter=None,
        )

        blocks = transformed[0]["content"]
        assert [b["type"] for b in blocks] == ["input_image", "input_file"]
        # Image failure should fall back to original data URL.
        assert blocks[0]["image_url"] == "data:image/png;base64,AAAA"
        # File should still be processed.
        assert blocks[1]["file_id"] == "file-ok"
        assert "file_url" not in blocks[1]


# ─────────────────────────────────────────────────────────────────────────────
# Documentation Compliance Tests
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
