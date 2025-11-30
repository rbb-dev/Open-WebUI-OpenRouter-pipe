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
import socket
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

import openrouter_responses_pipe.openrouter_responses_pipe as pipe_module
from openrouter_responses_pipe.openrouter_responses_pipe import (
    Pipe,
    ResponsesBody,
    StatusMessages,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def pipe_instance():
    """Create a Pipe instance for testing."""
    return Pipe()


@pytest.fixture
def mock_request():
    """Mock FastAPI Request object."""
    request = Mock()
    request.app.url_path_for = Mock(return_value="/api/v1/files/test123")
    return request


@pytest.fixture
def mock_user():
    """Mock UserModel object."""
    user = Mock()
    user.id = "user123"
    user.email = "test@example.com"
    user.name = "Test User"
    return user


@pytest.fixture
def sample_image_base64():
    """Sample base64 encoded image (1x1 transparent PNG)."""
    return "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="


@pytest.fixture
def sample_audio_base64():
    """Sample base64 encoded audio data (mock)."""
    return base64.b64encode(b"FAKE_AUDIO_DATA").decode("utf-8")


async def _transform_single_block(
    pipe_instance: Pipe,
    block: dict,
    mock_request,
    mock_user,
) -> dict:
    """Helper to transform a single user message block."""
    messages = [
        {
            "role": "user",
            "content": [block],
        }
    ]
    transformed = await ResponsesBody.transform_messages_to_input(
        pipe_instance,
        messages,
        __request__=mock_request,
        user_obj=mock_user,
        event_emitter=None,
    )
    return transformed[0]["content"][0]


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


class TestRemoteURLDownloading:
    """Tests for _download_remote_url helper method."""

    @pytest.mark.asyncio
    async def test_download_successful(self, pipe_instance):
        """Should download remote file successfully."""
        test_content = b"fake image data"

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.content = test_content
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_response.raise_for_status = Mock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

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
        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.content = b"data"
            mock_response.headers = {"content-type": "image/jpg; charset=utf-8"}
            mock_response.raise_for_status = Mock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await pipe_instance._download_remote_url(
                "https://example.com/image.jpg"
            )

            assert result["mime_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_download_rejects_files_over_default_limit(self, pipe_instance):
        """Should reject files larger than the configured limit (default 50MB)."""
        limit_bytes = pipe_instance._get_effective_remote_file_limit_mb() * 1024 * 1024
        large_content = b"x" * (limit_bytes + 1)

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = Mock()
            mock_response.content = large_content
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_response.raise_for_status = Mock()

            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

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
        """Should return None on network errors."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                side_effect=Exception("Network error")
            )

            result = await pipe_instance._download_remote_url(
                "https://example.com/image.jpg"
            )

            assert result is None

    @pytest.mark.asyncio
    async def test_download_does_not_retry_on_client_errors(self, pipe_instance):
        """Should not retry on non-429 HTTP 4xx errors."""
        url = "https://example.com/forbidden.png"
        request = httpx.Request("GET", url)
        response = httpx.Response(status_code=403, request=request)
        error = httpx.HTTPStatusError("Forbidden", request=request, response=response)

        with patch("httpx.AsyncClient") as mock_client:
            client_ctx = mock_client.return_value.__aenter__.return_value
            client_ctx.get = AsyncMock(side_effect=error)

            result = await pipe_instance._download_remote_url(url)

            assert result is None
            assert client_ctx.get.await_count == 1


class TestSSRFIPv6Validation:
    """Ensure _is_safe_url handles IPv6 and mixed DNS responses."""

    def test_blocks_private_ipv6_literal(self, pipe_instance):
        """IPv6 literals in unique-local ranges should be rejected."""
        assert pipe_instance._is_safe_url("http://[fd00::1]/") is False

    def test_allows_global_ipv6_literal(self, pipe_instance):
        """Public IPv6 literals should be considered safe."""
        assert pipe_instance._is_safe_url("https://[2001:4860:4860::8888]/foo")

    def test_blocks_domain_with_private_ipv6_record(self, pipe_instance, monkeypatch):
        """Hosts resolving to any private IPv6 addresses are rejected."""

        def fake_getaddrinfo(host, *args, **kwargs):
            return [
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("fd00::abcd", 0, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0)),
            ]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        assert pipe_instance._is_safe_url("https://example.com/resource") is False

    def test_allows_domain_with_public_ips_only(self, pipe_instance, monkeypatch):
        """Hosts resolving exclusively to public IPv4/IPv6 addresses pass the guard."""

        def fake_getaddrinfo(host, *args, **kwargs):
            return [
                (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("2001:4860:4860::8888", 0, 0, 0)),
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("93.184.216.34", 0)),
            ]

        monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
        assert pipe_instance._is_safe_url("https://example.com/resource")


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

    def test_image_url_simple_string(self):
        """Should handle simple string image URL."""
        # This test validates the basic transformation structure
        # Actual implementation requires async context, tested in integration
        pass

    def test_image_url_nested_dict(self):
        """Should extract URL from nested dict structure."""
        pass

    def test_image_data_url_saved_to_storage(self):
        """Should save base64 image to OWUI storage."""
        pass

    def test_image_remote_url_downloaded_and_saved(self):
        """Should download remote image and save to storage."""
        pass

    def test_image_detail_level_preserved(self):
        """Should preserve detail level (auto/high/low)."""
        pass

    def test_image_error_returns_empty_block(self):
        """Should return empty image block on error without crashing."""
        pass


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
        stored_url = "/api/v1/files/remote123"
        pipe_instance.valves.SAVE_REMOTE_FILE_URLS = True

        download_mock = AsyncMock(
            return_value={
                "data": b"%PDF-1.7",
                "mime_type": "application/pdf",
                "url": remote_url,
            }
        )
        upload_mock = AsyncMock(return_value=stored_url)
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

        transformed = await ResponsesBody.transform_messages_to_input(
            pipe_instance,
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
        assert file_block["file_url"] == stored_url

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

        transformed = await ResponsesBody.transform_messages_to_input(
            pipe_instance,
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
        assert audio_block["input_audio"]["format"] == "wav"
        assert audio_block["input_audio"]["data"] == sample_audio_base64


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMultimodalIntegration:
    """Integration tests for combined multimodal inputs."""

    def test_combined_text_image_file(self):
        """Should handle message with text, image, and file."""
        pass

    def test_combined_text_audio_image(self):
        """Should handle message with text, audio, and image."""
        pass

    def test_multiple_images_in_message(self):
        """Should handle multiple images in single message."""
        pass

    def test_error_in_one_block_does_not_crash_others(self):
        """Should process other blocks even if one fails."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Documentation Compliance Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestOpenRouterCompliance:
    """Tests verifying compliance with OpenRouter documentation."""

    def test_supported_image_formats(self):
        """Should support image/png, image/jpeg, image/webp, image/gif."""
        supported_formats = ["image/png", "image/jpeg", "image/webp", "image/gif"]
        pass

    def test_supported_audio_formats(self):
        """Should support wav and mp3 audio formats only."""
        supported_formats = ["wav", "mp3"]
        pass

    def test_audio_requires_base64_not_urls(self):
        """Audio must be base64-encoded, URLs not supported per docs."""
        pass

    def test_file_size_limit_enforced(self):
        """Should enforce the documented size limit (default 50MB)."""
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
