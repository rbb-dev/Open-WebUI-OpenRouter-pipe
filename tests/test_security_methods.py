"""Security-focused tests that exercise the real Pipe helpers."""

from __future__ import annotations

import socket
from unittest.mock import AsyncMock

import pytest

import open_webui_openrouter_pipe.pipe as pipe_module
from open_webui_openrouter_pipe import Pipe, StatusMessages


@pytest.fixture
def pipe_instance():
    pipe = Pipe()
    try:
        yield pipe
    finally:
        pipe.shutdown()


def _addrinfo_for_ip(ip: str):
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    if family == socket.AF_INET6:
        sockaddr = (ip, 0, 0, 0)
    else:
        sockaddr = (ip, 0)
    return [(family, socket.SOCK_STREAM, 0, "", sockaddr)]


@pytest.mark.asyncio
async def test_is_safe_url_allows_public_ips(pipe_instance_async, monkeypatch):
    """Publicly routable addresses should pass the SSRF guard."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: _addrinfo_for_ip("8.8.8.8"),
    )
    assert await pipe_instance_async._is_safe_url("https://example.com/image.jpg") is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ip",
    [
        "10.0.0.1",
        "192.168.1.1",
        "172.16.0.1",
        "127.0.0.1",
        "169.254.169.254",
        "224.0.0.1",
        "fd00::1",
    ],
)
async def test_is_safe_url_blocks_private_ranges(pipe_instance_async, monkeypatch, ip):
    """Private, loopback, link-local, multicast, and unique-local ranges must be rejected."""
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: _addrinfo_for_ip(ip),
    )
    assert await pipe_instance_async._is_safe_url("https://example.com/resource") is False


@pytest.mark.asyncio
async def test_is_safe_url_blocks_literal_loopback(pipe_instance_async):
    """Literal loopback hosts are rejected without DNS lookups."""
    assert await pipe_instance_async._is_safe_url("http://127.0.0.1/internal") is False


@pytest.mark.asyncio
async def test_is_safe_url_handles_dns_failures(pipe_instance_async, monkeypatch):
    """DNS errors should fall back to 'unsafe'."""
    def _raise(*_args, **_kwargs):
        raise socket.gaierror

    monkeypatch.setattr(socket, "getaddrinfo", _raise)
    assert await pipe_instance_async._is_safe_url("https://unknown.invalid") is False


@pytest.mark.asyncio
async def test_is_safe_url_rejects_urls_without_hostname(pipe_instance_async):
    """file:// and data: URLs should be rejected up front."""
    assert await pipe_instance_async._is_safe_url("file:///etc/passwd") is False
    assert await pipe_instance_async._is_safe_url("data:text/plain,hello") is False


@pytest.mark.asyncio
async def test_is_safe_url_respects_disabled_valve(pipe_instance_async):
    """When protection is disabled, even private URLs are allowed."""
    pipe_instance_async.valves.ENABLE_SSRF_PROTECTION = False
    assert await pipe_instance_async._is_safe_url("http://127.0.0.1/admin") is True


@pytest.mark.asyncio
async def test_download_remote_url_halts_when_ssrf_blocks(pipe_instance_async, monkeypatch):
    """_download_remote_url should short-circuit when SSRF validation fails."""
    guard = AsyncMock(return_value=False)
    monkeypatch.setattr(pipe_instance_async._multimodal_handler, "_is_safe_url", guard)

    class _FailingClient:
        def __init__(self, *args, **kwargs):
            raise AssertionError("HTTP client should not be created when SSRF blocks the URL")

    monkeypatch.setattr(pipe_module.httpx, "AsyncClient", _FailingClient)
    result = await pipe_instance_async._download_remote_url("https://internal.local/secret.png")
    assert result is None
    guard.assert_awaited_once()


def test_is_youtube_url_variants(pipe_instance):
    """_is_youtube_url should detect both standard and short formats."""
    assert pipe_instance._is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True
    assert pipe_instance._is_youtube_url("https://youtu.be/dQw4w9WgXcQ") is True
    assert pipe_instance._is_youtube_url("https://vimeo.com/123456") is False
    assert pipe_instance._is_youtube_url("") is False


def test_status_messages_are_strings():
    """StatusMessages entries should be non-empty strings with emoji cues."""
    emoji_required = {"IMAGE": "📥", "FILE": "📥", "VIDEO": "🎥", "AUDIO": "🎵"}
    for name in dir(StatusMessages):
        if name.startswith("_"):
            continue
        value = getattr(StatusMessages, name)
        assert isinstance(value, str) and value, f"{name} should be a non-empty string"
        prefix = name.split("_", 1)[0]
        expected_emoji = emoji_required.get(prefix)
        if expected_emoji:
            assert expected_emoji in value


def test_valve_defaults_are_sensible():
    """Key valve defaults should stay within documented bounds."""
    valves = Pipe.Valves()
    assert 1 <= valves.VIDEO_MAX_SIZE_MB <= 1000
    assert valves.ENABLE_SSRF_PROTECTION is True
