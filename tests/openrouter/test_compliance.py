"""OpenRouter compliance tests ensuring upstream requirements remain covered."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock

import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe


async def _transform_single_block(
    pipe_instance: Pipe,
    block: dict,
    mock_request,
    mock_user,
):
    messages = [{"role": "user", "content": [block]}]
    transformed = await pipe_instance.transform_messages_to_input(
        messages,
        __request__=mock_request,
        user_obj=mock_user,
        event_emitter=None,
    )
    return transformed[0]["content"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mime_type",
    ["image/png", "image/jpeg", "image/webp", "image/gif"],
)
async def test_supported_image_formats_are_inlined(
    pipe_instance,
    mock_request,
    mock_user,
    sample_image_base64,
    mime_type,
    monkeypatch,
):
    """All documented OpenRouter image formats should survive the transform pipeline."""

    inline_value = f"data:{mime_type};base64,{sample_image_base64}"

    monkeypatch.setattr(
        pipe_instance,
        "_resolve_storage_context",
        AsyncMock(return_value=(mock_request, mock_user)),
    )
    monkeypatch.setattr(
        pipe_instance,
        "_upload_to_owui_storage",
        AsyncMock(return_value="/api/v1/files/mock-image"),
    )
    monkeypatch.setattr(
        pipe_instance,
        "_inline_internal_file_url",
        AsyncMock(return_value=inline_value),
    )

    block = {"type": "image_url", "image_url": f"data:{mime_type};base64,{sample_image_base64}"}

    result = await _transform_single_block(pipe_instance, block, mock_request, mock_user)

    assert result["type"] == "input_image"
    assert result["image_url"] == inline_value
    pipe_instance._inline_internal_file_url.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("block", "expected_format"),
    [
        (
            {"type": "input_audio", "input_audio": {"data": "DATA", "format": "mp3"}},
            "mp3",
        ),
        (
            {"type": "audio", "mimeType": "audio/wav", "data": "DATA"},
            "wav",
        ),
        (
            {"type": "audio", "mimeType": "audio/mp3", "data": "DATA"},
            "mp3",
        ),
    ],
)
async def test_supported_audio_formats_map_correctly(
    pipe_instance,
    block,
    expected_format,
    sample_audio_base64,
    mock_request,
    mock_user,
    monkeypatch,
):
    """OpenRouter accepts only wav/mp3 audio and we enforce the same."""

    payload = dict(block)
    if isinstance(payload.get("input_audio"), dict):
        payload["input_audio"] = dict(payload["input_audio"])
        payload["input_audio"]["data"] = sample_audio_base64
    elif "data" in payload:
        payload["data"] = sample_audio_base64

    pipe_instance._emit_error = AsyncMock()

    result = await _transform_single_block(pipe_instance, payload, mock_request, mock_user)

    assert result["type"] == "input_audio"
    assert result["input_audio"]["data"] == sample_audio_base64
    assert result["input_audio"]["format"] == expected_format
    pipe_instance._emit_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_audio_requires_base64_not_urls(
    pipe_instance,
    mock_request,
    mock_user,
):
    """Remote URLs should be rejected per OpenRouter's audio spec."""

    pipe_instance._emit_error = AsyncMock()

    block = {"type": "input_audio", "input_audio": "https://example.com/audio.mp3"}

    result = await _transform_single_block(pipe_instance, block, mock_request, mock_user)

    assert result["type"] == "input_audio"
    assert result["input_audio"]["data"] == ""
    assert pipe_instance._emit_error.await_count == 1


def test_file_size_limit_enforced(pipe_instance):
    """Large base64 payloads must be rejected to honor OpenRouter limits."""

    pipe_instance.valves.BASE64_MAX_SIZE_MB = 1  # shrink for predictable test sizes
    small_payload = base64.b64encode(b"0" * (256 * 1024)).decode("ascii")
    large_payload = base64.b64encode(b"0" * (2 * 1024 * 1024)).decode("ascii")

    assert pipe_instance._validate_base64_size(small_payload) is True
    assert pipe_instance._validate_base64_size(large_payload) is False
