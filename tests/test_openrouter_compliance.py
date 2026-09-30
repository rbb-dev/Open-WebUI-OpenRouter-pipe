"""OpenRouter compliance tests ensuring upstream requirements remain covered."""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false

from __future__ import annotations

import base64
from unittest.mock import AsyncMock

import pytest

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.requests.transformer import (
    NO_AUDIO_DATA,
    transform_messages_to_input,
)
from open_webui_openrouter_pipe.storage.owui_files import InlinedFile


async def _transform_single_block(
    pipe_instance: Pipe,
    block: dict,
    mock_user,
):
    messages = [{"role": "user", "content": [block]}]
    transformed = await transform_messages_to_input(pipe_instance,
        messages,
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
    """All documented OpenRouter image formats survive the transform pipeline as they came, never stored.

    Each row's payload now carries the signature of the format it declares. The pipe
    types a `data:` URL from its own bytes, so a PNG declared as JPEG, WEBP or GIF is
    re-spelled to `image/png` -- correctly, and the point of B454 -- but it would leave
    this table asserting the same answer three times and covering none of the three
    formats it names.
    """

    bodies = {
        "image/png": base64.b64decode(sample_image_base64),
        "image/jpeg": b"\xff\xd8\xff\xe0" + b"\x00" * 32,
        "image/webp": b"RIFF" + b"\x1a\x00\x00\x00" + b"WEBP" + b"\x00" * 32,
        "image/gif": b"GIF89a" + b"\x00" * 32,
    }
    data_url = f"data:{mime_type};base64,{base64.b64encode(bodies[mime_type]).decode('ascii')}"
    ext = mime_type.split("/")[-1]
    if ext == "jpeg":
        ext = "jpg"

    monkeypatch.setattr(
        pipe_instance._file_gateway,
        "resolve_storage_context",
        AsyncMock(return_value=(mock_request, mock_user)),
    )
    monkeypatch.setattr(
        pipe_instance._file_gateway,
        "upload_to_owui_storage",
        AsyncMock(return_value="mock-image"),
    )
    monkeypatch.setattr(
        pipe_instance._file_gateway,
        "inline_owui_file_id",
        AsyncMock(return_value=InlinedFile(data_url="data:image/png;base64,STORED", filename=f"test.{ext}")),
    )

    block = {"type": "image_url", "image_url": data_url}

    result = await _transform_single_block(pipe_instance, block, mock_user)

    assert result["type"] == "input_image"
    assert result["image_url"] == data_url
    pipe_instance._file_gateway.upload_to_owui_storage.assert_not_awaited()
    pipe_instance._file_gateway.inline_owui_file_id.assert_not_awaited()


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

    pipe_instance._ensure_error_formatter()._emit_error = AsyncMock()

    result = await _transform_single_block(pipe_instance, payload, mock_user)

    assert result["type"] == "input_audio"
    assert result["input_audio"]["data"] == sample_audio_base64
    assert result["input_audio"]["format"] == expected_format
    pipe_instance._ensure_error_formatter()._emit_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_audio_requires_base64_not_urls(
    pipe_instance,
    mock_user,
):
    """Remote URLs should be rejected per OpenRouter's audio spec.

    The rejection is unchanged; what it is reported through is not. A remote URL is a
    malformed payload, and a malformed payload is reported as a `Files: skipped` status
    naming the reason -- not as a per-block error card, which is the wrong vocabulary
    for it and would put a second report beside the status. The turn also says what
    happened instead of leaving an `input_audio` block behind for OpenRouter: an empty
    `data` field is a claim that a clip was attached.
    """

    events: list = []

    async def emitter(event):
        events.append(event)

    block = {"type": "input_audio", "input_audio": "https://example.com/audio.mp3"}
    transformed = await transform_messages_to_input(
        pipe_instance,
        [{"role": "user", "content": [block]}],
        user_obj=mock_user,
        event_emitter=emitter,
    )
    result = transformed[0]["content"][0]

    assert result["type"] == "input_text"
    assert result["text"] == (
        "[An attached item was not sent: an audio clip must be base64-encoded; "
        "URLs are not supported.]"
    )
    statuses = [e for e in events if e.get("type") == "status"]
    assert [e["data"]["description"] for e in statuses] == [
        "Files: skipped 1 (an audio clip must be base64-encoded; URLs are not supported)."
    ], events
    assert not [e for e in events if isinstance((e.get("data") or {}).get("error"), dict)], (
        f"a malformed payload was reported as an error card as well: {events!r}"
    )


def test_file_size_limit_enforced(pipe_instance):
    """Large base64 payloads must be rejected to honor OpenRouter limits."""

    pipe_instance.valves.BASE64_MAX_SIZE_MB = 1  # shrink for predictable test sizes
    small_payload = base64.b64encode(b"0" * (256 * 1024)).decode("ascii")
    large_payload = base64.b64encode(b"0" * (2 * 1024 * 1024)).decode("ascii")

    assert pipe_instance._file_gateway.validate_base64_size(small_payload) is True
    assert pipe_instance._file_gateway.validate_base64_size(large_payload) is False
