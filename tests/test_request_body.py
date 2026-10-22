from __future__ import annotations

import base64
import json
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe.core.media_slot import MediaSlot
from open_webui_openrouter_pipe.core.request_body import StreamedJSONPayload


class _Sink:
    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    async def write(self, chunk: Any) -> None:
        self.chunks.append(bytes(chunk))


@pytest.mark.asyncio
async def test_the_streamed_body_is_the_json_dumps_bytes_and_is_sized_before_it_is_written(tmp_path):
    clip = bytes(range(256)) * 17
    path = tmp_path / "clip.bin"
    path.write_bytes(clip)
    speech = base64.b64encode(b"spoken words").decode("ascii")

    def body(picture: Any, audio: Any) -> dict[Any, Any]:
        return {
            "model": "vendor/model",
            "input": [{"type": "input_image", "image_url": picture, "detail": "auto"}],
            "messages": [{"content": [{"input_audio": {"data": audio, "format": "wav"}}]}],
            "note": "café   \"quoted\"",
            7: "an integer key",
        }

    with_slots = body(
        MediaSlot(mime="video/mp4", path=path),
        MediaSlot(mime="audio/wav", kind="bare", text=speech),
    )
    as_strings = body("data:video/mp4;base64," + base64.b64encode(clip).decode("ascii"), speech)

    payload = StreamedJSONPayload(with_slots, chunk_bytes=1000)
    sink = _Sink()
    await payload.write_with_length(cast(Any, sink), payload.size)

    expected = json.dumps(as_strings).encode("ascii")
    assert b"".join(sink.chunks) == expected
    assert payload.size == len(expected)
    assert len(sink.chunks) > 4


def test_a_slot_refuses_text_the_writer_would_have_to_escape():
    for text in ('QUJD"}]', "QUJD\u00e9"):
        with pytest.raises(ValueError):
            MediaSlot(mime="image/png", text=text)
