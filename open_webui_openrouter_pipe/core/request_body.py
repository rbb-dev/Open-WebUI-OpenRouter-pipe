from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import secrets
from typing import Any

from aiohttp.abc import AbstractStreamWriter
from aiohttp.payload import Payload

from .media_slot import MediaSlot

STREAMED_BODY_CHUNK_BYTES = 1024 * 1024

_TEXT_WRITE_SLICE_BYTES = 1024 * 1024


def _read_encoded_block(path: str, offset: int, size: int, expected: int | None) -> tuple[bytes, int]:
    with open(path, "rb") as handle:
        if expected is not None and os.fstat(handle.fileno()).st_size != expected:
            raise OSError("a media file changed after the request body was sized")
        handle.seek(offset)
        data = handle.read(size)
        if data and len(data) < size:
            parts = [data]
            have = len(data)
            while have < size:
                more = handle.read(size - have)
                if not more:
                    break
                parts.append(more)
                have += len(more)
            data = b"".join(parts)
    return base64.b64encode(data), len(data)


class StreamedJSONPayload(Payload):
    _autoclose = True

    def __init__(self, body: Any, *, chunk_bytes: int = STREAMED_BODY_CHUNK_BYTES) -> None:
        super().__init__(body, content_type="application/json")
        step = chunk_bytes if isinstance(chunk_bytes, int) and chunk_bytes > 0 else STREAMED_BODY_CHUNK_BYTES
        self._raw_step = max(3, step - step % 3)
        marker = secrets.token_hex(16)
        slots: list[MediaSlot] = []

        def _slot_marker(value: Any) -> str:
            if isinstance(value, MediaSlot):
                slots.append(value)
                return f"@@{marker}:{len(slots) - 1}@@"
            raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

        text = json.dumps(body, default=_slot_marker)
        self._segments: list[bytes | MediaSlot] = []
        size = 0
        cursor = 0
        for found in re.finditer(f'"@@{marker}:(\\d+)@@"', text):
            lead = text[cursor:found.start()].encode("ascii")
            slot = slots[int(found.group(1))]
            self._segments.extend((lead, slot))
            size += len(lead) + slot.json_len
            cursor = found.end()
        tail = text[cursor:].encode("ascii")
        self._segments.append(tail)
        self._size = size + len(tail)

    async def write(self, writer: AbstractStreamWriter) -> None:
        await self.write_with_length(writer, None)

    async def write_with_length(self, writer: AbstractStreamWriter, content_length: int | None) -> None:
        for segment in self._segments:
            if isinstance(segment, MediaSlot):
                await self._write_slot(writer, segment)
                continue
            view = memoryview(segment)
            for start in range(0, len(view), _TEXT_WRITE_SLICE_BYTES):
                await writer.write(view[start:start + _TEXT_WRITE_SLICE_BYTES])

    async def _write_slot(self, writer: AbstractStreamWriter, slot: MediaSlot) -> None:
        await writer.write(b'"' + slot.prefix.encode("ascii"))
        if slot.text is not None:
            step = self._raw_step * 4 // 3
            for start in range(0, len(slot.text), step):
                await writer.write(slot.text[start:start + step].encode("ascii"))
        else:
            assert slot.path is not None
            loop = asyncio.get_running_loop()
            written = 0
            while True:
                block, got = await loop.run_in_executor(
                    None, _read_encoded_block, slot.path, written, self._raw_step, slot.raw_len
                )
                if not got:
                    break
                written += got
                await writer.write(block)
            if written != slot.raw_len:
                raise OSError("a media file changed while the request was being written")
        await writer.write(b'"')

    def decode(self, encoding: str = "utf-8", errors: str = "strict") -> str:
        pieces: list[str] = []
        for segment in self._segments:
            if isinstance(segment, MediaSlot):
                pieces.append('"' + segment.prefix + _slot_base64_text(segment) + '"')
            else:
                pieces.append(segment.decode("ascii"))
        return "".join(pieces)


def _slot_base64_text(slot: MediaSlot) -> str:
    if slot.text is not None:
        return slot.text
    assert slot.path is not None
    with open(slot.path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("ascii")
