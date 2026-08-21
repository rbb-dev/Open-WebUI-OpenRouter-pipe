"""RED TEAM: does user-chosen content widen a process-global warn latch?"""
from __future__ import annotations

import base64
import logging
import struct
import zlib
from types import SimpleNamespace

import pytest


def _png(w: int, h: int) -> bytes:
    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IEND", b"")


@pytest.mark.asyncio
async def test_distinct_image_sizes_each_add_a_permanent_latch_entry(monkeypatch, caplog):
    from unittest.mock import AsyncMock, MagicMock
    from open_webui_openrouter_pipe.integrations import video as vm
    from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter

    vm._warned_dropped_video_param.clear()

    sizes = [(10, 10 + i) for i in range(40)]  # all below the 256px floor -> all rejected
    blobs = {f"ref-{i}": base64.b64encode(_png(w, h)).decode()
             for i, (w, h) in enumerate(sizes)}

    adapter = VideoGenerationAdapter.__new__(VideoGenerationAdapter)
    adapter.logger = logging.getLogger("openrouter.video.rt_latch")
    adapter._pipe = MagicMock()

    async def _read(file_obj, *_a, **_k):
        return blobs[file_obj.id]

    adapter._pipe._file_gateway.read_file_record_base64 = AsyncMock(side_effect=_read)

    async def _get_file(file_id, _logger):
        return SimpleNamespace(id=file_id, filename=f"{file_id}.png")

    monkeypatch.setattr(vm, "get_file_by_id", _get_file)
    monkeypatch.setattr(vm, "infer_file_mime_type", lambda _f: "image/png")

    valves = SimpleNamespace(
        VIDEO_FRAME_IMAGE_MAX_BYTES=1 << 20, REMOTE_VIDEO_MAX_SIZE_MB=1,
        VIDEO_FRAME_TOTAL_MAX_BYTES=1 << 20, IMAGE_UPLOAD_CHUNK_BYTES=1024,
        VIDEO_FRAME_IMAGE_MIME_ALLOWLIST="image/png,image/jpeg",
        SEND_MEDIA_VIA_FILE_HOST=False,
    )
    withheld: list[tuple[str, str]] = []
    with caplog.at_level(logging.WARNING, logger="openrouter.video.rt_latch"):
        encoded = await adapter._encode_input_references(
            {"input_references": [{"id": k} for k in blobs]},
            valves, withheld=withheld, companions=True,
        )
    assert encoded == []
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    print(f"\n  uploads={len(blobs)}  latch entries={len(vm._warned_dropped_video_param)}"
          f"  WARNING records={len(warnings)}")
    print("  sample latch keys:", sorted(vm._warned_dropped_video_param)[:3])
    assert len(vm._warned_dropped_video_param) == 1, (
        f"one cause must arm the latch once; it armed "
        f"{len(vm._warned_dropped_video_param)} times, one per user-chosen image size"
    )
