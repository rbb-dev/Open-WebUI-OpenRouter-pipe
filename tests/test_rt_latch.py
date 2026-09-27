"""RED TEAM: does user-chosen content widen a process-global warn latch?"""
from __future__ import annotations

import base64
import logging
from types import SimpleNamespace

import pytest


@pytest.mark.asyncio
async def test_distinct_image_sizes_each_add_a_permanent_latch_entry(monkeypatch, caplog):
    """40 user-chosen clip dimensions, all under one model's floor, must arm the latch once.

    The drop reason embeds the dimensions the user sent, so a latch keyed on the sentence
    would write 40 permanent entries into a process-global set from a single request.
    Keyed on the cause, one does.

    Nothing here is accepted, so the count cap -- which is a cap on what is *sent*, not
    on what is examined -- never applies and all 40 are probed. That is what makes the row
    worth writing: the latch sees 40 distinct user-chosen reasons from one request and
    must still hold one entry.
    """
    import itertools
    from unittest.mock import AsyncMock, MagicMock

    from open_webui_openrouter_pipe.integrations import video as vm
    from open_webui_openrouter_pipe.integrations.video import (
        _INPUT_PIXEL_FLOORS,
        VideoGenerationAdapter,
    )
    from open_webui_openrouter_pipe.media.frame_extraction import VideoMetadata

    vm._warned_dropped_video_param.clear()

    model_id = "bytedance/seedance-2.0"
    floor = _INPUT_PIXEL_FLOORS[model_id]
    # Every pair below trips the same floor, and no two are the same size.
    sizes = [
        (w, h)
        for w, h in itertools.product(range(20, 40), repeat=2)
        if 0 < w * h < floor.pixels
    ][:40]
    assert len({(w, h) for w, h in sizes}) == 40

    mp4 = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32
    blobs = {f"clip-{i}": base64.b64encode(mp4).decode() for i, _ in enumerate(sizes)}
    order = iter(sizes)

    adapter = VideoGenerationAdapter.__new__(VideoGenerationAdapter)
    adapter.logger = logging.getLogger("openrouter.video.rt_latch")
    adapter._pipe = MagicMock()

    async def _read(file_obj, *_a, **_k):
        return blobs[file_obj.id]

    adapter._pipe._file_gateway.read_file_record_base64 = AsyncMock(side_effect=_read)

    async def _get_file(file_id, _logger):
        return SimpleNamespace(
            id=file_id, filename=f"{file_id}.mp4", user_id="bob",
            content_type="video/mp4", mime_type=None,
            meta={"content_type": "video/mp4"},
        )

    async def _probe(_path):
        w, h = next(order)
        return VideoMetadata(
            duration_seconds=2.0, width=w, height=h, fps=24.0, has_audio=False
        )

    monkeypatch.setattr(vm, "get_file_by_id", _get_file)
    monkeypatch.setattr(vm, "infer_file_mime_type", lambda _f: "video/mp4")
    monkeypatch.setattr(vm, "probe_video", _probe)

    valves = SimpleNamespace(
        VIDEO_FRAME_IMAGE_MAX_BYTES=1 << 20, REMOTE_VIDEO_MAX_SIZE_MB=1,
        VIDEO_FRAME_TOTAL_MAX_BYTES=1 << 20, IMAGE_UPLOAD_CHUNK_BYTES=1024,
        VIDEO_FRAME_IMAGE_MIME_ALLOWLIST="image/png,image/jpeg",
        SEND_MEDIA_VIA_FILE_HOST=True,
        # The clip floor is checked on the relay path, which is the only path a clip takes.
        SEND_VIDEO_VIA_FILE_HOST=True,
    )
    owner = SimpleNamespace(id="bob", role="user")

    async def _relay(*_a, **_k):
        return "https://files.example.test/PUBLISHED.mp4"

    monkeypatch.setattr(
        VideoGenerationAdapter, "_relay_reference", _relay, raising=True
    )
    withheld: list[tuple[str, str]] = []
    with caplog.at_level(logging.WARNING, logger="openrouter.video.rt_latch"):
        encoded = await adapter._encode_input_references(
            {"input_references": [{"id": k} for k in blobs], "model_id": model_id},
            valves, withheld=withheld, companions=True, user_obj=owner,
            video_model={
                "id": model_id,
                "input_modalities": ["video", "image"],
            },
        )
    assert encoded == [], encoded
    assert len(withheld) == len(sizes), (
        f"{len(withheld)} records for {len(sizes)} clips; the count is what the latch "
        "sees, and a smaller one would mean some clips never reached the floor"
    )
    assert len({note for _name, note in withheld}) == len(withheld), (
        "the reason does not carry the user's own dimensions, so this test no longer "
        "drives the hazard it is about"
    )
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    print(f"\n  uploads={len(blobs)}  withheld={len(withheld)}  "
          f"latch entries={len(vm._warned_dropped_video_param)}  "
          f"WARNING records={len(warnings)}")
    print("  sample latch keys:", sorted(vm._warned_dropped_video_param)[:3])
    assert len(vm._warned_dropped_video_param) == 1, (
        f"one cause must arm the latch once; it armed "
        f"{len(vm._warned_dropped_video_param)} times, one per user-chosen clip size"
    )
