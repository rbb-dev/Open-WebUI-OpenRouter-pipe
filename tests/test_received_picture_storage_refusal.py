"""A picture received to disk that Open WebUI does not store is named, never inlined.

A generated picture large enough to be spooled to disk while the reply streams used to be
read back whole and written into the reply as a `data:` URL whenever Open WebUI's storage
did not take it, so a picture the admin's Max Upload Size refused reached the chat anyway,
whole, and was held in memory to get there. The reply now names the gap, and names Open
WebUI's Max Upload Size when that limit is what refused it (Open WebUI raises its 413 for a
file past `rag.file.max_size`).
"""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastapi import HTTPException

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.api.transforms import ResponsesBody
from open_webui_openrouter_pipe.core.errors import StatusMessages
from open_webui_openrouter_pipe.storage.receive_spool import PartRef, ReceiveSpool, scan_event
from tests.test_streaming_handler import _make_fake_stream


@pytest.mark.parametrize(
    ("failure", "notice"),
    [
        (HTTPException(status_code=413, detail="too large"), StatusMessages.IMAGES_REFUSED_BY_MAX_UPLOAD_SIZE),
        (RuntimeError("storage unavailable"), StatusMessages.IMAGES_NOT_STORED),
    ],
)
@pytest.mark.asyncio
async def test_a_spooled_picture_open_webui_does_not_store_is_named_and_never_inlined(
    failure, notice, monkeypatch, pipe_instance_async
):
    from open_webui_openrouter_pipe.storage import owui_files

    pipe = pipe_instance_async
    picture = b"\x89PNG\r\n\x1a\n" + bytes(range(251)) * 6000
    item = {
        "type": "image_generation_call",
        "id": "img-1",
        "status": "completed",
        "result": [{"b64_json": base64.b64encode(picture).decode("ascii")}],
    }
    blob = json.dumps({"type": "response.output_item.done", "item": item}).encode("utf-8")
    assert len(blob) > 1024 * 1024, "the event must cross the scan threshold"

    async def upload_file_handler(**_kwargs):
        raise failure

    async def storage_context(_request, _user_obj):
        return object(), SimpleNamespace(id="user-123")

    monkeypatch.setattr(owui_files, "upload_file_handler", upload_file_handler)
    monkeypatch.setattr(pipe._file_gateway, "resolve_storage_context", storage_context)
    emitted: list[dict[str, Any]] = []

    async def emitter(event):
        emitted.append(event)

    with ReceiveSpool() as spool:
        scanned = scan_event(blob, spool)
        assert scanned is not None, "the event was not scanned at all"
        assert isinstance(scanned["item"]["result"][0]["b64_json"], PartRef)
        events = [scanned, {"type": "response.completed", "response": {"output": [], "usage": {}}}]
        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _make_fake_stream(events))
        result = await pipe._streaming_handler._run_streaming_loop(
            ResponsesBody(model="test/model", input=[], stream=True),
            pipe.valves,
            emitter,
            metadata={"model": {"id": "test"}, "chat_id": "chat-1", "message_id": "msg-1"},
            tools={},
            session=cast(Any, object()),
            user_id="user-123",
        )

    named = notice.format(count=1)
    assert "data:image" not in result
    assert "/api/v1/files/" not in result
    assert named in result
    assert any(
        event.get("type") == "status" and (event.get("data") or {}).get("description") == named
        for event in emitted
    )
