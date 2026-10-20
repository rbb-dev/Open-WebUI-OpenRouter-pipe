"""A generated picture in a temporary chat is stored the way Open WebUI stores its own.

Open WebUI's own image generation (`routers/images.py` `upload_image`, called by
`tools/builtin.py` `generate_image`) stores the picture's bytes and file row in every chat and
links the file to the chat only when the chat is saved (`is_saved_chat_id`). The streaming path
and the image API path stopped storing anything for a temporary chat and wrote the picture into
the reply as a `data:` URL instead, which also left a picture received to disk as an empty one.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.api.transforms import ResponsesBody
from tests.test_image_api_path import (
    _adapter,
    _Emitter,
    _KeyPipe,
    _posted,
    _seed_contract,
    _StubResponsesBody,
    _StubValves,
)
from tests.test_streaming_handler import _make_fake_stream

TEMPORARY_CHAT = "temporary:3f1c9a"


@pytest.mark.asyncio
async def test_a_streamed_picture_in_a_temporary_chat_is_stored_without_a_chat_link(
    monkeypatch, pipe_instance_async, sample_image_base64
):
    from open_webui.models.chats import Chats

    from open_webui_openrouter_pipe.storage import owui_files

    pipe = pipe_instance_async
    uploads: list[dict[str, Any]] = []

    async def upload_file_handler(**kwargs):
        uploads.append(kwargs)
        return SimpleNamespace(id="file-tmp-1")

    async def storage_context(_request, _user_obj):
        return object(), SimpleNamespace(id="user-123")

    link = AsyncMock()
    monkeypatch.setattr(owui_files, "upload_file_handler", upload_file_handler)
    monkeypatch.setattr(Chats, "insert_chat_files", link, raising=False)
    monkeypatch.setattr(pipe._file_gateway, "resolve_storage_context", storage_context)
    events = [
        {
            "type": "response.output_item.done",
            "item": {
                "type": "image_generation_call",
                "id": "img-1",
                "result": f"data:image/png;base64,{sample_image_base64}",
            },
        },
        {"type": "response.completed", "response": {"output": [], "usage": {}}},
    ]
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", _make_fake_stream(events))

    result = await pipe._streaming_handler._run_streaming_loop(
        ResponsesBody(model="test/model", input=[], stream=True),
        pipe.valves,
        None,
        metadata={"model": {"id": "test"}, "chat_id": TEMPORARY_CHAT, "message_id": "msg-1"},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )

    assert "/api/v1/files/file-tmp-1/content" in result
    assert "data:image" not in result
    assert len(uploads) == 1
    assert "chat_id" not in uploads[0]["metadata"]
    link.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_dedicated_image_reply_in_a_temporary_chat_is_stored_rather_than_inlined():
    adapter = _adapter(_KeyPipe("sk-x"))
    _seed_contract(adapter, "m/x", [{}], at=time.monotonic())

    result = await _posted(
        adapter,
        body={},
        responses_body=_StubResponsesBody(
            [{"role": "user", "content": [{"type": "input_text", "text": "a leaf"}]}]
        ),
        valves=_StubValves("sk-x"),
        event_emitter=_Emitter(),
        metadata={"chat_id": TEMPORARY_CHAT, "message_id": "msg-1"},
        normalized_model_id="m.x",
        api_model_id="m/x",
    )

    assert [call["chat_id"] for call in result.calls] == [TEMPORARY_CHAT]
    assert "/api/v1/files/file-1/content" in str(result.content)
    assert "data:image" not in str(result.content)
