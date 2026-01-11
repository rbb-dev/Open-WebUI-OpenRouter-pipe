from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe


def _make_existing_model(model_id: str, meta: dict, params: dict | None = None):
    from open_webui.models.models import ModelMeta  # provided by test stubs

    return SimpleNamespace(
        id=model_id,
        base_model_id=None,
        name="Example",
        meta=ModelMeta(**meta),
        params=params or {},
        access_control=None,
        is_active=True,
    )


def test_auto_attach_direct_uploads_filter_and_persist_capabilities():
    pipe = Pipe()
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(model_id, meta={})
    update_mock = Mock()

    pipe_caps = {"file_input": True, "audio_input": False, "video_input": False, "vision": True}

    with patch(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.get_model_by_id",
        return_value=existing,
    ), patch(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.update_model_by_id",
        new=update_mock,
    ), patch(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelForm",
        new=lambda **kw: SimpleNamespace(**kw),
    ):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "Example",
            capabilities=None,
            profile_image_url=None,
            update_capabilities=False,
            update_images=False,
            direct_uploads_filter_function_id="openrouter_direct_uploads",
            direct_uploads_filter_supported=True,
            auto_attach_direct_uploads_filter=True,
            openrouter_pipe_capabilities=pipe_caps,
        )

    assert update_mock.call_count == 1
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["filterIds"] == ["openrouter_direct_uploads"]
    assert meta["openrouter_pipe"]["direct_uploads_filter_id"] == "openrouter_direct_uploads"
    assert meta["openrouter_pipe"]["capabilities"] == pipe_caps


def test_auto_attach_removes_direct_uploads_filter_when_unsupported():
    pipe = Pipe()
    model_id = "open_webui_openrouter_pipe.openai.gpt-4o"

    existing = _make_existing_model(
        model_id,
        meta={
            "filterIds": ["openrouter_direct_uploads"],
            "openrouter_pipe": {"direct_uploads_filter_id": "openrouter_direct_uploads"},
        },
    )
    update_mock = Mock()

    with patch(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.get_model_by_id",
        return_value=existing,
    ), patch(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.Models.update_model_by_id",
        new=update_mock,
    ), patch(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelForm",
        new=lambda **kw: SimpleNamespace(**kw),
    ):
        pipe._update_or_insert_model_with_metadata(
            model_id,
            "Example",
            capabilities=None,
            profile_image_url=None,
            update_capabilities=False,
            update_images=False,
            direct_uploads_filter_function_id="openrouter_direct_uploads",
            direct_uploads_filter_supported=False,
            auto_attach_direct_uploads_filter=True,
            openrouter_pipe_capabilities={"file_input": False, "audio_input": False, "video_input": False, "vision": False},
        )

    assert update_mock.call_count == 1
    updated_form = update_mock.call_args[0][1]
    meta = dict(updated_form.meta)
    assert meta["filterIds"] == []


@pytest.mark.asyncio
async def test_send_openrouter_streaming_request_respects_endpoint_override():
    pipe = Pipe()

    async def _chat_stream(*_args, **_kwargs):
        yield {"type": "chat"}

    async def _responses_stream(*_args, **_kwargs):
        yield {"type": "responses"}

    with patch.object(pipe, "send_openai_chat_completions_streaming_request", new=_chat_stream), patch.object(
        pipe, "send_openai_responses_streaming_request", new=_responses_stream
    ):
        session = cast(aiohttp.ClientSession, object())
        payload = {"model": "openai/gpt-4o", "input": "hi"}

        events = []
        async for evt in pipe.send_openrouter_streaming_request(
            session, payload, api_key="k", base_url="https://example.com", endpoint_override="chat_completions"
        ):
            events.append(evt)
            break

        assert events == [{"type": "chat"}]


@pytest.mark.asyncio
async def test_inline_internal_responses_input_files_inplace_rewrites_internal_urls():
    pipe = Pipe()
    payload = {
        "model": "google/gemini-3-flash-preview",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": "example.pdf",
                        "file_url": "/api/v1/files/abc123/content",
                    }
                ],
            }
        ],
    }

    pipe._inline_owui_file_id = AsyncMock(  # type: ignore[method-assign]
        return_value="data:application/pdf;base64,SGVsbG8="
    )

    await pipe._inline_internal_responses_input_files_inplace(payload, chunk_size=1024, max_bytes=1024 * 1024)
    pipe._inline_owui_file_id.assert_awaited_once_with("abc123", chunk_size=1024, max_bytes=1024 * 1024)

    block = cast(dict, payload["input"][0]["content"][0])
    assert block.get("file_data", "").startswith("data:application/pdf;base64,")
    assert "file_url" not in block
