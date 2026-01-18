from __future__ import annotations

from typing import cast
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from open_webui_openrouter_pipe import Pipe


@pytest.mark.asyncio
async def test_task_requests_ignore_direct_uploads_and_do_not_inject(pipe_instance_async):
    pipe = pipe_instance_async
    session = cast(aiohttp.ClientSession, object())

    body = {
        "model": "google/gemini-3-flash-preview",
        "messages": [{"role": "user", "content": "generate a title"}],
        "stream": False,
    }
    metadata = {
        "chat_id": "chat_1",
        "openrouter_pipe": {
            "direct_uploads": {
                "files": [{"id": "file_1", "name": "example.pdf", "content_type": "application/pdf", "size": 123}],
            }
        },
    }

    async def _fake_from_completions(*_args, **_kwargs):
        return SimpleNamespace(
            model="google/gemini-3-flash-preview",
            max_output_tokens=None,
            input=[],
            model_dump=lambda: {"model": "google/gemini-3-flash-preview", "input": []},
        )

    with patch(
        "open_webui_openrouter_pipe.transforms.ResponsesBody.from_completions",
        new=AsyncMock(side_effect=_fake_from_completions),
    ), patch.object(pipe, "_sanitize_request_input", new=lambda *_a, **_k: None), patch.object(
        pipe, "_apply_reasoning_preferences", new=lambda *_a, **_k: None
    ), patch.object(pipe, "_apply_gemini_thinking_config", new=lambda *_a, **_k: None), patch.object(
        pipe, "_apply_context_transforms", new=lambda *_a, **_k: None
    ), patch.object(
        pipe, "_get_user_by_id", new=AsyncMock(return_value=None)
    ), patch.object(
        pipe, "_run_task_model_request", new=AsyncMock(return_value="ok")
    ):
        result = await pipe._process_transformed_request(  # type: ignore[attr-defined]
            body,
            __user__={"id": "u1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__={},
            __task__="title_generation",
            __task_body__={},
            valves=pipe.valves,
            session=session,
            openwebui_model_id="pipe.model",
            pipe_identifier="open_webui_openrouter_pipe",
            allowlist_norm_ids=set(),
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
            user_id="u1",
        )

    assert result == "ok"
    # Metadata marker must not be mutated (it may be reused by a subsequent main chat request).
    assert "direct_uploads" in metadata["openrouter_pipe"]
    # And the task request body must not have file blocks injected.
    content = body["messages"][0]["content"]
    assert isinstance(content, str)


@pytest.mark.asyncio
async def test_task_first_does_not_consume_direct_uploads_for_following_chat(pipe_instance_async):
    pipe = pipe_instance_async
    session = cast(aiohttp.ClientSession, object())

    metadata = {
        "chat_id": "chat_1",
        "openrouter_pipe": {
            "direct_uploads": {
                "files": [{"id": "file_1", "name": "example.pdf", "content_type": "application/pdf", "size": 123}],
            }
        },
    }

    async def _fake_from_completions(*_args, **_kwargs):
        return SimpleNamespace(
            model="google/gemini-3-flash-preview",
            max_output_tokens=None,
            input=[],
            model_dump=lambda: {"model": "google/gemini-3-flash-preview", "input": []},
        )

    task_body = {
        "model": "google/gemini-3-flash-preview",
        "messages": [{"role": "user", "content": "generate a query"}],
        "stream": False,
    }

    with patch(
        "open_webui_openrouter_pipe.transforms.ResponsesBody.from_completions",
        new=AsyncMock(side_effect=_fake_from_completions),
    ), patch.object(pipe, "_sanitize_request_input", new=lambda *_a, **_k: None), patch.object(
        pipe, "_apply_reasoning_preferences", new=lambda *_a, **_k: None
    ), patch.object(pipe, "_apply_gemini_thinking_config", new=lambda *_a, **_k: None), patch.object(
        pipe, "_apply_context_transforms", new=lambda *_a, **_k: None
    ), patch.object(
        pipe, "_get_user_by_id", new=AsyncMock(return_value=None)
    ), patch.object(
        pipe, "_run_task_model_request", new=AsyncMock(return_value="ok")
    ):
        result = await pipe._process_transformed_request(  # type: ignore[attr-defined]
            task_body,
            __user__={"id": "u1"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__={},
            __task__="query_generation",
            __task_body__={},
            valves=pipe.valves,
            session=session,
            openwebui_model_id="pipe.model",
            pipe_identifier="open_webui_openrouter_pipe",
            allowlist_norm_ids=set(),
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
            user_id="u1",
        )

    assert result == "ok"
    assert "direct_uploads" in metadata["openrouter_pipe"]

    chat_body = {
        "model": "google/gemini-3-flash-preview",
        "messages": [{"role": "user", "content": "describe what you see."}],
        "stream": False,
    }

    with patch(
        "open_webui_openrouter_pipe.transforms.ResponsesBody.from_completions",
        new=AsyncMock(side_effect=RuntimeError("stop after injection")),
    ):
        with pytest.raises(RuntimeError, match="stop after injection"):
            await pipe._process_transformed_request(  # type: ignore[attr-defined]
                chat_body,
                __user__={"id": "u1"},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__=metadata,
                __tools__={},
                __task__=None,
                __task_body__=None,
                valves=pipe.valves,
                session=session,
                openwebui_model_id="pipe.model",
                pipe_identifier="open_webui_openrouter_pipe",
                allowlist_norm_ids=set(),
                enforced_norm_ids=set(),
                catalog_norm_ids=set(),
                features={},
                user_id="u1",
            )

    content = chat_body["messages"][0]["content"]
    assert isinstance(content, list)
    assert any(isinstance(block, dict) and block.get("type") == "file" for block in content)
