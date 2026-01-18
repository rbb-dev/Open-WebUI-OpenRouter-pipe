from __future__ import annotations

import base64
from typing import cast
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from open_webui_openrouter_pipe import Pipe


def _m4a_like_base64() -> str:
    # Ensure the decoded prefix has bytes[4:8] == b"ftyp" so the pipe sniffs it as "m4a".
    payload = b"\x00\x00\x00\x18ftypisom\x00\x00\x00\x00"
    return base64.b64encode(payload).decode("ascii")


@pytest.mark.asyncio
async def test_direct_uploads_audio_allowlist_makes_format_responses_eligible(pipe_instance_async):
    pipe = pipe_instance_async
    session = cast(aiohttp.ClientSession, object())
    body = {
        "model": "google/gemini-3-flash-preview",
        "messages": [{"role": "user", "content": "listen"}],
        "stream": False,
    }
    metadata = {
        "chat_id": "chat_1",
        "openrouter_pipe": {
            "direct_uploads": {
                "responses_audio_format_allowlist": "m4a",
                "audio": [{"id": "file_audio_1", "format": "mp3"}],
            }
        },
    }

    pipe._get_file_by_id = AsyncMock(return_value=SimpleNamespace(id="file_audio_1"))  # type: ignore[method-assign]
    pipe._read_file_record_base64 = AsyncMock(return_value=_m4a_like_base64())  # type: ignore[method-assign]

    def _unexpected_endpoint_selection(*_args, **_kwargs):
        raise AssertionError("Should not route to /chat/completions when allowlist permits the audio format.")

    async def _fail_fast_on_templated_error(*args, **kwargs):
        variables = kwargs.get("variables") or (args[2] if len(args) > 2 else {})
        reason = variables.get("reason") if isinstance(variables, dict) else None
        raise AssertionError(f"Direct uploads injection failed unexpectedly: {reason}")

    with patch.object(pipe, "_select_llm_endpoint_with_forced", new=_unexpected_endpoint_selection), patch(
        "open_webui_openrouter_pipe.pipe.Pipe._emit_templated_error",
        new=_fail_fast_on_templated_error,
    ), patch(
        "open_webui_openrouter_pipe.transforms.ResponsesBody.from_completions",
        new=AsyncMock(side_effect=RuntimeError("stop_after_injection")),
    ):
        with pytest.raises(RuntimeError, match="stop_after_injection"):
            await pipe._process_transformed_request(  # type: ignore[attr-defined]
                body,
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


@pytest.mark.asyncio
async def test_direct_uploads_audio_default_allowlist_routes_to_chat_when_format_not_eligible(pipe_instance_async):
    pipe = pipe_instance_async
    session = cast(aiohttp.ClientSession, object())
    body = {
        "model": "google/gemini-3-flash-preview",
        "messages": [{"role": "user", "content": "listen"}],
        "stream": False,
    }
    metadata = {
        "chat_id": "chat_1",
        "openrouter_pipe": {
            "direct_uploads": {
                "responses_audio_format_allowlist": "mp3,wav",
                "audio": [{"id": "file_audio_1", "format": "mp3"}],
            }
        },
    }

    pipe._get_file_by_id = AsyncMock(return_value=SimpleNamespace(id="file_audio_1"))  # type: ignore[method-assign]
    pipe._read_file_record_base64 = AsyncMock(return_value=_m4a_like_base64())  # type: ignore[method-assign]

    called = {"count": 0}

    def _endpoint_selection(*_args, **_kwargs):
        called["count"] += 1
        return "responses", False

    async def _fail_fast_on_templated_error(*args, **kwargs):
        variables = kwargs.get("variables") or (args[2] if len(args) > 2 else {})
        reason = variables.get("reason") if isinstance(variables, dict) else None
        raise AssertionError(f"Direct uploads injection failed unexpectedly: {reason}")

    with patch.object(pipe, "_select_llm_endpoint_with_forced", new=_endpoint_selection), patch(
        "open_webui_openrouter_pipe.pipe.Pipe._emit_templated_error",
        new=_fail_fast_on_templated_error,
    ), patch(
        "open_webui_openrouter_pipe.transforms.ResponsesBody.from_completions",
        new=AsyncMock(side_effect=RuntimeError("stop_after_injection")),
    ):
        with pytest.raises(RuntimeError, match="stop_after_injection"):
            await pipe._process_transformed_request(  # type: ignore[attr-defined]
                body,
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

    assert called["count"] == 1
