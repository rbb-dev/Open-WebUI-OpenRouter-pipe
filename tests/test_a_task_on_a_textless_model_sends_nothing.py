"""A background task on a model that cannot answer in text sends nothing (GitHub issue #61)."""

from __future__ import annotations

from typing import Any

import pytest
from aioresponses import aioresponses
from yarl import URL

import open_webui_openrouter_pipe.pipe as pipe_mod
from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.models.registry import (
    OpenRouterModelRegistry as Registry,
)

_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
_RESPONSES_URL = "https://openrouter.ai/api/v1/responses"
_TASK_ENDPOINTS = (URL(_CHAT_URL), URL(_RESPONSES_URL))

_PICTURE = "vendor/picture-only"
_VIDEO = "vendor/video-only"
_TEXT_AND_IMAGE = "vendor/text-and-image"
_NO_SPEC = "vendor/no-spec"

_CAPABILITIES = {"vision": False, "file_upload": False, "web_search": False, "image_generation": True,
                 "code_interpreter": False, "citations": False, "status_updates": True, "usage": True}
_SPECS = {
    _PICTURE: {
        "features": {"image_output", "image_gen_tool"},
        "capabilities": _CAPABILITIES,
        "architecture": {"input_modalities": ["text"], "output_modalities": ["image"]},
    },
    _VIDEO: {
        "features": {"video_generation"},
        "capabilities": {**_CAPABILITIES, "image_generation": False},
        "architecture": {"input_modalities": ["text"], "output_modalities": ["video"]},
    },
    _TEXT_AND_IMAGE: {
        "features": {"image_output", "image_gen_tool"},
        "capabilities": {**_CAPABILITIES, "vision": True, "file_upload": True},
        "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text", "image"]},
    },
}


def _norm(model_id: str) -> str:
    return model_id.replace("/", ".")


@pytest.fixture(autouse=True)
def _registry():
    saved = (
        dict(Registry._specs),
        dict(Registry._id_map),
        list(Registry._models),
        Registry._chat_catalog_norms,
        Registry._image_catalog_norms,
        Registry._video_catalog_norms,
    )
    Registry._specs = {}
    Registry._id_map = {}
    Registry._models = []
    Registry._chat_catalog_norms = frozenset()
    Registry._image_catalog_norms = frozenset()
    Registry._video_catalog_norms = frozenset()
    for model_id, spec in _SPECS.items():
        norm_id = _norm(model_id)
        Registry._specs[norm_id] = spec
        Registry._id_map[norm_id] = norm_id
        Registry._models.append({"id": model_id, "norm_id": norm_id, "original_id": model_id, "name": model_id})
    yield
    (
        Registry._specs,
        Registry._id_map,
        Registry._models,
        Registry._chat_catalog_norms,
        Registry._image_catalog_norms,
        Registry._video_catalog_norms,
    ) = saved


def _make_pipe(monkeypatch) -> Pipe:
    async def loaded(*_args: Any, **_kwargs: Any) -> None:
        return None

    pipe = Pipe()
    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test-key", None))
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", loaded)
    monkeypatch.setattr(
        pipe_mod.OpenRouterModelRegistry,
        "list_models",
        lambda: [{"id": n, "name": n, "norm_id": _norm(n)} for n in (*_SPECS, _NO_SPEC)],
    )
    return pipe


async def _task_turn(pipe: Pipe, model: str, kind: str) -> Any:
    return await pipe.pipe(
        body={"model": model, "messages": [{"role": "user", "content": "Draw a cat."}], "stream": False},
        __user__={"id": "user-1", "role": "user"},
        __request__=None,
        __event_emitter__=None,
        __event_call__=None,
        __metadata__={"task": kind, "chat_id": "chat-1", "user_id": "u1", "message_id": "m1"},
        __tools__=None,
        __task__=kind,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", (_PICTURE, _VIDEO))
async def test_a_task_on_a_textless_model_sends_nothing_and_answers_empty(monkeypatch, model):
    pipe = _make_pipe(monkeypatch)
    outcomes: list[str] = []

    async def _record(_usage: Any, status: str, **_kwargs: Any) -> None:
        outcomes.append(status)

    monkeypatch.setattr(pipe, "_dispatch_generation_complete", _record)
    with aioresponses():
        answer = await _task_turn(pipe, model, "title_generation")

    assert answer == "", f"{model} answered {answer!r} rather than the empty answer"
    assert "failed" not in outcomes, f"{model} reported a failure: {outcomes!r}"
    assert dict(pipe._circuit_breaker._breaker_records) == {}, (
        f"{model} wrote a breaker record: {pipe._circuit_breaker._breaker_records!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", (_TEXT_AND_IMAGE, _NO_SPEC))
async def test_a_model_that_can_answer_in_text_still_sends_its_task(monkeypatch, model):
    pipe = _make_pipe(monkeypatch)
    with aioresponses() as mocked:
        for _ in range(2):
            mocked.post(URL(_RESPONSES_URL), status=200, payload={
                "id": "resp-1",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "A cat"}]}],
            })
            mocked.post(URL(_CHAT_URL), status=200, payload={
                "id": "chat-1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "A cat"}, "finish_reason": "stop"}],
            })
        answer = await _task_turn(pipe, model, "title_generation")
        sent = [str(key[1]) for key in mocked.requests if key[1] in _TASK_ENDPOINTS]

    assert answer == "A cat", f"{model} answered {answer!r}"
    assert sent == [_RESPONSES_URL], f"{model} sent {sent!r}"
