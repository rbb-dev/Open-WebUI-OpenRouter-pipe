"""A temporary chat's cost snapshot keeps its spend but names neither the chat nor its message.

Open WebUI names a temporary chat "temporary:" + the browser's socket id, and the snapshot's message id names a
reply inside that chat, so a temporary chat's snapshot carries neither id. Saved chats and channels keep both.
The snapshot writer is not part of the dashboard plugin, so this module also runs in the no-plugins bundle; the
usage row's half is in test_a_temporary_chats_spend_is_recorded_without_its_chat.py.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from open_webui_openrouter_pipe.core.costs import maybe_dump_costs_snapshot

MODEL = "openai/gpt-5.4"
USAGE = {"input_tokens": 7, "output_tokens": 3, "cost": 0.25}


class _FakeRedis:
    def __init__(self) -> None:
        self.closed = False
        self.writes: list[tuple[str, dict, int | None]] = []

    def set(self, key, payload, ex=None):
        self.writes.append((key, json.loads(payload), ex))
        return True

    async def aclose(self) -> None:
        self.closed = True


async def _through_the_pipe(monkeypatch, pipe, *, messages, chat_id, model=MODEL, usage=None, metadata=None):
    import open_webui_openrouter_pipe.integrations.image_catalog as image_catalog
    import open_webui_openrouter_pipe.integrations.video_catalog as video_catalog
    import open_webui_openrouter_pipe.pipe as pipe_mod
    import open_webui_openrouter_pipe.requests.orchestrator as orchestrator_module
    from open_webui_openrouter_pipe import EncryptedStr, Pipe

    async def upstream(self, session, request_body, **_kwargs):
        yield {"type": "response.output_text.delta", "delta": "Notes."}
        yield {"type": "response.completed", "response": {"output": [], "usage": dict(usage or {})}}

    async def loaded(*_a, **_k):
        return None

    async def user_by_id(user_id, _logger):
        return SimpleNamespace(id=user_id, role="user", email="sam@example.com", name="Sam")

    monkeypatch.setattr(orchestrator_module, "get_user_by_id", user_by_id)
    monkeypatch.setattr(pipe_mod, "_OwuiConfig", None)
    monkeypatch.setattr(pipe, "_maybe_start_startup_checks", lambda: None)
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", upstream)
    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test-key", None))
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", loaded)
    monkeypatch.setattr(image_catalog, "ensure_image_catalog_loaded", loaded)
    monkeypatch.setattr(video_catalog, "ensure_video_catalog_loaded", loaded)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "list_models",
                        lambda: [{"id": model, "name": "GPT", "norm_id": "openai.gpt-5.4"}])
    valves = pipe.valves.model_copy(update={"TOOL_EXECUTION_MODE": "Pipeline"})
    valves.API_KEY = EncryptedStr("sk-test-key")
    pipe.valves = valves

    async def emitter(_event):
        return None

    result = await pipe.pipe(
        body={"model": "openai.gpt-5.4", "stream": True, "messages": messages},
        __user__={"id": "u1", "role": "user"}, __request__=MagicMock(), __event_emitter__=emitter,
        __event_call__=None,
        __metadata__={"model": {"id": model}, "chat_id": chat_id, "message_id": "m1", **(metadata or {})},
        __tools__={}, __task__=None,
    )
    async for _ in cast(Any, result):
        pass
