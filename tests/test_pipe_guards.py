from __future__ import annotations

import asyncio
import logging
import types
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import (
    OpenRouterAPIError,
    Pipe,
    ResponsesBody,
)


def test_wrap_safe_event_emitter_returns_none_for_missing():
    pipe = Pipe()
    try:
        assert pipe._wrap_safe_event_emitter(None) is None
    finally:
        pipe.shutdown()


@pytest.mark.asyncio
async def test_wrap_safe_event_emitter_swallows_transport_errors(caplog):
    pipe = Pipe()

    async def failing_emitter(_event):
        raise RuntimeError("boom")

    wrapped = pipe._wrap_safe_event_emitter(failing_emitter)
    assert wrapped is not None

    with caplog.at_level(logging.DEBUG):
        await wrapped({"type": "status"})

    try:
        assert any("Event emitter failure" in message for message in caplog.messages)
    finally:
        pipe.shutdown()


def test_resolve_pipe_identifier_returns_class_id():
    pipe = Pipe()
    try:
        result = pipe._resolve_pipe_identifier(None, fallback_model_id=None)
        assert result == "open_webui_openrouter_pipe"
    finally:
        pipe.shutdown()


@pytest.mark.asyncio
async def test_pipe_handles_job_failure(monkeypatch):
    pipe = Pipe()

    events: list[dict[str, Any]] = []

    async def emitter(event: dict[str, Any]) -> None:
        events.append(event)

    def fake_enqueue(self, job):
        job.future.set_exception(RuntimeError("boom"))
        return True

    monkeypatch.setattr(Pipe, "_enqueue_job", fake_enqueue)

    try:
        result = await pipe.pipe(
            body={},
            __user__={"valves": {}},
            __request__=None,
            __event_emitter__=emitter,
            __event_call__=None,
            __metadata__={},
            __tools__=None,
        )

        assert result == "Request failed. Please retry."
        assert events and events[0]["type"] == "chat:completion"
    finally:
        await pipe.close()


def test_merge_valves_no_overrides_returns_global():
    pipe = Pipe()
    try:
        baseline = pipe.Valves()
        merged = pipe._merge_valves(baseline, pipe.UserValves())
        assert merged is baseline
    finally:
        pipe.shutdown()


def test_merge_valves_applies_user_boolean_override():
    pipe = Pipe()
    try:
        user_valves = pipe.UserValves.model_validate({"ENABLE_REASONING": False})
        baseline = pipe.Valves()
        merged = pipe._merge_valves(baseline, user_valves)
        assert merged.ENABLE_REASONING is False
        assert merged.LOG_LEVEL == baseline.LOG_LEVEL
    finally:
        pipe.shutdown()


def test_merge_valves_honors_reasoning_retention_alias():
    pipe = Pipe()
    try:
        user_valves = pipe.UserValves.model_validate({"next_reply": "conversation"})
        merged = pipe._merge_valves(pipe.Valves(), user_valves)
        assert merged.PERSIST_REASONING_TOKENS == "conversation"
    finally:
        pipe.shutdown()


def test_redis_candidate_requires_full_multiworker_env(monkeypatch, caplog):
    monkeypatch.setenv("UVICORN_WORKERS", "4")
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("WEBSOCKET_MANAGER", "")
    monkeypatch.delenv("WEBSOCKET_REDIS_URL", raising=False)

    with caplog.at_level(logging.WARNING):
        pipe = Pipe()

    try:
        assert pipe._redis_candidate is False
        assert any("REDIS_URL is unset" in message for message in caplog.messages)
    finally:
        pipe.shutdown()


def test_redis_candidate_enabled_with_complete_env(monkeypatch):
    dummy_aioredis = types.SimpleNamespace(from_url=lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.aioredis",
        dummy_aioredis,
        raising=False,
    )

    monkeypatch.setenv("UVICORN_WORKERS", "3")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("WEBSOCKET_MANAGER", "redis")
    monkeypatch.setenv("WEBSOCKET_REDIS_URL", "redis://localhost:6380/0")

    pipe = Pipe()
    try:
        assert pipe._redis_candidate is True
    finally:
        pipe.shutdown()


def test_apply_task_reasoning_preferences_sets_effort(monkeypatch):
    pipe = Pipe()

    def fake_supported(cls, model_id):
        return frozenset({"reasoning"})

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelFamily.supported_parameters",
        classmethod(fake_supported),
    )

    try:
        body = ResponsesBody(model="fake", input=[])
        pipe._apply_task_reasoning_preferences(body, "high")
        assert body.reasoning == {"effort": "high", "enabled": True}
    finally:
        pipe.shutdown()


def test_apply_task_reasoning_preferences_include_only(monkeypatch):
    pipe = Pipe()

    def fake_supported(cls, model_id):
        return frozenset({"include_reasoning"})

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelFamily.supported_parameters",
        classmethod(fake_supported),
    )

    try:
        body = ResponsesBody(model="fake", input=[])
        pipe._apply_task_reasoning_preferences(body, "minimal")
        assert body.reasoning is None
        assert body.include_reasoning is False
        pipe._apply_task_reasoning_preferences(body, "none")
        assert body.include_reasoning is False
        pipe._apply_task_reasoning_preferences(body, "low")
        assert body.include_reasoning is True
    finally:
        pipe.shutdown()


def test_retry_without_reasoning_handles_thinking_error():
    pipe = Pipe()
    try:
        body = ResponsesBody(model="fake", input=[])
        body.include_reasoning = True
        body.thinking_config = {"include_thoughts": True}
        err = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="Unable to submit request because Thinking_config.include_thoughts is only enabled when thinking is enabled.",
        )
        assert pipe._should_retry_without_reasoning(err, body) is True
        assert body.include_reasoning is False
        assert body.reasoning is None
        assert body.thinking_config is None
    finally:
        pipe.shutdown()


def test_retry_without_reasoning_ignores_unrelated_errors():
    pipe = Pipe()
    try:
        body = ResponsesBody(model="fake", input=[])
        body.include_reasoning = True
        err = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="Some other provider issue",
        )
        assert pipe._should_retry_without_reasoning(err, body) is False
        assert body.include_reasoning is True
    finally:
        pipe.shutdown()


def test_apply_reasoning_preferences_prefers_reasoning_payload(monkeypatch):
    pipe = Pipe()

    def fake_supported(cls, model_id):
        return frozenset({"reasoning", "include_reasoning"})

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelFamily.supported_parameters",
        classmethod(fake_supported),
    )

    try:
        body = ResponsesBody(model="fake", input=[])
        body.include_reasoning = True
        pipe._apply_reasoning_preferences(body, pipe.Valves())
        assert isinstance(body.reasoning, dict)
        assert body.reasoning.get("enabled") is True
        assert body.include_reasoning is None
    finally:
        pipe.shutdown()


def test_apply_reasoning_preferences_legacy_only_sets_flag(monkeypatch):
    pipe = Pipe()

    def fake_supported(cls, model_id):
        return frozenset({"include_reasoning"})

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelFamily.supported_parameters",
        classmethod(fake_supported),
    )

    try:
        body = ResponsesBody(model="fake", input=[])
        valves = pipe.Valves(REASONING_EFFORT="high")
        pipe._apply_reasoning_preferences(body, valves)
        assert body.reasoning is None
        assert body.include_reasoning is True
        valves = pipe.Valves(REASONING_EFFORT="none")
        pipe._apply_reasoning_preferences(body, valves)
        assert body.include_reasoning is False
    finally:
        pipe.shutdown()


def test_retry_without_reasoning_handles_reasoning_dict():
    pipe = Pipe()
    try:
        body = ResponsesBody(model="fake", input=[])
        body.reasoning = {"enabled": True, "effort": "medium"}
        err = OpenRouterAPIError(
            status=400,
            reason="Bad Request",
            openrouter_message="Thinking_config.include_thoughts is only enabled when thinking is enabled.",
        )
        assert pipe._should_retry_without_reasoning(err, body) is True
        assert body.reasoning is None
    finally:
        pipe.shutdown()


def test_apply_gemini_thinking_config_sets_level(monkeypatch):
    pipe = Pipe()

    def fake_supported(cls, model_id):
        return frozenset({"reasoning"})

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelFamily.supported_parameters",
        classmethod(fake_supported),
    )

    try:
        valves = pipe.Valves(REASONING_EFFORT="high")
        body = ResponsesBody(model="google/gemini-3-pro-image-preview", input=[])
        pipe._apply_reasoning_preferences(body, valves)
        pipe._apply_gemini_thinking_config(body, valves)
        assert body.thinking_config == {"include_thoughts": True, "thinking_level": "HIGH"}
        assert body.reasoning is None
        assert body.include_reasoning is None
    finally:
        pipe.shutdown()


def test_apply_gemini_thinking_config_sets_budget(monkeypatch):
    pipe = Pipe()

    def fake_supported(cls, model_id):
        return frozenset({"reasoning"})

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelFamily.supported_parameters",
        classmethod(fake_supported),
    )

    try:
        valves = pipe.Valves(REASONING_EFFORT="medium", GEMINI_THINKING_BUDGET=512)
        body = ResponsesBody(model="google/gemini-2.5-flash", input=[])
        pipe._apply_reasoning_preferences(body, valves)
        pipe._apply_gemini_thinking_config(body, valves)
        assert body.thinking_config == {"include_thoughts": True, "thinking_budget": 512}
    finally:
        pipe.shutdown()


@pytest.mark.asyncio
async def test_task_reasoning_valve_applies_only_for_owned_models(monkeypatch):
    pipe = Pipe()
    pipe.valves = pipe.Valves(TASK_MODEL_REASONING_EFFORT="high")

    async def fake_ensure_loaded(*_, **__):
        return None

    def fake_list_models(cls):
        return [{"id": "fake.model", "norm_id": "fake.model", "name": "Fake"}]

    def fake_supported(cls, model_id):
        return frozenset({"reasoning"})

    captured: dict[str, Any] = {}

    async def fake_task_request(self, body, valves, *, session, task_context, **kwargs):
        captured["body"] = body
        captured["task_context"] = task_context
        return "ok"

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.OpenRouterModelRegistry.ensure_loaded",
        fake_ensure_loaded,
    )
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.OpenRouterModelRegistry.list_models",
        classmethod(fake_list_models),
    )
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelFamily.supported_parameters",
        classmethod(fake_supported),
    )
    monkeypatch.setattr(
        Pipe,
        "_run_task_model_request",
        fake_task_request,
    )

    try:
        body = {"model": "fake.model", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        metadata = {"chat_id": "c1", "message_id": "m1", "model": {"id": "fake.model"}}
        available_models = [{"id": "fake.model", "norm_id": "fake.model", "name": "Fake"}]
        allowed_models = pipe._select_models(pipe.valves.MODEL_ID, available_models)
        allowed_norm_ids = {m["norm_id"] for m in allowed_models}
        openwebui_model_id = metadata["model"]["id"]
        pipe_identifier = pipe._resolve_pipe_identifier(openwebui_model_id)
        result = await pipe._process_transformed_request(
            body,
            __user__={"id": "u1", "valves": {}},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__={"type": "title"},
            __task_body__=None,
            valves=pipe.valves,
            session=cast(Any, object()),
            openwebui_model_id=openwebui_model_id,
            pipe_identifier=pipe_identifier,
            pipe_identifier_for_artifacts=pipe_identifier,
            allowed_norm_ids=allowed_norm_ids,
            features={},
        )
        assert result == "ok"
        task_body = captured["body"]
        assert task_body["stream"] is True
        assert task_body.get("reasoning", {}).get("effort") == "high"
    finally:
        await pipe.close()


@pytest.mark.asyncio
async def test_task_reasoning_valve_skips_unowned_models(monkeypatch):
    pipe = Pipe()
    pipe.valves = pipe.Valves(TASK_MODEL_REASONING_EFFORT="high")

    async def fake_ensure_loaded(*_, **__):
        return None

    def fake_list_models(cls):
        return [{"id": "other.model", "norm_id": "other.model", "name": "Other"}]

    def fake_supported(cls, model_id):
        return frozenset({"reasoning"})

    captured: dict[str, Any] = {}

    async def fake_task_request(self, body, valves, *, session, task_context, **kwargs):
        captured["body"] = body
        return "ok"

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.OpenRouterModelRegistry.ensure_loaded",
        fake_ensure_loaded,
    )
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.OpenRouterModelRegistry.list_models",
        classmethod(fake_list_models),
    )
    monkeypatch.setattr(
        "open_webui_openrouter_pipe.open_webui_openrouter_pipe.ModelFamily.supported_parameters",
        classmethod(fake_supported),
    )
    monkeypatch.setattr(
        Pipe,
        "_run_task_model_request",
        fake_task_request,
    )

    try:
        body = {"model": "unlisted.model", "messages": [{"role": "user", "content": "hi"}], "stream": True}
        metadata = {"chat_id": "c1", "message_id": "m1", "model": {"id": "unlisted.model"}}
        available_models = [{"id": "other.model", "norm_id": "other.model", "name": "Other"}]
        allowed_models = pipe._select_models(pipe.valves.MODEL_ID, available_models)
        allowed_norm_ids = {m["norm_id"] for m in allowed_models}
        openwebui_model_id = metadata["model"]["id"]
        pipe_identifier = pipe._resolve_pipe_identifier(openwebui_model_id)
        result = await pipe._process_transformed_request(
            body,
            __user__={"id": "u1", "valves": {}},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__=metadata,
            __tools__=None,
            __task__={"type": "title"},
            __task_body__=None,
            valves=pipe.valves,
            session=cast(Any, object()),
            openwebui_model_id=openwebui_model_id,
            pipe_identifier=pipe_identifier,
            pipe_identifier_for_artifacts=pipe_identifier,
            allowed_norm_ids=allowed_norm_ids,
            features={},
        )
        assert result == "ok"
        task_body = captured["body"]
        assert task_body.get("reasoning", {}).get("effort") == pipe.valves.REASONING_EFFORT
        assert task_body["stream"] is True
    finally:
        await pipe.close()


@pytest.mark.asyncio
async def test_task_models_dump_costs_when_usage_available(monkeypatch):
    pipe = Pipe()
    pipe.valves = pipe.Valves(COSTS_REDIS_DUMP=True)
    captured: dict[str, Any] = {}

    async def fake_send(self, session, request_params, api_key, base_url):
        return {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "auto title"}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 2},
        }

    async def fake_dump(self, valves, *, user_id, model_id, usage, user_obj, pipe_id):
        captured["user_id"] = user_id
        captured["model_id"] = model_id
        captured["usage"] = usage
        captured["pipe_id"] = pipe_id

    monkeypatch.setattr(
        Pipe,
        "send_openai_responses_nonstreaming_request",
        fake_send,
    )
    monkeypatch.setattr(
        Pipe,
        "_maybe_dump_costs_snapshot",
        fake_dump,
    )

    try:
        result = await pipe._run_task_model_request(
            {"model": "openai.gpt-mini"},
            pipe.valves,
            session=cast(Any, object()),
            task_context={"type": "title"},
            user_id="u123",
            user_obj={"email": "user@example.com", "name": "User"},
            pipe_id="openrouter_responses_api_pipe",
            snapshot_model_id="openrouter_responses_api_pipe.openai.gpt-mini",
        )
        assert result == "auto title"
        assert captured["user_id"] == "u123"
        assert captured["model_id"] == "openrouter_responses_api_pipe.openai.gpt-mini"
        assert captured["pipe_id"] == "openrouter_responses_api_pipe"
        assert captured["usage"] == {"input_tokens": 5, "output_tokens": 2}
    finally:
        await pipe.close()
