from __future__ import annotations

import asyncio
import logging
import types

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe, ResponsesBody


def test_wrap_safe_event_emitter_returns_none_for_missing():
    pipe = Pipe()
    try:
        assert pipe._wrap_safe_event_emitter(None) is None
    finally:
        pipe.shutdown()


def test_wrap_safe_event_emitter_swallows_transport_errors(caplog):
    pipe = Pipe()

    async def failing_emitter(_event):
        raise RuntimeError("boom")

    wrapped = pipe._wrap_safe_event_emitter(failing_emitter)

    with caplog.at_level(logging.DEBUG):
        asyncio.run(wrapped({"type": "status"}))

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


def test_pipe_handles_job_failure(monkeypatch):
    async def runner():
        pipe = Pipe()

        events: list[dict] = []

        async def emitter(event):
            events.append(event)

        def fake_enqueue(self, job):
            job.future.set_exception(RuntimeError("boom"))
            return True

        monkeypatch.setattr(Pipe, "_enqueue_job", fake_enqueue)

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
        await pipe.close()

    asyncio.run(runner())


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
        pipe._apply_task_reasoning_preferences(body, "low")
        assert body.include_reasoning is True
    finally:
        pipe.shutdown()


def test_task_reasoning_valve_applies_only_for_owned_models(monkeypatch):
    async def runner():
        pipe = Pipe()
        pipe.valves = pipe.Valves(TASK_MODEL_REASONING_EFFORT="high")

        async def fake_ensure_loaded(*_, **__):
            return None

        def fake_list_models(cls):
            return [{"id": "fake.model", "norm_id": "fake.model", "name": "Fake"}]

        def fake_supported(cls, model_id):
            return frozenset({"reasoning"})

        captured: dict = {}

        async def fake_task_request(self, body, valves, *, session, task_context):
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
                session=object(),
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
            await pipe.close()
        finally:
            pipe.shutdown()

    asyncio.run(runner())


def test_task_reasoning_valve_skips_unowned_models(monkeypatch):
    async def runner():
        pipe = Pipe()
        pipe.valves = pipe.Valves(TASK_MODEL_REASONING_EFFORT="high")

        async def fake_ensure_loaded(*_, **__):
            return None

        def fake_list_models(cls):
            return [{"id": "other.model", "norm_id": "other.model", "name": "Other"}]

        def fake_supported(cls, model_id):
            return frozenset({"reasoning"})

        captured: dict = {}

        async def fake_task_request(self, body, valves, *, session, task_context):
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
                session=object(),
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
            await pipe.close()
        finally:
            pipe.shutdown()

    asyncio.run(runner())
