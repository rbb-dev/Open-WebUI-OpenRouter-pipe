from __future__ import annotations

import asyncio
import logging
import types
from decimal import Decimal
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import (
    EncryptedStr,
    OpenRouterAPIError,
    Pipe,
    CompletionsBody,
    ResponsesBody,
    ModelFamily,
    generate_item_id,
    _serialize_marker,
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
        assert pipe.id == "open_webui_openrouter_pipe"
    finally:
        pipe.shutdown()


@pytest.mark.asyncio
async def test_pipe_handles_job_failure(monkeypatch):
    """Test that job failures in the worker are handled correctly.

    This test verifies the real worker infrastructure by:
    1. Letting a real job be enqueued
    2. Having the HTTP request fail
    3. Verifying error handling in the worker
    4. Checking error emission to UI

    Uses HTTP boundary mocking instead of mocking internal methods.
    """
    from aioresponses import aioresponses

    pipe = Pipe()

    # Configure API key so auth succeeds
    pipe.valves.API_KEY = EncryptedStr(EncryptedStr.encrypt("test-api-key"))

    events: list[dict[str, Any]] = []

    async def emitter(event: dict[str, Any]) -> None:
        events.append(event)

    # Mock HTTP to fail - this tests real queue/worker error handling
    with aioresponses() as mock_http:
        # Mock model catalog endpoint (required for startup)
        mock_http.get(
            "https://openrouter.ai/api/v1/models",
            payload={
                "data": [
                    {
                        "id": "openrouter/test-model",
                        "name": "Test Model",
                        "pricing": {"prompt": "0", "completion": "0"},
                        "context_length": 4096,
                    }
                ]
            },
            repeat=True,
        )

        # Mock the actual request to fail with network error
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            exception=RuntimeError("Network failure"),
        )

        try:
            result = await pipe.pipe(
                body={"model": "openrouter/test-model", "messages": [{"role": "user", "content": "test"}]},
                __user__={"valves": {}},
                __request__=None,
                __event_emitter__=emitter,
                __event_call__=None,
                __metadata__={},
                __tools__=None,
            )

            # When HTTP fails, the real pipeline returns empty string and emits error via events
            assert result == ""
            assert events, "Expected error events to be emitted"

            # Verify error was emitted to UI via chat:completion event
            completion_events = [e for e in events if e.get("type") == "chat:completion"]
            assert completion_events, f"Expected chat:completion event, got: {events}"

            # Verify the completion event contains the error
            error_data = completion_events[0].get("data", {}).get("error", {})
            assert error_data.get("message") == "Error: Network failure"
            assert completion_events[0].get("data", {}).get("done") is True
        finally:
            await pipe.close()


@pytest.mark.asyncio
async def test_pipe_coerces_none_metadata(monkeypatch):
    """Test that the pipe handles None metadata without crashing.

    This test verifies the real worker infrastructure by:
    1. Passing None as metadata (edge case)
    2. Having the HTTP request succeed
    3. Verifying the pipeline processes it correctly
    4. Checking proper response handling

    Uses HTTP boundary mocking instead of mocking internal methods.
    """
    from aioresponses import aioresponses

    pipe = Pipe()

    # Configure API key
    pipe.valves.API_KEY = EncryptedStr(EncryptedStr.encrypt("test-api-key"))

    # Mock HTTP with successful response
    with aioresponses() as mock_http:
        # Mock model catalog
        mock_http.get(
            "https://openrouter.ai/api/v1/models",
            payload={"data": [{"id": "openrouter/test-model"}]},
            repeat=True,
        )

        # Mock successful response
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload={
                "id": "response-123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "test response"}],
                    }
                ],
            },
        )

        try:
            result = await pipe.pipe(
                body={"model": "openrouter/test-model", "messages": [{"role": "user", "content": "test"}]},
                __user__={"valves": {}},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__=None,  # type: ignore[arg-type]
                __tools__=None,
            )

            # Verify it doesn't crash and returns a response
            assert result is not None
            assert isinstance(result, (dict, str))
        finally:
            await pipe.close()


@pytest.mark.asyncio
async def test_handle_pipe_call_tolerates_metadata_model_none(monkeypatch):
    """Test that pipe handles None model in metadata without crashing.

    This test verifies the real infrastructure by:
    1. Passing metadata with model=None (edge case)
    2. Letting real registry load catalog
    3. Having the HTTP request succeed
    4. Verifying proper handling throughout pipeline

    Uses HTTP boundary mocking instead of mocking internal methods.
    """
    from aioresponses import aioresponses

    pipe = Pipe()

    # Configure API key
    pipe.valves.API_KEY = EncryptedStr(EncryptedStr.encrypt("test-api-key"))

    # Mock HTTP with successful response
    with aioresponses() as mock_http:
        # Mock model catalog - registry will load this
        mock_http.get(
            "https://openrouter.ai/api/v1/models",
            payload={"data": [{"id": "openrouter/test-model"}]},
            repeat=True,
        )

        # Mock successful response
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload={
                "id": "response-123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "test response"}],
                    }
                ],
            },
        )

        try:
            result = await pipe.pipe(
                body={"model": "openrouter/test-model", "messages": [{"role": "user", "content": "test"}]},
                __user__={"valves": {}},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__={"model": None},  # Edge case: None model in metadata
                __tools__=None,
            )

            # Verify it doesn't crash and returns a response
            assert result is not None
            assert isinstance(result, (dict, str))
        finally:
            await pipe.close()


@pytest.mark.asyncio
async def test_run_streaming_loop_tolerates_null_item(monkeypatch):
    """Test that streaming loop handles null items in SSE events without crashing.

    This test verifies the real infrastructure by:
    1. Mocking HTTP SSE stream with null items
    2. Letting real streaming parser handle the events
    3. Verifying null items don't cause crashes
    4. Checking proper completion

    Uses HTTP boundary mocking instead of mocking internal methods.
    """
    from aioresponses import aioresponses

    pipe = Pipe()

    # Configure API key
    pipe.valves.API_KEY = EncryptedStr(EncryptedStr.encrypt("test-api-key"))

    # Build SSE response with null items
    sse_response = (
        b"event: response.output_item.added\n"
        b"data: {\"type\": \"response.output_item.added\", \"item\": null}\n\n"
        b"event: response.output_item.done\n"
        b"data: {\"type\": \"response.output_item.done\", \"item\": null}\n\n"
        b"event: response.completed\n"
        b'data: {"type": "response.completed", "response": {"output": [], "usage": {}}}\n\n'
    )

    # Mock HTTP with SSE stream containing null items
    with aioresponses() as mock_http:
        # Mock model catalog
        mock_http.get(
            "https://openrouter.ai/api/v1/models",
            payload={"data": [{"id": "openrouter/test-model"}]},
            repeat=True,
        )

        # Mock SSE streaming response with null items
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            body=sse_response,
            headers={"Content-Type": "text/event-stream"},
        )

        try:
            result = await pipe.pipe(
                body={"model": "openrouter/test-model", "messages": [{"role": "user", "content": "test"}], "stream": True},
                __user__={"valves": {}},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__={},
                __tools__=None,
            )

            # For streaming, pipe() returns an async generator
            # Consume it to verify null items don't cause crashes
            chunks = []
            async for chunk in result:  # type: ignore[misc]
                chunks.append(chunk)

            # With null items and no content, we get minimal chunks or empty list
            assert isinstance(chunks, list)
        finally:
            await pipe.close()


@pytest.mark.asyncio
async def test_from_completions_preserves_system_message_in_input():
    pipe = Pipe()
    try:
        completions_body = CompletionsBody(
            model="openrouter/test",
            messages=[
                {"role": "system", "content": "You're an AI assistant."},
                {"role": "user", "content": "ping"},
            ],
            stream=True,
        )
        responses_body = await ResponsesBody.from_completions(
            completions_body,
            transformer_context=pipe,
        )
        assert responses_body.instructions is None
        assert isinstance(responses_body.input, list)

        system_items = [
            item
            for item in responses_body.input
            if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "system"
        ]
        assert system_items
        assert system_items[0]["content"][0]["type"] == "input_text"
        assert system_items[0]["content"][0]["text"] == "You're an AI assistant."

        user_items = [
            item
            for item in responses_body.input
            if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "user"
        ]
        assert user_items
        assert user_items[0]["content"][0]["type"] == "input_text"
        assert user_items[0]["content"][0]["text"] == "ping"
    finally:
        pipe.shutdown()


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
        "open_webui_openrouter_pipe.pipe.aioredis",
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


def test_apply_task_reasoning_preferences_sets_effort():
    pipe = Pipe()

    # Set real model capabilities using ModelFamily.set_dynamic_specs
    ModelFamily.set_dynamic_specs({
        "test.model": {
            "supported_parameters": ["reasoning"]
        }
    })

    try:
        body = ResponsesBody(model="test.model", input=[])
        pipe._apply_task_reasoning_preferences(body, "high")
        assert body.reasoning == {"effort": "high", "enabled": True}
    finally:
        pipe.shutdown()


def test_apply_task_reasoning_preferences_include_only():
    pipe = Pipe()

    # Set real model capabilities using ModelFamily.set_dynamic_specs
    ModelFamily.set_dynamic_specs({
        "test.model": {
            "supported_parameters": ["include_reasoning"]
        }
    })

    try:
        body = ResponsesBody(model="test.model", input=[])
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


def test_apply_reasoning_preferences_prefers_reasoning_payload():
    pipe = Pipe()

    # Set real model capabilities using ModelFamily.set_dynamic_specs
    ModelFamily.set_dynamic_specs({
        "test.model": {
            "supported_parameters": ["reasoning", "include_reasoning"]
        }
    })

    try:
        body = ResponsesBody(model="test.model", input=[])
        body.include_reasoning = True
        pipe._apply_reasoning_preferences(body, pipe.Valves())
        assert isinstance(body.reasoning, dict)
        assert body.reasoning.get("enabled") is True
        assert body.include_reasoning is None
    finally:
        pipe.shutdown()


def test_apply_reasoning_preferences_legacy_only_sets_flag():
    pipe = Pipe()

    # Set real model capabilities using ModelFamily.set_dynamic_specs
    ModelFamily.set_dynamic_specs({
        "test.model": {
            "supported_parameters": ["include_reasoning"]
        }
    })

    try:
        body = ResponsesBody(model="test.model", input=[])
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


@pytest.mark.asyncio
async def test_transform_messages_skips_image_generation_artifacts(monkeypatch):
    pipe = Pipe()
    marker = generate_item_id()
    serialized = _serialize_marker(marker)

    async def fake_loader(chat_id, message_id, ulids):
        assert ulids == [marker]
        return {
            marker: {
                "type": "image_generation_call",
                "id": "img1",
                "status": "completed",
                "result": "data:image/png;base64,AAA",
            }
        }

    messages = [
        {
            "role": "assistant",
            "message_id": "assistant-1",
            "content": [
                {
                    "type": "output_text",
                    "text": f"Here is your image {serialized}",
                }
            ],
        },
        {
            "role": "user",
            "message_id": "user-2",
            "content": [{"type": "input_text", "text": "Please continue"}],
        },
    ]

    monkeypatch.setattr(
        ModelFamily,
        "supports",
        classmethod(lambda cls, feature, model_id: True),
    )

    try:
        result = await pipe.transform_messages_to_input(
            messages,
            chat_id="chat-1",
            openwebui_model_id="provider/model",
            artifact_loader=fake_loader,
        )
        assert all(item.get("type") != "image_generation_call" for item in result)
    finally:
        pipe.shutdown()


def test_apply_gemini_thinking_config_sets_level():
    pipe = Pipe()

    # Set real model capabilities using ModelFamily.set_dynamic_specs
    ModelFamily.set_dynamic_specs({
        "google.gemini-3-pro-image-preview": {
            "supported_parameters": ["reasoning"]
        }
    })

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


def test_apply_gemini_thinking_config_sets_budget():
    pipe = Pipe()

    # Set real model capabilities using ModelFamily.set_dynamic_specs
    ModelFamily.set_dynamic_specs({
        "google.gemini-2.5-flash": {
            "supported_parameters": ["reasoning"]
        }
    })

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
    """Test that task model reasoning valve applies to owned models.

    This test verifies the real infrastructure by:
    1. Configuring TASK_MODEL_REASONING_EFFORT valve
    2. Calling _process_transformed_request with a task
    3. Letting real _run_task_model_request execute with real HTTP
    4. Capturing HTTP request to verify reasoning parameter

    Uses HTTP boundary mocking instead of mocking internal methods.
    """
    from aioresponses import aioresponses

    pipe = Pipe()
    pipe.valves = pipe.Valves(
        API_KEY=EncryptedStr(EncryptedStr.encrypt("test-api-key")),
        TASK_MODEL_REASONING_EFFORT="high"
    )

    # Set real model capabilities using ModelFamily.set_dynamic_specs
    ModelFamily.set_dynamic_specs({
        "fake.model": {
            "supported_parameters": ["reasoning"]
        }
    })

    try:
        captured_body: dict[str, Any] = {}

        def capture_request(url, **kwargs):
            if "json" in kwargs:
                captured_body.update(kwargs["json"])

        with aioresponses() as mock_http:
            # Mock model catalog
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={"data": [{"id": "fake.model", "norm_id": "fake.model", "name": "Fake"}]},
                repeat=True,
            )

            # Mock task request - capture the body
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                payload={
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Generated title"}],
                        }
                    ]
                },
                callback=capture_request,
            )

            try:
                body = {"model": "fake.model", "messages": [{"role": "user", "content": "hi"}], "stream": True}
                metadata = {"chat_id": "c1", "message_id": "m1", "model": {"id": "fake.model"}}
                available_models = [{"id": "fake.model", "norm_id": "fake.model", "name": "Fake"}]
                allowed_models = pipe._select_models(pipe.valves.MODEL_ID, available_models)
                allowed_norm_ids = {m["norm_id"] for m in allowed_models}
                catalog_norm_ids = {m["norm_id"] for m in available_models}
                openwebui_model_id = metadata["model"]["id"]
                pipe_identifier = pipe.id

                # Create real session for HTTP requests
                session = pipe._create_http_session(pipe.valves)

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
                    session=session,
                    openwebui_model_id=openwebui_model_id,
                    pipe_identifier=pipe_identifier,
                    allowlist_norm_ids=allowed_norm_ids,
                    enforced_norm_ids=allowed_norm_ids,
                    catalog_norm_ids=catalog_norm_ids,
                    features={},
                )

                # Verify result
                assert result == "Generated title"

                # Verify reasoning parameter was applied
                assert captured_body.get("reasoning", {}).get("effort") == "high"
                assert captured_body["stream"] is False
            finally:
                await pipe.close()
    finally:
        # Reset dynamic specs
        ModelFamily.set_dynamic_specs({})



@pytest.mark.asyncio
async def test_task_reasoning_valve_skips_unowned_models(monkeypatch):
    """Test that task model reasoning valve does NOT apply to unowned models.

    This test verifies that TASK_MODEL_REASONING_EFFORT is skipped
    when the model is not in the pipe's catalog.

    Uses HTTP boundary mocking instead of mocking internal methods.
    """
    from aioresponses import aioresponses

    pipe = Pipe()
    pipe.valves = pipe.Valves(
        API_KEY=EncryptedStr(EncryptedStr.encrypt("test-api-key")),
        TASK_MODEL_REASONING_EFFORT="high"
    )

    # Set real model capabilities using ModelFamily.set_dynamic_specs
    ModelFamily.set_dynamic_specs({
        "unlisted.model": {
            "supported_parameters": ["reasoning"]
        }
    })

    try:
        captured_body: dict[str, Any] = {}

        def capture_request(url, **kwargs):
            if "json" in kwargs:
                captured_body.update(kwargs["json"])

        with aioresponses() as mock_http:
            # Mock catalog with DIFFERENT model
            mock_http.get(
                "https://openrouter.ai/api/v1/models",
                payload={"data": [{"id": "other.model", "norm_id": "other.model", "name": "Other"}]},
                repeat=True,
            )

            # Mock task request
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                payload={
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Generated title"}],
                        }
                    ]
                },
                callback=capture_request,
            )

            try:
                body = {"model": "unlisted.model", "messages": [{"role": "user", "content": "hi"}], "stream": True}
                metadata = {"chat_id": "c1", "message_id": "m1", "model": {"id": "unlisted.model"}}
                available_models = [{"id": "other.model", "norm_id": "other.model", "name": "Other"}]
                allowed_models = pipe._select_models(pipe.valves.MODEL_ID, available_models)
                allowed_norm_ids = {m["norm_id"] for m in allowed_models}
                catalog_norm_ids = {m["norm_id"] for m in available_models}
                openwebui_model_id = metadata["model"]["id"]
                pipe_identifier = pipe.id

                session = pipe._create_http_session(pipe.valves)

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
                    session=session,
                    openwebui_model_id=openwebui_model_id,
                    pipe_identifier=pipe_identifier,
                    allowlist_norm_ids=allowed_norm_ids,
                    enforced_norm_ids=allowed_norm_ids,
                    catalog_norm_ids=catalog_norm_ids,
                    features={},
                )

                assert result == "Generated title"

                # Verify task reasoning was NOT applied (falls back to REASONING_EFFORT)
                assert captured_body.get("reasoning", {}).get("effort") == pipe.valves.REASONING_EFFORT
                assert captured_body["stream"] is False
            finally:
                await pipe.close()
    finally:
        # Reset dynamic specs
        ModelFamily.set_dynamic_specs({})


@pytest.mark.asyncio
async def test_task_models_dump_costs_when_usage_available(monkeypatch):
    """Test that task models dump costs when usage is available.

    This test verifies the real infrastructure by:
    1. Calling _run_task_model_request directly
    2. Letting real HTTP request execute
    3. Verifying cost dumping is triggered with correct parameters

    Uses HTTP boundary mocking and spy pattern for cost dumping.
    """
    from aioresponses import aioresponses

    pipe = Pipe()
    pipe.valves = pipe.Valves(
        API_KEY=EncryptedStr(EncryptedStr.encrypt("test-api-key")),
        COSTS_REDIS_DUMP=True
    )

    captured: dict[str, Any] = {}

    async def fake_dump(self, valves, *, user_id, model_id, usage, user_obj, pipe_id):
        captured["user_id"] = user_id
        captured["model_id"] = model_id
        captured["usage"] = usage
        captured["pipe_id"] = pipe_id

    monkeypatch.setattr(
        Pipe,
        "_maybe_dump_costs_snapshot",
        fake_dump,
    )

    with aioresponses() as mock_http:
        # Mock model catalog
        mock_http.get(
            "https://openrouter.ai/api/v1/models",
            payload={"data": [{"id": "openai.gpt-mini", "name": "GPT Mini"}]},
            repeat=True,
        )

        # Mock task request with usage
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload={
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "auto title"}],
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        )

        try:
            session = pipe._create_http_session(pipe.valves)

            result = await pipe._run_task_model_request(
                {"model": "openai.gpt-mini"},
                pipe.valves,
                session=session,
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


@pytest.mark.asyncio
async def test_task_models_apply_identifier_valves_to_payload(monkeypatch):
    """Test that task models apply identifier valves to payload.

    This test verifies that SEND_* valves properly add identifiers
    to task model requests.

    Uses HTTP boundary mocking with callback to capture request params.
    """
    from aioresponses import aioresponses

    pipe = Pipe()
    pipe.valves = pipe.Valves(
        API_KEY=EncryptedStr(EncryptedStr.encrypt("test-api-key")),
        SEND_END_USER_ID=True,
        SEND_SESSION_ID=True,
        SEND_CHAT_ID=True,
        SEND_MESSAGE_ID=True,
    )

    captured: dict[str, Any] = {}

    def capture_request(url, **kwargs):
        if "json" in kwargs:
            captured["request_params"] = kwargs["json"]

    with aioresponses() as mock_http:
        # Mock model catalog
        mock_http.get(
            "https://openrouter.ai/api/v1/models",
            payload={"data": [{"id": "openai.gpt-mini", "name": "GPT Mini"}]},
            repeat=True,
        )

        # Mock task request - capture params
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload={
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ]
            },
            callback=capture_request,
        )

        try:
            session = pipe._create_http_session(pipe.valves)

            result = await pipe._run_task_model_request(
                {"model": "openai.gpt-mini"},
                pipe.valves,
                session=session,
                task_context={"type": "title"},
                owui_metadata={
                    "user_id": "u_meta",
                    "session_id": "s1",
                    "chat_id": "c1",
                    "message_id": "m1",
                },
                user_id="u123",
            )

            assert result == "ok"

            request_params = captured["request_params"]
            assert request_params.get("user") == "u123"
            assert request_params.get("session_id") == "s1"

            metadata = request_params.get("metadata")
            assert isinstance(metadata, dict)
            assert metadata.get("user_id") == "u123"
            assert metadata.get("session_id") == "s1"
            assert metadata.get("chat_id") == "c1"
            assert metadata.get("message_id") == "m1"
        finally:
            await pipe.close()


def test_sum_pricing_values_walks_all_nodes():
    pricing = {
        "prompt": "0",
        "completion": 0,
        "image": {"base": "0.5", "modifier": "-1.5"},
        "extra": ["2", "not-a-number", 1],
    }
    total, numeric_count = Pipe._sum_pricing_values(pricing)
    assert total == Decimal("2.0")
    assert numeric_count == 6


def test_free_model_requires_numeric_pricing(monkeypatch):
    pipe = Pipe()
    pipe.valves = pipe.Valves(FREE_MODEL_FILTER="only", TOOL_CALLING_FILTER="all")

    def fake_spec(cls, model_id: str):
        return {"pricing": {"prompt": "not-a-number", "completion": ""}}

    monkeypatch.setattr(
        "open_webui_openrouter_pipe.registry.OpenRouterModelRegistry.spec",
        classmethod(fake_spec),
    )

    try:
        assert pipe._is_free_model("weird.model") is False
    finally:
        pipe.shutdown()


def test_model_filters_respect_free_mode():
    pipe = Pipe()
    pipe.valves = pipe.Valves(FREE_MODEL_FILTER="only", TOOL_CALLING_FILTER="all")

    # Set real model specs directly on the registry
    from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry
    original_specs = OpenRouterModelRegistry._specs.copy()
    try:
        OpenRouterModelRegistry._specs = {
            "free.model": {"pricing": {"prompt": "0", "completion": "0"}},
            "paid.model": {"pricing": {"prompt": "1", "completion": "0"}},
        }

        models = [
            {"id": "free.model", "norm_id": "free.model", "name": "Free"},
            {"id": "paid.model", "norm_id": "paid.model", "name": "Paid"},
        ]
        filtered = pipe._apply_model_filters(models, pipe.valves)
        assert [m["norm_id"] for m in filtered] == ["free.model"]
    finally:
        OpenRouterModelRegistry._specs = original_specs
        pipe.shutdown()


@pytest.mark.asyncio
async def test_model_restricted_template_includes_filter_name():
    """Test that model restriction errors include the filter name in the error message.

    Real infrastructure exercised:
    - Real pipe() method execution
    - Real model catalog loading and filtering
    - Real model restriction checking
    - Real error templating and event emission
    - Real _process_transformed_request validation
    """
    from aioresponses import aioresponses

    pipe = Pipe()
    pipe.valves = pipe.Valves(
        API_KEY=EncryptedStr(EncryptedStr.encrypt("test-api-key")),
        FREE_MODEL_FILTER="only",
        TOOL_CALLING_FILTER="all",
    )

    events: list[dict[str, Any]] = []

    async def emitter(event: dict[str, Any]) -> None:
        events.append(event)

    # Mock HTTP catalog with both free and paid models
    with aioresponses() as mock_http:
        mock_http.get(
            "https://openrouter.ai/api/v1/models",
            payload={
                "data": [
                    {
                        "id": "free/model",
                        "name": "Free Model",
                        "pricing": {"prompt": "0", "completion": "0"},
                        "supported_parameters": ["tools"],
                    },
                    {
                        "id": "paid/model",
                        "name": "Paid Model",
                        "pricing": {"prompt": "0.001", "completion": "0.002"},
                        "supported_parameters": ["tools"],
                    },
                ]
            },
            repeat=True,
        )

        try:
            # Request paid model when only free models are allowed
            result = await pipe.pipe(
                body={
                    "model": "paid.model",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                },
                __user__={"id": "u1", "valves": {}},
                __request__=None,
                __event_emitter__=emitter,
                __event_call__=None,
                __metadata__={"model": {"id": "paid.model"}},
                __tools__={},
            )

            # Should return empty string (error handled via events)
            assert result == ""

            # Verify error event was emitted with filter name
            # Error is emitted as chat:message with the restriction details
            error_events = [e for e in events if e.get("type") == "chat:message" and "restricted" in e.get("data", {}).get("content", "").lower()]
            assert error_events, f"Expected error event, got events: {events}"

            # Check that the error message mentions the FREE_MODEL_FILTER valve
            error_text = str(error_events[0].get("data", {}).get("content", ""))
            # The error should mention either the filter causing the restriction or show the filter setting
            assert ("FREE_MODEL_FILTER" in error_text or "free" in error_text.lower()), (
                f"Expected error to mention FREE_MODEL_FILTER, got: {error_text}"
            )
        finally:
            await pipe.close()
