from __future__ import annotations

from typing import Any

import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import (
    EncryptedStr,
    OpenRouterModelRegistry,
    Pipe,
    ResponsesBody,
)


@pytest.mark.asyncio
async def test_invalid_encrypted_api_key_returns_auth_error_and_skips_catalog(monkeypatch) -> None:
    monkeypatch.setenv("WEBUI_SECRET_KEY", "unit-test-secret")

    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"API_KEY": EncryptedStr("encrypted:not-a-valid-token")})
    session = pipe._create_http_session(valves)
    catalog_called = False

    async def fake_ensure_loaded(*_args: Any, **_kwargs: Any) -> None:
        nonlocal catalog_called
        catalog_called = True

    monkeypatch.setattr(OpenRouterModelRegistry, "ensure_loaded", fake_ensure_loaded)

    try:
        result = await pipe._handle_pipe_call(
            {"model": "openai/gpt-4o-mini", "stream": False},
            __user__={},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__={"model": {"id": "openai/gpt-4o-mini"}},
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=valves,
            session=session,
        )

        assert not catalog_called
        assert isinstance(result, dict)
        content = ((result.get("choices") or [{}])[0].get("message") or {}).get("content")
        assert isinstance(content, str)
        assert "Authentication Failed" in content
        assert "cannot be decrypted" in content
    finally:
        await session.close()
        await pipe.close()


@pytest.mark.asyncio
async def test_invalid_encrypted_api_key_task_returns_safe_stub(monkeypatch) -> None:
    monkeypatch.setenv("WEBUI_SECRET_KEY", "unit-test-secret")

    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"API_KEY": EncryptedStr("encrypted:not-a-valid-token")})
    session = pipe._create_http_session(valves)

    try:
        result = await pipe._handle_pipe_call(
            {"model": "openai/gpt-4o-mini", "stream": False},
            __user__={},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__={"model": {"id": "openai/gpt-4o-mini"}},
            __tools__=None,
            __task__="tags_generation",
            __task_body__={},
            valves=valves,
            session=session,
        )

        assert isinstance(result, dict)
        content = ((result.get("choices") or [{}])[0].get("message") or {}).get("content")
        assert isinstance(content, str)
        assert "\"tags\"" in content
    finally:
        await session.close()
        await pipe.close()


@pytest.mark.asyncio
async def test_process_transformed_request_accepts_string_task(monkeypatch) -> None:
    pipe = Pipe()
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    async def fake_from_completions(  # type: ignore[override]
        cls,  # noqa: ANN001 - classmethod signature
        *_args: Any,
        **_kwargs: Any,
    ) -> ResponsesBody:
        return ResponsesBody.model_validate({"model": "openai/gpt-4o-mini", "input": [], "stream": False})

    async def fake_run_task_model_request(*_args: Any, **_kwargs: Any) -> str:
        return "ok"

    monkeypatch.setattr(ResponsesBody, "from_completions", classmethod(fake_from_completions))
    monkeypatch.setattr(Pipe, "_run_task_model_request", fake_run_task_model_request)

    try:
        result = await pipe._process_transformed_request(
            {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            __user__={},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__={},
            __tools__=None,
            __task__="tags_generation",
            __task_body__={},
            valves=valves,
            session=session,
            openwebui_model_id="",
            pipe_identifier="pipe",
            allowlist_norm_ids={"some-other-model"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
            user_id="",
        )
        assert result == "ok"
    finally:
        await session.close()
        await pipe.close()

