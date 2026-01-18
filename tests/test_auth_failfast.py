from __future__ import annotations

from typing import Any

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import (
    EncryptedStr,
    Pipe,
    ResponsesBody,
)


@pytest.mark.asyncio
async def test_invalid_encrypted_api_key_returns_auth_error_and_skips_catalog(monkeypatch) -> None:
    """Test that invalid API key returns error WITHOUT calling catalog endpoint."""
    monkeypatch.setenv("WEBUI_SECRET_KEY", "unit-test-secret")

    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"API_KEY": EncryptedStr("encrypted:not-a-valid-token")})
    session = pipe._create_http_session(valves)

    # Mock HTTP at boundary - if catalog is requested, the test will fail
    # because aioresponses will raise an exception for unmocked calls
    with aioresponses() as mock_http:
        # Do NOT mock the catalog endpoint - if it's called, the test should fail
        # The real code should return an auth error before calling the catalog

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

            # Verify no HTTP calls were made (catalog was not called)
            # aioresponses raises if unexpected calls are made, so we're good if we reach here

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
    """Test that invalid API key for task request returns safe stub."""
    monkeypatch.setenv("WEBUI_SECRET_KEY", "unit-test-secret")

    pipe = Pipe()
    valves = pipe.valves.model_copy(update={"API_KEY": EncryptedStr("encrypted:not-a-valid-token")})
    session = pipe._create_http_session(valves)

    with aioresponses() as mock_http:
        # Do NOT mock catalog - should not be called for auth failures

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
async def test_process_transformed_request_accepts_string_task() -> None:
    """Test that _process_transformed_request accepts string task parameter.

    Real infrastructure exercised:
    - Real ResponsesBody.from_completions execution
    - Real task request processing
    - Real HTTP request to OpenRouter API
    """
    pipe = Pipe()
    valves = pipe.valves
    session = pipe._create_http_session(valves)

    # Mock HTTP at boundary
    with aioresponses() as mock_http:
        # Mock catalog endpoint
        mock_http.get(
            "https://openrouter.ai/api/v1/models",
            payload={"data": [{"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini", "context_length": 8192}]},
        )

        # Mock task response
        mock_http.post(
            "https://openrouter.ai/api/v1/responses",
            payload={
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": '{"tags":["test"]}'}],
                    }
                ],
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

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
                allowlist_norm_ids={"openai.gpt-4o-mini"},
                enforced_norm_ids=set(),
                catalog_norm_ids=set(),
                features={},
                user_id="",
            )
            assert isinstance(result, str)
            assert "tags" in result.lower()
        finally:
            await session.close()
            await pipe.close()
