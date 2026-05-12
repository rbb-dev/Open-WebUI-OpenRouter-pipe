"""Unit tests for the structured_task package (LLM task orchestration).

Mirrors patterns lifted from .external/seedream.py. These functions are
feature-agnostic — used by video intent classifier and (future) image
intent classifier.
"""
from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from open_webui_openrouter_pipe.structured_task import (
    build_response_format,
    call_with_candidates,
    consume_sse_line,
    downgrade_strict_for_provider,
    normalise_model_content,
    read_task_model_response_json,
    resolve_task_model_candidates,
    safe_log_payload,
)


def _fake_request(task_model: str = "", task_model_external: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                config=SimpleNamespace(
                    TASK_MODEL=task_model,
                    TASK_MODEL_EXTERNAL=task_model_external,
                ),
                MODELS={},
            ),
        ),
    )


class TestResolveCandidates:
    def test_internal_mode_picks_TASK_MODEL(self):
        req = _fake_request(task_model="local-llm", task_model_external="external-llm")
        result = resolve_task_model_candidates(
            request=req, mode="internal", fallback="none",
        )
        assert "local-llm" in result

    def test_external_mode_picks_TASK_MODEL_EXTERNAL(self):
        req = _fake_request(task_model="local-llm", task_model_external="external-llm")
        result = resolve_task_model_candidates(
            request=req, mode="external", fallback="none",
        )
        assert "external-llm" in result

    def test_fallback_other_task_model_appends(self):
        req = _fake_request(task_model="local", task_model_external="ext")
        result = resolve_task_model_candidates(
            request=req, mode="external", fallback="other_task_model",
        )
        assert result.index("ext") < result.index("local")

    def test_fallback_none_returns_only_primary(self):
        req = _fake_request(task_model_external="ext")
        result = resolve_task_model_candidates(
            request=req, mode="external", fallback="none",
        )
        assert result == ["ext"]

    def test_dedupes_when_internal_equals_external(self):
        req = _fake_request(task_model="same", task_model_external="same")
        result = resolve_task_model_candidates(
            request=req, mode="external", fallback="other_task_model",
        )
        assert result.count("same") == 1

    def test_returns_empty_when_unconfigured_no_fallback(self):
        req = _fake_request()
        result = resolve_task_model_candidates(
            request=req, mode="external", fallback="none",
        )
        assert result == []


class TestBuildResponseFormat:
    def test_wraps_schema_with_strict_true(self):
        rf = build_response_format(name="my_schema", schema={"type": "object"}, strict=True)
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "my_schema"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"] == {"type": "object"}

    def test_strict_false_when_specified(self):
        rf = build_response_format(name="x", schema={}, strict=False)
        assert rf["json_schema"]["strict"] is False

    def test_downgrade_when_structured_outputs_unsupported(self):
        rf = build_response_format(name="x", schema={}, strict=True)
        downgraded = downgrade_strict_for_provider(rf, supported_parameters=[])
        assert downgraded["json_schema"]["strict"] is False

    def test_downgrade_no_op_when_structured_outputs_supported(self):
        rf = build_response_format(name="x", schema={}, strict=True)
        out = downgrade_strict_for_provider(rf, supported_parameters=["structured_outputs"])
        assert out["json_schema"]["strict"] is True


class TestSafeLogPayload:
    def test_redacts_messages_keeps_roles_count(self):
        form_data = {
            "model": "gpt-5",
            "temperature": 0,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x", "strict": True},
            },
            "messages": [
                {"role": "system", "content": "secret system"},
                {"role": "user", "content": "secret user"},
            ],
            "metadata": {"chat_id": "abc"},
        }
        redacted = safe_log_payload(form_data)
        assert redacted["messages"]["count"] == 2
        assert redacted["messages"]["roles"] == ["system", "user"]
        # No raw content should leak
        assert "secret" not in str(redacted)

    def test_handles_missing_response_format(self):
        redacted = safe_log_payload({"model": "x", "messages": []})
        assert redacted["response_format"]["type"] is None
        assert redacted["messages"]["count"] == 0

    def test_handles_invalid_messages_field(self):
        redacted = safe_log_payload({"messages": "not-a-list"})
        assert redacted["messages"]["count"] == 0


class TestNormaliseModelContent:
    def test_string_passthrough(self):
        assert normalise_model_content("hello") == "hello"

    def test_list_of_text_parts_concatenates(self):
        parts = [
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "world"},
        ]
        assert "hello" in normalise_model_content(parts)
        assert "world" in normalise_model_content(parts)

    def test_none_returns_empty_string(self):
        assert normalise_model_content(None) == ""

    def test_dict_with_content_field(self):
        result = normalise_model_content({"content": "hello"})
        assert "hello" in result


class TestConsumeSseLine:
    def test_done_marker_ignored(self):
        parts: list[str] = []
        consume_sse_line("data: [DONE]", parts)
        assert parts == []

    def test_parses_choices_delta(self):
        parts: list[str] = []
        consume_sse_line(
            'data: {"choices":[{"delta":{"content":"hi"}}]}', parts,
        )
        assert parts == ["hi"]

    def test_ignores_non_data_line(self):
        parts: list[str] = []
        consume_sse_line("event: ping", parts)
        assert parts == []

    def test_handles_malformed_json(self):
        parts: list[str] = []
        consume_sse_line("data: {not-json", parts)
        assert parts == []


class TestReadTaskModelResponseJson:
    @pytest.mark.asyncio
    async def test_parses_dict_choice_message_content(self):
        response = {
            "choices": [
                {"message": {"content": '{"intent":"text_to_video"}'}},
            ],
        }
        result = await read_task_model_response_json(response)
        assert result["intent"] == "text_to_video"

    @pytest.mark.asyncio
    async def test_parses_dict_with_output_field(self):
        # OpenRouter Responses API style
        response = {
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": '{"intent":"x"}'}],
                },
            ],
        }
        result = await read_task_model_response_json(response)
        assert result["intent"] == "x"

    @pytest.mark.asyncio
    async def test_raises_on_empty_response(self):
        with pytest.raises(RuntimeError, match="empty"):
            await read_task_model_response_json({"choices": [{"message": {"content": ""}}]})

    @pytest.mark.asyncio
    async def test_raises_on_non_json_content(self):
        with pytest.raises(RuntimeError):
            await read_task_model_response_json(
                {"choices": [{"message": {"content": "not json at all"}}]}
            )


class TestCallWithCandidates:
    @pytest.mark.asyncio
    async def test_returns_first_success(self):
        async def invoke(form_data):
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

        result = await call_with_candidates(
            candidates=["m1", "m2"],
            build_form_data=lambda m: {"model": m, "messages": []},
            invoke=invoke,
            timeout_s=5.0,
            logger=MagicMock(),
        )
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_skips_to_next_on_first_failure(self):
        call_count = {"n": 0}

        async def invoke(form_data):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("first failed")
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

        result = await call_with_candidates(
            candidates=["m1", "m2"],
            build_form_data=lambda m: {"model": m, "messages": []},
            invoke=invoke,
            timeout_s=5.0,
            logger=MagicMock(),
        )
        assert result == {"ok": True}
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_fail(self):
        async def invoke(form_data):
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError):
            await call_with_candidates(
                candidates=["m1", "m2"],
                build_form_data=lambda m: {"model": m, "messages": []},
                invoke=invoke,
                timeout_s=5.0,
                logger=MagicMock(),
            )

    @pytest.mark.asyncio
    async def test_timeout_skips_to_next(self):
        attempts = {"n": 0}

        async def invoke(form_data):
            attempts["n"] += 1
            if attempts["n"] == 1:
                await asyncio.sleep(2.0)  # exceed timeout
                return {}
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

        result = await call_with_candidates(
            candidates=["m1", "m2"],
            build_form_data=lambda m: {"model": m, "messages": []},
            invoke=invoke,
            timeout_s=0.1,
            logger=MagicMock(),
        )
        assert result == {"ok": True}

    @pytest.mark.asyncio
    async def test_empty_candidates_raises(self):
        async def invoke(form_data):
            return {}

        with pytest.raises(RuntimeError):
            await call_with_candidates(
                candidates=[],
                build_form_data=lambda m: {},
                invoke=invoke,
                timeout_s=5.0,
                logger=MagicMock(),
            )
