"""Integration + invariant regression tests for the video intent classifier.

Covers:
- resolve_intent never raises (expanded)
- CancelledError propagation (resolve_intent + call_with_candidates)
- _intent_classifier_should_run short-circuit conditions
- streaming response branch in read_task_model_response_json
- _materialise_frame_plan integration paths
- _resolve_prior_video_file_id URL parser cases
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from open_webui_openrouter_pipe.integrations.video_intent import (
    FramePlanEntry,
    VideoIntentResult,
    resolve_intent,
)
from open_webui_openrouter_pipe.structured_task import call_with_candidates


def _fake_request(task_model_external: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(
            config=SimpleNamespace(TASK_MODEL="", TASK_MODEL_EXTERNAL=task_model_external),
            MODELS={},
        )),
    )


def _make_valves(**overrides) -> SimpleNamespace:
    defaults = dict(
        VIDEO_INTENT_ENABLED=True,
        VIDEO_INTENT_MAX_CLARIFICATIONS=1,
        VIDEO_INTENT_TASK_MODEL_MODE="external",
        VIDEO_INTENT_TASK_MODEL_FALLBACK="none",
        VIDEO_INTENT_TIMEOUT_S=5,
        VIDEO_INTENT_SKIP_WHEN_EMPTY_CHAT=True,
        VIDEO_INTENT_LOG_DECISIONS=False,
        VIDEO_INTENT_MAX_CALLS_PER_CHAT=0,
        VIDEO_INTENT_MAX_CALLS_PER_USER_DAY=0,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# -----------------------------------------------------------------------------
# resolve_intent never raises (expanded coverage)
# -----------------------------------------------------------------------------

class TestResolveIntentNeverRaises:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("body", [None, {}, [], "garbage", {"messages": "not-a-list"}])
    async def test_handles_invalid_body(self, body):
        result = await resolve_intent(
            body=body, video_meta={}, video_model={},
            valves=_make_valves(), request=_fake_request("task-llm"),
            user_obj=None, chat_id="c1", logger=logging.getLogger("test"),
            fallback_prompt_text="hi",
        )
        assert result.intent == "text_to_video"
        assert result.frame_plan == []

    @pytest.mark.asyncio
    async def test_handles_none_video_meta(self):
        result = await resolve_intent(
            body={"messages": [{"role": "user", "content": "hi"}]},
            video_meta=None, video_model={},  # type: ignore[arg-type]
            valves=_make_valves(), request=_fake_request("task-llm"),
            user_obj=None, chat_id="c1", logger=logging.getLogger("test"),
            fallback_prompt_text="hi",
        )
        assert result.intent == "text_to_video"

    @pytest.mark.asyncio
    async def test_handles_none_video_model(self):
        result = await resolve_intent(
            body={"messages": [{"role": "user", "content": "hi"}]},
            video_meta={}, video_model=None,  # type: ignore[arg-type]
            valves=_make_valves(), request=_fake_request("task-llm"),
            user_obj=None, chat_id="c1", logger=logging.getLogger("test"),
            fallback_prompt_text="hi",
        )
        assert result.intent == "text_to_video"

    @pytest.mark.asyncio
    async def test_handles_invoke_returning_list_not_dict(self):
        async def bad(form_data):
            return ["not", "a", "dict"]
        result = await resolve_intent(
            body={"messages": [{"role": "user", "content": "hi"}]},
            video_meta={}, video_model={},
            valves=_make_valves(), request=_fake_request("task-llm"),
            user_obj=None, chat_id="c1", logger=logging.getLogger("test"),
            invoke_chat_completion=bad, fallback_prompt_text="hi",
        )
        assert result.intent == "text_to_video"


# -----------------------------------------------------------------------------
# CancelledError propagation
# -----------------------------------------------------------------------------

class TestCancelledErrorPropagation:
    @pytest.mark.asyncio
    async def test_resolve_intent_re_raises_cancelled(self):
        async def boom(form_data):
            raise asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await resolve_intent(
                body={"messages": [{"role": "user", "content": "hi"}]},
                video_meta={}, video_model={},
                valves=_make_valves(), request=_fake_request("task-llm"),
                user_obj=None, chat_id="c1", logger=logging.getLogger("test"),
                invoke_chat_completion=boom, fallback_prompt_text="hi",
            )

    @pytest.mark.asyncio
    async def test_call_with_candidates_re_raises_cancelled(self):
        async def boom(form_data):
            raise asyncio.CancelledError()
        with pytest.raises(asyncio.CancelledError):
            await call_with_candidates(
                candidates=["m1"],
                build_form_data=lambda m: {"model": m, "messages": []},
                invoke=boom, timeout_s=5.0, logger=MagicMock(),
            )


# -----------------------------------------------------------------------------
# short-circuit conditions exhaustive
# -----------------------------------------------------------------------------

class TestShortCircuit:
    def _make_adapter(self):
        from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
        pipe = MagicMock()
        return VideoGenerationAdapter(pipe=pipe, logger=logging.getLogger("test"))

    def test_master_valve_off_returns_false(self):
        adapter = self._make_adapter()
        assert not adapter._intent_classifier_should_run(
            valves=_make_valves(VIDEO_INTENT_ENABLED=False),
            persisted_content="", prompt="hi",
            body={"messages": [{}, {}]}, video_meta={},
        )

    def test_empty_prompt_returns_false(self):
        adapter = self._make_adapter()
        assert not adapter._intent_classifier_should_run(
            valves=_make_valves(),
            persisted_content="", prompt="   ",
            body={"messages": [{}, {}]}, video_meta={},
        )

    def test_help_prompt_returns_false(self):
        adapter = self._make_adapter()
        assert not adapter._intent_classifier_should_run(
            valves=_make_valves(),
            persisted_content="", prompt="help",
            body={"messages": [{}, {}]}, video_meta={},
        )

    def test_resume_marker_returns_false(self):
        adapter = self._make_adapter()
        assert not adapter._intent_classifier_should_run(
            valves=_make_valves(),
            persisted_content="[openrouter:v1:videojob:abc123]: #",
            prompt="hi",
            body={"messages": [{}, {}]}, video_meta={},
        )

    def test_explicit_frame_images_no_longer_short_circuits(self):
        adapter = self._make_adapter()
        assert adapter._intent_classifier_should_run(
            valves=_make_valves(),
            persisted_content="", prompt="hi",
            body={"messages": [{}, {}]},
            video_meta={"frame_images": [{"id": "x"}]},
        )

    def test_empty_chat_returns_false_when_skip_valve_on(self):
        adapter = self._make_adapter()
        assert not adapter._intent_classifier_should_run(
            valves=_make_valves(VIDEO_INTENT_SKIP_WHEN_EMPTY_CHAT=True),
            persisted_content="", prompt="hi",
            body={"messages": [{"role": "user", "content": "hi"}]},
            video_meta={},
        )

    def test_per_chat_cap_exceeded_returns_false(self):
        adapter = self._make_adapter()
        adapter._intent_call_counts_per_chat["chat1"] = 5
        assert not adapter._intent_classifier_should_run(
            valves=_make_valves(VIDEO_INTENT_MAX_CALLS_PER_CHAT=5),
            persisted_content="", prompt="hi",
            body={"messages": [{}, {}]}, video_meta={}, chat_id="chat1",
        )

    def test_breaker_open_returns_false(self):
        adapter = self._make_adapter()
        adapter._intent_breaker_until_ts = time.time() + 60
        assert not adapter._intent_classifier_should_run(
            valves=_make_valves(),
            persisted_content="", prompt="hi",
            body={"messages": [{}, {}]}, video_meta={},
        )

    def test_happy_path_returns_true(self):
        adapter = self._make_adapter()
        assert adapter._intent_classifier_should_run(
            valves=_make_valves(),
            persisted_content="", prompt="make a video",
            body={"messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "make a video"},
            ]},
            video_meta={},
        )


# -----------------------------------------------------------------------------
# streaming response branch
# -----------------------------------------------------------------------------

class TestStreamingResponseBranch:
    @pytest.mark.asyncio
    async def test_parses_streaming_response_with_body_iterator(self):
        from open_webui_openrouter_pipe.structured_task.client import (
            read_task_model_response_json,
        )

        async def _gen():
            yield b'data: {"choices":[{"delta":{"content":"{\\\"intent\\\":"}}]}\n\n'
            yield b'data: {"choices":[{"delta":{"content":"\\\"text_to_video\\\"}"}}]}\n\n'
            yield b'data: [DONE]\n\n'

        response = SimpleNamespace(body_iterator=_gen())
        result = await read_task_model_response_json(response)
        assert result["intent"] == "text_to_video"


# -----------------------------------------------------------------------------
# _materialise_frame_plan integration tests
# -----------------------------------------------------------------------------

class TestMaterialiseFramePlan:
    def _make_adapter_with_mocks(self):
        from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
        pipe = MagicMock()
        pipe._multimodal_handler = MagicMock()
        pipe._multimodal_handler._get_file_by_id = AsyncMock()
        pipe._multimodal_handler._upload_to_owui_storage = AsyncMock(return_value="uploaded_id")
        adapter = VideoGenerationAdapter(pipe=pipe, logger=logging.getLogger("test"))
        return adapter, pipe

    def _make_intent_with_prior_video(self):
        return VideoIntentResult(
            intent="modify_prior_video",
            frame_plan=[FramePlanEntry(
                source="prior_video_first_frame", source_index=0,
                timestamp_seconds=None, target="first_frame",
            )],
            prompt="x", use_user_prompt=False, language="en", confidence="high",
            clarification=None, reason="x",
            prior_videos=[{"index": 0, "file_url": "/api/v1/files/abc/content"}],
        )

    @pytest.mark.asyncio
    async def test_skip_input_reference_target(self):
        adapter, pipe = self._make_adapter_with_mocks()
        intent = VideoIntentResult(
            intent="image_to_video",
            frame_plan=[FramePlanEntry(
                source="prior_video_first_frame", source_index=0,
                timestamp_seconds=None, target="input_reference",
            )],
            prompt="x", use_user_prompt=False, language="en", confidence="high",
            clarification=None, reason="x",
            prior_videos=[{"index": 0, "file_url": "/api/v1/files/abc/content"}],
        )
        # Mock all collaborators so we get to the input_reference branch
        with patch.object(adapter, "_resolve_owui_file_path", AsyncMock(return_value=Path("/tmp/x"))), \
             patch("open_webui_openrouter_pipe.integrations.video.extract_frame", AsyncMock(return_value=SimpleNamespace(
                 image_bytes=b"fake_png", width=10, height=10, downgrade_note="",
             ))):
            video_meta: dict = {}
            await adapter._materialise_frame_plan(
                intent=intent, video_meta=video_meta,
                request=None, user_obj=SimpleNamespace(id="u1"),
                chat_id="c1", message_id="m1",
            )
        # input_reference target should NOT be added to frame_images
        # (those are hard-anchor slots in OR's API).
        assert "frame_images" not in video_meta or len(video_meta.get("frame_images", [])) == 0
        # It SHOULD be added to input_references (the OR top-level
        # style-reference channel).
        irs = video_meta.get("input_references", [])
        assert len(irs) == 1
        assert irs[0]["content_type"] == "image/png"

    @pytest.mark.asyncio
    async def test_unresolvable_file_id_drops_entry(self):
        adapter, _ = self._make_adapter_with_mocks()
        intent = self._make_intent_with_prior_video()
        intent.prior_videos = [{"index": 0, "file_url": "garbage"}]  # bad URL
        video_meta: dict = {}
        await adapter._materialise_frame_plan(
            intent=intent, video_meta=video_meta,
            request=None, user_obj=SimpleNamespace(id="u1"),
            chat_id="c1", message_id="m1",
        )
        assert "frame_images" not in video_meta or video_meta.get("frame_images") == []
        assert any("unresolvable" in d for d in intent.downgrades)

    @pytest.mark.asyncio
    async def test_extract_frame_failure_drops_entry(self):
        from open_webui_openrouter_pipe.media import FrameExtractionError
        adapter, _ = self._make_adapter_with_mocks()
        intent = self._make_intent_with_prior_video()
        with patch.object(adapter, "_resolve_owui_file_path", AsyncMock(return_value=Path("/tmp/x"))), \
             patch("open_webui_openrouter_pipe.integrations.video.extract_frame",
                   AsyncMock(side_effect=FrameExtractionError("codec not supported"))):
            video_meta: dict = {}
            await adapter._materialise_frame_plan(
                intent=intent, video_meta=video_meta,
                request=None, user_obj=SimpleNamespace(id="u1"),
                chat_id="c1", message_id="m1",
            )
        # Downgrade added, NO raw exception text in the downgrade code
        assert any("frame_extract_failed" in d for d in intent.downgrades)
        assert not any("codec not supported" in d for d in intent.downgrades)

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates(self):
        adapter, _ = self._make_adapter_with_mocks()
        intent = self._make_intent_with_prior_video()
        with patch.object(adapter, "_resolve_owui_file_path",
                          AsyncMock(side_effect=asyncio.CancelledError())):
            with pytest.raises(asyncio.CancelledError):
                await adapter._materialise_frame_plan(
                    intent=intent, video_meta={},
                    request=None, user_obj=SimpleNamespace(id="u1"),
                    chat_id="c1", message_id="m1",
                )


# -----------------------------------------------------------------------------
# _resolve_prior_video_file_id URL parser
# -----------------------------------------------------------------------------

class TestUrlParser:
    @pytest.mark.parametrize("url,expected", [
        ("/api/v1/files/abc/content", "abc"),
        ("/api/v1/files/abc/content?token=x", "abc"),
        ("/api/v1/files/abc?ts=1", "abc"),
        ("/api/v1/files/abc#frag", "abc"),
        ("/api/v1/files/abc", "abc"),
        ("https://example.com/api/v1/files/abc/content", "abc"),
        ("not-a-files-url", ""),
        ("", ""),
        ("/api/v1/files//content", ""),
    ])
    @pytest.mark.asyncio
    async def test_url_parser_cases(self, url, expected):
        from open_webui_openrouter_pipe.integrations.video import VideoGenerationAdapter
        pipe = MagicMock()
        adapter = VideoGenerationAdapter(pipe=pipe, logger=logging.getLogger("test"))
        intent = VideoIntentResult(
            intent="modify_prior_video",
            frame_plan=[FramePlanEntry(
                source="prior_video_first_frame", source_index=0,
                timestamp_seconds=None, target="first_frame",
            )],
            prompt="x", use_user_prompt=False, language="en", confidence="high",
            clarification=None, reason="x",
            prior_videos=[{"index": 0, "file_url": url}],
        )
        result = await adapter._resolve_prior_video_file_id(
            intent.frame_plan[0], intent=intent, user_obj=None,
        )
        assert result == expected


# -----------------------------------------------------------------------------
# Disclosure persistence — assert intent block in pending_content
# -----------------------------------------------------------------------------

class TestDisclosurePersistence:
    @pytest.mark.asyncio
    async def test_pending_content_includes_intent_block_when_present(self):
        from open_webui_openrouter_pipe.integrations.video_intent import (
            render_intent_disclosure_block,
        )
        intent = VideoIntentResult(
            intent="modify_prior_video",
            frame_plan=[FramePlanEntry(
                source="prior_video_first_frame", source_index=0,
                timestamp_seconds=None, target="first_frame",
            )],
            prompt="a black cat", use_user_prompt=False, language="en", confidence="high",
            clarification=None, reason="x",
        )
        block = render_intent_disclosure_block(intent, thumb_urls=["/api/v1/files/T/content"])
        # Block must contain markers AND visible content
        assert "[openrouter:v1:intent_block_start:1]: #" in block
        assert "[openrouter:v1:intent_block_end:1]: #" in block
        assert "/api/v1/files/T/content" in block
        assert "a black cat" in block
