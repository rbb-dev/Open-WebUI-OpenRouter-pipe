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

    @pytest.mark.parametrize(
        "system_text",
        [None, "HOUSE STYLE: always cel-shaded", "STUDIO RULE: hand-held camera"],
    )
    def test_help_prompt_returns_false(self, system_text):
        """The user typed `help`, whatever the Workspace model prepends to it.

        The prompt that reaches the model is the system text and the user text joined,
        so comparing THAT to "help" misses on every Workspace model carrying a system
        prompt -- and the classifier bills an LLM call to interpret a request for the
        help panel.
        """
        adapter = self._make_adapter()
        turns = [] if system_text is None else [{"role": "system", "content": system_text}]
        turns += [{"role": "user", "content": "help"}]
        body = {"messages": turns}
        assert not adapter._intent_classifier_should_run(
            valves=_make_valves(),
            persisted_content="", prompt=adapter._extract_prompt(body),
            body=body, video_meta={"frame_images": [{"id": "x"}]},
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

    @pytest.mark.parametrize("cap", [1, 3])
    def test_the_per_chat_budget_is_filled_by_the_calls_that_were_made(self, cap):
        """The cap costs money, so the counter that fills it has to be the real one.

        Every other test of this cap pre-seeded `_intent_call_counts_per_chat` by hand,
        so `_intent_record_call` could do nothing at all and the budget would never be
        reached: the classifier is an extra LLM call per request, and the valve that
        bounds it would have bounded nothing.

        Two caps, so neither a constant nor a counter that saturates at one satisfies
        both.
        """
        adapter = self._make_adapter()
        valves = _make_valves(VIDEO_INTENT_MAX_CALLS_PER_CHAT=cap)

        def _ask():
            return adapter._intent_classifier_should_run(
                valves=valves,
                persisted_content="", prompt="make a video",
                body={"messages": [{}, {}]}, video_meta={}, chat_id="chat1",
            )

        allowed = 0
        for _turn in range(cap + 2):
            if not _ask():
                break
            allowed += 1
            adapter._intent_record_call("chat1", "")

        assert allowed == cap, (
            f"a cap of {cap} allowed {allowed} classifier call(s) before it closed"
        )
        assert adapter._intent_call_counts_per_chat["chat1"] == cap

    @pytest.mark.parametrize("cap", [1, 3])
    def test_the_per_user_day_budget_is_filled_by_the_calls_that_were_made(self, cap):
        """The same counter question for the per-user daily cap, which is keyed by date."""
        adapter = self._make_adapter()
        valves = _make_valves(VIDEO_INTENT_MAX_CALLS_PER_USER_DAY=cap)

        allowed = 0
        for _turn in range(cap + 2):
            if not adapter._intent_classifier_should_run(
                valves=valves,
                persisted_content="", prompt="make a video",
                body={"messages": [{}, {}]}, video_meta={}, user_id="user-1",
            ):
                break
            allowed += 1
            adapter._intent_record_call("", "user-1")

        assert allowed == cap, (
            f"a daily cap of {cap} allowed {allowed} classifier call(s) before it closed"
        )
        assert sum(adapter._intent_call_counts_per_user_day.values()) == cap

    def test_one_chats_calls_are_not_charged_to_another(self):
        """The counters are per chat and per user; sharing one would close both together."""
        adapter = self._make_adapter()
        adapter._intent_record_call("chat-a", "user-1")
        adapter._intent_record_call("chat-a", "user-1")

        assert adapter._intent_call_counts_per_chat == {"chat-a": 2}
        assert adapter._intent_classifier_should_run(
            valves=_make_valves(VIDEO_INTENT_MAX_CALLS_PER_CHAT=2),
            persisted_content="", prompt="make a video",
            body={"messages": [{}, {}]}, video_meta={}, chat_id="chat-b",
        ), "a second chat was refused for calls the first one made"

    def test_breaker_open_returns_false(self):
        adapter = self._make_adapter()
        adapter._intent_breaker_until_ts = time.time() + 60
        assert not adapter._intent_classifier_should_run(
            valves=_make_valves(),
            persisted_content="", prompt="hi",
            body={"messages": [{}, {}]}, video_meta={},
        )

    @pytest.mark.parametrize("chat_id", [None, "chat1"])
    def test_happy_path_returns_true(self, chat_id):
        """Parametrised over the chat id because the default cap is 0 = unlimited.

        Every previous happy-path case passed no chat_id, and the only case that did
        pass one used a non-zero cap. So `cap_chat > 0` could be relaxed to `>= 0` with
        the suite green -- and under the shipped default that makes `0 >= 0` true and
        `counts.get(chat_id, 0) >= 0` always true, so the classifier refuses to run for
        every request that has a chat id. Which is every real request: natural-language
        video generation would silently stop working out of the box.

        The pre-seeded count is what forces the comparison against the counter to be
        evaluated rather than short-circuited by an absent key.
        """
        adapter = self._make_adapter()
        if chat_id:
            adapter._intent_call_counts_per_chat[chat_id] = 7
        assert adapter._intent_classifier_should_run(
            valves=_make_valves(),
            persisted_content="", prompt="make a video",
            body={"messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "make a video"},
            ]},
            video_meta={},
            chat_id=chat_id,
        ), (
            "the classifier refused to run under the default cap of 0, which the valve "
            "documents as unlimited"
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
        pipe.valves.VIDEO_MAX_SIZE_MB = 500
        pipe.valves.ALLOW_UNKNOWN_SIZE_CLOUD_READS = False
        pipe._file_gateway.upload_to_owui_storage = AsyncMock(return_value="uploaded_id")
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

    @pytest.mark.asyncio
    async def test_resolve_returning_none_adds_download_failed_downgrade(self):
        """When `_resolve_owui_file_path` degrades to None (e.g. the gateway
        raised RequiredInternalFileError for an unauthorized prior video), the
        caller drops the entry and records a downgrade — no crash."""
        adapter, _ = self._make_adapter_with_mocks()
        intent = self._make_intent_with_prior_video()
        with patch.object(adapter, "_resolve_owui_file_path", AsyncMock(return_value=None)):
            video_meta: dict = {}
            await adapter._materialise_frame_plan(
                intent=intent, video_meta=video_meta,
                request=None, user_obj=SimpleNamespace(id="u_unauthorized"),
                chat_id="c1", message_id="m1",
            )
        assert "frame_images" not in video_meta or video_meta.get("frame_images") == []
        assert any("prior_video_download_failed" in d for d in intent.downgrades)

    @pytest.mark.asyncio
    async def test_resolved_temp_is_unlinked_after_extraction(self, tmp_path):
        """The private temp returned by `_resolve_owui_file_path` must be
        unlinked by `_materialise_frame_plan` once the frame is extracted —
        covers the per-prior-video temp leak fix."""
        adapter, _ = self._make_adapter_with_mocks()
        intent = self._make_intent_with_prior_video()
        real_temp = tmp_path / "orpipe-read-prior.mp4"
        real_temp.write_bytes(b"fake_video")
        assert real_temp.exists()
        with patch.object(adapter, "_resolve_owui_file_path", AsyncMock(return_value=real_temp)), \
             patch("open_webui_openrouter_pipe.integrations.video.extract_frame", AsyncMock(return_value=SimpleNamespace(
                 image_bytes=b"fake_png", width=10, height=10, downgrade_note="",
             ))):
            await adapter._materialise_frame_plan(
                intent=intent, video_meta={},
                request=None, user_obj=SimpleNamespace(id="u1"),
                chat_id="c1", message_id="m1",
            )
        assert not real_temp.exists(), "prior-video temp leaked; _materialise_frame_plan must unlink it"


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
