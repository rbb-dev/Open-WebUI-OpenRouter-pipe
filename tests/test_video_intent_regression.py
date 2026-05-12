"""Regression tests for the video intent classifier."""
from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import pytest

ConfidenceLit = Literal["high", "medium", "low"]

from open_webui_openrouter_pipe.integrations.video_intent import (
    ClarificationPayload,
    FramePlanEntry,
    IntentLiteral,
    VideoIntentResult,
    count_prior_clarifications,
    fallback_intent_result,
    render_clarification_message,
    should_emit_confirmation_footer,
    validate_intent_params,
)
from open_webui_openrouter_pipe.structured_task.client import (
    read_task_model_response_json,
)


_FALLBACK_PROMPT = "make a video"


def _empty_clar() -> dict:
    return {"needs": False, "question": "", "options": None, "reason": ""}


def _raw(
    *,
    intent: str = "image_to_video",
    frame_plan: list[dict] | None = None,
    confidence: str = "high",
    prompt: str = "x",
) -> dict:
    return {
        "intent": intent,
        "frame_plan": frame_plan or [],
        "prompt": prompt,
        "use_user_prompt": False,
        "language": "en",
        "confidence": confidence,
        "clarification": _empty_clar(),
        "reason": "",
    }


def _result(
    *,
    intent: IntentLiteral = "modify_prior_video",
    frame_plan: list[FramePlanEntry] | None = None,
    confidence: ConfidenceLit = "high",
    classifier_failed: bool = False,
) -> VideoIntentResult:
    return VideoIntentResult(
        intent=intent,
        frame_plan=frame_plan or [],
        prompt="x",
        use_user_prompt=False,
        language="en",
        confidence=confidence,
        clarification=None,
        reason="",
        classifier_failed=classifier_failed,
    )


# -----------------------------------------------------------------------------
# text-only video model: drop frame_plan, coerce intent
# -----------------------------------------------------------------------------

class TestNoFrameSupport:
    """When `supported_frame_images=[]`, the validator must drop frame entries
    so `_encode_frame_images` doesn't raise on the paid /videos call."""

    def test_prior_video_first_frame_dropped_on_empty_support(self):
        raw = _raw(
            intent="modify_prior_video",
            frame_plan=[{
                "source": "prior_video_last_frame", "source_index": 0,
                "timestamp_seconds": None, "target": "first_frame",
            }],
        )
        res = validate_intent_params(
            raw, attachments_count=0,
            prior_videos=[{
                "index": 0, "message_index": 0,
                "file_url": "/api/v1/files/abc/content",
                "model_id_if_known": "", "duration_seconds_if_known": None,
            }],
            video_model={"supported_frame_images": []},
            explicit_frame_images_present=False,
            prior_clarifications_in_session=0, max_clarifications=1,
            fallback_prompt=_FALLBACK_PROMPT,
        )
        assert res.intent == "text_to_video"
        assert res.frame_plan == []
        assert any("dropped_first_frame_no_frame_support" in d for d in res.downgrades)
        assert any("modify_prior_video_dropped_to_text" in d for d in res.downgrades)

    def test_uploaded_attachment_input_reference_dropped_on_empty_support(self):
        raw = _raw(
            intent="image_to_video",
            frame_plan=[{
                "source": "uploaded_attachment", "source_index": 0,
                "timestamp_seconds": None, "target": "input_reference",
            }],
        )
        res = validate_intent_params(
            raw, attachments_count=1, prior_videos=[],
            video_model={"supported_frame_images": []},
            explicit_frame_images_present=False,
            prior_clarifications_in_session=0, max_clarifications=1,
            fallback_prompt=_FALLBACK_PROMPT,
        )
        assert res.intent == "text_to_video"
        assert res.frame_plan == []
        assert any("dropped_input_reference_no_frame_support" in d for d in res.downgrades)

    def test_partial_support_downgrades_unsupported_target_to_input_reference(self):
        raw = _raw(
            intent="modify_prior_video",
            frame_plan=[{
                "source": "prior_video_first_frame", "source_index": 0,
                "timestamp_seconds": None, "target": "first_frame",
            }],
        )
        res = validate_intent_params(
            raw, attachments_count=0,
            prior_videos=[{
                "index": 0, "message_index": 0,
                "file_url": "/api/v1/files/abc/content",
                "model_id_if_known": "", "duration_seconds_if_known": None,
            }],
            video_model={"supported_frame_images": ["last_frame", "input_reference"]},
            explicit_frame_images_present=False,
            prior_clarifications_in_session=0, max_clarifications=1,
            fallback_prompt=_FALLBACK_PROMPT,
        )
        assert res.intent == "modify_prior_video"
        assert len(res.frame_plan) == 1
        assert res.frame_plan[0].target == "input_reference"
        assert any("downgraded_unsupported_target_first_frame" in d for d in res.downgrades)

    def test_full_support_keeps_target_unchanged(self):
        raw = _raw(
            intent="modify_prior_video",
            frame_plan=[{
                "source": "prior_video_first_frame", "source_index": 0,
                "timestamp_seconds": None, "target": "first_frame",
            }],
        )
        res = validate_intent_params(
            raw, attachments_count=0,
            prior_videos=[{
                "index": 0, "message_index": 0,
                "file_url": "/api/v1/files/abc/content",
                "model_id_if_known": "", "duration_seconds_if_known": None,
            }],
            video_model={"supported_frame_images": ["first_frame", "last_frame"]},
            explicit_frame_images_present=False,
            prior_clarifications_in_session=0, max_clarifications=1,
            fallback_prompt=_FALLBACK_PROMPT,
        )
        assert res.intent == "modify_prior_video"
        assert len(res.frame_plan) == 1
        assert res.frame_plan[0].target == "first_frame"
        assert res.downgrades == []


# -----------------------------------------------------------------------------
# explicit attachments: classifier still interprets text but
# can't substitute prior video frames; uploaded_attachment retargeting kept
# -----------------------------------------------------------------------------

class TestExplicitAttachmentsRetarget:
    def test_uploaded_attachment_retargeting_preserved_with_explicit(self):
        raw = _raw(
            intent="image_to_video",
            frame_plan=[{
                "source": "uploaded_attachment", "source_index": 0,
                "timestamp_seconds": None, "target": "last_frame",
            }],
        )
        res = validate_intent_params(
            raw, attachments_count=1, prior_videos=[],
            video_model={"supported_frame_images": ["first_frame", "last_frame"]},
            explicit_frame_images_present=True,
            prior_clarifications_in_session=0, max_clarifications=1,
            fallback_prompt=_FALLBACK_PROMPT,
        )
        assert res.intent == "image_to_video"
        assert len(res.frame_plan) == 1
        assert res.frame_plan[0].source == "uploaded_attachment"
        assert res.frame_plan[0].target == "last_frame"
        assert res.discarded_plan is False

    def test_prior_video_dropped_when_explicit_attachments_present(self):
        raw = _raw(
            intent="modify_prior_video",
            frame_plan=[{
                "source": "prior_video_first_frame", "source_index": 0,
                "timestamp_seconds": None, "target": "first_frame",
            }],
        )
        res = validate_intent_params(
            raw, attachments_count=1,
            prior_videos=[{
                "index": 0, "message_index": 0,
                "file_url": "/api/v1/files/abc/content",
                "model_id_if_known": "", "duration_seconds_if_known": None,
            }],
            video_model={"supported_frame_images": ["first_frame", "last_frame"]},
            explicit_frame_images_present=True,
            prior_clarifications_in_session=0, max_clarifications=1,
            fallback_prompt=_FALLBACK_PROMPT,
        )
        assert res.frame_plan == []
        assert any(
            "dropped_prior_video_entries_explicit_attachments" in d
            for d in res.downgrades
        )

    def test_short_circuit_no_longer_skips_when_frame_images_present(self):
        # Reach into the adapter without instantiating the full pipe.
        from open_webui_openrouter_pipe.integrations.video import (
            VideoGenerationAdapter,
        )

        pipe = MagicMock()
        adapter = VideoGenerationAdapter(pipe=pipe, logger=logging.getLogger("t"))
        valves = SimpleNamespace(
            VIDEO_INTENT_ENABLED=True,
            VIDEO_INTENT_SKIP_WHEN_EMPTY_CHAT=True,
            VIDEO_INTENT_MAX_CALLS_PER_CHAT=0,
            VIDEO_INTENT_MAX_CALLS_PER_USER_DAY=0,
        )
        # frame_images present + 2-message chat + non-empty prompt → must NOT
        # short-circuit (so the classifier can read "use this as the last frame")
        assert adapter._intent_classifier_should_run(
            valves=valves, persisted_content="", prompt="use this as the last frame",
            body={"messages": [{"role": "user", "content": "x"}, {"role": "user", "content": "x"}]},
            video_meta={"frame_images": [{"id": "abc", "kind": "first_frame"}]},
        )

    def test_apply_uploaded_attachment_retargeting_updates_kind(self):
        from open_webui_openrouter_pipe.integrations.video import (
            VideoGenerationAdapter,
        )

        pipe = MagicMock()
        adapter = VideoGenerationAdapter(pipe=pipe, logger=logging.getLogger("t"))
        intent = _result(
            intent="image_to_video",
            frame_plan=[FramePlanEntry(
                source="uploaded_attachment", source_index=0,
                timestamp_seconds=None, target="last_frame",
            )],
        )
        video_meta = {"frame_images": [{"id": "abc", "kind": "first_frame"}]}
        adapter._apply_uploaded_attachment_retargeting(intent, video_meta)
        assert video_meta["frame_images"][0]["kind"] == "last_frame"
        # Successful retargeting is tracked on the dedicated counter (NOT in
        # downgrades), so it doesn't surface to users as a 'non-critical step
        # was skipped' warning.
        assert intent.frames_retargeted == 1
        assert not any("retargeted" in d for d in intent.downgrades)


# -----------------------------------------------------------------------------
# clarification cap counts past current user turn
# -----------------------------------------------------------------------------

class TestClarificationCap:
    def _clar_text(self) -> str:
        return render_clarification_message(VideoIntentResult(
            intent="ambiguous", frame_plan=[], prompt="", use_user_prompt=False,
            language="en", confidence="low",
            clarification=ClarificationPayload(
                needs=True, question="Which one?",
                options=["A", "B"], reason="ambig",
            ),
            reason="",
        ))

    def test_returns_one_with_current_user_at_end(self):
        clar = self._clar_text()
        msgs = [
            {"role": "user", "content": "change colour"},
            {"role": "assistant", "content": clar},
            {"role": "user", "content": "still ambiguous"},
        ]
        assert count_prior_clarifications(msgs) == 1

    def test_returns_zero_when_no_prior_assistant_turn(self):
        msgs = [{"role": "user", "content": "fresh ambig"}]
        assert count_prior_clarifications(msgs) == 0

    def test_returns_zero_when_assistant_did_not_clarify(self):
        msgs = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": "regular reply"},
            {"role": "user", "content": "y"},
        ]
        assert count_prior_clarifications(msgs) == 0

    def test_streak_breaks_at_intervening_user_turn(self):
        # user → assistant(clarify) → user(reply) → assistant(clarify) → user(current)
        # Most-recent streak before current user is 1, not 2.
        clar = self._clar_text()
        msgs = [
            {"role": "user", "content": "orig"},
            {"role": "assistant", "content": clar},
            {"role": "user", "content": "reply1"},
            {"role": "assistant", "content": clar},
            {"role": "user", "content": "reply2"},
        ]
        assert count_prior_clarifications(msgs) == 1


# -----------------------------------------------------------------------------
# classifier infrastructure failures trip the breaker
# -----------------------------------------------------------------------------

class TestFailureMetadata:
    def test_fallback_result_has_classifier_failed_false(self):
        # Synthesized fallbacks for non-failure paths (e.g. no candidates) keep
        # classifier_failed=False so the breaker doesn't open spuriously.
        res = fallback_intent_result(_FALLBACK_PROMPT)
        assert res.classifier_failed is False
        assert res.failure_reason == ""

    def test_resolve_intent_marks_classifier_failed_on_exception(self):
        from open_webui_openrouter_pipe.integrations.video_intent import (
            resolve_intent,
        )

        async def boom_invoke(form_data: dict):
            raise RuntimeError("simulated task model down")

        async def main():
            valves = SimpleNamespace(
                VIDEO_INTENT_TASK_MODEL_MODE="external",
                VIDEO_INTENT_TASK_MODEL_FALLBACK="other_task_model",
                VIDEO_INTENT_TIMEOUT_S=2,
                VIDEO_INTENT_MAX_CLARIFICATIONS=1,
                VIDEO_INTENT_LOG_DECISIONS=False,
            )
            request = SimpleNamespace(
                app=SimpleNamespace(state=SimpleNamespace(
                    config=SimpleNamespace(TASK_MODEL_EXTERNAL="some/model"),
                )),
            )
            return await resolve_intent(
                body={"model": "x/y", "messages": [{"role": "user", "content": "hi"}]},
                video_meta={}, video_model={},
                valves=valves, request=request,
                user_obj=SimpleNamespace(id="u"),
                chat_id="c",
                logger=logging.getLogger("test"),
                fallback_prompt_text=_FALLBACK_PROMPT,
                invoke_chat_completion=boom_invoke,
            )

        res = asyncio.run(main())
        assert res.classifier_failed is True
        assert "RuntimeError" in res.failure_reason
        assert res.intent == "text_to_video"

    def test_resolve_intent_no_candidates_does_not_set_classifier_failed(self):
        # Admin hasn't configured a task model — that's not an infrastructure
        # outage, and shouldn't trip the breaker.
        from open_webui_openrouter_pipe.integrations.video_intent import (
            resolve_intent,
        )

        async def main():
            valves = SimpleNamespace(
                VIDEO_INTENT_TASK_MODEL_MODE="external",
                VIDEO_INTENT_TASK_MODEL_FALLBACK="none",
                VIDEO_INTENT_TIMEOUT_S=2,
                VIDEO_INTENT_MAX_CLARIFICATIONS=1,
                VIDEO_INTENT_LOG_DECISIONS=False,
            )
            request = SimpleNamespace(
                app=SimpleNamespace(state=SimpleNamespace(
                    config=SimpleNamespace(TASK_MODEL_EXTERNAL=""),
                )),
            )
            return await resolve_intent(
                body={"model": "x/y", "messages": [{"role": "user", "content": "hi"}]},
                video_meta={}, video_model={}, valves=valves, request=request,
                user_obj=SimpleNamespace(id="u"), chat_id="c",
                logger=logging.getLogger("test"),
                fallback_prompt_text=_FALLBACK_PROMPT,
            )

        res = asyncio.run(main())
        assert res.classifier_failed is False


# -----------------------------------------------------------------------------
# VIDEO_INTENT_CONFIRM_MODE + VIDEO_INTENT_FRAME_EXTRACTION_INDEX
# -----------------------------------------------------------------------------

class TestConfirmMode:
    def test_never_returns_false_even_with_frame_plan(self):
        intent = _result(frame_plan=[FramePlanEntry(
            source="prior_video_first_frame", source_index=0,
            timestamp_seconds=None, target="first_frame",
        )])
        assert should_emit_confirmation_footer(intent, confirm_mode="never") is False

    def test_always_returns_true_with_frame_plan(self):
        intent = _result(frame_plan=[FramePlanEntry(
            source="uploaded_attachment", source_index=0,
            timestamp_seconds=None, target="first_frame",
        )])
        assert should_emit_confirmation_footer(intent, confirm_mode="always") is True

    def test_always_returns_false_when_frame_plan_empty(self):
        intent = _result(frame_plan=[])
        assert should_emit_confirmation_footer(intent, confirm_mode="always") is False

    def test_on_reference_true_for_prior_video_source(self):
        intent = _result(frame_plan=[FramePlanEntry(
            source="prior_video_first_frame", source_index=0,
            timestamp_seconds=None, target="first_frame",
        )])
        assert should_emit_confirmation_footer(
            intent, confirm_mode="on_reference",
        ) is True

    def test_on_reference_false_for_single_uploaded_attachment(self):
        intent = _result(frame_plan=[FramePlanEntry(
            source="uploaded_attachment", source_index=0,
            timestamp_seconds=None, target="first_frame",
        )])
        assert should_emit_confirmation_footer(
            intent, confirm_mode="on_reference",
        ) is False

    def test_on_reference_true_for_multiple_attachments(self):
        intent = _result(frame_plan=[
            FramePlanEntry(
                source="uploaded_attachment", source_index=0,
                timestamp_seconds=None, target="first_frame",
            ),
            FramePlanEntry(
                source="uploaded_attachment", source_index=1,
                timestamp_seconds=None, target="last_frame",
            ),
        ])
        assert should_emit_confirmation_footer(
            intent, confirm_mode="on_reference",
        ) is True

    def test_low_confidence_only_when_confidence_low(self):
        intent_low = _result(
            frame_plan=[FramePlanEntry(
                source="uploaded_attachment", source_index=0,
                timestamp_seconds=None, target="first_frame",
            )],
            confidence="low",
        )
        intent_high = _result(
            frame_plan=[FramePlanEntry(
                source="uploaded_attachment", source_index=0,
                timestamp_seconds=None, target="first_frame",
            )],
            confidence="high",
        )
        assert should_emit_confirmation_footer(
            intent_low, confirm_mode="low_confidence",
        ) is True
        assert should_emit_confirmation_footer(
            intent_high, confirm_mode="low_confidence",
        ) is False

    def test_unknown_mode_falls_back_to_on_reference(self):
        intent = _result(frame_plan=[FramePlanEntry(
            source="uploaded_attachment", source_index=0,
            timestamp_seconds=None, target="first_frame",
        )])
        # Single uploaded attachment under on_reference → False
        assert should_emit_confirmation_footer(
            intent, confirm_mode="bogus_mode",
        ) is False


class TestOvershootFallback:
    def test_extract_frame_overshoot_uses_first_when_valve_first(self, tmp_path):
        # Don't depend on a real fixture — exercise the param plumbing.
        from open_webui_openrouter_pipe.media import frame_extraction

        captured: dict[str, float] = {}

        async def fake_probe(_):
            return frame_extraction.VideoMetadata(
                duration_seconds=4.0, width=1, height=1, fps=24.0, has_audio=False,
            )

        async def fake_ffmpeg(path, *, timestamp_seconds, logger):
            captured["ts"] = timestamp_seconds
            return (b"\x89PNG\r\n\x1a\n", 1, 1)

        original_probe = frame_extraction.probe_video
        original_ffmpeg = frame_extraction._extract_frame_ffmpeg
        frame_extraction.probe_video = fake_probe  # type: ignore[assignment]
        frame_extraction._extract_frame_ffmpeg = fake_ffmpeg  # type: ignore[assignment]
        try:
            video = tmp_path / "x.mp4"
            video.write_bytes(b"\x00")  # exists check passes; extraction is mocked
            frame = asyncio.run(frame_extraction.extract_frame(
                video, target="at_timestamp", timestamp_seconds=30.0,
                fallback_to_last_on_overshoot=True,
                overshoot_fallback_index="first",
            ))
            assert captured["ts"] == 0.0
            assert "first frame" in frame.downgrade_note.lower()
        finally:
            frame_extraction.probe_video = original_probe  # type: ignore[assignment]
            frame_extraction._extract_frame_ffmpeg = original_ffmpeg  # type: ignore[assignment]

    def test_extract_frame_overshoot_uses_last_by_default(self, tmp_path):
        from open_webui_openrouter_pipe.media import frame_extraction

        captured: dict[str, float] = {}

        async def fake_probe(_):
            return frame_extraction.VideoMetadata(
                duration_seconds=4.0, width=1, height=1, fps=24.0, has_audio=False,
            )

        async def fake_ffmpeg(path, *, timestamp_seconds, logger):
            captured["ts"] = timestamp_seconds
            return (b"\x89PNG\r\n\x1a\n", 1, 1)

        original_probe = frame_extraction.probe_video
        original_ffmpeg = frame_extraction._extract_frame_ffmpeg
        frame_extraction.probe_video = fake_probe  # type: ignore[assignment]
        frame_extraction._extract_frame_ffmpeg = fake_ffmpeg  # type: ignore[assignment]
        try:
            video = tmp_path / "x.mp4"
            video.write_bytes(b"\x00")
            frame = asyncio.run(frame_extraction.extract_frame(
                video, target="at_timestamp", timestamp_seconds=30.0,
                fallback_to_last_on_overshoot=True,
            ))
            assert captured["ts"] > 3.5  # near the 4.0s end
            assert "last frame" in frame.downgrade_note.lower()
        finally:
            frame_extraction.probe_video = original_probe  # type: ignore[assignment]
            frame_extraction._extract_frame_ffmpeg = original_ffmpeg  # type: ignore[assignment]


# -----------------------------------------------------------------------------
# task-model adapter returns plain str
# -----------------------------------------------------------------------------

class TestPlainStringResponse:
    def test_plain_json_string_parses(self):
        async def main():
            return await read_task_model_response_json(
                '{"intent": "text_to_video", "frame_plan": []}'
            )

        result = asyncio.run(main())
        assert result["intent"] == "text_to_video"

    def test_string_with_whitespace_parses(self):
        async def main():
            return await read_task_model_response_json(
                '   {"intent": "text_to_video"}  \n'
            )

        result = asyncio.run(main())
        assert result["intent"] == "text_to_video"

    def test_empty_string_raises_runtime_error(self):
        async def main():
            await read_task_model_response_json("   ")

        with pytest.raises(RuntimeError, match="task_model_empty_response"):
            asyncio.run(main())

    def test_invalid_json_string_raises_runtime_error(self):
        async def main():
            await read_task_model_response_json("not json {{{")

        with pytest.raises(RuntimeError, match="task_model_invalid_json"):
            asyncio.run(main())

    def test_oversize_string_raises_runtime_error(self):
        async def main():
            await read_task_model_response_json("x" * (256 * 1024 + 1))

        with pytest.raises(RuntimeError, match="task_model_response_too_large"):
            asyncio.run(main())


# -----------------------------------------------------------------------------
# pyright type for test helper
# -----------------------------------------------------------------------------

class TestIntentLiteralExport:
    def test_intent_literal_is_importable(self):
        # IntentLiteral is the public type alias tests rely on. If this
        # import breaks, pyright fails on the helper; this test guards it.
        from open_webui_openrouter_pipe.integrations.video_intent import (
            IntentLiteral as _IntentLiteral,  # noqa: F401
        )

        # Ensure the alias is the exact union pyright expects (5 variants).
        # Using a typing introspection that won't fail on Literal evaluation.
        from typing import get_args
        args = set(get_args(_IntentLiteral))
        assert args == {
            "text_to_video", "image_to_video", "modify_prior_video",
            "continue_prior_video", "ambiguous",
        }
