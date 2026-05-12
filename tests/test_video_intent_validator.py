"""Validator tests for video_intent — 13 worked examples + schema invariants.

Each example tests that the validator accepts a valid task-model output and
produces the expected VideoIntentResult.
"""
from __future__ import annotations

import pytest

from open_webui_openrouter_pipe.integrations.video_intent import (
    FramePlanEntry,
    VideoIntentResult,
    validate_intent_params,
)


def _validate(raw, **kwargs):
    """Convenience wrapper with sensible defaults."""
    return validate_intent_params(
        raw,
        attachments_count=kwargs.pop("attachments_count", 0),
        prior_videos=kwargs.pop("prior_videos", []),
        video_model=kwargs.pop("video_model", {"supported_frame_images": ["first_frame", "last_frame"]}),
        explicit_frame_images_present=kwargs.pop("explicit_frame_images_present", False),
        prior_clarifications_in_session=kwargs.pop("prior_clarifications_in_session", 0),
        max_clarifications=kwargs.pop("max_clarifications", 1),
        fallback_prompt=kwargs.pop("fallback_prompt", "fallback prompt"),
    )


# -----------------------------------------------------------------------------
# 13 worked examples
# -----------------------------------------------------------------------------

def _empty_clar():
    return {"needs": False, "question": "", "options": None, "reason": ""}


def test_example_01_text_only_first_turn():
    raw = {
        "intent": "text_to_video",
        "frame_plan": [],
        "prompt": "a robot walking through a neon-lit alley",
        "use_user_prompt": False,
        "language": "en",
        "confidence": "high",
        "clarification": _empty_clar(),
        "reason": "fresh",
    }
    result = _validate(raw)
    assert result.intent == "text_to_video"
    assert result.frame_plan == []
    assert result.prompt == "a robot walking through a neon-lit alley"
    assert result.confidence == "high"


def test_example_02_modify_prior_video():
    raw = {
        "intent": "modify_prior_video",
        "frame_plan": [{
            "source": "prior_video_first_frame",
            "source_index": -1,
            "timestamp_seconds": None,
            "target": "first_frame",
        }],
        "prompt": "a black cat walking through tall grass",
        "use_user_prompt": False,
        "language": "en",
        "confidence": "high",
        "clarification": _empty_clar(),
        "reason": "diff-style modify",
    }
    prior = [{"index": 0, "message_index": 1, "file_url": "/api/v1/files/abc/content"}]
    result = _validate(raw, prior_videos=prior)
    assert result.intent == "modify_prior_video"
    assert len(result.frame_plan) == 1
    entry = result.frame_plan[0]
    assert entry.source == "prior_video_first_frame"
    assert entry.source_index == 0  # -1 normalized to last index
    assert entry.target == "first_frame"


def test_example_03_fresh_start_despite_prior():
    raw = {
        "intent": "text_to_video",
        "frame_plan": [],
        "prompt": "a dog running on the beach",
        "use_user_prompt": False,
        "language": "en",
        "confidence": "high",
        "clarification": _empty_clar(),
        "reason": "fresh",
    }
    prior = [{"index": 0, "message_index": 1, "file_url": "/api/v1/files/abc/content"}]
    result = _validate(raw, prior_videos=prior)
    assert result.intent == "text_to_video"
    assert result.frame_plan == []


def test_example_04_explicit_attachment_last_frame():
    raw = {
        "intent": "image_to_video",
        "frame_plan": [{
            "source": "uploaded_attachment",
            "source_index": 0,
            "timestamp_seconds": None,
            "target": "last_frame",
        }],
        "prompt": "a sunrise over mountains",
        "use_user_prompt": False,
        "language": "en",
        "confidence": "high",
        "clarification": _empty_clar(),
        "reason": "explicit",
    }
    result = _validate(raw, attachments_count=1)
    assert result.intent == "image_to_video"
    assert result.frame_plan[0].target == "last_frame"


def test_example_05_two_attachments_first_plus_last():
    raw = {
        "intent": "image_to_video",
        "frame_plan": [
            {"source": "uploaded_attachment", "source_index": 0, "timestamp_seconds": None, "target": "first_frame"},
            {"source": "uploaded_attachment", "source_index": 1, "timestamp_seconds": None, "target": "last_frame"},
        ],
        "prompt": "transition from morning to night",
        "use_user_prompt": False,
        "language": "en",
        "confidence": "high",
        "clarification": _empty_clar(),
        "reason": "two attachments",
    }
    result = _validate(raw, attachments_count=2)
    assert len(result.frame_plan) == 2
    targets = [e.target for e in result.frame_plan]
    assert "first_frame" in targets and "last_frame" in targets


def test_example_06_timestamp_extraction():
    raw = {
        "intent": "continue_prior_video",
        "frame_plan": [{
            "source": "prior_video_at_timestamp",
            "source_index": -1,
            "timestamp_seconds": 5,
            "target": "first_frame",
        }],
        "prompt": "continuing with a slow pan",
        "use_user_prompt": False,
        "language": "en",
        "confidence": "high",
        "clarification": _empty_clar(),
        "reason": "explicit timestamp",
    }
    prior = [{"index": 0, "file_url": "/x"}]
    result = _validate(raw, prior_videos=prior)
    assert result.frame_plan[0].timestamp_seconds == 5.0


def test_example_07_ambiguous_make_it_red():
    raw = {
        "intent": "ambiguous",
        "frame_plan": [],
        "prompt": "",
        "use_user_prompt": False,
        "language": "en",
        "confidence": "low",
        "clarification": {
            "needs": True,
            "question": "Which video should I recolor?",
            "options": ["The cat video", "The dog video"],
            "reason": "two priors",
        },
        "reason": "ambiguous",
    }
    prior = [{"index": 0, "file_url": "/a"}, {"index": 1, "file_url": "/b"}]
    result = _validate(raw, prior_videos=prior)
    assert result.intent == "ambiguous"
    assert result.clarification is not None
    assert result.clarification.needs is True
    assert len(result.clarification.options or []) == 2


def test_example_08_italian_modify():
    raw = {
        "intent": "modify_prior_video",
        "frame_plan": [{
            "source": "prior_video_first_frame", "source_index": -1,
            "timestamp_seconds": None, "target": "first_frame",
        }],
        "prompt": "un gatto nero che cammina nell'erba alta",
        "use_user_prompt": False,
        "language": "it",
        "confidence": "high",
        "clarification": _empty_clar(),
        "reason": "italian",
    }
    result = _validate(raw, prior_videos=[{"index": 0, "file_url": "/x"}])
    assert result.language == "it"
    assert "gatto nero" in result.prompt


def test_example_09_verbatim_optout():
    raw = {
        "intent": "modify_prior_video",
        "frame_plan": [{
            "source": "prior_video_first_frame", "source_index": -1,
            "timestamp_seconds": None, "target": "first_frame",
        }],
        "prompt": "the previous scene, cinematic and slow",
        "use_user_prompt": False,
        "language": "en",
        "confidence": "medium",
        "clarification": _empty_clar(),
        "reason": "opt-out",
    }
    prior = [{"index": 0, "file_url": "/a"}, {"index": 1, "file_url": "/b"}]
    result = _validate(raw, prior_videos=prior)
    assert result.confidence == "medium"
    assert result.intent == "modify_prior_video"


def test_example_10_explicit_wiring_verbatim():
    raw = {
        "intent": "continue_prior_video",
        "frame_plan": [{
            "source": "prior_video_last_frame", "source_index": -1,
            "timestamp_seconds": None, "target": "first_frame",
        }],
        "prompt": "a dragon swooping low over the village",
        "use_user_prompt": True,
        "language": "en",
        "confidence": "high",
        "clarification": _empty_clar(),
        "reason": "verbatim",
    }
    result = _validate(raw, prior_videos=[{"index": 0, "file_url": "/x"}])
    assert result.use_user_prompt is True


def test_example_11_referenced_prior_but_none_exist():
    raw = {
        "intent": "ambiguous",
        "frame_plan": [],
        "prompt": "",
        "use_user_prompt": False,
        "language": "en",
        "confidence": "low",
        "clarification": {
            "needs": True,
            "question": "There's no previous video. Generate a new sunset video?",
            "options": ["Generate a new sunset video", "Cancel"],
            "reason": "no prior",
        },
        "reason": "no prior",
    }
    result = _validate(raw, prior_videos=[])
    assert result.intent == "ambiguous"
    assert result.clarification is not None
    assert result.clarification.needs is True


def test_example_12_spanish_clarification():
    raw = {
        "intent": "ambiguous",
        "frame_plan": [],
        "prompt": "",
        "use_user_prompt": False,
        "language": "es",
        "confidence": "low",
        "clarification": {
            "needs": True,
            "question": "¿Cuál de los videos quieres que vuelva a generar en rojo?",
            "options": ["El video del coche", "El video del autobús"],
            "reason": "two priors",
        },
        "reason": "ambiguous",
    }
    prior = [{"index": 0, "file_url": "/a"}, {"index": 1, "file_url": "/b"}]
    result = _validate(raw, prior_videos=prior)
    assert result.language == "es"
    assert result.clarification is not None
    assert "rojo" in result.clarification.question


def test_example_13_timestamp_out_of_range():
    raw = {
        "intent": "continue_prior_video",
        "frame_plan": [{
            "source": "prior_video_at_timestamp", "source_index": -1,
            "timestamp_seconds": 30, "target": "first_frame",
        }],
        "prompt": "continuing the prior scene",
        "use_user_prompt": False,
        "language": "en",
        "confidence": "high",
        "clarification": _empty_clar(),
        "reason": "explicit timestamp",
    }
    # The pipe-side validator doesn't know the duration; downgrade happens in
    # frame_extraction. Validator should accept the timestamp.
    result = _validate(raw, prior_videos=[{"index": 0, "file_url": "/x"}])
    assert result.frame_plan[0].timestamp_seconds == 30.0


# -----------------------------------------------------------------------------
# Schema invariant tests
# -----------------------------------------------------------------------------

class TestInvariants:
    def test_drops_unknown_uploaded_attachment_index(self):
        raw = {
            "intent": "image_to_video",
            "frame_plan": [{
                "source": "uploaded_attachment",
                "source_index": 99,
                "timestamp_seconds": None,
                "target": "first_frame",
            }],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw, attachments_count=1)
        # Bad index dropped — frame_plan empty, intent downgraded
        assert result.frame_plan == []

    def test_drops_out_of_range_prior_video(self):
        raw = {
            "intent": "modify_prior_video",
            "frame_plan": [{
                "source": "prior_video_first_frame",
                "source_index": 99,
                "timestamp_seconds": None,
                "target": "first_frame",
            }],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw, prior_videos=[{"index": 0, "file_url": "/x"}])
        assert result.frame_plan == []

    def test_resolves_minus_one_to_most_recent(self):
        raw = {
            "intent": "modify_prior_video",
            "frame_plan": [{
                "source": "prior_video_first_frame",
                "source_index": -1,
                "timestamp_seconds": None,
                "target": "first_frame",
            }],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        prior = [{"index": 0}, {"index": 1}, {"index": 2}]
        result = _validate(raw, prior_videos=prior)
        assert result.frame_plan[0].source_index == 2

    def test_downgrades_unsupported_target(self):
        raw = {
            "intent": "image_to_video",
            "frame_plan": [{
                "source": "uploaded_attachment",
                "source_index": 0,
                "timestamp_seconds": None,
                "target": "last_frame",
            }],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        # Model only supports first_frame
        model = {"supported_frame_images": ["first_frame"]}
        result = _validate(raw, attachments_count=1, video_model=model)
        # last_frame downgraded to input_reference
        assert result.frame_plan[0].target == "input_reference"

    def test_dedupes_duplicate_first_frame_targets(self):
        raw = {
            "intent": "image_to_video",
            "frame_plan": [
                {"source": "uploaded_attachment", "source_index": 0, "timestamp_seconds": None, "target": "first_frame"},
                {"source": "uploaded_attachment", "source_index": 1, "timestamp_seconds": None, "target": "first_frame"},
            ],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw, attachments_count=2)
        first_frames = [e for e in result.frame_plan if e.target == "first_frame"]
        assert len(first_frames) == 1

    def test_truncates_at_4_entries(self):
        raw = {
            "intent": "image_to_video",
            "frame_plan": [
                {"source": "uploaded_attachment", "source_index": i, "timestamp_seconds": None, "target": "input_reference"}
                for i in range(6)
            ],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw, attachments_count=10)
        assert len(result.frame_plan) <= 4

    def test_discards_classifier_plan_when_explicit_attachments_present(self):
        raw = {
            "intent": "modify_prior_video",
            "frame_plan": [{
                "source": "prior_video_first_frame",
                "source_index": -1,
                "timestamp_seconds": None,
                "target": "first_frame",
            }],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw, prior_videos=[{"index": 0}], explicit_frame_images_present=True)
        assert result.discarded_plan is True
        assert result.frame_plan == []

    def test_text_to_video_forces_empty_frame_plan(self):
        raw = {
            "intent": "text_to_video",
            "frame_plan": [{
                "source": "uploaded_attachment", "source_index": 0,
                "timestamp_seconds": None, "target": "first_frame",
            }],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw, attachments_count=1)
        assert result.frame_plan == []

    def test_image_to_video_without_attachment_downgrades_to_text(self):
        raw = {
            "intent": "image_to_video",
            "frame_plan": [],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw, attachments_count=0)
        assert result.intent == "text_to_video"

    def test_max_clarifications_cap_forces_text_to_video(self):
        raw = {
            "intent": "ambiguous",
            "frame_plan": [],
            "prompt": "",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "low",
            "clarification": {
                "needs": True, "question": "?", "options": None, "reason": "x",
            },
            "reason": "x",
        }
        result = _validate(raw, prior_clarifications_in_session=1, max_clarifications=1)
        assert result.intent == "text_to_video"
        assert result.clarification is None or not result.clarification.needs

    def test_clarification_needs_true_requires_question(self):
        raw = {
            "intent": "ambiguous",
            "frame_plan": [],
            "prompt": "",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "low",
            "clarification": {
                "needs": True, "question": "", "options": None, "reason": "x",
            },
            "reason": "x",
        }
        result = _validate(raw)
        # Empty question -> needs forced to False -> intent downgraded to text_to_video
        assert result.intent == "text_to_video"

    def test_strips_image_placeholders_from_prompt(self):
        raw = {
            "intent": "text_to_video",
            "frame_plan": [],
            "prompt": "use [image:0] and [video:1] for the scene",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw)
        assert "[image:" not in result.prompt
        assert "[video:" not in result.prompt

    def test_caps_prompt_length(self):
        raw = {
            "intent": "text_to_video",
            "frame_plan": [],
            "prompt": "x" * 5000,
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw)
        assert len(result.prompt) <= 2000

    def test_missing_intent_falls_back_to_text_to_video(self):
        raw = {
            "frame_plan": [],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw)
        assert result.intent == "text_to_video"

    def test_missing_confidence_treated_as_low(self):
        raw = {
            "intent": "text_to_video",
            "frame_plan": [],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw)
        assert result.confidence == "low"

    def test_invalid_source_dropped(self):
        raw = {
            "intent": "image_to_video",
            "frame_plan": [{
                "source": "garbage",
                "source_index": 0,
                "timestamp_seconds": None,
                "target": "first_frame",
            }],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw, attachments_count=1)
        assert result.frame_plan == []

    def test_invalid_target_dropped(self):
        raw = {
            "intent": "image_to_video",
            "frame_plan": [{
                "source": "uploaded_attachment",
                "source_index": 0,
                "timestamp_seconds": None,
                "target": "garbage",
            }],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw, attachments_count=1)
        assert result.frame_plan == []

    def test_negative_timestamp_dropped(self):
        raw = {
            "intent": "continue_prior_video",
            "frame_plan": [{
                "source": "prior_video_at_timestamp",
                "source_index": -1,
                "timestamp_seconds": -5,
                "target": "first_frame",
            }],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw, prior_videos=[{"index": 0}])
        assert result.frame_plan == []

    def test_ambiguous_without_clarification_downgrades(self):
        raw = {
            "intent": "ambiguous",
            "frame_plan": [],
            "prompt": "",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "low",
            "clarification": _empty_clar(),  # needs=False
            "reason": "x",
        }
        result = _validate(raw)
        assert result.intent == "text_to_video"

    def test_use_user_prompt_preserves_flag(self):
        raw = {
            "intent": "text_to_video",
            "frame_plan": [],
            "prompt": "x",
            "use_user_prompt": True,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw)
        assert result.use_user_prompt is True

    def test_returns_VideoIntentResult_dataclass(self):
        raw = {
            "intent": "text_to_video",
            "frame_plan": [],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw)
        assert isinstance(result, VideoIntentResult)
        assert isinstance(result.downgrades, list)

    def test_input_reference_target_dropped_without_any_frame_support(self):
        # When the model has zero frame support, the entry cannot be honored
        # (no slot exists). Drop it instead of letting _encode_frame_images
        # break the paid call.
        raw = {
            "intent": "image_to_video",
            "frame_plan": [{
                "source": "uploaded_attachment",
                "source_index": 0,
                "timestamp_seconds": None,
                "target": "input_reference",
            }],
            "prompt": "x",
            "use_user_prompt": False,
            "language": "en",
            "confidence": "high",
            "clarification": _empty_clar(),
            "reason": "x",
        }
        result = _validate(raw, attachments_count=1, video_model={"supported_frame_images": []})
        assert len(result.frame_plan) == 0
        assert result.intent == "text_to_video"
        assert any(
            "no_frame_support" in note for note in result.downgrades
        ), f"expected no_frame_support downgrade, got {result.downgrades}"
