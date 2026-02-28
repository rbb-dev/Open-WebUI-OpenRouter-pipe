"""Tests for adaptive context budgeting helpers."""

from __future__ import annotations

from open_webui_openrouter_pipe.core.context_budget import (
    apply_live_tool_output_budget,
    apply_replay_tool_output_budget,
    build_replayed_tool_omission_stub,
    compute_prompt_limit_tokens,
)
from open_webui_openrouter_pipe.models.registry import ModelFamily


def test_compute_prompt_limit_prefers_max_prompt_tokens() -> None:
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {
                    "max_prompt_tokens": 64000,
                    "context_length": 200000,
                    "top_provider": {"context_length": 4096},
                }
            }
        }
    )
    assert compute_prompt_limit_tokens("test/model") == 64000


def test_compute_prompt_limit_prefers_top_level_context_over_top_provider() -> None:
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "context_length": 100000,
                "max_completion_tokens": 20000,
                "full_model": {
                    "context_length": 100000,
                    "max_completion_tokens": 20000,
                    "top_provider": {
                        "context_length": 4096,
                        "max_completion_tokens": 1024,
                    },
                },
            }
        }
    )
    assert compute_prompt_limit_tokens("test/model") == 80000


def test_compute_prompt_limit_falls_back_to_top_provider() -> None:
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {
                    "top_provider": {
                        "context_length": 4096,
                        "max_completion_tokens": 1024,
                    }
                }
            }
        }
    )
    assert compute_prompt_limit_tokens("test/model") == 3072


def test_live_budget_omits_when_result_exceeds_half_remaining() -> None:
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {"max_prompt_tokens": 100},
                "context_length": 100,
            }
        }
    )
    outputs = [{"type": "function_call_output", "call_id": "call-1", "output": "x" * 240}]
    omitted = apply_live_tool_output_budget(
        outputs,
        existing_input_items=[],
        model_id="test/model",
    )

    assert omitted == {"call-1"}
    assert outputs[0]["output"].startswith("[Tool result omitted due to context budget.")


def test_live_budget_omits_when_remaining_is_zero() -> None:
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {"max_prompt_tokens": 50},
                "context_length": 50,
            }
        }
    )
    oversized_existing = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x" * 2000}]}
    ]
    outputs = [{"type": "function_call_output", "call_id": "call-1", "output": "ok"}]
    omitted = apply_live_tool_output_budget(
        outputs,
        existing_input_items=oversized_existing,
        model_id="test/model",
    )

    assert omitted == {"call-1"}
    assert outputs[0]["output"].startswith("[Tool result omitted due to context budget.")


def test_live_budget_tracks_multiple_outputs_within_iteration() -> None:
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {"max_prompt_tokens": 200},
                "context_length": 200,
            }
        }
    )
    outputs = [
        {"type": "function_call_output", "call_id": "call-1", "output": "a" * 300},
        {"type": "function_call_output", "call_id": "call-2", "output": "b" * 260},
    ]
    omitted = apply_live_tool_output_budget(
        outputs,
        existing_input_items=[],
        model_id="test/model",
    )

    assert "call-1" not in omitted
    assert outputs[0]["output"] == "a" * 300
    assert "call-2" in omitted
    assert outputs[1]["output"].startswith("[Tool result omitted due to context budget.")


def test_replay_budget_is_idempotent_for_existing_stub() -> None:
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {"max_prompt_tokens": 30},
                "context_length": 30,
            }
        }
    )
    original_stub = build_replayed_tool_omission_stub(result_chars=4000, remaining_tokens=0)
    items = [{"type": "function_call_output", "call_id": "call-1", "output": original_stub}]

    first = apply_replay_tool_output_budget(items, model_id="test/model")
    second = apply_replay_tool_output_budget(items, model_id="test/model")

    assert "call-1" in first
    assert "call-1" in second
    assert items[0]["output"] == original_stub
