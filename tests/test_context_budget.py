"""Tests for adaptive context budgeting helpers."""

from __future__ import annotations

import pytest

from open_webui_openrouter_pipe.core.context_budget import (
    apply_live_tool_output_budget,
    apply_replay_tool_output_budget,
    build_replayed_tool_omission_stub,
    compute_prompt_limit_tokens,
    estimate_serialized_chars,
    is_tool_omission_stub,
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


def test_live_budget_keeps_output_that_fits() -> None:
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {"max_prompt_tokens": 100},
                "context_length": 100,
            }
        }
    )
    # 100 tokens * 4 chars/token = 400 chars budget; 240 chars fits.
    outputs = [{"type": "function_call_output", "call_id": "call-1", "output": "x" * 240}]
    omitted = apply_live_tool_output_budget(
        outputs,
        existing_input_items=[],
        model_id="test/model",
    )

    assert not omitted
    assert outputs[0]["output"] == "x" * 240


def test_live_budget_omits_when_result_exceeds_remaining() -> None:
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {"max_prompt_tokens": 100},
                "context_length": 100,
            }
        }
    )
    # 100 tokens * 4 chars/token = 400 chars budget; 500 chars does not fit.
    outputs = [{"type": "function_call_output", "call_id": "call-1", "output": "x" * 500}]
    omitted = apply_live_tool_output_budget(
        outputs,
        existing_input_items=[],
        model_id="test/model",
    )

    assert omitted == {"call-1"}
    assert outputs[0]["output"].startswith("[Tool result omitted due to context budget.")


def test_replay_budget_leaves_results_alone_when_stubbing_cannot_help() -> None:
    """Same futility rule on the replay path, where one oversized reasoning item lives.

    A multi-megabyte reasoning artifact is counted in full when sizing the budget, so it
    alone can exceed the whole prompt limit. Stubbing every replayed tool result then
    destroys them without bringing the request under the limit.
    """
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {"max_prompt_tokens": 50},
                "context_length": 50,
            }
        }
    )
    items = [
        {"type": "reasoning", "id": "rs-1", "signature": "s" * 4000},
        {"type": "function_call_output", "call_id": "call-1", "output": "ok"},
    ]

    omitted = apply_replay_tool_output_budget(items, model_id="test/model")

    assert omitted == set()
    assert items[1]["output"] == "ok"


def test_live_budget_leaves_results_alone_when_stubbing_cannot_help() -> None:
    """When the request is already over budget without any tool output, stubbing is futile.

    Omitting a two-character result cannot bring a request that is ten times over the
    limit back under it -- it only destroys the result while the request still overflows.
    The provider's own context compression is the mechanism that handles this case.
    """
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

    assert omitted == set()
    assert outputs[0]["output"] == "ok"


def test_live_budget_tracks_multiple_outputs_within_iteration() -> None:
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {"max_prompt_tokens": 200},
                "context_length": 200,
            }
        }
    )
    # 200 tokens * 4 = 800 chars budget.
    # Tool 1: 500 chars — fits (800 remaining), leaves 300.
    # Tool 2: 400 chars — exceeds 300 remaining, gets stubbed.
    outputs = [
        {"type": "function_call_output", "call_id": "call-1", "output": "a" * 500},
        {"type": "function_call_output", "call_id": "call-2", "output": "b" * 400},
    ]
    omitted = apply_live_tool_output_budget(
        outputs,
        existing_input_items=[],
        model_id="test/model",
    )

    assert "call-1" not in omitted
    assert outputs[0]["output"] == "a" * 500
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


def _spec(max_prompt_tokens: int) -> None:
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {"max_prompt_tokens": max_prompt_tokens},
                "context_length": max_prompt_tokens,
            }
        }
    )


@pytest.mark.parametrize(("filler_chars", "expect_stub"), [(100, False), (3_000, True)])
def test_the_budget_subtracts_what_the_rest_of_the_request_needs(filler_chars, expect_stub) -> None:
    """Two fillers, one result size, opposite verdicts.

    Every other case here runs with an empty request or one already over the limit, and
    in both of those the subtraction is irrelevant to the outcome. Deleting it outright
    -- `remaining_chars = prompt_limit_chars` -- therefore changed nothing anywhere in
    the suite. A single case cannot fix that either: it is satisfied by returning the
    expected answer. Only a pair whose verdicts differ *because* of the filler can tell
    a real subtraction from a constant.
    """
    _spec(1_000)
    existing = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x" * filler_chars}]}
    ]
    outputs = [{"type": "function_call_output", "call_id": "call-1", "output": "z" * 1_200}]

    omitted = apply_live_tool_output_budget(
        outputs, existing_input_items=existing, model_id="test/model"
    )

    assert (omitted == {"call-1"}) is expect_stub
    assert is_tool_omission_stub(outputs[0]["output"]) is expect_stub


def test_a_tool_result_already_in_the_request_is_not_counted_as_fixed_cost() -> None:
    """The live path measured the request raw; the replay path blanked outputs first.

    The tool loop appends each accepted result to the same list the next iteration
    measures, so the previous result was charged as un-shrinkable -- and it is exactly
    what this function shrinks. One large accepted result then convinced the guard the
    request was hopeless and switched budgeting off for the rest of the turn.
    """
    _spec(200)
    existing = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "function_call_output", "call_id": "earlier", "output": "y" * 4_000},
    ]
    outputs = [{"type": "function_call_output", "call_id": "new", "output": "z" * 4_000}]

    omitted = apply_live_tool_output_budget(
        outputs, existing_input_items=existing, model_id="test/model"
    )

    assert omitted == {"new"}, (
        "an earlier accepted tool result was treated as fixed cost, so the budget gave "
        "up instead of trimming the new one"
    )


@pytest.mark.parametrize("payload_chars", [100, 1_000_000])
def test_an_inlined_image_is_not_measured_by_its_base64_length(payload_chars) -> None:
    """A model bills an image by resolution; base64 length says nothing about cost.

    Counting it literally reads an ordinary phone photo as hundreds of thousands of
    tokens, which on its own exceeds a large model's whole character budget and
    convinces the guard that a perfectly normal request cannot be trimmed. Parametrised
    over two payload sizes four orders of magnitude apart: any estimate that still
    scales with the base64 gives different answers for the two.
    """
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": "data:image/png;base64," + "A" * payload_chars}],
        }
    ]

    assert estimate_serialized_chars(items) == estimate_serialized_chars(
        [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
            }
        ]
    )


def test_a_live_result_shorter_than_the_stub_is_left_alone() -> None:
    """Substituting the stub has to make the request smaller, or it is pure loss.

    The stub is a fixed ~190 characters. Swapping a 40-character result for it destroys
    the result and grows the request, and the returned call id then makes the streaming
    layer skip persisting both halves of the pair -- so the real result survives
    nowhere.

    The regime assertion is load-bearing. The band that reaches this branch at all is
    narrow: a little less filler and the result simply fits, a little more and the
    futility guard returns before the loop runs. A first attempt at this test sat just
    outside it and passed whether the guard existed or not.
    """
    _spec(100)
    existing = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x" * 300}]}
    ]
    remaining = 400 - estimate_serialized_chars(existing)
    assert 0 < remaining < 40, f"regime broken: remaining={remaining}, nothing is being tested"
    outputs = [{"type": "function_call_output", "call_id": "tiny", "output": "z" * 40}]

    omitted = apply_live_tool_output_budget(
        outputs, existing_input_items=existing, model_id="test/model"
    )

    assert omitted == set(), "a result shorter than the stub was replaced by it"
    assert outputs[0]["output"] == "z" * 40


def test_a_replayed_result_shorter_than_the_stub_is_left_alone() -> None:
    """The replay stub is shorter than the live one but still far longer than 40 chars."""
    _spec(100)
    filler = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x" * 230}]}
    result = {"type": "function_call_output", "call_id": "tiny", "output": "z" * 40}
    remaining = 400 - estimate_serialized_chars([filler, {**result, "output": ""}])
    assert 0 < remaining < 40, f"regime broken: remaining={remaining}, nothing is being tested"

    items = [filler, result]
    omitted = apply_replay_tool_output_budget(items, model_id="test/model")

    assert omitted == set(), "a replayed result shorter than the stub was replaced by it"
    assert items[1]["output"] == "z" * 40
