"""Tests for adaptive context budgeting helpers."""

from __future__ import annotations

import base64
import json
import logging
import re
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe.core.context_budget import (
    effective_chars_per_token,
    measure_chars_per_token,
    record_chars_per_token,
    _CHARS_PER_TOKEN_HEURISTIC,
    _decode_window,
    _payload_windows,
    apply_live_tool_output_budget,
    apply_replay_tool_output_budget,
    _LIVE_OMISSION_PREFIX,
    _baseline_without_tool_outputs,
    build_live_tool_omission_stub,
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
    assert compute_prompt_limit_tokens("test/model") == 100000, (
        "with no reply allowance declared the whole window is available for the prompt"
    )
    assert compute_prompt_limit_tokens("test/model", reserved_output_tokens=20000) == 80000, (
        "a declared allowance is subtracted from the top-level window, not the provider's"
    )
    assert compute_prompt_limit_tokens("test/model", reserved_output_tokens=1) == 99999


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
    assert compute_prompt_limit_tokens("test/model") == 4096, (
        "the fallback window is the provider's context_length, whole"
    )
    assert compute_prompt_limit_tokens("test/model", reserved_output_tokens=1024) == 3072
    assert compute_prompt_limit_tokens("test/model", reserved_output_tokens=4096) == 4096, (
        "an allowance as large as the window cannot drive the prompt budget to zero"
    )


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
    ).omitted_call_ids

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
    ).omitted_call_ids

    assert omitted == {"call-1"}
    assert outputs[0]["output"].startswith("[Tool result omitted due to context budget.")


def test_replay_budget_still_trims_and_says_so_when_it_cannot_win() -> None:
    """When the request is over budget before any tool output, the pass trims and reports it.

    A result far larger than the stub is the load-bearing case. With a two-character
    result the guard's answer is indistinguishable from the stub-length short-circuit
    added alongside it -- both leave the output alone -- so the whole guard could be
    deleted with the suite still green. The regime assertion below keeps the two
    mechanisms separable: the stub genuinely would shrink this result.

    The contract here was inverted by an explicit product decision, not by a test being
    made to agree with new behaviour. Trimming used to STOP when the pass calculated that
    no amount of trimming could bring the request under the limit. That made request size
    discontinuous in conversation size: on a 16,000-character limit, 15,125 characters of
    user text dispatched 15,998 and 15,127 characters dispatched 54,475 -- two more
    characters typed, 3.4x more data on the wire, both rejected, the second slower and
    dearer. The owner chose: trim anyway and say so. The pipe's four-characters-per-token
    estimate is a rule of thumb, so a request it calls too big may well fit once trimmed,
    and an untrimmed one certainly will not.
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
        {"type": "function_call_output", "call_id": "call-1", "output": "y" * 4000},
    ]

    assert len(
        build_replayed_tool_omission_stub(result_chars=4000, remaining_tokens=0)
    ) < 4000, "regime broken: the stub-length short-circuit would give this answer too"
    outcome = apply_replay_tool_output_budget(items, model_id="test/model")

    assert outcome.futile is True, (
        "the pass did not report that it could not bring the request under the limit"
    )
    assert outcome.omitted_call_ids == {"call-1"}, (
        "the pass gave up instead of trimming what it could"
    )
    assert is_tool_omission_stub(items[1]["output"]), (
        "the oversized result was left at full size on a request already over the limit"
    )


def test_live_budget_still_trims_and_says_so_when_it_cannot_win() -> None:
    """When the request is already over budget without any tool output, stubbing is futile.

    A result far larger than the stub is the load-bearing case. With a two-character
    result the guard's answer is indistinguishable from the stub-length short-circuit
    added alongside it -- both leave the output alone -- so the whole guard could be
    deleted with the suite still green. The regime assertion below keeps the two
    mechanisms separable: the stub genuinely would shrink this result, so only the
    futility guard can explain the output surviving.
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
    outputs = [{"type": "function_call_output", "call_id": "call-1", "output": "y" * 4000}]
    assert len(
        build_live_tool_omission_stub(result_chars=4000, remaining_tokens=0)
    ) < 4000, "regime broken: the stub-length short-circuit would give this answer too"
    outcome = apply_live_tool_output_budget(
        outputs,
        existing_input_items=oversized_existing,
        model_id="test/model",
    )

    assert outcome.futile is True, (
        "the pass did not report that it could not bring the request under the limit"
    )
    assert outcome.omitted_call_ids == {"call-1"}, (
        "the pass gave up instead of trimming what it could"
    )
    assert is_tool_omission_stub(outputs[0]["output"]), (
        "the oversized result was left at full size on a request already over the limit"
    )


def _dispatch(existing: list, outputs: list, *, model_id: str = "test/model"):
    """Reproduce the dispatch path: the live pass, then the sanitiser's replay pass.

    `streaming_core` runs `apply_live_tool_output_budget`, extends `body.input` with
    the budgeted outputs, and then calls `_sanitize_request_input`, which runs
    `apply_replay_tool_output_budget` over the whole list. The request the provider
    receives is the result of BOTH passes, so that is what these tests assert on.
    """
    paired = _with_calls(existing) + [
        {
            "type": "function_call",
            "call_id": output.get("call_id"),
            "name": "lookup",
            "arguments": "{}",
        }
        for output in outputs
        if isinstance(output, dict) and output.get("type") == "function_call_output"
    ]
    outcome = apply_live_tool_output_budget(
        outputs, existing_input_items=paired, model_id=model_id
    )
    shipped = list(paired) + list(outputs)
    apply_replay_tool_output_budget(shipped, model_id=model_id)
    return outcome, shipped


def _with_calls(items: list) -> list:
    """Insert the `function_call` production always puts ahead of each output.

    `streaming_core` extends `body.input` with the normalised calls before the live
    pass runs, and `_validate_tool_call_pairs` drops any output whose call is missing --
    so a bare `function_call_output` is a shape the provider never receives. A fixture
    built from bare outputs is calibrated against a request that cannot exist.
    """
    paired: list = []
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            paired.append({
                "type": "function_call",
                "call_id": item.get("call_id"),
                "name": "lookup",
                "arguments": "{}",
            })
        paired.append(item)
    return paired


def _delivered(shipped: list) -> set:
    """call_ids whose result the shipped request still carries in full."""
    return {
        item["call_id"]
        for item in shipped
        if isinstance(item, dict)
        and item.get("type") == "function_call_output"
        and not is_tool_omission_stub(item.get("output") or "")
    }


def test_live_budget_tracks_multiple_outputs_within_iteration() -> None:
    """Several results in one iteration must leave a request that fits.

    The pass spends the budget across the whole batch, so it has to reserve what the
    later results will cost before it lets an earlier one have the room. Spending
    first-come and stubbing whatever is left over shipped 832 characters against an
    800-character limit -- the stub it wrote for the second result was never budgeted
    for. Asserted on the request after the sanitiser has run, because that is the one
    the provider receives.

    Five batches whose totals differ, so a pass that always stubs and a pass that never
    stubs each redden a row, and the reported set is checked against what the request
    actually delivers rather than against the loop's own intermediate arithmetic.
    """
    for max_prompt_tokens, sizes in (
        (200, (500, 400)),
        (200, (700, 700)),
        (300, (600, 500)),
        (200, (300, 10_000)),
        (400, (300, 200)),
    ):
        ModelFamily.set_dynamic_specs(
            {
                "test.model": {
                    "full_model": {"max_prompt_tokens": max_prompt_tokens},
                    "context_length": max_prompt_tokens,
                }
            }
        )
        limit_chars = max_prompt_tokens * _CHARS_PER_TOKEN_HEURISTIC
        call_ids = [f"call-{i}" for i in range(len(sizes))]
        outputs = [
            {"type": "function_call_output", "call_id": cid, "output": "a" * n}
            for cid, n in zip(call_ids, sizes)
        ]
        outcome, shipped = _dispatch([], outputs)
        delivered = _delivered(shipped)

        assert estimate_serialized_chars(shipped) <= limit_chars, (
            f"{sizes} against a {limit_chars} char limit shipped "
            f"{estimate_serialized_chars(shipped)} chars"
        )
        assert outcome.omitted_call_ids == set(call_ids) - delivered, (
            f"{sizes} against a {limit_chars} char limit: the pass reported "
            f"{sorted(outcome.omitted_call_ids)} but the shipped request delivered "
            f"{sorted(delivered)}. Spending the budget first-come lets an earlier "
            "result take room a later result's stub needs; the sanitiser then reclaims "
            "it from the earlier result, and the pass has already reported the wrong "
            "set -- the in-chat notification names a tool the model did receive, and "
            "citation harvesting stores one it never read"
        )


@pytest.mark.parametrize(
    ("tokens", "kept"),
    [(785, 3), (780, 2)],
    ids=["excess-fits", "excess-does-not-fit"],
)
def test_a_kept_result_costs_only_what_it_exceeds_its_own_stub_by(
    tokens: int, kept: int
) -> None:
    """Every result already has its stub reserved, so keeping one costs the difference.

    The floor charges each result `min(len, len(stub))` before the loop starts. A result
    the loop then decides to keep does not cost its whole length on top of that -- it
    costs only what it exceeds the reservation by. Charging the full length again would
    double-count the reserved part and stub results that comfortably fit.

    Each 900-character result reserves min(900, 127) = 127 and carries 773 of excess,
    2,319 in total. The two rows sit either side of that: at 785 tokens there are 2,321
    characters of room and all three survive, at 780 there are 2,301 and one must go.
    Charging the full 900 stubs the third, so the first row reddens and the second does
    not.

    The fixture pairs each output with its `function_call`, because that is the shape the
    provider receives -- `_validate_tool_call_pairs` drops an orphaned output. The
    boundary was re-derived by sweep for that shape rather than carried over: bare it sat
    at 724 tokens, paired it sits at 785, so the old rows were calibrated against a
    request that cannot exist.
    """
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": tokens}, "context_length": tokens}}
    )
    items = [
        {"type": "function_call_output", "call_id": f"c{i}", "output": "R" * 900}
        for i in range(3)
    ]
    apply_replay_tool_output_budget(_with_calls(items), model_id="test/model")
    survived = sum(1 for i in items if not is_tool_omission_stub(i["output"]))
    assert survived == kept, (
        f"at {tokens} tokens {survived} of 3 results survived, expected {kept}; a kept "
        "result is being charged for bytes its own floor already reserved"
    )


@pytest.mark.parametrize(
    ("outputs", "expect_futile"),
    [(20, True), (4, False)],
    ids=["status-tips-it-over", "room-to-spare"],
)
def test_the_baseline_carries_every_key_the_request_will_ship(
    outputs: int, expect_futile: bool
) -> None:
    """The shipped shape is `{type, call_id, output, status}` -- all four, not three.

    The baseline exists to measure what a request costs with its tool bodies emptied, so
    it has to carry exactly the keys `_strip_tool_item_extras` lets through. Dropping the
    display-only keys was the fix; dropping `status` with them would be the same error in
    the other direction, under-charging about 22 characters per result and shipping over.

    Twenty 40-character results with a status reserve 2,570 characters and without one
    2,110, so a 2,240-character limit sits between the two verdicts: the first row is
    futile only because the statuses are counted. The second row has room either way, so
    a constant answer fails one of them.
    """
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": 560}, "context_length": 560}}
    )
    items = [
        {
            "type": "function_call_output",
            "call_id": f"c{i}",
            "output": "R" * 40,
            "status": "completed",
        }
        for i in range(outputs)
    ]
    outcome = apply_replay_tool_output_budget(items, model_id="test/model")
    assert outcome.futile is expect_futile, (
        f"{outputs} results with a status reserved {outcome.irreducible_chars} characters "
        f"against a {outcome.limit_chars}-character limit; the baseline is not carrying "
        "the same keys the sanitiser ships"
    )


@pytest.mark.parametrize(
    ("limit_tokens", "result_chars", "files_kb", "embed_kb"),
    [(1_000, 500, 4, 0), (1_000, 1_200, 0, 8), (4_000, 900, 64, 32)],
)
def test_a_tool_result_is_budgeted_for_what_ships_not_for_its_ui_attachments(
    limit_tokens: int, result_chars: int, files_kb: int, embed_kb: int
) -> None:
    """Open WebUI hangs display payloads off a tool result that never reach the provider.

    `process_tool_result` puts whole base64 `data:image/...` strings into `files` and the
    entire decoded body of an inline HTML response into `embeds`, and the pipe forwards
    both. `_strip_tool_item_extras` then reduces every `function_call_output` to
    `{type, call_id, output, status}` before the request ships, so neither key leaves the
    pipe -- but the budget's baseline kept every key except `output`, so it charged them.

    Measured before the fix on a 4,000-character limit: 4 KB of `files` -- one small PNG
    from a tool -- took the irreducible cost from 372 to 4,546 characters, stubbed a
    500-character result, and told the user the conversation could not be made to fit,
    for a request that ships at 367 characters.

    The rows vary which key carries the weight and at which limit, so no constant
    satisfies them, and each compares a laden output against a bare one holding
    everything else equal -- the assertion is on the budget's arithmetic, not on whether
    the keys reach the wire, which the sanitiser already guarantees downstream.
    """
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": limit_tokens},
                        "context_length": limit_tokens}}
    )
    existing = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]
    bare = {
        "type": "function_call_output",
        "call_id": "c1",
        "output": "R" * result_chars,
        "status": "completed",
    }
    laden: dict[str, Any] = dict(bare) | {"id": "fco-1"}
    if files_kb:
        laden["files"] = [{"type": "image", "url": "data:image/png;base64," + "A" * (files_kb * 1024)}]
    if embed_kb:
        laden["embeds"] = [{"type": "html", "content": "<div>" + "x" * (embed_kb * 1024) + "</div>"}]

    bare_items, laden_items = [dict(bare)], [dict(laden)]
    bare_out = apply_live_tool_output_budget(
        bare_items, existing_input_items=existing, model_id="test/model"
    )
    laden_out = apply_live_tool_output_budget(
        laden_items, existing_input_items=existing, model_id="test/model"
    )

    assert laden_out.irreducible_chars == bare_out.irreducible_chars, (
        f"{files_kb} KB of files and {embed_kb} KB of embeds added "
        f"{laden_out.irreducible_chars - bare_out.irreducible_chars} characters to a budget "
        "for content the pipe strips before sending"
    )
    assert laden_out.futile == bare_out.futile
    assert laden_out.omitted_call_ids == bare_out.omitted_call_ids
    assert is_tool_omission_stub(laden_items[0]["output"]) is is_tool_omission_stub(
        bare_items[0]["output"]
    ), "a display-only attachment decided whether the model saw the result"


@pytest.mark.parametrize(
    ("results", "result_chars", "expect_futile"),
    [(20, 2, False), (50, 2, False), (70, 9_000, True)],
    ids=["few-tiny", "many-tiny", "genuinely-too-big"],
)
def test_a_result_is_never_reserved_more_room_than_it_occupies(
    results: int, result_chars: int, expect_futile: bool
) -> None:
    """A two-character result cannot cost a placeholder's worth of budget.

    The floor reserves `min(len(result), len(its placeholder))`. Drop the `min` and every
    result is reserved 123 characters whatever its size, so fifty results totalling 100
    characters of output reserve 13,530 against an 8,000-character limit -- rather than
    the 7,480 they really need -- and the pass reports that the conversation cannot fit.
    The user is told to shorten a message over 100 characters of tool output.

    The fixture pairs each output with its `function_call`, the shape the provider
    receives. Paired, the futility boundary is 54 results and the row sits at 50; bare it
    was 118, so the old row at 60 was nowhere near the edge it claims to test.

    The two short-result tests already in this file cannot see it: an inflated floor makes
    the excess negative, which still satisfies `excess <= remaining`, so the result is kept
    either way. The damage lands on the futility verdict, so that is what this asserts.
    The third row is genuinely too big, so neither a constant True nor a constant False
    satisfies the set.
    """
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": 2_000}, "context_length": 2_000}}
    )
    items = [
        {"type": "function_call_output", "call_id": f"c{i}", "output": "o" * result_chars}
        for i in range(results)
    ]
    outcome = apply_replay_tool_output_budget(_with_calls(items), model_id="test/model")
    assert outcome.futile is expect_futile, (
        f"{results} results of {result_chars} characters reserved "
        f"{outcome.irreducible_chars} against a {outcome.limit_chars}-character limit"
    )


@pytest.mark.parametrize(
    ("max_prompt_tokens", "expect_futile"),
    [(136, True), (137, False)],
    ids=["floor-exactly-fills-the-limit", "floor-one-token-under"],
)
def test_a_floor_that_exactly_fills_the_budget_is_futile(
    max_prompt_tokens: int, expect_futile: bool
) -> None:
    """Equality is the interesting case, and it was decided by an unpinned character.

    `irreducible_chars >= prompt_limit_chars` is what makes the verdict; changing it to
    `>` left the entire package suite green. That matters more than a one-off boundary
    usually would, because the row that decides WHICH pass's verdict the user sees --
    `live-pass-alone-says-hopeless` in the streaming tests -- sits exactly on this
    equality. A one-character change there silently turns that row into a duplicate of
    the row above it and disarms the only check on the notice's source, with nothing red.

    A request whose floor exactly fills the window has nothing left for a single token of
    the answer, so it cannot be brought under the limit and futile is the honest verdict.
    The floor is asserted before the verdict, so if the placeholder wording changes and
    the fixture drifts off the boundary this fails loudly instead of quietly testing
    nothing.
    """
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {"max_prompt_tokens": max_prompt_tokens},
                "context_length": max_prompt_tokens,
            }
        }
    )
    items = _with_calls(
        [
            {"type": "function_call_output", "call_id": f"c{i}", "output": "z" * 200}
            for i in range(2)
        ]
    )

    outcome = apply_replay_tool_output_budget(items, model_id="test/model")

    assert outcome.irreducible_chars == 544, (
        f"regime broken: this fixture's floor is {outcome.irreducible_chars}, not the 544 "
        f"that sits exactly on a {136}-token limit, so neither row is on the boundary any "
        "more and the comparison is unpinned again"
    )
    assert outcome.futile is expect_futile, (
        f"a floor of {outcome.irreducible_chars} against a limit of {outcome.limit_chars} "
        f"reported futile={outcome.futile}"
    )


@pytest.mark.parametrize(
    ("tokens", "results", "expect_futile"),
    [(500, 12, True), (500, 6, False), (1_000, 12, False), (2_000, 12, False)],
)
def test_a_pass_that_does_not_report_futility_leaves_the_request_under_the_limit(
    tokens: int, results: int, expect_futile: bool
) -> None:
    """The guard reserved the stubs already written and not the ones about to be written.

    A stub is not free -- it is about 128 characters -- so a pass that replaces eleven
    results adds ~1,400 characters the budget never accounted for. Measured before the
    fix: twelve 900-character results against a 2,000-character limit were trimmed to
    eleven stubs and shipped 3,082 characters, 54% over, and the guard did not fire
    because it only counted results that were ALREADY stubs.

    The rows straddle the boundary on purpose. Where the floor genuinely exceeds the
    limit the pass must say so rather than trim to no purpose; everywhere else the
    request it produces must actually fit. Reporting futility everywhere fails the
    lower three rows, and reporting it nowhere fails the first.
    """
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": tokens}, "context_length": tokens}}
    )
    items = [
        {"type": "function_call_output", "call_id": f"c{i}", "output": "R" * 900}
        for i in range(results)
    ]
    limit_chars = tokens * _CHARS_PER_TOKEN_HEURISTIC

    outcome = apply_replay_tool_output_budget(items, model_id="test/model")

    assert outcome.futile is expect_futile, (
        f"{results} results against a {limit_chars}-char limit: futile={outcome.futile}"
    )
    if outcome.futile:
        assert outcome.irreducible_chars >= limit_chars, (
            f"the pass reported futility while its own floor ({outcome.irreducible_chars}) "
            f"still fit inside the limit ({limit_chars})"
        )
        assert outcome.omitted_call_ids, (
            "the pass reported it could not win and then trimmed nothing, so a request it "
            "already knew was too large went out at full size"
        )
        shipped = estimate_serialized_chars(items)
        assert shipped <= outcome.irreducible_chars + 200, (
            f"a pass that reported futility shipped {shipped} chars, above its own computed "
            f"floor of {outcome.irreducible_chars}. This is the discontinuity: request size "
            "must stay smooth as the conversation grows, not jump when the verdict flips."
        )
    else:
        shipped = estimate_serialized_chars(items)
        assert shipped <= limit_chars, (
            f"the pass reported success and shipped {shipped} chars against a "
            f"{limit_chars}-char limit -- over by {shipped - limit_chars}"
        )


@pytest.mark.parametrize("later_stubs", [1, 2, 3, 5])
def test_a_later_stub_is_not_free_budget_for_an_earlier_result(later_stubs: int) -> None:
    """The same conversation, replayed in a different order, must ship the same request.

    The guard reserved every existing stub up front while the loop reserved none of
    them, subtracting each only as it walked past. So a large result sitting BEFORE the
    stubs was measured against a budget that still counted their characters as free, and
    was kept; the same result sitting after them was stubbed. Measured on a chronological
    transcript -- an early result followed by later oversized calls -- the request shipped
    up to 645 characters over a 2,000-character limit and the provider rejected it.

    Reversal is the witness, not the contract. The loop is greedy first-fit, which IS
    order-dependent when results differ in size -- over 300 randomised four-item size
    sets, 175 gave different verdicts forward versus reversed. What this fixture pins is
    narrower and is what the defect actually was: a stub appearing LATER in the list must
    not be treated as free budget by a result that precedes it. Replay order is
    chronological and stable, so the greedy order-dependence is not a runtime defect;
    claiming general order-independence here would be.

    The small result that must survive is what stops "omit everything" from satisfying
    the reversal check -- that answer is order-independent too, and trivially under the
    limit.
    """
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": 500}, "context_length": 500}}
    )
    stub = build_replayed_tool_omission_stub(result_chars=900_000, remaining_tokens=0)

    def build() -> list[dict]:
        items: list[dict] = [
            {"type": "function_call_output", "call_id": "big", "output": "R" * 1_700},
            {"type": "function_call_output", "call_id": "small", "output": "s" * 40},
        ]
        items += [
            {"type": "function_call_output", "call_id": f"s{i}", "output": stub}
            for i in range(later_stubs)
        ]
        return items

    forward, backward = build(), list(reversed(build()))
    apply_replay_tool_output_budget(forward, model_id="test/model").omitted_call_ids
    apply_replay_tool_output_budget(backward, model_id="test/model").omitted_call_ids

    verdicts = lambda items: sorted(
        (i["call_id"], is_tool_omission_stub(i["output"])) for i in items
    )
    assert verdicts(forward) == verdicts(backward), (
        f"with {later_stubs} later stub(s) the outcome changed with the order: "
        f"{verdicts(forward)} vs {verdicts(backward)}"
    )
    for label, items in (("forward", forward), ("reversed", backward)):
        small = next(i for i in items if i["call_id"] == "small")
        assert not is_tool_omission_stub(small["output"]), (
            f"the {label} pass omitted a 40-character result that always fits. Omitting "
            "everything is perfectly order-independent, so without this the reversal "
            "assertion is satisfied by a pass that trims indiscriminately."
        )
    assert estimate_serialized_chars(forward) <= 500 * _CHARS_PER_TOKEN_HEURISTIC, (
        f"the pass declared the request in bounds and shipped "
        f"{estimate_serialized_chars(forward)} chars against a "
        f"{500 * _CHARS_PER_TOKEN_HEURISTIC} char limit"
    )


_EXISTING_LIVE_STUB = build_live_tool_omission_stub(result_chars=4_000_000, remaining_tokens=0)


def _live_items(fresh_chars: int, *, stub_first: bool) -> list[dict]:
    fresh = {"type": "function_call_output", "call_id": "fresh", "output": "z" * fresh_chars}
    already = {"type": "function_call_output", "call_id": "already", "output": _EXISTING_LIVE_STUB}
    return [already, fresh] if stub_first else [fresh, already]


def test_a_live_pass_does_not_report_a_stub_it_did_not_write() -> None:
    """The live twin of the replay contract had no witness at all.

    Deleting this branch, re-adding the `omitted_call_ids.add` the changeset removed, or
    dropping its budget subtraction all left 675 tests green. The set feeds the in-chat
    notification that names the tools whose results the model did not receive, so
    reporting a result this pass never touched announces a tool that was, in fact,
    delivered -- and does it again on every later loop iteration.
    """
    for max_prompt_tokens, fresh_chars, expect_named in ((200, 40, False), (200, 4_000, True)):
        ModelFamily.set_dynamic_specs(
            {
                "test.model": {
                    "full_model": {"max_prompt_tokens": max_prompt_tokens},
                    "context_length": max_prompt_tokens,
                }
            }
        )
        outputs = _live_items(fresh_chars, stub_first=True)
        outcome, shipped = _dispatch([], outputs)

        assert ("fresh" in outcome.omitted_call_ids) is expect_named, (
            f"the pass reported {sorted(outcome.omitted_call_ids)} for a {fresh_chars} "
            f"char result against a {max_prompt_tokens * _CHARS_PER_TOKEN_HEURISTIC} "
            "char limit; it must name a result when, and only when, it replaced it"
        )
        assert "already" not in outcome.omitted_call_ids, (
            "the pass named a result that arrived already stubbed, so the in-chat "
            "notification announces a tool whose result was, in fact, delivered"
        )
        assert outcome.omitted_call_ids == {"fresh", "already"} - _delivered(shipped) - {"already"}, (
            "the ids the pass reports must be the ids the shipped request carries a "
            "stub for; the notification names the tools the model did not receive"
        )
        assert outputs[0]["output"] == _EXISTING_LIVE_STUB, (
            "a result that arrived already stubbed was stubbed again"
        )


@pytest.mark.parametrize("fresh_chars", [400, 900], ids=["fits-nothing", "far-over"])
def test_a_live_pre_existing_stub_still_costs_its_own_characters(fresh_chars: int) -> None:
    """Skipping a stub must not mean skipping its cost -- and order must not decide.

    The stub is 205 characters and the budget is 500. Charging the stub nothing lets
    the fresh result through and ships 743 characters; charging it only when it
    happens to come first makes the same two items ship 537 characters in one order
    and 743 in the other. Both rows carry the same two items in both orders, so the
    only thing they can be satisfied by is a budget that reserves the stub's cost
    before it spends anything.

    Asserted on the request after the sanitiser has run, because a limit the
    intermediate state respects and the shipped request does not is not a limit.
    """
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": 125}, "context_length": 125}}
    )
    limit_chars = 125 * _CHARS_PER_TOKEN_HEURISTIC
    seen = {}
    for stub_first in (True, False):
        outputs = _live_items(fresh_chars, stub_first=stub_first)
        outcome, shipped = _dispatch([], outputs)
        size = estimate_serialized_chars(shipped)

        ceiling = max(limit_chars, outcome.irreducible_chars)
        assert size <= ceiling, (
            f"stub_first={stub_first}: a {fresh_chars} char result behind a 205 char "
            f"stub shipped {size} chars against a ceiling of {ceiling}; the stub ahead "
            "of it was treated as free. Where the floor itself exceeds the limit the "
            "pass cannot win, but it must still trim down to that floor -- giving up "
            "instead is what made request size jump 3.4x on two characters of extra "
            "user text."
        )
        already = next(o for o in outputs if o["call_id"] == "already")
        assert already["output"] == _EXISTING_LIVE_STUB, "the existing stub was rewritten"
        seen[stub_first] = (size, frozenset(_delivered(shipped)))

    assert seen[True] == seen[False], (
        f"the same two items shipped {seen[True]} in one order and {seen[False]} in "
        "the other; which result survives must follow the budget, not the arrival order"
    )


_EXISTING_REPLAY_STUB = build_replayed_tool_omission_stub(
    result_chars=4_000_000, remaining_tokens=0
)


@pytest.mark.parametrize(
    ("max_prompt_tokens", "fresh_chars", "fresh_is_omitted", "first_pass_reports"),
    [
        (220, 700, True, {"fresh"}),
        (10_000, 700, False, set()),
        (220, 430, False, set()),
    ],
    ids=["budget-exhausted", "budget-ample", "budget-spent-by-a-kept-result"],
)
def test_replay_budget_is_idempotent_for_existing_stub(
    max_prompt_tokens: int,
    fresh_chars: int,
    fresh_is_omitted: bool,
    first_pass_reports: set,
) -> None:
    """A result already omitted must replay byte-for-byte, and a pass that changes
    nothing must report nothing.

    The fixture needs two items. With one, `remaining < len(stub)` holds exactly when
    `limit < irreducible` -- which is the futility guard's own condition -- so a
    single-item test can never reach the short-circuit in a state where it matters. The
    earlier version of this test proved that the hard way: it sat inside the guard, its
    subject never executed, and deleting the branch it is named for changed nothing
    anywhere in the suite. The fresh result in front is what spends the budget so the
    stub behind it is reached with the budget already gone.

    The fresh result is also the discriminator: it is stubbed in one row and kept in the
    other, so no constant return satisfies both. `second` is empty in BOTH rows, which is
    the contract -- the set names what this invocation replaced, not what is currently
    omitted, so an untouched stub is never in it and a second pass over an unchanged
    request reports an empty set rather than repeating the first pass's answer.

    What leaves the stub alone is `_output_floor_chars`: a result that is already a stub
    reserves its own full length, so its excess over that reservation is zero and the
    size test keeps it whatever the remaining budget is. An earlier version of this
    paragraph named a separate `is_tool_omission_stub` branch in the loop as the
    mechanism. That branch existed but decided nothing -- the floor had already made the
    outcome the same -- and deleting it changed no test. It is gone; the floor is the
    thing to break if you want this row to redden, and `_output_floor_chars` returning
    zero for a stub is CAUGHT by
    `test_a_later_stub_is_not_free_budget_for_an_earlier_result`.
    """
    ModelFamily.set_dynamic_specs(
        {
            "test.model": {
                "full_model": {"max_prompt_tokens": max_prompt_tokens},
                "context_length": max_prompt_tokens,
            }
        }
    )
    items = [
        {"type": "function_call_output", "call_id": "fresh", "output": "z" * fresh_chars},
        {"type": "function_call_output", "call_id": "already", "output": _EXISTING_REPLAY_STUB},
    ]
    probe = apply_replay_tool_output_budget(
        _with_calls([dict(item) for item in items]), model_id="test/model"
    )
    assert probe.futile is False, (
        f"regime broken: the pass reports it cannot bring this request under the limit "
        f"({probe.irreducible_chars} against {probe.limit_chars}), which is not the regime "
        "these rows are written for. The precondition asks the pass itself rather than "
        "recomputing the floor here -- an earlier version used a formula production had "
        "stopped using, and the two disagreed by 127 characters while both happened to "
        "sit under the limit."
    )

    first = apply_replay_tool_output_budget(
        _with_calls(items), model_id="test/model"
    ).omitted_call_ids
    after_first = [dict(item) for item in items]
    second = apply_replay_tool_output_budget(
        _with_calls(items), model_id="test/model"
    ).omitted_call_ids

    assert items[1]["output"] == _EXISTING_REPLAY_STUB, (
        "an existing stub was stubbed again, so the replayed text drifts every turn"
    )
    assert is_tool_omission_stub(items[0]["output"]) is fresh_is_omitted, (
        "the fresh result's fate did not follow the budget regime"
    )
    assert first == first_pass_reports, (
        f"the first pass reported {sorted(first)}; it must name exactly the call ids whose "
        "output it replaced"
    )
    assert second == set(), (
        f"a pass that changed nothing reported {sorted(second)}. The caller uses this set "
        "to decide whether the request was altered and to count it in a log line, so "
        "reporting a stub it did not touch overstates both."
    )
    assert items == after_first, "a second pass over an unchanged request changed it"


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
    ).omitted_call_ids

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
    ).omitted_call_ids

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

    tiny = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": "data:image/png;base64,AAAA"}],
        }
    ]
    assert estimate_serialized_chars(items) == estimate_serialized_chars(tiny)

    text_only = estimate_serialized_chars(
        [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "AAAA"}],
            }
        ]
    )
    charge = estimate_serialized_chars(tiny) - text_only
    assert 4_000 <= charge <= 12_000, (
        f"an inlined image is charged {charge} chars. Equality between two payload sizes "
        "is satisfied by any constant, including an empty placeholder -- which would let "
        "an attachment cost nothing at all, the mirror of the bug this fixes. The charge "
        "has to be real and bounded, so a band rather than an exact figure: the exact "
        "number also encodes JSON key overhead and would break on an unrelated rename."
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
    ).omitted_call_ids

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
    omitted = apply_replay_tool_output_budget(items, model_id="test/model").omitted_call_ids

    assert omitted == set(), "a replayed result shorter than the stub was replaced by it"
    assert items[1]["output"] == "z" * 40


@pytest.mark.parametrize(
    "block",
    [
        {"type": "input_image", "image_url": "data:x;base64," + "A" * 1_400_000},
        {"type": "image_url", "image_url": {"url": "data:x;base64," + "A" * 1_400_000}},
        {"type": "input_audio", "input_audio": {"data": "A" * 1_400_000, "format": "mp3"}},
        {"type": "input_file", "file_data": "data:application/pdf;base64," + "A" * 1_400_000},
        {"type": "video_url", "video_url": {"url": "data:x;base64," + "A" * 1_400_000}},
    ],
    ids=["input_image", "image_url", "input_audio", "input_file", "video_url"],
)
def test_an_attachment_does_not_switch_tool_trimming_off(block) -> None:
    """A payload the provider does not read as text must be charged at a flat rate.

    The `input_file` row names its media type explicitly. A bare `data:x;base64,` took
    the binary branch by accident, which hid that `_document_tokens` has two branches at
    all -- and a textual document genuinely is charged its literal length, deliberately,
    so this property was never true of every inline payload. It is true of every payload
    whose bytes the provider does not tokenise one-for-one.

    A model bills an attachment by its resolution or duration, not by how many base64
    characters carried it. Counting the base64 literally reads one ordinary file as
    hundreds of thousands of tokens, which alone exceeds a large model's whole
    character budget -- and the futility guard then concludes the request is hopeless
    and stops trimming tool output entirely, so a genuinely oversized result ships in
    full and the provider rejects the request.

    Parametrised over every block shape the pipe can actually place in the request --
    including the nested-dict forms, because a fixture with a flat string where
    production emits `{"data": ..., "format": ...}` passes while the real attachment is
    still counted by its base64 length.
    """
    _spec(128_000)
    existing = [
        {
            "type": "message",
            "role": "user",
            "content": [block],
        }
    ]
    outputs = [{"type": "function_call_output", "call_id": "big", "output": "z" * 900_000}]

    omitted = apply_live_tool_output_budget(
        outputs, existing_input_items=existing, model_id="test/model"
    ).omitted_call_ids

    assert omitted == {"big"}, (
        f"one {block['type']} attachment disabled tool-output trimming for the whole turn"
    )


@pytest.mark.parametrize(
    ("fresh_chars", "expect_stub"),
    [(100, False), (400, False), (900, True), (2000, True)],
)
def test_results_already_in_the_request_consume_the_remaining_budget(
    fresh_chars, expect_stub
) -> None:
    """A result accepted on an earlier tool loop is charged what it will really cost.

    The baseline blanks every tool output so the futility guard does not treat a
    shrinkable result as hopeless -- but the live pass then charged the same results
    at their FULL length, so the guard and the loop disagreed about the same items.
    Charging them at full length is not conservative, it is wrong in both directions:
    at 200 characters the pass approved a result the sanitiser then replaced, and the
    pipe cited and stored a result the model never read; at 600 it replaced a result
    the sanitiser would have kept, and the request shipped neither.

    The expectations are not hand-chosen. Each row asserts against the counterfactual
    -- what the sanitiser ships when the live pass touches nothing -- so the row is
    checkable against the request the user's model actually receives rather than
    against the loop's own arithmetic. The discriminating axis is the fresh result's own
    size: the budget spends newest-first, so a stale result no longer decides the fate of
    a fresh one, and what remains observable is whether the fresh result fits at all.

    The fixture pairs each output with the `function_call` production always puts ahead
    of it. Without that the rows were calibrated against a request the provider never
    receives -- `_validate_tool_call_pairs` drops an orphaned output -- and two of the
    three rows gave a different answer under the shape that actually ships. The limit was
    re-derived by sweep rather than carried over: at 240 tokens the three rows disagree
    exactly as they did before, non-monotonically, which is what the paragraph above is
    about, and the live pass agrees with the sanitiser-alone counterfactual there.
    """
    _spec(240)
    existing = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "function_call_output", "call_id": "earlier", "output": "y" * 600},
    ]
    outputs = [{"type": "function_call_output", "call_id": "new", "output": "z" * fresh_chars}]

    counterfactual = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "function_call_output", "call_id": "earlier", "output": "y" * 600},
        {"type": "function_call_output", "call_id": "new", "output": "z" * fresh_chars},
    ]
    counterfactual = _with_calls(counterfactual)
    apply_replay_tool_output_budget(counterfactual, model_id="test/model")

    outcome, shipped = _dispatch(existing, outputs)
    delivered = _delivered(shipped)

    assert ("new" in delivered) is not expect_stub, (
        f"a {fresh_chars} char fresh result left the shipped request "
        f"{'carrying' if 'new' in delivered else 'without'} the fresh 400 char result"
    )
    assert delivered == _delivered(counterfactual), (
        f"the live pass shipped {sorted(delivered)} where the sanitiser on its own "
        f"ships {sorted(_delivered(counterfactual))}; the live pass may choose the "
        "wording of a stub and which ids it reports, but not which results survive -- "
        "it cannot see less room than the request will actually have"
    )
    assert ("new" in outcome.omitted_call_ids) is ("new" not in delivered), (
        f"the pass reported {sorted(outcome.omitted_call_ids)} but the shipped request "
        f"delivered {sorted(delivered)}; that set gates citation harvesting, so a "
        "result the model did receive must not be announced as dropped, and one it "
        "never received must not be cited and stored as though it had been"
    )
    assert estimate_serialized_chars(shipped) <= 240 * _CHARS_PER_TOKEN_HEURISTIC


@pytest.mark.parametrize(
    ("fresh_chars", "expect_new_stubbed"),
    [(100, False), (400, False), (900, True), (2000, True)],
)
def test_the_live_pass_leaves_results_it_was_not_handed_alone(
    fresh_chars, expect_new_stubbed
) -> None:
    """The live pass may decide the fate of its own outputs, not of the ones already sent.

    `apply_live_tool_output_budget` measures its outputs against the request they will
    join, so it walks `existing_input_items` too. It used to rewrite those items as well,
    replacing a result the user watched arrive on an earlier tool loop with a placeholder
    -- and it deliberately kept those ids out of `omitted_call_ids`, which is the set that
    drives the "the model did not receive" notice and gates citation harvesting. So the
    result vanished from the outgoing request and nobody was told. Measured over
    `prior_chars` 376..899 that happened for every value, with an empty report every time.

    The `earlier` assertion is the same in every row, so the `new` column carries the
    discrimination: a pass that never stubs anything fails row 200, and one that stubs
    everything fails the other three. The replay pass that runs next re-decides `earlier`
    under its own floors, which is why leaving it alone here loses nothing.
    """
    _spec(240)
    earlier = {
        "type": "function_call_output",
        "call_id": "earlier",
        "output": "y" * 600,
    }
    new = {"type": "function_call_output", "call_id": "new", "output": "z" * fresh_chars}
    paired = _with_calls(
        [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
            earlier,
        ]
    ) + [{"type": "function_call", "call_id": "new", "name": "lookup", "arguments": "{}"}]

    outcome = apply_live_tool_output_budget(
        [new], existing_input_items=paired, model_id="test/model"
    )

    assert not is_tool_omission_stub(earlier["output"]), (
        f"a 600 char result already in the request was replaced with a "
        "placeholder by the live pass, which was not handed it; the user watched that "
        "result arrive on an earlier tool loop"
    )
    assert is_tool_omission_stub(new["output"]) is expect_new_stubbed, (
        f"the live pass was handed the fresh result and "
        f"{'kept' if not is_tool_omission_stub(new['output']) else 'stubbed'} it"
    )
    assert ("new" in outcome.omitted_call_ids) is expect_new_stubbed, (
        f"the pass reported {sorted(outcome.omitted_call_ids)} for a result it "
        f"{'stubbed' if expect_new_stubbed else 'kept'}; that set gates citation "
        "harvesting and the in-chat notice"
    )


def test_a_block_without_a_payload_is_not_charged_for_one() -> None:
    """The table selects a rate; it must not invent the payload.

    A block with no payload key at all -- what `_to_input_file` emits on its error path
    -- carries no bytes, so charging it the inline rate means a handful of them exceed
    a model's whole character budget and trip the futility guard with nothing inline
    anywhere in the request. A block whose payload key is present but *empty* is the
    same case: `_to_input_video` emits `{"url": ""}` on five separate failure paths.

    A `file_id` is the same case, and this test has now asserted it in both directions.
    The reference really is inlined at dispatch, so the request will carry a document --
    but nothing at estimation time bounds how big it is, and the identifier's own length
    says nothing about its target. Charging it a fixed guess is what made a handful of
    attachments exhaust a model's whole budget. The two failure directions are not
    symmetric: an over-charge silently and permanently drops the call and its result as a
    pair, while an under-charge produces a provider error with the conversation intact.

    The final pair is the discriminator. A payload the estimator can actually measure is
    still charged for its bytes, so "charge references nothing" cannot be satisfied by
    ignoring payloads, and a constant cannot satisfy both rows.
    """
    measurable = "data:application/pdf;base64," + "A" * 1_400_000

    assert estimate_serialized_chars([{"type": "input_file"}]) < 200
    assert estimate_serialized_chars([{"type": "video_url", "video_url": {"url": ""}}]) < 200
    assert estimate_serialized_chars(
        [{"type": "input_audio", "input_audio": {"data": "", "format": "mp3"}}]
    ) < 200
    assert estimate_serialized_chars([{"type": "input_file", "file_id": "abc"}]) < 200, (
        "a reference whose target size is unknowable was charged a fixed guess"
    )
    assert estimate_serialized_chars([{"type": "input_file", "file_url": "https://x/y.pdf"}]) < 200
    assert estimate_serialized_chars([{"type": "input_file", "file_data": measurable}]) > 8_000, (
        "a document whose bytes are right there was not charged for them"
    )


def test_estimating_a_request_does_not_alter_it() -> None:
    """The estimator is handed the live request body, not a copy.

    `apply_live_tool_output_budget` receives `body.input` itself, so if the shaping
    step mutated in place -- the obvious "why copy a big dict" optimisation -- every
    forwarded image would leave as a few thousand placeholder characters.
    """
    import copy

    items = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_image", "image_url": "data:image/png;base64," + "A" * 5_000}],
        }
    ]
    before = copy.deepcopy(items)

    estimate_serialized_chars(items)

    assert items == before, "estimating the request rewrote the payload it was measuring"


@pytest.mark.parametrize(
    ("block_type", "keys", "scales"),
    [
        ("input_image", ("image_url",), False),
        ("input_audio", ("input_audio",), True),
        ("input_file", ("file_data", "file_url"), True),
        ("video_url", ("video_url",), True),
    ],
)
def test_a_payload_that_scales_is_not_charged_a_constant(block_type, keys, scales) -> None:
    """A flat rate is honest for an image and dishonest for everything else.

    A model bills an image by resolution, so its cost really does saturate. Audio and
    video are billed by duration and a document by its pages, so charging a two-minute
    clip what a two-second one costs under-states it by roughly the ratio of their
    lengths -- and an under-stated request is the dangerous direction: it is approved,
    dispatched, and rejected by the provider, after the live pass has already told the
    pipe to persist and cite a result the model never read.
    """
    def one(size: int) -> int:
        payload = "data:x;base64," + "A" * size
        return estimate_serialized_chars([{"type": block_type, keys[0]: payload}])

    if scales:
        assert one(1_000) < one(10_000_000)
        payload_bytes = 10_000_000 * 3 // 4
        charged_tokens = one(10_000_000) // _CHARS_PER_TOKEN_HEURISTIC
        assert payload_bytes // 2_000 <= charged_tokens <= payload_bytes // 125, (
            f"{payload_bytes} bytes were charged {charged_tokens} tokens, outside the "
            "125-2000 bytes-per-token band. Monotonicity alone is satisfied by a divisor "
            "a thousand times too large, which is the under-count this test names as the "
            "dangerous direction: approved, dispatched, rejected by the provider."
        )
    else:
        assert one(1_000) == one(10_000_000)


@pytest.mark.parametrize("payload_chars", [1_000_000, 8_000_000])
def test_one_document_is_charged_once_however_many_keys_carry_it(payload_chars: int) -> None:
    """`_to_input_file` sets `file_data` and `file_url` in independent branches.

    Both name the same document, which the provider tokenises once. Two failures have to
    stay red and they push in opposite directions. Charging every matching key bills one
    document several times; charging only the first leaves the rest counted at full
    base64 length by the surrounding walk, which over-counts far worse. An earlier
    version asserted `both > one`, which is satisfied by either arrangement -- and under
    the charge-once rule it still passed, by the sixteen characters of `"file_url": ""`.

    Parametrised over two payload sizes so no constant charge can satisfy it.
    """
    blob = "data:x;base64," + "A" * payload_chars
    small = "data:x;base64," + "A" * (payload_chars // 8)
    huge = "data:x;base64," + "A" * (payload_chars * 8)

    alone = estimate_serialized_chars([{"type": "input_file", "file_data": blob}])
    twice = estimate_serialized_chars([{"type": "input_file", "file_data": blob, "file_url": blob}])
    largest_first = estimate_serialized_chars(
        [{"type": "input_file", "file_data": blob, "file_url": small}]
    )
    largest_second = estimate_serialized_chars(
        [{"type": "input_file", "file_data": small, "file_url": blob}]
    )
    bigger_document = estimate_serialized_chars([{"type": "input_file", "file_data": huge}])

    assert twice - alone < 100, (
        f"one document on two keys cost {twice} against {alone} for the same document on "
        "one key -- either a key escaped shaping and was counted raw, or the document "
        "was billed once per key naming it"
    )
    assert largest_first == largest_second, (
        "the charge depends on which key holds the larger payload, so the block is being "
        "charged for the first match rather than the largest"
    )
    assert abs(largest_first - alone) < 100, (
        "a block was charged for something other than its largest payload -- one-sided, "
        "this missed max() becoming min(), which charges an 8 MB document for the short "
        "URL beside it and survives the whole suite"
    )
    assert bigger_document > 4 * alone, (
        "an eight-fold larger document was not charged proportionally, so the estimate is "
        "a constant rather than a measurement"
    )


def _payload(kind: str, size: int) -> bytes:
    if kind == "text":
        return ("The quick brown fox jumps over the lazy dog. " * 4000)[:size].encode()
    if kind == "cp1252":
        return ("name,city\nRen\xe9e,Z\xfcrich\n" * 4000).encode("cp1252")[:size]
    if kind == "utf16":
        return ("hello world\n" * 6000).encode("utf-16")[:size]
    if kind == "control":
        return bytes(range(1, 9)) * (size // 8)
    if kind == "nul":
        return (b"abc " * 50 + b"\x00" + b"the quick brown fox. " * 4000)[:size]
    if kind == "ansi":
        return ("\x1b[32mPASS\x1b[0m \x1b[1msrc/a.ts\x1b[0m\n" * 8000)[:size].encode()
    if kind == "classic_pdf":
        head = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
        tail = (
            b"xref\n0 3\n0000000000 65535 f \n0000000015 00000 n \ntrailer\n"
            b"<< /Size 3 /Root 1 0 R >>\nstartxref\n" + b"9" * 400 + b"\n%%EOF\n"
        )
        middle = bytes(range(1, 256)) * (max(size - len(head) - len(tail), 0) // 255 + 1)
        return head + middle[: max(size - len(head) - len(tail), 0)] + tail
    if kind == "zip_directory_tail":
        compressed = bytes(range(0x80, 0x100)) * (max(size - 900, 0) // 128 + 1)
        directory = (
            b"PK\x01\x02" + b"word/document.xml" * 30 + b"PK\x05\x06"
        )
        room = max(size - len(directory), 0)
        return compressed[:room] + directory
    if kind == "tail_nul_text":
        body = ("the quick brown fox jumps over the lazy dog. " * 4000).encode()[:size]
        return body[:-100] + b"\x00" + body[-99:]
    if kind == "ascii_headed_deflate":
        head = (b"1 0 obj << /Type /Catalog >> endobj " * 20)[:600]
        return (head + bytes(range(256)) * (size // 256 + 1))[:size]
    if kind == "ascii_wrapped_pdf":
        head = b"%PDF-1.4\r\n" + b"1 0 obj\r\n<< /Type /Catalog /Pages 2 0 R >>\r\nendobj\r\n" * 12
        tail = (
            b"xref\r\n0 25\r\n"
            + b"".join(b"%010d 00000 n \r\n" % (i * 137) for i in range(25))
            + b"trailer\r\n<< /Size 25 /Root 1 0 R >>\r\nstartxref\r\n900123\r\n%%EOF\r\n"
        )
        room = max(size - len(head) - len(tail), 0)
        return head + (b"\x00\x80\xff\x01" * (room // 4 + 1))[:room] + tail
    if kind == "rtf":
        return (
            rb"{\rtf1\ansi\deff0 {\fonttbl{\f0 Times New Roman;}}\viewkind4\uc1\pard\f0\fs24 "
            + b"Hello world. " * 40000
        )[:size]
    if kind == "ascii_headed_binary":
        head = (b"%PDF-1.7\n" + b"1 0 obj\n<< /Type /Catalog >>\nendobj\n" * 20)[:600]
        return (head + bytes(range(256)) * (size // 256 + 1))[:size]
    if kind == "density_16":
        return (b"a" * 15 + b"\x01") * (size // 16)
    if kind == "density_32":
        return (b"a" * 31 + b"\x01") * (size // 32)
    if kind == "crlf_csv":
        return ("id,name,amount\r\n" * 12000).encode("ascii")[:size]
    if kind == "tsv":
        return ("a\tb\tc\td\n" * 20000).encode("ascii")[:size]
    if kind == "utf16be":
        return b"\xfe\xff" + ("hello world\n" * 8000).encode("utf-16-be")[: size - 2]
    if kind == "sparse_control":
        body = ("the quick brown fox. " * 4000).encode()[:size]
        return body[:12] + b"\x07" + body[13:]
    if kind == "split":
        return ("x" + "\u00e9" * 60000).encode()[:size]
    return (b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * (size // 256 + 1))[:size]


def _charge(raw: bytes, filename: str, media_type: str = "application/octet-stream") -> int:
    block: dict = {
        "type": "input_file",
        "file_data": f"data:{media_type};base64," + base64.b64encode(raw).decode("ascii"),
    }
    if filename:
        block["filename"] = filename
    return estimate_serialized_chars([block])


@pytest.mark.parametrize("filename", ["a.txt", "a.bin", ""], ids=["txt", "bin", "none"])
@pytest.mark.parametrize("kind", ["text", "binary"])
def test_an_attachment_is_charged_for_its_bytes_not_its_name(kind: str, filename: str) -> None:
    """The same 100,000 bytes cost 125x more as `report.txt` than as `report.bin`.

    `application/octet-stream` is what a client sends when it has no type to give -- the
    pipe writes that spelling itself and so does Open WebUI, so it is the common case.
    The charge for it was decided by the filename, a label the client supplies and the
    pipe never verifies, which made a rename a 125x swing in a number the futility guard
    reads. Six rows: two payloads across three names, so the name cannot be what answers.

    Both payloads carry positive byte evidence -- printable ASCII, or a NUL -- so the
    sniff reaches a verdict and the name never gets a vote. The undecided case, where the
    name legitimately decides, is `test_an_undecidable_window_falls_back_to_the_name`.
    """
    size = 100_000
    charged = _charge(_payload(kind, size), filename)

    if kind == "text":
        assert charged > size // 2, (
            f"text bytes named {filename or 'nothing'} were charged {charged} chars; "
            "text costs about four characters per token"
        )
    else:
        assert charged < size // 50, (
            f"binary bytes named {filename or 'nothing'} were charged {charged} chars, "
            "the text rate, for a payload that is not text"
        )


@pytest.mark.parametrize(
    ("kind", "filename", "text_rate", "decided_by"),
    [
        ("control", "a.txt", False, "the control-character ratio, alone"),
        ("nul", "a.txt", False, "the NUL check, alone"),
        ("split", "a.bin", True, "the mid-character retry, alone"),
        ("cp1252", "a.csv", True, "the name, because the window is undecidable"),
        ("cp1252", "a.bin", False, "the name, because the window is undecidable"),
        ("utf16", "a.bin", True, "the little-endian UTF-16 BOM, alone"),
        ("utf16be", "a.bin", True, "the big-endian UTF-16 BOM, alone"),
        ("crlf_csv", "a.bin", True, "that a carriage return is not a control byte"),
        ("tsv", "a.bin", True, "that a tab is not a control byte"),
        ("sparse_control", "a.bin", True, "that a lone control byte is not binary evidence"),
        ("ansi", "a.bin", True, "that the ANSI escape introducer is not a control byte"),
        ("ascii_headed_binary", "a.txt", False, "the deep window, alone -- the head is all ASCII"),
        ("density_16", "a.txt", False, "the control-density cap's value, from below"),
        ("density_32", "a.bin", True, "the control-density cap's value, from above"),
        ("ascii_wrapped_pdf", "a.txt", False, "the %PDF- signature, alone -- both windows read as ASCII"),
        ("rtf", "a.bin", True, "that ASCII document markup is still text"),
        ("tail_nul_text", "a.bin", True, "the tail's control RATIO -- one stray NUL is not a veto"),
        ("ascii_headed_deflate", "a.txt", False, "the tail veto, on a dense binary tail behind a clean ASCII head"),
        ("zip_directory_tail", "a.txt", True, "the name, because the head abstains and a tail may not vouch"),
        ("zip_directory_tail", "a.bin", False, "the name, because the head abstains and a tail may not vouch"),
    ],
)
def test_each_rule_in_the_sniff_decides_a_row_by_itself(
    kind: str, filename: str, text_rate: bool, decided_by: str
) -> None:
    """Every guard in the sniff must be the sole decider somewhere, or it is unpinned.

    Nine mutations once survived this file -- the window size, the NUL check, the
    control-density gate, the mid-character retry and the base64 scan width were all
    free to change. The cause was payload choice: every row was either clean ASCII or
    invalid at byte zero, so each verdict was over-determined and the first guard always
    answered. A payload that is obviously binary trips every guard at once and therefore
    observes none of them.

    The `crlf_csv`, `tsv` and `utf16be` rows each name one member of a constant. Drop
    `\r` from `_ALLOWED_CONTROL` and a Windows-authored CSV -- one carriage return every
    nine characters -- goes from 192,062 characters to 1,598, a 120x undercharge on
    probably the most common attachment this product sees. Drop `\t` and a TSV does the
    same. Drop the big-endian entry from `_UTF16_BOMS` and a UTF-16BE file is read as
    NUL-bearing binary. All three carry `a.bin` so the filename cannot supply the answer.

    The two `cp1252` rows are the regression this file exists to prevent in the other
    direction. Windows-authored CSV is not valid UTF-8, which is absence of evidence and
    not evidence of binary; charging it binary was a 116x undercharge that cleared the
    futility guard, dropped the user's tool results, and failed at the provider anyway.
    The window is undecidable, so the name decides -- and the two rows differ only in the
    name, so a sniff that answers instead of abstaining reddens one of them.
    """
    size = 100_000
    charged = _charge(_payload(kind, size), filename)
    verdict = "the text rate" if charged > size // 4 else "the binary rate"
    expected = "the text rate" if text_rate else "the binary rate"
    assert (charged > size // 4) is text_rate, (
        f"a {kind} payload named {filename} took {verdict} ({charged} chars); this row is "
        f"decided by {decided_by}, so {expected} is the only correct answer"
    )


@pytest.mark.parametrize("filename", ["", "a.bin", "statement.pdf"])
@pytest.mark.parametrize(
    ("kind", "text_rate"),
    [("classic_pdf", False), ("text", True), ("binary", False)],
)
def test_only_the_head_window_may_vouch_for_text(
    kind: str, filename: str, text_rate: bool
) -> None:
    """A tail may veto; it may never vouch. Every conformant PDF depends on this.

    PDF 32000-1 section 7.5.2 asks producers to put four bytes above 127 on line two so
    the file survives text-mode transport -- `%\xe2\xe3\xcf\xd3` from Ghostscript and
    Acrobat, `%\xd0\xd4\xc5\xd8` from pdfTeX. That head is not valid UTF-8 and holds no
    NUL, so the sniff can reach no verdict on it. A classic cross-reference table and
    trailer are pure ASCII. An earlier reduction dropped the undecidable head and
    answered from the surviving tail alone, so `all([True])` charged the text rate:
    measured, a 2,177,800-byte PDF cost 2,177,870 characters instead of 17,490.

    Over-charging is the worse direction. That number feeds the futility guard, which
    then switches tool trimming off for the whole turn and tells the user to remove an
    attachment that needs 3.4% of the context.

    Crossed with three filenames, none of them textual, so the same bytes must get the
    same answer whoever named them.
    """
    size = 100_000
    charged = _charge(_payload(kind, size), filename)
    assert (charged > size // 4) is text_rate, (
        f"a {kind} payload named {filename or 'nothing'} was charged {charged} chars; "
        "only the head window may grant the text rate"
    )


@pytest.mark.parametrize("wrapped", [False, True], ids=["one-line", "rfc2045"])
@pytest.mark.parametrize("extra", [0, 1, 2], ids=["pad-0", "pad-2", "pad-1"])
@pytest.mark.parametrize("padded", [True, False], ids=["padded", "unpadded"])
def test_the_tail_window_decodes_a_true_suffix_of_the_payload(
    padded: bool, extra: int, wrapped: bool
) -> None:
    """The tail window is offset from the START of the stream, never from its end.

    Slicing the last N characters and trimming `len % 4` aligns the window to its own
    end, which coincides with the base64 grid only when the whole payload is a multiple
    of four -- true for padded base64, false for a client that strips `=`. Fuzzed over
    lengths 1-900: 0 of 1,042 padded tail windows were misaligned and 464 of 569
    unpadded ones were. A misaligned window decodes shifted by six bits and returns a
    confident wrong verdict, not an abstention, so a real text file flipped to the
    binary rate at 125x under.

    The `wrapped` axis pins the whitespace strip in `_payload_windows`: `_decode_window`
    strips too, so skipping it there only moves the window BOUNDARIES -- invisible until
    the payload actually carries newlines, at which point the offset is computed on raw
    character positions rather than on the base64 grid.

    The `extra` axis is load-bearing. A payload whose length is a multiple of three
    produces base64 with no padding at all, so stripping `=` is a no-op and the unpadded
    rows are silently identical to the padded ones -- which is how an earlier version of
    this test passed while the misalignment went unobserved. The three lengths give
    remainders 0, 1 and 2, and the precondition below fails loudly if a row does not
    exhibit the shape its id claims.
    """
    raw = ("the quick brown fox jumps over the lazy dog. " * 4000).encode() + b"x" * extra
    body = (
        base64.encodebytes(raw).decode("ascii")
        if wrapped
        else base64.b64encode(raw).decode("ascii")
    )
    if not padded:
        body = body.rstrip("=\n").rstrip("=")
    assert ("\n" in body) is wrapped, (
        "fixture broken: this row does not exhibit the wrapping shape its id claims"
    )
    assert (len("".join(body.split())) % 4 == 0) is (padded or extra == 0), (
        "fixture broken: this row does not exhibit the padding shape it is named for"
    )
    windows = _payload_windows(body)
    assert len(windows) == 2, "the payload is long enough to carry a distinct tail window"
    decoded_tail = _decode_window(windows[1], at_tail=True)
    assert decoded_tail, "the tail window did not decode at all"
    assert raw.endswith(decoded_tail), (
        f"the {'padded' if padded else 'unpadded'} tail window decoded "
        f"{len(decoded_tail)} bytes that are not a suffix of the payload"
    )


@pytest.mark.parametrize(
    ("shape", "text_rate"),
    [
        ("text-head-binary-middle-text-tail", True),
        ("text-head-binary-tail", False),
        ("nul-at-byte-eight", False),
    ],
)
def test_the_sniff_reads_bounded_windows_at_both_ends(shape: str, text_rate: bool) -> None:
    """The sample has to represent the payload, and reading it has to stay bounded.

    An earlier version of this test asserted the opposite of the first two rows. It built
    a payload whose first 384 bytes were ASCII and whose body was binary, and demanded the
    TEXT rate -- under the name `tail-beyond-the-window-is-not-read`. That is the shape of
    a PDF with more than 384 bytes of object headers, and it was charged 125x over, which
    pushed the futility guard over its limit and disabled the tool-output budget entirely:
    nothing was trimmed and the request shipped ~600 KB past the limit for the provider to
    reject. Over-charging is the worse direction, because the guard reads the same number.

    The middle row is what keeps the read bounded. If the sniff decoded the whole payload
    the binary middle would answer and the row would flip, so it fails if the window grows
    without limit -- while the second row fails if the tail window is dropped, and the
    third if the head window shrinks below the eighth byte.
    """
    size = 100_000
    text = ("the quick brown fox jumps over it. " * 4000).encode()
    binary = bytes(range(256)) * 400
    raw = {
        "text-head-binary-middle-text-tail": text[:2000] + binary + text[:2000],
        "text-head-binary-tail": text[:2000] + binary,
        "nul-at-byte-eight": b"abcdefgh" + b"\x00" + text[:size],
    }[shape]
    charged = _charge(raw, "a.bin")
    assert (charged > len(raw) // 4) is text_rate, (
        f"the {shape} payload was charged {charged} chars against {len(raw)} bytes; "
        "the windows do not sample where the code says they do"
    )


def _encoded_charge(body: str, filename: str, params: str = "") -> tuple[int, int]:
    charged = estimate_serialized_chars([{
        "type": "input_file",
        "filename": filename,
        "file_data": f"data:application/octet-stream{params};base64,{body}",
    }])
    return charged, len("".join(body.split())) * 3 // 4


_ASCII_384 = ("the quick brown fox jumps over it. " * 400)[:120_000].encode("ascii")
_UNPARSEABLE = "AAAA" * 50 + "A=AA" + "AAAA" * 50


@pytest.mark.parametrize(
    ("payload_bytes", "residue"),
    [(204, 0), (205, 2), (206, 3)],
    ids=["aligned", "residue-2", "residue-3"],
)
def test_a_short_unpadded_payload_still_decodes(payload_bytes: int, residue: int) -> None:
    """A payload smaller than the sniff window has no slack to absorb a bad length.

    Every other alignment row here uses a payload far larger than
    `_PAYLOAD_HEAD_CHARS`, so the head slice is exactly 512 characters -- already a
    multiple of four -- and the truncation in `_decode_window` is a no-op. Under the
    window the slice is the whole body, so an unpadded stream arrives at a length that
    is not a multiple of four, `b64decode` raises, the head verdict is lost, and the
    charge falls back to the filename rule. Measured on 205 bytes of plain text named
    `report.dat`: 271 characters with the truncation, 67 without.

    The residue column is asserted, not assumed, so a row that stops exhibiting its
    shape fails loudly instead of passing vacuously. The aligned row is the control.
    """
    raw = ("the quick brown fox jumps over it. " * 20).encode("ascii")[:payload_bytes]
    body = base64.b64encode(raw).decode("ascii").rstrip("=")
    assert len(body) % 4 == residue, (
        f"fixture broken: {payload_bytes} bytes gives base64 residue "
        f"{len(body) % 4}, not {residue}"
    )
    charged = estimate_serialized_chars([{
        "type": "input_file",
        "filename": "report.dat",
        "file_data": "data:application/octet-stream;base64," + body,
    }])
    assert charged > payload_bytes // 2, (
        f"{payload_bytes} bytes of plain text with base64 residue {residue} was charged "
        f"{charged} characters -- the head window did not decode, so the sniff never ran"
    )


@pytest.mark.parametrize(
    ("body", "params", "filename", "text_rate", "pins"),
    [
        (_UNPARSEABLE, "", "a.txt", True, "a window that will not decode is undecided"),
        (_UNPARSEABLE, "", "a.bin", False, "a window that will not decode is undecided"),
        (
            base64.b64encode(_ASCII_384).decode("ascii"),
            ";name=" + "n" * 200,
            "a.bin",
            True,
            "the ';base64,' scan reads a prefix wide enough for real parameters",
        ),
        (
            base64.b64encode(_ASCII_384).decode("ascii"),
            ";name=" + "n" * 2_000,
            "a.bin",
            False,
            "a ';base64,' beyond the prefix is not a data URL, so only the name decides",
        ),
        (
            base64.encodebytes(_ASCII_384).decode("ascii"),
            "",
            "a.bin",
            True,
            "RFC 2045 line wrapping is stripped before the payload is decoded",
        ),
    ],
    ids=["unparseable-txt", "unparseable-bin", "long-parameter", "absurd-parameter", "line-wrapped"],
)
def test_the_sniff_survives_the_shapes_a_real_client_sends(
    body: str, params: str, filename: str, text_rate: bool, pins: str
) -> None:
    """Three encodings the pipe will meet in the wild, none of which any test covered.

    A data URL may carry parameters before `;base64,`, may arrive wrapped at 76
    characters per RFC 2045, and may simply be malformed. Each was silently charged the
    binary rate: line wrapping made the payload unparseable, a long parameter pushed the
    marker past the scan window, and an undecodable window was read as evidence of binary
    rather than as no evidence at all. The first two rows differ only in the filename,
    so a sniff that answers where it should abstain reddens one of them.
    """
    charged, size = _encoded_charge(body, filename, params)
    assert (charged > size // 2) is text_rate, (
        f"a payload named {filename} was charged {charged} chars against {size} bytes; "
        f"this row pins {pins}"
    )


@pytest.mark.parametrize("filename", ["", "a.txt", "a.bin"])
@pytest.mark.parametrize("kind", ["text", "binary"])
@pytest.mark.parametrize(
    ("media_type", "text_rate"),
    [
        ("application/pdf", False),
        ("image/png", False),
        ("text/plain", True),
        ("application/json", True),
    ],
)
def test_a_declared_type_outranks_the_bytes(
    media_type: str, kind: str, filename: str, text_rate: bool
) -> None:
    """A declaration that means something is evidence; the sniff is only for the ones
    that do not.

    The payload axis is crossed with the type axis on purpose. A previous version paired
    each type with exactly one payload, and the two axes were perfectly anti-correlated
    -- so discarding the declaration entirely and inverting the sniff passed all four
    rows. Crossing them means the charge must be the same whichever payload sits behind
    a given declaration, which is the actual claim.
    """
    size = 100_000
    charged = _charge(_payload(kind, size), filename, media_type)
    assert (charged > size // 4) is text_rate, (
        f"{media_type} carrying {kind} bytes was charged {charged} chars; a declaration "
        "that names a type is not overruled by the payload behind it"
    )


@pytest.mark.parametrize("excess_in_stubs", [False, True])
def test_an_irreducible_overage_still_trims_what_it_can(
    excess_in_stubs: bool,
) -> None:
    """An overage made of stubs is still trimmed as far as it goes, and reported.

    This test used to assert the opposite -- that the newest result is left whole when
    trimming cannot bring the request under. That contract was retired by an explicit
    decision after it was measured making request size discontinuous: on a 16,000-char
    limit, 15,125 characters of user text shipped 15,998 and 15,127 shipped 54,475.

    What makes the inversion safe rather than merely different is recorded below and in
    `a41ffcc`: the stub reaches only the outgoing request. The full result is persisted
    and replays intact on a later turn or a larger model, so trimming it here costs the
    model one turn's sight of it, not the text. `test_an_omitted_result_keeps_its_full_
    text_everywhere_but_upstream` is what holds that guarantee up.


    `_baseline_without_tool_outputs` blanks every `function_call_output`, including ones
    that are already omission stubs, so an overage made entirely of stubs is charged in
    full against the remaining budget. The budget reaches zero and the newest result is
    replaced by a stub roughly its own size, which cannot bring the request under a limit
    the stubs alone already exceed. Under the retired contract that was the argument for
    leaving it whole; under this one the pass trims what it can and says so, because
    shipping at full size a request it has already declared hopeless helps nobody.

    An earlier version of this docstring justified the guard by saying the stub is what
    gets persisted, so the original text was gone. That stopped being true one commit
    later: `a41ffcc` copies the outputs before budgeting them and removed the persistence
    skip, so the full text is stored and only the outgoing request carries the stub. The
    assertion below is still right; the reason it was written down no longer is.

    The two rows carry the same irreducible cost over the same limit and differ only in
    where it sits: fifteen existing stubs in one, the same volume of user text in the
    other. Both must stub the new result and report it, because the verdict is about the
    request and not about which item happens to carry the weight -- a guard reading the
    wrong number would spare the new result in exactly one of the two.
    """
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": 1_000}, "context_length": 1_000}}
    )
    stub = build_live_tool_omission_stub(result_chars=9_000, remaining_tokens=0)
    bulk = "x" * (15 * len(stub))
    if excess_in_stubs:
        existing = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x" * 1_200}]},
            *(
                {"type": "function_call_output", "call_id": f"old-{i}", "output": stub}
                for i in range(15)
            ),
        ]
    else:
        existing = [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "x" * 1_200 + bulk}]},
        ]

    assert estimate_serialized_chars(existing) >= 4_000, "regime broken: not over the limit"
    outputs = [{"type": "function_call_output", "call_id": "new", "output": "R" * 400}]

    outcome = apply_live_tool_output_budget(
        outputs, existing_input_items=existing, model_id="test/model"
    )

    assert outcome.futile is True, (
        "the pass did not report an overage it cannot trim its way out of"
    )
    assert outcome.omitted_call_ids == {"new"}, (
        "the pass reported it could not win and then trimmed nothing, so a request "
        "already over the limit went out at full size"
    )
    assert is_tool_omission_stub(outputs[0]["output"])


@pytest.mark.parametrize(
    ("media_type", "text_rate"),
    [
        ("text/plain", True),
        ("application/json", True),
        ("image/svg+xml", True),
        ("application/json;charset=utf-8", True),
        ("image/svg+xml; charset=utf-8", True),
        ("text/plain;charset=iso-8859-7", True),
        ("application/javascript", True),
        ("application/pdf", False),
        ("application/pdf;version=1.7", False),
        ("image/png", False),
        ("", False),
    ],
)
def test_a_document_is_charged_at_the_rate_its_own_media_type_implies(
    media_type: str, text_rate: bool
) -> None:
    """One flat rate for every `input_file` is wrong by two orders of magnitude.

    The module charges text at 4 chars per token and documents at 500 bytes per token,
    so the same bytes cost 125x less as an `input_file` than as `input_text`. A 200 KB
    plain-text attachment really costs about 50,000 tokens and was estimated at 409 --
    the request then ships far over the limit and the provider rejects it, which is the
    failure this module exists to prevent, arriving from the under-count direction.

    The media type is already in hand: a `data:` URL carries it in the prefix that
    `_payload_bytes` splits on. Rows either side of the boundary, so a single rate
    cannot satisfy them, and one row with no declared type at all -- which must take
    the conservative document rate rather than assuming text.

    The parametrised rows are the ones that matter: `charset` is the canonical companion
    of a textual type -- RFC 2397's own example is `data:text/plain;charset=iso-8859-7`
    -- and an earlier version compared the whole prefix, so every parametrised textual
    type silently fell through to the binary rate and was under-counted 122-fold.
    """
    payload_bytes = 200_000
    blob = f"data:{media_type};base64," + "A" * (payload_bytes * 4 // 3)
    charged = estimate_serialized_chars([{"type": "input_file", "file_data": blob}])

    if text_rate:
        assert charged > payload_bytes // 2, (
            f"a {media_type or 'declared-nothing'} document of {payload_bytes} bytes was "
            f"charged {charged} chars -- text costs about four characters per token, so "
            "this under-states the request by two orders of magnitude"
        )
    else:
        assert charged < payload_bytes // 50, (
            f"a {media_type or 'declared-nothing'} document was charged {charged} chars, "
            "the text rate, which over-states a binary payload the provider parses"
        )


@pytest.mark.parametrize(
    ("n_refs", "result_chars", "survives"),
    [(31, 100_000, True), (64, 5_000_000, False)],
    ids=["slope", "cliff"],
)
def test_references_neither_erode_nor_disable_the_tool_budget(
    n_refs: int, result_chars: int, survives: bool
) -> None:
    """Attachments the estimator cannot size must not decide what happens to tool output.

    Charging each reference a fixed guess produced two failures from one cause. Below the
    guard's threshold the guess ate the remaining budget, so a result that would have fit
    was stubbed -- and the stub is what gets persisted, so the original text is gone
    from every sink at once. At the threshold the same guess
    tripped the futility guard, and a five-megabyte result shipped whole.

    The two rows demand opposite verdicts from one mechanism, so no constant charge
    satisfies both: `slope` is red whenever references erode the budget, `cliff` is red
    whenever they disable it. Both were red before this rule, and both stay red under the
    alternative that re-bases only the guard.
    """
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": 128_000}, "context_length": 128_000}}
    )
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": f"9f2c1a7b-3e4d-4f5a-8b6c-{i:012d}"}
                for i in range(n_refs)
            ],
        }
    ]
    outputs = [{"type": "function_call_output", "call_id": "call-1", "output": "R" * result_chars}]

    omitted = apply_live_tool_output_budget(
        outputs, existing_input_items=items, model_id="test/model"
    ).omitted_call_ids

    import json as _json

    literal = len(_json.dumps(items))
    charged = estimate_serialized_chars(items)
    assert charged <= literal + 100, (
        f"{n_refs} references the pipe cannot size were charged {charged} chars against "
        f"the {literal} they literally occupy -- the omit decision below only reddens "
        "above ~13 KB per reference, so a wrong-but-smaller guess passes it"
    )
    if survives:
        assert omitted == set(), (
            f"{n_refs} references the pipe never read shrank the budget until a "
            f"{result_chars}-char result no longer fit"
        )
        assert outputs[0]["output"] == "R" * result_chars
    else:
        assert omitted == {"call-1"}, (
            f"{n_refs} references switched trimming off and a {result_chars}-char result "
            "shipped whole"
        )
        assert is_tool_omission_stub(outputs[0]["output"])


def test_a_data_url_inside_a_tool_result_is_still_counted_literally() -> None:
    """The flat rate is authorised by block type, never by what a string looks like.

    A tool that returns a screenshot as a data URL produces a `function_call_output`
    whose text the provider tokenises literally. Flat-rating it because it begins with
    `data:` would discount the one thing this module exists to trim, and would do it
    only when the payload happened to sit at offset zero.
    """
    payload = "data:image/png;base64," + "A" * 900_000

    charged = estimate_serialized_chars(
        [{"type": "function_call_output", "call_id": "c1", "output": payload}]
    )

    assert charged > 800_000, (
        "a data URL in a tool result was flat-rated; the estimate must follow the block "
        "type, not the shape of the string"
    )


# The block types the transformer and the responses adapter can put in `input`,
# with the keys each can populate with an inline payload. Two-way, like the warn-latch
# and valve-enrichment inventories: a new emitter key must be acknowledged here, and a
# table row with no emitter must be removed.
_EMITTED_PAYLOAD_KEYS: dict[str, tuple[str, ...]] = {
    "input_image": ("image_url",),
    "image_url": ("image_url",),
    "input_audio": ("input_audio",),
    "input_file": ("file_data", "file_id", "file_url"),
    "video_url": ("video_url",),
}


# Keys an emitter sets on a block that carry no payload of their own. Anything an
# emitter sets that is in neither this map nor the rate table is unaccounted for.
# Block types the emitters build whose whole content is text the provider tokenises
# literally. They carry no opaque payload, so counting them as written is correct --
# but they must be named, or a genuinely new opaque type is scanned and then skipped.
_RAW_COUNTED_BLOCK_TYPES: frozenset[str] = frozenset(
    {
        "file",
        "function",
        "function_call",
        "function_call_output",
        "input_text",
        "json_schema",
        "message",
        "object",
        "output_text",
        "text",
    }
)

_NON_PAYLOAD_BLOCK_KEYS: dict[str, set[str]] = {
    "input_file": {"type", "filename"},
    "input_image": {"type", "detail"},
    "input_audio": {"type", "format"},
    "file": {"type", "file"},
    "function": {"type", "name", "description", "parameters", "strict", "id", "function",
                 "cache_control"},
    "function_call": {"type", "call_id", "id", "name", "arguments", "status"},
    "function_call_output": {"type", "call_id", "output", "status", "id"},
    "input_text": {"type", "text"},
    "json_schema": {"type", "name", "schema", "strict", "description", "json_schema"},
    "message": {"type", "role", "content", "id", "status", "phase"},
    "object": {"type", "properties", "required", "additionalProperties"},
    "output_text": {"type", "text", "annotations"},
    "text": {"type", "text", "format", "cache_control"},
    "image_url": {"type", "detail"},
    "video_url": {"type"},
}


def _scan_emitter_block_keys() -> dict[str, set[str]]:
    """Read out of the emitters' own source which keys each block type can carry.

    The inventory above is hand-written, so comparing it to the rate table compares two
    literals in this file and cannot see the emitters at all -- adding a payload key to
    `_to_input_file` leaves both untouched. This walks the emitter modules instead,
    finds every dict literal that names a block type, and collects the keys later
    assigned onto that same variable. It is the warn-latch inventory's shape: one side
    derived from the package, one side declared, and drift between them is red.

    The root is anchored on this test file rather than on the package's `__file__`, as
    `tests/test_warn_latch_isolation.py` does. Under the flat bundle the package resolves
    to a single module at the repository root, so a package-relative path finds nothing
    and the scan raises instead of scanning.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe"
    found: dict[str, set[str]] = {}

    def block_type_of(node: ast.AST) -> str | None:
        if not isinstance(node, ast.Dict):
            return None
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "type"
                and isinstance(value, ast.Constant)
                and isinstance(value.value, str)
            ):
                return value.value
        return None

    def own_scope(function: ast.AST) -> list[ast.AST]:
        """Nodes belonging to this function, excluding any nested function's body.

        The emitters are closures inside `transform_messages_to_input`, and all three
        name their block `result`. Walking the outer function would merge them and
        attribute `input_image`'s keys to `input_file`.
        """
        nodes: list[ast.AST] = []
        stack = list(ast.iter_child_nodes(function))
        while stack:
            node = stack.pop()
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda):
                continue
            nodes.append(node)
            stack.extend(ast.iter_child_nodes(node))
        return nodes

    for relative in ("requests/transformer.py", "api/transforms.py"):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            scope = own_scope(function)
            holders: dict[str, str] = {}
            for node in scope:
                if not isinstance(node, ast.Dict):
                    continue
                block = block_type_of(node)
                if block is None:
                    continue
                found.setdefault(block, set()).update(
                    k.value
                    for k in node.keys
                    if isinstance(k, ast.Constant) and isinstance(k.value, str)
                )
            for node in scope:
                value = getattr(node, "value", None)
                if not isinstance(value, ast.Dict):
                    continue
                block = block_type_of(value)
                if block is None:
                    continue
                names = (
                    [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
                )
                for target in names:
                    if isinstance(target, ast.Name):
                        holders[target.id] = block
            for node in scope:
                if not isinstance(node, ast.Assign):
                    continue
                for target in node.targets:
                    if (
                        isinstance(target, ast.Subscript)
                        and isinstance(target.value, ast.Name)
                        and target.value.id in holders
                        and isinstance(target.slice, ast.Constant)
                        and isinstance(target.slice.value, str)
                    ):
                        found[holders[target.value.id]].add(target.slice.value)
    return found


def test_every_emitted_payload_key_has_a_rate() -> None:
    """The index can only be trusted if a miss cannot ship silently.

    Keying the rate table on block type is right -- the type is what says where the
    payload lives, and for audio, whose payload is bare base64 with no `data:` prefix,
    it is the only thing that can. The cost is that the table can be incomplete, and an
    incomplete index fails in both directions: a miss counts a blob at full base64
    length and trips the futility guard, a hit under-counts and ships a request the
    provider rejects. `file_url` was missed exactly this way.

    So the completeness is asserted rather than hoped for. This is the pattern the repo
    already uses for warn latches and for valve enrichment: an explicit inventory, two
    directions, so neither side can drift without a red test.
    """
    from open_webui_openrouter_pipe.core.context_budget import _OPAQUE_BLOCK_PAYLOADS

    tabled = {block: set(keys) for block, (keys, _rate) in _OPAQUE_BLOCK_PAYLOADS.items()}
    emitted = {block: set(keys) for block, keys in _EMITTED_PAYLOAD_KEYS.items()}

    assert emitted.keys() == tabled.keys(), (
        f"untabled emitters: {sorted(emitted.keys() - tabled.keys())}; "
        f"orphaned rows: {sorted(tabled.keys() - emitted.keys())}"
    )
    for block in sorted(emitted):
        assert emitted[block] == tabled[block], (
            f"{block}: emitter can set {sorted(emitted[block])} but the table charges "
            f"{sorted(tabled[block])} -- a key in neither column is counted raw"
        )

    scanned = _scan_emitter_block_keys()
    unacknowledged = scanned.keys() - tabled.keys() - _RAW_COUNTED_BLOCK_TYPES
    assert not unacknowledged, (
        f"the emitters build {sorted(unacknowledged)}, which is neither rated nor "
        "acknowledged as safe to count literally -- a new opaque block type ships "
        "counted at full base64 length by the surrounding walk"
    )
    assert not (tabled.keys() - scanned.keys()), (
        f"the scan located no emitter for {sorted(tabled.keys() - scanned.keys())}, so "
        "those rows are checked against nothing but a hand-written literal"
    )
    for block, keys in sorted(scanned.items()):
        unaccounted = (
            keys - set(tabled.get(block, ())) - _NON_PAYLOAD_BLOCK_KEYS.get(block, set())
        )
        assert not unaccounted, (
            f"{block}: the emitter sets {sorted(unaccounted)}, which is neither charged "
            "by the rate table nor listed as carrying no payload -- a new key ships "
            "counted at full base64 length by the surrounding walk"
        )


@pytest.mark.parametrize(
    ("results", "tokens", "expect_drop"),
    [(8, 800, 4), (14, 1_400, 6)],
    ids=["eight-results", "fourteen-results"],
)
def test_each_stub_quotes_the_budget_the_stubs_before_it_actually_spent(
    results: int, tokens: int, expect_drop: int
) -> None:
    """The placeholder tells the model how much room is left; that figure must be true.

    A result's floor reserves a placeholder built with `remaining_tokens=0`, but the
    placeholder actually written quotes the real figure, so it is a few characters longer
    than was reserved -- one per extra digit. `remaining_chars` is debited that difference
    after each stub is written. Drop the debit and every placeholder in the turn quotes
    the same starting figure, so the model is told there is room that earlier placeholders
    have already consumed.

    Measured across the whole package suite, removing the debit reddened nothing: the
    effect is confined to the digits inside the placeholder text, which no assertion read.
    The two rows expect different drops, so a constant correction satisfies neither, and
    the figures must fall monotonically because budget is only ever spent.
    """
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": tokens}, "context_length": tokens}}
    )
    items = [
        {"type": "function_call_output", "call_id": f"c{i}", "output": "R" * 3_000}
        for i in range(results)
    ]

    apply_replay_tool_output_budget(_with_calls(items), model_id="test/model")

    quoted = [
        int(match.group(1).replace(",", ""))
        for item in reversed(items)
        if (match := re.search(r"Remaining prompt budget: ~([\d,]+) tokens", item["output"]))
    ]
    assert len(quoted) == results, (
        f"regime broken: {len(quoted)} of {results} results carry a placeholder, so this "
        "fixture is not exercising a turn where the budget runs out"
    )
    assert quoted == sorted(quoted, reverse=True), (
        f"the quoted remaining budget went up between placeholders: {quoted}"
    )
    assert quoted[0] - quoted[-1] == expect_drop, (
        f"the first placeholder quoted {quoted[0]} tokens and the last {quoted[-1]}, a "
        f"drop of {quoted[0] - quoted[-1]} rather than {expect_drop}; each placeholder "
        "costs a little more than its reservation and the difference must be spent"
    )


def test_a_request_with_no_tool_results_is_not_reported_as_hopeless() -> None:
    """`BudgetOutcome`'s `futile` default is load-bearing on a real path.

    `_apply_tool_output_budget` returns `BudgetOutcome(frozenset())` before it computes
    anything when there is nothing to budget, and `body.input` really can be empty. That
    outcome is handed straight to `_warn_if_futile`, so flipping the dataclass default to
    True would tell every user with an empty request that their conversation cannot fit --
    and the full package suite stayed green when it was flipped.
    """
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": 100}, "context_length": 100}}
    )

    empty = apply_replay_tool_output_budget([], model_id="test/model")
    no_outputs = apply_live_tool_output_budget(
        [], existing_input_items=[], model_id="test/model"
    )

    assert empty.futile is False, "an empty request was reported as impossible to fit"
    assert no_outputs.futile is False, "a turn with no tool results was reported hopeless"
    assert empty.omitted_call_ids == frozenset()
    assert no_outputs.omitted_call_ids == frozenset()


@pytest.mark.parametrize("result_chars", [60_000, 200_000])
def test_a_tool_result_quoting_the_stub_marker_is_still_trimmed(result_chars: int) -> None:
    """A result is already-reduced only if this pipe reduced it. Text cannot establish that.

    `_output_floor_chars` reserves a result's FULL length when it recognises an omission
    stub, so that replaying one is idempotent. Recognition used to be a prefix match on
    the output text -- which is tool output, i.e. attacker-reachable. Any tool, MCP server
    or fetched page whose text begins "[Tool result omitted due to context budget." bought
    immunity from trimming AND drove the irreducible floor past the limit, so every
    innocent result in the turn was stubbed instead and the user was told the conversation
    was hopeless. Measured at 4,000 tokens with three results: the poisoned one kept all
    60,044 characters, the two innocent ones fell to 131 each, and the request shipped
    60,744 characters against a 16,000 limit while reporting futile.

    Recognition now requires the entire text to be a well-formed stub, so a marker with a
    payload appended is not one. The two rows are an order of magnitude apart, so a
    constant floor satisfies neither.
    """
    ModelFamily.set_dynamic_specs(
        {"test.model": {"full_model": {"max_prompt_tokens": 4_000}, "context_length": 4_000}}
    )
    poisoned = {
        "type": "function_call_output",
        "call_id": "poison",
        "output": _LIVE_OMISSION_PREFIX + " " + "Z" * result_chars,
    }
    innocent = [
        {"type": "function_call_output", "call_id": f"ok{i}", "output": "b" * 4_000}
        for i in range(2)
    ]
    items = _with_calls([poisoned, *innocent])

    outcome = apply_replay_tool_output_budget(items, model_id="test/model")

    assert outcome.omitted_call_ids == {"poison"}, (
        f"the pass omitted {sorted(outcome.omitted_call_ids)}; a {result_chars} char "
        "result that merely quotes the placeholder must be trimmed like any other, and "
        "the results that did not quote it must survive"
    )
    assert all(not is_tool_omission_stub(item["output"]) for item in innocent), (
        "an innocent tool result was evicted to make room for one that claimed to be a stub"
    )
    assert outcome.futile is False, (
        f"a request whose only oversize is one trimmable result reported futile "
        f"({outcome.irreducible_chars} against {outcome.limit_chars})"
    )


@pytest.mark.parametrize("remaining", [0, 1_234])
@pytest.mark.parametrize("result_chars", [1, 9_000, 4_000_000])
def test_both_stub_builders_produce_text_the_recogniser_accepts(
    result_chars: int, remaining: int
) -> None:
    """The recogniser is a full-shape match, so it must track both builders exactly.

    If a builder's wording changes and the pattern does not, a replayed stub stops being
    recognised, its floor drops from its own length to a placeholder's, and it is stubbed
    again every turn -- the drift `test_replay_budget_is_idempotent_for_existing_stub`
    exists to prevent. Six rows across both builders, including the thousands separator
    that only appears above 999.
    """
    for build in (build_live_tool_omission_stub, build_replayed_tool_omission_stub):
        stub = build(result_chars=result_chars, remaining_tokens=remaining)
        assert is_tool_omission_stub(stub), (
            f"{build.__name__} produced text the recogniser rejects: {stub!r}"
        )


@pytest.mark.parametrize("extra_kb", [1, 16])
def test_a_tool_call_is_budgeted_for_what_ships_not_for_what_the_provider_attached(
    extra_kb: int,
) -> None:
    """The baseline must reduce every item type the sanitiser reduces, not just outputs.

    `_strip_tool_item_extras` shapes a `function_call` down to
    {type, call_id, name, arguments} before dispatch, exactly as it shapes an output. The
    budget's baseline mirrored the output half and passed calls through whole -- so a
    replayed call was charged for keys that never travel. This is the real path, not a
    hypothetical: `normalize_persisted_item` puts `id` and `status` on every persisted
    call and copies whatever else the provider sent, and the streaming loop extends
    `body.input` with them. Measured before the fix, a call carrying a 4 KB provider key
    charged 4,210 characters against the 146 it actually ships.

    The number matters because `irreducible_chars` is what the futility verdict reads: an
    inflated floor tells the user a conversation cannot fit when it can. Two sizes so a
    constant difference cannot satisfy both rows.
    """
    bare = [
        {"type": "function_call", "call_id": "c1", "name": "lookup", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "r" * 100},
    ]
    laden = [
        dict(bare[0])
        | {
            "id": "fc_abc123",
            "status": "completed",
            "provider_extra": "P" * (extra_kb * 1024),
        },
        dict(bare[1]),
    ]

    bare_chars = estimate_serialized_chars(_baseline_without_tool_outputs(bare))
    laden_chars = estimate_serialized_chars(_baseline_without_tool_outputs(laden))

    assert laden_chars == bare_chars, (
        f"a call carrying {extra_kb} KB of provider-only keys added "
        f"{laden_chars - bare_chars} characters to the irreducible floor; the sanitiser "
        "ships the same four keys either way, and this number decides whether the user "
        "is told their conversation cannot fit"
    )


@pytest.mark.parametrize(
    ("declared_cap", "expect_limit_tokens"),
    [(8_000, 123_072), (40_000, 91_072), (None, 131_072)],
    ids=["small-cap", "large-cap", "no-cap"],
)
def test_the_reply_allowance_on_the_body_reaches_the_budget(
    declared_cap, expect_limit_tokens: int
) -> None:
    """The budget must spend the window this request actually leaves the model.

    The prompt limit used to be `context_length - max_completion_tokens`, the provider's
    largest POSSIBLE answer rather than the one this request asks for. Across the
    2026-08-19 catalog that put 319 of 447 models below 99% of their real window and 17
    below half of it, worst 2.3% -- so an ordinary conversation was declared unable to fit
    and every tool result in it was replaced by a placeholder.

    Asserted on the limit the sanitiser's own pass computed, not on the call site. A
    version that spells `reserved_output_tokens=` at the call site while passing `None`
    reddens the first two rows; the third row keeps a fix that always subtracts something
    from passing.
    """
    from open_webui_openrouter_pipe.api.transforms import ResponsesBody
    from open_webui_openrouter_pipe.requests.sanitizer import _sanitize_request_input

    ModelFamily.set_dynamic_specs(
        {"test.model": {"context_length": 131_072, "full_model": {"context_length": 131_072}}}
    )
    payload: dict[str, Any] = {
        "model": "test/model",
        "input": _with_calls(
            [{"type": "function_call_output", "call_id": "c1", "output": "r" * 100}]
        ),
    }
    if declared_cap is not None:
        payload["max_output_tokens"] = declared_cap
    body = ResponsesBody.model_validate(payload)

    class _Pipe:
        logger = logging.getLogger("test")

    outcome = _sanitize_request_input(cast(Any, _Pipe()), body)

    assert outcome is not None
    assert outcome.limit_chars == expect_limit_tokens * _CHARS_PER_TOKEN_HEURISTIC, (
        f"a request declaring {declared_cap} output tokens was budgeted against "
        f"{outcome.limit_chars // _CHARS_PER_TOKEN_HEURISTIC} prompt tokens, not "
        f"{expect_limit_tokens}; the window the model is left is what the budget spends"
    )


@pytest.mark.parametrize(
    ("output", "expect_kind"),
    [
        ({"rows": [{"name": "Ann's file", "ok": True}]}, dict),
        ([1, 2, {"k": "v"}], list),
        ("already a string", str),
    ],
    ids=["dict", "list", "string"],
)
def test_the_budget_reads_a_result_without_rewriting_it(output, expect_kind) -> None:
    """The budget measures `output`; only the sanitiser normalises it.

    The budget used to write `str(raw)` back onto any non-string output. That is Python
    `repr`, not JSON: a dict came out as `{'rows': [...], 'ok': True}` -- single quotes,
    `True` -- and because the value was then a `str`, the sanitiser's
    `json.dumps(..., ensure_ascii=False)` left it alone. The provider received Python.

    The write was also unguarded by the ownership predicate the stub write carries, so it
    reached items the pass was explicitly confined not to touch.

    The string row is what stops a pass that simply stopped measuring from passing: it
    must still be measured and still be eligible for stubbing.
    """
    _spec(100_000)
    item = {"type": "function_call_output", "call_id": "c1", "output": output}
    items = _with_calls([item])

    apply_replay_tool_output_budget(items, model_id="test/model")

    assert isinstance(item["output"], expect_kind), (
        f"the budget rewrote a {expect_kind.__name__} result as "
        f"{type(item['output']).__name__}; normalising `output` is the sanitiser's job "
        "and it produces JSON, not a Python repr"
    )
    assert item["output"] == output, "the budget altered a result it only had to measure"


@pytest.mark.parametrize("limit_tokens", [131_072, 32_768])
def test_a_request_of_escaped_results_ships_inside_its_own_limit(limit_tokens) -> None:
    """The budget must meter the characters the wire carries, not the raw string.

    `result_chars` was `len(output_text)` while the request carries the JSON-escaped
    string: every quote and newline in a tool result becomes two characters on the way
    out. Measured on pretty JSON the inflation is ~1.2x, and a request the budget believed
    was exactly at its limit dispatched 629,940 characters against 524,288 -- 20% over,
    with trimming already exhausted.

    Two limits an order of magnitude apart, so a constant cannot satisfy both.
    """
    _spec(limit_tokens)
    payload = json.dumps(
        {"rows": [{"id": i, "name": f"item {i}", "note": 'a "quoted" value'} for i in range(40)]},
        indent=2,
    )
    items: list[dict] = []
    for i in range(200):
        items.append({"type": "function_call", "call_id": f"c{i}", "name": "lookup", "arguments": "{}"})
        items.append({"type": "function_call_output", "call_id": f"c{i}", "output": payload})

    apply_replay_tool_output_budget(items, model_id="test/model")

    limit_chars = limit_tokens * _CHARS_PER_TOKEN_HEURISTIC
    shipped = len(json.dumps(items, ensure_ascii=False))
    assert shipped <= limit_chars, (
        f"the budget finished and the request still carries {shipped} chars against its "
        f"own {limit_chars}-char limit; it metered the raw strings, not the escaped ones"
    )


def test_the_irreducible_floor_covers_what_the_stubbed_request_really_costs() -> None:
    """`irreducible_chars` is the request's cost with every result already a placeholder.

    It is the number the futility verdict is decided on and the number quoted to the user,
    so it has to be measured on the same scale as the request. Summing raw string lengths
    under-reports it whenever the results carry quotes -- measured on 120 short, heavily
    escaped results: 26,780 reported against a 32,180 real floor, so a request the pipe
    calls reducible is already at its limit.

    Asserted against the serialized size of the fully-stubbed request rather than against
    a constant, so the two cannot drift apart.
    """
    _spec(900)
    payload = '{"a":"' + '\\"q\\" ' * 12 + '"}'
    items: list[dict] = []
    for i in range(120):
        items.append({"type": "function_call", "call_id": f"c{i}", "name": "t", "arguments": "{}"})
        items.append({"type": "function_call_output", "call_id": f"c{i}", "output": payload})

    outcome = apply_replay_tool_output_budget(items, model_id="test/model")

    assert outcome.futile, "this fixture must exercise the floor, not the spending loop"
    shipped = len(json.dumps(items, ensure_ascii=False))
    assert outcome.irreducible_chars >= shipped, (
        f"the pass reported a {outcome.irreducible_chars}-char floor for a request that "
        f"actually costs {shipped}; the floor is summed from raw lengths while the wire "
        "carries the escaped strings"
    )


@pytest.mark.parametrize(
    ("tokens", "expect_kept"),
    [(2000, 5), (2500, 9), (3000, 13)],
    ids=["room-for-five", "room-for-nine", "room-for-thirteen"],
)
def test_the_budget_keeps_the_newest_results_and_drops_the_oldest(
    tokens: int, expect_kept: int
) -> None:
    """When the allowance cannot cover every result, the survivors are the recent ones.

    The loop spent front-to-back, so the OLDEST results survived and the newest were
    replaced -- including the one the model had just asked for. Measured on the real
    dispatch path over a 40-turn conversation, from turn 15 onward the pipe discarded
    every freshly-executed result while 15-turn-stale ones rode along in full. The model
    asks again, the pipe drops it again, and nothing is emitted: a livelock the operator
    cannot see, because the transcript shows the result arriving.

    It also undid the retention pruner, which shortens the OLDEST turns and leaves the
    newest whole -- two trimmers with opposite policies, the second destroying what the
    first had just protected.

    The property is the surviving SUFFIX, not the last element: greedy spending still
    stubs a newest result too large to fit, which is why the two rows expect different
    counts and why "the newest is always delivered" is the wrong assertion.
    """
    _spec(tokens)
    items = [
        {"type": "function_call_output", "call_id": f"c{i:02d}", "output": "R" * 600}
        for i in range(20)
    ]

    apply_replay_tool_output_budget(_with_calls(items), model_id="test/model")

    kept = [i["call_id"] for i in items if not is_tool_omission_stub(i["output"])]
    assert len(kept) == expect_kept, (
        f"regime broken: {len(kept)} of 20 results survived, not {expect_kept}; this "
        "fixture must exercise a budget that runs out partway"
    )
    assert kept == [f"c{i:02d}" for i in range(20 - expect_kept, 20)], (
        f"the budget kept {kept}; it must keep the most recent {expect_kept} results, "
        "not the oldest -- the newest are the ones the model is still working with"
    )


@pytest.mark.parametrize("filename", ["report.pdf", "report.txt", ""])
@pytest.mark.parametrize("url_chars", [200, 20_000])
def test_a_reference_the_estimator_cannot_size_costs_its_own_characters(
    url_chars: int, filename: str
) -> None:
    """A string that only NAMES content is charged the characters it occupies.

    `_payload_bytes` could not tell a locator from a payload, so it measured the URL as
    though the URL were base64 -- `len(raw) * 3 // 4` -- and `_budget_shape` then BLANKED
    the key and charged a rate on that invented number. The characters that actually
    travel were erased. Measured on a 20,000-character external `file_url`: 175 charged
    against 20,057 on the wire when the block was named `.pdf`, and 15,059 when the same
    block was named `.txt`. A client-supplied filename swung the charge by 86x on bytes
    nobody had measured.

    Both filenames and the empty one are rows here, so the name provably has no vote, and
    the two URL lengths are two orders of magnitude apart so no constant satisfies them.
    """
    block: dict = {"type": "input_file", "file_url": "https://example.invalid/" + "a" * url_chars}
    if filename:
        block["filename"] = filename

    charged = estimate_serialized_chars([block])

    wire = len(json.dumps(block))
    assert charged >= wire, (
        f"a block whose payload the estimator cannot size was charged {charged} chars "
        f"against the {wire} it puts on the wire; an unsizable reference must cost at "
        "least what it occupies"
    )


@pytest.mark.parametrize(
    ("block_type", "payload_key"),
    [("input_image", "image_url"), ("image_url", "image_url")],
)
def test_a_reference_still_pays_a_rate_that_never_needed_its_size(
    block_type: str, payload_key: str
) -> None:
    """An image costs what an image costs, whether or not its bytes can be measured.

    The tempting fix for the defect above is to return `None` for anything that is not a
    data URL. That drops the block out of the rate table entirely, and a remote image goes
    from 6,842 charged characters to its 71 literal ones -- a 96x under-charge on a
    payload the provider really does fetch and tokenise. The flat image rate never needed
    a size, so a locator must not disable it.
    """
    block = {"type": block_type, payload_key: "https://example.invalid/pic.png"}

    charged = estimate_serialized_chars([block])

    assert charged > 6_000, (
        f"a remote image was charged {charged} chars; the flat image rate does not depend "
        "on measuring the bytes and must survive a reference it cannot size"
    )


@pytest.mark.parametrize(
    ("usage", "expect"),
    [
        ({"input_tokens": 10_811}, 3.7),
        ({"input_tokens": "10811"}, 3.7),
        ({"input_tokens": 10_811.0}, 3.7),
        ({"input_tokens": 10_811, "output_tokens": 60, "total_tokens": 10_871}, 3.7),
        ({}, None),
        ({"input_tokens": None}, None),
        ({"input_tokens": 0}, None),
        ({"input_tokens": -5}, None),
        ({"input_tokens": True}, None),
        ({"input_tokens": "many"}, None),
        ({"input_tokens": float("inf")}, None),
        ({"input_tokens": float("nan")}, None),
        ({"input_tokens": float("-inf")}, None),
        ({"input_tokens": 10 ** 400}, None),
        ({"input_tokens": 277}, None),
        ({"input_tokens": 10_811, "output_tokens": 60, "total_tokens": 99_999}, None),
        ({"input_tokens": 200_000}, None),
        ({"input_tokens": 160_000}, 0.25),
    ],
    ids=[
        "healthy", "numeric-string", "float", "counters-agree", "empty", "none", "zero",
        "negative", "bool", "not-a-number", "infinity", "nan", "negative-infinity",
        "400-digit-integer", "below-the-sample-gate", "counters-contradict",
        "below-the-physical-floor", "exactly-at-the-floor",
    ],
)
def test_only_a_sample_the_guards_stand_behind_becomes_a_ratio(usage, expect) -> None:
    """A ratio is only as good as the counter it divides by.

    `BASE_URL` is configurable, so usage arrives from arbitrary OpenAI-compatible
    gateways. `json.loads` turns a bare `Infinity` literal into `inf`, and the pipe's own
    `_coerce_positive_int` raises `OverflowError` on it -- so the coercion here cannot be
    that one. A 400-digit integer is legal JSON and `float()` raises on it too.

    The sample gate is 1,000 tokens because the estimator meters JSON scaffolding the
    provider's chat template does not tokenise, and that overhead is fixed: the error it
    causes decays as 1/tokens and falls below the constant's own 9.7% error at ~988.

    The floor is 0.25 and it is a physical bound, not a preference: a token spans at
    least one UTF-8 byte and a character is at most four of them, so no real tokenizer
    can report fewer than a quarter of a character per token.

    Sixteen rows, four of which must yield a number and twelve of which must not, so
    neither "always measure" nor "never measure" satisfies the set.
    """
    ratio = measure_chars_per_token(metered_chars=40_000, usage=usage)

    if expect is None:
        assert ratio is None, (
            f"usage {usage!r} produced a ratio of {ratio}; a counter the guards cannot "
            "stand behind must fall back to the constant, not steer the budget"
        )
    else:
        assert ratio is not None and abs(ratio - expect) < 0.01, (
            f"usage {usage!r} produced {ratio}, not ~{expect}"
        )


@pytest.mark.parametrize(
    ("ratio", "expect_limit"),
    [(None, 16_000), (4.5, 16_000), (4.0, 16_000), (3.0, 12_000), (2.0, 8_000), (0.25, 1_000)],
    ids=["no-sample", "above-the-cap", "at-the-cap", "tighter", "tighter-still", "floor"],
)
def test_a_measured_ratio_may_tighten_the_budget_and_may_never_loosen_it(
    ratio, expect_limit: int
) -> None:
    """The whole safety property in one row set.

    A measured ratio is allowed to say the conversation costs MORE tokens than four
    characters each, which shrinks the character budget. It is never allowed to say it
    costs fewer, because a rising limit admits content that was previously trimmed --
    the dispatched request then grows as the conversation grows, which is the
    discontinuity this repo forbids and the reason an earlier content-dependent limit
    was rejected.

    The three rows at or above the cap must all produce the identical limit a deployment
    with no usage at all gets, so a gateway reporting a loose ratio changes nothing.
    """
    _spec(4_000)
    items = _with_calls(
        [{"type": "function_call_output", "call_id": f"c{i}", "output": "R" * 900} for i in range(8)]
    )

    outcome = apply_replay_tool_output_budget(
        items, model_id="test/model", chars_per_token=ratio
    )

    assert outcome.limit_chars == expect_limit, (
        f"a ratio of {ratio} produced a {outcome.limit_chars}-char limit, not "
        f"{expect_limit}; at or above {_CHARS_PER_TOKEN_HEURISTIC} it must equal the "
        "limit a deployment with no usage gets"
    )
    assert outcome.limit_chars <= 4_000 * _CHARS_PER_TOKEN_HEURISTIC, (
        "a measured ratio loosened the budget above the constant's limit"
    )


def test_the_ratchet_keeps_the_tightest_sample_of_the_turn() -> None:
    """Within a turn the limit must never rise, so the store keeps the minimum.

    Feeding the latest sample instead lets a turn that measured 2.5 on one call go back
    to 3.9 on the next, raising the limit mid-turn and re-admitting results it had
    already dropped. A running mean has the same defect more slowly.
    """
    store: dict[str, float] = {}
    for sample in (3.9, 2.5, 3.2, 3.8):
        record_chars_per_token(store, "m", sample)

    assert store["m"] == 2.5, (
        f"the store holds {store['m']} after seeing 3.9, 2.5, 3.2, 3.8; it must keep the "
        "tightest, or the limit rises again mid-turn"
    )
    assert effective_chars_per_token(store, "m") == 2.5
    assert effective_chars_per_token(store, "other-model") is None, (
        "a ratio measured for one model was applied to another"
    )
    assert effective_chars_per_token({}, "m") is None
