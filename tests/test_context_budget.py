"""Tests for adaptive context budgeting helpers."""

from __future__ import annotations

import pytest

from open_webui_openrouter_pipe.core.context_budget import (
    _CHARS_PER_TOKEN_HEURISTIC,
    apply_live_tool_output_budget,
    apply_replay_tool_output_budget,
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
    items = [
        {"type": "reasoning", "id": "rs-1", "signature": "s" * 4000},
        {"type": "function_call_output", "call_id": "call-1", "output": "y" * 4000},
    ]

    assert len(
        build_replayed_tool_omission_stub(result_chars=4000, remaining_tokens=0)
    ) < 4000, "regime broken: the stub-length short-circuit would give this answer too"
    omitted = apply_replay_tool_output_budget(items, model_id="test/model")

    assert omitted == set()
    assert items[1]["output"] == "y" * 4000


def test_live_budget_leaves_results_alone_when_stubbing_cannot_help() -> None:
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
    omitted = apply_live_tool_output_budget(
        outputs,
        existing_input_items=oversized_existing,
        model_id="test/model",
    )

    assert omitted == set()
    assert outputs[0]["output"] == "y" * 4000


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

    assert items[0]["output"] == original_stub, (
        "an existing stub was stubbed again, so the replayed text drifts every turn"
    )
    assert first == second == set(), (
        "a pass that changed nothing reported an omission. The caller uses this set only "
        "to decide whether the request was altered and to count it in a log line, so "
        "reporting a stub it did not touch overstates both."
    )


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
    )

    assert omitted == {"big"}, (
        f"one {block['type']} attachment disabled tool-output trimming for the whole turn"
    )


@pytest.mark.parametrize(
    ("prior_chars", "expect_stub"), [(100, False), (200, False), (600, True)]
)
def test_results_already_in_the_request_consume_the_remaining_budget(
    prior_chars, expect_stub
) -> None:
    """A result accepted on an earlier tool loop still occupies the request.

    The baseline blanks every tool output so the futility guard does not treat a
    shrinkable result as hopeless -- but the space those results occupy is real, and
    charging nothing for it makes the live pass believe there is more room than there
    is. It then approves a result the replay pass immediately stubs, and because the
    live pass's return value is what gates citation harvesting and artifact
    persistence, the pipe cites and stores a result the model never actually read.

    Parametrised across the boundary rather than in one direction: charging the prior
    twice is as wrong as charging it not at all, and a single row that expects a stub
    is satisfied by both the correct arithmetic and an over-conservative one.
    """
    _spec(200)
    existing = [
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "function_call_output", "call_id": "earlier", "output": "y" * prior_chars},
    ]
    outputs = [{"type": "function_call_output", "call_id": "new", "output": "z" * 400}]

    omitted = apply_live_tool_output_budget(
        outputs, existing_input_items=existing, model_id="test/model"
    )

    assert (omitted == {"new"}) is expect_stub, (
        "the space an earlier result occupies must be charged exactly once: charging "
        "nothing approves a result that does not fit, charging it twice rejects one "
        "that does"
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


@pytest.mark.parametrize(
    ("media_type", "filename", "text_rate"),
    [
        ("application/octet-stream", "notes.md", True),
        ("", "data.json", True),
        ("application/octet-stream", "report.pdf", False),
        ("application/octet-stream", "", False),
        ("application/pdf", "report.txt", False),
        ("image/png", "x.txt", False),
    ],
    ids=["generic-md", "absent-json", "generic-pdf", "generic-noname", "pdf-wins", "png-wins"],
)
def test_a_filename_decides_only_when_the_declaration_names_nothing(
    media_type: str, filename: str, text_rate: bool
) -> None:
    """`application/octet-stream` is what a client sends when it has no type to give.

    The pipe writes that spelling itself, and so does Open WebUI, so it is the common
    case rather than an edge one -- and it took the binary rate, 125x under, which
    dispatches a request the provider then rejects after the live pass has already told
    the pipe to cite and persist a result the model never read.

    The last two rows are the guard: an explicit `application/pdf` on a file named
    `.txt` must stay binary. The filename is a fallback for the untyped case, never an
    override of a declaration that means something.
    """
    payload_bytes = 200_000
    block: dict = {
        "type": "input_file",
        "file_data": f"data:{media_type};base64," + "A" * (payload_bytes * 4 // 3),
    }
    if filename:
        block["filename"] = filename
    charged = estimate_serialized_chars([block])

    if text_rate:
        assert charged > payload_bytes // 2, (
            f"{media_type or 'an undeclared type'} named {filename!r} was charged "
            f"{charged} chars; text costs about four characters per token"
        )
    else:
        assert charged < payload_bytes // 50, (
            f"{media_type or 'an undeclared type'} named {filename!r} was charged "
            f"{charged} chars, the text rate, for a payload that is not text"
        )


@pytest.mark.parametrize("excess_in_stubs", [False, True])
def test_an_irreducible_overage_is_not_paid_for_with_the_newest_result(
    excess_in_stubs: bool,
) -> None:
    """A stub is the one thing trimming cannot shrink further.

    `_baseline_without_tool_outputs` blanks every `function_call_output`, including ones
    that are already omission stubs, so an overage made entirely of stubs is invisible to
    the futility guard and is charged in full against the remaining budget instead. The
    budget reaches zero, the newest result is replaced by a stub roughly its own size --
    and the stub is what is persisted in its place, so the original text is gone. The
    request still fails.

    The two rows carry the same irreducible cost over the same limit and differ only in
    where it sits. Both must leave the new result alone; if only the fixed-text row does,
    the guard is reading the wrong number.
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

    omitted = apply_live_tool_output_budget(
        outputs, existing_input_items=existing, model_id="test/model"
    )

    assert omitted == set(), (
        "a request whose overage is entirely made of stubs -- the one content trimming "
        "cannot shrink -- paid for it by destroying the newest result, which is dropped "
        "from persistence with its call and survives nowhere"
    )
    assert outputs[0]["output"] == "R" * 400


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
    )

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
