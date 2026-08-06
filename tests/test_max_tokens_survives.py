"""A `max_tokens` the user sets must reach OpenRouter, whatever the valve says.

Two different things had been conflated:

* `max_tokens` is the USER's parameter, set per request in Open WebUI's advanced params:
  "cap this reply at N tokens". `api/transforms.py` renames it to `max_output_tokens`
  because that is what the /responses API calls the same field.
* `USE_MODEL_MAX_OUTPUT_TOKENS` is an ADMIN valve about a DEFAULT: when the request does
  not carry a limit, should the pipe look up the provider's advertised maximum and send
  that? The number it injects comes from `ModelFamily.max_completion_tokens` -- the
  provider's, not the user's.

The valve defaulted to False and its `else` branch nulled the field outright, so every
`max_tokens` a user set was discarded while `temperature` and `top_p` on the same request
were forwarded. The task path had the same shape with `pop()`.

Driven end to end through `Pipe.pipe()` and asserted on the OUTBOUND PAYLOAD. The first
version of this file re-implemented the two shipped blocks as local copies and asserted
against those; a review reintroduced the bug in the real orchestrator with a
regex-evasive ternary and the whole suite stayed green. A copy of the rule tests the copy.
"""

from __future__ import annotations

import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.core.config import EncryptedStr
from tests.test_request_orchestrator import _consume_stream, _smart_callback

_MODEL = "openai/gpt-4o-mini"
# The registry reads the limit from top_provider.max_completion_tokens, NOT from a
# top-level key -- models/registry.py. A fixture that puts it top-level loads a spec with
# max_completion_tokens=None, so the valve-on arm would assert nothing.
_PROVIDER_MAX = 16384
_CATALOG = {
    "data": [
        {
            "id": _MODEL,
            "name": "GPT-4o Mini",
            "top_provider": {"max_completion_tokens": _PROVIDER_MAX},
        }
    ]
}


async def _outbound_max_output_tokens(*, valve: bool, user_limit: int | None) -> object:
    """Drive a real request and return the max_output_tokens that left the pipe."""
    pipe = Pipe()
    captured: list[dict] = []
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = "https://openrouter.ai/api/v1"
        pipe.valves.USE_MODEL_MAX_OUTPUT_TOKENS = valve

        async def _emit(_event):
            pass

        body: dict = {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        if user_limit is not None:
            body["max_tokens"] = user_limit

        with aioresponses() as http:
            http.post(
                "https://openrouter.ai/api/v1/responses",
                callback=_smart_callback(captured, "OK"),
                repeat=True,
            )
            http.get("https://openrouter.ai/api/v1/models", payload=_CATALOG, repeat=True)
            http.get(
                "https://openrouter.ai/api/v1/endpoints/zdr", payload={"data": []}, repeat=True
            )
            result = await pipe.pipe(
                body=body,
                __user__={"id": "u1", "valves": Pipe.UserValves()},
                __request__=None,
                __event_emitter__=_emit,
                __event_call__=None,
                __metadata__={"model": {"id": _MODEL}},
                __tools__=None,
                __task__=None,
                __task_body__=None,
            )
            await _consume_stream(result)
    finally:
        await pipe.close()

    assert captured, "no request reached OpenRouter, so this asserts nothing"
    return captured[-1].get("max_output_tokens")


@pytest.mark.asyncio
async def test_a_forwarded_limit_is_not_announced_as_dropped(caplog):
    """The operator must not be told the user's setting was discarded when it was not.

    `max_tokens` and `reasoning_effort` sat in `unsupported_fields`, so each one logged
    `Dropping unsupported parameter` at WARNING -- and was then re-emitted a few lines
    below as `max_output_tokens` / `reasoning.effort`. Every request that moved Open
    WebUI's Max Tokens slider produced an operator-visible line contradicting the valve
    description, the dashboard detail and the docs page.

    The genuinely-dropped fields must keep warning, so this is asserted in both
    directions: silence for a converted field, a record for an unsupported one.
    """
    import logging

    with caplog.at_level(logging.DEBUG):
        got = await _outbound_max_output_tokens(valve=False, user_limit=128)

    assert got == 128, "the limit did not reach OpenRouter, so the rest proves nothing"
    noisy = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING and "max_tokens" in r.getMessage()
    ]
    assert not noisy, (
        "the pipe warned that the user's max_tokens was dropped, and then forwarded it: "
        f"{noisy}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("valve", [False, True], ids=["valve-off", "valve-on"])
@pytest.mark.parametrize("user_limit", [128, 4096], ids=["128", "4096"])
async def test_a_user_set_limit_reaches_openrouter_in_both_valve_states(valve, user_limit):
    """The user asked for N; N is what goes out, whatever the valve is set to.

    Two distinct limits so a reader hardcoded to either cannot satisfy both arms.
    """
    got = await _outbound_max_output_tokens(valve=valve, user_limit=user_limit)
    assert got == user_limit, (
        f"with USE_MODEL_MAX_OUTPUT_TOKENS={valve}, a user-set max_tokens={user_limit} "
        f"left the pipe as {got!r}. The valve decides whether the PIPE supplies a "
        "default; it must not discard the limit the user asked for."
    )


@pytest.mark.asyncio
async def test_the_valve_off_sends_no_limit_of_the_pipes_own():
    """No user limit, valve off: nothing is sent, so provider defaults apply."""
    got = await _outbound_max_output_tokens(valve=False, user_limit=None)
    assert got is None, (
        f"the request carried no limit and the valve is off, yet {got!r} was sent"
    )


@pytest.mark.asyncio
async def test_the_valve_on_fills_the_providers_advertised_maximum():
    """No user limit, valve on: the provider's own maximum is filled in.

    Without this arm, deleting the block entirely satisfies every other test here.
    """
    got = await _outbound_max_output_tokens(valve=True, user_limit=None)
    assert got == _PROVIDER_MAX, (
        f"the valve is on and the request carried no limit, so the provider's advertised "
        f"maximum ({_PROVIDER_MAX}) should have been filled in; got {got!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("valve", [False, True], ids=["valve-off", "valve-on"])
@pytest.mark.parametrize("out_of_range", [0, -1, -2], ids=["zero", "minus-one", "minus-two"])
async def test_a_max_tokens_outside_openrouters_range_is_not_forwarded(valve, out_of_range):
    """OpenRouter documents max_tokens as "integer, 1 or above".

    Open WebUI's slider allows values below that. Forwarding one produces a 400 from
    OpenRouter that reads to the user as the pipe being broken, so a value outside the
    documented range is sent as no cap at all.

    Both valve states, because the valve must not turn an out-of-range request into a
    filled default either -- with the valve on and no usable limit, the provider's own
    maximum is the right answer, which the next assertion pins.
    """
    got = await _outbound_max_output_tokens(valve=valve, user_limit=out_of_range)
    expected = _PROVIDER_MAX if valve else None
    assert got == expected, (
        f"max_tokens={out_of_range} is outside OpenRouter's documented range (1 or "
        f"above) and left the pipe as {got!r}; expected {expected!r}."
    )


@pytest.mark.asyncio
async def test_the_smallest_valid_limit_is_forwarded():
    """The boundary: 1 is valid and must survive, so the guard is a range check.

    Without this, rejecting everything satisfies the test above.
    """
    got = await _outbound_max_output_tokens(valve=False, user_limit=1)
    assert got == 1, f"max_tokens=1 is valid per OpenRouter and left the pipe as {got!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "expected_choice", "should_warn"),
    [
        ({"function_call": "auto"}, "auto", False),
        ({"function_call": {"name": "my_func"}}, {"type": "function", "name": "my_func"}, False),
        ({"function_call": "my_func"}, None, False),
        ({"function_call": "auto", "tool_choice": "none"}, "none", False),
        ({"extra_tools": [{"type": "function", "name": "probe", "parameters": {}}]}, None, False),
        ({"n": 2}, None, True),
        ({"suffix": "tail"}, None, True),
    ],
    ids=[
        "converted-auto", "converted-named", "malformed-ignored", "tool_choice-wins",
        "extra_tools", "n", "suffix",
    ],
)
async def test_only_the_genuinely_dropped_fields_are_announced_as_dropped(
    kwargs, expected_choice, should_warn, pipe_instance_async, caplog
):
    """Seven arms over one code path, so no constant answer satisfies them.

    `function_call` is carried: it converts to `tool_choice` for "auto"/"none" and for a
    dict naming a function, and the conversion is skipped when `tool_choice` was supplied
    (the modern field wins). `extra_tools` is read off this same CompletionsBody by the
    orchestrator and reaches the advertised tool list. Warning that either was dropped is
    false -- the defect this fixes.

    A malformed `function_call` (any other string) converts to nothing and is silently
    ignored, deliberately: that is the carried-set convention -- `max_tokens` below 1 and
    a falsy `reasoning_effort` are dropped without a word at the same spot. Narrating
    per-value outcomes for one carried field and not its three siblings would be one rule
    with two mechanisms, and reusing "Dropping unsupported parameter" for a SUPPORTED
    field that was merely superseded would be a new false message of the same class.

    The payload assertions are not decoration: a name left out of BOTH sets is copied
    into sanitized_params, and ResponsesBody is extra="allow", so it goes on the wire to
    OpenRouter verbatim.
    """
    import logging

    from open_webui_openrouter_pipe.api.transforms import CompletionsBody, ResponsesBody

    body = CompletionsBody(
        model="gpt-4", messages=[{"role": "user", "content": "hi"}], **kwargs
    )
    with caplog.at_level(logging.WARNING):
        out = await ResponsesBody.from_completions(
            body, transformer_context=pipe_instance_async
        )

    noisy = [
        r.getMessage()
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and "Dropping unsupported parameter" in r.getMessage()
    ]
    dumped = out.model_dump(exclude_none=True)
    assert out.tool_choice == expected_choice
    assert bool(noisy) is should_warn, f"{kwargs}: {noisy}"
    assert "extra_tools" not in dumped, (
        "extra_tools was forwarded to OpenRouter as a foreign top-level key"
    )
    assert "function_call" not in dumped, (
        "function_call was forwarded to OpenRouter as a foreign top-level key"
    )
