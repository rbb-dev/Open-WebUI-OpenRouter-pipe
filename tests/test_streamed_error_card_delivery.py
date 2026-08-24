"""With streaming ON, an error card only arrives if it is joined to the answer so far.

The middleware stream emitter is append-only by construction. Its ``chat:message`` branch
forwards ``content[len(assistant_sent):]`` and only when ``content`` starts with what it
has already sent, so a bare card -- the card and nothing else -- is a non-prefix write the
moment any answer text has streamed, and the branch enqueues nothing at all. The user
watched the model start answering, the turn failed, and the browser was left holding a
sentence that stops mid-thought with no error anywhere on screen.

Every test drives the real ``_run_streaming_loop`` and the real
``_make_middleware_stream_emitter``. The stub is one seam BELOW the handler under test --
the event source the loop iterates -- so the handler body runs for real, and what is
asserted is the ``delta.content`` Open WebUI's own accumulator would concatenate, taken
off the queue rather than from a literal.

Each is parametrised over two rows differing in BOTH the answer text and the card, so
neither a hardcoded card nor a hardcoded partial can satisfy every row.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody
from open_webui_openrouter_pipe.core.errors import (
    OpenRouterAPIError,
    RequiredInternalFileError,
)

MODEL = "m1"


class _NoSession:
    """Stand-in for the aiohttp session; nothing under test ever calls it."""


def _stream_emitter(pipe: Pipe, queue: asyncio.Queue[Any]) -> Any:
    """The emitter Open WebUI is actually handed on the streaming leg."""
    job = SimpleNamespace(
        request_id="req-streamed-card",
        metadata={"model": {"id": MODEL}},
        body={"model": MODEL},
        valves=pipe.valves,
        future=asyncio.get_running_loop().create_future(),
        event_emitter=None,
        stream_queue=queue,
    )
    return pipe._event_emitter_handler._make_middleware_stream_emitter(cast(Any, job), queue)


def _drain(queue: asyncio.Queue[Any]) -> list[Any]:
    """Everything the pipe put on the queue Open WebUI reads, in order."""
    items: list[Any] = []
    while not queue.empty():
        items.append(queue.get_nowait())
    return items


def _content(items: list[Any]) -> str:
    """Concatenate exactly what Open WebUI's own accumulator would see."""
    parts: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        choices = item.get("choices")
        if not (isinstance(choices, list) and choices and isinstance(choices[0], dict)):
            continue
        delta = choices[0].get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("content"), str):
            parts.append(delta["content"])
    return "".join(parts)


def _banner(items: list[Any]) -> str:
    """The error text the same queue carries out of band, which OWUI shows in its banner."""
    return "\n".join(
        str(item["error"].get("message") or "")
        for item in items
        if isinstance(item, dict) and isinstance(item.get("error"), dict)
    )


def _accumulated(queue: asyncio.Queue[Any]) -> str:
    """Concatenate exactly what Open WebUI's own accumulator would see."""
    return _content(_drain(queue))


async def _drive_items(pipe: Pipe, source: Any) -> tuple[list[Any], str]:
    """Run the real loop on the streaming leg, returning (queue items, returned)."""
    queue: asyncio.Queue[Any] = asyncio.Queue()
    emitter = _stream_emitter(pipe, queue)
    body = ResponsesBody(model=MODEL, input=[], stream=True)
    returned = await pipe._streaming_handler._run_streaming_loop(
        body,
        pipe.valves,
        emitter,
        {},
        {},
        session=cast(Any, _NoSession()),
        user_id="u",
        event_source=source,
    )
    return _drain(queue), cast(str, returned)


async def _drive(pipe: Pipe, source: Any) -> tuple[str, str]:
    """Run the real loop on the streaming leg, returning (accumulated, returned)."""
    items, returned = await _drive_items(pipe, source)
    return _content(items), returned


def _assert_both_in_order(accumulated: str, partial: str, marker: str) -> None:
    assert partial in accumulated, (
        "the model produced this much before the failure and the browser had already "
        f"rendered it; replacing it with a card throws it away. got {accumulated!r}"
    )
    assert marker in accumulated, (
        "the turn failed and the browser was told nothing, so the user is left reading a "
        f"sentence that stops mid-thought. got {accumulated!r}"
    )
    assert accumulated.index(partial) < accumulated.index(marker), (
        f"the card must follow the answer it interrupted, not precede it. got {accumulated!r}"
    )


# ---------------------------------------------------------------------------
# The response loop's catch-all
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("partial", "card", "marker"),
    [
        ("The capital of France is", "### Something broke\n\nA {error_type} ended the turn.", "Something broke"),
        ("Photosynthesis begins when", "### Turn abandoned\n\nA {error_type} stopped it.", "Turn abandoned"),
    ],
)
async def test_the_catch_all_card_reaches_a_browser_that_already_has_text(
    pipe_instance_async, partial, card, marker
) -> None:
    """A card emitted alone here is silently dropped; joined to the answer it arrives."""
    pipe = pipe_instance_async
    pipe.valves.INTERNAL_ERROR_TEMPLATE = card

    async def source() -> Any:
        yield {"type": "response.created", "response": {"model": MODEL}}
        yield {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}}
        yield {"type": "response.output_text.delta", "output_index": 0, "delta": partial}
        raise ValueError("the upstream connection died")

    accumulated, returned = await _drive(pipe, source())

    _assert_both_in_order(accumulated, partial, marker)
    assert partial in returned and marker in returned, (
        "the non-streaming leg keeps only the return value, so the same join has to be "
        f"there too. got {returned!r}"
    )


# ---------------------------------------------------------------------------
# Its OpenRouter rejection sibling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("partial", "card", "marker", "status"),
    [
        ("Newton's second law says", "### Out of credits\n\n{openrouter_message}", "Out of credits", 402),
        ("The Treaty of Versailles was", "### Too many requests\n\n{openrouter_message}", "Too many requests", 429),
    ],
)
async def test_the_rejection_card_reaches_a_browser_that_already_has_text(
    pipe_instance_async, partial, card, marker, status
) -> None:
    """The sibling handler routes through ``_report_openrouter_error`` and dropped it too.

    The status also selects the template, so the two rows exercise two different renders
    and one constant cannot serve both.
    """
    pipe = pipe_instance_async
    pipe.valves.INSUFFICIENT_CREDITS_TEMPLATE = card
    pipe.valves.RATE_LIMIT_TEMPLATE = card

    async def source() -> Any:
        yield {"type": "response.created", "response": {"model": MODEL}}
        yield {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}}
        yield {"type": "response.output_text.delta", "output_index": 0, "delta": partial}
        raise OpenRouterAPIError(
            status=status,
            reason="rejected",
            openrouter_message=f"provider said {status}",
        )

    accumulated, returned = await _drive(pipe, source())

    _assert_both_in_order(accumulated, partial, marker)
    assert f"provider said {status}" in accumulated, (
        f"the provider's own words are what make the card actionable. got {accumulated!r}"
    )
    assert partial in returned and marker in returned, (
        "the non-streaming leg keeps only the return value, so the same join has to be "
        f"there too. got {returned!r}"
    )


# ---------------------------------------------------------------------------
# Its required-internal-file sibling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("partial", "denial"),
    [
        ("The ledger for March opens with", "Access to 'ledger-march.csv' was refused."),
        ("Reading the survey results,", "The file 'survey-2026.pdf' is no longer available."),
    ],
)
async def test_a_required_file_denial_keeps_the_answer_it_interrupted(
    pipe_instance_async, partial, denial
) -> None:
    """A file the turn needed mid-stream went missing, after the model had already answered.

    This branch does not render a card into the answer stream: the denial leaves on the
    ``error`` channel of the same queue, which Open WebUI shows in its banner. What the
    branch decides is what becomes of the answer already produced -- it used to be
    assigned over, which threw away text the browser had rendered and left the stored
    message holding only the denial. Two rows differing in BOTH the partial and the
    denial, so no constant on either side of the join satisfies both.
    """
    pipe = pipe_instance_async

    async def source() -> Any:
        yield {"type": "response.created", "response": {"model": MODEL}}
        yield {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}}
        yield {"type": "response.output_text.delta", "output_index": 0, "delta": partial}
        raise RequiredInternalFileError(denial, kind="file", denied=True)

    items, returned = await _drive_items(pipe, source())

    assert partial in _content(items), (
        f"the browser had already rendered this much. got {_content(items)!r}"
    )
    assert denial in _banner(items), (
        f"the denial never reached the browser's error banner. got {_banner(items)!r}"
    )
    _assert_both_in_order(returned, partial, denial)


# ---------------------------------------------------------------------------
# What already worked has to keep working
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("card", "marker"),
    [
        ("### Something broke\n\nA {error_type} ended the turn.", "Something broke"),
        ("### Turn abandoned\n\nA {error_type} stopped it.", "Turn abandoned"),
    ],
)
async def test_a_card_with_no_answer_before_it_arrives_exactly_as_it_did(
    pipe_instance_async, card, marker
) -> None:
    """Nothing streamed, so the card is the whole message and must not gain a separator."""
    pipe = pipe_instance_async
    pipe.valves.INTERNAL_ERROR_TEMPLATE = card

    async def source() -> Any:
        yield {"type": "response.created", "response": {"model": MODEL}}
        raise ValueError("died before a single token")

    accumulated, returned = await _drive(pipe, source())

    assert accumulated.startswith(marker) or accumulated.startswith("###"), (
        f"a leading separator would indent the card under a blank line. got {accumulated!r}"
    )
    assert marker in accumulated, f"the card never reached the browser. got {accumulated!r}"
    assert not accumulated.startswith("\n"), (
        f"there was no answer to append to, so nothing may precede the card. got {accumulated!r}"
    )
    assert returned.strip() == accumulated.strip(), (
        f"both legs must carry the same card. returned {returned!r}, streamed {accumulated!r}"
    )


# ---------------------------------------------------------------------------
# An attempt the orchestrator hands back for a retry
# ---------------------------------------------------------------------------


class _FakeItemModel:
    """Stands in for the SQLAlchemy artifact row class so `_make_db_row` runs for real."""


def _persisting(pipe: Pipe, monkeypatch: pytest.MonkeyPatch) -> None:
    """Let reasoning items reach the artifact store, stubbing only the database write."""
    pipe.valves.PERSIST_REASONING_TOKENS = "conversation"
    monkeypatch.setattr(pipe._artifact_store, "_item_model", _FakeItemModel, raising=False)

    persisted: list[str] = []

    async def _db_persist(rows: list[dict[str, Any]]) -> list[str]:
        ulids = [f"01JABCDEF0123456789ABCDE{len(persisted) + i:02d}" for i in range(len(rows))]
        persisted.extend(ulids)
        return ulids

    monkeypatch.setattr(pipe._artifact_store, "_db_persist", _db_persist)


async def _drive_shared(pipe: Pipe, emitter: Any, source: Any) -> str:
    """One attempt, against an emitter that outlives it -- exactly as the retry loop does."""
    body = ResponsesBody(model=MODEL, input=[], stream=True)
    returned = await pipe._streaming_handler._run_streaming_loop(
        body,
        pipe.valves,
        emitter,
        {"chat_id": "chat-retry", "message_id": "msg-retry"},
        {},
        session=cast(Any, _NoSession()),
        user_id="u",
        event_source=source,
    )
    return cast(str, returned)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("partial", "card", "marker", "status"),
    [
        ("Newton's second law says", "### Out of credits\n\n{openrouter_message}", "Out of credits", 402),
        ("The Treaty of Versailles was", "### Too many requests\n\n{openrouter_message}", "Too many requests", 429),
    ],
)
async def test_a_second_attempt_still_puts_the_card_on_screen(
    monkeypatch, pipe_instance_async, partial, card, marker, status
) -> None:
    """The translator's accumulator outlives the loop; the answer text does not.

    ``_run_streaming_loop`` is invoked once per attempt by the orchestrator's retry loop,
    against ONE ``_make_middleware_stream_emitter``. An attempt handed back for a retry was
    still emitting its persisted-artifact markers on the way out, so the shared accumulator
    ended up holding text no later attempt's snapshot could begin with -- and from then on
    every card was a non-prefix write the translator discarded in silence.

    Both attempts run the real loop and the real emitter; the only stub is the database
    write the artifact store performs, one seam below. Two partials and two cards, and the
    status selects the template, so no constant satisfies both rows.
    """
    pipe = pipe_instance_async
    pipe.valves.INSUFFICIENT_CREDITS_TEMPLATE = card
    pipe.valves.RATE_LIMIT_TEMPLATE = card
    _persisting(pipe, monkeypatch)

    queue: asyncio.Queue[Any] = asyncio.Queue()
    emitter = _stream_emitter(pipe, queue)

    def _rejection() -> OpenRouterAPIError:
        return OpenRouterAPIError(
            status=status,
            reason="rejected",
            openrouter_message=f"provider said {status}",
        )

    async def first() -> Any:
        yield {"type": "response.created", "response": {"model": MODEL}}
        yield {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "reasoning",
                "id": "rs-first",
                "summary": [{"type": "summary_text", "text": "weighing the options"}],
                "status": "completed",
            },
        }
        raise _rejection()

    with pytest.raises(OpenRouterAPIError):
        await _drive_shared(pipe, emitter, first())

    handed_back = _accumulated(queue)

    async def second() -> Any:
        yield {"type": "response.created", "response": {"model": MODEL}}
        yield {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message"}}
        yield {"type": "response.output_text.delta", "output_index": 0, "delta": partial}
        raise _rejection()

    returned = await _drive_shared(pipe, emitter, second())
    on_screen = handed_back + _accumulated(queue)

    _assert_both_in_order(on_screen, partial, marker)
    assert f"provider said {status}" in on_screen
    assert handed_back == "", (
        "an attempt the retry supersedes contributes nothing the browser keeps, so anything "
        f"it wrote is text no later attempt can start with. got {handed_back!r}"
    )
    assert partial in returned and marker in returned, (
        f"the non-streaming leg keeps only the return value. got {returned!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("phase", "card", "marker", "status"),
    [
        ("commentary", "### Out of credits\n\n{openrouter_message}", "Out of credits", 402),
        ("final_answer", "### Too many requests\n\n{openrouter_message}", "Too many requests", 429),
    ],
)
async def test_a_phase_marker_on_the_wire_ends_the_attempt_rather_than_being_retried(
    pipe_instance_async, phase, card, marker, status
) -> None:
    """The same accumulator damage, from the marker site that runs mid-stream instead.

    A completed assistant message item carrying a ``phase`` streams its hidden marker
    immediately, and that site set no retry barrier -- so the attempt could still be handed
    back after putting bytes in front of the reader. Once anything of this attempt's has
    been written, the attempt owns the turn: it reports rather than retries. Two phases and
    two cards, and the status selects the template.
    """
    pipe = pipe_instance_async
    pipe.valves.INSUFFICIENT_CREDITS_TEMPLATE = card
    pipe.valves.RATE_LIMIT_TEMPLATE = card

    queue: asyncio.Queue[Any] = asyncio.Queue()
    emitter = _stream_emitter(pipe, queue)

    async def only_attempt() -> Any:
        yield {"type": "response.created", "response": {"model": MODEL}}
        yield {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "message",
                "role": "assistant",
                "phase": phase,
                "content": [{"type": "output_text", "text": "a phase-tagged block"}],
            },
        }
        raise OpenRouterAPIError(
            status=status,
            reason="rejected",
            openrouter_message=f"provider said {status}",
        )

    returned = await _drive_shared(pipe, emitter, only_attempt())
    accumulated = _accumulated(queue)

    assert f"[P:{phase}]" in accumulated, (
        f"the phase marker is what makes this attempt unretryable. got {accumulated!r}"
    )
    assert marker in accumulated, (
        "the attempt was handed back for a retry after writing to the browser, so its card "
        f"was never rendered and the next attempt's was discarded. got {accumulated!r}"
    )
    assert accumulated.index(f"[P:{phase}]") < accumulated.index(marker)
    assert marker in returned, f"the non-streaming leg carries the card too. got {returned!r}"
