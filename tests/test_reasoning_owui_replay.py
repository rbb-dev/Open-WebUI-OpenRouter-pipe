"""Keystone end-to-end replay: the pipe's streamed events through Open WebUI's own output assembly.

This test proves that the events the pipe emits, consumed by the installed Open WebUI's output-assembly code, yield a
correct saved ``output`` array: distinct completed reasoning items, intact tool cards, and an intact answer message.

It works in two layers:

1. The REAL pipe streaming loop is driven with fake upstream streams (same harness idioms as
   ``tests/test_reasoning_native_items.py``: ``_make_timed_stream`` / ``_install_clock`` / ``_run``, monkeypatching
   ``Pipe.send_openrouter_streaming_request`` and ``THINKING_OUTPUT_MODE = "open_webui"``). The pipe emits its
   downstream events into a list.

2. Those events go through Open WebUI's own code, compiled from the installed ``utils/middleware.py`` rather than
   imported (conftest stubs ``open_webui``): ``handle_responses_streaming_event`` for the pipe's ``response.*``
   events; for its text, the chunk handler's text branch together with the reasoning- and solution-tag scan
   (``tag_output_handler``) that branch runs; and the chunk handler's end-of-stream cleanup. The text branch and the
   cleanup are inline code in ``stream_body_handler`` and the tag scan is nested in the response handler, so each is
   compiled from its own AST node. Only the loop that routes each event to them is written here. Base64 image
   conversion and the code-interpreter tag scan stay off, as they are by default, and the tag scan uses Open WebUI's
   default reasoning tags, as for a model with no ``reasoning_tags`` setting of its own.
"""

from __future__ import annotations

import __future__
import ast
import copy
import re
import time
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody
from open_webui_openrouter_pipe.core.utils import UNRETAINED_TOOL_RESULT
from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input
from open_webui_openrouter_pipe.streaming import streaming_core
from tests.package_sources import open_webui_package_dir
from tests.test_continue_reasoning_order import _real_artifact_store
from tests.test_continue_stores_once import _open_webui_convert_output_to_messages
from tests.test_tool_rounds_reach_the_model_once import _open_webui_streaming_handler

# ─────────────────────────────────────────────────────────────────────────────
# Open WebUI's output assembly, compiled from the installed source
# ─────────────────────────────────────────────────────────────────────────────

_TAG_CONSTANTS = {"DEFAULT_REASONING_TAGS", "DEFAULT_SOLUTION_TAGS", "DEFAULT_CODE_INTERPRETER_TAGS"}


class _BreakLeavesTheChunk(ast.NodeTransformer):
    """The text branch's `break` leaves Open WebUI's chunk loop; in a function of its own that is a `return`. A `break`
    inside a loop of the branch's own is left alone."""

    def visit_Break(self, node: ast.Break) -> ast.Return:
        return ast.copy_location(ast.Return(value=None), node)

    def visit_For(self, node: ast.AST) -> ast.AST:
        return node

    visit_AsyncFor = visit_For
    visit_While = visit_For


async def _nothing(*_args: Any, **_kwargs: Any) -> None:
    return None


async def _no_image_conversion(*_args: Any, **_kwargs: Any) -> str:
    raise AssertionError("base64 image conversion is off by default")


def _open_webui_assembly() -> dict[str, Any]:
    """Open WebUI's output assembly for one reply, in a namespace of its own: the shared streaming-event handler, plus
    the tag scan, the chunk handler's text branch and its end-of-stream cleanup, each compiled from its AST node in the
    installed `utils/middleware.py`, with the text branch and the cleanup wrapped in functions of their own."""
    namespace = _open_webui_streaming_handler()
    source = open_webui_package_dir() / "utils" / "middleware.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))

    module_level: list[ast.stmt] = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in {"_start_tag_pattern", "append_to_text_field"})
        or (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in _TAG_CONSTANTS for target in node.targets
        ))
    ]
    assert len(module_level) == 5, [getattr(node, "name", None) or ast.dump(node)[:60] for node in module_level]
    (tag_scan,) = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "tag_output_handler"]
    (chunk_handler,) = [
        n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef) and n.name == "stream_body_handler"
    ]
    (text_branch,) = [
        n
        for n in ast.walk(chunk_handler)
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "value"
        and any(isinstance(t, ast.Name) and t.id == "inside_tag_block" for t in ast.walk(n))
    ]
    (cleanup,) = [
        n for n in chunk_handler.body if isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == "output"
    ]

    text = ast.parse("async def _open_webui_text_branch(value):\n    pass\n").body[0]
    finish = ast.parse("def _open_webui_cleanup():\n    pass\n").body[0]
    assert isinstance(text, ast.AsyncFunctionDef) and isinstance(finish, ast.FunctionDef)
    text.body = [cast(ast.stmt, _BreakLeavesTheChunk().visit(copy.deepcopy(stmt))) for stmt in text_branch.body]
    finish.body = [cleanup]
    module = ast.fix_missing_locations(ast.Module(body=[*module_level, tag_scan, text, finish], type_ignores=[]))

    namespace.update(
        re=re, time=time, tag_scan_positions={}, tag_boundary_positions={},
        ENABLE_CHAT_RESPONSE_STREAM_INPLACE_APPEND=False, ENABLE_CHAT_RESPONSE_BASE64_IMAGE_URL_CONVERSION=False,
        convert_markdown_base64_images=_no_image_conversion, request=None, metadata={}, user=None,
        DETECT_REASONING_TAGS=True, DETECT_CODE_INTERPRETER=False, continuing=False,
        output=[], content_parts=[],
        event_emitter=_nothing, flush_pending_delta_data=_nothing, save_current_response_stream=_nothing,
        full_output=lambda: list(namespace["output"]),
    )
    exec(compile(module, str(source), "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True), namespace)
    namespace["reasoning_tags"] = namespace["DEFAULT_REASONING_TAGS"]
    return namespace


async def owui_assemble(emitted: list[dict]) -> list[dict]:
    """The pipe's events through Open WebUI's output assembly, routed as its chunk handler routes them: ``response.*``
    events to its streaming-event handler, ``chat:message:delta`` text to its text branch, everything else (status,
    chat:completion ...) to nothing that builds output. Then the end-of-stream cleanup."""
    assembly = _open_webui_assembly()
    for event in emitted:
        kind = event.get("type", "")
        if kind.startswith("response."):
            handled = assembly["handle_responses_streaming_event"](copy.deepcopy(event), assembly["output"])
            assert handled is not None, f"unhandled event shape: {kind}"
            assembly["output"] = handled[0]
        elif kind == "chat:message:delta":
            value = (event.get("data") or {}).get("content") or ""
            if value:
                await assembly["_open_webui_text_branch"](value)
    assembly["_open_webui_cleanup"]()
    return assembly["output"]


# ─────────────────────────────────────────────────────────────────────────────
# Pipe streaming harness (idioms copied from tests/test_reasoning_native_items.py)
# ─────────────────────────────────────────────────────────────────────────────


def _make_timed_stream(steps: list[tuple[float, dict[str, Any]]], clock: dict[str, float]):
    async def fake_stream(self, session, request_body, **_kwargs):
        for advance, event in steps:
            clock["now"] += advance
            yield event
    return fake_stream


def _install_clock(monkeypatch) -> dict[str, float]:
    clock = {"now": 1000.0}
    monkeypatch.setattr(streaming_core, "_monotonic", lambda: clock["now"])
    return clock


async def _run(pipe, valves, steps, clock, monkeypatch) -> list[dict]:
    body = ResponsesBody(model="test/model", input=[], stream=True)
    monkeypatch.setattr(
        Pipe, "send_openrouter_streaming_request", _make_timed_stream(steps, clock)
    )
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe._streaming_handler._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "test"}},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )
    return emitted


def _reasoning_items(output: list[dict]) -> list[dict]:
    return [i for i in output if i.get("type") == "reasoning"]


def _message_items(output: list[dict]) -> list[dict]:
    return [i for i in output if i.get("type") == "message"]


def _message_text(item: dict) -> str:
    return "".join(
        p.get("text", "")
        for p in (item.get("content") or [])
        if p.get("type") == "output_text"
    )


def _summary_text(item: dict) -> str:
    summary = item.get("summary") or []
    return summary[0].get("text", "") if summary else ""


def _assert_completed_reasoning(item: dict) -> None:
    assert item.get("type") == "reasoning"
    assert item.get("status") == "completed"
    assert item.get("attributes", {}).get("type") != "reasoning_content"
    assert isinstance(item.get("started_at"), float)
    assert isinstance(item.get("ended_at"), float)
    assert isinstance(item.get("duration"), float)
    assert item["duration"] >= 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Case A — snapshot interleave (gpt-5 shape) with a server tool
# ─────────────────────────────────────────────────────────────────────────────


class TestReplayCaseA:
    @pytest.mark.asyncio
    async def test_snapshot_interleave_yields_distinct_boxes_intact_tools_and_answer(
        self, monkeypatch, pipe_instance_async
    ):
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(
            update={"THINKING_OUTPUT_MODE": "open_webui", "SHOW_TOOL_CARDS": True}
        )
        steps = [
            (0.0, {"type": "response.output_item.added",
                   "item": {"type": "reasoning", "id": "rs-1", "summary": []}}),
            (0.5, {"type": "response.output_item.added",
                   "item": {"type": "openrouter:web_search", "id": "ws-1"}}),
            (2.0, {"type": "response.output_item.done",
                   "item": {"type": "openrouter:web_search", "id": "ws-1", "action": {}}}),
            (0.5, {"type": "response.output_item.done",
                   "item": {"type": "reasoning", "id": "rs-1", "status": "completed",
                            "summary": [{"type": "summary_text", "text": "First reasoning block."}],
                            "encrypted_content": "enc-1"}}),
            (0.5, {"type": "response.output_item.added",
                   "item": {"type": "reasoning", "id": "rs-2", "summary": []}}),
            (1.5, {"type": "response.output_item.done",
                   "item": {"type": "reasoning", "id": "rs-2", "status": "completed",
                            "summary": [{"type": "summary_text", "text": "Second reasoning block."}],
                            "encrypted_content": "enc-2"}}),
            (0.5, {"type": "response.output_item.added",
                   "item": {"type": "reasoning", "id": "rs-3", "summary": []}}),
            (0.5, {"type": "response.output_item.done",
                   "item": {"type": "reasoning", "id": "rs-3", "summary": [],
                            "encrypted_content": "enc-3"}}),
            (1.0, {"type": "response.output_text.delta", "delta": "Answer text."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)
        output = await owui_assemble(emitted)

        types = [i.get("type") for i in output]
        assert types == [
            "function_call",
            "function_call_output",
            "reasoning",
            "reasoning",
            "message",
        ], types

        # Exactly two completed reasoning boxes, distinct texts, rs-3 dropped.
        reasoning = _reasoning_items(output)
        assert len(reasoning) == 2
        assert [r.get("id") for r in reasoning] == ["rs-1", "rs-2"]
        for r in reasoning:
            _assert_completed_reasoning(r)
        assert _summary_text(reasoning[0]) == "First reasoning block."
        assert _summary_text(reasoning[1]) == "Second reasoning block."
        assert reasoning[0]["duration"] == pytest.approx(0.5)
        assert reasoning[1]["duration"] == pytest.approx(1.5)
        assert not any(r.get("id") == "rs-3" for r in reasoning)
        assert not any("enc-3" == r.get("encrypted_content") for r in reasoning)

        # Tool cards intact and NOT overwritten with any reasoning fields.
        fc = output[0]
        assert fc == {
            "type": "function_call",
            "id": "srv-ws-1",
            "call_id": "srv-ws-1",
            "name": "web_search",
            "arguments": "{}",
            "status": "completed",
        }
        fco = output[1]
        assert fco["type"] == "function_call_output"
        assert fco["call_id"] == "srv-ws-1"
        assert fco["status"] == "completed"
        for card in (fc, fco):
            assert "summary" not in card
            assert "duration" not in card
            assert "started_at" not in card
            assert card.get("attributes", {}).get("type") != "reasoning_content"

        # Exactly one message, carrying the answer.
        messages = _message_items(output)
        assert len(messages) == 1
        assert _message_text(messages[0]) == "Answer text."


# ─────────────────────────────────────────────────────────────────────────────
# Case B — Anthropic delta shape (late reasoning output_item.done)
# ─────────────────────────────────────────────────────────────────────────────


class TestReplayCaseB:
    @pytest.mark.asyncio
    async def test_anthropic_delta_yields_single_box_before_answer(
        self, monkeypatch, pipe_instance_async
    ):
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        full_reasoning = "Thinking through it. Almost there. "
        steps = [
            (0.0, {"type": "response.output_item.added",
                   "item": {"type": "reasoning", "id": "rs-a"}}),
            (1.0, {"type": "response.reasoning_text.delta",
                   "item_id": "rs-a", "delta": "Thinking through it. "}),
            (1.0, {"type": "response.reasoning_text.delta",
                   "item_id": "rs-a", "delta": "Almost there. "}),
            (1.0, {"type": "response.output_text.delta", "delta": "The answer."}),
            (2.0, {"type": "response.output_item.done",
                   "item": {"type": "reasoning", "id": "rs-a",
                            "content": [{"type": "reasoning_text", "text": full_reasoning}]}}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)
        output = await owui_assemble(emitted)

        types = [i.get("type") for i in output]
        assert types == ["reasoning", "message"], types

        reasoning = _reasoning_items(output)
        assert len(reasoning) == 1
        _assert_completed_reasoning(reasoning[0])
        assert _summary_text(reasoning[0]) == full_reasoning
        assert reasoning[0]["duration"] == pytest.approx(3.0)

        # Reasoning positioned strictly before the message.
        reasoning_idx = next(i for i, it in enumerate(output) if it.get("type") == "reasoning")
        message_idx = next(i for i, it in enumerate(output) if it.get("type") == "message")
        assert reasoning_idx < message_idx

        messages = _message_items(output)
        assert len(messages) == 1
        assert _message_text(messages[0]) == "The answer."

        # No item ever carries the reasoning_content attribute (native path only).
        for item in output:
            assert item.get("attributes", {}).get("type") != "reasoning_content"


# ─────────────────────────────────────────────────────────────────────────────
# Case C — tag-in-answer: literal <think> tag inside the answer text
# ─────────────────────────────────────────────────────────────────────────────


class TestReplayCaseC:
    @pytest.mark.asyncio
    async def test_literal_think_tag_in_answer_is_not_wrapped_in_reasoning(
        self, monkeypatch, pipe_instance_async
    ):
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        full_reasoning = "Thinking through it. Almost there. "
        answer = "The answer <think>not real</think> done."
        steps = [
            (0.0, {"type": "response.output_item.added",
                   "item": {"type": "reasoning", "id": "rs-a"}}),
            (1.0, {"type": "response.reasoning_text.delta",
                   "item_id": "rs-a", "delta": "Thinking through it. "}),
            (1.0, {"type": "response.reasoning_text.delta",
                   "item_id": "rs-a", "delta": "Almost there. "}),
            (1.0, {"type": "response.output_text.delta", "delta": answer}),
            (2.0, {"type": "response.output_item.done",
                   "item": {"type": "reasoning", "id": "rs-a",
                            "content": [{"type": "reasoning_text", "text": full_reasoning}]}}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        # The pipe never wraps the literal tag in a reasoning item: the answer
        # (including the tag) travels solely on chat:message:delta, and the only
        # reasoning emission is the native output_item.added box.
        answer_deltas = [
            (e.get("data") or {}).get("content", "")
            for e in emitted
            if e.get("type") == "chat:message:delta"
        ]
        assert "".join(answer_deltas) == answer
        assert "<think>not real</think>" in "".join(answer_deltas)
        reasoning_events = [
            e for e in emitted
            if e.get("type") == "response.output_item.added"
            and (e.get("item") or {}).get("type") == "reasoning"
        ]
        assert len(reasoning_events) == 1
        assert "<think>" not in _summary_text(reasoning_events[0]["item"])

        output = await owui_assemble(emitted)
        types = [i.get("type") for i in output]
        assert types == ["reasoning", "message"], types

        # Reasoning box intact and unaffected by the tag.
        reasoning = _reasoning_items(output)
        assert len(reasoning) == 1
        _assert_completed_reasoning(reasoning[0])
        assert _summary_text(reasoning[0]) == full_reasoning
        assert reasoning[0]["duration"] == pytest.approx(3.0)

        # The message item holds the literal tag text verbatim: Open WebUI's tag scan splits the
        # streamed text at the tag, but the pipe's record of the reply replaces that list at the end.
        messages = _message_items(output)
        assert len(messages) == 1
        assert _message_text(messages[0]) == answer
        assert "<think>not real</think>" in _message_text(messages[0])


def _chunks_read_by_a_program(frames: list[Any]) -> str:
    return "".join(
        frame["choices"][0]["delta"]["reasoning"]
        for frame in frames
        if isinstance(frame, dict) and (frame.get("choices") or [{}])[0].get("delta", {}).get("reasoning")
    )


async def _reasoning_stored_and_read(
    pipe: Any, monkeypatch: Any, events: list[dict], **valve_updates: Any
) -> tuple[list[str], str]:
    from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import CHAT, install, run

    install(pipe, monkeypatch, lambda _n, _request: events)
    in_a_chat, _raw = await run(pipe, mode="Pipeline", stream=True, metadata=dict(CHAT), **valve_updates)
    stored = await owui_assemble([frame for frame in in_a_chat if isinstance(frame, dict)])
    with_no_chat, _raw = await run(pipe, mode="Pipeline", stream=True, metadata={}, **valve_updates)
    return [_summary_text(item) for item in _reasoning_items(stored)], _chunks_read_by_a_program(with_no_chat)


class TestReplayWholeTextRepeats:
    @pytest.mark.asyncio
    async def test_a_piece_equal_to_the_start_or_end_of_the_text_so_far_is_kept_and_the_providers_repeat_adds_nothing(
        self, monkeypatch, pipe_instance_async
    ):
        text = "First, list 1 to 9999 and then First again."
        pieces = ["First, list 1 to 999", "9", " and then ", "First", " again."]
        item = {"type": "reasoning", "id": "rs-1", "status": "completed", "summary": [],
                "content": [{"type": "reasoning_text", "text": text}]}
        events = [
            {"type": "response.output_item.added", "output_index": 0,
             "item": {"type": "reasoning", "id": "rs-1", "status": "in_progress", "summary": []}},
            *({"type": "response.reasoning_text.delta", "item_id": "rs-1", "output_index": 0, "content_index": 0,
               "delta": piece} for piece in pieces),
            {"type": "response.reasoning_text.done", "item_id": "rs-1", "output_index": 0, "content_index": 0,
             "text": text},
            {"type": "response.content_part.done", "item_id": "rs-1", "output_index": 0, "content_index": 0,
             "part": {"type": "reasoning_text", "text": text}},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.output_text.delta", "delta": "Done."},
            {"type": "response.completed", "response": {"output": [item], "usage": {}}},
        ]

        stored, read = await _reasoning_stored_and_read(pipe_instance_async, monkeypatch, events)

        assert stored == [text]
        assert read == text

    @pytest.mark.asyncio
    async def test_summary_parts_the_provider_repeats_joined_differently_add_nothing_to_the_streamed_text(
        self, monkeypatch, pipe_instance_async
    ):
        parts = [
            ["**Plan the route**\n\n", "Pick the ", "coast road."],
            ["**Pack**\n\n", "Bring water."],
        ]
        texts = ["".join(part) for part in parts]
        events: list[dict] = [
            {"type": "response.output_item.added", "output_index": 0,
             "item": {"type": "reasoning", "id": "rs-1", "status": "in_progress", "summary": []}},
        ]
        for index, (pieces, whole) in enumerate(zip(parts, texts)):
            where = {"item_id": "rs-1", "output_index": 0, "summary_index": index}
            events += [
                {"type": "response.reasoning_summary_part.added", **where, "part": {"type": "summary_text", "text": ""}},
                *({"type": "response.reasoning_summary_text.delta", **where, "delta": piece} for piece in pieces),
                {"type": "response.reasoning_summary_text.done", **where, "text": whole},
                {"type": "response.reasoning_summary_part.done", **where, "part": {"type": "summary_text", "text": whole}},
            ]
        item = {"type": "reasoning", "id": "rs-1", "status": "completed",
                "summary": [{"type": "summary_text", "text": whole} for whole in texts]}
        events += [
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.output_text.delta", "delta": "Done."},
            {"type": "response.completed", "response": {"output": [item], "usage": {}}},
        ]

        stored, read = await _reasoning_stored_and_read(pipe_instance_async, monkeypatch, events)

        assert stored == ["".join(texts)]
        assert read == "".join(texts)

    @pytest.mark.asyncio
    async def test_a_chat_completions_summary_the_reasoning_field_repeats_reaches_the_box_and_the_program_once(
        self, monkeypatch, pipe_instance_async
    ):
        import json

        from aioresponses import aioresponses

        from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import CHAT, install, run

        def chunk(delta: dict, finish: str | None = None) -> str:
            return f"data: {json.dumps({'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish}]})}\n\n"

        pieces = ["**Check the forecast**\n\nI should", " look up", " Paris", " now."]
        body = (
            "".join(
                chunk({"role": "assistant", "content": "", "reasoning": piece, "reasoning_details": [
                    {"type": "reasoning.summary", "summary": piece, "format": "openai-responses-v1", "index": 0}
                ]})
                for piece in pieces
            )
            + chunk({"content": "Sunny."})
            + chunk({}, "stop")
            + "data: [DONE]\n\n"
        )
        pipe = pipe_instance_async
        real_transport = Pipe.send_openrouter_streaming_request
        install(pipe, monkeypatch, lambda _n, _request: [])
        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", real_transport)

        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/chat/completions",
                body=body.encode("utf-8"),
                headers={"Content-Type": "text/event-stream"},
                status=200,
                repeat=True,
            )
            in_a_chat, _raw = await run(
                pipe, mode="Pipeline", stream=True, metadata=dict(CHAT), DEFAULT_LLM_ENDPOINT="chat_completions"
            )
            with_no_chat, _raw = await run(
                pipe, mode="Pipeline", stream=True, metadata={}, DEFAULT_LLM_ENDPOINT="chat_completions"
            )

        stored = await owui_assemble([frame for frame in in_a_chat if isinstance(frame, dict)])
        assert [_summary_text(item) for item in _reasoning_items(stored)] == ["".join(pieces)]
        assert _chunks_read_by_a_program(with_no_chat) == "".join(pieces)

    @pytest.mark.asyncio
    async def test_a_summary_the_reasoning_field_repeats_reaches_the_box_once_when_the_reply_is_not_streamed_and_when_it_closes_the_stream(
        self, monkeypatch, pipe_instance_async
    ):
        import json

        from aioresponses import aioresponses

        from tests.test_a_tool_with_nothing_behind_it_goes_back_to_its_sender import CHAT, install, run

        text = "**Check the forecast**\n\nI should look up Paris now."
        message = {
            "role": "assistant", "content": "Sunny.", "reasoning": text,
            "reasoning_details": [
                {"type": "reasoning.summary", "summary": text, "format": "openai-responses-v1", "index": 0}
            ],
        }
        whole_reply = {
            "id": "gen-1", "model": "test/model", "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 9, "total_tokens": 14},
        }
        closing_chunk = (
            f"data: {json.dumps({'choices': [{'index': 0, 'message': message, 'finish_reason': 'stop'}]})}\n\n"
            "data: [DONE]\n\n"
        )
        pipe = pipe_instance_async
        real_streaming = Pipe.send_openrouter_streaming_request
        real_whole = Pipe.send_openrouter_nonstreaming_request_as_events
        install(pipe, monkeypatch, lambda _n, _request: [])
        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", real_streaming)
        monkeypatch.setattr(Pipe, "send_openrouter_nonstreaming_request_as_events", real_whole)

        with aioresponses() as mock_http:
            url = "https://openrouter.ai/api/v1/chat/completions"
            mock_http.post(url, payload=whole_reply)
            mock_http.post(url, body=closing_chunk.encode("utf-8"), headers={"Content-Type": "text/event-stream"})
            _answer, not_streamed = await run(
                pipe, mode="Pipeline", stream=False, metadata=dict(CHAT), DEFAULT_LLM_ENDPOINT="chat_completions"
            )
            streamed, _raw = await run(
                pipe, mode="Pipeline", stream=True, metadata=dict(CHAT), DEFAULT_LLM_ENDPOINT="chat_completions"
            )

        for frames in (not_streamed, streamed):
            stored = await owui_assemble([frame for frame in frames if isinstance(frame, dict)])
            assert [_summary_text(item) for item in _reasoning_items(stored)] == [text]

    @pytest.mark.asyncio
    async def test_a_whole_text_that_extends_what_was_streamed_adds_only_its_tail(
        self, monkeypatch, pipe_instance_async
    ):
        item = {"type": "reasoning", "id": "rs-1", "status": "completed", "summary": [],
                "content": [{"type": "reasoning_text", "text": "Hello world"}]}
        events = [
            {"type": "response.output_item.added", "output_index": 0,
             "item": {"type": "reasoning", "id": "rs-1", "status": "in_progress", "summary": []}},
            {"type": "response.reasoning_text.delta", "item_id": "rs-1", "output_index": 0, "delta": "Hello"},
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.output_text.delta", "delta": "Done."},
            {"type": "response.completed", "response": {"output": [item], "usage": {}}},
        ]

        stored, read = await _reasoning_stored_and_read(pipe_instance_async, monkeypatch, events)

        assert stored == ["Hello world"]
        assert read == "Hello world"

    @pytest.mark.asyncio
    async def test_a_thought_that_arrives_whole_under_an_id_an_earlier_round_used_is_a_box_of_its_own_unless_it_is_resent(
        self, monkeypatch, pipe_instance_async
    ):
        pipe = pipe_instance_async

        def thought(text: str) -> dict:
            return {"type": "reasoning", "id": "rs-1", "status": "completed",
                    "summary": [{"type": "summary_text", "text": text}]}

        def round_with_a_call(text: str, call_id: str) -> list[dict]:
            call = {"type": "function_call", "id": call_id, "call_id": call_id, "name": "lookup",
                    "arguments": "{}", "status": "completed"}
            return [
                {"type": "response.output_item.done", "output_index": 0, "item": thought(text)},
                {"type": "response.output_item.done", "output_index": 1, "item": call},
                {"type": "response.completed", "response": {"output": [thought(text), call], "usage": {}}},
            ]

        rounds = iter([
            round_with_a_call("Round one thought.", "call-1"),
            round_with_a_call("Round one thought.", "call-2"),
            [
                {"type": "response.output_item.done", "output_index": 0, "item": thought("Round three thought.")},
                {"type": "response.output_text.delta", "delta": "Done."},
                {"type": "response.completed",
                 "response": {"output": [thought("Round three thought.")], "usage": {}}},
            ],
        ])

        async def stream(self, session, request_body, **_kwargs):
            for event in next(rounds):
                yield event

        async def run_tool(calls, registry):
            return [{"type": "function_call_output", "call_id": c["call_id"], "output": "ok"} for c in calls]

        monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", stream)
        monkeypatch.setattr(pipe._ensure_tool_executor(), "_execute_function_calls", run_tool)
        emitted: list[dict] = []

        async def emitter(event):
            emitted.append(copy.deepcopy(event))

        await pipe._streaming_handler._run_streaming_loop(
            ResponsesBody(model="test/model", input=[], stream=True), pipe.valves, emitter,
            metadata={"model": {"id": "test"}}, tools={"lookup": {"callable": lambda **_k: "ok"}},
            session=cast(Any, object()), user_id="u1",
        )

        stored = await owui_assemble(emitted)
        assert [_summary_text(item) for item in _reasoning_items(stored)] == [
            "Round one thought.",
            "Round three thought.",
        ]


SEARCH_MODEL = "anthropic/claude-haiku-5.5"
SEARCH_ANSWER = "Dana Example is the chair of Northwind Gadgets."
SEARCH_PAGES = [
    {
        "url": "https://www.example.com/northwind-gadgets/about",
        "title": "About Northwind Gadgets",
        "content": (
            "Northwind Gadgets builds pocket-sized weather stations and ships them from one warehouse in Exampleton"
            "\n\n...\n\nBoard of directors: the chair is Dana Example. Dana has led the board since 2021 and studied"
            " hydrology at the Example Institute."
        ),
    },
    {
        "url": "https://www.example.org/news/northwind-gadgets-new-chair",
        "title": "Northwind Gadgets names a new chair",
        "content": "Minutes of the March meeting. Present: the chair, Dana Example, and six other directors.",
    },
]
SEARCH_ITEM = {
    "id": "st_tmp_1", "type": "openrouter:web_search", "status": "completed",
    "action": {
        "type": "search", "query": "current chair of Northwind Gadgets",
        "sources": [{"type": "url", "url": page["url"]} for page in SEARCH_PAGES],
    },
}
SEARCH_CITATIONS = [
    {"type": "url_citation", "start_index": 0, "end_index": len(SEARCH_ANSWER), **page} for page in SEARCH_PAGES
]
SEARCH_REASONING = {
    "id": "rs_1", "type": "reasoning", "status": "completed", "summary": [],
    "encrypted_content": "gAAAAA-short", "format": "openai-responses-v1",
}
SEARCH_MESSAGE = {
    "id": "msg_1", "type": "message", "status": "completed", "role": "assistant",
    "content": [{"type": "output_text", "text": SEARCH_ANSWER, "annotations": SEARCH_CITATIONS}],
}


def _search_opens(item: dict, index: int) -> list[dict]:
    return [
        {"type": "response.output_item.added", "output_index": index,
         "item": {"id": item["id"], "type": item["type"], "status": "in_progress"}},
        {"type": "response.output_item.done", "output_index": index, "item": item},
    ]


def _search_answer_events(*, with_search_item: bool) -> list[dict]:
    output = [SEARCH_REASONING, *([SEARCH_ITEM] if with_search_item else []), SEARCH_MESSAGE]
    return [
        {"type": "response.output_item.done", "output_index": 0, "item": SEARCH_REASONING},
        *(_search_opens(SEARCH_ITEM, 1) if with_search_item else []),
        *({"type": "response.output_text.annotation.added", "annotation": citation} for citation in SEARCH_CITATIONS),
        {"type": "response.output_text.delta", "delta": SEARCH_ANSWER},
        {"type": "response.output_item.done", "output_index": 2, "item": SEARCH_MESSAGE},
        {"type": "response.completed", "response": {"output": output, "usage": {}}},
    ]


async def _replays_after_a_search_turn(
    pipe: Any, monkeypatch: Any, responses: list[list[dict]], valves: Any, *, tools: dict | None = None
) -> tuple[list[dict], list[dict]]:
    queue = iter(responses)

    async def stream(self, session, request_body, **_kwargs):
        for event in next(queue):
            yield copy.deepcopy(event)

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", stream)
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(copy.deepcopy(event))

    with _real_artifact_store(pipe) as store:
        await pipe._streaming_handler._run_streaming_loop(
            ResponsesBody(model=SEARCH_MODEL, input=[], stream=True), valves, emitter,
            metadata={"model": {"id": SEARCH_MODEL}, "chat_id": "c1", "message_id": "m1"},
            tools=tools or {}, session=cast(Any, object()), user_id="u1",
        )
        saved = await owui_assemble(emitted)
        answer = _open_webui_convert_output_to_messages()(saved, raw=True, flatten_tool_images=True)
        history = [{"role": "user", "content": "q1"}, *answer, {"role": "user", "content": "q2"}]

        async def replay(replay_valves: Any) -> list[dict]:
            return await transform_messages_to_input(
                pipe, history, chat_id="c1", openwebui_model_id="owui", artifact_loader=store._db_fetch,
                model_id=SEARCH_MODEL, valves=replay_valves,
            )

        return await replay(valves), await replay(valves.model_copy(update={"PERSIST_TOOL_RESULTS": False}))


class TestReplayWebSearch:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("cards", [True, False], ids=["cards-on", "cards-off"])
    async def test_the_next_turn_gets_the_query_and_the_results_of_the_search_the_model_ran(
        self, monkeypatch, pipe_instance_async, cards
    ):
        pipe = pipe_instance_async
        valves = pipe.valves.model_copy(update={"PERSIST_TOOL_RESULTS": True, "SHOW_TOOL_CARDS": cards})

        kept, not_kept = await _replays_after_a_search_turn(
            pipe, monkeypatch, [_search_answer_events(with_search_item=True)], valves
        )

        assert [i["role"] if i["type"] == "message" else i["type"] for i in kept] == [
            "user", "reasoning", "function_call", "function_call_output", "assistant", "user",
        ]
        (call,) = [i for i in kept if i["type"] == "function_call"]
        (result,) = [i for i in kept if i["type"] == "function_call_output"]
        assert call["call_id"] == result["call_id"]
        assert SEARCH_ITEM["action"]["query"] in call["arguments"]
        for page in SEARCH_PAGES:
            assert page["url"] in result["output"]
        assert SEARCH_PAGES[0]["content"][:60] in result["output"]

        (call_not_kept,) = [i for i in not_kept if i["type"] == "function_call"]
        (result_not_kept,) = [i for i in not_kept if i["type"] == "function_call_output"]
        assert call_not_kept["arguments"] == call["arguments"]
        assert result_not_kept["output"] == UNRETAINED_TOOL_RESULT

    @pytest.mark.asyncio
    @pytest.mark.parametrize("cards", [True, False], ids=["cards-on", "cards-off"])
    async def test_a_search_followed_by_a_tool_call_takes_the_citations_the_answer_brings(
        self, monkeypatch, pipe_instance_async, cards
    ):
        pipe = pipe_instance_async
        call = {
            "type": "function_call", "id": "fc_1", "call_id": "call_lookup_1", "name": "lookup",
            "arguments": "{}", "status": "completed",
        }
        first = [
            *_search_opens(SEARCH_ITEM, 0),
            {"type": "response.output_item.added", "output_index": 1, "item": {**call, "arguments": "", "status": "in_progress"}},
            {"type": "response.output_item.done", "output_index": 1, "item": call},
            {"type": "response.completed", "response": {"output": [SEARCH_ITEM, call], "usage": {}}},
        ]

        async def run_tool(calls, registry):
            return [{"type": "function_call_output", "call_id": c["call_id"], "output": "ok"} for c in calls]

        monkeypatch.setattr(pipe._ensure_tool_executor(), "_execute_function_calls", run_tool)
        valves = pipe.valves.model_copy(update={"PERSIST_TOOL_RESULTS": True, "SHOW_TOOL_CARDS": cards})

        kept, _ = await _replays_after_a_search_turn(
            pipe, monkeypatch, [first, _search_answer_events(with_search_item=False)], valves,
            tools={"lookup": {"callable": lambda **_k: "ok"}},
        )

        result = next(i for i in kept if i["type"] == "function_call_output" and i["call_id"].startswith("srv-"))
        assert SEARCH_PAGES[0]["content"][:60] in result["output"]
