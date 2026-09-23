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
import sysconfig
import time
from pathlib import Path
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody
from open_webui_openrouter_pipe.streaming import streaming_core
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
    source = Path(sysconfig.get_paths()["purelib"]) / "open_webui" / "utils" / "middleware.py"
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
        assert fco["output"] == [
            {"type": "input_text",
             "text": "Search completed. Sources available in the citations panel below."}
        ]
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
