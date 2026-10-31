"""A tool the request itself brought goes back to whoever sent it.

Every request reaches the model with the tools it can actually use, and a function tool the pipe has nothing to run
behind goes back whole after exactly one upstream request: to the API caller that sent it, or to Open WebUI when the
request is one Open WebUI can pause. A name nobody offered is answered "Tool not found" inside the loop instead, and a
caller's schema is forwarded untouched.

The harness below drives one chat or API request through `Pipe.pipe()` with the model faked at the transport seam only,
so everything between Open WebUI's call and the HTTP request runs for real.
"""

from __future__ import annotations

import ast
import copy
import inspect
import json
import logging
import textwrap
import time
import uuid
from collections.abc import AsyncGenerator, Callable, Iterator
from typing import Any, cast

import pytest
from pydantic import BaseModel
from starlette.responses import StreamingResponse

import open_webui_openrouter_pipe.pipe as pipe_mod
from open_webui_openrouter_pipe import EncryptedStr, Pipe, generate_item_id
from open_webui_openrouter_pipe.integrations import image_catalog, video_catalog
from open_webui_openrouter_pipe.models.registry import ModelFamily
from tests.package_sources import open_webui_package_dir

MODEL = "test/model"
NORM = "test.model"
CHAT = {"chat_id": "chat-1", "message_id": "message-1", "session_id": "session-1"}
BASE_URL = "https://openrouter.ai/api/v1"
CATALOG = {"data": [{
    "id": NORM, "name": "Test", "norm_id": NORM,
    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
    "supported_parameters": ["tools", "tool_choice", "temperature", "max_tokens"],
}]}

ROW_WITH_TOOLS = {"features": {"function_calling"},
                  "supported_parameters": frozenset({"tools", "tool_choice", "temperature", "max_tokens"})}
ROW_WITHOUT_TOOLS = {"features": set(), "supported_parameters": frozenset({"temperature", "max_tokens", "seed"})}
def call(name: str, args: dict[str, Any], call_id: str = "call-1") -> dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": json.dumps(args),
            "status": "completed"}


def calls_round(*items: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        *({"type": "response.output_item.done", "item": item} for item in items),
        {"type": "response.completed", "response": {"output": list(items), "usage": {}}},
    ]


def answer_round(text: str) -> list[dict[str, Any]]:
    message = {"type": "message", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": text}]}
    return [
        {"type": "response.output_text.delta", "delta": text},
        {"type": "response.completed", "response": {"output": [message], "usage": {}}},
    ]


def outputs_in(request: dict[str, Any]) -> list[Any]:
    found: list[Any] = []
    for item in request.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            out = item.get("output")
            if isinstance(out, list):
                out = "".join(p.get("text", "") for p in out if isinstance(p, dict))
            found.append(str(out))
    return found


def offered(request: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for tool in request.get("tools") or []:
        if tool.get("type") == "function":
            names.append(f"function:{tool.get('name')}")
        else:
            names.append(str(tool.get("type")))
    return names


def install(pipe: Pipe, monkeypatch, script: Callable[[int, dict[str, Any]], list[dict[str, Any]]], *,
            supports_tools: bool = True, row: Any = "default") -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    async def model(self, session, request_body, **_kwargs):
        requests.append(copy.deepcopy(request_body))
        for event in script(len(requests), request_body):
            yield event

    async def loaded(*_a: Any, **_k: Any) -> None:
        return None

    rows: dict[str, Any] = {}

    async def persist(new_rows):
        ulids = [generate_item_id() for _ in new_rows]
        rows.update(zip(ulids, (r["payload"] for r in new_rows)))
        return ulids

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", model)
    monkeypatch.setattr(Pipe, "send_openrouter_nonstreaming_request_as_events", model)
    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test-key", None))
    monkeypatch.setattr(pipe, "_maybe_start_startup_checks", lambda: None)
    monkeypatch.setattr(pipe_mod, "_OwuiConfig", None)
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
    monkeypatch.setattr(pipe._artifact_store, "_make_db_row", lambda _c, _m, _mo, payload: {"payload": payload})
    monkeypatch.setattr(pipe._artifact_store, "_db_persist", persist)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", loaded)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "list_models",
                        lambda: [{"id": MODEL, "name": "Test", "norm_id": NORM}])
    monkeypatch.setattr(image_catalog, "ensure_image_catalog_loaded", loaded)
    monkeypatch.setattr(video_catalog, "ensure_video_catalog_loaded", loaded)
    if row == "default":
        row = ROW_WITH_TOOLS if supports_tools else ROW_WITHOUT_TOOLS
    ModelFamily.set_dynamic_specs({} if row is None else {NORM: dict(row)})
    return requests


def resolved_tool(name: str, fn, *, properties: dict[str, Any] | None = None) -> dict[str, Any]:
    spec = {"name": name, "description": f"{name} tool",
            "parameters": {"type": "object", "properties": properties or {"q": {"type": "string"}}}}
    return {"callable": fn, "spec": spec, "type": "function", "tool_id": name}


def open_webui_stream_content(form_data: dict[str, Any], pipe_call: Any, params: dict[str, Any]) -> Any:
    root = open_webui_package_dir()
    functions_text = (root / "functions.py").read_text(encoding="utf-8")
    misc_text = (root / "utils" / "misc.py").read_text(encoding="utf-8")

    def source_of(node: ast.AST, text: str) -> str:
        return textwrap.dedent(ast.get_source_segment(text, node, padded=True) or "")

    outer = next(
        node for node in ast.parse(functions_text).body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "generate_function_chat_completion"
    )
    nested = {
        node.name: node for node in ast.walk(outer) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    templates = [
        node for node in ast.parse(misc_text).body
        if isinstance(node, ast.FunctionDef)
        and node.name in ("openai_chat_message_template", "openai_chat_chunk_message_template")
    ]
    namespace: dict[str, Any] = {
        "inspect": inspect, "JSONCodec": json, "BaseModel": BaseModel, "StreamingResponse": StreamingResponse,
        "Iterator": Iterator, "AsyncGenerator": AsyncGenerator, "log": logging.getLogger("open_webui.functions"),
        "time": time, "uuid": uuid,
    }
    exec("\n".join(source_of(node, misc_text) for node in templates), namespace)
    wrapper = "\n".join(source_of(nested[name], functions_text) for name in ("process_line", "execute_pipe", "stream_content"))
    exec("def build(form_data, pipe, params):\n" + textwrap.indent(wrapper + "\nreturn stream_content\n", "    "), namespace)
    return namespace["build"](form_data, pipe_call, params)


def frames_of(pieces: list[Any]) -> list[Any]:
    frames: list[Any] = []
    text = "".join(piece.decode("utf-8") if isinstance(piece, bytes) else piece for piece in pieces)
    for block in text.split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                payload = line[len("data:"):].strip()
                frames.append("[DONE]" if payload == "[DONE]" else json.loads(payload))
    return frames


async def drain(result: Any) -> Any:
    if isinstance(result, StreamingResponse):
        pieces = [piece async for piece in result.body_iterator]
        return [frame for frame in frames_of(pieces) if frame != "[DONE]"]
    if hasattr(result, "__aiter__"):
        return [chunk async for chunk in cast(Any, result)]
    return result


def finish_reasons_of(frames: list[Any]) -> list[str]:
    return [
        choice["finish_reason"]
        for frame in frames
        if isinstance(frame, dict)
        for choice in frame.get("choices") or []
        if choice.get("finish_reason")
    ]


async def run(pipe: Pipe, *, mode: str, stream: bool, metadata: dict[str, Any] | None = None,
              body_tools: list[dict[str, Any]] | None = None, owui_tools: dict[str, Any] | None = None,
              loops: int = 4, params: dict[str, Any] | None = None, messages: list[dict[str, Any]] | None = None,
              body_extra: dict[str, Any] | None = None,
              user: dict[str, Any] | None = None, wire: bool = False,
              **valve_updates: Any) -> tuple[Any, list[dict[str, Any]]]:
    valves = pipe.valves.model_copy(update={"TOOL_EXECUTION_MODE": mode, "MAX_FUNCTION_CALL_LOOPS": loops,
                                            **valve_updates})
    valves.API_KEY = EncryptedStr("sk-test-key")
    pipe.valves = valves
    events: list[dict[str, Any]] = []

    async def emitter(event):
        events.append(event)

    meta = {"model": {"id": MODEL}, **(metadata or {})}
    if params is not None:
        meta["params"] = params
    if owui_tools is not None:
        meta["tools"] = owui_tools
    body: dict[str, Any] = {
        "model": NORM,
        "stream": stream,
        "messages": messages or [{"role": "user", "content": "look it up"}],
    }
    if body_tools is not None:
        body["tools"] = body_tools
    body.update(body_extra or {})
    call_params: dict[str, Any] = {
        "body": body, "__user__": user if user is not None else {"id": "user-1", "role": "user"},
        "__request__": None,
        "__event_emitter__": emitter if meta.get("chat_id") else None, "__event_call__": None,
        "__metadata__": meta, "__tools__": owui_tools,
    }
    if wire:
        sent = [piece async for piece in open_webui_stream_content(body, pipe.pipe, call_params)()]
        return frames_of(sent), events
    result: Any = await pipe.pipe(**call_params)
    return await drain(result), events


def _terminal(chunks: Any) -> bool:
    """Did the stream end the reply (terminal output record or a done completion) rather than wait for a call-back?"""
    for chunk in chunks or []:
        if not isinstance(chunk, dict):
            continue
        if chunk.get("type") == "response.completed":
            return True
        event = chunk.get("event")
        if isinstance(event, dict) and event.get("type") == "chat:completion" and (event.get("data") or {}).get("done"):
            return True
    return False


async def _lookup(q: str) -> str:
    return "ok"


# --- the hand-back budget is a reply's own ----------------------------------------------------------------------------


CAP = 2
def always_hands_back(n: int, _request: dict[str, Any]):
    """A turn the model answers with a tool the pipe has nothing to run behind, so it hands back."""
    return calls_round(call("filter_schema", {"q": "x"}, f"call-h{n}"))


def _probe(pipe: Pipe, monkeypatch) -> list[dict[str, Any]]:
    """Install the hand-back script and hand back the request list the tests bill against."""
    return install(pipe, monkeypatch, always_hands_back)


@pytest.mark.asyncio
async def test_a_streaming_request_with_no_chat_gets_its_own_tool_call_back_as_a_chunk_in_open_webui_mode(
    pipe_instance_async, monkeypatch
):
    declared = [{"type": "function", "function": {
        "name": "lookup", "description": "Look something up.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }}]
    pipe = pipe_instance_async
    requests = install(pipe, monkeypatch, lambda n, _request: calls_round(call("lookup", {"q": "x"}, f"call-{n}")))

    chunks, _raw = await run(pipe, mode="Open-WebUI", stream=True, metadata={}, body_tools=declared)
    requests_for_no_chat = len(requests)
    in_a_chat, _raw = await run(pipe, mode="Open-WebUI", stream=True, metadata=dict(CHAT), body_tools=declared)

    returned = [
        tool_call
        for chunk in chunks
        if isinstance(chunk, dict)
        for tool_call in ((chunk.get("choices") or [{}])[0].get("delta") or {}).get("tool_calls") or []
    ]
    assert [(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in returned] == [
        ("call-1", "lookup", {"q": "x"})
    ], chunks
    assert all(isinstance(chunk, dict) and isinstance(chunk.get("choices"), list) for chunk in chunks), chunks
    assert requests_for_no_chat == 1

    assert any(
        isinstance(frame, dict)
        and frame.get("type") == "response.output_item.added"
        and (frame.get("item") or {}).get("type") == "function_call"
        for frame in in_a_chat
    )
    assert not any(
        isinstance(frame, dict) and ((frame.get("choices") or [{}])[0].get("delta") or {}).get("tool_calls")
        for frame in in_a_chat
    )


@pytest.mark.asyncio
async def test_a_streaming_request_with_no_chat_that_ends_on_its_own_tool_call_finishes_with_tool_calls(
    pipe_instance_async, monkeypatch
):
    declared = [{"type": "function", "function": {
        "name": "lookup", "description": "Look something up.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }}]
    pipe = pipe_instance_async
    install(pipe, monkeypatch, lambda n, _request: calls_round(call("lookup", {"q": "x"}, f"call-{n}")))

    in_pipeline_mode, _raw = await run(
        pipe, mode="Pipeline", stream=True, metadata={}, body_tools=declared, wire=True
    )
    in_open_webui_mode, _raw = await run(
        pipe, mode="Open-WebUI", stream=True, metadata={}, body_tools=declared, wire=True
    )

    for read in (in_pipeline_mode, in_open_webui_mode):
        assert finish_reasons_of(read) == ["tool_calls"], read
        assert read[-1] == "[DONE]", read
        finished_at = next(i for i, frame in enumerate(read) if finish_reasons_of([frame]))
        assert any(
            isinstance(frame, dict) and ((frame.get("choices") or [{}])[0].get("delta") or {}).get("tool_calls")
            for frame in read[:finished_at]
        ), read


@pytest.mark.asyncio
async def test_a_request_with_no_chat_that_names_an_open_webui_tool_gets_each_call_back_and_the_pipe_runs_none(
    pipe_instance_async, monkeypatch
):
    ran: list[str] = []

    async def lookup(q: str) -> str:
        ran.append(q)
        return "found"

    registry = {"lookup": resolved_tool("lookup", lookup)}
    declared = [{"type": "function", "function": {
        "name": "lookup", "description": "lookup tool",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }}]
    pipe = pipe_instance_async
    requests = install(
        pipe, monkeypatch,
        lambda n, _request: calls_round(call("lookup", {"q": "x"}, f"call-{n}")) if n == 1 else answer_round("done"),
    )

    for mode in ("Pipeline", "Open-WebUI"):
        for stream in (True, False):
            requests.clear()
            result, _raw = await run(
                pipe, mode=mode, stream=stream, metadata={}, body_tools=declared, owui_tools=registry, wire=stream
            )
            if stream:
                returned = [
                    tool_call
                    for frame in result
                    if isinstance(frame, dict)
                    for tool_call in ((frame.get("choices") or [{}])[0].get("delta") or {}).get("tool_calls") or []
                ]
                finished = finish_reasons_of(result)
            else:
                choice = result["choices"][0]
                returned = choice["message"]["tool_calls"]
                finished = [choice["finish_reason"]]
            where = (mode, stream)
            assert [(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in returned] == [
                ("call-1", "lookup", {"q": "x"})
            ], (where, result)
            assert finished == ["tool_calls"], (where, result)
            assert len(requests) == 1, where
            assert offered(requests[0]) == ["function:lookup"], where
            assert ran == [], where

    requests.clear()
    await run(pipe, mode="Pipeline", stream=True, metadata=dict(CHAT), body_tools=declared, owui_tools=registry)
    assert ran == ["x"]
    assert outputs_in(requests[1]) == ["found"]


@pytest.mark.asyncio
async def test_a_request_with_no_chat_gets_a_call_to_a_tool_withheld_under_legacy_function_calling_back_streamed_or_not(
    pipe_instance_async, monkeypatch
):
    ran: list[str] = []

    async def lookup(q: str) -> str:
        ran.append(q)
        return "found"

    registry = {"lookup": resolved_tool("lookup", lookup)}
    legacy = {"function_calling": "legacy"}
    pipe = pipe_instance_async
    requests = install(
        pipe, monkeypatch,
        lambda n, _request: calls_round(call("lookup", {"q": "x"}, f"call-{n}")) if n == 1 else answer_round("done"),
    )

    for stream in (True, False):
        requests.clear()
        result, _raw = await run(
            pipe, mode="Pipeline", stream=stream, metadata={}, owui_tools=registry, params=legacy, wire=stream
        )
        if stream:
            returned = [
                tool_call
                for frame in result
                if isinstance(frame, dict)
                for tool_call in ((frame.get("choices") or [{}])[0].get("delta") or {}).get("tool_calls") or []
            ]
            finished = finish_reasons_of(result)
        else:
            choice = result["choices"][0]
            returned = choice["message"].get("tool_calls") or []
            finished = [choice["finish_reason"]]
        assert [(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in returned] == [
            ("call-1", "lookup", {"q": "x"})
        ], (stream, result)
        assert finished == ["tool_calls"], (stream, result)
        assert len(requests) == 1, stream
        assert offered(requests[0]) == [], stream
        assert ran == [], stream

    requests.clear()
    in_a_chat, _raw = await run(
        pipe, mode="Pipeline", stream=False, metadata=dict(CHAT), owui_tools=registry, params=legacy
    )
    assert len(requests) == 2
    assert in_a_chat["choices"][0]["message"]["content"].endswith("done")
    assert ran == []


@pytest.mark.asyncio
async def test_a_request_with_no_chat_gets_a_call_to_a_name_nobody_offered_back_and_a_chat_gets_it_answered(
    pipe_instance_async, monkeypatch
):
    declared = [{"type": "function", "function": {
        "name": "lookup", "description": "Look something up.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }}]
    pipe = pipe_instance_async
    requests = install(pipe, monkeypatch, lambda n, _request: calls_round(call("ghost", {"q": "x"}, f"call-{n}")))

    for stream in (True, False):
        requests.clear()
        result, _raw = await run(
            pipe, mode="Pipeline", stream=stream, metadata={}, body_tools=declared, loops=2, wire=stream
        )
        if stream:
            returned = [
                tool_call
                for frame in result
                if isinstance(frame, dict)
                for tool_call in ((frame.get("choices") or [{}])[0].get("delta") or {}).get("tool_calls") or []
            ]
            finished = finish_reasons_of(result)
        else:
            choice = result["choices"][0]
            returned = choice["message"].get("tool_calls") or []
            finished = [choice["finish_reason"]]
        assert [(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in returned] == [
            ("call-1", "ghost", {"q": "x"})
        ], (stream, result)
        assert finished == ["tool_calls"], (stream, result)
        assert len(requests) == 1, stream

    requests.clear()
    await run(pipe, mode="Pipeline", stream=True, metadata=dict(CHAT), body_tools=declared, loops=2)
    assert len(requests) > 1
    assert len(outputs_in(requests[1])) == 1


def reported_by(model: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**event, "response": {**event["response"], "model": model}} if event.get("type") == "response.completed" else event
        for event in events
    ]


@pytest.mark.asyncio
async def test_a_reply_that_is_not_streamed_carries_the_model_openrouter_reports_whether_it_hands_a_call_back_or_answers(
    pipe_instance_async, monkeypatch
):
    declared = [{"type": "function", "function": {
        "name": "lookup", "description": "Look something up.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }}]
    pipe = pipe_instance_async

    install(pipe, monkeypatch, lambda n, _request: reported_by("answering/model", calls_round(call("lookup", {"q": "x"}))))
    handed_back, _raw = await run(pipe, mode="Pipeline", stream=False, metadata={}, body_tools=declared)

    install(pipe, monkeypatch, lambda n, _request: reported_by("answering/model", answer_round("done")))
    answered, _raw = await run(pipe, mode="Pipeline", stream=False, metadata={}, body_tools=declared)

    assert handed_back["choices"][0]["finish_reason"] == "tool_calls"
    assert answered["choices"][0]["message"]["content"] == "done"
    assert (handed_back["model"], answered["model"]) == ("answering/model", "answering/model")
