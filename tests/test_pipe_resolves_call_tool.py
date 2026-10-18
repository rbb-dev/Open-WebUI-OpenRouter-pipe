"""The pipe resolves Open WebUI 0.12.2's deferred-tool dispatcher itself.

When Open WebUI's Tool Search holds large tool schemas behind a manifest, the model reaches them
through `call_tool`. On the pipe's own rounds the pipe resolves that dispatch with Open WebUI's own
`resolve_tool_call` and executes the target through its executor, so the reply keeps going and the
next request carries the call exactly as the model made it. Where Open WebUI runs the round itself
(a streamed Open-WebUI reply, a streamed ask-approval hand-back) the pipe only sheds the held-back
specs from its offer, the same ones Open WebUI's own request omits.
"""

from __future__ import annotations

import asyncio
import ast
import copy
import json
import re
import sysconfig
from pathlib import Path
from typing import Any, cast

import pytest

import open_webui_openrouter_pipe.requests.orchestrator as orchestrator_mod
import open_webui_openrouter_pipe.streaming.streaming_core as streaming_core_mod
from open_webui_openrouter_pipe import EncryptedStr, Pipe, ResponsesBody
from open_webui_openrouter_pipe.models.registry import ModelFamily
from tests.test_reasoning_skeleton_replay import _recorded_output, _stage_b
from tests.test_tools import _as_open_webui_resolves_them

_CALL_TOOL_ARGUMENTS = json.dumps({"tool_name": "lookup", "params": json.dumps({"q": "v"})})
_MALFORMED_CALL_ARGUMENTS = _CALL_TOOL_ARGUMENTS[:-6]


def _open_webui_tool_search_functions() -> dict[str, Any]:
    """Open WebUI's own `resolve_tool_call` and `strip_deferred_tools_manifest`, compiled from the
    installed source with the constants and codec they read.

    The suite's `open_webui` stub carries no `utils` submodules, so the pipe's own imports of these
    resolve to None; the tests rebind the pipe's module-level symbols to the real functions, compiled
    the same way as this suite's other Open WebUI seams. Nothing is added to `sys.modules`.
    """
    source = Path(sysconfig.get_paths()["purelib"]) / "open_webui" / "utils" / "tool_search.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    namespace: dict[str, Any] = {"re": re, "JSONCodec": json}
    wanted = {"resolve_tool_call", "strip_deferred_tools_manifest"}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) in {"CALL_TOOL_NAME", "SEARCH_TOOL_NAME", "_MANIFEST_RE"}
            for target in node.targets
        ):
            exec(  # noqa: S102 - the suite's established Open WebUI seam compile
                compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
                namespace,
            )
        elif isinstance(node, ast.FunctionDef) and node.name in wanted:
            exec(  # noqa: S102 - see above
                compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
                namespace,
            )
    assert namespace["CALL_TOOL_NAME"] == "call_tool", namespace["CALL_TOOL_NAME"]
    assert all(callable(namespace[name]) for name in wanted)
    return namespace


def _model_rounds(*, with_malformed_call: bool = False) -> tuple[Any, list[dict[str, Any]]]:
    step = [0]
    bodies: list[dict[str, Any]] = []

    async def model(self, session, request_body, **_kwargs):
        step[0] += 1
        bodies.append(copy.deepcopy(request_body))
        if step[0] == 1:
            call = {
                "type": "function_call",
                "call_id": "call_A",
                "name": "call_tool",
                "arguments": _CALL_TOOL_ARGUMENTS,
                "status": "completed",
            }
            output = [call]
            yield {"type": "response.output_item.done", "item": call}
            if with_malformed_call:
                malformed = {
                    "type": "function_call",
                    "call_id": "call_B",
                    "name": "call_tool",
                    "arguments": _MALFORMED_CALL_ARGUMENTS,
                    "status": "completed",
                }
                output.append(malformed)
                yield {"type": "response.output_item.done", "item": malformed}
            yield {"type": "response.completed", "response": {"output": output, "usage": {}}}
            return
        yield {"type": "response.output_text.delta", "delta": "lookup ran, all done."}
        yield {"type": "response.completed", "response": {"output": [], "usage": {}}}

    return model, bodies


def _metadata_with_deferred_lookup() -> dict[str, Any]:
    lookup_spec = {
        "name": "lookup",
        "description": "Look it up.",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
    }
    metadata = _as_open_webui_resolves_them({
        "chat_id": "c1",
        "message_id": "m1",
        "model": {"id": "test-model"},
        "tool_servers": [{"url": "wss://example", "specs": [dict(lookup_spec)]}],
    })
    metadata["tools"]["call_tool"] = {
        "tool_id": "builtin:call_tool",
        "type": "builtin",
        "spec": {
            "name": "call_tool",
            "description": "Invoke a tool found by search_tools.",
            "parameters": {
                "type": "object",
                "properties": {"tool_name": {"type": "string"}, "params": {"type": "string"}},
            },
        },
    }
    metadata["deferred_tools"] = ["lookup"]
    return metadata


def _dynamic_model_specs():
    return {
        "test-model": {
            "features": {"function_calling"},
            "supported_parameters": frozenset(["tools", "tool_choice"]),
        }
    }


@pytest.mark.asyncio
async def test_a_pipe_run_reply_resolves_the_models_call_tool_and_answers(monkeypatch):
    """The regression: a non-streamed reply whose round is a `call_tool` dispatch.

    The held-back target is a browser-run direct tool, the one shape with no callable of its own in
    Open WebUI's registry -- the pipe's bridge twin is the only executor there is. The reply must end
    as final assistant text with the target run exactly once, and the next request must carry the
    model's own `call_tool` call, so the model sees the call it made.
    """
    functions = _open_webui_tool_search_functions()
    monkeypatch.setattr(streaming_core_mod, "resolve_tool_call", functions["resolve_tool_call"], raising=False)
    monkeypatch.setattr(orchestrator_mod, "resolve_tool_call", functions["resolve_tool_call"], raising=False)
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("sk-test-key")
    model, model_bodies = _model_rounds(with_malformed_call=True)
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", model)
    monkeypatch.setattr(Pipe, "send_openrouter_nonstreaming_request_as_events", model)
    bridge_payloads: list[dict[str, Any]] = []

    async def event_call(payload: dict[str, Any]) -> Any:
        bridge_payloads.append(payload)
        return {"result": "lookup ran"}

    metadata = _metadata_with_deferred_lookup()
    malformed_refusal = functions["resolve_tool_call"](
        {"function": {"name": "call_tool", "arguments": _MALFORMED_CALL_ARGUMENTS}},
        metadata["tools"],
    )
    assert isinstance(malformed_refusal, str) and malformed_refusal.startswith("Error: "), malformed_refusal
    ModelFamily.set_dynamic_specs(_dynamic_model_specs())
    from open_webui_openrouter_pipe.tools.tool_executor import _ToolExecutionContext

    context = _ToolExecutionContext(
        queue=asyncio.Queue(maxsize=50),
        per_request_semaphore=asyncio.Semaphore(1),
        global_semaphore=None,
        timeout=5.0,
        batch_timeout=5.0,
        idle_timeout=None,
        user_id="u1",
        event_emitter=None,
    )
    context.workers.append(asyncio.create_task(pipe._ensure_tool_executor()._tool_worker_loop(context)))
    token = pipe._TOOL_CONTEXT.set(context)
    session = None
    try:
        session = pipe._create_http_session(pipe.valves)
        result = await pipe._process_transformed_request(
            {
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "call_tool",
                            "description": "Invoke a tool found by search_tools.",
                            "parameters": {
                                "type": "object",
                                "properties": {"tool_name": {"type": "string"}, "params": {"type": "string"}},
                            },
                        },
                    },
                ],
            },
            __user__={"id": "u1", "valves": {}},
            __request__=None,
            __event_emitter__=None,
            __event_call__=event_call,
            __metadata__=metadata,
            __tools__={},
            __task__=None,
            __task_body__=None,
            session=session,
            openwebui_model_id="test-model",
            pipe_identifier="pipe.test",
            allowlist_norm_ids=set(),
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            valves=pipe.valves,
            features={},
        )
    finally:
        pipe._TOOL_CONTEXT.reset(token)
        for worker in context.workers:
            worker.cancel()
        await asyncio.gather(*context.workers, return_exceptions=True)
        ModelFamily.set_dynamic_specs({})
        if session is not None:
            await session.close()
        await pipe.close()

    assert len(model_bodies) == 2, [body.get("model") for body in model_bodies]
    round_two_calls = [
        item for item in model_bodies[1].get("input") or []
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]
    assert [(call.get("name"), call.get("arguments")) for call in round_two_calls] == [
        ("call_tool", _CALL_TOOL_ARGUMENTS),
        ("call_tool", _MALFORMED_CALL_ARGUMENTS),
    ]
    assert any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        and item.get("call_id") == "call_A"
        for item in model_bodies[1].get("input") or []
    ), model_bodies[1].get("input")
    malformed_output = next(
        item.get("output") for item in model_bodies[1].get("input") or []
        if isinstance(item, dict) and item.get("type") == "function_call_output"
        and item.get("call_id") == "call_B"
    )
    assert malformed_output == malformed_refusal, (malformed_output, malformed_refusal)
    assert len(bridge_payloads) == 1, bridge_payloads
    assert bridge_payloads[0]["data"]["name"] == "lookup"
    assert bridge_payloads[0]["data"]["params"] == {"q": "v"}
    assert isinstance(result, str) and "all done." in result, result


@pytest.mark.asyncio
@pytest.mark.parametrize("cards", [True, False], ids=["cards-on", "cards-off"])
async def test_a_stored_call_tool_round_replays_as_the_model_made_it(monkeypatch, pipe_instance_async, cards):
    """Both replay consumers restore the model's own `call_tool` call from the stored round.

    With cards on, Open WebUI's converter rebuilds the round from the published item, which carries
    `provider_function`; with cards off, the pipe's own stored row does, through the replay formation.
    At the `PERSIST_TOOL_RESULTS` default only the round's result is withheld, so both routes must replay
    the call with the arguments the model gave it, and no replayed item may carry the resolver's own key.
    """
    functions = _open_webui_tool_search_functions()
    monkeypatch.setattr(streaming_core_mod, "resolve_tool_call", functions["resolve_tool_call"], raising=False)
    pipe = pipe_instance_async
    model, _model_bodies = _model_rounds()
    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", model)
    monkeypatch.setattr(Pipe, "send_openrouter_nonstreaming_request_as_events", model)

    async def lookup(**_kwargs):
        return "found"

    owui_lookup = {
        "tool_id": "local_lookup",
        "callable": lookup,
        "spec": {"name": "lookup", "parameters": {"type": "object", "properties": {}}},
    }
    call_tool_entry = {
        "tool_id": "builtin:call_tool",
        "type": "builtin",
        "spec": {"name": "call_tool", "parameters": {"type": "object", "properties": {}}},
    }
    persisted: dict[str, dict[str, Any]] = {}
    emitted: list[dict[str, Any]] = []

    def make_row(_chat_id, _message_id, _model_id, payload):
        return {"payload": copy.deepcopy(payload)}

    async def persist(rows):
        from open_webui_openrouter_pipe import (
            generate_item_id,
        )

        ulids = [generate_item_id() for _ in rows]
        persisted.update(zip(ulids, (row["payload"] for row in rows)))
        return ulids

    monkeypatch.setattr(pipe._artifact_store, "_make_db_row", make_row)
    monkeypatch.setattr(pipe._artifact_store, "_db_persist", persist)

    async def capture(event):
        emitted.append(event)

    valves = pipe.valves.model_copy(update={"SHOW_TOOL_CARDS": cards})
    content = await pipe._streaming_handler._run_streaming_loop(
        ResponsesBody(
            model="test/model",
            input=[{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            stream=True,
        ),
        valves,
        capture,
        metadata={
            "model": {"id": "test"},
            "chat_id": "c1",
            "message_id": "m1",
            "tools": {"lookup": owui_lookup, "call_tool": call_tool_entry},
        },
        tools={
            "call_tool": {**call_tool_entry, "source": call_tool_entry},
            "lookup": {**owui_lookup, "source": owui_lookup},
        },
        session=cast(Any, object()),
        user_id="u1",
    )
    assert "all done." in str(content), content

    stored_call = next(
        payload for payload in persisted.values()
        if isinstance(payload, dict) and payload.get("type") == "function_call"
    )
    assert stored_call["name"] == "lookup", stored_call
    assert stored_call["provider_function"]["name"] == "call_tool", stored_call
    assert stored_call["provider_function"]["arguments"] == _CALL_TOOL_ARGUMENTS, stored_call

    def _restored_call(replay: list[Any]) -> dict[str, Any]:
        found = [
            item for item in replay
            if isinstance(item, dict) and item.get("type") == "function_call" and item.get("call_id") == "call_A"
        ]
        assert len(found) == 1, replay
        return found[0]

    replay_cards = await _stage_b(pipe, valves, content, persisted, recorded=_recorded_output(emitted))
    replay_rows = await _stage_b(pipe, valves, content, persisted)
    cards_item = _restored_call(replay_cards)
    rows_item = _restored_call(replay_rows)
    for item in (cards_item, rows_item):
        assert item["name"] == "call_tool", item
        assert item["arguments"] == _CALL_TOOL_ARGUMENTS, item
        assert "provider_function" not in item, item
    assert (cards_item["name"], cards_item["arguments"]) == (rows_item["name"], rows_item["arguments"])
