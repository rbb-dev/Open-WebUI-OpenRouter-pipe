"""Open Terminal behaves through the pipe as it does in Open WebUI's own tool loop.

Open WebUI's order for a terminal tool (`utils/middleware.py`, its native loop) is reshape -> process -> tell the chat:

    terminal_file_result = build_terminal_file_tool_result(name, params, result, tool, metadata)
    if terminal_file_result: tool_result = terminal_file_result
    tool_result, files, embeds = await process_tool_result(..., direct_tool, ...)
    await terminal_event_handler(name, params, tool_result, event_emitter)

Measured live on the dev box, 2026-09-21: before the pipe did the same, a file the model wrote in Pipeline mode existed
on disk and never appeared in the chat's file browser. These drive the real job -- queue, workers, context -- with
Open WebUI's OWN three functions compiled from the installed source, so a wrong argument, a wrong order or a swallowed
helper changes what the model reads or what the chat is told, and a test sees it.
"""
from __future__ import annotations

import ast
import asyncio
import json
import logging
import mimetypes
import os
import re
from typing import Any, Optional, cast

import pytest

from tests.package_sources import open_webui_package_dir
from tests.test_tools import _as_open_webui_resolves_them

FILE = {"exists": True, "path": "/home/user/t.txt", "name": "t.txt"}
HEADERS = {"Content-Type": "application/json"}
CHAT = {"chat_id": "chat-1", "message_id": "m1", "session_id": "s1", "terminal_id": "T1"}


def _real_owui(*names: str) -> list[Any]:
    """Open WebUI's own functions, compiled from the installed `utils/middleware.py` together with the module-level
    helpers and constants they call, which the next release may add to (0.11.4 added two)."""
    path = open_webui_package_dir() / "utils" / "middleware.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    from starlette.responses import HTMLResponse

    async def _no_upload(*_a, **_k):
        return "http://files/x"

    namespace: dict[str, Any] = {
        "json": json, "JSONCodec": json, "mimetypes": mimetypes, "os": os, "re": re, "Any": Any, "Optional": Optional,
        "HTMLResponse": HTMLResponse, "log": logging.getLogger("open-webui-compiled"),
        "get_file_url_from_base64": _no_upload,
    }
    defined: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined[node.name] = node
        elif isinstance(node, ast.Assign):
            defined.update((target.id, node) for target in node.targets if isinstance(target, ast.Name))
    missing = set(names) - defined.keys()
    assert not missing, missing
    wanted: set[str] = set()
    pending = list(names)
    while pending:
        name = pending.pop()
        if name in wanted or name in namespace:
            continue
        wanted.add(name)
        pending.extend(
            used.id for used in ast.walk(defined[name]) if isinstance(used, ast.Name) and used.id in defined
        )
    for node in tree.body:
        if node in {defined[name] for name in wanted}:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return [namespace[name] for name in names]


_DECLARED_ARGUMENTS: dict[str, dict[str, Any]] = {
    "display_file": {"path": {"type": "string"}, "inline": {"type": "boolean"}, "page": {"type": "integer"}},
    "write_file": {"path": {"type": "string"}, "content": {"type": "string"}},
    "run_command": {"command": {"type": "string"}},
    "search": {"q": {"type": "string"}},
}


def _entry(name: str, callable_: Any, *, origin: str | None = None, **extra: Any) -> dict[str, Any]:
    """A resolved tool whose spec declares the arguments the tool takes, as Open WebUI's specs do: Open Terminal's own,
    plus the `inline` and `page` Open WebUI adds to `display_file` (`add_terminal_display_file_inline_param`)."""
    origin = origin or name
    return {"callable": callable_, "origin_name": origin, "exposed_name": name,
            "spec": {"name": origin, "parameters": {"type": "object",
                                                    "properties": dict(_DECLARED_ARGUMENTS.get(origin, {}))}},
            **extra}


async def _drive(pipe, monkeypatch, *, exposed: str, entry: dict[str, Any], args: dict[str, Any],
                 metadata: dict[str, Any] | None = None, valves=None, card_carries_the_result: bool = False,
                 spelled: str | None = None):
    """One tool call through the real job. `card_carries_the_result` says the streaming loop published this call's card
    in this round, as it records for each call a card will hold (cards on, a streamed turn, not a Fusion member).

    `spelled` is the name the MODEL wrote, and it defaults to the name the registry is keyed by. Letting the two differ
    is the only way to reach a padded spelling on the real path: the registry keys tools by the advertised (trimmed)
    name and the lookup strips, so `"display_file "` and `"display_file"` name the same tool and reach the same code
    with different `call["name"]` text. Nothing else in production may see the difference -- a padded spelling is the
    model's own text, and Open WebUI's loop keeps it."""
    calls = [{"type": "function_call", "call_id": "c1", "name": spelled or exposed, "arguments": json.dumps(args)}]

    async def work(_emitter, _valves):
        if card_carries_the_result:
            pipe._TOOL_CONTEXT.get().carded_calls = {"c1"}
        return await pipe._ensure_tool_executor()._execute_function_calls(calls, {exposed: entry})

    return await _in_job(pipe, monkeypatch, work, metadata=metadata, valves=valves)


async def _in_job(pipe, monkeypatch, work, *, metadata: dict[str, Any] | None = None, valves=None):
    from open_webui_openrouter_pipe import _PipeJob

    emitted: list[dict[str, Any]] = []

    async def emitter(event):
        emitted.append(event)

    valves = valves or pipe.Valves()
    await pipe._ensure_concurrency_controls(valves)
    holder: dict[str, Any] = {}

    async def handle(*_a, **_k):
        holder["outputs"] = await work(emitter, valves)
        return "done"

    monkeypatch.setattr(pipe, "_handle_pipe_call", handle)
    job = _PipeJob(
        pipe=pipe, body={}, user={"id": "user-1"}, request=None, event_emitter=emitter, event_call=None,
        metadata=dict(CHAT if metadata is None else metadata), tools=None, task=None, task_body=None, valves=valves,
        future=asyncio.get_running_loop().create_future(),
    )
    await pipe._execute_pipe_job(job)
    assert job.future.result() == "done"
    return holder["outputs"], emitted


@pytest.mark.asyncio
async def test_a_file_shown_inline_is_reported_displayed_only_when_a_card_shows_it(monkeypatch):
    from open_webui_openrouter_pipe import Pipe
    import open_webui_openrouter_pipe.tools.tool_executor as tool_executor

    build, handler, process = _real_owui(
        "build_terminal_file_tool_result", "terminal_event_handler", "process_tool_result"
    )
    monkeypatch.setattr(tool_executor, "_owui_build_terminal_file_tool_result", build)
    monkeypatch.setattr(tool_executor, "_owui_terminal_event_handler", handler)
    monkeypatch.setattr(tool_executor, "_owui_process_tool_result", process)

    async def display_file(**_kwargs):
        return dict(FILE)

    entry = _entry("display_file", display_file, tool_id="terminal:T1")
    pipe = Pipe()
    try:
        seen = {}
        for carded in (False, True):
            outputs, emitted = await _drive(
                pipe, monkeypatch, exposed="display_file", entry=entry,
                args={"path": FILE["path"], "inline": True}, card_carries_the_result=carded,
            )
            seen[carded] = (
                json.loads(outputs[0]["output"]).get("displayed") is True,
                [event["type"] for event in emitted],
            )
    finally:
        await pipe.close()
    assert seen == {False: (False, ["terminal:display_file"]), True: (True, [])}


@pytest.mark.asyncio
async def test_with_cards_off_a_chat_file_runs_the_file_tool_and_keeps_its_card_where_a_terminal_path_has_none(
    monkeypatch,
):
    from open_webui_openrouter_pipe import EncryptedStr, Pipe, ResponsesBody
    from open_webui_openrouter_pipe.tools.tool_registry import _build_collision_safe_tool_specs_and_registry

    ran: list[str] = []

    async def terminal_display(**_kwargs):
        ran.append("terminal")
        return dict(FILE)

    async def file_display(**kwargs):
        ran.append("file")
        return json.dumps(
            {"type": "file", "source": "open_webui", "exists": True, "displayed": True, "path": kwargs["path"]}
        )

    spec = _entry("display_file", None)["spec"]
    file_tool = {"tool_id": "builtin:display_file", "type": "builtin", "callable": file_display, "spec": spec}
    terminal = {
        "tool_id": "terminal:T1", "type": "terminal", "callable": terminal_display, "spec": spec,
        "file_tool": file_tool,
    }
    _tools, registry, _map = _build_collision_safe_tool_specs_and_registry(
        request_tool_specs=None, owui_registry={"display_file": terminal}, direct_registry=None,
        builtin_registry=None, extra_tools=None, strictify=False, owui_tool_passthrough=False, logger=None,
    )
    paths = ["3f2504e0-4f89-41d3-9a0c-0305e82c3301", FILE["path"]]
    step = [0]

    async def provider(_self, _session, _request_body, **_kwargs):
        index = step[0]
        step[0] += 1
        if index < len(paths):
            call = {
                "type": "function_call", "call_id": f"call-{index}", "name": "display_file",
                "arguments": json.dumps({"path": paths[index]}), "status": "completed",
            }
            yield {"type": "response.output_item.done", "item": call}
            yield {"type": "response.completed", "response": {"output": [call], "usage": {}}}
        else:
            yield {"type": "response.output_text.delta", "delta": "done"}
            yield {"type": "response.completed", "response": {"output": [], "usage": {}}}

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", provider)
    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("sk-test-key")
    try:
        async def work(emitter, valves):
            return await pipe._streaming_handler._run_streaming_loop(
                ResponsesBody(model="test/model", input=[], stream=True),
                valves,
                emitter,
                metadata={"model": {"id": "test"}, **CHAT},
                tools=registry,
                session=cast(Any, object()),
                user_id="user-1",
            )

        _outputs, emitted = await _in_job(
            pipe, monkeypatch, work, valves=pipe.valves.model_copy(update={"SHOW_TOOL_CARDS": False})
        )
    finally:
        await pipe.close()
    carded = {
        (event["item"]["type"], event["item"]["call_id"])
        for event in emitted
        if event.get("type") == "response.output_item.added"
        and event["item"].get("type") in ("function_call", "function_call_output")
    }
    assert (ran, carded) == (["file", "terminal"], {("function_call", "call-0"), ("function_call_output", "call-0")})
