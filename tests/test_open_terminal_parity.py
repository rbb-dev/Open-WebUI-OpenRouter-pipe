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
import importlib.metadata
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Optional, cast

import pytest

from tests.test_tools import _as_open_webui_resolves_them

FILE = {"exists": True, "path": "/home/user/t.txt", "name": "t.txt"}
HEADERS = {"Content-Type": "application/json"}
CHAT = {"chat_id": "chat-1", "message_id": "m1", "session_id": "s1", "terminal_id": "T1"}


def _real_owui(*names: str) -> list[Any]:
    """Open WebUI's own functions, compiled from the installed `utils/middleware.py` together with the module-level
    helpers and constants they call, which the next release may add to (0.11.4 added two)."""
    path = Path(str(importlib.metadata.distribution("open-webui").locate_file("open_webui/utils/middleware.py")))
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
    from open_webui_openrouter_pipe import _PipeJob

    calls = [{"type": "function_call", "call_id": "c1", "name": spelled or exposed, "arguments": json.dumps(args)}]
    emitted: list[dict[str, Any]] = []

    async def emitter(event):
        emitted.append(event)

    valves = valves or pipe.Valves()
    await pipe._ensure_concurrency_controls(valves)
    holder: dict[str, Any] = {}

    async def handle(*_a, **_k):
        if card_carries_the_result:
            pipe._TOOL_CONTEXT.get().carded_calls = {"c1"}
        holder["outputs"] = await pipe._ensure_tool_executor()._execute_function_calls(calls, {exposed: entry})
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
