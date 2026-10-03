"""A tool the request itself brought goes back to whoever sent it.

Every request reaches the model with the tools it can actually use, and a function tool the pipe has nothing to run
behind goes back whole after exactly one upstream request: to the API caller that sent it, or to Open WebUI when the
request is one Open WebUI can pause. A name nobody offered is answered "Tool not found" inside the loop instead, and a
caller's schema is forwarded untouched.

The harness below drives one chat or API request through `Pipe.pipe()` with the model faked at the transport seam only,
so everything between Open WebUI's call and the HTTP request runs for real.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Iterator
from typing import Any, cast

import pytest

import open_webui_openrouter_pipe.pipe as pipe_mod
from open_webui_openrouter_pipe import EncryptedStr, Pipe, generate_item_id
from open_webui_openrouter_pipe.integrations import image_catalog, video_catalog
from open_webui_openrouter_pipe.models.registry import ModelFamily

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


async def run(pipe: Pipe, *, mode: str, stream: bool, metadata: dict[str, Any] | None = None,
              body_tools: list[dict[str, Any]] | None = None, owui_tools: dict[str, Any] | None = None,
              loops: int = 4, params: dict[str, Any] | None = None, messages: list[dict[str, Any]] | None = None,
              body_extra: dict[str, Any] | None = None,
              user: dict[str, Any] | None = None,
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
    result: Any = await pipe.pipe(
        body=body, __user__=user if user is not None else {"id": "user-1", "role": "user"},
        __request__=None,
        __event_emitter__=emitter if meta.get("chat_id") else None, __event_call__=None,
        __metadata__=meta, __tools__=owui_tools,
    )
    if hasattr(result, "__aiter__"):
        chunks: list[Any] = []
        async for chunk in cast(Any, result):
            chunks.append(chunk)
        return chunks, events
    return result, events


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


# --- B131: the hand-back budget is a reply's own ----------------------------------------------------------------------


CAP = 2
def always_hands_back(n: int, _request: dict[str, Any]):
    """A turn the model answers with a tool the pipe has nothing to run behind, so it hands back."""
    return calls_round(call("filter_schema", {"q": "x"}, f"call-h{n}"))


def _probe(pipe: Pipe, monkeypatch) -> list[dict[str, Any]]:
    """Install the hand-back script and hand back the request list the tests bill against."""
    return install(pipe, monkeypatch, always_hands_back)
