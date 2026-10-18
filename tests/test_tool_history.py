"""The tool-history search: three pipe-native tools that search this chat's earlier tool calls.

The harness drives one chat through `Pipe.pipe()` with the model faked at the transport seam only, so the offer, the
registry, the executor and the transform run for real; the same pattern as the hand-back tests.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any, cast

import pytest

import open_webui_openrouter_pipe.pipe as pipe_mod
from open_webui_openrouter_pipe import EncryptedStr, Pipe, generate_item_id
from open_webui_openrouter_pipe.integrations import image_catalog, video_catalog
from open_webui_openrouter_pipe.models.registry import ModelFamily

MODEL = "test/model"
NORM = "test.model"
CHAT = {"chat_id": "chat-1", "message_id": "message-1", "session_id": "session-1"}
CATALOG = {"data": [{
    "id": NORM, "name": "Test", "norm_id": NORM,
    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
    "supported_parameters": ["tools", "tool_choice", "temperature", "max_tokens"],
}]}

ROW_WITH_TOOLS = {"features": {"function_calling"},
                  "supported_parameters": frozenset({"tools", "tool_choice", "temperature", "max_tokens"})}

HISTORY_NAMES = {"search_tool_history", "list_tool_history", "view_tool_call"}


def call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
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


def outputs_in(request: dict[str, Any]) -> list[str]:
    found: list[str] = []
    for item in request.get("input") or []:
        if isinstance(item, dict) and item.get("type") == "function_call_output":
            out = item.get("output")
            if isinstance(out, list):
                out = "".join(p.get("text", "") for p in out if isinstance(p, dict))
            found.append(str(out))
    return found


def tool_names_in(request: dict[str, Any]) -> set[str]:
    return {
        tool["name"]
        for tool in request.get("tools") or []
        if isinstance(tool, dict) and tool.get("type") == "function" and isinstance(tool.get("name"), str)
    }


def install(pipe: Pipe, monkeypatch, script: Callable[[int, dict[str, Any]], list[dict[str, Any]]]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []

    async def model(self, session, request_body, **_kwargs):
        requests.append(copy.deepcopy(request_body))
        for event in script(len(requests), request_body):
            yield event

    async def loaded(*_a: Any, **_k: Any) -> None:
        return None

    async def persist(new_rows):
        return [generate_item_id() for _ in new_rows]

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
    ModelFamily.set_dynamic_specs({NORM: dict(ROW_WITH_TOOLS)})
    return requests


async def run(pipe: Pipe, *, messages: list[dict[str, Any]], script: Callable[[int, dict[str, Any]], list[dict[str, Any]]],
              monkeypatch=None, stream: bool = True, mode: str = "Pipeline",
              metadata: dict[str, Any] | None = None, body_extra: dict[str, Any] | None = None,
              **valve_updates: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    requests: list[dict[str, Any]] = []
    if monkeypatch is not None:
        requests = install(pipe, monkeypatch, script)
    valves = pipe.valves.model_copy(update={"TOOL_EXECUTION_MODE": mode, **valve_updates})
    valves.API_KEY = EncryptedStr("sk-test-key")
    pipe.valves = valves
    events: list[dict[str, Any]] = []

    async def emitter(event):
        events.append(event)

    meta = {"model": {"id": MODEL}, **(metadata if metadata is not None else CHAT)}
    body: dict[str, Any] = {
        "model": NORM,
        "stream": stream,
        "messages": messages,
    }
    body.update(body_extra or {})
    result: Any = await pipe.pipe(
        body=body, __user__={"id": "user-1", "role": "user"},
        __request__=None,
        __event_emitter__=emitter, __event_call__=None,
        __metadata__=meta, __tools__=None,
    )
    if hasattr(result, "__aiter__"):
        chunks: list[Any] = []
        async for chunk in cast(Any, result):
            chunks.append(chunk)
        return requests, events
    return requests, events


def history_chat(*, result: str = "total 12\nbackup-ok.txt", self_round: bool = False) -> list[dict[str, Any]]:
    first_assistant: dict[str, Any] = {
        "role": "assistant",
        "content": "Running the backup check.",
        "tool_calls": [{"id": "call-9", "type": "function",
                        "function": {"name": "run_cmd", "arguments": json.dumps({"cmd": "ls /var/backups"})}}],
    }
    if self_round:
        first_assistant["content"] += "\n[a1b2c3d4e5f6g7h8i9j0]: #\nmid"
    second_assistant: dict[str, Any] = {"role": "assistant", "content": "The backup check finished."}
    if self_round:
        second_assistant["tool_calls"] = [
            {"id": "call-8", "type": "function",
             "function": {"name": "search_tool_history",
                          "arguments": json.dumps({"query": "backup-ok", "mode": "exact"})}}]
    return [
        {"role": "user", "content": "check the backup folder"},
        first_assistant,
        {"role": "tool", "tool_call_id": "call-9", "content": result},
        second_assistant,
        *([{"role": "tool", "tool_call_id": "call-8",
            "content": "[1.1] turn 1 (a while ago) - run_cmd"}] if self_round else []),
        {"role": "user", "content": "what did the backup check show?"},
    ]


def search_then_view_script(n: int, _request: dict[str, Any]) -> list[dict[str, Any]]:
    if n == 1:
        return calls_round(call("search_tool_history", {"query": "backup-ok", "mode": "exact"}, "call-s1"))
    if n == 2:
        return calls_round(call("view_tool_call", {"id": "1.1"}, "call-v1"))
    return answer_round("The backup check listed backup-ok.txt.")


@pytest.mark.asyncio
async def test_history_round_from_search_to_view_with_results_off(pipe_instance_async, monkeypatch):
    requests, _events = await run(
        pipe_instance_async, monkeypatch=monkeypatch, messages=history_chat(), script=search_then_view_script,
    )
    assert len(requests) >= 3
    first_tools = tool_names_in(requests[0])
    assert HISTORY_NAMES <= first_tools
    search_output = "\n".join(outputs_in(requests[1]))
    assert "[1.1] turn 1" in search_output
    assert "backup-ok.txt" in search_output
    view_output = "\n".join(outputs_in(requests[2]))
    assert "total 12" in view_output
    note = "\n".join(outputs_in(requests[1]))
    assert "view_tool_call 1.1 returns it" in note
    assert "[This call succeeded. Its result is not kept for later turns; view_tool_call 1.1 returns it.]" in note


@pytest.mark.asyncio
async def test_switching_the_feature_off_restores_the_previous_request(pipe_instance_async, monkeypatch):
    requests, _events = await run(
        pipe_instance_async, monkeypatch=monkeypatch, messages=history_chat(), script=search_then_view_script,
        TOOL_HISTORY_SEARCH=False,
    )
    assert len(requests) >= 2
    assert not (HISTORY_NAMES & tool_names_in(requests[0]))
    note = "\n".join(outputs_in(requests[0]))
    assert "[This call succeeded. Its result is not kept for later turns.]" in note
    assert "view_tool_call" not in note


@pytest.mark.asyncio
async def test_offered_only_where_the_pipe_runs_the_round(pipe_instance_async, monkeypatch):
    def script(n, _request):
        if n == 1:
            return calls_round(call("search_tool_history", {"query": "backup-ok", "mode": "exact"}, "call-s1"))
        return answer_round("listed.")

    requests, _events = await run(
        pipe_instance_async, monkeypatch=monkeypatch, messages=history_chat(), script=script,
        mode="Open-WebUI", stream=True,
    )
    assert requests
    assert not (HISTORY_NAMES & tool_names_in(requests[0]))
    requests, _events = await run(
        pipe_instance_async, monkeypatch=monkeypatch, messages=history_chat(), script=script,
        mode="Open-WebUI", stream=False,
    )
    assert requests
    assert HISTORY_NAMES <= tool_names_in(requests[0])


@pytest.mark.asyncio
async def test_not_offered_when_the_request_leaves_nothing_out(pipe_instance_async, monkeypatch):
    def script(n, _request):
        return answer_round("fine.")

    requests, _events = await run(
        pipe_instance_async, monkeypatch=monkeypatch,
        messages=[
            {"role": "user", "content": "hello there"},
            {"role": "assistant", "content": "Hello! How can I help?"},
            {"role": "user", "content": "what else?"},
        ],
        script=script,
    )
    assert requests
    assert not (HISTORY_NAMES & tool_names_in(requests[0]))
    requests, _events = await run(
        pipe_instance_async, monkeypatch=monkeypatch,
        messages=history_chat(), script=script, PERSIST_TOOL_RESULTS=True,
    )
    assert requests
    assert not (HISTORY_NAMES & tool_names_in(requests[0]))


@pytest.mark.asyncio
async def test_with_results_kept_a_round_beyond_the_window_is_found_and_returned(pipe_instance_async, monkeypatch):
    def script(n, _request):
        if n == 1:
            return calls_round(call("search_tool_history", {"query": "backup-ok", "mode": "exact"}, "call-s1"))
        if n == 2:
            return calls_round(call("view_tool_call", {"id": "1.1"}, "call-v1"))
        return answer_round("The earlier check listed backup-ok.txt.")

    requests, _events = await run(
        pipe_instance_async, monkeypatch=monkeypatch,
        messages=history_chat(), script=script,
        PERSIST_TOOL_RESULTS=True, TOOL_TURNS_TO_REMEMBER=0,
    )
    assert len(requests) >= 3
    assert HISTORY_NAMES <= tool_names_in(requests[0])
    search_output = "\n".join(outputs_in(requests[1]))
    assert "[1.1] turn 1" in search_output and "backup-ok.txt" in search_output
    view_output = "\n".join(outputs_in(requests[2]))
    assert "total 12" in view_output


@pytest.mark.asyncio
async def test_the_history_tools_own_rounds_are_never_found(pipe_instance_async, monkeypatch):
    def script(n, _request):
        if n == 1:
            return calls_round(call("search_tool_history", {"query": "RR-marker-unique-77", "mode": "exact"}, "call-s1"))
        return answer_round("done.")

    requests, _events = await run(
        pipe_instance_async, monkeypatch=monkeypatch,
        messages=history_chat(self_round=True, result="RR-marker-unique-77 in the log"), script=script,
    )
    assert len(requests) >= 2
    search_output = "\n".join(outputs_in(requests[1]))
    assert "[1.1]" in search_output
    assert "[1.2]" not in search_output


@pytest.mark.asyncio
async def test_view_pages_join_to_the_stored_text():
    long_result = "\n".join(f"backup line {i:04d} of the nightly log" for i in range(120))
    messages = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": "ok", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "run_cmd", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": long_result},
        {"role": "assistant", "content": "done"},
    ]
    from open_webui_openrouter_pipe.tools import tool_history as th

    entries = th.history_entries(
        None, messages=messages, chat_id=None, reply_id=None, window_turns=0,
        notes_in_view=True, ask_user_names=frozenset({"ask_user"}),
    )
    view = entries["view_tool_call"]["callable"]

    def body_of(page: str) -> str:
        lines = page.split("\n")
        start = next(i for i, line in enumerate(lines) if line.startswith("result: chars ")) + 1
        end = next(i for i in range(start, len(lines)) if lines[i].startswith("This is what the tool returned"))
        return "\n".join(lines[start:end])

    first = await view(id="1.1", offset=0, max_chars=500)
    second = await view(id="1.1", offset=500, max_chars=500)
    assert "chars 0-500 of " in first
    assert "chars 500-1," in second or "chars 500-1000" in second
    assert body_of(first) + body_of(second) == long_result[:1000]


@pytest.mark.asyncio
async def test_a_round_stored_only_by_the_pipe_is_found_through_the_same_loader(pipe_instance_async, monkeypatch):
    marker_a = generate_item_id()
    marker_b = generate_item_id()
    stored = {
        marker_a: {"type": "function_call", "call_id": "row-call-1", "name": "device_query",
                   "arguments": json.dumps({"host": "inv-backup-7"})},
        marker_b: {"type": "function_call_output", "call_id": "row-call-1",
                   "output": "row-only result RR-9k7q", "status": "completed"},
    }

    async def fake_db_fetch(chat_id, message_id, item_ids, *, reply_id=None, with_producers=False):
        return {uid: stored[uid] for uid in item_ids if uid in stored}

    monkeypatch.setattr(pipe_instance_async._artifact_store, "_db_fetch", fake_db_fetch)
    messages = history_chat(result="RR-marker-unique-77 in the log")
    messages.insert(4, {"role": "assistant",
                        "content": f"also\n[{marker_a}]: #\n[{marker_b}]: #"})

    def script(n, _request):
        if n == 1:
            return calls_round(call("search_tool_history", {"query": "RR-9k7q", "mode": "exact"}, "call-s1"))
        if n == 2:
            return calls_round(call("view_tool_call", {"id": "1.2"}, "call-v2"))
        return answer_round("found it.")

    requests, _events = await run(
        pipe_instance_async, monkeypatch=monkeypatch, messages=messages, script=script,
    )
    assert len(requests) >= 3
    note = "\n".join(outputs_in(requests[0]))
    assert "[This call succeeded. Its result is not kept for later turns; view_tool_call 1.2 returns it.]" in note
    search_output = "\n".join(outputs_in(requests[1]))
    assert "RR-9k7q" in search_output
    assert "device_query" in search_output
    view_output = "\n".join(outputs_in(requests[2]))
    assert "row-only result RR-9k7q" in view_output


@pytest.mark.asyncio
async def test_the_request_models_builtin_tools_box_gates_the_offer(pipe_instance_async, monkeypatch):
    def script(n, _request):
        return answer_round("listed.")

    requests, _events = await run(
        pipe_instance_async, monkeypatch=monkeypatch, messages=history_chat(), script=script,
        metadata={**CHAT, "model": {"id": MODEL,
                                    "info": {"meta": {"capabilities": {"builtin_tools": False}}}}},
    )
    assert requests
    assert not (HISTORY_NAMES & tool_names_in(requests[0]))
    requests, _events = await run(
        pipe_instance_async, monkeypatch=monkeypatch, messages=history_chat(), script=script,
        metadata={**CHAT, "model": {"id": MODEL}},
    )
    assert requests
    assert HISTORY_NAMES <= tool_names_in(requests[0])


@pytest.mark.asyncio
async def test_offered_only_when_a_tool_round_is_left_out(pipe_instance_async, monkeypatch):
    def script(n, _request):
        return answer_round("noted.")

    thinking_marker_chat = [
        {"role": "user", "content": "think about the backup"},
        {"role": "assistant", "content": "Thinking it through.\n[%s]: #" % generate_item_id()},
        {"role": "user", "content": "and now?"},
    ]
    requests, _events = await run(
        pipe_instance_async, monkeypatch=monkeypatch, messages=thinking_marker_chat, script=script,
    )
    assert requests
    assert not (HISTORY_NAMES & tool_names_in(requests[0]))

    marker_a = generate_item_id()
    marker_b = generate_item_id()
    stored = {
        marker_a: {"type": "function_call", "call_id": "row-call-9", "name": "device_query",
                   "arguments": "{}"},
        marker_b: {"type": "function_call_output", "call_id": "row-call-9",
                   "output": "row result", "status": "completed"},
    }

    async def fake_db_fetch(chat_id, message_id, item_ids, *, reply_id=None, with_producers=False):
        return {uid: stored[uid] for uid in item_ids if uid in stored}

    monkeypatch.setattr(pipe_instance_async._artifact_store, "_db_fetch", fake_db_fetch)
    cards_off_withheld = [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": "ran.\n[%s]: #\n[%s]: #" % (marker_a, marker_b)},
        {"role": "user", "content": "what did it return?"},
    ]
    requests, _events = await run(
        pipe_instance_async, monkeypatch=monkeypatch, messages=cards_off_withheld, script=script,
    )
    assert requests
    assert HISTORY_NAMES <= tool_names_in(requests[0])
