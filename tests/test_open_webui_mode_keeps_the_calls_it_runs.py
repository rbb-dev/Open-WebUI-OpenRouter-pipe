"""In Open-WebUI tool mode, Open WebUI runs the model's tool calls itself, so it must keep them.

Open WebUI builds its own `function_call` items from the `delta.tool_calls` chunks the pipe streams. A
`response.completed` whose `output` is not empty REPLACES Open WebUI's whole list, and after the stream Open WebUI only
updates the call items it still finds -- it never adds a missing one back. Its re-call of the pipe is built from that
list, and a result whose call is missing is dropped. So a response that hands calls to Open WebUI must not publish a
closing record; only the response that ends the reply does.
"""

from __future__ import annotations

import __future__
import ast
import copy
import json
import sysconfig
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody, generate_item_id
from tests.test_continue_stores_once import _open_webui_convert_output_to_messages
from tests.test_reasoning_skeleton_replay import MODEL, _stage_a, _valves


def _open_webui_streaming_handler() -> dict[str, Any]:
    """Open WebUI's own `handle_responses_streaming_event`, compiled from the installed release."""
    source = Path(sysconfig.get_paths()["purelib"]) / "open_webui" / "utils" / "middleware.py"
    text = source.read_text(encoding="utf-8")
    wanted = {"handle_responses_streaming_event", "deep_merge", "output_id"}
    segments = [ast.get_source_segment(text, node) or "" for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace: dict[str, Any] = {"uuid4": uuid4}
    exec(compile("\n\n".join(segments), str(source), "exec", flags=__future__.annotations.compiler_flag,
                 dont_inherit=True), namespace)
    return namespace


def _append_text(output: list[dict[str, Any]], value: str, output_id) -> None:
    last = output[-1] if output else None
    inside_tag_block = (
        last is not None and last.get("status") == "in_progress"
        and (last.get("attributes") or {}).get("type") != "reasoning_content"
        and (last.get("type") == "reasoning" or (last.get("type") == "message" and last.get("_tag_type") is not None))
    )
    if not inside_tag_block and (not output or output[-1].get("type") != "message"):
        output.append({"type": "message", "id": output_id("msg"), "status": "in_progress", "role": "assistant",
                       "content": [{"type": "output_text", "text": ""}]})
    parts = output[-1].get("content") or []
    if parts and parts[-1].get("type") == "output_text":
        parts[-1]["text"] += value
    else:
        output[-1]["content"] = [{"type": "output_text", "text": value}]


def _open_webui_backend(emitted: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Open WebUI 0.11.4's backend fed the pipe's events: text through its chunk branch, tool calls through its
    `delta.tool_calls` branch, `response.*` through its compiled handler, then its after-stream call update."""
    handler = _open_webui_streaming_handler()
    output: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for event in emitted:
        kind = event.get("type") or ""
        if kind == "chat:message:delta":
            value = (event.get("data") or {}).get("content") or ""
            if value:
                _append_text(output, value, handler["output_id"])
        elif kind == "chat:tool_calls":
            for delta in copy.deepcopy((event.get("data") or {}).get("tool_calls") or []):
                current = next((tc for tc in tool_calls if tc.get("index") == delta.get("index")), None)
                if current is None:
                    delta.setdefault("function", {})
                    delta["function"].setdefault("name", "")
                    delta["id"] = delta.get("id") or handler["output_id"]("fc")
                    delta["function"].setdefault("arguments", "")
                    tool_calls.append(delta)
                else:
                    function = delta.get("function") or {}
                    if function.get("name"):
                        current["function"]["name"] = function["name"]
                    if function.get("arguments") is not None:
                        current["function"]["arguments"] = current["function"].get("arguments", "") + function["arguments"]
            known = {it.get("call_id") for it in output if it.get("type") == "function_call"}
            for tc in tool_calls:
                if tc["id"] not in known:
                    output.append({"type": "function_call", "id": tc["id"], "call_id": tc["id"],
                                   "name": tc["function"].get("name", ""),
                                   "arguments": tc["function"].get("arguments", ""), "status": "in_progress"})
        elif kind.startswith("response."):
            handled = handler["handle_responses_streaming_event"](copy.deepcopy(event), output)
            if handled is not None:
                output = handled[0]
    for tc in tool_calls:
        for item in output:
            if item.get("type") == "function_call" and item.get("call_id") == tc["id"]:
                item["arguments"] = tc["function"].get("arguments", "{}")
                item["status"] = "completed"
                break
    return output
