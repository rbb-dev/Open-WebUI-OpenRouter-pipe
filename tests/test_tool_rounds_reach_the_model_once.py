"""Every tool round reaches the next request exactly once, where it happened, whatever the card switch says.

A round can travel two ways: Open WebUI saves it in the message and rebuilds it into the next request (tool cards
on, a streamed turn), or the pipe keeps it in its own storage and replays it through a hidden marker. Round 33 found
the round arriving twice, arriving nowhere, and arriving in the wrong place, depending on the switch, the result
setting, the kind of chat and the reasoning path. Each test here pins one of those, driven through the real
streaming loop and, where Open WebUI rebuilds the message, through Open WebUI's own converter.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.test_reasoning_skeleton_replay import (
    _recorded_output,
    _stage_a,
    _stage_b,
    _valves,
)


def _label(item: dict[str, Any]) -> str:
    kind = item.get("type")
    if kind == "reasoning":
        texts = [part.get("text", "") for part in item.get("content") or [] if isinstance(part, dict)]
        return "R:" + "".join(texts)
    if kind == "function_call":
        return f"call:{item.get('call_id')}"
    if kind == "function_call_output":
        return f"out:{item.get('call_id')}"
    if kind == "message":
        return f"msg:{item.get('role')}"
    return str(kind)


# --- T235: with cards off, the pipe's copy is the model's record of the round, not a reasoning scaffold -------------

import asyncio  # noqa: E402
import json  # noqa: E402

from open_webui_openrouter_pipe.api.transforms import (  # noqa: E402
    _filter_openrouter_request,
    _responses_payload_to_chat_completions_payload,
)
from open_webui_openrouter_pipe.models.reasoning_config import ReasoningConfigManager  # noqa: E402
from tests.test_reasoning_skeleton_replay import (  # noqa: E402
    PARALLEL_WITH_FINAL_REASONING,
    SEQUENTIAL_TWO_ROUNDS,
    _outgoing_body,
    _shape,
)


def _open_webui_streaming_handler() -> dict[str, Any]:
    """Open WebUI's own `handle_responses_streaming_event`, `deep_merge` and `output_id`, compiled from the installed
    source into a namespace of their own."""
    import __future__
    import ast
    import sysconfig
    from pathlib import Path
    from uuid import uuid4

    source = Path(sysconfig.get_paths()["purelib"]) / "open_webui" / "utils" / "middleware.py"
    text = source.read_text(encoding="utf-8")
    wanted = {"handle_responses_streaming_event", "deep_merge", "output_id"}
    segments = [
        ast.get_source_segment(text, node) or ""
        for node in ast.parse(text).body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert len(segments) == len(wanted)
    code = "\n\n".join(segments)
    assert "existing.get('type') == item.get('type')" in code, (
        "the installed Open WebUI no longer matches a published item to an earlier one by type and call id"
    )
    namespace: dict[str, Any] = {"uuid4": uuid4}
    exec(compile(code, str(source), "exec", flags=__future__.annotations.compiler_flag, dont_inherit=True), namespace)
    return namespace


# --- T233: one copy of each round, whichever carriers exist -----------------------------------------------------------

from open_webui_openrouter_pipe import generate_item_id  # noqa: E402
from open_webui_openrouter_pipe.core import utils as core_utils  # noqa: E402
from open_webui_openrouter_pipe.core.utils import _serialize_marker  # noqa: E402
from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input  # noqa: E402
from tests.test_continue_stores_once import _open_webui_convert_output_to_messages  # noqa: E402
from tests.test_reasoning_skeleton_replay import MODEL, RESULT_CANARY, _has_consecutive_reasoning  # noqa: E402


# --- T237: an OpenRouter server tool's round is carried by exactly one thing ---------------------------------------

from tests.test_tool_record_survives_cards_off import _run_server_tool  # noqa: E402

# --- T243: a Continue must not glue its first words onto the message's last hidden marker ----------------------------

import copy  # noqa: E402
import re  # noqa: E402
from typing import cast  # noqa: E402

from open_webui_openrouter_pipe import Pipe, ResponsesBody  # noqa: E402
