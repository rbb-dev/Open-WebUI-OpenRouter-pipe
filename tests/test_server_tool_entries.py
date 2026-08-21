"""Unit tests for _build_server_tool_entries — the server_tools metadata to
OpenRouter tools[] wire-format conversion."""
from __future__ import annotations

import logging
from typing import Any
from unittest.mock import AsyncMock, Mock

import aiohttp
import pytest

from conftest import Pipe
from open_webui_openrouter_pipe.requests.orchestrator import (
    RequestOrchestrator,
    _apply_server_tools_metadata,
    _build_server_tool_entries,
)
from open_webui_openrouter_pipe.api.transforms import ResponsesBody

_CHAT_VALVES = {
    "MAX_INPUT_IMAGES_PER_REQUEST": 0,
    "IMAGE_INPUT_SELECTION": "latest_user",
    "BASE64_MAX_SIZE_MB": 10,
    "IMAGE_UPLOAD_CHUNK_BYTES": 65536,
    "DIRECT_UPLOAD_FAILURE_TEMPLATE": "Direct upload failed: {reason}",
    "ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE": "Endpoint conflict: {reason}",
    "MODEL_RESTRICTED_TEMPLATE": "Model restricted: {restriction_reasons}",
    "OPENROUTER_ERROR_TEMPLATE": "Error: {detail}",
    "ZDR_ENFORCE": False,
    "ALLOW_USER_ZDR_OVERRIDE": False,
    "FUSION_BACKEND": "openrouter",
    "REASONING_EFFORT": "medium",
    "TASK_MODEL_REASONING_EFFORT": "low",
    "USE_MODEL_MAX_OUTPUT_TOKENS": False,
    "ENABLE_STRICT_TOOL_CALLING": False,
    "TOOL_EXECUTION_MODE": "Pipeline",
    "TOOL_OUTPUT_RETENTION_TURNS": 3,
    "ENABLE_ANTHROPIC_PROMPT_CACHING": False,
    "MODEL_ID": "",
    "FREE_MODEL_FILTER": "all",
    "TOOL_CALLING_FILTER": "all",
}


def test_basic_entry_shape():
    entries, _ = _build_server_tool_entries({"web_search": {"engine": "auto", "max_results": 5}})
    assert entries == [
        {"type": "openrouter:web_search", "parameters": {"engine": "auto", "max_results": 5}}
    ]


def test_empty_params_emits_no_parameters_key():
    entries, _ = _build_server_tool_entries({"datetime": {}})
    assert entries == [{"type": "openrouter:datetime"}]


def test_drops_none_and_empty_string():
    entries, _ = _build_server_tool_entries({"web_search": {"engine": "auto", "blank": "", "missing": None}})
    params = entries[0]["parameters"]
    assert params == {"engine": "auto"}


def test_preserves_zero_and_false_params():
    """Regression: the `!= 0` filter dropped temperature:0 and boolean False (False == 0).

    advisor/subagent accept temperature (0-2, where 0 is common) and boolean flags;
    these must survive the conversion verbatim.
    """
    entries, _ = _build_server_tool_entries(
        {"advisor": {"model": "x", "temperature": 0, "forward_transcript": False}}
    )
    params = entries[0]["parameters"]
    assert params["temperature"] == 0
    assert params["forward_transcript"] is False
    assert params["model"] == "x"


def test_skips_blank_tool_key():
    entries, _ = _build_server_tool_entries({"   ": {"x": 1}, "datetime": {}})
    assert entries == [{"type": "openrouter:datetime"}]


def test_chat_search_models_uses_experimental_type():
    """chat_search_models must map to the documented openrouter:experimental__search_models type."""
    entries, _ = _build_server_tool_entries({"chat_search_models": {}})
    assert entries == [{"type": "openrouter:experimental__search_models"}]


def test_chat_search_models_with_max_results():
    entries, _ = _build_server_tool_entries({"chat_search_models": {"max_results": 10}})
    assert entries == [
        {"type": "openrouter:experimental__search_models", "parameters": {"max_results": 10}}
    ]


def test_list_value_emits_one_entry_per_element():
    """A list value (multiple advisors) emits one tool entry per element."""
    entries, _ = _build_server_tool_entries(
        {"advisor": [{"name": "reviewer", "model": "x"}, {"name": "architect", "model": "y"}]}
    )
    assert entries == [
        {"type": "openrouter:advisor", "parameters": {"name": "reviewer", "model": "x"}},
        {"type": "openrouter:advisor", "parameters": {"name": "architect", "model": "y"}},
    ]


def test_apply_server_tools_metadata_injects_tools_and_stop_guard():
    """The integration seam: server_tools + stop_server_tools_when metadata reach responses_body."""
    body = ResponsesBody(model="x", input=[])
    meta = {
        "openrouter_pipe": {
            "server_tools": {"web_search": {"engine": "auto"}, "advisor": {"model": "m"}},
            "stop_server_tools_when": [{"type": "max_cost", "max_cost_in_dollars": 0.5}],
        }
    }
    _apply_server_tools_metadata(body, meta)
    types = [t["type"] for t in (body.tools or [])]
    assert "openrouter:web_search" in types
    assert "openrouter:advisor" in types
    assert body.stop_server_tools_when == [{"type": "max_cost", "max_cost_in_dollars": 0.5}]


def test_apply_server_tools_metadata_preserves_existing_tools():
    """Server tools are appended after any pre-existing (function) tools, not replacing them."""
    body = ResponsesBody(model="x", input=[], tools=[{"type": "function", "name": "f"}])
    _apply_server_tools_metadata(body, {"openrouter_pipe": {"server_tools": {"datetime": {}}}})
    types = [t.get("type") for t in (body.tools or [])]
    assert types == ["function", "openrouter:datetime"]


def test_apply_server_tools_metadata_noop_when_empty():
    body = ResponsesBody(model="x", input=[])
    _apply_server_tools_metadata(body, {})
    assert not body.tools
    assert getattr(body, "stop_server_tools_when", None) is None


def test_apply_server_tools_metadata_no_stop_guard_when_absent():
    body = ResponsesBody(model="x", input=[])
    _apply_server_tools_metadata(body, {"openrouter_pipe": {"server_tools": {"datetime": {}}}})
    assert getattr(body, "stop_server_tools_when", None) is None


@pytest.mark.parametrize(
    ("size", "ratio", "shape"),
    [("1024x1024", "16:9", "1:1"), ("1920x1080", "1:1", "16:9")],
)
@pytest.mark.asyncio
async def test_a_size_superseding_a_ratio_on_the_image_tool_is_told_to_the_user(
    monkeypatch, size, ratio, shape
):
    """The ratio the pipe removes from the image tool reaches the chat, not just the wire.

    Both controls are drawn together on 24 of the 40 recorded contracts, so choosing
    exact pixels and a shape that disagrees with them is an ordinary thing to do. The
    pixels win and the shape is withheld; a user who is not told sees a picture in the
    wrong shape with nothing to explain it.

    The assertion is on the notification the emitter received, because that is the only
    place the user reads. Two rows whose pixels imply DIFFERENT shapes, so a fixed
    string cannot satisfy both -- the second row keeps the ratio the first row drops.
    """
    pipe = Pipe()
    try:
        orchestrator = RequestOrchestrator(pipe, logging.getLogger("test_supersede"))
        events: list[dict[str, Any]] = []

        async def emitter(event):
            events.append(event)

        pipe._artifact_store._db_fetch = AsyncMock(return_value=None)
        pipe._ensure_reasoning_config_manager()._apply_reasoning_preferences = Mock()
        pipe._ensure_reasoning_config_manager()._apply_gemini_thinking_config = Mock()
        pipe._ensure_tool_executor()._build_direct_tool_server_registry = Mock(
            return_value=({}, [])
        )
        pipe._streaming_handler._select_llm_endpoint_with_forced = Mock(
            return_value=("chat_completions", False)
        )
        pipe._streaming_handler._run_streaming_loop = AsyncMock(return_value="drawn")

        valves = Mock()
        for name, value in _CHAT_VALVES.items():
            setattr(valves, name, value)

        await orchestrator.process_request(
            body={"model": "openai/gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            __user__={"id": "user1"},
            __request__=None,
            __event_emitter__=emitter,
            __event_call__=None,
            __metadata__={
                "openrouter_pipe": {
                    "server_tools": {
                        "image_generation": {
                            "model": "openai/gpt-image-2",
                            "size": size,
                            "aspect_ratio": ratio,
                        }
                    }
                }
            },
            __tools__=None,
            __task__=None,
            __task_body__=None,
            valves=valves,
            session=AsyncMock(spec=aiohttp.ClientSession),
            openwebui_model_id="openai/gpt-4o",
            pipe_identifier="test-pipe",
            allowlist_norm_ids={"openai/gpt-4o"},
            enforced_norm_ids=set(),
            catalog_norm_ids=set(),
            features={},
        )
    finally:
        await pipe.close()

    told = [
        str((event.get("data") or {}).get("content", ""))
        for event in events
        if event.get("type") == "notification"
    ]
    assert any(ratio in note and "was not sent" in note for note in told), (
        f"aspect_ratio={ratio!r} was removed because size={size!r} is {shape}, and the "
        f"user was never told; the notifications were {told!r}"
    )
