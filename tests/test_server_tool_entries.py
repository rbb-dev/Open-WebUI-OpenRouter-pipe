"""Unit tests for _build_server_tool_entries — the server_tools metadata to
OpenRouter tools[] wire-format conversion."""
from __future__ import annotations

from open_webui_openrouter_pipe.requests.orchestrator import (
    _apply_server_tools_metadata,
    _build_server_tool_entries,
)
from open_webui_openrouter_pipe.api.transforms import ResponsesBody


def test_basic_entry_shape():
    entries = _build_server_tool_entries({"web_search": {"engine": "auto", "max_results": 5}})
    assert entries == [
        {"type": "openrouter:web_search", "parameters": {"engine": "auto", "max_results": 5}}
    ]


def test_empty_params_emits_no_parameters_key():
    entries = _build_server_tool_entries({"datetime": {}})
    assert entries == [{"type": "openrouter:datetime"}]


def test_drops_none_and_empty_string():
    entries = _build_server_tool_entries({"web_search": {"engine": "auto", "blank": "", "missing": None}})
    params = entries[0]["parameters"]
    assert params == {"engine": "auto"}


def test_preserves_zero_and_false_params():
    """Regression: the `!= 0` filter dropped temperature:0 and boolean False (False == 0).

    advisor/subagent accept temperature (0-2, where 0 is common) and boolean flags;
    these must survive the conversion verbatim.
    """
    entries = _build_server_tool_entries(
        {"advisor": {"model": "x", "temperature": 0, "forward_transcript": False}}
    )
    params = entries[0]["parameters"]
    assert params["temperature"] == 0
    assert params["forward_transcript"] is False
    assert params["model"] == "x"


def test_skips_blank_tool_key():
    entries = _build_server_tool_entries({"   ": {"x": 1}, "datetime": {}})
    assert entries == [{"type": "openrouter:datetime"}]


def test_chat_search_models_uses_experimental_type():
    """chat_search_models must map to the documented openrouter:experimental__search_models type."""
    entries = _build_server_tool_entries({"chat_search_models": {}})
    assert entries == [{"type": "openrouter:experimental__search_models"}]


def test_chat_search_models_with_max_results():
    entries = _build_server_tool_entries({"chat_search_models": {"max_results": 10}})
    assert entries == [
        {"type": "openrouter:experimental__search_models", "parameters": {"max_results": 10}}
    ]


def test_list_value_emits_one_entry_per_element():
    """A list value (multiple advisors) emits one tool entry per element."""
    entries = _build_server_tool_entries(
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
