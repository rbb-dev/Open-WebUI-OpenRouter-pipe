"""Unit tests for _build_server_tool_entries — the server_tools metadata to
OpenRouter tools[] wire-format conversion."""
from __future__ import annotations

from open_webui_openrouter_pipe.requests.orchestrator import _build_server_tool_entries


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
