import json

from open_webui_openrouter_pipe import (
    _classify_function_call_artifacts,
    _dedupe_tools,
    _normalize_persisted_item,
)


def test_normalize_function_call_serializes_arguments_and_ids():
    item = {
        "type": "function_call",
        "name": "lookup",
        "arguments": {"city": "Lisbon", "units": "metric"},
    }

    normalized = _normalize_persisted_item(item)
    assert normalized is not None

    assert normalized["type"] == "function_call"
    assert json.loads(normalized["arguments"]) == {"city": "Lisbon", "units": "metric"}
    assert normalized["status"] == "completed"
    assert normalized["call_id"]
    assert normalized["id"]


def test_normalize_function_call_output_assigns_defaults():
    item = {"type": "function_call_output", "output": {"ok": True}}

    normalized = _normalize_persisted_item(item)
    assert normalized is not None

    assert isinstance(normalized["output"], str)
    assert normalized["call_id"]
    assert normalized["status"] == "completed"


def test_normalize_reasoning_and_file_search_payloads():
    reasoning = _normalize_persisted_item(
        {"type": "reasoning", "content": "Chain", "summary": "Done"}
    )
    assert reasoning is not None
    file_search = _normalize_persisted_item({"type": "file_search_call", "queries": "bad"})
    assert file_search is not None
    web_search = _normalize_persisted_item({"type": "web_search_call", "action": "invalid"})
    assert web_search is not None

    assert reasoning["content"] == [{"type": "reasoning_text", "text": "Chain"}]
    assert reasoning["summary"] == ["Done"]
    assert file_search["queries"] == []
    assert web_search["action"] == {}


def test_classify_function_call_artifacts_identifies_orphans():
    artifacts = {
        "a": {"type": "function_call", "call_id": "call-1"},
        "b": {"type": "function_call_output", "call_id": "call-1"},
        "c": {"type": "function_call", "call_id": "call-2"},
        "d": {"type": "function_call_output", "call_id": "call-3"},
        "e": {"type": "function_call_output", "call_id": "   "},
    }

    valid, orphaned_calls, orphaned_outputs = _classify_function_call_artifacts(artifacts)

    assert valid == {"call-1"}
    assert orphaned_calls == {"call-2"}
    assert orphaned_outputs == {"call-3"}


def test_dedupe_tools_prefers_latest_definition():
    tools = [
        {"type": "function", "name": "search", "parameters": {"type": "object", "properties": {}}},
        {
            "type": "function",
            "name": "search",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        },
        {"type": "code_interpreter", "mode": "safe"},
        {"type": "code_interpreter", "mode": "fast"},
        "skip-me",
    ]

    deduped = _dedupe_tools(tools)

    assert len(deduped) == 2
    assert deduped[0]["parameters"]["properties"] == {"query": {"type": "string"}}
    assert deduped[1]["mode"] == "fast"
