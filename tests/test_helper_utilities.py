import json

import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import (
    EncryptedStr,
    ModelFamily,
    OpenRouterAPIError,
    Pipe,
    ResponsesBody,
    _build_error_template_values,
    _classify_function_call_artifacts,
    _normalize_persisted_item,
    _pretty_json,
    _render_error_template,
    _strictify_schema,
    _template_value_present,
    build_tools,
    contains_marker,
    generate_item_id,
    split_text_by_markers,
)


def test_render_error_template_hides_missing_sections():
    template = (
        "{{#if heading}}## {heading}{{/if}}\n"
        "{{#if error_id}}error: {error_id}{{/if}}\n"
        "{{#if optional}}should not appear{{/if}}"
    )
    content = _render_error_template(
        template,
        {"heading": "Oops", "error_id": "E-123", "optional": ""},
    )
    assert "Oops" in content
    assert "E-123" in content
    assert "should not appear" not in content


def test_render_error_template_uses_default_template():
    rendered = _render_error_template(
        "",
        {"heading": "Fallback", "error_id": "ERR-1"},
    )
    assert "Fallback" in rendered
    assert "ERR-1" in rendered


def test_build_error_template_values_reports_model_limits():
    error = OpenRouterAPIError(
        status=400,
        reason="Bad Request",
        provider="Demo",
        openrouter_message='Shorten the prompt or use the "middle-out" transform.',
        metadata={"rate_limit_type": "minute"},
        requested_model="demo",
    )
    values = _build_error_template_values(
        error,
        heading="Demo",
        diagnostics=["- diag"],
        metrics={"context_limit": 16000, "max_output_tokens": 4096},
        model_identifier="demo",
        normalized_model_id="demo",
        api_model_id="demo",
    )
    assert values["include_model_limits"] is True
    assert values["context_limit_tokens"] == "16,000"
    assert "minute" in values["rate_limit_type"]


def test_template_utils_cover_edge_cases():
    assert _template_value_present("text") is True
    assert _template_value_present("") is False
    assert _template_value_present([1, 2]) is True
    assert _template_value_present([]) is False

    class NotSerializable:
        pass

    value = {NotSerializable()}
    result = _pretty_json(value)
    assert "NotSerializable" in result


def test_encrypted_str_roundtrip(monkeypatch):
    monkeypatch.setenv("WEBUI_SECRET_KEY", "super-secret-key")
    cipher = EncryptedStr.encrypt("token-value")
    assert cipher.startswith("encrypted:")
    assert EncryptedStr.decrypt(cipher) == "token-value"


def test_strictify_schema_enforces_required_and_nullability():
    schema = {
        "type": "object",
        "properties": {
            "required": {"type": "string"},
            "maybe": {"type": "integer"},
            "child": {
                "type": "object",
                "properties": {"flag": {"type": "boolean"}},
            },
        },
        "required": ["required"],
    }
    strict = _strictify_schema(schema)
    assert set(strict["required"]) == {"required", "maybe", "child"}
    assert strict["properties"]["required"]["type"] == "string"
    assert set(strict["properties"]["maybe"]["type"]) == {"integer", "null"}
    child_props = strict["properties"]["child"]["properties"]
    assert set(child_props["flag"]["type"]) == {"boolean", "null"}
    assert strict["properties"]["child"]["additionalProperties"] is False


def test_build_tools_combines_registry_and_extras(monkeypatch):
    responses_body = ResponsesBody(model="demo/tool", input=[])
    valves = Pipe.Valves(ENABLE_STRICT_TOOL_CALLING=True)

    def fake_supports(cls, feature, _model_id):
        return feature == "function_calling"

    monkeypatch.setattr(ModelFamily, "supports", classmethod(fake_supports))

    registry = {
        "alpha": {
            "spec": {
                "name": "alpha",
                "description": "alpha tool",
                "parameters": {"type": "object", "properties": {"foo": {"type": "string"}}},
            }
        }
    }
    extra_tools = [
        {"type": "function", "name": "alpha", "parameters": {"type": "object", "properties": {}}},
        {"type": "function", "name": "beta", "parameters": {"type": "object", "properties": {}}},
    ]

    tools = build_tools(responses_body, valves, __tools__=registry, extra_tools=extra_tools)

    function_names = [tool["name"] for tool in tools if tool["type"] == "function"]
    assert "beta" in function_names
    assert "alpha" in function_names  # dedup keeps last occurrence
    alpha_tool = next(tool for tool in tools if tool["type"] == "function" and tool["name"] == "alpha")
    assert alpha_tool["parameters"]["properties"] == {}


def test_tool_output_clamps_failed_status():
    pipe = Pipe()
    output = pipe._build_tool_output(
        {"call_id": "call-1"},
        "Tool not found",
        status="failed",
    )
    assert output["type"] == "function_call_output"
    assert output["status"] == "completed"


def test_normalize_persisted_items_and_classifier():
    call = _normalize_persisted_item({"type": "function_call", "name": "fetch", "arguments": {"foo": 1}})
    assert call is not None
    assert call["call_id"]
    assert json.loads(call["arguments"])["foo"] == 1

    output = _normalize_persisted_item(
        {"type": "function_call_output", "call_id": "abc", "output": {"data": 42}}
    )
    assert output is not None
    assert output["output"] == str({"data": 42})

    reasoning = _normalize_persisted_item({"type": "reasoning", "content": "step"})
    assert reasoning is not None
    assert reasoning["content"][0]["text"] == "step"

    artifacts = {
        "a": {"type": "function_call", "call_id": "shared"},
        "b": {"type": "function_call_output", "call_id": "shared"},
        "c": {"type": "function_call", "call_id": "orphan"},
        "d": {"type": "function_call_output", "call_id": "lonely-output"},
    }
    valid, orphan_calls, orphan_outputs = _classify_function_call_artifacts(artifacts)
    assert valid == {"shared"}
    assert orphan_calls == {"orphan"}
    assert orphan_outputs == {"lonely-output"}


def test_marker_helpers_and_ulids():
    ulid = generate_item_id()
    marker_line = f"[{ulid}]: #"
    text = f"prefix\n{marker_line}\npostfix"
    assert contains_marker(text) is True
    segments = split_text_by_markers(text)
    assert any(segment.get("marker") == ulid for segment in segments if segment["type"] == "marker")
    ids = {generate_item_id() for _ in range(10)}
    assert len(ids) == 10
