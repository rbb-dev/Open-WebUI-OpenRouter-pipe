from __future__ import annotations

from openrouter_responses_pipe.openrouter_responses_pipe import _strictify_schema


def test_strictify_preserves_required_fields():
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["path"],
    }

    strict = _strictify_schema(schema)

    assert strict["required"] == ["path"]
    assert strict["properties"]["path"]["type"] == "string"
    assert strict["properties"]["timeout"]["type"] == ["integer", "null"]


def test_strictify_keeps_optional_fields_optional():
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        # No required list → everything optional
    }

    strict = _strictify_schema(schema)

    assert strict["required"] == []
    assert strict["properties"]["query"]["type"] == ["string", "null"]
    assert strict["properties"]["limit"]["type"] == ["integer", "null"]
