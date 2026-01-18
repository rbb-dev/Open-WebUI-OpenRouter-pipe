from __future__ import annotations

from open_webui_openrouter_pipe import _strictify_schema


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

    assert strict["required"] == sorted(["path", "timeout"])
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

    assert strict["required"] == sorted(["query", "limit"])
    assert strict["properties"]["query"]["type"] == ["string", "null"]
    assert strict["properties"]["limit"]["type"] == ["integer", "null"]


def test_strictify_handles_nested_objects():
    schema = {
        "type": "object",
        "properties": {
            "filters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["status"],
            },
            "limit": {"type": "integer"},
        },
        "required": ["filters"],
    }

    strict = _strictify_schema(schema)

    assert strict["required"] == sorted(["filters", "limit"])
    assert strict["properties"]["limit"]["type"] == ["integer", "null"]

    nested = strict["properties"]["filters"]
    assert nested["required"] == ["status", "tags"]
    assert nested["properties"]["status"]["type"] == "string"
    assert nested["properties"]["tags"]["type"] == ["array", "null"]


# New tests for type inference (fixing empty schema bug)


def test_strictify_adds_type_to_empty_property():
    """Test that empty property schemas get default type 'object'"""
    schema = {
        "type": "object",
        "properties": {
            "session": {},  # Empty schema - missing type
            "user": {"type": "string"}
        }
    }
    strict = _strictify_schema(schema)

    # Should add default type "object"
    assert strict["properties"]["session"]["type"] == ["object", "null"]
    assert strict["properties"]["session"]["additionalProperties"] is False

    # Should preserve existing types
    assert "string" in strict["properties"]["user"]["type"]


def test_strictify_adds_type_to_schema_with_only_description():
    """Test that schemas with only description get default type"""
    schema = {
        "type": "object",
        "properties": {
            "metadata": {
                "description": "Additional metadata"
                # No type key
            }
        }
    }
    strict = _strictify_schema(schema)

    assert strict["properties"]["metadata"]["type"] == ["object", "null"]
    assert strict["properties"]["metadata"]["description"] == "Additional metadata"


def test_strictify_handles_nested_empty_schemas():
    """Test that nested empty schemas are handled correctly"""
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {
                    "inner": {}  # Nested empty
                }
            }
        },
        "required": ["outer"]
    }
    strict = _strictify_schema(schema)

    nested = strict["properties"]["outer"]["properties"]["inner"]
    assert "object" in nested["type"]


def test_strictify_adds_type_to_empty_items():
    """Test that empty items schemas get default type"""
    schema = {
        "type": "object",
        "properties": {
            "list": {
                "type": "array",
                "items": {}  # Empty items
            }
        }
    }
    strict = _strictify_schema(schema)

    items = strict["properties"]["list"]["items"]
    assert items["type"] == "object"


def test_strictify_handles_empty_anyof_branch():
    """Test that empty anyOf branches get default type"""
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "anyOf": [
                    {"type": "string"},
                    {}  # Empty branch
                ]
            }
        }
    }
    strict = _strictify_schema(schema)

    branches = strict["properties"]["value"]["anyOf"]
    assert branches[0]["type"] == "string"
    assert branches[1]["type"] == "object"


def test_strictify_infers_object_type_from_properties():
    """Test that schemas with properties but no type get 'object' inferred"""
    schema = {
        "type": "object",
        "properties": {
            "config": {
                # Has properties but no type - should infer "object"
                "properties": {
                    "enabled": {"type": "boolean"}
                }
            }
        }
    }
    strict = _strictify_schema(schema)

    # Should infer type as object and add null since it's optional
    assert "object" in strict["properties"]["config"]["type"]
    assert "null" in strict["properties"]["config"]["type"]


def test_strictify_infers_array_type_from_items():
    """Test that schemas with items but no type get 'array' inferred"""
    schema = {
        "type": "object",
        "properties": {
            "tags": {
                # Has items but no type - should infer "array"
                "items": {"type": "string"}
            }
        }
    }
    strict = _strictify_schema(schema)

    # Should infer type as array and add null since it's optional
    assert "array" in strict["properties"]["tags"]["type"]
    assert "null" in strict["properties"]["tags"]["type"]


def test_strictify_preserves_existing_behavior_for_valid_schemas():
    """Ensure fix doesn't break existing functionality"""
    schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "timeout": {"type": "integer"},
        },
        "required": ["path"],
    }

    strict = _strictify_schema(schema)

    # Existing behavior should be unchanged
    assert strict["required"] == sorted(["path", "timeout"])
    assert strict["properties"]["path"]["type"] == "string"
    assert strict["properties"]["timeout"]["type"] == ["integer", "null"]


def test_strictify_handles_auth_headers_scenario():
    """Test the exact scenario from the bug report"""
    schema = {
        "type": "object",
        "properties": {
            "session": {}  # This was causing the OpenAI error
        },
        "required": ["session"]
    }

    strict = _strictify_schema(schema)

    # Should now have a valid type
    assert "type" in strict["properties"]["session"]
    assert strict["properties"]["session"]["type"] == "object"
    assert strict["properties"]["session"]["additionalProperties"] is False
    assert "session" in strict["required"]
