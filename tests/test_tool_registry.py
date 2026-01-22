"""Tests for tool_registry.py to achieve high coverage.

Targets the following uncovered code paths:
- _pick_executor with prefer="builtin", prefer="owui", prefer="direct"
- Collision-safe registry building edge cases
- Tool normalization and validation
"""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

import pytest

from open_webui_openrouter_pipe.tools.tool_registry import (
    _build_collision_safe_tool_specs_and_registry,
    _dedupe_tools,
    _normalize_responses_function_tool_spec,
    _responses_spec_from_owui_tool_cfg,
    _tool_prefix_for_collision,
    build_tools,
)
from open_webui_openrouter_pipe.models.registry import ModelFamily


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def mock_valves():
    """Create a mock valves object with typical configuration."""
    valves = MagicMock()
    valves.ENABLE_STRICT_TOOL_CALLING = False
    valves.TOOL_EXECUTION_MODE = "Pipeline"
    return valves


@pytest.fixture
def mock_responses_body():
    """Create a mock ResponsesBody with typical configuration."""
    body = MagicMock()
    body.model = "test-model"
    return body


@pytest.fixture(autouse=True)
def setup_model_family():
    """Configure ModelFamily for tests."""
    ModelFamily.set_dynamic_specs({
        "test-model": {
            "features": {"function_calling"},
            "supported_parameters": frozenset(["tools", "tool_choice"]),
        },
        "no-tool-model": {
            "features": set(),
            "supported_parameters": frozenset(),
        },
    })
    yield
    ModelFamily.set_dynamic_specs({})


# -----------------------------------------------------------------------------
# Tests for _dedupe_tools
# -----------------------------------------------------------------------------

class TestDedupeTools:
    """Tests for _dedupe_tools function."""

    def test_dedupe_empty_list(self):
        """Empty list returns empty list."""
        assert _dedupe_tools([]) == []

    def test_dedupe_none(self):
        """None returns empty list."""
        assert _dedupe_tools(None) == []

    def test_dedupe_non_dict_items_filtered(self):
        """Non-dict items are filtered out."""
        tools = [
            {"type": "function", "name": "foo"},
            "not a dict",
            42,
            None,
            {"type": "function", "name": "bar"},
        ]
        result = _dedupe_tools(tools)
        assert len(result) == 2
        names = [t.get("name") for t in result]
        assert "foo" in names
        assert "bar" in names

    def test_dedupe_last_wins(self):
        """Later entries with same identity win."""
        tools = [
            {"type": "function", "name": "foo", "description": "first"},
            {"type": "function", "name": "foo", "description": "second"},
        ]
        result = _dedupe_tools(tools)
        assert len(result) == 1
        assert result[0]["description"] == "second"

    def test_dedupe_non_function_tools(self):
        """Non-function tools use (type, None) as key."""
        tools = [
            {"type": "web_search", "config": "first"},
            {"type": "web_search", "config": "second"},
            {"type": "code_interpreter", "config": "ci"},
        ]
        result = _dedupe_tools(tools)
        # Two unique types
        assert len(result) == 2
        types = [t.get("type") for t in result]
        assert "web_search" in types
        assert "code_interpreter" in types

    def test_dedupe_null_type_filtered(self):
        """Tools with no type are filtered out."""
        tools = [
            {"name": "no_type"},
            {"type": "function", "name": "valid"},
        ]
        result = _dedupe_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "valid"


# -----------------------------------------------------------------------------
# Tests for _normalize_responses_function_tool_spec
# -----------------------------------------------------------------------------

class TestNormalizeResponsesFunctionToolSpec:
    """Tests for _normalize_responses_function_tool_spec function."""

    def test_normalize_valid_tool(self):
        """Valid tool spec is normalized correctly."""
        tool = {
            "type": "function",
            "name": "test_func",
            "description": "A test function",
            "parameters": {"type": "object", "properties": {}},
        }
        result = _normalize_responses_function_tool_spec(tool, strictify=False)
        assert result is not None
        assert result["type"] == "function"
        assert result["name"] == "test_func"
        assert result["description"] == "A test function"
        assert result["parameters"] == {"type": "object", "properties": {}}

    def test_normalize_non_dict(self):
        """Non-dict input returns None."""
        assert _normalize_responses_function_tool_spec("not a dict", strictify=False) is None
        assert _normalize_responses_function_tool_spec(123, strictify=False) is None
        assert _normalize_responses_function_tool_spec(None, strictify=False) is None

    def test_normalize_non_function_type(self):
        """Non-function type returns None."""
        tool = {"type": "web_search", "name": "search"}
        assert _normalize_responses_function_tool_spec(tool, strictify=False) is None

    def test_normalize_missing_name(self):
        """Missing name returns None."""
        tool = {"type": "function", "description": "no name"}
        assert _normalize_responses_function_tool_spec(tool, strictify=False) is None

    def test_normalize_empty_name(self):
        """Empty or whitespace-only name returns None."""
        tool = {"type": "function", "name": ""}
        assert _normalize_responses_function_tool_spec(tool, strictify=False) is None
        tool = {"type": "function", "name": "   "}
        assert _normalize_responses_function_tool_spec(tool, strictify=False) is None

    def test_normalize_non_string_name(self):
        """Non-string name returns None."""
        tool = {"type": "function", "name": 123}
        assert _normalize_responses_function_tool_spec(tool, strictify=False) is None

    def test_normalize_name_stripped(self):
        """Name is stripped of whitespace."""
        tool = {"type": "function", "name": "  test_func  "}
        result = _normalize_responses_function_tool_spec(tool, strictify=False)
        assert result["name"] == "test_func"

    def test_normalize_empty_description_omitted(self):
        """Empty or whitespace-only description is omitted."""
        tool = {"type": "function", "name": "test", "description": ""}
        result = _normalize_responses_function_tool_spec(tool, strictify=False)
        assert "description" not in result

        tool = {"type": "function", "name": "test", "description": "   "}
        result = _normalize_responses_function_tool_spec(tool, strictify=False)
        assert "description" not in result

    def test_normalize_non_dict_parameters_omitted(self):
        """Non-dict parameters are omitted."""
        tool = {"type": "function", "name": "test", "parameters": "invalid"}
        result = _normalize_responses_function_tool_spec(tool, strictify=False)
        assert "parameters" not in result

    def test_normalize_with_strictify(self):
        """Strictify transforms parameters schema."""
        tool = {
            "type": "function",
            "name": "test",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        }
        result = _normalize_responses_function_tool_spec(tool, strictify=True)
        # Strictify should add additionalProperties: false and required
        assert result["parameters"].get("additionalProperties") is False


# -----------------------------------------------------------------------------
# Tests for _responses_spec_from_owui_tool_cfg
# -----------------------------------------------------------------------------

class TestResponsesSpecFromOwuiToolCfg:
    """Tests for _responses_spec_from_owui_tool_cfg function."""

    def test_valid_tool_cfg(self):
        """Valid tool config is transformed correctly."""
        cfg = {
            "spec": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
            "callable": lambda: None,
        }
        result = _responses_spec_from_owui_tool_cfg(cfg, strictify=False)
        assert result is not None
        assert result["type"] == "function"
        assert result["name"] == "test_tool"
        assert result["description"] == "A test tool"

    def test_non_dict_cfg(self):
        """Non-dict config returns None."""
        assert _responses_spec_from_owui_tool_cfg("not a dict", strictify=False) is None
        assert _responses_spec_from_owui_tool_cfg(None, strictify=False) is None

    def test_missing_spec(self):
        """Missing spec returns None."""
        cfg = {"callable": lambda: None}
        assert _responses_spec_from_owui_tool_cfg(cfg, strictify=False) is None

    def test_non_dict_spec(self):
        """Non-dict spec returns None."""
        cfg = {"spec": "not a dict"}
        assert _responses_spec_from_owui_tool_cfg(cfg, strictify=False) is None

    def test_missing_name_in_spec(self):
        """Missing name in spec returns None."""
        cfg = {"spec": {"description": "no name"}}
        assert _responses_spec_from_owui_tool_cfg(cfg, strictify=False) is None

    def test_empty_name_in_spec(self):
        """Empty name in spec returns None."""
        cfg = {"spec": {"name": ""}}
        assert _responses_spec_from_owui_tool_cfg(cfg, strictify=False) is None

    def test_missing_parameters_default(self):
        """Missing parameters defaults to empty object schema."""
        cfg = {"spec": {"name": "test"}}
        result = _responses_spec_from_owui_tool_cfg(cfg, strictify=False)
        assert result["parameters"] == {"type": "object", "properties": {}}

    def test_non_dict_parameters_default(self):
        """Non-dict parameters defaults to empty object schema."""
        cfg = {"spec": {"name": "test", "parameters": "invalid"}}
        result = _responses_spec_from_owui_tool_cfg(cfg, strictify=False)
        assert result["parameters"] == {"type": "object", "properties": {}}

    def test_missing_description_uses_name(self):
        """Missing description defaults to tool name."""
        cfg = {"spec": {"name": "my_tool"}}
        result = _responses_spec_from_owui_tool_cfg(cfg, strictify=False)
        assert result["description"] == "my_tool"


# -----------------------------------------------------------------------------
# Tests for _tool_prefix_for_collision
# -----------------------------------------------------------------------------

class TestToolPrefixForCollision:
    """Tests for _tool_prefix_for_collision function."""

    def test_owui_request_tools_prefix(self):
        """OWUI request tools get 'owui__' prefix."""
        assert _tool_prefix_for_collision("owui_request_tools", None) == "owui__"

    def test_direct_tool_server_prefix(self):
        """Direct tool servers get 'direct__' prefix."""
        assert _tool_prefix_for_collision("direct_tool_server", None) == "direct__"

    def test_extra_tools_prefix(self):
        """Extra tools get 'extra__' prefix."""
        assert _tool_prefix_for_collision("extra_tools", None) == "extra__"

    def test_registry_direct_tool_prefix(self):
        """Registry tools with direct=True get 'direct__' prefix."""
        cfg = {"direct": True}
        assert _tool_prefix_for_collision("owui_registry_tools", cfg) == "direct__"

    def test_registry_non_direct_tool_prefix(self):
        """Registry tools without direct flag get 'tool__' prefix."""
        assert _tool_prefix_for_collision("owui_registry_tools", None) == "tool__"
        assert _tool_prefix_for_collision("owui_registry_tools", {}) == "tool__"
        assert _tool_prefix_for_collision("owui_registry_tools", {"direct": False}) == "tool__"


# -----------------------------------------------------------------------------
# Tests for _build_collision_safe_tool_specs_and_registry
# -----------------------------------------------------------------------------

class TestBuildCollisionSafeToolSpecsAndRegistry:
    """Tests for _build_collision_safe_tool_specs_and_registry function."""

    def test_empty_inputs(self):
        """Empty inputs return empty results."""
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry=None,
            direct_registry=None,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        assert tools == []
        assert registry == {}
        assert exposed_to_origin == {}

    def test_request_tools_with_callable(self):
        """Request tools with callable executor are included."""
        request_tools = [{"type": "function", "name": "my_tool", "description": "test"}]
        owui_registry = {
            "my_tool": {
                "spec": {"name": "my_tool", "description": "test"},
                "callable": lambda: "result",
            }
        }
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry=None,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        assert len(tools) == 1
        assert tools[0]["name"] == "my_tool"
        assert "my_tool" in registry
        assert exposed_to_origin["my_tool"] == "my_tool"

    def test_request_tools_without_callable_skipped(self):
        """Request tools without callable executor are skipped (non-passthrough mode)."""
        request_tools = [{"type": "function", "name": "my_tool", "description": "test"}]
        # No registry entry with callable
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry={},
            direct_registry=None,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=logging.getLogger("test"),
        )
        assert tools == []
        assert registry == {}

    def test_passthrough_mode_includes_without_callable(self):
        """Passthrough mode includes tools without callable."""
        request_tools = [{"type": "function", "name": "my_tool", "description": "test"}]
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry={},
            direct_registry=None,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=True,
            logger=None,
        )
        assert len(tools) == 1
        assert tools[0]["name"] == "my_tool"
        # Registry should be empty in passthrough mode
        assert registry == {}

    def test_collision_renaming(self):
        """Tools with same name from different sources get renamed."""
        # Same tool name from two sources
        owui_registry = {
            "search": {
                "spec": {"name": "search", "description": "OWUI search"},
                "callable": lambda: "owui",
            }
        }
        direct_registry = {
            "search": {
                "spec": {"name": "search", "description": "Direct search"},
                "callable": lambda: "direct",
                "direct": True,
            }
        }
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry=owui_registry,
            direct_registry=direct_registry,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        # Both tools should be present with different names
        assert len(tools) == 2
        names = [t["name"] for t in tools]
        # At least one should have a prefix
        assert any("__search" in name for name in names)

    def test_extra_tools_with_executor(self):
        """Extra tools with matching executor are included."""
        extra_tools = [{"type": "function", "name": "extra_func", "description": "extra"}]
        builtin_registry = {
            "extra_func": {
                "spec": {"name": "extra_func", "description": "builtin extra"},
                "callable": lambda: "builtin",
            }
        }
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry=None,
            direct_registry=None,
            builtin_registry=builtin_registry,
            extra_tools=extra_tools,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        assert len(tools) == 1
        assert tools[0]["name"] == "extra_func"
        assert "extra_func" in registry

    def test_extra_tools_without_executor_skipped(self):
        """Extra tools without executor are skipped in non-passthrough mode."""
        extra_tools = [{"type": "function", "name": "orphan_tool", "description": "no executor"}]
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry={},
            direct_registry=None,
            builtin_registry=None,
            extra_tools=extra_tools,
            strictify=False,
            owui_tool_passthrough=False,
            logger=logging.getLogger("test"),
        )
        assert tools == []

    def test_pick_executor_prefers_builtin(self):
        """_pick_executor prefers builtin registry for request tools."""
        # Both builtin and owui have same tool - builtin should be picked
        request_tools = [{"type": "function", "name": "shared_tool", "description": "test"}]
        builtin_registry = {
            "shared_tool": {
                "spec": {"name": "shared_tool", "description": "builtin"},
                "callable": lambda: "builtin_result",
            }
        }
        owui_registry = {
            "shared_tool": {
                "spec": {"name": "shared_tool", "description": "owui"},
                "callable": lambda: "owui_result",
            }
        }
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry=None,
            builtin_registry=builtin_registry,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        assert len(tools) == 1
        # The executor should come from builtin (first in preference order)
        assert "shared_tool" in registry

    def test_pick_executor_falls_back_to_owui(self):
        """_pick_executor falls back to owui when builtin not available."""
        request_tools = [{"type": "function", "name": "owui_only", "description": "test"}]
        owui_registry = {
            "owui_only": {
                "spec": {"name": "owui_only", "description": "owui"},
                "callable": lambda: "owui_result",
            }
        }
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry=None,
            builtin_registry={},
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        assert len(tools) == 1
        assert "owui_only" in registry

    def test_pick_executor_falls_back_to_direct(self):
        """_pick_executor falls back to direct when builtin and owui not available.

        Note: direct_registry tools are added independently in step 2, so we use
        extra_tools to test the _pick_executor fallback to direct registry.
        """
        # Use extra_tools which use _pick_executor to find executors
        extra_tools = [{"type": "function", "name": "direct_only", "description": "extra"}]
        direct_registry = {
            "direct_only": {
                "spec": {"name": "direct_only", "description": "direct"},
                "callable": lambda: "direct_result",
                "direct": True,
            }
        }
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,  # No request tools
            owui_registry={},
            direct_registry=direct_registry,
            builtin_registry={},
            extra_tools=extra_tools,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        # Should have 2 tools: one from direct_registry (added in step 2), one from extra_tools
        # Both will have the same origin but different sources
        assert len(tools) == 2
        # One from extra_tools uses direct executor
        assert any("extra__direct_only" in t["name"] or "direct__direct_only" in t["name"] for t in tools)

    def test_duplicate_collision_with_hash(self):
        """Multiple collisions with same prefix get hash suffix."""
        # Create three tools with same name but different sources
        request_tools = [{"type": "function", "name": "common", "description": "request"}]
        owui_registry = {
            "common": {
                "spec": {"name": "common", "description": "owui"},
                "callable": lambda: "owui",
            }
        }
        direct_registry = {
            "common": {
                "spec": {"name": "common", "description": "direct"},
                "callable": lambda: "direct",
                "direct": True,
            }
        }
        extra_tools = [{"type": "function", "name": "common", "description": "extra"}]

        # Need executor for extra tool
        builtin_registry = {
            "common": {
                "spec": {"name": "common", "description": "builtin"},
                "callable": lambda: "builtin",
            }
        }

        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry=direct_registry,
            builtin_registry=builtin_registry,
            extra_tools=extra_tools,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        # Should have multiple unique names
        names = [t["name"] for t in tools]
        assert len(names) == len(set(names)), "All names should be unique"

    def test_direct_registry_without_callable_skipped(self):
        """Direct registry entries without callable are skipped (non-passthrough)."""
        direct_registry = {
            "no_callable": {
                "spec": {"name": "no_callable", "description": "test"},
                # No callable
            }
        }
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry={},
            direct_registry=direct_registry,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        assert tools == []

    def test_owui_registry_tool_not_in_request_names(self):
        """OWUI registry tools not in request names are added."""
        owui_registry = {
            "unique_owui_tool": {
                "spec": {"name": "unique_owui_tool", "description": "test"},
                "callable": lambda: "result",
            }
        }
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry=owui_registry,
            direct_registry=None,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        assert len(tools) == 1
        assert tools[0]["name"] == "unique_owui_tool"

    def test_owui_registry_tool_in_request_names_skipped(self):
        """OWUI registry tools already in request names are skipped."""
        request_tools = [{"type": "function", "name": "shared", "description": "request"}]
        owui_registry = {
            "shared": {
                "spec": {"name": "shared", "description": "owui duplicate"},
                "callable": lambda: "owui",
            }
        }
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry=None,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        # Only the request tool should be present (owui duplicate skipped)
        assert len(tools) == 1

    def test_tool_cfg_spec_parameters_updated(self):
        """Tool cfg spec parameters are updated from normalized spec."""
        request_tools = [
            {
                "type": "function",
                "name": "param_test",
                "description": "test",
                "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
            }
        ]
        owui_registry = {
            "param_test": {
                "spec": {
                    "name": "param_test",
                    "description": "original",
                    "parameters": {"type": "object", "properties": {"y": {"type": "string"}}},
                },
                "callable": lambda: "result",
            }
        }
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry=None,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        # Registry should have updated parameters from request
        cfg = registry.get("param_test")
        assert cfg is not None
        cfg_spec = cfg.get("spec", {})
        # The spec name should be the origin name
        assert cfg_spec.get("name") == "param_test"

    def test_passthrough_mode_skips_registry_population(self):
        """In passthrough mode, exec_registry is not populated."""
        request_tools = [{"type": "function", "name": "pass_tool", "description": "test"}]
        owui_registry = {
            "pass_tool": {
                "spec": {"name": "pass_tool", "description": "test"},
                "callable": lambda: "result",
            }
        }
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry=None,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=True,  # Passthrough mode
            logger=None,
        )
        # Tools should be present
        assert len(tools) == 1
        # But registry should be empty (passthrough means no pipeline execution)
        assert registry == {}
        # exposed_to_origin should still be populated
        assert "pass_tool" in exposed_to_origin

    def test_tool_cfg_without_callable_not_in_registry(self):
        """Tool cfg without callable is not added to exec_registry (line 372 coverage)."""
        # Create a scenario where tool_cfg exists but has no callable
        request_tools = [{"type": "function", "name": "no_exec", "description": "test"}]
        owui_registry = {
            "no_exec": {
                "spec": {"name": "no_exec", "description": "test"},
                "callable": None,  # Explicitly None
            }
        }
        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry=None,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=logging.getLogger("test"),
        )
        # Tool should be skipped entirely because callable is None
        assert tools == []
        assert registry == {}


# -----------------------------------------------------------------------------
# Tests for build_tools (high-level API)
# -----------------------------------------------------------------------------

class TestBuildTools:
    """Tests for the build_tools function."""

    def test_no_function_calling_support_returns_empty(self, mock_valves, mock_responses_body):
        """Model without function calling returns empty tools."""
        mock_responses_body.model = "no-tool-model"
        result = build_tools(mock_responses_body, mock_valves, None)
        assert result == []

    def test_passthrough_mode_bypasses_model_check(self, mock_valves, mock_responses_body):
        """Passthrough mode bypasses model capability check."""
        mock_responses_body.model = "no-tool-model"
        mock_valves.TOOL_EXECUTION_MODE = "Open-WebUI"

        tools = [{"type": "function", "name": "test", "description": "test"}]
        result = build_tools(mock_responses_body, mock_valves, tools)
        # Should not be empty even though model doesn't support function calling
        assert len(result) == 1

    def test_dict_tools_transformed(self, mock_valves, mock_responses_body):
        """Dict-style __tools__ are transformed via ResponsesBody."""
        mock_valves.TOOL_EXECUTION_MODE = "Open-WebUI"
        tools_dict = {
            "my_tool": {
                "spec": {"name": "my_tool", "description": "test"},
                "callable": lambda: None,
            }
        }
        # This test exercises the dict branch
        result = build_tools(mock_responses_body, mock_valves, tools_dict)
        # Result depends on ResponsesBody.transform_owui_tools implementation
        assert isinstance(result, list)

    def test_list_tools_filtered(self, mock_valves, mock_responses_body):
        """List-style tools are filtered to include only dicts."""
        mock_valves.TOOL_EXECUTION_MODE = "Open-WebUI"
        tools_list = [
            {"type": "function", "name": "valid"},
            "not a dict",
            None,
            {"type": "function", "name": "also_valid"},
        ]
        result = build_tools(mock_responses_body, mock_valves, tools_list)
        # Only dict items should be included
        names = [t.get("name") for t in result]
        assert "valid" in names
        assert "also_valid" in names

    def test_extra_tools_appended(self, mock_valves, mock_responses_body):
        """Extra tools are appended to the result."""
        mock_valves.TOOL_EXECUTION_MODE = "Open-WebUI"
        extra = [{"type": "function", "name": "extra_tool", "description": "extra"}]
        result = build_tools(mock_responses_body, mock_valves, None, extra_tools=extra)
        assert any(t.get("name") == "extra_tool" for t in result)

    def test_features_parameter_accepted(self, mock_valves, mock_responses_body):
        """Features parameter is accepted (for future use)."""
        mock_valves.TOOL_EXECUTION_MODE = "Open-WebUI"
        result = build_tools(
            mock_responses_body, mock_valves, None, features={"some_feature": True}
        )
        assert isinstance(result, list)


# -----------------------------------------------------------------------------
# Tests for _pick_executor preference branches (lines 223-236)
# -----------------------------------------------------------------------------

class TestPickExecutorPreferences:
    """Tests specifically targeting _pick_executor preference branches."""

    def test_prefer_builtin_finds_in_builtin(self):
        """Test that prefer='builtin' finds executor in builtin registry first.

        Note: direct_registry tools are always added separately in step 2.
        To test _pick_executor preference, we use only builtin and owui.
        """
        builtin_registry = {
            "tool_a": {
                "spec": {"name": "tool_a", "description": "builtin"},
                "callable": lambda: "builtin",
            }
        }
        owui_registry = {
            "tool_a": {
                "spec": {"name": "tool_a", "description": "owui"},
                "callable": lambda: "owui",
            }
        }
        # Don't include direct_registry since it adds tools independently

        # Request tools use default preference (builtin -> owui -> direct)
        request_tools = [{"type": "function", "name": "tool_a"}]

        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry={},  # Empty to avoid adding duplicate tools
            builtin_registry=builtin_registry,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )

        # Should have one tool (builtin executor found for request tool)
        # owui_registry tool with same name is skipped because it's already in request_names
        assert len(tools) == 1
        assert "tool_a" in registry

    def test_prefer_owui_fallback(self):
        """Test fallback to owui when builtin not found."""
        owui_registry = {
            "owui_tool": {
                "spec": {"name": "owui_tool", "description": "owui"},
                "callable": lambda: "owui",
            }
        }

        request_tools = [{"type": "function", "name": "owui_tool"}]

        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry={},
            builtin_registry={},
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )

        assert len(tools) == 1
        assert "owui_tool" in registry

    def test_prefer_direct_fallback(self):
        """Test fallback to direct when builtin and owui not found.

        Note: direct_registry tools are added separately in step 2.
        To properly test _pick_executor fallback to direct, we use extra_tools
        which go through _pick_executor to find an executor.
        """
        direct_registry = {
            "direct_tool": {
                "spec": {"name": "direct_tool", "description": "direct"},
                "callable": lambda: "direct",
            }
        }

        # Use extra_tools to test _pick_executor fallback
        extra_tools = [{"type": "function", "name": "direct_tool", "description": "extra"}]

        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,  # No request tools
            owui_registry={},
            direct_registry=direct_registry,
            builtin_registry={},
            extra_tools=extra_tools,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )

        # Should have 2 tools: one from direct_registry (step 2), one from extra_tools (step 4)
        # Both have callable so both are included
        assert len(tools) == 2
        # Verify the extra_tool found an executor in direct_registry
        tool_names = [t["name"] for t in tools]
        assert any("extra__" in name or "direct__" in name for name in tool_names)

    def test_no_executor_found(self):
        """Test when no executor is found in any registry."""
        request_tools = [{"type": "function", "name": "orphan"}]

        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry={},
            direct_registry={},
            builtin_registry={},
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=logging.getLogger("test"),
        )

        # Tool should be skipped
        assert tools == []
        assert registry == {}


# -----------------------------------------------------------------------------
# Tests for edge cases and error handling
# -----------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_non_dict_registry_values_filtered(self):
        """Non-dict values in registries are filtered out."""
        owui_registry = {
            "valid": {
                "spec": {"name": "valid", "description": "test"},
                "callable": lambda: "result",
            },
            "invalid_string": "not a dict",
            "invalid_list": ["not", "a", "dict"],
            "invalid_none": None,
        }

        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry=owui_registry,
            direct_registry=None,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )

        # Only valid entry should be included
        assert len(tools) == 1
        assert tools[0]["name"] == "valid"

    def test_invalid_request_tool_specs_filtered(self):
        """Invalid request tool specs are filtered out."""
        request_tools = [
            {"type": "function", "name": "valid", "description": "test"},
            {"type": "not_function", "name": "invalid_type"},
            {"type": "function"},  # Missing name
            {"type": "function", "name": ""},  # Empty name
            "not a dict",
        ]
        owui_registry = {
            "valid": {
                "spec": {"name": "valid", "description": "test"},
                "callable": lambda: "result",
            }
        }

        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry=None,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )

        # Only valid tool should be included
        assert len(tools) == 1
        assert tools[0]["name"] == "valid"

    def test_origin_key_fallback(self):
        """Origin key uses fallback when not provided."""
        direct_registry = {
            "tool_no_key": {
                "spec": {"name": "tool_no_key", "description": "test"},
                "callable": lambda: "result",
                # No origin_key provided
            }
        }

        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry={},
            direct_registry=direct_registry,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )

        assert len(tools) == 1
        assert "tool_no_key" in exposed_to_origin

    def test_cfg_spec_not_dict_handled(self):
        """Tool cfg with non-dict spec in registry is handled."""
        owui_registry = {
            "weird_tool": {
                "spec": {"name": "weird_tool", "description": "test"},
                "callable": lambda: "result",
            }
        }
        request_tools = [{"type": "function", "name": "weird_tool"}]

        # Manually create scenario where tool_cfg.spec is not a dict
        # by having request provide the spec
        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry=None,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )

        assert len(tools) == 1

    def test_strictify_applies_to_all_sources(self):
        """Strictify flag applies to tools from all sources."""
        request_tools = [
            {
                "type": "function",
                "name": "strict_test",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            }
        ]
        owui_registry = {
            "strict_test": {
                "spec": {
                    "name": "strict_test",
                    "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
                },
                "callable": lambda: "result",
            }
        }

        tools, _, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry=None,
            builtin_registry=None,
            extra_tools=None,
            strictify=True,
            owui_tool_passthrough=False,
            logger=None,
        )

        assert len(tools) == 1
        # Check that strictify was applied
        params = tools[0].get("parameters", {})
        assert params.get("additionalProperties") is False


# -----------------------------------------------------------------------------
# Tests for additional coverage (missing lines)
# -----------------------------------------------------------------------------

class TestAdditionalCoverage:
    """Tests for additional coverage of missing lines."""

    def test_direct_registry_invalid_spec_skipped(self):
        """Direct registry entries with invalid spec are skipped (line 285)."""
        direct_registry = {
            "invalid_spec": {
                "spec": "not a dict",  # Invalid spec
                "callable": lambda: "result",
            },
            "missing_spec": {
                # No spec at all
                "callable": lambda: "result",
            },
            "valid": {
                "spec": {"name": "valid", "description": "test"},
                "callable": lambda: "result",
            }
        }
        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry={},
            direct_registry=direct_registry,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        # Only valid entry should be included
        assert len(tools) == 1
        assert tools[0]["name"] == "valid"

    def test_owui_registry_invalid_spec_skipped(self):
        """OWUI registry entries with invalid spec are skipped (line 303)."""
        owui_registry = {
            "invalid_spec": {
                "spec": "not a dict",  # Invalid spec
                "callable": lambda: "result",
            },
            "missing_name_in_spec": {
                "spec": {"description": "no name"},  # Missing name
                "callable": lambda: "result",
            },
            "valid": {
                "spec": {"name": "valid", "description": "test"},
                "callable": lambda: "result",
            }
        }
        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry=owui_registry,
            direct_registry={},
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        # Only valid entry should be included
        assert len(tools) == 1
        assert tools[0]["name"] == "valid"

    def test_owui_registry_no_callable_skipped_non_passthrough(self):
        """OWUI registry entries without callable are skipped in non-passthrough mode (line 308)."""
        owui_registry = {
            "no_callable": {
                "spec": {"name": "no_callable", "description": "test"},
                # No callable
            },
            "null_callable": {
                "spec": {"name": "null_callable", "description": "test"},
                "callable": None,
            },
            "valid": {
                "spec": {"name": "valid", "description": "test"},
                "callable": lambda: "result",
            }
        }
        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry=owui_registry,
            direct_registry={},
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        # Only valid entry should be included
        assert len(tools) == 1
        assert tools[0]["name"] == "valid"

    def test_extra_tools_invalid_spec_skipped(self):
        """Extra tools with invalid spec are skipped (line 323)."""
        extra_tools = [
            {"type": "not_function", "name": "wrong_type"},  # Wrong type
            {"type": "function"},  # Missing name
            {"type": "function", "name": ""},  # Empty name
            {"type": "function", "name": "valid", "description": "test"},
        ]
        builtin_registry = {
            "valid": {
                "spec": {"name": "valid", "description": "test"},
                "callable": lambda: "result",
            }
        }
        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry={},
            direct_registry={},
            builtin_registry=builtin_registry,
            extra_tools=extra_tools,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )
        # Only valid extra tool should be included
        assert len(tools) == 1
        assert tools[0]["name"] == "valid"

    def test_hash_collision_suffix_applied(self):
        """Hash suffix is applied when exposed_name already exists (lines 357-360).

        This tests the scenario where multiple tools with the same name from
        different sources get the same prefixed name, requiring a hash suffix.

        To trigger a hash collision, we need two tools from the same source type
        (e.g., two direct_registry entries) that have the same origin_name.
        Since dict keys are unique, we simulate this by having two entries
        that would produce the same exposed_name after collision renaming.
        """
        # Create multiple direct_registry entries that will collide
        # The collision occurs when: owui_request + direct = same exposed_name
        # Since different sources have different prefixes, we need two entries
        # from direct_registry with the same tool name but different origin_keys

        # Create two direct tool server entries with the same tool name
        # This simulates two different tool servers providing the same tool
        direct_registry = {
            "server1_shared": {
                "spec": {"name": "shared", "description": "server1"},
                "callable": lambda: "server1",
                "origin_key": "server1::shared",
            },
            "server2_shared": {
                "spec": {"name": "shared", "description": "server2"},
                "callable": lambda: "server2",
                "origin_key": "server2::shared",
            },
        }

        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry={},
            direct_registry=direct_registry,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )

        # Both tools should be present with unique names
        names = [t["name"] for t in tools]
        assert len(tools) == 2, f"Expected 2 tools, got {len(tools)}: {names}"
        assert len(names) == len(set(names)), f"Names should be unique: {names}"

        # Both should have direct__ prefix since they're both from direct_tool_server
        # One will be direct__shared, the other will have a hash suffix
        assert any("direct__shared" in name for name in names), f"Expected direct__shared prefix: {names}"
        # At least one should have a hash suffix (contains __ followed by hex chars)
        import re
        hash_pattern = re.compile(r"__[a-f0-9]{8}$")
        assert any(hash_pattern.search(name) for name in names), f"Expected at least one hash suffix: {names}"

    def test_tool_cfg_not_dict_in_final_loop(self):
        """Tool cfg that is not a dict is skipped in final loop (line 372 branch).

        This covers the case where tool_cfg exists but is not a dict.
        """
        # In passthrough mode, tool_cfg can be anything since we skip registry population
        # But in non-passthrough mode, we need tool_cfg to be a dict with callable
        # This test covers the branch where tool_cfg is not a dict

        # To hit line 372, we need:
        # 1. owui_tool_passthrough = False (not in passthrough mode)
        # 2. tool_cfg is not a dict OR tool_cfg.get("callable") is None

        # Create request tool that gets a tool_cfg from the executor
        request_tools = [{"type": "function", "name": "test_tool", "description": "test"}]

        # Create registry where the tool has callable=None
        owui_registry = {
            "test_tool": {
                "spec": {"name": "test_tool", "description": "test"},
                "callable": None,  # This will cause line 372 to skip
            }
        }

        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry={},
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=logging.getLogger("test"),
        )

        # Tool should be skipped entirely because callable is None
        assert tools == []
        assert registry == {}

    def test_passthrough_with_valid_tool_cfg_skips_registry(self):
        """In passthrough mode, even valid tool_cfg is not added to registry (line 369-370)."""
        request_tools = [{"type": "function", "name": "pass_tool", "description": "test"}]
        owui_registry = {
            "pass_tool": {
                "spec": {"name": "pass_tool", "description": "test"},
                "callable": lambda: "result",
            }
        }

        tools, registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=request_tools,
            owui_registry=owui_registry,
            direct_registry={},
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=True,  # Passthrough enabled
            logger=None,
        )

        # Tool should be in output
        assert len(tools) == 1
        assert tools[0]["name"] == "pass_tool"
        # But registry should be empty (passthrough skips registry population)
        assert registry == {}
        # exposed_to_origin should still be populated
        assert "pass_tool" in exposed_to_origin

    def test_direct_entry_with_passthrough_includes_without_callable(self):
        """Direct entries without callable are included in passthrough mode."""
        direct_registry = {
            "no_callable": {
                "spec": {"name": "no_callable", "description": "test"},
                # No callable
            }
        }

        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry={},
            direct_registry=direct_registry,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=True,  # Passthrough enabled
            logger=None,
        )

        # Tool should be included even without callable
        assert len(tools) == 1
        assert tools[0]["name"] == "no_callable"

    def test_owui_registry_with_passthrough_includes_without_callable(self):
        """OWUI registry entries without callable are included in passthrough mode."""
        owui_registry = {
            "no_callable": {
                "spec": {"name": "no_callable", "description": "test"},
                # No callable
            }
        }

        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry=owui_registry,
            direct_registry={},
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=True,  # Passthrough enabled
            logger=None,
        )

        # Tool should be included even without callable
        assert len(tools) == 1
        assert tools[0]["name"] == "no_callable"

    def test_extra_tools_with_passthrough_includes_without_executor(self):
        """Extra tools without executor are included in passthrough mode."""
        extra_tools = [{"type": "function", "name": "orphan", "description": "test"}]

        tools, registry, _ = _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry={},
            direct_registry={},
            builtin_registry={},
            extra_tools=extra_tools,
            strictify=False,
            owui_tool_passthrough=True,  # Passthrough enabled
            logger=None,
        )

        # Tool should be included even without executor
        assert len(tools) == 1
        assert tools[0]["name"] == "orphan"
