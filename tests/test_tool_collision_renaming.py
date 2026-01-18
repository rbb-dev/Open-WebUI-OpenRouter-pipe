from __future__ import annotations

from typing import Any

import pytest


@pytest.mark.asyncio
async def test_collision_safe_tool_renaming_executes_both(pipe_instance_async):
    import open_webui_openrouter_pipe.pipe as pipe_mod

    async def builtin_search_web(**_kwargs: Any) -> str:
        return "builtin"

    async def direct_search_web(**_kwargs: Any) -> str:
        return "direct"

    request_tools = [
        {
            "type": "function",
            "name": "search_web",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]

    builtin_registry = {
        "search_web": {
            "spec": {
                "name": "search_web",
                "description": "builtin",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            "callable": builtin_search_web,
        }
    }

    direct_registry = {
        "search_web::0::0": {
            "spec": {
                "name": "search_web",
                "description": "direct",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            "direct": True,
            "origin_key": "search_web::0::0",
            "callable": direct_search_web,
        }
    }

    tools, exec_registry, exposed_to_origin = pipe_mod._build_collision_safe_tool_specs_and_registry(
        request_tool_specs=request_tools,
        owui_registry={},
        direct_registry=direct_registry,
        builtin_registry=builtin_registry,
        extra_tools=[],
        strictify=False,
        owui_tool_passthrough=False,
        logger=None,
    )

    assert exposed_to_origin["owui__search_web"] == "search_web"
    assert exposed_to_origin["direct__search_web"] == "search_web"
    assert {t["name"] for t in tools} == {"owui__search_web", "direct__search_web"}

    out1 = await pipe_instance_async._execute_function_calls(
        [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "owui__search_web",
                "arguments": '{"query":"x"}',
            }
        ],
        exec_registry,
    )
    out2 = await pipe_instance_async._execute_function_calls(
        [
            {
                "type": "function_call",
                "call_id": "c2",
                "name": "direct__search_web",
                "arguments": '{"query":"x"}',
            }
        ],
        exec_registry,
    )
    assert out1 and out1[0]["output"] == "builtin"
    assert out2 and out2[0]["output"] == "direct"


def test_collision_safe_tool_registry_passthrough_keeps_origin_map():
    import open_webui_openrouter_pipe.pipe as pipe_mod

    request_tools = [
        {
            "type": "function",
            "name": "search_web",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]

    direct_registry = {
        "search_web::0::0": {
            "spec": {
                "name": "search_web",
                "description": "direct",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            "direct": True,
            "origin_key": "search_web::0::0",
            "callable": lambda **_kwargs: None,
        }
    }

    tools, exec_registry, exposed_to_origin = pipe_mod._build_collision_safe_tool_specs_and_registry(
        request_tool_specs=request_tools,
        owui_registry={},
        direct_registry=direct_registry,
        builtin_registry={},
        extra_tools=[],
        strictify=False,
        owui_tool_passthrough=True,
        logger=None,
    )

    assert not exec_registry
    assert exposed_to_origin["owui__search_web"] == "search_web"
    assert exposed_to_origin["direct__search_web"] == "search_web"
    assert {t["name"] for t in tools} == {"owui__search_web", "direct__search_web"}


@pytest.mark.asyncio
async def test_registry_tool_ids_tool_still_executes(pipe_instance_async):
    import open_webui_openrouter_pipe.pipe as pipe_mod

    async def kb_tool(**_kwargs: Any) -> str:
        return "ok"

    owui_registry = {
        "kb_query": {
            "spec": {
                "name": "kb_query",
                "description": "KB query",
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            },
            "callable": kb_tool,
        }
    }

    tools, exec_registry, _map = pipe_mod._build_collision_safe_tool_specs_and_registry(
        request_tool_specs=[],
        owui_registry=owui_registry,
        direct_registry={},
        builtin_registry={},
        extra_tools=[],
        strictify=False,
        owui_tool_passthrough=False,
        logger=None,
    )

    assert {t["name"] for t in tools} == {"kb_query"}
    result = await pipe_instance_async._execute_function_calls(
        [{"type": "function_call", "call_id": "c1", "name": "kb_query", "arguments": '{"q":"x"}'}],
        exec_registry,
    )
    assert result and result[0]["output"] == "ok"

