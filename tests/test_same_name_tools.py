"""Tools that expose the same function name must each stay reachable, in both tool execution modes.

Open WebUI resolves a chat's tools into one dictionary. When two tools define the same function name it renames only
the dictionary key (the later one becomes ``<tool_id>_<name>``) and leaves ``spec["name"]`` as it was, then sends one
spec per tool for native function calling, so the request carries that name more than once. Open WebUI itself runs a
call by the dictionary key.
"""

from __future__ import annotations

import re
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody
from open_webui_openrouter_pipe.api.transforms import _chat_tools_to_responses_tools
from open_webui_openrouter_pipe.tools.tool_registry import _build_collision_safe_tool_specs_and_registry


def _returns(result: str):
    async def tool(**_kwargs):
        return result

    return tool


def _open_webui_tools_dict(tool_ids: tuple[str, ...], function_name: str = "search") -> dict[str, dict[str, Any]]:
    """`tools_dict` as Open WebUI's `get_tools` builds it when every tool defines the same function."""
    tools_dict: dict[str, dict[str, Any]] = {}
    for tool_id in tool_ids:
        spec = {
            "name": function_name,
            "description": f"Search with {tool_id}",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        }
        key = spec["name"]
        while key in tools_dict:  # Open WebUI's collision loop renames the key only
            key = f"{tool_id}_{key}"
        tools_dict[key] = {"tool_id": tool_id, "callable": _returns(f"ran {tool_id}"), "spec": spec}
    return tools_dict


def _build(tools_dict: dict[str, dict[str, Any]], *, passthrough: bool):
    # For native function calling Open WebUI sends one spec per tool: {"type": "function", "function": spec}.
    request_tools = _chat_tools_to_responses_tools(
        [{"type": "function", "function": tool["spec"]} for tool in tools_dict.values()]
    )
    return _build_collision_safe_tool_specs_and_registry(
        request_tool_specs=request_tools,
        owui_registry=tools_dict,
        direct_registry=None,
        builtin_registry=None,
        extra_tools=None,
        strictify=False,
        owui_tool_passthrough=passthrough,
        logger=None,
    )


SAME_NAME_TOOLS = [
    (
        ("tool_a", "tool_b"),
        {"search": "tool_a", "tool_b_search": "tool_b"},
    ),
    (
        ("tool_a", "tool_b", "tool_c"),
        {"search": "tool_a", "tool_b_search": "tool_b", "tool_c_search": "tool_c"},
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool_ids", "expected"), SAME_NAME_TOOLS)
async def test_in_pipeline_mode_each_same_name_tool_is_offered_once_and_runs_its_own_code(tool_ids, expected):
    tools, exec_registry, _ = _build(_open_webui_tools_dict(tool_ids), passthrough=False)

    offered = {}
    for tool in tools:
        cfg = exec_registry.get(tool["name"])
        offered[tool["name"]] = (tool.get("description"), await cfg["callable"]() if cfg else None)

    assert offered == {
        name: (f"Search with {tool_id}", f"ran {tool_id}") for name, tool_id in expected.items()
    }


@pytest.mark.parametrize(("tool_ids", "expected"), SAME_NAME_TOOLS)
def test_in_open_webui_mode_each_same_name_tool_maps_to_the_key_open_webui_runs(tool_ids, expected):
    tools_dict = _open_webui_tools_dict(tool_ids)
    tools, _, exposed_to_origin = _build(tools_dict, passthrough=True)

    offered = {tool["name"]: (tool.get("description"), exposed_to_origin.get(tool["name"])) for tool in tools}

    assert offered == {name: (f"Search with {tool_id}", name) for name, tool_id in expected.items()}
    # Each mapped name is a key Open WebUI can run, and it runs the tool whose description the model saw.
    for description, key in offered.values():
        assert key is not None
        assert tools_dict[key]["spec"]["description"] == description


# --- names a provider accepts -----------------------------------------------------------------------------------------

# A tool-server id is free text an admin types and a local tool id can be as long as its author likes, so the keys
# Open WebUI's collision loop builds from them are not always valid function names.
PROVIDER_FUNCTION_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
KEYS_OPEN_WEBUI_BUILDS = [
    pytest.param(("tool_a", "Weather API"), "search", id="server-id-with-a-space"),
    pytest.param(("tool_a", "acme.tools"), "search", id="server-id-with-a-dot"),
    pytest.param(
        ("tool_a", "company_internal_documentation_search_tool"),
        "search_the_company_knowledge_base",
        id="key-longer-than-64",
    ),
    pytest.param(("tool_a", "Weather API", "Weather_API"), "search", id="two-keys-that-clean-up-alike"),
    pytest.param(("tool_a", "Weather API", "Weather_API", "Weather.API"), "search",
                 id="three-keys-that-clean-up-alike"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("passthrough", [False, True], ids=["pipeline", "open-webui"])
@pytest.mark.parametrize(("tool_ids", "function_name"), KEYS_OPEN_WEBUI_BUILDS)
async def test_every_offered_name_is_a_valid_function_name_that_leads_back_to_its_own_tool(
    tool_ids, function_name, passthrough
):
    tools_dict = _open_webui_tools_dict(tool_ids, function_name)
    tools, exec_registry, exposed_to_origin = _build(tools_dict, passthrough=passthrough)

    names = [tool["name"] for tool in tools]
    assert all(PROVIDER_FUNCTION_NAME.fullmatch(name) for name in names), names
    # One offered name per Open WebUI key, each mapped back to the key Open WebUI runs ...
    assert sorted(exposed_to_origin[name] for name in names) == sorted(tools_dict)
    for tool in tools:
        key = exposed_to_origin[tool["name"]]
        # ... for the tool whose description the model saw; in Pipeline mode that tool's own code runs.
        assert tools_dict[key]["spec"]["description"] == tool["description"]
        if not passthrough:
            assert await exec_registry[tool["name"]]["callable"]() == f"ran {tools_dict[key]['tool_id']}"


# A tool that shares its name with nothing can still carry a name no provider accepts: Open WebUI names an MCP tool
# `<server id>_<tool>` in both its key and its spec, and a tool server's operationId is used as it is.
NAMES_FROM_A_SINGLE_TOOL = [
    pytest.param("My MCP_search", id="mcp-server-id-with-a-space"),
    pytest.param("acme.mcp_search", id="mcp-server-id-with-a-dot"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("passthrough", [False, True], ids=["pipeline", "open-webui"])
@pytest.mark.parametrize("name", NAMES_FROM_A_SINGLE_TOOL)
async def test_a_single_tool_whose_own_name_is_not_a_valid_function_name_is_offered_under_a_valid_one(name, passthrough):
    spec = {"name": name, "description": f"Tool {name}", "parameters": {"type": "object", "properties": {}}}
    tools_dict = {name: {"tool_id": "server:mcp:acme", "callable": _returns(f"ran {name}"), "spec": spec}}

    tools, exec_registry, exposed_to_origin = _build(tools_dict, passthrough=passthrough)

    names = [tool["name"] for tool in tools]
    assert len(names) == 1 and PROVIDER_FUNCTION_NAME.fullmatch(names[0]), names
    assert exposed_to_origin[names[0]] == name
    if not passthrough:
        assert await exec_registry[names[0]]["callable"]() == f"ran {name}"


@pytest.mark.asyncio
@pytest.mark.parametrize("passthrough", [False, True], ids=["pipeline", "open-webui"])
async def test_a_tool_server_operation_whose_id_is_not_a_valid_function_name_is_offered_under_a_valid_one(passthrough):
    spec = {"name": "users.list", "description": "List users", "parameters": {"type": "object", "properties": {}}}
    direct = {"users.list": {"direct": True, "callable": _returns("listed users"), "spec": spec}}

    tools, exec_registry, exposed_to_origin = _build_collision_safe_tool_specs_and_registry(
        request_tool_specs=_chat_tools_to_responses_tools([{"type": "function", "function": spec}]),
        owui_registry=None,
        direct_registry=direct,
        builtin_registry=None,
        extra_tools=None,
        strictify=False,
        owui_tool_passthrough=passthrough,
        logger=None,
    )

    names = [tool["name"] for tool in tools]
    assert names and all(PROVIDER_FUNCTION_NAME.fullmatch(name) for name in names), names
    assert {exposed_to_origin[name] for name in names} == {"users.list"}
    if not passthrough:
        assert {await exec_registry[name]["callable"]() for name in names} == {"listed users"}


# --- building again from the pipe's own registry ----------------------------------------------------------------------


def _no_parameters() -> dict[str, Any]:
    return {"type": "object", "properties": {}}


OPEN_WEBUI_TOOLS_NAMED_LIKE_A_DIRECT_TOOL = [
    pytest.param(
        "ask_user",
        {
            "tool_id": "builtin:ask_user",
            "type": "builtin",
            "callable": _returns("open webui ask_user"),
            "spec": {"name": "ask_user", "parameters": _no_parameters()},
        },
        id="builtin-ask-user",
    ),
    pytest.param(
        "lookup",
        {"tool_id": "local_lookup", "callable": _returns("open webui lookup"), "spec": {"name": "lookup", "parameters": _no_parameters()}},
        id="local-tool",
    ),
]


@pytest.mark.parametrize(("name", "open_webui_tool"), OPEN_WEBUI_TOOLS_NAMED_LIKE_A_DIRECT_TOOL)
def test_building_again_from_the_pipes_own_registry_keeps_each_renamed_tools_origin(name, open_webui_tool):
    # Internal Fusion hands the outer request's execution registry, keyed by the names the model saw, back to the
    # builder for every panel member, together with the same direct tool servers.
    direct = {
        name: {"direct": True, "callable": _returns(f"direct {name}"), "spec": {"name": name, "parameters": _no_parameters()}}
    }

    def build(registry):
        return _build_collision_safe_tool_specs_and_registry(
            request_tool_specs=None,
            owui_registry=registry,
            direct_registry=direct,
            builtin_registry=None,
            extra_tools=None,
            strictify=False,
            owui_tool_passthrough=False,
            logger=None,
        )

    _, outer_registry, _ = build({name: open_webui_tool})
    assert name not in outer_registry, sorted(outer_registry)
    _, inner_registry, inner_map = build(outer_registry)

    assert inner_registry
    assert {cfg["origin_name"] for cfg in inner_registry.values()} == {name}
    assert {inner_map[exposed] for exposed in inner_registry} == {name}


# --- the name Open WebUI stores for a call the pipe hands back --------------------------------------------------------

RENAMED_AND_PLAIN_KEYS = [
    pytest.param("My MCP_search", id="renamed-space"),
    pytest.param("acme.mcp_search", id="renamed-dot"),
    pytest.param("search", id="not-renamed"),
]


def _one_open_webui_tool(key: str) -> dict[str, dict[str, Any]]:
    spec = {"name": key, "description": f"Tool {key}", "parameters": {"type": "object", "properties": {}}}
    return {key: {"tool_id": "server:mcp:acme", "callable": _returns(f"ran {key}"), "spec": spec}}


def _names_open_webui_stores(chunks: list[dict[str, Any]]) -> dict[str, str]:
    """Fold `chat:tool_calls` chunks as Open WebUI's streaming handler does: a call keeps the name it was first sent,
    replaced only by a later chunk that carries a non-empty name."""
    names: dict[str, str] = {}
    for chunk in chunks:
        for call in chunk["data"]["tool_calls"]:
            name = (call.get("function") or {}).get("name")
            if name:
                names[call["id"]] = name
            else:
                names.setdefault(call["id"], "")
    return names


@pytest.mark.asyncio
@pytest.mark.parametrize("argument_deltas", [True, False], ids=["argument-deltas", "completion-only"])
@pytest.mark.parametrize("key", RENAMED_AND_PLAIN_KEYS)
async def test_open_webui_stores_a_handed_back_call_under_the_key_it_runs(
    monkeypatch, pipe_instance_async, key, argument_deltas
):
    tools, _, exposed_to_origin = _build(_one_open_webui_tool(key), passthrough=True)
    call = {"type": "function_call", "id": "fc-1", "call_id": "call-1", "name": tools[0]["name"],
            "arguments": '{"q": "x"}'}
    events: list[dict[str, Any]] = [
        {"type": "response.output_item.added", "output_index": 0, "item": dict(call, arguments="", status="in_progress")},
    ]
    if argument_deltas:
        events += [
            {"type": "response.function_call_arguments.delta", "item_id": "fc-1", "output_index": 0, "delta": '{"q": '},
            {"type": "response.function_call_arguments.delta", "item_id": "fc-1", "output_index": 0, "delta": '"x"}'},
            {"type": "response.function_call_arguments.done", "item_id": "fc-1", "output_index": 0,
             "arguments": '{"q": "x"}'},
        ]
    events.append({"type": "response.completed", "response": {"output": [dict(call, status="completed")], "usage": {}}})

    async def model(self, session, request_body, **_kwargs):
        for event in events:
            yield event

    emitted: list[dict[str, Any]] = []

    async def emitter(event):
        emitted.append(event)

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", model)
    pipe = pipe_instance_async
    valves = pipe.valves.model_copy(update={"TOOL_EXECUTION_MODE": "Open-WebUI"})
    await pipe._streaming_handler._run_streaming_loop(
        ResponsesBody(model="test/model", input=[], stream=True), valves, emitter,
        metadata={"model": {"id": "test"}, "_pipe_exposed_to_origin": exposed_to_origin},
        tools={}, session=cast(Any, object()), user_id="user-1",
    )

    chunks = [event for event in emitted if event.get("type") == "chat:tool_calls"]
    assert chunks, emitted
    assert _names_open_webui_stores(chunks) == {"call-1": key}


# --- tool names on the wire when Open WebUI replays a round ------------------------------------------------------------


def _called_names(called: str | list[str]) -> list[str]:
    return [called] if isinstance(called, str) else list(called)


async def _upstream_after_open_webui_replays(
    pipe, monkeypatch, *, tools_dict: dict[str, dict[str, Any]], called: str | list[str]
) -> tuple[set[str], list[str]]:
    """Open WebUI's next call after running `called`: the names the request advertises and the replayed call names."""
    import open_webui_openrouter_pipe.pipe as pipe_mod

    sent: list[dict[str, Any]] = []

    async def model(self, session, request_body, **_kwargs):
        sent.append(request_body)
        yield {"type": "response.output_text.delta", "delta": "Done."}
        yield {"type": "response.completed", "response": {"output": [], "usage": {}}}

    async def loaded(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def emitter(_event):
        return None

    monkeypatch.setattr(Pipe, "send_openrouter_streaming_request", model)
    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test", None))
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", loaded)
    monkeypatch.setattr(
        pipe_mod.OpenRouterModelRegistry, "list_models", lambda: [{"id": "m1", "name": "Model m1", "norm_id": "m1"}]
    )
    valves = pipe.valves.model_copy(update={"TOOL_EXECUTION_MODE": "Open-WebUI"})

    await pipe._handle_pipe_call(
        {
            "stream": True,
            "model": "m1",
            "messages": [
                {"role": "user", "content": "look it up"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": f"call-{n}", "type": "function", "function": {"name": name, "arguments": "{}"}}
                    for n, name in enumerate(_called_names(called))]},
                *[{"role": "tool", "tool_call_id": f"call-{n}", "content": "found it"}
                  for n, _ in enumerate(_called_names(called))],
            ],
            "tools": [{"type": "function", "function": entry["spec"]} for entry in tools_dict.values()],
        },
        {"id": "user-1"},
        None,
        emitter,
        None,
        {"chat_id": "chat-1", "message_id": "message-1", "model": {"id": "m1"}},
        tools_dict,
        None,
        None,
        valves=valves,
        session=cast(Any, object()),
    )

    assert len(sent) == 1, sent
    advertised = {tool.get("name") for tool in sent[0].get("tools") or [] if tool.get("type") == "function"}
    replayed = [item.get("name") for item in sent[0].get("input") or [] if item.get("type") == "function_call"]
    assert replayed, sent[0].get("input")
    return advertised, replayed


@pytest.mark.asyncio
@pytest.mark.parametrize("key", RENAMED_AND_PLAIN_KEYS)
async def test_every_tool_name_sent_upstream_is_one_the_request_advertises(monkeypatch, pipe_instance_async, key):
    """In Open-WebUI mode Open WebUI stores the key it ran and sends that name back in the history of its next call."""
    advertised, replayed = await _upstream_after_open_webui_replays(
        pipe_instance_async, monkeypatch, tools_dict=_one_open_webui_tool(key), called=key
    )

    assert set(replayed) <= advertised, (replayed, sorted(advertised))


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ["My MCP_search", "acme.mcp_search"])
async def test_a_replayed_tool_name_the_request_no_longer_offers_reaches_the_wire_as_a_valid_name(
    monkeypatch, pipe_instance_async, key
):
    """The tool that ran was disabled or removed since, so the request advertises only another tool; the history still
    names the key Open WebUI ran."""
    advertised, replayed = await _upstream_after_open_webui_replays(
        pipe_instance_async, monkeypatch, tools_dict=_one_open_webui_tool("lookup"), called=key
    )

    assert key not in advertised
    assert all(name in advertised or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) for name in replayed), (
        replayed,
        sorted(advertised),
    )


@pytest.mark.asyncio
async def test_two_removed_tools_replayed_in_one_history_keep_two_different_names(monkeypatch, pipe_instance_async):
    """Renaming a replayed call must keep two different tools apart.

    A history can name more than one tool the request no longer offers. If both are rewritten to the
    same name, the model reads its own past as having called one tool twice, and a support trace can no
    longer be followed back to the key that ran. A constant satisfies "matches the provider's name
    pattern"; two names that must differ does not.
    """
    advertised, replayed = await _upstream_after_open_webui_replays(
        pipe_instance_async, monkeypatch, tools_dict=_one_open_webui_tool("lookup"),
        called=["My MCP_search", "acme.mcp_search"],
    )

    assert len(replayed) == 2, replayed
    assert len(set(replayed)) == 2, replayed
    assert not set(replayed) & advertised, (replayed, sorted(advertised))
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) for name in replayed), replayed


@pytest.mark.asyncio
async def test_a_replayed_name_that_is_already_valid_is_left_alone(monkeypatch, pipe_instance_async):
    """A key that is already a usable function name is not mangled: a blanket rewrite would break a trace
    that is perfectly good, and the name the provider sees would stop matching the one Open WebUI ran."""
    advertised, replayed = await _upstream_after_open_webui_replays(
        pipe_instance_async, monkeypatch, tools_dict=_one_open_webui_tool("lookup"), called="search",
    )

    assert replayed == ["search"], (replayed, sorted(advertised))


def test_a_replayed_name_two_advertised_tools_could_mean_is_not_rewritten_to_either():
    """When two advertised tools share one origin, the pipe cannot tell which of them ran.

    Rewriting the replayed call to one of them would tell the provider - and the model reading its own
    history - that a specific tool ran, chosen by nothing better than dictionary order. The name is
    cleaned into something the provider accepts instead, and deliberately left out of the advertised
    set.
    """
    from open_webui_openrouter_pipe.tools.tool_registry import _advertised_names_for_replayed_calls

    exposed_to_origin = {
        "tool__My_MCP_search__2ae282af": "My MCP_search",
        "direct__My_MCP_search__c552683f": "My MCP_search",
        "lookup": "lookup",
    }
    items = [{"type": "function_call", "call_id": "c1", "name": "My MCP_search", "arguments": "{}"}]

    _advertised_names_for_replayed_calls(items, exposed_to_origin)

    assert items[0]["name"] not in exposed_to_origin, items[0]["name"]
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", items[0]["name"]), items[0]["name"]


@pytest.mark.parametrize("origin", ["x", "y" * 70], ids=["short-name", "name-over-the-limit"])
def test_every_candidate_keeps_a_name_of_its_own_even_when_three_share_one(origin):
    """The builder is named for collision safety, so no candidate may lose its name to another.

    Two same-named candidates are fine: the first keeps the plain name and the second takes the digest.
    A third produces the same digest as the second, because the digest is computed from the origin and
    all three share it - so the later entry overwrites the earlier one in the execution registry and the
    tool simply disappears, advertised to nobody and callable by no one.
    """
    from open_webui_openrouter_pipe.tools.tool_registry import _provider_tool_name

    used: set[str] = set()
    names = []
    for _ in range(3):
        name = _provider_tool_name(origin, "deadbeef", used)
        used.add(name)
        names.append(name)

    assert len(set(names)) == 3, names
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name) for name in names), names


def test_a_name_that_is_already_usable_is_returned_unchanged():
    """The disambiguator must not touch a name that needs nothing: only a clash or an unusable
    character earns a digest."""
    from open_webui_openrouter_pipe.tools.tool_registry import _provider_tool_name

    assert _provider_tool_name("ok_name", "deadbeef", set()) == "ok_name"
    assert len(_provider_tool_name("x" * 70, "deadbeef", set())) <= 64


# --- request specs the pipe has no tool for ----------------------------------------------------------------------

# Prescription 125 asked for a registry-level arm built from "three request specs named `x`". It is writable: in
# Open-WebUI mode the caller's specs are passed through and the collision loop names them apart. An earlier
# deviation claimed it could not be written because such specs were dropped in Pipeline mode; since the tool
# hand-back landed they are offered in both modes and the collision loop names them apart in both. Both modes are
# pinned here.
REQUEST_SPECS_WITH_NO_TOOL = [
    pytest.param(("x", "x", "x"), id="three-request-specs-sharing-one-name"),
    pytest.param(("alpha", "beta"), id="two-request-specs-with-distinct-names"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("names", REQUEST_SPECS_WITH_NO_TOOL)
async def test_request_specs_open_webui_will_run_keep_one_offered_name_each(names):
    """In Open-WebUI mode the caller's own specs go to the model -- Open WebUI runs them, not the pipe. Each gets
    one valid offered name of its own even when they share a name, and each maps back to the name Open WebUI
    knows it by. Nothing is executable here, which is the point: the pipe is not the one running them."""
    tools, exec_registry, exposed_to_origin = _build_request_only(names, passthrough=True)

    offered = [tool["name"] for tool in tools]
    assert len(offered) == len(names), offered
    assert len(set(offered)) == len(names), offered
    assert all(PROVIDER_FUNCTION_NAME.fullmatch(name) for name in offered), offered
    assert len(exposed_to_origin) == len(names), exposed_to_origin
    assert sorted(exposed_to_origin[name] for name in offered) == sorted(names), exposed_to_origin
    assert exec_registry == {}, exec_registry


def _build_request_only(names, *, passthrough: bool):
    specs = [
        {"name": name, "description": f"Tool {name}", "parameters": {"type": "object", "properties": {}}}
        for name in names
    ]
    return _build_collision_safe_tool_specs_and_registry(
        request_tool_specs=_chat_tools_to_responses_tools(
            [{"type": "function", "function": spec} for spec in specs]
        ),
        owui_registry=None,
        direct_registry=None,
        builtin_registry=None,
        extra_tools=None,
        strictify=False,
        owui_tool_passthrough=passthrough,
        logger=None,
    )
