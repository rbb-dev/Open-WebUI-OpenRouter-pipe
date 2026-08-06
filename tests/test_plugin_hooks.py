"""Task-0 tests: the three neutral observation hooks, request_id plumbing,
and the shared table_suffix helper."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.plugins.base import PluginBase
from open_webui_openrouter_pipe.plugins.registry import _PR_SUBSCRIBABLE_HOOKS, PluginRegistry
from open_webui_openrouter_pipe.storage.persistence import ArtifactStore, _sanitize_table_fragment
from open_webui_openrouter_pipe.tools.tool_executor import _ToolExecutionContext


def _registry_with(plugin: Any, hooks: list[str]) -> PluginRegistry:
    reg = PluginRegistry()
    reg._plugins = [plugin]
    for hook in hooks:
        reg._hook_subscribers[hook] = [(plugin, 50)]
    return reg


class _Recorder(PluginBase):
    plugin_id = "recorder"
    plugin_name = "Recorder"
    plugin_version = "0"

    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []

    async def on_tool_result(self, tool_name, status, **kwargs):
        self.calls.append(("tool", (tool_name, status), kwargs))

    async def on_request_retry(self, kind, **kwargs):
        self.calls.append(("retry", (kind,), kwargs))

    async def on_generation_complete(self, usage, status, **kwargs):
        self.calls.append(("gen", (usage, status), kwargs))


def test_no_duplicate_turn_total_cost_snapshot():
    """Regression: the per-generation cost snapshot is the SOLE snapshot per
    turn. A second turn-final snapshot double-counts every request in the
    costs:* keyspace — it must not exist."""
    from pathlib import Path

    src = Path("open_webui_openrouter_pipe/streaming/streaming_core.py").read_text()
    assert "turn_total" not in src
    assert src.count("maybe_dump_costs_snapshot(") == 1


def test_new_hooks_are_subscribable():
    assert {"on_tool_result", "on_request_retry", "on_generation_complete"} <= _PR_SUBSCRIBABLE_HOOKS


@pytest.mark.asyncio
async def test_new_dispatchers_noop_on_empty_registry():
    reg = PluginRegistry()
    await reg.dispatch_on_tool_result("t", "completed")
    await reg.dispatch_on_request_retry("reasoning")
    await reg.dispatch_on_generation_complete({"cost": 1}, "ok")


@pytest.mark.asyncio
async def test_new_dispatchers_reach_subscriber_with_kwargs():
    plugin = _Recorder()
    reg = _registry_with(plugin, ["on_tool_result", "on_request_retry", "on_generation_complete"])

    await reg.dispatch_on_tool_result("web_search", "failed", request_id="r1", metadata={"chat_id": "c"})
    await reg.dispatch_on_request_retry("signature", request_id="r1")
    await reg.dispatch_on_generation_complete({"cost": 0.5}, "cancelled", request_id="r1", task=None)

    kinds = [c[0] for c in plugin.calls]
    assert kinds == ["tool", "retry", "gen"]
    assert plugin.calls[0][1] == ("web_search", "failed")
    assert plugin.calls[0][2]["request_id"] == "r1"
    assert plugin.calls[1][1] == ("signature",)
    assert plugin.calls[2][1] == ({"cost": 0.5}, "cancelled")


@pytest.mark.asyncio
async def test_failing_subscriber_is_isolated():
    bad = Mock()
    bad.plugin_id = "bad"
    bad.on_tool_result = Mock(side_effect=RuntimeError("boom"))
    reg = _registry_with(bad, ["on_tool_result"])
    await reg.dispatch_on_tool_result("t", "completed")
    assert bad.on_tool_result.called


@pytest.mark.asyncio
async def test_dispatch_on_request_passes_request_id():
    plugin = Mock()
    plugin.plugin_id = "p"
    plugin.on_request = AsyncMock(return_value=None)
    reg = _registry_with(plugin, ["on_request"])
    await reg.dispatch_on_request({}, {}, {}, None, None, request_id="rid-9")
    assert plugin.on_request.call_args.kwargs["request_id"] == "rid-9"


@pytest.mark.asyncio
async def test_pipe_dispatch_plugin_event_guards():
    pipe = Pipe()
    assert pipe._plugin_registry is None
    await pipe._dispatch_plugin_event("dispatch_on_tool_result", "t", "completed", request_id="r")

    recorder = _Recorder()
    pipe._plugin_registry = _registry_with(recorder, ["on_tool_result"])
    pipe.valves.ENABLE_PLUGIN_SYSTEM = False
    await pipe._dispatch_plugin_event("dispatch_on_tool_result", "t", "completed", request_id="r")
    assert recorder.calls == []

    pipe.valves.ENABLE_PLUGIN_SYSTEM = True
    await pipe._dispatch_plugin_event("dispatch_on_tool_result", "t", "completed", request_id="r")
    assert recorder.calls and recorder.calls[0][0] == "tool"

    await pipe._dispatch_plugin_event("dispatch_missing_method")
    await pipe.close()


def test_tool_context_carries_request_id():
    import asyncio as _a

    ctx = _ToolExecutionContext(
        queue=Mock(),
        per_request_semaphore=Mock(spec=_a.Semaphore),
        global_semaphore=None,
        timeout=1.0,
        batch_timeout=None,
        idle_timeout=None,
        user_id="u",
        event_emitter=None,
        batch_cap=1,
        request_id="req-42",
    )
    assert ctx.request_id == "req-42"
    default_ctx = _ToolExecutionContext(
        queue=Mock(),
        per_request_semaphore=Mock(spec=_a.Semaphore),
        global_semaphore=None,
        timeout=1.0,
        batch_timeout=None,
        idle_timeout=None,
        user_id="u",
        event_emitter=None,
        batch_cap=1,
    )
    assert default_ctx.request_id == ""


def test_table_suffix_matches_artifact_formula():
    host: Any = SimpleNamespace(_encryption_key="k-secret", id="open_webui_openrouter_pipe")
    suffix = ArtifactStore.table_suffix(host)
    key_hash = hashlib.sha256("k-secretopen_webui_openrouter_pipe".encode("utf-8", "ignore")).hexdigest()
    expected = f"{_sanitize_table_fragment('open_webui_openrouter_pipe')}_{key_hash[:8]}"
    assert suffix == expected
    assert f"response_items_{suffix}".startswith("response_items_open_webui_openrouter_pipe_")


def test_table_suffix_requires_identifier():
    host: Any = SimpleNamespace(_encryption_key="", id="")
    with pytest.raises(RuntimeError):
        ArtifactStore.table_suffix(host)


@pytest.mark.asyncio
async def test_a_repeatedly_failing_plugin_hook_is_reported_once(caplog):
    """`_dispatch_plugin_event` runs per tool call, per generation, per request.

    It logged a full traceback at WARNING on every failure with no latch, while a path
    that runs *less* often -- the outbound user-header stamp, once per API call -- was
    given one in the same changeset. A plugin hook that raises consistently therefore
    produced one traceback per tool call, for the life of the worker.

    The latch keys on method plus exception type, so a different failure in a different
    hook still gets its own first report.
    """
    import logging

    from open_webui_openrouter_pipe import pipe as pipe_module

    pipe = Pipe()
    pipe.valves.ENABLE_PLUGIN_SYSTEM = True

    class _Exploding:
        async def on_tool_result(self, *_args, **_kwargs):
            raise RuntimeError("hook is broken")

        async def on_models(self, *_args, **_kwargs):
            raise ValueError("different failure")

    cast(Any, pipe)._plugin_registry = _Exploding()
    pipe_module._warned_plugin_dispatch.clear()

    def _warnings():
        return [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "dispatch failed" in r.getMessage()
        ]

    try:
        with caplog.at_level(logging.DEBUG, logger=pipe.logger.name):
            for _ in range(5):
                await pipe._dispatch_plugin_event("on_tool_result")
            after_five = list(_warnings())

            await pipe._dispatch_plugin_event("on_models")
            after_other = list(_warnings())
    finally:
        pipe_module._warned_plugin_dispatch.clear()
        cast(Any, pipe)._plugin_registry = None
        await pipe.close()

    assert len(after_five) == 1, (
        f"five failing dispatches of the same hook produced {len(after_five)} warnings "
        "with tracebacks. This runs once per tool call, so a broken plugin floods the "
        "operator's log indefinitely."
    )
    repeats = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG and "dispatch failed" in r.getMessage()
    ]
    assert len(repeats) == 4, (
        f"the four repeat occurrences produced {len(repeats)} DEBUG records. A latch "
        "with no repeat path is a net loss: before latching, these logged on every "
        "occurrence, so an operator raising LOG_LEVEL to DEBUG to diagnose a recurring "
        "fault now sees nothing at any level."
    )
    assert len(after_other) == 2, (
        f"a different hook failing a different way produced {len(after_other) - 1} new "
        "warnings; each distinct cause must still get its own first report"
    )
    assert after_five[0].exc_info is not None, (
        "the first report carries no traceback, so it says a hook failed without "
        "saying why"
    )


def _pipes_maintenance_functions() -> set[str]:
    """The enclosing function of every `warn_level(_warned_pipes_maintenance, ...)` call.

    Used to attribute log records to the sites under test. Without it the record filter
    also picks up unrelated warnings from the same run, and a site that emits NOTHING is
    indistinguishable from one that emitted once -- which is how a demotion to DEBUG
    passed a test named "reported once".
    """
    import ast
    import inspect

    from open_webui_openrouter_pipe import pipe as pipe_module

    tree = ast.parse(inspect.getsource(pipe_module))
    names: set[str] = set()
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for node in ast.walk(fn):
            if (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "warn_level"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "_warned_pipes_maintenance"
            ):
                names.add(fn.name)
                break
    return names


def _pipes_maintenance_causes() -> tuple[set[str], list[str]]:
    """Every `warn_level(_warned_pipes_maintenance, ...)` cause PREFIX, from pipe.py's AST.

    Discovered, not counted. Most keys are f-strings of the form `"<site>:{type(exc)}"`,
    so the stable part is the literal before the colon; `on_models` is a plain string.
    Shared with the two sibling censuses, which matched only string constants and so
    could not see this module's dominant idiom at all.
    """
    from open_webui_openrouter_pipe import pipe as pipe_module

    from tests.warn_latch_census import warn_level_causes

    return warn_level_causes(pipe_module, "_warned_pipes_maintenance")


@pytest.mark.asyncio
async def test_a_failing_filter_install_is_reported_once_across_repeated_pipes_calls(caplog):
    """`pipes()` runs on every /api/models request, not once at startup.

    Open WebUI's ENABLE_BASE_MODELS_CACHE defaults to False, so get_all_models ->
    get_function_models -> pipes() executes per request. This changeset promoted ten
    sites inside pipes() from debug to warning-with-traceback; unlatched, a read-only
    DB or a stale filter row emits ten tracebacks per model-list fetch, forever.

    A sibling forty lines above -- _dispatch_plugin_event -- was given a latch in the
    same changeset for exactly this reason, and these were missed.
    """
    import logging

    from aioresponses import aioresponses

    from open_webui_openrouter_pipe import pipe as pipe_module
    from open_webui_openrouter_pipe.core.config import EncryptedStr

    pipe = Pipe()
    pipe.valves.API_KEY = EncryptedStr("sk-test-key")
    pipe.valves.AUTO_INSTALL_WEB_TOOLS_FILTER = True
    pipe.valves.AUTO_INSTALL_IMAGE_GEN_FILTER = True
    pipe.valves.ENABLE_IMAGE_GENERATION = True
    pipe.valves.AUTO_INSTALL_VIDEO_FILTERS = True
    pipe.valves.ENABLE_VIDEO_GENERATION = True
    pipe.valves.AUTO_INSTALL_DIRECT_UPLOADS_FILTER = True
    pipe.valves.ADMIN_PROVIDER_ROUTING_MODELS = "openrouter/test"
    pipe.valves.ZDR_MODELS_ONLY = True
    # A model id and a variant base that the catalog does not hold: both are STABLE
    # misconfigurations, so unlatched they warned on every /api/models request for
    # the life of the worker -- the variant one once per bad entry.
    pipe.valves.MODEL_ID = "does/not-exist"
    pipe.valves.VARIANT_MODELS = "does/not-exist:nitro"

    def _explode(*_args, **_kwargs):
        raise RuntimeError("filter table is read-only")

    cast(Any, pipe)._ensure_filter_manager = _explode

    # The three sites the filter-manager stub alone never reaches. Without these the
    # cluster assertion below was a `>= 4` floor over ten sites, and provider_routing,
    # stale_prune and on_models had their lines executed by zero tests -- a broken
    # plugin on_models hook could go back to being completely silent, green.
    pipe.valves.ENABLE_PLUGIN_SYSTEM = True
    cast(Any, pipe)._ensure_plugin_registry = _explode

    catalog_mgr = pipe._ensure_catalog_manager()
    cast(Any, catalog_mgr).get_cached_provider_map = lambda: {"openrouter/test": ["a"]}

    async def _explode_async(*_args, **_kwargs):
        raise RuntimeError("catalog table is read-only")

    cast(Any, catalog_mgr).prune_stale_openrouter_filter_ids = _explode_async
    cast(Any, pipe)._stale_filter_ids_pruned = False

    pipe_module._warned_pipes_maintenance.clear()

    maintenance_functions = _pipes_maintenance_functions()

    def _warnings():
        return [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and r.funcName in maintenance_functions
        ]

    try:
        with caplog.at_level(logging.DEBUG, logger=pipe.logger.name):
            with aioresponses() as http:
                http.get(
                    "https://openrouter.ai/api/v1/models",
                    payload={
                        "data": [
                            {
                                "id": "openrouter/test",
                                "name": "T",
                                "supported_parameters": ["tools", "tool_choice"],
                                "architecture": {
                                    "input_modalities": ["text"],
                                    "output_modalities": ["text"],
                                },
                                "pricing": {"prompt": "0", "completion": "0"},
                                "context_length": 8192,
                            }
                        ]
                    },
                    repeat=True,
                )
                http.get(
                    "https://openrouter.ai/api/v1/endpoints/zdr",
                    exception=RuntimeError("ZDR endpoint down"),
                    repeat=True,
                )
                for _ in range(3):
                    await pipe.pipes()

            # Second phase: the catalog pair needs the fetch to FAIL, which the phase
            # above deliberately keeps healthy so the later sites are reachable at all.
            # Same latch, same no-repeat rule, so it belongs in the same drive.
            # The refresh window has to be zeroed or phase 1's cache means no refetch
            # happens and the error branch is never entered.
            pipe.valves.MODEL_CATALOG_REFRESH_SECONDS = 0
            with aioresponses() as http:
                http.get(
                    "https://openrouter.ai/api/v1/models",
                    exception=RuntimeError("catalog endpoint down"),
                    repeat=True,
                )
                http.get(
                    "https://openrouter.ai/api/v1/endpoints/zdr",
                    exception=RuntimeError("ZDR endpoint down"),
                    repeat=True,
                )
                for _ in range(3):
                    await pipe.pipes()

            emitted = _warnings()
            armed = {c.split(":", 1)[0] for c in pipe_module._warned_pipes_maintenance}
    finally:
        pipe_module._warned_pipes_maintenance.clear()
        await pipe.close()

    per_cause: dict[str, int] = {}
    for record in emitted:
        per_cause[record.getMessage()[:40]] = per_cause.get(record.getMessage()[:40], 0) + 1
    # Two-sided. `repeated` alone is an upper bound, and zero warnings satisfies it: a
    # site that logs at DEBUG still arms the latch, so `armed` and the AST scan both
    # stay green while the operator sees nothing at any level they would look at.
    assert len(emitted) == len(armed), (
        f"{len(armed)} causes armed a latch but {len(emitted)} warnings were emitted "
        f"({sorted(armed)}). Fewer means a site reports nothing an operator sees; more "
        "means a latch is not holding."
    )
    repeated = {k: v for k, v in per_cause.items() if v > 1}
    assert not repeated, (
        f"three pipes() calls emitted these warnings more than once: {repeated}. "
        "pipes() runs per /api/models request, so a stable configuration fault would "
        "repeat for the life of the worker."
    )
    # The two catalog sites need the chat-catalog REFRESH to fail, which this drive
    # cannot produce: phase 1 has to populate the registry for the later sites to be
    # reachable at all, and a populated registry is served from cache. They are covered
    # by no test today -- named here rather than dropped from the inventory, so the gap
    # is visible in the failure message instead of being an absence nobody can see.
    not_driven_here = {
        "catalog_refresh",
        "catalog_cached",
        # The chat path, which this drive never enters -- it calls pipes() only. Named
        # here rather than given a driver written to satisfy the census: a drive that
        # exists only to arm a latch passes for the wrong reason, and this file has
        # already paid for that twice.
        "enforcement_base_missing",
        "enforcement_base_unnormalized",
        "chat_catalog_refresh",
    }

    # Frozen, NOT discovered. An inventory read from the same source being mutated
    # self-heals: deleting a diagnostic removes it from the expected set as well as the
    # observed one, so the check passes and proves nothing. Discovery is still used --
    # below -- to catch a site ADDED without a driver, which a literal cannot see.
    expected_sites = {
        "catalog_refresh", "catalog_cached", "web_tools", "fusion", "image_gen",
        "video", "direct_uploads", "provider_routing", "stale_prune", "on_models",
        "zdr_list_unavailable", "models_missing", "variant_base_missing",
        "enforcement_base_missing", "enforcement_base_unnormalized",
        "chat_catalog_refresh",
    }
    from tests.warn_latch_census import UNRESOLVABLE_MESSAGE

    discovered, unresolvable = _pipes_maintenance_causes()
    assert not unresolvable, UNRESOLVABLE_MESSAGE + "\n  ".join(unresolvable)
    assert discovered == expected_sites, (
        f"the pipes() maintenance sites changed: only in source {sorted(discovered - expected_sites)}, "
        f"only in this list {sorted(expected_sites - discovered)}. Update it deliberately -- "
        "a new site needs a driver, and a deleted diagnostic needs to be intentional."
    )
    # Both directions, like the sibling censuses. An exemption is one line and always
    # locally justified, while removing one needs somebody to NOTICE it went stale --
    # so left unchecked this list only ever grows. If a later drive arms one of these,
    # the entry stops exempting anything and has to go.
    inert = sorted(not_driven_here & armed)
    assert not inert, (
        f"these exemptions no longer exempt anything -- the drive now arms them: "
        f"{inert}. Remove them from not_driven_here so the census covers them again."
    )

    missing = sorted(expected_sites - armed - not_driven_here)
    assert not missing, (
        f"these pipes() maintenance sites were never driven, so deleting their "
        f"diagnostic is invisible: {missing}. The floor this replaced was `>= 4` over "
        "ten sites, and three of them were executed by no test at all."
    )
