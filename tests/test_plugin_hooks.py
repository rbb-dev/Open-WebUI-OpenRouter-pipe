"""Task-0 tests: the three neutral observation hooks, request_id plumbing,
and the shared table_suffix helper."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any
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
