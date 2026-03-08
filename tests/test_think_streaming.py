"""Tests for the Think Streaming plugin components."""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, Mock, patch

import pytest

from open_webui_openrouter_pipe.plugins.pipe_stats.ephemeral_keys import EphemeralKeyStore
from open_webui_openrouter_pipe.plugins.think_streaming.session import (
    SessionRegistry,
    ThinkSession,
)
from open_webui_openrouter_pipe.plugins.think_streaming.wrapper import (
    ThinkStreamingEmitterWrapper,
)


# ── ThinkSession Tests ──


class TestThinkSession:
    def test_default_values(self):
        session = ThinkSession(key="test-key")
        assert session.key == "test-key"
        assert session.consumer_alive is True
        assert session.user_id == ""
        assert isinstance(session.queue, asyncio.Queue)

    def test_queue_has_bounded_size(self):
        session = ThinkSession(key="test-key")
        # Default queue maxsize should be > 0
        assert session.queue.maxsize > 0


# ── SessionRegistry Tests ──


class TestSessionRegistry:
    def test_create_and_get(self):
        registry = SessionRegistry()
        session = registry.create("key1", user_id="user-a")
        assert session.key == "key1"
        assert session.user_id == "user-a"
        assert registry.get("key1") is session

    def test_get_nonexistent_returns_none(self):
        registry = SessionRegistry()
        assert registry.get("nonexistent") is None

    def test_remove(self):
        registry = SessionRegistry()
        registry.create("key1")
        registry.remove("key1")
        assert registry.get("key1") is None

    def test_remove_nonexistent_is_noop(self):
        registry = SessionRegistry()
        registry.remove("nonexistent")  # Should not raise

    def test_active_count(self):
        registry = SessionRegistry()
        assert registry.active_count == 0
        registry.create("k1")
        assert registry.active_count == 1
        registry.create("k2")
        assert registry.active_count == 2
        registry.remove("k1")
        assert registry.active_count == 1

    def test_capacity_eviction(self):
        registry = SessionRegistry(max_sessions=3)
        s1 = registry.create("k1")
        registry.create("k2")
        registry.create("k3")
        assert registry.active_count == 3

        # Creating a 4th should evict the oldest (k1)
        registry.create("k4")
        assert registry.active_count == 3
        assert registry.get("k1") is None
        assert registry.get("k4") is not None

    def test_ttl_cleanup(self):
        registry = SessionRegistry(ttl=1.0)
        registry.create("k1")
        registry.create("k2")
        # Manually backdate k1
        registry._sessions["k1"].created_at = time.monotonic() - 10
        registry.cleanup_expired()
        assert registry.get("k1") is None
        assert registry.get("k2") is not None


# ── ThinkStreamingEmitterWrapper Tests ──


class TestThinkStreamingEmitterWrapper:
    @pytest.mark.asyncio
    async def test_reasoning_delta_copied_and_passed_through(self):
        """reasoning:delta events should go to BOTH queue AND original."""
        original = AsyncMock()
        session = ThinkSession(key="test")
        wrapper = ThinkStreamingEmitterWrapper(original, session)

        event = {
            "type": "reasoning:delta",
            "data": {"content": "Let me think", "delta": "think", "event": "content_block_delta"},
        }
        await wrapper(event)

        original.assert_called_once_with(event)
        assert session.queue.qsize() == 1
        item = session.queue.get_nowait()
        assert item is not None
        parsed = json.loads(item)
        assert parsed["type"] == "thinking_delta"
        assert parsed["delta"] == "think"
        assert parsed["content"] == "Let me think"

    @pytest.mark.asyncio
    async def test_reasoning_completed_copied_and_passed_through(self):
        """reasoning:completed events should go to BOTH queue AND original."""
        original = AsyncMock()
        session = ThinkSession(key="test")
        wrapper = ThinkStreamingEmitterWrapper(original, session)

        event = {
            "type": "reasoning:completed",
            "data": {"content": "Full reasoning text"},
        }
        await wrapper(event)

        original.assert_called_once_with(event)
        item = session.queue.get_nowait()
        assert item is not None
        parsed = json.loads(item)
        assert parsed["type"] == "thinking_done"
        assert parsed["content"] == "Full reasoning text"

    @pytest.mark.asyncio
    async def test_non_thinking_events_pass_through(self):
        """Regular events (text deltas, completions) should pass through."""
        original = AsyncMock()
        session = ThinkSession(key="test")
        wrapper = ThinkStreamingEmitterWrapper(original, session)

        event = {"type": "chat:message:delta", "data": {"content": "Hello"}}
        await wrapper(event)

        original.assert_called_once_with(event)
        assert session.queue.empty()

    @pytest.mark.asyncio
    async def test_chat_completion_passes_through(self):
        """chat:completion should pass through to original."""
        original = AsyncMock()
        session = ThinkSession(key="test")
        wrapper = ThinkStreamingEmitterWrapper(original, session)

        event = {"type": "chat:completion", "data": {"done": True, "content": "done"}}
        await wrapper(event)

        original.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_non_dict_events_pass_through(self):
        """Non-dict events should pass through unchanged."""
        original = AsyncMock()
        session = ThinkSession(key="test")
        wrapper = ThinkStreamingEmitterWrapper(original, session)

        await wrapper("some string event")
        original.assert_called_once_with("some string event")

    @pytest.mark.asyncio
    async def test_consumer_disconnect_skips_queue_but_passes_through(self):
        """When consumer_alive=False, events skip queue but still pass to original."""
        original = AsyncMock()
        session = ThinkSession(key="test")
        session.consumer_alive = False
        wrapper = ThinkStreamingEmitterWrapper(original, session)

        event = {
            "type": "reasoning:delta",
            "data": {"content": "test", "delta": "t", "event": "delta"},
        }
        await wrapper(event)

        # Should NOT be in queue (consumer dead)
        assert session.queue.empty()
        # Should still pass through to original emitter
        original.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_queue_full_drops_silently_but_passes_through(self):
        """When queue is full, queue push is dropped but event passes to original."""
        original = AsyncMock()
        session = ThinkSession(key="test")
        # Fill the queue
        for i in range(session.queue.maxsize):
            session.queue.put_nowait(f"item-{i}")

        wrapper = ThinkStreamingEmitterWrapper(original, session)
        event = {
            "type": "reasoning:delta",
            "data": {"content": "overflow", "delta": "o", "event": "delta"},
        }
        # Should not raise
        await wrapper(event)
        # Event still passes through to original even if queue is full
        original.assert_called_once_with(event)

    def test_think_streaming_active_flag(self):
        """Wrapper class should have _think_streaming_active = True."""
        assert ThinkStreamingEmitterWrapper._think_streaming_active is True

    def test_getattr_delegates_to_original(self):
        """__getattr__ should delegate to the original emitter."""
        original = AsyncMock()
        original.some_custom_attr = "custom_value"
        session = ThinkSession(key="test")
        wrapper = ThinkStreamingEmitterWrapper(original, session)

        assert wrapper.some_custom_attr == "custom_value"

    def test_simplify_reasoning_delta(self):
        event = {
            "type": "reasoning:delta",
            "data": {"content": "buffer", "delta": "d"},
        }
        result = ThinkStreamingEmitterWrapper._simplify(event)
        assert result == {"type": "thinking_delta", "delta": "d", "content": "buffer"}

    def test_simplify_reasoning_completed(self):
        event = {
            "type": "reasoning:completed",
            "data": {"content": "full text"},
        }
        result = ThinkStreamingEmitterWrapper._simplify(event)
        assert result == {"type": "thinking_done", "content": "full text"}

    def test_simplify_unknown_type(self):
        result = ThinkStreamingEmitterWrapper._simplify({"type": "something"})
        assert result == {"type": "unknown"}

    @pytest.mark.asyncio
    async def test_tool_function_call_copied_and_passed_through(self):
        """response.output_item.added with function_call should go to queue AND original."""
        original = AsyncMock()
        session = ThinkSession(key="test")
        wrapper = ThinkStreamingEmitterWrapper(original, session)

        event = {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "name": "exa_search",
                "call_id": "call_123",
                "arguments": '{"query": "test"}',
                "status": "in_progress",
            },
        }
        await wrapper(event)

        original.assert_called_once_with(event)
        assert session.queue.qsize() == 1
        item = session.queue.get_nowait()
        assert item is not None
        parsed = json.loads(item)
        assert parsed["type"] == "tool_start"
        assert parsed["name"] == "exa_search"
        assert parsed["call_id"] == "call_123"
        assert parsed["arguments"] == '{"query": "test"}'

    @pytest.mark.asyncio
    async def test_tool_function_call_output_copied_and_passed_through(self):
        """response.output_item.added with function_call_output should go to queue AND original."""
        original = AsyncMock()
        session = ThinkSession(key="test")
        wrapper = ThinkStreamingEmitterWrapper(original, session)

        event = {
            "type": "response.output_item.added",
            "item": {
                "type": "function_call_output",
                "call_id": "call_123",
                "status": "completed",
                "output": "search results here",
            },
        }
        await wrapper(event)

        original.assert_called_once_with(event)
        item = session.queue.get_nowait()
        assert item is not None
        parsed = json.loads(item)
        assert parsed["type"] == "tool_done"
        assert parsed["call_id"] == "call_123"
        assert parsed["status"] == "completed"
        assert parsed["output"] == "search results here"

    @pytest.mark.asyncio
    async def test_non_tool_output_item_passes_through(self):
        """response.output_item.added with message type should pass through."""
        original = AsyncMock()
        session = ThinkSession(key="test")
        wrapper = ThinkStreamingEmitterWrapper(original, session)

        event = {
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "msg_1"},
        }
        await wrapper(event)

        original.assert_called_once_with(event)
        assert session.queue.empty()

    def test_simplify_tool_start(self):
        event = {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "name": "exa", "call_id": "c1"},
        }
        result = ThinkStreamingEmitterWrapper._simplify(event)
        assert result == {"type": "tool_start", "name": "exa", "call_id": "c1"}

    def test_simplify_tool_start_with_arguments(self):
        event = {
            "type": "response.output_item.added",
            "item": {"type": "function_call", "name": "exa", "call_id": "c1", "arguments": '{"q":"test"}'},
        }
        result = ThinkStreamingEmitterWrapper._simplify(event)
        assert result["arguments"] == '{"q":"test"}'

    def test_simplify_tool_done(self):
        event = {
            "type": "response.output_item.added",
            "item": {"type": "function_call_output", "call_id": "c1", "status": "completed"},
        }
        result = ThinkStreamingEmitterWrapper._simplify(event)
        assert result == {"type": "tool_done", "call_id": "c1", "status": "completed"}

    def test_simplify_tool_done_with_output(self):
        event = {
            "type": "response.output_item.added",
            "item": {"type": "function_call_output", "call_id": "c1", "status": "completed", "output": "result data"},
        }
        result = ThinkStreamingEmitterWrapper._simplify(event)
        assert result["output"] == "result data"

    def test_simplify_tool_done_output_capped(self):
        """Output longer than 2000 chars should be truncated."""
        long_output = "x" * 3000
        event = {
            "type": "response.output_item.added",
            "item": {"type": "function_call_output", "call_id": "c1", "status": "completed", "output": long_output},
        }
        result = ThinkStreamingEmitterWrapper._simplify(event)
        assert len(result["output"]) == 2000


# ── EphemeralKeyStore max_keys parameter Tests ──


class TestEphemeralKeyStoreMaxKeys:
    def test_custom_max_keys(self):
        store = EphemeralKeyStore(max_keys=5)
        keys = [store.generate() for _ in range(5)]
        assert store.active_count == 5

        # 6th key should evict oldest
        new_key = store.generate()
        assert store.active_count == 5
        assert store.validate(new_key) is True
        assert store.validate(keys[0]) is False

    def test_default_max_keys_unchanged(self):
        """Default max_keys should still be 10 (backward compat)."""
        store = EphemeralKeyStore()
        assert store._max_keys == 10

    def test_large_max_keys(self):
        store = EphemeralKeyStore(max_keys=100)
        keys = [store.generate() for _ in range(50)]
        assert store.active_count == 50
        # No eviction yet
        for k in keys:
            assert store.validate(k) is True


# ── Plugin Registry on_emitter_wrap Tests ──


class TestPluginRegistryEmitterWrap:
    def test_on_emitter_wrap_in_subscribable_hooks(self):
        """on_emitter_wrap should be in the subscribable hooks set."""
        from open_webui_openrouter_pipe.plugins.registry import _PR_SUBSCRIBABLE_HOOKS
        assert "on_emitter_wrap" in _PR_SUBSCRIBABLE_HOOKS

    @pytest.mark.asyncio
    async def test_dispatch_on_emitter_wrap_no_subscribers(self):
        """With no subscribers, dispatch should return None."""
        from open_webui_openrouter_pipe.plugins.registry import PluginRegistry
        registry = PluginRegistry()
        registry._hook_subscribers = {}

        result = await registry.dispatch_on_emitter_wrap(
            AsyncMock(), raw_emitter=AsyncMock(),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_dispatch_on_emitter_wrap_wraps_emitter(self):
        """A plugin returning non-None should replace the emitter."""
        from open_webui_openrouter_pipe.plugins.registry import PluginRegistry
        from open_webui_openrouter_pipe.plugins.base import PluginBase

        wrapped_emitter = AsyncMock()

        class TestPlugin(PluginBase):
            plugin_id = "test-wrapper"
            hooks = {"on_emitter_wrap": 50}

            async def on_emitter_wrap(self, stream_emitter, **kwargs):
                return wrapped_emitter

        registry = PluginRegistry()
        plugin = TestPlugin()
        registry._hook_subscribers = {
            "on_emitter_wrap": [(plugin, 50)],
        }

        result = await registry.dispatch_on_emitter_wrap(
            AsyncMock(), raw_emitter=AsyncMock(),
        )
        assert result is wrapped_emitter

    @pytest.mark.asyncio
    async def test_dispatch_on_emitter_wrap_none_passthrough(self):
        """A plugin returning None should not change the emitter."""
        from open_webui_openrouter_pipe.plugins.registry import PluginRegistry
        from open_webui_openrouter_pipe.plugins.base import PluginBase

        class TestPlugin(PluginBase):
            plugin_id = "test-noop"
            hooks = {"on_emitter_wrap": 50}

            async def on_emitter_wrap(self, stream_emitter, **kwargs):
                return None

        registry = PluginRegistry()
        plugin = TestPlugin()
        registry._hook_subscribers = {
            "on_emitter_wrap": [(plugin, 50)],
        }

        original = AsyncMock()
        result = await registry.dispatch_on_emitter_wrap(
            original, raw_emitter=AsyncMock(),
        )
        assert result is None  # No change


# ── PluginBase on_emitter_wrap default Tests ──


class TestThinkStreamingPluginValves:
    """Verify on_emitter_wrap handles merged valve objects correctly."""

    @pytest.mark.asyncio
    async def test_user_valve_missing_from_merged_valves(self):
        """on_emitter_wrap must not crash when THINK_STREAMING_USER_ENABLE is absent.

        _merge_valves produces a Valves object (not UserValves), so user-only
        fields like THINK_STREAMING_USER_ENABLE are dropped during the merge.
        The plugin must handle this gracefully — either via getattr with a
        default, or by checking hasattr first.
        """
        from open_webui_openrouter_pipe.plugins.think_streaming.plugin import (
            ThinkStreamingPlugin,
        )

        plugin = ThinkStreamingPlugin()

        # Simulate a merged Valves object that has THINK_STREAMING_ENABLE
        # but NOT THINK_STREAMING_USER_ENABLE (which is a user valve only)
        valves = Mock()
        valves.THINK_STREAMING_ENABLE = True
        # Deliberately omit THINK_STREAMING_USER_ENABLE — accessing it raises
        del valves.THINK_STREAMING_USER_ENABLE

        raw_emitter = AsyncMock()
        stream_emitter = AsyncMock(spec=[])  # spec=[] prevents auto-attribute creation

        # This must NOT raise AttributeError
        result = await plugin.on_emitter_wrap(
            stream_emitter,
            valves=valves,
            raw_emitter=raw_emitter,
            job_metadata={"user_id": "u1", "request_id": "r1"},
        )

        # With user valve absent, default should be True → plugin should proceed
        assert result is not None, (
            "Plugin should return a wrapper when user valve is absent "
            "(default=True)"
        )

    @pytest.mark.asyncio
    async def test_user_valve_explicitly_false(self):
        """on_emitter_wrap should return None when user explicitly disables."""
        from open_webui_openrouter_pipe.plugins.think_streaming.plugin import (
            ThinkStreamingPlugin,
        )

        plugin = ThinkStreamingPlugin()

        valves = Mock()
        valves.THINK_STREAMING_ENABLE = True
        valves.THINK_STREAMING_USER_ENABLE = False

        result = await plugin.on_emitter_wrap(
            AsyncMock(),
            valves=valves,
            raw_emitter=AsyncMock(),
            job_metadata={"user_id": "u1", "request_id": "r1"},
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_no_valves_returns_none(self):
        """on_emitter_wrap should return None when valves is None."""
        from open_webui_openrouter_pipe.plugins.think_streaming.plugin import (
            ThinkStreamingPlugin,
        )

        plugin = ThinkStreamingPlugin()

        result = await plugin.on_emitter_wrap(
            AsyncMock(spec=[]),
            valves=None,
            raw_emitter=AsyncMock(),
            job_metadata={},
        )

        assert result is None


class TestPluginBaseEmitterWrap:
    @pytest.mark.asyncio
    async def test_default_returns_none(self):
        """PluginBase.on_emitter_wrap should return None by default."""
        from open_webui_openrouter_pipe.plugins.base import PluginBase
        plugin = PluginBase()
        result = await plugin.on_emitter_wrap(AsyncMock())
        assert result is None
