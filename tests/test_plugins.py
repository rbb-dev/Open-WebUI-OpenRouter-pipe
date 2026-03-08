"""Tests for the plugin system: registration, dispatch, priorities, error isolation."""

from __future__ import annotations

import pytest
import pytest_asyncio

from open_webui_openrouter_pipe.plugins.base import PluginBase, PluginContext
from open_webui_openrouter_pipe.plugins.registry import PluginRegistry


# ── Fixtures ──


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset class-level registry between tests."""
    original_classes = PluginRegistry._plugin_classes[:]
    original_valve_fields = dict(PluginRegistry._pending_valve_fields)
    original_user_valve_fields = dict(PluginRegistry._pending_user_valve_fields)
    PluginRegistry._plugin_classes.clear()
    PluginRegistry._pending_valve_fields.clear()
    PluginRegistry._pending_user_valve_fields.clear()
    yield
    PluginRegistry._plugin_classes.clear()
    PluginRegistry._plugin_classes.extend(original_classes)
    PluginRegistry._pending_valve_fields.clear()
    PluginRegistry._pending_valve_fields.update(original_valve_fields)
    PluginRegistry._pending_user_valve_fields.clear()
    PluginRegistry._pending_user_valve_fields.update(original_user_valve_fields)


def _make_mock_pipe():
    """Create a minimal mock Pipe object for testing."""
    from unittest.mock import Mock

    pipe = Mock()
    pipe.id = "test-pipe"
    pipe.valves = Mock()
    pipe.valves.ENABLE_PLUGIN_SYSTEM = True
    pipe._artifact_store = Mock()
    pipe._circuit_breaker = Mock()
    pipe._redis_client = None
    pipe._redis_enabled = False
    pipe._request_queue = None
    pipe._catalog_manager = None
    pipe._http_session = None
    return pipe


# ── Registration Tests ──


class TestPluginRegistration:
    def test_register_decorator(self):
        """@PluginRegistry.register registers the plugin class."""

        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "my-plugin"

        assert MyPlugin in PluginRegistry._plugin_classes

    def test_register_idempotent(self):
        """Registering the same class twice doesn't duplicate."""

        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "my-plugin"

        PluginRegistry.register(MyPlugin)
        assert PluginRegistry._plugin_classes.count(MyPlugin) == 1

    def test_register_direct_call(self):
        """register() works as a direct call, not just a decorator."""

        class MyPlugin(PluginBase):
            plugin_id = "my-plugin"

        PluginRegistry.register(MyPlugin)
        assert MyPlugin in PluginRegistry._plugin_classes


# ── Initialization Tests ──


class TestPluginInit:
    def test_init_plugins_calls_on_init(self):
        """init_plugins() instantiates plugins and calls on_init()."""
        init_called = []

        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "my-plugin"
            hooks = {"on_models": 10}

            def on_init(self, ctx, **kwargs):
                init_called.append(ctx)

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        assert len(init_called) == 1
        assert isinstance(init_called[0], PluginContext)

    def test_init_plugins_error_isolation(self):
        """A plugin that fails on_init() doesn't block others."""
        init_called = []

        @PluginRegistry.register
        class BadPlugin(PluginBase):
            plugin_id = "bad-plugin"

            def on_init(self, ctx, **kwargs):
                raise RuntimeError("init failed")

        @PluginRegistry.register
        class GoodPlugin(PluginBase):
            plugin_id = "good-plugin"

            def on_init(self, ctx, **kwargs):
                init_called.append(True)

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        assert len(init_called) == 1

    def test_empty_registry(self):
        """Empty registry initializes without error."""
        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        assert len(registry._plugins) == 0


# ── Hook Subscription Tests ──


class TestHookSubscription:
    def test_only_subscribed_hooks_dispatched(self):
        """Plugin only receives hooks it subscribes to."""
        calls = []

        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "my-plugin"
            hooks = {"on_models": 10}

            def on_models(self, models, **kwargs):
                calls.append("on_models")

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                calls.append("on_request")
                return None

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())

        # on_models is subscribed
        registry.dispatch_on_models([])
        assert "on_models" in calls

        # on_request is NOT subscribed — should not be dispatched
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            registry.dispatch_on_request({}, {}, {}, None, None)
        )
        assert "on_request" not in calls
        assert result is None

    def test_unsubscribed_hook_not_called(self):
        """Plugin without hook in hooks dict is not called for that hook."""
        called = []

        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "my-plugin"
            hooks = {}  # No subscriptions

            def on_models(self, models, **kwargs):
                called.append(True)

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        models = [{"id": "test"}]
        registry.dispatch_on_models(models)
        assert not called
        # Models list unchanged since no plugin ran
        assert models == [{"id": "test"}]


# ── Priority Tests ──


class TestPriorityOrdering:
    def test_higher_priority_runs_first(self):
        """Plugins with higher priority number run first."""
        order = []

        @PluginRegistry.register
        class LowPlugin(PluginBase):
            plugin_id = "low"
            hooks = {"on_models": 10}

            def on_models(self, models, **kwargs):
                order.append("low")

        @PluginRegistry.register
        class HighPlugin(PluginBase):
            plugin_id = "high"
            hooks = {"on_models": 50}

            def on_models(self, models, **kwargs):
                order.append("high")

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        registry.dispatch_on_models([])
        assert order == ["high", "low"]


# ── Dispatch Strategy Tests ──


class TestDispatchStrategies:
    def test_on_models_chain(self):
        """on_models: plugins mutate models list in place."""

        @PluginRegistry.register
        class PluginA(PluginBase):
            plugin_id = "a"
            hooks = {"on_models": 20}

            def on_models(self, models, **kwargs):
                models.append({"id": "added-by-a"})

        @PluginRegistry.register
        class PluginB(PluginBase):
            plugin_id = "b"
            hooks = {"on_models": 10}

            def on_models(self, models, **kwargs):
                models.append({"id": "added-by-b"})

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        models = [{"id": "original"}]
        registry.dispatch_on_models(models)
        ids = [m["id"] for m in models]
        assert "original" in ids
        assert "added-by-a" in ids
        assert "added-by-b" in ids

    @pytest.mark.asyncio
    async def test_on_request_chain_last_writer_wins(self):
        """on_request: chain dispatch — all run, each receives current_result; last non-None return wins."""
        calls = []

        @PluginRegistry.register
        class HighPlugin(PluginBase):
            plugin_id = "high"
            hooks = {"on_request": 50}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                calls.append("high")
                return {"response": "from-high"}

        @PluginRegistry.register
        class LowPlugin(PluginBase):
            plugin_id = "low"
            hooks = {"on_request": 10}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                calls.append("low")
                return {"response": "from-low"}

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        result = await registry.dispatch_on_request({}, {}, {}, None, None)

        # Both called in priority order
        assert calls == ["high", "low"]
        # Chain semantics: last writer wins (low runs after high)
        assert result == {"response": "from-low"}

    @pytest.mark.asyncio
    async def test_on_request_none_results_ignored(self):
        """on_request: None results are not selected."""

        @PluginRegistry.register
        class PassthroughPlugin(PluginBase):
            plugin_id = "passthrough"
            hooks = {"on_request": 50}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                return None  # Skip

        @PluginRegistry.register
        class ResponsePlugin(PluginBase):
            plugin_id = "responder"
            hooks = {"on_request": 10}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                return "low-priority-response"

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        result = await registry.dispatch_on_request({}, {}, {}, None, None)
        assert result == "low-priority-response"

    @pytest.mark.asyncio
    async def test_on_request_transform_chain(self):
        """on_request_transform: body mutated sequentially in place."""

        @PluginRegistry.register
        class TransformA(PluginBase):
            plugin_id = "transform-a"
            hooks = {"on_request_transform": 20}

            async def on_request_transform(self, body, model, valves, **kwargs):
                body["added_by_a"] = True

        @PluginRegistry.register
        class TransformB(PluginBase):
            plugin_id = "transform-b"
            hooks = {"on_request_transform": 10}

            async def on_request_transform(self, body, model, valves, **kwargs):
                body["added_by_b"] = True

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        body = {"original": True}
        await registry.dispatch_on_request_transform(body, "test-model", None)
        assert body["original"] is True
        assert body["added_by_a"] is True
        assert body["added_by_b"] is True

    @pytest.mark.asyncio
    async def test_on_response_transform_chain(self):
        """on_response_transform: plugins mutate completion_data in place."""
        calls = []

        @PluginRegistry.register
        class TransformA(PluginBase):
            plugin_id = "transform-a"
            hooks = {"on_response_transform": 20}

            async def on_response_transform(self, completion_data, model, metadata, **kwargs):
                calls.append(("a", completion_data["content"]))
                completion_data["content"] += " [a]"

        @PluginRegistry.register
        class TransformB(PluginBase):
            plugin_id = "transform-b"
            hooks = {"on_response_transform": 10}

            async def on_response_transform(self, completion_data, model, metadata, **kwargs):
                calls.append(("b", completion_data["content"]))
                completion_data["content"] += " [b]"

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        data = {"done": True, "content": "hello"}
        await registry.dispatch_on_response_transform(data, "model", {})
        assert ("a", "hello") in calls
        assert ("b", "hello [a]") in calls
        assert data["content"] == "hello [a] [b]"


# ── Error Isolation Tests ──


class TestErrorIsolation:
    def test_on_models_error_isolation(self):
        """Failing plugin in on_models doesn't crash dispatch."""

        @PluginRegistry.register
        class BadPlugin(PluginBase):
            plugin_id = "bad"
            hooks = {"on_models": 50}

            def on_models(self, models, **kwargs):
                raise ValueError("boom")

        @PluginRegistry.register
        class GoodPlugin(PluginBase):
            plugin_id = "good"
            hooks = {"on_models": 10}

            def on_models(self, models, **kwargs):
                models.append({"id": "added-by-good"})

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        models = []
        registry.dispatch_on_models(models)
        ids = [m["id"] for m in models]
        assert "added-by-good" in ids

    @pytest.mark.asyncio
    async def test_on_request_error_isolation(self):
        """Failing plugin in on_request doesn't crash dispatch."""
        calls = []

        @PluginRegistry.register
        class BadPlugin(PluginBase):
            plugin_id = "bad"
            hooks = {"on_request": 50}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                raise RuntimeError("boom")

        @PluginRegistry.register
        class GoodPlugin(PluginBase):
            plugin_id = "good"
            hooks = {"on_request": 10}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                calls.append(True)
                return "good-response"

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        result = await registry.dispatch_on_request({}, {}, {}, None, None)
        assert calls == [True]
        assert result == "good-response"

    def test_on_shutdown_error_isolation(self):
        """Failing plugin in on_shutdown doesn't crash dispatch."""
        calls = []

        @PluginRegistry.register
        class BadPlugin(PluginBase):
            plugin_id = "bad"

            def on_shutdown(self, **kwargs):
                raise RuntimeError("boom")

        @PluginRegistry.register
        class GoodPlugin(PluginBase):
            plugin_id = "good"

            def on_shutdown(self, **kwargs):
                calls.append(True)

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        registry.dispatch_on_shutdown()
        assert calls == [True]

    @pytest.mark.asyncio
    async def test_on_request_transform_error_isolation(self):
        """Failing plugin in on_request_transform doesn't block others."""
        @PluginRegistry.register
        class BadTransform(PluginBase):
            plugin_id = "bad-transform"
            hooks = {"on_request_transform": 50}

            async def on_request_transform(self, body, model, valves, **kwargs):
                raise RuntimeError("transform boom")

        @PluginRegistry.register
        class GoodTransform(PluginBase):
            plugin_id = "good-transform"
            hooks = {"on_request_transform": 10}

            async def on_request_transform(self, body, model, valves, **kwargs):
                body["transformed"] = True

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        body = {"original": True}
        await registry.dispatch_on_request_transform(body, "test-model", None)
        assert body["original"] is True
        assert body["transformed"] is True

    @pytest.mark.asyncio
    async def test_on_response_transform_error_isolation(self):
        """Failing plugin in on_response_transform doesn't block others."""
        calls = []

        @PluginRegistry.register
        class BadTransformer(PluginBase):
            plugin_id = "bad-transformer"
            hooks = {"on_response_transform": 50}

            async def on_response_transform(self, completion_data, model, metadata, **kwargs):
                raise RuntimeError("transformer boom")

        @PluginRegistry.register
        class GoodTransformer(PluginBase):
            plugin_id = "good-transformer"
            hooks = {"on_response_transform": 10}

            async def on_response_transform(self, completion_data, model, metadata, **kwargs):
                calls.append(completion_data["content"])
                completion_data["content"] += " [transformed]"

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        data = {"done": True, "content": "test-response"}
        await registry.dispatch_on_response_transform(data, "model", {})
        assert calls == ["test-response"]
        assert data["content"] == "test-response [transformed]"


# ── Chain Dispatch Tests (new semantics) ──


class TestChainDispatch:
    @pytest.mark.asyncio
    async def test_on_request_chain_all_plugins_run(self):
        """Three plugins: A returns response, B sees it via current_result, C passes through."""
        calls = []

        @PluginRegistry.register
        class PluginA(PluginBase):
            plugin_id = "a"
            hooks = {"on_request": 100}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                calls.append("a")
                return {"response": "from-a"}

        @PluginRegistry.register
        class PluginB(PluginBase):
            plugin_id = "b"
            hooks = {"on_request": 50}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                calls.append("b")
                prior = kwargs.get("current_result")
                return {"response": f"modified-by-b(saw:{prior})"}

        @PluginRegistry.register
        class PluginC(PluginBase):
            plugin_id = "c"
            hooks = {"on_request": 10}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                calls.append("c")
                return None  # Pass through

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        result = await registry.dispatch_on_request({}, {}, {}, None, None)

        # All three ran in priority order
        assert calls == ["a", "b", "c"]
        # B received A's result via current_result and built on it; C returned None so B's result persists
        assert result == {"response": "modified-by-b(saw:{'response': 'from-a'})"}

    @pytest.mark.asyncio
    async def test_on_request_chain_none_passthrough(self):
        """All plugins return None — dispatch returns None."""

        @PluginRegistry.register
        class PluginA(PluginBase):
            plugin_id = "a"
            hooks = {"on_request": 50}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                return None

        @PluginRegistry.register
        class PluginB(PluginBase):
            plugin_id = "b"
            hooks = {"on_request": 10}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                return None

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        result = await registry.dispatch_on_request({}, {}, {}, None, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_on_request_side_effect_still_runs(self):
        """High-priority returns response; low-priority still runs and receives accumulated result."""
        side_effects = []

        @PluginRegistry.register
        class ResponsePlugin(PluginBase):
            plugin_id = "responder"
            hooks = {"on_request": 100}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                return {"response": "intercepted"}

        @PluginRegistry.register
        class SideEffectPlugin(PluginBase):
            plugin_id = "side-effect"
            hooks = {"on_request": 10}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                side_effects.append("fired")
                return None  # Observe only

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        result = await registry.dispatch_on_request({}, {}, {}, None, None)

        assert result == {"response": "intercepted"}
        assert side_effects == ["fired"]


# ── Zombie Plugin / Init Safety Tests ──


class TestInitSafety:
    def test_on_init_failure_does_not_add_to_plugins(self):
        """Plugin whose on_init raises is NOT in _plugins or _hook_subscribers."""

        @PluginRegistry.register
        class BadPlugin(PluginBase):
            plugin_id = "bad-init"
            hooks = {"on_models": 50}

            def on_init(self, ctx, **kwargs):
                raise RuntimeError("init explosion")

        @PluginRegistry.register
        class GoodPlugin(PluginBase):
            plugin_id = "good-init"
            hooks = {"on_models": 10}

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())

        # Only good plugin should be present
        plugin_ids = [p.plugin_id for p in registry._plugins]
        assert "bad-init" not in plugin_ids
        assert "good-init" in plugin_ids

        # Hook subscribers should not contain the bad plugin
        subscriber_ids = [
            p.plugin_id for p, _ in registry._hook_subscribers.get("on_models", [])
        ]
        assert "bad-init" not in subscriber_ids
        assert "good-init" in subscriber_ids

    def test_per_plugin_logger(self):
        """Each plugin gets a context with a logger named after its plugin_id."""
        loggers = {}

        @PluginRegistry.register
        class PluginA(PluginBase):
            plugin_id = "alpha"
            hooks = {"on_models": 10}

            def on_init(self, ctx, **kwargs):
                loggers["alpha"] = ctx.logger

        @PluginRegistry.register
        class PluginB(PluginBase):
            plugin_id = "beta"
            hooks = {"on_models": 10}

            def on_init(self, ctx, **kwargs):
                loggers["beta"] = ctx.logger

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())

        assert "alpha" in loggers
        assert "beta" in loggers
        # Logger names should contain the plugin ID
        assert "alpha" in loggers["alpha"].name
        assert "beta" in loggers["beta"].name
        # Loggers should be different objects
        assert loggers["alpha"] is not loggers["beta"]

    def test_re_init_safety(self):
        """Calling init_plugins() twice should not duplicate plugins (idempotency guard)."""
        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "re-init-test"
            hooks = {"on_models": 10}

        pipe = _make_mock_pipe()
        registry = PluginRegistry()
        registry.init_plugins(pipe)
        registry.init_plugins(pipe)

        # Idempotency guard: second call is a no-op
        plugin_ids = [p.plugin_id for p in registry._plugins]
        assert plugin_ids.count("re-init-test") == 1


# ── Priority Edge Case Tests ──


class TestPriorityEdgeCases:
    def test_equal_priority_ordering(self):
        """Two plugins at same priority: registration order preserved (stable sort)."""
        order = []

        @PluginRegistry.register
        class PluginFirst(PluginBase):
            plugin_id = "first"
            hooks = {"on_models": 50}

            def on_models(self, models, **kwargs):
                order.append("first")

        @PluginRegistry.register
        class PluginSecond(PluginBase):
            plugin_id = "second"
            hooks = {"on_models": 50}

            def on_models(self, models, **kwargs):
                order.append("second")

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        registry.dispatch_on_models([])
        assert order == ["first", "second"]

    def test_three_plugin_priority_sort(self):
        """Priorities 10, 30, 50 execute in order 50, 30, 10."""
        order = []

        @PluginRegistry.register
        class Low(PluginBase):
            plugin_id = "low"
            hooks = {"on_models": 10}

            def on_models(self, models, **kwargs):
                order.append(10)

        @PluginRegistry.register
        class Mid(PluginBase):
            plugin_id = "mid"
            hooks = {"on_models": 30}

            def on_models(self, models, **kwargs):
                order.append(30)

        @PluginRegistry.register
        class High(PluginBase):
            plugin_id = "high"
            hooks = {"on_models": 50}

            def on_models(self, models, **kwargs):
                order.append(50)

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        registry.dispatch_on_models([])
        assert order == [50, 30, 10]

    def test_on_models_plugin_returns_none(self):
        """on_models void dispatch: plugin mutates list, returning None is expected."""

        @PluginRegistry.register
        class Adder(PluginBase):
            plugin_id = "adder"
            hooks = {"on_models": 50}

            def on_models(self, models, **kwargs):
                models.append({"id": "added"})

        @PluginRegistry.register
        class NoneReturner(PluginBase):
            plugin_id = "none-returner"
            hooks = {"on_models": 10}

            def on_models(self, models, **kwargs):
                pass  # Void — no return needed

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        models = [{"id": "original"}]
        registry.dispatch_on_models(models)
        ids = [m["id"] for m in models]
        assert "original" in ids
        assert "added" in ids

    @pytest.mark.asyncio
    async def test_dispatch_with_no_subscribers(self):
        """Dispatch methods work correctly when no plugins subscribe."""

        @PluginRegistry.register
        class NoHooksPlugin(PluginBase):
            plugin_id = "no-hooks"
            hooks = {}  # Subscribes to nothing

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())

        # on_models: void dispatch, models unchanged
        models = [{"id": "test"}]
        registry.dispatch_on_models(models)
        assert models == [{"id": "test"}]

        # on_request returns None
        result = await registry.dispatch_on_request({}, {}, {}, None, None)
        assert result is None

        # on_request_transform: void dispatch, body unchanged
        body = {"model": "test"}
        await registry.dispatch_on_request_transform(body, "test", None)
        assert body == {"model": "test"}

        # on_response_transform: void dispatch, data unchanged
        data = {"done": True, "content": "text"}
        await registry.dispatch_on_response_transform(data, "model", {})
        assert data["content"] == "text"


# ── Multi-Hook Subscription Tests ──


class TestMultiHookSubscription:
    def test_plugin_subscribes_to_multiple_hooks(self):
        """One plugin with different priorities per hook gets correct placement."""

        @PluginRegistry.register
        class MultiHook(PluginBase):
            plugin_id = "multi"
            hooks = {
                "on_models": 90,
                "on_request": 30,
                "on_request_transform": 60,
                "on_response_transform": 10,
            }

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())

        # Verify correct priority in each subscriber list
        for hook_name, expected_priority in [
            ("on_models", 90),
            ("on_request", 30),
            ("on_request_transform", 60),
            ("on_response_transform", 10),
        ]:
            subscribers = registry._hook_subscribers.get(hook_name, [])
            assert len(subscribers) == 1
            plugin, priority = subscribers[0]
            assert plugin.plugin_id == "multi"
            assert priority == expected_priority


# ── Lifecycle Invariant Tests ──


class TestLifecycleInvariants:
    def test_on_shutdown_called_with_empty_hooks(self):
        """on_shutdown is lifecycle — called on ALL plugins, regardless of hooks dict."""
        calls = []

        @PluginRegistry.register
        class NoHooksPlugin(PluginBase):
            plugin_id = "no-hooks"
            hooks = {}  # No subscribable hooks

            def on_shutdown(self, **kwargs):
                calls.append("no-hooks")

        @PluginRegistry.register
        class WithHooksPlugin(PluginBase):
            plugin_id = "with-hooks"
            hooks = {"on_models": 10}

            def on_shutdown(self, **kwargs):
                calls.append("with-hooks")

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        registry.dispatch_on_shutdown()

        # Both plugins receive on_shutdown regardless of hooks subscription
        assert "no-hooks" in calls
        assert "with-hooks" in calls


class TestBodyMutationVisibility:
    @pytest.mark.asyncio
    async def test_body_mutation_visible_to_next_plugin(self):
        """First plugin mutates body dict; second plugin sees the mutation."""

        @PluginRegistry.register
        class Mutator(PluginBase):
            plugin_id = "mutator"
            hooks = {"on_request_transform": 50}

            async def on_request_transform(self, body, model, valves, **kwargs):
                body["mutated_by_first"] = True

        @PluginRegistry.register
        class Observer(PluginBase):
            plugin_id = "observer"
            hooks = {"on_request_transform": 10}

            async def on_request_transform(self, body, model, valves, **kwargs):
                body["saw_mutation"] = body.get("mutated_by_first", False)

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        body = {"original": True}
        await registry.dispatch_on_request_transform(body, "test", None)
        assert body["mutated_by_first"] is True
        assert body["saw_mutation"] is True


class TestTransformNonePassthrough:
    @pytest.mark.asyncio
    async def test_none_return_passes_body_unchanged(self):
        """Plugin returning None from on_request_transform is fine (void)."""

        @PluginRegistry.register
        class NonePlugin(PluginBase):
            plugin_id = "none-plugin"
            hooks = {"on_request_transform": 50}

            async def on_request_transform(self, body, model, valves, **kwargs):
                pass  # Void — no mutation

        @PluginRegistry.register
        class ModifyPlugin(PluginBase):
            plugin_id = "modifier"
            hooks = {"on_request_transform": 10}

            async def on_request_transform(self, body, model, valves, **kwargs):
                body["modified"] = True

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        body = {"original": True}
        await registry.dispatch_on_request_transform(body, "test", None)
        # Original body passes through no-op plugin unchanged, modifier adds key
        assert body["original"] is True
        assert body["modified"] is True


# ── Timeout Isolation Tests (4.1) ──


class TestTimeoutIsolation:
    @pytest.mark.asyncio
    async def test_slow_plugin_does_not_block_others(self):
        """A plugin timing out in on_request doesn't block subsequent plugins."""
        import asyncio
        from unittest.mock import patch

        @PluginRegistry.register
        class SlowPlugin(PluginBase):
            plugin_id = "slow"
            hooks = {"on_request": 50}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                await asyncio.sleep(9999)
                return "never-returned"

        @PluginRegistry.register
        class FastPlugin(PluginBase):
            plugin_id = "fast"
            hooks = {"on_request": 10}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                return "fast-response"

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())

        # Patch timeout to 0.01s so the slow plugin times out quickly
        with patch("open_webui_openrouter_pipe.plugins.registry._PR_DISPATCH_TIMEOUT", 0.01):
            result = await registry.dispatch_on_request({}, {}, {}, None, None)

        # Fast plugin still ran and its result is returned
        assert result == "fast-response"


# ── Kwarg Forwarding Tests (4.2) ──


class TestKwargForwarding:
    @pytest.mark.asyncio
    async def test_on_request_forwards_valves_kwarg(self):
        """dispatch_on_request passes valves= kwarg through to plugin."""
        received_kwargs = {}

        @PluginRegistry.register
        class KwargPlugin(PluginBase):
            plugin_id = "kwargs-test"
            hooks = {"on_request": 50}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                received_kwargs.update(kwargs)
                return None

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        sentinel = object()
        await registry.dispatch_on_request({}, {}, {}, None, None, valves=sentinel)
        assert received_kwargs.get("valves") is sentinel

    @pytest.mark.asyncio
    async def test_on_request_transform_forwards_user_metadata_kwargs(self):
        """dispatch_on_request_transform passes user= and metadata= kwargs."""
        received_kwargs = {}

        @PluginRegistry.register
        class KwargPlugin(PluginBase):
            plugin_id = "kwargs-transform"
            hooks = {"on_request_transform": 50}

            async def on_request_transform(self, body, model, valves, **kwargs):
                received_kwargs.update(kwargs)

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        user_obj = {"id": "test-user"}
        meta_obj = {"chat_id": "abc"}
        await registry.dispatch_on_request_transform(
            {}, "model", None, user=user_obj, metadata=meta_obj,
        )
        assert received_kwargs.get("user") is user_obj
        assert received_kwargs.get("metadata") is meta_obj

    @pytest.mark.asyncio
    async def test_on_response_transform_forwards_user_id_and_user_kwargs(self):
        """dispatch_on_response_transform passes user_id= and user= kwargs."""
        received_kwargs = {}

        @PluginRegistry.register
        class KwargPlugin(PluginBase):
            plugin_id = "kwargs-response"
            hooks = {"on_response_transform": 50}

            async def on_response_transform(self, completion_data, model, metadata, **kwargs):
                received_kwargs.update(kwargs)

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        user_obj = {"name": "Admin"}
        data = {"done": True, "content": "text"}
        await registry.dispatch_on_response_transform(
            data, "model", {}, user_id="uid-123", user=user_obj,
        )
        assert received_kwargs.get("user_id") == "uid-123"
        assert received_kwargs.get("user") is user_obj


# ── Async on_models Detection Test (4.10) ──


class TestAsyncOnModelsGuard:
    def test_async_on_models_detected_and_skipped(self):
        """Plugin with async def on_models doesn't corrupt the model list."""

        @PluginRegistry.register
        class AsyncPlugin(PluginBase):
            plugin_id = "async-models"
            hooks = {"on_models": 50}

            async def on_models(self, models, **kwargs):  # type: ignore[override]
                # This would corrupt if not caught — but dispatch detects coroutine
                models.clear()
                models.append({"id": "corrupted"})

        @PluginRegistry.register
        class NormalPlugin(PluginBase):
            plugin_id = "normal-models"
            hooks = {"on_models": 10}

            def on_models(self, models, **kwargs):
                models.append({"id": "normal-added"})

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        models = [{"id": "original"}]
        registry.dispatch_on_models(models)
        ids = [m["id"] for m in models]
        # Async plugin's coroutine is detected and closed (body never executes)
        assert "corrupted" not in ids
        # Original models and normal plugin's contribution survive
        assert "original" in ids
        assert "normal-added" in ids


# ── Duplicate plugin_id Warning Test (4.12) ──


class TestDuplicatePluginIdWarning:
    def test_duplicate_plugin_id_warns_but_both_run(self):
        """Two plugins with same plugin_id: both register, warning logged."""

        @PluginRegistry.register
        class PluginA(PluginBase):
            plugin_id = "dupe-id"
            hooks = {"on_models": 50}

            def on_models(self, models, **kwargs):
                models.append({"id": "from-a"})

        @PluginRegistry.register
        class PluginB(PluginBase):
            plugin_id = "dupe-id"
            hooks = {"on_models": 10}

            def on_models(self, models, **kwargs):
                models.append({"id": "from-b"})

        registry = PluginRegistry()

        with pytest.raises(Exception) if False else \
                pytest.warns(None) if False else \
                _noop_context():
            pass  # Just ensuring we can detect the warning

        # Use caplog-style detection via logging
        with _capture_warnings() as warnings:
            registry.init_plugins(_make_mock_pipe())

        # Both plugins should be initialized and functional
        models = []
        registry.dispatch_on_models(models)
        ids = [m["id"] for m in models]
        assert "from-a" in ids
        assert "from-b" in ids

        # Verify a duplicate warning was logged
        assert any("Duplicate plugin_id" in w for w in warnings)


# ── current_result Chaining Tests ──


class TestCurrentResultChaining:
    @pytest.mark.asyncio
    async def test_plugin_b_reads_plugin_a_result(self):
        """Plugin B receives Plugin A's return via current_result kwarg."""
        captured = []

        @PluginRegistry.register
        class PluginA(PluginBase):
            plugin_id = "a"
            hooks = {"on_request": 50}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                return {"response": "from-a"}

        @PluginRegistry.register
        class PluginB(PluginBase):
            plugin_id = "b"
            hooks = {"on_request": 10}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                captured.append(kwargs.get("current_result"))
                return None

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        await registry.dispatch_on_request({}, {}, {}, None, None)

        assert len(captured) == 1
        assert captured[0] == {"response": "from-a"}


# ── current_result Kwarg Forwarding Test ──
# (added to TestKwargForwarding via standalone class to avoid
#  reopening the existing class definition)


class TestCurrentResultKwargForwarding:
    @pytest.mark.asyncio
    async def test_on_request_passes_current_result(self):
        """dispatch_on_request forwards current_result kwarg to later plugins."""
        captured = []

        @PluginRegistry.register
        class Producer(PluginBase):
            plugin_id = "producer"
            hooks = {"on_request": 50}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                return "produced-value"

        @PluginRegistry.register
        class Consumer(PluginBase):
            plugin_id = "consumer"
            hooks = {"on_request": 10}

            async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
                captured.append(kwargs.get("current_result"))
                return None

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        result = await registry.dispatch_on_request({}, {}, {}, None, None)

        assert result == "produced-value"
        assert len(captured) == 1
        assert captured[0] == "produced-value"


# ── Model Tracking Tests ──


class TestModelTracking:
    @pytest.mark.asyncio
    async def test_model_tracks_body_changes(self):
        """on_request_transform re-reads model from body after each plugin."""
        captured_models = []

        @PluginRegistry.register
        class ModelChanger(PluginBase):
            plugin_id = "model-changer"
            hooks = {"on_request_transform": 50}

            async def on_request_transform(self, body, model, valves, **kwargs):
                body["model"] = "new-model"

        @PluginRegistry.register
        class ModelObserver(PluginBase):
            plugin_id = "model-observer"
            hooks = {"on_request_transform": 10}

            async def on_request_transform(self, body, model, valves, **kwargs):
                captured_models.append(model)

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        await registry.dispatch_on_request_transform(
            {"model": "original-model"}, "original-model", None,
        )

        assert len(captured_models) == 1
        assert captured_models[0] == "new-model"


# ── Response Transform Chaining Tests ──


class TestResponseTransformChaining:
    @pytest.mark.asyncio
    async def test_completion_data_chains(self):
        """on_response_transform: content mutated in place through each plugin in order."""

        @PluginRegistry.register
        class AppenderA(PluginBase):
            plugin_id = "appender-a"
            hooks = {"on_response_transform": 50}

            async def on_response_transform(self, completion_data, model, metadata, **kwargs):
                completion_data["content"] += " [a]"

        @PluginRegistry.register
        class AppenderB(PluginBase):
            plugin_id = "appender-b"
            hooks = {"on_response_transform": 10}

            async def on_response_transform(self, completion_data, model, metadata, **kwargs):
                completion_data["content"] += " [b]"

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        data = {"done": True, "content": "original"}
        await registry.dispatch_on_response_transform(data, "model", {})
        assert data["content"] == "original [a] [b]"


# ── Void Dispatch Tests ──


class TestVoidDispatch:
    def test_void_dispatch_returns_none(self):
        """All three void dispatchers return None."""

        @PluginRegistry.register
        class TestPlugin(PluginBase):
            plugin_id = "void-test"
            hooks = {
                "on_models": 10,
                "on_request_transform": 10,
                "on_response_transform": 10,
            }

            def on_models(self, models, **kwargs):
                models.append({"id": "added"})

            async def on_request_transform(self, body, model, valves, **kwargs):
                body["key"] = "value"

            async def on_response_transform(self, completion_data, model, metadata, **kwargs):
                completion_data["content"] += " modified"

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())

        # dispatch_on_models returns None
        result = registry.dispatch_on_models([])
        assert result is None

    @pytest.mark.asyncio
    async def test_void_dispatch_async_returns_none(self):
        """Async void dispatchers return None."""

        @PluginRegistry.register
        class TestPlugin(PluginBase):
            plugin_id = "void-test"
            hooks = {
                "on_request_transform": 10,
                "on_response_transform": 10,
            }

            async def on_request_transform(self, body, model, valves, **kwargs):
                body["key"] = "value"

            async def on_response_transform(self, completion_data, model, metadata, **kwargs):
                completion_data["content"] += " modified"

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())

        # dispatch_on_request_transform returns None
        result = await registry.dispatch_on_request_transform({}, "model", None)
        assert result is None

        # dispatch_on_response_transform returns None
        data = {"done": True, "content": "text"}
        result = await registry.dispatch_on_response_transform(data, "model", {})
        assert result is None


class TestCompletionDataMutation:
    @pytest.mark.asyncio
    async def test_completion_data_mutation(self):
        """Plugin mutates content, usage, and adds sources — all visible after dispatch."""

        @PluginRegistry.register
        class RichMutator(PluginBase):
            plugin_id = "rich-mutator"
            hooks = {"on_response_transform": 50}

            async def on_response_transform(self, completion_data, model, metadata, **kwargs):
                completion_data["content"] += " [enriched]"
                completion_data["usage"] = {"prompt_tokens": 10, "completion_tokens": 20}
                completion_data["sources"] = [{"url": "https://example.com"}]

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        data = {"done": True, "content": "original"}
        await registry.dispatch_on_response_transform(data, "model", {})

        assert data["content"] == "original [enriched]"
        assert data["usage"] == {"prompt_tokens": 10, "completion_tokens": 20}
        assert data["sources"] == [{"url": "https://example.com"}]


class TestNoDictCopiesInResponseTransform:
    @pytest.mark.asyncio
    async def test_no_dict_copies_in_response_transform(self):
        """Pass shared metadata; plugin mutates it; caller sees mutation (no copy)."""

        @PluginRegistry.register
        class MetadataMutator(PluginBase):
            plugin_id = "meta-mutator"
            hooks = {"on_response_transform": 50}

            async def on_response_transform(self, completion_data, model, metadata, **kwargs):
                metadata["injected"] = True

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        shared_metadata = {"original": True}
        data = {"done": True, "content": "text"}
        await registry.dispatch_on_response_transform(data, "model", shared_metadata)

        # Caller sees the mutation — proving no dict copy was made
        assert shared_metadata["injected"] is True
        assert shared_metadata["original"] is True


class TestBodyMutationInPlace:
    @pytest.mark.asyncio
    async def test_body_mutation_in_place(self):
        """Plugins mutate body in on_request_transform; mutations persist without return."""

        @PluginRegistry.register
        class Adder(PluginBase):
            plugin_id = "adder"
            hooks = {"on_request_transform": 50}

            async def on_request_transform(self, body, model, valves, **kwargs):
                body["added_key"] = "added_value"
                body["messages"].append({"role": "system", "content": "injected"})

        @PluginRegistry.register
        class Modifier(PluginBase):
            plugin_id = "modifier"
            hooks = {"on_request_transform": 10}

            async def on_request_transform(self, body, model, valves, **kwargs):
                body["temperature"] = 0.5

        registry = PluginRegistry()
        registry.init_plugins(_make_mock_pipe())
        body = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
        }
        await registry.dispatch_on_request_transform(body, "test-model", None)

        assert body["added_key"] == "added_value"
        assert len(body["messages"]) == 2
        assert body["messages"][1]["content"] == "injected"
        assert body["temperature"] == 0.5


# ── Helpers for warning capture ──


import contextlib
import logging as _logging


@contextlib.contextmanager
def _noop_context():
    yield


@contextlib.contextmanager
def _capture_warnings():
    """Capture warning-level log messages from the plugin registry."""
    warnings: list[str] = []
    handler = _logging.Handler()
    handler.emit = lambda record: warnings.append(record.getMessage())  # type: ignore[assignment]
    handler.setLevel(_logging.WARNING)
    # Use PluginRegistry.__module__ to get the correct logger name in both
    # package mode ("open_webui_openrouter_pipe.plugins.registry") and
    # bundle mode ("owui_pipe_bundle").
    registry_logger = _logging.getLogger(PluginRegistry.__module__)
    registry_logger.addHandler(handler)
    try:
        yield warnings
    finally:
        registry_logger.removeHandler(handler)


# ── Plugin-Exported Valves Tests ──


class TestBuildExtendedValves:
    """Tests for build_extended_valves() and build_extended_user_valves()."""

    def test_no_plugin_fields_returns_base_unchanged(self):
        """When no plugins declare valves, the base class is returned as-is."""
        from pydantic import BaseModel

        class BaseValves(BaseModel):
            API_KEY: str = "default"

        result = PluginRegistry.build_extended_valves(BaseValves)
        assert result is BaseValves

    def test_no_plugin_user_fields_returns_base_unchanged(self):
        """When no plugins declare user valves, the base class is returned as-is."""
        from pydantic import BaseModel

        class BaseUserValves(BaseModel):
            SHOW_USAGE: bool = True

        result = PluginRegistry.build_extended_user_valves(BaseUserValves)
        assert result is BaseUserValves

    def test_plugin_fields_merged_into_valves(self):
        """Plugin-declared valve fields appear in the extended Valves class."""
        from pydantic import BaseModel, Field

        class BaseValves(BaseModel):
            API_KEY: str = "default"

        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "my-plugin"
            plugin_valves = {
                "MY_PLUGIN_ENABLE": (bool, Field(default=True)),
                "MY_PLUGIN_LIMIT": (int, Field(default=100)),
            }

        Extended = PluginRegistry.build_extended_valves(BaseValves)
        assert Extended is not BaseValves
        # Can instantiate with defaults
        instance = Extended()
        assert instance.API_KEY == "default"
        assert instance.MY_PLUGIN_ENABLE is True
        assert instance.MY_PLUGIN_LIMIT == 100

    def test_plugin_user_fields_merged_into_user_valves(self):
        """Plugin-declared user valve fields appear in the extended UserValves class."""
        from pydantic import BaseModel, Field

        class BaseUserValves(BaseModel):
            SHOW_USAGE: bool = True

        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "my-plugin"
            plugin_user_valves = {
                "USER_THEME": (str, Field(default="dark")),
            }

        Extended = PluginRegistry.build_extended_user_valves(BaseUserValves)
        assert Extended is not BaseUserValves
        instance = Extended()
        assert instance.SHOW_USAGE is True
        assert instance.USER_THEME == "dark"

    def test_extended_valves_accepts_custom_values(self):
        """Extended Valves can be instantiated with non-default values."""
        from pydantic import BaseModel, Field

        class BaseValves(BaseModel):
            API_KEY: str = "default"

        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "my-plugin"
            plugin_valves = {
                "MY_PLUGIN_LIMIT": (int, Field(default=100)),
            }

        Extended = PluginRegistry.build_extended_valves(BaseValves)
        instance = Extended(API_KEY="custom-key", MY_PLUGIN_LIMIT=500)
        assert instance.API_KEY == "custom-key"
        assert instance.MY_PLUGIN_LIMIT == 500

    def test_extended_valves_is_subclass_of_base(self):
        """Extended Valves is a proper subclass of the base Valves."""
        from pydantic import BaseModel, Field

        class BaseValves(BaseModel):
            API_KEY: str = "default"

        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "my-plugin"
            plugin_valves = {
                "MY_ENABLE": (bool, Field(default=True)),
            }

        Extended = PluginRegistry.build_extended_valves(BaseValves)
        assert issubclass(Extended, BaseValves)
        instance = Extended()
        assert isinstance(instance, BaseValves)

    def test_multiple_plugins_contribute_fields(self):
        """Multiple plugins can contribute different fields."""
        from pydantic import BaseModel, Field

        class BaseValves(BaseModel):
            API_KEY: str = "default"

        @PluginRegistry.register
        class PluginA(PluginBase):
            plugin_id = "plugin-a"
            plugin_valves = {
                "PLUGIN_A_ENABLE": (bool, Field(default=True)),
            }

        @PluginRegistry.register
        class PluginB(PluginBase):
            plugin_id = "plugin-b"
            plugin_valves = {
                "PLUGIN_B_LIMIT": (int, Field(default=50)),
            }

        Extended = PluginRegistry.build_extended_valves(BaseValves)
        instance = Extended()
        assert instance.PLUGIN_A_ENABLE is True
        assert instance.PLUGIN_B_LIMIT == 50

    def test_schema_includes_plugin_fields(self):
        """The JSON schema includes plugin-contributed fields (OWUI reads this)."""
        from pydantic import BaseModel, Field

        class BaseValves(BaseModel):
            API_KEY: str = "default"

        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "my-plugin"
            plugin_valves = {
                "MY_PLUGIN_ENABLE": (bool, Field(default=True, description="Enable my plugin")),
            }

        Extended = PluginRegistry.build_extended_valves(BaseValves)
        schema = Extended.model_json_schema()
        props = schema["properties"]
        assert "API_KEY" in props
        assert "MY_PLUGIN_ENABLE" in props
        assert props["MY_PLUGIN_ENABLE"]["description"] == "Enable my plugin"


class TestValveFieldCollision:
    """Tests for valve field name collision handling."""

    def test_collision_auto_renames_with_suffix(self):
        """When two plugins declare the same field name, the second is renamed."""
        from pydantic import BaseModel, Field

        @PluginRegistry.register
        class PluginA(PluginBase):
            plugin_id = "plugin-a"
            plugin_valves = {
                "SHARED_FIELD": (bool, Field(default=True)),
            }

        with _capture_warnings() as warnings:
            @PluginRegistry.register
            class PluginB(PluginBase):
                plugin_id = "plugin-b"
                plugin_valves = {
                    "SHARED_FIELD": (int, Field(default=99)),
                }

        # Should have been renamed, not raised
        assert "SHARED_FIELD" in PluginRegistry._pending_valve_fields
        assert "SHARED_FIELD_2" in PluginRegistry._pending_valve_fields
        # Warning was logged
        assert any("renamed" in w and "SHARED_FIELD_2" in w for w in warnings)

        # Both fields work in the extended class
        class BaseValves(BaseModel):
            pass

        Extended = PluginRegistry.build_extended_valves(BaseValves)
        instance = Extended()
        assert instance.SHARED_FIELD is True
        assert instance.SHARED_FIELD_2 == 99

    def test_triple_collision_increments_suffix(self):
        """Three-way collision produces _2 and _3 suffixes."""
        from pydantic import Field

        @PluginRegistry.register
        class P1(PluginBase):
            plugin_id = "p1"
            plugin_valves = {"DUP": (int, Field(default=1))}

        @PluginRegistry.register
        class P2(PluginBase):
            plugin_id = "p2"
            plugin_valves = {"DUP": (int, Field(default=2))}

        @PluginRegistry.register
        class P3(PluginBase):
            plugin_id = "p3"
            plugin_valves = {"DUP": (int, Field(default=3))}

        assert "DUP" in PluginRegistry._pending_valve_fields
        assert "DUP_2" in PluginRegistry._pending_valve_fields
        assert "DUP_3" in PluginRegistry._pending_valve_fields

    def test_user_valve_collision_auto_renames(self):
        """User valve field collisions are also renamed, not raised."""
        from pydantic import Field

        @PluginRegistry.register
        class PluginA(PluginBase):
            plugin_id = "plugin-a"
            plugin_user_valves = {
                "USER_PREF": (str, Field(default="a")),
            }

        with _capture_warnings() as warnings:
            @PluginRegistry.register
            class PluginB(PluginBase):
                plugin_id = "plugin-b"
                plugin_user_valves = {
                    "USER_PREF": (str, Field(default="b")),
                }

        assert "USER_PREF" in PluginRegistry._pending_user_valve_fields
        assert "USER_PREF_2" in PluginRegistry._pending_user_valve_fields
        assert any("renamed" in w for w in warnings)

    def test_no_collision_no_warning(self):
        """No warning when field names are unique."""
        from pydantic import Field

        with _capture_warnings() as warnings:
            @PluginRegistry.register
            class PluginA(PluginBase):
                plugin_id = "plugin-a"
                plugin_valves = {"FIELD_A": (bool, Field(default=True))}

            @PluginRegistry.register
            class PluginB(PluginBase):
                plugin_id = "plugin-b"
                plugin_valves = {"FIELD_B": (int, Field(default=0))}

        assert len(warnings) == 0


class TestPluginValvesIsolation:
    """Tests for __init_subclass__ isolation of plugin_valves dicts."""

    def test_plugin_valves_isolated_per_subclass(self):
        """Each subclass gets its own plugin_valves dict."""

        class PluginA(PluginBase):
            plugin_id = "a"
            plugin_valves = {"A_FIELD": (bool, True)}

        class PluginB(PluginBase):
            plugin_id = "b"

        # B should have empty dict, not A's
        assert "A_FIELD" not in PluginB.plugin_valves
        assert len(PluginB.plugin_valves) == 0

    def test_plugin_user_valves_isolated_per_subclass(self):
        """Each subclass gets its own plugin_user_valves dict."""

        class PluginA(PluginBase):
            plugin_id = "a"
            plugin_user_valves = {"USER_A": (str, "default")}

        class PluginB(PluginBase):
            plugin_id = "b"

        assert "USER_A" not in PluginB.plugin_user_valves
        assert len(PluginB.plugin_user_valves) == 0

    def test_mutation_does_not_affect_base(self):
        """Mutating a subclass dict doesn't affect PluginBase."""
        original_base = dict(PluginBase.plugin_valves)

        class MutantPlugin(PluginBase):
            plugin_id = "mutant"

        MutantPlugin.plugin_valves["SNEAKY"] = (bool, True)

        assert "SNEAKY" not in PluginBase.plugin_valves
        assert PluginBase.plugin_valves == original_base


class TestRegistryAccumulation:
    """Tests for _pending_valve_fields accumulation in register()."""

    def test_register_collects_valve_fields(self):
        """@register populates _pending_valve_fields."""
        from pydantic import Field

        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "test"
            plugin_valves = {
                "TEST_FIELD": (int, Field(default=10)),
            }

        assert "TEST_FIELD" in PluginRegistry._pending_valve_fields

    def test_register_collects_user_valve_fields(self):
        """@register populates _pending_user_valve_fields."""
        from pydantic import Field

        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "test"
            plugin_user_valves = {
                "USER_TEST": (str, Field(default="hello")),
            }

        assert "USER_TEST" in PluginRegistry._pending_user_valve_fields

    def test_register_collects_both_valve_types(self):
        """@register collects both plugin_valves and plugin_user_valves."""
        from pydantic import Field

        @PluginRegistry.register
        class MyPlugin(PluginBase):
            plugin_id = "test"
            plugin_valves = {"SYS_FIELD": (bool, Field(default=True))}
            plugin_user_valves = {"USER_FIELD": (int, Field(default=5))}

        assert "SYS_FIELD" in PluginRegistry._pending_valve_fields
        assert "USER_FIELD" in PluginRegistry._pending_user_valve_fields

    def test_plugin_without_valves_contributes_nothing(self):
        """A plugin with no plugin_valves contributes no fields."""

        @PluginRegistry.register
        class PlainPlugin(PluginBase):
            plugin_id = "plain"
            hooks = {"on_models": 50}

        assert len(PluginRegistry._pending_valve_fields) == 0
        assert len(PluginRegistry._pending_user_valve_fields) == 0

    def test_fixture_cleans_pending_fields(self):
        """The _clean_registry fixture resets _pending_valve_fields between tests."""
        # If the fixture works, this test starts with empty pending fields
        assert len(PluginRegistry._pending_valve_fields) == 0
        assert len(PluginRegistry._pending_user_valve_fields) == 0
