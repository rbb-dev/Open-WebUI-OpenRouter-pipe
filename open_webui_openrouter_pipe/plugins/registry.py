"""Plugin registry with self-registration, priority dispatch, and error isolation."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, TypeVar

from .base import PluginBase, PluginContext

_PluginT = TypeVar("_PluginT", bound=PluginBase)

if TYPE_CHECKING:
    from ..pipe import Pipe

_pr_logger = logging.getLogger(__name__)

# Per-plugin timeout for async dispatch calls (seconds).
# A hung plugin will be cancelled after this duration instead of blocking the request.
_PR_DISPATCH_TIMEOUT = 30.0

# Hook names that are subscription-based (controlled by plugin.hooks dict).
_PR_SUBSCRIBABLE_HOOKS = frozenset({
    "on_models",
    "on_request",
    "on_request_transform",
    "on_emitter_wrap",
    "on_response_transform",
})


class PluginRegistry:
    """Manages plugin registration, instantiation, and hook dispatch.

    Plugins register via ``@PluginRegistry.register`` class decorator.
    ``init_plugins()`` instantiates registered classes and builds per-hook
    subscriber lists sorted by priority (descending).
    """

    # Class-level registry — populated at import time by @register decorator.
    # Safe across uvicorn workers (each worker is a separate process).
    _plugin_classes: list[type[PluginBase]] = []

    # Accumulated valve field specs from all registered plugins.
    # Populated at import time by @register. Used by build_extended_valves().
    _pending_valve_fields: dict[str, tuple] = {}
    _pending_user_valve_fields: dict[str, tuple] = {}

    @classmethod
    def register(cls, plugin_class: type[_PluginT]) -> type[_PluginT]:
        """Register a plugin class. Works as a decorator or direct call."""
        if plugin_class not in cls._plugin_classes:
            cls._plugin_classes.append(plugin_class)
            # Collect plugin-contributed valve fields (system + user)
            cls._collect_valve_fields(
                plugin_class, "plugin_valves", cls._pending_valve_fields,
            )
            cls._collect_valve_fields(
                plugin_class, "plugin_user_valves", cls._pending_user_valve_fields,
            )
        return plugin_class

    @classmethod
    def _collect_valve_fields(
        cls,
        plugin_class: type[PluginBase],
        attr_name: str,
        target: dict[str, tuple],
    ) -> None:
        """Collect valve field specs from a plugin, auto-renaming on collision."""
        for field_name, field_spec in getattr(plugin_class, attr_name, {}).items():
            if field_name in target:
                original = field_name
                n = 2
                while field_name in target:
                    field_name = f"{original}_{n}"
                    n += 1
                _pr_logger.warning(
                    "Plugin '%s' %s field '%s' renamed to '%s' (name collision)",
                    plugin_class.plugin_id, attr_name, original, field_name,
                )
            target[field_name] = field_spec

    @classmethod
    def build_extended_valves(cls, base_valves_class: type) -> type:
        """Build a Valves subclass that includes plugin-contributed fields.

        Returns ``base_valves_class`` unchanged if no plugins declared fields.
        Uses ``pydantic.create_model()`` to dynamically extend the base class.
        """
        if not cls._pending_valve_fields:
            return base_valves_class
        from pydantic import create_model
        return create_model(
            "Valves",
            __base__=base_valves_class,
            **cls._pending_valve_fields,  # type: ignore[reportArgumentType]
        )

    @classmethod
    def build_extended_user_valves(cls, base_user_valves_class: type) -> type:
        """Build a UserValves subclass that includes plugin-contributed fields.

        Same pattern as ``build_extended_valves()`` but for per-user settings.
        """
        if not cls._pending_user_valve_fields:
            return base_user_valves_class
        from pydantic import create_model
        return create_model(
            "UserValves",
            __base__=base_user_valves_class,
            **cls._pending_user_valve_fields,  # type: ignore[reportArgumentType]
        )

    def __init__(self) -> None:
        # Per-instance state (per Pipe instance)
        self._plugins: list[PluginBase] = []
        self._hook_subscribers: dict[str, list[tuple[PluginBase, int]]] = {}

    def init_plugins(self, pipe: Pipe) -> None:
        """Instantiate all registered plugin classes and call on_init().

        Builds per-hook subscriber lists sorted by priority (descending).
        Idempotent — calling twice is a no-op.
        """
        if self._plugins:
            return  # Already initialized
        for cls in self._plugin_classes:
            try:
                plugin_id = getattr(cls, "plugin_id", "") or cls.__name__
                plugin_logger = logging.getLogger(f"{__name__}.{plugin_id}")
                ctx = PluginContext(pipe=pipe, logger=plugin_logger)
                instance = cls()
                instance.on_init(ctx)
                # Warn on duplicate plugin_id (copy-paste error detection)
                existing_ids = {p.plugin_id for p in self._plugins}
                if instance.plugin_id and instance.plugin_id in existing_ids:
                    _pr_logger.warning(
                        "Duplicate plugin_id '%s' — plugin '%s' shares ID with an existing plugin",
                        instance.plugin_id, cls.__name__,
                    )
                self._plugins.append(instance)
                _pr_logger.debug(
                    "Plugin '%s' v%s initialized",
                    instance.plugin_id or cls.__name__,
                    instance.plugin_version,
                )
            except Exception:
                _pr_logger.warning(
                    "Plugin '%s' failed to initialize",
                    getattr(cls, "plugin_id", cls.__name__),
                    exc_info=True,
                )

        # Build per-hook subscriber lists
        for hook_name in _PR_SUBSCRIBABLE_HOOKS:
            subscribers: list[tuple[PluginBase, int]] = []
            for plugin in self._plugins:
                if hook_name in plugin.hooks:
                    subscribers.append((plugin, plugin.hooks[hook_name]))
            # Sort by priority descending (higher number = runs first)
            subscribers.sort(key=lambda x: x[1], reverse=True)
            self._hook_subscribers[hook_name] = subscribers

    # ── Dispatch methods ──

    def dispatch_on_models(self, models: list[dict[str, Any]]) -> None:
        """Void dispatch: plugins mutate the models list in place.

        Guards against using ``async def`` instead of ``def`` (common mistake).
        """
        for plugin, _priority in self._hook_subscribers.get("on_models", ()):
            try:
                result = plugin.on_models(models)
                if asyncio.iscoroutine(result):
                    result.close()  # prevent ResourceWarning
                    _pr_logger.warning(
                        "Plugin '%s' on_models returned coroutine — use 'def', not 'async def'",
                        plugin.plugin_id,
                    )
            except Exception:
                _pr_logger.debug(
                    "Plugin '%s' on_models failed", plugin.plugin_id, exc_info=True,
                )

    async def dispatch_on_request(
        self,
        body: dict[str, Any],
        user: dict[str, Any],
        metadata: dict[str, Any],
        event_emitter: Any,
        task: Any,
        *,
        valves: Any = None,
    ) -> dict[str, Any] | str | None:
        """Chain dispatch: all subscribers run in priority order.

        Each plugin receives the current accumulated result via the
        ``current_result`` kwarg. Return non-None to set or replace the
        result. Return None to leave the current result unchanged.
        """
        result: dict[str, Any] | str | None = None

        for plugin, _priority in self._hook_subscribers.get("on_request", ()):
            try:
                plugin_result = await asyncio.wait_for(
                    plugin.on_request(
                        body, user, metadata, event_emitter, task,
                        valves=valves,
                        current_result=result,
                    ),
                    timeout=_PR_DISPATCH_TIMEOUT,
                )
                if plugin_result is not None:
                    result = plugin_result
            except asyncio.TimeoutError:
                _pr_logger.warning(
                    "Plugin '%s' on_request timed out after %.0fs",
                    plugin.plugin_id, _PR_DISPATCH_TIMEOUT,
                )
            except Exception:
                _pr_logger.debug(
                    "Plugin '%s' on_request failed", plugin.plugin_id, exc_info=True,
                )
        return result

    async def dispatch_on_request_transform(
        self,
        body: dict[str, Any],
        model: str,
        valves: Any,
        *,
        user: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Void dispatch: plugins mutate body dict in place.

        The ``model`` parameter is re-read from ``body["model"]`` after each
        plugin, so plugins that remap models see the updated value.
        """
        for plugin, _priority in self._hook_subscribers.get("on_request_transform", ()):
            try:
                await asyncio.wait_for(
                    plugin.on_request_transform(
                        body, model, valves,
                        user=user, metadata=metadata,
                    ),
                    timeout=_PR_DISPATCH_TIMEOUT,
                )
                # Re-read model from body after each plugin (plugins may remap)
                model = str(body.get("model", model))
            except asyncio.TimeoutError:
                _pr_logger.warning(
                    "Plugin '%s' on_request_transform timed out after %.0fs",
                    plugin.plugin_id, _PR_DISPATCH_TIMEOUT,
                )
            except Exception:
                _pr_logger.debug(
                    "Plugin '%s' on_request_transform failed",
                    plugin.plugin_id,
                    exc_info=True,
                )

    async def dispatch_on_emitter_wrap(
        self,
        stream_emitter: Any,
        *,
        raw_emitter: Any = None,
        job_metadata: dict[str, Any] | None = None,
        valves: Any = None,
    ) -> Any | None:
        """Chain dispatch: each subscriber may wrap the stream emitter.

        Plugins run in priority order (highest first).  Each receives the
        current emitter (which may already be wrapped by a prior plugin).
        Return non-None to replace the emitter.  Return ``None`` to leave it
        unchanged.  Returns the final emitter, or ``None`` if no plugin
        wrapped it.
        """
        current = stream_emitter
        changed = False
        for plugin, _priority in self._hook_subscribers.get("on_emitter_wrap", ()):
            try:
                result = await asyncio.wait_for(
                    plugin.on_emitter_wrap(
                        current,
                        raw_emitter=raw_emitter,
                        job_metadata=job_metadata or {},
                        valves=valves,
                    ),
                    timeout=_PR_DISPATCH_TIMEOUT,
                )
                if result is not None:
                    current = result
                    changed = True
            except asyncio.TimeoutError:
                _pr_logger.warning(
                    "Plugin '%s' on_emitter_wrap timed out after %.0fs",
                    plugin.plugin_id, _PR_DISPATCH_TIMEOUT,
                )
            except Exception:
                _pr_logger.debug(
                    "Plugin '%s' on_emitter_wrap failed",
                    plugin.plugin_id,
                    exc_info=True,
                )
        return current if changed else None

    async def dispatch_on_response_transform(
        self,
        completion_data: dict[str, Any],
        model: str,
        metadata: dict[str, Any],
        *,
        user_id: str = "",
        user: Any = None,
    ) -> None:
        """Void dispatch: plugins mutate completion_data dict in place.

        ``completion_data`` is the exact dict sent to OWUI via ``chat:completion``.
        All dicts are passed by reference — no copies.
        """
        for plugin, _priority in self._hook_subscribers.get("on_response_transform", ()):
            try:
                await asyncio.wait_for(
                    plugin.on_response_transform(
                        completion_data, model, metadata,
                        user_id=user_id, user=user,
                    ),
                    timeout=_PR_DISPATCH_TIMEOUT,
                )
            except asyncio.TimeoutError:
                _pr_logger.warning(
                    "Plugin '%s' on_response_transform timed out after %.0fs",
                    plugin.plugin_id, _PR_DISPATCH_TIMEOUT,
                )
            except Exception:
                _pr_logger.debug(
                    "Plugin '%s' on_response_transform failed",
                    plugin.plugin_id,
                    exc_info=True,
                )

    def dispatch_on_shutdown(self) -> None:
        """Broadcast: all plugins notified on shutdown."""
        for plugin in self._plugins:
            try:
                plugin.on_shutdown()
            except Exception:
                _pr_logger.debug(
                    "Plugin '%s' on_shutdown failed", plugin.plugin_id, exc_info=True,
                )
