from __future__ import annotations

import asyncio
import inspect
import logging
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from .base import PluginBase, PluginContext

_PluginT = TypeVar("_PluginT", bound=PluginBase)

if TYPE_CHECKING:
    from ..pipe import Pipe

logger = logging.getLogger(__name__)

_PR_DISPATCH_TIMEOUT = 30.0

_PR_SUBSCRIBABLE_HOOKS = frozenset({
    "on_models",
    "on_request",
    "on_request_transform",
    "on_emitter_wrap",
    "on_tool_result",
    "on_request_retry",
    "on_request_alive",
    "on_generation_complete",
})


class PluginRegistry:
    _plugin_classes: ClassVar[list[type[PluginBase]]] = []

    _pending_valve_fields: ClassVar[dict[str, tuple]] = {}
    _pending_user_valve_fields: ClassVar[dict[str, tuple]] = {}

    @classmethod
    def register(cls, plugin_class: type[_PluginT]) -> type[_PluginT]:
        if plugin_class not in cls._plugin_classes:
            cls._plugin_classes.append(plugin_class)
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
        for field_name, field_spec in getattr(plugin_class, attr_name, {}).items():
            if field_name in target:
                original = field_name
                n = 2
                while field_name in target:
                    field_name = f"{original}_{n}"
                    n += 1
                logger.warning(
                    "Plugin '%s' %s field '%s' renamed to '%s' (name collision)",
                    plugin_class.plugin_id, attr_name, original, field_name,
                )
            target[field_name] = field_spec

    @classmethod
    def build_extended_valves(cls, base_valves_class: type) -> type:
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
        if not cls._pending_user_valve_fields:
            return base_user_valves_class
        from pydantic import create_model
        return create_model(
            "UserValves",
            __base__=base_user_valves_class,
            **cls._pending_user_valve_fields,  # type: ignore[reportArgumentType]
        )

    def __init__(self) -> None:
        self._plugins: list[PluginBase] = []
        self._hook_subscribers: dict[str, list[tuple[PluginBase, int]]] = {}
        self._pipe: Pipe | None = None

    def init_plugins(self, pipe: Pipe) -> None:
        if self._plugins:
            return
        self._pipe = pipe
        for cls in self._plugin_classes:
            try:
                plugin_id = getattr(cls, "plugin_id", "") or cls.__name__
                plugin_logger = logging.getLogger(f"{__name__}.{plugin_id}")
                ctx = PluginContext(pipe=pipe, logger=plugin_logger)
                instance = cls()
                instance.on_init(ctx)
                existing_ids = {p.plugin_id for p in self._plugins}
                if instance.plugin_id and instance.plugin_id in existing_ids:
                    logger.warning(
                        "Duplicate plugin_id '%s' — plugin '%s' shares ID with an existing plugin",
                        instance.plugin_id, cls.__name__,
                    )
                self._plugins.append(instance)
                logger.debug(
                    "Plugin '%s' v%s initialized",
                    instance.plugin_id or cls.__name__,
                    instance.plugin_version,
                )
            except Exception:
                logger.warning(
                    "Plugin '%s' failed to initialize",
                    getattr(cls, "plugin_id", cls.__name__),
                    exc_info=True,
                )

        for hook_name in _PR_SUBSCRIBABLE_HOOKS:
            subscribers: list[tuple[PluginBase, int]] = []
            for plugin in self._plugins:
                if hook_name in plugin.hooks:
                    subscribers.append((plugin, plugin.hooks[hook_name]))
            subscribers.sort(key=lambda x: x[1], reverse=True)
            self._hook_subscribers[hook_name] = subscribers

    async def dispatch_on_models(self, models: list[dict[str, Any]]) -> None:
        if getattr(self._pipe, "_closed", False) is True:
            return
        for plugin, _priority in self._hook_subscribers.get("on_models", ()):
            try:
                result = plugin.on_models(models)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.debug(
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
        request_id: str = "",
    ) -> dict[str, Any] | str | None:
        result: dict[str, Any] | str | None = None

        for plugin, _priority in self._hook_subscribers.get("on_request", ()):
            try:
                plugin_result = await asyncio.wait_for(
                    plugin.on_request(
                        body, user, metadata, event_emitter, task,
                        valves=valves,
                        request_id=request_id,
                        current_result=result,
                    ),
                    timeout=_PR_DISPATCH_TIMEOUT,
                )
                if plugin_result is not None:
                    result = plugin_result
            except TimeoutError:
                logger.warning(
                    "Plugin '%s' on_request timed out after %.0fs",
                    plugin.plugin_id, _PR_DISPATCH_TIMEOUT,
                )
            except Exception:
                logger.debug(
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
        for plugin, _priority in self._hook_subscribers.get("on_request_transform", ()):
            try:
                await asyncio.wait_for(
                    plugin.on_request_transform(
                        body, model, valves,
                        user=user, metadata=metadata,
                    ),
                    timeout=_PR_DISPATCH_TIMEOUT,
                )
                model = str(body.get("model", model))
            except TimeoutError:
                logger.warning(
                    "Plugin '%s' on_request_transform timed out after %.0fs",
                    plugin.plugin_id, _PR_DISPATCH_TIMEOUT,
                )
            except Exception:
                logger.debug(
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
            except TimeoutError:
                logger.warning(
                    "Plugin '%s' on_emitter_wrap timed out after %.0fs",
                    plugin.plugin_id, _PR_DISPATCH_TIMEOUT,
                )
            except Exception:
                logger.debug(
                    "Plugin '%s' on_emitter_wrap failed",
                    plugin.plugin_id,
                    exc_info=True,
                )
        return current if changed else None


    async def dispatch_on_tool_result(self, tool_name: str, status: str, **kwargs: Any) -> None:
        for plugin, _priority in self._hook_subscribers.get("on_tool_result", ()):
            try:
                result = plugin.on_tool_result(tool_name, status, **kwargs)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.debug(
                    "Plugin '%s' on_tool_result failed", plugin.plugin_id, exc_info=True,
                )

    async def dispatch_on_request_retry(self, kind: str, **kwargs: Any) -> None:
        for plugin, _priority in self._hook_subscribers.get("on_request_retry", ()):
            try:
                result = plugin.on_request_retry(kind, **kwargs)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.debug(
                    "Plugin '%s' on_request_retry failed", plugin.plugin_id, exc_info=True,
                )

    async def dispatch_on_request_alive(self, request_id: str = "", **kwargs) -> None:
        for plugin, _priority in self._hook_subscribers.get("on_request_alive", ()):
            try:
                result = plugin.on_request_alive(request_id, **kwargs)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.debug(
                    "Plugin '%s' on_request_alive failed", plugin.plugin_id, exc_info=True,
                )

    async def dispatch_on_generation_complete(
        self,
        usage: dict[str, Any] | None,
        status: str,
        **kwargs: Any,
    ) -> None:
        for plugin, _priority in self._hook_subscribers.get("on_generation_complete", ()):
            try:
                result = plugin.on_generation_complete(usage, status, **kwargs)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.debug(
                    "Plugin '%s' on_generation_complete failed",
                    plugin.plugin_id,
                    exc_info=True,
                )

    def dispatch_on_shutdown(self) -> list[Any]:
        pending: list[Any] = []
        for plugin in self._plugins:
            try:
                result = plugin.on_shutdown()
            except Exception:
                logger.debug(
                    "Plugin '%s' on_shutdown failed", plugin.plugin_id, exc_info=True,
                )
                continue
            if inspect.isawaitable(result):
                pending.append(result)
        return pending
