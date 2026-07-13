"""Pipe Dashboard plugin — virtual model providing a live dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from pydantic import Field

from ..base import PluginBase, PluginContext
from ..registry import PluginRegistry
from .auth import ACCESS_DENIED_MD
from .command_registry import CommandRegistry
from .context import CommandContext
from .._utils import extract_task_name, extract_user_message
from .authz import can_view, model_id, resolve_user
from .http_routes import register_action_route, set_pipe_getter
from .dashboard_socket import register_socket_handler
from .dashboard_publisher import run_dashboard_publisher, set_snapshot_getter
from .session_tracker import SessionTracker
from .usage_store import UsageStore

# Trigger command auto-imports so @register_command decorators fire
from .commands.help_cmd import handle_help as _pd_commands_loaded  # noqa: F401, E402

_PIPE_DASHBOARD_MODEL_ID = "pipe-dashboard"


def _registry_pricing(model_id: str) -> dict[str, Any] | None:
    """Pricing dict for a model id (accepts dotted pipe-prefixed ids)."""
    try:
        from ...models.registry import OpenRouterModelRegistry

        norm = model_id.split(".", 1)[-1] if "." in model_id else model_id
        spec = OpenRouterModelRegistry._specs.get(norm) or OpenRouterModelRegistry._specs.get(model_id) or {}
        pricing = spec.get("pricing")
        return pricing if isinstance(pricing, dict) else None
    except Exception:
        return None


def _registry_model_name(model_id: str) -> str:
    try:
        from .formatters import build_model_name_map, resolve_model_name

        return resolve_model_name(model_id, build_model_name_map())
    except Exception:
        return model_id


@PluginRegistry.register
class PipeDashboardPlugin(PluginBase):
    """Virtual model that acts as a live dashboard.

    Subscribes to ``on_models`` (to inject the virtual model) and
    ``on_request`` (to intercept messages sent to it).
    """

    plugin_id = "pipe-dashboard"
    plugin_name = "Pipe Dashboard"
    plugin_version = "1.0.0"
    hooks = {
        "on_models": 50,
        "on_request": 50,
        "on_emitter_wrap": 50,
        "on_tool_result": 50,
        "on_request_retry": 50,
        "on_generation_complete": 50,
    }
    plugin_valves = {
        "PIPE_DASHBOARD_ENABLE": (bool, Field(
            default=False,
            title="Enable Pipe Dashboard plugin",
            description="Enable the Pipe Dashboard virtual model in the model selector.",
        )),
        "PIPE_DASHBOARD_USAGE_COLLECT": (bool, Field(
            default=False,
            title="Collect usage records for the Usage tab",
            description=(
                "Persist one record per completed request (user, model, tokens, tools, cost) "
                "to a dedicated dashboard_ table so the dashboard's Usage tab can show usage over time. "
                "Off by default; records are purged after the configured retention."
            ),
        )),
        "PIPE_DASHBOARD_USAGE_RETENTION_DAYS": (int, Field(
            default=30,
            ge=1,
            le=365,
            title="Usage record retention (days)",
            description="How long collected usage records are kept before the purge task deletes them.",
        )),
    }
    plugin_user_valves = {}

    def __init__(self) -> None:
        super().__init__()
        self._publisher_task: asyncio.Task[None] | None = None
        self._usage_store = UsageStore()
        self._tracker = SessionTracker(pricing_fn=_registry_pricing, name_fn=_registry_model_name)
        self._tracker.on_finalize = self._persist_usage_row

    def on_init(self, ctx: PluginContext, **kwargs: Any) -> None:
        self.ctx = ctx
        self._get_pipe = lambda: getattr(ctx, "pipe", None)  # noqa: E731

        register_socket_handler(self._get_pipe)
        set_pipe_getter(self._get_pipe)
        set_snapshot_getter(self._live_snapshot)
        register_action_route()

        # Start the per-worker stats publisher background task.
        # The publisher is idle until a dashboard joins the viewers room.
        self._maybe_start_publisher(self._get_pipe)

    def _maybe_start_publisher(self, get_pipe: Any) -> None:
        """Start the stats publisher if an event loop is available."""
        log = logging.getLogger(__name__)

        def _get_redis():
            pipe = get_pipe()
            if pipe is None:
                return None, False
            return getattr(pipe, "_redis_client", None), getattr(pipe, "_redis_enabled", False)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            log.debug("No event loop — stats publisher deferred")
            return

        pipe = get_pipe()
        namespace = getattr(pipe, "_redis_namespace", "openrouter") if pipe else "openrouter"

        if self._publisher_task is None or self._publisher_task.done():
            self._publisher_task = loop.create_task(
                run_dashboard_publisher(get_pipe, _get_redis, namespace),
                name="openrouter-dashboard-publisher",
            )
            log.debug("Dashboard publisher task created (ns=%s)", namespace)

    async def on_models(self, models: list[dict[str, Any]], **kwargs: Any) -> None:
        if not hasattr(self, "ctx"):
            return
        if not self.ctx.valves.PIPE_DASHBOARD_ENABLE:
            return
        _display_name = "Pipe Dashboard"
        _description = (
            "Live dashboard for pipe monitoring and diagnostics. "
            "Access: a read grant = view the dashboard; a write grant = run operator actions."
        )
        models.append({"id": _PIPE_DASHBOARD_MODEL_ID, "name": _display_name})
        # Write a clean display name into OWUI's Models table so the UI shows
        # "Pipe Dashboard" instead of the ugly concatenated format.
        await self._ensure_model_overlay(_display_name, _description)
        # Retry both on every model-list request — on_init may fire before
        # OWUI's socket module or the event loop is ready on this worker.
        get_pipe = getattr(self, "_get_pipe", None)
        register_socket_handler(get_pipe)
        if get_pipe is not None:
            set_pipe_getter(get_pipe)
            register_action_route()
            self._maybe_start_publisher(get_pipe)

    async def _ensure_model_overlay(self, display_name: str, description: str) -> None:
        """Create or update the OWUI Models table entry for this virtual model."""
        try:
            from open_webui.models.models import ModelForm, ModelMeta, ModelParams, Models
            from open_webui.models.users import Users

            owui_model_id = model_id(self.ctx.pipe)
            if not owui_model_id:
                return
            existing = await Models.get_model_by_id(owui_model_id)
            if existing is not None:
                # Already exists — update name/description only if they differ
                existing_meta = getattr(existing, "meta", None)
                existing_desc = getattr(existing_meta, "description", None) if existing_meta else None
                if existing.name == display_name and existing_desc == description:
                    return
                meta_dict: dict[str, Any] = {}
                if existing_meta:
                    meta_dict = existing_meta.model_dump() if hasattr(existing_meta, "model_dump") else dict(existing_meta)
                meta_dict["description"] = description
                form = ModelForm(
                    id=existing.id,
                    base_model_id=existing.base_model_id,
                    name=display_name,
                    meta=ModelMeta(**meta_dict),
                    params=existing.params if existing.params else ModelParams(),
                    is_active=existing.is_active,
                )
                await Models.update_model_by_id(owui_model_id, form)
            else:
                owner_id = ""
                try:
                    admin = await Users.get_super_admin_user()
                    owner_id = getattr(admin, "id", "") or ""
                except Exception:
                    owner_id = ""
                form = ModelForm(
                    id=owui_model_id,
                    base_model_id=None,
                    name=display_name,
                    meta=ModelMeta(description=description),
                    params=ModelParams(),
                    is_active=True,
                )
                await Models.insert_new_model(form, user_id=owner_id)
        except Exception:
            logging.getLogger(__name__).debug("pipe-dashboard model overlay ensure failed", exc_info=True)

    async def on_request(
        self,
        body: dict[str, Any],
        user: dict[str, Any],
        metadata: dict[str, Any],
        event_emitter: Any,
        task: Any,
        **kwargs: Any,
    ) -> dict[str, Any] | str | None:
        requested_model = str(body.get("model", ""))
        if not self._is_our_model(requested_model):
            try:
                self._tracker.start(
                    str(kwargs.get("request_id") or ""),
                    body=body,
                    user=user,
                    metadata=metadata,
                    task=task,
                )
            except Exception:
                logging.getLogger(__name__).debug("session track start failed", exc_info=True)
            return None  # Not for us — let the request continue

        # Plugin disabled — don't handle requests for our model
        if not self.ctx.valves.PIPE_DASHBOARD_ENABLE:
            return None

        # OWUI sends background tasks (title/tags/emoji) to ALL models.
        # Handle them before auth so they don't break the admin's UI.
        # Task stubs return harmless generic content only.
        task_name = self._extract_task_name(task)
        if task_name:
            return self.ctx.build_response(
                model=_PIPE_DASHBOARD_MODEL_ID,
                content=self._build_task_fallback(task_name),
            )

        acting_user = await resolve_user(user.get("id"))
        if not await can_view(acting_user, self.ctx.pipe):
            return self.ctx.build_response(
                model=_PIPE_DASHBOARD_MODEL_ID,
                content=ACCESS_DENIED_MD,
            )

        # Extract the user's message text
        command_text = self._extract_user_message(body) or "help"

        # Resolve and dispatch command
        entry, args = CommandRegistry.resolve(command_text)
        if entry is None:
            safe_text = command_text.replace("`", "'")
            return self.ctx.build_response(
                model=_PIPE_DASHBOARD_MODEL_ID,
                content=f"Unknown command: `{safe_text}`\n\nType `help` for available commands.",
            )

        try:
            result = await entry.handler(CommandContext(
                pipe=self.ctx.pipe,
                args=args,
                user=user,
                metadata=metadata,
                event_emitter=event_emitter,
            ))
        except Exception as exc:
            safe_exc = str(exc).replace("`", "'")
            result = f"## Command Error\n\n`{entry.name}` failed: {safe_exc}"
        return self.ctx.build_response(model=_PIPE_DASHBOARD_MODEL_ID, content=result)

    # ── Private helpers ──

    def _is_our_model(self, candidate: str) -> bool:
        """Check if the model ID refers to this Pipe Dashboard plugin.

        Open WebUI may prefix with ``<pipe-id>.`` for manifold pipes.
        Only the known pipe ID prefix is accepted — arbitrary prefixes
        like ``evil.pipe-dashboard`` are rejected.
        """
        if not candidate:
            return False
        mid = candidate.lower()
        if mid == _PIPE_DASHBOARD_MODEL_ID:
            return True
        dotted = model_id(self.ctx.pipe)
        return bool(dotted) and mid == dotted.lower()

    _extract_task_name = staticmethod(extract_task_name)
    _extract_user_message = staticmethod(extract_user_message)

    @staticmethod
    def _build_task_fallback(task_name: str) -> str:
        """Build OWUI task stub content (title/tags/emoji/follow-ups).

        Matching is substring-based, checked in order: follow, tag, title, emoji.
        """
        name = (task_name or "").strip().lower()
        if not name:
            return ""
        if "follow" in name:
            return json.dumps({"follow_ups": []})
        if "tag" in name:
            return json.dumps({"tags": ["Dashboard"]})
        if "title" in name:
            return json.dumps({"title": "Pipe Dashboard"})
        if "emoji" in name:
            return json.dumps({"emoji": ""})
        # Unknown task type — return empty JSON object as safe default
        return "{}"

    async def on_emitter_wrap(self, stream_emitter: Any, **kwargs: Any) -> Any | None:
        job_metadata = kwargs.get("job_metadata") or {}
        request_id = str(job_metadata.get("request_id") or "") if isinstance(job_metadata, dict) else ""
        if not request_id:
            return None
        tracker = self._tracker

        async def _wrapped(event: Any) -> Any:
            result = await stream_emitter(event)
            try:
                if isinstance(event, dict):
                    etype = event.get("type")
                    if etype == "chat:completion":
                        data = event.get("data") or {}
                        usage = data.get("usage") if isinstance(data, dict) else None
                        if isinstance(usage, dict):
                            tracker.update_usage(request_id, usage)
                        else:
                            tracker.mark_streaming(request_id)
                    elif etype == "response.output_item.added":
                        item = event.get("item") or {}
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "function_call"
                            and item.get("status") == "in_progress"
                        ):
                            tracker.tool_started(request_id, str(item.get("name") or "?"))
            except Exception:
                pass
            return result

        return _wrapped

    async def on_tool_result(self, tool_name: str, status: str, **kwargs: Any) -> None:
        try:
            self._tracker.tool_result(str(kwargs.get("request_id") or ""), str(status))
        except Exception:
            pass

    async def on_request_retry(self, kind: str, **kwargs: Any) -> None:
        try:
            self._tracker.retry(str(kwargs.get("request_id") or ""))
        except Exception:
            pass

    async def on_generation_complete(self, usage: Any, status: str, **kwargs: Any) -> None:
        try:
            self._tracker.finalize(str(kwargs.get("request_id") or ""), usage, str(status))
        except Exception:
            logging.getLogger(__name__).debug("session finalize failed", exc_info=True)

    def _persist_usage_row(self, entry: dict[str, Any]) -> None:
        """Finalize callback: enqueue a DB row when collection is enabled (live valve read)."""
        try:
            if not bool(getattr(self.ctx.valves, "PIPE_DASHBOARD_USAGE_COLLECT", False)):
                return
            get_pipe = getattr(self, "_get_pipe", None)
            pipe = get_pipe() if get_pipe else None
            store = getattr(pipe, "_artifact_store", None) if pipe else None
            if store is None:
                return
            if not self._usage_store.enabled and not self._usage_store.ensure(store):
                return
            self._usage_store.start_purge_task(self._retention_days)
            self._usage_store.record(self._tracker.db_row(entry))
        except Exception:
            logging.getLogger(__name__).debug("usage persist failed", exc_info=True)

    def _retention_days(self) -> int:
        try:
            return int(getattr(self.ctx.valves, "PIPE_DASHBOARD_USAGE_RETENTION_DAYS", 30))
        except Exception:
            return 30

    def _live_snapshot(self) -> tuple[list[dict[str, Any]], dict[str, float]]:
        try:
            self._tracker.sweep()
            return self._tracker.live_snapshot()
        except Exception:
            return [], {}

    def on_shutdown(self, **kwargs: Any) -> Any:
        pending: list[Any] = []
        task = self._publisher_task
        if task is not None and not task.done():
            task.cancel()
            pending.append(task)
        writer_running = getattr(self._usage_store, "writer_alive", False)
        joined_inline = False
        try:
            purge = self._usage_store.signal_stop()
            if purge is not None:
                pending.append(purge)
            if writer_running:
                try:
                    asyncio.get_running_loop()
                    pending.append(asyncio.to_thread(self._usage_store.join_writer))
                except RuntimeError:
                    self._usage_store.join_writer()
                    joined_inline = True
        except Exception:
            if writer_running and not joined_inline:
                try:
                    self._usage_store.join_writer()
                except Exception:
                    pass
        if not pending:
            return None
        if len(pending) == 1:
            return pending[0]

        async def _drain(items: list[Any]) -> None:
            await asyncio.gather(*items, return_exceptions=True)

        coro = _drain(pending)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            coro.close()
            return None
        return loop.create_task(coro)
