"""Pipe Stats Dashboard plugin — virtual model providing a live stats dashboard."""

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
from .ephemeral_keys import EphemeralKeyStore
from .._utils import configure_keystore_redis, extract_task_name, extract_user_message
from .sse_stats import register_sse_route
from .stats_publisher import run_stats_publisher

# Trigger command auto-imports so @register_command decorators fire
from . import commands as _commands  # noqa: F401, E402

_PIPE_STATS_MODEL_ID = "pipe-stats"


@PluginRegistry.register
class PipeStatsDashboardPlugin(PluginBase):
    """Virtual model that acts as a live stats dashboard.

    Subscribes to ``on_models`` (to inject the virtual model) and
    ``on_request`` (to intercept messages sent to it).
    """

    plugin_id = "pipe-stats"
    plugin_name = "Pipe Stats Dashboard"
    plugin_version = "1.0.0"
    hooks = {
        "on_models": 50,
        "on_request": 50,
    }
    plugin_valves = {
        "PIPE_STATS_ENABLE": (bool, Field(
            default=False,
            title="Enable Pipe Stats dashboard plugin",
            description="Enable the Pipe Stats Dashboard virtual model in the model selector.",
        )),
    }
    plugin_user_valves = {}

    def __init__(self) -> None:
        super().__init__()
        self._key_store = EphemeralKeyStore()
        self._publisher_task: asyncio.Task[None] | None = None

    def on_init(self, ctx: PluginContext, **kwargs: Any) -> None:
        self.ctx = ctx
        self._get_pipe = lambda: getattr(ctx, "pipe", None)  # noqa: E731

        # Wire Redis into key store for cross-worker key sharing
        configure_keystore_redis(self._key_store, ctx.pipe)

        # Register the SSE endpoint for live dashboard stats
        register_sse_route(self._key_store, self._get_pipe)

        # Start the per-worker stats publisher background task.
        # The publisher is idle (one Redis EXISTS every 5s) until
        # a dashboard SSE session sets the active flag.
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
                run_stats_publisher(get_pipe, _get_redis, namespace),
                name="openrouter-stats-publisher",
            )
            log.debug("Stats publisher task created (ns=%s)", namespace)

    def on_models(self, models: list[dict[str, Any]], **kwargs: Any) -> None:
        if not hasattr(self, "ctx"):
            return
        if not self.ctx.valves.PIPE_STATS_ENABLE:
            return
        _display_name = "Pipe Stats Dashboard"
        _description = "Live stats dashboard for pipe monitoring and diagnostics."
        models.append({"id": _PIPE_STATS_MODEL_ID, "name": _display_name})
        # Write a clean display name into OWUI's Models table so the UI shows
        # "Pipe Stats Dashboard" instead of the ugly concatenated format.
        self._ensure_model_overlay(_display_name, _description)
        # Lazy Redis retry — on_init may fire before Redis is ready
        if self._key_store._redis_client is None:
            configure_keystore_redis(self._key_store, self.ctx.pipe)
        # Ensure the stats publisher is running — on_models fires on every
        # model-list request, so this catches workers that missed on_init.
        if hasattr(self, "_get_pipe"):
            self._maybe_start_publisher(self._get_pipe)

    def _ensure_model_overlay(self, display_name: str, description: str) -> None:
        """Create or update the OWUI Models table entry for this virtual model."""
        try:
            from open_webui.models.models import ModelForm, ModelMeta, ModelParams, Models

            pipe_id = getattr(self.ctx.pipe, "id", "")
            if not pipe_id:
                return
            owui_model_id = f"{pipe_id}.{_PIPE_STATS_MODEL_ID}"
            existing = Models.get_model_by_id(owui_model_id)
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
                Models.update_model_by_id(owui_model_id, form)
            else:
                form = ModelForm(
                    id=owui_model_id,
                    base_model_id=None,
                    name=display_name,
                    meta=ModelMeta(description=description),
                    params=ModelParams(),
                    is_active=True,
                )
                Models.insert_new_model(form, user_id="")
        except Exception:
            # OWUI Models API not available — display name will use the default
            # concatenated format, which is acceptable as a fallback.
            pass

    async def on_request(
        self,
        body: dict[str, Any],
        user: dict[str, Any],
        metadata: dict[str, Any],
        event_emitter: Any,
        task: Any,
        **kwargs: Any,
    ) -> dict[str, Any] | str | None:
        model_id = str(body.get("model", ""))
        if not self._is_our_model(model_id):
            return None  # Not for us — let the request continue

        # Plugin disabled — don't handle requests for our model
        if not self.ctx.valves.PIPE_STATS_ENABLE:
            return None

        # OWUI sends background tasks (title/tags/emoji) to ALL models.
        # Handle them before auth so they don't break the admin's UI.
        # Task stubs return harmless generic content only.
        task_name = self._extract_task_name(task)
        if task_name:
            return self.ctx.build_response(
                model=_PIPE_STATS_MODEL_ID,
                content=self._build_task_fallback(task_name),
            )

        # Authorization check
        if user.get("role") != "admin":
            return self.ctx.build_response(
                model=_PIPE_STATS_MODEL_ID,
                content=ACCESS_DENIED_MD,
            )

        # Extract the user's message text
        command_text = self._extract_user_message(body) or "help"

        # Resolve and dispatch command
        entry, args = CommandRegistry.resolve(command_text)
        if entry is None:
            safe_text = command_text.replace("`", "'")
            return self.ctx.build_response(
                model=_PIPE_STATS_MODEL_ID,
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
        return self.ctx.build_response(model=_PIPE_STATS_MODEL_ID, content=result)

    # ── Private helpers ──

    def _is_our_model(self, model_id: str) -> bool:
        """Check if the model ID refers to this Pipe Stats Dashboard plugin.

        Open WebUI may prefix with ``<pipe-id>.`` for manifold pipes.
        Only the known pipe ID prefix is accepted — arbitrary prefixes
        like ``evil.pipe-stats`` are rejected.
        """
        if not model_id:
            return False
        model_id = model_id.lower()
        if model_id == _PIPE_STATS_MODEL_ID:
            return True
        # Handle pipe-prefixed form: "<pipe_id>.pipe-stats"
        pipe_id = getattr(self.ctx, "pipe_id", "openrouter").lower()
        if model_id == f"{pipe_id}.{_PIPE_STATS_MODEL_ID}":
            return True
        return False

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
            return json.dumps({"tags": ["Stats"]})
        if "title" in name:
            return json.dumps({"title": "Pipe Stats Dashboard"})
        if "emoji" in name:
            return json.dumps({"emoji": ""})
        # Unknown task type — return empty JSON object as safe default
        return "{}"

    def on_shutdown(self, **kwargs: Any) -> None:
        if self._publisher_task is not None and not self._publisher_task.done():
            self._publisher_task.cancel()
