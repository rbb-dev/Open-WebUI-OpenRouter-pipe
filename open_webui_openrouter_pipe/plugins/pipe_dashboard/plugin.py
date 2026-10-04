"""Pipe Dashboard plugin — virtual model providing a live dashboard."""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any, ClassVar

from pydantic import Field

from .._utils import extract_task_name, extract_user_message
from ..base import PluginBase, PluginContext
from ..registry import PluginRegistry
from .actions import _dashboard_enabled
from .auth import ACCESS_DENIED_MD, UNDETERMINED_MD
from .authz import UNDETERMINED, can_view, model_id, resolve_user
from .command_registry import CommandRegistry

# Trigger command auto-imports so @register_command decorators fire
from .commands.help_cmd import handle_help as _pd_commands_loaded  # noqa: F401
from .config_service import persisted_dashboard_enabled
from .context import CommandContext
from .dashboard_publisher import (
    clear_snapshot_getter,
    run_dashboard_publisher,
    set_snapshot_getter,
)
from .dashboard_socket import clear_socket_pipe_getter, register_socket_handler
from .http_routes import (
    _plugins_enabled,
    clear_fresh_dispatch,
    clear_routes_pipe_getter,
    register_action_route,
    set_pipe_getter,
)
from .session_tracker import _ST_SWEEP_INTERVAL, _ST_SWEEP_JITTER_S, SessionTracker
from .update_service import DEFAULT_REPO
from .usage_store import UsageStore

logger = logging.getLogger(__name__)

_PIPE_DASHBOARD_MODEL_ID = "pipe-dashboard"
_PD_OFF_META_KEY = "openrouter_pipe:dashboard_switched_off_by_pipe"


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _pd_stored_meta(row: Any) -> dict[str, Any]:
    stored = getattr(row, "meta", None)
    dump = getattr(stored, "model_dump", None)
    if callable(dump):
        stored = dump()
    return stored if isinstance(stored, dict) else {}


def _overlay_claims_off(row: Any) -> bool:
    return bool(_pd_stored_meta(row).get(_PD_OFF_META_KEY))


def _overlay_target(row: Any, enabled: bool) -> tuple[bool, bool]:
    active = bool(getattr(row, "is_active", False))
    claimed = _overlay_claims_off(row)
    if not enabled:
        return (False, True) if active else (active, claimed)
    if not active and claimed:
        return True, False
    return active, claimed


def _overlay_update_form(
    model_form_cls: Any,
    model_meta_cls: Any,
    model_params_cls: Any,
    fresh: Any,
    display_name: str,
    description: str,
    is_active: bool | None = None,
    claim: bool | None = None,
) -> Any:
    if fresh is None:
        return None
    fresh_meta = getattr(fresh, "meta", None)
    meta_dict: dict[str, Any] = {}
    if fresh_meta:
        meta_dict = fresh_meta.model_dump() if hasattr(fresh_meta, "model_dump") else dict(fresh_meta)
    meta_dict["description"] = description
    if claim is True:
        meta_dict[_PD_OFF_META_KEY] = True
    elif claim is False:
        meta_dict.pop(_PD_OFF_META_KEY, None)
    return model_form_cls(
        id=fresh.id,
        base_model_id=fresh.base_model_id,
        name=display_name,
        meta=model_meta_cls(**meta_dict),
        params=fresh.params if fresh.params else model_params_cls(),
        is_active=fresh.is_active if is_active is None else is_active,
    )


def _registry_pricing(model_id: str) -> dict[str, Any] | None:
    """Pricing dict for a model id (accepts dotted pipe-prefixed ids)."""
    try:
        from ...models.registry import OpenRouterModelRegistry

        norm = model_id.split(".", 1)[-1] if "." in model_id else model_id
        spec = OpenRouterModelRegistry._specs.get(norm) or OpenRouterModelRegistry._specs.get(model_id) or {}
        pricing = spec.get("pricing")
        return pricing if isinstance(pricing, dict) else None
    except (ImportError, AttributeError, TypeError):
        return None


def _dashboard_observability_needed(valves: Any) -> bool:
    return bool(getattr(valves, "PIPE_DASHBOARD_ENABLE", False)) or bool(
        getattr(valves, "PIPE_DASHBOARD_USAGE_COLLECT", False)
    )


def _update_auto_wanted(ctx: Any) -> bool:
    valves = getattr(ctx, "valves", None)
    return bool(getattr(valves, "PIPE_DASHBOARD_UPDATE_AUTO", False))


def _retired(ctx: Any) -> bool:
    return bool(getattr(getattr(ctx, "pipe", None), "_closed", False))


def _registry_model_name(model_id: str) -> str:
    try:
        from .formatters import build_model_name_map, resolve_model_name

        return resolve_model_name(model_id, build_model_name_map())
    except (ImportError, AttributeError, TypeError):
        return model_id


_ST_LIVENESS_EVENT_TYPES = frozenset({
    "chat:message:delta",
    "response.output_text.delta",
    "fusion:event",
    "fusion_inner:reasoning.delta",
})


@PluginRegistry.register
class PipeDashboardPlugin(PluginBase):
    """Virtual model that acts as a live dashboard.

    Subscribes to ``on_models`` (to inject the virtual model) and
    ``on_request`` (to intercept messages sent to it).
    """

    plugin_id = "pipe-dashboard"
    plugin_name = "Pipe Dashboard"
    plugin_version = "1.0.0"
    hooks: ClassVar[dict[str, int]] = {
        "on_models": 50,
        "on_request": 50,
        "on_emitter_wrap": 50,
        "on_tool_result": 50,
        "on_request_retry": 50,
        "on_request_alive": 50,
        "on_generation_complete": 50,
    }
    _master_switch_request_id: str | None = None
    _master_switch_on: bool = False

    plugin_valves: ClassVar[dict[str, tuple]] = {
        "PIPE_DASHBOARD_ENABLE": (bool, Field(
            default=False,
            title="Enable Pipe Dashboard plugin",
            description="Enable the Pipe Dashboard virtual model in the model selector. "
                        "Governs this install only: with two copies of the pipe installed, "
                        "switching one off closes that copy's panel and evicts that copy's "
                        "viewers, and leaves the other running. Read from the persisted row, "
                        "so a committed change holds on every worker without a restart; an "
                        "unreadable row refuses. The Open WebUI model row behind the dashboard "
                        "is switched off and back on with this valve, so the admin model lists "
                        "agree with the selector; a row an administrator switched off themselves "
                        "is left as they set it.",
        )),
        "PIPE_DASHBOARD_USAGE_COLLECT": (bool, Field(
            default=False,
            title="Collect usage records for the Usage tab",
            description=(
                "Persist one record per completed request (user, model, tokens, tools, cost) "
                "to a dedicated dashboard_ table so the dashboard's Usage tab can show usage over time. "
                "Off by default; records are purged after the configured retention. "
                "A request that never reaches a terminal state is recorded as `failed` after two hours "
                "of silence, and that happens on a timer rather than when someone is looking. "
                "That reaper is armed by the first request that starts tracking and by the model-list "
                "pass, so turning collection on brings it up within one chat turn or one model-list "
                "request, with no restart and whether or not the dashboard model itself is on. "
                "Turning it off stops records written by the background abandon sweep as well as by the "
                "request path, on every worker including ones that have served no request, with no "
                "restart; the setting is read again when the row is written, so a request already in flight when you switch it off records nothing. "
                "A stored row this server cannot decrypt counts as off, and the refusal is reported in the "
                "log naming the read."
            ),
        )),
        "PIPE_DASHBOARD_USAGE_RETENTION_DAYS": (int, Field(
            default=30,
            ge=1,
            le=365,
            title="Usage record retention (days)",
            description="How long collected usage records are kept before the purge task deletes them. Read from this saved setting on every pass, at the declared default when it cannot be read.",
        )),
        "PIPE_DASHBOARD_UPDATE_ENABLE": (bool, Field(
            default=True,
            title="Enable the Update tab",
            description=(
                "Show the dashboard's Update tab and allow its actions (check, apply, restore, "
                "delete snapshot). Off: the tab reports disabled and every update action fails "
                "closed, including auto-update. The tab also refuses when the stored valve set "
                "cannot be read at all — the database is unreachable, or the stored set will "
                "not decrypt under the server's application secret (`WEBUI_SECRET_KEY`, or the "
                "deprecated `WEBUI_JWT_SECRET_KEY` it falls back to) — and says so rather than "
                "reporting an admin disable. It also refuses when the update service is not up "
                "on this worker yet; the tab says so and retries by itself, and that is not an "
                "admin disable."
            ),
        )),
        "PIPE_DASHBOARD_UPDATE_SNAPSHOT_KEEP": (int, Field(
            default=3,
            ge=1,
            le=10,
            title="Update snapshots to keep",
            description=(
                "How many previous-version snapshots the updater retains in Open WebUI Files. "
                "Oldest snapshots are pruned (file record first, then blob) when a new one "
                "exceeds the limit. An unreadable stored row prunes nothing on that pass: the "
                "snapshot is still taken and the ring stays capped at ten slots regardless."
            ),
        )),
        "PIPE_DASHBOARD_UPDATE_REPO": (str, Field(
            default=DEFAULT_REPO,
            title="Update source repo (owner/repo)",
            description=(
                "GitHub repo the updater tracks for tagged releases — point it at a fork to "
                "self-update from your own builds. owner/repo on github.com only; the browser "
                "never supplies it. An unreadable stored row refuses the check rather than "
                "naming the upstream default."
            ),
        )),
        "PIPE_DASHBOARD_UPDATE_AUTO": (bool, Field(
            default=False,
            title="Auto-update",
            description=(
                "Apply eligible new releases automatically after the quarantine delay. Runs "
                "headless: whenever this, the Update tab and the plugin system master switch "
                "are all enabled, the background task keeps updating even while the Pipe "
                "Dashboard model itself is switched off. Turning the master switch off stops "
                "the next cycle rather than the current one, so an update already in flight "
                "still finishes. A deterministic failure pauses a release on this worker for "
                "a day at most, after which it is attempted again - so a database blip does "
                "not park an upgrade for the life of the process."
            ),
        )),
        "PIPE_DASHBOARD_UPDATE_AUTO_DELAY_HOURS": (int, Field(
            default=168,
            ge=0,
            le=720,
            title="Auto-update quarantine (hours)",
            description=(
                "A release must be at least this old before auto-update applies it — a bad "
                "release published and yanked within the window never reaches auto-updaters. "
                "0 = apply immediately, no quarantine. Default 168 = 7 days."
            ),
        )),
    }
    plugin_user_valves: ClassVar[dict[str, tuple]] = {}

    def __init__(self) -> None:
        super().__init__()
        self._publisher_task: asyncio.Task[None] | None = None
        self._auto_update_task: asyncio.Task[None] | None = None
        self._sweep_task: asyncio.Task[None] | None = None
        self.update_service: Any = None
        self._usage_store = UsageStore()
        self._tracker = SessionTracker(pricing_fn=_registry_pricing, name_fn=_registry_model_name)
        self._tracker.on_finalize = self._persist_usage_row

    def on_init(self, ctx: PluginContext, **kwargs: Any) -> None:
        self.ctx = ctx
        self._get_pipe = lambda: getattr(ctx, "pipe", None)
        self._master_switch_request_id: str | None = None
        self._master_switch_on: bool = False

        from .update_service import UpdateService

        self.update_service = UpdateService(self._get_pipe)

        self._re_register_registrations(self._get_pipe)

    def _re_register_registrations(self, get_pipe: Any) -> None:
        pipe_id = str(getattr(get_pipe(), "id", "") or "")
        register_socket_handler(pipe_id, get_pipe)
        set_pipe_getter(pipe_id, get_pipe)
        set_snapshot_getter(pipe_id, self._live_snapshot)
        register_action_route()

        # Start the per-worker stats publisher background task.
        # The publisher is idle until a dashboard joins the viewers room.
        self._maybe_start_publisher(get_pipe)
        self._maybe_start_sweep()
        self._maybe_start_auto_update()
        self._maybe_start_usage_purge()

    def _maybe_start_usage_purge(self) -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("No event loop — usage purge task deferred")
            return
        usage_store = getattr(self, "_usage_store", None)
        if usage_store is None:
            return
        get_pipe = getattr(self, "_get_pipe", None)
        pipe = get_pipe() if get_pipe else None
        store = getattr(pipe, "_artifact_store", None) if pipe else None
        if store is None:
            return
        try:
            usage_store.start_retention_purge(store, self._retention_days)
        except Exception:
            logger.debug("usage purge task start failed", exc_info=True)

    def _maybe_start_auto_update(self) -> None:
        """Start the auto-update loop task if an event loop is available."""
        if _retired(getattr(self, "ctx", None)):
            return
        if self.update_service is None:
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logging.getLogger(__name__).debug("No event loop — auto-update task deferred")
            return
        if self._auto_update_task is None or self._auto_update_task.done():
            self._auto_update_task = loop.create_task(
                self.update_service.run_auto_loop(), name="openrouter-update-auto"
            )

    async def _plugin_system_on_for(self, request_id: str = "") -> bool:
        key = str(request_id or "")
        if key and self._master_switch_request_id == key:
            return self._master_switch_on
        on = await _plugins_enabled(getattr(getattr(self, "ctx", None), "pipe", None))
        if key:
            self._master_switch_request_id = key
            self._master_switch_on = on
        return on

    def _maybe_start_sweep(self) -> None:
        if _retired(getattr(self, "ctx", None)):
            return
        valves = getattr(getattr(self, "ctx", None), "valves", None)
        if not _dashboard_observability_needed(valves):
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.debug("No event loop — abandon-sweep task deferred")
            return
        sweep_task = getattr(self, "_sweep_task", None)
        if sweep_task is None or sweep_task.done():
            self._sweep_task = loop.create_task(self._sweep_loop(), name="openrouter-dashboard-sweep")

    async def _sweep_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(_ST_SWEEP_INTERVAL + random.uniform(0.0, _ST_SWEEP_JITTER_S))
                self._tracker.sweep()
            except asyncio.CancelledError:
                return

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
        if getattr(pipe, "_closed", False):
            return
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
        if not await _plugins_enabled(self.ctx.pipe):
            return
        _display_name = "Pipe Dashboard"
        _description = (
            "Live dashboard for pipe monitoring and diagnostics. "
            "Access: a read grant = view the dashboard; a write grant = run operator actions; "
            "the Config tab = the admin role."
        )
        _dashboard_on, _gate_read_ok = await persisted_dashboard_enabled(self.ctx.pipe)
        self._maybe_start_sweep()
        if _update_auto_wanted(self.ctx):
            self._maybe_start_auto_update()
        # Write a clean display name into OWUI's Models table so the UI shows
        # "Pipe Dashboard" instead of the ugly concatenated format.
        await self._ensure_model_overlay(_display_name, _description, _dashboard_on, _gate_read_ok)
        if not _dashboard_on:
            return
        models.append({"id": _PIPE_DASHBOARD_MODEL_ID, "name": _display_name})
        # Retry both on every model-list request — on_init may fire before
        # OWUI's socket module or the event loop is ready on this worker.
        get_pipe = getattr(self, "_get_pipe", None)
        pipe_id = self._registration_pipe_id()
        register_socket_handler(pipe_id, get_pipe)
        if get_pipe is not None:
            self._re_register_registrations(get_pipe)

    async def _ensure_model_overlay(
        self,
        display_name: str,
        description: str,
        enabled: bool | None = None,
        read_ok: bool = True,
    ) -> None:
        """Create or update the OWUI Models table entry for this virtual model."""
        try:
            from open_webui.models.models import (
                ModelForm,
                ModelMeta,
                ModelParams,
                Models,
            )
            from open_webui.models.users import Users

            owui_model_id = model_id(self.ctx.pipe)
            if not owui_model_id:
                return
            if not read_ok:
                logger.debug(
                    "pipe-dashboard model overlay: the valve row could not be read, so the "
                    "overlay row is left exactly as it is stored"
                )
                return
            if enabled is None:
                enabled = await _dashboard_enabled(self.ctx.pipe)
            row = await Models.get_model_by_id(owui_model_id)
            reread = False
            if row is None:
                if not enabled:
                    return
                owner_id = ""
                try:
                    admin = await Users.get_super_admin_user()
                    owner_id = getattr(admin, "id", "") or ""
                except Exception:
                    logger.warning(
                        "pipe-dashboard overlay: super-admin lookup failed; inserting with empty owner",
                        exc_info=True,
                    )
                    owner_id = ""
                form = ModelForm(
                    id=owui_model_id,
                    base_model_id=None,
                    name=display_name,
                    meta=ModelMeta(description=description),
                    params=ModelParams(),
                    is_active=True,
                )
                inserted = await Models.insert_new_model(form, user_id=owner_id)
                if inserted is not None:
                    return
                row = await Models.get_model_by_id(owui_model_id)
                reread = True
                if row is None:
                    return
            described = row.name != display_name or _pd_stored_meta(row).get("description") != description
            active, claim = _overlay_target(row, bool(enabled))
            if (
                not described
                and active == bool(getattr(row, "is_active", False))
                and claim == _overlay_claims_off(row)
            ):
                return
            if described and not reread:
                row = await Models.get_model_by_id(owui_model_id)
                if row is None:
                    return
                active, claim = _overlay_target(row, bool(enabled))
            form = _overlay_update_form(
                ModelForm, ModelMeta, ModelParams, row, display_name, description,
                is_active=active, claim=claim,
            )
            if form is None:
                return
            await Models.update_model_by_id(owui_model_id, form)
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
        self._maybe_start_usage_purge()
        requested_model = str(body.get("model", ""))
        if not self._is_our_model(requested_model):
            if _dashboard_observability_needed(getattr(self.ctx, "valves", None)) and (
                await self._plugin_system_on_for(
                    str(kwargs.get("request_id") or "")
                )
            ):
                self._maybe_start_sweep()
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

        if not await _plugins_enabled(self.ctx.pipe):
            return None

        # Plugin disabled — don't handle requests for our model
        if not await _dashboard_enabled(self.ctx.pipe):
            return None

        acting_user = await resolve_user(user.get("id"))
        if acting_user is UNDETERMINED:
            return self.ctx.build_response(
                model=_PIPE_DASHBOARD_MODEL_ID,
                content=UNDETERMINED_MD,
            )
        if not await can_view(acting_user, self.ctx.pipe):
            return self.ctx.build_response(
                model=_PIPE_DASHBOARD_MODEL_ID,
                content=ACCESS_DENIED_MD,
            )

        task_name = self._extract_task_name(task)
        if task_name:
            return self.ctx.build_response(
                model=_PIPE_DASHBOARD_MODEL_ID,
                content=self._build_task_fallback(task_name),
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
            logger.warning("pipe-dashboard command %r failed", entry.name, exc_info=True)
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
        return ""

    async def on_emitter_wrap(self, stream_emitter: Any, **kwargs: Any) -> Any | None:
        job_metadata = kwargs.get("job_metadata") or {}
        request_id = str(job_metadata.get("request_id") or "") if isinstance(job_metadata, dict) else ""
        if not request_id:
            return None
        if not _dashboard_observability_needed(getattr(getattr(self, "ctx", None), "valves", None)):
            return None
        if not await self._plugin_system_on_for(request_id):
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
                    elif etype in _ST_LIVENESS_EVENT_TYPES:
                        tracker.mark_stream_alive(request_id)
                    elif etype == "response.output_item.added":
                        item = event.get("item") or {}
                        if (
                            isinstance(item, dict)
                            and item.get("type") == "function_call"
                            and item.get("status") == "in_progress"
                        ):
                            tracker.tool_started(request_id, str(item.get("name") or "?"))
            except (AttributeError, KeyError, TypeError, ValueError):
                pass
            return result

        return _wrapped

    async def on_tool_result(self, tool_name: str, status: str, **kwargs: Any) -> None:
        try:
            self._tracker.tool_result(str(kwargs.get("request_id") or ""), str(status))
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

    async def on_request_retry(self, kind: str, **kwargs: Any) -> None:
        try:
            self._tracker.retry(str(kwargs.get("request_id") or ""))
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

    async def on_request_alive(self, request_id: str = "", **kwargs: Any) -> None:
        try:
            if not _dashboard_observability_needed(getattr(getattr(self, "ctx", None), "valves", None)):
                return
            if not await self._plugin_system_on_for(str(request_id or "")):
                return
            self._tracker.mark_stream_alive(str(request_id or ""))
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

    async def on_generation_complete(self, usage: Any, status: str, **kwargs: Any) -> None:
        try:
            self._tracker.finalize(str(kwargs.get("request_id") or ""), usage, str(status))
        except Exception:
            logging.getLogger(__name__).debug("session finalize failed", exc_info=True)

    def _persist_usage_row(self, entry: dict[str, Any]) -> None:
        try:
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
            logger.debug("usage persist failed", exc_info=True)

    async def _retention_days(self) -> int:
        from .actions import _update_service_of

        try:
            get_pipe = getattr(self, "_get_pipe", None)
            pipe = get_pipe() if get_pipe else None
            if pipe is not None:
                svc = _update_service_of(pipe)
                if svc is not None:
                    try:
                        row, stored_read_ok = await svc._row_valves_checked()
                        if not stored_read_ok:
                            logger.warning(
                                "pipe_dashboard: the persisted usage valves are unreadable; "
                                "the usage purge is deleting at the declared default rather "
                                "than the in-memory copy, which would let a failed read "
                                "override an operator's setting"
                            )
                    except Exception:
                        logger.warning(
                            "pipe_dashboard: cannot read the persisted usage valves; the "
                            "usage purge is deleting at the declared default rather than the "
                            "in-memory copy, which would let a failed read override an "
                            "operator's setting",
                            exc_info=True,
                        )
                        row, stored_read_ok = {}, False
                    if stored_read_ok:
                        return int(row.get("PIPE_DASHBOARD_USAGE_RETENTION_DAYS", 30) or 30)
                    return 30
            return int(getattr(self.ctx.valves, "PIPE_DASHBOARD_USAGE_RETENTION_DAYS", 30))
        except (AttributeError, TypeError, ValueError):
            return 30

    def _live_snapshot(self) -> tuple[list[dict[str, Any]], dict[str, float], int]:
        self._tracker.sweep()
        return self._tracker.live_snapshot()

    def _registration_pipe_id(self) -> str:
        pipe = getattr(getattr(self, "ctx", None), "pipe", None)
        return str(getattr(pipe, "id", "") or "")

    def _clear_module_registrations(self) -> None:
        pipe_id = self._registration_pipe_id()
        for clear, name in (
            (clear_socket_pipe_getter, "_get_pipe"),
            (clear_routes_pipe_getter, "_get_pipe"),
            (clear_snapshot_getter, "_live_snapshot"),
        ):
            try:
                clear(self, name, pipe_id)
            except Exception:
                logger.debug("pipe_dashboard module-global teardown failed", exc_info=True)
        try:
            clear_fresh_dispatch(getattr(getattr(self, "ctx", None), "pipe", None))
        except Exception:
            logger.debug("pipe_dashboard module-global teardown failed", exc_info=True)

    def on_shutdown(self, **kwargs: Any) -> Any:
        self._clear_module_registrations()
        pending: list[Any] = []
        task = self._publisher_task
        if task is not None and not task.done():
            task.cancel()
            pending.append(task)
        auto_task = self._auto_update_task
        if auto_task is not None and not auto_task.done():
            auto_task.cancel()
            pending.append(auto_task)
        sweep_task = self._sweep_task
        if sweep_task is not None and not sweep_task.done():
            sweep_task.cancel()
            pending.append(sweep_task)
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
            logger.warning("pipe-dashboard usage-store shutdown failed", exc_info=True)
            if writer_running and not joined_inline:
                try:
                    self._usage_store.join_writer()
                except (AttributeError, RuntimeError):
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


def release_registrations_for(pipe: Any) -> None:
    registry = getattr(pipe, "_plugin_registry", None)
    for plugin in list(getattr(registry, "_plugins", None) or ()):
        try:
            plugin._clear_module_registrations()
        except Exception:
            logger.debug("pipe_dashboard module-global teardown failed", exc_info=True)
