"""OWUI socket.io integration for the dashboard.

The dashboard iframe emits ``openrouter:pipe_dashboard:sub`` after its
``user-join`` acknowledges; the handler joins that socket to the shared
viewers room. Room membership is the entire "who is watching" state —
socket.io removes members on disconnect and deletes the empty room.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ...core.warn_latch import warn_level
from .authz import (
    _VIEW_MODEL_UNREAD,
    can_view,
    can_view_known,
    remember_socket_user_id,
    resolve_socket_user_id,
    resolve_user,
    resolve_view_model,
)

logger = logging.getLogger(__name__)

VIEWERS_ROOM = "pipe_dashboard_viewers"
DASHBOARD_EVENT = "openrouter:pipe_dashboard"
SUB_EVENT = "openrouter:pipe_dashboard:sub"
DENIED_EVENT = "openrouter:pipe_dashboard:denied"
CONFIG_EVENT = "openrouter:pipe_dashboard:config"

_registered = False
_resync = False
_get_pipe: Any = None

_warned_import_sites: set[str] = set()


async def _socket_dashboard_enabled(pipe: Any) -> bool:
    from .config_service import stored_gate_valves

    merged, read_ok = await stored_gate_valves(
        getattr(pipe, "id", ""), getattr(pipe, "valves", None)
    )
    if not read_ok:
        return False
    if not bool(merged.get("ENABLE_PLUGIN_SYSTEM", False)):
        return False
    return bool(merged.get("PIPE_DASHBOARD_ENABLE", True))


def _current_pipe() -> Any:
    return _get_pipe() if _get_pipe else None


async def _evict(sio: Any, sid: str, reason: str) -> None:
    logger.warning("pipe_dashboard: evicting viewer sid=%s (%s)", sid, reason)
    try:
        await sio.leave_room(sid, VIEWERS_ROOM)
        await sio.emit(DENIED_EVENT, {}, room=sid)
    except Exception:
        logger.warning("pipe_dashboard viewer eviction failed for sid=%s", sid, exc_info=True)


async def _deny(sio: Any, sid: str) -> None:
    try:
        await sio.leave_room(sid, VIEWERS_ROOM)
    except Exception:
        logger.debug("pipe_dashboard leave_room failed for sid=%s", sid, exc_info=True)
    try:
        await sio.emit(DENIED_EVENT, {}, room=sid)
    except Exception:
        logger.debug("pipe_dashboard denied-notice emit failed for sid=%s", sid, exc_info=True)


async def _evict_every_viewer() -> None:
    try:
        from open_webui.socket.main import get_session_ids_from_room, sio
    except Exception:  # noqa: BLE001
        return
    for sid in list(get_session_ids_from_room(VIEWERS_ROOM) or []):
        await _evict(sio, sid, "dashboard disabled")


async def _pipe_dashboard_sub(sid: str, _data: Any = None) -> None:
    global _resync
    pipe = _current_pipe()
    if not await _socket_dashboard_enabled(pipe):
        logger.warning("pipe_dashboard: viewer sid=%s refused (dashboard disabled)", sid)
        try:
            from open_webui.socket.main import sio

            await _deny(sio, sid)
        except Exception:
            logger.debug("pipe_dashboard disabled-notice emit failed for sid=%s", sid, exc_info=True)
        return
    uid = await resolve_socket_user_id(sid)
    user = await resolve_user(uid)
    if not await can_view(user, pipe):
        try:
            from open_webui.socket.main import sio

            await _deny(sio, sid)
        except Exception:
            logger.debug("pipe_dashboard denied-notice emit failed for sid=%s", sid, exc_info=True)
        return
    try:
        from open_webui.socket.main import sio

        await sio.enter_room(sid, VIEWERS_ROOM)
    except Exception:
        logger.debug("pipe_dashboard sub failed for sid=%s", sid, exc_info=True)
        return
    if uid:
        await remember_socket_user_id(sid, uid)
    _resync = True


async def emit_config_changed(rev: Any) -> bool:
    try:
        from open_webui.socket.main import sio
    except ImportError:
        return False
    except Exception:
        logging.getLogger(__name__).warning(
            "open_webui.socket.main failed to import for a reason other than absence; "
            "the features that depend on it are now disabled",
            exc_info=True,
        )
        return False
    try:
        await sio.emit(CONFIG_EVENT, {"rev": rev}, room=VIEWERS_ROOM)
        return True
    except Exception:
        logger.debug("pipe_dashboard config emit failed", exc_info=True)
        return False


async def read_config_rev(pipe_id: str) -> Any:
    try:
        from open_webui.internal.db import get_async_db_context
        from open_webui.models.functions import Function
        from sqlalchemy import select

        async with get_async_db_context() as db:
            result = await db.execute(select(Function.updated_at).filter_by(id=pipe_id))
            rev = result.scalar_one_or_none()
    except Exception:
        logger.debug("pipe_dashboard config rev read failed", exc_info=True)
        return None
    if rev is None:
        logger.debug("pipe_dashboard config rev unavailable for %s", pipe_id)
        return None
    return rev


async def _emit_config_rev(pipe_id: str) -> None:
    await emit_config_changed(await read_config_rev(pipe_id))


_pending_emits: set[Any] = set()


class _ValveEventSink:
    async def handle_event(self, app: Any, event: Any, request: Any = None) -> None:
        name = getattr(event, "event", None)
        pipe = _current_pipe()
        pipe_id = getattr(pipe, "id", None)
        subject = getattr(event, "subject", None)
        if not pipe_id or not isinstance(subject, dict) or subject.get("id") != pipe_id:
            return
        if name == "function.deleted":
            from .plugin import release_registrations_for

            release_registrations_for(pipe)
            pipe.close_when_idle()
            return
        if name != "function.valves_updated":
            return
        if not await _socket_dashboard_enabled(pipe):
            await _evict_every_viewer()
        try:
            task = asyncio.create_task(_emit_config_rev(pipe_id))
            _pending_emits.add(task)
            task.add_done_callback(_pending_emits.discard)
        except RuntimeError:
            logger.debug("pipe_dashboard valve event: no running loop")


_valve_sink = _ValveEventSink()


def register_valve_event_sink() -> bool:
    try:
        from open_webui.events import EVENT_SINKS
    except Exception:
        _level = warn_level(_warned_import_sites, 'events')
        logger.log(
            _level,
            "pipe_dashboard: valve event sink unavailable; instant config push disabled",
            exc_info=True,
        )
        return False
    try:
        EVENT_SINKS[:] = [s for s in EVENT_SINKS if type(s).__name__ != "_ValveEventSink"]
        EVENT_SINKS.append(_valve_sink)
        return True
    except Exception:
        logger.debug("pipe_dashboard valve sink registration failed", exc_info=True)
        return False


async def _publish_function_event(
    name: str, pipe_id: str, actor: Any, request: Any, data: dict[str, Any] | None
) -> bool:
    try:
        from open_webui.events import EVENTS, publish_event
    except Exception:
        _level = warn_level(_warned_import_sites, 'events')
        logger.log(
            _level,
            "pipe_dashboard: Open WebUI events unavailable; dashboard writes are not "
            "announced to event functions or webhooks",
            exc_info=True,
        )
        return False
    if request is None:
        from .http_routes import get_owui_app

        request = get_owui_app()
    try:
        await publish_event(
            request, getattr(EVENTS, name), actor=actor, subject_id=pipe_id, data=data
        )
        return True
    except Exception:
        logger.debug("pipe_dashboard event publish failed", exc_info=True)
        return False


async def publish_valves_changed(pipe_id: str, actor: Any, request: Any = None) -> bool:
    return await _publish_function_event(
        "FUNCTION_VALVES_UPDATED", pipe_id, actor, request, None
    )


async def publish_function_updated(
    pipe_id: str, actor: Any, request: Any = None, data: dict[str, Any] | None = None
) -> bool:
    return await _publish_function_event("FUNCTION_UPDATED", pipe_id, actor, request, data)


def clear_socket_pipe_getter(instance: Any, name: str) -> None:
    global _get_pipe
    current = _get_pipe
    if current is None or current == getattr(instance, name, None):
        _get_pipe = None


def register_socket_handler(get_pipe: Any = None) -> bool:
    global _registered, _get_pipe
    if get_pipe is not None:
        _get_pipe = get_pipe
    register_valve_event_sink()
    if _registered:
        return True
    try:
        from open_webui.socket.main import sio
    except Exception:
        _level = warn_level(_warned_import_sites, 'register')
        logger.log(
            _level,
            "pipe_dashboard: OWUI socket unavailable; the live dashboard cannot start",
            exc_info=True,
        )
        return False
    try:
        sio.on(SUB_EVENT, _pipe_dashboard_sub)
    except Exception:
        logger.debug("pipe_dashboard socket handler registration failed", exc_info=True)
        return False
    _registered = True
    return True


def consume_resync() -> bool:
    global _resync
    if _resync:
        _resync = False
        return True
    return False


def local_viewer_sids() -> list[str]:
    try:
        from open_webui.socket.main import get_session_ids_from_room
    except Exception:
        _level = warn_level(_warned_import_sites, 'viewer_sids')
        logger.log(
            _level,
            "pipe_dashboard: viewer lookup unavailable; no dashboard payloads will be emitted",
            exc_info=True,
        )
        return []
    try:
        return list(get_session_ids_from_room(VIEWERS_ROOM) or [])
    except Exception:
        logger.debug("pipe_dashboard viewer lookup failed", exc_info=True)
        return []


async def emit_dashboard(payload: dict[str, Any]) -> bool:
    if not await _socket_dashboard_enabled(_current_pipe()):
        return False
    try:
        from open_webui.socket.main import sio
    except Exception:
        _level = warn_level(_warned_import_sites, 'emit')
        logger.log(
            _level,
            "pipe_dashboard: OWUI socket unavailable; dashboard payloads are being dropped",
            exc_info=True,
        )
        return False
    try:
        await sio.emit(DASHBOARD_EVENT, payload, room=VIEWERS_ROOM, ignore_queue=True)
        return True
    except Exception:
        logger.debug("pipe_dashboard emit failed", exc_info=True)
        return False


async def reauthorize_local_viewers() -> None:
    pipe = _current_pipe()
    try:
        from open_webui.socket.main import get_session_ids_from_room, sio
    except Exception:
        _level = warn_level(_warned_import_sites, 'reauth')
        logger.log(
            _level,
            "pipe_dashboard: OWUI socket unavailable; viewer revocation checks are disabled",
            exc_info=True,
        )
        return
    enabled = await _socket_dashboard_enabled(pipe)
    if not enabled:
        for sid in list(get_session_ids_from_room(VIEWERS_ROOM) or []):
            await _evict(sio, sid, "dashboard disabled")
        return
    sids = list(get_session_ids_from_room(VIEWERS_ROOM) or [])
    if not sids:
        return
    resolved, model = await resolve_view_model(pipe)
    verdicts: dict[str | None, bool | None] = {}
    for sid in sids:
        uid = await resolve_socket_user_id(sid)
        if uid not in verdicts:
            verdicts[uid] = await can_view_known(
                await resolve_user(uid), pipe, model if resolved else _VIEW_MODEL_UNREAD
            )
        if verdicts[uid] is False:
            await _evict(sio, sid, "authorization no longer holds")


register_socket_handler()
