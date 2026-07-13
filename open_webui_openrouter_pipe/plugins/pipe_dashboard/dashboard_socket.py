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

from .authz import can_view, resolve_socket_user_id, resolve_user

_pd_sock_log = logging.getLogger(__name__)

VIEWERS_ROOM = "pipe_dashboard_viewers"
DASHBOARD_EVENT = "openrouter:pipe_dashboard"
SUB_EVENT = "openrouter:pipe_dashboard:sub"
DENIED_EVENT = "openrouter:pipe_dashboard:denied"
CONFIG_EVENT = "openrouter:pipe_dashboard:config"

_registered = False
_resync = False
_get_pipe: Any = None


async def _pipe_dashboard_sub(sid: str, _data: Any = None) -> None:
    global _resync
    pipe = _get_pipe() if _get_pipe else None
    user = await resolve_user(resolve_socket_user_id(sid))
    if not await can_view(user, pipe):
        try:
            from open_webui.socket.main import sio

            await sio.emit(DENIED_EVENT, {}, room=sid)
        except Exception:
            pass
        return
    try:
        from open_webui.socket.main import sio

        await sio.enter_room(sid, VIEWERS_ROOM)
    except Exception:
        _pd_sock_log.debug("pipe_dashboard sub failed for sid=%s", sid, exc_info=True)
        return
    _resync = True


async def emit_config_changed(rev: Any) -> bool:
    try:
        from open_webui.socket.main import sio
    except Exception:
        return False
    try:
        await sio.emit(CONFIG_EVENT, {"rev": rev}, room=VIEWERS_ROOM)
        return True
    except Exception:
        _pd_sock_log.debug("pipe_dashboard config emit failed", exc_info=True)
        return False


async def read_config_rev(pipe_id: str) -> Any:
    try:
        from open_webui.models.functions import Functions

        function = await Functions.get_function_by_id(pipe_id)
        return getattr(function, "updated_at", None)
    except Exception:
        return None


async def _emit_config_rev(pipe_id: str) -> None:
    await emit_config_changed(await read_config_rev(pipe_id))


_pending_emits: set[Any] = set()


class _ValveEventSink:
    async def handle_event(self, app: Any, event: Any, request: Any = None) -> None:
        if getattr(event, "event", None) != "function.valves_updated":
            return
        pipe = _get_pipe() if _get_pipe else None
        pipe_id = getattr(pipe, "id", None)
        subject = getattr(event, "subject", None)
        if not pipe_id or not isinstance(subject, dict) or subject.get("id") != pipe_id:
            return
        try:
            task = asyncio.create_task(_emit_config_rev(pipe_id))
            _pending_emits.add(task)
            task.add_done_callback(_pending_emits.discard)
        except RuntimeError:
            _pd_sock_log.debug("pipe_dashboard valve event: no running loop")


_valve_sink = _ValveEventSink()


def register_valve_event_sink() -> bool:
    try:
        from open_webui.events import EVENT_SINKS
    except Exception:
        return False
    try:
        EVENT_SINKS[:] = [s for s in EVENT_SINKS if type(s).__name__ != "_ValveEventSink"]
        EVENT_SINKS.append(_valve_sink)
        return True
    except Exception:
        _pd_sock_log.debug("pipe_dashboard valve sink registration failed", exc_info=True)
        return False


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
        return False
    try:
        sio.on(SUB_EVENT, _pipe_dashboard_sub)
    except Exception:
        _pd_sock_log.debug("pipe_dashboard socket handler registration failed", exc_info=True)
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
        return []
    try:
        return list(get_session_ids_from_room(VIEWERS_ROOM) or [])
    except Exception:
        return []


async def emit_dashboard(payload: dict[str, Any]) -> bool:
    try:
        from open_webui.socket.main import sio
    except Exception:
        return False
    try:
        await sio.emit(DASHBOARD_EVENT, payload, room=VIEWERS_ROOM, ignore_queue=True)
        return True
    except Exception:
        _pd_sock_log.debug("pipe_dashboard emit failed", exc_info=True)
        return False


async def reauthorize_local_viewers() -> None:
    pipe = _get_pipe() if _get_pipe else None
    try:
        from open_webui.socket.main import get_session_ids_from_room, sio
    except Exception:
        return
    for sid in list(get_session_ids_from_room(VIEWERS_ROOM) or []):
        user = await resolve_user(resolve_socket_user_id(sid))
        if not await can_view(user, pipe):
            try:
                await sio.leave_room(sid, VIEWERS_ROOM)
                await sio.emit(DENIED_EVENT, {}, room=sid)
            except Exception:
                pass


register_socket_handler()
