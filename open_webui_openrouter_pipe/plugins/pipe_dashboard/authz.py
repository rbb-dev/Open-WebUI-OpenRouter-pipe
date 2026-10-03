"""Authorization chokepoint for the pipe_dashboard dashboard and actions.
"""

from __future__ import annotations

import logging
from typing import Any

from ...core.warn_latch import warn_level

logger = logging.getLogger(__name__)

UNDETERMINED = object()
_warned_undeterminable: dict[str, float] = {}

_VIEW_MODEL_UNSET = object()
_VIEW_MODEL_UNREAD = object()

_PD_MODEL_SUFFIX = "pipe-dashboard"


def model_id(pipe: Any) -> str | None:
    pid = getattr(pipe, "id", None)
    return f"{pid}.{_PD_MODEL_SUFFIX}" if pid else None


def _owui() -> Any:
    from types import SimpleNamespace

    from open_webui.config import BYPASS_ADMIN_ACCESS_CONTROL
    from open_webui.env import BYPASS_MODEL_ACCESS_CONTROL
    from open_webui.models.access_grants import AccessGrants
    from open_webui.models.models import Models
    from open_webui.models.users import Users
    from open_webui.utils.access_control import check_model_access
    from open_webui.utils.auth import get_verified_user

    return SimpleNamespace(
        Users=Users,
        Models=Models,
        AccessGrants=AccessGrants,
        get_verified_user=get_verified_user,
        check_model_access=check_model_access,
        BYPASS_ADMIN=BYPASS_ADMIN_ACCESS_CONTROL,
        BYPASS_MODEL=BYPASS_MODEL_ACCESS_CONTROL,
    )


async def resolve_user(user_id: str | None) -> Any | None:
    if not user_id:
        return None
    try:
        return await _owui().Users.get_user_by_id(user_id)
    except Exception as exc:
        _level = warn_level(
            _warned_undeterminable,
            f"viewer_authz:{type(exc).__name__}",
            cooldown_s=300.0,
        )
        logger.log(
            _level,
            "pipe_dashboard: could not resolve user %s; the viewer sweep asks again next tick",
            user_id,
            exc_info=True,
        )
        return UNDETERMINED


VIEWER_ID_KEY = "openrouter_pipe_dashboard_user_id"


async def remember_socket_user_id(sid: str, user_id: str) -> bool:
    """Pin an authorized viewer's id to the socket.io session for the socket's lifetime.

    Open WebUI identifies a socket through ``SESSION_POOL``, which its reaper deletes
    after ``SESSION_POOL_TIMEOUT`` seconds without a heartbeat while the socket is still
    connected and still in the viewers room. Reading only that store makes a live admin
    indistinguishable from an anonymous socket, and the reauthorization sweep evicts them
    for it. The socket.io session is owned by python-socketio, not by Open WebUI, so it
    exists on every Open WebUI version and dies only with the connection.
    """
    try:
        from open_webui.socket.main import sio

        try:
            existing = await sio.get_session(sid)
        except Exception:
            logger.debug("pipe_dashboard: no existing session for socket %s", sid, exc_info=True)
            existing = None
        merged = dict(existing) if isinstance(existing, dict) else {}
        merged[VIEWER_ID_KEY] = user_id
        await sio.save_session(sid, merged)
        return True
    except Exception:
        logger.debug("pipe_dashboard: could not pin viewer id for socket %s", sid, exc_info=True)
        return False


async def resolve_socket_user_id(sid: str) -> str | None:
    try:
        from open_webui.socket.main import sio

        session = await sio.get_session(sid)
        uid = session.get(VIEWER_ID_KEY) if isinstance(session, dict) else None
        if isinstance(uid, str) and uid:
            return uid
    except Exception:
        logger.debug("pipe_dashboard: no pinned viewer id for socket %s", sid, exc_info=True)
    try:
        from open_webui.socket.main import get_user_id_from_session_pool

        return get_user_id_from_session_pool(sid)
    except Exception:
        logger.debug("pipe_dashboard: no user for socket %s", sid, exc_info=True)
        return None


async def _authorized(
    user: Any, pipe: Any, permission: str, model: Any = _VIEW_MODEL_UNSET
) -> bool | None:
    from fastapi import HTTPException

    if user is None:
        return False
    mid = model_id(pipe)
    if not mid:
        return False
    try:
        o = _owui()
        o.get_verified_user(user)
        if model is _VIEW_MODEL_UNSET:
            model = await o.Models.get_model_by_id(mid)
        if permission == "read":
            await o.check_model_access(user, model, bypass_filter=o.BYPASS_MODEL)
            if o.BYPASS_MODEL or (user.role == "admin" and o.BYPASS_ADMIN):
                return True
            if model is None:
                return False
            if user.role != "admin":
                return True
            if user.id == model.user_id:
                return True
            return await o.AccessGrants.has_access(
                user_id=user.id, resource_type="model", resource_id=mid, permission="read",
            )
        if model is None:
            return False
        if user.role == "admin" and o.BYPASS_ADMIN:
            return True
        if user.id == model.user_id:
            return True
        return await o.AccessGrants.has_access(
            user_id=user.id, resource_type="model", resource_id=mid, permission="write",
        )
    except HTTPException:
        logger.debug(
            "pipe_dashboard: %s access denied for user %s on %s",
            permission,
            getattr(user, "id", None),
            mid,
            exc_info=True,
        )
        return False
    except Exception as exc:
        _level = warn_level(
            _warned_undeterminable,
            f"viewer_authz:{type(exc).__name__}",
            cooldown_s=300.0,
        )
        logger.log(
            _level,
            "pipe_dashboard: %s access undeterminable for user %s on %s; the viewer sweep "
            "asks again next tick",
            permission,
            getattr(user, "id", None),
            mid,
            exc_info=True,
        )
        return None


async def resolve_view_model(pipe: Any) -> tuple[bool, Any]:
    mid = model_id(pipe)
    if not mid:
        return False, None
    try:
        return True, await _owui().Models.get_model_by_id(mid)
    except Exception:
        logger.debug(
            "pipe_dashboard: the dashboard model row could not be read; the viewer sweep "
            "asks again next tick",
            exc_info=True,
        )
        return False, None


async def can_view_known(user: Any, pipe: Any, model: Any = _VIEW_MODEL_UNSET) -> bool | None:
    if user is UNDETERMINED or model is _VIEW_MODEL_UNREAD:
        return None
    return await _authorized(user, pipe, "read", model)


async def can_act_known(user: Any, pipe: Any) -> bool | None:
    if user is UNDETERMINED:
        return None
    return await _authorized(user, pipe, "write")


async def can_view(user: Any, pipe: Any, model: Any = _VIEW_MODEL_UNSET) -> bool:
    return await can_view_known(user, pipe, model) is True


async def can_act(user: Any, pipe: Any) -> bool:
    return await can_act_known(user, pipe) is True
