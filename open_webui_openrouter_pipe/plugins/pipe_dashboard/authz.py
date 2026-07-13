"""Authorization chokepoint for the pipe_dashboard dashboard and actions.

Reuses OWUI's own access decision with no access logic composed here:
read (view) via ``check_model_access``, write (operate) via OWUI's router
write-formula, and the ``{user, admin}`` role gate via ``get_verified_user``.
"""

from __future__ import annotations

from typing import Any

_PD_MODEL_SUFFIX = "pipe-dashboard"


def model_id(pipe: Any) -> str | None:
    pid = getattr(pipe, "id", None)
    return f"{pid}.{_PD_MODEL_SUFFIX}" if pid else None


def _owui() -> Any:
    from types import SimpleNamespace

    from open_webui.models.access_grants import AccessGrants
    from open_webui.models.models import Models
    from open_webui.models.users import Users
    from open_webui.utils.access_control import check_model_access
    from open_webui.utils.auth import get_verified_user
    from open_webui.config import BYPASS_ADMIN_ACCESS_CONTROL
    from open_webui.env import BYPASS_MODEL_ACCESS_CONTROL

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
    except Exception:
        return None


def resolve_socket_user_id(sid: str) -> str | None:
    try:
        from open_webui.socket.main import get_user_id_from_session_pool

        return get_user_id_from_session_pool(sid)
    except Exception:
        return None


async def _authorized(user: Any, pipe: Any, permission: str) -> bool:
    if user is None:
        return False
    mid = model_id(pipe)
    if not mid:
        return False
    try:
        o = _owui()
        o.get_verified_user(user)
        model = await o.Models.get_model_by_id(mid)
        if permission == "read":
            await o.check_model_access(user, model, bypass_filter=o.BYPASS_MODEL)
            return True
        if model is None:
            return False
        if user.role == "admin" and o.BYPASS_ADMIN:
            return True
        if user.id == model.user_id:
            return True
        return await o.AccessGrants.has_access(
            user_id=user.id, resource_type="model", resource_id=mid, permission="write",
        )
    except Exception:
        return False


async def can_view(user: Any, pipe: Any) -> bool:
    return await _authorized(user, pipe, "read")


async def can_act(user: Any, pipe: Any) -> bool:
    return await _authorized(user, pipe, "write")
