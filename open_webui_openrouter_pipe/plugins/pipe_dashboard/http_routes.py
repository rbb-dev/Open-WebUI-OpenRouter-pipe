"""Authenticated HTTP action route for the pipe_dashboard dashboard.

Auth is header-only (Authorization: Bearer) so the route is CSRF-safe
regardless of OWUI's cookie SameSite/CORS; it reuses OWUI's own token
validation. Registered as a FastAPI APIRoute before the SPA catch-all.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import threading
import time
from typing import Any

from fastapi import Request
from pydantic import BaseModel

from .actions import ACTIONS, _audit, dispatch_action
from .authz import can_view

_pd_http_log = logging.getLogger(__name__)

_ACTION_PATH = "/api/pipe/dashboard/action"
_registered_paths: set[str] = set()
_registration_lock = threading.Lock()

_PD_COARSE_MIN_INTERVAL = 0.25
_coarse_state: dict[str, float] = {}

_get_pipe: Any = None
_reconcile_lock = asyncio.Lock()
_fresh_dispatch: Any = None
_reconcile_attempted = False


def set_pipe_getter(get_pipe: Any) -> None:
    global _get_pipe
    _get_pipe = get_pipe


def _coarse_rate_limited(user_id: str) -> bool:
    now = time.monotonic()
    last = _coarse_state.get(user_id, 0.0)
    if now - last < _PD_COARSE_MIN_INTERVAL:
        return True
    _coarse_state[user_id] = now
    return False


class ActionBody(BaseModel):
    action: str
    args: dict = {}


async def bearer_user(request: Request) -> Any:
    from fastapi import HTTPException

    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401)
    token = auth[len("Bearer "):]
    try:
        from open_webui.models.users import Users
        from open_webui.utils.auth import decode_token, is_valid_token
    except Exception:
        raise HTTPException(status_code=401)
    try:
        data = decode_token(token)
    except Exception:
        data = None
    if not data or not data.get("id"):
        raise HTTPException(status_code=401)
    if not await is_valid_token(data, getattr(request.app.state, "redis", None)):
        raise HTTPException(status_code=401)
    user = await Users.get_user_by_id(data["id"])
    if user is None or user.role not in ("user", "admin"):
        raise HTTPException(status_code=401)
    return user


def _client_ip(request: Any) -> Any:
    fwd = request.headers.get("x-forwarded-for", "")
    first = fwd.split(",")[0].strip()
    if first:
        return first
    return request.client.host if request.client else None


async def _resolve_fresh(request: Any, fid: str) -> tuple[Any, Any] | None:
    try:
        from open_webui.functions import get_function_module_by_id

        fresh_pipe = await get_function_module_by_id(request, fid)
        mod = importlib.import_module("open_webui_openrouter_pipe.plugins.pipe_dashboard.actions")
        dispatch = getattr(mod, "dispatch_action", None)
        if dispatch is None:
            return None
        return dispatch, fresh_pipe
    except Exception:
        _pd_http_log.warning("pipe_dashboard action-route reconcile failed", exc_info=True)
        return None


async def _action_route(request: Request, body: ActionBody):
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    user = await bearer_user(request)
    if _coarse_rate_limited(user.id):
        _audit(user, body.action, "coarse_rate_limited", _client_ip(request))
        raise HTTPException(status_code=429)
    pipe = _get_pipe() if _get_pipe else None
    dispatch = dispatch_action
    if body.action not in ACTIONS and pipe is not None and getattr(pipe, "id", None):
        global _fresh_dispatch, _reconcile_attempted
        if _fresh_dispatch is None and not _reconcile_attempted and await can_view(user, pipe):
            async with _reconcile_lock:
                if _fresh_dispatch is None and not _reconcile_attempted:
                    _reconcile_attempted = True
                    _fresh_dispatch = await _resolve_fresh(request, pipe.id)
        if _fresh_dispatch is not None:
            dispatch, fresh_pipe = _fresh_dispatch
            if fresh_pipe is not None:
                pipe = fresh_pipe
    kwargs: dict[str, Any] = {"client_ip": _client_ip(request)}
    try:
        params = inspect.signature(dispatch).parameters
        if "request" in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        ):
            kwargs["request"] = request
    except (TypeError, ValueError):
        kwargs["request"] = request
    status, payload = await dispatch(pipe, user, body.action, body.args, **kwargs)
    return JSONResponse(payload, status_code=status)


def get_owui_app() -> Any | None:
    try:
        from open_webui.main import app

        return app
    except Exception:
        return None


def ensure_route_before_spa(app: Any) -> None:
    for i, route in enumerate(app.routes):
        if getattr(route, "name", "") == "spa-static-files":
            app.routes.append(app.routes.pop(i))
            break


def register_action_route() -> bool:
    """Register the action route on the OWUI app, replacing any existing route at the path."""
    with _registration_lock:
        app = get_owui_app()
        if app is None:
            return False
        try:
            for route in list(getattr(app, "routes", []) or []):
                if getattr(route, "path", None) == _ACTION_PATH:
                    try:
                        app.routes.remove(route)
                    except ValueError:
                        pass
            app.add_api_route(_ACTION_PATH, _action_route, methods=["POST"])
            ensure_route_before_spa(app)
        except Exception:
            _pd_http_log.debug("action route registration failed", exc_info=True)
            return False
        _registered_paths.add(_ACTION_PATH)
        return True
