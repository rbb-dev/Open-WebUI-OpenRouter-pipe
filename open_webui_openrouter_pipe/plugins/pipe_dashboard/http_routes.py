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
from typing import Any, cast

from fastapi import Depends, Request
from pydantic import BaseModel

from .actions import ACTIONS, _audit, _redacted_args
from .authz import can_view

logger = logging.getLogger(__name__)

_ACTIONS_MODNAME = "open_webui_openrouter_pipe.plugins.pipe_dashboard.actions"
_ROUTES_MODNAME = "open_webui_openrouter_pipe.plugins.pipe_dashboard.http_routes"

_ACTION_PATH = "/api/pipe/dashboard/action"
_registered_paths: set[str] = set()
_registration_lock = threading.Lock()

_PD_COARSE_MIN_INTERVAL = 0.25
_coarse_state: dict[str, float] = {}

_off_audit_state: dict[str, float] = {}
_PD_OFF_AUDIT_EVERY_S = 300.0

_PD_RECONCILE_BACKOFF_S = 5.0

_routes_get_pipe: Any = None
_reconcile_lock: asyncio.Lock | None = None
_fresh_dispatch: Any = None
_reconcile_retry_until: float = 0.0
_teardown_epoch: int = 0


def _current_reconcile_lock() -> asyncio.Lock:
    global _reconcile_lock
    current = asyncio.get_running_loop()
    lock = _reconcile_lock
    if lock is not None:
        try:
            lock_loop = getattr(cast(Any, lock), "_get_loop", lambda: None)()
            if lock_loop is not current:
                lock = None
        except RuntimeError:
            lock = None
    if lock is None:
        lock = asyncio.Lock()
        _reconcile_lock = lock
    return lock


def _live_reconcile_state() -> Any:
    try:
        mod = importlib.import_module(_ROUTES_MODNAME)
    except Exception:
        logger.warning("pipe_dashboard action-route reconcile-state lookup failed", exc_info=True)
        return None
    return mod


def set_pipe_getter(get_pipe: Any) -> None:
    state = _live_reconcile_state()
    if state is None:
        return
    if state._routes_get_pipe is get_pipe:
        return
    state._routes_get_pipe = get_pipe
    state._teardown_epoch += 1
    state._fresh_dispatch = None


def clear_routes_pipe_getter(instance: Any, name: str) -> None:
    global _routes_get_pipe
    current = _routes_get_pipe
    if current is None or current == getattr(instance, name, None):
        _routes_get_pipe = None


async def _plugins_enabled(pipe: Any) -> bool:
    from .config_service import stored_gate_valves

    merged, read_ok = await stored_gate_valves(
        getattr(pipe, "id", ""), getattr(pipe, "valves", None)
    )
    if not read_ok:
        return False
    return bool(merged.get("ENABLE_PLUGIN_SYSTEM", False))


def _audit_off(user: Any, action: str, client_ip: Any) -> None:
    from ...core.warn_latch import warn_level
    from .actions import _scrub
    from .actions import logger as _actions_logger

    uid = str(getattr(user, "id", None) or "-")
    _actions_logger.log(
        warn_level(_off_audit_state, f"{uid}|plugin_system_off", cooldown_s=_PD_OFF_AUDIT_EVERY_S),
        "pipe_dashboard action user=%s action=%s outcome=plugin_system_off ip=%s args=-",
        _scrub(uid), _scrub(action), _scrub(client_ip),
    )


async def _dispatch_unavailable(
    pipe: Any, user: Any, name: str, args: Any, *, client_ip: Any = None, request: Any = None
) -> tuple[int, dict[str, Any]]:
    _audit(user, name, "unavailable", client_ip, _redacted_args(pipe, args))
    return 503, {"error": "action unavailable"}


def clear_fresh_dispatch(pipe: Any) -> None:
    state = _live_reconcile_state()
    if state is None:
        return
    state._teardown_epoch += 1
    cached = state._fresh_dispatch
    if cached is not None and cached[1] is pipe:
        state._fresh_dispatch = None


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


_MAX_JSON_DEPTH = 64


def _exceeds_json_depth(raw: bytes, limit: int = _MAX_JSON_DEPTH) -> bool:
    depth = 0
    in_string = False
    escaped = False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte == 0x5B or byte == 0x7B:
            depth += 1
            if depth > limit:
                return True
        elif byte == 0x5D or byte == 0x7D:
            depth -= 1
    return False


async def _bounded_json_body(request: Request) -> None:
    from fastapi import HTTPException

    raw = await request.body()
    if _exceeds_json_depth(raw):
        raise HTTPException(status_code=400, detail="args nested too deeply")


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
        logger.warning(
            "pipe_dashboard: Open WebUI auth helpers are unavailable; denying the request",
            exc_info=True,
        )
        raise HTTPException(status_code=401)
    try:
        data = decode_token(token)
    except Exception:
        logger.debug("pipe_dashboard: bearer token rejected", exc_info=True)
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
        mod = importlib.import_module(_ACTIONS_MODNAME)
        dispatch = getattr(mod, "dispatch_action", None)
        if dispatch is None:
            return None
        return dispatch, fresh_pipe
    except Exception:
        logger.warning("pipe_dashboard action-route reconcile failed", exc_info=True)
        return None


def _live_actions() -> Any:
    try:
        mod = importlib.import_module(_ACTIONS_MODNAME)
    except Exception:
        logger.warning("pipe_dashboard action-route registry lookup failed", exc_info=True)
        return None
    return getattr(mod, "ACTIONS", None)


def _live_dispatch() -> Any:
    try:
        mod = importlib.import_module(_ACTIONS_MODNAME)
    except Exception:
        logger.warning("pipe_dashboard action-route dispatch lookup failed", exc_info=True)
        return None
    return getattr(mod, "dispatch_action", None)


def _live_routes_get_pipe() -> Any:
    try:
        mod = importlib.import_module(_ROUTES_MODNAME)
    except Exception:
        logger.warning("pipe_dashboard action-route pipe lookup failed", exc_info=True)
        return None
    getter = getattr(mod, "_routes_get_pipe", None)
    if getter is None:
        return None
    return getter()


def _preferred_dispatch(action: str) -> Any:
    live = _live_dispatch()
    if live is not None and action in (_live_actions() or {}):
        return live
    state = _live_reconcile_state()
    if state is not None:
        cached = state._fresh_dispatch
        if cached is not None and cached[1] is _live_routes_get_pipe():
            return cached[0]
    if live is not None:
        return live
    return _dispatch_unavailable


async def _current_dispatch(request: Any, user: Any, pipe: Any, fid: Any, action: str = "") -> tuple[Any, Any]:
    state = _live_reconcile_state()
    if state is not None and pipe is not None and fid and time.monotonic() >= state._reconcile_retry_until and await can_view(user, pipe):
        epoch = state._teardown_epoch
        async with _current_reconcile_lock():
            state = _live_reconcile_state()
            if state is not None and state._fresh_dispatch is None and epoch == state._teardown_epoch and time.monotonic() >= state._reconcile_retry_until:
                fresh = await _resolve_fresh(request, fid)
                state = _live_reconcile_state()
                if fresh is not None and state is not None and state._teardown_epoch == epoch:
                    state._fresh_dispatch = fresh
                    state._reconcile_retry_until = 0.0
                elif state is not None:
                    state._reconcile_retry_until = time.monotonic() + _PD_RECONCILE_BACKOFF_S
    return _preferred_dispatch(action), pipe


async def _action_route(
    request: Request,
    body: ActionBody,
    _depth: Any = Depends(_bounded_json_body),  # noqa: B008 - the guard must be declared here to run before the body is parsed
):
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse

    user = await bearer_user(request)
    pipe = _live_routes_get_pipe()
    if not await _plugins_enabled(pipe):
        _audit_off(user, body.action, _client_ip(request))
        return JSONResponse({"error": "plugin_system_off"}, status_code=404)
    if _coarse_rate_limited(user.id):
        _audit(user, body.action, "coarse_rate_limited", _client_ip(request))
        raise HTTPException(status_code=429)
    fid = getattr(pipe, "id", None) if pipe is not None else None
    if body.action not in ACTIONS and pipe is not None and fid:
        dispatch, pipe = await _current_dispatch(request, user, pipe, fid, body.action)
    else:
        dispatch, _p = await _current_dispatch(request, user, pipe, None, body.action)
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
        logger.debug("pipe_dashboard: Open WebUI app is not importable", exc_info=True)
        return None


def ensure_route_before_spa(app: Any) -> None:
    for i, route in enumerate(app.routes):
        if getattr(route, "name", "") == "spa-static-files":
            app.routes.append(app.routes.pop(i))
            break


def register_action_route() -> bool:
    with _registration_lock:
        app = get_owui_app()
        if app is None:
            return False
        try:
            for index, route in enumerate(list(getattr(app, "routes", []) or [])):
                if getattr(route, "path", None) != _ACTION_PATH:
                    continue
                endpoint = getattr(route, "endpoint", None)
                if getattr(endpoint, "__name__", None) != "_action_route" or (
                    "_live_routes_get_pipe" in getattr(endpoint, "__globals__", {})
                ):
                    _registered_paths.add(_ACTION_PATH)
                    ensure_route_before_spa(app)
                    return True
                app.add_api_route(_ACTION_PATH, _action_route, methods=["POST"])
                app.routes.insert(index, app.routes.pop())
                del app.routes[index + 1]
                ensure_route_before_spa(app)
                _registered_paths.add(_ACTION_PATH)
                return True
            app.add_api_route(_ACTION_PATH, _action_route, methods=["POST"])
            ensure_route_before_spa(app)
        except Exception:
            logger.debug("action route registration failed", exc_info=True)
            return False
        _registered_paths.add(_ACTION_PATH)
        return True
