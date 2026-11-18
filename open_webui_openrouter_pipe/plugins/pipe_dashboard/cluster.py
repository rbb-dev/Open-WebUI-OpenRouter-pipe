from __future__ import annotations

import importlib
import json
import logging
import secrets
from typing import Any, TypeGuard

from ...core.warn_latch import warn_level
from .authz import resolve_user

logger = logging.getLogger(__name__)

_CLUSTER_MODNAME = "open_webui_openrouter_pipe.plugins.pipe_dashboard.cluster"
_CLUSTER_SOCKET_MODNAME = "open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_socket"
_CLUSTER_ROUTES_MODNAME = "open_webui_openrouter_pipe.plugins.pipe_dashboard.http_routes"

_PD_KEY_TOPIC = "pipe_dashboard"
PANEL_REQUEST_KEY = "pipe_dashboard_panel"

_PD_LOCK_TTL_S = 10
_PD_MAX_TOKEN_LEN = 64
_PD_DEFAULT_KEY_PREFIX = "open-webui"

LOCK_RENEW = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('expire', KEYS[1], ARGV[2]) end return 0"
)
LOCK_RELEASE = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then "
    "return redis.call('del', KEYS[1]) end return 0"
)

_warned_sites: set[str] = set()


def owui_redis() -> Any:
    try:
        from open_webui.socket.main import REDIS
    except ImportError:
        return None
    except Exception:
        logger.debug("pipe_dashboard: Open WebUI's socket Redis could not be read", exc_info=True)
        return None
    return REDIS


def _key_prefix() -> str:
    try:
        from open_webui.env import REDIS_KEY_PREFIX
    except ImportError:
        return _PD_DEFAULT_KEY_PREFIX
    return str(REDIS_KEY_PREFIX)


def _redis_key(pipe_id: str, *parts: str) -> str:
    return ":".join((_key_prefix(), _PD_KEY_TOPIC, pipe_id, *parts))


def _text(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _valid_token(value: Any) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _PD_MAX_TOKEN_LEN
        and all(ch.isalnum() or ch in "-_" for ch in value)
    )


def _session_user_id(sid: str) -> str | None:
    try:
        from open_webui.socket.main import get_user_id_from_session_pool

        return get_user_id_from_session_pool(sid)
    except Exception:
        logger.debug("pipe_dashboard: the viewer's socket could not be looked up", exc_info=True)
        return None


def panel_request_of(text: Any) -> dict[str, Any] | None:
    if not isinstance(text, str) or PANEL_REQUEST_KEY not in text:
        return None
    routes = importlib.import_module(_CLUSTER_ROUTES_MODNAME)
    if routes._exceeds_json_depth(text.encode("utf-8")):
        return None
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    if not isinstance(parsed, dict) or set(parsed) != {PANEL_REQUEST_KEY}:
        return None
    request = parsed[PANEL_REQUEST_KEY]
    if not isinstance(request, dict) or request.get("op") not in ("sub", "act"):
        return None
    return request


async def _sent_with_bearer_header(http_request: Any) -> bool:
    if http_request is None:
        return False
    try:
        from open_webui.utils.auth import bearer_security
    except ImportError:
        return False
    return await bearer_security(http_request) is not None


async def answer_panel_request(
    pipe: Any, user_id: str, request: dict[str, Any], http_request: Any = None
) -> dict[str, Any]:
    pipe_id = str(getattr(pipe, "id", "") or "")
    if not await _sent_with_bearer_header(http_request):
        logger.warning(
            "pipe_dashboard: a panel request was refused because it did not carry the Authorization header the panel sends (user=%s)",
            user_id or "?",
        )
        if request.get("op") == "sub":
            return {"ok": False}
        return {"status": 401, "body": {"error": "authorization header required"}}
    sid = request.get("sid")
    if not user_id or not _valid_token(sid) or _session_user_id(sid) != user_id:
        logger.warning(
            "pipe_dashboard: a panel request was refused because it did not name a live socket of its sender (user=%s)",
            user_id or "?",
        )
        if request.get("op") == "sub":
            return {"ok": False}
        return {"status": 403, "body": {"error": "not from a live dashboard panel"}}
    if request.get("op") == "sub":
        socket_mod = importlib.import_module(_CLUSTER_SOCKET_MODNAME)
        return {"ok": bool(await socket_mod.admit_viewer(sid, pipe_id, user_id))}
    action = request.get("action")
    args = request.get("args") if request.get("args") is not None else {}
    routes = importlib.import_module(_CLUSTER_ROUTES_MODNAME)
    if not isinstance(action, str) or not action or not isinstance(args, dict):
        return {"status": 400, "body": {"error": "bad request"}}
    if routes._exceeds_json_depth(json.dumps(args).encode("utf-8")):
        return {"status": 413, "body": {"error": "args too large"}}
    user = await resolve_user(user_id)
    if not getattr(user, "id", None):
        return {"status": 503, "body": {"error": "user unavailable"}}
    status, body = await routes.run_action(
        user,
        pipe,
        action,
        args,
        request=http_request if http_request is not None else routes.internal_request(),
        client_ip=routes._client_ip(http_request) if http_request is not None else None,
    )
    return {"status": status, "body": body}


async def ensure_dashboard_pipe(pipe_id: str, request: Any = None) -> Any:
    if not pipe_id:
        return None
    pipe = importlib.import_module(_CLUSTER_SOCKET_MODNAME)._current_pipe(pipe_id)
    if pipe is not None:
        return pipe
    if request is None:
        request = importlib.import_module(_CLUSTER_ROUTES_MODNAME).internal_request()
    if request is None:
        return None
    try:
        from open_webui.functions import get_function_module_by_id

        loaded = await get_function_module_by_id(request, pipe_id)
    except Exception:
        logger.log(
            warn_level(_warned_sites, "start_load"),
            "pipe_dashboard: the pipe could not be loaded to start its dashboard on this worker",
            exc_info=True,
        )
        return None
    if not bool(getattr(getattr(loaded, "valves", None), "ENABLE_PLUGIN_SYSTEM", False)):
        return None
    start = getattr(loaded, "_ensure_plugin_registry", None)
    if not callable(start):
        return None
    try:
        start()
    except Exception:
        logger.warning("pipe_dashboard: the dashboard could not be started on this worker", exc_info=True)
        return None
    return importlib.import_module(_CLUSTER_SOCKET_MODNAME)._current_pipe(pipe_id)


async def record_viewer(pipe_id: str, sid: str, user_id: str) -> None:
    redis = owui_redis()
    if redis is None:
        return
    try:
        await redis.hset(_redis_key(pipe_id, "viewers"), sid, user_id)
    except Exception:
        logger.log(
            warn_level(_warned_sites, "viewer_write"),
            "pipe_dashboard: a viewer could not be shared with Open WebUI's other workers through "
            "Redis; it gets live updates only while its socket is on this worker",
            exc_info=True,
        )


async def forget_viewer(pipe_id: str, sid: str) -> None:
    redis = owui_redis()
    if redis is None:
        return
    try:
        await redis.hdel(_redis_key(pipe_id, "viewers"), sid)
    except Exception:
        logger.debug("pipe_dashboard: a viewer could not be removed from Open WebUI's Redis", exc_info=True)


async def shared_viewers(pipe_id: str) -> dict[str, str] | None:
    redis = owui_redis()
    if redis is None:
        return None
    try:
        stored = await redis.hgetall(_redis_key(pipe_id, "viewers")) or {}
    except Exception:
        logger.log(
            warn_level(_warned_sites, "viewer_read"),
            "pipe_dashboard: the viewers could not be read from Open WebUI's Redis; no live "
            "updates are sent until it answers",
            exc_info=True,
        )
        return {}
    return {_text(sid): _text(uid) for sid, uid in stored.items()}


class PublisherLock:
    def __init__(self, pipe_id: str) -> None:
        self.pipe_id = pipe_id
        self.key = _redis_key(pipe_id, "publisher")
        self.token = secrets.token_hex(16)
        self.held = False

    async def hold(self, redis: Any) -> bool:
        try:
            if self.held and await redis.eval(LOCK_RENEW, 1, self.key, self.token, _PD_LOCK_TTL_S):
                return True
            self.held = bool(await redis.set(self.key, self.token, nx=True, ex=_PD_LOCK_TTL_S))
        except Exception:
            logger.debug("pipe_dashboard: the live-update lock could not be checked", exc_info=True)
            self.held = False
        return self.held

    async def release(self, redis: Any) -> None:
        if not self.held or redis is None:
            return
        self.held = False
        try:
            await redis.eval(LOCK_RELEASE, 1, self.key, self.token)
        except Exception:
            logger.debug(
                "pipe_dashboard: the live-update lock could not be released; it expires on its own",
                exc_info=True,
            )
