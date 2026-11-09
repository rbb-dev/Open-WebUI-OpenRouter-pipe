from __future__ import annotations

import asyncio
import importlib
import json
import logging
import secrets
import time
from typing import Any, TypeGuard

from ...core.warn_latch import warn_level
from .authz import resolve_user

logger = logging.getLogger(__name__)

_CLUSTER_MODNAME = "open_webui_openrouter_pipe.plugins.pipe_dashboard.cluster"
_CLUSTER_SOCKET_MODNAME = "open_webui_openrouter_pipe.plugins.pipe_dashboard.dashboard_socket"
_CLUSTER_ROUTES_MODNAME = "open_webui_openrouter_pipe.plugins.pipe_dashboard.http_routes"

RELAY_TOPIC = "pipe_dashboard"
REPLY_EVENT = "openrouter:pipe_dashboard:reply"

_PD_CLAIM_TTL_S = 120
_PD_LOCK_TTL_S = 10
_PD_INBOX_REFRESH_S = 10.0
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
_known_users: dict[str, set[str]] = {}
_refreshed_at: dict[str, float] = {}
_relay_tasks: set[asyncio.Task[Any]] = set()


def relay_channel(user_id: str, pipe_id: str) -> str | None:
    if not user_id or not pipe_id or ":" in user_id or ":" in pipe_id:
        return None
    return f"{user_id}:{RELAY_TOPIC}:{pipe_id}"


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
    return ":".join((_key_prefix(), RELAY_TOPIC, pipe_id, *parts))


def _text(value: Any) -> str:
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else str(value)


def _valid_token(value: Any) -> TypeGuard[str]:
    return (
        isinstance(value, str)
        and 0 < len(value) <= _PD_MAX_TOKEN_LEN
        and all(ch.isalnum() or ch in "-_" for ch in value)
    )


class _RelayInbox:
    def __init__(self, user_id: str, pipe_id: str) -> None:
        self.user_id = user_id
        self.pipe_id = pipe_id

    async def put(self, data: Any) -> None:
        try:
            live = importlib.import_module(_CLUSTER_MODNAME)
            live.schedule_relay_message(self.user_id, self.pipe_id, data)
        except Exception:
            logger.warning(
                "pipe_dashboard: a dashboard request relayed by Open WebUI could not be started",
                exc_info=True,
            )


def _event_queues() -> dict[str, Any] | None:
    try:
        from open_webui.socket.main import EVENT_QUEUES
    except ImportError:
        return None
    except Exception:
        logger.log(
            warn_level(_warned_sites, "event_queues"),
            "pipe_dashboard: Open WebUI's relay between workers is unavailable; the dashboard "
            "answers only viewers whose socket is on a worker running the pipe",
            exc_info=True,
        )
        return None
    return EVENT_QUEUES


def register_inboxes(pipe_id: str, user_ids: Any) -> int:
    queues = _event_queues()
    if queues is None:
        return 0
    added = 0
    for user_id in user_ids:
        uid = str(user_id or "")
        channel = relay_channel(uid, pipe_id)
        if channel is None or channel in queues:
            continue
        queues[channel] = _RelayInbox(uid, pipe_id)
        added += 1
    return added


def unregister_inboxes(pipe_id: str) -> None:
    _known_users.pop(pipe_id, None)
    _refreshed_at.pop(pipe_id, None)
    queues = _event_queues()
    if queues is None:
        return
    for channel, inbox in list(queues.items()):
        if type(inbox).__name__ == "_RelayInbox" and getattr(inbox, "pipe_id", None) == pipe_id:
            queues.pop(channel, None)


async def remember_dashboard_user(pipe_id: str, user_id: str) -> None:
    if relay_channel(user_id, pipe_id) is None:
        return
    _known_users.setdefault(pipe_id, set()).add(user_id)
    register_inboxes(pipe_id, [user_id])
    redis = owui_redis()
    if redis is None:
        return
    try:
        await redis.hset(_redis_key(pipe_id, "users"), user_id, str(int(time.time())))
    except Exception:
        logger.log(
            warn_level(_warned_sites, "users_write"),
            "pipe_dashboard: the dashboard user could not be shared with Open WebUI's other "
            "workers through Redis; only this worker answers that user's dashboard",
            exc_info=True,
        )


def _local_session_user_ids() -> set[str]:
    try:
        from open_webui.socket.main import SESSION_POOL
    except ImportError:
        return set()
    except Exception:
        logger.debug("pipe_dashboard: Open WebUI's session pool could not be read", exc_info=True)
        return set()
    try:
        sessions = list(SESSION_POOL.values())
    except Exception:
        logger.debug("pipe_dashboard: Open WebUI's session pool could not be listed", exc_info=True)
        return set()
    return {str(user["id"]) for user in sessions if isinstance(user, dict) and user.get("id")}


async def refresh_inboxes(pipe_id: str) -> None:
    if not pipe_id:
        return
    user_ids = set(_known_users.get(pipe_id, ()))
    redis = owui_redis()
    if redis is None:
        user_ids.update(_local_session_user_ids())
    else:
        try:
            user_ids.update(_text(uid) for uid in await redis.hkeys(_redis_key(pipe_id, "users")) or [])
        except Exception:
            logger.log(
                warn_level(_warned_sites, "users_read"),
                "pipe_dashboard: the dashboard users could not be read from Open WebUI's Redis; "
                "viewers whose socket is on another worker may get no answer",
                exc_info=True,
            )
    register_inboxes(pipe_id, user_ids)
    _refreshed_at[pipe_id] = time.monotonic()


async def maybe_refresh_inboxes(pipe_id: str) -> None:
    last = _refreshed_at.get(pipe_id)
    if last is not None and time.monotonic() - last < _PD_INBOX_REFRESH_S:
        return
    await refresh_inboxes(pipe_id)


def schedule_relay_message(user_id: str, pipe_id: str, data: Any) -> None:
    task = asyncio.get_running_loop().create_task(handle_relay_message(user_id, pipe_id, data))
    _relay_tasks.add(task)
    task.add_done_callback(_relay_tasks.discard)


def _session_user_id(sid: str) -> str | None:
    try:
        from open_webui.socket.main import get_user_id_from_session_pool

        return get_user_id_from_session_pool(sid)
    except Exception:
        logger.debug("pipe_dashboard: the viewer's socket could not be looked up", exc_info=True)
        return None


async def _claim(pipe_id: str, rid: str) -> bool:
    redis = owui_redis()
    if redis is None:
        return True
    try:
        return bool(
            await redis.set(_redis_key(pipe_id, "request", rid), "1", nx=True, ex=_PD_CLAIM_TTL_S)
        )
    except Exception:
        logger.log(
            warn_level(_warned_sites, "claim"),
            "pipe_dashboard: a dashboard request could not be claimed in Open WebUI's Redis; "
            "it is left for the dashboard to send again",
            exc_info=True,
        )
        return False


async def _reply(sid: str, payload: dict[str, Any]) -> None:
    try:
        from open_webui.socket.main import sio

        await sio.emit(REPLY_EVENT, payload, to=sid)
    except Exception:
        logger.debug("pipe_dashboard: an answer could not be sent to the viewer's socket", exc_info=True)


async def handle_relay_message(user_id: str, pipe_id: str, data: Any) -> None:
    if not isinstance(data, dict):
        return
    op, rid, sid = data.get("op"), data.get("rid"), data.get("sid")
    if op not in ("sub", "act") or not _valid_token(rid) or not _valid_token(sid):
        return
    if _session_user_id(sid) != user_id:
        logger.debug("pipe_dashboard: a relayed request named a socket that is not its sender's")
        return
    if not await _claim(pipe_id, rid):
        return
    try:
        if op == "sub":
            socket_mod = importlib.import_module(_CLUSTER_SOCKET_MODNAME)
            admitted = await socket_mod.admit_viewer(sid, pipe_id, user_id)
            await _reply(sid, {"rid": rid, "ok": admitted})
        else:
            await _relay_action(user_id, pipe_id, sid, rid, data)
    except Exception:
        logger.warning("pipe_dashboard: a dashboard request relayed by Open WebUI failed", exc_info=True)
        await _reply(sid, {"rid": rid, "status": 500, "body": {"error": "internal error"}})


async def _relay_action(user_id: str, pipe_id: str, sid: str, rid: str, data: dict[str, Any]) -> None:
    action = data.get("action")
    args = data.get("args") if data.get("args") is not None else {}
    routes = importlib.import_module(_CLUSTER_ROUTES_MODNAME)
    if not isinstance(action, str) or not action or not isinstance(args, dict):
        await _reply(sid, {"rid": rid, "status": 400, "body": {"error": "bad request"}})
        return
    if routes._exceeds_json_depth(json.dumps(args).encode("utf-8")):
        await _reply(sid, {"rid": rid, "status": 413, "body": {"error": "args too large"}})
        return
    pipe = await ensure_dashboard_pipe(pipe_id)
    if pipe is None:
        await _reply(sid, {"rid": rid, "status": 404, "body": {"error": "unknown pipe"}})
        return
    user = await resolve_user(user_id)
    if not getattr(user, "id", None):
        await _reply(sid, {"rid": rid, "status": 503, "body": {"error": "user unavailable"}})
        return
    status, body = await routes.run_action(
        user, pipe, action, args, request=routes.internal_request(), client_ip=None
    )
    await _reply(sid, {"rid": rid, "status": status, "body": body})


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
