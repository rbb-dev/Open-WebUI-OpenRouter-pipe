from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
from typing import Any

logger = logging.getLogger(__name__)

RETIRE_ACTION = "retire"
RETIRE_UPDATED_EVENT = "function.updated"
RETIRE_DELETED_EVENT = "function.deleted"
_RETIRE_EVENTS = frozenset({RETIRE_UPDATED_EVENT, RETIRE_DELETED_EVENT})
_RETIRE_DEFAULT_KEY_PREFIX = "open-webui"
_RETIRE_RECONNECT_START_S = 1.0
_RETIRE_RECONNECT_MAX_S = 30.0


def _retire_key_prefix() -> str:
    try:
        from open_webui.env import REDIS_KEY_PREFIX
    except ImportError:
        return _RETIRE_DEFAULT_KEY_PREFIX
    return str(REDIS_KEY_PREFIX or _RETIRE_DEFAULT_KEY_PREFIX)


def retire_channel(function_id: str) -> str:
    return f"{_retire_key_prefix()}:openrouter_pipe:{function_id}:retire"


def retire_listener_app() -> Any:
    return getattr(sys.modules.get("open_webui.main"), "app", None)


def retire_listener_redis(app: Any) -> Any:
    return getattr(getattr(app, "state", None), "redis", None)


async def publish_retire(redis: Any, function_id: str, event: str) -> None:
    payload = json.dumps({"action": RETIRE_ACTION, "function_id": function_id, "event": event})
    channel = retire_channel(function_id)
    if hasattr(redis, "nodes_manager"):
        await redis.execute_command("PUBLISH", channel, payload)
    else:
        await redis.publish(channel, payload)


class RetireEventSink:
    async def handle_event(self, request: Any, event: Any) -> None:
        name = getattr(event, "event", None)
        if name not in _RETIRE_EVENTS:
            return
        subject = getattr(event, "subject", None)
        if not isinstance(subject, dict) or not subject.get("id"):
            return
        redis = retire_listener_redis(getattr(request, "app", None))
        if redis is None:
            return
        await publish_retire(redis, str(subject["id"]), str(name))


def register_retire_sink() -> bool:
    try:
        from open_webui.events import EVENT_SINKS
    except ImportError:
        return False
    EVENT_SINKS[:] = [sink for sink in EVENT_SINKS if type(sink).__name__ != RetireEventSink.__name__]
    EVENT_SINKS.append(RetireEventSink())
    return True


async def _retire_stored_row(function_id: str) -> Any:
    from open_webui.models.functions import Functions

    return await Functions.get_function_by_id(function_id)


async def should_retire(pipe: Any, app: Any, event: Any) -> bool:
    function_id = str(getattr(pipe, "id", "") or "")
    if not function_id:
        return False
    row = await _retire_stored_row(function_id)
    if event == RETIRE_DELETED_EVENT:
        return row is None
    if event != RETIRE_UPDATED_EVENT or row is None:
        return False
    state = getattr(app, "state", None)
    if (getattr(state, "FUNCTIONS", None) or {}).get(function_id) is not pipe:
        return False
    cached = (getattr(state, "FUNCTION_CONTENTS", None) or {}).get(function_id)
    return cached != getattr(row, "content", None)


async def _retired_by(pipe: Any, app: Any, data: Any) -> bool:
    try:
        command = json.loads(data)
    except (TypeError, ValueError):
        return False
    if not isinstance(command, dict) or command.get("action") != RETIRE_ACTION:
        return False
    if str(command.get("function_id")) != str(getattr(pipe, "id", "")):
        return False
    event = command.get("event")
    if not await should_retire(pipe, app, event):
        return False
    logger.info(
        "hot-reload: retiring this copy; the stored pipe was %s on another worker (id=%s)",
        "deleted" if event == RETIRE_DELETED_EVENT else "updated",
        getattr(pipe, "id", "?"),
    )
    pipe.close_when_idle()
    return True


async def run_retire_listener(pipe: Any, app: Any) -> None:
    redis = retire_listener_redis(app)
    if redis is None:
        return
    channel = retire_channel(str(getattr(pipe, "id", "")))
    reconnect = _RETIRE_RECONNECT_START_S
    while True:
        pubsub = None
        try:
            await redis.initialize()
            pubsub = redis.pubsub()
            await pubsub.subscribe(channel)
            reconnect = _RETIRE_RECONNECT_START_S
            async for message in pubsub.listen():
                if not isinstance(message, dict) or message.get("type") != "message":
                    continue
                if await _retired_by(pipe, app, message.get("data")):
                    return
            logger.warning("hot-reload: the retire listener stopped; retrying")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("hot-reload: the retire listener failed; retrying", exc_info=True)
        finally:
            if pubsub is not None:
                with contextlib.suppress(Exception):
                    await pubsub.aclose()
        await asyncio.sleep(reconnect)
        reconnect = min(reconnect * 2, _RETIRE_RECONNECT_MAX_S)
