from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_LOCK_TIMEOUT_S = 60


def owui_redis_lock(name: str, ttl: int) -> Any | None:
    try:
        from open_webui.env import (
            REDIS_KEY_PREFIX,
            WEBSOCKET_MANAGER,
            WEBSOCKET_REDIS_CLUSTER,
            WEBSOCKET_REDIS_URL,
            WEBSOCKET_SENTINEL_HOSTS,
            WEBSOCKET_SENTINEL_PORT,
        )
        from open_webui.socket.utils import RedisLock
        from open_webui.utils.redis import get_sentinels_from_env

        if str(WEBSOCKET_MANAGER or "").lower() != "redis" or not WEBSOCKET_REDIS_URL:
            return None
        return RedisLock(
            redis_url=WEBSOCKET_REDIS_URL,
            lock_name=f"{REDIS_KEY_PREFIX}:{name}",
            timeout_secs=ttl,
            redis_sentinels=get_sentinels_from_env(
                WEBSOCKET_SENTINEL_HOSTS, WEBSOCKET_SENTINEL_PORT
            ),
            redis_cluster=bool(WEBSOCKET_REDIS_CLUSTER),
        )
    except Exception:
        logger.warning("The cross-worker lock %s is unavailable", name, exc_info=True)
        return None


def lock_acquire(lock: Any) -> Any:
    return getattr(lock, "acquire_lock", None) or getattr(lock, "aquire_lock", None)


def owui_lock_timeout() -> int:
    try:
        from open_webui.env import WEBSOCKET_REDIS_LOCK_TIMEOUT

        return int(WEBSOCKET_REDIS_LOCK_TIMEOUT)
    except (ImportError, TypeError, ValueError):
        return _DEFAULT_LOCK_TIMEOUT_S
