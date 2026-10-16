from __future__ import annotations

from contextlib import suppress
from typing import Any, NamedTuple
from urllib.parse import urlparse

try:
    from redis.asyncio.client import Redis
    from redis.asyncio.cluster import RedisCluster
    from redis.asyncio.sentinel import Sentinel
except ImportError:
    Redis = None  # type: ignore
    RedisCluster = None  # type: ignore
    Sentinel = None  # type: ignore

try:
    from open_webui.utils import redis as _owui_redis
except Exception:  # noqa: BLE001 - a half-initialised open_webui package raises beyond ImportError
    _owui_redis = None

_ACCEPTED_SCHEMES = frozenset({"redis", "rediss"})

_REDISS_SENTINEL_NOTICE = (
    "REDIS_URL uses the rediss scheme together with Redis Sentinel hosts; "
    "Open WebUI's Sentinel setup keeps no TLS flag, so the Redis credentials would "
    "cross the sentinel and master links in plain text; use redis:// for the sentinel "
    "pair, or drop the sentinel hosts to keep TLS point-to-point"
)


class RedisTarget(NamedTuple):
    mode: str
    url: str
    service_name: str = ""
    nodes: tuple[tuple[str, int], ...] = ()
    db: int | None = None
    username: str | None = None
    password: str | None = None
    sentinel_kwargs: dict[str, Any] | None = None


class RedisUnavailable(RuntimeError):
    pass


class SentinelRefused(RuntimeError):
    pass


def _replica_parse_redis_url(url: str) -> dict[str, Any]:
    parts = urlparse(url)
    if parts.scheme not in _ACCEPTED_SCHEMES:
        raise ValueError(f"Invalid Redis URL scheme '{parts.scheme}'; expected 'redis' or 'rediss'.")
    return {
        "service": parts.netloc.rpartition("@")[2].split(":")[0] or "mymaster",
        "port": parts.port or 6379,
        "db": int(parts.path.lstrip("/") or "0"),
        "username": parts.username or None,
        "password": parts.password or None,
    }


def _replica_get_sentinels_from_env(
    hosts_csv: str | None, port: str | int | None
) -> list[tuple[str, int]]:
    if not hosts_csv:
        return []
    resolved_port = int(port) if port else 26379
    return [(host.strip(), resolved_port) for host in hosts_csv.split(",") if host.strip()]


def _replica_build_sentinel_kwargs(base_url: str) -> dict[str, Any]:
    cfg = _replica_parse_redis_url(base_url)
    return {"username": cfg["username"], "password": cfg["password"]}


def _parse_redis_url(url: str) -> dict[str, Any]:
    if _owui_redis is not None:
        return _owui_redis.parse_redis_url(url)
    return _replica_parse_redis_url(url)


def _sentinel_nodes(hosts_csv: str | None, port: str | int | None) -> list[tuple[str, int]]:
    if _owui_redis is not None:
        return _owui_redis.get_sentinels_from_env(hosts_csv, port)
    return _replica_get_sentinels_from_env(hosts_csv, port)


def _sentinel_kwargs(url: str) -> dict[str, Any] | None:
    if _owui_redis is not None:
        builder = getattr(_owui_redis, "build_sentinel_kwargs", None)
        if builder is not None:
            return builder(url, {})
        return None
    return _replica_build_sentinel_kwargs(url)


def parse_redis_target(
    redis_url: str,
    sentinel_hosts: str | None,
    sentinel_port: str | int | None,
    cluster: bool,
) -> RedisTarget:
    nodes = _sentinel_nodes(sentinel_hosts, sentinel_port)
    if nodes:
        if urlparse(redis_url).scheme.lower() == "rediss":
            raise SentinelRefused(_REDISS_SENTINEL_NOTICE)
        cfg = _parse_redis_url(redis_url)
        return RedisTarget(
            mode="sentinel",
            url=redis_url,
            service_name=cfg["service"],
            nodes=tuple(nodes),
            db=cfg["db"],
            username=cfg["username"],
            password=cfg["password"],
            sentinel_kwargs=_sentinel_kwargs(redis_url),
        )
    if cluster:
        return RedisTarget(mode="cluster", url=redis_url)
    return RedisTarget(mode="standalone", url=redis_url)


def build_redis_client(target: RedisTarget, aioredis_mod: Any) -> Any:
    if target.mode == "standalone":
        if aioredis_mod is None:
            raise RedisUnavailable("redis-py is unavailable")
        return aioredis_mod.from_url(
            target.url,
            encoding="utf-8",
            decode_responses=True,
            health_check_interval=30,
        )
    if Redis is None or Sentinel is None or RedisCluster is None:
        raise RedisUnavailable("redis-py is unavailable")
    if target.mode == "cluster":
        return RedisCluster.from_url(
            target.url,
            decode_responses=True,
            encoding="utf-8",
            health_check_interval=30,
        )
    connection_kwargs: dict[str, Any] = {
        "encoding": "utf-8",
        "decode_responses": True,
        "health_check_interval": 30,
        "db": target.db,
        "username": target.username,
        "password": target.password,
    }
    sentinel = Sentinel(target.nodes, sentinel_kwargs=target.sentinel_kwargs, **connection_kwargs)

    class _SentinelRootClosingRedis(Redis):
        async def aclose(self, close_connection_pool: bool | None = None) -> None:
            try:
                await super().aclose(close_connection_pool=close_connection_pool)
            finally:
                manager = getattr(self.connection_pool, "sentinel_manager", None)
                if manager is not None:
                    for node in list(manager.sentinels):
                        with suppress(Exception):
                            await node.aclose()

    return sentinel.master_for(target.service_name, redis_class=_SentinelRootClosingRedis)


def mget_many(client: Any, keys: Any) -> Any:
    if client is None:
        return None
    if RedisCluster is not None and isinstance(client, RedisCluster):
        return client.mget_nonatomic(list(keys))
    return client.mget(list(keys))
