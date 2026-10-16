from __future__ import annotations

from types import SimpleNamespace

import pytest
from redis.asyncio.cluster import RedisCluster

from open_webui_openrouter_pipe.storage import redis_client


def test_the_sentinel_helper_route_uses_open_webuis_own_parse(monkeypatch) -> None:
    seen: dict = {}

    def parse_redis_url(url: str) -> dict:
        seen["url"] = url
        return {
            "service": "MyMaster",
            "port": 6380,
            "db": 2,
            "username": "cache-user",
            "password": "cache-pass",
        }

    def get_sentinels_from_env(hosts_csv: str, port) -> list:
        seen["hosts"] = (hosts_csv, port)
        return [("sentinel-a", 26379), ("sentinel-b", 2640)]

    def build_sentinel_kwargs(base_url: str, redis_options: dict) -> dict:
        seen["options"] = dict(redis_options)
        return {"username": "cache-user", "password": "cache-pass", **redis_options}

    stand_in = SimpleNamespace(
        parse_redis_url=parse_redis_url,
        get_sentinels_from_env=get_sentinels_from_env,
        build_sentinel_kwargs=build_sentinel_kwargs,
    )
    monkeypatch.setattr(redis_client, "_owui_redis", stand_in)

    target = redis_client.parse_redis_target(
        "redis://MyMaster:6380/2", "sentinel-a, sentinel-b", "2640", True
    )

    assert seen["url"] == "redis://MyMaster:6380/2"
    assert seen["hosts"] == ("sentinel-a, sentinel-b", "2640")
    assert seen["options"] == {}
    assert target.mode == "sentinel"
    assert target.service_name == "MyMaster"
    assert target.nodes == (("sentinel-a", 26379), ("sentinel-b", 2640))
    assert target.db == 2
    assert target.username == "cache-user"
    assert target.password == "cache-pass"
    assert target.sentinel_kwargs == {"username": "cache-user", "password": "cache-pass"}


def test_without_open_webui_the_replicate_parses_as_0122_does(monkeypatch) -> None:
    monkeypatch.setattr(redis_client, "_owui_redis", None)

    target = redis_client.parse_redis_target(
        "redis://cache-user:cache-pass@MyMaster:6380/2", "sentinel-a", "", True
    )

    assert target.mode == "sentinel"
    assert target.service_name == "MyMaster"
    assert target.db == 2
    assert target.username == "cache-user"
    assert target.password == "cache-pass"
    assert target.nodes == (("sentinel-a", 26379),)
    assert target.sentinel_kwargs == {"username": "cache-user", "password": "cache-pass"}


def test_standalone_targets_skip_parsing_entirely(monkeypatch) -> None:
    monkeypatch.setattr(redis_client, "_owui_redis", None)

    for url in ("unix:///run/redis.sock", "redis://localhost:6379/0"):
        target = redis_client.parse_redis_target(url, "", "26379", False)
        assert target.mode == "standalone"
        assert target.url == url
        assert target.service_name == ""
        assert target.nodes == ()


def test_the_cluster_builder_hands_the_library_the_pipe_kwargs(monkeypatch) -> None:
    recorded: dict = {}

    class FakeCluster:
        @classmethod
        def from_url(cls, url: str, **kwargs):
            recorded["url"] = url
            recorded["kwargs"] = dict(kwargs)
            return object()

    monkeypatch.setattr(redis_client, "RedisCluster", FakeCluster)

    target = redis_client.parse_redis_target("redis://localhost:7000/2", "", "", True)
    redis_client.build_redis_client(target, None)

    assert recorded["url"] == "redis://localhost:7000/2"
    assert recorded["kwargs"] == {
        "decode_responses": True,
        "encoding": "utf-8",
        "health_check_interval": 30,
    }


@pytest.mark.asyncio
async def test_multi_key_reads_dispatch_on_the_client_kind() -> None:
    class FakeCluster(RedisCluster):
        def __init__(self, *args, **kwargs) -> None:
            self.cluster_keys: list = []

        async def mget_nonatomic(self, keys, *args):
            self.cluster_keys.append(list(keys))
            return ["c1", "c2"]

    class PlainClient:
        def __init__(self) -> None:
            self.plain_keys: list = []

        async def mget(self, keys):
            self.plain_keys.append(list(keys))
            return ["p1", "p2"]

    cluster = FakeCluster()
    plain = PlainClient()

    assert await redis_client.mget_many(cluster, ["k1", "k2"]) == ["c1", "c2"]
    assert cluster.cluster_keys == [["k1", "k2"]]
    assert await redis_client.mget_many(plain, ["k1", "k2"]) == ["p1", "p2"]
    assert plain.plain_keys == [["k1", "k2"]]
    assert redis_client.mget_many(None, ["k1"]) is None


@pytest.mark.asyncio
async def test_closing_the_pipe_client_closes_the_sentinel_nodes(monkeypatch) -> None:
    monkeypatch.setattr(redis_client, "_owui_redis", None)

    target = redis_client.parse_redis_target(
        "redis://MyMaster:6379/0", "sentinel-a,sentinel-b", "", False
    )
    client = redis_client.build_redis_client(target, None)

    closed: list[str] = []

    class RecordingNode:
        def __init__(self, name: str) -> None:
            self.name = name

        async def aclose(self) -> None:
            closed.append(self.name)

    manager = getattr(client.connection_pool, "sentinel_manager")
    manager.sentinels = [RecordingNode("a"), RecordingNode("b")]

    await client.aclose()

    assert closed == ["a", "b"]
