from __future__ import annotations

import asyncio
import importlib
import logging
import sys
import types
from typing import Any

import pytest

LIFECYCLE_REGISTRY_KEY = "_openrouter_pipe_lifecycle"
PIPE_ID = "retire_test_pipe"


@pytest.fixture(autouse=True)
def _enable_test_mode(monkeypatch):
    monkeypatch.setenv("OWUI_PIPE_TEST_MODE", "1")


@pytest.fixture
def fresh_registry():
    saved = sys.modules.pop(LIFECYCLE_REGISTRY_KEY, None)
    try:
        yield
    finally:
        sys.modules.pop(LIFECYCLE_REGISTRY_KEY, None)
        if saved is not None:
            sys.modules[LIFECYCLE_REGISTRY_KEY] = saved


class _Broker:
    def __init__(self) -> None:
        self.queues: dict[str, list[asyncio.Queue]] = {}

    def subscribed(self, channel: str) -> bool:
        return bool(self.queues.get(channel))


class _PubSub:
    def __init__(self, broker: _Broker) -> None:
        self._broker = broker
        self._queue: asyncio.Queue = asyncio.Queue()
        self._channels: list[str] = []

    async def subscribe(self, *channels: str) -> None:
        for channel in channels:
            self._broker.queues.setdefault(channel, []).append(self._queue)
            self._channels.append(channel)

    async def listen(self):
        while True:
            yield await self._queue.get()

    async def aclose(self) -> None:
        for channel in self._channels:
            queues = self._broker.queues.get(channel, [])
            if self._queue in queues:
                queues.remove(self._queue)


class _Redis:
    def __init__(self, broker: _Broker) -> None:
        self._broker = broker

    async def initialize(self) -> "_Redis":
        return self

    async def publish(self, channel: str, data: Any) -> int:
        queues = list(self._broker.queues.get(channel, ()))
        for queue in queues:
            queue.put_nowait({"type": "message", "channel": channel, "data": data})
        return len(queues)

    def pubsub(self) -> _PubSub:
        return _PubSub(self._broker)


def _open_webui_events(monkeypatch) -> Any:
    log = logging.getLogger("open_webui.events")
    events_mod = types.ModuleType("open_webui.events")
    sinks: list[Any] = []

    async def _dispatch_event(request, event_payload):
        for sink in sinks:
            try:
                await sink.handle_event(request, event_payload)
            except Exception:
                log.exception('Event sink failed for %s', event_payload.event)

    async def publish_event(request, event, *, actor=None, subject_id=None, subject_type=None, **_):
        parts = str(event).split(".")
        resource = ".".join(parts[:-1])
        subject = (
            {"type": subject_type or resource, "id": subject_id}
            if subject_id is not None or subject_type is not None
            else None
        )
        await _dispatch_event(request, types.SimpleNamespace(event=str(event), subject=subject, data={}))

    setattr(events_mod, "EVENT_SINKS", sinks)
    setattr(events_mod, "publish_event", publish_event)
    monkeypatch.setitem(sys.modules, "open_webui.events", events_mod)
    return events_mod


def _worker(broker: _Broker, **state: Any) -> Any:
    return types.SimpleNamespace(state=types.SimpleNamespace(redis=_Redis(broker), **state))


async def _settle(predicate, rounds: int = 200) -> None:
    for _ in range(rounds):
        if predicate():
            return
        await asyncio.sleep(0)


async def _run_update_across_workers(monkeypatch, *, event: str, stored: Any) -> Any:
    from open_webui.models.functions import Functions

    pipe_mod = importlib.import_module("open_webui_openrouter_pipe.pipe")
    events_mod = _open_webui_events(monkeypatch)
    broker = _Broker()

    old_copy = pipe_mod.Pipe()
    old_copy.id = PIPE_ID
    worker_b = _worker(broker, FUNCTIONS={PIPE_ID: old_copy}, FUNCTION_CONTENTS={PIPE_ID: "code 2.9.0"})
    monkeypatch.setitem(sys.modules, "open_webui.main", types.SimpleNamespace(app=worker_b))

    async def get_function_by_id(function_id, db=None):
        return stored if function_id == PIPE_ID else None

    monkeypatch.setattr(Functions, "get_function_by_id", staticmethod(get_function_by_id))

    old_copy._maybe_start_retire_listener()
    retire_channels = [name for name in broker.queues]
    await _settle(lambda: any(broker.subscribed(name) for name in broker.queues))
    assert any(broker.subscribed(name) for name in broker.queues), (
        f"the copy on worker B never subscribed to anything: {retire_channels}"
    )

    worker_a = _worker(broker)
    await events_mod.publish_event(types.SimpleNamespace(app=worker_a), event, subject_id=PIPE_ID)
    await _settle(lambda: old_copy._draining)
    for _ in range(20):
        await asyncio.sleep(0)
    return old_copy


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_code, retired", [("code 2.9.1", True), ("code 2.9.0", False)])
async def test_an_update_on_one_worker_retires_the_old_copy_another_worker_still_runs(
    fresh_registry, monkeypatch, stored_code, retired
):
    stored = types.SimpleNamespace(id=PIPE_ID, content=stored_code)
    copy = await _run_update_across_workers(monkeypatch, event="function.updated", stored=stored)
    try:
        assert copy._draining is retired, (
            f"stored code {stored_code!r} against worker B's cached 'code 2.9.0': "
            f"expected retired={retired}, got draining={copy._draining}"
        )
    finally:
        await copy.close()


@pytest.mark.asyncio
async def test_a_delete_on_one_worker_retires_the_copy_another_worker_still_runs(fresh_registry, monkeypatch):
    copy = await _run_update_across_workers(monkeypatch, event="function.deleted", stored=None)
    try:
        assert copy._draining is True, "worker B's copy kept running after the pipe was deleted on worker A"
    finally:
        await copy.close()
