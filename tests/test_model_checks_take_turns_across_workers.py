"""Workers that load a new pipe copy together run their model checks one at a time.

After an update, eight dev-box workers loaded the new copy within 0.4 s and each started its full
pass over every model entry at once; their parallel database reads ran Postgres out of
connections ("too many clients", 75 times), the passes misread existing entries as missing and
inserted them again, and two model lists failed with HTTP 500. Open WebUI runs work that only one
worker should do at a time under its `RedisLock`: a worker that finds it held waits a random half to
whole lock timeout and tries again, renews it while working, and releases it at the end. The
pipe's pass now takes that lock, so two workers' passes never overlap.
"""

from __future__ import annotations

import asyncio
import pytest

from open_webui_openrouter_pipe.models import catalog_manager as cm


class _SharedRedis:
    def __init__(self) -> None:
        self.holder: dict[str, str] = {}
        self.names: list[str] = []


class _OpenWebUILock:
    def __init__(self, redis: _SharedRedis, lock_name: str, timeout_secs: float) -> None:
        self.redis = redis
        self.lock_name = lock_name
        self.lock_id = f"id-{id(self)}"
        self.timeout_secs = timeout_secs
        redis.names.append(lock_name)

    def aquire_lock(self) -> bool:
        if self.lock_name in self.redis.holder:
            return False
        self.redis.holder[self.lock_name] = self.lock_id
        return True

    def renew_lock(self) -> bool:
        return self.redis.holder.get(self.lock_name) == self.lock_id

    def release_lock(self) -> None:
        if self.redis.holder.get(self.lock_name) == self.lock_id:
            del self.redis.holder[self.lock_name]


@pytest.mark.asyncio
async def test_two_workers_model_checks_never_run_at_the_same_time(monkeypatch, pipe_instance_async):
    redis = _SharedRedis()
    monkeypatch.setattr(
        cm,
        "owui_redis_lock",
        lambda name, ttl: _OpenWebUILock(redis, name, 0.04),
    )

    running = 0
    most_at_once = 0
    passes: list[str] = []

    def _pass_for(worker: str):
        async def _pass(_models, *, pipe_identifier):
            nonlocal running, most_at_once
            running += 1
            most_at_once = max(most_at_once, running)
            await asyncio.sleep(0.03)
            running -= 1
            passes.append(worker)
            return None

        return _pass

    pipe = pipe_instance_async
    pipe.valves.UPDATE_MODEL_CAPABILITIES = True
    workers: list[cm.ModelCatalogManager] = []
    for worker in ("worker-1", "worker-2"):
        manager = cm.ModelCatalogManager(
            pipe=pipe, multimodal_handler=pipe._multimodal_handler, logger=pipe.logger
        )
        monkeypatch.setattr(manager, "_sync_model_metadata_to_owui", _pass_for(worker))
        workers.append(manager)

    for manager in workers:
        manager.maybe_schedule_model_metadata_sync(
            [{"id": "openai/gpt-5-mini"}], pipe_identifier="open_webui_openrouter_pipe"
        )
    tasks = [manager._model_metadata_sync_task for manager in workers]
    assert all(task is not None for task in tasks), "a worker did not start its model check"
    await asyncio.wait_for(asyncio.gather(*(task for task in tasks if task is not None)), timeout=5)

    assert sorted(passes) == ["worker-1", "worker-2"], f"each worker must still run its own check: {passes}"
    assert most_at_once == 1, f"{most_at_once} workers ran their model check at the same time"
    assert redis.holder == {}, f"the lock was left held after the checks: {redis.holder}"
    assert set(redis.names) == {"open_webui_openrouter_pipe:model_sync_lock"}, redis.names
