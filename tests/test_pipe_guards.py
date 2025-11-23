from __future__ import annotations

import asyncio
import logging

from openrouter_responses_pipe.openrouter_responses_pipe import Pipe


def test_wrap_safe_event_emitter_returns_none_for_missing():
    pipe = Pipe()
    try:
        assert pipe._wrap_safe_event_emitter(None) is None
    finally:
        pipe.shutdown()


def test_wrap_safe_event_emitter_swallows_transport_errors(caplog):
    pipe = Pipe()

    async def failing_emitter(_event):
        raise RuntimeError("boom")

    wrapped = pipe._wrap_safe_event_emitter(failing_emitter)

    with caplog.at_level(logging.DEBUG):
        asyncio.run(wrapped({"type": "status"}))

    try:
        assert any("Event emitter failure" in message for message in caplog.messages)
    finally:
        pipe.shutdown()


def test_resolve_pipe_identifier_uses_fallback(caplog):
    pipe = Pipe()
    with caplog.at_level(logging.WARNING):
        result = pipe._resolve_pipe_identifier(None, fallback_model_id=None)
    try:
        assert result == "openrouter"
        assert any("Pipe identifier missing" in message for message in caplog.messages)
    finally:
        pipe.shutdown()


def test_pipe_handles_job_failure(monkeypatch):
    async def runner():
        pipe = Pipe()

        events: list[dict] = []

        async def emitter(event):
            events.append(event)

        def fake_enqueue(self, job):
            job.future.set_exception(RuntimeError("boom"))
            return True

        monkeypatch.setattr(Pipe, "_enqueue_job", fake_enqueue)

        result = await pipe.pipe(
            body={},
            __user__={"valves": {}},
            __request__=None,
            __event_emitter__=emitter,
            __event_call__=None,
            __metadata__={},
            __tools__=None,
        )

        assert result == "Request failed. Please retry."
        assert events and events[0]["type"] == "chat:completion"
        await pipe.close()

    asyncio.run(runner())
