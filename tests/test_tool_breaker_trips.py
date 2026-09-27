"""A tool that keeps failing must be skipped once its breaker opens, on the path real tool calls take.

The breaker only protects anything if a failure recorded while running one call is still there when the next call
asks. These tests drive ``_execute_function_calls`` with a real worker, so every call goes through the batch executor,
the per-call breaker gate and the retry wrapper, and they count how many times the tool itself actually ran.

Status cannot tell a skip from a failure: the executor reports both as ``incomplete``. The invocation count and the
skip wording can.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import aiohttp
import pytest
from aioresponses import aioresponses

from open_webui_openrouter_pipe import EncryptedStr, Pipe, _ToolExecutionContext

# The two places a breaker skip is built: before a call is queued (a later turn), and when a queued call reaches its
# batch (later in the same turn). Each wording is the one the model reads.
_SKIP_BEFORE_QUEUEING = "skipped due to repeated failures"
_SKIP_IN_BATCH = "temporarily disabled due to repeated errors"


def _is_skip(output: dict[str, Any]) -> bool:
    text = str(output.get("output"))
    return _SKIP_BEFORE_QUEUEING in text or _SKIP_IN_BATCH in text


def _registry(tool) -> dict[str, dict[str, Any]]:
    return {
        "flaky": {
            "type": "function",
            "callable": tool,
            "spec": {"name": "flaky", "parameters": {"type": "object", "properties": {}}},
        }
    }


def _call(index: int) -> dict[str, Any]:
    return {"name": "flaky", "call_id": f"call-{index}", "arguments": "{}"}


async def _with_tool_context(pipe, body, *, workers=1, batch_cap=1, request_slots=1, global_slots=None):
    """Run ``body(executor)`` inside a real tool context; by default one worker, one call per batch and one slot."""
    context = _ToolExecutionContext(
        queue=asyncio.Queue(maxsize=50),
        per_request_semaphore=asyncio.Semaphore(request_slots),
        global_semaphore=None if global_slots is None else asyncio.Semaphore(global_slots),
        timeout=5.0,
        batch_timeout=5.0,
        idle_timeout=None,
        user_id="user-1",
        event_emitter=None,
        batch_cap=batch_cap,
    )
    executor = pipe._ensure_tool_executor()
    context.workers.extend(asyncio.create_task(executor._tool_worker_loop(context)) for _ in range(workers))
    token = pipe._TOOL_CONTEXT.set(context)
    try:
        return await body(executor)
    finally:
        pipe._TOOL_CONTEXT.reset(token)
        for worker in context.workers:
            worker.cancel()
        await asyncio.gather(*context.workers, return_exceptions=True)


def _always_failing(ran: list[str]):
    async def flaky(**_kwargs):
        ran.append("fail")
        raise RuntimeError("tool exploded")

    return flaky


class _StoppedByItsLibrary(BaseException):
    """What a tool raises when its own library stops it: reported as a failure, but not an `Exception`."""


def _failing_with(ran: list[str], error: BaseException):
    async def flaky(**_kwargs):
        ran.append("fail")
        raise error

    return flaky


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
@pytest.mark.parametrize(
    "error",
    [RuntimeError("tool exploded"), _StoppedByItsLibrary("stopped")],
    ids=["an-ordinary-exception", "a-failure-that-is-not-an-exception"],
)
async def test_every_failure_the_model_is_told_about_counts_against_the_tool(pipe_instance_async, threshold, error):
    """The switch counts what the model was told, and the model is told about both kinds.

    A tool whose library stops it raises something that is not an `Exception`, and the pipe renders that
    to the model as Open WebUI's own `{"error": ...}` text exactly as it renders an ordinary one. Counting only the ordinary
    kind means such a tool is retried for ever, a whole call timeout at a time, while the model is told
    every round that it failed.

    The two arms differ only in which class the tool raises, so a switch that reads the class rather
    than the outcome passes one and fails the other.
    """
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = threshold
    ran: list[str] = []
    registry = _registry(_failing_with(ran, error))

    async def one_call_per_turn(executor):
        outputs = []
        for index in range(threshold + 2):
            outputs.extend(await executor._execute_function_calls([_call(index)], registry))
        return outputs

    outputs = await _with_tool_context(pipe, one_call_per_turn)

    assert len(ran) == threshold, (len(ran), [output.get("output") for output in outputs])
    assert [_is_skip(output) for output in outputs] == [False] * threshold + [True, True]


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_tool_that_keeps_failing_is_skipped_on_later_turns(pipe_instance_async, threshold):
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = threshold
    ran: list[str] = []
    registry = _registry(_always_failing(ran))

    async def one_call_per_turn(executor):
        outputs = []
        for index in range(threshold + 2):
            outputs.extend(await executor._execute_function_calls([_call(index)], registry))
        return outputs

    outputs = await _with_tool_context(pipe, one_call_per_turn)

    assert len(ran) == threshold
    assert [_is_skip(output) for output in outputs] == [False] * threshold + [True, True]
    # A later turn's calls are stopped as they are queued, before any batch picks them up.
    assert [_SKIP_BEFORE_QUEUEING in str(output.get("output")) for output in outputs] == [False] * threshold + [
        True,
        True,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_tool_that_keeps_failing_is_skipped_later_in_the_same_turn(pipe_instance_async, threshold):
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = threshold
    ran: list[str] = []
    registry = _registry(_always_failing(ran))

    async def many_calls_in_one_turn(executor):
        return await executor._execute_function_calls([_call(index) for index in range(threshold + 2)], registry)

    outputs = await _with_tool_context(pipe, many_calls_in_one_turn)

    assert len(ran) == threshold
    assert [_is_skip(output) for output in outputs] == [False] * threshold + [True, True]


def _failing_after_a_moment(ran: list[str]):
    async def flaky(**_kwargs):
        ran.append("fail")
        await asyncio.sleep(0.01)
        raise RuntimeError("tool exploded")

    return flaky


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_slots", "global_slots"),
    [(1, None), (5, 1)],
    ids=["waiting-for-a-request-slot", "waiting-for-a-global-slot"],
)
@pytest.mark.parametrize("threshold", [2, 3])
async def test_calls_waiting_for_a_slot_are_skipped_once_the_breaker_opens(
    pipe_instance_async, threshold, request_slots, global_slots
):
    # Several workers and several calls per batch, as by default: every call is past the queue before any has failed,
    # so only a check made once the call holds its slot can see the failures recorded while it waited.
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = threshold
    ran: list[str] = []
    registry = _registry(_failing_after_a_moment(ran))

    async def six_calls_in_one_turn(executor):
        return await executor._execute_function_calls([_call(index) for index in range(6)], registry)

    outputs = await _with_tool_context(
        pipe, six_calls_in_one_turn, workers=5, batch_cap=4, request_slots=request_slots, global_slots=global_slots
    )

    assert len(ran) == threshold
    assert sum(_is_skip(output) for output in outputs) == 6 - threshold


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_success_clears_the_failures_recorded_before_it(pipe_instance_async, threshold):
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = threshold
    # threshold-1 failures, a success, then threshold failures: only the failures after the success count, so the
    # call after them is the first one skipped.
    plan = ["fail"] * (threshold - 1) + ["ok"] + ["fail"] * threshold
    ran: list[str] = []

    async def flaky(**_kwargs):
        step = plan[len(ran)] if len(ran) < len(plan) else "ran-after-plan"
        ran.append(step)
        if step == "ok":
            return "ok"
        raise RuntimeError("tool exploded")

    registry = _registry(flaky)

    async def one_call_per_turn(executor):
        outputs = []
        for index in range(len(plan) + 1):
            outputs.extend(await executor._execute_function_calls([_call(index)], registry))
        return outputs

    outputs = await _with_tool_context(pipe, one_call_per_turn)

    assert ran == plan
    assert [_is_skip(output) for output in outputs] == [False] * len(plan) + [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_turns_where_every_call_was_skipped_do_not_lock_the_user_out(pipe_instance_async, threshold):
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = threshold
    ran: list[str] = []
    registry = _registry(_always_failing(ran))

    async def open_the_breaker_then_keep_calling(executor):
        outputs = []
        for index in range(threshold * 3):
            outputs.extend(await executor._execute_function_calls([_call(index)], registry))
        return outputs

    outputs = await _with_tool_context(pipe, open_the_breaker_then_keep_calling)

    # The breaker really opened and every later turn was nothing but a skip ...
    assert len(ran) == threshold
    assert all(_is_skip(output) for output in outputs[threshold:])
    # ... and those skipped-only turns did not count as failed requests for the user.
    assert pipe._circuit_breaker.allows("user-1") is True
    # Real request failures still count: the same breaker refuses once they reach the threshold, so the True above
    # did not come from a breaker that can never refuse.
    for _ in range(threshold):
        pipe._circuit_breaker.record_failure("user-1")
    assert pipe._circuit_breaker.allows("user-1") is False


# --- the breaker settings an admin saves -----------------------------------------------------------------------------


def _saved_breaker_settings(pipe, threshold: int, window_seconds: int = 600) -> None:
    """Assign the pipe's valves the way Open WebUI does before each request: a new instance holding the saved values."""
    valves = pipe.valves.model_copy(update={"BREAKER_MAX_FAILURES": threshold, "BREAKER_WINDOW_SECONDS": window_seconds})
    valves.API_KEY = EncryptedStr("sk-test-key")
    pipe.valves = valves


async def _request(pipe) -> Any:
    with aioresponses() as mock_http:
        mock_http.get("https://openrouter.ai/api/v1/models", payload={"data": []}, repeat=True)
        return await pipe.pipe(
            body={"model": "openrouter.test", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            __user__={"id": "user-1", "role": "user"},
            __request__=None,
            __event_emitter__=None,
            __event_call__=None,
            __metadata__={},
            __tools__=None,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_the_tool_breaker_uses_the_failure_count_saved_for_the_pipe(monkeypatch, threshold):
    pipe = Pipe()
    ran: list[str] = []
    registry = _registry(_always_failing(ran))

    async def the_model_calls_the_tool(*_args, **_kwargs):
        outputs = await pipe._ensure_tool_executor()._execute_function_calls([_call(len(ran))], registry)
        return str(outputs[0].get("output"))

    monkeypatch.setattr(pipe, "_handle_pipe_call", the_model_calls_the_tool)
    replies = []
    try:
        for _ in range(threshold + 1):
            _saved_breaker_settings(pipe, threshold)
            replies.append(await _request(pipe))
    finally:
        await pipe.close()

    assert len(ran) == threshold, replies
    assert _SKIP_BEFORE_QUEUEING in str(replies[-1]), replies


def _tools_that_answer(ran: list[str], *names: str) -> dict[str, dict[str, Any]]:
    def tool(name: str):
        async def answer(**_kwargs):
            ran.append(name)
            return f"{name} result"

        return answer

    return {
        name: {
            "type": "function",
            "callable": tool(name),
            "spec": {"name": name, "parameters": {"type": "object", "properties": {}}},
        }
        for name in names
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("saved_count", [3, 7])
async def test_a_tool_first_called_after_the_saved_failure_count_changes_still_runs(monkeypatch, saved_count):
    # The user called one tool; then an admin saved a different failure count, and the same Pipe serves the user's next
    # requests. A tool the user had not called before, alone or beside the known one, must run and answer for itself.
    pipe = Pipe()
    ran: list[str] = []
    registry = _tools_that_answer(ran, "lookup", "search")
    registry.update(_registry(_always_failing(ran)))
    turns = iter([["lookup"], ["search"], ["lookup", "search"], ["flaky"]])

    async def the_model_calls_tools(*_args, **_kwargs):
        calls = [{"name": name, "call_id": f"call-{name}-{len(ran)}", "arguments": "{}"} for name in next(turns)]
        outputs = await pipe._ensure_tool_executor()._execute_function_calls(calls, registry)
        return json.dumps([str(output.get("output")) for output in outputs])

    monkeypatch.setattr(pipe, "_handle_pipe_call", the_model_calls_tools)
    replies = []
    try:
        _saved_breaker_settings(pipe, 5)
        replies.append(await _request(pipe))
        for _ in range(3):
            _saved_breaker_settings(pipe, saved_count)
            replies.append(await _request(pipe))
    finally:
        await pipe.close()

    assert replies == [
        json.dumps(["lookup result"]),
        json.dumps(["search result"]),
        json.dumps(["lookup result", "search result"]),
        json.dumps([_T383_ERROR_TEXT]),
    ]
    assert sorted(ran) == ["fail", "lookup", "lookup", "search", "search"]


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_the_request_breaker_uses_the_failure_count_saved_for_the_pipe(monkeypatch, threshold):
    pipe = Pipe()

    async def the_model_answers(*_args, **_kwargs):
        return "answered"

    monkeypatch.setattr(pipe, "_handle_pipe_call", the_model_answers)
    try:
        _saved_breaker_settings(pipe, threshold)
        first = await _request(pipe)
        for _ in range(threshold):
            pipe._circuit_breaker.record_failure("user-1")
        _saved_breaker_settings(pipe, threshold)
        refused = await _request(pipe)
    finally:
        await pipe.close()

    assert first == "answered"
    assert "Temporarily disabled due to repeated errors" in str(refused), refused


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3, 6, 8])
async def test_the_database_breaker_uses_the_failure_count_saved_for_the_pipe(monkeypatch, threshold):
    pipe = Pipe()

    async def the_model_answers(*_args, **_kwargs):
        return "answered"

    monkeypatch.setattr(pipe, "_handle_pipe_call", the_model_answers)
    try:
        _saved_breaker_settings(pipe, threshold)
        await _request(pipe)
        store = pipe._artifact_store
        for _ in range(threshold - 1):
            store._record_db_failure("user-1")
        allowed_one_failure_short = store._db_breaker_allows("user-1")
        store._record_db_failure("user-1")
        allowed_at_the_saved_count = store._db_breaker_allows("user-1")
    finally:
        await pipe.close()

    assert allowed_one_failure_short is True
    assert allowed_at_the_saved_count is False


@pytest.mark.asyncio
@pytest.mark.parametrize(("first_count", "new_count"), [(6, 7), (6, 8)])
async def test_database_failures_recorded_before_the_saved_count_changes_still_count(
    monkeypatch, first_count, new_count
):
    pipe = Pipe()

    async def the_model_answers(*_args, **_kwargs):
        return "answered"

    monkeypatch.setattr(pipe, "_handle_pipe_call", the_model_answers)
    try:
        _saved_breaker_settings(pipe, first_count)
        await _request(pipe)
        store = pipe._artifact_store
        for _ in range(first_count - 1):
            store._record_db_failure("user-1")
        _saved_breaker_settings(pipe, new_count)
        await _request(pipe)
        for _ in range(new_count - first_count):
            store._record_db_failure("user-1")
        allowed_one_failure_short = store._db_breaker_allows("user-1")
        store._record_db_failure("user-1")
        allowed_at_the_new_count = store._db_breaker_allows("user-1")
    finally:
        await pipe.close()

    assert allowed_one_failure_short is True
    assert allowed_at_the_new_count is False


@pytest.mark.asyncio
@pytest.mark.parametrize(("window_seconds", "refused"), [(30, False), (90, True)])
async def test_the_request_breaker_uses_the_window_saved_for_the_pipe(monkeypatch, window_seconds, refused):
    pipe = Pipe()

    async def the_model_answers(*_args, **_kwargs):
        return "answered"

    monkeypatch.setattr(pipe, "_handle_pipe_call", the_model_answers)
    try:
        _saved_breaker_settings(pipe, 2, window_seconds)
        await _request(pipe)
        # Two failures recorded 45 s ago: inside a 90 s window, outside a 30 s one.
        records = pipe._circuit_breaker._breaker_records["user-1"]
        records.clear()
        records.extend([time.time() - 45] * 2)
        _saved_breaker_settings(pipe, 2, window_seconds)
        reply = await _request(pipe)
    finally:
        await pipe.close()

    assert ("Temporarily disabled due to repeated errors" in str(reply)) is refused, reply


@pytest.mark.asyncio
@pytest.mark.parametrize(("window_seconds", "allowed"), [(30, True), (90, False)])
async def test_the_database_breaker_uses_the_window_saved_for_the_pipe(monkeypatch, window_seconds, allowed):
    pipe = Pipe()

    async def the_model_answers(*_args, **_kwargs):
        return "answered"

    monkeypatch.setattr(pipe, "_handle_pipe_call", the_model_answers)
    try:
        _saved_breaker_settings(pipe, 2, window_seconds)
        await _request(pipe)
        store = pipe._artifact_store
        # Two database failures recorded 45 s ago: inside a 90 s window, outside a 30 s one.
        store._db_breakers["user-1"].extend([time.time() - 45] * 2)
        store_allows = store._db_breaker_allows("user-1")
    finally:
        await pipe.close()

    assert store_allows is allowed


# --- failed requests trip the request breaker ------------------------------------------------------------------------

_FAILED_UPSTREAM = {"status": 500, "payload": {"error": {"message": "upstream trouble"}}}
_ANSWERED_UPSTREAM = {
    "status": 200,
    "payload": {
        "id": "resp-1",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello."}]}],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    },
}


def _reach_the_model(monkeypatch, pipe) -> None:
    import open_webui_openrouter_pipe.pipe as pipe_mod

    async def loaded(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test-key", None))
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", loaded)
    monkeypatch.setattr(
        pipe_mod.OpenRouterModelRegistry, "list_models", lambda: [{"id": "m1", "name": "Model m1", "norm_id": "m1"}]
    )


async def _chat_turn(pipe, *, stream: bool) -> str:
    result = await pipe.pipe(
        body={"model": "m1", "messages": [{"role": "user", "content": "hi"}], "stream": stream},
        __user__={"id": "user-1", "role": "user"},
        __request__=None,
        __event_emitter__=None,
        __event_call__=None,
        __metadata__={"chat_id": "chat-1", "message_id": "message-1", "model": {"id": "m1"}},
        __tools__=None,
    )
    if hasattr(result, "__aiter__"):
        return "".join([str(chunk) async for chunk in result])
    return str(result)


def _posts(mock_http) -> int:
    return sum(len(calls) for (method, _url), calls in mock_http.requests.items() if method == "POST")


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [True, False], ids=["streaming", "not-streaming"])
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_user_whose_requests_keep_failing_upstream_is_refused_at_the_saved_count(monkeypatch, threshold, stream):
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            mock_http.post("https://openrouter.ai/api/v1/responses", repeat=True, **_FAILED_UPSTREAM)
            for _ in range(threshold + 1):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=stream))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert posts == threshold, replies
    assert "Temporarily disabled due to repeated errors" in replies[-1], replies
    assert not any("Temporarily disabled" in reply for reply in replies[:-1]), replies


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_request_that_ends_without_an_error_clears_the_failures_before_it(monkeypatch, threshold):
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    outcomes = [_FAILED_UPSTREAM] * (threshold - 1) + [_ANSWERED_UPSTREAM] + [_FAILED_UPSTREAM] * (threshold - 1)
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            for outcome in outcomes:
                mock_http.post("https://openrouter.ai/api/v1/responses", **outcome)
            mock_http.post("https://openrouter.ai/api/v1/responses", **_FAILED_UPSTREAM)
            for _ in range(len(outcomes) + 1):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert posts == len(outcomes) + 1, replies
    assert "Hello." in replies[threshold - 1], replies
    assert not any("Temporarily disabled" in reply for reply in replies), replies


_TOOL_CALL_UPSTREAM = {
    "status": 200,
    "payload": {
        "id": "resp-tools",
        "output": [
            {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}", "status": "completed"}
        ],
        "usage": {"input_tokens": 1, "output_tokens": 1},
    },
}


def _open_webui_tool_mode(pipe, threshold: int) -> dict[str, dict[str, Any]]:
    """Save the breaker settings as Open WebUI does, with the tool backend switched to Open WebUI, and offer one tool."""
    valves = pipe.valves.model_copy(
        update={"BREAKER_MAX_FAILURES": threshold, "BREAKER_WINDOW_SECONDS": 600, "TOOL_EXECUTION_MODE": "Open-WebUI"}
    )
    valves.API_KEY = EncryptedStr("sk-test-key")
    pipe.valves = valves

    async def lookup(**_kwargs: Any) -> str:
        return "found"

    return {
        "lookup": {
            "type": "function",
            "callable": lookup,
            "spec": {"name": "lookup", "description": "Look it up", "parameters": {"type": "object", "properties": {}}},
        }
    }


async def _chat_turn_with_tools(pipe, tools: dict[str, dict[str, Any]]) -> str:
    result = await pipe.pipe(
        body={"model": "m1", "messages": [{"role": "user", "content": "hi"}], "stream": False},
        __user__={"id": "user-1", "role": "user"},
        __request__=None,
        __event_emitter__=None,
        __event_call__=None,
        __metadata__={"chat_id": "chat-1", "message_id": "message-1", "model": {"id": "m1"}},
        __tools__=tools,
    )
    return str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_handing_tool_calls_back_to_open_webui_clears_the_failures_before_it(monkeypatch, threshold):
    """With streaming off, the Open WebUI tool backend answers the turn with the model's tool calls. That call to
    OpenRouter succeeded, so it clears the user's failures like any other request that ends without an error."""
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    try:
        with aioresponses() as mock_http:
            for _ in range(threshold - 1):
                mock_http.post("https://openrouter.ai/api/v1/responses", **_FAILED_UPSTREAM)
            mock_http.post("https://openrouter.ai/api/v1/responses", **_TOOL_CALL_UPSTREAM)
            for _ in range(threshold - 1):
                _saved_breaker_settings(pipe, threshold)
                await _chat_turn(pipe, stream=False)
            before = len(pipe._circuit_breaker._breaker_records.get("user-1", []))
            reply = await _chat_turn_with_tools(pipe, _open_webui_tool_mode(pipe, threshold))
            after = len(pipe._circuit_breaker._breaker_records.get("user-1", []))
    finally:
        await pipe.close()

    assert before == threshold - 1, before
    assert "tool_calls" in reply and "lookup" in reply, reply
    assert after == 0, (before, after, reply)


_TITLE_UPSTREAM = {
    "status": 200,
    "payload": {
        "id": "resp-title",
        "output": [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": '{"title": "Pipe title"}'}]}],
        "usage": {},
    },
}


_CHAT_TITLE_UPSTREAM = {
    "status": 200,
    "payload": {
        "id": "chatcmpl-title",
        "object": "chat.completion",
        "model": "m1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": '{"title": "Pipe title"}'},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    },
}
"""The same title in the shape /chat/completions delivers.

A task uses whichever endpoint a chat turn would (T379), so under
`DEFAULT_LLM_ENDPOINT=chat_completions` the title arrives here rather than on /responses.
"""


async def _title_task(pipe, task: str = "title_generation") -> str:
    result = await pipe.pipe(
        body={"model": "m1", "messages": [{"role": "user", "content": "Write a title for this chat."}], "stream": False},
        __user__={"id": "user-1", "role": "user"},
        __request__=None,
        __event_emitter__=None,
        __event_call__=None,
        __metadata__={"chat_id": "chat-1", "message_id": "message-1", "model": {"id": "m1"}, "task": task},
        __tools__=None,
        __task__=task,
    )
    return str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("task", ["title_generation", "moa_response_generation"])
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_background_task_that_succeeds_does_not_clear_a_users_failed_requests(monkeypatch, threshold, task):
    """Open WebUI runs title, tag and follow-up tasks after a chat turn, often on another model. If their successes
    cleared the count, a user whose chats keep failing would never be refused. A mixture-of-agents task runs through
    the same loop as a chat, so it reports a finished call like one."""
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            for _ in range(threshold - 1):
                mock_http.post("https://openrouter.ai/api/v1/responses", **_FAILED_UPSTREAM)
            mock_http.post("https://openrouter.ai/api/v1/responses", **_TITLE_UPSTREAM)
            mock_http.post("https://openrouter.ai/api/v1/responses", repeat=True, **_FAILED_UPSTREAM)
            for _ in range(threshold - 1):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            _saved_breaker_settings(pipe, threshold)
            title = await _title_task(pipe, task)
            for _ in range(2):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert "Pipe title" in title, title
    assert posts == threshold + 1, (title, replies)
    assert "Temporarily disabled due to repeated errors" in replies[-1], replies


async def _chat_turn_ending_in_an_error_of_its_own(monkeypatch, pipe, own_error: str) -> str:
    import open_webui_openrouter_pipe.pipe as pipe_mod

    with monkeypatch.context() as patch:
        if own_error == "missing-api-key":
            patch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: (None, "No API key configured."))
            patch.setattr(pipe, "_note_auth_failure", lambda: None)
        elif own_error == "configuration-error":
            async def misconfigured(*_args: Any, **_kwargs: Any) -> None:
                raise ValueError("the base URL is not valid")

            patch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", misconfigured)
        elif own_error == "catalog-unavailable":
            async def catalog_down(*_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("catalog down")

            patch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", catalog_down)
            patch.setattr(pipe_mod.OpenRouterModelRegistry, "list_models", lambda: [])
        else:
            async def request_could_not_be_built(*_args: Any, **_kwargs: Any) -> None:
                raise RuntimeError("request could not be built")

            patch.setattr(pipe, "_process_transformed_request", request_could_not_be_built)
        return await _chat_turn(pipe, stream=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "own_error", ["missing-api-key", "configuration-error", "catalog-unavailable", "error-before-streaming"]
)
async def test_a_request_that_ends_in_an_error_of_its_own_does_not_clear_earlier_failures(monkeypatch, own_error):
    threshold = 2
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            mock_http.post("https://openrouter.ai/api/v1/responses", repeat=True, **_FAILED_UPSTREAM)
            _saved_breaker_settings(pipe, threshold)
            replies.append(await _chat_turn(pipe, stream=False))
            _saved_breaker_settings(pipe, threshold)
            replies.append(await _chat_turn_ending_in_an_error_of_its_own(monkeypatch, pipe, own_error))
            for _ in range(2):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert posts == threshold, replies
    assert "Temporarily disabled due to repeated errors" in replies[-1], replies


_REFUSAL_HEADINGS = {
    "model-without-zero-data-retention": "Model restricted",
    "zero-data-retention-list-unavailable": "Model restricted",
    "model-outside-the-allowlist": "Model restricted",
    "attachment-that-cannot-be-read": "Direct Upload Issue",
    "endpoint-forced-against-a-preset": "Endpoint Override Conflict",
}


async def _chat_turn_refused_before_the_model_is_called(monkeypatch, pipe, refusal: str) -> str:
    import open_webui_openrouter_pipe.pipe as pipe_mod

    saved = pipe.valves
    body: dict[str, Any] = {"model": "m1", "messages": [{"role": "user", "content": "hi"}], "stream": False}
    metadata: dict[str, Any] = {"chat_id": "chat-1", "message_id": "message-1", "model": {"id": "m1"}}
    with monkeypatch.context() as patch:
        if refusal in {"model-without-zero-data-retention", "zero-data-retention-list-unavailable"}:
            capable = False if refusal == "model-without-zero-data-retention" else None
            pipe.valves = saved.model_copy(update={"ZDR_ENFORCE": True})
            patch.setattr(pipe_mod.OpenRouterModelRegistry, "is_zdr_capable", classmethod(lambda _cls, _model_id: capable))
        elif refusal == "model-outside-the-allowlist":
            pipe.valves = saved.model_copy(update={"MODEL_ID": "m2"})
            patch.setattr(
                pipe_mod.OpenRouterModelRegistry,
                "list_models",
                lambda: [{"id": "m1", "name": "Model m1", "norm_id": "m1"}, {"id": "m2", "name": "Model m2", "norm_id": "m2"}],
            )
        elif refusal == "attachment-that-cannot-be-read":
            metadata["openrouter_pipe"] = {"direct_uploads": {"audio": [{"id": "missing-audio", "format": "mp3"}]}}
        else:
            pipe.valves = saved.model_copy(update={"FORCE_RESPONSES_MODELS": "m1"})
            body["preset"] = "house-style"
        try:
            result = await pipe.pipe(
                body=body,
                __user__={"id": "user-1", "role": "user"},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__=metadata,
                __tools__=None,
            )
        finally:
            pipe.valves = saved
    return str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", list(_REFUSAL_HEADINGS))
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_request_refused_before_the_model_is_called_does_not_clear_earlier_failures(
    monkeypatch, threshold, refusal
):
    # Only a finished call clears the count. A refusal is not one, however early it comes.
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            mock_http.post("https://openrouter.ai/api/v1/responses", repeat=True, **_FAILED_UPSTREAM)
            for _ in range(threshold - 1):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            _saved_breaker_settings(pipe, threshold)
            refused = await _chat_turn_refused_before_the_model_is_called(monkeypatch, pipe, refusal)
            for _ in range(2):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert _REFUSAL_HEADINGS[refusal] in refused, refused
    assert posts == threshold, (refused, replies)
    assert "Temporarily disabled due to repeated errors" in replies[-1], replies


# --- image and video generations count toward the request breaker (decision T145) ------------------------------------

_IMAGE_MODEL = "black-forest-labs/flux.2-pro"
_VIDEO_MODEL = "openai/sora-2-pro"
_PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
_MP4_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32


def _reach_the_media_models(monkeypatch, pipe) -> None:
    import open_webui_openrouter_pipe.pipe as pipe_mod
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures"
    images = json.loads((fixtures / "openrouter_image_models.json").read_text(encoding="utf-8"))["data"]
    videos = json.loads((fixtures / "video_models_catalog.json").read_text(encoding="utf-8"))["data"]
    _reach_the_model(monkeypatch, pipe)
    pipe_mod.OpenRouterModelRegistry.register_image_models([m for m in images if m["id"] == _IMAGE_MODEL])
    pipe_mod.OpenRouterModelRegistry.register_video_models([m for m in videos if m["id"] == _VIDEO_MODEL])
    monkeypatch.setattr(
        pipe_mod.OpenRouterModelRegistry,
        "list_models",
        lambda: [{"id": "m1", "name": "Model m1", "norm_id": "m1"}]
        + [{"id": model.replace("/", "."), "name": model, "norm_id": model.replace("/", ".")} for model in (_IMAGE_MODEL, _VIDEO_MODEL)],
    )


class _ImageModelReplies:
    """Stands in for OpenRouter's image endpoint: each generation takes the next reply in order."""

    def __init__(self, replies: list[str]) -> None:
        self.replies = list(replies)
        self.sent = 0

    async def generate(self, payload: dict[str, Any], **_kwargs: Any):
        from open_webui_openrouter_pipe.integrations.image_types import GeneratedImage, ImageGenerationResult

        self.sent += 1
        reply = self.replies.pop(0)
        if reply == "dropped":
            raise aiohttp.ClientConnectionError("dropped")
        return ImageGenerationResult(images=[GeneratedImage(data=_PNG_BYTES, mime_type="image/png")], usage={})


def _answer_images_with(monkeypatch, pipe, replies: _ImageModelReplies) -> None:
    adapter = pipe._ensure_image_generation_adapter()

    async def no_endpoint_record(*_args: Any, **_kwargs: Any):
        return {}, None

    async def stored(*_args: Any, **_kwargs: Any) -> str:
        return "file-1"

    monkeypatch.setattr(adapter, "_endpoint_record", no_endpoint_record)
    monkeypatch.setattr(adapter, "_client", lambda *_args, **_kwargs: replies)
    monkeypatch.setattr(adapter, "_persist", stored)


async def _media_turn(pipe, model: str, prompt: str = "a lighthouse at dusk") -> str:
    norm_id = model.replace("/", ".")
    result = await pipe.pipe(
        body={"model": norm_id, "messages": [{"role": "user", "content": prompt}], "stream": False},
        __user__={"id": "user-1", "role": "user"},
        __request__=None,
        __event_emitter__=None,
        __event_call__=None,
        __metadata__={"chat_id": "chat-1", "message_id": f"message-{model}-{prompt[:8]}", "model": {"id": norm_id}},
        __tools__=None,
    )
    return str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_failed_image_generation_counts_once_toward_the_users_refusal(monkeypatch, threshold):
    pipe = Pipe()
    _reach_the_media_models(monkeypatch, pipe)
    images = _ImageModelReplies(["dropped"])
    _answer_images_with(monkeypatch, pipe, images)
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            mock_http.post("https://openrouter.ai/api/v1/responses", repeat=True, **_FAILED_UPSTREAM)
            for _ in range(threshold - 2):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            _saved_breaker_settings(pipe, threshold)
            image_reply = await _media_turn(pipe, _IMAGE_MODEL)
            recorded = len(pipe._circuit_breaker._breaker_records["user-1"])
            for _ in range(2):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert images.sent == 1, image_reply
    assert "Image generation failed" in image_reply, image_reply
    assert recorded == threshold - 1, (recorded, image_reply)
    assert posts == threshold - 1, (image_reply, replies)
    assert "Temporarily disabled due to repeated errors" in replies[-1], replies


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_delivered_image_clears_the_users_failures(monkeypatch, threshold):
    pipe = Pipe()
    _reach_the_media_models(monkeypatch, pipe)
    images = _ImageModelReplies(["delivered"])
    _answer_images_with(monkeypatch, pipe, images)
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            mock_http.post("https://openrouter.ai/api/v1/responses", repeat=True, **_FAILED_UPSTREAM)
            for _ in range(threshold - 1):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            _saved_breaker_settings(pipe, threshold)
            image_reply = await _media_turn(pipe, _IMAGE_MODEL)
            recorded = len(pipe._circuit_breaker._breaker_records["user-1"])
            for _ in range(threshold):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert "/api/v1/files/file-1/content" in image_reply, image_reply
    assert recorded == 0, (recorded, image_reply)
    assert posts == 2 * threshold - 1, (image_reply, replies)
    assert not any("Temporarily disabled" in reply for reply in replies), replies


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_an_image_request_refused_before_it_is_sent_neither_counts_nor_clears(monkeypatch, threshold):
    pipe = Pipe()
    _reach_the_media_models(monkeypatch, pipe)
    images = _ImageModelReplies([])
    _answer_images_with(monkeypatch, pipe, images)
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            mock_http.post("https://openrouter.ai/api/v1/responses", repeat=True, **_FAILED_UPSTREAM)
            for _ in range(threshold - 1):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            _saved_breaker_settings(pipe, threshold)
            image_reply = await _media_turn(pipe, _IMAGE_MODEL, prompt="   ")
            recorded = len(pipe._circuit_breaker._breaker_records["user-1"])
            for _ in range(2):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert images.sent == 0, image_reply
    assert recorded == threshold - 1, (recorded, image_reply)
    assert posts == threshold, (image_reply, replies)
    assert "Temporarily disabled due to repeated errors" in replies[-1], replies


class _VideoMemoryPersistence:
    async def load_message_content(self, *, chat_id: str, message_id: str) -> str:
        return ""


def _answer_videos_with(monkeypatch, pipe, outcome: str) -> list[str]:
    """Stand in for OpenRouter's video endpoints: a submit that drops, a job that fails, one still running, or one that finishes."""
    from typing import cast

    calls: list[str] = []
    adapter = pipe._ensure_video_generation_adapter()
    cast(Any, adapter)._persistence = _VideoMemoryPersistence()

    class _VideoReplies:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        async def submit(self, _payload: dict[str, Any]) -> dict[str, Any]:
            calls.append("submit")
            if outcome == "submit-dropped":
                raise aiohttp.ClientConnectionError("dropped")
            return {"id": "job-1", "status": "pending"}

        async def status(self, _job_id: str, polling_url: Any = None) -> dict[str, Any]:
            calls.append("status")
            if outcome == "job-failed":
                return {"status": "failed", "error": {"message": "provider stopped"}}
            if outcome == "still-running":
                return {"status": "in_progress"}
            return {"status": "completed", "usage": {"cost": 0.1}}

        def content_url(self, job_id: str, index: int = 0) -> str:
            return f"https://openrouter.ai/api/v1/videos/{job_id}/content"

        def bearer_header(self) -> dict[str, str]:
            return {"Authorization": "Bearer test"}

    async def downloaded(url: str, dest_path: Any, **_kwargs: Any) -> dict[str, Any]:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        dest_path.write_bytes(_MP4_BYTES)
        return {"path": dest_path, "mime_type": "video/mp4", "url": url, "size_bytes": len(_MP4_BYTES)}

    async def stored(*_args: Any, **_kwargs: Any) -> str:
        return "file-1"

    monkeypatch.setattr("open_webui_openrouter_pipe.integrations.video.OpenRouterVideoClient", _VideoReplies)
    monkeypatch.setattr(pipe._multimodal_handler, "_download_remote_url_streaming", downloaded)
    monkeypatch.setattr(pipe._file_gateway, "upload_to_owui_storage_from_path", stored)
    pipe.valves = pipe.valves.model_copy(
        update={
            "VIDEO_INTENT_ENABLED": False,
            "VIDEO_INITIAL_POLL_DELAY_SECONDS": 0,
            "VIDEO_POLL_INTERVAL_SECONDS": 0,
            "VIDEO_POLL_INTERVAL_MAX_SECONDS": 0,
        }
    )
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["submit-dropped", "job-failed"])
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_failed_video_generation_counts_once_toward_the_users_refusal(monkeypatch, threshold, outcome):
    pipe = Pipe()
    _reach_the_media_models(monkeypatch, pipe)
    calls = _answer_videos_with(monkeypatch, pipe, outcome)
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            mock_http.post("https://openrouter.ai/api/v1/responses", repeat=True, **_FAILED_UPSTREAM)
            for _ in range(threshold - 2):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            _saved_breaker_settings(pipe, threshold)
            video_reply = await _media_turn(pipe, _VIDEO_MODEL)
            recorded = len(pipe._circuit_breaker._breaker_records["user-1"])
            for _ in range(2):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert calls.count("submit") == 1, (calls, video_reply)
    assert "Video generation failed" in video_reply, video_reply
    assert recorded == threshold - 1, (recorded, video_reply)
    assert posts == threshold - 1, (video_reply, replies)
    assert "Temporarily disabled due to repeated errors" in replies[-1], replies


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_delivered_video_clears_the_users_failures(monkeypatch, threshold):
    pipe = Pipe()
    _reach_the_media_models(monkeypatch, pipe)
    _answer_videos_with(monkeypatch, pipe, "delivered")
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            mock_http.post("https://openrouter.ai/api/v1/responses", repeat=True, **_FAILED_UPSTREAM)
            for _ in range(threshold - 1):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            _saved_breaker_settings(pipe, threshold)
            video_reply = await _media_turn(pipe, _VIDEO_MODEL)
            recorded = len(pipe._circuit_breaker._breaker_records["user-1"])
            for _ in range(threshold):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert "/api/v1/files/file-1/content" in video_reply, video_reply
    assert recorded == 0, (recorded, video_reply)
    assert posts == 2 * threshold - 1, (video_reply, replies)
    assert not any("Temporarily disabled" in reply for reply in replies), replies


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_video_job_still_running_when_its_status_window_closes_neither_counts_nor_clears(monkeypatch, threshold):
    pipe = Pipe()
    _reach_the_media_models(monkeypatch, pipe)
    calls = _answer_videos_with(monkeypatch, pipe, "still-running")
    monkeypatch.setattr("open_webui_openrouter_pipe.integrations.video._video_stall_window", lambda _valves: 0.0)
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            mock_http.post("https://openrouter.ai/api/v1/responses", repeat=True, **_FAILED_UPSTREAM)
            for _ in range(threshold - 1):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            _saved_breaker_settings(pipe, threshold)
            video_reply = await _media_turn(pipe, _VIDEO_MODEL)
            recorded = len(pipe._circuit_breaker._breaker_records["user-1"])
            for _ in range(2):
                _saved_breaker_settings(pipe, threshold)
                replies.append(await _chat_turn(pipe, stream=False))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert calls.count("submit") == 1, (calls, video_reply)
    assert "OpenRouter is still working on this video" in video_reply, video_reply
    assert recorded == threshold - 1, (recorded, video_reply)
    assert posts == threshold, (video_reply, replies)
    assert "Temporarily disabled due to repeated errors" in replies[-1], replies


# --- the breaker refuses the next request, not work already under way ------------------------------------------------


def _sse_answer() -> bytes:
    """One complete streamed answer, in the shape the /responses transport delivers."""
    events = [
        {"type": "response.output_text.delta", "delta": "Hello."},
        {
            "type": "response.completed",
            "response": {
                "id": "resp-1",
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Hello."}]}
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
    ]
    return ("".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n").encode("utf-8")


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_request_under_way_keeps_the_retry_its_own_failure_would_refuse(monkeypatch, threshold):
    """Earlier failures plus this request's dropped connection reach the saved count while it is still running.

    The count refuses the user's NEXT request. This one keeps its own retry, so the user reads the answer rather
    than an unexpected-error card.
    """
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    try:
        with aioresponses() as mock_http:
            mock_http.post("https://openrouter.ai/api/v1/responses", exception=aiohttp.ClientConnectionError("dropped"))
            mock_http.post("https://openrouter.ai/api/v1/responses", body=_sse_answer(), status=200, repeat=True)
            _saved_breaker_settings(pipe, threshold)
            for _ in range(threshold - 1):
                pipe._circuit_breaker.record_failure("user-1")
            reply = await _chat_turn(pipe, stream=True)
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert posts == 2, reply
    assert "Hello." in reply, reply
    assert "Unexpected Error" not in reply, reply


# --- a dropped connection counts like an error reply, on every transport ---------------------------------------------

_RESPONSES_URL = "https://openrouter.ai/api/v1/responses"
_CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


def _chat_sse_answer() -> bytes:
    """One complete streamed answer, in the shape the /chat/completions transport delivers."""
    chunks = [
        {"choices": [{"index": 0, "delta": {"content": "Hello."}, "finish_reason": None}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    ]
    return ("".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n").encode("utf-8")


_ANSWER_BY_TRANSPORT: dict[str, dict[str, Any]] = {
    "responses-streaming": {"body": _sse_answer(), "status": 200},
    "responses-not-streaming": _ANSWERED_UPSTREAM,
    "chat-streaming": {"body": _chat_sse_answer(), "status": 200},
    "chat-not-streaming": {
        "status": 200,
        "payload": {"id": "chat-1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}}]},
    },
}


async def _one_call(pipe, session, transport: str, key: str) -> None:
    common: dict[str, Any] = {
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "valves": pipe.valves,
        "breaker_key": key,
    }
    if transport == "responses-streaming":
        request = {"model": "openai/gpt-4o", "stream": True, "input": []}
        async for _ in pipe.send_openai_responses_streaming_request(session, request, **common):
            pass
    elif transport == "responses-not-streaming":
        await pipe.send_openai_responses_nonstreaming_request(session, {"model": "openai/gpt-4o", "input": []}, **common)
    elif transport == "chat-streaming":
        request = {"model": "openai/gpt-4o", "stream": True, "input": []}
        async for _ in pipe.send_openai_chat_completions_streaming_request(session, request, **common):
            pass
    else:
        await pipe.send_openai_chat_completions_nonstreaming_request(
            session, {"model": "openai/gpt-4o", "input": []}, **common
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "drop",
    [aiohttp.ClientConnectionError("dropped"), asyncio.TimeoutError()],
    ids=["connection-dropped", "timed-out"],
)
@pytest.mark.parametrize("drops", [1, 2])
@pytest.mark.parametrize("transport", list(_ANSWER_BY_TRANSPORT))
async def test_every_transport_counts_each_dropped_connection_against_the_user(
    pipe_instance_async, transport, drops, drop
):
    """A connection that drops before OpenRouter answers is one failed call, however the request travels.

    The attempts that drop are retried and the last one answers, so the count is exactly the number of drops.
    """
    pipe = pipe_instance_async
    key = f"dropped-{transport}"
    url = _CHAT_URL if transport.startswith("chat") else _RESPONSES_URL
    session = pipe._create_http_session(pipe.valves)
    try:
        with aioresponses() as mock_http:
            for _ in range(drops):
                mock_http.post(url, exception=drop)
            mock_http.post(url, **_ANSWER_BY_TRANSPORT[transport])
            await _one_call(pipe, session, transport, key)
    finally:
        await session.close()

    assert len(pipe._circuit_breaker._breaker_records[key]) == drops


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("endpoint", "stream"),
    [("responses", True), ("responses", False), ("chat_completions", True), ("chat_completions", False)],
    ids=["responses-streaming", "responses-not-streaming", "chat-streaming", "chat-not-streaming"],
)
async def test_a_user_whose_calls_keep_dropping_is_refused_at_the_saved_count(monkeypatch, endpoint, stream):
    """During an outage the user reaches the limit and reads the retry-later notice, not a run of error cards."""
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    url = _RESPONSES_URL if endpoint == "responses" else _CHAT_URL
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            mock_http.post(url, exception=aiohttp.ClientConnectionError("dropped"), repeat=True)
            for _ in range(2):
                _saved_breaker_settings(pipe, 2)
                pipe.valves = pipe.valves.model_copy(update={"DEFAULT_LLM_ENDPOINT": endpoint})
                replies.append(await _chat_turn(pipe, stream=stream))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert posts == 3, replies
    assert "Temporarily disabled" not in replies[0], replies
    assert "Temporarily disabled due to repeated errors" in replies[-1], replies


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", list(_ANSWER_BY_TRANSPORT))
async def test_every_transport_counts_an_error_reply_exactly_once(pipe_instance_async, transport):
    """An error reply is one failed call on every transport: counted once, never twice by two handlers."""
    pipe = pipe_instance_async
    key = f"error-reply-{transport}"
    url = _CHAT_URL if transport.startswith("chat") else _RESPONSES_URL
    session = pipe._create_http_session(pipe.valves)
    try:
        with aioresponses() as mock_http:
            mock_http.post(url, **_FAILED_UPSTREAM)
            with pytest.raises(Exception):
                await _one_call(pipe, session, transport, key)
    finally:
        await session.close()

    assert len(pipe._circuit_breaker._breaker_records[key]) == 1


# --- a request that brings tool results finishes an answer already under way ----------------------------------------

_A_TOOL_ROUND = [
    {"role": "user", "content": "Look it up."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}],
    },
    {"role": "tool", "tool_call_id": "call-1", "content": "found it"},
]
_IMAGES_FROM_THE_TOOL_ROUND = [
    *_A_TOOL_ROUND,
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Here are the images from the tool results above. Please analyze them."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
        ],
    },
]


async def _turn_with_messages(pipe, messages: list[dict[str, Any]], **metadata: Any) -> str:
    result = await pipe.pipe(
        body={"model": "m1", "messages": messages, "stream": False},
        __user__={"id": "user-1", "role": "user"},
        __request__=None,
        __event_emitter__=None,
        __event_call__=None,
        __metadata__={"chat_id": "chat-1", "message_id": "message-1", "model": {"id": "m1"}, **metadata},
        __tools__=None,
    )
    return str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("messages", "metadata"),
    [
        (_A_TOOL_ROUND, {}),
        (_IMAGES_FROM_THE_TOOL_ROUND, {}),
        (_A_TOOL_ROUND, {"assistant_message_id": "message-1"}),
    ],
    ids=["ends-on-the-tool-result", "open-webuis-image-message-follows-the-result", "a-resume-names-the-message-under-way"],
)
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_request_bringing_tool_results_is_answered_at_the_saved_count(monkeypatch, threshold, messages, metadata):
    """Open WebUI runs a tool and calls the pipe again to finish the same answer: that answer is under way. On 0.11.1 and
    later a resume after a tool approval or an ask_user answer arrives the same way, naming the message it continues."""
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    try:
        with aioresponses() as mock_http:
            mock_http.post(_RESPONSES_URL, repeat=True, **_ANSWERED_UPSTREAM)
            _saved_breaker_settings(pipe, threshold)
            for _ in range(threshold):
                pipe._circuit_breaker.record_failure("user-1")
            reply = await _turn_with_messages(pipe, messages, **metadata)
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert posts == 1, reply
    assert "Temporarily disabled" not in reply, reply
    assert "Hello." in reply, reply


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_new_question_after_a_tool_round_is_still_refused_at_the_saved_count(monkeypatch, threshold):
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    try:
        with aioresponses() as mock_http:
            mock_http.post(_RESPONSES_URL, repeat=True, **_ANSWERED_UPSTREAM)
            _saved_breaker_settings(pipe, threshold)
            for _ in range(threshold):
                pipe._circuit_breaker.record_failure("user-1")
            reply = await _turn_with_messages(
                pipe,
                [*_A_TOOL_ROUND, {"role": "assistant", "content": "It found it."}, {"role": "user", "content": "Thanks."}],
            )
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert posts == 0, reply
    assert "Temporarily disabled due to repeated errors" in reply, reply


# --- a Fusion turn counts like any other request ----------------------------------------------------------------------


def _reach_fusion(monkeypatch, pipe) -> None:
    import open_webui_openrouter_pipe.pipe as pipe_mod
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager

    async def loaded(*_args: Any, **_kwargs: Any) -> None:
        return None

    models = [
        {"id": "m1", "name": "Model m1", "norm_id": "m1"},
        {"id": "openrouter/fusion", "name": "Fusion", "norm_id": "openrouter.fusion"},
    ]

    async def no_web_tools(*_args: Any, **_kwargs: Any) -> None:
        return None

    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test-key", None))
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", loaded)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "list_models", lambda: models)
    monkeypatch.setattr(FilterManager, "collect_installed_web_tools_config", no_web_tools)


async def _streamed_turn(pipe, model: str) -> str:
    result = await pipe.pipe(
        body={"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": True},
        __user__={"id": "user-1", "role": "user"},
        __request__=None,
        __event_emitter__=None,
        __event_call__=None,
        __metadata__={"chat_id": "chat-1", "message_id": "message-1", "model": {"id": model}},
        __tools__=None,
    )
    if hasattr(result, "__aiter__"):
        return "".join([str(chunk) async for chunk in result])
    return str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_fusion_turn_whose_whole_panel_failed_counts_toward_the_users_refusal(monkeypatch, threshold):
    pipe = Pipe()
    _reach_fusion(monkeypatch, pipe)
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            mock_http.post(_RESPONSES_URL, repeat=True, **_FAILED_UPSTREAM)
            _saved_breaker_settings(pipe, threshold)
            replies.append(await _streamed_turn(pipe, "m1"))
            _saved_breaker_settings(pipe, threshold)
            replies.append(await _streamed_turn(pipe, "openrouter.fusion"))
            posts_before_the_next_request = _posts(mock_http)
            _saved_breaker_settings(pipe, threshold)
            replies.append(await _streamed_turn(pipe, "m1"))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert "Every panel member failed" in replies[1], replies[1][-600:]
    assert posts == posts_before_the_next_request, replies[2]
    assert "Temporarily disabled due to repeated errors" in replies[2], replies[2]


@pytest.mark.asyncio
async def test_a_fusion_turn_where_some_panel_members_answered_clears_the_count(monkeypatch):
    from aioresponses import CallbackResult

    pipe = Pipe()
    _reach_fusion(monkeypatch, pipe)

    def m1_and_one_panel_member_fail(_url, **kwargs):
        model = str((kwargs.get("json") or {}).get("model") or "")
        if model == "m1" or "gemini" in model:
            return CallbackResult(status=500, payload={"error": {"message": "upstream trouble"}})
        return CallbackResult(status=200, body=_sse_answer())

    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            mock_http.post(_RESPONSES_URL, repeat=True, callback=m1_and_one_panel_member_fail)
            _saved_breaker_settings(pipe, 2)
            replies.append(await _streamed_turn(pipe, "m1"))
            _saved_breaker_settings(pipe, 2)
            replies.append(await _streamed_turn(pipe, "openrouter.fusion"))
            posts_before_the_next_request = _posts(mock_http)
            _saved_breaker_settings(pipe, 2)
            replies.append(await _streamed_turn(pipe, "m1"))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert "Every panel member failed" not in replies[1], replies[1][-600:]
    assert "Hello." in replies[1], replies[1][-600:]
    assert posts == posts_before_the_next_request + 1, replies[2]
    assert "Temporarily disabled" not in replies[2], replies[2]


# --- a failure OpenRouter reports inside a 200 response counts like an error reply -------------------------------------


def _sse_events(events: list[dict[str, Any]], *, done: bool = True) -> bytes:
    body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
    return (body + ("data: [DONE]\n\n" if done else "")).encode("utf-8")


_TEXT_DELTA = {"type": "response.output_text.delta", "delta": "Partial answer "}
_RESPONSE_FAILED = {
    "type": "response.failed",
    "response": {"id": "resp-1", "status": "failed", "error": {"code": "server_error", "message": "Provider disconnected unexpectedly"}},
}
_CHAT_PARTIAL = {"id": "cmpl-1", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": "Partial answer "}, "finish_reason": None}]}
_CHAT_ERROR_CHUNK = {
    "id": "cmpl-1",
    "object": "chat.completion.chunk",
    "error": {"code": "server_error", "message": "Provider disconnected unexpectedly"},
    "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}],
}
_REPORTED_INSIDE_A_200: dict[str, tuple[str, bool, dict[str, Any]]] = {
    "responses-stream-error-first": ("responses", True, {"status": 200, "body": _sse_events([_RESPONSE_FAILED])}),
    "responses-stream-error-after-text": ("responses", True, {"status": 200, "body": _sse_events([_TEXT_DELTA, _RESPONSE_FAILED])}),
    "chat-stream-error-first": ("chat_completions", True, {"status": 200, "body": _sse_events([_CHAT_ERROR_CHUNK])}),
    "chat-stream-error-after-text": ("chat_completions", True, {"status": 200, "body": _sse_events([_CHAT_PARTIAL, _CHAT_ERROR_CHUNK])}),
    "responses-body-error": (
        "responses",
        False,
        {"status": 200, "payload": {"id": "resp-1", "status": "failed", "error": {"code": 502, "message": "Provider returned error"}}},
    ),
    "chat-body-error": ("chat_completions", False, {"status": 200, "payload": {"error": {"code": 502, "message": "Provider returned error"}}}),
    "chat-body-choice-error": (
        "chat_completions",
        False,
        {
            "status": 200,
            "payload": {
                "id": "cmpl-1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Partial answer "},
                        "finish_reason": "error",
                        "error": {"code": 502, "message": "Provider returned error"},
                    }
                ],
            },
        },
    ),
    "responses-stream-cut-off-after-text": ("responses", True, {"status": 200, "body": _sse_events([_TEXT_DELTA], done=False)}),
    "chat-stream-cut-off-after-text": ("chat_completions", True, {"status": 200, "body": _sse_events([_CHAT_PARTIAL], done=False)}),
}


def _saved_settings_for(pipe, threshold: int, endpoint: str) -> None:
    _saved_breaker_settings(pipe, threshold)
    pipe.valves = pipe.valves.model_copy(update={"DEFAULT_LLM_ENDPOINT": endpoint})


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
@pytest.mark.parametrize("arm", list(_REPORTED_INSIDE_A_200))
async def test_a_failure_reported_inside_a_200_counts_once_and_the_next_request_is_refused(monkeypatch, arm, threshold):
    endpoint, stream, reply = _REPORTED_INSIDE_A_200[arm]
    url = _RESPONSES_URL if endpoint == "responses" else _CHAT_URL
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    replies: list[str] = []
    try:
        with aioresponses() as mock_http:
            for _ in range(threshold - 1):
                mock_http.post(url, **_FAILED_UPSTREAM)
            mock_http.post(url, **reply)
            mock_http.post(url, repeat=True, **_FAILED_UPSTREAM)
            for _ in range(threshold):
                _saved_settings_for(pipe, threshold, endpoint)
                replies.append(await _chat_turn(pipe, stream=stream))
            recorded = len(pipe._circuit_breaker._breaker_records["user-1"])
            posts_before_the_next_request = _posts(mock_http)
            _saved_settings_for(pipe, threshold, endpoint)
            replies.append(await _chat_turn(pipe, stream=stream))
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert recorded == threshold, replies
    assert posts == posts_before_the_next_request, replies[-1]
    assert "Temporarily disabled due to repeated errors" in replies[-1], replies[-1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shown_before",
    [[_TEXT_DELTA], [{"type": "response.created", "response": {"id": "resp-1", "status": "in_progress"}}]],
    ids=["after-answer-text", "before-anything-visible"],
)
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_response_that_ends_incomplete_is_a_finished_call_and_clears_the_count(monkeypatch, threshold, shown_before):
    incomplete = {
        "type": "response.incomplete",
        "response": {
            "id": "resp-1",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Partial answer "}]}],
            "usage": {"input_tokens": 7, "output_tokens": 11},
        },
    }
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    reply = ""
    try:
        with aioresponses() as mock_http:
            for _ in range(threshold - 1):
                mock_http.post(_RESPONSES_URL, **_FAILED_UPSTREAM)
            mock_http.post(_RESPONSES_URL, status=200, body=_sse_events([*shown_before, incomplete]))
            for _ in range(threshold):
                _saved_breaker_settings(pipe, threshold)
                reply = await _chat_turn(pipe, stream=True)
    finally:
        await pipe.close()

    assert "Response interrupted" not in reply, reply
    assert "Total tokens: 18 (Input: 7, Output: 11)" in reply, reply[-600:]
    assert len(pipe._circuit_breaker._breaker_records["user-1"]) == 0, reply


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_body", [b"", b"data: [DONE]\n\n"], ids=["nothing", "only-the-end-marker"])
@pytest.mark.parametrize("transport", ["responses-streaming", "chat-streaming"])
async def test_a_stream_that_closes_before_sending_anything_is_retried_and_counted_once(pipe_instance_async, transport, empty_body):
    pipe = pipe_instance_async
    key = f"empty-{transport}"
    url = _CHAT_URL if transport.startswith("chat") else _RESPONSES_URL
    session = pipe._create_http_session(pipe.valves)
    try:
        with aioresponses() as mock_http:
            mock_http.post(url, status=200, body=empty_body)
            mock_http.post(url, **_ANSWER_BY_TRANSPORT[transport])
            await _one_call(pipe, session, transport, key)
            posts = _posts(mock_http)
    finally:
        await session.close()

    assert posts == 2
    assert len(pipe._circuit_breaker._breaker_records[key]) == 1


_ENDINGS = {
    "completed": ([_TEXT_DELTA, {"type": "response.completed", "response": {"id": "resp-1", "status": "completed", "output": [], "usage": {}}}], 0),
    "incomplete": (
        [
            _TEXT_DELTA,
            {
                "type": "response.incomplete",
                "response": {"id": "resp-1", "status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}, "output": [], "usage": {}},
            },
        ],
        0,
    ),
    "cut-off": ([_TEXT_DELTA], 1),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", list(_ENDINGS))
async def test_the_responses_transport_counts_a_stream_by_how_it_ends(pipe_instance_async, ending):
    """Counted at the transport, before the request ends and clears the count: a stream that stops before its final event
    is a failed call, and `response.incomplete` (the answer reached a limit) is a finished one."""
    pipe = pipe_instance_async
    events, failures = _ENDINGS[ending]
    key = f"ending-{ending}"
    session = pipe._create_http_session(pipe.valves)
    try:
        with aioresponses() as mock_http:
            mock_http.post(_RESPONSES_URL, status=200, body=_sse_events(events, done=False))
            await _one_call(pipe, session, "responses-streaming", key)
            posts = _posts(mock_http)
    finally:
        await session.close()

    assert posts == 1
    assert len(pipe._circuit_breaker._breaker_records[key]) == failures


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type, still_named",
    [
        pytest.param("authentication", False, id="a-sign-in-failure-pauses-background-work"),
        pytest.param("content_policy_violation", True, id="a-content-block-does-not"),
        pytest.param("refusal", True, id="a-model-refusal-does-not"),
        pytest.param("permission_denied", True, id="a-guardrail-block-does-not"),
    ],
)
async def test_only_a_sign_in_failure_stops_the_pipe_naming_the_next_chats(monkeypatch, error_type, still_named):
    """A prompt a provider declines must not leave that person's chats unnamed for the next minute.

    The pause exists so a dead key stops being retried on every background task: while it holds, title,
    tag and follow-up generation return their canned answers without sending anything. A content filter
    or a model refusing one prompt is not a credential fault, and nothing on screen would explain why
    the next chats are suddenly called "Chat".

    The four arms differ only in the kind OpenRouter reported, and each asserts what the person gets
    (a real title or the fallback) together with whether the request was sent at all, so a pause that
    is recorded but has no effect cannot satisfy them.
    """
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    try:
        with aioresponses() as mock_http:
            mock_http.post(
                "https://openrouter.ai/api/v1/responses",
                status=200,
                payload={"id": "resp-1", "status": "failed",
                         "error": {"code": "server_error", "message": "This request was declined."},
                         "error_type": error_type},
            )
            mock_http.post("https://openrouter.ai/api/v1/responses", **_TITLE_UPSTREAM)
            await _chat_turn(pipe, stream=False)
            title = await _title_task(pipe)
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert ("Pipe title" in title) is still_named, title
    assert posts == (2 if still_named else 1), (posts, title)


_BLOCK_SHAPES = {
    "a-blocked-image-named-by-its-own-code": (
        "responses", 200,
        {"type": "error", "error": {"code": "image_content_policy_violation", "message": "This image was declined."}},
        False,
    ),
    "a-mid-reply-block-with-no-typed-kind": (
        "chat_completions", 200,
        {"id": "gen-1", "object": "chat.completion.chunk", "created": 1, "model": "m1", "provider": "Google",
         "error": {"code": 403, "message": "This request was declined.",
                   "metadata": {"reasons": ["hate"], "flagged_input": "the words that were flagged",
                                "provider_name": "Google"}},
         "choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "error"}]},
        False,
    ),
    "a-block-openrouter-returned-as-a-status": (
        "responses", 403,
        {"error": {"code": 403, "message": "This request was declined.",
                   "metadata": {"reasons": ["hate"], "flagged_input": "the words that were flagged"}}},
        False,
    ),
    "a-sign-in-failure": (
        "responses", 200,
        {"id": "resp-1", "status": "failed", "error": {"code": "server_error", "message": "Invalid credentials"},
         "error_type": "authentication"},
        True,
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", list(_BLOCK_SHAPES))
async def test_a_content_block_never_pauses_background_work_however_it_arrives(monkeypatch, shape):
    """The pause is about credentials, so how a block is delivered must not decide whether it fires.

    A provider can report a content block four ways: under OpenRouter's typed kind, under its own native code,
    as a bare `403` whose metadata carries the moderation reasons, or as a `403` status before the reply
    starts. Only the last of these existed before failures reported inside a reply carried a real status, and
    a predicate that reads the status alone treats all but the typed one as a dead key - taking the person's
    titles, tags and follow-ups with it for a minute.

    Each arm asserts what the person gets from the next background task and whether it was sent at all, so a
    pause that is recorded but harmless would still be caught.
    """
    endpoint, status, body, pauses = _BLOCK_SHAPES[shape]
    url = ("https://openrouter.ai/api/v1/responses" if endpoint == "responses"
           else "https://openrouter.ai/api/v1/chat/completions")
    pipe = Pipe()
    _reach_the_model(monkeypatch, pipe)
    pipe.valves = pipe.valves.model_copy(update={"DEFAULT_LLM_ENDPOINT": endpoint})
    streamed = endpoint == "chat_completions"
    try:
        with aioresponses() as mock_http:
            if streamed:
                mock_http.post(url, status=status, body=(f"data: {json.dumps(body)}\n\ndata: [DONE]\n\n").encode())
            else:
                mock_http.post(url, status=status, payload=body)
            mock_http.post("https://openrouter.ai/api/v1/responses", **_TITLE_UPSTREAM)
            if streamed:
                mock_http.post("https://openrouter.ai/api/v1/chat/completions", **_CHAT_TITLE_UPSTREAM)
            await _chat_turn(pipe, stream=streamed)
            title = await _title_task(pipe)
            posts = _posts(mock_http)
    finally:
        await pipe.close()

    assert ("Pipe title" in title) is not pauses, (shape, title)
    assert posts == (1 if pauses else 2), (shape, posts, title)


# --- the text a raising tool sends the model is Open WebUI's own (T383) -------------------------
#
# Open WebUI stores a failed tool call as `{'error': str(exc)}` and serialises it with
# `json.dumps(..., indent=2, ensure_ascii=False)`, so the model reads
# '{\n  "error": "…"\n}'. The pipe sent `Tool error: {str(exc) or type(exc).__name__}`,
# which is a different shape from the one the person already sees on the card, and the one
# Open WebUI itself produces when a tool raises. The only intended difference is the
# empty-message case, which names the exception class.

_T383_ERROR_TEXT = '{\n  "error": "tool exploded"\n}'
