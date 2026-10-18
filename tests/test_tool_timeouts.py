"""Tool time limits in Pipeline mode.

Open WebUI's builtin ask_user keeps a browser prompt open for one to four minutes, so the pipe must not cut it at its
ordinary per-call, batch or idle limits. An ask_user call that shares its turn with other calls is refused, the way
Open WebUI refuses it, because the browser can only show one prompt at a time. A tool that times out or raises tells
the model what happened.

The limits involved are minutes long. While a turn runs, every relative timeout asyncio offers is scaled down by
``SCALE`` and the fake tools sleep on the same scale, so the executor's real timers run unmodified, only faster.
"""

from __future__ import annotations

import __future__
import ast
import asyncio
import collections
from collections.abc import AsyncIterator
import contextlib
import functools
import inspect
import json
import math
import re
import sysconfig
import time as _real_time
import typing
from pathlib import Path
from typing import Any

import pytest
from tests.pipe_limits import limits_for
from aiohttp.helpers import TimerContext

import open_webui_openrouter_pipe.core.circuit_breaker as circuit_breaker_module
import open_webui_openrouter_pipe.tools.tool_executor as tool_executor_module
from open_webui_openrouter_pipe import _ToolExecutionContext
from open_webui_openrouter_pipe.plugins.base import PluginBase
from open_webui_openrouter_pipe.plugins.registry import PluginRegistry
import logging
import threading
from open_webui_openrouter_pipe.integrations import image_catalog, video_catalog
from tests._jumping_clock import event_loop_policy  # noqa: F401

# Time here is simulated, never waited out. The loop jumps its own clock to the next scheduled wake-up
# instead of sleeping until it, so a tool that "takes 300 seconds" costs nothing while still finishing
# after one that takes 295. Ordering is exact because the jump is to a timer the loop already holds --
# there is no race to lose. This replaced a scaled real clock whose margins shrank with the scale: at
# 0.01 the file took 156 s, at 0.002 34 s, and at 0.001 it failed two runs in three.
SCALE = 1.0
NEVER = 100_000


class _TimeTravelLoop(asyncio.SelectorEventLoop):
    """An event loop that advances to the next timer rather than waiting for it.

    `_scheduled` is the loop's own heap of pending wake-ups. When nothing is runnable, the earliest of
    those is the only thing that can happen next, so moving the clock there changes what the loop does
    next by exactly nothing -- except that it costs no real time.
    """

    # CPython internals typeshed does not declare. Naming them states what this clock rests on;
    # `test_the_event_loop_internals_the_jumping_clock_rests_on_still_exist` fails if one goes away.
    _ready: Any
    _scheduled: Any

    def __init__(self) -> None:
        super().__init__()
        self._skew = 0.0

    def time(self) -> float:
        return super().time() + self._skew

    def _run_once(self) -> None:
        if not self._ready and self._scheduled:
            gap = self._scheduled[0]._when - self.time()
            if gap > 0:
                self._skew += gap
        super()._run_once()  # pyright: ignore[reportAttributeAccessIssue]


def test_the_event_loop_internals_the_jumping_clock_rests_on_still_exist():
    """A silent rename in CPython would stop the clock jumping without failing anything."""
    loop = _TimeTravelLoop()
    try:
        assert isinstance(loop._ready, collections.deque)
        assert isinstance(loop._scheduled, list)
        loop.call_later(60.0, lambda: None)
        assert isinstance(loop._scheduled[0]._when, float)
        assert callable(asyncio.SelectorEventLoop._run_once)  # pyright: ignore[reportAttributeAccessIssue]
    finally:
        loop.close()


@contextlib.contextmanager
def _scaled_clock(_monkeypatch):
    """Kept so the call sites read unchanged; the loop itself supplies the simulated time."""
    yield


def _builtin_ask_user(tool, exposed: str = "ask_user") -> dict[str, Any]:
    """The registry entry the pipe builds from Open WebUI's builtin ask_user, renamed when names collide."""
    return {
        "tool_id": "builtin:ask_user",
        "type": "builtin",
        "callable": tool,
        "spec": {"name": "ask_user", "parameters": {"type": "object", "properties": {
            "questions": {"type": "array", "items": {"type": "object"}},
            "allow_other": {"type": "boolean"},
            "timeout_ms": {"type": "integer"},
        }, "required": ["questions"]}},
        "origin_source": "owui_registry_tools",
        "origin_name": "ask_user",
        "exposed_name": exposed,
    }


def _declared(tool) -> dict[str, Any]:
    """The arguments Open WebUI's spec for a tool declares: its named parameters, without the `__context__` ones Open
    WebUI binds itself."""
    return {
        name: {}
        for name, parameter in inspect.signature(tool).parameters.items()
        if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY) and not name.startswith("__")
    }


def _entry(tool, *, tool_type: str, name: str) -> dict[str, Any]:
    return {
        "type": tool_type,
        "callable": tool,
        "spec": {"name": name, "parameters": {"type": "object", "properties": _declared(tool)}},
        "origin_source": "owui_registry_tools",
        "origin_name": name,
        "exposed_name": name,
    }


def _ask(timeout_ms: Any) -> str:
    return json.dumps(
        {
            "questions": [
                {
                    "id": "q1",
                    "header": "Pick",
                    "question": "Which one?",
                    "options": [
                        {"label": "A", "description": "the first"},
                        {"label": "B", "description": "the second"},
                    ],
                }
            ],
            "timeout_ms": timeout_ms,
        }
    )


def _call(call_id: str, name: str, arguments: str = "{}") -> dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments}


def _answers_after(seconds: float, trace: list[str]):
    async def tool(**_kwargs):
        trace.append("asked")
        await asyncio.sleep(seconds * SCALE)
        trace.append("answered")
        return json.dumps({"status": "answered", "answers": {"q1": {"label": "A"}}})

    return tool


def _lookup(trace: list[str]):
    async def lookup(**_kwargs):
        trace.append("looked up")
        return "found it"

    return lookup


def _text(output: dict[str, Any]) -> str:
    return str(output.get("output"))


async def _run(pipe, monkeypatch, registry, calls, *, timeout=60.0, batch_timeout=120.0, idle_timeout=None,
               on_complete=None, event_emitter=None, card_carries_the_result=False, drain=0.0):
    """Run one turn's calls through the real queue, workers, batch executor and retry wrapper.

    Returns the outputs by call_id and how many (virtual) seconds the turn took. `drain` keeps the workers for that many
    more (virtual) seconds after the calls return, as the rest of a request does.
    """
    return await _run_with(
        pipe, monkeypatch, registry, calls,
        maxsize=50, workers=5, batch_cap=4,
        timeout=timeout, batch_timeout=batch_timeout, idle_timeout=idle_timeout,
        on_complete=on_complete, event_emitter=event_emitter,
        card_carries_the_result=card_carries_the_result, drain=drain,
    )


async def _run_with(pipe, monkeypatch, registry, calls, *, maxsize=50, workers=5, batch_cap=4, timeout=60.0,
                    batch_timeout=120.0, idle_timeout=None, on_complete=None, event_emitter=None,
                    card_carries_the_result=False, drain=0.0, queue=None):
    """`_run` with the queue's maxsize, the worker count and the batch cap as parameters.

    The enqueue at the end of `_execute_function_calls` blocks on `queue.put`, so a question about
    whether the idle allowance is charged for that wait needs a queue small enough to actually fill
    and few enough workers to be busy. No valve sets the queue's maxsize, so the only way to say so
    is to pass one; every existing caller keeps `_run`'s values.
    """
    context = _ToolExecutionContext(
        queue=queue if queue is not None else asyncio.Queue(maxsize=maxsize),
        per_request_semaphore=asyncio.Semaphore(5),
        global_semaphore=None,
        timeout=timeout,
        batch_timeout=batch_timeout,
        idle_timeout=idle_timeout,
        user_id="user-1",
        event_emitter=event_emitter,
        batch_cap=batch_cap,
    )
    context.on_complete = on_complete
    context.carded_calls = {str(call.get("call_id")) for call in calls} if card_carries_the_result else set()
    executor = pipe._ensure_tool_executor()
    context.workers.extend(asyncio.create_task(executor._tool_worker_loop(context)) for _ in range(workers))
    token = pipe._TOOL_CONTEXT.set(context)
    loop = asyncio.get_running_loop()
    started = loop.time()
    try:
        with _scaled_clock(monkeypatch):
            outputs = await executor._execute_function_calls(calls, registry)
            if drain:
                await asyncio.sleep(drain * SCALE)
    finally:
        pipe._TOOL_CONTEXT.reset(token)
        for worker in context.workers:
            worker.cancel()
        await asyncio.gather(*context.workers, return_exceptions=True)
    return {output["call_id"]: output for output in outputs}, (loop.time() - started) / SCALE


# --- ask_user keeps its prompt window -------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_ms", [90_000, 200_000])
async def test_an_ask_user_answered_while_its_prompt_is_open_reaches_the_model(
    pipe_instance_async, monkeypatch, timeout_ms
):
    trace: list[str] = []
    registry = {"ask_user": _builtin_ask_user(_answers_after(timeout_ms / 1000 - 5, trace))}

    outputs, _ = await _run(pipe_instance_async, monkeypatch, registry, [_call("c1", "ask_user", _ask(timeout_ms))])

    assert trace == ["asked", "answered"]
    assert outputs["c1"]["status"] == "completed"
    assert '"answered"' in _text(outputs["c1"])


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_ms", [90_000, 200_000])
async def test_an_unanswered_ask_user_is_cut_only_after_its_prompt_has_closed(
    pipe_instance_async, monkeypatch, timeout_ms
):
    trace: list[str] = []
    registry = {"ask_user": _builtin_ask_user(_answers_after(NEVER, trace))}

    outputs, elapsed = await _run(
        pipe_instance_async, monkeypatch, registry, [_call("c1", "ask_user", _ask(timeout_ms))]
    )

    prompt_seconds = timeout_ms / 1000
    assert trace == ["asked"]
    assert outputs["c1"]["status"] != "completed"
    # Cut after the browser closed the prompt, and not long after ...
    assert prompt_seconds < elapsed < prompt_seconds + 60
    # ... and the model is told which tool timed out and after how long.
    text = _text(outputs["c1"])
    numbers = [int(number) for number in re.findall(r"\d+", text)]
    assert "ask_user" in text
    assert numbers and max(numbers) > prompt_seconds, text


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_ms", [30_000, 90_000.0])
async def test_an_ask_user_timeout_open_webui_rejects_gets_open_webuis_default_prompt_time(
    pipe_instance_async, monkeypatch, timeout_ms
):
    # Open WebUI replaces a timeout it rejects (below 60 s, or not an integer) with 120 s instead of clamping it, and
    # the browser keeps the prompt open that long. A clamp would cut these two before the answer at 115 s.
    trace: list[str] = []
    registry = {"ask_user": _builtin_ask_user(_answers_after(115, trace))}

    outputs, _ = await _run(pipe_instance_async, monkeypatch, registry, [_call("c1", "ask_user", _ask(timeout_ms))])

    assert trace == ["asked", "answered"]
    assert outputs["c1"]["status"] == "completed"


def _open_webui_tool_wrapper():
    """Open WebUI's own `get_async_tool_function_and_apply_extra_params`, compiled from the installed source.

    Importing `open_webui.utils.tools` pulls in Open WebUI's configuration, so only this one function is compiled, in a
    namespace of its own; nothing is added to `sys.modules`.
    """
    source = Path(sysconfig.get_paths()["purelib"]) / "open_webui" / "utils" / "tools.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.AsyncFunctionDef) and item.name == "get_async_tool_function_and_apply_extra_params"
    )
    namespace: dict[str, Any] = {
        "inspect": inspect,
        "partial": functools.partial,
        "update_wrapper": functools.update_wrapper,
        "get_type_hints": typing.get_type_hints,
        "get_args": typing.get_args,
    }
    code = compile(
        ast.Module(body=[node], type_ignores=[]),
        str(source),
        "exec",
        flags=__future__.annotations.compiler_flag,
        dont_inherit=True,
    )
    exec(code, namespace)
    return namespace[node.name]


def _open_webui_builtin_ask_user(opened_for: list[Any]):
    """A function with the builtin ask_user's signature and range check, and a browser that keeps the dialog open for
    the `timeout_ms` it is sent and gets its answer 5 s before the dialog closes."""

    async def ask_user(
        questions: list[dict], allow_other: bool = True, timeout_ms: int = 120_000, __event_call__: Any = None
    ) -> str:
        if isinstance(timeout_ms, bool) or not isinstance(timeout_ms, int) or not 60_000 <= timeout_ms <= 240_000:
            timeout_ms = 120_000
        event = {"type": "request:user_input", "data": {"questions": questions, "timeout_ms": timeout_ms}}
        return json.dumps(await __event_call__(event))

    async def browser(event: dict[str, Any]) -> dict[str, Any]:
        opened_for.append(event["data"]["timeout_ms"])
        await asyncio.sleep((event["data"]["timeout_ms"] / 1000 - 5) * SCALE)
        return {"status": "answered", "answers": {"q1": {"label": "A"}}}

    return ask_user, browser


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_ms", ["150000", 200_000], ids=["text-150000", "number-200000"])
async def test_an_ask_user_timeout_sent_as_text_gets_the_same_time_in_the_browser_and_in_the_pipe(
    pipe_instance_async, monkeypatch, timeout_ms
):
    # Open WebUI's tool wrapper turns a quoted number into an int before the builtin opens its dialog, so the time the
    # pipe waits has to come from the value the browser receives.
    opened_for: list[Any] = []
    ask_user, browser = _open_webui_builtin_ask_user(opened_for)
    tool = await _open_webui_tool_wrapper()(ask_user, {"__event_call__": browser})
    registry = {"ask_user": _builtin_ask_user(tool)}

    outputs, _ = await _run(pipe_instance_async, monkeypatch, registry, [_call("c1", "ask_user", _ask(timeout_ms))])

    assert len(opened_for) == 1, opened_for
    assert outputs["c1"]["status"] == "completed", (opened_for, _text(outputs["c1"]))
    assert '"answered"' in _text(outputs["c1"])


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool_type", "name"), [("function", "ask_user"), ("builtin", "search_web")])
async def test_only_open_webuis_builtin_ask_user_gets_a_prompt_window(
    pipe_instance_async, monkeypatch, tool_type, name
):
    # A user's own tool that happens to be called ask_user, and a different builtin, keep the ordinary 60 s limit
    # even when their arguments carry an ask_user timeout.
    trace: list[str] = []
    registry = {name: _entry(_answers_after(80, trace), tool_type=tool_type, name=name)}

    outputs, elapsed = await _run(pipe_instance_async, monkeypatch, registry, [_call("c1", name, _ask(200_000))])

    assert trace == ["asked"]
    assert outputs["c1"]["status"] != "completed"
    assert elapsed < 80


@pytest.mark.asyncio
async def test_an_ask_user_renamed_for_a_name_collision_keeps_its_prompt_window(pipe_instance_async, monkeypatch):
    trace: list[str] = []
    registry = {"owui__ask_user": _builtin_ask_user(_answers_after(85, trace), exposed="owui__ask_user")}

    outputs, _ = await _run(
        pipe_instance_async, monkeypatch, registry, [_call("c1", "owui__ask_user", _ask(90_000))]
    )

    assert trace == ["asked", "answered"]
    assert outputs["c1"]["status"] == "completed"


@pytest.mark.asyncio
async def test_the_idle_timeout_does_not_cut_an_open_ask_user_prompt(pipe_instance_async, monkeypatch):
    trace: list[str] = []
    registry = {"ask_user": _builtin_ask_user(_answers_after(85, trace))}

    outputs, _ = await _run(
        pipe_instance_async, monkeypatch, registry, [_call("c1", "ask_user", _ask(90_000))], idle_timeout=30.0
    )

    assert trace == ["asked", "answered"]
    assert outputs["c1"]["status"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("timeout_ms", "batch_timeout"), [(60_000, 120.0), (120_000, 600.0)]
)
async def test_an_ask_user_nobody_answered_does_not_count_against_the_tool(
    pipe_instance_async, monkeypatch, timeout_ms, batch_timeout
):
    # Under the old 120 s batch limit a 60 s prompt is ended by the per-call limit, while 120 s and 200 s prompts
    # stretch the batch limit to their window and the batch limit ends the wait. With a roomy batch limit the
    # per-call limit ends a 120 s prompt.
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = 1
    registry = {
        "ask_user": _builtin_ask_user(_answers_after(NEVER, [])),
        "slow": _entry(_answers_after(NEVER, []), tool_type="function", name="slow"),
    }

    await _run(pipe, monkeypatch, registry, [_call("c1", "ask_user", _ask(timeout_ms))], batch_timeout=batch_timeout)
    await _run(pipe, monkeypatch, registry, [_call("c2", "slow")], batch_timeout=batch_timeout)

    assert pipe._circuit_breaker.tool_allows("user-1", "builtin", "ask_user") is True
    # An ordinary tool that times out still counts, so the True above is not a breaker that records nothing.
    assert pipe._circuit_breaker.tool_allows("user-1", "function", "slow") is False


# --- ask_user runs alone ----------------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("wording", ["Error: ask_user must run alone.", "Error: one prompt at a time."])
async def test_ask_user_sharing_its_turn_gets_open_webuis_own_refusal(pipe_instance_async, monkeypatch, wording):
    # Where Open WebUI has the checker, it decides and words the refusal; the pipe passes the words on unchanged.
    def open_webui_checker(tool_calls):
        asked = [call for call in tool_calls if call.get("function", {}).get("name") == "ask_user"]
        return (asked, wording) if asked and len(tool_calls) != 1 else (asked, None)

    assert hasattr(tool_executor_module, "_owui_get_ask_user_tool_calls")
    monkeypatch.setattr(tool_executor_module, "_owui_get_ask_user_tool_calls", open_webui_checker)
    trace: list[str] = []
    registry = {
        "ask_user": _builtin_ask_user(_answers_after(1, trace)),
        "lookup": _entry(_lookup(trace), tool_type="function", name="lookup"),
    }

    outputs, _ = await _run(
        pipe_instance_async, monkeypatch, registry, [_call("c1", "ask_user", _ask(90_000)), _call("c2", "lookup")]
    )

    assert trace == ["looked up"]
    assert _text(outputs["c1"]) == wording


@pytest.mark.asyncio
async def test_a_users_own_tool_named_ask_user_may_share_its_turn(pipe_instance_async, monkeypatch):
    trace: list[str] = []
    registry = {
        "ask_user": _entry(_answers_after(1, trace), tool_type="function", name="ask_user"),
        "lookup": _entry(_lookup(trace), tool_type="function", name="lookup"),
    }

    outputs, _ = await _run(
        pipe_instance_async, monkeypatch, registry, [_call("c1", "ask_user", _ask(90_000)), _call("c2", "lookup")]
    )

    assert sorted(trace) == ["answered", "asked", "looked up"]
    assert outputs["c1"]["status"] == "completed"
    assert outputs["c2"]["status"] == "completed"


@pytest.mark.asyncio
async def test_ask_user_sharing_its_turn_with_another_tool_is_refused_and_the_other_tool_runs(
    pipe_instance_async, monkeypatch
):
    # The bare spelling only. The collision-renamed `owui__ask_user` arm used to sit here beside it and
    # was refused for the same reason, on the same one-line defect: the synthetic list handed to Open WebUI
    # hard-coded the bare literal, so a name Open WebUI would not recognise as the builtin was judged by a
    # rule Open WebUI never applied to it. H2301-2 reversed that, and the arm moved to
    # `tests/test_the_ask_user_only_call_rule_is_open_webui_s_rule.py`, where the differential against the
    # real `get_ask_user_tool_calls` is what states the rule rather than a string written out here.
    trace: list[str] = []
    registry = {
        "ask_user": _builtin_ask_user(_answers_after(1, trace)),
        "lookup": _entry(_lookup(trace), tool_type="function", name="lookup"),
    }

    outputs, _ = await _run(
        pipe_instance_async, monkeypatch, registry, [_call("c1", "ask_user", _ask(90_000)), _call("c2", "lookup")]
    )

    assert trace == ["looked up"]
    assert outputs["c1"]["status"] != "completed"
    assert "ask_user" in _text(outputs["c1"])
    assert outputs["c2"]["status"] == "completed"


@pytest.mark.asyncio
async def test_two_ask_user_calls_in_one_turn_are_both_refused(pipe_instance_async, monkeypatch):
    trace: list[str] = []
    registry = {"ask_user": _builtin_ask_user(_answers_after(1, trace))}

    outputs, _ = await _run(
        pipe_instance_async,
        monkeypatch,
        registry,
        [_call("c1", "ask_user", _ask(90_000)), _call("c2", "ask_user", _ask(90_000))],
    )

    assert trace == []
    for call_id in ("c1", "c2"):
        assert outputs[call_id]["status"] != "completed"
        assert "ask_user" in _text(outputs[call_id])


@pytest.mark.asyncio
async def test_ask_user_alone_in_its_turn_runs(pipe_instance_async, monkeypatch):
    trace: list[str] = []
    registry = {
        "ask_user": _builtin_ask_user(_answers_after(1, trace)),
        "lookup": _entry(_lookup(trace), tool_type="function", name="lookup"),
    }

    outputs, _ = await _run(pipe_instance_async, monkeypatch, registry, [_call("c1", "ask_user", _ask(90_000))])

    assert trace == ["asked", "answered"]
    assert outputs["c1"]["status"] == "completed"


# --- what a failed tool tells the model -------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("seconds", "name"), [(7, "slow_lookup"), (13, "slow_render")])
async def test_a_timed_out_tool_tells_the_model_which_tool_and_how_long(
    pipe_instance_async, monkeypatch, seconds, name
):
    registry = {name: _entry(_answers_after(NEVER, []), tool_type="function", name=name)}

    outputs, _ = await _run(
        pipe_instance_async,
        monkeypatch,
        registry,
        [_call("c1", name)],
        timeout=float(seconds),
        batch_timeout=float(seconds * 10),
    )

    text = _text(outputs["c1"])
    assert name in text
    assert re.search(rf"(?<!\d){seconds}(?!\d)", text), text


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [KeyError, RuntimeError])
async def test_a_tool_exception_without_a_message_is_reported_by_its_type(pipe_instance_async, monkeypatch, error):
    async def broken(**_kwargs):
        raise error()

    registry = {"broken": _entry(broken, tool_type="function", name="broken")}

    outputs, _ = await _run(pipe_instance_async, monkeypatch, registry, [_call("c1", "broken")])

    assert error.__name__ in _text(outputs["c1"])


@pytest.mark.asyncio
@pytest.mark.parametrize(("error", "message"), [(RuntimeError, "tool exploded"), (ValueError, "bad input 42")])
async def test_a_tool_exception_with_a_message_keeps_it(pipe_instance_async, monkeypatch, error, message):
    async def broken(**_kwargs):
        raise error(message)

    registry = {"broken": _entry(broken, tool_type="function", name="broken")}

    outputs, _ = await _run(pipe_instance_async, monkeypatch, registry, [_call("c1", "broken")])

    assert message in _text(outputs["c1"])


@pytest.mark.asyncio
async def test_a_tool_that_raises_its_own_timeout_error_is_not_reported_as_cut_off(pipe_instance_async, monkeypatch):
    # A timeout inside the tool (its own HTTP call, say) is the tool's error, not the pipe's limit running out.
    async def times_out_inside(**_kwargs):
        raise TimeoutError()

    registry = {"fetch": _entry(times_out_inside, tool_type="function", name="fetch")}

    outputs, _ = await _run(pipe_instance_async, monkeypatch, registry, [_call("c1", "fetch")], timeout=45.0)

    text = _text(outputs["c1"])
    assert "TimeoutError" in text
    assert "45" not in text


def _times_out_inside(how: str):
    """A tool whose own timer runs out long before the pipe's limit; asyncio and aiohttp both raise the resulting
    TimeoutError chained to the cancellation, exactly as the pipe's own timer does."""

    async def fetch(**_kwargs):
        if how == "asyncio.wait_for":
            await asyncio.wait_for(asyncio.sleep(NEVER), 1)
        elif how == "asyncio.timeout":
            async with asyncio.timeout(1):
                await asyncio.sleep(NEVER)
        else:
            loop = asyncio.get_running_loop()
            timer = TimerContext(loop)
            loop.call_later(0.005, timer.timeout)
            with timer:
                await asyncio.sleep(NEVER)
        return "unreachable"

    return fetch


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [45.0])
@pytest.mark.parametrize("how", ["asyncio.wait_for", "aiohttp timer"])
async def test_a_tool_whose_own_timer_runs_out_is_not_reported_as_cut_off_by_the_pipe(
    pipe_instance_async, monkeypatch, how, limit
):
    registry = {"fetch": _entry(_times_out_inside(how), tool_type="function", name="fetch")}

    outputs, _ = await _run(
        pipe_instance_async, monkeypatch, registry, [_call("c1", "fetch")], timeout=limit, batch_timeout=limit * 10
    )

    text = _text(outputs["c1"])
    assert "TimeoutError" in text
    assert not re.search(rf"(?<!\d){limit:.0f}(?!\d)", text), text


class _StoppedByItsLibrary(BaseException):
    """What some libraries raise instead of an Exception (gevent's Timeout, for one)."""


def _fails_past_the_retry_wrapper(how: str):
    """A tool whose failure is not an Exception, so it passes the retry wrapper and reaches the batch's results."""

    async def fetch(**_kwargs):
        if how == "cancelled future":
            future = asyncio.get_running_loop().create_future()
            future.cancel()
            return await future
        raise _StoppedByItsLibrary("stopped by the library")

    return fetch


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("how", "reported"),
    [("cancelled future", "CancelledError"), ("library error with a message", "stopped by the library")],
)
async def test_a_tool_failure_that_is_not_an_exception_tells_the_model_what_happened(
    pipe_instance_async, monkeypatch, how, reported
):
    # A tool that awaited something another caller cancelled raises CancelledError, and by the time the batch collects
    # it the error carries no message.
    registry = {"fetch": _entry(_fails_past_the_retry_wrapper(how), tool_type="function", name="fetch")}

    outputs, _ = await _run(pipe_instance_async, monkeypatch, registry, [_call("c1", "fetch")])

    assert reported in _text(outputs["c1"])


def _takes(seconds: float, name: str, trace: list):
    async def tool(**_kwargs):
        await asyncio.sleep(seconds * SCALE)
        trace.append(name)
        return f"{name} done"

    return tool


@pytest.mark.asyncio
async def test_a_tool_card_is_emitted_when_its_call_finishes_before_the_calls_behind_it_start(
    pipe_instance_async, monkeypatch
):
    # Restated on WHICH CALLS HAD FINISHED rather than on the simulated second they finished at, because the
    # clock this loop answers `time()` with is real monotonic time plus its own skew: a loop thread that waits
    # a hundredth of a second to be scheduled is charged for the wait as though the tool had taken that long,
    # and this row is run on a machine shared with about seventy other agents.
    order = ("lookup", "render")
    trace: list[str] = []
    registry = {
        "lookup": _entry(_takes(5, "lookup", trace), tool_type="function", name="lookup"),
        "render": _entry(_takes(250, "render", trace), tool_type="function", name="render"),
    }
    cards: dict[str, list[str]] = {}

    async def on_complete(call, _result):
        cards[str(call.get("name"))] = list(trace)

    await _run(
        pipe_instance_async, monkeypatch, registry,
        [_call(f"c{i}", name) for i, name in enumerate(order)],
        timeout=600.0, batch_timeout=900.0, on_complete=on_complete,
    )

    assert trace == ["lookup", "render"], trace
    assert cards["lookup"] == ["lookup"], cards
    assert cards["render"] == ["lookup", "render"], cards


# --- the default limits leave room for long tools --------------------------------------------------------------------


def _works_for(seconds: float, trace: list[str]):
    async def tool(**_kwargs):
        trace.append("started")
        await asyncio.sleep(seconds * SCALE)
        trace.append("finished")
        return "done"

    return tool


async def _run_job(pipe, monkeypatch, registry, calls, *, model_seconds: float = 0, on_complete=None,
                   **valve_changes):
    """Run one turn inside `_execute_pipe_job`, so the tool limits come from the valves exactly as a request gets them.

    Only the call to the model is replaced: it takes ``model_seconds`` to answer, then runs ``calls`` through the tool
    context the job built.
    """
    from open_webui_openrouter_pipe import _PipeJob

    valves = pipe.Valves(**valve_changes)
    await pipe._ensure_concurrency_controls(valves)
    seen: dict[str, Any] = {}

    async def call_the_model_and_run_its_tools(*_args, **_kwargs):
        await asyncio.sleep(model_seconds * SCALE)
        if on_complete is not None:
            pipe._TOOL_CONTEXT.get().on_complete = on_complete
        seen["outputs"] = await pipe._ensure_tool_executor()._execute_function_calls(calls, registry)
        return "turn finished"

    monkeypatch.setattr(pipe, "_handle_pipe_call", call_the_model_and_run_its_tools)
    job = _PipeJob(
        pipe=pipe,
        body={},
        user={"id": "user-1"},
        request=None,
        event_emitter=None,
        event_call=None,
        metadata={},
        tools=None,
        task=None,
        task_body=None,
        future=asyncio.get_running_loop().create_future(),
        valves=valves,
    )
    with _scaled_clock(monkeypatch):
        await pipe._execute_pipe_job(job)
    assert job.future.result() == "turn finished"
    return {output["call_id"]: output for output in seen["outputs"]}


@pytest.mark.asyncio
async def test_default_limits_let_a_four_minute_tool_finish(pipe_instance_async, monkeypatch):
    trace: list[str] = []
    registry = {"render": _entry(_works_for(240, trace), tool_type="function", name="render")}

    outputs = await _run_job(pipe_instance_async, monkeypatch, registry, [_call("c1", "render")])

    assert trace == ["started", "finished"]
    assert outputs["c1"]["status"] == "completed"


@pytest.mark.asyncio
async def test_default_limits_let_two_long_calls_in_one_round_finish_one_after_the_other(
    pipe_instance_async, monkeypatch
):
    trace: list[str] = []
    registry = {"render": _entry(_works_for(200, trace), tool_type="function", name="render")}

    outputs = await _run_job(pipe_instance_async, monkeypatch, registry, [_call("c1", "render"), _call("c2", "render")])

    assert trace == ["started", "finished", "started", "finished"]
    assert outputs["c1"]["status"] == "completed"
    assert outputs["c2"]["status"] == "completed"


# --- a set idle limit only caps the wait for a result ----------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["plain tool", "ask_user"])
async def test_a_model_that_answers_after_the_idle_limit_still_gets_its_tools_run(pipe_instance_async, monkeypatch, tool):
    # The tool workers start before the model is called: a model that takes 40 s to ask for a tool must still get it
    # run under a 30 s idle limit.
    trace: list[str] = []
    if tool == "plain tool":
        registry = {"lookup": _entry(_lookup(trace), tool_type="function", name="lookup")}
        calls, expected = [_call("c1", "lookup")], ["looked up"]
    else:
        registry = {"ask_user": _builtin_ask_user(_answers_after(60, trace))}
        calls, expected = [_call("c1", "ask_user", _ask(90_000))], ["asked", "answered"]

    outputs = await _run_job(
        pipe_instance_async, monkeypatch, registry, calls, model_seconds=40, TOOL_IDLE_TIMEOUT_SECONDS=30
    )

    assert trace == expected
    assert outputs["c1"]["status"] == "completed", _text(outputs["c1"])


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_seconds", [20, 30])
async def test_a_set_idle_limit_still_ends_the_wait_for_a_tool_that_never_returns(
    pipe_instance_async, monkeypatch, idle_seconds
):
    registry = {"hangs": _entry(_answers_after(NEVER, []), tool_type="function", name="hangs")}

    outputs = await _run_job(
        pipe_instance_async, monkeypatch, registry, [_call("c1", "hangs")], TOOL_IDLE_TIMEOUT_SECONDS=idle_seconds
    )

    text = _text(outputs["c1"])
    assert outputs["c1"]["status"] != "completed"
    assert "idle" in text and re.search(rf"(?<!\d){idle_seconds}(?!\d)", text), text


async def _run_job_rounds(pipe, monkeypatch, registry, rounds, **valve_changes):
    """Run several model rounds inside `_execute_pipe_job`.

    Each round is ``(seconds the model thinks, calls, seconds it keeps writing afterwards)``. Returns every round's
    outputs by call_id.
    """
    from open_webui_openrouter_pipe import _PipeJob

    valves = pipe.Valves(**valve_changes)
    await pipe._ensure_concurrency_controls(valves)
    outputs: dict[str, dict[str, Any]] = {}

    async def the_model_works_round_after_round(*_args, **_kwargs):
        for think_seconds, calls, write_seconds in rounds:
            await asyncio.sleep(think_seconds * SCALE)
            for output in await pipe._ensure_tool_executor()._execute_function_calls(calls, registry):
                outputs[output["call_id"]] = output
            await asyncio.sleep(write_seconds * SCALE)
        return "turn finished"

    monkeypatch.setattr(pipe, "_handle_pipe_call", the_model_works_round_after_round)
    job = _PipeJob(
        pipe=pipe,
        body={},
        user={"id": "user-1"},
        request=None,
        event_emitter=None,
        event_call=None,
        metadata={},
        tools=None,
        task=None,
        task_body=None,
        future=asyncio.get_running_loop().create_future(),
        valves=valves,
    )
    with _scaled_clock(monkeypatch):
        await pipe._execute_pipe_job(job)
    assert job.future.result() == "turn finished"
    return outputs


def _sends_email(trace: list[str]):
    async def send_email(**_kwargs):
        trace.append("email sent")
        return "sent"

    return send_email


@pytest.mark.asyncio
@pytest.mark.parametrize("slots", [1, 2])
async def test_a_call_waiting_for_a_slot_past_the_idle_limit_still_runs_and_is_answered(
    pipe_instance_async, monkeypatch, slots
):
    trace: list[str] = []
    registry = {
        "hangs": _entry(_answers_after(NEVER, []), tool_type="function", name="hangs"),
        "send_email": _entry(_sends_email(trace), tool_type="function", name="send_email"),
    }
    rounds = [(5, [_call(f"h{index}", "hangs")], 0) for index in range(slots)]
    rounds += [(5, [_call("e1", "send_email")], 0), (5, [_call("e2", "send_email")], 400)]

    outputs = await _run_job_rounds(
        pipe_instance_async,
        monkeypatch,
        registry,
        rounds,
        MAX_PARALLEL_TOOLS_PER_REQUEST=slots,
        TOOL_IDLE_TIMEOUT_SECONDS=30,
    )

    assert outputs["e1"]["status"] == "completed", _text(outputs["e1"])
    assert outputs["e2"]["status"] == "completed", _text(outputs["e2"])
    assert trace == ["email sent", "email sent"]


@pytest.mark.asyncio
@pytest.mark.parametrize("work_seconds", [40, 55])
async def test_a_call_already_running_when_the_idle_limit_gives_up_finishes_once(
    pipe_instance_async, monkeypatch, work_seconds
):
    trace: list[str] = []
    registry = {"render": _entry(_works_for(work_seconds, trace), tool_type="function", name="render")}

    outputs = await _run_job_rounds(
        pipe_instance_async, monkeypatch, registry, [(5, [_call("c1", "render")], 60)], TOOL_IDLE_TIMEOUT_SECONDS=30
    )

    assert "idle timeout" in _text(outputs["c1"]), _text(outputs["c1"])
    assert trace == ["started", "finished"]


@pytest.mark.asyncio
@pytest.mark.parametrize("slots", [1, 2])
async def test_an_ask_user_waiting_for_a_worker_past_the_idle_limit_still_opens_its_question(
    pipe_instance_async, monkeypatch, slots
):
    asked: list[str] = []
    registry = {
        "hangs": _entry(_answers_after(NEVER, []), tool_type="function", name="hangs"),
        "ask_user": _builtin_ask_user(_answers_after(5, asked)),
    }
    rounds = [(5, [_call(f"h{index}", "hangs")], 0) for index in range(slots)]
    rounds += [(5, [_call("a1", "ask_user", _ask(90_000))], 400)]

    outputs = await _run_job_rounds(
        pipe_instance_async,
        monkeypatch,
        registry,
        rounds,
        MAX_PARALLEL_TOOLS_PER_REQUEST=slots,
        TOOL_IDLE_TIMEOUT_SECONDS=30,
    )

    assert outputs["a1"]["status"] == "completed", _text(outputs["a1"])
    assert asked == ["asked", "answered"]


# --- a reply's calls run one after another --------------------------------------------------------------------------


def _records_when_it_ran(ran: list[tuple[str, float, float]]):
    async def tool(label: str = "", seconds: float = 0, **_kwargs):
        began = asyncio.get_running_loop().time() / SCALE
        await asyncio.sleep(seconds * SCALE)
        ran.append((label, began, asyncio.get_running_loop().time() / SCALE))
        return f"{label} done"

    return tool


def _timed_call(label: str, name: str, seconds: float) -> dict[str, Any]:
    return _call(label, name, json.dumps({"label": label, "seconds": seconds}))


def _sub_agent_entry(tool) -> dict[str, Any]:
    return {**_entry(tool, tool_type="builtin", name="delegate_task"), "tool_id": "builtin:delegate_task"}


@pytest.mark.asyncio
@pytest.mark.parametrize("durations", [[250, 200], [250, 30, 30]], ids=["two-calls", "three-calls"])
async def test_a_rounds_calls_run_one_after_another_in_the_order_the_model_asked(
    pipe_instance_async, monkeypatch, durations
):
    ran: list[tuple[str, float, float]] = []
    registry = {"lookup": _entry(_records_when_it_ran(ran), tool_type="function", name="lookup")}
    calls = [_timed_call(f"c{index}", "lookup", seconds) for index, seconds in enumerate(durations, start=1)]

    outputs = await _run_job(pipe_instance_async, monkeypatch, registry, calls)

    assert [output["status"] for output in outputs.values()] == ["completed"] * len(calls)
    assert [label for label, _, _ in ran] == [call["call_id"] for call in calls], ran
    assert all(began >= ended for (_, _, ended), (_, began, _) in zip(ran, ran[1:])), ran


@pytest.mark.asyncio
async def test_delegate_task_calls_run_together_after_the_rounds_other_calls(pipe_instance_async, monkeypatch):
    ran: list[tuple[str, float, float]] = []
    registry = {
        "delegate_task": _sub_agent_entry(_records_when_it_ran(ran)),
        "lookup": _entry(_records_when_it_ran(ran), tool_type="function", name="lookup"),
    }
    calls = [
        _timed_call("first", "delegate_task", 50),
        _timed_call("middle", "lookup", 30),
        _timed_call("last", "delegate_task", 80),
    ]

    outputs = await _run_job(pipe_instance_async, monkeypatch, registry, calls)

    window = {label: (began, ended) for label, began, ended in ran}
    assert list(outputs) == ["first", "middle", "last"]
    assert [output["status"] for output in outputs.values()] == ["completed"] * 3
    assert window["middle"][1] <= min(window["first"][0], window["last"][0]), window
    assert window["first"][0] < window["last"][1] and window["last"][0] < window["first"][1], window


@pytest.mark.asyncio
async def test_a_call_that_waited_behind_earlier_calls_still_gets_its_whole_per_call_time(
    pipe_instance_async, monkeypatch
):
    ran: list[tuple[str, float, float]] = []
    registry = {"lookup": _entry(_records_when_it_ran(ran), tool_type="function", name="lookup")}
    calls = [_timed_call("c1", "lookup", 50), _timed_call("c2", "lookup", 50), _timed_call("c3", "lookup", 70)]

    outputs, elapsed = await _run(
        pipe_instance_async, monkeypatch, registry, calls, timeout=60.0, batch_timeout=120.0
    )

    assert [outputs[call_id]["status"] for call_id in ("c1", "c2")] == ["completed", "completed"]
    assert outputs["c3"]["status"] != "completed", outputs["c3"]
    assert [label for label, _, _ in ran] == ["c1", "c2"], ran
    assert 160 <= elapsed < 170, elapsed


@pytest.mark.asyncio
async def test_the_wait_for_a_result_is_limited_from_each_calls_own_start(pipe_instance_async, monkeypatch):
    ran: list[tuple[str, float, float]] = []
    registry = {"lookup": _entry(_records_when_it_ran(ran), tool_type="function", name="lookup")}
    calls = [_timed_call("c1", "lookup", 50), _timed_call("c2", "lookup", 50), _timed_call("c3", "lookup", 70)]

    outputs, elapsed = await _run(
        pipe_instance_async, monkeypatch, registry, calls, timeout=300.0, batch_timeout=600.0, idle_timeout=60.0
    )

    assert [outputs[call_id]["status"] for call_id in ("c1", "c2")] == ["completed", "completed"]
    assert outputs["c3"]["status"] != "completed", outputs["c3"]
    assert 160 <= elapsed < 170, elapsed


@pytest.mark.asyncio
@pytest.mark.parametrize("slots", [3, 1])
async def test_inside_a_fusion_answer_each_models_calls_run_one_after_another_while_the_models_run_side_by_side(
    pipe_instance_async, monkeypatch, slots
):
    from open_webui_openrouter_pipe.requests.fusion_engine import FusionInnerInvocation, run_fusion_member

    pipe = pipe_instance_async
    ran: list[tuple[str, float, float]] = []
    registry = {"lookup": _entry(_records_when_it_ran(ran), tool_type="function", name="lookup")}

    class _ModelCallingTwice:
        async def process_request(self, inner_body: dict[str, Any], *_args: Any, **kwargs: Any) -> str:
            model = inner_body["model"]
            await pipe._ensure_tool_executor()._execute_function_calls(
                [_timed_call(f"{model}-1", "lookup", 100), _timed_call(f"{model}-2", "lookup", 100)], registry
            )
            kwargs["outcome_sink"]["error_occurred"] = False
            return "an answer"

    outer = _ToolExecutionContext(
        queue=asyncio.Queue(maxsize=50),
        per_request_semaphore=asyncio.Semaphore(slots),
        global_semaphore=None,
        timeout=300.0,
        batch_timeout=600.0,
        idle_timeout=None,
        user_id="user-1",
        event_emitter=None,
        batch_cap=4,
    )
    invocation = FusionInnerInvocation(
        orchestrator=_ModelCallingTwice(),
        messages=[{"role": "user", "content": "Look everything up."}],
        outer_model_id="openrouter/fusion",
        user={"id": "user-1"},
        request=None,
        event_call=None,
        metadata={"chat_id": "chat-1", "message_id": "message-1"},
        tools=registry,
        valves=pipe.valves.model_copy(update={"MAX_PARALLEL_TOOLS_PER_REQUEST": slots}),
        session=None,
        pipe_identifier="test-pipe",
        user_id="user-1",
    )
    loop = asyncio.get_running_loop()
    token = pipe._TOOL_CONTEXT.set(outer)
    try:
        with _scaled_clock(monkeypatch):
            started = loop.time()
            results = await asyncio.gather(
                *(
                    run_fusion_member(
                        pipe,
                        invocation,
                        model=model,
                        messages=invocation.messages,
                        system_prompt="PANEL PROMPT",
                        max_tool_calls=8,
                        live_queue=None,
                    )
                    for model in ("panel/a", "panel/b")
                )
            )
            elapsed = (loop.time() - started) / SCALE
    finally:
        pipe._TOOL_CONTEXT.reset(token)

    assert [result.failed for result in results] == [False, False], results
    for model in ("panel/a", "panel/b"):
        mine = sorted((row for row in ran if row[0].startswith(model)), key=lambda row: row[1])
        assert [label for label, _, _ in mine] == [f"{model}-1", f"{model}-2"], ran
        assert mine[1][1] >= mine[0][2], mine
    assert len(ran) == 4, ran
    assert (elapsed < 250) if slots == 3 else (elapsed >= 400), elapsed


# --- ask_user does not wait for a tool slot --------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("slot_freed_at", [530, 560])
async def test_an_ask_user_opens_at_once_while_another_request_holds_every_tool_slot(
    pipe_instance_async, monkeypatch, slot_freed_at
):
    # Another request's tool holds the only process-wide tool slot for most of the ten-minute batch limit. ask_user
    # waits on a person, not on work, so its prompt must open at once and an answer 85 s into a 90 s prompt must arrive.
    pipe = pipe_instance_async
    held = asyncio.Semaphore(1)
    monkeypatch.setattr(limits_for(type(pipe)), "tool_semaphore", held)
    monkeypatch.setattr(limits_for(type(pipe)), "tool_limit", 1)
    await held.acquire()
    trace: list[str] = []
    registry = {"ask_user": _builtin_ask_user(_answers_after(85, trace))}

    async def the_other_request_finishes():
        await asyncio.sleep(slot_freed_at * SCALE)
        held.release()

    other_request = asyncio.ensure_future(the_other_request_finishes())
    try:
        outputs = await _run_job(
            pipe, monkeypatch, registry, [_call("c1", "ask_user", _ask(90_000))], MAX_PARALLEL_TOOLS_GLOBAL=1
        )
    finally:
        await other_request

    assert trace == ["asked", "answered"]
    assert outputs["c1"]["status"] == "completed", _text(outputs["c1"])


# --- the batch limit counts from each call's own start ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_round_that_takes_longer_than_the_round_limit_still_runs_every_call(
    pipe_instance_async, monkeypatch
):
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = 2
    trace: list[str] = []
    registry = {"render": _entry(_works_for(250, trace), tool_type="function", name="render")}
    calls = [_call(f"c{index}", "render") for index in range(1, 13)]

    outputs = await _run_job(pipe, monkeypatch, registry, calls)

    assert sorted(outputs) == sorted(f"c{index}" for index in range(1, 13))
    assert {output["status"] for output in outputs.values()} == {"completed"}
    assert (trace.count("started"), trace.count("finished")) == (12, 12)
    assert pipe._circuit_breaker.tool_allows("user-1", "function", "render") is True


@pytest.mark.asyncio
async def test_the_round_limit_still_ends_a_started_call_whose_body_ignores_the_per_call_cancellation(
    pipe_instance_async, monkeypatch
):
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = 1
    finish = asyncio.Event()

    async def stubborn(**_kwargs):
        while not finish.is_set():
            try:
                await asyncio.sleep(NEVER * SCALE)
            except asyncio.CancelledError:
                pass

    registry = {"stubborn": _entry(stubborn, tool_type="function", name="stubborn")}
    try:
        outputs, elapsed = await _run(
            pipe, monkeypatch, registry, [_call("c1", "stubborn")], timeout=300.0, batch_timeout=600.0
        )
    finally:
        finish.set()
        for task in list(pipe._abandoned_tool_tasks):
            task.cancel()

    assert "exceeded 600s" in _text(outputs["c1"]), _text(outputs["c1"])
    assert 600 <= elapsed < 620, elapsed
    assert pipe._circuit_breaker.tool_allows("user-1", "function", "stubborn") is False


# --- each call's result is handed back when that call finishes --------------------------------------------------------


def _takes_as_long_as_asked():
    """One tool whose calls take as many seconds as they ask for, or fail the way `_fails_past_the_retry_wrapper`
    does, so one tool serves calls that behave differently."""

    async def fetch(seconds: float = 0, fails: str = "", **_kwargs):
        if fails:
            return await _fails_past_the_retry_wrapper(fails)()
        await asyncio.sleep(seconds * SCALE)
        return f"fetched after {seconds:g}s"

    return fetch


@pytest.mark.asyncio
@pytest.mark.parametrize(("quick", "slow", "idle"), [(5, 80, 30), (10, 120, 45)])
@pytest.mark.parametrize("quick_first", [True], ids=["quick-call-listed-first"])
async def test_a_quick_call_keeps_its_result_and_the_slow_call_after_it_is_reported_timed_out_at_the_wait_limit(
    pipe_instance_async, monkeypatch, quick, slow, idle, quick_first
):
    # The calls run one after another. With a limit on the wait for results, the quick call ends well inside it
    # and keeps its result, and the slow call is reported timed out at the limit.
    registry = {"fetch": _entry(_takes_as_long_as_asked(), tool_type="function", name="fetch")}
    quick_call = _call("quick", "fetch", json.dumps({"seconds": quick}))
    slow_call = _call("slow", "fetch", json.dumps({"seconds": slow}))

    outputs, _ = await _run(
        pipe_instance_async,
        monkeypatch,
        registry,
        [quick_call, slow_call] if quick_first else [slow_call, quick_call],
        timeout=300.0,
        batch_timeout=600.0,
        idle_timeout=idle,
    )

    assert outputs["quick"]["status"] == "completed", outputs
    assert f"fetched after {quick}s" in _text(outputs["quick"]), outputs
    assert f"Tool 'fetch' timed out after {idle}s (idle timeout)." in _text(outputs["slow"]), outputs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("how", "reported"),
    [("cancelled future", "CancelledError"), ("library error with a message", "stopped by the library")],
)
async def test_a_call_that_fails_past_the_retry_wrapper_is_answered_with_its_failure(
    pipe_instance_async, monkeypatch, how, reported
):
    registry = {"fetch": _entry(_takes_as_long_as_asked(), tool_type="function", name="fetch")}
    calls = [_call("failing", "fetch", json.dumps({"fails": how})), _call("slow", "fetch", json.dumps({"seconds": 80}))]

    outputs, _ = await _run(
        pipe_instance_async, monkeypatch, registry, calls, timeout=300.0, batch_timeout=600.0, idle_timeout=30
    )

    assert reported in _text(outputs["failing"]), outputs


class _NotesWhenToolResultsArrive(PluginBase):
    plugin_id = "notes-when-tool-results-arrive"
    plugin_name = "Notes when tool results arrive"

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self.arrivals: list[float] = []

    async def on_tool_result(self, tool_name, status, **kwargs):
        self.arrivals.append(self._loop.time() / SCALE)


@pytest.mark.asyncio
@pytest.mark.parametrize(("quick", "slow"), [(5, 80), (10, 120)])
async def test_plugins_hear_about_each_tool_result_when_that_call_finishes(
    pipe_instance_async, monkeypatch, quick, slow
):
    pipe = pipe_instance_async
    loop = asyncio.get_running_loop()
    plugin = _NotesWhenToolResultsArrive(loop)
    plugins = PluginRegistry()
    plugins._plugins = [plugin]
    plugins._hook_subscribers["on_tool_result"] = [(plugin, 50)]
    pipe._plugin_registry = plugins
    pipe.valves = pipe.valves.model_copy(update={"ENABLE_PLUGIN_SYSTEM": True})
    registry = {"fetch": _entry(_takes_as_long_as_asked(), tool_type="function", name="fetch")}
    calls = [_call("quick", "fetch", json.dumps({"seconds": quick})), _call("slow", "fetch", json.dumps({"seconds": slow}))]
    started = loop.time() / SCALE

    await _run(pipe, monkeypatch, registry, calls, timeout=300.0, batch_timeout=600.0)

    arrivals = [arrival - started for arrival in plugin.arrivals]
    assert len(arrivals) == 2, arrivals
    assert arrivals[0] < slow / 2 <= arrivals[1], arrivals


@pytest.mark.asyncio
@pytest.mark.parametrize(("slow", "idle"), [(80, 30), (120, 45)])
async def test_a_call_still_running_when_its_request_ends_is_stopped_rather_than_left_to_finish(
    pipe_instance_async, monkeypatch, slow, idle
):
    trace: list[str] = []
    registry = {"fetch": _entry(_works_for(slow, trace), tool_type="function", name="fetch")}

    outputs, _ = await _run(
        pipe_instance_async,
        monkeypatch,
        registry,
        [_call("slow", "fetch")],
        timeout=300.0,
        batch_timeout=600.0,
        idle_timeout=idle,
    )
    await asyncio.sleep(slow * SCALE)

    assert f"Tool 'fetch' timed out after {idle}s (idle timeout)." in _text(outputs["slow"]), outputs
    assert trace == ["started"], trace


# --- a tool that keeps timing out ------------------------------------------------------------------------------------


class _VirtualTime:
    """Stands in for the breaker module's `time`, reading the scaled clock the tool timers run on."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def time(self) -> float:
        return self._loop.time() / SCALE

    def __getattr__(self, name: str) -> Any:
        return getattr(_real_time, name)


async def _hanging_rounds(pipe, monkeypatch, rounds: int, trace: list[str]) -> list[dict[str, Any]]:
    """One call per round to a tool that never returns, cut by a 90 s limit, with 5 s between rounds."""
    registry = {"hangs": _entry(_answers_after(NEVER, trace), tool_type="function", name="hangs")}
    outputs = []
    for index in range(rounds):
        result, _ = await _run(pipe, monkeypatch, registry, [_call(f"c{index}", "hangs")], timeout=90.0, batch_timeout=900.0)
        outputs.append(result[f"c{index}"])
        await asyncio.sleep(5 * SCALE)
    return outputs


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_tool_that_keeps_timing_out_is_skipped_once_its_failures_run_in_a_row(
    pipe_instance_async, monkeypatch, threshold
):
    # Each time-out is recorded when its 90 s limit fires, so consecutive failures land 95 s apart: always further
    # apart than the 60 s breaker window.
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = threshold
    pipe._circuit_breaker.window_seconds = 60
    monkeypatch.setattr(circuit_breaker_module, "time", _VirtualTime(asyncio.get_running_loop()))
    trace: list[str] = []

    outputs = await _hanging_rounds(pipe, monkeypatch, threshold + 1, trace)

    assert trace.count("asked") == threshold
    assert "skipped due to repeated failures" in _text(outputs[-1])


def _cancels_itself(trace: list[str]):
    """A tool that awaits something another caller cancelled, so CancelledError escapes without the pipe cutting it."""

    async def fetch(**_kwargs):
        trace.append("ran")
        future = asyncio.get_running_loop().create_future()
        future.cancel()
        return await future

    return fetch


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_tool_that_keeps_cancelling_itself_is_skipped_once_its_failures_run_in_a_row(
    pipe_instance_async, monkeypatch, threshold
):
    # The model is told this call failed, so it has to count against the tool like any other failure. Only the
    # cancellations the pipe itself causes -- a batch deadline, request shutdown -- are exempt.
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = threshold
    pipe._circuit_breaker.window_seconds = 600
    monkeypatch.setattr(circuit_breaker_module, "time", _VirtualTime(asyncio.get_running_loop()))
    trace: list[str] = []
    registry = {"fetch": _entry(_cancels_itself(trace), tool_type="function", name="fetch")}

    outputs, per_round = [], []
    for index in range(threshold + 1):
        before = len(trace)
        result, _ = await _run(pipe, monkeypatch, registry, [_call(f"c{index}", "fetch")])
        per_round.append(len(trace) - before)
        outputs.append(result[f"c{index}"])

    assert per_round[-1] == 0, per_round
    assert all(count > 0 for count in per_round[:-1]), per_round
    assert "skipped due to repeated failures" in _text(outputs[-1]), _text(outputs[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("builtin", "still_allowed"),
    [(True, True), (False, False)],
    ids=["open-webuis-ask_user", "a-users-own-tool-of-the-same-name"],
)
async def test_only_a_real_tool_is_counted_when_it_cancels_itself(
    pipe_instance_async, monkeypatch, builtin, still_allowed
):
    # `ask_user` waits on a person, so the pipe holds nothing against it -- the same exemption its timeout
    # already gets. Every other tool that cancels itself is a failure like any other, including one a user
    # happens to have named `ask_user`.
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = 1
    pipe._circuit_breaker.window_seconds = 600
    monkeypatch.setattr(circuit_breaker_module, "time", _VirtualTime(asyncio.get_running_loop()))
    trace: list[str] = []
    entry = (
        _builtin_ask_user(_cancels_itself(trace))
        if builtin
        else _entry(_cancels_itself(trace), tool_type="function", name="ask_user")
    )

    await _run(pipe, monkeypatch, {"ask_user": entry}, [_call("c1", "ask_user", _ask(90_000))])

    assert trace == ["ran"], trace
    tool_type = "builtin" if builtin else "function"
    assert pipe._circuit_breaker.tool_allows("user-1", tool_type, "ask_user") is still_allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_pause_longer_than_the_window_gives_a_timed_out_tool_back(pipe_instance_async, monkeypatch, threshold):
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = threshold
    pipe._circuit_breaker.window_seconds = 60
    monkeypatch.setattr(circuit_breaker_module, "time", _VirtualTime(asyncio.get_running_loop()))
    trace: list[str] = []

    await _hanging_rounds(pipe, monkeypatch, threshold, trace)
    await asyncio.sleep(61 * SCALE)
    outputs = await _hanging_rounds(pipe, monkeypatch, threshold + 1, trace)

    # The pause cleared the earlier failures: the tool gets a whole new run of them before it is skipped again.
    assert trace.count("asked") == 2 * threshold
    assert "skipped due to repeated failures" in _text(outputs[-1])


# --- inside a Fusion turn --------------------------------------------------------------------------------------------


def _models_keep_calling(tool_name: str, fed_back: list[str], *, think_after_first_result: float = 0):
    """Every model offered ``tool_name`` calls it each round; one not offered it just answers. A model that has just read
    its first tool result thinks for ``think_after_first_result`` virtual seconds before it calls again."""

    async def fake_stream(self, session, request_body, **_kwargs):
        outputs = [
            item for item in request_body.get("input") or [] if isinstance(item, dict) and item.get("type") == "function_call_output"
        ]
        fed_back.extend(str(item.get("output")) for item in outputs)
        offered = [tool.get("name") for tool in request_body.get("tools") or [] if tool.get("type") == "function"]
        if tool_name not in offered:
            yield {"type": "response.output_text.delta", "delta": "an answer"}
            yield {"type": "response.completed", "response": {"output": [], "usage": {}}}
            return
        if len(outputs) == 1:
            await asyncio.sleep(think_after_first_result * SCALE)
        call = {"type": "function_call", "call_id": f"call-{len(outputs)}", "name": tool_name, "arguments": "{}", "status": "completed"}
        yield {"type": "response.output_item.done", "item": call}
        yield {"type": "response.completed", "response": {"output": [call], "usage": {}}}

    return fake_stream


@pytest.mark.asyncio
@pytest.mark.parametrize("think_seconds", [0], ids=["models-answer-at-once"])
@pytest.mark.parametrize("threshold", [1, 12])
async def test_a_fusion_turn_stops_calling_a_tool_that_keeps_timing_out_without_touching_the_users_own_switch(
    monkeypatch, threshold, think_seconds
):
    """One Fusion turn, one tool that never returns, and the count of calls it may still make.

    Every panel member is offered the tool each round and calls it again as soon as it hears the first
    result, so the count is a property of the panel, of `MAX_PARALLEL_TOOLS_PER_REQUEST` and of the
    breaker's failure threshold -- the number of tool-call rounds the run survives is
    `ceil(threshold / min(panel, parallelism))`, not a slack constant. Both the panel and the
    parallelism are therefore pinned and derived here rather than typed: the panel from the plan the
    run itself resolves, the parallelism next to the threshold, and the assertion below is the exact
    count where the run spans one round and a derived ceiling where it spans more. Measured at panel 3
    and parallelism 5 the count is 3, 3, 3, 6, 7, 7, 9, 10, 12, 14 for thresholds 1 to 12, so the
    shipped default of 5 is a seven-dispatch turn and a bare equality would be red there.

    The members call at once, each first call times out at the 90 s limit, and the run's own breaker --
    not the user's switch, which stays off throughout -- then refuses the tool for the rest of the
    answer. That breaker is built with an infinite window, so a count that has been reached can never
    reopen by expiry. The user's own breaker still allows the tool for a separate reason: a Fusion
    turn's failures are recorded on the per-run breaker alone, never on the user's, so the user is
    never charged for them. Measured: the dispatches are sub-millisecond apart, 0.19-0.35 ms on an
    idle box at `SCALE = 1.0`, which is an order of magnitude and not a bound; the count, not the
    interval, is what the assertions below pin. The arms differ in whether a member deliberates before
    its second dispatch, so a window that reopened between rounds could not pass both: at an infinite
    window the pause changes nothing, and at a finite one the count is what moves.
    """
    import open_webui_openrouter_pipe.pipe as pipe_mod
    from open_webui_openrouter_pipe import EncryptedStr, Pipe
    from open_webui_openrouter_pipe.core.fusion_defaults import find_fusion_entry, resolve_fusion_run
    from open_webui_openrouter_pipe.filters.filter_manager import FilterManager
    from open_webui_openrouter_pipe.models.registry import ModelFamily
    from open_webui_openrouter_pipe.requests import fusion_engine as fusion_engine_module

    async def loaded(*_args: Any, **_kwargs: Any) -> None:
        return None

    async def no_web_tools(*_args: Any, **_kwargs: Any) -> None:
        return None

    pipe = Pipe()
    monkeypatch.setattr(pipe, "_maybe_start_startup_checks", lambda: None)
    monkeypatch.setattr(pipe, "_resolve_openrouter_api_key", lambda _valves: ("sk-test-key", None))
    monkeypatch.setattr(pipe._artifact_store, "_ensure_artifact_store", lambda *_a, **_k: None)
    monkeypatch.setattr(pipe_mod.OpenRouterModelRegistry, "ensure_loaded", loaded)
    monkeypatch.setattr(
        pipe_mod.OpenRouterModelRegistry,
        "list_models",
        lambda: [{"id": "openrouter/fusion", "name": "Fusion", "norm_id": "openrouter.fusion"}]
        + [
            {"id": m, "name": m, "norm_id": ModelFamily._norm(m)}
            for m in resolve_fusion_run(find_fusion_entry(None)).panel_models
        ],
    )
    monkeypatch.setattr(FilterManager, "collect_installed_web_tools_config", no_web_tools)
    monkeypatch.setattr(ModelFamily, "supports", classmethod(lambda cls, capability, model: capability == "function_calling"))
    fed_back: list[str] = []
    monkeypatch.setattr(
        Pipe,
        "send_openrouter_streaming_request",
        _models_keep_calling("hangs", fed_back, think_after_first_result=think_seconds),
    )
    monkeypatch.setattr(circuit_breaker_module, "time", _VirtualTime(asyncio.get_running_loop()))
    per_run_breakers: list[Any] = []
    _real_breaker = fusion_engine_module.CircuitBreaker

    def _spy(**kwargs):
        breaker = _real_breaker(**kwargs)
        per_run_breakers.append(breaker)
        return breaker

    monkeypatch.setattr(fusion_engine_module, "CircuitBreaker", _spy)
    invoked: list[int] = []

    async def hangs(**_kwargs):
        invoked.append(1)
        await asyncio.sleep(NEVER * SCALE)

    tools = {
        "hangs": {
            "tool_id": "mcp_dead",
            "type": "mcp",
            "callable": hangs,
            "spec": {"name": "hangs", "description": "a dead MCP tool", "parameters": {"type": "object", "properties": {}}},
        }
    }
    valves = pipe.valves.model_copy(
        update={"BREAKER_MAX_FAILURES": threshold, "TOOL_TIMEOUT_SECONDS": 90, "TOOL_BATCH_TIMEOUT_SECONDS": 900,
                "MAX_PARALLEL_TOOLS_PER_REQUEST": 5}
    )
    valves.API_KEY = EncryptedStr("sk-test-key")
    pipe.valves = valves
    try:
        with _scaled_clock(monkeypatch):
            result = await pipe.pipe(
                body={"model": "openrouter.fusion", "messages": [{"role": "user", "content": "Look it up."}], "stream": True},
                __user__={"id": "u1", "role": "user"},
                __request__=None,
                __event_emitter__=None,
                __event_call__=None,
                __metadata__={"chat_id": "chat-1", "message_id": "message-1", "model": {"id": "openrouter.fusion"}},
                __tools__=tools,
            )
            if isinstance(result, AsyncIterator):
                async for _chunk in result:
                    pass
    finally:
        await pipe.close()

    expected = len(resolve_fusion_run(find_fusion_entry(None)).panel_models)
    per_round = min(expected, valves.MAX_PARALLEL_TOOLS_PER_REQUEST)
    if per_round >= expected and threshold <= expected:
        # One round: every member dispatches exactly once and the run's own breaker closes the tool.
        assert len(invoked) == expected, ("a second dispatch escaped the first round",
                                          len(invoked), expected, threshold)
    else:
        # The count spans ceil(threshold / per_round) rounds, so bound it rather than name it.
        ceiling = expected * ((threshold + per_round - 1) // per_round) + per_round - 1
        assert 0 < len(invoked) <= ceiling, ("over-dispatched", len(invoked), ceiling,
                                             expected, per_round, threshold)
    assert any("repeated" in output for output in fed_back), fed_back
    assert pipe._circuit_breaker.tool_allows("u1", "mcp", "hangs") is True
    # New pin: the user's breaker is not merely unconvinced, it was never charged.
    assert pipe._circuit_breaker._tool_breakers.get("u1", {}) == {}, pipe._circuit_breaker._tool_breakers
    assert per_run_breakers, "no per-run Fusion breaker was built"
    inner = per_run_breakers[-1]
    assert inner.tool_allows("u1", "mcp", "hangs") is False
    assert inner._window_seconds == math.inf, (
        "the per-run Fusion breaker must not reopen by window expiry: with a finite "
        "window the count and the outer breaker stay green, so nothing else in this test pins it"
    )


def _finishes_after(finished: list[str]):
    async def tool(label: str = "", seconds: float = 0, **_kwargs):
        await asyncio.sleep(seconds * SCALE)
        finished.append(label)
        return f"{label} done"

    return tool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "calls",
    [
        [("a1", "delegate_task", 250), ("b1", "lookup", 30), ("c1", "delegate_task", 120)],
        [("x1", "delegate_task", 120), ("y1", "lookup", 30), ("z1", "delegate_task", 250)],
    ],
    ids=["slowest-called-first", "slowest-called-last"],
)
async def test_a_round_hands_back_its_results_in_call_order_however_the_calls_finish(
    pipe_instance_async, monkeypatch, calls
):
    """The order a round's results are handed back in is the order the model asked for them, not the order the
    tools happened to finish.

    This matters beyond the cards: the hand-back is also the moment each result is recorded for replay, so the
    order here becomes the order the next turn sends upstream. Reasoning anchors and skeleton rounds are numbered
    against that sequence. Only `delegate_task` calls run together, so they are what finishes out of the order the
    model asked in.
    """
    recorded: list[str] = []
    finished: list[str] = []
    registry = {
        "delegate_task": _sub_agent_entry(_finishes_after(finished)),
        "lookup": _entry(_finishes_after(finished), tool_type="function", name="lookup"),
    }

    async def _record(call, _output):
        recorded.append(call.get("call_id"))

    outputs = await _run_job(
        pipe_instance_async,
        monkeypatch,
        registry,
        [_call(cid, name, json.dumps({"label": cid, "seconds": seconds})) for cid, name, seconds in calls],
        on_complete=_record,
    )

    asked = [cid for cid, _, _ in calls]
    assert sorted(finished) == sorted(asked), finished
    assert finished != asked, f"the arm must finish out of order to test anything: {finished}"
    assert recorded == asked, recorded
    assert list(outputs) == asked, list(outputs)


# --- the four places the idle limit is described to an operator ---------------------------------------------


# --- Stop during a round counts only what the tools did (round 35, tools F1) ----------------------------------------


# A synchronous tool reaches the pipe wrapped in Open WebUI's async shell, so the pipe sees an async
# function and its body ran on the event loop, freezing every chat the worker serves while the body
# blocked. `Synchronous tools on threads` routes that shell through the pool like a plain function:
# the worker plants a per-call stand-in loop so the body's own asyncio use still works, and the
# body-raised check unwraps the shell before comparing code objects.


@pytest.mark.asyncio
async def test_a_wrapped_synchronous_tool_body_runs_on_a_pool_thread_and_leaves_the_loop_live(
    pipe_instance_async,
):
    loop_ident = threading.get_ident()
    started: threading.Event = threading.Event()
    release: threading.Event = threading.Event()
    seen: dict[str, Any] = {}

    def body(**_kwargs):
        seen["thread"] = threading.get_ident()
        started.set()
        release.wait(2)
        return "ran off the loop"

    shell = await _open_webui_tool_wrapper()(body, {})
    task = asyncio.ensure_future(pipe_instance_async._call_tool_callable(shell, {}))
    for _ in range(20000):
        if started.is_set():
            break
        await asyncio.sleep(0)
    assert started.is_set()
    assert seen["thread"] != loop_ident
    assert not task.done()
    release.set()
    assert await task == "ran off the loop"


@pytest.mark.asyncio
async def test_a_threaded_tool_body_reaches_the_main_loop_only_through_thread_safe_entries(
    pipe_instance_async,
):
    main_ident = threading.get_ident()
    seen: list[tuple[int, str]] = []

    async def delivered():
        seen.append((threading.get_ident(), "delivered"))

    def _run_async(coro):
        """Fire an async event-emitter coroutine from sync code, tolerating both a
        running loop (OWUI) and none."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(coro)
            else:
                loop.run_until_complete(coro)
        except Exception:
            pass

    def body(**_kwargs):
        _run_async(delivered())

    asyncio.get_running_loop().set_debug(True)
    shell = await _open_webui_tool_wrapper()(body, {})
    try:
        await pipe_instance_async._call_tool_callable(shell, {})
    finally:
        asyncio.get_running_loop().set_debug(False)
    for _ in range(1000):
        if seen:
            break
        await asyncio.sleep(0)
    assert seen == [(main_ident, "delivered")]


@pytest.mark.real_clock
@pytest.mark.asyncio
async def test_the_synchronous_tools_switch_places_a_wrapped_body_on_the_pool_or_the_loop(
    pipe_instance_async, monkeypatch
):
    loop_ident = threading.get_ident()
    seen: dict[str, Any] = {}

    def body(**_kwargs):
        seen["thread"] = threading.get_ident()
        return "ok"

    shell = await _open_webui_tool_wrapper()(body, {})
    registry = {"render": _entry(shell, tool_type="function", name="render")}

    await _run_job(pipe_instance_async, monkeypatch, registry, [_call("c1", "render")])
    assert seen["thread"] != loop_ident

    seen.clear()
    await _run_job(
        pipe_instance_async,
        monkeypatch,
        registry,
        [_call("c2", "render")],
        SYNC_TOOLS_ON_THREADS=False,
    )
    assert seen["thread"] == loop_ident


@pytest.mark.asyncio
async def test_a_threaded_tool_body_can_start_a_task_and_ask_for_the_running_loop(pipe_instance_async):
    main_ident = threading.get_ident()
    seen: list[tuple[str, int]] = []

    async def delivered(label: str) -> str:
        seen.append((label, threading.get_ident()))
        return label

    def body(**_kwargs):
        loop = asyncio.get_running_loop()
        started: list[Any] = [asyncio.create_task(delivered("module")), loop.create_task(delivered("loop"))]
        return [future.result(5) for future in started]

    shell = await _open_webui_tool_wrapper()(body, {})
    assert await pipe_instance_async._call_tool_callable(shell, {}) == ["module", "loop"]
    assert seen == [("module", main_ident), ("loop", main_ident)]


@pytest.mark.asyncio
async def test_a_threaded_tool_body_timer_fires_on_the_main_loop_unless_the_body_cancelled_it(
    pipe_instance_async,
):
    main_ident = threading.get_ident()
    fired: list[tuple[str, int]] = []
    body_done = threading.Event()

    def body(**_kwargs):
        try:
            loop = asyncio.get_event_loop()
            loop.call_later(5, lambda: fired.append(("kept", threading.get_ident())))
            loop.call_later(5, lambda: fired.append(("dropped", threading.get_ident()))).cancel()
        finally:
            body_done.set()

    shell = await _open_webui_tool_wrapper()(body, {})
    call = asyncio.ensure_future(pipe_instance_async._call_tool_callable(shell, {}))
    await asyncio.sleep(0)
    assert body_done.wait(5)
    await call
    await asyncio.sleep(10)
    assert fired == [("kept", main_ident)]


async def _tool_context_with_workers(
    pipe, *, per_request, idle_timeout=None, timeout=300.0, workers=8, global_semaphore=None
):
    context = _ToolExecutionContext(
        queue=asyncio.Queue(maxsize=50),
        per_request_semaphore=asyncio.Semaphore(per_request),
        global_semaphore=global_semaphore,
        timeout=timeout,
        batch_timeout=600.0,
        idle_timeout=idle_timeout,
        user_id="user-1",
        event_emitter=None,
        batch_cap=4,
    )
    executor = pipe._ensure_tool_executor()
    context.workers.extend(asyncio.create_task(executor._tool_worker_loop(context)) for _ in range(workers))
    return executor, context


async def _calls_in(pipe, executor, context, registry, calls):
    token = pipe._TOOL_CONTEXT.set(context)
    try:
        return await executor._execute_function_calls(calls, registry)
    finally:
        pipe._TOOL_CONTEXT.reset(token)


async def _stop_the_request_after(pipe, monkeypatch, registry, calls, seconds, **valve_changes):
    from open_webui_openrouter_pipe import _PipeJob

    valves = pipe.Valves(**valve_changes)
    await pipe._ensure_concurrency_controls(valves)

    async def the_model_runs_its_tools(*_args, **_kwargs):
        try:
            await pipe._ensure_tool_executor()._execute_function_calls(calls, registry)
        except asyncio.CancelledError:
            await asyncio.sleep(20 * SCALE)
            raise
        return "turn finished"

    monkeypatch.setattr(pipe, "_handle_pipe_call", the_model_runs_its_tools)
    job = _PipeJob(
        pipe=pipe,
        body={},
        user={"id": "user-1"},
        request=None,
        event_emitter=None,
        event_call=None,
        metadata={},
        tools=None,
        task=None,
        task_body=None,
        future=asyncio.get_running_loop().create_future(),
        valves=valves,
    )
    running = asyncio.ensure_future(pipe._execute_pipe_job(job))
    await asyncio.sleep(seconds * SCALE)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running


@pytest.mark.asyncio
async def test_a_call_waiting_for_a_pool_thread_past_every_limit_still_runs_and_is_answered(pipe_instance_async):
    pipe = pipe_instance_async
    pipe.valves.MAX_PARALLEL_TOOLS_GLOBAL = 1
    holding, release, ran = threading.Event(), threading.Event(), threading.Event()

    def blocker(**_kwargs):
        holding.set()
        release.wait(30)

    def victim(**_kwargs):
        ran.set()
        return "victim ran"

    wrap = _open_webui_tool_wrapper()
    registry = {
        "blocker": _entry(await wrap(blocker, {}), tool_type="function", name="blocker"),
        "victim": _entry(await wrap(victim, {}), tool_type="function", name="victim"),
    }
    executor, context = await _tool_context_with_workers(pipe, per_request=5, idle_timeout=30.0, timeout=40.0)
    try:
        blocker_round = asyncio.ensure_future(_calls_in(pipe, executor, context, registry, [_call("b1", "blocker")]))
        while not holding.is_set():
            await asyncio.sleep(0)
        victim_round = asyncio.ensure_future(_calls_in(pipe, executor, context, registry, [_call("v1", "victim")]))
        await asyncio.sleep(700)
        assert not victim_round.done()
        release.set()
        while not victim_round.done():
            await asyncio.sleep(0)
        outputs = victim_round.result()
        await blocker_round
    finally:
        release.set()
        for worker in context.workers:
            worker.cancel()
        await asyncio.gather(*context.workers, return_exceptions=True)

    assert ran.is_set()
    assert outputs[0]["status"] == "completed", _text(outputs[0])


@pytest.mark.asyncio
async def test_a_call_behind_the_running_one_when_the_request_is_stopped_never_starts_even_while_the_request_unwinds(
    pipe_instance_async, monkeypatch
):
    trace: list[str] = []

    def tool(name: str, seconds: float):
        async def run(**_kwargs):
            trace.append(f"{name} started")
            await asyncio.sleep(seconds * SCALE)
            trace.append(f"{name} finished")
            return name

        return run

    registry = {
        "long": _entry(tool("long", 53), tool_type="function", name="long"),
        "later": _entry(tool("later", 1), tool_type="function", name="later"),
    }

    await _stop_the_request_after(
        pipe_instance_async,
        monkeypatch,
        registry,
        [_call("c1", "long"), _call("c2", "later")],
        50,
    )

    assert trace == ["long started", "long finished"], trace


@pytest.mark.real_clock
@pytest.mark.asyncio
async def test_one_tools_file_whose_calls_hang_leaves_threads_for_other_tools(pipe_instance_async):
    pipe = pipe_instance_async
    pipe.valves.MAX_PARALLEL_TOOLS_GLOBAL = 8
    release = threading.Event()
    entered: list[int] = []

    def hangs(**_kwargs):
        entered.append(1)
        release.wait(30)

    def quick(**_kwargs):
        return "quick done"

    wrap = _open_webui_tool_wrapper()
    hanging = _entry(await wrap(hangs, {}), tool_type="function", name="hangs")
    hanging["tool_id"] = "tools:backend"
    other = _entry(await wrap(quick, {}), tool_type="function", name="quick")
    other["tool_id"] = "tools:other"
    registry = {"hangs": hanging, "quick": other}
    executor, quick_context = await _tool_context_with_workers(pipe, per_request=5)
    hung_contexts = [(await _tool_context_with_workers(pipe, per_request=5))[1] for _ in range(8)]

    hung_rounds = [
        asyncio.ensure_future(_calls_in(pipe, executor, context, registry, [_call(f"h{index}", "hangs")]))
        for index, context in enumerate(hung_contexts)
    ]
    try:
        for _ in range(500):
            if len(entered) >= 6:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        held = len(entered)
        outputs = await asyncio.wait_for(
            _calls_in(pipe, executor, quick_context, registry, [_call("q1", "quick")]), 5
        )
    finally:
        release.set()
        await asyncio.wait(hung_rounds, timeout=10)
        workers = [worker for context in [quick_context, *hung_contexts] for worker in context.workers]
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*hung_rounds, *workers, return_exceptions=True)

    assert held == 6, held
    assert outputs[0]["status"] == "completed", _text(outputs[0])


@pytest.mark.real_clock
@pytest.mark.asyncio
async def test_calls_waiting_for_a_hung_tools_file_hold_no_slot_so_a_healthy_call_still_runs(pipe_instance_async):
    pipe = pipe_instance_async
    pipe.valves.MAX_PARALLEL_TOOLS_GLOBAL = 8
    release = threading.Event()
    entered: list[int] = []

    def hangs(**_kwargs):
        entered.append(1)
        release.wait(30)

    def quick(**_kwargs):
        return "quick done"

    wrap = _open_webui_tool_wrapper()
    hanging = _entry(await wrap(hangs, {}), tool_type="function", name="hangs")
    hanging["tool_id"] = "tools:backend"
    healthy = _entry(await wrap(quick, {}), tool_type="function", name="quick")
    healthy["tool_id"] = "tools:other"
    registry = {"hangs": hanging, "quick": healthy}
    process_slots = asyncio.Semaphore(7)
    executor, healthy_context = await _tool_context_with_workers(pipe, per_request=5, global_semaphore=process_slots)
    hung_contexts = [
        (await _tool_context_with_workers(pipe, per_request=5, global_semaphore=process_slots))[1] for _ in range(10)
    ]

    hung_rounds = [
        asyncio.ensure_future(_calls_in(pipe, executor, context, registry, [_call(f"h{index}", "hangs")]))
        for index, context in enumerate(hung_contexts)
    ]
    try:
        for _ in range(500):
            if len(entered) >= 6:
                break
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        held = len(entered)
        answer = await asyncio.wait_for(_calls_in(pipe, executor, healthy_context, registry, [_call("q1", "quick")]), 5)
    finally:
        release.set()
        await asyncio.wait(hung_rounds, timeout=10)
        workers = [worker for context in [healthy_context, *hung_contexts] for worker in context.workers]
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*hung_rounds, *workers, return_exceptions=True)

    assert held == 6, held
    assert answer[0]["status"] == "completed", _text(answer[0])


@pytest.mark.real_clock
@pytest.mark.asyncio
async def test_a_valve_change_that_replaces_the_pool_keeps_a_tools_file_within_its_share_and_loses_no_waiting_call(
    pipe_instance_async,
):
    pipe = pipe_instance_async
    pipe.valves.MAX_PARALLEL_TOOLS_PER_REQUEST = 5
    pipe.valves.MAX_PARALLEL_TOOLS_GLOBAL = 4
    old_labels = [f"old{i}" for i in range(6)]
    new_labels = [f"new{i}" for i in range(6)]
    gates = {label: threading.Event() for label in old_labels + new_labels}
    lock = threading.Lock()
    entered: list[str] = []
    in_new_pool: set[str] = set()
    replaced = threading.Event()
    peak = [0]

    def hold(label="", **_kwargs):
        with lock:
            entered.append(label)
            if replaced.is_set():
                in_new_pool.add(label)
                peak[0] = max(peak[0], len(in_new_pool))
        gates[label].wait(30)
        with lock:
            in_new_pool.discard(label)
        return label

    def quick(**_kwargs):
        return "quick done"

    wrap = _open_webui_tool_wrapper()
    shared = _entry(await wrap(hold, {}), tool_type="function", name="hold")
    shared["tool_id"] = "tools:shared"
    other = _entry(await wrap(quick, {}), tool_type="function", name="quick")
    other["tool_id"] = "tools:other"
    registry = {"hold": shared, "quick": other}
    executor, other_context = await _tool_context_with_workers(pipe, per_request=5)
    old_contexts = [(await _tool_context_with_workers(pipe, per_request=10, workers=10))[1] for _ in old_labels]
    new_contexts = [(await _tool_context_with_workers(pipe, per_request=10, workers=10))[1] for _ in new_labels]
    contexts = [other_context, *old_contexts, *new_contexts]

    def round_for(context, label):
        return asyncio.ensure_future(
            _calls_in(pipe, executor, context, registry, [_call(label, "hold", json.dumps({"label": label}))])
        )

    old_rounds = [round_for(context, label) for context, label in zip(old_contexts, old_labels)]
    new_rounds: list[asyncio.Future] = []
    try:
        for _ in range(500):
            if len(entered) >= 4:
                break
            await asyncio.sleep(0.01)
        in_old_pool = list(entered)
        assert len(in_old_pool) == 4, in_old_pool
        pipe.valves.MAX_PARALLEL_TOOLS_GLOBAL = 8
        replaced.set()
        new_rounds = [round_for(context, label) for context, label in zip(new_contexts, new_labels)]
        for _ in range(500):
            if len(in_new_pool) >= 6:
                break
            await asyncio.sleep(0.01)
        gates[in_old_pool[0]].set()
        gates[in_old_pool[1]].set()
        await asyncio.sleep(0.15)
        assert peak[0] == 6, peak[0]
        answer = await asyncio.wait_for(_calls_in(pipe, executor, other_context, registry, [_call("q", "quick")]), 5)
        for gate in gates.values():
            gate.set()
        outputs = [output for finished in await asyncio.wait_for(asyncio.gather(*old_rounds, *new_rounds), 10) for output in finished]
    finally:
        for gate in gates.values():
            gate.set()
        rounds = [*old_rounds, *new_rounds]
        await asyncio.wait(rounds, timeout=10)
        workers = [worker for context in contexts for worker in context.workers]
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*rounds, *workers, return_exceptions=True)

    assert answer[0]["status"] == "completed", _text(answer[0])
    assert [output["status"] for output in outputs] == ["completed"] * 12


@pytest.mark.real_clock
@pytest.mark.asyncio
async def test_a_parked_call_still_starts_when_the_call_woken_ahead_of_it_is_withdrawn_before_it_reached_a_thread(
    pipe_instance_async,
):
    pipe = pipe_instance_async
    labels = [f"a{i}" for i in range(6)] + ["w1", "w2"]
    gates = {label: threading.Event() for label in labels}
    entered: list[str] = []

    def hangs(label="", **_kwargs):
        entered.append(label)
        gates[label].wait(30)
        return label

    wrap = _open_webui_tool_wrapper()
    hanging = _entry(await wrap(hangs, {}), tool_type="function", name="hangs")
    hanging["tool_id"] = "tools:backend"
    registry = {"hangs": hanging}
    executor, stopped_context = await _tool_context_with_workers(pipe, per_request=1, workers=1)
    holder_contexts = [(await _tool_context_with_workers(pipe, per_request=5))[1] for _ in range(6)]
    _, waiting_context = await _tool_context_with_workers(pipe, per_request=5)
    contexts = [stopped_context, waiting_context, *holder_contexts]

    def hung_calls(*names):
        return [_call(name, "hangs", json.dumps({"label": name})) for name in names]

    rounds = [
        asyncio.ensure_future(_calls_in(pipe, executor, context, registry, hung_calls(f"a{index}")))
        for index, context in enumerate(holder_contexts)
    ]
    try:
        for _ in range(500):
            if len(entered) >= 6:
                break
            await asyncio.sleep(0.01)
        stopped = asyncio.ensure_future(_calls_in(pipe, executor, stopped_context, registry, hung_calls("w1")))
        await asyncio.sleep(0.05)
        stopped_context.workers[0].cancel()
        await asyncio.gather(stopped_context.workers[0], return_exceptions=True)
        rounds.append(asyncio.ensure_future(_calls_in(pipe, executor, waiting_context, registry, hung_calls("w2"))))
        await asyncio.sleep(0.05)
        gates["a0"].set()
        await asyncio.sleep(0.05)
        stopped.cancel()
        await asyncio.gather(stopped, return_exceptions=True)
        for _ in range(100):
            if "w2" in entered:
                break
            await asyncio.sleep(0.01)
        seen = list(entered)
    finally:
        for gate in gates.values():
            gate.set()
        await asyncio.wait(rounds, timeout=10)
        workers = [worker for context in contexts for worker in context.workers]
        for worker in workers:
            worker.cancel()
        await asyncio.gather(*rounds, *workers, return_exceptions=True)

    assert "w2" in seen, seen
