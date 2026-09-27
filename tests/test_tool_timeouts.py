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
from aiohttp.helpers import TimerContext

import open_webui_openrouter_pipe.core.circuit_breaker as circuit_breaker_module
import open_webui_openrouter_pipe.tools.tool_executor as tool_executor_module
from open_webui_openrouter_pipe import _ToolExecutionContext
from open_webui_openrouter_pipe.plugins.base import PluginBase
from open_webui_openrouter_pipe.plugins.registry import PluginRegistry

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


class _TimeTravelPolicy(asyncio.DefaultEventLoopPolicy):
    def new_event_loop(self):
        return _TimeTravelLoop()


@pytest.fixture
def event_loop_policy(request):
    """The jumping clock, except where a test asks for a real one.

    Two tests deadlock on it: they arrange for a plugin to still be working when the batch deadline
    passes, and with every wake-up collapsed to the same instant the loop reaches a state where no
    timer is left to break the wait. They keep a real loop and shrink their own numbers instead --
    what they check is the order of three limits against each other, not the size of any of them.
    """
    if request.node.get_closest_marker("real_clock"):
        return asyncio.DefaultEventLoopPolicy()
    return _TimeTravelPolicy()


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
    context = _ToolExecutionContext(
        queue=asyncio.Queue(maxsize=50),
        per_request_semaphore=asyncio.Semaphore(5),
        global_semaphore=None,
        timeout=timeout,
        batch_timeout=batch_timeout,
        idle_timeout=idle_timeout,
        user_id="user-1",
        event_emitter=event_emitter,
        batch_cap=4,
    )
    context.on_complete = on_complete
    context.carded_calls = {str(call.get("call_id")) for call in calls} if card_carries_the_result else set()
    executor = pipe._ensure_tool_executor()
    context.workers.extend(asyncio.create_task(executor._tool_worker_loop(context)) for _ in range(5))
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
@pytest.mark.parametrize("timeout_ms", [90_000, 200_000])
async def test_an_ask_user_answered_just_before_a_late_opened_prompt_closes_reaches_the_model(
    pipe_instance_async, monkeypatch, timeout_ms
):
    # The browser starts its countdown when the dialog opens, after the pipe's call has started: here the dialog opened
    # 12 s late and the user answered 2 s before it closed.
    trace: list[str] = []
    registry = {"ask_user": _builtin_ask_user(_answers_after(timeout_ms / 1000 + 10, trace))}

    outputs, _ = await _run(pipe_instance_async, monkeypatch, registry, [_call("c1", "ask_user", _ask(timeout_ms))])

    assert trace == ["asked", "answered"]
    assert outputs["c1"]["status"] == "completed"


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
@pytest.mark.parametrize("timeout_ms", ["150000", "200000", 200_000], ids=["text-150000", "text-200000", "number-200000"])
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
    ("timeout_ms", "batch_timeout"), [(60_000, 120.0), (120_000, 120.0), (200_000, 120.0), (120_000, 600.0)]
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
@pytest.mark.parametrize("exposed", ["ask_user", "owui__ask_user"])
async def test_ask_user_sharing_its_turn_with_another_tool_is_refused_and_the_other_tool_runs(
    pipe_instance_async, monkeypatch, exposed
):
    trace: list[str] = []
    registry = {
        exposed: _builtin_ask_user(_answers_after(1, trace), exposed=exposed),
        "lookup": _entry(_lookup(trace), tool_type="function", name="lookup"),
    }

    outputs, _ = await _run(
        pipe_instance_async, monkeypatch, registry, [_call("c1", exposed, _ask(90_000)), _call("c2", "lookup")]
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
@pytest.mark.parametrize("limit", [45.0, 70.0])
@pytest.mark.parametrize("how", ["asyncio.wait_for", "asyncio.timeout", "aiohttp timer"])
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
@pytest.mark.parametrize(
    ("order", "lookup_card_at"),
    [(("lookup", "render"), 5.0), (("render", "lookup"), 250.0)],
    ids=["fast-call-first", "fast-call-second"],
)
async def test_a_tool_card_waits_only_for_the_calls_before_it_in_the_round(
    pipe_instance_async, monkeypatch, order, lookup_card_at
):
    # A call's card is emitted once that call and every call BEFORE it in the round have finished -- cards
    # follow call order. What it must not do is wait for a slower call that comes after it.
    trace: list[str] = []
    registry = {
        "lookup": _entry(_takes(5, "lookup", trace), tool_type="function", name="lookup"),
        "render": _entry(_takes(250, "render", trace), tool_type="function", name="render"),
    }
    cards: dict[str, float] = {}
    loop = asyncio.get_running_loop()
    started = loop.time()

    async def on_complete(call, _result):
        cards[str(call.get("name"))] = round((loop.time() - started) / SCALE, 1)

    await _run(
        pipe_instance_async, monkeypatch, registry,
        [_call(f"c{i}", name) for i, name in enumerate(order)],
        timeout=600.0, batch_timeout=900.0, on_complete=on_complete,
    )

    assert trace == ["lookup", "render"], trace
    assert cards["lookup"] == lookup_card_at, cards
    assert cards["render"] == 250.0, cards


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
async def test_default_limits_let_two_long_calls_in_one_batch_finish_one_after_the_other(
    pipe_instance_async, monkeypatch
):
    # With one tool running at a time, both calls share a batch and run back to back: 400 s for the whole batch.
    trace: list[str] = []
    registry = {"render": _entry(_works_for(200, trace), tool_type="function", name="render")}

    outputs = await _run_job(
        pipe_instance_async,
        monkeypatch,
        registry,
        [_call("c1", "render"), _call("c2", "render")],
        MAX_PARALLEL_TOOLS_PER_REQUEST=1,
    )

    assert trace == ["started", "finished", "started", "finished"]
    assert outputs["c1"]["status"] == "completed"
    assert outputs["c2"]["status"] == "completed"


@pytest.mark.asyncio
async def test_with_default_limits_a_call_is_cut_at_five_minutes_and_not_before(pipe_instance_async, monkeypatch):
    registry = {
        "shorter": _entry(_works_for(295, []), tool_type="function", name="shorter"),
        "longer": _entry(_works_for(305, []), tool_type="function", name="longer"),
    }

    outputs = await _run_job(
        pipe_instance_async, monkeypatch, registry, [_call("c1", "shorter"), _call("c2", "longer")]
    )

    assert outputs["c1"]["status"] == "completed"
    assert outputs["c2"]["status"] != "completed"
    assert re.search(r"(?<!\d)300(?!\d)", _text(outputs["c2"])), _text(outputs["c2"])


@pytest.mark.asyncio
async def test_with_default_limits_a_batch_is_cut_at_ten_minutes_and_not_before(pipe_instance_async, monkeypatch):
    # With one tool running at a time, a batch's calls run back to back: two 295 s calls take 590 s, three 205 s
    # calls take 615 s.
    fits = await _run_job(
        pipe_instance_async,
        monkeypatch,
        {"render": _entry(_works_for(295, []), tool_type="function", name="render")},
        [_call("c1", "render"), _call("c2", "render")],
        MAX_PARALLEL_TOOLS_PER_REQUEST=1,
    )
    overflows = await _run_job(
        pipe_instance_async,
        monkeypatch,
        {"render": _entry(_works_for(205, []), tool_type="function", name="render")},
        [_call("c1", "render"), _call("c2", "render"), _call("c3", "render")],
        MAX_PARALLEL_TOOLS_PER_REQUEST=1,
    )

    assert [output["status"] for output in fits.values()] == ["completed", "completed"]
    assert any(re.search(r"exceeded 600s", _text(output)) for output in overflows.values()), overflows


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
async def test_a_call_the_idle_limit_gave_up_on_before_it_started_never_runs_later(
    pipe_instance_async, monkeypatch, slots
):
    # Calls that never return hold every slot. Each later send_email is still waiting for a slot when the idle limit
    # tells the model it timed out; once a slot frees, the email must not go out after all.
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

    assert "idle timeout" in _text(outputs["e1"]), _text(outputs["e1"])
    assert "idle timeout" in _text(outputs["e2"]), _text(outputs["e2"])
    assert trace == []


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
async def test_an_ask_user_the_idle_limit_gave_up_on_before_it_started_never_opens_its_question_later(
    pipe_instance_async, monkeypatch, slots
):
    # ask_user takes no tool slot, but it still needs a worker, and calls that never return keep every worker busy until
    # the per-call limit cuts them. The idle limit tells the model the question timed out while it is still waiting; once
    # a worker frees up, the question must not open after all.
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

    assert "idle timeout" in _text(outputs["a1"]), _text(outputs["a1"])
    assert asked == []


# --- a call waits only for a slot ------------------------------------------------------------------------------------


def _records_its_start(name: str, seconds: float, starts: list[float], finishes: list[float]):
    async def tool(**_kwargs):
        starts.append(asyncio.get_running_loop().time() / SCALE)
        await asyncio.sleep(seconds * SCALE)
        finishes.append(asyncio.get_running_loop().time() / SCALE)
        return f"{name} done"

    return tool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "calls",
    [
        [("c1", "render", 250), ("c2", "lookup", 200)],
        [("c1", "render", 250), ("c2", "lookup", 200), ("c3", "fetch", 200)],
        [("c1", "render", 250), ("c2", "lookup", 30), ("c3", "lookup", 30), ("c4", "fetch", 30)],
    ],
    ids=["two-tools", "three-tools", "a-pair-between-two-tools"],
)
async def test_a_slow_call_does_not_hold_up_other_tools_while_slots_are_free(pipe_instance_async, monkeypatch, calls):
    # On default settings there are five slots, so every call in the response starts at once, and the round lasts as long
    # as its slowest call. Times are virtual seconds: a call held behind another tool's slow call would start 250 s later
    # and stretch the round past 400 s.
    starts: list[float] = []
    finishes: list[float] = []
    registry: dict[str, dict[str, Any]] = {}
    for _call_id, name, seconds in calls:
        registry.setdefault(name, _entry(_records_its_start(name, seconds, starts, finishes), tool_type="function", name=name))

    outputs = await _run_job(pipe_instance_async, monkeypatch, registry, [_call(cid, name) for cid, name, _ in calls])

    assert [output["status"] for output in outputs.values()] == ["completed"] * len(calls)
    assert len(starts) == len(calls)
    assert max(starts) - min(starts) < 1, [round(start - min(starts), 2) for start in starts]
    assert max(finishes) - min(starts) < 260, round(max(finishes) - min(starts), 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("slots", [3, 6])
async def test_inside_a_fusion_answer_a_slow_call_does_not_hold_up_other_tools_while_slots_are_free(
    pipe_instance_async, monkeypatch, slots
):
    # A Fusion model shares its request's tool slots. Here one model calls as many different tools as the request has
    # slots, with every slot free, so every call starts at once and the round lasts as long as its slowest call.
    from open_webui_openrouter_pipe.requests.fusion_engine import FusionInnerInvocation, run_fusion_member

    pipe = pipe_instance_async
    starts: list[float] = []
    finishes: list[float] = []
    names = [f"tool{index}" for index in range(slots)]
    registry = {
        name: _entry(_records_its_start(name, 250 if index == 0 else 200, starts, finishes), tool_type="function", name=name)
        for index, name in enumerate(names)
    }
    statuses: list[str] = []

    class _ModelCallingEveryTool:
        async def process_request(self, *_args: Any, **kwargs: Any) -> str:
            outputs = await pipe._ensure_tool_executor()._execute_function_calls(
                [_call(f"c{index}", name) for index, name in enumerate(names)], registry
            )
            statuses.extend(output["status"] for output in outputs)
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
    executor = pipe._ensure_tool_executor()
    outer.workers.extend(asyncio.create_task(executor._tool_worker_loop(outer)) for _ in range(slots))
    invocation = FusionInnerInvocation(
        orchestrator=_ModelCallingEveryTool(),
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
    token = pipe._TOOL_CONTEXT.set(outer)
    try:
        with _scaled_clock(monkeypatch):
            result = await run_fusion_member(
                pipe,
                invocation,
                model="panel/model",
                messages=invocation.messages,
                system_prompt="PANEL PROMPT",
                max_tool_calls=8,
                live_queue=None,
                bypass_restrictions=True,
            )
    finally:
        pipe._TOOL_CONTEXT.reset(token)
        for worker in outer.workers:
            worker.cancel()
        await asyncio.gather(*outer.workers, return_exceptions=True)

    assert result.failed is False, result
    assert statuses == ["completed"] * slots
    assert len(starts) == slots
    assert max(starts) - min(starts) < 1, [round(start - min(starts), 2) for start in starts]
    assert max(finishes) - min(starts) < 260, round(max(finishes) - min(starts), 2)


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
    monkeypatch.setattr(type(pipe), "_tool_global_semaphore", held)
    monkeypatch.setattr(type(pipe), "_tool_global_limit", 1)
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


# --- a batch that runs out of time -----------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", [2, 3])
async def test_a_batch_that_runs_out_of_time_keeps_the_results_of_calls_that_finished(
    pipe_instance_async, monkeypatch, threshold
):
    # Twelve 250 s calls on default settings make three batches of four sharing five slots. The third batch gets two
    # slots at 250 s and its last two at 500 s, so its 600 s limit fires while only those last two are still running.
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = threshold
    trace: list[str] = []
    registry = {"render": _entry(_works_for(250, trace), tool_type="function", name="render")}
    calls = [_call(f"c{index}", "render") for index in range(1, 13)]

    outputs = await _run_job(pipe, monkeypatch, registry, calls)

    completed = sorted(call_id for call_id, output in outputs.items() if output["status"] == "completed")
    cut = sorted(call_id for call_id, output in outputs.items() if "exceeded 600s" in _text(output))
    assert completed == sorted(f"c{index}" for index in range(1, 11))
    assert cut == ["c11", "c12"]
    # The two calls running when the limit fired were stopped, not left to finish their work.
    assert (trace.count("started"), trace.count("finished")) == (12, 10)
    # Only the two calls still running when the limit fired count against the tool.
    assert pipe._circuit_breaker.tool_allows("user-1", "function", "render") is (threshold > 2)


@pytest.mark.asyncio
async def test_a_call_still_waiting_for_its_slot_when_the_batch_limit_fires_does_not_count_against_the_tool(
    pipe_instance_async, monkeypatch
):
    # One slot and four 250 s calls in one batch: c1 and c2 finish, c3 is running when the 600 s limit fires and c4 is
    # still waiting for the slot. Only c3 was running, so a threshold of 2 still allows the tool.
    pipe = pipe_instance_async
    pipe._circuit_breaker.threshold = 2
    trace: list[str] = []
    registry = {"render": _entry(_works_for(250, trace), tool_type="function", name="render")}
    calls = [_call(f"c{index}", "render") for index in range(1, 5)]

    outputs = await _run_job(pipe, monkeypatch, registry, calls, MAX_PARALLEL_TOOLS_PER_REQUEST=1)

    assert sorted(call_id for call_id, output in outputs.items() if output["status"] == "completed") == ["c1", "c2"]
    assert sorted(call_id for call_id, output in outputs.items() if "exceeded 600s" in _text(output)) == ["c3", "c4"]
    assert (trace.count("started"), trace.count("finished")) == (3, 2)
    assert pipe._circuit_breaker.tool_allows("user-1", "function", "render") is True


# --- each call's result is handed back when that call finishes --------------------------------------------------------


def _takes_as_long_as_asked():
    """One tool whose calls take as many seconds as they ask for, or fail the way `_fails_past_the_retry_wrapper`
    does, so calls that behave differently still share a batch."""

    async def fetch(seconds: float = 0, fails: str = "", **_kwargs):
        if fails:
            return await _fails_past_the_retry_wrapper(fails)()
        await asyncio.sleep(seconds * SCALE)
        return f"fetched after {seconds:g}s"

    return fetch


@pytest.mark.asyncio
@pytest.mark.parametrize(("quick", "slow", "idle"), [(5, 80, 30), (10, 120, 45)])
@pytest.mark.parametrize("quick_first", [True, False], ids=["quick-call-listed-first", "slow-call-listed-first"])
async def test_a_quick_call_batched_with_a_slow_one_keeps_its_result_when_the_wait_for_results_is_limited(
    pipe_instance_async, monkeypatch, quick, slow, idle, quick_first
):
    # Both calls share one batch. With a limit on the wait for results, the wait for each call gives up after `idle`
    # seconds, so the quick call keeps its result only if it is handed back when the quick call ends, not when the
    # slow call does.
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
async def test_a_call_that_fails_past_the_retry_wrapper_is_answered_while_its_batch_is_still_running(
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


class _TakesItsTimeOverEachToolResult(PluginBase):
    plugin_id = "takes-its-time-over-each-tool-result"
    plugin_name = "Takes its time over each tool result"

    def __init__(self, seconds: float) -> None:
        self._seconds = seconds

    async def on_tool_result(self, tool_name, status, **kwargs):
        await asyncio.sleep(self._seconds * SCALE)


@pytest.mark.real_clock
@pytest.mark.asyncio
@pytest.mark.parametrize(("first", "second"), [(0.01, 0.02), (0.03, 0.05)])
async def test_a_call_that_finishes_while_a_plugin_is_busy_with_an_earlier_result_keeps_its_own_result(
    pipe_instance_async, monkeypatch, first, second
):
    # The plugin is still handling the first result when the batch limit passes, so the second call finished
    # while nothing was collecting results. It must still be answered with its own result. The three limits are
    # a thousandth of the real ones and keep their order -- plugin 0.7 s outlasts the 0.6 s batch, both inside
    # the 0.9 s idle limit -- because what is under test is which of them fires first.
    pipe = pipe_instance_async
    plugin = _TakesItsTimeOverEachToolResult(0.7)
    plugins = PluginRegistry()
    plugins._plugins = [plugin]
    plugins._hook_subscribers["on_tool_result"] = [(plugin, 50)]
    pipe._plugin_registry = plugins
    pipe.valves = pipe.valves.model_copy(update={"ENABLE_PLUGIN_SYSTEM": True})
    registry = {"fetch": _entry(_takes_as_long_as_asked(), tool_type="function", name="fetch")}
    calls = [
        _call("first", "fetch", json.dumps({"seconds": first})),
        _call("second", "fetch", json.dumps({"seconds": second})),
    ]

    outputs, _ = await _run(pipe, monkeypatch, registry, calls, timeout=3.0, batch_timeout=0.6, idle_timeout=0.9)

    assert outputs["second"]["status"] == "completed", outputs
    assert f"fetched after {second:g}s" in _text(outputs["second"]), outputs

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
@pytest.mark.parametrize("think_seconds", [0, 70], ids=["models-answer-at-once", "models-think-past-the-window"])
@pytest.mark.parametrize("threshold", [1, 2, 3, 4, 5, 6, 7, 8, 10, 12])
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
        lambda: [{"id": "openrouter/fusion", "name": "Fusion", "norm_id": "openrouter.fusion"}],
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_seconds", [90, 5], ids=["every-call-outlives-the-limit", "a-quick-call-listed-first"]
)
async def test_the_wait_for_results_is_limited_once_for_the_round_not_once_per_call(
    pipe_instance_async, monkeypatch, first_seconds
):
    """The limit caps how long a person waits for the round, not how long each call is looked at in turn.

    The calls are answered in order, so waiting for each one separately starts its clock when the previous
    one resolved. Four calls that start together and take the same time then get different answers - the
    one the pipe happens to be waiting on is reported as timed out and the rest as completed - the round
    still takes as long as the slowest call, and the timed-out call's real result, which the tool did
    produce, is thrown away.

    The arms differ in whether the first call is quick, so a fix that merely reorders the waiting passes
    one and fails the other.
    """
    registry = {"fetch": _entry(_takes_as_long_as_asked(), tool_type="function", name="fetch")}
    calls = [_call("c0", "fetch", json.dumps({"seconds": first_seconds}))]
    calls += [_call(f"c{n}", "fetch", json.dumps({"seconds": 90})) for n in (1, 2, 3)]

    outputs, elapsed = await _run(
        pipe_instance_async, monkeypatch, registry, calls, timeout=300.0, batch_timeout=600.0, idle_timeout=60.0
    )

    cut = [call_id for call_id, output in outputs.items() if "idle timeout" in str(output.get("output"))]
    if first_seconds == 90:
        assert sorted(cut) == ["c0", "c1", "c2", "c3"], outputs
    else:
        assert "fetched after 5s" in str(outputs["c0"].get("output")), outputs["c0"]
        assert sorted(cut) == ["c1", "c2", "c3"], outputs
    assert elapsed < 70, (elapsed, outputs)


def _finishes_after(call_id: str, seconds: float, finished: list[str]):
    async def tool(**_kwargs):
        await asyncio.sleep(seconds * SCALE)
        finished.append(call_id)
        return f"{call_id} done"

    return tool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "calls",
    [
        [("a1", 250), ("b1", 30), ("c1", 120)],
        [("x1", 30), ("y1", 250), ("z1", 120)],
    ],
    ids=["slowest-called-first", "slowest-called-second"],
)
async def test_a_round_hands_back_its_results_in_call_order_however_the_calls_finish(
    pipe_instance_async, monkeypatch, calls
):
    """The order a round's results are handed back in is the order the model asked for them, not the order the
    tools happened to finish.

    This matters beyond the cards: the hand-back is also the moment each result is recorded for replay, so the
    order here becomes the order the next turn sends upstream. Reasoning anchors and skeleton rounds are numbered
    against that sequence. Nothing pinned it before, which is why the question of whether the round may be driven
    by completion instead could not be answered by running the replay suites -- they never reach this code.
    """
    recorded: list[str] = []
    finished: list[str] = []
    registry: dict[str, Any] = {}
    for call_id, seconds in calls:
        registry[call_id] = _entry(_finishes_after(call_id, seconds, finished), tool_type="function", name=call_id)

    async def _record(call, _output):
        recorded.append(call.get("call_id"))

    outputs = await _run_job(
        pipe_instance_async, monkeypatch, registry, [_call(cid, cid) for cid, _ in calls], on_complete=_record
    )

    asked = [cid for cid, _ in calls]
    assert sorted(finished) == sorted(asked), finished
    assert finished != asked, f"the arm must finish out of order to test anything: {finished}"
    assert recorded == asked, recorded
    assert list(outputs) == asked, list(outputs)


# --- the four places the idle limit is described to an operator ---------------------------------------------

# Prescription 117 named four surfaces and the wording was fixed on all four, but nothing observed them: round
# 32's verify lens measured each revert SURVIVING the whole suite. The limit is one wait for the whole round,
# counted from the model's request -- not a fresh wait per call, in call order.
IDLE_LIMIT_SURFACES = [
    pytest.param("Field description", "open_webui_openrouter_pipe/core/config.py", id="field"),
    pytest.param("Config tab detail", "open_webui_openrouter_pipe/plugins/pipe_dashboard/config_meta.py",
                 id="config-tab"),
    pytest.param("tooling doc", "docs/tooling_and_integrations.md", id="tooling-doc"),
    pytest.param("valve atlas", "docs/valves_and_configuration_atlas.md", id="atlas"),
]
RETIRED_PER_CALL_PHRASES = ("for each tool call's result", "one by one in call order", "one at a time",
                            "for each tool result, in turn")


@pytest.mark.parametrize(("surface", "path"), IDLE_LIMIT_SURFACES)
def test_every_place_the_idle_limit_is_described_calls_it_one_wait_for_the_round(surface, path):
    import pathlib as _pathlib

    text = _pathlib.Path(path).read_text(encoding="utf-8")
    start = text.index("TOOL_IDLE_TIMEOUT_SECONDS")
    window = text[start : start + 4000]

    assert "in total for one response" in window, f"{surface} no longer calls it one wait for the round"
    for phrase in RETIRED_PER_CALL_PHRASES:
        assert phrase not in window, f"{surface} still promises a per-call wait: {phrase!r}"


# --- Stop during a round counts only what the tools did (round 35, tools F1) ----------------------------------------


async def _wait_for(predicate, *, seconds: float, what: str) -> None:
    """Wait until ``predicate`` holds, up to a wall-clock ``seconds`` budget, and say so by name if it never does.

    The budget is a deadline on the real monotonic clock, not a count of sleeps, so it means the same thing
    however slowly the machine runs the loop. Running the budget out is the helper raising with ``what``
    naming what never happened; the caller asserts the count it expects, so what reaches the report from a
    loaded machine is the wait that missed its target rather than a bare ``assert (3, 0) == (3, 3)`` that
    says a number is wrong without saying which wait gave up.
    """
    deadline = _real_time.monotonic() + seconds
    while not predicate():
        if _real_time.monotonic() >= deadline:
            raise AssertionError(what)
        await asyncio.sleep(0.01)
