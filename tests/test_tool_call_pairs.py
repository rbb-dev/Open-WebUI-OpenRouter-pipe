"""Tool calls and their results must pair up by COUNT on the wire, not by membership.

Call ids are not always unique. Until the pipe made its ids unique, the chat-completions adapters minted
`toolcall-{model}-{index}` with the index restarting each request, so in Open-WebUI tool mode -- where every round is
a separate pipe request -- two rounds of one answer could carry the same id, and chats saved then still do. Open WebUI
appends a `function_call` only for ids it does not already hold but a
`function_call_output` for every result, so its stored output ends up with one call and several results under one
id, and that is what comes back to the pipe.

Comparing sets cannot see that: the id is present on both sides either way. A Pipeline turn, meanwhile,
legitimately carries several genuine call/result pairs on one id, so dropping duplicates by id would delete real
results. Counting tells the two apart.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from open_webui_openrouter_pipe.requests.sanitizer import _validate_tool_call_pairs

_LOG = logging.getLogger(__name__)


def _user(text: str) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _call(call_id: str) -> dict[str, Any]:
    return {"type": "function_call", "call_id": call_id, "name": "lookup", "arguments": "{}"}


def _result(call_id: str, text: str = "ok") -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": call_id,
            "output": [{"type": "input_text", "text": text}], "status": "completed"}


def _shape(items: list[Any]) -> list[tuple[str, str]]:
    return [(str(item.get("type")), str(item.get("call_id"))) for item in items if isinstance(item, dict)]


@pytest.mark.parametrize(
    ("calls", "results", "kept"),
    [(1, 2, 1), (2, 2, 2), (1, 3, 1), (3, 2, 2)],
    ids=["one-call-two-results", "two-rounds-on-one-id", "one-call-three-results", "three-calls-two-results"],
)
def test_an_id_never_carries_more_results_than_it_has_calls(calls, results, kept):
    """Two rounds of one answer sharing an id is a real shape; so is one call answered twice, which is not."""
    items = [_user("q"), *[_call("c0") for _ in range(calls)], *[_result("c0", f"r{i}") for i in range(results)]]

    out = _shape(_validate_tool_call_pairs(list(items), logger=_LOG))

    assert sum(1 for kind, _ in out if kind == "function_call_output") == kept, out
    assert sum(1 for kind, _ in out if kind == "function_call") == calls, out


def test_the_results_that_survive_are_the_earliest_ones():
    """The first result belongs to the first round; a later surplus is Open WebUI's duplicate, not a new answer."""
    items = [_user("q"), _call("c0"), _result("c0", "first"), _result("c0", "second")]

    out = _validate_tool_call_pairs(list(items), logger=_LOG)

    outputs = [item for item in out if isinstance(item, dict) and item.get("type") == "function_call_output"]
    assert len(outputs) == 1, out
    assert outputs[0]["output"][0]["text"] == "first", outputs


def test_a_result_whose_call_is_absent_is_still_dropped_whole():
    items = [_user("q"), _call("c0"), _result("c0"), _result("stray")]

    out = _shape(_validate_tool_call_pairs(list(items), logger=_LOG))

    assert ("function_call_output", "stray") not in out, out
    assert ("function_call_output", "c0") in out, out


def test_results_under_different_ids_are_counted_apart():
    items = [_user("q"), _call("a"), _call("b"), _result("a"), _result("a"), _result("b")]

    out = _shape(_validate_tool_call_pairs(list(items), logger=_LOG))

    assert out.count(("function_call_output", "a")) == 1, out
    assert out.count(("function_call_output", "b")) == 1, out
