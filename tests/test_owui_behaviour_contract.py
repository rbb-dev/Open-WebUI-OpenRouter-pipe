"""What Open WebUI must still do for the replay tests to mean anything.

Several suites borrow Open WebUI's own `convert_output_to_messages` and `handle_responses_streaming_event`
and build fixtures around how they behave. They take them from the installed Open WebUI, which the pipe
requires to be at least the version its manifest names. What those suites cannot model is the borrowed
functions changing under them: their fixtures would still be built, still run, and still pass while
describing something Open WebUI no longer does.

This file states the behaviours they rely on and fails, naming the installed version, the day one stops
holding. It reads Open WebUI's source off disk and compiles the functions into a namespace of its own,
because importing the module pulls in Open WebUI's configuration.
"""

from __future__ import annotations

import __future__
import ast
import importlib.metadata
import json
import sysconfig
from pathlib import Path
from typing import Any

import pytest

_WANTED = {
    "misc.py": {
        "convert_output_to_messages",
        "reconcile_tool_pairs",
        "get_content_from_message",
        "get_output_text",
    },
    "middleware.py": {"handle_responses_streaming_event", "deep_merge", "merge_streamed_reasoning_details"},
}
_ONE_PIXEL_PNG = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _installed_version() -> str:
    try:
        return importlib.metadata.version("open-webui")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - open-webui is a test dependency
        return "not installed"


@pytest.fixture(scope="module")
def open_webui() -> dict[str, Any]:
    utils = Path(sysconfig.get_paths()["purelib"]) / "open_webui" / "utils"
    namespace: dict[str, Any] = {"json": json}
    for filename, names in _WANTED.items():
        source = utils / filename
        tree = ast.parse(source.read_text(encoding="utf-8"))
        nodes: list[ast.stmt] = [
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        missing = names - {node.name for node in nodes if isinstance(node, ast.FunctionDef)}
        assert not missing, f"Open WebUI {_installed_version()} no longer defines {sorted(missing)} in {filename}"
        code = compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(source),
            "exec",
            flags=__future__.annotations.compiler_flag,
            dont_inherit=True,
        )
        exec(code, namespace)
    return namespace


def _call(call_id: str, status: str = "completed") -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc-{call_id}",
        "call_id": call_id,
        "name": "lookup",
        "arguments": "{}",
        "status": status,
    }


def _result(call_id: str, *, image: bool) -> dict[str, Any]:
    parts: list[dict[str, Any]] = [{"type": "input_text", "text": "ok"}]
    if image:
        parts.append({"type": "input_image", "image_url": _ONE_PIXEL_PNG})
    return {
        "type": "function_call_output",
        "id": f"fco-{call_id}",
        "call_id": call_id,
        "output": parts,
        "status": "completed",
    }


def test_a_finished_round_is_replayed_as_a_tool_call_and_its_result(open_webui) -> None:
    """Every fixture in the replay suites is a finished round, and expects both messages back."""
    messages = open_webui["convert_output_to_messages"](
        [_call("c0"), _result("c0", image=False)], raw=True, flatten_tool_images=True
    )

    assert [message["role"] for message in messages] == ["assistant", "tool"], (
        f"Open WebUI {_installed_version()} no longer replays a finished tool round as an assistant "
        f"message carrying the call plus a tool message: {messages}"
    )
    assert messages[0]["tool_calls"][0]["id"] == "c0", messages
    assert messages[1]["tool_call_id"] == "c0", messages


def test_a_result_holding_an_image_is_followed_by_a_message_of_open_webuis_own(open_webui) -> None:
    """Open WebUI writes the round's images out as a message of its own, so a rebuilt turn contains a user
    message nobody typed. The turn scan that had to allow for that is gone (ledger 118); this pins the
    Open WebUI behaviour itself, which the pipe still reads back when it rebuilds history."""
    messages = open_webui["convert_output_to_messages"](
        [_call("c0"), _result("c0", image=True)], raw=True, flatten_tool_images=True
    )

    assert [message["role"] for message in messages] == ["assistant", "tool", "user"], (
        f"Open WebUI {_installed_version()} no longer writes the round's images out as a message of its "
        f"own: {[m['role'] for m in messages]}"
    )
    carrier = messages[-1]["content"]
    assert [part["type"] for part in carrier] == ["text", "image_url"], carrier


def test_rounds_run_back_to_back_are_batched_and_the_images_written_out_after_them(open_webui) -> None:
    """Where that message lands decides which round the images belong to in a rebuilt turn.

    A turn's rounds are held together and written out as one assistant message carrying both calls,
    then both results, then the images. Anything that is not a call or a result -- the model saying
    something between rounds -- ends the batch and brings the images forward to that point.
    """
    back_to_back = [_call("c0"), _result("c0", image=True), _call("c1"), _result("c1", image=False)]
    interrupted = [
        _call("c0"),
        _result("c0", image=True),
        {"type": "message", "id": "m", "role": "assistant", "status": "completed",
         "content": [{"type": "output_text", "text": "Let me check one more thing."}]},
        _call("c1"),
        _result("c1", image=False),
    ]

    batched = [m["role"] for m in open_webui["convert_output_to_messages"](
        back_to_back, raw=True, flatten_tool_images=True
    )]
    split = [m["role"] for m in open_webui["convert_output_to_messages"](
        interrupted, raw=True, flatten_tool_images=True
    )]

    assert batched == ["assistant", "tool", "tool", "user"], (
        f"Open WebUI {_installed_version()} no longer batches a turn's rounds with the images last: {batched}"
    )
    assert split == ["assistant", "tool", "user", "assistant", "tool"], (
        f"Open WebUI {_installed_version()} no longer brings the images forward when the model speaks "
        f"between rounds: {split}"
    )


def test_a_round_whose_call_never_finished_is_dropped_with_its_result(open_webui) -> None:
    """Why every fixture writes the finished status Open WebUI settles a call to once its tool has run.

    A round written as still running is discarded whole, so a fixture that models one is not testing a
    lenient case -- it is testing nothing at all, silently.
    """
    roles = [message["role"] for message in open_webui["convert_output_to_messages"](
        [_call("c0", status="in_progress"), _result("c0", image=False)], raw=True, flatten_tool_images=True
    )]

    assert roles == [], (
        f"Open WebUI {_installed_version()} now replays a round whose call never finished; fixtures that "
        f"model one are no longer silently testing nothing: {roles}"
    )


def test_an_item_the_pipe_opens_and_finishes_is_folded_into_the_stored_output(open_webui) -> None:
    """The recall suite drives Open WebUI's own handler with the events the pipe publishes.

    The pipe opens each item before it finishes it, so this is the order that matters.
    """
    output: list[dict[str, Any]] = []
    opened = dict(_call("c0"), status="in_progress", arguments="")

    output, _ = open_webui["handle_responses_streaming_event"](
        {"type": "response.output_item.added", "item": opened}, output
    )
    output, _ = open_webui["handle_responses_streaming_event"](
        {"type": "response.output_item.done", "output_index": 0, "item": _call("c0")}, output
    )

    assert [item.get("type") for item in output] == ["function_call"], (
        f"Open WebUI {_installed_version()} no longer folds a finished output item into the stored "
        f"output: {output}"
    )
    assert output[0]["call_id"] == "c0", output
    assert output[0]["status"] == "completed", (
        f"Open WebUI {_installed_version()} no longer lets the finished item replace the one that was "
        f"opened: {output[0]}"
    )
