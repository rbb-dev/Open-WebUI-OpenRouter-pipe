"""The one place a test reaches the pipe's process-level concurrency slots.

The request, tool and video semaphores used to be class attributes on `Pipe`, so a test
reached them as `Pipe._global_semaphore`. They are process-level now, keyed by pipe id, so
the key is written here once instead of in thirty-nine files: `limits_for` resolves the
holder that serves a pipe id, and `slot` / `set_slot` are the read and write of one named
slot on it.

`limits_for` accepts a `Pipe` instance or a `Pipe` class -- the class carries the same
`id` the instance does, and the tests that never build an instance use the class.
"""

from __future__ import annotations

from typing import Any

from open_webui_openrouter_pipe.pipe import _get_process_limits

REQUEST_SEMAPHORE = "request_semaphore"
REQUEST_LIMIT = "request_limit"
TOOL_SEMAPHORE = "tool_semaphore"
TOOL_LIMIT = "tool_limit"
VIDEO_SEMAPHORE = "video_semaphore"
VIDEO_LIMIT = "video_limit"

REQUEST_AND_TOOL_SLOTS = (REQUEST_SEMAPHORE, REQUEST_LIMIT, TOOL_SEMAPHORE, TOOL_LIMIT)
ALL_SLOTS = (*REQUEST_AND_TOOL_SLOTS, VIDEO_SEMAPHORE, VIDEO_LIMIT)


def limits_for(pipe: Any) -> Any:
    """The process-level slots that serve this pipe (or pipe class)'s id."""
    return _get_process_limits().for_id(getattr(pipe, "id", ""))


def slot(pipe: Any, name: str) -> Any:
    return getattr(limits_for(pipe), name)


def set_slot(pipe: Any, name: str, value: Any) -> None:
    setattr(limits_for(pipe), name, value)


def set_slots(pipe: Any, names: tuple[str, ...], values: tuple[Any, ...]) -> None:
    slots = limits_for(pipe)
    for name, value in zip(names, values):
        setattr(slots, name, value)


def reset_all_slots() -> None:
    """Null every slot the process has ever minted, for every pipe id."""
    holder = _get_process_limits()
    for pipe_id in list(holder._by_pipe_id):
        minted = holder.for_id(pipe_id)
        for name in ALL_SLOTS:
            setattr(minted, name, None if name.endswith("semaphore") else 0)
