"""One shape for the ffmpeg child the frame extractor spawns.

`media.frame_extraction` reads the child's stdout and stderr as two streams, reaps the
child, and kills it on every way out. A fake that only answered `communicate()` models a
process the module no longer talks to, so a change to the read path left every one of
these tests green while the real leg had stopped working; one fake that carries the real
shape moves them together with it.

Two properties of a real `asyncio.subprocess.Process` are load-bearing and modelled
here. `returncode` is `None` until the child has been reaped -- `communicate()` returning
is what used to set it -- so a fake that set it eagerly would model a child asyncio never
produces, and a fix that killed "the child that is still running" would be tested on a
child that was not running. And a killed child closes its pipes, so a read that is
blocked at the moment of the kill returns EOF; a fake whose read stays blocked forever
cannot model the cleanup under test, because the cleanup can never finish.

`hang` is the shape that models a wedged child: the read never returns on its own and the
only thing that ends it is `kill()`. `fault` is the shape that models a transport error
mid-stream, the one exit that leaks a child if nothing stops it.
"""

from __future__ import annotations

import asyncio
from typing import Any

__all__ = ["FfmpegChildStub", "child_exec_factory"]


class _ChildStream:
    """The read side of one of the child's two pipes."""

    def __init__(self, child: "FfmpegChildStub", name: str) -> None:
        self._child = child
        self._name = name
        self._offset = 0

    @property
    def _source(self) -> bytes:
        return (
            self._child.payload if self._name == "stdout" else self._child.stderr_bytes
        )

    async def read(self, size: int = -1) -> bytes:
        child = self._child
        if self._name == "stdout":
            child.stdout_reads.append(size)
            if child.started is not None:
                child.started.set()
        else:
            child.stderr_reads.append(size)
        if child.hang:
            await child.killed.wait()
            return b""
        if child.fault is not None:
            raise child.fault
        source = self._source
        if self._name == "stderr":
            self._offset = len(source)
            return source
        if self._offset >= len(source):
            return b""
        step = len(source) if size is None or size < 0 else size
        chunk = source[self._offset:self._offset + step]
        self._offset += len(chunk)
        if self._name == "stdout":
            child.bytes_served += len(chunk)
        return chunk

    async def readexactly(self, size: int) -> bytes:
        return await self.read(size)

    async def readline(self) -> bytes:
        return await self.read(-1)

    def at_eof(self) -> bool:
        return self._offset >= len(self._source)


class FfmpegChildStub:
    """A child that answers with a fixed payload, or with whatever mode it was built in."""

    def __init__(
        self,
        payload: bytes = b"",
        *,
        returncode: int = 0,
        stderr: bytes = b"",
        hang: bool = False,
        fault: BaseException | None = None,
        name: str = "child",
        started: asyncio.Event | None = None,
    ) -> None:
        self.name = name
        self.payload = payload
        self.stderr_bytes = stderr
        self.hang = hang
        self.fault = fault
        self.started = started
        self.returncode: int | None = None
        self.alive = True
        self.kill_calls = 0
        self.wait_calls = 0
        self.stdout_reads: list[int] = []
        self.stderr_reads: list[int] = []
        self.bytes_served = 0
        self.killed = asyncio.Event()
        self._exit_code = returncode
        self.stdout = _ChildStream(self, "stdout")
        self.stderr = _ChildStream(self, "stderr")

    def kill(self) -> None:
        self.kill_calls += 1
        self.alive = False
        self.killed.set()

    async def wait(self) -> int | None:
        self.wait_calls += 1
        if self.returncode is None:
            self.returncode = self._exit_code
        self.alive = False
        return self.returncode

    def _reap(self, returncode: int) -> None:
        self.returncode = returncode
        self.alive = False

    async def communicate(self, *_args: Any, **_kwargs: Any) -> tuple[bytes, bytes]:
        if self.started is not None:
            self.started.set()
        if self.hang:
            await self.killed.wait()
            return b"", self.stderr_bytes
        if self.fault is not None:
            raise self.fault
        self._reap(self._exit_code)
        return self.payload, self.stderr_bytes


def child_exec_factory(made: list[FfmpegChildStub], build: Any = None) -> Any:
    """An `asyncio.create_subprocess_exec` replacement that records what it handed out."""

    async def _fake_exec(*_argv: Any, **_kwargs: Any) -> FfmpegChildStub:
        child = build(len(made)) if build is not None else FfmpegChildStub()
        made.append(child)
        return child

    return _fake_exec
