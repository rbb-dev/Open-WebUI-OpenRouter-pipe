"""The jumping test clock, importable by the files that ask for it.

`conftest.py` used to define `event_loop_policy`, which made it directory-scope: pytest gives a
same-named fixture in a conftest priority over the plugin's own, so every async test in the suite
inherited a loop that jumps to its next timer. A file that then waits out a real limit never waits,
and a test whose product budget is supposed to stop the work instead sees the loop skip ahead.
The fixture therefore lives here, and a file opts in by importing it -- the import is the record of
which files run on a simulated clock.
"""

from __future__ import annotations

import asyncio

import pytest

from tests.conftest import _TimeTravelPolicy


@pytest.fixture
def event_loop_policy(request):
    """The jumping clock, except where a test asks for a real one.

    Opt-in, not ambient: a file that waits out a limit, or that drives a retry honouring a `Retry-After`,
    asks for this with `pytestmark = pytest.mark.usefixtures("event_loop_policy")` and pays no wall-clock.
    Two tests elsewhere deadlock on it: they arrange for a plugin to still be working when the batch
    deadline passes, and with every wake-up collapsed to the same instant the loop reaches a state where
    no timer is left to break the wait. They keep a real loop and shrink their own numbers instead --
    what they check is the order of three limits against each other, not the size of any of them.
    """
    if request.node.get_closest_marker("real_clock"):
        return asyncio.DefaultEventLoopPolicy()
    return _TimeTravelPolicy()
