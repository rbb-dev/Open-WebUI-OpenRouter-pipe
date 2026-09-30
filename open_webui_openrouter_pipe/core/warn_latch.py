"""The one implementation of "warn once, then DEBUG".

A leaf on purpose: it imports nothing from this package. The previous home,
`core/utils.py`, imports from `core/config.py`, so `config.py` could not reach the
helper and hand-rolled its own copy -- the same shape as the original defect, where the
helper sat in `pipe.py` at the top of the import graph and fifteen sites across six
modules below it each wrote their own latch. A module every layer can import is the only placement that makes
"there is exactly one implementation" enforceable rather than aspirational.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict


def warn_level(
    latch: set[str] | dict[str, float],
    cause: str,
    *,
    cooldown_s: float | None = None,
) -> int:
    """WARNING the first time a cause is seen, DEBUG on every repeat.

    Lives here rather than in pipe.py because pipe.py sits at the TOP of the import
    graph: storage/, plugins/ and api/gateway/ cannot import from it without inverting
    the layering, so every one of them wrote its own latch and none of them wrote the
    repeat path. Fifteen sites armed a latch, emitted one WARNING, and then went silent
    at EVERY level for the life of the worker -- an operator who raises the log level to
    diagnose a recurring fault sees nothing at all.

    Two keying modes, because both are in use and they are the same decision: a `set`
    latches a cause forever, a `dict` of monotonic timestamps re-warns after
    `cooldown_s`. Passing a dict without a cooldown is a permanent latch that also
    records when it armed.

    Returns a level rather than doing the logging, so the `logger.log(...)` call stays
    inside the `except` block -- where it is the reason the broad catch is acceptable,
    and where ruff's BLE001 can see it.
    """
    if isinstance(latch, dict):
        now = time.monotonic()
        previous = latch.get(cause)
        if previous is not None and (cooldown_s is None or now - previous < cooldown_s):
            return logging.DEBUG
        latch[cause] = now
        return logging.WARNING
    if cause in latch:
        return logging.DEBUG
    latch.add(cause)
    return logging.WARNING


_shared_latches: dict[str, set[str]] = {}


def shared_latch(name: str) -> set[str]:
    return _shared_latches.setdefault(name, set())


def bounded_warn_level(latch: OrderedDict[str, None], cause: str, window: int) -> int:
    if cause in latch:
        latch.move_to_end(cause)
        return logging.DEBUG
    latch[cause] = None
    while len(latch) > window:
        latch.popitem(last=False)
    return logging.WARNING
