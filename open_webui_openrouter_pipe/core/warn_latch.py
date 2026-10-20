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


def prune_expired(latch: dict, now: float, window: float) -> None:
    for key in [k for k, seen in latch.items() if now - seen >= window]:
        latch.pop(key, None)


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
