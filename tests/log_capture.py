"""Reading `caplog` when the pipe's logger hands the capture handler each record twice.

`SessionLogger.get_logger` sets ``propagate = False`` on the ``open_webui_openrouter_pipe``
logger and replaces propagation with ``_HostForwardHandler``, which re-emits every record
to the ancestor loggers' handlers. From pytest 9.1 ``catching_logs`` attaches the capture
handler to every non-propagating logger as well as to the root, so that one handler
instance receives each pipe record twice: once directly from the pipe logger, once when
the forwarder walks up to the root and calls the same object again.

The two arrivals are the SAME ``LogRecord``, so identity separates a redelivery from a
second emission -- two ``logger.warning(...)`` calls build two objects, one call delivered
twice builds one. Deduplicating on the rendered message would not: it collapses the flood
of identical lines that a broken warn-once latch produces, which is the regression these
counts exist to catch.
"""

from __future__ import annotations

import logging


def emitted(
    caplog,
    *,
    min_level: int = logging.NOTSET,
    level: int | None = None,
    containing: str | None = None,
) -> list[logging.LogRecord]:
    """The distinct records the code under test created, filtered by level and message.

    ``level`` pins an exact level, ``min_level`` a floor; pass one. Filtering by level
    rather than raising ``caplog.set_level`` keeps a first-call-degraded-to-DEBUG
    regression visible instead of discarding it before it can be counted.
    """
    seen: set[int] = set()
    records: list[logging.LogRecord] = []
    for record in caplog.records:
        if level is not None and record.levelno != level:
            continue
        if record.levelno < min_level:
            continue
        if containing is not None and containing not in record.getMessage():
            continue
        if id(record) in seen:
            continue
        seen.add(id(record))
        records.append(record)
    return records
