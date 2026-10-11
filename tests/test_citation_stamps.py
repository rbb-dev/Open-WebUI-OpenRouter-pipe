"""The citation date stamp must render in the server's local frame.

Lives outside the dashboard test module on purpose: that module opens with
``pytest.importorskip`` on the plugin package, so a guard over ``streaming/`` source
placed there disappears entirely in both no-plugins bundle modes -- the guard is gone
exactly where nobody is looking.
"""

from __future__ import annotations


import os
import re

from pathlib import Path

import pytest
from typing import Any, cast


def test_the_citation_helpers_render_in_the_servers_own_frame():
    """Assert the value under a timezone where UTC and local disagree TODAY.

    Comparing the helper's date against `datetime.now().astimezone().date()` in the
    ambient zone is not a test: for most of the day those agree even when the helper
    stamps UTC, so a re-normalising mutation passes. The zone below is chosen from the
    current UTC hour precisely so the two calendar days differ right now, which is the
    condition under which the bug is visible to a user.
    """
    import datetime as _dt
    import os
    import time

    if not hasattr(time, "tzset"):
        pytest.skip("TZ manipulation needs tzset (POSIX only)")


    from open_webui_openrouter_pipe.core import utils as _utils

    utc_now = _dt.datetime.now(_dt.UTC)
    zone = "Etc/GMT+12" if utc_now.hour < 12 else "Etc/GMT-14"

    saved_tz = os.environ.get("TZ")
    os.environ["TZ"] = zone
    time.tzset()
    try:
        local_now = _dt.datetime.now(_dt.UTC).astimezone()
        assert local_now.date() != _dt.datetime.now(_dt.UTC).date(), (
            f"{zone} was expected to put local and UTC on different calendar days at "
            f"{utc_now.isoformat()}, but it did not -- this test can no longer detect "
            "a UTC-stamped date and must be re-anchored."
        )

        day = _utils.citation_access_stamp().split("T")[0]
        assert day == local_now.date().isoformat(), (
            f"under {zone} the citation date is {day}, but the server's local date is "
            f"{local_now.date().isoformat()} (UTC is "
            f"{_dt.datetime.now(_dt.UTC).date().isoformat()}). The stamp is rendering "
            "the UTC calendar day, which is the wrong day for every reader in this "
            "zone and is persisted into chat exports."
        )

        assert _dt.date.fromisoformat(day) == local_now.date(), (
            f"{day!r} does not parse back to the local calendar day; the stamp is the "
            "value Open WebUI renders as-is into the citation"
        )
    finally:
        if saved_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved_tz
        time.tzset()


@pytest.mark.skipif(
    bool(os.environ.get("OWUI_PIPE_BUNDLE_PATH")),
    reason="in a bundle the loaded code is the artifact, not this source tree, so a source scan proves nothing about what is running",
)
def test_a_broken_version_lookup_does_not_break_the_import():
    """__init__ is the package's first executable statement.

    importlib.metadata raises more than PackageNotFoundError -- a malformed METADATA
    file raises ValueError -- and narrowing the handler made any such failure take the
    whole pipe down rather than falling back to the pinned version.
    """
    import ast
    from pathlib import Path

    init = Path(__file__).resolve().parents[1] / "open_webui_openrouter_pipe" / "__init__.py"
    tree = ast.parse(init.read_text(encoding="utf-8"), filename=str(init))

    handlers = [
        h
        for node in ast.walk(tree)
        if isinstance(node, ast.Try)
        for h in node.handlers
        if any("_get_version" in ast.unparse(s) for s in node.body)
    ]
    assert handlers, "the version lookup is no longer wrapped; scan is stale"
    for h in handlers:
        assert h.type is None or ast.unparse(h.type) == "Exception", (
            f"the version lookup catches only {ast.unparse(h.type)}; anything else "
            "importlib.metadata raises makes the entire package unimportable"
        )


@pytest.mark.asyncio
async def test_the_emitted_citation_stamp_is_the_local_calendar_day():
    """Assert the EMITTED value under a zone that is not UTC.

    Comparing against the ambient zone made this vacuous on a UTC host -- which is
    every CI runner: both sides are +00:00, so re-normalising the stamp back to UTC
    passed. The zone is forced here for the same reason the sibling forces it, and
    this one covers the emit path rather than the helper in isolation.
    """
    import datetime as _dt
    import os
    import time

    if not hasattr(time, "tzset"):
        pytest.skip("TZ manipulation needs tzset (POSIX only)")

    from open_webui_openrouter_pipe.streaming.event_emitter import EventEmitterHandler

    emitted: list[dict] = []

    async def _sink(event):
        emitted.append(event)

    # The zone is derived from the current UTC hour so that local and UTC ALWAYS land on
    # different calendar days. A fixed +10 only straddles midnight when the UTC hour is
    # 14 or later; for the other 14 hours the two dates are the same string and the
    # assertion below cannot tell a local stamp from a UTC one. CI runs in UTC.
    saved_tz = os.environ.get("TZ")
    os.environ["TZ"] = "Etc/GMT+12" if _dt.datetime.now(_dt.UTC).hour < 12 else "Etc/GMT-14"
    time.tzset()
    try:
        handler = EventEmitterHandler.__new__(EventEmitterHandler)
        await EventEmitterHandler._emit_citation(
            handler, _sink, {"document": "d", "source": {"url": "https://example.test"}}
        )

        assert emitted, "no source event was emitted"
        stamp = emitted[0]["data"]["metadata"][0]["date_accessed"]
        local_day = _dt.datetime.now(_dt.UTC).astimezone().date().isoformat()
        utc_day = _dt.datetime.now(_dt.UTC).date().isoformat()
        assert local_day != utc_day, (
            f"the forced zone put local ({local_day}) and UTC ({utc_day}) on the same "
            "calendar day, so the assertion below is satisfied by a UTC stamp too and "
            "this test proves nothing. Re-anchor it."
        )
        assert stamp.split("T")[0] == local_day, (
            f"{stamp!r} is not the server's local calendar day ({local_day}); Open "
            "WebUI renders the string as-is, so the citation shows a date the reader "
            "never experienced."
        )
        assert stamp.split("T")[0] != utc_day, (
            f"{stamp!r} carries the UTC calendar day, not the local one -- which is the "
            "whole point of the .astimezone() in citation_access_stamp"
        )
    finally:
        if saved_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = saved_tz
        time.tzset()


def test_the_access_stamp_carries_a_time_not_just_a_date(monkeypatch):
    """Precision preserved, not merely uniformity.

    `_emit_citation` stamped a full local timestamp; the other four sites stamped a bare
    date. Consolidating them onto one helper adopted the date-only form and silently
    dropped the time from the one that had it -- in a value Open WebUI persists into the
    chat record and exports verbatim.

    Parametrised over two distinct instants via a frozen clock so a hardcoded return
    cannot satisfy both, and asserted on the PARSED value rather than on the presence of
    a "T", which a date-only string with a suffix would also satisfy.
    """
    import datetime as _dt

    from open_webui_openrouter_pipe.core import utils as _utils

    seen = []
    for hour, minute in ((3, 7), (21, 44)):
        class _Frozen(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return _dt.datetime(2026, 3, 14, hour, minute, 5, tzinfo=tz or _dt.UTC)

        monkeypatch.setattr(_utils.datetime, "datetime", _Frozen)
        stamp = _utils.citation_access_stamp()
        parsed = _dt.datetime.fromisoformat(stamp)
        assert parsed.time() != _dt.time(0, 0), (
            f"citation_access_stamp() returned {stamp!r}, which has no time component. "
            "The value is persisted into chat exports; a bare date loses when the source "
            "was actually read."
        )
        seen.append(parsed)

    assert seen[0] != seen[1], (
        "both frozen clocks produced the same stamp, so the value does not come from the "
        "clock at all"
    )
