"""The reasoning-status throttle decides how often Open WebUI is told anything.

`ReasoningStatusThrottle.feed` is what stands between a reasoning stream and one socket
emit per token. It had no test of its own: the three that name it drive the streaming
loop end to end, and that loop finishes with `_maybe_emit_reasoning_status("", force=True)`
-- `force` skips the entire decision, so `assert any("Thinking" in ...)` was satisfied by
the final flush and never by the throttle. Inverting any of the three comparisons left
the whole suite green while turning 40 short deltas into 40 status updates instead of one.

Tested as what it is: pure decision logic with no I/O, so the *rate* is observable rather
than the eventual text. Written against the constants rather than their values, so
retuning them is not a test failure.
"""

from __future__ import annotations

import pytest

from open_webui_openrouter_pipe.streaming.constants import (
    REASONING_STATUS_IDLE_SECONDS,
    REASONING_STATUS_MAX_CHARS,
    REASONING_STATUS_MIN_CHARS,
    REASONING_STATUS_PUNCTUATION,
    ReasoningStatusThrottle,
)


def _primed() -> ReasoningStatusThrottle:
    """A throttle that has emitted once, so `_last_emit` is set.

    The first update fires as soon as MIN_CHARS is reached, because `elapsed is None`.
    Every test below is about what happens *after* that, which is where a stream spends
    all of its time.
    """
    throttle = ReasoningStatusThrottle()
    primer = "x" * (REASONING_STATUS_MIN_CHARS + 4)
    assert throttle.feed(primer) is not None, (
        "the throttle did not emit on the first buffer past MIN_CHARS, so these tests "
        "are not starting from the state they claim"
    )
    return throttle


def test_the_throttle_does_not_emit_once_per_delta(monkeypatch):
    """The defect the missing test allowed: one Open WebUI socket emit per token.

    Frozen clock, not a fast one. The idle branch is the third release condition and the
    only one this drive cannot control by construction; leaving it to the wall clock
    makes "every condition is unmet" a claim about how quickly 39 string appends run.
    """
    import open_webui_openrouter_pipe.streaming.constants as constants

    clock = {"t": 1000.0}
    monkeypatch.setattr(constants, "perf_counter", lambda: clock["t"])

    throttle = _primed()
    # Sized from the constant so retuning it is not a test failure: one delta short of
    # the cap, ending in no punctuation, inside the idle window. Every release condition
    # is unmet, so the correct answer is exactly zero.
    deltas = (REASONING_STATUS_MAX_CHARS // 4) - 1
    assert deltas >= 2, (
        f"the cap is {REASONING_STATUS_MAX_CHARS}, so this drives {deltas} deltas and "
        "an empty loop satisfies `emitted == []` without exercising the throttle at all"
    )
    emitted = [out for _ in range(deltas) if (out := throttle.feed("tok ")) is not None]
    assert throttle.pending.strip(), (
        f"{deltas} deltas left the buffer empty, so the throttle emitted and cleared it "
        "somewhere this test did not observe, or the deltas never arrived"
    )
    assert emitted == [], (
        f"{deltas} short deltas produced {len(emitted)} status updates with every "
        "release condition unmet. Each one is a socket emit to every viewer of the "
        "chat, on every streaming request with reasoning status enabled. Asserting a "
        "ceiling of 2 here admitted a doubling twice over, and admitted zero as well."
    )


def test_the_throttle_emits_once_the_buffer_reaches_the_cap(monkeypatch):
    """Without a cap a silent stream never updates at all -- and the cap is THE cap.

    Bounded above as well as below. `len(out) >= MAX_CHARS` alone is satisfied by a cap
    of 2*MAX_CHARS -- a doubling of the silence this exists to end -- because the loop
    bound is a multiple of the constant under test. The emit has to land within one
    chunk of the cap being crossed.
    """
    import open_webui_openrouter_pipe.streaming.constants as constants

    clock = {"t": 1000.0}
    monkeypatch.setattr(constants, "perf_counter", lambda: clock["t"])

    chunk = 8
    throttle = _primed()
    out = None
    fed = 0
    while out is None and fed < REASONING_STATUS_MAX_CHARS * 4:
        out = throttle.feed("a" * chunk)
        fed += chunk
    assert out is not None, (
        f"fed {fed} chars without an emit, past a cap of {REASONING_STATUS_MAX_CHARS}; "
        "a reasoning stream with no sentence breaks would show nothing at all"
    )
    assert REASONING_STATUS_MAX_CHARS <= len(out) < REASONING_STATUS_MAX_CHARS + chunk, (
        f"the buffer released at {len(out)} chars against a cap of "
        f"{REASONING_STATUS_MAX_CHARS}. It must fire on the first delta that crosses "
        "the cap, not some multiple of it."
    )


@pytest.mark.parametrize("mark", REASONING_STATUS_PUNCTUATION)
def test_the_throttle_emits_on_a_sentence_boundary(mark):
    """Every member of the tuple, so a dead one is visible.

    `"\n"` was dead: the check was `delta.rstrip().endswith(...)`, and `rstrip()`
    removes trailing whitespace, so the result can never end in a newline. Reasoning
    text that breaks by line rather than by punctuation never triggered a boundary and
    fell through to the character cap. Stripping only spaces and tabs keeps every other
    member working and makes this one reachable.

    The emitted text is `strip()`ped by `feed`, so a trailing newline is correctly not
    part of the status line -- the boundary fires, the whitespace does not travel.
    """
    throttle = _primed()
    body = "Considering the request"
    assert throttle.feed(body) is None, (
        "a fragment below the cap emitted before any boundary was reached"
    )
    assert throttle.feed(mark) == (body + mark).strip()


def test_a_fragment_below_the_minimum_does_not_emit():
    """The floor exists so a two-character update is never pushed on its own."""
    throttle = _primed()
    assert throttle.feed("a" * (REASONING_STATUS_MIN_CHARS - 2)) is None


def test_a_forced_flush_emits_whatever_is_buffered():
    """`force` is what end-of-stream uses, and what masked the three existing tests."""
    throttle = _primed()
    assert throttle.feed("ab") is None
    assert throttle.feed("", force=True) == "ab"


def test_the_buffer_is_cleared_after_an_emit():
    """Otherwise every later update repeats everything already shown."""
    throttle = _primed()
    first = throttle.feed("A first sentence.")
    assert first is not None
    assert throttle.pending == ""
    second = throttle.feed("A second sentence.")
    assert second is not None and first not in second


def test_the_idle_path_needs_both_the_minimum_and_the_quiet_period(monkeypatch):
    """Time is injected, so this asserts the rule rather than sleeping on it."""
    import open_webui_openrouter_pipe.streaming.constants as constants

    clock = {"t": 1000.0}
    monkeypatch.setattr(constants, "perf_counter", lambda: clock["t"])

    throttle = _primed()
    assert throttle.feed("a" * (REASONING_STATUS_MIN_CHARS + 2)) is None, (
        "emitted immediately after the previous emit; the idle period is not consulted"
    )
    clock["t"] += REASONING_STATUS_IDLE_SECONDS + 0.1
    assert throttle.feed("") is not None, (
        "still silent after the idle period elapsed with the buffer past MIN_CHARS"
    )


def test_a_non_string_delta_is_ignored():
    assert ReasoningStatusThrottle().feed(None) is None  # type: ignore[arg-type]
