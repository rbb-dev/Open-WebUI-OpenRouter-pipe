"""Tests for SSE streaming queue valves and deadlock prevention."""

import pytest
from open_webui_openrouter_pipe import Pipe


def test_queue_valves_unbounded_defaults(pipe_instance):
    """Verify unbounded (0) defaults prevent deadlock."""
    pipe = pipe_instance
    assert pipe.valves.STREAMING_CHUNK_QUEUE_MAXSIZE == 0, "Chunk queue should default to unbounded"
    assert pipe.valves.STREAMING_EVENT_QUEUE_MAXSIZE == 0, "Event queue should default to unbounded"
    assert pipe.valves.STREAMING_EVENT_QUEUE_WARN_SIZE == 1000, "Warning threshold should be 1000"


def test_valve_descriptions_warn_deadlock(pipe_instance):
    """Verify valve descriptions document deadlock risks for small bounded queues."""
    pipe = pipe_instance
    # Access model_fields from class, not instance (avoid Pydantic 2.11+ deprecation)
    chunk_desc = Pipe.Valves.model_fields['STREAMING_CHUNK_QUEUE_MAXSIZE'].description
    event_desc = Pipe.Valves.model_fields['STREAMING_EVENT_QUEUE_MAXSIZE'].description
    warn_desc = Pipe.Valves.model_fields['STREAMING_EVENT_QUEUE_WARN_SIZE'].description

    # Ensure descriptions are not None
    assert chunk_desc is not None, "Chunk queue should have a description"
    assert event_desc is not None, "Event queue should have a description"
    assert warn_desc is not None, "Warning size should have a description"

    # Both queue descriptions should warn about deadlock
    assert 'deadlock' in chunk_desc.lower(), "Chunk queue description should mention deadlock risk"
    assert 'deadlock' in event_desc.lower(), "Event queue description should mention deadlock risk"

    # Both should mention the unbounded (0) recommendation
    assert '0=' in chunk_desc, "Chunk queue should document 0=unbounded"
    assert '0=' in event_desc, "Event queue should document 0=unbounded"

    # Warning description should explain qsize monitoring
    assert 'qsize' in warn_desc.lower(), "Warning description should mention qsize()"
    assert 'warn' in warn_desc.lower(), "Warning description should mention warning behavior"


@pytest.mark.parametrize('qsize, warn_size, delta_time, expected', [
    # Below threshold - never warn
    (999, 1000, 0, False),
    (999, 1000, 30, False),
    (999, 1000, 60, False),

    # At threshold but time not elapsed - no warn
    (1000, 1000, 0, False),
    (1000, 1000, 29.9, False),

    # At threshold and time elapsed - warn
    (1000, 1000, 30.0, True),
    (1000, 1000, 60, True),

    # Above threshold but time not elapsed - no warn
    (1200, 1000, 0, False),
    (1200, 1000, 29, False),

    # Above threshold and time elapsed - warn
    (1200, 1000, 30, True),
    (1200, 1000, 100, True),

    # Edge cases
    (0, 1000, 30, False),     # Empty queue
    (5000, 1000, 30, True),   # Very large backlog
    (1000, 100, 30, True),    # Lower warning threshold
])
def test_warn_condition_logic(qsize, warn_size, delta_time, expected):
    """Test drain loop warning condition matches implementation logic.

    Implementation: Pipe._should_warn_event_queue_backlog(...)
    """
    should_warn = Pipe._should_warn_event_queue_backlog(
        qsize,
        warn_size,
        delta_time,
        0.0,
    )
    assert should_warn == expected, f"Failed for qsize={qsize}, warn_size={warn_size}, delta={delta_time}"


def test_queue_valve_constraints():
    """Verify valve constraints allow unbounded and bounded configurations."""
    # Access model_fields from class, not instance (avoid Pydantic 2.11+ deprecation)
    chunk_field = Pipe.Valves.model_fields['STREAMING_CHUNK_QUEUE_MAXSIZE']
    event_field = Pipe.Valves.model_fields['STREAMING_EVENT_QUEUE_MAXSIZE']
    warn_field = Pipe.Valves.model_fields['STREAMING_EVENT_QUEUE_WARN_SIZE']

    # Check ge=0 constraint
    assert chunk_field.metadata[0].ge == 0, "Chunk queue should allow ge=0"
    assert event_field.metadata[0].ge == 0, "Event queue should allow ge=0"
    assert warn_field.metadata[0].ge == 100, "Warning size should require ge=100"


@pytest.mark.parametrize('chunk_size, event_size', [
    (0, 0),       # Unbounded (recommended)
    (1000, 1000), # Large bounded (safe)
    (50, 50),     # Small bounded (documented risk)
    (100, 200),   # Asymmetric
])
def test_valve_accepts_various_queue_sizes(chunk_size, event_size, pipe_instance):
    """Verify valves accept various queue size configurations."""
    pipe = pipe_instance

    # These should all be valid configurations (though some are risky)
    pipe.valves.STREAMING_CHUNK_QUEUE_MAXSIZE = chunk_size
    pipe.valves.STREAMING_EVENT_QUEUE_MAXSIZE = event_size

    assert pipe.valves.STREAMING_CHUNK_QUEUE_MAXSIZE == chunk_size
    assert pipe.valves.STREAMING_EVENT_QUEUE_MAXSIZE == event_size


def test_warning_cooldown_prevents_spam():
    """Verify 30-second cooldown prevents log spam.

    Implementation initializes last_warn_ts=0.0, so first warning requires
    delta >= 30.0 from epoch 0.0.
    """
    # Simulating multiple checks within cooldown window
    # Implementation in pipe starts with event_queue_warn_last_ts = 0.0
    timestamps = [0.0, 10.0, 20.0, 29.9, 30.0, 60.0, 90.0]
    last_warn_ts = 0.0
    warnings_emitted = []

    for now in timestamps:
        if Pipe._should_warn_event_queue_backlog(1000, 1000, now, last_warn_ts):
            warnings_emitted.append(now)
            last_warn_ts = now

    # First warn at 30.0 (30.0-0.0>=30), next at 60.0 (60.0-30.0>=30), then 90.0
    assert warnings_emitted == [30.0, 60.0, 90.0], "Should respect 30s cooldown"


def test_unbounded_queue_semantics():
    """Document that maxsize=0 means unbounded in asyncio.Queue."""
    import asyncio

    # asyncio.Queue with maxsize=0 is unbounded (never blocks on put)
    unbounded_queue = asyncio.Queue(maxsize=0)
    bounded_queue = asyncio.Queue(maxsize=10)

    assert unbounded_queue.maxsize == 0, "maxsize=0 is the unbounded marker"
    assert bounded_queue.maxsize == 10, "Positive maxsize creates bounded queue"


@pytest.mark.parametrize('warn_size', [100, 500, 1000, 5000])
def test_warning_size_minimum(warn_size, pipe_instance):
    """Verify warning size minimum prevents spam on normal loads."""
    pipe = pipe_instance
    pipe.valves.STREAMING_EVENT_QUEUE_WARN_SIZE = warn_size

    # Should accept any value >= 100
    assert pipe.valves.STREAMING_EVENT_QUEUE_WARN_SIZE >= 100
    assert pipe.valves.STREAMING_EVENT_QUEUE_WARN_SIZE == warn_size
