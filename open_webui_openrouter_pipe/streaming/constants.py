"""Shared streaming constants to avoid duplication across modules."""

# Default reasoning status emission thresholds (streaming core + event emitter).
REASONING_STATUS_PUNCTUATION = (".", "!", "?", ":", "\n")
REASONING_STATUS_MAX_CHARS = 160
REASONING_STATUS_MIN_CHARS = 12
REASONING_STATUS_IDLE_SECONDS = 0.75

# Reasoning tracker thresholds (different cadence for tracker-based status).
REASONING_STATUS_TRACKER_PUNCTUATION = (".", "!", "?", ":", ";", ",")
REASONING_STATUS_TRACKER_MAX_CHARS = 200
REASONING_STATUS_TRACKER_MIN_CHARS = 40
REASONING_STATUS_TRACKER_IDLE_SECONDS = 0.5
