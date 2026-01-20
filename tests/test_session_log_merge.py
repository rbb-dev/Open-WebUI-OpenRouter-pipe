"""Tests for session log archive merging functionality.

These tests verify the merge helpers that prevent data loss when multiple
pipe invocations share the same message_id (main response, title generation,
tool calls, etc.).
"""

from __future__ import annotations

import io
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest

from open_webui_openrouter_pipe import Pipe


class TestConvertJsonlToInternal:
    """Tests for _convert_jsonl_to_internal helper."""

    def test_converts_iso_timestamp_to_epoch(self, pipe_instance) -> None:
        """ISO 'ts' field is converted to 'created' epoch float."""
        pipe = pipe_instance
        evt = {
            "ts": "2025-01-20T10:30:45.123000+00:00",
            "level": "INFO",
            "message": "test",
        }
        result = pipe._convert_jsonl_to_internal(evt)

        assert "created" in result
        assert "ts" not in result  # ts is removed after conversion
        assert isinstance(result["created"], float)
        # Verify approximate timestamp (2025-01-20 10:30:45 UTC)
        assert 1737369045 <= result["created"] <= 1737369046

    def test_converts_z_suffix_timestamp(self, pipe_instance) -> None:
        """'Z' suffix (Zulu time) is properly handled."""
        pipe = pipe_instance
        evt = {"ts": "2025-01-20T10:30:45.123Z", "message": "test"}
        result = pipe._convert_jsonl_to_internal(evt)

        assert "created" in result
        assert isinstance(result["created"], float)

    def test_preserves_existing_created_field(self, pipe_instance) -> None:
        """If 'created' already exists, 'ts' is ignored."""
        pipe = pipe_instance
        evt = {
            "ts": "2025-01-20T10:30:45.123Z",
            "created": 1234567890.0,
            "message": "test",
        }
        result = pipe._convert_jsonl_to_internal(evt)

        # created should be unchanged, ts should remain
        assert result["created"] == 1234567890.0
        assert "ts" in result  # ts is NOT removed when created exists

    def test_handles_missing_ts_field(self, pipe_instance) -> None:
        """Events without 'ts' field pass through unchanged."""
        pipe = pipe_instance
        evt = {"created": 1234567890.0, "message": "test"}
        result = pipe._convert_jsonl_to_internal(evt)

        assert result == evt

    def test_handles_invalid_timestamp_gracefully(self, pipe_instance) -> None:
        """Invalid timestamp falls back to current time."""
        pipe = pipe_instance
        before = time.time()
        evt = {"ts": "not-a-valid-timestamp", "message": "test"}
        result = pipe._convert_jsonl_to_internal(evt)
        after = time.time()

        assert "created" in result
        assert before <= result["created"] <= after

    def test_does_not_mutate_original_event(self, pipe_instance) -> None:
        """Original event dict is not modified."""
        pipe = pipe_instance
        original = {"ts": "2025-01-20T10:30:45.123Z", "message": "test"}
        original_copy = dict(original)
        pipe._convert_jsonl_to_internal(original)

        assert original == original_copy


class TestDedupeSessionLogEvents:
    """Tests for _dedupe_session_log_events helper."""

    def test_removes_exact_duplicates(self, pipe_instance) -> None:
        """Identical events are deduplicated."""
        pipe = pipe_instance
        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "hello"},
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "hello"},
        ]
        result = pipe._dedupe_session_log_events(events)

        assert len(result) == 1
        assert result[0]["message"] == "hello"

    def test_keeps_events_with_different_timestamps(self, pipe_instance) -> None:
        """Events with different timestamps are kept."""
        pipe = pipe_instance
        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "hello"},
            {"created": 1001.0, "request_id": "req1", "lineno": 42, "message": "hello"},
        ]
        result = pipe._dedupe_session_log_events(events)

        assert len(result) == 2

    def test_keeps_events_with_different_request_ids(self, pipe_instance) -> None:
        """Events from different request_ids are kept."""
        pipe = pipe_instance
        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "hello"},
            {"created": 1000.0, "request_id": "req2", "lineno": 42, "message": "hello"},
        ]
        result = pipe._dedupe_session_log_events(events)

        assert len(result) == 2

    def test_keeps_events_with_different_line_numbers(self, pipe_instance) -> None:
        """Events from different line numbers are kept."""
        pipe = pipe_instance
        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "hello"},
            {"created": 1000.0, "request_id": "req1", "lineno": 43, "message": "hello"},
        ]
        result = pipe._dedupe_session_log_events(events)

        assert len(result) == 2

    def test_keeps_events_with_different_messages(self, pipe_instance) -> None:
        """Events with different messages are kept."""
        pipe = pipe_instance
        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "hello"},
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "world"},
        ]
        result = pipe._dedupe_session_log_events(events)

        assert len(result) == 2

    def test_preserves_order_of_first_occurrence(self, pipe_instance) -> None:
        """First occurrence of duplicate is kept."""
        pipe = pipe_instance
        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "first", "extra": "A"},
            {"created": 1000.0, "request_id": "req1", "lineno": 42, "message": "first", "extra": "B"},
        ]
        result = pipe._dedupe_session_log_events(events)

        assert len(result) == 1
        assert result[0]["extra"] == "A"  # First occurrence kept

    def test_handles_missing_fields_gracefully(self, pipe_instance) -> None:
        """Events with missing fields are handled."""
        pipe = pipe_instance
        events = [
            {"message": "no created or request_id"},
            {"created": 1000.0, "message": "no request_id"},
            {"request_id": "req1", "message": "no created"},
        ]
        result = pipe._dedupe_session_log_events(events)

        # All are unique because of different content
        assert len(result) == 3

    def test_handles_empty_list(self, pipe_instance) -> None:
        """Empty event list returns empty list."""
        pipe = pipe_instance
        result = pipe._dedupe_session_log_events([])
        assert result == []


class TestReadSessionLogArchiveEvents:
    """Tests for _read_session_log_archive_events helper."""

    def test_reads_events_from_encrypted_zip(self, pipe_instance) -> None:
        """Events are read from an AES-encrypted zip archive."""
        import pyzipper

        pipe = pipe_instance

        # Create a test archive
        events = [
            {"ts": "2025-01-20T10:30:45.123Z", "level": "INFO", "message": "event 1"},
            {"ts": "2025-01-20T10:30:46.456Z", "level": "DEBUG", "message": "event 2"},
        ]
        jsonl_content = "\n".join(json.dumps(e) for e in events)

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = Path(f.name)

        try:
            password = b"test-password-123"
            with pyzipper.AESZipFile(
                zip_path, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES
            ) as zf:
                zf.setpassword(password)
                zf.writestr("logs.jsonl", jsonl_content.encode("utf-8"))

            settings = ("/tmp", password, "deflated", 6)
            result = pipe._read_session_log_archive_events(zip_path, settings)

            assert len(result) == 2
            # Verify conversion happened
            assert "created" in result[0]
            assert "created" in result[1]
            assert result[0]["message"] == "event 1"
            assert result[1]["message"] == "event 2"
        finally:
            zip_path.unlink(missing_ok=True)

    def test_returns_empty_list_for_missing_jsonl(self, pipe_instance) -> None:
        """Archive without logs.jsonl returns empty list."""
        import pyzipper

        pipe = pipe_instance

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = Path(f.name)

        try:
            password = b"test-password"
            with pyzipper.AESZipFile(
                zip_path, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES
            ) as zf:
                zf.setpassword(password)
                zf.writestr("other.txt", b"not jsonl")

            settings = ("/tmp", password, "deflated", 6)
            result = pipe._read_session_log_archive_events(zip_path, settings)

            assert result == []
        finally:
            zip_path.unlink(missing_ok=True)

    def test_skips_malformed_json_lines(self, pipe_instance) -> None:
        """Malformed JSON lines are skipped, valid ones are returned."""
        import pyzipper

        pipe = pipe_instance

        jsonl_content = (
            '{"ts": "2025-01-20T10:30:45.123Z", "message": "valid"}\n'
            "not valid json\n"
            '{"ts": "2025-01-20T10:30:46.456Z", "message": "also valid"}\n'
        )

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = Path(f.name)

        try:
            password = b"test-password"
            with pyzipper.AESZipFile(
                zip_path, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES
            ) as zf:
                zf.setpassword(password)
                zf.writestr("logs.jsonl", jsonl_content.encode("utf-8"))

            settings = ("/tmp", password, "deflated", 6)
            result = pipe._read_session_log_archive_events(zip_path, settings)

            assert len(result) == 2
            assert result[0]["message"] == "valid"
            assert result[1]["message"] == "also valid"
        finally:
            zip_path.unlink(missing_ok=True)

    def test_handles_empty_lines(self, pipe_instance) -> None:
        """Empty lines in JSONL are skipped."""
        import pyzipper

        pipe = pipe_instance

        jsonl_content = (
            '{"ts": "2025-01-20T10:30:45.123Z", "message": "one"}\n'
            "\n"
            "   \n"
            '{"ts": "2025-01-20T10:30:46.456Z", "message": "two"}\n'
        )

        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as f:
            zip_path = Path(f.name)

        try:
            password = b"test-password"
            with pyzipper.AESZipFile(
                zip_path, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES
            ) as zf:
                zf.setpassword(password)
                zf.writestr("logs.jsonl", jsonl_content.encode("utf-8"))

            settings = ("/tmp", password, "deflated", 6)
            result = pipe._read_session_log_archive_events(zip_path, settings)

            assert len(result) == 2
        finally:
            zip_path.unlink(missing_ok=True)


class TestMergeIntegration:
    """Integration tests for the full merge flow."""

    def test_merge_combines_existing_and_new_events(self, pipe_instance) -> None:
        """When archive exists, its events are merged with new DB events."""
        # This is a higher-level test that verifies the merge logic works together
        pipe = pipe_instance

        # Create some "existing" events (as if from archive)
        existing = [
            {"created": 1000.0, "request_id": "req1", "lineno": 10, "message": "existing 1"},
            {"created": 1001.0, "request_id": "req1", "lineno": 20, "message": "existing 2"},
        ]

        # Create some "new" events (as if from DB)
        new_events = [
            {"created": 1002.0, "request_id": "req2", "lineno": 30, "message": "new 1"},
            {"created": 1003.0, "request_id": "req2", "lineno": 40, "message": "new 2"},
        ]

        # Merge them
        merged = existing + new_events
        merged = pipe._dedupe_session_log_events(merged)
        merged.sort(key=lambda e: e.get("created", 0))

        assert len(merged) == 4
        assert merged[0]["message"] == "existing 1"
        assert merged[1]["message"] == "existing 2"
        assert merged[2]["message"] == "new 1"
        assert merged[3]["message"] == "new 2"

    def test_merge_dedupes_overlapping_events(self, pipe_instance) -> None:
        """Duplicate events between archive and DB are deduplicated."""
        pipe = pipe_instance

        # Create overlapping events
        existing = [
            {"created": 1000.0, "request_id": "req1", "lineno": 10, "message": "same event"},
        ]
        new_events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 10, "message": "same event"},
            {"created": 1001.0, "request_id": "req1", "lineno": 20, "message": "unique"},
        ]

        merged = existing + new_events
        merged = pipe._dedupe_session_log_events(merged)
        merged.sort(key=lambda e: e.get("created", 0))

        assert len(merged) == 2
        assert merged[0]["message"] == "same event"
        assert merged[1]["message"] == "unique"

    def test_sort_is_stable_for_equal_timestamps(self, pipe_instance) -> None:
        """Events with equal timestamps maintain relative order (stable sort)."""
        pipe = pipe_instance

        events = [
            {"created": 1000.0, "request_id": "req1", "lineno": 1, "message": "A"},
            {"created": 1000.0, "request_id": "req1", "lineno": 2, "message": "B"},
            {"created": 1000.0, "request_id": "req1", "lineno": 3, "message": "C"},
        ]

        # Sort should preserve insertion order for equal keys
        events.sort(key=lambda e: e.get("created", 0))

        assert events[0]["message"] == "A"
        assert events[1]["message"] == "B"
        assert events[2]["message"] == "C"
