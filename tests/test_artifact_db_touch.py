from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def test_db_fetch_sync_touches_created_at_best_effort(pipe_instance) -> None:
    """Test that _db_fetch_sync touches created_at timestamp during fetch.

    This test uses real SQLAlchemy mocking to verify the touch behavior.
    """
    # Mock the SQLAlchemy model and query chain
    mock_model = MagicMock()
    mock_query = MagicMock()
    mock_session = MagicMock()

    # Configure query chain
    mock_session.query.return_value = mock_query
    mock_query.filter.return_value = mock_query

    # Mock rows with proper attributes
    mock_row_a = MagicMock()
    mock_row_a.id = "a1"
    mock_row_a.payload = {"type": "tool_output", "output": "ok"}
    mock_row_a.is_encrypted = False

    mock_row_b = MagicMock()
    mock_row_b.id = "b2"
    mock_row_b.payload = {"type": "tool_output", "output": "ok"}
    mock_row_b.is_encrypted = False

    mock_query.all.return_value = [mock_row_a, mock_row_b]
    mock_query.update.return_value = 2

    # Set up pipe instance with mocked session factory
    pipe_instance._item_model = mock_model
    pipe_instance._session_factory = lambda: mock_session

    # Execute the fetch
    results = pipe_instance._db_fetch_sync("chat123", None, ["a1", "b2"])

    # Verify results
    assert results == {"a1": {"type": "tool_output", "output": "ok"}, "b2": {"type": "tool_output", "output": "ok"}}

    # Verify session was used correctly (2 sessions: one for read, one for touch)
    # The touch session should have committed
    assert mock_session.commit.called

    # Verify the touch update was called
    assert mock_query.update.called


def test_db_fetch_sync_touch_failure_never_breaks_read(pipe_instance) -> None:
    """Test that touch failures are gracefully handled and don't break reads.

    Uses real exception handling with SQLAlchemy mocks.
    """
    # Mock the SQLAlchemy model and sessions
    mock_model = MagicMock()

    # First session for reading (succeeds)
    read_session = MagicMock()
    read_query = MagicMock()
    read_session.query.return_value = read_query
    read_query.filter.return_value = read_query

    mock_row = MagicMock()
    mock_row.id = "a1"
    mock_row.payload = {"type": "tool_output", "output": "ok"}
    mock_row.is_encrypted = False

    read_query.all.return_value = [mock_row]

    # Second session for touching (fails)
    touch_session = MagicMock()
    touch_query = MagicMock()
    touch_session.query.return_value = touch_query
    touch_query.filter.return_value = touch_query
    touch_query.all.return_value = [mock_row]

    # Make update raise an exception
    touch_query.update.side_effect = RuntimeError("update failed")

    # Session factory returns different sessions
    sessions = [read_session, touch_session]
    pipe_instance._item_model = mock_model
    pipe_instance._session_factory = lambda: sessions.pop(0)

    # Execute the fetch - should succeed despite touch failure
    results = pipe_instance._db_fetch_sync("chat123", None, ["a1"])

    # Verify results are still correct
    assert results == {"a1": {"type": "tool_output", "output": "ok"}}

    # Verify touch session was rolled back after failure
    assert touch_session.rollback.called
