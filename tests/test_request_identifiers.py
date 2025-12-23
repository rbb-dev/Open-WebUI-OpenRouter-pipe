from __future__ import annotations

from typing import Any

import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import (
    Pipe,
    _apply_identifier_valves_to_payload,
    _filter_openrouter_request,
)


def _base_payload() -> dict[str, Any]:
    return {"model": "openrouter/test", "input": "ping", "stream": False}


def test_identifier_valves_default_omit_everything():
    valves = Pipe.Valves()
    payload = _base_payload()
    payload.update(
        {
            "user": "should-drop",
            "session_id": "should-drop",
            "metadata": {"user_id": "should-drop"},
        }
    )

    _apply_identifier_valves_to_payload(
        payload,
        valves=valves,
        owui_metadata={"session_id": "sess", "chat_id": "chat", "message_id": "msg"},
        owui_user_id="user",
    )
    assert "user" not in payload
    assert "session_id" not in payload
    assert "metadata" not in payload


@pytest.mark.parametrize(
    "valves_kwargs, expected_payload",
    [
        (
            {"SEND_END_USER_ID": True},
            {"user": "u1", "metadata": {"user_id": "u1"}},
        ),
        (
            {"SEND_SESSION_ID": True},
            {"session_id": "s1", "metadata": {"session_id": "s1"}},
        ),
        (
            {"SEND_CHAT_ID": True},
            {"metadata": {"chat_id": "c1"}},
        ),
        (
            {"SEND_MESSAGE_ID": True},
            {"metadata": {"message_id": "m1"}},
        ),
        (
            {"SEND_END_USER_ID": True, "SEND_SESSION_ID": True, "SEND_CHAT_ID": True, "SEND_MESSAGE_ID": True},
            {
                "user": "u1",
                "session_id": "s1",
                "metadata": {
                    "user_id": "u1",
                    "session_id": "s1",
                    "chat_id": "c1",
                    "message_id": "m1",
                },
            },
        ),
    ],
)
def test_identifier_valves_populate_expected_fields(valves_kwargs, expected_payload):
    valves = Pipe.Valves(**valves_kwargs)
    payload = _base_payload()
    _apply_identifier_valves_to_payload(
        payload,
        valves=valves,
        owui_metadata={"session_id": "s1", "chat_id": "c1", "message_id": "m1"},
        owui_user_id="u1",
    )

    for key, value in expected_payload.items():
        assert payload.get(key) == value


def test_filter_openrouter_request_sanitizes_metadata_constraints():
    valves = Pipe.Valves(SEND_END_USER_ID=True)
    payload = _base_payload()

    _apply_identifier_valves_to_payload(
        payload,
        valves=valves,
        owui_metadata={},
        owui_user_id="u1",
    )
    # Add invalid metadata entries that should be dropped by sanitizer.
    payload["metadata"].update(
        {
            "ok": "v",
            "bad[key]": "v",
            "too_long_value": "x" * 600,
        }
    )
    filtered = _filter_openrouter_request(payload)
    assert "metadata" in filtered
    assert filtered["metadata"]["user_id"] == "u1"
    assert filtered["metadata"]["ok"] == "v"
    assert "bad[key]" not in filtered["metadata"]
    assert "too_long_value" not in filtered["metadata"]

