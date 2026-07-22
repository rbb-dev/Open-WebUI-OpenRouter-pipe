# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false
from __future__ import annotations

import hashlib
import hmac
from typing import Any

import pytest

from open_webui_openrouter_pipe import (
    Pipe,
    _apply_identifier_valves_to_payload,
    _filter_openrouter_request,
)


def _base_payload() -> dict[str, Any]:
    return {"model": "openrouter/test", "input": "ping", "stream": False}


def test_identifier_valves_default_omit_everything():
    valves = Pipe.Valves(SEND_CACHE_SESSION_ID=False)
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
            {"metadata": {"session_id": "s1"}},
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


_STICKY_CHAT_ID = "chat-1234-abcd"
_STICKY_SECRET = "unit-test-secret"


def _expected_sticky(chat_id: str, secret: str) -> str:
    return hmac.new(secret.encode("utf-8"), chat_id.encode("utf-8"), hashlib.sha256).hexdigest()


def _sticky_session_for(chat_id: str, **valve_kwargs: Any) -> Any:
    payload = _base_payload()
    _apply_identifier_valves_to_payload(
        payload,
        valves=Pipe.Valves(**valve_kwargs),
        owui_metadata={"chat_id": chat_id} if chat_id else {},
        owui_user_id="u1",
    )
    return payload.get("session_id"), payload


def test_sticky_session_default_on_sets_opaque_session_id(monkeypatch):
    monkeypatch.setenv("WEBUI_SECRET_KEY", _STICKY_SECRET)
    sid, payload = _sticky_session_for(_STICKY_CHAT_ID)
    assert sid == _expected_sticky(_STICKY_CHAT_ID, _STICKY_SECRET)
    assert len(sid) == 64 and all(c in "0123456789abcdef" for c in sid)
    assert sid != _STICKY_CHAT_ID and _STICKY_CHAT_ID not in sid  # opaque, no raw leak
    assert "metadata" not in payload  # sticky adds no metadata


def test_sticky_session_deterministic_and_unique(monkeypatch):
    monkeypatch.setenv("WEBUI_SECRET_KEY", _STICKY_SECRET)
    a1, _ = _sticky_session_for("chat-A")
    assert a1 is not None and len(a1) == 64
    a2, _ = _sticky_session_for("chat-A")
    b1, _ = _sticky_session_for("chat-B")
    assert a1 == a2  # stable across turns of the same conversation
    assert a1 != b1  # unique per conversation


def test_sticky_session_skipped_without_secret(monkeypatch):
    monkeypatch.delenv("WEBUI_SECRET_KEY", raising=False)
    sid, payload = _sticky_session_for(_STICKY_CHAT_ID)
    assert sid is None  # no unkeyed fallback when the secret is unset
    assert "session_id" not in payload


def test_sticky_session_skipped_without_chat_id(monkeypatch):
    monkeypatch.setenv("WEBUI_SECRET_KEY", _STICKY_SECRET)
    sid, payload = _sticky_session_for("")
    assert sid is None
    assert "session_id" not in payload


def test_sticky_session_skipped_for_whitespace_chat_id(monkeypatch):
    monkeypatch.setenv("WEBUI_SECRET_KEY", _STICKY_SECRET)
    sid, payload = _sticky_session_for("   ")
    assert sid is None  # .strip() guard rejects whitespace; never HMAC("")
    assert "session_id" not in payload


def test_send_session_id_and_sticky_coexist(monkeypatch):
    monkeypatch.setenv("WEBUI_SECRET_KEY", _STICKY_SECRET)
    payload = _base_payload()
    _apply_identifier_valves_to_payload(
        payload,
        valves=Pipe.Valves(SEND_SESSION_ID=True),  # sticky stays default-on; they no longer compete
        owui_metadata={"session_id": "raw-sess", "chat_id": _STICKY_CHAT_ID},
        owui_user_id="u1",
    )
    # top-level session_id is the cache pin (hashed chat_id) — never the raw session
    assert payload.get("session_id") == _expected_sticky(_STICKY_CHAT_ID, _STICKY_SECRET)
    # the raw OWUI session goes to metadata only (observability)
    assert payload["metadata"]["session_id"] == "raw-sess"


def test_sticky_session_disabled(monkeypatch):
    monkeypatch.setenv("WEBUI_SECRET_KEY", _STICKY_SECRET)
    sid, payload = _sticky_session_for(_STICKY_CHAT_ID, SEND_CACHE_SESSION_ID=False)
    assert sid is None
    assert "session_id" not in payload


def test_invalid_metadata_session_dropped_but_sticky_pin_remains(monkeypatch):
    monkeypatch.setenv("WEBUI_SECRET_KEY", _STICKY_SECRET)
    payload = _base_payload()
    _apply_identifier_valves_to_payload(
        payload,
        valves=Pipe.Valves(SEND_SESSION_ID=True),
        owui_metadata={"session_id": "   ", "chat_id": _STICKY_CHAT_ID},
        owui_user_id="u1",
    )
    # whitespace session is dropped from metadata...
    assert "session_id" not in payload.get("metadata", {})
    # ...while the cache pin still occupies top-level session_id
    assert payload.get("session_id") == _expected_sticky(_STICKY_CHAT_ID, _STICKY_SECRET)


def test_inbound_session_id_is_stripped(monkeypatch):
    monkeypatch.delenv("WEBUI_SECRET_KEY", raising=False)  # no secret -> sticky no-ops
    payload = _base_payload()
    payload["session_id"] = "client-spoofed"
    _apply_identifier_valves_to_payload(
        payload,
        valves=Pipe.Valves(),
        owui_metadata={"chat_id": _STICKY_CHAT_ID},
        owui_user_id="u1",
    )
    assert "session_id" not in payload  # client-supplied session_id is never forwarded


def test_cache_pin_survives_openrouter_filters(monkeypatch):
    from open_webui_openrouter_pipe.api.transforms import _filter_openrouter_chat_request

    monkeypatch.setenv("WEBUI_SECRET_KEY", _STICKY_SECRET)
    expected = _expected_sticky(_STICKY_CHAT_ID, _STICKY_SECRET)

    responses_payload = _base_payload()
    _apply_identifier_valves_to_payload(
        responses_payload,
        valves=Pipe.Valves(),
        owui_metadata={"chat_id": _STICKY_CHAT_ID},
        owui_user_id="u1",
    )
    assert responses_payload.get("session_id") == expected
    assert _filter_openrouter_request(responses_payload).get("session_id") == expected

    chat_payload = {
        "model": "openrouter/test",
        "messages": [{"role": "user", "content": "ping"}],
        "session_id": expected,
    }
    assert _filter_openrouter_chat_request(chat_payload).get("session_id") == expected



def test_end_user_id_source_default_sends_guid():
    valves = Pipe.Valves(SEND_END_USER_ID=True, SEND_CACHE_SESSION_ID=False)
    payload = _base_payload()
    _apply_identifier_valves_to_payload(
        payload,
        valves=valves,
        owui_metadata={},
        owui_user_id="guid-1",
        owui_user={"id": "guid-1", "email": "person@example.com", "name": "Person"},
    )
    assert payload["user"] == "guid-1"
    assert payload["metadata"]["user_id"] == "guid-1"


def test_end_user_id_source_email_sends_email_keeps_guid_metadata():
    valves = Pipe.Valves(
        SEND_END_USER_ID=True, END_USER_ID_SOURCE="email", SEND_CACHE_SESSION_ID=False,
    )
    payload = _base_payload()
    _apply_identifier_valves_to_payload(
        payload,
        valves=valves,
        owui_metadata={},
        owui_user_id="guid-1",
        owui_user={"id": "guid-1", "email": "person@example.com", "name": "Person"},
    )
    assert payload["user"] == "person@example.com"
    assert payload["metadata"]["user_id"] == "guid-1"


def test_end_user_id_source_name_sends_display_name():
    valves = Pipe.Valves(
        SEND_END_USER_ID=True, END_USER_ID_SOURCE="name", SEND_CACHE_SESSION_ID=False,
    )
    payload = _base_payload()
    _apply_identifier_valves_to_payload(
        payload,
        valves=valves,
        owui_metadata={},
        owui_user_id="guid-1",
        owui_user={"id": "guid-1", "email": "person@example.com", "name": "Person"},
    )
    assert payload["user"] == "Person"
    assert payload["metadata"]["user_id"] == "guid-1"


def test_end_user_id_source_email_falls_back_to_guid_when_missing():
    valves = Pipe.Valves(
        SEND_END_USER_ID=True, END_USER_ID_SOURCE="email", SEND_CACHE_SESSION_ID=False,
    )
    payload = _base_payload()
    _apply_identifier_valves_to_payload(
        payload,
        valves=valves,
        owui_metadata={},
        owui_user_id="guid-1",
        owui_user={"id": "guid-1", "email": "", "name": "Person"},
    )
    assert payload["user"] == "guid-1"

    payload_no_user_obj = _base_payload()
    _apply_identifier_valves_to_payload(
        payload_no_user_obj,
        valves=valves,
        owui_metadata={},
        owui_user_id="guid-1",
    )
    assert payload_no_user_obj["user"] == "guid-1"


def test_end_user_id_source_oversized_email_falls_back_to_guid():
    valves = Pipe.Valves(
        SEND_END_USER_ID=True, END_USER_ID_SOURCE="email", SEND_CACHE_SESSION_ID=False,
    )
    payload = _base_payload()
    _apply_identifier_valves_to_payload(
        payload,
        valves=valves,
        owui_metadata={},
        owui_user_id="guid-1",
        owui_user={"email": "x" * 5000},
    )
    assert payload["user"] == "guid-1"


def test_end_user_id_source_email_reads_attribute_objects():
    from types import SimpleNamespace

    valves = Pipe.Valves(
        SEND_END_USER_ID=True, END_USER_ID_SOURCE="email", SEND_CACHE_SESSION_ID=False,
    )
    payload = _base_payload()
    user_model = SimpleNamespace(id="guid-1", email="person@example.com", name="Person")
    _apply_identifier_valves_to_payload(
        payload,
        valves=valves,
        owui_metadata={},
        owui_user_id="guid-1",
        owui_user=user_model,
    )
    assert payload["user"] == "person@example.com"
    assert payload["metadata"]["user_id"] == "guid-1"


def test_end_user_id_source_name_reads_attribute_objects():
    from types import SimpleNamespace

    valves = Pipe.Valves(
        SEND_END_USER_ID=True, END_USER_ID_SOURCE="name", SEND_CACHE_SESSION_ID=False,
    )
    payload = _base_payload()
    _apply_identifier_valves_to_payload(
        payload,
        valves=valves,
        owui_metadata={},
        owui_user_id="guid-1",
        owui_user=SimpleNamespace(id="guid-1", email="e@example.com", name="Person"),
    )
    assert payload["user"] == "Person"
