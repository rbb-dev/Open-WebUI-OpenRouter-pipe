from __future__ import annotations

from open_webui_openrouter_pipe import (
    _filter_openrouter_request,
    _responses_payload_to_chat_completions_payload,
)


def test_filter_openrouter_request_translates_response_format_to_text_format() -> None:
    payload = {
        "model": "openai/gpt-5",
        "input": [],
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "demo", "schema": {"type": "object"}},
        },
    }
    filtered = _filter_openrouter_request(payload)
    assert "response_format" not in filtered
    assert filtered["text"]["format"] == {
        "type": "json_schema",
        "name": "demo",
        "schema": {"type": "object"},
    }


def test_filter_openrouter_request_prefers_existing_text_format() -> None:
    payload = {
        "model": "openai/gpt-5",
        "input": [],
        "text": {"format": {"type": "json_object"}},
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "demo", "schema": {"type": "object"}},
        },
    }
    filtered = _filter_openrouter_request(payload)
    assert "response_format" not in filtered
    assert filtered["text"]["format"] == {"type": "json_object"}


def test_filter_openrouter_request_drops_invalid_text_format() -> None:
    payload = {
        "model": "openai/gpt-5",
        "input": [],
        "text": {"format": {"type": "nope"}},
    }
    filtered = _filter_openrouter_request(payload)
    assert "text" not in filtered


def test_responses_payload_to_chat_maps_text_format_to_response_format() -> None:
    payload = {
        "model": "openai/gpt-5",
        "stream": False,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ],
        "text": {"format": {"type": "json_object"}},
    }
    chat = _responses_payload_to_chat_completions_payload(payload)
    assert chat["response_format"] == {"type": "json_object"}


def test_responses_payload_to_chat_prefers_response_format_over_text_format() -> None:
    payload = {
        "model": "openai/gpt-5",
        "stream": False,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hi"}],
            }
        ],
        "response_format": {"type": "json_object"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "demo",
                "schema": {"type": "object"},
            }
        },
    }
    chat = _responses_payload_to_chat_completions_payload(payload)
    assert chat["response_format"] == {"type": "json_object"}

