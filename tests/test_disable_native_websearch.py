from __future__ import annotations

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import (
    _apply_disable_native_websearch_to_payload,
)


def test_disable_native_websearch_removes_web_plugin() -> None:
    payload = {
        "model": "openai/gpt-5",
        "disable_native_websearch": True,
        "plugins": [
            {"id": "web", "max_results": 3},
            {"id": "other"},
        ],
    }

    _apply_disable_native_websearch_to_payload(payload)

    assert payload["plugins"] == [{"id": "other"}]
    assert "disable_native_websearch" not in payload


def test_disable_native_websearch_removes_plugins_key_when_empty() -> None:
    payload = {
        "model": "openai/gpt-5",
        "disable_native_websearch": "true",
        "plugins": [{"id": "web"}],
    }

    _apply_disable_native_websearch_to_payload(payload)

    assert "plugins" not in payload
    assert "disable_native_websearch" not in payload


def test_disable_native_websearch_false_keeps_web_plugin() -> None:
    payload = {
        "model": "openai/gpt-5",
        "disable_native_websearch": False,
        "plugins": [{"id": "web"}],
    }

    _apply_disable_native_websearch_to_payload(payload)

    assert payload["plugins"] == [{"id": "web"}]
    assert "disable_native_websearch" not in payload


def test_disable_native_websearch_removes_web_search_options_alias_key() -> None:
    payload = {
        "model": "openai/gpt-5",
        "disable_native_web_search": "1",
        "web_search_options": {"search_context_size": "low"},
    }

    _apply_disable_native_websearch_to_payload(payload)

    assert "web_search_options" not in payload
    assert "disable_native_web_search" not in payload

