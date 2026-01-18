from __future__ import annotations

from open_webui_openrouter_pipe import _filter_openrouter_request


def test_filter_openrouter_request_forwards_numeric_top_k() -> None:
    payload = {"model": "openai/gpt-5", "input": [], "top_k": 50}
    filtered = _filter_openrouter_request(payload)
    assert filtered["top_k"] == 50.0


def test_filter_openrouter_request_parses_string_top_k() -> None:
    payload = {"model": "openai/gpt-5", "input": [], "top_k": " 50 "}
    filtered = _filter_openrouter_request(payload)
    assert filtered["top_k"] == 50.0


def test_filter_openrouter_request_drops_invalid_top_k() -> None:
    payload = {"model": "openai/gpt-5", "input": [], "top_k": "nope"}
    filtered = _filter_openrouter_request(payload)
    assert "top_k" not in filtered

