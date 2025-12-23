from __future__ import annotations

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import _apply_model_fallback_to_payload


def test_model_fallback_csv_to_models_array() -> None:
    payload = {
        "model": "openai/gpt-5",
        "model_fallback": " openai/gpt-5.1 , , anthropic/claude-sonnet-4.5,openai/gpt-5.1 ",
    }
    _apply_model_fallback_to_payload(payload)
    assert payload["models"] == ["openai/gpt-5.1", "anthropic/claude-sonnet-4.5"]
    assert "model_fallback" not in payload


def test_model_fallback_merges_with_existing_models_list() -> None:
    payload = {
        "model": "openai/gpt-5",
        "models": ["anthropic/claude-sonnet-4.5", "openai/gpt-5.1"],
        "model_fallback": "openai/gpt-5.1,google/gemini-2.5-pro",
    }
    _apply_model_fallback_to_payload(payload)
    assert payload["models"] == [
        "anthropic/claude-sonnet-4.5",
        "openai/gpt-5.1",
        "google/gemini-2.5-pro",
    ]

