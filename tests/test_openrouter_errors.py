import json

from openrouter_responses_pipe.openrouter_responses_pipe import (
    ModelFamily,
    OpenRouterAPIError,
    _build_openrouter_api_error,
    _resolve_error_model_context,
)


def test_openrouter_api_error_includes_provider_and_request_details():
    raw_metadata = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "message": "prompt is too long: 1015918 tokens > 1000000 maximum",
        },
        "request_id": "req_011CVgpGEQMnFmYrjfT7Zh9h",
    }
    body = json.dumps(
        {
            "error": {
                "message": "Provider returned error",
                "code": 400,
                "metadata": {
                    "provider_name": "Anthropic",
                    "raw": json.dumps(raw_metadata),
                },
            },
            "user_id": "org_123",
        }
    )

    err = _build_openrouter_api_error(400, "Bad Request", body)

    assert isinstance(err, OpenRouterAPIError)
    assert err.provider == "Anthropic"
    assert err.upstream_type == "invalid_request_error"
    assert err.request_id == raw_metadata["request_id"]

    md = err.to_markdown()
    assert "### 🚫 Anthropic could not process your request." in md
    assert "### Error:" in md
    assert raw_metadata["request_id"] in md
    assert "prompt is too long" in md


def test_openrouter_api_error_handles_plain_text_payload():
    err = _build_openrouter_api_error(400, "Bad Request", "plain text body")

    md = err.to_markdown()
    assert "could not process your request" in md
    assert "middle-out option" in md


def test_openrouter_api_error_includes_moderation_metadata():
    body = json.dumps(
        {
            "error": {
                "message": "Input was flagged by moderation",
                "code": 400,
                "metadata": {
                    "provider_name": "OpenRouter",
                    "model_slug": "anthropic/claude-3",
                    "reasons": ["violence", "hate"],
                    "flagged_input": "Some violent text…",
                },
            }
        }
    )

    err = _build_openrouter_api_error(400, "Bad Request", body)
    assert err.model_slug == "anthropic/claude-3"
    assert err.moderation_reasons == ["violence", "hate"]
    assert err.flagged_input == "Some violent text…"

    md = err.to_markdown()
    assert "**Moderation reasons:**" in md
    assert "Some violent text" in md


def test_openrouter_error_markdown_accepts_model_label_and_diagnostics():
    err = OpenRouterAPIError(
        status=400,
        reason="Bad Request",
        provider="Anthropic",
        openrouter_message="This endpoint's maximum context length is 400000 tokens. However, you requested about 564659 tokens. Please reduce the length or use the \"middle-out\" transform.",
    )
    md = err.to_markdown(
        model_label="anthropic/claude-3",
        diagnostics=["- **Context window**: 200,000 tokens"],
        metrics={"context_limit": 400000, "max_output_tokens": 128000},
        fallback_model="anthropic/claude-3",
    )
    assert "### 🚫 Anthropic: anthropic/claude-3 could not process your request." in md
    assert "Model limits" in md
    assert "400,000" in md
    assert "128,000" in md


def test_resolve_error_model_context_uses_registry_spec():
    spec = {
        "full_model": {"name": "Claude 3", "context_length": 200000},
        "context_length": 200000,
        "max_completion_tokens": 4000,
    }
    ModelFamily.set_dynamic_specs({"anthropic.claude-3": spec})
    err = OpenRouterAPIError(status=400, reason="Bad Request")
    label, diagnostics, metrics = _resolve_error_model_context(
        err,
        normalized_model_id="anthropic.claude-3",
        api_model_id="anthropic/claude-3",
    )
    assert label == "Claude 3"
    assert any("4,000" in line for line in diagnostics)
    assert metrics["max_output_tokens"] == 4000
    ModelFamily.set_dynamic_specs({})


def test_custom_error_template_skips_lines_for_missing_values():
    err = OpenRouterAPIError(status=400, reason="Bad Request")
    template = "Line A\n- Req: {request_id}\nLine B\n- Provider: {provider}"
    rendered = err.to_markdown(template=template, diagnostics=[], metrics={})
    assert rendered == "Line A\nLine B"
