import pytest


def test_chat_tools_to_responses_tools_converts_function_shape():
    import open_webui_openrouter_pipe.pipe as pipe_mod

    converted = pipe_mod._chat_tools_to_responses_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                        "required": ["q"],
                    },
                },
            }
        ]
    )

    assert converted == [
        {
            "type": "function",
            "name": "search_web",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        }
    ]


@pytest.mark.asyncio
async def test_responsesbody_from_completions_keeps_and_normalizes_tools():
    import open_webui_openrouter_pipe.pipe as pipe_mod

    class DummyTransformer:
        async def transform_messages_to_input(self, messages, **kwargs):  # noqa: ANN001
            return [{"role": "user", "content": "hi"}]

    completions = pipe_mod.CompletionsBody.model_validate(
        {
            "model": "openrouter/test",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_current_timestamp",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
    )

    rb = await pipe_mod.ResponsesBody.from_completions(
        completions_body=completions,
        transformer_context=DummyTransformer(),
    )

    assert rb.tools == [
        {
            "type": "function",
            "name": "get_current_timestamp",
            "parameters": {"type": "object", "properties": {}},
        }
    ]

