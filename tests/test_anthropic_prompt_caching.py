import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe


def _last_input_text_cache_control(message: dict) -> dict | None:
    content = message.get("content")
    if not isinstance(content, list):
        return None
    for block in reversed(content):
        if not isinstance(block, dict):
            continue
        if block.get("type") != "input_text":
            continue
        return block.get("cache_control")
    return None


@pytest.mark.asyncio
async def test_anthropic_prompt_caching_inserts_breakpoints():
    pipe = Pipe()
    valves = pipe.valves.model_copy(
        update={
            "ENABLE_ANTHROPIC_PROMPT_CACHING": True,
            "ANTHROPIC_PROMPT_CACHE_TTL": "5m",
        }
    )

    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "user-1"},
        {"role": "assistant", "content": "assistant-1"},
        {"role": "user", "content": "user-2"},
    ]

    input_items = await pipe.transform_messages_to_input(
        messages,
        model_id="anthropic/claude-sonnet-4.5",
        valves=valves,
    )

    system_messages = [
        item for item in input_items
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "system"
    ]
    user_messages = [
        item for item in input_items
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "user"
    ]
    assistant_messages = [
        item for item in input_items
        if isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "assistant"
    ]

    assert system_messages
    assert len(user_messages) == 2
    assert assistant_messages

    expected = {"type": "ephemeral", "ttl": "5m"}
    assert _last_input_text_cache_control(system_messages[-1]) == expected
    assert _last_input_text_cache_control(user_messages[-1]) == expected
    assert _last_input_text_cache_control(user_messages[-2]) == expected
    assert _last_input_text_cache_control(assistant_messages[-1]) is None


@pytest.mark.asyncio
async def test_non_anthropic_models_do_not_insert_cache_control():
    pipe = Pipe()
    messages = [
        {"role": "system", "content": "SYSTEM"},
        {"role": "user", "content": "user-1"},
        {"role": "assistant", "content": "assistant-1"},
        {"role": "user", "content": "user-2"},
    ]

    input_items = await pipe.transform_messages_to_input(
        messages,
        model_id="openai/gpt-5.1",
    )

    assert not Pipe._input_contains_cache_control(input_items)


def test_strip_cache_control_from_input_removes_markers():
    payload = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "hello",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                }
            ],
        }
    ]

    assert Pipe._input_contains_cache_control(payload)
    Pipe._strip_cache_control_from_input(payload)
    assert not Pipe._input_contains_cache_control(payload)

