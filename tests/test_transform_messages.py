from __future__ import annotations

import asyncio

from openrouter_responses_pipe.openrouter_responses_pipe import (
    ResponsesBody,
    _serialize_marker,
    generate_item_id,
)


def _assistant_message_with_markers(*markers: str) -> str:
    parts = ["Assistant note"]
    parts.extend(markers)
    return "\n".join(parts) + "\n"


def _run_transform(messages, artifacts):
    async def loader(chat_id, message_id, ulids):
        return {ulid: artifacts.get(ulid) for ulid in ulids if ulid in artifacts}

    return asyncio.run(
        ResponsesBody(model="test-model", input=[]).transform_messages_to_input(
            messages,
            chat_id="chat-1",
            openwebui_model_id="model-1",
            artifact_loader=loader,
        )
    )


def test_transform_messages_skips_orphaned_function_calls():
    call_ulid = generate_item_id()
    messages = [
        {
            "role": "assistant",
            "content": _assistant_message_with_markers(_serialize_marker(call_ulid)),
        }
    ]

    artifacts = {
        call_ulid: {
            "type": "function_call",
            "call_id": "tool_missing_output",
            "name": "tool_fetch_json_post",
            "arguments": "{}",
        }
    }

    result = _run_transform(messages, artifacts)
    tool_entries = [item for item in result if item.get("type") != "message"]
    assert all(entry.get("type") != "function_call" for entry in tool_entries)


def test_transform_messages_keeps_complete_function_call_pairs():
    call_ulid = generate_item_id()
    output_ulid = generate_item_id()
    messages = [
        {
            "role": "assistant",
            "content": _assistant_message_with_markers(
                _serialize_marker(call_ulid), _serialize_marker(output_ulid)
            ),
        }
    ]
    artifacts = {
        call_ulid: {
            "type": "function_call",
            "call_id": "tool_ok",
            "name": "tool_fetch_json_post",
            "arguments": "{}",
        },
        output_ulid: {
            "type": "function_call_output",
            "call_id": "tool_ok",
            "output": "done",
        },
    }

    result = _run_transform(messages, artifacts)
    tool_entries = [item for item in result if item.get("type") != "message"]
    assert [entry.get("type") for entry in tool_entries] == [
        "function_call",
        "function_call_output",
    ]
    assert all(entry.get("call_id") == "tool_ok" for entry in tool_entries)


def test_transform_messages_skips_orphaned_function_call_outputs():
    output_ulid = generate_item_id()
    messages = [
        {
            "role": "assistant",
            "content": _assistant_message_with_markers(_serialize_marker(output_ulid)),
        }
    ]
    artifacts = {
        output_ulid: {
            "type": "function_call_output",
            "call_id": "tool_without_call",
            "output": "data",
        }
    }

    result = _run_transform(messages, artifacts)
    tool_entries = [item for item in result if item.get("type") != "message"]
    assert all(entry.get("type") != "function_call_output" for entry in tool_entries)
