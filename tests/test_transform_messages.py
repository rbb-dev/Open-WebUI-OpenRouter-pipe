from __future__ import annotations

import asyncio

import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import (
    ModelFamily,
    Pipe,
    _classify_function_call_artifacts,
    _serialize_marker,
    generate_item_id,
)


def _assistant_message_with_markers(*markers: str) -> str:
    parts = ["Assistant note"]
    parts.extend(markers)
    return "\n".join(parts) + "\n"


def _run_transform(messages, artifacts):
    pipe = Pipe()

    async def loader(chat_id, message_id, ulids):
        return {ulid: artifacts.get(ulid) for ulid in ulids if ulid in artifacts}

    return asyncio.run(
        pipe.transform_messages_to_input(
            messages,
            chat_id="chat-1",
            openwebui_model_id="model-1",
            artifact_loader=loader,
            valves=pipe.valves,
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


def test_classify_function_call_artifacts_partitions_sets():
    payloads = {
        "m-valid-call": {"type": "function_call", "call_id": "shared"},
        "m-valid-output": {"type": "function_call_output", "call_id": "shared"},
        "m-orphan-call": {"type": "function_call", "call_id": "only_call"},
        "m-orphan-output": {"type": "function_call_output", "call_id": "only_output"},
        "m-irrelevant": {"type": "reasoning"},
    }
    valid, orphan_calls, orphan_outputs = _classify_function_call_artifacts(payloads)
    assert valid == {"shared"}
    assert orphan_calls == {"only_call"}
    assert orphan_outputs == {"only_output"}


@pytest.fixture(autouse=True)
def _reset_model_specs():
    ModelFamily.set_dynamic_specs({})
    yield
    ModelFamily.set_dynamic_specs({})


@pytest.mark.asyncio
async def test_transform_limits_user_images(monkeypatch):
    pipe = Pipe()
    captured_status: list[str] = []

    async def fake_emitter(event):
        if event.get("type") == "status":
            captured_status.append(event["data"]["description"])

    async def fake_inline(url, chunk_size, max_bytes):
        file_id = url.split("/")[-2]
        return f"data:image/png;base64,{file_id}"

    monkeypatch.setattr(pipe, "_inline_internal_file_url", fake_inline)
    ModelFamily.set_dynamic_specs({"vision-model": {"features": {"vision"}}})
    valves = pipe.valves.model_copy(update={"MAX_INPUT_IMAGES_PER_REQUEST": 1})
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": "/api/v1/files/img-a/content"},
                {"type": "image_url", "image_url": "/api/v1/files/img-b/content"},
            ],
        }
    ]
    transformed = await pipe.transform_messages_to_input(
        messages,
        model_id="vision-model",
        valves=valves,
        event_emitter=fake_emitter,
    )
    content = transformed[0]["content"]
    assert len(content) == 1
    assert content[0]["image_url"].endswith("img-a")
    assert any("Dropped" in status for status in captured_status)


@pytest.mark.asyncio
async def test_transform_falls_back_to_assistant_images(monkeypatch):
    pipe = Pipe()
    async def fake_inline(url, chunk_size, max_bytes):
        file_id = url.split("/")[-2]
        return f"data:image/png;base64,{file_id}"
    monkeypatch.setattr(pipe, "_inline_internal_file_url", fake_inline)
    ModelFamily.set_dynamic_specs({"vision-model": {"features": {"vision"}}})
    messages = [
        {
            "role": "assistant",
            "content": "![img](/api/v1/files/assistant-img/content)",
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "please edit"}],
        },
    ]
    transformed = await pipe.transform_messages_to_input(
        messages,
        model_id="vision-model",
        valves=pipe.valves,
    )
    content = transformed[-1]["content"]
    assert any(
        block.get("type") == "input_image"
        and block.get("image_url", "").endswith("assistant-img")
        for block in content
    )


@pytest.mark.asyncio
async def test_transform_respects_user_turn_only_selection(monkeypatch):
    pipe = Pipe()

    async def fake_inline(url, chunk_size, max_bytes):
        file_id = url.split("/")[-2]
        return f"data:image/png;base64,{file_id}"

    monkeypatch.setattr(pipe, "_inline_internal_file_url", fake_inline)
    ModelFamily.set_dynamic_specs({"vision-model": {"features": {"vision"}}})
    valves = pipe.valves.model_copy(update={"IMAGE_INPUT_SELECTION": "user_turn_only"})
    messages = [
        {
            "role": "assistant",
            "content": "![img](/api/v1/files/assistant-img/content)",
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "touch up"}],
        },
    ]
    transformed = await pipe.transform_messages_to_input(
        messages,
        model_id="vision-model",
        valves=valves,
    )
    content = transformed[-1]["content"]
    assert all(block.get("type") != "input_image" for block in content)


@pytest.mark.asyncio
async def test_transform_skips_images_when_model_lacks_vision(monkeypatch):
    pipe = Pipe()
    captured_status: list[str] = []

    async def fake_emitter(event):
        if event.get("type") == "status":
            captured_status.append(event["data"]["description"])

    async def fake_inline(url, chunk_size, max_bytes):
        return "data:image/png;base64,test"

    monkeypatch.setattr(pipe, "_inline_internal_file_url", fake_inline)
    ModelFamily.set_dynamic_specs({"text-only": {"features": set()}})
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": "/api/v1/files/img-a/content"}],
        }
    ]
    transformed = await pipe.transform_messages_to_input(
        messages,
        model_id="text-only",
        valves=pipe.valves,
        event_emitter=fake_emitter,
    )
    content = transformed[0]["content"]
    assert all(block.get("type") != "input_image" for block in content)
    assert any("does not accept image inputs" in status for status in captured_status)


@pytest.mark.asyncio
async def test_transform_preserves_system_and_developer_message_text_exactly():
    pipe = Pipe()
    messages = [
        {"role": "system", "content": "  keep leading\nand trailing  \n"},
        {
            "role": "developer",
            "content": [
                {"type": "text", "text": "dev  "},
                "  raw\n",
            ],
        },
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]

    transformed = await pipe.transform_messages_to_input(messages, valves=pipe.valves)
    assert transformed[0] == {
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": "  keep leading\nand trailing  \n"}],
    }
    assert transformed[1] == {
        "type": "message",
        "role": "developer",
        "content": [
            {"type": "input_text", "text": "dev  "},
            {"type": "input_text", "text": "  raw\n"},
        ],
    }
