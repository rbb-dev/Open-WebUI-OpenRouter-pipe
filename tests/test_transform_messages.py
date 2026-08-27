# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportRedeclaration=false, reportIncompatibleMethodOverride=false, reportGeneralTypeIssues=false, reportSelfClsParameterName=false, reportCallIssue=false, reportOptionalIterable=false
from __future__ import annotations

import asyncio

import pytest

from open_webui_openrouter_pipe import (
    ModelFamily,
    Pipe,
    _classify_function_call_artifacts,
    _serialize_marker,
    generate_item_id,
)
from open_webui_openrouter_pipe.requests.transformer import transform_messages_to_input
from open_webui_openrouter_pipe.storage.owui_files import InlinedFile


def _assistant_message_with_markers(*markers: str) -> str:
    parts = ["Assistant note"]
    parts.extend(markers)
    return "\n".join(parts) + "\n"


def _run_transform(messages, artifacts):
    pipe = Pipe()

    async def loader(chat_id, message_id, ulids):
        return {ulid: artifacts.get(ulid) for ulid in ulids if ulid in artifacts}

    async def _transform_and_close():
        try:
            return await transform_messages_to_input(pipe,
                messages,
                chat_id="chat-1",
                openwebui_model_id="model-1",
                artifact_loader=loader,
                valves=pipe.valves,
            )
        finally:
            await pipe.close()

    return asyncio.run(_transform_and_close())


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
async def test_transform_limits_user_images(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    captured_status: list[str] = []

    async def fake_emitter(event):
        if event.get("type") == "status":
            captured_status.append(event["data"]["description"])

    async def fake_inline(file_id, *, chunk_size=None, max_bytes=None, user=None):
        return InlinedFile(data_url=f"data:image/png;base64,{file_id}", filename=f"{file_id}.png")

    monkeypatch.setattr(pipe._file_gateway, "inline_owui_file_id", fake_inline)
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
    transformed = await transform_messages_to_input(pipe,
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
async def test_transform_falls_back_to_assistant_images(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    async def fake_inline(file_id, *, chunk_size=None, max_bytes=None, user=None):
        return InlinedFile(data_url=f"data:image/png;base64,{file_id}", filename=f"{file_id}.png")
    monkeypatch.setattr(pipe._file_gateway, "inline_owui_file_id", fake_inline)
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
    transformed = await transform_messages_to_input(pipe,
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
async def test_transform_rehydration_drops_uninlineable_assistant_images(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async

    async def fake_inline(file_id, *, chunk_size=None, max_bytes=None, user=None):
        if file_id == "missing-img":
            return None
        return InlinedFile(data_url=f"data:image/png;base64,{file_id}", filename=f"{file_id}.png")

    monkeypatch.setattr(pipe._file_gateway, "inline_owui_file_id", fake_inline)
    ModelFamily.set_dynamic_specs({"vision-model": {"features": {"vision"}}})
    messages = [
        {
            "role": "assistant",
            "content": "\n".join(
                [
                    "Here you go:",
                    "![img](/api/v1/files/missing-img/content)",
                    "![img](/api/v1/files/ok-img/content)",
                ]
            ),
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "please edit"}],
        },
    ]
    transformed = await transform_messages_to_input(pipe,
        messages,
        model_id="vision-model",
        valves=pipe.valves,
    )
    content = transformed[-1]["content"]
    images = [b for b in content if isinstance(b, dict) and b.get("type") == "input_image"]
    assert len(images) == 1
    assert images[0].get("image_url", "").startswith("data:image/png;base64,ok-img")


@pytest.mark.asyncio
async def test_transform_respects_user_turn_only_selection(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async

    async def fake_inline(file_id, *, chunk_size=None, max_bytes=None, user=None):
        return InlinedFile(data_url=f"data:image/png;base64,{file_id}", filename=f"{file_id}.png")

    monkeypatch.setattr(pipe._file_gateway, "inline_owui_file_id", fake_inline)
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
    transformed = await transform_messages_to_input(pipe,
        messages,
        model_id="vision-model",
        valves=valves,
    )
    content = transformed[-1]["content"]
    assert all(block.get("type") != "input_image" for block in content)


@pytest.mark.asyncio
async def test_transform_skips_images_when_model_lacks_vision(monkeypatch, pipe_instance_async):
    pipe = pipe_instance_async
    captured_status: list[str] = []

    async def fake_emitter(event):
        if event.get("type") == "status":
            captured_status.append(event["data"]["description"])

    async def fake_inline(_file_id, *, chunk_size=None, max_bytes=None, user=None):
        return InlinedFile(data_url="data:image/png;base64,test", filename="test.png")

    monkeypatch.setattr(pipe._file_gateway, "inline_owui_file_id", fake_inline)
    ModelFamily.set_dynamic_specs({"text-only": {"features": set()}})
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": "/api/v1/files/img-a/content"}],
        }
    ]
    transformed = await transform_messages_to_input(pipe,
        messages,
        model_id="text-only",
        valves=pipe.valves,
        event_emitter=fake_emitter,
    )
    content = transformed[0]["content"]
    assert all(block.get("type") != "input_image" for block in content)
    assert any("does not accept image inputs" in status for status in captured_status)


@pytest.mark.asyncio
async def test_transform_preserves_system_and_developer_message_text_exactly(pipe_instance_async):
    pipe = pipe_instance_async
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

    transformed = await transform_messages_to_input(pipe,messages, valves=pipe.valves)
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


def test_transform_survives_a_non_dict_content_block():
    """A truthy non-dict content part must not abort the whole request."""
    messages = [
        {
            "role": "user",
            "content": ["a bare string part", {"type": "text", "text": "a real block"}],
        }
    ]

    result = _run_transform(messages, {})

    texts = [
        part.get("text")
        for item in result
        if isinstance(item, dict)
        for part in (item.get("content") or [])
        if isinstance(part, dict)
    ]
    assert "a bare string part" in texts, result
    assert "a real block" in texts, result


def test_transform_drops_unrenderable_content_block_without_aborting():
    """A non-dict, non-str part is dropped; the rest of the message still transforms."""
    messages = [
        {
            "role": "user",
            "content": [12345, {"type": "text", "text": "kept"}],
        }
    ]

    result = _run_transform(messages, {})

    parts = [
        part
        for item in result
        if isinstance(item, dict)
        for part in (item.get("content") or [])
    ]
    assert all(isinstance(part, dict) for part in parts), (
        f"a raw non-dict block reached the outbound payload: {parts!r}"
    )
    assert 12345 not in parts, f"the unrenderable block was forwarded verbatim: {parts!r}"
    assert [p.get("text") for p in parts] == ["kept"], parts


@pytest.mark.parametrize(
    "content",
    [
        {"type": "text", "text": "the real user question"},
        {"content": "the real user question"},
        {"type": "text", "text": None, "content": "the real user question"},
        {"type": "text", "text": 123, "content": "the real user question"},
    ],
    ids=["text-key", "content-key", "text-key-is-none", "text-key-is-not-a-string"],
)
def test_transform_preserves_a_dict_shaped_user_content(content):
    """A dict `content` must yield the user's text, not the dict's keys.

    One shape is satisfied by reading one key: with only the `text` case, deleting the
    `content` fallback left the whole suite green while a user message nested under
    that key became an empty block list — the model then answers a turn it cannot see.

    The third id is the one that separates `not isinstance(text_val, str)` from
    `text_val is None`; a `text` key holding None is what a partially-built message
    dict looks like.
    """
    messages = [{"role": "user", "content": content}]

    result = _run_transform(messages, {})

    texts = [
        part.get("text")
        for item in result
        if isinstance(item, dict)
        for part in (item.get("content") or [])
        if isinstance(part, dict)
    ]
    assert texts == ["the real user question"], (
        f"the user's question was replaced by the container's keys: {texts!r}"
    )


def test_transform_drops_a_set_shaped_user_content():
    """A set `content` carries no usable text and must not leak its members."""
    messages = [{"role": "user", "content": {"a-set-member"}}]

    result = _run_transform(messages, {})

    texts = [
        part.get("text")
        for item in result
        if isinstance(item, dict)
        for part in (item.get("content") or [])
        if isinstance(part, dict)
    ]
    assert "a-set-member" not in texts, f"set members leaked into the payload: {texts!r}"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("url", "allow", "hosts", "kept"),
    [
        ("http://images.example.test/a.png", False, "", False),
        ("http://images.example.test/a.png", True, "images.example.test", True),
        ("https://images.example.test/a.png", False, "", True),
        ("HTTP://images.example.test/a.png", False, "", False),
        ("HTTP://images.example.test/a.png", True, "images.example.test", True),
        ("HTTPS://images.example.test/a.png", False, "", True),
    ],
)
async def test_an_insecure_http_image_url_is_dropped_unless_allowed(
    pipe_instance_async, url, allow, hosts, kept
):
    """The enforcement, not the predicate.

    `test_http_blocked_by_default` exercises `_is_insecure_http_allowed` in isolation.
    Nothing checked that any caller consults it: neutering all six sites in
    transform_messages_to_input -- `and False and pipe._multimodal_handler.
    _is_insecure_http_allowed(url)` -- left the whole suite green, and the block branch
    had never executed.

    Three rows on purpose. The blocking row alone is satisfied by a production edit
    that drops every image; the allowed and https rows are what make it a gate rather
    than a wall.

    Each row runs twice, once with the scheme upper-cased. Schemes are case-insensitive
    (RFC 3986 3.1) but every site here compared raw prefixes, so `HTTP://` skipped the
    gate entirely and the lower-case rows alone reported the whole thing green.
    """
    pipe = pipe_instance_async
    pipe.valves.ALLOW_INSECURE_HTTP = allow
    pipe.valves.ALLOW_INSECURE_HTTP_HOSTS = hosts
    ModelFamily.set_dynamic_specs({"vision-model": {"features": {"vision"}}})

    async def _emitter(_event):
        return None

    transformed = await transform_messages_to_input(
        pipe,
        [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}],
        model_id="vision-model",
        valves=pipe.valves,
        event_emitter=_emitter,
    )

    blocks = transformed[0]["content"] if transformed else []
    images = [
        b for b in blocks
        if isinstance(b, dict) and b.get("type") in {"input_image", "image_url"}
    ]
    assert bool(images) is kept, (
        f"url={url} ALLOW_INSECURE_HTTP={allow} hosts={hosts!r}: expected the image to "
        f"be {'kept' if kept else 'dropped'}, got blocks={blocks!r}. A plaintext URL "
        "reaching the outbound payload is fetched by OpenRouter over the network the "
        "operator disabled."
    )
