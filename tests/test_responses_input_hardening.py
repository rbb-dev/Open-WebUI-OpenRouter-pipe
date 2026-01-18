from __future__ import annotations

import pytest


def test_sanitize_request_input_strips_function_call_and_output_extras(pipe_instance):
    import open_webui_openrouter_pipe.pipe as pipe_mod

    body = pipe_mod.ResponsesBody.model_validate(
        {
            "model": "openrouter/test",
            "input": [
                {
                    "type": "function_call",
                    "id": "ulid-1",
                    "status": "completed",
                    "call_id": "call-1",
                    "name": "search_web",
                    "arguments": {"query": "x", "count": 1},
                },
                {
                    "type": "function_call_output",
                    "id": "ulid-2",
                    "status": "completed",
                    "call_id": "call-1",
                    "output": {"ok": True},
                },
            ],
            "stream": True,
        }
    )

    pipe_instance._sanitize_request_input(body)

    assert body.input == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "search_web",
            "arguments": '{"query": "x", "count": 1}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"ok": true}',
        },
    ]


def test_sanitize_request_input_falls_back_to_id_as_call_id(pipe_instance):
    import open_webui_openrouter_pipe.pipe as pipe_mod

    body = pipe_mod.ResponsesBody.model_validate(
        {
            "model": "openrouter/test",
            "input": [
                {
                    "type": "function_call",
                    "id": "tooluse_abc123",
                    "status": "completed",
                    "name": "search_web",
                    "arguments": "{}",
                }
            ],
        }
    )

    pipe_instance._sanitize_request_input(body)

    assert body.input == [
        {
            "type": "function_call",
            "call_id": "tooluse_abc123",
            "name": "search_web",
            "arguments": "{}",
        }
    ]

