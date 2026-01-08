from __future__ import annotations

from typing import Any

import pytest

from open_webui_openrouter_pipe.open_webui_openrouter_pipe import Pipe


@pytest.mark.asyncio
async def test_execute_function_calls_rejects_empty_string_args_when_required() -> None:
    pipe = Pipe()
    called: dict[str, Any] = {"count": 0}

    async def fetch_tool(**_kwargs: Any) -> str:
        called["count"] += 1
        return "ok"

    tools = {
        "fetch": {
            "spec": {
                "name": "fetch",
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            },
            "callable": fetch_tool,
        }
    }
    calls = [{"type": "function_call", "call_id": "call-1", "name": "fetch", "arguments": ""}]

    try:
        outputs = await pipe._execute_function_calls(calls, tools)
        assert called["count"] == 0
        assert outputs and outputs[0]["type"] == "function_call_output"
        assert "Missing tool arguments" in (outputs[0].get("output") or "")
    finally:
        await pipe.close()

