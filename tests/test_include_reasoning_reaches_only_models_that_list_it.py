"""The pipe's own `include_reasoning` goes only to a model whose catalog entry lists it (T369); reasoning fields the chat itself carries go out as Open WebUI sends them (O2).

Live, 2026-09-25, Open WebUI 0.11.4, pipe at 6c63a68: four chats with `openai/gpt-4.1-mini`
failed with HTTP 400 "Provider returned error"; the provider (Azure) said "Unknown parameter:
'include_reasoning'." The same requests with `google/gemini-2.5-flash` succeeded.

`gpt-4.1-mini`'s catalog row lists neither `reasoning` nor `include_reasoning` (132 of 460
models on 2026-09-25), and the pipe's "supports neither" branches wrote
`include_reasoning = False` anyway, believing it to be an off switch. OpenRouter documents the
flag as a deprecated alias for `reasoning.exclude`: it hides reasoning, it does not disable it,
so for a model that has no reasoning it carries nothing and can only be rejected. `False` is not
`None`, so it survived `model_dump(exclude_none=True)` and the request filters and went out on
every path. The Gemini 2.5 thinking translation wrote the same `False` for
`google/gemini-2.5-flash-image`, a Gemini 2.5 model whose row also lists neither, even with
reasoning switched off, and so did the retry that follows a provider's thinking-mismatch 400.

Every test drives `Pipe.pipe()` with only HTTP stubbed and asserts on the JSON the pipe POSTed
to OpenRouter, the only place the defect is visible. The real rows are copied from OpenRouter's
/models dump of 2026-09-12 (unchanged on 2026-09-25). Two rows are synthetic, because no model
in either dump lists exactly one of the two parameters: a legacy-only model, whose branch still
ships and is the branch a careless fix breaks, and a Gemini 2.5 model that lists `reasoning`
but not `include_reasoning`, the only shape that tells a gate on `include_reasoning` from a gate
on `reasoning`, which agree on every real row.

Each "absent" arm has a "present" twin on the same path, so neither a pipe that sends the flag
to everyone nor one that sends it to no one can satisfy the file, and every exact value is
asserted under two different settings, so a hardcoded constant cannot either.

The leak the removed valve let through, measured on this harness before the removal landed.
`ENABLE_REASONING` was a request gate, not a display control, and its early return sat above
the pass that filters a chat's own reasoning fields by the model's catalog row, so a valve-off
install sent whatever the chat carried to models that reject it: `openai/gpt-4.1-mini` with
the valve off and a chat `reasoning_effort: "high"` put `{"reasoning": {"effort": "high"}}` on
the wire against a row listing no reasoning parameter at all, and `google/gemini-2.5-flash-image`
carried the chat's own reasoning the same way. The switch also failed to hold on the owned-task
branch, which ran the task pass ungated, so an install with the switch off already paid for
thinking on every new saved, non-temporary chat with title auto-generation on. There is no
switch now: `REASONING_EFFORT` and `GEMINI_THINKING_BUDGET` are the surviving levers, and both
still ask for `{"effort": "none"}` rather than dropping the request's reasoning.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
from aioresponses import CallbackResult, aioresponses

from open_webui_openrouter_pipe import Pipe
from open_webui_openrouter_pipe.core.config import EncryptedStr
from open_webui_openrouter_pipe.models.registry import ModelFamily, sanitize_model_id
from tests.test_request_orchestrator import _consume_stream, _smart_callback

_BASE = "https://openrouter.ai/api/v1"

# Real rows: `name`, `supported_parameters` and modalities verbatim from /api/v1/models, 2026-09-12.
_GPT_41_MINI = {
    "id": "openai/gpt-4.1-mini",
    "name": "OpenAI: GPT-4.1 Mini",
    "supported_parameters": [
        "max_completion_tokens", "max_tokens", "response_format", "seed",
        "structured_outputs", "temperature", "tool_choice", "tools", "top_p",
    ],
    "architecture": {"input_modalities": ["image", "text", "file"], "output_modalities": ["text"]},
}
# Named like the Gemini 2.5 thinking family, so it reaches the Gemini translation, and lists
# neither parameter.
_GEMINI_25_FLASH_IMAGE = {
    "id": "google/gemini-2.5-flash-image",
    "name": "Google: Nano Banana (Gemini 2.5 Flash Image)",
    "supported_parameters": [
        "max_tokens", "response_format", "seed", "stop", "structured_outputs", "temperature", "top_p",
    ],
    "architecture": {"input_modalities": ["image", "text"], "output_modalities": ["image", "text"]},
}
_GEMINI_25_FLASH = {
    "id": "google/gemini-2.5-flash",
    "name": "Google: Gemini 2.5 Flash",
    "supported_parameters": [
        "include_reasoning", "max_tokens", "reasoning", "response_format", "seed", "stop",
        "structured_outputs", "temperature", "tool_choice", "tools", "top_p",
    ],
    "architecture": {"input_modalities": ["file", "image", "text", "audio", "video"], "output_modalities": ["text"]},
}
_GPT_5_MINI = {
    "id": "openai/gpt-5-mini",
    "name": "OpenAI: GPT-5 Mini",
    "supported_parameters": [
        "include_reasoning", "max_completion_tokens", "max_tokens", "reasoning", "reasoning_effort",
        "response_format", "seed", "structured_outputs", "tool_choice", "tools",
    ],
    "architecture": {"input_modalities": ["text", "image", "file"], "output_modalities": ["text"]},
}
_CLAUDE_SONNET_46 = {
    "id": "anthropic/claude-sonnet-4.6",
    "name": "Anthropic: Claude Sonnet 4.6",
    "supported_parameters": [
        "include_reasoning", "max_completion_tokens", "max_tokens", "reasoning", "reasoning_effort",
        "response_format", "stop", "structured_outputs", "temperature", "tool_choice", "tools",
        "top_k", "top_p", "verbosity",
    ],
    "architecture": {"input_modalities": ["text", "image", "file"], "output_modalities": ["text"]},
}
_FUSION = {
    "id": "openrouter/fusion",
    "name": "OpenRouter: Fusion",
    "supported_parameters": [],
    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
}
# Synthetic: see the module docstring.
_LEGACY_ONLY = {
    "id": "acme/legacy-thinker",
    "name": "Acme: Legacy Thinker",
    "supported_parameters": ["include_reasoning", "max_tokens", "temperature"],
    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
}
_GEMINI_25_REASONING_ONLY = {
    "id": "google/gemini-2.5-synthetic-reasoning-only",
    "name": "Synthetic: Gemini 2.5, reasoning only",
    "supported_parameters": ["max_tokens", "reasoning", "temperature"],
    "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
}
_CATALOG = [
    _GPT_41_MINI, _GEMINI_25_FLASH_IMAGE, _GEMINI_25_FLASH, _GPT_5_MINI, _CLAUDE_SONNET_46,
    _FUSION, _LEGACY_ONLY, _GEMINI_25_REASONING_ONLY,
]

_RESPONSES_UNSUPPORTED = {"message": "This model does not support the Responses API", "code": "unsupported_endpoint"}
def _rejection(error: dict[str, Any]) -> CallbackResult:
    return CallbackResult(
        status=400,
        body=json.dumps({"error": error}),
        headers={"Content-Type": "application/json"},
    )


async def _sent(
    model: str,
    *,
    stream: bool = True,
    task: str | None = None,
    valves: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    responses_unsupported: bool = False,
    reject_first: dict[str, Any] | None = None,
    catalog: list[dict[str, Any]] | None = None,
    tools: dict[str, Any] | None = None,
    user_valves: dict[str, Any] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Drive one real request through `Pipe.pipe()`; return (endpoint, JSON body) for every
    POST that left the pipe, in order."""
    return (await _sent_shown(model, stream=stream, task=task, valves=valves, extra_body=extra_body,
                              responses_unsupported=responses_unsupported, reject_first=reject_first,
                              catalog=catalog, tools=tools, user_valves=user_valves))[0]


async def _sent_shown(
    model: str,
    *,
    stream: bool = True,
    task: str | None = None,
    valves: dict[str, Any] | None = None,
    extra_body: dict[str, Any] | None = None,
    responses_unsupported: bool = False,
    reject_first: dict[str, Any] | None = None,
    catalog: list[dict[str, Any]] | None = None,
    tools: dict[str, Any] | None = None,
    user_valves: dict[str, Any] | None = None,
) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    """As `_sent`, and the text the person would read: every emitter payload plus a
    non-streamed return, the way `_t379_run` in test_task_model_adapter.py collects it.

    `user_valves` is the route to a stored user valve row. `Pipe._stored_user_valves` re-reads
    the row from Open WebUI whenever a `user_id` is present, so the `Pipe.UserValves()` instance
    passed in `__user__` is discarded and the stub returns `{}`; patching
    `Functions.get_user_valves_by_id_and_user_id` is the faithful way to reach it.

    The metadata carries a `chat_id` and a `message_id` because every assertion here reads a
    card off a reply, and a card is what a chat gets: a caller with neither key has no chat
    to write one into, so the pipe hands it the provider rejection as an HTTP error instead
    (`requests/orchestrator.py`). The tests that drive that leg pass no such keys on purpose.
    """
    import open_webui.models.functions as owui_functions


    pipe = Pipe()
    sent: list[tuple[str, dict[str, Any]]] = []
    shown: list[str] = []
    answer = _smart_callback([], "OK")

    def _record(url, **kwargs):
        endpoint = "chat/completions" if str(url).endswith("/chat/completions") else "responses"
        sent.append((endpoint, copy.deepcopy(kwargs.get("json") or {})))
        if reject_first is not None and len(sent) == 1:
            return _rejection(reject_first)
        if responses_unsupported and endpoint == "responses":
            return _rejection(_RESPONSES_UNSUPPORTED)
        return answer(url, **kwargs)

    async def _emit(event) -> None:
        payload = getattr(event, "data", event)
        if isinstance(payload, dict):
            payload = payload.get("content")
        if isinstance(payload, str):
            shown.append(payload)

    original_row = None
    if user_valves is not None:
        original_row = owui_functions.Functions.get_user_valves_by_id_and_user_id

        async def _stored_row(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return user_valves

        owui_functions.Functions.get_user_valves_by_id_and_user_id = _stored_row  # type: ignore[method-assign]
    try:
        pipe.valves.API_KEY = EncryptedStr("test-api-key")
        pipe.valves.BASE_URL = _BASE
        for name, value in (valves or {}).items():
            setattr(pipe.valves, name, value)
        with aioresponses() as http:
            http.post(f"{_BASE}/responses", callback=_record, repeat=True)
            http.post(f"{_BASE}/chat/completions", callback=_record, repeat=True)
            http.get(f"{_BASE}/models", payload={"data": catalog or _CATALOG}, repeat=True)
            http.get(f"{_BASE}/endpoints/zdr", payload={"data": []}, repeat=True)
            result = await pipe.pipe(
                body={
                    "model": model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": stream,
                    **(extra_body or {}),
                },
                __user__={"id": "u1", "valves": Pipe.UserValves()},
                __request__=None,
                __event_emitter__=_emit,
                __event_call__=None,
                __metadata__={"chat_id": "chat-1", "message_id": "message-1", "model": {"id": model}},
                __tools__=tools,
                __task__=task,
                __task_body__=None,
            )
            if isinstance(result, str):
                shown.append(result)
            shown.append(await _consume_stream(result))
    finally:
        await pipe.close()
        if original_row is not None:
            owui_functions.Functions.get_user_valves_by_id_and_user_id = original_row  # type: ignore[method-assign]
    assert sent, f"no request for {model} reached OpenRouter, so this asserts nothing"
    return sent, "".join(shown)
