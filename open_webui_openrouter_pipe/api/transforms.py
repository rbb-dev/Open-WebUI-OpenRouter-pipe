"""Request and response transformation layer.

This module handles bidirectional conversion between API formats:
- Responses API <-> Chat Completions API payload conversion
- Message format transforms
- Tool schema normalization
- Structured output format conversion
- Model fallback application
- Request filtering and enrichment

The transforms bridge Open WebUI's expectations with OpenRouter's API variants.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
from collections.abc import Awaitable, Callable, Iterator
from math import isfinite
from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

if TYPE_CHECKING:
    from ..pipe import Pipe

from ..core.config import (
    _EMPTY_TOOL_SCHEMA,
    _MAX_OPENROUTER_ID_CHARS,
    _MAX_OPENROUTER_METADATA_KEY_CHARS,
    _MAX_OPENROUTER_METADATA_PAIRS,
    _MAX_OPENROUTER_METADATA_VALUE_CHARS,
    _NON_REPLAYABLE_TOOL_ARTIFACTS,
    _PIPE_METADATA_KEY,
    _PROVIDER_SLUG_PATTERN,
    OPENAI_EMPTY_USER_TURN_FALLBACK,
)
from ..core.fusion_defaults import (
    _REQUIRED_TOOL_CHOICE,
    find_fusion_entry,
    has_active_fusion_entry,
)
from ..core.image_detail import image_detail_or_auto
from ..core.timing_logger import timed
from ..core.url_scheme import is_absolute_url, loggable_link, url_scheme
from ..core.utils import (
    OPEN_WEBUI_TOOL_IMAGES_TEXT,
    _coerce_bool,
    _parse_model_fallback_csv,
    _sticky_session_key,
    is_picture_output,
    is_text_part_output,
    opens_a_turn,
    recorded_tool_text,
    server_tool_arguments,
    server_tool_call_id,
    server_tool_result_text,
    server_tool_status,
    strip_hidden_marker_lines,
    tool_output_text_and_pictures,
)
from ..core.warn_latch import warn_level
from ..filters.fusion_filter_renderer import _fusion_base_model_id, is_fusion_model
from ..models.registry import ModelFamily
from ..storage.owui_files import is_temporary_chat, names_an_owui_file_path
from ..tools.tool_schema import _strictify_schema

# Pydantic Body Classes

logger = logging.getLogger(__name__)

_warned_previous_response_id: set[str] = set()
_warned_store: set[str] = set()

class CompletionsBody(BaseModel):
    """
    Represents the body of a completions request to OpenAI completions API.
    """
    model: str
    messages: list[dict[str, Any]]
    stream: bool = False
    response_format: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    function_call: str | dict[str, Any] | None = None
    tool_choice: str | dict[str, Any] | None = None
    model_config = ConfigDict(extra="allow")


class ResponsesBody(BaseModel):
    """
    Represents the body of a responses request to OpenAI Responses API.
    """
    
    # Core parameters
    model: str
    models: list[str] | None = None
    instructions: str | None = None
    input: str | list[dict[str, Any]]

    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: float | None = None
    min_p: float | None = None
    top_a: float | None = None
    max_output_tokens: int | None = None
    reasoning: dict[str, Any] | None = None
    include_reasoning: bool | None = None
    thinking_config: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    tool_choice: str | dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    plugins: list[dict[str, Any]] | None = None
    truncation: str | None = None
    text: dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    transforms: list[str] | None = None
    stop_server_tools_when: list[dict[str, Any]] | None = None
    user: str | None = None
    session_id: str | None = None
    _continued_turn: tuple[int, int, bool] | None = PrivateAttr(default=None)
    _continues_after_marker: bool = PrivateAttr(default=False)

    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    repetition_penalty: float | None = None
    logit_bias: dict[str, float] | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    response_format: dict[str, Any] | None = None
    structured_outputs: bool | None = None
    reasoning_effort: str | None = None
    verbosity: str | None = None
    web_search_options: dict[str, Any] | None = None
    stream_options: dict[str, Any] | None = None

    provider: dict[str, Any] | None = None
    route: str | None = None
    debug: dict[str, Any] | None = None
    image_config: dict[str, Any] | None = None
    modalities: list[str] | None = None
    input_file_sizes: dict[str, tuple[int, str, str]] | None = Field(
        default=None, exclude=True
    )
    budget_futility_notified: bool = Field(default=False, exclude=True)
    budget_reported_call_ids: set[str] = Field(default_factory=set, exclude=True)
    budget_chars_per_token: dict[str, float] = Field(default_factory=dict, exclude=True)
    model_config = ConfigDict(extra="allow")

    @staticmethod
    def _strip_blank_string(value: Any) -> Any:
        if isinstance(value, str):
            candidate = value.strip()
            return candidate or None
        return value

    @field_validator(
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "top_a",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        mode="before",
    )
    @classmethod
    def _coerce_float_fields(cls, value: Any) -> Any:
        return _coerced_sampling_float(cls._strip_blank_string(value))

    @field_validator(
        "max_output_tokens",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "top_logprobs",
        mode="before",
    )
    @classmethod
    def _coerce_int_fields(cls, value: Any) -> Any:
        value = cls._strip_blank_string(value)
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("Boolean is not a valid integer value.")  # noqa: TRY004 - pydantic converts ValueError to ValidationError; TypeError would propagate raw
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if not isfinite(value):
                return None
            return round(value)
        if isinstance(value, str):
            try:
                numeric = float(value)
            except ValueError as exc:
                raise ValueError(f"Invalid integer value: {value!r}") from exc
            if not isfinite(numeric):
                return None
            return round(numeric)
        raise ValueError(f"Invalid integer value: {value!r}")

    @field_validator("models", mode="before")
    @classmethod
    def _coerce_models_list(cls, value: Any) -> Any:
        value = cls._strip_blank_string(value)
        if value is None:
            return None
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
            models = [part for part in parts if part]
            return models or None
        if isinstance(value, list):
            models: list[str] = []
            for entry in value:
                if not isinstance(entry, str):
                    continue
                candidate = entry.strip()
                if candidate:
                    models.append(candidate)
            return models or None
        raise ValueError("models must be an array of strings (or a CSV string).")

    @model_validator(mode='after')
    def _normalize_model_id(self) -> ResponsesBody:
        normalized = ModelFamily.base_model(self.model or "")
        if normalized and normalized != self.model:
            self.model = normalized
        return self

    @staticmethod
    @timed
    def transform_owui_tools(__tools__: dict[str, dict] | None, *, strict: bool = False) -> list[dict]:
        """
        Convert Open WebUI __tools__ registry (dict of entries with {"spec": {...}}) into
        OpenAI Responses-API tool specs: {"type": "function", "name", ...}.

        """
        if not __tools__:
            return []

        tools: list[dict] = []
        for item in __tools__.values():
            spec = item.get("spec") or {}
            name = spec.get("name")
            if not name:
                continue

            params = spec.get("parameters") or {"type": "object", "properties": {}}

            tool = {
                "type": "function",
                "name": name,
                "description": spec.get("description") or name,
                "parameters": _strictify_schema(params) if strict else params,
            }
            if strict:
                tool["strict"] = True

            tools.append(tool)

        return tools

    @staticmethod
    def _convert_function_call_to_tool_choice(function_call: Any) -> Any | None:
        """
        Translate legacy OpenAI `function_call` payloads into modern `tool_choice` values.

        Returns either "auto"/"none" (strings) or {"type": "function", "name": "..."}.
        """
        if function_call is None:
            return None
        if isinstance(function_call, str):
            lowered = function_call.strip().lower()
            if lowered in {"auto", "none"}:
                return lowered
            return None

        if isinstance(function_call, dict):
            name = function_call.get("name")
            if not name and isinstance(function_call.get("function"), dict):
                name = function_call["function"].get("name")
            if isinstance(name, str) and name.strip():
                return {"type": "function", "name": name.strip()}
        return None

    @classmethod
    @timed
    async def from_completions(
        cls,
        completions_body: CompletionsBody,
        chat_id: str | None = None,
        openwebui_model_id: str | None = None,
        *,
        user_obj: Any | None = None,
        event_emitter: Callable | None = None,
        artifact_loader: Callable[
            [str | None, str | None, list[str]],
            Awaitable[dict[str, dict[str, Any]] | tuple[dict[str, dict[str, Any]], dict[str, str]]],
        ]
        | None = None,
        pruning_turns: int = 0,
        transformer_context: Any | None = None,
        transformer_valves: Pipe.Valves | None = None,
        capability_model_id: str | None = None,
        ask_user_names: frozenset[str] = frozenset(),
        **extra_params,
    ) -> ResponsesBody:
        """
        Convert CompletionsBody -> ResponsesBody.

        - Drops unsupported fields (clearly logged).
        - Converts max_tokens -> max_output_tokens.
        - Converts reasoning_effort -> reasoning.effort (without overwriting).
        - Builds messages in Responses API format.
        - Allows explicit overrides via kwargs.
        - Replays persisted artifacts by awaiting `artifact_loader` when provided.
        """
        completions_dict = completions_body.model_dump(exclude_none=True)

        unsupported_fields = {
            "n",
            "suffix",
            "functions",
        }
        # carried elsewhere: max_tokens -> max_output_tokens; reasoning_effort ->
        # reasoning.effort; function_call -> tool_choice (all below); extra_tools ->
        # read off this CompletionsBody by requests/orchestrator.py and appended to the
        # advertised list by tools/tool_registry.py
        carried_fields = {"reasoning_effort", "max_tokens", "function_call", "extra_tools"}
        sanitized_params = {}
        for key, value in completions_dict.items():
            if key in unsupported_fields:
                logger.warning("Dropping unsupported parameter: '%s'", key)
            elif key not in carried_fields:
                sanitized_params[key] = value

        if "max_tokens" in completions_dict:
            # OpenRouter documents max_tokens as "integer, 1 or above". Open WebUI's
            # slider allows values below that; forwarding one earns a 400 that reads as
            # the pipe's fault, so anything outside the documented range is sent as no
            # cap at all.
            cap = _coerced_token_cap(completions_dict["max_tokens"])
            if cap is not None and cap >= 1:
                sanitized_params["max_output_tokens"] = cap

        if "max_completion_tokens" in completions_dict:
            requested_completion_max = _coerced_token_cap(
                completions_dict["max_completion_tokens"]
            )
            if requested_completion_max is not None and requested_completion_max >= 1:
                sanitized_params["max_output_tokens"] = requested_completion_max

        effort = completions_dict.get("reasoning_effort")
        if effort:
            reasoning = sanitized_params.get("reasoning", {})
            reasoning.setdefault("effort", effort)
            sanitized_params["reasoning"] = reasoning

        if "tool_choice" not in sanitized_params and "function_call" in completions_dict:
            converted_choice = ResponsesBody._convert_function_call_to_tool_choice(
                completions_dict.get("function_call")
            )
            if converted_choice is not None:
                sanitized_params["tool_choice"] = converted_choice

        if "messages" in completions_dict:
            sanitized_params.pop("messages", None)
            replayed_reasoning_refs: list[tuple[str, str]] = []
            attachment_notices: list[str] = []
            if transformer_context is None:
                raise RuntimeError(
                    "ResponsesBody.from_completions requires a transformer_context (usually the Pipe instance) "
                    "so multimodal helpers (downloads, status events, etc.) are available."
                )
            transformer_owner = transformer_context
            raw_messages = completions_dict.get("messages", [])
            filtered_messages: list[dict[str, Any]] = []
            if isinstance(raw_messages, list):
                for msg in raw_messages:
                    if not isinstance(msg, dict):
                        continue
                    role = (msg.get("role") or "").lower()
                    if not role:
                        continue
                    filtered_messages.append(msg)

            from ..requests.transformer import transform_messages_to_input

            sanitized_params["input"] = await transform_messages_to_input(
                transformer_owner,
                filtered_messages,
                chat_id=chat_id,
                openwebui_model_id=openwebui_model_id,
                artifact_loader=artifact_loader,
                pruning_turns=pruning_turns,
                replayed_reasoning_refs=replayed_reasoning_refs,
                user_obj=user_obj,
                event_emitter=event_emitter,
                model_id=completions_dict.get("model"),
                valves=transformer_valves or getattr(transformer_owner, "valves", None),
                capability_model_id=capability_model_id,
                ask_user_names=ask_user_names,
                attachment_notices=attachment_notices,
            )
            if replayed_reasoning_refs:
                sanitized_params["_replayed_reasoning_refs"] = replayed_reasoning_refs
            if attachment_notices:
                sanitized_params["_attachment_notices"] = attachment_notices

        merged_params = {
            **sanitized_params,
            **extra_params,
        }
        tool_choice = merged_params.get("tool_choice")
        if isinstance(tool_choice, dict):
            t = tool_choice.get("type")
            if t == "function":
                name = tool_choice.get("name")
                if not isinstance(name, str) or not name.strip():
                    fn = tool_choice.get("function")
                    if isinstance(fn, dict):
                        name = fn.get("name")
                if isinstance(name, str) and name.strip():
                    merged_params["tool_choice"] = {"type": "function", "name": name.strip()}

        tools_value = merged_params.get("tools")
        if isinstance(tools_value, list):
            merged_params["tools"] = _chat_tools_to_responses_tools(tools_value)

        _normalise_openrouter_responses_text_format(merged_params)

        return cls(**merged_params)

ALLOWED_OPENROUTER_FIELDS = {
    "model",
    "models",
    "input",
    "instructions",
    "metadata",
    "stream",
    "max_output_tokens",
    "temperature",
    "top_k",
    "top_p",
    "reasoning",
    "include_reasoning",
    "tools",
    "tool_choice",
    "plugins",
    "truncation",
    "preset",
    "text",
    "parallel_tool_calls",
    "user",
    "session_id",
    "transforms",
    "stop_server_tools_when",
    "trace",
    "background",
    "frequency_penalty",
    "image_config",
    "include",
    "max_tool_calls",
    "modalities",
    "presence_penalty",
    "prompt",
    "prompt_cache_key",
    "safety_identifier",
    "service_tier",
    "store",
    "top_logprobs",
    "provider",
    "route",
    "debug",
    "thinking_config",
    "web_search_options",
}

ALLOWED_OPENROUTER_CHAT_FIELDS = {
    "model",
    "models",
    "preset",
    "messages",
    "stream",
    "stream_options",
    "max_tokens",
    "max_completion_tokens",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "top_a",
    "stop",
    "seed",
    "presence_penalty",
    "frequency_penalty",
    "repetition_penalty",
    "logit_bias",
    "logprobs",
    "top_logprobs",
    "response_format",
    "structured_outputs",
    "reasoning",
    "include_reasoning",
    "reasoning_effort",
    "verbosity",
    "web_search_options",
    "tools",
    "tool_choice",
    "parallel_tool_calls",
    "max_tool_calls",
    "plugins",
    "user",
    "session_id",
    "metadata",
    "trace",
    "provider",
    "route",
    "debug",
    "service_tier",
    "prompt_cache_key",
    "image_config",
    "modalities",
    "transforms",
    "stop_server_tools_when",
}


_SERVER_TOOL_PREFIX = "openrouter:"


def _has_server_tool(tools: Any) -> bool:
    items = tools if isinstance(tools, list) else []
    return any(
        isinstance(t, dict)
        and isinstance(t.get("type"), str)
        and t["type"].startswith(_SERVER_TOOL_PREFIX)
        for t in items
    )


# Request Filtering

def _filter_openrouter_chat_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Filter payload to fields accepted by OpenRouter's /chat/completions."""
    if not isinstance(payload, dict):
        return {}
    filtered: dict[str, Any] = {}
    for key, value in payload.items():
        if key in ALLOWED_OPENROUTER_CHAT_FIELDS:
            if key == "metadata":
                value = _sanitize_openrouter_metadata(value)
                if value is None:
                    continue
            filtered[key] = value
    return filtered


def _strip_disable_model_settings_params(payload: dict[str, Any]) -> None:
    """Remove OWUI-local per-model control flags from the outbound provider payload."""
    if not isinstance(payload, dict):
        return
    for key in (
        "disable_model_metadata_sync",
        "disable_capability_updates",
        "disable_image_updates",
        "disable_web_tools_auto_attach",
        "disable_web_tools_default_on",
        "disable_direct_uploads_auto_attach",
        "disable_description_updates",
        "disable_native_websearch",
        "disable_native_web_search",
        "openrouter_provider_ignore",
        "openrouter_provider_only",
        "openrouter_provider_order",
    ):
        payload.pop(key, None)

# Tool Schema Transforms

def _responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    """Convert Responses API tools -> Chat Completions tools schema."""
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type", "")
        if isinstance(tool_type, str) and tool_type.startswith("openrouter:"):
            out.append(dict(tool))
            continue
        if tool_type != "function":
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        function: dict[str, Any] = {
            "name": name,
        }
        for key, want in (("description", str), ("parameters", dict)):
            if isinstance(tool.get(key), want):
                function[key] = tool[key]
        if "strict" in tool:
            function["strict"] = tool["strict"]
        entry: dict[str, Any] = {"type": "function", "function": function}
        if isinstance(tool.get("cache_control"), dict):
            entry["cache_control"] = tool["cache_control"]
        out.append(entry)
    return out


def _chat_tools_to_responses_tools(tools: Any) -> list[dict[str, Any]]:
    """Convert Chat Completions `tools` schema -> Responses API tools schema."""
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        tool_type = tool.get("type", "")
        if isinstance(tool_type, str) and tool_type.startswith("openrouter:"):
            out.append(dict(tool))
            continue
        if tool_type != "function":
            continue

        fn = tool.get("function")
        name = tool.get("name")
        if (not isinstance(name, str) or not name.strip()) and isinstance(fn, dict):
            name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()

        description = tool.get("description")
        parameters = tool.get("parameters")
        if isinstance(fn, dict):
            if not isinstance(description, str):
                description = fn.get("description")
            if not isinstance(parameters, dict):
                parameters = fn.get("parameters")

        spec: dict[str, Any] = {"type": "function", "name": name}
        if isinstance(description, str) and description.strip():
            spec["description"] = description.strip()
        spec["parameters"] = parameters if isinstance(parameters, dict) else _EMPTY_TOOL_SCHEMA
        if isinstance(tool.get("cache_control"), dict):
            spec["cache_control"] = tool["cache_control"]
        if isinstance(fn, dict):
            spec["strict"] = fn.get("strict", False)

        out.append(spec)

    return out


def _responses_tool_choice_to_chat_tool_choice(value: Any) -> Any:
    """Convert Responses API tool_choice -> Chat Completions tool_choice."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return value
    t = value.get("type")
    name = value.get("name")
    if not (isinstance(name, str) and name.strip()):
        function_block = value.get("function")
        if isinstance(function_block, dict):
            name = function_block.get("name")
    if t == "function" and isinstance(name, str) and name.strip():
        return {"type": "function", "function": {"name": name.strip()}}
    return value


# Response Format Transforms

def _coerced_token_cap(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if not isinstance(value, (float, str)):
        return None
    try:
        return round(float(value))
    except (OverflowError, ValueError):
        return None


def _coerced_sampling_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        numeric = float(value)
    except (OverflowError, ValueError):
        return None
    if not isfinite(numeric):
        return None
    return numeric


def _chat_response_format_to_responses_text_format(value: Any) -> dict[str, Any] | None:
    """Convert Chat Completions `response_format` -> Responses `text.format`."""
    if not isinstance(value, dict):
        return None

    fmt_type = value.get("type")
    if fmt_type in {"text", "json_object"}:
        return {"type": fmt_type}

    if fmt_type == "json_schema":
        json_schema = value.get("json_schema")
        if not isinstance(json_schema, dict):
            return None
        name = json_schema.get("name")
        schema = json_schema.get("schema")
        if not (isinstance(name, str) and name.strip()):
            return None
        if not isinstance(schema, dict):
            return None

        out: dict[str, Any] = {"type": "json_schema", "name": name.strip(), "schema": schema}
        description = json_schema.get("description")
        if isinstance(description, str) and description.strip():
            out["description"] = description.strip()
        strict = json_schema.get("strict")
        if isinstance(strict, bool):
            out["strict"] = strict
        return out

    return None

def _responses_text_format_to_chat_response_format(value: Any) -> dict[str, Any] | None:
    """Convert Responses `text.format` -> Chat Completions `response_format`."""
    if not isinstance(value, dict):
        return None

    fmt_type = value.get("type")
    if fmt_type in {"text", "json_object"}:
        return {"type": fmt_type}

    if fmt_type == "json_schema":
        name = value.get("name")
        schema = value.get("schema")
        if not (isinstance(name, str) and name.strip()):
            return None
        if not isinstance(schema, dict):
            return None
        json_schema: dict[str, Any] = {"name": name.strip(), "schema": schema}
        description = value.get("description")
        if isinstance(description, str) and description.strip():
            json_schema["description"] = description.strip()
        strict = value.get("strict")
        if isinstance(strict, bool):
            json_schema["strict"] = strict
        return {"type": "json_schema", "json_schema": json_schema}

    return None

def _normalise_openrouter_responses_text_format(payload: dict[str, Any]) -> None:
    """Normalise structured output config for OpenRouter `/responses` requests.

    The OpenRouter `/responses` endpoint follows an OpenResponses-style schema:
    structured outputs are configured via `text.format`, not `response_format`.

    To keep the pipe blast-safe:
    - Never raise for malformed input.
    - Prefer `text.format` when both are present (endpoint-native).
    - Otherwise, accept Chat-style `response_format` as an alias and translate it.
    """
    if not isinstance(payload, dict):
        return

    response_format = payload.get("response_format")
    response_format_as_text = _chat_response_format_to_responses_text_format(response_format)

    existing_text = payload.get("text")
    if isinstance(existing_text, dict):
        existing_text = dict(existing_text)
    if existing_text is None:
        existing_text = {}
    if not isinstance(existing_text, dict):
        if response_format_as_text is None:
            return
        existing_text = {}

    existing_format = existing_text.get("format")
    existing_as_chat = _responses_text_format_to_chat_response_format(existing_format)
    existing_canonical = (
        _chat_response_format_to_responses_text_format(existing_as_chat) if existing_as_chat else None
    )
    if existing_format is not None and existing_canonical is None:
        existing_text.pop("format", None)
        logger.warning(
            "Dropping invalid `text.format` on /responses request payload."
        )

    final_format: dict[str, Any] | None = None
    if existing_canonical is not None:
        final_format = dict(existing_canonical)
        if response_format_as_text is not None and existing_canonical != response_format_as_text:
            logger.warning(
                "Conflicting structured output config: preferring `text.format` over `response_format` for /responses."
            )
    elif response_format_as_text is not None:
        final_format = dict(response_format_as_text)

    payload.pop("response_format", None)

    if final_format is None:
        if len(existing_text) == 0:
            payload.pop("text", None)
        else:
            payload["text"] = existing_text
        return

    existing_text["format"] = final_format
    payload["text"] = existing_text


# Message and Input Transforms

def _replay_payload_is_present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        for key in ("url", "data"):
            if key in value:
                return _replay_payload_is_present(value[key])
    return False


def _input_audio_from_string(value: str) -> dict[str, Any] | None:
    from ..requests.transformer import _map_audio_format

    stripped = value.strip()
    if not stripped:
        return None
    if stripped[:5].lower() == "data:":
        head, _, payload = stripped.partition(",")
        mime = head[5:].split(";", 1)[0].strip()
        if not mime.startswith("audio/"):
            return None
        audio_format = _map_audio_format(mime)
        if audio_format is None:
            return None
        return {
            "type": "input_audio",
            "input_audio": {"data": "".join(payload.split()), "format": audio_format},
        }
    cleaned = "".join(stripped.split())
    try:
        base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError):
        return None
    return {
        "type": "input_audio",
        "input_audio": {"data": cleaned, "format": _map_audio_format(None)},
    }


def _image_file_payload(block: dict[str, Any]) -> dict[str, Any] | None:
    file_id = block.get("file_id")
    if not isinstance(file_id, str) or not file_id.strip():
        return None
    return {"type": "file", "file": {"file_id": file_id.strip()}}


_CHAT_INTERNAL_PATH_REFUSAL = (
    "a link to this Open WebUI's own file endpoint, which no provider can fetch"
)


def _chat_link_refusal_reason(
    field: str, url: Any, *, allow_insecure: Callable[[str], bool], max_inline_bytes: int,
) -> str | None:
    from ..requests.transformer import _tool_picture_gate

    if not isinstance(url, str) or not url.strip():
        return None
    candidate = url.strip()
    if names_an_owui_file_path(candidate):
        return _CHAT_INTERNAL_PATH_REFUSAL
    if not url_scheme(candidate):
        if field == "file_data":
            return None
        if field == "file_url" and not is_absolute_url(candidate):
            return None
    _, refused = _tool_picture_gate(
        [candidate], max_inline_bytes=max_inline_bytes, allow_insecure=allow_insecure,
    )
    return refused[0][1] if refused else None


def _chat_media_url(block: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = block.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, dict):
            nested = value.get("url")
            if isinstance(nested, str) and nested.strip():
                return nested
    return None


def _chat_audio_url_refusal(block: dict[str, Any], *keys: str) -> str | None:
    from ..requests.transformer import _AUDIO_URL_REFUSAL

    url = _chat_media_url(block, *keys)
    if not isinstance(url, str):
        return None
    candidate = url.strip()
    if not candidate or url_scheme(candidate) == "data":
        return None
    return _AUDIO_URL_REFUSAL


def _replay_block_is_usable(block: Any, refused: dict[int, str] | None = None) -> bool:
    if refused is not None and id(block) in refused:
        return False
    if not isinstance(block, dict):
        return True
    btype = block.get("type")
    if btype in {"text", "input_text", "output_text"}:
        text = block.get("text")
        if not isinstance(text, str):
            return False
        return bool(strip_hidden_marker_lines(text).strip())
    if btype in {"file", "input_file"}:
        payload = block.get("file")
        if isinstance(payload, dict):
            return any(payload.get(key) for key in ("file_id", "file_data", "file_url"))
        return any(block.get(key) for key in ("file_id", "file_data", "file_url"))
    if btype in {"image_url", "input_image"}:
        return _replay_payload_is_present(block.get("image_url")) or _image_file_payload(block) is not None
    if btype == "input_audio":
        return _replay_payload_is_present(block.get("input_audio"))
    if btype in {"video_url", "input_video"}:
        video = block.get("video_url")
        return _replay_payload_is_present(
            video if video is not None else block.get("url")
        )
    return True


def _replay_block_refusal(block: Any, refused: dict[int, str] | None = None) -> str | None:
    if refused is not None and id(block) in refused:
        return refused[id(block)]
    if not isinstance(block, dict):
        return None
    btype = block.get("type")
    if btype in {"image_url", "input_image"}:
        return "an image carried no picture data"
    if btype == "input_audio":
        return "an audio clip carried no audio data"
    if btype in {"video_url", "input_video"}:
        return "a video clip carried no video data"
    if btype in {"file", "input_file"}:
        return "a file carried no contents"
    return None


def responses_refusal_text(item: Any) -> str | None:
    if not isinstance(item, dict):
        return None
    own = item.get("refusal")
    if isinstance(own, str) and own.strip():
        return own.strip()
    content = item.get("content")
    if not isinstance(content, list):
        return None
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "refusal":
            continue
        text = part.get("refusal")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None


def _replay_blocks_or_note(
    blocks: list[Any],
    siblings: list[Any] | None = None,
    *,
    role: str = "user",
    refused: dict[int, str] | None = None,
) -> list[Any]:
    originals = siblings or blocks
    survivors = [b for b in blocks if _replay_block_is_usable(b, refused)]
    refusals = [
        r
        for r in (
            _replay_block_refusal(b, refused)
            for b in originals
            if not _replay_block_is_usable(b, refused)
        )
        if r
    ]
    note = [
        {"type": "text", "text": f"[An attached item was not sent: {'; '.join(refusals)}.]"}
    ] if refusals else []
    if survivors:
        return survivors + note
    if not refusals:
        if role != "user":
            return blocks
        if any(_replay_block_is_usable(b, refused) for b in originals):
            return [b for b in originals if _replay_block_is_usable(b, refused)]
        return [{"type": "text", "text": OPENAI_EMPTY_USER_TURN_FALLBACK}]
    return note


async def _responses_input_to_chat_messages(
    input_value: Any,
    *,
    max_inline_bytes: int,
    allow_insecure: Callable[[str], bool],
    allow_unknown_fields: bool = False,
    pipe: Pipe | None = None,
    refused_out: list[tuple[str, str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Convert Responses API input array -> Chat Completions messages array.

    Best-effort mapping for supported multimodal blocks. Unsupported blocks are
    degraded to text notes rather than dropped.

    Args:
        input_value: Responses API input (list of message items, string, or None)
        allow_unknown_fields: When True, preserves unknown/custom fields from input items.
            When False (default), only copies known/validated fields for stricter behavior.
    """
    if input_value is None:
        return []
    if isinstance(input_value, str):
        text = strip_hidden_marker_lines(input_value).strip()
        return [{"role": "user", "content": text}] if text else []
    if not isinstance(input_value, list):
        return []

    messages: list[dict[str, Any]] = []
    tool_pictures: list[str] = []
    media_refusals: dict[int, str] = {}
    pending_reasoning_details: list[Any] = []

    async def _hand_over_tool_pictures() -> None:
        from ..requests.transformer import _tool_picture_gate_with_address

        if not tool_pictures:
            return
        if pipe is None:
            logger.warning(
                "A tool's picture reached the chat-completions handover with no pipe, so the "
                "address the provider would reach could not be checked; every http(s) one is refused"
            )
        kept, refused = await _tool_picture_gate_with_address(
            pipe, tool_pictures, max_inline_bytes=max_inline_bytes,
        )
        if refused and refused_out is not None:
            refused_out.extend(refused)
        for url, reason, cause in refused:
            logger.warning(
                "Not forwarding a tool's picture (%s): %s [cause=%s]",
                loggable_link(url), reason, cause,
            )
        tool_pictures.clear()
        if not kept:
            return
        messages.append({"role": "user", "content": [
            {"type": "text", "text": OPEN_WEBUI_TOOL_IMAGES_TEXT},
            *({"type": "image_url", "image_url": {"url": url}} for url in kept),
        ]})

    def _flush_pending_reasoning() -> None:
        if not pending_reasoning_details:
            return
        messages.append({
            "role": "assistant",
            "content": "",
            "reasoning_details": list(pending_reasoning_details),
        })
        pending_reasoning_details.clear()

    def _attach_reasoning_details(msg: dict[str, Any]) -> None:
        if not pending_reasoning_details:
            return
        if msg.get("role") != "assistant":
            return
        existing = msg.get("reasoning_details")
        merged = (
            [*(existing if isinstance(existing, list) else []), *pending_reasoning_details]
            if existing
            else list(pending_reasoning_details)
        )
        msg["reasoning_details"] = merged
        pending_reasoning_details.clear()

    def _to_text_block(text: str, *, cache_control: Any = None) -> dict[str, Any]:
        block: dict[str, Any] = {"type": "text", "text": text}
        if isinstance(cache_control, dict) and cache_control:
            block["cache_control"] = dict(cache_control)
        return block

    for index, item in enumerate(input_value):
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        if itype != "function_call_output":
            await _hand_over_tool_pictures()

        if opens_a_turn(input_value, index):
            _flush_pending_reasoning()

        if itype == "reasoning":
            details = item.get("reasoning_details")
            if isinstance(details, list):
                pending_reasoning_details.extend(
                    detail for detail in details
                    if isinstance(detail, dict) and (
                        detail.get("format") != "anthropic-claude-v1" or detail.get("signature")
                    )
                )
            else:
                encrypted = item.get("encrypted_content")
                if isinstance(encrypted, str) and encrypted:
                    entry: dict[str, Any] = {"type": "reasoning.encrypted", "data": encrypted}
                    if isinstance(item.get("signature"), str):
                        entry["signature"] = item["signature"]
                    if isinstance(item.get("format"), str):
                        entry["format"] = item["format"]
                    pending_reasoning_details.append(entry)
                elif isinstance(item.get("signature"), str) and item["signature"]:
                    parts = item.get("summary") or item.get("content") or []
                    if not isinstance(parts, list):
                        parts = [parts]
                    text = "".join(
                        str(part.get("text") or "") if isinstance(part, dict) else str(part)
                        for part in parts
                    )
                    if text:
                        pending_reasoning_details.append({
                            "type": "reasoning.text",
                            "text": text,
                            "signature": item["signature"],
                            "format": item.get("format"),
                        })
            continue

        if itype == "message":
            role = (item.get("role") or "").strip().lower()
            if not role:
                continue

            media_refusals.clear()
            raw_content = item.get("content")

            if not allow_unknown_fields:
                raw_annotations = item.get("annotations")
                msg_annotations: list[Any] = (
                    list(raw_annotations)
                    if isinstance(raw_annotations, list) and raw_annotations
                    else []
                )
                raw_reasoning_details = item.get("reasoning_details")
                msg_reasoning_details: list[Any] = (
                    list(raw_reasoning_details)
                    if isinstance(raw_reasoning_details, list) and raw_reasoning_details
                    else []
                )

                if isinstance(raw_content, str):
                    msg: dict[str, Any] = {
                        "role": role,
                        "content": strip_hidden_marker_lines(raw_content),
                    }
                    if msg_annotations:
                        msg["annotations"] = msg_annotations
                    if msg_reasoning_details:
                        msg["reasoning_details"] = msg_reasoning_details
                    _attach_reasoning_details(msg)
                    messages.append(msg)
                    continue

                blocks_out: list[dict[str, Any]] = []
                if isinstance(raw_content, list):
                    for block in raw_content:
                        if not isinstance(block, dict):
                            continue
                        btype = block.get("type")
                        if btype in {"input_text", "output_text"}:
                            text = block.get("text")
                            if isinstance(text, str) and text:
                                cleaned = strip_hidden_marker_lines(text)
                                if not cleaned:
                                    continue
                                blocks_out.append(
                                    _to_text_block(cleaned, cache_control=block.get("cache_control"))
                                )
                            continue
                        if btype == "input_image":
                            url = block.get("image_url")
                            if isinstance(url, str) and url.strip():
                                refusal = _chat_link_refusal_reason(
                                    "image_url", url, allow_insecure=allow_insecure,
                                    max_inline_bytes=max_inline_bytes,
                                )
                                if refusal is not None:
                                    media_refusals[id(block)] = refusal
                                    continue
                                image_url_obj: dict[str, Any] = {"url": url.strip()}
                                image_url_obj["detail"] = image_detail_or_auto(block.get("detail"))
                                blocks_out.append({"type": "image_url", "image_url": image_url_obj})
                                continue
                            payload = _image_file_payload(block)
                            if payload is not None:
                                blocks_out.append(payload)
                            continue
                        if btype == "image_url":
                            image_url_val = block.get("image_url")
                            refusal = _chat_link_refusal_reason(
                                "image_url", _chat_media_url(block, "image_url"),
                                allow_insecure=allow_insecure,
                                max_inline_bytes=max_inline_bytes,
                            )
                            if refusal is not None:
                                media_refusals[id(block)] = refusal
                                continue
                            if isinstance(image_url_val, dict):
                                blocks_out.append({"type": "image_url", "image_url": dict(image_url_val)})
                            elif isinstance(image_url_val, str) and image_url_val.strip():
                                blocks_out.append({"type": "image_url", "image_url": {"url": image_url_val.strip()}})
                            continue
                        if btype == "input_audio":
                            audio = block.get("input_audio")
                            if isinstance(audio, dict):
                                refusal = _chat_audio_url_refusal(block, "input_audio")
                                if refusal is not None:
                                    media_refusals[id(block)] = refusal
                                    continue
                                blocks_out.append({"type": "input_audio", "input_audio": dict(audio)})
                            elif isinstance(audio, str):
                                converted = _input_audio_from_string(audio)
                                if converted is not None:
                                    blocks_out.append(converted)
                            continue
                        if btype == "video_url":
                            video_url = block.get("video_url")
                            refusal = _chat_link_refusal_reason(
                                "video_url", _chat_media_url(block, "video_url"),
                                allow_insecure=allow_insecure,
                                max_inline_bytes=max_inline_bytes,
                            )
                            if refusal is not None:
                                media_refusals[id(block)] = refusal
                                continue
                            if isinstance(video_url, dict):
                                blocks_out.append({"type": "video_url", "video_url": dict(video_url)})
                            elif isinstance(video_url, str) and video_url.strip():
                                blocks_out.append({"type": "video_url", "video_url": {"url": video_url.strip()}})
                            continue
                        if btype == "input_video":
                            video = block.get("video_url")
                            if video is None:
                                video = block.get("url")
                            refusal = _chat_link_refusal_reason(
                                "video_url", _chat_media_url(block, "video_url", "url"),
                                allow_insecure=allow_insecure,
                                max_inline_bytes=max_inline_bytes,
                            )
                            if refusal is not None:
                                media_refusals[id(block)] = refusal
                                continue
                            if isinstance(video, str) and video.strip():
                                blocks_out.append({"type": "video_url", "video_url": {"url": video.strip()}})
                            elif isinstance(video, dict) and video.get("url"):
                                blocks_out.append({"type": "video_url", "video_url": dict(video)})
                            continue
                        if btype == "input_file":
                            filename = block.get("filename")
                            file_data = block.get("file_data")
                            file_url = block.get("file_url")
                            file_id = block.get("file_id")
                            file_payload: dict[str, Any] = {}
                            if isinstance(file_id, str) and file_id.strip():
                                file_payload["file_id"] = file_id.strip()
                            if isinstance(filename, str) and filename.strip():
                                file_payload["filename"] = filename.strip()
                            file_value: str | None = None
                            file_refusal: str | None = None
                            if isinstance(file_data, str) and file_data.strip():
                                file_refusal = _chat_link_refusal_reason(
                                    "file_data", file_data, allow_insecure=allow_insecure,
                                    max_inline_bytes=max_inline_bytes,
                                )
                                if file_refusal is None:
                                    file_value = file_data.strip()
                            elif isinstance(file_url, str) and file_url.strip():
                                file_refusal = _chat_link_refusal_reason(
                                    "file_url", file_url, allow_insecure=allow_insecure,
                                    max_inline_bytes=max_inline_bytes,
                                )
                                if file_refusal is None:
                                    file_value = file_url.strip()
                            if file_value:
                                file_payload["file_data"] = file_value
                            if file_refusal is not None and not (
                                file_payload.get("file_id") or file_value
                            ):
                                media_refusals[id(block)] = file_refusal
                            if file_payload:
                                blocks_out.append({"type": "file", "file": file_payload})
                            continue

                if isinstance(raw_content, list) and raw_content:
                    blocks_out = _replay_blocks_or_note(
                        blocks_out, raw_content, role=role, refused=media_refusals,
                    )

                if (
                    not blocks_out
                    and role == "assistant"
                    and not msg_annotations
                    and not msg_reasoning_details
                    and not pending_reasoning_details
                ):
                    continue

                if not blocks_out:
                    msg: dict[str, Any] = {"role": role, "content": []}
                    if msg_annotations:
                        msg["annotations"] = msg_annotations
                    if msg_reasoning_details:
                        msg["reasoning_details"] = msg_reasoning_details
                    _attach_reasoning_details(msg)
                    messages.append(msg)
                else:
                    msg = {"role": role, "content": blocks_out}
                    if msg_annotations:
                        msg["annotations"] = msg_annotations
                    if msg_reasoning_details:
                        msg["reasoning_details"] = msg_reasoning_details
                    _attach_reasoning_details(msg)
                    messages.append(msg)
                continue

            msg: dict[str, Any] = dict(item)
            msg.pop("type", None)
            msg["role"] = role

            if isinstance(raw_content, str):
                msg["content"] = strip_hidden_marker_lines(raw_content)
                _attach_reasoning_details(msg)
                messages.append(msg)
                continue

            blocks_out: list[dict[str, Any]] = []
            if isinstance(raw_content, list):
                for block in raw_content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")

                    if btype in {"input_text", "output_text"}:
                        transformed = dict(block)
                        transformed["type"] = "text"
                        text = transformed.get("text")
                        if isinstance(text, str) and text:
                            cleaned = strip_hidden_marker_lines(text)
                            if not cleaned:
                                continue
                            transformed["text"] = cleaned
                            blocks_out.append(transformed)
                        continue

                    if btype == "input_image":
                        transformed = dict(block)
                        transformed["type"] = "image_url"
                        url = transformed.pop("image_url", "")
                        refusal = _chat_link_refusal_reason(
                            "image_url", _chat_media_url(block, "image_url"),
                            allow_insecure=allow_insecure,
                            max_inline_bytes=max_inline_bytes,
                        )
                        if refusal is not None:
                            media_refusals[id(block)] = refusal
                            continue
                        image_url_obj: dict[str, Any] = {"url": url.strip() if isinstance(url, str) else ""}
                        detail = transformed.pop("detail", None)
                        image_url_obj["detail"] = image_detail_or_auto(detail)
                        transformed["image_url"] = image_url_obj
                        if image_url_obj["url"]:
                            blocks_out.append(transformed)
                            continue
                        payload = _image_file_payload(block)
                        if payload is not None:
                            blocks_out.append(payload)
                        continue

                    if btype == "image_url":
                        transformed = dict(block)
                        image_url_val = transformed.get("image_url")
                        refusal = _chat_link_refusal_reason(
                            "image_url", _chat_media_url(block, "image_url"),
                            allow_insecure=allow_insecure,
                            max_inline_bytes=max_inline_bytes,
                        )
                        if refusal is not None:
                            media_refusals[id(block)] = refusal
                            continue
                        if isinstance(image_url_val, dict):
                            transformed["image_url"] = dict(image_url_val)
                        elif isinstance(image_url_val, str) and image_url_val.strip():
                            transformed["image_url"] = {"url": image_url_val.strip()}
                        else:
                            continue
                        blocks_out.append(transformed)
                        continue

                    if btype == "input_audio":
                        # Pass through with copy
                        transformed = dict(block)
                        audio = transformed.get("input_audio")
                        if isinstance(audio, dict):
                            refusal = _chat_audio_url_refusal(block, "input_audio")
                            if refusal is not None:
                                media_refusals[id(block)] = refusal
                                continue
                            transformed["input_audio"] = dict(audio)
                            blocks_out.append(transformed)
                        continue

                    if btype == "video_url":
                        transformed = dict(block)
                        video_url = transformed.get("video_url")
                        refusal = _chat_link_refusal_reason(
                            "video_url", _chat_media_url(block, "video_url"),
                            allow_insecure=allow_insecure,
                            max_inline_bytes=max_inline_bytes,
                        )
                        if refusal is not None:
                            media_refusals[id(block)] = refusal
                            continue
                        if isinstance(video_url, dict):
                            transformed["video_url"] = dict(video_url)
                        elif isinstance(video_url, str) and video_url.strip():
                            transformed["video_url"] = {"url": video_url.strip()}
                        else:
                            continue
                        blocks_out.append(transformed)
                        continue

                    if btype == "input_file":
                        transformed = dict(block)
                        transformed["type"] = "file"
                        file_id = transformed.pop("file_id", None)
                        filename = transformed.pop("filename", None)
                        file_data = transformed.pop("file_data", None)
                        file_url = transformed.pop("file_url", None)
                        file_payload: dict[str, Any] = {}
                        if isinstance(file_id, str) and file_id.strip():
                            file_payload["file_id"] = file_id.strip()
                        if isinstance(filename, str) and filename.strip():
                            file_payload["filename"] = filename.strip()
                        file_value: str | None = None
                        file_refusal: str | None = None
                        if isinstance(file_data, str) and file_data.strip():
                            file_refusal = _chat_link_refusal_reason(
                                "file_data", file_data, allow_insecure=allow_insecure,
                                max_inline_bytes=max_inline_bytes,
                            )
                            if file_refusal is None:
                                file_value = file_data.strip()
                        elif isinstance(file_url, str) and file_url.strip():
                            file_refusal = _chat_link_refusal_reason(
                                "file_url", file_url, allow_insecure=allow_insecure,
                                max_inline_bytes=max_inline_bytes,
                            )
                            if file_refusal is None:
                                file_value = file_url.strip()
                        if file_value:
                            file_payload["file_data"] = file_value
                        if file_refusal is not None and not (
                            file_payload.get("file_id") or file_value
                        ):
                            media_refusals[id(block)] = file_refusal
                        if file_payload:
                            transformed["file"] = file_payload
                            blocks_out.append(transformed)
                        continue

                    blocks_out.append(dict(block))

            if isinstance(raw_content, list) and raw_content:
                blocks_out = _replay_blocks_or_note(
                    blocks_out, raw_content, role=role, refused=media_refusals,
                )

            msg["content"] = blocks_out
            _attach_reasoning_details(msg)
            messages.append(msg)
            continue

        if isinstance(itype, str) and itype.startswith("openrouter:"):
            call_id = server_tool_call_id(item.get("id"))
            arguments = server_tool_arguments(item)
            tool_call_msg: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": itype.split(":", 1)[1] or itype,
                            "arguments": json.dumps(arguments, ensure_ascii=False),
                        },
                    }
                ],
            }
            _attach_reasoning_details(tool_call_msg)
            messages.append(tool_call_msg)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": recorded_tool_text(server_tool_result_text(item), server_tool_status(item)),
                }
            )
            continue

        if itype == "function_call_output":
            call_id = item.get("call_id")
            output = item.get("output")
            if isinstance(call_id, str) and call_id.strip():
                if is_picture_output(output):
                    content, pictures = tool_output_text_and_pictures(output)
                    tool_pictures.extend(pictures)
                elif is_text_part_output(output):
                    content = tool_output_text_and_pictures(output)[0]
                else:
                    content = output if isinstance(output, str) else (json.dumps(output, ensure_ascii=False) if output is not None else "")
                messages.append({"role": "tool", "tool_call_id": call_id.strip(), "content": content})
            continue

        if itype == "function_call":
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            args = item.get("arguments")
            if not (isinstance(call_id, str) and call_id.strip() and isinstance(name, str) and name.strip()):
                continue
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False) if args is not None else "{}"
            call_msg: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id.strip(),
                        "type": "function",
                        "function": {"name": name.strip(), "arguments": args},
                    }
                ],
            }
            _attach_reasoning_details(call_msg)
            messages.append(call_msg)
            continue


    await _hand_over_tool_pictures()
    _flush_pending_reasoning()
    return messages


# Payload Transforms

def _coerce_openrouter_int(value: Any) -> int | None:
    return _coerced_token_cap(value)


def chat_payload_loses_fusion_entry(model_id: Any, plugins: Any) -> bool:
    if not is_fusion_model(str(model_id or "")):
        return False
    entry = find_fusion_entry(plugins)
    return isinstance(entry, dict) and entry.get("enabled") is not False


async def _responses_payload_to_chat_completions_payload(
    responses_payload: dict[str, Any],
    *,
    max_inline_bytes: int,
    allow_insecure: Callable[[str], bool],
    pipe: Pipe | None = None,
    refused_out: list[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Convert a Responses API request payload into a Chat Completions payload."""
    if not isinstance(responses_payload, dict):
        return {}
    chat_payload: dict[str, Any] = {}

    # Core routing and identifiers
    for key in ("model", "models", "preset", "user", "session_id", "metadata", "plugins", "provider", "route", "debug", "image_config", "modalities", "transforms", "stop_server_tools_when", "trace"):
        if key in responses_payload:
            chat_payload[key] = responses_payload[key]

    plugins = chat_payload.get("plugins")
    if isinstance(plugins, list) and is_fusion_model(
        _fusion_base_model_id(str(responses_payload.get("model") or ""))
    ):
        kept = [p for p in plugins if not (isinstance(p, dict) and p.get("id") == "fusion")]
        if kept:
            chat_payload["plugins"] = kept
        else:
            chat_payload.pop("plugins", None)

    # Streaming flags
    stream = bool(responses_payload.get("stream"))
    chat_payload["stream"] = stream
    if stream:
        existing_stream_options = responses_payload.get("stream_options")
        if isinstance(existing_stream_options, dict):
            chat_payload["stream_options"] = dict(existing_stream_options)

    passthrough = (
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "top_a",
        "stop",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "response_format",
        "structured_outputs",
        "reasoning",
        "include_reasoning",
        "reasoning_effort",
        "verbosity",
        "max_completion_tokens",
        "web_search_options",
        "parallel_tool_calls",
        "max_tool_calls",
        "service_tier",
        "prompt_cache_key",
    )
    for key in passthrough:
        if key in responses_payload:
            chat_payload[key] = responses_payload[key]

    for key in ("top_k", "seed", "top_logprobs", "max_tokens", "max_completion_tokens"):
        if key not in chat_payload:
            continue
        rounded = _coerce_openrouter_int(chat_payload.get(key))
        if rounded is None:
            chat_payload.pop(key, None)
        else:
            chat_payload[key] = rounded

    existing_response_format = chat_payload.get("response_format")
    if existing_response_format is not None and _chat_response_format_to_responses_text_format(existing_response_format) is None:
        chat_payload.pop("response_format", None)
        existing_response_format = None
        logger.warning("Dropping invalid `response_format` on /chat/completions payload.")

    responses_text = responses_payload.get("text")
    if isinstance(responses_text, dict):
        mapped_response_format = _responses_text_format_to_chat_response_format(responses_text.get("format"))
        if existing_response_format is None:
            if mapped_response_format is not None:
                chat_payload["response_format"] = mapped_response_format
        elif mapped_response_format is not None and existing_response_format != mapped_response_format:
            logger.warning(
                "Conflicting structured output config: preferring `response_format` over `text.format` for /chat/completions."
            )

        if "verbosity" not in chat_payload:
            verbosity = responses_text.get("verbosity")
            if isinstance(verbosity, str) and verbosity.strip():
                chat_payload["verbosity"] = verbosity.strip()

    # Token limit mapping
    explicit_cap = chat_payload.get("max_completion_tokens")
    if isinstance(explicit_cap, int) and explicit_cap < 1:
        chat_payload.pop("max_completion_tokens", None)
    max_output_tokens = responses_payload.get("max_output_tokens")
    if (
        isinstance(max_output_tokens, int)
        and not isinstance(max_output_tokens, bool)
        and max_output_tokens >= 1
        and "max_completion_tokens" not in chat_payload
    ):
        chat_payload["max_tokens"] = max_output_tokens

    # Tools
    tools = responses_payload.get("tools")
    chat_tools = _responses_tools_to_chat_tools(tools)
    if chat_tools:
        chat_payload["tools"] = chat_tools
    tool_choice = _responses_tool_choice_to_chat_tool_choice(responses_payload.get("tool_choice"))
    if tool_choice is not None:
        chat_payload["tool_choice"] = tool_choice

    if chat_payload.get("tool_choice") == _REQUIRED_TOOL_CHOICE \
            and not chat_payload.get("tools") \
            and not has_active_fusion_entry(chat_payload.get("plugins")):
        chat_payload.pop("tool_choice", None)

    chat_payload["messages"] = await _responses_input_to_chat_messages(
        responses_payload.get("input"),
        max_inline_bytes=max_inline_bytes,
        allow_insecure=allow_insecure,
        pipe=pipe,
        refused_out=refused_out,
    )

    instructions = responses_payload.get("instructions")
    if isinstance(instructions, str):
        instructions = instructions.strip()
    else:
        instructions = ""
    if instructions:
        messages = chat_payload.get("messages")
        if not isinstance(messages, list):
            messages = []
            chat_payload["messages"] = messages
        if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
            system_msg = messages[0]
            existing_content = system_msg.get("content")
            if isinstance(existing_content, str):
                existing_text = existing_content.strip()
                if existing_text:
                    system_msg["content"] = f"{instructions}\n\n{existing_text}"
                else:
                    system_msg["content"] = instructions
            elif isinstance(existing_content, list):
                merged_blocks: list[dict[str, Any]] = [{"type": "text", "text": instructions}]
                folded = False
                for block in existing_content:
                    if (
                        not folded
                        and isinstance(block, dict)
                        and block.get("type") == "text"
                        and isinstance(block.get("text"), str)
                        and block["text"].strip()
                    ):
                        merged_blocks[0] = {**block, "text": f"{instructions}\n\n{block['text']}"}
                        folded = True
                        continue
                    merged_blocks.append(block)
                system_msg["content"] = merged_blocks
            else:
                system_msg["content"] = instructions
        else:
            messages.insert(0, {"role": "system", "content": instructions})

    return chat_payload


# Model Fallback

def _fallback_off_carry(primary: str) -> dict[str, Any]:
    from ..core.utils import _select_best_effort_fallback

    contract = ModelFamily.reasoning_contract(primary)
    if contract.get("mandatory") is not True:
        return {"effort": "none"}
    supported = [e for e in contract.get("supported_efforts") or [] if e != "none"]
    floor = _select_best_effort_fallback("none", supported)
    return {"effort": floor} if floor else {}


def _drop_include_reasoning_for_unsupported_fallbacks(
    request_payload: dict[str, Any], logger: logging.Logger
) -> None:
    if request_payload.get("include_reasoning") is None:
        return
    models = request_payload.get("models")
    if not isinstance(models, list):
        return
    for model_id in models:
        if not isinstance(model_id, str) or not model_id:
            continue
        if "include_reasoning" in ModelFamily.supported_parameters(model_id):
            continue
        dropped = request_payload.pop("include_reasoning")
        if dropped is False:
            primary = str(request_payload.get("model") or "")
            if "reasoning" in ModelFamily.supported_parameters(primary):
                request_payload["reasoning"] = _fallback_off_carry(primary)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Dropped include_reasoning=%r: fallback %r does not list it.",
                dropped,
                model_id,
            )
        return


def _apply_model_fallback_to_payload(payload: dict[str, Any], *, logger: logging.Logger = logger) -> None:
    if not isinstance(payload, dict):
        return
    raw_fallback = payload.pop("model_fallback", None)
    fallback_models = _parse_model_fallback_csv(raw_fallback)

    existing_models_raw = payload.get("models")
    existing_models: list[str] = []
    if isinstance(existing_models_raw, list):
        for entry in existing_models_raw:
            if isinstance(entry, str) and entry.strip():
                existing_models.append(entry.strip())

    if not fallback_models and not existing_models:
        return

    merged: list[str] = []
    seen: set[str] = set()
    for candidate in existing_models + fallback_models:
        if candidate in seen:
            continue
        seen.add(candidate)
        merged.append(candidate)

    if not merged:
        payload.pop("models", None)
        return

    payload["models"] = merged
    if raw_fallback not in (None, "", []):
        logger.debug("Applied model_fallback -> models (%d fallback(s))", len(fallback_models))


def _apply_openrouter_trace_to_payload(
    payload: dict[str, Any],
    *,
    logger: logging.Logger = logger,
) -> None:
    """Map OWUI custom ``openrouter_trace`` model param to OpenRouter ``trace``.

    Admins can add an ``openrouter_trace`` key in the model's
    *Custom Parameters* section (Advanced → Custom Parameters).  The value
    should be a JSON object, e.g.::

        {"trace_name": "My Pipeline", "generation_name": "chat"}

    Open WebUI's ``apply_model_params_to_body_openai`` auto-parses JSON
    strings in ``custom_params``, so by the time we receive the payload the
    value is already a ``dict``.

    This helper **pops** ``openrouter_trace`` and writes it as ``trace``,
    merging with any pre-existing ``trace`` dict already in the payload.
    """
    if not isinstance(payload, dict):
        return

    trace_data = payload.pop("openrouter_trace", None)
    if trace_data is None:
        return

    if isinstance(trace_data, str):
        trace_data = trace_data.strip()
        if not trace_data:
            return
        try:
            import json as _json
            parsed = _json.loads(trace_data)
        except (ValueError, TypeError):
            logger.warning(
                "openrouter_trace value is not valid JSON — ignored: %.120s",
                trace_data,
            )
            return
        if not isinstance(parsed, dict):
            logger.warning(
                "openrouter_trace must be a JSON object, got %s — ignored",
                type(parsed).__name__,
            )
            return
        trace_data = parsed

    if not isinstance(trace_data, dict) or not trace_data:
        return

    existing_trace = payload.get("trace")
    if isinstance(existing_trace, dict):
        merged = {**existing_trace, **trace_data}
    else:
        merged = trace_data

    payload["trace"] = merged
    logger.debug(
        "Applied openrouter_trace -> trace (%d key(s))",
        len(merged),
    )


# Feature Application to Payloads

def _apply_disable_native_websearch_to_payload(
    payload: dict[str, Any],
    *,
    logger: logging.Logger = logger,
) -> None:
    """Apply OWUI per-model `disable_native_websearch` custom param.

    When truthy, this disables OpenRouter's built-in web search integration by removing:
    - `plugins` entries with `{"id": "web"}`
    - `web_search_options` (chat/completions)
    """
    if not isinstance(payload, dict):
        return

    raw_flag = payload.pop("disable_native_websearch", None)
    if raw_flag is None:
        raw_flag = payload.pop("disable_native_web_search", None)

    disable = _coerce_bool(raw_flag)
    if disable is not True:
        return

    removed = False
    if payload.pop("web_search_options", None) is not None:
        removed = True

    tools = payload.get("tools")
    if isinstance(tools, list) and tools:
        kept = [t for t in tools if not (isinstance(t, dict) and t.get("type") == "openrouter:web_search")]
        if len(kept) != len(tools):
            removed = True
            if kept:
                payload["tools"] = kept
            else:
                payload.pop("tools", None)
            if not _has_server_tool(payload.get("tools")):
                payload.pop("stop_server_tools_when", None)

    plugins = payload.get("plugins")
    if isinstance(plugins, list) and plugins:
        filtered: list[Any] = []
        for entry in plugins:
            if isinstance(entry, dict) and entry.get("id") == "web":
                removed = True
                continue
            filtered.append(entry)

        if removed:
            if filtered:
                payload["plugins"] = filtered
            else:
                payload.pop("plugins", None)

    if removed:
        logger.debug(
            "Native web search disabled via custom param (model=%s).",
            payload.get("model"),
        )


# -- Provider routing custom parameters --------------------------------------

def _parse_provider_csv(value: Any, *, logger: logging.Logger = logger) -> list[str]:
    """CSV string or list → validated, deduped, lowercase provider slugs."""
    if isinstance(value, list):
        parts = [str(v).strip() for v in value]
    elif isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
    else:
        return []
    seen: set[str] = set()
    result: list[str] = []
    for raw in parts:
        slug = raw.lower()
        if not slug or slug in seen:
            continue
        if not _PROVIDER_SLUG_PATTERN.match(slug):
            logger.warning("Dropping invalid provider slug %r from custom param.", slug)
            continue
        seen.add(slug)
        result.append(slug)
    return result


def _apply_provider_routing_params_to_payload(
    payload: dict[str, Any], *, logger: logging.Logger = logger,
) -> None:
    """Pop openrouter_provider_{ignore,only,order} and merge into provider dict."""
    if not isinstance(payload, dict):
        return
    try:
        parsed = {
            key: _parse_provider_csv(payload.pop(f"openrouter_provider_{key}", None), logger=logger)
            for key in ("ignore", "only", "order")
        }
        if not any(parsed.values()):
            return

        existing = payload.get("provider")
        provider: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
        for key, slugs in parsed.items():
            if not slugs:
                continue
            prev = provider.get(key)
            if isinstance(prev, list) and prev:
                seen = set(prev)
                provider[key] = list(prev) + [s for s in slugs if s not in seen]
            else:
                provider[key] = slugs
        payload["provider"] = provider
        logger.debug("Applied provider routing custom params: %s", {k: v for k, v in parsed.items() if v})
    except Exception:
        logger.debug("Failed to apply provider routing custom params", exc_info=True)


# Helper Functions

def _model_params_to_dict(params: Any) -> dict[str, Any]:
    """Best-effort conversion of OWUI ModelParams-like objects to a plain dict."""
    if params is None:
        return {}
    if isinstance(params, dict):
        return dict(params)
    model_dump = getattr(params, "model_dump", None)
    if callable(model_dump):
        try:
            dumped = model_dump()
            return dict(dumped) if isinstance(dumped, dict) else {}
        except Exception:
            logger.debug("Could not dump request params", exc_info=True)
            return {}
    return {}

def _get_disable_param(params: Any, key: str) -> bool:
    """Return True when params[key] is truthy as a bool-ish flag."""
    params_dict = _model_params_to_dict(params)
    sentinel = object()
    raw: Any = params_dict.get(key, sentinel)

    if raw is sentinel:
        custom_params = params_dict.get("custom_params")
        if isinstance(custom_params, dict) and key in custom_params:
            raw = custom_params.get(key, None)

    if raw is sentinel:
        for container_key in (_PIPE_METADATA_KEY, "openrouter", "pipe"):
            container = params_dict.get(container_key)
            if isinstance(container, dict) and key in container:
                raw = container.get(key, None)
                break

            custom_params = params_dict.get("custom_params")
            if isinstance(custom_params, dict):
                container = custom_params.get(container_key)
                if isinstance(container, dict) and key in container:
                    raw = container.get(key, None)
                    break
    if raw is sentinel:
        raw = None
    coerced = _coerce_bool(raw)
    return bool(coerced) if coerced is not None else False

def _sanitize_openrouter_metadata(raw: Any) -> dict[str, str] | None:
    """Return a validated OpenRouter `metadata` dict or None.

    OpenRouter's Responses schema documents `metadata` as a string->string map with:
    - max 16 pairs
    - key <= 64 chars, no brackets
    - value <= 512 chars
    """
    if not isinstance(raw, dict):
        return None

    sanitized: dict[str, str] = {}
    for key, value in raw.items():
        if len(sanitized) >= _MAX_OPENROUTER_METADATA_PAIRS:
            break
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        if len(key) > _MAX_OPENROUTER_METADATA_KEY_CHARS:
            continue
        if "[" in key or "]" in key:
            continue
        if len(value) > _MAX_OPENROUTER_METADATA_VALUE_CHARS:
            continue
        sanitized[key] = value

    return sanitized or None

def _apply_identifier_valves_to_payload(
    payload: dict[str, Any],
    *,
    valves: Pipe.Valves,
    owui_metadata: dict[str, Any],
    owui_user_id: str,
    owui_user: Any = None,
    logger: logging.Logger = logger,
) -> None:
    """Mutate request payload to include valve-gated identifiers.

    Rules (per operator requirements):
    - Only emit `metadata` when at least one identifier valve contributes a value.
    - When `SEND_END_USER_ID` is enabled, emit top-level `user` (GUID, email, or
      display name per `END_USER_ID_SOURCE`, falling back to the GUID) and keep
      `metadata.user_id` on the stable GUID.
    - `SEND_SESSION_ID`, `SEND_CHAT_ID`, `SEND_MESSAGE_ID` emit `metadata.<id>` only.
    """
    if not isinstance(payload, dict):
        return
    if not isinstance(owui_metadata, dict):
        owui_metadata = {}

    metadata_out: dict[str, str] = {}

    payload.pop("safety_identifier", None)

    if valves.SEND_END_USER_ID:
        user_value = (owui_user_id or "").strip()
        source = str(getattr(valves, "END_USER_ID_SOURCE", "id") or "id")
        alternate = ""
        if source in ("email", "name") and owui_user is not None:
            if isinstance(owui_user, dict):
                raw_alternate = owui_user.get(source)
            else:
                raw_alternate = getattr(owui_user, source, None)
            if isinstance(raw_alternate, str):
                alternate = raw_alternate.strip()
        display = (
            alternate
            if alternate and len(alternate) <= _MAX_OPENROUTER_ID_CHARS
            else user_value
        )
        if display and len(display) <= _MAX_OPENROUTER_ID_CHARS:
            payload["user"] = display
            payload["safety_identifier"] = display
            if user_value and len(user_value) <= _MAX_OPENROUTER_ID_CHARS:
                metadata_out["user_id"] = user_value
        else:
            payload.pop("user", None)
            logger.debug("SEND_END_USER_ID enabled but OWUI user id missing/invalid; omitting `user`.")
    else:
        payload.pop("user", None)

    pipe_meta = owui_metadata.get(_PIPE_METADATA_KEY)
    is_temporary = is_temporary_chat(owui_metadata.get("chat_id")) or (
        isinstance(pipe_meta, dict) and pipe_meta.get("temporary_chat") is True
    )

    payload.pop("session_id", None)
    if valves.SEND_CACHE_SESSION_ID:
        pin_source: str | None = None
        sticky_chat_id = owui_metadata.get("chat_id")
        if isinstance(sticky_chat_id, str) and sticky_chat_id.strip():
            pin_source = sticky_chat_id.strip()
        else:
            inner_meta = owui_metadata.get(_PIPE_METADATA_KEY)
            if isinstance(inner_meta, dict) and inner_meta.get("fusion_inner"):
                pin_source = None
            else:
                caller_session_id = owui_metadata.get("session_id")
                if isinstance(caller_session_id, str) and caller_session_id.strip():
                    pin_source = "api-session:" + caller_session_id.strip()
        if pin_source:
            sticky = _sticky_session_key(pin_source)
            if sticky:
                payload["session_id"] = sticky

    if valves.SEND_SESSION_ID and not is_temporary:
        session_id = owui_metadata.get("session_id")
        if isinstance(session_id, str):
            candidate = session_id.strip()
            if candidate:
                metadata_out["session_id"] = candidate[:_MAX_OPENROUTER_METADATA_VALUE_CHARS]

    if valves.SEND_CHAT_ID and not is_temporary:
        chat_id = owui_metadata.get("chat_id")
        if isinstance(chat_id, str):
            candidate = chat_id.strip()
            if candidate:
                metadata_out["chat_id"] = candidate[:_MAX_OPENROUTER_METADATA_VALUE_CHARS]

    if valves.SEND_MESSAGE_ID and not is_temporary:
        message_id = owui_metadata.get("message_id")
        if isinstance(message_id, str):
            candidate = message_id.strip()
            if candidate:
                metadata_out["message_id"] = candidate[:_MAX_OPENROUTER_METADATA_VALUE_CHARS]

    if metadata_out:
        payload["metadata"] = metadata_out
    else:
        payload.pop("metadata", None)


def _filter_openrouter_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop any keys not documented for the OpenRouter Responses API."""
    candidate = dict(payload or {})
    existing_text = candidate.get("text")
    if isinstance(existing_text, dict):
        candidate["text"] = dict(existing_text)
    _normalise_openrouter_responses_text_format(candidate)
    verbosity = candidate.get("verbosity")
    if isinstance(verbosity, str) and verbosity.strip():
        existing_text = candidate.get("text")
        text_value: dict[str, Any] = dict(existing_text) if isinstance(existing_text, dict) else {}
        current = text_value.get("verbosity")
        if not isinstance(current, str) or not current.strip():
            text_value["verbosity"] = verbosity.strip()
        candidate["text"] = text_value
        candidate.pop("verbosity", None)
    if candidate.pop("previous_response_id", None) is not None:
        logger.log(
            warn_level(_warned_previous_response_id, "previous_response_id"),
            "Dropped previous_response_id: OpenRouter's /responses endpoint is stateless and "
            "answers a non-null previous_response_id with a 400. The whole conversation is "
            "sent in 'input' instead.",
        )
    filtered: dict[str, Any] = {}
    for key, value in candidate.items():
        if key not in ALLOWED_OPENROUTER_FIELDS:
            continue

        if value is None:
            continue

        if key == "store":
            if value is not False:
                logger.log(
                    warn_level(_warned_store, "store"),
                    "Forced store=false: OpenRouter's /responses endpoint is stateless and "
                    "answers store=true with a 400.",
                )
            value = False

        if key == "top_k":
            coerced = _coerce_openrouter_int(value)
            if coerced is None:
                continue
            filtered[key] = coerced
            continue

        if key == "metadata":
            value = _sanitize_openrouter_metadata(value)
            if value is None:
                continue

        if key == "reasoning":
            if not isinstance(value, dict):
                continue
            allowed_reasoning = {}
            for field_name in ("effort", "max_tokens", "exclude", "enabled", "summary", "context", "mode"):
                if field_name in value:
                    allowed_reasoning[field_name] = value[field_name]
            if not allowed_reasoning:
                continue
            value = allowed_reasoning

        if key == "text":
            if not isinstance(value, dict):
                continue
            if not value:
                continue

        filtered[key] = value

    return filtered


def _filter_replayable_input_items(
    items: Any,
    *,
    logger: logging.Logger = logger,
) -> Any:
    """Strip tool artifacts we must not replay back to the provider."""
    if not isinstance(items, list):
        return items

    from ..requests.transformer import _as_replayed

    filtered: list[dict[str, Any]] = []
    changed = False
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            filtered.append(item)
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type in _NON_REPLAYABLE_TOOL_ARTIFACTS:
            logger.debug(
                "Input sanitizer removed %s artifact at index %d (id=%s).",
                item_type,
                idx,
                item.get("id"),
            )
            changed = True
            continue
        if item_type.startswith("openrouter:"):
            parts = _as_replayed(item)
            if len(parts) != 1 or parts[0] is not item:
                changed = True
            filtered.extend(parts)
            continue
        filtered.append(item)

    return filtered if changed else items


def apply_context_transforms(responses_body: ResponsesBody, *, auto_context_trimming: bool) -> None:
    """Set context trimming fields when not already explicitly configured."""
    if not auto_context_trimming:
        if responses_body.truncation is None:
            responses_body.truncation = "disabled"
        return
    plugins = list(responses_body.plugins or [])
    if not any(isinstance(entry, dict) and entry.get("id") == "context-compression" for entry in plugins):
        plugins.append({"id": "context-compression"})
        responses_body.plugins = plugins


def _parse_url_citation_annotations(raw_annotations: list[Any]) -> Iterator[tuple[str, str, str]]:
    """Parse url_citation annotations, yielding (url, title, content) triples.

    Handles both nested format (``annotation.url_citation.url``) and flat
    format (``annotation.url``).  Validates that url is a non-empty string,
    normalises the title, and surfaces the ``content`` excerpt OpenRouter
    supplies (empty string when absent).
    """
    for raw_ann in raw_annotations:
        if not isinstance(raw_ann, dict):
            continue
        if raw_ann.get("type") != "url_citation":
            continue
        payload = raw_ann.get("url_citation")
        if isinstance(payload, dict):
            url = payload.get("url")
            title = payload.get("title") or url
            content = payload.get("content")
        else:
            url = raw_ann.get("url")
            title = raw_ann.get("title") or url
            content = raw_ann.get("content")
        if not isinstance(url, str) or not url.strip():
            continue
        url = url.strip()
        if isinstance(title, str):
            title = title.strip() or url
        else:
            title = url
        content = content.strip() if isinstance(content, str) else ""
        yield url, title, content


def _unhandled_citation_types(raw_annotations: Any) -> set[str]:
    """Return citation annotation types the pipe cannot render (e.g. ``file_citation``).

    Any annotation whose ``type`` ends in ``_citation`` but is not ``url_citation``
    (the only type we surface today) is reported so callers can notify the user
    instead of silently dropping it.
    """
    types: set[str] = set()
    if not isinstance(raw_annotations, list):
        return types
    for raw_ann in raw_annotations:
        if not isinstance(raw_ann, dict):
            continue
        ann_type = raw_ann.get("type")
        if isinstance(ann_type, str) and ann_type.endswith("_citation") and ann_type != "url_citation":
            types.add(ann_type)
    return types
