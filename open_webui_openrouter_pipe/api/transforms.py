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

import json
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from fastapi import Request
    from ..pipe import Pipe

from ..core.config import (
    _MAX_OPENROUTER_ID_CHARS,
    _MAX_OPENROUTER_METADATA_PAIRS,
    _MAX_OPENROUTER_METADATA_KEY_CHARS,
    _MAX_OPENROUTER_METADATA_VALUE_CHARS,
    _NON_REPLAYABLE_TOOL_ARTIFACTS,
)
from ..core.timing_logger import timed
from ..models.registry import ModelFamily
from ..tools.tool_schema import _strictify_schema
from ..core.utils import _coerce_bool, _parse_model_fallback_csv

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Pydantic Body Classes
# -----------------------------------------------------------------------------

class CompletionsBody(BaseModel):
    """
    Represents the body of a completions request to OpenAI completions API.
    """
    model: str
    messages: List[Dict[str, Any]]
    stream: bool = False
    response_format: Optional[Dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = None
    function_call: Optional[Union[str, Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    model_config = ConfigDict(extra="allow")  # pass through additional OpenAI parameters


class ResponsesBody(BaseModel):
    """
    Represents the body of a responses request to OpenAI Responses API.
    """
    
    # Core parameters
    model: str
    models: Optional[List[str]] = None
    instructions: Optional[str] = None  # system/developer instructions
    input: Union[str, List[Dict[str, Any]]]  # plain text, or rich array

    stream: bool = False                          # SSE chunking
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[float] = None
    min_p: Optional[float] = None
    top_a: Optional[float] = None
    max_output_tokens: Optional[int] = None
    reasoning: Optional[Dict[str, Any]] = None    # {"effort":"high", ...}
    include_reasoning: Optional[bool] = None
    thinking_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    plugins: Optional[List[Dict[str, Any]]] = None
    text: Optional[Dict[str, Any]] = None
    parallel_tool_calls: Optional[bool] = None
    transforms: Optional[List[str]] = None
    user: Optional[str] = None
    session_id: Optional[str] = None

    # OpenRouter /chat/completions parameters (preserved for endpoint fallback).
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    seed: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    structured_outputs: Optional[bool] = None
    reasoning_effort: Optional[str] = None
    verbosity: Optional[str] = None
    web_search_options: Optional[Dict[str, Any]] = None
    stream_options: Optional[Dict[str, Any]] = None
    usage: Optional[Dict[str, Any]] = None

    # OpenRouter routing extras (best-effort passthrough for /chat/completions).
    provider: Optional[Dict[str, Any]] = None
    route: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None
    image_config: Optional[Union[str, float]] = None
    modalities: Optional[List[str]] = None
    model_config = ConfigDict(extra="allow")  # allow additional OpenAI parameters automatically

    @staticmethod
    @timed
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
    @timed
    def _coerce_float_fields(cls, value: Any) -> Any:
        return cls._strip_blank_string(value)

    @field_validator(
        "max_output_tokens",
        "max_tokens",
        "max_completion_tokens",
        "seed",
        "top_logprobs",
        mode="before",
    )
    @classmethod
    @timed
    def _coerce_int_fields(cls, value: Any) -> Any:
        value = cls._strip_blank_string(value)
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError("Boolean is not a valid integer value.")
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(round(value))
        if isinstance(value, str):
            try:
                return int(round(float(value)))
            except ValueError as exc:
                raise ValueError(f"Invalid integer value: {value!r}") from exc
        raise ValueError(f"Invalid integer value: {value!r}")

    @field_validator("models", mode="before")
    @classmethod
    @timed
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
    @timed
    def _normalize_model_id(self) -> "ResponsesBody":
        """Ensure the model name references the canonical base id (prefix/date stripped)."""
        normalized = ModelFamily.base_model(self.model or "")
        if normalized and normalized != self.model:
            self.model = normalized
        return self

    @staticmethod
    @timed
    def transform_owui_tools(__tools__: Dict[str, dict] | None, *, strict: bool = False) -> List[dict]:
        """
        Convert Open WebUI __tools__ registry (dict of entries with {"spec": {...}}) into
        OpenAI Responses-API tool specs: {"type": "function", "name", ...}.

        """
        if not __tools__:
            return []

        tools: List[dict] = []
        for item in __tools__.values():
            spec = item.get("spec") or {}
            name = spec.get("name")
            if not name:
                continue  # skip malformed entries

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
    @timed
    def _convert_function_call_to_tool_choice(function_call: Any) -> Optional[Any]:
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
        completions_body: "CompletionsBody",
        chat_id: Optional[str] = None,
        openwebui_model_id: Optional[str] = None,
        *,
        request: Optional[Request] = None,
        user_obj: Optional[Any] = None,
        event_emitter: Optional[Callable] = None,
        artifact_loader: Optional[
            Callable[[Optional[str], Optional[str], List[str]], Awaitable[Dict[str, Dict[str, Any]]]]
        ] = None,
        pruning_turns: int = 0,
        transformer_context: Optional[Any] = None,
        transformer_valves: Optional["Pipe.Valves"] = None,
        **extra_params,
    ) -> "ResponsesBody":
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

        # Step 1: Remove unsupported fields
        unsupported_fields = {
            # Fields that are not supported by OpenAI Responses API
            "n",
            "suffix", # Responses API does not support suffix
            "function_call", # Deprecated in favor of 'tool_choice'.
            "functions", # Deprecated in favor of 'tools'.

            # Fields that are dropped and manually handled in step 2.
            "reasoning_effort", "max_tokens",

            # Fields that are dropped and manually handled later in the pipe()
            "extra_tools", # Not a real OpenAI parm. Upstream filters may use it to add tools. The are appended to body["tools"] later in the pipe()

            # Fields not documented in OpenRouter's Responses API reference
            "store", "truncation",
            "user",
        }
        sanitized_params = {}
        for key, value in completions_dict.items():
            if key in unsupported_fields:
                logging.warning(f"Dropping unsupported parameter: '{key}'")
            else:
                sanitized_params[key] = value

        # Step 2: Apply transformations
        # Rename max_tokens -> max_output_tokens
        if "max_tokens" in completions_dict:
            sanitized_params["max_output_tokens"] = completions_dict["max_tokens"]

        # reasoning_effort -> reasoning.effort (without overwriting existing effort)
        effort = completions_dict.get("reasoning_effort")
        if effort:
            reasoning = sanitized_params.get("reasoning", {})
            reasoning.setdefault("effort", effort)
            sanitized_params["reasoning"] = reasoning

        # Legacy function_call -> modern tool_choice
        if "tool_choice" not in sanitized_params and "function_call" in completions_dict:
            converted_choice = ResponsesBody._convert_function_call_to_tool_choice(
                completions_dict.get("function_call")
            )
            if converted_choice is not None:
                sanitized_params["tool_choice"] = converted_choice

        # Transform input messages to OpenAI Responses API format
        if "messages" in completions_dict:
            sanitized_params.pop("messages", None)
            replayed_reasoning_refs: list[Tuple[str, str]] = []
            # Resolve the transformer context. When provided, the transformer_context is typically
            # the Pipe instance so helper methods (upload, emit_status, etc.) are available.
            if transformer_context is None:
                raise RuntimeError(
                    "ResponsesBody.from_completions requires a transformer_context (usually the Pipe instance) "
                    "so multimodal helpers (file uploads, status events, etc.) are available."
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

            sanitized_params["input"] = await transformer_owner.transform_messages_to_input(
                filtered_messages,
                chat_id=chat_id,
                openwebui_model_id=openwebui_model_id,
                artifact_loader=artifact_loader,
                pruning_turns=pruning_turns,
                replayed_reasoning_refs=replayed_reasoning_refs,
                __request__=request,
                user_obj=user_obj,
                event_emitter=event_emitter,
                model_id=completions_dict.get("model"),
                valves=transformer_valves or getattr(transformer_owner, "valves", None),
            )
            if replayed_reasoning_refs:
                sanitized_params["_replayed_reasoning_refs"] = replayed_reasoning_refs

        # Apply explicit overrides (e.g. custom Open WebUI model parameters) and then
        # normalise fields to the OpenRouter `/responses` schema.
        merged_params = {
            **sanitized_params,
            **extra_params,  # Extra parameters (e.g. custom Open WebUI model settings)
        }
        # Normalize OpenAI chat tool_choice shape -> OpenRouter /responses tool_choice shape.
        # OWUI uses: {"type":"function","function":{"name":"..."}}.
        # OpenRouter /responses expects: {"type":"function","name":"..."}.
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

        # Normalize OpenAI chat tools schema -> OpenRouter /responses tools schema.
        # OWUI sends: [{"type":"function","function":{...}}].
        # OpenRouter /responses expects: [{"type":"function","name":"...","parameters":{...}}].
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
    "preset",
    "text",
    "parallel_tool_calls",
    "user",
    "session_id",
    "transforms",
}

ALLOWED_OPENROUTER_CHAT_FIELDS = {
    # Core OpenAI-style chat completion fields (OpenRouter supports additional provider routing keys too).
    "model",
    "models",
    "messages",
    "stream",
    "stream_options",
    "usage",
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
    "plugins",
    "user",
    "session_id",
    "metadata",
    # OpenRouter routing extras (best-effort pass-through).
    "provider",
    "route",
    "debug",
    "image_config",
    "modalities",
    # Keep transforms for compatibility; OpenRouter may ignore if unsupported.
    "transforms",
}



# -----------------------------------------------------------------------------
# Request Filtering
# -----------------------------------------------------------------------------

@timed
def _filter_openrouter_chat_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Filter payload to fields accepted by OpenRouter's /chat/completions."""
    if not isinstance(payload, dict):
        return {}
    filtered: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in ALLOWED_OPENROUTER_CHAT_FIELDS:
            filtered[key] = value
    return filtered



@timed
def _strip_disable_model_settings_params(payload: dict[str, Any]) -> None:
    """Remove OWUI-local per-model control flags from the outbound provider payload."""
    if not isinstance(payload, dict):
        return
    for key in (
        "disable_model_metadata_sync",
        "disable_capability_updates",
        "disable_image_updates",
        "disable_openrouter_search_auto_attach",
        "disable_openrouter_search_default_on",
        "disable_direct_uploads_auto_attach",
        "disable_description_updates",
        "disable_native_websearch",
        "disable_native_web_search",
    ):
        payload.pop(key, None)

# -----------------------------------------------------------------------------
# Tool Schema Transforms
# -----------------------------------------------------------------------------

@timed
def _responses_tools_to_chat_tools(tools: Any) -> list[dict[str, Any]]:
    """Convert Responses API tools -> Chat Completions tools schema."""
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        function: dict[str, Any] = {
            "name": name,
        }
        if isinstance(tool.get("description"), str):
            function["description"] = tool["description"]
        if isinstance(tool.get("parameters"), dict):
            function["parameters"] = tool["parameters"]
        out.append({"type": "function", "function": function})
    return out



@timed
def _chat_tools_to_responses_tools(tools: Any) -> list[dict[str, Any]]:
    """Convert Chat Completions `tools` schema -> Responses API tools schema."""
    if not isinstance(tools, list):
        return []
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue

        fn = tool.get("function")
        name = tool.get("name")
        if not isinstance(name, str) or not name.strip():
            if isinstance(fn, dict):
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
        if isinstance(parameters, dict):
            spec["parameters"] = parameters

        out.append(spec)

    return out



@timed
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



# -----------------------------------------------------------------------------
# Response Format Transforms
# -----------------------------------------------------------------------------

@timed
def _chat_response_format_to_responses_text_format(value: Any) -> Optional[dict[str, Any]]:
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

@timed
def _responses_text_format_to_chat_response_format(value: Any) -> Optional[dict[str, Any]]:
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

@timed
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
        LOGGER.warning(
            "Dropping invalid `text.format` on /responses request payload."
        )

    final_format: Optional[dict[str, Any]] = None
    if existing_canonical is not None:
        final_format = dict(existing_canonical)
        if response_format_as_text is not None and existing_canonical != response_format_as_text:
            LOGGER.warning(
                "Conflicting structured output config: preferring `text.format` over `response_format` for /responses."
            )
    elif response_format_as_text is not None:
        final_format = dict(response_format_as_text)

    payload.pop("response_format", None)

    if final_format is None:
        if isinstance(existing_text, dict) and len(existing_text) == 0:
            payload.pop("text", None)
        return

    existing_text["format"] = final_format
    payload["text"] = existing_text



# -----------------------------------------------------------------------------
# Message and Input Transforms
# -----------------------------------------------------------------------------

@timed
def _responses_input_to_chat_messages(input_value: Any) -> list[dict[str, Any]]:
    """Convert Responses API input array -> Chat Completions messages array.

    Best-effort mapping for supported multimodal blocks. Unsupported blocks are
    degraded to text notes rather than dropped.
    """
    if input_value is None:
        return []
    if isinstance(input_value, str):
        text = input_value.strip()
        return [{"role": "user", "content": text}] if text else []
    if not isinstance(input_value, list):
        return []

    messages: list[dict[str, Any]] = []

    @timed
    def _to_text_block(text: str, *, cache_control: Any = None) -> dict[str, Any]:
        block: dict[str, Any] = {"type": "text", "text": text}
        if isinstance(cache_control, dict) and cache_control:
            block["cache_control"] = dict(cache_control)
        return block

    for item in input_value:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")

        if itype == "message":
            role = (item.get("role") or "").strip().lower()
            if not role:
                continue
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
            raw_content = item.get("content")
            if isinstance(raw_content, str):
                msg: dict[str, Any] = {"role": role, "content": raw_content}
                if msg_annotations:
                    msg["annotations"] = msg_annotations
                if msg_reasoning_details:
                    msg["reasoning_details"] = msg_reasoning_details
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
                            blocks_out.append(
                                _to_text_block(text, cache_control=block.get("cache_control"))
                            )
                        continue
                    if btype == "input_image":
                        url = block.get("image_url")
                        if isinstance(url, str) and url.strip():
                            image_url_obj: dict[str, Any] = {"url": url.strip()}
                            detail = block.get("detail")
                            if isinstance(detail, str) and detail in {"auto", "low", "high"}:
                                image_url_obj["detail"] = detail
                            blocks_out.append({"type": "image_url", "image_url": image_url_obj})
                        continue
                    if btype == "image_url":
                        image_url_val = block.get("image_url")
                        if isinstance(image_url_val, dict):
                            blocks_out.append({"type": "image_url", "image_url": dict(image_url_val)})
                        elif isinstance(image_url_val, str) and image_url_val.strip():
                            blocks_out.append({"type": "image_url", "image_url": {"url": image_url_val.strip()}})
                        continue
                    if btype == "input_audio":
                        audio = block.get("input_audio")
                        if isinstance(audio, dict):
                            blocks_out.append({"type": "input_audio", "input_audio": dict(audio)})
                        continue
                    if btype == "video_url":
                        video_url = block.get("video_url")
                        if isinstance(video_url, dict):
                            blocks_out.append({"type": "video_url", "video_url": dict(video_url)})
                        elif isinstance(video_url, str) and video_url.strip():
                            blocks_out.append({"type": "video_url", "video_url": {"url": video_url.strip()}})
                        continue
                    if btype == "input_file":
                        filename = block.get("filename")
                        file_data = block.get("file_data")
                        file_url = block.get("file_url")
                        file_payload: dict[str, Any] = {}
                        if isinstance(filename, str) and filename.strip():
                            file_payload["filename"] = filename.strip()
                        file_value: Optional[str] = None
                        if isinstance(file_data, str) and file_data.strip():
                            file_value = file_data.strip()
                        elif isinstance(file_url, str) and file_url.strip():
                            file_value = file_url.strip()
                        if file_value:
                            file_payload["file_data"] = file_value
                        if file_payload:
                            blocks_out.append({"type": "file", "file": file_payload})
                        continue

            if not blocks_out:
                # If the role message had no supported blocks, keep shape with empty content.
                msg: dict[str, Any] = {"role": role, "content": ""}
                if msg_annotations:
                    msg["annotations"] = msg_annotations
                if msg_reasoning_details:
                    msg["reasoning_details"] = msg_reasoning_details
                messages.append(msg)
            else:
                msg = {"role": role, "content": blocks_out}
                if msg_annotations:
                    msg["annotations"] = msg_annotations
                if msg_reasoning_details:
                    msg["reasoning_details"] = msg_reasoning_details
                messages.append(msg)
            continue

        if itype == "function_call_output":
            # Responses tool output -> chat tool message.
            call_id = item.get("call_id")
            output = item.get("output")
            if isinstance(call_id, str) and call_id.strip():
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id.strip(),
                        "content": output if isinstance(output, str) else (json.dumps(output, ensure_ascii=False) if output is not None else ""),
                    }
                )
            continue

        if itype == "function_call":
            # Responses tool call -> assistant tool_calls message.
            call_id = item.get("call_id") or item.get("id")
            name = item.get("name")
            args = item.get("arguments")
            if not (isinstance(call_id, str) and call_id.strip() and isinstance(name, str) and name.strip()):
                continue
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False) if args is not None else "{}"
            messages.append(
                {
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
            )
            continue

        # Drop other non-message artifacts (reasoning, web_search_call, etc.) by default.

    return messages



# -----------------------------------------------------------------------------
# Payload Transforms
# -----------------------------------------------------------------------------

@timed
def _responses_payload_to_chat_completions_payload(
    responses_payload: dict[str, Any],
) -> dict[str, Any]:
    """Convert a Responses API request payload into a Chat Completions payload."""
    if not isinstance(responses_payload, dict):
        return {}
    chat_payload: dict[str, Any] = {}

    # Core routing and identifiers
    for key in ("model", "models", "user", "session_id", "metadata", "plugins", "provider", "route", "debug", "image_config", "modalities", "transforms"):
        if key in responses_payload:
            chat_payload[key] = responses_payload[key]

    # Streaming flags
    stream = bool(responses_payload.get("stream"))
    chat_payload["stream"] = stream
    if stream:
        existing_stream_options = responses_payload.get("stream_options")
        if isinstance(existing_stream_options, dict):
            stream_options = dict(existing_stream_options)
        else:
            stream_options = {}
        stream_options.setdefault("include_usage", True)
        chat_payload["stream_options"] = stream_options

    # OpenRouter usage accounting (enables cost + cached/reasoning token metrics).
    existing_usage = responses_payload.get("usage")
    if isinstance(existing_usage, dict):
        merged_usage = dict(existing_usage)
        merged_usage["include"] = True
        chat_payload["usage"] = merged_usage
    else:
        chat_payload["usage"] = {"include": True}

    # Sampling + misc OpenAI params (may exist as extra fields on ResponsesBody)
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
        "web_search_options",
        "parallel_tool_calls",
    )
    for key in passthrough:
        if key in responses_payload:
            chat_payload[key] = responses_payload[key]

    @timed
    def _coerce_chat_int_param(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(round(value))
        if isinstance(value, str):
            candidate = value.strip()
            if not candidate:
                return None
            try:
                return int(round(float(candidate)))
            except ValueError:
                return value
        return value

    # OpenRouter's /chat/completions historically expects integer-ish values for some params
    # (notably top_k), while /responses schemas may accept floats. Round for chat.
    for key in ("top_k", "seed", "top_logprobs", "max_tokens", "max_completion_tokens"):
        if key not in chat_payload:
            continue
        rounded = _coerce_chat_int_param(chat_payload.get(key))
        if rounded is None:
            chat_payload.pop(key, None)
        else:
            chat_payload[key] = rounded

    # Structured outputs adapter: `/responses` uses `text.format` while `/chat/completions`
    # uses `response_format`. Prefer endpoint-native fields when both exist.
    existing_response_format = chat_payload.get("response_format")
    if existing_response_format is not None and _chat_response_format_to_responses_text_format(existing_response_format) is None:
        chat_payload.pop("response_format", None)
        existing_response_format = None
        LOGGER.warning("Dropping invalid `response_format` on /chat/completions payload.")

    responses_text = responses_payload.get("text")
    if isinstance(responses_text, dict):
        mapped_response_format = _responses_text_format_to_chat_response_format(responses_text.get("format"))
        if existing_response_format is None:
            if mapped_response_format is not None:
                chat_payload["response_format"] = mapped_response_format
        elif mapped_response_format is not None and existing_response_format != mapped_response_format:
            LOGGER.warning(
                "Conflicting structured output config: preferring `response_format` over `text.format` for /chat/completions."
            )

        # Best-effort mapping for text verbosity when falling back to /chat/completions.
        if "verbosity" not in chat_payload:
            verbosity = responses_text.get("verbosity")
            if isinstance(verbosity, str) and verbosity.strip():
                chat_payload["verbosity"] = verbosity.strip()

    # Token limit mapping
    max_output_tokens = responses_payload.get("max_output_tokens")
    if max_output_tokens is not None:
        chat_payload["max_tokens"] = max_output_tokens

    # Tools
    tools = responses_payload.get("tools")
    chat_tools = _responses_tools_to_chat_tools(tools)
    if chat_tools:
        chat_payload["tools"] = chat_tools
    tool_choice = _responses_tool_choice_to_chat_tool_choice(responses_payload.get("tool_choice"))
    if tool_choice is not None:
        chat_payload["tool_choice"] = tool_choice

    # Input -> messages
    chat_payload["messages"] = _responses_input_to_chat_messages(responses_payload.get("input"))

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
                prepend_blocks = [{"type": "text", "text": instructions}]
                if existing_content:
                    prepend_blocks.append({"type": "text", "text": ""})
                system_msg["content"] = prepend_blocks + list(existing_content)
            else:
                system_msg["content"] = instructions
        else:
            messages.insert(0, {"role": "system", "content": instructions})

    return chat_payload


# -----------------------------------------------------------------------------
# Model Fallback
# -----------------------------------------------------------------------------

@timed
def _apply_model_fallback_to_payload(payload: dict[str, Any], *, logger: logging.Logger = LOGGER) -> None:
    """Map OWUI custom `model_fallback` (CSV string) to OpenRouter `models` (array).

    OpenRouter supports `model` plus `models` where `models` is treated as the fallback list.
    This helper never prepends `model` into `models`.
    """
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

    # Merge existing + fallback (dedupe, preserve order). Existing models come first.
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



# -----------------------------------------------------------------------------
# Feature Application to Payloads
# -----------------------------------------------------------------------------

@timed
def _apply_disable_native_websearch_to_payload(
    payload: dict[str, Any],
    *,
    logger: logging.Logger = LOGGER,
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



# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

@timed
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
            return {}
    return {}

@timed
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
        # Some operators may choose to namespace pipe settings inside params JSON.
        for container_key in ("openrouter_pipe", "openrouter", "pipe"):
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

@timed
def _sanitize_openrouter_metadata(raw: Any) -> Optional[dict[str, str]]:
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

@timed
def _apply_identifier_valves_to_payload(
    payload: dict[str, Any],
    *,
    valves: "Pipe.Valves",
    owui_metadata: dict[str, Any],
    owui_user_id: str,
    logger: logging.Logger = LOGGER,
) -> None:
    """Mutate request payload to include valve-gated identifiers.

    Rules (per operator requirements):
    - Only emit `metadata` when at least one identifier valve contributes a value.
    - When `SEND_END_USER_ID` is enabled, emit both top-level `user` and `metadata.user_id`.
    - When `SEND_SESSION_ID` is enabled, emit both top-level `session_id` and `metadata.session_id`.
    - `chat_id` and `message_id` have no OpenRouter top-level fields; they go into `metadata` only.
    """
    if not isinstance(payload, dict):
        return
    if not isinstance(owui_metadata, dict):
        owui_metadata = {}

    metadata_out: dict[str, str] = {}

    if valves.SEND_END_USER_ID:
        candidate = (owui_user_id or "").strip()
        if candidate and len(candidate) <= _MAX_OPENROUTER_ID_CHARS:
            payload["user"] = candidate
            metadata_out["user_id"] = candidate
        else:
            payload.pop("user", None)
            logger.debug("SEND_END_USER_ID enabled but OWUI user id missing/invalid; omitting `user`.")
    else:
        payload.pop("user", None)

    if valves.SEND_SESSION_ID:
        session_id = owui_metadata.get("session_id")
        if isinstance(session_id, str):
            candidate = session_id.strip()
            if candidate and len(candidate) <= _MAX_OPENROUTER_ID_CHARS:
                payload["session_id"] = candidate
                metadata_out["session_id"] = candidate
            else:
                payload.pop("session_id", None)
                logger.debug("SEND_SESSION_ID enabled but OWUI session_id missing/invalid; omitting `session_id`.")
        else:
            payload.pop("session_id", None)
    else:
        payload.pop("session_id", None)

    if valves.SEND_CHAT_ID:
        chat_id = owui_metadata.get("chat_id")
        if isinstance(chat_id, str):
            candidate = chat_id.strip()
            if candidate:
                metadata_out["chat_id"] = candidate[:_MAX_OPENROUTER_METADATA_VALUE_CHARS]

    if valves.SEND_MESSAGE_ID:
        message_id = owui_metadata.get("message_id")
        if isinstance(message_id, str):
            candidate = message_id.strip()
            if candidate:
                metadata_out["message_id"] = candidate[:_MAX_OPENROUTER_METADATA_VALUE_CHARS]

    if metadata_out:
        # Let the central sanitizer enforce length/bracket/pair constraints.
        payload["metadata"] = metadata_out
    else:
        payload.pop("metadata", None)



@timed
def _filter_openrouter_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Drop any keys not documented for the OpenRouter Responses API."""
    candidate = dict(payload or {})
    _normalise_openrouter_responses_text_format(candidate)
    filtered: Dict[str, Any] = {}
    for key, value in candidate.items():
        if key not in ALLOWED_OPENROUTER_FIELDS:
            continue

        # Drop explicit nulls; OpenRouter rejects nulls for optional fields.
        if value is None:
            continue

        if key == "top_k":
            if isinstance(value, (int, float)):
                filtered[key] = float(value)
                continue
            if isinstance(value, str):
                candidate = value.strip()
                if not candidate:
                    continue
                try:
                    filtered[key] = float(candidate)
                except ValueError:
                    continue
                continue
            continue

        if key == "metadata":
            value = _sanitize_openrouter_metadata(value)
            if value is None:
                continue

        if key == "reasoning":
            if not isinstance(value, dict):
                continue
            allowed_reasoning = {}
            for field_name in ("effort", "max_tokens", "exclude", "enabled", "summary"):
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



@timed
def _filter_replayable_input_items(
    items: Any,
    *,
    logger: logging.Logger = LOGGER,
) -> Any:
    """Strip tool artifacts we must not replay back to the provider."""
    if not isinstance(items, list):
        return items

    filtered: list[dict[str, Any]] = []
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
            continue
        filtered.append(item)

    if len(filtered) != len(items):
        return filtered
    return items
