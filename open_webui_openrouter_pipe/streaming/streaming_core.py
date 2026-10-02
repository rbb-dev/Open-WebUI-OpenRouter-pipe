"""Core streaming loop and response handling.

This module provides StreamingHandler for SSE streaming, non-streaming responses,
and endpoint selection.
"""

from __future__ import annotations

import asyncio
import binascii
import contextlib
import copy
import functools
import json
import logging
import random
import re
import time
import uuid
from collections.abc import AsyncGenerator
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal

import aiohttp
from fastapi import Request

if TYPE_CHECKING:
    from ..api.transforms import ResponsesBody
    from ..pipe import Pipe
else:
    Pipe = Any
    ResponsesBody = Any

# Import error classes
# Import transform functions
from ..api.transforms import (
    _apply_disable_native_websearch_to_payload,
    _apply_identifier_valves_to_payload,
    _apply_model_fallback_to_payload,
    _apply_openrouter_trace_to_payload,
    _apply_provider_routing_params_to_payload,
    _drop_include_reasoning_for_unsupported_fallbacks,
    _parse_url_citation_annotations,
    _strip_disable_model_settings_params,
    _unhandled_citation_types,
    responses_refusal_text,
)

# Import config classes
from ..core.config import (
    _NON_REPLAYABLE_TOOL_ARTIFACTS,
    _PIPE_METADATA_KEY,
    _RAW_REPLAYED_SERVER_TOOLS,
    DEFAULT_STREAM_INTERRUPTED_TEMPLATE,
    NO_CONTENT_AFTER_TOOLS_FALLBACK,
)
from ..core.context_budget import (
    apply_live_tool_output_budget,
    build_futility_notice,
    effective_chars_per_token,
    estimate_serialized_chars,
    measure_chars_per_token,
    omitted_tool_names,
    record_chars_per_token,
)

# Import costs helper
from ..core.costs import maybe_dump_costs_snapshot
from ..core.errors import (
    OpenRouterAPIError,
    RequiredInternalFileError,
    StatusMessages,
    UpstreamBodyUnreadable,
)

# Import SessionLogger
from ..core.logging_system import SessionLogger, bounded_log_record_text

# Import timing instrumentation
from ..core.timing_logger import clear_timing_events, timed, timing_mark
from ..core.url_scheme import is_http_or_https_url, loggable_link

# Imports from core.utils
from ..core.utils import (
    BUILTIN_ASK_USER_ROUND_KEY,
    CONTINUED_REPLY,
    IMAGE_NO_IMAGES_REASON,
    OWUI_UNRESOLVABLE_CALL_STATUSES,
    PIPE_ONLY_TOOL_ROUND_KEY,
    REASONING_ANCHOR_SEQ_KEY,
    REASONING_FOLLOWING_ORDINAL_KEY,
    REASONING_FOLLOWING_SERVER_ITEM_KEY,
    REASONING_PRECEDING_ORDINAL_KEY,
    REASONING_TEXT_ORDINAL_KEY,
    _data_url_log_subject,
    _image_item_is_empty,
    _redact_payload_blobs,
    _safe_json_loads,
    _serialize_marker,
    _serialize_phase_marker,
    citation_access_stamp,
    continued_turn_counts,
    current_turn_items,
    is_picture_output,
    join_answer_and_card,
    merge_usage_stats,
    owui_call_status,
    parse_tool_arguments,
    picture_output,
    recorded_tool_text,
    server_tool_arguments,
    server_tool_call_id,
    server_tool_result_text,
    server_tool_status,
    split_tool_argument_objects,
    strip_hidden_marker_lines,
    tool_output_text_and_pictures,
    wrap_code_block,
)

# Import Anthropic integration
from ..integrations.anthropic import _maybe_apply_anthropic_prompt_caching

# Imports from models.registry
from ..models.registry import (
    ModelFamily,
    _matches_any_model_pattern,
    _parse_model_patterns,
)

# Import request sanitizer
from ..requests.sanitizer import (
    _request_overhead_chars,
    _sanitize_request_input,
    budget_model_id,
)
from ..requests.transformer import (
    _gate_round_output_pictures,
    _tool_picture_notice,
    _tool_picture_verdicts_for_input,
)

_OWUI_ORIGIN_SOURCES = frozenset({"owui_registry_tools", "owui_request_tools"})

ASK_USER_ROUND_NAME = "ask_user"
_BUILTIN_ASK_USER_NAMES_KEY = "_pipe_builtin_ask_user_names"

_FUSION_PANEL_FAILURE_REASON = (
    "Every Fusion panel member failed; this run has no deliberated answer."
)

_STREAM_INTERRUPTED_REASON = "Stream ended without completion event."


def _segment_status(
    was_cancelled: bool,
    error_occurred: bool,
    fusion_no_usable_member: bool,
    handed_back: bool,
) -> str:
    if was_cancelled:
        return "cancelled"
    if error_occurred or fusion_no_usable_member:
        return "error"
    if handed_back:
        return "needs_tool"
    return "complete"


def _generation_status(
    was_cancelled: bool,
    error_occurred: bool,
    fusion_no_usable_member: bool,
) -> str:
    if was_cancelled:
        return "cancelled"
    if error_occurred or fusion_no_usable_member:
        return "failed"
    return "ok"


def _is_no_usable_member_event(event_source: Any, etype: Any, event: Any) -> bool:
    if event_source is None:
        return False
    if etype != "response.output_text.done":
        return False
    return bool(event.get("no_usable_member"))


def _member_notice_text(event: Any) -> str:
    raw = event.get("data")
    data = raw if isinstance(raw, dict) else {}
    text = data.get("content")
    return text if isinstance(text, str) and text else ""


# Imports from storage.persistence
from ..storage.multimodal import (
    _SNIFF_PREFIX_BYTES,
    _decode_base64_in_quanta,
    _sniff_evidence,
    canonical_image_mime,
    head_is_not_a_picture,
    image_extension_for_mime,
    resolve_download_type,
)
from ..storage.owui_files import is_channel_chat, is_linkable_chat, is_temporary_chat
from ..storage.persistence import (
    _UNCLAIMED_LATCH,
    generate_item_id,
    normalize_persisted_item,
)
from ..tools.citation_harvester import (
    BUILTIN_CITATION_TOOLS,
    UNCITED_TOOLS,
    harvest_tool_citations,
)
from ..tools.tool_registry import open_webui_runs_the_calls
from .constants import (
    _REPLAY_DROPPED_OPENING,
    DEFERRED_REASONING_FLUSH,
    FUSION_EMBED_ATTEMPTS,
    ReasoningStatusThrottle,
)

# Import EventEmitter type alias
from .event_emitter import _UNGUARDED_ATTR, EventEmitter
from .fusion_embed import (
    FusionDeliberationState,
    FusionDeltaBatcher,
    build_fusion_embed_html,
)

# Import Open WebUI models
try:
    from open_webui.models.chats import Chats  # type: ignore[import-not-found]
except ImportError:
    Chats = None  # type: ignore
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.models.chats failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    Chats = None  # type: ignore

try:
    from open_webui.utils.middleware import (
        get_citation_source_from_tool_result,  # type: ignore[import-not-found]
    )
except ImportError:
    get_citation_source_from_tool_result = None  # type: ignore
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.utils.middleware failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    get_citation_source_from_tool_result = None  # type: ignore

try:
    from open_webui.utils.middleware import (
        apply_source_context_to_messages as _owui_apply_source_context,  # type: ignore[import-not-found]
    )
except ImportError:
    _owui_apply_source_context = None  # type: ignore
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.utils.middleware failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    _owui_apply_source_context = None  # type: ignore


_monotonic = time.monotonic


def _reply_key(user_id: Any, chat_id: Any, message_id: Any) -> tuple[Any, Any, Any]:
    return (user_id, "" if is_temporary_chat(chat_id) else chat_id, message_id)


def _citation_host(url: str) -> str:
    """Extract a display host from a citation URL, dropping a leading "www.".

    Uses ``removeprefix`` (not ``lstrip``, which strips a character *set* and
    would corrupt hosts beginning with 'w'/'.', e.g. weather.com -> eather.com).
    """
    host = url.split("//", 1)[-1].split("/", 1)[0].lower()
    return host.removeprefix("www.")


def _owui_citations_enabled(model_block: Any) -> bool:
    if not isinstance(model_block, dict):
        return True
    return bool(
        (((model_block.get("info") or {}).get("meta") or {}).get("capabilities") or {}).get(
            "citations", True
        )
    )


def _append_hidden_marker_block(text: str, marker: str) -> str:
    """Append a hidden marker line to text using the same spacing as ULID markers."""
    if text:
        if not text.endswith("\n"):
            text += "\n"
        if not text.endswith("\n\n"):
            text += "\n"
    text += marker
    if not text.endswith("\n"):
        text += "\n"
    return text


def _append_hidden_marker_lines(text: str, markers: list[str]) -> str:
    """Append multiple hidden marker lines to text."""
    updated = text
    for marker in markers:
        updated = _append_hidden_marker_block(updated, marker)
    return updated


def _phase_marker_for_output_item(item: dict[str, Any]) -> str | None:
    """Return the hidden phase marker for a stored assistant output item."""
    if item.get("type") != "message" or item.get("role") != "assistant":
        return None
    if "phase" not in item:
        return None

    phase_value = item.get("phase")
    if phase_value is None:
        return _serialize_phase_marker(None)
    if isinstance(phase_value, str):
        normalized_phase = phase_value.strip()
        if normalized_phase in {"commentary", "final_answer"}:
            return _serialize_phase_marker(normalized_phase)
    return None


def _read_arguments_as_open_webui_reads_them(item: dict[str, Any]) -> str:
    arguments = item.get("arguments", "{}")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return arguments.strip() or "{}"


_TOOL_RESULT_LOG_MAX_CHARS = 16_384


_SHELL_CALL_ARTIFACTS = frozenset(
    {
        "shell_call",
        "local_shell_call",
        "shell_call_output",
        "local_shell_call_output",
    }
)

_SHELL_CALL_OUTPUT_ARTIFACTS = frozenset({"shell_call_output", "local_shell_call_output"})


def _shell_call_commands(item: dict[str, Any]) -> list[str]:
    action = item.get("action")
    if not isinstance(action, dict):
        return []
    for key in ("commands", "command"):
        raw = action.get(key)
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, list):
            return [str(part) for part in raw]
    return []


def _shell_call_result_text(item: dict[str, Any]) -> str:
    output = item.get("output")
    if isinstance(output, str):
        return output
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for entry in output:
        if isinstance(entry, str):
            parts.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        for key in ("stdout", "stderr"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                parts.append(value)
    return "\n".join(parts)


def _tool_result_for_log(output: dict[str, Any]) -> str:
    text = tool_output_text_and_pictures(output.get("output"))[0]
    limit = _TOOL_RESULT_LOG_MAX_CHARS
    if len(text) > limit:
        omitted = len(text) - limit
        text = f"{text[:limit]}\n...(truncated: {omitted:,} characters omitted)..."
    return wrap_code_block(_data_url_log_subject(text))


def _stored_citation_identity(entry: Any) -> str:
    if isinstance(entry, dict):
        source = entry.get("source")
        if isinstance(source, dict):
            url = source.get("url")
            if isinstance(url, str) and url:
                return url
    return json.dumps(entry, sort_keys=True, default=str)


def _stored_entry_identity(entry: Any) -> str:
    return json.dumps(entry, sort_keys=True, default=str)


def _merge_stored_field(stored: Any, fresh: Any, *, key_fn: Any) -> list[Any]:
    merged: list[Any] = list(stored) if isinstance(stored, list) else []
    seen = {key_fn(entry) for entry in merged}
    for entry in fresh:
        identity = key_fn(entry)
        if identity in seen:
            continue
        seen.add(identity)
        merged.append(entry)
    return merged


_TURN_METADATA_FIELDS: tuple[tuple[str, str, str, Any], ...] = (
    ("sources", "citations", "citations", _stored_citation_identity),
    ("annotations", "annotations", "file annotations", _stored_entry_identity),
    ("reasoning_details", "reasoning_details", "reasoning details", _stored_entry_identity),
)


def _joined_labels(labels: list[str]) -> str:
    if len(labels) < 2:
        return labels[0] if labels else ""
    return ", ".join(labels[:-1]) + " or " + labels[-1]


async def _aclose_quietly(it: AsyncGenerator[dict[str, Any], None]) -> None:
    await it.aclose()


_REASONING_HEAD_CHARS = 1024


class _ReasoningTextBox:
    __slots__ = ("head", "length", "parts", "tail")

    def __init__(self) -> None:
        self.parts: list[str] = []
        self.length = 0
        self.head = ""
        self.tail = ""

    def add(self, append: str) -> None:
        if len(self.head) < _REASONING_HEAD_CHARS:
            self.head += append[: _REASONING_HEAD_CHARS - len(self.head)]
        self.parts.append(append)
        self.length += len(append)
        self.tail = (self.tail + append)[-_REASONING_HEAD_CHARS:]

    def text(self) -> str:
        return "".join(self.parts)

    def ends_with(self, candidate: str) -> bool:
        if not candidate or len(candidate) > len(self.tail):
            return False
        return self.tail.endswith(candidate)

    def is_prefix_of(self, text: str) -> bool:
        if not text or len(text) < self.length:
            return False
        offset = 0
        for part in self.parts:
            if text[offset : offset + len(part)] != part:
                return False
            offset += len(part)
        return True


class StreamingHandler:
    """Manages streaming response processing.

    This class encapsulates the core streaming logic including:
    - SSE event parsing and delta accumulation
    - Tool call extraction
    - Reasoning status tracking
    - Usage metrics aggregation
    - Endpoint selection (/responses vs /chat/completions)

    Dependencies:
        - logger: Logger instance for diagnostic output
        - valves: Configuration valves (Pipe.Valves instance)
        - model_registry: Model capability registry
        - Various Pipe methods and utilities
    """
    
    def __init__(
        self,
        logger: logging.Logger,
        valves: Any,
        model_registry: Any,
        pipe_instance: Any,
        valves_owner: Any | None = None,
    ):
        """Initialize StreamingHandler with dependencies.

        Args:
            logger: Logger instance for diagnostic output
            valves: Configuration valves (Pipe.Valves instance)
            model_registry: Model capability registry
            pipe_instance: Reference to parent Pipe instance for helper methods
        """
        self.logger = logger
        self._valves = valves
        self._valves_owner = valves_owner
        self._model_registry = model_registry
        self._pipe = pipe_instance

    @property
    def valves(self) -> Any:
        live = getattr(self._valves_owner, "valves", None)
        return self._valves if live is None else live

    def _audit_orphan_tool_cards(
        self,
        emitted_tool_call_items: set[str],
        emitted_tool_output_items: set[str],
    ) -> None:
        """Warn when a tool-card start was emitted without a matching function_call_output.

        Extracted from the closure-local audit so it's directly testable without
        having to desync internal sets through the public stream surface.
        """
        unmatched = emitted_tool_call_items - emitted_tool_output_items
        if unmatched:
            self.logger.warning(
                "Tool card(s) emitted without matching function_call_output: %s",
                sorted(unmatched),
            )

    @timed
    async def _run_streaming_loop(  # pyright: ignore[reportGeneralTypeIssues]
        self,
        body: ResponsesBody,
        valves: Pipe.Valves,
        event_emitter: EventEmitter | None,
        metadata: dict[str, Any] | None = None,
        tools: dict[str, dict[str, Any]] | list[dict[str, Any]] | None = None,
        session: aiohttp.ClientSession | None = None,
        user_id: str = "",
        *,
        endpoint_override: Literal["responses", "chat_completions"] | None = None,
        request_context: Request | None = None,
        user_obj: Any | None = None,
        pipe_identifier: str | None = None,
        fusion_live_enabled: bool = False,
        event_source: AsyncGenerator[dict[str, Any], None] | None = None,
        outcome_sink: dict[str, Any] | None = None,
        retry_handoff: dict[str, Any] | None = None,
        emitter_supplied: bool | None = None,
    ):
        """
        Stream assistant responses incrementally, handling function calls, status updates, and tool usage.
        """
        metadata = {} if metadata is None else metadata
        if session is None:
            raise RuntimeError("HTTP session is required for streaming")

        if emitter_supplied is None:
            emitter_supplied = event_emitter is not None
        continuation_newline_pending = bool(body._continues_after_marker)
        refusal_pending: list[str] = []
        continues_after_text = not continuation_newline_pending and bool((CONTINUED_REPLY.get() or "").strip())
        if event_emitter is None:
            event_emitter = _wrap_event_emitter(None)

        if not isinstance(metadata, dict):
            metadata = {}

        _loop_pipe_meta = metadata.get(_PIPE_METADATA_KEY)
        fusion_inner_call = bool(isinstance(_loop_pipe_meta, dict) and _loop_pipe_meta.get("fusion_inner"))

        async def _emit_budget_notice(text: str) -> None:
            if fusion_inner_call and event_emitter is not None:
                await event_emitter({"type": "pipe:member.notice", "data": {"content": text}})
                return
            await self._pipe._event_emitter_handler._emit_notification(
                event_emitter, text, level="warning"
            )

        async def _report_omissions(outcome: Any, opening: str) -> None:
            if outcome is None or not outcome.omitted_call_ids:
                return
            fresh = {
                call_id
                for call_id in outcome.omitted_call_ids
                if call_id not in body.budget_reported_call_ids
            }
            if not fresh:
                return
            body.budget_reported_call_ids.update(fresh)
            names = omitted_tool_names(
                type(outcome)(frozenset(fresh), False, 0, 0), body.input
            )
            await _emit_budget_notice(f"{opening} {', '.join(names)}.")

        async def _warn_if_futile(outcome: Any) -> None:
            if outcome is None or not outcome.futile:
                return
            if body.budget_futility_notified:
                return
            body.budget_futility_notified = True
            await _emit_budget_notice(build_futility_notice(outcome))

        breaker_key_value = None if fusion_inner_call else (user_id or None)

        owui_tool_passthrough = open_webui_runs_the_calls(valves, metadata, stream=bool(body.stream))
        persist_tools_enabled = valves.PERSIST_TOOL_RESULTS
        handed_back = False
        is_continuation = owui_tool_passthrough and any(
            item.get("type") == "function_call_output" for item in current_turn_items(body.input)
        )
        open_webui_keeps_stored_output = bool(metadata.get("assistant_message_id"))
        earlier_turn_calls, earlier_turn_texts, continues_after_reasoning = (
            body._continued_turn if body._continued_turn is not None else continued_turn_counts(body.input)
        )
        self.logger.debug(
            "🔧 TOOL_EXECUTION_MODE decision=owui_passthrough=%s PERSIST_TOOL_RESULTS=%s effective_persist_tools=%s is_continuation=%s",
            owui_tool_passthrough,
            valves.PERSIST_TOOL_RESULTS,
            persist_tools_enabled,
            is_continuation,
        )
        self.logger.debug("Streaming config: direct pass-through (no server batching)")
        streamed_tool_call_ids: set[str] = set()
        streamed_tool_call_indices: dict[str, int] = {}
        tool_call_names: dict[str, str] = {}

        def _origin_tool_name(exposed_name: str) -> str:
            raw_map = metadata.get("_pipe_exposed_to_origin") if isinstance(metadata, dict) else None
            origin = raw_map.get(exposed_name) if isinstance(raw_map, dict) else None
            return origin if isinstance(origin, str) and origin else exposed_name

        def _is_owui_origin_tool(exposed_name: str) -> bool:
            cfg = tool_registry.get(exposed_name)
            if not isinstance(cfg, dict):
                return True
            origin_source = cfg.get("origin_source")
            return origin_source is None or origin_source in _OWUI_ORIGIN_SOURCES

        def _is_owui_builtin_tool(exposed_name: str) -> bool:
            cfg = tool_registry.get(exposed_name)
            if not isinstance(cfg, dict):
                return False
            return cfg.get("type") == "builtin" and str(cfg.get("tool_id") or "").startswith("builtin:")

        def _cites_as_owui_builtin(exposed_name: str, origin_name: str) -> bool:
            return origin_name in BUILTIN_CITATION_TOOLS and _is_owui_builtin_tool(exposed_name)

        tool_call_item_ids: dict[str, str] = {}
        streamed_tool_call_args: dict[str, _ReasoningTextBox] = {}
        host_call_indices: dict[str, int] = {}
        host_call_item_ids: dict[str, str] = {}
        host_call_names: dict[str, str] = {}
        host_call_finalised: set[str] = set()
        host_call_closed: set[str] = set()
        emitted_tool_call_items: set[str] = set()
        emitted_model_call_items: set[str] = set()
        calls_carded_this_round: set[str] = set()
        emitted_tool_output_items: set[str] = set()
        committed_call_rows: set[str] = set()
        committed_output_rows: set[str] = set()
        committed_shell_calls: set[str] = set()
        stubbed_call_ids: set[str] = set()
        executed_tool_call_ids: set[tuple[str, str, str]] = set()
        emitted_response_output_items = False
        emitted_output_items: list[dict[str, Any]] = []
        published_item_ids: list[str] = []
        open_message_id: str | None = None
        seeded_output_items: list[dict[str, Any]] | None = None
        recorded_message_chars = 0
        incomplete_warning_emitted = False
        tool_loops_executed = False
        assistant_len_before_tool_loops = 0
        has_actionable_continuation = False
        session_log_reason: str = ""
        answer_truncated: str = ""

        raw_tools = tools or {}
        tool_registry: dict[str, dict[str, Any]] = {}
        if isinstance(raw_tools, dict):
            tool_registry = raw_tools
        elif isinstance(raw_tools, list):
            skipped_no_callable = False
            for entry in raw_tools:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name")
                if not name and isinstance(entry.get("spec"), dict):
                    name = entry["spec"].get("name")
                callable_obj = entry.get("callable")
                if callable_obj is None:
                    skipped_no_callable = True
                    continue
                if name:
                    tool_registry[name] = entry
            if skipped_no_callable and not tool_registry:
                self.logger.warning("Received list-based tools without callables; tool execution will be disabled for this request.")
        model_block = metadata.get("model")
        openwebui_model = model_block.get("id", "") if isinstance(model_block, dict) else ""
        citations_enabled = _owui_citations_enabled(model_block)
        assistant_message = ""
        assistant_len_before_tool_loops = len(assistant_message)
        pending_ulids: list[str] = []
        pending_items: list[dict[str, Any]] = []
        reasoning_anchor_state: dict[str, Any] = {
            "seq": 0,
            "calls_seen": earlier_turn_calls,
            "stream_calls": earlier_turn_calls,
            "text_chunks": earlier_turn_texts,
            "chars_at_last_chunk": -1 if continues_after_reasoning else 0,
            "awaiting": [],
        }
        total_usage: dict[str, Any] = {}
        reasoning_stream_active = False
        active_reasoning_item_id: str | None = None
        reasoning_stream_buffers: dict[str, _ReasoningTextBox] = {}
        reasoning_summary_consumed: dict[str, str] = {}
        reasoning_stream_completed: set[str] = set()
        reasoning_display: dict[str, dict[str, Any]] = {}
        unpublished_reasoning_keys: dict[str, None] = {}
        open_reasoning_windows: dict[str, None] = {}
        model_call_cards_at_round_start = 0
        round_saw_function_call = 0
        calls_in_this_round = 0
        named_tool_call = False
        deferred_reasoning_keys: dict[str, int] = {}
        ordinal_by_url: dict[str, int] = {}
        emitted_citations: list[dict] = []
        citation_excerpt_max = 1000
        unhandled_citation_notified = False
        chat_id = metadata.get("chat_id")
        message_id = metadata.get("message_id")
        reply_key = _reply_key(user_id, chat_id, message_id)
        offered_function_names = {
            str(t.get("name"))
            for t in (body.tools or [])
            if isinstance(t, dict) and t.get("type") == "function"
        }
        unclaimed_token: Any = None
        api_hold_key = ""
        try:
            may_hand_back = owui_tool_passthrough or bool(offered_function_names - set(tool_registry))
            holds_the_reply = bool(
                may_hand_back and body.stream and message_id and is_temporary_chat(chat_id)
            )
            _release_armed = holds_the_reply
            if holds_the_reply:
                self._pipe._artifact_store._reply_memory.open(chat_id, message_id)
            unclaimed_token = _UNCLAIMED_LATCH.set(
                set() if (chat_id and not message_id and not fusion_inner_call) else None
            )
            if (
                not chat_id
                and not is_temporary_chat(chat_id)
                and not fusion_inner_call
                and valves.API_CALL_ARTIFACT_MEMORY
            ):
                api_hold_key = SessionLogger.request_id.get() or ""
            persist_chat_id = None if fusion_inner_call else chat_id
            persist_message_id = message_id if message_id else (api_hold_key or None)
            if api_hold_key:
                self._pipe._artifact_store._api_reply_memory.open(persist_chat_id, persist_message_id)
                holds_the_reply = True
            model_started = asyncio.Event()
            responding_status_sent = False
            provider_status_seen = False
            generation_started_at: float | None = None
            generation_last_event_at: float | None = None
            response_completed_at: float | None = None
            stream_started_at: float | None = None
            surrogate_carry: dict[str, str] = {"assistant": "", "reasoning": ""}
            storage_context_cache: tuple[Request | None, Any | None] | None = None
            processed_image_item_ids: set[str] = set()
            opened_image_windows: set[str] = set()
            generated_image_count = 0
            skipped_image_reasons: list[str] = []
            thinking_mode = valves.THINKING_OUTPUT_MODE
            thinking_box_enabled = thinking_mode in {"open_webui", "both"}
            thinking_status_enabled = thinking_mode in {"status", "both"}
            reasoning_throttle = ReasoningStatusThrottle()

            fusion_armed = bool(fusion_live_enabled)
            fusion_state = FusionDeliberationState() if fusion_armed else None
            fusion_batcher = FusionDeltaBatcher() if fusion_armed else None
            fusion_embed_emitted = False
            fusion_embed_task: asyncio.Task[None] | None = None
            fell_back_to_chat = False

            def _add_fusion_name(names: dict[str, str], mid: str) -> None:
                if mid in names:
                    return
                display = ModelFamily.display_name(mid)
                if not display and mid.startswith("~"):
                    display = ModelFamily.display_name(mid[1:])
                if display:
                    names[mid] = display

            def _fusion_model_names() -> dict[str, str]:
                names: dict[str, str] = {}
                if fusion_state is None:
                    return names
                for fusion_ev in fusion_state.events:
                    if not isinstance(fusion_ev, dict):
                        continue
                    for _name_key in ("model", "judge_model"):
                        _mid = fusion_ev.get(_name_key)
                        if isinstance(_mid, str):
                            _add_fusion_name(names, _mid)
                    _resp = fusion_ev.get("response")
                    if isinstance(_resp, dict) and isinstance(_resp.get("model"), str):
                        _add_fusion_name(names, _resp["model"])
                return names

            async def _emit_fusion_embed_once() -> None:
                nonlocal fusion_embed_emitted
                if fusion_embed_emitted or not (fusion_armed and fusion_state is not None and event_emitter):
                    return
                _prior = 0
                if retry_handoff is not None:
                    _prior = retry_handoff.get(FUSION_EMBED_ATTEMPTS, 0)
                    retry_handoff[FUSION_EMBED_ATTEMPTS] = _prior + 1
                _card = build_fusion_embed_html(fusion_state, _fusion_model_names())
                fusion_embed_emitted = True
                _set = [_card]
                if _prior > 0 and chat_id and message_id and Chats is not None:
                    try:
                        _existing = await Chats.get_message_by_id_and_message_id(
                            str(chat_id), str(message_id)
                        )
                        _raw = (_existing or {}).get("embeds")
                        if isinstance(_raw, list):
                            _set = [
                                e for e in _raw
                                if not (isinstance(e, str) and "<title>OpenRouter Fusion" in e)
                            ] + [_card]
                    except Exception:
                        self.logger.debug(
                            "Failed to read message embeds for fusion retry merge", exc_info=True
                        )
                        _set = [_card]
                try:
                    await self._pipe._event_emitter_handler._emit_embeds(
                        event_emitter, _set,
                        replace=_prior > 0,
                    )
                except Exception:
                    self.logger.debug("Failed to emit fusion embed", exc_info=True)

            async def _emit_fusion_embed_after_roster() -> None:
                try:
                    await asyncio.sleep(0.05)
                except asyncio.CancelledError:
                    return
                await _emit_fusion_embed_once()

            async def _emit_fusion_event(fusion_ev: dict) -> None:
                if not (fusion_armed and event_emitter):
                    return
                try:
                    await event_emitter({"type": "fusion:event", "data": {"event": fusion_ev}})
                except Exception:
                    self.logger.debug("Failed to emit fusion event", exc_info=True)

            async def _emit_fusion_sources(raw_sources: Any) -> None:
                """Emit openrouter:fusion item-level sources as citations (per-URL dedup)."""
                if not isinstance(raw_sources, list):
                    return
                for src in raw_sources:
                    if not isinstance(src, dict):
                        continue
                    url = (src.get("url") or "").strip()
                    if not url or url in ordinal_by_url:
                        continue
                    ordinal_by_url[url] = len(ordinal_by_url) + 1
                    title = (src.get("title") or "").strip()
                    host = _citation_host(url)
                    citation = {
                        "source": {"name": host or "source", "url": url},
                        "document": [title or url],
                        "metadata": [{
                            "source": url,
                            "date_accessed": citation_access_stamp(),
                        }],
                    }
                    try:
                        await self._pipe._event_emitter_handler._emit_citation(event_emitter, citation)
                    except Exception as exc:
                        self.logger.debug("Failed to emit fusion source citation: %s", exc, exc_info=True)
                    emitted_citations.append(citation)

            async def _maybe_emit_reasoning_status(delta_text: str, *, force: bool = False) -> None:
                """Emit readable status updates for reasoning text without flooding."""
                if not thinking_status_enabled or not event_emitter:
                    return
                text = reasoning_throttle.feed(delta_text, force=force)
                if text:
                    await event_emitter({"type": "status", "data": {"description": text, "done": False}})

            async def _get_storage_context() -> tuple[Request | None, Any | None]:
                nonlocal storage_context_cache
                if storage_context_cache is None:
                    storage_context_cache = await self._pipe._file_gateway.resolve_storage_context(request_context, user_obj)
                return storage_context_cache or (None, None)

            @timed
            async def _persist_generated_image(data: bytes, mime_type: str) -> str | None:
                if not is_linkable_chat(chat_id):
                    return None
                upload_request, upload_user = await _get_storage_context()
                if not upload_request or not upload_user:
                    return None
                filename = f"generated-image-{uuid.uuid4().hex}.{image_extension_for_mime(mime_type)}"
                return await self._pipe._file_gateway.upload_to_owui_storage(
                    request=upload_request,
                    user=upload_user,
                    file_data=data,
                    filename=filename,
                    mime_type=mime_type,
                    chat_id=persist_chat_id if isinstance(persist_chat_id, str) else None,
                    message_id=message_id if isinstance(message_id, str) else None,
                    owui_user_id=user_id,
                )

            def _resolved_stored_mime(declared: str | None, decoded: bytes) -> str | None:
                head = decoded[:_SNIFF_PREFIX_BYTES]
                evidence = _sniff_evidence(head)
                if evidence is None and head_is_not_a_picture(head):
                    self.logger.debug(
                        "Not materialising an image entry: its head is one repeated byte, "
                        "which is not a container whatever it declares (%r).",
                        declared,
                    )
                    return None
                resolved = resolve_download_type(declared, evidence)
                mime_type = canonical_image_mime(resolved)
                if mime_type is None:
                    self.logger.debug(
                        "Not materialising an image entry: declared=%r resolves to %r, "
                        "which is not a storable image type",
                        declared, resolved,
                    )
                return mime_type

            @timed
            async def _materialize_image_from_str(data_str: str) -> str | None:
                text = (data_str or "").strip()
                if not text:
                    return None
                if text.startswith("data:"):
                    parsed = await asyncio.to_thread(self._pipe._multimodal_handler._parse_data_url, text)
                    if parsed:
                        mime_type = _resolved_stored_mime(parsed["mime_type"], parsed["data"])
                        if mime_type is None:
                            return None
                        stored = await _persist_generated_image(parsed["data"], mime_type)
                        if stored:
                            await self._pipe._event_emitter_handler._emit_status(event_emitter, StatusMessages.IMAGE_BASE64_SAVED, done=False)
                            return f"/api/v1/files/{stored}/content"
                        return text
                    return None
                if is_http_or_https_url(text):
                    if not is_linkable_chat(chat_id):
                        if await self._pipe._multimodal_handler._is_safe_url(text) is not True:
                            skipped_image_reasons.append("unlinkable_chat_unvetted")
                            return None
                        return text
                    downloaded = await self._pipe._multimodal_handler._download_remote_url(text)
                    if downloaded:
                        mime_type = _resolved_stored_mime(downloaded["mime_type"], downloaded["data"])
                        if mime_type is None:
                            return None
                        stored = await _persist_generated_image(downloaded["data"], mime_type)
                        if stored:
                            await self._pipe._event_emitter_handler._emit_status(event_emitter, StatusMessages.IMAGE_REMOTE_SAVED, done=False)
                            return f"/api/v1/files/{stored}/content"
                    if await self._pipe._multimodal_handler._is_safe_url(text) is not True:
                        skipped_image_reasons.append("unfetchable")
                        return None
                    return text
                if text.startswith("/"):
                    return text
                cleaned = text
                if "," in cleaned and ";base64" in cleaned.split(",", 1)[0]:
                    cleaned = cleaned.split(",", 1)[1]
                cleaned = cleaned.strip()
                if not cleaned:
                    return None
                if not self._pipe._file_gateway.validate_base64_size(cleaned):
                    return None
                try:
                    decoded = await _decode_base64_in_quanta(cleaned)
                except (binascii.Error, ValueError):
                    return None
                mime_type = _resolved_stored_mime(None, decoded)
                if mime_type is None:
                    return None
                stored = await _persist_generated_image(decoded, mime_type)
                if stored:
                    await self._pipe._event_emitter_handler._emit_status(event_emitter, StatusMessages.IMAGE_BASE64_SAVED, done=False)
                    return f"/api/v1/files/{stored}/content"
                return f"data:{mime_type};base64,{cleaned}"

            @timed
            async def _materialize_image_entry(entry: Any) -> str | None:
                if entry is None:
                    return None
                if isinstance(entry, str):
                    return await _materialize_image_from_str(entry)
                if isinstance(entry, dict):
                    for key in ("url", "image_url", "imageUrl", "content_url"):
                        candidate = entry.get(key)
                        if isinstance(candidate, str) and candidate.strip():
                            return await _materialize_image_from_str(candidate.strip())
                        if isinstance(candidate, dict):
                            nested = await _materialize_image_entry(candidate)
                            if nested:
                                return nested
                    for key in ("b64_json", "b64", "base64", "data", "image_base64", "imageB64"):
                        b64_val = entry.get(key)
                        if isinstance(b64_val, str) and b64_val.strip():
                            cleaned = b64_val.strip()
                            if not self._pipe._file_gateway.validate_base64_size(cleaned):
                                continue
                            try:
                                decoded = await _decode_base64_in_quanta(cleaned)
                            except (binascii.Error, ValueError):
                                continue
                            explicit_mime = (
                                entry.get("mime_type")
                                or entry.get("mimeType")
                                or entry.get("content_type")
                            )
                            mime_type = _resolved_stored_mime(explicit_mime, decoded)
                            if mime_type is None:
                                continue
                            stored = await _persist_generated_image(decoded, mime_type)
                            if stored:
                                await self._pipe._event_emitter_handler._emit_status(event_emitter, StatusMessages.IMAGE_BASE64_SAVED, done=False)
                                return f"/api/v1/files/{stored}/content"
                            return f"data:{mime_type};base64,{cleaned}"
                    nested_result = entry.get("result")
                    if nested_result is not None:
                        return await _materialize_image_entry(nested_result)
                return None

            async def _collect_image_output_urls(item: dict[str, Any]) -> list[str]:
                urls: list[str] = []
                seen_blobs: set[int] = set()

                async def _resolve_and_dedup(value: Any) -> None:
                    if value is None:
                        return
                    if isinstance(value, str) and len(value) > 256:
                        blob_hash = hash(value)
                        if blob_hash in seen_blobs:
                            return
                        seen_blobs.add(blob_hash)
                    resolved = await _materialize_image_entry(value)
                    if resolved:
                        urls.append(resolved)

                payload = item.get("result")
                if isinstance(payload, list):
                    for entry in payload:
                        await _resolve_and_dedup(entry)
                else:
                    await _resolve_and_dedup(payload)

                for extra_key in ("imageUrl", "imageB64"):
                    extra = item.get(extra_key)
                    if extra is not None:
                        await _resolve_and_dedup(extra)

                return urls

            async def _render_image_markdown(item: dict[str, Any]) -> list[str]:
                nonlocal generated_image_count
                urls = await _collect_image_output_urls(item)
                markdowns: list[str] = []
                for url in urls:
                    generated_image_count += 1
                    label = item.get("label") or f"Generated image {generated_image_count}"
                    alt_text = re.sub(r"[\r\n]+", " ", str(label)).strip() or f"Generated image {generated_image_count}"
                    markdowns.append(f"![{alt_text}]({url})")
                if skipped_image_reasons:
                    await self._pipe._event_emitter_handler._emit_status(
                        event_emitter,
                        StatusMessages.IMAGES_SKIPPED_UNFETCHABLE.format(
                            count=len(skipped_image_reasons)
                        ),
                        done=False,
                    )
                    skipped_image_reasons.clear()
                return markdowns

            def _append_output_block(current: str, block: str) -> str:
                snippet = (block or "").strip()
                if not snippet:
                    return current
                if current:
                    if not current.endswith("\n"):
                        current += "\n"
                    if not current.endswith("\n\n"):
                        current += "\n"
                return f"{current}{snippet}\n"

            async def _capture_seeded_output() -> list[dict[str, Any]]:
                nonlocal seeded_output_items
                if seeded_output_items is not None:
                    return seeded_output_items
                seeded_output_items = []
                if open_webui_keeps_stored_output:
                    return seeded_output_items
                if chat_id and message_id and Chats is not None:
                    try:
                        stored = await Chats.get_message_by_id_and_message_id(
                            str(chat_id), str(message_id)
                        )
                    except Exception:
                        self.logger.debug(
                            "Could not read seeded output items (chat_id=%s message_id=%s)",
                            chat_id,
                            message_id,
                            exc_info=True,
                        )
                        return seeded_output_items
                    prior = (stored or {}).get("output")
                    if isinstance(prior, list):
                        stored_items = [
                            copy.deepcopy(entry) for entry in prior if isinstance(entry, dict)
                        ]
                        seeded_output_items = stored_items
                return seeded_output_items

            def _unjoined_segment(text: str, earlier_items: list[dict[str, Any]]) -> str:
                if not text.startswith("\n"):
                    return text
                for entry in earlier_items:
                    if entry.get("type") != "message":
                        continue
                    joined = "".join(
                        str(part.get("text") or "") for part in entry.get("content") or []
                    )
                    if joined.strip():
                        return text[1:]
                return text

            def _flush_recorded_message(current_text: str) -> dict[str, Any] | None:
                nonlocal recorded_message_chars, open_message_id
                pending = current_text[recorded_message_chars:]
                segment_id = open_message_id or f"msg-{uuid.uuid4().hex}"
                open_message_id = None
                if not pending:
                    return None
                pending = _unjoined_segment(pending, emitted_output_items)
                recorded_message_chars = len(current_text)
                item = {
                    "type": "message",
                    "id": segment_id,
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": pending}],
                }
                emitted_output_items.append(item)
                return item

            async def _record_output_item(item: dict[str, Any], current_text: str) -> None:
                await _capture_seeded_output()
                recorded = copy.deepcopy(item)
                item_id = recorded.get("id")
                if item_id:
                    for index, existing in enumerate(emitted_output_items):
                        if existing.get("id") == item_id:
                            emitted_output_items[index] = recorded
                            return
                _flush_recorded_message(current_text)
                emitted_output_items.append(recorded)

            def _terminal_output_items(current_text: str) -> list[dict[str, Any]]:
                seeded = seeded_output_items or []
                combined = seeded + emitted_output_items
                result_status_by_call_id: dict[str, Any] = {}
                for entry in combined:
                    if entry.get("type") != "function_call_output":
                        continue
                    result_id = entry.get("call_id")
                    if isinstance(result_id, str) and result_id:
                        result_status_by_call_id[result_id] = entry.get("status")
                resolved: list[dict[str, Any]] = []
                for index, entry in enumerate(combined):
                    item = copy.deepcopy(entry)
                    if item.get("type") == "function_call":
                        call_id = item.get("call_id") or item.get("id")
                        resolvable = item.get("status") not in OWUI_UNRESOLVABLE_CALL_STATUSES
                        addressable = index >= len(seeded) or call_id in result_status_by_call_id
                        if resolvable and addressable and call_id in result_status_by_call_id:
                            item["status"] = owui_call_status(result_status_by_call_id.get(call_id))
                    resolved.append(item)
                trailing = current_text[recorded_message_chars:]
                trailing = _unjoined_segment(trailing, resolved)
                if trailing:
                    resolved.append({
                        "type": "message",
                        "id": open_message_id or f"msg-{uuid.uuid4().hex}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": trailing}],
                    })
                return resolved

            def _output_index(item: dict[str, Any]) -> int:
                item_id = item.get("id")
                if isinstance(item_id, str) and item_id in published_item_ids:
                    return published_item_ids.index(item_id)
                published_item_ids.append(item_id if isinstance(item_id, str) and item_id else f"#{len(published_item_ids)}")
                return len(published_item_ids) - 1

            async def _publish_host_call_item(call_id: str, item_id: str, tool_name: str) -> int:
                if call_id in host_call_indices:
                    return host_call_indices[call_id]
                published_id = item_id or f"fc_{uuid.uuid4().hex}"
                reserved = published_id not in published_item_ids
                index = _output_index({"id": published_id})
                try:
                    await _publish_turn_frame(
                        {
                            "type": "response.output_item.added",
                            "output_index": index,
                            "item": {
                                "type": "function_call",
                                "id": published_id,
                                "call_id": call_id,
                                "name": tool_name,
                                "arguments": "",
                                "status": "in_progress",
                            },
                        }
                    )
                except Exception:
                    if reserved:
                        published_item_ids.remove(published_id)
                    raise
                host_call_item_ids[call_id] = published_id
                host_call_names[call_id] = tool_name
                host_call_indices[call_id] = index
                return index

            async def _close_host_call_item(call_id: str, arguments: str) -> None:
                if call_id in host_call_closed:
                    return
                host_call_closed.add(call_id)
                index = host_call_indices.get(call_id)
                if index is None:
                    return
                published_id = host_call_item_ids.get(call_id) or f"fc_{uuid.uuid4().hex}"
                if call_id not in host_call_finalised:
                    host_call_finalised.add(call_id)
                    await _publish_turn_frame(
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": published_id,
                            "output_index": index,
                            "arguments": arguments,
                        }
                    )
                await _publish_turn_frame(
                    {
                        "type": "response.output_item.done",
                        "output_index": index,
                        "item": {
                            "type": "function_call",
                            "id": published_id,
                            "call_id": call_id,
                            "name": host_call_names.get(call_id, ""),
                            "arguments": arguments,
                            "status": "completed",
                        },
                    }
                )

            def _output_index_before_open_message(item: dict[str, Any]) -> int:
                if open_message_id not in published_item_ids:
                    return _output_index(item)
                position = published_item_ids.index(open_message_id)
                published_item_ids.insert(position, str(item["id"]))
                return position

            async def _place_item(item: dict[str, Any], current_text: str) -> int:
                nonlocal retry_barrier_crossed
                if emitter_supplied:
                    retry_barrier_crossed = True
                pending = current_text[recorded_message_chars:]
                pending_shows_text = bool(strip_hidden_marker_lines(pending).strip())
                if item.get("type") == "function_call" and not pending_shows_text:
                    recorded = [*(await _capture_seeded_output() or []), *emitted_output_items]
                    if recorded and recorded[-1].get("type") == "function_call_output":
                        divider: dict[str, Any] = {
                            "type": "message",
                            "id": f"msg-{uuid.uuid4().hex}",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": ""}],
                        }
                        await event_emitter({
                            "type": "response.output_item.added",
                            "output_index": await _place_item(divider, current_text),
                            "item": divider,
                        })
                if pending and not pending_shows_text:
                    await _capture_seeded_output()
                    emitted_output_items.append(copy.deepcopy(item))
                    return _output_index_before_open_message(item)
                if open_message_id is None and pending_shows_text:
                    await _publish_pending_message(current_text)
                await _record_output_item(item, current_text)
                return _output_index(item)

            async def _publish_pending_message(current_text: str) -> None:
                if open_message_id is not None:
                    return
                pending_now = current_text[recorded_message_chars:]
                if not strip_hidden_marker_lines(pending_now).strip():
                    return
                published_segment = _flush_recorded_message(current_text)
                if published_segment is not None:
                    await event_emitter({
                        "type": "response.output_item.added",
                        "output_index": _output_index(published_segment),
                        "item": published_segment,
                    })

            async def _open_message() -> None:
                nonlocal open_message_id
                if open_message_id is not None or not (body.stream and emitter_supplied):
                    return
                open_message_id = f"msg-{uuid.uuid4().hex}"
                opened: dict[str, Any] = {
                    "type": "message",
                    "id": open_message_id,
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [{"type": "output_text", "text": ""}],
                }
                await event_emitter({"type": "response.output_item.added", "output_index": _output_index(opened), "item": opened})

            async def _publish_turn_frame(event: dict[str, Any]) -> None:
                nonlocal retry_barrier_crossed
                published = await event_emitter(event)
                if body.stream and published is not False:
                    retry_barrier_crossed = True

            def _shows_a_file_inline(name: str) -> bool:
                context = self._pipe._TOOL_CONTEXT.get()
                return name == "display_file" and bool(
                    context and context.terminal_files_inline and not context.fusion_inner
                )

            async def _emit_tool_start(
                *,
                call_id: str,
                name: str,
                arguments: str,
                status: str = "in_progress",
                current_text: str,
                model_call: bool = True,
            ) -> str:
                """Emit a function_call tool card. Returns the effective call_id (UUID-generated if input was empty).

                CALLERS MUST capture the return value and pass it to _emit_tool_result for correct pairing.
                If the caller threads the wrong id, _emit_tool_result will short-circuit and the card will
                stay in spinner state indefinitely.
                """
                nonlocal emitted_response_output_items
                if not event_emitter or not (valves.SHOW_TOOL_CARDS or _shows_a_file_inline(_origin_tool_name(name))):
                    return call_id
                effective_id = call_id or f"st-{uuid.uuid4().hex}"
                if effective_id in emitted_tool_call_items:
                    return effective_id
                emitted_tool_call_items.add(effective_id)
                if model_call:
                    emitted_model_call_items.add(effective_id)
                calls_carded_this_round.add(effective_id)
                emitted_response_output_items = True
                call_item: dict[str, Any] = {
                    "type": "function_call",
                    "id": effective_id,
                    "call_id": effective_id,
                    "name": _origin_tool_name(name) if not valves.SHOW_TOOL_CARDS else name,
                    "arguments": arguments,
                    "status": status,
                }
                call_index = await _place_item(call_item, current_text)
                await event_emitter({
                    "type": "response.output_item.added",
                    "output_index": call_index,
                    "item": call_item,
                })
                await _flush_deferred_reasoning_items(len(emitted_model_call_items), current_text)
                return effective_id

            async def _emit_tool_result(
                *,
                call_id: str,
                result_text: str,
                files: list | None = None,
                embeds: list | None = None,
                status: str,
                pictures: list[str] | None = None,
                current_text: str,
            ) -> None:
                """Emit a tool result card. Used by both pipeline and server tools.

                The status is a parameter, not a constant: this is the display card for
                EVERY output, including the failure stubs built when a tool raises or the
                call loop is cut short. Open WebUI appends the item verbatim into the
                persisted message, so hardcoding "completed" stores a failure as a success.
                """
                nonlocal emitted_response_output_items
                if not event_emitter or not call_id or call_id in emitted_tool_output_items:
                    return
                if not valves.SHOW_TOOL_CARDS and call_id not in emitted_tool_call_items:
                    return
                result_text = recorded_tool_text(result_text, status)
                emitted_tool_output_items.add(call_id)
                emitted_response_output_items = True
                output_item: dict[str, Any] = {
                    "type": "function_call_output",
                    "id": f"fco-{uuid.uuid4().hex}",
                    "call_id": call_id,
                    "output": picture_output(result_text, pictures or []),
                    "status": status,
                }
                if files:
                    output_item["files"] = files
                if embeds:
                    output_item["embeds"] = embeds
                output_index = await _place_item(output_item, current_text)
                await event_emitter({"type": "response.output_item.added", "output_index": output_index, "item": output_item})
                call_item = next(
                    (entry for entry in emitted_output_items
                     if entry.get("type") == "function_call" and entry.get("call_id") == call_id),
                    None,
                )
                if call_item is not None and call_item.get("status") not in OWUI_UNRESOLVABLE_CALL_STATUSES:
                    settled = owui_call_status(status)
                    if call_item.get("status") != settled:
                        call_item["status"] = settled
                        await event_emitter({
                            "type": "response.output_item.added",
                            "output_index": _output_index(call_item),
                            "item": copy.deepcopy(call_item),
                        })

            def _handed_to_open_webui(call_id: str) -> bool:
                return bool(emitter_supplied and call_id in calls_carded_this_round)

            def _tool_rows(payloads: list[dict[str, Any]], call_id: str) -> list[dict[str, Any]]:
                rows: list[dict[str, Any]] = []
                for payload in payloads:
                    if not _handed_to_open_webui(call_id):
                        payload[PIPE_ONLY_TOOL_ROUND_KEY] = True
                    normalized = normalize_persisted_item(payload)
                    row = (
                        self._pipe._artifact_store._make_db_row(persist_chat_id, persist_message_id, openwebui_model, normalized)
                        if normalized
                        else None
                    )
                    if row:
                        rows.append(row)
                return rows

            def _round_call_row(call: dict[str, Any], cid: str) -> list[dict[str, Any]]:
                from ..tools.tool_executor import is_builtin_ask_user

                payload = dict(call)
                exposed = str(payload.get("name") or "").strip()
                if is_builtin_ask_user(tool_registry.get(exposed)):
                    payload[BUILTIN_ASK_USER_ROUND_KEY] = True
                return _tool_rows([payload], cid)

            def _handed_back_round_rows(
                call: dict[str, Any], cid: str, exposed_name: str, tool_name: str, args_text: str
            ) -> list[dict[str, Any]]:
                if exposed_name != ASK_USER_ROUND_NAME and tool_name != ASK_USER_ROUND_NAME:
                    return []
                builtin_names = metadata.get(_BUILTIN_ASK_USER_NAMES_KEY) if isinstance(metadata, dict) else None
                payload: dict[str, Any] = {
                    "type": "function_call",
                    "call_id": cid,
                    "name": tool_name,
                    "arguments": args_text,
                    "status": str(call.get("status") or "completed"),
                    BUILTIN_ASK_USER_ROUND_KEY: bool(
                        isinstance(builtin_names, (set, frozenset)) and exposed_name in builtin_names
                    ),
                }
                return _tool_rows([payload], cid)

            def _round_output_row(output: dict[str, Any], cid: str) -> list[dict[str, Any]]:
                recorded = output.get("output")
                if is_picture_output(recorded):
                    text, pictures = tool_output_text_and_pictures(recorded)
                    recorded = picture_output(recorded_tool_text(text, output.get("status")), pictures)
                elif isinstance(recorded, str):
                    recorded = recorded_tool_text(recorded, output.get("status"))
                return _tool_rows([{**output, "output": recorded}], cid)

            async def _commit_server_tool_round(
                call_id: str, name: str, status: str, *, item_type: str, result_text: str, arguments: str = "{}",
                raw_item: dict[str, Any] | None = None, current_text: str,
            ) -> str:
                if not message_id and not api_hold_key:
                    return current_text
                if persist_tools_enabled and item_type in _RAW_REPLAYED_SERVER_TOOLS:
                    normalized = normalize_persisted_item(raw_item) if raw_item else None
                    row = (
                        self._pipe._artifact_store._make_db_row(persist_chat_id, persist_message_id, openwebui_model, normalized)
                        if normalized
                        else None
                    )
                    rows = [row] if row else []
                else:
                    rows = _tool_rows(
                        [
                            {"type": "function_call", "call_id": call_id, "name": name, "arguments": arguments},
                            {"type": "function_call_output", "call_id": call_id, "status": status,
                             "output": recorded_tool_text(result_text, status)},
                        ],
                        call_id,
                    )
                ulids = await _persist_rows(rows, "server_tool") if rows else []
                if ulids and not api_hold_key:
                    current_text = await _append_assistant_hidden_markers(
                        current_text, [_serialize_marker(ulid) for ulid in ulids]
                    )
                return current_text

            def _normalize_surrogate_chunk(text: str, bucket: str) -> str:
                """Coalesce surrogate pairs in streaming chunks to keep UTF-8 happy."""
                prev = surrogate_carry.get(bucket, "")
                combined = f"{prev}{text or ''}"
                if not combined:
                    surrogate_carry[bucket] = ""
                    return ""
                new_carry = ""
                try:
                    normalized = combined.encode("utf-16", "surrogatepass").decode("utf-16")
                except UnicodeDecodeError:
                    if combined:
                        last_char = combined[-1]
                        if 0xD800 <= ord(last_char) <= 0xDBFF:
                            new_carry = last_char
                            combined = combined[:-1]
                    normalized = combined.encode("utf-16", "surrogatepass").decode("utf-16", "ignore")
                surrogate_carry[bucket] = new_carry
                return normalized

            def _extract_reasoning_text(event: dict[str, Any]) -> str:
                """Return best-effort reasoning text from assorted event payloads."""
                if not isinstance(event, dict):
                    return ""
                for key in ("delta", "text"):
                    value = event.get(key)
                    if isinstance(value, str) and value:
                        return value
                part = event.get("part")
                if isinstance(part, dict):
                    part_text = part.get("text")
                    if isinstance(part_text, str) and part_text:
                        return part_text
                    content = part.get("content")
                    if isinstance(content, list):
                        fragments: list[str] = []
                        for entry in content:
                            if isinstance(entry, dict):
                                text_val = entry.get("text")
                                if isinstance(text_val, str):
                                    fragments.append(text_val)
                            elif isinstance(entry, str):
                                fragments.append(entry)
                        if fragments:
                            return "".join(fragments)
                return ""

            def _reasoning_stream_key(event: dict[str, Any], etype: str | None) -> str:
                """Associate reasoning deltas/snapshots with a stable upstream item id when possible."""
                item_id = event.get("item_id")
                if isinstance(item_id, str) and item_id:
                    return item_id
                if etype in {"response.output_item.added", "response.output_item.done"}:
                    item_raw = event.get("item")
                    item = item_raw if isinstance(item_raw, dict) else {}
                    iid = item.get("id")
                    if isinstance(iid, str) and iid:
                        return iid
                if active_reasoning_item_id:
                    return active_reasoning_item_id
                return "__reasoning__"

            def _extract_reasoning_text_from_item(item: dict[str, Any]) -> str:
                """Extract reasoning content/summary from a completed output item."""
                if not isinstance(item, dict):
                    return ""
                fragments: list[str] = []
                content = item.get("content")
                if isinstance(content, list):
                    for entry in content:
                        if isinstance(entry, dict):
                            text_val = entry.get("text")
                            if isinstance(text_val, str) and text_val:
                                fragments.append(text_val)
                if fragments:
                    return "".join(fragments)
                summary = item.get("summary")
                if isinstance(summary, list):
                    for entry in summary:
                        if isinstance(entry, dict):
                            text_val = entry.get("text")
                            if isinstance(text_val, str) and text_val:
                                fragments.append(text_val)
                return "".join(fragments)

            def _append_reasoning_text(
                key: str, incoming: str, *, allow_misaligned: bool, consumed: str = ""
            ) -> str:
                """Coalesce cumulative/snapshot reasoning payloads into a single stream without replay."""
                candidate = (incoming or "")
                if not candidate:
                    return ""
                if consumed:
                    if candidate == consumed or consumed.startswith(candidate):
                        return ""
                    if candidate.startswith(consumed):
                        candidate = candidate[len(consumed) :]
                        if not candidate:
                            return ""
                box = reasoning_stream_buffers.get(key)
                if box is None or box.length == 0:
                    append = candidate
                elif len(candidate) == box.length:
                    append = "" if candidate == box.text() else (candidate if allow_misaligned else "")
                elif len(candidate) > box.length:
                    current = box.text()
                    append = (
                        candidate[box.length :]
                        if candidate.startswith(current)
                        else (candidate if allow_misaligned else "")
                    )
                elif len(candidate) <= len(box.head):
                    if box.head.startswith(candidate):
                        append = ""
                    else:
                        append = "" if box.ends_with(candidate) else (candidate if allow_misaligned else "")
                else:
                    current = box.text()
                    append = (
                        ""
                        if current.startswith(candidate) or current.endswith(candidate)
                        else (candidate if allow_misaligned else "")
                    )
                if append:
                    if box is None:
                        box = _ReasoningTextBox()
                        reasoning_stream_buffers[key] = box
                    box.add(append)
                    unpublished_reasoning_keys[key] = None
                return append

            def _append_summary_tail(key: str, raw_text: str) -> str:
                consumed = reasoning_summary_consumed.get(key, "")
                if not consumed or not raw_text.startswith(consumed) or len(raw_text) <= len(consumed):
                    return ""
                tail = raw_text[len(consumed) :]
                title_match = re.findall(r"\*\*(.+?)\*\*", tail)
                title = title_match[-1].strip() if title_match else "Thinking…"
                content = re.sub(r"\*\*(.+?)\*\*", "", tail).strip()
                summary = title if not content else f"{title}\n{content}"
                normalized_summary = _normalize_surrogate_chunk(summary, "reasoning") if summary else ""
                if not normalized_summary:
                    return ""
                box = reasoning_stream_buffers.get(key)
                delivered = box.text() if box is not None else ""
                return _append_reasoning_text(
                    key, delivered + normalized_summary, allow_misaligned=False
                )

            def _reasoning_display_state(key: str) -> dict[str, Any]:
                state = reasoning_display.get(key)
                if state is None:
                    state = {
                        "wall_open": time.time(),
                        "mono_open": _monotonic(),
                        "mono_close": None,
                        "emitted": False,
                        "published_len": 0,
                        "published_round": None,
                        "published_id": None,
                    }
                    reasoning_display[key] = state
                    _reopen_reasoning_window(key, state)
                return state

            def _reopen_reasoning_window(key: str, state: dict[str, Any]) -> None:
                state["mono_close"] = None
                open_reasoning_windows[key] = None

            def _close_open_reasoning_windows() -> None:
                now = _monotonic()
                for key in list(open_reasoning_windows):
                    state = reasoning_display[key]
                    if state["mono_close"] is None:
                        state["mono_close"] = now
                open_reasoning_windows.clear()

            def _rearm_reasoning_window(key: str, state: dict[str, Any]) -> None:
                if state.get("published_round") is None or state["published_round"] == loop_index:
                    return
                state["wall_open"] = time.time()
                state["mono_open"] = _monotonic()
                _reopen_reasoning_window(key, state)

            async def _emit_reasoning_item(key: str, current_text: str, *, closing: bool = False) -> None:
                nonlocal emitted_response_output_items
                if event_emitter is None or not thinking_box_enabled:
                    return
                box = reasoning_stream_buffers.get(key)
                length = box.length if box is not None else 0
                state = _reasoning_display_state(key)
                published = int(state.get("published_len", 0) or 0)
                if length <= published:
                    return
                text = box.text() if box is not None else ""
                if not text.strip():
                    return
                if state.get("published_round") == loop_index:
                    await _replace_published_reasoning_item(key, text, state, current_text, closing=closing)
                    return
                mono_end = state["mono_close"] if state["mono_close"] is not None else _monotonic()
                duration = max(0.1, round(mono_end - state["mono_open"], 1))
                if published or key == "__reasoning__":
                    item_id = f"rs-{uuid.uuid4().hex}"
                else:
                    item_id = key
                state["published_len"] = len(text)
                state["published_round"] = loop_index
                state["emitted"] = True
                reasoning_stream_completed.add(key)
                unpublished_reasoning_keys.pop(key, None)
                emitted_response_output_items = True
                reasoning_item: dict[str, Any] = {
                    "type": "reasoning",
                    "id": item_id,
                    "summary": [{"type": "summary_text", "text": text[published:]}],
                    "status": "completed",
                    "started_at": state["wall_open"],
                    "ended_at": time.time(),
                    "duration": duration,
                }
                reasoning_index = await _place_item(reasoning_item, current_text)
                await event_emitter(
                    {
                        "type": "response.output_item.added",
                        "output_index": reasoning_index,
                        "item": reasoning_item,
                    }
                )
                state["published_id"] = item_id

            async def _replace_published_reasoning_item(
                key: str, text: str, state: dict[str, Any], current_text: str, *, closing: bool = False
            ) -> None:
                item_id = state.get("published_id")
                if not item_id:
                    return
                published = int(state.get("published_len", 0) or 0)
                if len(text) <= published:
                    return
                mono_end = state["mono_close"] if state["mono_close"] is not None else _monotonic()
                duration = max(0.1, round(mono_end - state["mono_open"], 1))
                reasoning_item: dict[str, Any] = {
                    "type": "reasoning",
                    "id": item_id,
                    "summary": [{"type": "summary_text", "text": text}],
                    "status": "completed",
                    "started_at": state["wall_open"],
                    "ended_at": time.time(),
                    "duration": duration,
                }
                await _record_output_item(reasoning_item, current_text)
                reasoning_index = _output_index(reasoning_item)
                tail = text[published:]
                if closing or not tail:
                    await event_emitter(
                        {
                            "type": "response.output_item.done",
                            "output_index": reasoning_index,
                            "item": reasoning_item,
                        }
                    )
                else:
                    await event_emitter(
                        {
                            "type": "response.reasoning_summary_text.delta",
                            "item_id": item_id,
                            "output_index": reasoning_index,
                            "summary_index": 0,
                            "delta": tail,
                        }
                    )
                state["published_len"] = len(text)
                unpublished_reasoning_keys.pop(key, None)

            async def _close_and_emit_reasoning_items(current_text: str) -> None:
                _close_open_reasoning_windows()
                for reasoning_key in list(unpublished_reasoning_keys):
                    await _emit_reasoning_item(reasoning_key, current_text, closing=True)

            async def _publish_open_reasoning_before_a_card(current_text: str) -> None:
                if not reasoning_display:
                    return
                _close_open_reasoning_windows()
                for reasoning_key in list(reasoning_display):
                    if reasoning_key in deferred_reasoning_keys:
                        continue
                    await _emit_reasoning_item(reasoning_key, current_text)

            def _defer_reasoning_keys_after_a_call(completed: Any) -> None:
                if not isinstance(completed, dict):
                    return
                calls_before = 0
                for entry in completed.get("output") or []:
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("type") == "function_call":
                        calls_before += 1
                    elif entry.get("type") == "reasoning" and calls_before:
                        entry_id = entry.get("id")
                        if isinstance(entry_id, str) and entry_id:
                            due = model_call_cards_at_round_start + calls_before
                            deferred_reasoning_keys[entry_id] = min(
                                due, deferred_reasoning_keys.get(entry_id, due)
                            )

            async def _flush_deferred_reasoning_items(upto_calls: int, current_text: str) -> None:
                if not deferred_reasoning_keys:
                    return
                for reasoning_key, calls_before in list(deferred_reasoning_keys.items()):
                    if calls_before > upto_calls:
                        continue
                    del deferred_reasoning_keys[reasoning_key]
                    await _emit_reasoning_item(reasoning_key, current_text)

            async def _flush_trailing_reasoning(current_text: str) -> None:
                if event_emitter is None or not thinking_box_enabled:
                    return
                for reasoning_key in list(unpublished_reasoning_keys):
                    try:
                        await _emit_reasoning_item(reasoning_key, current_text, closing=True)
                    except Exception:
                        self.logger.exception("Failed to emit trailing reasoning item")

            @timed
            async def _persist_rows(rows: list[dict[str, Any]], reason: str) -> list[str]:
                try:
                    return await self._pipe._artifact_store._db_persist(rows)
                except Exception:
                    self.logger.exception("Failed to persist response artifacts (%s)", reason)
                    if event_emitter:
                        await event_emitter(
                            {
                                "type": "status",
                                "data": {"description": "⚠️ Tool storage unavailable", "done": False},
                            }
                        )
                    return []

            async def _flush_pending(reason: str) -> None:
                if not pending_items:
                    return
                rows = pending_items[:]
                pending_items.clear()
                try:
                    pending_ulids.extend(await _persist_rows(rows, reason))
                except asyncio.CancelledError:
                    pending_items[:0] = [row for row in rows if row.get("payload") is not None]
                    raise

            async def _mark_committed_rows(current_text: str) -> str:
                if not pending_ulids:
                    return current_text
                ulids = pending_ulids[:]
                pending_ulids.clear()
                if api_hold_key:
                    return current_text
                return await _append_assistant_hidden_markers(
                    current_text, [_serialize_marker(ulid) for ulid in ulids]
                )

            thinking_tasks: list[asyncio.Task] = []
            thinking_cancelled = False
            if event_emitter and not is_continuation:
                async def _later(delay: float, msg: str) -> None:
                    """Emit a delayed status update to reassure the user during long thoughts."""
                    try:
                        await asyncio.wait_for(model_started.wait(), timeout=delay)
                        return
                    except TimeoutError:
                        if model_started.is_set():
                            return
                    await event_emitter({"type": "status", "data": {"description": msg}})

                thinking_tasks = []
                for delay, msg in [
                    (0, "Thinking…"),
                    (1.5, "Reading the user's question…"),
                    (4.0, "Gathering my thoughts…"),
                    (6.0, "Exploring possible responses…"),
                    (7.0, "Building a plan…"),
                ]:
                    if delay == 0:
                        await event_emitter({"type": "status", "data": {"description": msg}})
                        continue
                    thinking_tasks.append(
                        asyncio.create_task(_later(delay + random.uniform(0, 0.5), msg))
                    )

            def cancel_thinking() -> None:
                """Cancel any scheduled reasoning status updates once the loop completes."""
                nonlocal thinking_cancelled
                if thinking_cancelled:
                    return
                thinking_cancelled = True
                for t in thinking_tasks:
                    t.cancel()

            def note_model_activity() -> None:
                """Mark the stream as active and stop any pending thinking statuses."""
                if not model_started.is_set():
                    model_started.set()
                    cancel_thinking()

            def note_generation_activity() -> None:
                """Record when output tokens start/continue streaming."""
                nonlocal generation_started_at, generation_last_event_at
                now = perf_counter()
                generation_last_event_at = now
                if generation_started_at is None:
                    generation_started_at = now

            def _continuation_lead(current_text: str) -> str:
                nonlocal continuation_newline_pending
                lead = "\n" if continuation_newline_pending and not current_text else ""
                continuation_newline_pending = False
                return lead

            async def _append_assistant_hidden_markers(current_text: str, markers: list[str]) -> str:
                if not markers:
                    return current_text
                current_text += _continuation_lead(current_text)
                msg_before = len(current_text)
                if not current_text and continues_after_text:
                    current_text = "\n\n"
                current_text = _append_hidden_marker_lines(current_text, markers)
                if reasoning_anchor_state["chars_at_last_chunk"] == msg_before:
                    reasoning_anchor_state["chars_at_last_chunk"] = len(current_text)
                marker_delta = current_text[msg_before:]
                if body.stream:
                    await _open_message()
                    await _publish_turn_frame(
                        {"type": "chat:message:delta", "data": {"content": marker_delta}}
                    )
                elif content_handed_back:
                    self.logger.warning(
                        "Committed artifact row(s) left unaddressed: the content was handed back before its "
                        "markers were added (chat_id=%s markers=%s)",
                        chat_id,
                        markers,
                    )
                else:
                    self.logger.debug(
                        "Hidden markers added to the content this turn returns (chat_id=%s markers=%s)",
                        chat_id,
                        markers,
                    )
                return current_text

            def _extract_call_id(item: Any) -> str:
                """Best-effort call_id extraction for tool call/output items."""
                if not isinstance(item, dict):
                    return ""
                candidate = item.get("call_id") or item.get("id")
                if isinstance(candidate, str):
                    return candidate.strip()
                return ""

            def _extract_call_key(call: Any) -> tuple[str, str, str] | None:
                call_id = _extract_call_id(call)
                if not call_id or not isinstance(call, dict):
                    return None
                raw_args = call.get("arguments")
                if isinstance(raw_args, str):
                    args_text = raw_args.strip()
                else:
                    args_text = json.dumps(raw_args, ensure_ascii=False) if raw_args is not None else "{}"
                return (call_id, (call.get("name") or "").strip(), args_text)

            async def _notify_unhandled_citations(raw_annotations: Any) -> None:
                """Warn (status + toast) once if the response carries a citation type we can't
                render yet (e.g. file_citation). Never alters or halts the answer."""
                nonlocal unhandled_citation_notified
                if unhandled_citation_notified:
                    return
                unhandled = _unhandled_citation_types(raw_annotations)
                if not unhandled:
                    return
                unhandled_citation_notified = True
                type_label = ", ".join(sorted(unhandled))
                self.logger.debug("Unhandled citation annotation type(s): %s", type_label)
                await self._pipe._event_emitter_handler._emit_status(
                    event_emitter,
                    "Some source citations in this response couldn't be displayed.",
                    done=True,
                )
                await self._pipe._event_emitter_handler._emit_notification(
                    event_emitter,
                    f"This response included a citation type this pipe can't render yet ({type_label}). "
                    "Your answer is unaffected — please report this so support can be added.",
                    level="warning",
                )

            async def _emit_annotation_citations(raw_annotations: Any) -> None:
                """Emit url_citation citations, skipping any URL already emitted (per-URL dedup)."""
                await _notify_unhandled_citations(raw_annotations)
                if not isinstance(raw_annotations, list) or not raw_annotations:
                    return
                for url, title, content in _parse_url_citation_annotations(raw_annotations):
                    url = url.removesuffix("?utm_source=openai")
                    if url in ordinal_by_url:
                        continue
                    ordinal_by_url[url] = len(ordinal_by_url) + 1
                    host = _citation_host(url)
                    citation = {
                        "source": {"name": host or "source", "url": url},
                        "document": [content[:citation_excerpt_max] if content else title],
                        "metadata": [{
                            "source": url,
                            "date_accessed": citation_access_stamp(),
                        }],
                    }
                    try:
                        await self._pipe._event_emitter_handler._emit_citation(event_emitter, citation)
                    except Exception as exc:
                        self.logger.debug("Failed to emit annotation citation (final): %s", exc, exc_info=True)
                    emitted_citations.append(citation)

            request_started_at = perf_counter()

            error_occurred = False
            was_cancelled = False
            fusion_no_usable_member = False
            loop_limit_reached = False
            loop_limit_announced = False
            ran_out = False
            last_round_had_calls = False
            retry_barrier_crossed = False
            handed_back_for_retry = False
            content_handed_back = False

            def _round_annotations_and_reasoning(
                response: dict[str, Any] | None,
            ) -> tuple[list[Any], list[Any]]:
                out_annotations: list[Any] = []
                out_reasoning_details: list[Any] = []
                if not response or not isinstance(response.get("output"), list):
                    return out_annotations, out_reasoning_details
                for item in response.get("output") or []:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") != "message":
                        continue
                    if item.get("role") != "assistant":
                        continue
                    content = item.get("content")
                    if isinstance(content, list):
                        for part in content:
                            if not isinstance(part, dict) or part.get("type") != "output_text":
                                continue
                            raw_part_annotations = part.get("annotations")
                            if isinstance(raw_part_annotations, list) and raw_part_annotations:
                                out_annotations.extend(raw_part_annotations)
                    raw_annotations = item.get("annotations")
                    if isinstance(raw_annotations, list) and raw_annotations:
                        out_annotations.extend(raw_annotations)
                    raw_reasoning_details = item.get("reasoning_details")
                    if isinstance(raw_reasoning_details, list) and raw_reasoning_details:
                        out_reasoning_details.extend(raw_reasoning_details)
                return out_annotations, out_reasoning_details

            def _record_outcome() -> None:
                if outcome_sink is None:
                    return
                outcome_sink["error_occurred"] = error_occurred
                outcome_sink["was_cancelled"] = was_cancelled
                outcome_sink["reason"] = session_log_reason or None
                outcome_sink["answer_truncated"] = answer_truncated or None

            final_response: dict[str, Any] | None = None
            _finalise_cancelled: BaseException | None = None
            round_annotations: list[Any] = []
            round_reasoning_details: list[Any] = []
            dispatched_metered_chars: int | None = None
            dispatched_model_id: str = ""
            _release_armed = False
            event_iter: AsyncGenerator[dict[str, Any], None] | None = None
        except BaseException:
            if _release_armed:
                self._pipe._artifact_store._reply_memory.release(chat_id, message_id)
            if api_hold_key:
                self._pipe._artifact_store._api_reply_memory.release("", api_hold_key)
            raise

        try:
            max_loops = max(1, int(valves.MAX_FUNCTION_CALL_LOOPS))
            input_is_sanitized = False
            _answer_carry = [assistant_message]
            for loop_index in range(max_loops + 2):
                if loop_index > max_loops and not loop_limit_reached:
                    ran_out = True
                    break

                if loop_index > 0:
                    retry_barrier_crossed = True
                    await _close_and_emit_reasoning_items(assistant_message)
                    active_reasoning_item_id = None
                    reasoning_stream_buffers.pop("__reasoning__", None)
                    reasoning_summary_consumed.pop("__reasoning__", None)
                    reasoning_stream_completed.discard("__reasoning__")
                    reasoning_display.pop("__reasoning__", None)
                    unpublished_reasoning_keys.pop("__reasoning__", None)
                    open_reasoning_windows.pop("__reasoning__", None)
                    model_call_cards_at_round_start = len(emitted_model_call_items)
                    calls_in_this_round = 0
                    named_tool_call = False
                    final_response = None
                if event_source is not None:
                    if loop_index > 0:
                        break
                    model_for_cache = body.model
                    event_iter = event_source
                else:
                    if not input_is_sanitized:
                        _replay_budget = _sanitize_request_input(
                            self._pipe, body,
                            verdicts=await _tool_picture_verdicts_for_input(
                                self._pipe, body.input,
                            ),
                        )
                        await _warn_if_futile(_replay_budget)
                        await _report_omissions(_replay_budget, _REPLAY_DROPPED_OPENING)
                        input_is_sanitized = True
                    dispatched_model_id = budget_model_id(body)
                    model_for_cache = dispatched_model_id
                    items = getattr(body, "input", None)
                    if isinstance(items, list):
                        tools_list = getattr(body, "tools", None)
                        _maybe_apply_anthropic_prompt_caching(
                            items,
                            model_id=model_for_cache,
                            valves=valves,
                            tools=tools_list if isinstance(tools_list, list) else None,
                        )
                    request_payload = body.model_dump(exclude_none=True)
                    if dispatched_model_id:
                        request_payload["model"] = dispatched_model_id
                    request_payload.pop("api_model", None)
                    _apply_identifier_valves_to_payload(
                        request_payload,
                        valves=valves,
                        owui_metadata=metadata,
                        owui_user_id=user_id,
                        owui_user=user_obj,
                        logger=self.logger,
                    )
                    _apply_model_fallback_to_payload(request_payload, logger=self.logger)
                    _drop_include_reasoning_for_unsupported_fallbacks(
                        request_payload, self.logger
                    )
                    _apply_openrouter_trace_to_payload(request_payload, logger=self.logger)
                    _apply_disable_native_websearch_to_payload(request_payload, logger=self.logger)
                    _apply_provider_routing_params_to_payload(request_payload, logger=self.logger)
                    _strip_disable_model_settings_params(request_payload)
                    dispatched_metered_chars = None
                    dispatch_overhead = _request_overhead_chars(body)

                    api_key_value, api_key_error = (
                        self._pipe._resolve_openrouter_api_key(valves)
                    )
                    if api_key_error:
                        error_occurred = True
                        assistant_message = await self._pipe._ensure_error_formatter(
                        )._emit_templated_error(
                            event_emitter,
                            template=valves.AUTHENTICATION_ERROR_TEMPLATE,
                            variables={
                                "openrouter_code": 401,
                                "openrouter_message": api_key_error,
                            },
                            log_message=f"Auth configuration error: {api_key_error}",
                            log_level=logging.WARNING,
                            partial_answer=assistant_message,
                            terminal=False,
                        )
                        break
                    is_streaming = bool(request_payload.get("stream"))
                    if is_streaming:
                        event_iter = self._pipe.send_openrouter_streaming_request(
                            session,
                            request_payload,
                            event_emitter=event_emitter,
                            api_key=api_key_value,
                            base_url=valves.BASE_URL,
                            valves=valves,
                            endpoint_override=endpoint_override,
                            workers=valves.SSE_WORKERS_PER_REQUEST,
                            breaker_key=breaker_key_value,
                            delta_char_limit=valves.STREAMING_DELTA_CHAR_LIMIT,
                            idle_flush_ms=valves.STREAMING_IDLE_FLUSH_MS,
                            nagle_min_chars=valves.STREAMING_NAGLE_MIN_FLUSH_CHARS,
                            chunk_queue_maxsize=valves.STREAMING_CHUNK_QUEUE_MAXSIZE,
                            chunk_queue_warn_size=valves.STREAMING_CHUNK_QUEUE_WARN_SIZE,
                            event_queue_maxsize=valves.STREAMING_EVENT_QUEUE_MAXSIZE,
                            event_queue_warn_size=valves.STREAMING_EVENT_QUEUE_WARN_SIZE,
                            user=user_obj,
                            owui_chat_id=chat_id if isinstance(chat_id, str) else None,
                        )
                    else:
                        event_iter = self._pipe.send_openrouter_nonstreaming_request_as_events(
                            session,
                            request_payload,
                            api_key=api_key_value,
                            base_url=valves.BASE_URL,
                            valves=valves,
                            endpoint_override=endpoint_override,
                            breaker_key=breaker_key_value,
                            user=user_obj,
                            owui_chat_id=chat_id if isinstance(chat_id, str) else None,
                        )
                timing_mark("event_iteration_start")
                first_event_logged = False
                async for event in event_iter:
                    if stream_started_at is None:
                        stream_started_at = perf_counter()
                    if not first_event_logged:
                        first_event_logged = True
                        timing_mark("first_event_received")
                    etype = event.get("type")
                    if etype == "openrouter_pipe.chat_fallback":
                        fell_back_to_chat = True
                        continue

                    if etype == "openrouter_pipe.unhandled_citations":
                        raw_types = event.get("types")
                        if isinstance(raw_types, list) and raw_types:
                            await _notify_unhandled_citations(
                                [{"type": entry} for entry in raw_types]
                            )
                        continue

                    is_delta_event = bool(etype and etype.endswith(".delta"))
                    if not is_delta_event and SessionLogger.debug_enabled(self.logger):
                        redacted_event = _redact_payload_blobs(event)
                        self.logger.debug(
                            "OpenRouter payload: %s",
                            bounded_log_record_text(
                                json.dumps(redacted_event, indent=2, ensure_ascii=False)
                            ),
                        )

                    if etype:
                        if etype == "response.output_item.added":
                            item_raw = event.get("item")
                            item = item_raw if isinstance(item_raw, dict) else {}
                            if item.get("type") == "reasoning":
                                iid = item.get("id")
                                if not (isinstance(iid, str) and iid):
                                    iid = f"rs-{uuid.uuid4().hex}"
                                active_reasoning_item_id = iid
                                _reasoning_display_state(iid)

                        is_reasoning_event = (
                            etype.startswith("response.reasoning")
                            and etype != "response.reasoning_summary_text.done"
                        )
                        part = event.get("part") if isinstance(event, dict) else None
                        reasoning_part_types = {"reasoning_text", "reasoning_summary_text", "summary_text"}
                        is_reasoning_part_event = (
                            etype.startswith("response.content_part")
                            and isinstance(part, dict)
                            and part.get("type") in reasoning_part_types
                        )
                        if is_reasoning_event or is_reasoning_part_event:
                            reasoning_stream_active = True
                            note_model_activity()

                            key = _reasoning_stream_key(event, etype)
                            is_incremental = etype.endswith((".delta", ".added"))
                            is_final = etype.endswith((".done", ".completed"))

                            delta_text = _extract_reasoning_text(event)
                            normalized_delta = _normalize_surrogate_chunk(delta_text, "reasoning") if delta_text else ""

                            append = ""
                            if normalized_delta:
                                append = _append_reasoning_text(
                                    key,
                                    normalized_delta,
                                    allow_misaligned=is_incremental,
                                )
                            if append:
                                note_generation_activity()
                                display_state = _reasoning_display_state(key)
                                _rearm_reasoning_window(key, display_state)
                                _reopen_reasoning_window(key, display_state)
                                if fusion_inner_call and event_emitter is not None:
                                    await event_emitter({"type": "fusion_inner:reasoning.delta", "data": {"delta": append}})
                                await _maybe_emit_reasoning_status(append)
                            if is_final and event_emitter:
                                await _maybe_emit_reasoning_status("", force=True)
                            continue

                    if fusion_armed and fusion_state is not None and etype and (
                        etype.startswith("response.fusion_call")
                        or (
                            etype in ("response.output_item.added", "response.output_item.done")
                            and isinstance(event.get("item"), dict)
                            and event["item"].get("type") == "openrouter:fusion"
                        )
                    ):
                        if etype in ("response.fusion_call.panel.delta",
                                     "response.fusion_call.panel.reasoning.delta",
                                     "response.fusion_call.analysis.reasoning.delta",
                                     "response.fusion_call.synthesis.reasoning.delta"):
                            fusion_state.record(event)
                            if fusion_batcher is not None:
                                _batched = fusion_batcher.add(event)
                                if _batched is not None:
                                    await _emit_fusion_event(_batched)
                            continue
                        if etype == "response.fusion_call.panel.completed":
                            event = fusion_state.augment_panel_completed(event)
                            if fusion_batcher is not None and isinstance(event.get("model"), str):
                                for _tail in fusion_batcher.flush_model(event["model"]):
                                    await _emit_fusion_event(_tail)
                        elif etype == "response.fusion_call.analysis.in_progress":
                            if fusion_batcher is not None:
                                for _straggler in fusion_batcher.flush_all():
                                    await _emit_fusion_event(_straggler)
                        elif etype == "response.fusion_call.analysis.completed":
                            if fusion_batcher is not None:
                                for _straggler in fusion_batcher.flush_all():
                                    await _emit_fusion_event(_straggler)
                            event = fusion_state.augment_analysis_completed(event)
                        _fusion_ms = fusion_state.record(event)
                        if _fusion_ms == "fusion_open":
                            assistant_message = ""
                            if fusion_embed_task is None:
                                fusion_embed_task = asyncio.create_task(
                                    _emit_fusion_embed_after_roster()
                                )
                        if _fusion_ms:
                            await _emit_fusion_event(event)
                        if (
                            etype == "response.output_item.done"
                            and isinstance(event.get("item"), dict)
                            and event["item"].get("type") == "openrouter:fusion"
                        ):
                            _synth = fusion_state.synthesize_missing_analysis()
                            if _synth is not None:
                                await _emit_fusion_event(_synth)
                            await _emit_fusion_sources(event["item"].get("sources"))
                        continue

                    if etype == "pipe:member.notice" and event_source is not None:
                        await self._pipe._event_emitter_handler._emit_notification(
                            event_emitter, _member_notice_text(event), level="warning"
                        )
                        continue

                    if _is_no_usable_member_event(event_source, etype, event):
                        fusion_no_usable_member = True
                        event = {k: v for k, v in event.items() if k != "no_usable_member"}
                        if not fusion_armed and not assistant_message:
                            assistant_message = str(event.get("text") or "")
                            if event_emitter:
                                await event_emitter(
                                    {"type": "chat:message:delta",
                                     "data": {"content": assistant_message}}
                                )

                    if fusion_armed and fusion_state is not None and etype in ("response.created", "response.in_progress"):
                        fusion_state.record(event)

                    if fusion_armed and fusion_state is not None:
                        _ans_oi = event.get("output_index")
                        _post_fusion = (
                            fusion_state.fusion_index is not None
                            and isinstance(_ans_oi, int)
                            and _ans_oi > fusion_state.fusion_index
                        )
                        if fusion_state.fusion_index is None and etype in (
                            "response.output_text.delta",
                            "response.output_text.done",
                        ):
                            fusion_state.record(event)
                        elif _post_fusion and etype in (
                            "response.output_text.delta",
                            "response.output_text.done",
                        ):
                            if etype == "response.output_text.done":
                                if fusion_batcher is not None:
                                    for _straggler in fusion_batcher.flush_all():
                                        await _emit_fusion_event(_straggler)
                                event = fusion_state.augment_final_answer(event)
                            fusion_state.record(event)
                            await _emit_fusion_embed_once()
                            if etype == "response.output_text.done" and not assistant_message:
                                assistant_message = event.get("text") or ""
                            await _emit_fusion_event(event)
                            retry_barrier_crossed = True

                    if etype == "response.output_text.delta":
                        note_model_activity()
                        if reasoning_display:
                            _close_open_reasoning_windows()
                            for reasoning_key in list(unpublished_reasoning_keys):
                                if reasoning_key in deferred_reasoning_keys:
                                    continue
                                await _emit_reasoning_item(reasoning_key, assistant_message)
                        delta = event.get("delta") or ""
                        normalized_delta = _normalize_surrogate_chunk(delta, "assistant") if delta else ""
                        if normalized_delta:
                            note_generation_activity()
                            if (
                                not provider_status_seen
                                and not responding_status_sent
                                and not reasoning_stream_active
                                and event_emitter
                                and not fusion_armed
                            ):
                                provider_status_seen = True
                                responding_status_sent = True
                                await event_emitter(
                                    {
                                        "type": "status",
                                        "data": {"description": "Responding to the user…"},
                                    }
                                )
                            if normalized_delta:
                                normalized_delta = _continuation_lead(assistant_message) + normalized_delta
                            assistant_message += normalized_delta
                            if not fusion_armed:
                                await _open_message()
                                await _publish_turn_frame(
                                    {
                                        "type": "chat:message:delta",
                                        "data": {
                                            "content": normalized_delta,
                                        },
                                    }
                                )
                        continue

                    if etype in ("response.refusal.delta", "response.refusal.done"):
                        note_model_activity()
                        if reasoning_display:
                            _close_open_reasoning_windows()
                            for reasoning_key in list(reasoning_display):
                                await _emit_reasoning_item(reasoning_key, assistant_message)
                        if etype == "response.refusal.delta":
                            piece = event.get("delta") or ""
                            if isinstance(piece, str) and piece:
                                refusal_pending.append(piece)
                            continue
                        whole = event.get("refusal") or event.get("delta") or ""
                        if not isinstance(whole, str) or not whole:
                            whole = "".join(refusal_pending)
                        refusal_pending.clear()
                        normalized_refusal = (
                            _normalize_surrogate_chunk(whole, "assistant") if whole else ""
                        )
                        if normalized_refusal:
                            note_generation_activity()
                            if (
                                not provider_status_seen
                                and not responding_status_sent
                                and not reasoning_stream_active
                                and event_emitter
                                and not fusion_armed
                            ):
                                provider_status_seen = True
                                responding_status_sent = True
                                await event_emitter(
                                    {
                                        "type": "status",
                                        "data": {"description": "Responding to the user…"},
                                    }
                                )
                            if assistant_message:
                                normalized_refusal = (
                                    _continuation_lead(assistant_message) or "\n\n"
                                ) + normalized_refusal
                            assistant_message += normalized_refusal
                            if not fusion_armed:
                                await _open_message()
                                await _publish_turn_frame(
                                    {
                                        "type": "chat:message:delta",
                                        "data": {
                                            "content": normalized_refusal,
                                        },
                                    }
                                )
                        continue

                    if etype == "response.content_part.done":
                        part = event.get("part") if isinstance(event, dict) else None
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            await _emit_annotation_citations(part.get("annotations"))
                        continue

                    if owui_tool_passthrough and event_emitter and etype in {
                        "response.function_call_arguments.delta",
                        "response.function_call_arguments.done",
                    }:
                        try:
                            raw_item_id = event.get("item_id")
                            item_id = raw_item_id.strip() if isinstance(raw_item_id, str) else ""
                            raw_call_id = tool_call_item_ids.get(item_id) or event.get("call_id") or item_id or event.get("id")
                            call_id = raw_call_id.strip() if isinstance(raw_call_id, str) else ""
                            if not call_id:
                                continue
                            streamed_tool_call_indices.setdefault(
                                call_id, len(streamed_tool_call_indices)
                            )
                            raw_name = event.get("name") or tool_call_names.get(call_id)
                            tool_name = _origin_tool_name(raw_name.strip()) if isinstance(raw_name, str) else ""
                            if not tool_name:
                                continue
                            index = await _publish_host_call_item(call_id, item_id, tool_name)
                            published_id = host_call_item_ids.get(call_id) or item_id

                            if etype == "response.function_call_arguments.delta":
                                raw_delta = (
                                    event.get("delta")
                                    or event.get("arguments_delta")
                                    or event.get("arguments")
                                )
                                delta_text = raw_delta if isinstance(raw_delta, str) else ""
                                if not delta_text:
                                    continue
                                streamed_tool_call_ids.add(call_id)
                                await _publish_turn_frame(
                                    {
                                        "type": "response.function_call_arguments.delta",
                                        "item_id": published_id,
                                        "output_index": index,
                                        "delta": delta_text,
                                    }
                                )
                                continue

                            raw_args = event.get("arguments")
                            args_text = raw_args.strip() if isinstance(raw_args, str) else ""
                            if args_text:
                                streamed_tool_call_ids.add(call_id)
                                streamed_tool_call_args[call_id] = _ReasoningTextBox()
                                streamed_tool_call_args[call_id].add(args_text)
                                if call_id in host_call_finalised:
                                    continue
                                host_call_finalised.add(call_id)
                                await _publish_turn_frame(
                                    {
                                        "type": "response.function_call_arguments.done",
                                        "item_id": published_id,
                                        "output_index": index,
                                        "arguments": args_text,
                                    }
                                )
                        except Exception as exc:
                            self.logger.warning(
                                "Failed to stream tool-call arguments: %s",
                                exc,
                                exc_info=True,
                            )
                        continue

                    # --- Emit reasoning summary once done -----------------------
                    if etype == "response.reasoning_summary_text.done":
                        raw_text = (event.get("text") or "").strip()
                        if raw_text:
                            title_match = re.findall(r"\*\*(.+?)\*\*", raw_text)
                            title = title_match[-1].strip() if title_match else "Thinking…"
                            content = re.sub(r"\*\*(.+?)\*\*", "", raw_text).strip()
                            summary = title if not content else f"{title}\n{content}"
                            if event_emitter:
                                note_model_activity()
                                key = _reasoning_stream_key(event, etype)
                                if thinking_box_enabled:
                                    normalized_summary = (
                                        _normalize_surrogate_chunk(summary, "reasoning") if summary else ""
                                    )
                                    append = ""
                                    if normalized_summary:
                                        append = _append_reasoning_text(
                                            key,
                                            normalized_summary,
                                            allow_misaligned=False,
                                        )
                                    if not append:
                                        append = _append_summary_tail(key, raw_text)
                                    if append:
                                        reasoning_summary_consumed[key] = raw_text
                                        note_generation_activity()
                                        reasoning_stream_active = True
                                        display_state = _reasoning_display_state(key)
                                        _rearm_reasoning_window(key, display_state)
                                        _reopen_reasoning_window(key, display_state)
                                if thinking_status_enabled:
                                    cancel_thinking()
                                    await event_emitter(
                                        {
                                            "type": "status",
                                            "data": {"description": summary},
                                        }
                                    )
                        continue

                    if etype == "response.output_text.annotation.added":
                        ann = event.get("annotation") or {}
                        await _notify_unhandled_citations([ann])
                        if ann.get("type") == "url_citation":
                            payload = ann.get("url_citation") if isinstance(ann.get("url_citation"), dict) else ann
                            url = (payload.get("url") or "").strip()
                            url = url.removesuffix("?utm_source=openai")
                            title = (payload.get("title") or url).strip()
                            ann_content = payload.get("content")
                            ann_content = ann_content.strip() if isinstance(ann_content, str) else ""

                            if not url or url in ordinal_by_url:
                                continue

                            ordinal_by_url[url] = len(ordinal_by_url) + 1

                            host = _citation_host(url)
                            citation = {
                                "source": {"name": host or "source", "url": url},
                                "document": [ann_content[:citation_excerpt_max] if ann_content else title],
                                "metadata": [{
                                    "source": url,
                                    "date_accessed": citation_access_stamp(),
                                }],
                            }
                            try:
                                await self._pipe._event_emitter_handler._emit_citation(event_emitter, citation)
                            except Exception as exc:
                                self.logger.debug("Failed to emit annotation citation: %s", exc, exc_info=True)
                            emitted_citations.append(citation)

                        continue


                    if etype == "response.output_item.added":
                        item_raw = event.get("item")
                        item = item_raw if isinstance(item_raw, dict) else {}
                        item_type = item.get("type", "")
                        item_status = item.get("status", "")

                        if item_type and item_type != "reasoning" and reasoning_display:
                            _close_open_reasoning_windows()

                        if item_type == "reasoning":
                            iid = item.get("id")
                            if isinstance(iid, str) and iid:
                                active_reasoning_item_id = iid
                            continue

                        if item_type == "message" and item_status == "in_progress":
                            provider_status_seen = True
                            if (not responding_status_sent) and (not reasoning_stream_active) and event_emitter:
                                responding_status_sent = True
                                await event_emitter(
                                    {
                                        "type": "status",
                                        "data": {"description": "Responding to the user…"},
                                    }
                                )
                            continue

                        if owui_tool_passthrough and item_type == "function_call":
                            raw_call_id = item.get("call_id") or item.get("id")
                            call_id = raw_call_id.strip() if isinstance(raw_call_id, str) else ""
                            raw_item_id = item.get("id")
                            item_id = raw_item_id.strip() if isinstance(raw_item_id, str) else ""
                            raw_name = item.get("name")
                            tool_name = raw_name.strip() if isinstance(raw_name, str) else ""
                            if tool_name:
                                named_tool_call = True
                            if call_id:
                                if item_id:
                                    tool_call_item_ids[item_id] = call_id
                                streamed_tool_call_indices.setdefault(
                                    call_id, len(streamed_tool_call_indices)
                                )
                                if tool_name:
                                    tool_call_names[call_id] = tool_name
                                if event_emitter and tool_name:
                                    try:
                                        await _publish_host_call_item(
                                            call_id, item_id, _origin_tool_name(tool_name)
                                        )
                                    except Exception as exc:
                                        self.logger.warning(
                                            "Failed to stream tool-call arguments: %s",
                                            exc,
                                            exc_info=True,
                                        )
                            continue

                        if item_type.startswith("openrouter:") and event_emitter:
                            tool_label = {
                                "openrouter:web_search": "Searching the web…",
                                "openrouter:web_fetch": "Fetching web page…",
                                "openrouter:datetime": "Getting current time…",
                                "openrouter:image_generation": "Generating image…",
                                "openrouter:advisor": "Consulting advisor…",
                                "openrouter:subagent": "Delegating to worker…",
                                "openrouter:experimental__search_models": "Searching models…",
                                "openrouter:shell": "Running shell commands…",
                            }.get(item_type, f"Running {item_type}…")
                            await self._pipe._event_emitter_handler._emit_status(
                                event_emitter, tool_label, done=False
                            )
                            opened_image_windows.add(str(item.get("id") or ""))

                    if etype == "response.output_item.done":
                        item_raw = event.get("item")
                        item = item_raw if isinstance(item_raw, dict) else {}
                        item_type = item.get("type", "")
                        item_name = item.get("name", "unnamed_tool")

                        if item_type == "message":
                            content = item.get("content")
                            if isinstance(content, list):
                                for content_part in content:
                                    if not isinstance(content_part, dict):
                                        continue
                                    if content_part.get("type") != "output_text":
                                        continue
                                    await _emit_annotation_citations(content_part.get("annotations"))
                            whole_refusal = responses_refusal_text(item)
                            buffered = "".join(piece for piece in refusal_pending if piece)
                            refusal_pending.clear()
                            if whole_refusal and whole_refusal not in assistant_message:
                                buffered = whole_refusal if not buffered else buffered
                            if buffered and buffered not in assistant_message:
                                note_model_activity()
                                if reasoning_display:
                                    _close_open_reasoning_windows()
                                    for reasoning_key in list(reasoning_display):
                                        await _emit_reasoning_item(reasoning_key, assistant_message)
                                published = _normalize_surrogate_chunk(buffered, "assistant")
                                if published:
                                    note_generation_activity()
                                    if (
                                        not provider_status_seen
                                        and not responding_status_sent
                                        and not reasoning_stream_active
                                        and event_emitter
                                        and not fusion_armed
                                    ):
                                        provider_status_seen = True
                                        responding_status_sent = True
                                        await event_emitter(
                                            {
                                                "type": "status",
                                                "data": {"description": "Responding to the user…"},
                                            }
                                        )
                                    if assistant_message:
                                        published = (
                                            _continuation_lead(assistant_message) or "\n\n"
                                        ) + published
                                    assistant_message += published
                                    if not fusion_armed:
                                        await _open_message()
                                        await _publish_turn_frame(
                                            {
                                                "type": "chat:message:delta",
                                                "data": {"content": published},
                                            }
                                        )
                            await _emit_annotation_citations(item.get("annotations"))
                            phase_marker = _phase_marker_for_output_item(item)
                            if not api_hold_key:
                                if phase_marker:
                                    assistant_message = await _append_assistant_hidden_markers(assistant_message, [phase_marker])
                                reasoning_anchor_state["text_chunks"] += 1
                                reasoning_anchor_state["chars_at_last_chunk"] = len(
                                    assistant_message
                                )
                            continue

                        should_persist = False
                        if item_type == "reasoning":
                            should_persist = valves.PERSIST_REASONING_TOKENS in {"next_reply", "conversation"}

                        elif item_type in _NON_REPLAYABLE_TOOL_ARTIFACTS:
                            should_persist = False

                        elif item_type == "function_call":
                            should_persist = False
                            round_saw_function_call += 1
                            calls_in_this_round += 1
                            if item.get("name"):
                                named_tool_call = True
                                reasoning_anchor_state["stream_calls"] += len(
                                    split_tool_argument_objects(
                                        _read_arguments_as_open_webui_reads_them(item)
                                    )
                                )

                        elif isinstance(item_type, str) and item_type.startswith("openrouter:"):
                            should_persist = False

                        else:
                            should_persist = True

                        if isinstance(item_type, str) and item_type.startswith("openrouter:") and item.get("id"):
                            for _pending_reasoning, _stream_pos, _text_pos in reasoning_anchor_state["awaiting"]:
                                if _stream_pos == reasoning_anchor_state["stream_calls"]:
                                    _pending_reasoning.setdefault(REASONING_FOLLOWING_SERVER_ITEM_KEY, str(item["id"]))

                        if should_persist:
                            normalized_item = normalize_persisted_item(item)
                            if normalized_item:
                                if item_type == "reasoning":
                                    normalized_item[REASONING_ANCHOR_SEQ_KEY] = reasoning_anchor_state["seq"]
                                    reasoning_anchor_state["seq"] += 1
                                    _text_ordinal = reasoning_anchor_state["text_chunks"] + (
                                        1
                                        if len(assistant_message)
                                        > reasoning_anchor_state["chars_at_last_chunk"]
                                        else 0
                                    )
                                    reasoning_anchor_state["awaiting"].append(
                                        (
                                            normalized_item,
                                            reasoning_anchor_state["stream_calls"],
                                            _text_ordinal,
                                        )
                                    )
                                row = self._pipe._artifact_store._make_db_row(
                                    persist_chat_id, persist_message_id, openwebui_model, normalized_item
                                )
                                if row:
                                    pending_items.append(row)


                        title = f"Running `{item_name}`"
                        content = ""
                        image_markdowns: list[str] = []

                        if item_type == "function_call":
                            title = f"Running the {item_name} tool…"
                            raw_arguments = item.get("arguments")
                            if isinstance(raw_arguments, str) and not raw_arguments.strip():
                                title = f"Tool call requested: {item_name} (arguments pending)"
                                content = ""
                            else:
                                parsed_arguments: dict[str, Any] | None = None
                                if isinstance(raw_arguments, dict):
                                    parsed_arguments = raw_arguments
                                elif isinstance(raw_arguments, str):
                                    parsed = _safe_json_loads(raw_arguments)
                                    if isinstance(parsed, dict):
                                        parsed_arguments = parsed

                                if parsed_arguments is not None:
                                    args_formatted = ", ".join(
                                        f"{k}={json.dumps(v, ensure_ascii=False)}"
                                        for k, v in parsed_arguments.items()
                                    )
                                    if len(args_formatted) > 1000:
                                        args_formatted = args_formatted[:1000] + "…"
                                    invocation = (
                                        f"{item_name}({args_formatted})"
                                        if args_formatted
                                        else f"{item_name}()"
                                    )
                                else:
                                    raw_text = ""
                                    if isinstance(raw_arguments, str):
                                        raw_text = raw_arguments.strip()
                                    elif raw_arguments is not None:
                                        with contextlib.suppress(TypeError, ValueError):
                                            raw_text = json.dumps(raw_arguments, ensure_ascii=False)
                                        if not raw_text:
                                            raw_text = str(raw_arguments)

                                    if not raw_text or raw_text == "{}":
                                        invocation = f"{item_name}()"
                                    else:
                                        truncated = (
                                            raw_text
                                            if len(raw_text) <= 1000
                                            else (raw_text[:1000] + "…")
                                        )
                                        invocation = (
                                            "# Unparsed tool arguments "
                                            "(provider sent invalid/non-object JSON)\n"
                                            f"{item_name}(raw_arguments={json.dumps(truncated, ensure_ascii=False)})"
                                        )

                                content = wrap_code_block(invocation, "python")

                        elif item_type == "web_search_call":
                            action = item.get("action", {}) or {}

                            if action.get("type") == "search":
                                query = action.get("query")
                                sources = action.get("sources") or []
                                urls = [s.get("url") for s in sources if s.get("url")]

                                if event_emitter:
                                    if query:
                                        await event_emitter({
                                            "type": "status",
                                            "data": {
                                                "action": "web_search_queries_generated",
                                                "description": "Searching",
                                                "queries": [query],
                                                "done": False,
                                            },
                                        })

                                    if urls:
                                        await event_emitter({
                                            "type": "status",
                                            "data": {
                                                "action": "web_search",
                                                "description": "Reading through {{count}} sites",
                                                "query": query,
                                                "urls": urls,
                                                "done": False,
                                            },
                                        })

                            elif action.get("type") == "open_page":
                                if event_emitter:
                                    raw_title = action.get("title")
                                    raw_host = action.get("host")
                                    raw_url = action.get("url")
                                    title = raw_title.strip() if isinstance(raw_title, str) else ""
                                    host = raw_host.strip() if isinstance(raw_host, str) else ""
                                    url = raw_url.strip() if isinstance(raw_url, str) else ""
                                    description = "Opening page"
                                    if host:
                                        description = f"Opening {host}"
                                    elif title:
                                        description = f"Opening {title}"
                                    elif url:
                                        description = f"Opening {url}"
                                    await event_emitter({
                                        "type": "status",
                                        "data": {
                                            "action": "open_page",
                                            "description": description,
                                            "title": raw_title or (title or None),
                                            "host": raw_host or (host or None),
                                            "url": raw_url or (url or None),
                                            "done": False,
                                        },
                                    })
                                continue
                            elif action.get("type") == "find_in_page":
                                if event_emitter:
                                    raw_query = next(
                                        (
                                            value
                                            for value in (
                                                action.get("needle"),
                                                action.get("query"),
                                                action.get("text"),
                                            )
                                            if isinstance(value, str) and value.strip()
                                        ),
                                        "",
                                    )
                                    query = raw_query.strip() if isinstance(raw_query, str) else ""
                                    raw_page_url = action.get("url")
                                    page_url = raw_page_url.strip() if isinstance(raw_page_url, str) else ""
                                    description = "Searching within page"
                                    if query:
                                        description = f"Searching for {query}"
                                    await event_emitter({
                                        "type": "status",
                                        "data": {
                                            "action": "find_in_page",
                                            "description": description,
                                            "needle": raw_query or (query or None),
                                            "url": raw_page_url or (page_url or None),
                                            "done": False,
                                        },
                                    })
                                continue
                                    
                            continue

                        elif item_type == "file_search_call":
                            title = "Let me skim those files…"
                        elif item_type in ("image_generation_call", "openrouter:image_generation"):
                            title = "Let me create that image…"
                            image_handled = False
                            if (
                                item_type == "openrouter:image_generation"
                                and _image_item_is_empty(item)
                                and server_tool_status(item) == "incomplete"
                                and not item.get("error")
                            ):
                                self.logger.warning(
                                    "Image generation returned no image for item '%s'",
                                    item.get("id") or "<unknown>",
                                )
                                await self._pipe._event_emitter_handler._emit_notification(
                                    event_emitter,
                                    f"Image generation failed: {IMAGE_NO_IMAGES_REASON}",
                                    level="warning",
                                )
                                image_handled = True
                            elif server_tool_status(item) == "incomplete":
                                error_msg = item.get("error") or "Image generation failed"
                                self.logger.warning("Image generation error: %s", error_msg)
                                await self._pipe._event_emitter_handler._emit_notification(
                                    event_emitter,
                                    f"Image generation failed: {error_msg}",
                                    level="warning",
                                )
                                image_handled = True
                            if not image_handled:
                                item_id = item.get("id")
                                if item_id and item_id in processed_image_item_ids:
                                    self.logger.debug("Skipping duplicate image item '%s'", item_id)
                                else:
                                    if item_id:
                                        processed_image_item_ids.add(item_id)
                                    try:
                                        image_markdowns = await _render_image_markdown(item)
                                        for blob_key in ("result", "imageUrl", "imageB64"):
                                            val = item.get(blob_key)
                                            if isinstance(val, str) and len(val) > 1024:
                                                item[blob_key] = "[image persisted to storage]"
                                    except Exception:
                                        self.logger.exception(
                                            "Failed to process generated image for item '%s'",
                                            item_id or "<unknown>",
                                        )
                                        await self._pipe._event_emitter_handler._emit_status(
                                            event_emitter,
                                            "⚠️ Unable to process generated image output",
                                            done=False,
                                        )
                                        image_markdowns = []
                            if image_handled and str(item.get("id") or "") in opened_image_windows:
                                await self._pipe._event_emitter_handler._emit_status(
                                    event_emitter, "", done=True
                                )
                        elif item_type == "openrouter:datetime":
                            title = None
                            dt_val = item.get("datetime", "")
                            tz_val = item.get("timezone", "")
                            result_text = json.dumps({"datetime": dt_val, "timezone": tz_val}, indent=2)
                            effective_id = server_tool_call_id(item.get("id"))
                            if emitter_supplied:
                                await _publish_open_reasoning_before_a_card(assistant_message)
                                effective_id = await _emit_tool_start(
                                    call_id=effective_id,
                                    name="datetime",
                                    arguments="{}",
                                    status=server_tool_status(item),
                                    current_text=assistant_message,
                                    model_call=False,
                                )
                                await _emit_tool_result(
                                    call_id=effective_id,
                                    result_text=result_text,
                                    status=server_tool_status(item),
                                    current_text=assistant_message,
                                )
                            assistant_message = await _commit_server_tool_round(
                                effective_id, "datetime", server_tool_status(item),
                                item_type=item_type, result_text=result_text, arguments="{}", current_text=assistant_message,
                            )
                            await self._pipe._event_emitter_handler._emit_status(event_emitter, "", done=True)
                        elif item_type == "openrouter:web_search":
                            title = None
                            result_text = (
                                "Search completed. Sources available in the citations panel below."
                                if server_tool_status(item) == "completed"
                                else "Search did not complete."
                            )
                            effective_id = server_tool_call_id(item.get("id"))
                            if emitter_supplied:
                                await _publish_open_reasoning_before_a_card(assistant_message)
                                effective_id = await _emit_tool_start(
                                    call_id=effective_id,
                                    name="web_search",
                                    arguments="{}",
                                    status=server_tool_status(item),
                                    current_text=assistant_message,
                                    model_call=False,
                                )
                                await _emit_tool_result(
                                    call_id=effective_id,
                                    result_text=result_text,
                                    status=server_tool_status(item),
                                    current_text=assistant_message,
                                )
                            assistant_message = await _commit_server_tool_round(
                                effective_id, "web_search", server_tool_status(item),
                                item_type=item_type, result_text=result_text, arguments="{}", current_text=assistant_message,
                            )
                            action = item.get("action") if isinstance(item.get("action"), dict) else {}
                            search_urls: list[str] = []
                            for u in (action.get("sources") or []):
                                if isinstance(u, dict):
                                    href = u.get("url")
                                    if isinstance(href, str) and href:
                                        search_urls.append(href)
                                elif isinstance(u, str) and u:
                                    search_urls.append(u)
                            if search_urls and event_emitter:
                                await event_emitter({"type": "status", "data": {
                                    "action": "web_search",
                                    "description": f"Searched {len(search_urls)} sites",
                                    "done": True,
                                    "urls": search_urls,
                                }})
                            await self._pipe._event_emitter_handler._emit_status(event_emitter, "", done=True)
                        elif item_type == "openrouter:web_fetch":
                            title = None
                            fetch_url = item.get("url", "")
                            fetch_error = item.get("error")
                            fetch_content = item.get("content")
                            if fetch_error:
                                result_text = str(fetch_error)
                            elif isinstance(fetch_content, str) and fetch_content:
                                result_text = fetch_content
                            else:
                                http_status = item.get("httpStatus")
                                result_text = f"Fetch failed (HTTP {http_status})" if http_status else "Fetch failed or returned no content."
                            args_text = json.dumps({"url": fetch_url}, ensure_ascii=False) if fetch_url else "{}"
                            effective_id = server_tool_call_id(item.get("id"))
                            if emitter_supplied:
                                await _publish_open_reasoning_before_a_card(assistant_message)
                                effective_id = await _emit_tool_start(
                                    call_id=effective_id,
                                    name="web_fetch",
                                    arguments=args_text,
                                    status=server_tool_status(item),
                                    current_text=assistant_message,
                                    model_call=False,
                                )
                                await _emit_tool_result(
                                    call_id=effective_id,
                                    result_text=result_text,
                                    status=server_tool_status(item),
                                    current_text=assistant_message,
                                )
                            assistant_message = await _commit_server_tool_round(
                                effective_id, "web_fetch", server_tool_status(item),
                                item_type=item_type, result_text=result_text, arguments=args_text, current_text=assistant_message,
                            )
                            await self._pipe._event_emitter_handler._emit_status(event_emitter, "", done=True)
                        elif item_type in _SHELL_CALL_ARTIFACTS:
                            title = None
                            args_text = json.dumps(
                                {"commands": _shell_call_commands(item)}, ensure_ascii=False
                            )
                            result_text = _shell_call_result_text(item)
                            raw_call_id = item.get("call_id")
                            effective_id = (
                                raw_call_id.strip()
                                if isinstance(raw_call_id, str) and raw_call_id.strip()
                                else server_tool_call_id(item.get("id"))
                            )
                            if emitter_supplied:
                                effective_id = await _emit_tool_start(
                                    call_id=effective_id,
                                    name="shell",
                                    arguments=args_text,
                                    status=server_tool_status(item),
                                    current_text=assistant_message,
                                )
                                if result_text:
                                    await _emit_tool_result(
                                        call_id=effective_id,
                                        result_text=result_text,
                                        status=server_tool_status(item),
                                        current_text=assistant_message,
                                    )
                            if effective_id not in committed_shell_calls:
                                committed_shell_calls.add(effective_id)
                                assistant_message = await _commit_server_tool_round(
                                    effective_id, "shell", server_tool_status(item),
                                    item_type=item_type, result_text=result_text, arguments=args_text, current_text=assistant_message,
                                )
                            await self._pipe._event_emitter_handler._emit_status(event_emitter, "", done=True)
                        elif isinstance(item_type, str) and item_type.startswith("openrouter:"):
                            title = None
                            tool_name = item_type.split(":", 1)[1] or item_type
                            result_text = server_tool_result_text(item)
                            server_arguments = json.dumps(server_tool_arguments(item), ensure_ascii=False)
                            effective_id = server_tool_call_id(item.get("id"))
                            if emitter_supplied:
                                await _publish_open_reasoning_before_a_card(assistant_message)
                                effective_id = await _emit_tool_start(
                                    call_id=effective_id,
                                    name=tool_name,
                                    arguments=server_arguments,
                                    status=server_tool_status(item),
                                    current_text=assistant_message,
                                    model_call=False,
                                )
                                await _emit_tool_result(
                                    call_id=effective_id,
                                    result_text=result_text,
                                    status=server_tool_status(item),
                                    current_text=assistant_message,
                                )
                            assistant_message = await _commit_server_tool_round(
                                effective_id, tool_name, server_tool_status(item),
                                item_type=item_type, result_text=result_text, arguments=server_arguments, raw_item=item,
                                current_text=assistant_message,
                            )
                            await self._pipe._event_emitter_handler._emit_status(event_emitter, "", done=True)
                        elif item_type == "reasoning":
                            title = None
                            key = _reasoning_stream_key(event, etype)
                            snapshot = _extract_reasoning_text_from_item(item)
                            normalized_snapshot = (
                                _normalize_surrogate_chunk(snapshot, "reasoning") if snapshot else ""
                            )
                            append = ""
                            if normalized_snapshot:
                                append = _append_reasoning_text(
                                    key,
                                    normalized_snapshot,
                                    allow_misaligned=True,
                                    consumed=reasoning_summary_consumed.get(key, ""),
                                )
                            if append:
                                reasoning_stream_active = True
                                note_model_activity()
                                note_generation_activity()
                                opened = reasoning_display.get(key)
                                if opened is not None:
                                    _rearm_reasoning_window(key, opened)
                                await _maybe_emit_reasoning_status(append)
                                await _maybe_emit_reasoning_status("", force=True)
                            if emitter_supplied and calls_in_this_round:
                                _reasoning_display_state(key)
                                deferred_reasoning_keys.setdefault(key, round_saw_function_call)
                            else:
                                await _emit_reasoning_item(key, assistant_message)

                        if title:
                            desc = title if not content else f"{title}\n{content}"
                            if thinking_tasks:
                                cancel_thinking()
                            self.logger.debug("Tool status update: %s", desc)

                        if image_markdowns:
                            note_model_activity()
                            note_generation_activity()
                            msg_before = len(assistant_message)
                            assistant_message += _continuation_lead(assistant_message)
                            for snippet in image_markdowns:
                                assistant_message = _append_output_block(assistant_message, snippet)
                            if event_emitter:
                                image_delta = assistant_message[msg_before:]
                                await _open_message()
                                await _publish_turn_frame(
                                    {"type": "chat:message:delta", "data": {"content": image_delta}}
                                )
                            if (
                                item_type in ("image_generation_call", "openrouter:image_generation")
                                and str(item.get("id") or "") in opened_image_windows
                            ):
                                await self._pipe._event_emitter_handler._emit_status(
                                    event_emitter, "", done=True
                                )

                        continue

                    if etype in ("response.completed", "response.done", "response.incomplete"):
                        if reasoning_display:
                            _close_open_reasoning_windows()
                            _defer_reasoning_keys_after_a_call(event.get("response"))
                            for reasoning_key in list(unpublished_reasoning_keys):
                                if reasoning_key in deferred_reasoning_keys:
                                    continue
                                await _emit_reasoning_item(reasoning_key, assistant_message)
                        if fusion_armed and fusion_batcher is not None:
                            for _straggler in fusion_batcher.flush_all():
                                await _emit_fusion_event(_straggler)
                        if fusion_armed and fusion_state is not None:
                            _fusion_resp = event.get("response")
                            if isinstance(_fusion_resp, dict) and "elapsed_seconds" not in _fusion_resp:
                                _fc = _fusion_resp.get("created_at")
                                _fco = _fusion_resp.get("completed_at")
                                if isinstance(_fc, (int, float)) and isinstance(_fco, (int, float)) and _fco >= _fc:
                                    _fusion_resp["elapsed_seconds"] = round(float(_fco) - float(_fc), 1)
                                else:
                                    _fstart = stream_started_at or request_started_at
                                    if _fstart is not None:
                                        _fusion_resp["elapsed_seconds"] = round(max(0.0, perf_counter() - _fstart), 1)
                            if fusion_state.record(event):
                                await _emit_fusion_embed_once()
                                _synth_terminal = fusion_state.synthesize_missing_analysis()
                                if _synth_terminal is not None:
                                    await _emit_fusion_event(_synth_terminal)
                                await _emit_fusion_event(event)
                        note_model_activity()
                        _terminal_response = event.get("response")
                        final_response = (
                            _terminal_response
                            if isinstance(_terminal_response, dict) and _terminal_response
                            else None
                        )
                        _round_annotations, _round_reasoning_details = (
                            _round_annotations_and_reasoning(final_response)
                        )
                        round_annotations.extend(_round_annotations)
                        round_reasoning_details.extend(_round_reasoning_details)
                        response_completed_at = perf_counter()
                        if generation_started_at is not None:
                            generation_last_event_at = response_completed_at
                        break

                if final_response is None:
                    error_occurred = not fusion_inner_call
                    answer_truncated = _STREAM_INTERRUPTED_REASON
                    if not fusion_inner_call:
                        session_log_reason = _STREAM_INTERRUPTED_REASON
                    if fusion_inner_call:
                        if outcome_sink is not None:
                            outcome_sink["member_ended_early"] = True
                        self.logger.warning("Stream ended without completion event for model=%s", body.model)
                        if event_emitter:
                            await self._pipe._event_emitter_handler._emit_completion(
                                event_emitter,
                                content=None if fusion_armed else assistant_message,
                                done=True,
                            )
                        break
                    reported = await self._pipe._ensure_error_formatter()._emit_templated_error(
                        event_emitter,
                        template=valves.STREAM_INTERRUPTED_TEMPLATE,
                        variables={"model": body.model or ""},
                        log_message=f"Stream ended without completion event for model={body.model}",
                        log_level=logging.WARNING,
                        terminal=False,
                        partial_answer=assistant_message,
                        fallback_template=DEFAULT_STREAM_INTERRUPTED_TEMPLATE,
                    )
                    if reported:
                        assistant_message = reported
                    if event_emitter and fusion_armed:
                        await self._pipe._event_emitter_handler._emit_completion(
                            event_emitter, content=None, done=True
                        )
                    break

                incomplete_details = final_response.get("incomplete_details")
                response_status = final_response.get("status")
                is_incomplete = response_status == "incomplete" or incomplete_details is not None
                if is_incomplete and not incomplete_warning_emitted:
                    incomplete_warning_emitted = True
                    reason = ""
                    if isinstance(incomplete_details, dict):
                        raw_reason = incomplete_details.get("reason") or incomplete_details.get("type")
                        if isinstance(raw_reason, str):
                            reason = raw_reason.strip()
                    _continues = any(
                        isinstance(i, dict)
                        and i.get("type") == "function_call"
                        and (
                            str(i.get("name") or "").strip()
                            or str(i.get("call_id") or i.get("id") or "").strip()
                        )
                        for i in (final_response.get("output") or [])
                    )
                    if _continues:
                        warning_msg = (
                            "Model response ended incomplete; attempting best-effort continuation."
                        )
                    else:
                        warning_msg = (
                            "Model response ended incomplete; the answer was cut short here. "
                            "Press Continue Response if your account has it, or regenerate, "
                            "to ask for the rest."
                        )
                    if reason:
                        warning_msg = f"{warning_msg} Reason: {reason}."
                    await _emit_budget_notice(warning_msg)

                raw_usage = final_response.get("usage") or {}
                usage = dict(raw_usage) if isinstance(raw_usage, dict) else {}

                priced_chars = dispatched_metered_chars
                if priced_chars is None and dispatched_model_id:
                    priced_chars = estimate_serialized_chars(
                        body.input,
                        referenced_sizes=getattr(body, "input_file_sizes", None),
                    ) + dispatch_overhead
                if priced_chars is not None:
                    measured = measure_chars_per_token(
                        metered_chars=priced_chars,
                        usage=usage,
                        model_id=dispatched_model_id,
                    )
                    if measured is not None:
                        record_chars_per_token(
                            body.budget_chars_per_token, dispatched_model_id, measured
                        )

                if usage:
                    usage["turn_count"] = 1
                    usage["function_call_count"] = sum(
                        1
                        for i in (final_response.get("output") or [])
                        if isinstance(i, dict) and i.get("type") == "function_call"
                    )
                    total_usage = merge_usage_stats(total_usage, usage)
                    intermediate_content = (
                        None
                        if (emitted_response_output_items or fusion_armed or open_webui_keeps_stored_output)
                        else (assistant_message if assistant_message else None)
                    )
                    await self._pipe._event_emitter_handler._emit_completion(
                        event_emitter,
                        content=intermediate_content,
                        usage=total_usage,
                        done=False,
                    )

                metadata_model = None
                if isinstance(metadata, dict):
                    model_block = metadata.get("model")
                    if isinstance(model_block, dict):
                        metadata_model = model_block.get("id")
                snapshot_model_id = self._pipe._qualify_model_for_pipe(
                    pipe_identifier,
                    metadata_model or body.model,
                )

                await maybe_dump_costs_snapshot(
                    self._pipe,
                    valves,
                    user_id=user_id or "",
                    model_id=snapshot_model_id,
                    usage=usage if usage else {},
                    user_obj=user_obj,
                    pipe_id=pipe_identifier,
                    chat_id=str(metadata.get("chat_id") or "") or None,
                    message_id=str(metadata.get("message_id") or "") or None,
                    kind="generation",
                )

                continuation_input_items: list[dict[str, Any]] = []
                reasoning_count = 0
                message_count = 0
                call_items: list[dict[str, Any]] = []
                invalid_call_outputs: list[dict[str, Any]] = []
                _carried_positions: list[int] = []
                for _position, item in enumerate(final_response.get("output") or []):
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "reasoning" and (item.get("encrypted_content") or item.get("signature")):
                        continuation_input_items.append(item)
                        reasoning_count += 1
                    elif item_type == "message":
                        continuation_input_items.append(item)
                        message_count += 1
                    elif item_type == "function_call":
                        arguments = _read_arguments_as_open_webui_reads_them(item)
                        parts = (
                            split_tool_argument_objects(arguments)
                            if not owui_tool_passthrough else [arguments]
                        )
                        carried: list[dict[str, Any]] = []
                        for _part in parts:
                            try:
                                _params = parse_tool_arguments(_part)
                            except ValueError:
                                _params = None
                            _stored = (
                                json.dumps(_params, ensure_ascii=False)
                                if isinstance(_params, dict) else _part
                            )
                            _candidate = normalize_persisted_item(
                                {
                                    **item,
                                    "arguments": _stored,
                                    **(
                                        {"id": generate_item_id(),
                                         "call_id": f"call_{uuid.uuid4().hex[:24]}"}
                                        if len(parts) > 1 else {}
                                    ),
                                }
                            )
                            if _candidate is None:
                                break
                            carried.append(_candidate)
                            _carried_positions.append(_position)
                        if carried:
                            call_items.extend(carried)
                            continuation_input_items.extend(carried)
                            continue
                        raw_call_id = item.get("call_id") or item.get("id")
                        call_id = raw_call_id.strip() if isinstance(raw_call_id, str) else ""
                        if not call_id:
                            self.logger.warning(
                                "Dropping malformed function_call without call_id (name=%s)",
                                item.get("name"),
                            )
                            continue
                        normalized_output = normalize_persisted_item(
                            {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": "Tool call missing name",
                                "status": "incomplete",
                            }
                        )
                        if normalized_output:
                            self.logger.warning(
                                "Refused a function_call with no usable name (call_id=%s, name=%r); "
                                "it was not run and no answer for it can reach the model",
                                call_id, item.get("name"),
                            )
                            invalid_call_outputs.append(normalized_output)
                        else:
                            self.logger.warning(
                                "Failed to normalize tool error output for call_id=%s",
                                call_id,
                            )

                _ordered = final_response.get("output") or []
                _fc_local = _carried_positions
                _calls_seen = reasoning_anchor_state["calls_seen"]
                _derived: list[tuple[str, int]] = []
                _derived_by_id: dict[str, tuple[str, int]] = {}
                _signed_by_id: dict[str, dict[str, str]] = {}
                for _i, _o in enumerate(_ordered):
                    if not (isinstance(_o, dict) and _o.get("type") == "reasoning"):
                        continue
                    _next = next((_m for _m, _p in enumerate(_fc_local) if _p > _i), None)
                    if _next is not None:
                        _entry = ("following", _calls_seen + _next)
                    else:
                        _prevs = [_m for _m, _p in enumerate(_fc_local) if _p < _i]
                        if _prevs:
                            _entry = ("preceding", _calls_seen + _prevs[-1])
                        elif _calls_seen > earlier_turn_calls:
                            _entry = ("preceding", _calls_seen - 1)
                        else:
                            _entry = ("", -1)
                    _derived.append(_entry)
                    _rid = _o.get("id")
                    if _rid:
                        _derived_by_id[_rid] = _entry
                        _carried: dict[str, str] = {}
                        _sig = _o.get("signature")
                        if isinstance(_sig, str) and _sig:
                            _carried["signature"] = _sig
                        _enc = _o.get("encrypted_content")
                        if isinstance(_enc, str) and _enc:
                            _carried["encrypted_content"] = _enc
                        _fmt = _o.get("format")
                        if isinstance(_fmt, str) and _fmt:
                            _carried["format"] = _fmt
                        if _carried:
                            _signed_by_id[_rid] = _carried
                if valves.PERSIST_REASONING_TOKENS in {"next_reply", "conversation"}:
                    _awaiting = reasoning_anchor_state["awaiting"]
                    _awaiting_ids = {
                        str(_pending.get("id"))
                        for _pending, _stream_pos, _text_pos in _awaiting
                        if _pending.get("id")
                    }
                    for _o in _ordered:
                        if not (isinstance(_o, dict) and _o.get("type") == "reasoning"):
                            continue
                        _rid = _o.get("id")
                        if not _rid or str(_rid) in _awaiting_ids:
                            continue
                        _normalized = normalize_persisted_item(_o)
                        if not _normalized:
                            continue
                        _normalized[REASONING_ANCHOR_SEQ_KEY] = reasoning_anchor_state["seq"]
                        reasoning_anchor_state["seq"] += 1
                        _awaiting.append(
                            (
                                _normalized,
                                reasoning_anchor_state["stream_calls"],
                                reasoning_anchor_state["text_chunks"],
                            )
                        )
                        _awaiting_ids.add(str(_rid))
                        _row = self._pipe._artifact_store._make_db_row(
                            persist_chat_id, persist_message_id, openwebui_model, _normalized
                        )
                        if _row:
                            pending_items.append(_row)
                _stream_total = reasoning_anchor_state["stream_calls"]
                _stream_consistent = _stream_total == _calls_seen + len(_fc_local)
                for _idx, (_pending_reasoning, _stream_pos, _text_pos) in enumerate(reasoning_anchor_state["awaiting"]):
                    _pending_id = _pending_reasoning.get("id")
                    if _pending_id and _pending_id in _signed_by_id:
                        _pending_reasoning.update(_signed_by_id[_pending_id])
                    if _pending_id and _pending_id in _derived_by_id:
                        _mode, _ordinal = _derived_by_id[_pending_id]
                    elif _stream_consistent and isinstance(_stream_pos, int):
                        if _stream_pos < _stream_total:
                            _mode, _ordinal = "following", _stream_pos
                        elif _stream_pos > earlier_turn_calls:
                            _mode, _ordinal = "preceding", _stream_pos - 1
                        else:
                            _mode, _ordinal = "", -1
                    elif _idx < len(_derived):
                        _mode, _ordinal = _derived[_idx]
                    elif _calls_seen > earlier_turn_calls:
                        _mode, _ordinal = "preceding", _calls_seen - 1
                    else:
                        _mode, _ordinal = "", -1
                    if _mode == "following":
                        _pending_reasoning[REASONING_FOLLOWING_ORDINAL_KEY] = _ordinal
                    elif _mode == "preceding":
                        _pending_reasoning[REASONING_PRECEDING_ORDINAL_KEY] = _ordinal
                    else:
                        _pending_reasoning[REASONING_TEXT_ORDINAL_KEY] = _text_pos
                reasoning_anchor_state["awaiting"] = []
                reasoning_anchor_state["calls_seen"] = _calls_seen + len(_fc_local)

                _origin_names = frozenset(
                    o for o in (_origin_tool_name(nm) for nm in offered_function_names)
                    if isinstance(o, str) and o
                )

                def _pipe_runs(name: str) -> bool:
                    return name in tool_registry or _origin_tool_name(name) in tool_registry

                def _open_webui_runs(name: str) -> bool:
                    owui_tools = metadata.get("tools") if isinstance(metadata, dict) else None
                    if not isinstance(owui_tools, dict):
                        return False
                    return name in owui_tools or _origin_tool_name(name) in owui_tools

                def _open_webui_owns_the_round(items: list[dict[str, Any]]) -> bool:
                    return bool(items) and all(
                        _open_webui_runs(str(c.get("name") or ""))
                        and not _pipe_runs(str(c.get("name") or ""))
                        for c in items
                    )

                hand_back = bool(call_items) and (
                    owui_tool_passthrough
                    or any(
                        (
                            (nm := str(c.get("name") or "")) in offered_function_names
                            or _origin_tool_name(nm) in _origin_names
                        )
                        and nm not in tool_registry
                        for c in call_items
                    )
                    and any(
                        str(c.get("name") or "") not in tool_registry
                        or _origin_tool_name(str(c.get("name") or "")) not in tool_registry
                        for c in call_items
                    )
                )
                if hand_back and chat_id and message_id and not metadata.get("task"):
                    hand_back_seen = self._pipe._hand_back_seen
                    now_seen = time.monotonic()
                    hand_back_seen[reply_key] = now_seen
                    if now_seen >= self._pipe._hand_back_swept_at:
                        self._pipe._hand_back_swept_at = now_seen + 3600.0
                        for stale_key, stale_at in list(hand_back_seen.items()):
                            if now_seen - stale_at > 3600.0:
                                hand_back_seen.pop(stale_key, None)
                                self._pipe._hand_back_counts.pop(stale_key, None)
                    counts = self._pipe._hand_back_counts
                    max_keys = max(
                        self._pipe._HAND_BACK_MAX_KEYS,
                        valves.MAX_CONCURRENT_REQUESTS + 2,
                    )
                    if len(counts) >= max_keys:
                        for oldest in list(counts):
                            if oldest != reply_key and counts[oldest] > valves.MAX_FUNCTION_CALL_LOOPS:
                                counts.pop(oldest, None)
                                break
                        else:
                            for oldest in list(counts):
                                if oldest != reply_key:
                                    counts.pop(oldest, None)
                                    break
                    while len(hand_back_seen) > max_keys:
                        for orphan in list(hand_back_seen):
                            if orphan != reply_key and orphan not in counts:
                                hand_back_seen.pop(orphan, None)
                                break
                        else:
                            break
                    spent = counts[reply_key] = min(
                        counts.get(reply_key, 0) + 1, valves.MAX_FUNCTION_CALL_LOOPS + 1
                    )
                    if spent > valves.MAX_FUNCTION_CALL_LOOPS and not _open_webui_owns_the_round(call_items):
                        hand_back = False
                        self.logger.debug(
                            "Hand-back cap reached for this reply (%d); answering in the loop instead",
                            valves.MAX_FUNCTION_CALL_LOOPS,
                        )

                if (call_items or invalid_call_outputs) and not hand_back:
                    note_model_activity()
                    if continuation_input_items:
                        body.input.extend(continuation_input_items)
                        input_is_sanitized = False
                    if reasoning_count:
                        self.logger.debug(
                            "🧠 Preserving %d reasoning item(s) with encrypted_content for tool continuation",
                            reasoning_count,
                        )
                    if message_count:
                        self.logger.debug(
                            "💬 Preserving %d message item(s) for tool continuation",
                            message_count,
                        )
                    if call_items:
                        self.logger.debug(
                            "📞 Preserving %d function_call item(s) for tool continuation",
                            len(call_items),
                        )
                    if not input_is_sanitized:
                        _replay_budget = _sanitize_request_input(
                            self._pipe, body,
                            verdicts=await _tool_picture_verdicts_for_input(
                                self._pipe, body.input,
                            ),
                        )
                        await _warn_if_futile(_replay_budget)
                        await _report_omissions(_replay_budget, _REPLAY_DROPPED_OPENING)
                        input_is_sanitized = True

                self.logger.debug("📞 Found %d function_call items in response", len(call_items))
                function_outputs: list[dict[str, Any]] = []
                if call_items or invalid_call_outputs:
                    if call_items and loop_index >= max_loops:
                        loop_limit_reached = True
                        ran_out = True

                    if call_items and hand_back:
                        handed_back = True
                        tool_calls_payload: list[dict[str, Any]] = []
                        try:
                            for call in call_items:
                                raw_call_id = call.get("call_id") or call.get("id")
                                call_id = raw_call_id.strip() if isinstance(raw_call_id, str) else ""
                                raw_name = call.get("name")
                                exposed_name = raw_name.strip() if isinstance(raw_name, str) else ""
                                tool_name = _origin_tool_name(exposed_name)
                                raw_args = call.get("arguments")
                                if isinstance(raw_args, str):
                                    args_text = raw_args.strip() or "{}"
                                else:
                                    args_text = json.dumps(raw_args or {}, ensure_ascii=False)
                                if not call_id or not tool_name:
                                    continue

                                tool_calls_payload.append(
                                    {
                                        "id": call_id,
                                        "type": "function",
                                        "function": {"name": tool_name, "arguments": args_text},
                                    }
                                )
                                pending_items.extend(
                                    _handed_back_round_rows(call, call_id, exposed_name, tool_name, args_text)
                                )

                                if body.stream and event_emitter:
                                    if owui_tool_passthrough:
                                        streamed_tool_call_indices.setdefault(
                                            call_id, len(streamed_tool_call_indices)
                                        )
                                        await _publish_host_call_item(
                                            call_id, str(call.get("id") or ""), tool_name
                                        )
                                        await _close_host_call_item(call_id, args_text)
                                    else:
                                        idx = streamed_tool_call_indices.setdefault(
                                            call_id, len(streamed_tool_call_indices)
                                        )
                                        await event_emitter(
                                            {
                                                "type": "chat:tool_calls",
                                                "data": {
                                                    "tool_calls": [
                                                        {
                                                            "index": idx,
                                                            "id": call_id,
                                                            "type": "function",
                                                            "function": {
                                                                "name": tool_name,
                                                                "arguments": args_text,
                                                            },
                                                        }
                                                    ]
                                                },
                                            }
                                        )
                        except Exception as exc:
                            self.logger.warning(
                                "Tool pass-through failed while building tool_calls payload: %s",
                                exc,
                                exc_info=True,
                            )

                        if not body.stream:
                            content_handed_back = True
                            try:
                                model_for_response = ""
                                metadata_model = metadata.get("model") if isinstance(metadata, dict) else None
                                if isinstance(metadata_model, dict):
                                    model_for_response = str(metadata_model.get("id") or "")
                                if not model_for_response:
                                    model_for_response = str(body.model or "pipe")
                                response = {
                                    "id": f"{model_for_response}-{uuid.uuid4()}",
                                    "object": "chat.completion",
                                    "created": int(time.time()),
                                    "model": model_for_response,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "message": {
                                                "role": "assistant",
                                                "content": assistant_message or None,
                                                **({"tool_calls": tool_calls_payload} if tool_calls_payload else {}),
                                            },
                                            "finish_reason": "tool_calls",
                                            "logprobs": None,
                                        }
                                    ],
                                    **({"usage": total_usage} if total_usage else {}),
                                }
                                if SessionLogger.debug_enabled(self.logger) and tool_calls_payload:
                                    summaries: list[dict[str, Any]] = []
                                    for call in tool_calls_payload:
                                        if not isinstance(call, dict):
                                            continue
                                        fn_raw = call.get("function")
                                        fn = fn_raw if isinstance(fn_raw, dict) else {}
                                        args = fn.get("arguments")
                                        summaries.append(
                                            {
                                                "id": call.get("id"),
                                                "type": call.get("type"),
                                                "name": fn.get("name"),
                                                "args_len": len(args) if isinstance(args, str) else None,
                                                "args_empty": (isinstance(args, str) and not args.strip()),
                                            }
                                        )
                                    self.logger.debug(
                                        "Returning non-streaming tool_calls response (request_id=%s): %s",
                                        (SessionLogger.request_id.get() or ""),
                                        json.dumps(summaries, ensure_ascii=False),
                                    )
                                _record_outcome()
                                return response
                            except Exception as exc:
                                self.logger.warning(
                                    "Failed to build non-streaming tool_calls response: %s",
                                    exc,
                                    exc_info=True,
                                )
                                _record_outcome()
                                return assistant_message

                        break

                    has_actionable_continuation = last_round_had_calls = bool(call_items or invalid_call_outputs)

                    if call_items:
                        if not tool_loops_executed:
                            assistant_len_before_tool_loops = len(assistant_message)
                        tool_loops_executed = True
                        committed_call_rows.clear()
                        committed_output_rows.clear()
                        calls_carded_this_round.clear()
                        stubbed_before_this_round: set[str] = set()

                        if emitter_supplied:
                            try:
                                for call in call_items:
                                    call_id = _extract_call_id(call)
                                    if not call_id:
                                        continue
                                    tool_name = (call.get("name") or "").strip()
                                    raw_args = call.get("arguments") or "{}"
                                    if not tool_name:
                                        continue
                                    args_text = (
                                        raw_args.strip()
                                        if isinstance(raw_args, str)
                                        else json.dumps(raw_args, ensure_ascii=False)
                                    )
                                    await _emit_tool_start(
                                        call_id=call_id,
                                        name=tool_name,
                                        arguments=args_text,
                                        status="in_progress" if _origin_tool_name(tool_name) == "ask_user" else "completed",
                                        current_text=assistant_message,
                                    )
                            except Exception as exc:
                                self.logger.warning("Failed to emit in-progress tool cards: %s", exc, exc_info=True)

                        if loop_limit_reached:
                            has_actionable_continuation = False
                            limit = max_loops
                            if not loop_limit_announced:
                                loop_limit_announced = True
                                self.logger.info(
                                    "Tool-call loop limit reached (%d/%d); injecting stubs for %d pending call(s).",
                                    limit, limit, len(call_items),
                                )
                            function_outputs = []
                            stubbed_before_this_round = set(stubbed_call_ids)
                            for call in call_items:
                                call_id = _extract_call_id(call) or f"call-{uuid.uuid4().hex}"
                                stubbed_call_ids.add(call_id)
                                function_outputs.append(
                                    {
                                        "type": "function_call_output",
                                        "call_id": call_id,
                                        "output": (
                                            f"TOOL_CALL_SKIPPED: tool-call loop limit reached ({limit}/{limit}); "
                                            "call not executed, so no output is available. "
                                            "Respond to the user using existing context (no further tool calls)."
                                        ),
                                        "status": "incomplete",
                                    }
                                )
                            if loop_index > max_loops and emitter_supplied:
                                try:
                                    for output in function_outputs:
                                        result_str, pictures = tool_output_text_and_pictures(
                                            output.get("output")
                                        )
                                        await _emit_tool_result(
                                            call_id=_extract_call_id(output) or "",
                                            result_text=result_str,
                                            files=output.get("files") or None,
                                            embeds=output.get("embeds") or None,
                                            status=str(output.get("status") or "completed"),
                                            pictures=pictures,
                                            current_text=assistant_message,
                                        )
                                except Exception as exc:
                                    self.logger.warning(
                                        "Failed to emit the skipped tool cards: %s", exc, exc_info=True
                                    )
                        else:
                            fresh_calls = [
                                call for call in call_items
                                if _extract_call_key(call) not in executed_tool_call_ids
                            ]
                            repeated_calls = [
                                call for call in call_items
                                if _extract_call_key(call) in executed_tool_call_ids
                            ]
                            repeated_outputs = [
                                {
                                    "type": "function_call_output",
                                    "call_id": _extract_call_id(call) or f"call-{uuid.uuid4().hex}",
                                    "output": (
                                        "TOOL_CALL_ALREADY_RUN: this identical call already has a result "
                                        "in this reply, so the tool is not run again. Use the result "
                                        "already provided, or call the tool again with a new call_id."
                                    ),
                                    "status": "incomplete",
                                }
                                for call in repeated_calls
                            ]
                            if not fresh_calls:
                                function_outputs = repeated_outputs
                                if loop_index > max_loops:
                                    break
                            else:
                                executed_tool_call_ids.update(
                                    key for key in (_extract_call_key(call) for call in fresh_calls)
                                    if key is not None
                                )
                                call_items = fresh_calls

                                _tool_ctx = self._pipe._TOOL_CONTEXT.get()
                                if _tool_ctx:
                                    async def _on_tool_complete(call: dict, result: dict) -> None:
                                        cid = _extract_call_id(call) or _extract_call_id(result)
                                        if not cid:
                                            return
                                        result_str, pictures = tool_output_text_and_pictures(result.get("output"))
                                        await _emit_tool_result(
                                            call_id=cid,
                                            result_text=result_str,
                                            files=result.get("files") or None,
                                            embeds=result.get("embeds") or None,
                                            status=str(result.get("status") or "completed"),
                                            pictures=pictures,
                                            current_text=_answer_carry[0],
                                        )
                                        if persist_message_id and cid in committed_call_rows and cid not in committed_output_rows:
                                            committed_output_rows.add(cid)
                                            rows = _round_output_row(result, cid)
                                            ulids = await _persist_rows(rows, "tool_result") if rows else []
                                            if ulids and not api_hold_key:
                                                _answer_carry[0] = await _append_assistant_hidden_markers(
                                                    _answer_carry[0],
                                                    [_serialize_marker(ulid) for ulid in ulids],
                                                )

                                    _tool_ctx.on_complete = _on_tool_complete
                                    _tool_ctx.carded_calls = (
                                        calls_carded_this_round if emitter_supplied and not _tool_ctx.fusion_inner else set()
                                    )

                                call_rows_at_start: list[dict[str, Any]] = []
                                for call in call_items if persist_message_id else []:
                                    cid = _extract_call_id(call)
                                    if cid and cid not in committed_call_rows:
                                        committed_call_rows.add(cid)
                                        call_rows_at_start.extend(_round_call_row(call, cid))
                                if call_rows_at_start:
                                    if thinking_tasks:
                                        cancel_thinking()
                                    pending_items.extend(call_rows_at_start)
                                    await _flush_pending("tool_calls")
                                    assistant_message = await _mark_committed_rows(assistant_message)

                                _answer_carry[0] = assistant_message

                                try:
                                    function_outputs = await self._pipe._ensure_tool_executor()._execute_function_calls(
                                        call_items,
                                        tool_registry,
                                    )
                                except Exception as exc:
                                    self.logger.warning(
                                        "Tool execution failed; continuing loop with model-visible error outputs: %s",
                                        exc,
                                        exc_info=True,
                                    )
                                    function_outputs = []
                                    for call in call_items:
                                        call_id = _extract_call_id(call) or f"call-{uuid.uuid4().hex}"
                                        function_outputs.append(
                                            {
                                                "type": "function_call_output",
                                                "call_id": call_id,
                                                "output": self._pipe._ensure_tool_executor()._tool_error_text(exc),
                                                "status": "incomplete",
                                            }
                                        )
                                finally:
                                    if _tool_ctx and _tool_ctx.on_complete is not None:
                                        _tool_ctx.on_complete = None
                                        _tool_ctx.carded_calls = set()
                                assistant_message = _answer_carry[0]
                                if repeated_outputs:
                                    function_outputs = list(function_outputs) + repeated_outputs

                        all_function_outputs = list(function_outputs)
                        budgeted_outputs = [
                            dict(output) if isinstance(output, dict) else output
                            for output in all_function_outputs
                        ]
                        budget = apply_live_tool_output_budget(
                            budgeted_outputs,
                            existing_input_items=body.input,
                            model_id=model_for_cache,
                            logger=self.logger,
                            referenced_sizes=getattr(body, "input_file_sizes", None),
                            reserved_output_tokens=getattr(body, "max_output_tokens", None),
                            chars_per_token=effective_chars_per_token(
                                body.budget_chars_per_token, budget_model_id(body)
                            ),
                            fixed_overhead_chars=_request_overhead_chars(body),
                        )

                        call_by_id: dict[str, dict] = {}
                        for call in call_items:
                            cid = _extract_call_id(call)
                            if cid:
                                call_by_id[cid] = call
                        output_by_call_id: dict[str, dict] = {}
                        for output in all_function_outputs:
                            cid = _extract_call_id(output)
                            if cid:
                                output_by_call_id[cid] = output

                        omitted_call_ids = budget.omitted_call_ids
                        await _report_omissions(
                            budget,
                            "Too large for the remaining context this turn, so the model "
                            "did not receive:",
                        )
                        if emitter_supplied and all_function_outputs:
                            try:
                                for cid, output in output_by_call_id.items():
                                    result_str, pictures = tool_output_text_and_pictures(output.get("output"))
                                    await _emit_tool_result(
                                        call_id=cid,
                                        result_text=result_str,
                                        files=output.get("files") or None,
                                        embeds=output.get("embeds") or None,
                                        status=str(output.get("status") or "completed"),
                                        pictures=pictures,
                                        current_text=assistant_message,
                                    )
                            except Exception as exc:
                                self.logger.warning("Failed to emit completed tool cards: %s", exc, exc_info=True)

                        for cid, output in output_by_call_id.items():
                            if cid in omitted_call_ids:
                                continue
                            call = call_by_id.get(cid)
                            if not call:
                                continue
                            tool_name = _origin_tool_name((call.get("name") or "").strip())
                            if output.get("status") != "completed" or tool_name in UNCITED_TOOLS:
                                continue
                            exposed_name = (call.get("name") or "").strip()
                            if not _cites_as_owui_builtin(exposed_name, tool_name):
                                continue
                            if not citations_enabled:
                                continue
                            try:
                                tool_result = output.get("output") or ""
                                if get_citation_source_from_tool_result is not None:
                                    tool_params = _safe_json_loads(call.get("arguments") or "{}")
                                    citations = get_citation_source_from_tool_result(
                                        tool_name=tool_name,
                                        tool_params=tool_params if isinstance(tool_params, dict) else {},
                                        tool_result=tool_result,
                                        tool_id=cid,
                                    )
                                    for source in citations:
                                        if isinstance(source, dict):
                                            emitted_citations.append(source)
                                        if event_emitter:
                                            await self._pipe._event_emitter_handler._emit_citation(event_emitter, source)
                                            self.logger.debug(
                                                "Emitted citation from tool=%s: %s",
                                                tool_name,
                                                source.get("source", {}).get("name", "unknown"),
                                            )
                                    continue
                                result_text = tool_result if isinstance(tool_result, str) else str(tool_result)
                                for url, title, snippet in harvest_tool_citations(result_text):
                                    if url in ordinal_by_url:
                                        continue
                                    ordinal_by_url[url] = len(ordinal_by_url) + 1
                                    host = _citation_host(url)
                                    citation = {
                                        "source": {"name": host or "source", "url": url},
                                        "document": [(snippet or title or url)[:citation_excerpt_max]],
                                        "metadata": [{
                                            "source": url,
                                            "date_accessed": citation_access_stamp(),
                                        }],
                                    }
                                    emitted_citations.append(citation)
                                    if event_emitter:
                                        await self._pipe._event_emitter_handler._emit_citation(event_emitter, citation)
                                        self.logger.debug(
                                            "Emitted citation from tool=%s: %s", tool_name, url,
                                        )
                            except Exception as exc:
                                self.logger.warning(
                                    "Failed to extract citations from tool=%s: %s",
                                    tool_name,
                                    exc,
                                    exc_info=True,
                                )

                        call_rows: list[dict[str, Any]] = []
                        output_rows: list[dict[str, Any]] = []
                        for output in all_function_outputs if persist_message_id else []:
                            cid = _extract_call_id(output)
                            if cid in stubbed_before_this_round:
                                continue
                            call = call_by_id.get(cid) if cid else None
                            if not call:
                                continue
                            if cid not in committed_call_rows:
                                committed_call_rows.add(cid)
                                call_rows.extend(_round_call_row(call, cid))
                            if cid not in committed_output_rows:
                                committed_output_rows.add(cid)
                                output_rows.extend(_round_output_row(output, cid))
                        round_rows = call_rows + output_rows
                        if round_rows:
                            if thinking_tasks:
                                cancel_thinking()
                            pending_items.extend(round_rows)
                            await _flush_pending("tool_round")
                            assistant_message = await _mark_committed_rows(assistant_message)

                        for output in all_function_outputs:
                            if thinking_tasks:
                                cancel_thinking()
                            self.logger.debug("Received tool result\n%s", _tool_result_for_log(output))
                        if loop_index > max_loops:
                            break
                        round_refusals: list[tuple[str, str, str]] = []
                        for position, round_output in enumerate(budgeted_outputs):
                            if not isinstance(round_output, dict):
                                continue
                            gated_output, round_refused = await _gate_round_output_pictures(
                                self._pipe,
                                round_output.get("output"),
                                self._pipe.valves.BASE64_MAX_SIZE_MB * 1024 * 1024,
                                allow_insecure=(
                                    self._pipe._multimodal_handler._is_insecure_http_allowed
                                ),
                            )
                            if round_refused:
                                budgeted_outputs[position] = {**round_output, "output": gated_output}
                                round_refusals.extend(round_refused)
                        if round_refusals:
                            for url, reason, cause in round_refusals:
                                self.logger.warning(
                                    "Not forwarding a tool's picture (%s): %s [cause=%s]",
                                    loggable_link(url), reason, cause,
                                )
                            await self._pipe._event_emitter_handler._emit_status(
                                event_emitter, _tool_picture_notice(round_refusals), done=False,
                            )
                        body.input.extend(budgeted_outputs)
                        input_is_sanitized = False
                        shipped_budget = _sanitize_request_input(
                            self._pipe, body,
                            verdicts=await _tool_picture_verdicts_for_input(
                                self._pipe, body.input,
                            ),
                        )
                        await _warn_if_futile(shipped_budget)
                        await _report_omissions(shipped_budget, _REPLAY_DROPPED_OPENING)
                        input_is_sanitized = True
                    elif invalid_call_outputs:
                        if not tool_loops_executed:
                            assistant_len_before_tool_loops = len(assistant_message)
                        tool_loops_executed = True
                        all_function_outputs = list(invalid_call_outputs)
                        for output in all_function_outputs:
                            if thinking_tasks:
                                cancel_thinking()
                            self.logger.debug("Received tool result\n%s", _tool_result_for_log(output))
                    else:
                        break
                else:
                    has_actionable_continuation = False
                    last_round_had_calls = False
                    break

            if ran_out and tool_loops_executed and last_round_had_calls:
                limit_note = f"Tool-call limit reached ({max_loops} iterations)."
                if not emitter_supplied:
                    assistant_message = join_answer_and_card(assistant_message, limit_note)
                else:
                    await self._pipe._event_emitter_handler._emit_notification(
                        event_emitter, limit_note, level="warning"
                    )
                    if (not was_cancelled) and chat_id and message_id and Chats is not None:
                        with contextlib.suppress(Exception):
                            await Chats.upsert_message_to_chat_by_id_and_message_id(
                                chat_id, message_id, {"error": {"content": limit_note}}
                            )

            if (
                tool_loops_executed
                and not strip_hidden_marker_lines(assistant_message[assistant_len_before_tool_loops:]).strip()
                and not has_actionable_continuation
            ):
                await self._pipe._event_emitter_handler._emit_notification(
                    event_emitter,
                    "No assistant content was produced after tool execution; sending fallback guidance.",
                    level="warning",
                )
                fallback = NO_CONTENT_AFTER_TOOLS_FALLBACK
                joined = join_answer_and_card(assistant_message, fallback)
                delta = joined[len(assistant_message):]
                assistant_message = joined
                if event_emitter:
                    await _open_message()
                    await event_emitter(
                        {
                            "type": "chat:message:delta",
                            "data": {"content": delta},
                        }
                    )

        except asyncio.CancelledError:
            was_cancelled = True
            session_log_reason = "cancelled"
            _record_outcome()
            raise
        except OpenRouterAPIError as exc:
            error_occurred = True
            session_log_reason = str(exc)
            cancel_thinking()
            if not retry_barrier_crossed:
                handed_back_for_retry = True
                _record_outcome()
                raise
            reported = await self._pipe._ensure_error_formatter()._report_openrouter_error(
                exc,
                event_emitter=event_emitter,
                normalized_model_id=body.model,
                api_model_id=getattr(body, "api_model", None),
                usage=total_usage,
                partial_answer=assistant_message,
                terminal=False,
            )
            if reported:
                assistant_message = reported
        except RequiredInternalFileError as e:
            error_occurred = True
            session_log_reason = str(e)
            cancel_thinking()
            reported = await self._pipe._ensure_error_formatter()._emit_error(
                event_emitter,
                e.user_message,
                show_error_message=True,
                done=True,
                partial_answer=assistant_message,
                terminal=False,
            )
            if reported:
                assistant_message = reported
            self.logger.warning("Required internal file unavailable in streaming loop: %s", e.user_message)
        except UpstreamBodyUnreadable as e:
            error_occurred = True
            session_log_reason = str(e)
            if bool((metadata or {}).get("chat_id")) and bool((metadata or {}).get("message_id")):
                cancel_thinking()
                reported = await self._pipe._ensure_error_formatter()._emit_templated_error(
                    event_emitter,
                    template=valves.SERVICE_ERROR_TEMPLATE,
                    variables={"error_type": type(e).__name__, "status_code": "502",
                               "reason": e.summary(),
                               "body_excerpt": e.body_excerpt_block()},
                    log_message=(
                        f"Unreadable upstream body from {e.endpoint}: {e} "
                        f"(Content-Type: {e.content_type}): {e.body_excerpt[:200]}"
                    ),
                    partial_answer=assistant_message,
                    terminal=False,
                )
                if reported:
                    assistant_message = reported
            else:
                handed_back_for_retry = True
                _record_outcome()
                raise
        except Exception as e:  # pragma: no cover - network errors
            error_occurred = True
            session_log_reason = str(e)
            self.logger.exception("Unexpected error in streaming loop")
            if isinstance(e, (TimeoutError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError)):
                if strip_hidden_marker_lines(assistant_message).strip() or named_tool_call:
                    template, variables = valves.STREAM_INTERRUPTED_TEMPLATE, {"model": body.model or ""}
                elif isinstance(e, aiohttp.ConnectionTimeoutError):
                    template, variables = valves.NETWORK_TIMEOUT_TEMPLATE, {"timeout_seconds": valves.HTTP_CONNECT_TIMEOUT_SECONDS}
                elif isinstance(e, aiohttp.SocketTimeoutError):
                    template, variables = valves.NETWORK_TIMEOUT_TEMPLATE, {"timeout_seconds": valves.HTTP_SOCK_READ_SECONDS}
                elif isinstance(e, TimeoutError):
                    template, variables = valves.NETWORK_TIMEOUT_TEMPLATE, {"timeout_seconds": valves.HTTP_TOTAL_TIMEOUT_SECONDS}
                else:
                    template, variables = valves.CONNECTION_ERROR_TEMPLATE, {"error_type": type(e).__name__}
                reported = await self._pipe._ensure_error_formatter()._emit_templated_error(
                    event_emitter,
                    template=template,
                    variables=variables,
                    log_message=f"OpenRouter call failed in streaming loop: {type(e).__name__}: {e}",
                    partial_answer=assistant_message,
                    terminal=False,
                )
            else:
                reported = await self._pipe._ensure_error_formatter()._emit_templated_error(
                    event_emitter,
                    template=self._pipe.valves.INTERNAL_ERROR_TEMPLATE,
                    variables={"error_type": type(e).__name__},
                    log_message=f"Unexpected error in streaming loop: {e}",
                    partial_answer=assistant_message,
                    terminal=False,
                )
            if reported:
                assistant_message = reported

        finally:
            if unclaimed_token is not None:
                _UNCLAIMED_LATCH.reset(unclaimed_token)
            if event_iter is not None:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await asyncio.shield(_aclose_quietly(event_iter))
            cancel_thinking()
            for t in thinking_tasks:
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await t
            if handed_back_for_retry and retry_handoff is not None:
                retry_handoff[DEFERRED_REASONING_FLUSH] = functools.partial(
                    _flush_trailing_reasoning, assistant_message
                )
            else:
                await _flush_trailing_reasoning(assistant_message)
            surrogate_carry["assistant"] = ""
            surrogate_carry["reasoning"] = ""

            if fusion_no_usable_member:
                session_log_reason = _FUSION_PANEL_FAILURE_REASON

            terminal = bool(was_cancelled or error_occurred or not handed_back) and not handed_back_for_retry

            answer_delivered = bool(
                assistant_message.strip()
                or emitted_response_output_items
                or emitted_output_items
            )
            if outcome_sink is not None:
                outcome_sink["answer_delivered"] = answer_delivered

            if (
                terminal
                and not error_occurred
                and not was_cancelled
                and not fusion_inner_call
                and not answer_delivered
            ):
                error_occurred = True

            generation_status = _generation_status(
                was_cancelled, error_occurred, fusion_no_usable_member
            )
            if not handed_back_for_retry and not fusion_inner_call:
                try:
                    dispatch = asyncio.ensure_future(
                        self._pipe._dispatch_generation_complete(
                            total_usage if isinstance(total_usage, dict) else None,
                            generation_status,
                            request_id=SessionLogger.request_id.get() or "",
                            metadata=metadata,
                            task=None,
                        )
                    )
                    if outcome_sink is not None:
                        outcome_sink["generation_complete_dispatch"] = dispatch
                    await asyncio.shield(dispatch)
                except (asyncio.CancelledError, Exception):
                    self.logger.debug("generation-complete dispatch failed", exc_info=True)

            if (not error_occurred) and (not was_cancelled) and (not handed_back):
                try:
                    await self._cleanup_replayed_reasoning(body, valves, message_id)
                except (asyncio.CancelledError, Exception) as _exc:
                    if isinstance(_exc, asyncio.CancelledError):
                        _finalise_cancelled = _exc
                    self.logger.debug(
                        "Replayed-reasoning cleanup failed; this turn's own rows and the "
                        "terminal frames are unaffected",
                        exc_info=True,
                    )
            if (not error_occurred) and (not was_cancelled) and event_emitter:
                effective_start = stream_started_at or request_started_at
                elapsed = max(0.0, perf_counter() - effective_start)
                stream_window = None
                last_generation_stamp = generation_last_event_at or response_completed_at
                if last_generation_stamp is not None:
                    duration = max(0.0, last_generation_stamp - effective_start)
                    if duration > 0:
                        stream_window = duration
                if terminal or handed_back:
                    description = self._pipe._ensure_error_formatter()._format_final_status_description(
                        elapsed=elapsed,
                        total_usage=total_usage,
                        valves=valves,
                        stream_duration=stream_window,
                    )
                    try:
                        await event_emitter(
                            {
                                "type": "status",
                                "data": {
                                    "description": description,
                                    "done": True,
                                },
                            }
                        )
                    except (asyncio.CancelledError, Exception):
                        self.logger.exception("Failed to emit final status in finally")

            resolved_chat_id = str(metadata.get("chat_id") or "")
            from ..logging.session_log_manager import resolve_message_id
            resolved_message_id = resolve_message_id(metadata)
            request_id = SessionLogger.request_id.get() or ""
            if request_id:
                with SessionLogger._state_lock:
                    log_events = list(SessionLogger.logs.get(request_id, []))
                if log_events and self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug("Collected %d session log entries for request %s.", len(log_events), request_id)
                resolved_user_id = str(user_id or metadata.get("user_id") or "")
                resolved_session_id = str(metadata.get("session_id") or "")
                segment_status = _segment_status(
                    was_cancelled, error_occurred, fusion_no_usable_member, handed_back
                )
                try:
                    await asyncio.shield(
                        self._pipe._session_log_manager.persist_segment_to_db(
                            valves,
                            user_id=resolved_user_id,
                            session_id=resolved_session_id,
                            chat_id=resolved_chat_id,
                            message_id=resolved_message_id,
                            request_id=request_id,
                            log_events=log_events,
                            terminal=terminal,
                            status=segment_status,
                            reason=session_log_reason,
                            pipe_identifier=pipe_identifier,
                            task=str(metadata.get("task") or ""),
                        )
                    )
                except (asyncio.CancelledError, Exception):
                    self.logger.debug(
                        "Failed to persist session log segment (chat_id=%s message_id=%s request_id=%s terminal=%s)",
                        resolved_chat_id,
                        resolved_message_id,
                        request_id,
                        terminal,
                        exc_info=True,
                    )

            if fusion_embed_task is not None:
                if was_cancelled or handed_back_for_retry:
                    fusion_embed_task.cancel()
                try:
                    await fusion_embed_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    self.logger.warning(
                        "The Fusion panel card could not be built (model=%s); the turn continues without it",
                        body.model,
                        exc_info=True,
                    )

            if (
                fusion_armed and fusion_state is not None
                and fusion_state.fusion_index is None
                and not was_cancelled and not error_occurred
                and endpoint_override != "chat_completions"
                and any(
                    isinstance(p, dict) and p.get("id") == "fusion" and p.get("enabled") is not False
                    for p in (body.plugins or [])
                )
            ):
                if fell_back_to_chat:
                    self.logger.warning(
                        "No structured fusion events observed for fusion request model=%s: "
                        "the request was retried on /chat/completions, which cannot carry the "
                        "Fusion panel, so it did not run",
                        body.model,
                    )
                else:
                    self.logger.warning(
                        "No structured fusion events observed for fusion request model=%s",
                        body.model,
                    )

            if (
                fusion_armed and fusion_state is not None
                and fusion_state.fusion_index is not None and not was_cancelled
            ):
                _terminal_synth = fusion_state.synthesize_missing_analysis()
                if _terminal_synth is not None:
                    await _emit_fusion_event(_terminal_synth)

            if (
                fusion_armed and fusion_state is not None and fusion_state.fusion_index is not None
                and assistant_message and not error_occurred and not was_cancelled
            ):
                if event_emitter:
                    fusion_answer_item = {
                        "type": "message",
                        "id": open_message_id or f"msg-{uuid.uuid4().hex}",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": assistant_message}],
                    }
                    try:
                        recorded_message_chars = len(assistant_message)
                        await _record_output_item(fusion_answer_item, assistant_message)
                        answer_index = _output_index(fusion_answer_item)
                        await event_emitter({
                            "type": "response.output_item.added",
                            "output_index": answer_index,
                            "item": fusion_answer_item,
                        })
                        await event_emitter({
                            "type": "response.output_item.done",
                            "output_index": answer_index,
                            "item": fusion_answer_item,
                        })
                        emitted_response_output_items = True
                    except Exception:
                        self.logger.debug(
                            "Failed to emit fusion answer output item", exc_info=True
                        )
                assistant_message = (
                    '<details type="fusion_answer" done="true">\n'
                    '<summary>Final answer</summary>\n\n'
                    + assistant_message
                    + '\n</details>'
                )
                recorded_message_chars = len(assistant_message)

            if (
                fusion_armed and fusion_state is not None and fusion_state.fusion_index is not None
                and fusion_state.events and Chats is not None
                and not was_cancelled and not error_occurred
            ):
                try:
                    if resolved_chat_id and resolved_message_id:
                        _fusion_html = build_fusion_embed_html(
                            fusion_state, _fusion_model_names(), final=True
                        )
                        _content = assistant_message

                        async def _persist_fusion_snapshot() -> None:
                            _kept: list[object] = []
                            _existing = await Chats.get_message_by_id_and_message_id(
                                resolved_chat_id, resolved_message_id
                            )
                            _raw = (_existing or {}).get("embeds")
                            if isinstance(_raw, list):
                                _kept = [
                                    e for e in _raw
                                    if not (isinstance(e, str) and "<title>OpenRouter Fusion" in e)
                                ]
                            _kept.append(_fusion_html)
                            _payload: dict[str, object] = {"embeds": _kept}
                            if _content:
                                _payload["content"] = _content
                            await Chats.upsert_message_to_chat_by_id_and_message_id(
                                resolved_chat_id, resolved_message_id, _payload
                            )

                        await asyncio.shield(_persist_fusion_snapshot())
                except asyncio.CancelledError:
                    raise
                except Exception:
                    self.logger.warning("Failed to persist terminal fusion snapshot", exc_info=True)

            if was_cancelled or handed_back_for_retry:
                if was_cancelled:
                    with contextlib.suppress(BaseException):
                        await _flush_pending("abandoned")
                if pending_ulids:
                    self.logger.warning(
                        "An abandoned turn left %d committed artifact row(s) with no marker "
                        "addressing them; they are unreachable until retention removes them "
                        "(reason=%s chat_id=%s ulids=%s)",
                        len(pending_ulids),
                        "cancelled" if was_cancelled else "retry_handback",
                        chat_id,
                        list(pending_ulids),
                    )
                if pending_items:
                    self.logger.warning(
                        "An abandoned turn left %d artifact row(s) that no flush wrote: "
                        "either its best-effort flush was cancelled or failed with it, or "
                        "the turn was handed back for a retry that re-mints them "
                        "(reason=%s chat_id=%s item_types=%s)",
                        len(pending_items),
                        "cancelled" if was_cancelled else "retry_handback",
                        chat_id,
                        sorted({str(row.get("item_type")) for row in pending_items}),
                    )
            else:
                await _flush_pending("finalize")
                assistant_message = await _mark_committed_rows(assistant_message)

            if (
                chat_id
                and message_id
                and not metadata.get("task")
                and not metadata.get("assistant_message_id")
                and terminal
                and not handed_back_for_retry
                and not ran_out
                and not error_occurred
                and not was_cancelled
            ):
                _finished_key = reply_key
                self._pipe._hand_back_counts.pop(_finished_key, None)
                self._pipe._hand_back_seen.pop(_finished_key, None)

            reply_over = bool(
                terminal and not handed_back_for_retry and message_id and is_temporary_chat(chat_id)
            )
            if reply_over:
                self._pipe._artifact_store._reply_memory.release(chat_id, message_id)
            if api_hold_key:
                self._pipe._artifact_store._api_reply_memory.release(persist_chat_id, persist_message_id)

            terminal_output: list[dict[str, Any]] = []
            if (
                (not handed_back_for_retry)
                and (not was_cancelled)
                and terminal
                and (
                    emitted_output_items
                    or not error_occurred
                    or open_message_id is not None
                    or assistant_message
                )
            ):
                if not error_occurred or emitted_output_items:
                    await _capture_seeded_output()
                terminal_output = _terminal_output_items(assistant_message)
            if (
                outcome_sink is not None
                and terminal_output
                and terminal
                and (body.stream or not error_occurred or emitted_output_items)
            ):
                outcome_sink["output"] = terminal_output
                if total_usage:
                    outcome_sink["usage"] = dict(total_usage)
            if terminal_output and event_emitter:
                try:
                    await event_emitter({
                        "type": "response.completed",
                        "response": {"output": terminal_output},
                    })
                except (asyncio.CancelledError, Exception) as _exc:
                    if isinstance(_exc, asyncio.CancelledError):
                        _finalise_cancelled = _exc
                    self.logger.warning(
                        "Could not publish the terminal output array; tool calls and "
                        "reasoning may be missing from this turn's stored history",
                        exc_info=True,
                    )

            if not was_cancelled:
                if not error_occurred:
                    self._audit_orphan_tool_cards(emitted_tool_call_items, emitted_tool_output_items)
                try:
                    if terminal:
                        final_content = (
                            None
                            if (
                                body.stream
                                and not (error_occurred and is_channel_chat(chat_id))
                                and (emitted_response_output_items or open_webui_keeps_stored_output)
                            )
                            else assistant_message
                        )
                        final_output = None
                        if (
                            final_content is None
                            and terminal_output
                            and isinstance(chat_id, str)
                            and is_channel_chat(chat_id)
                        ):
                            final_output = terminal_output
                        await self._pipe._event_emitter_handler._emit_completion(
                            event_emitter,
                            content=final_content,
                            output=final_output,
                            usage=total_usage,
                            done=True,
                        )
                    else:
                        await self._pipe._event_emitter_handler._emit_completion(
                            event_emitter,
                            content=None,
                            usage=total_usage,
                            done=False,
                        )
                except (asyncio.CancelledError, Exception) as _exc:
                    if isinstance(_exc, asyncio.CancelledError):
                        _finalise_cancelled = _exc
                    self.logger.warning(
                        "Could not publish the terminal chat:completion frame; this turn's "
                        "stored record is still written",
                        exc_info=True,
                    )

            # Clear logs
            if request_id:
                SessionLogger.release(request_id)
                clear_timing_events(request_id)
            SessionLogger.cleanup()

            chat_id = metadata.get("chat_id")
            message_id = metadata.get("message_id")

            turn_values = {
                "sources": emitted_citations,
                "annotations": round_annotations,
                "reasoning_details": (
                    round_reasoning_details
                    if valves.PERSIST_REASONING_TOKENS in {"next_reply", "conversation"}
                    else []
                ),
            }
            payload: dict[str, Any] = {
                field: turn_values[field]
                for field, _log_label, _notify_label, _key_fn in _TURN_METADATA_FIELDS
                if turn_values[field]
            }
            if (not was_cancelled) and (not handed_back_for_retry) and chat_id and message_id \
                    and payload and Chats is not None and is_linkable_chat(chat_id):
                stored_message: dict[str, Any] = {}
                try:
                    chat_row = await Chats.get_chat_by_id(chat_id)
                    stored_message = (
                        (chat_row.chat or {}).get("history", {}).get("messages", {}).get(message_id, {})
                        if chat_row is not None and isinstance(chat_row.chat, dict)
                        else {}
                    )
                except Exception:
                    self.logger.debug(
                        "Could not read the stored message to merge this turn's fields into "
                        "(chat_id=%s message_id=%s)",
                        chat_id, message_id,
                        exc_info=True,
                    )
                    stored_message = {}
                if not isinstance(stored_message, dict):
                    self.logger.debug(
                        "Stored message for chat_id=%s message_id=%s is not an object; "
                        "writing this turn's own values",
                        chat_id, message_id,
                    )
                    stored_message = {}
                for field, _log_label, _notify_label, key_fn in _TURN_METADATA_FIELDS:
                    if field in payload:
                        payload[field] = _merge_stored_field(
                            stored_message.get(field), payload[field], key_fn=key_fn
                        )
                try:
                    await Chats.upsert_message_to_chat_by_id_and_message_id(
                        chat_id, message_id, payload
                    )
                except Exception as exc:
                    self.logger.warning(
                        "Failed to persist %s for chat_id=%s message_id=%s: %s",
                        _joined_labels([log for _f, log, _n, _k in _TURN_METADATA_FIELDS if _f in payload]),
                        chat_id, message_id, exc,
                        exc_info=True,
                    )
                    await self._pipe._event_emitter_handler._emit_notification(
                        event_emitter,
                        f"Unable to save {_joined_labels([n for _f, _l, n, _k in _TURN_METADATA_FIELDS if _f in payload])} for this response. Output was delivered successfully.",
                        level="warning",
                    )

        _record_outcome()
        if _finalise_cancelled is not None:
            raise _finalise_cancelled
        return assistant_message


    @timed
    async def _run_nonstreaming_loop(
        self,
        body: ResponsesBody,
        valves: Pipe.Valves,
        event_emitter: EventEmitter | None,
        metadata: dict[str, Any] | None = None,
        tools: dict[str, dict[str, Any]] | list[dict[str, Any]] | None = None,
        session: aiohttp.ClientSession | None = None,
        user_id: str = "",
        *,
        endpoint_override: Literal["responses", "chat_completions"] | None = None,
        request_context: Request | None = None,
        user_obj: Any | None = None,
        pipe_identifier: str | None = None,
        fusion_live_enabled: bool = False,
        event_source: AsyncGenerator[dict[str, Any], None] | None = None,
        outcome_sink: dict[str, Any] | None = None,
        retry_handoff: dict[str, Any] | None = None,
    ) -> str | dict[str, Any]:
        """Reuse the streaming loop logic, but honour `stream=False` at the HTTP layer.

        This delegates to `_run_streaming_loop` with a wrapped emitter so incremental
        `chat:message` frames are suppressed while still running all value-add logic
        (tools, citations, usage snapshots, persistence).
        """
        metadata = {} if metadata is None else metadata

        emitter_supplied = event_emitter is not None
        wrapped_emitter = (
            None
            if event_emitter is None
            else _wrap_event_emitter(
                event_emitter,
                suppress_chat_messages=True,
                suppress_completion=False,
            )
        )

        if session is None:
            raise RuntimeError("HTTP session is required for non-streaming")

        answer = await self._run_streaming_loop(
            body,
            valves,
            wrapped_emitter,
            metadata,
            tools or {},
            session=session,
            user_id=user_id,
            endpoint_override=endpoint_override,
            request_context=request_context,
            user_obj=user_obj,
            pipe_identifier=pipe_identifier,
            fusion_live_enabled=fusion_live_enabled,
            event_source=event_source,
            outcome_sink=outcome_sink,
            retry_handoff=retry_handoff,
            emitter_supplied=emitter_supplied,
        )

        from ..requests.orchestrator import _is_api_caller

        notices = getattr(body, "_attachment_notices", None)
        if (
            _is_api_caller(metadata)
            and isinstance(notices, list)
            and notices
            and isinstance(answer, str)
        ):
            return join_answer_and_card(answer, " ".join(notices))
        return answer


    @timed
    async def _cleanup_replayed_reasoning(
        self, body: ResponsesBody, valves: Pipe.Valves, message_id: str | None = None
    ) -> None:
        """Delete once-used reasoning artifacts when retention is limited to the next reply."""
        if valves.PERSIST_REASONING_TOKENS == "conversation":
            return
        refs = getattr(body, "_replayed_reasoning_refs", None)
        if not refs:
            return
        deleted = await self._pipe._artifact_store._delete_artifacts(refs, keep_message_id=message_id)
        if deleted:
            setattr(body, "_replayed_reasoning_refs", [])  # noqa: B010 - undeclared dynamic attribute; setattr keeps pyright quiet


    def _resolve_llm_endpoint(
        self,
        model_id: str,
        *,
        valves: Pipe.Valves,
    ) -> tuple[Literal["responses", "chat_completions"], bool]:
        base_id = ModelFamily.base_model(model_id or "") or (model_id or "")
        undated_id = ModelFamily.undated(base_id) or base_id
        force_chat = _parse_model_patterns(valves.FORCE_CHAT_COMPLETIONS_MODELS)
        force_responses = _parse_model_patterns(valves.FORCE_RESPONSES_MODELS)
        if _matches_any_model_pattern(base_id, force_responses) or _matches_any_model_pattern(
            undated_id, force_responses
        ):
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "LLM endpoint selection: model_id=%s base_id=%s -> responses (FORCE_RESPONSES_MODELS=%s)",
                    model_id,
                    base_id,
                    force_responses,
                )
            return "responses", True
        if _matches_any_model_pattern(base_id, force_chat) or _matches_any_model_pattern(
            undated_id, force_chat
        ):
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "LLM endpoint selection: model_id=%s base_id=%s -> chat_completions (FORCE_CHAT_COMPLETIONS_MODELS=%s)",
                    model_id,
                    base_id,
                    force_chat,
                )
            return "chat_completions", True
        default_endpoint = valves.DEFAULT_LLM_ENDPOINT
        selected = "chat_completions" if default_endpoint == "chat_completions" else "responses"
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                "LLM endpoint selection: model_id=%s base_id=%s default=%s -> %s",
                model_id,
                base_id,
                default_endpoint,
                selected,
            )
        return selected, False


    def _select_llm_endpoint(
        self,
        model_id: str,
        *,
        valves: Pipe.Valves,
    ) -> Literal["responses", "chat_completions"]:
        return self._resolve_llm_endpoint(model_id, valves=valves)[0]


    def _select_llm_endpoint_with_forced(
        self,
        model_id: str,
        *,
        valves: Pipe.Valves,
    ) -> tuple[Literal["responses", "chat_completions"], bool]:
        """Return (endpoint, forced) where forced=True when a FORCE_* valve matched the model id."""
        return self._resolve_llm_endpoint(model_id, valves=valves)


    @staticmethod
    def _looks_like_responses_unsupported(exc: BaseException) -> bool:
        """Heuristic: detect 'model doesn't support /responses' so we can retry via /chat/completions."""
        if getattr(exc, "is_streaming_error", False):
            return False
        if isinstance(exc, OpenRouterAPIError):
            code = exc.openrouter_code
            code_lower = code.strip().lower() if isinstance(code, str) else ""
            if code_lower in {
                "unsupported_endpoint",
                "unsupported_feature",
                "endpoint_not_supported",
                "responses_not_supported",
            }:
                return True

            message_parts = [
                exc.openrouter_message or "",
                exc.upstream_message or "",
                exc.raw_body or "",
                str(exc),
            ]
            haystack = " ".join(part for part in message_parts if part).lower()
            if not _NAMES_THE_RESPONSES_ENDPOINT.search(haystack):
                return False
            if any(token in haystack for token in ("not supported", "unsupported", "does not support")):
                return True
            if any(token in haystack for token in ("chat/completions", "chat completions")):
                return True
            return bool(any(token in haystack for token in ("openai-responses-v1", "xai-responses-v1")))

        haystack_parts: list[str] = [str(exc)]
        for attr in ("openrouter_message", "upstream_message", "raw_body"):
            value = getattr(exc, attr, None)
            if isinstance(value, str) and value:
                haystack_parts.append(value)
        haystack = " ".join(haystack_parts).lower()
        if not _NAMES_THE_RESPONSES_ENDPOINT.search(haystack):
            return False
        if any(token in haystack for token in ("not supported", "unsupported", "does not support")):
            return True
        if any(token in haystack for token in ("chat/completions", "chat completions")):
            return True
        return bool(any(token in haystack for token in ("openai-responses-v1", "xai-responses-v1")))


_NAMES_THE_RESPONSES_ENDPOINT = re.compile(r"\bresponses\b|\bresponse\s+(?:api|endpoint)\b")


def _wrap_event_emitter(
    emitter: EventEmitter | None,
    *,
    suppress_chat_messages: bool = False,
    suppress_completion: bool = False,
    suppress_status: bool = False,
):
    """
    Wrap the given event emitter and optionally suppress specific event types.

    Use-case: reuse the streaming loop for non-stream requests by swallowing
    incremental 'chat:message' frames while allowing status/citation/usage
    events through.
    """
    if emitter is None:
        async def _noop(_event: dict[str, Any]) -> None:
            """Swallow events when no emitter is provided."""
            return

        return _noop

    async def _wrapped(event: dict[str, Any]) -> None:
        """Proxy emitter that suppresses selected event types."""
        etype = (event or {}).get("type")
        if suppress_chat_messages and etype in ("chat:message", "chat:message:delta"):
            return
        if suppress_completion and etype == "chat:completion":
            return
        if suppress_status and etype == "status":
            return
        await emitter(event)

    inner = getattr(emitter, _UNGUARDED_ATTR, None)
    if inner is not None:
        setattr(_wrapped, _UNGUARDED_ATTR, inner)
    return _wrapped
