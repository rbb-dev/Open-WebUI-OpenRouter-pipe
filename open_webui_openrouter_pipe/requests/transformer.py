"""Message transformation to provider input format.

This module handles the transformation of Open WebUI message format
to OpenRouter/OpenAI Responses API input format, including:
- Multimodal content handling (images, files, audio)
- Tool call artifact management
- Persistence layer integration
- Content pruning and filtering
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from ..core.config import (
    _MARKDOWN_IMAGE_RE,
    _NON_REPLAYABLE_TOOL_ARTIFACTS,
    _RAW_REPLAYED_SERVER_TOOLS,
)

# Import status messages
from ..core.errors import RequiredInternalFileError, StatusMessages
from ..core.image_detail import image_detail_or_auto
from ..core.url_scheme import (
    is_cleartext_http_url,
    is_http_or_https_url,
    is_inline_data_url,
    split_base64_data_url,
    url_scheme,
    url_site,
)

# Import utility functions
from ..core.utils import (
    OPEN_WEBUI_TOOL_IMAGES_TEXT,
    PIPE_ONLY_TOOL_ROUND_KEY,
    REASONING_ANCHOR_KEYS,
    REASONING_ANCHOR_SEQ_KEY,
    REASONING_FOLLOWING_ORDINAL_KEY,
    REASONING_FOLLOWING_SERVER_ITEM_KEY,
    REASONING_PRECEDING_ORDINAL_KEY,
    REASONING_TEXT_ORDINAL_KEY,
    SERVER_TOOL_CALL_PREFIX,
    TOOL_ROUND_SKELETON_KEY,
    _extract_plain_text_content,
    _tool_result_failed,
    contains_marker,
    is_picture_output,
    is_server_tool_call_id,
    is_tool_image_handoff,
    opens_a_turn,
    picture_output,
    recorded_tool_text,
    server_tool_arguments,
    server_tool_call_id,
    server_tool_result_text,
    server_tool_status,
    split_text_by_markers,
    split_text_by_phase_markers,
    strip_hidden_marker_lines,
    tool_output_text_and_pictures,
    unretained_tool_result,
)
from ..core.warn_latch import warn_level

# Import Anthropic integration
from ..integrations.anthropic import _maybe_apply_anthropic_prompt_caching

# Import from registry
from ..models.registry import ModelFamily, supports_phase_model

# Import from storage
from ..storage.multimodal import (
    _SNIFF_PREFIX_BYTES,
    _sniff_evidence,
    resolve_download_type,
)
from ..storage.owui_files import (
    InlineFileTooLargeError,
    extract_internal_file_id,
    is_internal_file_url,
    is_temporary_chat,
)

# Import from persistence
from ..storage.persistence import generate_item_id, normalize_persisted_item
from ..tools.tool_schema import (
    _classify_function_call_artifacts,
)

if TYPE_CHECKING:
    from ..pipe import Pipe

# Tool output pruning constants
_TOOL_OUTPUT_PRUNE_MIN_LENGTH = 800
_TOOL_OUTPUT_PRUNE_HEAD_CHARS = 256
_TOOL_OUTPUT_PRUNE_TAIL_CHARS = 128

def _strip_reasoning_anchor_keys(item: dict[str, Any]) -> dict[str, Any]:
    """Return *item* without the internal anchor keys used only for replay
    ordering; they must never reach the provider on the reasoning block."""
    if not any(k in item for k in REASONING_ANCHOR_KEYS):
        return item
    return {k: v for k, v in item.items() if k not in REASONING_ANCHOR_KEYS}


logger = logging.getLogger(__name__)


def _server_round(
    item: dict[str, Any], arguments: str, output: Any, fallback_id: Any = None
) -> list[dict[str, Any]]:
    item_type = str(item.get("type") or "")
    call_id = server_tool_call_id(item.get("id") or fallback_id)
    return [
        {"type": "function_call", "call_id": call_id, "name": item_type.split(":", 1)[1] or item_type,
         "arguments": arguments},
        {"type": "function_call_output", "call_id": call_id, "output": output},
    ]


def _as_replayed(item: dict[str, Any], fallback_id: Any = None) -> list[dict[str, Any]]:
    item_type = item.get("type")
    if not (isinstance(item_type, str) and item_type.startswith("openrouter:")) or (
        item_type in _RAW_REPLAYED_SERVER_TOOLS
    ):
        return [item]
    return _server_round(
        item,
        json.dumps(server_tool_arguments(item), ensure_ascii=False),
        recorded_tool_text(server_tool_result_text(item), server_tool_status(item)),
        fallback_id,
    )


def _without_tool_result(item: dict[str, Any], names: dict[str, str]) -> list[dict[str, Any]] | None:
    item_type = item.get("type")
    call_id = item.get("call_id")
    if item_type == "function_call":
        name = str(item.get("name") or "")
        if isinstance(call_id, str):
            names.setdefault(call_id, name)
        return None if name == "ask_user" else [{**item, "arguments": "{}"}]
    if item_type == "function_call_output":
        if names.get(str(call_id)) == "ask_user":
            return None
        text = tool_output_text_and_pictures(item.get("output"))[0]
        return [{**item, "output": unretained_tool_result(_tool_result_failed(text, item.get("status")))}]
    if isinstance(item_type, str) and item_type.startswith("openrouter:"):
        return _server_round(item, "{}", unretained_tool_result(server_tool_status(item) != "completed"))
    return None


def _tool_name_for_round(messages: list[dict[str, Any]], position: int) -> str:
    target = str(messages[position].get("tool_call_id") or "")
    if not target:
        return ""
    for offset in range(position - 1, -1, -1):
        message = messages[offset]
        if not isinstance(message, dict):
            continue
        if (message.get("role") or "").lower() == "tool":
            continue
        calls = [
            call for call in (message.get("tool_calls") or [])
            if isinstance(call, dict) and str(call.get("id") or "") == target
        ]
        if not calls:
            break
        ordinal = sum(
            1 for prior in messages[offset + 1: position]
            if isinstance(prior, dict) and str(prior.get("tool_call_id") or "") == target
        )
        if ordinal >= len(calls):
            break
        function = calls[ordinal].get("function")
        return str((function or {}).get("name") or "") if isinstance(function, dict) else ""
    return ""


_REUSE_DOWNLOAD_MEMO_MAX_BYTES = 8 * 1024 * 1024
_REUSE_WARN_COOLDOWN_S = 30.0
_reuse_download_memo: OrderedDict[tuple[str, str], tuple[bytes, str]] = OrderedDict()
_warned_image_reuse: dict[str, float] = {}
_warned_oversized_inline: dict[str, float] = {}


class ImageRefusal(NamedTuple):
    reason: str
    cause: str
    severity: Literal["status", "error", "fatal"] = "status"
    subject: str = ""


def _inline_payload_bytes(value: str) -> int:
    if url_scheme(value) != "data":
        return (len(value) * 3) // 4
    split = split_base64_data_url(value)
    if split is not None:
        return (len(split[1]) * 3) // 4
    return len(value.partition(",")[2])


def _inline_media_type(value: str) -> str:
    if url_scheme(value) != "data":
        return "raw base64"
    return value.partition(",")[0][len("data:"):].split(";", 1)[0][:64]


def _reinterleave_region(region: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-interleave reasoning within one assistant turn using call ORDINALS.

    Each reasoning carries the ordinal (0,1,2...) of the call it sits next to:
    a following-ordinal block goes immediately before the Nth ``function_call``;
    a preceding-ordinal block goes immediately after that call's OWN
    ``function_call_output`` (matched by ``call_id`` so a missing/error output for
    one call cannot shift another call's reasoning between a tool_use and its
    tool_result). A call with no output (a failed/orphaned tool) places the block
    immediately after the bare call, where the sanitizer's stub output then lands
    between them -- keeping the reasoning after the result.
    Ordinals are unique within the turn even when call_ids repeat across rounds.
    Reasoning whose ordinal is out of range, or which has no anchor, is left in
    place; nothing is dropped. Anchor keys are stripped from every item.
    """
    movable: list[tuple[int, dict[str, Any], str, int, str | None]] = []
    skeleton: list[dict[str, Any]] = []
    for it in region:
        if isinstance(it, dict) and it.get("type") == "reasoning":
            raw_seq = it.get(REASONING_ANCHOR_SEQ_KEY)
            seq = raw_seq if isinstance(raw_seq, int) else 0
            following = it.get(REASONING_FOLLOWING_ORDINAL_KEY)
            preceding = it.get(REASONING_PRECEDING_ORDINAL_KEY)
            server_item = it.get(REASONING_FOLLOWING_SERVER_ITEM_KEY)
            server_item = server_item if isinstance(server_item, str) and server_item else None
            stripped = _strip_reasoning_anchor_keys(it)
            text_ordinal = it.get(REASONING_TEXT_ORDINAL_KEY)
            if isinstance(following, int):
                movable.append((seq, stripped, "before", following, server_item))
            elif isinstance(preceding, int):
                movable.append((seq, stripped, "after", preceding, server_item))
            elif isinstance(text_ordinal, int):
                movable.append((seq, stripped, "text", text_ordinal, server_item))
            else:
                skeleton.append(stripped)
        elif isinstance(it, dict):
            skeleton.append(_strip_reasoning_anchor_keys(it))
        else:
            skeleton.append(it)

    if not movable:
        return skeleton

    fc_items = [
        (i, e) for i, e in enumerate(skeleton)
        if isinstance(e, dict) and e.get("type") == "function_call" and not is_server_tool_call_id(e.get("call_id"))
    ]
    server_positions: dict[str, int] = {}
    for i, e in enumerate(skeleton):
        if not isinstance(e, dict):
            continue
        if e.get("type") == "function_call" and is_server_tool_call_id(e.get("call_id")):
            server_positions.setdefault(str(e["call_id"])[len(SERVER_TOOL_CALL_PREFIX):], i)
        elif str(e.get("type") or "").startswith("openrouter:") and isinstance(e.get("id"), str):
            server_positions.setdefault(e["id"], i)
    fco_items = [
        (i, e) for i, e in enumerate(skeleton)
        if isinstance(e, dict) and e.get("type") == "function_call_output"
    ]
    output_index_for_call: dict[int, int] = {}
    remaining_outputs = list(fco_items)
    for call_ordinal, (_, call_item) in enumerate(fc_items):
        call_id = call_item.get("call_id")
        for j, (fco_idx, fco_item) in enumerate(remaining_outputs):
            if fco_item.get("call_id") == call_id:
                output_index_for_call[call_ordinal] = fco_idx
                remaining_outputs.pop(j)
                break

    msg_items = [
        (i, e) for i, e in enumerate(skeleton)
        if isinstance(e, dict) and e.get("type") == "message" and e.get("role") == "assistant"
    ]

    inserts_before: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    inserts_after: dict[int, list[tuple[int, dict[str, Any]]]] = {}
    for seq, item, mode, ordinal, server_item in sorted(movable, key=lambda a: a[0]):
        pos: int | None = server_positions.get(server_item) if server_item else None
        if pos is not None:
            inserts_before.setdefault(pos, []).append((seq, item))
            continue
        if mode == "before":
            if 0 <= ordinal < len(fc_items):
                pos = fc_items[ordinal][0]
            bucket = inserts_before
        elif mode == "text":
            if 0 <= ordinal < len(msg_items):
                pos = msg_items[ordinal][0]
            bucket = inserts_before
        else:
            if ordinal in output_index_for_call:
                pos = output_index_for_call[ordinal]
                after = skeleton[pos + 1] if pos + 1 < len(skeleton) else None
                if (
                    isinstance(after, dict) and after.get("type") == "message" and after.get("role") == "user"
                    and not opens_a_turn(skeleton, pos + 1)
                ):
                    pos += 1
            elif 0 <= ordinal < len(fc_items):
                pos = fc_items[ordinal][0]
            bucket = inserts_after
        if pos is None:
            skeleton.append(item)
            continue
        bucket.setdefault(pos, []).append((seq, item))

    rebuilt: list[dict[str, Any]] = []
    for i, entry in enumerate(skeleton):
        for _, ins in sorted(inserts_before.get(i, []), key=lambda a: a[0]):
            rebuilt.append(ins)
        rebuilt.append(entry)
        for _, ins in sorted(inserts_after.get(i, []), key=lambda a: a[0]):
            rebuilt.append(ins)
    return rebuilt


def _reinterleave_reasoning_by_anchor(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Re-interleave reasoning to its true generated position, scoped per assistant
    turn. Turns are delimited by user messages so a reasoning anchor only binds to
    a tool call within its own turn -- tool ``call_id`` values are not unique across
    turns (the chat-completions adapter assigns index-based ids that repeat).
    """
    out: list[dict[str, Any]] = []
    region: list[dict[str, Any]] = []
    for index, it in enumerate(items):
        if opens_a_turn(items, index):
            out.extend(_reinterleave_region(_one_copy_per_round(region)))
            region = []
            out.append(_strip_reasoning_anchor_keys(it))
        else:
            region.append(it)
    out.extend(_reinterleave_region(_one_copy_per_round(region)))
    return out


_PIPE_STORAGE_KEY = "_anchor_from_pipe_storage"
_TRANSPORT_ONLY_KEYS = (_PIPE_STORAGE_KEY, PIPE_ONLY_TOOL_ROUND_KEY, TOOL_ROUND_SKELETON_KEY)


def _from_pipe_storage(item: dict[str, Any]) -> dict[str, Any]:
    kind = str(item.get("type") or "")
    if kind in ("function_call", "function_call_output") or kind.startswith("openrouter:"):
        return {**item, _PIPE_STORAGE_KEY: True}
    return item


def _one_copy_per_round(region: list[Any]) -> list[Any]:
    supplied = {
        it.get("call_id")
        for it in region
        if isinstance(it, dict) and it.get("type") == "function_call" and not it.get(_PIPE_STORAGE_KEY)
    }
    raw_server_ids = {
        it["id"]
        for it in region
        if isinstance(it, dict)
        and it.get(_PIPE_STORAGE_KEY)
        and str(it.get("type") or "").startswith("openrouter:")
        and isinstance(it.get("id"), str)
    }
    shadowed = raw_server_ids | {server_tool_call_id(raw_id) for raw_id in raw_server_ids}
    kept: list[Any] = []
    pictures: list[str] = []
    for it in region:
        if pictures and not (isinstance(it, dict) and it.get("type") == "function_call_output"):
            kept.append(_tool_images_message(pictures))
            pictures = []
        replayed_pictures = (
            isinstance(it, dict)
            and bool(it.get(_PIPE_STORAGE_KEY))
            and it.get("type") == "function_call_output"
            and is_picture_output(it.get("output"))
        )
        if isinstance(it, dict) and it.get("type") in ("function_call", "function_call_output"):
            from_pipe = bool(it.get(_PIPE_STORAGE_KEY))
            pipe_only = bool(it.get(PIPE_ONLY_TOOL_ROUND_KEY) or it.get(TOOL_ROUND_SKELETON_KEY))
            if from_pipe and not pipe_only and it.get("call_id") in supplied:
                continue
            if not from_pipe and it.get("call_id") in shadowed:
                continue
        if isinstance(it, dict) and any(key in it for key in _TRANSPORT_ONLY_KEYS):
            it = {key: value for key, value in it.items() if key not in _TRANSPORT_ONLY_KEYS}
            if it.get("type") == "function_call_output":
                it.pop("status", None)
        if replayed_pictures:
            text, shown = tool_output_text_and_pictures(it["output"])
            it = {**it, "output": text}
            pictures.extend(shown)
        kept.append(it)
    if pictures:
        kept.append(_tool_images_message(pictures))
    return kept


def _tool_images_message(pictures: list[str]) -> dict[str, Any]:
    return {"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": OPEN_WEBUI_TOOL_IMAGES_TEXT},
        *({"type": "input_image", "image_url": url, "detail": "auto"} for url in pictures),
    ]}


async def _memo_hit_is_still_permitted(
    pipe: Pipe, memo_key: Any, url: str
) -> bool:
    if not await pipe._multimodal_handler._is_safe_url(url):
        _reuse_download_memo.pop(memo_key, None)
        return False
    return True


async def transform_messages_to_input(
    pipe: Pipe,
    messages: list[dict[str, Any]],
    chat_id: str | None = None,
    openwebui_model_id: str | None = None,
    artifact_loader: Callable[[str | None, str | None, list[str]], Awaitable[dict[str, dict[str, Any]]]] | None = None,
    pruning_turns: int = 0,
    replayed_reasoning_refs: list[tuple[str, str]] | None = None,
    user_obj: Any | None = None,
    event_emitter: Callable | None = None,
    *,
    model_id: str | None = None,
    valves: Pipe.Valves | None = None,
    capability_model_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Build an OpenAI Responses-API `input` array from Open WebUI-style messages.

    Parameters `chat_id` and `openwebui_model_id` are optional. When both are
    supplied and the messages contain empty-link encoded item references, the
    function fetches persisted items from the database and injects them in the
    correct order. When either parameter is missing, the messages are simply
    converted without attempting to fetch persisted items.

    When provided, `artifact_loader` is awaited with `(chat_id, message_id, ulids)`
    for each assistant message so database-backed artifacts can be replayed.
    When `replayed_reasoning_refs` is supplied, the function appends each
    `(chat_id, artifact_id)` pair for reasoning items so the caller can clean
    them up after they have been replayed once.

    Returns
    -------
    List[dict] : The fully-formed `input` list for the OpenAI Responses API.
    """

    active_valves = valves or pipe.valves
    image_limit = active_valves.MAX_INPUT_IMAGES_PER_REQUEST
    selection_mode = active_valves.IMAGE_INPUT_SELECTION
    image_reuse_turns = active_valves.IMAGE_REUSE_MAX_TURNS
    chunk_size = active_valves.IMAGE_UPLOAD_CHUNK_BYTES
    max_inline_bytes = active_valves.BASE64_MAX_SIZE_MB * 1024 * 1024
    video_max_size_mb = active_valves.VIDEO_MAX_SIZE_MB
    target_model_id = model_id or openwebui_model_id or ""
    vision_lookup_id = capability_model_id or target_model_id
    phase_lookup_id = capability_model_id or target_model_id
    phase_supported = supports_phase_model(phase_lookup_id)
    if vision_lookup_id:
        vision_supported = ModelFamily.supports("vision", vision_lookup_id)
    else:
        vision_supported = True

    openai_input: list[dict] = []
    last_image_blocks: list[dict[str, Any]] = []
    last_image_turn: int | None = None

    def _message_identifier(entry: dict[str, Any]) -> str | None:
        """Return the most specific identifier available on ``entry``."""
        for key in ("id", "_id", "message_id"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None

    def _compute_turn_indices() -> tuple[list[int | None], int]:
        """Label each message with a turn index and return the total count."""
        indices: list[int | None] = []
        current_turn = -1
        max_turn = -1
        last_dialog_role: str | None = None

        for position, msg in enumerate(messages):
            role = (msg.get("role") or "").lower()
            turn_idx: int | None = None

            if role == "user" and position and is_tool_image_handoff(messages[position - 1], msg):
                turn_idx = current_turn if current_turn >= 0 else None
            elif role == "user":
                if last_dialog_role != "user":
                    current_turn += 1
                turn_idx = current_turn
                last_dialog_role = "user"
            elif role == "assistant":
                current_turn = max(current_turn, 0)
                turn_idx = current_turn
                last_dialog_role = "assistant"
            else:
                turn_idx = current_turn if current_turn >= 0 else None

            if turn_idx is not None and turn_idx > max_turn:
                max_turn = turn_idx
            indices.append(turn_idx)

        total_turns = max_turn + 1 if max_turn >= 0 else 0
        return indices, total_turns

    def _markdown_images_from_text(text: str) -> list[str]:
        """Extract inline Markdown image URLs from a text block."""
        if not isinstance(text, str):
            return []
        return [
            match.group("url").strip()
            for match in _MARKDOWN_IMAGE_RE.finditer(text)
            if match.group("url").strip()
        ]

    def _is_old_turn(turn_index: int | None, *, threshold: int | None) -> bool:
        """Return True when a message turn falls outside the retention window."""
        return (
            threshold is not None
            and turn_index is not None
            and turn_index < threshold
        )

    def _prune_tool_output(
        item: dict[str, Any],
        *,
        marker: str | None,
        turn_index: int | None,
        retention_turns: int,
    ) -> bool:
        """Shorten oversized tool output strings while leaving markers intact."""
        if item.get("type") != "function_call_output":
            return False

        output_value = item.get("output")
        if output_value is None:
            return False
        if is_picture_output(output_value):
            text, pictures = tool_output_text_and_pictures(output_value)
            probe = {"type": "function_call_output", "call_id": item.get("call_id"), "output": text}
            if not _prune_tool_output(probe, marker=marker, turn_index=turn_index, retention_turns=retention_turns):
                return False
            item["output"] = picture_output(probe["output"], pictures)
            return True
        if not isinstance(output_value, str):
            try:
                output_text = json.dumps(output_value, ensure_ascii=False)
            except (TypeError, ValueError):
                output_text = str(output_value)
        else:
            output_text = output_value

        if len(output_text) < _TOOL_OUTPUT_PRUNE_MIN_LENGTH:
            return False

        head = output_text[:_TOOL_OUTPUT_PRUNE_HEAD_CHARS].rstrip()
        tail = output_text[-_TOOL_OUTPUT_PRUNE_TAIL_CHARS:].lstrip()
        removed_chars = len(output_text) - len(head) - len(tail)
        if removed_chars <= 0:
            return False

        ellipsis = "..."
        turn_label = (
            f"{retention_turns} turn" if retention_turns == 1 else f"{retention_turns} turns"
        )
        note = (
            f"{ellipsis}\n"
            f"[tool output pruned: removed {removed_chars} char"
            f"{'' if removed_chars == 1 else 's'}, older than {turn_label}]\n"
            f"{ellipsis}"
        )
        item["output"] = "\n".join(
            part for part in (head, note, tail) if part
        )
        logger.debug("Pruned tool output (marker=%s, call_id=%s, turn=%s, removed_chars=%d, retention=%d)", marker, item.get("call_id"), turn_index, removed_chars, retention_turns)
        return True

    turn_indices, total_turns = _compute_turn_indices()
    current_turn_people = [
        position
        for position, message in enumerate(messages)
        if (message.get("role") or "").lower() == "user"
        and turn_indices[position] is not None
        and turn_indices[position] == total_turns - 1
        and not (position and is_tool_image_handoff(messages[position - 1], message))
    ]
    tool_handoff_positions = [
        position
        for position, message in enumerate(messages)
        if (message.get("role") or "").lower() == "user"
        and position
        and is_tool_image_handoff(messages[position - 1], message)
    ]
    last_tool_handoff_index = tool_handoff_positions[-1] if tool_handoff_positions else -1
    person_images_this_turn = False
    temporary_chat = is_temporary_chat(chat_id)
    request_memo: dict[tuple[str, str], tuple[bytes, str]] = {}
    tool_names_by_call_id: dict[str, str] = {
        str(call.get("id")): str((call.get("function") or {}).get("name") or "")
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("tool_calls"), list)
        for call in message["tool_calls"]
        if isinstance(call, dict) and call.get("id") and isinstance(call.get("function"), dict)
    }

    def _withheld(turn_index: int | None) -> bool:
        return not active_valves.PERSIST_TOOL_RESULTS and turn_index is not None and turn_index < total_turns - 1

    prune_before_turn: int | None = None
    if pruning_turns > 0 and total_turns > pruning_turns:
        prune_before_turn = total_turns - pruning_turns

    def _sanitize_free_text(text: str) -> str:
        """Strip hidden transport markers from non-assistant free text."""
        return strip_hidden_marker_lines(text)

    missing_artifact_markers: list[str] = []
    for idx, msg in enumerate(messages):
        raw_role = msg.get("role")
        role = (raw_role or "").lower()
        raw_content = msg.get("content", "")
        msg_id = msg.get("message_id") or _message_identifier(msg)
        msg_turn_index = turn_indices[idx]
        raw_tool_calls = msg.get("tool_calls")
        msg_tool_calls: list[dict[str, Any]] = (
            list(raw_tool_calls)
            if isinstance(raw_tool_calls, list) and raw_tool_calls
            else []
        )

        if role in {"system", "developer"}:
            blocks: list[dict[str, Any]] = []

            if isinstance(raw_content, str):
                cleaned = _sanitize_free_text(raw_content)
                if cleaned or raw_content == "":
                    blocks.append({"type": "input_text", "text": cleaned})
            elif isinstance(raw_content, list):
                for entry in raw_content:
                    if isinstance(entry, str):
                        cleaned = _sanitize_free_text(entry)
                        if cleaned:
                            blocks.append({"type": "input_text", "text": cleaned})
                        continue
                    if isinstance(entry, dict):
                        text_val = entry.get("text")
                        if not isinstance(text_val, str):
                            text_val = entry.get("content")
                        if isinstance(text_val, str):
                            cleaned = _sanitize_free_text(text_val)
                            if cleaned:
                                blocks.append({"type": "input_text", "text": cleaned})
            elif isinstance(raw_content, dict):
                text_val = raw_content.get("text")
                if not isinstance(text_val, str):
                    text_val = raw_content.get("content")
                if isinstance(text_val, str):
                    cleaned = _sanitize_free_text(text_val)
                    if cleaned:
                        blocks.append({"type": "input_text", "text": cleaned})

            if blocks:
                openai_input.append(
                    {
                        "type": "message",
                        "role": role,
                        "content": blocks,
                    }
                )
            continue

        if role == "tool":
            call_id = msg.get("tool_call_id") or msg.get("id") or msg.get("call_id")
            call_id = call_id.strip() if isinstance(call_id, str) else ""
            if not call_id:
                continue

            tool_content = raw_content
            if tool_content is None:
                tool_content_text = ""
            elif isinstance(tool_content, str):
                tool_content_text = tool_content
            else:
                try:
                    tool_content_text = json.dumps(tool_content, ensure_ascii=False)
                except (TypeError, ValueError):
                    tool_content_text = str(tool_content)

            if _withheld(msg_turn_index) and _tool_name_for_round(messages, idx) != "ask_user":
                tool_content_text = unretained_tool_result(_tool_result_failed(tool_content_text))

            tool_item: dict[str, Any] = {
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_content_text,
            }
            if pruning_turns > 0 and _is_old_turn(msg_turn_index, threshold=prune_before_turn):
                _prune_tool_output(tool_item, marker=None, turn_index=msg_turn_index, retention_turns=pruning_turns)
            openai_input.append(tool_item)
            continue

        if role == "user":
            tool_images = bool(idx) and is_tool_image_handoff(messages[idx - 1], msg)
            if tool_images:
                last_image_blocks, last_image_turn = [], None
            if tool_images and _withheld(msg_turn_index):
                continue
            content_blocks = msg.get("content") or []
            if isinstance(content_blocks, str):
                cleaned = _sanitize_free_text(content_blocks)
                content_blocks = [{"type": "text", "text": cleaned}] if cleaned else []
            elif isinstance(content_blocks, dict):
                text_val = content_blocks.get("text")
                if not isinstance(text_val, str):
                    text_val = content_blocks.get("content")
                cleaned = _sanitize_free_text(text_val) if isinstance(text_val, str) else ""
                content_blocks = [{"type": "text", "text": cleaned}] if cleaned else []
            elif not isinstance(content_blocks, list):
                pipe.logger.warning(
                    "Ignoring user message %s: content is %s, expected a string, list or dict",
                    msg_id,
                    type(content_blocks).__name__,
                )
                content_blocks = []

            def _image_subject(source: str) -> str:
                return url_site(source) if is_http_or_https_url(source) else _inline_media_type(source)

            def _refuse(reason: str, cause: str, *, subject: str) -> ImageRefusal:
                return ImageRefusal(reason, cause, subject=subject)

            async def _to_input_image(
                block: dict,
                *,
                mode: Literal["reuse", "inline"] = "reuse" if tool_images else "inline",
            ) -> dict[str, Any] | ImageRefusal | None:
                """Convert Open WebUI image block into Responses format.

                Supported Image Formats (per OpenRouter docs):
                    - image/png
                    - image/jpeg
                    - image/webp
                    - image/gif

                Args:
                    block: Content block from Open WebUI message

                """
                try:
                    image_payload = block.get("image_url")
                    detail: str | None = None
                    url: str = ""

                    if isinstance(image_payload, dict):
                        url = image_payload.get("url", "")
                        detail = image_payload.get("detail")
                    elif isinstance(image_payload, str):
                        url = image_payload
                        block_detail = block.get("detail")
                        if isinstance(block_detail, str):
                            detail = block_detail

                    if not url:
                        return None

                    if (
                        is_cleartext_http_url(url)
                        and not is_internal_file_url(url)
                        and not pipe._multimodal_handler._is_insecure_http_allowed(url)
                    ):
                            return ImageRefusal(
                                "served over plain HTTP, which is blocked by security policy; "
                                "set ALLOW_INSECURE_HTTP and list the host in "
                                "ALLOW_INSECURE_HTTP_HOSTS to permit it",
                                "insecure_http",
                                severity="error",
                                subject=url_site(url),
                            )

                    if is_inline_data_url(url):
                        url = f"data:{url.partition(':')[2]}"
                        try:
                            split = split_base64_data_url(url)
                            if split is None:
                                return ImageRefusal(
                                    "a data URL that is not base64-encoded, which "
                                    "OpenRouter does not accept",
                                    "unencoded_inline",
                                    subject=_inline_media_type(url),
                                )
                            parsed = pipe._multimodal_handler._parse_data_url(url)
                            if not parsed:
                                oversized = (len(split[1]) * 3) // 4 > max_inline_bytes
                                return ImageRefusal(
                                    f"larger than the {max_inline_bytes}-byte inline limit"
                                    if oversized
                                    else "not decodable as base64",
                                    "oversized_inline" if oversized else "undecodable_inline",
                                    subject=_inline_media_type(url),
                                )
                        except Exception as exc:
                            pipe.logger.exception("Failed to process base64 image")
                            await pipe._ensure_error_formatter()._emit_error(
                                event_emitter,
                                f"Failed to process base64 image: {exc}",
                                show_error_message=False
                            )
                            return ImageRefusal(
                                f"could not be processed: {exc}", "base64_processing_error", subject=url[:64]
                            )

                    elif is_http_or_https_url(url) and not is_internal_file_url(url):
                        memo_key = (chat_id, url) if chat_id else None
                        remembered = (
                            request_memo.get(memo_key)
                            if mode == "reuse" and memo_key is not None
                            else None
                        )
                        if remembered is None and mode == "reuse" and not temporary_chat:
                            remembered = (
                                _reuse_download_memo.get(memo_key)
                                if memo_key is not None
                                else None
                            )
                        if remembered is not None and not await (
                            _memo_hit_is_still_permitted(pipe, memo_key, url)
                        ):
                            remembered = None
                        try:
                            downloaded = (
                                {"data": remembered[0], "mime_type": remembered[1]}
                                if remembered is not None
                                else await pipe._multimodal_handler._download_remote_url(url)
                            )
                        except Exception:
                            pipe.logger.exception("Failed to download remote image %s", url_site(url))
                            downloaded = None
                        if downloaded and not downloaded.get("data"):
                            downloaded = None
                        if not downloaded and not await pipe._multimodal_handler._is_safe_url(url):
                            return _refuse(
                                "could not be fetched, so it was not sent",
                                "remote_unfetched",
                                subject=url_site(url),
                            )
                        if downloaded:
                            oversized = len(downloaded["data"]) > max_inline_bytes
                            if oversized:
                                return _refuse(
                                    f"{len(downloaded['data'])} bytes, over the "
                                    f"{max_inline_bytes}-byte limit, so it was not sent",
                                    "oversized_remote",
                                    subject=url_site(url),
                                )
                            if mode == "reuse" and memo_key is not None:
                                request_memo[memo_key] = (
                                    downloaded["data"],
                                    downloaded.get("mime_type") or "",
                                )
                            if (
                                mode == "reuse"
                                and not temporary_chat
                                and remembered is None
                                and memo_key is not None
                                and len(downloaded["data"])
                                <= _REUSE_DOWNLOAD_MEMO_MAX_BYTES
                            ):
                                held = sum(
                                    len(data) for data, _ in _reuse_download_memo.values()
                                )
                                while (
                                    _reuse_download_memo
                                    and held + len(downloaded["data"])
                                    > _REUSE_DOWNLOAD_MEMO_MAX_BYTES
                                ):
                                    _, evicted = _reuse_download_memo.popitem(last=False)
                                    held -= len(evicted[0])
                                _reuse_download_memo[memo_key] = (
                                    downloaded["data"],
                                    downloaded.get("mime_type") or "",
                                )
                            url = (
                                f"data:{downloaded.get('mime_type') or ''};base64,"
                                + base64.b64encode(downloaded["data"]).decode("ascii")
                            )
                    owui_file_id = extract_internal_file_id(url) if is_internal_file_url(url) else None

                    if owui_file_id:
                        try:
                            inlined = await pipe._file_gateway.inline_owui_file_id(
                                owui_file_id,
                                chunk_size=chunk_size,
                                max_bytes=max_inline_bytes,
                                user=user_obj,
                            )
                        except InlineFileTooLargeError:
                            return ImageRefusal(
                                f"larger than the {max_inline_bytes}-byte inline limit",
                                "oversized_inline",
                                subject=owui_file_id,
                            )
                        if not inlined:
                            return ImageRefusal(
                                "no longer available in Open WebUI storage",
                                "owui_file_unavailable",
                                severity="fatal",
                                subject=owui_file_id,
                            )
                        url = inlined.data_url

                    if mode == "reuse":
                        split = split_base64_data_url(url)
                        head, body = split if split is not None else ("", "")
                        if not body:
                            if is_http_or_https_url(url):
                                return _refuse(
                                    "could not be fetched, so it was not sent",
                                    "remote_unfetched",
                                    subject=_image_subject(url),
                                )
                            return ImageRefusal(
                                "could not be fetched, so its type could not be established",
                                "reuse_unfetched",
                                subject=_image_subject(url),
                            )
                        declared = head[len("data:") :].split(";", 1)[0].strip().lower()
                        try:
                            sniffed = base64.b64decode(
                                "".join(body.split())[: (_SNIFF_PREFIX_BYTES + 2) // 3 * 4]
                            )
                        except (binascii.Error, ValueError):
                            sniffed = b""
                        resolved = resolve_download_type(declared, _sniff_evidence(sniffed))
                        if not resolved.startswith("image/"):
                            return ImageRefusal(
                                "not identifiable as an image",
                                "reuse_untyped",
                            )
                        if resolved != declared:
                            url = f"data:{resolved};base64,{body}"

                    result: dict[str, Any] = {"type": "input_image", "image_url": url}
                    if url:
                        result["detail"] = image_detail_or_auto(detail)

                    return result

                except RequiredInternalFileError:
                    raise
                except Exception:
                    pipe.logger.exception("Error in _to_input_image")
                    _raw = block.get("image_url")
                    _source = _raw.get("url", "") if isinstance(_raw, dict) else _raw
                    return ImageRefusal(
                        "could not be processed",
                        "processing_error",
                        subject=_image_subject(_source if isinstance(_source, str) else ""),
                    )

            async def _to_input_file(block: dict) -> dict | ImageRefusal:
                """Convert Open WebUI file blocks into Responses API format.

                Responses API File Input Fields (per OpenAPI spec):
                    - type: "input_file" (required)
                    - file_id: string | null (optional)
                    - file_data: string (optional) - base64 or data URL
                    - filename: string (optional) - for model context
                    - file_url: string (optional) - URL to file

                Args:
                    block: Content block from Open WebUI message

                Returns:
                    Responses API input_file block with all available fields

                Note:
                    All errors are caught and logged with status emissions.
                """
                try:
                    result = {"type": "input_file"}

                    nested_file = block.get("file")
                    source = nested_file if isinstance(nested_file, dict) else block

                    file_id = source.get("file_id")
                    file_data = source.get("file_data")
                    filename = source.get("filename")
                    file_url = source.get("file_url")

                    if isinstance(file_url, str) and file_url.strip() and is_internal_file_url(file_url.strip()):
                        extracted = extract_internal_file_id(file_url.strip())
                        if extracted:
                            file_id = extracted
                            file_url = None
                    if isinstance(file_data, str) and file_data.strip() and is_internal_file_url(file_data.strip()):
                        extracted = extract_internal_file_id(file_data.strip())
                        if extracted:
                            file_id = extracted
                            file_data = None

                    if (
                        isinstance(file_data, str)
                        and is_cleartext_http_url(file_data)
                        and not is_internal_file_url(file_data)
                        and not pipe._multimodal_handler._is_insecure_http_allowed(file_data)
                    ):
                        pipe.logger.error("Blocked insecure HTTP file_data URL by default: %s", file_data)
                        await pipe._ensure_error_formatter()._emit_error(
                            event_emitter,
                            "File URL blocked by security policy (HTTP disabled by default). "
                            "Enable ALLOW_INSECURE_HTTP + ALLOW_INSECURE_HTTP_HOSTS to allow specific hosts.",
                            show_error_message=True,
                        )
                        if file_id:
                            file_data = None
                        else:
                            return ImageRefusal(
                                "served over plain HTTP, which is blocked by security policy; "
                                "set ALLOW_INSECURE_HTTP and list the host in "
                                "ALLOW_INSECURE_HTTP_HOSTS to permit it",
                                "insecure_http_file",
                                subject="file_data",
                            )

                    if (
                        isinstance(file_url, str)
                        and is_cleartext_http_url(file_url)
                        and not is_internal_file_url(file_url)
                        and not pipe._multimodal_handler._is_insecure_http_allowed(file_url)
                    ):
                        pipe.logger.error("Blocked insecure HTTP file_url by default: %s", file_url)
                        await pipe._ensure_error_formatter()._emit_error(
                            event_emitter,
                            "File URL blocked by security policy (HTTP disabled by default). "
                            "Enable ALLOW_INSECURE_HTTP + ALLOW_INSECURE_HTTP_HOSTS to allow specific hosts.",
                            show_error_message=True,
                        )
                        if file_id:
                            file_url = None
                        else:
                            return ImageRefusal(
                                "served over plain HTTP, which is blocked by security policy; "
                                "set ALLOW_INSECURE_HTTP and list the host in "
                                "ALLOW_INSECURE_HTTP_HOSTS to permit it",
                                "insecure_http_file",
                                subject="file_url",
                            )

                    oversized = {
                        name: value
                        for name, value in (("file_data", file_data), ("file_url", file_url))
                        if isinstance(value, str)
                        and value
                        and (url_scheme(value) == "data" or (name == "file_data" and not url_scheme(value)))
                        and _inline_payload_bytes(value) > max_inline_bytes
                    }
                    if oversized and not file_id:
                        return ImageRefusal(
                            f"larger than the {max_inline_bytes}-byte inline limit",
                            "oversized_inline_file",
                            subject=_inline_media_type(next(iter(oversized.values()))),
                        )
                    if "file_data" in oversized:
                        file_data = None
                    if "file_url" in oversized:
                        file_url = None

                    if file_id:
                        result["file_id"] = file_id
                    if file_data:
                        result["file_data"] = file_data
                    if filename:
                        result["filename"] = filename
                    if file_url:
                        result["file_url"] = file_url

                    return result

                except Exception as exc:
                    pipe.logger.exception("Error in _to_input_file")
                    await pipe._ensure_error_formatter()._emit_error(
                        event_emitter,
                        f"File processing error: {exc}",
                        show_error_message=False
                    )
                    return {"type": "input_file"}

            async def _to_input_audio(block: dict) -> dict | None:
                """Convert Open WebUI audio blocks into Responses API format.

                Handles audio content blocks, transforming various input formats into
                the Responses API audio input format.

                Responses API Audio Input Format (per OpenAPI spec):
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "<base64_audio_data>",
                            "format": "mp3" | "wav"
                        }
                    }

                OpenRouter Audio Requirements (per documentation):
                    - Audio must be base64-encoded (URLs NOT supported)
                    - Supported formats: wav, mp3 only
                    - See: https://openrouter.ai/docs/guides/overview/multimodal/audio

                Input Formats Handled:
                    1. Chat Completions: {"type": "input_audio", "input_audio": "<base64>"}
                    2. Tool output: {"type": "audio", "mimeType": "audio/mp3", "data": "<base64>"}
                    3. Already correct: {"type": "input_audio", "input_audio": {"data": "...", "format": "..."}}

                Args:
                    block: Content block from Open WebUI message

                MIME Type to Format Mapping:
                    - audio/mpeg, audio/mp3 -> "mp3"
                    - audio/wav, audio/wave, audio/x-wav -> "wav"
                    - audio/flac, audio/x-flac -> "flac"
                    - audio/mp4, audio/m4a, audio/x-m4a -> "m4a"
                    - audio/ogg -> "ogg"; audio/aiff, audio/x-aiff -> "aiff"; audio/aac -> "aac"
                    - An explicit format already in the supported set (mp3, wav,
                      flac, m4a, ogg, aiff, aac, pcm16, pcm24) is preserved as-is.
                    - Unknown types default to "mp3"

                Note:
                    All errors are caught and logged with status emissions.
                    Failed processing returns minimal valid block rather than crashing.
                """
                format_map = {
                    "audio/mpeg": "mp3",
                    "audio/mp3": "mp3",
                    "audio/wav": "wav",
                    "audio/wave": "wav",
                    "audio/x-wav": "wav",
                    "audio/flac": "flac",
                    "audio/x-flac": "flac",
                    "audio/mp4": "m4a",
                    "audio/m4a": "m4a",
                    "audio/x-m4a": "m4a",
                    "audio/ogg": "ogg",
                    "audio/aiff": "aiff",
                    "audio/x-aiff": "aiff",
                    "audio/aac": "aac",
                    "audio/webm": "webm",
                    "audio/x-webm": "webm",
                }
                supported_formats = {
                    "mp3", "wav", "flac", "m4a", "ogg", "aiff", "aac", "pcm16", "pcm24",
                    "webm",
                }

                def _map_format(mime: str | None) -> str:
                    if not isinstance(mime, str):
                        return "mp3"
                    return format_map.get(mime.lower(), "mp3")

                def _empty_audio_block() -> dict[str, Any]:
                    return {
                        "type": "input_audio",
                        "input_audio": {
                            "data": "",
                            "format": "mp3",
                        },
                    }

                async def _refuse_oversized_inline(estimate: int) -> None:
                    pipe.logger.warning(
                        "Audio payload rejected: ~%.1fMB is over the %dMB inline limit.",
                        estimate / (1024 * 1024),
                        pipe.valves.BASE64_MAX_SIZE_MB,
                    )
                    await pipe._ensure_error_formatter()._emit_error(
                        event_emitter,
                        f"Audio input is larger than the {max_inline_bytes}-byte inline limit and was not sent.",
                        show_error_message=True,
                    )

                def _normalize_base64(data: str) -> str | None:
                    if not data:
                        return None
                    cleaned = "".join(data.split())
                    if not cleaned:
                        return None
                    try:
                        base64.b64decode(cleaned, validate=True)
                    except (binascii.Error, ValueError):
                        return None
                    return cleaned

                def _resolved_mime_hint(payload: dict[str, Any] | None = None) -> str | None:
                    candidates: list[Any] = []
                    if isinstance(payload, dict):
                        candidates.extend(
                            [
                                payload.get("mimeType"),
                                payload.get("mime_type"),
                            ]
                        )
                    candidates.extend(
                        [
                            block.get("mimeType"),
                            block.get("mime_type"),
                            block.get("contentType"),
                            block.get("content_type"),
                        ]
                    )
                    for value in candidates:
                        if isinstance(value, str):
                            stripped = value.strip()
                            if stripped:
                                return stripped
                    return None

                def _normalize_format(
                    explicit_format: str | None,
                    mime_hint: str | None,
                ) -> str:
                    if isinstance(explicit_format, str):
                        normalized = explicit_format.strip().lower()
                        if normalized in supported_formats:
                            return normalized
                    return _map_format(mime_hint)

                def _build_audio_block(data: str, audio_format: str) -> dict[str, Any]:
                    return {
                        "type": "input_audio",
                        "input_audio": {
                            "data": data,
                            "format": audio_format,
                        },
                    }

                try:
                    audio_payload = block.get("input_audio") or block.get("data") or block.get("blob")

                    if isinstance(audio_payload, dict) and "data" in audio_payload and "format" in audio_payload:
                        _data = audio_payload.get("data", "")
                        if isinstance(_data, str) and _inline_payload_bytes(_data) > max_inline_bytes:
                            return await _refuse_oversized_inline(_inline_payload_bytes(_data))
                        cleaned = _normalize_base64(_data)
                        if not cleaned:
                            pipe.logger.warning("Audio payload rejected: invalid base64 data.")
                            await pipe._ensure_error_formatter()._emit_error(
                                event_emitter,
                                "Audio input was not valid base64.",
                                show_error_message=False,
                            )
                            return _empty_audio_block()
                        audio_format = _normalize_format(audio_payload.get("format"), _resolved_mime_hint(audio_payload))
                        return _build_audio_block(cleaned, audio_format)

                    if isinstance(audio_payload, dict):
                        raw_data = audio_payload.get("data")
                        if isinstance(raw_data, str):
                            if _inline_payload_bytes(raw_data) > max_inline_bytes:
                                return await _refuse_oversized_inline(_inline_payload_bytes(raw_data))
                            cleaned = _normalize_base64(raw_data)
                            if not cleaned:
                                pipe.logger.warning("Audio payload rejected: invalid base64 data.")
                                await pipe._ensure_error_formatter()._emit_error(
                                    event_emitter,
                                    "Audio input was not valid base64.",
                                    show_error_message=False,
                                )
                                return _empty_audio_block()
                            mime_hint = _resolved_mime_hint(audio_payload)
                            audio_format = _normalize_format(audio_payload.get("format"), mime_hint)
                            return _build_audio_block(cleaned, audio_format)

                    if isinstance(audio_payload, str):
                        sanitized = audio_payload.strip()
                        lowercase = sanitized.lower()
                        if is_http_or_https_url(sanitized):
                            pipe.logger.warning("Audio payload rejected: remote URLs are not supported.")
                            await pipe._ensure_error_formatter()._emit_error(
                                event_emitter,
                                "Audio input must be base64-encoded. URLs are not supported.",
                                show_error_message=False,
                            )
                            return _empty_audio_block()

                        if _inline_payload_bytes(sanitized) > max_inline_bytes:
                            return await _refuse_oversized_inline(_inline_payload_bytes(sanitized))

                        if lowercase.startswith("data:"):
                            parsed = pipe._multimodal_handler._parse_data_url(sanitized if sanitized.startswith("data:") else f"data:{sanitized.split(':', 1)[1]}")
                            if not parsed or not parsed.get("mime_type", "").startswith("audio/"):
                                pipe.logger.warning("Audio payload rejected: invalid data URL.")
                                await pipe._ensure_error_formatter()._emit_error(
                                    event_emitter,
                                    "Audio input must be base64-encoded audio data.",
                                    show_error_message=False,
                                )
                                return _empty_audio_block()
                            audio_format = _map_format(parsed.get("mime_type"))
                            return _build_audio_block(parsed.get("b64", ""), audio_format)

                        cleaned = _normalize_base64(sanitized)
                        if not cleaned:
                            pipe.logger.warning("Audio payload rejected: invalid base64 data.")
                            await pipe._ensure_error_formatter()._emit_error(
                                event_emitter,
                                "Audio input was not valid base64.",
                                show_error_message=False,
                            )
                            return _empty_audio_block()

                        mime_type = _resolved_mime_hint()
                        audio_format = _map_format(mime_type)
                        return _build_audio_block(cleaned, audio_format)

                    # Invalid/empty
                    pipe.logger.warning("Invalid audio payload format, returning empty audio block")
                    return _empty_audio_block()

                except Exception as exc:
                    pipe.logger.exception("Error in _to_input_audio")
                    await pipe._ensure_error_formatter()._emit_error(
                        event_emitter,
                        f"Audio processing error: {exc}",
                        show_error_message=False,
                    )
                    return _empty_audio_block()

            async def _to_input_video(block: dict) -> dict:
                """Convert Open WebUI video blocks into Chat Completions video format.

                Note: The Responses API doesn't have explicit `input_video` type.
                Videos use the Chat Completions `video_url` format, which OpenRouter
                handles internally.

                Video Support by Provider (per OpenRouter docs):
                    - Gemini AI Studio: YouTube links only
                    - Most providers: Limited or no video support
                    - Check model's input_modalities for "video" capability

                Supported Video Formats:
                    - Remote URLs: YouTube links, direct video URLs
                    - Data URLs: data:video/mp4;base64,... (rarely used due to size)

                Internal OWUI file URLs are never forwarded or base64-inlined
                (videos can reach VIDEO_MAX_SIZE_MB); they hard-fail via
                ``RequiredInternalFileError`` so the internal URL never leaks.

                Chat Completions Video Format:
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": "https://youtube.com/..." or "data:video/mp4;base64,..."
                        }
                    }

                Input Formats Handled:
                    1. {"type": "video_url", "video_url": {"url": "..."}}
                    2. {"type": "video_url", "video_url": "..."}
                    3. {"type": "video", "url": "...", "mimeType": "video/mp4"}

                Args:
                    block: Content block from Open WebUI message

                Returns:
                    Chat Completions video_url block

                Note:
                    Videos are NOT downloaded/stored due to large size.
                    URL validation is minimal - OpenRouter will validate provider support.
                """
                try:
                    video_payload = block.get("video_url")
                    url: str = ""

                    if isinstance(video_payload, dict):
                        url = video_payload.get("url", "")
                    elif isinstance(video_payload, str):
                        url = video_payload

                    if not url:
                        url = block.get("url", "")

                    if not url:
                        pipe.logger.warning("Video block has no URL")
                        return {"type": "video_url", "video_url": {"url": ""}}

                    if is_internal_file_url(url):
                        raise RequiredInternalFileError(
                            "Internal video URLs cannot be forwarded to the provider.",
                            kind="video",
                        )

                    if (
                        is_cleartext_http_url(url)
                        and not pipe._multimodal_handler._is_insecure_http_allowed(url)
                    ):
                        pipe.logger.error("Blocked insecure HTTP video URL by default: %s", url)
                        await pipe._ensure_error_formatter()._emit_error(
                            event_emitter,
                            "Video URL blocked by security policy (HTTP disabled by default). "
                            "Enable ALLOW_INSECURE_HTTP + ALLOW_INSECURE_HTTP_HOSTS to allow specific hosts.",
                            show_error_message=True,
                        )
                        return {"type": "video_url", "video_url": {"url": ""}}

                    if url_scheme(url) == "data":
                        if "," in url:
                            b64_data = url.split(",", 1)[1]
                            estimated_size_bytes = (len(b64_data) * 3) // 4
                            max_size_bytes = video_max_size_mb * 1024 * 1024
                            if estimated_size_bytes > max_size_bytes:
                                estimated_size_mb = estimated_size_bytes / (1024 * 1024)
                                pipe.logger.warning(
                                    f"Base64 video size (~{estimated_size_mb:.1f}MB) exceeds configured limit "
                                    f"({video_max_size_mb}MB), rejecting to prevent memory issues"
                                )
                                await pipe._ensure_error_formatter()._emit_error(
                                    event_emitter,
                                    f"Video too large (~{estimated_size_mb:.1f}MB, max: {video_max_size_mb}MB)",
                                    show_error_message=True
                                )
                                return {"type": "video_url", "video_url": {"url": ""}}

                        await pipe._event_emitter_handler._emit_status(
                            event_emitter,
                            StatusMessages.VIDEO_BASE64,
                            done=False
                        )
                    elif pipe._multimodal_handler._is_youtube_url(url):
                        await pipe._event_emitter_handler._emit_status(
                            event_emitter,
                            StatusMessages.VIDEO_YOUTUBE,
                            done=False
                        )
                    elif is_http_or_https_url(url):
                        if not await pipe._multimodal_handler._is_safe_url(url):
                            pipe.logger.error(f"SSRF protection blocked video URL: {url}")
                            await pipe._ensure_error_formatter()._emit_error(
                                event_emitter,
                                "Video URL blocked by security policy (private network)",
                                show_error_message=True
                            )
                            return {"type": "video_url", "video_url": {"url": ""}}

                        await pipe._event_emitter_handler._emit_status(
                            event_emitter,
                            StatusMessages.VIDEO_REMOTE,
                            done=False
                        )
                    else:
                        await pipe._event_emitter_handler._emit_status(
                            event_emitter,
                            StatusMessages.VIDEO_REMOTE,
                            done=False
                        )

                    return {
                        "type": "video_url",
                        "video_url": {
                            "url": url
                        }
                    }

                except RequiredInternalFileError:
                    raise
                except Exception as exc:
                    pipe.logger.exception("Error in _to_input_video")
                    await pipe._ensure_error_formatter()._emit_error(
                        event_emitter,
                        f"Video processing error: {exc}",
                        show_error_message=False
                    )
                    return {"type": "video_url", "video_url": {"url": ""}}

            def _identity_block(b: dict[str, Any]) -> dict[str, Any]:
                return b

            block_transform = {
                "text":       lambda b: {"type": "input_text",  "text": b.get("text", "")},
                "image_url":  _to_input_image,
                "input_image": _to_input_image,
                "input_file": _to_input_file,
                "file":       _to_input_file,
                "input_audio": _to_input_audio,
                "audio":      _to_input_audio,
                "video_url":  _to_input_video,
                "video":      _to_input_video,
            }

            converted_blocks: list[dict[str, Any]] = []
            user_images_used = 0
            dropped_images = 0
            refused_images: list[str] = []
            refused_files: list[str] = []
            encountered_user_images = False
            reusable_image_blocks: list[dict[str, Any]] = []
            vision_warning_sent = False
            latest_user_message = role == "user" and idx in current_turn_people
            include_user_images = (
                ((latest_user_message and vision_supported) or tool_images) and image_limit > 0
            )

            for block_idx, block in enumerate(content_blocks):
                if not block:
                    continue
                if not isinstance(block, dict):
                    if isinstance(block, str):
                        cleaned = _sanitize_free_text(block)
                        if cleaned:
                            converted_blocks.append({"type": "input_text", "text": cleaned})
                        continue
                    pipe.logger.warning(
                        "Dropping unsupported %s content block at index %d of the %s message %s",
                        type(block).__name__,
                        block_idx,
                        role,
                        msg_id,
                    )
                    continue
                raw_block_type = block.get("type")
                block_type = raw_block_type if isinstance(raw_block_type, str) else ""
                transformer = block_transform.get(block_type, _identity_block)
                is_image_block = block_type in {"image_url", "input_image"}

                if is_image_block:
                    if not (latest_user_message or tool_images):
                        reusable_image_blocks.append(block)
                    if not include_user_images:
                        if latest_user_message and not vision_supported and not vision_warning_sent:
                            await pipe._event_emitter_handler._emit_status(
                                event_emitter,
                                "Model does not accept image inputs; skipping user attachments.",
                                done=False,
                            )
                            vision_warning_sent = True
                        continue
                    if not tool_images and user_images_used >= image_limit:
                        dropped_images += 1
                        encountered_user_images = True
                        continue

                try:
                    if asyncio.iscoroutinefunction(transformer):
                        result = await transformer(block)
                    else:
                        result = transformer(block)
                    if isinstance(result, ImageRefusal):
                        if not is_image_block:
                            pipe.logger.log(
                                warn_level(
                                    _warned_oversized_inline,
                                    result.cause,
                                    cooldown_s=_REUSE_WARN_COOLDOWN_S,
                                ),
                                "Skipping an attached file (%s): %s",
                                result.subject or "no source",
                                result.reason,
                            )
                            refused_files.append(result.reason)
                            continue
                        if result.severity == "fatal":
                            raise RequiredInternalFileError(
                                f"A referenced image ({result.subject}) is "
                                f"{result.reason}.",
                                kind="image",
                            )
                        pipe.logger.log(
                            warn_level(
                                _warned_oversized_inline,
                                result.cause,
                                cooldown_s=_REUSE_WARN_COOLDOWN_S,
                            ),
                            "Skipping an attached image (%s): %s [cause=%s]",
                            result.subject or "no source",
                            result.reason,
                            result.cause,
                        )
                        if result.severity == "error":
                            await pipe._event_emitter_handler._emit_error_event(
                                event_emitter,
                                f"An attached image was {result.reason}.",
                                show_error_message=True,
                            )
                        else:
                            refused_images.append(result.reason)
                        encountered_user_images = True
                        continue
                    if result is None:
                        if is_image_block and idx == current_turn_people[-1]:
                            encountered_user_images = True
                        continue
                    if isinstance(result, dict):
                        text_value = result.get("text")
                        if isinstance(text_value, str):
                            result = dict(result)
                            cleaned = _sanitize_free_text(text_value)
                            if not cleaned:
                                continue
                            result["text"] = cleaned
                    if is_image_block and result:
                        user_images_used += 1
                        encountered_user_images = True
                    converted_blocks.append(result)
                except RequiredInternalFileError:
                    raise
                except Exception as exc:
                    pipe.logger.exception("Failed to transform block type '%s'", block_type)
                    await pipe._ensure_error_formatter()._emit_error(
                        event_emitter,
                        f"Block transformation error for '{block_type}': {exc}",
                        show_error_message=False
                    )
                    if not is_image_block:
                        converted_blocks.append(block)

            if (
                latest_user_message
                and idx == current_turn_people[-1]
                and not person_images_this_turn
                and selection_mode == "user_then_assistant"
                and include_user_images
                and user_images_used == 0
                and not encountered_user_images
                and last_image_blocks
                and last_image_turn is not None
                and msg_turn_index is not None
                and (msg_turn_index - last_image_turn) <= image_reuse_turns
            ):
                fallback_slots = min(image_limit, len(last_image_blocks))
                fallback_blocks: list[dict[str, Any]] = []
                for source_block in last_image_blocks[:fallback_slots]:
                    try:
                        transformed = await _to_input_image(source_block, mode="reuse")
                        if isinstance(transformed, ImageRefusal):
                            pipe.logger.log(
                                warn_level(
                                    _warned_image_reuse,
                                    transformed.cause,
                                    cooldown_s=_REUSE_WARN_COOLDOWN_S,
                                ),
                                "Not reusing an earlier image: %s [cause=%s]",
                                transformed.reason,
                                transformed.cause,
                            )
                            refused_images.append(transformed.reason)
                        elif transformed is not None:
                            fallback_blocks.append(transformed)
                    except RequiredInternalFileError as exc:
                        pipe.logger.log(
                            logging.WARNING
                            if exc.denied
                            else warn_level(
                                _warned_image_reuse,
                                "reuse_unavailable",
                                cooldown_s=_REUSE_WARN_COOLDOWN_S,
                            ),
                            "Not reusing an earlier image: %s",
                            exc.user_message,
                        )
                        refused_images.append(exc.user_message)
                    except Exception:
                        pipe.logger.exception("Failed to reuse assistant image")
                if fallback_blocks:
                    pipe.logger.debug(
                        "Rehydrating %d assistant-generated image(s) due to empty user attachments (selection_mode=%s, limit=%d).",
                        len(fallback_blocks),
                        selection_mode,
                        image_limit,
                    )
                    converted_blocks = fallback_blocks + converted_blocks
                    user_images_used = len(fallback_blocks)

            if latest_user_message and (user_images_used or encountered_user_images):
                person_images_this_turn = True
            if reusable_image_blocks:
                last_image_blocks = reusable_image_blocks
                last_image_turn = msg_turn_index

            image_notices: list[str] = []
            if refused_images:
                image_notices.append(
                    f"skipped {len(refused_images)} ({'; '.join(refused_images)})"
                )
            if dropped_images:
                image_notices.append(
                    f"dropped {dropped_images} over the limit of {image_limit}"
                )
            notices = ["Images: " + "; ".join(image_notices) + "."] if image_notices else []
            if refused_files:
                notices.append(f"Files: skipped {len(refused_files)} ({'; '.join(refused_files)}).")
            if notices and (latest_user_message or (tool_images and idx == last_tool_handoff_index)):
                await pipe._event_emitter_handler._emit_status(
                    event_emitter,
                    " ".join(notices),
                    done=False,
                )

            if not converted_blocks and (refused_images or refused_files):
                reasons = "; ".join(reason.rstrip(".") for reason in refused_files + refused_images)
                converted_blocks.append({
                    "type": "input_text",
                    "text": f"[An attached item was not sent: {reasons}.]",
                })
            openai_input.append({
                "type": "message",
                "role": "user",
                "content": converted_blocks,
            })
            continue

        raw_msg_annotations = msg.get("annotations")
        msg_annotations: list[Any] = (
            list(raw_msg_annotations)
            if isinstance(raw_msg_annotations, list) and raw_msg_annotations
            else []
        )
        raw_msg_reasoning_details = msg.get("reasoning_details")
        msg_reasoning_details: list[Any] = (
            list(raw_msg_reasoning_details)
            if isinstance(raw_msg_reasoning_details, list) and raw_msg_reasoning_details
            else []
        )
        assistant_text = (
            raw_content
            if isinstance(raw_content, str)
            else _extract_plain_text_content(raw_content)
        )
        is_old_message = _is_old_turn(msg_turn_index, threshold=prune_before_turn)
        assistant_image_urls = _markdown_images_from_text(assistant_text)
        if assistant_image_urls:
            last_image_blocks = [
                {"type": "image_url", "image_url": url, "detail": "auto"}
                for url in assistant_image_urls
            ]
            last_image_turn = msg_turn_index

        appended_text_chunks: list[dict[str, Any]] = []

        def _append_assistant_text_chunks(
            text: str,
            appended: list[dict[str, Any]] = appended_text_chunks,
            msg_annotations: list[Any] = msg_annotations,
            msg_reasoning_details: list[Any] = msg_reasoning_details,
        ) -> None:
            chunk_items: list[dict[str, Any]] = []
            for phase_chunk in split_text_by_phase_markers(text):
                cleaned_text = phase_chunk["text"].strip()
                if not cleaned_text:
                    continue
                item_out: dict[str, Any] = {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": cleaned_text}],
                }
                if phase_supported and phase_chunk.get("phase_present"):
                    item_out["phase"] = phase_chunk.get("phase")
                chunk_items.append(item_out)

            if not chunk_items:
                return
            openai_input.extend(chunk_items)
            appended.extend(chunk_items)

        if contains_marker(assistant_text):
            segments = split_text_by_markers(assistant_text)
            markers = [seg["marker"] for seg in segments if seg.get("type") == "marker"]

            db_artifacts: dict[str, dict] = {}
            orphaned_call_ids: set[str] = set()
            orphaned_output_ids: set[str] = set()
            if artifact_loader and chat_id and openwebui_model_id and markers:
                try:
                    db_artifacts = await artifact_loader(chat_id, msg_id, markers)
                    (
                        _,
                        orphaned_call_ids,
                        orphaned_output_ids,
                    ) = _classify_function_call_artifacts(db_artifacts)
                    if orphaned_call_ids:
                        logger.debug(
                            "Dropping %d persisted function_call artifact(s) missing outputs (chat_id=%s message_id=%s call_ids=%s)",
                            len(orphaned_call_ids),
                            chat_id,
                            msg_id,
                            sorted(orphaned_call_ids),
                        )
                    if orphaned_output_ids:
                        logger.warning(
                            "Dropping %d persisted function_call_output artifact(s) missing calls (chat_id=%s message_id=%s call_ids=%s)",
                            len(orphaned_output_ids),
                            chat_id,
                            msg_id,
                            sorted(orphaned_output_ids),
                        )
                except Exception:
                    logger.warning("Artifact loader failed for chat_id=%s message_id=%s", chat_id, msg_id, exc_info=True)
                    db_artifacts = {}

            for segment in segments:
                if segment["type"] == "marker":
                    artifact_payload = db_artifacts.get(segment["marker"])
                    if artifact_payload is None:
                        missing_artifact_markers.append(segment["marker"])
                        continue
                    if (
                        artifact_payload.get("type") == "reasoning"
                        and replayed_reasoning_refs is not None
                        and chat_id
                    ):
                        replayed_reasoning_refs.append((chat_id, segment["marker"]))
                    item = normalize_persisted_item(artifact_payload)
                    if item is not None:
                        item_type = ((item.get("type") or "").lower())
                        if item_type in _NON_REPLAYABLE_TOOL_ARTIFACTS:
                            pipe.logger.debug(
                                "Skipping %s artifact when rebuilding provider context (not replayable).",
                                item_type,
                            )
                            continue
                        if (
                            item_type == "function_call"
                            and item.get("call_id") in orphaned_call_ids
                        ):
                            logger.debug(
                                "Skipping orphaned function_call artifact (call_id=%s chat_id=%s message_id=%s)",
                                item.get("call_id"),
                                chat_id,
                                msg_id,
                            )
                            continue
                        if (
                            item_type == "function_call_output"
                            and item.get("call_id") in orphaned_output_ids
                        ):
                            logger.debug(
                                "Skipping orphaned function_call_output artifact (call_id=%s chat_id=%s message_id=%s)",
                                item.get("call_id"),
                                chat_id,
                                msg_id,
                            )
                            continue
                        if item_type == "function_call_output" and is_picture_output(item.get("output")):
                            last_image_blocks, last_image_turn = [], None
                        if _withheld(msg_turn_index):
                            withheld_items = _without_tool_result(item, tool_names_by_call_id)
                            if withheld_items is not None:
                                openai_input.extend(_from_pipe_storage(withheld) for withheld in withheld_items)
                                continue
                        for part in _as_replayed(item, fallback_id=segment["marker"]):
                            if (
                                is_old_message
                                and pruning_turns > 0
                                and prune_before_turn is not None
                            ):
                                _prune_tool_output(
                                    part,
                                    marker=segment["marker"],
                                    turn_index=msg_turn_index,
                                    retention_turns=pruning_turns,
                                )
                            openai_input.append(_from_pipe_storage(part))
                elif segment["type"] == "text":
                    _append_assistant_text_chunks(segment["text"])
        else:
            _append_assistant_text_chunks(assistant_text)

        if appended_text_chunks:
            if msg_annotations:
                appended_text_chunks[-1]["annotations"] = msg_annotations
            if msg_reasoning_details:
                appended_text_chunks[-1]["reasoning_details"] = msg_reasoning_details

        if msg_tool_calls:
            for index, tool_call in enumerate(msg_tool_calls):
                if not isinstance(tool_call, dict):
                    continue
                if tool_call.get("type") not in (None, "function"):
                    continue

                tool_call_id = tool_call.get("id") or tool_call.get("call_id")
                tool_call_id = tool_call_id.strip() if isinstance(tool_call_id, str) else ""
                if not tool_call_id:
                    tool_call_id = f"call_{generate_item_id()}_{index}"

                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                name = name.strip() if isinstance(name, str) else ""
                if not name:
                    continue

                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    args_text = arguments.strip() or "{}"
                else:
                    try:
                        args_text = json.dumps(arguments or {}, ensure_ascii=False)
                    except (TypeError, ValueError):
                        args_text = "{}"
                if _withheld(msg_turn_index) and name != "ask_user":
                    args_text = "{}"

                openai_input.append(
                    {
                        "type": "function_call",
                        "id": tool_call_id,
                        "call_id": tool_call_id,
                        "name": name,
                        "arguments": args_text,
                    }
                )

    if missing_artifact_markers:
        distinct_missing = sorted(set(missing_artifact_markers))
        logger.log(
            logging.DEBUG if temporary_chat else logging.WARNING,
            "Missing %d artifact(s) across %d marker reference(s) for chat_id=%s: %s",
            len(distinct_missing),
            len(missing_artifact_markers),
            chat_id,
            distinct_missing,
        )

    openai_input = _reinterleave_reasoning_by_anchor(openai_input)

    _maybe_apply_anthropic_prompt_caching(
        openai_input,
        model_id=target_model_id,
        valves=active_valves,
    )
    return openai_input
