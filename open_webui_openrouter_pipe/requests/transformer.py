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
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

from ..core.config import (
    _NON_REPLAYABLE_TOOL_ARTIFACTS,
    _RAW_REPLAYED_SERVER_TOOLS,
    OPENAI_ATTACHMENT_NOT_SENT_PREFIX,
    OPENAI_EMPTY_USER_TURN_FALLBACK,
    markdown_image_spans,
)

# Import status messages
from ..core.context_budget import inline_payload_bytes
from ..core.errors import RequiredInternalFileError, StatusMessages
from ..core.image_detail import image_detail_or_auto
from ..core.url_scheme import (
    base64_data_url_payload_chars,
    base64_data_url_payload_len,
    first_n_non_whitespace,
    is_absolute_url,
    is_cleartext_http_url,
    is_http_or_https_url,
    is_inline_data_url,
    link_media_type,
    loggable_link,
    split_base64_data_url,
    url_scheme,
)

# Import utility functions
from ..core.utils import (
    _MARKER_SUFFIX,
    BUILTIN_ASK_USER_ROUND_KEY,
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
    _iter_marker_spans,
    _tool_result_failed,
    is_picture_output,
    is_server_tool_call_id,
    is_text_part_output,
    is_tool_image_handoff,
    is_tool_image_handoff_for_round,
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
    _NO_VERDICT,
    _SNIFF_PREFIX_BYTES,
    ADDRESS_CHECK_BUDGET_SECONDS,
    ADDRESS_CHECK_SECONDS,
    _sniff_evidence,
    remote_file_limit_scope,
    resolve_download_type,
)
from ..storage.owui_files import (
    InlineFileTooLargeError,
    extract_internal_file_id,
    is_temporary_chat,
    names_an_owui_file_path,
)

# Import from persistence
from ..storage.persistence import generate_item_id, normalize_persisted_item
from ..tools.tool_schema import (
    _classify_function_call_artifacts,
)
from .tool_names import _tool_names_by_position

if TYPE_CHECKING:
    from ..pipe import Pipe

# Tool output pruning constants
_REPLAYABLE_TOOL_ARTIFACTS = frozenset({"function_call", "function_call_output"})
_TOOL_OUTPUT_PRUNE_MIN_LENGTH = 800
_TOOL_OUTPUT_PRUNE_HEAD_CHARS = 256
_TOOL_OUTPUT_PRUNE_TAIL_CHARS = 128

_PROSE_KEYS = frozenset({"text", "input_text", "output_text", "summary_text", "content"})
_INTERNAL_FILE_PATH = "/api/v1/files/"

_ARTIFACT_GROUP_CONCURRENCY = 8

_AUDIO_URL_REFUSAL = "an audio clip must be base64-encoded; URLs are not supported"


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


_NOT_A_TOOL_ROUND = frozenset({"message", "reasoning"})


def _withheld_provider_round(item: dict[str, Any]) -> list[dict[str, Any]]:
    item_type = str(item.get("type") or "")
    call_id = str(item.get("call_id") or item.get("id") or "")
    name = item.get("name")
    return [
        {"type": "function_call", "call_id": call_id,
         "name": name.strip() if isinstance(name, str) and name.strip() else item_type,
         "arguments": "{}"},
        {"type": "function_call_output", "call_id": call_id,
         "output": unretained_tool_result(server_tool_status(item) != "completed")},
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


def _round_keeps_its_ask_user_answer(
    call_id: Any,
    name: str,
    ask_user_names: frozenset[str],
    recorded_rounds: frozenset[tuple[str, str]],
    builtin_rounds: frozenset[tuple[str, str]],
) -> bool:
    key = (str(call_id or ""), name)
    if key in builtin_rounds:
        return True
    if key in recorded_rounds:
        return False
    return name in ask_user_names


def _orphaned_round_markers(
    segments: list[dict[str, Any]], artifacts: dict[str, Any]
) -> tuple[set[str], set[str]]:
    calls_by_id: dict[str, list[str]] = {}
    outputs_by_id: dict[str, list[str]] = {}
    for segment in segments:
        if segment.get("type") != "marker":
            continue
        marker = str(segment.get("marker") or "")
        if not marker:
            continue
        item = artifacts.get(marker)
        if not isinstance(item, dict):
            continue
        item_type = str(item.get("type") or "").lower()
        if item_type not in _REPLAYABLE_TOOL_ARTIFACTS:
            continue
        call_id = str(item.get("call_id") or "")
        if not call_id:
            continue
        bucket = calls_by_id if item_type == "function_call" else outputs_by_id
        bucket.setdefault(call_id, []).append(marker)

    orphaned_calls: set[str] = set()
    orphaned_outputs: set[str] = set()
    for call_id, calls in calls_by_id.items():
        orphaned_calls.update(calls[len(outputs_by_id.get(call_id) or []):])
    for call_id, outputs in outputs_by_id.items():
        orphaned_outputs.update(outputs[len(calls_by_id.get(call_id) or []):])
    return orphaned_calls, orphaned_outputs


def _replay_round_name(item_type: str, name: str, call_id: Any, pending: dict[str, list[str]]) -> str:
    key = str(call_id or "")
    if item_type == "function_call":
        pending.setdefault(key, []).append(name)
        return name
    if item_type == "function_call_output":
        queue = pending.get(key) or []
        return queue.pop(0) if queue else ""
    return name


def _without_tool_result(
    item: dict[str, Any],
    name: str,
    ask_user_names: frozenset[str],
    recorded_rounds: frozenset[tuple[str, str]],
    builtin_rounds: frozenset[tuple[str, str]],
) -> list[dict[str, Any]] | None:
    item_type = item.get("type")
    if item_type == "function_call":
        exempt = _round_keeps_its_ask_user_answer(
            item.get("call_id"), name, ask_user_names, recorded_rounds, builtin_rounds
        )
        return None if exempt else [{**item, "arguments": "{}"}]
    if item_type == "function_call_output":
        if _round_keeps_its_ask_user_answer(
            item.get("call_id"), name, ask_user_names, recorded_rounds, builtin_rounds
        ):
            return None
        text = tool_output_text_and_pictures(item.get("output"))[0]
        return [{**item, "output": unretained_tool_result(_tool_result_failed(text, item.get("status")))}]
    if isinstance(item_type, str) and item_type.startswith("openrouter:"):
        return _server_round(item, "{}", unretained_tool_result(server_tool_status(item) != "completed"))
    if isinstance(item_type, str) and item_type.lower() in _NOT_A_TOOL_ROUND:
        return None
    return _withheld_provider_round(item)


_REUSE_DOWNLOAD_MEMO_MAX_BYTES = 8 * 1024 * 1024
_BASE64_SCAN_CHUNK = 1 << 12
_REUSE_WARN_COOLDOWN_S = 30.0


class _ReuseDownloadMemo(OrderedDict):
    __slots__ = ("held",)

    def __init__(self) -> None:
        super().__init__()
        self.held: int = 0

    @staticmethod
    def _value_bytes(value: tuple[str | None, bytes, str]) -> int:
        return len(value[1])

    def __setitem__(self, key, value) -> None:
        old = self.get(key)
        if old is not None:
            self.held -= self._value_bytes(old)
        super().__setitem__(key, value)
        self.held += self._value_bytes(value)

    def __delitem__(self, key) -> None:
        old = self[key]
        super().__delitem__(key)
        self.held -= self._value_bytes(old)

    _pop_sentinel: Any = object()

    def pop(self, key, default=_pop_sentinel):
        if key in self:
            self.held -= self._value_bytes(self[key])
        if default is self._pop_sentinel:
            return super().pop(key)
        return super().pop(key, default)

    def popitem(self, last: bool = True):
        key, value = super().popitem(last=last)
        self.held -= self._value_bytes(value)
        return key, value

    def clear(self) -> None:
        super().clear()
        self.held = 0

    def update(self, *args, **kwargs) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value


_reuse_download_memo: _ReuseDownloadMemo = _ReuseDownloadMemo()
_warned_image_reuse: dict[str, float] = {}
_warned_oversized_inline: dict[str, float] = {}


def _b64encode_ascii(raw: bytes) -> str:
    pieces: list[str] = []
    whole_bytes = (len(raw) // 3) * 3
    if whole_bytes:
        pieces.append(base64.b64encode(raw[:whole_bytes]).decode("ascii"))
    leftover = raw[whole_bytes:]
    if leftover:
        pieces.append(base64.b64encode(leftover).decode("ascii"))
    return "".join(pieces)


def _base64_quantum_is_valid(payload: str) -> bool:
    try:
        binascii.a2b_base64(payload, strict_mode=True)
    except (binascii.Error, ValueError):
        return False
    return True


def _is_well_formed_base64(cleaned: str) -> bool:
    if not cleaned:
        return False
    n = len(cleaned)
    end = n - (n % _BASE64_SCAN_CHUNK) if n >= _BASE64_SCAN_CHUNK else n
    for i in range(0, end, _BASE64_SCAN_CHUNK):
        if not _base64_quantum_is_valid(cleaned[i : i + _BASE64_SCAN_CHUNK]):
            return _is_valid_base64_as_a_whole(cleaned)
    tail = cleaned[end:]
    if not tail:
        return True
    return _base64_quantum_is_valid(tail) or _is_valid_base64_as_a_whole(cleaned)


def _is_valid_base64_as_a_whole(cleaned: str) -> bool:
    try:
        base64.b64decode(cleaned, validate=True)
    except (binascii.Error, ValueError):
        return False
    return True


async def _validate_inline_payload(cleaned: str) -> bool:
    return await asyncio.to_thread(_is_well_formed_base64, cleaned)


AUDIO_FORMAT_MAP: dict[str, str] = {
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
}

SUPPORTED_AUDIO_FORMATS: frozenset[str] = frozenset(
    {"mp3", "wav", "flac", "m4a", "ogg", "aiff", "aac", "pcm16", "pcm24"}
)


def _map_audio_format(mime: str | None) -> str | None:
    if not isinstance(mime, str) or not mime.strip():
        return "mp3"
    return AUDIO_FORMAT_MAP.get(mime.lower())


def _normalize_audio_format(
    explicit_format: str | None,
    mime_hint: str | None,
) -> str | None:
    if isinstance(explicit_format, str) and explicit_format.strip():
        normalized = explicit_format.strip().lower()
        if normalized in SUPPORTED_AUDIO_FORMATS:
            return normalized
        return None
    return _map_audio_format(mime_hint)


class ImageRefusal(NamedTuple):
    reason: str
    cause: str
    severity: Literal["status", "error", "fatal"] = "status"
    subject: str = ""


def _resolve_inline_type(
    head: str,
    body: str,
    *,
    cause: str = "reuse_untyped",
    subject: str = "",
    split_from: str = "",
) -> tuple[str, ImageRefusal | None]:
    declared = head[len("data:"):].split(";", 1)[0].strip().lower()
    try:
        sniffed = base64.b64decode(
            first_n_non_whitespace(body, (_SNIFF_PREFIX_BYTES + 2) // 3 * 4)
        )
    except (binascii.Error, ValueError):
        sniffed = b""
    resolved = resolve_download_type(declared, _sniff_evidence(sniffed))
    if not resolved.startswith("image/"):
        return head, ImageRefusal("not identifiable as an image", cause, subject=subject)
    if resolved != declared:
        return f"data:{resolved};base64,{body}", None
    if split_from:
        return split_from, None
    return f"{head},{body}", None


async def _gate_inline_data_url(
    url: str,
    max_inline_bytes: int,
    *,
    resolve_type: bool,
) -> tuple[str, tuple[str, str] | None, ImageRefusal | None]:
    payload_chars = base64_data_url_payload_len(url)
    if payload_chars is None:
        return url, None, ImageRefusal(
            "a data URL that is not base64-encoded, which OpenRouter does not accept",
            "unencoded_inline",
            subject=loggable_link(url),
        )
    if (payload_chars * 3) // 4 > max_inline_bytes:
        folded = base64_data_url_payload_chars(url)
        if folded is None or (folded * 3) // 4 > max_inline_bytes:
            return url, None, ImageRefusal(
                f"larger than the {max_inline_bytes}-byte inline limit",
                "oversized_inline",
                subject=loggable_link(url),
            )
    split = split_base64_data_url(url)
    assert split is not None
    body = "".join(split[1].split())
    if not body or not await _validate_inline_payload(body):
        return url, None, ImageRefusal(
            "not decodable as base64", "undecodable_inline", subject=loggable_link(url),
        )
    head = "data:" + split[0].partition(":")[2]
    unchanged = (
        type(url) is str
        and url[: url.find(",")] == head
        and body == split[1]
    )
    if not resolve_type:
        return (url if unchanged else head + "," + body), (head, body), None
    resolved, refusal = _resolve_inline_type(head, body, split_from=url if unchanged else "")
    return resolved, None, refusal


async def _gate_inline_tool_pictures(
    pictures: list[str],
    max_inline_bytes: int,
    *,
    allow_insecure: Callable[[str], bool],
) -> tuple[list[str], list[tuple[str, str, str]]]:
    kept: list[str] = []
    refused: list[tuple[str, str, str]] = []
    for url in pictures:
        if not is_inline_data_url(url):
            admitted, not_a_link = _tool_picture_gate(
                [url], max_inline_bytes=max_inline_bytes, allow_insecure=allow_insecure,
            )
            kept.extend(admitted)
            refused.extend(not_a_link)
            continue
        _gated, _parts, refusal = await _gate_inline_data_url(
            url, max_inline_bytes, resolve_type=True,
        )
        if refusal is not None:
            refused.append((url, refusal.reason, refusal.cause))
            continue
        kept.append(_gated)
    return kept, refused


async def _gated_tool_pictures_with_address(
    pipe: Pipe,
    pictures: list[str],
    *,
    max_inline_bytes: int,
    allow_insecure: Callable[[str], bool],
    seen: dict[str, bool | None] | None = None,
    deadline: float | None = None,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    typed, refused = await _gate_inline_tool_pictures(
        pictures, max_inline_bytes, allow_insecure=allow_insecure,
    )
    admitted, unfetchable = await _tool_picture_address_gate(
        pipe, typed, seen=seen, deadline=deadline,
    )
    return admitted, [*refused, *unfetchable]


async def _gate_round_output_pictures(
    pipe: Pipe,
    output: Any,
    max_inline_bytes: int,
    *,
    allow_insecure: Callable[[str], bool],
    seen: dict[str, bool | None] | None = None,
    deadline: float | None = None,
) -> tuple[Any, list[tuple[str, str, str]]]:
    if not is_picture_output(output):
        return output, []
    text, pictures = tool_output_text_and_pictures(output)
    kept, refused = await _gated_tool_pictures_with_address(
        pipe, pictures, max_inline_bytes=max_inline_bytes, allow_insecure=allow_insecure,
        seen=seen, deadline=deadline,
    )
    return picture_output(text, kept), refused


def _inline_payload_bytes(value: str) -> int:
    return inline_payload_bytes(value)


def _payload_is_present(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        for key in ("url", "data"):
            if key in value:
                return _payload_is_present(value[key])
    return False


def _block_is_usable(block: dict[str, Any]) -> bool:
    btype = block.get("type")
    if btype == "input_text":
        return isinstance(block.get("text"), str) and bool(block["text"].strip())
    if btype == "input_file":
        return any(block.get(key) for key in ("file_id", "file_data", "file_url"))
    if btype in {"input_image", "image_url"}:
        return _payload_is_present(block.get("image_url"))
    if btype == "input_audio":
        return _payload_is_present(block.get("input_audio"))
    if btype == "video_url":
        return _payload_is_present(block.get("video_url"))
    return _unconverted_block_reason(block) is None


NO_AUDIO_DATA = "an audio clip carried no audio data"
NO_VIDEO_DATA = "a video clip carried no video data"


def _unconverted_block_reason(block: dict[str, Any]) -> str | None:
    btype = block.get("type")
    if btype in {"input_image", "image_url", "image"}:
        return "an image carried no picture data" if not _payload_is_present(
            block.get("image_url", block.get("url"))
        ) else None
    if btype in {"input_audio", "audio"}:
        return NO_AUDIO_DATA if not _payload_is_present(
            block.get("input_audio", block.get("audio", block.get("data")))
        ) else None
    if btype in {"video_url", "video"}:
        return NO_VIDEO_DATA if not _payload_is_present(
            block.get("video_url", block.get("url"))
        ) else None
    if btype in {"input_file", "file"}:
        return "a file carried no contents" if not any(
            block.get(key) for key in ("file_id", "file_data", "file_url", "file")
        ) else None
    return None


def _oversized_inline_refusal(
    limit_bytes: int, cause: str, subject: str
) -> ImageRefusal:
    return ImageRefusal(
        f"larger than the {limit_bytes}-byte inline limit", cause,
        severity="error", subject=subject,
    )


def _is_text_ordinal_upper_bound(
    msg_items: list[dict[str, Any]],
    ordinal: int,
) -> bool:
    return bool(msg_items) and 0 <= ordinal <= len(msg_items) + 1


def _text_ordinal_anchor(
    msg_items: list[tuple[int, dict[str, Any]]],
    ordinal: int,
) -> int | None:
    if 0 <= ordinal < len(msg_items):
        return msg_items[ordinal][0]
    if not msg_items or ordinal < 0:
        return None
    return msg_items[-1][0]


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
    region_messages = [
        it
        for it in region
        if isinstance(it, dict) and it.get("type") == "message" and it.get("role") == "assistant"
    ]
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
                if _is_text_ordinal_upper_bound(region_messages, text_ordinal):
                    movable.append((seq, stripped, "text", text_ordinal, server_item))
                else:
                    skeleton.append(stripped)
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
            pos = _text_ordinal_anchor(msg_items, ordinal)
            bucket = inserts_after if (msg_items and ordinal >= len(msg_items)) else inserts_before
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
_REPLAY_GATED_PICTURES_KEY = "_gated_replay_pictures"
_TRANSPORT_ONLY_KEYS = (
    _PIPE_STORAGE_KEY,
    _REPLAY_GATED_PICTURES_KEY,
    PIPE_ONLY_TOOL_ROUND_KEY,
    BUILTIN_ASK_USER_ROUND_KEY,
    TOOL_ROUND_SKELETON_KEY,
)
_LIFTED_TEXT_IMAGE_PLACEHOLDER = "image"

_MEDIA_BLOCK_TYPES = frozenset({
    "image_url", "input_image", "image", "input_file", "file",
    "input_audio", "audio", "video_url", "input_video", "video",
})


def _source_url_of(block: dict[str, Any]) -> str:
    payload = block.get("image_url")
    candidate: Any = payload.get("url") if isinstance(payload, dict) else payload
    if not isinstance(candidate, str) or not candidate:
        fallback = block.get("url")
        candidate = fallback if isinstance(fallback, str) else ""
    return candidate


def _from_pipe_storage(item: dict[str, Any]) -> dict[str, Any]:
    kind = str(item.get("type") or "")
    if kind in ("function_call", "function_call_output") or kind.startswith("openrouter:"):
        return {**item, _PIPE_STORAGE_KEY: True}
    return item


def _pipe_row_is_shadowed(
    it: dict[str, Any],
    supplied: set[Any],
    owui_answered: set[Any],
    shadowed: set[Any],
) -> bool:
    if it.get("type") == "function_call_output":
        return it.get("call_id") not in owui_answered and it.get("call_id") not in shadowed
    return False


def _move_kept_outputs_after_their_calls(kept: list[Any]) -> list[Any]:
    first_call: dict[Any, int] = {}
    for index, it in enumerate(kept):
        if isinstance(it, dict) and it.get("type") == "function_call":
            first_call.setdefault(it.get("call_id"), index)
    misplaced = [
        (index, it)
        for index, it in enumerate(kept)
        if isinstance(it, dict)
        and it.get("type") == "function_call_output"
        and it.get("call_id") in first_call
        and index < first_call[it.get("call_id")]
    ]
    if not misplaced:
        return kept
    by_call: dict[Any, list[Any]] = {}
    for _source_index, source in misplaced:
        by_call.setdefault(source.get("call_id"), []).append(source)
    drop = {index for index, _ in misplaced}
    out: list[Any] = []
    for index, it in enumerate(kept):
        if index in drop:
            continue
        out.append(it)
        if isinstance(it, dict) and it.get("type") == "function_call":
            call_id = it.get("call_id")
            if call_id in first_call and first_call[call_id] == index:
                out.extend(by_call.get(call_id, ()))
    return out


def _one_copy_per_round(
    region: list[Any],
) -> list[Any]:
    supplied = {
        it.get("call_id")
        for it in region
        if isinstance(it, dict) and it.get("type") == "function_call" and not it.get(_PIPE_STORAGE_KEY)
    }
    owui_answered = {
        it.get("call_id")
        for it in region
        if isinstance(it, dict)
        and it.get("type") == "function_call_output"
        and not it.get(_PIPE_STORAGE_KEY)
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
    kept_output_ids: set[Any] = set()
    pictures: list[str] = []
    refused_pictures = False
    for it in region:
        if pictures and not (isinstance(it, dict) and it.get("type") == "function_call_output"):
            message = _tool_images_message(pictures, lead_in=refused_pictures)
            if message is not None:
                kept.append(message)
            pictures = []
            refused_pictures = False
        replayed_pictures = (
            isinstance(it, dict)
            and bool(it.get(_PIPE_STORAGE_KEY))
            and it.get("type") == "function_call_output"
            and is_picture_output(it.get("output"))
        )
        row_gated = it.get(_REPLAY_GATED_PICTURES_KEY) if replayed_pictures else None
        if isinstance(it, dict) and it.get("type") in ("function_call", "function_call_output"):
            from_pipe = bool(it.get(_PIPE_STORAGE_KEY))
            pipe_only = bool(it.get(PIPE_ONLY_TOOL_ROUND_KEY) or it.get(TOOL_ROUND_SKELETON_KEY))
            if from_pipe and not pipe_only and it.get("call_id") in supplied:
                if from_pipe and _pipe_row_is_shadowed(
                    it, supplied, owui_answered, shadowed
                ) and it.get("call_id") not in kept_output_ids:
                    pass
                else:
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
            if row_gated is not None:
                kept_shown = row_gated
            else:
                kept_shown = []
            refused_pictures = refused_pictures or bool(shown) and not kept_shown
            pictures.extend(kept_shown)
        kept.append(it)
        if isinstance(it, dict) and it.get("type") == "function_call_output":
            kept_output_ids.add(it.get("call_id"))
    if pictures or refused_pictures:
        message = _tool_images_message(pictures, lead_in=refused_pictures)
        if message is not None:
            kept.append(message)
    return _move_kept_outputs_after_their_calls(kept)


def _round_results_across(
    messages: list[dict[str, Any]], offsets: Iterable[int], results: list[Any],
) -> bool:
    for offset in offsets:
        message = messages[offset]
        role = (message.get("role") or "").lower()
        if role == "tool":
            results.append(message)
            continue
        if role in ("assistant", "system", "developer"):
            continue
        return False
    return True


def _handoff_ahead(messages: list[dict[str, Any]], position: int) -> bool:
    results: list[Any] = [messages[position]]
    for offset in range(position + 1, len(messages)):
        if is_tool_image_handoff_for_round(results, messages[offset]):
            return True
        if not _round_results_across(messages, (offset,), results):
            return False
    return False


def _handoff_back(messages: list[dict[str, Any]], position: int) -> bool:
    results: list[Any] = []
    _round_results_across(messages, range(position - 1, -1, -1), results)
    return is_tool_image_handoff_for_round(results, messages[position])


def _tool_picture_gate(
    pictures: list[str],
    *,
    max_inline_bytes: int,
    allow_insecure: Callable[[str], bool],
) -> tuple[list[str], list[tuple[str, str, str]]]:
    kept: list[str] = []
    refused: list[tuple[str, str, str]] = []
    for url in pictures:
        if (
            is_cleartext_http_url(url)
            and not names_an_owui_file_path(url)
            and not allow_insecure(url)
        ):
            refused.append((url, ("served over plain HTTP, which is blocked by security policy; "
                                  "set ALLOW_INSECURE_HTTP and list the host in "
                                  "ALLOW_INSECURE_HTTP_HOSTS to permit it"), "insecure_http"))
            continue
        if not (url_scheme(url) in ("data", "http", "https") or names_an_owui_file_path(url)):
            refused.append((url, "not a link the pipe can resolve into an image", "unusable_link"))
            continue
        if is_inline_data_url(url):
            payload_len = base64_data_url_payload_len(url)
            if payload_len is None:
                refused.append((url, ("a data URL that is not base64-encoded, which OpenRouter "
                                      "does not accept"), "unencoded_inline"))
                continue
            if (payload_len * 3) // 4 > max_inline_bytes:
                folded = base64_data_url_payload_chars(url)
                if folded is None or (folded * 3) // 4 > max_inline_bytes:
                    refused.append((url, f"larger than the {max_inline_bytes}-byte inline limit",
                                    "oversized_inline"))
                    continue
        kept.append(url)
    return kept, refused


async def _tool_picture_address_gate(
    pipe: Pipe,
    pictures: list[str],
    *,
    seen: dict[str, bool | None] | None = None,
    deadline: float | None = None,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    admitted: list[str] = []
    refused: list[tuple[str, str, str]] = []
    for url in pictures:
        if not (is_http_or_https_url(url) and not names_an_owui_file_path(url)):
            admitted.append(url)
            continue
        if seen is not None and url in seen:
            permitted = seen[url]
        else:
            if deadline is None:
                deadline = time.monotonic() + ADDRESS_CHECK_BUDGET_SECONDS
            try:
                permitted = await pipe._multimodal_handler._is_safe_url(
                    url, seconds=_remaining_address_seconds(deadline),
                )
            except Exception:
                pipe.logger.warning(
                    "The address check for a tool's picture could not run, so it was not sent: %s",
                    loggable_link(url), exc_info=True,
                )
                permitted = _NO_VERDICT
            if seen is not None:
                seen[url] = permitted
        if permitted is not True:
            if permitted is _NO_VERDICT:
                refused.append((url, "could not be checked in time, so it was not sent", "uncheckable_tool_picture"))
            else:
                refused.append((url, "could not be fetched, so it was not sent", "remote_unfetched"))
            continue
        admitted.append(url)
    return admitted, refused


def _tool_picture_verdict_gate(
    kept: list[str], verdicts: dict[str, bool | None] | None
) -> tuple[list[str], list[tuple[str, str, str]]]:
    admitted: list[str] = []
    refused: list[tuple[str, str, str]] = []
    for url in kept:
        if names_an_owui_file_path(url) or not is_http_or_https_url(url):
            admitted.append(url)
            continue
        if verdicts is None:
            if url_scheme(url) != "https":
                admitted.append(url)
                continue
        elif verdicts.get(url) is True:
            admitted.append(url)
            continue
        refused.append((url, "could not be fetched, so it was not sent", "remote_unfetched"))
    return admitted, refused


async def _tool_picture_verdicts_for_input(
    pipe: Pipe,
    input_items: Any,
    *,
    seen: dict[str, bool | None] | None = None,
    deadline: float | None = None,
) -> dict[str, bool | None]:
    verdicts = {} if seen is None else seen
    if not isinstance(input_items, list):
        return verdicts
    pictures: list[str] = []
    for row in input_items:
        if not (isinstance(row, dict) and row.get("type") == "function_call_output"):
            continue
        output = row.get("output")
        if not is_picture_output(output):
            continue
        _text, urls = tool_output_text_and_pictures(output)
        pictures.extend(urls)
    if pictures:
        await _tool_picture_address_gate(pipe, pictures, seen=verdicts, deadline=deadline)
    return verdicts


async def _tool_picture_gate_with_address(
    pipe: Pipe | None,
    pictures: list[str],
    *,
    max_inline_bytes: int,
    seen: dict[str, bool | None] | None = None,
    deadline: float | None = None,
) -> tuple[list[str], list[tuple[str, str, str]]]:
    if pipe is None:
        kept, refused = _tool_picture_gate(
            pictures,
            max_inline_bytes=max_inline_bytes,
            allow_insecure=lambda _url: True,
        )
        return [
            url for url in kept
            if not (is_http_or_https_url(url) and not names_an_owui_file_path(url))
        ], refused + [
            (url, "could not be fetched, so it was not sent", "remote_unfetched")
            for url in kept
            if is_http_or_https_url(url) and not names_an_owui_file_path(url)
        ]
    kept, refused = _tool_picture_gate(
        pictures,
        max_inline_bytes=max_inline_bytes,
        allow_insecure=pipe._multimodal_handler._is_insecure_http_allowed,
    )
    admitted, unfetchable = await _tool_picture_address_gate(
        pipe, kept, seen=seen, deadline=deadline,
    )
    refused.extend(unfetchable)
    return admitted, refused


def _tool_picture_notice(refusals: list[tuple[str, str, str]]) -> str:
    return "Images: skipped {count} ({reasons}).".format(
        count=len(refusals),
        reasons="; ".join(reason for _url, reason, _cause in refusals),
    )


def _tool_images_message(
    pictures: list[str], *, lead_in: bool = False,
) -> dict[str, Any] | None:
    if not pictures and not lead_in:
        return None
    return {"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": OPEN_WEBUI_TOOL_IMAGES_TEXT},
        *({"type": "input_image", "image_url": url, "detail": "auto"} for url in pictures),
    ]}


_TEMPORARY_CHAT_LOG_SUBJECT = "<temporary chat>"


def _chat_log_subject(chat_id: Any) -> Any:
    return _TEMPORARY_CHAT_LOG_SUBJECT if is_temporary_chat(chat_id) else chat_id


def _memo_owner_key(user_obj: Any | None) -> str | None:
    raw = getattr(user_obj, "id", None)
    return raw.strip() if isinstance(raw, str) and raw.strip() else None


def _note_memo_use(
    memo_key: Any,
    remembered: tuple[str | None, bytes, str] | None,
    *,
    mode: str,
    temporary_chat: bool,
) -> None:
    if mode != "reuse" or temporary_chat or memo_key is None:
        return
    if remembered is not None:
        if memo_key in _reuse_download_memo:
            _reuse_download_memo.move_to_end(memo_key)
        return
    if memo_key in _reuse_download_memo:
        _reuse_download_memo.move_to_end(memo_key)


def _remaining_address_seconds(deadline: float) -> float:
    return min(ADDRESS_CHECK_SECONDS, deadline - time.monotonic())


async def _effective_remote_bytes(pipe: Pipe, seen: list[int | None] | None) -> int:
    if seen is not None and seen[0] is not None:
        return seen[0]
    limit = await pipe._multimodal_handler._get_effective_remote_file_limit_mb() * 1024 * 1024
    if seen is not None:
        seen[0] = limit
    return limit


async def _memo_hit_is_still_permitted(
    pipe: Pipe,
    memo_key: Any,
    url: str,
    seen: dict[str, bool | None] | None = None,
    deadline: float | None = None,
    *,
    payload_bytes: int | None = None,
    size_seen: list[int | None] | None = None,
) -> bool:
    if payload_bytes is not None and payload_bytes > await _effective_remote_bytes(pipe, size_seen):
        _reuse_download_memo.pop(memo_key, None)
        return False
    if seen is not None and url in seen:
        permitted = seen[url]
    else:
        if deadline is None:
            deadline = time.monotonic() + ADDRESS_CHECK_BUDGET_SECONDS
        permitted = await pipe._multimodal_handler._is_safe_url(
            url, seconds=_remaining_address_seconds(deadline),
        )
        if seen is not None:
            seen[url] = permitted
    if permitted is False:
        _reuse_download_memo.pop(memo_key, None)
        return False
    return True


async def transform_messages_to_input(
    pipe: Pipe,
    messages: list[dict[str, Any]],
    chat_id: str | None = None,
    openwebui_model_id: str | None = None,
    artifact_loader: Callable[
        [str | None, str | None, list[str]],
        Awaitable[dict[str, dict[str, Any]] | tuple[dict[str, dict[str, Any]], dict[str, str]]],
    ]
    | None = None,
    pruning_turns: int = 0,
    replayed_reasoning_refs: list[tuple[str, str]] | None = None,
    user_obj: Any | None = None,
    event_emitter: Callable | None = None,
    *,
    model_id: str | None = None,
    valves: Pipe.Valves | None = None,
    capability_model_id: str | None = None,
    ask_user_names: frozenset[str] = frozenset(),
    attachment_notices: list[str] | None = None,
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


    async with remote_file_limit_scope():

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
        window_armed_at: set[int] = set()
        lift_candidates: list[tuple[dict[str, Any], list[tuple[str, int, int]]]] = []
        pictures_in_request: set[str] = set()
        _deferred_tool_pictures: list[str] = []
        _deferred_tool_refusals: list[tuple[str, str, str]] = []

        def _message_identifier(entry: dict[str, Any]) -> str | None:
            """Return the most specific identifier available on ``entry``."""
            for key in ("id", "_id", "message_id"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return None

        async def _flush_deferred_tool_pictures() -> None:
            had_pictures = bool(_deferred_tool_pictures)
            if _deferred_tool_pictures:
                admitted, gated_refusals = await _gate_inline_tool_pictures(
                    _deferred_tool_pictures, max_inline_bytes,
                    allow_insecure=pipe._multimodal_handler._is_insecure_http_allowed,
                )
                _deferred_tool_refusals.extend(gated_refusals)
                _deferred_tool_pictures.clear()
                message = _tool_images_message(admitted, lead_in=had_pictures)
                if message is not None:
                    openai_input.append(message)
            if _deferred_tool_refusals:
                for url, reason, refusal_cause in _deferred_tool_refusals:
                    pipe.logger.warning(
                        "Not forwarding a tool's picture (%s): %s [cause=%s]",
                        loggable_link(url), reason, refusal_cause,
                    )
                await pipe._event_emitter_handler._emit_status(
                    event_emitter, _tool_picture_notice(_deferred_tool_refusals), done=False,
                )
                _deferred_tool_refusals.clear()

        def _compute_turn_indices() -> tuple[list[int | None], int]:
            """Label each message with a turn index and return the total count."""
            indices: list[int | None] = []
            current_turn = -1
            max_turn = -1
            last_dialog_role: str | None = None

            for position, msg in enumerate(messages):
                role = (msg.get("role") or "").lower()
                turn_idx: int | None = None

                if role == "user" and position and (
                    is_tool_image_handoff(messages[position - 1], msg)
                    or _handoff_back(messages, position)
                ):
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
            and not (
                position
                and (
                    is_tool_image_handoff(messages[position - 1], message)
                    or _handoff_back(messages, position)
                )
            )
        ]
        last_person_position = current_turn_people[-1] if current_turn_people else -1
        tool_handoff_positions = [
            position
            for position, message in enumerate(messages)
            if (message.get("role") or "").lower() == "user"
            and position
            and (
                is_tool_image_handoff(messages[position - 1], message)
                or _handoff_back(messages, position)
            )
        ]
        last_tool_handoff_index = tool_handoff_positions[-1] if tool_handoff_positions else -1
        person_images_this_turn = False
        turn_images_used = 0
        turn_images_dropped = 0
        turn_images_index: int | None = None
        temporary_chat = is_temporary_chat(chat_id)
        memo_owner = _memo_owner_key(user_obj)
        request_memo: dict[tuple[str, str], tuple[str | None, bytes, str]] = {}

        async def _remote_limit_bytes() -> int:
            return (
                await pipe._multimodal_handler._get_effective_remote_file_limit_mb()
                * 1024
                * 1024
            )

        address_verdicts: dict[str, bool | None] = {}
        reuse_limit_seen: list[int | None] = [None]
        tool_name_at, issuer_at = _tool_names_by_position(messages)

        def _withheld(turn_index: int | None) -> bool:
            return not active_valves.PERSIST_TOOL_RESULTS and turn_index is not None and turn_index < total_turns - 1

        def _tool_round_withheld(turn_index: int | None) -> bool:
            if active_valves.PERSIST_TOOL_RESULTS:
                return False
            return turn_index is None or _withheld(turn_index)

        prune_before_turn: int | None = None
        if pruning_turns > 0 and total_turns > pruning_turns:
            prune_before_turn = total_turns - pruning_turns

        def _sanitize_free_text(text: str) -> str:
            """Strip hidden transport markers from non-assistant free text."""
            return strip_hidden_marker_lines(text)

        message_texts: dict[int, str] = {}
        marker_spans_by_text: dict[int, list[dict[str, Any]]] = {}
        marker_segments_by_text: dict[int, list[dict[str, Any]]] = {}

        def _message_text(index: int, entry: dict[str, Any]) -> str:
            text = message_texts.get(index)
            if text is None:
                entry_content = entry.get("content", "")
                text = (
                    entry_content
                    if isinstance(entry_content, str)
                    else _extract_plain_text_content(entry_content)
                )
                message_texts[index] = text
            return text

        def _marker_spans_for(text: str) -> list[dict[str, Any]]:
            if _MARKER_SUFFIX not in text:
                return []
            key = id(text)
            spans = marker_spans_by_text.get(key)
            if spans is None:
                spans = _iter_marker_spans(text)
                marker_spans_by_text[key] = spans
            return spans

        def _marker_segments_for(text: str, spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
            key = id(text)
            segments = marker_segments_by_text.get(key)
            if segments is None:
                segments = split_text_by_markers(text, spans)
                marker_segments_by_text[key] = segments
            return segments

        artifact_groups: dict[str | None, dict[str, dict]] = {}
        artifact_producers: dict[str | None, dict[str, str]] = {}
        if artifact_loader and chat_id and openwebui_model_id:
            wanted_by_group: dict[str | None, list[str]] = {}
            for entry_index, entry in enumerate(messages):
                entry_role = (entry.get("role") or "").lower()
                if entry_role in {"tool", "user"}:
                    continue
                entry_text = _message_text(entry_index, entry)
                entry_spans = _marker_spans_for(entry_text)
                if not entry_spans:
                    continue
                group_id = entry.get("message_id") or _message_identifier(entry)
                group_markers = wanted_by_group.setdefault(group_id, [])
                for segment in _marker_segments_for(entry_text, entry_spans):
                    if segment.get("type") == "marker" and segment["marker"] not in group_markers:
                        group_markers.append(segment["marker"])
            async def _load_group(
                gate: asyncio.Semaphore,
                load_group_id: str | None,
                load_group_markers: list[str],
            ) -> tuple[str | None, dict[str, dict], dict[str, str]]:
                async with gate:
                    try:
                        loaded = await artifact_loader(
                            chat_id, load_group_id, load_group_markers
                        )
                        if isinstance(loaded, tuple):
                            payloads, producers = loaded
                        else:
                            payloads, producers = loaded, {}
                        return load_group_id, payloads, producers
                    except Exception:
                        logger.warning("Artifact loader failed for chat_id=%s message_id=%s", _chat_log_subject(chat_id), load_group_id, exc_info=True)
                        return load_group_id, {}, {}

            pending_groups = [
                (group_id, group_markers)
                for group_id, group_markers in wanted_by_group.items()
                if group_markers
            ]
            if pending_groups:
                gate = asyncio.Semaphore(_ARTIFACT_GROUP_CONCURRENCY)
                for group_id, loaded, producers in await asyncio.gather(
                    *(_load_group(gate, gid, markers) for gid, markers in pending_groups)
                ):
                    artifact_groups[group_id] = loaded
                    artifact_producers[group_id] = producers

        _tool_context = pipe._TOOL_CONTEXT.get()
        _context_deadline = _tool_context.address_deadline if _tool_context is not None else None
        address_deadline = (
            _context_deadline if _context_deadline is not None
            else time.monotonic() + ADDRESS_CHECK_BUDGET_SECONDS
        )
        normalized_rows: dict[int, Any] = {}

        def _normalized_row(row: Any) -> Any:
            key = id(row)
            if key not in normalized_rows:
                normalized_rows[key] = normalize_persisted_item(row)
            return normalized_rows[key]

        recorded_rounds: set[tuple[str, str]] = set()
        builtin_rounds: set[tuple[str, str]] = set()
        for entry_index, entry in enumerate(messages):
            if (entry.get("role") or "").lower() != "assistant":
                continue
            entry_text = _message_text(entry_index, entry)
            entry_spans = _marker_spans_for(entry_text)
            if not entry_spans:
                continue
            group_id = entry.get("message_id") or _message_identifier(entry)
            batch = artifact_groups.get(group_id) or {}
            for segment in _marker_segments_for(entry_text, entry_spans):
                if segment.get("type") != "marker":
                    continue
                payload = batch.get(segment["marker"])
                if payload is None:
                    continue
                stored = _normalized_row(payload)
                if not isinstance(stored, dict) or stored.get("type") != "function_call":
                    continue
                if BUILTIN_ASK_USER_ROUND_KEY not in stored:
                    continue
                call_id = str(stored.get("call_id") or "")
                if not call_id:
                    continue
                round_key = (call_id, str(stored.get("name") or ""))
                if stored.get(BUILTIN_ASK_USER_ROUND_KEY):
                    builtin_rounds.add(round_key)
                else:
                    recorded_rounds.add(round_key)
        stamped_ask_user_rounds = frozenset(builtin_rounds)
        recorded_ask_user_rounds = frozenset(recorded_rounds)

        missing_artifact_markers: list[str] = []
        deferred_vision_skips = 0
        for idx, msg in enumerate(messages):
            raw_role = msg.get("role")
            role = (raw_role or "").lower()
            raw_content = msg.get("content", "")
            msg_id = msg.get("message_id") or _message_identifier(msg)
            msg_turn_index = turn_indices[idx]
            if msg_turn_index != turn_images_index:
                turn_images_used = 0
                turn_images_dropped = 0
                turn_images_index = msg_turn_index
            raw_tool_calls = msg.get("tool_calls")
            msg_tool_calls: list[dict[str, Any]] = (
                list(raw_tool_calls)
                if isinstance(raw_tool_calls, list) and raw_tool_calls
                else []
            )

            if role != "tool" and (_deferred_tool_pictures or _deferred_tool_refusals):
                await _flush_deferred_tool_pictures()

            if role in {"system", "developer"}:
                blocks: list[dict[str, Any]] = []

                if isinstance(raw_content, str):
                    cleaned = _sanitize_free_text(raw_content)
                    if cleaned:
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

                round_name = tool_name_at[idx] or ""
                issuer = issuer_at[idx]
                round_exempt = _round_keeps_its_ask_user_answer(
                    call_id, round_name, ask_user_names, recorded_ask_user_rounds, stamped_ask_user_rounds
                )
                if not round_exempt and not (
                    issuer >= 0 and issuer in window_armed_at and not is_picture_output(raw_content)
                ):
                    last_image_blocks, last_image_turn = [], None

                tool_content = raw_content
                tool_pictures: list[str] = []
                if tool_content is None:
                    tool_content_text = ""
                elif is_picture_output(tool_content):
                    tool_content_text, tool_pictures = tool_output_text_and_pictures(tool_content)
                elif is_text_part_output(tool_content):
                    tool_content_text = tool_output_text_and_pictures(tool_content)[0]
                elif isinstance(tool_content, str):
                    tool_content_text = tool_content
                else:
                    try:
                        tool_content_text = json.dumps(tool_content, ensure_ascii=False)
                    except (TypeError, ValueError):
                        tool_content_text = str(tool_content)

                if is_picture_output(tool_content):
                    last_image_blocks, last_image_turn = [], None

                if _tool_round_withheld(msg_turn_index):
                    if not round_exempt:
                        tool_content_text = unretained_tool_result(_tool_result_failed(tool_content_text))
                    tool_pictures = []

                tool_item: dict[str, Any] = {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": tool_content_text,
                }
                if not round_exempt and pruning_turns > 0 and _is_old_turn(msg_turn_index, threshold=prune_before_turn):
                    _prune_tool_output(tool_item, marker=None, turn_index=msg_turn_index, retention_turns=pruning_turns)
                openai_input.append(tool_item)
                if tool_pictures and not _handoff_ahead(messages, idx):
                    admitted, tool_refusals = await _tool_picture_gate_with_address(
                        pipe, tool_pictures, max_inline_bytes=max_inline_bytes,
                        seen=address_verdicts, deadline=address_deadline,
                    )
                    _deferred_tool_refusals.extend(tool_refusals)
                    _deferred_tool_pictures.extend(admitted)
                continue

            if role == "user":
                tool_images = bool(idx) and (
                    is_tool_image_handoff(messages[idx - 1], msg)
                    or _handoff_back(messages, idx)
                )
                if tool_images:
                    last_image_blocks, last_image_turn = [], None
                if tool_images and _tool_round_withheld(msg_turn_index):
                    continue
                raw_content_value = msg.get("content")
                content_blocks = raw_content_value or []
                carried_blocks = (isinstance(raw_content_value, str) and bool(raw_content_value)) or isinstance(raw_content_value, (dict, list))
                if isinstance(content_blocks, str):
                    cleaned = _sanitize_free_text(content_blocks)
                    content_blocks = [{"type": "text", "text": cleaned}] if cleaned else []
                elif isinstance(content_blocks, dict):
                    text_val = content_blocks.get("text")
                    if not isinstance(text_val, str):
                        text_val = content_blocks.get("content")
                    cleaned = _sanitize_free_text(text_val) if isinstance(text_val, str) else ""
                    if content_blocks.get("type") in _MEDIA_BLOCK_TYPES:
                        content_blocks = [
                            *([{"type": "text", "text": cleaned}] if cleaned else []),
                            content_blocks,
                        ]
                    else:
                        content_blocks = [{"type": "text", "text": cleaned}] if cleaned else []
                elif not isinstance(content_blocks, list):
                    pipe.logger.warning(
                        "Ignoring user message %s: content is %s, expected a string, list or dict",
                        msg_id,
                        type(content_blocks).__name__,
                    )
                    content_blocks = []

                def _image_subject(source: str) -> str:
                    return loggable_link(source)

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
                            image_block_detail = block.get("detail")
                            if isinstance(image_block_detail, str):
                                detail = image_block_detail
                            url = block.get("url", "") if isinstance(block.get("url"), str) else ""

                        if not url:
                            return None

                        owui_internal = names_an_owui_file_path(url)
                        held_len: int | None = None
                        held_over_cap = False

                        if (
                            is_cleartext_http_url(url)
                            and not owui_internal
                            and not pipe._multimodal_handler._is_insecure_http_allowed(url)
                        ):
                                return ImageRefusal(
                                    "served over plain HTTP, which is blocked by security policy; "
                                    "set ALLOW_INSECURE_HTTP and list the host in "
                                    "ALLOW_INSECURE_HTTP_HOSTS to permit it",
                                    "insecure_http",
                                    subject=loggable_link(url),
                                )

                        if not (
                            url_scheme(url) in ("data", "http", "https")
                            or owui_internal
                        ):
                            return ImageRefusal(
                                "not a link the pipe can resolve into an image",
                                "unusable_link",
                                subject=loggable_link(url),
                            )

                        gated_split: tuple[str, str] | None = None
                        if is_inline_data_url(url):
                            try:
                                url, gated_split, refusal = await _gate_inline_data_url(
                                    url, max_inline_bytes, resolve_type=False,
                                )
                                if refusal is not None:
                                    return refusal
                            except Exception as exc:
                                pipe.logger.exception("Failed to process base64 image")
                                await pipe._ensure_error_formatter()._emit_error(
                                    event_emitter,
                                    f"Failed to process base64 image: {exc}",
                                    show_error_message=False
                                )
                                return ImageRefusal(
                                    f"could not be processed: {exc}", "base64_processing_error",
                                    subject=loggable_link(url),
                                )

                        elif is_http_or_https_url(url) and not owui_internal:
                            memo_key = (chat_id, url) if chat_id else None
                            remembered = (
                                request_memo.get(memo_key)
                                if mode == "reuse" and memo_key is not None
                                else None
                            )
                            if mode == "reuse" and not temporary_chat and remembered is None:
                                remembered = (
                                    _reuse_download_memo.get(memo_key)
                                    if memo_key is not None
                                    else None
                                )
                            if remembered is not None and remembered[0] != memo_owner:
                                remembered = None
                            _note_memo_use(
                                memo_key, remembered,
                                mode=mode, temporary_chat=temporary_chat,
                            )
                            held_len = len(remembered[1]) if remembered is not None else None
                            held_over_cap = (
                                held_len is not None
                                and memo_key is not None
                                and held_len > await _remote_limit_bytes()
                            )
                            if remembered is not None and not await (
                                _memo_hit_is_still_permitted(
                                    pipe, memo_key, url, address_verdicts, address_deadline,
                                    payload_bytes=len(remembered[1]), size_seen=reuse_limit_seen,
                                )
                            ):
                                remembered = None
                            try:
                                downloaded = (
                                    {"data": remembered[1], "mime_type": remembered[2]}
                                    if remembered is not None
                                    else await pipe._multimodal_handler._download_remote_url(
                                        url,
                                        seconds=_remaining_address_seconds(address_deadline),
                                    )
                                )
                            except Exception:
                                pipe.logger.exception("Failed to download remote image %s", loggable_link(url))
                                downloaded = None
                            if downloaded and not downloaded.get("data"):
                                downloaded = None
                            if not downloaded and not (
                                _cold_verdict := address_verdicts[url]
                                if url in address_verdicts
                                else address_verdicts.setdefault(
                                    url,
                                    await pipe._multimodal_handler._is_safe_url(
                                        url,
                                        seconds=_remaining_address_seconds(address_deadline),
                                    ),
                                )
                            ):
                                return _refuse(
                                    "could not be fetched, so it was not sent",
                                    "remote_unfetched",
                                    subject=loggable_link(url),
                                )
                            if downloaded:
                                oversized = len(downloaded["data"]) > max_inline_bytes
                                if oversized:
                                    return _refuse(
                                        f"{len(downloaded['data'])} bytes, over the "
                                        f"{max_inline_bytes}-byte limit, so it was not sent",
                                        "oversized_remote",
                                        subject=loggable_link(url),
                                    )
                                declared_type = str(downloaded.get("mime_type") or "").split(";", 1)[0].strip().lower()
                                resolved_type = resolve_download_type(
                                    declared_type,
                                    _sniff_evidence(bytes(downloaded["data"][:_SNIFF_PREFIX_BYTES])),
                                )
                                if not resolved_type.startswith("image/"):
                                    return _refuse(
                                        "not identifiable as an image",
                                        "inline_untyped",
                                        subject=loggable_link(url),
                                    )
                                if mode == "reuse" and memo_key is not None:
                                    request_memo[memo_key] = (
                                        memo_owner,
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
                                    held = _reuse_download_memo.held
                                    while (
                                        _reuse_download_memo
                                        and held + len(downloaded["data"])
                                        > _REUSE_DOWNLOAD_MEMO_MAX_BYTES
                                    ):
                                        _reuse_download_memo.popitem(last=False)
                                        held = _reuse_download_memo.held
                                    _reuse_download_memo[memo_key] = (
                                        memo_owner,
                                        downloaded["data"],
                                        downloaded.get("mime_type") or "",
                                    )
                                url = f"data:{resolved_type};base64," + await asyncio.to_thread(
                                    _b64encode_ascii, downloaded["data"]
                                )
                        owui_file_id = extract_internal_file_id(url) if owui_internal else None
                        if owui_internal and not owui_file_id:
                            raise RequiredInternalFileError(
                                "An image Open WebUI is serving cannot be read, so it was not sent.",
                                kind="image",
                            )

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

                        split = (
                            gated_split
                            if gated_split is not None
                            else split_base64_data_url(url)
                        )
                        if mode == "reuse" and not (split[1] if split is not None else ""):
                            if is_http_or_https_url(url):
                                return _refuse(
                                    f"{held_len} bytes, over the "
                                    f"{await _remote_limit_bytes()}-byte download limit, "
                                    "so it was not sent"
                                    if held_over_cap
                                    else "could not be fetched, so it was not sent",
                                    "oversized_remote" if held_over_cap else "remote_unfetched",
                                    subject=_image_subject(url),
                                )
                            return ImageRefusal(
                                "could not be fetched, so its type could not be established",
                                "reuse_unfetched",
                                subject=_image_subject(url),
                            )

                        if split is not None and split[1]:
                            url, refusal = _resolve_inline_type(
                                split[0],
                                split[1],
                                cause="reuse_untyped" if mode == "reuse" else "inline_untyped",
                                subject=loggable_link(url),
                                split_from=url,
                            )
                            if refusal is not None:
                                return refusal

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

                def _no_source_refusal() -> ImageRefusal:
                    return ImageRefusal("has no readable source", "no_source")

                async def _to_input_file(block: dict) -> dict | ImageRefusal | None:
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

                        if (
                            isinstance(file_url, str)
                            and file_url.strip()
                            and names_an_owui_file_path(file_url.strip())
                        ):
                            extracted = extract_internal_file_id(file_url.strip())
                            if extracted:
                                file_id = extracted
                                file_url = None
                        if (
                            isinstance(file_data, str)
                            and file_data.strip()
                            and names_an_owui_file_path(file_data.strip())
                            and url_scheme(file_data.strip()) != "data"
                        ):
                            extracted = extract_internal_file_id(file_data.strip())
                            if extracted:
                                file_id = extracted
                                file_data = None
                        if (
                            isinstance(file_url, str)
                            and file_url.strip()
                            and names_an_owui_file_path(file_url.strip())
                        ):
                            if not file_id:
                                raise RequiredInternalFileError(
                                    "A file Open WebUI is serving cannot be read, so it was not sent.",
                                    kind="file",
                                )
                            file_url = None
                        if (
                            isinstance(file_data, str)
                            and file_data.strip()
                            and names_an_owui_file_path(file_data.strip())
                            and url_scheme(file_data.strip()) != "data"
                        ):
                            if not file_id:
                                raise RequiredInternalFileError(
                                    "A file Open WebUI is serving cannot be read, so it was not sent.",
                                    kind="file",
                                )
                            file_data = None

                        if (
                            isinstance(file_data, str)
                            and is_cleartext_http_url(file_data)
                            and not names_an_owui_file_path(file_data)
                            and not pipe._multimodal_handler._is_insecure_http_allowed(file_data)
                        ):
                            pipe.logger.error(
                                "Blocked insecure HTTP file_data URL by default (blocked by security policy): %s",
                                loggable_link(file_data),
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
                            and not names_an_owui_file_path(file_url)
                            and not pipe._multimodal_handler._is_insecure_http_allowed(file_url)
                        ):
                            pipe.logger.error(
                                "Blocked insecure HTTP file_url by default (blocked by security policy): %s",
                                loggable_link(file_url),
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

                        def _is_inline_payload(name: str, value: str) -> bool:
                            scheme = url_scheme(value)
                            return scheme == "data" or (name == "file_data" and not scheme)

                        def _is_provider_fetched_link(name: str, value: Any) -> bool:
                            if not isinstance(value, str) or not value:
                                return False
                            if _is_inline_payload(name, value):
                                return False
                            return bool(url_scheme(value)) or (
                                name == "file_url" and is_absolute_url(value)
                            )

                        _gate_fields = (("file_data", file_data), ("file_url", file_url))
                        _scheme_verdicts: dict[str, bool | None] = {}
                        for _name, _value in _gate_fields:
                            if not isinstance(_value, str) or not _is_provider_fetched_link(
                                _name, _value
                            ):
                                continue
                            _gate_memo = (
                                address_verdicts
                                if is_http_or_https_url(_value)
                                else _scheme_verdicts
                            )
                            if not (
                                _file_verdict := _gate_memo[_value]
                                if _value in _gate_memo
                                else _gate_memo.setdefault(
                                    _value,
                                    await pipe._multimodal_handler._is_safe_url(
                                        _value,
                                        seconds=_remaining_address_seconds(address_deadline),
                                    ),
                                )
                            ):
                                if _file_verdict is None and is_http_or_https_url(_value):
                                    _reason = (
                                        "not checked against the address policy in time, so it "
                                        "was not sent"
                                    )
                                    _cause = "uncheckable_file_url"
                                    pipe.logger.warning(
                                        "Address check for %s file link %s reached no verdict "
                                        "within the request's address budget, so the link was "
                                        "not sent",
                                        _name, loggable_link(_value),
                                    )
                                elif not is_http_or_https_url(_value):
                                    _reason = (
                                        "served from a link that is neither http nor https, "
                                        "which is blocked by security policy"
                                    )
                                    _cause = "unsupported_scheme_file"
                                    pipe.logger.error(
                                        "Blocked %s file link by security policy: %s",
                                        _name, loggable_link(_value),
                                    )
                                else:
                                    _reason = (
                                        "served from a link on a private, link-local or "
                                        "unresolvable address, which is blocked by security "
                                        "policy; set ENABLE_SSRF_PROTECTION to False to permit it"
                                    )
                                    _cause = "private_network_file"
                                    pipe.logger.error(
                                        "Blocked %s file link by security policy: %s",
                                        _name, loggable_link(_value),
                                    )
                                if not file_id:
                                    return ImageRefusal(_reason, _cause, subject=_name)
                                if _name == "file_data":
                                    file_data = None
                                else:
                                    file_url = None

                        oversized = {
                            name: value
                            for name, value in (("file_data", file_data), ("file_url", file_url))
                            if isinstance(value, str)
                            and value
                            and _is_inline_payload(name, value)
                            and _inline_payload_bytes(value) > max_inline_bytes
                        }
                        if oversized and not file_id:
                            return ImageRefusal(
                                f"larger than the {max_inline_bytes}-byte inline limit",
                                "oversized_inline_file",
                                subject=link_media_type(next(iter(oversized.values()))),
                            )
                        if "file_data" in oversized:
                            file_data = None
                        if "file_url" in oversized:
                            file_url = None

                        if not (file_id or file_data or file_url):
                            return _no_source_refusal()

                        if file_id:
                            result["file_id"] = file_id
                        if file_data:
                            result["file_data"] = file_data
                        if filename:
                            result["filename"] = filename
                        if file_url:
                            result["file_url"] = file_url

                        return result

                    except RequiredInternalFileError:
                        raise
                    except Exception as exc:
                        pipe.logger.exception("Error in _to_input_file")
                        await pipe._ensure_error_formatter()._emit_error(
                            event_emitter,
                            f"File processing error: {exc}",
                            show_error_message=False
                        )
                        return _no_source_refusal()

                async def _to_input_audio(block: dict) -> dict | ImageRefusal | None:
                    """Convert Open WebUI audio blocks into Responses API format.

                    Handles audio content blocks, transforming various input formats into
                    the Responses API audio input format.

                    Responses API Audio Input Format (per OpenAPI spec):
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": "<base64_audio_data>",
                            }
                        }

                    OpenRouter Audio Requirements (per documentation):
                        - Audio must be base64-encoded (URLs NOT supported)
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

                    Note:
                        All errors are caught and logged with status emissions.
                        Failed processing returns minimal valid block rather than crashing.
                    """
                    def _refuse_audio(why: str, cause: str) -> ImageRefusal:
                        return ImageRefusal(why, cause, subject="audio")

                    async def _refuse_oversized_inline(estimate: int) -> ImageRefusal:
                        pipe.logger.warning(
                            "Audio payload rejected: ~%.1fMB is over the %dMB inline limit.",
                            estimate / (1024 * 1024),
                            max_inline_bytes // (1024 * 1024),
                        )
                        await pipe._ensure_error_formatter()._emit_error(
                            event_emitter,
                            f"Audio input is larger than the {max_inline_bytes}-byte inline limit and was not sent.",
                            show_error_message=True,
                        )
                        return _oversized_inline_refusal(
                            max_inline_bytes, "oversized_inline_audio",
                            "an attached audio clip",
                        )

                    async def _normalize_base64(data: str) -> str | None:
                        if not data:
                            return None
                        stripped = data.strip()
                        if stripped[:5].lower() == "data:":
                            parsed = await asyncio.to_thread(
                                pipe._multimodal_handler._parse_data_url, stripped
                            )
                            if not parsed:
                                return None
                            cleaned = "".join(str(parsed.get("b64") or "").split())
                        else:
                            cleaned = "".join(stripped.split())
                        if not cleaned:
                            return None
                        if not await asyncio.to_thread(_is_well_formed_base64, cleaned):
                            return None
                        return cleaned

                    def _resolved_mime_hint(payload: dict[str, Any] | None = None) -> str | None:
                        candidates: list[Any] = []
                        if isinstance(payload, dict):
                            head = payload.get("data")
                            if isinstance(head, str) and is_inline_data_url(head.strip()):
                                candidates.append(link_media_type(head.strip()))
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
                            if isinstance(_data, str) and (_data_bytes := _inline_payload_bytes(_data)) > max_inline_bytes:
                                return await _refuse_oversized_inline(_data_bytes)
                            cleaned = await _normalize_base64(_data)
                            if not cleaned:
                                pipe.logger.warning("Audio payload rejected: invalid base64 data.")
                                return _refuse_audio(
                                    "an audio clip was not valid base64",
                                    "audio_not_base64",
                                )
                            declared = link_media_type(_data.strip()) if is_inline_data_url(_data.strip()) else ""
                            if declared:
                                audio_format = _map_audio_format(declared)
                                if audio_format is None:
                                    pipe.logger.warning(
                                        "Audio payload rejected: mime type %r.",
                                        declared,
                                    )
                                    return _refuse_audio(
                                        "an audio clip was in a format the pipe will not rename",
                                        "audio_unsupported_format",
                                    )
                                return _build_audio_block(cleaned, audio_format)
                            hint = _resolved_mime_hint(audio_payload)
                            audio_format = _normalize_audio_format(audio_payload.get("format"), hint)
                            if audio_format is None:
                                pipe.logger.warning(
                                    "Audio payload rejected: format %r, mime hint %r.",
                                    audio_payload.get("format"),
                                    hint,
                                )
                                return _refuse_audio(
                                    "an audio clip was in a format the pipe will not rename",
                                    "audio_unsupported_format",
                                )
                            return _build_audio_block(cleaned, audio_format)

                        if isinstance(audio_payload, dict):
                            raw_data = audio_payload.get("data")
                            if isinstance(raw_data, str):
                                raw_bytes = _inline_payload_bytes(raw_data)
                                if raw_bytes > max_inline_bytes:
                                    return await _refuse_oversized_inline(raw_bytes)
                                cleaned = await _normalize_base64(raw_data)
                                if not cleaned:
                                    pipe.logger.warning("Audio payload rejected: invalid base64 data.")
                                    return _refuse_audio(
                                        "an audio clip was not valid base64",
                                        "audio_not_base64",
                                    )
                                mime_hint = _resolved_mime_hint(audio_payload)
                                audio_format = _normalize_audio_format(
                                    audio_payload.get("format"), mime_hint
                                )
                                if audio_format is None:
                                    pipe.logger.warning(
                                        "Audio payload rejected: format %r, mime hint %r.",
                                        audio_payload.get("format"),
                                        mime_hint,
                                    )
                                    return _refuse_audio(
                                        "an audio clip was in a format the pipe will not rename",
                                        "audio_unsupported_format",
                                    )
                                return _build_audio_block(cleaned, audio_format)

                        if isinstance(audio_payload, str):
                            sanitized = audio_payload.strip()
                            if is_http_or_https_url(sanitized):
                                pipe.logger.warning("Audio payload rejected: remote URLs are not supported.")
                                return _refuse_audio(
                                    _AUDIO_URL_REFUSAL,
                                    "audio_remote_url",
                                )

                            sanitized_bytes = _inline_payload_bytes(sanitized)
                            if sanitized_bytes > max_inline_bytes:
                                return await _refuse_oversized_inline(sanitized_bytes)

                            if sanitized[:5].lower() == "data:":
                                parsed = await asyncio.to_thread(pipe._multimodal_handler._parse_data_url, sanitized if sanitized.startswith("data:") else f"data:{sanitized.split(':', 1)[1]}")
                                if not parsed or not parsed.get("mime_type", "").startswith("audio/"):
                                    pipe.logger.warning("Audio payload rejected: invalid data URL.")
                                    return _refuse_audio(
                                        "an audio clip was not an audio data URL",
                                        "audio_bad_data_url",
                                    )
                                audio_format = _map_audio_format(parsed.get("mime_type"))
                                audio_b64 = "".join(str(parsed.get("b64", "")).split())
                                if not audio_b64:
                                    pipe.logger.warning("Audio payload rejected: invalid base64 data.")
                                    return _refuse_audio(
                                        "an audio clip was not valid base64",
                                        "audio_not_base64",
                                    )
                                if audio_format is None:
                                    pipe.logger.warning(
                                        "Audio payload rejected: mime type %r.",
                                        parsed.get("mime_type"),
                                    )
                                    return _refuse_audio(
                                        "an audio clip was in a format the pipe will not rename",
                                        "audio_unsupported_format",
                                    )
                                return _build_audio_block(audio_b64, audio_format)

                            cleaned = await _normalize_base64(sanitized)
                            if not cleaned:
                                pipe.logger.warning("Audio payload rejected: invalid base64 data.")
                                return _refuse_audio(
                                    "an audio clip was not valid base64",
                                    "audio_not_base64",
                                )

                            mime_type = _resolved_mime_hint()
                            audio_format = _map_audio_format(mime_type)
                            if audio_format is None:
                                pipe.logger.warning(
                                    "Audio payload rejected: mime hint %r.", mime_type
                                )
                                return _refuse_audio(
                                    "an audio clip was in a format the pipe will not rename",
                                    "audio_unsupported_format",
                                )
                            return _build_audio_block(cleaned, audio_format)

                        # Invalid/empty
                        pipe.logger.warning("Invalid audio payload format, refusing the audio block")
                        return _refuse_audio(
                            "an audio block carried no audio data",
                            "audio_no_payload",
                        )

                    except Exception as exc:
                        pipe.logger.exception("Error in _to_input_audio")
                        await pipe._ensure_error_formatter()._emit_error(
                            event_emitter,
                            f"Audio processing error: {exc}",
                            show_error_message=False,
                        )
                        return _refuse_audio(
                            "an audio clip could not be read", "audio_conversion_failed",
                        )

                async def _to_input_video(block: dict) -> dict | ImageRefusal | None:
                    """Convert Open WebUI video blocks into Chat Completions video format.

                    Video Support by Provider (per OpenRouter docs):
                        - Gemini AI Studio: YouTube links only
                        - Most providers: Limited or no video support
                        - Check model's input_modalities for "video" capability

                    Supported Video Formats:
                        - Remote URLs: YouTube links, direct video URLs
                        - Data URLs: data:video/mp4;base64,... (rarely used due to size)

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
                            return ImageRefusal(
                                "a video block carried no video URL",
                                "video_no_url",
                                subject="video",
                            )

                        if not is_inline_data_url(url) and names_an_owui_file_path(url):
                            raise RequiredInternalFileError(
                                "A video link whose path names this Open WebUI's own file "
                                "endpoint, which a provider cannot fetch and the pipe will "
                                "not forward.",
                                kind="video",
                            )

                        if (
                            is_cleartext_http_url(url)
                            and not pipe._multimodal_handler._is_insecure_http_allowed(url)
                        ):
                            pipe.logger.error("Blocked insecure HTTP video URL by default: %s", loggable_link(url))
                            await pipe._ensure_error_formatter()._emit_error(
                                event_emitter,
                                "Video URL blocked by security policy (HTTP disabled by default). "
                                "Enable ALLOW_INSECURE_HTTP + ALLOW_INSECURE_HTTP_HOSTS to allow specific hosts.",
                                show_error_message=True,
                            )
                            return ImageRefusal(
                                "served over plain HTTP, which is blocked by security "
                                "policy; use an https link, or set ALLOW_INSECURE_HTTP "
                                "and list the host in ALLOW_INSECURE_HTTP_HOSTS",
                                "insecure_http_video",
                                severity="error",
                                subject=loggable_link(url),
                            )

                        if url_scheme(url) == "data":
                            estimated_size_bytes = _inline_payload_bytes(url)
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
                                return _oversized_inline_refusal(
                                    max_size_bytes, "oversized_inline_video",
                                    "an attached video clip",
                                )

                            await pipe._event_emitter_handler._emit_status(
                                event_emitter,
                                StatusMessages.VIDEO_BASE64,
                                done=False
                            )
                        elif not (
                            _verdict := await pipe._multimodal_handler._is_safe_url(
                                url, seconds=_remaining_address_seconds(address_deadline)
                            )
                        ):
                            if _verdict is None and is_http_or_https_url(url):
                                pipe.logger.log(
                                    logging.WARNING,
                                    "Address check for video URL %s reached no verdict within "
                                    "the request's address budget, so the link was not sent",
                                    loggable_link(url),
                                )
                                await pipe._ensure_error_formatter()._emit_error(
                                    event_emitter,
                                    "Video URL could not be checked against the address policy "
                                    "in time, so it was not sent",
                                    show_error_message=True
                                )
                                return ImageRefusal(
                                    "not checked against the address policy in time, so it "
                                    "was not sent",
                                    "uncheckable_video_url",
                                    severity="error",
                                    subject=loggable_link(url),
                                )
                            pipe.logger.error(
                                "SSRF protection blocked video URL: %s", loggable_link(url)
                            )
                            await pipe._ensure_error_formatter()._emit_error(
                                event_emitter,
                                "Video URL blocked by security policy (only http and https links are allowed)"
                                if not is_http_or_https_url(url)
                                else "Video URL blocked by security policy (private network)",
                                show_error_message=True
                            )
                            return ImageRefusal(
                                "not an http or https link, which is blocked by security "
                                "policy"
                                if not is_http_or_https_url(url)
                                else "a link to a private network address, which is "
                                "blocked by security policy",
                                "unsafe_video_url",
                                severity="error",
                                subject=loggable_link(url),
                            )
                        else:
                            await pipe._event_emitter_handler._emit_status(
                                event_emitter,
                                StatusMessages.VIDEO_YOUTUBE
                                if pipe._multimodal_handler._is_youtube_url(url)
                                else StatusMessages.VIDEO_REMOTE,
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
                        return ImageRefusal(
                            "a video clip that could not be prepared, so it was not sent",
                            "video_conversion_error",
                            subject="video",
                        )

                def _carries_url(value: Any) -> bool:
                    if isinstance(value, str):
                        return (
                            value.startswith(("http://", "https://", "ftp://", "gopher://", "//"))
                            or _INTERNAL_FILE_PATH in value
                        )
                    if isinstance(value, dict):
                        return any(
                            key not in _PROSE_KEYS and _carries_url(item)
                            for key, item in value.items()
                        )
                    if isinstance(value, (list, tuple)):
                        return any(_carries_url(item) for item in value)
                    return False

                def _identity_block(b: dict[str, Any]) -> dict[str, Any] | None:
                    if _carries_url(b):
                        nonlocal dropped_unknown_block
                        dropped_unknown_block = True
                        return None
                    return b

                block_transform = {
                    "text":       lambda b: {"type": "input_text",  "text": b.get("text", "")},
                    "input_text": lambda b: {"type": "input_text",  "text": b.get("text", "")},
                    "image_url":  _to_input_image,
                    "input_image": _to_input_image,
                    "image":      _to_input_image,
                    "input_file": _to_input_file,
                    "file":       _to_input_file,
                    "input_audio": _to_input_audio,
                    "audio":      _to_input_audio,
                    "video_url":  _to_input_video,
                    "input_video": _to_input_video,
                    "video":      _to_input_video,
                }

                converted_blocks: list[dict[str, Any]] = []
                user_images_used = 0
                refused_images: list[str] = []
                refused_files: list[str] = []
                status_files: list[str] = []
                carded_refusals: list[str] = []
                encountered_user_images = False
                reusable_image_blocks: list[dict[str, Any]] = []
                vision_warning_sent = False
                dropped_unknown_block = False
                latest_user_message = role == "user" and idx in current_turn_people
                is_last_current_turn_person = bool(current_turn_people) and idx == current_turn_people[-1]
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
                    is_image_block = block_type in {"image_url", "input_image", "image"}

                    if is_image_block:
                        if not (latest_user_message or tool_images) and _unconverted_block_reason(block) is None:
                            reusable_image_blocks.append(block)
                        if not include_user_images:
                            if latest_user_message and not vision_supported and not vision_warning_sent:
                                await pipe._event_emitter_handler._emit_status(
                                    event_emitter,
                                    "Model does not accept image inputs; skipping user attachments.",
                                    done=False,
                                )
                                vision_refusal = "the selected model does not accept picture inputs"
                                refused_images.append(vision_refusal)
                                carded_refusals.append(vision_refusal)
                                vision_warning_sent = True
                            elif not (latest_user_message or tool_images) and not vision_supported and image_limit > 0:
                                deferred_vision_skips += 1
                            continue
                        if not tool_images and turn_images_used >= image_limit:
                            turn_images_dropped += 1
                            encountered_user_images = True
                            continue

                    try:
                        if asyncio.iscoroutinefunction(transformer):
                            result = await transformer(block)
                        else:
                            result = transformer(block)
                        if isinstance(block, dict):
                            _void = result is None or (
                                isinstance(result, dict) and not _block_is_usable(result)
                            )
                            _unconverted = (
                                (_unconverted_block_reason(result) if isinstance(result, dict) else None)
                                or _unconverted_block_reason(block)
                            ) if _void else None
                            if _unconverted is not None:
                                if block.get("type") in {"input_image", "image_url", "image"}:
                                    refused_images.append(_unconverted)
                                else:
                                    refused_files.append(_unconverted)
                                    status_files.append(_unconverted)
                                result = None
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
                                if result.severity != "error":
                                    status_files.append(result.reason)
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
                            refused_images.append(result.reason)
                            encountered_user_images = True
                            continue
                        if result is None:
                            if is_image_block and is_last_current_turn_person:
                                encountered_user_images = True
                            if dropped_unknown_block:
                                dropped_unknown_block = False
                                pipe.logger.warning(
                                    "Dropping unsupported %s content block at index %d of the %s message %s",
                                    type(block).__name__,
                                    block_idx,
                                    role,
                                    msg_id,
                                )
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
                            turn_images_used += 1
                            encountered_user_images = True
                            pictures_in_request.add(_source_url_of(block))
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
                        if not is_image_block and not _carries_url(block):
                            converted_blocks.append(block)

                if (
                    latest_user_message
                    and is_last_current_turn_person
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
                                if transformed.cause == "oversized_inline":
                                    pictures_in_request.add(_source_url_of(source_block))
                            elif transformed is not None:
                                pictures_in_request.add(_source_url_of(source_block))
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
                        turn_images_used += len(fallback_blocks)

                if latest_user_message and (user_images_used or encountered_user_images):
                    person_images_this_turn = True
                if reusable_image_blocks:
                    last_image_blocks = reusable_image_blocks
                    last_image_turn = msg_turn_index

                image_notices: list[str] = []
                if latest_user_message and deferred_vision_skips:
                    image_notices.append(
                        f"left out {deferred_vision_skips} from earlier turns "
                        "(this model does not accept image inputs)"
                    )
                    deferred_vision_skips = 0
                unreported_images = [r for r in refused_images if r not in carded_refusals]
                if unreported_images:
                    image_notices.append(
                        f"skipped {len(unreported_images)} ({'; '.join(unreported_images)})"
                    )
                if turn_images_dropped and is_last_current_turn_person:
                    image_notices.append(
                        f"dropped {turn_images_dropped} over the limit of {image_limit}"
                    )
                    turn_images_dropped = 0
                notices = ["Images: " + "; ".join(image_notices) + "."] if image_notices else []
                if status_files:
                    notices.append(f"Files: skipped {len(status_files)} ({'; '.join(status_files)}).")
                if notices and (
                    latest_user_message
                    or (tool_images and idx == last_tool_handoff_index)
                    or status_files
                ):
                    await pipe._event_emitter_handler._emit_status(
                        event_emitter,
                        " ".join(notices),
                        done=False,
                    )
                if attachment_notices is not None and notices:
                    attachment_notices.append(" ".join(notices))

                if carried_blocks and not any(_block_is_usable(b) for b in converted_blocks):
                    converted_blocks = [b for b in converted_blocks if _block_is_usable(b)]
                    if not converted_blocks:
                        if refused_images or refused_files:
                            reasons = "; ".join(
                                reason.rstrip(".") for reason in refused_files + refused_images
                            )
                            converted_blocks.append({
                                "type": "input_text",
                                "text": f"{OPENAI_ATTACHMENT_NOT_SENT_PREFIX}{reasons}.]",
                            })
                        else:
                            converted_blocks.append({
                                "type": "input_text",
                                "text": OPENAI_EMPTY_USER_TURN_FALLBACK,
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
            msg_model = msg.get("model")
            same_model = (
                not isinstance(msg_model, str)
                or not msg_model.strip()
                or str(msg_model) == str(target_model_id)
            )
            raw_msg_reasoning_details = msg.get("reasoning_details")
            if (
                not same_model
                and isinstance(raw_msg_reasoning_details, list)
                and raw_msg_reasoning_details
            ):
                pipe.logger.debug(
                    "Dropping %d reasoning detail(s) from message %s: produced by %s, this request "
                    "is answered by %s",
                    len(raw_msg_reasoning_details),
                    msg_id,
                    msg_model,
                    target_model_id,
                )
            msg_reasoning_details: list[Any] = (
                list(raw_msg_reasoning_details)
                if (
                    active_valves.PERSIST_REASONING_TOKENS != "disabled"
                    and same_model
                    and isinstance(raw_msg_reasoning_details, list)
                    and raw_msg_reasoning_details
                )
                else []
            )
            assistant_text = _message_text(idx, msg)
            is_old_message = _is_old_turn(msg_turn_index, threshold=prune_before_turn)

            assistant_spans = markdown_image_spans(assistant_text)
            assistant_image_urls = [url for url, _start, _end in assistant_spans]
            if assistant_image_urls:
                last_image_blocks = [
                    {"type": "image_url", "image_url": url, "detail": "auto"}
                    for url in assistant_image_urls
                ]
                last_image_turn = msg_turn_index
                window_armed_at.add(idx)

            appended_text_chunks: list[dict[str, Any]] = []

            def _append_assistant_text_chunks(
                text: str,
                appended: list[dict[str, Any]] = appended_text_chunks,
                msg_annotations: list[Any] = msg_annotations,
                msg_reasoning_details: list[Any] = msg_reasoning_details,
                at: int = idx,
                spans: list[tuple[str, int, int]] = assistant_spans,
                text_base: int = 0,
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
                    if selection_mode == "user_then_assistant" and at < last_person_position:
                        base = text_base + phase_chunk["start"]
                        limit = base + len(phase_chunk["text"])
                        lead = len(phase_chunk["text"]) - len(
                            phase_chunk["text"].lstrip()
                        )
                        lift_candidates.append((
                            item_out,
                            [
                                (url, start - base - lead, end - base - lead)
                                for url, start, end in spans
                                if base <= start and end <= limit
                            ],
                        ))

                if not chunk_items:
                    return
                openai_input.extend(chunk_items)
                appended.extend(chunk_items)

            assistant_marker_spans = _marker_spans_for(assistant_text)
            if assistant_marker_spans:
                withhold_replayed_reasoning = active_valves.PERSIST_REASONING_TOKENS == "disabled"
                segments = _marker_segments_for(assistant_text, assistant_marker_spans)
                segment_cursor = 0
                markers = [seg["marker"] for seg in segments if seg.get("type") == "marker"]

                db_artifacts: dict[str, dict] = {}
                group_producers: dict[str, str] = {}
                replayable: dict[str, dict[str, Any]] = {}
                orphaned_call_ids: set[str] = set()
                orphaned_output_ids: set[str] = set()
                orphaned_call_markers: set[str] = set()
                orphaned_output_markers: set[str] = set()
                if artifact_loader and chat_id and openwebui_model_id and markers:
                    batch = artifact_groups.get(msg_id) or {}
                    group_producers = artifact_producers.get(msg_id) or {}
                    db_artifacts = {marker: batch[marker] for marker in markers if marker in batch}
                    for marker, row in db_artifacts.items():
                        normalized = _normalized_row(row)
                        if normalized is not None:
                            replayable[marker] = normalized
                    (
                        _,
                        orphaned_call_ids,
                        orphaned_output_ids,
                    ) = _classify_function_call_artifacts(replayable)
                    (
                        orphaned_call_markers,
                        orphaned_output_markers,
                    ) = _orphaned_round_markers(segments, replayable)
                    if orphaned_call_ids:
                        logger.debug(
                            "Dropping %d persisted function_call artifact(s) missing outputs (chat_id=%s message_id=%s call_ids=%s)",
                            len(orphaned_call_ids),
                            _chat_log_subject(chat_id),
                            msg_id,
                            sorted(orphaned_call_ids),
                        )
                    if orphaned_output_ids:
                        logger.warning(
                            "Dropping %d persisted function_call_output artifact(s) missing calls (chat_id=%s message_id=%s call_ids=%s)",
                            len(orphaned_output_ids),
                            _chat_log_subject(chat_id),
                            msg_id,
                            sorted(orphaned_output_ids),
                        )

                replay_pending: dict[str, list[str]] = {}
                for segment in segments:
                    if segment["type"] == "marker":
                        artifact_payload = db_artifacts.get(segment["marker"])
                        if artifact_payload is None:
                            missing_artifact_markers.append(segment["marker"])
                            continue
                        if artifact_payload.get("type") == "reasoning":
                            row_producer = group_producers.get(segment["marker"])
                            if (
                                isinstance(row_producer, str)
                                and row_producer.strip()
                                and str(row_producer) != str(target_model_id)
                            ):
                                pipe.logger.debug(
                                    "Dropping stored reasoning artifact %s: produced by %s, this "
                                    "request is answered by %s",
                                    segment["marker"],
                                    row_producer,
                                    target_model_id,
                                )
                                continue
                        if (
                            artifact_payload.get("type") == "reasoning"
                            and replayed_reasoning_refs is not None
                            and chat_id
                        ):
                            replayed_reasoning_refs.append((chat_id, segment["marker"]))
                            if withhold_replayed_reasoning:
                                continue
                        item = replayable.get(segment["marker"])
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
                                and segment["marker"] in orphaned_call_markers
                            ):
                                logger.debug(
                                    "Skipping orphaned function_call artifact (call_id=%s chat_id=%s message_id=%s)",
                                    item.get("call_id"),
                                    _chat_log_subject(chat_id),
                                    msg_id,
                                )
                                continue
                            if (
                                item_type == "function_call_output"
                                and segment["marker"] in orphaned_output_markers
                            ):
                                logger.debug(
                                    "Skipping orphaned function_call_output artifact (call_id=%s chat_id=%s message_id=%s)",
                                    item.get("call_id"),
                                    _chat_log_subject(chat_id),
                                    msg_id,
                                )
                                continue
                            replay_round_name = _replay_round_name(
                                item_type, str(item.get("name") or ""), item.get("call_id"), replay_pending
                            )
                            replay_round_exempt = _round_keeps_its_ask_user_answer(
                                item.get("call_id"),
                                replay_round_name,
                                ask_user_names,
                                recorded_ask_user_rounds,
                                stamped_ask_user_rounds,
                            )
                            if item_type == "function_call_output" and not (
                                replay_round_exempt
                                or (idx in window_armed_at and not is_picture_output(item.get("output")))
                            ):
                                last_image_blocks, last_image_turn = [], None
                            if _tool_round_withheld(msg_turn_index):
                                withheld_items = _without_tool_result(
                                    item,
                                    replay_round_name,
                                    ask_user_names,
                                    recorded_ask_user_rounds,
                                    stamped_ask_user_rounds,
                                )
                                if withheld_items is not None:
                                    openai_input.extend(_from_pipe_storage(withheld) for withheld in withheld_items)
                                    continue
                            for part in _as_replayed(item, fallback_id=segment["marker"]):
                                if (
                                    not replay_round_exempt
                                    and is_old_message
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
                        _append_assistant_text_chunks(
                            segment["text"],
                            text_base=assistant_text.find(segment["text"], segment_cursor),
                        )
                        segment_cursor += len(segment["text"])
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
                    if _tool_round_withheld(msg_turn_index) and not _round_keeps_its_ask_user_answer(
                        tool_call_id, name, ask_user_names, recorded_ask_user_rounds, stamped_ask_user_rounds
                    ):
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

        if _deferred_tool_pictures or _deferred_tool_refusals:
            await _flush_deferred_tool_pictures()

        if missing_artifact_markers:
            distinct_missing = sorted(set(missing_artifact_markers))
            logger.log(
                logging.DEBUG if temporary_chat else logging.WARNING,
                "Missing %d artifact(s) across %d marker reference(s) for chat_id=%s: %s",
                len(distinct_missing),
                len(missing_artifact_markers),
                _chat_log_subject(chat_id),
                distinct_missing,
            )

        for item_out, spans in lift_candidates:
            cursor = 0
            spliced: list[str] = []
            for destination, start, end in spans:
                if destination not in pictures_in_request:
                    continue
                spliced.append(item_out["content"][0]["text"][cursor:start])
                spliced.append(_LIFTED_TEXT_IMAGE_PLACEHOLDER)
                cursor = end
            if not spliced:
                continue
            spliced.append(item_out["content"][0]["text"][cursor:])
            item_out["content"][0]["text"] = "".join(spliced)

        replay_refusals: list[tuple[str, str, str]] = []
        for row in openai_input:
            if not (
                isinstance(row, dict)
                and row.get(_PIPE_STORAGE_KEY)
                and row.get("type") == "function_call_output"
                and is_picture_output(row.get("output"))
            ):
                continue
            _replay_text, shown = tool_output_text_and_pictures(row["output"])
            admitted, refused_shown = await _gated_tool_pictures_with_address(
                pipe, shown, max_inline_bytes=max_inline_bytes,
                allow_insecure=pipe._multimodal_handler._is_insecure_http_allowed,
                seen=address_verdicts, deadline=address_deadline,
            )
            for url, reason, cause in refused_shown:
                logger.warning(
                    "Not replaying a stored tool's picture (%s): %s [cause=%s]",
                    loggable_link(url), reason, cause,
                )
            replay_refusals.extend(refused_shown)
            row[_REPLAY_GATED_PICTURES_KEY] = admitted
        if replay_refusals:
            await pipe._event_emitter_handler._emit_status(
                event_emitter, _tool_picture_notice(replay_refusals), done=False,
            )

        openai_input = _reinterleave_reasoning_by_anchor(openai_input)

        _maybe_apply_anthropic_prompt_caching(
            openai_input,
            model_id=target_model_id,
            valves=active_valves,
        )
        return openai_input
