from __future__ import annotations

import ast
import asyncio
import codecs
import datetime
import hashlib
import hmac
import inspect
import json
import logging
import math
import re
import uuid
from collections.abc import Awaitable, Iterable, Iterator
from contextvars import ContextVar
from typing import Any, TypeVar, cast

import aiohttp

from .config import (
    CROCKFORD_ALPHABET,
    DEFAULT_OPENROUTER_ERROR_TEMPLATE,
    ULID_LENGTH,
    _application_secret,
)
from .media_slot import MediaSlot
from .url_scheme import (
    _BASE64_FOLD_SLICE,
    _BASE64_FOLDED_TABLE,
    _STRIPPED_SCHEME_BYTES,
    base64_data_url_payload_len,
    loggable_link,
    media_type_or_empty,
)
from .warn_latch import shared_latch, warn_level

logger = logging.getLogger(__name__)

_OWUI_CLASSIFIER_IMPORT_LATCH: set[str] = shared_latch("owui_classifier_import")

try:
    from open_webui.utils.middleware import (
        _is_tool_result_error as _owui_is_tool_result_error,  # type: ignore[import-not-found]
    )
except ImportError:
    _owui_is_tool_result_error = None  # type: ignore
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui.utils.middleware failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    _owui_is_tool_result_error = None  # type: ignore

if _owui_is_tool_result_error is None:
    logger.log(
        warn_level(_OWUI_CLASSIFIER_IMPORT_LATCH, "tool_classifier_import"),
        "open_webui.utils.middleware is unavailable, so the pipe is using its own copy "
        "of the tool-result classifier",
    )

_T = TypeVar("_T")

TOOL_CALL_STATUSES = frozenset({"in_progress", "completed", "incomplete"})



SERVER_TOOL_EXTRA_SUCCESS = frozenset({"ok"})
SERVER_TOOL_IN_FLIGHT_STATUSES = frozenset({"in_progress", "generating", "searching"})
SERVER_TOOL_SUCCESS_STATUSES = frozenset({"completed"}) | SERVER_TOOL_EXTRA_SUCCESS
SERVER_TOOL_FAILURE_STATUSES = frozenset({"incomplete", "failed"})
OWUI_SETTLED_CALL_STATUSES = frozenset({"completed", "failed", "rejected"})
OWUI_UNRESOLVABLE_CALL_STATUSES = frozenset({"pending", "queued", "requires_approval", "rejected"})


def owui_call_status(result_status: str | None) -> str:
    if result_status in OWUI_SETTLED_CALL_STATUSES:
        return str(result_status)
    if result_status in SERVER_TOOL_SUCCESS_STATUSES:
        return "completed"
    return "failed"

_TEMPLATE_IF_TOKEN_RE = re.compile(r"\{\{\s*(#if\s+(\w+)|/if)\s*\}\}")
_TEMPLATE_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
_FENCE_RUN_RE = re.compile(r"(`{3,}|~{3,})")
_BLOCKQUOTE_PREFIX_RE = re.compile(r"^(?:>[ \t]?)+")
_FENCE_OWNED_KEYS = frozenset({"raw_body", "flagged_excerpt", "metadata_json", "provider_raw_json", "body_excerpt"})
_MARKER_SUFFIX = "]: #"
_CROCKFORD_SET = frozenset(CROCKFORD_ALPHABET)
_PHASE_MARKER_RE = re.compile(r"^\[P:([a-z_]+)\]: #$")
_PHASE_MARKER_VALUES = frozenset({"commentary", "final_answer", "null"})

REASONING_ANCHOR_SEQ_KEY = "_anchor_seq"
REASONING_FOLLOWING_ORDINAL_KEY = "_anchor_following_call_ordinal"
REASONING_PRECEDING_ORDINAL_KEY = "_anchor_preceding_call_ordinal"
REASONING_TEXT_ORDINAL_KEY = "_anchor_text_ordinal"
REASONING_FOLLOWING_SERVER_ITEM_KEY = "_anchor_following_server_item"
_ROW_MODEL_KEY = "_row_model"
TOOL_ROUND_SKELETON_KEY = "_anchor_tool_round_skeleton"
PIPE_ONLY_TOOL_ROUND_KEY = "_anchor_pipe_only_tool_round"
BUILTIN_ASK_USER_ROUND_KEY = "_anchor_builtin_ask_user"
PIPE_TOOL_IMAGES_KEY = "_anchor_pipe_tool_images_message"
UNRETAINED_TOOL_RESULT = "[This call succeeded. Its result is not kept for later turns.]"
UNRETAINED_FAILED_TOOL_RESULT = "[This call failed. Its result is not kept for later turns.]"
_EARLIER_UNRETAINED_FAILED_TOOL_RESULT = "[tool call failed; result not retained]"
TOOL_FAILURE_LINE = "Error: the tool call did not complete."

IMAGE_NO_IMAGES_REASON = "OpenRouter image generation returned no images."


def image_failure_billing_suffix(billed: dict[str, Any]) -> str:
    return "OpenRouter billed it before it stopped." if billed else "Nothing was billed."


SERVER_TOOL_CALL_PREFIX = "srv-"
REASONING_ANCHOR_KEYS = (
    REASONING_ANCHOR_SEQ_KEY,
    REASONING_FOLLOWING_ORDINAL_KEY,
    REASONING_PRECEDING_ORDINAL_KEY,
    REASONING_TEXT_ORDINAL_KEY,
    REASONING_FOLLOWING_SERVER_ITEM_KEY,
    _ROW_MODEL_KEY,
)


def parse_tool_arguments(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    if not text.strip():
        return {}
    try:
        params = json.loads(text)
    except (ValueError, RecursionError):
        try:
            params = ast.literal_eval(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            return None
    if not isinstance(params, dict):
        raise ValueError("Tool call arguments must be a JSON object.")  # noqa: TRY004
    return params


def split_tool_argument_objects(raw: Any) -> list[str]:
    if not isinstance(raw, str):
        return [raw]
    decoder = json.JSONDecoder()
    found: list[str] = []
    position = 0
    while position < len(raw):
        while position < len(raw) and raw[position].isspace():
            position += 1
        if position >= len(raw):
            break
        try:
            _value, end = decoder.raw_decode(raw, position)
        except (ValueError, RecursionError):
            return [raw]
        found.append(raw[position:end].strip())
        position = end
    return found if len(found) > 1 else [raw]


def server_tool_call_id(item_id: Any) -> str:
    return f"{SERVER_TOOL_CALL_PREFIX}{item_id if isinstance(item_id, str) and item_id else uuid.uuid4().hex}"


def is_server_tool_call_id(call_id: Any) -> bool:
    return isinstance(call_id, str) and call_id.startswith(SERVER_TOOL_CALL_PREFIX)


def unretained_tool_result(failed: bool) -> str:
    return UNRETAINED_FAILED_TOOL_RESULT if failed else UNRETAINED_TOOL_RESULT


def unretained_tool_result_ref(failed: bool, ref: str | None) -> str:
    note = UNRETAINED_FAILED_TOOL_RESULT if failed else UNRETAINED_TOOL_RESULT
    if not ref:
        return note
    return note[:-2] + f"; view_tool_call {ref} returns it.]"


def _own_is_tool_result_error(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text.startswith(("error:", "exception:", "traceback", "http error!")):
            return True

    parsed = value
    while isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except (TypeError, ValueError):
            break

    if not isinstance(parsed, dict):
        return False

    error = parsed.get("error")
    if isinstance(error, str):
        has_error = bool(error.strip())
    else:
        has_error = isinstance(error, (dict, list)) and bool(error)
    if has_error:
        return True

    status = parsed.get("status")
    if isinstance(status, str) and status.strip().lower() in {"error", "failed"}:
        return True

    if parsed.get("success") is False or parsed.get("ok") is False:
        message = parsed.get("message")
        return has_error or (
            bool(message.strip()) if isinstance(message, str) else isinstance(message, (dict, list)) and bool(message)
        )

    return False


def _tool_result_failed(text: str, status: Any = None) -> bool:
    if isinstance(status, str) and status and status != "completed":
        return True
    if text in (UNRETAINED_FAILED_TOOL_RESULT, _EARLIER_UNRETAINED_FAILED_TOOL_RESULT):
        return True
    if _owui_is_tool_result_error is None:
        return _own_is_tool_result_error(text)
    try:
        return bool(_owui_is_tool_result_error(text))
    except Exception:
        logger.debug("Open WebUI could not classify a tool result", exc_info=True)
        return _own_is_tool_result_error(text)


def is_picture_output(output: Any) -> bool:
    return (
        isinstance(output, list)
        and bool(output)
        and all(isinstance(part, dict) and part.get("type") in ("input_text", "input_image") for part in output)
        and any(part.get("type") == "input_image" for part in output)
    )


def is_media_part_output(output: Any) -> bool:
    return (
        isinstance(output, list)
        and bool(output)
        and all(
            isinstance(part, dict)
            and part.get("type") in ("input_text", "input_image", "input_file")
            for part in output
        )
    )


def is_text_part_output(output: Any) -> bool:
    return (
        isinstance(output, list)
        and bool(output)
        and all(isinstance(part, dict) and part.get("type") == "input_text" for part in output)
    )


def _picture_link(part: Any) -> str:
    value = part.get("image_url")
    if isinstance(value, dict):
        value = value.get("url")
    return value.strip() if isinstance(value, str) and value.strip() else ""


def tool_output_text_and_pictures(output: Any) -> tuple[str, list[str]]:
    if is_picture_output(output) or is_text_part_output(output):
        text = "".join(str(part.get("text") or "") for part in output if part.get("type") == "input_text")
        links = [_picture_link(part) for part in output if part.get("type") == "input_image"]
        return text, [link for link in links if link]
    return (output if isinstance(output, str) else ("" if output is None else str(output))), []


_DATA_URL_LOG_SCAN = re.compile(
    r"""(?<![A-Za-z0-9+.-])d[\t\r\n]*a[\t\r\n]*t[\t\r\n]*a[\t\r\n]*:(?:[^\s"')\]}]|[\t\r\n])*""",
    re.IGNORECASE,
)
_DATA_URL_LOG_MARKER_RE = re.compile(r"\s*\[redacted\]")
_DATA_URL_NAME_PARAM = re.compile(r";name=[^;,]*", re.IGNORECASE)
_DATA_URL_PRESENT = re.compile(r"d[\t\r\n]*a[\t\r\n]*t[\t\r\n]*a[\t\r\n]*:", re.IGNORECASE)


def _base64_run_end(text: str, body_at: int, stop: int) -> int:
    newline = text.find("\n", body_at, stop + 1)
    end = stop if newline < 0 else newline
    while True:
        step = _BASE64_FOLD_STEP.match(text, end)
        if step is None or step.end() > stop:
            return end
        end = step.end()


def _is_a_bare_media_type(head: str) -> bool:
    candidate = head[len("data:") :]
    media_type = media_type_or_empty(candidate)
    return bool(media_type) and candidate.strip().lower() == media_type


def _data_url_log_subject(text: str) -> str:
    return _truncate_base64_runs(_data_url_tokens(text), 256)


def _data_url_tokens(text: str) -> str:
    if not _DATA_URL_PRESENT.search(text):
        return text

    out: list[str] = []
    last = 0
    for match in _DATA_URL_LOG_SCAN.finditer(text):
        start, stop = match.span()
        if start < last:
            continue
        comma_at = text.find(",", start, stop)
        head = text[start:comma_at] if comma_at >= 0 else text[start:stop]
        comma = comma_at - start if comma_at >= 0 else -1
        if comma < 0 and (
            _is_a_bare_media_type(head) or _DATA_URL_LOG_MARKER_RE.match(text, stop)
        ):
            continue
        if comma >= 0 and ";base64" not in head.lower():
            newline = text.find("\n", comma_at)
            end = len(text) if newline < 0 else newline
        else:
            end = _base64_run_end(text, comma_at + 1 if comma >= 0 else stop, stop)
        flat = head.translate(_STRIPPED_SCHEME_BYTES).split()
        candidate = flat[0] if flat else head
        out.append(text[last:start])
        out.append(f"data:{media_type_or_empty(candidate[len('data:') :])} [redacted]")
        last = end
    out.append(text[last:])
    return "".join(out)


def picture_output(text: str, pictures: list[str]) -> list[dict[str, Any]]:
    return [{"type": "input_text", "text": text}, *({"type": "input_image", "image_url": url} for url in pictures)]


def recorded_tool_text(text: str, status: Any) -> str:
    if status in (None, "completed") or _tool_result_failed(text):
        return text
    return f"{TOOL_FAILURE_LINE}\n{text}" if text else TOOL_FAILURE_LINE


_IMAGE_ITEM_TYPE = "openrouter:image_generation"
_IMAGE_ITEM_BLOB_KEYS = ("result", "imageUrl", "imageB64")
_IMAGE_ENTRY_URL_KEYS = ("url", "image_url", "imageUrl", "content_url")
_IMAGE_ENTRY_B64_KEYS = ("b64_json", "b64", "base64", "data", "image_base64", "imageB64")


def _image_entry_carries_an_image(entry: Any, _depth: int = 0) -> bool:
    if _depth > 6:
        return False
    if isinstance(entry, str):
        return bool(entry.strip())
    if isinstance(entry, (list, tuple)):
        return any(_image_entry_carries_an_image(item, _depth + 1) for item in entry)
    if isinstance(entry, dict):
        for key in _IMAGE_ENTRY_URL_KEYS:
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return True
            if isinstance(value, (dict, list, tuple)) and _image_entry_carries_an_image(value, _depth + 1):
                return True
        if any(isinstance(entry.get(key), str) and entry[key].strip() for key in _IMAGE_ENTRY_B64_KEYS):
            return True
        nested = entry.get("result")
        if nested is not None and _image_entry_carries_an_image(nested, _depth + 1):
            return True
    return False


def _image_item_is_empty(item: dict[str, Any]) -> bool:
    for key in _IMAGE_ITEM_BLOB_KEYS:
        value = item.get(key)
        if isinstance(value, (list, tuple)):
            if any(_image_entry_carries_an_image(entry) for entry in value):
                return False
        elif _image_entry_carries_an_image(value):
            return False
    return True


def server_tool_status(item: dict[str, Any]) -> str:
    reported = item.get("status")
    if isinstance(reported, str) and reported:
        if reported in SERVER_TOOL_IN_FLIGHT_STATUSES:
            return "in_progress"
        if reported in SERVER_TOOL_FAILURE_STATUSES:
            return "incomplete"
        if reported not in SERVER_TOOL_SUCCESS_STATUSES:
            return "incomplete"
    if item.get("error"):
        return "incomplete"
    http_status = item.get("httpStatus")
    if http_status is not None:
        try:
            code = int(http_status)
        except (TypeError, ValueError):
            return "incomplete"
        if not 200 <= code < 300:
            return "incomplete"
    if item.get("type") == _IMAGE_ITEM_TYPE and _image_item_is_empty(item):
        return "incomplete"
    return "completed"


_SERVER_TOOL_ARGUMENT_KEYS = {"openrouter:advisor": ("prompt",), "openrouter:subagent": ("task_name", "task_description")}
_SEARCH_MODELS_ITEM_TYPE = "openrouter:experimental__search_models"


def _search_models_arguments(item: dict[str, Any]) -> dict[str, Any]:
    raw = item.get("arguments")
    try:
        parsed = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, RecursionError):
        parsed = None
    if isinstance(parsed, dict) and parsed:
        return parsed
    query = item.get("query")
    return {"query": query} if isinstance(query, str) and query else {}


def server_tool_arguments(item: dict[str, Any]) -> dict[str, Any]:
    item_type = str(item.get("type") or "")
    if item_type == _SEARCH_MODELS_ITEM_TYPE:
        return _search_models_arguments(item)
    return {key: item[key] for key in _SERVER_TOOL_ARGUMENT_KEYS.get(item_type, ()) if item.get(key)}


def server_tool_result_text(item: dict[str, Any]) -> str:
    item_type = str(item.get("type") or "")
    if item_type in ("openrouter:advisor", "openrouter:subagent"):
        error = item.get("error")
        if error:
            return str(error)
        return str(item.get("advice" if item_type == "openrouter:advisor" else "outcome") or "")
    result_data = item.get("result")
    if result_data is None:
        result_data = {k: v for k, v in item.items() if k not in ("type", "id", "status")} or None
    try:
        return (
            json.dumps(result_data, indent=2, ensure_ascii=False)
            if result_data is not None
            else str(item.get("status") or "completed")
        )
    except (TypeError, ValueError):
        return str(result_data)


def current_turn_items(items: Any) -> list[dict[str, Any]]:
    turn: list[dict[str, Any]] = []
    if isinstance(items, list):
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            if opens_a_turn(items, index):
                turn = []
            else:
                turn.append(item)
    return turn


def continued_turn_counts(items: Any) -> tuple[int, int, bool]:
    turn = current_turn_items(items)
    calls = sum(
        1 for item in turn if item.get("type") == "function_call" and not is_server_tool_call_id(item.get("call_id"))
    )
    texts = sum(1 for item in turn if item.get("type") == "message" and item.get("role") == "assistant")
    return calls, texts, bool(turn) and turn[-1].get("type") == "reasoning"


OPEN_WEBUI_TOOL_IMAGES_TEXT = "Here are the images from the tool results above. Please analyze them."


def opens_a_turn(items: list[Any], index: int) -> bool:
    item = items[index]
    if not (isinstance(item, dict) and item.get("type") == "message" and item.get("role") == "user"):
        return False
    content = item.get("content")
    first, *rest = content if isinstance(content, list) and content else [None]
    if not (
        isinstance(first, dict)
        and first.get("type") == "input_text"
        and first.get("text") == OPEN_WEBUI_TOOL_IMAGES_TEXT
        and (
            item.get(PIPE_TOOL_IMAGES_KEY) if not rest
            else all(isinstance(part, dict) and part.get("type") == "input_image" for part in rest)
        )
    ):
        return True
    before = None
    for pos in range(index - 1, -1, -1):
        candidate = items[pos]
        if not (isinstance(candidate, dict) and candidate.get("type") == "reasoning"):
            before = candidate
            break
    return not (isinstance(before, dict) and before.get("type") == "function_call_output")


def is_tool_image_handoff(previous: Any, message: Any) -> bool:
    if not (isinstance(previous, dict) and isinstance(message, dict)):
        return False
    content = message.get("content")
    if not (previous.get("role") == "tool" and message.get("role") == "user" and isinstance(content, list)):
        return False
    first, *images = content or [None]
    if not (isinstance(first, dict) and first.get("type") == "text"
            and first.get("text") == OPEN_WEBUI_TOOL_IMAGES_TEXT
            and bool(images)
            and all(isinstance(part, dict) and part.get("type") == "image_url" for part in images)):
        return False
    handoff = _handoff_urls(message)
    if not handoff:
        return False
    round_pictures = _picture_urls(previous.get("content"))
    if not round_pictures:
        return True
    return any(url in round_pictures for url in handoff)


def _picture_urls(blocks: Any) -> list[str]:
    urls: list[str] = []
    for part in blocks if isinstance(blocks, list) else []:
        if not (isinstance(part, dict) and part.get("type") in ("input_image", "image_url")):
            continue
        url = part.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
        if isinstance(url, str) and url:
            urls.append(url)
    return urls


def _handoff_urls(message: Any) -> list[str]:
    if not isinstance(message, dict) or message.get("role") != "user":
        return []
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return []
    first, *images = content
    if not (
        isinstance(first, dict)
        and first.get("type") == "text"
        and first.get("text") == OPEN_WEBUI_TOOL_IMAGES_TEXT
        and images
        and all(isinstance(part, dict) and part.get("type") == "image_url" for part in images)
    ):
        return []
    urls: list[str] = []
    for part in images:
        url = part.get("image_url")
        if isinstance(url, dict):
            url = url.get("url")
        if isinstance(url, str) and url:
            urls.append(url)
    return urls


def is_tool_image_handoff_for_round(results: list[Any], message: Any) -> bool:
    handoff = _handoff_urls(message)
    if not handoff:
        return False
    seen: set[str] = set()
    for result in results:
        if not (isinstance(result, dict) and result.get("role") == "tool"):
            continue
        seen.update(_picture_urls(result.get("content")))
    return any(url in seen for url in handoff)


def brings_tool_results(body: dict[str, Any]) -> bool:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages or not isinstance(messages[-1], dict):
        return False
    if messages[-1].get("role") == "tool":
        return True
    return len(messages) > 1 and is_tool_image_handoff(messages[-2], messages[-1])



def _stable_crockford_id(seed: str, *, length: int = ULID_LENGTH) -> str:
    length_int = int(length)
    if length_int <= 0:
        raise ValueError("length must be positive")

    digest = hashlib.sha256((seed or "").encode("utf-8", "ignore")).digest()
    bits_needed = length_int * 5
    total_bits = len(digest) * 8
    if bits_needed > total_bits:
        raise ValueError("seed hash too short for requested id length")

    value = int.from_bytes(digest, byteorder="big", signed=False)
    shift = total_bits - bits_needed
    value = value >> shift

    out: list[str] = []
    for i in range(length_int):
        idx = (value >> ((length_int - 1 - i) * 5)) & 0x1F
        out.append(CROCKFORD_ALPHABET[idx])
    return "".join(out)


def _sticky_session_key(chat_id: str) -> str | None:
    secret = _application_secret()
    if not secret:
        return None
    return hmac.new(secret.encode("utf-8"), chat_id.encode("utf-8"), hashlib.sha256).hexdigest()

def _render_error_template(template: str, values: dict[str, Any]) -> str:
    if not template:
        template = DEFAULT_OPENROUTER_ERROR_TEMPLATE

    rendered_lines: list[str] = []
    condition_stack: list[bool] = []
    fence_marker = ""
    emit_marker = ""
    fence_open = False
    fence_carried_content = False
    pending_opener = -1

    def _conditions_active() -> bool:
        return all(condition_stack) if condition_stack else True

    def _strip_fence(value: str) -> str:
        text = value.strip("\n")
        runs = _FENCE_RUN_RE.findall(text)
        if len(runs) < 2 or runs[0][0] != runs[-1][0] or not text.startswith(runs[0]):
            return value
        tail = text[len(runs[0]) :]
        language = ""
        newline = tail.find("\n")
        if newline < 0:
            return value
        language = tail[:newline].strip()
        if language and not re.fullmatch(r"[\w.+-]*", language):
            return value
        return tail[newline + 1 :][: -len(runs[-1])].rstrip("\n")

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            return match.group(0)
        value = values[name]
        if fence_open and name in _FENCE_OWNED_KEYS:
            value = _strip_fence(str(value))
        return "" if value is None else str(value)

    def _fence_language(run: str, between: str) -> str:
        candidate = between.strip()
        if candidate and re.fullmatch(r"[\w.+-]*", candidate):
            return candidate
        return ""

    def _body_fence(run: str, body: str) -> str:
        char = run[0]
        longest = max((len(m.group(0)) for m in re.finditer(re.escape(char) + "+", body)), default=0)
        return char * max(3, len(run), longest + 1)

    def _wrapped_fence_block(line: str) -> list[str]:
        match = _TEMPLATE_PLACEHOLDER_RE.search(line)
        if match is None or match.group(1) not in _FENCE_OWNED_KEYS:
            return []
        opener = _FENCE_RUN_RE.search(line[: match.start()])
        closer = _FENCE_RUN_RE.search(line[match.end() :])
        if opener is None or closer is None or opener.group(1)[0] != closer.group(1)[0]:
            return []
        if len(closer.group(1)) < len(opener.group(1)):
            return []
        prefix = line[: opener.start()]
        quoted = _BLOCKQUOTE_PREFIX_RE.match(prefix)
        quote = quoted.group(0) if quoted else ""
        label = prefix[len(quote) :].rstrip()
        tail = line[match.end() :][closer.end() :].strip()
        body = _strip_fence(str(values.get(match.group(1), "")))
        if not body:
            return [quote + label] if label else []
        language = _fence_language(opener.group(0), line[opener.end() : match.start()]) or _fence_language(
            opener.group(0), label
        )
        fence = _body_fence(opener.group(0), body)
        block = ([quote + label] if label else []) + [quote + fence + language]
        block.extend(quote + row if row else quote.rstrip() for row in body.split("\n"))
        block.append(quote + fence)
        if tail:
            block.append(quote + tail)
        return block

    def _wrapped_fence_key(line: str) -> bool:
        match = _TEMPLATE_PLACEHOLDER_RE.search(line)
        if match is None or match.group(1) not in _FENCE_OWNED_KEYS:
            return False
        opener = _FENCE_RUN_RE.search(line[: match.start()])
        closer = _FENCE_RUN_RE.search(line[match.end() :])
        if opener is None or closer is None or opener.group(1)[0] != closer.group(1)[0]:
            return False
        return len(closer.group(1)) >= len(opener.group(1))

    for raw_line in template.splitlines():
        last_index = 0
        line_parts: list[str] = []

        for match in _TEMPLATE_IF_TOKEN_RE.finditer(raw_line):
            segment = raw_line[last_index:match.start()]
            if segment and _conditions_active():
                line_parts.append(segment)

            token = match.group(1) or ""
            var_name = match.group(2)
            if token.startswith("#if"):
                condition_stack.append(_template_value_present(values.get(var_name or "")))
            else:
                if condition_stack:
                    condition_stack.pop()

            last_index = match.end()

        tail_segment = raw_line[last_index:]
        if tail_segment and _conditions_active():
            line_parts.append(tail_segment)

        if not line_parts:
            if raw_line.strip():
                continue
            if not _conditions_active():
                continue
            rendered_lines.append("")
            continue

        line = "".join(line_parts)
        quoted_line = _BLOCKQUOTE_PREFIX_RE.match(line)
        line_prefix = quoted_line.group(0) if quoted_line else ""
        remainder = line[len(line_prefix) :]

        def _emit(text: str, prefix: str = line_prefix) -> None:
            rendered_lines.extend(
                prefix + row if row else prefix.rstrip() for row in text.split("\n")
            )

        if any(
            f"{{{name}}}" in line and not _template_value_present(value)
            for name, value in values.items()
        ):
            continue
        opened_on_this_line = fence_open
        closed_here = False
        stripped = remainder.strip()
        if not opened_on_this_line and _wrapped_fence_key(line):
            rendered_lines.extend(_wrapped_fence_block(line))
            continue
        if not opened_on_this_line:
            leading = _FENCE_RUN_RE.match(stripped)
            if leading and stripped[: len(leading.group(0))] == leading.group(0):
                if stripped[len(leading.group(0)) :].strip().startswith("`"):
                    _emit(_TEMPLATE_PLACEHOLDER_RE.sub(_replace, remainder))
                    continue
                fence_marker = leading.group(0)
                emit_marker = fence_marker
                fence_open = True
                fence_carried_content = False
                pending_opener = len(rendered_lines)
        elif stripped and set(stripped) == set(emit_marker) and len(stripped) >= len(fence_marker):
            fence_open = False
            closed_here = True
            if not fence_carried_content:
                del rendered_lines[pending_opener:]
                emit_marker = ""
                continue
            if len(emit_marker) > len(stripped):
                remainder = remainder[: len(remainder) - len(remainder.lstrip())] + emit_marker
        elif stripped:
            fence_carried_content = True
        remainder = _TEMPLATE_PLACEHOLDER_RE.sub(_replace, remainder)
        if opened_on_this_line and not closed_here:
            widened = _body_fence(emit_marker, remainder)
            if len(widened) > len(emit_marker):
                emit_marker = widened
                rendered_lines[pending_opener] = rendered_lines[pending_opener].replace(
                    fence_marker, widened, 1
                )
        _emit(remainder)
    return "\n".join(rendered_lines).strip()


def _pretty_json(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, bytes)):
        text = value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value
        return text.strip()
    if isinstance(value, (dict, list, tuple)) and not value:
        return ""
    try:
        return json.dumps(value, indent=2, ensure_ascii=False)
    except (RecursionError, TypeError, ValueError):
        return str(value)


def _safe_json_loads(payload: str | None) -> Any:
    if not payload:
        return None
    try:
        return json.loads(payload)
    except (RecursionError, TypeError, ValueError):
        return None


def _coerce_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return coerced if coerced > 0 else None


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


def _coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, int):
        return bool(value)
    return None


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for entry in value:
        text = _normalize_optional_str(entry)
        if text:
            items.append(text)
    return items


def _parse_model_fallback_csv(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        parts = [entry if isinstance(entry, str) else "" for entry in value]
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        parts = raw.split(",")
    else:
        return []
    models: list[str] = []
    seen: set[str] = set()
    for part in parts:
        candidate = part.strip()
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        models.append(candidate)
    return models


def _select_best_effort_fallback(requested: str, supported: list[str]) -> str | None:
    ordering = ["none", "minimal", "low", "medium", "high", "xhigh"]
    if not supported:
        return None
    requested_lower = (requested or "").strip().lower()
    supported_lower = [value.strip().lower() for value in supported if value]
    if requested_lower in supported_lower:
        return requested_lower
    try:
        requested_idx = ordering.index(requested_lower)
    except ValueError:
        return supported_lower[0] if supported_lower else None
    indexed: list[tuple[int, str]] = []
    for value in supported_lower:
        try:
            indexed.append((ordering.index(value), value))
        except ValueError:
            continue
    if not indexed:
        return supported_lower[0] if supported_lower else None
    indexed.sort()
    min_idx, min_value = indexed[0]
    max_idx, max_value = indexed[-1]
    if requested_idx <= min_idx:
        return min_value
    if requested_idx >= max_idx:
        return max_value
    closest = None
    distance = float("inf")
    for idx, value in indexed:
        delta = abs(idx - requested_idx)
        if delta < distance or (delta == distance and value != "none"):
            closest = value
            distance = delta
    return closest


def _extract_marker_ulid(line: str) -> str | None:
    if not line:
        return None
    stripped = line.strip()
    if not stripped.startswith("[") or not stripped.endswith(_MARKER_SUFFIX):
        return None
    body = stripped[1 : -len(_MARKER_SUFFIX)]
    if len(body) != ULID_LENGTH:
        return None
    for char in body:
        if char not in _CROCKFORD_SET:
            return None
    return body


def contains_marker(text: str) -> bool:
    return _MARKER_SUFFIX in text and bool(_iter_marker_spans(text))


def is_hidden_marker_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("["):
        return False
    return (
        _extract_phase_marker_value(stripped) is not None
        or bool(_extract_marker_ulid(stripped))
        or _extract_kind_marker(stripped) is not None
    )


def ends_on_hidden_marker_line(text: Any) -> bool:
    if not isinstance(text, str):
        return False
    lines = text.rstrip().splitlines()
    return bool(lines) and is_hidden_marker_line(lines[-1])


def split_text_by_markers(text: str, spans: list[dict[str, Any]] | None = None) -> list[dict]:
    segments: list[dict[str, Any]] = []
    last = 0
    for span in _iter_marker_spans(text) if spans is None else spans:
        if span["start"] > last:
            segments.append({"type": "text", "text": text[last:span["start"]]})
        if span["marker"]:
            segments.append(
                {
                    "type": "marker",
                    "marker": span["marker"],
                }
            )
        last = span["end"]
    if last < len(text):
        segments.append({"type": "text", "text": text[last:]})
    return segments


def _extract_phase_marker_value(line: str) -> str | None:
    if not line:
        return None
    match = _PHASE_MARKER_RE.match(line.strip())
    if not match:
        return None
    token = match.group(1)
    if token not in _PHASE_MARKER_VALUES:
        return None
    return token


def split_text_by_phase_markers(text: str) -> list[dict[str, Any]]:
    if not text:
        return []

    segments: list[dict[str, Any]] = []
    last = 0
    for span in _iter_phase_marker_spans(text):
        segments.append(
            {
                "text": text[last:span["start"]],
                "start": last,
                "phase": None if span["phase_token"] == "null" else span["phase_token"],
                "phase_present": True,
            }
        )
        last = span["end"]
    if last < len(text):
        segments.append(
            {"text": text[last:], "start": last, "phase": None, "phase_present": False}
        )
    return segments


def strip_hidden_marker_lines(text: str) -> str:
    if not text:
        return ""

    kept_segments: list[str] = []
    removed = False
    for segment in text.splitlines(True):
        stripped = segment.strip()
        if not stripped:
            kept_segments.append(segment)
            continue
        if is_hidden_marker_line(stripped):
            removed = True
            continue
        kept_segments.append(segment)

    if not removed:
        return text
    return "".join(kept_segments)


_OPEN_WEBUI_CONFIG_MODULE: Any | None = None


def _get_open_webui_config_module() -> Any | None:
    global _OPEN_WEBUI_CONFIG_MODULE
    if _OPEN_WEBUI_CONFIG_MODULE is not None:
        return _OPEN_WEBUI_CONFIG_MODULE
    try:
        import open_webui.config as ow_config  # type: ignore
    except Exception:
        logger.debug("Open WebUI config module unavailable", exc_info=True)
        return None
    _OPEN_WEBUI_CONFIG_MODULE = ow_config
    return ow_config


_ADAPTER_CACHE = '''_ADAPTERS: dict = {}


def _adapters_for(cls: type) -> dict:
    return _ADAPTERS.setdefault(cls, {})'''


_KEEP_WHAT_STILL_FITS = '''        @model_validator(mode="before")
        @classmethod
        def _keep_what_still_fits(cls, data: Any) -> Any:
            """Drop stored values the model no longer publishes, keep the rest.

            These fields track a live contract, so a provider joining the model can
            narrow a range or remove a ratio while a value the user chose earlier is
            still stored. Open WebUI builds this class from that stored dict and passes
            no valves at all if construction raises -- so one stale entry silently threw
            away every other choice the user had made.
            """
            if not isinstance(data, dict):
                return data
            kept = {}
            table = _adapters_for(cls)
            for name in data:
                field = cls.model_fields.get(name)
                if field is None:
                    continue
                adapter = table.get(name)
                if adapter is None:
                    metadata = field.metadata
                    annotated = (
                        Annotated[(field.annotation, *metadata)]
                        if metadata
                        else field.annotation
                    )
                    adapter = table[name] = TypeAdapter(annotated)
                try:
                    adapter.validate_python(data[name])
                except ValidationError:
                    continue
                kept[name] = data[name]
            return kept'''


_DROP_UNUSABLE_PRIORITY = '''        @field_validator("priority", mode="before")
        @classmethod
        def _drop_unusable_priority(cls, value: Any, info: ValidationInfo) -> Any:
            field = cls.model_fields[info.field_name]
            table = _adapters_for(cls)
            adapter = table.get(info.field_name)
            if adapter is None:
                adapter = table[info.field_name] = TypeAdapter(field.annotation)
            try:
                adapter.validate_python(value)
            except ValidationError:
                kept = field.get_default()
                marker = (info.field_name, str(value), str(kept))
                seen = cls.__dict__.get("_warned_unusable_priority")
                if seen is None:
                    seen = set()
                    setattr(cls, "_warned_unusable_priority", seen)
                if marker not in seen:
                    seen.add(marker)
                    logging.getLogger(__name__).warning(
                        "Filter valve %s: stored value %r is not usable by this filter "
                        "build; using the field default %r",
                        info.field_name,
                        value,
                        kept,
                    )
                return field.get_default(call_default_factory=True)
            return value
'''


_PRIORITY_FIELD = (
    '        priority: int = Field(\n'
    '            default=0,\n'
    '            description="Priority level for the filter operations.",\n'
    '        )'
)


def _unwrap_config_value(value: Any) -> Any:
    if value is None:
        return None
    return getattr(value, "value", value)


_REDACTED_DATA_URL_MARKER = "[REDACTED]"

_BARE_BASE64_KEYS = frozenset({
    "b64_json", "b64", "image_base64", "base64", "data", "imageB64",
    "input_audio", "audio",
})

_MEDIA_URL_KEYS = frozenset({
    "image_url", "file_url", "video_url", "url", "file_data", "content_url",
    "audio", "last_image", "video", "videos", "images",
})


def _payload_key(key: str) -> str:
    return key.replace("_", "").casefold()


_MEDIA_KEY_STEMS = frozenset(_payload_key(k) for k in _MEDIA_URL_KEYS)

_BARE_BASE64_SHAPE = re.compile(r"[A-Za-z0-9+/_-]{1024,}={0,2}\Z")

_BARE_BASE64_RUN = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{1024,}")

_BARE_BASE64_RUN_PROBE = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{1024}")

_BASE64_FOLD_STEP = re.compile(r"\r?\n[ \t]*[A-Za-z0-9+/_-]+={0,2}[ \t]*(?=\r?\n|\Z)")

_BASE64_FOLD_SEED = re.compile(
    r"(?<![A-Za-z0-9+/_-])[ \t]*[A-Za-z0-9+/_-]+[ \t]*\r?\n"
    r"[ \t]*[A-Za-z0-9+/_-]+={0,2}[ \t]*(?=\r?\n|\Z)"
)

_BASE64_ALPHABET = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/-_="
)

_BARE_BASE64_FLOOR = 1024


def _folded_base64_end(text: str, stop: int) -> int:
    end = stop
    while True:
        step = _BASE64_FOLD_STEP.match(text, end)
        if step is None:
            return end
        end = step.end()


def _base64_span_size(text: str, start: int, end: int) -> int:
    return sum(
        len(text[at : min(at + _BASE64_FOLD_SLICE, end)].translate(_BASE64_FOLDED_TABLE))
        for at in range(start, end, _BASE64_FOLD_SLICE)
    )


def _bare_base64_spans(text: str) -> Iterator[tuple[int, int, int]]:
    last = 0
    runs = _BARE_BASE64_RUN.finditer(text)
    folds = _BASE64_FOLD_SEED.finditer(text)
    run = next(runs, None)
    fold = next(folds, None)
    while True:
        if run is not None and (fold is None or run.start() <= fold.start()):
            start, stop, folded = run.start(), run.end(), False
            run = next(runs, None)
        elif fold is not None:
            start, stop, folded = fold.start(), fold.end(), True
            fold = next(folds, None)
        else:
            return
        if start < last:
            continue
        if folded:
            end = _folded_base64_end(text, stop)
            size = _base64_span_size(text, start, end)
        else:
            end = stop
            size = end - start
        if size < _BARE_BASE64_FLOOR:
            continue
        last = end
        yield start, end, size


def _truncate_base64_runs(text: str, max_chars: int) -> str:
    if _BARE_BASE64_RUN_PROBE.search(text) is None and "\n" not in text:
        return text
    keep = max(8, min(64, max_chars // 4))
    pieces: list[str] = []
    last = 0
    for start, end, size in _bare_base64_spans(text):
        pieces.append(text[last:start])
        pieces.append(
            f"{text[start : start + keep]}…{_REDACTED_DATA_URL_MARKER}({size} chars)…"
        )
        last = end
    if not pieces:
        return text
    pieces.append(text[last:])
    return "".join(pieces)


def _truncate_bare_blob(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(8, min(64, max_chars // 4))
    return f"{text[:keep]}…{_REDACTED_DATA_URL_MARKER}({len(text)} chars)…"


def _stripped_len(text: str) -> int:
    start, end = 0, len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return end - start


def _redact_payload_blobs(value: Any, *, max_chars: int = 256) -> Any:
    def _redact_data_url(text: str) -> str:
        payload_len = base64_data_url_payload_len(text)
        if payload_len is None:
            return _data_url_log_subject(text)
        comma = text.find(",")
        header = _DATA_URL_NAME_PARAM.sub("", text[:comma])
        if _stripped_len(text) <= max_chars:
            return _data_url_log_subject(text)
        keep = max(8, min(64, max_chars // 4))
        body_at = comma + 1
        return (
            f"{header},{text[body_at : body_at + keep]}…{_REDACTED_DATA_URL_MARKER}"
            f"({payload_len} chars)…"
        )

    def _citation_url_holder(obj: dict[Any, Any]) -> bool:
        declared = obj.get("type")
        return isinstance(declared, str) and declared == "url_citation"

    def _walk(obj: Any, key: str = "", citation: bool = False) -> Any:
        if isinstance(obj, dict):
            own_citation = _citation_url_holder(obj)
            out: dict[Any, Any] = {}
            for k, v in obj.items():
                name = k if isinstance(k, str) else ""
                if own_citation and name == "url_citation" and isinstance(v, dict):
                    out[k] = _walk(v, name, True)
                elif isinstance(v, (dict, list, tuple)):
                    out[k] = _walk(v, name, False)
                else:
                    out[k] = _walk(v, name, own_citation or citation)
            return out
        if isinstance(obj, list):
            return [_walk(v, key, False) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_walk(v, key, False) for v in obj)
        if isinstance(obj, MediaSlot):
            lead = f"data:{obj.mime};base64," if obj.kind == "data_url" else ""
            return f"{lead}…{_REDACTED_DATA_URL_MARKER}({obj.b64_len} chars)…"
        if isinstance(obj, str):
            if _payload_key(key) in _MEDIA_KEY_STEMS and not (
                citation and _payload_key(key) == "url"
            ):
                return loggable_link(obj) or _REDACTED_DATA_URL_MARKER
            if base64_data_url_payload_len(obj) is not None:
                return _redact_data_url(obj)
            if key in _BARE_BASE64_KEYS or key.endswith("_b64"):
                return _truncate_bare_blob(obj, max_chars)
            return _redact_data_url(obj)
        return obj

    safe_max = int(max_chars) if isinstance(max_chars, int) else 256
    safe_max = max(64, min(safe_max, 8192))
    max_chars = safe_max
    return _walk(value)


def _extract_plain_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                text_val = block.get("text") or block.get("content")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return "\n".join(parts)
    if isinstance(content, dict):
        text_val = content.get("text") or content.get("content")
        if isinstance(text_val, str):
            return text_val
    return str(content or "")


def _extract_feature_flags(__metadata__: dict[str, Any]) -> dict[str, Any]:
    raw_features = __metadata__.get("features") if isinstance(__metadata__, dict) else None
    return dict(raw_features) if isinstance(raw_features, dict) else {}


_USAGE_SUMMABLE_KEYS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost",
        "total_cost",
        "input_cost",
        "output_cost",
        "prompt_cost",
        "completion_cost",
        "prompt_tokens",
        "completion_tokens",
        "cache_discount",
        "turn_count",
        "function_call_count",
    }
)

_USAGE_DETAIL_KEYS = frozenset(
    {
        "prompt_tokens_details",
        "completion_tokens_details",
        "input_tokens_details",
        "output_tokens_details",
    }
)


def _is_numeric_usage_value(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _merge_usage_map(total, new) -> None:
    for k, v in new.items():
        if isinstance(v, dict):
            target = total.get(k)
            if not isinstance(target, dict):
                target = {}
                total[k] = target
            _merge_usage_map(target, v)
            continue
        if _is_numeric_usage_value(v):
            previous = total.get(k)
            total[k] = (previous if _is_numeric_usage_value(previous) else 0) + v
        else:
            total[k] = v


def _merge_usage_into(total, new) -> None:
    for k, v in new.items():
        if k in _USAGE_DETAIL_KEYS:
            if isinstance(v, dict):
                target = total.get(k)
                if not isinstance(target, dict):
                    target = {}
                    total[k] = target
                _merge_usage_map(target, v)
            else:
                total[k] = v
            continue
        if k in _USAGE_SUMMABLE_KEYS and _is_numeric_usage_value(v):
            previous = total.get(k)
            total[k] = (previous if _is_numeric_usage_value(previous) else 0) + v
            continue
        if v is None and _is_numeric_usage_value(total.get(k)) and k in _USAGE_SUMMABLE_KEYS:
            continue
        total[k] = v


def merge_usage_stats(total, new):
    _merge_usage_into(total, new)
    return total


def wrap_code_block(text: str, language: str = "python") -> str:
    longest = max((len(m.group(0)) for m in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{language}\n{text}\n{fence}"


def _template_value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return True
    return bool(value)


def _normalize_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _clean_str(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _csv_set(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {part.strip().lower() for part in value.split(",") if part.strip()}


def _sanitize_path_component(value: str, *, fallback: str = "unknown", max_length: int = 128) -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    cleaned = cleaned.strip("._-") or fallback
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip("._-") or fallback
    return cleaned


def _retry_after_seconds(value: str | None) -> float | None:
    import datetime
    import email.utils

    if not value:
        return None
    trimmed = value.strip()
    if not trimmed:
        return None
    try:
        seconds = float(trimmed)
    except ValueError:
        pass
    else:
        return max(0.0, seconds) if math.isfinite(seconds) else None
    try:
        dt = email.utils.parsedate_to_datetime(trimmed)
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.UTC)
        now = datetime.datetime.now(datetime.UTC)
        seconds = (dt - now).total_seconds()
        return max(0.0, seconds)
    except (TypeError, ValueError, OverflowError):
        return None


def _apply_retry_after_metadata(meta: dict[str, Any], headers: Any) -> None:
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if not retry_after:
        return
    meta["retry_after"] = retry_after
    parsed = _retry_after_seconds(retry_after)
    if parsed is not None:
        meta["retry_after_seconds"] = round(parsed)


def _resolve_retry_after_seconds(meta: Any) -> int | None:
    if not isinstance(meta, dict):
        return None
    for key in ("retry_after_seconds", "retry_after"):
        value = meta.get(key)
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                continue
            return round(max(0.0, value))
        seconds = _retry_after_seconds(str(value))
        if seconds is not None:
            return round(seconds)
    return None


def _serialize_marker(ulid: str) -> str:
    return f"[{ulid}{_MARKER_SUFFIX}"


def _serialize_phase_marker(phase: str | None) -> str:
    marker_value = "null" if phase is None else phase.strip()
    if marker_value not in _PHASE_MARKER_VALUES:
        raise ValueError(f"Unsupported phase marker value: {marker_value!r}")
    return f"[P:{marker_value}{_MARKER_SUFFIX}"


def _iter_marker_spans(text: str) -> list[dict[str, Any]]:
    if not text:
        return []

    spans: list[dict[str, Any]] = []
    cursor = 0
    for segment in text.splitlines(True):
        stripped = segment.strip()
        marker_ulid = _extract_marker_ulid(stripped)
        if marker_ulid:
            offset = segment.find(stripped)
            start = cursor + (max(offset, 0))
            spans.append(
                {
                    "start": start,
                    "end": start + len(stripped),
                    "marker": marker_ulid,
                }
            )
        else:
            kind_marker = _extract_kind_marker(stripped)
            if kind_marker is not None:
                offset = segment.find(stripped)
                start = cursor + (max(offset, 0))
                spans.append(
                    {
                        "start": start,
                        "end": start + len(stripped),
                        "marker": None,
                        "kind": kind_marker[0],
                    }
                )
        cursor += len(segment)

    spans.sort(key=lambda span: span["start"])
    return spans


def _iter_phase_marker_spans(text: str) -> list[dict[str, Any]]:
    if not text:
        return []

    spans: list[dict[str, Any]] = []
    cursor = 0
    for segment in text.splitlines(True):
        stripped = segment.strip()
        phase_token = _extract_phase_marker_value(stripped)
        if phase_token is not None:
            offset = segment.find(stripped)
            start = cursor + (max(offset, 0))
            spans.append(
                {
                    "start": start,
                    "end": start + len(stripped),
                    "phase_token": phase_token,
                }
            )
        cursor += len(segment)

    spans.sort(key=lambda span: span["start"])
    return spans


_KIND_MARKER_NAMESPACE = "openrouter:v1:"
_KIND_MARKER_RE = re.compile(
    r"^\[" + re.escape(_KIND_MARKER_NAMESPACE) + r"([a-z][a-z0-9_-]*):([^\]]+)\]: #$"
)
_KIND_NAME_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_KIND_FORBIDDEN_BODY_CHARS = (
    "\n", "\r", "[", "]", "\x1c", "\x1d", "\x1e", "\x85", " ", " ",
)


def _serialize_kind_marker(kind: str, body: str) -> str:
    if not isinstance(kind, str) or not kind:
        raise ValueError("kind must be a non-empty string")
    if not _KIND_NAME_RE.match(kind):
        raise ValueError(f"invalid kind format: {kind!r}")
    if not isinstance(body, str) or not body:
        raise ValueError("body must be a non-empty string")
    for ch in _KIND_FORBIDDEN_BODY_CHARS:
        if ch in body:
            raise ValueError(f"marker body contains forbidden character: {ord(ch):#x}")
    return f"[{_KIND_MARKER_NAMESPACE}{kind}:{body}{_MARKER_SUFFIX}"


def _safe_marker_body(body: str) -> str:
    if not isinstance(body, str):
        body = str(body or "")
    for ch in _KIND_FORBIDDEN_BODY_CHARS:
        body = body.replace(ch, " ")
    body = body.strip()
    return body or "_"


def _extract_kind_marker(line: str) -> tuple[str, str] | None:
    if not line:
        return None
    match = _KIND_MARKER_RE.match(line.strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def _iter_kind_marker_spans(text: str, *, kind: str | None = None) -> list[dict[str, Any]]:
    if not text:
        return []
    spans: list[dict[str, Any]] = []
    cursor = 0
    for segment in text.splitlines(True):
        stripped = segment.strip()
        match = _extract_kind_marker(stripped)
        if match is not None:
            mk_kind, body = match
            if kind is None or mk_kind == kind:
                offset = segment.find(stripped)
                start = cursor + (max(offset, 0))
                spans.append(
                    {
                        "start": start,
                        "end": start + len(stripped),
                        "kind": mk_kind,
                        "body": body,
                    }
                )
        cursor += len(segment)
    spans.sort(key=lambda span: span["start"])
    return spans


def _find_first_kind_marker_body(text: str, kind: str) -> str:
    for span in _iter_kind_marker_spans(text, kind=kind):
        return str(span.get("body") or "")
    return ""


OWUI_FUNCTION_ID_ILLEGAL_RE = re.compile(r"[^a-zA-Z0-9_]")


async def _await_if_needed(
    value: Awaitable[_T] | _T,
    *,
    timeout: float | None = None,
) -> _T:
    if inspect.isawaitable(value):
        coroutine = cast(Awaitable[_T], value)
        if timeout is None:
            return cast(_T, await coroutine)
        return cast(_T, await asyncio.wait_for(coroutine, timeout=timeout))
    return cast(_T, value)


def citation_access_stamp() -> str:
    return datetime.datetime.now(datetime.UTC).astimezone().isoformat()


_TEXT_LIMIT = 120


def scrub_surrogates(text: str) -> str:
    return text.encode("utf-8", "surrogatepass").decode("utf-8", "replace")


def clamp_text(text: Any, limit: int = _TEXT_LIMIT) -> str:
    rendered = scrub_surrogates(text if isinstance(text, str) else str(text))
    return rendered if len(rendered) <= limit else f"{rendered[:limit]}…"


def summarise_names(names: list[str], limit: int = 4, width: int = _TEXT_LIMIT) -> str:
    shown = [clamp_text(name, width) for name in names[:limit]]
    tail = f" and {len(names) - len(shown)} more" if len(names) > len(shown) else ""
    return f"{'; '.join(shown)}{tail}"


_ANSWERED_BY_SHOWN = 3


def _answering_model_key(model_id: str) -> str:
    return model_id.casefold().split("@", 1)[0].split(":", 1)[0].strip()


def answered_by_text(requested_model: Any, answering_models: Iterable[Any]) -> str:
    if not (isinstance(requested_model, str) and requested_model.strip()):
        return ""
    requested_key = _answering_model_key(requested_model)
    by_key: dict[str, str] = {}
    for candidate in answering_models:
        if isinstance(candidate, str) and candidate.strip():
            shown = clamp_text(" ".join(candidate.split()))
            by_key.setdefault(_answering_model_key(shown), shown)
    if all(key == requested_key for key in by_key):
        return ""
    ids = list(by_key.values())
    text = ", ".join(ids[:_ANSWERED_BY_SHOWN])
    hidden = len(ids) - _ANSWERED_BY_SHOWN
    return f"{text} and {hidden} more" if hidden > 0 else text


CONTINUED_REPLY: ContextVar[str | None] = ContextVar("continued_reply", default=None)

ATTACHMENT_NOTICES: ContextVar[list[str] | None] = ContextVar("attachment_notices", default=None)


def note_for_the_end(notice: str) -> bool:
    collector = ATTACHMENT_NOTICES.get()
    if collector is None:
        return False
    collector.append(notice)
    return True


def closing_status_descriptions(
    description: str | None, notices: Iterable[str], *, join: bool
) -> list[str]:
    end_notice = " ".join(dict.fromkeys(notices))
    if not end_notice:
        return [] if description is None else [description]
    if not description:
        return [end_notice]
    if join:
        return [f"{description} | {end_notice}"]
    return [description, end_notice]


def take_closing_notices(body: Any, *, silenced: bool = False) -> list[str]:
    notices = getattr(body, "_attachment_notices", None)
    if silenced or not isinstance(notices, list) or not notices:
        return []
    if getattr(body, "_attachment_notices_shown", False):
        return []
    body._attachment_notices_shown = True
    return list(notices)


def continued_reply_text(body: Any, metadata: Any) -> str | None:
    if not (isinstance(metadata, dict) and metadata.get("assistant_message_id")):
        return None
    messages = body.get("messages") if isinstance(body, dict) else None
    last = messages[-1] if isinstance(messages, list) and messages else None
    if not (isinstance(last, dict) and last.get("role") == "assistant"):
        return None
    content = last.get("content")
    return content if isinstance(content, str) else ""


def join_answer_and_card(answer: str, card: str) -> str:
    if not card:
        return answer
    card = card.lstrip("\n")
    if not answer:
        return f"\n\n{card}" if CONTINUED_REPLY.get() is not None else card
    return f"{answer}\n\n{card}"


def utf8_stream_decoder() -> codecs.IncrementalDecoder:
    return codecs.getincrementaldecoder("utf-8")(errors="replace")


class _NoValves:
    HTTP_CONNECT_TIMEOUT_SECONDS = None
    HTTP_TOTAL_TIMEOUT_SECONDS = None
    HTTP_SOCK_READ_SECONDS = None



_DEFAULT_VALVES = _NoValves()


def http_timeout(valves: Any) -> aiohttp.ClientTimeout:
    connect = getattr(valves, "HTTP_CONNECT_TIMEOUT_SECONDS", None)
    total_value = getattr(valves, "HTTP_TOTAL_TIMEOUT_SECONDS", None)
    total = total_value if total_value else None
    sock_read = getattr(valves, "HTTP_SOCK_READ_SECONDS", None) if total is None else None
    return aiohttp.ClientTimeout(total=total, connect=connect, sock_read=sock_read)


def http_timeout_str(valves: Any) -> str:
    timeout = http_timeout(valves)
    return (
        f"connect={timeout.connect}s "
        f"total={timeout.total if timeout.total is not None else 'disabled'} "
        f"sock_read={timeout.sock_read if timeout.sock_read is not None else 'disabled'}"
    )
