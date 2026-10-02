"""Configuration management for OpenRouter pipe.

This module contains all configuration schemas, constants, and valve definitions:
- Valves: Global configuration (API keys, timeouts, model lists, etc.)
- UserValves: Per-user configuration overrides
- EncryptedStr: Secret value encryption wrapper
- Error template constants
- Pipe configuration constants
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import os
import re
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Annotated, Any, Literal, cast

from cryptography.fernet import Fernet, InvalidToken
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from pydantic_core import core_schema

from .fusion_defaults import (
    DEFAULT_FUSION_JUDGE_SYSTEM_PROMPT,
    DEFAULT_FUSION_PANEL_SYSTEM_PROMPT,
    DEFAULT_FUSION_SYNTHESIS_SYSTEM_PROMPT,
)
from .url_scheme import (
    base64_data_url_payload_len,
    is_http_or_https_url,
    url_scheme,
)
from .valve_salvage import (
    _STALE_VALVES_WARN_EVERY_S,  # noqa: F401
    _VALVE_SCHEMA_CACHE,  # noqa: F401
    _valve_schema,  # noqa: F401
    _warned_stale_valves,  # noqa: F401
    drop_unvalidatable,  # noqa: F401
    is_secret_field,  # noqa: F401
    repair_unvalidatable,
)
from .warn_latch import warn_level

try:
    from open_webui import env as _owui_env
    from open_webui.utils.headers import (
        include_user_info_headers as _owui_include_user_info_headers,
    )
except ImportError:
    _owui_env = None
    _owui_include_user_info_headers = None
except Exception:
    logging.getLogger(__name__).warning(
        "open_webui failed to import for a reason other than absence; "
        "the features that depend on it are now disabled",
        exc_info=True,
    )
    _owui_env = None
    _owui_include_user_info_headers = None

logger = logging.getLogger(__name__)

_warned_forward_headers: set[str] = set()
_warned_bzip2_compresslevel: set[str] = set()

_BZIP2_COMPRESSLEVEL_ADAPTER: TypeAdapter[int | None] = TypeAdapter(int | None)


def _warn_bzip2_level_floored() -> None:
    logger.log(
        warn_level(_warned_bzip2_compresslevel, "0"),
        "Session log zip compression is bzip2 at level 0, which bzip2 cannot use; "
        "writing archives at level 1. bzip2 takes levels 1-9: set Session log zip "
        "compress level to 1-9, or choose another codec. The stored setting still "
        "reads 0 until it is saved again.",
    )


# Constants

_OPENROUTER_TITLE = "Open WebUI plugin for OpenRouter Responses API"
_OPENROUTER_CATEGORIES = "general-chat"
_OPENROUTER_REFERER = "https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/"
_DEFAULT_PIPE_ID = "open_webui_openrouter_pipe"
_FUNCTION_MODULE_PREFIX = "function_"
_OPENROUTER_FRONTEND_MODELS_URL = "https://openrouter.ai/api/frontend/v1/catalog/models"
_OPENROUTER_MODEL_ENDPOINTS_URL_TEMPLATE = "https://openrouter.ai/api/v1/models/{slug}/endpoints"
_OPENROUTER_SITE_URL = "https://openrouter.ai"
_OPENROUTER_HOST = "openrouter.ai"
_MAX_MODEL_PROFILE_IMAGE_BYTES = 2 * 1024 * 1024
_MAX_MODEL_PROFILE_IMAGE_PIXELS = 25_000_000
_MAX_OPENROUTER_ID_CHARS = 128
_MAX_OPENROUTER_METADATA_PAIRS = 16
_MAX_OPENROUTER_METADATA_KEY_CHARS = 64
_MAX_OPENROUTER_METADATA_VALUE_CHARS = 512

_PIPE_METADATA_KEY = "openrouter_pipe"

_BOOLEAN_PLACEHOLDER_RULE = (
    "A boolean placeholder renders `True` when the value is true, and its line is dropped when the "
    "value is false; the `{{#if}}` form of the same value behaves the same way."
)

_CHANNEL_CARD_RULE = (
    "On a channel chat, which every member of the room reads, the values behind session_id, user_id, "
    "detail, sanitized_detail, reason, openrouter_message, upstream_message, moderation_reasons, "
    "flagged_excerpt, raw_body, metadata_json, provider_raw_json and body_excerpt are withheld, and the card "
    "is rendered "
    "as though each were empty. Wrap a line that uses one in {{#if name}} and it is left out; otherwise the whole "
    "line is omitted, exactly as any other line whose value came out empty; a name the pipe never "
    "supplies at all is the one left in the text verbatim. error_id, the model, "
    "the provider, openrouter_code and status_code still render on a channel, and error_id is the handle to "
    "quote when following up there."
)

_WRITABLE_FRAME_MIMES = frozenset({"image/jpeg", "image/png", "image/webp"})


def _frame_allowlist_is_satisfiable(value: Any) -> bool:
    from .utils import _csv_set

    return bool(_csv_set(value) & _WRITABLE_FRAME_MIMES)


def _check_frame_allowlist(value: str) -> str:
    from .utils import _csv_set

    if not _csv_set(value):
        return value
    if not _frame_allowlist_is_satisfiable(value):
        raise ValueError("name at least one of image/jpeg, image/png or image/webp")
    return value


_DEFAULT_RESPONSES_AUDIO_FORMATS = frozenset({"mp3", "wav"})

_UNMAPPABLE_AUDIO_FORMATS = frozenset({"webm"})

# OpenRouter Web Tools filter
_OPENROUTER_WEB_TOOLS_FILTER_MARKER = "openrouter_pipe:web_tools_filter:v1"
_OPENROUTER_WEB_TOOLS_FILTER_PREFERRED_FUNCTION_ID = "openrouter_web_tools"

# OpenRouter Image Generation filter
_OPENROUTER_IMAGE_GEN_FILTER_MARKER = "openrouter_pipe:image_gen_filter:v1"
_OPENROUTER_IMAGE_GEN_FILTER_PREFERRED_FUNCTION_ID = "openrouter_image_gen"
_OPENROUTER_IMAGE_GEN_FILTER_DEFAULT_MODEL = "openai/gpt-5-image-mini"

# OpenRouter Video Generation filter
_OPENROUTER_VIDEO_GEN_FILTER_MARKER = "openrouter_pipe:video_filter:v1"

_OPENROUTER_IMAGE_FILTER_MARKER = "openrouter_pipe:image_filter:v1"

_OPENROUTER_FUSION_FILTER_MARKER = "openrouter_pipe:fusion_filter:v1"
_OPENROUTER_FUSION_FILTER_PREFERRED_FUNCTION_ID = "openrouter_fusion"

_DIRECT_UPLOADS_FILTER_MARKER = "openrouter_pipe:direct_uploads_filter:v1"
_DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID = "openrouter_direct_uploads"

_PROVIDER_ROUTING_FILTER_MARKER_PREFIX = "openrouter_pipe:provider_routing:"
_PROVIDER_ROUTING_FILTER_MARKER_VERSION = "v1"
_PROVIDER_ROUTING_FILTER_ID_PREFIX = "openrouter_provider_"
_PROVIDER_SLUG_PATTERN = re.compile(r"^[a-z0-9-]+(?:/[a-z0-9-]+)?$")
_PROVIDER_ROUTING_ORDER_PERMUTATION_MAX = 4
_PROVIDER_ROUTING_MAX_PROVIDERS = 100
_PROVIDER_ROUTING_OVERLAY_MAX_MODELS = 50

_NON_REPLAYABLE_TOOL_ARTIFACTS = frozenset(
    {
        "image_generation_call",
        "openrouter:image_generation",
        "openrouter:web_search",
        "openrouter:web_fetch",
        "openrouter:datetime",
        "web_search_call",
        "file_search_call",
        "local_shell_call",
    }
)

_RAW_REPLAYED_SERVER_TOOLS = frozenset(
    {
        "openrouter:advisor",
        "openrouter:subagent",
        "openrouter:experimental__search_models",
    }
)

_EMPTY_TOOL_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}

_REMOTE_FILE_MAX_SIZE_DEFAULT_MB = 50
_REMOTE_FILE_MAX_SIZE_MAX_MB = 500
_INTERNAL_FILE_ID_PATTERN = re.compile(r"/files/([A-Za-z0-9-]+)(?:[/?#]|$)")
_RUN = r"(?:[^ \t()\n]++|\n(?![\"'(]))"
_MARKDOWN_IMAGE_RE = re.compile(
    r"!\[[^\]]*+\]\(\s*(?:<(?P<angled>[^<>\n]*)>|"
    r"(?P<bare>" + _RUN + r"*(?:\(" + _RUN + r"*\)" + _RUN + r"*)*))"
    r"(?:\s+(?:\"[^\"]*\"|'[^']*'|\([^()]*\)))?\s*\)"
)


_MARKDOWN_IMAGE_BODY_RE = re.compile(
    r"data:[A-Za-z0-9#$&^_.+-]*/[A-Za-z0-9#$&^_.+-]*"
    r"(?:;[A-Za-z0-9#$&^_.+-]+=[A-Za-z0-9#$&^_.+-]+)*"
    r";base64,[A-Za-z0-9+/=]{4096,}"
)
_MARKDOWN_IMAGE_BLANK = "\x00" * 32


def _blank_long_image_bodies(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    blanks: list[tuple[int, int, str]] = []
    pieces: list[str] = []
    cursor = 0
    length = 0
    for match in _MARKDOWN_IMAGE_BODY_RE.finditer(text):
        pieces.append(text[cursor:match.start()])
        length += match.start() - cursor
        blanks.append((length, length + len(_MARKDOWN_IMAGE_BLANK), match.group(0)))
        pieces.append(_MARKDOWN_IMAGE_BLANK)
        length += len(_MARKDOWN_IMAGE_BLANK)
        cursor = match.end()
    pieces.append(text[cursor:])
    return "".join(pieces), blanks


def _restore_blanked_image_body(
    shrunk: str, blanks: list[tuple[int, int, str]], start: int, end: int
) -> tuple[str, int]:
    parts: list[str] = []
    cursor = start
    shift = 0
    for blank_start, blank_end, original in blanks:
        if blank_end <= start:
            shift += len(original) - (blank_end - blank_start)
            continue
        if blank_start >= end:
            break
        parts.append(shrunk[cursor:blank_start])
        parts.append(original)
        cursor = blank_end
    parts.append(shrunk[cursor:end])
    return "".join(parts), start + shift


def markdown_image_spans(text: str) -> list[tuple[str, int, int]]:
    if not isinstance(text, str) or "](" not in text:
        return []
    shrunk, blanks = _blank_long_image_bodies(text)
    spans: list[tuple[str, int, int]] = []
    for match in _MARKDOWN_IMAGE_RE.finditer(shrunk):
        group = "angled" if match.group("angled") is not None else "bare"
        destination = match.group(group)
        if not destination:
            continue
        if blanks:
            restored, start = _restore_blanked_image_body(
                shrunk, blanks, *match.span(group)
            )
        else:
            restored, start = destination, match.start(group)
        destination = restored.strip()
        if not destination:
            continue
        lead = len(restored) - len(restored.lstrip())
        spans.append((destination, start + lead, start + lead + len(destination)))
    return spans


def markdown_image_destinations(text: str) -> list[str]:
    return [url for url, _start, _end in markdown_image_spans(text)]


_ENTRY_DATA_URL_KEYS = ("url", "content", "data")


def entry_data_url(entry: Any) -> str:
    if not isinstance(entry, dict):
        return ""
    for key in _ENTRY_DATA_URL_KEYS:
        value = entry.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = value.strip()
        if url_scheme(candidate) == "data" and base64_data_url_payload_len(candidate) is not None:
            return candidate
    return ""


# ULID generation constants
ULID_LENGTH = 20
ULID_TIME_LENGTH = 16
ULID_RANDOM_LENGTH = ULID_LENGTH - ULID_TIME_LENGTH
CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_TIME_MASK = (1 << (ULID_TIME_LENGTH * 5)) - 1

NO_CONTENT_AFTER_TOOLS_FALLBACK = (
    "I couldn't produce a final answer after running tools. "
    "Please retry with a narrower tool query or a shorter context window."
)

OPENAI_EMPTY_USER_TURN_FALLBACK = "[The user sent an empty message.]"
OPENAI_ATTACHMENT_NOT_SENT_PREFIX = "[An attached item was not sent: "

DEFAULT_OPENROUTER_ERROR_TEMPLATE = (
    "{{#if heading}}\n"
    "### 🚫 {heading} could not process your request.\n\n"
    "{{/if}}\n"
    "{{#if error_id}}\n"
    "- **Error ID**: `{error_id}`\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "- **Time**: {timestamp}\n"
    "{{/if}}\n"
    "{{#if session_id}}\n"
    "- **Session**: `{session_id}`\n"
    "{{/if}}\n"
    "{{#if user_id}}\n"
    "- **User**: `{user_id}`\n"
    "{{/if}}\n"
    "{{#if sanitized_detail}}\n"
    "### Error: `{sanitized_detail}`\n\n"
    "{{/if}}\n"
    "{{#if openrouter_message}}\n"
    "- **OpenRouter message**: `{openrouter_message}`\n"
    "{{/if}}\n"
    "{{#if upstream_message}}\n"
    "- **Provider message**: `{upstream_message}`\n"
    "{{/if}}\n"
    "{{#if model_identifier}}\n"
    "- **Model**: `{model_identifier}`\n"
    "{{/if}}\n"
    "{{#if provider}}\n"
    "- **Provider**: `{provider}`\n"
    "{{/if}}\n"
    "{{#if requested_model}}\n"
    "- **Requested model**: `{requested_model}`\n"
    "{{/if}}\n"
    "{{#if api_model_id}}\n"
    "- **API model id**: `{api_model_id}`\n"
    "{{/if}}\n"
    "{{#if normalized_model_id}}\n"
    "- **Normalized model id**: `{normalized_model_id}`\n"
    "{{/if}}\n"
    "{{#if openrouter_code}}\n"
    "- **OpenRouter code**: `{openrouter_code}`\n"
    "{{/if}}\n"
    "{{#if upstream_type}}\n"
    "- **Provider error**: `{upstream_type}`\n"
    "{{/if}}\n"
    "{{#if reason}}\n"
    "- **Reason**: `{reason}`\n"
    "{{/if}}\n"
    "{{#if request_id}}\n"
    "- **Request ID**: `{request_id}`\n"
    "{{/if}}\n"
    "{{#if native_finish_reason}}\n"
    "- **Finish reason**: `{native_finish_reason}`\n"
    "{{/if}}\n"
    "{{#if error_chunk_id}}\n"
    "- **Chunk ID**: `{error_chunk_id}`\n"
    "{{/if}}\n"
    "{{#if error_chunk_created}}\n"
    "- **Chunk time**: {error_chunk_created}\n"
    "{{/if}}\n"
    "{{#if streaming_provider}}\n"
    "- **Streaming provider**: `{streaming_provider}`\n"
    "{{/if}}\n"
    "{{#if streaming_model}}\n"
    "- **Streaming model**: `{streaming_model}`\n"
    "{{/if}}\n"
    "{{#if include_model_limits}}\n"
    "\n**Model limits:**\n"
    "{{#if context_limit_tokens}}Context window: {context_limit_tokens} tokens\n{{/if}}\n"
    "{{#if max_output_tokens}}Max output tokens: {max_output_tokens} tokens\n{{/if}}\n"
    "Shorten the conversation or lower the requested output to stay within these limits. "
    "An admin can also switch on `Auto-trim overlong prompts`, which lets OpenRouter compress an "
    "over-long prompt to fit instead of rejecting it.\n"
    "{{/if}}\n"
    "{{#if moderation_reasons}}\n"
    "\n**Moderation reasons:**\n"
    "{moderation_reasons}\n"
    "Please review the flagged content or contact your administrator if you believe this is a mistake.\n"
    "{{/if}}\n"
    "{{#if flagged_excerpt}}\n"
    "\n**Flagged text excerpt:**\n"
    "{flagged_excerpt}\n"
    "Provide this excerpt when following up with your administrator.\n"
    "{{/if}}\n"
    "{{#if raw_body}}\n"
    "\n**Raw provider response:**\n"
    "{raw_body}\n"
    "{{/if}}\n"
    "{{#if metadata_json}}\n"
    "\n**Metadata:**\n"
    "{metadata_json}\n"
    "{{/if}}\n"
    "{{#if provider_raw_json}}\n"
    "\n**Provider raw error:**\n"
    "{provider_raw_json}\n"
    "{{/if}}\n\n"
    "Please adjust the request and try again, or contact your administrator if it keeps failing.\n"
    "{{#if request_id_reference}}\n"
    "{request_id_reference}\n"
    "{{/if}}\n"
)

DEFAULT_NETWORK_TIMEOUT_TEMPLATE = (
    "### ⏱️ Request Timeout\n\n"
    "The request to OpenRouter took too long to complete.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if timeout_seconds}}\n"
    "**Timeout:** {timeout_seconds}s\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**Possible causes:**\n"
    "- OpenRouter's servers are slow or overloaded\n"
    "- Network congestion\n"
    "- Large request taking longer than expected\n\n"
    "**What to do:**\n"
    "- Wait a few moments and try again\n"
    "- Try a smaller request if possible\n"
    "- Check [OpenRouter Status](https://status.openrouter.ai/)\n"
    "{{#if support_email}}\n"
    "- Contact support: {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_CONNECTION_ERROR_TEMPLATE = (
    "### 🔌 Connection Failed\n\n"
    "The connection to OpenRouter failed, or ended before a reply arrived.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if error_type}}\n"
    "**Error type:** `{error_type}`\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**Possible causes:**\n"
    "- Network connectivity issues\n"
    "- Firewall blocking HTTPS traffic\n"
    "- DNS resolution failure\n"
    "- OpenRouter closed the connection without sending a reply\n"
    "- OpenRouter service outage\n\n"
    "**What to do:**\n"
    "1. Try the message again\n"
    "2. Check your internet connection\n"
    "3. Verify firewall allows HTTPS (port 443)\n"
    "4. Check [OpenRouter Status](https://status.openrouter.ai/)\n"
    "5. Contact your network administrator if the issue persists\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_SERVICE_ERROR_TEMPLATE = (
    "### 🔴 OpenRouter Service Error\n\n"
    "OpenRouter returned a server-side error instead of a reply.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if request_id}}\n"
    "**OpenRouter request ID:** `{request_id}`\n"
    "{{/if}}\n"
    "{{#if status_code}}\n"
    "**Status:** {status_code}\n"
    "{{/if}}\n"
    "{{#if reason}}\n"
    "**Details:** {reason}\n"
    "{{/if}}\n"
    "{{#if body_excerpt}}\n"
    "**Response body (not an OpenRouter reply):**\n"
    "{body_excerpt}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**What the status means:**\n"
    "- `502` — the chosen model is down, or it returned something OpenRouter could not read, or a proxy, "
    "CDN or WAF in front of this deployment rewrote the reply\n"
    "- `503` — the provider is momentarily overloaded, or no provider was available that satisfies the "
    "routing requirements sent with this request\n"
    "- `504` — the provider did not answer in time\n"
    "- any other `5xx` — a fault inside OpenRouter itself\n\n"
    "**What to do:**\n"
    "- Retry in a few minutes; an overloaded provider or a model outage normally clears on its own\n"
    "- Try a different model, which routes to a different set of providers\n"
    "- If a `503` keeps repeating rather than clearing, it may be the routing constraints rather than load: "
    "an admin can review `Enforce ZDR routing` and the provider-routing settings for this model\n"
    "- Check [OpenRouter Status](https://status.openrouter.ai/) for a platform-wide incident\n"
    "- Check the network path, and any proxy between this host and OpenRouter, if the details quote a body "
    "that is not an OpenRouter response\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_INTERNAL_ERROR_TEMPLATE = (
    "### ⚠️ Unexpected Error\n\n"
    "Something unexpected went wrong while processing your request.\n\n"
    "**Error ID:** `{error_id}` -- Share this with support\n"
    "{{#if error_type}}\n"
    "**Error type:** `{error_type}`\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "The error has been logged and will be investigated.\n\n"
    "**What to do:**\n"
    "- Try your request again\n"
    "- If the problem persists, contact support with the Error ID above\n"
    "{{#if support_email}}\n"
    "- Email: {support_email}\n"
    "{{/if}}\n"
    "{{#if support_url}}\n"
    "- Support: {support_url}\n"
    "{{/if}}\n"
)

DEFAULT_ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE = (
    "### ⚠️ Endpoint Override Conflict\n\n"
    "This request requires a different OpenRouter endpoint than the one enforced for the selected model.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if requested_model}}\n"
    "**Model:** `{requested_model}`\n"
    "{{/if}}\n"
    "{{#if required_endpoint}}\n"
    "**Required endpoint:** `{required_endpoint}`\n"
    "{{/if}}\n"
    "{{#if enforced_endpoint}}\n"
    "**Enforced endpoint:** `{enforced_endpoint}`\n"
    "{{/if}}\n"
    "{{#if reason}}\n"
    "**Reason:** {reason}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**What to do:**\n"
    "- Ask an admin to adjust the model endpoint override (or choose a different model)\n"
    "- Or remove the attachment(s) and retry, unless the requirement is the model's own\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_DIRECT_UPLOAD_FAILURE_TEMPLATE = (
    "### ⚠️ Direct Upload Issue\n\n"
    "OpenRouter Direct Uploads could not be applied to this request.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if requested_model}}\n"
    "**Model:** `{requested_model}`\n"
    "{{/if}}\n"
    "{{#if reason}}\n"
    "**Reason:** {reason}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**What to do:**\n"
    "- Split your attachments into separate messages (e.g. documents in one message, audio/video in another)\n"
    "- Temporarily disable one of the Direct Uploads valves (Files / Audio / Video) and retry\n"
    "- Convert media to a supported format (for example: audio to mp3/wav)\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_AUTHENTICATION_ERROR_TEMPLATE = (
    "### 🔐 Authentication Failed\n\n"
    "This request was not authorised: OpenRouter rejected the pipe's API key, or the pipe could not read one.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if request_id}}\n"
    "**OpenRouter request ID:** `{request_id}`\n"
    "{{/if}}\n"
    "{{#if openrouter_code}}\n"
    "**Status:** {openrouter_code}\n"
    "{{/if}}\n"
    "{{#if openrouter_message}}\n"
    "**Details:** {openrouter_message}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**What an admin should check:**\n"
    "1. The `OpenRouter API key` valve in this pipe's settings — a blank or truncated value fails here\n"
    "2. Whether `WEBUI_SECRET_KEY`, or the deprecated `WEBUI_JWT_SECRET_KEY` it falls back to (a default, so an empty primary is not a fallback) changed since that key was saved; the stored value can no longer be decrypted, "
    "so the key has to be entered again\n"
    "3. Whether the key itself was disabled or deleted — issue a replacement at https://openrouter.ai/keys\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE = (
    "### 💳 Insufficient Credits\n\n"
    "OpenRouter could not complete this request: the account is out of credits.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if request_id}}\n"
    "**OpenRouter request ID:** `{request_id}`\n"
    "{{/if}}\n"
    "{{#if openrouter_code}}\n"
    "**Status:** {openrouter_code}\n"
    "{{/if}}\n"
    "{{#if openrouter_message}}\n"
    "**Details:** {openrouter_message}\n"
    "{{/if}}\n"
    "{{#if required_cost}}\n"
    "**Estimated cost:** ${required_cost}\n"
    "{{/if}}\n"
    "{{#if account_balance}}\n"
    "**Current balance:** ${account_balance}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**What to do:**\n"
    "- Add credits at https://openrouter.ai/credits\n"
    "- Review what the account has been spending on the Activity page: https://openrouter.ai/activity\n"
    "- Turn on auto top up so the balance refills before it runs out\n"
    "- A negative balance blocks the `:free` model variants too; clearing it restores them\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_RATE_LIMIT_TEMPLATE = (
    "### ⏸️ Rate Limit Exceeded\n\n"
    "OpenRouter could not complete this request: the account has reached one of its request limits.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if request_id}}\n"
    "**OpenRouter request ID:** `{request_id}`\n"
    "{{/if}}\n"
    "{{#if openrouter_code}}\n"
    "**Status:** {openrouter_code}\n"
    "{{/if}}\n"
    "{{#if retry_after_seconds}}\n"
    "**Retry after:** {retry_after_seconds}s\n"
    "{{/if}}\n"
    "{{#if rate_limit_type}}\n"
    "**Limit type:** {rate_limit_type}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**Tips:**\n"
    "- Back off and retry with exponential delays, honouring the retry-after value when one is shown\n"
    "- Queue requests or lower parallelism when the limit is the per-minute one\n"
    "- `:free` model variants carry their own per-minute and per-day caps. Those caps count every "
    "`:free` request the account makes, whichever free model it names, so moving to a different free "
    "model does not lift them; buying credits raises the daily one\n"
    "- On a paid model the limits differ from model to model, so switching to another paid model "
    "spreads the load\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_SERVER_TIMEOUT_TEMPLATE = (
    "### 🕒 OpenRouter Timed Out\n\n"
    "OpenRouter cancelled the request: it exceeded its time limit.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if request_id}}\n"
    "**OpenRouter request ID:** `{request_id}`\n"
    "{{/if}}\n"
    "{{#if openrouter_code}}\n"
    "**Status:** {openrouter_code}\n"
    "{{/if}}\n"
    "{{#if openrouter_message}}\n"
    "**Details:** {openrouter_message}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**Next steps:**\n"
    "- Retry shortly; the upstream provider may be busy\n"
    "- Reduce prompt size or requested work\n"
    "- Contact support if the issue persists\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_PAYLOAD_TOO_LARGE_TEMPLATE = (
    "### 📦 Request Too Large\n\n"
    "The request payload exceeds the size limit accepted by OpenRouter. This is a cap on how large the "
    "request itself may be, not on the model's context window.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if request_id}}\n"
    "**OpenRouter request ID:** `{request_id}`\n"
    "{{/if}}\n"
    "{{#if openrouter_code}}\n"
    "**Status:** {openrouter_code}\n"
    "{{/if}}\n"
    "{{#if openrouter_message}}\n"
    "**Details:** {openrouter_message}\n"
    "{{/if}}\n"
    "{{#if model_identifier}}\n"
    "**Model:** `{model_identifier}`\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**What to do:**\n"
    "- Remove or compress attachments; inlined images, audio and documents dominate the payload size\n"
    "- Shorten the conversation history, which is resent in full on every turn\n"
    "- Send large media in its own message rather than alongside a long prompt\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_MODEL_RESTRICTED_TEMPLATE = (
    "### 🚫 Model restricted\n\n"
    "This pipe rejected the requested model due to configuration restrictions.\n\n"
    "- **Requested model**: `{requested_model}`\n"
    "{{#if normalized_model_id}}\n"
    "- **Normalized model id**: `{normalized_model_id}`\n"
    "{{/if}}\n"
    "{{#if restriction_reasons}}\n"
    "- **Restricted by**: {restriction_reasons}\n"
    "{{/if}}\n"
    "{{#if model_id_filter}}\n"
    "- **Model allowlist**: `{model_id_filter}`\n"
    "{{/if}}\n"
    "{{#if free_model_filter}}\n"
    "- **Free model visibility**: `{free_model_filter}`\n"
    "{{/if}}\n"
    "{{#if tool_calling_filter}}\n"
    "- **Tool-calling model filter**: `{tool_calling_filter}`\n"
    "{{/if}}\n\n"
    "Choose an allowed model or ask your admin to update the pipe filters.\n"
)

DEFAULT_VIDEO_CATALOG_LOADING_TEMPLATE = (
    "### ⏳ Video catalogue still loading\n\n"
    "This pipe is still reading the video model catalogue, so `{requested_model}` is not "
    "in its list yet. Nothing is misconfigured: the next turn will carry this model.\n\n"
    "- **Requested model**: `{requested_model}`\n"
    "{{#if normalized_model_id}}\n"
    "- **Normalized model id**: `{normalized_model_id}`\n"
    "{{/if}}\n"
)

DEFAULT_STREAM_INTERRUPTED_TEMPLATE = (
    "---\n"
    "### ⚠️ Response interrupted\n\n"
    "The stream ended unexpectedly before the model finished responding.\n\n"
    "{{#if model}}\n"
    "**Model:** `{model}`\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "Retry the message — this is usually a transient issue.\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
    "{{#if support_url}}\n"
    "\n**Support:** {support_url}\n"
    "{{/if}}\n"
)


# EncryptedStr and Helper Functions

_FERNET_MIN_BODY = 73
_FERNET_VERSION_BYTE = 0x80
_FERNET_VERSION_HEAD = base64.urlsafe_b64encode(bytes([_FERNET_VERSION_BYTE])).decode()[0]


def _application_secret() -> str:
    return os.getenv("WEBUI_SECRET_KEY", os.getenv("WEBUI_JWT_SECRET_KEY", ""))


class EncryptedStr(str):
    """String wrapper that automatically encrypts/decrypts valve values."""

    _ENCRYPTION_PREFIX = "encrypted:"

    @classmethod
    def _get_encryption_key(cls) -> bytes | None:
        """Return the Fernet key derived from ``WEBUI_SECRET_KEY``.

        Returns:
            Optional[bytes]: URL-safe base64 Fernet key or ``None`` when unset.
        """
        secret = _application_secret()
        if not secret:
            return None
        hashed_key = hashlib.sha256(secret.encode()).digest()
        return base64.urlsafe_b64encode(hashed_key)

    @classmethod
    def encrypt(cls, value: str) -> str:
        """Encrypt ``value`` when an application secret is configured.

        Args:
            value: Plain-text string supplied by the user.

        Returns:
            str: Ciphertext prefixed with ``encrypted:`` or the original value.
        """
        if not value or value.startswith(cls._ENCRYPTION_PREFIX):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value
        fernet = Fernet(key)
        encrypted = fernet.encrypt(value.encode())
        return f"{cls._ENCRYPTION_PREFIX}{encrypted.decode()}"

    @classmethod
    def decrypt(cls, value: str) -> str:
        """Decrypt values produced by :meth:`encrypt`.

        Args:
            value: Ciphertext string, typically prefixed with ``encrypted:``.
        """
        if not value or not value.startswith(cls._ENCRYPTION_PREFIX):
            return value
        key = cls._get_encryption_key()
        if not key:
            return cls._undecryptable()
        try:
            encrypted_part = value[len(cls._ENCRYPTION_PREFIX) :]
            fernet = Fernet(key)
            decrypted = fernet.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except InvalidToken:
            logger.warning("Failed to decrypt value: invalid token or key mismatch")
            return cls._undecryptable()
        except (ValueError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to decrypt value: {type(e).__name__}: {e}")
            return cls._undecryptable()

    @classmethod
    def _undecryptable(cls) -> str:
        return ""

    @classmethod
    def _is_ciphertext(cls, value: str) -> bool:
        body = value[len(cls._ENCRYPTION_PREFIX) :]
        if not re.fullmatch(r"[A-Za-z0-9_-]+={0,2}", body):
            return False
        try:
            raw = base64.urlsafe_b64decode(body)
        except (binascii.Error, ValueError):
            return False
        return len(raw) >= _FERNET_MIN_BODY and raw[0] == _FERNET_VERSION_BYTE and (len(raw) - 57) % 16 == 0

    @classmethod
    def _looks_like_ciphertext(cls, value: str) -> bool:
        body = value[len(cls._ENCRYPTION_PREFIX) :]
        return bool(re.fullmatch(r"[A-Za-z0-9_-]+={0,2}", body)) and body.startswith(_FERNET_VERSION_HEAD)

    @classmethod
    def is_unreadable(cls, value: str) -> bool:
        if not value or not value.startswith(cls._ENCRYPTION_PREFIX):
            return False
        if not cls._looks_like_ciphertext(value):
            return False
        key = cls._get_encryption_key()
        if key is None:
            return True
        try:
            Fernet(key).decrypt(value[len(cls._ENCRYPTION_PREFIX) :].encode())
        except (InvalidToken, ValueError):
            return True
        return False

    @classmethod
    def read(cls, value: str) -> str | None:
        if not value:
            return None
        if not value.startswith(cls._ENCRYPTION_PREFIX):
            return value
        if cls._get_encryption_key() is None:
            return None
        plain = cls.decrypt(value)
        return plain or (None if cls._looks_like_ciphertext(value) else value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Expose a union schema so plain strings auto-wrap as EncryptedStr."""
        return core_schema.union_schema(
            [
                core_schema.is_instance_schema(cls),
                core_schema.chain_schema(
                    [
                        core_schema.str_schema(),
                        core_schema.no_info_plain_validator_function(
                            lambda value: cls(cls.encrypt(value) if value else value)
                        ),
                    ]
                ),
            ],
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: str(instance)
            ),
        )
_ALLOWED_LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def _default_api_key() -> EncryptedStr:
    """Return the API key env default as EncryptedStr."""
    return EncryptedStr((os.getenv("OPENROUTER_API_KEY") or "").strip())


def _default_artifact_encryption_key() -> EncryptedStr:
    """Provide an EncryptedStr placeholder for artifact encryption."""
    return EncryptedStr("")


def _resolve_log_level_default() -> Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    """The GLOBAL_LOG_LEVEL env var as a valve value, via the one resolver.

    Open WebUI gates the same variable on `logging.getLevelNamesMapping()`, which
    contains `WARN` and `FATAL`. A membership test against the five canonical names
    rejected both and fell back to INFO, so an operator who set `WARN` -- a spelling the
    host accepts -- got a WARNING floor at import and an INFO floor once the pipe read
    its own valve. Routing through `resolve_level` and naming the result maps the
    aliases onto their canonical spelling instead of discarding them.

    The membership test stays as the last step: a level registered by a third party
    through `addLevelName` resolves to a name outside the valve's Literal, and the UI
    dropdown only offers these five.

    Imported inside the function because `logging_system` reaches `core.utils`, which
    imports this module: at module scope the chain closes into a cycle. `default_factory`
    runs at `Valves()` instantiation, long after imports settle.
    """
    from .logging_system import resolve_level

    raw = (os.getenv("GLOBAL_LOG_LEVEL") or "INFO").strip().upper()
    value = logging.getLevelName(resolve_level(raw, logging.INFO))
    if value not in _ALLOWED_LOG_LEVELS:
        value = "INFO"
    return cast(Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], value)


def _detect_runtime_pipe_id(default: str = _DEFAULT_PIPE_ID) -> str:
    """Infer the Open WebUI function id from the module name.

    Some loaders (Open WebUI hot reload, runpy, unit tests) execute this module via exec/run
    without setting a real __name__. Returning the default keeps imports working in those cases.
    """
    module_name = globals().get("__name__", "")
    if isinstance(module_name, str) and module_name.startswith(_FUNCTION_MODULE_PREFIX):
        candidate = module_name[len(_FUNCTION_MODULE_PREFIX) :].strip()
        if candidate:
            return candidate
    return default


_PIPE_RUNTIME_ID = _detect_runtime_pipe_id()

def _is_template_valve(name: Any) -> bool:
    return isinstance(name, str) and name.endswith("_TEMPLATE")


_ADMIN_OFF_STAYS_OFF = (
    " A filter you switch off yourself in Open WebUI's Functions list stays off: the "
    "pipe keeps its code up to date but never switches it back on. The one exception is "
    "a write Open WebUI itself refused, which the pipe retries until it lands; once it "
    "lands the retry stops, so a filter you switch off after that stays off."
    " A row an earlier install of this pipe wrote — the pipe function was renamed or "
    "re-created, so its record names an id Open WebUI no longer loads as a pipe — is "
    "retired too."
)
_PIPE_OFF_COMES_BACK = (
    " One this version switched off itself comes back on its own when you enable the "
    "feature again."
)
_ROUTING_ADMIN_OFF_STAYS_OFF = (
    " A filter you switch off yourself in Open WebUI's Functions list stays off: the "
    "pipe keeps its code up to date but never switches it back on, or when the model "
    "is listed again. A write Open WebUI itself refused is the one thing the pipe does "
    "retry, and only until it lands."
    " A row an earlier install of this pipe wrote — the pipe function was renamed or "
    "re-created, so its record names an id Open WebUI no longer loads as a pipe — is "
    "retired too."
)
_ROUTING_ADMIN_RE_ENABLED_STAYS_ON = (
    " An entry you switch back on in the same list while its model is still listed "
    "stays on, and the pipe stops writing to it: no switching off, no re-stamping. "
    "Take the model out of the routing lists and the pipe retires the entry again."
)


class Valves(BaseModel):
    """Global valve configuration shared across sessions."""

    @model_validator(mode="before")
    @classmethod
    def _restore_blanked_templates(cls, values):
        if not isinstance(values, Mapping):
            return values
        restored: dict[str, Any] | None = None
        for name, value in values.items():
            if not _is_template_valve(name) or not isinstance(value, str) or value.strip():
                continue
            field = cls.model_fields.get(name)
            if field is None:
                continue
            default = field.get_default(call_default_factory=True)
            if not isinstance(default, str) or default == value:
                continue
            if restored is None:
                restored = dict(values)
            restored[name] = default
        return values if restored is None else restored

    @model_validator(mode="before")
    @classmethod
    def _floor_the_tool_loop_count(cls, values):
        if not isinstance(values, Mapping):
            return values
        count = values.get("MAX_FUNCTION_CALL_LOOPS")
        if isinstance(count, str):
            try:
                count = int(count.strip())
            except ValueError:
                return values
        if isinstance(count, float) and count.is_integer():
            count = int(count)
        if isinstance(count, int) and count < 1:
            return dict(values, MAX_FUNCTION_CALL_LOOPS=1)
        return values

    @model_validator(mode="before")
    @classmethod
    def _floor_the_bzip2_compression_level(cls, values):
        if not isinstance(values, Mapping):
            return values
        if values.get("SESSION_LOG_ZIP_COMPRESSION") != "bzip2":
            return values
        try:
            coerced = _BZIP2_COMPRESSLEVEL_ADAPTER.validate_python(
                values.get("SESSION_LOG_ZIP_COMPRESSLEVEL")
            )
        except ValidationError:
            return values
        if coerced != 0:
            return values
        _warn_bzip2_level_floored()
        return dict(values, SESSION_LOG_ZIP_COMPRESSLEVEL=1)

    @model_validator(mode="wrap")
    @classmethod
    def _drop_unvalidatable(cls, values, handler):
        return repair_unvalidatable(cls, values, handler)

    # Connection & Auth
    BASE_URL: str = Field(
        default=((os.getenv("OPENROUTER_API_BASE_URL") or "").strip() or "https://openrouter.ai/api/v1"),
        description=(
            "OpenRouter API base URL. Override this if you are using a gateway or proxy. "
            "Changing it makes the next image request re-read every image model's published "
            "settings, so the controls a gateway offers are never taken from the host they were "
            "read from before the change. A gateway on a private address is exempt from the "
            "address policy by origin -- scheme, hostname and effective port, compared as a "
            "parsed origin and never as a string prefix -- for the generated clip the pipe "
            "itself downloads from it, so a self-hosted gateway the pipe cannot reach is not "
            "refused by its own gate. Nothing else on that origin is exempt: a link a person "
            "attaches or pastes is still checked, so this does not make the gateway a fetch "
            "target a user can aim at. A cleartext `http://` gateway is still refused unless "
            "ALLOW_INSECURE_HTTP is set and the host is allowlisted, because the exemption is "
            "about addresses and not about transport. The exempted origin is the one place a "
            "download is not pinned to a validated address, so a DNS answer for that host can "
            "move mid-download."
        ),
    )
    DEFAULT_LLM_ENDPOINT: Literal["responses", "chat_completions"] = Field(
        default="responses",
        description=(
            "Which OpenRouter endpoint to use by default. "
            "`responses` uses /responses (best feature coverage; the whole prompt is cached as one unit). "
            "`chat_completions` uses /chat/completions (cache markers placed on individual message blocks; needed for Bedrock/Vertex-routed Claude caching and some provider features)."
        ),
    )
    FORCE_CHAT_COMPLETIONS_MODELS: str = Field(
        default="",
        description=(
            "Comma-separated glob patterns of model ids that must use /chat/completions "
            "(e.g. 'anthropic/*, openai/gpt-4.1-mini'). Matches both slash and dotted model ids. "
            "Globs are literal about the `~` prefix; add '~anthropic/*' to cover router aliases. "
            "A match on a Fusion model is not overridden: on the hosted OpenRouter backend the valve "
            "holds and the request is refused with the endpoint-conflict card, because Fusion "
            "renders only from /responses and on /chat/completions returns a flattened text "
            "transcript with no structured events. On the internal backend the Fusion panel "
            "runs in the pipe and the model never reaches OpenRouter, so there is no conflict to refuse "
            "and the turn runs — unless the request carries "
            "`{\"id\": \"fusion\", \"enabled\": false}`, or is a housekeeping task request (a chat "
            "title, summary or tags): with no panel to run, the model does reach OpenRouter and the pin "
            "refuses it with the same endpoint-conflict card."
        ),
    )
    FORCE_RESPONSES_MODELS: str = Field(
        default="",
        description=(
            "Comma-separated glob patterns of model ids that must use /responses "
            "(overrides FORCE_CHAT_COMPLETIONS_MODELS when both match). A model pinned "
            "here is not retried on /chat/completions when /responses fails; the error "
            "surfaces instead. Housekeeping task requests — chat titles, summaries, tags — "
            "are exempt and still fall back, so a pinned model never turns a title into an "
            "error string."
        ),
    )
    AUTO_FALLBACK_CHAT_COMPLETIONS: bool = Field(
        default=True,
        description=(
            "When True, retry the request against /chat/completions if /responses fails with an "
            "endpoint/model support error before any streaming output is produced. A failure OpenRouter "
            "reports inside a reply already under way is not retried against /chat/completions - the "
            "endpoint is not the problem once a reply has started. If it is a temporary failure (a 429 "
            "or 5xx) and nothing has been shown yet, it is re-sent on the same endpoint, governed by "
            "TRANSIENT_RETRY_MAX_ATTEMPTS; otherwise, and once those tries are spent, its message is "
            "shown straight away, appended below whatever answer was already streamed. The retry is "
            "skipped for a model pinned by FORCE_RESPONSES_MODELS to /responses, and its error "
            "surfaces; a model that merely sits on the responses default is still retried. Housekeeping "
            "task requests — chat titles, summaries, tags — are exempt and still fall back, so a pinned "
            "model never turns a title into an error string. It is also skipped for a Fusion turn whose "
            "panel the /chat/completions payload would lose - the plugin entry cannot travel there, so the "
            "retry would answer a different request as a plain completion with no panel - and that turn's "
            "error surfaces instead; a Fusion turn whose entry is disabled, and a Fusion entry on a "
            "non-Fusion model, still retry. A /responses failure the fallback "
            "repairs does not spend the user's failure budget, and the refund is immediate rather "
            "than at the end of the turn: the failed leg's own strike is taken back as soon as "
            "the fallback is decided, so a second request from the same user is not refused for "
            "a call this valve went on to answer. A /chat/completions failure of its own still counts."
        ),
    )
    API_KEY: EncryptedStr = Field(
        default_factory=_default_api_key,
        title="OpenRouter API key",
        description=(
            "Your OpenRouter API key. Defaults to the OPENROUTER_API_KEY environment variable. "
            "Clearing it removes the stored value and returns the setting to its default: the "
            "OPENROUTER_API_KEY environment value when one is set, and no key at all when one is "
            "not. Rotating it makes the next image request re-read every image model's published "
            "settings under the new account, so one account's published limits are never applied "
            "to another's. Re-saving an unchanged key does the same, because the stored value is "
            "re-encrypted with a fresh nonce on every save."
        ),
    )
    HTTP_REFERER_OVERRIDE: str = Field(
        default="",
        description=(
            "Override the `HTTP-Referer` header sent to OpenRouter for app attribution. "
            "Must be a full URL including scheme (e.g. https://example.com), not just a hostname. "
            "Applies to every request to an openrouter.ai host, including the catalogue, "
            "endpoint and maker-page refresh reads; not to user- or model-supplied asset "
            "downloads or the GitHub self-update check. "
            "Surrounding whitespace is trimmed, so a URL pasted with a trailing newline "
            "is used as typed. "
            "When empty, the pipe uses its default project URL. "
            "A value that is still not a full http(s) URL after trimming is ignored, a "
            "warning is shown, and the default is used."
        ),
    )
    HTTP_CONNECT_TIMEOUT_SECONDS: int = Field(
        default=10,
        ge=1,
        description="Seconds to wait for the TCP/TLS connection to OpenRouter before failing.",
    )
    HTTP_TOTAL_TIMEOUT_SECONDS: int | None = Field(
        default=None,
        ge=1,
        description="Overall HTTP timeout (seconds) for OpenRouter requests. Set to null to disable the total timeout so long-running streaming responses are not interrupted.",
    )
    HTTP_SOCK_READ_SECONDS: int = Field(
        default=300,
        ge=1,
        description="Idle read timeout (seconds) applied to active streams when HTTP_TOTAL_TIMEOUT_SECONDS is disabled. Generous default favors smoother User Interface behavior for slow providers. A stored value this release no longer accepts is left at this valve's default and named in the log.",
    )

    # Remote File/Image Download Settings
    REMOTE_DOWNLOAD_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts for downloading a picture from a link or a video a model generated. Set to 0 to disable retries.",
    )
    REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Initial delay in seconds before the first retry attempt. Subsequent retries use exponential backoff (delay * 2^attempt).",
    )
    REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS: int = Field(
        default=45,
        ge=5,
        le=300,
        description="Maximum total time in seconds to spend on retry attempts. Retries will stop if this time limit is exceeded. It also caps any single retry wait, including one a server asked for in a Retry-After header.",
    )

    TRANSIENT_RETRY_MAX_ATTEMPTS: int = Field(
        default=2,
        ge=0,
        le=10,
        description=(
            "How many extra tries a chat request to OpenRouter gets after a temporary failure "
            "(HTTP 429, HTTP 5xx, a 408 that names the provider-timeout kind, the same failure reported "
            "inside a reply, or a connection that drops or times out), on top of the first try. "
            "A body carrying a content decision is the exception and is never retried, whatever status it "
            "arrived on, because re-sending a prompt a guardrail declined cannot unblock it. "
            "So is a reply OpenRouter had already accepted: if that body dies on the way, the generation "
            "is billed whatever reached the pipe, and the fault is reported on that one request. "
            "2 means at most three requests in all. 0 makes a "
            "temporary failure final: the error is shown at once and nothing is retried. These are the "
            "chat request valves and are independent of the remote download valves above; one failed "
            "chat request is counted against the breaker once, however many tries it took."
        ),
    )
    TRANSIENT_RETRY_MAX_WAIT_SECONDS: int = Field(
        default=30,
        ge=1,
        le=300,
        description=(
            "The longest single wait, in seconds, between two tries of one chat request to OpenRouter. "
            "A Retry-After header OpenRouter sends is honoured up to this cap, so a longer Retry-After is "
            "truncated to it rather than replaced by it. Independent of the remote download valves above, "
            "and a wait here can be cancelled by stopping the request."
        ),
    )

    REMOTE_FILE_MAX_SIZE_MB: int = Field(
        default=_REMOTE_FILE_MAX_SIZE_DEFAULT_MB,
        ge=1,
        le=_REMOTE_FILE_MAX_SIZE_MAX_MB,
        description="Maximum size in MB for downloading a picture from a link: one in the conversation, or one a model returns for a picture it generated. A picture over this limit is not downloaded. A picture the pipe is already holding from an earlier turn is measured against this limit again on every send, exactly as one just downloaded is; one that no longer fits is given up and fetched once more, which this limit then refuses, and the refusal names the limit in force and the byte count it is over. The Open WebUI Max Upload Size counts the same way on that leg as on a fresh download. A file link a person attaches is not downloaded, so this limit does not apply to it; it is passed on to the provider instead, subject to the SSRF valve and the plaintext-HTTP policy, which can refuse it first. When Open WebUI RAG is enabled, the pipe also holds downloads to the cap the Open WebUI admin last saved under Admin > Settings > Documents > Max Upload Size. That setting is read from Open WebUI's own store, so changing it takes effect without restarting anything. Normally the smaller of the two wins, with one exception: this valve left at its 50 MB default gives way to a larger admin cap, clipped to the pipe's own 500 MB ceiling. Set it away from its default to keep full control in both directions; a valve off its default is never overridden. Clearing the admin's box lifts Open WebUI's cap, not the pipe's, so with this valve at its 50 MB default a cleared box still refuses anything over 50 MB. An environment variable set before Open WebUI first started is used only if that store cannot be read, so a RAG_FILE_MAX_SIZE value set after Open WebUI's first boot no longer has any effect: the stored admin value wins.",
    )
    BASE64_MAX_SIZE_MB: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum size in MB for inline files, images and audio. A base64 payload is measured as its decoded size; any other inline payload is measured as its own length. Larger payloads are dropped, to prevent memory issues and excessive HTTP request sizes: an uploaded payload is left out with a note, a file a tool returned that the pipe could not store is not shown, and a warning names the tool that returned it, and a picture inside one generated-image reply is dropped from that reply and named in the chat, while the pictures in that reply that did fit are still delivered. Every entry in such a reply is named for what it is, over that ceiling or not a picture, and raising this limit cannot make a non-picture entry fit: an entry that is not a picture does not count against the reply's ceiling at all, so a bad entry cannot displace a good one. It bounds a tool's file result too, which is then neither stored nor shown. A picture a tool returns is capped here on every path, as its own picture: it is left out of that round and named on the turn whose round carries it, while the round's text result and lead-in still go out. A picture a tool returns as an inline `data:` URL is capped where a file link a person attaches is not: the pipe forwards such a picture as it stands rather than downloading it, so the cap is what stands between it and the request.",
    )
    IMAGE_UPLOAD_CHUNK_BYTES: int = Field(
        default=1 * 1024 * 1024,
        ge=64 * 1024,
        le=8 * 1024 * 1024,
        description="Maximum number of bytes to buffer at a time when loading Open WebUI-hosted images before forwarding them to a provider. Lower values reduce peak memory usage when multiple users edit images concurrently. A stored value this release no longer accepts is left at this valve's default and named in the log.",
    )
    VIDEO_MAX_SIZE_MB: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum size in MB for inline (data:) video payloads and for stored videos re-read to extract frames. A base64 payload is measured as its decoded size; a token-free payload is measured as its own length, and is a real gate on it. An oversized video is not sent at all: the block is dropped from the turn and the refusal is reported; remote http(s) and YouTube video links are forwarded to the provider unmeasured.",
    )
    FALLBACK_STORAGE_EMAIL: str = Field(
        default=(os.getenv("OPENROUTER_STORAGE_USER_EMAIL") or "openrouter-pipe@system.local"),
        description="Owner email for the pictures a model generates in a request with no signed-in user (e.g., API automations).",
    )
    FALLBACK_STORAGE_NAME: str = Field(
        default=(os.getenv("OPENROUTER_STORAGE_USER_NAME") or "OpenRouter Pipe Storage"),
        description="Display name for the fallback storage owner.",
    )
    FALLBACK_STORAGE_ROLE: str = Field(
        default=(os.getenv("OPENROUTER_STORAGE_USER_ROLE") or "pending"),
        description="Role assigned to the fallback storage account when auto-created. Defaults to the low-privilege 'pending' role; override if your deployment needs a custom service role.",
    )
    ENABLE_SSRF_PROTECTION: bool = Field(
        default=True,
description="Enable SSRF (Server-Side Request Forgery) protection for remote URL downloads. When enabled, a remote address is fetched only if it is provably globally routable, so loopback, 10.x/172.16.x/192.168.x, link-local, carrier-grade NAT (100.64.0.0/10 -- also Tailscale's default range) and IPv6 site-local are all refused, as is any range the registries do not mark as globally routable. IPv6 addresses that wrap an IPv4 one (::ffff:, 6to4, Teredo, NAT64) are judged on the address they carry. A refused address is not sent either: the person sees `Images: skipped N (could not be fetched, so it was not sent).` A picture a tool result carries as a link is covered by that too: the address the provider would reach is checked before the link is forwarded, and a refused one is neither forwarded nor counted towards the turn's pictures. That check draws on the same request-wide `ADDRESS_CHECK_BUDGET_SECONDS` as the rest of this turn's, and a tool picture left with no time is not sent. A public `https://` link the pipe merely failed to download is still forwarded for the provider to fetch. The same gate covers a picture the MODEL wrote into a reply: a generated image that arrives as a `http(s)` address is checked before it is published, including when the pipe could not store it and the chat is temporary, local or a channel -- so a refused address is never written into the reply, and with this valve on, a check that reaches no verdict inside its budget is a refusal there too. It also gates every non-`data:` video link before it is passed on, whichever way it is written: the link is not downloaded, and the check is on the address the provider would reach, so a video link is refused when its host is not public and a link whose scheme is neither `http` nor `https` is refused outright. It walks a generation request's provider options too, including a value that is a JSON document containing an address, so an address written inside one is put to the same gate; a control too large or too deeply nested to certify is refused rather than sent, and the refusal names the path the address was found at. The one exemption is the admin's own `BASE_URL` origin: a generated clip the pipe itself downloads from it is not put to the address check, so a self-hosted gateway on a private address can deliver its own clips, and the comparison is by parsed origin (scheme, hostname, effective port) and never by string prefix. It covers that one download and nothing else: an attached `file_url`, a link in a tool result and a video link in a chat are all still checked, even on the gateway's own host, so it does not make that host a fetch target a user can aim at. The exemption is about addresses and not about transport, so a cleartext `http://` gateway still needs ALLOW_INSECURE_HTTP and an allowlisted host; and the exempted origin is the one download not pinned to a validated address, so a DNS answer for that host can move mid-download. Every link in `file_data` or `file_url` that the provider would have to fetch is gated the same way, in both fields, and it is not downloaded either. A link whose scheme the provider could not dial at all is refused on its scheme and never enters the request's address-verdict memo, because that memo holds verdicts about addresses and no address is consulted for such a link; the gate is still asked about it, once, so the refusal can say what it refused and not merely that something was. The two reasons stay distinct: a link on a private address and a link on a scheme the provider cannot fetch are not the same refusal and are not worded the same way.  a file link on a non-public host is refused, the person sees `Files: skipped N (...).`, and a block that also carries a `file_id` keeps that id and drops only the refused field. An inline `data:` URL, raw base64 in `file_data` and an Open WebUI file path are not links and are never checked. Those file checks draw on the same request-wide `ADDRESS_CHECK_BUDGET_SECONDS` as the pictures and the video links, and a repeated link is resolved once per request, so N attached links cost at most that budget in total rather than a lookup apiece. A deployment whose users attach documents by link to an internal host is refused with this valve on, exactly as pictures and videos are; set it to False to restore the old forwarding, which also restores the plaintext exposure this valve exists to close. A failed download costs one further address check, and the download's own check and that further one both draw on the same request-wide `ADDRESS_CHECK_BUDGET_SECONDS` as the video links, each check still held to at most one `ADDRESS_CHECK_SECONDS`; a picture left with no time is not sent, and a repeated link is checked once per request. A turn's remote video links draw on the same request-wide `ADDRESS_CHECK_BUDGET_SECONDS` as its pictures, so a message with many links is bounded by that budget rather than by one check per link, and a video link that is left with no time is not sent. That budget is spent by address checks alone: it is not charged to the time this turn's stored artefacts took to load. The address checks run on a dedicated bounded thread pool, so a stalled resolver is bounded there rather than queued behind everything else the process does; with this valve on, a check that cannot start inside its own budget reaches no verdict at all, and the person is told the check reached no verdict rather than that the address was refused: on a download or generation path that still sends no bytes, while a stalled re-check of a picture the pipe already holds no longer drops the stored copy, and a stalled check on a video link or a file link is refused with its own wording, naming the check that did not finish rather than a private network address; a stalled check on a picture link a tool result carries is refused the same way and with its own words again, saying the check could not be checked in time rather than that the link could not be fetched, because that link is forwarded rather than downloaded and it is the check that ran out of time, not a download. The pool's width follows `MAX_CONCURRENT_REQUESTS` and is re-made when that valve changes, in both directions. HTTP is disabled by default; see ALLOW_INSECURE_HTTP_* for explicit opt-in.",
    )
    ALLOW_INSECURE_HTTP: bool = Field(
        default=False,
        description="Allow plaintext HTTP remote URLs when explicitly enabled. HTTP is disabled by default; only enable with a narrow allowlist in ALLOW_INSECURE_HTTP_HOSTS. It covers a picture a tool result carries as a link in the same way as one the person attached. A refused picture is reported once, as an 'Images: skipped N (...)' status naming both valves, and a turn whose only content was that picture is sent as a placeholder naming the reason.",
    )
    ALLOW_INSECURE_HTTP_HOSTS: str = Field(
        default="",
        description=(
            "Comma-separated list of hosts or host:port entries allowed for plaintext HTTP. "
            "Exact match only (no wildcards). Empty means no HTTP allowed. "
            "Example: 'example.com, example.org:8080, 203.0.113.10'."
        ),
    )
    VIDEO_REFERENCE_ALLOWED_DOMAINS: str = Field(
        default="",
        description=(
            "Comma-separated host allowlist for the per-user reference links a video filter "
            "forwards to OpenRouter, on the filter's own reference fields and on free-text "
            "provider.options alike. An entry matches exactly or as a parent domain, "
            "case-insensitively, so example.com covers cdn.example.com. A bare host matches on "
            "any port; a host:port entry names that host on that port only, and a URL with no "
            "port is read as the port its own scheme implies. Empty - the default - "
            "means unrestricted. This is an additional restriction: the https:// and SSRF "
            "address check still applies, and runs either way. It takes no '!' block entries "
            "and no CIDR ranges, unlike the Plaintext HTTP host allowlist, which is "
            "exact-match rather than parent-domain. It does not govern a link the media relay "
            "published for this request: those are recorded as the pipe's own and go out "
            "whatever this list holds, and a host address typed by a user is not recorded and "
            "is not exempt."
        ),
    )
    ALLOW_UNKNOWN_SIZE_CLOUD_READS: bool = Field(
        default=False,
        description=(
            "Allow reading an Open WebUI file from a cloud or unrecognised storage provider when its "
            "recorded file size is missing or invalid. Default false to avoid an unbounded "
            "download. When true, the file is still copied to a private temporary file and capped by "
            "BASE64_MAX_SIZE_MB after download. Operator recovery path for legacy rows without size "
            "metadata; not an authorization bypass."
        ),
    )

    # Models
    MODEL_ID: str = Field(
        default="auto",
        title="Model allowlist",
        description=(
            "Comma separated OpenRouter model IDs to expose in Open WebUI. "
            "Set to 'auto' to import every available Responses-capable model. "
            "Each id may be written as OpenRouter's `author/model` form or as the exact id "
            "Open WebUI shows in the model picker, which begins with this pipe's function id "
            "followed by a dot. Both forms resolve to the same model. "
            "An entry holding `*`, `?` or `[` is a glob pattern rather than one id: it is "
            "matched case-insensitively against every catalog row, on both the `author/model` "
            "and the dotted spelling, so `deepseek/*` publishes that whole family. A `!`-prefixed "
            "entry excludes, and exclusions run after the includes, so "
            "`deepseek/*, !deepseek/*:free` keeps the paid rows and drops the `:free` one. A list "
            "made only of exclusions starts from the whole catalog, which `auto, !openai/*` also "
            "says; a `!`-only value used to publish nothing and now publishes the catalog minus "
            "what it removes. A pattern is matched as written, never split on `:` or `@`. "
            "Two glob effects are worth knowing: a `*` crosses a date stamp, so `openai/gpt-4o*` "
            "reaches `openai/gpt-4o-2024-08-06`; and globs are literal about the `~` prefix, so "
            "`openai/*` does not cover `~openai/...` and `!~openai/*` is what removes one. "
            "A restricting list that matches nothing publishes nothing and refuses every request; "
            "the one exception is a value of only commas or spaces, which is read as blank and "
            "imports the whole catalog. A pattern that matches no row fails closed the same way "
            "an unresolvable id does, and one `WARNING` names the entries it could not match. "
            "This list governs every model the pipe calls, including each Fusion panel member, "
            "the Fusion judge and the final answer, and including preset members: one this list "
            "excludes is a failed panel member carrying the reason, never silently substituted "
            "or dropped. A `~` pin is part of a model's identity for this list, so "
            "`~anthropic/claude-opus-latest` must be allowlisted with the tilde; written without "
            "it, the entry matches nothing in the catalog. "
            "An '@preset/slug' entry resolves to the model before the '@'."
        ),
    )
    MODEL_CATALOG_REFRESH_SECONDS: int = Field(
        default=60 * 60,
        ge=60,
        description=(
            "How long to cache the OpenRouter model catalog (in seconds) before refreshing. "
            "The refresh backoff after a failed fetch is tracked per OpenRouter account, so "
            "one account's outage never holds up another account's catalog read. The image "
            "catalog's model-list clock and its published-settings clock are stamped with the "
            "same account and keyed the same way, so a changed key refetches the image list and "
            "re-reads every model's settings at once rather than answering the new account from "
            "the old one's state. The video catalog's clock is stamped with the account that "
            "asked as well, so a changed key refetches the video model list for the same reason. "
            "The image models' published settings are cached on the same "
            "window, and that cache is dropped at once when the base URL or the API key changes, "
            "rather than being kept for the rest of the window. The 30-second window a failed read of one "
            "model's published settings opens is dropped on that same event, so the new "
            "base URL or key makes its first read rather than waiting out a pause that "
            "belongs to the credential before it. A request arriving during a media-catalog "
            "refresh is answered from the previous catalog rather than waiting for it, so "
            "a newly registered media model appears on the next turn; where the video catalogue has not "
            "been written yet, the request is told the video catalogue is still loading and that the next "
            "turn will carry the model, rather than that the pipe rejected it."
        ),
    )
    NEW_MODEL_ACCESS_CONTROL: Literal["public", "admins"] = Field(
        default="admins",
        description=(
            "Default access grants for new OpenRouter model entries added to Open WebUI. "
            "'public' grants read access to all users (wildcard access grant). "
            "'admins' creates no access grants (private) and relies on Open WebUI's "
            "BYPASS_ADMIN_ACCESS_CONTROL for admin access; otherwise admins must be granted access explicitly. "
            "Applies only when a model is first added; existing access grants are kept when it is refreshed."
        ),
    )
    FREE_MODEL_FILTER: Literal["all", "only", "exclude"] = Field(
        default="all",
        title="Free model visibility",
        description=(
            "Filter models based on OpenRouter pricing totals. "
            "'all' disables filtering. "
            "'only' restricts to models whose summed pricing fields equal 0. "
            "'exclude' hides those free models."
        ),
    )
    TOOL_CALLING_FILTER: Literal["all", "only", "exclude"] = Field(
        default="all",
        title="Tool-calling model filter",
        description=(
            "Filter models based on tool-calling capability (supported_parameters includes 'tools' or 'tool_choice'). "
            "'all' disables filtering. "
            "'only' restricts to tool-capable models. "
            "'exclude' hides tool-capable models."
        ),
    )
    ZDR_MODELS_ONLY: bool = Field(
        default=False,
        title="Show only ZDR models",
        description=(
            "When enabled, hide models that are not ZDR-capable (based on OpenRouter's /endpoints/zdr list). "
            "A hidden model is also refused if requested directly. It never sends provider.zdr=true -- "
            "use Enforce ZDR routing for that. Filtering is skipped only if the ZDR list has never been read; "
            "if a later read fails, the last list read successfully stays in force and filtering carries on from it. "
            "Video models are filtered like any other model. The list is per credential: the pipe re-reads whenever the credential that produced the list in force is not the credential in hand, so a different account sees no list rather than another's, and is refused while Enforce ZDR routing is on until its own read succeeds."
        ),
    )
    ZDR_ENFORCE: bool = Field(
        default=False,
        title="Enforce ZDR routing",
        description=(
            "When enabled, all requests include provider.zdr=true and will be rejected if the selected model "
            "does not have any ZDR endpoints. Requests are refused outright only if the ZDR list has never been read; "
            "if a later read fails, the last list read successfully stays in force, so an outage does not lock out a model "
            "that was answering a moment ago -- and every request still carries provider.zdr=true, which is what makes "
            "OpenRouter hold it to a no-retention endpoint. The list is per credential: the pipe re-reads whenever the credential that produced the list in force is not the credential in hand, so a different account sees no list rather than another's, and is refused while Enforce ZDR routing is on until its own read succeeds."
        ),
    )
    ALLOW_USER_ZDR_OVERRIDE: bool = Field(
        default=True,
        title="Allow user ZDR override",
        description=(
            "When enabled, users can toggle 'Request ZDR' per chat. "
            "If Enforce ZDR routing is enabled, user overrides are ignored. The list is per credential: the pipe re-reads whenever the credential that produced the list in force is not the credential in hand, so a different account sees no list rather than another's, and is refused while Enforce ZDR routing is on until its own read succeeds."
        ),
    )
    VARIANT_MODELS: str = Field(
        default="",
        title="Variant models",
        description=(
            "Comma-separated list of variant model entries in format 'base_id:variant_tag'. "
            "Example: 'openai/gpt-4o:exacto,anthropic/claude-sonnet-4.5:extended'. "
            "Each entry creates a virtual model that inherits the base model's metadata "
            "(description, icon, capabilities) with the variant tag appended to the display name. "
            "A routing variant also gets the base model's reasoning settings, "
            "unless the catalog lists the suffixed id itself; "
            "a `base_id@preset/slug` entry gets none of the pipe's own, so the preset's saved ones apply, "
            "though a reasoning effort the chat itself sets still goes out and overrides them for that request. "
            "Supported tags: free, thinking, online, nitro, exacto, extended. "
            "A variant of an image model is the same model, so the image settings panel "
            "covers the variant too. "
            "A `*` or `!` entry is `MODEL_ID` selection syntax, not a variant spec: this valve "
            "takes exact base ids, and an id `MODEL_ID` excludes is not added as a variant either. "
            "The full model ID with ':variant' suffix is sent to OpenRouter for specialized routing."
        ),
    )

    THINKING_OUTPUT_MODE: Literal["open_webui", "status", "both"] = Field(
        default="open_webui",
        title="Thinking output",
        description=(
            "Controls where in-progress thinking is surfaced while a response is being generated. "
            "'open_webui' streams reasoning in the Open WebUI reasoning box and nowhere else; "
            "'status' shows thinking only as status messages; "
            "'both' enables both outputs. "
            "The pipe's own short progress lines above a first reply - 'Thinking…' and the few that follow it - "
            "are sent at every setting of this valve, in every mode: they are the pipe saying the model has "
            "started, not a report of its thinking, and this valve does not move them."
        ),
    )
    ENABLE_ANTHROPIC_INTERLEAVED_THINKING: bool = Field(
        default=True,
        title="Anthropic interleaved thinking",
        description=(
            "When True, enables Claude's interleaved thinking mode by sending "
            "`interleaved-thinking-2025-05-14` in the `x-anthropic-beta` header "
            "on Claude models that support it "
            "(including the `~anthropic/...` router aliases)."
        ),
    )
    ENABLE_ANTHROPIC_PROMPT_CACHING: bool = Field(
        default=True,
        title="Anthropic prompt caching",
        description=(
            "When True and the selected model is `anthropic/...` (including `~anthropic/...` router aliases), "
            "enable Claude prompt caching to reduce "
            "per-turn costs for large stable prefixes (system prompts, tools, RAG context). On /responses a "
            "the whole prompt is cached as one unit (routes Anthropic-direct, excluding Bedrock/Vertex); on "
            "/chat/completions cache markers placed on individual message blocks are used (works across all providers)."
        ),
    )
    ANTHROPIC_PROMPT_CACHE_TTL: Literal["5m", "1h"] = Field(
        default="5m",
        title="Anthropic prompt cache TTL",
        description=(
            "TTL for Claude prompt caching breakpoints (ephemeral cache). Default is '5m'. "
            "Note: Longer TTLs can increase cache write costs."
        ),
    )
    SEND_CACHE_SESSION_ID: bool = Field(
        default=True,
        title="Prompt-cache session affinity",
        description=(
            "When True (default), send a stable per-conversation `session_id` = "
            "HMAC-SHA256(`WEBUI_SECRET_KEY`, or the deprecated `WEBUI_JWT_SECRET_KEY` it falls back to (a default, so an empty primary is not a fallback), chat_id) so OpenRouter keeps each conversation on one "
            "provider and maximizes prompt-cache hits across turns. Opaque; no raw identifiers. "
            "Skipped if neither of those is set. A temporary chat is pinned like any "
            "other chat: the value is a keyed hash, so no identifier is exposed. The pin is "
            "computed from the chat id, and a Fusion panel-member call is pinned like any "
            "other turn of the same conversation: it follows the outer turn's identity "
            "rules exactly, so a saved or `channel:` chat is pinned and a temporary chat "
            "is pinned by hash.\n\n"
            "A call that carries no chat id falls back to a hash of the caller's own "
            "`session_id` under an `api-session:` prefix, so the raw value never leaves the "
            "pipe. The API caller is the only party that knows what \"the same conversation\" "
            "means for its own traffic, so the pin is applied only when that caller actually "
            "sends a `session_id`. A call that sends `parent_id: null` is pinned to a "
            "conversation id Open WebUI mints fresh for each turn, so its cache never warms; "
            "the pipe does not override a real `chat_id`. The `session_id` fallback applies "
            "only to a body that carries no `chat_id` at all - send `session_id` and omit "
            "`chat_id` to get one."
        ),
    )
    ENABLE_PLUGIN_SYSTEM: bool = Field(
        default=False,
        title="Enable plugin system",
        description=(
            "Master switch for the plugin system. When False, plugins are never called at all. "
            "Takes hold on each worker's next request or model-list build; the dashboard's own "
            "route reads the persisted row and closes at once."
        ),
    )
    AUTO_CONTEXT_TRIMMING: bool = Field(
        default=True,
        title="Auto-trim overlong prompts",
        description=(
            "When enabled, automatically enables OpenRouter's `context-compression` plugin so long prompts "
            "are trimmed from the middle instead of failing with context errors. Disable if your deployment "
            "manages context compression manually."
        ),
    )
    REASONING_EFFORT: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = Field(
        default="medium",
        title="Reasoning effort",
        description=(
            "Default reasoning effort to request from supported models. 'none' switches reasoning off where the "
            "model allows it; a model that always reasons gets the lightest level its catalog entry lists other than "
            "`none` instead, and no level at all when it lists no other level. On a model that takes only the legacy "
            "`include_reasoning` there is no effort to substitute, so the pipe writes `true` instead of the off. "
            "On such a model the pipe keeps asking regardless, and in a chat it says so in a status line naming the "
            "model; a background task shows nothing. "
            "A request that only hides the reasoning trace (`reasoning.exclude` of `true`) is not one of these offs: "
            "it keeps the depth this setting chose and draws no status line. "
            "A request that carries its own reasoning.max_tokens overrides this default, except while this setting "
            "is 'none': there a per-chat thinking budget is ignored, because 'none' switches reasoning off on every "
            "path. A request that carries its own verbosity keeps it on both endpoints, whichever of the two "
            "spellings it used, and this setting never overrides it. Use 'xhigh' when maximum depth is desired "
            "(only on supporting models)."
        ),
    )
    REASONING_SUMMARY_MODE: Literal["auto", "concise", "detailed", "disabled"] = Field(
        default="auto",
        title="Reasoning summary",
        description="Controls the reasoning summary emitted by supported models (auto/concise/detailed). Set to 'disabled' to skip requesting reasoning summaries.",
    )
    GEMINI_THINKING_BUDGET: int = Field(
        default=1024,
        ge=0,
        le=65536,
        title="Gemini 2.5 thinking budget",
        description=(
            "Base thinking budget (tokens) for Gemini 2.5 models, sent as OpenRouter's reasoning.max_tokens and "
            "scaled by reasoning effort (minimal -> smaller, xhigh -> larger). When 0, thinking is switched off, "
            "except on Gemini 2.5 Pro, which cannot stop thinking; on such a model the pipe keeps asking regardless, "
            "and in a chat it says so in a status line naming the model (a background task shows nothing). "
            "A request that carries its own reasoning.max_tokens overrides this budget, except while reasoning is "
            "switched off; there a per-chat thinking budget is ignored. A request that carries its own output limit has the budget reduced to fit inside it, the "
            "limit itself is never changed, and when the limit leaves no room the pipe asks for no bounded budget and "
            "the model decides. The same reserve applies to a request's own reasoning.max_tokens on any "
            "reasoning-capable model, not only the Gemini 2.5 family."
        ),
    )
    PERSIST_REASONING_TOKENS: Literal["disabled", "next_reply", "conversation"] = Field(
        default="conversation",
        title="Reasoning retention",
        description="Reasoning retention: 'disabled' keeps nothing, and covers the reasoning already stored as well as the new - the first turn after the switch stops replaying the rows an earlier one stored and deletes them, it does not put them on the wire again, and it does not save each reply's reasoning_details onto the stored chat message - and the round's own tool markers still go out, so a tool round is sent without the thinking block that preceded it. 'next_reply' keeps thoughts only until the following assistant reply finishes, when that reply happens in this chat; rows whose answering request never arrives are dropped by the periodic cleanup, and 'conversation' keeps them for the full chat history. The copy of the reasoning written onto the Open WebUI assistant message follows the same setting, and at 'next_reply' that copy is not removed when the following reply finishes; only the rows are. Reasoning is kept when the provider sends it as a replayable output item; reasoning that arrives only as streamed deltas is shown in the thinking box but not replayed on later turns. A temporary chat stores no reasoning; in Open-WebUI tool mode the thinking of a streamed reply is held in memory for that reply only, and dropped when the pipe answers its last call back, or when the provider refuses a call-back the pipe was waiting for, or when the reply is stopped, or after 15 minutes unused. A user setting the pipe cannot read (an undecodable stored row, after a rotation of `WEBUI_SECRET_KEY`, or the deprecated `WEBUI_JWT_SECRET_KEY` it falls back to (a default, so an empty primary is not a fallback)) falls back to that field's own per-user default, whichever side of this site-wide value that default sits on, and never to the value set here. A call that carries no chat_id has its reasoning and tool records held in memory for the length of that request only and never written to the database (see API_CALL_ARTIFACT_MEMORY), so an API call's records last for the request, not the conversation.",
    )
    TASK_MODEL_REASONING_EFFORT: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = Field(
        default="low",
        title="Task reasoning effort",
        description=(
            "Reasoning effort requested for Open WebUI background tasks (titles, tags, etc.) when they target this pipe's models. "
            "Low is the default balance between speed and quality; set to 'minimal' to prioritize fastest runs, "
            "or use medium/high for progressively deeper background reasoning at higher cost. "
            "Use 'none' to switch reasoning off for those tasks, where the model allows it; a model that always "
            "reasons gets the lightest level its catalog entry lists other than 'none' instead, or, on a model that "
            "takes only the legacy `include_reasoning`, `include_reasoning: true` since there is no effort to "
            "substitute. A task request "
            "that only hides the reasoning trace (`reasoning.exclude` of `true`) is not one of these offs: it "
            "keeps the depth this setting chose and draws no status line."
        ),
    )

    # Tool execution behavior
    TOOL_EXECUTION_MODE: Literal["Pipeline", "Open-WebUI"] = Field(
        default="Pipeline",
        title="Tool execution mode",
        description=(
            "Where to execute tools. 'Pipeline' executes tool calls inside this pipe "
            "(with its own batching, failure limits, and special tool handling). 'Open-WebUI' hands a streamed reply's "
            "tool calls back rather than running them here, so Open WebUI executes them and renders the native tool UI, "
            "until the reply's hand-back budget of `MAX_FUNCTION_CALL_LOOPS` turns is spent; after that the pipe runs the "
            "remaining round itself with the tools it advertised. A round of browser-run tools and Open WebUI's own "
            "builtins is handed back whatever that budget says, because Open WebUI is what runs those, and its own "
            "iteration bound ends such a reply. That budget is one user's spend against one reply - two users naming "
            "the same chat and message are charged to two budgets and neither spends the other's - and it is charged "
            "in a Temporary Chat too, where only the reply's own budget is held and its chat id is not. "
            "The pipe runs a non-streamed reply's calls and a Fusion panel model's calls in either mode. A tool the "
            "request itself declared with nothing behind it goes back to its sender instead. With 'ask' tool approval, "
            "a streamed saved chat hands every call to Open WebUI in both modes. With legacy function calling, no "
            "Open WebUI tool is offered. A model the catalogue rules out for tool use is sent none of the tools Open WebUI or this pipe added; a tool the request itself declared is the caller's and goes out as sent, with its `tool_choice` beside it."
        ),
    )
    SHOW_TOOL_CARDS: bool = Field(
        default=True,
        title="Show tool execution cards",
        description="Show each tool the model uses as a collapsible card in the chat, with its name, arguments and result, as Open WebUI does for the tools it runs itself. As in Open WebUI's own tool loop, a picture a tool returns as image data goes only to the model; a picture Open WebUI has stored as a file, such as an MCP tool's, goes to the model and the chat, as Open WebUI does since its fix after 0.11.4; other files go only to the chat. When off, the tools this pipe runs and OpenRouter's server tools get no card, except that a file the model shows through Open Terminal keeps its card for a person whose Open WebUI shows terminal files inline. On its next turn the model still learns which tools it used, except in a temporary chat, for which the pipe keeps nothing; after Stop, it learns of the calls before the first one still running if the reply was streamed, and of none if it was not. Open WebUI draws its own cards for the calls it runs, which now means a streamed reply in Open-WebUI mode and calls approved under 'ask'. With tool cards on, a call the loop cut off at `MAX_FUNCTION_CALL_LOOPS` is shown as a card marked failed, so a round that never ran is visible rather than missing.",
    )
    PERSIST_TOOL_RESULTS: bool = Field(
        default=False,
        title="Keep tool results",
        description="Give the model the full arguments and results of tool calls from earlier turns. When disabled, the model sees each tool call from an earlier turn, and each round that arrived before the chat's first turn -- an API caller, an imported or reordered chat, or a filter posts one -- as its name and a short note on whether it succeeded (the built-in ask_user question and the person's answer always go back), and relies on its own earlier answers or runs the tool again. Each round is judged by its own call, so two rounds that happen to share one call id are kept or withheld separately, and one round's exemption never carries to another behind it. The turn the request ends in is never withheld: a round is replaced only when its turn is earlier than the final one, so a round whose assistant message opens the chat is kept while no later turn follows it. The setting applies in both tool execution modes and decides what the model is handed, not whether results are stored: a shown tool card keeps the full result in the message, and the pipe's own copy of each tool round keeps the full call and result, pictures included, encrypted only while ARTIFACT_ENCRYPTION_KEY is set and ENCRYPT_ALL is on. A temporary chat stores none of its tool rounds or thinking; in Open-WebUI tool mode the rounds and thinking of a streamed reply are held in memory for that reply only, and dropped when the pipe answers its last call back, or when the provider refuses a call-back the pipe was waiting for, or when the reply is stopped, or after 15 minutes unused. A user setting the pipe cannot read (an undecodable stored row, after a rotation of `WEBUI_SECRET_KEY`, or the deprecated `WEBUI_JWT_SECRET_KEY` it falls back to (a default, so an empty primary is not a fallback)) falls back to that field's own per-user default, whichever side of this site-wide value that default sits on, and never to the value set here. A call that carries no chat_id has its tool records held in memory for the length of that request only and never written to the database (see API_CALL_ARTIFACT_MEMORY), so an API call's records last for the request, not the conversation.",
    )
    API_CALL_ARTIFACT_MEMORY: bool = Field(
        default=True,
        title="Keep API-call artifacts in memory",
        description=(
            "When True (default), the reasoning and tool records of a call that carries no `chat_id` are held in "
            "memory for the length of that request only, keyed on the request id, and are never written to the "
            "database. Off, it does nothing for existing callers, because such a call writes nothing "
            "either way; the difference is that on, the records exist for the request and are dropped when it ends. "
            "The same limits apply as for a temporary chat's held reply - 15 minutes idle and 64 MiB per pool, "
            "fixed rather than set here - and this is a pool of its own, so the total memory ceiling is twice one pool. "
            "A temporary chat never opens this bucket, a Fusion inner call never does either, and a call that sends "
            "`parent_id: null` is given a real chat id by Open WebUI, so that shape is not covered here. "
            "Its records have no message id to be stored under, so none of them is persisted either, and the skip is "
            "announced once per request - at warning level for the first record and at debug level for the rest - "
            "rather than once per record. "
            "No marker line is ever added to the caller's response, so a program's bytes in and bytes out are "
            "unchanged."
        ),
    )
    ARTIFACT_ENCRYPTION_KEY: EncryptedStr = Field(
        default_factory=_default_artifact_encryption_key,
        description="Use at least 16 chars. Encrypt reasoning tokens (and optionally all persisted artifacts). Changing the key creates a new table; prior artifacts become inaccessible. Clearing it stops artifact encryption and returns the setting to its default, which is empty. Both the artifact table and the usage-history table are named from a hash of this key, so new writes after a clear go to a fresh, unencrypted pair of tables and everything already saved under the previous key is stranded there, unread. A value that cannot be read under the current application secret (`WEBUI_SECRET_KEY`, or the deprecated `WEBUI_JWT_SECRET_KEY` it falls back to (a default, so an empty primary is not a fallback)) cannot be used either, whether it was stored under a key that no longer decrypts or is a damaged row that still looks like a Fernet token: the pipe refuses to write artifacts while the key is unreadable rather than storing them in the clear, and the person in the chat is told once, on the turn it happens, that the items will be missing from later turns; the key must be re-entered here before writes resume. A row damaged out of that shape is not read as a ciphertext, so it does not arm this refusal — which is also why an install that leaves Open WebUI's ENABLE_VALVE_ENCRYPTION at its default, and so stores this valve as plain JSON, is not stopped by it. A passphrase typed here that begins with encrypted: and continues with an all-base64 character body is read as a damaged stored value and refused the same way, so enter it without the prefix. The cipher is rebuilt against the current key on every call, so a rotation never leaves the store using a retired one; a write already inside that cipher build when the change lands is still written under the previous key, cannot be read afterwards, and is dropped with a warning naming its artifact kind.",
    )
    ENCRYPT_ALL: bool = Field(
        default=True,
        description="Encrypt every persisted artifact when ARTIFACT_ENCRYPTION_KEY is set. When False, only reasoning tokens are encrypted. This decides what is written; a row already stored encrypted stays encrypted, and the replay cache keeps it encrypted, whatever this is set to. If ARTIFACT_ENCRYPTION_KEY is set but cannot be decrypted after a rotation of `WEBUI_SECRET_KEY`, or the deprecated `WEBUI_JWT_SECRET_KEY` it falls back to (a default, so an empty primary is not a fallback), the pipe stops writing artifacts rather than storing them in the clear; that refusal is armed by a stored row that still looks like a Fernet token and will not open, so a plain-JSON valve row, which is what an install with Open WebUI's ENABLE_VALVE_ENCRYPTION at its default has, never arms it, and the person in the chat is told once, on the turn it happens, that the items will be missing from later turns.",
    )
    ENABLE_LZ4_COMPRESSION: bool = Field(
        default=True,
        description="When True (and lz4 is available), compress large encrypted artifacts to reduce database read/write overhead. A compression attempt that raises disables compression for the rest of the process: the failure is logged once, no later payload is sent to that codec, and toggling this valve off and on again does not re-arm it. A restart is what re-enables it.",
    )
    MIN_COMPRESS_BYTES: int = Field(
        default=0,
        ge=0,
        description="Payloads at or above this size (in bytes) are candidates for LZ4 compression before encryption. The default 0 always attempts compression; raise the value to skip tiny payloads.",
    )

    ENABLE_STRICT_TOOL_CALLING: bool = Field(
        default=True,
        description=(
            "When True, converts Open WebUI registry tools to strict JSON Schema for OpenAI tools, "
            "enforcing explicit types, required fields, and disallowing additionalProperties. Only the "
            "registry tools this pipe runs are made strict; a schema that will be handed back keeps the caller's own "
            "fields, and a tool that arrived in Chat Completions shape states its strictness explicitly - `false` when "
            "the caller wrote none, that endpoint's own default - while a tool that arrived in Responses shape is "
            "passed through with exactly the keys it arrived with. "
            "A tool whose root schema is a string or an array is not supported: strict mode requires an object "
            "root, so the pipe wraps such a schema in a single `value` property and the model sends its argument "
            "under that name. "
            "Any root that is not an object is wrapped this way - a number, a boolean, an `enum`/`const`, a "
            "`oneOf`/`anyOf`, or a schema that declares no `properties`. A root carried by `$ref` or `allOf` is "
            "resolved and its properties advertised, as far as the resolver's depth and budget reach. "
            "Every property becomes required, and an optional one instead accepts null - a `null` beside the type it "
            "already had, or, for a node that states no type of its own such as a bare `$ref` or an `anyOf`/`oneOf`, a "
            "`null` branch added to it with the node itself kept as the first branch. A missing field type is filled in "
            "too: a property, an array's `items` and an `anyOf`/`oneOf` branch whose values are pinned by `enum` or "
            "`const` is given the type those values have - `string`, `integer`, `number`, `boolean` or `null` - and one "
            "that pins no values is given `object`, or `array` where it declares `items`. A node that declares "
            "`anyOf`, `oneOf` or `allOf` states its own type, so none of those three places gives it one: the "
            "composition is left as it was written rather than constrained by a type beside it that no value would "
            "satisfy. "
            "Annotations and definitions a node carries are kept, not only its resolved properties: a single-`$ref` "
            "`allOf` unwrap leaves the node's own `title` and `description` in place, and a definition a nested node "
            "carries is reachable from the document root and strictified in place. A name the root already uses is the "
            "root's, and the nested body is renamed on the way up, with its pointer following it to the fresh name. "
            "Tools are also sent with `strict: true` on `/chat/completions`, nested under each `function`; "
            "a provider that does not support strict tool calling there will reject the request. A caller's own "
            "Chat-shaped tool goes out on that leg with `strict: false` under the same `function` when the caller "
            "wrote no `strict` - the value that endpoint already defaults to, stated rather than left to it. "
            "On the Responses route the registry and direct-tool specs the pipe advertises also carry "
            "`strict: true`, so the provider enforces the strictified schema. A tool the pipe does not "
            "strictify (Open WebUI tool mode, a hand-back, or this valve off) is sent there with an "
            "explicit `strict: false`, so that endpoint's own `true` default never applies to a schema "
            "the pipe did not strictify. The strictified schema is not "
            "renamed to meet strict mode's property-name rules, so a tool whose author gave a property a name "
            "with a dot, a space, or more than 64 characters, or a free-form array whose `items` declare no "
            "properties, can be rejected by a strict provider; turn this valve off for such a tool. "
            "When False, the schema is forwarded untouched and the executor still filters a call's arguments "
            "against the names that schema declares, with a root carried by `$ref` or `allOf` resolved for that "
            "purpose only; a root that is not an object and is not advertised under the pipe's `value` envelope "
            "declares no names, so nothing is delivered for it."
        ),
    )
    MAX_FUNCTION_CALL_LOOPS: int = Field(
        default=25,
        description=(
            "Maximum number of full execution cycles (loops) allowed per request whenever "
            "this pipe runs the calls. Each loop involves the model generating "
            "one or more function/tool calls, executing all requested functions, and feeding "
            "the results back into the model. The count is of rounds in which tools actually "
            "run, not of model requests: a request that only carries the skipped-call stub and "
            "the one after it that writes the answer are both outside it. When the limit is "
            "reached, pending tool calls are returned to the model marked as skipped so it can "
            "write a final answer, and, with tool cards on, each skipped call is shown in the "
            "transcript as a failed call card rather than being dropped silently. "
            "The model always gets at least one generation turn, so 0 and below are stored as 1. "
            "A reply Open WebUI re-asks carries a hand-back budget of this many turns, and once it is spent the pipe "
            "runs the remaining round itself with the tools it advertised. "
            "Has no effect on the calls Open WebUI runs, where the round limit is managed by Open WebUI: a round of "
            "browser-run tools and Open WebUI's own builtins is handed back whatever this budget says, and Open WebUI's "
            "own iteration bound is what ends that reply."
        )
    )

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default_factory=_resolve_log_level_default,
        description="Select logging level.  Recommend INFO or WARNING for production use. DEBUG is useful for development and debugging.",
    )
    SESSION_LOG_STORE_ENABLED: bool = Field(
        default=False,
        description=(
            "When True, save the full log of each request to encrypted zip files on disk. "
            "Archives capture the full OpenRouter request/response (prompts, model output, tool calls, provider errors) plus request identifiers — treat as sensitive conversation data at rest. "
            "One zip is written per message turn, plus one for each housekeeping task that resolves to a message id on the route that dispatched it, named <message_id>.<task>.zip — a message id too long for that key keeps its stem and gains a short digest of the exact id rather than being cut, so two ids cannot share one file, and an archive written for such an id by an earlier release is renamed. Open WebUI defines nine task types in its TASKS enum plus three more named inline (context_compaction, memory_review, context_summary), and the pipe dispatches a thirteenth of its own, video_intent_v1, so how many of those one turn writes depends on which of them the route that dispatched them carried an id for, and there is no fixed ceiling. context_summary is one of the qualifiers that does not resolve - Open WebUI dispatches it from its realtime voice session, which carries no message id - so it contributes no task archive of its own, and a task invocation that resolves to no message id is skipped whether or not Archive API calls is on. An internal-Fusion turn writes its inner calls beside those, each in the request-keyed api/ tree rather than beside the turn's own archives. "
            "Persistence needs a user_id and a request_id; with it on, a call that carries no usable chat_id or message_id is archived under "
            "`api/api-<request_id>.zip` with no session_id recorded (see SESSION_LOG_ARCHIVE_API_CALLS), and every temporary chat is still dropped; that drop is logged as a warning on each of the two archive paths (segment persist, bundle assembly) and again after a five-minute cooldown, once per person on the one path that runs and once per worker process on bundle assembly, which is called without a user. Only the segment-persist path runs for a request today, so that is the one that warns. "
            "A Fusion panel member is the other shape that carries no message id: it has none of its own, but `run_fusion_member` restores the outer turn's chat_id onto it, so its traffic takes the request surrogate and is written as `api/api-<request_id>.zip` while Archive API calls is on, and skipped when it is off - one file per inner call, so an N-model panel turn writes N+2 of them (the members, the judge and the synthesis), or N+3 with the judge's second pass, on top of the turn's own archives. "
            "A turn whose chat id names a saved chat the caller does not own is not staged either, and is refused by the same rule Open WebUI applies to a chat message (admins excepted); the assembler separately refuses any bundle whose staged segments do not all name the same user, keeping every segment for retry and writing no archive under either name. "
            "Turning this off also stops the retention sweep, leaving every archive already on disk untouched until it is re-enabled and the retention window passes. "
            "A write already inside an assembly pass is read again at the write itself, so one that is already under way when this is switched off publishes nothing and its staged segments stay in the database for a later pass."
        ),
    )
    SESSION_LOG_ARCHIVE_API_CALLS: bool = Field(
        default=True,
        description=(
            "When True (default), a call that arrives with no usable chat id or message id - the plain API route, "
            "where Open WebUI supplies neither - is archived like any other turn, under "
            "`<SESSION_LOG_DIR>/<user_id>/api/api-<request_id>.zip`. The key is the request id, which is unique per "
            "request, so two API calls never share one archive file. An archive written this way records no session_id at "
            "all, at any of its three writers; a caller that wants its session id correlated should give the call a chat_id, "
            "or accept the request-keyed archive as it is. Requires SESSION_LOG_STORE_ENABLED. "
            "This is a staging gate: it decides whether the call's segment is written to the database at all. A "
            "segment already staged while this was on is still packed and written by the assembler, so a handful of "
            "archives can appear after you switch it off. A `parent_id: null` body is given a real chat id but no "
            "message id, so it takes this path too and is keyed on the request id, not on that chat. "
            "This valve governs that plain API route and the `parent_id: null` shape only: a task invocation that "
            "resolves to no message id is skipped whether it is on or off. "
            "This valve is never read again after staging: a row already written is finished under it, exactly as a "
            "terminal segment that lands after a turn has ended. The master SESSION_LOG_STORE_ENABLED is different - it is read "
            "at the write itself, so an assembly pass already under way when it is switched off publishes nothing and keeps its rows."
        ),
    )
    SESSION_LOG_DIR: str = Field(
        default="session_logs",
        description=(
            "Base directory for encrypted session log archives. "
            "Files are stored under <SESSION_LOG_DIR>/<user_id>/<chat_id>/<message_id>.zip, "
            "with a housekeeping task's own archive beside the answer's as <message_id>.<task>.zip. "
            "<user_id> is the user every event in that file belongs to: a turn whose chat id names a "
            "saved chat the caller does not own is not staged, and a bundle whose staged segments disagree "
            "about the user is never written under either name. "
            "Surrounding whitespace is ignored, and a value that is blank once trimmed counts as unset. "
            "A path component holding a character outside [0-9A-Za-z._-], or long enough to be cut, keeps its sanitized stem "
            "and gains a short digest of the exact id, so two ids that would otherwise land on the same name cannot; the composed "
            "<message_id>.<task> key obeys the same rule one stage earlier, so a message id too long for that key keeps its stem and gains the "
            "same short digest rather than being cut, and two such ids cannot share a task archive. The exact ids stay in "
            "the archive's meta.json under ids, and a turn with no usable user id gets a directory of its own rather than a shared user/."
        ),
    )
    SESSION_LOG_ZIP_PASSWORD: EncryptedStr = Field(
        default=EncryptedStr(""),
        description=(
            "Password used to encrypt session log zip files (AES-encrypted zip). "
            "Recommend using a long random passphrase and encrypting the value (requires "
            "`WEBUI_SECRET_KEY`, or the deprecated `WEBUI_JWT_SECRET_KEY` it falls back to (a default, so an empty primary is not a fallback)). "
            "Clearing it stops all archive writing, and so does a stored value that cannot be read under "
            "the current application secret, whether it was stored under a key that no longer decrypts or is a "
            "damaged row: no new archive is written until a passphrase is re-entered here. A segment staged while "
            "the passphrase did resolve is still packed once it resolves again, so a few archives can appear after "
            "you clear it. A passphrase typed here "
            "that begins with encrypted: and continues with an all-base64 character body is read as a damaged "
            "stored value and refused the same way, so enter it without the prefix."
        ),
    )
    SESSION_LOG_RETENTION_DAYS: int = Field(
        default=90,
        ge=1,
        description=(
            "Retention window for stored session log archives and for their staging rows. Cleanup deletes zip files older than this many days, and reaps the staged session-log segment rows in the artifact table that are older than this many days, measured from when they were written."
            " The sweep reads this valve on every pass, so a changed window applies from the next cleanup"
            " with no restart and no new turn."
        ),
    )
    SESSION_LOG_CLEANUP_INTERVAL_SECONDS: int = Field(
        default=3600,
        ge=60,
        description=(
            "How often (in seconds) to run the session log cleanup loop when storage is enabled. "
            "A pass never prunes a directory a write is in the middle of, in this process or a peer "
            "worker's, so an archive is never lost to the sweep: the write publishes a reservation "
            "file inside the directory it is filling, and a directory holding anything at all is left "
            "alone. "
            "A write that fails leaves its directory to a later pass, while a process that dies mid-write leaves "
            "a temporary file and its reservation that no pass reaps. A warning about a lost archive is itself "
            "filtered by LOG_LEVEL."
        ),
    )
    SESSION_LOG_ZIP_COMPRESSION: Literal["stored", "deflated", "bzip2", "lzma"] = Field(
        default="lzma",
        description="Zip compression algorithm for session log archives (default lzma).",
    )
    SESSION_LOG_ZIP_COMPRESSLEVEL: int | None = Field(
        default=None,
        ge=0,
        le=9,
        description=(
            "Compression level for deflated (0-9) and bzip2 (1-9) zip compression; 0 "
            "stores a deflated entry without compressing it, and bzip2 has no such "
            "level. "
            "Ignored for stored/lzma. "
            "A 0 under bzip2 is read as 1."
        ),
    )
    SESSION_LOG_MAX_LINES: int = Field(
        default=20000,
        ge=100,
        le=200000,
        description=(
            "Maximum number of log records held in memory per request (older entries are dropped). "
            "This counts records, not bytes: a record's own size is bounded separately, so a large "
            "tool result is truncated to a fixed number of characters with the cut named in the record."
        ),
    )
    SESSION_LOG_FORMAT: Literal["jsonl", "text", "both"] = Field(
        default="jsonl",
        description=(
            "Format written inside session log archives. "
            "logs.jsonl is always written; 'jsonl' writes only it (one JSON object per record), while 'text' and 'both' additionally write a plain-text logs.txt (so 'text' and 'both' produce the same files). "
            "logs.txt writes one record per physical line, except for a record's exception block."
        ),
    )
    SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS: int = Field(
        default=30,
        ge=1,
        description=(
            "How often (in seconds) to check the database for log pieces waiting to be packed and build one zip per message. "
            "The same number is the wall-clock budget one pass gets, counted from when the pass starts assembling: it takes no new turn once this much time has elapsed, "
            "and leaves the rest of its window to the next pass. The pass spends its stale-lock sweep and its two candidate listings before that window opens, "
            "so on a slow host a pass can run longer than this number. Read on every pass, so a change applies with no restart."
        ),
    )
    SESSION_LOG_ASSEMBLER_JITTER_SECONDS: int = Field(
        default=10,
        ge=0,
        description="Random extra delay (0..N seconds) added to each wait between archiving passes so multiple workers do not run in lockstep.",
    )
    SESSION_LOG_ASSEMBLER_BATCH_SIZE: int = Field(
        default=25,
        ge=1,
        le=500,
        description=(
            "Maximum number of message bundles listed per archiving pass, in each of the two listings "
            "(finished turns and crash-stranded ones), so a pass lists up to twice this many; counted in turns, not rows. "
            "The count is of turns that can be assembled: one the assembler refuses by policy — a temporary chat's, which "
            "the pipe never archives — does not consume a slot, and is offered at most once per listing per pass, so a "
            "stranded one cannot hold the window. A turn whose lock another worker is already holding does the same for "
            "the rest of that pass: the pass sets it aside and lists again without it, so a contended head cannot shut the "
            "window either — which is why a pass that meets one may offer more than this number across its re-listings. "
            "A pass then assembles as many of those as SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS of wall clock allows, "
            "leaving the rest staged for the next pass."
        ),
    )
    SESSION_LOG_STALE_FINALIZE_SECONDS: int = Field(
        default=6 * 7200,
        ge=300,
        description=(
            "If a message has staged session-log segments but never signals that it finished "
            "(the worker crashed or was killed), finalize an incomplete zip after this many seconds since the last piece. "
            "This is a cutoff on the last segment, not on the turn: a turn still running when it passes is sealed as "
            "incomplete too, and a segment it stages afterwards is left stranded until the next assembly, unless it lands before "
            "the sealing pass has read that turn's segments - a pass that finds a terminal segment among the rows it loaded writes the "
            "turn complete instead; one that lands after that read is not folded into that archive by that pass and is "
            "picked up by a later one, unless the archive already records that turn as finished - "
            "a segment that lands after that read on a turn whose archive already records the turn as "
            "finished is folded in by that same pass, which writes the turn complete rather than sealing it "
            "again and also preserves the outcome that archive already recorded. That "
            "exposure is why the default is long. Each pass takes the oldest stranded bundles first and seals a "
            "bundle only if the sealed write succeeds, keeping the segments for a retry otherwise."
            "The incomplete marker is written at most once per archive: a pass that finds the turn still stale re-stamps that one marker "
            "rather than adding another - only its count is stable, and its timestamp is the last such pass - and a pass that finds the turn complete retires it. "
            "**Warning:** The minimum is `300` seconds (five minutes): a lower value is refused when the configuration is saved, and a stored one that no longer validates falls back to the default rather than being raised to it."
        ),
    )
    SESSION_LOG_LOCK_STALE_SECONDS: int = Field(
        default=1800,
        ge=60,
        description="Stale lock timeout (seconds) for DB-backed session log assembly locks; stale locks are reclaimed. It is also the write-failure backoff: a bundle whose archive could not be written is skipped for this long before it is retried, so one bundle the pipe cannot write does not hold the window. The one carve-out is the stranded-turn rescue: a turn whose existing archive could not be read is exempt from the backoff only while its own three-strike rescue budget lasts, and once that budget is spent it is set aside for this interval like any other bundle the pipe could not write. A turn whose rows were rescued to a separate archive is exempt on the pass that wrote that archive and set aside from the next one like any other bundle the pipe could not write. The rescue bookkeeping is itself bounded: at most 32 stranded turns are tracked at once, and beyond that the oldest is set aside early. A lock held by another worker is not a write failure and is never backed off.",
    )
    ENABLE_TIMING_LOG: bool = Field(
        default=False,
        description=(
            "When True, record how long each internal step of a request takes. "
            "Writes to TIMING_LOG_FILE path directly (not session archives); the per-request "
            "in-memory copy of those events is released when the request ends - by its own "
            "job completing, by its being refused before it enqueues, or by its being "
            "discarded without running - so only the file output persists. "
            "Useful for performance profiling and debugging latency issues."
        ),
    )
    TIMING_LOG_FILE: str = Field(
        default="logs/timing.jsonl",
        description=(
            "File path for timing log output when ENABLE_TIMING_LOG is True. "
            "Events are appended in JSONL format (one JSON object per line). "
            "Parent directories are created automatically if they don't exist."
        ),
    )
    MAX_CONCURRENT_REQUESTS: int = Field(
        default=200,
        ge=1,
        le=2000,
        description="Maximum number of in-flight OpenRouter requests allowed per process. Takes effect without a restart, in both directions: a higher value admits more requests at once, and a lower one binds from the moment it is saved, counting the requests already running, which finish first. A request holds its slot until its own tool calls have finished cleanup, so a tool-bearing request occupies its slot a little longer than its answer; a request the person stopped, or one that failed before its request body ran, returns its slot immediately, and so does one the worker is shut down while it is still running — a code reload cancels it and gives the slot back before the new worker starts taking work. The wait list behind this limit is bounded: the pipe queues further requests and sheds load once that queue is full — a chat caller sees a \"Server busy (503)\" card, and an API caller gets a 503 response carrying the same sentence. The same refusal reaches a request already waiting for a permit when the pipe is superseded, and is answered rather than cancelled.",
    )
    SSE_WORKERS_PER_REQUEST: int = Field(
        default=4,
        ge=1,
        le=8,
        description="Number of per-request workers that decode the streamed chunks of one reply.",
    )
    STREAMING_CHUNK_QUEUE_MAXSIZE: int = Field(
        default=0,
        ge=0,
        description="Maximum number of raw SSE chunks buffered before the pipe stops reading from OpenRouter until the backlog clears. 0=unbounded (cannot stall, recommended); bounded values risk stalls on tool-heavy loads, slow database writes, or a slow browser (a slow reader fills the decoded-event backlog, which fills the raw-chunk backlog, which stops the pipe reading from OpenRouter). A consumer that stops reading entirely is no longer held by that backpressure forever: cancelling the request tears the pipeline down and returns, leaving no background task behind.",
    )
    STREAMING_EVENT_QUEUE_MAXSIZE: int = Field(
        default=0,
        ge=0,
        description="Maximum number of decoded events buffered before the rest of the pipe handles them. 0=unbounded (cannot stall, recommended); bounded values risk stalls on tool-heavy loads, slow database writes, or a slow browser (a slow reader fills the decoded-event backlog, which fills the raw-chunk backlog, which stops the pipe reading from OpenRouter). A worker mid-put on this queue when the reply ends no longer holds the stream: cancelling the request tears the pipeline down and returns, leaving no background task behind.",

    )
    STREAMING_CHUNK_QUEUE_WARN_SIZE: int = Field(
        default=1000,
        ge=100,
        description="Log a warning when the raw-chunk backlog reaches this many chunks, so an unbounded buffer is still watched. What bounds those log records is time rather than this threshold: the queue logs one `WARNING` at the crossing and repeats it at `DEBUG` at most once every second while the backlog stays high, so a long turn costs a handful of lines instead of one per buffered chunk, and raising this threshold does not reduce that volume. The minimum of 100 keeps the crossing a sign of real pressure rather than of ordinary queue depth.",
    )
    STREAMING_EVENT_QUEUE_WARN_SIZE: int = Field(
        default=1000,
        ge=100,
        description="Log a warning when a buffered-event backlog reaches this many events, so an unbounded buffer is still watched. It covers two queues: the decoded-event backlog of the `/responses` path, and the pump queue that sits behind a `/chat/completions` reply, whose own buffer limit does not reach it. What bounds those log records is time rather than this threshold: the queue logs one `WARNING` at the crossing and repeats it at `DEBUG` at most once every second while the backlog stays high, so a long turn costs a handful of lines instead of one per buffered event, and raising this threshold does not reduce that volume. The minimum of 100 keeps the crossing a sign of real pressure rather than of ordinary queue depth.",
    )
    STREAMING_DELTA_CHAR_LIMIT: int = Field(
        default=256,
        ge=0,
        description=(
            "On/off switch for batching streamed output. "
            "When > 0 (default 256) batching is active — only whether the value exceeds 0 matters, its magnitude has no effect on batch size. Small pieces of streamed text are held back "
            "and combined whenever the browser cannot keep up, with STREAMING_NAGLE_MIN_FLUSH_CHARS "
            "controlling the minimum batch size. "
            "When 0 (and STREAMING_IDLE_FLUSH_MS is also 0), no batching: every piece is sent exactly as it arrives."
        ),
    )
    STREAMING_IDLE_FLUSH_MS: int = Field(
        default=30,
        ge=0,
        description=(
            "How long buffered text waits (ms) when the model pauses. "
            "Anything held back is sent after this interval so nothing sits on screen half-finished. "
            "0 turns off the time-based send (not recommended — buffered text is then only sent when the next chunk arrives or the reply ends)."
        ),
    )
    STREAMING_NAGLE_MIN_FLUSH_CHARS: int = Field(
        default=3,
        ge=1,
        description=(
            "Minimum buffered characters before a batch is sent at the end of "
            "each send pass. Default 3 avoids sending one character at a time when traffic is light. "
            "Set to 1 to send whatever has built up on every pass; 5-10 to batch harder and cut update events. The idle timeout "
            "(STREAMING_IDLE_FLUSH_MS) still guarantees delivery within its interval."
        ),
    )
    MIDDLEWARE_STREAM_QUEUE_MAXSIZE: int = Field(
        default=0,
        ge=0,
        description=(
            "Maximum number of per-request items buffered for the Open WebUI layer that streams the reply to the browser. "
            "The cap bounds incremental items: the item that reports a turn's failure and the item that ends the turn are put past it on a ceiling of their own, "
            "except on a turn the person stopped, where that record is put without waiting and a full buffer drops it. "
            "A third item is put past it too, and never went near the queue at all: the answer a job returns without having streamed, which the generator hands to the reader directly. "
            "0=unbounded (default behavior). A dropped item is re-sent by the next whole-message frame rather than lost."
        ),
    )
    MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS: float = Field(
        default=1.0,
        ge=0,
        description=(
            "When MIDDLEWARE_STREAM_QUEUE_MAXSIZE>0, maximum seconds to wait while adding one item to that buffer; "
            "if the wait expires the update is re-sent by the next whole-message frame rather than lost, and the stream continues. "
            "A dropped item is not recorded as sent, so the next snapshot re-derives the text this one would have carried. "
            "This wait covers incremental items only: the item that reports a turn's failure and the item that ends the turn are put past that valve on a ceiling of their own, which 0 does not remove, "
            "except on a turn the person stopped, where that record is put without waiting at all and a full buffer drops it. "
            "A third item is put past it too, and never went near the queue at all: the answer a job returns without having streamed, which the generator hands to the reader directly. "
            "0 disables the timeout (not recommended; a stalled browser can hold up the pipe indefinitely)."
        ),
    )
    OPENROUTER_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_OPENROUTER_ERROR_TEMPLATE,
        description=(
            "Markdown template used when OpenRouter rejects a request with a status that has no template of its own (400, 403, 404, 422, and so on), and when a failure reported inside a started reply resolves to such a status, from the kind OpenRouter named or, when that kind is unknown, from the code it sent. It is also the template for a content decision on any status: a body carrying `metadata.patterns`, `metadata.reasons` or `metadata.flagged_input` is a decline rather than a rate limit, a server fault, a timeout, an exhausted balance or an oversized request, so it renders here whatever status it arrived on -- the status itself does not change, and the `{status_code}` row and the pipe's own retry and latch decisions still see the number OpenRouter sent. The one exception is a `401`, which stays a rejected credential whatever the body beside it carries. Clear this box and save to restore this built-in text. Placeholders such as {heading}, {detail}, {sanitized_detail}, {provider}, {model_identifier}, {requested_model}, {api_model_id}, {normalized_model_id}, {openrouter_code}, {upstream_type}, {reason}, {request_id}, {request_id_reference}, {openrouter_message}, {upstream_message}, {moderation_reasons}, {flagged_excerpt}, {raw_body}, {context_limit_tokens}, {max_output_tokens}, {include_model_limits}, {metadata_json}, {provider_raw_json}, {error_id}, {timestamp}, {session_id}, {user_id}, {native_finish_reason}, {error_chunk_id}, {error_chunk_created}, {is_streaming_error}, {streaming_provider}, {streaming_model}, {retry_after_seconds}, {rate_limit_type}, {required_cost}, and {account_balance} are replaced when values are available. Lines whose **own** placeholder resolves to a missing or empty value are omitted automatically; a value that itself contains a placeholder in braces is shown verbatim, never re-read as a placeholder. A template whose rendered card comes out empty -- every line inside a `{{#if}}` whose value is absent, which is what happens when a card is written entirely out of the values a channel chat withholds -- falls back to the built-in provider-error card rather than emitting nothing, so a reader always gets a card and a support handle to quote. `{streaming_provider}` and `{streaming_model}` are filled only for a failure reported inside a reply that has already started, on a rejected request they are empty however the provider is named, and they are also empty when such a failure names no provider at all. "
            + _BOOLEAN_PLACEHOLDER_RULE
            + " The pipe does the span and fence work on these values itself: a value placed inside a backtick span, on a `### ` heading, or on a bare `**…**` / `- ` line arrives as one logical line with its backticks removed, and a value placed in a fenced block arrives inside a fence long enough to contain it, so a custom template does not have to. A fence written inside a blockquote keeps that quote on every line the renderer emits from it, and a `>` alone on that line is not a label. The pipe's own numbers and labels (`status_code`, `retry_after_seconds`, `context_limit_tokens`, `max_output_tokens`, `diagnostics`) are already single-line. {error_chunk_created} is rendered as a Z-suffixed UTC ISO-8601 instant when the provider sends a Unix epoch, and as the provider's own text otherwise. Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to show them only when that value is set. {metadata_json} and {provider_raw_json} are the provider's metadata and its raw error block, including any field the pipe itself adds; so are the five copies of the provider's own message the card carries inline -- {detail}, {sanitized_detail}, {reason}, {upstream_message} and {openrouter_message} -- and the joined {moderation_reasons} list. All seven are cut at 16,384 characters, each with a marker naming how many characters were removed, so a value that arrives cut is no longer the whole one. The two marker shapes differ: a fenced JSON value carries its marker on a line of its own inside the fence, while a cut inline value carries it inside the value, space-joined onto the card's line, because an inline value is always one logical line. {raw_body} and {flagged_excerpt} are never cut: they are the provider's own bytes and reach the card whole on purpose, so that what a person reads is what the provider sent. The complete values stay on the error object and in the session log, which is where an operator who needs the whole payload reads it."
            + _CHANNEL_CARD_RULE
        ),
    )
    ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE: str = Field(
        default=DEFAULT_ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE,
        description=(
            "Markdown template used when a request and the endpoint its model is forced to disagree: a request that "
            "requires /chat/completions (e.g. direct video uploads) on a model explicitly forced to /responses by "
            "endpoint override valves, or a request that requires /responses (e.g. a Fusion model, whose panel "
            "renders only from /responses — on /chat/completions Fusion returns a flattened text transcript "
            "with no structured events) on a model forced to /chat/completions."
        ),
    )
    DIRECT_UPLOAD_FAILURE_TEMPLATE: str = Field(
        default=DEFAULT_DIRECT_UPLOAD_FAILURE_TEMPLATE,
        description=(
            "Markdown template used when OpenRouter Direct Uploads cannot be applied (e.g. incompatible attachment combinations, "
            "missing stored files, or other checks that fail before the request is sent)."
        ),
    )

    AUTHENTICATION_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_AUTHENTICATION_ERROR_TEMPLATE,
        description=(
            "Markdown template for HTTP 401 errors, for an authentication failure OpenRouter reports, on either path, and for the pipe's own failure to read a usable API key. All three fill {error_id}, {timestamp}, {session_id}, {user_id}, {support_email}, {support_url}, {openrouter_code} and {openrouter_message}. A 401 returned by OpenRouter also fills the shared error-context fields — {request_id}, {provider}, {model_identifier}, {requested_model}, {reason}, {metadata_json} and the rest of the set the rejected-request template lists. A failure reported inside a started reply fills the same set, except for the fields OpenRouter has to send for them to exist: {request_id} only when the failure carries an id: the generation id on Chat Completions, the failed response's id on Responses, and {provider} only when the error itself names the provider. Nothing is sent when the key itself cannot be read, so on that path those extra fields have no value and any line using one prints the braces verbatim; wrap such a line in {{#if request_id}}...{{/if}} and it is left out instead. A name nothing supplies is never substituted, whichever path rendered the card. If the whole card comes out empty, because every line sat inside a `{{#if}}` whose value that path does not supply, the built-in authentication-failure card is shown instead of nothing, so a reader always gets the reason and an error id to quote."
            + _CHANNEL_CARD_RULE
        ),
    )

    INSUFFICIENT_CREDITS_TEMPLATE: str = Field(
        default=DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE,
        description=(
            "Markdown template for HTTP 402 errors when the account is out of credits, and for a payment_required failure OpenRouter reports, on either path. A decline carrying `metadata.patterns`, `metadata.reasons` or `metadata.flagged_input` does not reach it, whatever status the decline arrived on: that is OPENROUTER_ERROR_TEMPLATE, which renders the moderation rows. Supports {error_id}, {timestamp}, {openrouter_code}, {openrouter_message}, {request_id}, {required_cost}, {account_balance}, {support_email}, and other shared context variables. {request_id} is OpenRouter's own reference for the rejected request, which its support can look up; the built-in text shows it on its own row whenever the rejection carried one."
            + _CHANNEL_CARD_RULE
        ),
    )

    RATE_LIMIT_TEMPLATE: str = Field(
        default=DEFAULT_RATE_LIMIT_TEMPLATE,
        description=(
            "Markdown template for HTTP 429 rate-limit errors, and for a rate_limit_exceeded failure OpenRouter reports, on either path. A decline carrying `metadata.patterns`, `metadata.reasons` or `metadata.flagged_input` does not reach it even under the rate_limit_exceeded kind, whatever status the decline arrived on: that is OPENROUTER_ERROR_TEMPLATE, which renders the moderation rows. Use placeholders such as {error_id}, {timestamp}, {openrouter_code}, {retry_after_seconds}, {rate_limit_type}, {request_id}, {support_email}, and the standard context variables. {request_id} is OpenRouter's own reference for the rejected request, which its support can look up; the built-in text shows it on its own row whenever the rejection carried one. A {{#if}} block renders its body for any value the pipe supplies, and a number is a value at every number: a {retry_after_seconds} parsed from an HTTP-date that has already expired is 0, and its row renders as 0s, which says the delay is over."
        ),
    )

    SERVER_TIMEOUT_TEMPLATE: str = Field(
        default=DEFAULT_SERVER_TIMEOUT_TEMPLATE,
        description=(
            "Markdown template for a 408 from OpenRouter (server-side timeout), whether that is the reply's own status or the code reported inside a reply already under way. A provider timeout reported inside a started reply is documented as 504 and renders SERVICE_ERROR_TEMPLATE instead. A `408` is not retried. A `408` that names the provider-timeout kind is retried, because the kind resolves it to a `504`. A decline carrying `metadata.patterns`, `metadata.reasons` or `metadata.flagged_input` does not reach it, whatever status the decline arrived on: that is OPENROUTER_ERROR_TEMPLATE, which renders the moderation rows. Supports the common context variables plus {openrouter_message}, {openrouter_code}, {request_id}, and support contact placeholders. {request_id} is OpenRouter's own reference for the timed-out request, which its support can look up; the built-in text shows it on its own row whenever the response carried one."
            + _CHANNEL_CARD_RULE
        ),
    )

    PAYLOAD_TOO_LARGE_TEMPLATE: str = Field(
        default=DEFAULT_PAYLOAD_TOO_LARGE_TEMPLATE,
        description=(
            "Markdown template for HTTP 413 errors when the request payload exceeds size limits, and for a payload_too_large failure OpenRouter reports, on either path. A decline carrying `metadata.patterns`, `metadata.reasons` or `metadata.flagged_input` does not reach it, whatever status the decline arrived on: that is OPENROUTER_ERROR_TEMPLATE, which renders the moderation rows. Supports {error_id}, {timestamp}, {openrouter_code}, {openrouter_message}, {model_identifier}, {request_id}, {support_email}, and other shared context variables. {request_id} is OpenRouter's own reference for the rejected request, which its support can look up; the built-in text shows it on its own row whenever the rejection carried one."
            + _CHANNEL_CARD_RULE
        ),
    )

    # Support configuration
    SUPPORT_EMAIL: str = Field(
        default="",
        description=(
            "Support email displayed in error messages. "
            "Leave empty if self-hosted without dedicated support."
        )
    )

    SUPPORT_URL: str = Field(
        default="",
        description=(
            "Support URL (e.g., internal ticket system, Slack channel). "
            "Shown in error messages if provided."
        )
    )

    # Additional error templates
    NETWORK_TIMEOUT_TEMPLATE: str = Field(
        default=DEFAULT_NETWORK_TIMEOUT_TEMPLATE,
        description=(
            "Markdown template a chat reply shows, once the retries are spent, when its call to OpenRouter times out before any of the answer arrives. A timeout before the first byte is a temporary failure: the request is re-sent up to TRANSIENT_RETRY_MAX_ATTEMPTS extra times (three attempts in all by default) and is counted once against the breaker however many attempts it took. Picture-only image models, video models and the panel, judge and final-answer calls inside internal Fusion report failures in their own way. {timeout_seconds} is the limit that ran out: HTTP_CONNECT_TIMEOUT_SECONDS while connecting, HTTP_SOCK_READ_SECONDS while waiting for data, or HTTP_TOTAL_TIMEOUT_SECONDS for the whole request. Once OpenRouter has accepted the request - after part of the answer has arrived, or while the body of a stream it has already answered is still arriving - STREAM_INTERRUPTED_TEMPLATE is used instead and nothing is retried, because a request that was accepted is billed again by every re-send; a tool call the model has already named closes that window on its own, with no answer text of its own. Available variables: {error_id}, {timeout_seconds}, {timestamp}, {session_id}, {user_id}, {support_email}, {support_url}. Supports Handlebars-style conditionals: wrap a section in {{#if variable}}...{{/if}} to show it only when that value is set."
            + _CHANNEL_CARD_RULE
        )
    )

    CONNECTION_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_CONNECTION_ERROR_TEMPLATE,
        description=(
            "Markdown template a chat reply shows, once the retries are spent, when its connection to OpenRouter fails before any of the answer arrives: the connection cannot be opened or drops, or, on every attempt, OpenRouter closes the stream without sending anything, or sends frames the pipe cannot read, or sends frames the pipe can read that published nothing the reader could use, or a non-streamed 200 answers on /responses with no `output` key at all, or with an `output` that is neither a list nor null, or on /chat/completions with no `choices`. A connection that fails before the first byte is a temporary failure: the request is re-sent up to TRANSIENT_RETRY_MAX_ATTEMPTS extra times (three attempts in all by default) and is counted once against the breaker however many attempts it took. Picture-only image models, video models and the panel, judge and final-answer calls inside internal Fusion report failures in their own way. A timeout uses NETWORK_TIMEOUT_TEMPLATE instead, and once OpenRouter has accepted the request - after part of the answer has arrived, or while the body of a stream it has already answered is still arriving - or the model has named the tool it is calling, STREAM_INTERRUPTED_TEMPLATE is used and nothing is retried, because a request that was accepted is billed again by every re-send. A reply that arrives on an accepted status but whose body is not a JSON object is not a connection failure: the connection worked, and SERVICE_ERROR_TEMPLATE reports it; a well-formed object that carries no answer on either route because the key is absent is a different thing and is reported here. A non-streamed 200 on /responses whose `output` is present but empty or null is neither: the connection worked and the model returned nothing, so it is retried on the same TRANSIENT_RETRY_MAX_ATTEMPTS budget and then reported by OPENROUTER_ERROR_TEMPLATE, whose reason says the model returned an empty answer. Available variables: {error_id}, {error_type}, {timestamp}, {session_id}, {user_id}, {support_email}, {support_url}. Supports Handlebars-style conditionals: wrap a section in {{#if variable}}...{{/if}} to show it only when that value is set."
            + _CHANNEL_CARD_RULE
        )
    )

    SERVICE_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_SERVICE_ERROR_TEMPLATE,
        description=(
            "Markdown template for OpenRouter 5xx errors, for an accepted response whose body is not a JSON object (a proxy, CDN or WAF in front of this deployment rewrote the reply), and for a failure OpenRouter reports inside a reply it has already started under one of its own typed codes: provider_unavailable, provider_overloaded, timeout, server or unmapped, or under its native code server_error. A decline is not one of those failures whatever 5xx it arrived on: a body carrying `metadata.patterns`, `metadata.reasons` or `metadata.flagged_input` renders OPENROUTER_ERROR_TEMPLATE instead, which names the moderation rows, and the status it keeps is still the 5xx the pipe retried on and counted against the breaker. Available variables: {error_id}, {status_code}, {reason}, {timestamp}, {session_id}, {user_id}, {support_email}. A 5xx that OpenRouter itself returned also fills {request_id}, its own reference for that request. A 5xx reported inside a started reply fills it whenever the failure carries an id: the failed response's id on Responses, the generation id on Chat Completions, also available as {error_chunk_id}; {provider} likewise appears only when the error names the provider. A 5xx raised by the connection to OpenRouter carries no such reference and a line using it prints the braces verbatim unless it is wrapped in a conditional. An accepted response whose body is not decodable at all also fills {body_excerpt} with the provider's own first 200 characters, already inside a code fence: {reason} then names only the endpoint and the Content-Type, and the template must NOT fence {body_excerpt} again -- a second fence either nests inside the value's or, in a stored row that already fences the placeholder, collapses onto it. Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to show them only when that value is set."
            + _CHANNEL_CARD_RULE
        )
    )

    INTERNAL_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_INTERNAL_ERROR_TEMPLATE,
        description=(
            "Markdown template for unexpected internal errors the pipe itself owns. "
            "A response that arrived on an accepted status but whose body is not a JSON object is not one of those: SERVICE_ERROR_TEMPLATE reports that, because the network path, not the pipe, produced it. "
            "Available variables: {error_id}, {error_type}, {timestamp}, "
            "{session_id}, {user_id}, {support_email}, {support_url}. "
            "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to show them only when that value is set."
            + _CHANNEL_CARD_RULE
        )
    )

    MODEL_RESTRICTED_TEMPLATE: str = Field(
        default=DEFAULT_MODEL_RESTRICTED_TEMPLATE,
        description=(
            "Markdown template emitted when the requested model is blocked by MODEL_ID and/or model filter valves. "
            "Available variables: {requested_model}, {normalized_model_id}, {restriction_reasons}, "
            "{model_id_filter}, {free_model_filter}, {tool_calling_filter}, plus standard context variables "
            "like {error_id}, {timestamp}, {session_id}, {user_id}, {support_email}, and {support_url}."
            + _CHANNEL_CARD_RULE
        ),
    )
    VIDEO_CATALOG_LOADING_TEMPLATE: str = Field(
        default=DEFAULT_VIDEO_CATALOG_LOADING_TEMPLATE,
        description=(
            "Markdown template emitted when the requested model is absent from the catalogue only because the "
            "video model catalogue is still being read for this worker, so the sweep has not registered it yet. "
            "Available variables: {requested_model}, {normalized_model_id}, plus standard context variables "
            "like {error_id}, {timestamp}, {session_id}, {user_id}, {support_email}, and {support_url}."
            + _CHANNEL_CARD_RULE
        ),
    )
    STREAM_INTERRUPTED_TEMPLATE: str = Field(
        default=DEFAULT_STREAM_INTERRUPTED_TEMPLATE,
        description=(
            "Markdown template appended to the assistant message when a reply's stream stops before its final event: the stream closes early or, after answer text has arrived or the model has named the tool it is calling, its connection fails, drops or times out. Any partial content is kept, with this notice after it. The panel, judge and final-answer calls inside internal Fusion never get this notice. When none of the answer has arrived, NETWORK_TIMEOUT_TEMPLATE is used instead for a timeout, and CONNECTION_ERROR_TEMPLATE for a failed connection or a stream that sent nothing. Available variables: {model}, {timestamp}, {support_email}, {support_url}."
        ),
    )

    MAX_PARALLEL_TOOLS_GLOBAL: int = Field(
        default=200,
        ge=1,
        le=2000,
        description=(
            "Global ceiling for simultaneously executing tool calls; Open WebUI's ask_user takes no slot. "
            "Takes effect without a restart, in both directions: a higher value admits more calls at once, "
            "and a lower one binds from the moment it is saved, counting the calls already running, which finish first. "
            "It also sizes the pool of threads a tool written as a plain function runs on, up to a fixed cap of 8, so a blocking one "
            "cannot consume the threads Open WebUI's own requests and this pipe's storage work use; raising it above 8 admits no "
            "further plain-function tool concurrency, and a tool written as `async def` never takes a thread at all."
        ),
    )
    MAX_PARALLEL_TOOLS_PER_REQUEST: int = Field(
        default=5,
        ge=1,
        le=50,
        description=(
            "Per-request limit on simultaneously executing tool calls, and the number of tool workers each request that runs tools starts. Each internal Fusion model that runs tools starts the same number of workers and shares the request's slots; Open WebUI's ask_user takes no slot."
        ),
    )
    BREAKER_MAX_FAILURES: int = Field(
        default=5,
        ge=1,
        le=50,
        description=(
            "Number of failures one user may accumulate before their requests are refused, a failing tool is skipped, or their database reads and writes are skipped; raise it for fewer trips in noisy environments. A request failure is a failed chat call to OpenRouter (an error reply; a connection that cannot be opened, drops or times out; an error reported inside a response; or a stream that stops before its final event). A request counts once however many attempts it took: a 429, a 5xx, a 408 that names the provider-timeout kind, or that failure reported inside a response before anything has been shown is retried first, up to TRANSIENT_RETRY_MAX_ATTEMPTS extra tries, and the request is a single failure whichever attempt gave up on it. A /responses failure that AUTO_FALLBACK_CHAT_COMPLETIONS repairs on /chat/completions is the one exception and is not counted at all: that one strike is taken back as soon as the fallback is decided, and the user's other failures in the window are left standing. A body carrying a content decision is the exception and is never retried, whatever status it arrived on. An internal Fusion run is one such request: its panel, judge and final-answer calls each spend nothing, and the run spends one failure of its own when no panel model answered, whatever the panel's size and however many of those calls failed. A generation on a picture-only image model or a video model that fails after it was sent to OpenRouter is also a request failure, counted once, except a generation whose response arrived whole and was not an OpenRouter document (a body a proxy, CDN or WAF rewrote, or one that is not a JSON object at all): that is not a failure of the request and never counts toward the limit. Request and database failures count within BREAKER_WINDOW_SECONDS. Raising or lowering the setting mid-session re-reads the failures already recorded: it does not discard them, and a lowered setting applies from the next request without evicting anything. Request failures clear when a request ends without an error (for a picture-only image model or a video model, only once its result is delivered whole; a stream that breaks after the finished image arrived is a delivered result and a failed call, counted once and clearing nothing, and for an internal Fusion run, only if a panel model answered); a request the user stops clears no request failures. Housekeeping tasks such as title generation neither count nor clear; Open WebUI's merge-responses task counts but never clears. Database failures also clear when a database operation succeeds. A fault in the artifact cache refill after a successful read is neither counted nor reported as a lost round: the rows are returned, the window still clears, and a WARNING names the cache rather than the database. A fault sealing rows for storage on the way to the database is likewise neither counted nor blamed on it: nothing reached the database, the round is dropped, and a WARNING names the seal and the affected row count. The request breaker never refuses a request whose last message is a tool result, or is Open WebUI's own message that comes right after a tool result and hands the model a tool's images; a question or picture the user sends is refused like any other request. The exemption is from the refusal only: a request like that that fails still counts against the limit, and one that ends without an error still clears the count. Each tool counts its failures in a row: errors it raises, per-call timeouts, running calls cut off by TOOL_BATCH_TIMEOUT_SECONDS, and calls whose tool server cannot be reached or answers with an HTTP error status; an ask_user timeout and a call to an MCP tool whose session has closed do not count. The count belongs to the tool the call resolved to, so a name the model padded with surrounding whitespace is the same tool and spends the same budget. A SystemExit, KeyboardInterrupt or GeneratorExit from a tool is shown as failed but is not a failure of that tool: it signals the process rather than the tool, and it never adds to the count or clears it. An error the tool reports in a result it returns normally is shown as failed but neither adds to the count nor clears it, whether the judgement is made by Open WebUI's own classifier or by the pipe's copy of it A tool the breaker has taken out of service for a call it refuses before the batch is queued is announced once per round, however many such calls that round asked for; a call it takes out of service after the batch was queued is announced for that call alone. A rejected credential is a separate mechanism this valve does not govern: a 401, or a 403 that names no kind and carries no content decision, arms a fixed 60-second pause on that user's background tasks rather than a failure this valve records, and neither this valve nor BREAKER_WINDOW_SECONDS has any say in it. That pause is one timestamp rather than a count - no setting changes its length, a request that succeeds does not clear it, and another rejected credential re-arms it - and while it holds, that user's Open WebUI background tasks are refused, each with `OpenRouter access is temporarily disabled after an authentication failure.`, whereas their chat requests are not refused and reach the model as usual. The dashboard's `Auth` row is how many users are inside that pause at once."
        ),
    )
    BREAKER_WINDOW_SECONDS: int = Field(
        default=60,
        ge=5,
        le=900,
        description=(
            "Number of seconds a failure keeps counting. The request and database breakers use it as a trailing window: it is what retires a recorded request failure, at the next check. A tool's failure count clears on a success, or when the tool is next called more than this long after its last failure; inside an internal Fusion run, only a success clears the run's count for that tool."
        ),
    )
    TOOL_BATCH_CAP: int = Field(
        default=4,
        ge=1,
        le=32,
        description=(
            "Maximum number of consecutive calls to one tool that may run in a single batch."
        ),
    )
    TOOL_OUTPUT_RETENTION_TURNS: int = Field(
        default=10,
        ge=0,
        description=(
            "How many recent logical turns have their tool outputs sent in full. A turn starts at a person's message that follows the assistant's reply and runs to the next such message; messages the person sends back to back belong to the same turn. In older turns, long tool outputs are shortened to save tokens, whether they come from a saved tool card or the pipe's own storage; OpenRouter's own advisor, subagent and model-search items go back whole. Apart from answers given through ask_user, this matters only while tool results are kept across turns. Set to 0 to keep every tool output in full."
        ),
    )
    TOOL_TIMEOUT_SECONDS: int = Field(
        default=300,
        ge=1,
        le=600,
        description=(
            "Maximum seconds one tool call may run, counted from when it starts; Open WebUI's built-in ask_user waits for its question window plus 15 seconds instead. A timed-out call counts as a failure of that tool; the generous default reduces disruption for real-world tools."
        ),
    )
    TOOL_BATCH_TIMEOUT_SECONDS: int = Field(
        default=600,
        ge=1,
        description=(
            "Maximum seconds a batch of calls to one tool may take once a worker has picked it up, and never less than TOOL_TIMEOUT_SECONDS. When the limit is reached, finished calls keep their results and calls still running or waiting are cancelled; the round ends as soon as the cancelled calls have had five seconds to acknowledge, and one that does not is dropped from the round, keeps running, and its result never reaches the model. It is also the ceiling on the time one response may take putting its calls on the queue: a round whose calls outnumber the free workers stops waiting and reports the calls it never started. Open WebUI's built-in ask_user raises this limit to its question window plus 15 seconds when that is longer, and the cut-batch message then says that the limit it quotes is that question window rather than this one, giving this valve's value alongside."
        ),
    )
    TOOL_IDLE_TIMEOUT_SECONDS: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Maximum seconds to wait in total for one response's tool results, counted once from when the model asked; every call that had started and whose result has not arrived by then is reported as timed out, however long it has been running, and one still waiting for a tool worker or a free slot is reported as not started rather than as an idle timeout; Open WebUI's ask_user waits at least its question window. On timeout, a call that is already running continues until it finishes, another tool limit ends it, or request cleanup cancels it after TOOL_SHUTDOWN_TIMEOUT_SECONDS (inside internal Fusion, without that wait, as soon as the calling model's answer ends). The model never receives the late result. Outside internal Fusion, files or embeds the call returns still appear in the chat: a streamed reply waits for the call, bounded by TOOL_SHUTDOWN_TIMEOUT_SECONDS, so a call that returns during that wait is read before the reply ends, and a file it shows through Open Terminal opens in the preview panel (or nowhere for a person whose Open WebUI shows terminal files inline). A call still waiting for a slot or a worker never starts. Null means no limit, leaving TOOL_TIMEOUT_SECONDS and TOOL_BATCH_TIMEOUT_SECONDS in charge. It is not the limit on getting calls started: a round whose calls outnumber the free workers is additionally bounded by TOOL_BATCH_TIMEOUT_SECONDS, and the calls it never started are reported as not started rather than as idle timeouts."
        ),
    )
    TOOL_SHUTDOWN_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        ge=0,
        description=(
            "Maximum seconds to wait for a request's unfinished tool calls to finish during cleanup; after a Stop, calls that had not started can still start in this time. A streamed reply ends after this wait and is bounded by it, so raising the valve raises how long a browser waits for a reply whose tools are still running. 0 disables the graceful wait and cancels workers immediately. Inside internal Fusion, a model's tool workers are cancelled without this wait as soon as that model's answer ends."
        ),
    )
    ENABLE_REDIS_CACHE: bool = Field(
        default=True,
        description="Buffer artifact writes through Redis when REDIS_URL and more than one worker are detected. It is re-read on each request, so flipping it applies to the next message with no restart. A connection attempt that fails is retried on an interval rather than on every request: 300 seconds, a fixed constant and not a valve. Turning this valve off and on again, and a changed REDIS_URL, both clear that wait. The valve is authoritative at the write in the very turn that turns it off, that turn's row goes straight to the database instead of Redis, and everything still buffered is drained to the database to completion before the Redis client is closed, and the same drain runs when the worker shuts down, before its Redis tasks are cancelled, so a row the drain cannot commit stays in the pending queue for another worker. While it is off no artifact data is written to Redis, but a delete is not data and still runs: a cleanup in that window still writes its delete marker and still deletes the cache entries of the rows it was told to forget, and the drain uses its pending queue and flush lock only to empty the queue. It stays off until you turn it on again, and turning it back on brings it up on the next request without a restart, once the worker reconnects to Redis. A drain that never starts - a shutdown, a hot reload, or a loop swapped between the off edge and the drain's first step - leaves the off edge disarmed rather than latched, so the next time you turn it on it re-arms on the next request.",
    )
    REDIS_CACHE_TTL_SECONDS: int = Field(
        default=600,
        ge=60,
        le=3600,
        description="TTL applied to Redis artifact cache entries (seconds).",
    )
    REDIS_PENDING_WARN_THRESHOLD: int = Field(
        default=100,
        ge=1,
        le=10000,
        description="Emit a warning when the Redis pending queue exceeds this number of artifacts.",
    )
    REDIS_FLUSH_FAILURE_LIMIT: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Log a critical alert after this many consecutive flush failures - a flush that fails, in the database or in Redis, because taking the lock, reading the queue depth, popping a batch and releasing the lock are all part of the same pass. Buffering is not disabled: the pipe waits longer between attempts and keeps retrying, resuming when writes succeed. New writes keep queueing meanwhile and only bypass Redis if the enqueue itself fails. A flush interrupted by a cancellation (worker shutdown, hot reload, valve flip) also puts its uncommitted batch back on the queue rather than dropping it.",
    )
    COSTS_REDIS_DUMP: bool = Field(
        default=False,
        description="When True, push per-request usage snapshots into Redis for downstream cost analytics. Only written while Redis buffering of artifact writes is active (same prerequisites as ENABLE_REDIS_CACHE); the ENABLE_REDIS_CACHE valve is authoritative at the write in the very turn that turns it off, so no snapshot is written from that turn on while it is off.",
    )
    COSTS_REDIS_TTL_SECONDS: int = Field(
        default=900,
        ge=60,
        le=3600,
        description="TTL (seconds) applied to cost analytics Redis snapshots.",
    )
    ARTIFACT_CLEANUP_DAYS: int = Field(
        default=90,
        ge=1,
        le=365,
        description="Days an artifact is kept before cleanup. Its stored timestamp is refreshed on every read of the artifact that goes through the database store — whether that read is served from the database or from the cache — so retention runs from last access, not creation. Reads that never reach the database store do not refresh it, either because the database breaker is open or because the artifact store failed to initialise. The cache only serves reads on a deployment with the Redis artifact cache enabled. The sweep also removes the Redis cache entries of the rows it deletes, so a purged artifact is not replayed from the cache either. Rows a temporary chat left behind are deleted at the next cleanup, whatever their age. The sweep covers the pipe's own artifacts only: it does not touch the pipe's bookkeeping rows in the same table — the staged session-log segments and the coordination locks — and those are removed by their owners instead, staged segments on SESSION_LOG_RETENTION_DAYS and locks by a stale-lock rule. The sweep also does not cover files the pipe put in Open WebUI's storage.",
    )
    ARTIFACT_CLEANUP_INTERVAL_HOURS: float = Field(
        default=1.0,
        ge=0.5,
        le=24,
        description="Frequency (hours) for the artifact cleanup worker to wake up.",
    )
    DB_BATCH_SIZE: int = Field(
        default=10,
        ge=5,
        le=20,
        description="Number of artifacts to commit per DB batch.",
    )
    USE_MODEL_MAX_OUTPUT_TOKENS: bool = Field(
        default=False,
        description="When enabled, and the request does not already set a limit, fill in an output allowance: the smaller of the model's advertised max_output_tokens and half its context window, or the advertised value alone when no context window is known. Models advertising neither are left unset. Disable to send no limit of the pipe's own. A routing variant such as base:nitro resolves through its base's catalog row, so it inherits the base's ceiling. This valve controls the automatic value, not yours: a max_tokens, max_output_tokens or max_completion_tokens of 1 or above is forwarded as a whole number whatever its spelling, and a fractional value rounds. OpenRouter documents the parameters as 1 or above and Open WebUI's slider reaches -2, so a value below 1 is sent as no cap -- which means the automatic ceiling applies if this valve is on.",
    )
    SHOW_FINAL_USAGE_STATUS: bool = Field(
        default=True,
        description="When True, the final status line includes elapsed time, token usage, and the cost of the generation, with the cost shown only when it is above zero; when False, that status line reports elapsed time alone.",
    )
    FINAL_USAGE_STATUS_STYLE: Literal["text", "icons"] = Field(
        default="text",
        description="Choose text labels or icons for the final usage status line.",
    )
    USAGE_STATUS_ICON_SET: str = Field(
        default="⧗,$,⇅,▲,▼,↺,▽",
        description=(
            "CSV of icons for the final usage status fields "
            "(time,cost,total,input,output,cached,reasoning). "
            "Only used when FINAL_USAGE_STATUS_STYLE is set to 'icons'."
        ),
    )
    SEND_END_USER_ID: bool = Field(
        default=False,
        description="When True, send OpenRouter `user` and `safety_identifier` (both carrying the value chosen by END_USER_ID_SOURCE), and also include `metadata.user_id` with the Open WebUI user GUID. A `user` or `safety_identifier` a client supplies is discarded, so neither can be forged.",
    )
    END_USER_ID_SOURCE: Literal["id", "email", "name"] = Field(
        default="id",
        description="What the OpenRouter `user` field carries when SEND_END_USER_ID is on: the Open WebUI GUID, the user's email, or their display name. Email/name fall back to the GUID when empty. Sending email or name shares PII with OpenRouter.",
    )
    SEND_SESSION_ID: bool = Field(
        default=False,
        description="When True, include the Open WebUI session_id as `metadata.session_id` (metadata only). A temporary chat's session id is never sent: for such a chat the valve is skipped.",
    )
    SEND_CHAT_ID: bool = Field(
        default=False,
        description="When True, include the Open WebUI chat_id as `metadata.chat_id` (metadata only). A temporary chat's chat id is never sent: for such a chat the valve is skipped.",
    )
    SEND_MESSAGE_ID: bool = Field(
        default=False,
        description="When True, include the Open WebUI message_id as `metadata.message_id` (metadata only). A temporary chat's message id is never sent: for such a chat the valve is skipped.",
    )
    MAX_INPUT_IMAGES_PER_REQUEST: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of images one turn forwards to the provider, counting a picture reused from earlier in the conversation, and counting every message of the person's in that turn -- including the back-to-back messages a caller's `messages` array produces, which are one turn and not several. Pictures are kept in the order they were sent, up to the limit, so on a turn carrying more than the limit the earliest survive and the rest are dropped with one `Images: dropped N over the limit of M.` for the turn. Pictures a tool returns are not counted: they are never cut by this limit. Whenever a tool round's result reaches the model, every one of its pictures that clears the tool-picture gate goes with it, as in Open WebUI's own tool loop: none of them is cut by this limit, and one the gate refuses is named in the chat rather than dropped silently.",
    )
    IMAGE_INPUT_SELECTION: Literal["user_turn_only", "user_then_assistant"] = Field(
        default="user_then_assistant",
        description=(
            "Controls which images are forwarded to the provider. "
            "'user_turn_only' restricts inputs to the images supplied with the current turn. "
            "'user_then_assistant' falls back to the most recent image already in the conversation, from either side, when the user did not attach any; a tool round ends the window for pictures from before it, so nothing older is reused after one whether or not it returned a picture. A round that asked you a question is not a media round and does not end the window, and neither does a picture the model shows you in its own reply to a round. It also governs a picture the model wrote into an earlier reply: that picture is re-sent as a picture where the reuse window can still reach it, and is otherwise left in the reply as the destination it was written as, which is a base64 data URL for a chat with nowhere to store a file and an Open WebUI file link for a saved one. The window cannot reach it when the reply is more than IMAGE_REUSE_MAX_TURNS turns back, when a tool round has closed the window, when you attach a picture of your own, when the model takes no image input, or when MAX_INPUT_IMAGES_PER_REQUEST has no room left for it. One picture is left in the reply neither way: one over BASE64_MAX_SIZE_MB, which the pipe has already refused to send, is reported and keeps its placeholder rather than riding along as the text it was written as; 'user_turn_only' sends it to nobody."
        ),
    )
    IMAGE_REUSE_MAX_TURNS: int = Field(
        default=3,
        ge=1,
        le=50,
        description=(
            "How many turns an earlier picture stays available for reuse under "
            "'user_then_assistant' when the user attaches nothing. This is a budget on "
            "the pixels the pipe resends: past it a picture is no longer sent again as "
            "an image, and stays in the conversation as the text the model wrote it in, "
            "so a long text conversation stops paying to resend an image nobody is "
            "talking about any more."
        ),
    )

    # Model metadata synchronization
    UPDATE_MODEL_IMAGES: bool = Field(
        default=True,
        description="When enabled, automatically sync profile image URLs from OpenRouter's frontend catalog to Open WebUI model metadata, falling back to a model maker's logo when the catalog answered and has no icon for that model. A pass whose catalog read did not answer leaves every stored icon alone, whoever put it there, and the next pass that does answer applies the fallback. While the remembered source URL is unchanged the download is skipped, and for a model taking its maker's logo the maker's page is not re-fetched either. The two sweeps this runs -- the maker pages, and the icon downloads -- are each bounded as a whole, not only per read, so a slow icon host delays one pass by a bounded time rather than by the size of the catalogue. A model the budget abandoned keeps the icon it already has and is offered again on the next pass. Disable to manage images manually.",
    )
    UPDATE_MODEL_CAPABILITIES: bool = Field(
        default=True,
        description="When enabled, automatically sync model capabilities (vision, file_upload, web_search, etc.) from OpenRouter's API catalog to Open WebUI model metadata. The web_search and citations checkboxes are written only where the model has no setting of its own yet, so a value set by hand is kept. Disable to manage capabilities manually.",
    )
    DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS: bool = Field(
        default=True,
        description=(
            "Turn off Open WebUI's built-in tools for models that produce images or video. "
            "A model that answers with a picture or a clip and no text cannot make a tool "
            "call, and offering it web search, code execution and the rest tends to make a "
            "turn fail or come back empty. With this on, each such model arrives in "
            "your workspace with 'Built-in tools' already unticked, so you can see the "
            "setting rather than wonder why tools are quiet. Tick it back on for any single "
            "model and your choice stays put; the pipe only sets it the first time it adds "
            "the model. A model that also answers with text is a chat model that can draw, "
            "and the box is left where you have it - a router such as openrouter/auto, which "
            "publishes a picture among its possible outputs, or a fixed model such as "
            "google/gemini-2.5-flash-image. The exception is the one model that cannot call "
            "tools at all: if its own catalog row names neither 'tools' nor 'tool_choice', "
            "as gemini's does, 'Built-in tools' is unticked on it anyway, because there is "
            "nothing there to call. 'File context' follows the same split, and answers in "
            "text alone is enough to keep it: neither box is cleared on such a model going "
            "forward. A model that synced before this change already has both boxes "
            "cleared, and the sync only fills a box that is still empty, so tick them back "
            "by hand if you want attachments and built-in tools on it. "
            "Requires model capability syncing to be enabled."
        ),
    )
    UPDATE_MODEL_DESCRIPTIONS: bool = Field(
        default=False,
        description=(
            "When enabled, automatically sync model descriptions from OpenRouter's frontend catalog to Open WebUI model metadata. "
            "Disable to manage model descriptions manually (or set per-model disable_description_updates)."
        ),
    )
    ENABLE_WEB_SEARCH: bool = Field(
        default=True,
        description="Enable the OpenRouter Web Search server tool. When disabled, web search toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it; every Web Tools filter this pipe maintains (one it installed, or one carrying no install record) that still offers it is rewritten without it, whether it is on or off, at the next model-list refresh, or after the first message that still asks for it. On a Fusion model the panel is never attached, so there is no per-chat switch there: the setting stored for you governs what the internal Fusion panel may use.",
    )
    ENABLE_WEB_FETCH: bool = Field(
        default=True,
        description="Enable the OpenRouter Web Fetch server tool. When disabled, web fetch toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it; every Web Tools filter this pipe maintains (one it installed, or one carrying no install record) that still offers it is rewritten without it, whether it is on or off, at the next model-list refresh, or after the first message that still asks for it.",
    )
    ENABLE_DATETIME: bool = Field(
        default=True,
        description="Enable the OpenRouter Datetime server tool (free, no additional cost). When disabled, datetime toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it; every Web Tools filter this pipe maintains (one it installed, or one carrying no install record) that still offers it is rewritten without it, whether it is on or off, at the next model-list refresh, or after the first message that still asks for it.",
    )
    ENABLE_ADVISOR: bool = Field(
        default=True,
        description="Enable the OpenRouter Advisor server tool (consult a higher-intelligence model mid-generation). When disabled, advisor toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it; every Web Tools filter this pipe maintains (one it installed, or one carrying no install record) that still offers it is rewritten without it, whether it is on or off, at the next model-list refresh, or after the first message that still asks for it.",
    )
    ENABLE_SUBAGENT: bool = Field(
        default=True,
        description="Enable the OpenRouter Subagent server tool (delegate tasks to a worker model an admin chooses). When disabled, subagent toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it; every Web Tools filter this pipe maintains (one it installed, or one carrying no install record) that still offers it is rewritten without it, whether it is on or off, at the next model-list refresh, or after the first message that still asks for it.",
    )
    ENABLE_SEARCH_MODELS: bool = Field(
        default=True,
        description="Enable the OpenRouter model-search server tool (let the model search the OpenRouter catalog). When disabled, model-search toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it; every Web Tools filter this pipe maintains (one it installed, or one carrying no install record) that still offers it is rewritten without it, whether it is on or off, at the next model-list refresh, or after the first message that still asks for it.",
    )
    ENABLE_IMAGE_GENERATION: bool = Field(
        default=True,
        description="Enable the OpenRouter Image Generation server tool. When disabled, image generation toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it.",
    )
    ENABLE_VIDEO_GENERATION: bool = Field(
        default=True,
        description=(
            "Add OpenRouter's video-generation models, which render in the background, to the model list. "
            "Video models are judged by OpenRouter's ZDR list like any other model, so a ZDR-only picker "
            "excludes them unless OpenRouter lists a ZDR endpoint for them."
            + _PIPE_OFF_COMES_BACK
            + " Turning it off deactivates all installed per-model video filter rows on the next model-list refresh;"
            + " AUTO_INSTALL_VIDEO_FILTERS is the install valve for that family. Turning it back on re-activates"
            + " the ones still in the catalogue that the pipe itself switched off, while that valve is on;"
            + " with that valve off its retirement governs, and the rows stay off until it comes back on."
            + " The rows are identified by their source, so a copy you made by hand of one of these"
            + " filters' source is switched off too."
        ),
    )

    AUTO_INSTALL_WEB_TOOLS_FILTER: bool = Field(
        default=True,
        description=(
            "Automatically install/update the OpenRouter Web Tools filter function in Open WebUI. When off, the pipe neither installs nor updates it, except that a web tool switched off on the pipe is taken out of every Web Tools filter this pipe maintains (one it installed, or one carrying no install record): that filter's code is replaced with the pipe's current version for the tools it still offers (hand edits in it are lost), and a warning is logged. Switching the tool back on does not add it back to that stored code, which is rewritten only while the tool is off. With it on — the default — the next model-list refresh rewrites the row this pipe maintains from the current valve set, so the tool comes back there; another copy the repair touched keeps it out until an admin switches it on in the Functions list."
            + " With every web tool off, every Web Tools filter is switched off that this "
            "pipe installed or that carries no install record, and one you switch off "
            "yourself there stays off until you switch it on again. The row's on/off "
            "state is the other half of the same switch and moves on its own: with this "
            "valve off as well, a row the pipe switched off is switched back on as soon "
            "as any web tool is on again, whether the pipe installed that row or you did."
            " A filter the pipe cannot read is reported as unrepaired, and the pass is "
            "retried once per five-minute window until the row is updated or removed."
            + _PIPE_OFF_COMES_BACK
            + _ADMIN_OFF_STAYS_OFF
        ),
    )
    AUTO_ATTACH_WEB_TOOLS_FILTER: bool = Field(
        default=True,
        description="Automatically attach the OpenRouter Web Tools per-chat switch to every pipe model that is not a picture-only model, a video-generation or a Fusion model (a model that answers with text as well as pictures is a chat model that can draw, and it keeps the switch, so the toggle appears in the Integrations menu). Turning this off detaches the filters the pipe attached, and also releases the default the pipe seeded for them; a filter id an admin attached by hand is left alone, and so is a Default Filter ticked by hand in the model editor, which this valve never seeds (seeding a default is the separate AUTO_DEFAULT_WEB_TOOLS_FILTER setting). A pass that cannot install the filter, because Open WebUI refused the write, is not a decision to detach: the switch and the default it already carried stay exactly where they are, and the install is tried again at the next catalog fetch. This relies on the ownership record the pipe writes, which it keeps up to date on every pass where the pipe attaches it, so a record that has drifted is repaired on the next sync. A model carrying the panel with no record -- one attached before this build wrote records -- is not given one: detach the panel by hand once and the next sync records the current one.",
    )
    AUTO_DEFAULT_WEB_TOOLS_FILTER: bool = Field(
        default=False,
        description="When enabled, marks the OpenRouter Web Tools filter as a Default Filter on every pipe model that is not a picture-only model, a video-generation or a Fusion model (a model that answers with text as well as pictures is a chat model that can draw, and it keeps the switch, pre-enabled per chat; users can still turn it off). Turning it off removes the already-seeded default from models on the next sync, and so does switching every Web Tool off, and turning it back on reclaims a default the operator re-ticked in between, so the next turn-off still removes it. A default seeded under an id the panel no longer has is released too, even after the panel has been reinstalled under a new id, and a default seeded by a build older than the durable seed record, whose row carries only the attach record, is left in place and the admin removes it in the model editor. A default an installer hiccup left in place is not one of them: a blank filter id is a lookup that failed, not a decision to release, so a seeded default survives it, and with the valve on it is never looked for.",
    )

    AUTO_INSTALL_IMAGE_GEN_FILTER: bool = Field(
        default=True,
        description=(
            "Automatically install/update the OpenRouter Image Generation filter function "
            "in Open WebUI."
            + _PIPE_OFF_COMES_BACK
            + _ADMIN_OFF_STAYS_OFF
            + " The six controls on that tool's panel are built from the same "
            "published-contract sweep as the per-model image panels, so either half of "
            "this pair, or either half of AUTO_INSTALL_IMAGE_FILTERS / "
            "AUTO_ATTACH_IMAGE_FILTERS, pays for that read; turning the per-model pair off "
            "does not turn this one off, and with all four off the contracts are not read "
            "at all."
        ),
    )
    AUTO_ATTACH_IMAGE_GEN_FILTER: bool = Field(
        default=True,
        description="Automatically attach the OpenRouter Image Generation filter to every pipe model that can send the tool: not a model whose catalogue entry rules tool use out, not a picture-only model, not a video model, not the hosted Fusion model. A model that stops qualifying loses the switch at the next refresh, and the id the pipe had recorded for it is released from `filterIds` on that same refresh. Turning this off detaches the filters the pipe attached; a filter id an admin attached by hand is left alone. A pass that cannot install the filter, because Open WebUI refused the write, is not a decision to detach: the switch stays exactly where it was and the install is tried again at the next catalog fetch. On its own, with the per-model image filter pair off, this valve also buys the published-contract sweep the tool's six controls are drawn from and checked against, so the choices on them are the drawing model's own.",
    )
    ENABLE_OPENROUTER_IMAGE_GENERATION: bool = Field(
        default=True,
        description=(
            "Expose OpenRouter's image-generation models (Sourceful, Flux, Seedream and "
            "the rest) as models you can pick in chat. Models that produce both text and "
            "images stay where they already are in the chat list; this only adds the "
            "image-only ones. Turning it off also withdraws the published contracts of the "
            "models it drops; a text+image chat model's own contract is untouched, so its "
            "image_config vetting continues."
        ),
    )
    AUTO_INSTALL_IMAGE_FILTERS: bool = Field(
        default=True,
        description=(
            "Install and keep up to date one settings panel per image model, built from "
            "the settings that model tells OpenRouter it accepts -- so nobody is shown an "
            "aspect ratio their model rejects. A routing variant or preset of an image "
            "model is the same model, so its panel covers the variant too. Alongside those, every panel carries "
            "Output size: a size tier typed there is checked against the tiers that model "
            "publishes, or against 512, 1K, 2K and 4K where it publishes none, and a tier "
            "the model does not list is dropped before the request goes out, while exact "
            "pixels such as 1024x1024 travel as typed. A model that answers with a picture "
            "and no text carries three more the panel supplies rather than the model: "
            "Provider options, Reference images and Reference image links. If a model's "
            "settings list cannot be read on a refresh, it keeps the settings from the "
            "last successful read; a model never read gets no panel at all rather than a "
            "guessed set. If a whole refresh cannot install the panels or the Fusion "
            "panel at all, every model keeps the panels it already had and the pass is "
            "tried again at the next catalog fetch. This is not the only valve that reads "
            "those settings: the Image Generation tool's own panel is built from the same "
            "sweep, so either half of AUTO_INSTALL_IMAGE_GEN_FILTER / "
            "AUTO_ATTACH_IMAGE_GEN_FILTER reads them too, and with all four filter valves "
            "off nothing reads them at all."
            + _ADMIN_OFF_STAYS_OFF
        ),
    )
    AUTO_ATTACH_IMAGE_FILTERS: bool = Field(
        default=True,
        description=(
            "Attach each image model's own settings panel to it, so the settings appear "
            "in the chat controls when that model is selected. Turn this off to install "
            "the panels but leave attaching them to you; turning it off also detaches the "
            "panels the pipe attached, while a panel an admin attached by hand is left alone."
            " Detaching is this valve's doing and the model's loss of image support, not a "
            "panel whose install failed once: a model whose image panel could not be "
            "written keeps the panel it already carries and the default it already had, "
            "and takes them off at the next refresh that installs it, or at the next one "
            "that finds it no longer offers images. A pass that cannot find the panel it "
            "was told to attach leaves the existing one in place and tries again at the "
            "next catalog fetch."
        ),
    )
    AUTO_DEFAULT_IMAGE_FILTERS: bool = Field(
        default=True,
        description=(
            "Keep the attached image filters enabled by default on "
            "image-output models. Reapplied on every catalogue or settings change; turning it off "
            "clears the default the pipe seeded, leaving the filter attached. A pass that "
            "cannot find the panel it was told to attach leaves the existing one in place "
            "and tries again at the next catalog fetch."
        ),
    )

    AUTO_INSTALL_VIDEO_FILTERS: bool = Field(
        default=True,
        description=(
            "Automatically install/update the OpenRouter Video Generation companion filter "
            "function in Open WebUI. A model whose catalogue entry publishes no video "
            "contract is left as it is: any filter it already has is kept, and none is "
            "installed for it, and the same holds for a model whose install this pass could "
            "not write. With this valve off the pipe logs an installed row whose "
            "stored source is out of date but will not rewrite it, so fixes to that filter "
            "stay undelivered until this is on."
            + _PIPE_OFF_COMES_BACK
            + _ADMIN_OFF_STAYS_OFF
        ),
    )
    AUTO_ATTACH_VIDEO_FILTERS: bool = Field(
        default=True,
        description="Automatically attach the OpenRouter Video Generation filter to OpenRouter video-generation models. Turning this off detaches the filters the pipe attached; a filter id an admin attached by hand is left alone. A pass that cannot find the panel it was told to attach leaves the existing one in place and tries again at the next catalog fetch.",
    )
    AUTO_DEFAULT_VIDEO_FILTERS: bool = Field(
        default=True,
        description="Keep the per-model video filter enabled by default on its video model. Reapplied on every catalogue or settings change; turning it off clears the default the pipe seeded, leaving the filter attached. A pass that cannot find the panel it was told to attach leaves the existing one in place and tries again at the next catalog fetch. Models that require a per-model parameter (e.g. Veo's personGeneration) cannot be driven without it; parameter-free models still generate.",
    )
    ENABLE_OPENROUTER_FUSION: bool = Field(
        default=True,
        description=(
            "Master switch for OpenRouter Fusion support. When enabled, the pipe installs the 'OpenRouter Fusion' filter and attaches it to the fusion models automatically."
            + _PIPE_OFF_COMES_BACK
            + " Turning it off deactivates the installed filter on the next model-list refresh,"
            + " and a `{\"id\": \"fusion\"}` plugin entry the request already carries is removed"
            + " before anything is sent - on any model and either engine - so an entry a filter"
            + " row, a saved chat or a direct API caller brought along cannot deliberate;"
            + " a task or title request never carries one, whatever the valve says, so the off"
            + " state does not wait for that refresh;"
            + " AUTO_INSTALL_FUSION_FILTER is the install valve for that family. Turning it back on re-activates a"
            + " filter the pipe itself switched off whose family's install valve is still on; a row that valve has"
            + " retired stays off until that valve comes back on."
        ),
    )
    AUTO_INSTALL_FUSION_FILTER: bool = Field(
        default=True,
        description=(
            "Automatically install/update the OpenRouter Fusion filter function in Open WebUI."
            + _PIPE_OFF_COMES_BACK
            + _ADMIN_OFF_STAYS_OFF
            + " Its fixes live in the stored row rather than in the pipe, so an installed"
            " deployment receives them only while this valve is on."
        ),
    )
    AUTO_ATTACH_FUSION_FILTER: bool = Field(
        default=True,
        description="Automatically attach the OpenRouter Fusion filter to the fusion models only — `openrouter/fusion`, `openrouter/fusion-flash`, and their `:tag` variant and `@preset/…` rows (so their panel/judge options appear in the Integrations menu). Never attaches to any other model. Turning this off detaches the filters the pipe attached; a filter id an admin attached by hand is left alone. Auto-install off with this on — the install-by-hand mode — no longer detaches on a pass that finds no panel: the attached filter stays and the next catalog fetch tries again. Neither does a pass that could not install the panel because Open WebUI refused the write: the attached filter and the default it carried stay, and the install is tried again at the next catalog fetch.",
    )
    AUTO_DEFAULT_FUSION_FILTER: bool = Field(
        default=True,
        description="Mark the OpenRouter Fusion filter as a Default Filter on the fusion models (pre-enabled per chat) — including their `:tag` variant and `@preset/…` rows. Does NOT force Fusion to run — the per-user 'Always run Fusion' toggle is off by default. Reapplied on every catalogue or settings change; turning it off clears the default the pipe seeded, leaving the filter attached. A pass that cannot find the filter it was told to attach leaves the existing one in place and tries again at the next catalog fetch.",
    )
    FUSION_BACKEND: Literal["openrouter", "internal"] = Field(
        default="internal",
        title="Fusion Backend",
        description="Which engine runs deliberation for the dedicated fusion models. 'openrouter': requests go to OpenRouter's hosted Fusion — the panel and judge execute on their servers. 'internal': the pipe runs the same panel → judge → synthesis flow itself as ordinary pipe model calls, so panel members can use every Open WebUI tool that user has (knowledge bases, tool servers, pipe server tools), every dial (ZDR, reasoning effort, cost attribution) applies per member, and one member failing does not kill the whole stream. The live panel UI is identical on both backends.",
    )
    FUSION_PANEL_SYSTEM_PROMPT: str = Field(
        default=DEFAULT_FUSION_PANEL_SYSTEM_PROMPT,
        title="Fusion Panel System Prompt",
        description="System prompt sent to every panel member on the internal fusion backend. It enforces independent, committed, source-cited answers and forbids members from revealing the deliberation machinery. Edit to reshape panel behaviour; the shipped default was produced by a multi-model design tournament and is tuned to pair with the judge and synthesis prompts. Clearing the box, or leaving only whitespace in it, restores the shipped default.",
    )
    FUSION_JUDGE_SYSTEM_PROMPT: str = Field(
        default=DEFAULT_FUSION_JUDGE_SYSTEM_PROMPT,
        title="Fusion Judge System Prompt",
        description="System prompt for the internal fusion judge (runs at temperature 0). CAUTION: the judge must emit a single strict JSON object with exactly the five keys consensus / contradictions / partial_coverage / unique_insights / blind_spots — the live Analysis panel and the synthesis stage are depend on exactly that JSON. Keep the output-format rules intact when editing; if the judge stops producing valid JSON the run continues without the analysis. Clearing the box, or leaving only whitespace in it, restores the shipped default.",
    )
    FUSION_SYNTHESIS_SYSTEM_PROMPT: str = Field(
        default=DEFAULT_FUSION_SYNTHESIS_SYSTEM_PROMPT,
        title="Fusion Synthesis System Prompt",
        description="System prompt for the internal fusion synthesis call — the model that receives the panel drafts plus the judge's analysis and writes the final user-facing answer. It enforces 'compose, never average', honest handling of contradictions, and total secrecy about the deliberation machinery. Edit to change the final answer's voice or composition rules. Clearing the box, or leaving only whitespace in it, restores the shipped default.",
    )
    VIDEO_INITIAL_POLL_DELAY_SECONDS: float = Field(
        default=5.0,
        ge=0.0,
        le=60.0,
        description="Initial delay before polling a newly submitted OpenRouter video generation job.",
    )
    VIDEO_POLL_INTERVAL_SECONDS: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Base polling interval for OpenRouter video generation jobs.",
    )
    VIDEO_POLL_BACKOFF_FACTOR: float = Field(
        default=1.2,
        ge=1.0,
        le=4.0,
        description="Backoff multiplier applied after each status check that reports the job is still running.",
    )
    VIDEO_POLL_INTERVAL_MAX_SECONDS: float = Field(
        default=20.0,
        ge=1.0,
        le=120.0,
        description="Maximum interval between video generation status polls.",
    )
    VIDEO_MAX_POLL_TIME_SECONDS: int = Field(
        default=1800,
        ge=30,
        le=7200,
        description=(
            "How long a video generation job may go without a word from OpenRouter before "
            "this stops watching it. The clock restarts every time a status check comes "
            "back saying the job is still running, so a slow render is never cut off for "
            "taking a long time \u2014 only one that has gone quiet is. This is one of two "
            "limits that do the same thing: a run of unanswered status checks stops the "
            "watching too, without cancelling the job. When either one runs out the chat "
            "keeps the job's card and says the render is still going at "
            "OpenRouter: nothing is cancelled and OpenRouter still bills the job, and the "
            "person can press Continue Response on that message to pick it back up. The "
            "wait is never allowed to be shorter than a single status check can take, so "
            "anything below `Maximum poll interval` plus the HTTP read timeout is raised "
            "to that. A stored value this release no longer accepts is left at this "
            "valve's default and named in the log."
        ),
    )
    VIDEO_STATUS_POLL_MAX_ERRORS: int = Field(
        default=5,
        ge=1,
        le=25,
        description=(
            "How many status checks in a row may come back as an error before this stops "
            "watching the job. A status endpoint that is not answering says nothing about "
            "the job itself, so the card is kept as still running and resumable rather "
            "than written off as a failure: nothing is cancelled, the render is still "
            "going at OpenRouter, and the person can press Continue Response on that "
            "message to pick it back up. A later check that reports the job as failed or "
            "expired ends it for real, as it would have anyway."
        ),
    )
    SEND_MEDIA_VIA_FILE_HOST: bool = Field(
        default=False,
        description=(
            "Put an attached clip or sound file behind a public link so the model can fetch "
            "it. OpenRouter takes reference media only as a link it can download, so without "
            "this an attachment cannot reach a video model at all. Turning it on uploads the "
            "file to the third-party host chosen below, where anyone holding the link can "
            "watch it: on litterbox until it expires, on catbox for good. Only a file the "
            "person asking uploaded themselves is sent this way, never one they can merely "
            "open because it was shared with them. Off unless you decide otherwise."
        ),
    )
    MEDIA_FILE_HOST: Literal["litterbox", "catbox"] = Field(
        default="litterbox",
        description=(
            "Which file host receives the upload. litterbox deletes the file by itself after "
            "the time set below. catbox keeps it for good: the upload carries no account, so "
            "nobody here can take a file down again once it is up. Choose catbox only when a "
            "link has to outlive the job."
        ),
    )
    MEDIA_FILE_HOST_RETENTION: Literal["1h", "12h", "24h", "72h"] = Field(
        default="1h",
        description=(
            "How long litterbox keeps the file before deleting it. The model fetches it within "
            "seconds of the request, so an hour is ample; raise it only if your provider queues "
            "jobs for longer. catbox ignores this and keeps everything."
        ),
    )
    USE_THE_OTHER_FILE_HOST_IF_ONE_IS_DOWN: bool = Field(
        default=False,
        description=(
            "If the chosen host cannot take the file, try the other one instead of failing "
            "the request. Off by default because the two keep files for very different "
            "lengths of time: falling back from litterbox to catbox turns a file that would "
            "have deleted itself within the hour into one that stays up for good, with no "
            "way for anyone here to remove it. Turn it on only if you would rather the "
            "request succeed. This valve decides only whether a second copy is ever made: "
            "when a host's answer leaves it possible that it stored the file anyway, the "
            "user is told that a copy may already be there whether or not this is on. "
            "Retention is given per host: when a second copy does land on the other one, "
            "the notice names that host with its own retention, so a copy on the "
            "self-deleting host is not described as permanent. "
            "With the fallback taken, the correction naming the host that keeps the file "
            "is a precondition too, so while TELL_USERS_ABOUT_THE_FILE_HOST is on a chat "
            "that will not accept it gets the request failed and an error naming the host "
            "the file was on its way to. The 300s upload budget is shared across the turn's "
            "attachments rather than given to the first one, so a slow first upload does "
            "not spend the time the ones behind it were going to get."
        ),
    )
    MEDIA_FILE_HOST_MAX_SIZE_MB: int = Field(
        default=200,
        ge=1,
        le=1024,
        description=(
            "Largest attachment that will be uploaded to the file host, and the most one "
            "request may upload in total. A file over this, or a set of attachments coming "
            "to more than this, is refused and the request stops rather than generating "
            "without it."
        ),
    )
    SEND_VIDEO_VIA_FILE_HOST: bool = Field(
        default=True,
        description=(
            "Include attached video clips when the file host is in use. OpenRouter refuses a "
            "clip sent any other way, so a video attachment needs this to reach the model."
        ),
    )
    SEND_AUDIO_VIA_FILE_HOST: bool = Field(
        default=True,
        description=(
            "Include attached sound files when the file host is in use. OpenRouter refuses "
            "audio sent any other way. It also only accepts a sound reference alongside a "
            "picture or a clip, never on its own."
        ),
    )
    SEND_IMAGES_VIA_FILE_HOST: bool = Field(
        default=False,
        description=(
            "Include attached pictures as well. They do not need it: a picture travels inside "
            "the request already and never leaves this server. The per-picture cap in "
            "`Maximum single frame size` holds whichever way the picture travels, and an "
            "oversize one is left out of the request with a note in the chat naming it. What "
            "this adds on top is the host's own `Largest attachment to upload` ceiling, so a "
            "picture is the one reference kind you can hand over as a link."
        ),
    )
    TELL_USERS_ABOUT_THE_FILE_HOST: bool = Field(
        default=True,
        description=(
            "Warn in the chat before a user's attachment is uploaded to the file host. Their "
            "own media leaves this server, so they are warned by default -- and while this is "
            "on, an upload that could not be announced does not happen: the request fails "
            "instead of publishing the file unannounced. That holds for every publication, "
            "a fallback to a second host included: the second host is announced before it "
            "receives anything, and a chat that refuses that one is refused the same way. "
            "That includes a warning with no "
            "words to give, and one that could not be rendered, so an empty notice below "
            "stops the upload too. Switch it off if "
            "you have told your users another way. Either way the finished message keeps a "
            "written record of what was uploaded and where, and where a host's answer left "
            "it possible that it stored the file anyway, that record says the file may have "
            "been uploaded rather than that it was; only this advance warning is optional."
        ),
    )
    FILE_HOST_NOTICE: str = Field(
        default=(
            "Sending your {kind} to {host} so the model can read it, {retention}."
        ),
        description=(
            "The wording of that advance warning. Rewrite it in your own words or your own "
            "language. {kind} becomes clip, sound file or picture; {host} names the file "
            "host; {retention} says how long each host named keeps the file, and where two "
            "of them keep files for different lengths of time it names both. Leave out any "
            "you do not want, "
            "but keep enough that a sentence is left: an empty or blank setting leaves the "
            "warning nothing to say, and so does a placeholder nothing can fill -- one "
            "misspelled, one numbered, or one carrying a formatting flag. The wording you "
            "typed is shown to you here as written and the placeholder is named in the "
            "server log, and it stops the upload, or the request, the same way an empty "
            "one does. While the setting above is on, a warning that "
            "cannot be said stops the upload, or the request, instead -- turn "
            "TELL_USERS_ABOUT_THE_FILE_HOST off if you want no warning at all. "
            "{kind} and {retention} are written in English, so if you are writing this in "
            "another language, say those parts yourself rather than using the placeholders. "
            "The record kept in the finished message is written separately and is not this "
            "template."
        ),
    )
    REMOTE_VIDEO_MAX_SIZE_MB: int = Field(
        default=500,
        ge=1,
        le=2048,
        description="Maximum downloaded generated video size in MB before the lifecycle is failed.",
    )
    VIDEO_DOWNLOAD_CHUNK_SIZE: int = Field(
        default=1024 * 1024,
        ge=64 * 1024,
        le=8 * 1024 * 1024,
        description="Chunk size in bytes used when downloading generated video content. A stored value this release no longer accepts is left at this valve's default and named in the log.",
    )
    MAX_CONCURRENT_VIDEO_GENS: int = Field(
        default=2,
        ge=1,
        le=100,
        description="Maximum number of video generation jobs running per pipe process. A permit is taken before the turn does any media work, so frame extraction, frame encoding and the file-host relay all run inside the held window, not only submission and polling. Takes effect without a restart, at the next generation: a lower value binds from that moment and the jobs already running finish first.",
    )
    MAX_CONCURRENT_VIDEO_GENS_PER_USER: int = Field(
        default=2,
        ge=1,
        le=25,
        description="Maximum number of video generation jobs running per user per pipe process. The check happens before any frame is extracted or reference relayed, so a turn over the cap costs no media work. On a resumed turn the job found in the message is left running and pickable, and the refusal is written as a resumable card rather than a failure.",
    )
    VIDEO_FRAME_IMAGE_MAX_BYTES: int = Field(
        default=12 * 1024 * 1024,
        ge=64 * 1024,
        le=64 * 1024 * 1024,
        description="Maximum decoded size for a single image frame passed to OpenRouter video generation. A frame the pipe itself extracts from a previous video is bounded to 1920 on its long edge before it is encoded, so at this default such a frame never meets this cap; an attached frame still can, and still fails the whole request over it.",
    )
    VIDEO_FRAME_TOTAL_MAX_BYTES: int = Field(
        default=50 * 1024 * 1024,
        ge=64 * 1024,
        le=128 * 1024 * 1024,
        description="Maximum combined decoded size for all image frames passed to one video generation request.",
    )
    VIDEO_FRAME_IMAGE_MIME_ALLOWLIST: Annotated[
        str, AfterValidator(_check_frame_allowlist)
    ] = Field(
        default="image/jpeg,image/png,image/webp",
        description="Comma-separated MIME allowlist for video generation frame images. Frames the pipe itself extracts from a prior video are re-encoded to a type on this list before upload, so dropping image/png no longer breaks frame reuse. Name at least one of `image/jpeg`, `image/png` or `image/webp`: a list naming none of those three cannot be saved, because there would be no type left for a frame the pipe extracted to be converted into. An empty value is accepted and permits no type at all, so every frame image is refused; a value already stored that names none of the three is left empty on load, which refuses every frame rather than turning frames back on, and the log names the valve and the value it replaced. Either way a frame the pipe extracted is left out with a note in the chat and the video still renders, while an attached frame of a type that is not listed still fails the whole request.",
    )
    VIDEO_OUTPUT_MIME_ALLOWLIST: str = Field(
        default="video/mp4,video/webm",
        description="Comma-separated MIME allowlist for generated video downloads, applied to the declared type or, where that is not listed, to the format identified from the file's leading bytes.",
    )
    VIDEO_INTENT_ENABLED: bool = Field(
        default=True,
        description=(
            "Master switch for the video intent classifier. When False, video "
            "requests bypass the classifier entirely and only the latest user "
            "message is sent to the video model — no cross-turn context, no "
            "clarifying questions, no automatic frame reuse from prior videos."
        ),
    )
    VIDEO_INTENT_TASK_MODEL_MODE: Literal["internal", "external"] = Field(
        default="external",
        description=(
            "Which of Open WebUI's two Task Models, as configured in Open WebUI's "
            "admin Task Model settings, to use as the intent classifier. 'internal' "
            "uses the one for local models, 'external' the one for API models."
        ),
    )
    VIDEO_INTENT_TASK_MODEL_FALLBACK: Literal["none", "other_task_model"] = Field(
        default="other_task_model",
        description=(
            "Fallback strategy when the chosen task model fails. 'none' disables "
            "fallback; 'other_task_model' switches between internal/external. If "
            "neither task model is configured at all, the classifier is skipped for "
            "the turn and the turn is recorded as a classifier failure with reason "
            "'no_task_model_candidates' -- the video still generates, without "
            "cross-turn intent analysis, and the person in the chat is warned once "
            "per chat."
        ),
    )
    VIDEO_INTENT_SKIP_WHEN_EMPTY_CHAT: bool = Field(
        default=True,
        description=(
            "When True, skip the classifier on the very first turn of a fresh "
            "chat that has no attachments. There is nothing for the classifier "
            "to reference in that case (no prior video, no attached media), so "
            "the call is wasted task-model spend. Turn off only if you want "
            "the classifier to ask a clarifying question on opening prompts "
            "like 'make it red' that have no context."
        ),
    )
    VIDEO_INTENT_MAX_CLARIFICATIONS: int = Field(
        default=1,
        ge=0,
        le=3,
        description=(
            "Per-chat cap on clarifying questions. 0 disables the "
            "clarification loop entirely (always proceeds with best-guess interpretation); "
            "at the administrator's 0 that wins over a saved per-model value, so the "
            "chat asks nothing however high the per-model setting is. "
            "Default 1 = at most one question in the whole chat, after which it proceeds "
            "with its best guess on every later ambiguous request."
        ),
    )
    VIDEO_INTENT_FRAME_EXTRACTION_INDEX: Literal["first", "last"] = Field(
        default="last",
        description=(
            "When a requested moment in a prior video runs past the end the pipe can "
            "measure, which frame - 'first' or 'last' - is substituted for it. It also "
            "decides the frame when the seek to an in-range moment comes back empty, "
            "and the frame the pixel cap makes the pipe substitute for a refused one: "
            "that substitute is a smaller copy of the frame the cap would not decode, "
            "and the disclosure footer says so. On a model accepting only a first frame "
            "the first frame is substituted there whatever this setting says, and the "
            "disclosure names it. "
            "A request that names a first or last frame directly gets that frame - "
            "except on a clip, damaged or not, whose length the host cannot measure, "
            "where no end-seek hop reads a frame at all and the file's first frame is "
            "substituted for the one asked for, which the disclosure footer says. "
            "A frame read out of an earlier video is bounded to 1920 on its long edge "
            "whatever size was asked for, so a clip generated past that on its long edge "
            "is reduced before it is sent, and nothing is disclosed for it. "
            "'last' matches 'continue this scene' intent. On a model that accepts only a "
            "first frame, a moment the earlier video has is sent as asked and nothing is "
            "substituted, so the disclosure footer stays silent; a request for the final "
            "frame of the earlier video, or for a moment it does not have, is answered "
            "with its opening frame, and the footer says so."
        ),
    )
    VIDEO_INTENT_TIMEOUT_S: int = Field(
        default=8,
        ge=1,
        le=60,
        description=(
            "Hard timeout (seconds) for the classifier task-model call. The window "
            "covers the whole classifier, all candidates, so a fallback task model does "
            "not double the wait; each candidate still to run keeps a quarter of the "
            "window reserved for it, so at a small value a slow fallback may still not "
            "fit. The limit covers the model call only, and the lookup of those Task "
            "Model settings happens before it and is a database read, so it is not "
            "counted against this window. The calls themselves are unchanged in number: "
            "a fallback candidate is a second call, and a candidate whose reply cannot "
            "be parsed is retried once more, but they all happen inside that one window. "
            "A retry is only issued while the window still leaves it a usable share: with "
            "a candidate still to run, the loop falls through to that candidate rather than "
            "spending a call it cannot finish, and the last candidate keeps its retry unless "
            "the window has closed. "
            "If the calls exceed this or fail for any reason, the pipe falls back to "
            "sending only the latest user message to the video model. The paid video "
            "generation request still proceeds — the classifier never blocks generation."
        ),
    )
    VIDEO_INTENT_CONFIRM_MODE: Literal["always", "on_reference", "low_confidence", "never"] = Field(
        default="on_reference",
        description=(
            "When to show the confirmation footer under a generated video. 'always' = "
            "every generation; 'on_reference' (default) = only when a prior video's frame is reused "
            "or more than one frame is combined; a lone attached image on its own does not "
            "trigger it; 'low_confidence' = only when classifier confidence is low; "
            "'never' = no confirmation. A rewritten prompt, or a best-guess turn once "
            "the clarifying question limit is reached, shows the block in every mode "
            "except 'never'. A turn where the classifier changed what is sent is "
            "confirmed in every mode, 'never' excepted. At the administrator's "
            "'never' that wins over a saved per-model value, so the block is never "
            "shown and no thumbnail is taken however the per-model setting is set."
        ),
    )
    VIDEO_INTENT_MAX_TURNS_PER_CHAT: int = Field(
        default=0,
        ge=0,
        description=(
            "Cost guard: maximum video turns per chat session that run the classifier. 0 "
            "(default) = unlimited. Admin sets a positive integer to enforce a per-chat "
            "ceiling. A turn is charged when it is admitted, so concurrent turns in one chat "
            "share the ceiling; a turn costs one billable task-model call per candidate it "
            "tries, so one to four at the shipped default. The tally is per worker process "
            "and in memory, keyed on a one-way SHA-256 prefix of the chat id rather than the "
            "id itself, so no chat id is held in it; a temporary chat is charged like any "
            "other. It keeps the most recent 300 chats, so a chat pushed out of that window "
            "by other chats in between starts a fresh budget. This ceiling needs a chat id: "
            "a request that carries none - an external API client that omits chat_id, "
            "which Open WebUI normalises to an empty string - is not charged it at all, "
            "so such a request is bounded only by the per-user-per-day ceiling."
        ),
    )
    VIDEO_INTENT_MAX_TURNS_PER_USER_DAY: int = Field(
        default=0,
        ge=0,
        description=(
            "Cost guard: maximum video turns per user per day that run the classifier. 0 "
            "(default) = unlimited. Admin sets a positive integer to enforce a "
            "per-user-per-day ceiling. A turn is charged when it is admitted, so concurrent "
            "turns by one user share the ceiling; a turn costs one billable task-model call "
            "per candidate it tries, so one to four at the shipped default. This ceiling "
            "is keyed on the user, not on the chat, so a request that carries no chat id "
            "is still charged it against its own user's bucket. A caller that reaches the "
            "pipe with no user id at all falls back to the literal string 'anonymous', so "
            "every such caller shares one daily budget."
        ),
    )
    VIDEO_INTENT_LOG_DECISIONS: bool = Field(
        default=False,
        description=(
            "When True, log the per-turn intent classification summary (intent, "
            "confidence, detected language, frame counts, latency, and fallback/"
            "failure flags, with the chat id hashed) at INFO instead of DEBUG. The "
            "record is always written; this only raises its log level. What it holds "
            "is derived classification metadata: intent, confidence, language, frame "
            "counts, latency, the fallback and failure flags, a hashed chat id, "
            "whether the classifier failed, and a failure_reason carrying a bounded "
            "fault code or the exception's class name - describing the last "
            "candidate's failure, so a later candidate's fault can displace an "
            "earlier one's, or the bare code 'no_task_model_candidates' on a turn "
            "where no candidate ran at all. It also raises the level of the cost-guard "
            "line a turn refused by VIDEO_INTENT_MAX_TURNS_PER_CHAT or "
            "VIDEO_INTENT_MAX_TURNS_PER_USER_DAY emits, which names the cap it reached "
            "and the cap's value. Neither the user's verbatim prompt nor the task model's "
            "own free-text reason appears at any level. Off by default to keep this "
            "prompt-derived metadata out of normal INFO logs."
        ),
    )
    AUTO_ATTACH_DIRECT_UPLOADS_FILTER: bool = Field(
        default=True,
        description=(
            "When enabled, automatically attaches the OpenRouter Direct Uploads toggleable filter to models that support "
            "at least one of OpenRouter direct file/audio/video inputs (so the switch appears in the Integrations menu only where it can work). Turning this off detaches the filters the pipe attached; a filter id an admin attached by hand is left alone. A pass that cannot install the filter, because Open WebUI refused the write, is not a decision to detach: the switch stays exactly where it was and the install is tried again at the next catalog fetch."
        ),
    )
    AUTO_INSTALL_DIRECT_UPLOADS_FILTER: bool = Field(
        default=True,
        description=(
            "When enabled, automatically installs/updates the companion OpenRouter Direct Uploads filter function in Open WebUI. "
            "This is required for AUTO_ATTACH_DIRECT_UPLOADS_FILTER when the filter hasn't been installed manually."
            + _ADMIN_OFF_STAYS_OFF
        ),
    )

    # Provider Routing Filters
    ADMIN_PROVIDER_ROUTING_MODELS: str = Field(
        default="",
        description=(
            "Comma-separated list of model slugs (e.g., 'openai/gpt-4o, anthropic/claude-3.5-sonnet') for which "
            "to generate admin-only provider routing filters. These filters enforce provider preferences (order, "
            "only, ignore, sort, quantizations, etc.) that users cannot override or disable. "
            "At most 50 of the listed models are fetched for provider data per sync, counted across this list and "
            "the user provider routing list together, so a long list here spends that budget first; a correctly "
            "spelled slug past the 50th keeps the provider data from the last cycle, its routing entry offers fewer "
            "providers, and the log names it. "
            "Leave empty to disable admin provider routing filters. This is a per-model list: "
            "clicking Global on one of its rows in Workspace > Functions would apply that model's "
            "preferences to every model, and the pipe reverts the click on the next model-list refresh."
            + _ROUTING_ADMIN_OFF_STAYS_OFF
            + _ROUTING_ADMIN_RE_ENABLED_STAYS_ON
        ),
    )
    USER_PROVIDER_ROUTING_MODELS: str = Field(
        default="",
        description=(
            "Comma-separated list of model slugs (e.g., 'meta-llama/llama-3.2-3b-instruct') for which "
            "to generate user-configurable provider routing filters. Users can toggle these filters per-chat "
            "and configure their own provider preferences in their per-user settings. "
            "At most 50 of the listed models are fetched for provider data per sync, counted across this list and "
            "the admin provider routing list together, so a long admin list spends that budget before this one is "
            "reached; a correctly spelled slug past the 50th keeps the provider data from the last cycle, its routing "
            "entry offers fewer providers, and the log names it. "
            "Leave empty to disable user provider routing filters. This is a per-model list: "
            "clicking Global on one of its rows in Workspace > Functions would apply that model's "
            "preferences to every model, and the pipe reverts the click on the next model-list refresh."
            + _ROUTING_ADMIN_OFF_STAYS_OFF
            + _ROUTING_ADMIN_RE_ENABLED_STAYS_ON
        ),
    )
    AUTO_DEFAULT_PROVIDER_ROUTING_FILTERS: bool = Field(
        default=True,
        description=(
            "Enable attached provider routing filters by default in new chats, so saved provider "
            "preferences apply without users having to switch the filter on per chat. The filter does "
            "nothing until preferences are actually configured, so defaulting it on is free. "
            "Disable to make users opt in per chat; on the next sync this also clears the default "
            "the pipe seeded, leaving the filters attached. A pass that could not read the filter "
            "table, or could not write a new routing filter, leaves the filters it already found "
            "in place and tries again at the next catalog fetch."
        ),
    )


class UserValves(BaseModel):
    """Per-user valve overrides."""

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_inherit(cls, values):
        """Treat the literal string 'inherit' (any case) as an unset value.

        """
        if not isinstance(values, dict):
            return values

        normalized: dict[str, Any] = {}
        for key, val in values.items():
            if isinstance(val, str):
                stripped = val.strip()
                lowered = stripped.lower()
                if lowered == "inherit":
                    continue
            normalized[key] = val
        return normalized

    SHOW_FINAL_USAGE_STATUS: bool = Field(
        default=True,
        title="Show usage details",
        description="Display tokens, time, and cost at the end of each reply; the cost appears only when it is above zero.",
    )
    THINKING_OUTPUT_MODE: Literal["open_webui", "status", "both"] = Field(
        default="open_webui",
        title="Thinking output",
        description=(
            "Choose where to show the model's thinking while it works: "
            "'open_webui' uses the Open WebUI reasoning box, "
            "'status' uses status messages, "
            "or 'both' shows both."
        ),
    )
    ENABLE_ANTHROPIC_INTERLEAVED_THINKING: bool = Field(
        default=True,
        title="Interleaved thinking (Claude)",
        description=(
            "When enabled, request Claude's interleaved thinking stream by sending "
            "`interleaved-thinking-2025-05-14` in the `x-anthropic-beta` header "
            "on Claude models that support it "
            "(including the `~anthropic/...` router aliases)."
        ),
    )
    REASONING_EFFORT: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = Field(
        default="medium",
        title="Reasoning depth",
        description="Choose how much thinking the AI should do before answering (higher depth is slower but more thorough). 'none' switches reasoning off where the model allows it; a model that always reasons gets the lightest level its catalog entry lists other than `none` instead, and no level at all when it lists no other level. A request that only hides the reasoning trace (`reasoning.exclude` of `true`) is not one of these offs: it keeps the depth you chose here and draws no status line. A request that carries its own verbosity keeps it on both endpoints, whichever of the two spellings it used. Use 'xhigh' for maximum depth when available.",
    )
    REASONING_SUMMARY_MODE: Literal["auto", "concise", "detailed", "disabled"] = Field(
        default="auto",
        title="Reasoning explanation detail",
        description="Pick how detailed the reasoning summary should be (auto, concise, detailed, or disabled).",
    )
    PERSIST_REASONING_TOKENS: Literal["disabled", "next_reply", "conversation"] = Field(
        default="next_reply",
        title="How long to keep reasoning",
        description="Choose whether reasoning is kept just for the next reply, the entire conversation, or not at all: 'disabled' writes nothing, withholds rows already stored from every later turn and deletes them on the same terms as 'next_reply', and does not save each reply's reasoning_details onto the stored chat message. The copy of the reasoning written onto the Open WebUI assistant message follows the same choice, and at 'next_reply' that copy is not removed when the following reply finishes; only the rows are. A setting the pipe cannot read (an undecodable stored row, after a rotation of `WEBUI_SECRET_KEY`, or the deprecated `WEBUI_JWT_SECRET_KEY` it falls back to (a default, so an empty primary is not a fallback)) falls back to this valve's own per-user default rather than to the administrator's site-wide value.",
    )
    PERSIST_TOOL_RESULTS: bool = Field(
        default=False,
        title="Remember tool and search results",
        description="Let the AI reuse outputs from tools (for example pages it fetched or other apps) later in the conversation, using more tokens on long chats. When off, the AI relies on its own summaries and can re-run tools as needed, but a question the AI's built-in ask_user tool asked you and your answer always go back to it. A round that arrived before the chat's first turn counts as earlier too. The turn the request ends in is never withheld: a round is replaced only when its turn is earlier than the final one, so a round whose assistant message opens the chat is kept while no later turn follows it. Each round is judged by its own call, so two rounds that happen to share one call id are kept or withheld separately. A temporary chat stores none of its tool results. Tool cards in the chat, while shown, still show every result. A setting the pipe cannot read (an undecodable stored row, after a rotation of `WEBUI_SECRET_KEY`, or the deprecated `WEBUI_JWT_SECRET_KEY` it falls back to (a default, so an empty primary is not a fallback)) falls back to this valve's own per-user default rather than to the administrator's site-wide value.",
    )
    TOOL_EXECUTION_MODE: Literal["Pipeline", "Open-WebUI"] = Field(
        default="Pipeline",
        title="Tool execution mode",
        description=(
            "Where to execute tools. 'Pipeline' executes tool calls inside this pipe. "
            "'Open-WebUI' hands a streamed reply's tool calls to Open WebUI to run instead, until the reply's hand-back "
            "budget of `MAX_FUNCTION_CALL_LOOPS` turns is spent; after that the pipe runs the remaining round itself "
            "with the tools it advertised. A round of browser-run tools and Open WebUI's own builtins is handed back "
            "whatever that budget says, because Open WebUI is what runs those, and its own iteration bound ends such a "
            "reply. That budget is one user's spend against one reply - two users naming the same chat and message are "
            "charged to two budgets - and it is charged in a Temporary Chat too, where only the reply's own budget is "
            "held and its chat id is not."
        ),
    )
    SHOW_TOOL_CARDS: bool = Field(
        default=True,
        title="Show tool execution cards",
        description="Show each tool the AI uses as a card in the chat. When off, no card appears, except for a file the AI shows through Open Terminal while Open WebUI is set to show terminal files inline; when it is not, a file the AI asks to show inline opens in the preview panel. The AI still remembers which tools it used, except in a temporary chat, where nothing is kept; after Stop, it remembers the calls before the first one still running if the reply was streamed, and none if it was not. Open WebUI draws its own cards for the calls it runs, which now means a streamed reply in Open-WebUI mode and calls approved under 'ask'. With tool cards on, a call the loop cut off at `MAX_FUNCTION_CALL_LOOPS` is shown as a card marked failed, so a round that never ran is visible rather than missing.",
    )
    REQUEST_ZDR: bool = Field(
        default=False,
        title="Request ZDR",
        description="Request Zero Data Retention routing for this chat.",
    )


def parse_user_valves(
    raw: Any, *, model: type[UserValves] = UserValves
) -> tuple[UserValves, list[str]]:
    """The one place `__user__["valves"]` becomes a UserValves. Never raises.

    Open WebUI hands this over as a MODEL INSTANCE, not a mapping -- `functions.py`
    assigns `params["__user__"]["valves"] = function_module.UserValves(**user_valves)`.
    A reader that gates on `isinstance(raw, dict)` therefore reads nothing in
    production while passing every test that hands it a dict.

    `model` is a parameter because `Pipe.UserValves` may be a plugin-extended SUBCLASS
    that `PluginRegistry` builds from `_pending_user_valve_fields`. This module sits below
    `pipe.py` and cannot see it, so hardcoding the base class here would silently drop
    any plugin-contributed field arriving as a mapping -- and report it as absent rather
    than rejected, which is the one distinction this function exists to preserve.

    Validation here is per-field rather than all-or-nothing. `model_validate` on the
    whole payload discards every setting the user has when any ONE of them is stale,
    which is how a REQUEST_ZDR=True got dropped -- and, at the pipe entry point, how a
    single unreadable field failed the request outright with a generic message. Fields
    that cannot be read are dropped and NAMED, so a caller that cares about a specific
    one can tell "the user did not set it" from "we could not read it" and decide for
    itself which way to fail.
    """
    if isinstance(raw, model):
        return raw, []
    if isinstance(raw, BaseModel):
        # exclude_unset: a bare model_dump() emits defaults too, and model_validate then
        # marks every one of them explicitly set. _merge_valves reads model_fields_set,
        # so one user-set field would override eleven admin valves.
        raw = raw.model_dump(exclude_unset=True)
    if not isinstance(raw, Mapping):
        return model(), []

    candidate = dict(raw)
    renamed = {
        name
        for name, value in raw.items()
        if value is not None and name not in model.model_fields
    }
    rejected: list[str] = []
    for _ in range(len(candidate) + 1):
        try:
            return model.model_validate(candidate), sorted(set(rejected) | renamed)
        except ValidationError as exc:
            bad = {
                str(err["loc"][0])
                for err in exc.errors()
                if err.get("loc") and str(err["loc"][0]) in candidate
            }
            if not bad:
                return model(), sorted(set(rejected) | set(candidate) | renamed)
            rejected.extend(sorted(bad))
            for name in bad:
                candidate.pop(name, None)
    return model(), sorted(set(rejected) | renamed)


def _select_openrouter_http_referer(valves: Any | None) -> str:
    """Select HTTP referer for OpenRouter requests, with optional valve override."""
    override = valves.HTTP_REFERER_OVERRIDE if valves else ""
    candidate = override.strip() if isinstance(override, str) else ""
    if candidate and is_http_or_https_url(candidate):
        return candidate
    return _OPENROUTER_REFERER


def openrouter_attribution_headers(valves: Any | str | None) -> dict[str, str]:
    if isinstance(valves, str):
        return {"HTTP-Referer": valves or _OPENROUTER_REFERER}
    return {"HTTP-Referer": _select_openrouter_http_referer(valves)}


OWUI_REQUEST: ContextVar[Any] = ContextVar("owui_request", default=None)
OWUI_CHAT_ID: ContextVar[str] = ContextVar("owui_chat_id", default="")


def _apply_owui_forward_user_headers(headers: dict, user: Any, chat_id: Any = None) -> dict:
    """Stamp Open WebUI user-identity headers (and Chat-Id) on an outbound request, matching a native OWUI connection; does nothing unless Open WebUI is present with ENABLE_FORWARD_USER_INFO_HEADERS set."""
    if _owui_env is None or _owui_include_user_info_headers is None:
        return headers
    if not getattr(_owui_env, "ENABLE_FORWARD_USER_INFO_HEADERS", False):
        return headers
    if user is None or isinstance(user, dict):
        return headers
    try:
        headers = _owui_include_user_info_headers(headers, user, request=OWUI_REQUEST.get())
        if chat_id:
            name = getattr(_owui_env, "FORWARD_SESSION_INFO_HEADER_CHAT_ID", "X-OpenWebUI-Chat-Id")
            headers[name] = str(chat_id)
    except Exception as exc:
        logger.log(
            warn_level(_warned_forward_headers, type(exc).__name__),
            "Could not attach Open WebUI session info headers; requests will be "
            "sent without them",
            exc_info=True,
        )
        return headers
    return headers


def _owui_forwarded_header_names() -> set[str]:
    """Lowercased names of the headers OWUI forwarding may emit, read from OWUI's own env config (its FORWARD_*_HEADER_* constants); used to redact them from debug logs."""
    names: set[str] = set()
    env = _owui_env
    if env is None:
        return names
    for attr in dir(env):
        if not attr.startswith("FORWARD_") or "_HEADER_" not in attr:
            continue
        if attr.endswith(("_SECRET", "_EXPIRES_SECONDS")):
            continue
        val = getattr(env, attr, None)
        if isinstance(val, str) and val.strip():
            names.add(val.strip().lower())
    return names
