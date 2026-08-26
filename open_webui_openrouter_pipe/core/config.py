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
import hashlib
import logging
import os
import re
from collections.abc import Mapping
from typing import Any, Literal, cast

from cryptography.fernet import Fernet, InvalidToken
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    GetCoreSchemaHandler,
    ValidationError,
    model_validator,
)
from pydantic_core import core_schema

from .fusion_defaults import (
    DEFAULT_FUSION_JUDGE_SYSTEM_PROMPT,
    DEFAULT_FUSION_PANEL_SYSTEM_PROMPT,
    DEFAULT_FUSION_SYNTHESIS_SYSTEM_PROMPT,
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


# Constants

_OPENROUTER_TITLE = "Open WebUI plugin for OpenRouter Responses API"
_OPENROUTER_CATEGORIES = "general-chat"
_OPENROUTER_REFERER = "https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/"
_DEFAULT_PIPE_ID = "open_webui_openrouter_pipe"
_FUNCTION_MODULE_PREFIX = "function_"
_OPENROUTER_FRONTEND_MODELS_URL = "https://openrouter.ai/api/frontend/v1/catalog/models"
_OPENROUTER_MODEL_ENDPOINTS_URL_TEMPLATE = "https://openrouter.ai/api/v1/models/{slug}/endpoints"
_OPENROUTER_SITE_URL = "https://openrouter.ai"
_MAX_MODEL_PROFILE_IMAGE_BYTES = 2 * 1024 * 1024
_MAX_OPENROUTER_ID_CHARS = 128
_MAX_OPENROUTER_METADATA_PAIRS = 16
_MAX_OPENROUTER_METADATA_KEY_CHARS = 64
_MAX_OPENROUTER_METADATA_VALUE_CHARS = 512

_PIPE_METADATA_KEY = "openrouter_pipe"

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

_REMOTE_FILE_MAX_SIZE_DEFAULT_MB = 50
_REMOTE_FILE_MAX_SIZE_MAX_MB = 500
_INTERNAL_FILE_ID_PATTERN = re.compile(r"/files/([A-Za-z0-9-]+)(?:/|\\?|$)")
_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?P<url>[^)]+)\)")
_TEMPLATE_VAR_PATTERN = re.compile(r"{(\w+)}")
_TEMPLATE_IF_OPEN_RE = re.compile(r"\{\{\s*#if\s+(\w+)\s*\}\}")
_TEMPLATE_IF_CLOSE_RE = re.compile(r"\{\{\s*/if\s*\}\}")

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
    "```\n{flagged_excerpt}\n```\n"
    "Provide this excerpt when following up with your administrator.\n"
    "{{/if}}\n"
    "{{#if raw_body}}\n"
    "\n**Raw provider response:**\n"
    "```\n{raw_body}\n```\n"
    "{{/if}}\n"
    "{{#if metadata_json}}\n"
    "\n**Metadata:**\n"
    "```\n{metadata_json}\n```\n"
    "{{/if}}\n"
    "{{#if provider_raw_json}}\n"
    "\n**Provider raw error:**\n"
    "```\n{provider_raw_json}\n```\n"
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
    "Unable to reach OpenRouter's servers.\n\n"
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
    "- OpenRouter service outage\n\n"
    "**What to do:**\n"
    "1. Check your internet connection\n"
    "2. Verify firewall allows HTTPS (port 443)\n"
    "3. Check [OpenRouter Status](https://status.openrouter.ai/)\n"
    "4. Contact your network administrator if the issue persists\n"
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
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**What the status means:**\n"
    "- `502` — the chosen model is down, or it returned something OpenRouter could not read\n"
    "- `503` — no provider was available that satisfies the routing requirements sent with this request\n"
    "- any other `5xx` — a fault inside OpenRouter itself\n\n"
    "**What to do:**\n"
    "- Retry in a few minutes; a model or provider outage normally clears on its own\n"
    "- Try a different model, which routes to a different set of providers\n"
    "- On a `503` that keeps repeating, the blocker is the routing constraints rather than an outage: "
    "an admin should review `Enforce ZDR routing` and the provider-routing settings for this model\n"
    "- Check [OpenRouter Status](https://status.openrouter.ai/) for a platform-wide incident\n"
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
    "This request includes attachments that require a different OpenRouter endpoint than the one enforced for the selected model.\n\n"
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
    "- Or remove the attachment(s) and retry\n"
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
    "2. Whether `WEBUI_SECRET_KEY` changed since that key was saved; the stored value can no longer be decrypted, "
    "so the key has to be entered again\n"
    "3. Whether the key itself was disabled or deleted — issue a replacement at https://openrouter.ai/keys\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE = (
    "### 💳 Insufficient Credits\n\n"
    "OpenRouter could not run this request because the account is out of credits.\n\n"
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
    "OpenRouter refused this request because the account has reached one of its request limits.\n\n"
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
    "OpenRouter cancelled the request: the operation exceeded its time limit before any output was produced.\n\n"
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

class EncryptedStr(str):
    """String wrapper that automatically encrypts/decrypts valve values."""

    _ENCRYPTION_PREFIX = "encrypted:"

    @classmethod
    def _get_encryption_key(cls) -> bytes | None:
        """Return the Fernet key derived from ``WEBUI_SECRET_KEY``.

        Returns:
            Optional[bytes]: URL-safe base64 Fernet key or ``None`` when unset.
        """
        secret = os.getenv("WEBUI_SECRET_KEY")
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

        Returns:
            str: Decrypted plain text or the original value when keyless.
        """
        if not value or not value.startswith(cls._ENCRYPTION_PREFIX):
            return value
        key = cls._get_encryption_key()
        if not key:
            return value[len(cls._ENCRYPTION_PREFIX) :]
        try:
            encrypted_part = value[len(cls._ENCRYPTION_PREFIX) :]
            fernet = Fernet(key)
            decrypted = fernet.decrypt(encrypted_part.encode())
            return decrypted.decode()
        except InvalidToken:
            logger.warning("Failed to decrypt value: invalid token or key mismatch")
            return value
        except (ValueError, UnicodeDecodeError) as e:
            logger.warning(f"Failed to decrypt value: {type(e).__name__}: {e}")
            return value

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

    # Connection & Auth
    BASE_URL: str = Field(
        default=((os.getenv("OPENROUTER_API_BASE_URL") or "").strip() or "https://openrouter.ai/api/v1"),
        description="OpenRouter API base URL. Override this if you are using a gateway or proxy.",
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
            "Globs are literal about the `~` prefix; add '~anthropic/*' to cover router aliases."
        ),
    )
    FORCE_RESPONSES_MODELS: str = Field(
        default="",
        description=(
            "Comma-separated glob patterns of model ids that must use /responses "
            "(overrides FORCE_CHAT_COMPLETIONS_MODELS when both match)."
        ),
    )
    AUTO_FALLBACK_CHAT_COMPLETIONS: bool = Field(
        default=True,
        description=(
            "When True, retry the request against /chat/completions if /responses fails with an "
            "endpoint/model support error before any streaming output is produced."
        ),
    )
    API_KEY: EncryptedStr = Field(
        default_factory=_default_api_key,
        title="OpenRouter API key",
        description="Your OpenRouter API key. Defaults to the OPENROUTER_API_KEY environment variable.",
    )
    HTTP_REFERER_OVERRIDE: str = Field(
        default="",
        description=(
            "Override the `HTTP-Referer` header sent to OpenRouter for app attribution. "
            "Must be a full URL including scheme (e.g. https://example.com), not just a hostname. "
            "When empty, the pipe uses its default project URL."
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
        description="Idle read timeout (seconds) applied to active streams when HTTP_TOTAL_TIMEOUT_SECONDS is disabled. Generous default favors smoother User Interface behavior for slow providers.",
    )

    # Remote File/Image Download Settings
    REMOTE_DOWNLOAD_MAX_RETRIES: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum number of retry attempts for downloading remote images and files. Set to 0 to disable retries.",
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
        description="Maximum total time in seconds to spend on retry attempts. Retries will stop if this time limit is exceeded.",
    )

    REMOTE_FILE_MAX_SIZE_MB: int = Field(
        default=_REMOTE_FILE_MAX_SIZE_DEFAULT_MB,
        ge=1,
        le=_REMOTE_FILE_MAX_SIZE_MAX_MB,
        description="Maximum size in MB for downloading remote files/images. Files exceeding this limit are skipped. When Open WebUI RAG is enabled, the pipe automatically caps downloads to Open WebUI's FILE_MAX_SIZE (if set).",
    )
    SAVE_REMOTE_FILE_URLS: bool = Field(
        default=True,
        description="When True, remote URLs and data URLs in the file_url field are downloaded/parsed and re-hosted in Open WebUI storage (default; keeps chats replayable if the source link later dies, at the cost of storage growth). When False, file_url values pass through untouched. Note: This valve only affects the file_url field; see SAVE_FILE_DATA_CONTENT for file_data behavior.",
    )
    SAVE_FILE_DATA_CONTENT: bool = Field(
        default=True,
        description="When True, base64 content and URLs in the file_data field are parsed/downloaded and re-hosted in Open WebUI storage to prevent chat history bloat. When False, file_data values pass through untouched. Recommended: Keep enabled to avoid large inline payloads in chat history.",
    )
    BASE64_MAX_SIZE_MB: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum size in MB for base64-encoded files/images before decoding. Larger payloads will be rejected to prevent memory issues and excessive HTTP request sizes.",
    )
    IMAGE_UPLOAD_CHUNK_BYTES: int = Field(
        default=1 * 1024 * 1024,
        ge=64 * 1024,
        le=8 * 1024 * 1024,
        description="Maximum number of bytes to buffer at a time when loading Open WebUI-hosted images before forwarding them to a provider. Lower values reduce peak memory usage when multiple users edit images concurrently.",
    )
    VIDEO_MAX_SIZE_MB: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum size in MB for inline base64 (data:) video payloads and for stored videos re-read to extract frames. Oversized videos are rejected/skipped; remote http(s) and YouTube video links are forwarded to the provider unmeasured.",
    )
    FALLBACK_STORAGE_EMAIL: str = Field(
        default=(os.getenv("OPENROUTER_STORAGE_USER_EMAIL") or "openrouter-pipe@system.local"),
        description="Owner email used when multimodal uploads occur without a chat user (e.g., API automations).",
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
        description="Enable SSRF (Server-Side Request Forgery) protection for remote URL downloads. When enabled, blocks requests to private IP ranges (localhost, 192.168.x.x, 10.x.x.x, etc.) to prevent internal network probing. HTTP is disabled by default; see ALLOW_INSECURE_HTTP_* for explicit opt-in.",
    )
    ALLOW_INSECURE_HTTP: bool = Field(
        default=False,
        description="Allow plaintext HTTP remote URLs when explicitly enabled. HTTP is disabled by default; only enable with a narrow allowlist in ALLOW_INSECURE_HTTP_HOSTS.",
    )
    ALLOW_INSECURE_HTTP_HOSTS: str = Field(
        default="",
        description=(
            "Comma-separated list of hosts or host:port entries allowed for plaintext HTTP. "
            "Exact match only (no wildcards). Empty means no HTTP allowed. "
            "Example: 'example.com, example.org:8080, 203.0.113.10'."
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
            "Set to 'auto' to import every available Responses-capable model."
        ),
    )
    MODEL_CATALOG_REFRESH_SECONDS: int = Field(
        default=60 * 60,
        ge=60,
        description="How long to cache the OpenRouter model catalog (in seconds) before refreshing.",
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
            "use Enforce ZDR routing for that -- and filtering is skipped if the ZDR list cannot be loaded, except video models, which have no ZDR endpoints and stay hidden."
        ),
    )
    ZDR_ENFORCE: bool = Field(
        default=False,
        title="Enforce ZDR routing",
        description=(
            "When enabled, all requests include provider.zdr=true and will be rejected if the selected model "
            "does not have any ZDR endpoints."
        ),
    )
    ALLOW_USER_ZDR_OVERRIDE: bool = Field(
        default=True,
        title="Allow user ZDR override",
        description=(
            "When enabled, users can toggle 'Request ZDR' per chat. "
            "If Enforce ZDR routing is enabled, user overrides are ignored."
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
            "Supported tags: free, thinking, online, nitro, exacto, extended. "
            "The full model ID with ':variant' suffix is sent to OpenRouter for specialized routing."
        ),
    )

    ENABLE_REASONING: bool = Field(
        default=True,
        title="Show live reasoning",
        description="Request live reasoning traces whenever the selected model supports them.",
    )
    THINKING_OUTPUT_MODE: Literal["open_webui", "status", "both"] = Field(
        default="open_webui",
        title="Thinking output",
        description=(
            "Controls where in-progress thinking is surfaced while a response is being generated. "
            "'open_webui' streams reasoning in the Open WebUI reasoning box only; "
            "'status' shows thinking only as status messages; "
            "'both' enables both outputs."
        ),
    )
    ENABLE_ANTHROPIC_INTERLEAVED_THINKING: bool = Field(
        default=True,
        title="Anthropic interleaved thinking",
        description=(
            "When True, enables Claude's interleaved thinking mode by sending "
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
            "HMAC-SHA256(WEBUI_SECRET_KEY, chat_id) so OpenRouter keeps each conversation on one "
            "provider and maximizes prompt-cache hits across turns. Opaque; no raw identifiers. "
            "Skipped if WEBUI_SECRET_KEY is unset."
        ),
    )
    ENABLE_PLUGIN_SYSTEM: bool = Field(
        default=False,
        title="Enable plugin system",
        description=(
            "Master switch for the plugin system. When False, plugins are never called at all. "
            "Takes effect immediately without restart."
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
            "Default reasoning effort to request from supported models. Use 'none' to skip reasoning entirely "
            "or 'xhigh' when maximum depth is desired (only on supporting models)."
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
            "Base thinking budget (tokens) for Gemini 2.5 models. When 0, thinking is disabled. "
            "When non-zero, the pipe scales this value based on reasoning effort (minimal -> smaller, xhigh -> larger)."
        ),
    )
    PERSIST_REASONING_TOKENS: Literal["disabled", "next_reply", "conversation"] = Field(
        default="conversation",
        title="Reasoning retention",
        description="Reasoning retention: 'disabled' keeps nothing, 'next_reply' keeps thoughts only until the following assistant reply finishes, and 'conversation' keeps them for the full chat history.",
    )
    TASK_MODEL_REASONING_EFFORT: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = Field(
        default="low",
        title="Task reasoning effort",
        description=(
            "Reasoning effort requested for Open WebUI background tasks (titles, tags, etc.) when they target this pipe's models. "
            "Low is the default balance between speed and quality; set to 'minimal' to prioritize fastest runs, "
            "or use medium/high for progressively deeper background reasoning at higher cost."
        ),
    )

    # Tool execution behavior
    TOOL_EXECUTION_MODE: Literal["Pipeline", "Open-WebUI"] = Field(
        default="Pipeline",
        title="Tool execution mode",
        description=(
            "Where to execute tools. 'Pipeline' executes tool calls inside this pipe "
            "(with its own batching, failure limits, and special tool handling). 'Open-WebUI' hands tool calls back rather than "
            "running them here, so Open WebUI executes them and renders the native tool UI."
        ),
    )
    SHOW_TOOL_CARDS: bool = Field(
        default=False,
        title="Show tool execution cards",
        description="Show collapsible cards in chat with tool name, arguments, and results. When disabled, tools run silently without visual status indicators.",
    )
    PERSIST_TOOL_RESULTS: bool = Field(
        default=False,
        title="Keep tool results",
        description="Persist tool call results across conversation turns. When disabled, tool results stay ephemeral and the model relies on its own summaries or re-runs tools.",
    )
    ARTIFACT_ENCRYPTION_KEY: EncryptedStr = Field(
        default_factory=_default_artifact_encryption_key,
        description="Use at least 16 chars. Encrypt reasoning tokens (and optionally all persisted artifacts). Changing the key creates a new table; prior artifacts become inaccessible.",
    )
    ENCRYPT_ALL: bool = Field(
        default=True,
        description="Encrypt every persisted artifact when ARTIFACT_ENCRYPTION_KEY is set. When False, only reasoning tokens are encrypted.",
    )
    ENABLE_LZ4_COMPRESSION: bool = Field(
        default=True,
        description="When True (and lz4 is available), compress large encrypted artifacts to reduce database read/write overhead.",
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
            "enforcing explicit types, required fields, and disallowing additionalProperties."
        ),
    )
    MAX_FUNCTION_CALL_LOOPS: int = Field(
        default=25,
        description=(
            "Maximum number of full execution cycles (loops) allowed per request when "
            "TOOL_EXECUTION_MODE is 'Pipeline'. Each loop involves the model generating "
            "one or more function/tool calls, executing all requested functions, and feeding "
            "the results back into the model. When the limit is reached, pending tool calls "
            "are returned to the model marked as skipped so it can write a final answer. "
            "Has no effect when TOOL_EXECUTION_MODE is 'Open-WebUI' (the round limit is managed "
            "by Open WebUI in that mode)."
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
            "Persistence is skipped when any required IDs are missing (user_id, chat_id, message_id, request_id)."
        ),
    )
    SESSION_LOG_DIR: str = Field(
        default="session_logs",
        description=(
            "Base directory for encrypted session log archives. "
            "Files are stored under <SESSION_LOG_DIR>/<user_id>/<chat_id>/<message_id>.zip."
        ),
    )
    SESSION_LOG_ZIP_PASSWORD: EncryptedStr = Field(
        default=EncryptedStr(""),
        description=(
            "Password used to encrypt session log zip files (AES-encrypted zip). "
            "Recommend using a long random passphrase and encrypting the value (requires WEBUI_SECRET_KEY)."
        ),
    )
    SESSION_LOG_RETENTION_DAYS: int = Field(
        default=90,
        ge=1,
        description="Retention window for stored session log archives. Cleanup deletes zip files older than this many days.",
    )
    SESSION_LOG_CLEANUP_INTERVAL_SECONDS: int = Field(
        default=3600,
        ge=60,
        description="How often (in seconds) to run the session log cleanup loop when storage is enabled.",
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
            "Compression level (0-9) for deflated/bzip2 zip compression. "
            "Ignored for stored/lzma."
        ),
    )
    SESSION_LOG_MAX_LINES: int = Field(
        default=20000,
        ge=100,
        le=200000,
        description="Maximum number of log records held in memory per request (older entries are dropped).",
    )
    SESSION_LOG_FORMAT: Literal["jsonl", "text", "both"] = Field(
        default="jsonl",
        description=(
            "Format written inside session log archives. "
            "logs.jsonl is always written; 'jsonl' writes only it (one JSON object per record), while 'text' and 'both' additionally write a plain-text logs.txt (so 'text' and 'both' produce the same files)."
        ),
    )
    SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS: int = Field(
        default=30,
        ge=1,
        description="How often (in seconds) to check the database for log pieces waiting to be packed and build one zip per message.",
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
        description="Maximum number of message bundles to assemble per archiving pass.",
    )
    SESSION_LOG_STALE_FINALIZE_SECONDS: int = Field(
        default=6 * 7200,
        ge=60,
        description=(
            "If a message has staged session-log segments but never signals that it finished "
            "(the worker crashed or was killed), finalize an incomplete zip after this many seconds since the last piece."
        ),
    )
    SESSION_LOG_LOCK_STALE_SECONDS: int = Field(
        default=1800,
        ge=60,
        description="Stale lock timeout (seconds) for DB-backed session log assembly locks; stale locks are reclaimed.",
    )
    ENABLE_TIMING_LOG: bool = Field(
        default=False,
        description=(
            "When True, record how long each internal step of a request takes. "
            "Writes to TIMING_LOG_FILE path directly (not session archives). "
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
        description="Maximum number of in-flight OpenRouter requests allowed per process.",
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
        description="Maximum number of raw SSE chunks buffered before the pipe stops reading from OpenRouter until the backlog clears. 0=unbounded (cannot stall, recommended); bounded values &lt;500 risk stalls on tool-heavy loads, slow database writes, or a slow browser (a slow reader fills the decoded-event backlog, which fills the raw-chunk backlog, which stops the pipe reading from OpenRouter).",
    )
    STREAMING_EVENT_QUEUE_MAXSIZE: int = Field(
        default=0,
        ge=0,
        description="Maximum number of decoded events buffered before the rest of the pipe handles them. 0=unbounded (cannot stall, recommended); bounded values &lt;500 risk stalls on tool-heavy loads, slow database writes, or a slow browser (a slow reader fills the decoded-event backlog, which fills the raw-chunk backlog, which stops the pipe reading from OpenRouter).",

    )
    STREAMING_CHUNK_QUEUE_WARN_SIZE: int = Field(
        default=1000,
        ge=100,
        description="Log a warning when the raw-chunk backlog reaches this many chunks, so an unbounded buffer is still watched. The minimum of 100 avoids flooding the log under sustained high load; raise it on busy servers.",
    )
    STREAMING_EVENT_QUEUE_WARN_SIZE: int = Field(
        default=1000,
        ge=100,
        description="Log a warning when the decoded-event backlog reaches this many events, so an unbounded buffer is still watched. The minimum of 100 avoids flooding the log under sustained high load; raise it on busy servers.",
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
            "0=unbounded (default behavior)."
        ),
    )
    MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS: float = Field(
        default=1.0,
        ge=0,
        description=(
            "When MIDDLEWARE_STREAM_QUEUE_MAXSIZE>0, maximum seconds to wait while adding one item to that buffer before dropping that single item and continuing the stream. "
            "0 disables the timeout (not recommended; a stalled browser can hold up the pipe indefinitely)."
        ),
    )
    OPENROUTER_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_OPENROUTER_ERROR_TEMPLATE,
        description=(
            "Markdown template used when OpenRouter rejects a request with a status that has no template of its own "
            "(400, 403, 404, 422, and so on). Clear this box and save to restore this built-in text. "
            "Placeholders such as {heading}, {detail}, {sanitized_detail}, {provider}, {model_identifier}, "
            "{requested_model}, {api_model_id}, {normalized_model_id}, {openrouter_code}, {upstream_type}, "
            "{reason}, {request_id}, {request_id_reference}, {openrouter_message}, {upstream_message}, "
            "{moderation_reasons}, {flagged_excerpt}, {raw_body}, {context_limit_tokens}, {max_output_tokens}, "
            "{include_model_limits}, {metadata_json}, {provider_raw_json}, {error_id}, {timestamp}, {session_id}, {user_id}, "
            "{native_finish_reason}, {error_chunk_id}, {error_chunk_created}, {streaming_provider}, {streaming_model}, "
            "{retry_after_seconds}, {rate_limit_type}, {required_cost}, and {account_balance} are replaced when values are available. "
            "Lines containing placeholders are omitted automatically when the referenced value is missing or empty. "
            "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to show them only when that value is set."
        ),
    )
    ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE: str = Field(
        default=DEFAULT_ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE,
        description=(
            "Markdown template used when a request requires /chat/completions (e.g. direct video uploads) but the model is "
            "explicitly forced to /responses by endpoint override valves."
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
            "Markdown template for HTTP 401 errors, and for the pipe's own failure to read a usable API key. "
            "Both cases fill {error_id}, {timestamp}, {session_id}, {user_id}, {support_email}, {support_url}, {openrouter_code} and {openrouter_message}. "
            "A 401 returned by OpenRouter also fills the shared error-context fields — {request_id}, {provider}, {model_identifier}, {requested_model}, {reason}, {metadata_json} and the rest of the set the rejected-request template lists. "
            "Nothing is sent when the key itself cannot be read, so on that path those extra fields have no value and any line using one prints the braces verbatim; wrap such a line in {{#if request_id}}...{{/if}} and it is left out instead. "
            "A name nothing supplies is never substituted, whichever path rendered the card."
        ),
    )

    INSUFFICIENT_CREDITS_TEMPLATE: str = Field(
        default=DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE,
        description=(
            "Markdown template for HTTP 402 errors when the account is out of credits. Supports {error_id}, {timestamp}, {openrouter_code}, "
            "{openrouter_message}, {request_id}, {required_cost}, {account_balance}, {support_email}, and other shared context variables. "
            "{request_id} is OpenRouter's own reference for the rejected request, which its support can look up; the built-in text shows it on its own row whenever the rejection carried one."
        ),
    )

    RATE_LIMIT_TEMPLATE: str = Field(
        default=DEFAULT_RATE_LIMIT_TEMPLATE,
        description=(
            "Markdown template for HTTP 429 rate-limit errors. Use placeholders such as {error_id}, {timestamp}, {openrouter_code}, {retry_after_seconds}, "
            "{rate_limit_type}, {request_id}, {support_email}, and the standard context variables. "
            "{request_id} is OpenRouter's own reference for the rejected request, which its support can look up; the built-in text shows it on its own row whenever the rejection carried one."
        ),
    )

    SERVER_TIMEOUT_TEMPLATE: str = Field(
        default=DEFAULT_SERVER_TIMEOUT_TEMPLATE,
        description=(
            "Markdown template for HTTP 408 errors returned by OpenRouter (server-side timeout). Supports the common context variables plus "
            "{openrouter_message}, {openrouter_code}, {request_id}, and support contact placeholders. "
            "{request_id} is OpenRouter's own reference for the timed-out request, which its support can look up; the built-in text shows it on its own row whenever the response carried one."
        ),
    )

    PAYLOAD_TOO_LARGE_TEMPLATE: str = Field(
        default=DEFAULT_PAYLOAD_TOO_LARGE_TEMPLATE,
        description=(
            "Markdown template for HTTP 413 errors when the request payload exceeds size limits. Supports {error_id}, {timestamp}, {openrouter_code}, "
            "{openrouter_message}, {model_identifier}, {request_id}, {support_email}, and other shared context variables. "
            "{request_id} is OpenRouter's own reference for the rejected request, which its support can look up; the built-in text shows it on its own row whenever the rejection carried one."
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
            "Markdown template for network timeout errors. "
            "Available variables: {error_id}, {timeout_seconds}, {timestamp}, "
            "{session_id}, {user_id}, {support_email}. "
            "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to show them only when that value is set."
        )
    )

    CONNECTION_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_CONNECTION_ERROR_TEMPLATE,
        description=(
            "Markdown template for connection failures. "
            "Available variables: {error_id}, {error_type}, {timestamp}, "
            "{session_id}, {user_id}, {support_email}. "
            "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to show them only when that value is set."
        )
    )

    SERVICE_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_SERVICE_ERROR_TEMPLATE,
        description=(
            "Markdown template for OpenRouter 5xx errors. "
            "Available variables: {error_id}, {status_code}, {reason}, {timestamp}, "
            "{session_id}, {user_id}, {support_email}. "
            "A 5xx that OpenRouter itself returned also fills {request_id}, its own reference for that request; a 5xx raised by the connection to OpenRouter, or by a failure inside the pipe, carries no such reference and a line using it prints the braces verbatim unless it is wrapped in a conditional. "
            "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to show them only when that value is set."
        )
    )

    INTERNAL_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_INTERNAL_ERROR_TEMPLATE,
        description=(
            "Markdown template for unexpected internal errors. "
            "Available variables: {error_id}, {error_type}, {timestamp}, "
            "{session_id}, {user_id}, {support_email}, {support_url}. "
            "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to show them only when that value is set."
        )
    )

    MODEL_RESTRICTED_TEMPLATE: str = Field(
        default=DEFAULT_MODEL_RESTRICTED_TEMPLATE,
        description=(
            "Markdown template emitted when the requested model is blocked by MODEL_ID and/or model filter valves. "
            "Available variables: {requested_model}, {normalized_model_id}, {restriction_reasons}, "
            "{model_id_filter}, {free_model_filter}, {tool_calling_filter}, plus standard context variables "
            "like {error_id}, {timestamp}, {session_id}, {user_id}, {support_email}, and {support_url}."
        ),
    )
    STREAM_INTERRUPTED_TEMPLATE: str = Field(
        default=DEFAULT_STREAM_INTERRUPTED_TEMPLATE,
        description=(
            "Markdown template appended to the assistant message when the streaming response ends "
            "without a completion event. The partial content is preserved and this notice is appended. "
            "Available variables: {model}, {timestamp}, {support_email}, {support_url}."
        ),
    )

    MAX_PARALLEL_TOOLS_GLOBAL: int = Field(
        default=200,
        ge=1,
        le=2000,
        description="Global ceiling for simultaneously executing tool calls.",
    )
    MAX_PARALLEL_TOOLS_PER_REQUEST: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Per-request concurrency limit for tool execution workers.",
    )
    BREAKER_MAX_FAILURES: int = Field(
        default=5,
        ge=1,
        le=50,
        description=(
            "Number of failures allowed per breaker window before that user's requests, tools, or database writes are temporarily blocked. "
            "Set higher to reduce trip frequency in noisy environments."
        ),
    )
    BREAKER_WINDOW_SECONDS: int = Field(
        default=60,
        ge=5,
        le=900,
        description="Sliding window length (in seconds) used when counting breaker failures.",
    )
    BREAKER_HISTORY_SIZE: int = Field(
        default=5,
        ge=1,
        le=200,
        description=(
            "Maximum failures the per-user database circuit breaker remembers. Keep at or above BREAKER_MAX_FAILURES so history is not truncated below the trip count (the request and per-tool breakers size their own history automatically)."
        ),
    )
    TOOL_BATCH_CAP: int = Field(
        default=4,
        ge=1,
        le=32,
        description="Maximum number of compatible tool calls that may be executed in a single batch.",
    )
    TOOL_OUTPUT_RETENTION_TURNS: int = Field(
        default=10,
        ge=0,
        description=(
            "Number of most recent logical turns whose tool outputs are sent in full. "
            "A turn starts when a user speaks and includes the assistant/tool responses "
            "that follow until the next user message. Older turns have their persisted "
            "tool outputs shortened to save tokens. Set to 0 to keep every tool output in full."
        ),
    )
    TOOL_TIMEOUT_SECONDS: int = Field(
        default=60,
        ge=1,
        le=600,
        description="Max seconds to wait for an individual tool to finish before timing out. Generous default reduces disruption for real-world tools.",
    )
    TOOL_BATCH_TIMEOUT_SECONDS: int = Field(
        default=120,
        ge=1,
        description="Max seconds to wait for a batch of tool calls to complete before timing out. Longer default keeps complex batches from being interrupted prematurely.",
    )
    TOOL_IDLE_TIMEOUT_SECONDS: int | None = Field(
        default=None,
        ge=1,
        description="Idle timeout (seconds) between tool executions in a queue. Set to null for unlimited idle time so intermittent tool usage does not fail unexpectedly.",
    )
    TOOL_SHUTDOWN_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        ge=0,
        description=(
            "Maximum seconds to wait for that request's running tools to finish and stop during cleanup. "
            "0 disables the graceful wait and cancels workers immediately."
        ),
    )
    ENABLE_REDIS_CACHE: bool = Field(
        default=True,
        description="Buffer artifact writes through Redis when REDIS_URL and more than one worker are detected.",
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
        description="Log a critical alert after this many consecutive failures writing the buffered artifacts to the database. Buffering is not disabled: the pipe waits longer between attempts and keeps retrying, resuming when writes succeed (new writes fall back to direct DB meanwhile).",
    )
    COSTS_REDIS_DUMP: bool = Field(
        default=False,
        description="When True, push per-request usage snapshots into Redis for downstream cost analytics.",
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
        description="Days an artifact is kept before cleanup. Its stored timestamp is refreshed on every database read, so retention runs from last access, not creation.",
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
        description="When enabled, and the request does not already set a limit, fill in the provider's advertised max_output_tokens. Disable to send no limit of the pipe's own. This valve controls the automatic value, not yours: A `max_tokens` of 1 or above is forwarded unchanged. OpenRouter documents the parameter as 1 or above and Open WebUI's slider reaches -2, so a value below 1 is sent as no cap -- which means the automatic ceiling applies if this valve is on.",
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
        description="When True, send OpenRouter `user` (value chosen by END_USER_ID_SOURCE), and also include `metadata.user_id` with the Open WebUI user GUID.",
    )
    END_USER_ID_SOURCE: Literal["id", "email", "name"] = Field(
        default="id",
        description="What the OpenRouter `user` field carries when SEND_END_USER_ID is on: the Open WebUI GUID, the user's email, or their display name. Email/name fall back to the GUID when empty. Sending email or name shares PII with OpenRouter.",
    )
    SEND_SESSION_ID: bool = Field(
        default=False,
        description="When True, include the Open WebUI session_id as `metadata.session_id` (metadata only).",
    )
    SEND_CHAT_ID: bool = Field(
        default=False,
        description="When True, include the Open WebUI chat_id as `metadata.chat_id` (metadata only).",
    )
    SEND_MESSAGE_ID: bool = Field(
        default=False,
        description="When True, include the Open WebUI message_id as `metadata.message_id` (metadata only).",
    )
    MAX_INPUT_IMAGES_PER_REQUEST: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of image inputs (images attached by the user, plus reused images from earlier replies) to include in a single provider request.",
    )
    IMAGE_INPUT_SELECTION: Literal["user_turn_only", "user_then_assistant"] = Field(
        default="user_then_assistant",
        description=(
            "Controls which images are forwarded to the provider. "
            "'user_turn_only' restricts inputs to the images supplied with the current user message. "
            "'user_then_assistant' falls back to the most recent assistant-generated images when the user did not attach any."
        ),
    )

    # Model metadata synchronization
    UPDATE_MODEL_IMAGES: bool = Field(
        default=True,
        description="When enabled, automatically sync profile image URLs from OpenRouter's frontend catalog to Open WebUI model metadata. Disable to manage images manually.",
    )
    UPDATE_MODEL_CAPABILITIES: bool = Field(
        default=True,
        description="When enabled, automatically sync model capabilities (vision, file_upload, web_search, etc.) from OpenRouter's API catalog to Open WebUI model metadata. Disable to manage capabilities manually.",
    )
    DISABLE_BUILTIN_TOOLS_ON_MEDIA_MODELS: bool = Field(
        default=True,
        description=(
            "Turn off Open WebUI's built-in tools for models that produce images or video. "
            "These models answer with a picture or a clip rather than a tool call, and "
            "offering them web search, code execution and the rest tends to make a turn "
            "fail or come back empty. With this on, each image or video model arrives in "
            "your workspace with 'Built-in tools' already unticked, so you can see the "
            "setting rather than wonder why tools are quiet. Tick it back on for any single "
            "model and your choice stays put; the pipe only sets it the first time it adds "
            "the model. Requires model capability syncing to be enabled."
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
        description="Enable the OpenRouter Web Search server tool. When disabled, web search toggles are hidden from users.",
    )
    ENABLE_WEB_FETCH: bool = Field(
        default=True,
        description="Enable the OpenRouter Web Fetch server tool. When disabled, web fetch toggles are hidden from users.",
    )
    ENABLE_DATETIME: bool = Field(
        default=True,
        description="Enable the OpenRouter Datetime server tool (free, no additional cost). When disabled, datetime toggles are hidden from users.",
    )
    ENABLE_ADVISOR: bool = Field(
        default=True,
        description="Enable the OpenRouter Advisor server tool (consult a higher-intelligence model mid-generation). When disabled, advisor toggles are hidden from users.",
    )
    ENABLE_SUBAGENT: bool = Field(
        default=True,
        description="Enable the OpenRouter Subagent server tool (delegate tasks to a worker model an admin chooses). When disabled, subagent toggles are hidden from users.",
    )
    ENABLE_SEARCH_MODELS: bool = Field(
        default=True,
        description="Enable the OpenRouter model-search server tool (let the model search the OpenRouter catalog). When disabled, model-search toggles are hidden from users.",
    )
    ENABLE_IMAGE_GENERATION: bool = Field(
        default=True,
        description="Enable the OpenRouter Image Generation server tool. When disabled, image generation toggles are hidden from users.",
    )
    ENABLE_VIDEO_GENERATION: bool = Field(
        default=True,
        description=(
            "Add OpenRouter's video-generation models, which render in the background, to the model list. "
            "Video models are never treated as ZDR-capable."
        ),
    )

    AUTO_INSTALL_WEB_TOOLS_FILTER: bool = Field(
        default=True,
        description="Automatically install/update the OpenRouter Web Tools filter function in Open WebUI.",
    )
    AUTO_ATTACH_WEB_TOOLS_FILTER: bool = Field(
        default=True,
        description="Automatically attach the OpenRouter Web Tools filter to all pipe models (so the toggle appears in the Integrations menu).",
    )
    AUTO_DEFAULT_WEB_TOOLS_FILTER: bool = Field(
        default=False,
        description="When enabled, marks the OpenRouter Web Tools filter as a Default Filter on all pipe models (pre-enabled per chat; users can still turn it off).",
    )

    AUTO_INSTALL_IMAGE_GEN_FILTER: bool = Field(
        default=True,
        description="Automatically install/update the OpenRouter Image Generation filter function in Open WebUI.",
    )
    AUTO_ATTACH_IMAGE_GEN_FILTER: bool = Field(
        default=True,
        description="Automatically attach the OpenRouter Image Generation filter to all pipe models.",
    )
    ENABLE_OPENROUTER_IMAGE_GENERATION: bool = Field(
        default=True,
        description=(
            "Expose OpenRouter's image-generation models (Sourceful, Flux, Seedream and "
            "the rest) as models you can pick in chat. Models that produce both text and "
            "images stay where they already are in the chat list; this only adds the "
            "image-only ones."
        ),
    )
    AUTO_INSTALL_IMAGE_FILTERS: bool = Field(
        default=True,
        description=(
            "Install and keep up to date one settings panel per image model, built from "
            "the settings that model tells OpenRouter it accepts -- so nobody is shown an "
            "aspect ratio their model rejects. Alongside those, every panel carries "
            "Output size: a size tier typed there is checked against the tiers that model "
            "publishes, or against 512, 1K, 2K and 4K where it publishes none, and a tier "
            "the model does not list is dropped before the request goes out, while exact "
            "pixels such as 1024x1024 travel as typed. A model that answers with a picture "
            "and no text carries three more the panel supplies rather than the model: "
            "Provider options, Reference images and Reference image links. If a model's "
            "settings list cannot be read on a refresh, it keeps the settings from the "
            "last successful read; a model never read gets no panel at all rather than a "
            "guessed set."
        ),
    )
    AUTO_ATTACH_IMAGE_FILTERS: bool = Field(
        default=True,
        description=(
            "Attach each image model's own settings panel to it, so the settings appear "
            "in the chat controls when that model is selected. Turn this off to install "
            "the panels but leave attaching them to you."
        ),
    )
    AUTO_DEFAULT_IMAGE_FILTERS: bool = Field(
        default=True,
        description=(
            "Always keep the attached image filters enabled by default on "
            "image-output models. Reapplied at every catalog refresh."
        ),
    )

    AUTO_INSTALL_VIDEO_FILTERS: bool = Field(
        default=True,
        description="Automatically install/update the OpenRouter Video Generation companion filter function in Open WebUI.",
    )
    AUTO_ATTACH_VIDEO_FILTERS: bool = Field(
        default=True,
        description="Automatically attach the OpenRouter Video Generation filter to OpenRouter video-generation models.",
    )
    AUTO_DEFAULT_VIDEO_FILTERS: bool = Field(
        default=True,
        description="Always keep the per-model video filter enabled by default on its video model. Reapplied at every catalog refresh. Models that require a per-model parameter (e.g. Veo's personGeneration) cannot be driven without it; parameter-free models still generate.",
    )
    ENABLE_OPENROUTER_FUSION: bool = Field(
        default=True,
        description="Master switch for OpenRouter Fusion support. When enabled, the pipe installs the 'OpenRouter Fusion' filter and attaches it to the openrouter/fusion model automatically.",
    )
    AUTO_INSTALL_FUSION_FILTER: bool = Field(
        default=True,
        description="Automatically install/update the OpenRouter Fusion filter function in Open WebUI.",
    )
    AUTO_ATTACH_FUSION_FILTER: bool = Field(
        default=True,
        description="Automatically attach the OpenRouter Fusion filter to the openrouter/fusion model only (so its panel/judge options appear in the Integrations menu). Never attaches to other models.",
    )
    AUTO_DEFAULT_FUSION_FILTER: bool = Field(
        default=True,
        description="Mark the OpenRouter Fusion filter as a Default Filter on the openrouter/fusion model (pre-enabled per chat). Does NOT force Fusion to run — the per-user 'Always run Fusion' toggle is off by default. Reapplied at every catalog refresh.",
    )
    FUSION_BACKEND: Literal["openrouter", "internal"] = Field(
        default="internal",
        title="Fusion Backend",
        description="Which engine runs deliberation for the dedicated fusion models. 'openrouter': requests go to OpenRouter's hosted Fusion — the panel and judge execute on their servers. 'internal': the pipe runs the same panel → judge → synthesis flow itself as ordinary pipe model calls, so panel members can use every Open WebUI tool that user has (knowledge bases, tool servers, pipe server tools), every dial (ZDR, reasoning effort, cost attribution) applies per member, and one member failing does not kill the whole stream. The live panel UI is identical on both backends.",
    )
    FUSION_PANEL_SYSTEM_PROMPT: str = Field(
        default=DEFAULT_FUSION_PANEL_SYSTEM_PROMPT,
        title="Fusion Panel System Prompt",
        description="System prompt sent to every panel member on the internal fusion backend. It enforces independent, committed, source-cited answers and forbids members from revealing the deliberation machinery. Edit to reshape panel behaviour; the shipped default was produced by a multi-model design tournament and is tuned to pair with the judge and synthesis prompts.",
    )
    FUSION_JUDGE_SYSTEM_PROMPT: str = Field(
        default=DEFAULT_FUSION_JUDGE_SYSTEM_PROMPT,
        title="Fusion Judge System Prompt",
        description="System prompt for the internal fusion judge (runs at temperature 0). CAUTION: the judge must emit a single strict JSON object with exactly the five keys consensus / contradictions / partial_coverage / unique_insights / blind_spots — the live Analysis panel and the synthesis stage are depend on exactly that JSON. Keep the output-format rules intact when editing; if the judge stops producing valid JSON the run continues without the analysis.",
    )
    FUSION_SYNTHESIS_SYSTEM_PROMPT: str = Field(
        default=DEFAULT_FUSION_SYNTHESIS_SYSTEM_PROMPT,
        title="Fusion Synthesis System Prompt",
        description="System prompt for the internal fusion synthesis call — the model that receives the panel drafts plus the judge's analysis and writes the final user-facing answer. It enforces 'compose, never average', honest handling of contradictions, and total secrecy about the deliberation machinery. Edit to change the final answer's voice or composition rules.",
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
            "taking a long time \u2014 only one that has gone quiet is. When it does run "
            "out the chat keeps the job's card and says the render is still going at "
            "OpenRouter: nothing is cancelled and OpenRouter still bills the job, and the "
            "person can press Continue Response on that message to pick it back up. The "
            "wait is never allowed to be shorter than a single status check can take, so "
            "anything below `Maximum poll interval` plus the HTTP read timeout is raised "
            "to that."
        ),
    )
    VIDEO_STATUS_POLL_MAX_ERRORS: int = Field(
        default=5,
        ge=1,
        le=25,
        description="Maximum consecutive video status polling errors before the job is marked failed in the chat.",
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
            "request succeed."
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
            "the request already and never leaves this server. Turn it on only if large "
            "reference images are being refused for size."
        ),
    )
    TELL_USERS_ABOUT_THE_FILE_HOST: bool = Field(
        default=True,
        description=(
            "Warn in the chat before a user's attachment is uploaded to the file host. Their "
            "own media leaves this server, so they are warned by default -- and while this is "
            "on, an upload that could not be announced does not happen: the request fails "
            "instead of publishing the file unannounced. Switch it off if you have told your "
            "users another way. Either way the finished message keeps a written record of "
            "what was uploaded and where; only this advance warning is optional."
        ),
    )
    FILE_HOST_NOTICE: str = Field(
        default=(
            "Sending your {kind} to {host} so the model can read it, {retention}."
        ),
        description=(
            "The wording of that advance warning. Rewrite it in your own words or your own "
            "language. {kind} becomes clip, sound file or picture; {host} names the file "
            "host; {retention} says how long it stays there. Leave out any you do not want. "
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
        description="Chunk size in bytes used when downloading generated video content.",
    )
    MAX_CONCURRENT_VIDEO_GENS: int = Field(
        default=2,
        ge=1,
        le=100,
        description="Maximum number of video generation jobs running per pipe process.",
    )
    MAX_CONCURRENT_VIDEO_GENS_PER_USER: int = Field(
        default=2,
        ge=1,
        le=25,
        description="Maximum number of video generation jobs running per user per pipe process.",
    )
    VIDEO_FRAME_IMAGE_MAX_BYTES: int = Field(
        default=12 * 1024 * 1024,
        ge=64 * 1024,
        le=64 * 1024 * 1024,
        description="Maximum decoded size for a single image frame passed to OpenRouter video generation.",
    )
    VIDEO_FRAME_TOTAL_MAX_BYTES: int = Field(
        default=50 * 1024 * 1024,
        ge=64 * 1024,
        le=128 * 1024 * 1024,
        description="Maximum combined decoded size for all image frames passed to one video generation request.",
    )
    VIDEO_FRAME_IMAGE_MIME_ALLOWLIST: str = Field(
        default="image/jpeg,image/png,image/webp",
        description="Comma-separated MIME allowlist for video generation frame images.",
    )
    VIDEO_OUTPUT_MIME_ALLOWLIST: str = Field(
        default="video/mp4,video/webm",
        description="Comma-separated MIME allowlist for generated video downloads after the format is identified from the downloaded file.",
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
            "Which Open WebUI global Task Model to use as the intent classifier. "
            "'internal' uses TASK_MODEL, 'external' uses TASK_MODEL_EXTERNAL."
        ),
    )
    VIDEO_INTENT_TASK_MODEL_FALLBACK: Literal["none", "other_task_model"] = Field(
        default="other_task_model",
        description=(
            "Fallback strategy when the chosen task model fails. 'none' disables "
            "fallback; 'other_task_model' switches between internal/external."
        ),
    )
    VIDEO_INTENT_SKIP_WHEN_EMPTY_CHAT: bool = Field(
        default=True,
        description=(
            "When True, skip the classifier on the very first turn of a fresh "
            "chat that has no attachments. There is nothing for the classifier "
            "to reference in that case (no prior video, no attached image), so "
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
            "Per-session cap on consecutive clarifying questions. 0 disables the "
            "clarification loop entirely (always proceeds with best-guess interpretation). "
            "Default 1 = at most one question, then it proceeds with its best guess."
        ),
    )
    VIDEO_INTENT_FRAME_EXTRACTION_INDEX: Literal["first", "last"] = Field(
        default="last",
        description=(
            "When extracting a frame from a prior video for image-to-video continuation, "
            "which frame to grab by default. 'last' matches 'continue this scene' intent."
        ),
    )
    VIDEO_INTENT_TIMEOUT_S: int = Field(
        default=8,
        ge=1,
        le=60,
        description=(
            "Hard timeout (seconds) for the classifier task-model call. If the "
            "call exceeds this or fails for any reason, the pipe falls back to "
            "sending only the latest user message to the video model. The paid "
            "video generation request still proceeds — the classifier never "
            "blocks generation."
        ),
    )
    VIDEO_INTENT_CONFIRM_MODE: Literal["always", "on_reference", "low_confidence", "never"] = Field(
        default="on_reference",
        description=(
            "When to show the confirmation footer under a generated video. 'always' = "
            "every generation; 'on_reference' (default) = only when reusing a prior video's frame "
            "or attached image; 'low_confidence' = only when classifier confidence is low; "
            "'never' = no confirmation."
        ),
    )
    VIDEO_INTENT_MAX_CALLS_PER_CHAT: int = Field(
        default=0,
        ge=0,
        description=(
            "Cost guard: maximum task-model calls per chat session. 0 (default) = unlimited. "
            "Admin sets a positive integer to enforce a per-chat ceiling."
        ),
    )
    VIDEO_INTENT_MAX_CALLS_PER_USER_DAY: int = Field(
        default=0,
        ge=0,
        description=(
            "Cost guard: maximum task-model calls per user per day. 0 (default) = unlimited. "
            "Admin sets a positive integer to enforce a per-user-per-day ceiling."
        ),
    )
    VIDEO_INTENT_LOG_DECISIONS: bool = Field(
        default=False,
        description=(
            "When True, log the per-turn intent classification summary (intent, "
            "confidence, detected language, frame counts, latency, and fallback/"
            "failure flags, with the chat id hashed) at INFO instead of DEBUG. The "
            "record is always written; this only raises its log level. It excludes "
            "the user's verbatim prompt and the model's free-text reason. Off by "
            "default to keep this prompt-derived metadata out of normal INFO logs."
        ),
    )
    AUTO_ATTACH_DIRECT_UPLOADS_FILTER: bool = Field(
        default=True,
        description=(
            "When enabled, automatically attaches the OpenRouter Direct Uploads toggleable filter to models that support "
            "at least one of OpenRouter direct file/audio/video inputs (so the switch appears in the Integrations menu only where it can work)."
        ),
    )
    AUTO_INSTALL_DIRECT_UPLOADS_FILTER: bool = Field(
        default=True,
        description=(
            "When enabled, automatically installs/updates the companion OpenRouter Direct Uploads filter function in Open WebUI. "
            "This is required for AUTO_ATTACH_DIRECT_UPLOADS_FILTER when the filter hasn't been installed manually."
        ),
    )

    # Provider Routing Filters
    ADMIN_PROVIDER_ROUTING_MODELS: str = Field(
        default="",
        description=(
            "Comma-separated list of model slugs (e.g., 'openai/gpt-4o, anthropic/claude-3.5-sonnet') for which "
            "to generate admin-only provider routing filters. These filters enforce provider preferences (order, "
            "only, ignore, sort, quantizations, etc.) that users cannot override or disable. "
            "Leave empty to disable admin provider routing filters."
        ),
    )
    USER_PROVIDER_ROUTING_MODELS: str = Field(
        default="",
        description=(
            "Comma-separated list of model slugs (e.g., 'meta-llama/llama-3.2-3b-instruct') for which "
            "to generate user-configurable provider routing filters. Users can toggle these filters per-chat "
            "and configure their own provider preferences in their per-user settings. "
            "Leave empty to disable user provider routing filters."
        ),
    )
    AUTO_DEFAULT_PROVIDER_ROUTING_FILTERS: bool = Field(
        default=True,
        description=(
            "Enable attached provider routing filters by default in new chats, so saved provider "
            "preferences apply without users having to switch the filter on per chat. The filter does "
            "nothing until preferences are actually configured, so defaulting it on is free. "
            "Disable to make users opt in per chat."
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
    ENABLE_REASONING: bool = Field(
        default=True,
        title="Show reasoning steps",
        description="While the AI works, show its step-by-step reasoning when supported.",
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
            "on Claude models that support it "
            "(including the `~anthropic/...` router aliases)."
        ),
    )
    REASONING_EFFORT: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = Field(
        default="medium",
        title="Reasoning depth",
        description="Choose how much thinking the AI should do before answering (higher depth is slower but more thorough). Use 'none' to disable reasoning or 'xhigh' for maximum depth when available.",
    )
    REASONING_SUMMARY_MODE: Literal["auto", "concise", "detailed", "disabled"] = Field(
        default="auto",
        title="Reasoning explanation detail",
        description="Pick how detailed the reasoning summary should be (auto, concise, detailed, or hidden).",
    )
    PERSIST_REASONING_TOKENS: Literal["disabled", "next_reply", "conversation"] = Field(
        default="next_reply",
        title="How long to keep reasoning",
        description="Choose whether reasoning is kept just for the next reply or the entire conversation.",
    )
    PERSIST_TOOL_RESULTS: bool = Field(
        default=False,
        title="Remember tool and search results",
        description="Let the AI reuse outputs from tools (for example web searches or other apps) later in the conversation. Uses more tokens on long chats; when off, the AI relies on its own summaries and can re-run tools as needed.",
    )
    TOOL_EXECUTION_MODE: Literal["Pipeline", "Open-WebUI"] = Field(
        default="Pipeline",
        title="Tool execution mode",
        description=(
            "Where to execute tools. 'Pipeline' executes tool calls inside this pipe. "
            "'Open-WebUI' hands tool calls to Open WebUI to run instead."
        ),
    )
    SHOW_TOOL_CARDS: bool = Field(
        default=False,
        title="Show tool execution cards",
        description="Show collapsible cards in chat with tool name, arguments, and results. When disabled, tools run silently.",
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
    rejected: list[str] = []
    for _ in range(len(candidate) + 1):
        try:
            return model.model_validate(candidate), rejected
        except ValidationError as exc:
            bad = {
                str(err["loc"][0])
                for err in exc.errors()
                if err.get("loc") and str(err["loc"][0]) in candidate
            }
            if not bad:
                return model(), sorted(set(rejected) | set(candidate))
            rejected.extend(sorted(bad))
            for name in bad:
                candidate.pop(name, None)
    return model(), sorted(set(rejected))


def _select_openrouter_http_referer(valves: Any | None) -> str:
    """Select HTTP referer for OpenRouter requests, with optional valve override."""
    override = valves.HTTP_REFERER_OVERRIDE if valves else ""
    if override and override.startswith(("http://", "https://")):
        return override
    return _OPENROUTER_REFERER


def _apply_owui_forward_user_headers(headers: dict, user: Any, chat_id: Any = None) -> dict:
    """Stamp Open WebUI user-identity headers (and Chat-Id) on an outbound request, matching a native OWUI connection; does nothing unless Open WebUI is present with ENABLE_FORWARD_USER_INFO_HEADERS set."""
    if _owui_env is None or _owui_include_user_info_headers is None:
        return headers
    if not getattr(_owui_env, "ENABLE_FORWARD_USER_INFO_HEADERS", False):
        return headers
    if user is None or isinstance(user, dict):
        return headers
    try:
        headers = _owui_include_user_info_headers(headers, user)
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
