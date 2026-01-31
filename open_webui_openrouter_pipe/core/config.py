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
from typing import Any, Literal, Optional, cast

from cryptography.fernet import Fernet, InvalidToken
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import core_schema
from pydantic import GetCoreSchemaHandler
from ..core.timing_logger import timed

LOGGER = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

_OPENROUTER_TITLE = "Open WebUI plugin for OpenRouter Responses API"
_OPENROUTER_REFERER = "https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe/"
_DEFAULT_PIPE_ID = "open_webui_openrouter_pipe"
_FUNCTION_MODULE_PREFIX = "function_"
_OPENROUTER_FRONTEND_MODELS_URL = "https://openrouter.ai/api/frontend/models"
_OPENROUTER_SITE_URL = "https://openrouter.ai"
_MAX_MODEL_PROFILE_IMAGE_BYTES = 2 * 1024 * 1024
_MAX_OPENROUTER_ID_CHARS = 128
_MAX_OPENROUTER_METADATA_PAIRS = 16
_MAX_OPENROUTER_METADATA_KEY_CHARS = 64
_MAX_OPENROUTER_METADATA_VALUE_CHARS = 512

_ORS_FILTER_MARKER = "openrouter_pipe:ors_filter:v1"
_ORS_FILTER_FEATURE_FLAG = "openrouter_web_search"
_ORS_FILTER_PREFERRED_FUNCTION_ID = "openrouter_search"

_DIRECT_UPLOADS_FILTER_MARKER = "openrouter_pipe:direct_uploads_filter:v1"
_DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID = "openrouter_direct_uploads"

_PROVIDER_ROUTING_FILTER_MARKER_PREFIX = "openrouter_pipe:provider_routing:"
_PROVIDER_ROUTING_FILTER_MARKER_VERSION = "v1"
_PROVIDER_ROUTING_FILTER_ID_PREFIX = "openrouter_provider_"

_NON_REPLAYABLE_TOOL_ARTIFACTS = frozenset(
    {
        "image_generation_call",
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
_TEMPLATE_IF_TOKEN_RE = re.compile(r"\{\{\s*(#if\s+(\w+)|/if)\s*\}\}")

# ULID generation constants
ULID_LENGTH = 20
ULID_TIME_LENGTH = 16
ULID_RANDOM_LENGTH = ULID_LENGTH - ULID_TIME_LENGTH
CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ULID_TIME_MASK = (1 << (ULID_TIME_LENGTH * 5)) - 1

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
    "Adjust your prompt or requested output to stay within these limits.\n"
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
    "Please adjust the request and try again, or ask your admin to enable the middle-out option.\n"
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
    "OpenRouter's servers are experiencing issues.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if status_code}}\n"
    "**Status:** {status_code} {reason}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "This is **not** a problem with your request. The issue is on OpenRouter's side.\n\n"
    "**What to do:**\n"
    "- Wait a few minutes and try again\n"
    "- Check [OpenRouter Status](https://status.openrouter.ai/) for updates\n"
    "- If the problem persists for more than 15 minutes, contact OpenRouter support\n"
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
    "OpenRouter rejected your credentials.\n\n"
    "**Error ID:** `{error_id}`\n"
    "{{#if openrouter_code}}\n"
    "**Status:** {openrouter_code}\n"
    "{{/if}}\n"
    "{{#if openrouter_message}}\n"
    "**Details:** {openrouter_message}\n"
    "{{/if}}\n"
    "{{#if timestamp}}\n"
    "**Time:** {timestamp}\n"
    "{{/if}}\n\n"
    "**What to do:**\n"
    "1. Verify the API key configured for this pipe\n"
    "2. Generate a new key at https://openrouter.ai/keys if needed\n"
    "3. If using OAuth, re-authenticate your session\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE = (
    "### 💳 Insufficient Credits\n\n"
    "OpenRouter could not run this request because the account is out of credits.\n\n"
    "**Error ID:** `{error_id}`\n"
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
    "- Review usage at https://openrouter.ai/usage\n"
    "- Consider enabling auto-recharge to avoid interruptions\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_RATE_LIMIT_TEMPLATE = (
    "### ⏸️ Rate Limit Exceeded\n\n"
    "OpenRouter is protecting the service because too many requests were made quickly.\n\n"
    "**Error ID:** `{error_id}`\n"
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
    "- Back off and retry with exponential delays\n"
    "- Queue requests or lower parallelism\n"
    "- Contact OpenRouter if you need higher limits\n"
    "{{#if support_email}}\n"
    "\n**Support:** {support_email}\n"
    "{{/if}}\n"
)

DEFAULT_SERVER_TIMEOUT_TEMPLATE = (
    "### 🕒 OpenRouter Timed Out\n\n"
    "OpenRouter started the request but couldn't finish within its timeout window.\n\n"
    "**Error ID:** `{error_id}`\n"
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

DEFAULT_MAX_FUNCTION_CALL_LOOPS_REACHED_TEMPLATE = (
    "### 🧰 Tool step limit reached\n\n"
    "This chat needed more tool rounds than allowed, so it stopped early to prevent infinite loops.\n\n"
    "{{#if max_function_call_loops}}\n"
    "**Current limit:** `{max_function_call_loops}`\n"
    "{{/if}}\n\n"
    "**What you can do:**\n"
    "- Increase the *Max Function Call Loops* valve (e.g. 25-50) and retry\n"
    "- Simplify the request or reduce tool chaining\n"
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
    "- **MODEL_ID**: `{model_id_filter}`\n"
    "{{/if}}\n"
    "{{#if free_model_filter}}\n"
    "- **FREE_MODEL_FILTER**: `{free_model_filter}`\n"
    "{{/if}}\n"
    "{{#if tool_calling_filter}}\n"
    "- **TOOL_CALLING_FILTER**: `{tool_calling_filter}`\n"
    "{{/if}}\n\n"
    "Choose an allowed model or ask your admin to update the pipe filters.\n"
)



# -----------------------------------------------------------------------------
# EncryptedStr and Helper Functions
# -----------------------------------------------------------------------------

class EncryptedStr(str):
    """String wrapper that automatically encrypts/decrypts valve values."""

    _ENCRYPTION_PREFIX = "encrypted:"

    @classmethod
    @timed
    def _get_encryption_key(cls) -> Optional[bytes]:
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
    @timed
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
    @timed
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
            # Invalid encryption key or corrupted data - return original value
            LOGGER.warning("Failed to decrypt value: invalid token or key mismatch")
            return value
        except (ValueError, UnicodeDecodeError) as e:
            # Decoding or encoding error - return original value
            LOGGER.warning(f"Failed to decrypt value: {type(e).__name__}: {e}")
            return value

    @classmethod
    @timed
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


@timed
def _default_api_key() -> EncryptedStr:
    """Return the API key env default as EncryptedStr."""
    return EncryptedStr((os.getenv("OPENROUTER_API_KEY") or "").strip())


@timed
def _default_artifact_encryption_key() -> EncryptedStr:
    """Provide an EncryptedStr placeholder for artifact encryption."""
    return EncryptedStr("")


@timed
def _resolve_log_level_default() -> Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
    """Normalize env-provided log level to the allowed literal set."""
    value = (os.getenv("GLOBAL_LOG_LEVEL") or "INFO").strip().upper()
    if value not in _ALLOWED_LOG_LEVELS:
        value = "INFO"
    return cast(Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], value)


@timed
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


# Runtime pipe ID (computed at module import time)
_PIPE_RUNTIME_ID = _detect_runtime_pipe_id()

# -----------------------------------------------------------------------------
# Valves and UserValves Configuration Classes
# -----------------------------------------------------------------------------

class Valves(BaseModel):
    """Global valve configuration shared across sessions."""
    # Connection & Auth
    BASE_URL: str = Field(
        default=((os.getenv("OPENROUTER_API_BASE_URL") or "").strip() or "https://openrouter.ai/api/v1"),
        description="OpenRouter API base URL. Override this if you are using a gateway or proxy.",
    )
    DEFAULT_LLM_ENDPOINT: Literal["responses", "chat_completions"] = Field(
        default="responses",
        description=(
            "Which OpenRouter endpoint to use by default. "
            "`responses` uses /responses (best feature coverage). "
            "`chat_completions` uses /chat/completions (needed for some providers/features like cache_control breakpoints on Anthropic)."
        ),
    )
    FORCE_CHAT_COMPLETIONS_MODELS: str = Field(
        default="",
        description=(
            "Comma-separated glob patterns of model ids that must use /chat/completions "
            "(e.g. 'anthropic/*, openai/gpt-4.1-mini'). Matches both slash and dotted model ids."
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
    HTTP_TOTAL_TIMEOUT_SECONDS: Optional[int] = Field(
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
        description="When True, remote URLs and data URLs in the file_url field are downloaded/parsed and re-hosted in Open WebUI storage. When False, file_url values pass through untouched. Note: This valve only affects the file_url field; see SAVE_FILE_DATA_CONTENT for file_data behavior. Recommended: Keep disabled to avoid unexpected storage growth.",
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
        description="Maximum size in MB for video files (remote URLs or data URLs). Videos exceeding this limit will be rejected to prevent memory and bandwidth issues.",
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

    # Models
    MODEL_ID: str = Field(
        default="auto",
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
            "Default access_control for new OpenRouter model overlays inserted into Open WebUI. "
            "'public' sets access_control=None (any user can read). "
            "'admins' sets access_control={} (private) and relies on Open WebUI's "
            "BYPASS_ADMIN_ACCESS_CONTROL for admin access; otherwise admins must be granted access explicitly. "
            "Applies only on insert; existing access_control values are preserved."
        ),
    )
    FREE_MODEL_FILTER: Literal["all", "only", "exclude"] = Field(
        default="all",
        title="Free model filter",
        description=(
            "Filter models based on OpenRouter pricing totals. "
            "'all' disables filtering. "
            "'only' restricts to models whose summed pricing fields equal 0. "
            "'exclude' hides those free models."
        ),
    )
    TOOL_CALLING_FILTER: Literal["all", "only", "exclude"] = Field(
        default="all",
        title="Tool calling filter",
        description=(
            "Filter models based on tool-calling capability (supported_parameters includes 'tools' or 'tool_choice'). "
            "'all' disables filtering. "
            "'only' restricts to tool-capable models. "
            "'exclude' hides tool-capable models."
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
            "'status' emits thinking as status events only; "
            "'both' enables both outputs."
        ),
    )
    ENABLE_ANTHROPIC_INTERLEAVED_THINKING: bool = Field(
        default=True,
        title="Anthropic interleaved thinking",
        description=(
            "When True, enables Claude's interleaved thinking mode by sending "
            "`x-anthropic-beta: interleaved-thinking-2025-05-14` for `anthropic/...` models."
        ),
    )
    ENABLE_ANTHROPIC_PROMPT_CACHING: bool = Field(
        default=True,
        title="Anthropic prompt caching",
        description=(
            "When True and the selected model is `anthropic/...`, insert `cache_control` breakpoints into "
            "system/user text blocks to enable Claude prompt caching and reduce per-turn costs for large "
            "stable prefixes (system prompts, tools, RAG context)."
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
    AUTO_CONTEXT_TRIMMING: bool = Field(
        default=True,
        title="Auto context trimming",
        description=(
            "When enabled, automatically attaches OpenRouter's `middle-out` transform so long prompts "
            "are trimmed from the middle instead of failing with context errors. Disable if your deployment "
            "manages `transforms` manually."
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
            "Reasoning effort requested for Open WebUI task payloads (titles, tags, etc.) when they target this pipe's models. "
            "Low is the default balance between speed and quality; set to 'minimal' to prioritize fastest runs (and disable auto web-search), "
            "or use medium/high for progressively deeper background reasoning at higher cost."
        ),
    )

    # Tool execution behavior
    TOOL_EXECUTION_MODE: Literal["Pipeline", "Open-WebUI"] = Field(
        default="Pipeline",
        title="Tool execution mode",
        description=(
            "Where to execute tools. 'Pipeline' executes tool calls inside this pipe "
            "(batching/breakers/special backends). 'Open-WebUI' bypasses the internal executor and "
            "passes tool calls through so Open WebUI executes them and renders the native tool UI."
        ),
    )
    SHOW_TOOL_CARDS: bool = Field(
        default=False,
        title="Show tool execution cards",
        description="Show collapsible cards in chat with tool name, arguments, and results. When disabled, tools run silently without visual status indicators.",
    )
    PERSIST_TOOL_RESULTS: bool = Field(
        default=True,
        title="Keep tool results",
        description="Persist tool call results across conversation turns. When disabled, tool results stay ephemeral.",
    )
    ARTIFACT_ENCRYPTION_KEY: EncryptedStr = Field(
        default_factory=_default_artifact_encryption_key,
        description="Min 16 chars. Encrypt reasoning tokens (and optionally all persisted artifacts). Changing the key creates a new table; prior artifacts become inaccessible.",
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
            "Maximum number of full execution cycles (loops) allowed per request. "
            "Each loop involves the model generating one or more function/tool calls, "
            "executing all requested functions, and feeding the results back into the model. "
            "Looping stops when this limit is reached or when the model no longer requests "
            "additional tool or function calls."
        )
    )

    # Web search
    WEB_SEARCH_MAX_RESULTS: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of web results to request when the web-search plugin is enabled (1-10). Set to null to use the provider default.",
    )

    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default_factory=_resolve_log_level_default,
        description="Select logging level.  Recommend INFO or WARNING for production use. DEBUG is useful for development and debugging.",
    )
    SESSION_LOG_STORE_ENABLED: bool = Field(
        default=False,
        description=(
            "When True, persist per-request SessionLogger output to encrypted zip files on disk. "
            "Persistence is skipped when any required IDs are missing (user_id, session_id, chat_id, message_id)."
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
            "Password used to encrypt session log zip files (pyzipper AES). "
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
    SESSION_LOG_ZIP_COMPRESSLEVEL: Optional[int] = Field(
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
        description="Maximum number of in-memory SessionLogger records retained per request (older entries are dropped).",
    )
    SESSION_LOG_FORMAT: Literal["jsonl", "text", "both"] = Field(
        default="jsonl",
        description=(
            "Format written inside session log archives. "
            "'jsonl' writes logs.jsonl (one JSON object per log record). "
            "'text' writes logs.txt (plain text). "
            "'both' writes both files."
        ),
    )
    SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS: int = Field(
        default=30,
        ge=1,
        description="How often (in seconds) to scan staged DB session-log segments and assemble per-message zip archives.",
    )
    SESSION_LOG_ASSEMBLER_JITTER_SECONDS: int = Field(
        default=10,
        ge=0,
        description="Random jitter (0..N seconds) added to assembler sleeps so multiple workers do not run in lockstep.",
    )
    SESSION_LOG_ASSEMBLER_BATCH_SIZE: int = Field(
        default=25,
        ge=1,
        le=500,
        description="Maximum number of message bundles to assemble per assembler cycle.",
    )
    SESSION_LOG_STALE_FINALIZE_SECONDS: int = Field(
        default=6 * 7200,
        ge=60,
        description=(
            "If a message has staged session-log segments but never reaches a terminal state "
            "(crash/kill), finalize an incomplete zip after this many seconds since the last segment."
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
            "When True, capture function entrance/exit timing data. "
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
        description="Number of per-request SSE worker tasks that parse streamed chunks.",
    )
    STREAMING_CHUNK_QUEUE_MAXSIZE: int = Field(
        default=0,
        ge=0,
        description="Maximum number of raw SSE chunks buffered before applying backpressure to the OpenRouter stream. 0=unbounded (deadlock-proof, recommended); bounded values &lt;500 risk hangs on tool-heavy loads or slow DB/emit (drain block -> event full -> workers block -> chunk full -> producer halt).",
    )
    STREAMING_EVENT_QUEUE_MAXSIZE: int = Field(
        default=0,
        ge=0,
        description="Maximum number of parsed SSE events buffered ahead of downstream processing. 0=unbounded (deadlock-proof, recommended); bounded values &lt;500 risk hangs on tool-heavy loads or slow DB/emit (drain block -> event full -> workers block -> chunk full -> producer halt).",

    )
    STREAMING_EVENT_QUEUE_WARN_SIZE: int = Field(
        default=1000,
        ge=100,
        description="Log warning when event_queue.qsize() hits this threshold (unbounded queue monitoring); ge=100 avoids spam on sustained high load. Tune higher for noisy envs.",
    )
    MIDDLEWARE_STREAM_QUEUE_MAXSIZE: int = Field(
        default=0,
        ge=0,
        description=(
            "Maximum number of per-request items buffered for the Open WebUI middleware streaming bridge. "
            "0=unbounded (default behavior)."
        ),
    )
    MIDDLEWARE_STREAM_QUEUE_PUT_TIMEOUT_SECONDS: float = Field(
        default=1.0,
        ge=0,
        description=(
            "When MIDDLEWARE_STREAM_QUEUE_MAXSIZE>0, maximum seconds to wait while enqueueing a stream item before aborting the request. "
            "0 disables the timeout (not recommended; a stalled client can hang producers)."
        ),
    )
    OPENROUTER_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_OPENROUTER_ERROR_TEMPLATE,
        description=(
            "Markdown template used when OpenRouter rejects a request with status 400. "
            "Placeholders such as {heading}, {detail}, {sanitized_detail}, {provider}, {model_identifier}, "
            "{requested_model}, {api_model_id}, {normalized_model_id}, {openrouter_code}, {upstream_type}, "
            "{reason}, {request_id}, {request_id_reference}, {openrouter_message}, {upstream_message}, "
            "{moderation_reasons}, {flagged_excerpt}, {raw_body}, {context_limit_tokens}, {max_output_tokens}, "
            "{include_model_limits}, {metadata_json}, {provider_raw_json}, {error_id}, {timestamp}, {session_id}, {user_id}, "
            "{native_finish_reason}, {error_chunk_id}, {error_chunk_created}, {streaming_provider}, {streaming_model}, "
            "{retry_after_seconds}, {rate_limit_type}, {required_cost}, and {account_balance} are replaced when values are available. "
            "Lines containing placeholders are omitted automatically when the referenced value is missing or empty. "
            "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to render only when truthy."
        ),
    )
    ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE: str = Field(
        default=DEFAULT_ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE,
        description=(
            "Markdown template used when a request requires /chat/completions (e.g. direct video uploads) but the model is "
            "explicitly forced to /responses by endpoint override valves (or vice versa)."
        ),
    )
    DIRECT_UPLOAD_FAILURE_TEMPLATE: str = Field(
        default=DEFAULT_DIRECT_UPLOAD_FAILURE_TEMPLATE,
        description=(
            "Markdown template used when OpenRouter Direct Uploads cannot be applied (e.g. incompatible attachment combinations, "
            "missing storage objects, or other pre-flight validation failures)."
        ),
    )

    AUTHENTICATION_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_AUTHENTICATION_ERROR_TEMPLATE,
        description=(
            "Markdown template for HTTP 401 errors. Available placeholders include {error_id}, {timestamp}, {openrouter_code}, {openrouter_message}, "
            "{session_id}, {user_id}, {support_email}, {support_url}, and any metadata extracted from the OpenRouter payload."
        ),
    )

    INSUFFICIENT_CREDITS_TEMPLATE: str = Field(
        default=DEFAULT_INSUFFICIENT_CREDITS_TEMPLATE,
        description=(
            "Markdown template for HTTP 402 errors when the account is out of credits. Supports {error_id}, {timestamp}, {openrouter_code}, "
            "{openrouter_message}, {required_cost}, {account_balance}, {support_email}, and other shared context variables."
        ),
    )

    RATE_LIMIT_TEMPLATE: str = Field(
        default=DEFAULT_RATE_LIMIT_TEMPLATE,
        description=(
            "Markdown template for HTTP 429 rate-limit errors. Use placeholders such as {error_id}, {timestamp}, {openrouter_code}, {retry_after_seconds}, "
            "{rate_limit_type}, {support_email}, and the standard context variables."
        ),
    )

    SERVER_TIMEOUT_TEMPLATE: str = Field(
        default=DEFAULT_SERVER_TIMEOUT_TEMPLATE,
        description=(
            "Markdown template for HTTP 408 errors returned by OpenRouter (server-side timeout). Supports the common context variables plus "
            "{openrouter_message}, {openrouter_code}, and support contact placeholders."
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
            "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to render only when truthy."
        )
    )

    CONNECTION_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_CONNECTION_ERROR_TEMPLATE,
        description=(
            "Markdown template for connection failures. "
            "Available variables: {error_id}, {error_type}, {timestamp}, "
            "{session_id}, {user_id}, {support_email}. "
            "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to render only when truthy."
        )
    )

    SERVICE_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_SERVICE_ERROR_TEMPLATE,
        description=(
            "Markdown template for OpenRouter 5xx errors. "
            "Available variables: {error_id}, {status_code}, {reason}, {timestamp}, "
            "{session_id}, {user_id}, {support_email}. "
            "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to render only when truthy."
        )
    )

    INTERNAL_ERROR_TEMPLATE: str = Field(
        default=DEFAULT_INTERNAL_ERROR_TEMPLATE,
        description=(
            "Markdown template for unexpected internal errors. "
            "Available variables: {error_id}, {error_type}, {timestamp}, "
            "{session_id}, {user_id}, {support_email}, {support_url}. "
            "Supports Handlebars-style conditionals: wrap sections in {{#if variable}}...{{/if}} to render only when truthy."
        )
    )

    MAX_FUNCTION_CALL_LOOPS_REACHED_TEMPLATE: str = Field(
        default=DEFAULT_MAX_FUNCTION_CALL_LOOPS_REACHED_TEMPLATE,
        description=(
            "Markdown template emitted when a request reaches MAX_FUNCTION_CALL_LOOPS while the model is still "
            "requesting additional tool/function calls. Available variables: {max_function_call_loops}."
        ),
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
            "Number of failures allowed per breaker window before requests, tools, or DB ops are temporarily blocked. "
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
            "Maximum failures remembered per user/tool breaker. Increase when using very high BREAKER_MAX_FAILURES so history is not truncated."
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
            "tool outputs pruned to save tokens. Set to 0 to keep every tool output."
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
    TOOL_IDLE_TIMEOUT_SECONDS: Optional[int] = Field(
        default=None,
        ge=1,
        description="Idle timeout (seconds) between tool executions in a queue. Set to null for unlimited idle time so intermittent tool usage does not fail unexpectedly.",
    )
    TOOL_SHUTDOWN_TIMEOUT_SECONDS: float = Field(
        default=10.0,
        ge=0,
        description=(
            "Maximum seconds to wait for per-request tool workers to drain/stop during request cleanup. "
            "0 disables the graceful wait and cancels workers immediately."
        ),
    )
    ENABLE_REDIS_CACHE: bool = Field(
        default=True,
        description="Enable Redis write-behind cache when REDIS_URL + multi-worker detected.",
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
        description="Disable Redis caching after this many consecutive flush failures (falls back to direct DB writes).",
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
        description="Retention window (days) for artifact cleanup scheduler.",
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
        description="When enabled, automatically include the provider's max_output_tokens in each request. Disable to omit the parameter entirely.",
    )
    SHOW_FINAL_USAGE_STATUS: bool = Field(
        default=True,
        description="When True, the final status message includes elapsed time, cost, and token usage.",
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
    ENABLE_STATUS_CSS_PATCH: bool = Field(
        default=True,
        description="When True, injects a CSS tweak via __event_call__ to show multi-line status descriptions in Open WebUI (experimental).",
    )
    SEND_END_USER_ID: bool = Field(
        default=False,
        description="When True, send OpenRouter `user` using the OWUI user GUID, and also include `metadata.user_id`.",
    )
    SEND_SESSION_ID: bool = Field(
        default=False,
        description="When True, send OpenRouter `session_id` using OWUI metadata and also include `metadata.session_id`.",
    )
    SEND_CHAT_ID: bool = Field(
        default=False,
        description="When True, include OWUI chat_id as `metadata.chat_id` (metadata only).",
    )
    SEND_MESSAGE_ID: bool = Field(
        default=False,
        description="When True, include OWUI message_id as `metadata.message_id` (metadata only).",
    )
    MAX_INPUT_IMAGES_PER_REQUEST: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of image inputs (user attachments plus assistant fallbacks) to include in a single provider request.",
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
    UPDATE_MODEL_DESCRIPTIONS: bool = Field(
        default=False,
        description=(
            "When enabled, automatically sync model descriptions from OpenRouter's API catalog to Open WebUI model metadata. "
            "Disable to manage model descriptions manually (or set per-model disable_description_updates)."
        ),
    )
    AUTO_ATTACH_ORS_FILTER: bool = Field(
        default=True,
        description=(
            "When enabled, automatically attaches the OpenRouter Search toggleable filter to models that support "
            "OpenRouter native web search (so the OpenRouter Search switch appears in the Integrations menu only where it works)."
        ),
    )
    AUTO_INSTALL_ORS_FILTER: bool = Field(
        default=True,
        description=(
            "When enabled, automatically installs/updates the companion OpenRouter Search filter function in Open WebUI. "
            "This is required for AUTO_ATTACH_ORS_FILTER when the filter hasn't been installed manually."
        ),
    )
    AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER: bool = Field(
        default=True,
        description=(
            "When enabled, automatically marks OpenRouter Search as a Default Filter on models that support OpenRouter native "
            "web search (by updating the model's meta.defaultFilterIds). This replicates \"web search enabled by default\" "
            "behavior while still allowing operators/users to turn it off per model or per chat."
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
            "and configure their own provider preferences via UserValves. "
            "Leave empty to disable user provider routing filters."
        ),
    )


class UserValves(BaseModel):
    """Per-user valve overrides."""

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="before")
    @classmethod
    @timed
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
                    normalized[key] = None
                    continue
            normalized[key] = val
        return normalized

    SHOW_FINAL_USAGE_STATUS: bool = Field(
        default=True,
        title="Show usage details",
        description="Display tokens, time, and cost at the end of each reply.",
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
            "`x-anthropic-beta: interleaved-thinking-2025-05-14` for `anthropic/...` models."
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
        default=True,
        title="Remember tool and search results",
        description="Let the AI reuse outputs from tools (for example web searches or other apps) later in the conversation.",
    )
    TOOL_EXECUTION_MODE: Literal["Pipeline", "Open-WebUI"] = Field(
        default="Pipeline",
        title="Tool execution mode",
        description=(
            "Where to execute tools. 'Pipeline' executes tool calls inside this pipe. "
            "'Open-WebUI' bypasses the internal executor and lets Open WebUI run tools."
        ),
    )
    SHOW_TOOL_CARDS: bool = Field(
        default=False,
        title="Show tool execution cards",
        description="Show collapsible cards in chat with tool name, arguments, and results. When disabled, tools run silently.",
    )


@timed
def _select_openrouter_http_referer(valves: Any | None) -> str:
    """Select HTTP referer for OpenRouter requests, with optional valve override."""
    override = (getattr(valves, "HTTP_REFERER_OVERRIDE", "") or "").strip()
    if override:
        if override.startswith(("http://", "https://")):
            return override
    return _OPENROUTER_REFERER
