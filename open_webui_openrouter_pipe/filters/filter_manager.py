"""Filter management for OWUI filter functions.

This module manages three types of filter functions:
1. OpenRouter Search (ORS) - Enables OpenRouter's native web search
2. Direct Uploads - Bypasses OWUI RAG for file uploads
3. Provider Routing - Per-model provider/quantization preferences

FilterManager handles:
- Generating filter source code (static methods)
- Installing/updating filters in OWUI Functions table (instance methods)
- Security sanitization for code generation (static methods)
"""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import logging
import re
from typing import TYPE_CHECKING, Any

from ..core.timing_logger import timed
from ..core.config import (
    _ORS_FILTER_MARKER,
    _ORS_FILTER_FEATURE_FLAG,
    _ORS_FILTER_PREFERRED_FUNCTION_ID,
    _DIRECT_UPLOADS_FILTER_MARKER,
    _DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID,
    _PROVIDER_ROUTING_FILTER_MARKER_PREFIX,
    _PROVIDER_ROUTING_FILTER_MARKER_VERSION,
    _PROVIDER_ROUTING_FILTER_ID_PREFIX,
    _PROVIDER_SLUG_PATTERN,
)

# Security: Quantization level validation (alphanumeric + underscore/hyphen)
# Used for validating quantization values like "int4", "int8", "fp16", "bf16"
_QUANTIZATION_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

if TYPE_CHECKING:
    from ..pipe import Pipe

# Regex patterns for provider name validation
_PROVIDER_NAME_ALLOWLIST_RE = re.compile(r"[^A-Za-z0-9 \-_.]")
_PROVIDER_NAME_COLLAPSE_RE = re.compile(r"[ _]{2,}")


class FilterManager:
    """Manages OWUI filter functions for the OpenRouter pipe.

    Static methods handle code generation and security sanitization.
    Instance methods handle filter installation/updates in OWUI.
    """

    # Class-level state hash to prevent unnecessary filter regeneration
    _provider_routing_state_hash: str = ""

    def __init__(
        self,
        pipe: "Pipe",
        valves: Any,
        logger: logging.Logger,
    ) -> None:
        """Initialize FilterManager.

        Args:
            pipe: Reference to parent Pipe instance
            valves: Pipe valves configuration
            logger: Logger instance for this manager
        """
        self._pipe = pipe
        self.valves = valves
        self.logger = logger

    # =========================================================================
    # SECURITY/UTILITY METHODS (Static)
    # =========================================================================

    @staticmethod
    def safe_literal_string(s: str) -> str:
        """Safely escape a string for use in Python Literal type annotations.

        Uses Python's built-in repr() to produce a safely escaped string
        representation that can be interpolated into generated Python source
        code without risk of code injection.

        Security Purpose:
            Provider names from the OpenRouter API are untrusted external input.
            When interpolating these into generated filter source code (particularly
            in Literal['...'] type annotations), malicious input like:
                "foo', None); import os; os.system('rm -rf /'); x = ('"
            could escape the string context and execute arbitrary code.

            Using repr() produces properly escaped output that remains a safe,
            quoted string literal.

        Args:
            s: The raw string to escape (e.g., provider name, model slug).

        Returns:
            A repr()-escaped string safe for interpolation into Python source.
            The returned string includes surrounding quotes.

        Example:
            >>> FilterManager.safe_literal_string("Amazon Bedrock")
            "'Amazon Bedrock'"
            >>> FilterManager.safe_literal_string("evil'; import os; #")
            "\"evil'; import os; #\""
        """
        if not isinstance(s, str):
            s = str(s) if s is not None else ""
        return repr(s)

    @staticmethod
    def validate_provider_name(name: str, slug: str = "") -> str:
        """Validate and sanitize a provider name for safe use in generated code.

        Enforces strict character allowlists and length limits on provider names
        to prevent injection attacks and ensure predictable behavior in generated
        filter source code.

        Security Purpose:
            Provider names from external APIs may contain unexpected characters
            that could:
            1. Break Python syntax when interpolated into source code
            2. Exploit edge cases in string parsing
            3. Cause display issues in the UI
            4. Enable homograph attacks with Unicode lookalikes

            This function sanitizes to ASCII alphanumeric + limited punctuation,
            and truncates long names with a hash suffix to maintain uniqueness.

        Args:
            name: The raw provider name from the API.
            slug: Optional provider slug for hash uniqueness when truncating.

        Returns:
            A sanitized provider name safe for use in generated code.
            - Only ASCII letters, digits, spaces, hyphens, underscores, periods
            - Maximum 64 characters (truncated with hash suffix if longer)
            - Empty/whitespace-only input returns "Unknown"

        Example:
            >>> FilterManager.validate_provider_name("Amazon Bedrock")
            'Amazon Bedrock'
            >>> FilterManager.validate_provider_name("Evil<script>Provider")
            'EvilscriptProvider'
        """
        # Handle empty/None input
        if not name or not isinstance(name, str):
            return "Unknown"

        # Strip leading/trailing whitespace
        cleaned = name.strip()
        if not cleaned:
            return "Unknown"

        # Remove any characters not in our allowlist (use pre-compiled pattern)
        # Allow: ASCII letters, digits, space, hyphen, underscore, period
        cleaned = _PROVIDER_NAME_ALLOWLIST_RE.sub("", cleaned)

        # Collapse multiple spaces/underscores (use pre-compiled pattern)
        cleaned = _PROVIDER_NAME_COLLAPSE_RE.sub(" ", cleaned)
        cleaned = cleaned.strip()

        if not cleaned:
            return "Unknown"

        # Enforce maximum length with hash suffix for uniqueness
        max_length = 64
        if len(cleaned) > max_length:
            # Use slug for hash if provided, otherwise use the cleaned name itself
            hash_source = slug if slug else cleaned
            hash_suffix = hashlib.md5(hash_source.encode("utf-8")).hexdigest()[:8]
            # Truncate to leave room for underscore + 8-char hash
            truncated = cleaned[: max_length - 9].rstrip(" _-.")
            # Handle edge case: if truncated is empty after rstrip, use fallback
            if not truncated:
                cleaned = f"Provider_{hash_suffix}"
            else:
                cleaned = f"{truncated}_{hash_suffix}"

        return cleaned

    @staticmethod
    def sanitize_model_for_filter_id(model_slug: str) -> str:
        """Convert model slug to safe filter ID component.

        Example: 'openai/gpt-4o' -> 'openai_gpt_4o'
        """
        return model_slug.replace("/", "_").replace("-", "_").replace(".", "_")

    @staticmethod
    def validate_filter_source(source: str) -> tuple[bool, str | None]:
        """Validate generated Python source code for syntactic correctness.

        Uses Python's ast.parse() to verify that generated filter source code
        is syntactically valid before it is stored and potentially executed
        by Open WebUI.

        Security Purpose:
            This serves as a defense-in-depth measure. Even if string escaping
            and validation functions work correctly, this provides a final
            safety check that:
            1. The generated code is valid Python syntax
            2. No injection attacks have produced malformed code
            3. Template rendering hasn't introduced syntax errors

        Note:
            This validates SYNTAX only, not semantic safety. It cannot detect
            syntactically valid but malicious code. The earlier sanitization
            functions (safe_literal_string, validate_provider_name) are the
            primary defense against injection.

        Args:
            source: The complete generated Python source code to validate.

        Returns:
            A tuple of (is_valid, error_message):
            - (True, None) if the source is syntactically valid
            - (False, error_description) if parsing fails

        Example:
            >>> FilterManager.validate_filter_source("x = 1")
            (True, None)
            >>> FilterManager.validate_filter_source("x = ")
            (False, "Line 1: unexpected EOF while parsing")
        """
        if not source or not isinstance(source, str):
            return False, "Empty or invalid source"

        try:
            ast.parse(source)
            return True, None
        except SyntaxError as e:
            # Include line number and message for debugging
            # Note: Use str(e) rather than e.msg for cross-version compatibility
            if e.lineno:
                error_msg = f"Line {e.lineno}: {e}"
            else:
                error_msg = str(e)
            return False, error_msg
        except Exception as e:
            # Catch any other parsing errors
            return False, f"Parse error: {str(e)}"

    # =========================================================================
    # ORS FILTER (OpenRouter Search)
    # =========================================================================

    @staticmethod
    def render_ors_filter_source() -> str:
        """Return the canonical OWUI filter source for the OpenRouter Search toggle."""
        # NOTE: This file is inserted into Open WebUI's Functions table as a *filter* function.
        # It must not depend on this pipe module at runtime.
        return f'''"""
title: OpenRouter Search
author: Open-WebUI-OpenRouter-pipe
	author_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
	id: openrouter_search
	description: Enables OpenRouter's web-search plugin for the OpenRouter pipe and disables Open WebUI Web Search for this request (OpenRouter Search overrides Web Search).
	version: 0.1.0
	license: MIT
	"""

from __future__ import annotations

import logging
from typing import Any

from open_webui.env import SRC_LOG_LEVELS

OWUI_OPENROUTER_PIPE_MARKER = "{_ORS_FILTER_MARKER}"
_FEATURE_FLAG = "{_ORS_FILTER_FEATURE_FLAG}"


class Filter:
    # Toggleable filter (shows a switch in the Integrations menu).
    toggle = True

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.search.toggle")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True

    def inlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Signal the pipe via request metadata (preferred path).
        if __metadata__ is not None and not isinstance(__metadata__, dict):
            return body

        features = body.get("features")
        if not isinstance(features, dict):
            features = {{}}
            body["features"] = features

        # Enforce: OpenRouter Search overrides Web Search (prevent OWUI native web search handler).
        features["web_search"] = False

        if isinstance(__metadata__, dict):
            meta_features = __metadata__.get("features")
            if meta_features is None:
                meta_features = {{}}
                __metadata__["features"] = meta_features

            # OWUI builds __metadata__["features"] as a reference to body["features"].
            # Break the reference so we can preserve the marker for the pipe while forcing
            # body.features.web_search = False.
            if meta_features is features:
                meta_features = dict(meta_features)
                __metadata__["features"] = meta_features

            if isinstance(meta_features, dict):
                meta_features[_FEATURE_FLAG] = True

        self.log.debug("Enabled OpenRouter Search; disabled OWUI web_search")
        return body
'''

    @timed
    def ensure_ors_filter_function_id(self) -> str | None:
        """Ensure the OpenRouter Search companion filter exists (and is up to date), returning its OWUI function id."""
        try:
            from open_webui.models.functions import Functions  # type: ignore
        except Exception:
            return None

        desired_source = self.render_ors_filter_source().strip() + "\n"
        desired_name = "OpenRouter Search"
        desired_meta = {
            "description": (
                "Enable OpenRouter native web search for this request. "
                "When OpenRouter Search is enabled, OWUI Web Search is disabled to avoid double-search."
            ),
            "toggle": True,
            "manifest": {
                "title": "OpenRouter Search",
                "id": "openrouter_search",
                "version": "0.1.0",
                "license": "MIT",
            },
        }

        def _matches_candidate(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            if _ORS_FILTER_MARKER in content:
                return True
            # Back-compat for manual installs of earlier drafts: detect by the feature flag string.
            return _ORS_FILTER_FEATURE_FLAG in content and "class Filter" in content

        try:
            filters = Functions.get_functions_by_type("filter", active_only=False)
        except Exception:
            return None

        candidates = [f for f in filters if _matches_candidate(getattr(f, "content", ""))]
        chosen = None
        if candidates:
            # Prefer the canonical marker when present.
            marked = [f for f in candidates if _ORS_FILTER_MARKER in (getattr(f, "content", "") or "")]
            if marked:
                candidates = marked
            chosen = sorted(candidates, key=lambda f: int(getattr(f, "updated_at", 0) or 0), reverse=True)[0]
            if len(candidates) > 1:
                self.logger.warning(
                    "Multiple OpenRouter Search filter candidates found (%d); using '%s'.",
                    len(candidates),
                    getattr(chosen, "id", ""),
                )

        if chosen is None:
            if not getattr(self.valves, "AUTO_INSTALL_ORS_FILTER", False):
                return None

            candidate_id = _ORS_FILTER_PREFERRED_FUNCTION_ID
            suffix = 0
            while True:
                existing = None
                try:
                    existing = Functions.get_function_by_id(candidate_id)
                except Exception:
                    existing = None
                if existing is None:
                    break
                suffix += 1
                candidate_id = f"{_ORS_FILTER_PREFERRED_FUNCTION_ID}_{suffix}"
                if suffix > 50:
                    return None

            try:
                from open_webui.models.functions import FunctionForm, FunctionMeta  # type: ignore
            except Exception:
                return None

            meta_obj = FunctionMeta(**desired_meta)
            form = FunctionForm(
                id=candidate_id,
                name=desired_name,
                content=desired_source,
                meta=meta_obj,
            )
            created = Functions.insert_new_function("", "filter", form)
            if not created:
                return None
            Functions.update_function_by_id(candidate_id, {"is_active": True, "is_global": False, "name": desired_name, "meta": desired_meta})
            self.logger.info("Installed OpenRouter Search filter: %s", candidate_id)
            return candidate_id

        function_id = str(getattr(chosen, "id", "") or "").strip()
        if not function_id:
            return None

        if getattr(self.valves, "AUTO_INSTALL_ORS_FILTER", False):
            existing_content = (getattr(chosen, "content", "") or "").strip() + "\n"
            if existing_content != desired_source:
                self.logger.info("Updating OpenRouter Search filter: %s", function_id)
                Functions.update_function_by_id(
                    function_id,
                    {
                        "content": desired_source,
                        "name": desired_name,
                        "meta": desired_meta,
                        "type": "filter",
                        "is_active": True,
                        "is_global": False,
                    },
                )
            else:
                Functions.update_function_by_id(
                    function_id,
                    {
                        "name": desired_name,
                        "meta": desired_meta,
                        "type": "filter",
                        "is_active": True,
                        "is_global": False,
                    },
                )

        return function_id

    # =========================================================================
    # DIRECT UPLOADS FILTER
    # =========================================================================

    @staticmethod
    def render_direct_uploads_filter_source() -> str:
        """Return the canonical OWUI filter source for the OpenRouter Direct Uploads toggle."""
        # NOTE: This file is inserted into Open WebUI's Functions table as a *filter* function.
        # It must not depend on this pipe module at runtime.
        template = '''"""
title: Direct Uploads
author: Open-WebUI-OpenRouter-pipe
author_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: __FILTER_ID__
description: Bypass Open WebUI RAG for chat uploads and forward them to OpenRouter as direct file/audio/video inputs (user-controlled via valves).
version: 0.1.0
license: MIT
"""

from __future__ import annotations

import fnmatch
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from open_webui.env import SRC_LOG_LEVELS

OWUI_OPENROUTER_PIPE_MARKER = "__MARKER__"


class Filter:
    # Toggleable filter (shows a switch in the Integrations menu).
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )
        DIRECT_TOTAL_PAYLOAD_MAX_MB: int = Field(
            default=50,
            ge=1,
            le=500,
            description="Maximum total size (MB) across all diverted direct uploads in a single request.",
        )
        DIRECT_FILE_MAX_UPLOAD_SIZE_MB: int = Field(
            default=50,
            ge=1,
            le=500,
            description="Maximum size (MB) for a single diverted direct file upload.",
        )
        DIRECT_AUDIO_MAX_UPLOAD_SIZE_MB: int = Field(
            default=25,
            ge=1,
            le=500,
            description="Maximum size (MB) for a single diverted direct audio upload.",
        )
        DIRECT_VIDEO_MAX_UPLOAD_SIZE_MB: int = Field(
            default=20,
            ge=1,
            le=500,
            description="Maximum size (MB) for a single diverted direct video upload.",
        )
        DIRECT_FILE_MIME_ALLOWLIST: str = Field(
            default="application/pdf,text/plain,text/markdown,application/json,text/csv",
            description="Comma-separated MIME allowlist for diverted direct generic files.",
        )
        DIRECT_AUDIO_MIME_ALLOWLIST: str = Field(
            default="audio/*",
            description="Comma-separated MIME allowlist for diverted direct audio files.",
        )
        DIRECT_VIDEO_MIME_ALLOWLIST: str = Field(
            default="video/mp4,video/mpeg,video/quicktime,video/webm",
            description="Comma-separated MIME allowlist for diverted direct video files.",
        )
        DIRECT_AUDIO_FORMAT_ALLOWLIST: str = Field(
            default="wav,mp3,aiff,aac,ogg,flac,m4a,pcm16,pcm24",
            description="Comma-separated audio format allowlist (derived from filename/MIME).",
        )
        DIRECT_RESPONSES_AUDIO_FORMAT_ALLOWLIST: str = Field(
            default="wav,mp3",
            description="Comma-separated audio formats eligible for /responses input_audio.format.",
        )

    class UserValves(BaseModel):
        DIRECT_FILES: bool = Field(
            default=False,
            description="When enabled, uploads files directly to the model.",
        )
        DIRECT_AUDIO: bool = Field(
            default=False,
            description="When enabled, uploads audio directly to the model.",
        )
        DIRECT_VIDEO: bool = Field(
            default=False,
            description="When enabled, uploads video directly to the model.",
        )
        DIRECT_PDF_PARSER: Literal["Native", "PDF Text", "Mistral OCR"] = Field(
            default="Native",
            description="OpenRouter PDF engine selection for direct uploads (requires Direct Files enabled).",
        )

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.direct.uploads")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    @staticmethod
    def _to_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return int(stripped)
            except ValueError:
                return None
        return None

    @staticmethod
    def _csv_set(value: Any) -> set[str]:
        if not isinstance(value, str):
            return set()
        parts = []
        for raw in value.split(","):
            item = (raw or "").strip().lower()
            if item:
                parts.append(item)
        return set(parts)

    @staticmethod
    def _mime_allowed(mime: str, allowlist_csv: str) -> bool:
        mime = (mime or "").strip().lower()
        if not mime:
            return False
        allowlist = Filter._csv_set(allowlist_csv)
        if not allowlist:
            return False
        for pattern in allowlist:
            if fnmatch.fnmatch(mime, pattern):
                return True
        return False

    @staticmethod
    def _infer_audio_format(name: Any, mime: Any) -> str:
        mime_str = (mime or "").strip().lower() if isinstance(mime, str) else ""
        if mime_str in {"audio/wav", "audio/wave", "audio/x-wav"}:
            return "wav"
        if mime_str in {"audio/mpeg", "audio/mp3"}:
            return "mp3"
        filename = (name or "").strip().lower() if isinstance(name, str) else ""
        if "." in filename:
            ext = filename.rsplit(".", 1)[-1].strip().lower()
            if ext:
                return ext
        return ""

    @staticmethod
    def _model_caps(__model__: Any) -> dict[str, bool]:
        if not isinstance(__model__, dict):
            return {}
        meta = __model__.get("info", {}).get("meta", {})
        if not isinstance(meta, dict):
            return {}
        pipe_meta = meta.get("openrouter_pipe", {})
        if not isinstance(pipe_meta, dict):
            return {}
        caps = pipe_meta.get("capabilities", {})
        return caps if isinstance(caps, dict) else {}

    def inlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
        __model__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(body, dict):
            return body
        if __metadata__ is not None and not isinstance(__metadata__, dict):
            return body
        if __user__ is not None and not isinstance(__user__, dict):
            __user__ = None

        user_valves = None
        if isinstance(__user__, dict):
            user_valves = __user__.get("valves")
        if not isinstance(user_valves, BaseModel):
            user_valves = self.UserValves()

        enable_files = bool(getattr(user_valves, "DIRECT_FILES", False))
        enable_audio = bool(getattr(user_valves, "DIRECT_AUDIO", False))
        enable_video = bool(getattr(user_valves, "DIRECT_VIDEO", False))
        pdf_parser = getattr(user_valves, "DIRECT_PDF_PARSER", "Native")

        files = body.get("files", None)
        if not isinstance(files, list) or not files:
            return body

        caps = self._model_caps(__model__)
        supports_files = bool(caps.get("file_input", False))
        supports_audio = bool(caps.get("audio_input", False))
        supports_video = bool(caps.get("video_input", False))

        diverted: dict[str, list[dict[str, Any]]] = {"files": [], "audio": [], "video": []}
        retained: list[Any] = []
        warnings: list[str] = []
        total_bytes = 0
        pdf_seen = False

        total_limit = int(self.valves.DIRECT_TOTAL_PAYLOAD_MAX_MB) * 1024 * 1024
        file_limit = int(self.valves.DIRECT_FILE_MAX_UPLOAD_SIZE_MB) * 1024 * 1024
        audio_limit = int(self.valves.DIRECT_AUDIO_MAX_UPLOAD_SIZE_MB) * 1024 * 1024
        video_limit = int(self.valves.DIRECT_VIDEO_MAX_UPLOAD_SIZE_MB) * 1024 * 1024

        audio_formats_allowed = self._csv_set(self.valves.DIRECT_AUDIO_FORMAT_ALLOWLIST)

        for item in files:
            if not isinstance(item, dict):
                retained.append(item)
                continue
            if bool(item.get("legacy", False)):
                retained.append(item)
                continue
            if (item.get("type") or "file") != "file":
                retained.append(item)
                continue
            file_id = item.get("id")
            if not isinstance(file_id, str) or not file_id.strip():
                retained.append(item)
                continue

            content_type = (
                item.get("content_type")
                or item.get("contentType")
                or item.get("mime_type")
                or item.get("mimeType")
                or ""
            )
            content_type = content_type.strip().lower() if isinstance(content_type, str) else ""
            name = item.get("name") or ""
            filename = name.strip().lower() if isinstance(name, str) else ""
            is_pdf = ("pdf" in content_type) or filename.endswith(".pdf")

            size_bytes = self._to_int(item.get("size"))
            if size_bytes is None or size_bytes < 0:
                raise Exception("Direct uploads: uploaded file missing a valid size.")

            kind = "files"
            if content_type.startswith("audio/"):
                kind = "audio"
            elif content_type.startswith("video/"):
                kind = "video"

            if kind == "files":
                if not enable_files:
                    retained.append(item)
                    continue
                if not supports_files:
                    warnings.append("Direct file uploads not supported by the selected model; falling back to Open WebUI.")
                    retained.append(item)
                    continue
                if not self._mime_allowed(content_type, self.valves.DIRECT_FILE_MIME_ALLOWLIST):
                    # Fail-open: leave unsupported types on the normal OWUI path (RAG/Knowledge).
                    retained.append(item)
                    continue
                if size_bytes > file_limit:
                    raise Exception(
                        f"Direct file '{name or file_id}' is too large ({size_bytes} bytes; max {self.valves.DIRECT_FILE_MAX_UPLOAD_SIZE_MB} MB)."
                    )
                total_bytes += size_bytes
                if total_bytes > total_limit:
                    raise Exception(
                        f"Direct uploads exceed total limit ({self.valves.DIRECT_TOTAL_PAYLOAD_MAX_MB} MB)."
                    )
                diverted["files"].append(
                    {
                        "id": file_id,
                        "name": name,
                        "size": size_bytes,
                        "content_type": content_type,
                    }
                )
                if is_pdf:
                    pdf_seen = True
                continue

            if kind == "audio":
                if not enable_audio:
                    retained.append(item)
                    continue
                if not supports_audio:
                    warnings.append("Direct audio uploads not supported by the selected model; falling back to Open WebUI.")
                    retained.append(item)
                    continue
                if not self._mime_allowed(content_type, self.valves.DIRECT_AUDIO_MIME_ALLOWLIST):
                    retained.append(item)
                    continue
                audio_format = self._infer_audio_format(name, content_type)
                if not audio_format or (audio_formats_allowed and audio_format not in audio_formats_allowed):
                    retained.append(item)
                    continue
                if size_bytes > audio_limit:
                    raise Exception(
                        f"Direct audio '{name or file_id}' is too large ({size_bytes} bytes; max {self.valves.DIRECT_AUDIO_MAX_UPLOAD_SIZE_MB} MB)."
                    )
                total_bytes += size_bytes
                if total_bytes > total_limit:
                    raise Exception(
                        f"Direct uploads exceed total limit ({self.valves.DIRECT_TOTAL_PAYLOAD_MAX_MB} MB)."
                    )
                diverted["audio"].append(
                    {
                        "id": file_id,
                        "name": name,
                        "size": size_bytes,
                        "content_type": content_type,
                        "format": audio_format,
                    }
                )
                continue

            if kind == "video":
                if not enable_video:
                    retained.append(item)
                    continue
                if not supports_video:
                    warnings.append("Direct video uploads not supported by the selected model; falling back to Open WebUI.")
                    retained.append(item)
                    continue
                if not self._mime_allowed(content_type, self.valves.DIRECT_VIDEO_MIME_ALLOWLIST):
                    retained.append(item)
                    continue
                if size_bytes > video_limit:
                    raise Exception(
                        f"Direct video '{name or file_id}' is too large ({size_bytes} bytes; max {self.valves.DIRECT_VIDEO_MAX_UPLOAD_SIZE_MB} MB)."
                    )
                total_bytes += size_bytes
                if total_bytes > total_limit:
                    raise Exception(
                        f"Direct uploads exceed total limit ({self.valves.DIRECT_TOTAL_PAYLOAD_MAX_MB} MB)."
                    )
                diverted["video"].append(
                    {
                        "id": file_id,
                        "name": name,
                        "size": size_bytes,
                        "content_type": content_type,
                    }
                )
                continue

            retained.append(item)

        diverted_any = bool(diverted["files"] or diverted["audio"] or diverted["video"])
        # OWUI "File Context" reads `body["metadata"]["files"]`, but OWUI also rebuilds metadata.files
        # from `body["files"]` after inlet filters. To reliably bypass OWUI RAG for diverted uploads,
        # update both.
        if diverted_any:
            body["files"] = retained
            if isinstance(__metadata__, dict):
                __metadata__["files"] = retained

        if isinstance(__metadata__, dict) and (diverted_any or warnings):
            prev_pipe_meta = __metadata__.get("openrouter_pipe")
            pipe_meta = dict(prev_pipe_meta) if isinstance(prev_pipe_meta, dict) else {}
            __metadata__["openrouter_pipe"] = pipe_meta

            if warnings:
                prev_warnings = pipe_meta.get("direct_uploads_warnings")
                merged_warnings: list[str] = []
                seen: set[str] = set()
                if isinstance(prev_warnings, list):
                    for warning in prev_warnings:
                        if isinstance(warning, str) and warning and warning not in seen:
                            seen.add(warning)
                            merged_warnings.append(warning)
                for warning in warnings:
                    if warning and warning not in seen:
                        seen.add(warning)
                        merged_warnings.append(warning)
                pipe_meta["direct_uploads_warnings"] = merged_warnings

            if diverted_any:
                prev_attachments = pipe_meta.get("direct_uploads")
                attachments = dict(prev_attachments) if isinstance(prev_attachments, dict) else {}
                pipe_meta["direct_uploads"] = attachments
                # Persist the /responses audio format allowlist into metadata so the pipe can honor it at injection time.
                attachments["responses_audio_format_allowlist"] = self.valves.DIRECT_RESPONSES_AUDIO_FORMAT_ALLOWLIST
                if pdf_seen and isinstance(pdf_parser, str) and pdf_parser.strip():
                    attachments["pdf_parser"] = pdf_parser.strip()

                for key in ("files", "audio", "video"):
                    items = diverted.get(key) or []
                    if items:
                        existing = attachments.get(key)
                        merged: list[dict[str, Any]] = []
                        seen: set[str] = set()
                        if isinstance(existing, list):
                            for entry in existing:
                                if isinstance(entry, dict):
                                    eid = entry.get("id")
                                    if isinstance(eid, str) and eid and eid not in seen:
                                        seen.add(eid)
                                        merged.append(entry)
                        for entry in items:
                            eid = entry.get("id")
                            if isinstance(eid, str) and eid and eid not in seen:
                                seen.add(eid)
                                merged.append(entry)
                        attachments[key] = merged

        if diverted_any:
            self.log.debug("Diverted %d byte(s) for direct upload forwarding", total_bytes)
        return body
'''

        return (
            template.replace("__FILTER_ID__", _DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID)
            .replace("__MARKER__", _DIRECT_UPLOADS_FILTER_MARKER)
        )

    @timed
    def ensure_direct_uploads_filter_function_id(self) -> str | None:
        """Ensure the OpenRouter Direct Uploads companion filter exists (and is up to date), returning its OWUI function id."""
        try:
            from open_webui.models.functions import Functions  # type: ignore
        except Exception:
            return None

        desired_source = self.render_direct_uploads_filter_source().strip() + "\n"
        desired_name = "Direct Uploads"
        desired_meta = {
            "description": (
                "Bypass Open WebUI RAG for chat uploads and forward them to OpenRouter as direct file/audio/video inputs. "
                "Enable files/audio/video via filter user valves."
            ),
            "toggle": True,
            "manifest": {
                "title": "Direct Uploads",
                "id": _DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID,
                "version": "0.1.0",
                "license": "MIT",
            },
        }

        def _matches_candidate(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            return _DIRECT_UPLOADS_FILTER_MARKER in content and "class Filter" in content

        try:
            filters = Functions.get_functions_by_type("filter", active_only=False)
        except Exception:
            return None

        candidates = [f for f in filters if _matches_candidate(getattr(f, "content", ""))]
        chosen = None
        if candidates:
            chosen = sorted(candidates, key=lambda f: int(getattr(f, "updated_at", 0) or 0), reverse=True)[0]
            if len(candidates) > 1:
                self.logger.warning(
                    "Multiple OpenRouter Direct Uploads filter candidates found (%d); using '%s'.",
                    len(candidates),
                    getattr(chosen, "id", ""),
                )

        if chosen is None:
            if not getattr(self.valves, "AUTO_INSTALL_DIRECT_UPLOADS_FILTER", False):
                return None

            candidate_id = _DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID
            suffix = 0
            while True:
                existing = None
                try:
                    existing = Functions.get_function_by_id(candidate_id)
                except Exception:
                    existing = None
                if existing is None:
                    break
                suffix += 1
                candidate_id = f"{_DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID}_{suffix}"
                if suffix > 50:
                    return None

            try:
                from open_webui.models.functions import FunctionForm, FunctionMeta  # type: ignore
            except Exception:
                return None

            meta_obj = FunctionMeta(**desired_meta)
            form = FunctionForm(
                id=candidate_id,
                name=desired_name,
                content=desired_source,
                meta=meta_obj,
            )
            created = Functions.insert_new_function("", "filter", form)
            if not created:
                return None
            Functions.update_function_by_id(
                candidate_id,
                {
                    "is_active": True,
                    "is_global": False,
                    "name": desired_name,
                    "meta": desired_meta,
                },
            )
            self.logger.info("Installed OpenRouter Direct Uploads filter: %s", candidate_id)
            return candidate_id

        function_id = str(getattr(chosen, "id", "") or "").strip()
        if not function_id:
            return None

        if getattr(self.valves, "AUTO_INSTALL_DIRECT_UPLOADS_FILTER", False):
            existing_content = (getattr(chosen, "content", "") or "").strip() + "\n"
            if existing_content != desired_source:
                self.logger.info("Updating OpenRouter Direct Uploads filter: %s", function_id)
                Functions.update_function_by_id(
                    function_id,
                    {
                        "content": desired_source,
                        "name": desired_name,
                        "meta": desired_meta,
                        "type": "filter",
                        "is_active": True,
                        "is_global": False,
                    },
                )
            else:
                Functions.update_function_by_id(
                    function_id,
                    {
                        "name": desired_name,
                        "meta": desired_meta,
                        "type": "filter",
                        "is_active": True,
                        "is_global": False,
                    },
                )

        return function_id

    # =========================================================================
    # PROVIDER ROUTING FILTER
    # =========================================================================

    @staticmethod
    def compute_provider_routing_hash(
        admin_models: str,
        user_models: str,
        provider_map: dict[str, dict[str, list[str]]],
    ) -> str:
        """Compute hash of provider routing state to detect changes."""
        # Include sorted versions of relevant data
        admin_sorted = sorted([m.strip() for m in admin_models.split(",") if m.strip()])
        user_sorted = sorted([m.strip() for m in user_models.split(",") if m.strip()])

        # Include provider info for models we care about
        relevant_slugs = set(admin_sorted) | set(user_sorted)
        provider_data = {
            slug: provider_map.get(slug, {})
            for slug in sorted(relevant_slugs)
        }

        data = f"{admin_sorted}|{user_sorted}|{provider_data}"
        return hashlib.md5(data.encode()).hexdigest()

    @staticmethod
    def _generate_inlet_logic(visibility: str) -> str:
        """Generate the inlet method logic based on visibility.

        IMPORTANT: User valves are injected by OWUI into __user__["valves"], NOT into
        self.user_valves. The self.user_valves from __init__ only contains defaults.
        We must extract the injected valves from __user__ to get actual user settings.
        """
        # Common provider dict building logic
        logic = '''        provider: dict[str, Any] = {}

        # Determine which valve source to use
        admin_set: set[str] = set()
        user_set: set[str] = set()
'''
        if visibility in ("admin", "both"):
            logic += '''        if hasattr(self, "valves"):
            admin_set = self.valves.model_fields_set
'''
        if visibility in ("user", "both"):
            # Extract user valves from OWUI-injected __user__["valves"]
            logic += '''
        # OWUI injects user valves into __user__["valves"], not self.user_valves
        user_valves = __user__.get("valves") if __user__ else None
        if user_valves is not None:
            user_set = user_valves.model_fields_set
'''

        # Field processing - user overrides admin for 'both'
        logic += '''
        # String fields: user if set and non-empty, else admin if set and non-empty
        def get_str(field: str) -> str:
            val = ""
'''
        if visibility in ("user", "both"):
            logic += '''            if field in user_set and user_valves is not None:
                v = getattr(user_valves, field, "")
                if isinstance(v, str) and v.strip():
                    val = v.strip()
'''
        if visibility in ("admin", "both"):
            logic += '''            if not val and field in admin_set and hasattr(self, "valves"):
                v = getattr(self.valves, field, "")
                if isinstance(v, str) and v.strip():
                    val = v.strip()
'''
        logic += '''            return val

        def get_literal(field: str) -> str:
            """Get Literal field value, returning empty string if _NO_PREF."""
            val = get_str(field)
            return "" if val == _NO_PREF else val

        def get_bool(field: str, api_default: bool = False) -> bool | None:
            """Get boolean value. Returns None if not explicitly set or matches API default."""
'''
        if visibility in ("user", "both"):
            logic += '''            if field in user_set and user_valves is not None:
                val = getattr(user_valves, field, api_default)
                if val != api_default:
                    return val
'''
        if visibility in ("admin", "both"):
            logic += '''            if field in admin_set and hasattr(self, "valves"):
                val = getattr(self.valves, field, api_default)
                if val != api_default:
                    return val
'''
        logic += '''            return None

        def get_float(field: str) -> float:
'''
        if visibility in ("user", "both"):
            logic += '''            if field in user_set and user_valves is not None:
                val = getattr(user_valves, field, 0)
                if isinstance(val, (int, float)) and val > 0:
                    return float(val)
'''
        if visibility in ("admin", "both"):
            logic += '''            if field in admin_set and hasattr(self, "valves"):
                val = getattr(self.valves, field, 0)
                if isinstance(val, (int, float)) and val > 0:
                    return float(val)
'''
        logic += '''            return 0

        # Build provider object
        # ORDER: Map display value to provider slug list using _ORDER_MAP
        order_display = get_literal("ORDER")
        if order_display:
            order_slugs = _ORDER_MAP.get(order_display)
            if order_slugs:
                provider["order"] = order_slugs
            else:
                self.log.warning("ORDER value %r not found in _ORDER_MAP", order_display)

        # ONLY: Map display name to slug using _PROVIDER_MAP
        only_display = get_literal("ONLY")
        if only_display:
            only_slug = _PROVIDER_MAP.get(only_display)
            if only_slug:
                provider["only"] = [only_slug]
            else:
                self.log.warning("ONLY value %r not found in _PROVIDER_MAP", only_display)

        # IGNORE: Map display name to slug using _PROVIDER_MAP
        ignore_display = get_literal("IGNORE")
        if ignore_display:
            ignore_slug = _PROVIDER_MAP.get(ignore_display)
            if ignore_slug:
                provider["ignore"] = [ignore_slug]
            else:
                self.log.warning("IGNORE value %r not found in _PROVIDER_MAP", ignore_display)

        # SORT: Literal values map directly
        sort_val = get_literal("SORT")
        if sort_val:
            provider["sort"] = sort_val

        # QUANTIZATION: Literal dropdown maps directly
        quant_val = get_literal("QUANTIZATION")
        if quant_val:
            provider["quantizations"] = [quant_val]

        # DATA_COLLECTION: Literal values map directly
        data_collection = get_literal("DATA_COLLECTION")
        if data_collection:
            provider["data_collection"] = data_collection

        # Boolean fields (only include if explicitly set and differs from API default)
        allow_fallbacks = get_bool("ALLOW_FALLBACKS", api_default=True)
        if allow_fallbacks is not None:
            provider["allow_fallbacks"] = allow_fallbacks

        require_params = get_bool("REQUIRE_PARAMETERS", api_default=False)
        if require_params is not None:
            provider["require_parameters"] = require_params

        zdr = get_bool("ZDR", api_default=False)
        if zdr is not None:
            provider["zdr"] = zdr

        distillable = get_bool("ENFORCE_DISTILLABLE_TEXT", api_default=False)
        if distillable is not None:
            provider["enforce_distillable_text"] = distillable

        # Numeric fields (only include if > 0)
        min_throughput = get_float("MIN_THROUGHPUT")
        if min_throughput > 0:
            provider["preferred_min_throughput"] = min_throughput

        max_latency = get_float("MAX_LATENCY")
        if max_latency > 0:
            provider["preferred_max_latency"] = max_latency

        # Price limits
        max_price_prompt = get_float("MAX_PRICE_PROMPT")
        max_price_completion = get_float("MAX_PRICE_COMPLETION")
        max_price_image = get_float("MAX_PRICE_IMAGE")
        max_price_audio = get_float("MAX_PRICE_AUDIO")
        max_price_request = get_float("MAX_PRICE_REQUEST")
        if (
            max_price_prompt > 0
            or max_price_completion > 0
            or max_price_image > 0
            or max_price_audio > 0
            or max_price_request > 0
        ):
            provider["max_price"] = {}
            if max_price_prompt > 0:
                provider["max_price"]["prompt"] = max_price_prompt
            if max_price_completion > 0:
                provider["max_price"]["completion"] = max_price_completion
            if max_price_image > 0:
                provider["max_price"]["image"] = max_price_image
            if max_price_audio > 0:
                provider["max_price"]["audio"] = max_price_audio
            if max_price_request > 0:
                provider["max_price"]["request"] = max_price_request

        # Inject into metadata if we have any provider settings
        if provider:
            if __metadata__ is None:
                __metadata__ = {}
            pipe_meta = __metadata__.setdefault("openrouter_pipe", {})
            pipe_meta["provider"] = provider
            self.log.debug("Injected provider routing: %s", provider)
'''
        return logic

    @staticmethod
    def _render_provider_routing_filter_source(
        model_slug: str,
        providers: list[str],
        quantizations: list[str],
        visibility: str,  # "admin", "user", or "both"
        *,
        short_name: str = "",
        provider_names: dict[str, str] | None = None,
    ) -> str:
        """Generate filter source code for a specific model's provider routing.

        Args:
            model_slug: The OpenRouter model slug (e.g., 'openai/gpt-4o')
            providers: List of available provider slugs
            quantizations: List of available quantization levels
            visibility: Who can configure - 'admin' (enforced), 'user' (optional), or 'both'
            short_name: Human-readable model name for filter title (e.g., 'GPT-4o')
            provider_names: Mapping of provider slug to display name (e.g., {'openai': 'OpenAI'})
        """
        safe_id = FilterManager.sanitize_model_for_filter_id(model_slug)
        filter_id = f"{_PROVIDER_ROUTING_FILTER_ID_PREFIX}{safe_id}"

        # SECURITY: Validate model_slug format before interpolating into generated code
        # Model slugs should be: vendor/model-name (e.g., "openai/gpt-4o")
        if not isinstance(model_slug, str) or not model_slug:
            raise ValueError("model_slug must be a non-empty string")
        # Use json.dumps for consistent double-quote escaping (repr uses single quotes
        # by default which causes quote-type mismatch when interpolated into "..." templates)
        safe_model_slug_escaped = json.dumps(model_slug)[1:-1]  # Strip surrounding quotes

        # Use short_name for display, fallback to model_slug if not provided
        display_name = short_name.strip() if short_name else model_slug.split("/")[-1]
        # SECURITY: Sanitize display name for use in filter title
        safe_display_name = FilterManager.validate_provider_name(display_name, slug=model_slug)
        safe_display_name_escaped = json.dumps(safe_display_name)[1:-1]

        marker = f"{_PROVIDER_ROUTING_FILTER_MARKER_PREFIX}{model_slug}:{_PROVIDER_ROUTING_FILTER_MARKER_VERSION}"
        # Escape marker for safe interpolation (contains model_slug)
        safe_marker_escaped = json.dumps(marker)[1:-1]

        # SECURITY: Validate provider slugs before interpolation (defense-in-depth)
        # Provider slugs from API should be lowercase ASCII + hyphens (e.g., "amazon-bedrock")
        safe_providers = [
            p for p in providers
            if isinstance(p, str) and _PROVIDER_SLUG_PATTERN.match(p) and len(p) <= 64
        ]

        # Build provider slug -> display name mapping (for dropdowns)
        prov_names = provider_names or {}
        # Create display options for ONLY/IGNORE dropdowns: "Display Name" -> "slug"
        # Format: Literal["(no preference)", "OpenAI", "Azure", ...]
        provider_display_options: list[str] = []
        provider_slug_map_entries: list[str] = []  # For the _PROVIDER_MAP dict
        for pslug in safe_providers:
            # Get display name, fallback to titlecased slug
            disp = prov_names.get(pslug, pslug.replace("-", " ").title())
            # SECURITY: Sanitize display name
            safe_disp = FilterManager.validate_provider_name(disp, slug=pslug)
            provider_display_options.append(safe_disp)
            # Map: "OpenAI" -> "openai"
            provider_slug_map_entries.append(f'    {FilterManager.safe_literal_string(safe_disp)}: {FilterManager.safe_literal_string(pslug)}')

        # Build Literal type string for ONLY/IGNORE fields
        # Include "(no preference)" as the default/empty option
        no_pref = "(no preference)"
        only_ignore_options = [no_pref] + provider_display_options
        only_ignore_literal = ", ".join(FilterManager.safe_literal_string(opt) for opt in only_ignore_options)

        # Build provider map code block
        provider_map_code = "{\n" + ",\n".join(provider_slug_map_entries) + "\n}" if provider_slug_map_entries else "{}"

        # Build ORDER permutations - FULL permutations with ALL providers in each entry
        # 2 providers = 2! = 2 choices, 3 providers = 3! = 6 choices
        # Display format: "OpenAI > Azure > Together" (all providers in priority order)
        order_display_options: list[str] = []
        order_map_entries: list[str] = []  # For the _ORDER_MAP dict

        # Create mapping from display name to slug for lookups
        display_to_slug = dict(zip(provider_display_options, safe_providers))

        # Generate all n! full permutations of ALL providers
        for perm in itertools.permutations(provider_display_options):
            # Display: "OpenAI > Azure > Together"
            perm_disp = " > ".join(perm)
            order_display_options.append(perm_disp)
            # Slugs: ["openai", "azure", "together"]
            perm_slugs = [display_to_slug[d] for d in perm]
            slugs_literal = ", ".join(FilterManager.safe_literal_string(s) for s in perm_slugs)
            order_map_entries.append(f'    {FilterManager.safe_literal_string(perm_disp)}: [{slugs_literal}]')

        # Build Literal type string for ORDER field
        order_options = [no_pref] + order_display_options
        order_literal = ", ".join(FilterManager.safe_literal_string(opt) for opt in order_options)

        # Build order map code block
        order_map_code = "{\n" + ",\n".join(order_map_entries) + "\n}" if order_map_entries else "{}"

        # SECURITY: Validate quantization levels before interpolation
        # Quantizations should be alphanumeric (e.g., "int4", "fp16", "bf16")
        safe_quantizations = [
            q for q in quantizations
            if isinstance(q, str) and _QUANTIZATION_PATTERN.match(q) and len(q) <= 32
        ]

        # Build Literal type string for QUANTIZATIONS dropdown
        quant_options = [no_pref] + safe_quantizations
        quantizations_literal = ", ".join(FilterManager.safe_literal_string(q) for q in quant_options)

        # Determine toggle setting based on visibility
        # ADMIN-only: toggle=False (always runs, user can't disable)
        # USER or BOTH: toggle=True (user can toggle per-chat)
        toggle_value = "False" if visibility == "admin" else "True"

        # Generate Valves class (for admin) if visibility is 'admin' or 'both'
        # Field order matches OpenRouter API docs: provider-selection.md
        valves_class = ""
        if visibility in ("admin", "both"):
            valves_class = f'''
    class Valves(BaseModel):
        """Admin-level provider routing preferences (cannot be disabled by users)."""
        ORDER: Literal[{order_literal}] = Field(default=_NO_PREF, description="Provider priority order")
        ALLOW_FALLBACKS: bool = Field(default=True, description="Allow backup providers if preferred unavailable")
        REQUIRE_PARAMETERS: bool = Field(default=False, description="Only use providers supporting all request params")
        DATA_COLLECTION: Literal[_NO_PREF, "allow", "deny"] = Field(default=_NO_PREF, description="Data collection policy")
        ZDR: bool = Field(default=False, description="Zero Data Retention - only ZDR endpoints")
        ONLY: Literal[{only_ignore_literal}] = Field(default=_NO_PREF, description="Use only this provider")
        IGNORE: Literal[{only_ignore_literal}] = Field(default=_NO_PREF, description="Avoid this provider")
        QUANTIZATION: Literal[{quantizations_literal}] = Field(default=_NO_PREF, description="Filter by quantization")
        SORT: Literal[_NO_PREF, "price", "throughput", "latency"] = Field(default=_NO_PREF, description="Sort providers by")
        MIN_THROUGHPUT: float = Field(default=0, ge=0, description="Min throughput (tokens/sec), 0=no pref")
        MAX_LATENCY: float = Field(default=0, ge=0, description="Max latency (seconds), 0=no pref")
        MAX_PRICE_PROMPT: float = Field(default=0, ge=0, description="Max price for prompt ($/M tokens), 0=no limit")
        MAX_PRICE_COMPLETION: float = Field(default=0, ge=0, description="Max price for completion ($/M tokens), 0=no limit")
        MAX_PRICE_IMAGE: float = Field(default=0, ge=0, description="Max price per image ($/image), 0=no limit")
        MAX_PRICE_AUDIO: float = Field(default=0, ge=0, description="Max price for audio ($/unit), 0=no limit")
        MAX_PRICE_REQUEST: float = Field(default=0, ge=0, description="Max price per request ($/request), 0=no limit")
'''

        # Generate UserValves class (for user) if visibility is 'user' or 'both'
        # Field order matches OpenRouter API docs: provider-selection.md
        user_valves_class = ""
        if visibility in ("user", "both"):
            user_valves_class = f'''
    class UserValves(BaseModel):
        """User-level provider routing preferences (can override admin defaults)."""
        ORDER: Literal[{order_literal}] = Field(default=_NO_PREF, description="Provider priority order")
        ALLOW_FALLBACKS: bool = Field(default=True, description="Allow backup providers if preferred unavailable")
        REQUIRE_PARAMETERS: bool = Field(default=False, description="Only use providers supporting all request params")
        DATA_COLLECTION: Literal[_NO_PREF, "allow", "deny"] = Field(default=_NO_PREF, description="Data collection policy")
        ZDR: bool = Field(default=False, description="Zero Data Retention - only ZDR endpoints")
        ONLY: Literal[{only_ignore_literal}] = Field(default=_NO_PREF, description="Use only this provider")
        IGNORE: Literal[{only_ignore_literal}] = Field(default=_NO_PREF, description="Avoid this provider")
        QUANTIZATION: Literal[{quantizations_literal}] = Field(default=_NO_PREF, description="Filter by quantization")
        SORT: Literal[_NO_PREF, "price", "throughput", "latency"] = Field(default=_NO_PREF, description="Sort providers by")
        MIN_THROUGHPUT: float = Field(default=0, ge=0, description="Min throughput (tokens/sec), 0=no pref")
        MAX_LATENCY: float = Field(default=0, ge=0, description="Max latency (seconds), 0=no pref")
        MAX_PRICE_PROMPT: float = Field(default=0, ge=0, description="Max price for prompt ($/M tokens), 0=no limit")
        MAX_PRICE_COMPLETION: float = Field(default=0, ge=0, description="Max price for completion ($/M tokens), 0=no limit")
        MAX_PRICE_IMAGE: float = Field(default=0, ge=0, description="Max price per image ($/image), 0=no limit")
        MAX_PRICE_AUDIO: float = Field(default=0, ge=0, description="Max price for audio ($/unit), 0=no limit")
        MAX_PRICE_REQUEST: float = Field(default=0, ge=0, description="Max price per request ($/request), 0=no limit")
'''

        # Generate init based on visibility
        init_body = "        self.log = logging.getLogger(f\"openrouter.provider.{MODEL_SLUG}\")\n        self.log.setLevel(SRC_LOG_LEVELS.get(\"OPENAI\", logging.INFO))"
        if visibility in ("admin", "both"):
            init_body += "\n        self.valves = self.Valves()"
        if visibility in ("user", "both"):
            init_body += "\n        self.user_valves = self.UserValves()"

        # Generate inlet logic based on visibility
        inlet_logic = FilterManager._generate_inlet_logic(visibility)

        return f'''"""
title: Provider: {safe_display_name_escaped}
author: Open-WebUI-OpenRouter-pipe
author_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: {filter_id}
description: Provider routing for {safe_display_name_escaped}
version: 0.2.0
license: MIT
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field
from open_webui.env import SRC_LOG_LEVELS

OWUI_OPENROUTER_PIPE_MARKER = "{safe_marker_escaped}"
MODEL_SLUG = "{safe_model_slug_escaped}"

# Sentinel value for "no preference" dropdown option
_NO_PREF = "(no preference)"

# Map display names to provider slugs
_PROVIDER_MAP: dict[str, str] = {provider_map_code}

# Map ORDER display values to provider slug lists
_ORDER_MAP: dict[str, list[str]] = {order_map_code}


class Filter:
    toggle = {toggle_value}
{valves_class}{user_valves_class}
    def __init__(self) -> None:
{init_body}

    def inlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
        __model__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Inject provider routing preferences into request metadata."""
{inlet_logic}
        return body
'''

    def ensure_provider_routing_filters(
        self,
        admin_models_csv: str,
        user_models_csv: str,
        provider_map: dict[str, dict[str, list[str]]],
        models: list[dict[str, Any]],
        pipe_identifier: str,
    ) -> dict[str, str]:
        """Ensure provider routing filters exist for specified models.

        Creates, updates, or disables filters based on current valve configuration.

        Returns:
            Mapping of model slug -> filter function ID for filters that should be attached.
        """
        try:
            from open_webui.models.functions import Functions, FunctionForm, FunctionMeta
        except Exception:
            return {}

        # Parse model lists
        admin_models = {m.strip() for m in admin_models_csv.split(",") if m.strip()}
        user_models = {m.strip() for m in user_models_csv.split(",") if m.strip()}

        # Compute state hash - skip generation if unchanged but still return existing mappings
        current_hash = self.compute_provider_routing_hash(admin_models_csv, user_models_csv, provider_map)
        hash_unchanged = current_hash == FilterManager._provider_routing_state_hash
        if hash_unchanged:
            self.logger.info("Provider routing state unchanged (hash=%s), returning existing filter mappings", current_hash[:8])
        else:
            self.logger.info(
                "Provider routing state changed (new_hash=%s), processing %d admin + %d user model(s)",
                current_hash[:8],
                len(admin_models),
                len(user_models),
            )
            # NOTE: Hash update moved to end of function to prevent race condition
            # (hash was being set before filter work completed)

        # Determine visibility for each model
        all_models = admin_models | user_models
        model_visibility: dict[str, str] = {}
        for slug in all_models:
            if slug in admin_models and slug in user_models:
                model_visibility[slug] = "both"
            elif slug in admin_models:
                model_visibility[slug] = "admin"
            else:
                model_visibility[slug] = "user"

        # Find existing provider routing filters
        try:
            all_filters = Functions.get_functions_by_type("filter", active_only=False)
        except Exception:
            all_filters = []

        existing_filters: dict[str, Any] = {}  # model_slug -> filter object
        for f in all_filters:
            content = getattr(f, "content", "") or ""
            if _PROVIDER_ROUTING_FILTER_MARKER_PREFIX in content:
                # Extract model slug from marker
                for line in content.split("\n"):
                    if _PROVIDER_ROUTING_FILTER_MARKER_PREFIX in line:
                        # Format: OWUI_OPENROUTER_PIPE_MARKER = "openrouter_pipe:provider_routing:slug:v1"
                        try:
                            marker_val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            parts = marker_val.split(":")
                            if len(parts) >= 4 and parts[0] == "openrouter_pipe" and parts[1] == "provider_routing":
                                slug = parts[2]
                                existing_filters[slug] = f
                        except Exception:
                            pass
                        break

        # Track slug -> filter_id mappings for attachment
        slug_to_filter_id: dict[str, str] = {}

        # Check if any filters are missing (deleted by user)
        missing_filters = {slug for slug in all_models if slug not in existing_filters}

        # If hash unchanged AND all filters exist, just return existing mappings
        if hash_unchanged and not missing_filters:
            for slug in all_models:
                existing = existing_filters.get(slug)
                if existing:
                    existing_id = getattr(existing, "id", "")
                    if existing_id:
                        slug_to_filter_id[slug] = existing_id
            self.logger.info(
                "Returning %d existing provider routing filter mappings",
                len(slug_to_filter_id),
            )
            return slug_to_filter_id

        # If filters are missing, we need to create them even if hash unchanged
        if missing_filters:
            self.logger.info(
                "Provider routing filters missing for %d model(s), recreating: %s",
                len(missing_filters),
                ", ".join(sorted(missing_filters)),
            )

        # Create/update filters for requested models
        created = 0
        updated = 0
        for slug, visibility in model_visibility.items():
            model_info = provider_map.get(slug, {})
            providers = model_info.get("providers", [])
            quantizations = model_info.get("quantizations", [])
            # Extract short_name and provider_names with type safety
            raw_short_name = model_info.get("short_name", "")
            short_name: str = raw_short_name if isinstance(raw_short_name, str) else ""
            raw_prov_names = model_info.get("provider_names", {})
            prov_names: dict[str, str] = raw_prov_names if isinstance(raw_prov_names, dict) else {}

            if not providers:
                self.logger.warning("Skipping filter for %s: no providers found in catalog (check slug spelling)", slug)
                continue

            safe_id = self.sanitize_model_for_filter_id(slug)
            filter_id = f"{_PROVIDER_ROUTING_FILTER_ID_PREFIX}{safe_id}"

            desired_source = self._render_provider_routing_filter_source(
                slug, providers, quantizations, visibility,
                short_name=short_name,
                provider_names=prov_names,
            ).strip() + "\n"

            # SECURITY: Validate generated source before storing (defense-in-depth)
            is_valid, validation_error = self.validate_filter_source(desired_source)
            if not is_valid:
                self.logger.error(
                    "Generated filter for %s failed syntax validation: %s. Skipping.",
                    slug,
                    validation_error,
                )
                continue  # Skip this filter, don't store invalid code

            # Use short_name for display, fallback to slug
            display_name = short_name if short_name else slug.split("/")[-1]
            # Sanitize for safety
            safe_display = self.validate_provider_name(display_name, slug=slug)
            desired_name = f"Provider: {safe_display}"
            desired_meta = {
                "description": f"Provider routing preferences for {slug}",
                "toggle": visibility != "admin",  # ADMIN-only: not toggleable
                "manifest": {
                    "title": f"Provider Routing: {slug}",
                    "id": filter_id,
                    "version": "0.1.0",
                    "license": "MIT",
                },
            }

            existing = existing_filters.get(slug)
            if existing:
                # Update existing filter
                existing_id = getattr(existing, "id", "")
                existing_content = (getattr(existing, "content", "") or "").strip() + "\n"
                if existing_content != desired_source:
                    Functions.update_function_by_id(
                        existing_id,
                        {
                            "content": desired_source,
                            "name": desired_name,
                            "meta": desired_meta,
                            "is_active": True,
                        },
                    )
                    updated += 1
                    self.logger.info("Updated provider routing filter: %s", existing_id)
                else:
                    # Just ensure it's active
                    Functions.update_function_by_id(existing_id, {"is_active": True})
                # Track for attachment
                if existing_id:
                    slug_to_filter_id[slug] = existing_id
            else:
                # Create new filter
                candidate_id = filter_id
                suffix = 0
                while True:
                    try:
                        existing_func = Functions.get_function_by_id(candidate_id)
                    except Exception:
                        existing_func = None
                    if existing_func is None:
                        break
                    suffix += 1
                    candidate_id = f"{filter_id}_{suffix}"
                    if suffix > 50:
                        self.logger.warning("Could not find unique ID for provider routing filter: %s", slug)
                        break

                if suffix <= 50:
                    meta_obj = FunctionMeta(**desired_meta)
                    form = FunctionForm(
                        id=candidate_id,
                        name=desired_name,
                        content=desired_source,
                        meta=meta_obj,
                    )
                    created_func = Functions.insert_new_function("", "filter", form)
                    if created_func:
                        Functions.update_function_by_id(candidate_id, {"is_active": True, "is_global": False})
                        created += 1
                        self.logger.info("Created provider routing filter: %s", candidate_id)
                        # Track for attachment
                        slug_to_filter_id[slug] = candidate_id

        # Disable filters for models no longer in the lists
        disabled = 0
        for slug, existing in existing_filters.items():
            if slug not in all_models:
                existing_id = getattr(existing, "id", "")
                if existing_id:
                    Functions.update_function_by_id(existing_id, {"is_active": False})
                    disabled += 1
                    self.logger.info("Disabled provider routing filter: %s", existing_id)

        if created or updated or disabled:
            self.logger.info(
                "Provider routing filters: created=%d, updated=%d, disabled=%d (total models=%d)",
                created, updated, disabled, len(all_models),
            )

        # Update hash AFTER all filter work completes (prevents race condition where
        # hash is set but work failed, causing next call to incorrectly skip)
        if created > 0 or updated > 0 or disabled > 0 or len(all_models) == 0:
            FilterManager._provider_routing_state_hash = current_hash
            self.logger.debug("Provider routing state hash updated: %s", current_hash[:8])
        else:
            self.logger.warning(
                "No filter changes made; state hash NOT updated (will retry on next call)"
            )

        self.logger.info(
            "Returning %d provider routing filter mappings for attachment: %r",
            len(slug_to_filter_id),
            slug_to_filter_id,
        )
        return slug_to_filter_id
