"""Filter management for OWUI filter functions.

This module manages filter functions:
1. OpenRouter Web Tools - Web Search, Web Fetch, and Datetime server tools
2. OpenRouter Image Generation - Image generation server tool
3. Direct Uploads - Bypasses OWUI RAG for file uploads
4. Provider Routing - Per-model provider/quantization preferences

FilterManager handles:
- Generating filter source code (static methods)
- Installing/updating filters in OWUI Functions table (instance methods)
- Security sanitization for code generation (static methods)
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, ClassVar

from ..core.config import (
    _DIRECT_UPLOADS_FILTER_MARKER,
    _DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID,
    _OPENROUTER_FUSION_FILTER_MARKER,
    _OPENROUTER_FUSION_FILTER_PREFERRED_FUNCTION_ID,
    _OPENROUTER_IMAGE_FILTER_MARKER,
    _OPENROUTER_IMAGE_GEN_FILTER_DEFAULT_MODEL,
    _OPENROUTER_IMAGE_GEN_FILTER_MARKER,
    _OPENROUTER_IMAGE_GEN_FILTER_PREFERRED_FUNCTION_ID,
    _OPENROUTER_VIDEO_GEN_FILTER_MARKER,
    _OPENROUTER_WEB_TOOLS_FILTER_MARKER,
    _OPENROUTER_WEB_TOOLS_FILTER_PREFERRED_FUNCTION_ID,
    _PIPE_METADATA_KEY,
    _PROVIDER_ROUTING_FILTER_ID_PREFIX,
    _PROVIDER_ROUTING_FILTER_MARKER_PREFIX,
    _PROVIDER_ROUTING_FILTER_MARKER_VERSION,
    _PROVIDER_ROUTING_MAX_PROVIDERS,
    _PROVIDER_ROUTING_ORDER_PERMUTATION_MAX,
    _PROVIDER_SLUG_PATTERN,
)
from ..core.timing_logger import timed
from ..core.utils import OWUI_FUNCTION_ID_ILLEGAL_RE as _MODEL_FILTER_ID_RE
from ..core.warn_latch import warn_level
from ..integrations.provider_options import CHAT_PROVIDER_KEYS, TRANSPORT_PROVIDER_KEYS

_ROUTING_CONTROL_KEYS: dict[str, str] = {
    "ORDER": "order",
    "ALLOW_FALLBACKS": "allow_fallbacks",
    "REQUIRE_PARAMETERS": "require_parameters",
    "DATA_COLLECTION": "data_collection",
    "ZDR": "zdr",
    "ENFORCE_DISTILLABLE_TEXT": "enforce_distillable_text",
    "ONLY": "only",
    "IGNORE": "ignore",
    "QUANTIZATION": "quantizations",
    "SORT": "sort",
    "SORT_PARTITION": "sort",
    "MIN_THROUGHPUT": "preferred_min_throughput",
    "MAX_LATENCY": "preferred_max_latency",
    "MAX_PRICE_PROMPT": "max_price",
    "MAX_PRICE_COMPLETION": "max_price",
    "MAX_PRICE_IMAGE": "max_price",
    "MAX_PRICE_AUDIO": "max_price",
    "MAX_PRICE_REQUEST": "max_price",
}

_QUANTIZATION_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

if TYPE_CHECKING:
    from ..pipe import Pipe

_PROVIDER_NAME_ALLOWLIST_RE = re.compile(r"[^A-Za-z0-9 \-_.]")
_PROVIDER_NAME_COLLAPSE_RE = re.compile(r"[ _]{2,}")

_warned_stale_filter_rows: set[str] = set()

_REPLACE_IMPORTS_REFUSAL = (
    "Open WebUI rewrites this source when it loads it and stores the result, so the pipe "
    "would rewrite it back on the next refresh: its unanchored replace of 'from utils', "
    "'from apps', 'from main' or 'from config' matched somewhere in the generated text"
)


class FilterManager:
    """Manages OWUI filter functions for the OpenRouter pipe.

    Static methods handle code generation and security sanitization.
    Instance methods handle filter installation/updates in OWUI.
    """

    _provider_routing_state_hash: str = ""

    def __init__(
        self,
        pipe: Pipe,
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

        cleaned = _PROVIDER_NAME_ALLOWLIST_RE.sub("", cleaned)

        cleaned = _PROVIDER_NAME_COLLAPSE_RE.sub(" ", cleaned)
        cleaned = cleaned.strip()

        if not cleaned:
            return "Unknown"

        max_length = 64
        if len(cleaned) > max_length:
            hash_source = slug if slug else cleaned
            hash_suffix = hashlib.md5(hash_source.encode("utf-8")).hexdigest()[:8]
            truncated = cleaned[: max_length - 9].rstrip(" _-.")
            if not truncated:
                cleaned = f"Provider_{hash_suffix}"
            else:
                cleaned = f"{truncated}_{hash_suffix}"

        return cleaned

    @staticmethod
    def sanitize_model_for_filter_id(model_slug: str) -> str:
        """Convert model slug to safe filter ID component.

        Example: 'openai/gpt-4o' -> 'openai_gpt_4o'

        Every character outside ``[A-Za-z0-9_]`` is replaced, not an enumerated few:
        a chain of ``.replace()`` calls only covers the separators someone thought of,
        and the catalog contains ids they did not -- the tilde aliases
        (``~anthropic/claude-sonnet-latest``) are real, supported ids that a
        colon-only fix leaves broken. The sibling renderers use the same character
        class and are injective for the same reason; they additionally lower-case and
        prepend their own prefix, which this one must not, because the id it produces
        is matched against already-installed filters.
        """
        return _MODEL_FILTER_ID_RE.sub("_", model_slug)

    @staticmethod
    def validate_filter_source(source: str) -> tuple[bool, str | None]:
        """Validate generated Python source code for syntactic correctness.

        Compiles the source the way Open WebUI will, before it is stored and executed.
        ``ast.parse`` is a weaker check than it looks: it accepts a file whose
        ``from __future__`` import is no longer first, which ``compile`` rejects, so a
        broken filter could pass validation and then fail to load forever.

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
            compile(source, "<generated-filter>", "exec")
        except SyntaxError as e:
            if e.lineno:
                error_msg = f"Line {e.lineno}: {e}"
            else:
                error_msg = str(e)
            return False, error_msg
        except (RecursionError, MemoryError, ValueError) as e:
            return False, f"Parse error: {e!s}"

        try:
            import open_webui.utils.plugin as owp

            rewritten = owp.replace_imports(source)
        except ImportError:
            return True, None
        except Exception:
            logging.getLogger(__name__).warning(
                "open_webui.utils.plugin.replace_imports is unavailable, so generated "
                "filter sources are not checked against Open WebUI's import rewrite",
                exc_info=True,
            )
            return True, None

        if rewritten != source:
            return False, _REPLACE_IMPORTS_REFUSAL
        return True, None

    # GENERIC FILTER INSTALL / UPDATE

    @timed
    async def _ensure_filter_installed(
        self,
        *,
        desired_source: str,
        desired_name: str,
        desired_meta: dict[str, Any],
        preferred_id: str,
        auto_install_valve: str,
        log_label: str,
        matches_candidate: Callable[[str], bool],
        primary_marker: str | None = None,
    ) -> str | None:
        """Generic filter install/update lifecycle shared by all filter types.

        Args:
            desired_source: The canonical filter source (already stripped + newline-terminated).
            desired_name: Display name for the filter function.
            desired_meta: Metadata dict for the filter function.
            preferred_id: Preferred OWUI function ID when auto-installing.
            auto_install_valve: Valve attribute name controlling auto-install/update.
            log_label: Human-readable label for log messages.
            matches_candidate: Predicate that returns True if a filter's content matches this filter type.
            primary_marker: If set, narrow candidates to those containing this marker (back-compat support).
        """
        try:
            from open_webui.models.functions import Functions  # type: ignore
        except ImportError:
            return None
        except Exception:
            logging.getLogger(__name__).warning(
                "open_webui.models.functions failed to import for a reason other than absence; "
                "the features that depend on it are now disabled",
                exc_info=True,
            )
            return None

        try:
            filters = await Functions.get_functions_by_type("filter", active_only=False)
        except Exception:
            self.logger.warning(
                "Cannot enumerate OWUI filter functions; %s will not be installed or updated",
                log_label,
                exc_info=True,
            )
            return None

        candidates = [f for f in filters if matches_candidate(getattr(f, "content", ""))]
        chosen = None
        if candidates:
            if primary_marker:
                marked = [f for f in candidates if primary_marker in (getattr(f, "content", "") or "")]
                if marked:
                    candidates = marked
            chosen = max(candidates, key=lambda f: int(getattr(f, "updated_at", 0) or 0))
            if len(candidates) > 1:
                self.logger.warning(
                    "Multiple %s candidates found (%d); using '%s'.",
                    log_label, len(candidates), getattr(chosen, "id", ""),
                )

        if chosen is None:
            if not getattr(self.valves, auto_install_valve, False):
                return None

            candidate_id = preferred_id
            suffix = 0
            while True:
                existing = await Functions.get_function_by_id(candidate_id)
                if existing is None:
                    break
                suffix += 1
                candidate_id = f"{preferred_id}_{suffix}"
                if suffix > 50:
                    return None

            try:
                from open_webui.models.functions import (  # type: ignore
                    FunctionForm,
                    FunctionMeta,
                )
            except ImportError:
                return None
            except Exception:
                logging.getLogger(__name__).warning(
                    "open_webui.models.functions failed to import for a reason other than absence; "
                    "the features that depend on it are now disabled",
                    exc_info=True,
                )
                return None

            meta_obj = FunctionMeta(**desired_meta)
            form = FunctionForm(
                id=candidate_id,
                name=desired_name,
                content=desired_source,
                meta=meta_obj,
            )
            created = await Functions.insert_new_function("", "filter", form)
            if not created:
                return None
            await Functions.update_function_by_id(candidate_id, {"is_active": True, "is_global": False, "name": desired_name, "meta": desired_meta})
            self.logger.info("Installed %s: %s", log_label, candidate_id)
            return candidate_id

        function_id = str(getattr(chosen, "id", "") or "").strip()
        if not function_id:
            return None

        existing_content = (getattr(chosen, "content", "") or "").strip() + "\n"
        if getattr(self.valves, auto_install_valve, False):
            if existing_content != desired_source:
                self.logger.info("Updating %s: %s", log_label, function_id)
                await Functions.update_function_by_id(
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
                await Functions.update_function_by_id(
                    function_id,
                    {
                        "name": desired_name,
                        "meta": desired_meta,
                        "type": "filter",
                        "is_active": True,
                        "is_global": False,
                    },
                )
        elif existing_content != desired_source:
            self.logger.log(
                warn_level(_warned_stale_filter_rows, f"stale_row:{function_id}"),
                "%s %r is installed and in use but its stored source is out of date. "
                "%s is off, so the pipe will not rewrite it and every fix to this filter "
                "stays undelivered. Turn %s on to let the pipe update it.",
                log_label,
                function_id,
                auto_install_valve,
                auto_install_valve,
            )

        return function_id


    @staticmethod
    def render_openrouter_web_tools_filter_source(
        *,
        enable_web_search: bool = True,
        enable_web_fetch: bool = True,
        enable_datetime: bool = True,
        enable_advisor: bool = True,
        enable_subagent: bool = True,
        enable_search_models: bool = True,
    ) -> str:
        """Return the canonical OWUI filter source for the OpenRouter Web Tools filter.

        The generated filter exposes admin Valves (engine selection, limits) and
        UserValves (per-tool toggles, preferences) for whichever tools are enabled.
        Tools disabled via the gate parameters are excluded from the template entirely.
        """
        valves_fields = [
            ('        priority: int = Field(\n'
            '            default=0,\n'
            '            description="Priority level for the filter operations.",\n'
            '        )'),
        ]
        if enable_web_search:
            valves_fields.extend([
                ('        WEB_SEARCH_ENGINE: Literal["auto", "native", "exa", "firecrawl", "parallel", "perplexity"] = Field(\n'
                '            default="auto",\n'
                '            description="Web search backend. auto lets OpenRouter choose, native uses the model provider, others use specific engines.",\n'
                '        )'),
                ('        WEB_SEARCH_MAX_RESULTS: int = Field(\n'
                '            default=5,\n'
                '            ge=1,\n'
                '            le=25,\n'
                '            description="Maximum number of search results per query.",\n'
                '        )'),
                ('        WEB_SEARCH_MAX_TOTAL_RESULTS: int = Field(\n'
                '            default=0,\n'
                '            ge=0,\n'
                '            description="Cap on total search results across all queries in one request. 0 means no cap.",\n'
                '        )'),
                ('        WEB_SEARCH_MAX_CHARACTERS: int = Field(\n'
                '            default=0,\n'
                '            ge=0,\n'
                '            le=100000,\n'
                '            description="Max characters of content per search result (1-100000). 0 means no cap. Takes precedence over context size when set.",\n'
                '        )'),
                ('        WEB_SEARCH_ALLOWED_DOMAINS: str = Field(\n'
                '            default="",\n'
                '            description="Comma-separated list of domains to restrict search results to. Empty means no restriction.",\n'
                '        )'),
                ('        WEB_SEARCH_EXCLUDED_DOMAINS: str = Field(\n'
                '            default="",\n'
                '            description="Comma-separated list of domains to exclude from search results.",\n'
                '        )'),
            ])
        if enable_web_fetch:
            valves_fields.extend([
                ('        WEB_FETCH_ENGINE: Literal["auto", "native", "exa", "openrouter", "firecrawl", "parallel"] = Field(\n'
                '            default="auto",\n'
                '            description="Web fetch backend. auto lets OpenRouter choose the best engine for each URL.",\n'
                '        )'),
                ('        WEB_FETCH_MAX_USES: int = Field(\n'
                '            default=0,\n'
                '            ge=0,\n'
                '            description="Maximum number of URL fetches per request. 0 means no limit.",\n'
                '        )'),
                ('        WEB_FETCH_MAX_CONTENT_TOKENS: int = Field(\n'
                '            default=0,\n'
                '            ge=0,\n'
                '            description="Maximum tokens of fetched content to return per URL. 0 means no limit.",\n'
                '        )'),
                ('        WEB_FETCH_ALLOWED_DOMAINS: str = Field(\n'
                '            default="",\n'
                '            description="Comma-separated list of domains allowed for fetching. Empty means allow all.",\n'
                '        )'),
                ('        WEB_FETCH_BLOCKED_DOMAINS: str = Field(\n'
                '            default="",\n'
                '            description="Comma-separated list of domains blocked from fetching.",\n'
                '        )'),
            ])
        if enable_advisor:
            valves_fields.append(
                '        ADVISOR_MODEL: str = Field(\n'
                '            default="",\n'
                '            description="Advisor model to consult (any OpenRouter model). Empty uses the chat\'s own model.",\n'
                '        )'
            )
        if enable_subagent:
            valves_fields.append(
                '        SUBAGENT_MODEL: str = Field(\n'
                '            default="",\n'
                '            description="Worker model for delegated subagent tasks. Empty uses the chat\'s own model.",\n'
                '        )'
            )
        valves_fields.append(
            '        SERVER_TOOLS_MAX_COST_USD: float = Field(\n'
            '            default=0.0,\n'
            '            ge=0,\n'
            '            description="Cap cumulative server-tool loop cost per request in USD. 0 means no cap.",\n'
            '        )'
        )

        # -- UserValves fields ------------------------------------------------
        user_valves_fields: list[str] = []
        if enable_web_search:
            user_valves_fields.extend([
                ('        WEB_SEARCH: bool = Field(\n'
                '            default=True,\n'
                '            description="Enable OpenRouter web search for this chat.",\n'
                '        )'),
                ('        WEB_SEARCH_CONTEXT_SIZE: Literal["low", "medium", "high"] = Field(\n'
                '            default="medium",\n'
                '            description="Amount of search context to include (low saves tokens, high is more thorough).",\n'
                '        )'),
                ('        WEB_SEARCH_LOCATION_CITY: str = Field(\n'
                '            default="",\n'
                '            description="City for location-aware search results.",\n'
                '        )'),
                ('        WEB_SEARCH_LOCATION_REGION: str = Field(\n'
                '            default="",\n'
                '            description="Region/state for location-aware search results.",\n'
                '        )'),
                ('        WEB_SEARCH_LOCATION_COUNTRY: str = Field(\n'
                '            default="",\n'
                '            description="Country code (e.g. AU, US) for location-aware search results.",\n'
                '        )'),
                ('        WEB_SEARCH_LOCATION_TIMEZONE: str = Field(\n'
                '            default="",\n'
                '            description="Timezone (e.g. Australia/Sydney) for location-aware search results.",\n'
                '        )'),
            ])
        if enable_web_fetch:
            user_valves_fields.append(
                '        WEB_FETCH: bool = Field(\n'
                '            default=False,\n'
                '            description="Enable OpenRouter web fetch (URL reading) for this chat.",\n'
                '        )'
            )
        if enable_datetime:
            user_valves_fields.extend([
                ('        DATETIME: bool = Field(\n'
                '            default=True,\n'
                '            description="Enable OpenRouter datetime tool for this chat (free, no extra cost).",\n'
                '        )'),
                ('        DATETIME_TIMEZONE: str = Field(\n'
                '            default="",\n'
                '            description="Timezone for the datetime tool (e.g. Australia/Sydney). Empty uses UTC.",\n'
                '        )'),
            ])
        if enable_advisor:
            user_valves_fields.append(
                '        ADVISOR: bool = Field(\n'
                '            default=False,\n'
                '            description="Enable the OpenRouter advisor tool (consult a higher-intelligence model mid-generation).",\n'
                '        )'
            )
        if enable_subagent:
            user_valves_fields.append(
                '        SUBAGENT: bool = Field(\n'
                '            default=False,\n'
                '            description="Enable the OpenRouter subagent tool (delegate tasks to a cheaper worker model).",\n'
                '        )'
            )
        if enable_search_models:
            user_valves_fields.append(
                '        SEARCH_MODELS: bool = Field(\n'
                '            default=False,\n'
                '            description="Enable the OpenRouter model-search tool (let the model search the OpenRouter catalog).",\n'
                '        )'
            )

        inlet_tool_blocks: list[str] = []
        if enable_web_search:
            inlet_tool_blocks.append(
                '        if user_valves.WEB_SEARCH:\n'
                '            ws_params: dict[str, Any] = {}\n'
                '            ws_params["engine"] = self.valves.WEB_SEARCH_ENGINE\n'
                '            ws_params["max_results"] = self.valves.WEB_SEARCH_MAX_RESULTS\n'
                '            if self.valves.WEB_SEARCH_MAX_TOTAL_RESULTS > 0:\n'
                '                ws_params["max_total_results"] = self.valves.WEB_SEARCH_MAX_TOTAL_RESULTS\n'
                '            if self.valves.WEB_SEARCH_MAX_CHARACTERS > 0:\n'
                '                ws_params["max_characters"] = self.valves.WEB_SEARCH_MAX_CHARACTERS\n'
                '            ws_params["search_context_size"] = user_valves.WEB_SEARCH_CONTEXT_SIZE\n'
                '            allowed = self._csv_list(self.valves.WEB_SEARCH_ALLOWED_DOMAINS)\n'
                '            if allowed:\n'
                '                ws_params["allowed_domains"] = allowed\n'
                '            excluded = self._csv_list(self.valves.WEB_SEARCH_EXCLUDED_DOMAINS)\n'
                '            if excluded:\n'
                '                ws_params["excluded_domains"] = excluded\n'
                '            location: dict[str, str] = {}\n'
                '            if user_valves.WEB_SEARCH_LOCATION_CITY:\n'
                '                location["city"] = user_valves.WEB_SEARCH_LOCATION_CITY\n'
                '            if user_valves.WEB_SEARCH_LOCATION_REGION:\n'
                '                location["region"] = user_valves.WEB_SEARCH_LOCATION_REGION\n'
                '            if user_valves.WEB_SEARCH_LOCATION_COUNTRY:\n'
                '                location["country"] = user_valves.WEB_SEARCH_LOCATION_COUNTRY\n'
                '            if user_valves.WEB_SEARCH_LOCATION_TIMEZONE:\n'
                '                location["timezone"] = user_valves.WEB_SEARCH_LOCATION_TIMEZONE\n'
                '            if location:\n'
                '                ws_params["user_location"] = location\n'
                '            server_tools["web_search"] = ws_params\n'
                '            suppress_owui_web_search = True'
            )
        if enable_web_fetch:
            inlet_tool_blocks.append(
                '        if user_valves.WEB_FETCH:\n'
                '            wf_params: dict[str, Any] = {}\n'
                '            wf_params["engine"] = self.valves.WEB_FETCH_ENGINE\n'
                '            if self.valves.WEB_FETCH_MAX_USES > 0:\n'
                '                wf_params["max_uses"] = self.valves.WEB_FETCH_MAX_USES\n'
                '            if self.valves.WEB_FETCH_MAX_CONTENT_TOKENS > 0:\n'
                '                wf_params["max_content_tokens"] = self.valves.WEB_FETCH_MAX_CONTENT_TOKENS\n'
                '            allowed = self._csv_list(self.valves.WEB_FETCH_ALLOWED_DOMAINS)\n'
                '            if allowed:\n'
                '                wf_params["allowed_domains"] = allowed\n'
                '            blocked = self._csv_list(self.valves.WEB_FETCH_BLOCKED_DOMAINS)\n'
                '            if blocked:\n'
                '                wf_params["blocked_domains"] = blocked\n'
                '            server_tools["web_fetch"] = wf_params'
            )
        if enable_datetime:
            inlet_tool_blocks.append(
                '        if user_valves.DATETIME:\n'
                '            dt_params: dict[str, Any] = {}\n'
                '            if user_valves.DATETIME_TIMEZONE:\n'
                '                dt_params["timezone"] = user_valves.DATETIME_TIMEZONE\n'
                '            server_tools["datetime"] = dt_params'
            )
        if enable_advisor:
            inlet_tool_blocks.append(
                '        if user_valves.ADVISOR:\n'
                '            adv_params: dict[str, Any] = {}\n'
                '            if self.valves.ADVISOR_MODEL:\n'
                '                adv_params["model"] = self.valves.ADVISOR_MODEL\n'
                '            server_tools["advisor"] = adv_params'
            )
        if enable_subagent:
            inlet_tool_blocks.append(
                '        if user_valves.SUBAGENT:\n'
                '            sub_params: dict[str, Any] = {}\n'
                '            if self.valves.SUBAGENT_MODEL:\n'
                '                sub_params["model"] = self.valves.SUBAGENT_MODEL\n'
                '            server_tools["subagent"] = sub_params'
            )
        if enable_search_models:
            inlet_tool_blocks.append(
                '        if user_valves.SEARCH_MODELS:\n'
                '            server_tools["chat_search_models"] = {}'
            )

        inlet_tools_code = "\n\n".join(inlet_tool_blocks)

        # -- Suppress OWUI web search block -----------------------------------
        suppress_block = ""
        if enable_web_search:
            suppress_block = (
                '\n'
                '        if suppress_owui_web_search:\n'
                '            features = body.get("features")\n'
                '            if not isinstance(features, dict):\n'
                '                features = {}\n'
                '                body["features"] = features\n'
                '            features["web_search"] = False\n'
            )

        # -- Build the template -----------------------------------------------
        template = '"""\n'
        template += 'title: OR Web Tools\n'
        template += 'author: Open-WebUI-OpenRouter-pipe\n'
        template += 'author_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe\n'
        template += 'id: __FILTER_ID__\n'
        template += 'description: Configures OpenRouter server tools (web search, web fetch, datetime, advisor, subagent, model search) for the OpenRouter pipe.\n'
        template += 'version: 0.1.0\n'
        template += 'license: MIT\n'
        template += '"""\n'
        template += '\n'
        template += 'from __future__ import annotations\n'
        template += '\n'
        template += 'import logging\n'
        template += 'from typing import Any, Literal\n'
        template += '\n'
        template += 'from pydantic import BaseModel, Field\n'
        template += '\n'
        template += 'try:' + '\n'
        template += '    from open_webui.env import SRC_LOG_LEVELS' + '\n'
        template += 'except Exception:  # noqa: BLE001 - open_webui.env does filesystem work on import' + '\n'
        template += '    SRC_LOG_LEVELS = {}' + '\n'
        template += '\n'
        template += 'OWUI_OPENROUTER_PIPE_MARKER = "__MARKER__"\n'
        template += '\n'
        template += '\n'
        template += 'class Filter:\n'
        template += '    toggle = True\n'
        template += '\n'

        # Valves class
        template += '    class Valves(BaseModel):\n'
        template += '\n'.join(valves_fields) + '\n'
        template += '\n'

        template += '    class UserValves(BaseModel):\n'
        if user_valves_fields:
            template += '\n'.join(user_valves_fields) + '\n'
        else:
            template += '        pass\n'
        template += '\n'

        # __init__
        template += '    def __init__(self) -> None:\n'
        template += '        self.log = logging.getLogger("openrouter.web.tools")\n'
        template += '        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))\n'
        template += '        self.toggle = True\n'
        template += '        self.valves = self.Valves()\n'
        template += '\n'

        # _csv_list helper
        template += '    @staticmethod\n'
        template += '    def _csv_list(value: str) -> list[str]:\n'
        template += '        if not isinstance(value, str):\n'
        template += '            return []\n'
        template += '        return [item.strip() for item in value.split(",") if item.strip()]\n'
        template += '\n'

        template += '    def inlet(\n'
        template += '        self,\n'
        template += '        body: dict[str, Any],\n'
        template += '        __metadata__: dict[str, Any] | None = None,\n'
        template += '        __user__: dict[str, Any] | None = None,\n'
        template += '        __model__: dict[str, Any] | None = None,\n'
        template += '    ) -> dict[str, Any]:\n'
        template += '        if not isinstance(body, dict):\n'
        template += '            return body\n'
        template += '        if __metadata__ is not None and not isinstance(__metadata__, dict):\n'
        template += '            return body\n'
        template += '\n'
        template += '        user_valves = None\n'
        template += '        if isinstance(__user__, dict):\n'
        template += '            user_valves = __user__.get("valves")\n'
        template += '        if not isinstance(user_valves, BaseModel):\n'
        template += '            user_valves = self.UserValves()\n'
        template += '\n'
        template += '        prev_st = (__metadata__.get("__PIPE_META_KEY__") or {}).get("server_tools") if isinstance(__metadata__, dict) else None\n'
        template += '        server_tools: dict[str, Any] = dict(prev_st) if isinstance(prev_st, dict) else {}\n'
        if enable_web_search:
            template += '        suppress_owui_web_search = False\n'
        template += '\n'
        template += inlet_tools_code + '\n'
        template += '\n'

        # Write to metadata
        template += '        if server_tools and isinstance(__metadata__, dict):\n'
        template += '            prev_pipe_meta = __metadata__.get("__PIPE_META_KEY__")\n'
        template += '            pipe_meta = dict(prev_pipe_meta) if isinstance(prev_pipe_meta, dict) else {}\n'
        template += '            __metadata__["__PIPE_META_KEY__"] = pipe_meta\n'
        template += '            pipe_meta["server_tools"] = server_tools\n'
        template += '            if self.valves.SERVER_TOOLS_MAX_COST_USD > 0:\n'
        template += '                pipe_meta["stop_server_tools_when"] = [\n'
        template += '                    {"type": "max_cost", "max_cost_in_dollars": self.valves.SERVER_TOOLS_MAX_COST_USD}\n'
        template += '                ]\n'

        # OWUI web search suppression
        template += suppress_block
        template += '\n'
        template += '        return body\n'

        return (
            template
            .replace("__FILTER_ID__", _OPENROUTER_WEB_TOOLS_FILTER_PREFERRED_FUNCTION_ID)
            .replace("__MARKER__", _OPENROUTER_WEB_TOOLS_FILTER_MARKER)
            .replace("__PIPE_META_KEY__", _PIPE_METADATA_KEY)
        )

    _inner_web_tools_module_cache: ClassVar[dict[str, Any]] = {}

    async def collect_installed_web_tools_config(
        self, user_id: str
    ) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
        from open_webui.models.functions import Functions

        function_id = _OPENROUTER_WEB_TOOLS_FILTER_PREFERRED_FUNCTION_ID
        try:
            row = await Functions.get_function_by_id(function_id)
        except Exception as exc:
            self.logger.debug("Web tools filter lookup failed: %s", exc, exc_info=True)
            return None
        if row is None or not getattr(row, "is_active", False):
            return None
        content = getattr(row, "content", None)
        if not isinstance(content, str) or "class Filter" not in content:
            return None
        cache_key = str(hash(content))
        module_ns = self._inner_web_tools_module_cache.get(cache_key)
        if module_ns is None:
            import sys
            import types

            module_name = f"_owui_or_webtools_inner_{abs(hash(content)) % 10**10}"
            module = types.ModuleType(module_name)
            module.__file__ = "<installed-web-tools-filter>"
            try:
                sys.modules[module_name] = module
                exec(compile(content, "<installed-web-tools-filter>", "exec"), module.__dict__)  # noqa: S102 - loading the installed filter module is this function's purpose
            except Exception as exc:
                self.logger.warning(
                    "Installed web tools filter failed to load for fusion inner calls: %s",
                    exc,
                    exc_info=True,
                )
                sys.modules.pop(module_name, None)
                return None
            module_ns = module.__dict__
            self._inner_web_tools_module_cache.clear()
            self._inner_web_tools_module_cache[cache_key] = module_ns
        filter_cls = module_ns.get("Filter")
        if filter_cls is None:
            return None
        try:
            instance = filter_cls()
            stored_valves = await Functions.get_function_valves_by_id(function_id) or {}
            instance.valves = filter_cls.Valves(**{k: v for k, v in stored_valves.items() if v is not None})
            stored_user = {}
            if user_id:
                stored_user = await Functions.get_user_valves_by_id_and_user_id(function_id, user_id) or {}
            user_valves = filter_cls.UserValves(**{k: v for k, v in stored_user.items() if v is not None})
            metadata: dict[str, Any] = {}
            instance.inlet({"model": "fusion-inner"}, __metadata__=metadata, __user__={"valves": user_valves})
        except Exception as exc:
            self.logger.warning(
                "Installed web tools filter inlet failed for fusion inner calls: %s", exc, exc_info=True
            )
            return None
        pipe_meta = metadata.get(_PIPE_METADATA_KEY)
        if not isinstance(pipe_meta, dict):
            return {}, []
        server_tools = pipe_meta.get("server_tools")
        stop_when = pipe_meta.get("stop_server_tools_when")
        return (
            dict(server_tools) if isinstance(server_tools, dict) else {},
            list(stop_when) if isinstance(stop_when, list) else [],
        )

    @timed
    async def ensure_openrouter_web_tools_filter_function_id(
        self,
        *,
        enable_web_search: bool = True,
        enable_web_fetch: bool = True,
        enable_datetime: bool = True,
        enable_advisor: bool = True,
        enable_subagent: bool = True,
        enable_search_models: bool = True,
    ) -> str | None:
        """Ensure the OpenRouter Web Tools filter exists (and is up to date), returning its OWUI function id."""

        def _matches(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            return _OPENROUTER_WEB_TOOLS_FILTER_MARKER in content and "class Filter" in content

        return await self._ensure_filter_installed(
            desired_source=self.render_openrouter_web_tools_filter_source(
                enable_web_search=enable_web_search,
                enable_web_fetch=enable_web_fetch,
                enable_datetime=enable_datetime,
                enable_advisor=enable_advisor,
                enable_subagent=enable_subagent,
                enable_search_models=enable_search_models,
            ).strip() + "\n",
            desired_name="OR Web Tools",
            desired_meta={
                "description": (
                    "OpenRouter server tools: Web Search, Web Fetch, and Datetime. "
                    "Toggle individual tools on/off via user valves."
                ),
                "toggle": True,
                "manifest": {
                    "title": "OR Web Tools",
                    "id": _OPENROUTER_WEB_TOOLS_FILTER_PREFERRED_FUNCTION_ID,
                    "version": "0.1.0",
                    "license": "MIT",
                },
            },
            preferred_id=_OPENROUTER_WEB_TOOLS_FILTER_PREFERRED_FUNCTION_ID,
            auto_install_valve="AUTO_INSTALL_WEB_TOOLS_FILTER",
            log_label="OpenRouter Web Tools filter",
            matches_candidate=_matches,
            primary_marker=_OPENROUTER_WEB_TOOLS_FILTER_MARKER,
        )

    # OPENROUTER FUSION FILTER

    async def ensure_openrouter_fusion_filter_function_id(self) -> str | None:
        """Ensure the OpenRouter Fusion filter exists (and is up to date), returning its OWUI function id."""
        from .fusion_filter_renderer import (
            FUSION_FILTER_DISPLAY_NAME,
            render_openrouter_fusion_filter_source,
        )

        def _matches(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            return _OPENROUTER_FUSION_FILTER_MARKER in content and "class Filter" in content

        return await self._ensure_filter_installed(
            desired_source=render_openrouter_fusion_filter_source(
                marker=_OPENROUTER_FUSION_FILTER_MARKER,
            ).strip() + "\n",
            desired_name=FUSION_FILTER_DISPLAY_NAME,
            desired_meta={
                "description": (
                    "Configure OpenRouter Fusion (multi-model judge panel): panel models, judge, "
                    "preset, and optional force-run. Acts on the openrouter/fusion model."
                ),
                "toggle": True,
                "manifest": {
                    "title": FUSION_FILTER_DISPLAY_NAME,
                    "id": _OPENROUTER_FUSION_FILTER_PREFERRED_FUNCTION_ID,
                    "version": "0.1.0",
                    "license": "MIT",
                },
            },
            preferred_id=_OPENROUTER_FUSION_FILTER_PREFERRED_FUNCTION_ID,
            auto_install_valve="AUTO_INSTALL_FUSION_FILTER",
            log_label="OpenRouter Fusion filter",
            matches_candidate=_matches,
            primary_marker=_OPENROUTER_FUSION_FILTER_MARKER,
        )

    # OPENROUTER IMAGE GENERATION FILTER

    @staticmethod
    def render_openrouter_image_gen_filter_source(
        *,
        model_id: str = "",
        image_model: dict[str, Any] | None = None,
        endpoint_record: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> str:
        """Return the canonical OWUI filter source for the OpenRouter Image Generation filter."""
        from .image_filter_renderer import (
            build_image_model_filter_spec,
            render_image_gen_filter_source,
        )

        resolved = (model_id or "").strip() or _OPENROUTER_IMAGE_GEN_FILTER_DEFAULT_MODEL
        return render_image_gen_filter_source(
            build_image_model_filter_spec(resolved, image_model, endpoint_record),
            catalog_match=isinstance(image_model, dict),
            selected_model=resolved,
        )

    async def image_gen_filter_inputs(
        self,
    ) -> tuple[str, dict[str, Any] | None, list[dict[str, Any]] | None]:
        from ..models.registry import OpenRouterModelRegistry

        selected = (await self.image_gen_filter_selected_model()).strip()
        model_id = selected or _OPENROUTER_IMAGE_GEN_FILTER_DEFAULT_MODEL
        spec = OpenRouterModelRegistry.spec(model_id)
        if not isinstance(spec, dict) or not spec:
            return model_id, None, None
        image_model = spec.get("image_model")
        if not isinstance(image_model, dict):
            image_model = {"id": model_id, "name": spec.get("name") or model_id}
        return model_id, image_model, OpenRouterModelRegistry.image_endpoint(model_id)

    @timed
    async def ensure_openrouter_image_gen_filter_function_id(self) -> str | None:
        """Ensure the OpenRouter Image Generation filter exists (and is up to date), returning its OWUI function id."""

        def _matches(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            return _OPENROUTER_IMAGE_GEN_FILTER_MARKER in content and "class Filter" in content

        from .image_filter_renderer import (
            build_image_model_filter_spec,
            image_gen_model_note,
            render_image_gen_filter_source,
        )

        model_id, image_model, endpoint_record = await self.image_gen_filter_inputs()
        spec = build_image_model_filter_spec(model_id, image_model, endpoint_record)
        catalog_match = isinstance(image_model, dict)
        desired_source = render_image_gen_filter_source(
            spec, catalog_match=catalog_match, selected_model=model_id
        ).strip() + "\n"
        valid, error = self.validate_filter_source(desired_source)
        if not valid:
            raise ValueError(f"Generated OpenRouter Image Generation filter is invalid: {error}")

        return await self._ensure_filter_installed(
            desired_source=desired_source,
            desired_name="OR Image Gen",
            desired_meta={
                "description": (
                    "Let the model generate images from text prompts via OpenRouter's image "
                    "generation server tool. "
                    + image_gen_model_note(spec, catalog_match=catalog_match)
                ),
                "toggle": True,
                "manifest": {
                    "title": "OR Image Gen",
                    "id": _OPENROUTER_IMAGE_GEN_FILTER_PREFERRED_FUNCTION_ID,
                    "version": "0.1.0",
                    "license": "MIT",
                },
            },
            preferred_id=_OPENROUTER_IMAGE_GEN_FILTER_PREFERRED_FUNCTION_ID,
            auto_install_valve="AUTO_INSTALL_IMAGE_GEN_FILTER",
            log_label="OpenRouter Image Generation filter",
            matches_candidate=_matches,
            primary_marker=_OPENROUTER_IMAGE_GEN_FILTER_MARKER,
        )

    async def image_gen_filter_selected_model(self) -> str:
        try:
            from open_webui.models.functions import Functions  # type: ignore
        except ImportError:
            return ""
        except Exception:
            logging.getLogger(__name__).warning(
                "open_webui.models.functions failed to import for a reason other than "
                "absence; the features that depend on it are now disabled",
                exc_info=True,
            )
            return ""

        try:
            rows = await Functions.get_functions_by_type("filter", active_only=False)
            chosen = next(
                (
                    row
                    for row in rows or []
                    if _OPENROUTER_IMAGE_GEN_FILTER_MARKER
                    in (getattr(row, "content", "") or "")
                ),
                None,
            )
            if chosen is None:
                return ""
            stored = await Functions.get_function_valves_by_id(
                str(getattr(chosen, "id", "") or "")
            )
        except Exception as exc:
            self.logger.debug(
                "Could not read the image generation filter's selected model: %s",
                exc,
                exc_info=True,
            )
            return ""

        selected = (stored or {}).get("IMAGE_GENERATION_MODEL")
        return selected.strip() if isinstance(selected, str) else ""

    def render_openrouter_video_gen_filter_source(
        self,
        *,
        model_id: str = "openrouter/video",
        video_model: dict[str, Any] | None = None,
    ) -> str:
        from .video_filter_renderer import render_video_filter_source

        return render_video_filter_source(
            model_id=model_id,
            video_model=video_model,
            pipe_metadata_key=_PIPE_METADATA_KEY,
            admin_valves=self.valves,
        )

    @timed
    async def ensure_openrouter_video_gen_filter_function_ids(
        self,
        models: list[dict[str, Any]],
    ) -> dict[str, str]:
        from ..models.registry import ModelFamily, OpenRouterModelRegistry

        installed: dict[str, str] = {}
        for model in models:
            model_id = model.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            model_id = model_id.strip()
            try:
                if not ModelFamily.supports("video_generation", model_id):
                    continue
            except (AttributeError, TypeError):
                continue
            spec = OpenRouterModelRegistry.spec(model_id)
            video_model = spec.get("video_model") if isinstance(spec, dict) else None
            if not isinstance(video_model, dict):
                video_model = dict(model)
            original_id = model.get("original_id")
            canonical_id = original_id if isinstance(original_id, str) and original_id.strip() else model_id
            function_id = await self._ensure_single_video_gen_filter_function_id(
                model_id=canonical_id,
                video_model=video_model,
            )
            if function_id:
                installed[model_id] = function_id
                if isinstance(original_id, str) and original_id.strip():
                    installed[original_id.strip()] = function_id
        return installed

    async def _ensure_single_video_gen_filter_function_id(
        self,
        *,
        model_id: str,
        video_model: dict[str, Any] | None,
    ) -> str | None:
        from .video_filter_renderer import build_video_filter_spec

        spec = build_video_filter_spec(model_id, video_model)

        model_id_token = f"VIDEO_MODEL_ID = {spec.model_id!r}"

        def _matches(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            if _OPENROUTER_VIDEO_GEN_FILTER_MARKER not in content:
                return False
            if model_id_token not in content:
                return False
            return "class Filter" in content

        desired_source = self.render_openrouter_video_gen_filter_source(
            model_id=model_id,
            video_model=video_model,
        ).strip() + "\n"
        valid, error = self.validate_filter_source(desired_source)
        if not valid:
            raise ValueError(f"Generated OpenRouter Video Generation filter is invalid: {error}")

        return await self._ensure_filter_installed(
            desired_source=desired_source,
            desired_name=f" {spec.display_name}"[:80],
            desired_meta={
                "description": (
                    f"Configure OpenRouter async video generation for {spec.display_name}."
                ),
                "toggle": True,
                "manifest": {
                    "title": spec.display_name[:80],
                    "id": spec.function_id,
                    "version": "0.1.0",
                    "license": "MIT",
                },
            },
            preferred_id=spec.function_id,
            auto_install_valve="AUTO_INSTALL_VIDEO_FILTERS",
            log_label=f"OpenRouter Video Generation filter for {spec.model_id}",
            matches_candidate=_matches,
        )



    @staticmethod
    def render_openrouter_image_filter_source(
        *,
        model_id: str,
        image_model: dict[str, Any] | None = None,
        endpoint_record: list[dict[str, Any]] | dict[str, Any] | None = None,
    ) -> str:
        from .image_filter_renderer import (
            build_image_model_filter_spec,
            render_image_model_filter_source,
        )

        return render_image_model_filter_source(
            build_image_model_filter_spec(model_id, image_model, endpoint_record)
        )

    @timed
    async def ensure_openrouter_image_filter_function_ids(
        self,
        models: list[dict[str, Any]],
    ) -> dict[str, list[str]]:
        """Install one filter per image model and return its attachment list.

        Each filter offers exactly the knobs that model's published contract names. The
        seven fixed variants this replaces assigned knobs by a regex on the model id, so a
        model was handed the same ten aspect ratios whatever it actually accepted.
        """
        from ..models.registry import ModelFamily, OpenRouterModelRegistry

        installed: dict[str, list[str]] = {}
        for model in models:
            model_id = model.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            model_id = model_id.strip()
            try:
                if not ModelFamily.supports("image_output", model_id):
                    continue
            except (AttributeError, TypeError):
                continue

            original_id = model.get("original_id")
            canonical_id = (
                original_id if isinstance(original_id, str) and original_id.strip() else model_id
            )
            spec = OpenRouterModelRegistry.spec(model_id)
            image_model = spec.get("image_model") if isinstance(spec, dict) else None
            endpoint_record = OpenRouterModelRegistry.image_endpoint(canonical_id)
            if endpoint_record is None:
                endpoint_record = OpenRouterModelRegistry.image_endpoint(model_id)
            if not isinstance(image_model, dict):
                image_model = dict(model)

            try:
                function_id = await self._ensure_single_image_filter_function_id(
                    model_id=canonical_id,
                    image_model=image_model,
                    endpoint_record=endpoint_record,
                )
            except Exception as exc:
                # One model's install failure costs that model its filter and nothing
                # else. The catch is deliberately broad: the install path reaches Open
                # WebUI's database, whose driver errors are not in any tuple this module
                # could enumerate, and one of them must not skip every remaining model.
                self.logger.warning(
                    "Image filter install failed for %r: %s", canonical_id, exc, exc_info=True
                )
                continue

            if function_id:
                installed[model_id] = [function_id]
                if isinstance(original_id, str) and original_id.strip() and original_id != model_id:
                    installed[original_id] = [function_id]

        await self._retire_variant_image_filters()
        return installed

    async def _image_filter_exists(self, function_id: str) -> bool:
        """Whether a filter row is already installed under this id."""
        try:
            from open_webui.models.functions import Functions

            return await Functions.get_function_by_id(function_id) is not None
        except Exception as exc:
            self.logger.debug(
                "Could not check for an existing filter %r: %s", function_id, exc, exc_info=True
            )
            return False

    async def _retire_variant_image_filters(self) -> None:
        """Deactivate image filters left over from the fixed-variant design.

        Those rows carry the image marker but no ``IMAGE_FILTER_MODEL_ID``, so nothing
        re-selects or overwrites them. The generic one has no model gate at all, so left
        active and attached it keeps writing its invented ratio list into every request
        for the model -- and it stays attached precisely to the models that now get no
        filter of their own.
        """
        try:
            from open_webui.models.functions import Functions

            rows = await Functions.get_functions_by_type("filter", active_only=True)
        except Exception as exc:
            self.logger.debug("Could not list filters to retire old ones: %s", exc, exc_info=True)
            return

        for row in rows or []:
            content = getattr(row, "content", "")
            row_id = getattr(row, "id", "")
            if not isinstance(content, str) or not row_id:
                continue
            if _OPENROUTER_IMAGE_FILTER_MARKER not in content:
                continue
            if "IMAGE_FILTER_MODEL_ID" in content:
                continue
            try:
                await Functions.update_function_by_id(row_id, {"is_active": False})
            except Exception as exc:
                self.logger.warning(
                    "Could not retire superseded image filter %r: %s", row_id, exc, exc_info=True
                )
                continue
            self.logger.info(
                "Retired superseded image filter %r; each model now has its own.", row_id
            )

    async def _ensure_single_image_filter_function_id(
        self,
        *,
        model_id: str,
        image_model: dict[str, Any] | None,
        endpoint_record: list[dict[str, Any]] | dict[str, Any] | None,
    ) -> str | None:
        from .image_filter_renderer import build_image_model_filter_spec

        spec = build_image_model_filter_spec(model_id, image_model, endpoint_record)
        if spec.knob_count == 0 and (
            not spec.contract_read or not await self._image_filter_exists(spec.function_id)
        ):
            # Nothing to offer and nothing already installed, so install nothing. If a
            # filter IS installed, fall through and overwrite it: a contract that shrank
            # to nothing must not leave the previous controls on screen, writing values
            # the model no longer accepts into every request.
            return None

        # Built with the same expression the renderer emits, not a reconstruction of it.
        # Rebuilding the literal by hand missed any id whose repr needs an escape -- an
        # invisible soft hyphen was enough -- and a filter that cannot be re-identified is
        # installed again under a new suffix on every catalog refresh.
        model_id_token = f"IMAGE_FILTER_MODEL_ID = {spec.model_id!r}"

        def _matches(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            if _OPENROUTER_IMAGE_FILTER_MARKER not in content:
                return False
            if model_id_token not in content:
                return False
            return "class Filter" in content

        desired_source = self.render_openrouter_image_filter_source(
            model_id=model_id,
            image_model=image_model,
            endpoint_record=endpoint_record,
        ).strip() + "\n"
        valid, error = self.validate_filter_source(desired_source)
        if not valid:
            raise ValueError(f"Generated OpenRouter image filter is invalid: {error}")

        return await self._ensure_filter_installed(
            desired_source=desired_source,
            desired_name=spec.display_name[:80],
            desired_meta={
                "description": (
                    f"Configure OpenRouter native image generation for {spec.display_name}."
                ),
                "toggle": True,
                "manifest": {
                    "title": spec.display_name[:80],
                    "id": spec.function_id,
                    "version": "0.1.0",
                    "license": "MIT",
                },
            },
            preferred_id=spec.function_id,
            auto_install_valve="AUTO_INSTALL_IMAGE_FILTERS",
            log_label=f"OpenRouter image filter for {spec.model_id}",
            matches_candidate=_matches,
        )

    # DIRECT UPLOADS FILTER

    @staticmethod
    def render_direct_uploads_filter_source() -> str:
        """Return the canonical OWUI filter source for the OpenRouter Direct Uploads toggle."""
        template = '''"""
title: OR Direct Uploads
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
from typing import Any, Literal

from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # noqa: BLE001 - open_webui.env does filesystem work on import
    SRC_LOG_LEVELS = {}

OWUI_OPENROUTER_PIPE_MARKER = "__MARKER__"


class DirectUploadError(Exception):
    """Rejects a direct upload; Open WebUI shows the message to the user verbatim."""


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
    def _to_int(value: Any) -> int | None:
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
        pipe_meta = meta.get("__PIPE_META_KEY__", {})
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
                raise DirectUploadError("Direct uploads: uploaded file missing a valid size.")

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
                    raise DirectUploadError(
                        f"Direct file '{name or file_id}' is too large ({size_bytes} bytes; max {self.valves.DIRECT_FILE_MAX_UPLOAD_SIZE_MB} MB)."
                    )
                total_bytes += size_bytes
                if total_bytes > total_limit:
                    raise DirectUploadError(
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
                    raise DirectUploadError(
                        f"Direct audio '{name or file_id}' is too large ({size_bytes} bytes; max {self.valves.DIRECT_AUDIO_MAX_UPLOAD_SIZE_MB} MB)."
                    )
                total_bytes += size_bytes
                if total_bytes > total_limit:
                    raise DirectUploadError(
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
                    raise DirectUploadError(
                        f"Direct video '{name or file_id}' is too large ({size_bytes} bytes; max {self.valves.DIRECT_VIDEO_MAX_UPLOAD_SIZE_MB} MB)."
                    )
                total_bytes += size_bytes
                if total_bytes > total_limit:
                    raise DirectUploadError(
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
        if diverted_any:
            body["files"] = retained
            if isinstance(__metadata__, dict):
                __metadata__["files"] = retained

        if isinstance(__metadata__, dict) and (diverted_any or warnings):
            prev_pipe_meta = __metadata__.get("__PIPE_META_KEY__")
            pipe_meta = dict(prev_pipe_meta) if isinstance(prev_pipe_meta, dict) else {}
            __metadata__["__PIPE_META_KEY__"] = pipe_meta

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
            .replace("__PIPE_META_KEY__", _PIPE_METADATA_KEY)
        )

    @timed
    async def ensure_direct_uploads_filter_function_id(self) -> str | None:
        """Ensure the OpenRouter Direct Uploads companion filter exists (and is up to date), returning its OWUI function id."""

        def _matches(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            return _DIRECT_UPLOADS_FILTER_MARKER in content and "class Filter" in content

        return await self._ensure_filter_installed(
            desired_source=self.render_direct_uploads_filter_source().strip() + "\n",
            desired_name="OR Direct Uploads",
            desired_meta={
                "description": (
                    "Bypass Open WebUI RAG for chat uploads and forward them to OpenRouter as direct file/audio/video inputs. "
                    "Enable files/audio/video via filter user valves."
                ),
                "toggle": True,
                "manifest": {
                    "title": "OR Direct Uploads",
                    "id": _DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID,
                    "version": "0.1.0",
                    "license": "MIT",
                },
            },
            preferred_id=_DIRECT_UPLOADS_FILTER_PREFERRED_FUNCTION_ID,
            auto_install_valve="AUTO_INSTALL_DIRECT_UPLOADS_FILTER",
            log_label="OpenRouter Direct Uploads filter",
            matches_candidate=_matches,
        )

    # PROVIDER ROUTING FILTER

    @staticmethod
    def compute_provider_routing_hash(
        admin_models: str,
        user_models: str,
        provider_map: dict[str, dict[str, list[str]]],
    ) -> str:
        """Compute hash of provider routing state to detect changes."""
        admin_sorted = sorted([m.strip() for m in admin_models.split(",") if m.strip()])
        user_sorted = sorted([m.strip() for m in user_models.split(",") if m.strip()])

        relevant_slugs = set(admin_sorted) | set(user_sorted)
        provider_data = {
            slug: provider_map.get(slug, {})
            for slug in sorted(relevant_slugs)
        }

        transports = {
            slug: FilterManager.model_transport(slug) for slug in sorted(relevant_slugs)
        }
        data = f"{admin_sorted}|{user_sorted}|{provider_data}|{transports}"
        return hashlib.md5(data.encode()).hexdigest()

    @staticmethod
    def model_transport(model_slug: str) -> str:
        from ..models.registry import OpenRouterModelRegistry, uses_dedicated_image_api

        spec = OpenRouterModelRegistry.spec(model_slug)
        if not isinstance(spec, dict):
            return "chat"
        if "video_generation" in set(spec.get("features") or ()):
            return "video"
        return "image" if uses_dedicated_image_api(spec) else "chat"

    @staticmethod
    def _routing_controls(transport: str) -> frozenset[str]:
        accepted = TRANSPORT_PROVIDER_KEYS.get(transport, CHAT_PROVIDER_KEYS)
        return frozenset(
            control
            for control, key in _ROUTING_CONTROL_KEYS.items()
            if key in accepted
        )

    @staticmethod
    def _generate_inlet_logic(visibility: str, transport: str = "chat") -> str:
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
            logic += '''
        # OWUI injects user valves into __user__["valves"], not self.user_valves
        user_valves = __user__.get("valves") if __user__ else None
        if user_valves is not None:
            user_set = user_valves.model_fields_set
'''

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

'''
        drawn = FilterManager._routing_controls(transport)
        if "ORDER" in drawn:
            logic += '''        # ORDER: Map display value to provider slug list using _ORDER_MAP
        order_display = get_literal("ORDER")
        if order_display:
            order_slugs = _ORDER_MAP.get(order_display)
            if order_slugs:
                provider["order"] = order_slugs
            else:
                self.log.warning("ORDER value %r not found in _ORDER_MAP", order_display)

'''
        if "ONLY" in drawn:
            logic += '''        # ONLY: Map display name to slug using _PROVIDER_MAP
        only_display = get_literal("ONLY")
        if only_display:
            only_slug = _PROVIDER_MAP.get(only_display)
            if only_slug:
                provider["only"] = [only_slug]
            else:
                self.log.warning("ONLY value %r not found in _PROVIDER_MAP", only_display)

'''
        if "IGNORE" in drawn:
            logic += '''        # IGNORE: Map display name to slug using _PROVIDER_MAP
        ignore_display = get_literal("IGNORE")
        if ignore_display:
            ignore_slug = _PROVIDER_MAP.get(ignore_display)
            if ignore_slug:
                provider["ignore"] = [ignore_slug]
            else:
                self.log.warning("IGNORE value %r not found in _PROVIDER_MAP", ignore_display)

'''
        if "SORT" in drawn:
            logic += '''        # SORT: a bare strategy, or an object when a partition is chosen too
        sort_val = get_literal("SORT")
        sort_partition = get_literal("SORT_PARTITION")
        if sort_val and sort_partition:
            provider["sort"] = {"by": sort_val, "partition": sort_partition}
        elif sort_val:
            provider["sort"] = sort_val
        elif sort_partition:
            provider["sort"] = {"partition": sort_partition}

'''
        if "QUANTIZATION" in drawn:
            logic += '''        # QUANTIZATION: Literal dropdown maps directly
        quant_val = get_literal("QUANTIZATION")
        if quant_val:
            provider["quantizations"] = [quant_val]

'''
        if "DATA_COLLECTION" in drawn:
            logic += '''        # DATA_COLLECTION: Literal values map directly
        data_collection = get_literal("DATA_COLLECTION")
        if data_collection:
            provider["data_collection"] = data_collection

'''
        if "ALLOW_FALLBACKS" in drawn:
            logic += '''        allow_fallbacks = get_bool("ALLOW_FALLBACKS", api_default=True)
        if allow_fallbacks is not None:
            provider["allow_fallbacks"] = allow_fallbacks

'''
        if "REQUIRE_PARAMETERS" in drawn:
            logic += '''        require_params = get_bool("REQUIRE_PARAMETERS", api_default=False)
        if require_params is not None:
            provider["require_parameters"] = require_params

'''
        if "ZDR" in drawn:
            logic += '''        zdr = get_bool("ZDR", api_default=False)
        if zdr is not None:
            provider["zdr"] = zdr

'''
        if "ENFORCE_DISTILLABLE_TEXT" in drawn:
            logic += '''        distillable = get_bool("ENFORCE_DISTILLABLE_TEXT", api_default=False)
        if distillable is not None:
            provider["enforce_distillable_text"] = distillable

'''
        if "MIN_THROUGHPUT" in drawn:
            logic += '''        min_throughput = get_float("MIN_THROUGHPUT")
        if min_throughput > 0:
            provider["preferred_min_throughput"] = min_throughput

'''
        if "MAX_LATENCY" in drawn:
            logic += '''        max_latency = get_float("MAX_LATENCY")
        if max_latency > 0:
            provider["preferred_max_latency"] = max_latency

'''
        if "MAX_PRICE_PROMPT" in drawn:
            logic += '''        # Price limits
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

'''
        logic += '''        # Inject into metadata if we have any provider settings
        if provider:
            if __metadata__ is None:
                __metadata__ = {}
            pipe_meta = __metadata__.setdefault("__PIPE_META_KEY__", {})
            pipe_meta["provider"] = provider
            self.log.debug("Injected provider routing: %s", provider)
'''
        return logic

    @staticmethod
    def _render_provider_routing_filter_source(
        model_slug: str,
        providers: list[str],
        quantizations: list[str],
        visibility: str,
        *,
        short_name: str = "",
        provider_names: dict[str, str] | None = None,
        transport: str = "chat",
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

        if not isinstance(model_slug, str) or not model_slug:
            raise ValueError("model_slug must be a non-empty string")
        safe_model_slug_escaped = json.dumps(model_slug)[1:-1]

        display_name = short_name.strip() if short_name else model_slug.split("/")[-1]
        safe_display_name = FilterManager.validate_provider_name(display_name, slug=model_slug)
        safe_display_name_escaped = json.dumps(safe_display_name)[1:-1]

        marker = f"{_PROVIDER_ROUTING_FILTER_MARKER_PREFIX}{model_slug}:{_PROVIDER_ROUTING_FILTER_MARKER_VERSION}"
        safe_marker_escaped = json.dumps(marker)[1:-1]

        safe_providers = [
            p for p in providers
            if isinstance(p, str) and _PROVIDER_SLUG_PATTERN.match(p) and len(p) <= 64
        ][:_PROVIDER_ROUTING_MAX_PROVIDERS]

        prov_names = provider_names or {}
        provider_display_options: list[str] = []
        provider_slug_map_entries: list[str] = []
        for pslug in safe_providers:
            disp = prov_names.get(pslug, pslug.replace("-", " ").title())
            safe_disp = FilterManager.validate_provider_name(disp, slug=pslug)
            provider_display_options.append(safe_disp)
            provider_slug_map_entries.append(f'    {FilterManager.safe_literal_string(safe_disp)}: {FilterManager.safe_literal_string(pslug)}')

        no_pref = "(no preference)"
        only_ignore_options = [no_pref] + provider_display_options
        only_ignore_literal = ", ".join(FilterManager.safe_literal_string(opt) for opt in only_ignore_options)

        # Build provider map code block
        provider_map_code = "{\n" + ",\n".join(provider_slug_map_entries) + "\n}" if provider_slug_map_entries else "{}"

        order_display_options: list[str] = []
        order_map_entries: list[str] = []

        display_to_slug = dict(zip(provider_display_options, safe_providers))

        if len(provider_display_options) <= _PROVIDER_ROUTING_ORDER_PERMUTATION_MAX:
            for perm in itertools.permutations(provider_display_options):
                perm_disp = " > ".join(perm)
                order_display_options.append(perm_disp)
                perm_slugs = [display_to_slug[d] for d in perm]
                slugs_literal = ", ".join(FilterManager.safe_literal_string(s) for s in perm_slugs)
                order_map_entries.append(f'    {FilterManager.safe_literal_string(perm_disp)}: [{slugs_literal}]')
        else:
            for disp in provider_display_options:
                first_disp = f"{disp} first"
                order_display_options.append(first_disp)
                slug_literal = FilterManager.safe_literal_string(display_to_slug[disp])
                order_map_entries.append(f'    {FilterManager.safe_literal_string(first_disp)}: [{slug_literal}]')

        order_options = [no_pref] + order_display_options
        order_literal = ", ".join(FilterManager.safe_literal_string(opt) for opt in order_options)

        # Build order map code block
        order_map_code = "{\n" + ",\n".join(order_map_entries) + "\n}" if order_map_entries else "{}"

        safe_quantizations = [
            q for q in quantizations
            if isinstance(q, str) and _QUANTIZATION_PATTERN.match(q) and len(q) <= 32
        ][:_PROVIDER_ROUTING_MAX_PROVIDERS]

        quant_options = [no_pref] + safe_quantizations
        quantizations_literal = ", ".join(FilterManager.safe_literal_string(q) for q in quant_options)

        toggle_value = "False" if visibility == "admin" else "True"

        drawn = FilterManager._routing_controls(transport)
        control_lines = {
            "ORDER": f'        ORDER: Literal[{order_literal}] = Field(default=_NO_PREF, description="Provider priority order")',
            "ALLOW_FALLBACKS": '        ALLOW_FALLBACKS: bool = Field(default=True, description="Allow backup providers if preferred unavailable")',
            "REQUIRE_PARAMETERS": '        REQUIRE_PARAMETERS: bool = Field(default=False, description="Only use providers supporting all request params")',
            "DATA_COLLECTION": '        DATA_COLLECTION: Literal[_NO_PREF, "allow", "deny"] = Field(default=_NO_PREF, description="Data collection policy")',
            "ZDR": '        ZDR: bool = Field(default=False, description="Zero Data Retention - only ZDR endpoints")',
            "ENFORCE_DISTILLABLE_TEXT": '        ENFORCE_DISTILLABLE_TEXT: bool = Field(default=False, description="Only use providers whose author allows text distillation")',
            "ONLY": f'        ONLY: Literal[{only_ignore_literal}] = Field(default=_NO_PREF, description="Use only this provider")',
            "IGNORE": f'        IGNORE: Literal[{only_ignore_literal}] = Field(default=_NO_PREF, description="Avoid this provider")',
            "QUANTIZATION": f'        QUANTIZATION: Literal[{quantizations_literal}] = Field(default=_NO_PREF, description="Filter by quantization")',
            "SORT": '        SORT: Literal[_NO_PREF, "price", "throughput", "latency", "exacto"] = Field(default=_NO_PREF, description="Sort providers by; exacto favours endpoints that reproduce the model most faithfully")',
            "SORT_PARTITION": '        SORT_PARTITION: Literal[_NO_PREF, "model", "none"] = Field(default=_NO_PREF, description="Whether sorting groups endpoints by model first (model) or ranks them all together (none)")',
            "MIN_THROUGHPUT": '        MIN_THROUGHPUT: float = Field(default=0, ge=0, description="Min throughput (tokens/sec), 0=no pref")',
            "MAX_LATENCY": '        MAX_LATENCY: float = Field(default=0, ge=0, description="Max latency (seconds), 0=no pref")',
            "MAX_PRICE_PROMPT": '        MAX_PRICE_PROMPT: float = Field(default=0, ge=0, description="Max price for prompt ($/M tokens), 0=no limit")',
            "MAX_PRICE_COMPLETION": '        MAX_PRICE_COMPLETION: float = Field(default=0, ge=0, description="Max price for completion ($/M tokens), 0=no limit")',
            "MAX_PRICE_IMAGE": '        MAX_PRICE_IMAGE: float = Field(default=0, ge=0, description="Max price per image ($/image), 0=no limit")',
            "MAX_PRICE_AUDIO": '        MAX_PRICE_AUDIO: float = Field(default=0, ge=0, description="Max price for audio ($/unit), 0=no limit")',
            "MAX_PRICE_REQUEST": '        MAX_PRICE_REQUEST: float = Field(default=0, ge=0, description="Max price per request ($/request), 0=no limit")',
        }
        if set(control_lines) != set(_ROUTING_CONTROL_KEYS):
            raise ValueError(
                "every routing control needs both a field and a request field to write; "
                f"unpaired: {sorted(set(control_lines) ^ set(_ROUTING_CONTROL_KEYS))}"
            )
        rendered_controls = "\n".join(
            line for name, line in control_lines.items() if name in drawn
        )
        guarded = [
            name
            for name in ("ORDER", "ONLY", "IGNORE", "QUANTIZATION", "SORT", "SORT_PARTITION")
            if name in drawn
        ]
        guarded_literal = ", ".join(json.dumps(name) for name in guarded)
        stale_choice_guard = f'''
        @field_validator({guarded_literal}, mode="before")
        @classmethod
        def _coerce_stale_choice(cls, value: Any, info: ValidationInfo) -> Any:
            options = get_args(cls.model_fields[info.field_name].annotation)
            return value if value in options else _NO_PREF
''' if guarded else ""

        valves_class = ""
        if visibility in ("admin", "both"):
            valves_class = f'''
    class Valves(BaseModel):
        """Admin-level provider routing preferences."""
{rendered_controls}
{stale_choice_guard}'''

        user_valves_class = ""
        if visibility in ("user", "both"):
            user_valves_class = f'''
    class UserValves(BaseModel):
        """User-level provider routing preferences (can override admin defaults)."""
{rendered_controls}
{stale_choice_guard}'''

        # Generate init based on visibility
        init_body = "        self.log = logging.getLogger(f\"openrouter.provider.{MODEL_SLUG}\")\n        self.log.setLevel(SRC_LOG_LEVELS.get(\"OPENAI\", logging.INFO))"
        if visibility in ("admin", "both"):
            init_body += "\n        self.valves = self.Valves()"
        if visibility in ("user", "both"):
            init_body += "\n        self.user_valves = self.UserValves()"

        inlet_logic = FilterManager._generate_inlet_logic(visibility, transport)

        return (f'''"""
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
from typing import Any, Literal, get_args

from pydantic import BaseModel, Field, ValidationInfo, field_validator

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # noqa: BLE001 - open_webui.env does filesystem work on import
    SRC_LOG_LEVELS = {{}}

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
''').replace("__PIPE_META_KEY__", _PIPE_METADATA_KEY)

    async def ensure_provider_routing_filters(
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
            from open_webui.models.functions import (
                FunctionForm,
                FunctionMeta,
                Functions,
            )
        except ImportError:
            return {}
        except Exception:
            logging.getLogger(__name__).warning(
                "open_webui.models.functions failed to import for a reason other than absence; "
                "the features that depend on it are now disabled",
                exc_info=True,
            )
            return {}

        # Parse model lists
        admin_models = {m.strip() for m in admin_models_csv.split(",") if m.strip()}
        user_models = {m.strip() for m in user_models_csv.split(",") if m.strip()}

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

        try:
            all_filters = await Functions.get_functions_by_type("filter", active_only=False)
        except Exception:
            self.logger.exception(
                "Cannot enumerate OWUI filter functions; aborting provider routing sync "
                "to avoid creating duplicate filters"
            )
            return {}

        filters_by_slug: dict[str, list[Any]] = {}
        for f in all_filters:
            content = getattr(f, "content", "") or ""
            if _PROVIDER_ROUTING_FILTER_MARKER_PREFIX in content:
                # Extract model slug from marker
                for line in content.split("\n"):
                    if _PROVIDER_ROUTING_FILTER_MARKER_PREFIX in line:
                        try:
                            marker_val = line.split("=", 1)[1].strip().strip('"').strip("'")
                            parts = marker_val.split(":", 2)
                            if (
                                len(parts) == 3
                                and parts[0] == _PIPE_METADATA_KEY
                                and parts[1] == "provider_routing"
                                and ":" in parts[2]
                            ):
                                slug = parts[2].rsplit(":", 1)[0]
                                if slug:
                                    filters_by_slug.setdefault(slug, []).append(f)
                        except IndexError:
                            self.logger.debug(
                                "Ignoring malformed provider routing marker in filter %s",
                                getattr(f, "id", "?"),
                            )
                        break

        existing_filters: dict[str, Any] = {}
        orphan_filters: list[Any] = []
        for slug, found in filters_by_slug.items():
            canonical_id = f"{_PROVIDER_ROUTING_FILTER_ID_PREFIX}{self.sanitize_model_for_filter_id(slug)}"
            canonical = next(
                (f for f in found if getattr(f, "id", "") == canonical_id), found[0]
            )
            existing_filters[slug] = canonical
            orphan_filters.extend(f for f in found if f is not canonical)

        slug_to_filter_id: dict[str, str] = {}

        missing_filters = {
            slug
            for slug in all_models
            if slug not in existing_filters
            and (provider_map.get(slug) or {}).get("providers")
            and self._routing_controls(self.model_transport(slug))
        }

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

        if missing_filters:
            self.logger.info(
                "Provider routing filters missing for %d model(s), recreating: %s",
                len(missing_filters),
                ", ".join(sorted(missing_filters)),
            )

        created = 0
        updated = 0
        undeliverable: list[str] = []
        for slug, visibility in model_visibility.items():
            model_info = provider_map.get(slug, {})
            providers = model_info.get("providers", [])
            quantizations = model_info.get("quantizations", [])
            raw_short_name = model_info.get("short_name", "")
            short_name: str = raw_short_name if isinstance(raw_short_name, str) else ""
            raw_prov_names = model_info.get("provider_names", {})
            prov_names: dict[str, str] = raw_prov_names if isinstance(raw_prov_names, dict) else {}

            if not providers:
                self.logger.warning("Skipping filter for %s: no providers found in catalog (check slug spelling)", slug)
                continue

            transport = self.model_transport(slug)
            if not self._routing_controls(transport):
                undeliverable.append(slug)
                self.logger.warning(
                    "Not installing a provider routing filter for %s: its request format "
                    "carries none of these settings, so every control would be accepted "
                    "and then dropped.",
                    slug,
                )
                continue

            safe_id = self.sanitize_model_for_filter_id(slug)
            filter_id = f"{_PROVIDER_ROUTING_FILTER_ID_PREFIX}{safe_id}"

            desired_source = self._render_provider_routing_filter_source(
                slug, providers, quantizations, visibility,
                short_name=short_name,
                provider_names=prov_names,
                transport=transport,
            ).strip() + "\n"

            is_valid, validation_error = self.validate_filter_source(desired_source)
            if not is_valid:
                self.logger.error(
                    "Generated filter for %s failed syntax validation: %s. Skipping.",
                    slug,
                    validation_error,
                )
                continue

            display_name = short_name if short_name else slug.split("/")[-1]
            # Sanitize for safety
            safe_display = self.validate_provider_name(display_name, slug=slug)
            desired_name = f"Provider: {safe_display}"
            desired_meta = {
                "description": f"Provider routing preferences for {slug}",
                "toggle": visibility != "admin",
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
                    await Functions.update_function_by_id(
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
                    await Functions.update_function_by_id(existing_id, {"is_active": True})
                # Track for attachment
                if existing_id:
                    slug_to_filter_id[slug] = existing_id
            else:
                # Create new filter
                candidate_id = filter_id
                suffix = 0
                while True:
                    existing_func = await Functions.get_function_by_id(candidate_id)
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
                    created_func = await Functions.insert_new_function("", "filter", form)
                    if created_func:
                        await Functions.update_function_by_id(candidate_id, {"is_active": True, "is_global": False})
                        created += 1
                        self.logger.info("Created provider routing filter: %s", candidate_id)
                        # Track for attachment
                        slug_to_filter_id[slug] = candidate_id

        disabled = 0
        for orphan in orphan_filters:
            orphan_id = getattr(orphan, "id", "")
            if orphan_id:
                await Functions.update_function_by_id(orphan_id, {"is_active": False})
                disabled += 1
                self.logger.warning("Disabled duplicate provider routing filter: %s", orphan_id)

        for slug, existing in existing_filters.items():
            if slug in undeliverable or slug not in all_models:
                existing_id = getattr(existing, "id", "")
                if existing_id:
                    await Functions.update_function_by_id(existing_id, {"is_active": False})
                    disabled += 1
                    self.logger.info("Disabled provider routing filter: %s", existing_id)

        if created or updated or disabled:
            self.logger.info(
                "Provider routing filters: created=%d, updated=%d, disabled=%d (total models=%d)",
                created, updated, disabled, len(all_models),
            )

        FilterManager._provider_routing_state_hash = current_hash
        self.logger.debug("Provider routing state hash updated: %s", current_hash[:8])

        self.logger.info(
            "Returning %d provider routing filter mappings for attachment: %r",
            len(slug_to_filter_id),
            slug_to_filter_id,
        )
        return slug_to_filter_id
