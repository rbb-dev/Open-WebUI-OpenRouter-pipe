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

import ast
import hashlib
import itertools
import json
import logging
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from functools import lru_cache
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple

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
from ..core.utils import (
    _ADAPTER_CACHE,
    _KEEP_WHAT_STILL_FITS,
    _clean_str,
)
from ..core.utils import OWUI_FUNCTION_ID_ILLEGAL_RE as _MODEL_FILTER_ID_RE
from ..core.warn_latch import bounded_warn_level, warn_level
from ..integrations.provider_options import CHAT_PROVIDER_KEYS, TRANSPORT_PROVIDER_KEYS
from ..models.catalog_manager import WEB_TOOL_SWITCHES, every_web_tool_is_off

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

_PRIORITY_FIELD = (
    '        priority: int = Field(\n'
    '            default=0,\n'
    '            description="Priority level for the filter operations.",\n'
    '        )'
)

if TYPE_CHECKING:
    from ..pipe import Pipe

_PROVIDER_NAME_ALLOWLIST_RE = re.compile(r"[^A-Za-z0-9 \-_.]")
_PROVIDER_NAME_COLLAPSE_RE = re.compile(r"[ _]{2,}")

_warned_stale_filter_rows: set[str] = set()
_warned_write_refusals: dict[str, float] = {}
_warned_image_filter_installs: OrderedDict[str, None] = OrderedDict()
_warned_video_filter_installs: OrderedDict[str, None] = OrderedDict()
_PER_MODEL_INSTALL_WARN_WINDOW = 300

_PIPE_OFF_META_KEY = "openrouter_pipe:switched_off_by_pipe"
_PIPE_OFF_STAMP_META_KEY = "openrouter_pipe:switched_off_at"
_PIPE_INSTALLED_META_KEY = "openrouter_pipe:installed_by"
_DISPLAY_NAME_MAX_CHARS = 80
_PIPE_OFF_LANDED_AT: dict[str, int] = {}
_OFF_STAMP_SETTLE_ATTEMPTS = 2


class _FilterRows(NamedTuple):
    all_rows: list[Any] | None
    active_rows: list[Any] | None
    available: bool


def _stored_meta(row: Any) -> dict[str, Any]:
    stored = getattr(row, "meta", None)
    dump = getattr(stored, "model_dump", None)
    if callable(dump):
        stored = dump()
    return stored if isinstance(stored, dict) else {}


def _manifest_id_for(desired_meta: dict[str, Any], function_id: str) -> dict[str, Any]:
    manifest = desired_meta.get("manifest")
    if not isinstance(manifest, dict) or "id" not in manifest or manifest["id"] == function_id:
        return desired_meta
    return {**desired_meta, "manifest": {**manifest, "id": function_id}}


def _switched_off_by_pipe(row: Any) -> bool:
    return bool(_stored_meta(row).get(_PIPE_OFF_META_KEY))


def _is_already_switched_off(row: Any) -> bool:
    return not getattr(row, "is_active", False) and _switched_off_by_pipe(row)


def _installed_by(row: Any) -> str:
    value = _stored_meta(row).get(_PIPE_INSTALLED_META_KEY)
    return value if isinstance(value, str) else ""


def _owned_by(row: Any, owner: str) -> bool:
    return _installed_by(row) == owner


def _claimable_by(row: Any, owner: str) -> bool:
    return _installed_by(row) in ("", owner)


def _merged_meta(
    row: Any,
    desired_meta: dict[str, Any],
    *,
    off_by_pipe: bool | None = None,
) -> dict[str, Any]:
    merged = {**_stored_meta(row), **desired_meta}
    if off_by_pipe is True:
        merged[_PIPE_OFF_META_KEY] = True
        merged[_PIPE_OFF_STAMP_META_KEY] = int(time.time())
    elif off_by_pipe is False:
        merged.pop(_PIPE_OFF_META_KEY, None)
        merged.pop(_PIPE_OFF_STAMP_META_KEY, None)
    return merged


def _unconfirmed_switch_off_clause(row: Any) -> str:
    if _switched_off_by_pipe(row) and not _pipe_owns_the_off(row):
        return (
            " (the pipe switched this off itself, and this pass could not confirm it; "
            "the switch-off stamp is not durable)"
        )
    return ""


def _maintained_meta(row: Any, desired_meta: dict[str, Any]) -> dict[str, Any]:
    return _merged_meta(row, desired_meta, off_by_pipe=None if not _switch_on(row) else False)


def _pipe_owns_the_off(row: Any) -> bool:
    stored = _stored_meta(row)
    if not stored.get(_PIPE_OFF_META_KEY):
        return False
    stamp = stored.get(_PIPE_OFF_STAMP_META_KEY)
    if not isinstance(stamp, int) or isinstance(stamp, bool):
        return True
    updated_at = int(getattr(row, "updated_at", 0) or 0)
    if updated_at <= stamp:
        return True
    return updated_at == _PIPE_OFF_LANDED_AT.get(str(getattr(row, "id", "") or ""))


def _switch_on(row: Any) -> bool:
    return bool(getattr(row, "is_active", False)) or _pipe_owns_the_off(row)


def _operator_re_enabled_the_pipe_off(row: Any) -> bool:
    return (
        bool(getattr(row, "is_active", False))
        and _switched_off_by_pipe(row)
        and not _pipe_owns_the_off(row)
    )


def switched_off_meta(row: Any) -> dict[str, Any]:
    return _merged_meta(row, {}, off_by_pipe=True)


def _stored_source(row: Any) -> str:
    content = getattr(row, "content", "")
    if not isinstance(content, str):
        return "\n"
    return content.strip() + "\n"


_REFUSED_FILTER_WRITES: set[str] = set()


class _WriteOutcome:
    __slots__ = ("refused",)

    def __init__(self) -> None:
        self.refused = False


async def _write_function(Functions, function_id, updates, what, logger, raised=None, *, settle: bool = True, landed_out: list | None = None) -> bool:
    try:
        landed = await Functions.update_function_by_id(function_id, updates)
        if landed_out is not None:
            landed_out.append(landed)
    except Exception as exc:  # noqa: BLE001 - a database driver's own error type is not enumerable here
        if raised is not None:
            raised.append(exc)
        landed = None
    if landed is None:
        _REFUSED_FILTER_WRITES.add(str(function_id))
        logger.log(
            warn_level(_warned_write_refusals, f"refused:{function_id}", cooldown_s=3600),
            "Open WebUI refused the write to %s while %s; it will be retried on the next pass.",
            function_id,
            what,
        )
        return False
    if settle:
        await _settle_the_off_stamp(Functions, function_id, updates, landed, logger)
    return True


async def _settle_the_off_stamp(Functions, function_id, updates, landed, logger) -> None:
    desired_meta = updates.get("meta") if isinstance(updates, dict) else None
    if not isinstance(desired_meta, dict) or desired_meta.get(_PIPE_OFF_META_KEY) is not True:
        return
    stamp = desired_meta.get(_PIPE_OFF_STAMP_META_KEY)
    if not isinstance(stamp, int) or isinstance(stamp, bool):
        return
    landed_at = int(getattr(landed, "updated_at", 0) or 0)
    _PIPE_OFF_LANDED_AT[str(function_id)] = landed_at
    row = landed
    for _ in range(_OFF_STAMP_SETTLE_ATTEMPTS):
        if landed_at <= int(stamp):
            return
        settled: list = []
        await _write_function(
            Functions,
            function_id,
            {
                "meta": {
                    **_stored_meta(row),
                    _PIPE_OFF_STAMP_META_KEY: landed_at,
                },
            },
            f"settling the switch-off stamp on {function_id} to the second the write landed",
            logger,
            settle=False,
            landed_out=settled,
        )
        if not settled or settled[0] is None:
            return
        row = settled[0]
        stamp = landed_at
        landed_at = int(getattr(row, "updated_at", 0) or 0)
        _PIPE_OFF_LANDED_AT[str(function_id)] = landed_at
    if landed_at > int(stamp):
        logger.log(
            warn_level(_warned_stale_filter_rows, f"off_stamp:{function_id}"),
            "The switch-off stamp on %s names second %s while Open WebUI last wrote that "
            "row at %s.",
            function_id,
            stamp,
            landed_at,
        )


def a_filter_write_was_refused(family_id: str = "") -> bool:
    if not family_id:
        return bool(_REFUSED_FILTER_WRITES)
    return any(fid.startswith(family_id) for fid in _REFUSED_FILTER_WRITES)


def _row_needs_update(row: Any, desired_name: str, desired_meta: dict[str, Any]) -> bool:
    desired_is_active = _switch_on(row)
    return (
        bool(getattr(row, "is_active", False)) != desired_is_active
        or bool(getattr(row, "is_global", False))
        or (getattr(row, "name", "") or "") != desired_name
        or (getattr(row, "type", "") or "") != "filter"
        or _maintained_meta(row, desired_meta) != _meta_dict(row)
    )


def _meta_dict(row: Any) -> dict[str, Any]:
    stored = getattr(row, "meta", None)
    dump = getattr(stored, "model_dump", None)
    if callable(dump):
        try:
            stored = dump()
        except Exception:  # noqa: BLE001 - any pydantic/meta failure means "not a dict"
            stored = None
    return stored if isinstance(stored, dict) else {}


def _is_web_tools_filter(content: Any) -> bool:
    return (
        isinstance(content, str)
        and _OPENROUTER_WEB_TOOLS_FILTER_MARKER in content
        and "class Filter" in content
    )


def _is_filter_carrying(content: Any, marker: str) -> bool:
    return isinstance(content, str) and marker in content and "class Filter" in content


def _no_search_suffix(switched_off) -> str:
    return (
        " Chats using it get no web search at all until you do."
        if "WEB_SEARCH" in switched_off
        else ""
    )


def _is_video_gen_filter(content: Any) -> bool:
    return _is_filter_carrying(content, _OPENROUTER_VIDEO_GEN_FILTER_MARKER)


@lru_cache(maxsize=32)
def _offered_web_tools(content: str) -> frozenset[str] | None:
    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    toggles = {toggle for _, toggle, _ in WEB_TOOL_SWITCHES}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Filter":
            for inner in node.body:
                if isinstance(inner, ast.ClassDef) and inner.name == "UserValves":
                    declared = [
                        (stmt.target.id, stmt.annotation)
                        for stmt in inner.body
                        if isinstance(stmt, ast.AnnAssign)
                        and isinstance(stmt.target, ast.Name)
                    ]
                    if any(
                        isinstance(annotation, ast.Name) and annotation.id == "bool" and name not in toggles
                        for name, annotation in declared
                    ):
                        return None
                    return frozenset(name for name, _ in declared if name in toggles)
    return None

_PROVIDER_ROUTING_OWNER_PREFIX = "OWUI_PIPE_OWNER"


def _row_owner(row: Any) -> str:
    content = getattr(row, "content", None)
    if not isinstance(content, str) or not content:
        return ""
    start = 0
    while True:
        idx = content.find(_PROVIDER_ROUTING_OWNER_PREFIX, start)
        if idx < 0:
            return ""
        if idx == 0 or content[idx - 1] == "\n":
            end = content.find("\n", idx)
            line = content[idx:] if end < 0 else content[idx:end]
            _, _, raw = line.partition("=")
            return raw.strip().strip('"').strip("'")
        start = idx + 1


def _routing_display_names(model_info: dict[str, Any]) -> tuple[str, dict[str, str]]:
    raw_prov_names = model_info.get("provider_names", {})
    prov_names: dict[str, str] = raw_prov_names if isinstance(raw_prov_names, dict) else {}
    raw_short_name = model_info.get("short_name", "")
    short_name: str = raw_short_name if isinstance(raw_short_name, str) else ""
    return short_name, prov_names


def _is_pipe_video_filter_row(content: Any, row_id: Any) -> bool:
    if not isinstance(content, str) or not isinstance(row_id, str) or not row_id:
        return False
    if not row_id.startswith("openrouter_video_"):
        return False
    return _OPENROUTER_VIDEO_GEN_FILTER_MARKER in content


_IMAGE_FILTER_MODEL_ID_TOKEN = "IMAGE_FILTER_MODEL_ID"
_IMAGE_FILTER_LINE_BREAKS = "\n\r\v\f\x1c\x1d\x1e\x85  "


def _line_break_before(content: str, at: int) -> int:
    start = 0
    for char in _IMAGE_FILTER_LINE_BREAKS:
        found = content.rfind(char, 0, at)
        if found >= 0:
            start = max(start, found + 1)
    return start


def _line_break_after(content: str, at: int) -> int:
    end = len(content)
    for char in _IMAGE_FILTER_LINE_BREAKS:
        found = content.find(char, at)
        if found >= 0:
            end = min(end, found)
    return end


def _stored_image_model_id(content: str) -> str | None:
    import ast

    if not content:
        return None
    start = 0
    while True:
        idx = content.find(_IMAGE_FILTER_MODEL_ID_TOKEN, start)
        if idx < 0:
            return None
        start = idx + 1
        if content[_line_break_before(content, idx) : idx].strip():
            continue
        rest = content[idx : _line_break_after(content, idx)]
        name, sep, rhs = rest.partition("=")
        if not sep or name.strip() != _IMAGE_FILTER_MODEL_ID_TOKEN:
            continue
        try:
            value = ast.literal_eval(rhs.strip())
        except (ValueError, SyntaxError):
            return None
        return value if isinstance(value, str) else None


def _row_is_off_identity(content: str, row_id: str) -> bool:
    from .image_filter_renderer import sanitize_image_filter_id

    if not isinstance(row_id, str) or not row_id.startswith("openrouter_image_filter_"):
        return False
    stored = _stored_image_model_id(content)
    return stored is not None and sanitize_image_filter_id(stored) != row_id


def _image_model_panel_id(content: str) -> str | None:
    from .image_filter_renderer import sanitize_image_filter_id

    stored = _stored_image_model_id(content)
    return sanitize_image_filter_id(stored) if stored is not None else None


def _index_active_image_panels(rows: list[Any]) -> dict[str, list[Any]]:
    panels: dict[str, list[Any]] = {}
    for row in rows or []:
        content = getattr(row, "content", "")
        row_id = getattr(row, "id", "")
        if not isinstance(content, str) or not row_id:
            continue
        if "IMAGE_FILTER_MODEL_ID" not in content:
            continue
        if not bool(getattr(row, "is_active", False)):
            continue
        if _row_is_off_identity(content, row_id):
            continue
        panel = _image_model_panel_id(content)
        if panel is None:
            continue
        panels.setdefault(panel, []).append(row)
    return panels


def _on_identity_active_sibling_exists(
    rows: list[Any], row_id: Any, row_content: str, panels: dict[str, list[Any]]
) -> bool:
    model_panel_id = _image_model_panel_id(row_content) if isinstance(row_content, str) else None
    if model_panel_id is None:
        return False
    return any(getattr(r, "id", "") != row_id for r in panels.get(model_panel_id, ()))


_AUTO_INSTALL_FAMILY_MARKERS: tuple[tuple[str, str], ...] = (
    ("AUTO_INSTALL_WEB_TOOLS_FILTER", _OPENROUTER_WEB_TOOLS_FILTER_MARKER),
    ("AUTO_INSTALL_IMAGE_GEN_FILTER", _OPENROUTER_IMAGE_GEN_FILTER_MARKER),
    ("AUTO_INSTALL_VIDEO_FILTERS", _OPENROUTER_VIDEO_GEN_FILTER_MARKER),
    ("AUTO_INSTALL_IMAGE_FILTERS", _OPENROUTER_IMAGE_FILTER_MARKER),
    ("AUTO_INSTALL_FUSION_FILTER", _OPENROUTER_FUSION_FILTER_MARKER),
    ("AUTO_INSTALL_DIRECT_UPLOADS_FILTER", _DIRECT_UPLOADS_FILTER_MARKER),
)


_REPLACE_IMPORTS_REFUSAL = (
    "Open WebUI rewrites this source when it loads it and stores the result, so the pipe "
    "would rewrite it back on the next refresh: its unanchored replace of 'from " "utils', "
    "'from " "apps', 'from " "main' or 'from " "config' matched somewhere in the generated text"
)


class _FilterEnumerationUnavailable(RuntimeError):
    pass


def _is_install_enumeration_failure(exc: BaseException) -> bool:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, _FilterEnumerationUnavailable):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _newest_marked_row(rows, marker, *, prefer_id=None, tie_break_id=False, owner=None):
    candidates = [row for row in rows or [] if marker in (getattr(row, "content", "") or "")]
    if owner is not None:
        owned = [row for row in candidates if _owned_by(row, owner)]
        candidates = owned or [row for row in candidates if _claimable_by(row, owner)]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            bool(getattr(row, "is_active", False)),
            bool(prefer_id is not None and getattr(row, "id", None) == prefer_id),
            int(getattr(row, "updated_at", 0) or 0),
        ) + ((str(getattr(row, "id", "") or ""),) if tie_break_id else ()),
    )


def _byte_identical_foreign_row(rows, desired_source):
    foreign = [row for row in rows or [] if _stored_source(row) == desired_source]
    if not foreign:
        return None
    return max(
        foreign,
        key=lambda row: (
            bool(getattr(row, "is_active", False)),
            int(getattr(row, "updated_at", 0) or 0),
            str(getattr(row, "id", "") or ""),
        ),
    )


def _sweep_candidate_index(
    rows: list[Any], name: str, marker: str
) -> tuple[dict[str, list[Any]], list[Any]]:
    assign = f"{name} = "
    index: dict[str, list[Any]] = {}
    unindexed: list[Any] = []
    for row in rows or []:
        content = getattr(row, "content", None)
        if not isinstance(content, str) or not content:
            continue
        if marker not in content or "class Filter" not in content:
            continue
        keys: list[str] = []
        unparsed = False
        start = 0
        while True:
            at = content.find(assign, start)
            if at < 0:
                break
            start = at + len(assign)
            end = content.find("\n", start)
            rhs = (content[start:] if end < 0 else content[start:end]).strip()
            try:
                value = ast.literal_eval(rhs)
            except (ValueError, SyntaxError):
                unparsed = True
                break
            if not isinstance(value, str) or isinstance(value, bool):
                unparsed = True
                break
            keys.append(f"{assign}{value!r}")
        if unparsed or not keys:
            unindexed.append(row)
        for key in keys:
            index.setdefault(key, []).append(row)
    return index, unindexed


def _sweep_candidates(
    index: dict[str, list[Any]], unindexed: list[Any], *tokens: str
) -> list[Any]:
    candidates: list[Any] = []
    seen: set[int] = set()
    for row in itertools.chain(*(index.get(token, ()) for token in tokens), unindexed):
        if id(row) in seen:
            continue
        seen.add(id(row))
        candidates.append(row)
    return candidates


def _kept_candidates(candidates: list[Any], matches_candidate: Callable[[str], bool]) -> list[Any]:
    return [
        row for row in candidates if matches_candidate(getattr(row, "content", ""))
    ]


class FilterManager:
    """Manages OWUI filter functions for the OpenRouter pipe.

    Static methods handle code generation and security sanitization.
    Instance methods handle filter installation/updates in OWUI.
    """

    def _install_owner(self) -> str:
        return str(getattr(self._pipe, "id", "") or "")

    _unresolved_image_filter_ids: frozenset[str] = frozenset()
    _unresolved_video_filter_ids: frozenset[str] = frozenset()
    _unresolved_fusion_filter_id: bool = False
    _installed_image_gen_model: str | None = None

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
        self._valves = valves
        self._provider_routing_state_hash = ""
        self._provider_routing_ids_known = True
        self.logger = logger
        self._unresolved_image_filter_ids = frozenset()
        self._unresolved_video_filter_ids = frozenset()
        self._unresolved_fusion_filter_id = False
        self._installed_image_gen_model = None

    @property
    def valves(self) -> Any:
        live = getattr(self._pipe, "valves", None)
        return self._valves if live is None else live

    @property
    def unresolved_image_filter_ids(self) -> frozenset[str]:
        return self._unresolved_image_filter_ids

    @property
    def unresolved_video_filter_ids(self) -> frozenset[str]:
        return self._unresolved_video_filter_ids

    @property
    def unresolved_fusion_filter_id(self) -> bool:
        return self._unresolved_fusion_filter_id

    @property
    def installed_image_gen_model(self) -> str | None:
        return self._installed_image_gen_model


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
        prefer_id: str | None = None,
        rows: _FilterRows | None = None,
        tie_break_id: bool = False,
        candidates: list[Any] | None = None,
    ) -> tuple[str | None, _WriteOutcome]:
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
            return None, _WriteOutcome()
        except Exception:
            logging.getLogger(__name__).warning(
                "open_webui.models.functions failed to import for a reason other than absence; "
                "the features that depend on it are now disabled",
                exc_info=True,
            )
            return None, _WriteOutcome()

        if rows is None:
            try:
                filters = await Functions.get_functions_by_type("filter", active_only=False)
            except Exception as exc:
                self.logger.warning(
                    "Cannot enumerate OWUI filter functions; %s will not be installed or updated",
                    log_label,
                    exc_info=True,
                )
                raise _FilterEnumerationUnavailable(str(exc)) from exc
        elif not rows.available:
            return None, _WriteOutcome()
        else:
            filters = rows.all_rows or []

        return await self._install_from_rows(
            filters, desired_source, desired_name, desired_meta, preferred_id,
            auto_install_valve, log_label, matches_candidate, primary_marker, prefer_id,
            tie_break_id, candidates,
        )

    def _validate_before_write(self, desired_source: str, log_label: str) -> None:
        valid, error = self.validate_filter_source(desired_source)
        if not valid:
            raise ValueError(f"Generated {log_label} is invalid: {error}")

    async def _install_from_rows(
        self,
        filters: list[Any],
        desired_source: str,
        desired_name: str,
        desired_meta: dict[str, Any],
        preferred_id: str,
        auto_install_valve: str,
        log_label: str,
        matches_candidate: Callable[[str], bool],
        primary_marker: str | None,
        prefer_id: str | None = None,
        tie_break_id: bool = False,
        candidates: list[Any] | None = None,
    ) -> tuple[str | None, _WriteOutcome]:
        from open_webui.models.functions import Functions  # type: ignore

        outcome = _WriteOutcome()
        candidates = (
            candidates
            if candidates is not None
            else [f for f in filters if matches_candidate(getattr(f, "content", ""))]
        )
        chosen = None
        owner = self._install_owner()
        if candidates:
            effective_marker = ""
            if primary_marker:
                marked = [f for f in candidates if primary_marker in (getattr(f, "content", "") or "")]
                if marked:
                    candidates = marked
                    effective_marker = primary_marker
            claimed = [f for f in candidates if _claimable_by(f, owner)]
            foreign = [f for f in candidates if not _claimable_by(f, owner)]
            if effective_marker:
                chosen = (
                    _newest_marked_row(
                        claimed,
                        effective_marker,
                        prefer_id=prefer_id,
                        tie_break_id=tie_break_id,
                    )
                    if claimed
                    else _byte_identical_foreign_row(foreign, desired_source)
                )
            else:
                chosen = _newest_marked_row(
                    candidates, effective_marker, prefer_id=prefer_id, tie_break_id=tie_break_id
                )
            if len(candidates) > 1:
                self.logger.warning(
                    "Multiple %s candidates found (%d); using '%s'.",
                    log_label, len(candidates), getattr(chosen, "id", ""),
                )

        if chosen is not None and not _claimable_by(chosen, owner):
            return str(getattr(chosen, "id", "") or "").strip(), outcome

        if chosen is None:
            if not getattr(self.valves, auto_install_valve, False):
                return None, outcome

            self._validate_before_write(desired_source, log_label)
            desired_meta = {
                **desired_meta,
                _PIPE_INSTALLED_META_KEY: owner,
                _PIPE_OFF_META_KEY: True,
                _PIPE_OFF_STAMP_META_KEY: int(time.time()),
            }

            candidate_id = preferred_id
            suffix = 0
            while True:
                existing = await Functions.get_function_by_id(candidate_id)
                if existing is None:
                    break
                if matches_candidate(_stored_source(existing)) and _claimable_by(existing, owner):
                    if getattr(existing, "is_active", False):
                        return str(getattr(existing, "id", "") or ""), outcome
                    if not (_pipe_owns_the_off(existing) or not _switched_off_by_pipe(existing)):
                        chosen = existing
                        break
                    if await _write_function(
                        Functions,
                        candidate_id,
                        {
                            "is_active": True,
                            "is_global": False,
                            "meta": _merged_meta(
                                existing,
                                _manifest_id_for(desired_meta, candidate_id),
                                off_by_pipe=False,
                            ),
                        },
                        f"activating the inert {log_label} this pass found",
                        self.logger,
                    ):
                        self.logger.info("Activated inert %s: %s", log_label, candidate_id)
                        return candidate_id, outcome
                    outcome.refused = True
                    return None, outcome
                suffix += 1
                candidate_id = f"{preferred_id}_{suffix}"
                if suffix == 1:
                    self.logger.log(
                        warn_level(_warned_stale_filter_rows, f"id_taken:{preferred_id}"),
                        "The function id %r is already taken by a filter the pipe did not "
                        "install, so the pipe installs this filter under the numbered id %r "
                        "instead. The id the models carry does not change with it; remove the "
                        "row holding %r to get the canonical id back.",
                        preferred_id,
                        candidate_id,
                        preferred_id,
                    )
                if suffix > 50:
                    self._write_not_installed = True
                    self.logger.warning(
                        "Open WebUI would not give the %s an id: %r and its numbered variants "
                        "are all held by rows the pipe did not install, so nothing was "
                        "installed and the filter is left exactly as it is.",
                        log_label,
                        preferred_id,
                    )
                    return None, outcome

            if chosen is None:
                try:
                    from open_webui.models.functions import (  # type: ignore
                        FunctionForm,
                        FunctionMeta,
                    )
                except ImportError:
                    return None, outcome
                except Exception:
                    logging.getLogger(__name__).warning(
                        "open_webui.models.functions failed to import for a reason other than absence; "
                        "the features that depend on it are now disabled",
                        exc_info=True,
                    )
                    return None, outcome

                desired_meta = _manifest_id_for(desired_meta, candidate_id)
                meta_obj = FunctionMeta(**desired_meta)
                form = FunctionForm(
                    id=candidate_id,
                    name=desired_name,
                    content=desired_source,
                    meta=meta_obj,
                )
                created = await Functions.insert_new_function("", "filter", form)
                if not created:
                    created = await Functions.get_function_by_id(candidate_id)
                if not created:
                    outcome.refused = True
                    return None, outcome
                if not await _write_function(
                    Functions,
                    candidate_id,
                    {"is_active": True, "is_global": False, "name": desired_name, "meta": _merged_meta(created, desired_meta, off_by_pipe=False)},
                    f"activating the newly installed {log_label}",
                    self.logger,
                ):
                    self.logger.log(
                        warn_level(_warned_stale_filter_rows, f"not_activated:{candidate_id}"),
                        "OpenRouter %s filter %r was not activated; treating it as not installed",
                        log_label,
                        candidate_id,
                    )
                    removed = await Functions.delete_function_by_id(candidate_id)
                    if not removed:
                        self.logger.warning(
                            "Open WebUI refused to remove the inert %s %r this pass created; it stays "
                            "switched off, and the pipe switches it back on at the next pass "
                            "that can write.",
                            log_label, candidate_id,
                        )
                    outcome.refused = True
                    return None, outcome
                self.logger.info("Installed %s: %s", log_label, candidate_id)
                return candidate_id, outcome

        function_id = str(getattr(chosen, "id", "") or "").strip()
        if not function_id:
            raise _FilterEnumerationUnavailable(
                f"the chosen {log_label} row carries no id, so no row can be maintained"
            )

        existing_content = _stored_source(chosen)
        if getattr(self.valves, auto_install_valve, False):
            desired_meta = _manifest_id_for(
                {**desired_meta, _PIPE_INSTALLED_META_KEY: self._install_owner()}, function_id
            )
            switch_on = _switch_on(chosen)
            if not switch_on:
                self.logger.log(
                    warn_level(_warned_stale_filter_rows, f"admin_off:{function_id}"),
                    "%s %r is switched off in Open WebUI; the pipe keeps its code up to date "
                    "and leaves it off. Switch it on in Workspace > Functions to get it "
                    "back.%s",
                    log_label,
                    function_id,
                    _unconfirmed_switch_off_clause(chosen),
                )
            if existing_content != desired_source:
                self._validate_before_write(desired_source, log_label)
                live = await Functions.get_function_by_id(function_id)
                row = live if live is not None else chosen
                if await _write_function(
                    Functions,
                    function_id,
                    {
                        "content": desired_source,
                        "name": desired_name,
                        "meta": _maintained_meta(row, desired_meta),
                        "type": "filter",
                        "is_active": _switch_on(row),
                        "is_global": False,
                    },
                    f"updating the stored source of the installed {log_label}",
                    self.logger,
                ):
                    self.logger.info("Updating %s: %s", log_label, function_id)
            else:
                needs_write = _row_needs_update(chosen, desired_name, desired_meta)
                if needs_write:
                    live = await Functions.get_function_by_id(function_id)
                    row = live if live is not None else chosen
                    await _write_function(
                        Functions,
                        function_id,
                        {
                            "name": desired_name,
                            "meta": _maintained_meta(row, desired_meta),
                            "type": "filter",
                            "is_active": _switch_on(row),
                            "is_global": False,
                        },
                        f"refreshing the stored settings of the installed {log_label}",
                        self.logger,
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

        return function_id, outcome


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
            '            description="Cap cumulative server-tool loop cost per request in USD. 0 means no cap. The cap is sent only while an openrouter: server tool is on the request. One request is one call the pipe makes, and an internal Fusion turn is one call per panel member plus the judge and the synthesis, each sent the whole cap, so the ceiling for the turn is that multiple.",\n'
            '        )'
        )

        # -- UserValves fields ------------------------------------------------
        user_valves_fields: list[str] = []
        if enable_web_search:
            user_valves_fields.extend([
                ('        WEB_SEARCH: bool = Field(\n'
                '            default=True,\n'
                '            description="Enable OpenRouter web search for you.",\n'
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
                '            description="Enable OpenRouter web fetch (URL reading) for you.",\n'
                '        )'
            )
        if enable_datetime:
            user_valves_fields.extend([
                ('        DATETIME: bool = Field(\n'
                '            default=True,\n'
                '            description="Enable OpenRouter datetime tool for you (free, no extra cost).",\n'
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
                '            description="Enable the OpenRouter subagent tool (delegate tasks to a worker model an admin chooses).",\n'
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
        template += 'from typing import Annotated, Any, Literal\n'
        template += '\n'
        template += 'from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator\n'
        template += '\n'
        template += _ADAPTER_CACHE + '\n'
        template += '\n'
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
        template += _KEEP_WHAT_STILL_FITS + '\n'
        template += '\n'
        template += '\n'.join(valves_fields) + '\n'
        template += '\n'

        template += '    class UserValves(BaseModel):\n'
        template += _KEEP_WHAT_STILL_FITS + '\n'
        template += '\n'
        if user_valves_fields:
            template += '\n'.join(user_valves_fields) + '\n'
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
        template += '        if not isinstance(user_valves, BaseModel) or user_valves.__class__.__module__ != self.UserValves.__module__:\n'
        template += '            user_valves = self.UserValves()\n'
        template += '        elif not isinstance(user_valves, self.UserValves):\n'
        template += '            try:\n'
        template += '                user_valves = self.UserValves.model_validate(user_valves.model_dump(exclude_unset=True))\n'
        template += '            except Exception:  # noqa: BLE001 - values that fail to validate fall back to defaults\n'
        template += '                user_valves = self.UserValves()\n'
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
            if (
                row is None
                or not getattr(row, "is_active", False)
                or not _is_web_tools_filter(getattr(row, "content", None))
            ):
                row = None
                picked = _newest_marked_row(
                    await Functions.get_functions_by_type("filter", active_only=False),
                    _OPENROUTER_WEB_TOOLS_FILTER_MARKER,
                    prefer_id=_OPENROUTER_WEB_TOOLS_FILTER_PREFERRED_FUNCTION_ID,
                )
                if (
                    picked is not None
                    and getattr(picked, "is_active", False)
                    and _is_web_tools_filter(getattr(picked, "content", None))
                ):
                    row = picked
                    resolved = str(getattr(picked, "id", "") or "")
                    if resolved and resolved != function_id:
                        self.logger.log(
                            warn_level(
                                _warned_stale_filter_rows,
                                f"web_tools_id_taken:{resolved}",
                                cooldown_s=3600,
                            ),
                            "The OpenRouter Web Tools function id %r is held by a row this "
                            "pipe does not own; the per-user configuration the panel "
                            "members run against is read from %r instead.",
                            function_id,
                            resolved,
                        )
                        function_id = resolved
        except Exception as exc:
            self.logger.debug("Web tools filter lookup failed: %s", exc, exc_info=True)
            return None
        if row is None:
            return None
        content = getattr(row, "content", None)
        if not isinstance(content, str):
            return None
        cache_key = str(hash(content))
        module_ns = self._inner_web_tools_module_cache.get(cache_key)
        if module_ns is None:
            import sys
            import types

            module_name = f"function_{function_id}"
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
        rows: _FilterRows | None = None,
    ) -> str | None:
        """Ensure the OpenRouter Web Tools filter exists (and is up to date), returning its OWUI function id."""

        function_id, _outcome = await self._ensure_filter_installed(
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
            matches_candidate=_is_web_tools_filter,
            primary_marker=_OPENROUTER_WEB_TOOLS_FILTER_MARKER,
            prefer_id=_OPENROUTER_WEB_TOOLS_FILTER_PREFERRED_FUNCTION_ID,
            rows=rows,
        )
        return function_id

    async def repair_web_tools_filters(self, rows: _FilterRows | None = None) -> bool:
        switched_off = {toggle for switch, toggle, _ in WEB_TOOL_SWITCHES if not getattr(self.valves, switch)}
        if not switched_off:
            return True
        try:
            from open_webui.models.functions import Functions  # type: ignore
        except ImportError:
            return True
        except Exception as exc:
            self.logger.warning(
                "open_webui.models.functions failed to import for a reason other than absence; "
                "the Web Tools filters cannot be repaired",
                exc_info=True,
            )
            raise _FilterEnumerationUnavailable(str(exc)) from exc

        try:
            if rows is not None and rows.all_rows is not None:
                found = rows.all_rows
            else:
                found = await Functions.get_functions_by_type("filter", active_only=False)
        except Exception as exc:
            self.logger.warning("Could not list the installed Web Tools filters", exc_info=True)
            raise _FilterEnumerationUnavailable(str(exc)) from exc
        owner = self._install_owner()
        rows_owned = [
            row for row in found
            if _is_web_tools_filter(getattr(row, "content", ""))
            and _claimable_by(row, owner)
        ]

        complete = True
        if every_web_tool_is_off(self.valves):
            for row in rows_owned:
                if getattr(row, "is_active", False):
                    row_id = getattr(row, "id", "")
                    raised: list[Exception] = []
                    landed = await _write_function(
                        Functions,
                        str(row_id or ""),
                        {"is_active": False, "meta": switched_off_meta(row)},
                        "every OpenRouter Web Tools tool being disabled",
                        self.logger,
                        raised,
                    )
                    if raised:
                        self.logger.log(
                            warn_level(_warned_stale_filter_rows, f"web_tools_deactivate_raised:{row_id}"),
                            "OpenRouter Web Tools filter %r could not be switched off with every web tool "
                            "disabled, so it is still on: switch it off by hand, or let the repair try again.%s",
                            row_id,
                            _no_search_suffix(switched_off),
                            exc_info=raised[0],
                        )
                        complete = False
                        continue
                    if not landed:
                        self.logger.log(
                            warn_level(_warned_stale_filter_rows, f"web_tools_deactivate_refused:{row_id}"),
                            "OpenRouter Web refused the write that switches OpenRouter Web Tools filter %r "
                            "off while every web tool is disabled, so it is still on and still offering %s: switch "
                            "it off by hand, or let the repair try again.%s",
                            row_id,
                            ", ".join(sorted(switched_off)),
                            _no_search_suffix(switched_off),
                        )
                        complete = False
                        continue
                    self.logger.info("Disabled OpenRouter Web Tools filter %r (all tools disabled)", row.id)
            return complete

        for row in rows_owned:
            offered = _offered_web_tools(getattr(row, "content", ""))
            row_id = getattr(row, "id", "")
            if offered is None:
                self.logger.log(
                    warn_level(_warned_stale_filter_rows, f"web_tools_unreadable:{row_id}:{','.join(sorted(switched_off))}"),
                    "OpenRouter Web Tools filter %r still offers %s, which this pipe has switched off, but its "
                    "code could not be read, so it is left exactly as it is: update it or remove it.%s",
                    row_id,
                    ", ".join(sorted(switched_off)),
                    _no_search_suffix(switched_off),
                )
                complete = False
                continue
            dropped = offered & switched_off
            if not dropped:
                continue
            source = self.render_openrouter_web_tools_filter_source(
                **{kwarg: (toggle in offered and toggle not in dropped) for _, toggle, kwarg in WEB_TOOL_SWITCHES}
            ).strip() + "\n"
            raised = []
            landed = await _write_function(
                Functions,
                row_id,
                {"content": source},
                "taking a switched-off web tool out of the installed filter",
                self.logger,
                raised,
            )
            if raised:
                self.logger.log(
                    warn_level(_warned_stale_filter_rows, f"web_tools_rewrite_raised:{row_id}"),
                    "OpenRouter Web Tools filter %r still offers %s, which this pipe has switched off, and its "
                    "code could not be replaced, so it is left exactly as it is: update it or remove it.%s",
                    row_id,
                    ", ".join(sorted(dropped)),
                    _no_search_suffix(switched_off),
                    exc_info=raised[0],
                )
                complete = False
                continue
            if not landed:
                self.logger.log(
                    warn_level(_warned_stale_filter_rows, f"web_tools_rewrite_refused:{row_id}"),
                    "OpenRouter Web Tools filter %r still offers %s, which this pipe has switched off. Its code "
                    "could not be replaced, so it is left exactly as it is: update it or remove it.%s",
                    row_id,
                    ", ".join(sorted(dropped)),
                    _no_search_suffix(switched_off),
                )
                complete = False
                continue
            self.logger.warning(
                "OpenRouter Web Tools filter %r still offered %s, which this pipe has switched off. Its code was "
                "replaced with the pipe's current version for the tools it still offers (%s); its name, settings "
                "and on/off state are kept, and any hand edit in its code is gone. While AUTO_INSTALL_WEB_TOOLS_FILTER "
                "is off, switching the tool back on does not add it back; with it on, the next model-list "
                "refresh rewrites the row from the current valve set.",
                row_id,
                ", ".join(sorted(dropped)),
                ", ".join(sorted(offered - dropped)) or "none",
            )
        return complete

    async def deactivate_video_gen_filters(self) -> None:
        if self.valves.ENABLE_VIDEO_GENERATION:
            return
        try:
            from open_webui.models.functions import Functions  # type: ignore
        except ImportError:
            return
        except Exception:
            self.logger.warning(
                "open_webui.models.functions failed to import for a reason other than absence; "
                "the Video Generation filters cannot be deactivated",
                exc_info=True,
            )
            return
        try:
            found = await Functions.get_functions_by_type("filter", active_only=True)
        except Exception:
            self.logger.warning("Could not list the installed Video Generation filters", exc_info=True)
            return
        for row in found or []:
            if not _is_video_gen_filter(getattr(row, "content", "")):
                continue
            if not _claimable_by(row, self._install_owner()):
                continue
            if not getattr(row, "is_active", False):
                continue
            if await _write_function(
                Functions,
                str(getattr(row, "id", "") or ""),
                {"is_active": False, "meta": switched_off_meta(row)},
                "disabling a Video Generation filter ENABLE_VIDEO_GENERATION switched off",
                self.logger,
            ):
                self.logger.info("Disabled OpenRouter Video Generation filter %r (ENABLE_VIDEO_GENERATION=False)", row.id)

    async def reactivate_video_gen_filters(self, rows: _FilterRows | None = None) -> None:
        if not self.valves.ENABLE_VIDEO_GENERATION:
            return
        await self.reactivate_filters_by_marker(
            _OPENROUTER_VIDEO_GEN_FILTER_MARKER, log_label="Video Generation", rows=rows
        )

    async def reactivate_filters_by_marker(
        self, marker: str, *, log_label: str, rows: _FilterRows | None = None
    ) -> None:
        owner = self._install_owner()
        try:
            from open_webui.models.functions import Functions  # type: ignore
        except ImportError:
            return
        except Exception:
            self.logger.warning(
                "open_webui.models.functions failed to import for a reason other than absence; "
                f"the {log_label} filters cannot be reactivated",
                exc_info=True,
            )
            return
        try:
            if rows is not None and rows.all_rows is not None:
                found = rows.all_rows
            else:
                found = await Functions.get_functions_by_type("filter", active_only=False)
        except Exception as exc:
            self.logger.warning(
                "Could not list the installed %s filters", log_label, exc_info=True
            )
            raise _FilterEnumerationUnavailable(str(exc)) from exc
        retired_valves = {
            family_marker: valve
            for valve, family_marker in _AUTO_INSTALL_FAMILY_MARKERS
            if not getattr(self.valves, valve, False)
        }
        for row in found or []:
            if not _is_filter_carrying(getattr(row, "content", ""), marker):
                continue
            if getattr(row, "is_active", False):
                continue
            if not _pipe_owns_the_off(row):
                continue
            if not _claimable_by(row, owner):
                continue
            if marker in retired_valves and owner and _installed_by(row) == owner:
                continue
            if await _write_function(
                Functions,
                str(getattr(row, "id", "") or ""),
                {"is_active": True, "meta": _merged_meta(row, {}, off_by_pipe=False)},
                f"re-enabling a {log_label} filter the pipe had switched off",
                self.logger,
            ):
                self.logger.info("Re-enabled OpenRouter %s filter %r", log_label, row.id)

    # OPENROUTER FUSION FILTER

    async def ensure_openrouter_fusion_filter_function_id(
        self, rows: _FilterRows | None = None
    ) -> tuple[str | None, bool]:
        """Ensure the OpenRouter Fusion filter exists (and is up to date), returning its OWUI function id."""
        from .fusion_filter_renderer import (
            FUSION_FILTER_DISPLAY_NAME,
            render_openrouter_fusion_filter_source,
        )

        def _matches(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            return _OPENROUTER_FUSION_FILTER_MARKER in content and "class Filter" in content

        function_id, outcome = await self._ensure_filter_installed(
            desired_source=render_openrouter_fusion_filter_source(
                marker=_OPENROUTER_FUSION_FILTER_MARKER,
                pipe_id=self._install_owner(),
            ).strip() + "\n",
            desired_name=FUSION_FILTER_DISPLAY_NAME,
            desired_meta={
                "description": (
                    "Configure OpenRouter Fusion (multi-model judge panel): panel models, judge, "
                    "preset, and optional force-run. Acts on the fusion models."
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
            rows=rows,
        )
        unresolved = not function_id and bool(
            outcome.refused
            or getattr(self.valves, "AUTO_ATTACH_FUSION_FILTER", False)
        )
        self._unresolved_fusion_filter_id = unresolved
        return function_id, unresolved

    # OPENROUTER IMAGE GENERATION FILTER

    @staticmethod
    def render_openrouter_image_gen_filter_source(
        *,
        model_id: str = "",
        image_model: dict[str, Any] | None = None,
        endpoint_record: list[dict[str, Any]] | dict[str, Any] | None = None,
        dedicated_image_api: bool,
    ) -> str:
        """Return the canonical OWUI filter source for the OpenRouter Image Generation filter."""
        from .image_filter_renderer import (
            build_image_model_filter_spec,
            render_image_gen_filter_source,
        )

        resolved = (model_id or "").strip() or _OPENROUTER_IMAGE_GEN_FILTER_DEFAULT_MODEL
        return render_image_gen_filter_source(
            build_image_model_filter_spec(
                resolved,
                image_model,
                endpoint_record,
                dedicated_image_api=dedicated_image_api,
            ),
            catalog_match=isinstance(image_model, dict),
            selected_model=resolved,
        )

    async def image_gen_filter_inputs(
        self, rows: _FilterRows | None = None
    ) -> tuple[str | None, dict[str, Any] | None, list[dict[str, Any]] | None, bool]:
        from ..models.registry import OpenRouterModelRegistry, uses_dedicated_image_api

        selected = await self.image_gen_filter_selected_model(rows)
        if selected is None:
            return None, None, None, False
        model_id = selected.strip() or _OPENROUTER_IMAGE_GEN_FILTER_DEFAULT_MODEL
        spec = OpenRouterModelRegistry.spec(model_id)
        if not isinstance(spec, dict) or not spec:
            return model_id, None, None, False
        image_model = spec.get("image_model")
        if not isinstance(image_model, dict):
            image_model = {"id": model_id, "name": spec.get("name") or model_id}
        return (
            model_id,
            image_model,
            OpenRouterModelRegistry.image_endpoint(model_id),
            uses_dedicated_image_api(spec),
        )

    @timed
    async def ensure_openrouter_image_gen_filter_function_id(
        self, rows: _FilterRows | None = None
    ) -> str | None:
        """Ensure the OpenRouter Image Generation filter exists (and is up to date), returning its OWUI function id."""

        self._installed_image_gen_model = None

        def _matches(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            return _OPENROUTER_IMAGE_GEN_FILTER_MARKER in content and "class Filter" in content

        from .image_filter_renderer import (
            build_image_model_filter_spec,
            image_gen_model_note,
            render_image_gen_filter_source,
        )

        model_id, image_model, endpoint_record, dedicated_image_api = (
            await self.image_gen_filter_inputs(rows)
        )
        if model_id is None:
            return None
        self._installed_image_gen_model = model_id
        spec = build_image_model_filter_spec(
            model_id, image_model, endpoint_record, dedicated_image_api=dedicated_image_api
        )
        catalog_match = isinstance(image_model, dict)
        desired_source = render_image_gen_filter_source(
            spec, catalog_match=catalog_match, selected_model=model_id
        ).strip() + "\n"
        valid, error = self.validate_filter_source(desired_source)
        if not valid:
            raise ValueError(f"Generated OpenRouter Image Generation filter is invalid: {error}")

        function_id, _outcome = await self._ensure_filter_installed(
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
            tie_break_id=True,
            rows=rows,
        )
        return function_id

    async def image_gen_filter_selected_model(self, rows: _FilterRows | None = None) -> str | None:
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
            if rows is not None and rows.all_rows is not None:
                candidates = rows.all_rows
            else:
                candidates = await Functions.get_functions_by_type("filter", active_only=False)
            chosen = _newest_marked_row(
                candidates, _OPENROUTER_IMAGE_GEN_FILTER_MARKER, tie_break_id=True
            )
            if chosen is None:
                return ""
            stored = await Functions.get_function_valves_by_id(
                str(getattr(chosen, "id", "") or "")
            )
        except Exception as exc:
            self.logger.log(
                warn_level(_warned_stale_filter_rows, f"valve_read:{type(exc).__name__}"),
                "Could not read the image generation filter's selected model, so the "
                "installed filter is left as it is rather than rebuilt for the default "
                "model: %s",
                exc,
                exc_info=True,
            )
            return None

        if stored is None:
            self.logger.log(
                warn_level(_warned_stale_filter_rows, "valve_read:none"),
                "Could not read the image generation filter's selected model, so the "
                "installed filter is left as it is rather than rebuilt for the default "
                "model",
            )
            return None

        selected = (stored or {}).get("IMAGE_GENERATION_MODEL")
        return selected.strip() if isinstance(selected, str) else ""

    def render_openrouter_video_gen_filter_source(
        self,
        *,
        model_id: str = "openrouter/video",
        video_model: dict[str, Any] | None = None,
        variant_ids: Any = None,
        spec: Any = None,
    ) -> str:
        from .video_filter_renderer import render_video_filter_source

        return render_video_filter_source(
            model_id=model_id,
            video_model=video_model,
            pipe_metadata_key=_PIPE_METADATA_KEY,
            admin_valves=self.valves,
            variant_ids=variant_ids,
            spec=spec,
        )

    @timed
    async def ensure_openrouter_video_gen_filter_function_ids(
        self,
        models: list[dict[str, Any]],
        rows: _FilterRows | None = None,
    ) -> tuple[dict[str, str], frozenset[str]]:
        from ..models.registry import ModelFamily, OpenRouterModelRegistry

        if rows is None:
            prefetched = await self._filter_rows()
            rows = _FilterRows(prefetched, None, prefetched is not None)
        if not rows.available or rows.all_rows is None:
            return {}, frozenset()
        index, unindexed = _sweep_candidate_index(
            rows.all_rows, "VIDEO_MODEL_ID", _OPENROUTER_VIDEO_GEN_FILTER_MARKER
        )

        from ..models.registry import sanitize_model_id

        variant_ids_by_canonical: dict[str, set[str]] = {}
        for model in models:
            model_id = model.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            model_id = model_id.strip()
            original_id = model.get("original_id")
            canonical_id = (
                original_id if isinstance(original_id, str) and original_id.strip() else model_id
            )
            if sanitize_model_id(model_id) != sanitize_model_id(canonical_id):
                variant_ids_by_canonical.setdefault(canonical_id, set()).add(model_id)

        installed: dict[str, str] = {}
        unresolved: set[str] = set()
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
            video_spec_id = _clean_str(video_model.get("id")) or _clean_str(canonical_id)
            try:
                function_id, refused = await self._install_single_video_gen_filter(
                    model_id=canonical_id,
                    video_model=video_model,
                    rows=rows,
                    variant_ids=tuple(
                        sorted(variant_ids_by_canonical.get(canonical_id, ()))
                    ),
                    candidates=_sweep_candidates(
                        index, unindexed, f"VIDEO_MODEL_ID = {video_spec_id!r}"
                    ),
                )
            except Exception as exc:
                self.logger.log(
                    bounded_warn_level(
                        _warned_video_filter_installs,
                        f"{canonical_id}:{type(exc).__name__}",
                        _PER_MODEL_INSTALL_WARN_WINDOW,
                    ),
                    "Video filter install failed for %r: %s", canonical_id, exc, exc_info=True,
                )
                unresolved.add(model_id)
                if isinstance(original_id, str) and original_id.strip() and original_id != model_id:
                    unresolved.add(original_id)
                continue

            if function_id:
                installed[model_id] = function_id
                if isinstance(original_id, str) and original_id.strip():
                    installed[original_id.strip()] = function_id
            elif refused:
                unresolved.add(model_id)
                if isinstance(original_id, str) and original_id.strip() and original_id != model_id:
                    unresolved.add(original_id)

        self._unresolved_video_filter_ids = frozenset(unresolved)
        return installed, frozenset(unresolved)

    async def _ensure_single_video_gen_filter_function_id(
        self,
        *,
        model_id: str,
        video_model: dict[str, Any] | None,
        rows: _FilterRows | None = None,
        candidates: list[Any] | None = None,
        variant_ids: tuple[str, ...] = (),
    ) -> str | None:
        function_id, _refused = await self._install_single_video_gen_filter(
            model_id=model_id,
            video_model=video_model,
            rows=rows,
            candidates=candidates,
            variant_ids=variant_ids,
        )
        return function_id

    async def _install_single_video_gen_filter(
        self,
        *,
        model_id: str,
        video_model: dict[str, Any] | None,
        rows: _FilterRows | None = None,
        candidates: list[Any] | None = None,
        variant_ids: tuple[str, ...] = (),
    ) -> tuple[str | None, bool]:
        from .video_filter_renderer import build_video_filter_spec

        spec = build_video_filter_spec(
            model_id, video_model, admin_valves=self.valves, variant_ids=variant_ids
        )
        if not spec.contract_read:
            self.logger.info(
                "Catalogue entry for %s publishes no video contract, so no OpenRouter Video "
                "Generation filter is installed or refreshed for it", spec.model_id,
            )
            return None, False

        model_id_token = f"VIDEO_MODEL_ID = {spec.model_id!r}"

        def _matches(content: str) -> bool:
            if not _is_video_gen_filter(content):
                return False
            return model_id_token in (content or "")

        if candidates is not None:
            candidates = _kept_candidates(candidates, _matches)

        desired_source = self.render_openrouter_video_gen_filter_source(
            model_id=model_id,
            video_model=video_model,
            variant_ids=variant_ids,
            spec=spec,
        ).strip() + "\n"
        function_id, _outcome = await self._ensure_filter_installed(
            desired_source=desired_source,
            desired_name=f" {spec.display_name}"[:_DISPLAY_NAME_MAX_CHARS],
            desired_meta={
                "description": (
                    f"Configure OpenRouter async video generation for {spec.display_name}."
                ),
                "toggle": True,
                "manifest": {
                    "title": spec.display_name[:_DISPLAY_NAME_MAX_CHARS],
                    "id": spec.function_id,
                    "version": "0.1.0",
                    "license": "MIT",
                },
            },
            preferred_id=spec.function_id,
            auto_install_valve="AUTO_INSTALL_VIDEO_FILTERS",
            log_label=f"OpenRouter Video Generation filter for {spec.model_id}",
            matches_candidate=_matches,
            rows=rows,
            candidates=candidates,
        )
        return function_id, _outcome.refused


    @staticmethod
    def render_openrouter_image_filter_source(
        *,
        model_id: str,
        image_model: dict[str, Any] | None = None,
        endpoint_record: list[dict[str, Any]] | dict[str, Any] | None = None,
        dedicated_image_api: bool,
        variant_ids: tuple[str, ...] = (),
        spec: Any = None,
    ) -> str:
        from .image_filter_renderer import (
            build_image_model_filter_spec,
            render_image_model_filter_source,
        )

        if spec is not None:
            return render_image_model_filter_source(spec)
        return render_image_model_filter_source(
            build_image_model_filter_spec(
                model_id,
                image_model,
                endpoint_record,
                dedicated_image_api=dedicated_image_api,
                variant_ids=variant_ids,
            )
        )

    async def retire_families_whose_install_valve_is_off(
        self,
        *,
        web_tools_still_offered: bool = False,
        rows: _FilterRows | None = None,
    ) -> None:
        owner = self._install_owner()
        if not owner:
            return
        families = [
            (valve, marker)
            for valve, marker in _AUTO_INSTALL_FAMILY_MARKERS
            if not getattr(self.valves, valve, False)
            and not (valve == "AUTO_INSTALL_WEB_TOOLS_FILTER" and web_tools_still_offered)
        ]
        if not families:
            return
        try:
            from open_webui.models.functions import Functions  # type: ignore
        except ImportError:
            return
        except Exception:
            self.logger.warning(
                "Cannot enumerate OWUI filter functions; a family whose install valve is "
                "off cannot be retired",
                exc_info=True,
            )
            return
        if rows is None:
            prefetched = await self._filter_rows()
            rows = _FilterRows(prefetched, None, prefetched is not None)
        table = rows.all_rows
        if not table:
            return
        live_installs: set[str] | None = None
        for valve, marker in families:
            for row in table:
                content = getattr(row, "content", None)
                if not isinstance(content, str) or marker not in content:
                    continue
                if not getattr(row, "is_active", False):
                    continue
                function_id = str(getattr(row, "id", "") or "")
                if not _owned_by(row, owner):
                    stamped = _installed_by(row)
                    past_install = False
                    if stamped:
                        if live_installs is None:
                            live_installs = await self._live_pipe_installs(Functions)
                        past_install = live_installs is not None and stamped not in live_installs
                    if not past_install:
                        if function_id:
                            self.logger.log(
                                warn_level(
                                    _warned_stale_filter_rows, f"foreign_stamp:{function_id}"
                                ),
                                "Left the %s filter %r active: its install record names %r, "
                                "which Open WebUI still loads as a pipe.",
                                valve,
                                function_id,
                                stamped,
                            )
                        continue
                    self.logger.info(
                        "Retired the %s filter %r: its install record names %r, which Open "
                        "WebUI no longer loads as a pipe, and %s is off. Delete the row to "
                        "remove it; the pipe will not write to it again.",
                        valve,
                        function_id,
                        stamped,
                        valve,
                    )
                if not function_id:
                    continue
                if await _write_function(
                    Functions,
                    function_id,
                    {"is_active": False, "meta": switched_off_meta(row)},
                    f"retiring a {valve} filter whose install valve is off",
                    self.logger,
                ):
                    self.logger.info("Switched off %s filter %r (%s is off)", valve, function_id, valve)

    async def _live_pipe_installs(self, Functions: Any) -> set[str] | None:
        try:
            live = await Functions.get_functions_by_type("pipe", active_only=True)
        except Exception:
            self.logger.warning(
                "Cannot read which pipe functions Open WebUI has loaded, so a filter row "
                "stamped with another install's id is left as it is rather than retired",
                exc_info=True,
            )
            return None
        return {str(getattr(row, "id", "") or "") for row in live or ()}

    async def _read_filter_rows(self, *, active_only: bool = False) -> list[Any] | None:
        return await self._filter_rows(active_only=active_only)

    async def _filter_rows(self, *, active_only: bool = False) -> list[Any] | None:
        try:
            from open_webui.models.functions import Functions  # type: ignore
        except ImportError:
            return None
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "open_webui.models.functions failed to import for a reason other than absence; "
                "the features that depend on it are now disabled",
                exc_info=True,
            )
            raise _FilterEnumerationUnavailable(str(exc)) from exc
        try:
            return list(
                await Functions.get_functions_by_type("filter", active_only=active_only)
            )
        except Exception as exc:
            self.logger.warning(
                "Cannot enumerate OWUI filter functions; the filters that depend on it "
                "will not be installed or updated",
                exc_info=True,
            )
            raise _FilterEnumerationUnavailable(str(exc)) from exc

    @timed
    async def ensure_openrouter_image_filter_function_ids(
        self,
        models: list[dict[str, Any]],
        rows: _FilterRows | None = None,
    ) -> tuple[dict[str, list[str]], frozenset[str]]:
        """Install one filter per image model and return its attachment list.

        Each filter offers exactly the knobs that model's published contract names. The
        seven fixed variants this replaces assigned knobs by a regex on the model id, so a
        model was handed the same ten aspect ratios whatever it actually accepted.
        """
        from ..models.registry import (
            ModelFamily,
            OpenRouterModelRegistry,
            sanitize_model_id,
            uses_dedicated_image_api,
        )

        if rows is None:
            prefetched = await self._filter_rows()
            rows = _FilterRows(prefetched, None, prefetched is not None)
        if not rows.available or rows.all_rows is None:
            return {}, frozenset()

        unresolved: set[str] = set()
        installed: dict[str, list[str]] = {}
        index, unindexed = _sweep_candidate_index(
            rows.all_rows, "IMAGE_FILTER_MODEL_ID", _OPENROUTER_IMAGE_FILTER_MARKER
        )

        variant_ids_by_canonical: dict[str, set[str]] = {}
        for model in models:
            model_id = model.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            model_id = model_id.strip()
            original_id = model.get("original_id")
            canonical_id = (
                original_id if isinstance(original_id, str) and original_id.strip() else model_id
            )
            if sanitize_model_id(model_id) != sanitize_model_id(canonical_id):
                variant_ids_by_canonical.setdefault(canonical_id, set()).add(model_id)

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
            image_model = {**image_model, "id": sanitize_model_id(canonical_id)}
            image_spec_id = str(image_model.get("id") or canonical_id or "").strip()

            try:
                function_id, outcome = await self._ensure_single_image_filter_function_id(
                    model_id=canonical_id,
                    image_model=image_model,
                    endpoint_record=endpoint_record,
                    dedicated_image_api=uses_dedicated_image_api(spec),
                    rows=rows,
                    variant_ids=tuple(sorted(variant_ids_by_canonical.get(canonical_id, ()))),
                    candidates=_sweep_candidates(
                        index,
                        unindexed,
                        f"IMAGE_FILTER_MODEL_ID = {image_spec_id!r}",
                        f"IMAGE_FILTER_MODEL_ID = {canonical_id!r}",
                    ),
                )
            except Exception as exc:
                # One model's install failure costs that model its filter and nothing
                # else. The catch is deliberately broad: the install path reaches Open
                # WebUI's database, whose driver errors are not in any tuple this module
                # could enumerate, and one of them must not skip every remaining model.
                if _is_install_enumeration_failure(exc):
                    raise
                self.logger.log(
                    bounded_warn_level(
                        _warned_image_filter_installs,
                        f"{canonical_id}:{type(exc).__name__}",
                        _PER_MODEL_INSTALL_WARN_WINDOW,
                    ),
                    "Image filter install failed for %r: %s", canonical_id, exc, exc_info=True,
                )
                unresolved.add(model_id)
                if isinstance(original_id, str) and original_id.strip() and original_id != model_id:
                    unresolved.add(original_id)
                continue

            if function_id:
                installed[model_id] = [function_id]
                if isinstance(original_id, str) and original_id.strip() and original_id != model_id:
                    installed[original_id] = [function_id]
            elif outcome.refused:
                unresolved.add(model_id)
                if isinstance(original_id, str) and original_id.strip() and original_id != model_id:
                    unresolved.add(original_id)

        self._unresolved_image_filter_ids = frozenset(unresolved)
        await self._retire_variant_image_filters(rows)
        return installed, frozenset(unresolved)

    async def _retire_variant_image_filters(self, rows: _FilterRows | None = None) -> set[str]:
        """Deactivate image filters left over from the fixed-variant design.

        Those rows carry the image marker but no ``IMAGE_FILTER_MODEL_ID``, so nothing
        re-selects or overwrites them. The generic one has no model gate at all, so left
        active and attached it keeps writing its invented ratio list into every request
        for the model -- and it stays attached precisely to the models that now get no
        filter of their own.
        """
        retired: set[str] = set()
        try:
            from open_webui.models.functions import Functions

            if rows is None:
                active = await Functions.get_functions_by_type("filter", active_only=True)
            elif rows.active_rows is not None:
                active = rows.active_rows
            elif rows.all_rows is not None:
                active = [r for r in rows.all_rows if getattr(r, "is_active", False)]
            else:
                active = await Functions.get_functions_by_type("filter", active_only=True)
        except Exception as exc:
            self.logger.debug("Could not list filters to retire old ones: %s", exc, exc_info=True)
            return retired

        panels = _index_active_image_panels(active)
        for row in active or []:
            content = getattr(row, "content", "")
            row_id = getattr(row, "id", "")
            if not isinstance(content, str) or not row_id:
                continue
            if _OPENROUTER_IMAGE_FILTER_MARKER not in content:
                continue
            if "IMAGE_FILTER_MODEL_ID" in content:
                if not _row_is_off_identity(content, row_id):
                    continue
                if not _on_identity_active_sibling_exists(active, row_id, content, panels):
                    continue
            if getattr(row, "is_active", True) is False:
                continue
            if not _claimable_by(row, self._install_owner()):
                continue
            if not await _write_function(
                Functions,
                row_id,
                {"is_active": False, "meta": switched_off_meta(row)},
                "retiring a superseded image filter",
                self.logger,
            ):
                continue
            retired.add(row_id)
            self.logger.info(
                "Retired superseded image filter %r; each model now has its own.", row_id
            )
        return retired

    async def _retire_variant_video_filters(self, rows: _FilterRows | None = None) -> None:
        try:
            from open_webui.models.functions import Functions

            if rows is None or rows.active_rows is None:
                active = await Functions.get_functions_by_type("filter", active_only=True)
            else:
                active = rows.active_rows
        except Exception as exc:
            self.logger.debug("Could not list filters to retire old video ones: %s", exc, exc_info=True)
            return

        for row in active or []:
            content = getattr(row, "content", "")
            row_id = getattr(row, "id", "")
            if not _is_pipe_video_filter_row(content, row_id):
                continue
            if not _claimable_by(row, self._install_owner()):
                continue
            if getattr(row, "is_active", True) is False:
                continue
            if not await _write_function(
                Functions,
                row_id,
                {"is_active": False, "meta": switched_off_meta(row)},
                "retiring a superseded per-model video filter",
                self.logger,
            ):
                continue
            self.logger.info(
                "Deactivated per-model video filter %r; video generation is off.", row_id
            )

    async def _ensure_single_image_filter_function_id(
        self,
        *,
        model_id: str,
        image_model: dict[str, Any] | None,
        endpoint_record: list[dict[str, Any]] | dict[str, Any] | None,
        dedicated_image_api: bool,
        rows: _FilterRows | None = None,
        variant_ids: tuple[str, ...] = (),
        candidates: list[Any] | None = None,
    ) -> tuple[str | None, _WriteOutcome]:
        from .image_filter_renderer import build_image_model_filter_spec

        spec = build_image_model_filter_spec(
            model_id,
            image_model,
            endpoint_record,
            dedicated_image_api=dedicated_image_api,
            variant_ids=variant_ids,
        )
        if not spec.contract_read:
            self.logger.info(
                "No published contract for %s was in hand this pass, so its settings panel is "
                "left exactly as it is rather than rebuilt from nothing",
                spec.model_id,
            )
            outcome = _WriteOutcome()
            outcome.refused = True
            return None, outcome

        # Built with the same expression the renderer emits, not a reconstruction of it.
        # Rebuilding the literal by hand missed any id whose repr needs an escape -- an
        # invisible soft hyphen was enough -- and a filter that cannot be re-identified is
        # installed again under a new suffix on every catalog refresh.
        model_id_token = f"IMAGE_FILTER_MODEL_ID = {spec.model_id!r}"
        legacy_id_token = f"IMAGE_FILTER_MODEL_ID = {model_id!r}"

        def _matches(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            if _OPENROUTER_IMAGE_FILTER_MARKER not in content:
                return False
            if model_id_token not in content and legacy_id_token not in content:
                return False
            return "class Filter" in content

        if candidates is not None:
            candidates = _kept_candidates(candidates, _matches)

        desired_source = self.render_openrouter_image_filter_source(
            model_id=model_id,
            image_model=image_model,
            endpoint_record=endpoint_record,
            dedicated_image_api=dedicated_image_api,
            variant_ids=variant_ids,
            spec=spec,
        ).strip() + "\n"
        function_id, _outcome = await self._ensure_filter_installed(
            desired_source=desired_source,
            desired_name=spec.display_name[:_DISPLAY_NAME_MAX_CHARS],
            desired_meta={
                "description": (
                    f"Configure OpenRouter native image generation for {spec.display_name}."
                ),
                "toggle": True,
                "manifest": {
                    "title": spec.display_name[:_DISPLAY_NAME_MAX_CHARS],
                    "id": spec.function_id,
                    "version": "0.1.0",
                    "license": "MIT",
                },
            },
            preferred_id=spec.function_id,
            auto_install_valve="AUTO_INSTALL_IMAGE_FILTERS",
            log_label=f"OpenRouter image filter for {spec.model_id}",
            matches_candidate=_matches,
            prefer_id=spec.function_id,
            rows=rows,
            candidates=candidates,
        )
        return function_id, _outcome

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
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, model_validator

__ADAPTER_CACHE__


try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # noqa: BLE001 - open_webui.env does filesystem work on import
    SRC_LOG_LEVELS = {}

OWUI_OPENROUTER_PIPE_MARKER = "__MARKER__"

_UNMAPPABLE_AUDIO_FORMATS = frozenset({"webm"})


class DirectUploadError(Exception):
    """Rejects a direct upload; Open WebUI shows the message to the user verbatim."""


class Filter:
    # Toggleable filter (shows a switch in the Integrations menu).
    toggle = True

    class Valves(BaseModel):
__KEEP_WHAT_STILL_FITS__

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
            description="Comma-separated MIME allowlist for diverted direct generic files. The pattern is matched with `fnmatch` against the declared type, so a wildcard admits declared values that are not media types at all; Non-allowlisted types are fail-open: the item stays on the normal OWUI RAG/Knowledge path instead.",
        )
        DIRECT_AUDIO_MIME_ALLOWLIST: str = Field(
            default="audio/*",
            description="Comma-separated MIME allowlist for diverted direct audio files.",
        )
        DIRECT_VIDEO_MIME_ALLOWLIST: str = Field(
            default="video/mp4,video/mpeg,video/quicktime,video/webm",
            description="Comma-separated MIME allowlist for diverted direct video files. The pattern is matched with `fnmatch` against the declared type, so a wildcard admits declared values that are not media types at all; Non-allowlisted types are fail-open: the item stays on the normal OWUI RAG/Knowledge path instead.",
        )
        DIRECT_AUDIO_FORMAT_ALLOWLIST: str = Field(
            default="wav,mp3,aiff,aac,ogg,flac,m4a,pcm16,pcm24",
            description="Comma-separated audio format allowlist (derived from filename/MIME). Listing a format here lets a direct audio upload through even when it is outside the nine the pipe sends natively; naming an undocumented one here diverts the clip onto the pipe's path, and the pipe then leaves it out of the request rather than relabelling it `mp3` -- the clip comes back with a note on the chat's status line saying an audio clip was in a format the pipe will not rename. The pipe reads this valve from the installed filter's stored row, not from the copy this filter writes into each turn's metadata, so narrowing it here takes effect on the next turn with no reinstall. Only the formats listed here are diverted; a `webm` container is never diverted, listed or not, and stays on Open WebUI's path, because OpenRouter documents no `webm` format on either endpoint; a cleared value diverts no audio at all.",
        )
        DIRECT_RESPONSES_AUDIO_FORMAT_ALLOWLIST: str = Field(
            default="wav,mp3",
            description="Comma-separated audio formats eligible for /responses input_audio.format. It selects only among the nine formats OpenRouter documents; naming an undocumented one such as webm here is not a route back to sending it.",
        )

    class UserValves(BaseModel):
__KEEP_WHAT_STILL_FITS__

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
        if not isinstance(user_valves, self.UserValves):
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
                size_bytes = -1

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
                if size_bytes < 0:
                    raise DirectUploadError("Direct uploads: uploaded file missing a valid size.")
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
                if audio_format in _UNMAPPABLE_AUDIO_FORMATS:
                    warnings.append("Direct audio 'webm' is not sent directly: OpenRouter documents no 'webm' format, so the file stays on Open WebUI.")
                    retained.append(item)
                    continue
                if not audio_format or audio_format not in audio_formats_allowed:
                    retained.append(item)
                    continue
                if size_bytes < 0:
                    raise DirectUploadError("Direct uploads: uploaded file missing a valid size.")
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
                if size_bytes < 0:
                    raise DirectUploadError("Direct uploads: uploaded file missing a valid size.")
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
                attachments["audio_format_allowlist"] = self.valves.DIRECT_AUDIO_FORMAT_ALLOWLIST
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
            .replace("__ADAPTER_CACHE__", _ADAPTER_CACHE)
            .replace("__KEEP_WHAT_STILL_FITS__", _KEEP_WHAT_STILL_FITS)
        )

    @timed
    async def ensure_direct_uploads_filter_function_id(
        self, rows: _FilterRows | None = None
    ) -> str | None:
        """Ensure the OpenRouter Direct Uploads companion filter exists (and is up to date), returning its OWUI function id."""

        def _matches(content: str) -> bool:
            if not isinstance(content, str) or not content:
                return False
            return _DIRECT_UPLOADS_FILTER_MARKER in content and "class Filter" in content

        function_id, _outcome = await self._ensure_filter_installed(
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
            rows=rows,
        )
        return function_id

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
        from open_webui_openrouter_pipe import __version__

        data = f"{admin_sorted}|{user_sorted}|{provider_data}|{transports}|{__version__}"
        return hashlib.md5(data.encode()).hexdigest()

    def _routing_drift(
        self,
        all_models: set[str],
        existing_filters: dict[str, Any],
        undeliverable_slugs: set[str],
        provider_map: dict[str, dict[str, list[str]]],
        model_visibility: dict[str, str],
        pipe_identifier: str,
    ) -> tuple[set[str], set[str], set[str], dict[str, str]]:
        content_drifted: set[str] = set()
        deliverable_off: set[str] = set()
        global_rows: set[str] = set()
        desired_sources: dict[str, str] = {}
        for slug in all_models:
            existing = existing_filters.get(slug)
            if existing is None or slug in undeliverable_slugs:
                continue
            if _row_owner(existing) not in ("", pipe_identifier):
                continue
            if getattr(existing, "is_global", False):
                global_rows.add(slug)
            if not getattr(existing, "is_active", False):
                deliverable_off.add(slug)
            model_info = provider_map.get(slug) or {}
            transport = self.model_transport(slug)
            if not self._routing_controls(transport) or not model_info.get("providers"):
                continue
            short_name, prov_names = _routing_display_names(model_info)
            desired_source = self._render_provider_routing_filter_source(
                slug,
                list(model_info.get("providers") or []),
                list(model_info.get("quantizations") or []),
                model_visibility.get(slug, "user"),
                short_name=short_name,
                provider_names=prov_names,
                transport=transport,
                owner=pipe_identifier,
            ).strip() + "\n"
            desired_sources[slug] = desired_source
            if desired_source != _stored_source(existing):
                content_drifted.add(slug)
        return content_drifted, deliverable_off, global_rows, desired_sources

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
        if not isinstance(user_valves, self.UserValves):
            user_valves = None
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
            """Get a boolean: None when this holder never set the field, else the value it holds (an explicit False is returned)."""
            for holder, holder_set in __BOOL_HOLDERS__:
                if holder is not None and field in holder_set:
                    return bool(getattr(holder, field, api_default))
            return None

        def get_float(field: str) -> float | None:
'''
        _bool_holder_terms = []
        if visibility in ("user", "both"):
            _bool_holder_terms.append("(user_valves, user_set)")
        if visibility in ("admin", "both"):
            _bool_holder_terms.append('(getattr(self, "valves", None), admin_set)')
        logic = logic.replace(
            "__BOOL_HOLDERS__", "[" + ", ".join(_bool_holder_terms) + "]"
        )
        if visibility in ("user", "both"):
            logic += '''            if field in user_set and user_valves is not None:
                val = getattr(user_valves, field, None)
                if isinstance(val, (int, float)):
                    return float(val)
'''
        if visibility in ("admin", "both"):
            logic += '''            if field in admin_set and hasattr(self, "valves"):
                val = getattr(self.valves, field, None)
                if isinstance(val, (int, float)):
                    return float(val)
'''
        logic += '''            return None

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
        if min_throughput is not None and min_throughput > 0:
            provider["preferred_min_throughput"] = min_throughput

'''
        if "MAX_LATENCY" in drawn:
            logic += '''        max_latency = get_float("MAX_LATENCY")
        if max_latency is not None and max_latency > 0:
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
            (max_price_prompt is not None and max_price_prompt > 0)
            or (max_price_completion is not None and max_price_completion > 0)
            or (max_price_image is not None and max_price_image > 0)
            or (max_price_audio is not None and max_price_audio > 0)
            or (max_price_request is not None and max_price_request > 0)
        ):
            provider["max_price"] = {}
            if max_price_prompt is not None and max_price_prompt > 0:
                provider["max_price"]["prompt"] = max_price_prompt
            if max_price_completion is not None and max_price_completion > 0:
                provider["max_price"]["completion"] = max_price_completion
            if max_price_image is not None and max_price_image > 0:
                provider["max_price"]["image"] = max_price_image
            if max_price_audio is not None and max_price_audio > 0:
                provider["max_price"]["audio"] = max_price_audio
            if max_price_request is not None and max_price_request > 0:
                provider["max_price"]["request"] = max_price_request

'''
        logic += '''        # Inject into metadata if we have any provider settings
        if provider:
            if __metadata__ is None:
                __metadata__ = {}
            pipe_meta = __metadata__.setdefault("__PIPE_META_KEY__", {})
            existing = pipe_meta.get("provider")
            merged = dict(existing) if isinstance(existing, dict) else {}
            merged.update(provider)
            pipe_meta["provider"] = merged
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
        owner: str = "",
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

        from open_webui_openrouter_pipe import __version__

        if not isinstance(model_slug, str) or not model_slug:
            raise ValueError("model_slug must be a non-empty string")
        safe_model_slug_escaped = json.dumps(model_slug)[1:-1]

        display_name = short_name.strip() if short_name else model_slug.split("/")[-1]
        safe_display_name = FilterManager.validate_provider_name(display_name, slug=model_slug)
        safe_display_name_escaped = json.dumps(safe_display_name)[1:-1]

        marker = f"{_PROVIDER_ROUTING_FILTER_MARKER_PREFIX}{model_slug}:{_PROVIDER_ROUTING_FILTER_MARKER_VERSION}"
        safe_marker_escaped = json.dumps(marker)[1:-1]
        owner_assignment = f"{_PROVIDER_ROUTING_OWNER_PREFIX} = {json.dumps(str(owner))}" if owner else ""

        safe_providers = [
            p for p in providers
            if isinstance(p, str) and _PROVIDER_SLUG_PATTERN.match(p) and len(p) <= 64
        ][:_PROVIDER_ROUTING_MAX_PROVIDERS]

        prov_names = provider_names or {}
        provider_display_options: list[str] = []
        provider_slug_map_entries: list[str] = []
        display_to_slug: dict[str, str] = {}
        seen_labels: set[str] = set()
        for pslug in safe_providers:
            disp = prov_names.get(pslug, pslug.replace("-", " ").title())
            safe_disp = FilterManager.validate_provider_name(disp, slug=pslug)
            if safe_disp in seen_labels:
                safe_disp = f"{safe_disp} ({pslug})"
            seen_labels.add(safe_disp)
            provider_display_options.append(safe_disp)
            display_to_slug[safe_disp] = pslug
            provider_slug_map_entries.append(f'    {FilterManager.safe_literal_string(safe_disp)}: {FilterManager.safe_literal_string(pslug)}')

        no_pref = "(no preference)"
        only_ignore_options = [no_pref] + provider_display_options
        only_ignore_literal = ", ".join(FilterManager.safe_literal_string(opt) for opt in only_ignore_options)

        # Build provider map code block
        provider_map_code = "{\n" + ",\n".join(provider_slug_map_entries) + "\n}" if provider_slug_map_entries else "{}"

        order_display_options: list[str] = []
        order_map_entries: list[str] = []

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
            "MIN_THROUGHPUT": '        MIN_THROUGHPUT: float = Field(default=0, ge=0, description="Min throughput (tokens/sec); 0 clears the admin default, omit for no constraint")',
            "MAX_LATENCY": '        MAX_LATENCY: float = Field(default=0, ge=0, description="Max latency (seconds); 0 clears the admin default, omit for no constraint")',
            "MAX_PRICE_PROMPT": '        MAX_PRICE_PROMPT: float = Field(default=0, ge=0, description="Max price for prompt ($/M tokens); 0 clears the admin default, omit for no limit")',
            "MAX_PRICE_COMPLETION": '        MAX_PRICE_COMPLETION: float = Field(default=0, ge=0, description="Max price for completion ($/M tokens); 0 clears the admin default, omit for no limit")',
            "MAX_PRICE_IMAGE": '        MAX_PRICE_IMAGE: float = Field(default=0, ge=0, description="Max price per image ($/image); 0 clears the admin default, omit for no limit")',
            "MAX_PRICE_AUDIO": '        MAX_PRICE_AUDIO: float = Field(default=0, ge=0, description="Max price for audio ($/unit); 0 clears the admin default, omit for no limit")',
            "MAX_PRICE_REQUEST": '        MAX_PRICE_REQUEST: float = Field(default=0, ge=0, description="Max price per request ($/request); 0 clears the admin default, omit for no limit")',
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
            if value in options:
                return value
            kept = _NO_PREF
            if info.field_name == "ORDER" and isinstance(value, str):
                head = value.split(" > ")[0].strip()
                if head and f"{{head}} first" in options:
                    kept = f"{{head}} first"
            _warn_stale_choice(info.field_name, value, kept)
            return kept
''' if guarded else ""
        dropped = sorted(
            name for name in drawn
            if name not in guarded and name != "DATA_COLLECTION"
        )
        dropped_literal = ", ".join(json.dumps(name) for name in dropped)
        stale_value_guard = f'''
        @field_validator({dropped_literal}, mode="before")
        @classmethod
        def _drop_unusable_setting(cls, value: Any, info: ValidationInfo) -> Any:
            field = cls.model_fields[info.field_name]
            table = _adapters_for(cls)
            adapter = table.get(info.field_name)
            if adapter is None:
                metadata = field.metadata
                annotated = (
                    Annotated[(field.annotation, *metadata)]
                    if metadata
                    else field.annotation
                )
                adapter = table[info.field_name] = TypeAdapter(annotated)
            try:
                adapter.validate_python(value)
            except ValidationError:
                _warn_unusable_setting(info.field_name, value, field.get_default())
                return field.get_default(call_default_factory=True)
            return value
''' if dropped else ""
        priority_guard = '''
        @field_validator("priority", mode="before")
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
                _warn_unusable_setting(info.field_name, value, field.get_default())
                return field.get_default(call_default_factory=True)
            return value
'''

        admin_controls = (
            f"{rendered_controls}\n{stale_choice_guard}{stale_value_guard}"
            if visibility in ("admin", "both")
            else ""
        )
        valves_class = f'''
    class Valves(BaseModel):
        """Admin-level provider routing preferences."""
{_PRIORITY_FIELD}
{admin_controls}{priority_guard}'''

        user_valves_class = ""
        if visibility in ("user", "both"):
            user_valves_class = f'''
    class UserValves(BaseModel):
        """User-level provider routing preferences (can override admin defaults)."""
{rendered_controls}
{stale_choice_guard}{stale_value_guard}'''

        # Generate init based on visibility
        init_body = "        self.log = logging.getLogger(f\"openrouter.provider.{MODEL_SLUG}\")\n        self.log.setLevel(SRC_LOG_LEVELS.get(\"OPENAI\", logging.INFO))"
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
from typing import Annotated, Any, Literal, get_args

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, ValidationInfo, field_validator

{_ADAPTER_CACHE}

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # noqa: BLE001 - open_webui.env does filesystem work on import
    SRC_LOG_LEVELS = {{}}

OWUI_OPENROUTER_PIPE_MARKER = "{safe_marker_escaped}"
{owner_assignment}
MODEL_SLUG = "{safe_model_slug_escaped}"
OPENROUTER_PIPE_VERSION = {__version__!r}

# Sentinel value for "no preference" dropdown option
_NO_PREF = "(no preference)"

# Map display names to provider slugs
_PROVIDER_MAP: dict[str, str] = {provider_map_code}

# Map ORDER display values to provider slug lists
_ORDER_MAP: dict[str, list[str]] = {order_map_code}

_WARNED_STALE_CHOICES: set[tuple[str, str, str]] = set()
_WARNED_UNUSABLE_SETTINGS: set[tuple[str, str, str]] = set()


def _warn_stale_choice(field: str, value: Any, kept: str) -> None:
    marker = (field, str(value), kept)
    if marker in _WARNED_STALE_CHOICES:
        return
    _WARNED_STALE_CHOICES.add(marker)
    logging.getLogger(MODEL_SLUG).warning(
        "Provider routing valve %s: %r is no longer offered; using %r",
        field,
        value,
        kept,
    )


def _warn_unusable_setting(field: str, value: Any, kept: Any) -> None:
    marker = (field, str(value), str(kept))
    if marker in _WARNED_UNUSABLE_SETTINGS:
        return
    _WARNED_UNUSABLE_SETTINGS.add(marker)
    logging.getLogger(MODEL_SLUG).warning(
        "Provider routing valve %s: stored value %r is not usable by this filter "
        "build; using the field default %r",
        field,
        value,
        kept,
    )


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
        rows: _FilterRows | None = None,
        *,
        not_fetched_slugs: Iterable[str] = (),
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
        not_fetched = {s.strip() for s in not_fetched_slugs if s and s.strip()}

        current_hash = self.compute_provider_routing_hash(admin_models_csv, user_models_csv, provider_map)
        hash_unchanged = current_hash == self._provider_routing_state_hash
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

        self._provider_routing_ids_known = True
        try:
            if rows is not None and rows.all_rows is not None:
                all_filters = rows.all_rows
            else:
                all_filters = await Functions.get_functions_by_type("filter", active_only=False)
        except Exception:
            self._provider_routing_ids_known = False
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

        undeliverable_slugs = {
            slug
            for slug in all_models
            if slug not in not_fetched
            and (
                not self._routing_controls(self.model_transport(slug))
                or not (provider_map.get(slug) or {}).get("providers")
            )
        }
        stale_active = {
            slug
            for slug in undeliverable_slugs
            if getattr(existing_filters.get(slug), "is_active", False)
            and not _operator_re_enabled_the_pipe_off(existing_filters.get(slug))
        }

        orphaned_active = {
            slug for slug, existing in existing_filters.items()
            if slug not in all_models and getattr(existing, "is_active", False)
        }

        content_drifted: set[str] = set()
        deliverable_off: set[str] = set()
        global_rows: set[str] = set()
        drift_sources: dict[str, str] = {}
        if hash_unchanged:
            content_drifted, deliverable_off, global_rows, drift_sources = self._routing_drift(
                all_models,
                existing_filters,
                undeliverable_slugs,
                provider_map,
                model_visibility,
                pipe_identifier,
            )
            for slug in sorted(deliverable_off):
                existing_id = getattr(existing_filters.get(slug), "id", "")
                self.logger.log(
                    warn_level(_warned_stale_filter_rows, f"admin_off:{existing_id}"),
                    "Provider routing filter %r is switched off in Open WebUI; the pipe "
                    "keeps its code up to date and leaves it off. Switch it on in "
                    "Workspace > Functions to get it back.%s",
                    existing_id,
                    _unconfirmed_switch_off_clause(existing_filters.get(slug)),
                )

        if (
            hash_unchanged and not content_drifted
            and not missing_filters and not stale_active and not orphaned_active
            and not global_rows
            and not deliverable_off
        ):
            for slug in all_models:
                if slug in undeliverable_slugs:
                    continue
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
        writes_ok = True
        undeliverable: list[str] = []
        for slug, visibility in model_visibility.items():
            model_info = provider_map.get(slug, {})
            providers = model_info.get("providers", [])
            quantizations = model_info.get("quantizations", [])
            short_name, prov_names = _routing_display_names(model_info)

            if not providers:
                if slug in not_fetched:
                    skipped_id = getattr(existing_filters.get(slug), "id", "")
                    if skipped_id and _row_owner(existing_filters.get(slug)) in ("", pipe_identifier):
                        slug_to_filter_id[slug] = skipped_id
                    self.logger.debug(
                        "Provider routing slug %s was not fetched this cycle (endpoint cap); "
                        "its existing filter row is left exactly as it is.",
                        slug,
                    )
                    continue
                undeliverable.append(slug)
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

            desired_source = drift_sources.get(slug) or self._render_provider_routing_filter_source(
                slug, providers, quantizations, visibility,
                short_name=short_name,
                provider_names=prov_names,
                transport=transport,
                owner=pipe_identifier,
            ).strip() + "\n"

            def _validate_or_skip(slug: str, source: str) -> bool:
                is_valid, validation_error = self.validate_filter_source(source)
                if is_valid:
                    return True
                self.logger.error(
                    "Generated filter for %s failed syntax validation: %s. Skipping.",
                    slug,
                    validation_error,
                )
                return False

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
                if _row_owner(existing) not in ("", pipe_identifier):
                    existing_id = getattr(existing, "id", "")
                    if existing_id:
                        slug_to_filter_id[slug] = existing_id
                    continue
                switch_on = _switch_on(existing)
                if not switch_on:
                    self.logger.log(
                        warn_level(_warned_stale_filter_rows, f"admin_off:{existing_id}"),
                        "Provider routing filter %r is switched off in Open WebUI; the pipe "
                        "keeps its code up to date and leaves it off. Switch it on in "
                        "Workspace > Functions to get it back.%s",
                        existing_id,
                        _unconfirmed_switch_off_clause(existing),
                    )
                if existing_content != desired_source:
                    if not _validate_or_skip(slug, desired_source):
                        continue
                    live = await Functions.get_function_by_id(existing_id)
                    row = live if live is not None else existing
                    if await _write_function(
                        Functions,
                        existing_id,
                        {
                            "content": desired_source,
                            "name": desired_name,
                            "meta": _maintained_meta(row, desired_meta),
                            "is_active": _switch_on(row),
                            "is_global": False,
                        },
                        f"updating the stored source of the provider routing filter for {slug}",
                        self.logger,
                    ):
                        updated += 1
                        self.logger.info("Updated provider routing filter: %s", existing_id)
                    else:
                        writes_ok = False
                elif _row_needs_update(existing, desired_name, desired_meta):
                    live = await Functions.get_function_by_id(existing_id)
                    row = live if live is not None else existing
                    if not await _write_function(
                        Functions,
                        existing_id,
                        {
                            "is_active": _switch_on(row),
                            "meta": _maintained_meta(row, desired_meta),
                            "name": desired_name,
                            "type": "filter",
                            "is_global": False,
                        },
                        f"re-asserting the provider routing filter for {slug}",
                        self.logger,
                    ):
                        writes_ok = False
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
                    own_marker = f"{_PROVIDER_ROUTING_FILTER_MARKER_PREFIX}{slug}:"
                    if own_marker in _stored_source(existing_func):
                        break
                    suffix += 1
                    candidate_id = f"{filter_id}_{suffix}"
                    if suffix > 50:
                        self.logger.warning("Could not find unique ID for provider routing filter: %s", slug)
                        break

                if suffix <= 50:
                    if not _validate_or_skip(slug, desired_source):
                        continue
                    desired_meta = {
                        **desired_meta,
                        _PIPE_OFF_META_KEY: True,
                        _PIPE_OFF_STAMP_META_KEY: int(time.time()),
                    }
                    meta_obj = FunctionMeta(**desired_meta)
                    form = FunctionForm(
                        id=candidate_id,
                        name=desired_name,
                        content=desired_source,
                        meta=meta_obj,
                    )
                    created_func = await Functions.insert_new_function("", "filter", form)
                    if not created_func:
                        created_func = await Functions.get_function_by_id(candidate_id)
                    if created_func:
                        if await _write_function(
                            Functions,
                            candidate_id,
                            {"is_active": True, "is_global": False, "meta": _merged_meta(created_func, desired_meta, off_by_pipe=False)},
                            "activating the new provider routing filter",
                            self.logger,
                        ):
                            created += 1
                            self.logger.info("Created provider routing filter: %s", candidate_id)
                            # Track for attachment
                            slug_to_filter_id[slug] = candidate_id
                        else:
                            writes_ok = False
                            self._provider_routing_ids_known = False
                            removed = await Functions.delete_function_by_id(candidate_id)
                            if not removed:
                                self.logger.warning(
                                    "Open WebUI refused to remove the inert provider routing "
                                    "filter %r; it stays switched off, and the pipe switches "
                                    "it back on at the next pass that can write.",
                                    candidate_id,
                                )
                    else:
                        writes_ok = False
                        self._provider_routing_ids_known = False
                        self.logger.warning(
                            "Open WebUI refused to create the provider routing filter for %r; "
                            "the models it covers keep the filter they already have, and the "
                            "create is retried on the next catalog refresh.",
                            slug,
                        )

        disabled = 0
        for orphan in orphan_filters:
            orphan_id = getattr(orphan, "id", "")
            if orphan_id and _row_owner(orphan) in ("", pipe_identifier):
                if _is_already_switched_off(orphan):
                    continue
                if await _write_function(
                    Functions,
                    orphan_id,
                    {"is_active": False, "meta": switched_off_meta(orphan)},
                    "disabling a duplicate provider routing filter",
                    self.logger,
                ):
                    disabled += 1
                    self.logger.warning("Disabled duplicate provider routing filter: %s", orphan_id)
                else:
                    writes_ok = False

        for slug, existing in existing_filters.items():
            if slug in undeliverable or slug not in all_models:
                existing_id = getattr(existing, "id", "")
                if existing_id and _row_owner(existing) in ("", pipe_identifier):
                    if slug in all_models and _operator_re_enabled_the_pipe_off(existing):
                        continue
                    if not getattr(existing, "is_active", False):
                        continue
                    deactivation = {
                        "is_active": False,
                        "meta": switched_off_meta(existing),
                    }
                    if await _write_function(
                        Functions,
                        existing_id,
                        deactivation,
                        "disabling a provider routing filter the routing valves no longer publish",
                        self.logger,
                    ):
                        disabled += 1
                        self.logger.info("Disabled provider routing filter: %s", existing_id)
                    else:
                        writes_ok = False

        if created or updated or disabled:
            self.logger.info(
                "Provider routing filters: created=%d, updated=%d, disabled=%d (total models=%d)",
                created, updated, disabled, len(all_models),
            )

        if writes_ok:
            self._provider_routing_state_hash = current_hash
            self.logger.debug("Provider routing state hash updated: %s", current_hash[:8])

        self.logger.info(
            "Returning %d provider routing filter mappings for attachment: %r",
            len(slug_to_filter_id),
            slug_to_filter_id,
        )
        return slug_to_filter_id
