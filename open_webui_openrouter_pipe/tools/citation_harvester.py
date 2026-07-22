"""Generic URL-citation harvesting from tool result strings.

Extracts (url, title, snippet) candidates from two strict shapes only:
whole-string JSON with exact ``url``/``link``/``href`` keys, and labeled-line
text as emitted by web tools ("Title:"/"URL:" blocks or a "# heading" + "URL:"
head). Anything else yields no candidates. All lookups are bounded and the
public function never raises.

``BUILTIN_CITATION_TOOLS`` mirrors the tool names Open WebUI's own
``get_citation_source_from_tool_result`` special-cases (Open WebUI 0.10.x);
those names keep using the Open WebUI extractor at the call site.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlparse

BUILTIN_CITATION_TOOLS = frozenset(
    {"search_web", "fetch_url", "view_file", "view_knowledge_file", "query_knowledge_files"}
)

_MAX_INPUT_CHARS = 1_000_000
_MAX_SOURCES = 15
_MAX_DEPTH = 8
_MAX_NODES = 2000
_MAX_URL_LEN = 2048
_MAX_TITLE_LEN = 300
_MAX_SNIPPET_LEN = 500
_MAX_LINES = 4000
_MAX_BLOCKS = 15
_URL_KEYS = ("url", "link", "href")
_TITLE_KEYS = ("title", "name")
_SNIPPET_KEYS = ("text", "snippet", "description", "content")
_SEARCH_BLOCK_DELIMITER = "\n\n---\n\n"
_URL_LINE_RE = re.compile(r"URL:\s*(\S+)\s*\Z")


def _sanitize_text(value: Any, limit: int) -> str:
    if not isinstance(value, str) or not value:
        return ""
    cleaned_chars: list[str] = []
    for ch in value:
        code = ord(ch)
        if 0xD800 <= code <= 0xDFFF:
            continue
        if ch not in "\n\t" and (code < 0x20 or 0x7F <= code <= 0x9F):
            continue
        cleaned_chars.append(ch)
    return "".join(cleaned_chars).strip()[:limit]


def _valid_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    candidate = value.strip()
    if not candidate or len(candidate) > _MAX_URL_LEN:
        return ""
    for ch in candidate:
        code = ord(ch)
        if code < 0x21 or 0x7F <= code <= 0x9F or 0xD800 <= code <= 0xDFFF:
            return ""
    try:
        parsed = urlparse(candidate)
    except Exception:
        return ""
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""
    return candidate


def _candidate_from_node(node: dict[str, Any]) -> tuple[str, str, str] | None:
    for key in _URL_KEYS:
        url = _valid_url(node.get(key))
        if not url:
            continue
        title = ""
        for title_key in _TITLE_KEYS:
            title = _sanitize_text(node.get(title_key), _MAX_TITLE_LEN)
            if title:
                break
        snippet = ""
        for snippet_key in _SNIPPET_KEYS:
            snippet = _sanitize_text(node.get(snippet_key), _MAX_SNIPPET_LEN)
            if snippet:
                break
        return (url, title, snippet)
    return None


def _harvest_json(parsed: Any) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    stack: list[tuple[Any, int]] = [(parsed, 0)]
    visited = 0
    while stack and len(found) < _MAX_SOURCES:
        node, depth = stack.pop()
        visited += 1
        if visited > _MAX_NODES:
            break
        children: list[Any] = []
        if isinstance(node, dict):
            candidate = _candidate_from_node(node)
            if candidate is not None:
                found.append(candidate)
            children = list(node.values())
        elif isinstance(node, list):
            children = node
        if depth >= _MAX_DEPTH:
            continue
        for child in reversed(children):
            if visited + len(stack) >= _MAX_NODES:
                break
            if isinstance(child, (dict, list)):
                stack.append((child, depth + 1))
    return found


def _harvest_fetch_shape(lines: list[str]) -> list[tuple[str, str, str]]:
    match = _URL_LINE_RE.fullmatch(lines[1].strip()) if len(lines) > 1 else None
    if match is None:
        return []
    url = _valid_url(match.group(1))
    if not url:
        return []
    title = _sanitize_text(lines[0][2:], _MAX_TITLE_LEN)
    snippet = _sanitize_text("\n".join(lines[2:40]), _MAX_SNIPPET_LEN)
    return [(url, title, snippet)]


def _harvest_search_shape(text: str) -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for block in text.split(_SEARCH_BLOCK_DELIMITER)[:_MAX_BLOCKS]:
        block_lines = block.split("\n", 60)
        if len(block_lines) < 2 or not block_lines[0].startswith("Title:"):
            continue
        match = _URL_LINE_RE.fullmatch(block_lines[1].strip())
        if match is None:
            continue
        if not any(line.startswith("Highlights:") for line in block_lines[:6]):
            continue
        url = _valid_url(match.group(1))
        if not url:
            continue
        title = _sanitize_text(block_lines[0][len("Title:"):], _MAX_TITLE_LEN)
        highlight_index = next(
            (i for i, line in enumerate(block_lines[:6]) if line.startswith("Highlights:")), None
        )
        snippet = ""
        if highlight_index is not None:
            snippet = _sanitize_text(
                "\n".join(block_lines[highlight_index + 1 : highlight_index + 20]),
                _MAX_SNIPPET_LEN,
            )
        found.append((url, title, snippet))
    return found


def _harvest_labeled(text: str) -> list[tuple[str, str, str]]:
    lines = text.split("\n", _MAX_LINES)[:_MAX_LINES]
    if not lines:
        return []
    if lines[0].startswith("# "):
        return _harvest_fetch_shape(lines)
    if lines[0].startswith("Title:"):
        return _harvest_search_shape(text)
    return []


def harvest_tool_citations(tool_result: Any) -> list[tuple[str, str, str]]:
    try:
        if not isinstance(tool_result, str) or not tool_result:
            return []
        if len(tool_result) > _MAX_INPUT_CHARS:
            return []
        parse_failed = False
        parsed: Any = None
        try:
            parsed = json.loads(tool_result)
        except Exception:
            parse_failed = True
        raw = _harvest_labeled(tool_result) if parse_failed else _harvest_json(parsed)
        deduped: list[tuple[str, str, str]] = []
        seen_urls: set[str] = set()
        for url, title, snippet in raw:
            if url in seen_urls:
                continue
            seen_urls.add(url)
            deduped.append((url, title, snippet))
            if len(deduped) >= _MAX_SOURCES:
                break
        return deduped
    except Exception:
        return []
