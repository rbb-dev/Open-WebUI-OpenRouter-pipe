from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any
from urllib.parse import urlsplit

from .utils import server_tool_status

SEARCH_SNIPPET_CHAR_CAP = 1600
SEARCH_SNIPPET_CHAR_FLOOR = 100
SEARCH_ROUND_SNIPPET_CHAR_BUDGET = 8000
SEARCH_MAX_RESULTS = 25
SEARCH_REPORTED_NO_SOURCES = "The search reported no sources."
SEARCH_DID_NOT_COMPLETE = "Search did not complete."

_OPENAI_TRACKING_SUFFIX = "?utm_source=openai"
_PASSAGE_SEPARATOR = "\n[...]\n"
_CUT_MARK = "…"


def comparable_url(url: str) -> str:
    text = url.strip().removesuffix(_OPENAI_TRACKING_SUFFIX)
    try:
        parts = urlsplit(text)
    except ValueError:
        return text.lower()
    return f"{parts.netloc.lower()}{parts.path.rstrip('/')}?{parts.query}"


def _search_action(item: dict[str, Any]) -> dict[str, Any]:
    action = item.get("action")
    return action if isinstance(action, dict) else {}


def search_query(item: dict[str, Any]) -> str:
    query = _search_action(item).get("query")
    return query.strip() if isinstance(query, str) else ""


def search_source_urls(item: dict[str, Any]) -> list[str]:
    sources = _search_action(item).get("sources")
    if not isinstance(sources, list):
        return []
    urls: list[str] = []
    for source in sources:
        link = source.get("url") if isinstance(source, dict) else source
        if isinstance(link, str) and link.strip():
            urls.append(link.strip())
    return urls


def render_search_args(item: dict[str, Any]) -> str:
    query = search_query(item)
    return json.dumps({"query": query}, ensure_ascii=False) if query else "{}"


def merge_search_citations(parsed: Iterable[tuple[str, str, str]]) -> list[dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for url, title, content in parsed:
        key = comparable_url(url)
        entry = merged.get(key)
        if entry is None:
            merged[key] = {
                "url": url.removesuffix(_OPENAI_TRACKING_SUFFIX),
                "title": "" if title == url else title,
                "content": content,
            }
            continue
        if content and content not in entry["content"]:
            entry["content"] = f"{entry['content']}{_PASSAGE_SEPARATOR}{content}" if entry["content"] else content
        if not entry["title"] and title != url:
            entry["title"] = title
    return list(merged.values())


def attribute_search_citations(
    searches: list[dict[str, Any]], citations: list[dict[str, str]]
) -> list[list[dict[str, str]]]:
    listed = [{comparable_url(url) for url in search_source_urls(item)} for item in searches]
    owned: list[list[dict[str, str]]] = [[] for _ in searches]
    for citation in citations:
        key = comparable_url(citation["url"])
        for index, urls in enumerate(listed):
            if key in urls:
                owned[index].append(citation)
    return owned


def _counted(count: int, noun: str) -> str:
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _cut_note(beyond_cap: int, excerpts_withheld: int, excerpts_shortened: int) -> str:
    not_included = []
    if beyond_cap:
        not_included.append(_counted(beyond_cap, "more result"))
    if excerpts_withheld:
        not_included.append(f"the excerpts of {_counted(excerpts_withheld, 'result')}")
    sentences = []
    if not_included:
        sentences.append(f"Not included: {'; '.join(not_included)}.")
    if excerpts_shortened:
        sentences.append(f"Shortened: {_counted(excerpts_shortened, 'excerpt')}.")
    return " ".join(sentences)


def render_search_result(item: dict[str, Any], citations: list[dict[str, str]]) -> str:
    if server_tool_status(item) != "completed":
        error = item.get("error")
        detail = error.strip() if isinstance(error, str) else ""
        return f"{SEARCH_DID_NOT_COMPLETE} {detail}" if detail else SEARCH_DID_NOT_COMPLETE
    cited_by_link = {comparable_url(citation["url"]): citation for citation in citations}
    results: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for link in search_source_urls(item):
        key = comparable_url(link)
        if key in seen:
            continue
        seen.add(key)
        cited = cited_by_link.get(key, {})
        results.append((cited.get("title", ""), link, cited.get("content", "")))
    if not results:
        return SEARCH_REPORTED_NO_SOURCES
    kept = results[:SEARCH_MAX_RESULTS]
    budget = SEARCH_ROUND_SNIPPET_CHAR_BUDGET
    withheld = 0
    shortened = 0
    entries: list[dict[str, str]] = []
    for title, link, snippet in kept:
        if snippet and budget < SEARCH_SNIPPET_CHAR_FLOOR:
            withheld += 1
            snippet = ""
        elif snippet:
            room = min(SEARCH_SNIPPET_CHAR_CAP, budget)
            if len(snippet) > room:
                snippet = snippet[: room - 1].rstrip() + _CUT_MARK
                shortened += 1
            budget -= len(snippet)
        entries.append(
            {name: value for name, value in (("title", title), ("link", link), ("snippet", snippet)) if value}
        )
    note = _cut_note(len(results) - len(kept), withheld, shortened)
    if note:
        entries.append({"note": note})
    return json.dumps(entries, ensure_ascii=False)
