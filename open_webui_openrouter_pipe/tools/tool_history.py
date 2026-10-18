from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from typing import Any

logger = logging.getLogger(__name__)

try:
    import re2
except Exception:
    logger.debug("re2 is unavailable; regex mode answers that it is not installed", exc_info=True)
    re2 = None

try:
    import regex as _regex_mod
except Exception:
    logger.debug("the regex module is unavailable; the tokenizer falls back to str", exc_info=True)
    _regex_mod = None

try:
    import open_webui.env as _viewer_env

    _VIEW_DEFAULT = int(getattr(_viewer_env, "VIEW_FILE_DEFAULT_MAX_CHARS", 0)) or 10_000
    _VIEW_CAP = int(getattr(_viewer_env, "VIEW_FILE_MAX_CHARS", 0)) or 100_000
except Exception:
    logger.debug("Open WebUI's viewer limits are unavailable; the environment fallback applies", exc_info=True)
    _VIEW_DEFAULT = 0
    _VIEW_CAP = 0

from ..core.config import _NON_REPLAYABLE_TOOL_ARTIFACTS
from ..core.utils import (
    _MARKER_SUFFIX,
    _iter_marker_spans,
    _tool_result_failed,
    is_picture_output,
    is_text_part_output,
    recorded_tool_text,
    server_tool_arguments,
    server_tool_result_text,
    server_tool_status,
    split_text_by_markers,
    strip_hidden_marker_lines,
    tool_output_text_and_pictures,
)
from ..requests.transformer import (
    _as_replayed,
    _from_pipe_storage,
    _handoff_back_answers,
    _one_copy_per_round,
    _orphaned_round_markers,
    _restored_call_tool_replay,
    _strip_markers_from_tool_content,
    compute_turn_indices,
    history_round_entry_ok,
    history_self_name,
    message_identifier,
)
from ..storage.persistence import normalize_persisted_item

HISTORY_TOOL_NAMES = ("search_tool_history", "list_tool_history", "view_tool_call")
HISTORY_PIPE_TOOL_PREFIX = "pipe:"
SEARCH_DEFAULT = 10
SEARCH_MAX = 50
LIST_PAGE = 30
VIEW_LINES = 100
PATTERN_MAX = 4096
MATCH_BUDGET_SECONDS = 2.0
SNIPPET_CHARS = 300
ARGUMENT_PREVIEW_CHARS = 200
OPENING_CHARS = 100
ARGUMENTS_INLINE_CHARS = 2000
_LISTING_SNIPPET_CHARS = 120
DATA_URL = re.compile(r"data:[a-z]+/[a-z0-9.+-]+;base64,[A-Za-z0-9+/=]{200,}")
QUOTED = re.compile(r'"([^"]+)"')
CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
CJK = re.compile(r"([぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]+)")
WORD = _regex_mod.compile(r"[\p{L}\p{N}\p{M}]+") if _regex_mod is not None else re.compile(r"[^\W_]+")
STOP = frozenset((
    "the", "a", "an", "of", "to", "and", "in", "for", "on", "with", "by", "from", "at", "as", "is", "was",
    "are", "be", "this", "that", "it", "not", "or", "what", "how", "why", "when", "did", "do", "does",
))

_HISTORY_POS_KEY = "_history_pos"
_HISTORY_ROW_LOAD_CONCURRENCY = 4


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def view_limits() -> tuple[int, int]:
    cap = _VIEW_CAP or _env_int("VIEW_FILE_MAX_CHARS", 100_000)
    default = _VIEW_DEFAULT or _env_int("VIEW_FILE_DEFAULT_MAX_CHARS", 10_000)
    cap = max(1, cap)
    return max(1, min(default, cap)), cap


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for word in WORD.findall(unicodedata.normalize("NFKC", text or "")):
        for part in CAMEL.split(word):
            for chunk in CJK.split(part.casefold()):
                if CJK.fullmatch(chunk):
                    out.extend([chunk[i : i + 2] for i in range(len(chunk) - 1)] or [chunk])
                elif chunk:
                    out.append(chunk)
    return out


def _flatten(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            lines.extend(_flatten(item, f"{prefix}{key}."))
        return lines or [f"{prefix.rstrip('.')}: {{}}"]
    if isinstance(value, list):
        lines = []
        for index, item in enumerate(value):
            lines.extend(_flatten(item, f"{prefix}{index}."))
        return lines or [f"{prefix.rstrip('.')}: []"]
    if value is None:
        shown = "null"
    elif value is True:
        shown = "true"
    elif value is False:
        shown = "false"
    else:
        shown = str(value)
    return [f"{prefix.rstrip('.')}: {shown}"]


def _hide_data_urls(text: str) -> str:
    return DATA_URL.sub(lambda match: f"[data URL, {len(match.group(0)):,} chars]", text)


def decode_value(text: Any) -> str:
    if not isinstance(text, str):
        text = json.dumps(text, ensure_ascii=False)
    stripped = text.lstrip()
    if stripped[:1] in ("{", "["):
        try:
            return _hide_data_urls("\n".join(_flatten(json.loads(stripped))))
        except (ValueError, RecursionError):
            pass
    return _hide_data_urls(text)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    parts.append(part["text"])
                elif part.get("type") in ("image_url", "input_image"):
                    parts.append("[picture]")
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


class HistoryCall:
    __slots__ = (
        "_decoded", "_values", "answered", "arguments", "arguments_raw", "call_id", "failed",
        "id", "name", "result_chars", "result_raw", "turn",
    )

    def __init__(self, turn: int, position: int, call_id: str, name: str, arguments_raw: str):
        self.id = f"{turn}.{position}"
        self.turn = turn
        self.call_id = call_id
        self.name = name
        self.arguments_raw = arguments_raw
        self.arguments = self._decode_arguments(arguments_raw)
        self.result_raw: str | None = None
        self.result_chars = 0
        self.answered = False
        self.failed = False
        self._decoded: str | None = None
        self._values: set[str] | None = None

    @staticmethod
    def _decode_arguments(raw_text: str) -> str:
        if not raw_text.strip():
            return ""
        try:
            parsed = json.loads(raw_text)
        except (ValueError, RecursionError):
            return _hide_data_urls(raw_text)
        if isinstance(parsed, dict):
            if not parsed:
                return ""
            return _hide_data_urls("\n".join(_flatten(parsed)))
        return _hide_data_urls(raw_text)

    def result_text(self) -> str | None:
        if not self.answered:
            return None
        if self._decoded is None:
            self._decoded = decode_value(self.result_raw)
        return self._decoded

    def argument_values(self) -> set[str]:
        if self._values is None:
            values: set[str] = set()
            try:
                parsed = json.loads(self.arguments_raw)
            except (ValueError, RecursionError):
                parsed = None
            if parsed is None:
                if self.arguments_raw.strip():
                    values.add(self.arguments_raw.strip().casefold())
            else:
                stack = [parsed]
                while stack:
                    item = stack.pop()
                    if isinstance(item, str):
                        stripped = item.strip().casefold()
                        if stripped:
                            values.add(stripped)
                    elif isinstance(item, dict):
                        stack.extend(item.values())
                    elif isinstance(item, list):
                        stack.extend(item)
            self._values = values
        return self._values


class HistoryHit:
    __slots__ = (
        "field", "line", "line_number", "matched", "of", "phrase_unit", "record", "search_in", "typed", "whole", "words",
    )

    def __init__(
        self,
        record: HistoryCall,
        field: str | None = None,
        line_number: int | None = None,
        line: str | None = None,
        matched: list[str] | None = None,
        of: int = 0,
        search_in: str = "all",
        whole: bool = False,
        typed: list[str] | None = None,
        words: list[str] | None = None,
        phrase_unit: str = "",
    ) -> None:
        self.phrase_unit = phrase_unit
        self.record = record
        self.field = field
        self.line_number = line_number
        self.line = line
        self.matched = matched
        self.of = of
        self.search_in = search_in
        self.whole = whole
        self.typed = typed or []
        self.words = words or []


class History:
    def __init__(
        self,
        messages: list[dict[str, Any]],
        artifacts: dict[str | None, dict[str, dict[str, Any]]],
        *,
        window_turns: int,
        notes_in_view: bool,
        ask_user_names: frozenset[str] | set[str],
        self_names: frozenset[str] | set[str],
    ) -> None:
        self.calls: list[HistoryCall] = []
        self.by_id: dict[str, HistoryCall] = {}
        self.opening: dict[int, str] = {}
        self.ask_user_names = frozenset(ask_user_names)
        self.self_names = frozenset(self_names)
        self.notes_in_view = bool(notes_in_view)
        self.results_in_view = not bool(notes_in_view)
        handoff_back_at = _handoff_back_answers(messages)

        def _carries_compaction(message: Any) -> bool:
            if not isinstance(message, dict):
                return False
            if message.get("contextSummary") or message.get("context_summary"):
                return True
            content = message.get("content")
            return (
                (message.get("role") or "").lower() in ("system", "developer")
                and isinstance(content, str)
                and content.startswith("[CONVERSATION SUMMARY]")
            )

        turn_indices, total_turns = compute_turn_indices(messages, handoff_back_at)
        self.compacted = any(_carries_compaction(message) for message in messages)
        self.current_turn = max(total_turns, 1)
        self.first_view_turn = max(1, self.current_turn - max(int(window_turns or 0), 0))
        regions: dict[Any, list[dict[str, Any]]] = {}
        positions: dict[Any, int] = defaultdict(int)

        def region_of(turn: Any) -> list[dict[str, Any]]:
            return regions.setdefault(turn, [])

        def next_position(turn: Any) -> int:
            positions[turn] += 1
            return positions[turn]

        def stamp(item: dict[str, Any], turn: Any, position: int) -> None:
            item[_HISTORY_POS_KEY] = (turn, position)

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            role = (message.get("role") or "").lower()
            turn = turn_indices[index]
            if role == "user":
                if turn is not None:
                    turn1 = turn + 1
                    if turn1 not in self.opening:
                        self.opening[turn1] = _message_text(message.get("content"))[:OPENING_CHARS]
                continue
            if role == "assistant":
                self._region_from_markers(message, turn, region_of, next_position, stamp, artifacts)
                raw_tool_calls = message.get("tool_calls")
                for tool_call in raw_tool_calls if isinstance(raw_tool_calls, list) else []:
                    if not isinstance(tool_call, dict) or not history_round_entry_ok(tool_call):
                        continue
                    function = tool_call.get("function")
                    if not isinstance(function, dict):
                        continue
                    name = str(function.get("name") or "").strip()
                    call_id = tool_call.get("id") or tool_call.get("call_id")
                    call_id = call_id.strip() if isinstance(call_id, str) else ""
                    arguments = function.get("arguments")
                    raw_text = (
                        arguments
                        if isinstance(arguments, str)
                        else json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
                    )
                    position = next_position(turn)
                    item = {
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": raw_text.strip() or "{}",
                    }
                    if not history_self_name(name, self.self_names):
                        stamp(item, turn, position)
                    region_of(turn).append(item)
                continue
            if role == "tool":
                raw_call_id = message.get("tool_call_id") or message.get("id") or message.get("call_id")
                call_id = raw_call_id.strip() if isinstance(raw_call_id, str) else ""
                if not call_id:
                    continue
                cleaned = _strip_markers_from_tool_content(message.get("content"), strip_hidden_marker_lines)
                if cleaned is None:
                    text = ""
                elif is_picture_output(cleaned) or is_text_part_output(cleaned):
                    text = tool_output_text_and_pictures(cleaned)[0]
                elif isinstance(cleaned, str):
                    text = cleaned
                else:
                    try:
                        text = json.dumps(cleaned, ensure_ascii=False)
                    except (TypeError, ValueError):
                        text = str(cleaned)
                region_of(turn).append({"type": "function_call_output", "call_id": call_id, "output": text})

        for turn, region in regions.items():
            kept = _one_copy_per_round(region)
            turn1 = (turn + 1) if isinstance(turn, int) else 0
            self._collect(kept, turn1)

    def _region_from_markers(
        self,
        message: dict[str, Any],
        turn: Any,
        region_of,
        next_position,
        stamp,
        artifacts: dict[str | None, dict[str, dict[str, Any]]],
    ) -> None:
        text = message.get("content")
        text = text if isinstance(text, str) else _message_text(text)
        if _MARKER_SUFFIX not in text:
            return
        spans = _iter_marker_spans(text)
        if not spans:
            return
        segments = split_text_by_markers(text, spans)
        markers = [seg["marker"] for seg in segments if isinstance(seg, dict) and seg.get("type") == "marker"]
        if not markers:
            return
        group_id = message.get("message_id") or message_identifier(message)
        batch = (artifacts or {}).get(group_id) or {}
        replayable: dict[str, dict[str, Any]] = {}
        for marker in markers:
            row = batch.get(marker)
            if row is None:
                continue
            normalized = normalize_persisted_item(row)
            normalized = _restored_call_tool_replay(normalized) if isinstance(normalized, dict) else normalized
            if isinstance(normalized, dict):
                replayable[marker] = normalized
        if not replayable:
            return
        orphaned_calls, orphaned_outputs = _orphaned_round_markers(segments, replayable)
        for segment in segments:
            if not isinstance(segment, dict) or segment.get("type") != "marker":
                continue
            marker = str(segment.get("marker") or "")
            item = replayable.get(marker)
            if item is None:
                continue
            item_type = str(item.get("type") or "").lower()
            if item_type == "reasoning" or item_type in _NON_REPLAYABLE_TOOL_ARTIFACTS:
                continue
            if (item_type == "function_call" and marker in orphaned_calls) or (
                item_type == "function_call_output" and marker in orphaned_outputs
            ):
                continue
            callish = item_type == "function_call" or item_type.startswith("openrouter:")
            position = next_position(turn) if callish else None
            item_name = item.get("name")
            item_name = item_name.strip() if isinstance(item_name, str) else ""
            self_named = history_self_name(item_name, self.self_names) if item_type == "function_call" else False
            for part in _as_replayed(item, fallback_id=marker):
                if not isinstance(part, dict):
                    continue
                part_type = str(part.get("type") or "")
                if callish and position is not None and not self_named and part_type == "function_call":
                    stamp(part, turn, position)
                region_of(turn).append(_from_pipe_storage(part))

    def _collect(self, kept: list[Any], turn1: int) -> None:
        pending: dict[str, list[HistoryCall]] = defaultdict(list)
        for it in kept:
            if not isinstance(it, dict):
                continue
            item_type = str(it.get("type") or "")
            if item_type == "function_call":
                turn_position = it.pop(_HISTORY_POS_KEY, None)
                if not isinstance(turn_position, tuple) or len(turn_position) != 2:
                    continue
                call_turn, position = turn_position
                call_id = it.get("call_id")
                call_id = call_id.strip() if isinstance(call_id, str) else ""
                name = it.get("name")
                name = name.strip() if isinstance(name, str) else ""
                record = HistoryCall(
                    call_turn + 1 if isinstance(call_turn, int) else 0,
                    position,
                    call_id,
                    name,
                    str(it.get("arguments") or ""),
                )
                pending[record.call_id].append(record)
                self.calls.append(record)
                self.by_id[record.id] = record
            elif item_type == "function_call_output":
                call_id = it.get("call_id")
                call_id = call_id.strip() if isinstance(call_id, str) else ""
                queue = pending.get(call_id)
                record = queue.pop(0) if queue else None
                if record is None:
                    continue
                output = it.get("output")
                text = output if isinstance(output, str) else json.dumps(output, ensure_ascii=False) if output is not None else ""
                record.answered = True
                record.result_raw = text
                record.result_chars = len(text)
                record.failed = _tool_result_failed(text, it.get("status"))
            elif item_type.startswith("openrouter:"):
                turn_position = it.pop(_HISTORY_POS_KEY, None)
                if not isinstance(turn_position, tuple) or len(turn_position) != 2:
                    continue
                call_turn, position = turn_position
                name = item_type.split(":", 1)[1] or item_type
                record = HistoryCall(
                    call_turn + 1 if isinstance(call_turn, int) else 0,
                    position,
                    str(it.get("id") or ""),
                    name,
                    json.dumps(server_tool_arguments(it), ensure_ascii=False),
                )
                record.answered = True
                record.result_raw = recorded_tool_text(server_tool_result_text(it), server_tool_status(it))
                record.result_chars = len(record.result_raw)
                record.failed = server_tool_status(it) != "completed"
                self.calls.append(record)
                self.by_id[record.id] = record
            else:
                continue

    def in_view(self, record: HistoryCall) -> bool:
        if record.name in self.ask_user_names:
            return True
        return self.results_in_view and record.turn >= self.first_view_turn

    def scope_line(self, search_in: str) -> str:
        earlier = [record for record in self.calls if not self.in_view(record)]
        if search_in == "tool_name":
            where = "tool names"
        elif search_in == "arguments":
            where = "arguments"
        elif search_in == "result":
            where = "results"
        else:
            where = "tool names, arguments and results"
        if self.notes_in_view:
            line = (
                f"This chat has {len(self.calls):,} tool {'call' if len(self.calls) == 1 else 'calls'} in turns "
                f"1-{self.current_turn}. Their results are not in this request: each earlier one stands in it as a "
                "note naming the call's id, and this tool returns them. A turn starts at a person's message that "
                f"follows the assistant's reply; this request is turn {self.current_turn}. Searched: {where}."
            )
        else:
            line = (
                f"This chat has {len(self.calls):,} tool {'call' if len(self.calls) == 1 else 'calls'} in turns "
                f"1-{self.current_turn}, {len(earlier):,} of them in turns left out of this request. A turn starts at a "
                f"person's message that follows the assistant's reply; this request is turn {self.current_turn}. "
                f"Searched: {where}."
            )
            if self.first_view_turn < self.current_turn:
                line += (
                    f" Turns {self.first_view_turn}-{self.current_turn} are already in this request and are counted, "
                    "not listed."
                )
        if self.compacted:
            line += (
                " This request carries Open WebUI's conversation summary; the turns it stands for are not in the"
                " request and are not searched here."
            )
        if self.ask_user_names and any(record.name in self.ask_user_names for record in self.calls):
            line += " The person's ask_user rounds of every turn are already in this request."
        return line

    def selected(self, tool, status, from_turn, to_turn):
        for record in self.calls:
            if tool and record.name != tool:
                continue
            if status == "failed" and not record.failed:
                continue
            if status == "succeeded" and record.failed:
                continue
            if from_turn and record.turn < from_turn:
                continue
            if to_turn and record.turn > to_turn:
                continue
            yield record

    @staticmethod
    def fields(record: HistoryCall, search_in: str) -> list[tuple[str, str]]:
        names = ("tool_name", "arguments", "result") if search_in == "all" else (search_in,)
        out = []
        for field_name in names:
            if field_name == "tool_name":
                out.append((field_name, record.name))
            elif field_name == "arguments":
                if record.arguments:
                    out.append((field_name, record.arguments))
            else:
                result_text = record.result_text()
                if result_text is not None:
                    out.append((field_name, result_text))
        return out

    @staticmethod
    def locate(text: str, position: int) -> tuple[int, str]:
        line_number = text.count("\n", 0, position) + 1
        start = text.rfind("\n", 0, position) + 1
        end = text.find("\n", position)
        return line_number, text[start : end if end >= 0 else len(text)]

    def status_word(self, record: HistoryCall) -> str:
        return "failed" if record.failed else "succeeded"

    def result_note(self, record: HistoryCall) -> str:
        if not record.answered:
            return "no result stored"
        return f"result {record.result_chars:,} chars"

    def age(self, record: HistoryCall) -> str:
        turns = self.current_turn - record.turn
        if turns <= 0:
            return "this turn"
        return "1 turn ago" if turns == 1 else f"{turns} turns ago"

    def hit_block(self, hit: HistoryHit) -> str:
        record = hit.record
        if hit.line is None and hit.search_in is not None:
            self.resolve_snippet(hit)
        view = " · in this request" if self.in_view(record) else ""
        arguments = record.arguments.replace("\n", " | ")
        if len(arguments) > ARGUMENT_PREVIEW_CHARS:
            arguments = arguments[:ARGUMENT_PREVIEW_CHARS] + "…"
        out = [
            (
                f"[{record.id}] turn {record.turn} ({self.age(record)}) · {record.name} · {self.status_word(record)} · "
                f"{self.result_note(record)}{view}"
            ),
            f'  turn {record.turn} began: "{self.opening.get(record.turn, "")}"',
            f"  arguments: {arguments if arguments else '(none)'}",
        ]
        if hit.field and hit.line is not None:
            snippet = hit.line.strip()
            if len(snippet) > SNIPPET_CHARS:
                snippet = snippet[:SNIPPET_CHARS] + "…"
            label = hit.field.replace("_", " ")
            out.append(f"  {label} line {hit.line_number}: {snippet}")
        if hit.matched is not None and hit.of:
            out.append(f"  matched {len(hit.matched)} of {hit.of} words: {', '.join(hit.matched)}")
        return "\n".join(out)

    def resolve_snippet(self, hit: HistoryHit) -> None:
        texts = [(field_name, field_text, field_text.casefold()) for field_name, field_text in self.fields(hit.record, hit.search_in)]
        choices: list[str] = []
        if hit.whole:
            choices.append(hit.phrase_unit)
        choices.extend(hit.typed)
        choices.extend(hit.words)
        for unit in choices:
            for field_name, field_text, low in texts:
                position = low.find(unit)
                if position >= 0:
                    line_number = low.count("\n", 0, position) + 1
                    lines = field_text.split("\n")
                    hit.field = field_name
                    hit.line_number = line_number
                    hit.line = lines[line_number - 1] if line_number - 1 < len(lines) else ""
                    return
        if not texts:
            hit.field, hit.line_number, hit.line = "result", 1, ""
            return
        field_name, field_text, _low = texts[0]
        hit.field, hit.line_number, hit.line = field_name, 1, field_text.split("\n", 1)[0]

    @staticmethod
    def matcher(query: str, mode: str):
        if mode == "exact" and re2 is None:
            needle = query.casefold()

            def casefold_search(text):
                position = text.casefold().find(needle)
                return (position, position + len(needle)) if position >= 0 else None

            return casefold_search, None
        if re2 is None:
            return None, "Error: regex mode is not available on this install."
        options = re2.Options()
        options.case_sensitive = False
        options.literal = mode == "exact"
        options.max_mem = 1 << 20
        options.log_errors = False
        try:
            compiled = re2.compile(query, options=options)
        except re2.error as exc:
            return None, f"Error: invalid or unsupported regex (RE2 syntax: no lookarounds or backreferences): {exc}"

        def search(text):
            match = compiled.search(text)
            return (match.start(), match.end()) if match else None

        return search, None

    @staticmethod
    def query_units(query: str):
        phrase = query.strip().casefold()
        quoted = [part.strip().casefold() for part in QUOTED.findall(query) if part.strip()]
        rest = QUOTED.sub(" ", query)
        typed = list(dict.fromkeys(quoted + [token for token in rest.casefold().split() if len(token) > 2 and token not in STOP]))
        words = [
            word for word in dict.fromkeys(tokenize(query)) if word not in STOP and (len(word) >= 2 or CJK.fullmatch(word))
        ]
        return phrase, typed, words

    def keyword_hits(self, query, selected, search_in, deadline):
        phrase, typed, words = self.query_units(query)
        units = set(typed)
        if phrase:
            units.add(phrase)
        by_argument = search_in in ("all", "arguments")
        matched = []
        for record in selected:
            if time.monotonic() > deadline:
                return self.over_budget()
            lows = [(field_name, field_text.casefold()) for field_name, field_text in self.fields(record, search_in)]
            if not lows:
                continue
            per_word = {}
            for word in words:
                for field_name, low in lows:
                    count = low.count(word)
                    if count:
                        per_word.setdefault(word, {})[field_name] = count
            whole = bool(phrase) and any(phrase in low for _field_name, low in lows)
            if not per_word and not whole:
                continue
            equal = len(units & record.argument_values()) if by_argument else 0
            intact = frozenset(token for token in typed if any(token in low for _field, low in lows))
            sizes = {field_name: len(low) for field_name, low in lows}
            matched.append((record, equal, whole, intact, per_word, sizes))
        total = max(len(selected), 1)
        frequency = Counter(word for entry in matched for word in entry[4])
        unit_frequency = Counter(unit for entry in matched for unit in entry[3])
        lengths = defaultdict(list)
        for entry in matched:
            for field_name, size in entry[5].items():
                lengths[field_name].append(size)
        average = {field_name: max(1.0, sum(values) / len(values)) for field_name, values in lengths.items()}
        ranked = []
        for record, equal, whole, intact, per_word, size in matched:
            unit_score = round(
                sum(math.log(1 + (total - unit_frequency[unit] + 0.5) / (unit_frequency[unit] + 0.5)) for unit in intact), 6
            )
            score = 0.0
            for word, per_field in per_word.items():
                idf = math.log(1 + (total - frequency[word] + 0.5) / (frequency[word] + 0.5))
                best = 0.0
                for field_name, count in per_field.items():
                    norm = 0.25 + 0.75 * size[field_name] / average[field_name]
                    best = max(best, idf * count * 2.2 / (count + 1.2 * norm))
                score += best
            key = ((equal, whole), unit_score, len(per_word), score, record.turn, record.id)
            present = [word for word in words if word in per_word]
            hit = HistoryHit(
                record, matched=present, of=len(words), search_in=search_in, whole=whole,
                typed=[token for token in typed if token in intact],
                words=present, phrase_unit=phrase,
            )
            ranked.append((key, hit))
        ranked.sort(key=lambda entry: entry[0], reverse=True)
        return [hit for _key, hit in ranked]

    @staticmethod
    def over_budget() -> str:
        return f"Error: the search took over {MATCH_BUDGET_SECONDS:g}s; narrow the query or add a tool or turn filter."

    def other_fields_note(self, query, mode, selected, search_in):
        if search_in == "all":
            return ""
        counts = {}
        phrase, words, test = "", [], None
        if mode == "keyword":
            phrase, _typed, words = self.query_units(query)
        else:
            test, error = self.matcher(query, mode)
            if error:
                return ""
        for field_name in ("tool_name", "arguments", "result"):
            if field_name == search_in:
                continue
            found = 0
            for record in selected:
                for _name, field_text in self.fields(record, field_name):
                    if mode == "keyword":
                        low = field_text.casefold()
                        if (phrase and phrase in low) or any(word in low for word in words):
                            found += 1
                    elif test and test(field_text):
                        found += 1
            if found:
                counts[field_name] = found
        if not counts:
            return ""
        return (
            "Other fields hold it: "
            + ", ".join(f"{field_name.replace('_', ' ')} of {total:,} {'call' if total == 1 else 'calls'}" for field_name, total in counts.items())
            + "; search_in=all looks at all of them."
        )

    def search(self, query, mode, search_in, tool, status, from_turn, to_turn, count, offset):
        selected = list(self.selected(tool, status, from_turn, to_turn))
        searched_calls = [record for record in selected if not self.in_view(record)]
        deadline = time.monotonic() + MATCH_BUDGET_SECONDS
        if mode in ("exact", "regex"):
            search, error = self.matcher(query, mode)
            if error or search is None:
                return error or "Error: the pattern could not be compiled."
            hits = []
            for record in selected:
                if time.monotonic() > deadline:
                    return self.over_budget()
                for field_name, field_text in self.fields(record, search_in):
                    found = search(field_text)
                    if found:
                        line_number, line = self.locate(field_text, found[0])
                        hits.append(HistoryHit(record, field_name, line_number, line))
                        break
            hits.sort(key=lambda hit: (hit.record.turn, hit.record.id), reverse=True)
        else:
            hits = self.keyword_hits(query, selected, search_in, deadline)
            if isinstance(hits, str):
                return hits
        listed = [hit for hit in hits if not self.in_view(hit.record)]
        counted = [hit for hit in hits if self.in_view(hit.record)]
        head = [self.scope_line(search_in)]
        shown_query = f"Query ({mode}): {query}"
        if tool:
            shown_query += f" · tool {tool}"
        if status and status != "any":
            shown_query += f" · {status}"
        if from_turn or to_turn:
            shown_query += f" · turns {from_turn or 1}-{to_turn or self.current_turn}"
        head.append(shown_query)
        plus = ""
        if counted:
            low_turn = min(hit.record.turn for hit in counted)
            high_turn = max(hit.record.turn for hit in counted)
            span = f"turn {low_turn}" if low_turn == high_turn else f"turns {low_turn}-{high_turn}"
            plus = f"Plus {len(counted):,} {'match' if len(counted) == 1 else 'matches'} in {span}, which this request already shows."
        if not listed:
            head.append(f"No call in the turns left out of this request matched, among the {len(searched_calls):,} searched.")
            if mode == "keyword":
                _phrase, typed, words = self.query_units(query)
                tried = words or typed
                head.append(
                    "Words tried: " + (", ".join(tried) if tried else "none (the query has no letters or digits; use mode exact for symbols)") + "."
                )
            note = self.other_fields_note(query, mode, searched_calls, search_in)
            if note:
                head.append(note)
            if plus:
                head.append(plus)
            return "\n".join(head)
        by_tool = Counter(hit.record.name for hit in listed)
        turns = len({hit.record.turn for hit in listed})
        head.append(
            f"{len(listed):,} {'call' if len(listed) == 1 else 'calls'} matched in {turns} {'turn' if turns == 1 else 'turns'}; by tool: "
            + ", ".join(f"{name} {total}" for name, total in by_tool.most_common(5))
            + "."
        )
        if plus:
            head.append(plus)
        page = listed[offset : offset + count]
        if not page:
            head.append(f"Offset {offset} is past the last of the {len(listed):,} matching calls.")
            return "\n".join(head)
        shown = f"Showing {offset + 1}-{offset + len(page)}, " + ("best first" if mode == "keyword" else "newest first")
        if offset + len(page) < len(listed):
            shown += f"; next page: offset={offset + len(page)}"
        head.append(shown + ". Read a call in full with view_tool_call.")
        return "\n".join(head) + "\n\n" + "\n".join(self.hit_block(hit) for hit in page)

    def listing(self, tool, status, from_turn, to_turn, offset):
        selected = list(self.selected(tool, status, from_turn, to_turn))
        records = [record for record in selected if not self.in_view(record)]
        in_view = [record for record in selected if self.in_view(record)]
        single_turn = bool(from_turn and to_turn and from_turn == to_turn)
        lines = []
        if tool or single_turn:
            for record in reversed(records):
                arguments = record.arguments.replace("\n", " | ")
                arguments = arguments[:_LISTING_SNIPPET_CHARS] + ("…" if len(arguments) > _LISTING_SNIPPET_CHARS else "")
                lines.append(
                    f"{record.id} {record.name} · {self.status_word(record)} · {self.result_note(record)} · "
                    f"{arguments if arguments else '(no arguments)'}"
                )
            unit = "calls"
        else:
            per_turn = defaultdict(list)
            for record in records:
                per_turn[record.turn].append(record)
            for turn in sorted(per_turn, reverse=True):
                names = Counter(record.name for record in per_turn[turn])
                failed = sum(1 for record in per_turn[turn] if record.failed)
                lines.append(
                    f"turn {turn} · {len(per_turn[turn])} calls: "
                    + ", ".join(f"{name} ×{total}" for name, total in names.most_common(4))
                    + (" …" if len(names) > 4 else "")
                    + (f" · {failed} failed" if failed else "")
                    + f' · began: "{self.opening.get(turn, "")[:70]}"'
                )
            unit = "turns"
        head = [
            f"This chat has {len(self.calls):,} tool calls in turns 1-{self.current_turn}; this request is turn {self.current_turn}."
        ]
        if not self.notes_in_view and self.first_view_turn <= self.current_turn and (
            in_view or self.first_view_turn < self.current_turn
        ):
            head.append(
                (
                    f"Turns {self.first_view_turn}-{self.current_turn} are already in this request and are not listed"
                    if self.first_view_turn < self.current_turn
                    else f"Turn {self.current_turn} is already in this request and is not listed"
                )
                + (f" ({len(in_view):,} matching {'call' if len(in_view) == 1 else 'calls'} there)." if in_view else ".")
            )
        names = Counter(record.name for record in self.calls)
        if not tool and names:
            head.append("Tools used: " + ", ".join(f"{name} {total}" for name, total in names.most_common()) + ".")
        if not lines:
            head.append("No call in the turns left out of this request matched.")
            return "\n".join(head)
        page = lines[offset : offset + LIST_PAGE]
        if not page:
            head.append(f"Offset {offset} is past the last of the {len(lines):,} {unit}.")
            return "\n".join(head)
        summary = f"{len(lines):,} {unit} with tool calls; showing {offset + 1}-{offset + len(page)}, newest first"
        if offset + len(page) < len(lines):
            summary += f"; next page: offset={offset + len(page)}"
        head.append(summary + ".")
        return "\n".join(head) + "\n" + "\n".join(page)

    def view(self, call_id, part, offset, max_chars, start_line, end_line):
        offset = int(offset or 0)
        max_chars = int(max_chars) if max_chars else view_limits()[0]
        record = self.by_id.get(call_id)
        if record is None:
            return (
                f"Error: no tool call {call_id} on this chat as it is now; ids look like 37.4 (turn 37, fourth call) "
                "and name a position, so an edit, a deleted message or a branch switch can move them."
            )
        head = [
            f"[{record.id}] turn {record.turn} ({self.age(record)}) · {record.name} · {self.status_word(record)} · call id {record.call_id}",
            f'turn {record.turn} began: "{self.opening.get(record.turn, "")}"',
        ]
        if self.in_view(record):
            head.append("This turn is in this request" + (" with a note in place of the result." if self.notes_in_view else "."))
        if part == "arguments":
            text = record.arguments or "(no arguments)"
        else:
            shown = record.arguments
            if len(shown) <= ARGUMENTS_INLINE_CHARS:
                head.append("arguments:" + ("\n" + shown if shown else " (none)"))
            else:
                head.append(
                    f"arguments: {len(shown):,} chars; read them with part=arguments\n" + shown[:ARGUMENTS_INLINE_CHARS] + "…"
                )
            if not record.answered:
                head.append("result: none stored for this call.")
                return "\n".join(head)
            text = record.result_text() or ""
        total = len(text)
        if start_line is not None:
            lines = text.split("\n")
            first = max(1, int(start_line)) - 1
            last = min(len(lines), int(end_line) if end_line else first + VIEW_LINES)
            body = "\n".join(f"{first + index + 1}: {line}" for index, line in enumerate(lines[first:last]))
            navigation = f"{part}: lines {first + 1}-{last} of {len(lines):,} ({total:,} chars)"
            if last < len(lines):
                navigation += f"; next: start_line={last + 1}"
        else:
            body = text[offset : offset + max_chars]
            navigation = f"{part}: chars {offset:,}-{offset + len(body):,} of {total:,}"
            if offset + len(body) < total:
                navigation += f"; next: offset={offset + len(body)}"
        tail = (
            f"This is what the tool returned in turn {record.turn}; run the tool again for the current state."
            if part == "result"
            else f"These are the arguments of the call made in turn {record.turn}."
        )
        return "\n".join(head + [navigation, body, tail])


_SEARCH_DESC_HEAD_ON = (
    "Search the tool calls made in earlier turns of this chat: their tool names, arguments and results. Tool calls from "
    "older turns of this chat are left out of what you see; the ones this chat still stores can be searched with this "
    "tool. "
)
_SEARCH_DESC_HEAD_OFF = (
    "Search the tool calls made in earlier turns of this chat: their tool names, arguments and results. Tool calls from "
    "older turns of this chat are left out of what you see, and each earlier result stands in the conversation as a note "
    "naming the id that view_tool_call reads; the ones this chat still stores can be searched with this tool. "
)
_SEARCH_DESC_TAIL = (
    "Use it when the person refers to something a tool returned or did earlier, before answering from memory, and before "
    "running a tool again only to see what it returned then. A stored result shows what the tool returned at that time; "
    "for the current state, run the tool again. Each hit gives an id such as 37.4 (turn 37, fourth call) to read in full "
    "with view_tool_call, the person's message that began that turn, the arguments, and the matching line. Ids are "
    "positions on this chat's current path, so an edit, a deleted message or a branch switch can move them."
)
_LIST_DESC = (
    "List the tool calls made in earlier turns of this chat, newest first. With no tool and no single turn given you get "
    "one line per turn: the person's message that began it and the tools it used. With a tool, or from_turn equal to "
    "to_turn, you get one line per call with its id for view_tool_call."
)
_VIEW_DESC = (
    "Read one tool call from an earlier turn of this chat in full: its arguments and its result as stored. Long text "
    "comes in pages; the reply says how to get the next one. The result is what the tool returned in that turn, not the "
    "current state."
)

_SEARCH_PARAMETERS = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to look for: words, exact text or an RE2 pattern, as mode says."},
        "mode": {
            "type": "string",
            "enum": ["keyword", "exact", "regex"],
            "description": "keyword (default): calls holding the whole query first, then calls holding more of its words, each word matched case-insensitively; text in double quotes is one unit. exact: the text exactly as given, ignoring case, no special characters; use it for paths, error messages and IDs. regex: an RE2 pattern, case-insensitive (no lookarounds or backreferences).",
        },
        "search_in": {
            "type": "string",
            "enum": ["all", "tool_name", "arguments", "result"],
            "description": "Where to look. Default all.",
        },
        "tool": {"type": "string", "description": "Only calls of this tool, by its name as the earlier turn recorded it; list_tool_history shows the names."},
        "status": {
            "type": "string",
            "enum": ["any", "succeeded", "failed"],
            "description": "Only calls that succeeded or only calls that failed, as the call's card would show it. Default any.",
        },
        "from_turn": {"type": "integer", "description": "Only calls from this turn on. Turns are numbered from the person's first message; every reply of these tools says the current turn's number."},
        "to_turn": {"type": "integer", "description": "Only calls up to and including this turn."},
        "count": {"type": "integer", "description": "Hits per page, 1-50. Default 10."},
        "offset": {"type": "integer", "description": "Hits to skip, for the next page. Default 0."},
    },
    "required": ["query"],
}

_LIST_PARAMETERS = {
    "type": "object",
    "properties": {
        "tool": {"type": "string", "description": "Only calls of this tool, by its name as the earlier turn recorded it."},
        "status": {
            "type": "string",
            "enum": ["any", "succeeded", "failed"],
            "description": "Only calls that succeeded or only calls that failed. Default any.",
        },
        "from_turn": {"type": "integer", "description": "Only calls from this turn on. Turns are numbered from the person's first message; every reply of these tools says the current turn's number."},
        "to_turn": {"type": "integer", "description": "Only calls up to and including this turn."},
        "offset": {"type": "integer", "description": "Lines to skip, for the next page. Default 0."},
    },
}

_VIEW_PARAMETERS = {
    "type": "object",
    "properties": {
        "id": {"type": "string", "description": "The call's id from search_tool_history or list_tool_history, such as 37.4 (turn 37, fourth call)."},
        "part": {
            "type": "string",
            "enum": ["result", "arguments"],
            "description": "Which part to page through. Default result; the arguments are shown with it when short.",
        },
        "offset": {"type": "integer", "description": "Character to start at. Default 0."},
        "max_chars": {"type": "integer", "description": "Characters to return, up to the page ceiling. Default the standard page size."},
        "start_line": {"type": "integer", "description": "First line to return (1-based), instead of offset; numbers each line."},
        "end_line": {"type": "integer", "description": "Last line to return. Default start_line plus 99."},
    },
    "required": ["id"],
}


def _marker_ids(message: dict[str, Any]) -> list[str]:
    text = message.get("content")
    text = text if isinstance(text, str) else _message_text(text)
    if _MARKER_SUFFIX not in text:
        return []
    spans = _iter_marker_spans(text)
    if not spans:
        return []
    return [
        str(seg["marker"])
        for seg in split_text_by_markers(text, spans)
        if isinstance(seg, dict) and seg.get("type") == "marker"
    ]


def _chat_log_subject_safe(chat_id: Any) -> str:
    return "temporary" if not chat_id else str(chat_id)


class _ReplyHistory:
    def __init__(
        self,
        pipe: Any,
        messages: list[dict[str, Any]],
        *,
        chat_id: str | None,
        reply_id: str | None,
        window_turns: int,
        notes_in_view: bool,
        ask_user_names: frozenset[str],
    ) -> None:
        self.pipe = pipe
        self.messages = messages
        self.chat_id = chat_id
        self.reply_id = reply_id
        self.window_turns = window_turns
        self.notes_in_view = notes_in_view
        self.ask_user_names = frozenset(ask_user_names)
        self.self_names: frozenset[str] = frozenset()
        self._history: History | None = None
        self._lock = asyncio.Lock()

    async def history(self) -> History:
        async with self._lock:
            if self._history is None:
                artifacts: dict[str | None, dict[str, dict[str, Any]]] = {}
                groups: dict[str | None, list[str]] = {}
                for message in self.messages:
                    if not isinstance(message, dict) or (message.get("role") or "").lower() != "assistant":
                        continue
                    markers = _marker_ids(message)
                    if not markers:
                        continue
                    group_id = message.get("message_id") or message_identifier(message)
                    groups.setdefault(group_id, []).extend(markers)
                if groups and self.pipe._artifact_store is not None:
                    gate = asyncio.Semaphore(_HISTORY_ROW_LOAD_CONCURRENCY)

                    async def _load(group_id: str | None, markers: list[str]):
                        async with gate:
                            try:
                                return group_id, await self.pipe._artifact_store._db_fetch(
                                    self.chat_id, group_id, list(dict.fromkeys(markers)),
                                    reply_id=self.reply_id, with_producers=False,
                                )
                            except Exception:
                                logger.warning(
                                    "The tool history's stored-row load failed for chat %s",
                                    _chat_log_subject_safe(self.chat_id), exc_info=True,
                                )
                                return group_id, {}

                    for group_id, payload in await asyncio.gather(
                        *(_load(group_id, markers) for group_id, markers in groups.items())
                    ):
                        if payload:
                            artifacts[group_id] = payload
                self._history = await asyncio.to_thread(
                    History,
                    self.messages,
                    artifacts,
                    window_turns=self.window_turns,
                    notes_in_view=self.notes_in_view,
                    ask_user_names=self.ask_user_names,
                    self_names=self.self_names,
                )
            return self._history


def _out_of_range(name: str, value: Any, low: int, high: int | None = None) -> str:
    if high is None:
        return f"Error: {name} must be at least {low}; got {value}."
    return f"Error: {name} must be between {low} and {high}; got {value}."


def _turn_arguments(from_turn, to_turn, offset) -> str | None:
    if from_turn is not None and from_turn < 1:
        return _out_of_range("from_turn", from_turn, 1)
    if to_turn is not None and to_turn < 1:
        return _out_of_range("to_turn", to_turn, 1)
    if from_turn is not None and to_turn is not None and from_turn > to_turn:
        return f"Error: from_turn {from_turn} is after to_turn {to_turn}."
    if offset is None or offset < 0:
        return _out_of_range("offset", offset, 0)
    return None


def history_entries(
    pipe: Any,
    *,
    messages: list[dict[str, Any]],
    chat_id: str | None,
    reply_id: str | None,
    window_turns: int,
    notes_in_view: bool,
    ask_user_names: frozenset[str],
) -> dict[str, dict[str, Any]]:
    state = _ReplyHistory(
        pipe,
        list(messages or ()),
        chat_id=chat_id,
        reply_id=reply_id,
        window_turns=window_turns,
        notes_in_view=notes_in_view,
        ask_user_names=ask_user_names,
    )

    async def list_tool_history(
        tool: str | None = None,
        status: str = "any",
        from_turn: int | None = None,
        to_turn: int | None = None,
        offset: int = 0,
    ) -> str:
        if status not in ("any", "succeeded", "failed"):
            return f"Error: status must be any, succeeded or failed; got {status}."
        argument_error = _turn_arguments(from_turn, to_turn, offset)
        if argument_error:
            return argument_error
        history = await state.history()
        return await asyncio.to_thread(history.listing, tool or None, status, from_turn, to_turn, int(offset or 0))

    async def search_tool_history(
        query: str,
        mode: str = "keyword",
        search_in: str = "all",
        tool: str | None = None,
        status: str = "any",
        from_turn: int | None = None,
        to_turn: int | None = None,
        count: int = SEARCH_DEFAULT,
        offset: int = 0,
    ) -> str:
        if not isinstance(query, str) or not query.strip():
            return "Error: query is empty. To see what was done, use list_tool_history."
        if len(query) > PATTERN_MAX:
            return f"Error: queries over {PATTERN_MAX:,} characters are not supported."
        if mode not in ("keyword", "exact", "regex"):
            return f"Error: mode must be keyword, exact or regex; got {mode}."
        if search_in not in ("all", "tool_name", "arguments", "result"):
            return f"Error: search_in must be all, tool_name, arguments or result; got {search_in}."
        if status not in ("any", "succeeded", "failed"):
            return f"Error: status must be any, succeeded or failed; got {status}."
        if count is None or not 1 <= count <= SEARCH_MAX:
            return _out_of_range("count", count, 1, SEARCH_MAX)
        argument_error = _turn_arguments(from_turn, to_turn, offset)
        if argument_error:
            return argument_error
        history = await state.history()
        return await asyncio.to_thread(
            history.search, query, mode, search_in, tool or None, status, from_turn, to_turn, int(count), int(offset or 0)
        )

    async def view_tool_call(
        id: str,
        part: str = "result",
        offset: int = 0,
        max_chars: int | None = None,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        call_id = str(id).strip() if isinstance(id, str) else ""
        if not re.fullmatch(r"\d+\.\d+", call_id):
            return f"Error: {call_id!r} is not a call id; ids look like 37.4 (turn 37, fourth call)."
        if part not in ("result", "arguments"):
            return f"Error: part must be result or arguments; got {part}."
        default_chars, cap_chars = view_limits()
        if max_chars is None:
            max_chars = default_chars
        if not 1 <= max_chars <= cap_chars:
            return _out_of_range("max_chars", max_chars, 1, cap_chars)
        if offset is None or offset < 0:
            return _out_of_range("offset", offset, 0)
        if start_line is not None and start_line < 1:
            return _out_of_range("start_line", start_line, 1)
        if end_line is not None and (start_line is None or end_line < start_line):
            return "Error: end_line needs a start_line and must not be before it."
        history = await state.history()
        return await asyncio.to_thread(history.view, call_id, part, int(offset or 0), int(max_chars), start_line, end_line)

    state.self_names = frozenset(HISTORY_TOOL_NAMES)

    def _entry(name: str, description: str, parameters: dict[str, Any], callable_) -> dict[str, Any]:
        return {
            "spec": {"type": "function", "name": name, "description": description, "parameters": parameters},
            "callable": callable_,
            "type": "pipe",
            "tool_id": f"{HISTORY_PIPE_TOOL_PREFIX}{name}",
        }

    search_description = (_SEARCH_DESC_HEAD_OFF if notes_in_view else _SEARCH_DESC_HEAD_ON) + _SEARCH_DESC_TAIL
    view_default, view_cap = view_limits()
    view_parameters = {
        **_VIEW_PARAMETERS,
        "properties": {
            **_VIEW_PARAMETERS["properties"],
            "max_chars": {
                "type": "integer",
                "description": f"Characters to return, up to {view_cap}. Default {view_default}.",
            },
        },
    }
    return {
        "search_tool_history": _entry("search_tool_history", search_description, _SEARCH_PARAMETERS, search_tool_history),
        "list_tool_history": _entry("list_tool_history", _LIST_DESC, _LIST_PARAMETERS, list_tool_history),
        "view_tool_call": _entry("view_tool_call", _VIEW_DESC, view_parameters, view_tool_call),
    }
