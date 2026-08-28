"""Keystone end-to-end replay: pipe emit stream -> OWUI 0.11.1 output assembly.

This test proves that the events the pipe emits, when consumed by Open WebUI
0.11.1's ACTUAL output-assembly logic, yield a correct final ``output`` array:
distinct completed reasoning items, intact tool cards, and an intact answer
message.

It works in two layers:

1. The REAL pipe streaming loop is driven with fake upstream streams (same
   harness idioms as ``tests/test_reasoning_native_items.py``:
   ``_make_timed_stream`` / ``_install_clock`` / ``_run``, monkeypatching
   ``Pipe.send_openrouter_streaming_request`` and ``THINKING_OUTPUT_MODE
   = "open_webui"``). The pipe emits its downstream events into a list.

2. Those emitted events are fed through OWUI output-assembly logic VENDORED
   (copied) from
   ``.external/open-webui/backend/open_webui/utils/middleware.py`` — we do NOT
   import ``open_webui`` (conftest stubs it). The vendored pieces are:
     * ``deep_merge``                       (middleware.py ~403-421)
     * ``handle_responses_streaming_event`` (middleware.py ~597-900, including
       0.11.1's ``.added`` trichotomy: replace on matching id/call_id, insert at
       an in-range ``output_index``, otherwise append)
     * ``tag_output_handler``               (middleware.py 4219-4496 verbatim,
       dedented to module level; its two closure dicts ``tag_scan_positions`` /
       ``tag_boundary_positions`` become module globals reset per assembly)
     * ``_start_tag_pattern``               (middleware.py 240-243)
     * ``DEFAULT_REASONING_TAGS`` / ``DEFAULT_SOLUTION_TAGS`` /
       ``DEFAULT_CODE_INTERPRETER_TAGS`` (middleware.py 225-237)
     * ``_process_value_chunk``             (the running-content / reasoning-close
       / inside-tag-block / trailing-message logic that lives inline inside
       ``stream_body_handler`` and cannot be imported in isolation)
     * ``_finalize_output``                 (post-loop cleanup: empty-trailing-
       message pop + last-reasoning close)
   ``output_id`` is replaced with a deterministic counter-based stand-in
   producing ``r_<n>`` / ``msg_<n>`` (the real one returns ``prefix_<hex>``;
   only the shape matters for assembly).

   ``owui_assemble`` takes ``detect_reasoning_tags`` so both sides of OWUI's
   ``DETECT_REASONING_TAGS`` gate (``middleware.py:4580``, an identity check
   against literal ``False``) can be exercised.

DEVIATIONS from the vendored OWUI logic (intentional, per task scope):
  * The base64 image conversion (guarded by
    ``ENABLE_CHAT_RESPONSE_BASE64_IMAGE_URL_CONVERSION``) is omitted — no
    base64 in this data.
  * ``ENABLE_REALTIME_CHAT_SAVE`` DB persistence is omitted — not
    output-building.
  * The fusion answer path is not exercised here; fusion resets
    ``assistant_message`` and emits its own message item, so its interaction
    with the pipe's terminal ``response.completed`` array is uncovered.
"""
# pyright: reportArgumentType=false, reportOptionalSubscript=false, reportAttributeAccessIssue=false, reportOptionalMemberAccess=false, reportCallIssue=false

from __future__ import annotations

import re
import time
from typing import Any, cast

import pytest

from open_webui_openrouter_pipe import Pipe, ResponsesBody
from open_webui_openrouter_pipe.streaming import streaming_core


# ─────────────────────────────────────────────────────────────────────────────
# Vendored OWUI 0.10.2 output-assembly logic (see module docstring for sources)
# ─────────────────────────────────────────────────────────────────────────────

_ID_STATE = {"n": 0}


def _reset_output_id() -> None:
    _ID_STATE["n"] = 0
    tag_scan_positions.clear()
    tag_boundary_positions.clear()


def output_id(prefix: str) -> str:
    """Counter-based stand-in for OWUI's ``output_id`` (real: ``prefix_<hex>``)."""
    n = _ID_STATE["n"]
    _ID_STATE["n"] = n + 1
    return f"{prefix}_{n}"


# Verbatim ports from middleware.py: the tag constants (:225-237), the start-tag
# pattern builder (:240-243), and the two per-response position dicts (:4215-4216)
# that tag_output_handler carries as closure state.
DEFAULT_REASONING_TAGS = [
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reason>", "</reason>"),
    ("<reasoning>", "</reasoning>"),
    ("<thought>", "</thought>"),
    ("<Thought>", "</Thought>"),
    ("<|begin_of_thought|>", "<|end_of_thought|>"),
    ("\u25c1think\u25b7", "\u25c1/think\u25b7"),
]

DEFAULT_SOLUTION_TAGS = [("<|begin_of_solution|>", "<|end_of_solution|>")]
DEFAULT_CODE_INTERPRETER_TAGS = [("<code_interpreter>", "</code_interpreter>")]

tag_scan_positions: dict = {}
tag_boundary_positions: dict = {}


def _start_tag_pattern(start_tag: str) -> str:
    """Verbatim port of middleware.py:240-243."""
    if start_tag.startswith("<") and start_tag.endswith(">"):
        return rf"<{re.escape(start_tag[1:-1])}(\s.*?)?>"
    return re.escape(start_tag)


def tag_output_handler(content_type, tags, output):
    """
    Detect special tags (reasoning, solution, code_interpreter) in streaming
    content and create corresponding OR-aligned output items directly.
    Operates on output items instead of content_blocks.

    Uses the text from the output items themselves for tag detection,
    eliminating state divergence between accumulated content and items.
    """
    end_flag = False

    def extract_attributes(tag_content):
        """Extract attributes from a tag if they exist."""
        attributes = {}
        if not tag_content:
            return attributes
        matches = re.findall(r'(\w+)\s*=\s*"([^"]+)"', tag_content)
        for key, value in matches:
            attributes[key] = value
        return attributes

    def get_last_text(out):
        """Get text from last message item, or empty string."""
        if out and out[-1].get('type') == 'message':
            parts = out[-1].get('content', [])
            if parts and parts[-1].get('type') == 'output_text':
                return parts[-1].get('text', '')
        return ''

    def set_last_text(out, text):
        """Set text on last message item's output_text."""
        if out and out[-1].get('type') == 'message':
            parts = out[-1].get('content', [])
            if parts and parts[-1].get('type') == 'output_text':
                parts[-1]['text'] = text

    def get_scanned_length(item, text):
        item_id = item.get('id')
        if not item_id:
            return 0

        scanned_length = tag_scan_positions.get((item_id, content_type), 0)
        return scanned_length if scanned_length <= len(text) else 0

    def save_scanned_length(item, text):
        item_id = item.get('id')
        if item_id:
            tag_scan_positions[(item_id, content_type)] = len(text)

    def clear_scanned_length(item):
        item_id = item.get('id')
        if item_id:
            tag_scan_positions.pop((item_id, content_type), None)
            tag_boundary_positions.pop((item_id, content_type), None)

    def get_tag_boundaries(item, text, scanned_length):
        """Index of the last '<', and of the last '>' or newline, before scanned_length."""
        key = (item.get('id'), content_type)
        scanned, last_open, last_boundary = tag_boundary_positions.get(key, (0, -1, -1))
        if scanned > scanned_length:  # the item was rewritten, so the cached positions are stale
            scanned, last_open, last_boundary = 0, -1, -1

        if scanned < scanned_length:
            # only text added since the last call can move either position
            open_tag = text.rfind('<', scanned, scanned_length)
            if open_tag != -1:
                last_open = open_tag
            boundary = max(
                text.rfind('>', scanned, scanned_length),
                text.rfind('\n', scanned, scanned_length),
            )
            if boundary != -1:
                last_boundary = boundary
            tag_boundary_positions[key] = (scanned_length, last_open, last_boundary)

        return last_open, last_boundary

    # Map content_type to output item type
    output_type_map = {
        'reasoning': 'reasoning',
        'solution': 'message',  # solution tags just produce text
        'code_interpreter': 'open_webui:code_interpreter',
    }
    output_item_type = output_type_map.get(content_type, content_type)

    last_type = output[-1].get('type', '') if output else ''

    if last_type == 'message':
        # Use the output item's own text for tag detection
        item = output[-1]
        item_text = get_last_text(output)
        scanned_length = get_scanned_length(item, item_text)
        max_start_tag_length = max((len(start_tag) for start_tag, _ in tags), default=1)
        search_start = max(0, scanned_length - max_start_tag_length + 1)

        if scanned_length and any(
            start_tag.startswith('<') and start_tag.endswith('>') for start_tag, _ in tags
        ):
            open_tag_start, last_tag_boundary = get_tag_boundaries(item, item_text, scanned_length)
            if open_tag_start > last_tag_boundary:
                search_start = min(search_start, open_tag_start)

        for start_tag, end_tag in tags:
            match = re.compile(_start_tag_pattern(start_tag)).search(item_text, search_start)
            if match:
                clear_scanned_length(item)
                try:
                    attr_content = match.group(1) if match.group(1) else ''
                except Exception:
                    attr_content = ''

                attributes = extract_attributes(attr_content)

                before_tag = item_text[: match.start()]
                after_tag = item_text[match.end() :]

                # Keep only text before the tag in the message
                set_last_text(output, before_tag)

                if not before_tag.strip():
                    # Remove empty message item
                    if output and output[-1].get('type') == 'message':
                        output.pop()

                # Append the new output item
                if output_item_type == 'reasoning':
                    output.append(
                        {
                            'type': 'reasoning',
                            'id': output_id('r'),
                            'status': 'in_progress',
                            'start_tag': start_tag,
                            'end_tag': end_tag,
                            'attributes': attributes,
                            'content': [],
                            'summary': None,
                            'started_at': time.time(),
                        }
                    )
                elif output_item_type == 'open_webui:code_interpreter':
                    output.append(
                        {
                            'type': 'open_webui:code_interpreter',
                            'id': output_id('ci'),
                            'status': 'in_progress',
                            'start_tag': start_tag,
                            'end_tag': end_tag,
                            'attributes': attributes,
                            'lang': attributes.get('lang', 'python'),
                            'code': '',
                            'output': None,
                            'started_at': time.time(),
                        }
                    )
                else:
                    # solution or other text-producing tag
                    output.append(
                        {
                            'type': 'message',
                            'id': output_id('msg'),
                            'status': 'in_progress',
                            'role': 'assistant',
                            'content': [{'type': 'output_text', 'text': ''}],
                            '_tag_type': content_type,
                            'start_tag': start_tag,
                            'end_tag': end_tag,
                            'attributes': attributes,
                            'started_at': time.time(),
                        }
                    )

                if after_tag:
                    # Set the after_tag content on the new item
                    if output_item_type == 'reasoning':
                        output[-1]['content'] = [{'type': 'output_text', 'text': after_tag}]
                    elif output_item_type == 'open_webui:code_interpreter':
                        output[-1]['code'] = after_tag
                    else:
                        set_last_text(output, after_tag)

                    _, recursive_end = tag_output_handler(content_type, tags, output)
                    if recursive_end:
                        end_flag = True

                break
        else:
            save_scanned_length(item, item_text)

    elif (
        (last_type == 'reasoning' and content_type == 'reasoning')
        or (last_type == 'open_webui:code_interpreter' and content_type == 'code_interpreter')
        or (last_type == 'message' and output[-1].get('_tag_type') == content_type)
    ):
        item = output[-1]
        start_tag = item.get('start_tag', '')
        end_tag = item.get('end_tag', '')

        # Get the block content from the item itself
        if last_type == 'reasoning':
            parts = item.get('content', [])
            block_content = ''
            if parts and parts[-1].get('type') == 'output_text':
                block_content = parts[-1].get('text', '')
        elif last_type == 'open_webui:code_interpreter':
            block_content = item.get('code', '')
        else:
            block_content = get_last_text(output)

        scanned_length = get_scanned_length(item, block_content)
        end_tag_search_start = max(0, scanned_length - max(len(end_tag), 1) + 1)

        if block_content.find(end_tag, end_tag_search_start) != -1:
            clear_scanned_length(item)
            end_flag = True

            # Strip start and end tags from content
            start_tag_pattern = _start_tag_pattern(start_tag)
            block_content = re.sub(start_tag_pattern, '', block_content).strip()

            end_tag_pattern = rf'{re.escape(end_tag)}'
            end_tag_regex = re.compile(end_tag_pattern, re.DOTALL)
            split_content = end_tag_regex.split(block_content, maxsplit=1)

            block_content = split_content[0].strip() if split_content else ''
            leftover_content = split_content[1].strip() if len(split_content) > 1 else ''

            if block_content:
                # Update the item with final content
                if last_type == 'reasoning':
                    item['content'] = [{'type': 'output_text', 'text': block_content}]
                    item['ended_at'] = time.time()
                    item['duration'] = int(item['ended_at'] - item['started_at'])
                    item['status'] = 'completed'
                elif last_type == 'open_webui:code_interpreter':
                    item['code'] = block_content
                    item['ended_at'] = time.time()
                    item['duration'] = int(item['ended_at'] - item['started_at'])
                else:
                    set_last_text(output, block_content)
                    item['ended_at'] = time.time()

                # Reset by appending a new message item for leftover
                output.append(
                    {
                        'type': 'message',
                        'id': output_id('msg'),
                        'status': 'in_progress',
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'output_text',
                                'text': leftover_content,
                            }
                        ],
                    }
                )
            else:
                # Remove the block if content is empty
                output.pop()
                output.append(
                    {
                        'type': 'message',
                        'id': output_id('msg'),
                        'status': 'in_progress',
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'output_text',
                                'text': leftover_content,
                            }
                        ],
                    }
                )
        else:
            save_scanned_length(item, block_content)

    return output, end_flag


def deep_merge(target, source):
    """Verbatim port of middleware.py ~403-421."""
    if isinstance(target, dict) and isinstance(source, dict):
        new_target = target.copy()
        for k, v in source.items():
            if k in new_target:
                new_target[k] = deep_merge(new_target[k], v)
            else:
                new_target[k] = v
        return new_target
    elif isinstance(target, str) and isinstance(source, str):
        return target + source
    else:
        return source


def handle_responses_streaming_event(data: dict, current_output: list):
    """Verbatim port of middleware.py ~424-750 (OWUI-internal imports stripped)."""
    event_type = data.get("type", "")

    if event_type == "response.output_item.added":
        item = data.get("item", {})
        if item:
            new_output = list(current_output)
            output_index = data.get("output_index", len(new_output))
            existing_index = next(
                (
                    idx
                    for idx, existing in enumerate(new_output)
                    if (item.get("id") and existing.get("id") == item.get("id"))
                    or (item.get("call_id") and existing.get("call_id") == item.get("call_id"))
                ),
                None,
            )
            if existing_index is not None:
                new_output[existing_index] = item
            elif 0 <= output_index < len(new_output):
                new_output.insert(output_index, item)
            else:
                new_output.append(item)
            return new_output, None
        return current_output, None

    elif event_type == "response.content_part.added":
        part = data.get("part", {})
        output_index = data.get("output_index", len(current_output) - 1)

        if current_output and 0 <= output_index < len(current_output):
            new_output = list(current_output)
            item = new_output[output_index].copy()
            new_output[output_index] = item

            if "content" not in item:
                item["content"] = []
            else:
                item["content"] = list(item["content"])

            if item.get("type") == "reasoning":
                pass
            else:
                item["content"].append(part)
            return new_output, None
        return current_output, None

    elif event_type == "response.reasoning_summary_part.added":
        part = data.get("part", {})
        output_index = data.get("output_index", len(current_output) - 1)

        if current_output and 0 <= output_index < len(current_output):
            new_output = list(current_output)
            item = new_output[output_index].copy()
            new_output[output_index] = item

            if "summary" not in item:
                item["summary"] = []
            else:
                item["summary"] = list(item["summary"])

            item["summary"].append(part)
            return new_output, None
        return current_output, None

    elif event_type.startswith("response.") and event_type.endswith(".delta"):
        parts = event_type.split(".")
        if len(parts) >= 3:
            delta_type = parts[1]
            delta = data.get("delta", "")

            output_index = data.get("output_index", len(current_output) - 1)

            if current_output and 0 <= output_index < len(current_output):
                new_output = list(current_output)
                item = new_output[output_index].copy()
                new_output[output_index] = item
                item_type = item.get("type", "")

                if delta_type == "function_call_arguments":
                    key = "arguments"
                    if item_type == "function_call":
                        item[key] = item.get(key, "") + str(delta)
                else:
                    if item_type == "message":
                        if delta_type in ["text", "output_text"]:
                            key = "text"
                        elif delta_type in ["reasoning_text", "reasoning_summary_text"]:
                            return new_output, None
                        else:
                            key = delta_type

                        content_index = data.get("content_index", 0)
                        if "content" not in item:
                            item["content"] = []
                        else:
                            item["content"] = list(item["content"])
                        content_list = item["content"]

                        while len(content_list) <= content_index:
                            content_list.append({"type": "text", "text": ""})

                        part = content_list[content_index].copy()
                        content_list[content_index] = part

                        current_val = part.get(key)
                        if current_val is None:
                            current_val = {} if isinstance(delta, dict) else ""

                        part[key] = deep_merge(current_val, delta)

                    elif item_type == "reasoning":
                        if delta_type == "reasoning_summary_text":
                            key = "text"
                            summary_index = data.get("summary_index", 0)
                            if "summary" not in item:
                                item["summary"] = []
                            else:
                                item["summary"] = list(item["summary"])
                            summary_list = item["summary"]

                            while len(summary_list) <= summary_index:
                                summary_list.append({"type": "summary_text", "text": ""})

                            part = summary_list[summary_index].copy()
                            summary_list[summary_index] = part

                            target_val = part.get(key, "")
                            part[key] = deep_merge(target_val, delta)

                        elif delta_type == "reasoning_text":
                            key = "text"
                            content_index = data.get("content_index", 0)
                            if "content" not in item:
                                item["content"] = []
                            else:
                                item["content"] = list(item["content"])
                            content_list = item["content"]

                            while len(content_list) <= content_index:
                                content_list.append({"type": "text", "text": ""})

                            part = content_list[content_index].copy()
                            content_list[content_index] = part

                            target_val = part.get(key, "")
                            part[key] = deep_merge(target_val, delta)

                        elif delta_type in ["text", "output_text"]:
                            return new_output, None
                        else:
                            pass

                    else:
                        if delta_type in ["text", "output_text"]:
                            key = "text"
                        else:
                            key = delta_type

                        current_val = item.get(key)
                        if current_val is None:
                            current_val = {} if isinstance(delta, dict) else ""
                        item[key] = deep_merge(current_val, delta)

            # Faithful to middleware.py:613 — OWUI returns new_output OUTSIDE the
            # bounds guard, so an out-of-range delta raises UnboundLocalError there
            # too. The port must keep that behavior.
            return new_output, None  # pyright: ignore[reportPossiblyUnboundVariable]

    elif event_type.startswith("response.") and event_type.endswith(".done"):
        parts = event_type.split(".")
        if len(parts) >= 3:
            type_name = parts[1]

            if type_name == "content_part":
                part = data.get("part")
                output_index = data.get("output_index", len(current_output) - 1)

                if part and current_output and 0 <= output_index < len(current_output):
                    new_output = list(current_output)
                    item = new_output[output_index].copy()
                    new_output[output_index] = item

                    if "content" in item:
                        item["content"] = list(item["content"])
                        content_index = data.get("content_index", len(item["content"]) - 1)
                        if 0 <= content_index < len(item["content"]):
                            item["content"][content_index] = part
                            return new_output, {}
                return current_output, None

            elif type_name == "reasoning_summary_part":
                part = data.get("part")
                output_index = data.get("output_index", len(current_output) - 1)

                if part and current_output and 0 <= output_index < len(current_output):
                    new_output = list(current_output)
                    item = new_output[output_index].copy()
                    new_output[output_index] = item

                    if "summary" in item:
                        item["summary"] = list(item["summary"])
                        summary_index = data.get("summary_index", len(item["summary"]) - 1)
                        if 0 <= summary_index < len(item["summary"]):
                            item["summary"][summary_index] = part
                            return new_output, {}
                return current_output, None

            if type_name == "output_item":
                pass

            elif type_name not in ["completed", "failed"]:
                output_index = data.get("output_index", len(current_output) - 1)
                if current_output and 0 <= output_index < len(current_output):
                    key = (
                        "text"
                        if type_name
                        in ["text", "output_text", "reasoning_text", "reasoning_summary_text"]
                        else type_name
                    )
                    if type_name == "function_call_arguments":
                        key = "arguments"

                    if key in data:
                        final_value = data[key]
                        new_output = list(current_output)
                        item = new_output[output_index].copy()
                        new_output[output_index] = item
                        item_type = item.get("type", "")

                        if type_name == "function_call_arguments":
                            if item_type == "function_call":
                                item["arguments"] = final_value
                        elif item_type == "message":
                            content_index = data.get("content_index", 0)
                            if "content" in item:
                                item["content"] = list(item["content"])
                                if len(item["content"]) > content_index:
                                    part = item["content"][content_index].copy()
                                    item["content"][content_index] = part
                                    part[key] = final_value
                        elif item_type == "reasoning":
                            item["status"] = "completed"
                        else:
                            item[key] = final_value

                        return new_output, {}

        return current_output, None

    elif event_type == "response.output_item.done":
        item = data.get("item")
        output_index = data.get("output_index", len(current_output) - 1)

        new_output = list(current_output)
        if item and 0 <= output_index < len(current_output):
            new_output[output_index] = item
        elif item:
            new_output.append(item)
        return new_output, {}

    elif event_type == "response.completed":
        response_data = data.get("response", {})
        final_output = response_data.get("output")

        new_output = final_output if final_output is not None else current_output

        if new_output:
            for item in new_output:
                if item.get("type") == "reasoning" and item.get("status") != "completed":
                    item["status"] = "completed"

        return new_output, {
            "usage": response_data.get("usage"),
            "done": True,
            "response_id": response_data.get("id"),
        }

    elif event_type == "response.in_progress":
        return current_output, None

    elif event_type == "response.failed":
        error = data.get("response", {}).get("error", {})
        return current_output, {"error": error}

    else:
        return current_output, None


def _process_value_chunk(
    value: str, output: list, content: str, detect_reasoning_tags: bool = True
) -> str:
    """Port of middleware.py ~4361-4479 (base64 conv + tag_output_handler stripped).

    Mutates ``output`` in place, returns the updated running ``content`` string.
    """
    if value:
        # (a) a still-open reasoning-content box closes when answer text starts.
        if (
            output
            and output[-1].get("type") == "reasoning"
            and output[-1].get("attributes", {}).get("type") == "reasoning_content"
        ):
            reasoning_item = output[-1]
            reasoning_item["ended_at"] = time.time()
            reasoning_item["duration"] = int(
                reasoning_item["ended_at"] - reasoning_item["started_at"]
            )
            reasoning_item["status"] = "completed"

            output.append(
                {
                    "type": "message",
                    "id": output_id("msg"),
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": ""}],
                }
            )

        content = f"{content}{value}"

        # (b) inside an in-progress tag block -> append INTO that item.
        last_item = output[-1] if output else None
        last_item_type = last_item.get("type", "") if last_item else ""
        inside_tag_block = (
            last_item is not None
            and last_item.get("status") == "in_progress"
            and last_item.get("attributes", {}).get("type") != "reasoning_content"
            and (
                last_item_type == "reasoning"
                or last_item_type == "open_webui:code_interpreter"
                or (last_item_type == "message" and last_item.get("_tag_type") is not None)
            )
        )

        if inside_tag_block:
            if last_item_type == "open_webui:code_interpreter":
                last_item["code"] = last_item.get("code", "") + value
            elif last_item_type == "reasoning":
                parts = last_item.get("content", [])
                if parts and parts[-1].get("type") == "output_text":
                    parts[-1]["text"] += value
                else:
                    last_item["content"] = [{"type": "output_text", "text": value}]
            else:
                msg_parts = last_item.get("content", [])
                if msg_parts and msg_parts[-1].get("type") == "output_text":
                    msg_parts[-1]["text"] += value
                else:
                    last_item["content"] = [{"type": "output_text", "text": value}]
        # (c) otherwise create/extend a trailing message item.
        else:
            if not output or output[-1].get("type") != "message":
                output.append(
                    {
                        "type": "message",
                        "id": output_id("msg"),
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": ""}],
                    }
                )

            msg_parts = output[-1].get("content", [])
            if msg_parts and msg_parts[-1].get("type") == "output_text":
                msg_parts[-1]["text"] += value
            else:
                output[-1]["content"] = [{"type": "output_text", "text": value}]

        if detect_reasoning_tags:
            scanned, _ = tag_output_handler("reasoning", DEFAULT_REASONING_TAGS, output)
            output[:] = scanned
            scanned, _ = tag_output_handler("solution", DEFAULT_SOLUTION_TAGS, output)
            output[:] = scanned

    return content


def _finalize_output(output: list) -> list:
    """Verbatim port of middleware.py ~4547-4573 post-loop cleanup."""
    if output:
        if output[-1].get("type") == "message":
            parts = output[-1].get("content", [])
            if parts and parts[-1].get("type") == "output_text":
                parts[-1]["text"] = parts[-1]["text"].strip()

                if not parts[-1]["text"]:
                    output.pop()

                    if not output:
                        output.append(
                            {
                                "type": "message",
                                "id": output_id("msg"),
                                "status": "in_progress",
                                "role": "assistant",
                                "content": [{"type": "output_text", "text": ""}],
                            }
                        )

        if output[-1].get("type") == "reasoning":
            reasoning_item = output[-1]
            if reasoning_item.get("ended_at") is None:
                reasoning_item["ended_at"] = time.time()
                reasoning_item["duration"] = int(
                    reasoning_item["ended_at"] - reasoning_item["started_at"]
                )
                reasoning_item["status"] = "completed"

    return output


def owui_assemble(emitted: list[dict], detect_reasoning_tags: bool = True) -> list[dict]:
    """Consume the pipe's emitted events exactly as OWUI 0.10.2 would.

    Mapping (per the pipe's middleware emitter contract):
      * ``response.*``          -> handle_responses_streaming_event(event, output)
      * ``chat:message:delta``  -> value-chunk processor with data.content
      * everything else (status / chat:completion / …) -> ignored (not
        output-building)
    Then run the post-loop finalizer.
    """
    _reset_output_id()
    output: list[dict] = []
    content = ""
    for event in emitted:
        etype = event.get("type", "")
        if etype.startswith("response."):
            result = handle_responses_streaming_event(event, output)
            # OWUI's elif chain implicitly returns None for a malformed 2-part
            # ".delta" event type; the pipe never emits such shapes.
            assert result is not None, f"unhandled event shape: {etype}"
            output, _meta = result
        elif etype == "chat:message:delta":
            value = (event.get("data") or {}).get("content") or ""
            content = _process_value_chunk(value, output, content, detect_reasoning_tags)
    _finalize_output(output)
    return output


# ─────────────────────────────────────────────────────────────────────────────
# Pipe streaming harness (idioms copied from tests/test_reasoning_native_items.py)
# ─────────────────────────────────────────────────────────────────────────────


def _make_timed_stream(steps: list[tuple[float, dict[str, Any]]], clock: dict[str, float]):
    async def fake_stream(self, session, request_body, **_kwargs):
        for advance, event in steps:
            clock["now"] += advance
            yield event
    return fake_stream


def _install_clock(monkeypatch) -> dict[str, float]:
    clock = {"now": 1000.0}
    monkeypatch.setattr(streaming_core, "_monotonic", lambda: clock["now"])
    return clock


async def _run(pipe, valves, steps, clock, monkeypatch) -> list[dict]:
    body = ResponsesBody(model="test/model", input=[], stream=True)
    monkeypatch.setattr(
        Pipe, "send_openrouter_streaming_request", _make_timed_stream(steps, clock)
    )
    emitted: list[dict] = []

    async def emitter(event):
        emitted.append(event)

    await pipe._streaming_handler._run_streaming_loop(
        body,
        valves,
        emitter,
        metadata={"model": {"id": "test"}},
        tools={},
        session=cast(Any, object()),
        user_id="user-123",
    )
    return emitted


def _reasoning_items(output: list[dict]) -> list[dict]:
    return [i for i in output if i.get("type") == "reasoning"]


def _message_items(output: list[dict]) -> list[dict]:
    return [i for i in output if i.get("type") == "message"]


def _message_text(item: dict) -> str:
    return "".join(
        p.get("text", "")
        for p in (item.get("content") or [])
        if p.get("type") == "output_text"
    )


def _summary_text(item: dict) -> str:
    summary = item.get("summary") or []
    return summary[0].get("text", "") if summary else ""


def _assert_completed_reasoning(item: dict) -> None:
    assert item.get("type") == "reasoning"
    assert item.get("status") == "completed"
    assert item.get("attributes", {}).get("type") != "reasoning_content"
    assert isinstance(item.get("started_at"), float)
    assert isinstance(item.get("ended_at"), float)
    assert isinstance(item.get("duration"), float)
    assert item["duration"] >= 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Case A — snapshot interleave (gpt-5 shape) with a server tool
# ─────────────────────────────────────────────────────────────────────────────


class TestReplayCaseA:
    @pytest.mark.asyncio
    async def test_snapshot_interleave_yields_distinct_boxes_intact_tools_and_answer(
        self, monkeypatch, pipe_instance_async
    ):
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(
            update={"THINKING_OUTPUT_MODE": "open_webui", "SHOW_TOOL_CARDS": True}
        )
        steps = [
            (0.0, {"type": "response.output_item.added",
                   "item": {"type": "reasoning", "id": "rs-1", "summary": []}}),
            (0.5, {"type": "response.output_item.added",
                   "item": {"type": "openrouter:web_search", "id": "ws-1"}}),
            (2.0, {"type": "response.output_item.done",
                   "item": {"type": "openrouter:web_search", "id": "ws-1", "action": {}}}),
            (0.5, {"type": "response.output_item.done",
                   "item": {"type": "reasoning", "id": "rs-1", "status": "completed",
                            "summary": [{"type": "summary_text", "text": "First reasoning block."}],
                            "encrypted_content": "enc-1"}}),
            (0.5, {"type": "response.output_item.added",
                   "item": {"type": "reasoning", "id": "rs-2", "summary": []}}),
            (1.5, {"type": "response.output_item.done",
                   "item": {"type": "reasoning", "id": "rs-2", "status": "completed",
                            "summary": [{"type": "summary_text", "text": "Second reasoning block."}],
                            "encrypted_content": "enc-2"}}),
            (0.5, {"type": "response.output_item.added",
                   "item": {"type": "reasoning", "id": "rs-3", "summary": []}}),
            (0.5, {"type": "response.output_item.done",
                   "item": {"type": "reasoning", "id": "rs-3", "summary": [],
                            "encrypted_content": "enc-3"}}),
            (1.0, {"type": "response.output_text.delta", "delta": "Answer text."}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)
        output = owui_assemble(emitted)

        types = [i.get("type") for i in output]
        assert types == [
            "function_call",
            "function_call_output",
            "reasoning",
            "reasoning",
            "message",
        ], types

        # Exactly two completed reasoning boxes, distinct texts, rs-3 dropped.
        reasoning = _reasoning_items(output)
        assert len(reasoning) == 2
        assert [r.get("id") for r in reasoning] == ["rs-1", "rs-2"]
        for r in reasoning:
            _assert_completed_reasoning(r)
        assert _summary_text(reasoning[0]) == "First reasoning block."
        assert _summary_text(reasoning[1]) == "Second reasoning block."
        assert reasoning[0]["duration"] == pytest.approx(0.5)
        assert reasoning[1]["duration"] == pytest.approx(1.5)
        assert not any(r.get("id") == "rs-3" for r in reasoning)
        assert not any("enc-3" == r.get("encrypted_content") for r in reasoning)

        # Tool cards intact and NOT overwritten with any reasoning fields.
        fc = output[0]
        assert fc == {
            "type": "function_call",
            "id": "ws-1",
            "call_id": "ws-1",
            "name": "web_search",
            "arguments": "{}",
            "status": "completed",
        }
        fco = output[1]
        assert fco["type"] == "function_call_output"
        assert fco["call_id"] == "ws-1"
        assert fco["status"] == "completed"
        assert fco["output"] == [
            {"type": "input_text",
             "text": "Search completed. Sources available in the citations panel below."}
        ]
        for card in (fc, fco):
            assert "summary" not in card
            assert "duration" not in card
            assert "started_at" not in card
            assert card.get("attributes", {}).get("type") != "reasoning_content"

        # Exactly one message, carrying the answer.
        messages = _message_items(output)
        assert len(messages) == 1
        assert _message_text(messages[0]) == "Answer text."


# ─────────────────────────────────────────────────────────────────────────────
# Case B — Anthropic delta shape (late reasoning output_item.done)
# ─────────────────────────────────────────────────────────────────────────────


class TestReplayCaseB:
    @pytest.mark.asyncio
    async def test_anthropic_delta_yields_single_box_before_answer(
        self, monkeypatch, pipe_instance_async
    ):
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        full_reasoning = "Thinking through it. Almost there. "
        steps = [
            (0.0, {"type": "response.output_item.added",
                   "item": {"type": "reasoning", "id": "rs-a"}}),
            (1.0, {"type": "response.reasoning_text.delta",
                   "item_id": "rs-a", "delta": "Thinking through it. "}),
            (1.0, {"type": "response.reasoning_text.delta",
                   "item_id": "rs-a", "delta": "Almost there. "}),
            (1.0, {"type": "response.output_text.delta", "delta": "The answer."}),
            (2.0, {"type": "response.output_item.done",
                   "item": {"type": "reasoning", "id": "rs-a",
                            "content": [{"type": "reasoning_text", "text": full_reasoning}]}}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)
        output = owui_assemble(emitted)

        types = [i.get("type") for i in output]
        assert types == ["reasoning", "message"], types

        reasoning = _reasoning_items(output)
        assert len(reasoning) == 1
        _assert_completed_reasoning(reasoning[0])
        assert _summary_text(reasoning[0]) == full_reasoning
        assert reasoning[0]["duration"] == pytest.approx(3.0)

        # Reasoning positioned strictly before the message.
        reasoning_idx = next(i for i, it in enumerate(output) if it.get("type") == "reasoning")
        message_idx = next(i for i, it in enumerate(output) if it.get("type") == "message")
        assert reasoning_idx < message_idx

        messages = _message_items(output)
        assert len(messages) == 1
        assert _message_text(messages[0]) == "The answer."

        # No item ever carries the reasoning_content attribute (native path only).
        for item in output:
            assert item.get("attributes", {}).get("type") != "reasoning_content"


# ─────────────────────────────────────────────────────────────────────────────
# Case C — tag-in-answer: literal <think> tag inside the answer text
# ─────────────────────────────────────────────────────────────────────────────


class TestReplayCaseC:
    @pytest.mark.asyncio
    async def test_literal_think_tag_in_answer_is_not_wrapped_in_reasoning(
        self, monkeypatch, pipe_instance_async
    ):
        pipe = pipe_instance_async
        clock = _install_clock(monkeypatch)
        valves = pipe.valves.model_copy(update={"THINKING_OUTPUT_MODE": "open_webui"})
        full_reasoning = "Thinking through it. Almost there. "
        answer = "The answer <think>not real</think> done."
        steps = [
            (0.0, {"type": "response.output_item.added",
                   "item": {"type": "reasoning", "id": "rs-a"}}),
            (1.0, {"type": "response.reasoning_text.delta",
                   "item_id": "rs-a", "delta": "Thinking through it. "}),
            (1.0, {"type": "response.reasoning_text.delta",
                   "item_id": "rs-a", "delta": "Almost there. "}),
            (1.0, {"type": "response.output_text.delta", "delta": answer}),
            (2.0, {"type": "response.output_item.done",
                   "item": {"type": "reasoning", "id": "rs-a",
                            "content": [{"type": "reasoning_text", "text": full_reasoning}]}}),
            (0.0, {"type": "response.completed", "response": {"output": [], "usage": {}}}),
        ]
        emitted = await _run(pipe, valves, steps, clock, monkeypatch)

        # The pipe never wraps the literal tag in a reasoning item: the answer
        # (including the tag) travels solely on chat:message:delta, and the only
        # reasoning emission is the native output_item.added box.
        answer_deltas = [
            (e.get("data") or {}).get("content", "")
            for e in emitted
            if e.get("type") == "chat:message:delta"
        ]
        assert "".join(answer_deltas) == answer
        assert "<think>not real</think>" in "".join(answer_deltas)
        reasoning_events = [
            e for e in emitted
            if e.get("type") == "response.output_item.added"
            and (e.get("item") or {}).get("type") == "reasoning"
        ]
        assert len(reasoning_events) == 1
        assert "<think>" not in _summary_text(reasoning_events[0]["item"])

        output = owui_assemble(emitted)
        types = [i.get("type") for i in output]
        assert types == ["reasoning", "message"], types

        # Reasoning box intact and unaffected by the tag.
        reasoning = _reasoning_items(output)
        assert len(reasoning) == 1
        _assert_completed_reasoning(reasoning[0])
        assert _summary_text(reasoning[0]) == full_reasoning
        assert reasoning[0]["duration"] == pytest.approx(3.0)

        # The message item holds the literal tag text verbatim (tag_output_handler
        # is intentionally out of scope here — see module docstring).
        messages = _message_items(output)
        assert len(messages) == 1
        assert _message_text(messages[0]) == answer
        assert "<think>not real</think>" in _message_text(messages[0])
