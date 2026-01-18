"""Request input sanitization.

This module handles cleaning and normalizing request input before sending
to the provider API. It removes non-replayable artifacts and normalizes
tool call items to ensure consistent format.
"""

import json
from typing import Any, TYPE_CHECKING

from ..api.transforms import _filter_replayable_input_items
from ..core.timing_logger import timed

if TYPE_CHECKING:
    from ..pipe import Pipe
    from ..api.transforms import ResponsesBody


@timed
def _sanitize_request_input(self: "Pipe", body: "ResponsesBody") -> None:
    """Remove non-replayable artifacts that may have snuck into body.input."""
    items = getattr(body, "input", None)
    if not isinstance(items, list):
        return
    sanitized = _filter_replayable_input_items(items, logger=self.logger)
    removed = len(items) - len(sanitized)

    @timed
    def _strip_tool_item_extras(item: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Return a minimal, portable /responses input shape for tool items."""
        changed = False
        item_type = item.get("type")
        if item_type == "function_call":
            call_id = item.get("call_id")
            if not (isinstance(call_id, str) and call_id.strip()):
                candidate = item.get("id")
                if isinstance(candidate, str) and candidate.strip():
                    call_id = candidate.strip()
                    changed = True
            name = item.get("name")
            if not (isinstance(name, str) and name.strip()):
                return item, False
            args = item.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args or {}, ensure_ascii=False)
                changed = True
            minimal = {
                "type": "function_call",
                "call_id": call_id,
                "name": name.strip(),
                "arguments": args,
            }
            if set(item.keys()) != set(minimal.keys()):
                changed = True
            return minimal, changed
        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if not (isinstance(call_id, str) and call_id.strip()):
                return item, False
            output = item.get("output")
            if not isinstance(output, str):
                output = json.dumps(output, ensure_ascii=False)
                changed = True
            minimal = {
                "type": "function_call_output",
                "call_id": call_id.strip(),
                "output": output,
            }
            if set(item.keys()) != set(minimal.keys()):
                changed = True
            return minimal, changed
        return item, False

    stripped_any = False
    normalized: list[dict[str, Any]] = []
    for entry in sanitized:
        if not isinstance(entry, dict):
            normalized.append(entry)
            continue
        stripped, changed = _strip_tool_item_extras(entry)
        if changed:
            stripped_any = True
        normalized.append(stripped)

    if removed or stripped_any or (sanitized is not items):
        if removed:
            self.logger.debug(
                "Sanitized provider input: removed %d non-replayable artifact(s).",
                removed,
            )
        if stripped_any:
            self.logger.debug("Sanitized provider input: stripped extra tool item fields.")
        body.input = normalized
