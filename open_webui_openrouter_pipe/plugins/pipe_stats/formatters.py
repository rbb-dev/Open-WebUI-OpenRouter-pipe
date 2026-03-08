"""Markdown, Mermaid, and display formatting utilities for Pipe Stats Dashboard output."""

from __future__ import annotations

import html as _html
import time
from typing import Any, Sequence


def _escape_pipe(text: str) -> str:
    """Escape pipe characters so they don't break markdown table cells."""
    return text.replace("|", "\\|")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Build a markdown table string.

    Pipe characters in cell values are escaped to prevent breaking table structure.

    >>> markdown_table(["A", "B"], [["1", "2"]])
    '| A | B |\\n| --- | --- |\\n| 1 | 2 |'
    """
    if not headers:
        return ""
    header_line = "| " + " | ".join(_escape_pipe(str(h)) for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = []
    for row in rows:
        cells = [_escape_pipe(str(c)) for c in row]
        # Pad if row is shorter than headers
        while len(cells) < len(headers):
            cells.append("")
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_line, separator, *body_lines])


def mermaid_pie(title: str, data: dict[str, int | float]) -> str:
    """Build a Mermaid pie chart fenced code block."""
    if not data:
        return ""
    lines = [f'```mermaid\npie title {title}']
    for label, value in data.items():
        lines.append(f'    "{label}" : {value}')
    lines.append("```")
    return "\n".join(lines)


def mermaid_bar(
    title: str,
    x_label: str,
    y_label: str,
    categories: Sequence[str],
    values: Sequence[int | float],
) -> str:
    """Build a Mermaid xychart-beta bar chart fenced code block."""
    if not categories or not values:
        return ""
    cats = "[" + ", ".join(f'"{c}"' for c in categories) + "]"
    vals = "[" + ", ".join(str(v) for v in values) + "]"
    return (
        f"```mermaid\nxychart-beta\n"
        f'    title "{title}"\n'
        f'    x-axis {x_label} {cats}\n'
        f'    y-axis "{y_label}"\n'
        f"    bar {vals}\n"
        f"```"
    )


def collapsible(summary: str, content: str) -> str:
    """Wrap content in an HTML ``<details>`` collapsible block.

    The summary is HTML-escaped to prevent injection.
    """
    return f"<details>\n<summary>{_html.escape(summary)}</summary>\n\n{content}\n\n</details>"


def mask_sensitive(value: str, visible_chars: int = 4) -> str:
    """Mask a sensitive value, showing only the last N characters.

    >>> mask_sensitive("sk-or-v1-abc123xyz")
    '***3xyz'
    """
    if not value or not isinstance(value, str):
        return ""
    value = str(value).strip()
    if len(value) <= visible_chars:
        return "***"
    return "***" + value[-visible_chars:]


def format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration.

    >>> format_duration(5.2)
    '5.2s'
    >>> format_duration(125)
    '2.1m'
    >>> format_duration(7200)
    '2.0h'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def format_bytes(n: int) -> str:
    """Format a byte count into a human-readable string.

    >>> format_bytes(0)
    '0 B'
    >>> format_bytes(1023)
    '1023 B'
    >>> format_bytes(1536)
    '1.5 KB'
    >>> format_bytes(2_621_440)
    '2.5 MB'
    >>> format_bytes(5_368_709_120)
    '5.0 GB'
    """
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.1f} GB"


_HUMANIZE_LABELS: dict[str, str] = {
    "function_call": "Function Call",
    "function_call_output": "Function Call Output",
    "reasoning": "Reasoning",
    "web_search_call": "Web Search",
    "file_search_call": "File Search",
    "image_generation_call": "Image Generation",
    "local_shell_call": "Shell Command",
}


def build_model_name_map() -> dict[str, str]:
    """Build a mapping from any model ID form to its display name."""
    id_to_name: dict[str, str] = {}
    try:
        from ...models.registry import OpenRouterModelRegistry
        for m in OpenRouterModelRegistry.list_models():
            name = m.get("name", "")
            if not name:
                continue
            for key in ("id", "norm_id", "original_id"):
                mid = m.get(key)
                if mid:
                    id_to_name[mid] = name
    except Exception:
        pass
    return id_to_name


def resolve_model_name(model_id: str, name_map: dict[str, str]) -> str:
    """Resolve a model ID to its display name, falling back to the raw ID.

    DB stores model IDs in OWUI's prefixed format: ``{pipe_id}.{author}.{model}``.
    The name map contains keys like ``author.model`` (norm_id) and
    ``author/model`` (original_id). We try progressively stripping prefixes
    until we find a match.
    """
    if not model_id or model_id == "unknown":
        return "Unknown"
    # Direct match
    name = name_map.get(model_id)
    if name:
        return name
    # Strip known pipe prefixes (e.g. "open_webui_openrouter_pipe.openai.gpt-5.2")
    # by progressively removing the first dot-segment until we match.
    candidate = model_id
    while "." in candidate:
        candidate = candidate.split(".", 1)[1]
        name = name_map.get(candidate)
        if name:
            return name
        # Also try slash form (author/model)
        slash_form = candidate.replace(".", "/", 1)
        name = name_map.get(slash_form)
        if name:
            return name
    return model_id


def format_ago(ts: float) -> str:
    """Human-readable 'time ago' from a Unix timestamp.

    >>> format_ago(0)
    'never'
    """
    if ts <= 0:
        return "never"
    delta = time.time() - ts
    if delta < 0:
        return "just now"
    if delta < 60:
        return f"{int(delta)}s ago"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{delta / 3600:.1f}h ago"
    return f"{delta / 86400:.1f}d ago"


def format_datetime(dt: Any) -> str:
    """Format a datetime to a short display string.

    >>> format_datetime(None)
    '-'
    """
    if dt is None:
        return "-"
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m-%d %H:%M")
    return str(dt)[:16]


def format_number(n: int | float) -> str:
    """Format a number with comma separators.

    >>> format_number(1234567)
    '1,234,567'
    """
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


def humanize_type(raw_type: str) -> str:
    """Convert a snake_case artifact type to a human-readable label.

    >>> humanize_type("function_call_output")
    'Function Call Output'
    >>> humanize_type("reasoning")
    'Reasoning'
    >>> humanize_type("image_generation_call")
    'Image Generation'
    """
    return _HUMANIZE_LABELS.get(raw_type, raw_type.replace("_", " ").title())
