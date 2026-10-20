from __future__ import annotations

import html as _html
import logging
import time
from collections.abc import Sequence
from typing import Any

logger = logging.getLogger(__name__)


def _escape_pipe(text: str) -> str:
    return text.replace("|", "\\|")


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    if not headers:
        return ""
    header_line = "| " + " | ".join(_escape_pipe(str(h)) for h in headers) + " |"
    separator = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = []
    for row in rows:
        cells = [_escape_pipe(str(c)) for c in row]
        while len(cells) < len(headers):
            cells.append("")
        body_lines.append("| " + " | ".join(cells) + " |")
    return "\n".join([header_line, separator, *body_lines])


def mermaid_pie(title: str, data: dict[str, int | float]) -> str:
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
    return f"<details>\n<summary>{_html.escape(summary)}</summary>\n\n{content}\n\n</details>"


def mask_sensitive(value: str, visible_chars: int = 4) -> str:
    if not value or not isinstance(value, str):
        return ""
    value = str(value).strip()
    if len(value) <= visible_chars:
        return "***"
    return "***" + value[-visible_chars:]


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def format_bytes(n: int) -> str:
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


def build_model_name_map(refresh: bool = False) -> dict[str, str]:
    try:
        from ...models.registry import OpenRouterModelRegistry
        cached = OpenRouterModelRegistry._name_map
    except (AttributeError, ImportError, TypeError):
        logger.debug("model id->name map build failed", exc_info=True)
        return {}
    if not refresh and cached is not None:
        return cached
    id_to_name: dict[str, str] = {}
    try:
        for m in OpenRouterModelRegistry.list_models():
            name = m.get("name", "")
            if not name:
                continue
            for key in ("id", "norm_id", "original_id"):
                mid = m.get(key)
                if mid:
                    id_to_name[mid] = name
    except (AttributeError, TypeError):
        logger.debug("model id->name map build failed", exc_info=True)
    OpenRouterModelRegistry._name_map = id_to_name
    return id_to_name


def resolve_model_name(model_id: str, name_map: dict[str, str]) -> str:
    if not model_id or model_id == "unknown":
        return "Unknown"
    name = name_map.get(model_id)
    if name:
        return name
    candidate = model_id
    while "." in candidate:
        candidate = candidate.split(".", 1)[1]
        name = name_map.get(candidate)
        if name:
            return name
        slash_form = candidate.replace(".", "/", 1)
        name = name_map.get(slash_form)
        if name:
            return name
    return model_id


def format_ago(ts: float) -> str:
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
    if dt is None:
        return "-"
    if hasattr(dt, "strftime"):
        return dt.strftime("%Y-%m-%d %H:%M")
    return str(dt)[:16]


def format_number(n: float) -> str:
    if isinstance(n, float):
        return f"{n:,.1f}"
    return f"{n:,}"


def humanize_type(raw_type: str) -> str:
    return _HUMANIZE_LABELS.get(raw_type, raw_type.replace("_", " ").title())
