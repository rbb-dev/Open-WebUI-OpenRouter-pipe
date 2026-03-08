"""Config command — show all valves or only non-default ones."""

from __future__ import annotations

from typing import Any

from ..command_registry import register_command
from ..context import CommandContext
from ..formatters import markdown_table, mask_sensitive

# Valve field names that contain secrets and should be masked
_SENSITIVE_FIELDS = frozenset({"API_KEY", "ARTIFACT_ENCRYPTION_KEY", "SESSION_LOG_ZIP_PASSWORD"})


def _is_template_field(name: str) -> bool:
    """Return True for error-template valves whose multi-line values break markdown tables."""
    return name.endswith("_TEMPLATE")


def _format_value(field_name: str, value: Any) -> str:
    """Format a valve value for display, masking sensitive fields and sanitizing text."""
    text = str(value)
    if field_name in _SENSITIVE_FIELDS:
        if value is None or text == "":
            return "*(not set)*"
        return f"`{mask_sensitive(text)}`"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        # Collapse newlines to avoid breaking the markdown table
        text = text.replace("\n", " ").replace("\r", "")
        if len(text) > 80:
            return f"`{text[:77]}...`"
        return f"`{text}`" if text else "*(empty)*"
    return f"`{text}`"


@register_command("config", summary="Show all valve settings", category="Configuration", usage="config")
async def handle_config(ctx: CommandContext) -> str:
    """Display all valves with their current values (templates excluded)."""
    pipe = ctx.pipe
    valves = pipe.valves

    rows: list[list[str]] = []
    skipped_templates = 0
    for field_name in sorted(valves.model_fields.keys()):
        if _is_template_field(field_name):
            skipped_templates += 1
            continue
        value = getattr(valves, field_name, "")
        rows.append([f"`{field_name}`", _format_value(field_name, value)])

    header = "## Current Configuration\n\n"
    table = markdown_table(["Valve", "Value"], rows)
    footer = ""
    if skipped_templates:
        footer = f"\n\n*{skipped_templates} error template(s) hidden — edit in Valves UI.*"
    return header + table + footer


@register_command("config diff", summary="Show non-default valve settings", category="Configuration", usage="config diff")
async def handle_config_diff(ctx: CommandContext) -> str:
    """Display only valves that differ from their default values."""
    pipe = ctx.pipe
    valves = pipe.valves

    # Create a default Valves instance for comparison
    defaults = type(valves)()

    rows: list[list[str]] = []
    skipped_templates = 0
    for field_name in sorted(valves.model_fields.keys()):
        current = getattr(valves, field_name, "")
        default = getattr(defaults, field_name, "")
        if str(current) != str(default):
            if _is_template_field(field_name):
                skipped_templates += 1
                continue
            rows.append([
                f"`{field_name}`",
                _format_value(field_name, default),
                _format_value(field_name, current),
            ])

    if not rows and not skipped_templates:
        return "## Configuration Diff\n\nAll valves are at default values."

    sections = ["## Configuration Diff\n\n"]
    if rows:
        sections.append(markdown_table(["Valve", "Default", "Current"], rows))
    else:
        sections.append("All non-template valves are at default values.")
    if skipped_templates:
        sections.append(f"\n\n*{skipped_templates} error template(s) also differ — edit in Valves UI.*")
    return "".join(sections)
