"""Help command — list all commands or show details for a specific command."""

from __future__ import annotations

from ..command_registry import CommandRegistry, register_command
from ..context import CommandContext
from ..formatters import markdown_table


@register_command("help", summary="Show available commands", category="General", usage="help [command]")
async def handle_help(ctx: CommandContext) -> str:
    """Show all commands, or detailed help for a specific command."""
    if ctx.args:
        # Specific command help
        entry, _ = CommandRegistry.resolve(ctx.args)
        if entry is None:
            safe_args = ctx.args.replace("`", "'")
            return f"Unknown command: `{safe_args}`\n\nType `help` for available commands."
        lines = [
            f"## `{entry.name}`",
            "",
            f"**Summary:** {entry.summary}",
        ]
        if entry.usage:
            lines.append(f"**Usage:** `{entry.usage}`")
        if entry.aliases:
            lines.append(f"**Aliases:** {', '.join(f'`{a}`' for a in entry.aliases)}")
        return "\n".join(lines)

    # General help — single flat table, sorted by name
    commands = CommandRegistry.all_commands()
    if not commands:
        return "No commands registered."

    rows = [[f"`{e.name}`", e.summary] for e in sorted(commands, key=lambda e: e.name)]

    sections: list[str] = [
        "## Pipe Stats Dashboard\n",
        markdown_table(["Command", "Description"], rows),
        "",
        "Type `help <command>` for detailed usage.",
    ]
    return "\n".join(sections)
