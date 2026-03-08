"""Command registry with decorator-based registration and longest-prefix matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

from .context import CommandContext

# Type alias for command handlers
CommandHandler = Callable[[CommandContext], Awaitable[str]]


@dataclass
class CommandEntry:
    """A registered command."""

    name: str
    handler: CommandHandler
    summary: str = ""
    usage: str = ""
    category: str = "General"
    aliases: list[str] = field(default_factory=list)


class CommandRegistry:
    """Registry of Pipe Stats Dashboard commands with longest-prefix matching."""

    _commands: dict[str, CommandEntry] = {}

    @classmethod
    def register(
        cls,
        name: str,
        *,
        summary: str = "",
        usage: str = "",
        category: str = "General",
        aliases: list[str] | None = None,
    ) -> Callable[[CommandHandler], CommandHandler]:
        """Decorator to register a command handler."""

        def decorator(handler: CommandHandler) -> CommandHandler:
            entry = CommandEntry(
                name=name,
                handler=handler,
                summary=summary,
                usage=usage,
                category=category,
                aliases=aliases or [],
            )
            cls._commands[name.lower()] = entry
            for alias in entry.aliases:
                cls._commands[alias.lower()] = entry
            return handler

        return decorator

    @classmethod
    def resolve(cls, text: str) -> tuple[CommandEntry | None, str]:
        """Find the best matching command using longest-prefix matching.

        Command names are matched case-insensitively, but the remaining
        arguments preserve their original casing.

        Returns ``(entry, remaining_args)`` or ``(None, "")`` if no match.
        """
        stripped = text.strip()
        normalized = stripped.lower()
        if not normalized:
            return None, ""

        # Try longest prefix first
        best_entry: CommandEntry | None = None
        best_len = 0

        for cmd_name, entry in cls._commands.items():
            if normalized == cmd_name or normalized.startswith(cmd_name + " "):
                if len(cmd_name) > best_len:
                    best_entry = entry
                    best_len = len(cmd_name)

        if best_entry is None:
            return None, ""

        # Preserve original casing in remaining args
        remaining = stripped[best_len:].strip()
        return best_entry, remaining

    @classmethod
    def all_commands(cls) -> list[CommandEntry]:
        """Return deduplicated list of all commands (excludes alias duplicates)."""
        seen: set[str] = set()
        result: list[CommandEntry] = []
        for entry in cls._commands.values():
            if entry.name not in seen:
                seen.add(entry.name)
                result.append(entry)
        return result

    @classmethod
    def _reset(cls) -> None:
        """Reset registry — for testing ONLY.

        WARNING: This wipes ALL commands including built-ins (help, health,
        config, storage). Only safe in test fixtures that restore the
        original state afterward.
        """
        cls._commands = {}


# Convenience alias — preserves the full type signature for IDE autocomplete
register_command = CommandRegistry.register
