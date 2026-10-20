from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import ClassVar

from .context import CommandContext

CommandHandler = Callable[[CommandContext], Awaitable[str]]


@dataclass
class CommandEntry:
    name: str
    handler: CommandHandler
    summary: str = ""
    usage: str = ""
    category: str = "General"
    aliases: list[str] = field(default_factory=list)


class CommandRegistry:
    _commands: ClassVar[dict[str, CommandEntry]] = {}

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
        stripped = text.strip()
        normalized = stripped.lower()
        if not normalized:
            return None, ""

        best_entry: CommandEntry | None = None
        best_len = 0

        for cmd_name, entry in cls._commands.items():
            if (
                normalized == cmd_name or normalized.startswith(cmd_name + " ")
            ) and len(cmd_name) > best_len:
                best_entry = entry
                best_len = len(cmd_name)

        if best_entry is None:
            return None, ""

        remaining = stripped[best_len:].strip()
        return best_entry, remaining

    @classmethod
    def all_commands(cls) -> list[CommandEntry]:
        seen: set[str] = set()
        result: list[CommandEntry] = []
        for entry in cls._commands.values():
            if entry.name not in seen:
                seen.add(entry.name)
                result.append(entry)
        return result

    @classmethod
    def _reset(cls) -> None:
        cls._commands = {}


register_command = CommandRegistry.register
