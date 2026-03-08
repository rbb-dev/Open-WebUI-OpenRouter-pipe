"""Command execution context for Pipe Stats Dashboard commands."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...pipe import Pipe


@dataclass
class CommandContext:
    """Context passed to every command handler."""

    pipe: Pipe
    """Full pipe reference — admin-only tool, dig into anything."""

    args: str
    """Remaining args after command prefix match."""

    user: dict[str, Any]
    """The __user__ dict from Open WebUI."""

    metadata: dict[str, Any]
    """The __metadata__ dict from Open WebUI."""

    event_emitter: Any = None
    """OWUI event emitter for rich UI embeds (HTML iframes)."""

    async def emit_html(self, html: str) -> None:
        """Emit an HTML string as a sandboxed iframe embed in the chat."""
        if self.event_emitter is not None:
            await self.event_emitter(
                {"type": "embeds", "data": {"embeds": [html]}}
            )
