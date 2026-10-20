from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...pipe import Pipe


@dataclass
class CommandContext:
    pipe: Pipe

    args: str

    user: dict[str, Any]

    metadata: dict[str, Any]

    event_emitter: Any = None

    async def emit_html(self, html: str) -> None:
        if self.event_emitter is not None:
            await self.event_emitter(
                {"type": "embeds", "data": {"embeds": [html]}}
            )
