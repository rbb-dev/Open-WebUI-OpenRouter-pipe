from __future__ import annotations

from typing import Any

import aiohttp

from ..core.config import _select_openrouter_http_referer
from .image_client import OpenRouterImageClient
from .video_client import OpenRouterVideoClient


def _build_catalog_client(
    client_cls: type[OpenRouterVideoClient | OpenRouterImageClient],
    session: aiohttp.ClientSession,
    *,
    valves: Any,
    api_key: str,
    logger: Any,
) -> Any:
    return client_cls(
        session,
        base_url=valves.BASE_URL,
        api_key=api_key,
        logger=logger,
        http_referer=_select_openrouter_http_referer(valves),
        valves=valves,
    )
