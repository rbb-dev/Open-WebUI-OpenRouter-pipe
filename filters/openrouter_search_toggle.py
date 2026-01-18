"""
title: OpenRouter Search
author: Open-WebUI-OpenRouter-pipe
author_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: openrouter_search
description: Enables OpenRouter's web-search plugin for the OpenRouter pipe and disables Open WebUI Web Search for this request (OpenRouter Search overrides Web Search).
version: 0.1.0
license: MIT
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from open_webui.env import SRC_LOG_LEVELS
except ImportError:
    SRC_LOG_LEVELS = {}

OWUI_OPENROUTER_PIPE_MARKER = "openrouter_pipe:ors_filter:v1"
_FEATURE_FLAG = "openrouter_web_search"


class Filter:
    # Toggleable filter (shows a switch in the Integrations menu).
    toggle = True

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.search.toggle")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True

    def inlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # Signal the pipe via request metadata (preferred path).
        if __metadata__ is not None and not isinstance(__metadata__, dict):
            return body

        features = body.get("features")
        if not isinstance(features, dict):
            features = {}
            body["features"] = features

        # Enforce: OpenRouter Search overrides Web Search (prevent Open WebUI native web search handler).
        features["web_search"] = False

        if isinstance(__metadata__, dict):
            meta_features = __metadata__.get("features")
            if meta_features is None:
                meta_features = {}
                __metadata__["features"] = meta_features

            # If OWUI set __metadata__["features"] to be the same dict as body["features"],
            # break the reference so we can enforce OpenRouter Search overrides Web Search
            # without losing the marker for the downstream pipe.
            if meta_features is features:
                meta_features = dict(meta_features)
                __metadata__["features"] = meta_features

            if isinstance(meta_features, dict):
                meta_features[_FEATURE_FLAG] = True

        self.log.debug("Enabled OpenRouter Search; disabled Open WebUI Web Search")
        return body
