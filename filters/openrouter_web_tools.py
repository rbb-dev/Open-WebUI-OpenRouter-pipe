"""
title: OR Web Tools
author: Open-WebUI-OpenRouter-pipe
author_url: https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe
id: openrouter_web_tools
description: Configures OpenRouter server tools (web search, web fetch, datetime) for the OpenRouter pipe.
version: 0.1.0
license: MIT
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, Field

from open_webui.env import SRC_LOG_LEVELS

OWUI_OPENROUTER_PIPE_MARKER = "openrouter_pipe:web_tools_filter:v1"


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )
        WEB_SEARCH_ENGINE: Literal["auto", "native", "exa", "firecrawl", "parallel"] = Field(
            default="auto",
            description="Web search backend. auto lets OpenRouter choose, native uses the model provider, others use specific engines.",
        )
        WEB_SEARCH_MAX_RESULTS: int = Field(
            default=5,
            ge=1,
            le=25,
            description="Maximum number of search results per query.",
        )
        WEB_SEARCH_MAX_TOTAL_RESULTS: int = Field(
            default=0,
            ge=0,
            description="Cap on total search results across all queries in one request. 0 means no cap.",
        )
        WEB_SEARCH_ALLOWED_DOMAINS: str = Field(
            default="",
            description="Comma-separated list of domains to restrict search results to. Empty means no restriction.",
        )
        WEB_SEARCH_EXCLUDED_DOMAINS: str = Field(
            default="",
            description="Comma-separated list of domains to exclude from search results.",
        )
        WEB_FETCH_ENGINE: Literal["auto", "native", "exa", "openrouter", "firecrawl"] = Field(
            default="auto",
            description="Web fetch backend. auto lets OpenRouter choose the best engine for each URL.",
        )
        WEB_FETCH_MAX_USES: int = Field(
            default=0,
            ge=0,
            description="Maximum number of URL fetches per request. 0 means no limit.",
        )
        WEB_FETCH_MAX_CONTENT_TOKENS: int = Field(
            default=0,
            ge=0,
            description="Maximum tokens of fetched content to return per URL. 0 means no limit.",
        )
        WEB_FETCH_ALLOWED_DOMAINS: str = Field(
            default="",
            description="Comma-separated list of domains allowed for fetching. Empty means allow all.",
        )
        WEB_FETCH_BLOCKED_DOMAINS: str = Field(
            default="",
            description="Comma-separated list of domains blocked from fetching.",
        )

    class UserValves(BaseModel):
        WEB_SEARCH: bool = Field(
            default=True,
            description="Enable OpenRouter web search for this chat.",
        )
        WEB_SEARCH_CONTEXT_SIZE: Literal["low", "medium", "high"] = Field(
            default="medium",
            description="Amount of search context to include (low saves tokens, high is more thorough).",
        )
        WEB_SEARCH_LOCATION_CITY: str = Field(
            default="",
            description="City for location-aware search results.",
        )
        WEB_SEARCH_LOCATION_REGION: str = Field(
            default="",
            description="Region/state for location-aware search results.",
        )
        WEB_SEARCH_LOCATION_COUNTRY: str = Field(
            default="",
            description="Country code (e.g. AU, US) for location-aware search results.",
        )
        WEB_SEARCH_LOCATION_TIMEZONE: str = Field(
            default="",
            description="Timezone (e.g. Australia/Sydney) for location-aware search results.",
        )
        WEB_FETCH: bool = Field(
            default=False,
            description="Enable OpenRouter web fetch (URL reading) for this chat.",
        )
        DATETIME: bool = Field(
            default=True,
            description="Enable OpenRouter datetime tool for this chat (free, no extra cost).",
        )
        DATETIME_TIMEZONE: str = Field(
            default="",
            description="Timezone for the datetime tool (e.g. Australia/Sydney). Empty uses UTC.",
        )

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.web.tools")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    @staticmethod
    def _csv_list(value: str) -> list[str]:
        if not isinstance(value, str):
            return []
        return [item.strip() for item in value.split(",") if item.strip()]

    def inlet(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any] | None = None,
        __user__: dict[str, Any] | None = None,
        __model__: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(body, dict):
            return body
        if __metadata__ is not None and not isinstance(__metadata__, dict):
            return body

        user_valves = None
        if isinstance(__user__, dict):
            user_valves = __user__.get("valves")
        if not isinstance(user_valves, BaseModel):
            user_valves = self.UserValves()

        prev_st = (__metadata__.get("openrouter_pipe") or {}).get("server_tools") if isinstance(__metadata__, dict) else None
        server_tools: dict[str, Any] = dict(prev_st) if isinstance(prev_st, dict) else {}
        suppress_owui_web_search = False

        if user_valves.WEB_SEARCH:
            ws_params: dict[str, Any] = {}
            ws_params["engine"] = self.valves.WEB_SEARCH_ENGINE
            ws_params["max_results"] = self.valves.WEB_SEARCH_MAX_RESULTS
            if self.valves.WEB_SEARCH_MAX_TOTAL_RESULTS > 0:
                ws_params["max_total_results"] = self.valves.WEB_SEARCH_MAX_TOTAL_RESULTS
            ws_params["search_context_size"] = user_valves.WEB_SEARCH_CONTEXT_SIZE
            allowed = self._csv_list(self.valves.WEB_SEARCH_ALLOWED_DOMAINS)
            if allowed:
                ws_params["allowed_domains"] = allowed
            excluded = self._csv_list(self.valves.WEB_SEARCH_EXCLUDED_DOMAINS)
            if excluded:
                ws_params["excluded_domains"] = excluded
            location: dict[str, str] = {}
            if user_valves.WEB_SEARCH_LOCATION_CITY:
                location["city"] = user_valves.WEB_SEARCH_LOCATION_CITY
            if user_valves.WEB_SEARCH_LOCATION_REGION:
                location["region"] = user_valves.WEB_SEARCH_LOCATION_REGION
            if user_valves.WEB_SEARCH_LOCATION_COUNTRY:
                location["country"] = user_valves.WEB_SEARCH_LOCATION_COUNTRY
            if user_valves.WEB_SEARCH_LOCATION_TIMEZONE:
                location["timezone"] = user_valves.WEB_SEARCH_LOCATION_TIMEZONE
            if location:
                ws_params["user_location"] = location
            server_tools["web_search"] = ws_params
            suppress_owui_web_search = True

        if user_valves.WEB_FETCH:
            wf_params: dict[str, Any] = {}
            wf_params["engine"] = self.valves.WEB_FETCH_ENGINE
            if self.valves.WEB_FETCH_MAX_USES > 0:
                wf_params["max_uses"] = self.valves.WEB_FETCH_MAX_USES
            if self.valves.WEB_FETCH_MAX_CONTENT_TOKENS > 0:
                wf_params["max_content_tokens"] = self.valves.WEB_FETCH_MAX_CONTENT_TOKENS
            allowed = self._csv_list(self.valves.WEB_FETCH_ALLOWED_DOMAINS)
            if allowed:
                wf_params["allowed_domains"] = allowed
            blocked = self._csv_list(self.valves.WEB_FETCH_BLOCKED_DOMAINS)
            if blocked:
                wf_params["blocked_domains"] = blocked
            server_tools["web_fetch"] = wf_params

        if user_valves.DATETIME:
            dt_params: dict[str, Any] = {}
            if user_valves.DATETIME_TIMEZONE:
                dt_params["timezone"] = user_valves.DATETIME_TIMEZONE
            server_tools["datetime"] = dt_params

        if server_tools and isinstance(__metadata__, dict):
            prev_pipe_meta = __metadata__.get("openrouter_pipe")
            pipe_meta = dict(prev_pipe_meta) if isinstance(prev_pipe_meta, dict) else {}
            __metadata__["openrouter_pipe"] = pipe_meta
            pipe_meta["server_tools"] = server_tools

        if suppress_owui_web_search:
            features = body.get("features")
            if not isinstance(features, dict):
                features = {}
                body["features"] = features
            features["web_search"] = False

        return body
