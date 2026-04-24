# OpenRouter Server Tools

**Scope:** How OpenRouter server tools (Web Search, Web Fetch, Datetime, Image Generation) are configured, surfaced to users, and injected into API requests.

> **Quick navigation:** [Docs Home](README.md) · [Valves Atlas](valves_and_configuration_atlas.md) · [Web Search: OWUI vs OpenRouter](web_search_owui_vs_openrouter_search.md) · [Tooling & Integrations](tooling_and_integrations.md)

---

## What are OpenRouter server tools?

OpenRouter server tools are tools that OpenRouter executes **server-side** on behalf of the model. Unlike Open WebUI registry tools (which run locally), server tools are passed in the `tools` array of the API request and executed by OpenRouter's infrastructure when the model decides to call them.

Available server tools:

| Tool | Purpose | Cost |
| --- | --- | --- |
| `web_search` | Search the web and return results to the model | Per-query pricing (varies by engine) |
| `web_fetch` | Fetch and read the content of a URL | Per-fetch pricing (varies by engine) |
| `datetime` | Return the current date and time | Free |
| `image_generation` | Generate images from text prompts | Per-image pricing (varies by model) |

The model decides **when** to call these tools based on the conversation context. The pipe does not invoke them directly; it includes the tool definitions in the outgoing request and OpenRouter handles execution.

---

## Two companion filters

Server tools are configured through **companion filter functions** that the pipe auto-installs into Open WebUI. Each filter runs as an inlet (pre-processing step) that writes tool configuration into request metadata, which the pipe then reads and injects into the API request.

### OpenRouter Web Tools filter

Bundles three tools into a single toggleable filter:
- **Web Search** (user valve, default: on)
- **Web Fetch** (user valve, default: off)
- **Datetime** (user valve, default: on)

Users see a single "OpenRouter Web Tools" switch in the Integrations menu. Individual tools are toggled via the filter's user valves (the knobs/settings UI for the filter).

### OpenRouter Image Generation filter

A separate filter for image generation:
- **Image Generation** (user valve, default: off)

Separated from Web Tools because image generation has distinct cost implications and a different set of configuration options (model selection, quality, size, format).

---

## Pipe admin valves (server tool gates)

These valves on the pipe control which server tools are available and how the companion filters are managed.

### Tool gate valves

Each tool has an enable gate. When a gate is disabled, the corresponding tool's user valves are excluded from the generated filter source entirely (users cannot see or enable the tool).

| Valve | Type | Default | Purpose |
| --- | --- | --- | --- |
| `ENABLE_WEB_SEARCH` | `bool` | `True` | Enable the OpenRouter Web Search server tool. When disabled, web search toggles are hidden from users. |
| `ENABLE_WEB_FETCH` | `bool` | `True` | Enable the OpenRouter Web Fetch server tool. When disabled, web fetch toggles are hidden from users. |
| `ENABLE_DATETIME` | `bool` | `True` | Enable the OpenRouter Datetime server tool (free, no additional cost). When disabled, datetime toggles are hidden from users. |
| `ENABLE_IMAGE_GENERATION` | `bool` | `True` | Enable the OpenRouter Image Generation server tool. When disabled, image generation toggles are hidden from users. |

### Filter lifecycle valves

These control auto-installation, auto-attachment, and default-on behavior for the companion filters.

| Valve | Type | Default | Purpose |
| --- | --- | --- | --- |
| `AUTO_INSTALL_WEB_TOOLS_FILTER` | `bool` | `True` | Automatically install/update the OpenRouter Web Tools filter function in Open WebUI. |
| `AUTO_ATTACH_WEB_TOOLS_FILTER` | `bool` | `True` | Automatically attach the OpenRouter Web Tools filter to all pipe models (so the toggle appears in the Integrations menu). |
| `AUTO_DEFAULT_WEB_TOOLS_FILTER` | `bool` | `True` | Automatically mark the OpenRouter Web Tools filter as a Default Filter on models (enabled by default, users can turn off per chat). |
| `AUTO_INSTALL_IMAGE_GEN_FILTER` | `bool` | `True` | Automatically install/update the OpenRouter Image Generation filter function in Open WebUI. |
| `AUTO_ATTACH_IMAGE_GEN_FILTER` | `bool` | `True` | Automatically attach the OpenRouter Image Generation filter to all pipe models. |

---

## Filter admin valves

These are configured on the companion filter functions themselves (Open WebUI Admin > Functions > filter > Valves), not on the pipe.

### OpenRouter Web Tools filter valves (admin)

| Valve | Type | Default | Purpose |
| --- | --- | --- | --- |
| `priority` | `int` | `0` | Priority level for the filter operations. |
| `WEB_SEARCH_ENGINE` | `Literal["auto","native","exa","firecrawl","parallel"]` | `auto` | Web search backend. `auto` lets OpenRouter choose, `native` uses the model provider, others use specific engines. |
| `WEB_SEARCH_MAX_RESULTS` | `int` | `5` | Maximum number of search results per query (1-25). |
| `WEB_SEARCH_MAX_TOTAL_RESULTS` | `int` | `0` | Cap on total search results across all queries in one request. 0 means no cap. |
| `WEB_SEARCH_ALLOWED_DOMAINS` | `str` | `""` | Comma-separated list of domains to restrict search results to. Empty means no restriction. |
| `WEB_SEARCH_EXCLUDED_DOMAINS` | `str` | `""` | Comma-separated list of domains to exclude from search results. |
| `WEB_FETCH_ENGINE` | `Literal["auto","native","exa","openrouter","firecrawl"]` | `auto` | Web fetch backend. `auto` lets OpenRouter choose the best engine for each URL. |
| `WEB_FETCH_MAX_USES` | `int` | `0` | Maximum number of URL fetches per request. 0 means no limit. |
| `WEB_FETCH_MAX_CONTENT_TOKENS` | `int` | `0` | Maximum tokens of fetched content to return per URL. 0 means no limit. |
| `WEB_FETCH_ALLOWED_DOMAINS` | `str` | `""` | Comma-separated list of domains allowed for fetching. Empty means allow all. |
| `WEB_FETCH_BLOCKED_DOMAINS` | `str` | `""` | Comma-separated list of domains blocked from fetching. |

### OpenRouter Image Generation filter valves (admin)

| Valve | Type | Default | Purpose |
| --- | --- | --- | --- |
| `priority` | `int` | `0` | Priority level for the filter operations. |
| `IMAGE_GENERATION_MODEL` | `str` | `openai/gpt-image-1` | OpenRouter model ID for image generation. Controls pricing and capabilities. |
| `IMAGE_GENERATION_MODERATION` | `Literal["auto","low"]` | `auto` | Content moderation level. `auto` = standard. `low` = reduced filtering. |

---

## Filter user valves

These appear in the filter's user-facing knobs UI and control per-user, per-chat behavior.

### OpenRouter Web Tools filter user valves

| Valve | Type | Default | Purpose |
| --- | --- | --- | --- |
| `WEB_SEARCH` | `bool` | `True` | Enable OpenRouter web search for this chat. |
| `WEB_SEARCH_CONTEXT_SIZE` | `Literal["low","medium","high"]` | `medium` | Amount of search context to include (low saves tokens, high is more thorough). |
| `WEB_SEARCH_LOCATION_CITY` | `str` | `""` | City for location-aware search results. |
| `WEB_SEARCH_LOCATION_REGION` | `str` | `""` | Region/state for location-aware search results. |
| `WEB_SEARCH_LOCATION_COUNTRY` | `str` | `""` | Country code (e.g. AU, US) for location-aware search results. |
| `WEB_SEARCH_LOCATION_TIMEZONE` | `str` | `""` | Timezone (e.g. Australia/Sydney) for location-aware search results. |
| `WEB_FETCH` | `bool` | `False` | Enable OpenRouter web fetch (URL reading) for this chat. |
| `DATETIME` | `bool` | `True` | Enable OpenRouter datetime tool for this chat (free, no extra cost). |
| `DATETIME_TIMEZONE` | `str` | `""` | Timezone for the datetime tool (e.g. Australia/Sydney). Empty uses UTC. |

### OpenRouter Image Generation filter user valves

| Valve | Type | Default | Purpose |
| --- | --- | --- | --- |
| `IMAGE_GENERATION` | `bool` | `False` | Let the model generate images from text prompts. Incurs additional cost per image. |
| `IMAGE_QUALITY` | `Literal["","low","medium","high"]` | `""` | Quality level for generated images. Empty = model default. |
| `IMAGE_SIZE` | `Literal["","1024x1024","1536x1024","1024x1536","512x512"]` | `""` | Image dimensions. Empty = model default. |
| `IMAGE_ASPECT_RATIO` | `Literal["","1:1","16:9","4:3","3:2"]` | `""` | Aspect ratio for generated images. Empty = model default. |
| `IMAGE_BACKGROUND` | `Literal["","transparent","opaque"]` | `""` | `transparent` removes background (PNG only). Empty = model default. |
| `IMAGE_OUTPUT_FORMAT` | `Literal["","png","jpeg","webp"]` | `""` | Output format. `png` supports transparency. Empty = model default. |
| `IMAGE_OUTPUT_COMPRESSION` | `int` | `0` | Compression level for jpeg/webp (0-100). 0 = model default. |

---

## How it works

### Data flow

```
User chat message
    |
    v
[Filter inlet] -- reads user valves, writes server_tools dict to __metadata__["openrouter_pipe"]["server_tools"]
    |
    v
[Pipe orchestrator] -- reads __metadata__["openrouter_pipe"]["server_tools"], injects into API request tools array
    |
    v
[OpenRouter API] -- model calls tools as needed, OpenRouter executes them server-side
    |
    v
[Response] -- tool results are inline in the model's response
```

### Filter side (inlet)

Each filter's `inlet` method:

1. Reads user valves to determine which tools the user has enabled.
2. Reads admin valves for engine/limit configuration.
3. Builds a `server_tools` dict mapping tool names to their parameters.
4. Writes the dict into `__metadata__["openrouter_pipe"]["server_tools"]`.
5. (Web Tools filter only) When web search is enabled, suppresses Open WebUI's native web search by setting `body["features"]["web_search"] = False` to prevent double-searching.

The Image Generation filter merges into any existing `server_tools` dict (so both filters can run on the same request without overwriting each other).

### Pipe side (orchestrator)

The pipe's request orchestrator:

1. Reads `__metadata__["openrouter_pipe"]["server_tools"]`.
2. For each tool in the dict, builds a tool spec (`{"type": "<tool_name>", ...params}`) and appends it to the `tools` array in the outgoing API request body.
3. The tools array is sent alongside any Open WebUI registry tools or Direct Tool Server tools.

---

## Migration from old OpenRouter Search filter

Previous versions of this pipe used a single "OpenRouter Search" filter (marker: `openrouter_pipe:ors_filter:*`) that injected web search as a `plugins` entry. This has been replaced by the Web Tools filter, which uses the `tools` array instead.

On startup, the pipe automatically detects and **disables** the old OpenRouter Search filter if it still exists in the Functions DB. No manual cleanup is required.

The old pipe valves (`AUTO_ATTACH_ORS_FILTER`, `AUTO_INSTALL_ORS_FILTER`, `AUTO_DEFAULT_OPENROUTER_SEARCH_FILTER`, `WEB_SEARCH_MAX_RESULTS`) have been replaced by the new server tool valves documented above.

---

## Per-model overrides (Advanced Parameters)

Two per-model custom parameters control Web Tools filter attachment on a per-model basis. These are set in Open WebUI's model Advanced Parameters:

| Parameter | Effect |
| --- | --- |
| `disable_openrouter_search_auto_attach` | Prevents auto-attaching the Web Tools filter to this model (the toggle will not appear in Integrations). |
| `disable_openrouter_search_default_on` | Prevents auto-enabling the Web Tools filter by default for this model (the toggle appears but starts off). |

These parameters are respected even when the global `AUTO_ATTACH_WEB_TOOLS_FILTER` and `AUTO_DEFAULT_WEB_TOOLS_FILTER` valves are enabled.

See: [OpenRouter Integrations & Telemetry](openrouter_integrations_and_telemetry.md) for the full list of per-model custom parameters.

---

## Recommended operator settings

### All server tools available (current defaults)

All `ENABLE_*` gates are `True`, all `AUTO_INSTALL_*` and `AUTO_ATTACH_*` valves are `True`, `AUTO_DEFAULT_WEB_TOOLS_FILTER` is `True`. Users see Web Search and Datetime enabled by default, with Web Fetch and Image Generation available as opt-in.

### Disable image generation

Set `ENABLE_IMAGE_GENERATION=False`. The Image Generation filter will not be generated or installed. Alternatively, set `AUTO_INSTALL_IMAGE_GEN_FILTER=False` to keep the gate open but skip auto-installation.

### Web search opt-in (lower cost)

Set `AUTO_DEFAULT_WEB_TOOLS_FILTER=False`. The Web Tools toggle remains available on models, but will not be enabled by default. Users must manually enable it per chat.

### Restrict search domains

Configure the Web Tools filter's admin valve `WEB_SEARCH_ALLOWED_DOMAINS` with a comma-separated domain list (e.g. `docs.python.org, stackoverflow.com`). Search results will be restricted to those domains.
