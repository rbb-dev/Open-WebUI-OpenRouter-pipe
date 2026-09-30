# OpenRouter Server Tools

**Scope:** How OpenRouter server tools (Web Search, Web Fetch, Datetime, Advisor, Subagent, Model Search, Image Generation) are configured, surfaced to users, and injected into API requests.

> **Quick navigation:** [Docs Home](README.md) · [Valves Atlas](valves_and_configuration_atlas.md) · [Web Search: OWUI vs OpenRouter](web_search_owui_vs_openrouter_search.md) · [Tooling & Integrations](tooling_and_integrations.md)

---

## What are OpenRouter server tools?

OpenRouter server tools are tools that OpenRouter executes **server-side** on behalf of the model. Unlike Open WebUI registry tools (which run locally), server tools are passed in the `tools` array of the API request and executed by OpenRouter's infrastructure when the model decides to call them. One request carries one entry per `openrouter:*` type, whichever of the two places it was written in.

Available server tools:

| Tool | Purpose |
| --- | --- |
| `web_search` | Search the web and return results to the model |
| `web_fetch` | Fetch and read the content of a URL |
| `datetime` | Return the current date and time |
| `image_generation` | Generate images from text prompts |
| `advisor` | Consult a higher-intelligence model mid-generation |
| `subagent` | Delegate a self-contained task to a worker model an admin chooses |
| `chat_search_models` | Let the model search the OpenRouter model catalog |

> Advisor and subagent each spawn an **additional model call**; both default **off** per chat and are gated by `ENABLE_ADVISOR` / `ENABLE_SUBAGENT`. The `SERVER_TOOLS_MAX_COST_USD` filter valve bounds the server-tool agent loop via the OpenRouter `stop_server_tools_when` request parameter (overrides `max_tool_calls`); it is sent only while an `openrouter:` server tool is on the request, and dropped when every tool it bounds has been switched off or stripped. It is the cost bound, not the only one: OpenRouter treats `stop_server_tools_when` as an override, so a request that also carries a caller-set top-level `max_tool_calls` is stopped by whichever fires first, and at the default `0.0` — no cap, field absent — that step count is the only bound on the loop. The cap is per request: one request is one call the pipe makes, and an internal Fusion turn is one call per panel member plus the judge and the synthesis, each sent the whole cap, so the ceiling for that turn is that multiple. What a model charges is on OpenRouter's pricing page.

The model decides **when** to call these tools based on the conversation context. The pipe does not invoke them directly; it includes the tool definitions in the outgoing request and OpenRouter handles execution.

---

## Two companion filters

Server tools are configured through **companion filter functions** that the pipe auto-installs into Open WebUI. Each filter runs as an inlet (pre-processing step) that writes tool configuration into request metadata, which the pipe then reads and injects into the API request.

### OpenRouter Web Tools filter

Bundles six tools into a single toggleable filter:
- **Web Search** (user valve, default: on)
- **Web Fetch** (user valve, default: off)
- **Datetime** (user valve, default: on)
- **Advisor** (user valve, default: off)
- **Subagent** (user valve, default: off)
- **Model Search** (user valve, default: off)

Users see a single "OpenRouter Web Tools" switch in the Integrations menu. Individual tools are toggled via the filter's user valves (the knobs/settings UI for the filter).

### OpenRouter Image Generation filter

A separate filter for image generation:
- **Image Generation** (user valve, default: off)

Separated from Web Tools because image generation has distinct cost implications and a different set of configuration options (model selection, quality, size, format).

---

## Pipe admin valves (server tool gates)

These valves on the pipe control which server tools are available and how the companion filters are managed.

### Tool gate valves

Each tool has an enable gate. When a gate is disabled, the corresponding tool's user valves are excluded from the generated filter source entirely (users cannot see or enable the tool). The gate is also re-checked on every request: while it is off, the pipe never sends that tool, whether a filter writes it, the request itself lists it, an internal-Fusion member re-runs a filter inlet, or the request is a housekeeping task turn (a title generation, a tag pass, any other Task Model call). Every Web Tools filter this pipe maintains (one it installed, or one carrying no install record) that still offers a switched-off tool is rewritten without it, whatever its id and whether it is on or off (see below).

| Valve | Type | Default | Purpose |
| --- | --- | --- | --- |
| `ENABLE_WEB_SEARCH` | `bool` | `True` | Enable the OpenRouter Web Search server tool. When disabled, web search toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it; every Web Tools filter this pipe maintains (one it installed, or one carrying no install record) that still offers it is rewritten without it, whether it is on or off, at the next model-list refresh, or after the first message that still asks for it. On a Fusion model the panel is never attached, so there is no per-chat switch there: the setting stored for you governs what the internal Fusion panel may use. |
| `ENABLE_WEB_FETCH` | `bool` | `True` | Enable the OpenRouter Web Fetch server tool. When disabled, web fetch toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it; every Web Tools filter this pipe maintains (one it installed, or one carrying no install record) that still offers it is rewritten without it, whether it is on or off, at the next model-list refresh, or after the first message that still asks for it.  |
| `ENABLE_DATETIME` | `bool` | `True` | Enable the OpenRouter Datetime server tool (free, no additional cost). When disabled, datetime toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it; every Web Tools filter this pipe maintains (one it installed, or one carrying no install record) that still offers it is rewritten without it, whether it is on or off, at the next model-list refresh, or after the first message that still asks for it.  |
| `ENABLE_ADVISOR` | `bool` | `True` | Enable the OpenRouter Advisor server tool (consult a higher-intelligence model mid-generation). When disabled, advisor toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it; every Web Tools filter this pipe maintains (one it installed, or one carrying no install record) that still offers it is rewritten without it, whether it is on or off, at the next model-list refresh, or after the first message that still asks for it.  |
| `ENABLE_SUBAGENT` | `bool` | `True` | Enable the OpenRouter Subagent server tool (delegate tasks to a worker model an admin chooses). When disabled, subagent toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it; every Web Tools filter this pipe maintains (one it installed, or one carrying no install record) that still offers it is rewritten without it, whether it is on or off, at the next model-list refresh, or after the first message that still asks for it.  |
| `ENABLE_SEARCH_MODELS` | `bool` | `True` | Enable the OpenRouter model-search server tool (let the model search the OpenRouter catalog). When disabled, model-search toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it; every Web Tools filter this pipe maintains (one it installed, or one carrying no install record) that still offers it is rewritten without it, whether it is on or off, at the next model-list refresh, or after the first message that still asks for it.  |
| `ENABLE_IMAGE_GENERATION` | `bool` | `True` | Enable the OpenRouter Image Generation server tool. When disabled, image generation toggles are hidden from users, and the pipe stops sending the tool at once, even while an out-of-date filter or the request itself still asks for it. |

### Filter lifecycle valves

These control auto-installation, auto-attachment, and default-on behavior for the companion filters.

| Valve | Type | Default | Purpose |
| --- | --- | --- | --- |
| `AUTO_INSTALL_WEB_TOOLS_FILTER` | `bool` | `True` | Automatically install/update the OpenRouter Web Tools filter function in Open WebUI. When off, the pipe neither installs nor updates it, except that a web tool switched off on the pipe is taken out of every Web Tools filter this pipe maintains (one it installed, or one carrying no install record): that filter's code is replaced with the pipe's current version for the tools it still offers (hand edits in it are lost), and a warning is logged. Switching the tool back on does not add it back. | With every web tool off, every Web Tools filter is switched off that this pipe installed or that carries no install record, and one you switch off yourself there stays off until you switch it on again.
| `AUTO_ATTACH_WEB_TOOLS_FILTER` | `bool` | `True` | Automatically attach the OpenRouter Web Tools per-chat switch to every pipe model that is not an image-output, a video-generation or a Fusion model (so the toggle appears in the Integrations menu). On a Fusion model the filter is never auto-attached, but it is still the source: the internal engine gives each member the `openrouter:*` tools from the admin's `ENABLE_*` valves **and** from this filter's stored per-user toggles for the chatting user, so a per-chat toggle set on another chat governs the panel. |
| `AUTO_DEFAULT_WEB_TOOLS_FILTER` | `bool` | `False` | When enabled, marks the OpenRouter Web Tools filter as a Default Filter on models (pre-enabled per chat; users can still turn it off). |
| `AUTO_INSTALL_IMAGE_GEN_FILTER` | `bool` | `True` | Automatically install/update the OpenRouter Image Generation filter function in Open WebUI. |
| `AUTO_ATTACH_IMAGE_GEN_FILTER` | `bool` | `True` | Automatically attach the OpenRouter Image Generation filter to every pipe model that can send the tool: not a model whose catalogue entry rules tool use out, not a picture-only model, not a video model, not the hosted Fusion model. A model that stops qualifying loses the switch at the next refresh. |

---

## Filter admin valves

These are configured on the companion filter functions themselves (Open WebUI Admin > Functions > filter > Valves), not on the pipe.

### OpenRouter Web Tools filter valves (admin)

| Valve | Type | Default | Purpose |
| --- | --- | --- | --- |
| `priority` | `int` | `0` | Priority level for the filter operations. |
| `WEB_SEARCH_ENGINE` | `Literal["auto","native","exa","firecrawl","parallel","perplexity"]` | `auto` | Web search backend. `auto` lets OpenRouter choose, `native` uses the model provider, others use specific engines. |
| `WEB_SEARCH_MAX_RESULTS` | `int` | `5` | Maximum number of search results per query (1-25). |
| `WEB_SEARCH_MAX_TOTAL_RESULTS` | `int` | `0` | Cap on total search results across all queries in one request. 0 means no cap. |
| `WEB_SEARCH_MAX_CHARACTERS` | `int` | `0` | Max characters of content per search result (1-100000). 0 means no cap. Takes precedence over context size when set. |
| `WEB_SEARCH_ALLOWED_DOMAINS` | `str` | `""` | Comma-separated list of domains to restrict search results to. Empty means no restriction. |
| `WEB_SEARCH_EXCLUDED_DOMAINS` | `str` | `""` | Comma-separated list of domains to exclude from search results. |
| `WEB_FETCH_ENGINE` | `Literal["auto","native","exa","openrouter","firecrawl","parallel"]` | `auto` | Web fetch backend. `auto` lets OpenRouter choose the best engine for each URL. |
| `WEB_FETCH_MAX_USES` | `int` | `0` | Maximum number of URL fetches per request. 0 means no limit. |
| `WEB_FETCH_MAX_CONTENT_TOKENS` | `int` | `0` | Maximum tokens of fetched content to return per URL. 0 means no limit. |
| `WEB_FETCH_ALLOWED_DOMAINS` | `str` | `""` | Comma-separated list of domains allowed for fetching. Empty means allow all. |
| `WEB_FETCH_BLOCKED_DOMAINS` | `str` | `""` | Comma-separated list of domains blocked from fetching. |
| `ADVISOR_MODEL` | `str` | `""` | Advisor model to consult (any OpenRouter model). Empty uses the chat's own model. |
| `SUBAGENT_MODEL` | `str` | `""` | Worker model for delegated subagent tasks. Empty uses the chat's own model. |
| `SERVER_TOOLS_MAX_COST_USD` | `float` | `0.0` | Cap cumulative server-tool loop cost per request in USD (sets OpenRouter `stop_server_tools_when`). 0 means no cap — and then a caller-set top-level `max_tool_calls`, which the pipe forwards unchanged on both endpoints, is the fallback step bound, because OpenRouter's `stop_server_tools_when` overrides `max_tool_calls` rather than combining with it. The cap is sent only while an `openrouter:` server tool is on the request. One request is one call the pipe makes, and an internal Fusion turn is one call per panel member plus the judge and the synthesis, each sent the whole cap, so the ceiling for the turn is that multiple. |

### OpenRouter Image Generation filter valves (admin)

| Valve | Type | Default | Purpose |
| --- | --- | --- | --- |
| `priority` | `int` | `0` | Priority level for the filter operations. |
| `IMAGE_GENERATION_MODEL` | `str` | `openai/gpt-5-image-mini` | Which OpenRouter model draws the picture. The pipe's default is `openai/gpt-5-image-mini`; OpenRouter documents `openai/gpt-5-image` as its own, so the two differ. The default was chosen because it costs less per image than OpenRouter's documented default, and a value an admin stores in this valve takes its place. Clearing the box, or filling it with only spaces, counts as unset, and the pipe's own default draws the picture. Its own published contract is what the user valves below are built from, so changing it changes them on the next catalog refresh. Its description names the model in force and, when no settings are offered, says which of the four reasons applies. A refresh whose valve read **fails** leaves the installed filter exactly as it is, rather than rebuilding it for the default model, until a later refresh succeeds. Each distinct cause is warned about once at WARNING and then dropped to DEBUG for the life of the worker, latched separately for a read that raised and for a read that returned no result; the message reads "Could not read the image generation filter's selected model, so the installed filter is left as it is rather than rebuilt for the default model". The model-ruled-out warning is separate and latched per model over a sliding 300-model window: once per model inside the window, and again at WARNING for a model that falls out of it, so a model that keeps being ruled out stays visible rather than going quiet forever. |
| `IMAGE_GENERATION_MODERATION` | `Literal["auto","low"]` | `auto` | How strictly the company running the model screens what it will draw. |

A stored admin value from an older version of the pipe that no longer fits its field falls
back to that field's default rather than failing the request, and every other stored value
on the same row is kept. Open WebUI builds this class straight from the stored row with no
error handling around it, so without that a single retired moderation option or a
non-string model — both of which a refresh can produce, because the option list and the
default model both move — would abort the chat rather than lose one field. The stored value
is repaired when the class is built, not written back: the row keeps what was saved until
the form is next saved, which is the same treatment the user valves get.

---

## Filter user valves

These appear in the filter's user-facing knobs UI and control per-user, per-chat behavior.

### OpenRouter Web Tools filter user valves

| Valve | Type | Default | Purpose |
| --- | --- | --- | --- |
| `WEB_SEARCH` | `bool` | `True` | Enable OpenRouter web search for you. |
| `WEB_SEARCH_CONTEXT_SIZE` | `Literal["low","medium","high"]` | `medium` | Amount of search context to include (low saves tokens, high is more thorough). |
| `WEB_SEARCH_LOCATION_CITY` | `str` | `""` | City for location-aware search results. |
| `WEB_SEARCH_LOCATION_REGION` | `str` | `""` | Region/state for location-aware search results. |
| `WEB_SEARCH_LOCATION_COUNTRY` | `str` | `""` | Country code (e.g. AU, US) for location-aware search results. |
| `WEB_SEARCH_LOCATION_TIMEZONE` | `str` | `""` | Timezone (e.g. Australia/Sydney) for location-aware search results. |
| `WEB_FETCH` | `bool` | `False` | Enable OpenRouter web fetch (URL reading) for you. |
| `DATETIME` | `bool` | `True` | Enable OpenRouter datetime tool for you (free, no extra cost). |
| `DATETIME_TIMEZONE` | `str` | `""` | Timezone for the datetime tool (e.g. Australia/Sydney). Empty uses UTC. |
| `ADVISOR` | `bool` | `False` | Enable the OpenRouter advisor tool (consult a higher-intelligence model mid-generation). Incurs an extra model call. |
| `SUBAGENT` | `bool` | `False` | Enable the OpenRouter subagent tool (delegate tasks to a worker model an admin chooses). Runs an extra model call. |
| `SEARCH_MODELS` | `bool` | `False` | Enable the OpenRouter model-search tool (let the model search the OpenRouter catalog). |

A stored value from an older version of the pipe that no longer fits its field falls back to that field's default rather than failing the request, and every other stored value on the same row is kept.

### OpenRouter Image Generation filter user valves

Six controls, always six. The filter is re-rendered for whichever model
`IMAGE_GENERATION_MODEL` names, reading that model's own published list of what it
accepts — the same list, read the same way, as the per-model image filters described in
[OpenRouter Image Generation](openrouter_image_generation.md). Five of the six are always
the same five; the sixth depends on the model. What that list changes most is what each
control will let a user pick: where the model publishes values, the control becomes a
dropdown of exactly those, and where it publishes none the control takes free text and
names in its description what OpenRouter's image API accepts.

Turning image generation on for a chat is the filter's own on/off switch in the message
box, not one of these settings.

| Valve | On screen | Type | Shown when |
| --- | --- | --- | --- |
| `IMAGE_QUALITY` | Quality | `Literal` over the published levels, otherwise `str` | Always. |
| `IMAGE_ASPECT_RATIO` | Aspect ratio | `Literal` over the published ratios, otherwise `str` | Always. |
| `IMAGE_BACKGROUND` | Background | `Literal` over the published treatments, otherwise `str` | Always. |
| `IMAGE_OUTPUT_FORMAT` | Output format | `Literal` over the published formats, otherwise `str` | Always. |
| `IMAGE_OUTPUT_COMPRESSION` | Output compression | `int \| None`, bounded by the published range where there is one and by OpenRouter's documented 0-to-100 otherwise | Always. |
| `IMAGE_RESOLUTION` | Resolution | `Literal` over the published tiers | The model publishes a tier list. |
| `IMAGE_SIZE` | Output size | `str` | The model publishes no tier list. |

The last two are alternatives, never both at once: a model that publishes size tiers gets
the **Resolution** dropdown, and one that does not gets **Output size** instead, where a
tier name or exact pixels can be typed. Across the fifty-one image models recorded in this
repository, nineteen draw **Resolution** and thirty-two draw **Output size**.

Where a control falls back to `str` — because the model publishes no values for it — its
own description names what OpenRouter's image API accepts there, so an admin or user still
knows what to type. `IMAGE_OUTPUT_COMPRESSION` behaves the same way: bounded by the model's
own range when it publishes one, and by OpenRouter's own 0-to-100 range otherwise. A bound
is not a value list: `0` to `100` refuses only what OpenRouter itself refuses, and says
nothing about which number this model honours, so it can come from the API-wide schema
where a set of named choices cannot.

`IMAGE_SIZE` is the one control that is checked further along, and only on the
server-tool path: the `size` the pipe sends is measured against the selected model's
published `resolution` list, and a tier that model does not publish is withheld from
the request and named in a toast. On a model that publishes no `resolution` list there
is nothing to measure against, so the typed value goes out as typed, exactly as on the
direct path. The other five controls are not gated — whatever they hold reaches the
service as typed.

A model the pipe's image model list does not carry, or whose settings could not be read
this time, still gets all six — with `IMAGE_SIZE` as the sixth, and every one of them
offering what OpenRouter's image API accepts in general rather than that model's own
values. `IMAGE_GENERATION_MODEL`'s own description says which case it is.

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
[Pipe orchestrator: measure the image tool's size against the selected model] -- reads __metadata__["openrouter_pipe"]["server_tools"], injects into API request tools array
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
5. (Web Tools filter only) When web search is enabled, suppresses Open WebUI's native web search by setting `body["features"]["web_search"] = False` to prevent double-searching, so the suppression follows the live switches rather than the ones captured when the chat started.

The Image Generation filter merges into any existing `server_tools` dict (so both filters can run on the same request without overwriting each other).

### When the image tool returns no image

The image tool can report that it ran and hand back nothing to show. On the server-tool
path that used to pass silently: the turn carried a tool result saying `completed`, a
status line reading *Generating image…* stayed up, and no picture appeared. The person now
gets a warning naming the condition, and the status line resolves as the item is handled —
but only when this loop opened that line, so an image item never closes a status window
belonging to another tool.

An image may arrive in `result`, in `imageUrl` or in `imageB64`; all three are read, since
OpenRouter's own success item carries `imageUrl` and no `result` key at all. A nested
`result` is read too, because the renderer descends into one. The check is on the item the
service sent, not on whether a picture rendered — an image that arrived but could not be
written to storage is a storage problem, and keeps its own wording rather than being
reported as an empty result.

The empty branch records no tool result at all: it warns, resolves the window it opened,
and moves on, so the turn carries no result claiming a picture. `server_tool_status` reads
the same item as `incomplete`, which is what the recorded text would say if a result were
written.

This covers the **server-tool** path. The direct `image_config` path already failed loudly,
and it has two conditions worth telling apart rather than the server-tool path's one. A
response carrying no images at all raises *"OpenRouter image generation returned no
images."* — the same string the server-tool path now uses. A response that carried images
and had all of them rejected raises *"OpenRouter image generation returned no usable
images"*, with the rejected names appended; the server-tool path does not model that
second case, because it has no equivalent of a rejected candidate to count.

Both paths reach the emitter, including a non-streamed reply: the non-streamed loop
delegates to the streaming one, and the same warning and the same clear are sent.

### When a web tool is switched off

Switching a web tool off affects **every** Web Tools filter this pipe maintains, whatever its id, and whether auto-install is on or off:

- While at least one web tool is still on, every Web Tools filter this pipe maintains that still offers a switched-off tool is rewritten without it, whether it is on or off, so its Integrations toggle disappears at the next model-list refresh, or after the first message that still asks for it. Until then that chat gets neither search nor fetch.
- With **every** web tool off, every Web Tools filter this pipe maintains is switched off. Nothing is added back. A filter you switch off yourself in Open WebUI's Functions list stays off: the pipe keeps its code up to date but never switches it back on. One this version switched off itself comes back on its own when you enable the feature again.
- Where several copies exist, the pipe maintains and attaches the one with the id `openrouter_web_tools`, or the most recently updated copy if none has that id. Turning a web tool back on revives **the copy the pipe maintains** and leaves the others switched off until an admin switches them on in the Functions list.
- If the row you need was one **you** switched off, the pipe will not bring it back; switch it on there.
- **Upgrading:** a filter that was already off before this version stays off. The pipe only re-arms a filter it switched off itself, and it records that in the filter's own meta. The same record is what makes retirement safe: turning an `AUTO_INSTALL_*` valve off switches off only the rows the pipe itself installed, so a copy an admin added by hand — Web Tools, Direct Uploads or anything else — is left alone. A row that was already off when you upgraded carries no such record. Switch it on in Workspace > Functions if you want it.

The same rule holds for every other filter the pipe installs (Fusion, image generation, the per-model panels, Direct Uploads, provider routing).

### Pipe side (orchestrator)

The pipe's request orchestrator:

1. Reads `__metadata__["openrouter_pipe"]["server_tools"]`.
2. For each tool in the dict, builds a tool spec (`{"type": "<tool_name>", ...params}`) and appends it to the `tools` array in the outgoing API request body, leaving out every tool whose `ENABLE_*` gate is off, whether a filter writes it or the request lists it. A type the request body already lists is **not** appended a second time: the filter's parameters are the ones sent, and the body's own entry for a type the filter does not ask for is left exactly as it was written. For the image tool, the `size` it puts in that spec is measured against the selected model's published contract, and a tier the model does not publish is withheld and named in a toast.
3. The tools array is sent alongside any Open WebUI registry tools or Direct Tool Server tools.
4. If the chat asked for a web tool whose gate is now off, it schedules a background repair of the Web Tools filters, so the filter stops offering a switched-off tool without waiting for the next model-list refresh. The repair runs at most once every five minutes: a pass that could not reach a verdict -- the function store could not be imported or listed, or the auto-install raised -- leaves that window unarmed and is retried on the next request rather than waiting one out, while a pass that ran to its end consumes the window whether or not the write was accepted, so a standing refusal costs one write attempt per window and not one per request.

### How web-tool counts are reported
A server tool that runs inside a model call is counted in that call's `usage` block, under `server_tool_use_details` (for example `{"web_search": {"executed": 2}}`). When a turn makes several upstream calls -- the tool loop, or a Fusion panel where each member can run its own web tool -- those blocks are merged into one, and `server_tool_use_details` is **not** one of the keys that are added up: the last call's figure replaces the accumulated one.

This is deliberate. The OpenRouter schema's "do not sum the two" applies *within* one block, not across generations, and the reason for last-wins across generations is that the block is a per-request figure rather than a per-token one -- the same reason Open WebUI's own merge treats it that way. Token and cost keys (`input_tokens`, `output_tokens`, `cost`, and the `*_tokens_details` maps) are still summed, so the turn's token and cost totals are the whole turn's.

The practical consequence is scoped to Fusion: on a panel, `server_tool_use_details.web_search.executed` reports the **last member to finish**, not the panel total. Nothing in this repository or in Open WebUI's backend reads that key, so the figure is not displayed anywhere today; `docs/openrouter_integrations_and_telemetry.md` lists every key that sums and every key that does not.

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
| `disable_web_tools_auto_attach` | Prevents auto-attaching the Web Tools filter to this model (the toggle will not appear in Integrations). |
| `disable_web_tools_default_on` | Prevents auto-enabling the Web Tools filter by default for this model (the toggle appears but starts off on the next sync, and a default the pipe had already seeded for it is released on that same sync). |

These parameters are respected even when the global `AUTO_ATTACH_WEB_TOOLS_FILTER` and `AUTO_DEFAULT_WEB_TOOLS_FILTER` valves are enabled: `disable_web_tools_default_on` releases a default the pipe seeded for that model on the next sync, and neither of them detaches a panel the pipe attached. A default seeded under an id the panel no longer resolves is released on that same sync, including after the panel has been reinstalled under a new id, and turning the valve back on reclaims a default re-ticked in between so the next turn-off still removes it.

See: [OpenRouter Integrations & Telemetry](openrouter_integrations_and_telemetry.md) for the full list of per-model custom parameters.

---

## Recommended operator settings

### All server tools available (current defaults)

All `ENABLE_*` gates are `True` and all `AUTO_INSTALL_*` and `AUTO_ATTACH_*` valves are `True`, while `AUTO_DEFAULT_WEB_TOOLS_FILTER` is `False`. The OpenRouter Web Tools toggle appears on every model that is not an image-output, a video-generation or a Fusion model but starts off; users enable it per chat, and within it Web Search and Datetime are on by default while Web Fetch, Advisor, Subagent, and Model Search are opt-in.

### When a web tool is switched off

Switching one of the six web tools off changes more than the request: a Web Tools filter written before the switch
still offers the tool, and its inlet still suppresses Open WebUI's own search, so the switch would appear to do
nothing. The pipe repairs the rows:

- **Which filters are affected:** every Web Tools filter this pipe maintains -- one it installed, or one carrying no install record -- whatever its id and whether it is on or off, whether auto-install is on or off.
- **What "offers" means:** the per-chat switches the filter's `UserValves` declares, read from its code without running it.
- **What is written:** the code is replaced with the pipe's current version offering the tools it still offers. Its
  name, settings and on/off state are kept; a filter never gains a tool it did not offer, so switching a tool back on
  does not add it back. Hand edits in the code are lost and a warning names the row and the tools removed.
- **A filter the pipe cannot read** (its code does not parse, or has no `Filter.UserValves`) is left exactly as it is
  and named in a warning; repeats for the same row and set of switched-off tools log below WARNING.
- **With every web tool off,** every Web Tools filter this pipe maintains is switched off, and the default the pipe seeded is removed from every model it seeded it on.
- **When it happens:** at the next model-list refresh, or in the background after a message that still asks for a
  switched-off web tool. That message has already gone out without Open WebUI's search; the repair is for the chats
  after it.
- **With several copies:** the pipe maintains and attaches `openrouter_web_tools` whenever a row with that id exists, and
  otherwise the copy the pipe most recently rewrote.
- **Turning a web tool back on** revives **the copy the pipe maintains** and leaves the others switched off until an
  admin switches them on in the Functions list.
- **A filter you switched off yourself stays off:** the pipe keeps its code up to date but never switches it back on.
  Switch it on in Open WebUI's Functions list if you want it. One this version switched off itself comes back on its own
  when you enable the feature again.
- **Upgrading:** a filter that was already off before this version stays off. The pipe only re-arms a filter it switched
  off itself, and it records that in the filter's own meta. The same record is what makes retirement safe: turning an `AUTO_INSTALL_*` valve off switches off only the rows the pipe itself installed, so a copy an admin added by hand — Web Tools, Direct Uploads or anything else — is left alone. A row that was already off when you upgraded carries no
  such record. Switch it on in Workspace > Functions if you want it.

The same rule holds for every other filter the pipe installs (Fusion, image generation, the per-model panels, Direct
Uploads, provider routing).

### Disable image generation

Set `ENABLE_IMAGE_GENERATION=False`. The Image Generation filter will not be generated or installed, the tool stops being sent at once, and the filter is switched off at the next model-list refresh. Alternatively, set `AUTO_INSTALL_IMAGE_GEN_FILTER=False` to keep the gate open but skip auto-installation.

### Web search opt-in (lower cost)

Set `AUTO_DEFAULT_WEB_TOOLS_FILTER=False`. The Web Tools toggle remains available on models, but will not be enabled by default, and a default the pipe had already seeded for one of them is released on the next sync, including one seeded under an id the panel no longer resolves. Users must manually enable it per chat.

### Restrict search domains

Configure the Web Tools filter's admin valve `WEB_SEARCH_ALLOWED_DOMAINS` with a comma-separated domain list (e.g. `docs.python.org, stackoverflow.com`). Search results will be restricted to those domains.
