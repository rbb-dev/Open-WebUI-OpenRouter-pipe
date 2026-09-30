# Model Catalog & Routing Intelligence

This document explains how the pipe loads OpenRouter’s `/models` catalog, derives per-model capabilities/features, and uses that metadata to shape requests (reasoning, multimodal inputs, web search plugin attachment, and model allowlists).

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [OpenRouter integration](openrouter_integrations_and_telemetry.md) · [Multimodal](multimodal_ingestion_pipeline.md)

---

## 1. Catalog ingestion and caching

### 1.1 ID normalization (sanitized vs normalized IDs)

OpenRouter model IDs use slash-separated provider slugs like `vendor/model`. Open WebUI function model IDs are dot-prefixed and dot-separated. The pipe therefore normalizes IDs in two steps:

- `sanitize_model_id("vendor/model")` → `vendor.model` (slash-to-dot conversion for Open WebUI display).
- `ModelFamily.base_model(...)` → lowercase, with:
  - pipe prefix stripped when present (`<pipe-id>.…`) - so a valve entry may be written either as OpenRouter's `author/model` form or as the exact id Open WebUI shows in the model picker, and both forms resolve to the same model,
  - date suffixes like `-YYYY-MM-DD` **kept**.

A dated snapshot is a model OpenRouter prices and serves separately, so it is its own key: a request
for `openai/gpt-4o-2024-11-20` goes out with that id, and `spec()`, `supported_parameters`,
`max_completion_tokens` and ZDR membership are read from that row. This is also what Open WebUI
does with a connection's `model_ids` — it publishes exactly the ids that are listed, and never
normalises dates.

`ModelFamily.undated(...)` is the date-insensitive form (`openai.gpt-4o-2024-11-20` →
`openai.gpt-4o`, with any `:variant`/`:preset` suffix kept). Only the rules documented to be
date-insensitive use it: `Models forced to chat completions`, `Models forced to responses` (a
pattern that names a date stamp still matches only that snapshot) and the phase-capable model list.

The normalized ID is used as the stable key for catalog specs and feature lookups. Routing resolves the **exact** catalogue id: a dated snapshot and its rolling id are separate rows with separate keys, and the id map is keyed by the exact form, so it is never a merge key. An id the catalogue does not list — the routing variants the pipe synthesises — falls through to reconstruction.

### 1.2 Fetching `/models` and caching results

The catalog is loaded via `OpenRouterModelRegistry.ensure_loaded(...)`:

- Requires a non-empty API key (`API_KEY` valve or `OPENROUTER_API_KEY` env).
- Fetches `GET {BASE_URL}/models` and parses the JSON payload.
- Stores:
  - a sorted list of models (for Open WebUI selectors), and
  - a spec map keyed by normalized ID, and an id map keyed by the exact catalogue ID (for routing decisions).

Caching:
- `MODEL_CATALOG_REFRESH_SECONDS` controls how long the cache is considered fresh (default `3600` seconds). The cache is keyed on the API key as well as the clock, so a key change refetches immediately instead of waiting out the TTL — on that credential's first round; once it has a failure window of its own, that window is what it waits out (see below). The Zero Data Retention roster that comes back is kept per credential rather than in one process-wide slot, for the most recent 4 credentials: a write re-inserts its own key last and drops the oldest over that cap, so the credential in hand is never the victim of its own write, and one install — which runs a single key, since the pipe reads `API_KEY` and nothing else — never reaches the cap at all. A credential whose roster is dropped is re-read when it returns rather than answered from cache, because the read stamp that would have suppressed the fetch belongs to whichever credential was read most recently.
- A refresh is single-flight: concurrent callers on a stale cache queue behind the one already fetching, and the waiters re-check the clock rather than each starting their own — including when the fetch in flight fails or comes back empty, so a burst of callers cannot multiply into a burst of requests. The cost of collapsing a burst that failed is that the same pass stamps both the model clock and the contract clock, so the next retry after an OpenRouter outage is suppressed for `MODEL_CATALOG_REFRESH_SECONDS`; that is deliberate, and it is what makes the image catalogue loader behave like the video one. An `asyncio.Lock` binds to the loop that first contends it and never gives it up, so a single shared lock cannot serve two loops. Each of the four locks that guard these loaders -- one for the model list, one for the image catalogue's contract sweep, plus the `image` and `video` catalogue locks themselves -- is therefore keyed on the running loop and held weakly, so a loop that finishes takes its entry with it and the loop object does not stay reachable for the life of the process. The mapping is read without the guard and written under it, so the cost of a cache hit is one dictionary lookup. What this guarantees is that a lock, once given to a loop, is only ever returned to that loop: no loop is handed another loop's fetch. Single-flight within a loop holds exactly as described above, and the two loops a test session runs are independent of each other rather than serialised against one another.
- On refresh failures, the registry records an exponential backoff window. If cached models exist, the pipe can continue serving them; if no cache exists yet, the error is surfaced to the caller. The window is `5 s × 2^(failures−1)` with the exponent capped at 5, so the doubling stops at `160 s` however long the outage runs, and `MODEL_CATALOG_REFRESH_SECONDS` is a further cap on it that binds only when set below 160. A failure's own window is its retry deadline, so the ramp above is reachable from the first failure; `_record_refresh_success` replaces it, and resets that account's failure count to zero, for a successful fetch **under the same account**. Both the attempt record and the failure count are per credential, so one account's outage no longer makes another account believe the key changed on every call — that would otherwise settle the newcomer on `0.0` and issue a request per model-list build. On an empty spec map — a fresh install, before anything has ever been fetched — the cold path consults only the window that credential's own last failure recorded, never the freshness horizon, which is derived from a fetch that has not happened. A model-list build inside that window issues no request and still refuses the request with the catalogue-unavailable card, rather than returning an empty model list that the caller would read as a model it does not carry.
- A per-credential settle rides on top of the backoff. After the second consecutive failing round under one credential, the window that round recorded becomes the clock that credential's next call consults, so a raising endpoint issues no further request per model-list build until that window expires; the first fetch after it expires is a normal one, and any successful round clears the record. The settle is keyed on the raising credential and reads that credential's own recorded value, so one account's outage is never another account's reason to skip a read. It is bounded the same way as the roster — the most recent 4 credentials — which is why the cap is 4 and not 1: a cap of one would let an account that reads first suppress the read of the account that reads second, and on a multi-key install the two interleave on every model-list build. A credential older than the cap has its settle record dropped with its roster, so a returning credential finds no recorded window and re-reads rather than inheriting another account's backoff.

The video catalog is registered on top of the chat catalog, and where they meet, the chat field wins: a normalized ID that already holds a chat spec keeps its `features`, `capabilities`, `context_length`, `max_completion_tokens`, `description`, `pricing`, `architecture` and `full_model`, and the display name the model picker publishes for it, so the picker and the error card name the same model the same way whichever fetch ran last. The video catalog contributes the row's `video_model` and the features and video-only capability flag that go with it, but not its name. The image catalog does not merge: it leaves an existing chat spec alone and registers no image row at all for an ID the chat catalog already lists. A normalized ID in neither catalog registers from its own row alone. The chat side's `image_output` / `image_gen_tool` features are the exception: they are dropped from a dual model, which has no `image_model` row to serve them, and a chat refresh in turn wins over a video or image spec on the same ID.

Ownership runs the other way too, on the retirement side: where a normalized ID is in both catalogs, neither the video catalog's refresh nor the image catalog's sweep may retire it, and the image sweep may not re-register it either — a row the chat catalog still lists keeps the chat catalog's spec, features, `supported_parameters`, `context_length` and `description`, not the bare endpoint listing the image catalog publishes for the same slug. This is why a model with `image` and no `text` in `output_modalities`, which belongs to both catalogues, keeps its chat row. A refresh that stops listing such an ID — a sweep that publishes fewer rows than the last one, or none at all, since a `/videos/models` answer of zero rows is a withdrawal of every row it published — returns its row to the chat catalog's own features, capabilities, `supported_parameters` and `full_model`: the video claims (`video_generation`, `video_output`, `video_model`, `capabilities.video_generation`) are dropped rather than subtracted, because the video row's `supported_frame_images` also contributed `vision`, which is a legitimate chat feature and has to survive, and the passthrough set is restored to the chat catalog's own because the video row's `allowed_passthrough_parameters` is the video endpoint's parameter list, not the chat model's — leaving it behind reads a tool-capable chat model as one that publishes no tools, and the orchestrator then strips every tool the request carried. Only a normalized ID in neither catalog is retired, the moment it leaves `/videos/models` or the image catalog's own list — including through the image catalog's empty-list (master-disable) arm, which retires the same image-only rows a non-empty sweep does.

### 1.3 Derived spec fields (what the pipe computes)

For each model, the registry stores the full catalog entry (`full_model`) and derives:

- `supported_parameters`: the provider-reported supported parameter set (stored as a `frozenset`).
- `features`: a set of higher-level flags derived from `supported_parameters`, model architecture, and pricing metadata:
  - `function_calling` (based on support for tools-related parameters)
  - `reasoning` and `reasoning_summary` (based on reasoning-related parameters)
  - modality flags: `vision`, `audio_input`, `video_input`, `file_input`
  - `image_gen_tool` (based on output modalities)
- `capabilities`: a dictionary of Open WebUI “capability checkboxes” used for UI affordances (for example `vision`, `file_upload`, `web_search`, `image_generation`), plus always-on UI toggles (`code_interpreter`, `citations`, `status_updates`, `usage`). The always-on list is what the *catalogue* writes: it always publishes `citations: True` for a chat model. What a row can *say* is a separate matter once an operator has taken the box over — `citations` is now seeded rather than assigned, like `web_search`, so an admin's untick survives a refresh, and the pipe honours it on the tool path by publishing no tool-derived source for that model (Open WebUI's own `url_citation` annotations and the pipe's Fusion item sources are not gated, because Open WebUI does not gate its own). A row that carries no `citations` key at all is on.
- When enabled, the pipe can also sync these capability checkboxes into Open WebUI model metadata (`meta.capabilities`) so the UI reflects OpenRouter’s catalog.
- `web_search` is seeded from the model’s published `web_search` pricing and is never overwritten once you have set it, so a model OpenRouter does not price for search can still get Open WebUI’s native search by a manual tick. This applies to rows the pipe writes; a row synced by an earlier version keeps whatever `web_search` value it already has until an admin edits it. `citations` is seeded the same way, with the catalogue's own `True` as the value a row with no setting receives.
- `max_completion_tokens`: taken from the model’s `top_provider.max_completion_tokens` field when present.

The derived specs are shared with `ModelFamily` via `ModelFamily.set_dynamic_specs(...)`, so the rest of the pipe can use `ModelFamily.supports(...)`, `ModelFamily.capabilities(...)`, and `ModelFamily.supported_parameters(...)` without depending directly on the registry.

---

## 2. How models are exposed to Open WebUI (`pipes()`)

Open WebUI requests the available models from the function by calling `Pipe.pipes()`.

Behavior:
- The pipe loads/refreshes the OpenRouter catalog (best-effort; may serve cached models on failure).
- The tail of the call is best-effort too: a failure in the model-metadata sync, in the image-generation filter lookup, or in the plugin `on_models` dispatch is logged and the pipe still returns the models it has. A raise at any of those points would make Open WebUI's handler serve an empty model list, so every picker would lose the pipe.
- The system valve `MODEL_ID` selects which models are exposed:
  - `auto` exposes the full catalog.
  - A comma-separated list restricts the exposed models. An entry holding `*`, `?` or `[` is a glob pattern rather than one id, matched case-insensitively against every catalog row on both the `author/model` and the dotted spelling, so `deepseek/*` publishes that whole family; a pattern is matched as written and is never split on `:` or `@`. A `!`-prefixed entry excludes, and exclusions run after the includes, so `deepseek/*, !deepseek/*:free` publishes the paid rows and drops the `:free` one. A list made only of exclusions starts from the whole catalog, which `auto, !openai/*` also says. A `*` crosses a date stamp, so `openai/gpt-4o*` reaches `openai/gpt-4o-2024-08-06`; globs are literal about the `~` prefix, so `openai/*` does not cover `~openai/...` and `!~openai/*` is what removes one. A list that resolves to nothing exposes nothing, and the pipe refuses every request rather than serving the whole catalog; one `WARNING` names the IDs it could not resolve, and so does an include pattern that matches no row. The one exception is a value of only commas or spaces, which is read as blank and imports the whole catalog. An `@preset/slug` entry resolves to the model before the `@`. A `base_id:tag` entry resolves to that exact tagged model when the catalog lists it as one in its own right — `openai/gpt-4o:free` publishes the `:free` model, not the paid base — and otherwise to its base.
- The pipe returns a minimal `{"id","name"}` list for the model selector.
- The special `openrouter/auto` model is included in the catalog and can be selected like any other model. Auto Router configuration (allowed model patterns) is managed in the OpenRouter UI (Settings → Plugins) and is not surfaced in Open WebUI.
- Optional: the pipe can schedule a background “model metadata sync” that writes Open WebUI model metadata:
  - `meta.capabilities` (capability checkboxes), and
  - `meta.profile_image_url` (model icon as a PNG data URL, on either a fetched or an inline `data:` source, and only when the decoded bytes are within 2 MiB and the pixels within 25 Mpx — an icon over either bound is skipped with a `debug` log), stamped with the source URL the icon was downloaded from (`image_source_url`; an inline `data:` source is not stamped, since the stamp is the raw catalogue URL and the payload would then ship whole on every model-list page) and with which of the two sources it was (`image_source_kind`: `frontend` for a catalogue icon, `maker` for a maker's logo) so an unchanged source is not downloaded again, and for a model taking its maker's logo the maker's page is not re-fetched either, which keeps a hand-picked icon on that row until its source URL changes — a change to the image at the same URL is therefore not picked up. The stamp records provenance, not a verdict, so the two ways a `frontend` row can leave that state are told apart by the read rather than by the stamp: a pass whose catalogue read DID NOT ANSWER leaves the row exactly as it is and scrapes no maker page, while a read that answered and no longer names that model's icon moves the row to its maker's logo — and
  - `meta.description` (model description text).
  This behavior is controlled by `UPDATE_MODEL_CAPABILITIES`, `UPDATE_MODEL_IMAGES`, and `UPDATE_MODEL_DESCRIPTIONS`. A sync that fails or is cancelled is retried on the next model-list refresh and its exception is logged at ERROR, so a stale sync is distinguishable from a healthy one. See: [OpenRouter Integrations & Telemetry](openrouter_integrations_and_telemetry.md).
  - New model access control defaults are set **on insert only, and read once per pass**: `NEW_MODEL_ACCESS_CONTROL` determines whether newly inserted OpenRouter overlays are public (wildcard read grant) or private (no access grants), with the `admins` option relying on Open WebUI's `BYPASS_ADMIN_ACCESS_CONTROL` for admin access. A pass already running when you save the valve finishes under the value it started with, so every row it writes carries the same policy.

---

## 3. Model allowlists and enforcement in requests

At request time, the pipe computes the allowed model set based on `MODEL_ID` and the loaded catalog:

- The catalog set, the allowlist set, the enforced set and the virtual-variant map are computed **once per (catalogue, specs, credential, valve snapshot)** on the pipe instance and reused while all four of those are unchanged. A change to any of them — a catalogue or spec refresh, a media-model registration, a different OpenRouter credential in play, or a change to `MODEL_ID`, `VARIANT_MODELS`, `FREE_MODEL_FILTER`, `TOOL_CALLING_FILTER` or `ZDR_MODELS_ONLY` — is visible on the very next request.
- `VARIANT_MODELS` is a second source of admission: a variant whose base is outside `MODEL_ID` is refused at request time, however it is named.
- For normal chat/API calls:
  - if the requested model is not in the allowed set, the pipe emits a user-facing error telling the user to choose an allowed model.
  - The allowed set is computed from the resolved `MODEL_ID` allowlist, so an unresolvable allowlist leaves it empty and every model is refused — and so does an include pattern that matches no row.
- For Open WebUI “task” requests (`__task__`):
  - **housekeeping tasks** bypass the model whitelist (so titles/tags/follow-ups and similar flows can still run even when end-user models are locked down).
  - `moa_response_generation` follows the normal chat restriction path and does **not** bypass the whitelist.
  - task-specific reasoning overrides apply only to housekeeping tasks when the task model is one of this pipe’s owned/allowed models.

See also: [Task Models & Housekeeping](task_models_and_housekeeping.md).

---

## 4. Capability-driven behavior (how the catalog influences routing)

### 4.1 Multimodal gating (vision and attachments)

The pipe uses catalog-derived capabilities to decide whether to forward image inputs:
- If the selected model is not vision-capable, user image attachments are skipped and a status message is emitted so users understand why attachments were ignored. The status is emitted once per request, on the request's latest user turn, and it names the pictures the gate left out on *any* turn of the request - the turn that carried them and the turns before it. An operator should not expect one status per picture: the count is the number, so a chat with three skipped pictures reports one status saying three.

Details are in: [Multimodal Intake Pipeline](multimodal_ingestion_pipeline.md).

### 4.2 Tooling and function calling

Tool definitions are built from Open WebUI tool registries and other configured sources, but the pipe only attaches `responses_body.tools` when the selected model supports function calling per catalog-derived feature flags.

A request-parameter gate has the same shape and the same caveat. `include_reasoning` is sent only when the primary model **and every fallback in `models`** list it, because OpenRouter forwards the key to whichever model ends up serving and a provider that does not know the parameter rejects the whole request rather than its own leg of it. An id the catalogue does not know counts as not listing it. The gate cannot live in `reasoning_config.py`, which reads the primary only: `model_fallback` is merged into `models` later, on the request payload, after every reasoning decision has been made.

See: [Tooling & Integrations](tooling_and_integrations.md).

### 4.3 Reasoning defaults and compatibility

The pipe decides how to request reasoning from the selected model's catalog entry: its `supported_parameters` and its `reasoning` object (`ModelFamily.catalog_norm_id`, `ModelFamily.supported_parameters` and `ModelFamily.reasoning_contract`). Which entry is used:

- A routing variant with no catalog entry of its own (for example `:nitro` or `:online`) uses its base model's entry, so it is sent what its base model is sent.
- A suffixed id the catalog lists as a model of its own (for example a `:free` model) uses its own entry. The tags OpenRouter lists as a model in its own right are `:free` today, and `:batch` since 2026-08-09.
- A preset model (`base_id@preset/slug`) gets no reasoning field of the pipe's own, so the preset's saved settings apply; a reasoning field the chat itself carries still goes out and overrides them for that request.

The pipe decides how to request reasoning from the model's catalog entry:

- If the model supports `reasoning`, the pipe populates a `reasoning` object (with defaults from valves such as `REASONING_EFFORT` and `REASONING_SUMMARY_MODE`).
- If the model does not support `reasoning` but supports the legacy `include_reasoning`, the pipe uses that fallback, and the flag is that model's only reasoning channel: an off the request itself carries (`reasoning.enabled` of `false`, `reasoning.exclude` of `true`, or an effort of `none`, including the `reasoning_effort` spelling that becomes that object) decides it, and `REASONING_EFFORT` decides it only when the request says nothing about reasoning. A request that asks for reasoning does not switch an install whose `REASONING_EFFORT` is `none` back on. An owned background task is the exception: it is the task valve that decides there.
- If neither is supported, the pipe adds no reasoning field of its own; a reasoning effort or reasoning parameter the chat itself carries goes out as Open WebUI would send it.

Gemini 2.5 models:

- The thinking budget (`GEMINI_THINKING_BUDGET`, scaled by the reasoning effort) is sent as OpenRouter's `reasoning.max_tokens`, with no `effort` beside it. A budget the request brought keeps whatever `effort` the request or the valve put beside it.
- A budget of `0` sends `reasoning: {"effort": "none"}`, which switches thinking off, except on Gemini 2.5 Pro, which cannot stop thinking and thinks at its own default.
- Every Gemini 2.5 request the pipe shapes carries a `reasoning` object, so the pipe does not send this flag on its own; the one exception is the retry after a provider rejects reasoning, which resends with `include_reasoning: false` alone (see [Error handling](error_handling_and_user_experience.md)).

A request that sets its own `reasoning.max_tokens` overrides the valve: the pipe forwards that number, reduced to `min(number, cap - 64)` when the request carries its own output cap and not at all when that leaves no room for an answer, and writes no competing thinking budget, on every model family and both endpoints. While `REASONING_EFFORT` is `none`, though, that per-chat thinking budget is ignored: on a model that can stop reasoning the request goes out as exactly `{"effort": "none"}`, and on a mandatory model too, where the pipe refuses the off and substitutes the lowest effort the row supports. A request that sets none is unaffected by this paragraph.

Effort `none`:

- A model whose reasoning is mandatory (its entry's `reasoning.mandatory`) is never sent `none`: it gets the lowest level its entry lists other than `none`, and no level at all when it lists no other level.
- On other models, a `none` that comes from the pipe's own settings goes out as exactly `reasoning: {"effort": "none"}`, OpenRouter's off switch.
- A mandatory model cannot be asked to stop reasoning at all: an off arriving from the request or from the `REASONING_EFFORT` / `TASK_MODEL_REASONING_EFFORT` / `GEMINI_THINKING_BUDGET` valves is replaced by an effort the row supports, and a chat says so in a status line naming the model.
- A Gemini 2.5 request that sets its own `reasoning.max_tokens` and asks for `none` beside it is the one input on which both controls are honoured as sent: the off switch wins and the request goes out as `reasoning: {"effort": "none"}`, the per-chat number with it. Everywhere else the `none` in that paragraph is the pipe's own, and a per-chat number beside it is dropped.

Provider mismatch recovery:
- If a provider rejects reasoning due to a “thinking” configuration mismatch, the pipe may retry once with reasoning disabled (see [Error Handling & User Experience](error_handling_and_user_experience.md)). A row the catalogue marks `reasoning.mandatory` is not retried that way.

### 4.4 Web search server tool attachment

When the **OpenRouter Web Tools** filter is active and the user has enabled Web Search in their valves, the pipe attaches the `openrouter:web_search` server tool to the API request. The model decides when and whether to search — it may search zero or multiple times per request.

For the full User Interface story (Open WebUI Web Search vs OpenRouter Web Tools, and why OpenRouter Web Tools overrides Web Search), see:
[Web Search (Open WebUI) vs OpenRouter Web Tools](web_search_owui_vs_openrouter_search.md).

### 4.5 Output token cap selection

When `USE_MODEL_MAX_OUTPUT_TOKENS=True` and the request carries no limit of its own, the pipe fills `max_output_tokens` with the smaller of the provider-advertised `max_completion_tokens` in the catalog and half that model's context window, or the advertised value alone when its context window is unknown. When it is disabled, the pipe adds no limit of its own and provider defaults apply. The valve controls the pipe's automatic value, not the caller's: a `max_tokens`, `max_output_tokens` or `max_completion_tokens` of 1 or above is forwarded as a whole number whatever its spelling, and a fractional value rounds. OpenRouter documents the parameter as "1 or above" and Open WebUI's slider reaches -2, so a value below 1 is sent as no cap — and the automatic ceiling then applies if the valve is on. A routing variant resolves through its base's row, so it gets the base's ceiling.

On Gemini 2.5 the same cap bounds the thinking budget: `budget = min(budget, cap - 64)`, never the other way round, because reasoning tokens count against the cap and a budget equal to the cap is the documented failure boundary. When the cap leaves no room the pipe writes no bounded budget at all, writes no off flag, and leaves the cap as the caller sent it.

### 4.6 Auto context trimming (context-compression plugin)

When `AUTO_CONTEXT_TRIMMING=True`, the pipe enables OpenRouter’s `context-compression` plugin by appending `{"id": "context-compression"}` to the request’s `plugins` array only when no context-compression plugin is already present.

See: [OpenRouter Integrations & Telemetry](openrouter_integrations_and_telemetry.md).

---

## 5. Helper APIs you can rely on

| Helper | Returns | Usage |
| --- | --- | --- |
| `ModelFamily.base_model(model_id)` | normalized model key | Use for stable comparisons and allowlist checks. |
| `ModelFamily.supports(feature, model_id)` | boolean | Feature gates (vision, tools, web search support, etc.). |
| `ModelFamily.capabilities(model_id)` | `dict[str,bool]` | Open WebUI capability checkboxes for UI affordances. |
| `ModelFamily.supported_parameters(model_id)` | `frozenset[str]` | Provider-supported request parameter set (used for reasoning compatibility decisions). |
| `ModelFamily.max_completion_tokens(model_id)` | `int \| None` | Provider-advertised max completion tokens, used when `USE_MODEL_MAX_OUTPUT_TOKENS=True`. |
| `OpenRouterModelRegistry.api_model_id(model_id)` | provider slug, reconstructed non-catalog slug, or `None` | Maps the normalized/sanitized model ID back to the provider’s original ID for outbound API calls. The exact catalogue row always wins over its base family; only ids the catalogue does not list fall through, and then to reconstruction. A single leading pipe function-id prefix is removed on the reconstruction path too, and only when what follows it names a vendor segment. An id that is already double-prefixed, or whose remainder is empty or carries no vendor segment, is left alone, so there the function id does remain the provider's namespace. That is the boundary the two spellings of `open_webui_openrouter_pipe.gpt-4o` force: `sanitize_model_id` maps both a pipe-prefixed id and a catalogue id for a vendor slugged `open_webui_openrouter_pipe` to the same key, and the reconstruction path is reached only when the catalogue has nothing to say, so it cannot tell them apart. Leaving a bare remainder alone keeps a catalogue id for that vendor slug reachable instead of rerouting it to a different provider's model. |

---

## 6. Failure modes and operator signals

- Missing API key: catalog load fails with a configuration error and the pipe cannot expose models.
- Refresh failures with cache: the pipe can continue serving cached models; logs will show a warning about serving cached catalog data.
- Refresh failures with no cache: the error propagates, and requests that require the catalog cannot proceed.
- Empty catalog: the registry treats an empty model list as an error.
- Missing provider dropdown: the provider map is retained per slug, so a frontend-catalog cycle that returns nothing for a model the catalog still lists keeps that slug's previously fetched providers instead of dropping them.
- Failed stored-row read: a metadata pass reads its model rows in one batched query per chunk of 1000 ids. If that read raises, or answers with no rows at all, the pipe logs a warning naming the count of ids it could not read and falls back to reading each model on its own, so a broken batch can never be read as "these models do not exist" and cannot produce a duplicate overlay insert per model.

Operator guidance:
- Treat catalog failures like an upstream connectivity/credential issue first (API key, network egress, proxy/gateway, OpenRouter availability), then inspect logs for the last refresh failure.
