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
- `MODEL_CATALOG_REFRESH_SECONDS` controls how long the cache is considered fresh (default `3600` seconds).
- A refresh is single-flight: concurrent callers on a stale cache queue behind the one already fetching, and the waiters re-check the clock rather than each starting their own — including when the fetch in flight fails or comes back empty, so a burst of callers cannot multiply into a burst of requests. The cost of collapsing a burst that failed is that the same pass stamps both the model clock and the contract clock, so the next retry after an OpenRouter outage is suppressed for `MODEL_CATALOG_REFRESH_SECONDS`; that is deliberate, and it is what makes the image catalogue loader behave like the video one.
- On refresh failures, the registry records an exponential backoff window. If cached models exist, the pipe can continue serving them; if no cache exists yet, the error is surfaced to the caller.

### 1.3 Derived spec fields (what the pipe computes)

For each model, the registry stores the full catalog entry (`full_model`) and derives:

- `supported_parameters`: the provider-reported supported parameter set (stored as a `frozenset`).
- `features`: a set of higher-level flags derived from `supported_parameters`, model architecture, and pricing metadata:
  - `function_calling` (based on support for tools-related parameters)
  - `reasoning` and `reasoning_summary` (based on reasoning-related parameters)
  - `web_search_tool` (based on web search pricing metadata)
  - modality flags: `vision`, `audio_input`, `video_input`, `file_input`
  - `image_gen_tool` (based on output modalities)
- `capabilities`: a dictionary of Open WebUI “capability checkboxes” used for UI affordances (for example `vision`, `file_upload`, `web_search`, `image_generation`), plus always-on UI toggles (`code_interpreter`, `citations`, `status_updates`, `usage`).
- When enabled, the pipe can also sync these capability checkboxes into Open WebUI model metadata (`meta.capabilities`) so the UI reflects OpenRouter’s catalog.
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
  - A comma-separated list restricts the exposed models. A list that resolves to nothing exposes nothing, and the pipe refuses every request rather than serving the whole catalog; one `WARNING` names the IDs it could not resolve. The one exception is a value of only commas or spaces, which is read as blank and imports the whole catalog. An `@preset/slug` entry resolves to the model before the `@`; a bare `:tag` is not an allowlist entry.
- The pipe returns a minimal `{"id","name"}` list for the model selector.
- The special `openrouter/auto` model is included in the catalog and can be selected like any other model. Auto Router configuration (allowed model patterns) is managed in the OpenRouter UI (Settings → Plugins) and is not surfaced in Open WebUI.
- Optional: the pipe can schedule a background “model metadata sync” that writes Open WebUI model metadata:
  - `meta.capabilities` (capability checkboxes), and
  - `meta.profile_image_url` (model icon as a PNG data URL), and
  - `meta.description` (model description text).
  This behavior is controlled by `UPDATE_MODEL_CAPABILITIES`, `UPDATE_MODEL_IMAGES`, and `UPDATE_MODEL_DESCRIPTIONS`. A sync that fails or is cancelled is retried on the next model-list refresh and its exception is logged at ERROR, so a stale sync is distinguishable from a healthy one. See: [OpenRouter Integrations & Telemetry](openrouter_integrations_and_telemetry.md).
  - New model access control defaults are set **on insert only**: `NEW_MODEL_ACCESS_CONTROL` determines whether newly inserted OpenRouter overlays are public (wildcard read grant) or private (no access grants), with the `admins` option relying on Open WebUI's `BYPASS_ADMIN_ACCESS_CONTROL` for admin access.

---

## 3. Model allowlists and enforcement in requests

At request time, the pipe computes the allowed model set based on `MODEL_ID` and the loaded catalog:

- `VARIANT_MODELS` is a second source of admission: a variant whose base is outside `MODEL_ID` is refused at request time, however it is named.
- For normal chat/API calls:
  - if the requested model is not in the allowed set, the pipe emits a user-facing error telling the user to choose an allowed model.
  - The allowed set is computed from the resolved `MODEL_ID` allowlist, so an unresolvable allowlist leaves it empty and every model is refused.
  - A `VARIANT_MODELS` entry is expanded against the whole catalogue, so a variant whose base is outside the allowlist still enters this set even though the picker does not publish it. This gap is documented rather than closed.
- For Open WebUI “task” requests (`__task__`):
  - **housekeeping tasks** bypass the model whitelist (so titles/tags/follow-ups and similar flows can still run even when end-user models are locked down).
  - `moa_response_generation` follows the normal chat restriction path and does **not** bypass the whitelist.
  - task-specific reasoning overrides apply only to housekeeping tasks when the task model is one of this pipe’s owned/allowed models.

See also: [Task Models & Housekeeping](task_models_and_housekeeping.md).

---

## 4. Capability-driven behavior (how the catalog influences routing)

### 4.1 Multimodal gating (vision and attachments)

The pipe uses catalog-derived capabilities to decide whether to forward image inputs:
- If the selected model is not vision-capable, user image attachments are skipped and a status message is emitted so users understand why attachments were ignored.

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
- If the model does not support `reasoning` but supports the legacy `include_reasoning`, the pipe uses that fallback.
- If neither is supported, the pipe adds no reasoning field of its own; a reasoning effort or reasoning parameter the chat itself carries goes out as Open WebUI would send it.

Gemini 2.5 models:

- The thinking budget (`GEMINI_THINKING_BUDGET`, scaled by the reasoning effort) is sent as OpenRouter's `reasoning.max_tokens`, with no `effort` beside it.
- A budget of `0` sends `reasoning: {"effort": "none"}`, which switches thinking off, except on Gemini 2.5 Pro, which cannot stop thinking and thinks at its own default.
- Every Gemini 2.5 request the pipe shapes carries a `reasoning` object, so the pipe does not send this flag on its own; the one exception is the retry after a provider rejects reasoning, which resends with `include_reasoning: false` alone (see [Error handling](error_handling_and_user_experience.md)).

A request that sets its own `reasoning.max_tokens` wins over the valve: the pipe forwards that number and writes no competing thinking budget, on every model family and both endpoints. A request that sets none is unaffected by this paragraph.

Effort `none`:

- A model whose reasoning is mandatory (its entry's `reasoning.mandatory`) is never sent `none`: it gets the lowest level its entry lists other than `none`, and no level at all when it lists no other level.
- On other models, a `none` that comes from the pipe's own settings goes out as exactly `reasoning: {"effort": "none"}`, OpenRouter's off switch.
- A mandatory model cannot be asked to stop reasoning at all: an off arriving from the request or from the `REASONING_EFFORT` / `TASK_MODEL_REASONING_EFFORT` / `GEMINI_THINKING_BUDGET` valves is replaced by an effort the row supports, and a chat says so in a status line naming the model.

Provider mismatch recovery:
- If a provider rejects reasoning due to a “thinking” configuration mismatch, the pipe may retry once with reasoning disabled (see [Error Handling & User Experience](error_handling_and_user_experience.md)). A row the catalogue marks `reasoning.mandatory` is not retried that way.

### 4.4 Web search server tool attachment

When the **OpenRouter Web Tools** filter is active and the user has enabled Web Search in their valves, the pipe attaches the `openrouter:web_search` server tool to the API request. The model decides when and whether to search — it may search zero or multiple times per request.

For the full User Interface story (Open WebUI Web Search vs OpenRouter Web Tools, and why OpenRouter Web Tools overrides Web Search), see:
[Web Search (Open WebUI) vs OpenRouter Web Tools](web_search_owui_vs_openrouter_search.md).

### 4.5 Output token cap selection

When `USE_MODEL_MAX_OUTPUT_TOKENS=True` and the request carries no limit of its own, the pipe fills `max_output_tokens` with the smaller of the provider-advertised `max_completion_tokens` in the catalog and half that model's context window, or the advertised value alone when its context window is unknown. When it is disabled, the pipe adds no limit of its own and provider defaults apply. The valve controls the pipe's automatic value, not the caller's: a `max_tokens` of 1 or above is forwarded unchanged. OpenRouter documents the parameter as "1 or above" and Open WebUI's slider reaches -2, so a value below 1 is sent as no cap — and the automatic ceiling then applies if the valve is on. A routing variant resolves through its base's row, so it gets the base's ceiling.

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
| `OpenRouterModelRegistry.api_model_id(model_id)` | provider slug, reconstructed non-catalog slug, or `None` | Maps the normalized/sanitized model ID back to the provider’s original ID for outbound API calls. The exact catalogue row always wins over its base family; only ids the catalogue does not list fall through, and then to reconstruction. |

---

## 6. Failure modes and operator signals

- Missing API key: catalog load fails with a configuration error and the pipe cannot expose models.
- Refresh failures with cache: the pipe can continue serving cached models; logs will show a warning about serving cached catalog data.
- Refresh failures with no cache: the error propagates, and requests that require the catalog cannot proceed.
- Empty catalog: the registry treats an empty model list as an error.

Operator guidance:
- Treat catalog failures like an upstream connectivity/credential issue first (API key, network egress, proxy/gateway, OpenRouter availability), then inspect logs for the last refresh failure.
