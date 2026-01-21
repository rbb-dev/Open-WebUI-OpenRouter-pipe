# Model Catalog & Routing Intelligence

This document explains how the pipe loads OpenRouter’s `/models` catalog, derives per-model capabilities/features, and uses that metadata to shape requests (reasoning, multimodal inputs, web search plugin attachment, and model allowlists).

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [OpenRouter integration](openrouter_integrations_and_telemetry.md) · [Multimodal](multimodal_ingestion_pipeline.md)

---

## 1. Catalog ingestion and caching

### 1.1 ID normalization (sanitized vs normalized IDs)

OpenRouter model IDs use slash-separated provider slugs like `vendor/model`. Open WebUI function model IDs are dot-prefixed and dot-separated. The pipe therefore normalizes IDs in two steps:

- `sanitize_model_id("vendor/model")` → `vendor.model` (slash-to-dot conversion for Open WebUI display).
- `ModelFamily.base_model(...)` → lowercase, with:
  - pipe prefix stripped when present (`<pipe-id>.…`), and
  - date suffixes like `-YYYY-MM-DD` removed.

The normalized ID is used as the stable key for catalog specs and feature lookups.

### 1.2 Fetching `/models` and caching results

The catalog is loaded via `OpenRouterModelRegistry.ensure_loaded(...)`:

- Requires a non-empty API key (`API_KEY` valve or `OPENROUTER_API_KEY` env).
- Fetches `GET {BASE_URL}/models` and parses the JSON payload.
- Stores:
  - a sorted list of models (for Open WebUI selectors), and
  - a spec map keyed by normalized ID (for routing decisions).

Caching:
- `MODEL_CATALOG_REFRESH_SECONDS` controls how long the cache is considered fresh (default `3600` seconds).
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
- The system valve `MODEL_ID` selects which models are exposed:
  - `auto` exposes the full catalog.
  - A comma-separated list restricts the exposed models.
- The pipe returns a minimal `{"id","name"}` list for the model selector.
- Optional: the pipe can schedule a background “model metadata sync” that writes Open WebUI model metadata:
  - `meta.capabilities` (capability checkboxes), and
  - `meta.profile_image_url` (model icon as a PNG data URL), and
  - `meta.description` (model description text).
  This behavior is controlled by `UPDATE_MODEL_CAPABILITIES`, `UPDATE_MODEL_IMAGES`, and `UPDATE_MODEL_DESCRIPTIONS`. See: [OpenRouter Integrations & Telemetry](openrouter_integrations_and_telemetry.md).
  - New model access control defaults are set **on insert only**: `NEW_MODEL_ACCESS_CONTROL` determines whether newly inserted OpenRouter overlays are public (`access_control=None`) or private (`access_control={}`), with the `admins` option relying on Open WebUI’s `BYPASS_ADMIN_ACCESS_CONTROL` for admin access.

---

## 3. Model allowlists and enforcement in requests

At request time, the pipe computes the allowed model set based on `MODEL_ID` and the loaded catalog:

- For normal chat/API calls:
  - if the requested model is not in the allowed set, the pipe emits a user-facing error telling the user to choose an allowed model.
- For Open WebUI “task” requests (`__task__`):
  - the pipe **bypasses** the model whitelist (so housekeeping tasks can still run even when end-user models are locked down).
  - the pipe only applies task-specific reasoning overrides when the task model is one of this pipe’s owned/allowed models.

See also: [Task Models & Housekeeping](task_models_and_housekeeping.md).

---

## 4. Capability-driven behavior (how the catalog influences routing)

### 4.1 Multimodal gating (vision and attachments)

The pipe uses catalog-derived capabilities to decide whether to forward image inputs:
- If the selected model is not vision-capable, user image attachments are skipped and a status message is emitted so users understand why attachments were ignored.

Details are in: [Multimodal Intake Pipeline](multimodal_ingestion_pipeline.md).

### 4.2 Tooling and function calling

Tool definitions are built from Open WebUI tool registries and other configured sources, but the pipe only attaches `responses_body.tools` when the selected model supports function calling per catalog-derived feature flags.

See: [Tooling & Integrations](tooling_and_integrations.md).

### 4.3 Reasoning defaults and compatibility

When `ENABLE_REASONING=True`, the pipe decides how to request reasoning based on `ModelFamily.supported_parameters(...)` for the selected model:

- If the model supports `reasoning`, the pipe populates a `reasoning` object (with defaults from valves such as `REASONING_EFFORT` and `REASONING_SUMMARY_MODE`).
- If the model does not support `reasoning` but supports the legacy `include_reasoning`, the pipe uses that fallback.
- If neither is supported, the pipe disables reasoning for that request.

Provider mismatch recovery:
- If a provider rejects reasoning due to a “thinking” configuration mismatch, the pipe may retry once with reasoning disabled (see [Error Handling & User Experience](error_handling_and_user_experience.md)).

### 4.4 Web search plugin attachment (guarded by reasoning effort)

When the selected model supports OpenRouter web search, the pipe can attach the OpenRouter web search plugin (`plugins: [{"id":"web", ...}]`) when the **OpenRouter Search** toggle is enabled for the request (per chat, or enabled by default via Default Filters).

Important compatibility behavior:
- If effective `reasoning.effort` is `minimal`, the pipe skips attaching the web search plugin (to avoid provider/tool incompatibility errors).

For the full User Interface story (Open WebUI Web Search vs OpenRouter Search, and why OpenRouter Search overrides Web Search), see:
[Web Search (Open WebUI) vs OpenRouter Search](web_search_owui_vs_openrouter_search.md).

### 4.5 Output token cap selection

When `USE_MODEL_MAX_OUTPUT_TOKENS=True`, the pipe can set `max_output_tokens` using the provider-advertised `max_completion_tokens` derived from the catalog. When it is disabled, the pipe clears `max_output_tokens` so provider limits and upstream defaults apply.

### 4.6 Auto context trimming (transforms)

When `AUTO_CONTEXT_TRIMMING=True`, the pipe may attach OpenRouter’s `middle-out` transform by setting `transforms=["middle-out"]` only when the request does not already specify `transforms`.

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
| `OpenRouterModelRegistry.api_model_id(model_id)` | provider slug or `None` | Maps the normalized/sanitized model ID back to the provider’s original ID for outbound API calls. |

---

## 6. Failure modes and operator signals

- Missing API key: catalog load fails with a configuration error and the pipe cannot expose models.
- Refresh failures with cache: the pipe can continue serving cached models; logs will show a warning about serving cached catalog data.
- Refresh failures with no cache: the error propagates, and requests that require the catalog cannot proceed.
- Empty catalog: the registry treats an empty model list as an error.

Operator guidance:
- Treat catalog failures like an upstream connectivity/credential issue first (API key, network egress, proxy/gateway, OpenRouter availability), then inspect logs for the last refresh failure.
