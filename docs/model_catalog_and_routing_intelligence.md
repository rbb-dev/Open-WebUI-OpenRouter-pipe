# model catalog & routing intelligence

**file:** `docs/model_catalog_and_routing_intelligence.md`
**related source:** `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py:543-1100`, `Pipe._pick_model`, `Pipe._apply_model_capabilities`

This document explains how the pipe imports OpenRouter's `/models` catalog, derives per-model capability maps, and applies that metadata when routing user requests. Understanding this layer is essential before tweaking valves such as `MODEL_ID`, `MODEL_CATALOG_REFRESH_SECONDS`, or any of the reasoning/modality toggles.

---

## 1. catalog ingestion pipeline

1. **Sanitization** (`sanitize_model_id`). Provider IDs such as `author/model/variant` are converted into dot-friendly identifiers (`author.model.variant`) so Open WebUI can safely store them. Date suffixes (e.g., `-2024-08-01`) are also stripped via `ModelFamily.base_model` to ensure capability lookups remain stable across dated releases.
2. **Fetching** (`OpenRouterModelRegistry._refresh`). The registry issues a GET to `${BASE_URL}/models` using the decrypted `API_KEY`. The raw JSON, including descriptions and pricing, is stored verbatim (`raw_specs`) so downstream logic can inspect any field the provider exposes.
3. **Derivation** (`_derive_features`, `_derive_capabilities`). For each model the registry derives:
   * `features`: `function_calling`, `reasoning`, `reasoning_summary`, `web_search_tool`, `image_gen_tool`, `vision`, `audio_input`, `video_input`, `file_input`.
   * `capabilities`: the booleans Open WebUI expects (`vision`, `file_upload`, `web_search`, `image_generation`, plus always-on `code_interpreter`, `citations`, `status_updates`, `usage`).
   * `supported_parameters`: a frozen set of flags that indicate which Responses fields the provider recognizes (`tools`, `parallel_tool_calls`, `reasoning`, etc.).
   * `max_completion_tokens`: pulled from `top_provider.max_completion_tokens` whenever it exists.
4. **Caching**. The registry caches both a sorted list of models (for UI selectors) and the derived specs keyed by normalized ID. `ModelFamily.set_dynamic_specs` shares the spec map with helper predicates like `ModelFamily.supports`.
5. **Refresh cadence**. `MODEL_CATALOG_REFRESH_SECONDS` (default 3600) controls how long the cache is considered fresh. Every failure increases `_consecutive_failures`, which drives an exponential backoff via `_record_refresh_failure`. If the first fetch fails there is no cache to fall back to and the error propagates to the caller.

> **Tip:** If you see the warning `OpenRouter catalog refresh failed... Serving N cached model(s)`, the registry is intentionally serving stale data. Use DEBUG logs to inspect `_last_error` and `_last_error_time` before retrying.

---

## 2. selecting a model per request

1. **User/valve inputs**. `MODEL_ID` can be a comma-separated allowlist (`gpt-4o-mini,gpt-4.1`) or the literal `"auto"` (default) to expose everything OpenRouter returns. Users may also specify a `model` field directly in the Open WebUI UI; the pipe normalizes it through `ModelFamily.base_model` before forwarding to OpenRouter.
2. **Capability guardrails**. The chosen model"s features drive multiple switches:
   * If `vision` is false, `_transform_messages_to_input` drops `input_image` blocks and emits a status update so the user knows attachments were ignored.
   * If `function_calling` is false, the pipe refuses to inject Open WebUI tools and disables multi-step tool loops regardless of per-user settings.
   * If `web_search_tool` is true and the `ENABLE_WEB_SEARCH_TOOL` valve is enabled, the OpenRouter `web` plugin is appended automatically.
   * If `supported_parameters` lacks `parallel_tool_calls`, the pipe clears `parallel_tool_calls` on the outgoing request even when the UI requested it.
3. **Max tokens**. When `USE_MODEL_MAX_OUTPUT_TOKENS` is true, the derived `max_completion_tokens` is forwarded as `max_output_tokens` so the provider enforces its own limit. Otherwise the pipe lets Open WebUI"s `max_output_tokens` input pass through unmodified.
4. **Reasoning toggles**. Setting `ENABLE_REASONING=False` either globally or per user removes the `reasoning` payload from the Responses request and disables persistence for reasoning markers. `ModelFamily.supports("reasoning")` is checked before enabling the feature; unsupported models simply skip the block regardless of valves.
5. **Task contexts bypass allowlists (and honor a dedicated effort valve).** Open WebUI schedules several internal tasks (title generation, tag suggestions, etc.) that execute via `__task__` payloads. Those flows still ignore the `MODEL_ID` whitelist so housekeeping keeps working, *but* when the referenced model belongs to this pipe we now consult the `TASK_MODEL_REASONING_EFFORT` valve and (when the effort is `minimal`) skip the OpenRouter `web` plugin entirely. Streaming preferences remain whatever Open WebUI requested; the new default `low` simply gives tasks a small reasoning budget with web-search still available when permitted. Drop to `minimal` for the fastest housekeeping path or raise the valve for richer task workflows. Interactive chats and API calls continue to respect the allowlist.
6. **Web-search plugin honors minimal reasoning.** The pipe only attaches the OpenRouter `web` plugin when the model advertises the `web_search_tool` feature *and* the request’s effective `reasoning.effort` is higher than `minimal`. When the effort resolves to `minimal` (either because the user requested it or a valve forced it) we skip the plugin and emit a debug log (`Skipping web-search plugin because reasoning.effort is set to 'minimal'`). This prevents OpenRouter 400s such as `The following tools cannot be used with reasoning.effort 'minimal': web_search`. To re-enable search, bump the effort to `low` or higher or disable the auto-attach valve entirely.

---

## 3. capability helpers you can rely on

| Helper | Returns | Common uses |
| --- | --- | --- |
| `ModelFamily.supports(feature, model_id)` | `True/False` for derived features | Drop/enable image, audio, video, web search, reasoning, or tool blocks. |
| `ModelFamily.capabilities(model_id)` | Dict mirroring `OpenRouterModelRegistry._derive_capabilities` | Injects capability metadata into Open WebUI so the UI toggles vision/file upload switches correctly. |
| `ModelFamily.supported_parameters(model_id)` | Frozen set of provider-supported request parameters | Used when pruning unsupported fields (`parallel_tool_calls`, `response_format`, etc.). |
| `ModelFamily.max_completion_tokens(model_id)` | Provider-advertised output limit | Enables `USE_MODEL_MAX_OUTPUT_TOKENS`. |
| `OpenRouterModelRegistry.api_model_id(sanitized_id)` | Original provider ID | Ensures the outbound API call references the exact model slug OpenRouter expects. |

All helpers expect sanitized IDs (period-separated). Internally they normalize everything via `ModelFamily.base_model`, so suffixes such as `.latest` or `pipe-id.` prefixes do not break lookups.

---

## 4. routing failure modes & UX

* **Missing catalog**: If `_refresh` fails before any models are cached, the exception propagates and the pipe emits a status message telling the operator to check credentials or network reachability. Requests abort before sending any user data to OpenRouter.
* **Unknown model**: When a user references a model that was not imported (e.g., `MODEL_ID` allowlist), the pipe responds with an actionable status (`model '<id>' is not available; allowed values: ...`) instead of falling back silently.
* **Capability drift**: If OpenRouter changes capabilities between refreshes, the pipe respects whatever the provider reports during the next successful refresh. There is no baked-in allowlist; every decision is data-driven. If you detect mismatches, temporarily override behavior via valves while coordinating with OpenRouter support.
* **Backoff exhaustion**: When repeated failures push `_next_refresh_after` far into the future, you can manually reset the cache (restart the pipe) or drop `OpenRouterModelRegistry._specs` via a REPL. The registry logs `_consecutive_failures` and the backoff ETA to help debugging.

---

## 5. extending the catalog logic

When adding new derived metadata:

1. Update `_derive_features` or `_derive_capabilities` to compute the new flag.
2. Store it inside `specs[norm_id]` so `ModelFamily` helpers can surface it.
3. Document the new behavior here and mention whether a valve controls it.
4. If the flag affects UI toggles, also emit it through `OpenRouterModelRegistry.list_models()` so Open WebUI can show the capability badge.

Never hard-code provider-specific rules elsewhere in the pipe. By keeping every routing decision rooted in the catalog, the manifold stays resilient to OpenRouter changes and keeps behavior explainable.
