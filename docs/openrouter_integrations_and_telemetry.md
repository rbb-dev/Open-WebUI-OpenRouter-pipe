# OpenRouter Integrations & Telemetry

This document covers behaviors that are specific to the OpenRouter Responses API integration: request shaping, model catalog behavior, OpenRouter-specific parameters, and optional telemetry exports.

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [Identifiers](request_identifiers_and_abuse_attribution.md) · [Errors](error_handling_and_user_experience.md)

---

## 1. Endpoint and OpenRouter headers

- The pipe targets the OpenRouter base URL configured by `BASE_URL` (default `https://openrouter.ai/api/v1`), using the `/responses` endpoint.
- Requests include OpenRouter-identifying headers:
  - `X-OpenRouter-Title` (pipe title)
  - `HTTP-Referer` (default project URL; can be overridden via `HTTP_REFERER_OVERRIDE` — must be a full URL including scheme, and must not carry a CR, LF or NUL inside it; one that does is ignored, warned about, and replaced by the default). Sent on every request to an openrouter.ai host, the catalogue, per-model endpoint and maker-page refresh reads included; not on user- or model-supplied asset downloads or the GitHub self-update check.
  - `X-OpenRouter-Categories` (pipe category identifier for analytics attribution)
- Optional provider beta headers:
  - For Anthropic models (`anthropic/...` slugs and `~anthropic/...` router aliases), when `ENABLE_ANTHROPIC_INTERLEAVED_THINKING=True`, the pipe sends `x-anthropic-beta: interleaved-thinking-2025-05-14` to opt into Claude “interleaved thinking” streaming (reasoning may appear in multiple blocks during a single answer).

---

## 2. Request shaping and schema enforcement

### 2.1 Allowed request fields (OpenRouter Responses allowlist)
Before sending requests to OpenRouter, the pipe filters request bodies to the allowlist below (`ALLOWED_OPENROUTER_FIELDS`). Any keys not in this list are dropped. Explicit `null` values are also dropped because OpenRouter rejects `null` for optional fields. A key that is on the list can still be rewritten rather than forwarded: `store` is the one such key, and the pipe forces it to `false` (see its row).

| Field | Purpose / notes |
| --- | --- |
| `model` | Primary model for the request (selected in Open WebUI). |
| `models` | Fallback model list (OpenRouter will try these if the primary `model` fails). **The array is the pipe's to own**: a caller-supplied `models` key in the request body is discarded before the request is sent, on every leg, and the only writer is the deployment's own `model_fallback`. Every entry the deployment does supply is checked against the same model restriction valves as the primary — `MODEL_ID`, `FREE_MODEL_FILTER`, `TOOL_CALLING_FILTER` and `ZDR_MODELS_ONLY` — and an entry one of them refuses ends the turn with the `MODEL_RESTRICTED_TEMPLATE` card naming that id, before any request is sent. |
| `input` | Responses input payload (constructed from Open WebUI messages and content blocks). |
| `instructions` | Additional instructions passed through to OpenRouter when present. |
| `metadata` | OpenRouter metadata map; sanitized to string→string with length/pair constraints (invalid entries dropped). |
| `stream` | Enables streaming mode. |
| `max_output_tokens` | Output token cap. The pipe may set/omit this depending on `USE_MODEL_MAX_OUTPUT_TOKENS` and routing decisions. A caller's own cap — `max_output_tokens`, or `max_completion_tokens` renamed to this name — occupies the field first, so only one cap is ever sent. |
| `temperature` | Sampling parameter. See the family rule on the `top_k` row: a value that is not a finite number is dropped, not refused. |
| `top_k` | Sampling parameter, sent as a JSON integer on both endpoints: a float is rounded, a numeric string is parsed and then rounded. **The rule for the whole sampling family** — `temperature`, `top_p`, `top_k`, `min_p`, `top_a`, `presence_penalty`, `frequency_penalty` and `repetition_penalty`: a value that is not a finite number contributes nothing, and the request is still sent. A boolean (dropped, not read as `1`/`0`), a list, a dict, a non-numeric string, a blank string, and any non-finite number — `inf`, `-inf`, `nan`, `1e400`, `float("inf")`, `float("nan")` — are each dropped. A finite number or numeric string is forwarded as its float; `0` is a real level, not an absent one, and survives as `0.0`. `top_k` additionally reaches the wire as a JSON integer: `0` is forwarded as the documented disabled value, and a negative value is forwarded unchanged and refused by OpenRouter. |
| `top_p` | Sampling parameter. See the family rule on the `top_k` row: a value that is not a finite number is dropped, not refused. |
| `reasoning` | Reasoning configuration; only recognized subfields are forwarded (unknown keys dropped). The recognized set is `context`, `effort`, `enabled`, `exclude`, `max_tokens`, `mode` and `summary`; a `reasoning` object that holds none of them is dropped rather than sent as `{}`. `max_tokens` is sent as a JSON integer: a numeric string is read as the number it spells and a fractional value rounds, while a value that expresses no number — letters, a blank string, a list, a boolean, or a non-finite number such as `inf` or `1e400` — is dropped rather than forwarded, and never refused. |
| `include_reasoning` | OpenRouter's deprecated alias for `reasoning.exclude`. The pipe adds it only when every model the request can be served by lists it, checked against the primary `model` — where a routing variant with no catalog entry of its own is read from its base model's entry — and against every fallback in `models`; an id the catalogue does not know counts as not listing it. OpenRouter forwards the key to whichever model ends up serving, and a provider that does not know the parameter rejects the whole request: measured 2026-09-26, `/responses` with `models=[openai/gpt-4.1-mini]` and `include_reasoning: false` answered HTTP 400 from Azure, `Unknown parameter: 'include_reasoning'.` When the flag has to go and its value was `false` and the primary lists `reasoning`, thinking off is carried by merging that effort into the `reasoning` object the request already carries, so its `max_tokens` and `summary` survive, which the same session measured as accepted with zero reasoning tokens. |
| `tools` | Tool definitions (merged from Open WebUI registry tools plus Open WebUI Direct Tool Servers when present). |
| `tool_choice` | Tool selection directive. |
| `plugins` | Legacy plugin configuration (retained for backward compatibility). |
| `truncation` | Context-truncation strategy — what OpenRouter does when a request exceeds the model's context window. **The pipe owns this field on both arms of `AUTO_CONTEXT_TRIMMING`**, whatever the request body carried (`apply_context_transforms`): off pins `truncation: "disabled"`, on sends no `truncation` at all and carries the `context-compression` plugin instead. It is on the `/responses` allowlist only, so `/chat/completions` never carries it. See §4. |
| `preset` | OpenRouter preset slug for pre-configured LLM settings (system prompts, provider routing, parameters). See [Model Variants & Presets](model_variants_and_presets.md#presets). |
| `text` | Response text configuration (`text.format` for structured outputs / JSON mode; `text.verbosity` when supported). `text.verbosity` is the `/responses` spelling of the verbosity setting and top-level `verbosity` is the `/chat/completions` one; **either spelling may be set by the caller** (a Custom Parameter, or Open WebUI's own field), and a caller who sets one is never overwritten by the pipe's own value. For one request the provider receives the same verbosity on both endpoints, under each endpoint's own spelling — including on the `AUTO_FALLBACK_CHAT_COMPLETIONS` leg, where a refused `/responses` re-sends the turn on `/chat/completions` without changing the depth; when both spellings are set and disagree, `text.verbosity` is the one honoured, on both endpoints. The pipe writes the value itself only from `REASONING_EFFORT=xhigh` on models whose catalog entry lists `verbosity`. Housekeeping requests are governed by the task valve, not this one. |
| `parallel_tool_calls` | Tool parallelism hint (when supported). |
| `user` | OpenRouter user identifier (optional; controlled by identifier valves). |
| `session_id` | OpenRouter session identifier (optional; controlled by identifier valves). |
| `trace` | OpenRouter Broadcast observability metadata; a JSON object forwarded as-is to OpenRouter's tracing destinations (Datadog, Langfuse, LangSmith, webhook, etc.). This pipe supports `openrouter_trace` as an OWUI convenience mapping to this field. |
| `transforms` | Deprecated OpenRouter transforms list (forwarded only if explicitly supplied). Automatic context trimming now uses the `context-compression` plugin — see §4. |
| `stop_server_tools_when` | Conditions under which OpenRouter stops a server-side tool run. The pipe writes a `max_cost` condition from the `SERVER_TOOLS_MAX_COST_USD` valve, which bounds a request's tool spend, and strips the field again when the tools it guarded are removed from the request. An internal Fusion turn is several such requests, one per panel member plus the judge and the synthesis, each sent the whole cap, so the ceiling for that turn is that multiple. OpenRouter applies this field as an **override** of `max_tool_calls`: wherever both are sent, the cost condition decides and the step count is ignored. It is sent on **both** endpoints. |
| `background` | Run the request in the background (OpenRouter extension). |
| `frequency_penalty` | Frequency penalty for sampling (-2 to 2). See the family rule on the `top_k` row: a value that is not a finite number is dropped, not refused. |
| `image_config` | Image generation configuration (OpenRouter extension). |
| `include` | Response include directives, forwarded verbatim. The values OpenRouter's responses API accepts are `file_search_call.results`, `message.input_image.image_url`, `computer_call_output.output.image_url`, `reasoning.encrypted_content` and `code_interpreter_call.outputs`; anything else is answered with a 400. `include` is a Responses field and is dropped when the request routes to `/chat/completions`. The pipe reads usage from the response's own usage object, so there is nothing to ask for. |
| `max_tool_calls` | Limit the number of tool call iterations (OpenRouter extension). Carried on **both** endpoints: whatever value the request body holds is sent on `/responses` and on `/chat/completions`, unchanged, and the pipe neither substitutes nor validates it — the int-rounding the sampling fields get is not applied here. A body that never carries it grows no such key. The step cap is the fallback bound, not the strongest one: `stop_server_tools_when` overrides it wherever both are set, so with the `SERVER_TOOLS_MAX_COST_USD` cap at its default `0.0` (no cap, field absent) this is the only bound on the server-tool loop. That cap is a per-request bound on one OpenRouter call, and an internal Fusion turn is one such call per panel member plus the judge and the synthesis, so the turn's tool ceiling is that multiple of the cap (see the `stop_server_tools_when` row). This is the top-level request field and is unrelated to the per-panelist budget Fusion writes inside `plugins`. |
| `modalities` | Output modalities (e.g. `["text"]`, `["text", "audio"]`). |
| `presence_penalty` | Presence penalty for sampling (-2 to 2). See the family rule on the `top_k` row: a value that is not a finite number is dropped, not refused. |
| `prompt` | Direct prompt input (alternative to structured `input`). |
| `prompt_cache_key` | Cache key for prompt caching (OpenRouter extension). |
| `prompt_cache_options` | Request-level prompt-caching controls. `mode: "explicit"` turns OpenAI-managed breakpoints off, so only blocks you marked with `prompt_cache_breakpoint` are cached; `ttl` requests a cache duration (e.g. `"30m"`). Carried on **both** endpoints, unchanged — the pipe neither substitutes nor validates it, so the shape you send is the shape OpenRouter receives. The request-level field is OpenAI-only: OpenRouter translates the block-level markers for the provider a turn routes to, and leaves this one alone. |
| `safety_identifier` | Safety configuration identifier (OpenRouter extension). Pipe-owned, not a free pass-through: a caller's value is removed, and the field is sent only when `SEND_END_USER_ID` is on, carrying the same value as `user`. `/responses` only. |
| `service_tier` | Service tier selection (default: `"auto"`). |
| `store` | Privacy signal — send `false` to tell the provider not to store your prompt/completion data. OpenRouter only accepts `false` for this field, so the pipe **forces** it: any other value a request body carries is rewritten to `false` before the request is sent, and the first such rewrite in a worker is logged at WARNING naming the field and the substitute. A body that never mentions `store` does not grow the key. |
| `top_logprobs` | Number of top log probabilities to return (0–20). |
| `provider` | Provider routing preferences — `only`, `ignore` and `order`. The pipe builds this dict itself from the `openrouter_provider_only` / `_ignore` / `_order` custom parameters described in §2.5. |
| `route` | Routing strategy, forwarded to OpenRouter. |
| `debug` | OpenRouter debug options, forwarded. |
| `thinking_config` | Gemini-family native thinking configuration; the pipe builds it for Gemini models. |
| `web_search_options` | Native web-search options (e.g. `search_context_size`). Forwarded on `/responses`; removed on **both** endpoints when `disable_native_websearch` is set. |

Operational note:
- `previous_response_id` is deliberately **not** forwarded. OpenRouter's `/responses` endpoint is stateless and answers a non-null `previous_response_id` with a 400, so the pipe drops it and sends the whole conversation in `input` instead; the first drop in a worker is logged as a WARNING naming the field and the substitute. A host running Open WebUI with `ENABLE_RESPONSES_API_STATEFUL=true` is relying on the field — that setting replaces the outgoing messages with the system message plus the prior output, so the history the host stopped sending is the history this pipe cannot recover. Leave stateful mode off against OpenRouter.
- The pipe always constructs a canonical "Responses-style" request first, then converts it to a Chat Completions payload only when needed (forced endpoint selection or automatic fallback).
- Some parameters are Chat-only (for example `stop`, `seed`, `logprobs`, `preset`). These are ignored when calling `/responses`, but are preserved so they can be used if the request is sent via `/chat/completions`. The token cap is not one of them: a caller's `max_completion_tokens` is sent as itself on `/chat/completions` and renamed to `max_output_tokens` on `/responses`, so the same cap reaches either endpoint under that endpoint's own spelling and only `max_output_tokens` is sent on that one.
- `max_tool_calls`, `service_tier`, `prompt_cache_key` and `prompt_cache_options` are carried on **both** endpoints: whatever value the request carries is sent on `/responses` and on `/chat/completions`, unchanged, and the pipe neither substitutes nor validates it. A request that carries none of them grows none of them. `prompt_cache_key` is OpenRouter's sticky-routing cache key; it is only a fallback for `session_id` and `x-session-id`, so a `SEND_CACHE_SESSION_ID` install (the default) routes on `session_id` instead. The request-level `prompt_cache_options` stays OpenAI-only, per OpenRouter's own routing — the block-level markers are the ones translated between providers, and a `cache_control` `ttl` is dropped toward OpenAI rather than becoming this field's.
- An explicit `max_completion_tokens` beats the pipe's automatic ceiling on **both** endpoints: when a request carries one, the value `USE_MODEL_MAX_OUTPUT_TOKENS` would otherwise fill is not sent, so only one token cap is ever on the wire. A `max_completion_tokens` below 1 is dropped exactly as a `max_tokens` below 1 is, so the valve's ceiling applies to it if enabled. A cap written as a numeric string is read as that number wherever it appears — `max_tokens`, `max_output_tokens` or `max_completion_tokens` — so all three spellings make the same decision about the same value, and a fractional value rounds.
- When a `preset` parameter is present in the request body, the pipe automatically forces `/chat/completions` because presets only work on that endpoint. For presets that work with `/responses`, use the VARIANT_MODELS approach with `@preset/slug` syntax instead. See [Model Variants & Presets](model_variants_and_presets.md#presets).

### 2.2 Advanced Model Parameters (per-model overrides)
Open WebUI allows per-model “Advanced Model Parameters” (stored on the model in `model.params`, typically under `model.params.custom_params`). This pipe supports a small set of per-model parameters that:

- affect **how requests are shaped** before they are sent to OpenRouter, and/or
- affect what the pipe will **auto-sync into Open WebUI model metadata** (icons, capability checkboxes, integrations toggles, descriptions).

The pipe accepts the following Advanced Model Parameters:

| Advanced param | Type | Applies to | What it does |
| --- | --- | --- | --- |
| `model_fallback` | `str` (CSV) or a JSON array | Requests | Convenience mapping for OpenRouter fallbacks: converts a CSV list or a JSON array into the OpenRouter `models` array (order-preserving, de-duplicated), **replacing** any `models` key the request already carried rather than merging with it. Open WebUI JSON-parses a custom param before the pipe sees it, so the two admin spellings produce one behaviour. Every parsed entry goes through the same model restriction check as the primary (`MODEL_ID` and the model filters), and a refused entry ends the turn with the `MODEL_RESTRICTED_TEMPLATE` card naming it, so a caller who POSTs this key cannot put a restricted model on the wire. Open WebUI cannot tell an admin's value from a caller's body key, which is why the entries are gated rather than trusted. |
| `openrouter_trace` | `str` (JSON) | Requests | Convenience mapping for OpenRouter Broadcast observability: parses a JSON object and writes it as the OpenRouter `trace` field. See §2.4. |
| `disable_native_websearch` | `bool-ish` | Requests | Prevents OpenRouter native web search from being used for this model by stripping OpenRouter web search server tools and related request fields. |
| `openrouter_provider_ignore` | `str` (CSV) | Requests | Comma-separated provider slugs to exclude from routing. Maps to `provider.ignore`. See §2.5. |
| `openrouter_provider_only` | `str` (CSV) | Requests | Comma-separated provider slugs to restrict routing to. Maps to `provider.only`. See §2.5. |
| `openrouter_provider_order` | `str` (CSV) | Requests | Comma-separated ordered list of preferred provider slugs. Maps to `provider.order`. See §2.5. |
| `disable_model_metadata_sync` | `bool-ish` | Model metadata sync | Master switch: the pipe will not “manage” this model’s settings at all (capabilities, icon, description, auto-attached integrations, default-on integrations). |
| `disable_capability_updates` | `bool-ish` | Model metadata sync | Prevents overwriting Open WebUI capability checkboxes (`meta.capabilities`). |
| `disable_image_updates` | `bool-ish` | Model metadata sync | Prevents overwriting the model icon/profile image (`meta.profile_image_url`). |
| `disable_description_updates` | `bool-ish` | Model metadata sync | Prevents overwriting the model description (`meta.description`). |
| `disable_web_tools_auto_attach` | `bool-ish` | Model metadata sync | Prevents auto-attaching the **OR Web Tools** integration toggle (filter id) for this model. |
| `disable_web_tools_default_on` | `bool-ish` | Model metadata sync | Prevents auto-enabling **OR Web Tools** by default for this model (prevents seeding `meta.defaultFilterIds`, and releases one the pipe had already seeded for it). |
| `disable_direct_uploads_auto_attach` | `bool-ish` | Model metadata sync | Prevents auto-attaching the **Direct Uploads** integration toggle (filter id) for this model, and prevents removing it from `meta.defaultFilterIds`. |

Notes:
- “bool-ish” accepts JSON booleans (`true/false`) and common string/int forms (`"true"`, `"1"`, `"on"`, etc.).
- `disable_native_websearch` has an alias key: `disable_native_web_search`.

### 2.3 `model_fallback` → OpenRouter `models`
OpenRouter supports a primary `model` plus a fallback list `models` (array). Open WebUI does not expose a first-class UI for OpenRouter’s `models` field, so this pipe supports a convenience parameter:

- Custom param: `model_fallback` (a CSV string or a JSON array)
- Pipe behavior:
  - Reads the value as a list of model ids, whether it arrived as a CSV string or as an array. Each element is trimmed, skipped if it is not a string, and de-duplicated (order-preserving), exactly as a CSV part is.
  - Checks every entry against the same model restriction valves as the primary model — `MODEL_ID`, `FREE_MODEL_FILTER`, `TOOL_CALLING_FILTER` and `ZDR_MODELS_ONLY` — and refuses the whole turn with the `MODEL_RESTRICTED_TEMPLATE` card naming the offending id, before any request is sent.
  - **Replaces** the request's `models` list rather than merging with it. A caller-supplied `models` array is not the deployment's to send: it is discarded, so the ids that reach the wire are the ones the deployment supplied.
  - Writes the final list to `models` (fallback list only).
  - Removes `model_fallback` from the outgoing OpenRouter payload.

A task request (`__task__`, such as Open WebUI's title generation) follows the same chain, on the
task leg as on the chat legs.

Open WebUI JSON-parses a custom param before the pipe sees it, so both of these admin spellings
reach the pipe as a list and produce the same `models` array:

```text
openai/gpt-5,openai/gpt-5.1,anthropic/claude-sonnet-4.5
```

```json
["openai/gpt-5", "openai/gpt-5.1", "anthropic/claude-sonnet-4.5"]
```

An entry that is not a string is skipped rather than stringified, so a typo inside the array
cannot become a model id. A value that is neither a string nor a list or tuple (a number, or a
JSON object) is ignored.

### 2.4 `openrouter_trace` → OpenRouter `trace` (Broadcast / Observability)

OpenRouter's [Broadcast](https://openrouter.ai/docs/guides/features/broadcast) feature sends traces from your LLM requests to observability backends — Datadog, Langfuse, LangSmith, or any webhook endpoint. The `trace` field is a top-level JSON object in the request body that lets you attach tracing metadata to each request.

This pipe supports `openrouter_trace` as a per-model custom parameter that maps to the OpenRouter `trace` field:

- Custom param: `openrouter_trace` (JSON string)
- Pipe behavior:
  - Parses the JSON string into a dict (Open WebUI's `custom_params` handler auto-parses JSON strings, so the value typically arrives as a dict already).
  - If the value is still a raw JSON string (edge case), the pipe attempts `json.loads()` as a fallback.
  - Merges with any existing `trace` dict already in the request (model-level values take precedence on key conflicts).
  - Writes the merged dict to `trace` in the outgoing OpenRouter payload.
  - Removes `openrouter_trace` from the outgoing payload.
  - Invalid values (non-JSON strings, arrays, empty dicts, non-dict types) are silently ignored.

#### OpenRouter `trace` field reference

OpenRouter recognizes the following special keys in the `trace` object. These map to OpenTelemetry (OTLP) span attributes in the Broadcast payload:

| Key | OTLP Mapping | Description |
| --- | --- | --- |
| `trace_id` | `traceId` | Group multiple requests into a single trace. |
| `trace_name` | Span `name` | Custom name for the root span. |
| `span_name` | Span `name` | Name for intermediate spans in the hierarchy. |
| `generation_name` | Span `name` | Name for the LLM generation span. |
| `parent_span_id` | `parentSpanId` | Link to an existing span in your trace hierarchy. |

Any additional keys are forwarded as custom metadata, appearing in the OTLP payload under the `trace.metadata.*` namespace.

The `user` and `session_id` request fields (configured separately via identifier valves) map to `user.id` and `session.id` in the OTLP span attributes.

#### How to configure in Open WebUI

1. Open the model editor in Open WebUI (Admin → Models → Edit).
2. Scroll to **Advanced Parameters** → **Custom Parameters**.
3. Click **Add** and create a key-value pair:

   ```
   Key:   openrouter_trace
   Value: {"trace_name": "Customer Support Pipeline", "generation_name": "support-chat"}
   ```

4. Save the model.

All requests using this model will now include the `trace` field in the OpenRouter payload.

#### Example: Full trace configuration

The following custom parameter value demonstrates all OpenRouter trace fields plus custom metadata:

```json
{"trace_id": "order_processing_001", "trace_name": "Order Processing Pipeline", "generation_name": "Extract Order Details", "order_id": "ORD-12345", "priority": "high"}
```

This results in the following OpenRouter request body (other fields omitted):

```json
{
  "model": "openai/gpt-4o",
  "messages": [{"role": "user", "content": "Process this order..."}],
  "trace": {
    "trace_id": "order_processing_001",
    "trace_name": "Order Processing Pipeline",
    "generation_name": "Extract Order Details",
    "order_id": "ORD-12345",
    "priority": "high"
  }
}
```

In the Broadcast payload sent to your observability backend (Datadog, Langfuse, webhook, etc.), the custom metadata keys appear under `trace.metadata.*`:

```json
{
  "resourceSpans": [{
    "scopeSpans": [{
      "spans": [{
        "attributes": [
          {"key": "gen_ai.request.model", "value": {"stringValue": "openai/gpt-4o"}},
          {"key": "trace.metadata.order_id", "value": {"stringValue": "ORD-12345"}},
          {"key": "trace.metadata.priority", "value": {"stringValue": "high"}}
        ]
      }]
    }]
  }]
}
```

#### Merging behavior

If a request already contains a `trace` field (for example, from a client-side tool or another transform), the `openrouter_trace` model parameter is **merged** into it. Model-level values take precedence when the same key exists in both:

```
Existing trace:       {"trace_id": "from_client", "session": "abc123"}
openrouter_trace:     {"trace_name": "My Pipeline", "trace_id": "from_model"}
→ Final trace:        {"trace_id": "from_model", "trace_name": "My Pipeline", "session": "abc123"}
```

#### Direct `trace` passthrough

If you send `trace` directly in the request body (without using the `openrouter_trace` model parameter), it passes through the pipe's allowlist filter unchanged — no special configuration required. The `openrouter_trace` convenience parameter is only needed when you want to configure trace metadata per-model in Open WebUI's model settings.

### 2.5 `disable_native_websearch` → disable OpenRouter web search server tool
Open WebUI can attach per-model custom parameters (model settings → `custom_params`). This pipe supports a boolean custom parameter to **disable OpenRouter’s built-in web search** for specific models.

- Custom param: `disable_native_websearch` (bool-ish; accepts `true/false`, `1/0`, etc.)
  - Alias: `disable_native_web_search`
- Pipe behavior (when truthy):
  - Removes `tools` entries with `{"type": "openrouter:web_search"}`.
  - Removes legacy `plugins` entries with `{"id": "web"}`.
  - Removes `web_search_options` when present.
  - Holds on the internal-Fusion panel, judge and final member calls too, and on the housekeeping-task path (title, tags, summary, follow-ups), which runs its own request path.

This is useful when OpenRouter Web Tools is enabled by default (via the model’s Default Filters / `AUTO_DEFAULT_WEB_TOOLS_FILTER`) but you want to block provider-native web search on selected models.

Internal-Fusion reach: each member call is built from scratch as `{model, stream, messages}`, so a flag on the outer request would otherwise never reach the member’s own applier. The resolved value is therefore carried on the internal invocation and copied into the member body under the primary spelling, resolved from the outer body under the primary name first and the alias second, mirroring the precedence the outer applier applies. The carried field defaults to `None`, not `False`: `None` means the outer request never mentioned the flag, and that is why the member-body guard tests `is not None` rather than truthiness — a `False` default would put `disable_native_websearch=False` on every member body of requests that never asked about it. The value is copied, never written back into `pipe_meta`, so one member’s tools are not shared by reference across the fan-out.

Note: Open WebUI’s built-in **Web Search** is separate (OWUI-native) and is not controlled by this parameter.
See: [Web Search (Open WebUI) vs OpenRouter Web Tools](web_search_owui_vs_openrouter_search.md).

### 2.6 `disable_model_metadata_sync` → opt out of all model metadata sync for a model
This is a per-model “master kill switch” for the pipe’s Open WebUI model metadata sync behavior.

- Custom param: `disable_model_metadata_sync` (bool-ish)
- Pipe behavior (when truthy):
  - Skips all metadata writes for that model, even if sync valves are enabled (`UPDATE_MODEL_*`, `AUTO_ATTACH_*`, `AUTO_DEFAULT_*`).
  - This preserves operator-edited settings (icons, descriptions, capabilities, and integration toggle defaults).

### 2.7 `disable_capability_updates` → preserve capability checkboxes
- Custom param: `disable_capability_updates` (bool-ish)
- Pipe behavior (when truthy):
  - Leaves `meta.capabilities` as-is for that model (no checkbox overwrites), even when `UPDATE_MODEL_CAPABILITIES=True`.
  - A preserved `citations` value is honoured, not merely stored: on the chat path the pipe publishes no tool-derived source for a model whose row says citations are off, so Open WebUI persists no `sources` for that message either.

### 2.8 `disable_image_updates` → preserve the model icon
- Custom param: `disable_image_updates` (bool-ish)
- Pipe behavior (when truthy):
  - Leaves `meta.profile_image_url` as-is for that model, even when `UPDATE_MODEL_IMAGES=True`.

### 2.9 `disable_description_updates` → preserve the model description
- Custom param: `disable_description_updates` (bool-ish)
- Pipe behavior (when truthy):
  - Leaves `meta.description` as-is for that model, even when `UPDATE_MODEL_DESCRIPTIONS=True`.

### 2.10 `disable_web_tools_auto_attach` → preserve the OpenRouter Web Tools toggle wiring
- Custom param: `disable_web_tools_auto_attach` (bool-ish)
- Pipe behavior (when truthy):
  - The pipe will not add/remove the OpenRouter Web Tools filter id in `meta.filterIds` for that model, even when `AUTO_ATTACH_WEB_TOOLS_FILTER=True`.
  - As a consequence, default-on seeding is also avoided (the pipe will not mark a filter as default unless it is attached).

### 2.11 `disable_web_tools_default_on` → stop seeding, and release a default already seeded
- Custom param: `disable_web_tools_default_on` (bool-ish)
- Pipe behavior (when truthy):
  - The pipe will not seed OpenRouter Web Tools into `meta.defaultFilterIds` for that model, even when `AUTO_DEFAULT_WEB_TOOLS_FILTER=True`.
  - A default the pipe had already seeded for that model is released on the next sync — that is, the next catalogue or settings change, or a worker restart, or saving the model: the id is removed from `meta.defaultFilterIds` and the seeding latch is cleared in the same write, so the row is not rewritten on any sync after that. It comes back on the next such sync if the parameter is later removed.
  - The OpenRouter Web Tools toggle may still be auto-attached if `AUTO_ATTACH_WEB_TOOLS_FILTER=True` and the model supports it — the release is of the *default*, never of the wiring, so the toggle stays in the Integrations menu to be switched back on.

### 2.12 `disable_direct_uploads_auto_attach` → preserve the Direct Uploads toggle wiring
- Custom param: `disable_direct_uploads_auto_attach` (bool-ish)
- Pipe behavior (when truthy):
  - The pipe will not add/remove the Direct Uploads filter id in `meta.filterIds` for that model, even when `AUTO_ATTACH_DIRECT_UPLOADS_FILTER=True`.
  - Nor will it remove the id from `meta.defaultFilterIds`: the opt-out is hands-off, so a default the pipe never detached stays ticked.

### 2.13 `disable_image_filter_auto_attach` → preserve the native image filter wiring
- Custom param: `disable_image_filter_auto_attach` (bool-ish)
- Pipe behavior (when truthy):
  - The pipe will not attach the native image filters to that image-output model in `meta.filterIds`, even when `AUTO_ATTACH_IMAGE_FILTERS=True`.
  - Turning the valve off detaches what the pipe attached; `AUTO_DEFAULT_IMAGE_FILTERS=False` instead clears the default it seeded in `meta.defaultFilterIds` and leaves the filter attached. Independently of that valve, a superseded image panel the pipe's retirement sweep switched off is dropped from `meta.defaultFilterIds` as well as `meta.filterIds` on the next sync, whether or not the ownership record ever named it.

### 2.14 `disable_video_gen_auto_attach` → preserve the Video Generation filter wiring
- Custom param: `disable_video_gen_auto_attach` (bool-ish)
- Pipe behavior (when truthy):
  - The pipe will not attach the per-model video generation filter to that video model in `meta.filterIds`, even when `AUTO_ATTACH_VIDEO_FILTERS=True`.
  - Turning the valve off detaches what the pipe attached; `AUTO_DEFAULT_VIDEO_FILTERS=False` instead clears the default it seeded in `meta.defaultFilterIds` and leaves the filter attached.

### 2.15 Provider routing custom parameters → OpenRouter `provider` dict

OpenRouter routes models through multiple infrastructure providers (e.g. OpenAI direct, Azure, Together). Some providers have different content filtering, latency, or pricing. These custom parameters let you control provider selection per-model without needing the full filter-based provider routing system.

- Custom params:
  - `openrouter_provider_ignore` (CSV string) — providers to exclude
  - `openrouter_provider_only` (CSV string) — providers to restrict to
  - `openrouter_provider_order` (CSV string) — preferred provider priority

- Pipe behavior:
  - Parses each CSV into a validated list of lowercase provider slugs
  - Invalid slugs (not matching `^[a-z0-9-]+(/[a-z0-9-]+)?$`) are silently dropped with a log warning
  - Merges into any existing `provider` dict on the request (from filter-injected routing or ZDR enforcement) — never overwrites
  - For list fields, existing entries come first, then new entries are appended (deduplicated)
  - Removes the custom param keys from the outgoing payload
  - Works on both Responses API and Chat Completions paths

- Provider slug format: lowercase, alphanumeric + hyphens, optional single slash segment (e.g. `azure`, `openai`, `together`, `deepinfra/turbo`)

#### Example: Skip Azure for GPT-5.4

Useful when Azure's content filter blocks responses that OpenAI direct would allow.

Open WebUI custom parameter:
```
openrouter_provider_ignore: azure
```

Resulting OpenRouter request field:
```json
{"provider": {"ignore": ["azure"]}}
```

#### Example: Prefer OpenAI then Together, never Azure

```
openrouter_provider_order: openai, together
openrouter_provider_ignore: azure
```

Result:
```json
{"provider": {"order": ["openai", "together"], "ignore": ["azure"]}}
```

#### Example: Force a single provider

```
openrouter_provider_only: openai
```

Result:
```json
{"provider": {"only": ["openai"]}}
```

#### Notes

- `only` and `ignore` are independent fields in the OpenRouter API. Setting both is technically valid but uncommon — `only` restricts the allowlist, `ignore` removes from it.
- These params are simple CSV strings that survive OWUI's Advanced Parameters editor re-serialization (unlike nested JSON objects which can be mangled).
- For more advanced per-model provider routing with UI dropdowns, see the [Provider Routing Filters](openrouter_provider_routing.md) system.

### 2.16 Prompt-cache session affinity (`session_id`)

To maximize prompt-cache hits, the pipe sends OpenRouter a stable per-conversation `session_id` so every turn of a conversation routes to the same provider, keeping that provider's prompt cache warm. A Fusion panel member is one turn of that conversation and is pinned like any other. The value is an opaque `HMAC-SHA256(`WEBUI_SECRET_KEY`, or the deprecated `WEBUI_JWT_SECRET_KEY` it falls back to (a default, so an empty primary is not a fallback), chat_id)` digest — never the raw chat id — and costs zero tokens. Any client-supplied `session_id` in the top-level wire field is dropped.

A call that carries no `chat_id` — the plain API route — is pinned instead on `HMAC-SHA256(`WEBUI_SECRET_KEY`, or the deprecated `WEBUI_JWT_SECRET_KEY` it falls back to (a default, so an empty primary is not a fallback), "api-session:" + session_id)`, the caller's own `session_id` under a namespaced prefix. The raw caller value never reaches the wire, and the prefix namespaces the fallback so it cannot collide with a real chat id unless the caller deliberately chooses a `chat_id` of the form `api-session:<x>` — chat ids are caller-supplied, so that one shape does collide, and a chatless caller sending `session_id=<x>` gets the same pin. The fallback fires only when the caller actually sends a `session_id`; that caller is the only party that knows what "the same conversation" means for its own traffic. A call that sends `parent_id: null` is given a fresh conversation id by Open WebUI on every turn, so it is pinned to a new value each time and its cache never warms; the pipe does not override a real `chat_id`.

Controlled by `SEND_CACHE_SESSION_ID` (default **on**); skipped when neither the primary name nor its deprecated fallback is set. A manually pinned `provider.order` overrides it.

---

## 3. Model catalog and capability-aware routing

The pipe loads OpenRouter’s `/models` catalog and caches it to drive capability-aware behavior (for example: vision inputs, web search eligibility, reasoning toggles, token caps).

Key valves:
- `MODEL_ID` (default `auto`) controls whether the pipe exposes the full catalog or a comma-separated allowlist. An entry holding `*`, `?` or `[` is a glob matched case-insensitively against every catalog row, and a `!`-prefixed entry excludes, applied after the includes. A restricting allowlist that resolves to nothing — including an include pattern that matches no row — now publishes nothing and refuses every request, instead of failing open to the full catalog. The one exception is a value of only commas or spaces, which is read as blank and imports the whole catalog.
- `MODEL_CATALOG_REFRESH_SECONDS` controls refresh cadence.
- `USE_MODEL_MAX_OUTPUT_TOKENS` controls whether the pipe states an explicit output allowance for requests that carry none, the smaller of the advertised ceiling and half the model's context window, or the advertised ceiling alone when OpenRouter publishes no context length.

### 3.1 Open WebUI model metadata sync (icons + capabilities)

Open WebUI stores additional per-model UI metadata (capabilities checkboxes and profile images) in its own Models table. This pipe can **automatically sync that metadata** for the OpenRouter models it exposes.

What it syncs (best-effort):
- `meta.profile_image_url`: downloads the model icon — or reads it inline when the catalogue serves it as a `data:` URL, in which case nothing is downloaded — converts it to **PNG**, and stores it as a `data:image/png;base64,...` data URL (Open WebUI does not process remote image URLs here). Either way the icon must be within 2 MiB decoded and 25 Mpx to be stored; an icon over either bound is skipped with a `debug` log. The source URL is stamped in the pipe's own metadata under `image_source_url`, alongside the `image_source_kind` recording which of the two sources it was — the frontend catalogue (`frontend`) or the maker's page (`maker`) — so an icon whose source URL has not changed is not downloaded again, and for a model taking its maker's logo the maker's page is not re-fetched either; a row stamped from the frontend catalogue is always re-fetched, so a model whose catalogue icon is retired still reaches its maker's logo. A change to the image at the same URL is therefore not picked up, and a card keeps its old icon — a hand-picked one included — until its source URL changes. The other half of that promise is the catalogue read itself: a model whose catalogue icon is *unreadable this pass* keeps the icon it has, and no maker page is scraped on its behalf, so an outage neither overwrites a stored icon nor costs one scrape per maker. On the first pass after upgrading from a version that did not record the kind, each maker's page is fetched once more and the rows converge again.
  - SVG icons are rasterized to PNG (requires `cairosvg`).
  - Other images are converted to PNG (requires `Pillow`).
- `meta.description`: writes the model’s user-facing description from OpenRouter’s `/models` catalog when present.
- `meta.capabilities`: writes the Open WebUI capability checkboxes (for example `vision`, `file_upload`, `web_search`, `image_generation`).
  - The `web_search` checkbox mirrors the model’s published `web_search` pricing, and is filled in only where the model has no setting of its own yet, so a value you set by hand is kept on every later refresh.
  - `meta.builtinTools` carries the pipe’s `Files` untick, which follows from its own `capabilities.file_context` untick on the same pass rather than from any operator action. The keys it filled are named under `meta.openrouter_pipe.builtin_tool_defaults`, so a `Files` box you set by hand is neither recorded as the pipe’s nor changed by it.

Data sources / egress:
- Fetches `https://openrouter.ai/api/frontend/v1/catalog/models` (no auth) to discover icons and descriptions.
- When provider routing valves list models, fetches `https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints` (no auth) for each listed model to build the full provider list for the routing filter dropdowns. That sweep is bounded as a whole and not only per read, like the two icon sweeps below, so the length of the routing list does not appear in how long one metadata pass takes; a slug the budget abandons keeps the provider data it already had and has its routing row left exactly as it is.
- Downloads each icon URL (absolute or relative to `https://openrouter.ai`) and may fall back to a maker page OpenGraph image (`https://openrouter.ai/<maker>`); while the stamped source is unchanged, neither the page nor its image is fetched again. The maker-page fallback is a fallback for a catalogue read that ANSWERED and had no icon for the model: a pass whose read did not answer scrapes no maker page at all and leaves stored icons as they are, so an outage does not turn into one request per maker per refresh window. Both icon sweeps — the maker pages and the downloads — are bounded as a whole and not only per read, so the size of the catalogue does not appear in how long one metadata pass takes; a leg the budget abandons keeps whatever the row already had, is named in a `WARNING` that separates it from a leg that ran and found nothing, and is offered again on the next sync-key change — a new catalogue fetch, a settings change such as a valve edit, or a restart.

All three of those fetches carry the `HTTP-Referer` attribution header, because each names an openrouter.ai host. An icon download is a different matter: its URL can name any host (a provider's favicon is fetched from a third-party CDN), so it carries none, and the vetted transport re-decides per redirect hop rather than replaying the header onto whatever a `Location` names.

Controls:
- `UPDATE_MODEL_IMAGES` (default `True`): enable/disable profile image sync.
- `UPDATE_MODEL_DESCRIPTIONS` (default `False`): enable/disable model description sync.
- `UPDATE_MODEL_CAPABILITIES` (default `True`): enable/disable capability checkbox sync.
- `NEW_MODEL_ACCESS_CONTROL` (default `admins`): sets the access grants applied when the pipe **inserts** a new OpenRouter model overlay into Open WebUI (existing access grants are preserved on update). One pass uses one value for all of its rows, and it is read once per pass, so a pass already running when you save the valve finishes under the value it started with. Saving it also schedules a pass, because the valve is a term of the sync key, so the value you just chose applies to every model the pipe imports next. Use `admins` to create no access grants (private), which relies on Open WebUI's `BYPASS_ADMIN_ACCESS_CONTROL` for admin access.
Per-model opt-outs:
- `disable_model_metadata_sync`: disables all metadata sync for the model.
- `disable_image_updates`, `disable_description_updates`, `disable_capability_updates`: disable specific metadata fields for the model.

Operational note:
- This sync updates Open WebUI’s Models table using Open WebUI’s own helper APIs (not raw SQL), but it is still a **write** to Open WebUI’s model metadata. Those helpers swallow a database fault and answer that they did not save, so the pipe reads that answer: a row the helper reports as not saved is counted as a failure for that row, named in the log, and retried. Disable the valves if you want to manage model icons/capabilities manually — and note that the `web_search` and `Citations` boxes are left alone either way, so an admin’s tick or untick of either now survives every refresh and is honoured on the chat path rather than only stored.

---

## 4. Auto context trimming (context-compression plugin)

When `AUTO_CONTEXT_TRIMMING=True`, the pipe enables OpenRouter’s `context-compression` plugin by appending `{"id": "context-compression"}` to the request’s `plugins` array **only when no context-compression plugin is already present**. (This replaces the deprecated top-level `transforms=["middle-out"]` shape; `middle-out` is now the plugin’s internal compression engine.)

The same valve owns the request's `truncation` field on `/responses`, in **both** states and whatever the request body carried:

| `AUTO_CONTEXT_TRIMMING` | `truncation` on the wire | `context-compression` plugin |
| --- | --- | --- |
| `False` | `"disabled"` | absent |
| `True` | absent | present |

A `truncation` the caller supplied is overridden in both states — off because the pin is the valve's answer, on because sending `disabled` beside the compression plugin would ship two instructions that contradict each other. `truncation` is on the `/responses` allowlist only, so this table does not describe `/chat/completions`.

Operational guidance:
- Leave this enabled if you want long prompts to degrade gracefully instead of failing due to context limits.
- Disable it if you manage context compression explicitly in your deployment.

---

## 5. Tooling and plugins

- Web search:
  - When the **OpenRouter Web Tools** toggle is enabled for the request (per chat, or enabled by default via Default Filters), and the selected model/provider supports OpenRouter web search, the pipe attaches OpenRouter web search as a server tool (`tools: [{"type": "openrouter:web_search", ...}]`). One entry per type per request, and where a filter and the request body both name the type, the filter's parameters are the ones sent.
  - The filter's admin valves (`WEB_SEARCH_MAX_RESULTS`, `WEB_SEARCH_ENGINE`, etc.) control search parameters.
- Response healing:
  - The OpenRouter response-healing plugin is intentionally **not** exposed by this pipe.
  - We prefer fail-fast behavior for malformed/invalid outputs so errors remain visible.
- Tools:
  - Tool schemas are built from Open WebUI’s `__tools__` registry plus any selected Open WebUI **Direct Tool Servers**.
  - Direct Tool Servers are executed client-side via Open WebUI (the pipe emits `execute:tool` via Socket.IO).
- Tool schema strictness:
  - `ENABLE_STRICT_TOOL_CALLING` is off by default: OpenAI rejects a strict tool whose parameters contain a free-form
    object, and Open WebUI's built-in `ask_user` is one, so strict mode is opt-in.
  - When `ENABLE_STRICT_TOOL_CALLING=True`, the pipe strictifies tool schemas for more predictable function calling.
  - When `ENABLE_STRICT_TOOL_CALLING=True` and the pipe runs the tool (not `Open-WebUI` mode,
    and not under `ask` approval in a saved chat), the tools it advertises on the Responses route
    carry `strict: true`. A tool the pipe does not strictify is sent there with an explicit
    `strict: false`, which is insurance rather than a workaround: measured on 2026-10-02 against
    OpenRouter's own `/responses`, a function tool with no `strict` key is treated as non-strict
    (200), and only `strict: true` enforces the strict-schema rules (400 on a non-strict schema),
    so OpenRouter applies no tool-level default today. A tool that arrived already in Responses
    shape keeps whatever `strict` it carried, and one that carried none is sent with none.
    OpenRouter strips `strict` from a tool on Anthropic models unless the
    `structured-outputs-2025-11-13` header is passed; the pipe does not pass that header, so on
    those models the field does not take effect and the call routes normally.

See also: [Tooling & Integrations](tooling_and_integrations.md).
And: [Web Search (Open WebUI) vs OpenRouter Web Tools](web_search_owui_vs_openrouter_search.md).

---

## 6. User-visible telemetry (status and usage)

### Final usage status line
When `SHOW_FINAL_USAGE_STATUS` resolves True — the reader's own copy where they have set it, otherwise the site default an administrator chooses — the pipe emits a final status line that can include timing, token counts, and any OpenRouter charge above zero the upstream usage payload carries.

The valve exists on both `Valves` and `UserValves`, and the merge overrides only the fields a reader has actually set. A generation reported at exactly zero prints the token counts the reply carried and the timing where that merge resolves True; it prints the timing alone where it resolves False, and also where the reply carried no token counts at all, since only the counters OpenRouter actually sends are ever read.

This is intended as user-visible telemetry and operator troubleshooting signal (not as an authoritative billing record).

### Which usage keys sum, and which do not
A turn that costs several upstream calls -- the tool loop, or a Fusion panel -- has several `usage` payloads, and they are merged into the one block the turn reports. The merge is an **allow-list**, mirroring Open WebUI's `merge_usage`: only per-token and per-cost quantities are added together, and every other key takes the incoming value.

Summed:

| Keys | Why |
| --- | --- |
| `input_tokens`, `output_tokens`, `total_tokens` | Per-token quantities. |
| `cost`, `total_cost`, `input_cost`, `output_cost`, `prompt_cost`, `completion_cost` | Per-call money quantities. |
| `prompt_tokens`, `completion_tokens` | The OpenRouter spellings of the first two, summed on their own account. |
| `input_tokens_details`, `output_tokens_details`, `prompt_tokens_details`, `completion_tokens_details` | The `*_tokens_details` maps are merged key by key and their numbers added -- this is what makes `cached_tokens` and `reasoning_tokens` cumulative. |
| `cache_discount` | A per-response money quantity, not a rate, and **signed**: a provider that reports a cache *write* sends a negative value, because the write costs more than a plain prompt. Nothing downstream may take its absolute value — the Usage tab floors this at zero, so the charge appears once, in `cost`, and never as a saving. The pipe reads it from the *merged* accumulator, so dropping it would under-report the Usage tab's `cache_savings` card by construction. |
| `turn_count`, `function_call_count` | Written once per generation before the merge, **on the streamed path only** -- a non-streamed turn carries neither key, so "written once per generation" does not hold everywhere; summing them gives the turn's generation and tool-call counts. `function_call_count` counts the `function_call` items in each round's completed response, summed over the turn's rounds. A host-side tool the model ran itself (`openrouter:web_search` and its siblings) is not one of them: the pipe did not execute it and cannot size a loop budget against it. One output item carrying several argument objects counts once, because the figure is what the model asked for -- the pipe splits it and runs it several times, and a budget is sized against the ask. |

Last-wins (the incoming value replaces the accumulated one):

- `cache_discount_pct` and every other rate or percentage. Adding three generations of a 20 % discount to get 60 % is meaningless.
- `server_tool_use_details` and `cost_details`. Both are **per-request** figures, not per-token. The OpenRouter schema's "do not sum the two" applies *within* one such block, not across generations, and neither is a per-token quantity, so the whole map is replaced -- the same rule Open WebUI applies. **On a Fusion panel this makes the reported figure the last member's, not the panel total** (`cost_details.upstream_inference_cost` over a five-member panel goes from a total to the fifth member's value). No reader in this repository or in Open WebUI's backend consumes either key, but the reduction is a decision rather than an oversight. Only these two keys change: the panel's `input_tokens` is still the full sum.
- Any other key, including flags such as `stream`. A `bool` is never added: three generations of `stream: true` stay `true` and not `3`.

A payload carrying **both** `input_tokens` and `prompt_tokens` gets both summed, so the quantity is counted twice in the block. Through the pipe's own request path the two spellings never coexist -- `core/costs.py` builds a fresh dict and copies one -- but `merge_usage_stats` is public API with no shape restriction, so a caller passing a raw chat-completions `usage` block sees it. This is unchanged from earlier releases and is pinned by a test rather than accidental.

---

## 7. Optional telemetry export: cost snapshots to Redis

When enabled, the pipe can write per-request usage snapshots into Redis for downstream analytics and chargeback workflows.

Valves:
- `COSTS_REDIS_DUMP` (default `False`) enables/disables the feature.
- `COSTS_REDIS_TTL_SECONDS` (default `900`) controls retention in Redis.

Behavior (as implemented):
- Writes occur only while `ENABLE_REDIS_CACHE` is on and Redis is available (`_redis_enabled=True`). The valve is authoritative at the write in the very turn that turns it off: that turn's snapshot is not written, and neither is any later one, for as long as the valve is off.
- Snapshots are written only when all required fields are present:
  - Open WebUI user ID (`guid`)
  - user `email`
  - user `name`
  - model ID
  - OpenRouter usage payload
- Keys are namespaced per pipe identifier:

```text
costs:{pipe_namespace}:{user_id}:{uuid}:{epoch_seconds}
```

Payload fields include:
- `guid`, `email`, `name`, `model`, `usage`, `ts`
- On a snapshot for a chat's answer, a background task, a picture or a clip: `kind` (`generation` or `task`), `chat_id` and `message_id`. A picture and a clip are the generation of that turn in that chat, so they carry `kind: "generation"` like an answer does. A temporary chat's snapshot carries no `chat_id` or `message_id`; saved and channel chats keep both. A call that arrives with no chat at all — the plain API route — keeps its `kind` and names neither id.

Snapshots an earlier release wrote for a temporary chat are not rewritten; they expire after `COSTS_REDIS_TTL_SECONDS`.

Privacy guidance:
- These snapshots include user identity fields (email/name) from Open WebUI. Treat Redis access as sensitive, apply TTLs, and avoid using this feature if you do not need per-user cost attribution.

---

## 8. Persistence and encryption defaults (OpenRouter workloads)

OpenRouter reasoning outputs can be large, so persistence controls matter for operational cost and storage growth.

Relevant valves:
- `PERSIST_REASONING_TOKENS` (system default `conversation`)
- `ARTIFACT_ENCRYPTION_KEY` (enables encryption when set)
- `ENCRYPT_ALL` (default `True`; when encryption is enabled, encrypts all artifacts vs reasoning-only; a row already stored encrypted stays encrypted, in the table and in the cache, whatever it is set to, and the cache entry is written in the form the row was stored in, on the buffered flush as well as on a read)
- `ENABLE_LZ4_COMPRESSION` (default `True`, when `lz4` is available. If a compression attempt raises, the failure is logged once and compression stays off for the rest of the process, so a broken native library is not retried on every request; only a restart re-arms it.)

See [Persistence, Encryption & Storage](persistence_encryption_and_storage.md) for the full behavior description.

---

## Related topics

- [Valves & Configuration Atlas](valves_and_configuration_atlas.md)
- [Request Identifiers & Abuse Attribution](request_identifiers_and_abuse_attribution.md)
- [Model Catalog & Routing Intelligence](model_catalog_and_routing_intelligence.md)
- [Tooling & Integrations](tooling_and_integrations.md)
- [Error Handling & User Experience](error_handling_and_user_experience.md)
