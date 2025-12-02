# openrouter-specific integrations & telemetry

**file:** `docs/openrouter_integrations_and_telemetry.md`
**related source:** `openrouter_responses_pipe/openrouter_responses_pipe.py`

This note collects the features that are unique to the OpenRouter variant of the Responses manifold. Use it whenever you need a quick reminder of the “extras” we ship for OpenRouter beyond the baseline OpenAI manifold.

---

## 1. status telemetry & usage string

* Location: `openrouter_responses_pipe/openrouter_responses_pipe.py` inside `_emit_completion`.
* When a request finishes, we emit a status message like:
  ```text
  Time: 80.31s  4007.6 tps | Cost $1.163295 | Total tokens: 323103 (Input: 1274, Output: 321829, Reasoning: 315177)
  ```
* The string is tailored to OpenRouter’s billing model:
  * Pulls cost from the Responses API usage payload (USD with six decimals).
  * Includes throughput (`tokens / elapsed_seconds`).
  * Breaks out reasoning tokens explicitly because OpenRouter exposes them per request.
* Controlled by valves:
  * `SHOW_FINAL_USAGE_STATUS` (system + user) toggles the entire final status block.
  * `STREAMING_UPDATE_PROFILE` and related valves influence how quickly interim token counts appear.

---

## 2. catalog-smart routing against /models

* The pipe imports the entire `/models` payload (not just names) via `OpenRouterModelRegistry` and derives rich metadata:
  * `supported_parameters` (parallel tool calls, reasoning, response_format, etc.).
  * Capability flags for `vision`, `audio_input`, `video_input`, `web_search_tool`, etc.
  * Provider-reported `max_completion_tokens` so we can honor provider caps via `USE_MODEL_MAX_OUTPUT_TOKENS`.
* `ModelFamily.supports("feature", model_id)` is used throughout the pipe to enable/disable behaviors automatically (no separate allowlists).
* Valve hook: `MODEL_ID=auto` imports the entire OpenRouter catalog, or you can specify a comma-separated shortlist.

---

## 3. automatic plugin + MCP wiring

* When `ENABLE_WEB_SEARCH_TOOL=True` and the selected model advertises the paid `web_search` capability, the pipe automatically attaches OpenRouter’s `web` plugin. Users get web search without remembering to toggle anything in the UI.
* `REMOTE_MCP_SERVERS_JSON` lets admins define workspace-wide MCP servers in valve JSON; the pipe merges those definitions with registry tools and OpenRouter-provided tools so they all appear in the same Responses request.
* Strict tool schemas: when `ENABLE_STRICT_TOOL_CALLING=True`, OpenRouter tools are sent with `strict: true` + JSON Schema pruning so function calling behaves predictably across OpenRouter providers.

---

## 4. OpenRouter-friendly concurrency & retries

* `_RetryableHTTPStatusError` + Tenacity waiters honor OpenRouter’s `Retry-After` headers, so we do not hammer `/responses` after throttling.
* Admission control is tuned for OpenRouter’s concurrency envelope: `MAX_CONCURRENT_REQUESTS`, `_QUEUE_MAXSIZE`, and per-user breakers prioritize fast failure (HTTP 503) instead of allowing overloads.
* `_stream_responses` merges usage blocks across multi-step tool loops, matching OpenRouter’s nested usage payload format.

---

## 5. status CSS patch for Open WebUI

* Valve: `ENABLE_STATUS_CSS_PATCH` (default True).
* On every request, `_process_transformed_request` injects a CSS snippet via `__event_call__` so multi-line status descriptions (like the final usage string above) are fully visible in Open WebUI’s sidebar. This mirrors the OpenAI manifold but we keep it optional.

---

## 6. persistence tuned for OpenRouter artifacts

* Reasoning payloads are much larger on OpenRouter (hundreds of thousands of tokens), so we:
  * Default to encrypting reasoning items (`ENCRYPT_ALL` optional for everything else).
  * Use LZ4 compression whenever available to keep DB/Redis traffic sane.
  * Prune tool outputs via `_prune_tool_output` once they fall outside `TOOL_OUTPUT_RETENTION_TURNS`.
* Redis write-behind is namespaced to the pipe ID (which includes “openrouter” by default) so multi-tenant OpenRouter deployments can run multiple copies of the pipe safely.

---

## 7. multimodal guardrails aligned with OpenRouter limits

* Remote download + base64 size caps default to `50 MB`, matching OpenRouter’s documentation for image/file attachments.
* `_download_remote_url` enforces SSRF bans and honors OpenRouter’s expected MIME types (`image/png`, `image/jpeg`, `image/webp`, `image/gif`).
* Valve `MAX_INPUT_IMAGES_PER_REQUEST` is clamped to `<=20`, matching OpenRouter’s current limit on `input_image` blocks.

---

## 8. developer-friendly valve catalog

* The valve names mirror OpenRouter semantics (e.g., `REASONING_EFFORT`, `REASONING_SUMMARY_MODE`, `WEB_SEARCH_MAX_RESULTS`).
* Each valve defaults to the OpenRouter-recommended value (reasoning enabled, web search enabled, Redis cache auto-on for multi-worker deployments).
* See `docs/valves_and_configuration_atlas.md` for the full table, but this document highlights the ones that make the OpenRouter experience richer out of the box.

---

## 9. user-facing 400 error templates

* Valve: `OPENROUTER_ERROR_TEMPLATE`.
* When OpenRouter returns a 400 (prompt too long, moderation block, provider invalid request, etc.) the manifold surfaces a Markdown card instead of crashing the stream. The template is completely admin-configurable and now supports Handlebars-style blocks: wrap optional sections in `{{#if variable}} ... {{/if}}` and they render only when the underlying value is truthy. Individual lines still auto-drop if a placeholder resolves to an empty string, so legacy templates continue to work.
* Token limit callouts use `{context_limit_tokens}` / `{max_output_tokens}` and are only populated when the upstream error text contains the OpenRouter hint `or use the "middle-out"` **and** the registry exposed limits for that model.
* Recommended workflow:
  1. Copy the default template from `Pipe.Valves.OPENROUTER_ERROR_TEMPLATE`.
  2. Add or remove Markdown sections as needed for your organization (e.g., internal ticket instructions, support URLs).
  3. Keep `{request_id_reference}` somewhere in the template so operators can correlate user reports with OpenRouter telemetry.

### Template variables

You can mix and match the following placeholders with `{{#if ...}}` blocks:

| Placeholder | Description |
| --- | --- |
| `{heading}` | Friendly label such as `Anthropic: claude-3`. |
| `{detail}` / `{sanitized_detail}` | Raw provider message (sanitized version escapes backticks). |
| `{provider}` | Provider name supplied by OpenRouter. |
| `{model_identifier}`, `{requested_model}`, `{api_model_id}`, `{normalized_model_id}` | Various model ids (Open WebUI, requested alias, canonical slug). |
| `{openrouter_code}` | Numeric OpenRouter error code. |
| `{upstream_type}` | Provider-specific error classification. |
| `{reason}` | User-friendly summary of the failure. |
| `{request_id}` / `{request_id_reference}` | Provider request id plus a short string you can drop into support tickets. |
| `{moderation_reasons}` | Bullet-list of moderation flags (already prefixed with `-`). |
| `{flagged_excerpt}` | Redacted moderation snippet. |
| `{raw_body}` | Trimmed raw provider response body. |
| `{context_limit_tokens}` / `{max_output_tokens}` | Formatted token limits (available only when OpenRouter suggests “middle-out”). |
| `{include_model_limits}` | Boolean helper used with `{{#if include_model_limits}} ... {{/if}}` to gate the entire limit block. |
| `{openrouter_message}` / `{upstream_message}` | Original strings from OpenRouter/provider before formatting. |

### Example error template with macros

```markdown
### 🚫 {heading} could not process this request

### Error: `{sanitized_detail}`

- **Requested model:** `{requested_model}`
- **Provider:** `{provider}`
- **OpenRouter code:** `{openrouter_code}`
- **Request ID:** `{request_id}`

{{#if include_model_limits}}
**Model limits**
- Context window: {context_limit_tokens} tokens
- Max output tokens: {max_output_tokens} tokens
Please trim your prompt or enable the middle-out transform.
{{/if}}

{{#if moderation_reasons}}
**Moderation reasons**
{moderation_reasons}
{{/if}}

{{#if flagged_excerpt}}
**Flagged text excerpt**
~~~
{flagged_excerpt}
~~~
{{/if}}

{{#if raw_body}}
**Raw provider response**
~~~
{raw_body}
~~~
{{/if}}

Need help? Share `{request_id_reference}` with the on-call engineer.
```

Example of the template:
<img width="1021" height="765" alt="image" src="https://github.com/user-attachments/assets/533554b0-cd2c-442c-bd15-29055ca43860" />


Because each placeholder is optional, empty sections disappear automatically—e.g., if there are no moderation hits, the `{{#if moderation_reasons}}` block is skipped before the card is sent to the user.

---

## 10. auto context trimming (message transforms)

* Valve: `AUTO_CONTEXT_TRIMMING` (default True). When enabled, `_apply_context_transforms` automatically sets `ResponsesBody.transforms = ["middle-out"]` unless the caller already provided a `transforms` list.
* Why: OpenRouter’s [message transforms guide](https://github.com/openrouter-team/openrouter-docs/blob/main/manual/docs/guides/features/message-transforms.md) recommends applying `middle-out` to large prompts so requests degrade gracefully instead of failing with 400 “prompt too long”.
* How to override:
  * Set the valve to `False` if you want full manual control per request (e.g., you supply `transforms` yourself or use a custom trimming strategy).
  * Provide `transforms` inside the request body to bypass the auto-injected array for a single call.
