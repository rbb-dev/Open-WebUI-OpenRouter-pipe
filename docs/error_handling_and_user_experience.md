# Error Handling & User Experience

This document describes how the pipe renders user-facing error messages, how operators correlate errors with backend logs, and how to customize error templates safely via valves.

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [OpenRouter integration](openrouter_integrations_and_telemetry.md) · [Security](security_and_encryption.md)

---

## Overview

The pipe aims to avoid raw exception traces surfacing in the Open WebUI UI. Instead, it:

- Logs technical details server-side with a unique `error_id` for correlation.
- Emits user-facing Markdown messages (templates) with actionable remediation guidance.
- Includes contextual identifiers (session/user) when available.

There are two main error rendering paths:

1. **OpenRouter “request rejected” templates** (OpenRouter HTTP status handling and parsed provider errors).
2. **Generic templated errors** (timeouts/connectivity/internal failures handled by `_emit_templated_error`).

---

## Error IDs and enriched context

For templated errors, the pipe generates:
- `error_id`: 16 hex characters (`secrets.token_hex(8)`)
- `timestamp`: ISO 8601 UTC timestamp
- `session_id` and `user_id` (when available)
- `support_email` and `support_url` (from valves `SUPPORT_EMAIL` and `SUPPORT_URL`)

Operator logs include the `error_id`, and templates can include it in user-facing text for support correlation.

---

## Exception/status mapping (which template is used)

### A) OpenRouter “request rejected” errors (status-aware templates)

The pipe selects an OpenRouter template based on the HTTP status:

| Status | Template valve |
| --- | --- |
| `401` | `AUTHENTICATION_ERROR_TEMPLATE` |
| `402` | `INSUFFICIENT_CREDITS_TEMPLATE` |
| `408` | `SERVER_TIMEOUT_TEMPLATE` |
| `413` | `PAYLOAD_TOO_LARGE_TEMPLATE` |
| `429` | `RATE_LIMIT_TEMPLATE` |
| `>= 500` | `SERVICE_ERROR_TEMPLATE` |
| other / default | `OPENROUTER_ERROR_TEMPLATE` |

“Other / default” is every remaining case, including a missing status: `403`, `404`, `422`, a `400` for a malformed reference URL, and a provider moderation refusal all render `OPENROUTER_ERROR_TEMPLATE`. Anything written into that template is therefore read by failures that have nothing in common beyond not having a template of their own, so advice specific to one cause belongs in a `{{#if}}` block rather than in the template's unconditional text.

**A `403` is not one thing, and a content decision does not pause the user's background work.** OpenRouter uses the status for at least four different answers: a key it rejected, a key that is valid but lacks permission for the model, a provider content policy, and a guardrail that declined the prompt before it reached any provider. The first is the person's to fix by signing in again; the other three are not, so only the first arms the sixty-second credential pause that otherwise leaves the next chats titled "Chat" with no error shown. A guardrail block is the case that carries no type at all — its only marker is the `patterns` the guardrail matched — and it is the case a status-only reading blames on the key. Rendering is unaffected either way: every one of these renders `OPENROUTER_ERROR_TEMPLATE` and shows OpenRouter's own reason.

These templates are used for the `OpenRouterAPIError` path (and for certain HTTP status errors that are converted into an OpenRouter error object by reading the response body best-effort).

**A failure reported inside a reply that has already started uses the same table.** OpenRouter commits `200 OK` as soon as a provider accepts a request, so a rate limit, an overloaded or unreachable provider, a provider timeout, an exhausted balance or a sign-in failure occurring after that point arrives inside the reply instead of as a status. The pipe reads the kind of failure OpenRouter names there - `error.metadata.error_type` on Chat Completions, the top-level `error_type` on Responses, the same fields the model-limits section below uses - and renders the template for the status that kind is documented with, so the person reads the message written for that failure rather than the rejected-request one. A kind the pipe does not recognise falls back to the status reported alongside it, and then to `OPENROUTER_ERROR_TEMPLATE`. One consequence matters when editing `SERVER_TIMEOUT_TEMPLATE`: a provider timeout reported this way is documented as `504`, so it renders `SERVICE_ERROR_TEMPLATE`. The timeout template is reached by a `408`, whether that is the reply's own status or the code reported inside the reply.

**Every caller uses the table.** Status selection is the only thing that chooses one of these templates; no call site can pass a template that overrides it. The chat orchestrator, the outer request handler, image generation, video generation, and a streaming failure arriving after output has begun all render the same template for the same status.

**One exception, and it matters for the wording of `AUTHENTICATION_ERROR_TEMPLATE`.** When the pipe cannot read its own OpenRouter API key it renders that template directly. Nothing was sent, so there is no status and no rejection by OpenRouter — the cause is local: the key setting is blank, or the value stored in it was encrypted under a `WEBUI_SECRET_KEY` that has since changed and can no longer be decrypted. The `401` shown on the card in that case is a display value the pipe supplies, not a status any server returned. Wording written for that box therefore has to fit a local configuration fault as well as a key OpenRouter rejected, which is why the built-in text says the request was not authorised "or the pipe could not read one" rather than asserting that OpenRouter refused the credentials. It is the only template in the table above reached this way; every other template is chosen by status alone, from the table in section B below.

**OpenRouter's own request reference travels with the card.** When a rejection carries one, every template in the table above renders it on its own row, separate from the `error_id` the pipe generates: the pipe's id correlates the pipe's logs, and OpenRouter's is what its support can look up. A rejection that carries none renders no such row. The reference is unavailable on the paths where no request reached OpenRouter — a key the pipe could not read, and a `5xx` raised by the connection itself — and on a failure event that carries no id (a Responses `response.error` or plain `error` event); on Chat Completions OpenRouter's generation id is the reference, also available as `error_chunk_id`. A template edited to show the reference should therefore keep the row inside a conditional, as the built-in text does.

**Clearing a template box and saving restores its built-in text.** Every error template valve behaves this way: leave the box empty (or containing only spaces or newlines), save, and the pipe writes that valve's factory default back, so reopening the Config tab shows the original wording ready to edit again. Only the valve that was cleared is affected. This is how an operator recovers from an edit that went wrong, since the built-in text is not otherwise visible in the interface. The same restore applies when valves are edited through Open WebUI's own Functions valve panel.

### B) Generic templated errors (network/5xx/internal)

When a chat reply's call to OpenRouter fails without being rejected, or anything else goes wrong in the chat reply loop, the pipe picks one of these templates:

| Condition | Template valve |
| --- | --- |
| A timeout before any answer text has arrived | `NETWORK_TIMEOUT_TEMPLATE` |
| A connection that cannot be opened or drops, or a stream that sent nothing on every attempt, before any answer text has arrived | `CONNECTION_ERROR_TEMPLATE` |
| A timeout, a failed or dropped connection, or a stream that sent nothing, after answer text arrived earlier in the reply | `STREAM_INTERRUPTED_TEMPLATE`, appended after the kept text |
| Any other exception | `INTERNAL_ERROR_TEMPLATE` |

The timeout template's `timeout_seconds` is the limit that ran out: `HTTP_CONNECT_TIMEOUT_SECONDS` while connecting, `HTTP_SOCK_READ_SECONDS` while waiting for data, or `HTTP_TOTAL_TIMEOUT_SECONDS` for the whole request. The connection template's `error_type` names the failure (for example `ClientConnectorError`). A stream that closes early without an error, once any of its events has arrived, also gets `STREAM_INTERRUPTED_TEMPLATE`; see [Streaming Pipeline & Emitters](streaming_pipeline_and_emitters.md). Picture-only image models, video models and the panel, judge and final-answer calls inside internal Fusion report failures in their own way, and a required internal file that cannot be read shows its own message.

---

## Template rendering rules

Templates are Markdown strings with:
- Placeholder variables like `{error_id}` and `{timestamp}`
- Optional conditional blocks:
  - `{{#if some_variable}} ... {{/if}}`

Rendering rules implemented by the pipe:
- A line is dropped when a placeholder it uses was supplied but came out empty; a name the message never supplies is left in the text verbatim, so wrap such a line in `{{#if name}}` and it is left out instead.
- A `{name}` token belongs to the template, never to a value: a substituted value is shown verbatim and is never re-read as a placeholder.
- Provider-controlled values arrive already shaped for the card: inside a backtick span, on a heading or on a bare card line they are collapsed to one line and their backticks removed; inside a fence they arrive fenced, so a template must not fence such a value again.
- A boolean placeholder renders `True`/`False`, and its line is dropped when the value is false; the `{{#if}}` form of the same value is equivalent.
- A value made only of backticks renders as `_` rather than deleting its own line. Backticks are stripped after whitespace collapses, so a value of ` ``` ` would otherwise become empty and the drop rule above would remove the whole bullet — on the shipped card, one such word from a provider erased the entire `### Error:` section. A value that is legitimately empty (`""`, `"   "`, a blank line) still reads as empty, so its `{{#if}}` gate and the drop rule keep working.
- Conditional blocks render only when the referenced variable is “present”.

Minimal example:

```markdown
### Request failed
**Error ID:** `{error_id}`
{{#if support_email}}**Support:** {support_email}{{/if}}
```

---

## Common template variables

### Variables available to most templates
The pipe always provides these for `_emit_templated_error` templates:
- `error_id`, `timestamp`, `session_id`, `user_id`, `support_email`, `support_url`

It then merges in per-error variables (for example `status_code`, `reason`, `timeout_seconds`, `error_type`).

### Variables for OpenRouter “request rejected” templates
The OpenRouter error formatter supports a larger set of optional values, including:
- `heading`, `detail`, `sanitized_detail`
- `openrouter_code`, `openrouter_message`
- `upstream_type`, `upstream_message`
- `provider`, `requested_model`, `api_model_id`, `normalized_model_id`
- `retry_after_seconds`, `rate_limit_type`
- `include_model_limits`, `context_limit_tokens`, `max_output_tokens`
- `metadata_json`, `provider_raw_json`, `diagnostics`
- `error_chunk_id`, `error_chunk_created`, `is_streaming_error`, `streaming_provider`, `streaming_model`, `native_finish_reason`, `request_id_reference`

Because OpenRouter/provider responses vary, treat these fields as optional and wrap them in `{{#if ...}}` blocks.

`max_output_tokens` on an error card is the provider's **advertised** ceiling, read straight from the catalog entry (`core/errors.py`). It is deliberately not the value the pipe sends when `USE_MODEL_MAX_OUTPUT_TOKENS` is on: that is the smaller of the advertised ceiling and half the model's context window. A diagnostic about a failure should report the provider's own limit, so the two numbers are meant to differ.

The pipe does the span and fence work on these values itself: a value placed inside a backtick span, on a `### ` heading, or on a bare `**…**` / `- ` line arrives as one logical line with its backticks removed, and a value placed in a fenced block arrives inside a fence long enough to contain it, so a custom template does not have to. The pipe's own numbers and labels (`status_code`, `retry_after_seconds`, `context_limit_tokens`, `max_output_tokens`, `diagnostics`) are already single-line, and a boolean placeholder renders `True`/`False`.

### When the model-limits block renders

`include_model_limits` guards the section that prints the model's context window and output cap. It is set when the rejection looks like a context overflow **and** the catalog knows at least one of those two numbers for the model.

A rejection counts as a context overflow when either holds:
- OpenRouter tags the error with the typed code `context_length_exceeded`. This is read from `error.metadata.error_type` on Chat Completions and from the top-level `error_type` on Responses, the two places OpenRouter documents it. On Responses the native error code is lossy — `context_length_exceeded` collapses into `invalid_prompt` — so the typed field is the only reliable signal there.
- The message text names the remedy, matched against both the current wording (“context compression”) and the wording OpenRouter used before the feature was renamed (`or use the "middle-out"`).

The provider's own error type (surfaced as `upstream_type`) is deliberately **not** consulted: it carries the upstream vendor's category, such as `invalid_request_error`, which covers far more than context overflows.

---

## Provider-specific recovery (reasoning/thinking mismatches)

Some providers reject “reasoning” requests when their own “thinking” mode is not enabled for that model/provider combination (for example error messages referencing `thinking_config.include_thoughts`).

When the pipe detects this condition, it can retry once with reasoning disabled by:
- clearing `reasoning`, or setting it to `{"effort": "none"}`
- removing the legacy `include_reasoning` flag when any id in `models` fails to list it — and, where the dropped value was `false` and the primary lists `reasoning`, carrying thinking off as `reasoning: {"effort": "none"}` instead; on a request with no `model_fallback` the flag is simply turned off

This retry is only available while the attempt has published nothing the reader keeps: once a thinking box or a tool card is on screen the attempt owns the turn, and the error is reported below what the reader already has instead.

This behavior is intended to convert certain provider-side “configuration mismatch” failures into a successful answer without requiring the user to change settings mid-conversation.

A row the catalogue marks `reasoning.mandatory` is the exception: the pipe does not retry it, because stripping reasoning from a request that must think is the very shape the row refuses. The first response is returned to the user with the provider's own diagnostic rather than a resend that would fail the same way and hide it.

---

## Valve configuration (where to customize)

### Support contact valves
- `SUPPORT_EMAIL`
- `SUPPORT_URL`

### Template override valves
- `OPENROUTER_ERROR_TEMPLATE`
- `AUTHENTICATION_ERROR_TEMPLATE`
- `INSUFFICIENT_CREDITS_TEMPLATE`
- `RATE_LIMIT_TEMPLATE`
- `SERVER_TIMEOUT_TEMPLATE`
- `PAYLOAD_TOO_LARGE_TEMPLATE`
- `NETWORK_TIMEOUT_TEMPLATE`
- `CONNECTION_ERROR_TEMPLATE`
- `SERVICE_ERROR_TEMPLATE`
- `INTERNAL_ERROR_TEMPLATE`
- `ENDPOINT_OVERRIDE_CONFLICT_TEMPLATE`
- `DIRECT_UPLOAD_FAILURE_TEMPLATE`
- `MODEL_RESTRICTED_TEMPLATE`
- `STREAM_INTERRUPTED_TEMPLATE`

Every valve above restores itself: clear its box and save, and the built-in text is written back for that valve alone. A box holding only spaces or newlines counts as cleared.

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for defaults and descriptions.

---

## Customization workflow (recommended)

1. Start from the default templates (edit minimally).
2. Wrap optional fields in `{{#if ...}}` blocks so missing variables do not render blank or confusing lines.
3. Keep user-facing messages actionable (what happened, what to do next).
4. Validate changes by triggering known failure modes in a controlled environment (see “Testing”).
5. To start over on any one of these valves, clear its box and save. The built-in text for that valve is written back and appears in the box on the next load, ready to edit again; nothing else is changed.

---

## Customization examples

### Example: minimal corporate support footer

```markdown
### ⚠️ Request failed
**Error ID:** `{error_id}`
{{#if timestamp}}**Time:** {timestamp}{{/if}}

If this persists, contact support and include the Error ID.
{{#if support_url}}**Support:** {support_url}{{/if}}
```

### Example: operator-forward template (adds diagnostic JSON)

A value that arrives already fenced must not be fenced again by the template: `metadata_json` carries its
own fence, so a template that adds a ```json … ``` pair around it closes that fence early and the JSON
escapes the block.

You do not have to edit a stored template that already fences one of these values. The renderer tracks
the template's own fences, so a row written as ` ```json ` / `{metadata_json}` / ` ``` ` — the shape
shipped before 2.7.4, and the shape the pipe's own example used to show — renders as exactly one
code block holding the payload, with the label above it and the card text after it. A template that
does not fence the value is unaffected. The same holds for `{raw_body}`, `{flagged_excerpt}` and
`{provider_raw_json}`; a value is only unwrapped when the template supplies the fence.

````markdown
### 🧾 Provider error
**Error ID:** `{error_id}`
{{#if provider}}**Provider:** {provider}{{/if}}
{{#if requested_model}}**Model:** {requested_model}{{/if}}

{{#if metadata_json}}
**Metadata:**
{metadata_json}
{{/if}}
````

---

## Troubleshooting (template system)

- Template changes do not apply:
  - Ensure you edited the correct function’s valves in Open WebUI (and saved).
  - Ensure you edited the correct template valve for the error type you’re testing.
- Conditionals do not render:
  - The variable may be empty for that error path; wrap the whole section in `{{#if var}}`.
- Templates render blank:
  - A placeholder on a line can cause the entire line to be dropped if the variable is empty; prefer multi-line blocks with conditionals for optional sections.
  - A line whose **own** placeholder resolves to a missing or empty value is omitted; a value that itself contains `{name}` is shown verbatim, never re-read as a placeholder.

---

## Operator runbook (correlating user reports with logs)

1. Ask the user for the `error_id` displayed in the UI.
2. Search backend logs for `[{error_id}]`.
3. Use `session_id` and `user_id` (when available) to correlate with other telemetry (Redis cost snapshots, session log archives, etc.).

See also: [Session Log Storage](session_log_storage.md) and [Request Identifiers & Abuse Attribution](request_identifiers_and_abuse_attribution.md).

---

## Testing

This repository includes tests for template behavior and error rendering. Prefer running the specific test modules first, then the full suite:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_error_handling.py tests/test_template_valve_restore.py -q
PYTHONPATH=. .venv/bin/pytest tests -q
```
