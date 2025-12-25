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
| `429` | `RATE_LIMIT_TEMPLATE` |
| other / default | `OPENROUTER_ERROR_TEMPLATE` |

These templates are used for the `OpenRouterAPIError` path (and for certain HTTP status errors that are converted into an OpenRouter error object by reading the response body best-effort).

### B) Generic templated errors (network/5xx/internal)

The pipe also renders Markdown templates for other exception categories:

| Condition | Template valve |
| --- | --- |
| `httpx.TimeoutException` | `NETWORK_TIMEOUT_TEMPLATE` |
| `httpx.ConnectError` | `CONNECTION_ERROR_TEMPLATE` |
| `httpx.HTTPStatusError` where `status_code >= 500` | `SERVICE_ERROR_TEMPLATE` |
| any other exception | `INTERNAL_ERROR_TEMPLATE` |

**Note:** The timeout template path may display a fallback `timeout_seconds` value when `HTTP_TOTAL_TIMEOUT_SECONDS` is unset (`null`). Use the timeout valves in [Valves & Configuration Atlas](valves_and_configuration_atlas.md) as the source of truth for runtime behavior.

---

## Template rendering rules

Templates are Markdown strings with:
- Placeholder variables like `{error_id}` and `{timestamp}`
- Optional conditional blocks:
  - `{{#if some_variable}} ... {{/if}}`

Rendering rules implemented by the pipe:
- If a placeholder value is empty/missing, the entire line containing it is dropped.
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

It then merges in per-error variables (for example `status_code`, `reason`, `endpoint`, `timeout_seconds`).

### Variables for OpenRouter “request rejected” templates
The OpenRouter error formatter supports a larger set of optional values, including:
- `heading`, `detail`, `sanitized_detail`
- `openrouter_code`, `openrouter_message`
- `upstream_type`, `upstream_message`
- `provider`, `requested_model`, `api_model_id`, `normalized_model_id`
- `retry_after_seconds`, `rate_limit_type`
- `metadata_json`, `provider_raw_json`, `diagnostics`

Because OpenRouter/provider responses vary, treat these fields as optional and wrap them in `{{#if ...}}` blocks.

---

## Provider-specific recovery (reasoning/thinking mismatches)

Some providers reject “reasoning” requests when their own “thinking” mode is not enabled for that model/provider combination (for example error messages referencing `thinking_config.include_thoughts`).

When the pipe detects this condition, it can retry once with reasoning disabled by:
- clearing `reasoning`
- disabling legacy `include_reasoning`
- clearing `thinking_config`

This behavior is intended to convert certain provider-side “configuration mismatch” failures into a successful answer without requiring the user to change settings mid-conversation.

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
- `NETWORK_TIMEOUT_TEMPLATE`
- `CONNECTION_ERROR_TEMPLATE`
- `SERVICE_ERROR_TEMPLATE`
- `INTERNAL_ERROR_TEMPLATE`

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for defaults and descriptions.

---

## Customization workflow (recommended)

1. Start from the default templates (edit minimally).
2. Wrap optional fields in `{{#if ...}}` blocks so missing variables do not render blank or confusing lines.
3. Keep user-facing messages actionable (what happened, what to do next).
4. Validate changes by triggering known failure modes in a controlled environment (see “Testing”).

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

````markdown
### 🧾 Provider error
**Error ID:** `{error_id}`
{{#if provider}}**Provider:** {provider}{{/if}}
{{#if requested_model}}**Model:** {requested_model}{{/if}}

{{#if metadata_json}}
**Metadata:**
```json
{metadata_json}
```
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

---

## Operator runbook (correlating user reports with logs)

1. Ask the user for the `error_id` displayed in the UI.
2. Search backend logs for `[{error_id}]`.
3. Use `session_id` and `user_id` (when available) to correlate with other telemetry (Redis cost snapshots, session log archives, etc.).

See also: [Session Log Storage](session_log_storage.md) and [Request Identifiers & Abuse Attribution](request_identifiers_and_abuse_attribution.md).

---

## Testing

This repository includes tests for template behavior and error rendering. Prefer running the specific test module first, then the full suite:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_error_templates.py -q
PYTHONPATH=. .venv/bin/pytest tests -q
```
