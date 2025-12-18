# error handling & user experience

**file:** `docs/error_handling_and_user_experience.md`
**related source:** `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py` (lines 177-265, 3179-3235, 6249-6312, 8236-8310)

This document covers the comprehensive error template system that catches all exception types and renders user-friendly Markdown error cards instead of letting Open WebUI display raw error boxes.

> **Quick Navigation**: [📑 Index](documentation_index.md) | [🏗️ Architecture](developer_guide_and_architecture.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🔒 Security](security_and_encryption.md)

---

## overview

The error template system prevents ugly Open WebUI error boxes by intercepting all exceptions in `_handle_pipe_call()` and emitting clean, professional Markdown cards with:

- **User-friendly messages** – Actionable guidance without technical jargon
- **Unique error IDs** – 16-character hex identifiers (via `secrets.token_hex(8)`) for support correlation
- **Timestamps** – ISO 8601 UTC timestamps for incident investigation
- **Session context** – Session ID and User ID for multi-tenant debugging
- **Support contacts** – Optional `SUPPORT_EMAIL` and `SUPPORT_URL` from valves

Operators see technical details in logs prefixed with error IDs (e.g., `[a3f8b2c1] Network timeout: ...`), while users see friendly messages.

---

## exception hierarchy & templates

The pipe catches exceptions in this order:

1. **OpenRouterAPIError** (400 errors from OpenRouter) → `OPENROUTER_ERROR_TEMPLATE`
2. **httpx.TimeoutException** (network timeouts) → `NETWORK_TIMEOUT_TEMPLATE`
3. **httpx.ConnectError** (connection failures) → `CONNECTION_ERROR_TEMPLATE`
4. **httpx.HTTPStatusError** (5xx service errors) → `SERVICE_ERROR_TEMPLATE`
5. **Exception** (catch-all for unexpected errors) → `INTERNAL_ERROR_TEMPLATE`

Each template supports Handlebars-style `{{#if variable}}...{{/if}}` conditionals for optional sections.

---

## 1. network timeout errors

**Exception:** `httpx.TimeoutException`

**When it happens:** Request to OpenRouter takes longer than `HTTP_TOTAL_TIMEOUT_SECONDS` (default 120s)

**Default template** (from `open_webui_openrouter_pipe.py:177-198`):

```markdown
### ⏱️ Request Timeout

The request to OpenRouter took too long to complete.

**Error ID:** `{error_id}`
{{#if timeout_seconds}}
**Timeout:** {timeout_seconds}s
{{/if}}
{{#if timestamp}}
**Time:** {timestamp}
{{/if}}

**Possible causes:**
- OpenRouter's servers are slow or overloaded
- Network congestion
- Large request taking longer than expected

**What to do:**
- Wait a few moments and try again
- Try a smaller request if possible
- Check [OpenRouter Status](https://status.openrouter.ai/)
{{#if support_email}}
- Contact support: {support_email}
{{/if}}

---

## provider-specific recovery (Gemini thinking errors)

Certain providers reject `include_reasoning` when their own "thinking" feature is disabled. Google's Gemini stack, for example, returns:

```
Unable to submit request because Thinking_config.include_thoughts is only enabled when thinking is enabled.
```

Without special handling this manifested as a 400 error card. Starting in `open_webui_openrouter_pipe.py:6956-7002` the pipe now:

1. Detects that specific error text (`Thinking_config.include_thoughts…`) via `_should_retry_without_reasoning`.
2. Automatically removes the `reasoning` payload (and the legacy `include_reasoning` flag when present) from the outgoing request.
3. Retries the request exactly once, keeping the original stream/non-stream mode.
4. Falls back to the normal error template only if the retry also fails.

This makes Gemini usable even when `ENABLE_REASONING` is set globally. Operators still see the warning in DEBUG logs (`Retrying without reasoning…`) so they can adjust valves or per-model settings later, but end users get the final answer instead of a provider error.

> **See also**: For comprehensive coverage of OpenRouter provider-specific behaviors, including catalog routing, automatic plugin wiring, and multimodal guardrails, see [OpenRouter Integrations and Telemetry](openrouter_integrations_and_telemetry.md).

```

**Example rendered output:**

---

### ⏱️ Request Timeout

The request to OpenRouter took too long to complete.

**Error ID:** `a3f8b2c1d4e5f6a7`
**Timeout:** 120s
**Time:** 2025-12-06T15:42:33Z

**Possible causes:**
- OpenRouter's servers are slow or overloaded
- Network congestion
- Large request taking longer than expected

**What to do:**
- Wait a few moments and try again
- Try a smaller request if possible
- Check [OpenRouter Status](https://status.openrouter.ai/)
- Contact support: support@example.com

---

**Logged as:**
```
ERROR [a3f8b2c1d4e5f6a7] Network timeout: httpx.TimeoutException (session=sess_abc123, user=user_789)
```

---

## 2. connection failure errors

**Exception:** `httpx.ConnectError`

**When it happens:** Cannot reach OpenRouter's servers (DNS failure, firewall block, network down)

**Default template** (from `open_webui_openrouter_pipe.py:200-223`):

```markdown
### 🔌 Connection Failed

Unable to reach OpenRouter's servers.

**Error ID:** `{error_id}`
{{#if error_type}}
**Error type:** `{error_type}`
{{/if}}
{{#if timestamp}}
**Time:** {timestamp}
{{/if}}

**Possible causes:**
- Network connectivity issues
- Firewall blocking HTTPS traffic
- DNS resolution failure
- OpenRouter service outage

**What to do:**
1. Check your internet connection
2. Verify firewall allows HTTPS (port 443)
3. Check [OpenRouter Status](https://status.openrouter.ai/)
4. Contact your network administrator if the issue persists
{{#if support_email}}

**Support:** {support_email}
{{/if}}
```

**Example rendered output:**

---

### 🔌 Connection Failed

Unable to reach OpenRouter's servers.

**Error ID:** `b7c8d9e0f1a2b3c4`
**Error type:** `ConnectError`
**Time:** 2025-12-06T15:45:12Z

**Possible causes:**
- Network connectivity issues
- Firewall blocking HTTPS traffic
- DNS resolution failure
- OpenRouter service outage

**What to do:**
1. Check your internet connection
2. Verify firewall allows HTTPS (port 443)
3. Check [OpenRouter Status](https://status.openrouter.ai/)
4. Contact your network administrator if the issue persists

**Support:** support@example.com

---

**Logged as:**
```
ERROR [b7c8d9e0f1a2b3c4] Connection failed: httpx.ConnectError('connection refused') (session=sess_abc123, user=user_789)
```

---

## 3. service error (5xx responses)

**Exception:** `httpx.HTTPStatusError` with status code >= 500

**When it happens:** OpenRouter's servers return 500, 502, 503, 504, etc.

**Default template** (from `open_webui_openrouter_pipe.py:225-243`):

```markdown
### 🔴 OpenRouter Service Error

OpenRouter's servers are experiencing issues.

**Error ID:** `{error_id}`
{{#if status_code}}
**Status:** {status_code} {reason}
{{/if}}
{{#if timestamp}}
**Time:** {timestamp}
{{/if}}

This is **not** a problem with your request. The issue is on OpenRouter's side.

**What to do:**
- Wait a few minutes and try again
- Check [OpenRouter Status](https://status.openrouter.ai/) for updates
- If the problem persists for more than 15 minutes, contact OpenRouter support
{{#if support_email}}

**Support:** {support_email}
{{/if}}
```

**Example rendered output:**

---

### 🔴 OpenRouter Service Error

OpenRouter's servers are experiencing issues.

**Error ID:** `c5d6e7f8a9b0c1d2`
**Status:** 502 Bad Gateway
**Time:** 2025-12-06T15:48:07Z

This is **not** a problem with your request. The issue is on OpenRouter's side.

**What to do:**
- Wait a few minutes and try again
- Check [OpenRouter Status](https://status.openrouter.ai/) for updates
- If the problem persists for more than 15 minutes, contact OpenRouter support

**Support:** support@example.com

---

**Logged as:**
```
ERROR [c5d6e7f8a9b0c1d2] OpenRouter service error: 502 Bad Gateway (session=sess_abc123, user=user_789)
```

---

## 4. internal / unexpected errors

**Exception:** Any unhandled exception (ValueError, KeyError, etc.)

**When it happens:** Unexpected bugs, coding errors, or edge cases

**Default template** (from `open_webui_openrouter_pipe.py:245-265`):

```markdown
### ⚠️ Unexpected Error

Something unexpected went wrong while processing your request.

**Error ID:** `{error_id}` — Share this with support
{{#if error_type}}
**Error type:** `{error_type}`
{{/if}}
{{#if timestamp}}
**Time:** {timestamp}
{{/if}}

The error has been logged and will be investigated.

**What to do:**
- Try your request again
- If the problem persists, contact support with the Error ID above
{{#if support_email}}
- Email: {support_email}
{{/if}}
{{#if support_url}}
- Support: {support_url}
{{/if}}
```

**Example rendered output:**

---

### ⚠️ Unexpected Error

Something unexpected went wrong while processing your request.

**Error ID:** `d3e4f5a6b7c8d9e0` — Share this with support
**Error type:** `ValueError`
**Time:** 2025-12-06T15:51:22Z

The error has been logged and will be investigated.

**What to do:**
- Try your request again
- If the problem persists, contact support with the Error ID above
- Email: support@example.com
- Support: https://support.example.com/tickets

---

**Logged as:**
```
ERROR [d3e4f5a6b7c8d9e0] Unexpected error: ValueError('invalid model configuration') (session=sess_abc123, user=user_789)
```

---

## 5. openrouter 400 errors (existing system)

**Exception:** `OpenRouterAPIError` (prompt too long, moderation block, invalid request)

**When it happens:** OpenRouter rejects the request before processing

**Default template** (from `open_webui_openrouter_pipe.py:141-175`):

```markdown
### 🚫 {heading} could not process your request.

### Error: `{sanitized_detail}`

- **Model**: `{model_identifier}`
- **Provider**: `{provider}`
- **Requested model**: `{requested_model}`
- **API model id**: `{api_model_id}`
- **Normalized model id**: `{normalized_model_id}`
- **OpenRouter code**: `{openrouter_code}`
- **Provider error**: `{upstream_type}`
- **Reason**: `{reason}`
- **Request ID**: `{request_id}`
{{#if include_model_limits}}

**Model limits:**
Context window: {context_limit_tokens} tokens
Max output tokens: {max_output_tokens} tokens
Adjust your prompt or requested output to stay within these limits.
{{/if}}
{{#if moderation_reasons}}

**Moderation reasons:**
{moderation_reasons}
Please review the flagged content or contact your administrator if you believe this is a mistake.
{{/if}}
{{#if flagged_excerpt}}

**Flagged text excerpt:**
```
{flagged_excerpt}
```
Provide this excerpt when following up with your administrator.
{{/if}}
{{#if raw_body}}

**Raw provider response:**
```
{raw_body}
```
{{/if}}

Please adjust the request and try again, or ask your admin to enable the middle-out option.
{request_id_reference}
```

**Note:** This template has many more placeholders than the other error types because OpenRouter 400 responses include rich provider-specific metadata.

### Template variables

You can mix and match the following placeholders with `{{#if ...}}` blocks:

| Placeholder | Description |
| --- | --- |
| `{heading}` | Friendly label such as `Anthropic: claude-3`. |
| `{error_id}` / `{timestamp}` | Unique identifier + ISO timestamp injected by `_emit_templated_error()` / `_report_openrouter_error()`. |
| `{session_id}` / `{user_id}` | Open WebUI session + user identifiers when available. |
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
| `{metadata_json}` / `{provider_raw_json}` | Pretty-printed JSON blocks for OpenRouter metadata and the provider's raw error payload. |
| `{context_limit_tokens}` / `{max_output_tokens}` | Formatted token limits (available only when OpenRouter suggests "middle-out"). |
| `{include_model_limits}` | Boolean helper used with `{{#if include_model_limits}} ... {{/if}}` to gate the entire limit block. |
| `{openrouter_message}` / `{upstream_message}` | Original strings from OpenRouter/provider before formatting. |
| `{native_finish_reason}` | Streaming-only finish reason emitted with `native_finish_reason`. |
| `{error_chunk_id}` / `{error_chunk_created}` | Identifiers and timestamps for the SSE chunk that contained the error. |
| `{streaming_provider}` / `{streaming_model}` | Provider/model reported by the streaming event (may differ from the initial request). |
| `{retry_after_seconds}` / `{rate_limit_type}` | When OpenRouter includes rate-limit hints, these expose the wait window and which limiter triggered. |
| `{required_cost}` / `{account_balance}` | Present on HTTP 402 responses when OpenRouter tells you how many credits were needed vs. remaining balance. |

### Customization workflow

When customizing the OpenRouter 400 error template:

1. Copy the default template from `Pipe.Valves.OPENROUTER_ERROR_TEMPLATE`.
2. Add or remove Markdown sections as needed for your organization (e.g., internal ticket instructions, support URLs).
3. Keep `{request_id_reference}` somewhere in the template so operators can correlate user reports with OpenRouter telemetry.
4. Token limit callouts use `{context_limit_tokens}` / `{max_output_tokens}` and are only populated when the upstream error text contains the OpenRouter hint `or use the "middle-out"` **and** the registry exposed limits for that model.

**Example rendered output for prompt-too-long:**

---

### 🚫 Anthropic: claude-sonnet-4 could not process this request

### Error: `Prompt exceeds model context window`

- **Requested model:** `anthropic/claude-sonnet-4`
- **Provider:** `Anthropic`
- **OpenRouter code:** `400`
- **Request ID:** `gen_abc123xyz789`

**Model limits**
- Context window: 200,000 tokens
- Max output tokens: 8,192 tokens

Please trim your prompt or enable the middle-out transform.

Need help? Share `OR-abc123` with the on-call engineer.

---

**Logged as:**
```
ERROR [e1f2a3b4c5d6e7f8] OpenRouter 400: Prompt exceeds context window (session=sess_abc123, user=user_789)
```

---

### Mid-stream errors (streaming mode)

When OpenRouter sends an error *after* streaming has started, the pipe now promotes the SSE payload (`response.failed`, `response.error`, or `error`) into an `OpenRouterAPIError`. That means:

1. The error is routed through `_report_openrouter_error()` so the same templating system is used.
2. Streaming-specific variables (`{native_finish_reason}`, `{error_chunk_id}`, `{streaming_provider}`, `{streaming_model}`) are populated from the SSE event.
3. The entire SSE payload is preserved via `{provider_raw_json}` and `{metadata_json}` so administrators can inspect the provider’s raw response.

These additions make it clear when a failure happened mid-stream and capture the exact chunk metadata for debugging.

### HTTP-status templates

OpenRouter frequently signals different remediation steps via HTTP status codes, so dedicated templates now exist for each major category:

| Status | Valve | Default guidance |
| --- | --- | --- |
| `401 Unauthorized` | `AUTHENTICATION_ERROR_TEMPLATE` | Reminds admins to check API keys / OAuth sessions. |
| `402 Payment Required` | `INSUFFICIENT_CREDITS_TEMPLATE` | Surfaces `{required_cost}` / `{account_balance}` when OpenRouter includes billing hints. |
| `408 Request Timeout` | `SERVER_TIMEOUT_TEMPLATE` | Indicates OpenRouter timed out the request (distinct from client-side network timeouts). |
| `429 Too Many Requests` | `RATE_LIMIT_TEMPLATE` | Surfaces `{retry_after_seconds}` and `{rate_limit_type}` from headers/metadata so users know when to retry. |

Each template accepts the common context placeholders (`{error_id}`, `{timestamp}`, `{session_id}`, `{user_id}`, `{support_email}`, `{support_url}`) plus any fields emitted in the response metadata. You can override these valves the same way as the other templates if your organization has custom guidance for auth, billing, or rate-limit incidents.

**Example override (rate limits):**

```markdown
### ⏸️ Slow down

Your organization hit the `{rate_limit_type}` limiter on OpenRouter.

**Error ID:** `{error_id}`
{{#if retry_after_seconds}}
**Retry after:** {retry_after_seconds}s
{{/if}}
{{#if openrouter_message}}
_Provider hint_: `{openrouter_message}`
{{/if}}

Next steps:
1. Wait for the retry window to expire.
2. Reduce concurrency or token usage.
3. Contact support ({support_email}) if you require a higher limit.
```

## valve configuration

All templates can be customized via valves:

### support contact valves

```python
SUPPORT_EMAIL: str = ""  # Optional: support@example.com
SUPPORT_URL: str = ""    # Optional: https://support.example.com/tickets
```

### template override valves

```python
NETWORK_TIMEOUT_TEMPLATE: str = DEFAULT_NETWORK_TIMEOUT_TEMPLATE
CONNECTION_ERROR_TEMPLATE: str = DEFAULT_CONNECTION_ERROR_TEMPLATE
SERVICE_ERROR_TEMPLATE: str = DEFAULT_SERVICE_ERROR_TEMPLATE
INTERNAL_ERROR_TEMPLATE: str = DEFAULT_INTERNAL_ERROR_TEMPLATE
OPENROUTER_ERROR_TEMPLATE: str = DEFAULT_OPENROUTER_ERROR_TEMPLATE
```

---

## template customization examples

### example 1: corporate support workflow

```markdown
### ⚠️ System Error

An unexpected error occurred. Our team has been notified.

**Incident ID:** `{error_id}`
**Time:** {timestamp}

**Next steps:**
1. Try your request again in a few minutes
2. If the issue persists, create a ticket at https://tickets.corp.example.com
3. Include this Incident ID: `{error_id}`

**Internal employees:** Check #eng-support in Slack for real-time updates.
```

### example 2: minimal template (no support info)

```markdown
### Error

Request failed. Please try again.

**Error ID:** `{error_id}`
```

### example 3: detailed debugging template

```markdown
### 🐛 Error Details

**Error ID:** `{error_id}`
**Type:** `{error_type}`
**Timestamp:** {timestamp}
**Session:** {session_id}
**User:** {user_id}

{{#if support_url}}
[Report this error]({support_url}?error_id={error_id})
{{/if}}
```

---

## admin guide: customizing error templates

This section explains how to customize error messages for your organization **without needing to be a developer**. All customization happens through the Open WebUI admin interface.

### step 1: accessing template settings

1. Open the Open WebUI admin panel
2. Navigate to **Admin → Functions → Valves**
3. Find the OpenRouter Responses Pipe
4. Scroll to the error template valves:
   - `NETWORK_TIMEOUT_TEMPLATE`
   - `CONNECTION_ERROR_TEMPLATE`
   - `SERVICE_ERROR_TEMPLATE`
   - `INTERNAL_ERROR_TEMPLATE`
   - `OPENROUTER_ERROR_TEMPLATE`
   - `AUTHENTICATION_ERROR_TEMPLATE`
   - `INSUFFICIENT_CREDITS_TEMPLATE`
   - `RATE_LIMIT_TEMPLATE`
   - `SERVER_TIMEOUT_TEMPLATE`

### step 2: understanding placeholders

Templates use **placeholders** that get replaced with real information. Placeholders are words inside curly braces like `{error_id}`.

**Always available placeholders:**

| Placeholder | What it shows | Example |
|-------------|---------------|---------|
| `{error_id}` | Unique error code for support | `a3f8b2c1d4e5f6a7` |
| `{timestamp}` | When error occurred | `2025-12-06T15:42:33Z` |
| `{session_id}` | Session identifier (if logged in) | `sess_abc123` |
| `{user_id}` | User identifier (if logged in) | `user_789` |
| `{support_email}` | From `SUPPORT_EMAIL` valve | `support@example.com` |
| `{support_url}` | From `SUPPORT_URL` valve | `https://tickets.example.com` |

**Error-specific placeholders:**

For **timeout errors**:
- `{timeout_seconds}` – How long we waited (e.g., `120`)

For **connection errors**:
- `{error_type}` – Technical error name (e.g., `ConnectError`)

For **authentication errors (401)**:
- `{openrouter_message}` – Reason returned by OpenRouter (e.g., “invalid API key”).

For **insufficient credits (402)**:
- `{required_cost}` – Estimated cost of the attempted request.
- `{account_balance}` – Credits remaining on the OpenRouter account.

For **rate limits (429)**:
- `{retry_after_seconds}` – Wait time returned by OpenRouter headers.
- `{rate_limit_type}` – Scope of the limiter (key/user/account) when provided.

For **OpenRouter timeouts (408)**:
- `{openrouter_code}` / `{openrouter_message}` – Server-side timeout context.

For **service errors** (5xx):
- `{status_code}` – HTTP status code (e.g., `502`)
- `{reason}` – Status description (e.g., `Bad Gateway`)

For **internal errors**:
- `{error_type}` – Technical error name (e.g., `ValueError`)

### step 3: using conditional sections

Sometimes you want text to appear only when certain information is available. Use **conditional blocks** for this.

**Important:** All placeholders `{variable}` must be wrapped in conditional blocks. If a variable might be empty, wrap it with `{{#if}}...{{/if}}` to prevent the line from appearing when the variable is missing.

**Basic pattern:**
```markdown
{{#if variable_name}}
This text only shows if variable_name has a value
{{/if}}
```

**Real-world example:**

```markdown
Contact support for help.
{{#if support_email}}
Email: {support_email}
{{/if}}
{{#if support_url}}
Ticket system: {support_url}
{{/if}}
```

**What happens:**
- If you set `SUPPORT_EMAIL = "help@example.com"`, users see: "Email: help@example.com"
- If you set `SUPPORT_URL = "https://tickets.example.com"`, users see: "Ticket system: https://tickets.example.com"
- If both valves are empty, users only see: "Contact support for help."

**Important rules:**
1. Opening tag: `{{#if variable_name}}`
2. Closing tag: `{{/if}}`
3. Must match exactly – even spacing matters
4. **All placeholders must be inside `{{#if}}` blocks** – This ensures clean output when variables are empty
5. The section between tags only appears if the variable has a value

### step 4: practical customization examples

#### example: add company branding

**Change:**
```markdown
### ⏱️ Request Timeout
```

**To:**
```markdown
### ⏱️ Acme Corp AI Assistant - Request Timeout
```

#### example: add internal ticket system link

**Change:**
```markdown
**What to do:**
- Wait a few moments and try again
- Try a smaller request if possible
```

**To:**
```markdown
**What to do:**
- Wait a few moments and try again
- Try a smaller request if possible
{{#if support_url}}
- [Create support ticket]({support_url}?error_id={error_id})
{{/if}}
```

**Note:** The URL automatically includes the error ID so your support team can look it up.

#### example: simplify for non-technical users

**Change:**
```markdown
**Error ID:** `{error_id}` — Share this with support
{{#if error_type}}
**Error type:** `{error_type}`
{{/if}}
{{#if timestamp}}
**Time:** {timestamp}
{{/if}}
```

**To:**
```markdown
**Reference number:** `{error_id}`

Share this number with our support team if you need help.
```

**Result:** Users only see the error ID, no technical details like error type or timestamp.

#### example: add specific escalation paths

**For internal errors template:**
```markdown
### ⚠️ System Error

Something unexpected happened. Don't worry – this has been logged automatically.

**Reference number:** `{error_id}`

**What to do next:**

For **internal users:**
1. Check #eng-alerts in Slack for known issues
2. If no alert exists, create a P2 ticket at https://tickets.corp.example.com
3. Include reference number: `{error_id}`

For **external users:**
{{#if support_email}}
Email our support team at {support_email} with reference number `{error_id}`
{{/if}}
```

### step 5: testing your templates

After editing a template:

1. **Save** the valve changes in Open WebUI
2. **Restart** Open WebUI workers (templates are loaded at startup)
3. **Trigger a test error** by setting an invalid configuration temporarily:
   - For timeout: Set `HTTP_TOTAL_TIMEOUT_SECONDS = 1`
   - For connection: Set `BASE_URL = https://invalid.example.com`
4. **Send a test message** and verify the error appears correctly
5. **Restore** the original configuration

### step 6: common mistakes to avoid

**Mistake 1: Forgetting closing tag**
```markdown
{{#if support_email}}
Email: {support_email}
<!-- Missing {{/if}} – this breaks the template! -->
```

**Mistake 2: Unwrapped placeholder**
```markdown
**Error ID:** `{error_id}`  ❌ Wrong! Must wrap in {{#if}}
```

**Correct:**
```markdown
{{#if error_id}}
**Error ID:** `{error_id}`
{{/if}}
```

**Mistake 3: Typo in conditional**
```markdown
{{#if suport_email}}  ❌ Wrong! Should be support_email
Email: {support_email}
{{/if}}
```

**Mistake 4: Forgetting to set support valves**
If you use `{support_email}` in your template but don't set the `SUPPORT_EMAIL` valve, users see nothing where the email should appear. Make sure to configure both:
- `SUPPORT_EMAIL` (e.g., `support@example.com`)
- `SUPPORT_URL` (e.g., `https://tickets.example.com`)

### quick reference: template variables

**Network timeout template** (`NETWORK_TIMEOUT_TEMPLATE`):
- `{error_id}`, `{timestamp}`, `{session_id}`, `{user_id}`, `{support_email}`, `{support_url}`
- `{timeout_seconds}` – How long we waited

**Connection error template** (`CONNECTION_ERROR_TEMPLATE`):
- `{error_id}`, `{timestamp}`, `{session_id}`, `{user_id}`, `{support_email}`, `{support_url}`
- `{error_type}` – Technical name of connection error

**Service error template** (`SERVICE_ERROR_TEMPLATE`):
- `{error_id}`, `{timestamp}`, `{session_id}`, `{user_id}`, `{support_email}`, `{support_url}`
- `{status_code}` – HTTP error code (502, 503, etc.)
- `{reason}` – Error description (Bad Gateway, etc.)

**Internal error template** (`INTERNAL_ERROR_TEMPLATE`):
- `{error_id}`, `{timestamp}`, `{session_id}`, `{user_id}`, `{support_email}`, `{support_url}`
- `{error_type}` – Technical name of error

**OpenRouter 400 error template** (`OPENROUTER_ERROR_TEMPLATE`):
- See Section 9 of `openrouter_integrations_and_telemetry.md` for full variable list

---

## troubleshooting guide

### error IDs not appearing in logs

**Problem:** User reports error ID but logs don't show it

**Solution:** Check log level is INFO or lower; error IDs are logged at ERROR level by default

### template rendering failures

**Problem:** Users see fallback error message instead of template

**Logged as:**
```
ERROR [abc123] Template rendering failed: KeyError('unknown_variable')
```

**Solution:** Check template syntax and ensure all `{placeholders}` match available variables

### conditionals not working

**Problem:** `{{#if}}` blocks always appear or always disappear

**Solution:**
- Verify exact syntax: `{{#if variable}}` not `{{if variable}}`
- Check variable name matches available variables list
- Ensure closing tag: `{{/if}}`

### customization not applied

**Problem:** Changed valve but still seeing default template

**Solution:**
- Restart Open WebUI workers after valve changes
- Verify valve is saved (check Admin → Functions → Valves)
- Check you're editing the correct valve (system vs user valves)

---

## operator runbook

### correlating user reports with logs

1. **User reports error:** "I got Error ID `a3f8b2c1d4e5f6a7`"
2. **Search logs:** `grep "a3f8b2c1d4e5f6a7" /var/log/openwebui/*.log`
3. **Find full context:** Log line includes session ID and user ID
4. **Trace session:** `grep "sess_abc123" /var/log/openwebui/*.log`

### common patterns

**Pattern:** Spike in timeout errors (Error ID prefix `[`)

**Action:**
1. Check OpenRouter status: https://status.openrouter.ai/
2. Review `HTTP_TOTAL_TIMEOUT_SECONDS` valve (may be too low)
3. Check network latency to OpenRouter

**Pattern:** Connection errors during specific hours

**Action:**
1. Check firewall logs for blocks during those hours
2. Verify DNS resolution is stable
3. Check for network maintenance windows

**Pattern:** Internal errors with specific `error_type`

**Action:**
1. Group errors by `error_type` in logs
2. File bug report with stack trace
3. Check if recent code/config changes coincide with errors

---

## testing error templates

Use the comprehensive test suite:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_error_templates.py -v
```

**Test coverage:**
- `TestEmitTemplatedError` – Template rendering, conditionals, variable injection
- `TestNetworkTimeoutError` – Timeout exception handling
- `TestConnectionError` – Connection failure handling
- `TestServiceError` – 5xx service error handling
- `TestInternalError` – Generic exception handling
- `TestTemplateCustomization` – Custom template usage

**Manual testing:**

Trigger specific errors by setting valves:
- Timeout: Set `HTTP_TOTAL_TIMEOUT_SECONDS=1` and make a request
- Connection: Set `BASE_URL=https://invalid.example.com`
- Service: (Requires actual OpenRouter 5xx response)
- Internal: Introduce a code bug in a test environment

---

## best practices

1. **Keep templates concise** – Users want quick guidance, not essays
2. **Provide actionable steps** – "Check X" or "Contact Y" is more helpful than "Something went wrong"
3. **Use conditionals** – Hide empty sections with `{{#if}}...{{/if}}`
4. **Include error IDs** – Always show `{error_id}` so users can report issues
5. **Link to status pages** – External status pages (OpenRouter, company) help users self-diagnose
6. **Test templates** – Render with both populated and empty variables to ensure clean output
7. **Log technical details** – Operators need stack traces; users don't

---

## future enhancements

**Planned:**
- Health endpoint exposing error rate by type
- Dead-letter queue for unrecoverable errors
- Prometheus metrics export (error_count by type/template)
- Error retry strategies (automatic retry for transient errors)

**Under consideration:**
- User-visible error history (last 5 errors in UI)
- Webhook notifications for critical errors
- Template preview in Admin UI


---

## Related Topics

**Error Sources & Context:**
- **Provider Errors**: [OpenRouter Integrations and Telemetry](openrouter_integrations_and_telemetry.md) - Provider-specific behaviors, catalog routing, automatic plugin wiring
- **Connection & Resilience**: [Concurrency Controls and Resilience](concurrency_controls_and_resilience.md) - Timeout configuration, breaker patterns, admission control
- **Security Errors**: [Security and Encryption](security_and_encryption.md) - SSRF blocks, encryption failures, key rotation issues

**Configuration & Customization:**
- **Valve Reference**: [Valves and Configuration Atlas](valves_and_configuration_atlas.md) - Error template configuration, timeout settings, retry limits
- **Testing Errors**: [Testing, Bootstrap, and Operations](testing_bootstrap_and_operations.md) - Incident response, troubleshooting workflows

**Architecture:**
- **System Overview**: [Developer Guide and Architecture](developer_guide_and_architecture.md) - Error propagation through the request lifecycle
- **Streaming Errors**: [Streaming Pipeline and Emitters](streaming_pipeline_and_emitters.md) - How errors are emitted through SSE events

