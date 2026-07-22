# Request identifiers and abuse attribution

**Scope:** How the pipe emits OpenRouter abuse-attribution identifiers (`user`, `metadata`) in multi-user deployments.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🔒 Security](security_and_encryption.md)

---

## Why this matters (multi-user deployments)

If you operate Open WebUI for multiple end-users, sending stable identifiers helps OpenRouter/provider safety systems and your own operators:

* correlate suspicious requests across time,
* narrow abuse investigations to *one* user/session/thread,
* apply targeted mitigations (block/limit the abusive user) rather than broad, account-wide disruption.

This does **not** guarantee that an entire account can never be actioned (serious or repeated abuse can still result in account-level enforcement). It does, however, materially improve attribution and reduces “unknown actor” ambiguity.

If you have a trust & safety, privacy, compliance, or incident response team, consult them before enabling these valves so you align on:

* retention/logging expectations,
* what identifiers are acceptable to share with third parties,
* escalation paths when abuse is reported.

---

## What gets sent (and what does not)

When enabled, the pipe sends **opaque OWUI identifiers** (GUIDs/UUIDs) only:

* **No email**
* **No username**
* **No message content**

These identifiers are only meaningful inside your Open WebUI database and logs.

Note: While these are not “direct identifiers” (like email), they may still be considered *pseudonymous identifiers* in some privacy regimes. Treat them as potentially sensitive operational data.

### OpenRouter request fields (pipe-controlled)

Depending on valves, the pipe can include:

* `user` (top-level): the OWUI user GUID (`__user__["id"]`) — OpenRouter's end-user identifier for abuse detection
* `metadata` (top-level): a `Dict[str, str]` built by the pipe (not OWUI’s full `__metadata__` blob)

`metadata` is only sent when at least one metadata entry is being populated. The pipe also sends a top-level `session_id`, but that is a prompt-cache pin (not an attribution field) — see [Prompt-cache session affinity](openrouter_integrations_and_telemetry.md#216-prompt-cache-session-affinity-session_id).

Important: the pipe **removes** any user-supplied `user`, `session_id`, or `metadata` fields and replaces them with valve-gated values. This prevents clients/users from spoofing attribution identifiers.

### Identifier mapping

Each identifier is gated by a valve. When enabled, the pipe sources IDs from Open WebUI context and maps them into OpenRouter fields as follows:

| Valve | OpenRouter top-level | OpenRouter metadata key | Source in Open WebUI context |
|---|---|---|---|
| `SEND_END_USER_ID` | `user` | `user_id` | `__user__["id"]`, or `__user__["email"]` / `__user__["name"]` per `END_USER_ID_SOURCE` (metadata stays on the id) |
| `SEND_SESSION_ID` | *(none)* | `session_id` | `__metadata__["session_id"]` |
| `SEND_CHAT_ID` | *(none)* | `chat_id` | `__metadata__["chat_id"]` |
| `SEND_MESSAGE_ID` | *(none)* | `message_id` | `__metadata__["message_id"]` |

### Sanitization and constraints

The pipe enforces OpenRouter’s documented `metadata` constraints:

* Maximum **16** key/value pairs.
* Keys must be **≤ 64 chars** and must not contain `[` or `]`.
* Values must be **≤ 512 chars**.

Additionally, the top-level `user` field is capped at **128 characters**. If a source value is missing or invalid, the corresponding field is omitted even when its valve is enabled.

---

## Forwarding Open WebUI user headers to a proxy or gateway

When `BASE_URL` points at a proxy or gateway — for example a **LiteLLM** instance or a PII-masking proxy — instead of OpenRouter directly, the pipe forwards Open WebUI's user-identity **HTTP headers** on every per-user request, matching what a native Open WebUI connection sends. This lets the gateway attribute traffic per user (for example LiteLLM's `user_header_mappings`, which reads request headers, not body fields).

This is a **separate mechanism** from the request-body identifiers above: `SEND_END_USER_ID` writes the OpenRouter body `user` field, while header forwarding stamps HTTP headers a proxy reads. The two are independent and can be combined.

### How to enable

There is **no pipe valve** for this — it is governed entirely by Open WebUI's own setting. Set `ENABLE_FORWARD_USER_INFO_HEADERS=True` in the Open WebUI environment (the same flag that drives header forwarding for native connections). When set, the pipe forwards the headers; when unset (the default), it forwards nothing.

The pipe inherits Open WebUI's header configuration verbatim, including custom header names (such as an `X-Amzn-Bedrock-AgentCore-Runtime-Custom-` prefix) and JWT mode. When `FORWARD_USER_INFO_HEADER_JWT_SECRET` is set, the four `X-OpenWebUI-User-*` headers collapse into one signed `X-OpenWebUI-User-Jwt`.

### Headers sent

| Header (default name) | Value |
|---|---|
| `X-OpenWebUI-User-Name` | display name (URL-encoded) |
| `X-OpenWebUI-User-Id` | user GUID |
| `X-OpenWebUI-User-Email` | email |
| `X-OpenWebUI-User-Role` | role |
| `X-OpenWebUI-Chat-Id` | current chat id |

### Privacy and scope

Unlike the body identifiers above (opaque GUIDs only), these headers can include the user's **name and email**, so forwarding is off by default and happens only when the operator opts in via the Open WebUI setting. When enabled, the headers reach whatever `BASE_URL` points at — including OpenRouter directly if no proxy is in use.

As with any native Open WebUI connection, enabling forwarding means trusting the `BASE_URL` endpoint: like OWUI's own connections, the pipe follows HTTP redirects, so a `BASE_URL` that issues a cross-origin redirect could pass the identity headers onward. Point `BASE_URL` only at an endpoint you trust.

Header forwarding applies to every per-user OpenRouter call: chat, responses, task-model requests (such as title and tag generation), image generation, and video generation (including the video content download from OpenRouter's own endpoint). It is **not** applied to infrastructure calls (model-catalog fetches, warmup pings) or to fetches of third-party media (such as remote image URLs), which must never receive user identity.

---

## Pairing with encrypted session log archives (recommended)

If you’re enabling request identifiers specifically for abuse attribution / incident response, consider also enabling **encrypted session log storage**.

Why:

* OpenRouter/provider support can reference `user` and/or `metadata.*` when reporting abuse or asking you to investigate.
* Encrypted on-disk session log archives give operators a durable record of what happened for a specific request, without needing to run the whole system at `LOG_LEVEL=DEBUG`.
* Archives are stored using the same IDs (`<user_id>/<chat_id>/<message_id>.zip`) so they’re directly searchable from the identifiers you already see in Open WebUI / provider reports.

This is optional: you can keep request identifiers enabled without storing archives, and you can store archives purely as local backups even if you choose not to send identifiers to OpenRouter.

Deep-dive: see [Encrypted session log storage (optional)](session_log_storage.md).

---

## Example payloads

These examples show only the relevant identifier fields; request bodies vary depending on the model and features enabled.

### Minimal (only `user`)

```json
{
  "model": "openrouter/...",
  "input": [...],
  "user": "a3d0d2c1-7f49-4b6b-9a3b-9d3b2a54c2d1",
  "metadata": {
    "user_id": "a3d0d2c1-7f49-4b6b-9a3b-9d3b2a54c2d1"
  }
}
```

### Full attribution

```json
{
  "model": "openrouter/...",
  "input": [...],
  "user": "a3d0d2c1-7f49-4b6b-9a3b-9d3b2a54c2d1",
  "metadata": {
    "user_id": "a3d0d2c1-7f49-4b6b-9a3b-9d3b2a54c2d1",
    "session_id": "0f6b31b0-8c9f-4c3b-a1e7-0d7d2c6b5a33",
    "chat_id": "b52f9c2e-5c01-4c47-8a2e-7b4f8e9a1d00",
    "message_id": "c0d9ad44-0d8b-4e6f-b6f3-8d6a9d1b2c3e"
  }
}
```

---

## Configuration (valves)

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for the canonical list and defaults. The relevant valves are:

* `SEND_END_USER_ID` (default: false) — top-level `user` (abuse attribution)
* `SEND_SESSION_ID` (default: false) — `metadata.session_id` only
* `SEND_CHAT_ID` (default: false) — `metadata.chat_id` only
* `SEND_MESSAGE_ID` (default: false) — `metadata.message_id` only

---

## Operational guidance

* `SEND_END_USER_ID` — sends the `user` field, OpenRouter's end-user abuse identifier (shown as "Client User ID" in their activity view). Enable in multi-user deployments.
* `END_USER_ID_SOURCE` — chooses what that `user` field carries: the OWUI GUID (default), the user's email, or their display name. Email and name make OpenRouter's dashboard human-readable but send PII with every request; empty values fall back to the GUID, and `metadata.user_id` always keeps the stable GUID so analytics keyed on it never fragment.
* `SEND_SESSION_ID` / `SEND_CHAT_ID` / `SEND_MESSAGE_ID` — add the raw id to `metadata` for incident-response traceability.
* Treat the emitted IDs as **non-secret** but sensitive operational data; they can correlate events.
