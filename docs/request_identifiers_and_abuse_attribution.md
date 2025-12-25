# Request identifiers and abuse attribution

**Scope:** How the pipe emits OpenRouter request identifiers (`user`, `session_id`, `metadata`) to support abuse attribution and operations in multi-user deployments.

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

* `user` (top-level): the OWUI user GUID (`__user__["id"]`)
* `session_id` (top-level): the OWUI session id (`__metadata__["session_id"]`)
* `metadata` (top-level): a `Dict[str, str]` built by the pipe (not OWUI’s full `__metadata__` blob)

`metadata` is only sent when at least one metadata entry is being populated.

Important: the pipe **removes** any user-supplied `user`, `session_id`, or `metadata` fields and replaces them with valve-gated values. This prevents clients/users from spoofing attribution identifiers.

### Identifier mapping

Each identifier is gated by a valve. When enabled, the pipe sources IDs from Open WebUI context and maps them into OpenRouter fields as follows:

| Valve | OpenRouter top-level | OpenRouter metadata key | Source in Open WebUI context |
|---|---|---|---|
| `SEND_END_USER_ID` | `user` | `user_id` | `__user__["id"]` |
| `SEND_SESSION_ID` | `session_id` | `session_id` | `__metadata__["session_id"]` |
| `SEND_CHAT_ID` | *(none)* | `chat_id` | `__metadata__["chat_id"]` |
| `SEND_MESSAGE_ID` | *(none)* | `message_id` | `__metadata__["message_id"]` |

### Sanitization and constraints

The pipe enforces OpenRouter’s documented `metadata` constraints:

* Maximum **16** key/value pairs.
* Keys must be **≤ 64 chars** and must not contain `[` or `]`.
* Values must be **≤ 512 chars**.

Additionally, for top-level `user` and `session_id`, the pipe enforces a maximum length of **128 characters**. If the source value is missing or invalid, the corresponding field is omitted even when the valve is enabled.

---

## Pairing with encrypted session log archives (recommended)

If you’re enabling request identifiers specifically for abuse attribution / incident response, consider also enabling **encrypted session log storage**.

Why:

* OpenRouter/provider support can reference `user`, `session_id`, and/or `metadata.*` when reporting abuse or asking you to investigate.
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

### Full attribution (`user` + `session_id` + `metadata`)

```json
{
  "model": "openrouter/...",
  "input": [...],
  "user": "a3d0d2c1-7f49-4b6b-9a3b-9d3b2a54c2d1",
  "session_id": "0f6b31b0-8c9f-4c3b-a1e7-0d7d2c6b5a33",
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

* `SEND_END_USER_ID` (default: false)
* `SEND_SESSION_ID` (default: false)
* `SEND_CHAT_ID` (default: false)
* `SEND_MESSAGE_ID` (default: false)

---

## Operational guidance

* Enable `SEND_END_USER_ID` in multi-user deployments unless you have a strict policy against sharing identifiers.
* Enable `SEND_SESSION_ID` when you want provider-side grouping for multi-step agent flows / long sessions.
* Enable `SEND_CHAT_ID`/`SEND_MESSAGE_ID` if you want a direct pointer back to a specific OWUI thread/turn during incident response.
* Treat the emitted IDs as **non-secret** (they may appear in logs), but still handle them as sensitive operational data (they can enable correlation across events).
