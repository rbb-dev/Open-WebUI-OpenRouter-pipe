# Encrypted session log storage (optional)

**Scope:** Optional, valve-gated persistence of per-request `SessionLogger` output to encrypted zip archives on local disk.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🔒 Security](security_and_encryption.md)

This feature is intended for operators running multi-user Open WebUI deployments who want a durable, encrypted trail of per-request logs that can be correlated using existing Open WebUI/OpenRouter identifiers.

This document covers **local filesystem storage only**. For the companion feature that sends identifiers to OpenRouter for abuse attribution, see [Request identifiers and abuse attribution](request_identifiers_and_abuse_attribution.md).

---

## What gets stored

When enabled, the pipe writes **one encrypted zip file per completed request** containing:

- `meta.json` — a small JSON document with:
  - `created_at` (UTC ISO timestamp)
  - `ids` (`user_id`, `session_id`, `chat_id`, `message_id`)
  - `request_id` (internal per-request key used for in-memory buffering)
  - `log_format` (`text`)
- `logs.txt` — plain text log lines captured in the request’s in-memory session log buffer.

Notes:

- `logs.txt` is **not** JSONL. Each line is a formatted log string and may include multi-line payloads for some messages.
- The `LOG_LEVEL` valve controls what is written to stdout/backend logs for a request. The stored archive is sourced from the in-memory session buffer and can include entries that are not emitted to stdout.
- Session logs can contain sensitive content (prompts, tool arguments, provider errors). Enable this only if you understand your retention and access controls.

---

## When an archive is written (and when it is skipped)

The pipe **skips persistence** when any of the following are true:

- `SESSION_LOG_STORE_ENABLED` is disabled.
- Any required IDs are missing/empty for the request: `user_id`, `session_id`, `chat_id`, `message_id`.
- The `pyzipper` package is unavailable at runtime.
- `SESSION_LOG_DIR` is empty.
- `SESSION_LOG_ZIP_PASSWORD` is empty/unconfigured.
- The request produced no captured log lines.

If persistence is skipped, the request still completes normally; the archive is simply not written.

Non-blocking behavior:

- Archives are written asynchronously via a bounded internal queue.
- If the archive queue is full, the pipe logs a warning and drops the archive for that request (it does not block the response).

---

## Archive layout (filesystem)

Archives are written under the base directory `SESSION_LOG_DIR` using a directory layout derived from request identifiers:

```
<SESSION_LOG_DIR>/
  <user_id>/
    <chat_id>/
      <message_id>.zip
```

Path safety:

- For filesystem safety, the `<user_id>`, `<chat_id>`, and `<message_id>` path components are **sanitized** (non-alphanumeric characters are replaced, and components are length-limited).
- The original (unsanitized) identifiers are preserved in `meta.json` under `ids.*` for correlation.

---

## Encryption and compression

Archives are written with `pyzipper` using AES encryption (`WZ_AES`).

- `SESSION_LOG_ZIP_PASSWORD` controls the password used to encrypt each archive.
  - Recommendation: set a long random passphrase and store it as an encrypted valve value (requires `WEBUI_SECRET_KEY` so Open WebUI can encrypt/decrypt the stored valve value).
- `SESSION_LOG_ZIP_COMPRESSION` selects the compression algorithm: `stored`, `deflated`, `bzip2`, or `lzma` (default `lzma`).
- `SESSION_LOG_ZIP_COMPRESSLEVEL` applies only to `deflated` and `bzip2` (0–9). It is ignored for `stored` and `lzma`.

Key rotation note:

- Changing `SESSION_LOG_ZIP_PASSWORD` affects only **future** archives. Previously written archives are not re-encrypted and require the prior password to decrypt.

---

## Retention and cleanup

When storage is enabled, a background cleanup loop periodically:

1. Deletes `*.zip` files older than `SESSION_LOG_RETENTION_DAYS` (based on file modification time).
2. Removes empty directories left behind (including the base directory if it becomes empty).

Cleanup runs every `SESSION_LOG_CLEANUP_INTERVAL_SECONDS`.

Operational note:

- Cleanup is performed per-process and tracks the session log directories that have been used in that process. In multi-worker deployments, each worker performs its own cleanup pass.

---

## Relevant valves

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for the canonical list and defaults. Session log storage is controlled by:

| Valve | Type | Default (verified) | Purpose |
|---|---:|---:|---|
| `SESSION_LOG_STORE_ENABLED` | bool | `false` | Enables writing encrypted session log archives to disk. |
| `SESSION_LOG_DIR` | str | `session_logs` | Base directory for archives. |
| `SESSION_LOG_ZIP_PASSWORD` | encrypted str | *(empty)* | Password used to encrypt archives (required to store). |
| `SESSION_LOG_RETENTION_DAYS` | int | `90` | Retention window for stored archives. |
| `SESSION_LOG_CLEANUP_INTERVAL_SECONDS` | int | `3600` | Cleanup loop interval. |
| `SESSION_LOG_ZIP_COMPRESSION` | enum | `lzma` | Zip compression algorithm. |
| `SESSION_LOG_ZIP_COMPRESSLEVEL` | int? | `null` | Compression level for deflated/bzip2. |
| `SESSION_LOG_MAX_LINES` | int | `20000` | Max in-memory lines retained per request before older lines are dropped. |

---

## Operational guidance

- Restrict access to `SESSION_LOG_DIR` (filesystem permissions, encrypted volume, backups with access controls).
- Treat archives as sensitive: they are encrypted at rest, but anyone with the zip password can decrypt them.
- If you enable request identifiers for provider-side attribution, keep the same identifiers available for operators: see [Request identifiers and abuse attribution](request_identifiers_and_abuse_attribution.md).
