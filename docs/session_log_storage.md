# encrypted session log storage (optional)

**file:** [session_log_storage.md](session_log_storage.md)  
**scope:** optional, valve-gated persistence of per-request SessionLogger output to encrypted zip archives.

This feature is intended for operators running multi-user Open WebUI deployments who want a durable, encrypted trail of per-request debug logs that can be correlated using only existing OpenRouter/Open WebUI identifiers.

This document covers **local filesystem storage only**. For the companion feature that sends identifiers to OpenRouter for abuse attribution, see `docs/request_identifiers_and_abuse_attribution.md`.

The pipe **never** sends additional identifiers to OpenRouter beyond the request identifiers feature (`user`, `session_id`, `metadata.user_id/session_id/chat_id/message_id`).

---

## what gets stored

When enabled, the pipe writes one encrypted zip file per completed request containing:

- `meta.json` — a small JSON document with timestamps and the four identifiers (plus an internal `request_id` used only as an in-memory log buffer key).
- `logs.txt` — plain text lines captured by the pipe’s `SessionLogger` in-memory formatter.

Notes:

- Archives include **DEBUG** log lines regardless of the configured `LOG_LEVEL`. The `LOG_LEVEL` valve only controls what is emitted to stdout/backend logs.
- `logs.txt` is **not** JSONL. Each line is a formatted log string (timestamp + level + user marker + message) and may include multi-line payloads for some log messages.
- Session logs can contain sensitive data (prompts, tool arguments, provider errors). Enable this only if you understand your retention and access controls.

---

## required identifiers (skip rule)

To avoid writing archives that cannot be correlated later, the pipe **skips persistence** unless **all** of the following are present for the request:

- `user_id`
- `session_id`
- `chat_id`
- `message_id`

If any are missing, the request still completes normally; the archive is simply not written.

In most abuse-attribution deployments you should enable this feature together with `docs/request_identifiers_and_abuse_attribution.md`, so the same identifiers you send to OpenRouter/provider support can be used to locate the matching archive on disk.

---

## archive layout (filesystem)

Archives are stored using only existing IDs so operators can locate the right file when OpenRouter/Open WebUI support references an identifier:

```
<SESSION_LOG_DIR>/
  <user_id>/
    <chat_id>/
      <message_id>.zip
```

Example:

```
session_logs/
  714affbb-d092-4e39-af42-0c59ee82ab8d/
    8ec723e5-6553-41bd-a0c6-2ab885c1bcbc/
      2e5b92d9-04b6-42ae-a1c7-602a3efdb1d2.zip
```

---

## encryption + compression

Archives are written with `pyzipper` using AES encryption (`WZ_AES`).

- `SESSION_LOG_ZIP_PASSWORD` controls the password used to encrypt each archive.
- `SESSION_LOG_ZIP_COMPRESSION` and `SESSION_LOG_ZIP_COMPRESSLEVEL` control compression.

If `pyzipper` is not available or `SESSION_LOG_ZIP_PASSWORD` is empty, the pipe logs a warning and skips persistence.

---

## retention cleanup

When storage is enabled, a background cleanup loop periodically:

1. Deletes `*.zip` files older than `SESSION_LOG_RETENTION_DAYS`.
2. Removes any empty directories left behind (including the base directory if it becomes empty).

Cleanup runs every `SESSION_LOG_CLEANUP_INTERVAL_SECONDS`.

---

## valves

See `docs/valves_and_configuration_atlas.md` for the full valve listing. The relevant knobs are:

- `SESSION_LOG_STORE_ENABLED`
- `SESSION_LOG_DIR`
- `SESSION_LOG_ZIP_PASSWORD`
- `SESSION_LOG_RETENTION_DAYS`
- `SESSION_LOG_CLEANUP_INTERVAL_SECONDS`
- `SESSION_LOG_ZIP_COMPRESSION`
- `SESSION_LOG_ZIP_COMPRESSLEVEL`
- `SESSION_LOG_MAX_LINES`
