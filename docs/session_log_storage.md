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
  - `log_format` (`jsonl`, `text`, or `both`)
- `logs.jsonl` — newline-delimited JSON (one JSON object per log record).
- `logs.txt` — plain text log output (optional; only when `SESSION_LOG_FORMAT=text|both`).
- `timing.jsonl` — function timing events (written separately to `TIMING_LOG_FILE`, not in session archives).

Notes:

- `logs.jsonl` records are always **one JSON object per line**. Multi-line payloads in the original message are stored as JSON strings with `\n` escapes.
- `logs.txt` may include multi-line payloads for some messages (for example pretty-printed JSON), so a single log record can span multiple physical lines.
- The `LOG_LEVEL` valve controls what is written to stdout/backend logs for a request. The stored archive is sourced from the in-memory session buffer and can include entries that are not emitted to stdout.
- Session logs can contain sensitive content (prompts, tool arguments, provider errors). Enable this only if you understand your retention and access controls.

---

## JSONL schema (`logs.jsonl`)

Each line in `logs.jsonl` is a single JSON object with the following keys:

- `ts` (string): UTC ISO 8601 timestamp (milliseconds), for example `2026-01-01T06:44:31.491Z`.
- `level` (string): Log level name (e.g. `DEBUG`, `INFO`, `WARNING`, `ERROR`).
- `logger` (string): Logger name.
- `request_id` (string): Internal per-request identifier used for in-memory buffering.
- `user_id` (string): Open WebUI user id (from request context).
- `session_id` (string): Open WebUI session id (from request context).
- `chat_id` (string): Open WebUI chat id (from archive metadata).
- `message_id` (string): Open WebUI message id (from archive metadata).
- `event_type` (string): Coarse classification derived from the message prefix (for example `openrouter.request.payload`, `openrouter.sse.event`, `pipe.tools`, `pipe`).
- `module` (string): Python module name.
- `func` (string): Function name.
- `lineno` (int): Source line number.
- `exception` (object, optional): When present, contains `text` (string) with the formatted exception/traceback.
- `message` (string): The formatted log message (may be large).

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
| `SESSION_LOG_MAX_LINES` | int | `20000` | Max in-memory log records retained per request before older entries are dropped. |
| `SESSION_LOG_FORMAT` | enum | `jsonl` | Archive log file format: `jsonl` writes `logs.jsonl`, `text` writes `logs.txt`, `both` writes both files. |
| `ENABLE_TIMING_LOG` | bool | `false` | Capture function entrance/exit timing data. |
| `TIMING_LOG_FILE` | str | `logs/timing.jsonl` | File path for timing log output. |

---

## Timing instrumentation

When `ENABLE_TIMING_LOG=True`, timing events are written directly to `TIMING_LOG_FILE` (default: `logs/timing.jsonl`) with high-precision function entrance/exit data.

### JSONL schema (`timing.jsonl`)

Each line is a single JSON object:

- `ts` (string): UTC ISO 8601 timestamp (milliseconds), for example `2026-01-18T12:34:56.789Z`.
- `perf_ts` (float): High-resolution monotonic counter (`time.perf_counter()`) for precise elapsed-time calculations.
- `event` (string): One of `enter`, `exit`, or `mark`.
- `label` (string): Function or scope name (for example `streaming.streaming_core.StreamingHandler._run_streaming_loop`).
- `elapsed_ms` (float, optional): Elapsed time in milliseconds (only present on `exit` events).

Example output:

```jsonl
{"ts":"2026-01-18T12:34:56.001Z","perf_ts":0.001234,"event":"enter","label":"streaming.streaming_core.StreamingHandler._run_streaming_loop"}
{"ts":"2026-01-18T12:34:56.002Z","perf_ts":0.002345,"event":"mark","label":"event_iteration_start"}
{"ts":"2026-01-18T12:34:56.050Z","perf_ts":0.050678,"event":"mark","label":"first_event_received"}
{"ts":"2026-01-18T12:34:58.123Z","perf_ts":2.123456,"event":"exit","label":"streaming.streaming_core.StreamingHandler._run_streaming_loop","elapsed_ms":2122.22}
```

### When to use timing

Enable timing when diagnosing performance issues:

- **Slow response times**: Identify which functions take the most time.
- **Unexpected delays**: Find gaps between function exits and entries.
- **Optimization verification**: Measure before/after improvements.

### Adding timing to new functions (for developers)

The timing system provides three mechanisms:

**1. `@timed` decorator** — automatic function entrance/exit timing:

```python
from open_webui_openrouter_pipe.core.timing_logger import timed

@timed
async def my_function():
    # Function calls are automatically timed
    ...
```

**2. `timing_scope()` context manager** — time specific code blocks:

```python
from open_webui_openrouter_pipe.core.timing_logger import timing_scope

async def process_request():
    with timing_scope("http_post"):
        async with session.post(...) as resp:
            ...
```

**3. `timing_mark()` function** — record point-in-time events:

```python
from open_webui_openrouter_pipe.core.timing_logger import timing_mark

async def stream_events():
    timing_mark("stream_start")
    async for event in event_iter:
        if first_event:
            timing_mark("first_event_received")
        ...
```

**Notes:**

- Timing only records when `ENABLE_TIMING_LOG=True` and a request context is active.
- Zero overhead when disabled (context variable check short-circuits).
- Events are written immediately to `TIMING_LOG_FILE` (not session archives).
- Each event includes a `request_id` field for correlation with session logs.
- Maximum 10,000 events per request in memory (oldest dropped if exceeded).

---

## Operational guidance

- Restrict access to `SESSION_LOG_DIR` (filesystem permissions, encrypted volume, backups with access controls).
- Treat archives as sensitive: they are encrypted at rest, but anyone with the zip password can decrypt them.
- If you enable request identifiers for provider-side attribution, keep the same identifiers available for operators: see [Request identifiers and abuse attribution](request_identifiers_and_abuse_attribution.md).
