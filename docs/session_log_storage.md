# Encrypted session log storage (optional)

**Scope:** Optional, valve-gated persistence of `SessionLogger` output to encrypted zip archives on local disk, assembled per **message turn**.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🔒 Security](security_and_encryption.md)

This feature is intended for operators running multi-user Open WebUI deployments who want a durable, encrypted trail of per-request logs that can be correlated using existing Open WebUI/OpenRouter identifiers.

This document covers **local filesystem storage only**. For the companion feature that sends identifiers to OpenRouter for abuse attribution, see [Request identifiers and abuse attribution](request_identifiers_and_abuse_attribution.md).

---

## What gets stored

When enabled, the pipe writes **one encrypted zip file per message turn** (Open WebUI `message_id`) containing logs from:

A call that arrives with no usable `chat_id`/`message_id` — the plain API route — is the one exception to the "per message turn" shape: it is keyed on its request id and written as `api/api-<request_id>.zip`, one file per request, while `SESSION_LOG_ARCHIVE_API_CALLS` is on.

- the initial user send → model response
- any intermediate OpenRouter traffic
- any tool calls/results that occur within the turn

One zip is written per message turn, plus one for each housekeeping task Open WebUI dispatches on that turn, named `<message_id>.<task>.zip`. Open WebUI defines nine task types, so a turn that triggers all of them produces up to ten archives. That is deliberate: the operator debugs from these archives, and a title or tags task folded into the answer's file would make its traffic unattributable. A task invocation with no resolvable message id is skipped entirely.

A reply that may hand a call back -- in Open-WebUI mode, or in Pipeline mode for a tool the pipe cannot run -- may be re-invoked for the same `message_id` during its tool loops. In that case, the pipe stages per-invocation log “segments” into the persistence layer and a background assembler merges them into a single archive.

- `meta.json` — a small JSON document with:
  - `created_at` (UTC ISO timestamp)
  - `ids` (`user_id`, `session_id`, `chat_id`, `message_id`) — `api` / `api-<request_id>` for an API call
  - `request_id` (a representative internal per-request key used for in-memory buffering)
  - `request_ids` (optional; sorted unique list of per-request identifiers found in the bundled events)
  - `log_format` (`jsonl`, `text`, or `both`)
- `logs.jsonl` — newline-delimited JSON (one JSON object per log record).
- `logs.txt` — plain text log output (optional; only when `SESSION_LOG_FORMAT=text|both`).
- `timing.jsonl` — function timing events (written separately to `TIMING_LOG_FILE`, not in session archives).

Notes:

- `logs.jsonl` records are always **one JSON object per line**. Multi-line payloads in the original message are stored as JSON strings with `\n` escapes.
- `logs.txt` writes **one log record per physical line**, with one exception. A control character in a *message* (including the newline-like separators `\x85`, U+2028 and U+2029) is replaced with a space rather than emitted, so a newline inside a user-supplied URL or a provider error cannot append a line shaped like a real record. The text stays present and readable, escaped in place. A record's **exception block** — the traceback `logger.exception(...)` captured — may span lines, because a traceback legitimately does; any line in it that would itself render as a record is de-formed first, so the forged line cannot stand on its own. The `logs.jsonl` guarantee above is unchanged and needs no counterpart here: it is one JSON object per line by construction, and that is exactly what re-assembly reads.
- `logs.jsonl` is the **evidence** and the text sinks are the **presentation**, so for a record carrying a neutralised character the two renderings differ: the archive keeps the caller's original bytes, while the console and `logs.txt` show the neutralised text.
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
- Any required IDs are missing/empty for the request: `user_id` or `request_id`.
- The request carries no usable `chat_id` or `message_id` and `SESSION_LOG_ARCHIVE_API_CALLS` is off. With it on (the default) such a call is archived as `api/api-<request_id>.zip` instead of being skipped.
- The `pyzipper` package is unavailable at runtime.
- `SESSION_LOG_DIR` is empty.
- `SESSION_LOG_ZIP_PASSWORD` is empty/unconfigured.
- The request produced no captured log lines.
- The chat is a temporary chat, which Open WebUI keeps only in the browser; the pipe stores nothing for it either. In Open-WebUI tool mode the tool rounds and thinking of a streamed reply are held in memory for that reply only, and dropped when the pipe answers its last call back or after 15 minutes unused.

If persistence is skipped, the request still completes normally; the archive is simply not written.

### Assembly timing

Archives are written by a background assembler thread when:

- a “terminal” segment is staged for the message key (final assistant answer, error, or cancellation). The key is `(chat_id, message_id)` for a chat turn and the request surrogate `("api", "api-<request_id>")` for an API call, or
- no terminal segment arrives for a long time (configurable “stale finalize”) — an **incomplete** archive is written so crash/cancel cases still leave a durable log trail. Each pass takes the oldest stranded bundles first and builds each archive from whatever segments exist at that moment, so a turn whose newest segment predates the delay is sealed **even if the turn is still running**; a segment it stages after that seal is left stranded until the next assembly, which may never come. The delay is what bounds that exposure, and it is why the default is long. The seal itself is conditional: if the log directory or passphrase no longer resolves, or the sealed write fails, the segments are kept for a retry and are removed only by the `ARTIFACT_CLEANUP_DAYS` sweep, which is not gated on the session-log valve. A later terminal segment for the same turn merges into that same zip and removes the finalized-incomplete line.

The incomplete marker appears **at most once** per archive: a later pass that finds the turn complete retires the marker rather than adding a second, and a pass that finds it already present leaves the count at one. A **refused pass resets the staleness clock**, so a stale turn waits a full window again before the assembler retries it.

Non-blocking behavior:

- Archives are written asynchronously via a bounded internal queue.
- If the archive queue is full, the pipe logs a warning and drops the archive for that request (it does not block the response).
- When the archive cannot be staged in the database, the pipe writes it through that same bounded queue instead of writing it inline, so the request never waits for compression. The path is therefore lossy in the same way: during a long database outage a busy queue fills, and the session logs for later requests in that period are dropped with the warning above rather than queued indefinitely.

---

## Archive layout (filesystem)

Archives are written under the base directory `SESSION_LOG_DIR` using a directory layout derived from request identifiers:

```
<SESSION_LOG_DIR>/
  <user_id>/
    <chat_id>/
      <message_id>.zip
      <message_id>.<task>.zip
```

An API call has no `chat_id` and no `message_id`, so letting the empty strings through would collide: sanitisation turns `""` into a fixed literal per slot and every API call from every user would land on one shared file. The pipe therefore substitutes the request-scoped surrogate pair `api` / `api-<request_id>` before staging, which is the same key the assembler gates on, so a surrogate archive is packed and written like any other:

```
<SESSION_LOG_DIR>/
  <user_id>/
    api/
      api-<request_id>.zip
```

The task files sit beside the answer's in the same `<chat_id>/` directory and keep the `message_id` as their prefix, so `ls <chat_id>/` and `grep <message_id>` both still work. The message id is truncated from the right to fit a 64-character column, with the task name's space reserved first — the qualifier is never the part that gets cut.

**Scope:** task archives are written for the housekeeping tasks Open WebUI dispatches with a resolvable `message_id` and a `task` name. **Fusion panel members are not archived** — their metadata deliberately carries no `chat_id` or `message_id` and no `task`, so they resolve to nothing and their segments are dropped.

Path safety:

- For filesystem safety, the `<user_id>`, `<chat_id>`, and `<message_id>` path components are **sanitized** (non-alphanumeric characters are replaced, and components are length-limited).
- The original (unsanitized) identifiers are preserved in `meta.json` under `ids.*` for correlation. A task archive's `ids.message_id` is the **bare** Open WebUI message id — the one that field can be joined against — with the task qualifier in the **filename** and in a separate `ids.task`, not appended to `ids.message_id`. An answer archive has no qualifier, so it carries no `ids.task` at all.

---

## Encryption and compression

Archives are written with `pyzipper` using AES encryption (`WZ_AES`).

- `SESSION_LOG_ZIP_PASSWORD` controls the password used to encrypt each archive.
  - Recommendation: set a long random passphrase and store it as an encrypted valve value (requires `WEBUI_SECRET_KEY` so Open WebUI can encrypt/decrypt the stored valve value).
- `SESSION_LOG_ZIP_COMPRESSION` selects the compression algorithm: `stored`, `deflated`, `bzip2`, or `lzma` (default `lzma`).
- `SESSION_LOG_ZIP_COMPRESSLEVEL` applies only to `deflated` and `bzip2` (0–9). It is ignored for `stored` and `lzma`.

Key rotation note:

- Changing `SESSION_LOG_ZIP_PASSWORD` affects only **future** archives. Previously written archives are not re-encrypted and require the prior password to decrypt.
- An archive sealed under a previous passphrase is left **byte-for-byte alone** and keeps that passphrase until retention expires it. **A message turn that is still receiving segments when you rotate can never be completed**: the existing archive cannot be read, so the pass refuses, and the turn's remaining segments are never merged into it. After three refused passes they are instead written to a separate `<message_id>.<request_id>.zip` that opens with the **new** passphrase, while `<message_id>.zip` still needs the old one; those segments are then removed from the database, and the turn's own archive still does not contain them. That separate file ages like any other archive and is reaped by `Archive retention period`. **Rotate between turns, not during one.** To capture a turn's segments before rotating, copy them out of the database.

---

## Retention and cleanup

When storage is enabled, a background cleanup loop periodically:

1. Deletes `*.zip` files older than `SESSION_LOG_RETENTION_DAYS` (based on file modification time).
2. Removes empty directories left behind (including the base directory if it becomes empty).

Cleanup runs every `SESSION_LOG_CLEANUP_INTERVAL_SECONDS`. Turning `SESSION_LOG_STORE_ENABLED` off stops the sweep entirely: no archive is deleted and no directory is pruned, and every archive already on disk is left exactly where it is until you re-enable storage and the retention window passes again.

With storage disabled the pipe neither writes nor deletes archives: turning the valve off stops the sweep on the next pass, and archives already on disk are left untouched until it is on again. Anything past `SESSION_LOG_RETENTION_DAYS` at that point is reclaimed on the first pass after it is turned back on.

Additionally, once an archive is assembled, the per-invocation DB segments used to build it are deleted. A separate “stale finalize” path can assemble + delete segments for abandoned turns after a long timeout; it is a cutoff on each turn’s last segment rather than on the turn itself, so a turn still running when it passes is sealed and its later segment is stranded until the next assembly.

Operational note:

- Cleanup is performed per-process and tracks the session log directories that have been used in that process. In multi-worker deployments, each worker performs its own cleanup pass.

---

## Relevant valves

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for the canonical list and defaults. Session log storage is controlled by:

| Valve | Type | Default (verified) | Purpose |
|---|---:|---:|---|
| `SESSION_LOG_STORE_ENABLED` | bool | `false` | Enables writing encrypted session log archives to disk. |
| `SESSION_LOG_ARCHIVE_API_CALLS` | bool | `true` | Archives a request with no usable `chat_id`/`message_id` under `api/api-<request_id>.zip`. A staging gate only — a segment staged while it was on is still written after it is switched off. |
| `SESSION_LOG_DIR` | str | `session_logs` | Base directory for archives. |
| `SESSION_LOG_ZIP_PASSWORD` | encrypted str | *(empty)* | Password used to encrypt archives (required to store). |
| `SESSION_LOG_RETENTION_DAYS` | int | `90` | Retention window for stored archives. |
| `SESSION_LOG_CLEANUP_INTERVAL_SECONDS` | int | `3600` | Cleanup loop interval. |
| `SESSION_LOG_ZIP_COMPRESSION` | enum | `lzma` | Zip compression algorithm. |
| `SESSION_LOG_ZIP_COMPRESSLEVEL` | int? | `null` | Compression level for deflated/bzip2. |
| `SESSION_LOG_MAX_LINES` | int | `20000` | Max in-memory log records retained per request before older entries are dropped. |
| `SESSION_LOG_FORMAT` | enum | `jsonl` | Archive log file format. `logs.jsonl` is always written as the canonical record; `jsonl` writes only `logs.jsonl`, while `text` and `both` additionally write `logs.txt` (so `text` and `both` produce an identical file set). `logs.txt` writes one record per physical line, except for a record's exception block. |
| `SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS` | int | `30` | How often each process scans the DB for completed/stale turns to assemble into zip archives. |
| `SESSION_LOG_ASSEMBLER_JITTER_SECONDS` | int | `10` | Per-process jitter added to the assembler loop to avoid multi-worker lockstep. |
| `SESSION_LOG_ASSEMBLER_BATCH_SIZE` | int | `25` | Max turns processed per assembler tick — a cap on turns, not rows, so a heavily split turn still takes one slot. |
| `SESSION_LOG_STALE_FINALIZE_SECONDS` | int | `43200` | If no terminal segment arrives for a turn, assemble an **incomplete** archive after this timeout. A cutoff on the last segment, not on the turn: a turn **still running** when it passes is sealed as incomplete too, and a segment it stages afterwards is left stranded until the next assembly. That exposure is why the default is long. Each pass takes the oldest stranded bundles first and seals a bundle only if the sealed write succeeds, keeping the segments for a retry otherwise. |
| `SESSION_LOG_LOCK_STALE_SECONDS` | int | `1800` | DB lock row stale timeout (multi-worker safety). |
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
- `request_id` (string): The pipe's per-request correlation id; present on every event.
- `elapsed_ms` (float, optional): Elapsed time in milliseconds (only present on `exit` events).

Example output:

```jsonl
{"ts":"2026-01-18T12:34:56.001Z","perf_ts":0.001234,"event":"enter","label":"streaming.streaming_core.StreamingHandler._run_streaming_loop","request_id":"a1b2c3d4e5f6a7b8"}
{"ts":"2026-01-18T12:34:56.002Z","perf_ts":0.002345,"event":"mark","label":"event_iteration_start","request_id":"a1b2c3d4e5f6a7b8"}
{"ts":"2026-01-18T12:34:56.050Z","perf_ts":0.050678,"event":"mark","label":"first_event_received","request_id":"a1b2c3d4e5f6a7b8"}
{"ts":"2026-01-18T12:34:58.123Z","perf_ts":2.123456,"event":"exit","label":"streaming.streaming_core.StreamingHandler._run_streaming_loop","request_id":"a1b2c3d4e5f6a7b8","elapsed_ms":2122.22}
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
