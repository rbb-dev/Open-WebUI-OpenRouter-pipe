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

One zip is written per message turn, plus one for each housekeeping task Open WebUI dispatches on that turn, named `<message_id>.<task>.zip`. Open WebUI defines nine task types in its `TASKS` enum plus three more it names inline (`context_compaction`, `memory_review`, `context_summary`), so a turn that triggers all of them produces up to thirteen archives. The count is checkable rather than asserted: `tests/test_session_logs.py::test_the_archive_key_split_takes_every_task_name_open_webui_sends` reads the names out of the Open WebUI source under `.external/` and fails when a name it has not recorded appears. That is deliberate: the operator debugs from these archives, and a title or tags task folded into the answer's file would make its traffic unattributable. A task invocation with no resolvable message id is skipped entirely.

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
- Every media field appears as scheme, host and port only — the path and query are dropped, so a signed CDN link cannot be recovered from an archive. (The fields are the media-URL keys the pipe keys on, not a list maintained here: it goes stale the moment one is added.) A `data:` URL contributes only its media type (for example `data:image/png`); its bytes never reach a log, including when the URL carries no comma. A `data:` URL in a tool result, in any record, contributes only its media type and never its `;name=` parameter — except where the URL's own header carries whitespace before its payload.

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
- The chat is a temporary chat, which Open WebUI keeps only in the browser; the pipe stores nothing for it either. In Open-WebUI tool mode the tool rounds and thinking of a streamed reply are held in memory for that reply only, and only for the user who opened that reply, and dropped when the pipe answers its last call back, or when the provider refuses a call-back the pipe was waiting for, or after 15 minutes unused.

If persistence is skipped, the request still completes normally; the archive is simply not written.

Each skip names itself at the level the code uses for it. On the segment-persist path the missing-id and valve-off skips are at `INFO`, the store-disabled, no-events and archive-settings-unavailable skips at `DEBUG`, and the passphrase, log-directory and `pyzipper` skips at `WARNING`. The enqueue path records the archive-settings skips at `WARNING` and the missing-id and valve-off skips at `INFO`, but records nothing at all when storage is off or when the request produced no captured log lines. The temporary-chat skips on the three archive paths (enqueue, segment persist, bundle assembly) warn once per path and warn again after a five-minute cooldown.

### Assembly timing

Archives are written by a background assembler thread when:

- a “terminal” segment is staged for the message key (final assistant answer, error, or cancellation). The key is `(chat_id, message_id)` for a chat turn and the request surrogate `("api", "api-<request_id>")` for an API call, or
- no terminal segment arrives for a long time (configurable “stale finalize”) — an **incomplete** archive is written so crash/cancel cases still leave a durable log trail. Each pass takes the oldest stranded bundles first and builds each archive from whatever segments exist at that moment, so a turn whose newest segment predates the delay is sealed **even if the turn is still running**; a segment it stages after that seal is picked up by the next assembly, which re-checks for a terminal segment first - one that lands among the rows the sealing pass loads is folded into that same archive by that pass, and one that lands later — while the pass is still writing that archive, or after it — is still picked up by a later pass, which is not guaranteed to come if the archive directory or passphrase stops resolving. The delay is what bounds that exposure, and it is why the default is long. The seal itself is conditional: if the log directory or passphrase no longer resolves, or the sealed write fails, the segments are kept for a retry and are removed only by the `ARTIFACT_CLEANUP_DAYS` sweep, which is not gated on the session-log valve. A later terminal segment for the same turn merges into that same zip and removes the finalized-incomplete line.

A bundle the assembler cannot write does not hold the window: the failed turn keeps its segments for retry, and is skipped by the next `SESSION_LOG_ASSEMBLER_BATCH_SIZE` window so one unwritable bundle delays only itself. `SESSION_LOG_LOCK_STALE_SECONDS` is the retry interval — the failed turn is dropped from both listings for that long, after which it is retried on the normal schedule and the pass after that seals it — so a transient failure recovers on its own, and nothing is discarded. Each of the two loss points is reported at WARNING naming the turn it holds up: a bundle whose sealed write fails names the turn and the archive path, and a bundle whose archive settings do not resolve (a blank passphrase, a blank log directory, a missing `pyzipper`) is reported the same way and per turn, rather than once for the worker's lifetime.

The incomplete marker appears **at most once** per archive: a later pass that finds the turn complete retires the marker rather than adding a second, and a pass that finds it already present leaves the count at one. A **refused pass does not restart the finalize countdown**: the staleness clock advances only when a bundle is published, so a turn whose write failed is offered again rather than waiting a full window.

Non-blocking behavior:

- Archives are written asynchronously via a bounded internal queue.
- If the archive queue is full, the pipe logs a warning and drops the archive for that request (it does not block the response).
- When the archive cannot be staged in the database, the pipe writes it through that same bounded queue instead of writing it inline, so the request never waits for compression. The path is therefore lossy in the same way: during a long database outage a busy queue fills, and the session logs for later requests in that period are dropped with the warning above rather than queued indefinitely.
- On stop, a shutdown or an Open WebUI hot reload, the writer drains the archives it has already accepted within a bounded window (about one second) rather than abandoning the queue the moment the stop signal arrives, and reports at WARNING how many queued archives it could not write. A restart is therefore no longer the point at which an in-flight burst is lost, including on the route where the manager is collected during shutdown, where the same bounded drain runs and the count is reported through the module logger because there is no manager left to log through. The request queue is drained on the same contract: `close()` finishes its drain even when the worker and its jobs are bound to a closed loop, and each abandoned job's counter is released rather than left held.

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

Each archive is written to a temporary file in the same directory, flushed to disk, and then published with an atomic rename. The temporary file's name carries the writer's process id and a random suffix, so concurrent writers to one turn never share it. The result is that when two writers race, each publishes a complete archive rather than interleaving into one: a published archive is always openable, even though a later publish replaces an earlier one's content rather than merging with it.

Before it creates anything, a write **reserves** the directory it is about to fill, and the cleanup sweep checks that reservation and removes the directory under the same lock, so a sweep can never take the directory out from under a write in progress. The guarantee is per process, and it is about the reservation being registered *before* the sweep's check — which is why the claim is taken before the directory is created rather than after. Once the write returns the reservation is dropped and a later sweep prunes the directory as usual.

**Scope:** task archives are written for the housekeeping tasks Open WebUI dispatches with a resolvable `message_id` and a `task` name. **Fusion panel members are not archived** — they carry no `message_id` and no `task`, so they resolve to nothing and their segments are dropped.

Path safety:

- For filesystem safety, the `<user_id>`, `<chat_id>`, and `<message_id>` path components are **sanitized** (non-alphanumeric characters are replaced, and components are length-limited).
- The original (unsanitized) identifiers are preserved in `meta.json` under `ids.*` for correlation. A task archive's `ids.message_id` is the **bare** Open WebUI message id — the one that field can be joined against — with the task qualifier in the **filename** and in a separate `ids.task`, not appended to `ids.message_id`. An answer archive has no qualifier, so it carries no `ids.task` at all.
- The split tests the key's tail against a known list of task names, not against a shape. A name Open WebUI sends that the pipe has never seen leaves the qualifier **in** `ids.message_id` (`msg-1.brand_new_task`) and leaves `ids.task` absent. That is a loud, visible gap rather than a silent misreading: the composed key is what the filename already says, so the archive is still findable — but `ids.message_id` does not join against an Open WebUI message. `tests/test_session_logs.py::test_the_archive_key_split_takes_every_task_name_open_webui_sends` is the row that turns a new name into a failing test the day Open WebUI ships it; note that it reads `.external/`, which is gitignored, so it **skips on CI and on a fresh checkout** and guards only where the Open WebUI source is present (local and farm runs).

---

## Encryption and compression

Archives are written with `pyzipper` using AES encryption (`WZ_AES`).

- `SESSION_LOG_ZIP_PASSWORD` controls the password used to encrypt each archive.
  - Recommendation: set a long random passphrase and store it as an encrypted valve value (requires `WEBUI_SECRET_KEY` so Open WebUI can encrypt/decrypt the stored valve value).
- `SESSION_LOG_ZIP_COMPRESSION` selects the compression algorithm: `stored`, `deflated`, `bzip2`, or `lzma` (default `lzma`).
- `SESSION_LOG_ZIP_COMPRESSLEVEL` applies only to `deflated` and `bzip2` (0–9). It is ignored for `stored` and `lzma`.

Key rotation note:

- Changing `SESSION_LOG_ZIP_PASSWORD` affects only **future** archives. Previously written archives are not re-encrypted and require the prior password to decrypt.
- An archive sealed under a previous passphrase is left **byte-for-byte alone** and keeps that passphrase until retention expires it. **A message turn that is still receiving segments when you rotate can never be completed**: the existing archive cannot be read, so the pass refuses, and the turn's remaining segments are never merged into it. After three refused passes they are instead written to a separate `<message_id>.<request_id>.zip` that opens with the **new** passphrase, while `<message_id>.zip` still needs the old one; those segments are then removed from the database — if that separate write itself fails, the segments stay staged for a later pass — and the turn's own archive still does not contain them. That separate file ages like any other archive and is reaped by `Archive retention period`. **Rotate between turns, not during one.** To capture a turn's segments before rotating, copy them out of the database.

---

## Retention and cleanup

When storage is enabled, a background cleanup loop periodically:

1. Deletes `*.zip` files older than `SESSION_LOG_RETENTION_DAYS` (based on file modification time).
2. Removes empty directories left behind (including the base directory if it becomes empty), except a directory a write in this process is currently filling; that one is left alone. A write that fails leaves its directory to a later sweep, while a process that dies mid-write leaves a `*.zip.tmp` that no pass reaps, because the prune leg only removes directories that are empty.

Cleanup runs every `SESSION_LOG_CLEANUP_INTERVAL_SECONDS`. Turning `SESSION_LOG_STORE_ENABLED` off stops the sweep entirely: no archive is deleted and no directory is pruned, and every archive already on disk is left exactly where it is until you re-enable storage and the retention window passes again.

With storage disabled the pipe neither writes nor deletes archives: turning the valve off stops the sweep on the next pass, and archives already on disk are left untouched until it is on again. The sweep reads `SESSION_LOG_RETENTION_DAYS` on every pass, so a changed window applies from the next sweep with no restart and no new turn.

Additionally, once an archive is assembled, the per-invocation DB segments used to build it are deleted. A separate “stale finalize” path can assemble + delete segments for abandoned turns after a long timeout; it is a cutoff on each turn’s last segment rather than on the turn itself, so a turn still running when it passes is sealed, and its later segment is picked up by the next assembly, which re-checks for a terminal segment first; one that lands among the rows the sealing pass loads is folded into that same archive by that pass, and one that lands later — while the pass is still writing that archive, or after it — is still picked up by a later pass, which is not guaranteed to come if the archive directory or passphrase stops resolving.

Operational note:

- Cleanup is performed per-process and tracks the session log directories that have been used in that process. In multi-worker deployments, each worker performs its own cleanup pass.
- A write reserves the directory it is filling, and the sweep honours that reservation, so a sweep does not remove a directory a write in the same process is in the middle of. The reservation is **per process**: another Open WebUI worker on the same host runs its own sweep with its own set of reservations, so a directory one worker is filling can still be pruned by a peer. Within a process the check and the removal are under one lock, so a write that reserves a directory before the sweep's check is never removed by that pass.

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
| `SESSION_LOG_STALE_FINALIZE_SECONDS` | int | `43200` | If no terminal segment arrives for a turn, assemble an **incomplete** archive after this timeout. A cutoff on the last segment, not on the turn: a turn **still running** when it passes is sealed as incomplete too, and a segment it stages afterwards is left stranded until the next assembly, unless it lands before the sealing pass has read that turn's segments - a pass that finds a terminal segment among the rows it loaded writes the turn complete instead; one that lands after that read is not folded into that archive by that pass and is picked up by a later one. That exposure is why the default is long. Each pass takes the oldest stranded bundles first and seals a bundle only if the sealed write succeeds, keeping the segments for a retry otherwise. |
| `SESSION_LOG_LOCK_STALE_SECONDS` | int | `1800` | DB lock row stale timeout (multi-worker safety), and the write-failure backoff: a bundle whose archive could not be written is skipped for this long before it is retried. A lock held by a peer is not a write failure, so a turn another worker is already archiving is never backed off at all. |
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
- `request_id` (string): The pipe's per-request correlation id. Present on every event that belongs to a request; the pipe's process-lifetime background workers — the request dispatcher, the log worker and the artifact-cleanup sweep — are started on an empty context and are deliberately not attributed to any request, so their frames do not appear. The one-shot startup warmup does carry the id of the request that started it.
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

### The in-memory copy

Alongside the file the logger keeps an in-memory copy of each request's events, so a running request can be read without parsing the file. That copy is a duplicate of a durable record and is released as soon as the request's job completes — the file is what persists, and only the file. `MAX_TIMING_REQUESTS` is the backstop for requests that end abnormally and never reach the release, not the normal retention path.

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
