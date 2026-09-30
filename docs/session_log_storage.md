# Encrypted session log storage (optional)

**Scope:** Optional, valve-gated persistence of `SessionLogger` output to encrypted zip archives on local disk, assembled per **message turn**.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🔒 Security](security_and_encryption.md)

This feature is intended for operators running multi-user Open WebUI deployments who want a durable, encrypted trail of per-request logs that can be correlated using existing Open WebUI/OpenRouter identifiers.

This document covers **local filesystem storage only**. For the companion feature that sends identifiers to OpenRouter for abuse attribution, see [Request identifiers and abuse attribution](request_identifiers_and_abuse_attribution.md).

---

## What gets stored

When enabled, the pipe writes **one encrypted zip file per message turn** (Open WebUI `message_id`) containing logs from — with two sources of calls keyed on the request id instead, and landing in the `api/` tree rather than in a turn's own directory:

Two sources of calls arrive with no usable `message_id`, and both are the exception to the "per message turn" shape. The first is the plain API route, which Open WebUI supplies neither id to. The second is every inner call an internal Fusion deliberation makes — each panel member, the judge, the judge's second pass when it runs, and the synthesis — which carries the outer turn's `chat_id` but no `message_id` of its own. Both are keyed on their own request id and written as `api/api-<request_id>.zip`, one file per call, while `SESSION_LOG_ARCHIVE_API_CALLS` is on.

- the initial user send → model response
- any intermediate OpenRouter traffic
- any tool calls/results that occur within the turn

One zip is written per message turn, plus one for each housekeeping task Open WebUI dispatches on that turn, named `<message_id>.<task>.zip`. Open WebUI defines nine task types in its `TASKS` enum plus three more it names inline (`context_compaction`, `memory_review`, `context_summary`), and the pipe dispatches a thirteenth of its own — the video-intent classifier, as `video_intent_v1` — so a turn that triggers all of them produces up to fourteen archives. The count is checkable rather than asserted: `tests/test_session_logs.py::test_the_archive_key_split_takes_every_task_name_open_webui_sends` reads the names out of the Open WebUI source under `.external/` and fails when a name it has not recorded appears. That is deliberate: the operator debugs from these archives, and a title or tags task folded into the answer's file would make its traffic unattributable. A task invocation with no resolvable message id is skipped entirely.

A reply that may hand a call back -- in Open-WebUI mode, or in Pipeline mode for a tool the pipe cannot run -- may be re-invoked for the same `message_id` during its tool loops. In that case, the pipe stages per-invocation log “segments” into the persistence layer and a background assembler merges them into a single archive.

- `meta.json` — a small JSON document with:
  - `created_at` (UTC ISO timestamp)
  - `ids` (`user_id`, `session_id`, `chat_id`, `message_id`) — `api` / `api-<request_id>` for an API call
  - `request_id` (a representative internal per-request key used for in-memory buffering)
  - `request_ids` (optional; sorted unique list of per-request identifiers found in the bundled events)
  - `status` (optional; how the turn ended — `complete`, `error`, `cancelled` or `needs_tool`; **absent** when no terminal segment recorded one and no earlier pass recorded one for that archive)
  - `reason` (optional; the cause, from the same segment as `status`; absent when empty)
  - `log_format` (`jsonl`, `text`, or `both`)
  - `terminal` (`true` when the pass that wrote this archive had a terminal segment for the turn; `false` when it sealed the turn as incomplete. A later pass reads it, so a segment that lands after the turn finished is merged without sealing the turn again)
- `logs.jsonl` — newline-delimited JSON (one JSON object per log record).
- `logs.txt` — plain text log output (optional; only when `SESSION_LOG_FORMAT=text|both`).
- `timing.jsonl` — function timing events (written separately to `TIMING_LOG_FILE`, not in session archives).

`status` and `reason` are taken from the **terminal** segment of the bundle, so a tool-loop turn records `complete` rather than the `needs_tool` of its first round. Neither key is written when the turn never reached a terminal segment and no earlier pass recorded one; `logs.jsonl` records `Session log finalized as incomplete` in that case instead.

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
- The invocation is a **task** that resolved to no message id. It is skipped whether `SESSION_LOG_ARCHIVE_API_CALLS` is on or off: it belongs to no message turn, and the valve's promise is about the plain API route, not about it. A task that *does* resolve keeps its own `<message_id>.<task>.zip` whatever its chat id.
- The `pyzipper` package is unavailable at runtime.
- `SESSION_LOG_DIR` is empty or only whitespace.
- `SESSION_LOG_ZIP_PASSWORD` is empty/unconfigured.
- The request produced no captured log lines.
- The chat is a temporary chat, which Open WebUI keeps only in the browser; the pipe stores nothing for it either. In Open-WebUI tool mode the tool rounds and thinking of a streamed reply are held in memory for that reply only, and only for the user who opened that reply, and dropped when the pipe answers its last call back, or when the provider refuses a call-back the pipe was waiting for, or when the reply is stopped, or after 15 minutes unused.

If persistence is skipped, the request still completes normally; the archive is simply not written.

Each skip names itself at the level the code uses for it. On the segment-persist path the missing-id and valve-off skips warn once each at `WARNING` and record every later skip at `DEBUG`, the store-disabled, no-events and archive-settings-unavailable skips at `DEBUG`, and the passphrase, log-directory and `pyzipper` skips at `WARNING`. An archive dropped because the writer queue is full also warns once per five-minute cooldown per worker and records every later drop at `DEBUG`, each record naming the running count of archives that worker has dropped. The store-disabled record names the `chat_id` and `message_id` it would otherwise have used, so an operator can find the archive that is not there, except for a temporary chat: its store-disabled record names the `request_id` alone, because a temporary chat's id is the browser's socket id and the record that follows the same convention on the enabled path names nothing but the request id. The temporary-chat skips on the two archive paths (segment persist, bundle assembly) warn again after a five-minute cooldown, each on the scope it can actually tell apart: the segment-persist skip is keyed on the user, so it warns once per person, and the bundle-assembly skip is called with a chat id and a message id and no user at all, so it warns once per worker process. A temporary chat's refused turn is not recorded as a write failure, so on that path "skipped" means the turn was dropped and will not be retried, where a saved chat whose bundle cannot be written is backed off and offered again.

### Assembly timing

Archives are written by a background assembler thread when:

- a “terminal” segment is staged for the message key (final assistant answer, error, or cancellation). The key is `(chat_id, message_id)` for a chat turn and the request surrogate `("api", "api-<request_id>")` for an API call, or
- no terminal segment arrives for a long time (configurable “stale finalize”) — an **incomplete** archive is written so crash/cancel cases still leave a durable log trail. Each pass takes the oldest stranded bundles first and builds each archive from whatever segments exist at that moment, so a turn whose newest segment predates the delay is sealed **even if the turn is still running**; a segment it stages after that seal is picked up by the next assembly, which re-checks for a terminal segment first - one that lands among the rows the sealing pass loads is folded into that same archive by that pass, and one that lands later — while the pass is still writing that archive, or after it — is still picked up by a later pass, which is not guaranteed to come if the archive directory or passphrase stops resolving., unless the archive already records that turn as finished - a segment that lands after that read on a turn whose archive already records the turn as finished is folded in by that same pass, which writes the turn complete rather than sealing it again and also preserves the outcome that archive already recorded. The delay is what bounds that exposure, and it is why the default is long. The seal itself is conditional: if the log directory or passphrase no longer resolves, or the sealed write fails, the segments are kept for a retry and are removed by this subsystem's own retention, on `SESSION_LOG_RETENTION_DAYS`. The `ARTIFACT_CLEANUP_DAYS` sweep does not touch them: it is scoped to the pipe's artifacts, and shortening that window cannot delete log history. The guarantee survives — the segments are still not gated on a valve the operator would shorten to save disk — but its home has moved from the artifact sweep to the one that owns them. A later terminal segment for the same turn merges into that same zip and removes the finalized-incomplete line.

A bundle the assembler cannot write does not hold the window: the failed turn keeps its segments for retry, and is skipped by the next `SESSION_LOG_ASSEMBLER_BATCH_SIZE` window so one unwritable bundle delays only itself. `SESSION_LOG_LOCK_STALE_SECONDS` is the retry interval — the failed turn is dropped from both listings for that long, after which it is retried on the normal schedule and the pass after that seals it — so a transient failure recovers on its own, and nothing is discarded. One bounded carve-out: a turn waiting on the stranded-turn rescue is exempt from that backoff only while its own three-strike rescue budget lasts, and once that budget is spent it is set aside for this interval like any other bundle the pipe could not write. Each loss point is reported at WARNING naming the turn it holds up: a bundle whose sealed write fails names the turn and the archive path; a bundle whose archive settings do not resolve (a blank passphrase, a blank log directory, a missing `pyzipper`) is reported the same way and per turn, rather than once for the worker's lifetime — though the record of turns already named is a bounded window of the most recent 32 per latch, oldest evicted first, so a very old turn that faults again after 32 others have warned is named again; and a bundle whose settings resolved but whose write found `SESSION_LOG_STORE_ENABLED` switched off under it says the turn's segments stay staged for a later pass.

A pass also has a **wall-clock budget**: it takes no new turn once `SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS` of wall clock has elapsed since it started assembling, which is the same valve that spaces the passes, read fresh on every pass so a change applies with no restart. The clock starts when the pass reaches its candidates, after the stale-lock sweep and both listings, so a pass whose database scan outlasts the interval is no longer emptied by it: it always assembles the first turn it listed, and a whole pass can run longer than this number on a slow host. One turn's seal is an AES-encrypted zip write, so a full `2 × SESSION_LOG_ASSEMBLER_BATCH_SIZE` window can take several times the interval on a slow machine; without the budget the pass overruns every tick for as long as the backlog lasts. The check runs **before** each turn rather than after, so the last turn a pass takes cannot overrun, and a turn the budget did not reach is neither failed nor backed off — its segments stay staged and unmarked, and the next pass offers it again as if nothing happened. The trade is fairness between the two kinds of turn: finished turns are still listed before crash-stranded ones, so a large backlog of finished turns spends the budget first and the stranded ones wait. Raising `SESSION_LOG_ASSEMBLER_BATCH_SIZE` above what one interval can actually pack does not make a pass take longer; it makes a pass take more of the window and leave the rest to the next one.


The incomplete marker appears **at most once** per archive: a later pass that finds the turn complete retires the marker rather than adding a second, and a pass that finds it already present leaves the count at one. A **refused pass does not restart the finalize countdown**: the staleness clock advances only when a bundle is published, so a turn whose write failed is offered again rather than waiting a full window.

Non-blocking behavior:

- Archives are written asynchronously via a bounded internal queue.
- If the archive queue is full, the pipe drops the archive for that request (it does not block the response), and the notice is the rate-limited one below: a `WARNING` per five-minute cooldown window per worker, `DEBUG` on every later drop, every record carrying the running number of archives that worker has dropped.
- When the archive cannot be staged in the database, the pipe writes it through that same bounded queue instead of writing it inline, so the request never waits for compression. The path is therefore lossy in the same way: during a long database outage a busy queue fills, and the session logs for later requests in that period are dropped with that same rate-limited notice rather than queued indefinitely.
- On stop, a shutdown or an Open WebUI hot reload, the writer drains the archives it has already accepted within a bounded window (about one second) rather than abandoning the queue the moment the stop signal arrives, and reports at WARNING how many queued archives it could not write. A restart is therefore no longer the point at which an in-flight burst is lost, including on the route where the manager is collected during shutdown, where the same bounded drain runs and the count is reported through the module logger because there is no manager left to log through. A cleanup sweep that outlasts the bounded stop is repaired on the next start, per thread, and the restarted thread runs on a freshly armed stop event, so nothing accepted in that window is left unwritten. That repair is not conditional on the *other* thread having retired: when a write and a sweep are both still inside their operations when the stop returns, the next start still re-arms the event and still starts a fresh sweep, because a thread still alive on a set event is about to retire and so does not count as running. The writer is the exception and is deliberately not replaced while its thread is alive — its own drain window expires and it retires, and the archive it was holding is written by the writer a later start brings back (the thread slot is a contract, pinned by `tests/test_session_logs.py:1215`, so a second writer is a concurrency change and not a repair). The request queue is drained on the same contract: `close()` finishes its drain even when the worker and its jobs are bound to a closed loop, and each abandoned job's counter is released rather than left held; the setup-time drop of a request queue bound to a dead loop drains the abandoned one the same way.

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

That layout is exact for every identifier Open WebUI mints: a user id and a chat id are a uuid, and a message id is one the caller chooses. A component that needs escaping is still one directory and one file deep — only its name changes, as the last bullet under Path safety describes.

A call in either group has no usable `message_id`, so letting the empty string through would collide: sanitisation turns `""` into a fixed literal per slot and every such call from every user would land on one shared file. The user slot is guarded the same way, so a turn whose segments name no usable owner gets a directory derived from that empty id rather than a shared one. The pipe therefore substitutes the request-scoped surrogate pair `api` / `api-<request_id>` before staging, which is the same key the assembler gates on, so a surrogate archive is packed and written like any other:

```
<SESSION_LOG_DIR>/
  <user_id>/
    api/
      api-<request_id>.zip
      api-fusion-inner-<12 hex>.zip
```

The two sit in one directory rather than one each. An inner Fusion call mints its own request id, `fusion-inner-<12 hex>`, so its file is `api-fusion-inner-<12 hex>.zip`: a panel member, the judge and the synthesis are told apart from a plain API call by that prefix, not by where they are filed. An N-model panel turn therefore writes N+2 archives here, or N+3 when the judge runs a second pass, on top of the turn's own archives.

The task files sit beside the answer's in the same `<chat_id>/` directory and keep the `message_id` as their prefix, so `ls <chat_id>/` and `grep <message_id>` both still work for an id that needs no escaping. The message id is truncated from the right to fit a 64-character column, with the task name's space reserved first — the qualifier is never the part that gets cut. The pipe's own video-intent classifier is a task archive too, but Open WebUI withholds a `message_id` from it (`utils/middleware.py:3286` builds a live emitter only when `chat_id` **and** `message_id` are present, and the classifier has only the former), so it resolves through `user_message.childrenIds[0]` and its file is keyed on the **first** assistant reply under the user's message rather than the turn's own. On the plain API route there is no id at all and, like any other id-less task, it is skipped rather than filed under `api/`.

Each archive is written to a temporary file in the same directory, flushed to disk, and then published with an atomic rename. The temporary file's name carries the writer's process id and a random suffix, so concurrent writers to one turn never share it. The result is that when two writers race, each publishes a complete archive rather than interleaving into one: a published archive is always openable, even though a later publish replaces an earlier one's content rather than merging with it.

Before it creates anything, a write **reserves** the directory it is about to fill, and the cleanup sweep checks that reservation and removes the directory under the same lock, so a sweep can never take the directory out from under a write in progress. The guarantee is per process, and it is about the reservation being registered *before* the sweep's check — which is why the claim is taken before the directory is created rather than after. Once the write returns the reservation is dropped and a later sweep prunes the directory as usual.

**Scope:** task archives are written for the housekeeping tasks Open WebUI dispatches with a resolvable `message_id` and a `task` name. **A Fusion panel member is no task archive, but it is archived**: it carries no `message_id` and no `task`, so it takes the request surrogate instead and is written to `api/api-<request_id>.zip`, one file per inner call, on top of the turn's own archives.

Path safety:

- For filesystem safety, the `<user_id>`, `<chat_id>`, and `<message_id>` path components are **sanitized** (non-alphanumeric characters are replaced, and components are length-limited). Sanitisation is lossy, so it is not a one-to-one naming: `a/b` and `a:b` both reduce to `a_b`, and a 129-character id is cut to its first 128. A component that does not survive sanitisation unchanged therefore keeps the sanitized stem **and gains a short digest of the exact id** (`a_b-01J9XZ…`), which is what keeps two ids that would otherwise share a name in two files; an id that needs no escaping keeps its exact current name. The digest is taken over the raw value, not the sanitized one, which is why the two members of a lossy pair separate.
- The original (unsanitized) identifiers are preserved in `meta.json` under `ids.*` for correlation. A task archive's `ids.message_id` is the **bare** Open WebUI message id — the one that field can be joined against — with the task qualifier in the **filename** and in a separate `ids.task`, not appended to `ids.message_id`. An answer archive has no qualifier, so it carries no `ids.task` at all.
- The split tests the key's tail against a known list of task names, not against a shape. A name Open WebUI sends that the pipe has never seen leaves the qualifier **in** `ids.message_id` (`msg-1.brand_new_task`) and leaves `ids.task` absent. That is a loud, visible gap rather than a silent misreading: the composed key is what the filename already says, so the archive is still findable — but `ids.message_id` does not join against an Open WebUI message. `tests/test_session_logs.py::test_the_archive_key_split_takes_every_task_name_open_webui_sends` is the row that turns a new name into a failing test the day Open WebUI ships it; note that it reads `.external/`, which is gitignored, so it **skips on CI and on a fresh checkout** and guards only where the Open WebUI source is present (local and farm runs).

---

## Encryption and compression

Archives are written with `pyzipper` using AES encryption (`WZ_AES`).

- `SESSION_LOG_ZIP_PASSWORD` controls the password used to encrypt each archive.
  - Recommendation: set a long random passphrase and store it as an encrypted valve value (requires `WEBUI_SECRET_KEY` so Open WebUI can encrypt/decrypt the stored valve value).
- `SESSION_LOG_ZIP_COMPRESSION` selects the compression algorithm: `stored`, `deflated`, `bzip2`, or `lzma` (default `lzma`).
- `SESSION_LOG_ZIP_COMPRESSLEVEL` applies only to `deflated` and `bzip2`, and its range is per codec: `deflated` takes 0–9, where 0 stores the entry without compressing it, and `bzip2` takes 1–9 because bzip2 has no level 0 — a `0` entered under `bzip2` is read as `1`. It is ignored for `stored` and `lzma`.

Key rotation note:

- Changing `SESSION_LOG_ZIP_PASSWORD` affects only **future** archives. Previously written archives are not re-encrypted and require the prior password to decrypt.
- An archive sealed under a previous passphrase is left **byte-for-byte alone** and keeps that passphrase until retention expires it. **A message turn that is still receiving segments when you rotate can never be completed**: the existing archive cannot be read, so the pass refuses, and the turn's remaining segments are never merged into it. After three refused passes, while `Enable session log storage` is on, they are instead written to a separate `<message_id>.<request_id>.zip` that opens with the **new** passphrase, while `<message_id>.zip` still needs the old one; those segments are then removed from the database — if that separate write itself fails, the segments stay staged for a later pass, and are eventually reaped by `Archive retention period` — and the turn's own archive still does not contain them. That separate file ages like any other archive and is reaped by `Archive retention period`. It carries the **same** `ids.message_id` and `ids.task` as the turn's own archive — the bare message id and the task qualifier, never the composite — so one correlation query covers both files; the `<request_id>` in its name is the terminal segment's request id where the turn staged one, and the first staged request id where it did not. **Rotate between turns, not during one.** To capture a turn's segments before rotating, copy them out of the database.

---

## Retention and cleanup

When storage is enabled, a background cleanup loop periodically:

1. Deletes `*.zip` files older than `SESSION_LOG_RETENTION_DAYS` (based on file modification time).
2. Reaps staged `session_log_segment` / `session_log_segment_terminal` rows in the artifact table older than `SESSION_LOG_RETENTION_DAYS` (from `created_at`), 500 at a time, on the assembler pass that already reaps stale locks. This is the only window that bounds them: the artifact sweep is scoped away from the pipe's bookkeeping rows, so without this leg a segment whose archive can never be written would sit in the table — with its full request and response content — indefinitely.
3. Removes empty directories left behind (including the base directory if it becomes empty), except a directory a write in this process is currently filling; that one is left alone. A write that fails leaves its directory to a later sweep, while a process that dies mid-write leaves a `*.zip.tmp` that no pass reaps, because the prune leg only removes directories that are empty.

Cleanup runs every `SESSION_LOG_CLEANUP_INTERVAL_SECONDS`. Turning `SESSION_LOG_STORE_ENABLED` off stops both sweeps entirely: no archive is deleted, no directory is pruned and no staged segment is reaped, and every archive already on disk is left exactly where it is until you re-enable storage and the retention window passes again.

With storage disabled the pipe neither writes nor deletes archives: turning the valve off stops the sweep on the next pass, and archives already on disk are left untouched until it is on again. A write already inside an assembly pass is read again at the write itself, so a pass that is under way when the valve goes off publishes nothing, keeps its staged segments, restores the turn's staleness stamps and releases its lock, and says the turn is still staged rather than that it was saved. The sweep reads `SESSION_LOG_RETENTION_DAYS` on every pass, so a changed window applies from the next sweep with no restart and no new turn.

Additionally, once an archive is assembled, the per-invocation DB segments used to build it are deleted. A separate “stale finalize” path can assemble + delete segments for abandoned turns after a long timeout; it is a cutoff on each turn’s last segment rather than on the turn itself, so a turn still running when it passes is sealed, and its later segment is picked up by the next assembly, which re-checks for a terminal segment first; one that lands among the rows the sealing pass loads is folded into that same archive by that pass, and one that lands later — while the pass is still writing that archive, or after it — is still picked up by a later pass, which is not guaranteed to come if the archive directory or passphrase stops resolving., unless the archive already records that turn as finished - a segment that lands after that read on a turn whose archive already records the turn as finished is folded in by that same pass, which writes the turn complete rather than sealing it again.

Operational note:

- Cleanup is performed per-process and tracks the session log directories that have been used in that process. In multi-worker deployments, each worker performs its own cleanup pass.
- A write reserves the directory it is filling, and the sweep honours that reservation, so a sweep does not remove a directory a write in the same process is in the middle of. The reservation is **per process**: another Open WebUI worker on the same host runs its own sweep with its own set of reservations, so a directory one worker is filling can still be pruned by a peer. Within a process the check and the removal are under one lock, so a write that reserves a directory before the sweep's check is never removed by that pass.

---

## Relevant valves

See [Valves & Configuration Atlas](valves_and_configuration_atlas.md) for the canonical list and defaults. Session log storage is controlled by:

| Valve | Type | Default (verified) | Purpose |
|---|---:|---:|---|
| `SESSION_LOG_STORE_ENABLED` | bool | `false` | Enables writing encrypted session log archives to disk. A write already inside an assembly pass is read again at the write itself: one already under way when this is switched off publishes nothing, keeps its staged segments, restores the turn's staleness stamps and releases its lock. |
| `SESSION_LOG_ARCHIVE_API_CALLS` | bool | `true` | Archives a request with no usable `chat_id`/`message_id` under `api/api-<request_id>.zip`. A staging gate only — a segment staged while it was on is still written after it is switched off. That is this valve alone: `SESSION_LOG_STORE_ENABLED` is read again at the write, so a pass already under way when it is switched off publishes nothing and keeps its rows. Governs the plain API route and the `parent_id: null` shape only: a task invocation that resolves to no message id is skipped whether it is on or off. |
| `SESSION_LOG_DIR` | str | `session_logs` | Base directory for archives. Surrounding whitespace is ignored, and a value that is blank once trimmed counts as unset. |
| `SESSION_LOG_ZIP_PASSWORD` | encrypted str | *(empty)* | Password used to encrypt archives (required to store). |
| `SESSION_LOG_RETENTION_DAYS` | int | `90` | Retention window for stored archives, and for the staged segment rows they are assembled from — the same window, measured from file modification time for the archives and from `created_at` for the rows. |
| `SESSION_LOG_CLEANUP_INTERVAL_SECONDS` | int | `3600` | Cleanup loop interval. |
| `SESSION_LOG_ZIP_COMPRESSION` | enum | `lzma` | Zip compression algorithm. |
| `SESSION_LOG_ZIP_COMPRESSLEVEL` | int? | `null` | Compression level for deflated (0–9, where 0 stores without compressing) and bzip2 (1–9; a `0` there is read as `1`). Ignored for stored/lzma. |
| `SESSION_LOG_MAX_LINES` | int | `20000` | Max in-memory log records retained per request before older entries are dropped. |
| `SESSION_LOG_FORMAT` | enum | `jsonl` | Archive log file format. `logs.jsonl` is always written as the canonical record; `jsonl` writes only `logs.jsonl`, while `text` and `both` additionally write `logs.txt` (so `text` and `both` produce an identical file set). `logs.txt` writes one record per physical line, except for a record's exception block. |
| `SESSION_LOG_ASSEMBLER_INTERVAL_SECONDS` | int | `30` | How often each process scans the DB for completed/stale turns to assemble into zip archives, and the wall-clock budget one pass gets to take new turns — read fresh on every pass, so a change applies with no restart. |
| `SESSION_LOG_ASSEMBLER_JITTER_SECONDS` | int | `10` | Per-process jitter added to the assembler loop to avoid multi-worker lockstep. |
| `SESSION_LOG_ASSEMBLER_BATCH_SIZE` | int | `25` | Max turns **listed** per assembler tick in each of the two listings — a cap on turns, not rows, so a heavily split turn still takes one slot. A tick lists up to twice this many and then takes as many as its wall-clock budget allows, leaving the rest to the next tick. |
| `SESSION_LOG_STALE_FINALIZE_SECONDS` | int | `43200` | If no terminal segment arrives for a turn, assemble an **incomplete** archive after this timeout. A cutoff on the last segment, not on the turn: a turn **still running** when it passes is sealed as incomplete too, and a segment it stages afterwards is left stranded until the next assembly, unless it lands before the sealing pass has read that turn's segments - a pass that finds a terminal segment among the rows it loaded writes the turn complete instead; one that lands after that read is not folded into that archive by that pass and is picked up by a later one, unless the archive already records that turn as finished - a segment that lands after that read on a turn whose archive already records the turn as finished is folded in by that same pass, which writes the turn complete rather than sealing it again and also preserves the outcome that archive already recorded. That exposure is why the default is long. Each pass takes the oldest stranded bundles first and seals a bundle only if the sealed write succeeds, keeping the segments for a retry otherwise. **Warning:** The minimum is `300` seconds (five minutes): a lower value is refused when the configuration is saved, and a stored one that no longer validates falls back to the default rather than being raised to it. |
| `SESSION_LOG_LOCK_STALE_SECONDS` | int | `1800` | DB lock row stale timeout (multi-worker safety), and the write-failure backoff: a bundle whose archive could not be written is skipped for this long before it is retried. A lock held by a peer is not a write failure, so a turn another worker is already archiving is never backed off at all. One bounded carve-out: a turn waiting on the stranded-turn rescue is exempt from that backoff only while its own three-strike rescue budget lasts, and once that budget is spent it is set aside for this interval like any other bundle the pipe could not write. |
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
- `label` (string): What was measured, never what was said. A label names a function or scope (for example `streaming.streaming_core.StreamingHandler._run_streaming_loop`), a scope such as `tool_run:<tool_name>:start`, a worker, or a transport read by its number and byte length (`chunk_1_len_4096`). It never carries prompt or response content: a timing record is a measurement, and a slice of the provider's body is not one.
- `request_id` (string): The pipe's per-request correlation id. Present on every event that belongs to a request; the pipe's process-lifetime background workers — the request dispatcher, the log worker and the artifact-cleanup sweep — are started on an empty context and are deliberately not attributed to any request, so their frames do not appear. So is a video generation job's lifecycle task, which outlives the request that submitted it, and the web-tools filter repair, which is scheduled from inside a request; both start on an empty context so neither can put a finished request's id back into the log, and neither is silenced by it — with no request in scope they answer to the process log level. The one-shot startup warmup does carry the id of the request that started it.
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

Alongside the file the logger keeps an in-memory copy of each request's events, so a running request can be read without parsing the file. That copy is a duplicate of a durable record and is released as soon as the request's job completes — the file is what persists, and only the file. A nested request (a Fusion panel member, judge, judge-repair or synthesis call) keeps its own copy for the length of *that* call and releases it when that call returns, so one turn can hold several at once. The timestamp index that says which requests are still in flight is released with that copy, and `SessionLogger.cleanup()`, which drops both maps for anything older than its one-hour cutoff, is the backstop for the index on a request that ends abnormally. `MAX_TIMING_REQUESTS` is the backstop for requests that end abnormally and never reach the release, not the normal retention path; at that bound the entry removed is the one whose last recorded event is oldest, so a request that is still recording is never the victim.

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
- Near-zero overhead when disabled (the context variable check short-circuits). The one cost that survives it is the label string the `/responses` producer builds for each transport read before the mark is offered to the recorder; on a 50 KB reply that is about 10 µs in total.
- Events are written immediately to `TIMING_LOG_FILE` (not session archives).
- Each event includes a `request_id` field for correlation with session logs.
- Maximum 10,000 events per request in memory (oldest dropped if exceeded). That cap is a backstop, not a budget: a call made once per event or once per transport read used to fill it from within the stream and evict the request's own pipeline marks, and the flood that did so was removed rather than accommodated.
- A generator abandoned mid-stream — built, then never iterated to exhaustion or closed — leaves an `enter` with no `exit`. That is ordinary span semantics, not a lost record: the `enter` says the function was called, and nothing says how long it ran.

---

## Operational guidance

- Restrict access to `SESSION_LOG_DIR` (filesystem permissions, encrypted volume, backups with access controls).
- Treat archives as sensitive: they are encrypted at rest, but anyone with the zip password can decrypt them.
- If you enable request identifiers for provider-side attribution, keep the same identifiers available for operators: see [Request identifiers and abuse attribution](request_identifiers_and_abuse_attribution.md).
