# Concurrency Controls & Resilience

This document describes the pipe’s admission control, concurrency limits, breakers, and background workers that protect Open WebUI from overload and cascading failures.

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [Errors](error_handling_and_user_experience.md) · [Persistence](persistence_encryption_and_storage.md)

---

## Admission control (requests)

The pipe applies multiple layers of admission control per process:

- **Request queue:** requests are wrapped into jobs and enqueued into an in-process asyncio queue (`maxsize=1000`). If the queue is full, the request is rejected with a user-facing “Server busy (503)” message.
- **Global request semaphore:** `MAX_CONCURRENT_REQUESTS` limits in-flight requests per process. The semaphore is re-made for the running event loop, so a process whose event loop is replaced (a test runner that calls `asyncio.run` twice, a supervisor that restarts the loop) starts again at the configured value rather than serving a semaphore bound to a dead loop. Increasing the valve can take effect immediately; decreasing it requires a restart to fully reduce concurrency, except across a loop swap, where the semaphore is rebuilt at the configured value in both directions. A request holds its slot from before its own tool workers start until after they have all stopped, on the normal path and on the raise path alike. The trade is admission latency: the next request's wait rises by up to `TOOL_SHUTDOWN_TIMEOUT_SECONDS` per tool-bearing request, and a still-running tool holds a `MAX_PARALLEL_TOOLS_GLOBAL` permit across that same wait (taken when the call starts running, released only when it returns; Open WebUI's `ask_user` excepted, which takes no tool slot), so at a low `MAX_PARALLEL_TOOLS_GLOBAL` a slow drain can occupy a request slot and a tool slot at once. That is a sizing note rather than a defect — both pools default to 200.
- **Startup warmup gate:** the pipe runs background warmup checks once an API key is configured. If warmup has failed, requests are rejected with “Service unavailable due to startup issues” until a subsequent warmup succeeds.

---

## HTTP client pool (one per event loop)

Outbound requests share a single `aiohttp.ClientSession` per event loop rather than building one per request, so a reply no longer repeats DNS and a TLS handshake and a streamed reply reuses its connection across its own round trips. The session is closed when the pipe shuts down, and a session left behind by a dead event loop is retired rather than reused. It is pooled per event loop because a connector is bound to the loop that created it.

The pool is not a second admission control. Its per-host limit is unlimited and its total connection limit follows the concurrency limit actually in force — `MAX_CONCURRENT_REQUESTS` as adjusted at startup — so it never queues work the semaphore has already admitted. Because that limit is fixed when the connector is built, lowering `MAX_CONCURRENT_REQUESTS` at runtime takes effect only after a restart, exactly as it does for the semaphore itself.

Timeout valves (`HTTP_CONNECT_TIMEOUT_SECONDS`, `HTTP_TOTAL_TIMEOUT_SECONDS`, `HTTP_SOCK_READ_SECONDS`) are applied **per call** at every site that issues an outbound request, so a valve changed at runtime governs the next request. The pooled session also carries the values from the request that built it; that default applies only to a site that supplies no `timeout=` of its own.

---

## Tool concurrency (per request and global)

Tool execution is constrained to prevent a single request (or a single user) from consuming all compute:

- **Global tool semaphore:** `MAX_PARALLEL_TOOLS_GLOBAL` caps the total number of tool executions across all requests in the process. Like the request semaphore, it is re-made when the process's event loop changes, so the limit is re-established for the current loop rather than surviving a loop swap.
- **Per-request tool semaphore:** `MAX_PARALLEL_TOOLS_PER_REQUEST` caps tool parallelism within a single request.
- **`ask_user` exemption:** Open WebUI's built-in `ask_user` waits on a person, so it takes no slot from either tool semaphore.
- **Batching and timeouts:** tool loop ceilings and timeouts are controlled by `MAX_FUNCTION_CALL_LOOPS`, `TOOL_BATCH_CAP`, `TOOL_TIMEOUT_SECONDS`, `TOOL_BATCH_TIMEOUT_SECONDS`, and `TOOL_IDLE_TIMEOUT_SECONDS`.

See [Tooling & Integrations](tooling_and_integrations.md) for tool-specific behavior and schema handling.

---

## Breakers (fast-fail protection)

Breakers stop repeated failures from turning into continuous retries and log storms. The three breakers below are per user; each internal Fusion run also keeps one count per tool, shared by all of its models, which uses `BREAKER_MAX_FAILURES` but ignores the window. The per-user breakers are governed by:

- `BREAKER_MAX_FAILURES`
- `BREAKER_WINDOW_SECONDS`

Breaker scopes include:

- **Per-user request breaker:** refuses a user's new requests once `BREAKER_MAX_FAILURES` failed calls to OpenRouter fall within the breaker window. For chat calls, the failures that count are error replies; connections that cannot be opened, drop or time out; errors OpenRouter reports after accepting a call; and streams that stop before their final event. A generation on a picture-only image model or a video model counts once, when it fails after being sent to OpenRouter. Failed panel, judge and final-answer calls made by internal Fusion count too. A request the pipe retried counts once for the whole request, however many attempts it took, and a request stopped during one of its waits counts none. A request that ends without an error clears the count, but a request to a picture-only image model or a video model clears it only once its result is delivered, and an internal Fusion run clears it only if a panel model answered; a request the user stops does not clear it. Housekeeping tasks such as title generation neither count nor clear; Open WebUI's merge-responses task counts but never clears. This breaker never refuses a request whose last message is a tool result, or is Open WebUI's own message that comes right after a tool result and hands the model a tool's images: such a request is how Open WebUI finishes an answer already under way. A question or picture the user sends is refused like any other request.
- **Per-user persistence breaker:** skips a user's database reads and writes after repeated DB failures within the breaker window; requests continue with reduced durability. A successful read or write clears the count; where Redis buffers writes, a write succeeds once Redis has taken it. Nothing is attempted while the breaker is open, so only failures ageing out reopen it.
- **Per-user, per-tool breaker:** skips a specific tool, keyed by tool type and tool name, after it fails `BREAKER_MAX_FAILURES` times in a row; other tools keep working. The failures that count are an error the tool raises (unless its MCP session has closed), a per-call timeout other than an `ask_user` one, a running call cancelled by the batch deadline, and a call whose tool server cannot be reached or answers with an HTTP error status. When the tool reports a failure in a result it returns normally, the call is shown as failed but neither counts nor clears the count. A successful call clears the count, and so does a gap longer than the breaker window between the tool's last failure and its next call, which is why a slow tool that keeps timing out still trips. Each internal Fusion run keeps one count per tool, shared by all of its models: once a tool fails `BREAKER_MAX_FAILURES` times in a row during the run, it is skipped from then on, even after a quiet spell, unless a call to it that was already running succeeds. The user's own count is left unaffected.

Breakers also recover on their own: the request and persistence breakers once their failures age out of the window, and a user's tool breaker once a call to the tool comes more than a full window after its last failure. A turn whose tool calls are all skipped by a tool breaker does not count as a failed request, so a broken tool cannot lock a user out of the pipe.

---

## Background workers (what runs in the process)

The pipe uses background tasks/threads to keep request handling responsive:

- **Request worker loop:** dequeues jobs from the request queue and spawns per-request tasks. The loop also gates those spawns: it takes a `MAX_CONCURRENT_REQUESTS` permit immediately before creating a task and the job keeps that slot for its whole life, so a queued request waits for a permit instead of being spawned and running anyway.
- **Async log worker:** drains a bounded async log queue (`maxsize=1000`) used by `SessionLogger` so log emission does not block request execution.
- **Redis tasks (optional):** when Redis caching is enabled and available, the pipe starts a Redis client and associated listener/flush tasks.
- **Artifact cleanup worker:** periodically deletes persisted artifacts older than `ARTIFACT_CLEANUP_DAYS` (as measured from `created_at`, which is refreshed on every read, database or cache) on an interval controlled by `ARTIFACT_CLEANUP_INTERVAL_HOURS`.
- **Session log storage threads (optional):** when session log storage is enabled, the pipe uses background threads to write, assemble and clean up encrypted zip archives. All three are per-Pipe-instance and stop on every teardown path: an orderly `stop_workers()` signals them and joins, and a collection without one ends each of them within one wake slice (the writer on its next queue timeout, the cleanup and assembler threads on their next 30 s slice).

**State ownership:**
- **Instance-level**: request queue, log queue, worker tasks, and locks are owned by each Pipe instance. A request is counted in the instance's in-flight number from when `pipe()` is entered until its job's cleanup tail has run to completion, not merely until its bytes were delivered, so a superseded generation is closed only once its last request has finished tidying up rather than while that request is still shutting down.
- **Per-instance plugin registrations**: when the plugin system is enabled, the pipe dashboard registers its pipe getter and live-snapshot getter into three module-level globals (the socket handler, the HTTP routes and the publisher) so those request-serving paths can reach the current pipe, and the action route caches a fourth `Pipe` of its own on first use, reconciling against the worker's function-module cache. The plugin's `on_shutdown` clears all four, but only where the global still refers to that plugin instance — a newer pipe that has already re-registered them (hot reload, in either order) is left registered, and the action route's cached pipe is dropped only when it is this pipe by identity, so a newer pipe's live cache survives another pipe's teardown. The same helper runs on every `on_models`, so a pipe that is still live puts the three registrations back after a newer pipe's teardown dropped them; the clear is therefore not a one-way door for a surviving instance, only a way of not stomping on a newer registration.
- **Class-level**: rate-limiting semaphores are shared across all instances in the same process (per-process concurrency control), and are re-made when the process's event loop changes. A job already past the semaphore's `acquire()` holds a permit on the old object and releases it into the orphan when it finishes; the release is harmless, and no live slot is taken from the new semaphore by it.

---

## Operational tuning (practical guidance)

- If you see “Server busy (503)” frequently, that is a real diagnostic and the first time it has been one. It means the pipe refused a request outright, which is a capacity problem and not a stall: the requests ahead of it are holding every concurrency permit and the bounded request queue behind them is full. Lower upstream load, raise `MAX_CONCURRENT_REQUESTS` toward the CPU/RAM headroom, or raise `_QUEUE_MAXSIZE` if the requests are merely queued rather than slow. The dashboard’s `requests` / `requests_max` fields give the queue depth that caused the refusal.
- For rate limits and upstream instability, use breakers plus conservative retry windows; avoid “infinite retries”. The chat retry window is bounded by `TRANSIENT_RETRY_MAX_ATTEMPTS` tries and `TRANSIENT_RETRY_MAX_WAIT_SECONDS` per wait — a `Retry-After` is truncated to that cap rather than replaced by it — and a request the user stops mid-wait adds no strike against the breaker.
- For multi-worker deployments, consider enabling Redis caching (when appropriate) to improve artifact replay performance and reduce DB contention.

All tunables referenced here are documented (with verified defaults) in [Valves & Configuration Atlas](valves_and_configuration_atlas.md).
