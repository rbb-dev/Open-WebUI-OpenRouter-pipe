# Concurrency Controls & Resilience

This document describes the pipe’s admission control, concurrency limits, breakers, and background workers that protect Open WebUI from overload and cascading failures.

> **Quick navigation:** [Docs Home](README.md) · [Valves](valves_and_configuration_atlas.md) · [Errors](error_handling_and_user_experience.md) · [Persistence](persistence_encryption_and_storage.md)

---

## Admission control (requests)

The pipe applies multiple layers of admission control per process:

- **Request queue:** requests are wrapped into jobs and enqueued into an in-process asyncio queue (`maxsize=500`). If the queue is full, the request is rejected with a user-facing “Server busy (503)” message.
- **Global request semaphore:** `MAX_CONCURRENT_REQUESTS` limits in-flight requests per process. Increasing the valve can take effect immediately; decreasing it requires a restart to fully reduce concurrency.
- **Startup warmup gate:** the pipe runs background warmup checks once an API key is configured. If warmup has failed, requests are rejected with “Service unavailable due to startup issues” until a subsequent warmup succeeds.

---

## Tool concurrency (per request and global)

Tool execution is constrained to prevent a single request (or a single user) from consuming all compute:

- **Global tool semaphore:** `MAX_PARALLEL_TOOLS_GLOBAL` caps the total number of tool executions across all requests in the process.
- **Per-request tool semaphore:** `MAX_PARALLEL_TOOLS_PER_REQUEST` caps tool parallelism within a single request.
- **Batching and timeouts:** tool loop ceilings and timeouts are controlled by `MAX_FUNCTION_CALL_LOOPS`, `TOOL_BATCH_CAP`, `TOOL_TIMEOUT_SECONDS`, `TOOL_BATCH_TIMEOUT_SECONDS`, and `TOOL_IDLE_TIMEOUT_SECONDS`.

See [Tooling & Integrations](tooling_and_integrations.md) for tool-specific behavior and schema handling.

---

## Breakers (fast-fail protection)

Breakers prevent repeated failures from cascading into continuous retries and log storms. They are governed by:

- `BREAKER_MAX_FAILURES`
- `BREAKER_WINDOW_SECONDS`
- `BREAKER_HISTORY_SIZE`

Breaker scopes include:

- **Per-user request breaker:** blocks new requests for a user when repeated failures occur within the breaker window.
- **Per-user persistence breaker:** skips database persistence work for a user when repeated DB failures occur (requests can continue with reduced durability).
- **Per-user/tool-type breaker:** skips tool execution for specific tool types when repeated tool failures occur.

Breakers are self-healing: once failures age out of the window (or a successful operation occurs where applicable), normal operation resumes without operator intervention.

---

## Background workers (what runs in the process)

The pipe uses background tasks/threads to keep request handling responsive:

- **Request worker loop:** dequeues jobs from the request queue and spawns per-request tasks.
- **Async log worker:** drains a bounded async log queue (`maxsize=1000`) used by `SessionLogger` so log emission does not block request execution.
- **Redis tasks (optional):** when Redis caching is enabled and available, the pipe starts a Redis client and associated listener/flush tasks.
- **Artifact cleanup worker:** periodically deletes persisted artifacts older than `ARTIFACT_CLEANUP_DAYS` (as measured from `created_at`, which is refreshed on DB reads) on an interval controlled by `ARTIFACT_CLEANUP_INTERVAL_HOURS`.
- **Session log storage threads (optional):** when session log storage is enabled, the pipe uses background threads to write and clean up encrypted zip archives.

**State ownership:**
- **Instance-level**: request queue, log queue, worker tasks, and locks are owned by each Pipe instance.
- **Class-level**: rate-limiting semaphores are shared across all instances in the same process (per-process concurrency control).

---

## Operational tuning (practical guidance)

- If you see “Server busy (503)” frequently, lower upstream load or increase capacity; then tune `MAX_CONCURRENT_REQUESTS` and tool parallelism valves based on CPU/RAM headroom.
- For rate limits and upstream instability, use breakers plus conservative retry windows; avoid “infinite retries”.
- For multi-worker deployments, consider enabling Redis caching (when appropriate) to improve artifact replay performance and reduce DB contention.

All tunables referenced here are documented (with verified defaults) in [Valves & Configuration Atlas](valves_and_configuration_atlas.md).
