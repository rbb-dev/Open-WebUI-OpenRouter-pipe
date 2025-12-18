# concurrency, guardrails, and resilience

**file:** `docs/concurrency_controls_and_resilience.md`
**related source:** `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py:2880-3600`, `tool worker` + `breaker` sections, `_maybe_start_*` helpers

OpenRouter workloads are bursty. This pipe is engineered to accept hundreds of concurrent chats without starving the host or corrupting state. This document walks through every concurrency primitive, admission policy, breaker, and watchdog that keeps the manifold stable.

> **Quick Navigation**: [📑 Index](documentation_index.md) | [🏗️ Architecture](developer_guide_and_architecture.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🔒 Security](security_and_encryption.md)

---

## 1. admission control

| Component | Location | Behavior |
| --- | --- | --- |
| Queue | `Pipe._request_queue` (max 500) | Every incoming request becomes a `_PipeJob` and is placed on the queue. If the queue is full, the pipe returns HTTP 503 immediately with a user-facing status ("Server busy"). |
| Worker task | `_request_worker_loop` | Pulls jobs from the queue, spawns async tasks, and wires cancellation: if Open WebUI cancels the future, the in-flight task receives `cancel()` and cleans up gracefully. |
| Global semaphore | `_global_semaphore` | Limit determined by `MAX_CONCURRENT_REQUESTS`. Lowering the valve at runtime logs a warning (requires restart) while raising it releases additional permits immediately. |
| Startup checks | `_maybe_start_startup_checks` | Defers actual work until an API key exists and a warmup probe succeeds. While `_warmup_failed` is true, every request is rejected with "Service unavailable due to startup issues" to avoid partial execution. |

---

## 2. tool concurrency

* **Global limit** -- `_tool_global_semaphore` enforces `MAX_PARALLEL_TOOLS_GLOBAL` so all chats combined cannot overwhelm CPU-bound tool workers.
* **Per-request limit** -- `_ToolExecutionContext` caps workers at `MAX_PARALLEL_TOOLS_PER_REQUEST`. Combined with queue batching, this stops a single chat from locking every worker.
* **Batch cap** -- `TOOL_BATCH_CAP` sets how many identical tool calls run in parallel when the model emits multiple calls with the same arguments.
* **Idle timeout** -- `TOOL_IDLE_TIMEOUT_SECONDS` (optional) cancels worker tasks when no tools run for the specified duration, preventing leaks.

---

## 3. background workers

| Worker | Trigger | Notes |
| --- | --- | --- |
| Log worker (`_log_worker_loop`) | Started lazily when the first request runs. Processes `SessionLogger` records asynchronously so slow sinks can"t block request handlers. |
| Redis listener / flusher | Spawned when `_redis_candidate` is true. Listens for `db-flush` pub/sub messages and also runs a periodic timer. Uses a short-lived lock to ensure only one worker flushes at a time. |
| Artifact cleanup worker | Created when persistence initializes. Sleeps for `ARTIFACT_CLEANUP_INTERVAL_HOURS` + jitter, deletes rows older than `ARTIFACT_CLEANUP_DAYS`. |
| Startup warmup task | Only runs once per process. Warms TLS/DNS caches by hitting `/models?limit=1`. Logs warnings instead of crashing when the API key is missing; failures flip `_warmup_failed` until the next successful probe. |

All workers are cancelled in `_stop_request_worker`, `_stop_log_worker`, `_shutdown_tool_context`, and `_run_cleanup_once` when the pipe is reloaded.

---

## 4. breaker summary

See `docs/tooling_and_integrations.md` for tool-specific breakers. This section covers the shared ones:

| Breaker | Scope | Threshold | Action |
| --- | --- | --- | --- |
| `_breaker_records` | per-user request-level | `BREAKER_MAX_FAILURES` hits within `BREAKER_WINDOW_SECONDS` (defaults: 5 / 60s) | Stops admitting new jobs for that user until the window clears. Emits a `chat:status` warning. |
| `_db_breakers` | per-user persistence | Same valves (`BREAKER_MAX_FAILURES` / `BREAKER_WINDOW_SECONDS`) scoped to persistence failures | Skips DB work for the affected user (falls back to ephemeral mode) until the breaker heals. |
| Redis pending queue warning | global | queue depth > `REDIS_PENDING_WARN_THRESHOLD` | Emits WARN logs so operators know flushers are behind. |

Breakers automatically drain; no manual reset is required.

---

## 5. resilience tactics

* **Retry-aware HTTP client** -- `_create_http_session` configures `httpx` with retries and `_RetryableHTTPStatusError` so throttling/backoff signals from OpenRouter are honored.
* **ContextVars** -- `_TOOL_CONTEXT` and `SessionLogger` store per-request metadata thread-safely even when work jumps between worker queues and thread pools.
* **Graceful degradation** -- If Redis or the DB fails, `_persist_artifacts` logs, surfaces a user warning, and falls back to the next-best storage path rather than crashing the entire request.
* **Size guards everywhere** -- From `_download_remote_url` through streaming queues, every pipeline has hard caps with descriptive valve names so operators can tune behavior without code changes.

---

## 6. tuning checklist

1. **High traffic** -- Raise `MAX_CONCURRENT_REQUESTS` only if CPU/RAM allow it and the upstream OpenRouter rate limit is high enough. Monitor 503 counts.
2. **Tool-heavy workloads** -- Increase `MAX_PARALLEL_TOOLS_GLOBAL` + `MAX_PARALLEL_TOOLS_PER_REQUEST` together, but keep breakers enabled so runaway tools still trip quickly.
3. **Slow networks** -- Increase `HTTP_CONNECT_TIMEOUT_SECONDS`/`HTTP_SOCK_READ_SECONDS` rather than disabling timeouts. Streaming reliability benefits more from steady limits than from unbounded waits.
4. **Verbose debugging** -- Set global `LOG_LEVEL=DEBUG` (or override per user) only temporarily; the async log queue is bounded to 1000 records.

Properly tuned valves, combined with the guardrails above, let this pipe sustain hundreds of concurrent reasoning + tool-heavy chats without starving the server. When you change any guardrail, update this doc and the valve reference so operators can keep the system stable.
