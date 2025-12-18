# developer overview & architecture map

**file:** `docs/developer_guide_and_architecture.md`
**related source:** `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py`, `pytest_bootstrap.py`, `tests/`

This document explains how the OpenRouter Responses manifold is organized, how requests flow through it, and how to contribute safely. Everything lives in a single Python module for compatibility with Open WebUI pipes, but the code is intentionally segmented into layers that mirror a multi-file project. Use the map below to orient yourself before diving into subsystem deep dives.

> **Quick Navigation**: [📑 Index](documentation_index.md) | [🏗️ Architecture](developer_guide_and_architecture.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🔒 Security](security_and_encryption.md)

---

## 1. layered architecture inside a single file

The pipe is delivered as `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py`. Internally it is split into clearly labeled sections:

| Section label | Responsibility | Key types/functions |
| --- | --- | --- |
| Imports & constants (top ~400 lines) | Standard/3rd-party/Open WebUI imports plus global regexes, limits, retry presets. | `_STREAMING_PRESETS`, `_REMOTE_FILE_MAX_SIZE_DEFAULT_MB`, `_MARKDOWN_IMAGE_RE`, `_RetryableHTTPStatusError`, `_RetryWait` |
| TypedDicts & dataclasses | Shapes for events, tool calls, and persisted payloads. | `StatusMessages`, `FunctionCall`, `ToolCall`, `Message`, `UsageStats`, `ArtifactPayload`, `_PipeJob`, `_QueuedToolCall`, `_ToolExecutionContext` |
| Security & helpers | Encryption wrapper, catalog helpers, ULID utilities, HTTP guards, chunked encoders, SSRF filters. | `EncryptedStr`, `ModelFamily`, `OpenRouterModelRegistry`, `_is_safe_url`, `_validate_base64_size`, `_inline_internal_file_url`, etc. |
| Request body models | Pydantic models translating Open WebUI "Completions" payloads to the Responses API schema. | `CompletionsBody`, `ResponsesBody` |
| Main controller | `Pipe` class: valves, queue/semaphore init, multimodal transforms, tool orchestrators, SSE streaming, persistence, Redis integration. | Everything from `class Pipe` through tool runners, streaming engine, persistence layer. |
| Session logging | Async logging queue & context propagation for per-request log capture. | `SessionLogger` |

Even though this is one Python file, treat each region as its own module. When you edit a subsection, update the matching doc listed in `docs/documentation_index.md`.

---

## 2. high-level request lifecycle

1. **Admission & warmup** (`Pipe.__call__`, `_ensure_concurrency_controls`, `_maybe_start_startup_checks`). A request arriving from Open WebUI enters a bounded queue (`_QUEUE_MAXSIZE = 500`) guarded by `MAX_CONCURRENT_REQUESTS`. If the API key is missing or the warmup probe has failed, the request short-circuits with a status message.
2. **Valve resolution** (`_merge_user_valves`, `_resolve_valves`). System valves (`Pipe.Valves`) provide defaults; per-user overrides (`Pipe.UserValves`) are merged after handling the special `
"inherit" literal so optional fields fall back cleanly. Valve values drive everything: timeouts, retries, image caps, Redis TTLs.
3. **History & multimodal transforms** (`_transform_messages_to_input`, `_to_input_*`). Messages from Open WebUI are normalized into Responses blocks. This includes decoding persisted markers, encoding images/files/audio/video, and replaying earlier tool outputs based on the retention valves.
4. **Model routing** (`OpenRouterModelRegistry.ensure_loaded`, `_pick_model`, `_inject_model_capabilities`). The registry fetches `/models`, caches the payload for `MODEL_CATALOG_REFRESH_SECONDS`, and exposes helper predicates (vision, reasoning, plugins). The chosen model controls downstream switches (e.g., drop image blocks when `vision` is false).
5. **Tool orchestration** (`_get_tool_context`, `_tool_worker_loop`, `_execute_tool_batch`). Function calls emitted by the model are executed via per-request queues bounded by both global and per-request semaphores. Outputs are persisted (optionally encrypted) and transformed back into Responses `tool_output` blocks before looping.
6. **Streaming engine** (`_stream_responses`, `_consume_sse`, `_emit_update`, `_emit_completion`). SSE chunks flow through bounded queues, multiple workers parse them, and emitters send chat deltas, reasoning summaries, citations, tool notifications, and final cost/usage stats.
7. **Persistence & cleanup** (`_ensure_db_initialized`, `_persist_artifacts`, `_redis_enqueue_rows`, `_artifact_cleanup_worker`). Reasoning/tool artifacts are written to scoped SQL tables, cached in Redis (when enabled), and periodically pruned based on retention valves.
8. **Shutdown paths** (`_stop_request_worker`, `_stop_log_worker`, `_shutdown_tool_context`). Cleanup runs when Open WebUI reloads the pipe or pytest tears it down, ensuring background tasks exit without hanging the worker.

Every one of those steps has a companion doc in `docs/` for deeper dives.

---

## 3. dev loop & bootstrap

### 3.1 repo layout

```
open_webui_openrouter_pipe/
├─ open_webui_openrouter_pipe.py   # single-file pipe
├─ pytest_bootstrap.py            # pytest plugin hook
├─ docs/                          # this documentation set
├─ tests/                         # isolated unit/contract tests
└─ *.md                           # compatibility docs kept for legacy readers
```

### 3.2 running tests

1. Create/activate a Python 3.12 virtualenv.
2. Install dependencies that Open WebUI would normally inject:
   ```bash
   pip install -r <(python - <<'PY'
import pkg_resources, json
print("aiohttp cryptography fastapi httpx lz4 pydantic sqlalchemy tenacity")
PY
   )
   ```
3. Export `PYTHONPATH=.` so `pytest` can import `pytest_bootstrap.py`.
4. Run targeted suites:
   ```bash
   PYTHONPATH=. pytest tests/test_multimodal_inputs.py
   PYTHONPATH=. pytest tests/test_pipe_guards.py
   ```
5. When editing the streaming or persistence layers, add/extend tests near the affected helper to keep coverage meaningful.

### 3.3 pytest bootstrap

`pytest_bootstrap.py` registers lightweight shims for Open WebUI modules so the pipe can be imported outside the server. It injects stub `open_webui` modules into `sys.modules`, patches SQLAlchemy engines to use SQLite in memory, and keeps tenacity retries deterministic. Always import `pytest_bootstrap` *before* the pipe inside new tests; the helper ensures CI runs do not hit the real database or network.

---

## 4. coding conventions

* **Async first:** Every path that touches OpenRouter, Redis, SQLAlchemy, or Open WebUI HTTP endpoints is async. Synchronous work (SQLAlchemy sessions, Fernet, LZ4 compression) is kept inside `ThreadPoolExecutor` offloads.
* **Bounded queues everywhere:** Request, streaming, tool, Redis, and logging pipelines all use explicit queue sizes so the process fails fast when overloaded. Preserve those limits when refactoring.
* **ContextVars for observability:** `SessionLogger` stores the session ID, user ID, and per-request log level. When you add new log statements, use `self.logger` (already context-aware) and include a concise prefix so status citations stay readable.
* **Valves as the single source of configuration:** Never read env vars directly in handlers. Either add a valve (and document it in `docs/valves_and_configuration_atlas.md`) or reuse the decrypted value stored on the `Pipe` instance.
* **User-facing status before exceptions:** Helper methods return human-readable messages via `_emit_status` / `_emit_notification` before raising or logging errors. Follow this pattern so Open WebUI users see progress and failure reasons even when logs are inaccessible.

---

## 5. onboarding checklist

1. Read `docs/documentation_index.md` → `docs/developer_guide_and_architecture.md` (you are here) → the subsystem doc you plan to modify.
2. Search for existing TODOs inside the target section of `open_webui_openrouter_pipe.py`. Many have context (e.g., why Redis write-behind is optional or why ProcessPool offload is deferred).
3. Update or add docs alongside code changes. The docs are opinionated; missing updates will block review.
4. Run targeted pytest files plus any new coverage you add.
5. Keep filenames lower-case when adding new docs to stay aligned with the navigation index.

If you follow this workflow you will naturally touch the right docs, tests, and code sections. The rest of `docs/` dives into the specifics--model catalog, history transforms, multimodal intake, tooling, streaming, persistence, ops, and valves.

---

## 6. automatic startup services

Before any request runs, the pipe bootstraps background helpers:

| Helper | Trigger | Notes |
| --- | --- | --- |
| `_maybe_start_log_worker()` | First request | Spins up an async logging queue so `SessionLogger` can stream request-scoped logs without blocking. |
| `_maybe_start_startup_checks()` | First request with an API key | Launches an OpenRouter warmup task (`/models?limit=1`). Failures set `_warmup_failed`, causing the pipe to reject new requests with "Service unavailable due to startup issues" until the probe succeeds. |
| `_maybe_start_redis()` | Whenever Redis preconditions are met | Detects multi-worker deployments (`UVICORN_WORKERS>1`, Redis URLs, valve enabled). Initializes namespaced pending queues, cache prefixes, and flush workers lazily. |
| `_maybe_start_cleanup()` | When persistence is available | Starts the artifact cleanup scheduler using the cadence from `ARTIFACT_CLEANUP_INTERVAL_HOURS`. |

These helpers are idempotent; they skip work if a previous run is still active. Understanding them is useful when debugging warmup failures or ensuring Redis/write-behind is enabled at runtime.

---

## 7. observability & per-request logging

`SessionLogger` is the backbone of diagnostics:

* **Context propagation** -- The logger stores `session_id` and `user_id` in ContextVars so every log line carries request identity. Request handlers set these values at the start of `pipe()`; cleanup drains them explicitly.
* **Async queue** -- `_maybe_start_log_worker()` installs an async logging queue. All log records go through `_enqueue()`, which either writes immediately or schedules the log on the main loop. Slow log sinks can"t block request threads.
* **In-memory buffer** -- Each session keeps up to 2k formatted log lines (`SessionLogger.logs[session_id]`). `_emit_error(..., show_error_log_citation=True)` can expose this buffer to end users as a citation, making debugging collaborative.
* **Automatic GC** -- `SessionLogger.cleanup()` removes sessions that have been idle for more than an hour, preventing unbounded memory usage.

When you add new logging statements, use `self.logger` so they inherit the per-session metadata automatically. If you need to inspect logs for a specific user, filter by the `session_id` displayed in the console formatter.


---

## Related Topics

**Core Systems:**
- **Model Selection & Routing**: [Model Catalog and Routing Intelligence](model_catalog_and_routing_intelligence.md) - How models are selected and routed based on capabilities
- **Message Transformation**: [History Reconstruction and Context](history_reconstruction_and_context.md) - How Open WebUI messages become API inputs
- **Configuration Reference**: [Valves and Configuration Atlas](valves_and_configuration_atlas.md) - Complete valve reference with defaults and ranges

**Feature Deep Dives:**
- **Multimodal Content**: [Multimodal Ingestion Pipeline](multimodal_ingestion_pipeline.md) - File, image, audio, and video handling
- **Tool Execution**: [Tooling and Integrations](tooling_and_integrations.md) - Function calling, MCP, web search
- **Streaming**: [Streaming Pipeline and Emitters](streaming_pipeline_and_emitters.md) - SSE events and worker pools
- **Data Persistence**: [Persistence, Encryption & Storage](persistence_encryption_and_storage.md) - Database schema and encryption

**Operations:**
- **Error Handling**: [Error Handling and User Experience](error_handling_and_user_experience.md) - Error templates and troubleshooting
- **Testing & Deployment**: [Testing, Bootstrap, and Operations](testing_bootstrap_and_operations.md) - Testing workflows and operational runbooks
- **Security**: [Security and Encryption](security_and_encryption.md) - Security procedures and compliance
- **Production Readiness**: [Production Readiness Report](production_readiness_report.md) - Pre-deployment checklist

