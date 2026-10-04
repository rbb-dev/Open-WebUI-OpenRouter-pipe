# Developer guide and architecture

**Scope:** High-level architecture map for contributors/operators who need to navigate the codebase safely and understand request flow.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🧪 Testing](testing_bootstrap_and_operations.md) | [🔒 Security](security_and_encryption.md)

This repository ships an Open WebUI pipe as a modular Python package implementing multiple subsystems (model registry, transforms, streaming, tools, persistence, Redis, session logs). This guide points you to the correct entry points and the deeper docs for each subsystem.

---

## Repository layout (what to read first)

```
open_webui_openrouter_pipe/
├── __init__.py          # Package entry point with lazy loading
├── pipe.py              # Main Pipe class and request handling
├── api/                 # Gateway adapters and transforms
│   ├── transforms.py    # Request/response transforms
│   └── gateway/         # OpenRouter API adapters
├── filters/             # Regenerable reference filter copies (scripts/build_reference_filters.py)
├── core/                # Config, logging, circuit breaker, timing
│   ├── config.py        # Valve definitions
│   ├── logging_system.py # Session logging
│   ├── timing_logger.py # Performance instrumentation
│   └── circuit_breaker.py
├── models/              # Model registry and capabilities
├── requests/            # Request orchestration and debug
├── storage/             # Artifact and file handling
├── streaming/           # SSE parsing, event emission
└── tools/               # Tool execution and persistence
```

- `tests/`: unit tests scoped by subsystem.
- `docs/`: documentation set (this folder).

---

## Architectural building blocks (code-level)

Key components you will see repeatedly:

- `Pipe`: the Open WebUI pipe controller. Owns valves, request admission, streaming/non-streaming execution, persistence, and background workers.
- `CompletionsBody` and `ResponsesBody`: request models that translate Open WebUI chat-completions-style payloads into OpenRouter Responses API payloads.
- `OpenRouterModelRegistry` and `ModelFamily`: model catalog loading, normalization, and capability/supported-parameter helpers.
- `SessionLogger`: per-request logging (stdout + in-memory buffer) keyed by a per-request `request_id` with `session_id`/`user_id` attached via context variables. The package logger is wired once per logger name, and a later `Pipe` re-attaches rather than re-wiring, so a hot reload cannot silence a live turn.

---

## High-level request lifecycle (normal chat requests)

At a high level, a request follows this shape:

1. **Admission and isolation**
   - Requests are queued into a bounded per-process request queue and executed under a per-process concurrency semaphore.
   - Each request gets its own per-request logging context, and shares one pooled `aiohttp.ClientSession` per event loop, which is closed at shutdown. Because that session outlives a single request, the timeout valves are applied per call at every site that issues an outbound request; the pooled session's own default tracks the valves of the request that most recently took it, and that default applies only to a site that supplies no `timeout=` of its own.

2. **Normalization and transforms**
   - The incoming Open WebUI payload is normalized into a `ResponsesBody` (history reconstruction, multimodal transforms, request defaults).
   - Identifier valves are applied (`SEND_*`), and the outbound request is filtered to the OpenRouter allowlist.

3. **Provider call and streaming**
   - The pipe calls the OpenRouter Responses API in streaming mode and emits Open WebUI events (`status`, `chat:message:delta` for streamed answer text, `chat:completion`, citations, and optional reasoning events); `chat:message` is reserved for whole-message snapshots that carry a card (see [Streaming Pipeline & Emitters](streaming_pipeline_and_emitters.md)).

4. **Tool-call loop (between Responses calls)**
   - When a Responses run completes, the pipe inspects the returned `output` items.
   - `function_call` items are executed locally against the Open WebUI tool registry, converted into `function_call_output` items, appended to the next request’s `input[]`, and the loop continues until no further tool calls are produced or `MAX_FUNCTION_CALL_LOOPS` is reached (at which point the model gets a synthesis turn). Applies whenever the pipe runs the calls, whatever the mode.
   - A separate, smaller loop governs hand-backs: when a reply hands a call back to its sender, Open WebUI re-asks the same `(user_id, chat_id, message_id)` rather than the pipe running it, and the pipe charges that hand-back against a per-reply budget that belongs to that one user's spend, so two users naming the same chat and message are charged to two budgets and neither spends the other's so a runaway re-ask cannot exceed `MAX_FUNCTION_CALL_LOOPS` upstream requests. When the budget is spent the pipe answers in its own loop, against the registry it built, so the post-cap round runs the tools it advertised — except a round of browser-run tools, Open WebUI's own builtins, or the tools this request withheld from the model under `ask` in a streamed saved chat or under `legacy` function calling, which is handed back whatever the budget says, because Open WebUI is what runs those; such a reply is bounded by Open WebUI's own iteration limit instead. The budget is per-reply **only for replies that carry a `chat_id` and a `message_id`**; an internal-Fusion panel member and any chat-less `/chat/completions` or `/responses` caller is handed back once per request instead, because only a reply carrying both ids is charged at all. A reply's budget is released on the next hand-back once its key has been idle for an hour; a reply that is still being re-asked keeps it. A temporary chat is charged too — it is charged under a key that carries no chat id at all, because its id is the browser's socket id and the budget is held in a process-lifetime map — and its budget is still per-reply, keyed on the `message_id`, so a Continue on a temporary chat carries the budget on exactly as it does in a saved chat. The reply's own budget entry is released when the reply ends, and the key it was charged under never carried the chat id, so a temporary chat's id is never held in a process-lifetime structure after its own turn.
   - On the `/chat/completions` transport, a streamed tool-call frame that carries no `index` continues the call its `id` or `name` names; all three arms carry the same parseability condition, so a match requires the candidate call's arguments to be still incomplete and two calls to the same tool stay two; and a frame carrying no `index`, `id` or `name` continues the most recent call when, and only when, exactly one call is open.

5. **Persistence (optional)**
   - Depending on valves, artifacts (reasoning/tool outputs) are persisted to SQL storage (optionally encrypted and/or compressed) and may be cached in Redis in multi-worker configurations.

---

## Background workers (when they start and why)

The pipe starts helper workers lazily:

- **Request queue worker**: drains the bounded request queue and isolates per-request context.
- **Log worker**: drains log records asynchronously so logging does not block request handling.
- **Artifact cleanup loop** (when persistence is available): periodically deletes old rows based on retention valves.
- **Redis workers** (when enabled and prerequisites are met): write-behind flush and pub/sub listeners for multi-worker cache behavior. Whatever is still in the pending queue is drained before these tasks are cancelled. The artifact cleanup loop, the Redis workers and the warmup are never started on a superseded instance: the helpers that start them refuse once that instance has been closed, so a model-list refresh Open WebUI still makes on a retired generation cannot leave a worker nothing will ever stop. The same is true of everything else `pipes()` maintains rather than merely serves: on a retired generation the refresh returns that instance's own rows and writes nothing -- no filter install, retire, repair or reactivate, no startup stale-id prune, no metadata sync, no `on_models` dispatch, and so no plugin re-point of the dashboard's module globals. That covers a refresh already past the guard when the close lands under it, not only one entered on a retired instance: the write region is gated again after the catalogue load, so the whole region is skipped whichever of the two the refresh meets. It returns rows rather than refusing, because Open WebUI substitutes an empty list for anything `pipes()` raises.
- **Session log writer/cleanup threads** (when enabled): writes encrypted session log archives and prunes old archives.
- **Plain-function tool pool** (started on the first sync tool call): a `ThreadPoolExecutor` `min(MAX_PARALLEL_TOOLS_GLOBAL, 8)` threads wide that plain-`def` tool bodies run on, so a user-supplied blocking tool cannot take the threads Open WebUI's own requests and this pipe's storage work use. Its width is recomputed on every call, so a valve change resizes it without a restart, and a teardown or a resize never cancels work already admitted to it.

**State ownership:**
- **Instance-level**: request queue, log queue, worker tasks, and locks are owned by each Pipe instance (prevents event loop contamination across async contexts). The plain-function tool pool belongs here as well, not to the class-level half below.
- **Process-level**: the rate-limiting semaphores (`request_semaphore`, `tool_semaphore`, `video_semaphore`, each with its limit slot beside it) are shared across every instance in the same process to enforce global concurrency limits, and are written only by a live (non-retired) instance. They are keyed by pipe id and live in a holder under a key in `sys.modules` rather than on the `Pipe` class, because a hot reload re-executes the pipe into a fresh module and builds a fresh subclass of `Pipe`: class-level slots gave each generation its own pool, so for as long as a reload overlapped the generation it replaced the worker admitted the ceiling twice. Two installed copies are two ids with two valves and stay separate.

---

## Contribution workflow (practical)

- Keep changes scoped: update one subsystem at a time and add/extend tests in the corresponding `tests/test_*.py`.
- Update docs alongside behavior changes (prefer the subsystem doc under `docs/` rather than embedding long comments in code).
- Run the relevant unit tests and then the full suite (see [Testing, bootstrap, and operational playbook](testing_bootstrap_and_operations.md)).

---

## Debugging and performance profiling

### Timing instrumentation

The pipe includes a built-in timing system for diagnosing performance issues. Enable it via the `ENABLE_TIMING_LOG` valve.

When enabled, timing events are written directly to `TIMING_LOG_FILE` (default: `logs/timing.jsonl`) with each event tagged by `request_id` for correlation. The system provides three mechanisms:

```python
from open_webui_openrouter_pipe.core.timing_logger import timed, timing_scope, timing_mark

# 1. @timed decorator - automatic function entrance/exit
@timed
async def my_function():
    ...

@timed
async def my_generator():
    # an async generator's span brackets its whole iteration,
    # not its construction
    yield "one"

# 2. timing_scope() - time specific code blocks
with timing_scope("expensive_operation"):
    do_work()

# 3. timing_mark() - record point-in-time events
timing_mark("first_chunk_received")
```

All three record only when timing is enabled *at call entry*, and only when a request id is bound. A body that enables timing mid-flight gets no `enter`/`exit` for itself, and a function whose worker was started on an empty context records nothing at all — which frames a worker inherits decide what its jobs record, so a background loop started inside a request must be given its own context if its work is to be attributed anywhere. The same holds for an async generator: its `enter`/`exit` pair is bound to the request id set at the moment the generator was *called*, and its span covers the whole iteration — first `__anext__` to exhaustion, `aclose()`, or a thrown exception — rather than the construction of the generator object. A generator built and then dropped without being iterated records its `enter` and no `exit`.

Key functions already instrumented:

- `StreamingHandler._run_streaming_loop` — the main streaming loop
- `StreamingHandler._select_llm_endpoint` — endpoint selection logic

For full documentation including JSONL schema and usage examples, see [Session Log Storage → Timing instrumentation](session_log_storage.md#timing-instrumentation).

---

## Related topics (deep dives)

Core systems:

- [Valves & Configuration Atlas](valves_and_configuration_atlas.md)
- [Model Catalog & Routing Intelligence](model_catalog_and_routing_intelligence.md)
- [History Reconstruction & Context Replay](history_reconstruction_and_context.md)

Feature deep dives:

- [Multimodal Intake Pipeline](multimodal_ingestion_pipeline.md)
- [Tools, plugins, and integrations](tooling_and_integrations.md)
- [Streaming Pipeline & Emitters](streaming_pipeline_and_emitters.md)
- [Persistence, Encryption & Storage](persistence_encryption_and_storage.md)

Operations:

- [Security & Encryption](security_and_encryption.md)
- [Error Handling & User Experience](error_handling_and_user_experience.md)
- [Session Log Storage](session_log_storage.md) — encrypted log archives and timing instrumentation
- [Testing, bootstrap, and operational playbook](testing_bootstrap_and_operations.md)
- [Production readiness report (OpenRouter Responses Pipe)](production_readiness_report.md)
