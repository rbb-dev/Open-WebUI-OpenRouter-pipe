# Developer guide and architecture

**Scope:** High-level architecture map for contributors/operators who need to navigate the codebase safely and understand request flow.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🧪 Testing](testing_bootstrap_and_operations.md) | [🔒 Security](security_and_encryption.md)

This repository ships a single Open WebUI pipe module, but it implements multiple subsystems (model registry, transforms, streaming, tools, persistence, Redis, session logs). This guide points you to the correct entry points and the deeper docs for each subsystem.

---

## Repository layout (what to read first)

- `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py`: the pipe implementation (single module).
- `tests/`: unit tests scoped by subsystem.
- `docs/`: documentation set (this folder).

---

## Architectural building blocks (code-level)

Key components you will see repeatedly:

- `Pipe`: the Open WebUI pipe controller. Owns valves, request admission, streaming/non-streaming execution, persistence, and background workers.
- `CompletionsBody` and `ResponsesBody`: request models that translate Open WebUI chat-completions-style payloads into OpenRouter Responses API payloads.
- `OpenRouterModelRegistry` and `ModelFamily`: model catalog loading, normalization, and capability/supported-parameter helpers.
- `SessionLogger`: per-request logging (stdout + in-memory buffer) keyed by a per-request `request_id` with `session_id`/`user_id` attached via context variables.

---

## High-level request lifecycle (normal chat requests)

At a high level, a request follows this shape:

1. **Admission and isolation**
   - Requests are queued into a bounded per-process request queue and executed under a per-process concurrency semaphore.
   - Each request receives its own `aiohttp.ClientSession` and per-request logging context.

2. **Normalization and transforms**
   - The incoming Open WebUI payload is normalized into a `ResponsesBody` (history reconstruction, multimodal transforms, request defaults).
   - Identifier valves are applied (`SEND_*`), and the outbound request is filtered to the OpenRouter allowlist.

3. **Provider call and streaming**
   - The pipe calls the OpenRouter Responses API in streaming mode and emits Open WebUI events (`status`, `chat:message`, `chat:completion`, citations, and optional reasoning events).

4. **Tool-call loop (between Responses calls)**
   - When a Responses run completes, the pipe inspects the returned `output` items.
   - `function_call` items are executed locally against the Open WebUI tool registry, converted into `function_call_output` items, appended to the next request’s `input[]`, and the loop continues until no further tool calls are produced or `MAX_FUNCTION_CALL_LOOPS` is reached.

5. **Persistence (optional)**
   - Depending on valves, artifacts (reasoning/tool outputs) are persisted to SQL storage (optionally encrypted and/or compressed) and may be cached in Redis in multi-worker configurations.

---

## Background workers (when they start and why)

The pipe starts helper workers lazily (per process):

- **Request queue worker**: drains the bounded request queue and isolates per-request context.
- **Log worker**: drains log records asynchronously so logging does not block request handling.
- **Artifact cleanup loop** (when persistence is available): periodically deletes old rows based on retention valves.
- **Redis workers** (when enabled and prerequisites are met): write-behind flush and pub/sub listeners for multi-worker cache behavior.
- **Session log writer/cleanup threads** (when enabled): writes encrypted session log archives and prunes old archives.

---

## Contribution workflow (practical)

- Keep changes scoped: update one subsystem at a time and add/extend tests in the corresponding `tests/test_*.py`.
- Update docs alongside behavior changes (prefer the subsystem doc under `docs_codex/` rather than embedding long comments in code).
- Run the relevant unit tests and then the full suite (see [Testing, bootstrap, and operational playbook](testing_bootstrap_and_operations.md)).

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
- [Testing, bootstrap, and operational playbook](testing_bootstrap_and_operations.md)
- [Production readiness report (OpenRouter Responses Pipe)](production_readiness_report.md)
