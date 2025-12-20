# Open WebUI pipe for OpenRouter Responses API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.10-blue.svg)](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe)
[![Open WebUI Compatible](https://img.shields.io/badge/Open%20WebUI-0.6.28%2B-green.svg)](https://openwebui.com/)

A production-grade Open WebUI pipe for OpenRouter's Responses API, built for reliability and scale. Three core capabilities set it apart:

1. **Catalog-smart routing** – Imports full model metadata from OpenRouter to automatically enable/disable features per model
2. **Durable artifact persistence** – Encrypted, compressed storage for reasoning payloads and tool outputs with ULID-based replay
3. **Resilient streaming & tool orchestration** – Breaker-protected execution, Redis-aware scaling, and comprehensive telemetry

Every layer—from model registry to SSE emitters—was designed for production workloads with multi-user concurrency, graceful degradation, and operational visibility.

![output](https://github.com/user-attachments/assets/c937443b-f1be-4091-9555-b49789f16a97)

## Overview

This pipe focuses on the capabilities unique to OpenRouter Responses deployments:
- **Model intelligence**: Imports the full `/models` payload (not just names) and derives modality/tool flags, context limits, and supported parameters so Open WebUI exposes exactly what each model can do.
- **Artifact durability**: Every remote/file upload is saved into Open WebUI storage, inline payloads are size-guarded, and reasoning/tool outputs land in per-pipe SQL tables with optional Fernet + LZ4 to keep chats recoverable months later.
- **Streaming + tool orchestration**: Dedicated SSE workers, breaker-aware tool queues, and usage-rich emitters keep long-running reasoning/tool chains responsive while surfacing time/cost/token telemetry to operators.

### Feature Highlights
- **Catalog-smart Responses import** – Async registry fetches `/models`, derives modality/tool flags, and backs off automatically when OpenRouter throttles.
- **Multimodal intake with guard rails** – Remote downloads flow through SSRF filters, retries, and MB caps, while base64/audio/video payloads are validated before decoding.
- **Secure persistence** – Uses Open WebUI's existing database connection (no additional configuration needed). Per-pipe SQLAlchemy tables with optional Fernet encryption, ULID markers, and LZ4 compression keep reasoning/tool artifacts safe yet replayable.
- **Resilient tool + streaming pipeline** – FIFO tool queues with breaker windows, SSE worker pools, and telemetry-rich emitters keep multi-minute reasoning/tool chains responsive without starving other users.
- **Error handling & user experience** – Comprehensive error templates for network timeouts, connection failures, 5xx service errors, and internal exceptions. Each error includes unique error IDs, timestamps, and session context for troubleshooting. Production-ready defaults with full admin customization via valves.
- **Operational safeguards** – Request queue + global semaphore, Redis write-behind cache (auto-enables only when Open WebUI sets `UVICORN_WORKERS>1`, `REDIS_URL`, and `WEBSOCKET_REDIS_URL`), artifact cleanup workers, and per-session logging via `SessionLogger` keep it production-ready.
- **Opt-in cost telemetry snapshots** – `COSTS_REDIS_DUMP` and `COSTS_REDIS_TTL_SECONDS` valves stream per-turn usage payloads into Redis for downstream billing or anomaly detection without impacting the main response path. See [OpenRouter Integrations & Telemetry → Opt-in cost snapshots](docs/openrouter_integrations_and_telemetry.md#11-opt-in-cost-snapshots-redis-export) for details.

## Documentation

**[Documentation Index](docs/documentation_index.md)** – Central navigation hub with persona-based reading paths (Developer, Operator, Security, Auditor) and a visual relationship map showing how all 15+ docs interconnect. Features quick navigation callouts and Related Topics sections throughout.

**[Changelog](CHANGELOG.md)** – Complete commit-by-commit history tracking every change since project inception.

  **Developer & Architecture Guides**
  - [Developer Guide & Architecture](docs/developer_guide_and_architecture.md) – High-level tour of manifold wiring, layering conventions, request lifecycle, and development workflow.
  - [Model Catalog & Routing Intelligence](docs/model_catalog_and_routing_intelligence.md) – Registry loader, capability detection, reasoning toggles, and routing helpers.

  **Modality & Interface Layers**
  - [History Reconstruction & Context](docs/history_reconstruction_and_context.md) – How Open WebUI messages become Responses `input[]` blocks with ULID markers and artifact replay.
  - [Multimodal Ingestion Pipeline](docs/multimodal_ingestion_pipeline.md) – Complete walkthrough of image/file/audio/video handling, SSRF protections, and persistence.
  - [Task Models & Housekeeping](docs/task_models_and_housekeeping.md) – Background task configuration, model selection, and troubleshooting.
  - [Tooling & Integrations](docs/tooling_and_integrations.md) – Tool sources, strict schema enforcement, batching, MCP servers, and OpenRouter's web search plugin.
  - [Streaming Pipeline & Emitters](docs/streaming_pipeline_and_emitters.md) – SSE queues, worker pools, reasoning/citation events, and completion finalizers.

  **Durability & State**
  - [Persistence, Encryption & Storage](docs/persistence_encryption_and_storage.md) – SQLAlchemy models, ULID markers, encryption, compression, Redis write-behind, and cleanup workers.
  - [Concurrency Controls & Resilience](docs/concurrency_controls_and_resilience.md) – Admission control, ContextVars, session logging, breaker windows, and overload fallbacks.
  - [Testing, Bootstrap & Operations](docs/testing_bootstrap_and_operations.md) – Pytest bootstrap plugin, dev setup, CI guidance, warmup probes, and production readiness checks.

  **Reference Materials**
  - [Valves & Configuration Atlas](docs/valves_and_configuration_atlas.md) – Exhaustive listing of system + user valves with defaults, ranges, and rationale.
  - [Security & Encryption Guide](docs/security_and_encryption.md) – Encryption setup, key rotation, SSRF protection, and compliance guidance.
  - [OpenRouter Integrations & Telemetry](docs/openrouter_integrations_and_telemetry.md) – OpenRouter-specific features including usage strings, catalog routing, plugin wiring, and 400 error templates.
  - [Error Handling & User Experience](docs/error_handling_and_user_experience.md) – Comprehensive error template system with rendered examples, troubleshooting guide, and operator runbook for all exception types.
  - [Production Readiness Report](docs/production_readiness_report.md) – Comprehensive audit covering secrets, persistence, multimodal guardrails, concurrency, streaming, and observability.

## What Makes This Pipe Unique

### Production-Grade OpenRouter Integration
This is not a simple API wrapper. The pipe has been extensively enhanced and hardened for production OpenRouter deployments with:
- **Intelligent catalog routing** – Automatically detects model capabilities (vision, audio, reasoning, web search, MCP) from OpenRouter's `/models` endpoint and exposes only what each model can do
- **Durable artifact persistence** – Every reasoning payload and tool output is encrypted (Fernet), compressed (LZ4), and stored with ULID markers so conversations survive server restarts and can be replayed months later
- **Redis-aware scaling** – Detects multi-worker Open WebUI deployments and automatically enables distributed caching with write-behind batching and breaker-protected fallbacks
- **Comprehensive error handling** – All exception types (timeouts, connection failures, 5xx errors, internal exceptions) render as user-friendly Markdown cards with unique error IDs for support correlation

### Security & Resilience at Scale
- **Encryption at rest** – Fernet encryption for reasoning tokens and tool outputs. Requires `WEBUI_SECRET_KEY` environment variable. See **[Security & Encryption Guide](docs/security_and_encryption.md)** for setup.
- **SSRF protection** on remote file downloads with configurable retry/backoff and MB caps that honor Open WebUI's RAG upload ceiling
- **Breaker-protected tool execution** with per-user/per-tool failure windows to prevent runaway retries
- **Request queue + semaphores** that return HTTP 503 when saturated instead of crashing workers
- **Session-aware logging** that ties every log line to a request via contextvars for troubleshooting

### Developer & Operator Experience
- **Valve-driven configuration** – Concurrency limits, retry policies, encryption keys, streaming profiles, and other operational settings are configurable through Open WebUI's valve system with production-ready defaults
- **Built-in telemetry** – Optional status messages show elapsed time, cost, tokens, and throughput per request when `SHOW_FINAL_USAGE_STATUS` is enabled
- **Comprehensive documentation** – 15+ interconnected docs with persona-based navigation (Developer, Operator, Security, Auditor), quick navigation callouts, visual relationship map, and Related Topics sections covering architecture, multimodal pipelines, persistence, streaming, testing, and operations

For detailed technical documentation, see the [Documentation Index](docs/documentation_index.md).

## Installation

1. **Prerequisites**
   - Open WebUI 0.6.28 or later.
   - OpenRouter API key (set via valves or environment variables).

2. **Add the pipe in Open WebUI**
   - Admin → Functions → New Function.
   - Upload the `open_webui_openrouter_pipe.py` file.

3. **Dependencies**
   - Automatically installed by Open WebUI via the `requirements:` header (`aiohttp`, `cryptography`, `fastapi`, `httpx`, `lz4`, `pydantic`, `sqlalchemy`, `tenacity`, etc.).

4. **Enable the pipe**
   - Navigate to the model selector in Open WebUI.
   - Newly imported models appear automatically after enabling the pipe.

## Usage

- **Model selection**: Pick any imported OpenRouter model (e.g., `openrouter.gpt-4o-mini`) inside Open WebUI.
- **Conversations**: Send messages; the pipe injects reasoning/tool settings automatically if the model supports them and persists artifacts for future turns.
- **Tools & plugins**: Enable registered tools, web search, or MCP servers via valves. Tool outputs are stored securely and replayed when needed.
- **Monitoring**: Watch streaming deltas, citations, notifications, and the final usage/status summary in the UI.

## Configuration (Valves)

All configuration is managed through Open WebUI's valve system. For comprehensive documentation of all system and user valves, including defaults, ranges, and configuration guidance, see **[Valves & Configuration Atlas](docs/valves_and_configuration_atlas.md)**.

## Security & Encryption

Artifact encryption requires both `WEBUI_SECRET_KEY` (environment variable) and `ARTIFACT_ENCRYPTION_KEY` (valve) to be configured. Without `WEBUI_SECRET_KEY`, all artifacts are stored unencrypted regardless of other settings.

For complete security documentation including encryption setup, key rotation procedures, SSRF protection, and multi-tenant isolation, see **[Security & Encryption Guide](docs/security_and_encryption.md)**.

## Testing

`tests/conftest.py` stubs Open WebUI, FastAPI, SQLAlchemy, pydantic-core, and tenacity so the pipe can be imported without the full server. Because the pytest plugin `open_webui_openrouter_pipe.pytest_bootstrap` lives inside this repository, set `PYTHONPATH=.` (or install the package in editable mode) whenever you run the suite; otherwise pytest cannot locate the plugin.

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_multimodal_inputs.py   # multimodal + size guard coverage
PYTHONPATH=. .venv/bin/pytest tests/test_pipe_guards.py         # request + valve guard coverage
PYTHONPATH=. .venv/bin/pytest tests/test_error_templates.py     # error template + exception handling coverage
```

Extend the suite with additional files under `tests/` when contributing new features or bug fixes. For detailed testing guidelines, see the [Testing, Bootstrap & Operations](docs/testing_bootstrap_and_operations.md) guide.

## Acknowledgments

This project builds on the [OpenAI Responses Manifold](https://github.com/jrkropp/open-webui-developer-toolkit/tree/development/functions/pipes/openai_responses_manifold) by jrkropp while adapting it for OpenRouter's ecosystem and adding the operational hardening needed for production deployments.
