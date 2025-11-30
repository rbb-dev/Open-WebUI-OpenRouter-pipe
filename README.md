# OpenRouter Responses API Manifold for Open WebUI

## This pipe is under active development, design might change.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.5-blue.svg)](https://github.com/rbb-dev/openrouter-responses-pipe)
[![Open WebUI Compatible](https://img.shields.io/badge/Open%20WebUI-0.6.28%2B-green.svg)](https://openwebui.com/)

This manifold is tailored to OpenRouter’s Responses API. It specializes in (1) catalog-smart routing against OpenRouter’s rich model metadata, (2) durable multimodal artifact handling with encryption + compression, and (3) high-throughput streaming/tool orchestration backed by breakers, Redis, and deep telemetry. Everything—from model registry to SSE emitters—was re-architected with these production requirements in mind.

![clean](https://github.com/user-attachments/assets/65a23a29-33b9-485d-9d0b-7caeca502ae6)

## Overview

This pipe focuses on the capabilities unique to OpenRouter Responses deployments:
- **Model intelligence**: Imports the full `/models` payload (not just names) and derives modality/tool flags, context limits, and supported parameters so Open WebUI exposes exactly what each model can do.
- **Artifact durability**: Every remote/file upload is saved into Open WebUI storage, inline payloads are size-guarded, and reasoning/tool outputs land in per-pipe SQL tables with optional Fernet + LZ4 to keep chats recoverable months later.
- **Streaming + tool orchestration**: Dedicated SSE workers, breaker-aware tool queues, and usage-rich emitters keep long-running reasoning/tool chains responsive while surfacing time/cost/token telemetry to operators.

### Feature Highlights
- **Catalog-smart Responses import** – Async registry fetches `/models`, derives modality/tool flags, and backs off automatically when OpenRouter throttles.
- **Multimodal intake with guard rails** – Remote downloads flow through SSRF filters, retries, and MB caps, while base64/audio/video payloads are validated before decoding.
- **Secure persistence** – Per-pipe SQLAlchemy tables, optional Fernet encryption, ULID markers, and LZ4 compression keep reasoning/tool artifacts safe yet replayable.
- **Resilient tool + streaming pipeline** – FIFO tool queues with breaker windows, SSE worker pools, and telemetry-rich emitters keep multi-minute reasoning/tool chains responsive without starving other users.
- **Operational safeguards** – Request queue + global semaphore, Redis write-behind cache (auto-enables only when Open WebUI sets `UVICORN_WORKERS>1`, `REDIS_URL`, and `WEBSOCKET_REDIS_URL`), artifact cleanup workers, and per-session logging via `SessionLogger` keep it production-ready.

## Documentation
- **[Multimodal Input Pipeline](MULTIMODAL_IMPLEMENTATION.md)** – end-to-end walkthrough of image/file/audio handling, SSRF protections, and persistence.
- **[Valve Reference](VALVES_REFERENCE.md)** – full catalog of every system and user valve with defaults, ranges, and tuning guidance.

## Features

### Model & Request Pipeline
- **Dynamic catalog import**: The registry fetches OpenRouter's `/models` endpoint, normalizes IDs, and caches feature flags (vision, audio, reasoning, web search, MCP, etc.).
- **Capability-aware routing**: `ModelFamily` helpers (e.g., `supports("function_calling")`) ensure the pipe only enables features the selected model can honor.
- **Completions → Responses transforms**: `ResponsesBody.from_completions` rewrites Open WebUI messages into Responses `input[]`, preserves `response_format`/`parallel_tool_calls`, injects persisted ULID artifacts, and keeps reasoning/tool state turn after turn.

### Multimodal Intake & Storage
- **Remote download safeguards**: HTTP downloads enforce SSRF bans, configurable retries/backoff, and a MB limit that also honors Open WebUI's RAG upload ceiling when one is configured.
- **Base64/video validation**: `_validate_base64_size` and `_parse_data_url` reject oversized inline payloads before decoding so large files can't exhaust memory.
- **Open WebUI storage integration**: Images/files are saved through Open WebUI's own storage endpoints, preventing link rot and keeping chat history portable.
- **Encrypted, compressed artifacts**: Per-pipe SQLAlchemy tables are created automatically; artifacts can be encrypted with `ARTIFACT_ENCRYPTION_KEY`, optionally LZ4-compressed above `MIN_COMPRESS_BYTES`, and tagged with ULIDs for replay/pruning.

### Artifact Persistence Flow
- **Per-pipe namespaces**: Artifact tables are named `response_items_<pipe>_<hash>` where the hash includes the encryption key. Rotating `ARTIFACT_ENCRYPTION_KEY` intentionally creates a new table so older ciphertexts remain unreadable.
- **ULID-based replay**: Every reasoning/tool payload is stored with a ULID marker so `_transform_messages_to_input()` can replay prior turns verbatim (and prune older ones when retention rules say so).
- **Selective encryption + compression**: Reasoning payloads are encrypted by default (or all artifacts when `ENCRYPT_ALL` is true); each payload carries a tiny header indicating whether it was LZ4-compressed.
- **Redis-backed write-behind**: When Open WebUI runs multiple workers *and* provides both `REDIS_URL` and `WEBSOCKET_REDIS_URL` (with `WEBSOCKET_MANAGER=redis`), artifacts queue into Redis, flush in batches, cache with TTLs for fast replays, and fall back to direct DB writes if breakers trip.
- **Cleanup + retention**: Scheduled jobs delete stale artifacts (based on `ARTIFACT_CLEANUP_*` valves) and retention settings (e.g., “reasoning only until next reply”) trigger `_delete_artifacts` to purge rows + cache entries once they have been replayed.

### Tooling & Streaming
- **Strict tool schemas**: Open WebUI registry entries are strictified for deterministic tool calling and deduped so the latest definition wins.
- **MCP + OpenRouter plugins**: Remote MCP servers (via JSON definition) and OpenRouter's `web` plugin are attached automatically for models that declare support.
- **Tool executor with breakers**: Each request gets a FIFO queue, per-request semaphore, global semaphore, batch cap, idle timeout, and per-user/per-tool breaker windows to prevent runaway retries.
- **Streaming worker pool**: Configurable SSE worker count plus chunk/event queues keep streams flowing while `_wrap_event_emitter` shields emitters from client disconnects.
- **Telemetry-rich emitters**: Citations, notifications, and the final completion frame include elapsed time, cost, tokens, and tokens-per-second when `SHOW_FINAL_USAGE_STATUS` is enabled, giving operators insight without extra dashboards.

### Security & Resilience
- **Session-aware logging**: `SessionLogger` ties every log line to a request via contextvars so troubleshooting output can optionally be surfaced as citations.
- **Request queue + semaphores**: A bounded queue and process-wide semaphore return HTTP 503 when the manifold is saturated instead of crashing workers.
- **Redis write-behind cache**: Auto-detects when Open WebUI is configured for multi-worker Redis ( `UVICORN_WORKERS>1`, `REDIS_URL`, `WEBSOCKET_MANAGER=redis`, `WEBSOCKET_REDIS_URL` ), batches artifacts through pending queues, and guards flushes with distributed locks; repeated failures automatically fall back to direct DB writes.
- **Artifact cleanup & size controls**: Scheduled cleanup removes stale rows while configurable caps for remote/base64/video payloads keep disk and RAM usage predictable.

### Roadmap
- Health endpoint exposing queue depth, breaker stats, and error rates.
- Dead-letter queue for unrecoverable tool outputs.
- Optional ProcessPool offload for CPU-bound or blocking tools.
- Prometheus-friendly metrics export (latency, TPS, cache hit rate).

## Operational Architecture

This pipe is tuned for multi-user workloads:
- **Request queue + semaphore**: Incoming jobs must acquire a global slot; overflow results in 503 "server busy" responses rather than worker crashes.
- **Redis write-behind cache**: Auto-detects Redis-backed multi-worker deployments ( `UVICORN_WORKERS>1`, `REDIS_URL`, `WEBSOCKET_MANAGER=redis`, `WEBSOCKET_REDIS_URL` ) and batches artifacts through pending queues guarded by distributed locks; repeated failures automatically fall back to direct DB writes.
- **Breakers and batching**: Per-user/per-tool breaker windows cap consecutive failures, while compatible tool calls can batch together, each bounded by per-call, per-batch, and idle timeouts.
- **Resilient emitters**: SSE/notification/citation emitters are wrapped so slow or disconnected clients never crash the request; errors are logged and processing continues.
- **Payload controls**: Remote downloads honor the configured MB ceiling (and Open WebUI's own upload limit when defined) while base64/video payloads are validated before decoding.
- **Usage-forward finalizer**: A closing completion event summarizes elapsed time, tokens, throughput, and cost so admins can monitor efficiency without extra instrumentation.

## Installation

1. **Prerequisites**
   - Open WebUI 0.6.28 or later.
   - OpenRouter API key (set via valves or environment variables).

2. **Add the pipe in Open WebUI**
   - Admin → Functions → New Function.
   - Upload the `openrouter_responses_pipe.py` file.

3. **Dependencies**
   - Automatically installed by Open WebUI via the `requirements:` header (`aiohttp`, `cryptography`, `fastapi`, `httpx`, `lz4`, `pydantic`, `sqlalchemy`, `tenacity`, etc.).

4. **Restart Open WebUI**
   - Newly imported models appear in the model selector.

## Usage

- **Model selection**: Pick any imported OpenRouter model (e.g., `openrouter.gpt-4o-mini`) inside Open WebUI.
- **Conversations**: Send messages; the pipe injects reasoning/tool settings automatically if the model supports them and persists artifacts for future turns.
- **Tools & plugins**: Enable registered tools, web search, or MCP servers via valves. Tool outputs are stored securely and replayed when needed.
- **Monitoring**: Watch streaming deltas, citations, notifications, and the final usage/status summary in the UI.

Configure `ARTIFACT_ENCRYPTION_KEY` if you need encrypted storage. Changing the key intentionally rotates to a new table and makes older artifacts inaccessible (defense-in-depth).

## Configuration (Valves)

See **[VALVES_REFERENCE.md](VALVES_REFERENCE.md)** for the full system + user valve catalog. Common knobs:

| Valve | Purpose |
| --- | --- |
| `API_KEY` | Encrypted OpenRouter credential injected into every request. |
| `MODEL_ID` | Comma-separated list of catalog IDs (`auto` imports everything). |
| `ARTIFACT_ENCRYPTION_KEY` | Enables encrypted reasoning/tool storage; key changes rotate tables. |
| `REMOTE_FILE_MAX_SIZE_MB` | Caps remote downloads and matches Open WebUI's upload ceiling when RAG uploads are enabled. |
| `ENABLE_REDIS_CACHE` | Governs Redis write-behind mode (requires `UVICORN_WORKERS>1`, `REDIS_URL`, `WEBSOCKET_MANAGER=redis`, and `WEBSOCKET_REDIS_URL`). |
| `LOG_LEVEL` | Pipe-level logging verbosity (DEBUG / INFO / …). |

Per-user overrides (log level, reasoning effort, streaming profile, etc.) are also documented in `VALVES_REFERENCE.md`.

## Testing

`tests/conftest.py` stubs Open WebUI, FastAPI, SQLAlchemy, pydantic-core, and tenacity so the pipe can be imported without the full server. Because the pytest plugin `openrouter_responses_pipe.pytest_bootstrap` lives inside this repository, set `PYTHONPATH=.` (or install the package in editable mode) whenever you run the suite; otherwise pytest cannot locate the plugin.

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_multimodal_inputs.py   # multimodal + size guard coverage
PYTHONPATH=. .venv/bin/pytest tests/test_pipe_guards.py         # request + valve guard coverage
```

Extend the suite with additional files under `tests/` when contributing new features or bug fixes.

## Acknowledgments

This project builds on the [OpenAI Responses Manifold](https://github.com/jrkropp/open-webui-developer-toolkit/tree/development/functions/pipes/openai_responses_manifold) by jrkropp while adapting it for OpenRouter's ecosystem and adding the operational hardening needed for production deployments.
