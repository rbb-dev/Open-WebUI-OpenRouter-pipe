# OpenRouter Responses API Manifold for Open WebUI

## This pipe is under active development, design might change. 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.4-blue.svg)](https://github.com/rbb-dev/openrouter-responses-pipe) <!-- Update with your repo link -->
[![Open WebUI Compatible](https://img.shields.io/badge/Open%20WebUI-0.6.28%2B-green.svg)](https://openwebui.com/)

This pipe provides seamless integration between Open WebUI and OpenRouter's Responses API. It automatically discovers and imports all Responses-capable models from OpenRouter, translates Open WebUI requests into the Responses format, and manages persistent storage of conversation artifacts. Designed for developers building robust AI applications, it ensures secure, efficient handling of multi-turn interactions, tools, and reasoning while leveraging Open WebUI's ecosystem.

<img width="778" height="500" alt="image" src="https://github.com/user-attachments/assets/00da49a3-6ff0-400e-bb6c-7c540090a593" />

## Overview

Open WebUI is a versatile platform for AI-driven interfaces. This pipe extends it by bridging to OpenRouter's Responses API, which supports advanced features like structured reasoning, tool calling, and plugins. It handles request transformation, model capabilities detection, and artifact persistence in Open WebUI's database—ensuring compatibility across various database backends (e.g., SQLite, PostgreSQL).

Key capabilities include:
- Dynamic model import from OpenRouter's catalog.
- Secure storage of reasoning traces and tool results with encryption and compression.
- Optimized tool schemas for predictable function calling.
- Real-time streaming with citations and usage metrics.

This project is an extension of the original [OpenAI Responses Manifold](https://github.com/jrkropp/open-webui-developer-toolkit/tree/development/functions/pipes/openai_responses_manifold) by jrkropp, adapted and enhanced for OpenRouter.

## Features

- **Model Auto-Discovery**: Fetches OpenRouter's model catalog via the `/models` endpoint, importing all Responses-capable models with detailed metadata (e.g., function calling, reasoning, plugin support). This provides more granular feature detection than OpenAI's equivalent, enabling precise tool integration.
- **Request Translation**: Converts Open WebUI's Completions-style requests to Responses format, supporting tools, plugins, and reasoning parameters.
- **Artifact Persistence**: Stores reasoning traces and tool results in Open WebUI's internal database (via `open_webui.internal.db`) using SQLAlchemy. Artifacts are saved in a dedicated, per-pipe table for isolation and scalability.
- **Security**: Encrypts artifacts using a user-provided key (via `ARTIFACT_ENCRYPTION_KEY`). Changing the key creates a new table by design, rendering old artifacts inaccessible to prevent unauthorized access.
- **Efficiency**: Applies LZ4 compression for large payloads; auto-installs dependencies via Open WebUI's `requirements:` header.
- **Tool Management**: Strictifies schemas for reliable function calling; deduplicates tools; supports MCP servers and web search plugins.
- **Streaming and Metrics**: Handles SSE streaming with delta optimization, inline citations, and final usage reports (e.g., tokens, cost).
- **Configurable Valves**: Expose settings for logging, compression thresholds, tool persistence, and loop limits.
- **Future Enhancements**: Planned scheduler for cleaning unused tables (e.g., drop after 30 days of inactivity, configurable via valves) to manage storage.

## Improvements Over the Original

This pipe builds on jrkropp's foundational OpenAI Responses Manifold, which provided efficient request handling and basic persistence. Key extensions include:
- **OpenRouter Adaptation**: Added dynamic catalog fetching from OpenRouter's `/models` endpoint (richer than OpenAI's), model normalization, and support for OpenRouter-specific features like web search plugins.
- **Enhanced Persistence**: Integrated with Open WebUI's internal DB connection (`open_webui.internal.db`) for all operations, removing fallback logic and ensuring compatibility with OWUI's configured backend (e.g., SQLite or PostgreSQL). Added encryption with key rotation (new tables on change for security) and LZ4 compression.
- **Tool Optimizations**: Introduced schema strictification, deduplication, and MCP server support, leveraging OpenRouter's detailed metadata for more accurate feature detection.
- **Security and Reliability**: Expanded error handling, logging, and metrics; version bumped to 1.0.4 with removal of standalone DB fallbacks to fully rely on OWUI's infrastructure.
- **Other**: Improved async handling, multi-line status updates via CSS injection, and configurable valves for finer control.

These changes maintain the original's efficiency while tailoring it for OpenRouter's ecosystem and enhancing security/scalability.

## Operational Architecture

This pipe is tuned for multi-user deployments and adds several protective layers on top of Open WebUI's standard execution:

- **Request queue + global semaphore**: every request acquires a slot from a process-wide semaphore, so bursts are back-pressured with a 503 "Server busy" instead of crashing the worker.
- **Redis write-behind cache (optional)**: when Redis is detected, tool artifacts are enqueued through a pending queue guarded by distributed locks. A background flusher persists batched rows to the Open WebUI database without blocking the hot path.
- **Breakers and batching**: per-user/per-tool breaker windows prevent runaway retries, while the tool executor batches compatible calls and enforces per-call, per-batch, and idle timeouts.
- **Resilient event emitters**: every status/SSE emitter is wrapped so client disconnects or slow browsers cannot crash the pipe; failures are logged while the request finishes cleanup.

These mechanisms keep the manifold responsive under load spikes, unreliable tools, or flaky networks.

## Installation

1. **Prerequisites**:
   - Open WebUI version 0.6.28 or later.
   - OpenRouter API key (set via valves or environment).

2. **Add the pipe in Open WebUI**:
   - Go to Admin > Functions -> New Function.
   - Upload the pipe Python file from this repository.

3. **Dependencies**: Automatically installed by Open WebUI via the `requirements:` header in the code (e.g., `aiohttp`, `cryptography`, etc.).

4. **Restart Open WebUI**: Imported models will appear in the model selection UI.

## Usage

- **Model Selection**: Choose an imported OpenRouter model in Open WebUI (e.g., "openrouter.gpt-4o").
- **Conversations**: Send messages; features like tools or reasoning activate if supported by the model. Artifacts persist in OWUI's DB for context retention.
- **Tools/Plugins**: Enable via valves (e.g., web search); results are stored securely.
- **Monitoring**: Observe streaming updates, citations, and usage metrics in the UI.

For persistence: Configure `ARTIFACT_ENCRYPTION_KEY` in valves. Note that key changes create new tables (by design for security)—old data becomes inaccessible.

## Configuration (Valves)

Adjust in Open WebUI:
- **API_KEY**: OpenRouter API key (encrypted).
- **MODEL_ID**: CSV of models to import ("auto" for all).
- **PERSIST_TOOL_RESULTS**: Enable/disable tool storage.
- **ARTIFACT_ENCRYPTION_KEY**: Encryption key (min 16 chars).
- **ENABLE_WEB_SEARCH_TOOL**: Activate web search.
- **LOG_LEVEL**: Set verbosity (e.g., "DEBUG").

Full schema in the code.

## Testing

Unit tests live under `tests/` and include a `conftest.py` that stubs Open WebUI, pydantic v2 hooks, pydantic-core, SQLAlchemy, and tenacity so the pipe can be imported outside of an OWUI runtime. To run the current regression suite (which covers the guard rails added in this release):

```bash
pytest tests/test_pipe_guards.py
```

Extend the suite with additional files under `tests/` when contributing new features or bug fixes.

## Acknowledgments

This project extends the [OpenAI Responses Manifold](https://github.com/jrkropp/open-webui-developer-toolkit/tree/development/functions/p
