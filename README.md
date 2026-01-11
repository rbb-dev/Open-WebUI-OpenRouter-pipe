# Open WebUI pipe for OpenRouter Responses API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.1.1-blue.svg)](https://github.com/rbb-dev/Open-WebUI-OpenRouter-pipe)
[![Open WebUI Compatible](https://img.shields.io/badge/Open%20WebUI-0.6.28%2B-green.svg)](https://openwebui.com/)

A production-focused Open WebUI “pipe” that routes chat-completions-style traffic through OpenRouter’s Responses API, with capability-aware routing, multimodal transforms, tool calling, optional persistence, and operator controls via valves.

![output](https://github.com/user-attachments/assets/c937443b-f1be-4091-9555-b49789f16a97)

## Overview

Eight core capabilities:

1. **Catalog-smart routing**: loads OpenRouter model metadata and uses it to gate features by capability.
2. **Request shaping + schema enforcement**: filters outgoing requests to a documented OpenRouter Responses allowlist and normalizes common inputs (including `model_fallback` → `models`).
3. **Streaming SSE pipeline**: streaming Responses ingestion with Open WebUI event emission (status, message deltas, citations, completion/usage).
4. **Tool orchestration**: executes `function_call` outputs against Open WebUI’s tool registry and Direct Tool Servers between Responses calls, with worker pools and concurrency controls.
5. **Multimodal intake guardrails**: validated/normalized image/file/audio/video inputs, with SSRF/size controls where remote retrieval is involved.
6. **Optional persistence + replay**: persists selected reasoning/tool artifacts and replays them into later turns via marker-based reconstruction.
7. **Operational resilience**: bounded admission controls plus breaker-style degradation for repeated request/tool/DB failures.
8. **Operator telemetry and attribution**: optional cost snapshots to Redis and valve-gated request identifiers (`user`, `session_id`, `metadata.*`) for operations/abuse attribution.

## Compatibility

- Open WebUI: requires **0.6.28+** (per pipe manifest header).
- Python: **3.11+** (per `pyproject.toml`; matches official Open WebUI Docker images).

## Feature highlights

- **Catalog + capability detection**: capability-aware behavior for tool calling, vision inputs, supported parameters, and plugin attachment.
- **Automatic model metadata sync (optional)**: syncs OpenRouter model icons and capability checkboxes into Open WebUI model metadata (valves: `UPDATE_MODEL_IMAGES`, `UPDATE_MODEL_CAPABILITIES`).
- **Dual web search toggles (Web Search vs OpenRouter Search)**: Open WebUI’s built-in **Web Search** stays Open WebUI-native, while OpenRouter-native web search is surfaced as a separate **OpenRouter Search** toggle. The pipe can auto-install the companion toggle, show it only on models where it can work, and enable it by default for those models (admins can turn the default off per model; users can toggle it per chat). If both are enabled, OpenRouter Search takes precedence to avoid double-search.
- **Multimodal guardrails**: validated/normalized image/file/audio/video inputs; remote retrieval is size-limited and SSRF-guarded.
- **Tool execution**: choose between pipe-run tools (`TOOL_EXECUTION_MODE="Pipeline"`) and Open WebUI pass-through (`TOOL_EXECUTION_MODE="Open-WebUI"`); supports per-request/global parallelism controls (see [Tools, plugins, and integrations](docs/tooling_and_integrations.md)).
- **Operational controls**: valve-driven configuration for concurrency, timeouts, retention, templates, and telemetry.
- **User experience**: status updates and templated provider errors with operator-friendly context.

## Documentation

The documentation set lives in `docs/`

Key entry points:

- Start here: [Docs Home](docs/README.md)
- Configuration reference: [Valves & Configuration Atlas](docs/valves_and_configuration_atlas.md)
- Security: [Security & Encryption](docs/security_and_encryption.md)
- Troubleshooting and error templates: [Error Handling & User Experience](docs/error_handling_and_user_experience.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)

Complete documentation list:

- [Concurrency Controls & Resilience](docs/concurrency_controls_and_resilience.md)
- [Developer guide and architecture](docs/developer_guide_and_architecture.md)
- [Error Handling & User Experience](docs/error_handling_and_user_experience.md)
- [History Reconstruction & Context Replay](docs/history_reconstruction_and_context.md)
- [Model Catalog & Routing Intelligence](docs/model_catalog_and_routing_intelligence.md)
- [Multimodal Intake Pipeline](docs/multimodal_ingestion_pipeline.md)
- [OpenRouter Integrations & Telemetry](docs/openrouter_integrations_and_telemetry.md)
- [OpenRouter Direct Uploads (bypass OWUI RAG)](docs/openrouter_direct_uploads.md)
- [Persistence, Encryption & Storage](docs/persistence_encryption_and_storage.md)
- [Production readiness report (OpenRouter Responses Pipe)](docs/production_readiness_report.md)
- [Request identifiers and abuse attribution](docs/request_identifiers_and_abuse_attribution.md)
- [Security & Encryption](docs/security_and_encryption.md)
- [Encrypted session log storage (optional)](docs/session_log_storage.md)
- [Streaming Pipeline & Emitters](docs/streaming_pipeline_and_emitters.md)
- [Task models and housekeeping](docs/task_models_and_housekeeping.md)
- [Testing, bootstrap, and operational playbook](docs/testing_bootstrap_and_operations.md)
- [Tools, plugins, and integrations](docs/tooling_and_integrations.md)
- [Web Search (Open WebUI) vs OpenRouter Search](docs/web_search_owui_vs_openrouter_search.md)
- [Valves & Configuration Atlas](docs/valves_and_configuration_atlas.md)

## Installation (Open WebUI)

1. In Open WebUI, go to **Admin → Functions → New Function**.
2. Upload the pipe module file: `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py`
3. Configure an API key:
   - via valves (`API_KEY`), or
   - via `OPENROUTER_API_KEY` in the environment (deployment dependent).
4. Enable the pipe and select a model in Open WebUI.

Notes:

- Dependencies are installed by Open WebUI from the pipe manifest `requirements:` header.
- If you run multiple installations, use distinct pipe ids and review persistence/Redis namespace behavior (see [Persistence, Encryption & Storage](docs/persistence_encryption_and_storage.md)).

## Configuration

All runtime configuration is via valves.

- Reference: [Valves & Configuration Atlas](docs/valves_and_configuration_atlas.md)
- Common operator entry points: [Concurrency Controls & Resilience](docs/concurrency_controls_and_resilience.md), [Security & Encryption](docs/security_and_encryption.md)

## Telemetry and identifiers (optional)

- **Request identifiers for attribution**: valve-gated `user`, `session_id`, and `metadata.*` emission to OpenRouter (see [Request identifiers and abuse attribution](docs/request_identifiers_and_abuse_attribution.md)).
- **Cost snapshots**: optional Redis export of usage/cost payloads for downstream billing/monitoring (see [OpenRouter Integrations & Telemetry](docs/openrouter_integrations_and_telemetry.md)).
- **Encrypted session log archives**: optional encrypted zip archives per request for incident response (see [Encrypted session log storage (optional)](docs/session_log_storage.md)).

## Security notes (read before production)

- Remote downloads accept both `http://` and `https://` URLs; if you require HTTPS-only, enforce it via network egress policy.
- Artifact encryption is enabled when `ARTIFACT_ENCRYPTION_KEY` is set. If you store secrets as encrypted valve values, `WEBUI_SECRET_KEY` must be configured so Open WebUI can decrypt them at runtime.
- This pipe does not implement MCP server connectivity inside the pipe (to avoid bypassing Open WebUI’s tool server RBAC). If you need MCP tools, use an MCP→OpenAPI proxy/aggregator (for example MCPO / MetaMCP) and add it as an OpenAPI tool server in Open WebUI.
- This project alone does not guarantee GDPR/HIPAA/SOC2 compliance; use it as part of a broader operational and governance program.

Read: [Security & Encryption](docs/security_and_encryption.md) and [Production readiness report (OpenRouter Responses Pipe)](docs/production_readiness_report.md).

## Development and testing (contributors)

See [Testing, bootstrap, and operational playbook](docs/testing_bootstrap_and_operations.md).

## Acknowledgments

This project builds on the [OpenAI Responses Manifold](https://github.com/jrkropp/open-webui-developer-toolkit/) by jrkropp while adapting it for OpenRouter's ecosystem and adding the operational hardening needed for production deployments.

## License

MIT. See `LICENSE`.
