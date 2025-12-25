# OpenRouter Responses Pipe for Open WebUI — Documentation

This documentation set covers installation, configuration, operations, and security for the Open WebUI pipe that targets the OpenRouter **Responses API**.

---

## Who this is for

### Everyday users
Use these docs if you want to install the pipe, select OpenRouter models in Open WebUI, and troubleshoot common errors.

### Enterprise admins / operators
Use these docs if you run Open WebUI for multiple users and need secure configuration, deployment guidance, operational controls, and incident-ready logging.

### Contributors
Use these docs if you are modifying the pipe, working on tests, or extending the integrations.

---

## Quick start (install and use)

### Prerequisites
- Open WebUI `0.6.28` or later (the pipe declares this requirement in its manifest header).
- An OpenRouter API key, provided either via the `OPENROUTER_API_KEY` environment variable or via the pipe’s valve configuration.

### Install (Open WebUI UI)
1. In Open WebUI, go to **Admin → Functions → New Function**.
2. Upload the pipe file: `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py`.
3. Save and enable the function.

### Dependencies
Open WebUI installs the Python dependencies declared in the function header (`requirements:`). This pipe declares (at least) `aiohttp`, `cryptography`, `fastapi`, `httpx`, `lz4`, `pydantic`, `pydantic_core`, `sqlalchemy`, `tenacity`, and `pyzipper`.

### Configure
- Open WebUI **Valves** are the configuration surface for this pipe. At minimum, configure the OpenRouter API key (or set `OPENROUTER_API_KEY` in the Open WebUI environment).
- For the authoritative list of valves (including verified defaults), see: [Valves & Configuration Atlas](valves_and_configuration_atlas.md).

### Use
After enabling the function, Open WebUI will expose OpenRouter models (imported from OpenRouter’s model catalog) in the model selector. Choose a model and chat normally.

---

## Common troubleshooting

### Models do not appear / no OpenRouter options
- Confirm the function is enabled in **Admin → Functions**.
- Confirm the pipe is configured with a valid API key (valve or `OPENROUTER_API_KEY` environment variable).
- If your deployment blocks outbound internet access, ensure it can reach the OpenRouter API endpoint configured by `BASE_URL`.
  - The model list is populated by refreshing the OpenRouter catalog; when the refresh fails and no cached catalog exists, Open WebUI will show no models until connectivity and credentials are fixed.

### Authentication errors (401/403) or “invalid API key”
- Validate the configured key and confirm it is being provided where you expect (valves vs environment variables).
- If you run Open WebUI behind a proxy/gateway, verify `BASE_URL` points to your gateway and that the gateway is correctly forwarding auth headers.

### Remote files or images fail to load
- Remote downloads are subject to SSRF filtering and size limits. Review your remote download settings in [Valves & Configuration Atlas](valves_and_configuration_atlas.md) and the deep-dive in [Multimodal Ingestion Pipeline](multimodal_ingestion_pipeline.md).

### Requests fail under load
- Review concurrency limits, queueing behavior, and breaker controls in [Concurrency Controls & Resilience](concurrency_controls_and_resilience.md).
- See [Error Handling & User Experience](error_handling_and_user_experience.md) for operator-facing troubleshooting and error template behavior.

---

## Where to go next

### Start here (most users and admins)
- [Valves & Configuration Atlas](valves_and_configuration_atlas.md) — authoritative configuration reference (defaults verified against code/tests).
- [Error Handling & User Experience](error_handling_and_user_experience.md) — what users see, how errors are structured, and how to troubleshoot.
- [OpenRouter Integrations & Telemetry](openrouter_integrations_and_telemetry.md) — identifiers, metadata, optional telemetry exports, and OpenRouter-facing headers.

### Security and compliance guidance (admins)
- [Security & Encryption](security_and_encryption.md) — key handling, SSRF controls, and hardening guidance.
- [Persistence, Encryption & Storage](persistence_encryption_and_storage.md) — what is persisted, how retention works, and operational considerations.
- [Request Identifiers & Abuse Attribution](request_identifiers_and_abuse_attribution.md) — multi-user identifiers and privacy guidance.
- [Session Log Storage](session_log_storage.md) — optional encrypted, per-request archives for incident response.

### Operations and performance (admins/operators)
- [Concurrency Controls & Resilience](concurrency_controls_and_resilience.md) — admission control, breaker behavior, and tuning guidance.
- [Streaming Pipeline & Emitters](streaming_pipeline_and_emitters.md) — streaming lifecycle, emitters, and UX/performance tradeoffs.
- [Testing, Bootstrap & Operations](testing_bootstrap_and_operations.md) — test harness, dev bootstrap, and operational runbooks.
- [Production Readiness Report](production_readiness_report.md) — an assessment-style document (treat as guidance, not a compliance guarantee).

### Deep technical references (contributors)
- [Developer Guide & Architecture](developer_guide_and_architecture.md)
- [Model Catalog & Routing Intelligence](model_catalog_and_routing_intelligence.md)
- [History Reconstruction & Context](history_reconstruction_and_context.md)
- [Multimodal Ingestion Pipeline](multimodal_ingestion_pipeline.md)
- [Task Models & Housekeeping](task_models_and_housekeeping.md)
- [Tooling & Integrations](tooling_and_integrations.md)
- `CHANGELOG.md` (repo root) — audit trail for changes.
