# openrouter responses manifold -- documentation index

**file:** `docs/documentation_index.md`
**scope:** navigation map for every deep-dive document that ships with the OpenRouter Responses pipe.

The `docs/` directory is the canonical reference for this project. It mirrors the structure of the pipe itself: model catalog intake, history reconstruction, multimodal transforms, tool orchestration, streaming, persistence, and operational guardrails. Every file below is AI- and human-friendly, written to stand alone while linking back to the code in `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py`.

Use this page to decide which document to open next.

---

## 1. developer and architecture guides

### 1.0 developer overview & architecture map
**file:** `docs/developer_guide_and_architecture.md`

* High-level tour of how the manifold is wired from the Open WebUI pipe shim through the streaming engine.
* Explains layering conventions, tracing utilities, request lifecycle, and dev loop expectations (tests, bootstrap, linting).
* Includes a "map" that correlates code regions (imports, valves, adapters, engine, utilities) with the doc set.

### 1.1 model catalog and routing intelligence
**file:** `docs/model_catalog_and_routing_intelligence.md`

* Documents the OpenRouter model registry loader, refresh cadence, capability detection, and routing helpers.
* Covers reasoning toggles, modality gates, web-search/MCP wiring, and how valves feed into catalog queries.

### 1.2 history reconstruction & context replay
**file:** `docs/history_reconstruction_and_context.md`

* Explains how Open WebUI messages become Responses `input[]` blocks, including developer/system instructions, persisted artifacts, and marker decoding.
* Details ULID markers, history compaction, reasoning/tool retention knobs, and how failures are surfaced to the UI.

---

## 2. modality & interface layers

### 2.0 multimodal intake pipeline
**file:** `docs/multimodal_ingestion_pipeline.md`

* Complete walkthrough of file/image/audio/video handling, SSRF protections, retries, buffering, and fallback storage identities.
* Includes flowcharts for `_to_input_image`, `_to_input_file`, `_to_input_audio`, and `_to_input_video` plus size-guard tables and error surfacing notes.

### 2.1 task models & housekeeping
**file:** `docs/task_models_and_housekeeping.md`

* Explains how Open WebUI “task” requests differ from user-facing chats, including the fast-path controller, retries, and reasoning overrides.
* Recommends ideal mini-tier models (GPT-4.1 Mini, GPT-5 Mini), empty system prompts, and no-tool configurations so chores stay cheap and predictable.
* Provides a troubleshooting table for common misconfigurations (verbose titles, 400s from unintended tools, runaway costs) and wiring steps in the Open WebUI admin panel.

### 2.1 tools, plugins, and extra integrations
**file:** `docs/tooling_and_integrations.md`

* Catalogs every tool source: Open WebUI registry, model-provided tools, filter-injected extras, MCP servers, and OpenRouter"s `web` search plugin.
* Documents strict schema enforcement, batching rules, loop ceilings, breaker telemetry, and how persisted tool outputs are referenced later.

### 2.2 streaming engine & emitters
**file:** `docs/streaming_pipeline_and_emitters.md`

* Dissects the SSE consumer/producer queues, worker pools, UTF-16 safety, reasoning/citation events, and completion finalizers.
* Shows how `STREAMING_*` valves and user overrides reshape latency vs. throughput, and how disconnects are handled without dropping artifacts.

## 3. durability & state

### 3.0 persistence, storage, and cleanup
**file:** `docs/persistence_encryption_and_storage.md`

* Deep dive on SQLAlchemy models, ULID markers, encryption, compression, Redis write-behind, cache invalidation, and cleanup workers.
* Explains how encryption keys impact table names, what happens during key rotation, and how Redis breakers fail open.

### 3.1 concurrency, breakers, and resilience
**file:** `docs/concurrency_controls_and_resilience.md`

* Details admission control (queues + semaphores), request-scoped ContextVars, session logging, breaker windows, and overload fallbacks.
* Includes diagrams for request lifecycle under success vs. failure scenarios, plus tuning guidance for each guardrail valve.

### 3.2 testing, bootstrap, and operational runbooks
**file:** `docs/testing_bootstrap_and_operations.md`

* Documents the pytest bootstrap plugin, local dev setup, CI guidance, recommended test suites, warmup probes, and production readiness checks.
* Summarizes alerting hooks, log patterns, and how to verify Redis + DB health on deploy.

---

## 4. reference materials

### 4.0 valves & configuration atlas
**file:** `docs/valves_and_configuration_atlas.md`

* Exhaustive listing of every system + user valve with defaults, ranges, and rationale.
* Structured tables that mirror the order of `Pipe.Valves` / `Pipe.UserValves` definitions so you can cross-reference quickly.

### 4.1 security & encryption guide
**file:** `docs/security_and_encryption.md`

* Comprehensive security guide covering encryption requirements, key rotation procedures, SSRF protection, secret management, and multi-tenant isolation.
* Includes production deployment checklists, incident response runbooks, and compliance guidance (GDPR, HIPAA, SOC 2).

### 4.2 production readiness audit
**file:** `docs/production_readiness_report.md`

* Expanded audit covering secrets, persistence guarantees, multimodal guardrails, concurrency controls, streaming, observability, and outstanding risks.
* Updated for the OpenRouter manifold; supersedes the legacy root-level document.

### 4.3 openrouter integrations & telemetry
**file:** `docs/openrouter_integrations_and_telemetry.md`

* Highlights the OpenRouter-specific behaviors (usage strings, catalog routing, automatic plugin wiring, CSS patches, multimodal guardrails) so you can quickly explain what differentiates this manifold from the OpenAI version.
* Documents 400 error templates with template variable reference and Handlebars conditional syntax.

### 4.4 error handling & user experience
**file:** `docs/error_handling_and_user_experience.md`

* Comprehensive guide to the error template system covering all exception types: network timeouts, connection failures, 5xx service errors, and internal exceptions.
* Includes rendered examples with realistic data, valve configuration, troubleshooting guide, operator runbook, and template customization patterns.
* Shows exactly what users see when errors occur and how operators correlate error IDs with backend logs.

---

## 5. root-level references

### 5.0 changelog
**file:** `CHANGELOG.md`

* Chronological, commit-by-commit history from the initial import forward, useful for auditing when specific valves, docs, or integrations landed.
* Each entry includes the date, author, SHA, and original commit message/body so you can trace context without digging through `git log`.

---

## 6. reading order suggestions

1. Start with `developer_guide_and_architecture.md` to understand how the manifold is layered.
2. If you are touching model selection or capability toggles, read `model_catalog_and_routing_intelligence.md` followed by `history_reconstruction_and_context.md`.
3. For multimodal or storage changes, pair `multimodal_ingestion_pipeline.md` with `persistence_encryption_and_storage.md`.
4. Tooling, streaming, or concurrency work should reference `tooling_and_integrations.md`, `streaming_pipeline_and_emitters.md`, and `concurrency_controls_and_resilience.md` respectively.
5. Always consult `valves_and_configuration_atlas.md` before adding or editing configuration knobs, and skim `testing_bootstrap_and_operations.md` before submitting PRs.

Each document is intentionally verbose--expect tables, callouts, and excerpts from the source. Use the navigation breadcrumbs at the top of each file to jump between related sections.
