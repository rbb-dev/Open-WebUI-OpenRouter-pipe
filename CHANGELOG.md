# Changelog

All notable changes to this project will be documented in this file. The format roughly follows the [Keep a Changelog](https://keepachangelog.com/) convention and lists the most recent changes first. Dates use the commit timestamps available in git history.

## [Unreleased]
### Added
- Inline Open WebUI-hosted images directly into Responses payloads and auto-host remote `file_url` references so OpenRouter never depends on third-party storage (c809594, efbba1c, 7e158f7).
- OpenRouter-specific integrations overview covering usage telemetry, catalog routing, plugin wiring, and guardrails (`docs/openrouter_integrations_and_telemetry.md`).
- Customizable OpenRouter error template with auto-pruned placeholders plus regression tests and docs (`openrouter_responses_pipe.py`, `tests/test_openrouter_errors.py`).

### Changed
- Hardened remote download pipeline: asynchronous SSRF validation, Retry-After aware backoff, filtered retry conditions, and explicit valve to trust user-provided size limits (0245e0b, 7d3f71d, bf50d8c, 9b54ac4).
- Default artifact persistence to always attempt LZ4 compression when available (39396e2) and run model metadata lookups inside a threadpool to avoid blocking the event loop (bcaae30).
- Redis write-behind now requires Open WebUI multi-worker settings before enabling, preventing accidental single-worker misconfiguration (969ac44).
- OpenRouter error templating now moves moderation/model-limit copy into the public template with macros and middle-out-aware context limit values (openrouter_responses_pipe.py, tests/test_openrouter_errors.py).

### Fixed
- Streaming citations render reliably and no longer interleave with text deltas (7e70e86).
- Strict tool schema generation preserves optional fields, required lists, and trust in valve bounds (1f7206c, 1f353c5, 036146c, e2c8c45).
- User valve overrides (log level, reasoning toggles) honor inheritance semantics without clobbering defaults (a5768fb, 3a9bb06).
- Multimodal transformers always re-host uploads, handle audio edge cases, and carry request/user context so storage works outside the UI (8066068, 2d6835b, 6da4ae0).
- Streaming completion frames now include the last assistant snapshot so concurrent workers/usage updates can’t wipe the rendered reply in Open WebUI (openrouter_responses_pipe.py, tests/test_streaming_completion.py).
- OpenRouter 400 responses no longer bubble up as worker-crashing exceptions; `_run_streaming_loop` reports the templated error and exits cleanly (openrouter_responses_pipe.py).

### Documentation
- Rebuilt the documentation tree with professional names (`docs/documentation_index.md`) and lower-case ASCII, added an expanded production readiness report, valve atlas, and OpenRouter integration guide (multiple commits up to f57a356).
- README now references the new docs and highlights OpenRouter telemetry improvements (9532b3a and follow-ups).
- Documented Auto Context Trimming + message transforms and provided an `OPENROUTER_ERROR_TEMPLATE` example in `docs/openrouter_integrations_and_telemetry.md`; updated valve atlas accordingly.

## [1.0.5] - 2025-11-19
### Added
- Comprehensive multimodal input support with retry logic, fallback storage identity valves, and configurable size limits for downloads, base64 blobs, audio, and video (55a6b9f, c529866, efbba1c, 9c3ffde).
- Tunable streaming, tool, and Redis valves so operators can balance concurrency, latency, and resource usage (cb945ab, b5e28dd).
- Detailed documentation for valves, artifact persistence, and architecture (`docs/valves_reference.md`, `docs/MULTIMODAL_IMPLEMENTATION.md`, `docs/PRODUCTION_READINESS_AUDIT.md`).

### Changed
- Major overhaul of concurrency controls: request queue, shared semaphores, breaker windows, Redis warmup, and cancellation propagation to OpenRouter (fc76384, b5e28dd, bd29fd3).
- Modernized the codebase with Pydantic v2, improved type safety, and normalized logging output (ae67976, ee18006).
- Improved reasoning streaming UI by surfacing progress updates, semantic chunking, and throttled status emissions (191ee05, 5a6b46b, b6656ba).

### Fixed
- Security fixes across multimodal ingestion, SSRF validation, remote downloads, and reasoning/tool artifact replay (a2dafe8, 55a6b9f, 9c3ffde, 4c81877).
- Numerous test infrastructure updates so pytest passes in isolation, including bootstrap shims, Pydantic compatibility, and regression coverage (966343c, c990c28, 73ebfa5).

## [Initial release] - 2025-11-18
- Initial OpenRouter Responses pipe with streaming engine, persistence layer (encrypted artifacts + Redis write-behind), tool orchestration, and README aligned to the OpenRouter ecosystem (0cb59d9, 8f06828, early follow-up commits).
