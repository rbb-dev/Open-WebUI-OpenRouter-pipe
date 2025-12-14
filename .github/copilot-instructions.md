# OpenRouter Responses Pipe - Copilot Instructions

## Big Picture Architecture
Single-file Open WebUI pipe (`open_webui_openrouter_pipe/open_webui_openrouter_pipe.py`) layers Completions → OpenRouter Responses API:
- **Registry** (`OpenRouterModelRegistry`): Fetches/caches `/models`, derives capabilities (vision/reasoning/tools).
- **Transforms** (`_transform_messages_to_input`): Multimodal (image/file/audio/video) → storage URLs + ULID artifact replay.
- **Pipe** core: Valves-driven queues/semaphors → tool orchestration → SSE streaming → per-pipe SQL persistence (Redis optional).
Data flow: OWUI request → bounded queue → per-req isolation (ContextVars) → OpenRouter → tool loops → emitters + cleanup.

Key decisions: Bounded everything (queues=50-500, breakers); async I/O + ThreadPool sync; valve-only config; docs mirror layers.

## Developer Workflows
- **Venv/Tests**: `source .venv/bin/activate && PYTHONPATH=. pytest tests/test_<focus>.py` (bootstrap stubs OWUI/DB/network).
- **Backups First**: Before edits: `cp <file> backups/<file>-$(date +%Y-%m-%d-%H-%M-%S)`.
- **Validate**: Post-edit: `get_errors` → fix loops ≤3x; `run_in_terminal` only for git/mv.
- **Debug**: `SessionLogger` (per-session ContextVars); status emitters before exceptions.

## Conventions & Patterns
- **Async/Sync**: Async (HTTP/Redis/SQL); ThreadPool (DB/Fernet/LZ4).
- **Config**: `Pipe.Valves`/`UserValves` only—no env reads. Merge user overrides post-decrypt.
- **Queues**: Preserve bounds (`_request_queue maxsize=500`, tools=50); semaphores global/per-req.
- **Errors**: `_emit_templated_error` + templates; `OpenRouterAPIError` for 400s.
- **Persistence**: Per-pipe tables (`pipe_<id>_artifacts`); ULID markers; encrypt via `WEBUI_SECRET_KEY`+valve.
- **Tools**: FIFO queues, breakers per-user/type; strict schemas; MCP/web_search auto.
- **No generics**: Project-specific (e.g., replay artifacts, catalog routing)—cite `docs/<subsystem>.md`.

## Integrations & Boundaries
- **OWUI**: DB (`open_webui.internal.db`), models/files/users, `__event_emitter__`, valves.
- **OpenRouter**: `/models` + `/responses`; retry w/ `tenacity` + Retry-After.
- **Redis**: Auto-detect multi-worker; write-behind + pub/sub flushes.
- **Ext**: `.external/open-webui` (read-only); `.external/openrouter_docs`; live models JSON.

## Key Files/Dirs
| Purpose | Paths |
|---------|-------|
| Core | `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py` (sections: imports→Pipe→helpers) |
| Tests | `tests/*` (multimodal/guards/errors); `conftest.py` fixtures |
| Docs | `docs/*.md` (valves atlas, multimodal pipeline, concurrency) |
| Config | `pyproject.toml`, `pyrightconfig.json`, `pytest.ini` |
| Refs | `AGENTS.md` (strict rules), `README.md` (user-facing) |

**Before changes**: Read `docs/developer_guide_and_architecture.md` + relevant doc; search `grep_search` for patterns; backup; test surgically.

Feedback on unclear/incomplete sections?