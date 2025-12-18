# testing, bootstrap, and operational playbook

**file:** `docs/testing_bootstrap_and_operations.md`
**related source:** `pytest_bootstrap.py`, `tests/`, `PRODUCTION_READINESS_AUDIT.md`, startup helpers inside `open_webui_openrouter_pipe/open_webui_openrouter_pipe.py`

This document covers the local developer workflow (tests, linting, fixtures) and the operational checks you should run before/after deploying the pipe into Open WebUI.

> **Pre-deployment**: Before deploying to production, review the comprehensive [Production Readiness Report](production_readiness_report.md) for end-to-end security verification, compliance guidance, and deployment checklists.

> **Quick Navigation**: [📑 Index](documentation_index.md) | [🏗️ Architecture](developer_guide_and_architecture.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🔒 Security](security_and_encryption.md)

---

## 1. local testing workflow

> **Conventions:** All commands below run from the repository root (`Open-WebUI-OpenRouter-pipe/`). Paths like `.venv/bin/...` are relative to that root.

### 1.1 Python + virtualenv bootstrapping (Open WebUI requires Python 3.12)

```bash
# confirm python 3.12 is installed (install via pyenv/homebrew/apt if missing)
python3.12 --version

# create the project-local virtual environment
python3.12 -m venv .venv

# activate it for the current shell (prompt now shows "(.venv)")
source .venv/bin/activate
# (.venv) user@host:Open-WebUI-OpenRouter-pipe$

# run ad-hoc commands without activating the shell-wide venv
.venv/bin/python -V
.venv/bin/pip list
```

### 1.2 install dependencies inside the venv

```bash
# all commands run from repo root with the venv active (prompt shows .venv)
(.venv) pip install --upgrade pip setuptools wheel

# install Open WebUI so imports like open_webui.models.* resolve during tests.
# if you already have the Open WebUI repo locally, you can run
# (.venv) pip install -e ../open-webui
# instead of pulling from PyPI.
(.venv) pip install open_webui

# install everything listed in the pipe's requirements header
(.venv) pip install aiohttp cryptography fastapi httpx lz4 pydantic pydantic_core sqlalchemy tenacity

# test tooling
(.venv) pip install pytest pytest-asyncio

# without activating the venv, prefix commands with .venv/bin/
.venv/bin/pip install --upgrade pip setuptools wheel
```

To ensure `pytest` can import the bundle + bootstrap module, set `PYTHONPATH=.`:

```bash
(.venv) export PYTHONPATH=.
# per-command alternative:
PYTHONPATH=. .venv/bin/pytest tests/test_multimodal_inputs.py
```

### 1.3 pytest bootstrap & suites

1. **Pytest bootstrap** -- `pytest_bootstrap.py` preloads shims for Open WebUI modules (`open_webui.models.*`, `open_webui.config`, etc.), configures SQLAlchemy to use SQLite, and blocks real network calls. Always `import pytest_bootstrap` as the first line in new test files.
2. **Existing suites**:
   * `tests/test_multimodal_inputs.py` -- `_download_remote_url`, `_parse_data_url`, `_inline_internal_file_url`, and multimodal edge cases.
   * `tests/test_pipe_guards.py` -- queue admission, valve validation, breaker logic, Redis eligibility.
3. **Running tests** (from repo root):
   ```bash
   # inside the venv
   (.venv) pytest tests/test_multimodal_inputs.py
   (.venv) pytest tests/test_pipe_guards.py

   # without activating the venv
   PYTHONPATH=. .venv/bin/pytest tests/test_multimodal_inputs.py
   ```
4. **Adding coverage** -- When you modify a subsystem, add or extend tests nearby. Use `tests/conftest.py` fixtures for mocked `Request` objects, fake Redis, and Open WebUI shims to keep tests deterministic.

---

## 2. linting & formatting conventions

* Use `ruff` or `flake8` locally if desired, but the project intentionally avoids tooling that rewrites the single pipe file automatically. Keep edits focused.
* When editing docs, keep filenames lower-case (mirrors this `docs/` layout) and include file/line references when citing code.
* Comments should explain *why* a non-obvious block exists; avoid narrating simple assignments.

---

## 3. deployment checklist

| Step | Why |
| --- | --- |
| Confirm Open WebUI ≥ 0.6.28 | The pipe relies on the new Responses manifold interfaces introduced in 0.6.28. |
| Set `OPENROUTER_API_KEY` in OWUI or env | Without it, warmup probes keep retrying and requests fail with "API key missing". |
| Configure `WEBUI_SECRET_KEY` | Required so `EncryptedStr` can decrypt `ARTIFACT_ENCRYPTION_KEY` and other secret valves. |
| Decide on `ARTIFACT_ENCRYPTION_KEY` | Set it before first launch if you ever plan to encrypt persisted artifacts. Rotating later intentionally strands old rows. |
| Enable Redis when scaling out | Provide `REDIS_URL`, `WEBSOCKET_MANAGER=redis`, `WEBSOCKET_REDIS_URL`, and `UVICORN_WORKERS>1` so write-behind caching activates. |
| Assign unique pipe IDs when running more than one instance | Each pipe"s `id` influences both the SQL table name and Redis namespace (see `docs/persistence_encryption_and_storage.md`). Give every installation a distinct ID so artifacts never collide. |
| Set `FALLBACK_STORAGE_*` if defaults clash | Ensure the fallback upload account uses a valid email + role inside your Open WebUI deployment. |

---

## 4. smoke tests after install

1. **Catalog fetch** -- Open the model selector in Open WebUI. If it lists OpenRouter models with correct capability toggles (vision/search badges), the registry is healthy.
2. **Multimodal** -- Upload an image, send a prompt, and confirm the responder mentions "📥 downloading..." in the status tray. Regenerate to confirm the image is replayed without re-uploading.
3. **Tool call** -- Enable a simple tool (e.g., `web_search`), ask a question, and watch for tool prompts + outputs in the transcript. Verify subsequent turns replay the tool summaries via markers.
4. **Redis flush** (if enabled) -- Run two simultaneous chats, then check Redis for the `pending` key. After a minute it should be empty, confirming the flusher drained it.
5. **Warmup retry** -- Temporarily remove `OPENROUTER_API_KEY`, restart, and confirm requests are rejected with "Service unavailable due to startup issues". Re-set the key, request a chat, and verify the warmup probe succeeds (warning disappears).
6. **Cleanup worker** -- Temporarily lower `ARTIFACT_CLEANUP_DAYS` + `ARTIFACT_CLEANUP_INTERVAL_HOURS`, restart, and confirm logs show "Cleanup removed ... rows". Reset valves after the test.

---

## 5. observability & alerting

* **Logs** -- `SessionLogger` prefixes each line with the session ID so central log stores can slice per-user. Use `LOG_LEVEL` valves to capture DEBUG logs for a single user when triaging issues.
* **Status stream** -- Encourage operators to enable the Open WebUI status tray. The pipe emits detailed progress (downloads, tool execution, Redis fallbacks) so most incidents are visible without SSH.
* **Metrics (roadmap)** -- Future work includes emitting Prometheus-friendly counters for queue depth, Redis flush latency, and tool failure rates. Until then, scrape logs or add custom instrumentation inside `_emit_completion`.

---

## 6. incident response quick refs

| Symptom | Likely cause | Mitigation |
| --- | --- | --- |
| Continuous "warmup failed" logs | API key missing, DNS blocked, or OpenRouter unreachable. | Verify credentials + outbound network, then restart the pipe to retry the warmup task. |
| Users see "DB ops skipped..." | Database unavailable or migrations missing. | Check DB connectivity; breaker heals automatically once writes succeed. |
| Redis pending queue never drains | Flush lock stuck or DB writes failing. | Restart one worker to release the lock; inspect logs for DB errors; consider disabling Redis until resolved. |
| Attachments ignored | Selected model lacks `vision`/`file_input`, or valve caps were exceeded. | Pick a capable model or adjust `MAX_INPUT_IMAGES_PER_REQUEST` / size valves. |
| Tool loops stop at 10 iterations | `MAX_FUNCTION_CALL_LOOPS` hit. Either raise the valve (with caution) or improve the tool logic so fewer loops are needed. |

Document any new operational procedures here as you add features. The goal is for on-call engineers (or AI agents) to resolve incidents quickly without reading the entire pipe file.
