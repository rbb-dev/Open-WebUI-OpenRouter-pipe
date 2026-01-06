# Testing, bootstrap, and operational playbook

**Scope:** Local developer workflow (tests/lint) and operator checks before/after deploying the pipe into Open WebUI.

> **Pre-deployment**: Review [Production readiness report (OpenRouter Responses Pipe)](production_readiness_report.md) for environment hardening, security guidance, and deployment considerations.

> **Quick Navigation**: [📘 Docs Home](README.md) | [⚙️ Configuration](valves_and_configuration_atlas.md) | [🔒 Security](security_and_encryption.md) | [🧯 Errors](error_handling_and_user_experience.md)

---

## Local development and testing

Conventions:

- Commands run from the repository root.
- Prefer prefixing Python tooling with `PYTHONPATH=.` so imports resolve consistently when running from source.

### Python + virtualenv bootstrapping (Python 3.11+)

This project declares `requires-python = ">=3.11"` in `pyproject.toml`. Use Python 3.11+ for local development and tests (Python 3.11 matches official Open WebUI Docker images).

```bash
python3.11 --version  # or python3.12
python3.11 -m venv .venv  # or python3.12
source .venv/bin/activate
```

### Recreate the full dev venv (recommended)

To reproduce the maintained developer environment (including `open-webui`, this repo installed editable, and common test/lint tooling), use:

```bash
# From the repo root. Recreates `.venv` from scratch.
FORCE_RECREATE=1 bash scripts/repro_venv.sh
```

Notes:

- **Python selection**: `PYTHON_BIN=python3.11 FORCE_RECREATE=1 bash scripts/repro_venv.sh` (or `python3.12`)
- **Alternate venv dir**: `VENV_DIR=.venv2 FORCE_RECREATE=1 bash scripts/repro_venv.sh`
- **Slow installs** (Open WebUI can take a long time): the script sets long pip timeouts/retries; override if needed:
  `PIP_DEFAULT_TIMEOUT=1200 PIP_RETRIES=10 FORCE_RECREATE=1 bash scripts/repro_venv.sh`

### Install dependencies

Install the package in editable mode (this installs runtime dependencies from `pyproject.toml`):

```bash
(.venv) pip install --upgrade pip setuptools wheel
(.venv) pip install -e .
(.venv) pip install pytest pytest-asyncio
```

Optional:

- Install `open_webui` in the venv if you are running integration-style experiments locally. Unit tests in this repo do not require a full Open WebUI installation.
- Install `ruff` or `flake8` if you use a linter locally.

### Pytest bootstrap behavior

`pytest` is configured via `pytest.ini` to load `open_webui_openrouter_pipe/pytest_bootstrap.py` before test collection. This bootstrap:

- Forces `tempfile` to use `/tmp` on WSL/Windows to avoid file capture edge cases.
- Disables global/system pytest plugin auto-loading to keep collection deterministic.

You do not need to import any bootstrap module in individual test files.

### Running tests

Run a single file first, then the full suite:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_multimodal_inputs.py -q
PYTHONPATH=. .venv/bin/pytest tests -q
```

### Test suite map (high level)

The suite is organized by subsystem. Common entry points:

- `tests/test_multimodal_inputs.py`: multimodal URL/data handling and SSRF-related input guards.
- `tests/test_request_identifiers.py`: `SEND_*` valves and OpenRouter identifier mapping.
- `tests/test_session_log_storage.py`: encrypted session log storage skip rules and archive behavior.
- `tests/test_tool_schema.py`: strict tool schema transformations.
- `tests/test_direct_tool_servers.py`: Open WebUI Direct Tool Servers (Socket.IO `execute:tool`) wiring.
- `tests/test_transform_messages.py`: history reconstruction/marker replay behavior.
- `tests/test_pipe_guards.py`: admission controls, breakers, and runtime guards.

---

## Linting and formatting conventions

- Use `ruff` or `flake8` locally if desired, but avoid large automated rewrites.
- Keep changes focused and reviewable.

---

## Deployment checklist (operators)

| Step | Why |
| --- | --- |
| Confirm Open WebUI ≥ 0.6.28 | The pipe manifest requires 0.6.28. |
| Set `OPENROUTER_API_KEY` (valve or env) | Required for provider requests. |
| Configure `WEBUI_SECRET_KEY` | Recommended so Open WebUI can encrypt/decrypt secret valve values stored via `EncryptedStr`. |
| Decide on `ARTIFACT_ENCRYPTION_KEY` | Set before first launch if you plan to encrypt persisted artifacts; rotating later creates a new table and strands old rows. |
| Enable Redis when scaling out | Provide `REDIS_URL`, `WEBSOCKET_MANAGER=redis`, `WEBSOCKET_REDIS_URL`, and `UVICORN_WORKERS>1` to activate multi-worker cache behavior. |
| Assign unique pipe IDs for multiple installs | Pipe id influences SQL table names and Redis namespaces; keep them distinct (see [Persistence, Encryption & Storage](persistence_encryption_and_storage.md)). |
| Set `FALLBACK_STORAGE_*` if defaults clash | Ensure fallback uploads map to a valid Open WebUI account in your deployment. |

---

## Smoke tests after install

1. **Basic chat**: send a short prompt; confirm a normal completion returns successfully.
2. **Catalog/registry**: confirm the model list is populated and models resolve to OpenRouter ids as expected.
3. **Multimodal**: upload an image/file and confirm the request succeeds on a model that supports the capability (see [Multimodal Intake Pipeline](multimodal_ingestion_pipeline.md)).
4. **Tool calling**: enable a simple Open WebUI function tool and confirm the model can request and receive tool outputs (see [Tools, plugins, and integrations](tooling_and_integrations.md)).
5. **Persistence/replay** (if enabled): run two turns that include tool calls and confirm the second turn can reference persisted outputs (see [History Reconstruction & Context Replay](history_reconstruction_and_context.md)).
6. **Session logs** (if enabled): confirm that encrypted `.zip` archives appear under `SESSION_LOG_DIR` for requests that have all required IDs (see [Encrypted session log storage (optional)](session_log_storage.md)).

---

## Observability and incident response

Operator tools you can use immediately:

- **Backend logs**: use `LOG_LEVEL` to control verbosity and correlate with `session_id`/`user_id` (see [Request identifiers and abuse attribution](request_identifiers_and_abuse_attribution.md)).
- **Encrypted session logs**: enable `SESSION_LOG_STORE_ENABLED` for a durable per-request log bundle during incident response (see [Encrypted session log storage (optional)](session_log_storage.md)).
- **User-visible error templates**: tune the UI-facing templates for provider errors and timeouts (see [Error Handling & User Experience](error_handling_and_user_experience.md)).

---

## Incident response quick refs

| Symptom | Likely cause | Mitigation |
| --- | --- | --- |
| Repeated startup/warmup failures | API key missing, DNS blocked, or provider unreachable. | Verify credentials and outbound network; inspect backend logs for the root cause; restart to re-run startup checks. |
| Users see DB/persistence warnings | Database unavailable or migrations missing. | Check DB connectivity; breakers heal automatically once writes succeed. |
| Redis queue never drains | Flush lock stuck or DB writes failing. | Inspect logs for DB errors; restart one worker to release locks; consider disabling Redis until resolved. |
| Attachments ignored | Selected model lacks the capability, or size/count valves were exceeded. | Pick a capable model or adjust relevant multimodal valves. |
| Tool loops stop early | `MAX_FUNCTION_CALL_LOOPS` reached. | Reduce tool loop requirements or raise the valve with caution. |
