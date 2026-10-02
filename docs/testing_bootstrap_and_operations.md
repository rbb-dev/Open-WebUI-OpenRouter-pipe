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
(.venv) pip install -e ".[test]"
```

Optional:

- Install `open_webui` in the venv if you are running integration-style experiments locally. Unit tests in this repo do not require a full Open WebUI installation.
- Install `ruff` or `flake8` if you use a linter locally.

### Open WebUI's pinned source (`.external/`)

Some tests read Open WebUI's own code rather than a transcription of it, and they read it
from `.external/open-webui`, which is gitignored. CI checks the pinned tag into that path
before the suite runs:

```bash
git clone --depth 1 --branch v0.11.4 https://github.com/open-webui/open-webui .external/open-webui
```

The pin is `v0.11.4`, the same version the test extra installs from PyPI. The clone is
shallow and CI asks for `backend` and `src` only -- the frontend is not optional, because
`tests/test_browser_tool_failures_count.py` reads `src/lib/apis/index.ts`.

Without the tree those readers **skip**, naming the missing path, rather than failing:
`tests/vetting_helpers.py::owui_source` is the one place that says so, and
`tests/test_owui_source_guards.py` holds every reader of `.external/` to declaring it, so a
checkout that has not fetched the pin is a smaller suite and never a different one. No
reader does its work at module scope either: a module-level read is a *collection* error,
which interrupts the run before any test executes.

Do not `pip download open-webui==0.11.4` and unzip it instead. A wheel carries no
`src/lib`, so the frontend readers would still skip while the fix looked complete.

### Pytest bootstrap behavior

`pytest` is configured via `pytest.ini` to load `open_webui_openrouter_pipe/pytest_bootstrap.py` before test collection. This bootstrap:

- Forces `tempfile` to use `/tmp` on WSL/Windows to avoid file capture edge cases.
- Disables global/system pytest plugin auto-loading to keep collection deterministic.

You do not need to import any bootstrap module in individual test files.

A probe subprocess that imports the real `open_webui` must be given its own `DATA_DIR`: the import builds a chromadb store under it, so a probe that inherits the ambient one races every concurrent collection for the same directory. The assignment must be **unconditional** — `os.environ['DATA_DIR'] = ...`, not `setdefault`. The parent process exports `DATA_DIR` and `tests/owui_stubs.py` sets it before any test runs, so a child always inherits a set value and a `setdefault` there is a line that reads compliant and does nothing.

Give it from a fixture rather than at module import — a process-wide timeout does not cover a module-import probe, so a probe that costs tens of seconds there runs entirely outside the budget CI enforces on tests. `pytest.ini` sets no `--timeout` of its own; the only per-test ceiling in this repository is `--timeout=60` in `.github/workflows/verify.yml`, so "the budget the suite enforces" describes a CI run and not a local or agent run. `tests/test_a_test_module_starts_no_probe_at_import.py` is the census that keeps a module-scope probe from being added, and `tests/test_a_probe_script_path_is_not_shared_across_workers.py` is the one that keeps a probe's *script* off a path another worker can name.

### Running tests

Run a single file first, then the full suite:

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_multimodal_inputs.py -q
PYTHONPATH=. .venv/bin/pytest tests -q
```

When a test fakes a *dashboard worker other than this one*, derive that worker's pid
from `os.getpid()` (offset it, e.g. `os.getpid() + 1_000_003`), never a literal such as
`111`. A literal can collide with a real pytest worker's own pid, which suppresses the
self-append in the publisher and turns a green suite red on a loaded machine rather than
on a change.

### Test suite map (high level)

The suite is organized by subsystem. Common entry points:

- `tests/test_multimodal_inputs.py`: multimodal URL/data handling and SSRF-related input guards (HTTPS-only defaults; HTTP allowlist coverage).
- `tests/test_request_identifiers.py`: `SEND_*` valves and OpenRouter identifier mapping.
- `tests/test_session_log_storage.py`: encrypted session log storage skip rules and archive behavior.
- `tests/test_tool_schema.py`: strict tool schema transformations.
- `tests/test_direct_tool_servers.py`: Open WebUI Direct Tool Servers (Socket.IO `execute:tool`) wiring.
- `tests/test_transform_messages.py`: history reconstruction/marker replay behavior.
- `tests/test_pipe_guards.py`: admission controls, breakers, and runtime guards.
- `tests/test_a_free_form_items_is_not_sealed_shut.py`: worked example of the probe-subprocess convention above — a real `open_webui` import, run from a session fixture over its own `TemporaryDirectory` under a file-scoped `pytest.mark.timeout`.
- `tests/test_dashboard_socket_isolation.py`: worked example of the per-test reset convention — arms `dashboard_socket._get_pipe`/`._registered`/`._resync` and `http_routes._routes_get_pipe` in one test and reads them in the next, so conftest's `_reset_dashboard_socket_state` has a witness.
- `tests/test_module_state_census.py`: the inventory of every module-level container the package writes, and why each is either reset per test or exempt — the guard that makes a new container a reviewed row rather than a silent leak.
- `tests/test_channel_error_cards.py`, `tests/test_a_channel_card_carries_nobody_elsses_identity.py`: a second instance of the same convention, obeying the same rule. Each writes its driver scripts into a per-process `tempfile.mkdtemp` directory, and hands the child a `DATA_DIR` of its own through `own_data_dir()` in `tests/conftest.py` — once in the child's `env` and once as an argv the prelude hard-assigns before its first `open_webui` import, so neither guard is the one a later simplification can quietly drop. The census is `tests/test_a_probe_script_path_is_not_shared_across_workers.py`.
- `tests/test_a_test_module_starts_no_probe_at_import.py`: the census for the convention itself — no collected test module may start a subprocess, await an event loop, sleep, or import the real `open_webui` at module scope.
- `tests/test_a_test_module_does_not_leave_a_stub_module_rebound.py`: no test may rebind an attribute of an `open_webui` module without a `monkeypatch` or a `finally` that restores it.
- `tests/test_a_process_wide_record_is_reset_between_tests.py`: every module-level container that one function writes and another reads to decide something is cleared between tests.

---

## Linting and formatting conventions

- Use `ruff` or `flake8` locally if desired, but avoid large automated rewrites.
- Keep changes focused and reviewable.

---

## Deployment checklist (operators)

| Step | Why |
| --- | --- |
| Confirm Open WebUI ≥ 0.11.4 | The pipe manifest requires 0.11.4. |
| Set `OPENROUTER_API_KEY` (valve or env) | Required for provider requests. |
| Configure `WEBUI_SECRET_KEY` | Recommended so Open WebUI can encrypt/decrypt secret valve values stored via `EncryptedStr`. |
| Decide on `ARTIFACT_ENCRYPTION_KEY` | Set before first launch if you plan to encrypt persisted artifacts; rotating later creates a new table and strands old rows, and a write already inside the cipher build when the rotation lands is dropped and logged with its artifact kind. |
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
- **Encrypted session logs**: enable `SESSION_LOG_STORE_ENABLED` for a durable per-request log bundle during incident response (see [Encrypted session log storage (optional)](session_log_storage.md)). API calls, which carry no chat or message id, are covered too while `SESSION_LOG_ARCHIVE_API_CALLS` is on, and land under `<user>/api/api-<request_id>.zip`.
- **User-visible error templates**: tune the UI-facing templates for provider errors and timeouts (see [Error Handling & User Experience](error_handling_and_user_experience.md)).

---

## Incident response quick refs

| Symptom | Likely cause | Mitigation |
| --- | --- | --- |
| Repeated startup/warmup failures | API key missing, DNS blocked, or provider unreachable. | Verify credentials and outbound network; inspect backend logs for the root cause; restart to re-run startup checks. |
| Users see DB/persistence warnings | Database unavailable or migrations missing; or artifact persistence disabled because the artifact table could not be reconciled. | A `WARNING` naming the columns it could not add or reconcile means the schema is stale, not the connection: restart the pipe after the columns can be added (see [Persistence, encryption and storage](persistence_encryption_and_storage.md)), and the artifacts already in the table are preserved. Otherwise check DB connectivity. While the persistence breaker is open the pipe attempts no database work, so it reopens only when the failures age out of `BREAKER_WINDOW_SECONDS`; a successful read or write then clears the count. |
| Redis queue never drains | Flush lock stuck or DB writes failing. | Inspect logs for DB errors; restart one worker to release locks; consider disabling Redis until resolved. |
| Attachments ignored | Selected model lacks the capability, or size/count valves were exceeded. | Pick a capable model or adjust relevant multimodal valves. |
| Tool loops stop early | `MAX_FUNCTION_CALL_LOOPS` reached (Pipeline mode only). The model receives stub responses for pending calls and gets a synthesis turn. With tool cards on, the skipped calls appear in the transcript as failed call cards rather than vanishing, which is how a cut-off round is told apart from one that returned nothing. | Raise the valve if the model needs more rounds, or simplify the request. |
