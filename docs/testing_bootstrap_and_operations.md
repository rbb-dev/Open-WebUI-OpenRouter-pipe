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

The `test` extra does not install Open WebUI. Install it into the same environment (`pip install open-webui`, as `scripts/repro_venv.sh` does): most tests stand in for Open WebUI, but some read the source files of an installed copy, as the next section describes.

Optional:

- Install `ruff` or `flake8` if you use a linter locally.

### Open WebUI's own code in the tests

Some tests read Open WebUI's own source files rather than a transcription of them, so they
follow whichever Open WebUI release the environment holds. They parse the files and do not
import Open WebUI, but the files have to be on disk. One helper, `open_webui_package_dir()` in
`tests/package_sources.py`, finds them in two steps: the `open-webui` package installed in the
environment, then, where no such package is installed, a top-level `open_webui` folder that Python
finds on `sys.path` (set `PYTHONPATH` to the folder that contains it). Open WebUI's 0.12.4 Docker
image has that layout: its code is in `/app/backend/open_webui`, with no package metadata, so
`/app/backend` is the folder to put on `PYTHONPATH` there. When neither is found, each test that
needs the files fails with one message that names both places it looked and what to change. CI
installs the package with the other dependencies and pins no version.

No test reads a checkout of Open WebUI's repository: there is nothing to clone into
`.external/` for the suite, and CI does not fetch one.

### Pytest bootstrap behavior

`pytest` is configured via `pytest.ini` to load `open_webui_openrouter_pipe/pytest_bootstrap.py` before test collection. This bootstrap:

- Forces `tempfile` to use `/tmp` on WSL/Windows to avoid file capture edge cases.
- Disables global/system pytest plugin auto-loading to keep collection deterministic.

You do not need to import any bootstrap module in individual test files.

`pytest.ini` also sets `timeout = 60` with `timeout_method = thread`, so a hung test is failed and named on every path — including a hand-typed `pytest tests/`, which bypasses `scripts/gate.sh` entirely. The value is the repository's own CI budget, and CI's `--timeout=60 --timeout-method=thread` still overrides it, as does any other `--timeout` on the command line. A `@pytest.mark.timeout(...)` marker wins over both.

A probe subprocess that imports the real `open_webui` must be given its own `DATA_DIR`: the import builds a chromadb store under it, so a probe that inherits the ambient one races every concurrent collection for the same directory. The assignment must be **unconditional** — `os.environ['DATA_DIR'] = ...`, not `setdefault`. The parent process exports `DATA_DIR` and `tests/owui_stubs.py` sets it before any test runs, so a child always inherits a set value and a `setdefault` there is a line that reads compliant and does nothing.

Give it from a fixture rather than at module import — a process-wide timeout does not cover a module-import probe, so a probe that costs tens of seconds there runs entirely outside the budget CI enforces on tests.

A test module may not construct a `Pipe` at import time; `Pipe.id` is a class attribute and needs no instance. `Pipe.__init__` wires the package logger, claims the id in `_LifecycleRegistry` and starts the session-log machinery, so an import-time `Pipe()` decides the logger baseline of every later test in the same worker in an order the collector chooses.

That budget also stops at the end of the run. The timeout cannot cover interpreter shutdown, where `concurrent.futures.thread._python_exit` joins every `ThreadPoolExecutor` worker with `t.join()` and the alarm has already been reset — so a test that leaves a non-daemon thread running hangs the process instead of failing a test, and the symptom is a wall rather than a name. `tests/conftest.py` therefore names those threads separately and fails the session with the node ids, which is why a leak is reported against the test that caused it rather than against whichever one ran next.

### Running tests

Run a single file first, then the full suite. Both inherit the timeout above; a caller does not need to supply one.

```bash
PYTHONPATH=. .venv/bin/pytest tests/test_multimodal.py -q
PYTHONPATH=. .venv/bin/pytest tests -q
```

`scripts/gate.sh batch` and `release` hold every package-suite test to two seconds for its call: a test over that is re-run alone, and fails the gate only when it is still over. Fixture setup is not limited. Work that costs seconds and serves several tests -- a subprocess, an Open WebUI load, a bundle build -- belongs in one module-scoped fixture, which the gate lists with its time; a wait belongs on the simulated clock, which a test file or class asks for with `pytest.mark.jumping_clock` (`tests/conftest.py` then runs its async tests on an event loop that skips ahead to its next timer instead of waiting for it).

When a test fakes a *dashboard worker other than this one*, derive that worker's pid
from `os.getpid()` (offset it, e.g. `os.getpid() + 1_000_003`), never a literal such as
`111`. A literal can collide with a real pytest worker's own pid, which suppresses the
self-append in the publisher and turns a green suite red on a loaded machine rather than
on a change.

### Test suite map (high level)

The suite is organized by subsystem. Common entry points:

- `tests/test_multimodal.py`: multimodal URL/data handling and SSRF-related input guards (HTTPS-only defaults; HTTP allowlist coverage).
- `tests/test_request_identifiers.py`: `SEND_*` valves and OpenRouter identifier mapping.
- `tests/test_session_logs.py`: encrypted session log storage skip rules and archive behavior.
- `tests/test_tools.py`: tool execution, including the strict tool schema transformations.
- `tests/test_direct_tool_servers.py`: Open WebUI Direct Tool Servers (Socket.IO `execute:tool`) wiring.
- `tests/test_transform_messages.py`: history reconstruction/marker replay behavior.
- `tests/test_concurrency_limits.py`, `tests/test_circuit_breaker.py`: admission controls (the request queue and semaphores) and the request, tool and database breakers.

---

## Linting and formatting conventions

- Use `ruff` or `flake8` locally if desired, but avoid large automated rewrites.
- Keep changes focused and reviewable.

---

## Deployment checklist (operators)

| Step | Why |
| --- | --- |
| Confirm Open WebUI ≥ 0.12.0 | The pipe manifest requires 0.12.0. |
| Set `OPENROUTER_API_KEY` (valve or env) | Required for provider requests. |
| Configure `WEBUI_SECRET_KEY` | Recommended so Open WebUI can encrypt/decrypt secret valve values stored via `EncryptedStr`. |
| Decide on `ARTIFACT_ENCRYPTION_KEY` | Set before first launch if you plan to encrypt persisted artifacts; rotating later creates a new table and strands old rows, and a write already inside the cipher build when the rotation lands is dropped and logged with its artifact kind. |
| Enable Redis when scaling out | Provide `REDIS_URL`, `WEBSOCKET_MANAGER=redis`, `WEBSOCKET_REDIS_URL`, and `UVICORN_WORKERS>1` to activate multi-worker cache behavior. The same settings extend to the client choice: `REDIS_SENTINEL_HOSTS` (with `REDIS_SENTINEL_PORT`) selects a Sentinel client built on Open WebUI's own URL parsing, and `REDIS_CLUSTER=true` a cluster client built by the Redis library; with sentinel hosts set, a `rediss://` URL is refused, because that path keeps no TLS flag and the credentials would cross in plain text. |
| Assign unique pipe IDs for multiple installs | The pipe id is the identity everything per-install keys on: SQL table names and Redis namespaces (see [Persistence, Encryption & Storage](persistence_encryption_and_storage.md)), and now the dashboard too. Two copies with distinct ids each authorise, emit into and clear their own dashboard; two copies sharing one id share one pipe id, so keep them distinct. |
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
