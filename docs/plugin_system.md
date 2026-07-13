# Plugin System — Developer Guide

> Technical reference for building plugins that extend the OpenRouter Responses API pipe.

> **Status:** The plugin framework is active. `on_models` may be sync or async — an async implementation can `await` (e.g. OWUI's async Models API), and a plain `def on_models` still works via an awaitable check.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Shared Plugin Infrastructure](#shared-plugin-infrastructure)
3. [Plugin Lifecycle](#plugin-lifecycle)
4. [Understanding Hook Points](#understanding-hook-points)
5. [Creating a Plugin](#creating-a-plugin)
6. [Plugin-Exported Valves](#plugin-exported-valves)
7. [Hook Reference](#hook-reference)
   - [on_init](#on_init)
   - [on_models](#on_models)
   - [on_request](#on_request)
   - [on_request_transform](#on_request_transform)
   - [on_emitter_wrap](#on_emitter_wrap)
   - [on_tool_result](#on_tool_result)
   - [on_request_retry](#on_request_retry)
   - [on_generation_complete](#on_generation_complete)
   - [on_shutdown](#on_shutdown)
8. [Hook Dispatch Semantics](#hook-dispatch-semantics)
9. [Priority System](#priority-system)
10. [PluginContext — Accessing Pipe Internals](#plugincontext--accessing-pipe-internals)
11. [Streaming Compatibility](#streaming-compatibility)
12. [Error Isolation](#error-isolation)
13. [Bundle Compatibility](#bundle-compatibility)
14. [Complete Example: Counter Plugin](#complete-example-counter-plugin)
15. [Pipe Internals Reference](#pipe-internals-reference)
16. [Testing Plugins](#testing-plugins)
17. [See Also](#see-also)

---

## Architecture Overview

The plugin system consists of three layers:

```
┌─────────────────────────────────────────────────┐
│  Pipe (pipe.py)                                 │
│  Dispatches hooks at key lifecycle points       │
├─────────────────────────────────────────────────┤
│  PluginRegistry (plugins/registry.py)           │
│  Manages registration, init, priority sorting   │
├─────────────────────────────────────────────────┤
│  PluginBase (plugins/base.py)                   │
│  Base class plugins inherit from                │
│  PluginContext — gateway to pipe internals       │
└─────────────────────────────────────────────────┘
```

**Key files:**

| File | Purpose |
|------|---------|
| `plugins/base.py` | `PluginBase` class and `PluginContext` |
| `plugins/registry.py` | `PluginRegistry` — registration, init, dispatch |
| `plugins/_utils.py` | Shared utilities: `extract_task_name()`, `extract_user_message()` |
| `plugins/__init__.py` | Auto-discovery via `pkgutil` + explicit imports |

---

## Shared Plugin Infrastructure

`plugins/_utils.py` provides reusable building blocks for virtual-model plugins that intercept OWUI chat requests:

- `extract_task_name(task)` → `str` — normalize OWUI background-task metadata (title/tags/emoji/follow-ups) whether it arrives as a string or a dict.
- `extract_user_message(body)` → `str` — pull the last user message text from a chat completions body, handling both plain and multimodal content.

Both helpers are **synchronous, side-effect-free pure functions** — no I/O, safe to call from any hook (sync or async):

- `extract_task_name(task)` returns a stripped name. For a dict it reads the `type`, `task`, then `name` keys in that order; it returns `""` for `None`, a missing key, or any unrecognized shape. A **non-empty** result marks the call as an OWUI background task rather than a user turn.
- `extract_user_message(body)` scans `body["messages"]` newest-first for the last `role == "user"` entry and returns its stripped text, unwrapping a multimodal content list to the first `type == "text"` part. It returns `""` when `messages` is absent or not a list, or when no user text is found.

```python
from .._utils import extract_task_name, extract_user_message


async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
    """Route a virtual model: answer OWUI background tasks quietly, echo prompts."""
    if str(body.get("model", "")) != "echo-bot":
        return None  # not our virtual model — let the request proceed

    if extract_task_name(task):
        # Title/tags/emoji/follow-up generation — return an empty payload
        return self.ctx.build_response(model="echo-bot", content="")

    prompt = extract_user_message(body)  # "" if no user text is present
    return self.ctx.build_response(model="echo-bot", content=f"You said: {prompt}")
```

### Reusing Open WebUI's own infrastructure

The framework itself ships no HTTP or Socket.IO helpers, but through `ctx.pipe` a plugin can reuse Open WebUI's existing infrastructure directly:

- **Live server→browser push:** register a custom handler on OWUI's authenticated Socket.IO server (`/ws/socket.io`) and emit to rooms — the same pattern OWUI's own `realtime:*` features use — instead of standing up a custom HTTP endpoint. Register at module import (idempotent; `sio.on` replaces rather than stacks), gate room joins on the resource's access by reusing OWUI's own decision (e.g. `check_model_access`) so grants and revocations are honored, and inline the Socket.IO browser client into your embed (OWUI serves none).
- **Authenticated action routes:** expose a FastAPI route that reuses OWUI's token decode plus user lookup for header-only bearer auth (CSRF-safe regardless of cookie settings) and reuses OWUI's model access decision for authorization — no bespoke auth logic.

See [plugins_pipe_dashboard_internals.md](plugins_pipe_dashboard_internals.md) for the reference plugin that implements both.

---

## Plugin Lifecycle

### 1. Import Time — Registration

When the pipe module loads, all plugin submodules under `plugins/` are auto-imported. The `@PluginRegistry.register` decorator fires for each plugin class, adding it to a class-level list:

```python
# plugins/registry.py
class PluginRegistry:
    _plugin_classes: list[type[PluginBase]] = []

    @classmethod
    def register(cls, plugin_class: type[PluginBase]) -> type[PluginBase]:
        if plugin_class not in cls._plugin_classes:
            cls._plugin_classes.append(plugin_class)
        return plugin_class
```

### 2. First Use — Initialization

The registry is lazily created on first access via `_ensure_plugin_registry()`:

```python
# pipe.py
def _ensure_plugin_registry(self) -> PluginRegistry:
    if self._plugin_registry is None:
        from .plugins.registry import PluginRegistry
        self._plugin_registry = PluginRegistry()
        self._plugin_registry.init_plugins(self)
    return self._plugin_registry
```

> **Master switch:** The `ENABLE_PLUGIN_SYSTEM` valve (default: `False`) controls whether plugins run. It is **not** checked inside `_ensure_plugin_registry()` — the first call to that method always builds the registry and runs `init_plugins()`. Instead, every dispatch site guards on the valve first: the inline hooks (`on_models`, `on_emitter_wrap`, `on_request`, `on_request_transform`) check `ENABLE_PLUGIN_SYSTEM` before touching the registry, and the observer hooks route through `_dispatch_plugin_event()`, which returns immediately when the valve is `False` and deliberately does **not** force registry creation. So when the valve is `False`, no hook ever fires and the registry is typically never even built — a zero-overhead no-op. Enable it in the pipe's valve settings to activate plugins.

`init_plugins()` is idempotent — calling it twice is a no-op. On first call, it does the following for each registered class:
1. Creates a per-plugin `PluginContext(pipe=pipe, logger=plugin_logger)` with a logger named `open_webui_openrouter_pipe.plugins.registry.<plugin_id>`
2. Instantiates the plugin: `instance = cls()`
3. Calls `instance.on_init(ctx)` — only adds the plugin to the active list on success. Init failures are logged at WARNING level.
4. Builds per-hook subscriber lists, sorted by priority (descending)

### 3. Runtime — Hook Dispatch

Hooks fire at specific points in the request pipeline (see [Hook Reference](#hook-reference)).

### 4. Shutdown — Cleanup

`dispatch_on_shutdown()` is called during `Pipe.shutdown()`, before the artifact store and session log manager are closed.

---

## Understanding Hook Points

The plugin system exposes nine hook points that let you tap into different stages of the request lifecycle. Here's what each one can do for you, in plain terms:

### on_init — Setup

Your plugin's setup moment. This is where you store the `PluginContext` reference, initialize counters, set up data structures, or prepare anything your plugin needs before it starts handling requests. Runs exactly once when the plugin system first loads. It's synchronous, so don't do anything slow here — no network calls, no database connections, no file I/O.

### on_models — Control the Model Selector

This hook fires every time Open WebUI refreshes its model dropdown list. You receive the full list of models and can modify it however you want:

- **Inject virtual models** — add entries that don't correspond to real LLMs (e.g. a diagnostics plugin that turns a virtual model into an interactive command interface)
- **Hide models** — remove entries you don't want users to see (e.g., filter out expensive models for non-admin users)
- **Reorder or sort** — put preferred models at the top, or sort by price
- **Annotate** — tag model names (e.g., append "[FREE]" to zero-cost models)
- **Filter by capability** — remove models that lack specific features (vision, tool calling, etc.)

You get the full model data from OpenRouter (pricing, capabilities, context length), so filtering decisions can be precise.

### on_request — Intercept or Inspect Requests

The most powerful hook. It fires at the very start of request processing, before authentication checks and before anything is sent to OpenRouter. This is the **only hook that can intercept a request** — return a response and the request never reaches OpenRouter at all.

Use cases:

- **Virtual models** — intercept requests to your injected model IDs and return custom responses (markdown, data, commands — anything)
- **Rate limiting** — track request frequency per user and block when limits are exceeded
- **Access control** — restrict certain models or features to specific user roles
- **Audit logging** — record every request that passes through, without modifying anything
- **Request routing** — mutate the body to change the target model or add parameters, then return `None` to let it proceed

Key detail: this is the only hook where the return value matters. Return `None` to let the request continue normally. Return a dict or string to intercept it entirely.

### on_request_transform — Shape Outbound Requests

Fires just before the request is sent to OpenRouter, after authentication has been verified. Unlike `on_request`, this hook cannot intercept — it can only transform. You mutate the request body in place:

- **Inject system messages** — prepend instructions or persona definitions
- **Strip PII** — scrub personally identifiable information (emails, SSNs, credit card numbers) from user messages before they leave your infrastructure
- **Remap models** — silently redirect users to different models based on their role or organization
- **Add parameters** — inject provider preferences, reasoning effort settings, or custom OpenRouter headers
- **Enforce policies** — ensure all outbound requests comply with organizational rules

By the time this hook fires, you know the request is legitimate (auth passed). Focus purely on shaping the payload.

### on_emitter_wrap — Intercept the Event Stream

This hook fires once per streaming request, just after the middleware stream emitter is created but before the streaming loop begins. You receive the stream emitter (the callable that receives every SSE event — thinking deltas, message chunks, tool calls, completions) and can wrap or replace it.

This is the only hook that operates on the **emitter itself** rather than on request or response data. It's a pipeline insertion point — your wrapper sits between the streaming engine and Open WebUI's event handler:

- **Live reasoning display** — intercept `reasoning:delta` and `reasoning:completed` events to stream model thinking into a custom UI
- **Tool execution monitoring** — capture `response.output_item.added` events for function calls to build real-time tool status displays
- **Event logging** — observe every event that flows through the streaming pipeline without modifying any of them
- **Custom event injection** — emit additional events (like iframe embeds) at specific points during streaming
- **Content filtering** — inspect and modify individual chunks before they reach Open WebUI

The hook uses chain dispatch: if multiple plugins subscribe, each receives the (possibly already-wrapped) emitter from the previous plugin. Return a callable to replace it, or `None` to leave it unchanged. Background tasks (title/tags/emoji generation) are automatically excluded — this hook only fires for user-facing streaming requests.

### on_tool_result — Observe Tool Outcomes

Fires once for every tool call the pipe resolves, as each result lands — including every tool still unfinished when its batch exceeds its timeout (those resolve as `failed`). You receive the tool name and its real execution status (`completed`, `failed`, `skipped`, or `cancelled`, before it is flattened for emission). This is an observe-only hook: there is no return value and you cannot alter the tool result. Use it for per-request tool tallies, success-rate metrics, or audit trails. It runs on the hot path with no timeout, so keep the work trivial.

### on_request_retry — Observe Retry Decisions

Fires when the request orchestrator decides to retry an OpenRouter call after a recoverable rejection — currently the reasoning-effort downgrade, the signed-reasoning strip, and the reasoning-disable retries. You receive the retry `kind`. Observe-only, no return value. Note that only orchestrator-visible retries are hooked, so retries handled deeper in the streaming path may be undercounted.

### on_generation_complete — Observe Terminal State

Fires when a request reaches its terminal state — every chat turn (streaming or not) and every task-model call (title/tags/emoji). You receive the summed `usage` for the turn (which may be `None` if nothing was received) and a `status` of `ok`, `failed`, or `cancelled`, derived from the pipe's own terminal flags rather than from emitted events. A per-request backstop guards early-return and exception paths, so it fires **at least once** — observers that count or accumulate must **dedupe by `request_id`**. Observe-only. Ideal for completion counters, cost accounting, and usage analytics. Like the other observer hooks, it runs as a plain await with no timeout.

### on_shutdown — Cleanup

Runs when the pipe shuts down. Close connections, flush buffers, save state, log final statistics. Synchronous, like `on_init` — but it may **return an awaitable** (e.g. a background task it just cancelled); the async teardown path awaits returned awaitables (bounded at 5 s) so cleanup `finally` blocks complete before `close()` returns. Called for every plugin regardless of hook subscriptions.

---

## Creating a Plugin

### Minimal Plugin

```python
# plugins/my_plugin/plugin.py
from __future__ import annotations
from typing import Any
from ..base import PluginBase, PluginContext
from ..registry import PluginRegistry


@PluginRegistry.register
class MyPlugin(PluginBase):
    plugin_id = "my-plugin"
    plugin_name = "My Plugin"
    plugin_version = "1.0.0"

    # Subscribe to hooks with priorities (higher = runs first)
    hooks = {
        "on_models": 50,
    }

    def on_init(self, ctx: PluginContext, **kwargs: Any) -> None:
        self.ctx = ctx
        self.ctx.logger.info("MyPlugin initialized")

    def on_models(self, models: list[dict[str, Any]], **kwargs: Any) -> None:
        # Add a virtual model (mutate the list in place)
        models.append({"id": "my-virtual-model", "name": "My Virtual Model"})
```

### File Structure

```
plugins/
├── __init__.py              # Auto-discovery + explicit imports
├── base.py                  # PluginBase, PluginContext
├── registry.py              # PluginRegistry
├── _utils.py                # Shared helpers (always bundled)
└── my_plugin/
    ├── __init__.py          # Re-export: from .plugin import MyPlugin
    └── plugin.py            # @PluginRegistry.register class
```

A plugin may grow into a multi-module package of its own. For a fully worked example, see [plugins_pipe_dashboard_internals.md](plugins_pipe_dashboard_internals.md).

### Registration Checklist

1. Decorate your plugin class with `@PluginRegistry.register`
2. Set `plugin_id`, `plugin_name`, `plugin_version`
3. Declare subscribed hooks in the `hooks` dict with priority values

> Always define `hooks` as a class-level attribute (`hooks = {"on_models": 50}`). Never mutate it at instance level (e.g., `self.hooks["on_models"] = 50` in `on_init()`). Each subclass automatically gets its own isolated `hooks` dict via `__init_subclass__`.

4. Add an explicit import in `plugins/__init__.py` for bundle compatibility:
   ```python
   from . import my_plugin as _my_plugin  # noqa: E402, F401
   ```
5. Re-export from your `__init__.py`:
   ```python
   from .plugin import MyPlugin
   __all__ = ["MyPlugin"]
   ```

---

## Plugin-Exported Valves

Plugins can declare configuration fields that appear in the Open WebUI settings UI alongside the pipe's built-in valves. This lets plugins expose admin-level and per-user settings without editing the main Valves class.

### How It Works

1. Plugin declares `plugin_valves` and/or `plugin_user_valves` as class attributes
2. `@PluginRegistry.register` collects field specs at import time
3. `build_extended_valves()` / `build_extended_user_valves()` merge them into `Pipe.Valves` / `Pipe.UserValves` via `pydantic.create_model()`
4. OWUI reads the merged class schema — plugin fields appear in Settings

**Merge timing and failure mode:** The merge runs **once at import time**, when the pipe module finishes loading and reassigns `Pipe.Valves` / `Pipe.UserValves`. Plugin valve fields are therefore fixed for the process — a plugin cannot add or remove valve fields at runtime. When no plugin declares fields, `build_extended_valves()` returns the base `Valves` class unchanged (zero overhead). The merge is wrapped in `try/except`: if a field spec is malformed, the extension is skipped silently and `Pipe.Valves` keeps only its built-in fields, so the plugin's fields simply do not appear.

### System Valves (admin-only)

Declared via `plugin_valves`. Only admins see and modify these in the OWUI Settings UI.

```python
from pydantic import Field
from ..base import PluginBase
from ..registry import PluginRegistry

@PluginRegistry.register
class MyPlugin(PluginBase):
    plugin_id = "my-plugin"
    plugin_valves = {
        "MY_PLUGIN_ENABLE": (bool, Field(
            default=True,
            description="Enable the My Plugin feature.",
        )),
        "MY_PLUGIN_LIMIT": (int, Field(
            default=100, ge=1, le=1000,
            description="Maximum items to process.",
        )),
    }
```

Access at runtime: `self.ctx.valves.MY_PLUGIN_ENABLE`

### User Valves (per-user)

Declared via `plugin_user_valves`. Each user gets their own copy of these settings.

```python
@PluginRegistry.register
class MyPlugin(PluginBase):
    plugin_id = "my-plugin"
    plugin_user_valves = {
        "USER_MY_PLUGIN_THEME": (str, Field(
            default="auto",
            description="UI theme preference.",
        )),
    }
```

Access at runtime (from hooks that receive `user`):

```python
async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
    user_valves = user.get("valves")
    theme = getattr(user_valves, "USER_MY_PLUGIN_THEME", "auto") if user_valves else "auto"
```

### Field Type → UI Control Mapping

OWUI renders controls based on Pydantic JSON Schema output:

| Pydantic Type | OWUI Renders |
|---|---|
| `bool` | Toggle switch |
| `int` with `ge`/`le` | Slider |
| `float` with `ge`/`le` | Slider (decimal) |
| `str` | Text input |
| `SecretStr` | Password input (masked) |
| `Literal["a", "b", "c"]` | Dropdown |

### Naming Convention

Prefix all field names with your plugin ID to avoid collisions:

```
COUNTER_ENABLE         ✓  (prefixed with COUNTER_)
MY_PLUGIN_LIMIT        ✓  (prefixed with MY_PLUGIN_)
ENABLE                 ✗  (too generic, will collide)
```

If two plugins claim the same field name, the second is **auto-renamed** with a `_2` suffix and a warning is logged. No crash, no data loss — but the admin sees a confusingly named field. Use unique prefixes.

### Isolation

Each plugin subclass gets its own `plugin_valves` and `plugin_user_valves` dicts via `__init_subclass__`. Mutating one plugin's dict never affects another plugin or the base class.

---

## Hook Reference

### `on_init`

```python
def on_init(self, ctx: PluginContext, **kwargs: Any) -> None:
```

**When:** Called once during `init_plugins()` (first access to plugin registry). Synchronous.

**Purpose:** Store the `PluginContext` for later use. Initialize plugin state.

**Not subscription-based:** Always called for all plugins, regardless of the `hooks` dict.

> `on_init` and `on_shutdown` are synchronous and have NO timeout protection — they must be non-blocking (no network I/O, database connections, or other blocking work inside the hook body). For async cleanup, `on_shutdown` may return an awaitable (typically a cancelled task); the host awaits returned awaitables during async teardown with a 5 s bound.

**Example:**
```python
def on_init(self, ctx: PluginContext, **kwargs: Any) -> None:
    self.ctx = ctx
    self.request_count = 0
    self.ctx.logger.info("Plugin ready, pipe_id=%s", ctx.pipe_id)
```

---

### `on_models`

```python
def on_models(self, models: list[dict[str, Any]], **kwargs: Any) -> None:
```

**When:** Called at the end of `Pipe.pipes()` — every time Open WebUI fetches the model list. Synchronous.

**Dispatch location:** `Pipe.pipes()` (`pipe.py`):
```python
# Plugins see full model data (pricing, capabilities) — trusted code
if self.valves.ENABLE_PLUGIN_SYSTEM:
    try:
        await self._ensure_plugin_registry().dispatch_on_models(selected_models)
    except Exception:
        self.logger.debug("Plugin on_models dispatch failed", exc_info=True)
# Return simple id/name list — OWUI's get_function_models() only reads these fields
return [
    {"id": m["id"], "name": m.get("name", m["id"])}
    for m in selected_models
    if isinstance(m, dict) and "id" in m
]
```

**Dispatch type:** **Void/Mutation** — all subscribers receive a reference to the same list and mutate it in place. Plugins run in priority order; mutations made by higher-priority plugins are immediately visible to lower-priority ones.

**Return:** `None`. This is a void hook. Mutate the `models` list directly (append, remove, reorder, modify dicts in place). Do **not** reassign the parameter (`models = [...]`) -- that rebinds the local variable and the caller never sees the change.

**Model dict format:** Plugins receive the **full model data** from OpenRouter (pricing, capabilities, context length, provider info, etc.). The pipe strips to `{id, name}` **after** plugin dispatch. Plugin-injected models with minimal fields work fine.

> **Note:** `on_models` may be either `def` or `async def`. The dispatcher awaits the result when the hook returns an awaitable, so async implementations can call async APIs (e.g. OWUI's Models table) directly.

**Example 1 — add a virtual model (simple):**
```python
def on_models(self, models: list[dict[str, Any]], **kwargs: Any) -> None:
    """Append a virtual model that the plugin handles via on_request."""
    if getattr(self.ctx.valves, "ENABLE_MY_FEATURE", True):
        models.append({"id": "my-feature", "name": "My Feature"})
```

**Example 2 — filter models by capability (moderate):**
```python
def on_models(self, models: list[dict[str, Any]], **kwargs: Any) -> None:
    """Remove models that only support vision input (no text chat)."""
    # Iterate backwards so index stays valid while removing
    for i in range(len(models) - 1, -1, -1):
        arch = models[i].get("architecture", {})
        input_modalities = arch.get("modality", "").split("->")[0] if arch else ""
        # Keep models that accept text; remove vision-only
        if "text" not in input_modalities and "image" in input_modalities:
            models.pop(i)
```

**Example 3 — sort by pricing and add metadata (advanced):**
```python
def on_models(self, models: list[dict[str, Any]], **kwargs: Any) -> None:
    """Sort models by prompt price (cheapest first) and tag free models."""
    for m in models:
        pricing = m.get("pricing") or {}
        prompt_cost = float(pricing.get("prompt", "0") or "0")
        if prompt_cost == 0:
            m["name"] = f"{m.get('name', m['id'])} [FREE]"
        m["_sort_cost"] = prompt_cost

    models.sort(key=lambda m: m.get("_sort_cost", float("inf")))

    # Clean up temp key
    for m in models:
        m.pop("_sort_cost", None)
```

---

### `on_request`

```python
async def on_request(
    self,
    body: dict[str, Any],
    user: dict[str, Any],
    metadata: dict[str, Any],
    event_emitter: Any,
    task: Any,
    **kwargs: Any,
) -> dict[str, Any] | str | None:
```

**When:** Very early in `_handle_pipe_call()`, **before the API key check**. This is the first hook that fires during request processing.

**Extra kwargs:** `valves` — merged per-request valves (Valves instance), `request_id` — the pipe's per-request id (from `SessionLogger.request_id`), `current_result` — accumulated result from prior plugins in the chain (or `None` if no plugin has returned a value yet).

**Dispatch location:** `_handle_pipe_call` (`pipe.py`):
```python
plugin_result = await self._ensure_plugin_registry().dispatch_on_request(
    body, __user__, __metadata__, __event_emitter__, __task__,
    valves=valves,
    request_id=SessionLogger.request_id.get() or "",
)
if plugin_result is not None:
    # Streaming fix: emit content to stream_queue
    ...
    return plugin_result
```

**Dispatch type:** **Chain** — all subscribers run in priority order. Each plugin receives the current accumulated result via the `current_result` kwarg. Return non-`None` to set or replace the result; return `None` to leave the current result unchanged. The final accumulated result after all plugins is returned.

> **Chain semantics:** ALL subscribers run in priority order regardless of what earlier plugins return. Each plugin sees the accumulated result from prior plugins in `current_result` (starts as `None`). A plugin can inspect, refine, or replace the current result by returning a non-`None` value. Return `None` to leave it unchanged.

**Can short-circuit:** If the final result is non-`None`, the entire request is intercepted. No API call to OpenRouter is made.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `body` | `dict` | Full request body (`model`, `messages`, `stream`, etc.) |
| `user` | `dict` | Open WebUI user object (`role`, `id`, `name`, `email`) |
| `metadata` | `dict` | Request metadata from Open WebUI |
| `event_emitter` | `callable` | Async event emitter (wraps stream queue for streaming) |
| `task` | `Any` | Open WebUI task metadata (`"title_generation"`, `"tags_generation"`, etc.) or `None` |
| `**kwargs` | `Any` | Forward-compatibility kwargs. Currently: `valves` (merged per-request valves), `request_id` (the pipe's per-request id), `current_result` (accumulated result from prior plugins, or `None` if no plugin has returned a value yet) |

**Return values:**

| Return | Effect |
|--------|--------|
| `None` | No intercept — request continues to OpenRouter |
| `dict` | Chat completion format — returned as the response |
| `str` | Raw string — returned as the response |

**Example 1 — intercept a command keyword (simple):**
```python
async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
    """Intercept messages starting with '!ping' and return a canned response."""
    messages = body.get("messages", [])
    if not messages:
        return None
    last_msg = messages[-1].get("content", "")
    if isinstance(last_msg, str) and last_msg.strip().lower() == "!ping":
        return self.ctx.build_response(model="ping", content="Pong!")
    return None
```

**Example 2 — rate-limit by user (moderate):**
```python
async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
    """Enforce a per-user rate limit of 10 requests per minute."""
    import time

    user_id = user.get("id", "anonymous")
    now = time.monotonic()

    # Expire old timestamps
    window = self._requests.get(user_id, [])
    window = [t for t in window if now - t < 60]
    self._requests[user_id] = window

    if len(window) >= 10:
        return self.ctx.build_response(
            model=str(body.get("model", "system")),
            content="Rate limit exceeded. Please wait before sending another request.",
        )

    window.append(now)
    return None  # Under limit — continue to OpenRouter
```

**Example 3 — add provider routing preferences (advanced):**
```python
async def on_request(self, body, user, metadata, event_emitter, task, **kwargs):
    """Add OpenRouter provider preferences for compliance-sensitive models.

    Mutates the body to enforce specific infrastructure providers, then
    returns None to let the request proceed with the added preferences.
    """
    model = str(body.get("model", ""))
    if not model.startswith("anthropic/"):
        return None

    # Enforce Anthropic-direct routing (no third-party providers)
    body.setdefault("provider", {})
    body["provider"]["order"] = ["Anthropic"]
    body["provider"]["allow_fallbacks"] = False

    self.ctx.logger.debug("Enforced Anthropic-direct routing for %s", model)
    return None  # Continue — the pipe sends the mutated body to OpenRouter
```

---

### `on_request_transform`

```python
async def on_request_transform(
    self,
    body: dict[str, Any],
    model: str,
    valves: Any,
    **kwargs: Any,
) -> None:
```

**When:** After API key checks, just before the request is sent to OpenRouter.

**Extra kwargs:** `user` — user dict, `metadata` — request metadata.

**Dispatch location:** `_handle_pipe_call` (`pipe.py`):
```python
await self._ensure_plugin_registry().dispatch_on_request_transform(
    body, str(body.get("model", "")), valves,
    user=__user__, metadata=__metadata__,
)
```

**Dispatch type:** **Void/Mutation** — all subscribers receive a reference to the same `body` dict and mutate it in place. The `model` parameter is re-read from `body["model"]` after each plugin, so if a plugin changes the model in the body, the next plugin sees the updated model string.

**Return:** `None`. This is a void hook. Mutate the `body` dict directly (add keys, modify values, alter `messages`). Do **not** reassign the parameter (`body = {...}`) -- that rebinds the local variable and the caller never sees the change.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `body` | `dict` | Full request body (`model`, `messages`, `stream`, etc.). Mutate in place. |
| `model` | `str` | Current model ID (re-read from `body["model"]` after each plugin) |
| `valves` | `Valves` | Merged per-request valves |
| `**kwargs` | `Any` | Forward-compatibility kwargs. Currently: `user`, `metadata` |

**Example 1 — inject a system message (simple):**
```python
async def on_request_transform(self, body, model, valves, **kwargs):
    """Prepend a system message if none exists."""
    messages = body.get("messages", [])
    has_system = any(m.get("role") == "system" for m in messages)
    if not has_system:
        messages.insert(0, {
            "role": "system",
            "content": "You are a helpful assistant.",
        })
```

**Example 2 — strip PII patterns from messages (moderate):**
```python
import re

_PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN REDACTED]"),       # SSN
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL REDACTED]"),
    (re.compile(r"\b\d{16}\b"), "[CARD REDACTED]"),                   # Credit card
]

async def on_request_transform(self, body, model, valves, **kwargs):
    """Scrub personally identifiable information from user messages."""
    for msg in body.get("messages", []):
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        for pattern, replacement in _PII_PATTERNS:
            content = pattern.sub(replacement, content)
        msg["content"] = content
```

**Example 3 — remap model IDs based on user role (advanced):**
```python
_ROLE_MODEL_MAP = {
    "pending": "meta-llama/llama-3.1-8b-instruct:free",    # Free tier for unverified users
    "user":    "openai/gpt-4o-mini",                         # Budget model for regular users
    # "admin" keeps whatever model was selected
}

async def on_request_transform(self, body, model, valves, **kwargs):
    """Enforce model access tiers based on user role.

    Admins keep their selected model. Regular and pending users are
    silently remapped to their tier's allowed model.
    """
    user = kwargs.get("user", {})
    role = user.get("role", "user")

    replacement = _ROLE_MODEL_MAP.get(role)
    if replacement and body.get("model") != replacement:
        self.ctx.logger.debug(
            "Remapping model %s -> %s for role=%s", body.get("model"), replacement, role,
        )
        body["model"] = replacement
```

---

### `on_emitter_wrap`

```python
async def on_emitter_wrap(
    self,
    stream_emitter: Any,
    **kwargs: Any,
) -> Any | None:
```

**When:** After the middleware stream emitter is created in `_execute_pipe_job()`, but before the streaming loop begins. Fires once per streaming request. Does **not** fire for background tasks (title/tags/emoji generation).

**Extra kwargs:** `raw_emitter` — the OWUI-facing event emitter (bypasses the middleware stream queue; use for embeds), `job_metadata` — dict with `user_id`, `chat_id`, `message_id`, `request_id`, `valves` — merged system+user valves for the request.

**Dispatch location:** `_execute_pipe_job` (`pipe.py`):
```python
if (
    self.valves.ENABLE_PLUGIN_SYSTEM
    and stream_emitter is not None
    and not job.task  # Skip background tasks
):
    wrapped = await self._ensure_plugin_registry().dispatch_on_emitter_wrap(
        stream_emitter,
        raw_emitter=job.event_emitter,
        job_metadata={
            "user_id": job.user_id,
            "chat_id": job.metadata.get("chat_id", ""),
            "message_id": job.metadata.get("message_id", ""),
            "request_id": job.request_id,
        },
        valves=job.valves,
    )
    if wrapped is not None and wrapped is not stream_emitter:
        stream_emitter = wrapped
```

**Dispatch type:** **Chain/Wrap** — all subscribers run in priority order. Each plugin receives the current emitter (which may already be wrapped by a prior plugin). Return a callable to replace the emitter, or `None` to leave it unchanged. The final emitter is used by the streaming loop.

**Return values:**

| Return | Effect |
|--------|--------|
| `None` | No wrapping — emitter unchanged |
| callable | Replaces the stream emitter for the current request |

**Important details:**

- If the inner emitter exposes a `flush_reasoning_status` callable, a wrapper must proxy it (e.g. via `__getattr__`): the post-stream reasoning flush reads that attribute off the outermost emitter, so a bare-function wrapper that does not expose it skips the flush.
- The `raw_emitter` kwarg is a direct line to OWUI's event handler — events sent through it bypass the middleware stream queue, so a plugin can emit iframe embeds immediately rather than waiting for the stream queue to process them.
- The `job_metadata` dict provides per-request identifiers useful for session tracking and cleanup.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `stream_emitter` | callable | The middleware stream emitter (or a wrapper from a prior plugin) |
| `**kwargs` | `Any` | Forward-compatibility kwargs. Currently: `raw_emitter`, `job_metadata`, `valves` |

**Example 1 — observe all events (simple):**
```python
async def on_emitter_wrap(self, stream_emitter, **kwargs):
    """Log every event that flows through the streaming pipeline."""
    import logging
    log = logging.getLogger(__name__)

    async def _wrapper(event):
        if isinstance(event, dict):
            log.debug("Stream event: type=%s", event.get("type"))
        await stream_emitter(event)

    return _wrapper
```

**Example 2 — intercept thinking events for a custom display (moderate):**
```python
async def on_emitter_wrap(self, stream_emitter, **kwargs):
    """Intercept reasoning events and copy them to a session queue
    for a live iframe display, while still passing them through to
    OWUI for DB persistence.
    """
    valves = kwargs.get("valves")
    if not getattr(valves, "MY_FEATURE_ENABLE", False):
        return None

    raw_emitter = kwargs.get("raw_emitter")
    job_metadata = kwargs.get("job_metadata", {})

    # Create a session (per-request queue for the iframe consumer)
    session = self._create_session(job_metadata)

    class ReasoningWrapper:
        _reasoning_wrapper_active = True  # Prevent double-wrapping

        async def __call__(self, event):
            if isinstance(event, dict) and event.get("type") == "reasoning:delta":
                # Copy to session queue for the iframe consumer
                session.queue.put_nowait(event.get("data", {}).get("delta", ""))
            # Always pass through to OWUI
            await stream_emitter(event)

        def __getattr__(self, name):
            return getattr(stream_emitter, name)

    return ReasoningWrapper()
```

**Example 3 — inject an iframe embed on first thinking event (advanced):**
```python
async def on_emitter_wrap(self, stream_emitter, **kwargs):
    """Emit an iframe embed via raw_emitter on the first thinking event.

    Deferred emission means the iframe only appears when there is actual
    thinking content to show — no empty embed for non-reasoning models.
    """
    raw_emitter = kwargs.get("raw_emitter")
    iframe_html = self._build_iframe_html()
    emitted = False

    async def _wrapper(event):
        nonlocal emitted
        if isinstance(event, dict) and event.get("type") == "reasoning:delta":
            if not emitted:
                emitted = True
                # Emit iframe via raw_emitter (direct to OWUI, bypasses stream queue)
                if raw_emitter:
                    await raw_emitter({
                        "type": "embeds",
                        "data": {"embeds": [iframe_html]},
                    })
        await stream_emitter(event)

    return _wrapper
```

---

### `on_tool_result`

```python
async def on_tool_result(
    self,
    tool_name: str,
    status: str,
    **kwargs: Any,
) -> None:
```

**When:** Fires once per resolved tool call, from `_execute_tool_batch` — for each tool as its result lands, and once for every tool still unfinished when its batch exceeds its timeout (those resolve as `failed`). `status` is the executor's real per-tool outcome (`completed`, `failed`, `skipped`, or `cancelled`) **before** it is flattened for emission.

**Extra kwargs:** `request_id` — the pipe's per-request id, `metadata` — the request's OWUI metadata dict.

**Dispatch location:** `_execute_tool_batch` (`pipe.py`):
```python
await self._dispatch_plugin_event(
    "dispatch_on_tool_result",
    str(item.call.get("name") or "?"),
    resolved_status,
    request_id=context.request_id,
    metadata=context.metadata or {},
)
```

**Dispatch type:** **Void broadcast (observer)** — all subscribers run in priority order, each awaited in turn inside a per-plugin `try/except`. Unlike the transform hooks, observer dispatch is a plain `await` with **no** 30 s `wait_for` timeout — these are hot-path observers, so keep the work trivial. Dispatch is routed through `_dispatch_plugin_event()`, which is a no-op when `ENABLE_PLUGIN_SYSTEM` is `False` or the registry was never created.

**Return:** `None`. Observe only — you cannot alter the tool result.

**Example — tally tool outcomes per request:**
```python
async def on_tool_result(self, tool_name, status, **kwargs):
    """Count tool successes and failures for the current request."""
    request_id = kwargs.get("request_id", "")
    bucket = self._tool_stats.setdefault(request_id, {"ok": 0, "failed": 0})
    if status == "completed":
        bucket["ok"] += 1
    elif status == "failed":
        bucket["failed"] += 1
```

---

### `on_request_retry`

```python
async def on_request_retry(
    self,
    kind: str,
    **kwargs: Any,
) -> None:
```

**When:** Fires when the request orchestrator decides to retry an OpenRouter call after a recoverable rejection. `kind` names the retry class — currently `reasoning_effort` (effort downgrade), `signature` (drop signed reasoning), or `reasoning` (disable reasoning). **Undercount caveat:** only orchestrator-visible retries are hooked; retries handled deeper in the streaming path are not counted.

**Extra kwargs:** `request_id` — the pipe's per-request id.

**Dispatch location:** `requests/orchestrator.py` (at each retry decision, e.g. the signed-reasoning strip):
```python
await self._pipe._dispatch_plugin_event(
    "dispatch_on_request_retry",
    "signature",  # or "reasoning_effort" / "reasoning"
    request_id=SessionLogger.request_id.get() or "",
)
```

**Dispatch type:** **Void broadcast (observer)** — same semantics as `on_tool_result`: priority order, per-plugin `try/except`, plain `await` with no timeout, routed through `_dispatch_plugin_event()`.

**Return:** `None`. Observe only.

**Example — count retries per request:**
```python
async def on_request_retry(self, kind, **kwargs):
    """Increment a per-request retry counter."""
    request_id = kwargs.get("request_id", "")
    self._retries[request_id] = self._retries.get(request_id, 0) + 1
```

---

### `on_generation_complete`

```python
async def on_generation_complete(
    self,
    usage: dict[str, Any] | None,
    status: str,
    **kwargs: Any,
) -> None:
```

**When:** Fires **at least once** when a request reaches its terminal state. `usage` is the summed usage for the turn (or `None` if nothing was received); `status` is `ok`, `failed`, or `cancelled`, derived from the pipe's own terminal flags (never from emitted events). Because a per-request backstop guards early-return and exception paths, observers that count or accumulate **must dedupe by `request_id`** (a real terminal and the backstop can both fire for one request). Four code paths emit it:

- **Chat turns** — the streaming core's terminal `finally` block emits `ok`/`failed`/`cancelled` (from `was_cancelled` / `error_occurred`). It is **skipped** when the error is handed back to the orchestrator for a retry (`handed_back_for_retry`), so a retried turn still fires exactly once, at its true terminal.
- **Orchestrator give-up** — when all retries are exhausted, the orchestrator emits `failed` with `usage=None`.
- **Task-model calls** (title/tags/emoji) — the task-model adapter emits `ok` (with usage) on success, or `failed` when all attempts fail.
- **Terminal backstop** — `_execute_pipe_job` re-emits `ok`/`failed`/`cancelled` (with `usage=None`) for any job with a `request_id`, covering early-return and exception paths the three primary emitters miss. This is the source of a possible duplicate — dedupe by `request_id`.

**Extra kwargs:** `request_id` — the pipe's per-request id, `metadata` — the request's OWUI metadata dict, `task` — the task name string, or `None` for chat turns. (`metadata` and `task` are omitted on the orchestrator give-up path.)

**Dispatch location:** the streaming loop's terminal `finally` (`streaming/streaming_core.py`):
```python
generation_status = "cancelled" if was_cancelled else ("failed" if error_occurred else "ok")
if not handed_back_for_retry:  # skip when the error goes back for a retry
    await asyncio.shield(
        self._pipe._dispatch_plugin_event(
            "dispatch_on_generation_complete",
            total_usage if isinstance(total_usage, dict) else None,
            generation_status,
            request_id=SessionLogger.request_id.get() or "",
            metadata=metadata,
            task=None,
        )
    )
```

**Dispatch type:** **Void broadcast (observer)** — priority order, per-plugin `try/except`, plain `await` with no timeout, routed through `_dispatch_plugin_event()`.

**Return:** `None`. Observe only.

**Example — accumulate token usage and completion counts:**
```python
async def on_generation_complete(self, usage, status, **kwargs):
    """Count terminal completions by status and sum tokens."""
    self._completions[status] = self._completions.get(status, 0) + 1
    if isinstance(usage, dict):
        self._total_tokens += usage.get("total_tokens", 0)
```

---

### `on_shutdown`

```python
def on_shutdown(self, **kwargs: Any) -> Any:
```

**When:** During `Pipe.shutdown()`, before the artifact store and session log manager are closed. Synchronous; runs on both hot-reload and process shutdown.

**Return value:** `None`, or an awaitable (e.g. a background task the hook just cancelled). Returned awaitables are gathered and awaited by the async teardown path (5 s bound) so the task's cleanup `finally` completes before `close()` returns. The sync-only path (`__del__`) cannot await them — cleanup there is best-effort.

**Not subscription-based:** Always called for all plugins, like `on_init`.

**Example:**
```python
def on_shutdown(self, **kwargs: Any) -> None:
    self.ctx.logger.info(
        "Plugin shutting down. Processed %d requests.", self.request_count,
    )
```

---

## Hook Dispatch Semantics

Each hook has a specific dispatch pattern:

| Hook | Dispatch Type | Return | Description |
|------|---------------|--------|-------------|
| `on_init` | **Always** | `None` | Called for all plugins (not subscription-based) |
| `on_models` | **Void/Mutation** | `None` | All subscribers mutate the same list in place |
| `on_request` | **Chain** | `dict \| str \| None` | All subscribers run; each receives `current_result` kwarg; return non-`None` to set/replace result |
| `on_request_transform` | **Void/Mutation** | `None` | All subscribers mutate the same body dict in place |
| `on_emitter_wrap` | **Chain/Wrap** | `callable \| None` | All subscribers run; each wraps or replaces the current emitter |
| `on_tool_result` | **Void broadcast (observer)** | `None` | Observe one resolved tool call; plain await, no timeout |
| `on_request_retry` | **Void broadcast (observer)** | `None` | Observe one orchestrator retry decision; plain await, no timeout |
| `on_generation_complete` | **Void broadcast (observer)** | `None` | Observe the terminal state once per request; plain await, no timeout |
| `on_shutdown` | **Always** | `None` | Called for all plugins (not subscription-based) |

### Parameter Availability

All hooks accept `**kwargs` for forward compatibility. Extra keyword arguments are passed by the dispatch layer:

| Hook | Positional Params | Extra kwargs |
|------|-------------------|-------------|
| `on_init` | `ctx` | *(none currently)* |
| `on_models` | `models` | *(none currently)* |
| `on_request` | `body`, `user`, `metadata`, `event_emitter`, `task` | `valves`, `request_id`, `current_result` |
| `on_request_transform` | `body`, `model`, `valves` | `user`, `metadata` |
| `on_emitter_wrap` | `stream_emitter` | `raw_emitter`, `job_metadata`, `valves` |
| `on_tool_result` | `tool_name`, `status` | `request_id`, `metadata` |
| `on_request_retry` | `kind` | `request_id` |
| `on_generation_complete` | `usage`, `status` | `request_id`, `metadata`, `task` |
| `on_shutdown` | *(none)* | *(none currently)* |

### Void/Mutation Dispatch (on_models, on_request_transform)

```
         mutates in place             mutates in place           mutates in place
Shared → Plugin A (priority 100) → Plugin B (priority 50) → Plugin C (priority 10) → Caller sees
Object                                                                                 all mutations
```

All plugins receive a **reference** to the same object (list or dict). There are no copies, no return values, and no reassignment. Each plugin mutates the object directly, and its changes are immediately visible to subsequent plugins and to the caller.

**Python mutation gotcha:** Because the object is passed by reference, you must mutate it in place. Reassigning the local parameter creates a new local variable and silently discards the change:

```python
# CORRECT — mutates the dict in place; caller sees the change
body["temperature"] = 0.2

# WRONG — rebinds the local variable; caller never sees this
body = {"temperature": 0.2}
```

The same applies to lists (`models`): use `models.append(...)`, `models.pop(...)`, `models.sort(...)`, etc. Do **not** write `models = [...]`.

For `on_request_transform`, the `model` parameter is re-read from `body["model"]` after each plugin, so if a plugin changes the model in the body, the next plugin sees the updated model string.

### Chain Dispatch (on_request)

```
Input → Plugin A (priority 100) → Plugin B (priority 50) → Plugin C (priority 10) → Output
         may return result            may refine result        may refine result
```

`on_request` uses chain-with-return semantics. The accumulated result (starting at `None`) is passed to each plugin via the `current_result` kwarg. Each plugin can set or replace it by returning non-`None`, or leave it unchanged by returning `None`. All subscribers always run regardless of what earlier plugins return.

### Chain/Wrap Dispatch (on_emitter_wrap)

```
                     may wrap                    may wrap                    may wrap
Original emitter → Plugin A (priority 100) → Plugin B (priority 50) → Plugin C (priority 10) → Final emitter
```

`on_emitter_wrap` uses chain-wrap semantics. Each plugin receives the current emitter (which may already be wrapped by a prior plugin). Return a callable to replace it, or `None` to leave it unchanged. All subscribers always run. The final emitter — the outermost wrapper — is used by the streaming loop.

This pattern enables layered interception: Plugin A's wrapper calls Plugin B's wrapper, which calls the original emitter. Each layer can observe, modify, or suppress events independently.

---

## Priority System

Priorities are integer values declared in the `hooks` dict. **Higher values run first.**

```python
class MyPlugin(PluginBase):
    hooks = {
        "on_models": 100,     # Runs before plugins with priority < 100
        "on_request": 50,     # Default priority
    }
```

### Priority Guidelines

| Range | Recommended Use |
|-------|-------------|
| 90–100 | Security plugins (auth enforcement, rate limiting) |
| 50–89 | Feature plugins (virtual models, request modification) |
| 10–49 | Observability plugins (logging, analytics, auditing) |
| 1–9 | Fallback / catch-all plugins |

Priority **50** is the conventional default. The Counter Plugin below declares **30** for all its hooks so it runs after higher-priority feature plugins.

### Subscriber List Construction

During `init_plugins()`, the registry builds sorted subscriber lists:

```python
# plugins/registry.py
for hook_name in _PR_SUBSCRIBABLE_HOOKS:
    subscribers = []
    for plugin in self._plugins:
        if hook_name in plugin.hooks:
            subscribers.append((plugin, plugin.hooks[hook_name]))
    subscribers.sort(key=lambda x: x[1], reverse=True)  # Descending
    self._hook_subscribers[hook_name] = subscribers
```

Subscribable hooks: `on_models`, `on_request`, `on_request_transform`, `on_emitter_wrap`, `on_tool_result`, `on_request_retry`, `on_generation_complete`.

Lifecycle hooks (`on_init`, `on_shutdown`) are **not** subscription-based and always fire.

---

## PluginContext — Accessing Pipe Internals

`PluginContext` is the gateway to all pipe internals. It's passed to `on_init()` and should be stored for later use.

### Core Properties

```python
class PluginContext:
    @property
    def pipe(self) -> Pipe:
        """Full Pipe instance — dig into any subsystem."""

    @property
    def valves(self) -> Valves:
        """Live reference to pipe.valves (all configuration)."""

    @property
    def pipe_id(self) -> str:
        """Pipe runtime identifier (e.g., 'open_webui_openrouter_pipe')."""

    @property
    def artifact_store(self) -> ArtifactStore:
        """Artifact persistence (DB + Redis + encryption)."""

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Per-user failure tracking and request gating."""

    # Instance attribute (set in __init__, not a property)
    logger: logging.Logger  # Per-plugin logger (named after the plugin_id)
```

### Response Builders

```python
def build_response(self, *, model: str | None, content: str) -> dict[str, Any]:
    """Build a chat.completion-style response dict.

    ``model`` is normalized to ``model_id`` (stripped; empty or ``None``
    becomes ``"pipe"``), which is used for both the ``id`` and ``model``
    fields. The ``id`` uses the format ``"{model_id}-{uuid4}"``.

    Returns:
        {
            "id": "{model_id}-{uuid4}",
            "object": "chat.completion",
            "created": unix_timestamp,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
                "logprobs": None,
            }],
        }
    """
```

### Accessing Subsystems via `ctx.pipe`

> **Warning:** `ctx.pipe` is an escape hatch for trusted first-party plugins.
> It exposes the full `Pipe` instance including private attributes. Do not
> expose to untrusted third-party plugin code.

The `pipe` property is the escape hatch to any subsystem:

```python
# Direct access to initialized subsystems
store = ctx.pipe._artifact_store          # ArtifactStore
breaker = ctx.pipe._circuit_breaker       # CircuitBreaker
session = ctx.pipe._http_session          # aiohttp.ClientSession (may be None)
emitter = ctx.pipe._event_emitter_handler # EventEmitterHandler
streamer = ctx.pipe._streaming_handler    # StreamingHandler
multimodal = ctx.pipe._multimodal_handler # MultimodalHandler

# Lazy-initialized subsystems (safe to call — creates on first access)
catalog = ctx.pipe._ensure_catalog_manager()         # ModelCatalogManager
errors = ctx.pipe._ensure_error_formatter()          # ErrorFormatter
reasoning = ctx.pipe._ensure_reasoning_config_manager()  # ReasoningConfigManager
tools = ctx.pipe._ensure_tool_executor()             # ToolExecutor
filters = ctx.pipe._ensure_filter_manager()          # FilterManager
orchestrator = ctx.pipe._ensure_request_orchestrator()   # RequestOrchestrator

# Configuration
valves = ctx.pipe.valves                  # All valve settings
log_level = ctx.pipe.valves.LOG_LEVEL     # Individual valve value

# Model registry (class-level, no pipe reference needed)
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry
models = OpenRouterModelRegistry.list_models()   # Full model catalog
spec = OpenRouterModelRegistry.spec("gpt-4o")    # Model capabilities
```

---

## Streaming Compatibility

When a plugin intercepts a request via `on_request`, the pipe may be in streaming mode (`body["stream"] == True`). In streaming mode, `pipe()` returns an async generator that reads from a `stream_queue`. Plugin responses are complete dicts on `job.future`, which the generator never reads.

The pipe automatically handles this: when `on_request` returns a non-`None` result and the request is streaming, the content is emitted via a `chat:message:delta` event to the stream emitter:

```python
# In _handle_pipe_call (automatic — no plugin code needed)
if bool(body.get("stream")) and __event_emitter__:
    _pcontent = ...  # extracted from plugin result (dict or str)
    if isinstance(_pcontent, str) and _pcontent:
        await __event_emitter__(
            {"type": "chat:message:delta", "data": {"content": _pcontent}}
        )
    # Always signal stream completion when a plugin intercepts the request
    await __event_emitter__(
        {"type": "chat:completion", "data": {"done": True}}
    )
```

**Plugin developers do not need to handle streaming themselves.** Return a dict from `on_request` and both streaming and non-streaming paths work automatically.

---

## Error Isolation

The plugin system is designed to never crash the pipe:

| Failure Point | Behavior | Log Level |
|---------------|----------|-----------|
| Plugin import fails | Other plugins still load | DEBUG |
| `on_init()` throws | Plugin skipped, others continue | WARNING |
| Hook dispatch throws | Request continues without plugin result | DEBUG |
| Hook dispatch times out | Request continues after 30s timeout (transform/chain hooks only) | WARNING |

The three transform/chain hooks — `on_request`, `on_request_transform`, and `on_emitter_wrap` — wrap each plugin call in `asyncio.wait_for()` with a 30-second timeout (`_PR_DISPATCH_TIMEOUT`) and `try/except`. `on_models` and the three observer hooks (`on_tool_result`, `on_request_retry`, `on_generation_complete`) are dispatched with a plain `await` and **no** timeout — they run on the hot path, so a hung observer is treated as a plugin bug rather than a guarded case (it is still isolated by the per-plugin `try/except`). The chain-dispatch shape:

```python
# Example from registry.py — dispatch_on_request (a timeout-guarded hook)
try:
    plugin_result = await asyncio.wait_for(
        plugin.on_request(
            body, user, metadata, event_emitter, task,
            valves=valves,
            request_id=request_id,
            current_result=result,
        ),
        timeout=_PR_DISPATCH_TIMEOUT,  # 30 seconds
    )
    if plugin_result is not None:
        result = plugin_result
except asyncio.TimeoutError:
    logger.warning("Plugin '%s' on_request timed out after %.0fs", ...)
except Exception:
    logger.debug("Plugin '%s' on_request failed", plugin.plugin_id, exc_info=True)
```

---

## Bundle Compatibility

The pipe ships in three modes: **package**, **flat bundle**, and **compressed bundle**.

### The Problem

In compressed bundles, `__path__` is empty, causing `pkgutil.iter_modules()` to return nothing. Plugins would silently fail to register.

### The Solution

Explicit fallback imports after `pkgutil` discovery:

```python
# plugins/__init__.py — at the bottom
from . import my_plugin as _my_plugin  # noqa: E402, F401
from . import counter as _counter  # noqa: E402, F401
```

**When adding a new plugin, always add an explicit import line.**

---

## Complete Example: Counter Plugin

A full plugin that tracks completion counts by terminal status and exposes a virtual model to view stats:

```python
# plugins/counter/plugin.py
"""Request counter plugin — tracks per-model request counts."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from ..base import PluginBase, PluginContext
from ..registry import PluginRegistry

_COUNTER_MODEL_ID = "request-counter"


@PluginRegistry.register
class CounterPlugin(PluginBase):
    plugin_id = "request-counter"
    plugin_name = "Request Counter"
    plugin_version = "1.0.0"

    hooks = {
        "on_models": 30,                # Observability priority (runs after feature plugins)
        "on_request": 30,               # Observability priority (runs after feature plugins)
        "on_generation_complete": 30,   # Observe terminal completions
    }

    def on_init(self, ctx: PluginContext, **kwargs: Any) -> None:
        self.ctx = ctx
        self.counts: dict[str, int] = defaultdict(int)
        self.total_tokens: int = 0
        self.seen_request_ids: set[str] = set()

    def on_models(self, models: list[dict[str, Any]], **kwargs: Any) -> None:
        models.append({"id": _COUNTER_MODEL_ID, "name": "Request Counter"})

    async def on_request(
        self,
        body: dict[str, Any],
        user: dict[str, Any],
        metadata: dict[str, Any],
        event_emitter: Any,
        task: Any,
        **kwargs: Any,
    ) -> dict[str, Any] | str | None:
        model = str(body.get("model", ""))
        if not model.lower().endswith(_COUNTER_MODEL_ID):
            return None  # Not our model

        # Handle OWUI background tasks
        if task:
            import json
            task_name = task if isinstance(task, str) else ""
            name = task_name.strip().lower()
            content = ""
            if "title" in name: content = json.dumps({"title": "Request Counter"})
            elif "tag" in name: content = json.dumps({"tags": ["Counter"]})
            elif "emoji" in name: content = json.dumps({"emoji": ""})
            elif "follow" in name: content = json.dumps({"follow_ups": []})
            return self.ctx.build_response(model=_COUNTER_MODEL_ID, content=content)

        # Build stats output
        lines = ["## Request Counter\n"]
        if not self.counts:
            lines.append("No completions recorded yet.")
        else:
            lines.append("| Status | Completions |")
            lines.append("| --- | ---: |")
            for status, count in sorted(
                self.counts.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"| {status} | {count:,} |")
            lines.append(f"\n**Total tokens:** {self.total_tokens:,}")

        return self.ctx.build_response(
            model=_COUNTER_MODEL_ID,
            content="\n".join(lines),
        )

    async def on_generation_complete(
        self,
        usage: dict[str, Any] | None,
        status: str,
        **kwargs: Any,
    ) -> None:
        # Observe only — count terminal completions by status and sum tokens.
        # on_generation_complete fires at least once per request (a backstop may
        # re-emit), so dedupe by request_id before counting.
        request_id = kwargs.get("request_id")
        if request_id and request_id in self.seen_request_ids:
            return
        if request_id:
            self.seen_request_ids.add(request_id)
        self.counts[status] += 1
        if isinstance(usage, dict):
            self.total_tokens += usage.get("total_tokens", 0)

    def on_shutdown(self, **kwargs: Any) -> None:
        total = sum(self.counts.values())
        self.ctx.logger.info("Counter plugin: %d total requests tracked", total)
```

---

## Pipe Internals Reference

Complete list of `Pipe` attributes accessible via `ctx.pipe`:

### Always-Initialized Subsystems

| Attribute | Type | Description |
|-----------|------|-------------|
| `valves` | `Valves` | Active configuration (100+ settings) |
| `logger` | `SessionLogger` | Structured logger |
| `_artifact_store` | `ArtifactStore` | DB + Redis artifact persistence, encryption, cleanup |
| `_circuit_breaker` | `CircuitBreaker` | Per-user failure tracking; `allows(user_id)`, `record_failure(user_id)`, `reset(user_id)` |
| `_event_emitter_handler` | `EventEmitterHandler` | Emits UI events: status, errors, citations |
| `_streaming_handler` | `StreamingHandler` | SSE streaming: delta parsing, chunk buffering |
| `_multimodal_handler` | `MultimodalHandler` | File/image operations: uploads, downloads, inline handling |
| `_session_log_manager` | `SessionLogManager` | Session log archival to disk/zip |

### Lazy-Initialized Subsystems

Use `_ensure_*()` methods — safe to call, creates on first access:

| Method | Returns | Description |
|--------|---------|-------------|
| `_ensure_catalog_manager()` | `ModelCatalogManager` | Model metadata sync from OpenRouter to Open WebUI |
| `_ensure_error_formatter()` | `ErrorFormatter` | Formats and emits error messages using templates |
| `_ensure_reasoning_config_manager()` | `ReasoningConfigManager` | Reasoning/thinking output configuration |
| `_ensure_nonstreaming_adapter()` | `NonStreamingAdapter` | Non-streaming OpenRouter requests |
| `_ensure_task_model_adapter()` | `TaskModelAdapter` | Task model (title/tags/emoji) requests |
| `_ensure_tool_executor()` | `ToolExecutor` | Tool/function call execution with retries |
| `_ensure_responses_adapter()` | `ResponsesAdapter` | OpenRouter Responses API streaming |
| `_ensure_chat_completions_adapter()` | `ChatCompletionsAdapter` | OpenAI-compatible chat completions |
| `_ensure_request_orchestrator()` | `RequestOrchestrator` | Routes requests to appropriate adapter |
| `_ensure_filter_manager()` | `FilterManager` | ORS and direct uploads filter management |
| `_ensure_plugin_registry()` | `PluginRegistry` | Plugin coordination (avoid calling from within a plugin) |

### Redis State

| Attribute | Type | Description |
|-----------|------|-------------|
| `_redis_enabled` | `bool` | Whether Redis is connected and active |
| `_redis_client` | `Optional[redis.Redis]` | Connected Redis client |
| `_redis_namespace` | `str` | Key namespace for this pipe instance |

### Request Queue

| Attribute | Type | Description |
|-----------|------|-------------|
| `_request_queue` | `Optional[asyncio.Queue]` | Pending pipe jobs |

### Class-Level

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Pipe runtime identifier |
| `name` | `str` | Display name |

---

## Testing Plugins

### Test Setup

```python
import logging
from unittest.mock import Mock
import pytest
from open_webui_openrouter_pipe.plugins.base import PluginContext
from open_webui_openrouter_pipe.plugins.registry import PluginRegistry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset class-level registry between tests."""
    original_classes = PluginRegistry._plugin_classes[:]
    original_valve_fields = dict(PluginRegistry._pending_valve_fields)
    original_user_valve_fields = dict(PluginRegistry._pending_user_valve_fields)
    PluginRegistry._plugin_classes.clear()
    PluginRegistry._pending_valve_fields.clear()
    PluginRegistry._pending_user_valve_fields.clear()
    yield
    PluginRegistry._plugin_classes.clear()
    PluginRegistry._plugin_classes.extend(original_classes)
    PluginRegistry._pending_valve_fields.clear()
    PluginRegistry._pending_valve_fields.update(original_valve_fields)
    PluginRegistry._pending_user_valve_fields.clear()
    PluginRegistry._pending_user_valve_fields.update(original_user_valve_fields)


def _make_mock_pipe():
    """Create a minimal mock Pipe for plugin tests."""
    pipe = Mock()
    pipe.id = "test-pipe"
    pipe.valves = Mock()
    pipe.valves.ENABLE_PLUGIN_SYSTEM = True
    pipe.valves.model_fields = {}
    pipe._artifact_store = Mock()
    pipe._artifact_store._session_factory = None
    pipe._artifact_store._item_model = None
    pipe._circuit_breaker = Mock()
    pipe._circuit_breaker._open_circuits = {}
    pipe._redis_client = None
    pipe._redis_enabled = False
    pipe._request_queue = None
    pipe._catalog_manager = None
    pipe._http_session = None
    return pipe


def _make_plugin_ctx(pipe=None):
    if pipe is None:
        pipe = _make_mock_pipe()
    return PluginContext(pipe=pipe, logger=logging.getLogger("test"))
```

### Testing a Plugin

```python
from my_plugin import MyPlugin

class TestMyPlugin:
    @pytest.mark.asyncio
    async def test_intercepts_own_model(self):
        pipe = _make_mock_pipe()
        ctx = _make_plugin_ctx(pipe)
        plugin = MyPlugin()
        plugin.on_init(ctx)

        body = {"model": "my-virtual-model", "messages": [{"role": "user", "content": "hello"}]}
        result = await plugin.on_request(body, {"role": "admin"}, {}, None, None)
        assert result is not None
        assert "hello" in result["choices"][0]["message"]["content"]

    @pytest.mark.asyncio
    async def test_ignores_other_models(self):
        ctx = _make_plugin_ctx()
        plugin = MyPlugin()
        plugin.on_init(ctx)

        body = {"model": "gpt-4o", "messages": []}
        result = await plugin.on_request(body, {}, {}, None, None)
        assert result is None  # Passthrough
```

---

## See Also

- [Pipe Dashboard — Operations Guide](plugins_pipe_dashboard.md) — the built-in example plugin: a virtual model that opens a live admin dashboard and configuration editor. Build and extension reference: [Pipe Dashboard Internals](plugins_pipe_dashboard_internals.md).
- [OpenRouter Fusion](openrouter_fusion.md) — the Fusion live panel documents the shared Socket.IO embed posture (same-origin requirement, inlined client, CSP guidance).
- [Valves & Configuration Atlas](valves_and_configuration_atlas.md) — authoritative configuration reference for all pipe valves.
- [Developer Guide & Architecture](developer_guide_and_architecture.md) — pipe internals, subsystem architecture, and adapter patterns.
