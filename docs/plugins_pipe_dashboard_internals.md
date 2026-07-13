# Pipe Dashboard Plugin -- Internals & Extension Reference

> How the reference Pipe Dashboard plugin is built, and how to extend it with new commands and actions. To enable and use the feature, see the [Operations Guide](plugins_pipe_dashboard.md). To build your own plugin, see the [Plugin System -- Developer Guide](plugin_system.md).


---

## Table of Contents

1. [Architecture](#architecture)
2. [Live Dashboard Transport (Socket.IO)](#live-dashboard-transport-socketio)
3. [Command Registration](#command-registration)
4. [CommandContext](#commandcontext)
5. [Command Resolution](#command-resolution)
6. [CommandEntry](#commandentry)
7. [Adding Commands](#adding-commands)
8. [Actions (Write-Side Extensions)](#actions-write-side-extensions)
9. [Authorization Helpers](#authorization-helpers)
10. [HTTP Action Route](#http-action-route)
11. [Formatting Utilities](#formatting-utilities)
12. [Running DB Queries from Commands](#running-db-queries-from-commands)
13. [Resolving Model Display Names](#resolving-model-display-names)
14. [Session Tracking & Usage Records](#session-tracking--usage-records)
15. [Testing Commands](#testing-commands)

---

## Architecture

The Pipe Dashboard plugin lives under `plugins/pipe_dashboard/` and is organized as follows:

```
plugins/pipe_dashboard/
├── __init__.py           # Re-export PipeDashboardPlugin
├── plugin.py             # PipeDashboardPlugin class (model injection + request intercept)
├── auth.py               # ACCESS_DENIED_MD message
├── authz.py              # Authorization chokepoint — can_view/can_act (reuses OWUI model access)
├── actions.py            # Action registry + dispatcher (authorize-first, audited)
├── http_routes.py        # Authenticated POST /api/pipe/dashboard/action (header-only bearer)
├── context.py            # CommandContext dataclass
├── command_registry.py   # CommandRegistry, CommandEntry, register_command
├── formatters.py         # Markdown/Mermaid output helpers
├── runtime_metrics.py      # Tiered data collectors (identity, fast, medium, slow, system resources)
├── _collectors.py        # Shared low-level collectors (concurrency, queues, breakers, sessions gauge)
├── session_tracker.py    # In-memory live-session registry (hook-fed; task-fold; recent ring)
├── usage_store.py        # Valve-gated usage-record persistence (writer thread + purge task)
├── usage_queries.py      # Usage-tab range analytics (buckets, by-model/by-user, 30s memo)
├── dashboard_socket.py       # OWUI socket.io integration (subscribe handler, viewers room, emit)
├── _socketio_client.py   # Generated: vendored socket.io client (scripts/build_pipe_dashboard_socketio.py)
├── dashboard_publisher.py    # Per-worker slice publishing + aggregation + viewer emit loop
└── commands/
    ├── __init__.py       # Auto-imports command modules
    ├── help_cmd.py       # Built-in: help
    └── dashboard_cmd.py      # Built-in: dashboard (live dashboard)
```

**Request flow:**

```
User selects "Pipe Dashboard" model in OWUI dropdown
  → types "dashboard" and sends message
    → PipeDashboardPlugin.on_request() intercepts (model ID matches)
      → Auth check: authz.can_view (OWUI read grant / owner / admin)?
        → CommandRegistry.resolve("dashboard") → (entry, args)
          → entry.handler(CommandContext(...)) → emits HTML dashboard via event_emitter
            → Dashboard opens its own OWUI socket.io connection for live updates
```

The plugin subscribes to six hooks, all at priority **50**:

| Hook | Purpose |
|------|---------|
| `on_models` | Appends `{"id": "pipe-dashboard", "name": "Pipe Dashboard"}` to the model list |
| `on_request` | Intercepts requests sent to the `pipe-dashboard` model ID; starts live-session tracking for every other request |
| `on_emitter_wrap` | Wraps the stream emitter to capture usage snapshots and tool-start events for the Live feed |
| `on_tool_result` | Records each resolved tool call's outcome on the live session |
| `on_request_retry` | Increments the live session's retry counter |
| `on_generation_complete` | Finalizes the live session and persists a usage row when collection is enabled |

---

## Live Dashboard Transport (Socket.IO)


The `dashboard` command emits an HTML shell containing an empty dashboard layout. All data is populated dynamically over Open WebUI's own authenticated Socket.IO channel, with tiered update frequencies. This is a fully dynamic dashboard — no static data is embedded in the HTML. Everything arrives as socket events.

### How It Works — The Socket.IO Transport Architecture

Open WebUI already runs a Socket.IO server at `/ws/socket.io` for its own realtime features. The dashboard rides that channel instead of registering any custom HTTP endpoint:

```
Pipe import (every worker)
  │
  ▼
dashboard_socket.register_socket_handler()
  │  → sio.on("openrouter:pipe_dashboard:sub", ...) on OWUI's Socket.IO server
  │    (idempotent; retried from on_init / on_models)
  ▼
Dashboard command ("dashboard")
  │
  ▼
Emits dashboard HTML via event_emitter (embeds → srcdoc iframe)
  │
  ▼
Dashboard JS: io(origin, {path: "/ws/socket.io", auth: {token}})
  │  token = localStorage["token"] (requires same-origin iframe — see Requirements)
  ▼
on connect → emit "user-join" → in its ACK → emit "openrouter:pipe_dashboard:sub"
  │  (the ACK ordering guarantees OWUI has registered the session first)
  ▼
Subscribe handler: session must exist in OWUI's SESSION_POOL,
  │  then sio.enter_room(sid, "pipe_dashboard_viewers")
  ▼
Publisher loop on the worker holding the socket:
  │  aggregates stats → sio.emit("openrouter:pipe_dashboard", payload,
  │                              room="pipe_dashboard_viewers", ignore_queue=True)
  ▼
Dashboard JS updates DOM sections as payloads arrive
```

**Key insights:**

- **Room membership is the entire viewer state.** Socket.IO removes a socket from `pipe_dashboard_viewers` on disconnect and deletes the empty room — no registry, no keys, no TTLs, no custom HTTP surface.
- **Rooms are per-worker-local**, so the worker that holds a viewer's socket detects it locally and emits **locally** (`ignore_queue=True`) — the stats push never needs the cross-worker socket backplane. Redis carries the stats aggregation.
- **Reconnects self-heal.** Socket.IO re-fires `connect` with a fresh sid; the client re-emits `user-join` and the subscribe, rejoining the room automatically.
- **The subscribe emit lives inside the `user-join` ACK** because OWUI's server (`always_connect=True`) confirms the connection *before* it finishes validating the token and populating `SESSION_POOL`; emitting the subscribe bare on `connect` would race that.
- The Socket.IO browser client is **inlined into the dashboard HTML** from the generated `_socketio_client.py` module (OWUI does not serve a standalone client). It is generated from the same SHA-384-pinned vendor file the Fusion feature uses: `python scripts/build_pipe_dashboard_socketio.py`.

**Event & room names** (constants in `dashboard_socket.py`):

| Constant | Value | Direction | Purpose |
|----------|-------|-----------|---------|
| `SUB_EVENT` | `openrouter:pipe_dashboard:sub` | client → server | Dashboard requests to join the viewers room (emitted inside the `user-join` ACK) |
| `DASHBOARD_EVENT` | `openrouter:pipe_dashboard` | server → client | Aggregated stats payload pushed to the room |
| `DENIED_EVENT` | `openrouter:pipe_dashboard:denied` | server → client | Sent to a socket refused the room (no read grant, or grant later revoked) |
| `VIEWERS_ROOM` | `pipe_dashboard_viewers` | — | The Socket.IO room whose membership is the entire viewer state |

### Data Tiers

Data collection is split into tiers to balance freshness against collection cost:

| Tier | Frequency | Data | Collection Cost |
|------|-----------|------|-----------------|
| **Identity** | Tick 0 + every ~16s | Version, pipe ID, worker count | Negligible |
| **Fast** | Every 2s | Concurrency, queues, rate limits, sessions, uptime, PID | Cheap (in-memory reads) |
| **Medium** | Every ~16s | Models catalog status, system health | Moderate (subsystem inspection) |
| **Slow** | Every ~60s (30s recompute floor) | Storage stats, configuration, plugins | Expensive (DB queries) |

A new viewer joining the room resets the tick counter, so the next emit carries the **full** tier set for an instant first paint — within ~2s when a worker is already streaming, or within one idle poll interval (~5s) on a cold start (no dashboards were open). The slow tier is additionally guarded by a wall-clock floor: rapid re-subscribes reuse the cached slow payload instead of re-running the storage queries.

The `runtime_metrics.py` module implements each tier as a separate collector function. Collectors read directly from pipe internals (`ctx.pipe._circuit_breaker`, `ctx.pipe._request_queue`, etc.) — they have full access via `PluginContext.pipe`.

### Multi-Worker Aggregation

In multi-worker deployments (multiple uvicorn workers behind a load balancer), each worker only sees its own process state. The dashboard uses Redis for cross-worker aggregation; every worker runs the same background task (`dashboard_publisher.py`) in one of three modes:

1. **Emitting** — this worker has local members in the `pipe_dashboard_viewers` room. It renews the `{ns}:dashboard:active` flag, writes its own slice, reads all workers' slices from Redis (guaranteeing its own is included even before its first write lands), aggregates them, merges the tiered collectors, and emits to the room. Delivery is local — the viewer's socket lives on this worker.
2. **Publishing** — no local viewers, but another worker set the active flag: write this worker's slice to `{ns}:dashboard:worker:{pid}` every 2s so the emitting worker can aggregate it. On the idle→active transition the emitter waits ~1s so freshly woken workers land their first slice before the first aggregate.
3. **Idle** — no viewers anywhere: one Redis `EXISTS` per 5s, woken instantly via the `{ns}:dashboard:wake` pub/sub channel — near-zero overhead.

In single-worker mode (no Redis), the worker with viewers emits directly from its local collectors — the same payload shape, minus the multi-worker `workers` table.

### Payload Shape

Each `openrouter:pipe_dashboard` event carries a JSON object. Keys are present only when that tier fires:

```json
{
  "tick": 5,
  "worker_count": 3,
  "concurrency": {"active_requests": 2, "max_requests": 50, "...": "..."},
  "queues": {"requests": 0, "requests_max": 1000, "...": "..."},
  "rate_limits": {"tracked_users": 3, "tripped_users": 0, "...": "..."},
  "videos": {"active": 0, "max": 4},
  "sessions": {"in_flight": 1},
  "sessions_live": [{"user": "sam", "model_id": "...", "model_name": "...", "kind": "chat", "status": "streaming", "started": 1751690000.0, "done": null, "elapsed_s": 12.3, "tokens_in": 1200, "tokens_cached": 900, "tokens_out": 80, "tools_ok": 1, "tools_failed": 0, "cost": 0.012, "task_cost": 0.0, "worker_pid": 12345}],
  "workers_rss": 1987654321,
  "system": {"cpu_pct": 12.0, "mem_used_pct": 61.0, "mem_total": 16000000000, "disk_free": 142000000000, "disk_total": 250000000000},
  "workers": [{"pid": 12345, "uptime_s": 3600.5, "last_seen_age": 0.4, "active_requests": 2, "health": {"init": 1, "wf": 0, "http": 1, "r": 1, "rss": 123456789}}],
  "uptime_s": 3600.5,
  "pid": 12345
}
```

`degraded: true` appears when a transient Redis read error made the emitter reuse the last known worker set instead of collapsing to a single-worker view. Storage payloads carry `state` (`connected` / `unavailable` / `degraded`) so the dashboard can distinguish "not initialized on this worker yet" from a genuine failure; the collector wires the shared DB itself on first use, and by-type/by-model "Least/Most recent" columns are access times (the retention sweep touches `created_at` on every read).

On tick 0, all tiers fire simultaneously for instant dashboard population. The JavaScript checks key existence and updates only the sections whose data arrived in that tick. The payload arrives raw — direct custom emits do not use the `{chat_id, message_id, data}` envelope of OWUI's shared `events` channel, so no client-side filtering is needed.

### Access Model & Live-Mode Requirements

There are no capability keys — access rides Open WebUI's own model access control (`check_model_access` for read, OWUI's own model write-formula for write). The pipe composes no access logic of its own:

1. **The command, the live feed, and read-only actions require a *read* grant (viewer).** The `dashboard` command handler and every socket subscribe resolve a fresh `UserModel` and gate on `authz.can_view` → OWUI's `check_model_access` (honoring owner, admin, direct-user grant, group grant, `user:*` public, and `BYPASS_MODEL_ACCESS_CONTROL`). An ungranted socket is refused the viewers room and told so via a `denied` event; the publisher periodically re-checks each local viewer and evicts any whose grant was revoked, so revocation takes effect without a reconnect.
2. **State-changing actions require a *write* grant (operator).** Actions are invoked over an authenticated `POST /api/pipe/dashboard/action` route and gated on `authz.can_act` → OWUI's own model write-formula (`admin` + `BYPASS_ADMIN_ACCESS_CONTROL` / owner / `has_access(..., "write")`). Classify each action by **effect**: side-effect-free introspection is `read`; anything mutating shared state (clear a cache, trigger an update) is `write` — the same consume-vs-mutate rule OWUI applies across models, KBs, tools, and channels. `register_action` defaults to `write` (fail-restrictive). OWUI's editor pairs a read grant with every write grant, so an operator can always view.
3. **The action route is CSRF-safe unconditionally.** Its authentication is **header-only** (`Authorization: Bearer <token>`, reusing OWUI's `decode_token` + `is_valid_token` + a fresh `Users.get_user_by_id`); it never reads the session cookie, so a cross-origin page cannot forge a call with the victim's token regardless of OWUI's CORS/SameSite settings. Every outcome is audited (user, action, outcome, client IP; write args included).
4. **Live mode requires the same-origin iframe setting.** The dashboard reads the session token from `localStorage`, which only works when Open WebUI's **Settings → Interface → "iframe sandbox allow same origin"** is enabled — the identical requirement as the [OpenRouter Fusion live panel](openrouter_fusion.md). With it off, the dashboard renders statically with a notice and opens no socket. If a restrictive `IFRAME_CSP` is configured, the same policy documented for Fusion applies (`script-src 'unsafe-inline'` + `connect-src 'self'`).

The payload contains only aggregate operational counters (concurrency, queue depths, breaker trip counts, uptime, worker PIDs) — no per-user data, no chat content, no secrets. The connection bar includes **Disconnect** / **Connect** buttons: disconnecting tears down the socket (Socket.IO removes it from the viewers room server-side automatically); reconnecting re-runs the connect → `user-join` → subscribe sequence. When the last viewer disconnects, all workers return to idle within seconds.

---

## Command Registration

Commands are registered using the `@register_command` decorator, which is a convenience alias for `CommandRegistry.register`:

```python
# plugins/pipe_dashboard/commands/my_cmd.py
from ..command_registry import register_command
from ..context import CommandContext


@register_command(
    "mycommand",
    summary="Short description for help listing",
    category="General",
    usage="mycommand [args]",
    aliases=["mc"],
)
async def handle_mycommand(ctx: CommandContext) -> str:
    """Longer docstring (not shown in help)."""
    pipe = ctx.pipe
    args = ctx.args      # Remaining text after command prefix
    user = ctx.user      # Open WebUI user dict
    metadata = ctx.metadata

    return "## My Command Output\n\nHello!"
```

**Decorator parameters:**

| Parameter | Type | Required | Description |
|-----------|------|:--------:|-------------|
| `name` | `str` | Yes | Primary command name (lowercased on registration) |
| `summary` | `str` | No | One-line description shown in `help` output |
| `usage` | `str` | No | Usage pattern shown in help (e.g., `"mycommand [args]"`) |
| `category` | `str` | No | Grouping for the help listing (default: `"General"`) |
| `aliases` | `list[str]` | No | Alternative names that also resolve to this command |

The handler function must be `async` and return a markdown string. The return value is wrapped in a `chat.completion` response dict by the plugin.

---

## CommandContext

Every command handler receives a `CommandContext` instance:

```python
@dataclass
class CommandContext:
    pipe: Pipe                    # Full pipe reference
    args: str                     # Remaining args after command match
    user: dict[str, Any]          # __user__ dict from Open WebUI
    metadata: dict[str, Any]      # __metadata__ dict from Open WebUI
    event_emitter: Any = None     # OWUI event emitter for HTML embeds
```

| Field | Description |
|-------|-------------|
| `pipe` | The full `Pipe` instance. Provides access to `pipe.valves`, `pipe._artifact_store`, `pipe._circuit_breaker`, and all `_ensure_*()` lazy subsystems -- dig into anything. |
| `args` | The text remaining after the command prefix was matched. For input `"dashboard extra"` matched against command `"dashboard"`, `args` is `"extra"`. |
| `user` | The Open WebUI user dict containing `role`, `id`, `name`, `email`. Gated by OWUI model access (`can_view`) before dispatch, so a handler runs only for a user granted at least read (viewer) on the pipe-dashboard model -- not necessarily an admin. |
| `metadata` | The Open WebUI request metadata dict. |
| `event_emitter` | The OWUI event emitter callable for rich UI embeds (HTML iframes). Used by the `dashboard` command to emit the dashboard shell. |

### `emit_html(html)`

`CommandContext` exposes one async helper that wraps `event_emitter` for the common case of rendering an HTML panel in the chat. It emits an `embeds` event so Open WebUI renders the string inside a sandboxed iframe. When `event_emitter` is `None` (as in unit tests) it is a safe no-op, so a handler can still return its markdown fallback:

```python
async def handle_panel(ctx: CommandContext) -> str:
    await ctx.emit_html("<h3>Hello from a panel</h3>")
    return "Panel rendered above."  # chat-bubble fallback text
```

The `dashboard` command uses exactly this pattern: it calls `emit_html` with the dashboard shell, then returns a short line telling the user to enable iframe embeds if no panel appears.

---

## Command Resolution

The `CommandRegistry.resolve(text)` method uses **longest-prefix matching**. Command names and aliases are **lowercased** on both registration and resolution (case-insensitive matching, but remaining args preserve original casing).

```
Input: "dashboard extra"
  ↓
Tries: "dashboard extra" → no match
       "dashboard" → MATCH (entry: "dashboard", args: "extra")
```

This allows multi-word commands to coexist with shorter commands. Longest-prefix matching ensures the most specific command is preferred.

**Resolution algorithm:**

```python
# Simplified from command_registry.py
for cmd_name, entry in cls._commands.items():
    if normalized == cmd_name or normalized.startswith(cmd_name + " "):
        if len(cmd_name) > best_len:
            best_entry = entry
            best_len = len(cmd_name)
```

If no command matches, `resolve()` returns `(None, "")` and the plugin responds with an "Unknown command" message suggesting `help`.

---

## CommandEntry

Each registered command is stored as a `CommandEntry` dataclass:

```python
@dataclass
class CommandEntry:
    name: str                     # Primary command name
    handler: CommandHandler        # async (CommandContext) -> str
    summary: str                  # One-line description
    usage: str                    # Usage pattern for help
    category: str                 # Grouping (General, Diagnostics, etc.)
    aliases: list[str]            # Alternative names
```

`CommandHandler` is typed as `Callable[[CommandContext], Awaitable[str]]`.

The `CommandRegistry` stores entries in a class-level dict `_commands: dict[str, CommandEntry]`. Both the primary name and all aliases are keyed in this dict (lowercased), pointing to the same `CommandEntry` instance.

---

## Adding Commands

### Step 1: Create the command file

```python
# plugins/pipe_dashboard/commands/my_cmd.py
from __future__ import annotations
from ..command_registry import register_command
from ..context import CommandContext


@register_command("mycommand", summary="Do something", category="General")
async def handle_mycommand(ctx: CommandContext) -> str:
    return "## My Command\n\nDone."
```

### Step 2: Add explicit import

Add an import line in `plugins/pipe_dashboard/commands/__init__.py` for bundle compatibility (compressed bundles cannot use `pkgutil` auto-discovery):

```python
from . import my_cmd as _my_cmd  # noqa: E402, F401
```

### Multi-word commands

Commands with spaces are supported. Register them with the full name:

```python
@register_command("mycommand details", summary="Show details", category="General")
async def handle_mycommand_details(ctx: CommandContext) -> str:
    return "## Details\n\n..."
```

Both `mycommand` and `mycommand details` can coexist. Longest-prefix matching ensures `mycommand details` is preferred when the input starts with those two words.

---

## Actions (Write-Side Extensions)

Commands render read-only panels in the chat. **Actions** are the write-side surface: short, authorized JSON calls the live dashboard makes over an authenticated HTTP route (panel buttons, the Usage tab's data fetch). Each action is a small async function registered in `actions.py`; the dispatcher authorizes it, validates its arguments, rate-limits it, runs it, and audits every outcome.

Register an action with the `@register_action` decorator:

```python
from open_webui_openrouter_pipe.plugins.pipe_dashboard.actions import register_action


@register_action("cache_clear", permission="write", schema={"scope": str})
async def _cache_clear(pipe, user, args):
    scope = args["scope"]
    # ... mutate shared state ...
    return {"cleared": scope}
```

**Decorator parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | — | Action name the client sends in the request body |
| `permission` | `str` | `"write"` | `"read"` (side-effect-free introspection, gated on `can_view`) or `"write"` (mutates shared state, gated on `can_act`). Defaults to `write` — fail-restrictive |
| `schema` | `dict[str, type] \| None` | `None` | Per-key type check on `args`; `None` skips validation |

Each registered action is stored as an `ActionEntry`:

```python
@dataclass
class ActionEntry:
    name: str
    permission: str                                    # "read" | "write"
    schema: dict[str, type] | None
    handler: Callable[..., Awaitable[dict[str, Any]]]  # async (pipe, user, args) -> dict
```

The handler receives the pipe, the resolved OWUI user, and the validated `args` dict, and returns a JSON-serializable dict.

**Dispatch pipeline.** `dispatch_action(pipe, user, name, args, *, client_ip=None)` runs each call through a fixed sequence and returns a `(status_code, envelope)` tuple:

1. **Authorize** -- `can_act` for a write action, `can_view` for a read action (an unknown action is checked as read).
2. **Resolve** -- look the action up in the registry.
3. **Validate** -- type-check `args` against the schema.
4. **Rate-limit** -- one call per `(user, action)` per second.
5. **Run + audit** -- invoke the handler and audit the terminal outcome (write args are included in the audit line, read args are not).

| Status | Envelope | When |
|--------|----------|------|
| `200` | `{"ok": true, "result": <dict>}` | Handler succeeded |
| `400` | `{"error": "missing or invalid: <key>"}` | `args` failed schema validation |
| `403` | `{"error": "forbidden"}` | Caller lacks the required grant |
| `404` | `{"error": "unknown action"}` | No action registered under that name |
| `429` | `{"error": "rate limited"}` | Second call within the 1 s per-`(user, action)` window |
| `500` | `{"error": "action failed"}` | Handler raised |

**Built-in actions:**

| Action | Permission | Schema | Returns |
|--------|:----------:|--------|---------|
| `whoami` | `read` | — | `{user_id, role, can_view, can_act}` |
| `echo` | `write` | `{"message": str}` | `{"message": <echoed>}` |
| `usage_stats` | `read` | `{"range": str, "tz_offset_min": int, "include_tasks": bool}` | The Usage-tab analytics dict (see [Session Tracking & Usage Records](#session-tracking--usage-records)) |
| `config_get` | `read` | — | Valve specs with current values (secrets masked to a set/not-set flag) and the current config revision |
| `config_set` | `write` | `{"edits": dict}` | Persists the changed subset and returns the new revision — or a conflict payload when the client's revision was stale |

`whoami` and `echo` are reference implementations; `usage_stats` powers the Usage tab; `config_get` and `config_set` power the Config tab (see the [Operations Guide](plugins_pipe_dashboard.md#editing-configuration)). A new action is registered by importing its module at plugin load -- the same explicit-import requirement as commands.

---

## Authorization Helpers

`authz.py` is the single authorization chokepoint. It composes no access logic of its own -- every decision delegates to Open WebUI's own model access control, so the dashboard inherits the grants an admin sets in the model's Access editor. Reuse these helpers instead of writing role checks:

| Function | Signature | Semantics |
|----------|-----------|-----------|
| `model_id(pipe)` | `(pipe) -> str \| None` | The OWUI model id for this overlay, `"{pipe.id}.pipe-dashboard"` (or `None` when the pipe has no id) |
| `resolve_user(user_id)` | `async (str \| None) -> UserModel \| None` | Load the OWUI `UserModel` by id; `None` on any failure |
| `can_view(user, pipe)` | `async (user, pipe) -> bool` | **Read** grant -- `check_model_access` (honors owner, admin, direct/group grant, `user:*` public, `BYPASS_MODEL_ACCESS_CONTROL`) |
| `can_act(user, pipe)` | `async (user, pipe) -> bool` | **Write** grant -- admin + `BYPASS_ADMIN_ACCESS_CONTROL`, owner, or a `write` access grant |

A handler that needs to branch on the caller's grant resolves the user first, then asks:

```python
from open_webui_openrouter_pipe.plugins.pipe_dashboard.authz import can_act, resolve_user


async def handle_maybe_privileged(ctx: CommandContext) -> str:
    user = await resolve_user(ctx.user.get("id"))
    if await can_act(user, ctx.pipe):
        return "You are an operator."
    return "You are a viewer."
```

Classify by **effect**: side-effect-free introspection is `read` (viewer); anything mutating shared state is `write` (operator) -- the same consume-vs-mutate rule OWUI applies across models, KBs, tools, and channels.

---

## HTTP Action Route

Actions are invoked over one authenticated route registered on Open WebUI's own FastAPI app (ahead of the SPA catch-all), not over the Socket.IO channel:

| Property | Value |
|----------|-------|
| Method + path | `POST /api/pipe/dashboard/action` |
| Auth | Header-only `Authorization: Bearer <token>` -- reuses OWUI's `decode_token` + `is_valid_token` + a fresh `Users.get_user_by_id`; the session cookie is never read, so the route is CSRF-safe regardless of OWUI's CORS/SameSite settings |
| Request body | `{"action": <str>, "args": <object>}` |
| Response | The `dispatch_action` envelope, at its status code |

The route adds two guards in front of the dispatcher: a `401` when the bearer token is absent, malformed, invalid, or maps to a role outside `{user, admin}`, and a coarse per-user `429` (one request per 0.25 s) ahead of the per-action rate limit. The dashboard calls it from `callAction(...)` in the shell, forwarding the same `localStorage` token it uses for the socket.

---

## Formatting Utilities

The `formatters` module provides helpers for producing consistent command output. Import from the relative path within the `pipe_dashboard` package:

```python
from ..formatters import (
    markdown_table,        # pipe-delimited markdown table
    format_bytes,          # 2_621_440 -> "2.5 MB"
    format_duration,       # 125 -> "2.1m"
    format_number,         # 1234567 -> "1,234,567"
    format_ago,            # unix ts -> "5m ago" / "never"
    format_datetime,       # datetime -> "2026-07-05 16:30" / "-"
    humanize_type,         # "function_call" -> "Function Call"
    mask_sensitive,        # "sk-...xyz" -> "***3xyz"
    collapsible,           # summary + body -> <details> block
    mermaid_pie,           # title + data -> mermaid pie block
    mermaid_bar,           # xychart-beta bar block
    build_model_name_map,  # {id_variant: display_name}
    resolve_model_name,    # (model_id, name_map) -> display name
)
```

Every helper below is pure and side-effect-free, so it is safe to call from a synchronous section of a handler.

### Function signatures

**`markdown_table(headers, rows)`** -- Build a pipe-delimited markdown table. Pipe characters in cell values are auto-escaped.

```python
markdown_table(["Model", "Requests"], [["gpt-4o", "42"], ["claude-3", "17"]])
# | Model | Requests |
# | --- | --- |
# | gpt-4o | 42 |
# | claude-3 | 17 |
```

**`format_bytes(n)`** -- Human-readable byte count.

```python
format_bytes(0)           # "0 B"
format_bytes(1536)        # "1.5 KB"
format_bytes(2_621_440)   # "2.5 MB"
format_bytes(5_368_709_120)  # "5.0 GB"
```

**`format_duration(seconds)`** -- Human-readable duration.

```python
format_duration(5.2)    # "5.2s"
format_duration(125)    # "2.1m"
format_duration(7200)   # "2.0h"
```

**`humanize_type(raw_type)`** -- Convert snake_case artifact types to display labels. Uses a built-in lookup table with fallback to `str.title()`.

```python
humanize_type("function_call")         # "Function Call"
humanize_type("web_search_call")       # "Web Search"
humanize_type("image_generation_call") # "Image Generation"
```

**`mask_sensitive(value, visible_chars=4)`** -- Mask secrets, showing only the last N characters.

```python
mask_sensitive("sk-or-v1-abc123xyz")  # "***3xyz"
mask_sensitive("short")               # "***hort"
mask_sensitive("ab")                  # "***"
```

**`collapsible(summary, content)`** -- Wrap content in an HTML `<details>` block. The summary text is HTML-escaped.

```python
collapsible("Click to expand", "Hidden content here")
# <details>
# <summary>Click to expand</summary>
#
# Hidden content here
#
# </details>
```

**`mermaid_pie(title, data)`** -- Generate a Mermaid pie chart code block.

```python
mermaid_pie("Usage by Model", {"GPT-4o": 42, "Claude 3": 17})
# ```mermaid
# pie title Usage by Model
#     "GPT-4o" : 42
#     "Claude 3" : 17
# ```
```

**`mermaid_bar(title, x_label, y_label, categories, values)`** -- Generate a Mermaid xychart-beta bar chart.

```python
mermaid_bar("Requests", "Model", "Count", ["GPT-4o", "Claude"], [42, 17])
# ```mermaid
# xychart-beta
#     title "Requests"
#     x-axis Model ["GPT-4o", "Claude"]
#     y-axis "Count"
#     bar [42, 17]
# ```
```

**`format_number(n)`** -- Comma-separated number; floats render with one decimal place.

```python
format_number(1234567)   # "1,234,567"
format_number(1234.5)    # "1,234.5"
```

**`format_ago(ts)`** -- Human-readable "time ago" from a Unix timestamp. `0` or a negative value renders `"never"`.

```python
format_ago(0)                 # "never"
format_ago(time.time() - 90)  # "1m ago"
```

**`format_datetime(dt)`** -- Short `YYYY-MM-DD HH:MM` display for a datetime (falls back to the first 16 characters of `str(dt)`). `None` renders `"-"`.

```python
format_datetime(None)  # "-"
# datetime(2026, 7, 5, 16, 30) -> "2026-07-05 16:30"
```

**`build_model_name_map()`** -- Build a `{id_variant: display_name}` map across every known model ID form (`id`, `norm_id`, `original_id`). Pair it with `resolve_model_name` (see [Resolving Model Display Names](#resolving-model-display-names)) to label stored IDs; it returns an empty map if the registry is unavailable.

```python
name_map = build_model_name_map()
resolve_model_name("openai/gpt-4o", name_map)  # "GPT-4o" (or the raw ID if unknown)
```

---

## Running DB Queries from Commands

Commands that need database access (e.g., storage stats) must use `run_in_threadpool` to avoid blocking the async event loop. Access the artifact store's SQLAlchemy session factory through `ctx.pipe._artifact_store`:

```python
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import func

async def handle_my_storage_cmd(ctx: CommandContext) -> str:
    store = ctx.pipe._artifact_store
    session_factory = getattr(store, "_session_factory", None)
    item_model = getattr(store, "_item_model", None)

    if session_factory is None or item_model is None:
        return "Artifact store not initialized."

    def _query():
        from open_webui_openrouter_pipe.storage.persistence import _db_session
        with _db_session(session_factory) as session:
            total = session.query(func.count(item_model.id)).scalar() or 0
            return total

    total = await run_in_threadpool(_query)
    return f"**Total artifacts:** {total:,}"
```

**Key points:**
- The `_db_session` context manager handles connection lifecycle (open, commit/rollback, close).
- Always check that `_session_factory` and `_item_model` are not `None` -- the artifact store may not be initialized in all environments.
- Wrap the synchronous SQLAlchemy query in a plain `def` and run it via `run_in_threadpool`.

---

## Resolving Model Display Names

Commands that display model information (e.g., usage stats) often need to map raw model IDs (like `openai/gpt-4o`) to human-readable display names (like `GPT-4o`). Use the `OpenRouterModelRegistry`:

```python
from open_webui_openrouter_pipe.models.registry import OpenRouterModelRegistry

def _build_model_name_map() -> dict[str, str]:
    """Map model IDs to display names."""
    id_to_name: dict[str, str] = {}
    for m in OpenRouterModelRegistry.list_models():
        name = m.get("name", "")
        if not name:
            continue
        for key in ("id", "norm_id", "original_id"):
            mid = m.get(key)
            if mid:
                id_to_name[mid] = name
    return id_to_name

# Usage in a command handler:
# name_map = _build_model_name_map()
# display = name_map.get(stored_model_id, stored_model_id)
```

This maps all known ID variants (`id`, `norm_id`, `original_id`) to the same display name, so lookups work regardless of which ID form was stored.

---

## Session Tracking & Usage Records

The Live and Usage tabs are backed by two layers: an in-memory `SessionTracker` (live rows) and a valve-gated `UsageStore` (historical rows, queried by `usage_queries`).

**Live session rows.** `SessionTracker.live_sessions()` returns the rows the Live tab renders and the publisher ships as `sessions_live`. Each row:

| Field | Type | Notes |
|-------|------|-------|
| `user` | str | User name (or email, or `?`) |
| `model_id` | str | Raw model id |
| `model_name` | str | Server-resolved display name |
| `kind` | str | `chat` or `task` |
| `status` | str | `queued` / `streaming` / `tool:<name>` / `completed` / `failed` / `cancelled` |
| `started`, `done` | float / null | Unix timestamps; `done` is `null` while in flight |
| `elapsed_s` | float | Seconds since `started`, frozen at `done` |
| `tokens_in`, `tokens_cached`, `tokens_out` | int | Cumulative token counts |
| `tools_ok`, `tools_failed` | int | Resolved tool-call outcomes |
| `cost`, `task_cost` | float | Running cost; `task_cost` is the folded-in task portion |
| `worker_pid` | int | The worker that owns the row |

**Usage records.** With `PIPE_DASHBOARD_USAGE_COLLECT` on, each finalized session is mapped by `SessionTracker.db_row(...)` and written to the `dashboard_{suffix}` table. The columns (`USAGE_ROW_FIELDS`, plus a generated `id`):

| Column | Type | Meaning |
|--------|------|---------|
| `id` | str(26) | Generated primary key |
| `ts` | datetime | Completion time (indexed) |
| `started_at` | datetime | Request start |
| `kind` | str(8) | `chat` or `task` (indexed) |
| `user_id`, `user_name` | str | Caller identity (`user_id` indexed) |
| `chat_id`, `session_id` | str | Conversation identifiers (`chat_id` indexed) |
| `model_id` | str(128) | Raw model id (indexed) |
| `task_name` | str(32) / null | Task type for `kind="task"` rows |
| `status` | str(12) | `ok` / `failed` / `cancelled` |
| `duration_ms` | int | Wall-clock duration |
| `tokens_in`, `tokens_out`, `tokens_reasoning`, `tokens_cached` | int | Token counts |
| `tools_ok`, `tools_failed`, `retries` | int | Per-request counters |
| `cost`, `cache_savings` | float | Billed cost and estimated cache savings |
| `worker_pid` | int | Writing worker |

**Range analytics.** The `usage_stats` action calls `run_usage_query(plugin, pipe, args)`, which validates the range, memoizes for 30 s, and runs one windowed aggregation in the store's DB executor. The `args` keys are `range` (default `"24h"`), `include_tasks` (default `True`), and `tz_offset_min` (default `0`, clamped to ±900). Supported ranges (`USAGE_RANGES`) and their bucket sizes:

| Range | Span | Bucket |
|-------|------|--------|
| `1h` | 1 hour | 5 min |
| `6h` | 6 hours | 15 min |
| `24h` | 24 hours | 1 hour |
| `7d` | 7 days | 6 hours |
| `30d` | 30 days | 1 day |

On success the result is `{"available": true, "cards", "prev", "buckets", "by_model", "by_user", "totals", "meta"}`. When it cannot answer it returns `{"available": false, "reason": ...}` with one of:

| `reason` | Cause |
|----------|-------|
| `unknown range` | `range` is not one of `USAGE_RANGES` |
| `range exceeds retention` | The window is longer than `PIPE_DASHBOARD_USAGE_RETENTION_DAYS` |
| `storage unavailable` | The artifact store or usage table is not ready on this worker |
| `plugin unavailable` | The pipe-dashboard plugin instance could not be located |

---

## Testing Commands

### Test Setup

Use the same mock infrastructure as plugin tests. The key fixtures reset both the `PluginRegistry` and the `CommandRegistry` between tests:

```python
import logging
from unittest.mock import Mock
import pytest
from open_webui_openrouter_pipe.plugins.base import PluginContext
from open_webui_openrouter_pipe.plugins.registry import PluginRegistry
from open_webui_openrouter_pipe.plugins.pipe_dashboard.command_registry import CommandRegistry


@pytest.fixture(autouse=True)
def _clean_registries():
    """Reset registries between tests to avoid cross-contamination."""
    original_plugins = PluginRegistry._plugin_classes[:]
    original_valve_fields = dict(PluginRegistry._pending_valve_fields)
    original_user_valve_fields = dict(PluginRegistry._pending_user_valve_fields)
    original_commands = dict(CommandRegistry._commands)
    yield
    PluginRegistry._plugin_classes.clear()
    PluginRegistry._plugin_classes.extend(original_plugins)
    PluginRegistry._pending_valve_fields.clear()
    PluginRegistry._pending_valve_fields.update(original_valve_fields)
    PluginRegistry._pending_user_valve_fields.clear()
    PluginRegistry._pending_user_valve_fields.update(original_user_valve_fields)
    CommandRegistry._commands = original_commands


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
    pipe._circuit_breaker._threshold = 5
    pipe._circuit_breaker._window_seconds = 60.0
    pipe._circuit_breaker._breaker_records = {}
    pipe._circuit_breaker._tool_breakers = {}
    pipe._active_pipes_calls = 0
    pipe._video_global_semaphore = None
    pipe._video_global_limit = 0
    pipe._video_active_tasks = {}
    pipe._redis_client = None
    pipe._redis_enabled = False
    pipe._request_queue = None
    pipe._catalog_manager = None
    pipe._http_session = None
    return pipe
```

### Writing Command Tests

Test commands by constructing a `CommandContext` with a mock pipe, then calling the handler function directly:

```python
from open_webui_openrouter_pipe.plugins.pipe_dashboard.context import CommandContext

class TestMyCommand:
    @pytest.mark.asyncio
    async def test_mycommand_output(self):
        pipe = _make_mock_pipe()
        ctx = CommandContext(pipe=pipe, args="", user={"role": "admin"}, metadata={})

        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.my_cmd import handle_mycommand
        result = await handle_mycommand(ctx)
        assert "My Command" in result

    @pytest.mark.asyncio
    async def test_mycommand_with_args(self):
        pipe = _make_mock_pipe()
        ctx = CommandContext(pipe=pipe, args="--verbose", user={"role": "admin"}, metadata={})

        from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.my_cmd import handle_mycommand
        result = await handle_mycommand(ctx)
        assert isinstance(result, str)
```

**Testing tips:**
- Import the handler function directly, not via `CommandRegistry.resolve()`. This isolates the test to the handler logic.
- To test command resolution, use `CommandRegistry.resolve("mycommand args")` and assert on the returned `(entry, remaining_args)` tuple.
- The `_clean_registries` fixture prevents command registration from leaking between test files.

---

## See Also

- [Pipe Dashboard -- Operations Guide](plugins_pipe_dashboard.md) -- How to enable, access, and use the dashboard.
- [Plugin System -- Developer Guide](plugin_system.md) -- Hook system reference, plugin lifecycle, `PluginContext`, priority system, and general plugin development patterns.
- [OpenRouter Fusion](openrouter_fusion.md) -- The Fusion live panel shares the identical Socket.IO transport posture (same-origin requirement, inlined client, CSP guidance).
