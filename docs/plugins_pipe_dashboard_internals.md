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
├── usage_store.py        # Valve-gated usage-record persistence (writer thread + purge task; neither returns after signal_stop())
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
        → Background task (title/tags/emoji/follow-ups)? → the task's stub, granted viewers only
        → CommandRegistry.resolve("dashboard") → (entry, args)
          → entry.handler(CommandContext(...)) → emits HTML dashboard via event_emitter
            → Dashboard opens its own OWUI socket.io connection for live updates
```

The plugin subscribes to six hooks, all at priority **50**:

| Hook | Purpose |
|------|---------|
| `on_models` | Appends `{"id": "pipe-dashboard", "name": "Pipe Dashboard"}` to the model list, and ensures the row behind it exists. The hook **owns** the row's `name` and the whole of `meta`: every `meta` column is written from the row being updated, with only `meta.description` replaced, and every column outside `meta` — including `is_active`, `base_model_id` and `params` — is borrowed from a read taken immediately before the write, and the borrow is a point-in-time read, so a change landing after it is still lost. Open WebUI's `update_model_by_id` writes a whole row, so there is no partial write; keeping a borrowed column safe means writing a *current* value, not omitting it |
| `on_request` | Intercepts requests sent to the `pipe-dashboard` model ID; starts live-session tracking for every other request |
| `on_emitter_wrap` | Wraps the stream emitter to capture usage snapshots and tool-start events for the Live feed |
| `on_tool_result` | Records on the live session the outcome of each tool call the pipe runs in a batch |
| `on_request_retry` | Increments the live session's retry counter |
| `on_generation_complete` | Finalizes the live session and persists a usage row when collection is enabled |

**A Fusion chat in a channel loses the answer panel.** `streaming_core.py` records the Fusion answer item into `terminal_output` *before* it applies the `<details type="fusion_answer">` wrapper to `assistant_message`, so the `_is_channel_chat` guard routes a channel through `output`, which by construction holds the **unwrapped** answer. A Fusion channel row therefore renders the plain-text reconstruction, not the collapsible panel. `__channel_emitter__` has branches only for `chat:completion`, `response:completion`, `files`/`chat:message:files` and `chat:message:error`, so `fusion:event` and `embeds` reach no branch at all.

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
Dashboard JS: io(origin, {path: "/ws/socket.io", auth: fn -> {token}})
  │  token read from localStorage at each handshake (requires same-origin iframe
  │  — see Requirements); a value captured once would replay a rotated token
  ▼
on connect → emit "user-join" → in its ACK → emit "openrouter:pipe_dashboard:sub"
  │  (the ACK ordering guarantees OWUI has registered the session first)
  ▼
Subscribe handler: resolve the viewer, then sio.enter_room(sid,
  │  "pipe_dashboard_viewers") and pin the authorized id to the socket.io session
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
- **Viewer identity is pinned to the socket, not to OWUI's heartbeat bookkeeping.** OWUI identifies a socket through `SESSION_POOL`, which its reaper deletes after `SESSION_POOL_TIMEOUT` seconds without a heartbeat — while the socket is still connected and still in the viewers room. Resolving identity from that store alone made a live administrator indistinguishable from an anonymous socket, and the reauthorization sweep evicted them under a message claiming their access had been revoked. The subscribe handler therefore writes the authorized user id into the socket.io session under its own namespaced key, and identity resolution reads that first with `SESSION_POOL` as fallback. The session belongs to python-socketio rather than to Open WebUI, so the store exists regardless of Open WebUI version, and it is destroyed with the connection.
- **The panel heartbeats every 30 seconds**, matching Open WebUI's own frontend, so `SESSION_POOL` stays populated and Open WebUI stops logging the panel as an orphaned session. Identity no longer depends on it.
- **A replayed panel does not auto-connect.** Open WebUI persists the dashboard HTML into the chat message and re-runs it on every later chat load. The shell records in `localStorage`, under the panel's own id, when this browser first saw that panel, and dials out only while that record is fresh; an older panel opens in the disconnected state with its Connect button showing, so scrolling back through history cannot silently pin the publisher's emit loop. Both the write and the comparison happen in the browser: a server-rendered timestamp measured against the client clock fails silently in both directions under host skew, disconnecting a brand-new panel when the client runs fast and disabling the bound entirely when it runs slow.
- **A denial hangs up before offering reconnect.** Revocation removes the socket from the room but leaves it connected, and the Socket.IO client's `connect()` is a no-op on a connected socket — so the panel disconnects first, then shows the Connect button. Retrying is safe: the subscribe handler re-adjudicates from scratch.
- **The subscribe emit lives inside the `user-join` ACK** because OWUI's server (`always_connect=True`) confirms the connection *before* it finishes validating the token and populating `SESSION_POOL`; emitting the subscribe bare on `connect` would race that.
- The Socket.IO browser client is **inlined into the dashboard HTML** from the generated `_socketio_client.py` module (OWUI does not serve a standalone client). It is generated from the same SHA-384-pinned vendor file the Fusion feature uses: `python scripts/build_pipe_dashboard_socketio.py`.

**Event & room names** (constants in `dashboard_socket.py`):

| Constant | Value | Direction | Purpose |
|----------|-------|-----------|---------|
| `SUB_EVENT` | `openrouter:pipe_dashboard:sub` | client → server | Dashboard requests to join the viewers room (emitted inside the `user-join` ACK) |
| `DASHBOARD_EVENT` | `openrouter:pipe_dashboard` | server → client | Aggregated stats payload pushed to the room |
| `DENIED_EVENT` | `openrouter:pipe_dashboard:denied` | server → client | Sent to a socket refused the room (no read grant, or grant later revoked); the refusal also removes the socket from the room, whichever of the two refusals sent it |
| `VIEWERS_ROOM` | `pipe_dashboard_viewers` | — | The Socket.IO room whose membership is the entire viewer state |

### Data Tiers

Data collection is split into tiers to balance freshness against collection cost:

| Tier | Frequency | Data | Collection Cost |
|------|-----------|------|-----------------|
| **Identity** | Tick 0 + every ~16s | Version, pipe ID, worker count | Negligible |
| **Fast** | Every 2s | Concurrency, queues, rate limits, sessions, uptime, the emitting worker's PID | Cheap (in-memory reads) |
| **Medium** | Every ~16s | Models catalog status, system health | Moderate (subsystem inspection) |
| **Slow** | Every ~60s (30s recompute floor) | Storage stats, configuration, plugins | Expensive (DB queries), run on the artifact store's DB thread pool rather than the request loop; on a publisher-owned single-worker thread when the store has no pool (closed by `close()`, or a build that keeps failing) |

A new viewer joining the room resets the tick counter, so the next emit carries the **full** tier set for an instant first paint — within ~2s when a worker is already streaming, or within one idle poll interval (~5s) on a cold start (no dashboards were open). The slow tier is additionally guarded by a wall-clock floor: rapid re-subscribes reuse the cached slow payload instead of re-running the storage queries.

**The emit gate's cost.** `emit_dashboard` runs the persisted-row read on the live-feed cadence, so a worker with a dashboard open pays one extra `Functions` primary-key read per tick (one row, no join, no scan) on top of the tier work above, and it is paid before the emit rather than after it — a database slow to answer delays the tick rather than dropping it. That is the trade this fix makes deliberately: the alternative is a gate answered from a value the operator cannot reach. To measure it on a live deployment, time `emit_dashboard` with the dashboard open against the same worker with it closed (the closed path returns before the read), over at least a minute of ticks, and compare the two means against the tick interval — the read is only worth caching if it is not comfortably inside it. The follow-up if it is not is a `function.valves_updated`-driven cache of the switch, which is its own item; no cache is built here, because a cache is a new subsystem and a field read is the thing being corrected.

The `runtime_metrics.py` module implements each tier as a separate collector function. Collectors read directly from pipe internals (`ctx.pipe._circuit_breaker`, `ctx.pipe._request_queue`, etc.) — they have full access via `PluginContext.pipe`.

### Multi-Worker Aggregation

In multi-worker deployments (multiple uvicorn workers behind a load balancer), each worker only sees its own process state. The dashboard uses Redis for cross-worker aggregation; every worker runs the same background task (`dashboard_publisher.py`) in one of three modes:

1. **Emitting** — this worker has local members in the `pipe_dashboard_viewers` room. It renews the `{ns}:dashboard:active` flag, writes its own slice, reads each distinct worker slice from Redis once (guaranteeing its own is included even before its first write lands), aggregates them, merges the tiered collectors, and emits to the room. Delivery is local — the viewer's socket lives on this worker.
2. **Publishing** — no local viewers, but another worker set the active flag: write this worker's slice to `{ns}:dashboard:worker:{host_tag}:{pid}` every 2s so the emitting worker can aggregate it. A pid is unique only inside one kernel, so a two-host deployment has two processes claiming the same number and the second write would replace the first: the loser's row is destroyed before anyone reads it, so the aggregate reports one worker, one host's live sessions vanish, and nothing is flagged. `host_tag` is a memoised sha256 prefix of the hostname -- hashed, so no raw hostname reaches a Redis key or a dashboard payload. The emitting worker's self-heal, which re-appends its own payload when the read came back short, compares host *and* pid for the same reason. On the idle→active transition the emitter waits ~1s so freshly woken workers land their first slice before the first aggregate.
3. **Idle** — no viewers anywhere: one Redis `EXISTS` per 5s, woken instantly via the `{ns}:dashboard:wake` pub/sub channel — near-zero overhead.

A worker's slice expires 10s after it is written. Besides that worker's collector figures it carries its live session rows, the count of tracked non-task sessions it is running (`sessions.live_active`, computed before the row cap, so the dashboard's `Active` tile is not capped), and the cost of each finished background task whose chat's row is not on that worker, both keyed by chat id **and** user id, so the emitting worker can add a task's cost to its own user's row in that chat; the emitter then drops the ids from every row before anything reaches the browser. The row list itself **is** a capped view — 30 in-flight rows per worker, newest first, then 300 finished rows per worker, and the caps apply to the chat population: a task entry occupies a slot in the worker's recent ring without ever being a row, and is evicted only when the chat rows fill their own cap — and the aggregate keeps 300 across the cluster, dropping finished rows first because the sort puts actives ahead of them. The `tc` map's inputs did not shrink with that: a task entry in the ring is exactly what publishes a finished task's cost for a chat row living on another worker, so the ring holds up to 600 entries (300 per kind) rather than 300. The count is summed from each worker's own number rather than read off the merged rows, so `_PD_SESSIONS_CAP` cannot cap it; a slice from a worker on an older release carries no count at all, and that worker is left out of the sum rather than counted as contributing zero, so a rolling deploy reports a partial sum instead of a silently-lowered total. The `tc` map's key is the two ids joined by `\x1f` (U+001F, the separator `requests/task_model_adapter.py` also uses), as a flat string rather than a nested object because the slice is JSON: `"c1\u001fu1": 0.004`. A temporary chat is never written under its own id: its rows and task costs carry an anonymous stand-in instead, an HMAC of the chat id keyed by `WEBUI_SECRET_KEY` under a label only the dashboard uses, so it is neither the chat id, nor the socket id Open WebUI builds that id from, nor the key the pipe sends OpenRouter. Only the chat half of a temporary chat's `tc` key is replaced by the stand-in; the user half is the user id either way. Without `WEBUI_SECRET_KEY`, a temporary chat's rows carry no id and none of its task costs are published, so a task cost it ran on another worker is left out of its row. Saved and channel chats keep their own id. The worker's own memory always keeps the real id; only what it writes to Redis carries the stand-in.

In single-worker mode (no Redis), the worker with viewers emits directly from its local collectors — the same payload shape, minus the multi-worker `workers` table. In multi-worker mode an incomplete read is never dressed up as a complete one: the emitter marks the payload `degraded` rather than presenting the workers it could still see as the whole cluster, and while that flag is set the dashboard's footer makes no worker-count claim at all, because no payload key carries the last-known total.

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
  "sessions": {"in_flight": 1, "live_active": 4},
  "sessions_live": [{"user": "sam", "model_id": "...", "model_name": "...", "kind": "chat", "status": "streaming", "started": 1751690000.0, "done": null, "elapsed_s": 12.3, "tokens_in": 1200, "tokens_cached": 900, "tokens_out": 80, "tools_ok": 1, "tools_failed": 0, "tools_skipped": 0, "cost": 0.012, "task_cost": 0.0, "worker_pid": 12345}],
  "workers_rss": 1987654321,
  "system": {"cpu_pct": 12.0, "mem_used_pct": 61.0, "mem_total": 16000000000, "disk_free": 142000000000, "disk_total": 250000000000},
  "workers": [{"pid": 12345, "host": "9f2c1a4b7e0d", "uptime_s": 3600.5, "last_seen_age": 0.4, "active_requests": 2, "health": {"init": 1, "wf": 0, "http": 1, "r": 1, "rss": 123456789}}],
  "uptime_s": 3600.5,
  "pid": 12345,
  "host": "9f2c1a4b7e0d"
}
```

The top-level `pid` is the emitting worker -- the process that built this payload and holds the viewer's socket -- on every path: the Redis aggregate, the degraded replay and the single-worker emit all overwrite it with `os.getpid()`, so it never names a worker the read happened to return first. It is a different number from the same field inside a `workers` entry, which is that worker's own pid; `uptime_s` at the top level is the oldest worker's.

The `sessions.in_flight` sample counts a request from the moment it enters the pipe until its job's cleanup tail has finished, so a request whose answer has already been delivered still reads as in flight while it is shutting down. `sessions.live_active` is a different population: the tracked, non-task, not-yet-finalized sessions, counted **before** the row cap, so the `Active` tile reads 4 in the sample above even though `sessions_live` shows one row. It is absent from a payload whose worker is on an older release.

`degraded: true` appears on **every** tick whose Redis read failed -- the first one and every one after it, with or without a cached set. On such a tick the emitter replays its last known worker set rather than presenting a partial cluster as the whole one, and says so in the banner instead of quietly dropping to a single worker. The replayed set is age-bounded: it carries the timestamp of the successful read that produced it, and once that is older than three times `_PD_KEY_TTL` (each worker's own Redis key has by then lapsed three times over, so nothing in the set can be vouched for) the replayed set is dropped, the payload reports only the emitting worker's own slice (`worker_count: 1`), and it still reports `degraded: true` — the banner is the only thing distinguishing that one-worker view from a genuinely single-worker deployment. An empty *successful* read is not degradation and never sets the flag. Storage payloads carry `state` (`connected` / `unavailable` / `degraded`) so the dashboard can distinguish "not initialized on this worker yet" from a genuine failure; the collector wires the shared DB itself on first use, and `unavailable` is also the answer for a store whose pool is gone — a closed store reports it rather than claiming a connection it cannot make, which is why `db_connected` asks for the executor and not only for the session factory and the item model; and by-type/by-model "Least/Most recent" columns are access times (the retention sweep touches `created_at` on every read).

On tick 0, all tiers fire simultaneously for instant dashboard population. The JavaScript checks key existence and updates only the sections whose data arrived in that tick. The payload arrives raw — direct custom emits do not use the `{chat_id, message_id, data}` envelope of OWUI's shared `events` channel, so no client-side filtering is needed.

### Access Model & Live-Mode Requirements

There are no capability keys — access rides Open WebUI's own model access control (`check_model_access` for read, OWUI's own model write-formula for write). The pipe composes no access logic of its own:

1. **The command, the live feed, and read-only actions require a *read* grant (viewer).** The `dashboard` command handler and every socket subscribe resolve a fresh `UserModel` and gate on `authz.can_view` → OWUI's `check_model_access` (honoring owner, admin, direct-user grant, group grant, `user:*` public, and `BYPASS_MODEL_ACCESS_CONTROL`). An ungranted socket is refused the viewers room and told so via a `denied` event; the publisher re-authorizes every local viewer **before every payload it emits**, and evicts any whose grant was revoked, so revocation takes effect on the next tick -- at most `_PD_PUBLISH_INTERVAL` -- without a reconnect. Nothing about that decision is reused across ticks, which is what keeps the bound above exact, and its cost is a known number: **three database reads per person per tick** -- the user row from `authz.resolve_user`, then Open WebUI's own `check_model_access` reads the member's groups and the grant -- plus **one** model-row read for the whole tick, hoisted out of the loop by the `model_read` latch. A person holding several tabs is decided once per tick, so the count follows people and not sockets, and the whole thing costs nothing at all while the dashboard is off, because the dedup, the model read and the grant check all sit below the `not enabled` branch. The sweep evicts on a **definite** refusal only: a check that could not complete — a user row that raised rather than came back empty, a grant check that raised something other than OWUI's `HTTPException` — leaves the viewer in the room, is announced once as undeterminable, and is decided again on the next tick, while the **subscribe** path still refuses on the same failure, so nothing unverified is ever admitted. A row that reads as *absent* is a positive answer and evicts as usual, and the dashboard-off arm sits above the read entirely and evicts everyone whatever the user table is doing. Already admitted and not yet known to be lost is not the same as refused. The distinction is worth having because neither upstream answer is a verdict: OWUI's `get_current_user` turns only a `None` into a 401 and lets a database error propagate, and `Users.get_user_by_id` returns `None` for a missing row and raises for a real failure. That sweep reads the dashboard's own model row **once per tick** (`authz.resolve_view_model`) and decides every viewer against that one read, rather than re-reading the same row once per viewer; when that single read cannot be made, no viewer is decided from it at all and the whole room is re-decided on the next tick.
2. **State-changing actions require a *write* grant (operator).** Actions are invoked over an authenticated `POST /api/pipe/dashboard/action` route and gated on `authz.can_act` → OWUI's own model write-formula (`admin` + `BYPASS_ADMIN_ACCESS_CONTROL` / owner / `has_access(..., "write")`). Classify each action by **effect**: side-effect-free introspection is `read`; anything mutating shared state (clear a cache, trigger an update) is `write` — the same consume-vs-mutate rule OWUI applies across models, KBs, tools, and channels. `register_action` defaults to `write` (fail-restrictive). OWUI's editor pairs a read grant with every write grant, so an operator can always view.
3. **Five actions require the `admin` role on top of the grant.** `config_get`, `config_set`, `update_apply`, `update_restore` and `update_snapshot_delete` are registered with `admin_only=True`, which `dispatch_action` enforces with `user.role == "admin"` after the grant check and before the unknown-name branch. This matches Open WebUI, whose own valve routes are all `Depends(get_admin_user)`; the gate lives in the dispatcher rather than in a handler body because a reconcile-swapped copy of `actions` answers through whichever `dispatch_action` is loaded, and a body-level check would be absent from a freshly imported module.
4. **The action route is CSRF-safe unconditionally.** Its authentication is **header-only** (`Authorization: Bearer <token>`, reusing OWUI's `decode_token` + `is_valid_token` + a fresh `Users.get_user_by_id`); it never reads the session cookie, so a cross-origin page cannot forge a call with the victim's token regardless of OWUI's CORS/SameSite settings. Every outcome is audited (user, action, outcome, client IP; write args included, secret values masked at any depth, and for a value under a secret-named key at any depth) -- including the `unavailable` outcome the `503 {"error": "action unavailable"}` responder writes when no dispatcher could be resolved, whose write args are redacted exactly as on every other outcome. A value whose own key is a declared non-secret valve or a protocol field the action defines keeps the value the operator wrote -- that setting is stored and shown in cleartext anyway, so a secret typed there is logged as typed; under any other key, and inside any nested list or dict, the value is written as a `<redacted TYPE>` marker instead. A secret under a non-secret valve is therefore visible in the log by construction, which is why that valve must not be used as a place to keep one.
5. **The whole interactive dashboard requires the same-origin iframe setting.** The dashboard reads the session token from `localStorage`, which only works when Open WebUI's **Settings → Interface → "iframe sandbox allow same origin"** is enabled — the identical requirement as the [OpenRouter Fusion live panel](openrouter_fusion.md). With it off, the iframe runs with an opaque origin: the socket never opens *and* every server-backed action fails for lack of a token, so the Config, Usage, and Update tabs are inert, not just the live feed (the Update tab detects this and points at the setting). Native browser dialogs are additionally blocked in the sandbox regardless of settings, which is why every dashboard confirmation is an inline click-again button. If a restrictive `IFRAME_CSP` is configured, the same policy documented for Fusion applies (`script-src 'unsafe-inline'` + `connect-src 'self'`).

That same payload carries per-request rows for live and recent requests that name **every other user's** display name, model and spend, not only the aggregate operational counters (concurrency, queue depths, breaker trip counts, uptime, worker PIDs). A `sessions_live` / `sl` row is one request, with these keys: `user` (the requester's `user_name`), `model_id`, `model_name`, `kind` (`chat` or `task`), `status`, `started`, `done`, `elapsed_s`, `tokens_in`/`tokens_cached`/`tokens_out`, `tools_ok`/`tools_failed`, `cost`, `task_cost`, `worker_pid`, `chat_id` and `user_id`. The fold that adds a background task's cost to a chat row matches on `(chat_id, user_id)` together, so one member of a shared chat id never carries another's task spend; both ids are removed before the payload is emitted. A row is none of that — no chat content, no secrets. The connection bar includes **Disconnect** / **Connect** buttons: disconnecting tears down the socket (Socket.IO removes it from the viewers room server-side automatically); reconnecting re-runs the connect → `user-join` → subscribe sequence. When the last viewer disconnects, all workers return to idle within seconds.

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
| `user` | The Open WebUI user dict containing `role`, `id`, `name`, `email`. Gated by OWUI model access (`can_view`) before dispatch, so a handler runs only for a user granted at least read (viewer) on the pipe-dashboard model -- not necessarily an admin. Five actions are the exception -- `config_get`, `config_set`, `update_apply`, `update_restore` and `update_snapshot_delete` -- and are additionally refused to any user whose `role` is not `admin`. |
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
| `schema` | `dict[str, SchemaValue] \| None` | `None` | Per-key type check on `args`; a value is a type (or a tuple of types) that the key must carry, or `optional(<type>)`, meaning *type-checked when present, defaulted by the handler when absent*. `None` skips validation |

Each registered action is stored as an `ActionEntry`:

```python
@dataclass
class ActionEntry:
    name: str
    permission: str                                    # "read" | "write"
    schema: dict[str, type | tuple[type, ...] | _OptionalKey] | None
    handler: Callable[..., Awaitable[dict[str, Any]]]  # async (pipe, user, args) -> dict
```

The handler receives the pipe, the resolved OWUI user, and the validated `args` dict, and returns a JSON-serializable dict.

**Dispatch pipeline.** `dispatch_action(pipe, user, name, args, *, client_ip=None)` runs each call through a fixed sequence and returns a `(status_code, envelope)` tuple:

1. **Authorize** -- `can_act` for a write action, `can_view` for a read action (an unknown action is checked as read).
2. **Resolve** -- look the action up in the registry.
3. **Validate** -- type-check `args` against the schema.
4. **Rate-limit** -- one call per `(user, action)` per second.
5. **Run + audit** -- invoke the handler and audit the terminal outcome (write args are included in the audit line with secret values masked, read args are not). A failed write records the exception *type* -- and the text of the pipe's own `_ClientMessage` -- but never a traceback, because the traceback carries the rejected input verbatim: pydantic formats the failing field's `input_value` into the `ValidationError` message that `exc_info` prints. The recursion inside a submitted value stops at depth 32, which is a DoS bound and not a masking one; above it only the 200-character scrub on the audit line keeps a value out.

| Status | Envelope | When |
|--------|----------|------|
| `200` | `{"ok": true, "result": <dict>}` | Handler succeeded |
| `400` | `{"error": "missing or invalid: <key>"}` | `args` failed schema validation: a required key is absent, or a supplied one (including a `null`) has the wrong type. A key declared `optional(...)` and simply omitted is *not* a failure -- the handler's own default is used |
| `403` | `{"error": "forbidden"}` | Caller lacks the required grant, or holds it but is not an `admin` and the action is one of the five admin-gated actions |
| `404` | `{"error": "unknown action"}` | No action registered under that name, either in the copy this route holds or (see the reconcile retry below) in a freshly re-imported one |
| `429` | `{"error": "rate limited"}` | Second call within the 1 s per-`(user, action)` window |
| `500` | `{"error": "action failed", "detail": <cause>}` | Handler raised. `detail` is a sentence naming the cause for a `config_set` refusal -- the store refused the write, or the stored schema would not accept the edited field -- and the exception's class name otherwise. |

**Built-in actions:**

| Action | Permission | Schema | Returns |
|--------|:----------:|--------|---------|
| `whoami` | `read` | — | `{user_id, role, can_view, can_act}` |
| `echo` | `write` | `{"message": str}` | `{"message": <echoed>}` |
| `usage_stats` | `read` | `{"range"?: str, "tz_offset_min"?: int, "include_tasks"?: bool}` | The Usage-tab analytics dict (see [Session Tracking & Usage Records](#session-tracking--usage-records)) |
| `config_get` | `read` + `admin` | — | Valve specs with current values (secrets masked to a set/not-set flag) and the current config revision, plus a `config_unreadable` boolean that is `true` when the stored settings could not be read at all. The snapshot also carries `reset`, a sorted list of the stored values the current schema rejects; the tab counts them in its note bar and the pipe log names them, and the next save is what removes them. A name the schema no longer carries at all is in that list too, and is named twice rather than once: here for the Config tab's note bar, and independently by the `valve_salvage` scan the boot path runs on the decrypted row, which names every stored key outside `Valves.model_fields` and never quotes the value it was holding. When the stored configuration cannot be read, the action still answers: the tab renders the in-memory values under a banner with Save disabled, rather than a panel that will not draw or a default-filled view passed off as what is stored. A stored set that will not decrypt is told apart from an empty one by the raw column probe below, so it reports `config_unreadable` instead of rendering factory defaults |
| `config_set` | `write` + `admin` | `{"edits": dict, "rev": int \| str \| None}` | Persists the changed subset and returns the new revision, a `reset` list naming the stored values the current schema rejected, and the stored value of every valve it wrote, secrets excluded, read back the same way `config_get` reads — or, when that read did not succeed, no value for any edited setting, and the tab keeps the admin's own edit; the write is committed either way — plus a `secrets` block of `{name: {set, stored}}` for the secret valves the save named — the same two flags `config_get` reports, read from the row that was written rather than from the edit. The tab's Clear control and its hint are driven from that block, because the store's answer and the edit's are not the same: clearing a key that sits under an environment default leaves the setting configured, and a clear that removed nothing leaves nothing to clear. A save that names no secret carries no entry for it; a save whose echo is dropped carries no block at all, and the tab then keeps the projection it had. When the read-back that follows a committed write is itself not readable, that drop is the whole answer: `values` is `{}`, the `secrets` and `post_reset` blocks are empty, and the write stands, because a value the pipe did not read out of the store is not a saved value. — or a conflict payload when the client's revision was stale. Two administrators saving at once on one worker are serialised by a per-pipe lock held across the read, the merge, the write and the echo, with the revision re-read inside it, so one of them takes the conflict path; `not_saved` names the edits the loop never interpreted — a name the schema does not carry, or a blank secret box on a key that was never stored — and `saved` counts the rest. A value restored to its factory default, and a `..._TEMPLATE` box cleared so the built-in text goes back, are both saves: the row carries no name for either, because a value that now equals its default is not a customisation, and the row is what holds the result. A *clear* is the one edit the store legitimately does not hold: it is counted as saved, and it is collected in the same pass that honours it, so a `None` on a secret and a blank on a nullable number are both recognised as clears rather than as un-persisted edits. The `reset` list is what the tab names on the save that dropped them, beside the count in the note bar; the next `config_get` reads a repaired row, returns `reset: []`, and clears the note. When the stored configuration could not be read at all it returns `{"unreadable": <reason>, "rev"}` and writes nothing, and *not* a conflict payload: the conflict banner claims another administrator changed the configuration, and the pipe can name this cause, so the two would contradict each other. A refused **write** and a **rejected edit value** are a different fault and answer `500 {"error": "action failed", "detail": <sentence>}` instead -- `the database refused the write, so nothing was saved` when the store takes the merge and refuses the write, and `the stored settings would not accept: <field names>` when the stored schema rejects the edit, which names the fields and never the values (pydantic's own text would carry the rejected `input_value`, and `edits` legitimately carries `API_KEY`). Either way the tab toasts `Save failed: nothing was saved — <detail>`, so the "nothing was saved" half is the tab's. The conflict payload is still correct for the unreadable-*revision* cause, where no current state is known and the write is blocked anyway |
| `update_apply` | `write` + `admin` | `{"rev": int \| str, "compressed"?: bool}` | Applies the newest available release, snapshotting the installed function source first. Refused with `403 {"error": "forbidden"}` to anyone who is not an `admin`, and with `{"error": "disabled"}` / `{"error": "valve_unreadable"}` in its body when `PIPE_DASHBOARD_UPDATE_ENABLE` is off or the stored valve set cannot be read |
| `update_restore` | `write` + `admin` | `{"file_id": str, "rev": int \| str}` | Restores a stored snapshot over the installed function source. Same two refusals as `update_apply` |
| `update_snapshot_delete` | `write` + `admin` | `{"file_id": str, "sha256": str}` | Deletes a stored snapshot. Same two refusals as `update_apply` |

With `PIPE_DASHBOARD_ENABLE` off, three paths refuse, and the valve is read from the stored row at each of them, so a toggle takes effect on the very next request on every worker, with no restart -- and a row that cannot be read refuses rather than falling back to the in-memory copy, which is what lets a failed read override an operator's disable: the action route answers `404 {"error": "unknown action"}` (indistinguishable from an unregistered name) and audits the refusal as `outcome=disabled`; a new socket subscribe is refused with a `logger.warning` naming the sid and never joins the viewers room; and every viewer already in that room is evicted -- immediately on the `function.valves_updated` event, and again by the re-authorization that precedes every emit -- while `emit_dashboard` itself refuses, so the live feed stops for a tab that was open when the valve went off. That re-authorization is not deferred by a cache: it re-decides every local viewer from the database on every emit. Both dashboard write paths also announce themselves to Open WebUI's own event bus: a `config_set` that commits publishes `function.valves_updated`, and a committed apply or restore publishes `function.updated`, each carrying the pipe's function id as `subject.id`, the row's own `type` and `name` on the content event, and the acting administrator as `actor` -- so an event function or webhook subscribed to either sees the change Open WebUI's own route publishes for it. That is the event reaching the sink, and the sink then decides on the same persisted row every other gate reads, so the Config tab's own save evicts the panel it just locked out as immediately as Open WebUI's function editor does. The absent key is the off state: the Config writer drops any field equal to its declared default, and that default is `False`, so a stored row that omits `PIPE_DASHBOARD_ENABLE` is a row that says off. An unauthorized caller is still answered `403` first: the valve gate sits below the grant check and below the rate limiter, so turning the dashboard off never turns an authorization refusal into a not-found and never removes the 429 arm. A fourth thing the same valve switches off is the request path itself: with `PIPE_DASHBOARD_ENABLE` *and* `PIPE_DASHBOARD_USAGE_COLLECT` both off, `on_request` registers no session for a non-dashboard model at all and `on_emitter_wrap` returns `None` under the same `ENABLE or COLLECT` predicate, so the pipe holds no live-session state for a request nobody observes and builds no emitter wrapper for it either. A completed turn the tracker holds nothing for reads no persisted valve row either, so an install with the plugin off pays no query per generation: `on_generation_complete` asks `SessionTracker.has(request_id)` before it reads, and a turn with no entry to finalize has no row for `_persist_usage_row` to gate. The second switch is part of the condition because `_persist_usage_row` is bound to `on_finalize` and is gated on `PIPE_DASHBOARD_USAGE_COLLECT` alone — the Usage tab's rows are written from `finalize`, not from a viewer, so an install that collects but does not display keeps recording. The abandon reaper starts under the same `ENABLE or COLLECT` predicate, for the same reason: it is what finalizes those rows, and it has to keep running on a worker nobody is watching.

**The update service's error taxonomy.** `_reload_via_loader` in `update_service.py` snapshots the whole `sys.modules` mapping and `sys.meta_path` before it hands the content to Open WebUI's loader, and `_commit` wraps every exit after that loader in one `except BaseException` that restores that snapshot and re-raises -- so a raised error can never leave the pipe half-configured. The snapshot is still the whole mapping (that is what makes the compressed bundle's *deletions* recoverable), but the restore is scoped to what the attempt itself wrote: every snapshotted key in the attempt's own namespaces -- `function_<pipe_id>`, `open_webui_openrouter_pipe`, `open_webui_openrouter_pipe.*` -- is put back by identity, a key the attempt created in those namespaces is removed, and a bundle-owned `sys.meta_path` finder is restored at the index it held before, so a generation swap that removed the previous generation's hook gets it back. Everything else is left exactly as a concurrent request left it, which is the same shape as Open WebUI's own narrower cleanup on a failed load (`utils/plugin.py:308-313`, `del sys.modules[module_name]`): the attempt cleans up after itself rather than resetting the worker. `BaseException`, not `Exception`, because a cancelled commit raises `CancelledError`, which is not an `Exception`; the refused-write arm restores before it revives the cached instance, so a `write_failed` leaves a serving pipe behind as well. Every `UpdateError` carries one of these codes, and the tab renders each differently, because they call for different operator responses:

| Code | Means | Transient (`_TRANSIENT_CODES`)? |
| --- | --- | --- |
| `storage_unavailable` | The store could not be reached or read -- a database outage, a snapshot store that is down, a row that will not decrypt. Nothing about the content is wrong. | yes |
| `row_unreadable` | The store would not return the pipe's own function row. Open WebUI's `get_function_by_id` answers `None` on any database error as well as on a deleted row, so a blip here says nothing about the content. | yes -- a database fault clears on its own, and retrying costs one read |
| `validation_failed` | The content itself is unacceptable: the frontmatter, the digest, the size cap, the Open WebUI version gate, or an HTTP fault from the download. | no |
| `write_failed` | The content passed every check and the function-row write was refused anyway. The freshly loaded code is rolled back and the previous version really does stay active. The revived instance's `Pipe.valves` is restored the way Open WebUI restores it (`functions.py`: the module's own `Valves`, built from the pre-attempt row with the null entries filtered out) -- and state derived inside `__init__` before that rebind is stale in both, so the rollback is not a full restore of anything `__init__` computed. | no -- a refused write means the store is unhealthy, and retrying it is exactly the retry to avoid |
| `exec_failed` | The new bundle did not survive Open WebUI's own loader. The module state is restored and the previous code keeps serving. The row's `is_active` is left exactly as the operator had it: a row that was on has the loader's own `is_active: False` repaired back to on, and a row the operator had already switched off is left off with no write at all, because the repair is skipped on that arm and the code stays `exec_failed` rather than `exec_failed_inactive`. So this code reports three states, not one: repaired-and-serving, admin-off-and-still-off, and a refused repair -- the last of which is the row below. | no |
| `exec_failed_inactive` | The bundle did not load AND the repair write was refused, so the function row is left switched off and Open WebUI lists no OpenRouter model for it until somebody switches it on in Workspace > Functions. | no -- a refused write means the store is unhealthy, and retrying it is exactly the retry to avoid |
| `offline` / `rate_limited` | GitHub could not be reached, or is rate-limiting this server. | yes |
| `repo_not_found` / `bad_repo_valve` | The configured repository valve is wrong, or GitHub has no such repository or no releases. | yes -- the operator is expected to fix the valve; the tick backs off rather than pausing the version |
| `stale_rev` / `update_in_progress` | Another administrator's save, or another update, got there first. | yes |
| `digest_mismatch` | The release asset carries no `sha256:` digest, or the bytes that arrived do not hash to the one it published. A content problem, not a transport one: re-fetching the same asset fetches the same wrong bytes. | no -- and that is the whole point. The auto-updater writes `_auto_skip` for this version on this worker, so a mismatch pauses the version for the life of the worker until an apply succeeds, while a **manual** apply shows "Try again in a moment", which is true for a person who can fix the cause in between. The two surfaces therefore promise different retry behaviour for one code, and neither is wrong for its own caller. |
| `not_found` | A snapshot row, or the file behind it, is gone. | no -- the thing is not coming back on its own; the operator has to re-take the snapshot |
| `stale_snapshot` | The snapshot changed after the list it was chosen from was loaded, so the delete carried a digest that no longer matches. The tab says to refresh and retry. | no -- it is a lost-update guard, not a fault, and a retry without a refresh would hit it again |
| `package_mode` | The installed row is a package/stub rather than a bundle, so there is nothing to apply here: a package install updates through its pinned requirement. | no -- the valve or the install shape has to change first, and the auto-updater must not pause a version over it |
| `no_matching_asset` | The latest release carries no asset of the shape this installation wants (the compressed bundle, or the flat one). | no -- the release will not grow an asset mid-tick; the operator needs the other install mode, or a different release |
| `incompatible_owui` | The release's frontmatter asks for a newer Open WebUI than the one running. | no -- upgrading Open WebUI is the fix, and the version cannot arrive by retrying the asset |
| `internal` | The request the service was handed cannot be acted on: a manual apply or restore arrived with no request object. A caller-side defect, not a store or release one. | no |

The split is the point: a `storage_unavailable` and a `row_unreadable` are both retried on the next tick -- one is the snapshot store and one is the function row, and neither is a statement about the content -- a `validation_failed` pauses the version so a bad release is not re-fetched, and a `write_failed` pauses it too because the code is fine and the store is not. `_TRANSIENT_CODES` is the single list the auto-updater consults, so a code added there is retried everywhere at once. The `Previous versions` list carries its own flag, `snapshot_storage_error`, and the tab branches on it BEFORE the empty-list test, so "storage down" is never drawn as "No snapshots yet."

`whoami` and `echo` are reference implementations; `usage_stats` powers the Usage tab; `config_get` and `config_set` power the Config tab (see the [Operations Guide](plugins_pipe_dashboard.md#editing-configuration)). A new action is registered by importing its module at plugin load -- the same explicit-import requirement as commands.

`config_get` returns a `drift` report with two lists: `unenriched` (valves with no `CONFIG_META` entry) and `orphaned` (entries with no valve). The Config tab renders `unenriched` only. So `orphaned` names settings this plugin declares and the Config tab cannot show, and nothing on screen says so; a `drift.orphaned` that is not empty is a signal to read, not a state any install should be left in.

**Why the stored config is read more than once, and what "unreadable" means.** `_read_stored_valves(pipe_id)` in `actions.py` returns `(stored, read_ok)` and separates the transport read from its verdict, so the raising case and the `None` case reach the same marker. `None` is unreadable: Open WebUI's `get_function_valves_by_id` catches its own DB errors and returns `None`. `{}` is NOT reliably "nothing stored" -- `decrypt_valves` returns `{}` on `InvalidToken`, i.e. on a failed decrypt, which is what a rotated `WEBUI_SECRET_KEY` produces. `stored_row_readable(pipe_id, stored)` in `config_service.py` tells those two apart by reading the raw `Function.valves` column, the same probe `update_service._row_valves_checked` uses: ciphertext present but nothing decoded is unreadable; the column empty is genuinely unset. Only a *positively observed* ciphertext downgrades the read -- an Open WebUI without the async db reader, or a transient failure on it, is not evidence of corruption, and treating it as such would mark every such deployment unreadable. `_effective_valves_and_state(pipe)` rebuilds the valve model from that subset and returns `(valves, dropped, stored, read_ok)`, keeping the stored subset separate from the reconstructed one; `_effective_valves_and_drops(pipe)` is the `(valves, dropped)` view of it that the config tab's own callers use, and `_effective_valves(pipe)` the valves-only view. The two name one key from two directions, which is why the same stored name can appear in a note bar and in the pipe log for one save: `readable_stored` reports what the reconstruction could not keep, and `valve_salvage.drop_unvalidatable` reports what it could not keep from the row itself -- a value the schema rejects, and separately a name the schema does not publish, whose stored value it never quotes because there is no annotation left to decide that it is a secret. `config_set` refuses to write when the read is not ok, from either the conflict arm or the pre-write gate, and both paths carry `config_unreadable` so the client renders the banner instead of a Reload button that would re-read the same undecodable blob. A read that returned `None` keeps its own `unreadable` refusal payload on the save path, so the admin is not told a concurrent editor caused it. Both readers of a persisted valve row -- the update surface's `update_service._row_valves_checked` and the dashboard gates' `config_service.stored_gate_valves` -- resolve a key the stored row omits to that field's declared default, and neither consults the worker's own copy for it, so a setting the administrator reset to default reads as the default on every worker at once instead of as whatever a stale worker happens to be holding. A key the valve class does not declare is left out of the result entirely, and the call site's own absent-field default stands: a pipe that declares neither gate is not the dashboard and must keep working. A committed save reads the row a third time, in `_saved_values`, and that read belongs to the echo alone: the two reads above gate the write, and by the time the echo runs the write is already committed. It is the one step of the three that reads the row for reporting rather than for the decision, so it is also the one that has to be read under the same exclusion as the write -- the tab adopts the echo as its new baseline, and a read-back taken after the lock is released can report a second administrator's write as this save's own. So when the third read is not ok the echo is dropped -- `values` empty, no `secrets` block, `post_reset` empty -- rather than rebuilt from `pipe.valves`, which is the value the store has just contradicted, or from `Valves()`, which is a set of defaults for a row nobody could read. The pipe log names the arm: the warnings above describe the panel's view, this one describes an echo that could not be made after a commit.

**Why the Clear control is gated on the stored subset, not on `secret_set`.** (Amends the original design, which said the button is "rendered only when `v.secret_set` is true".) The design is overridden: a control must be able to remove what it promises. `secret_set` is derived from the *reconstructed* valve set, which includes the `OPENROUTER_API_KEY` default, so on the common env-only deployment it is true for a key the pipe never persisted -- a Clear button would render, and clicking it would stage a clear that saves "1 setting" and leaves the store untouched. The button is therefore gated on `v.secret_set && v.secret_stored`, where `secret_stored` is put on the spec by `_config_snapshot(valves, stored)` from the stored subset. `secret_set` keeps its meaning (an env-supplied key genuinely *is* configured, and the placeholder, the hint and the Default cell key off it): a `stored`-driven control and a `secret_set`-driven placeholder are different questions and do not share one flag. The write-gate refusal site has read nothing, so it passes no `stored`, `secret_stored` is False there and no Clear button renders -- correct, because the tab is being told the store is unreadable and Save is off anyway. `commitSave` used to derive both flags from the edit it sent, and the store does not have to honour that prediction: it overwrites both from the response's `secrets` block now, so the Clear control and the hint follow the row that was written. `secret_stored` is the half nothing client-side can predict at all — a key typed for the first time has no stored value until the server says so — which is why a first-typed secret was not clearable until something forced a reload. The tab applies the `secret_set`/`secret_stored` the `config_set` reply carries, and the env-only case is resolved there, because the server is the only party that knows whether a value came from the environment and the request says nothing about it. That local projection survives only as the fallback for a server that returns no `secrets` block at all, and follows the same rule: it clears `secret_set` only when a clear actually removed something stored, so an env-only secret does not flip to "not set" locally and back to "configured" on reload.

**Why the save path keeps a secret only when its plaintext is non-empty and differs from the default.** `_is_clear_edit(fld, value, current)` is the explicit clear: a `None` edit for a secret. `None` and `""` are different intents and collapsing them is the defect. `""` is "the box is empty", which is what a stray keystroke followed by a backspace looks like, so it keeps meaning unchanged; `None` is a distinct wire value the client sends only from its Clear control, so it means "remove what is stored" whether or not anything is stored under that name -- both outcomes are the same stored state. A `None` edit pops the key inside `merge_for_save_with_drops` (the two-value form `merge_for_save` wraps), but `full = valves_cls(**merged).model_dump()` is a dump of a *fully-defaulted* model and always contains every field, so the key is not absent from `full`; it is dropped by the subset filter, which keeps a secret only when `plain` (its decrypted value) is truthy AND differs from the default. A clear therefore lands on the stored subset as "the key is not stored", which is decision 2: the setting returns to its default.

**Why `_tool_counts` returns three and `db_row` keeps two.** `_tool_counts(entry)` yields the three tool-outcome counters (`tools_ok`, `tools_failed`, `tools_skipped`) as ints and is used by the live row. `tools_skipped` is a breaker-refused or no-longer-awaited call, not a failure. `db_row` narrows the same three to the two the usage table actually has -- `USAGE_ROW_FIELDS` is the insert projection, so a third key there would raise `OperationalError` against every table an earlier release created, and `_persist_sync`'s batch loop swallows it at DEBUG.

---

## Authorization Helpers

`authz.py` is the single authorization chokepoint. It composes no access logic of its own -- every decision delegates to Open WebUI's own model access control, so the dashboard inherits the grants an admin sets in the model's Access editor. Reuse these helpers for **model-access** decisions instead of writing your own role checks against OWUI's tables:

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

Actions are invoked over one authenticated route registered on Open WebUI's own FastAPI app (ahead of the SPA catch-all), not over the Socket.IO channel. Registration is idempotent: the route is added once and never removed, so the path is absent from `app.routes` for no instant and a request arriving during any number of later registrations still matches. Each call re-checks the live `app.routes` list under the registration lock rather than trusting the module-global `_registered_paths` set, which says "registered" even after a fresh app object replaces the old one.

| Property | Value |
|----------|-------|
| Method + path | `POST /api/pipe/dashboard/action` |
| Auth | Header-only `Authorization: Bearer <token>` -- reuses OWUI's `decode_token` + `is_valid_token` + a fresh `Users.get_user_by_id`; the session cookie is never read, so the route is CSRF-safe regardless of OWUI's CORS/SameSite settings |
| Request body | `{"action": <str>, "args": <object>}` |
| Response | The `dispatch_action` envelope, at its status code |

The route adds three guards in front of the dispatcher: a `401` when the bearer token is absent, malformed, invalid, or maps to a role outside `{user, admin}`; a `404` (`{"error": "plugin_system_off"}` — the same envelope shape the route's own unknown-action answer uses, so the panel's `error` branch can tell "the plugin system is off" from "the stored settings could not be read"; an absent route answers a different body entirely, `{"detail": "Not Found"}`, and this one is indistinguishable from it only to a client that checks the status code) when the master switch `ENABLE_PLUGIN_SYSTEM` is off on the resolved pipe, answered from the stored row on every request and refusing when that row cannot be read rather than serving the in-memory copy (a dashboard switched off while the plugin system is on passes this guard and is refused by the dispatcher from the same stored row, with its own `{"error": "unknown action"}` 404 and `outcome=disabled`), and that pipe is the one currently serving — resolved by name from `sys.modules` on each call, so a `Pipe` retired by an in-place update cannot leave the switch reading a frozen copy — the route is registered once and never unregistered, so a worker that started with the switch on would otherwise keep serving the whole admin surface; and a coarse per-user `429` (one request per 0.25 s) ahead of the per-action rate limit, which the 404 is answered above and so never spends a slot. A pipe the getter cannot resolve, or one carrying no `.valves`, reads the switch as off and is refused on the same terms. Past the guards the route never raises: when neither the live `actions` module nor the reconcile yields a dispatcher -- a partially-initialised module in `sys.modules`, or one the hot update deleted -- `_preferred_dispatch` hands the call to `_dispatch_unavailable`, which takes `dispatch_action`'s signature verbatim and answers `503 {"error": "action unavailable"}` after writing one audit record with `outcome=unavailable` and its write args redacted exactly as on every other outcome (with no action entry in hand, so every registered schema is vouched at once -- the widest set, and the only one that cannot be stale on the arm where the registry is what failed to resolve). That is the envelope every other refusal on this route uses, so the panel's `callAction` renders it, and 503 is retryable by the poll. The responder lives in `http_routes.py`, never in `actions.py`: the live lookup resolves `dispatch_action` from the live module, so a responder parked there is exactly what that lookup could hand back. The 404 sits **above** the per-action grant check, because the master switch is not a per-action grant: nobody is asked for permission, and the audit records `plugin_system_off` rather than `forbidden`. The dashboard calls it from `callAction(...)` in the shell, forwarding the same `localStorage` token it uses for the socket.

**Post-update reconcile.** `http_routes.py` imports `ACTIONS` and `dispatch_action` **by value** at module import, so a pipe updated in place leaves this route holding the *old* copy: a newly registered action 404s even though the on-disk code defines it. When the requested name is not in the held registry, the route therefore tries to re-resolve the action module once (`_resolve_fresh` → `get_function_module_by_id`) and, on success, keeps the fresh `(dispatch, pipe)` pair for the rest of the process. The re-resolve requires a read grant, is serialised by `_reconcile_lock` with a double-check, and is retried at most once every `_PD_RECONCILE_BACKOFF_S` (5.0 s, on `time.monotonic()`, so a wall-clock step cannot break it) — a reconcile that returns nothing therefore does not latch, and an update that lands while the fresh module is temporarily unresolvable is repaired by a later request. The lock is re-created whenever the running event loop is not the one it was bound to — the same guard `pipe.py` keeps on `_queue_worker_lock` (and on `_log_worker_lock`) and the fourth site `actions._config_write_lock` keeps on the per-pipe admin-save lock — because an `asyncio.Lock` binds itself to the first loop that makes it wait and refuses to be taken by any other, so a lock a closed loop left behind would otherwise stop the repair instead of serialising it. That cache also prunes: an entry whose lock is bound to a **closed** loop is deleted on the next call, because a contended acquire holds a strong reference to its loop, so an unpruned entry pins that loop — and everything it referenced — for the life of the process. A lock that is merely *unbound* is kept, never pruned: two `config_set` coroutines that must serialise can both arrive before anything has contended the lock, and handing them two different locks would lose an edit. Success is terminal: once a fresh pair is installed, the reconcile block is short-circuited forever -- *until* a teardown takes the pair away, which is what the epoch counts. A `clear_fresh_dispatch` or a `set_pipe_getter` bumps it, and a re-resolve that overlaps either discards its result and arms the same 5 s backoff instead of installing a dispatch for a pipe that is gone, so the two ways a reconcile can publish nothing are now the same arm. Once-only registration is still sufficient, but not for the reason it used to be given. The endpoint in `app.routes` is the *first* generation's `_action_route` function object, and after an in-place update its `__globals__` belong to a superseded namespace rather than the one the process is running — so "a module-level function name that reads the module globals at call time" is not a safety property here, it is the bug. `_action_route` therefore resolves the dispatcher, the action registry **and the serving pipe** by name from `sys.modules` on every call; the reconcile above is one of those per-call reads, not the mechanism that makes idempotence sufficient, and what is left of its job is the window between a reload and that reload's first `pipes()`. A route registered by a version that predates the live pipe read is replaced once, by the first version that has it, so a worker that has not restarted picks the fix up rather than serving a retired `Pipe` for the life of the process; a fixed route is never replaced, so a later update — including a rollback to an older version — cannot replace the code of the one route an operator needs to **roll that update back**. What does *not* follow a reload is the endpoint's own glue: the bearer check, the JSON depth guard (bound into the dependant at `add_api_route` time) and the coarse per-user limiter with its per-generation state all stay at the version that registered the route, so a change to any of them takes effect at the next worker restart. The current request still answers `404` when the re-resolve fails; the retry happens on the *next* one, and the route never raises for this.

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

Commands that need database access (e.g., storage stats) must use `run_in_threadpool` to avoid blocking the async event loop. The same rule governs the publisher's slow tier: `collect_slow_stats` is submitted to the artifact store's own DB thread pool, which also serves the chat request path, so the dashboard's full-table scan of the artifact payload column can make in-flight artifact persist/fetch/delete queue behind it and can show up in the "Write pool backlog" figure. That is the arm for a store with a pool. A store that has none — closed by `close()`, or one whose build keeps failing — is submitted to a publisher-owned single-worker thread named `responses-slowstats`, one short-lived pool per slow tick (at most one a minute per emitting worker); a pipe with **no** store at all is the only case left running inline, because the collector does no I/O there. Access the artifact store's SQLAlchemy session factory through `ctx.pipe._artifact_store`:

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

**The abandon sweep.** `SessionTracker.sweep()` runs periodically and finalizes any active request that has produced no liveness signal for two hours (`_ST_ABANDON_S`), as `failed`. It has two callers: a plugin-owned timer task (`_sweep_loop`, every `_ST_SWEEP_INTERVAL` = 300 s, started from `_re_register_registrations` whenever `PIPE_DASHBOARD_ENABLE` or `PIPE_DASHBOARD_USAGE_COLLECT` is on) and `_live_snapshot`, which a viewer's cadence reaches every 2 s. The timer is what makes the sweep independent of anybody watching: an install with no dashboard open still reaps its abandoned requests, which is the case the Usage tab's Failed count is otherwise short on. With both valves off the tracker is not populated at all, so there is nothing for it to sweep and no task is created. Liveness is the `seen` stamp, which `mark_streaming`, `tool_started`, `tool_result`, `retry` and `update_usage` all refresh, and which `mark_stream_alive` refreshes on each of the liveness-carrying event types the wrapped emitter sees — `chat:message:delta` and `response.output_text.delta` for a plain streamed turn, and `fusion:event` / `fusion_inner:reasoning.delta` for an internal Fusion turn, which emits no text delta of its own because `streaming_core` consumes the upstream one and republishes the synthesis as `fusion:event`; `started` is the fallback for an entry that predates the stamp. **`seen` is not published**: it is the tracker's internal sweep clock, held on the in-memory entry only. `started` remains the elapsed / `duration_ms` clock. A row is finalized once and never re-finalized. The per-chunk refresh is coalesced: it writes at most once per `_ST_STREAM_STAMP_COALESCE_S` (5 s) per session, gated on a monotonic comparison made before the lock, so the hottest path in the pipe costs one float comparison per chunk rather than one lock-guarded dict write. That holds for a *tracked* session: an id the tracker does not hold costs one dict-membership test in `mark_stream_alive` and nothing else, and with both valves off the wrap is not installed at all, so the chunk never reaches the tracker. So a request that truly never returns still produces a `failed` row, and a request that is still running **and producing liveness signals** — a streaming generation, in either of the shapes above — when the sweep passes is finalized by whoever really ends it, so its recorded status, duration and token counts are the real ones, in exactly one row: staleness is re-tested at the moment of removal, in the same critical section as the removal itself, so a liveness signal landing between the sweep's snapshot and the pop saves the request rather than being discarded. A stretch that only *reasons* is not one of those shapes: `streaming_core` consumes reasoning deltas and does not republish them, and the only thing that surfaces is a throttled `status`, so a turn that reasons for hours and then produces no text (no `output_text` delta, no tool) emits nothing the stamp covers and is abandoned as `failed` even though the model is working. No heartbeat covers that subpopulation either, and closing it would need one rather than another event type on the stamp. The `mark_streaming`, `tool_started` and `update_usage` stamps are gated on `on_emitter_wrap`, which `pipe.py` only builds when `stream_queue is not None` (`wants_stream`); the `tool_result` and `retry` stamps are not. `on_tool_result` is dispatched from `_hand_back_tool_result` in `pipe.py` and `on_request_retry` from the shared retry loop in `requests/orchestrator.py`, so both reach a request that was sent with `"stream": false` — its `seen` moves past its `started` and the sweep leaves it alone, with its real cost and token counts intact. The subpopulation this fix does NOT cover is the one that never streams, never runs a tool and is never retried: there `seen` stays equal to `started`, both cross the two-hour mark together, and the sweep abandons it as `failed` exactly as before. A heartbeat for that subpopulation is not implemented.

**Live session rows.** `SessionTracker.live_sessions()` returns the rows the Live tab renders and the publisher ships as `sessions_live`. The rows are a **capped view of a larger population**, and the caps are what the tile numbers have to be read against: `_ST_ACTIVE_CAP` keeps the **30 newest** in-flight rows per worker, `_ST_RECENT_CAP` keeps 300 finished rows per worker, and `_PD_SESSIONS_CAP` keeps 300 rows across the cluster after aggregation, dropping finished rows first. Both per-worker caps bound **rendered chat rows**, not tracked entries: `_live_sessions_locked` filters the task entries out of both populations *before* it slices, so a task entry can no longer consume a slot a chat row would have rendered in. The two populations are capped **separately** — `_trim_recent_locked` keeps the newest 300 chats and the newest 300 task entries, so the recent ring holds up to 600 entries rather than 300, and a finished chat is never evicted by a newer task entry. The task entries stay in the ring on purpose: `_fold_task_into_parent` reads it to fold a task's cost into its parent chat's row, and `_task_costs_locked` reads it to publish (`tc`) the cost of a finished task whose chat row is not on this worker, so the ring is a storage bound as well as a display one — the cap is why the `Done` tile cannot read more than 300, and a count shipped beside the rows could never read more than 300. `live_snapshot()` returns a third value, `active_total`: the number of tracked, non-task, not-yet-finalized sessions on that worker, computed **before** `_ST_ACTIVE_CAP` and through the same predicate the row loop uses, so the `Active` tile describes the population and the table the capped view of it. The client-side **Keep completed** filter only ever drops rows carrying a `done` stamp, so it cannot change that count. `Done`, `Cost` and `Tokens` remain totals over the rows shown, and above the cap they therefore exclude the oldest in-flight requests. Each row:

| Field | Type | Notes |
|-------|------|-------|
| `user` | str | User name (or email, or `?`) |
| `chat_id` | str | The chat the request belongs to, used to add a background task's cost to the same user's row in that chat; removed before the rows reach the browser. In a worker's Redis slice a temporary chat carries an anonymous stand-in, or nothing without `WEBUI_SECRET_KEY` (see Multi-Worker Aggregation) |
| `user_id` | str | The user the request belongs to, the other half of the fold's identity, and present on the row for that match only; removed before the rows reach the browser, exactly as `chat_id` is. Without it a background task's cost is added to whichever member of a shared chat id is on the row |
| `model_id` | str | Raw model id |
| `model_name` | str | Server-resolved display name |
| `kind` | str | `chat` or `task` |
| `status` | str | `queued` / `streaming` / `tool:<name>` / `completed` / `failed` / `cancelled` |
| `started`, `done` | float / null | Unix timestamps; `done` is `null` while in flight |
| `elapsed_s` | float | Seconds since `started`, frozen at `done` |
| `tokens_in`, `tokens_cached`, `tokens_out` | int | Cumulative token counts |
| `tools_ok`, `tools_failed`, `tools_skipped` | int | The three outcomes of the tool calls the pipe ran in a batch, as `on_tool_result` reports them: succeeded, failed, and skipped. `tools_skipped` comes from three places: a call the tool breaker refused to let through, a `future.done()` call whose result was no longer awaited when the batch got to it, and a `cancelled` call -- one the pipe started and then abandoned, so the tool itself did not fail (`_run_tool_unless_breaker_open` returns `"skipped"` for the first two, and the tracker's status table maps `cancelled` to the same counter). Either way it is a wait, not an error, and the Live tab shows it with its own glyph rather than folding it into `tools_failed`. Live rows only; `db_row` persists the first two, and the by-model and by-user tables aggregate them. |
| `cost`, `task_cost` | float | Running cost; `task_cost` is the folded-in task portion |
| `worker_pid` | int | The worker that owns the row, identified by host **and** pid: a pid is only unique inside one machine, so the same number on two hosts is two workers |

`tools_skipped` is a **live-row key only**. It is deliberately not persisted and not aggregated: `USAGE_ROW_FIELDS` and the ORM model are unchanged, and the by-model and by-user tables keep counting only `tools_ok` and `tools_failed`, so they still under-report skips. That is a deferral, not an oversight, and what is deferred is the **aggregation**, not a DDL migration: `UsageStore.ensure` reconciles against the model with `ALTER TABLE ... ADD COLUMN`, so adding `tools_skipped` to `_usage_model_columns()` *is* the migration, and rows an earlier release wrote read the new column as `NULL`, which the aggregations already treat as `0`. `test_the_usage_row_column_set_is_stable_across_a_release` is the tripwire: it builds one engine from the current model and a second from the column set an earlier release created, and fails if they stop mapping 1:1, so the day someone adds the column the suite says so out loud.

**Usage records.** With `PIPE_DASHBOARD_USAGE_COLLECT` on, each finalized session is mapped by `SessionTracker.db_row(...)` and written to the `dashboard_{suffix}` table. The table name is keyed on `(ARTIFACT_ENCRYPTION_KEY, pipe_id)` and is therefore stable across upgrades, so `UsageStore.ensure()` reconciles missing columns with `ALTER TABLE ... ADD COLUMN` before the model is published; rows written by an earlier release read their new columns as `NULL`, which the aggregations already treat as `0` (`usage_queries.py` sums each column bare, which ignores a `NULL` the same way `or 0` did, and wraps the addends in `coalesce` only where one expression meets two columns, where a `NULL` would otherwise take its row's value out of the whole sum). The window the purge deletes at is read from the persisted valve row on each pass, at the valve's declared default when that row cannot be read - the same reader and the same fallback `usage_queries.py` uses, so the purge and the Usage tab can never report different windows for one setting. The writer re-checks that table signature on its own write path, so a rotated `ARTIFACT_ENCRYPTION_KEY` moves usage rows to a new table on the next write without a restart, and the pre-rotation table is left behind for the operator to drop - the retention window does not cover it, and nothing writes or purges it once the writer has moved on. A rotation whose new table **cannot be created** is different: the writer withdraws the pre-rotation model, arms the same retry interval, and the Usage tab says `storage unavailable` until the interval elapses and a later attempt succeeds, so no rows are written to the table the operator rotated away from. `ensure()` does not move off the event loop for this: the throttled retry costs one short DDL per worker per `_US_RECONCILE_RETRY_S` (300 s) on a host that is already misconfigured, and reaching off the loop would mean changing the signature of four modules on the request path. Revisit that if a first-use DDL is measured above ~100 ms, or a deployment reports loop stalls at request completion; the off-loop route already exists as `_warm_usage_store` through `run_in_executor` (`usage_queries.py`), so it needs no new mechanism. Shutdown is the other direction of the same rule: once `signal_stop()` has been called the store is retired and starts no writer and no purge loop again -- a row finalised after the stop is still written, by a writer that drains it and exits immediately, but the purge loop does not come back, so a hot reload leaves no predecessor thread, purge task or DB connection behind. A batch the writer cannot write is not lost quietly: it is held and retried with the rows that arrive next, bounded by the batch size (and every row the hold has to shed to stay inside it is counted as dropped, as is whatever is still held when the writer stops). A pass that raises reports its row count at WARNING with the traceback -- once per outage, then at DEBUG -- and a pass that only finds the reconciliation gate closed is reported by `ensure()` itself, at the table it could not prepare. So a total that comes up short is visible in the log even when the database is the reason. A table that cannot be **created** at all -- a read-only role, a missing `CREATE` grant, an unowned schema -- is retried at most once per `_US_RECONCILE_RETRY_S` (300 s) rather than once per completed request, because `ensure()` is reached synchronously from the request path. The `storage unavailable` reason is unchanged throughout and clears within that interval once the database can create the table. The columns (`USAGE_ROW_FIELDS`, plus a generated `id`):

| Column | Type | Meaning |
|--------|------|---------|
| `id` | str(26) | Generated primary key |
| `ts` | datetime | Completion time (indexed) |
| `started_at` | datetime | Request start |
| `kind` | str(8) | `chat` or `task` (indexed) |
| `user_id`, `user_name` | str | Caller identity (`user_id` indexed) |
| `chat_id`, `session_id` | str | Conversation identifiers (`chat_id` indexed). Both are written empty for a temporary chat (as `is_temporary_chat` decides; `channel:` chats keep them), whose row keeps every other column; `UsageStore` applies this on every write, and its purge clears them from rows an earlier release wrote, keeping the rows |
| `model_id` | str(128) | Raw model id (indexed) |
| `task_name` | str(32) / null | Task type for `kind="task"` rows |
| `status` | str(12) | `ok` / `failed` / `cancelled` |
| `duration_ms` | int | Wall-clock duration |
| `tokens_in`, `tokens_out`, `tokens_reasoning`, `tokens_cached` | int | Token counts |
| `tools_ok`, `tools_failed`, `retries` | int | Per-request counters |
| `cost`, `cache_savings` | float | Billed cost, and cache savings: the provider-reported `cache_discount` when the response sends one, otherwise the pipe's own read-pricing estimate. Either way it is floored at zero and never negative, so a cache *write* — which the provider reports as a charge — contributes nothing here; the charge itself is in `cost`. |
| `worker_pid` | int | Writing worker, identified by host **and** pid for the same reason |

**Declared widths are reconciled, and a width never takes the store down.** `ensure()` compares column *names* and adds what is missing, but a release that also *widens* a `String(n)` needs more: the deployed column is still `n` wide, and the writer would start sending longer values into it — on PostgreSQL that is a rejected insert, and since a batch is one transaction a single rejection used to take all 50 rows with it. So the reconcile also runs a width pass (`_reconcile_widths`), driven by the model's declared lengths rather than a list of names, and issues the dialect's own `ALTER TABLE ... ALTER COLUMN ... TYPE VARCHAR(n)` for each column the table holds **narrower** than the model declares. A column already wider is left alone: narrowing it back would destroy the values that only fit the wider shape.

**If the database refuses the widen, recording continues.** An account without `ALTER` rights, and SQLite — which cannot retype a column at all and enforces no `VARCHAR(n)` anyway — leave the column at its on-disk width. That is a WARNING, not a gate failure: `ensure()` still answers `True`, `enabled` stays on and `record()` keeps writing, because a display name is a label and the cost is the data. The writer then fits every string to the **narrower of the declared and the reflected width**, so the row lands with the value truncated rather than rejected, and one warning per column names both widths and the `ALTER` that fixes it. A warning fires once per column per worker, not once per `ensure()`.

**A rejected row costs only itself.** `_persist_sync` writes the batch inside one transaction with a `SAVEPOINT` per row, so the first row the database refuses is rolled back to its own savepoint and every other row — each carrying its own cost — commits with it. `add_all` plus a single commit was all-or-nothing over the whole batch. Each rejection is logged at **WARNING** naming the row's `id`, `chat_id`, `user_id`, `model_id` and `cost` with the exception type, and is counted in `UsageStore.persist_failed`, surfaced through `_table_info_sync()` and copied into the Usage tab's `meta` by `run_usage_query` as `persist_failed`: a loss the operator cannot see is half the defect. The batch is never re-queued — a poison row in it would be an infinite retry loop.

**Range analytics.** The `usage_stats` action calls `run_usage_query(plugin, pipe, args)`, which validates the range, memoizes for 30 s, and runs one windowed aggregation in the store's DB executor, then hands the payload's deep copy to the default executor on the way out -- a copy of the whole user set is tens of milliseconds of Python, and the DB pool also serves the chat request path, so neither hop belongs on the event loop. The aggregation is SQL: one grouped query per window for the cards, one for the buckets, one per grouping for `by_model` and `by_user`, and the all-retained `totals` aggregate beside them, so what crosses into Python is bounded by the size of the answer -- buckets plus distinct models plus distinct users -- and not by the number of usage rows in the window. **The bucket edges are computed in Python and handed to SQL as range predicates**, and that is load-bearing rather than incidental: `ts` is a naive `DateTime` holding server-local time (`usage_ts_from_epoch` is the writer and its own docstring says the frame is), so an edge derived in SQL -- `strftime('%s', ts)` on sqlite, `EXTRACT(EPOCH FROM ts)` on postgres -- reads the column as UTC and is off by a constant on every deployment east or west of Greenwich, silently, with no log line. The edges come from the window (`b0 = int((start + off) // bucket_s * bucket_s - off)`) and the predicates are `ts >= b AND ts < b + bucket_s` on the naive column, which is the same comparison `usage_queries.py` always made. Two more consequences of aggregating in SQL: a `NULL` `kind` is coalesced to `chat` in the per-model grouping, so a model never splits into two rows, and the per-user display name is the name on that user's **newest** row (greatest `ts`, by a per-user ordering), not a collated maximum -- `max(user_name)` would keep `zoe@old` for a user who has since become `Aaron`. **`totals` is all-retained by design, not windowed**: it pairs with `meta.records` and `meta.approx_bytes`, which are all-retained counts, and narrowing it would move a number an operator reads.  The memo is keyed on the usage table, the request shape *and* both persisted valves it reports -- `(table, range, include_tasks, tz_offset_min, collect_on, retention_days)` -- so two pipes, and a rotated `ARTIFACT_ENCRYPTION_KEY`, never share an entry, and a change to `PIPE_DASHBOARD_USAGE_COLLECT` or `PIPE_DASHBOARD_USAGE_RETENTION_DAYS` is visible on the tab's next poll instead of after the 30 s TTL. The `args` keys are `range` (default `"24h"`), `include_tasks` (default `True`), and `tz_offset_min` (default `0`, clamped to ±900). All three are declared `optional(...)` in the action's schema, so omitting one takes that default and the empty `args` object `{}` is a valid call; a key that *is* present with the wrong type, `null` included, is still a `400 {"error": "missing or invalid: <key>"}` — a supplied `null` is not an omission, and `include_tasks: null` in particular would coerce to `False` rather than the documented `True`. Supported ranges (`USAGE_RANGES`) and their bucket sizes:

| Range | Span | Bucket |
|-------|------|--------|
| `1h` | 1 hour | 1 min |
| `6h` | 6 hours | 2 min |
| `24h` | 24 hours | 5 min |
| `7d` | 7 days | 1 hour |
| `30d` | 30 days | 4 hours |

On success the result is `{"available": true, "cards", "prev", "buckets", "by_model", "by_user", "totals", "meta"}`. When it cannot answer it returns `{"available": false, "reason": ...}` with one of:

| `reason` | Cause |
|----------|-------|
| `unknown range` | `range` is not one of `USAGE_RANGES` |
| `range exceeds retention` | The window is longer than `PIPE_DASHBOARD_USAGE_RETENTION_DAYS` |
| `storage unavailable` | The artifact store or usage table is not ready on this worker, including a usage table an earlier pipe version left behind whose columns could not be reconciled, and a usage table this worker's database role cannot create; the pipe log carries a WARNING naming the missing columns, or naming the table whose creation was refused and the interval before it is attempted again |
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
    pipe._multimodal_handler = None
    return pipe
```

The readiness collectors read the HTTP-session state off `pipe._multimodal_handler`,
which owns the address-vetting transport, so a mock pipe that needs a non-idle reading
gives it a handler whose `transport_session_state()` returns `active` or `closed`.

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
