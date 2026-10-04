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
├── auth.py               # ACCESS_DENIED_MD / UNDETERMINED_MD messages
├── authz.py              # Authorization chokepoint — can_view/can_act (reuses OWUI model access)
├── actions.py            # Action registry + dispatcher (authorize-first, audited)
├── http_routes.py        # Authenticated POST /api/pipe/dashboard/action (bare-bearer guard + OWUI's get_verified_user)
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
| `on_models` | Appends `{"id": "pipe-dashboard", "name": "Pipe Dashboard"}` to the model list, and ensures the row behind it exists. A retired generation does not run it at all, so its model-list refresh writes nothing and does not re-point the module globals above; `on_init` remains the re-arm path for a live generation. The hook **owns** the row's `name` and the whole of `meta`: every `meta` column is written from the row being updated, with only `meta.description` replaced, and every column outside `meta` — `base_model_id` and `params` — is borrowed from a read taken immediately before the write, and the borrow is a point-in-time read, so a change landing after it is still lost. Open WebUI's `update_model_by_id` writes a whole row, so there is no partial write; keeping a borrowed column safe means writing a *current* value, not omitting it. `is_active` is the one column the pipe owns: the reconcile runs on every pass, above the gate that stops the *listing*, and a row the pipe created active is switched **off** with the valve, claimed in `meta` under `openrouter_pipe:dashboard_switched_off_by_pipe`; the claim is what lets the round trip off → on switch it back on again, while an `is_active=False` an administrator set in Open WebUI carries no claim and is left exactly as it is. A valve row that cannot be read writes nothing at all — a failed read is not an administrator's switch — and no row is inserted while the valve is off. Ahead of that gate the same pass arms the plugin's own background loops: the abandon sweep (see Session Tracking & Usage Records) whenever `PIPE_DASHBOARD_ENABLE` or `PIPE_DASHBOARD_USAGE_COLLECT` is on, and the auto-update loop whenever `PIPE_DASHBOARD_UPDATE_AUTO` is, so a worker with the dashboard model switched off picks both up on its next `/api/models` rather than waiting for a model-list rebuild. All three starters -- sweep, publisher and auto-update -- stand down once the `Pipe` behind them is `_closed`, which is the guard the registry also puts in front of the whole `on_models` dispatch: `Pipe.shutdown` notifies plugins without removing them, so a retired instance can still be handed a hook whose `on_shutdown` has already run, and a task re-armed there would be owned by nobody. `_closed` alone, never `_draining` or `_closing`: a merely draining instance has not had its `close()` run, its workers are still wanted, and suppressing them would leave a live instance with none |
| `on_request` | Intercepts requests sent to the `pipe-dashboard` model ID; starts live-session tracking for every other request |
| `on_emitter_wrap` | Wraps the stream emitter to capture usage snapshots and tool-start events for the Live feed |
| `on_tool_result` | Records on the live session the outcome of each tool call the pipe runs in a batch |
| `on_request_retry` | Increments the live session's retry counter |
| `on_generation_complete` | Finalizes the live session and persists a usage row when collection is enabled. A video turn reports its own terminal state, so a video row carries the job's real tokens and real cost, and a failed or stalled video row reads Failed rather than Completed |

**A Fusion chat in a channel keeps the answer panel but loses the panel card.** `streaming_core.py` applies the `<details type="fusion_answer">` wrapper to `assistant_message` *before* it builds, records and emits the Fusion answer item, so the item, the terminal `output` array and the string the turn returns all carry the same wrapped text — which matters because the `is_channel_chat` guard routes a channel through `output` and `__channel_emitter__` folds that array in place of `content`, with no `Chats` row behind it to fall back on. `__channel_emitter__` has branches only for `chat:completion`, `response:completion`, `files`/`chat:message:files` and `chat:message:error`, so `fusion:event` and `embeds` reach no branch at all. The same row is where nothing else would repair a truncated stored `content`: the terminal snapshot writes `content` only on a turn Open WebUI is not already holding, and on a Continue Open WebUI holds it — so the panel is added and the stored answer is left to the writer that already owns it.


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
  │    (idempotent; retried on a live on_models, and from on_init)
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
  │  (the ACK ordering guarantees OWUI has registered the session first; the payload
  │   carries {"pipe": "<this panel's Pipe.id>"}, which is what the handler resolves)
  ▼
Subscribe handler: resolve the install the payload named against this worker's own
  │  registrations, authorise against THAT install's model row, then
  │  sio.enter_room(sid, "pipe_dashboard_viewers:<Pipe.id>") and pin the
  │  authorized id to the socket.io session
  ▼
Publisher loop on the worker holding the socket:
  │  aggregates stats → sio.emit("openrouter:pipe_dashboard", payload,
  │                       room="pipe_dashboard_viewers:<Pipe.id>", ignore_queue=True)
  ▼
Dashboard JS updates DOM sections as payloads arrive
```

**Key insights:**

- **Room membership is the entire viewer state.** Socket.IO removes a socket from its room on disconnect and deletes the empty room — no registry, no keys, no TTLs, no custom HTTP surface. The room is named per install (`viewers_room(pipe_id)`), so two installed copies on one worker never share a viewer set.
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
| `SUB_EVENT` | `openrouter:pipe_dashboard:sub` | client → server | Dashboard requests to join its viewers room, carrying `{"pipe": "<Pipe.id>"}` (emitted inside the `user-join` ACK). An id this worker has no registration for is refused with `denied` — never resolved to whichever install registered last |
| `DASHBOARD_EVENT` | `openrouter:pipe_dashboard` | server → client | Aggregated stats payload pushed to the room |
| `DENIED_EVENT` | `openrouter:pipe_dashboard:denied` | server → client | Sent to a socket refused the room (no read grant, or grant later revoked); the refusal also removes the socket from the room, whichever of the two refusals sent it |
| `CONFIG_EVENT` | `openrouter:pipe_dashboard:config` | server → client | One valve write, to the room: `{"rev": <Function.updated_at>, "change": "<pid>-<n>"}`. `change` is minted once per save (`new_config_change`, `dashboard_socket.py`) and threaded to every carrier of that save — this event, the save response, and the `function.valves_updated` event the pipe publishes — so one save announced twice reads as one change at the tab. The pid qualifies it because `sio` runs under an `AsyncRedisManager` when `WEBSOCKET_MANAGER=redis`, so an emit minted on one worker reaches tabs attached to another; the value is compared for **equality, never for order**, so neither cross-worker delivery nor a restart can wedge the tab |
| `VIEWERS_ROOM_PREFIX` + `viewers_room(pipe_id)` | `pipe_dashboard_viewers` / `pipe_dashboard_viewers:<Pipe.id>` | — | The Socket.IO room whose membership is the entire viewer state. Prefixed, not fixed: membership is per install, so a viewer of one copy's dashboard is never in the other copy's room |

### Data Tiers

Data collection is split into tiers to balance freshness against collection cost:

| Tier | Frequency | Data | Collection Cost |
|------|-----------|------|-----------------|
| **Identity** | Tick 0 + every ~16s | Version, pipe ID, worker count | Negligible |
| **Fast** | Every 2s | Concurrency, queues, rate limits, sessions, uptime, the emitting worker's PID | Cheap (in-memory reads) |
| **Medium** | Every ~16s | Models catalog status, system health | Moderate (subsystem inspection) -- run on the event loop's **default executor**, never on the loop itself and never on the artifact store's database pool, which the chat path writes through and which this tier has no reason to occupy: it opens no database session of its own, so a private single-worker pool would buy nothing but a thread to shut down. The session-log figures it reports are counted, not walked: the buffer and record counts are per-deque lengths and the size is a running per-request total read under `SessionLogger._state_lock`, so the work the tier does does not grow with what the buffers hold. |
| **Slow** | Every ~60s (30s recompute floor) | Storage stats, configuration, plugins | Expensive (DB queries), run on the artifact store's DB thread pool rather than the request loop; on a publisher-owned single-worker thread when the store has no pool (closed by `close()`, or a build that keeps failing) |

A new viewer joining the room resets the tick counter, so the next emit carries the **full** tier set for an instant first paint — within ~2s when a worker is already streaming, or within one idle poll interval (~5s) on a cold start (no dashboards were open). The slow tier is additionally guarded by a wall-clock floor: rapid re-subscribes reuse the cached slow payload instead of re-running the storage queries.

**The Config tab's live-update guard is three early returns, one per suppression class.** `cfgOnEvent(rev, change, state)` in `config_tab_assets.py` drops the event, and applies nothing, when a save is in flight or the announcement carries no revision; when the event carries a change identity the tab has already applied, or one older than a revision the tab holds (strictly — a second save inside one second arrives at the revision already applied and must still fire); and, for an announcement with no identity at all, when it is at or below either revision the tab holds (non-strictly, which is what stops the ~16 s `cfgRev` republish from becoming a `config_get` every sixteen seconds, forever) and its digest is either absent or the one the tab recorded. Everything else is applied: a quiet reload when no edit is staged, the conflict banner when one is. The tie-break is **identity equality, not the revision** — the revision alone cannot separate two saves in one second, and minting the identity per announcement would make one save (announced twice, directly and by the pipe's own valve-event sink) read as two changes.

Every **medium** payload also carries the two keys the Config tab's live-update guard reads: `cfgRev`, the stored row's own `Function.updated_at`, and `cfgState`, a 16-hex-character SHA-256 digest of the stored `function.valves` ciphertext (`config_state_digest`, `dashboard_socket.py`, reached through the same `read_config_rev` select). Both are `null` when the row cannot be read. The revision alone is not enough to notice a write: it is a whole-second column, so two writes inside one second -- two workers, one second -- leave it where it was, and the republish would read as "nothing new" forever. The digest moves on every write, so the tab can still be told; its cost is one hash of a few hundred bytes every ~16 s per emitting worker. A tab records the digest it last saw and drops a republish that repeats it, records the first one it is shown without prompting (otherwise every tab reloads once on every load, and once more after every save it applied by another route), and applies the change when the digest differs from the one it recorded.

**The emit gate's cost.** The persisted-row read runs on the live-feed cadence, so a worker with a dashboard open pays one `Functions` primary-key read per tick (one row, no join, no scan) on top of the tier work above, and it is paid before the emit rather than after it — a database slow to answer delays the tick rather than dropping it. One read serves both consumers: the re-authorization that precedes the emit decides the gate and hands its answer to `emit_dashboard`, so the two are not the same read done twice. That is the trade this fix makes deliberately: the alternative is a gate answered from a value the operator cannot reach. To measure it on a live deployment, time `emit_dashboard` with the dashboard open against the same worker with it closed (the closed path returns before the read), over at least a minute of ticks, and compare the two means against the tick interval — the read is only worth caching if it is not comfortably inside it. The follow-up if it is not is a `function.valves_updated`-driven cache of the switch, which is its own item; no cache is built here, because a cache is a new subsystem and a field read is the thing being corrected.

The `runtime_metrics.py` module implements each tier as a separate collector function. Collectors read directly from pipe internals (`ctx.pipe._circuit_breaker`, `ctx.pipe._request_queue`, etc.) — they have full access via `PluginContext.pipe`.

### Multi-Worker Aggregation

In multi-worker deployments (multiple uvicorn workers behind a load balancer), each worker only sees its own process state. The dashboard uses Redis for cross-worker aggregation; every worker runs the same background task (`dashboard_publisher.py`) in one of three modes:

1. **Emitting** — this worker has local members in the `pipe_dashboard_viewers` room. It renews the `{ns}:dashboard:active` flag, writes its own slice, reads each distinct worker slice from Redis once (guaranteeing its own is included even before its first write lands: when the read comes back without naming this worker, the self-heal re-appends **the payload the write already built**, rather than collecting the worker payload a second time), aggregates them, merges the tiered collectors, and emits to the room. Delivery is local — the viewer's socket lives on this worker.
2. **Publishing** — no local viewers, but another worker set the active flag: write this worker's slice to `{ns}:dashboard:worker:{host_tag}:{pid}` every 2s so the emitting worker can aggregate it. A pid is unique only inside one kernel, so a two-host deployment has two processes claiming the same number and the second write would replace the first: the loser's row is destroyed before anyone reads it, so the aggregate reports one worker, one host's live sessions vanish, and nothing is flagged. `host_tag` is a memoised sha256 prefix of the hostname -- hashed, so no raw hostname reaches a Redis key or a dashboard payload. The emitting worker's self-heal, which re-appends the payload it already built for the write when the read came back short, compares host *and* pid for the same reason. On the idle→active transition the emitter waits ~1s so freshly woken workers land their first slice before the first aggregate.
3. **Idle** — no viewers anywhere: one Redis `EXISTS` per 5s, woken instantly via the `{ns}:dashboard:wake` pub/sub channel — near-zero overhead.

A worker's slice expires 10s after it is written. Besides that worker's collector figures it carries its live session rows, the count of tracked non-task sessions it is running (`sessions.live_active`, computed before the row cap, so the dashboard's `Active` tile is not capped by the display bound — its own ceiling is `_ST_ACTIVE_HARD_CAP`, the tracked-entry bound, which no healthy worker approaches), and the cost of each finished background task whose chat's row is not on that worker, both keyed by chat id **and** user id, so the emitting worker can add a task's cost to its own user's row in that chat; the emitter then drops the ids from every row before anything reaches the browser. The row list itself **is** a capped view — 30 in-flight rows per worker, newest first, then 300 finished rows per worker, and the caps apply to the chat population: a task entry occupies a slot in the worker's recent ring without ever being a row, and is evicted only when the chat rows fill their own cap — and the aggregate keeps 300 across the cluster, dropping finished rows first because the sort puts actives ahead of them. The `tc` map's inputs did not shrink with that: a task entry in the ring is exactly what publishes a finished task's cost for a chat row living on another worker, so the ring holds up to 600 entries (300 per kind) rather than 300. The count is summed from each worker's own number rather than read off the merged rows, so `_PD_SESSIONS_CAP` cannot cap it; a slice from a worker on an older release carries no count at all, and that worker is left out of the sum rather than counted as contributing zero, so a rolling deploy reports a partial sum instead of a silently-lowered total. The `tc` map's key is the two ids joined by `\x1f` (U+001F, the separator `requests/task_model_adapter.py` also uses), as a flat string rather than a nested object because the slice is JSON: `"c1\u001fu1": 0.004`. A temporary chat is never written under its own id: its rows and task costs carry an anonymous stand-in instead, an HMAC of the chat id keyed by `WEBUI_SECRET_KEY` under a label only the dashboard uses, so it is neither the chat id, nor the socket id Open WebUI builds that id from, nor the key the pipe sends OpenRouter. Only the chat half of a temporary chat's `tc` key is replaced by the stand-in; the user half is the user id either way. Without `WEBUI_SECRET_KEY`, a temporary chat's rows carry no id and none of its task costs are published, so a task cost it ran on another worker is left out of its row. Saved and channel chats keep their own id. The worker's own memory always keeps the real id; only what it writes to Redis carries the stand-in.

**Every Redis operation the publisher issues is bounded** by `_PD_REDIS_OP_TIMEOUT` (1.0 s), hardcoded beside `_PD_PUBLISH_INTERVAL` and deliberately not a valve: the pipe builds its own client, so Open WebUI's `REDIS_SOCKET_TIMEOUT` / `REDIS_SOCKET_CONNECT_TIMEOUT` do not reach it, and an operation with no bound of its own blocks the event loop — the same loop that serves chat requests — for as long as the far end stays quiet. The bound is shorter than the 2 s tick so the failure is reported inside the tick that saw it. The scan is bounded as a whole and its generator is closed on the way out, so a timeout cannot leak the connection, and the live-feed liveness ping is already bounded more tightly still (0.25 s) and now runs **before** the first read: a server that has stopped answering is found in one ping rather than after the whole read budget has been spent on it. Nothing is bounded by a wall-clock comparison — every bound here is an `asyncio.wait_for` around the operation itself.

In single-worker mode (no Redis), the worker with viewers emits directly from its local collectors — the same payload shape, minus the multi-worker `workers` table. In multi-worker mode an incomplete read is never dressed up as a complete one, and neither is a collect that failed: the emitter marks the payload `degraded` rather than presenting the workers it could still see as the whole cluster, and while that flag is set the dashboard's footer makes no worker-count claim at all, because no payload key carries the last-known total.

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

The top-level `pid` is the emitting worker -- the process that built this payload and holds the viewer's socket -- on every path: the Redis aggregate, the degraded replay and the single-worker emit all overwrite it with `os.getpid()`, so it never names a worker the read happened to return first. It is a different number from the same field inside a `workers` entry, which is that worker's own pid. `uptime_s` is not a process age at all: it is `time.monotonic() - PROCESS_START`, and `PROCESS_START` is a module-level constant in `_collectors.py` that a fresh module re-evaluates, so it is the age of one *generation* of the pipe and restarts at zero on every in-place reload. The top-level value is a **max** over the workers, so it is the oldest *generation's* age — and after a rolling deploy, where each worker restarted at a different moment, not even the oldest worker's. The key is unchanged; the panel labels it accordingly.

The `sessions.in_flight` sample counts a request from the moment it enters the pipe until its job's cleanup tail has finished, so a request whose answer has already been delivered still reads as in flight while it is shutting down. `sessions.live_active` is a different population: the tracked, non-task, not-yet-finalized sessions, counted **before** the row cap, so the `Active` tile reads 4 in the sample above even though `sessions_live` shows one row. The tile's own ceiling is `_ST_ACTIVE_HARD_CAP`, the bound on tracked entries rather than on rendered rows: a worker that reached it would have evicted its least-recently-seen session as `failed` to stay under, so the tile tops out there while the table is still the 30-row view. It is absent from a payload whose worker is on an older release.

`degraded: true` appears on **every** tick whose Redis read failed **or timed out** -- every client operation the publisher issues is bounded by `_PD_REDIS_OP_TIMEOUT` (1.0 s, hardcoded beside `_PD_PUBLISH_INTERVAL`, no valve), because the pipe builds its own Redis client and Open WebUI's `REDIS_SOCKET_*` options do not reach it -- and the tick asks the server whether it is alive **before** it reads anything, so a server that has stopped answering is found in one bounded ping instead of after the whole read budget. A timeout is a failure, not a quiet event: it lands in the same degraded path a raised read lands in. **or whose own worker slice could not be collected** -- the first one and every one after it, with or without a cached set. On such a tick the emitter replays its last known worker set rather than presenting a partial cluster as the whole one, and says so in the banner instead of quietly dropping to a single worker. A collect failure sets the flag for the same reason and does not empty the payload: the peers' slices were read successfully and are real, so they are still aggregated and shown. A set that is missing this worker is never written into the replay cache on such a tick, so the degraded replay keeps naming the last set that did -- otherwise the first failing tick would leave a short cluster as the last-known-good one long after the fault cleared. The replayed set is age-bounded: it carries the timestamp of the successful read that produced it, and once that is older than three times `_PD_KEY_TTL` (each worker's own Redis key has by then lapsed three times over, so nothing in the set can be vouched for) the replayed set is dropped, the payload reports only the emitting worker's own slice (`worker_count: 1`), and it still reports `degraded: true` — the banner is the only thing distinguishing that one-worker view from a genuinely single-worker deployment. An empty *successful* read is not degradation and never sets the flag. Storage payloads carry `state` (`connected` / `unavailable` / `degraded`) so the dashboard can distinguish "not initialized on this worker yet" from a genuine failure; the collector wires the shared DB itself on first use, and `unavailable` is also the answer for a store whose pool is gone — a closed store reports it rather than claiming a connection it cannot make, which is why `db_connected` asks for the executor and not only for the session factory and the item model; and by-type/by-model "Least/Most recent" columns are access times (the retention sweep touches `created_at` on every read).

On tick 0, all tiers fire simultaneously for instant dashboard population. The JavaScript checks key existence and updates only the sections whose data arrived in that tick. The payload arrives raw — direct custom emits do not use the `{chat_id, message_id, data}` envelope of OWUI's shared `events` channel, so no client-side filtering is needed.

### Access Model & Live-Mode Requirements

There are no capability keys — access rides Open WebUI's own model access control, but the two arms delegate to the two DIFFERENT answers Open WebUI itself gives, and each then applies its own arms on top: **read** calls `check_model_access` first (`utils/access_control/__init__.py:381-386`) and only then adds an admin / owner / grant chain of its own on the arms that call leaves open, while **write** reimplements the three-term disjunction Open WebUI's mutating model routes enforce — `routers/models.py:888`, `:956`, `:1077` and `:1126` — with the admin term asked first, and none of those four carries a `BYPASS_ADMIN_ACCESS_CONTROL` term. So the write answer is deliberately NOT the `write_access` formula `get_model_by_id` reports (`routers/models.py:725-733`), which does carry that valve: the pipe answers what Open WebUI's routers enforce rather than what its editor's affordances display, and a hardened install sees the dashboard's write surface agree with the UI beside it:

1. **The command, the live feed, and read-only actions require a *read* grant (viewer).** The `dashboard` command handler and every socket subscribe resolve a fresh `UserModel` and gate on `authz.can_view` → OWUI's `check_model_access` (honoring owner, admin, direct-user grant, group grant, `user:*` public, and `BYPASS_MODEL_ACCESS_CONTROL`). An ungranted socket is refused the viewers room and told so via a `denied` event; the publisher re-authorizes every local viewer **before every payload it emits**, and evicts any whose grant was revoked, so revocation takes effect on the next tick -- at most `_PD_PUBLISH_INTERVAL` -- without a reconnect. Nothing about that decision is reused across ticks, which is what keeps the bound above exact, and its cost is a known number: **three database reads per person per tick** -- the user row from `authz.resolve_user`, then Open WebUI's own `check_model_access` reads the member's groups and the grant -- plus **one** model-row read for the whole tick, hoisted out of the loop by the `model_read` latch. A person holding several tabs is decided once per tick, so the count follows people and not sockets, and the whole thing costs nothing at all while the dashboard is off, because the dedup, the model read and the grant check all sit below the `not enabled` branch. The sweep evicts on a **definite** refusal only: a check that could not complete — a user row that raised rather than came back empty, a grant check that raised something other than OWUI's `HTTPException` — leaves the viewer in the room, is announced once as undeterminable, and is decided again on the next tick, while the **subscribe** path still refuses on the same failure, so nothing unverified is ever admitted. A row that reads as *absent* is a positive answer and evicts as usual. The dashboard-off arm sits above that read, but it is a **committed** disable or nothing: it reads the same stored valve row through `_socket_dashboard_state`, which carries the verdict and whether the row could be read at all, so an unreadable row evicts nobody on this path or on the `function.valves_updated` event, and is announced once as undeterminable instead (the warning is latched, so a thirty-second outage is one line rather than fifteen). An undetermined tick decides no viewer at all — the per-viewer grant sweep is below it and returns with it — and the next tick, at most `_PD_PUBLISH_INTERVAL` away, decides the whole room again. That is the asymmetry to keep in mind while reading the rest of this page: **new work is refused on a row that cannot be read, and an admitted viewer is not evicted for one**, because the valve row's own gates (`_pipe_dashboard_sub`, `emit_dashboard`, the action route) refuse fail-closed on an unconfirmed gate, while eviction is a punitive act against somebody already admitted. The action route keeps the same distinction rather than collapsing it: it asks `can_view_known`/`can_act_known` and fails closed on the third answer, so a store fault is `500 {"error": "access_undeterminable"}` audited as `outcome=undeterminable`, with no handler invoked and nothing served from a cached or in-memory answer, while `False` stays `403 {"error": "forbidden"}`. That asymmetry is deliberate on both surfaces and is the point of the tri-state existing: new work is refused on a check that could not complete, and work already admitted is not taken away for one. Already admitted and not yet known to be lost is not the same as refused. The distinction is worth having because neither upstream answer is a verdict: OWUI's `get_current_user` turns only a `None` into a 401 and lets a database error propagate, and `Users.get_user_by_id` returns `None` for a missing row and raises for a real failure. That sweep reads the dashboard's own model row **once per tick** (`authz.resolve_view_model`) and decides every viewer against that one read, rather than re-reading the same row once per viewer; the access decision itself is still made per viewer, per tick, except that a second socket of a user already decided on that tick reuses that verdict; when that single read cannot be made, no viewer is decided from it at all and the whole room is re-decided on the next tick.
2. **State-changing actions require a *write* grant (operator).** Actions are invoked over an authenticated `POST /api/pipe/dashboard/action` route and gated on `authz.can_act` → the three-term write disjunction Open WebUI's own mutating model routes enforce (`admin` / owner / `has_access(..., "write")`), the four of which -- `toggle_model_by_id` (`routers/models.py:888`), `update_model_by_id` (`:956`), `update_model_access_by_id` (`:1077`) and `delete_model_by_id` (`:1126`) -- all evaluate and none of which carries a `BYPASS_ADMIN_ACCESS_CONTROL` term. The pipe **reimplements** that disjunction with the admin term first, so it answers a different question from the read gate above and gets a different answer from it on purpose; a next reader copying `routers/models.py:725-733` -- the `write_access` the model editor reports, which *does* carry that valve -- into the write arm is copying the wrong line. The reach is `BYPASS_ADMIN_ACCESS_CONTROL=False` with two or more admin accounts: the second admin, who is neither the model row's owner nor holds a `write` grant, still opens the panel -- read is unaffected -- and is still admitted for every state-changing action, because a gate stricter than the routers beside it would be the one surface where a hardened deployment disagrees with its own UI. Classify each action by **effect**: side-effect-free introspection is `read`; anything mutating shared state (clear a cache, trigger an update) is `write` — the same consume-vs-mutate rule OWUI applies across models, KBs, tools, and channels. `register_action` defaults to `write` (fail-restrictive). OWUI's editor pairs a read grant with every write grant, so an operator can always view.
3. **Five actions require the `admin` role on top of the grant.** `config_get`, `config_set`, `update_apply`, `update_restore` and `update_snapshot_delete` are registered with `admin_only=True`, which `dispatch_action` enforces with `user.role == "admin"` after the grant check and before the unknown-name branch. This matches Open WebUI, whose own valve routes are all `Depends(get_admin_user)`; the gate lives in the dispatcher rather than in a handler body because a reconcile-swapped copy of `actions` answers through whichever `dispatch_action` is loaded, and a body-level check would be absent from a freshly imported module.
4. **The action route is CSRF-safe unconditionally.** The header is parsed exactly as OWUI's own `HTTPBearer(auto_error=False)` (`utils/auth.py:175`) parses it: `get_authorization_scheme_param` partitions on the first space, strips the credentials and case-folds only the **scheme**, so `bearer` / `BEARER` / `BeArEr` and a padded `Bearer␣␣<token>` or `Bearer <token>␣` are all admitted; the **credentials are never case-folded**, because they are case-sensitive base64url and a lowercased signature never verifies. That is the host's own parser rather than a second copy of it, and it reads the header and nothing else -- the header-only property below is preserved deliberately, not incidentally. Its authentication is **header-only** (`Authorization: Bearer <token>`, reusing OWUI's `decode_token` + `is_valid_token` + a fresh `Users.get_user_by_id` + OWUI's `WEBUI_AUTH_TRUSTED_EMAIL_HEADER` comparison + the last-active refresh, the comparison in OWUI's own order: the header value is lowercased, compared against the stored `user.email`, and only when the setting is on **and** the header actually carries a value, so an API-token caller that sends no such header is unaffected); it never reads the session cookie, so a cross-origin page cannot forge a call with the victim's token regardless of OWUI's CORS/SameSite settings -- the trusted-identity comparison is itself header-only, so adding it costs that property nothing. Every outcome is audited (user, action, outcome, and the peer address of the connection -- `request.client.host`, exactly what Open WebUI's own audit record writes at `utils/audit.py:299`, with no forwarding header consulted; write args included, secret values masked at any depth, and for a value under a secret-named key at any depth) -- including the `unavailable` outcome the `503 {"error": "action unavailable"}` responder writes when no dispatcher could be resolved, whose write args are redacted exactly as on every other outcome. The mask's walk descends 32 levels and no further, because `edits` is caller-supplied and `json.loads` nests 200 deep happily; a container the walk will not descend into is written as a `<redacted TYPE>` marker naming its type, exactly as any other unvouched container is, so what it holds is never written at any depth. The 200-character scrub that follows the masked args is a log-cost control -- it keeps one audit line bounded -- and is not what keeps a value out; the mask above is. A value whose own key is a declared non-secret valve or a protocol field the action defines keeps the value the operator wrote -- that setting is stored and shown in cleartext anyway, so a secret typed there is logged as typed; under any other key, and inside any nested list or dict, the value is written as a `<redacted TYPE>` marker instead. When the valve schema cannot be read at all -- the pipe object publishes no `model_fields` and the fallback import of `Pipe` raises -- the mask runs against an **empty** field map, so **every** submitted value that is not a protocol field becomes a marker: with nothing known about any key, nothing a caller supplied may be written, and the key names and the revision survive so the line is still worth reading. The one value this leaves in cleartext is the protocol-vouched one: `echo`'s `message` keeps the operator's own text under this rule exactly as it does on every other path, which is the protocol-vouch rule above behaving as it was written rather than a hole opened by the fail-closed path. Whether that peer is the caller's real address is the operator's startup decision, not the pipe's: Open WebUI's own launcher passes `forwarded_allow_ips='*'` (`open_webui/__init__.py:88,108`, `backend/start.sh:74,111`), and uvicorn's `ProxyHeadersMiddleware` then rewrites `scope["client"]` from `X-Forwarded-For` before any route sees it. The pipe reads no forwarding header of its own, so it stops adding a second, weaker, caller-chosen source; it cannot undo the trust decision made at startup. On an install with `FORWARDED_ALLOW_IPS` pinned to the real proxy, the recorded address is the true peer. A secret under a non-secret valve is therefore visible in the log by construction, which is why that valve must not be used as a place to keep one.
5. **The whole interactive dashboard requires the same-origin iframe setting.** The dashboard reads the session token from `localStorage`, which only works when Open WebUI's **Settings → Interface → "iframe sandbox allow same origin"** is enabled — the identical requirement as the [OpenRouter Fusion live panel](openrouter_fusion.md). With it off, the iframe runs with an opaque origin: the socket never opens *and* every server-backed action fails for lack of a token, so the Config, Usage, and Update tabs are inert, not just the live feed (the Update tab detects this and points at the setting). Native browser dialogs are additionally blocked in the sandbox regardless of settings, which is why every dashboard confirmation is an inline click-again button. If a restrictive `IFRAME_CSP` is configured, the same policy documented for Fusion applies (`script-src 'unsafe-inline'` + `connect-src 'self'`).

That same payload carries per-request rows for live and recent requests that name **every other user's** display name, model and spend, not only the aggregate operational counters (concurrency, queue depths, breaker trip counts, uptime, worker PIDs). A `sessions_live` / `sl` row is one request, with these keys: `user` (the requester's `user_name`), `model_id`, `model_name`, `kind` (`chat` or `task`), `status`, `started`, `done`, `elapsed_s`, `tokens_in`/`tokens_cached`/`tokens_out`, `tools_ok`/`tools_failed`/`tools_skipped`, `cost`, `task_cost`, `worker_pid`, `chat_id` and `user_id`. The fold that adds a background task's cost to a chat row matches on `(chat_id, user_id)` together and on time: the parent is the latest-started turn that was already running when the task started, or the earliest same-key turn if the task predates them all, so one member of a shared chat id never carries another's task spend and an overlapping turn never carries a task that began before it did; both ids are removed before the payload is emitted. A row is none of that — no chat content, no secrets. **The rule the surface follows is that a stable account identifier never reaches a viewer**: no key carries a principal's Open WebUI account id at any depth, and the stored rows keep theirs because the id is the join key. `chat_id` is the documented exception and is not an account identifier — it is the conversation a request belongs to, and it is what makes a shared channel's rows joinable at all; a `channel:` chat id is visible to every member by construction. The connection bar includes **Disconnect** / **Connect** buttons: disconnecting tears down the socket (Socket.IO removes it from the viewers room server-side automatically); reconnecting re-runs the connect → `user-join` → subscribe sequence. When the last viewer disconnects, all workers return to idle within seconds.

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
5. **Run + audit** -- invoke the handler and audit the terminal outcome (write args are included in the audit line with secret values masked, read args are not). A failed write records the exception *type* -- and the text of the pipe's own `_ClientMessage` -- but never a traceback, because the traceback carries the rejected input verbatim: pydantic formats the failing field's `input_value` into the `ValidationError` message that `exc_info` prints. The recursion inside a submitted value stops at depth 32, which is a DoS bound and a masking floor at once: a container the walk will not descend into is written as a `<redacted TYPE>` marker rather than handed back, so nothing below it reaches the line. The 200-character scrub on the audit line is a line-length rule, not a secrecy rule, and is not what masks a value.

| Status | Envelope | When |
|--------|----------|------|
| `200` | `{"ok": true, "result": <dict>}` | Handler succeeded |
| `400` | `{"error": "missing or invalid: <key>"}` | `args` failed schema validation: a required key is absent, or a supplied one (including a `null`) has the wrong type. A key declared `optional(...)` and simply omitted is *not* a failure -- the handler's own default is used |
| `400` | `{"error": "action failed", "detail": <sentence>}` | A `config_set` edit the *stored schema* rejects: `detail` is `the stored settings would not accept: <field names>`, naming the fields and never the values. The cause is the value the client supplied, so a `4xx` on this route is only ever a fault the requesting client could have avoided. A `config_set` is the only action that raises this |
| `403` | `{"error": "forbidden"}` | Caller lacks the required grant, or holds it but is not an `admin` and the action is one of the five admin-gated actions |
| `404` | `{"error": "unknown action"}` | No action registered under that name, either in the copy this route holds or (see the reconcile retry below) in a freshly re-imported one |
| `404` | `{"error": "dashboard_off"}` | The stored valve row says `PIPE_DASHBOARD_ENABLE` is off, or cannot be read at all. The status is B242's and does not move; the code is its own, because `unknown action` carries the wrong remedy for this state -- it tells the administrator to reload a panel that reloads exactly the same way -- and because a genuinely unregistered name (a tab left open across a hot update) still needs that advice and keeps that code |
| `429` | `{"error": "rate limited"}` | Second call within the 1 s per-`(user, action)` window |
| `500` | `{"error": "action failed", "detail": <cause>}` | Handler raised.
| `500` | `{"error": "access_undeterminable"}` | The ACL could not be READ, so the pipe decided nothing and no handler ran. A store fault, not a verdict: the same `500` Open WebUI's own admin routes answer when their database read fails (`utils/auth.py:455` lets `Users.get_user_by_id` propagate), and a distinct code from `action unavailable`, which means this worker has no dispatcher at all. The audit records `outcome=undeterminable`. It is latched on `(user, action)` for 300 s, because a one-second poll during a thirty-second outage would otherwise write thirty WARNING lines that bury the fault they report. Fail-closed: the same call answers `200` as soon as the store does | `detail` is a sentence naming the cause for a `config_set` refusal -- the store refused the write -- and the exception's class name otherwise. |

**Built-in actions:**

| Action | Permission | Schema | Returns |
|--------|:----------:|--------|---------|
| `whoami` | `read` | — | `{user_id, role, can_view, can_act}` |
| `echo` | `write` | `{"message": str}` | `{"message": <echoed>}` |
| `usage_stats` | `read` | `{"range"?: str, "tz_offset_min"?: int, "include_tasks"?: bool}` | The Usage-tab analytics dict (see [Session Tracking & Usage Records](#session-tracking--usage-records)). Its `by_user` rows name each person by `user_name` and carry no account id: the rollup is keyed and sorted by the id **server-side**, and only the name crosses the projection. The table renders ten of them, so the query returns ten -- `LIMIT 10` on `ORDER BY cost DESC, user_id` -- and `meta` carries what the wire no longer does: `meta.user_count`, the number of distinct users in the window, counted by a `COUNT(*) OVER ()` that is evaluated before the `LIMIT`, and `meta.by_user_sums`, the same nine numeric columns -- the eight the `by_user` rows carry plus their `task_cost` -- summed over **every** user in the window. The client subtracts the ten shown rows from those sums to build the roll-up, which is why the row can say `(total - shown) + ' others'` rather than `rest.length + ' others'`: the rows past the tenth are not on the client to be counted. |
| `update_check` | `read` | `{"force"?: bool}` | The Update-tab status dict: the installed and latest versions, whether an update is available, and the stored `snapshots` list. `latest` is `None` for three reasons — a cold check on a worker that has never reached GitHub, a check that failed with no last-known release to fall back to, and a check that failed more than `_PD_UPDATE_STALE_RELEASE_S` after the release it would have fallen back to was fetched, at which point the fallback is withdrawn rather than served. `checked_at` is the instant of the call that ran whatever its outcome, and the failure's onset rides beside it in `last_check_error.ts`. Each snapshot row carries the acting user's **name** in `actor_name` (`auto` for the auto-updater) and never the account id; `force` is honoured only for an `admin`, so a viewer can ask the question but not trigger the fetch. `auto.this_worker` carries a parked release's version, code, timestamp and message; the pause ages out and the entry is dropped from the map it came from, so a `check()` more than `_PD_UPDATE_AUTO_PAUSE_S` after the pause reports the last attempt under `code`/`message` instead of naming a `paused_for_version`. |
| `config_get` | `read` + `admin` | — | Valve specs with current values (secrets masked to a set/not-set flag, plus an opaque `secret_fingerprint` per secret — a keyed HMAC of the stored value, not reversible to it and not a part of the key) and the current config revision, plus a `config_unreadable` boolean that is `true` when the stored settings could not be read at all. The snapshot also carries `reset`, a sorted list of the stored values the current schema rejects; the tab counts them in its note bar and the pipe log names them, and the next save is what removes them. A name the schema no longer carries at all is in that list too, and is named twice rather than once: here for the Config tab's note bar, and independently by the `valve_salvage` scan the boot path runs on the decrypted row, which names every stored key outside `Valves.model_fields` and never quotes the value it was holding. When the stored configuration cannot be read, the action still answers: the tab renders the in-memory values under a banner with Save disabled, rather than a panel that will not draw or a default-filled view passed off as what is stored. A stored set that will not decrypt is told apart from an empty one by the raw column probe below, so it reports `config_unreadable` instead of rendering factory defaults |
| `config_set` | `write` + `admin` | `{"edits": dict, "rev": int \| str \| None, "base"?: dict}` | Persists the changed subset and returns the new revision and the `change` identity minted for it (`"<pid>-<n>"`, the same value the `CONFIG_EVENT` announcement carries, so the tab drops its own save by identity rather than by `inflightSave` winning a race), a `reset` list naming the stored values the current schema rejected, and the stored value of every valve it wrote, secrets excluded, read back the same way `config_get` reads — or, when that read did not succeed, no value for any edited setting, and the tab keeps the admin's own edit; the write is committed either way — plus a `secrets` block of `{name: {set, stored, fingerprint}}` for the secret valves the save named — the same two flags and the same fingerprint `config_get` reports, read from the row that was written rather than from the edit. The tab's Clear control and its hint are driven from that block, because the store's answer and the edit's are not the same: clearing a key that sits under an environment default leaves the setting configured, and a clear that removed nothing leaves nothing to clear. A save that names no secret carries no entry for it; a save whose echo is dropped carries no block at all, and the tab then keeps the projection it had — or a conflict payload when the client's revision was stale. When the read-back that follows a committed write is itself not readable, that drop is the whole answer: `values` is `{}`, the `secrets` and `post_reset` blocks are empty, and the write stands, because a value the pipe did not read out of the store is not a saved value. Two administrators saving at once are serialised by a per-pipe lock held across the read, the merge, the write and the echo, with the revision re-read inside it, so one of them takes the conflict path; that lock is a module global, so on a multi-worker install it is a lock per worker and nothing more, and the save is additionally held by a Redis lease under a lock name of its own (`config_lock`, deliberately not the `update_lock` a hot update holds -- sharing it would put a valve save in contention with a code swap and surface the wrong sentence). The lease is taken before the revision read, because the read is the window, and released the moment `Functions.update_function_valves_by_id` returns, before `emit_config_changed` and `publish_valves_changed`, so no 30-second lease is ever held across a socket emit or the event bus. Where Redis is not configured (`WEBSOCKET_MANAGER` is not `redis`, or the lease cannot be built) it is absent and one `logger.warning` says so: the save then behaves exactly as it always has, guarded by the in-process lock and the revision alone, because refusing a save on an install that cannot have this race would break it to fix nothing. A save that meets the lease already taken writes nothing and is refused with the module's own `_ClientMessage`, which the dispatcher answers as `500 {"error": "action failed", "detail": <sentence>}` and the tab toasts as `Save failed: nothing was saved -- <sentence>`; the sentence names the other administrator's save as the cause. That is the one new user-visible outcome here, and it replaces a silently lost edit; the revision cannot be the whole guard, because `Function.updated_at` is a whole-second timestamp (`models/functions.py:31`, written as `int(time.time())` at `:326`, `:345`, `:408`) and two saves inside one second publish the same one — so a save that carries `base` is checked per field under the same lock: each edited field's fresh stored value is compared with the value the client says it started from, an equal field is written and a differing one is left alone and named in `conflicts`, and a secret is compared on the opaque fingerprint of its stored value the Config tab received from `config_get` \u2014 a boolean base still compares set-ness, which is what an API caller and a tab drawn by an older worker send \u2014 and that fingerprint is a keyed HMAC of the value under the application secret, never of the plaintext and never of any part of the key, so nothing about the credential is recoverable from it. A save carrying no `base` keeps the whole-save conflict unchanged, and gets no per-field check at all when its revision matches the stored one — so a hand-written `config_set` in that state is guarded by the revision alone, and a caller that wants a same-second edit refused has to send `base`. The Config tab always sends it.. The whole-save conflict is the other shape a refusal takes, and it is not the same payload: every field in `edits` was contested, so nothing is written and the answer is the snapshot itself — `conflicts: [<names>]`, the `valves` specs, `drift`, `rev` and `config_unreadable`, and **no `secrets` block at all**. The two flags are in that snapshot under their own per-valve names instead, `valves[*].secret_set` and `valves[*].secret_stored`, so the tab reads both spellings from one place (`adoptSecretFlags` in the Config tab's `commitSave`) rather than reading `secrets` alone and guessing on the arm that has no `secrets`; the fingerprint rides along with them, because a tab that kept the one it loaded would conflict with the store on its very next save. A refused secret's flags are therefore the store's answer on both arms, never a value inferred from the edit, and the `base` the next save sends for that field is built from it. The partial answer is a normal save payload plus `conflict: true` and `conflicts: [<names>]`, so the tab adopts the saved fields as its new baseline and leaves the refused ones staged; `not_saved` names the edits the loop never interpreted — a name the schema does not carry, or a blank secret box -- whitespace included, which is the commoner accident of the two — and `saved` counts the rest. A value restored to its factory default, and a `..._TEMPLATE` box cleared so the built-in text goes back, are both saves: the row carries no name for either, because a value that now equals its default is not a customisation, and the row is what holds the result. A *clear* is the one edit the store legitimately does not hold: it is counted as saved, and it is collected in the same pass that honours it, so a `None` on a secret and a blank on a nullable number are both recognised as clears rather than as un-persisted edits. The `reset` list is what the tab names on the save that dropped them, beside the count in the note bar; the next `config_get` reads a repaired row, returns `reset: []`, and clears the note. When the stored configuration could not be read at all it returns `{"unreadable": <reason>, "rev"}` and writes nothing, and *not* a conflict payload: the conflict banner claims another administrator changed the configuration, and the pipe can name this cause, so the two would contradict each other. A refused **write** and a **rejected edit value** are a different fault and answer differently -- `the stored settings would not accept: <field names>` answers `400 {"error": "action failed", "detail": <sentence>}` when the stored schema rejects the edit, naming the fields and never the values (pydantic's own text would carry the rejected `input_value`, and `edits` legitimately carries `API_KEY`), and `the database refused the write, so nothing was saved` answers `500 {"error": "action failed", "detail": <sentence>}` when the store takes the merge and refuses the write. The split is carried by the raise site: a client-input refusal is a `_ClientInput`, which the dispatcher answers `400` and audits as `bad_args`, while a bare `_ClientMessage` keeps `500` and `error`, so a store fault or an exception the pipe did not choose is never reported as the caller's mistake. The two do not reach the operator the same way. The refused write is a store condition that recurs on every save until the store takes one again, so the tab raises its durable `#conflict` strip for it — the store-fault text, no control of any kind, the admin's edits still staged and the save gate released so the same edits can be written once the database is back — and it still toasts `Save failed: nothing was saved — <detail>`. The strip is cleared by the next successful save or by the next `config_get` (the tab's own snapshot load hides it), and by nothing else. No Reload control is mounted there on purpose: that control arms into "Discard changes?" and then reloads, which would destroy the very edits the refusal preserved. The rejected edit value is an administrator's own mistyped field, which the editor names and they fix by correcting it, so it toasts `Save failed: nothing was saved — <detail>` and banners nothing; the "nothing was saved" half of that sentence is the tab's. The conflict payload is still correct for the unreadable-*revision* cause, where no current state is known and the write is blocked anyway |
| `update_apply` | `write` + `admin` | `{"rev": int \| str, "compressed"?: bool}` | Applies the newest available release, snapshotting the installed function source once the loader has exec-validated the bundle and the function's revision token has not moved. The token `check()` publishes is `"<updated_at>:<content digest>"` -- the stored whole-second stamp and the first 16 hex of `sha256` over the row's `content` (`Function.updated_at` is `int(time.time())`, so the stamp half alone is a whole second wide and cannot see a hand-paste made inside the tab's own second). A bare int is still accepted, and is then guarded on the stamp half alone, so a tab drawn by a worker on the old code keeps working. `valves` is not in the token: `_commit` writes only `content`, so a settings-only save inside that second destroys nothing and is not a refusal. Refused with `403 {"error": "forbidden"}` to anyone who is not an `admin`, and with `{"error": "disabled"}` / `{"error": "valve_unreadable"}` in its body when `PIPE_DASHBOARD_UPDATE_ENABLE` is off or the stored valve set cannot be read |
| `update_restore` | `write` + `admin` | `{"file_id": str, "rev": int \| str}` | Restores a stored snapshot over the installed function source, against the same `<updated_at>:<content digest>` token and with the same bare-int fallback. Same two refusals as `update_apply`, plus `package_mode` -- a package/stub install is refused before the snapshot store is read, so the attempt spends no ring slot |
| `update_snapshot_delete` | `write` + `admin` | `{"file_id": str, "sha256": str}` | Deletes a stored snapshot. Same two refusals as `update_apply` |

With `PIPE_DASHBOARD_ENABLE` off, three paths refuse, and each is evaluated **for the install the request names**: the subscribe handler resolves the id its payload carried against this worker's own registrations before it reaches a gate, the valve event resolves its own `subject.id` the same way rather than against whichever install registered last, and switching one installed copy off closes that copy's feed and evicts that copy's viewers without touching the other's. The valve is read from the stored row at each of them, so a toggle takes effect on the very next request on every worker, with no restart -- and a row that cannot be read refuses rather than falling back to the in-memory copy, which is what lets a failed read override an operator's disable: the action route answers `404 {"error": "dashboard_off"}` -- its own code, not `unknown action`, so the tab can name the switch that is off instead of telling the administrator to reload a page that reloads the same way -- and audits the refusal as `outcome=disabled`; a name no version of this pipe registers still answers `unknown action`, which is the honest answer for a tab left open across a hot update and keeps its own reload sentence; a new socket subscribe is refused with a `logger.warning` naming the sid and never joins the viewers room; and every viewer already in that room is evicted -- immediately on the `function.valves_updated` event, and by the re-authorization that precedes every emit, whose answer it is handed rather than read again -- on a **committed** off only, never on a row that could not be read (see the access-model section above; that case refuses and evicts nobody, and is announced once as undeterminable) -- while `emit_dashboard` itself refuses, so the live feed stops for a tab that was open when the valve went off. That re-authorization is not deferred by a cache: it re-decides every local viewer from the database on every emit. Both dashboard write paths also announce themselves to Open WebUI's own event bus: a `config_set` that commits publishes `function.valves_updated`, and a committed apply or restore publishes `function.updated`, each carrying the pipe's function id as `subject.id`, the row's own `type` and `name` on the content event, and the acting administrator as `actor`. **The valve event now carries data where it carried none**: a `config_set` publishes `function.valves_updated` with `data={"pipe_config_change": "<pid>-<n>"}` -- the same change identity the socket announcement and the save response carry, which is how the pipe's own `_ValveEventSink` re-announces that save's echo under the identity the tab has already applied instead of minting a second one. An operator's event function or webhook sees this new key on that event; a write made anywhere else (Open WebUI's own Functions screen, an activation toggle) publishes the same event with no data, and the sink mints an identity for it. The key name is load-bearing: Open WebUI's `_sanitize` (`events.py:947-971`) drops any data key ending in `_token`, `_key` or `_secret`, so a differently named carrier would arrive stripped and the sink would mint a second identity -- so an event function or webhook subscribed to either sees the change Open WebUI's own route publishes for it. That is the event reaching the sink, and the sink then decides on the same persisted row every other gate reads, so the Config tab's own save evicts the panel it just locked out as immediately as Open WebUI's function editor does. A third event reaches the same sink and means something else: `function.deleted` for this pipe's own `subject.id` stands the generation down on this worker. It releases that pipe id's three per-pipe registry entries through the plugin's own identity-guarded teardown -- so a registration a newer generation has already taken is left alone, and another installed copy's entries are never in the blast radius -- and then calls the pipe's `close_when_idle()`, which drains rather than closes: a request already holding the pipe keeps its artifact store and log worker, and once its last in-flight call returns the close lands. Without that, a deleted row pinned one `Pipe`, its session-log threads, its storage handle and its Redis client for the life of the worker, because nothing else ever cleared those registrations on a delete. The sink is registered at import (`register_socket_handler` calls `register_valve_event_sink`), not by a valve, so the stand-down happens with `PIPE_DASHBOARD_ENABLE` off as well as on. The event is delivered **per worker**: Open WebUI's events module contains no Redis relay, so on a multi-worker deployment only the worker that served the DELETE reacts, and the others keep their pipe until their next hot reload -- exactly as Open WebUI keeps its own per-worker function cache. The gate's answer does not change with any of this: after the row is gone `stored_gate_valves` returns `({}, False)`, so the route and the socket gates keep refusing. The absent key is the off state: the Config writer drops any field equal to its declared default, and that default is `False`, so a stored row that omits `PIPE_DASHBOARD_ENABLE` is a row that says off. An unauthorized caller is still answered `403` first: the valve gate sits below the grant check and below the rate limiter, so turning the dashboard off never turns an authorization refusal into a not-found and never removes the 429 arm. A fourth thing the same valve switches off is the request path itself: with `PIPE_DASHBOARD_ENABLE` *and* `PIPE_DASHBOARD_USAGE_COLLECT` both off, `on_request` registers no session for a non-dashboard model at all and `on_emitter_wrap` returns `None` under the same `ENABLE or COLLECT` predicate, so the pipe holds no live-session state for a request nobody observes and builds no emitter wrapper for it either. A completed turn reads no persisted valve row either, so an install with the plugin off pays no query per generation and a tracked one pays none either: `on_generation_complete` does not read the row at all, and neither does a sweep tick, because the gate that decides whether a row is written is the writer's own per-batch read below. The second switch is part of the condition because `_persist_usage_row` is bound to `on_finalize`, and the rows it feeds are written under `PIPE_DASHBOARD_USAGE_COLLECT` alone — the Usage tab's rows are written from `finalize`, not from a viewer, so an install that collects but does not display keeps recording. That gate is not in `_persist_usage_row`: it is in `UsageStore._persist_sync`, below every caller, and it reads the persisted function row (see Usage records). A committed flip still takes hold on the next completed turn, because the writer re-reads the persisted row every batch. The abandon reaper starts under the same `ENABLE or COLLECT` predicate, for the same reason: it is what finalizes those rows, and it has to keep running on a worker nobody is watching. All four sites that evaluate that predicate — `on_request`, `on_emitter_wrap`, `on_request_alive` and the reaper's own `_maybe_start_sweep` — read it off the copy Open WebUI has just rebound into `pipe.valves` (`functions.py:52-68`, unconditionally before every call into the pipe), not off the stored row: the request path therefore pays no query per request, per emitter wrap or per heartbeat, and a valve committed after boot takes hold on the very next turn or model list with no restart, whether or not the dashboard model is on.

**The update service's error taxonomy.** `_reload_via_loader` in `update_service.py` snapshots the whole `sys.modules` mapping and `sys.meta_path` before it hands the content to Open WebUI's loader, and `_commit` wraps every exit after that loader in one `except BaseException` that restores that snapshot and re-raises -- so a raised error can never leave the pipe half-configured. The snapshot is still the whole mapping (that is what makes the compressed bundle's *deletions* recoverable), but the restore is scoped to what the attempt itself wrote: every snapshotted key in the attempt's own namespaces -- `function_<pipe_id>`, `<pipe_id>`, `<pipe_id>.*`, all read from the runtime id rather than from the module-level `FUNCTION_ID` constant, so a second installed copy under its own id restores its own keys and leaves the first copy's alone -- is put back by identity, a key the attempt created in those namespaces is removed, and a bundle-owned `sys.meta_path` finder is restored at the index it held before, so a generation swap that removed the previous generation's hook gets it back. The restore does **not** cover `app.state.FUNCTIONS[pipe_id]` or `app.state.FUNCTION_CONTENTS[pipe_id]`, and it does not need to: the two cache writes are the last statements in the try, after the `function.updated` publish, so any failure lands before they run and the caches still hold the pre-attempt generation while the row holds the committed content. Open WebUI reconciles that on its own next load -- `get_function_module_from_cache` compares the cached content against the row and reloads on a mismatch (`utils/plugin.py:399-401`) -- so the guarantee is stated by the ordering rather than left to luck. The published instance is the one Open WebUI's loader built, carrying whatever valves `Pipe.__init__` gave it: a successful commit deliberately does NOT rebind the stored row onto it, because Open WebUI binds the persisted valves itself on every retrieval (`open_webui/functions.py:52-68`) and every `.pipe` / `.pipes()` call site goes through it -- so on the manual path the cached instance is corrected on the way out, and on the automatic path, which writes no cache at all, the next read rebuilds from the row. Only the refused-write revival has to do that construction by hand (`functions.py:61`, null entries filtered), because the instance it builds is one nothing is going to re-read. Everything else is left exactly as a concurrent request left it, which is the same shape as Open WebUI's own narrower cleanup on a failed load (`utils/plugin.py:308-313`, `del sys.modules[module_name]`): the attempt cleans up after itself rather than resetting the worker. `BaseException`, not `Exception`, because a cancelled commit raises `CancelledError`, which is not an `Exception`; the refused-write arm restores before it revives the cached instance, so a `write_failed` leaves a serving pipe behind as well. Every `UpdateError` carries one of these codes, and the tab renders each differently, because they call for different operator responses:

| Code | Means | Transient (`_TRANSIENT_CODES`)? |
| --- | --- | --- |
| `storage_unavailable` | The store could not be reached or read -- a database outage, a snapshot store that is down, a row that will not decrypt. Nothing about the content is wrong. | yes |
| `row_unreadable` | The store would not return the pipe's own function row. Open WebUI's `get_function_by_id` answers `None` on any database error as well as on a deleted row, so a blip here says nothing about the content. | yes -- a database fault clears on its own, and retrying costs one read |
| `validation_failed` | The content itself is unacceptable: the frontmatter, the digest, the size cap, the Open WebUI version gate, or an HTTP fault from the download. Two refusals in this family are about VERSIONS rather than about the file: a release before `MIN_UPDATE_VERSION` (`2.7.0`, which predates the self-updater and would remove it), and a bundle whose declared version is not newer than the installed one -- `apply()` passes `require_newer=True` for BOTH actors, so a manual apply re-runs the identical guard and refuses identically, and a restart only re-arms into it. `require_newer` also lets an EQUAL version through, so a reinstall is not a downgrade. The auto path pauses that version on this worker for at most `_PD_UPDATE_AUTO_PAUSE_S`, and the tab names the refusal's own sentence beside the version it stopped on. | no |
| `write_failed` | The content passed every check and the function-row write was refused anyway. The freshly loaded code is rolled back and the previous version really does stay active. The revived instance's `Pipe.valves` is restored the way Open WebUI restores it (`functions.py`: the module's own `Valves`, built from the pre-attempt row with the null entries filtered out) -- and state derived inside `__init__` before that rebind is stale in both, so the rollback is not a full restore of anything `__init__` computed. That staleness has two arms, not one: the same is true of the instance `_commit` publishes on a successful apply, which keeps its `__init__` valves until Open WebUI's next retrieval rebinds the stored row (see the taxonomy paragraph above). The revived generation also re-registers the dashboard's per-pipe bindings -- one entry each in `http_routes._routes_get_pipes`, `dashboard_socket._pipe_getters` and `dashboard_publisher._pd_snapshot_getters`, all keyed by `Pipe.id` -- by running its own `_ensure_plugin_registry`, which is the same path its first request would take and gated on the same `ENABLE_PLUGIN_SYSTEM` the row carries -- so the panel follows the rebuilt instance, and registering through the plugin is also what makes those entries clearable at that generation's retirement rather than pinned for the life of the worker. The cache repair runs first, so a registration that raises costs the panel its state and not the next chat. | no -- `update_function_by_id` answers `None` on a deleted row and on a model-validation failure as well as on a storage fault, so the code cannot tell them apart. The pause it writes is bounded by `_PD_UPDATE_AUTO_PAUSE_S`: the store gets one attempt a day per leader worker, and a database blip never retires an upgrade for the life of the process |
| `exec_failed` | The new bundle did not survive Open WebUI's own loader. The module state is restored and the previous code keeps serving. The row's `is_active` is left exactly as the operator had it: a row that was on has the loader's own `is_active: False` repaired back to on, and a row the operator had already switched off is left off with no write at all, because the repair is skipped on that arm and the code stays `exec_failed` rather than `exec_failed_inactive`. So this code reports three states, not one: repaired-and-serving, admin-off-and-still-off, and a refused repair -- the last of which is the row below. | no |
| `exec_failed_inactive` | The bundle did not load AND the repair write was refused, so the function row is left switched off and Open WebUI lists no OpenRouter model for it until somebody switches it on in Workspace > Functions. | no -- a refused write means the store is unhealthy, and retrying it is exactly the retry to avoid |
| `deps_failed` | Open WebUI installs the release's frontmatter requirements BEFORE it binds or executes anything (`utils/plugin.py:272-275`), and that install shells out to pip and re-raises whatever pip raised (`:441-447`). A package index that is down, or a wheel that will not build, therefore arrives as a `CalledProcessError` from a step that never reached the `exec` whose failure writes `is_active: False` (`:307-313`). The module state is restored, the previous code keeps serving, and NO `is_active` write is made at all: the repair arm has nothing to repair, and writing the live value back would bump `updated_at` on a row nobody asked to change. The exception type alone does not decide this -- a bundle body that shells out to pip at import has necessarily been exec'd and stays `exec_failed` -- and the classification is never widened to `OSError`, because the pipe's own media stack shells out (`media/frame_extraction.py`) and those failures are not the release's. | yes -- the fault is this worker's package install or the network in front of it, and both clear on their own; a code that is not transient writes `_auto_skip` for the life of the worker, which is how one unreachable index retires a good release until a restart |
| `offline` / `rate_limited` | GitHub could not be reached, or is rate-limiting this server. | yes |
| `repo_not_found` / `bad_repo_valve` | The configured repository valve is wrong, or GitHub has no such repository or no releases. | yes -- the operator is expected to fix the valve; the tick backs off rather than pausing the version |
| `stale_rev` / `update_in_progress` | Another administrator's save, or another update, got there first, under one of four causes: the stored revision could not be read, the client's could not, the two differ, or the row's `content` no longer hashes to the digest half the client carried -- a same-second code write, which moves no revision at all and so reports a revision that did not move. The snapshot lands after the guard that raises this, so a refusal costs no rollback point. | yes |
| `digest_mismatch` | The release asset carries no `sha256:` digest, or the bytes that arrived do not hash to the one it published. A content problem, not a transport one: re-fetching the same asset fetches the same wrong bytes. | no -- and that is the whole point. `_http_get_bytes` caps by size rather than by the asset's declared size, so a truncated transfer and a corrupt release are indistinguishable here, and calling the code transient would put both on the backoff ladder for ever. The auto-updater writes `_auto_skip` for this version on this worker, bounded by `_PD_UPDATE_AUTO_PAUSE_S`, so a mismatch pauses the version for a day and the release is attempted again after that; a **manual** apply shows "Try again in a moment", which is true for a person who can fix the cause in between. |
| `not_found` | A snapshot row, or the file behind it, is gone. | no -- the thing is not coming back on its own; the operator has to re-take the snapshot |
| `stale_snapshot` | The snapshot changed after the list it was chosen from was loaded, so the delete carried a digest that no longer matches. The tab says to refresh and retry. | no -- it is a lost-update guard, not a fault, and a retry without a refresh would hit it again |
| `package_mode` | The installed row is a package/stub rather than a bundle, so there is nothing to apply or restore here: a package install updates through its pinned requirement, and every snapshot is a bundle, so restoring one would overwrite the pin with an older bundle. | no -- the valve or the install shape has to change first, and the auto-updater must not pause a version over it |
| `no_matching_asset` | The latest release carries no asset of the shape this installation wants (the compressed bundle, or the flat one). | no -- the release will not grow an asset mid-tick; the operator needs the other install mode, or a different release |
| `incompatible_owui` | The release's frontmatter asks for a newer Open WebUI than the one running. | no -- upgrading Open WebUI is the fix, and the version cannot arrive by retrying the asset |
| `internal` | The request the service was handed cannot be acted on: a manual apply or restore arrived with no request object. A caller-side defect, not a store or release one. | no |

The split is the point: a `storage_unavailable` and a `row_unreadable` are both retried on the next tick -- one is the snapshot store and one is the function row, and neither is a statement about the content -- a `validation_failed` pauses the version so a bad release is not re-fetched, and a `write_failed` pauses it too because the code is fine and the store is not. Every pause these write is bounded by `_PD_UPDATE_AUTO_PAUSE_S` and the expired entries are dropped on the same pass, so a parked release is reconsidered after a day with no manual action, no restart and no newer release, and the map cannot grow. `_TRANSIENT_CODES` is the single list the auto-updater consults, so a code added there is retried everywhere at once. A pause renders through the same per-code vocabulary as every other error -- `updErrText(this_worker)`, so the code's own sentence plus the refusal's own message in brackets -- and the pause record keeps that message in `_auto_skip[version]["message"]`, projects it into `check()["auto"]["this_worker"]["message"]` and logs it on the pause line, so the reason is on screen rather than only in the code. The `(apply manually, or wait out the day-long pause, or restart to re-arm)` hint is NOT part of that vocabulary: it is shown only for a pause reason one of those can clear, and withheld for `validation_failed`, where a manual apply re-runs the same guard on the same bytes and a restart re-arms into the identical refusal. The `Previous versions` list carries its own flag, `snapshot_storage_error`, and the tab branches on it BEFORE the empty-list test, so "storage down" is never drawn as "No snapshots yet."

`whoami` and `echo` are reference implementations; `usage_stats` powers the Usage tab; `config_get` and `config_set` power the Config tab (see the [Operations Guide](plugins_pipe_dashboard.md#editing-configuration)). A new action is registered by importing its module at plugin load -- the same explicit-import requirement as commands.

`config_get` returns a `drift` report with two lists: `unenriched` (valves with no `CONFIG_META` entry) and `orphaned` (entries with no valve). The Config tab renders `unenriched` only. So `orphaned` names settings this plugin declares and the Config tab cannot show, and nothing on screen says so; a `drift.orphaned` that is not empty is a signal to read, not a state any install should be left in.

**Why the stored config is read more than once, and what "unreadable" means.** `_read_stored_valves(pipe_id)` in `actions.py` returns `(stored, read_ok)` and separates the transport read from its verdict, so the raising case and the `None` case reach the same marker. `None` is unreadable: Open WebUI's `get_function_valves_by_id` catches its own DB errors and returns `None`. `{}` is NOT reliably "nothing stored" -- `decrypt_valves` returns `{}` on `InvalidToken`, i.e. on a failed decrypt, which is what a rotated `WEBUI_SECRET_KEY` produces. `stored_row_readable(pipe_id, stored)` in `config_service.py` tells those two apart by reading the raw `Function.valves` column, and `update_service._row_valves_checked` asks the same predicate -- `storage.persistence.raw_valve_column_decodes` -- rather than re-deriving the key and decrypting the raw string itself, so the probe reads the SHAPE before the cipher and the two readers cannot disagree on any column. Ciphertext present but nothing decoded is unreadable; so is a column whose shape is a token and whose cipher will not open under this host's key. Everything else is readable, and that list is longer than "ciphertext or empty": a plain JSON object, an array, an empty-string column and any text that is not token-shaped at all are readable, because the update surface has no key evidence for them. A column that is a token with a mangled version head is readable for the same reason -- the shape does not claim to be a ciphertext this server wrote, and refusing on the strength of a guess about a key it never applied would deny the update surface on rows nobody has shown to be undecodable. Only a *positively observed* ciphertext downgrades the read -- an Open WebUI without the async db reader, or a transient failure on it, is not evidence of corruption, and treating it as such would mark every such deployment unreadable. `_effective_valves_and_state(pipe)` rebuilds the valve model from that subset and returns `(valves, dropped, stored, read_ok)`, keeping the stored subset separate from the reconstructed one; `_effective_valves_and_drops(pipe)` is the `(valves, dropped)` view of it that the config tab's own callers use, and `_effective_valves(pipe)` the valves-only view. The two name one key from two directions, which is why the same stored name can appear in a note bar and in the pipe log for one save: `readable_stored` reports what the reconstruction could not keep, and `valve_salvage.drop_unvalidatable` reports what it could not keep from the row itself -- a value the schema rejects, and separately a name the schema does not publish, whose stored value it never quotes because there is no annotation left to decide that it is a secret. `config_set` refuses to write when the read is not ok, from either the conflict arm or the pre-write gate, and both paths carry `config_unreadable` so the client renders the banner instead of a Reload button that would re-read the same undecodable blob. A read that returned `None` keeps its own `unreadable` refusal payload on the save path, so the admin is not told a concurrent editor caused it. Both readers of a persisted valve row -- the update surface's `update_service._row_valves_checked` and the dashboard gates' `config_service.stored_gate_valves` -- resolve a key the stored row omits to that field's declared default, and neither consults the worker's own copy for it, so a setting the administrator reset to default reads as the default on every worker at once instead of as whatever a stale worker happens to be holding. A key the valve class does not declare is left out of the result entirely, and the call site's own absent-field default stands: a pipe that declares neither gate is not the dashboard and must keep working. A committed save reads the row a third time, in `_saved_values`, and that read belongs to the echo alone: the two reads above gate the write, and by the time the echo runs the write is already committed. It is the one step of the three that reads the row for reporting rather than for the decision, so it is also the one that has to be read under the same exclusion as the write -- the tab adopts the echo as its new baseline, and a read-back taken after the lock is released can report a second administrator's write as this save's own. So when the third read is not ok the echo is dropped -- `values` empty, no `secrets` block, `post_reset` empty -- rather than rebuilt from `pipe.valves`, which is the value the store has just contradicted, or from `Valves()`, which is a set of defaults for a row nobody could read. The pipe log names the arm: the warnings above describe the panel's view, this one describes an echo that could not be made after a commit.

**Why the Clear control is gated on the stored subset, not on `secret_set`.** (Amends the original design, which said the button is "rendered only when `v.secret_set` is true".) The design is overridden: a control must be able to remove what it promises. `secret_set` is derived from the *reconstructed* valve set, which includes the `OPENROUTER_API_KEY` default, so on the common env-only deployment it is true for a key the pipe never persisted -- a Clear button would render, and clicking it would stage a clear that saves "1 setting" and leaves the store untouched. The button is therefore gated on `v.secret_set && v.secret_stored`, where `secret_stored` is put on the spec by `_config_snapshot(valves, stored)` from the stored subset. `secret_set` keeps its meaning (an env-supplied key genuinely *is* configured, and the placeholder, the hint and the Default cell key off it): a `stored`-driven control and a `secret_set`-driven placeholder are different questions and do not share one flag. The write-gate refusal site has read nothing, so it passes no `stored`, `secret_stored` is False there and no Clear button renders -- correct, because the tab is being told the store is unreadable and Save is off anyway. `commitSave` used to derive both flags from the edit it sent, and the store does not have to honour that prediction: it overwrites both from the response's `secrets` block now, so the Clear control and the hint follow the row that was written. `secret_stored` is the half nothing client-side can predict at all — a key typed for the first time has no stored value until the server says so — which is why a first-typed secret was not clearable until something forced a reload. The tab applies the `secret_set`/`secret_stored` the `config_set` reply carries, and the env-only case is resolved there, because the server is the only party that knows whether a value came from the environment and the request says nothing about it. That local projection survives only as the fallback for a server that returns no `secrets` block at all, and follows the same rule: it clears `secret_set` only when a clear actually removed something stored, so an env-only secret does not flip to "not set" locally and back to "configured" on reload.

**Why the save path keeps a secret only when its plaintext is non-empty and differs from the default.** `_is_clear_edit(fld, value, current)` is the explicit clear: a `None` edit for a secret. `None` and `""` are different intents and collapsing them is the defect. Empty **or only whitespace** is "the box is blank", which is what a stray keystroke -- a space, and backspace over it -- looks like, so both spellings keep meaning unchanged; `None` alone is the clear. `None` is a distinct wire value the client sends only from its Clear control, so it means "remove what is stored" whether or not anything is stored under that name -- both outcomes are the same stored state. A `None` edit pops the key inside `merge_for_save_with_drops` (the two-value form `merge_for_save` wraps), but `full = valves_cls(**merged).model_dump()` is a dump of a *fully-defaulted* model and always contains every field, so the key is not absent from `full`; it is dropped by the subset filter, which keeps a secret only when `plain` (its decrypted value) is truthy AND differs from the default. A clear therefore lands on the stored subset as "the key is not stored", which is decision 2: the setting returns to its default.

**Why `_tool_counts` returns three and so does `db_row`.** `_tool_counts(entry)` yields the three tool-outcome counters (`tools_ok`, `tools_failed`, `tools_skipped`) as ints and is used by the live row. `tools_skipped` is a breaker-refused or no-longer-awaited call, not a failure, so it is never folded into `tools_failed`. `db_row` persists all three, because `USAGE_ROW_FIELDS` and the ORM model both carry the third column and `ensure()` reconciles an older table up to the model before the first write. What keeps the projection and the writer in step is `test_db_row_mapping`'s `set(row) == set(USAGE_ROW_FIELDS)`: `db_row` is the only writer of a persisted row, so that one assertion is where a column added to the projection and forgotten in the writer is caught. Such a name also raises, and it raises in a way the batch cannot survive: `UsageStore._persist_sync` calls `model(**self._fit_row(data))`, so SQLAlchemy's declarative constructor raises `TypeError` on an unknown keyword -- while the batch's instances are being built, before the per-row `session.begin_nested()`, so it aborts the whole batch. `_write_now` catches it and logs "usage batch persist failed; N row(s) held for retry, not written" at a warn-gated level, and the rows are held rather than written.

---

## Authorization Helpers

`authz.py` is the single authorization chokepoint. It composes no access logic of its own -- every decision delegates to Open WebUI's own model access control, so the dashboard inherits the grants an admin sets in the model's Access editor. Each helper delegates to the Open WebUI answer for its own question, and the two questions have different answers: `can_view` mirrors the read decision (`routers/models.py:722-734`, which carries `BYPASS_ADMIN_ACCESS_CONTROL`), `can_act` mirrors the write disjunction (`:947-957`, which does not). The write decision therefore does not depend on the model row existing -- the admin term is answered on the role alone, ahead of the `model is None` clause -- while a row read that **raises** stays undeterminable rather than becoming a verdict. Reuse these helpers for **model-access** decisions instead of writing your own role checks against OWUI's tables:

| Function | Signature | Semantics |
|----------|-----------|-----------|
| `model_id(pipe)` | `(pipe) -> str \| None` | The OWUI model id for this overlay, `"{pipe.id}.pipe-dashboard"` (or `None` when the pipe has no id) |
| `resolve_user(user_id)` | `async (str \| None) -> UserModel \| None` | Load the OWUI `UserModel` by id; `None` on any failure |
| `can_view(user, pipe)` | `async (user, pipe) -> bool` | **Read** grant -- `check_model_access` (honors owner, admin, direct/group grant, `user:*` public, `BYPASS_MODEL_ACCESS_CONTROL`) |
| `can_act(user, pipe)` | `async (user, pipe) -> bool` | **Write** grant -- admin, owner, or a `write` access grant. The admin term is decided without reading the model row (`routers/models.py:947-957`); a row that cannot be read at all answers undeterminable. That admin term carries **no** `BYPASS_ADMIN_ACCESS_CONTROL` conjunct, unlike the `write_access` the model editor reports (`routers/models.py:725-733`) and unlike `can_view` above |

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

Actions are invoked over one authenticated route registered on Open WebUI's own FastAPI app (ahead of the SPA catch-all), not over the Socket.IO channel. Registration is idempotent: the route is added once and never removed, so the path is absent from `app.routes` for no instant and a request arriving during any number of later registrations still matches. Each call re-checks the live `app.routes` list under the registration lock rather than trusting the module-global `_registered_paths` set, which says "registered" even after a fresh app object replaces the old one. The socket module's `_registered` and `_resync` are process-wide singletons of the same kind, and they stay that way: `_registered` guards the one `sio.on(SUB_EVENT, ...)` this process installs, and `_resync` is a tick signal that carries no viewer state and no data (a viewer joining one install's room forces the next payload to be a full one, which under rule 3 is not worth a keyed container). The three bindings the pipe writes -- `dashboard_socket._pipe_getters`, `http_routes._routes_get_pipes` and `dashboard_publisher._pd_snapshot_getters` -- are **not** singletons: each is a dict keyed by `Pipe.id`, so they outlive any single test and the suite empties them between tests rather than asking the pipe to forget them.

| Property | Value |
|----------|-------|
| Method + path | `POST /api/pipe/dashboard/action` |
| Auth | Header-only `Authorization: Bearer <token>`, decided by Open WebUI: two route-level dependencies bound at registration, a bare-bearer precondition and `get_verified_user`. The precondition depends on OWUI's own `bearer_security` -- `HTTPBearer(auto_error=False)` (`utils/auth.py:175`) -- so the scheme is partitioned and case-folded and the credentials stripped exactly as OWUI does: any casing of `bearer` and any surrounding padding are accepted while the credentials themselves are handed on byte-for-byte (they are case-sensitive base64url, so they are never case-folded), and an empty credential is `None` rather than a pass. Behind it OWUI's `get_current_user` + `get_verified_user` supply the token decode, `is_valid_token`, a fresh `Users.get_user_by_id`, the `WEBUI_AUTH_TRUSTED_EMAIL_HEADER` comparison (set + present + disagreeing with `user.email` is a `401`) and the last-active refresh; the endpoint reads the principal from `request.state.user`. The session cookie and `request.state.token` are never read, so the route is CSRF-safe regardless of OWUI's CORS/SameSite settings. A dependency's `Response` is not usable here: the endpoint returns a `JSONResponse`, and FastAPI discards cookies and headers set on a dependency's response parameter on both the success and the error path -- so OWUI's `delete_cookie` in `get_current_user`'s failure tail (`utils/auth.py:492-504`) is inert on this route |
| Request body | `{"action": <str>, "args": <object>, "pipe": <str>}` -- `pipe` is the `Pipe.id` of the install the panel belongs to, and it is how the route decides which one to serve on a worker that hosts several. It defaults to `""`, and an empty or unresolvable id is answered `404 {"error": "unknown pipe"}` before the plugin-system gate, the rate limiter and any ACL read |
| Request body size | 1 MiB (`_PD_MAX_BODY_BYTES`). A body over it is refused `413` (`{"detail": "args too large"}`) from the `Content-Length` header when it admits the size, and again from the bytes actually read when the header understates or omits it, so neither a lying header nor chunked encoding gets past the cap. The cap is checked after authentication and before the depth scan, so an unauthenticated caller is answered `401` without the route reading or scanning a byte, and an over-cap body is refused without the scan |
| Response | The `dispatch_action` envelope, at its status code |

The route adds six guards in front of the dispatcher: a `413` (`{"detail": "args too large"}`) for a body over `_PD_MAX_BODY_BYTES` (1 MiB), taken from the `Content-Length` header when that header admits the size and from the bytes actually read when it understates or omits it, so neither a lying header nor a chunked upload gets past the cap, and the depth scan never runs over it; a bare-bearer precondition, which refuses a request carrying no usable `Authorization` credential before any identity work happens -- an absent header, or a `Bearer ` whose credential part is empty, which OWUI's own `HTTPBearer(auto_error=False)` reports as `None` rather than as a pass, and which is why a cross-origin page cannot fall through to OWUI's session cookie or `request.state.token`; then OWUI's own `get_current_user` + `get_verified_user`, whose `401` covers a token that is absent, malformed -- which after the parser above covers only a non-`bearer` scheme, since padded and differently-cased spellings are accepted -- invalid, maps to a role outside `{user, admin}`, or names a person the trusted-identity header disagrees with (`WEBUI_AUTH_TRUSTED_EMAIL_HEADER` set, the header present with a value, and that value not equal to the resolved user's email, compared exactly as OWUI's `get_current_user` compares it -- the header lowercased, the stored email as it is, and no requirement that the header be present). A header that is absent, or present and empty, is not a mismatch: the comparison is skipped exactly as OWUI's own `get_current_user` skips it, so a proxy that stamps the header on `/` but not on this route does not lose its admin surface -- including the update path that would fix it -- authentication is bound into the route as a dependency list declared ahead of the body guard, so an unauthenticated request is refused without the route reading or scanning a byte of it, and the effort of a call the caller is not authorised for does not depend on how many bytes they sent; a `404` (`{"error": "unknown pipe"}`) when the body names no install, or one this worker holds no registration for, answered before the plugin-system gate, the coarse rate limit and any ACL read, so a refusal cannot be an authorisation oracle for an install this worker does not serve; a `404` (`{"error": "plugin_system_off"}` — the same envelope shape the route's own unknown-action answer uses, so the panel's `error` branch can tell "the plugin system is off" from "the stored settings could not be read"; an absent route answers a different body entirely, `{"detail": "Not Found"}`, and this one is indistinguishable from it only to a client that checks the status code) when the master switch `ENABLE_PLUGIN_SYSTEM` is off on the resolved pipe, answered from the stored row on every request and refusing when that row cannot be read rather than serving the in-memory copy (a dashboard switched off while the plugin system is on passes this guard and is refused by the dispatcher from the same stored row, with its own `{"error": "dashboard_off"}` 404 and `outcome=disabled`), and that pipe is the one the request named — resolved by that id against `sys.modules` on each call, so a `Pipe` retired by an in-place update cannot leave the switch reading a frozen copy and a second installed copy cannot take over the first's route — the route is registered once and never unregistered, so a worker that started with the switch on would otherwise keep serving the whole admin surface; and a coarse per-user `429` (one request per 0.25 s) ahead of the per-action rate limit, which the 404 is answered above and so never spends a slot. A pipe the getter cannot resolve, or one carrying no `.valves`, reads the switch as off and is refused on the same terms. **The same branch refreshes last-active.** Once the caller is admitted, OWUI's own `get_current_user` fires a detached `asyncio.create_task(Users.update_last_active_by_id(user.id))` (`utils/auth.py:483`) and does not retrieve its exception, exactly as it does on every route it has, so an operator reading that field to spot idle admins no longer sees dashboard-only admins as inactive; the task is detached and runs after the route has answered, so a refresh that raised changes neither the answer nor the response. Its exception is consequently unconsumed here as it is there, which is OWUI's behaviour rather than a regression in this one route. Past the guards the route never raises: when neither the live `actions` module nor the reconcile yields a dispatcher -- a partially-initialised module in `sys.modules`, or one the hot update deleted -- `_preferred_dispatch` hands the call to `_dispatch_unavailable`, which takes `dispatch_action`'s signature verbatim and answers `503 {"error": "action unavailable"}` after writing one audit record with `outcome=unavailable` and its write args redacted exactly as on every other outcome (with no action entry in hand, so every registered schema is vouched at once -- the widest set, and the only one that cannot be stale on the arm where the registry is what failed to resolve). `outcome=undeterminable` is the other 5xx this audit can carry, and the two must not be read as one thing: `unavailable` means this worker has no dispatcher to ask, while `undeterminable` means the dispatcher asked and the ACL read failed under it, so the answer is `500 {"error": "access_undeterminable"}` -- the status Open WebUI's own admin routes answer for the same fault -- rather than the 503 above, and its own code because the remedies differ: a retry, not a reload. That is the envelope every other refusal on this route uses, so the panel's `callAction` renders it, and 503 is retryable by the poll. The responder lives in `http_routes.py`, never in `actions.py`: the live lookup resolves `dispatch_action` from the live module, so a responder parked there is exactly what that lookup could hand back. The 404 sits **above** the per-action grant check, because the master switch is not a per-action grant: nobody is asked for permission, and the audit records `plugin_system_off` rather than `forbidden`. The plugin's own surfaces carry the same persisted read: `PipeDashboardPlugin.on_models` and `.on_request` import `_plugins_enabled` from this module — one expression of "is the plugin system on", shared with the route, so the two cannot drift — and the tracker and usage writer take the same key from the merged row they already read. So a switch committed off closes the model list, the dashboard chat, the session tracker and the usage writer as well, on any worker, whatever its in-memory copy says. The dashboard calls it from `callAction(...)` in the shell, forwarding the same `localStorage` token it uses for the socket.

**Post-update reconcile.** `http_routes.py` imports `ACTIONS` and `dispatch_action` **by value** at module import, so a pipe updated in place leaves this route holding the *old* copy: a newly registered action 404s even though the on-disk code defines it. When the requested name is not in the held registry, the route therefore tries to re-resolve the action module once (`_resolve_fresh` → `get_function_module_by_id`) and, on success, keeps the fresh `(dispatch, pipe)` pair for the rest of the process. The re-resolve requires a read grant, is serialised by `_reconcile_lock` with a double-check, and is retried at most once every `_PD_RECONCILE_BACKOFF_S` (5.0 s, on `time.monotonic()`, so a wall-clock step cannot break it) — a reconcile that returns nothing therefore does not latch, and an update that lands while the fresh module is temporarily unresolvable is repaired by a later request. The lock is re-created whenever the running event loop is not the one it was bound to — the same guard `pipe.py` keeps on `_queue_worker_lock` (and on `_log_worker_lock`) and the fourth site `actions._config_write_lock` keeps on the per-pipe admin-save lock — because an `asyncio.Lock` binds itself to the first loop that makes it wait and refuses to be taken by any other, so a lock a closed loop left behind would otherwise stop the repair instead of serialising it. That cache also prunes: an entry whose lock is bound to a **closed** loop is deleted on the next call, because a contended acquire holds a strong reference to its loop, so an unpruned entry pins that loop — and everything it referenced — for the life of the process. A lock that is merely *unbound* is kept, never pruned: two `config_set` coroutines that must serialise can both arrive before anything has contended the lock, and handing them two different locks would lose an edit. Success is terminal: once a fresh pair is installed, the reconcile block is short-circuited forever -- *until* a teardown takes the pair away, which is what the epoch counts, or the pair names a pipe the route is no longer serving, which the use-time check in `_preferred_dispatch` notices and drops. A `set_pipe_getter` that installs a different getter takes the pair with it as well, since a new getter is exactly the previous-generation case and needs no teardown to arrive. The comparison is **per id**: `set_pipe_getter` returns early when the getter already registered under that id is the same object, so re-registering the same install on every `/api/models` moves nothing, while a new pipe id or a genuinely different getter for the same id still bumps it. That per-key guard is also what keeps two installed copies alternating on one model-list request from reading as a teardown on every hover of the model picker. A `clear_fresh_dispatch`, or a `set_pipe_getter` that installs a different getter, bumps it, and a re-resolve that overlaps either discards its result and arms the same 5 s backoff instead of installing a dispatch for a pipe that is gone, so the two ways a reconcile can publish nothing are now the same arm. The reconcile's own state is read and written the same way: `_fresh_dispatch`, `_teardown_epoch`, `_reconcile_retry_until` and `_routes_get_pipes` are resolved through `sys.modules` on every access by `_live_reconcile_state()`, not as bare module-level globals, so a teardown or a re-registration from **any** generation reaches them. That is why the read goes through `sys.modules` rather than by name even for state the module itself defines: the endpoint's own `__globals__` belong to the first generation, which after an in-place update is a superseded namespace no running plugin mutates. Once-only registration is still sufficient, but not for the reason it used to be given. The endpoint in `app.routes` is the *first* generation's `_action_route` function object, and after an in-place update its `__globals__` belong to a superseded namespace rather than the one the process is running — so "a module-level function name that reads the module globals at call time" is not a safety property here, it is the bug. `_action_route` therefore resolves the dispatcher, the action registry **and the serving pipe** by name from `sys.modules` on every call; the reconcile above is one of those per-call reads, not the mechanism that makes idempotence sufficient, and what is left of its job is the window between a reload and that reload's first `pipes()`. A route registered by a version that predates the live pipe read is replaced once, by the first version that has it, so a worker that has not restarted picks the fix up rather than serving a retired `Pipe` for the life of the process; a fixed route is never replaced, so a later update — including a rollback to an older version — cannot replace the code of the one route an operator needs to **roll that update back**. What does *not* follow a reload is the endpoint's own glue: the bare-bearer precondition and OWUI's `get_verified_user` (bound into the dependant at `add_api_route` time, as a dependency list declared ahead of the endpoint's own guards), the JSON depth guard and the 1 MiB body cap, and the coarse per-user limiter with its per-generation state all stay at the version that registered the route, so a change to any of them takes effect at the next worker restart. The current request still answers `404` when the re-resolve fails; the retry happens on the *next* one, and the route never raises for this. The refused-write arm reaches the same epoch through the plugin rather than around it: after a `write_failed` the revived generation re-registers the pipe getter and the snapshot getter by building its own registry, so the route resolves the instance the cache now serves, and its teardown clears them by the same identity test every other generation uses -- which is what keeps a retiring generation's `on_shutdown` from dropping a newer registration.

**Two installed copies.** Open WebUI never merges two installs of one function: `get_function_models` namespaces every model as `<pipe id>.<model id>` and `get_pipe_id` splits on the first `.` to route a request back to exactly the function that published it, and `authz.model_id` composes the dashboard's row the same way. Every dashboard binding follows that key rather than a single module-level value -- the three per-pipe registries, the socket room (`viewers_room(pipe_id)`), the subscribe payload's `{"pipe": ...}` and the action route's `pipe` body field -- so a subscribe, an emit, a Config-tab save and a valve event each resolve against the install that asked for it, and two copies each see two grants rather than one shared one. The re-authorization sweep reads the model row of the install whose room it is walking, and the action route refuses an id this worker has no registration for with `404 {"error": "unknown pipe"}` **before** the plugin-system gate, the rate limit and any ACL read, so that refusal is not an authorisation oracle for an install this worker does not serve. Each worker keys its own registry, so on a multi-worker deployment worker 1 may serve one copy while worker 2 serves the other for the same request: that is correct per-copy behaviour and matches how Open WebUI's per-worker function cache already behaves, and what changes for a viewer is that the panel is consistent for its own copy instead of flapping. One consequence reaches an existing user: a dashboard panel persisted in a chat message before this change carries no `pipe`, so its Config and Update tabs now answer `unknown pipe` until the panel is re-rendered -- the shell's `unknown_pipe` reason says so in words rather than showing a raw refusal.

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

**The abandon sweep.** `SessionTracker.sweep()` runs periodically and finalizes any active request that has produced no liveness signal for two hours (`_ST_ABANDON_S`), as `failed`. It has two callers: a plugin-owned timer task (`_sweep_loop`, every `_ST_SWEEP_INTERVAL` = 300 s plus up to `_ST_SWEEP_JITTER_S` = 60 s of per-worker jitter, started from `_re_register_registrations` at `on_init`, from `on_request` whenever that request starts tracking a session, and from the model-list pass (`on_models`) *above* its dashboard-off return, so the sweep is armed on the shipped default where only collection is on, each reading `PIPE_DASHBOARD_ENABLE` or `PIPE_DASHBOARD_USAGE_COLLECT` off the copy Open WebUI has just rebound, and never on an instance whose `Pipe` is already `_closed`, since that instance's `on_shutdown` has run and nothing would cancel a task re-armed afterwards) and `_live_snapshot`, which any worker that is emitting or publishing its slice reaches every 2 s — that is a local viewer, or, with Redis, another worker's active flag. The residual is that a worker which never receives a model-list request after the switch does not arm. The jitter is what keeps the workers on one install from reading the valve row in lockstep every five minutes; it only ever delays a tick and never advances one, so no request is abandoned measurably early against the two-hour threshold. The timer is what makes the sweep independent of anybody watching: an install with no dashboard open still reaps its abandoned requests, which is the case the Usage tab's Failed count is otherwise short on. With both valves off the tracker is not populated at all, so there is nothing for it to sweep and no task is created. Liveness is the `seen` stamp, which `mark_streaming`, `tool_started`, `tool_result`, `retry` and `update_usage` all refresh, and which the per-request heartbeat's `on_request_alive` refreshes on the request loop's own cadence, and which `mark_stream_alive` refreshes on each of the liveness-carrying event types the wrapped emitter sees — `chat:message:delta` and `response.output_text.delta` for a plain streamed turn, and `fusion:event` / `fusion_inner:reasoning.delta` for an internal Fusion turn, which emits no text delta of its own because `streaming_core` consumes the upstream one and republishes the synthesis as `fusion:event`; `started` is the fallback for an entry that predates the stamp. **`seen` is not published**: it is the tracker's internal sweep clock, held on the in-memory entry only. `started` remains the elapsed / `duration_ms` clock. A row is finalized once and never re-finalized. The per-chunk refresh is coalesced: it writes at most once per `_ST_STREAM_STAMP_COALESCE_S` (5 s) per session, gated on a monotonic comparison made before the lock, so the hottest path in the pipe costs one float comparison per chunk rather than one lock-guarded dict write. That holds for a *tracked* session: an id the tracker does not hold costs one dict-membership test in `mark_stream_alive` and nothing else, and with both valves off the wrap is not installed at all, so the chunk never reaches the tracker. So a request that truly never returns still produces a `failed` row — while the persisted collect valve is on, since the write gate reads that row on the writer thread for every path alike — and a request that is still running **and producing liveness signals** — a streaming generation, in either of the shapes above — when the sweep passes is finalized by whoever really ends it, so its recorded status, duration and token counts are the real ones, in exactly one row: staleness is re-tested at the moment of removal, in the same critical section as the removal itself, so a liveness signal landing between the sweep's snapshot and the pop saves the request rather than being discarded. A stretch that only *reasons* is not one of those shapes: `streaming_core` consumes reasoning deltas and does not republish them, and the only thing that surfaces is a throttled `status`, so a turn that reasons for hours and then produces no text (no `output_text` delta, no tool) emits nothing the stamp covers and is abandoned as `failed` even though the model is working. The heartbeat below covers that subpopulation too, and it needs no event type on the stamp: it is driven by the request loop, not by an event. The `mark_streaming`, `tool_started` and `update_usage` stamps are gated on `on_emitter_wrap`, which `pipe.py` only builds when `stream_queue is not None` (`wants_stream`); the `tool_result` and `retry` stamps are not. `on_tool_result` is dispatched from `_hand_back_tool_result` in `pipe.py`, from `_append_and_notify` and the two give-up sites (`_dispatch_settled`) in `tools/tool_executor.py`, and `on_request_retry` from the shared retry loop in `requests/orchestrator.py`, so all of them reach a request that was sent with `"stream": false` — its `seen` moves past its `started` and the sweep leaves it alone, with its real cost and token counts intact. A heartbeat covers the subpopulation the stamps do not: one that never streams, never runs a tool and is never retried. Once per `_PD_LIVENESS_INTERVAL_S` (60 s, a module constant in `pipe.py`, not a valve) every request the pipe's own request loop is still executing dispatches `on_request_alive` for its `job.request_id`, which reaches the same `mark_stream_alive` the streamed path uses, so its `seen` moves on the non-streaming path too and the sweep leaves it alone with its real cost and token counts intact. It is per-request-loop rather than per-event, which is what closes the reasoning-only stretch as well: the events a reasoning-only turn emits are consumed inside `streaming_core` and never reach the wrapped emitter, so no additional event type on the stamp would cover them. A request the pipe is genuinely no longer running, and which has also stopped emitting, is abandoned exactly as before.

**Live session rows.** `SessionTracker.live_sessions()` returns the rows the Live tab renders and the publisher ships as `sessions_live`. The rows are a **capped view of a larger population**, and the caps are what the tile numbers have to be read against: `_ST_ACTIVE_CAP` keeps the **30 newest** in-flight rows per worker, `_ST_RECENT_CAP` keeps 300 finished rows per worker, and `_PD_SESSIONS_CAP` keeps 300 rows across the cluster after aggregation, dropping finished rows first. Both per-worker caps bound **rendered chat rows**, not tracked entries: `_live_sessions_locked` filters the task entries out of both populations *before* it slices, so a task entry can no longer consume a slot a chat row would have rendered in. Those two are display bounds and are not the only ones: `_ST_ACTIVE_HARD_CAP` (8 192) is a separate bound on **tracked entries** in `_active`, four times the pipe's own `MAX_CONCURRENT_REQUESTS` ceiling of 2 000, and it is what actually stops the two per-tick walks from growing without limit. It is never merged with `_ST_ACTIVE_CAP` — that one says how many rows the Live tab renders, this one says how many sessions a worker will hold. When `start()` finds the map at the hard cap it removes one entry: the one whose `seen` stamp is oldest, which is the same judgement `_is_stale` makes and the reason a turn still streaming is never the victim (the cap exists to bound what *leaked* entries hold, so the leak is what goes). The evicted entry is treated exactly as the abandon sweep treats one — recorded as `failed`, `done` stamped, carrying that entry's own cost and token counts, folded into `_recent`, and fired through `on_finalize` outside the lock, so it produces exactly one usage row — and its own later `finalize` finds nothing to pop and writes no second. One latched warning names the bound and the count, because reaching it means the population was unbounded.
The two populations are capped **separately** — `_trim_recent_locked` keeps the newest 300 chats and the newest 300 task entries, so the recent ring holds up to 600 entries rather than 300, and a finished chat is never evicted by a newer task entry. The task entries stay in the ring on purpose: `_fold_task_into_parent` reads it to fold a task's cost into its parent chat's row -- onto the same-`(chat_id, user_id)` row with the greatest `started` **at or before the task's own `started`**, so an overlapping second turn (one that started after the task did) is never the parent and a turn that was already running when the task started is; when no candidate is at or before it, the fold takes the **earliest** same-key candidate, because `_task_costs_locked` builds its `folded_here` set from every chat entry carrying the chat id rather than from the rows the fold touched, so a task that folded onto nothing would be skipped as already-folded and its cost would leave both `live_sessions()` and `task_costs_by_chat()` entirely -- and `_task_costs_locked` reads it to publish (`tc`) the cost of a finished task whose chat row is not on this worker -- the cross-worker arm picks the latest-started row per `(chat_id, user_id)` from a map that carries no task timestamp, so it cannot honour the time rule above and is deliberately left as it is; so the ring is a storage bound as well as a display one — the cap is why the `Done` tile cannot read more than 300, and a count shipped beside the rows could never read more than 300. `live_snapshot()` returns a third value, `active_total`: the number of tracked, non-task, not-yet-finalized sessions on that worker, computed **before** `_ST_ACTIVE_CAP` and through the same predicate the row loop uses, so the `Active` tile describes the population and the table the capped view of it. The client-side **Keep completed** filter only ever drops rows carrying a `done` stamp, so it cannot change that count. `Done`, `Cost` and `Tokens` remain totals over the rows shown, and above the cap they therefore exclude the oldest in-flight requests. Each row:

| Field | Type | Notes |
|-------|------|-------|
| `user` | str | User name (or email, or `?`) |
| `chat_id` | str | The chat the request belongs to, used to add a background task's cost to the same user's row in that chat; removed before the rows reach the browser. In a worker's Redis slice a temporary chat carries an anonymous stand-in, or nothing without `WEBUI_SECRET_KEY` (see Multi-Worker Aggregation) |
| `user_id` | str | The user the request belongs to, the other half of the fold's identity (the other half of which is time: the fold also picks the latest turn **at or before** the task's own start, falling back to the earliest same-key candidate when every turn started later), and present on the row for that match only; removed before the rows reach the browser, exactly as `chat_id` is. Without it a background task's cost is added to whichever member of a shared chat id is on the row |
| `model_id` | str | Raw model id |
| `model_name` | str | Server-resolved display name |
| `kind` | str | `chat` or `task` |
| `status` | str | `queued` / `streaming` / `tool:<name>` / `completed` / `failed` / `cancelled`. The three in-flight values are Live-only. `completed` is the session archive's `complete`, `failed` is its `error`, and `cancelled` is spelled the same in both; see [Session log storage](session_log_storage.md) |
| `started`, `done` | float / null | Unix timestamps; `done` is `null` while in flight |
| `elapsed_s` | float | Seconds since `started`, frozen at `done` |
| `tokens_in`, `tokens_cached`, `tokens_out` | int | Cumulative token counts |
| `tools_ok`, `tools_failed`, `tools_skipped` | int | The three outcomes of the tool calls the pipe ran in a batch, as `on_tool_result` reports them: succeeded, failed, and skipped. `tools_failed` counts a tool that raised or timed out, a call the breaker let through and that then failed, and the two calls `_execute_function_calls` gives up on itself: the enqueue cut (the queue was still full when the batch ceiling ran out, so the call never started) and the idle settle under `TOOL_IDLE_TIMEOUT_SECONDS`, whether the call had started or was still waiting for a worker. `tools_skipped` comes from four places: a call the tool breaker refused to let through, a `future.done()` call whose result was no longer awaited when the batch got to it, a `cancelled` call -- one the pipe started and then abandoned -- and an `incomplete` call, the one handed back to Open WebUI to run itself because its entry carries `OWUI_OWNS_KEY`, so the tool neither failed nor ran here (`_run_tool_unless_breaker_open` returns `"skipped"` for the first two, the round's own loop dispatches `"incomplete"` for the fourth, and the tracker's status table maps `cancelled` and `incomplete` to the same counter). Either way it is a wait, not an error, and the Live tab shows it with its own glyph rather than folding it into `tools_failed`. All three are persisted by `db_row` and aggregated by every window aggregate. |
| `cost`, `task_cost` | float | Running cost; `task_cost` is the folded-in task portion |
| `worker_pid` | int | The worker that owns the row, identified by host **and** pid: a pid is only unique inside one machine, so the same number on two hosts is two workers |

`tools_skipped` is persisted and aggregated like the other two. `USAGE_ROW_FIELDS` and the ORM model both carry it, `db_row` writes it, and the cards, buckets, by-model, by-user and totals aggregates all sum it -- so the Usage tab's Tools figure equals the Live tab's over the same turns. What made the column cheap to add is that `UsageStore.ensure` reconciles against the model with `ALTER TABLE ... ADD COLUMN`: adding `tools_skipped` to `_usage_model_columns()` *is* the migration, and rows an earlier release wrote read the new column as `NULL`, which every aggregate coalesces to `0`. That is why each of the five tool-count aggregates goes through the one `_tools_sum(model)` helper and keeps all three `coalesce` terms: a bare `SUM` would drop such a row out of the sum instead of counting its other two counters as zero. `test_the_persisted_row_column_set_is_unchanged_since_the_last_release` is the tripwire, and it compares declarations to declarations with no database in between: `set(USAGE_ROW_FIELDS) == set(_RELEASED_USAGE_FIELDS)` (a frozen literal of what the last release declared, never read from the current model, so the comparison cannot go circular) and `set(_usage_model_columns()) == set(USAGE_ROW_FIELDS)`. Together they fail on a **narrowing** or a rename, and `test_db_row_mapping` fails on a name added to the projection and forgotten in `db_row`. Adding a persisted column is a supported migration, and it is deliberately **not** something the tripwire reds on by itself: the author declares it by editing the frozen list and this table, which is the friction the tripwire exists to impose. What the `ALTER TABLE ... ADD COLUMN` reconcile still does is add the column to the physical tables an earlier release created; the tripwire simply no longer goes through it to notice.

**Usage records.** With `PIPE_DASHBOARD_USAGE_COLLECT` on, each finalized session is mapped by `SessionTracker.db_row(...)` and written to the `dashboard_{suffix}` table. A video turn is finalized by the video adapter's own report rather than by the terminal backstop, so a video row carries the job's real tokens and real cost where it used to carry zeros. A video that failed, stored no clip, or stalled is written as Failed where it used to be written as Completed, and a second request that attached to an already-running job finalizes without that job's usage, so one billed clip is one row's spend rather than two. The write gate is at the writer, not the caller: `_persist_sync` reads the persisted `function.valves` row for this pipe on the writer thread, once per batch, before it builds the row instances, and a batch is written only if that row carries the key truthy. An absent key is off, and so is a row that cannot be read, cannot be decrypted, or raises — the valve's own default is `False`, so a failed read denies rather than falling back to the worker's in-memory copy, and it says so in one WARNING naming the read. Because the gate sits below both callers, the request path and the off-request sweep are decided by the same read, and a future caller of `UsageStore.record` is covered by construction rather than by remembering to check a valve. The table name is keyed on `(ARTIFACT_ENCRYPTION_KEY, pipe_id)` and is therefore stable across upgrades, so `UsageStore.ensure()` reconciles missing columns with `ALTER TABLE ... ADD COLUMN` before the model is published; rows written by an earlier release read their new columns as `NULL`, which the aggregations already treat as `0` (`usage_queries.py` sums each column bare, which ignores a `NULL` the same way `or 0` did, and wraps the addends in `coalesce` only where one expression meets two columns, where a `NULL` would otherwise take its row's value out of the whole sum). The window the purge deletes at is read from the persisted valve row on each pass, at the valve's declared default when that row cannot be read - the same reader and the same fallback `usage_queries.py` uses, so the purge and the Usage tab can never report different windows for one setting. The writer re-checks that table signature on its own write path, so a rotated `ARTIFACT_ENCRYPTION_KEY` moves usage rows to a new table on the next write without a restart, and the pre-rotation table is left behind for the operator to drop - but it is not orphaned: a rotation moves the table by changing only the hash, so the fragment - and therefore the prefix - is unchanged, and every pass enumerates the `dashboard_<fragment>_*` tables on the engine and holds each one this store published and abandoned to the same window, de-identifying its temporary-chat rows exactly as it does the current table's. The names come from `sa_inspect(engine).get_table_names()` once per pass rather than from anything a worker remembers, so a rotation applied while the workers were down - by editing the valve file and restarting - is still seen. What a pass will not do is reach a table that belongs to another installed copy: the fragment is `_sanitize_table_fragment(pipe_id)`, which lowercases and folds every character outside `[a-z0-9_]`, so two installed function ids that differ only outside that alphabet (`my-pipe` and `my_pipe`) share this prefix exactly while keeping separate tables, and a table under a fragment that another installed function id also sanitizes to is left alone and named once in the log - at the cost of this pipe's own retired tables going unswept on such an install. Each target is locked, purged and unlocked under its own lock id, derived from the table being purged rather than from the published one, so a pass over one table cannot delete a peer worker's lock on another. The pass never drops a table; it empties it, and the operator may still drop it. The purge also runs independently of `PIPE_DASHBOARD_USAGE_COLLECT`: it is started from the plugin's own lifecycle, and only for an install whose usage table already exists, so switching collection off cannot keep rows this pipe already stored past their window.  A rotation whose new table **cannot be created** is different: the writer withdraws the pre-rotation model, arms the same retry interval, and the Usage tab says `storage unavailable` until the interval elapses and a later attempt succeeds, so no rows are written to the table the operator rotated away from. `ensure()` does not move off the event loop for this: the throttled retry costs one short DDL per worker per `_US_RECONCILE_RETRY_S` (300 s) on a host that is already misconfigured, and reaching off the loop would mean changing the signature of four modules on the request path. `ensure()` is serialised, so a second thread waits for the first's build instead of running a second one, and a build that *raises* is throttled like the two refusal arms. Revisit that if a first-use DDL is measured above ~100 ms, or a deployment reports loop stalls at request completion; the off-loop route already exists as `_warm_usage_store` through `run_in_executor` (`usage_queries.py`), so it needs no new mechanism. Shutdown is the other direction of the same rule: once `signal_stop()` has been called the store is retired and starts no writer and no purge loop again -- a row finalised after the stop is still written, by a writer that drains it and exits immediately, but the purge loop does not come back, so a hot reload leaves no predecessor thread, purge task or DB connection behind. A batch the writer cannot write is not lost quietly: it is held and retried with the rows that arrive next, bounded by the batch size (and every row the hold has to shed to stay inside it is counted as dropped, as is whatever is still held when the writer stops). A pass that raises reports its row count at WARNING with the traceback -- once per outage, then at DEBUG -- and a pass that only finds the reconciliation gate closed is reported by `ensure()` itself, at the table it could not prepare. So a total that comes up short is visible in the log even when the database is the reason. A table that cannot be **created** at all -- a read-only role, a missing `CREATE` grant, an unowned schema -- is retried at most once per `_US_RECONCILE_RETRY_S` (300 s) rather than once per completed request, because `ensure()` is reached synchronously from the request path. The `storage unavailable` reason is unchanged throughout and clears within that interval once the database can create the table. The columns (`USAGE_ROW_FIELDS`, plus a generated `id`):

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
| `status` | str(12) | `ok` / `failed` / `cancelled`. This is the **third** spelling of the same outcome — the session archive writes `complete` / `error` / `cancelled`, so `ok` here is the archive's `complete` and `failed` is its `error`. `usage_queries.py` counts `status == "failed"` against this persisted column, which is why the two vocabularies are not merged |
| `duration_ms` | int | Wall-clock duration |
| `tokens_in`, `tokens_out`, `tokens_reasoning`, `tokens_cached` | int | Token counts |
| `tools_ok`, `tools_failed`, `tools_skipped`, `retries` | int | Per-request counters |
| `cost`, `cache_savings` | float | Billed cost, and cache savings: the provider-reported `cache_discount` when the response sends one, otherwise the pipe's own read-pricing estimate, which resolves the model through the catalogue's canonical id (so a variant tag, a `/`-spelled id or a mixed-case one prices the same as the base it names). Either way it is floored at zero and never negative, so a cache *write* — which the provider reports as a charge — contributes nothing here; the charge itself is in `cost`. |
| `worker_pid` | int | Writing worker, identified by host **and** pid for the same reason |

**Declared widths are reconciled, and a width never takes the store down.** `ensure()` compares column *names* and adds what is missing, but a release that also *widens* a `String(n)` needs more: the deployed column is still `n` wide, and the writer would start sending longer values into it — on PostgreSQL that is a rejected insert, and since a batch is one transaction a single rejection used to take all 50 rows with it. So the reconcile also runs a width pass (`_reconcile_widths`), driven by the model's declared lengths rather than a list of names, and issues the dialect's own `ALTER TABLE ... ALTER COLUMN ... TYPE VARCHAR(n)` for each column the table holds **narrower** than the model declares. A column already wider is left alone: narrowing it back would destroy the values that only fit the wider shape.

**If the database refuses the widen, recording continues.** An account without `ALTER` rights, and SQLite — which cannot retype a column at all and enforces no `VARCHAR(n)` anyway — leave the column at its on-disk width. That is a WARNING, not a gate failure: `ensure()` still answers `True`, `enabled` stays on and `record()` keeps writing, because a display name is a label and the cost is the data. The writer then fits every string to the **narrower of the declared and the reflected width**, so the row lands with the value truncated rather than rejected, and one warning per column names both widths and the `ALTER` that fixes it. A warning fires once per column per worker, not once per `ensure()`.

**A rejected row costs only itself.** `_persist_sync` writes the batch inside one transaction with a `SAVEPOINT` per row, so the first row the database refuses is rolled back to its own savepoint and every other row — each carrying its own cost — commits with it. `add_all` plus a single commit was all-or-nothing over the whole batch. Each rejection is logged at **WARNING** naming the row's `id`, `chat_id`, `user_id`, `model_id` and `cost` with the exception type, and is counted in `UsageStore.persist_failed`, surfaced through `_table_info_sync()` and copied into the Usage tab's `meta` by `run_usage_query` as `persist_failed`: a loss the operator cannot see is half the defect. The batch is never re-queued — a poison row in it would be an infinite retry loop.

**Range analytics.** The `usage_stats` action calls `run_usage_query(plugin, pipe, args)`, which validates the range, memoizes for 30 s, and runs one windowed aggregation in the store's DB executor, then hands the payload's deep copy to the default executor on the way out -- a copy of the answer -- ten user rows, not the whole user set -- is a far smaller hop, and the DB pool also serves the chat request path, so neither hop belongs on the event loop. The aggregation is SQL: one grouped query per window for the cards, one for the buckets, one per grouping for `by_model` and `by_user`, and the all-retained `totals` aggregate beside them, so what crosses into Python is bounded by the size of the answer -- buckets, distinct models, and at most ten distinct users plus the count and sums of all of them -- and not by the number of usage rows in the window. **The bucket edges are computed in Python and handed to SQL as range predicates**, and that is load-bearing rather than incidental: `ts` is a naive `DateTime` holding server-local time (`usage_ts_from_epoch` is the writer and its own docstring says the frame is), so an edge derived in SQL -- `strftime('%s', ts)` on sqlite, `EXTRACT(EPOCH FROM ts)` on postgres -- reads the column as UTC and is off by a constant on every deployment east or west of Greenwich, silently, with no log line. The edges come from the window (`b0 = int((start + off) // bucket_s * bucket_s - off)`) and the predicates are `ts >= b AND ts < b + bucket_s` on the naive column, which is the same comparison `usage_queries.py` always made. Two more consequences of aggregating in SQL: a `NULL` `kind` is coalesced to `chat` in the per-model grouping, so a model never splits into two rows, and the per-user display name is the name on that user's **newest** row (greatest `ts`, by a per-user ordering), not a collated maximum -- `max(user_name)` would keep `zoe@old` for a user who has since become `Aaron`. **`totals` is all-retained by design, not windowed**: it pairs with `meta.records` and `meta.approx_bytes`, which are all-retained counts, and narrowing it would move a number an operator reads.  The memo is keyed on the usage table, the request shape *and* both persisted valves it reports -- `(table, range, include_tasks, tz_offset_min, collect_on, retention_days)` -- so two pipes, and a rotated `ARTIFACT_ENCRYPTION_KEY`, never share an entry, and a change to `PIPE_DASHBOARD_USAGE_COLLECT` or `PIPE_DASHBOARD_USAGE_RETENTION_DAYS` is visible on the tab's next poll instead of after the 30 s TTL. The `args` keys are `range` (default `"24h"`), `include_tasks` (default `True`), and `tz_offset_min` (default `0`, clamped to ±900). All three are declared `optional(...)` in the action's schema, so omitting one takes that default and the empty `args` object `{}` is a valid call; a key that *is* present with the wrong type, `null` included, is still a `400 {"error": "missing or invalid: <key>"}` — a supplied `null` is not an omission, and `include_tasks: null` in particular would coerce to `False` rather than the documented `True`. Supported ranges (`USAGE_RANGES`) and their bucket sizes:

| Range | Span | Bucket |
|-------|------|--------|
| `1h` | 1 hour | 1 min |
| `6h` | 6 hours | 2 min |
| `24h` | 24 hours | 5 min |
| `7d` | 7 days | 1 hour |
| `30d` | 30 days | 4 hours |

On success the result is `{"available": true, "cards", "prev", "buckets", "by_model", "by_user", "totals", "meta"}`. A `by_user` row carries `user_name` and no account id at all: the rollup is grouped by the id and sorted by cost **server-side**, and the projection that reaches a viewer drops the id, so the rows can be ordered and counted but not joined to an account. When it cannot answer it returns `{"available": false, "reason": ...}` with one of:

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
