# Think Streaming Plugin -- Developer Guide

> Live reasoning and tool execution display via SSE-powered iframe embeds.

![Think Streaming](images/plugin_think_streaming.gif)

---

## Overview

The Think Streaming plugin intercepts model thinking tokens and tool execution events from the streaming pipeline and mirrors them into a live, auto-scrolling iframe embed inside the chat message. Instead of Open WebUI's native disclosure boxes ("Thought for N seconds" / tool cards that appear after completion), users see reasoning happen in real time — character by character — in a terminal-like display.

**How it works in one sentence:** The plugin wraps the stream event emitter, copies thinking and tool events into an `asyncio.Queue`, and serves them to an iframe via a dynamically-registered SSE endpoint on OWUI's own FastAPI application.

**Prerequisite:** `ENABLE_PLUGIN_SYSTEM` must be `True` (default: `False`). This pipe-level valve controls whether any plugins are loaded. See [Plugin System -- Developer Guide](plugin_system.md).

**Plugin-exported valves:**

| Valve | Type | Default | Scope | Description |
|-------|------|---------|-------|-------------|
| `THINK_STREAMING_USER_ENABLE` | bool | `False` | Per-user | Show live thinking/tool execution in an interactive embed during streaming |

There is no admin-level system valve — the per-user valve is the only control. Users must opt in by enabling it in their valve settings.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [The SSE Transport — How the Pipe Talks to OWUI](#the-sse-transport--how-the-pipe-talks-to-owui)
4. [Event Flow](#event-flow)
5. [Deferred Iframe Emission](#deferred-iframe-emission)
6. [CSS Injection — Hiding Native OWUI Boxes](#css-injection--hiding-native-owui-boxes)
7. [Emitter Wrapper Pattern](#emitter-wrapper-pattern)
8. [Session Management](#session-management)
9. [Ephemeral Key Authentication](#ephemeral-key-authentication)
10. [Graceful Degradation](#graceful-degradation)
11. [The Dashboard UI](#the-dashboard-ui)
12. [Limitations](#limitations)
13. [File Structure](#file-structure)
14. [See Also](#see-also)

---

## Architecture

```
User sends message to a reasoning model (e.g., Claude with extended thinking)
  │
  ▼
pipe() → _execute_pipe_job()
  │
  ▼
_make_middleware_stream_emitter() creates stream_emitter
  │
  ▼
dispatch_on_emitter_wrap(stream_emitter, raw_emitter=job.event_emitter)
  │
  ├──► ThinkStreamingPlugin.on_emitter_wrap():
  │       1. Check user valve (THINK_STREAMING_USER_ENABLE)
  │       2. Create ThinkSession (asyncio.Queue per request)
  │       3. Generate ephemeral key for SSE endpoint
  │       4. Build iframe HTML (not emitted yet — deferred)
  │       5. Return ThinkStreamingEmitterWrapper(stream_emitter, session)
  │
  ▼
Streaming loop (_run_streaming_loop) emits events through wrapper:
  │
  ├── reasoning:delta ──────► COPY to queue → SSE → iframe
  │                           AND pass through to OWUI (for DB persistence)
  │
  ├── reasoning:completed ──► COPY to queue → SSE → iframe
  │                           AND pass through to OWUI
  │
  ├── response.output_item.added (function_call) ──► COPY + pass through
  │
  ├── response.output_item.added (function_call_output) ──► COPY + pass through
  │
  ├── chat:message:delta ──► Pass through unchanged (normal text)
  │
  └── chat:completion ─────► Pass through unchanged
  │
  ▼
on_response_transform fires:
  Plugin pushes None sentinel → SSE generator yields {"type":"done"} → iframe stops
```

The plugin subscribes to two hooks at priority **50**:

| Hook | Purpose |
|------|---------|
| `on_emitter_wrap` | Wrap the stream emitter to copy thinking/tool events to the SSE queue |
| `on_response_transform` | Push completion sentinel to close the SSE stream cleanly |

---

## The SSE Transport — How the Pipe Talks to OWUI

This is the most architecturally interesting part of the plugin. Open WebUI pipes run as Python functions inside OWUI's process — they have no HTTP server, no WebSocket connection, and no way to push data to the browser outside of the event emitter. The Think Streaming plugin solves this limitation with a pattern that's worth understanding in detail.

### The Problem

OWUI's event emitter (`__event_emitter__`) sends events to the browser via Socket.IO. But it's designed for structured events like `chat:message:delta`, `status`, and `chat:completion`. There's no built-in way to stream arbitrary live data (like reasoning tokens) into a custom UI component in real time.

You could emit all reasoning content through `chat:message:delta`, but that mixes thinking with regular text. You could emit it through `status` events, but those are transient UI overlays, not persistent content. Neither approach gives you a dedicated, scrollable, live-updating display.

### The Solution

The plugin dynamically registers a FastAPI endpoint on OWUI's own web application:

```python
# sse_endpoint.py — simplified
from .._utils import register_sse_endpoint

async def _think_streaming_sse(key: str):
    # Validate ephemeral key, look up session, stream events
    if not await key_store.async_validate(key):
        return PlainTextResponse("Invalid or expired key", status_code=403)
    session = session_registry.get(key)
    return StreamingResponse(
        _generate(session, key_store, key),
        media_type="text/event-stream",
    )

register_sse_endpoint(
    "/api/pipe/think_streaming/{key}",
    _think_streaming_sse,
    logger=log,
)
```

**What just happened:**

1. `register_sse_endpoint()` imported OWUI's FastAPI `app` object (they share the same process)
2. It registered a new GET route on that app — OWUI's web server now serves this endpoint
3. It pushed OWUI's SPA catch-all mount (`SPAStaticFiles` at `/`) to the end of the route list, so the new route gets matched first
4. The handler uses `async_validate()` which checks local dict first, then Redis — enabling cross-worker key validation

The pipe now has a live HTTP endpoint. It uses this to serve Server-Sent Events (SSE) to an iframe embed:

```
┌────────────────────────────────────────────────────┐
│  Browser                                           │
│                                                    │
│  ┌──────────────────────────────────────────────┐  │
│  │  OWUI Chat Message                           │  │
│  │                                              │  │
│  │  ┌────────────────────────────────────────┐  │  │
│  │  │  <iframe> — Think Streaming embed      │  │  │
│  │  │                                        │  │  │
│  │  │  EventSource('/api/pipe/               │  │  │
│  │  │    think_streaming/{KEY}')             │  │  │
│  │  │         ▲                              │  │  │
│  │  └─────────┼────────────────────────────┘  │  │
│  │            │                               │  │
│  └────────────┼───────────────────────────────┘  │
│               │ SSE stream                        │
└───────────────┼───────────────────────────────────┘
                │
    ┌───────────┴────────────────────────────┐
    │  OWUI Server (same process as pipe)    │
    │                                        │
    │  GET /api/pipe/think_streaming/{key}   │
    │       │                                │
    │       ▼                                │
    │  SSE generator ← asyncio.Queue         │
    │                      ▲                 │
    │                      │ put_nowait()    │
    │  ThinkStreamingEmitterWrapper          │
    │       ▲                                │
    │       │ events from streaming loop     │
    │  _run_streaming_loop()                 │
    └────────────────────────────────────────┘
```

The iframe's `EventSource` connects to the pipe's SSE endpoint. The emitter wrapper copies events to the session queue. The SSE generator drains the queue and pushes events to the browser. All of this happens within a single OWUI process — no external services, no WebSocket setup, no additional ports.

### Why This Pattern Matters

This is a general-purpose technique for any pipe plugin that needs to serve custom data to the browser:

1. Define an async handler function
2. Call `register_sse_endpoint(path, handler)` — it handles OWUI app import, route registration, idempotency, and SPA reordering
3. Use `EphemeralKeyStore` for authentication (with automatic Redis dual-write in multi-worker deployments)
4. Emit an iframe or other HTML that connects to your endpoint

The Pipe Stats Dashboard plugin uses exactly the same pattern for its live dashboard. Both plugins share `register_sse_endpoint()` and `EphemeralKeyStore` from `plugins/_utils.py`.

---

## Event Flow

### Events Copied to SSE Queue

| Internal Event Type | SSE Type | Data Fields |
|---|---|---|
| `reasoning:delta` | `thinking_delta` | `delta` (incremental text), `content` (full buffer) |
| `reasoning:completed` | `thinking_done` | `content` (full reasoning text) |
| `response.output_item.added` (function_call) | `tool_start` | `name`, `call_id`, `arguments` |
| `response.output_item.added` (function_call_output) | `tool_done` | `call_id`, `status`, `output` (capped at 2000 chars) |

### Events Passed Through Only

| Event Type | Notes |
|---|---|
| `chat:message:delta` | Normal text content — goes straight to OWUI |
| `chat:completion` | End-of-response signal — goes straight to OWUI |
| `status` | General status messages — passed through |
| All others | Passed through unchanged |

**Critical design choice:** Events are **copied**, not redirected. Every event reaches OWUI's middleware regardless of what the wrapper does. This ensures OWUI can build its `output` items for DB persistence (`serialize_output`). When a user refreshes the page, OWUI's native reasoning/tool boxes render from the stored data — the iframe is gone (SSE endpoint expired), but the content is preserved.

---

## Deferred Iframe Emission

The iframe is not emitted immediately when the emitter is wrapped. Instead, emission is **deferred to the first thinking or tool event**:

```python
# wrapper.py — simplified
async def __call__(self, event):
    if event.get("type") in ("reasoning:delta", "reasoning:completed"):
        await self._maybe_emit_iframe()  # First call emits, subsequent calls are no-ops
        self._route(event)               # Copy to SSE queue
        await self._original(event)      # Pass through to OWUI
```

**Why defer?** Not all models reason. If a user sends a message to a non-reasoning model with Think Streaming enabled, there would be no thinking content — just an empty iframe taking up space. Deferred emission means the iframe only appears when there is actual content to show.

The iframe HTML is built in `on_emitter_wrap` and passed to the wrapper. The wrapper uses `raw_emitter` (a direct line to OWUI's event handler that bypasses the middleware stream queue) to inject the iframe as an embed event:

```python
await self._raw_emitter({
    "type": "embeds",
    "data": {"embeds": [iframe_html]},
})
```

Using `raw_emitter` instead of the stream queue ensures the iframe appears **immediately** — before any thinking content starts flowing.

---

## CSS Injection — Hiding Native OWUI Boxes

When the iframe is active, OWUI's native reasoning disclosure boxes ("Thought for N seconds") and tool cards would appear alongside the iframe, creating duplicate content. The plugin hides them using CSS injection:

```javascript
// Inject CSS into the parent document
var s = parent.document.createElement('style');
s.textContent = '.ts-active div.w-full.space-y-1:has(> .w-fit.text-gray-500){display:none!important}';
parent.document.head.appendChild(s);
```

**Lifecycle:**
- **On SSE connect:** CSS is injected, native boxes hidden. A `.ts-active` class is added to the message container to scope the rule.
- **On SSE completion (done event):** CSS rule **persists** — the iframe remains the sole display for thinking/tool content.
- **On SSE error (endpoint dead, page refresh):** CSS rule is **removed**, iframe hides itself, and OWUI's native boxes render from DB data.

This creates a seamless experience: during the current session, thinking content is shown live in the iframe. After a page refresh, the same content appears in OWUI's native rendering. No data is ever lost.

**Note:** This relies on `allow-same-origin` being enabled on the OWUI instance, as the iframe needs to access `parent.document` to inject CSS.

---

## Emitter Wrapper Pattern

The `ThinkStreamingEmitterWrapper` is a callable class that implements the emitter interface:

```python
class ThinkStreamingEmitterWrapper:
    _think_streaming_active = True  # Class flag for double-wrap prevention

    async def __call__(self, event):
        # Route thinking/tool events to queue AND pass through
        # Pass everything else through unchanged

    def __getattr__(self, name):
        # Delegate to original emitter (preserves flush_reasoning_status, etc.)
        return getattr(self._original, name)
```

Key design decisions:

1. **Class-level `_think_streaming_active` flag**: Checked by `on_emitter_wrap` to prevent double-wrapping if the hook fires twice.
2. **`__getattr__` delegation**: Any attribute not defined on the wrapper is forwarded to the original emitter. This preserves `flush_reasoning_status` and any future attributes that downstream code may access.
3. **Non-blocking queue push**: `put_nowait()` with `QueueFull` catch. If the queue fills (500 items), events are dropped rather than blocking the streaming loop.
4. **Consumer-alive check**: If the SSE consumer disconnects, the wrapper stops pushing to the queue but continues passing events through to OWUI — no data loss.

---

## Session Management

Each streaming request with Think Streaming enabled gets a `ThinkSession`:

```python
@dataclass
class ThinkSession:
    key: str                          # Ephemeral key for SSE endpoint
    queue: asyncio.Queue[str | None]  # maxsize=500
    created_at: float                 # time.monotonic()
    consumer_alive: bool = True       # False when SSE client disconnects
    user_id: str = ""
```

The `SessionRegistry` manages active sessions with layered cleanup:

| Mechanism | When | Purpose |
|---|---|---|
| **Normal completion** | `on_response_transform` pushes `None` sentinel | SSE generator yields `{"type":"done"}`, session removed |
| **Consumer disconnect** | SSE generator `finally` block | Sets `consumer_alive = False`, session stays until TTL |
| **TTL expiry** | Background task every 60s | Removes sessions older than 10 minutes |
| **Capacity eviction** | On new session creation | If at 100 sessions, oldest evicted |

---

## Ephemeral Key Authentication

SSE connections use the same `EphemeralKeyStore` class (from `plugins/_utils.py`) as the Pipe Stats Dashboard, but with a separate instance and higher capacity for per-request sessions:

| Setting | Pipe Stats | Think Streaming |
|---|---|---|
| TTL | 300s (5 min) | 600s (10 min) |
| Max keys | 10 | 100 |

Keys are 256-bit cryptographic tokens generated via `secrets.token_hex(32)`. Both plugins call `configure_redis()` on their key store during `on_init()` when Redis is available, enabling cross-worker key validation in multi-worker deployments. Key generation uses `await async_generate()` and validation uses `await async_validate()` — which checks local dict first, then falls back to Redis on miss.

---

## Graceful Degradation

The plugin is designed to degrade gracefully in every failure scenario:

| Scenario | Behavior |
|---|---|
| Non-reasoning model | No iframe emitted (deferred emission); no visual change |
| SSE consumer disconnects mid-stream | Wrapper stops pushing to queue; events still reach OWUI |
| Queue fills up | Events dropped (no block); OWUI still gets everything |
| OWUI app not importable | Plugin init skips SSE registration; wrapper still passes through |
| Page refresh | SSE endpoint expired; iframe hides; native OWUI boxes render from DB |
| Multi-worker deployment | Keys shared via Redis; SSE sessions remain process-local (warning logged if Redis unavailable) |
| Plugin valve disabled | `on_emitter_wrap` returns `None`; no wrapping occurs |

---

## The Dashboard UI

The iframe contains two sections:

### Thinking Section
- Monospace, auto-scrolling box with configurable visible lines (10/15/20/25/50)
- Top fade gradient indicates truncated history
- "Waiting for reasoning..." placeholder until first token arrives
- Shows full reasoning buffer (not just deltas)

### Tools Section
- Hidden until first tool event
- Each tool is a `<details>` disclosure element (collapsed by default)
- Running tools show an amber circle icon; completed tools show a green checkmark
- Expanding shows Input (parsed JSON, pretty-printed) and Output sections
- Configurable visible lines with its own selector

### Theme Sync
The iframe syncs its color theme with the parent document by observing the `dark` class on `<html>` via `MutationObserver`. CSS custom properties switch between light and dark palettes.

### Height Management
The iframe reports its content height to the parent via `postMessage({type: 'iframe:height', height: h})`. To handle height shrinking correctly (when thinking text wraps into fewer lines), `reportHeight()` temporarily sets `body.style.height = '0px'`, measures `scrollHeight`, then clears the override.

---

## Limitations

- **SSE sessions are process-local**: While ephemeral keys are now shared across workers via Redis, the `asyncio.Queue` instances and `SessionRegistry` remain process-local. In multi-worker deployments, the SSE request may hit a different worker than the one running the streaming loop. The plugin logs a warning at init when Redis is not available and `UVICORN_WORKERS > 1`. When Redis is available, key validation works cross-worker, but the session data itself must be on the same worker as the streaming loop.
- **Requires `allow-same-origin`**: CSS injection into the parent document requires the OWUI instance to have `allow-same-origin` enabled on iframes.
- **Tool output capped at 2000 characters**: Large tool outputs are truncated in the SSE payload to prevent memory issues.

---

## File Structure

```
plugins/think_streaming/
├── __init__.py           # Re-export ThinkStreamingPlugin
├── plugin.py             # ThinkStreamingPlugin class, iframe HTML builder
├── wrapper.py            # ThinkStreamingEmitterWrapper callable class
├── session.py            # ThinkSession dataclass, SessionRegistry
└── sse_endpoint.py       # SSE route registration + async generator
```

| File | Responsibility |
|------|---------------|
| `plugin.py` | Plugin class, valve declarations, `on_emitter_wrap` / `on_response_transform` hooks, iframe HTML generation (inline CSS + JS) |
| `wrapper.py` | Emitter wrapper that inspects events, copies thinking/tool events to session queue, defers iframe emission, delegates attribute access to original |
| `session.py` | `ThinkSession` dataclass (queue + metadata), `SessionRegistry` (creation, lookup, TTL cleanup, capacity eviction) |
| `sse_endpoint.py` | Registers `GET /api/pipe/think_streaming/{key}` on OWUI's FastAPI app, SSE generator with heartbeats and completion sentinel |

---

## See Also

- [Plugin System -- Developer Guide](plugin_system.md) -- Hook system reference, including the `on_emitter_wrap` hook that this plugin uses.
- [Pipe Stats Dashboard Plugin](plugins_pipe_stats.md) -- Uses the same SSE transport pattern for a live admin diagnostics dashboard.
- [Streaming Pipeline & Emitters](streaming_pipeline_and_emitters.md) -- Details on the streaming loop, event types, and emitter architecture that this plugin wraps.
