"""Think Streaming plugin — UI enhancement for live reasoning and tool display.

Intercepts ``reasoning:delta``, ``reasoning:completed``, and tool execution
events from the streaming loop and mirrors them into a per-session
``asyncio.Queue`` for a live SSE-powered iframe embed.  Events are also
passed through to OWUI so its middleware can build ``output`` items for DB
persistence (``serialize_output``).

A CSS rule injected into the parent document hides OWUI's native
reasoning/tool boxes while the iframe is active.  The rule persists after
streaming completes so the iframe remains the sole display; on page refresh
the iframe is gone (SSE endpoint expired) and OWUI's native boxes render
from the persisted DB data.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
from typing import Any

from pydantic import Field

from ..base import PluginBase, PluginContext
from .._utils import EphemeralKeyStore, configure_keystore_redis
from ..registry import PluginRegistry
from .session import SessionRegistry
from .sse_endpoint import register_think_streaming_route
from .wrapper import ThinkStreamingEmitterWrapper

_ts_plugin_log = logging.getLogger(__name__)


def _build_think_streaming_html(ephemeral_key: str) -> str:
    """Build the iframe HTML for the Think Streaming embed.

    Key behaviours:
    - On page refresh (SSE endpoint dead), ``es.onerror`` hides the body
      and reports height 0 — no empty block.
    - On SSE connect, a ``<style>`` tag is injected into the parent
      document to hide OWUI's native reasoning/tool boxes during
      streaming.  Removed on completion so persisted boxes render on
      page refresh.
    - Each tool line is a ``<details>`` element (collapsed by default)
      with a disclosure arrow.  Expanding shows arguments and output.
    """
    sid = "ts-" + secrets.token_hex(4)
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{
  --ts-text: #334155;
  --ts-text-dim: #64748b;
  --ts-text-muted: #94a3b8;
  --ts-border: #e2e8f0;
  --ts-box-bg: #f1f5f9;
  --ts-fade-from: #f1f5f9;
  --ts-tool-ok: #16a34a;
  --ts-tool-run: #d97706;
}}
:root.dark {{
  --ts-text: #cbd5e1;
  --ts-text-dim: #94a3b8;
  --ts-text-muted: #475569;
  --ts-border: rgba(255,255,255,0.08);
  --ts-box-bg: rgba(255,255,255,0.04);
  --ts-fade-from: rgba(30,41,59,0.95);
  --ts-tool-ok: #4ade80;
  --ts-tool-run: #fbbf24;
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body {{ background: transparent; }}
body {{
  line-height: 1.5; color: var(--ts-text);
  padding: 4px;
}}
.ts-box {{
  position: relative; overflow-y: auto;
  padding: 6px 8px; border-radius: 6px;
  border: 1px solid var(--ts-border); background: var(--ts-box-bg);
  white-space: pre-wrap; word-break: break-word;
}}
.ts-fade {{
  position: absolute; top: 0; left: 0; right: 0; height: 20px;
  pointer-events: none; border-radius: 6px 6px 0 0;
  background: linear-gradient(to bottom, var(--ts-fade-from), transparent);
}}
.ts-header {{
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 3px;
}}
.ts-label {{
  font-size: 0.75em; font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.05em; color: var(--ts-text-muted);
  margin-left: 4px;
}}
.ts-lines-select {{
  font-family: inherit; font-size: 0.75em; color: var(--ts-text-muted);
  background: transparent; border: 1px solid var(--ts-border);
  border-radius: 3px; padding: 0 2px; cursor: pointer;
  outline: none;
}}
.ts-lines-select option {{ background: var(--ts-box-bg); color: var(--ts-text); }}
.ts-empty {{ color: var(--ts-text-muted); font-style: italic; }}
.ts-section {{ margin-bottom: 4px; }}
.ts-section.hidden {{ display: none; }}
/* Tool details disclosure */
.ts-tool-item {{ margin: 1px 0; }}
.ts-tool-item summary {{
  cursor: pointer; list-style: none; padding: 1px 0;
  display: flex; align-items: center; gap: 4px;
}}
.ts-tool-item summary::-webkit-details-marker {{ display: none; }}
.ts-tool-arrow {{
  display: inline-block; width: 10px; font-size: 8px;
  color: var(--ts-text-muted); transition: transform 0.15s;
}}
.ts-tool-item[open] .ts-tool-arrow {{ transform: rotate(90deg); }}
.ts-tool-icon {{ display: inline-block; width: 14px; text-align: center; }}
.ts-tool-icon.running {{ color: var(--ts-tool-run); }}
.ts-tool-icon.done {{ color: var(--ts-tool-ok); }}
.ts-tool-name {{ color: var(--ts-text-dim); }}
.ts-tool-detail {{
  margin: 2px 0 4px 18px; padding: 3px 6px;
  font-size: 12px; border-left: 2px solid var(--ts-border);
  color: var(--ts-text-dim);
  white-space: pre-wrap; word-break: break-all;
}}
.ts-tool-detail-label {{
  font-size: 10px; font-weight: 600; text-transform: uppercase;
  color: var(--ts-text-muted); margin-bottom: 1px;
}}
</style>
</head>
<body>
<div class="ts-section" id="{sid}-think-section">
  <div class="ts-header">
    <div class="ts-label">Thinking</div>
    <select class="ts-lines-select" id="{sid}-lines" title="Visible lines">
      <option value="10" selected>10 lines</option>
      <option value="15">15 lines</option>
      <option value="20">20 lines</option>
      <option value="25">25 lines</option>
      <option value="50">50 lines</option>
    </select>
  </div>
  <div class="ts-box" id="{sid}-box">
    <div class="ts-fade" id="{sid}-fade"></div>
    <span class="ts-empty" id="{sid}-empty">Waiting for reasoning...</span>
    <span id="{sid}-content" style="display:none"></span>
  </div>
</div>
<div class="ts-section hidden" id="{sid}-tool-section">
  <div class="ts-header">
    <div class="ts-label">Tools</div>
    <select class="ts-lines-select" id="{sid}-tool-lines" title="Visible lines">
      <option value="10" selected>10 lines</option>
      <option value="15">15 lines</option>
      <option value="20">20 lines</option>
      <option value="25">25 lines</option>
      <option value="50">50 lines</option>
    </select>
  </div>
  <div class="ts-box" id="{sid}-tool-box"></div>
</div>
<script>
(function() {{
  var ID = "{sid}";
  var KEY = "{ephemeral_key}";
  var LINE_H = 18;
  var HIDE_STYLE_ID = 'ts-hide-native-' + ID;

  // Thinking elements
  var thinkSection = document.getElementById(ID + '-think-section');
  var box = document.getElementById(ID + '-box');
  var content = document.getElementById(ID + '-content');
  var empty = document.getElementById(ID + '-empty');
  var linesSel = document.getElementById(ID + '-lines');
  var hasThinking = false;

  // Tool elements
  var toolSection = document.getElementById(ID + '-tool-section');
  var toolBox = document.getElementById(ID + '-tool-box');
  var toolLinesSel = document.getElementById(ID + '-tool-lines');
  var toolsByCallId = {{}};

  function setBoxLines(b, n) {{ b.style.maxHeight = (n * LINE_H) + 'px'; }}
  linesSel.onchange = function() {{ setBoxLines(box, parseInt(this.value, 10)); reportHeight(); }};
  toolLinesSel.onchange = function() {{ setBoxLines(toolBox, parseInt(this.value, 10)); reportHeight(); }};
  setBoxLines(box, 10);
  setBoxLines(toolBox, 10);

  // Find this iframe element in the parent document
  function findMyIframe() {{
    try {{
      var iframes = parent.document.querySelectorAll('iframe');
      for (var i = 0; i < iframes.length; i++) {{
        try {{ if (iframes[i].contentWindow === window) return iframes[i]; }} catch(e) {{}}
      }}
    }} catch(e) {{}}
    return null;
  }}

  // Find the message container (div.chat-assistant) that holds both embeds and content
  function findMsgContainer() {{
    var iframe = findMyIframe();
    if (!iframe) return null;
    var el = iframe;
    while (el && el.parentElement) {{
      el = el.parentElement;
      if (el.classList && (el.classList.contains('chat-assistant') || el.classList.contains('chat-user'))) return el;
    }}
    return null;
  }}

  // Inject CSS into parent to hide OWUI native reasoning/tool boxes during streaming.
  // OWUI's Collapsible.svelte and ToolCallDisplay.svelte both render as:
  //   <div class="w-full space-y-1"> (outer)
  //     <div class="w-fit text-gray-500 ..."> (button area)
  // We target that DOM pattern, scoped to the current message via .ts-active marker.
  function injectHideCSS() {{
    try {{
      var container = findMsgContainer();
      if (container) container.classList.add('ts-active');
      var s = parent.document.createElement('style');
      s.id = HIDE_STYLE_ID;
      s.textContent = '.ts-active div.w-full.space-y-1:has(> .w-fit.text-gray-500){{display:none!important}}';
      parent.document.head.appendChild(s);
    }} catch(e) {{}}
  }}

  // Remove the injected CSS so OWUI native boxes show after completion
  function removeHideCSS() {{
    try {{
      var s = parent.document.getElementById(HIDE_STYLE_ID);
      if (s) s.remove();
      var container = findMsgContainer();
      if (container) container.classList.remove('ts-active');
    }} catch(e) {{}}
  }}

  // Hide the iframe container in the parent document (prevents empty block)
  function hideIframeContainer() {{
    try {{
      var iframe = findMyIframe();
      if (iframe) {{
        var wrapper = iframe.parentElement;
        if (wrapper) wrapper.style.display = 'none';
      }}
    }} catch(e) {{}}
  }}

  // Theme sync
  function syncTheme() {{
    try {{
      var isDark = parent.document.documentElement.classList.contains('dark');
      document.documentElement.classList.toggle('dark', isDark);
    }} catch (e) {{}}
  }}
  syncTheme();
  try {{
    new MutationObserver(syncTheme).observe(
      parent.document.documentElement,
      {{ attributes: true, attributeFilter: ['class'] }}
    );
  }} catch (e) {{}}

  function reportHeight() {{
    try {{
      // Temporarily collapse body to measure natural content height.
      // Without this, scrollHeight returns max(content, viewport) and
      // the iframe can never shrink after expanding.
      document.body.style.height = '0px';
      var h = document.body.scrollHeight;
      document.body.style.height = '';
      parent.postMessage({{ type: 'iframe:height', height: h }}, '*');
    }} catch(e) {{}}
  }}

  function escHtml(s) {{
    return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  }}

  function addToolLine(callId, name, args) {{
    toolSection.classList.remove('hidden');
    var det = document.createElement('details');
    det.className = 'ts-tool-item';
    det.id = ID + '-tool-' + callId;
    var summary = document.createElement('summary');
    summary.innerHTML = '<span class="ts-tool-arrow">&#9654;</span>' +
      '<span class="ts-tool-icon running">&#9711;</span> ' +
      '<span class="ts-tool-name">' + escHtml(name) + '</span>';
    det.appendChild(summary);
    // Arguments section
    var argsDiv = document.createElement('div');
    argsDiv.className = 'ts-tool-detail';
    argsDiv.id = ID + '-tool-args-' + callId;
    if (args) {{
      try {{
        var parsed = JSON.parse(args);
        argsDiv.innerHTML = '<div class="ts-tool-detail-label">Input</div>' +
          escHtml(JSON.stringify(parsed, null, 2));
      }} catch(e) {{
        argsDiv.innerHTML = '<div class="ts-tool-detail-label">Input</div>' + escHtml(args);
      }}
    }} else {{
      argsDiv.innerHTML = '<div class="ts-tool-detail-label">Input</div><em style="color:var(--ts-text-muted)">pending...</em>';
    }}
    det.appendChild(argsDiv);
    // Output placeholder
    var outDiv = document.createElement('div');
    outDiv.className = 'ts-tool-detail';
    outDiv.id = ID + '-tool-out-' + callId;
    outDiv.style.display = 'none';
    det.appendChild(outDiv);
    // Re-measure iframe height when tool details expand/collapse.
    // requestAnimationFrame defers until after browser layout recalculation.
    det.addEventListener('toggle', function() {{ requestAnimationFrame(reportHeight); }});
    toolBox.appendChild(det);
    toolBox.scrollTop = toolBox.scrollHeight;
    toolsByCallId[callId] = {{ el: det, name: name }};
    reportHeight();
  }}

  function completeToolLine(callId, output) {{
    var t = toolsByCallId[callId];
    if (!t) return;
    var icon = t.el.querySelector('.ts-tool-icon');
    if (icon) {{ icon.className = 'ts-tool-icon done'; icon.innerHTML = '&#10003;'; }}
    // Show output if available
    if (output) {{
      var outDiv = document.getElementById(ID + '-tool-out-' + callId);
      if (outDiv) {{
        outDiv.style.display = '';
        try {{
          var parsed = JSON.parse(output);
          outDiv.innerHTML = '<div class="ts-tool-detail-label">Output</div>' +
            escHtml(JSON.stringify(parsed, null, 2));
        }} catch(e) {{
          outDiv.innerHTML = '<div class="ts-tool-detail-label">Output</div>' + escHtml(output);
        }}
      }}
    }}
    reportHeight();
  }}

  // Inject CSS immediately to pre-hide native boxes
  injectHideCSS();

  var es = new EventSource('/api/pipe/think_streaming/' + KEY);
  es.onmessage = function(e) {{
    try {{
      var d = JSON.parse(e.data);
      if (d.type === 'thinking_delta') {{
        hasThinking = true;

        empty.style.display = 'none';
        content.style.display = '';
        content.textContent = d.content || '';
        box.scrollTop = box.scrollHeight;
      }}
      if (d.type === 'thinking_done') {{
        hasThinking = true;

        empty.style.display = 'none';
        content.style.display = '';
        if (d.content) content.textContent = d.content;
      }}
      if (d.type === 'tool_start') {{
        addToolLine(d.call_id, d.name, d.arguments || '');
      }}
      if (d.type === 'tool_done') {{
        completeToolLine(d.call_id, d.output || '');
      }}
      if (d.type === 'done') {{
        es.close();
        // Keep native boxes hidden — they reappear on page refresh from DB
        var hasTools = toolBox.children.length > 0;
        if (!hasThinking && !hasTools) {{
          document.body.style.display = 'none';
          reportHeight();
          hideIframeContainer();
          return;
        }}
        if (!hasThinking) thinkSection.style.display = 'none';
      }}
      reportHeight();
    }} catch(err) {{}}
  }};
  es.onerror = function() {{
    es.close();
    removeHideCSS();
    document.body.style.display = 'none';
    reportHeight();
    hideIframeContainer();
  }};
  reportHeight();
  new MutationObserver(reportHeight).observe(document.body, {{ childList: true, subtree: true }});
}})();
</script>
</body>
</html>"""


@PluginRegistry.register
class ThinkStreamingPlugin(PluginBase):
    """Streams model thinking tokens into a live iframe embed.

    Subscribes to ``on_emitter_wrap`` (to intercept reasoning events)
    and ``on_response_transform`` (to signal session completion).
    """

    plugin_id = "think-streaming"
    plugin_name = "Think Streaming"
    plugin_version = "1.0.0"
    hooks = {
        "on_emitter_wrap": 50,
        "on_response_transform": 50,
    }
    plugin_valves = {
        "THINK_STREAMING_USER_ENABLE": (bool, Field(
            default=False,
            title="Enable UI enhancement plugin",
            description="Show live thinking and tool execution in an interactive embed during streaming.",
        )),
    }
    plugin_user_valves = {
        "THINK_STREAMING_USER_ENABLE": (bool, Field(
            default=False,
            title="Enable UI enhancement plugin",
            description="Show live thinking and tool execution in an interactive embed during streaming.",
        )),
    }

    def __init__(self) -> None:
        super().__init__()
        self._key_store = EphemeralKeyStore(ttl=600.0)
        self._session_registry = SessionRegistry(max_sessions=100, ttl=600.0)
        self._request_sessions: dict[str, str] = {}  # request_id → ephemeral_key
        self._cleanup_task: asyncio.Task[None] | None = None

    def on_init(self, ctx: PluginContext, **kwargs: Any) -> None:
        self.ctx = ctx

        # Wire Redis into key store for cross-worker key sharing
        configure_keystore_redis(self._key_store, self.ctx.pipe)

        # Register the SSE endpoint
        register_think_streaming_route(self._key_store, self._session_registry)

        # Start background cleanup task
        self._maybe_start_cleanup()

        # Warn if multi-worker and no Redis (sessions are process-local)
        workers = os.environ.get("UVICORN_WORKERS", "1")
        if workers != "1" and self._key_store._redis_client is None:
            _ts_plugin_log.warning(
                "Think Streaming: multi-worker detected (%s) but no Redis. "
                "SSE sessions are process-local and may not work reliably.",
                workers,
            )

    def _maybe_start_cleanup(self) -> None:
        """Start the periodic cleanup background task if an event loop is available."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = loop.create_task(
                self._cleanup_loop(),
                name="think-streaming-cleanup",
            )

    async def _cleanup_loop(self) -> None:
        """Periodically remove expired sessions and stale request mappings."""
        while True:
            await asyncio.sleep(60)
            try:
                self._session_registry.cleanup_expired()
                # Clean stale request mappings (sessions that no longer exist)
                stale = [
                    rid for rid, key in self._request_sessions.items()
                    if self._session_registry.get(key) is None
                ]
                for rid in stale:
                    del self._request_sessions[rid]
            except Exception:
                _ts_plugin_log.debug("Think Streaming cleanup error", exc_info=True)

    async def on_emitter_wrap(
        self,
        stream_emitter: Any,
        **kwargs: Any,
    ) -> Any | None:
        valves = kwargs.get("valves")
        if valves is None:
            return None
        raw_emitter = kwargs.get("raw_emitter")
        job_metadata = kwargs.get("job_metadata", {})

        # Check user valve — lives on UserValves, may be absent from merged Valves
        if not getattr(valves, "THINK_STREAMING_USER_ENABLE", False):
            return None

        # Prevent double-wrapping
        if getattr(stream_emitter, "_think_streaming_active", False):
            return None

        # Lazy Redis configuration (handles case where Redis initialises after on_init)
        if self._key_store._redis_client is None:
            configure_keystore_redis(self._key_store, self.ctx.pipe)

        # Create session
        key = await self._key_store.async_generate()
        user_id = job_metadata.get("user_id", "")
        session = self._session_registry.create(key, user_id=user_id)

        # Track request → session mapping for cleanup in on_response_transform
        request_id = job_metadata.get("request_id", "")
        if request_id:
            self._request_sessions[request_id] = key

        # Build iframe HTML — emission is deferred to the wrapper on first
        # thinking or tool event so the iframe only appears when needed.
        html = _build_think_streaming_html(key)

        return ThinkStreamingEmitterWrapper(
            stream_emitter, session,
            raw_emitter=raw_emitter,
            iframe_html=html,
        )

    async def on_response_transform(
        self,
        completion_data: dict[str, Any],
        model: str,
        metadata: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Push completion sentinel so the SSE generator closes cleanly."""
        # Find the session for this request via SessionLogger context var
        try:
            from ...core.logging_system import SessionLogger
            request_id = SessionLogger.request_id.get() or ""
        except Exception:
            request_id = ""

        if not request_id:
            return

        key = self._request_sessions.pop(request_id, None)
        if not key:
            return

        session = self._session_registry.get(key)
        if session is None:
            return

        # Push completion sentinel so SSE generator closes
        try:
            session.queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    def on_shutdown(self, **kwargs: Any) -> None:
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
