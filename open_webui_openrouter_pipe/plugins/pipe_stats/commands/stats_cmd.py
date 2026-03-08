"""Stats command — fully dynamic HTML dashboard powered by SSE.

The ``stats`` command emits an HTML shell with empty containers and
JavaScript that connects to the SSE endpoint.  All data arrives via
tiered JSON events — nothing is rendered statically.

The dashboard uses a tabbed layout: Live, System, Storage, About.
"""

from __future__ import annotations

import html
import secrets
from typing import Any

from ..command_registry import register_command
from ..context import CommandContext


def _safe(val: Any) -> str:
    """HTML-escape a value for safe embedding."""
    return html.escape(str(val))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_stats_plugin(pipe: Any) -> Any | None:
    """Find the pipe-stats plugin instance from the pipe's plugin registry."""
    pr = getattr(pipe, "_plugin_registry", None)
    if pr is None:
        return None
    for plugin in getattr(pr, "_plugins", []):
        if getattr(plugin, "plugin_id", None) == "pipe-stats":
            return plugin
    return None


def _get_ephemeral_key(pipe: Any) -> str | None:
    """Generate an ephemeral key from the Pipe Stats Dashboard plugin's key store."""
    plugin = _get_stats_plugin(pipe)
    if plugin is None:
        return None
    store = getattr(plugin, "_key_store", None)
    if store is not None:
        return store.generate()
    return None


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_PS_STATS_CSS = """\
/* Light-mode defaults */
:root {
  --text: #1e293b;
  --text-dim: #475569;
  --text-muted: #64748b;
  --text-faint: #94a3b8;
  --border: #e2e8f0;
  --border-light: #cbd5e1;
  --sticky-bg: rgba(255,255,255,0.95);
  --expired-bg: rgba(245,158,11,0.06);
  --expired-border: rgba(245,158,11,0.25);
  --tab-bg: rgba(0,0,0,0.03);
  --tab-active-bg: rgba(99,102,241,0.1);
  --tab-active-text: #6366f1;
}
/* Dark override — mirrors OWUI .dark class from parent */
:root.dark {
  --text: #e2e8f0;
  --text-dim: #94a3b8;
  --text-muted: #64748b;
  --text-faint: #475569;
  --border: rgba(255,255,255,0.08);
  --border-light: rgba(255,255,255,0.12);
  --sticky-bg: rgba(17,24,39,0.95);
  --expired-bg: rgba(245,158,11,0.08);
  --expired-border: rgba(245,158,11,0.2);
  --tab-bg: rgba(255,255,255,0.04);
  --tab-active-bg: rgba(99,102,241,0.15);
  --tab-active-text: #818cf8;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body { background: transparent; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: var(--text); padding: 16px; line-height: 1.4;
}
.dash { }

/* Connection bar */
.live-bar { display: flex; align-items: center; gap: 8px; margin-bottom: 14px;
  justify-content: space-between; flex-wrap: wrap; }
.live-indicator { display: flex; align-items: center; gap: 5px; }
.live-dot { width: 7px; height: 7px; border-radius: 50%; background: #22c55e;
  animation: pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
.live-label { font-size: 11px; font-weight: 700; color: #22c55e;
  text-transform: uppercase; letter-spacing: 0.1em; }
.live-meta { font-size: 11px; color: var(--text-faint); font-family: 'JetBrains Mono', monospace; }

/* Tab bar */
.tab-bar { display: flex; gap: 2px; margin-bottom: 14px; border-bottom: 1px solid var(--border);
  padding-bottom: 0; }
.tab-btn { padding: 7px 14px; font-size: 12px; font-weight: 600; color: var(--text-muted);
  background: none; border: none; cursor: pointer; border-bottom: 2px solid transparent;
  margin-bottom: -1px; transition: all 0.15s; font-family: inherit;
  text-transform: uppercase; letter-spacing: 0.04em; }
.tab-btn:hover { color: var(--text-dim); background: var(--tab-bg); }
.tab-btn.active { color: var(--tab-active-text); border-bottom-color: var(--tab-active-text);
  background: var(--tab-active-bg); }
.tab-pane { display: none; }
.tab-pane.active { display: block; }

/* Section headings */
.section-h { font-size: 13px; color: var(--text-dim); font-weight: 600; margin: 16px 0 8px;
  text-transform: uppercase; letter-spacing: 0.06em; }

/* Two-column grid */
.g { display: grid; grid-template-columns: 1fr 1fr; gap: 2px 24px; margin-bottom: 12px; }
.gc { display: flex; justify-content: space-between; padding: 5px 0;
  border-bottom: 1px solid var(--border); font-size: 14px; }
.gc-l { color: var(--text-dim); }
.gc-v { color: var(--text); font-family: 'JetBrains Mono', monospace; font-size: 13px;
  text-align: right; }

/* Tables */
.tbl { width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 10px; }
.tbl th { text-align: left; color: var(--text-dim); font-weight: 600; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.06em; padding: 5px 8px;
  border-bottom: 1px solid var(--border-light); white-space: nowrap;
  background: var(--sticky-bg); }
.tbl th.r { text-align: right; }
.tbl td { padding: 5px 8px; border-bottom: 1px solid var(--border);
  color: var(--text-dim); vertical-align: top; }
.tbl td.r { text-align: right; font-family: 'JetBrains Mono', monospace; font-size: 12px;
  white-space: nowrap; }
.tbl td.name { color: var(--text); font-weight: 500; max-width: 200px;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.tbl tr:last-child td { border-bottom: none; }

/* Table overflow wrapper */
.tbl-wrap { overflow-x: auto; -webkit-overflow-scrolling: touch; }

/* Scrollable table — sticky header */
.scroll-table { max-height: 200px; overflow-y: auto; overflow-x: auto;
  -webkit-overflow-scrolling: touch; margin-bottom: 10px; }
.scroll-table .tbl thead th { position: sticky; top: 0; background: var(--sticky-bg);
  z-index: 1; box-shadow: 0 1px 0 var(--border-light); }

/* Sub-headings */
.sub-h { font-size: 11px; color: var(--text-muted); font-weight: 600; text-transform: uppercase;
  letter-spacing: 0.06em; margin: 12px 0 6px; }

/* Badges */
.b { display: inline-block; padding: 2px 8px; border-radius: 5px;
  font-size: 12px; font-weight: 600; }
.b-ok { background: rgba(34,197,94,0.15); color: #22c55e; }
.b-warn { background: rgba(245,158,11,0.15); color: #f59e0b; }
.b-off { background: rgba(100,116,139,0.15); color: #64748b; }
.b-err { background: rgba(239,68,68,0.15); color: #ef4444; }
.b-info { background: rgba(99,102,241,0.15); color: #6366f1; }

/* Loading placeholder */
.loading { color: var(--text-faint); font-style: italic; font-size: 13px; padding: 6px 0; }

/* Expired / Error states */
.live-expired { font-size: 13px; color: #f59e0b; text-align: center;
  padding: 10px; background: var(--expired-bg); border: 1px solid var(--expired-border);
  border-radius: 8px; margin-bottom: 12px; }
.live-error { font-size: 12px; color: #ef4444; text-align: center;
  padding: 6px; margin-bottom: 8px; }

/* Footer */
.footer { font-size: 11px; color: var(--text-faint);
  margin-top: 14px; padding-top: 8px; border-top: 1px solid var(--border); }

/* Mobile responsive */
@media (max-width: 480px) {
  body { padding: 10px; }
  .g { grid-template-columns: 1fr; gap: 0; }
  .tab-btn { padding: 6px 10px; font-size: 11px; }
  .gc { font-size: 13px; }
  .gc-v { font-size: 12px; }
  .tbl { font-size: 12px; }
  .tbl th, .tbl td { padding: 4px 6px; }
}
"""


# ---------------------------------------------------------------------------
# Dashboard shell
# ---------------------------------------------------------------------------

def _build_dashboard_shell(dash_id: str, ephemeral_key: str) -> str:
    """Build the fully dynamic dashboard HTML shell.

    All containers are empty — JavaScript populates them from SSE events.
    Uses a tabbed layout: Live, System, Storage, About.
    """
    sid = _safe(dash_id)
    key = _safe(ephemeral_key)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{_PS_STATS_CSS}</style>
</head>
<body>
  <div class="dash">
    <!-- Connection bar -->
    <div class="live-bar">
      <div class="live-indicator">
        <div class="live-dot" id="{sid}-dot"></div>
        <span class="live-label" id="{sid}-status">CONNECTING</span>
      </div>
      <div>
        <span class="live-meta" id="{sid}-uptime"></span>
      </div>
    </div>
    <div id="{sid}-expired" class="live-expired" style="display:none;">
      Dashboard session expired. Run <code>stats</code> again to start a new session.
    </div>
    <div id="{sid}-error" class="live-error" style="display:none;"></div>

    <!-- Tab bar -->
    <div class="tab-bar" id="{sid}-tabs">
      <button class="tab-btn active" data-tab="live">Live</button>
      <button class="tab-btn" data-tab="system">System</button>
      <button class="tab-btn" data-tab="storage">Storage</button>
      <button class="tab-btn" data-tab="about">About</button>
    </div>

    <!-- ═══ Tab: Live ═══ -->
    <div class="tab-pane active" id="{sid}-tab-live">
      <div class="g">
        <div class="gc"><span class="gc-l">Active requests</span><span class="gc-v" id="{sid}-req-val" style="color:#6366f1;">-</span></div>
        <div class="gc"><span class="gc-l">Active tools</span><span class="gc-v" id="{sid}-tool-val" style="color:#8b5cf6;">-</span></div>
        <div class="gc"><span class="gc-l">Sessions</span><span class="gc-v" id="{sid}-sess-val" style="color:#14b8a6;">-</span></div>
        <div class="gc"><span class="gc-l">Pending requests</span><span class="gc-v" id="{sid}-rq-val" style="color:#f97316;">-</span></div>
        <div class="gc"><span class="gc-l">Log queue</span><span class="gc-v" id="{sid}-lq-val">-</span></div>
        <div class="gc"><span class="gc-l">Archive queue</span><span class="gc-v" id="{sid}-aq-val">-</span></div>
        <div class="gc"><span class="gc-l">Uptime</span><span class="gc-v" id="{sid}-up-val">-</span></div>
      </div>

      <!-- User Circuit Breakers -->
      <div id="{sid}-rl-section" style="display:none;">
        <div class="section-h">User Circuit Breakers</div>
        <div id="{sid}-rl-panel"><div class="loading">Waiting for data...</div></div>
      </div>

      <!-- Models -->
      <div id="{sid}-models-section" style="display:none;">
        <div class="section-h">Models</div>
        <div id="{sid}-models-panel"><div class="loading">Waiting for data...</div></div>
      </div>
    </div>

    <!-- ═══ Tab: System ═══ -->
    <div class="tab-pane" id="{sid}-tab-system">
      <div id="{sid}-health-section" style="display:none;">
        <div class="section-h">Health</div>
        <div id="{sid}-health-panel"><div class="loading">Waiting for data...</div></div>
      </div>

      <div id="{sid}-config-section" style="display:none;">
        <div class="section-h">Configuration</div>
        <div id="{sid}-config-panel"><div class="loading">Waiting for data...</div></div>
      </div>

      <!-- Workers (multi-worker only) -->
      <div id="{sid}-workers-section" style="display:none;">
        <div class="section-h">Workers</div>
        <div id="{sid}-workers-panel"><div class="loading">Waiting for data...</div></div>
      </div>
    </div>

    <!-- ═══ Tab: Storage ═══ -->
    <div class="tab-pane" id="{sid}-tab-storage">
      <div id="{sid}-storage-section" style="display:none;">
        <div id="{sid}-storage-panel"><div class="loading">Waiting for data...</div></div>
      </div>
    </div>

    <!-- ═══ Tab: About ═══ -->
    <div class="tab-pane" id="{sid}-tab-about">
      <div id="{sid}-plugins-section" style="display:none;">
        <div class="section-h">Plugins</div>
        <div id="{sid}-plugins-panel"><div class="loading">Waiting for data...</div></div>
      </div>
    </div>

    <div class="footer" id="{sid}-footer"></div>
  </div>

  <script>
  (function() {{
    // ── Theme sync: mirror OWUI's .dark class from parent ──
    function syncTheme() {{
      try {{
        var isDark = parent.document.documentElement.classList.contains('dark');
        document.documentElement.classList.toggle('dark', isDark);
      }} catch (e) {{ /* cross-origin — fall back to light */ }}
    }}
    syncTheme();
    try {{
      new MutationObserver(syncTheme).observe(
        parent.document.documentElement,
        {{ attributes: true, attributeFilter: ['class'] }}
      );
    }} catch (e) {{ }}

    var ID = "{sid}";
    var KEY = "{key}";
    var state = {{}};

    function $(id) {{ return document.getElementById(id); }}
    function esc(s) {{ var d = document.createElement('div'); d.textContent = s; return d.innerHTML; }}

    // ── Tab switching ──
    var tabBar = $(ID + '-tabs');
    tabBar.addEventListener('click', function(e) {{
      var btn = e.target.closest('.tab-btn');
      if (!btn) return;
      var tab = btn.getAttribute('data-tab');
      // Deactivate all
      var btns = tabBar.querySelectorAll('.tab-btn');
      for (var i = 0; i < btns.length; i++) btns[i].classList.remove('active');
      var panes = document.querySelectorAll('.tab-pane');
      for (var j = 0; j < panes.length; j++) panes[j].classList.remove('active');
      // Activate selected
      btn.classList.add('active');
      var pane = $(ID + '-tab-' + tab);
      if (pane) pane.classList.add('active');
      reportHeight();
    }});

    // ── UI helpers ──
    function setLive() {{
      $(ID + '-status').textContent = 'LIVE';
      $(ID + '-status').style.color = '#22c55e';
    }}
    function setExpired() {{
      $(ID + '-expired').style.display = 'block';
      $(ID + '-dot').style.background = '#f59e0b';
      $(ID + '-dot').style.animation = 'none';
      $(ID + '-status').textContent = 'EXPIRED';
      $(ID + '-status').style.color = '#f59e0b';
    }}
    function setError(msg) {{
      $(ID + '-dot').style.background = '#ef4444';
      $(ID + '-dot').style.animation = 'none';
      $(ID + '-status').textContent = 'ERROR';
      $(ID + '-status').style.color = '#ef4444';
      $(ID + '-error').style.display = 'block';
      $(ID + '-error').textContent = msg;
    }}

    function fmtUptime(s) {{
      if (s < 60) return Math.round(s) + 's';
      if (s < 3600) return Math.floor(s/60) + 'm ' + Math.round(s%60) + 's';
      var h = Math.floor(s/3600);
      var m = Math.floor((s%3600)/60);
      return h + 'h ' + m + 'm';
    }}
    function badge(text, level) {{
      return '<span class="b b-' + level + '">' + esc(text) + '</span>';
    }}
    function gc(label, value, raw, rawLabel) {{
      var l = rawLabel ? label : esc(label);
      var v = raw ? value : esc(String(value));
      return '<div class="gc"><span class="gc-l">' + l + '</span><span class="gc-v">' + v + '</span></div>';
    }}
    function reportHeight() {{
      var h = document.documentElement.scrollHeight;
      parent.postMessage({{ type: 'iframe:height', height: h }}, '*');
    }}

    // ── Section updaters ──

    function updateIdentity(d) {{
      var wc = d.worker_count || 1;
      var workerText = wc > 1 ? 'Aggregated from ' + wc + ' workers' : 'Single worker';
      $(ID + '-footer').textContent = 'v' + (d.version || '?') + ' \u00b7 ' + (d.pipe_id || '?') + ' \u00b7 ' + workerText;
    }}

    function updateConcurrency(c) {{
      var ar = c.active_requests || 0, mr = c.max_requests || 0;
      $(ID + '-req-val').textContent = ar + ' / ' + mr;
      var at = c.active_tools || 0, mt = c.max_tools || 0;
      $(ID + '-tool-val').textContent = at + ' / ' + mt;
    }}

    function updateQueues(q) {{
      var rqs = q.requests || 0, rqm = q.requests_max || 1000;
      $(ID + '-rq-val').textContent = rqs + ' / ' + rqm;
      $(ID + '-lq-val').textContent = String(q.logs || 0);
      $(ID + '-aq-val').textContent = String(q.archive || 0);
    }}

    function updateSessions(s) {{
      $(ID + '-sess-val').textContent = (s.active || 0).toString();
    }}

    function updateRateLimits(rl) {{
      $(ID + '-rl-section').style.display = '';
      var tu = rl.tracked_users || 0, fu = rl.users_with_failures || 0;
      var tru = rl.tripped_users || 0, th = rl.threshold || 0;
      var ws = rl.window_s || 0;
      var tt = rl.tool_tracked || 0, tp = rl.tool_tripped || 0;
      var aa = rl.auth_failures_active || 0;
      var h = '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
        '<th>Type</th><th class="r">Tracked</th><th class="r">Failures</th>' +
        '<th class="r">Tripped</th><th class="r">Threshold</th><th class="r">Window</th>' +
        '</tr></thead><tbody>';
      var reqLevel = tru > 0 ? 'err' : (fu > 0 ? 'warn' : 'ok');
      h += '<tr><td>Requests</td><td class="r">' + tu + '</td>' +
        '<td class="r">' + fu + '</td>' +
        '<td class="r">' + badge(String(tru), reqLevel) + '</td>' +
        '<td class="r">' + th + '</td>' +
        '<td class="r">' + ws + 's</td></tr>';
      var toolLevel = tp > 0 ? 'err' : 'ok';
      h += '<tr><td>Tools</td><td class="r">' + tt + '</td>' +
        '<td class="r">-</td>' +
        '<td class="r">' + badge(String(tp), toolLevel) + '</td>' +
        '<td class="r">' + th + '</td>' +
        '<td class="r">' + ws + 's</td></tr>';
      var authLevel = aa > 0 ? 'warn' : 'ok';
      h += '<tr><td>Auth</td><td class="r">-</td><td class="r">-</td>' +
        '<td class="r">' + badge(String(aa), authLevel) + '</td>' +
        '<td class="r">-</td><td class="r">-</td></tr>';
      h += '</tbody></table></div>';
      $(ID + '-rl-panel').innerHTML = h;
    }}

    function updateModels(m) {{
      $(ID + '-models-section').style.display = '';
      var statusBadge;
      if (m.status === 'healthy') statusBadge = badge('Healthy', 'ok');
      else if (m.status === 'degraded') statusBadge = badge('Degraded', 'warn');
      else if (m.status === 'failing') statusBadge = badge('Failing', 'err');
      else if (m.status === 'pending') statusBadge = badge('Pending', 'info');
      else statusBadge = badge(m.status || 'Unknown', 'off');

      var h = '<div class="g">';
      h += gc('Models loaded', m.loaded || 0);
      h += gc('Specs cached', m.specs_cached || 0);
      h += gc('Last fetch', m.last_fetch_ago || 'never');
      h += gc('Failures', m.failures || 0);
      h += gc('Status', statusBadge, true);
      if (m.failures > 0 && m.last_error) {{
        h += gc('Last error', m.last_error);
      }}
      h += '</div>';
      $(ID + '-models-panel').innerHTML = h;
    }}

    function updateStorage(s) {{
      $(ID + '-storage-section').style.display = '';
      var h = '<div class="g">';
      h += gc('Database', s.connected ? badge('Connected', 'ok') : badge('Not connected', 'off'), true);
      if (s.table) h += gc('Table', s.table);
      if (s.total_items && s.total_items !== '-') h += gc('Total items', s.total_items);
      if (s.total_size && s.total_size !== '-') h += gc('Payload size', s.total_size);
      if (s.db_file_size && s.db_file_size !== '-') h += gc('DB file size', s.db_file_size);
      if (s.encrypted_count && s.encrypted_count !== '-') h += gc('Encrypted items', s.encrypted_count);
      h += gc('Encryption', badge(s.encryption_mode || 'Unknown', s.encryption_mode === 'Disabled' ? 'off' : 'ok'), true);
      var compLabel = (s.compression_mode || 'Disabled');
      if (s.compression_mode === 'LZ4') compLabel = badge('LZ4', 'ok') + ' \\u2265 ' + esc(s.compress_min_bytes || '0') + ' B';
      else compLabel = badge('Disabled', 'off');
      h += gc('Compression', compLabel, true);
      h += gc('Redis cache', s.redis_cache ? badge('Enabled', 'ok') : badge('Disabled', 'off'), true);
      h += '</div>';

      // By-type table
      if (s.by_type && s.by_type.length > 0) {{
        h += '<div class="sub-h">By Type</div>';
        h += '<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Type</th><th class="r">Count</th><th class="r">Size</th><th class="r">Oldest</th><th class="r">Newest</th></tr></thead><tbody>';
        for (var i = 0; i < s.by_type.length; i++) {{
          var t = s.by_type[i];
          h += '<tr><td class="name">' + esc(t.type) + '</td><td class="r">' + esc(t.count) + '</td><td class="r">' + esc(t.size) + '</td><td class="r">' + esc(t.oldest) + '</td><td class="r">' + esc(t.newest) + '</td></tr>';
        }}
        h += '</tbody></table></div>';
      }}

      // By-model table (scrollable)
      if (s.by_model && s.by_model.length > 0) {{
        h += '<div class="sub-h">By Model</div>';
        h += '<div class="scroll-table"><table class="tbl"><thead><tr><th>Model</th><th class="r">Items</th><th class="r">Size</th><th class="r">Chats</th><th class="r">Oldest</th><th class="r">Newest</th></tr></thead><tbody>';
        for (var j = 0; j < s.by_model.length; j++) {{
          var bm = s.by_model[j];
          h += '<tr><td class="name" title="' + esc(bm.model_id) + '">' + esc(bm.name) + '</td><td class="r">' + esc(bm.count) + '</td><td class="r">' + esc(bm.size) + '</td><td class="r">' + esc(bm.chats) + '</td><td class="r">' + esc(bm.oldest) + '</td><td class="r">' + esc(bm.newest) + '</td></tr>';
        }}
        h += '</tbody></table></div>';
      }}

      $(ID + '-storage-panel').innerHTML = h;
    }}

    function updateHealth(d) {{
      $(ID + '-health-section').style.display = '';
      var h = '<div class="g">';
      // Initialization status — _initialized is lazy (first request),
      // so "not initialized" on an idle worker just means no requests yet.
      if (d.initialized && d.startup_complete) {{
        h += gc('Status', badge('Ready', 'ok'), true);
      }} else if (d.warmup_failed) {{
        h += gc('Status', badge('Warmup failed', 'err'), true);
      }} else if (d.initialized) {{
        h += gc('Status', badge('Starting up', 'warn'), true);
      }} else {{
        h += gc('Status', badge('Idle', 'ok'), true);
      }}
      // HTTP session — created lazily on first request; "none" is normal for idle workers
      var httpBadge;
      if (d.http_session === 'active') httpBadge = badge('Active', 'ok');
      else if (d.http_session === 'closed') httpBadge = badge('Closed', 'err');
      else httpBadge = badge('Idle', 'off');
      h += gc('HTTP session', httpBadge, true);
      // Session logging
      if (d.logging_enabled) {{
        var logLabel = d.log_worker_alive ? badge('Active', 'ok') : badge('Worker stopped', 'warn');
        if (d.log_retention_days) logLabel += ' <span style="color:var(--text-faint);font-size:11px;">keep ' + esc(String(d.log_retention_days)) + 'd</span>';
        h += gc('Session logging', logLabel, true);
      }} else {{
        h += gc('Session logging', badge('Disabled', 'off'), true);
      }}
      // Redis
      if (d.redis_enabled) {{
        h += gc('Redis', d.redis_connected ? badge('Connected', 'ok') : badge('Disconnected', 'warn'), true);
      }} else {{
        h += gc('Redis', badge('Disabled', 'off'), true);
      }}
      h += '</div>';
      $(ID + '-health-panel').innerHTML = h;
    }}

    function updateConfig(c) {{
      if (!c || Object.keys(c).length === 0) {{
        $(ID + '-config-section').style.display = 'none';
        return;
      }}
      $(ID + '-config-section').style.display = '';
      var h = '<div class="g">';
      if (c.endpoint) h += gc('Default endpoint', c.endpoint);
      if (c.breaker) h += gc('Rate limit config', c.breaker);
      if (c.timing_log !== undefined) h += gc('Timing log', c.timing_log ? badge('Enabled', 'ok') : badge('Disabled', 'off'), true);
      if (c.artifact_cleanup) h += gc('Artifact cleanup', c.artifact_cleanup);
      if (c.log_retention) h += gc('Log retention', c.log_retention);
      if (c.redis_ttl) h += gc('Redis cache TTL', c.redis_ttl);
      if (c.stream_idle_flush) h += gc('Stream idle flush', c.stream_idle_flush);
      h += '</div>';
      $(ID + '-config-panel').innerHTML = h;
    }}

    function updateWorkers(workers) {{
      if (!workers || workers.length <= 1) {{
        $(ID + '-workers-section').style.display = 'none';
        return;
      }}
      $(ID + '-workers-section').style.display = '';
      var h = '<table class="tbl"><thead><tr>' +
        '<th>PID</th><th class="r">Uptime</th><th class="r">Status</th>' +
        '</tr></thead><tbody>';
      for (var i = 0; i < workers.length; i++) {{
        var w = workers[i];
        var up = fmtUptime(w.uptime_s || 0);
        var statusBadge = badge('Active', 'ok');
        h += '<tr><td class="name">' + esc(String(w.pid)) + '</td>' +
          '<td class="r">' + esc(up) + '</td>' +
          '<td class="r">' + statusBadge + '</td></tr>';
      }}
      h += '</tbody></table>';
      $(ID + '-workers-panel').innerHTML = h;
    }}

    function updatePlugins(list) {{
      $(ID + '-plugins-section').style.display = '';
      if (!list || list.length === 0) {{
        $(ID + '-plugins-panel').innerHTML = '<div class="loading">No plugins loaded</div>';
        return;
      }}
      var h = '<div class="g">';
      for (var i = 0; i < list.length; i++) {{
        var p = list[i];
        h += gc(esc(p.name) + ' <span style="color:var(--text-faint);font-size:11px;">' + esc(p.id) + '</span>', '<span style="color:#8b5cf6;">v' + esc(p.version) + '</span>', true, true);
      }}
      h += '</div>';
      $(ID + '-plugins-panel').innerHTML = h;
    }}

    // ── Master update dispatcher ──
    function updateDashboard(d) {{
      if (d.identity) {{ state.identity = d.identity; updateIdentity(d.identity); }}
      if (d.concurrency) updateConcurrency(d.concurrency);
      if (d.queues) updateQueues(d.queues);
      if (d.rate_limits) updateRateLimits(d.rate_limits);
      if (d.sessions) updateSessions(d.sessions);
      if (d.uptime_s !== undefined) {{
        $(ID + '-up-val').textContent = fmtUptime(d.uptime_s);
        $(ID + '-uptime').textContent = fmtUptime(d.uptime_s);
      }}
      if (d.models) {{ state.models = d.models; updateModels(d.models); }}
      if (d.health) updateHealth(d.health);
      if (d.storage) {{ state.storage = d.storage; updateStorage(d.storage); }}
      if (d.config) updateConfig(d.config);
      if (d.workers) {{ state.workers = d.workers; updateWorkers(d.workers); }}
      if (d.plugins) {{ state.plugins = d.plugins; updatePlugins(d.plugins); }}
      // Update worker_count from top-level field (sent every 4th tick in Redis mode)
      if (d.worker_count !== undefined && state.identity) {{
        state.identity.worker_count = d.worker_count;
        updateIdentity(state.identity);
      }}
      reportHeight();
    }}

    // ── SSE connection ──
    var es = new EventSource('/api/pipe/stats/' + KEY);
    var gotData = false;

    es.onmessage = function(e) {{
      try {{
        var data = JSON.parse(e.data);
        if (data.status === 'expired') {{
          es.close();
          setExpired();
          return;
        }}
        if (data.error) return;
        if (!gotData) {{
          gotData = true;
          setLive();
        }}
        $(ID + '-error').style.display = 'none';
        updateDashboard(data);
      }} catch (err) {{ }}
    }};

    es.onerror = function() {{
      if (!gotData) {{
        es.close();
        setError('Live updates unavailable (SSE endpoint not reachable)');
      }}
    }};

    // Initial height report
    reportHeight();
    new MutationObserver(reportHeight).observe(document.body, {{ childList: true, subtree: true }});
    window.addEventListener('resize', reportHeight);
  }})();
  </script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

@register_command("stats", summary="Show pipe statistics dashboard", category="Diagnostics", usage="stats")
async def handle_stats(ctx: CommandContext) -> str:
    """Display a fully dynamic HTML dashboard powered by SSE."""
    dash_id = "dash-" + secrets.token_hex(4)
    ephemeral_key = _get_ephemeral_key(ctx.pipe)
    if not ephemeral_key:
        return "Dashboard requires the pipe-stats plugin with ephemeral key support."
    dashboard_html = _build_dashboard_shell(dash_id, ephemeral_key)
    await ctx.emit_html(dashboard_html)
    return ""
