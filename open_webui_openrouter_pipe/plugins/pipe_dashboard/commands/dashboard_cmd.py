"""Dashboard command — fully dynamic HTML dashboard over OWUI socket.io.

The ``dashboard`` command emits an HTML shell with empty containers and
JavaScript that opens an authenticated socket.io connection to OWUI,
subscribes to the viewers room, and renders the tiered JSON
payloads the publisher emits.

The dashboard uses a tabbed layout: Live, Usage, Health, System,
Storage, About.
"""

from __future__ import annotations

import html
import secrets
from typing import Any

from .._socketio_client import SOCKETIO_UMD
from ..command_registry import register_command
from ..config_tab_assets import CONFIG_TAB_CSS, CONFIG_TAB_JS
from ..update_tab_assets import UPDATE_TAB_CSS, UPDATE_TAB_JS
from ..context import CommandContext
from ..dashboard_socket import CONFIG_EVENT, DENIED_EVENT, DASHBOARD_EVENT, SUB_EVENT, register_socket_handler


def _safe(val: Any) -> str:
    """HTML-escape a value for safe embedding."""
    return html.escape(str(val))


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_PD_CSS = """\
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
.dash.stale { opacity: 0.55; }
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
.live-btn { padding: 3px 10px; font-size: 11px; font-weight: 600; color: var(--text-muted);
  background: var(--tab-bg); border: 1px solid var(--border); border-radius: 6px;
  cursor: pointer; font-family: inherit; margin-left: 8px; }
.live-btn:hover { color: var(--text-dim); border-color: var(--border-light); }

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
.tbl tfoot .tot-row td { color: #6366f1; font-weight: 700; border-top: 2px solid var(--border); background: rgba(99, 102, 241, 0.06); }
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

/* Live sessions */
.b-live { background: rgba(34,197,94,0.15); color: #22c55e; }
.b-live::before { content: '\\25CF '; animation: pulse-dot 1.5s ease-in-out infinite; }
.b-tool { background: rgba(139,92,246,0.15); color: #8b5cf6; }
.tbl tr.done td { opacity: 0.55; }
.run-cost { color: #6366f1; }
.run-cost::after { content: ' \\2197'; font-size: 10px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; flex-shrink: 0; }
.seg button { padding: 3px 10px; font-size: 11px; font-weight: 600; color: var(--text-muted);
  background: var(--tab-bg); border: none; cursor: pointer; font-family: inherit; white-space: nowrap; }
.seg button.active { color: var(--tab-active-text); background: var(--tab-active-bg); }
.keep-select { font-size: 11px; font-weight: 600; color: var(--text-muted); background: var(--tab-bg);
  border: 1px solid var(--border); border-radius: 6px; padding: 3px 6px; font-family: inherit; }
.tbl-tools { margin-bottom: 6px; }
.tbl-search { width: 100%; max-width: 260px; font-size: 12px; padding: 4px 8px;
  border: 1px solid var(--border); border-radius: 6px; background: var(--tab-bg);
  color: var(--text); font-family: inherit; }
.th-sort { cursor: pointer; user-select: none; white-space: nowrap; }
.th-sort:hover { color: var(--text); }
.caret { font-size: 9px; }
.sess-controls { display: flex; align-items: center; gap: 14px; font-weight: 600; font-size: 11px;
  color: var(--text-muted); text-transform: none; letter-spacing: 0; }

/* Usage cards */
.us-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; margin-bottom: 12px; }
.us-card { border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
.us-card-h { font-size: 10px; font-weight: 700; color: var(--text-faint); letter-spacing: 0.08em;
  text-transform: uppercase; display: flex; justify-content: space-between; align-items: center;
  gap: 6px; margin-bottom: 4px; }
.us-chip { font-size: 10px; font-weight: 700; padding: 1px 6px; border-radius: 5px;
  background: rgba(99,102,241,0.12); color: var(--tab-active-text); }
.us-card-row { display: flex; justify-content: space-between; align-items: flex-end; gap: 8px; }
.us-card-v { font-size: 20px; font-weight: 800; font-family: 'JetBrains Mono', monospace;
  letter-spacing: -0.02em; color: var(--text); }
.us-card-sub { font-size: 10px; color: var(--text-faint); margin-top: 2px; }
/* Compact second row — smaller padding + value font */
.us-cards.us-sm .us-card { padding: 8px 11px; }
.us-cards.us-sm .us-card-h { margin-bottom: 3px; }
.us-cards.us-sm .us-card-v { font-size: 17px; }
.us-meta { font-size: 11px; color: var(--text-faint); }

/* Expired / Error states */
.live-expired { font-size: 13px; color: #f59e0b; text-align: center;
  padding: 10px; background: var(--expired-bg); border: 1px solid var(--expired-border);
  border-radius: 8px; margin-bottom: 12px; }
.live-error { font-size: 12px; color: #ef4444; text-align: center;
  padding: 6px; margin-bottom: 8px; }

/* Footer */
.footer-legend { font-size: 11px; color: var(--text-faint);
  margin-top: 14px; padding-top: 8px; border-top: 1px solid var(--border); }
.footer-legend b { color: var(--text-dim); font-weight: 600; }
.footer { font-size: 11px; color: var(--text-faint); margin-top: 5px; }

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

def _build_dashboard_shell(dash_id: str) -> str:
    """Build the fully dynamic dashboard HTML shell.

    All containers are empty — JavaScript populates them from socket.io
    stats events.  Uses a tabbed layout: Live, Usage, Health, System, Storage, About.
    """
    sid = _safe(dash_id)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script>{SOCKETIO_UMD}</script>
<style>{_PD_CSS}{CONFIG_TAB_CSS}{UPDATE_TAB_CSS}</style>
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
        <button class="live-btn" id="{sid}-btn-disconnect">Disconnect</button>
        <button class="live-btn" id="{sid}-btn-connect" style="display:none;">Connect</button>
      </div>
    </div>
    <div id="{sid}-notice" class="live-expired" style="display:none;"></div>
    <div id="{sid}-degraded" class="live-expired" style="display:none;">Aggregation degraded — showing last known workers.</div>
    <div id="{sid}-error" class="live-error" style="display:none;"></div>

    <!-- Tab bar -->
    <div class="tab-bar" id="{sid}-tabs">
      <button class="tab-btn active" data-tab="live">Live</button>
      <button class="tab-btn" data-tab="usage">Usage</button>
      <button class="tab-btn" data-tab="health">Health</button>
      <button class="tab-btn" data-tab="system">System</button>
      <button class="tab-btn" data-tab="storage">Storage</button>
      <button class="tab-btn" data-tab="config">Config</button>
      <button class="tab-btn" data-tab="update">Update</button>
      <button class="tab-btn" data-tab="about">About</button>
    </div>

    <!-- ═══ Tab: Live (sessions) ═══ -->
    <div class="tab-pane active" id="{sid}-tab-live">
      <div class="g" style="grid-template-columns:repeat(auto-fit,minmax(160px,1fr));">
        <div class="gc"><span class="gc-l">Active</span><span class="gc-v" id="{sid}-ls-active" style="color:#22c55e;">-</span></div>
        <div class="gc"><span class="gc-l" title="sessions that finished in the window (completed, failed, or cancelled)">Done</span><span class="gc-v" id="{sid}-ls-done">-</span></div>
        <div class="gc"><span class="gc-l">Cost</span><span class="gc-v" id="{sid}-ls-cost" style="color:#6366f1;">-</span></div>
        <div class="gc"><span class="gc-l">Tokens</span><span class="gc-v" id="{sid}-ls-tokens">-</span></div>
      </div>
      <div class="section-h" style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
        <span>Sessions</span>
        <span class="sess-controls">
          <span>Model: <span class="seg" id="{sid}-nameseg"><button type="button" class="active" data-mode="names">Names</button><button type="button" data-mode="slugs">Slugs</button></span></span>
          <span>Keep completed: <select class="keep-select" id="{sid}-keep"><option value="5">5 min</option><option value="10" selected>10 min</option><option value="15">15 min</option><option value="20">20 min</option><option value="25">25 min</option><option value="30">30 min</option><option value="60">1 hour</option><option value="120">2 hours</option><option value="180">3 hours</option></select></span>
        </span>
      </div>
      <div id="{sid}-sessions-panel"><div class="loading">Waiting for sessions...</div></div>
    </div>

    <!-- ═══ Tab: Usage (usage analytics) ═══ -->
    <div class="tab-pane" id="{sid}-tab-usage">
      <div class="section-h" style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
        <span>Usage — <span id="{sid}-us-range-label" style="text-transform:none;">last 24 hours</span></span>
        <span class="sess-controls">
          <span class="seg" id="{sid}-us-range">
            <button type="button" data-range="1h">1h</button><button type="button" data-range="6h">6h</button><button type="button" class="active" data-range="24h">24h</button><button type="button" data-range="7d">7d</button><button type="button" data-range="30d">30d</button>
          </span>
          <label style="display:flex;align-items:center;gap:5px;cursor:pointer;"><input type="checkbox" id="{sid}-us-tasks" checked> Incl. task requests</label>
          <span>Auto-refresh: <select class="keep-select" id="{sid}-us-refresh"><option value="1">1 min</option><option value="5">5 min</option><option value="15" selected>15 min</option><option value="30">30 min</option></select></span>
          <span class="us-meta" id="{sid}-us-meta"></span>
        </span>
      </div>
      <div id="{sid}-us-note" class="loading">Open this tab to load usage data.</div>
      <div id="{sid}-us-body" style="display:none;">
        <div id="{sid}-us-cards"></div>
        <div class="us-cards us-sm" id="{sid}-us-system"></div>
        <div class="section-h" style="display:flex;align-items:center;justify-content:space-between;gap:12px;">
          <span>Usage trend <span id="{sid}-us-legend" style="font-weight:400;text-transform:none;letter-spacing:0;color:var(--text-faint);font-size:11px;"></span></span>
          <span style="font-weight:400;text-transform:none;letter-spacing:0;font-size:11px;color:var(--text-dim);white-space:nowrap;"><span style="color:#14b8a6;">&#9679;</span> Tokens&nbsp;&nbsp;<span style="color:#f59e0b;">&#9679;</span> Cost</span>
        </div>
        <div id="{sid}-us-chart"></div>
        <div class="section-h">By model <span style="font-weight:400;color:var(--text-faint);text-transform:none;letter-spacing:0;font-size:11px;">share = share of cost</span></div>
        <div id="{sid}-us-models"></div>
        <div class="section-h">By user — top spenders</div>
        <div id="{sid}-us-users"></div>
        <div id="{sid}-us-totals"></div>
      </div>
    </div>

    <!-- ═══ Tab: Health ═══ -->
    <div class="tab-pane" id="{sid}-tab-health">
      <div class="g">
        <div class="gc"><span class="gc-l">Active requests</span><span class="gc-v" id="{sid}-req-val" style="color:#6366f1;">-</span></div>
        <div class="gc"><span class="gc-l">Active tools</span><span class="gc-v" id="{sid}-tool-val" style="color:#8b5cf6;">-</span></div>
        <div class="gc"><span class="gc-l">In-flight calls</span><span class="gc-v" id="{sid}-sess-val" style="color:#14b8a6;">-</span></div>
        <div class="gc"><span class="gc-l" title="semaphore waiters + queued jobs">Pending requests</span><span class="gc-v" id="{sid}-rq-val" style="color:#f97316;">-</span></div>
        <div class="gc" id="{sid}-video-row" style="display:none;"><span class="gc-l">Active video gens</span><span class="gc-v" id="{sid}-video-val" style="color:#ec4899;">-</span></div>
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
      <div class="loading" id="{sid}-load-system">Waiting for data...</div>
      <div id="{sid}-health-section" style="display:none;">
        <div class="section-h">Readiness</div>
        <div id="{sid}-health-panel"><div class="loading">Waiting for data...</div></div>
      </div>

      <div id="{sid}-config-section" style="display:none;">
        <div class="section-h">Configuration</div>
        <div id="{sid}-config-panel"><div class="loading">Waiting for data...</div></div>
      </div>

      <div id="{sid}-db-section" style="display:none;">
        <div class="section-h">Artifact DB</div>
        <div id="{sid}-db-panel"><div class="loading">Waiting for data...</div></div>
      </div>

      <!-- Workers (multi-worker only) -->
      <div id="{sid}-workers-section" style="display:none;">
        <div class="section-h">Workers</div>
        <div id="{sid}-workers-panel"><div class="loading">Waiting for data...</div></div>
      </div>
    </div>

    <!-- ═══ Tab: Storage ═══ -->
    <div class="tab-pane" id="{sid}-tab-storage">
      <div class="loading" id="{sid}-load-storage">Waiting for data...</div>
      <div id="{sid}-storage-section" style="display:none;">
        <div id="{sid}-storage-panel"><div class="loading">Waiting for data...</div></div>
      </div>
    </div>

    <!-- ═══ Tab: Config ═══ -->
    <div class="tab-pane" id="{sid}-tab-config">
      <div class="cfgroot" id="{sid}-cfgroot">
        <div class="topbar">
          <div class="searchwrap"><input id="search" placeholder="Search settings" autocomplete="off"><span id="searchcount" class="searchcount"></span></div>
          <div class="toolbar">
            <span id="driftnote" class="driftnote"></span>
            <div class="savebar" id="savebar"><span class="dirtybadge"><span id="dirtyN">0</span> unsaved</span>
              <button id="discard" class="btn ghost">Discard</button><button id="save" class="btn primary" disabled>Save</button></div>
            <label class="chk"><input type="checkbox" id="chgToggle"> Changed only</label>
            <button id="expandAll" class="btn">Expand all</button><button id="collapseAll" class="btn">Collapse all</button>
          </div>
        </div>
        <div id="conflict" class="conflict" style="display:none;"></div>
        <div class="body"><nav id="tree" class="tree"><div class="empty" style="margin-top:60px">Open this tab to load configuration.</div></nav><main id="detail" class="detail"><div class="empty">Select a setting to view and edit it.</div></main></div>
        <div class="modal" id="modal"></div><div class="toast" id="toast"></div>
      </div>
    </div>

    <!-- ═══ Tab: Update ═══ -->
    <div class="tab-pane" id="{sid}-tab-update">
      <div class="upd-wrap">
        <div id="upd-msg" class="upd-msg"></div>
        <div class="upd-grid">
          <div class="upd-card" id="upd-installed">
            <h3>Installed</h3>
            <div id="upd-installed-body">Loading...</div>
          </div>
          <div class="upd-card" id="upd-latest">
            <h3>Latest release</h3>
            <div id="upd-latest-body">Loading...</div>
            <div class="upd-actions">
              <button class="upd-btn" id="upd-check-now">Check now</button>
              <button class="upd-btn primary" id="upd-apply-btn" style="display:none">Update...</button>
            </div>
          </div>
        </div>
        <details id="upd-notes"><summary id="upd-notes-summary">Changelog</summary><pre id="upd-notes-body"></pre></details>
        <div class="upd-card">
          <h3>Previous versions</h3>
          <div id="upd-snapshots">No snapshots yet.</div>
        </div>
        <div id="upd-modal" class="upd-modal" style="display:none">
          <div class="upd-modal-box">
            <h3>Apply update</h3>
            <div id="upd-modal-body"></div>
            <label class="upd-check"><input type="checkbox" id="upd-compressed"> Compressed bundle</label>
            <div id="upd-modal-note" class="upd-note"></div>
            <div class="upd-actions">
              <button class="upd-btn" id="upd-modal-cancel">Cancel</button>
              <button class="upd-btn primary" id="upd-modal-confirm">Install</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ═══ Tab: About ═══ -->
    <div class="tab-pane" id="{sid}-tab-about">
      <div class="loading" id="{sid}-load-about">Waiting for data...</div>
      <div id="{sid}-plugins-section" style="display:none;">
        <div class="section-h">Plugins</div>
        <div id="{sid}-plugins-panel"><div class="loading">Waiting for data...</div></div>
      </div>
    </div>

    <div class="footer-legend">
      <b>+t</b> cost incl. folded task-model requests &middot; <b>&#10003; / &#10007;</b> tools succeeded / failed
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
    var state = {{}};

    function $(id) {{ return document.getElementById(id); }}
    function esc(s) {{ var d = document.createElement('div'); d.textContent = s; return d.innerHTML.replace(/"/g, '&quot;').replace(/'/g, '&#39;'); }}

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
      if (tab === 'usage' && !usLoaded) {{ usLoaded = true; usFetch(); }}
      if (tab === 'config' && !cfgLoaded) {{ cfgLoaded = true; cfgFetch(); }}
      if (tab === 'update') {{ if (!updLoaded) {{ updLoaded = true; updFetch(); }} else {{ updEnterTab(); }} }}
      reportHeight();
    }});

    // ── UI helpers ──
    var dashEl = document.querySelector('.dash');
    function setLive() {{
      $(ID + '-dot').style.background = '#22c55e';
      $(ID + '-dot').style.animation = '';
      $(ID + '-status').textContent = 'LIVE';
      $(ID + '-status').style.color = '#22c55e';
      $(ID + '-notice').style.display = 'none';
      $(ID + '-btn-disconnect').style.display = '';
      $(ID + '-btn-connect').style.display = 'none';
      if (dashEl) dashEl.classList.remove('stale');
    }}
    function setDisconnected() {{
      $(ID + '-dot').style.background = '#f59e0b';
      $(ID + '-dot').style.animation = 'none';
      $(ID + '-status').textContent = 'DISCONNECTED';
      $(ID + '-status').style.color = '#f59e0b';
      $(ID + '-btn-disconnect').style.display = 'none';
      $(ID + '-btn-connect').style.display = '';
      if (dashEl) dashEl.classList.add('stale');
    }}
    function setReconnecting() {{
      $(ID + '-dot').style.background = '#f59e0b';
      $(ID + '-status').textContent = 'RECONNECTING';
      $(ID + '-status').style.color = '#f59e0b';
      if (dashEl) dashEl.classList.add('stale');
    }}
    function setStatic(msg) {{
      $(ID + '-dot').style.background = '#64748b';
      $(ID + '-dot').style.animation = 'none';
      $(ID + '-status').textContent = 'STATIC';
      $(ID + '-status').style.color = '#64748b';
      $(ID + '-notice').style.display = 'block';
      $(ID + '-notice').textContent = msg;
      $(ID + '-btn-disconnect').style.display = 'none';
      $(ID + '-btn-connect').style.display = 'none';
      if (dashEl) dashEl.classList.add('stale');
    }}
    function setError(msg) {{
      $(ID + '-dot').style.background = '#ef4444';
      $(ID + '-dot').style.animation = 'none';
      $(ID + '-status').textContent = 'ERROR';
      $(ID + '-status').style.color = '#ef4444';
      $(ID + '-error').style.display = 'block';
      $(ID + '-error').textContent = msg;
      if (dashEl) dashEl.classList.add('stale');
    }}

    function fmtUptime(s) {{
      var t = Math.floor(s);
      if (t < 60) return t + 's';
      var m = Math.floor(t / 60);
      if (t < 3600) return m + 'm ' + (t % 60) + 's';
      var h = Math.floor(t / 3600);
      m = Math.floor((t % 3600) / 60);
      return h + 'h ' + m + 'm';
    }}
    function hideLoad(tab) {{
      var el = $(ID + '-load-' + tab);
      if (el) el.style.display = 'none';
    }}
    function pending(v) {{
      if (v === undefined || v === null || v === '-') {{
        return '<span title="no data yet" style="color:var(--text-faint);">\\u2026</span>';
      }}
      return esc(String(v));
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

    // ── Live sessions ──
    var nameMode = 'names';
    var keepMinutes = 10;

    function fmtCost(v) {{
      v = v || 0;
      return '$' + v.toFixed(3);
    }}
    function fmtTok(n) {{
      n = n || 0;
      if (n >= 999500) return (n / 1000000).toFixed(1) + 'M';
      if (n >= 1000) return (n / 1000).toFixed(1) + 'k';
      return String(Math.round(n));
    }}
    function tok3(tin, tcached, tout) {{
      return esc(fmtTok(tin)) + ' \\u2192 ' + esc(fmtTok(tcached)) + ' \\u2192 ' + esc(fmtTok(tout));
    }}
    function toolsCell(total, failed) {{
      total = total || 0; failed = failed || 0;
      if (!total) return '\\u2014';
      var ok = total - failed; if (ok < 0) ok = 0;
      var s = esc(String(ok)) + ' \\u2713';
      if (failed) s += ' <span style="color:#ef4444;">' + esc(String(failed)) + ' \\u2717</span>';
      return s;
    }}
    function sessBadge(row) {{
      var st = row.status || 'queued';
      if (st === 'streaming') return badge('Streaming', 'live');
      if (st === 'queued') return badge('Queued', 'warn');
      if (st.indexOf('tool:') === 0) return badge('Tool: ' + st.slice(5), 'tool');
      if (st === 'completed') return badge('Completed', 'ok');
      if (st === 'failed') return badge('Failed', 'err');
      if (st === 'cancelled') return badge('Cancelled', 'off');
      return badge(st, 'off');
    }}

    function updateLiveSessions(rows) {{
      state.sessionsLive = Array.isArray(rows) ? rows : [];
      renderLiveSessions();
    }}

    function makeSortTable(mountId, opts) {{
      var st = {{ key: null, dir: -1, q: '' }};
      var lastRows = [];
      function headHtml() {{
        var out = '';
        for (var i = 0; i < opts.cols.length; i++) {{
          var c = opts.cols[i];
          out += '<th class="' + (c.cls || '') + ' th-sort" data-k="' + c.k + '"' +
            (c.title ? ' title="' + esc(c.title) + '"' : '') +
            (c.style ? ' style="' + c.style + '"' : '') +
            '>' + esc(c.label) + '<span class="caret"></span></th>';
        }}
        return out;
      }}
      function sortView() {{
        var q = st.q.trim().toLowerCase();
        var view = q ? lastRows.filter(function(r) {{ return opts.match(r, q); }}) : lastRows.slice();
        if (st.key) {{
          var col = null;
          for (var i = 0; i < opts.cols.length; i++) if (opts.cols[i].k === st.key) col = opts.cols[i];
          if (col) view.sort(function(a, b) {{
            var va = col.get(a), vb = col.get(b);
            if (col.num) return ((+va || 0) - (+vb || 0)) * st.dir;
            va = String(va).toLowerCase(); vb = String(vb).toLowerCase();
            return (va < vb ? -1 : va > vb ? 1 : 0) * st.dir;
          }});
        }}
        if (!q && opts.cap && opts.aggregate && view.length > opts.cap) {{
          var capped = view.slice(0, opts.cap);
          capped.push(opts.aggregate(view.slice(opts.cap)));
          view = capped;
        }}
        return view;
      }}
      function paint() {{
        var mount = $(mountId); if (!mount) return;
        var tb = mount.querySelector('tbody'); if (!tb) return;
        var view = sortView(), body = '';
        for (var i = 0; i < view.length; i++) body += opts.row(view[i]);
        tb.innerHTML = body || '<tr><td colspan="' + opts.cols.length + '" class="loading" style="text-align:center;padding:14px;">' + esc(st.q ? 'No matches.' : (opts.empty || 'No data.')) + '</td></tr>';
        var tf = mount.querySelector('tfoot');
        if (tf) tf.innerHTML = opts.footerRow ? opts.footerRow(view) : '';
        var ths = mount.querySelectorAll('th[data-k]');
        for (var j = 0; j < ths.length; j++) {{
          var k = ths[j].getAttribute('data-k');
          ths[j].querySelector('.caret').textContent = st.key === k ? (st.dir > 0 ? ' \\u25b2' : ' \\u25bc') : '';
        }}
      }}
      return {{
        update: function(rows) {{
          lastRows = Array.isArray(rows) ? rows : [];
          var mount = $(mountId); if (!mount) return;
          if (!mount.querySelector('tbody')) {{
            mount.innerHTML =
              '<div class="tbl-tools"><input type="search" class="tbl-search" placeholder="Filter\\u2026" aria-label="Filter rows"></div>' +
              '<div class="tbl-wrap"><table class="tbl"><thead><tr>' + headHtml() + '</tr></thead><tbody></tbody>' + (opts.footerRow ? '<tfoot></tfoot>' : '') + '</table></div>';
            var search = mount.querySelector('.tbl-search');
            search.value = st.q;
            search.addEventListener('input', function() {{ st.q = search.value; paint(); }});
            mount.querySelector('thead').addEventListener('click', function(e) {{
              var th = e.target.closest('th[data-k]'); if (!th) return;
              var k = th.getAttribute('data-k'), col = null;
              for (var i = 0; i < opts.cols.length; i++) if (opts.cols[i].k === k) col = opts.cols[i];
              if (st.key === k) st.dir = -st.dir;
              else {{ st.key = k; st.dir = (col && col.num) ? -1 : 1; }}
              paint();
            }});
          }}
          paint();
        }}
      }};
    }}

    function liveRowHtml(r2) {{
      var now = Date.now() / 1000;
      var modelText = nameMode === 'slugs' ? (r2.model_id || '') : (r2.model_name || r2.model_id || '');
      var elapsed = r2.done ? (r2.done - r2.started) : (now - (r2.started || now));
      var tools = (r2.tools_ok || r2.tools_failed)
        ? (r2.tools_ok || 0) + ' \\u2713' + (r2.tools_failed ? ' <span style="color:#ef4444;">' + (r2.tools_failed || 0) + ' \\u2717</span>' : '')
        : '\\u2013';
      var costCell = r2.done
        ? esc(fmtCost(r2.cost)) + (r2.task_cost ? ' <span style="color:var(--text-faint);font-size:10px;" title="includes task-model cost">+t</span>' : '')
        : '<span class="run-cost">' + esc(fmtCost(r2.cost)) + '</span>';
      var agoMin = r2.done ? Math.round((now - r2.done) / 60) : 0;
      var ago = r2.done
        ? '<span style="color:var(--text-faint);font-size:11px;margin-left:6px;">' + esc(agoMin < 1 ? 'now' : agoMin + 'm ago') + '</span>'
        : '';
      return '<tr' + (r2.done ? ' class="done"' : '') + '>' +
        '<td class="name">' + esc(r2.user || '?') + '</td>' +
        '<td class="name" title="' + esc(r2.model_id || '') + '">' + esc(modelText) + '</td>' +
        '<td>' + sessBadge(r2) + ago + '</td>' +
        '<td class="r">' + esc(fmtUptime(elapsed)) + '</td>' +
        '<td class="r">' + tools + '</td>' +
        '<td class="r">' + tok3(r2.tokens_in, r2.tokens_cached, r2.tokens_out) + '</td>' +
        '<td class="r">' + costCell + '</td>' +
        '<td class="r">' + esc(String(r2.worker_pid || '-')) + '</td></tr>';
    }}
    function modelRowHtml(r) {{
      var name = nameMode === 'slugs' ? r.model_id : (r.model_name || r.model_id);
      return '<tr><td class="name" title="' + esc(r.model_id) + '">' + esc(name) + '</td>' +
        '<td class="r">' + esc(String(r.sessions)) + '</td>' +
        '<td class="r">' + tok3(r.tokens_in, r.tokens_cached, r.tokens_out) + '</td>' +
        '<td class="r">' + toolsCell(r.tools, r.tools_failed) + '</td>' +
        '<td class="r">' + esc(fmtCost(r.cost)) + '</td>' +
        '<td class="r">' + esc(fmtCost(r.avg_cost)) + '</td>' +
        '<td><div style="height:6px;border-radius:3px;background:rgba(99,102,241,0.18);position:relative;min-width:90px;"><i style="position:absolute;left:0;top:0;bottom:0;border-radius:3px;background:#6366f1;width:' + Math.min(100, r.share_pct || 0) + '%;"></i></div></td></tr>';
    }}
    function userRowHtml(r) {{
      var now = Date.now() / 1000;
      var last = r.last_active ? fmtUptime(Math.max(0, now - r.last_active)) + ' ago' : '\\u2014';
      return '<tr><td class="name">' + esc(r.user_name || '?') + '</td>' +
        '<td class="r">' + esc(String(r.sessions)) + '</td>' +
        '<td class="r">' + tok3(r.tokens_in, r.tokens_cached, r.tokens_out) + '</td>' +
        '<td class="r">' + toolsCell(r.tools, r.tools_failed) + '</td>' +
        '<td class="r">' + esc(fmtCost(r.cost)) + '</td>' +
        '<td class="r">' + esc(last) + '</td></tr>';
    }}

    var liveTable = makeSortTable(ID + '-sessions-panel', {{
      empty: 'No sessions in the window.', row: liveRowHtml,
      match: function(r, q) {{ return (String(r.user || '') + ' ' + String(r.model_name || r.model_id || '') + ' ' + String(r.status || '')).toLowerCase().indexOf(q) >= 0; }},
      cols: [
        {{ k: 'user', label: 'User', get: function(r) {{ return r.user || ''; }} }},
        {{ k: 'model', label: 'Model', get: function(r) {{ return r.model_name || r.model_id || ''; }} }},
        {{ k: 'status', label: 'Status', get: function(r) {{ return r.status || ''; }} }},
        {{ k: 'elapsed', label: 'Elapsed', cls: 'r', num: true, get: function(r) {{ return r.elapsed_s || 0; }} }},
        {{ k: 'tools', label: 'Tools', cls: 'r', num: true, get: function(r) {{ return (r.tools_ok || 0) + (r.tools_failed || 0); }} }},
        {{ k: 'tokens', label: 'Tokens in \\u2192 cached \\u2192 out', cls: 'r', num: true, title: 'cached input tokens shown in the middle', get: function(r) {{ return (r.tokens_in || 0) + (r.tokens_out || 0); }} }},
        {{ k: 'cost', label: 'Cost', cls: 'r', num: true, get: function(r) {{ return r.cost || 0; }} }},
        {{ k: 'worker', label: 'Worker', cls: 'r', num: true, get: function(r) {{ return r.worker_pid || 0; }} }}
      ]
    }});
    var modelsTable = makeSortTable(ID + '-us-models', {{
      empty: 'No model usage in range.', row: modelRowHtml,
      match: function(r, q) {{ return (String(r.model_name || '') + ' ' + String(r.model_id || '')).toLowerCase().indexOf(q) >= 0; }},
      cols: [
        {{ k: 'model', label: 'Model', get: function(r) {{ return r.model_name || r.model_id || ''; }} }},
        {{ k: 'sessions', label: 'Sessions', cls: 'r', num: true, get: function(r) {{ return r.sessions || 0; }} }},
        {{ k: 'tokens', label: 'Tokens in \\u2192 cached \\u2192 out', cls: 'r', num: true, title: 'cached input tokens shown in the middle', get: function(r) {{ return (r.tokens_in || 0) + (r.tokens_out || 0); }} }},
        {{ k: 'tools', label: 'Tools', cls: 'r', num: true, get: function(r) {{ return r.tools || 0; }} }},
        {{ k: 'cost', label: 'Cost', cls: 'r', num: true, get: function(r) {{ return r.cost || 0; }} }},
        {{ k: 'avg', label: 'Avg $/sess', cls: 'r', num: true, get: function(r) {{ return r.avg_cost || 0; }} }},
        {{ k: 'share', label: 'Share', num: true, style: 'width:110px;', get: function(r) {{ return r.share_pct || 0; }} }}
      ]
    }});
    var usersTable = makeSortTable(ID + '-us-users', {{
      empty: 'No user activity in range.', row: userRowHtml,
      match: function(r, q) {{ return String(r.user_name || '').toLowerCase().indexOf(q) >= 0; }},
      cap: 10,
      aggregate: function(rest) {{
        var o = {{ user_name: rest.length + ' others', sessions: 0, tokens_in: 0, tokens_cached: 0, tokens_out: 0, tools: 0, tools_failed: 0, cost: 0, last_active: 0 }};
        for (var i = 0; i < rest.length; i++) {{
          var r = rest[i];
          o.sessions += r.sessions || 0; o.tokens_in += r.tokens_in || 0; o.tokens_cached += r.tokens_cached || 0;
          o.tokens_out += r.tokens_out || 0; o.tools += r.tools || 0; o.tools_failed += r.tools_failed || 0; o.cost += r.cost || 0;
        }}
        return o;
      }},
      footerRow: function(rows) {{
        var s = 0, ti = 0, tc = 0, to = 0, tl = 0, tlf = 0, co = 0;
        for (var i = 0; i < rows.length; i++) {{
          var r = rows[i];
          s += r.sessions || 0; ti += r.tokens_in || 0; tc += r.tokens_cached || 0;
          to += r.tokens_out || 0; tl += r.tools || 0; tlf += r.tools_failed || 0; co += r.cost || 0;
        }}
        return '<tr class="tot-row"><td class="name">Totals</td>' +
          '<td class="r">' + esc(String(s)) + '</td>' +
          '<td class="r">' + tok3(ti, tc, to) + '</td>' +
          '<td class="r">' + toolsCell(tl, tlf) + '</td>' +
          '<td class="r">' + esc(fmtCost(co)) + '</td>' +
          '<td class="r">\\u2014</td></tr>';
      }},
      cols: [
        {{ k: 'user', label: 'User', get: function(r) {{ return r.user_name || ''; }} }},
        {{ k: 'sessions', label: 'Sessions', cls: 'r', num: true, get: function(r) {{ return r.sessions || 0; }} }},
        {{ k: 'tokens', label: 'Tokens in \\u2192 cached \\u2192 out', cls: 'r', num: true, title: 'cached input tokens shown in the middle', get: function(r) {{ return (r.tokens_in || 0) + (r.tokens_out || 0); }} }},
        {{ k: 'tools', label: 'Tools', cls: 'r', num: true, get: function(r) {{ return r.tools || 0; }} }},
        {{ k: 'cost', label: 'Cost', cls: 'r', num: true, get: function(r) {{ return r.cost || 0; }} }},
        {{ k: 'lastactive', label: 'Last active', cls: 'r', num: true, get: function(r) {{ return r.last_active || 0; }} }}
      ]
    }});

    function renderLiveSessions() {{
      var rows = state.sessionsLive || [];
      var now = Date.now() / 1000;
      var keepS = keepMinutes * 60;
      var visible = [];
      for (var i = 0; i < rows.length; i++) {{
        var r = rows[i];
        if (r && r.done && now - r.done > keepS) continue;
        if (r) visible.push(r);
      }}
      var active = 0, done = 0, cost = 0, tin = 0, tcached = 0, tout = 0;
      for (var k = 0; k < visible.length; k++) {{
        var v0 = visible[k];
        if (v0.done) done++; else active++;
        cost += v0.cost || 0;
        tin += v0.tokens_in || 0;
        tcached += v0.tokens_cached || 0;
        tout += v0.tokens_out || 0;
      }}
      $(ID + '-ls-active').textContent = String(active);
      $(ID + '-ls-done').textContent = String(done);
      $(ID + '-ls-cost').textContent = fmtCost(cost);
      $(ID + '-ls-tokens').textContent = fmtTok(tin) + ' in / ' + fmtTok(tcached) + ' cached / ' + fmtTok(tout) + ' out';
      liveTable.update(visible);
    }}

    var nameSeg = $(ID + '-nameseg');
    if (nameSeg) nameSeg.addEventListener('click', function(e) {{
      var btn = e.target.closest('button');
      if (!btn) return;
      var btns = nameSeg.querySelectorAll('button');
      for (var i = 0; i < btns.length; i++) btns[i].classList.remove('active');
      btn.classList.add('active');
      nameMode = btn.getAttribute('data-mode') || 'names';
      renderLiveSessions();
      if (state.usLast) {{ modelsTable.update(state.usLast.by_model); }}
    }});
    var keepSel = $(ID + '-keep');
    if (keepSel) keepSel.addEventListener('change', function() {{
      keepMinutes = parseInt(keepSel.value, 10) || 10;
      renderLiveSessions();
    }});

    // ── Usage tab (usage analytics via the usage_stats read-action) ──
    var usRange = '24h';
    var usTasks = true;
    var usLoaded = false;
    var cfgLoaded = false;
    var updLoaded = false;
    var cfgFetch = function() {{}};
    var cfgOnEvent = function() {{}};
    var usRefreshMs = 900000;
    var usRefreshTimer = null;
    var usCaptions = {{ '1h': 'last hour', '6h': 'last 6 hours', '24h': 'last 24 hours', '7d': 'last 7 days', '30d': 'last 30 days' }};
    var usSpans = {{ '1h': 3600, '6h': 21600, '24h': 86400, '7d': 604800, '30d': 2592000 }};

    function usDelta(cur, prevVal) {{
      if (prevVal === undefined || prevVal === null) return '';
      if (!prevVal) return cur ? '<span class="us-chip" title="vs previous period of equal length">new</span>' : '';
      var pct = Math.round((cur - prevVal) / prevVal * 100);
      var arrow = pct > 0 ? '\\u25B2' : (pct < 0 ? '\\u25BC' : '\\u2192');
      return '<span class="us-chip" title="vs previous period of equal length">' + arrow + ' ' + Math.abs(pct) + '%</span>';
    }}

    function usSpark(buckets, key, color) {{
      if (!buckets.length) return '';
      var max = 0;
      for (var i = 0; i < buckets.length; i++) max = Math.max(max, buckets[i][key] || 0);
      if (!max) max = 1;
      var pts = [];
      for (var j = 0; j < buckets.length; j++) {{
        var x = buckets.length > 1 ? (j / (buckets.length - 1)) * 90 : 45;
        var y = 28 - (buckets[j][key] || 0) / max * 24;
        pts.push(x.toFixed(1) + ',' + y.toFixed(1));
      }}
      return '<svg width="90" height="30" viewBox="0 0 90 30"><polyline fill="none" stroke="' + color + '" stroke-width="1.5" points="' + pts.join(' ') + '"/></svg>';
    }}

    function usFill(meta, buckets, off) {{
      var by = {{}};
      for (var i = 0; i < buckets.length; i++) by[buckets[i].t] = buckets[i];
      var b = meta.bucket_s || 3600;
      var startB = Math.floor((meta.start + off) / b) * b - off;
      var out = [];
      for (var t = startB; t <= meta.now; t += b) {{
        out.push(by[t] || {{ t: t, tokens: 0, cost: 0, sessions: 0, tools: 0 }});
      }}
      return out;
    }}

    function usCard(title, chip, value, sub, spark) {{
      return '<div class="us-card"><div class="us-card-h"><span>' + esc(title) + '</span>' + chip + '</div>' +
        '<div class="us-card-row"><div><div class="us-card-v">' + esc(String(value)) + '</div>' +
        '<div class="us-card-sub">' + esc(sub) + '</div></div>' + (spark || '') + '</div></div>';
    }}

    function usCards(cards, prev, buckets) {{
      var p = prev || {{}};
      var h = '<div class="us-cards">';
      h += usCard('Sessions', usDelta(cards.sessions.count, p.sessions && p.sessions.count),
        cards.sessions.count,
        cards.sessions.failed + ' failed \\u00b7 ' + cards.sessions.cancelled + ' cancelled \\u00b7 ' + cards.sessions.retried + ' retried',
        usSpark(buckets, 'sessions', '#6366f1'));
      h += usCard('Tokens', usDelta(cards.tokens.total, p.tokens && p.tokens.total),
        fmtTok(cards.tokens.total),
        fmtTok(cards.tokens.input) + ' in \\u00b7 ' + fmtTok(cards.tokens.cached) + ' cached \\u00b7 ' + fmtTok(cards.tokens.output) + ' out \\u00b7 ' + fmtTok(cards.tokens.reasoning) + ' reasoning',
        '');
      h += usCard('Cost', usDelta(cards.cost.total, p.cost && p.cost.total),
        fmtCost(cards.cost.total),
        'avg ' + fmtCost(cards.cost.avg_per_session) + ' / session' + (cards.cost.task_portion ? ' \\u00b7 incl. ' + fmtCost(cards.cost.task_portion) + ' task models' : ''),
        usSpark(buckets, 'cost', '#f59e0b'));
      h += usCard('Tools', usDelta(cards.tools.count, p.tools && p.tools.count),
        cards.tools.count,
        cards.tools.failed + ' failed' + (cards.tools.count ? ' (' + Math.round(cards.tools.failed / cards.tools.count * 100) + '%)' : ''),
        usSpark(buckets, 'tools', '#8b5cf6'));
      h += '</div>';
      return h;
    }}

    function fmtRange(t0, t1) {{
      var MON = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
      function p2(v) {{ return ('0' + v).slice(-2); }}
      function dp(d) {{ return MON[d.getMonth()] + ' ' + d.getDate(); }}
      function tp(d) {{ return p2(d.getHours()) + ':' + p2(d.getMinutes()); }}
      var a = new Date(t0 * 1000), b = new Date(t1 * 1000);
      if ((t1 - t0) <= 90000) {{
        var sameDay = a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
        return sameDay ? dp(a) + ', ' + tp(a) + ' \\u2013 ' + tp(b)
                       : dp(a) + ' ' + tp(a) + ' \\u2013 ' + dp(b) + ' ' + tp(b);
      }}
      return dp(a) + ' \\u2013 ' + dp(b);
    }}
    var _usGeom = null;
    function usChart(buckets) {{
      if (!buckets.length) return '<div class="loading">No data in range.</div>';
      var W = 940, H = 200, PL = 42, PR = 48, PT = 10, PB = 22;
      var iw = W - PL - PR, ih = H - PT - PB;
      var rawTok = 0, rawCost = 0;
      for (var i = 0; i < buckets.length; i++) {{
        rawTok = Math.max(rawTok, buckets[i].tokens || 0);
        rawCost = Math.max(rawCost, buckets[i].cost || 0);
      }}
      function niceStep(rawMax, target) {{
        if (!(rawMax > 0) || !isFinite(rawMax)) return 1;
        var raw = rawMax / target;
        var base = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
        var f = raw / base;
        var nf = f <= 1 ? 1 : (f <= 2 ? 2 : (f <= 2.5 ? 2.5 : (f <= 5 ? 5 : 10)));
        return nf * base;
      }}
      function ticksFor(rawMax, target) {{
        if (!(rawMax > 0) || !isFinite(rawMax)) return {{ max: 1, vals: [0, 1] }};
        var step = niceStep(rawMax, target);
        var mx = Math.ceil(rawMax / step) * step, out = [];
        for (var v = 0; v <= mx + step * 0.0001; v += step) out.push(v);
        return {{ max: mx, vals: out }};
      }}
      var tokAxis = ticksFor(rawTok, 7), costAxis = ticksFor(rawCost, 7);
      var maxTok = tokAxis.max, maxCost = costAxis.max;
      var n = buckets.length;
      var slot = n > 1 ? iw / (n - 1) : 0;
      function xAt(j) {{ return n > 1 ? PL + j * slot : PL + iw / 2; }}
      var tokPts = [], costPts = [];
      for (var j = 0; j < n; j++) {{
        var x = xAt(j);
        tokPts.push([x, PT + ih - (buckets[j].tokens || 0) / maxTok * ih]);
        costPts.push([x, PT + ih - (buckets[j].cost || 0) / maxCost * ih]);
      }}
      function smoothPath(p) {{
        if (!p.length) return '';
        if (p.length < 3) return 'M' + p.map(function(q) {{ return q[0].toFixed(1) + ',' + q[1].toFixed(1); }}).join(' L');
        var lo = PT, hi = PT + ih;
        function clampY(v) {{ return v < lo ? lo : (v > hi ? hi : v); }}
        var d = 'M' + p[0][0].toFixed(1) + ',' + p[0][1].toFixed(1);
        for (var k = 0; k < p.length - 1; k++) {{
          var a = p[k > 0 ? k - 1 : 0], b = p[k], c = p[k + 1], e = p[k + 2 < p.length ? k + 2 : k + 1];
          var c1x = b[0] + (c[0] - a[0]) / 6, c1y = clampY(b[1] + (c[1] - a[1]) / 6);
          var c2x = c[0] - (e[0] - b[0]) / 6, c2y = clampY(c[1] - (e[1] - b[1]) / 6);
          d += ' C' + c1x.toFixed(1) + ',' + c1y.toFixed(1) + ' ' + c2x.toFixed(1) + ',' + c2y.toFixed(1) + ' ' + c[0].toFixed(1) + ',' + c[1].toFixed(1);
        }}
        return d;
      }}
      function areaPath(p) {{
        var line = smoothPath(p);
        if (!line) return '';
        return line + ' L' + p[p.length - 1][0].toFixed(1) + ',' + (PT + ih).toFixed(1) +
               ' L' + p[0][0].toFixed(1) + ',' + (PT + ih).toFixed(1) + ' Z';
      }}
      function axisCost(v, step) {{
        var dec = 0;
        if (step > 0 && step < 1) {{
          dec = 6;
          for (var q = 1; q <= 6; q++) {{
            var scaled = step * Math.pow(10, q);
            if (Math.abs(scaled - Math.round(scaled)) < 1e-9) {{ dec = q; break; }}
          }}
        }}
        return '$' + v.toFixed(dec);
      }}
      function two(v) {{ return ('0' + v).slice(-2); }}
      function lbl(t) {{
        var d = new Date(t * 1000);
        return (buckets[n - 1].t - buckets[0].t) > 90000
          ? (d.getMonth() + 1) + '/' + d.getDate()
          : two(d.getHours()) + ':' + two(d.getMinutes());
      }}
      function yTok(v) {{ return PT + ih - v / maxTok * ih; }}
      function yCost(v) {{ return PT + ih - v / maxCost * ih; }}
      var grid = '';
      for (var gi = 0; gi < tokAxis.vals.length; gi++) {{
        var gy = yTok(tokAxis.vals[gi]);
        grid += '<line x1="' + PL + '" y1="' + gy.toFixed(1) + '" x2="' + (W - PR) + '" y2="' + gy.toFixed(1) + '" stroke="var(--border)" stroke-width="0.5"/>';
        grid += '<text x="' + (PL - 4) + '" y="' + (gy + 3).toFixed(1) + '" font-size="9" fill="#94a3b8" text-anchor="end">' + esc(fmtTok(tokAxis.vals[gi])) + '</text>';
      }}
      var costStep = costAxis.vals.length > 1 ? (costAxis.vals[1] - costAxis.vals[0]) : (costAxis.max || 1);
      for (var ci = 0; ci < costAxis.vals.length; ci++) {{
        var cy2 = yCost(costAxis.vals[ci]);
        grid += '<text x="' + (W - PR + 4) + '" y="' + (cy2 + 3).toFixed(1) + '" font-size="9" fill="#94a3b8">' + esc(axisCost(costAxis.vals[ci], costStep)) + '</text>';
      }}
      var xlabels = '', XN = Math.min(9, n);
      for (var xi = 0; xi < XN; xi++) {{
        var bi = XN > 1 ? Math.round(xi * (n - 1) / (XN - 1)) : 0;
        var lx = xAt(bi);
        var anchor = xi === 0 ? 'start' : (xi === XN - 1 ? 'end' : 'middle');
        xlabels += '<text x="' + lx.toFixed(1) + '" y="' + (H - 6) + '" font-size="9" fill="#94a3b8" text-anchor="' + anchor + '">' + esc(lbl(buckets[bi].t)) + '</text>';
      }}
      var dots = '';
      if (n === 1) {{
        dots = '<circle cx="' + tokPts[0][0].toFixed(1) + '" cy="' + tokPts[0][1].toFixed(1) + '" r="3" fill="#14b8a6"/>' +
               '<circle cx="' + costPts[0][0].toFixed(1) + '" cy="' + costPts[0][1].toFixed(1) + '" r="3" fill="#f59e0b"/>';
      }}
      var xs = [], tokY = [], costY = [], times = [], tvals = [], cvals = [];
      for (var m = 0; m < n; m++) {{
        xs.push(tokPts[m][0]); tokY.push(tokPts[m][1]); costY.push(costPts[m][1]);
        times.push(lbl(buckets[m].t)); tvals.push(buckets[m].tokens || 0); cvals.push(buckets[m].cost || 0);
      }}
      _usGeom = {{ xs: xs, tokY: tokY, costY: costY, times: times, tv: tvals, cv: cvals, W: W, H: H }};
      return '<div style="position:relative;">' +
        '<svg width="100%" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" style="border:1px solid var(--border);border-radius:10px;">' +
        '<defs>' +
        '<linearGradient id="' + ID + '-gtok" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="rgba(20,184,166,0.30)"/><stop offset="1" stop-color="rgba(20,184,166,0)"/></linearGradient>' +
        '<linearGradient id="' + ID + '-gcost" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="rgba(245,158,11,0.28)"/><stop offset="1" stop-color="rgba(245,158,11,0)"/></linearGradient>' +
        '</defs>' + grid +
        '<rect x="' + PL + '" y="' + PT + '" width="' + iw + '" height="' + ih + '" fill="transparent" pointer-events="all"/>' +
        '<path fill="url(#' + ID + '-gtok)" stroke="none" d="' + areaPath(tokPts) + '"/>' +
        '<path fill="none" stroke="#14b8a6" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="' + smoothPath(tokPts) + '"/>' +
        '<path fill="url(#' + ID + '-gcost)" stroke="none" d="' + areaPath(costPts) + '"/>' +
        '<path fill="none" stroke="#f59e0b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" d="' + smoothPath(costPts) + '"/>' +
        '<line id="' + ID + '-us-hline" x1="0" y1="' + PT + '" x2="0" y2="' + (PT + ih) + '" stroke="#94a3b8" stroke-width="1" stroke-dasharray="3 3" style="visibility:hidden;pointer-events:none;"/>' +
        '<circle id="' + ID + '-us-dtok" r="3.5" fill="#14b8a6" stroke="#fff" stroke-width="1.5" style="visibility:hidden;pointer-events:none;"/>' +
        '<circle id="' + ID + '-us-dcost" r="3.5" fill="#f59e0b" stroke="#fff" stroke-width="1.5" style="visibility:hidden;pointer-events:none;"/>' +
        dots + xlabels + '</svg>' +
        '<div id="' + ID + '-us-tip" style="position:absolute;pointer-events:none;visibility:hidden;background:rgba(15,23,42,0.92);color:#fff;font-size:11px;line-height:1.5;padding:6px 9px;border-radius:6px;white-space:nowrap;box-shadow:0 2px 8px rgba(0,0,0,0.3);z-index:5;"></div>' +
        '</div>';
    }}

    function usBindHover() {{
      var box = $(ID + '-us-chart');
      if (!box) return;
      var svg = box.querySelector('svg'), g = _usGeom;
      if (!svg || !g || !g.xs.length) return;
      var hl = $(ID + '-us-hline'), dt = $(ID + '-us-dtok'), dc = $(ID + '-us-dcost'), tip = $(ID + '-us-tip');
      if (!hl || !dt || !dc || !tip) return;
      function hide() {{ hl.style.visibility = 'hidden'; dt.style.visibility = 'hidden'; dc.style.visibility = 'hidden'; tip.style.visibility = 'hidden'; }}
      svg.addEventListener('mousemove', function(ev) {{
        var rect = svg.getBoundingClientRect();
        if (!rect.width) return;
        var sx = (ev.clientX - rect.left) / rect.width * g.W;
        var best = 0, bd = 1e12;
        for (var i = 0; i < g.xs.length; i++) {{ var d = Math.abs(g.xs[i] - sx); if (d < bd) {{ bd = d; best = i; }} }}
        var x = g.xs[best];
        hl.setAttribute('x1', x.toFixed(1)); hl.setAttribute('x2', x.toFixed(1)); hl.style.visibility = 'visible';
        dt.setAttribute('cx', x.toFixed(1)); dt.setAttribute('cy', g.tokY[best].toFixed(1)); dt.style.visibility = 'visible';
        dc.setAttribute('cx', x.toFixed(1)); dc.setAttribute('cy', g.costY[best].toFixed(1)); dc.style.visibility = 'visible';
        tip.innerHTML = '<div style="font-weight:600;margin-bottom:2px;">' + esc(g.times[best]) + '</div>' +
          '<div><span style="color:#14b8a6;">\\u25cf</span> Tokens: ' + esc(fmtTok(g.tv[best])) + '</div>' +
          '<div><span style="color:#f59e0b;">\\u25cf</span> Cost: ' + esc(fmtCost(g.cv[best])) + '</div>';
        var px = x / g.W * rect.width, tw = tip.offsetWidth || 96;
        tip.style.left = (px + 14 + tw > rect.width ? px - 14 - tw : px + 14) + 'px';
        tip.style.top = Math.max(0, Math.min(g.tokY[best], g.costY[best]) / g.H * rect.height - 8) + 'px';
        tip.style.visibility = 'visible';
      }});
      svg.addEventListener('mouseleave', hide);
    }}

    function usTotals(meta) {{
      var sinceText = meta.since ? new Date(meta.since * 1000).toLocaleDateString() : '\\u2014';
      return '<div style="font-size:11px;color:var(--text-faint);margin-top:6px;">Figures cover activity since collection was enabled (' + esc(sinceText) + '); task models served outside this pipe are not counted.</div>';
    }}

    function usMetaLine(meta) {{
      var parts = [];
      parts.push('retention ' + (meta.retention_days || '?') + 'd');
      if (meta.records !== undefined && meta.records !== null) parts.push(meta.records + ' records');
      if (meta.approx_bytes) parts.push('\\u2248 ' + fmtBytes(meta.approx_bytes));
      parts.push('times in your timezone');
      return 'collection ' + (meta.collect_on ? 'ON' : 'OFF') + ' \\u00b7 ' + parts.join(' \\u00b7 ');
    }}

    function usApplyRetention(meta) {{
      if (!meta || !meta.retention_days) return;
      var btns = $(ID + '-us-range').querySelectorAll('button');
      for (var i = 0; i < btns.length; i++) {{
        var rk = btns[i].getAttribute('data-range');
        var over = (usSpans[rk] || 0) > meta.retention_days * 86400;
        btns[i].disabled = over;
        btns[i].title = over ? 'beyond configured retention (' + meta.retention_days + 'd)' : '';
        btns[i].style.opacity = over ? '0.35' : '';
      }}
    }}

    function usRender(res) {{
      var note = $(ID + '-us-note'), body = $(ID + '-us-body');
      usApplyRetention(res && res.meta);
      if (!res || res.available === false) {{
        var reason = (res && res.reason) || 'request failed';
        if (state.usHasData) {{
          note.style.display = '';
          note.textContent = 'Refresh failed (' + reason + ') \\u2014 showing last loaded data.';
          reportHeight();
          return;
        }}
        body.style.display = 'none';
        note.style.display = '';
        if (res && res.meta && res.meta.collect_on === false) {{
          note.textContent = 'Usage collection is off. Enable the PIPE_DASHBOARD_USAGE_COLLECT valve to start recording usage for this tab.';
        }} else {{
          note.textContent = 'Usage data unavailable: ' + reason;
        }}
        reportHeight();
        return;
      }}
      if (res.meta && res.meta.collect_on === false && (!res.totals || !res.totals.sessions)) {{
        body.style.display = 'none';
        note.style.display = '';
        note.textContent = 'Usage collection is off \\u2014 no records yet. Enable the PIPE_DASHBOARD_USAGE_COLLECT valve to start recording; usage appears here from that point on.';
        reportHeight();
        return;
      }}
      state.usHasData = true;
      state.usLast = res;
      state.lastCards = res.cards;
      note.style.display = 'none';
      body.style.display = '';
      var off = -(new Date().getTimezoneOffset()) * 60;
      var buckets = usFill(res.meta, res.buckets || [], off);
      $(ID + '-us-cards').innerHTML = usCards(res.cards, res.prev, buckets);
      renderSystemCards();
      var stepr = buckets.length > 1 ? (buckets[1].t - buckets[0].t) : 3600;
      $(ID + '-us-legend').textContent = buckets.length ? (fmtRange(buckets[0].t, buckets[buckets.length - 1].t + stepr) + ' \\u00b7 your timezone') : '';
      $(ID + '-us-chart').innerHTML = usChart(buckets);
      usBindHover();
      modelsTable.update(res.by_model);
      usersTable.update(res.by_user);
      $(ID + '-us-totals').innerHTML = usTotals(res.meta);
      $(ID + '-us-meta').textContent = usMetaLine(res.meta);
      reportHeight();
    }}

    function usFetch() {{
      var note = $(ID + '-us-note');
      note.style.display = '';
      note.textContent = 'Loading usage data...';
      $(ID + '-us-range-label').textContent = usCaptions[usRange] || usRange;
      callAction('usage_stats', {{ range: usRange, tz_offset_min: -(new Date().getTimezoneOffset()), include_tasks: usTasks }})
        .then(function(resp) {{
          if (resp && resp.ok && resp.result) usRender(resp.result);
          else usRender({{ available: false, reason: (resp && resp.error) || 'request failed' }});
        }})
        .catch(function() {{ usRender({{ available: false, reason: 'request failed' }}); }});
      usScheduleRefresh();
    }}

    function usScheduleRefresh() {{
      if (usRefreshTimer) {{ clearInterval(usRefreshTimer); usRefreshTimer = null; }}
      if (!usRefreshMs) return;
      usRefreshTimer = setInterval(function() {{
        var pane = $(ID + '-tab-usage');
        if (pane && pane.classList.contains('active')) usFetch();
      }}, usRefreshMs);
    }}

    function fmtBytes(n) {{
      n = n || 0;
      if (n >= 1073741824) return (n / 1073741824).toFixed(1) + ' GB';
      if (n >= 1048576) return (n / 1048576).toFixed(1) + ' MB';
      if (n >= 1024) return (n / 1024).toFixed(1) + ' KB';
      return n + ' B';
    }}

    function renderSystemCards() {{
      var el = $(ID + '-us-system');
      if (!el) return;
      var h = '';
      var cards = state.lastCards;
      if (cards) {{
        h += usCard('Errors', '', Math.round((cards.errors.rate || 0) * 100) + '%',
          cards.sessions.failed + ' failed sessions', '');
        h += usCard('Cached input', '', Math.round((cards.cached.pct || 0) * 100) + '%',
          '\\u2248 ' + fmtCost(cards.cached.savings) + ' saved', '');
      }}
      var sys = state.system;
      if (sys) {{
        var cpuVal = sys.cpu_pct !== undefined ? Math.round(sys.cpu_pct) + '%'
          : (sys.load1 !== undefined ? sys.load1.toFixed(2) : '?');
        var cpuSub = sys.load1 !== undefined
          ? 'load ' + sys.load1.toFixed(2) + ' / ' + (sys.cores || '?') + ' cores'
          : ((sys.cores || '?') + ' cores');
        h += usCard('CPU', '', cpuVal, cpuSub, '');
        var memVal = sys.mem_used_pct !== undefined ? Math.round(sys.mem_used_pct) + '%' : '?';
        var memSub = state.workersRss ? 'workers ' + fmtBytes(state.workersRss) : 'host';
        if (sys.mem_total) memSub += ' \\u00b7 host ' + fmtBytes(sys.mem_total * (sys.mem_used_pct || 0) / 100) + ' / ' + fmtBytes(sys.mem_total);
        h += usCard('Memory', '', memVal, memSub, '');
        if (sys.disk_total) {{
          var usedPct = Math.round((1 - (sys.disk_free || 0) / sys.disk_total) * 100);
          h += usCard('Disk (data dir)', '', fmtBytes(sys.disk_free), 'free of ' + fmtBytes(sys.disk_total) + ' (' + usedPct + '% used)', '');
        }}
      }}
      el.innerHTML = h;
    }}

    var usRangeSeg = $(ID + '-us-range');
    if (usRangeSeg) usRangeSeg.addEventListener('click', function(e) {{
      var btn = e.target.closest('button');
      if (!btn || btn.disabled) return;
      var btns = usRangeSeg.querySelectorAll('button');
      for (var i = 0; i < btns.length; i++) btns[i].classList.remove('active');
      btn.classList.add('active');
      usRange = btn.getAttribute('data-range') || '24h';
      usFetch();
    }});
    var usTasksBox = $(ID + '-us-tasks');
    if (usTasksBox) usTasksBox.addEventListener('change', function() {{
      usTasks = !!usTasksBox.checked;
      usFetch();
    }});
    var usRefreshSel = $(ID + '-us-refresh');
    if (usRefreshSel) usRefreshSel.addEventListener('change', function() {{
      var m = parseInt(usRefreshSel.value, 10);
      usRefreshMs = (m > 0 ? m : 15) * 60000;
      usScheduleRefresh();
    }});

    // ── Section updaters ──

    function updateIdentity(d) {{
      var wc = d.worker_count || 1;
      var workerText = wc > 1 ? 'Aggregated from ' + wc + ' workers' : 'Single worker';
      var pidText = state.pid ? ' \\u00b7 emitter pid ' + state.pid : '';
      $(ID + '-footer').textContent = 'v' + (d.version || '?') + ' \u00b7 ' + (d.pipe_id || '?') + ' \u00b7 ' + workerText + pidText;
    }}

    function updateConcurrency(c) {{
      var ar = c.active_requests || 0, mr = c.max_requests || 0;
      $(ID + '-req-val').textContent = ar + ' / ' + mr;
      var at = c.active_tools || 0, mt = c.max_tools || 0;
      $(ID + '-tool-val').textContent = at + ' / ' + mt;
    }}

    function updateQueues(q) {{
      var backlog = (q.waiting || 0) + (q.requests || 0);
      var rq = $(ID + '-rq-val');
      rq.textContent = String(backlog);
      rq.title = 'waiters ' + (q.waiting || 0) + ' + queued ' + (q.requests || 0) + ' / ' + (q.requests_max || 0);
      $(ID + '-lq-val').textContent = (q.logs || 0) + (q.logs_max ? ' / ' + q.logs_max : '');
      $(ID + '-aq-val').textContent = (q.archive || 0) + (q.archive_max ? ' / ' + q.archive_max : '');
      var toolEl = $(ID + '-tool-val');
      if (toolEl) toolEl.title = 'waiting for a tool slot: ' + (q.tool_waiting || 0);
    }}

    function updateVideos(v) {{
      var row = $(ID + '-video-row');
      if (!v || !v.max) {{ row.style.display = 'none'; return; }}
      row.style.display = '';
      $(ID + '-video-val').textContent = (v.active || 0) + ' / ' + v.max;
    }}

    function updateSessions(s) {{
      $(ID + '-sess-val').textContent = String(s.in_flight || 0);
    }}

    function updateRateLimits(rl) {{
      var section = $(ID + '-rl-section');
      section.style.display = '';
      var head = section.querySelector('.section-h');
      if (head) head.title = (state.workerCount > 1)
        ? 'aggregate across workers — a user active on multiple workers is counted once per worker' : '';
      var tu = rl.tracked_users || 0, fu = rl.users_with_failures || 0;
      var tru = rl.tripped_users || 0, th = rl.threshold || 0;
      var ws = rl.window_s || 0;
      var tt = rl.tool_tracked || 0, tf = rl.tool_with_failures || 0, tp = rl.tool_tripped || 0;
      var aa = rl.auth_failures_active || 0;
      var h = '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
        '<th>Type</th><th class="r" title="distinct users/pairs seen since worker start">Seen</th>' +
        '<th class="r">Users w/ fail</th>' +
        '<th class="r">Tripped</th><th class="r">Threshold</th><th class="r">Window</th>' +
        '</tr></thead><tbody>';
      var reqLevel = tru > 0 ? 'err' : (fu > 0 ? 'warn' : 'ok');
      h += '<tr><td>Requests</td><td class="r">' + tu + '</td>' +
        '<td class="r">' + fu + '</td>' +
        '<td class="r">' + badge(String(tru), reqLevel) + '</td>' +
        '<td class="r">' + th + '</td>' +
        '<td class="r">' + ws + 's</td></tr>';
      var toolLevel = tp > 0 ? 'err' : (tf > 0 ? 'warn' : 'ok');
      h += '<tr><td>Tools</td><td class="r">' + tt + '</td>' +
        '<td class="r">' + tf + '</td>' +
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
      var modParts = [];
      if (m.text !== undefined) modParts.push(esc(String(m.text)) + ' text');
      if (m.image !== undefined) modParts.push(esc(String(m.image)) + ' image');
      if (m.video !== undefined) modParts.push(esc(String(m.video)) + ' video');
      var modSub = modParts.length ? '<div style="font-size:11px;color:var(--text-faint);margin-top:2px;">' + modParts.join(' \\u00b7 ') + '</div>' : '';
      h += '<div class="gc" style="flex-direction:column;align-items:stretch;">' +
        '<div style="display:flex;justify-content:space-between;align-items:baseline;"><span class="gc-l">Models loaded</span><span class="gc-v">' + esc(String(m.loaded || 0)) + '</span></div>' + modSub + '</div>';
      if (m.zdr !== undefined && m.zdr !== null) h += gc('ZDR-capable', m.zdr);
      h += gc('Specs cached', m.specs_cached || 0);
      h += gc('Last fetch (chat)', m.last_fetch_ago || 'never');
      if (m.video_fetch_ago) h += gc('Last fetch (video)', m.video_fetch_ago);
      if (m.image_fetch_ago) h += gc('Last fetch (image)', m.image_fetch_ago);
      if (m.video_attempt_ago) h += gc('Last attempt (video)', m.video_attempt_ago);
      if (m.image_attempt_ago) h += gc('Last attempt (image)', m.image_attempt_ago);
      h += gc('Failures (chat)', m.failures || 0);
      h += gc('<span title="status of the chat-catalog fetch loop; image/video freshness shown above">Chat catalog</span>', statusBadge, true, true);
      if (m.last_error) {{
        var errText = m.last_error + (m.last_error_ago ? ' (' + m.last_error_ago + ')' : '');
        h += gc('Last error', errText);
      }}
      h += '</div>';
      $(ID + '-models-panel').innerHTML = h;
    }}

    function updateStorage(s) {{
      $(ID + '-storage-section').style.display = '';
      var h = '';
      if (s.as_of) {{
        var age = Math.max(0, Math.round(Date.now() / 1000 - s.as_of));
        h += '<div style="font-size:11px;color:var(--text-faint);margin-bottom:6px;">data as of ' +
          (age < 5 ? 'now' : age + 's ago') + ' \\u00b7 refreshes about every 60s \\u00b7 timestamps are access times</div>';
      }}
      if (s.state === 'unavailable') {{
        h += '<div class="live-expired">Storage unavailable on this worker' + (s.error ? ' (' + esc(s.error) + ')' : '') + '</div>';
      }} else if (s.state === 'degraded') {{
        h += '<div class="live-expired">Storage queries degraded' + (s.error ? ' (' + esc(s.error) + ')' : '') + '</div>';
      }}
      h += '<div class="g">';
      h += gc('Database', s.connected ? badge('Connected', 'ok') : badge('Not connected', 'off'), true);
      if (s.table) h += gc('Table', s.table);
      h += gc('Total items', pending(s.total_items), true);
      h += gc('<span title="approximate: JSON text length of stored artifacts, not disk bytes">Pipe table size</span>', pending(s.total_size), true, true);
      h += gc('Encrypted items', pending(s.encrypted_count), true);
      h += gc('Encryption', badge(s.encryption_mode || 'Unknown', s.encryption_mode === 'Disabled' ? 'off' : 'ok'), true);
      var compLabel;
      if (s.compression_mode === 'LZ4') {{
        compLabel = badge('LZ4', 'ok');
        var minB = parseInt(String(s.compress_min_bytes || '0').replace(/,/g, ''), 10);
        if (minB > 0) compLabel += ' \\u2265 ' + esc(s.compress_min_bytes) + ' B';
      }} else {{
        compLabel = badge('Disabled', 'off');
      }}
      h += gc('Compression', compLabel, true);
      h += '</div>';

      var ageTitle = 'access time — updated whenever an item is read';
      if (s.by_type && s.by_type.length > 0) {{
        h += '<div class="sub-h">By Type</div>';
        h += '<div class="tbl-wrap"><table class="tbl"><thead><tr><th>Type</th><th class="r">Count</th><th class="r">Size</th><th class="r" title="' + ageTitle + '">Least recent</th><th class="r" title="' + ageTitle + '">Most recent</th></tr></thead><tbody>';
        for (var i = 0; i < s.by_type.length; i++) {{
          var t = s.by_type[i];
          h += '<tr><td class="name">' + esc(t.type) + '</td><td class="r">' + esc(t.count) + '</td><td class="r">' + esc(t.size) + '</td><td class="r">' + esc(t.oldest) + '</td><td class="r">' + esc(t.newest) + '</td></tr>';
        }}
        h += '</tbody></table></div>';
      }}

      if (s.by_model && s.by_model.length > 0) {{
        h += '<div class="sub-h">By Model</div>';
        h += '<div class="scroll-table"><table class="tbl"><thead><tr><th>Model</th><th class="r">Items</th><th class="r">Size</th><th class="r">Chats</th><th class="r" title="' + ageTitle + '">Least recent</th><th class="r" title="' + ageTitle + '">Most recent</th></tr></thead><tbody>';
        for (var j = 0; j < s.by_model.length; j++) {{
          var bm = s.by_model[j];
          h += '<tr><td class="name" title="' + esc(bm.model_id) + '">' + esc(bm.name) + '</td><td class="r">' + esc(bm.count) + '</td><td class="r">' + esc(bm.size) + '</td><td class="r">' + esc(bm.chats) + '</td><td class="r">' + esc(bm.oldest) + '</td><td class="r">' + esc(bm.newest) + '</td></tr>';
        }}
        h += '</tbody></table></div>';
      }}

      $(ID + '-storage-panel').innerHTML = h;
      hideLoad('storage');
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
      // Session logging (tri-state: the archive thread starts lazily on first persist)
      var lw = d.log_worker || (d.logging_enabled ? 'idle' : 'disabled');
      if (lw === 'active') h += gc('Session logging', badge('Active', 'ok'), true);
      else if (lw === 'idle') h += gc('Session logging', badge('Idle (starts on first archive)', 'off'), true);
      else if (lw === 'stopped') h += gc('Session logging', badge('Worker stopped', 'err'), true);
      else h += gc('Session logging', badge('Disabled', 'off'), true);
      if (d.log_buffers !== undefined) {{
        h += gc('Log buffers (RAM)', (d.log_buffers || 0) + ' buf / ' + (d.log_events_buffered || 0) + ' events');
      }}
      // Redis (pipe-level client; liveness-probed by the publisher)
      if (d.redis_enabled) {{
        h += gc('Redis (pipe)', d.redis_connected ? badge('Connected', 'ok') : badge('Disconnected', 'warn'), true);
      }} else {{
        h += gc('Redis (pipe)', badge('Disabled', 'off'), true);
      }}
      h += '</div>';
      $(ID + '-health-panel').innerHTML = h;
      hideLoad('system');
    }}

    function updateDb(db) {{
      $(ID + '-db-section').style.display = '';
      var h = '<div class="g">';
      h += gc('Write pool backlog', (db.pool_pending || 0) + ' pending / ' + (db.pool_workers || 0) + ' workers');
      var lvl = (db.breakers_tripped || 0) > 0 ? 'err' : 'ok';
      h += gc('DB breakers', badge((db.breakers_tripped || 0) + ' tripped', lvl) +
        ' <span style="color:var(--text-faint);font-size:11px;">' + esc(String(db.breakers_tracked || 0)) + ' seen</span>', true);
      h += '</div>';
      $(ID + '-db-panel').innerHTML = h;
      hideLoad('system');
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
      hideLoad('system');
    }}

    function updateWorkers(workers) {{
      if (!workers || workers.length < 1) {{
        $(ID + '-workers-section').style.display = 'none';
        return;
      }}
      $(ID + '-workers-section').style.display = '';
      var h = '<div class="tbl-wrap"><table class="tbl"><thead><tr>' +
        '<th>PID</th><th class="r">Uptime</th><th class="r">Active</th>' +
        '<th class="r" title="age of this worker\\'s last published slice">Seen</th><th class="r">Status</th>' +
        '</tr></thead><tbody>';
      for (var i = 0; i < workers.length; i++) {{
        var w = workers[i];
        var age = (w.last_seen_age === undefined || w.last_seen_age === null) ? null : w.last_seen_age;
        var hw = w.health || {{}};
        var st;
        if (hw.wf) st = badge('Warmup failed', 'err');
        else if (age !== null && age > 6) st = badge('Stale', 'warn');
        else st = badge('Active', 'ok');
        var seen = age === null ? '-' : (age < 3 ? 'now' : Math.round(age) + 's ago');
        h += '<tr><td class="name">' + esc(String(w.pid)) + '</td>' +
          '<td class="r">' + esc(fmtUptime(w.uptime_s || 0)) + '</td>' +
          '<td class="r">' + esc(String(w.active_requests || 0)) + '</td>' +
          '<td class="r">' + esc(seen) + '</td>' +
          '<td class="r">' + st + '</td></tr>';
      }}
      h += '</tbody></table></div>';
      $(ID + '-workers-panel').innerHTML = h;
      hideLoad('system');
    }}

    function updatePlugins(list) {{
      $(ID + '-plugins-section').style.display = '';
      hideLoad('about');
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
      if (d.worker_count !== undefined) state.workerCount = d.worker_count;
      if (d.pid !== undefined) state.pid = d.pid;
      $(ID + '-degraded').style.display = d.degraded ? 'block' : 'none';
      if (d.identity) {{ state.identity = d.identity; updateIdentity(d.identity); }}
      if (d.concurrency) updateConcurrency(d.concurrency);
      if (d.queues) updateQueues(d.queues);
      if (d.videos) updateVideos(d.videos);
      if (d.rate_limits) updateRateLimits(d.rate_limits);
      if (d.sessions) updateSessions(d.sessions);
      if (d.sessions_live !== undefined) updateLiveSessions(d.sessions_live);
      if (d.uptime_s !== undefined) {{
        var upTitle = (state.workerCount > 1) ? 'oldest worker' : '';
        var upEl = $(ID + '-up-val');
        upEl.textContent = fmtUptime(d.uptime_s);
        upEl.title = upTitle;
        var topEl = $(ID + '-uptime');
        topEl.textContent = fmtUptime(d.uptime_s);
        topEl.title = upTitle;
      }}
      if (d.models) {{ state.models = d.models; updateModels(d.models); }}
      if (d.system) {{ state.system = d.system; renderSystemCards(); }}
      if (d.workers_rss !== undefined) {{ state.workersRss = d.workers_rss; renderSystemCards(); }}
      if (d.health) updateHealth(d.health);
      if (d.db) updateDb(d.db);
      if (d.storage) {{ state.storage = d.storage; updateStorage(d.storage); }}
      if (d.config) updateConfig(d.config);
      if (d.workers) {{ state.workers = d.workers; updateWorkers(d.workers); }}
      if (d.plugins) {{ state.plugins = d.plugins; updatePlugins(d.plugins); }}
      // Update worker_count from top-level field (sent every tick in Redis mode)
      if (d.worker_count !== undefined && state.identity) {{
        state.identity.worker_count = d.worker_count;
        updateIdentity(state.identity);
      }}
      reportHeight();
    }}

    function callAction(name, args) {{
      var token = null;
      try {{ token = localStorage.getItem("token"); }} catch (e) {{}}
      if (!token) return Promise.reject("no token");
      return fetch("/api/pipe/dashboard/action", {{
        method: "POST",
        headers: {{ "Content-Type": "application/json", "Authorization": "Bearer " + token }},
        body: JSON.stringify({{ action: name, args: args || {{}} }})
      }}).then(function(r) {{ return r.json(); }});
    }}
    window.pipeDashboardCallAction = callAction;

    {CONFIG_TAB_JS}

    {UPDATE_TAB_JS}

    var sock = null;
    var gotData = false;

    function connectDashboard() {{
      if (typeof io === "undefined") {{ setStatic('Live updates unavailable (socket client missing).'); return; }}
      var token = null;
      try {{ token = localStorage.getItem("token"); }} catch (e) {{}}
      if (!token) {{ setStatic('Live updates require Open WebUI\\'s iframe same-origin setting (Settings \\u2192 Interface).'); return; }}
      var origin = "";
      try {{ origin = parent.location.origin; }} catch (e) {{ origin = ""; }}
      sock = io(origin || undefined, {{ reconnection: true, reconnectionDelay: 1000, reconnectionDelayMax: 5000, randomizationFactor: 0.5, path: "/ws/socket.io", transports: ["websocket", "polling"], auth: {{ token: token }} }});
      sock.on("connect", function() {{
        try {{
          sock.emit("user-join", {{ auth: {{ token: token }} }}, function() {{
            try {{ sock.emit("{SUB_EVENT}"); }} catch (e) {{}}
          }});
        }} catch (e) {{}}
      }});
      sock.on("{DASHBOARD_EVENT}", function(data) {{
        if (!data || typeof data !== "object") return;
        gotData = true;
        setLive();
        $(ID + '-error').style.display = 'none';
        updateDashboard(data);
        if (cfgLoaded && data.cfgRev != null) cfgOnEvent(data.cfgRev);
      }});
      sock.on("{CONFIG_EVENT}", function(d) {{
        if (cfgLoaded && d && d.rev != null) cfgOnEvent(d.rev);
      }});
      sock.on("{DENIED_EVENT}", function() {{
        setStatic("Access to this dashboard was revoked.");
      }});
      sock.on("disconnect", function(reason) {{
        if (reason === "io client disconnect" || reason === "io server disconnect") setDisconnected(); else setReconnecting();
      }});
      sock.on("connect_error", function() {{
        if (!gotData) setError('Live updates unavailable (socket connection failed)');
      }});
    }}

    $(ID + '-btn-disconnect').addEventListener('click', function() {{
      if (sock) sock.disconnect();
      setDisconnected();
    }});
    $(ID + '-btn-connect').addEventListener('click', function() {{
      $(ID + '-status').textContent = 'CONNECTING';
      $(ID + '-status').style.color = '#64748b';
      $(ID + '-btn-disconnect').style.display = '';
      $(ID + '-btn-connect').style.display = 'none';
      if (sock) sock.connect(); else connectDashboard();
    }});

    connectDashboard();

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

@register_command("dashboard", summary="Open the live, multi-tab admin dashboard and config editor", category="Diagnostics", usage="dashboard")
async def handle_dashboard(ctx: CommandContext) -> str:
    """Display the live dashboard (OWUI socket.io transport)."""
    register_socket_handler()
    dash_id = "dash-" + secrets.token_hex(4)
    await ctx.emit_html(_build_dashboard_shell(dash_id))
    return (
        "Live dashboard rendered above. If no panel appears, enable iframe embeds "
        "(Open WebUI Settings → Interface → iframe sandbox allow same origin)."
    )
