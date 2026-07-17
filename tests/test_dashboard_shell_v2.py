"""Shell guards for the Live-sessions + Usage tabs (additive tab strategy)."""

from __future__ import annotations

import pytest
pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import _build_dashboard_shell


def test_tab_order_and_health_rename():
    html = _build_dashboard_shell("dash-v2")
    live = html.index('data-tab="live"')
    stats = html.index('data-tab="usage"')
    health = html.index('data-tab="health"')
    system = html.index('data-tab="system"')
    assert live < stats < health < system
    assert 'id="dash-v2-tab-health"' in html
    assert 'id="dash-v2-tab-live"' in html
    assert 'id="dash-v2-tab-usage"' in html


def test_internal_ids_unchanged_by_rename():
    html = _build_dashboard_shell("dash-v2")
    for marker in ("-req-val", "-tool-val", "-sess-val", "-rq-val", "-rl-section",
                   "-models-section", "-health-section", "-workers-section"):
        assert f"dash-v2{marker}" in html


def test_live_sessions_markers():
    html = _build_dashboard_shell("dash-v2")
    for marker in ("-sessions-panel", "-nameseg", "-keep", "-ls-active", "-ls-cost",
                   'data-mode="slugs"', "Keep completed", 'value="10" selected',
                   "sessBadge", "renderLiveSessions", "sessions_live",
                   "tok3(r2.tokens_in, r2.tokens_cached, r2.tokens_out)",
                   "fmtTok(tcached) + ' cached / '"):
        assert marker in html


def test_usage_tab_markers():
    html = _build_dashboard_shell("dash-v2")
    for marker in ("-us-range", "-us-tasks", "-us-cards", "-us-chart", "-us-models",
                   "-us-users", "-us-system", "Incl. task requests",
                   "share = share of cost", "top spenders", "usage_stats",
                   "PIPE_DASHBOARD_USAGE_COLLECT", "vs previous period of equal length",
                   'data-range="30d"', "usApplyRetention", "renderSystemCards",
                   "state.usLast", "usSpark(buckets, 'tools'",
                   "-us-refresh", "Auto-refresh", "usScheduleRefresh",
                   "tok3(r.tokens_in, r.tokens_cached, r.tokens_out)",
                   "cached input tokens shown in the middle",
                   "fmtTok(cards.tokens.cached)", "toolsCell(r.tools, r.tools_failed)"):
        assert marker in html


def test_sortable_searchable_tables_and_chart():
    html = _build_dashboard_shell("dash-v2")
    # Reusable sort/search controller drives all three tables (D).
    assert "makeSortTable" in html
    for m in ("liveTable", "modelsTable", "usersTable"):
        assert m in html
    assert "th-sort" in html and "tbl-search" in html
    assert "toolsCell" in html      # shared ok/failed tools cell (C)
    assert "ticksFor" in html and "usBindHover" in html  # nice-tick axes + hover tooltip
    # Live 'Keep completed' extended to hours (E).
    for opt in ('value="60">1 hour', 'value="120">2 hours', 'value="180">3 hours'):
        assert opt in html
    # 30d segment no longer clips (A).
    assert "flex-shrink: 0" in html


def test_footer_legend_and_models_breakdown():
    html = _build_dashboard_shell("dash-v2")
    # Footer abbreviations legend sits above the version line (item 3).
    assert "footer-legend" in html
    assert "zero-data-retention" not in html  # ZDR dropped: no ZDR marker is ever rendered
    assert "tools succeeded / failed" in html
    assert "&#10003; / &#10007;" in html
    # Health "Models loaded" folds Text/Image/Video into one sub-line (item 7);
    # the stray middot-prefixed standalone cards are gone.
    assert "modParts" in html
    assert "\\u00b7 Text / chat" not in html
    assert "\\u00b7 Image" not in html
    assert "\\u00b7 Video" not in html


def test_config_tab_between_storage_and_about():
    html = _build_dashboard_shell("dash-v2")
    storage = html.index('data-tab="storage"')
    config = html.index('data-tab="config"')
    about = html.index('data-tab="about"')
    assert storage < config < about
    assert 'id="dash-v2-tab-config"' in html
    assert 'id="dash-v2-cfgroot"' in html


def test_config_tab_wired_live_not_embedded():
    html = _build_dashboard_shell("dash-v2")
    assert 'callAction("config_get"' in html
    assert 'callAction("config_set"' in html
    assert "tab === 'config'" in html and "cfgFetch()" in html
    assert ".cfgroot .item" in html
    assert ".cfgroot{--bg:transparent" in html and "--ink:var(--text)" in html
    assert "REASONING_EFFORT" not in html and "MODEL_ID" not in html


def test_config_tab_overlay_affordances_and_no_dangling():
    html = _build_dashboard_shell("dash-v2")
    assert "Not documented yet" in html
    assert "Another administrator changed" in html
    assert "driftnote" in html
    assert "advToggle" not in html
    assert "v.advanced" not in html and "needs_review" not in html


def test_config_secret_control_reads_secret_set():
    import re

    html = _build_dashboard_shell("dash-v2")
    assert re.search(r"secret_set\s*\?\s*[^:]*configured\s*—\s*type to replace", html)
    assert re.search(r"secret_set\s*\?\s*['\"]?«configured»", html)
    assert "not set — type to set" in html


def test_shell_only_calls_registered_actions():
    import re

    from open_webui_openrouter_pipe.plugins.pipe_dashboard import actions

    html = _build_dashboard_shell("dash-v2")
    called = set(re.findall(r'callAction\(\s*["\']([^"\']+)["\']', html))
    assert called
    unregistered = called - set(actions.ACTIONS)
    assert not unregistered, f"shell calls unregistered actions: {sorted(unregistered)}"


def test_config_live_update_wiring():
    import re

    from open_webui_openrouter_pipe.plugins.pipe_dashboard.config_tab_assets import CONFIG_TAB_JS

    html = _build_dashboard_shell("dash-v2")
    assert 'sock.on("openrouter:pipe_dashboard:config"' in html
    assert "if (cfgLoaded && data.cfgRev != null) cfgOnEvent(data.cfgRev)" in html
    assert re.search(r"cfgOnEvent\s*=\s*function\(rev\)", CONFIG_TAB_JS)
    assert not re.search(r"\b(var|let|const|function)\s+cfgOnEvent\b", CONFIG_TAB_JS)
    assert "rev<=REV" in CONFIG_TAB_JS and "rev<=lastSeenRev" in CONFIG_TAB_JS
    assert "function quietReload(" in html
    assert "inflightSave" in html and "lastSeenRev" in html
    assert "Nothing is saved until you confirm." in html
    assert "Nothing is written" not in html
    assert "Discard changes?" in html and "cfgArmed" in html


def test_update_tab_markers():
    html = _build_dashboard_shell("dash-v2")
    config = html.index('data-tab="config"')
    update = html.index('data-tab="update"')
    about = html.index('data-tab="about"')
    assert config < update < about
    assert 'id="dash-v2-tab-update"' in html
    assert "updLoaded" in html and "updFetch" in html
    for marker in ("upd-installed", "upd-latest", "upd-snapshots", "upd-modal", "upd-notes"):
        assert marker in html
    assert "update_check" in html and "update_apply" in html
    assert "update_restore" in html and "update_snapshot_delete" in html
    for fn in ("updFetch", "updRender", "updOpenModal", "updModalRows", "updRunWrite", "updSetBusy",
               "updNotesHtml", "updFetchFailed", "updEnterTab"):
        assert html.count(fn) >= 2, f"{fn} is defined but never called"
    assert "upd-notes-summary" in html
    assert "Reinstall" in html
    assert "Checking" in html
    assert "Confirm restore" in html and "Confirm delete" in html
    assert "this worker:" in html
    import re as _re

    native_dialogs = _re.findall(r"(?:(?<![\w$.])|window\.)(?:confirm|alert|prompt)\s*\(", html)
    assert not native_dialogs, (
        f"native dialog calls found in the dashboard shell: {native_dialogs} - "
        "confirm/alert/prompt are silently blocked inside Open WebUI's sandboxed "
        "iframe (no allow-modals); use inline arm/confirm UI instead"
    )
