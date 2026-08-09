"""Shell guards for the Live-sessions + Usage tabs (additive tab strategy)."""

from __future__ import annotations

import pytest
pytest.importorskip("open_webui_openrouter_pipe.plugins.pipe_dashboard")

from open_webui_openrouter_pipe.plugins.pipe_dashboard._socketio_client import SOCKETIO_UMD
from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands import dashboard_cmd
from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import _build_dashboard_shell


def _own_js(dash_id: str = "dash-v2") -> str:
    """The shell with the vendored socket.io client removed.

    The bundled client contains ``emitReserved(\"heartbeat\")``, so a bare
    ``\"heartbeat\" in html`` passes with no heartbeat code written at all. Excising it
    leaves only JS this repo authored.
    """
    return _build_dashboard_shell(dash_id).replace(SOCKETIO_UMD, "")


def _js_fn_body(js: str, name: str) -> str:
    """The body of a JS function declaration, by brace matching.

    Asserting that an identifier *appears* says nothing about what it does: a
    ``freshToken`` that memoises its first read satisfies every name-based check while
    replaying the render-time token forever.
    """
    start = js.index("function " + name)
    open_brace = js.index("{", start)
    depth = 0
    for i in range(open_brace, len(js)):
        if js[i] == "{":
            depth += 1
        elif js[i] == "}":
            depth -= 1
            if depth == 0:
                return js[open_brace + 1:i]
    raise AssertionError(f"unbalanced braces in {name}")


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


# ── Socket session lifetime: heartbeat, recoverable denial, fresh credentials ──


def test_shell_heartbeats_within_owui_session_timeout():
    """OWUI reaps a session 120s after its last heartbeat; the panel must beat that.

    Without this the pool entry dies while the socket is still connected, identity stops
    resolving, and the reauthorization sweep evicts an authorized admin.
    """
    import re

    js = _own_js()
    # Containment, not co-presence: an emit sitting just after the setInterval call
    # fires once on connect and never again, and the session is reaped at 120s.
    beat = re.search(r"hbTimer = setInterval\(function\s*\(\)\s*\{(.*?)\}\s*,\s*(\d+)\)", js, re.S)
    assert beat is not None, "heartbeat interval not found"
    assert 'sock.emit("heartbeat"' in beat.group(1), "the emit is not inside the interval callback"
    assert int(beat.group(2)) == dashboard_cmd._PD_HEARTBEAT_MS
    assert js.count('sock.emit("heartbeat"') == 1


def test_heartbeat_timer_is_cleared_on_every_path_that_ends_the_socket():
    """One arm site, and a clear at each of the three places the socket can stop.

    A reconnect re-runs the connect handler on the same socket, so the arm site must
    clear first or every reconnect leaks another interval.
    """
    js = _own_js()
    for site, window in (
        ("connect re-arm", js[js.index('sock.on("connect"'):js.index("hbTimer = setInterval(")]),
        ("disconnect handler", js[js.index('sock.on("disconnect"'):][:300]),
        ("manual disconnect button", js[js.index("'-btn-disconnect').addEventListener"):][:300]),
    ):
        assert "clearInterval(hbTimer)" in window, f"heartbeat timer not cleared at: {site}"


def test_heartbeat_period_beats_the_owui_reaper():
    """Pinned to OWUI's SESSION_POOL_TIMEOUT of 120s, with room for a missed beat."""
    assert dashboard_cmd._PD_HEARTBEAT_MS * 2 <= 120_000


def test_denied_hangs_up_before_offering_reconnect():
    """``sock.connect()`` is a no-op on a connected socket.

    A denial only removes the sid from the viewers room; the socket stays up. Showing the
    Connect button without disconnecting first yields a button that does nothing.
    """
    js = _own_js()
    i = js.index('sock.on("openrouter:pipe_dashboard:denied"')
    handler = js[i:i + 400]
    assert "sock.disconnect()" in handler
    assert "setRevoked(" in handler
    assert "setStatic(" not in handler
    assert handler.index("sock.disconnect()") < handler.index("setRevoked(")


def test_revoked_state_leaves_a_way_back():
    js = _own_js()
    i = js.index("function setRevoked(")
    body = js[i:i + 500]
    assert "'-btn-connect').style.display = '';" in body
    assert "'-btn-disconnect').style.display = 'none';" in body


def test_denial_message_does_not_claim_revocation_for_the_never_authorized():
    import re

    js = _own_js()
    arms = re.search(r'setRevoked\(gotData\s*\?\s*"([^"]+)"\s*:\s*"([^"]+)"\)', js, re.S)
    assert arms is not None, "the denial message is not selected by gotData"
    had_access, never_had = arms.group(1), arms.group(2)
    assert "revoked" in had_access
    assert "revoked" not in never_had, "a user who never had access is told it was revoked"
    assert "not authorized" in never_had


def test_credentials_are_read_at_handshake_not_at_render():
    """A captured token literal is replayed forever; a function form is re-invoked per
    handshake, so a rotated token reaches the server."""
    import re

    js = _own_js()
    assert "function freshToken()" in js
    # Bind the assertion to the handshake callback body. The old needle looked for
    # "auth: { token: token }", which the function form does not spell, so a callback
    # closing over the render-time token was invisible to it.
    handshake = re.search(r"auth:\s*function\s*\(cb\)\s*\{\s*cb\(\{\s*token:\s*([^}]+?)\s*\}\)", js)
    assert handshake is not None, "handshake auth is not a function form"
    assert handshake.group(1).strip().startswith("freshToken()"), "handshake replays a captured token"
    join = re.search(r'user-join",\s*\{\{?\s*auth:\s*\{\{?\s*token:\s*([^}]+?)\s*\}\}?', js)
    assert join is not None, "user-join payload not found"
    # startswith, not "in": `token || freshToken()` lets the render-time token win.
    assert join.group(1).strip().startswith("freshToken()"), "the captured token wins the fallback"

    # And the helper must actually read storage every call, holding nothing between them.
    body = _js_fn_body(js, "freshToken")
    assert 'localStorage.getItem("token")' in body, "freshToken does not read storage"
    assert re.search(r"[^=!<>]=(?!=)", body) is None, \
        "freshToken keeps state across calls, so a rotated token never reaches the server"


# ── Replayed panels must not silently pin the publisher ──


def test_autoconnect_is_gated_on_a_single_clock():
    """OWUI persists this shell into the chat message and re-runs it on every chat load.

    Before the socket lifetime was fixed, a replayed panel self-terminated in ~4 minutes
    because its session was reaped -- the very bug being fixed here. Fixing that removed
    the accidental bound, so every historical /dashboard message would pin the
    publisher's 2-second emit loop for the life of the browser tab.

    The baseline must be written and read by the SAME clock. A server-rendered timestamp
    compared against Date.now() fails silently in both directions: a client running fast
    opens a brand-new panel disconnected, and one running slow disables the bound.

    It must also outlive the tab. ``sessionStorage`` is per tab-session, so opening an old
    chat in a fresh tab finds no baseline, takes the never-seen branch, and reconnects
    every historical panel -- the failure the gate exists to prevent.
    """
    import re

    js = _own_js()
    assert "RENDERED_AT" not in js, "baseline still comes from the server clock"
    assert "sessionStorage" not in js, "a per-tab baseline does not survive a new tab"
    assert "localStorage.setItem(FIRST_SEEN_KEY" in js
    assert "localStorage.getItem(FIRST_SEEN_KEY" in js
    window = re.search(r"Date\.now\(\) - Number\(firstSeen\) < (\d+) \* 1000", js)
    assert window is not None, "freshness comparison not found, or not single-clock"
    assert int(window.group(1)) == dashboard_cmd._PD_AUTOCONNECT_MAX_AGE_S


def test_a_never_seen_panel_connects_and_a_stale_one_does_not():
    """Per-branch, not per-count. Counting call sites cannot tell which branch they are in,
    so moving the connect into the stale branch would pass a count-based test."""
    import re

    js = _own_js()
    branches = re.search(
        r"if \(firstSeen === null\) \{(.*?)\} else if \((.*?)\) \{(.*?)\} else \{(.*?)\n    \}",
        js, re.S)
    assert branches is not None, "the three-way freshness branch was not found"
    first, fresh, stale = branches.group(1), branches.group(3), branches.group(4)
    assert "localStorage.setItem" in first and first.count("connectDashboard();") == 1
    assert fresh.count("connectDashboard();") == 1 and "localStorage.setItem" not in fresh
    assert stale.count("connectDashboard();") == 0, "a stale panel still dials out"
    assert "setReplayed(" in stale
    # The baseline is written once, on first sighting only -- rewriting it on the fresh
    # branch turns the cap into a sliding window that never expires.
    assert js.count("localStorage.setItem") == 1


def test_the_stale_state_explains_itself():
    """The only unrequested transition into a non-live state, so it must write the notice.

    Every tab keeps its "Waiting for data..." placeholder, so a silent DISCONNECTED panel
    reads as simultaneously idle and waiting for data that will never arrive.
    """
    js = _own_js()
    body = js[js.index("function setReplayed("):][:600]
    assert "-notice').style.display = 'block'" in body
    assert "-notice').textContent = msg" in body
    assert "'-btn-connect').style.display = '';" in body
    assert "no longer updates on its own" in js


def test_autoconnect_window_is_short_enough_to_bound_a_replay():
    """Pinned against the publisher cadence it protects, not against its own literal."""
    from open_webui_openrouter_pipe.plugins.pipe_dashboard import dashboard_publisher

    assert dashboard_cmd._PD_AUTOCONNECT_MAX_AGE_S <= 900
    # Must still comfortably exceed the time between rendering and the first payload.
    assert dashboard_cmd._PD_AUTOCONNECT_MAX_AGE_S > dashboard_publisher._PD_PUBLISH_INTERVAL * 10


def test_each_panel_gets_a_distinct_high_entropy_id():
    """The id is the sessionStorage key for the freshness baseline.

    Two panels sharing an id makes a brand-new one inherit a stale baseline and open
    disconnected with no live data, so the width is load-bearing rather than cosmetic.
    Driven through the real handler rather than by reading the source, because in a
    compressed bundle there is no source file to read.
    """
    import asyncio
    import re
    from typing import Any, cast

    from open_webui_openrouter_pipe.plugins.pipe_dashboard.commands.dashboard_cmd import (
        handle_dashboard,
    )
    from open_webui_openrouter_pipe.plugins.pipe_dashboard.context import CommandContext

    emitted: list[str] = []

    class _Ctx(CommandContext):
        async def emit_html(self, html: str) -> None:
            emitted.append(html)

    async def _render() -> None:
        await handle_dashboard(
            _Ctx(pipe=cast(Any, object()), args="", user={}, metadata={}, event_emitter=None)
        )

    asyncio.run(_render())
    asyncio.run(_render())
    assert len(emitted) == 2

    ids = [re.search(r'var ID = "(dash-[0-9a-f]+)"', h) for h in emitted]
    assert all(i is not None for i in ids), "panel id not found in the rendered shell"
    first, second = ids[0].group(1), ids[1].group(1)  # type: ignore[union-attr]
    assert first != second, "two panels rendered with the same id"
    for value in (first, second):
        assert len(value.split("-", 1)[1]) >= 16, "panel id entropy reduced"
    assert "'owui_pd_first_' + ID" in emitted[0]


def test_the_internals_doc_describes_the_baseline_that_exists():
    """A doc that names the wrong mechanism is worse than one that says nothing.

    The bullet claimed the shell "stamps its render time" -- a server-side timestamp,
    which is precisely the design this changeset rejected for clock skew.
    """
    import pathlib

    doc = pathlib.Path(__file__).resolve().parents[1] / "docs" / "plugins_pipe_dashboard_internals.md"
    text = doc.read_text(encoding="utf-8")
    for forbidden in ("stamps its render time", "RENDERED_AT"):
        assert forbidden not in text, f"internals doc still claims a server render stamp: {forbidden!r}"
    assert "localStorage" in text, "internals doc does not name the store the shell uses"
