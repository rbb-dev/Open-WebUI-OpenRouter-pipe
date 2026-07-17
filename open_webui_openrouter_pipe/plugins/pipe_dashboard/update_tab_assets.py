"""CSS/JS assets for the pipe_dashboard Update tab.

Both constants are spliced verbatim into the dashboard shell: the CSS into the
single <style> block, the JS inside the main app IIFE (so it shares esc(),
callAction() and reportHeight() with the rest of the dashboard, exactly like
the config tab). Plain string constants — no f-string placeholders here.
"""

from __future__ import annotations

UPDATE_TAB_CSS = """
.upd-wrap { display: flex; flex-direction: column; gap: 12px; }
.upd-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
.upd-card { border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: 10px 12px; }
.upd-card h3 { margin: 0 0 8px 0; font-size: 13px; opacity: .8; }
.upd-row { display: flex; justify-content: space-between; gap: 10px; padding: 2px 0; font-size: 13px; }
.upd-k { opacity: .65; }
.upd-v { text-align: right; word-break: break-word; }
.upd-msg { font-size: 13px; min-height: 1em; }
.upd-msg.err { color: #d33; }
.upd-note { font-size: 12px; opacity: .75; margin: 6px 0; }
.upd-actions { display: flex; gap: 8px; margin-top: 8px; justify-content: flex-end; }
.upd-btn { padding: 4px 12px; border: 1px solid rgba(128,128,128,.45); border-radius: 6px;
  background: transparent; color: inherit; cursor: pointer; font-size: 13px;
  transition: background .15s ease, border-color .15s ease, transform .05s ease; }
.upd-btn.primary { border-color: #4a7dfc; }
.upd-btn.upd-armed { border-color: #b91c1c; background: rgba(185,28,28,.15); color: #b91c1c; font-weight: 600; }
.upd-btn:hover:not(:disabled) { border-color: #4a7dfc; background: rgba(74,125,252,.10); }
.upd-btn:active:not(:disabled) { transform: translateY(1px); }
.upd-btn:disabled { opacity: .5; cursor: default; }
.upd-msg.busy::before { content: ''; display: inline-block; width: 11px; height: 11px;
  margin-right: 6px; border: 2px solid rgba(128,128,128,.4); border-top-color: #4a7dfc;
  border-radius: 50%; animation: updspin .8s linear infinite; vertical-align: -1px; }
@keyframes updspin { to { transform: rotate(360deg); } }
.upd-badge { display: inline-block; padding: 0 6px; border-radius: 8px; font-size: 11px;
  border: 1px solid rgba(128,128,128,.45); margin-left: 6px; }
#upd-notes pre { max-height: 260px; overflow: auto; white-space: pre-wrap; font-size: 12px; }
.upd-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.upd-table th, .upd-table td { text-align: left; padding: 4px 6px; border-bottom: 1px solid rgba(128,128,128,.25); }
.upd-modal { position: fixed; inset: 0; background: rgba(0,0,0,.45); display: flex;
  align-items: center; justify-content: center; z-index: 40; }
.upd-modal-box { background: var(--upd-bg, #fff); color: inherit; border-radius: 10px;
  padding: 16px; min-width: 280px; max-width: 420px; border: 1px solid rgba(128,128,128,.4); }
.dark .upd-modal-box { --upd-bg: #1e1e24; }
.upd-check { display: flex; gap: 8px; align-items: center; font-size: 13px; margin-top: 8px; }
"""

UPDATE_TAB_JS = """
    var updData = null, updBusy = false, updLastFetch = 0;
    var UPD_ERRORS = {
      rate_limited: 'GitHub is rate-limiting update checks from this server.',
      offline: 'GitHub could not be reached from the server. Check its internet access or proxy, then press Check now.',
      no_matching_asset: 'The latest release does not include a bundle file for this install variant. Check the release assets on GitHub.',
      stale_rev: 'The function changed while this tab was open. The view has been refreshed \\u2014 please try again.',
      stale_snapshot: 'That snapshot changed since this list was loaded. The view has been refreshed \\u2014 please try again.',
      package_mode: 'This is a package/stub install. Update it by bumping the pinned version; this tab cannot modify it.',
      digest_mismatch: 'The downloaded file did not match the release checksum, so nothing was changed. Try again in a moment.',
      validation_failed: 'The downloaded bundle failed validation, so nothing was changed.',
      incompatible_owui: 'This release needs a newer Open WebUI than this server runs. Upgrade Open WebUI first.',
      update_in_progress: 'Another update is already running. Wait for it to finish, then press Check now.',
      disabled: 'Updates are switched off by the "Enable the Update tab" valve.',
      bad_repo_valve: 'The update repo valve is not a valid owner/repo value. Fix it in the Config tab.',
      repo_not_found: 'GitHub has no such repo or no releases for it. Check the update repo valve.',
      not_found: 'That snapshot no longer exists. The list has been refreshed.',
      'rate limited': 'Requests are limited to one per second. Give it a moment and try again.',
      'Too Many Requests': 'You are clicking faster than the dashboard allows. Give it a second and try again.',
      'Unauthorized': 'Your Open WebUI session is no longer valid here. Reload the page and sign in again.',
      'server_error': 'The server returned an unexpected response and nothing was changed. Try again.',
      'action failed': 'The action hit an unexpected server error and was not completed. Details are in the server log.',
      'unknown action': 'The server does not recognise this action \\u2014 the pipe was probably updated. Reload the dashboard.',
      'request unavailable': 'The server could not process this action \\u2014 reload the dashboard and try again.',
      exec_failed: 'The new code failed to load, so the pipe still runs the previous version. Nothing is broken.',
      unavailable: 'The update service is not ready on this worker yet.',
      forbidden: 'This action needs an Open WebUI admin account.',
      internal: 'Something unexpected went wrong and nothing was changed. Details are in the server log.'
    };
    function updEl(id) { return document.getElementById(id); }
    function updErrText(res) {
      var code = res.error || res.code;
      var base = UPD_ERRORS[code] || esc(String(code || 'error'));
      if (res.message) { base += ' (' + esc(String(res.message)) + ')'; }
      var reset = parseFloat(res.reset);
      if (reset > 0) {
        var resetEpoch = reset < 1e6 ? (Date.now() / 1000) + reset : reset;
        base += ' Try again after ' + updFmtDate(resetEpoch) + '.';
      }
      return base;
    }
    function updSetMsg(text, isErr, isBusy) {
      var m = updEl('upd-msg');
      if (m) {
        m.innerHTML = text || '';
        m.className = 'upd-msg' + (isErr ? ' err' : '') + (isBusy ? ' busy' : '');
      }
      reportHeight();
    }
    function updFmtDate(v) {
      if (v === null || v === undefined || v === '') return '\\u2014';
      try {
        var d = (typeof v === 'number') ? new Date(v * 1000) : new Date(v);
        if (isNaN(d.getTime())) return '\\u2014';
        return d.toLocaleString();
      } catch (e) { return '\\u2014'; }
    }
    function updRow(k, v) {
      return '<div class="upd-row"><span class="upd-k">' + k + '</span><span class="upd-v">' + v + '</span></div>';
    }
    var updRetried = false;
    function updFetchFailed(res) {
      var code = String(res.error || res.code || '');
      var text = updErrText(res);
      if (code === 'unavailable') {
        text += updRetried
          ? ' Press Check now in a few seconds.'
          : ' Retrying automatically\\u2026';
      }
      updSetMsg(text, true);
      if (!updData) {
        updEl('upd-installed-body').innerHTML = '<div class="upd-note">Not available yet.</div>';
        updEl('upd-latest-body').innerHTML = '<div class="upd-note">Press Check now to retry.</div>';
      }
      reportHeight();
      if (code === 'unavailable' && !updRetried) {
        updRetried = true;
        setTimeout(function () { updFetch(true); }, 1500);
      }
    }
    function updEnterTab() {
      if (updBusy || Date.now() - updLastFetch < 1000) return;
      updFetch();
    }
    function updFetch(force, quiet) {
      if (updBusy) return;
      updLastFetch = Date.now();
      updSetBusy(true);
      if (!quiet) updSetMsg('Checking for updates\\u2026', false, true);
      callAction('update_check', { force: !!force }).then(function (r) {
        updSetBusy(false);
        if (!r || r.ok !== true) {
          if (!quiet) updFetchFailed({ code: String((r && (r.error || r.detail)) || 'server_error') });
          return;
        }
        var res = r.result || {};
        if (res.error) { if (!quiet) updFetchFailed(res); return; }
        updData = res;
        if (!quiet) updSetMsg('');
        updRender();
      }).catch(function (reason) {
        updSetBusy(false);
        if (quiet) return;
        updSetMsg(reason === 'no token'
          ? 'Could not read the sign-in token. If reloading this page does not help, enable '
            + 'Settings \\u2192 Interface \\u2192 iframe sandbox: allow same origin \\u2014 the '
            + 'dashboard needs it to authenticate.'
          : 'Could not reach the Open WebUI server. Check your connection and try again.', true);
      });
    }
    function updNotesHtml(notes) {
      var out = esc(String(notes || ''));
      out = out.replace(/\\[([^\\]\\n]{1,120})\\]\\((https:\\/\\/[^)\\s]{1,300})\\)/g,
        '<code>$1</code>');
      out = out.replace(/`([^`\\n]{1,120})`/g, '<code>$1</code>');
      out = out.replace(/^## (.{1,160})$/gm, '<strong>$1</strong>');
      return out;
    }
    function updRender() {
      var d = updData || {};
      if (d.enabled === false) {
        updEl('upd-installed-body').innerHTML = 'Updates are disabled by the admin valve.';
        updEl('upd-latest-body').innerHTML = '';
        var cb0 = updEl('upd-check-now');
        if (cb0) cb0.style.display = 'none';
        reportHeight();
        return;
      }
      var cb1 = updEl('upd-check-now');
      if (cb1) cb1.style.display = '';
      var inst = d.installed || {};
      var instHtml = updRow('Version', esc(String(inst.version || '?')))
        + updRow('Install', esc(String(inst.mode || '?'))
            + (inst.mode === 'bundle' ? '<span class="upd-badge">' + (inst.compressed ? 'compressed' : 'flat') + '</span>' : ''));
      if (inst.mode !== 'bundle') {
        instHtml += '<div class="upd-note">Package/stub installs update by bumping the pinned requirement; this tab stays read-only.</div>';
      }
      updEl('upd-installed-body').innerHTML = instHtml;

      var latest = d.latest, latestHtml = '';
      if (latest) {
        latestHtml = updRow('Version', esc(String(latest.version || '?')))
          + updRow('Released', updFmtDate(latest.published_at));
        var assets = latest.assets || {};
        var mine = inst.compressed ? assets.compressed : assets.flat;
        if (mine && mine.size) { latestHtml += updRow('Size', esc(String(Math.round(mine.size / 1024))) + ' KiB'); }
      } else {
        latestHtml = '<div class="upd-note">No release information yet.</div>';
      }
      latestHtml += updRow('Last checked', updFmtDate(d.checked_at));
      if (d.last_check_error) {
        latestHtml += '<div class="upd-msg err">Check failing since ' + updFmtDate(d.last_check_error.ts)
          + ': ' + updErrText(d.last_check_error) + '</div>';
      }
      latestHtml += updRow('Updates from', esc(String(d.repo || ''))
        + (d.repo_is_default === false ? '<span class="upd-badge">fork</span>' : ''));
      var auto = d.auto || {};
      if (auto.enabled) {
        var autoTxt = 'on';
        if (auto.delay_hours === 0) { autoTxt += ' \\u2014 eligible immediately (no quarantine)'; }
        else if (d.update_available && auto.eligible_at) { autoTxt += ' \\u2014 eligible from ' + updFmtDate(auto.eligible_at); }
        if (auto.this_worker && auto.this_worker.paused_for_version) {
          autoTxt += ' \\u2014 paused for ' + esc(String(auto.this_worker.paused_for_version))
            + ' on this worker after ' + esc(String(auto.this_worker.code))
            + ' (apply manually or restart to re-arm)';
        }
        if (auto.last_success) {
          autoTxt += ' \\u2014 last auto-update ' + esc(String(auto.last_success.from_version || '?'))
            + ' \\u2192 ' + esc(String(auto.last_success.to_version || '?'))
            + ' at ' + updFmtDate(auto.last_success.ts);
        }
        latestHtml += updRow('Auto-update', autoTxt);
      } else {
        latestHtml += updRow('Auto-update', 'off');
      }
      updEl('upd-latest-body').innerHTML = latestHtml;

      var applyBtn = updEl('upd-apply-btn');
      var sameVersion = !!(latest && inst.version && latest.version === inst.version);
      var canApply = !!(inst.mode === 'bundle' && latest && latest.supported !== false
        && (d.update_available || sameVersion));
      applyBtn.style.display = canApply ? '' : 'none';
      applyBtn.textContent = d.update_available ? 'Update...' : 'Reinstall...';
      if (d.no_matching_asset) {
        applyBtn.style.display = 'none';
        updSetMsg(UPD_ERRORS.no_matching_asset, true);
      }

      var notes = (latest && latest.notes) ? latest.notes : '';
      updEl('upd-notes-body').innerHTML = notes ? updNotesHtml(notes) : 'changelog unavailable';
      var summary = updEl('upd-notes-summary');
      if (summary) {
        summary.textContent = 'Changelog' + (latest && latest.version ? ' — v' + latest.version : '');
      }

      var snaps = d.snapshots || [];
      if (!snaps.length) {
        updEl('upd-snapshots').innerHTML = 'No snapshots yet.';
      } else {
        var rows = '';
        for (var i = 0; i < snaps.length; i++) {
          var s = snaps[i];
          rows += '<tr><td>' + esc(String(s.version || '?')) + '</td><td>' + updFmtDate(s.ts)
            + '</td><td>' + (s.size ? Math.round(s.size / 1024) + ' KiB' : '\\u2014')
            + '</td><td>' + esc(String(s.actor_name || s.actor || '?')) + '</td>'
            + '<td><button class="upd-btn" data-upd-restore="' + esc(String(s.file_id)) + '">Restore</button> '
            + '<button class="upd-btn" data-upd-delete="' + esc(String(s.file_id))
            + '" data-upd-sha="' + esc(String(s.sha256 || '')) + '">Delete</button></td></tr>';
        }
        updEl('upd-snapshots').innerHTML =
          '<table class="upd-table"><tr><th>Version</th><th>Date</th><th>Size</th><th>Actor</th><th></th></tr>'
          + rows + '</table>';
      }
      reportHeight();
    }
    function updModalRows() {
      var d = updData || {}, inst = d.installed || {}, latest = d.latest || {};
      var assets = latest.assets || {};
      var chosen = updEl('upd-compressed').checked ? assets.compressed : assets.flat;
      updEl('upd-modal-body').innerHTML =
        updRow('From', esc(String(inst.version || '?')))
        + updRow('To', esc(String(latest.version || '?')))
        + updRow('Size', chosen && chosen.size ? Math.round(chosen.size / 1024) + ' KiB' : '\\u2014');
    }
    function updOpenModal() {
      var d = updData || {}, inst = d.installed || {}, latest = d.latest || {};
      var assets = latest.assets || {};
      var cb = updEl('upd-compressed');
      cb.checked = !!inst.compressed;
      cb.disabled = false;
      var note = 'The variant choice applies to this update only.';
      if (!assets.flat) { cb.checked = true; cb.disabled = true; note = 'Only the compressed bundle exists on this release.'; }
      if (!assets.compressed) { cb.checked = false; cb.disabled = true; note = 'Only the flat bundle exists on this release.'; }
      if (latest && inst.version === latest.version) {
        note = 'Reinstalls the current version — use the checkbox to switch the bundle variant. ' + note;
      }
      updEl('upd-modal-note').textContent = note
        + ' This worker pauses briefly during install (up to ~90s); the dashboard may blink.';
      updModalRows();
      updEl('upd-modal').style.display = 'flex';
      reportHeight();
    }
    function updCloseModal() { updEl('upd-modal').style.display = 'none'; }
    function updSetBusy(b) {
      updBusy = b;
      var ids = ['upd-apply-btn', 'upd-check-now', 'upd-modal-confirm'];
      for (var i = 0; i < ids.length; i++) { var e = updEl(ids[i]); if (e) e.disabled = b; }
      var snapBtns = updEl('upd-snapshots');
      if (snapBtns) {
        var btns = snapBtns.getElementsByTagName('button');
        for (var j = 0; j < btns.length; j++) { btns[j].disabled = b; }
      }
      var checkBtn = updEl('upd-check-now');
      if (checkBtn) { checkBtn.textContent = b ? 'Working\\u2026' : 'Check now'; }
    }
    function updRunWrite(action, args, okText, busyText) {
      if (updBusy) return;
      updSetBusy(true);
      updSetMsg(busyText || 'Working\\u2026', false, true);
      callAction(action, args).then(function (r) {
        updSetBusy(false);
        if (!r || r.ok !== true) {
          updSetMsg(updErrText({ error: String((r && (r.error || r.detail)) || 'server_error') }), true);
          updFetch(true, true);
          return;
        }
        var res = r.result || {};
        if (res.error) { updSetMsg(updErrText(res), true); }
        else { updSetMsg(okText.replace('%v', esc(String(res.to_version || '')))); }
        updFetch(true, true);
      }).catch(function (reason) {
        updSetBusy(false);
        updSetMsg(reason === 'no token'
          ? 'Could not read the sign-in token. If reloading this page does not help, enable '
            + 'Settings \\u2192 Interface \\u2192 iframe sandbox: allow same origin \\u2014 the '
            + 'dashboard needs it to authenticate.'
          : 'Could not reach the Open WebUI server \\u2014 the operation may still be running. Press Check now in a minute.', true);
        updFetch(true, true);
      });
    }
    (function updBind() {
      updEl('upd-check-now').addEventListener('click', function () { updFetch(true); });
      updEl('upd-apply-btn').addEventListener('click', updOpenModal);
      updEl('upd-compressed').addEventListener('change', updModalRows);
      updEl('upd-modal-cancel').addEventListener('click', updCloseModal);
      updEl('upd-modal-confirm').addEventListener('click', function () {
        updCloseModal();
        updRunWrite('update_apply',
          { rev: (updData || {}).rev, compressed: updEl('upd-compressed').checked },
          'Updated to %v \\u2014 reload the dashboard to load the new UI.',
          'Installing\\u2026 this worker pauses briefly (up to ~90 seconds); the dashboard may blink.');
      });
      updEl('upd-snapshots').addEventListener('click', function (ev) {
        var t = ev.target || {};
        if (!t.getAttribute) return;
        var rid = t.getAttribute('data-upd-restore');
        var did = t.getAttribute('data-upd-delete');
        if (!rid && !did) return;
        if (updBusy) return;
        if (t.getAttribute('data-upd-armed') !== '1') {
          t.setAttribute('data-upd-armed', '1');
          t.setAttribute('data-upd-label', t.textContent);
          t.textContent = rid ? 'Confirm restore' : 'Confirm delete';
          t.className += ' upd-armed';
          setTimeout(function () {
            if (document.body.contains(t) && t.getAttribute('data-upd-armed') === '1') {
              t.removeAttribute('data-upd-armed');
              t.textContent = t.getAttribute('data-upd-label') || (rid ? 'Restore' : 'Delete');
              t.className = t.className.replace(' upd-armed', '');
            }
          }, 5000);
          return;
        }
        t.removeAttribute('data-upd-armed');
        t.textContent = t.getAttribute('data-upd-label') || (rid ? 'Restore' : 'Delete');
        t.className = t.className.replace(' upd-armed', '');
        if (rid) {
          updRunWrite('update_restore', { rev: (updData || {}).rev, file_id: rid },
            'Restored %v \\u2014 reload the dashboard to load the matching UI.',
            'Restoring\\u2026 this worker pauses briefly (up to ~90 seconds); the dashboard may blink.');
        } else {
          updRunWrite('update_snapshot_delete',
            { file_id: did, sha256: (t.getAttribute('data-upd-sha') || '') },
            'Snapshot deleted.', 'Deleting snapshot\\u2026');
        }
      });
    })();
"""
