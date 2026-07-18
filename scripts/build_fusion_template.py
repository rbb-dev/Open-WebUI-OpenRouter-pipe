#!/usr/bin/env python3
"""Prune scripts/fusion-live.html into the production template
and inject it into fusion_embed.py between the GENERATED markers.

    python scripts/build_fusion_template.py
"""

from __future__ import annotations

import base64
import hashlib
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_HTML = PROJECT_ROOT / "scripts" / "fusion-live.html"
TARGET_PY = PROJECT_ROOT / "open_webui_openrouter_pipe" / "streaming" / "fusion_embed.py"

BEGIN_MARKER = "# === BEGIN GENERATED TEMPLATE ==="
END_MARKER = "# === END GENERATED TEMPLATE ==="

PRODUCTION_BOOTSTRAP = r'''
  var EVENTS = /*__FUSION_EVENTS_JSON__*/[];
  var _FUSION_FINAL = /*__FUSION_FINAL__*/false;
  try { EVENTS.forEach(function(ev){ try { window.FusionUI.push(ev); } catch(_e){} }); } catch(_e){}
  try { if (_FUSION_FINAL && EVENTS.length && _clockStart && _clockFrozen == null) { if (_clockTimer) { clearInterval(_clockTimer); _clockTimer = null; } _clockStart = 0; var _elx = document.getElementById('elapsed'); if (_elx) _elx.textContent = '—'; } } catch(_e){}
  (function fusionLiveSocket(){
    var msgId = null;
    try {
      var host = window.frameElement;
      var wrap = host && host.closest && host.closest('[id^="message-"]');
      if (wrap) msgId = wrap.id.slice("message-".length);
    } catch(_e){}
    var token = null;
    try { token = localStorage.getItem("token"); } catch(_e){}
    if (!token) return;
    function connect(){
      if (typeof io === "undefined") return;
      try {
        var origin = "";
        try { origin = parent.location.origin; } catch(_e){ origin = ""; }
        var sock = io(origin || undefined, { reconnection: true, reconnectionDelay: 1000, reconnectionDelayMax: 5000, randomizationFactor: 0.5, path: "/ws/socket.io", transports: ["websocket", "polling"], auth: { token: token } });
        sock.on("connect", function(){ try { sock.emit("user-join", { auth: { token: token } }); } catch(_e){} });
        sock.on("events", function(payload){
          try {
            if (!payload || typeof payload !== "object") return;
            if (msgId && payload.message_id && payload.message_id !== msgId) return;
            var d = payload.data;
            if (d && d.type === "fusion:event" && d.data && d.data.event) { window.FusionUI.push(d.data.event); }
          } catch(_e){}
        });
      } catch(_e){}
    }
    connect();
  })();
'''

SENTINEL = "/*__FUSION_EVENTS_JSON__*/[]"
SOCKETIO_VENDOR = PROJECT_ROOT / "scripts" / "vendor" / "socket.io.min.js"
SOCKETIO_SHA384 = "sha384-kzavj5fiMwLKzzD1f8S7TeoVIEi7uKHvbTA3ueZkrzYq75pNQUiUi6Dy98Q3fxb0"


def cut_between(html: str, start: str, end: str, *, must_contain: list[str],
                keep_end: bool = True, label: str = "") -> str:
    """Remove ``start``..``end``, asserting unique anchors and expected tokens."""
    si = html.find(start)
    assert si != -1, f"[{label}] start anchor not found: {start!r}"
    assert html.find(start, si + 1) == -1, f"[{label}] start anchor not unique: {start!r}"
    ei = html.find(end, si + len(start))
    assert ei != -1, f"[{label}] end anchor not found after start: {end!r}"
    removed = html[si:ei]
    for tok in must_contain:
        assert tok in removed, f"[{label}] expected token missing from removed region: {tok!r}"
    tail = html[ei:] if keep_end else html[ei + len(end):]
    return html[:si] + tail


def replace_once(html: str, old: str, new: str, *, label: str = "") -> str:
    """Replace ``old`` with ``new``, asserting ``old`` occurs exactly once."""
    count = html.count(old)
    assert count == 1, f"[{label}] expected exactly 1 occurrence of anchor, found {count}: {old!r}"
    return html.replace(old, new, 1)


def assert_absent(html: str, tokens: list[str]) -> None:
    for tok in tokens:
        assert tok not in html, f"OMIT token still present in generated template: {tok!r}"


def assert_present(html: str, tokens: list[str]) -> None:
    for tok in tokens:
        assert tok in html, f"KEEP token missing from generated template: {tok!r}"


def strip_comments(html: str) -> str:
    """Drop HTML/CSS/JS comments, keeping the events sentinel. Runs before socket.io is inlined."""
    sentinel = "\x00FUSION_SENTINEL\x00"
    sentinel_names = "\x00FUSION_NAMES_SENTINEL\x00"
    html = html.replace("/*__FUSION_EVENTS_JSON__*/", sentinel)
    html = html.replace("/*__FUSION_NAMES_JSON__*/", sentinel_names)
    html = html.replace("/*__FUSION_FINAL__*/", "FUSIONFINALSENTINEL")
    html = re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)
    html = re.sub(r"/\*.*?\*/", "", html, flags=re.DOTALL)
    html = re.sub(r"[ \t]*(?<![:/])//[^\n]*", "", html)
    html = html.replace(sentinel, "/*__FUSION_EVENTS_JSON__*/")
    html = html.replace(sentinel_names, "/*__FUSION_NAMES_JSON__*/")
    html = html.replace("FUSIONFINALSENTINEL", "/*__FUSION_FINAL__*/")
    return re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", html)


def inject_socketio(html: str) -> str:
    src = SOCKETIO_VENDOR.read_text(encoding="utf-8")
    digest = "sha384-" + base64.b64encode(hashlib.sha384(src.encode("utf-8")).digest()).decode("ascii")
    assert digest == SOCKETIO_SHA384, f"vendored socket.io integrity mismatch: {digest}"
    assert "</script" not in src, "vendored socket.io contains </script which would break the inline tag"
    src = re.sub(r"//# sourceMappingURL=\S*", "", src)
    return replace_once(html, "</head>", "<script>" + src + "</script></head>", label="28 socketio-inline")


def build_template(html: str) -> str:
    html = cut_between(
        html,
        "/* ---- demo controls (NOT part of the real interface) ---- */",
        "@media (max-width:560px)",
        must_contain=[".demo-bar", ".demo-btn", ".demo-tag"],
        label="1 demo-css",
    )

    html = cut_between(
        html,
        "<!-- demo-only chrome:",
        '<div class="fusion">',
        must_contain=['id="demoBg"', 'id="themeBtn"', 'id="replayBtn"', 'id="panelSel"'],
        label="3 demo-chrome",
    )

    html = cut_between(
        html,
        "<!-- ============================================================\n     FUSION DEMO DATA (placeholder)",
        "</script>",
        must_contain=['id="fusion-feed"'],
        keep_end=False,
        label="5 captured-data",
    )

    html = cut_between(
        html,
        "  // Optional parent-frame transport: postMessage",
        "  /* ============================================================\n     LIVE THEME MIRROR",
        must_contain=["__fusion", "addEventListener('message'"],
        label="8 postmessage",
    )

    html = cut_between(
        html,
        "  /* ============================================================\n     DEMO-ONLY:",
        "})();\n</script>\n</body>\n</html>",
        must_contain=["function runFeed", "buildPanel", "updateThemeLabel", "runFeed();"],
        label="9 demo-driver",
    )

    html = replace_once(
        html,
        "  var DATA = JSON.parse(document.getElementById('fusion-feed').textContent);",
        "  var DATA = {};",
        label="10 DATA",
    )

    html = replace_once(
        html,
        "      document.documentElement.classList.toggle('dark', isDark);\n      updateThemeLabel();\n",
        "      document.documentElement.classList.toggle('dark', isDark);\n",
        label="11 syncTheme",
    )

    html = replace_once(
        html,
        "  function esc(s){ var d=document.createElement('div'); d.textContent = (s==null?'':String(s)); return d.innerHTML; }",
        r"""  function esc(s){return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');}""",
        label="12 esc",
    )

    html = replace_once(
        html,
        r"""  function safeHref(h){ h=h.trim(); return /^(https?:\/\/|\/|#|mailto:)/i.test(h) ? h : '#'; }""",
        r"""  function safeHref(h){h=String(h==null?'':h).trim(); if(/["'<>\s]/.test(h)) return '#'; return /^(https?:\/\/|\/|#|mailto:)/i.test(h) ? h : '#';}""",
        label="13 safeHref",
    )

    fusion_assign = (
        "  window.FusionUI = {\n"
        "    push: push,\n"
        "    reset: function(){ _judge=null; _fusionIdx=null; _live={}; _finalStarted=false; preambleBuf=''; finalBuf=''; "
        "_collected={panels:[],analysis:null,judge:null,usage:null,model:null}; resetUI(); }\n"
        "  };\n"
    )
    html = replace_once(
        html,
        fusion_assign,
        fusion_assign + PRODUCTION_BOOTSTRAP,
        label="15 bootstrap",
    )

    html = replace_once(
        html,
        "  var DATA = {};",
        "  var DATA = {};\n  var panelCount = 0;",
        label="19 declare panelCount",
    )

    html = replace_once(
        html,
        "    document.getElementById('rosterCount').textContent = roster.children.length;",
        "    document.getElementById('rosterCount').textContent = roster.children.length;\n"
        "    panelCount = roster.children.length;\n"
        "    document.getElementById('rosterNote').textContent = panelCount + ' models deliberating in parallel';",
        label="20 sync panelCount",
    )

    html = replace_once(
        html,
        ".md{font-size:13.5px;line-height:1.62;color:var(--text-dim);max-width:72ch;}\n"
        '/* "Wide" toggle (html.wide) lifts the readable measure so long prose runs edge-to-edge */\n'
        "html.wide .md{max-width:none;}",
        ".md{font-size:13.5px;line-height:1.62;color:var(--text-dim);overflow-wrap:break-word;word-break:break-word;}\n"
        ".md pre,.md table{overflow-wrap:normal;word-break:normal;}",
        label="22 always-wide+wrap",
    )

    html = replace_once(
        html,
        '\n    <button class="ctl" id="wideBtn" type="button" title="Toggle wide / readable width" aria-pressed="false">\n'
        '      <span class="ctl-dot"></span>Wide\n'
        "    </button>",
        "",
        label="23 wide-button",
    )

    html = replace_once(
        html,
        "WIDE / READABLE WIDTH TOGGLE + MASTER COPY",
        "MASTER COPY",
        label="24a wide-comment",
    )

    html = cut_between(
        html,
        "  function applyWide(on){",
        "  (function initCopyFinal(){",
        must_contain=["function initWide", "fusionWide", "applyWide(on)"],
        label="24b wide-js",
    )

    html = replace_once(
        html,
        "  var finalBuf = '';",
        "  var finalBuf = ''; var _finalRaf = 0; var _hRaf = 0; var _frozenLen = 0; var _lastH = 0;",
        label="25 raf-state",
    )

    html = replace_once(
        html,
        "  function reportHeight(){\n"
        "    try{\n"
        "      var h = document.documentElement.scrollHeight;\n"
        "      parent.postMessage({type:'iframe:height', height:h}, '*');\n"
        "    }catch(e){}\n"
        "  }",
        "  function reportHeight(){\n"
        "    if (_hRaf) return;\n"
        "    _hRaf = requestAnimationFrame(function(){\n"
        "      _hRaf = 0;\n"
        "      try{\n"
        "        var h = Math.max(document.body.scrollHeight, document.body.offsetHeight) + 1;\n"
        "        if (h === _lastH) return;\n"
        "        _lastH = h;\n"
        "        parent.postMessage({type:'iframe:height', height:h}, '*');\n"
        "      }catch(e){}\n"
        "    });\n"
        "  }",
        label="26 reportHeight-shrink-throttle",
    )

    html = replace_once(
        html,
        "  function appendFinal(delta){\n"
        "    finalBuf += delta;\n"
        "    document.getElementById('finalMd').innerHTML = renderMarkdown(finalBuf);\n"
        "    document.getElementById('finalMd').classList.add('cursor');\n"
        "  }\n"
        "  function endFinal(){ document.getElementById('finalMd').classList.remove('cursor'); }",
        "  function _splitFrozen(buf, from){\n"
        "    var n = buf.length, i = from, start = from;\n"
        "    function lineEnd(p){ var nl = buf.indexOf('\\n', p); return nl === -1 ? n : nl; }\n"
        "    while (i < n){\n"
        "      var le = lineEnd(i);\n"
        "      var line = buf.slice(i, le);\n"
        "      if (/^\\s*```(\\w*)\\s*$/.test(line)){\n"
        "        if (le >= n) break;\n"
        "        var j = le + 1, closed = false;\n"
        "        while (j < n){\n"
        "          var le2 = lineEnd(j);\n"
        "          if (le2 < n && /^\\s*```\\s*$/.test(buf.slice(j, le2))){\n"
        "            var after = (le2 < n) ? le2 + 1 : n;\n"
        "            start = after; i = after; closed = true; break;\n"
        "          }\n"
        "          if (le2 >= n) break;\n"
        "          j = le2 + 1;\n"
        "        }\n"
        "        if (!closed) break;\n"
        "        continue;\n"
        "      }\n"
        "      if (/^\\s*$/.test(line)){\n"
        "        var blankEnd = (le < n) ? le + 1 : n;\n"
        "        start = blankEnd; i = blankEnd;\n"
        "        continue;\n"
        "      }\n"
        "      if (le >= n) break;\n"
        "      i = le + 1;\n"
        "    }\n"
        "    return start;\n"
        "  }\n"
        "  function _renderFinal(){\n"
        "    _finalRaf = 0;\n"
        "    if (!document.getElementById('finalFrozen') || !document.getElementById('finalLive')) return;\n"
        "    var frozenEnd = _splitFrozen(finalBuf, _frozenLen);\n"
        "    if (frozenEnd > _frozenLen){\n"
        "      document.getElementById('finalFrozen').insertAdjacentHTML('beforeend', renderMarkdown(finalBuf.slice(_frozenLen, frozenEnd)));\n"
        "      _frozenLen = frozenEnd;\n"
        "    }\n"
        "    document.getElementById('finalLive').innerHTML = renderMarkdown(finalBuf.slice(_frozenLen));\n"
        "    reportHeight();\n"
        "  }\n"
        "  function appendFinal(delta){\n"
        "    finalBuf += delta;\n"
        "    if (!_finalRaf) _finalRaf = requestAnimationFrame(_renderFinal);\n"
        "  }\n"
        "  function endFinal(){\n"
        "    if (_finalRaf){ cancelAnimationFrame(_finalRaf); _finalRaf = 0; }\n"
        "    if (finalBuf.length > _frozenLen){\n"
        "      document.getElementById('finalFrozen').insertAdjacentHTML('beforeend', renderMarkdown(finalBuf.slice(_frozenLen)));\n"
        "      _frozenLen = finalBuf.length;\n"
        "    }\n"
        "    var live = document.getElementById('finalLive');\n"
        "    if (live){ live.innerHTML = ''; live.classList.remove('cursor'); }\n"
        "    reportHeight();\n"
        "  }",
        label="27 final-append",
    )

    html = replace_once(
        html,
        "    finalBuf = '';\n"
        "    var md = document.getElementById('finalMd');\n"
        "    md.classList.add('cursor');",
        "    finalBuf = ''; _frozenLen = 0;\n"
        "    var md = document.getElementById('finalMd');\n"
        "    md.innerHTML = '<div id=\"finalFrozen\"></div><div id=\"finalLive\"></div>';\n"
        "    document.getElementById('finalLive').classList.add('cursor');",
        label="27b startFinal-split",
    )

    html = replace_once(
        html,
        "  function addPanelModel(id){\n"
        "    reveal('sec-roster');",
        "  function addPanelModel(id){\n"
        "    if (!id || document.getElementById('m-'+id.replace(/[^a-z0-9]/gi,''))) return;\n"
        "    reveal('sec-roster');",
        label="29 panel-dedup",
    )

    html = replace_once(
        html,
        "_finalStarted=false; preambleBuf=''; finalBuf='';",
        "_finalStarted=false; preambleBuf=''; finalBuf=''; _frozenLen=0;",
        label="27c reset-frozenlen",
    )

    html = replace_once(
        html,
        "html,body{background:transparent;margin:0;padding:0;}",
        "html,body{background:transparent;margin:0;padding:0;overflow:hidden;}",
        label="30 no-iframe-scroll",
    )

    html = replace_once(
        html,
        "#demoBg{position:fixed;inset:0;z-index:-1;display:none;background:#eef1f6;}\n"
        ":root.dark #demoBg{background:#0e131c;}",
        "",
        label="31 demoBg-css",
    )

    html = replace_once(
        html,
        "'</button>'\n"
        "      + '<div class=\"mstate-toggle\"><span class=\"show-lbl\">Read</span><svg class=\"chev\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M6 9l6 6 6-6\"/></svg></div>'",
        "'</button>'",
        label="32a remove-toggle-from-row",
    )

    html = replace_once(
        html,
        "approxWords(ev.content)+' words</div>'",
        "approxWords(ev.content)+' words<div class=\"mstate-toggle\"><span class=\"show-lbl\">Show</span><svg class=\"chev\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M6 9l6 6 6-6\"/></svg></div></div>'",
        label="32b toggle-to-bottom",
    )

    html = replace_once(
        html,
        "card.classList.contains('open') ? 'Hide' : 'Read';",
        "card.classList.contains('open') ? 'Hide' : 'Show';",
        label="33 show-lbl-toggle",
    )

    html = replace_once(
        html,
        ".an-judge .av{width:16px;height:16px;border-radius:5px;background:var(--m-judge);color:#fff;display:flex;\n"
        "  align-items:center;justify-content:center;font-size:9px;font-weight:700;}",
        ".an-judge .av{width:16px;height:16px;border-radius:5px;background:var(--m-judge);color:#fff;display:flex;\n"
        "  align-items:center;justify-content:center;font-size:9px;font-weight:700;}\n"
        ".an-head{cursor:pointer;}\n"
        ".an-toggle{display:flex;align-items:center;gap:3px;font-size:11px;color:var(--text-faint);}\n"
        ".an-toggle .chev{width:13px;height:13px;transition:transform .25s;}\n"
        ".analysis:not(.collapsed) .an-toggle .chev{transform:rotate(180deg);}\n"
        ".analysis.collapsed > :not(.an-head){display:none;}",
        label="36 analysis-collapse-css",
    )

    html = replace_once(
        html,
        "       +   '<button class=\"copy-btn\" id=\"copyAnalysisBtn\" type=\"button\" title=\"Copy the analysis\" aria-label=\"Copy analysis\">'+copyIcon()+'</button>'\n"
        "       + '</div>';",
        "       +   '<button class=\"copy-btn\" id=\"copyAnalysisBtn\" type=\"button\" title=\"Copy the analysis\" aria-label=\"Copy analysis\">'+copyIcon()+'</button>'\n"
        "       +   '<div class=\"an-toggle\"><span class=\"ant-lbl\">Hide</span><svg class=\"chev\" viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" stroke-width=\"2.2\" stroke-linecap=\"round\" stroke-linejoin=\"round\"><path d=\"M6 9l6 6 6-6\"/></svg></div>'\n"
        "       + '</div>';",
        label="37 analysis-toggle-head",
    )

    html = replace_once(
        html,
        "      if (writeClipboard(wrapHtml(analysisToHtml(a, judgeId)), analysisToText(a, judgeId))) flashCopied(_ab);\n"
        "    });\n"
        "  }",
        "      if (writeClipboard(wrapHtml(analysisToHtml(a, judgeId)), analysisToText(a, judgeId))) flashCopied(_ab);\n"
        "    });\n"
        "    var _an = document.getElementById('analysis');\n"
        "    if (_an) _an.classList.remove('collapsed');\n"
        "    var _ah = _an ? _an.querySelector('.an-head') : null;\n"
        "    if (_ah) _ah.addEventListener('click', function(e){\n"
        "      if (e.target.closest('a') || e.target.closest('.copy-btn')) return;\n"
        "      _an.classList.toggle('collapsed');\n"
        "      var _lbl = _ah.querySelector('.ant-lbl');\n"
        "      if (_lbl) _lbl.textContent = _an.classList.contains('collapsed') ? 'Show' : 'Hide';\n"
        "      reportHeight();\n"
        "    });\n"
        "  }",
        label="38 analysis-toggle-handler",
    )

    html = replace_once(
        html,
        "  var SCALE = 7.5;",
        "",
        label="39a drop-scale",
    )

    html = replace_once(
        html,
        "  var startTs = 0, timer = null, frozen = null;\n"
        "  function fmt(ms){ var s=Math.floor(ms/1000); var m=Math.floor(s/60); s=s%60; return m+'m '+(s<10?'0':'')+s+'s'; }\n"
        "  function tick(){\n"
        "    if (frozen!=null){ document.getElementById('elapsed').textContent = fmt(frozen); return; }\n"
        "    var real = (Date.now()-startTs)*SCALE;\n"
        "    document.getElementById('elapsed').textContent = fmt(real);\n"
        "  }\n"
        "  function startClock(){ startTs=Date.now(); frozen=null; if(timer)clearInterval(timer); timer=setInterval(tick,90); tick(); }\n"
        "  function freezeClock(ms){ frozen=ms; if(timer){clearInterval(timer);timer=null;} tick(); }",
        "  var _clockStart = 0, _clockTimer = null, _clockFrozen = null;\n"
        "  function fmt(ms){ var s=Math.round(ms/1000); if(s<0)s=0; var m=Math.floor(s/60); s=s%60; return m+'m '+(s<10?'0':'')+s+'s'; }\n"
        "  function tick(){\n"
        "    var el = document.getElementById('elapsed'); if(!el) return;\n"
        "    if (_clockFrozen!=null){ el.textContent = fmt(_clockFrozen); return; }\n"
        "    if (!_clockStart){ el.textContent = ''; return; }\n"
        "    el.textContent = fmt(Date.now() - _clockStart);\n"
        "  }\n"
        "  function startClock(startMs){ _clockStart = startMs || Date.now(); _clockFrozen = null; if(_clockTimer)clearInterval(_clockTimer); _clockTimer = setInterval(tick, 500); tick(); }\n"
        "  function freezeClock(ms){ _clockFrozen = ms; if(_clockTimer){clearInterval(_clockTimer);_clockTimer=null;} tick(); }",
        label="39b real-clock",
    )

    html = replace_once(
        html,
        "      case 'response.created':\n"
        "      case 'response.in_progress':\n"
        "        if (ev.response && ev.response.model) document.getElementById('modelLine').textContent = clean(ev.response.model);\n"
        "        break;",
        "      case 'response.created':\n"
        "      case 'response.in_progress':\n"
        "        if (ev.response && ev.response.model) document.getElementById('modelLine').textContent = clean(ev.response.model);\n"
        "        if (ev.type === 'response.created' && !_clockStart) startClock(ev.response && ev.response.created_at ? ev.response.created_at*1000 : Date.now());\n"
        "        break;",
        label="40 start-clock-wire",
    )

    html = replace_once(
        html,
        "        if (ev.response){ endFinal(); renderFooter(ev.response.usage, ev.response.model); }",
        "        if (ev.response){ endFinal(); var _el = (ev.response.elapsed_seconds!=null) ? ev.response.elapsed_seconds : (_clockStart ? (Date.now()-_clockStart)/1000 : 0); freezeClock(_el*1000); renderFooter(ev.response.usage, ev.response.model, _el); }",
        label="41 freeze-clock-wire",
    )

    html = replace_once(
        html,
        "  function renderFooter(u, model){",
        "  function renderFooter(u, model, elapsedSec){",
        label="42 footer-sig",
    )

    html = replace_once(
        html,
        "      + stat('Elapsed', '~2m 28s', 'wall clock')",
        "      + stat('Elapsed', (elapsedSec != null ? fmt(elapsedSec*1000) : '\\u2014'), 'wall clock')",
        label="43 footer-elapsed",
    )

    html = replace_once(
        html,
        "    {key:'panel',    lbl:'Panel',    sub:'~95s'},\n"
        "    {key:'answers',  lbl:'Answers',  sub:''},\n"
        "    {key:'judge',    lbl:'Judge',    sub:'~27s'},\n"
        "    {key:'answer',   lbl:'Answer',   sub:'~24s'}",
        "    {key:'panel',    lbl:'Panel',    sub:''},\n"
        "    {key:'answers',  lbl:'Answers',  sub:''},\n"
        "    {key:'judge',    lbl:'Judge',    sub:''},\n"
        "    {key:'answer',   lbl:'Answer',   sub:''}",
        label="45 rail-no-fake-times",
    )

    html = replace_once(
        html,
        "      + '<div class=\"cost-note\">Fusion ran <b>'+(panelCount+1)+' models</b> across <b>~148s</b> to produce this answer — roughly <b>4–5× the cost</b> of a single call. The depth above is what you paid for.</div>';",
        "      + '<div class=\"cost-note\">'+(_sawSynthesis ? 'Fusion made <b>'+(panelCount+2)+' model calls</b> to produce this answer.' : 'Fusion ran <b>'+(panelCount+1)+' models</b> to produce this answer.')+'</div>';",
        label="44 cost-note",
    )

    html = replace_once(
        html,
        "      + stat('Models', panelCount + ' + 1', 'panel + judge')",
        "      + stat('Models', _sawSynthesis ? (panelCount + ' + 1 + 1') : (panelCount + ' + 1'), _sawSynthesis ? 'panel + judge + synthesis' : 'panel + judge')",
        label="49 models-synthesis-count",
    )

    html = replace_once(
        html,
        "  var DATA = {};\n  var panelCount = 0;",
        "  var DATA = {};\n  var panelCount = 0;\n  var NAMES = /*__FUSION_NAMES_JSON__*/{};",
        label="46 names-sentinel",
    )

    html = replace_once(
        html,
        "  var MODELS = {\n"
        "    '~google/gemini-flash-latest': {label:'Gemini Flash', vendor:'Google',    initial:'G', cls:'gemini'},\n"
        "    'deepseek/deepseek-v4-flash':  {label:'DeepSeek V4 Flash', vendor:'DeepSeek', initial:'D', cls:'deepseek'},\n"
        "    '~moonshotai/kimi-latest':     {label:'Kimi', vendor:'Moonshot AI',        initial:'K', cls:'kimi'},\n"
        "    '~anthropic/claude-opus-latest':{label:'Claude Opus', vendor:'Anthropic',  initial:'C', cls:'judge'}\n"
        "  };\n"
        "  // unknown models get a stable palette colour. Panel-assigned ids (IDCLS) win so a\n"
        "  // given roster is always visually distinct; otherwise derive from a hash of the id.\n"
        "  var IDCLS = {};\n"
        "  function hashIdx(s){ var h=0; for(var i=0;i<s.length;i++){ h=(h*31 + s.charCodeAt(i))>>>0; } return (h%8)+1; }\n"
        "  function modelMeta(id){\n"
        "    if (MODELS[id]) return MODELS[id];\n"
        "    var c = clean(id); var parts = c.split('/');\n"
        "    var name = (parts[parts.length-1]||c).replace(/-(latest|preview|instruct)$/i,'');\n"
        "    return {label: name, vendor: parts[0]||'', initial:(name||'?')[0].toUpperCase(), cls: IDCLS[id] || ('c'+hashIdx(c))};\n"
        "  }\n"
        "  var COLORVAR = {gemini:'--m-gemini',deepseek:'--m-deepseek',kimi:'--m-kimi',judge:'--m-judge',\n"
        "    c1:'--c1',c2:'--c2',c3:'--c3',c4:'--c4',c5:'--c5',c6:'--c6',c7:'--c7',c8:'--c8'};",
        "  function hashIdx(s){ var h=0; for(var i=0;i<s.length;i++){ h=(h*31 + s.charCodeAt(i))>>>0; } return (h%8)+1; }\n"
        "  function prettySlug(s){ return String(s||'').split(/[-_]/).map(function(w){ return w ? w.charAt(0).toUpperCase()+w.slice(1) : w; }).join(' '); }\n"
        "  var VENDORCLS = {anthropic:'anthropic', google:'google', 'google-vertex':'google', openai:'openai', deepseek:'deepseek', moonshotai:'moonshot', 'x-ai':'xai', mistralai:'mistral', 'meta-llama':'meta', qwen:'qwen', cohere:'cohere', microsoft:'meta', perplexity:'xai', amazon:'meta', nvidia:'meta'};\n"
        "  function prettyLabel(label, vk){\n"
        "    if (!label) return label;\n"
        "    var ci = label.indexOf(': ');\n"
        "    if (ci > 0 && ci <= 16) return label.slice(ci + 2);\n"
        "    var v = (vk||'').replace(/[^a-z0-9]/gi,'');\n"
        "    var sp = label.indexOf(' ');\n"
        "    if (v && sp > 0 && label.slice(0, sp).toLowerCase().replace(/[^a-z0-9]/gi,'') === v) return label.slice(sp + 1);\n"
        "    return label;\n"
        "  }\n"
        "  function modelMeta(id){\n"
        "    var c = clean(id); var parts = c.split('/');\n"
        "    var vk = (parts[0]||'').toLowerCase();\n"
        "    var slug = (parts[parts.length-1]||c).replace(/-(latest|preview|instruct)$/i,'');\n"
        "    var label = prettyLabel(NAMES[c] || NAMES[id] || prettySlug(slug), vk);\n"
        "    return {label: label, vendor: parts[0]||'', initial: ((parts[0]||'?').charAt(0)||'?').toUpperCase(), cls: VENDORCLS[vk] || ('c'+hashIdx(vk||c))};\n"
        "  }\n"
        "  var COLORVAR = {anthropic:'--m-judge', judge:'--m-judge', google:'--m-gemini', openai:'--c8', deepseek:'--m-deepseek', moonshot:'--m-kimi', xai:'--c5', mistral:'--c7', meta:'--c6', qwen:'--c4', cohere:'--c3',\n"
        "    c1:'--c1',c2:'--c2',c3:'--c3',c4:'--c4',c5:'--c5',c6:'--c6',c7:'--c7',c8:'--c8'};",
        label="47 generalize-modelmeta",
    )

    html = replace_once(
        html,
        "      case 'response.completed':\n"
        "        if (ev.response){ endFinal();",
        "      case 'response.done':\n"
        "      case 'response.completed':\n"
        "        if (ev.response){ endFinal();",
        label="48 response-done-case",
    )

    html = strip_comments(html)
    html = inject_socketio(html)

    return html


OMIT_TOKENS = [
    'id="demoBg"', 'id="panelSel"', 'id="themeBtn"', 'id="replayBtn"',
    "function runFeed", "var DATA = JSON.parse", 'id="fusion-feed"',
    "addEventListener('message'",
    ".demo-bar", "updateThemeLabel", "buildPanel",
    'id="wideBtn"', "function applyWide", "function initWide", "html.wide",
    "fusionWide", "max-width:72ch", "document.documentElement.scrollHeight",
    "__SOCKETIO_B64__", "atob(", "cdn.socket.io",
    "#demoBg", 'show-lbl">Read',
    "~148s", "~2m 28s", "~95s", "~27s", "~24s", "SCALE = 7.5", "The depth above",
    "var MODELS", "var IDCLS", "Gemini Flash",
]

KEEP_TOKENS = [
    "window.FusionUI", "function renderAnalysis", "function syncTheme",
    "function reportHeight", 'id="copyAllBtn"',
    "function writeClipboard", "function analysisToHtml", SENTINEL,
    "var finalBuf = ''", "var panelCount = 0", "function initCopyAll",
    "function startFinal", "function appendFinal", "function endFinal",
    "function ensureFinal", 'id="sec-final"', 'id="finalMd"', "initCopyFinal",
    "&quot;", "&#39;", '["\'<>\\s]',
    "fusionLiveSocket", "user-join",
    "document.body.scrollHeight", "requestAnimationFrame", "cancelAnimationFrame",
    "function _renderFinal", "function _splitFrozen", "insertAdjacentHTML",
    'id="finalFrozen"', 'id="finalLive"',
    "overflow-wrap:break-word", 'show-lbl">Show',
    "an-toggle", "ant-lbl", ".analysis.collapsed",
    "function freezeClock", "_clockStart", "elapsedSec",
    "var NAMES =", "VENDORCLS", "function prettySlug", "function prettyLabel", "/*__FUSION_NAMES_JSON__*/{}",
    "function panelReasoningDelta", "function panelAnswerDelta",
    "'response.fusion_call.panel.delta'", "'response.fusion_call.panel.reasoning.delta'",
    'class="ticker"', "tick-words", "think-lbl", "think-toggle", "thinking-open",
    "function _thinkRowHtml", "function _bindThinking", "_lastH",
    'id="synthArea"', "think-view", "tv-scroll", "tv-expand", "think-block",
    "function synthesisStarting", "function stageReasoningDelta", "function buildThinkBlock",
    "function _viewportHtml", "function finishSynthesisThinking", "function finishStageThinking",
    "'response.fusion_call.synthesis.in_progress'",
    "'response.fusion_call.analysis.reasoning.delta'",
    "'response.fusion_call.synthesis.reasoning.delta'",
    "_sawSynthesis", "panel + judge + synthesis", "Fusion made ",
    'think-lbl">Thinking</span><span class="show-lbl">Show',
]


DANGLING_TOKENS = [
    "updateThemeLabel", "runFeed", "buildPanel", "clearFeed", "chunkText",
]


def self_check(template: str) -> None:
    assert_absent(template, OMIT_TOKENS)
    assert_absent(template, DANGLING_TOKENS)
    assert_present(template, KEEP_TOKENS)
    assert template.count(SENTINEL) == 1, (
        f"sentinel must appear exactly once, found {template.count(SENTINEL)}"
    )
    assert "'''" not in template, "template contains ''' which collides with the raw-string delimiter"
    assert not template.rstrip("\n").endswith("\\"), "template ends with a backslash (unsafe for r'''...''')"
    assert template.rstrip().endswith("})();\n</script>\n</body>\n</html>".rstrip()), (
        "template no longer ends with the kept IIFE close + document tags"
    )


def inject(template: str) -> None:
    module_src = TARGET_PY.read_text(encoding="utf-8")
    bi = module_src.find(BEGIN_MARKER)
    ei = module_src.find(END_MARKER)
    assert bi != -1, f"BEGIN marker not found in {TARGET_PY}"
    assert ei != -1, f"END marker not found in {TARGET_PY}"
    assert bi < ei, "BEGIN marker must precede END marker"

    block = (
        BEGIN_MARKER + "\n"
        + "_FUSION_TEMPLATE_HTML = r'''" + template + "'''\n"
        + END_MARKER
    )
    new_src = module_src[:bi] + block + module_src[ei + len(END_MARKER):]
    TARGET_PY.write_text(new_src, encoding="utf-8")


def main() -> None:
    assert SOURCE_HTML.exists(), f"source HTML missing: {SOURCE_HTML}"
    assert TARGET_PY.exists(), f"target module missing: {TARGET_PY}"

    html = SOURCE_HTML.read_text(encoding="utf-8")
    template = build_template(html)
    self_check(template)
    inject(template)

    import importlib.util
    import sys

    mod_name = "_fusion_embed_check"
    spec = importlib.util.spec_from_file_location(mod_name, TARGET_PY)
    assert spec and spec.loader, "could not load spec for written module"
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(mod_name, None)
    assert hasattr(mod, "_FUSION_TEMPLATE_HTML"), "_FUSION_TEMPLATE_HTML missing after injection"
    assert hasattr(mod, "FusionDeliberationState"), "FusionDeliberationState missing"
    assert hasattr(mod, "build_fusion_embed_html"), "build_fusion_embed_html missing"

    state = mod.FusionDeliberationState()
    state.record({"type": "response.fusion_call.panel.added", "output_index": 1, "model": "x/y"})
    out = mod.build_fusion_embed_html(state)
    assert SENTINEL not in out, "sentinel was not replaced by build_fusion_embed_html"
    assert "x/y" in out, "injected event payload not found in built HTML"

    print(f"OK: wrote pruned template ({len(template)} chars) into {TARGET_PY}")
    print(f"    template lines: {template.count(chr(10)) + 1}")
    print(f"    sentinel occurrences: {template.count(SENTINEL)}")
    print("    self-check + import smoke + sample build all passed")


if __name__ == "__main__":
    main()
