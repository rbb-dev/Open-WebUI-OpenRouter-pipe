"""CSS + JS assets (raw strings) for the dashboard Config tab."""

CONFIG_TAB_CSS = r"""
.cfgroot{--bg:transparent;--panel:#ffffff;--ink:var(--text);--muted:var(--text-muted);--faint:var(--text-faint);--border2:var(--border-light);--accent:var(--tab-active-text);--accent-weak:var(--tab-active-bg);--accent-ink:var(--tab-active-text);--warn:#d97706;--warn-bg:#fffbeb;--warn-bd:#fde68a;--tip:#2563eb;--tip-bg:#eff6ff;--tip-bd:#bfdbfe;--rose:#e11d48;--violet:#7c3aed;--radius:10px;--shadow:0 1px 2px rgba(15,23,42,.04),0 4px 12px rgba(15,23,42,.05);--mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif}
.cfgroot .topbar{display:flex;align-items:center;gap:16px;padding:12px 20px;background:var(--panel);border-bottom:1px solid var(--border);flex:0 0 auto;z-index:5}
.cfgroot .searchwrap{position:relative;flex:1;max-width:460px}
.cfgroot #search{width:100%;padding:9px 14px 9px 38px;border:1px solid var(--border2);border-radius:9px;font-size:14px;font-family:var(--sans);background:#f8fafc;transition:.15s}
.cfgroot #search:focus{outline:none;border-color:var(--accent);background:#fff;box-shadow:0 0 0 3px var(--accent-weak)}
.cfgroot .searchwrap::before{content:"\1F50D";position:absolute;left:12px;top:50%;transform:translateY(-50%);opacity:.5;font-size:13px}
.cfgroot .searchcount{position:absolute;right:12px;top:50%;transform:translateY(-50%);font-size:12px;color:var(--muted)}
.cfgroot .toolbar{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.cfgroot .chk{display:flex;align-items:center;gap:6px;font-size:13px;color:var(--muted);cursor:pointer;user-select:none;white-space:nowrap}
.cfgroot .btn{padding:7px 12px;border:1px solid var(--border2);background:#fff;border-radius:8px;font-size:13px;color:var(--ink);cursor:pointer;transition:.12s;font-family:var(--sans);white-space:nowrap}
.cfgroot .btn:hover{background:#f8fafc;border-color:var(--faint)}
.cfgroot .btn.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
.cfgroot .btn.primary:hover{background:#5457e0}
.cfgroot .btn.primary:disabled{background:#c7d2fe;border-color:#c7d2fe;cursor:not-allowed}
.cfgroot .btn.ghost{color:var(--muted)}
.cfgroot .savebar{display:none;align-items:center;gap:10px;padding-right:10px;margin-right:2px;border-right:1px solid var(--border)}
.cfgroot .savebar.show{display:flex}
.cfgroot .dirtybadge{display:flex;align-items:center;gap:7px;font-size:13px;color:var(--warn);font-weight:650;white-space:nowrap}
.cfgroot .dirtybadge::before{content:"";width:8px;height:8px;border-radius:50%;background:var(--warn)}
.cfgroot .body{display:flex;flex:1;min-height:0}
.cfgroot .tree{flex:0 0 340px;overflow-y:auto;border-right:1px solid var(--border);background:var(--panel);padding:8px 0}
.cfgroot .detail{flex:1;overflow-y:auto;padding:30px 40px}
.cfgroot .group{margin:1px 8px}
.cfgroot .ghead{display:flex;align-items:center;gap:9px;padding:8px 10px;border-radius:8px;cursor:pointer;user-select:none;font-weight:600;font-size:13px}
.cfgroot .ghead:hover{background:#f1f5f9}
.cfgroot .chev{width:14px;font-size:10px;color:var(--faint);transition:transform .15s}
.cfgroot .group.open>.ghead .chev{transform:rotate(90deg)}
.cfgroot .gicon{font-size:15px}
.cfgroot .gname{flex:1}
.cfgroot .gcount{font-size:11px;color:var(--faint);background:#f1f5f9;padding:1px 7px;border-radius:10px;font-weight:600}
.cfgroot .gbody{display:none;padding:2px 0 6px}
.cfgroot .group.open>.gbody{display:block}
.cfgroot .sub{margin:4px 0 4px 22px}
.cfgroot .shead{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--faint);font-weight:600;padding:5px 10px 3px}
.cfgroot .item{display:flex;align-items:center;gap:8px;padding:6px 10px 6px 14px;margin:0 6px 0 14px;border-radius:7px;cursor:pointer;font-size:13px;color:#334155;border-left:2px solid transparent}
.cfgroot .item:hover{background:#f8fafc}
.cfgroot .item.sel{background:var(--accent-weak);color:var(--accent-ink);border-left-color:var(--accent);font-weight:550}
.cfgroot .item .ilabel{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.cfgroot .item.mod .ilabel::after{content:"\25CF";color:var(--warn);font-size:9px;margin-left:7px;vertical-align:middle}
.cfgroot .dot{width:6px;height:6px;border-radius:50%;flex:0 0 auto}
.cfgroot .dot.sec{background:var(--rose)}
.cfgroot .dot.tpl{background:var(--violet)}
.cfgroot mark{background:#fde68a;border-radius:2px;padding:0 1px}
.cfgroot .empty{color:var(--faint);text-align:center;margin-top:120px;font-size:15px}
.cfgroot .crumb{font-size:12px;color:var(--muted);margin-bottom:6px}
.cfgroot .crumb b{color:var(--ink)}
.cfgroot .dtitle{font-size:24px;font-weight:680;letter-spacing:-.02em;margin-bottom:4px}
.cfgroot .dname{font-family:var(--mono);font-size:12px;color:var(--muted);background:#f1f5f9;padding:2px 8px;border-radius:6px;display:inline-block}
.cfgroot .badges{display:flex;gap:6px;margin:14px 0 20px;flex-wrap:wrap}
.cfgroot .badge{font-size:11px;font-weight:600;padding:3px 9px;border-radius:20px}
.cfgroot .badge.sec{background:#fff1f2;color:var(--rose)}
.cfgroot .badge.usr{background:#f5f3ff;color:var(--violet)}
.cfgroot .pending{display:flex;align-items:center;gap:10px;flex-wrap:wrap;background:var(--warn-bg);border:1px solid var(--warn-bd);border-radius:8px;padding:10px 14px;margin-bottom:18px;font-size:13.5px}
.cfgroot .pending .pk{font-weight:700;color:var(--warn)}
.cfgroot .pending .oldv{color:var(--muted);text-decoration:line-through}
.cfgroot .pending .newv{color:var(--accent-ink);font-weight:600}
.cfgroot .pending .arrow{color:var(--faint)}
.cfgroot .revert{margin-left:auto;border:1px solid var(--border2);background:#fff;border-radius:6px;padding:4px 11px;font-size:12px;cursor:pointer;color:var(--ink)}
.cfgroot .revert:hover{background:#f8fafc}
.cfgroot .card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:18px 20px;box-shadow:var(--shadow);margin-bottom:24px}
.cfgroot .card .clabel{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--faint);font-weight:600;margin-bottom:12px}
.cfgroot .switch{position:relative;display:inline-flex;align-items:center;gap:12px;cursor:pointer}
.cfgroot .switch input{display:none}
.cfgroot .track{width:42px;height:24px;background:var(--border2);border-radius:20px;position:relative;transition:.18s}
.cfgroot .track::after{content:"";position:absolute;top:2px;left:2px;width:20px;height:20px;background:#fff;border-radius:50%;box-shadow:0 1px 3px rgba(0,0,0,.2);transition:.18s}
.cfgroot .switch input:checked+.track{background:var(--accent)}
.cfgroot .switch input:checked+.track::after{transform:translateX(18px)}
.cfgroot .swval{font-weight:600;font-size:14px}
.cfgroot select,.cfgroot input.txt,.cfgroot input.num{font-family:var(--sans);font-size:14px;padding:9px 12px;border:1px solid var(--border2);border-radius:8px;background:#fff;color:var(--ink);min-width:240px;transition:.12s}
.cfgroot select:focus,.cfgroot input.txt:focus,.cfgroot input.num:focus{outline:none;border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-weak)}
.cfgroot .num.err,.cfgroot .txt.err{border-color:var(--rose);box-shadow:0 0 0 3px #fecdd3}
.cfgroot .errmsg{color:var(--rose);font-size:12px;margin-top:7px;font-weight:500}
.cfgroot .hint{font-size:12px;color:var(--muted);margin-top:8px}
.cfgroot .secretrow{display:flex;align-items:center;gap:10px}
.cfgroot .secretrow input{font-family:var(--mono);letter-spacing:.08em}
.cfgroot .eye{border:1px solid var(--border2);background:#fff;border-radius:8px;padding:8px 10px;cursor:pointer}
.cfgroot .editor{border:1px solid var(--border);border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow)}
.cfgroot .etabs{display:flex;gap:2px;background:#f8fafc;border-bottom:1px solid var(--border);padding:6px 6px 0}
.cfgroot .etab{padding:7px 14px;font-size:13px;font-weight:600;color:var(--muted);cursor:pointer;border-radius:7px 7px 0 0}
.cfgroot .etab.on{background:#fff;color:var(--accent-ink);box-shadow:0 -1px 0 var(--border)}
.cfgroot .etab.dis{opacity:.4;cursor:not-allowed}
.cfgroot .etool{margin-left:auto;font-size:11px;color:var(--faint);padding:8px 10px}
.cfgroot textarea.tpl{width:100%;min-height:340px;resize:vertical;border:0;padding:16px;font-family:var(--mono);font-size:13px;line-height:1.6;color:#0f172a;background:#fff;tab-size:2}
.cfgroot textarea.tpl:focus{outline:none}
.cfgroot .preview{padding:20px 22px;min-height:340px;background:#fff}
.cfgroot .diff{font-family:var(--mono);font-size:12.5px;line-height:1.65;background:#fff;max-height:420px;overflow:auto}
.cfgroot .dl{padding:1px 14px;white-space:pre-wrap;word-break:break-word}
.cfgroot .dl.add{background:#ecfdf5;color:#065f46}
.cfgroot .dl.add::before{content:"+ ";opacity:.7}
.cfgroot .dl.del{background:#fef2f2;color:#991b1b}
.cfgroot .dl.del::before{content:"\2212 ";opacity:.7}
.cfgroot .dl.ctx{color:var(--muted)}
.cfgroot .dl.ctx::before{content:"\2007\2007"}
.cfgroot .help{max-width:66ch;font-size:14.5px;line-height:1.65;color:#1e293b}
.cfgroot .help p{margin:0 0 12px}
.cfgroot .help ul{margin:0 0 12px;padding-left:4px;list-style:none}
.cfgroot .help li{position:relative;padding-left:20px;margin:5px 0}
.cfgroot .help li::before{content:"";position:absolute;left:4px;top:9px;width:5px;height:5px;border-radius:50%;background:var(--accent)}
.cfgroot .help code{font-family:var(--mono);font-size:.86em;background:#f1f5f9;color:#be185d;padding:1.5px 6px;border-radius:5px}
.cfgroot .help strong{font-weight:650}
.cfgroot .callout{border-radius:8px;padding:10px 14px;margin:6px 0 14px;font-size:13.5px;border:1px solid}
.cfgroot .callout.warning{background:var(--warn-bg);border-color:var(--warn-bd);color:#92400e}
.cfgroot .callout.tip{background:var(--tip-bg);border-color:var(--tip-bd);color:#1e40af}
.cfgroot .secth{font-size:12px;text-transform:uppercase;letter-spacing:.05em;color:var(--faint);font-weight:600;margin:26px 0 12px}
.cfgroot .meta{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
.cfgroot .mcell{background:var(--panel);border:1px solid var(--border);border-radius:8px;padding:11px 14px}
.cfgroot .mk{font-size:11px;color:var(--faint);text-transform:uppercase;letter-spacing:.04em;font-weight:600}
.cfgroot .mv{font-family:var(--mono);font-size:13px;margin-top:3px;word-break:break-word}
.cfgroot .modal{position:fixed;inset:0;background:rgba(15,23,42,.45);display:none;align-items:center;justify-content:center;z-index:50;padding:20px}
.cfgroot .modal.show{display:flex}
.cfgroot .modalcard{background:#fff;border-radius:14px;box-shadow:0 24px 70px rgba(0,0,0,.32);width:min(640px,94vw);max-height:84vh;display:flex;flex-direction:column;overflow:hidden}
.cfgroot .mhead{padding:18px 22px;border-bottom:1px solid var(--border);font-size:16px;font-weight:660}
.cfgroot .mhead small{display:block;font-size:12px;font-weight:400;color:var(--muted);margin-top:2px}
.cfgroot .mbody{padding:4px 22px;overflow-y:auto}
.cfgroot .mrow{padding:13px 0;border-bottom:1px solid var(--border)}
.cfgroot .mrow:last-child{border-bottom:0}
.cfgroot .mrl{font-size:13.5px;font-weight:600;margin-bottom:3px;display:flex;align-items:center;gap:8px}
.cfgroot .mrn{font-family:var(--mono);font-size:11px;color:var(--faint)}
.cfgroot .mrv{font-size:13px;margin-top:4px}
.cfgroot .mrv .oldv{color:var(--muted);text-decoration:line-through}
.cfgroot .mrv .newv{color:var(--accent-ink);font-weight:600}
.cfgroot .mrv .arrow{color:var(--faint)}
.cfgroot .chip{font-size:11px;font-weight:600;padding:1px 8px;border-radius:10px;background:#f1f5f9;color:#475569}
.cfgroot .expand{font-size:12px;color:var(--accent);cursor:pointer;user-select:none;margin-top:6px;display:inline-block}
.cfgroot .mfoot{padding:16px 22px;border-top:1px solid var(--border);display:flex;gap:10px;justify-content:flex-end;align-items:center}
.cfgroot .mfoot .sp{margin-right:auto;font-size:12px;color:var(--muted)}
.cfgroot .toast{position:fixed;bottom:26px;left:50%;transform:translateX(-50%) translateY(10px);background:var(--ink);color:#fff;padding:11px 20px;border-radius:10px;font-size:13.5px;font-weight:500;opacity:0;pointer-events:none;transition:.28s;z-index:60}
.cfgroot .toast.show{opacity:1;transform:translateX(-50%) translateY(0)}
.cfgroot .tree::-webkit-scrollbar,.cfgroot .detail::-webkit-scrollbar,.cfgroot .diff::-webkit-scrollbar,.cfgroot .mbody::-webkit-scrollbar{width:10px}
.cfgroot .tree::-webkit-scrollbar-thumb,.cfgroot .detail::-webkit-scrollbar-thumb{background:#e2e8f0;border-radius:8px;border:3px solid var(--panel)}
.cfgroot{display:flex;flex-direction:column;color:var(--ink);font-family:var(--sans);font-size:14px;line-height:1.5;-webkit-font-smoothing:antialiased}
.cfgroot .topbar{border-radius:10px 10px 0 0}
.cfgroot .body{height:600px;flex:0 0 auto;border:1px solid var(--border);border-top:0;border-radius:0 0 10px 10px}
.cfgroot .driftnote{font-size:12px;color:var(--warn);font-weight:600;white-space:nowrap}
.cfgroot .conflict{display:flex;align-items:center;gap:12px;flex-wrap:wrap;background:var(--warn-bg);border:1px solid var(--warn-bd);color:#92400e;border-radius:8px;padding:10px 14px;margin:10px 0 0;font-size:13.5px}
.cfgroot .conflict button{margin-left:auto;border:1px solid var(--border2);background:var(--panel);border-radius:6px;padding:4px 12px;font-size:12px;cursor:pointer;color:var(--ink)}

:root.dark .cfgroot{--panel:rgba(255,255,255,.03);--warn:#fbbf24;--warn-bg:rgba(245,158,11,.12);--warn-bd:rgba(245,158,11,.3);--tip:#60a5fa;--tip-bg:rgba(59,130,246,.12);--tip-bd:rgba(59,130,246,.32);--rose:#fb7185;--violet:#a78bfa}
:root.dark .cfgroot .conflict{color:#fde68a}
:root.dark .cfgroot .ghead:hover{background:rgba(255,255,255,.05)}
:root.dark .cfgroot .item{color:#cbd5e1}
:root.dark .cfgroot .item:hover{background:rgba(255,255,255,.04)}
:root.dark .cfgroot .gcount,:root.dark .cfgroot .dname,:root.dark .cfgroot .chip{background:rgba(255,255,255,.07)}
:root.dark .cfgroot .chip{color:#cbd5e1}
:root.dark .cfgroot mark{background:rgba(250,204,21,.28);color:inherit}
:root.dark .cfgroot select,:root.dark .cfgroot input.txt,:root.dark .cfgroot input.num,:root.dark .cfgroot textarea.tpl{background:rgba(255,255,255,.03);color:var(--ink)}
:root.dark .cfgroot #search{background:rgba(255,255,255,.03)}
:root.dark .cfgroot #search:focus{background:rgba(255,255,255,.05)}
:root.dark .cfgroot .btn,:root.dark .cfgroot .revert,:root.dark .cfgroot .eye{background:rgba(255,255,255,.04)}
:root.dark .cfgroot .btn:hover,:root.dark .cfgroot .revert:hover{background:rgba(255,255,255,.08)}
:root.dark .cfgroot .btn.primary{background:var(--accent);color:#0b1220}
:root.dark .cfgroot .btn.primary:disabled{background:rgba(129,140,248,.3);border-color:rgba(129,140,248,.3)}
:root.dark .cfgroot .etabs{background:rgba(255,255,255,.03)}
:root.dark .cfgroot .etab.on{background:var(--panel);color:var(--accent-ink)}
:root.dark .cfgroot .preview,:root.dark .cfgroot .diff{background:transparent}
:root.dark .cfgroot .help{color:#cbd5e1}
:root.dark .cfgroot .help code{background:rgba(255,255,255,.08);color:#f9a8d4}
:root.dark .cfgroot .dl.add{background:rgba(16,185,129,.12);color:#6ee7b7}
:root.dark .cfgroot .dl.del{background:rgba(239,68,68,.12);color:#fca5a5}
:root.dark .cfgroot .modalcard{background:#0f172a}
:root.dark .cfgroot .badge.sec{background:rgba(251,113,133,.15);color:#fb7185}
:root.dark .cfgroot .badge.usr{background:rgba(167,139,250,.15);color:#a78bfa}
:root.dark .cfgroot .toast{background:#334155;color:#f8fafc}
:root.dark .cfgroot .tree::-webkit-scrollbar-thumb,:root.dark .cfgroot .detail::-webkit-scrollbar-thumb{background:rgba(255,255,255,.1)}
"""

CONFIG_TAB_JS = r"""
(function(){
const CFGROOT=document.getElementById(ID+"-cfgroot"); if(!CFGROOT){return;}

const $=s=>CFGROOT.querySelector(s);
const el=(t,c,h)=>{const e=document.createElement(t);if(c)e.className=c;if(h!=null)e.innerHTML=h;return e;};
const esc=s=>(s==null?"":String(s)).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
const ICON={"Connection & Routing":"🔌","Models & Catalog":"🧠","Files & Media":"🖼️","Filters & Integrations":"🧩","Tools":"🛠️","Reasoning & Thinking":"💭","Prompt Caching":"⚡","Streaming & Performance":"📡","Reliability":"🛡️","Security":"🔒","Storage":"💾","Usage & Status":"📊","Error Messages":"⚠️","Logging":"📝","Plugins":"🔌"};
let VALVES=[],byName={},baseline={},REV=null,lastSeenRev=null,inflightSave=false;
const edits={}; const invalid=new Set();
let SEL=null, Q="", CHANGED=false;
const openG={};

function md(src){
  const lines=esc(src).split("\n"); let out="",inList=false;
  const inline=t=>t.replace(/`([^`]+)`/g,'<code>$1</code>').replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>');
  for(let raw of lines){
    const line=raw.replace(/\s+$/,"");
    if(/^\s*-\s+/.test(line)){if(!inList){out+="<ul>";inList=true;}out+="<li>"+inline(line.replace(/^\s*-\s+/,""))+"</li>";continue;}
    if(inList){out+="</ul>";inList=false;}
    if(line===""){continue;}
    const m=line.match(/^\*\*(Warning|Tip):\*\*\s*(.*)$/);
    if(m){out+='<div class="callout '+m[1].toLowerCase()+'"><strong>'+m[1]+':</strong> '+inline(m[2])+'</div>';continue;}
    out+="<p>"+inline(line)+"</p>";
  }
  if(inList)out+="</ul>"; return out;
}
function lineDiff(o,n){
  const a=String(o==null?"":o).split("\n"), b=String(n==null?"":n).split("\n");
  const m=a.length,k=b.length,dp=Array.from({length:m+1},()=>new Array(k+1).fill(0));
  for(let i=m-1;i>=0;i--)for(let j=k-1;j>=0;j--)dp[i][j]=a[i]===b[j]?dp[i+1][j+1]+1:Math.max(dp[i+1][j],dp[i][j+1]);
  const out=[]; let i=0,j=0;
  while(i<m&&j<k){ if(a[i]===b[j]){out.push(["ctx",a[i]]);i++;j++;} else if(dp[i+1][j]>=dp[i][j+1]){out.push(["del",a[i]]);i++;} else {out.push(["add",b[j]]);j++;} }
  while(i<m)out.push(["del",a[i++]]); while(j<k)out.push(["add",b[j++]]);
  return out;
}
function diffCount(o,n){ const d=lineDiff(o,n); return {add:d.filter(x=>x[0]=="add").length, del:d.filter(x=>x[0]=="del").length}; }
function diffHtml(o,n){ return '<div class="diff">'+lineDiff(o,n).map(([t,l])=>'<div class="dl '+t+'">'+esc(l||" ")+'</div>').join("")+'</div>'; }
const norm=x=>x===true?"true":x===false?"false":x==null?"":String(x);
const isDirty=n=>n in edits;
const curVal=v=>v.name in edits?edits[v.name]:baseline[v.name];
function fmtVal(v,val){ if(v.widget.startsWith("toggle"))return (val===true||val==="True"||val==="true")?"Enabled":"Disabled"; if(val===""||val==null)return "(empty)"; return String(val); }

function setEdit(v,val){
  const base=baseline[v.name];
  const numEq = v.widget.startsWith("number") && norm(val)!=="" && norm(base)!=="" && Number.isFinite(+val) && Number.isFinite(+base) && (+val===+base);
  if(v.secret){ if(norm(val).length) edits[v.name]=val; else delete edits[v.name]; }
  else if(numEq || norm(val)===norm(base)) delete edits[v.name];
  else edits[v.name]=val;
  updateBar();
  const it=CFGROOT.querySelector(".item.sel"); if(it&&SEL===v.name) it.classList.toggle("mod",isDirty(v.name));
  if(CHANGED) buildTree();
}
function updateBar(){
  const n=Object.keys(edits).length;
  $("#dirtyN").textContent=n;
  $("#savebar").classList.toggle("show",n>0);
  const bad=invalid.size>0;
  const sv=$("#save"); sv.disabled=(n===0||bad); sv.textContent=bad?("Fix "+invalid.size+" error"+(invalid.size>1?"s":"")):("Save"+(n?" ("+n+")":""));
}

function matches(v){
  if(CHANGED && !isDirty(v.name)) return false;
  if(!Q) return true;
  return (v.name+" "+v.title+" "+v.top+" "+v.sub+" "+v.detail).toLowerCase().includes(Q);
}
function hl(text){ if(!Q)return esc(text); const i=text.toLowerCase().indexOf(Q); if(i<0)return esc(text);
  return esc(text.slice(0,i))+"<mark>"+esc(text.slice(i,i+Q.length))+"</mark>"+esc(text.slice(i+Q.length)); }

function buildTree(){
  const tree=$("#tree"); tree.innerHTML="";
  const tops={}; let shown=0;
  for(const v of VALVES){ if(!matches(v))continue; shown++; (tops[v.top]=tops[v.top]||{}); (tops[v.top][v.sub]=tops[v.top][v.sub]||[]).push(v); }
  for(const top of Object.keys(tops).sort((a,b)=>a.localeCompare(b))){
    const subs=tops[top]; const cnt=Object.values(subs).reduce((a,b)=>a+b.length,0);
    const filt=Q||CHANGED; const isOpen=filt?true:(openG[top]??false);
    const g=el("div","group"+(isOpen?" open":""));
    const head=el("div","ghead",'<span class="chev"'+(filt?' style="visibility:hidden"':'')+'>▶</span><span class="gicon">'+(ICON[top]||"●")+'</span><span class="gname">'+esc(top)+'</span><span class="gcount">'+cnt+'</span>');
    if(filt){head.style.cursor="default";}else{head.onclick=()=>{openG[top]=!(openG[top]??false);buildTree();};}
    g.appendChild(head);
    const body=el("div","gbody");
    const subNames=[...new Set(VALVES.filter(v=>v.top===top).map(v=>v.sub))].filter(s=>subs[s]).sort((a,b)=>a.localeCompare(b));
    const multi=subNames.length>1;
    for(const sn of subNames){
      const sub=el("div","sub"); if(multi) sub.appendChild(el("div","shead",esc(sn)));
      for(const v of subs[sn].sort((a,b)=>a.title.localeCompare(b.title))){
        const it=el("div","item"+(SEL===v.name?" sel":"")+(isDirty(v.name)?" mod":""));
        const kind=v.secret?"sec":(v.is_template?"tpl":"");
        it.innerHTML='<span class="dot '+kind+'"></span><span class="ilabel">'+hl(v.title)+'</span>';
        it.onclick=()=>{SEL=v.name;renderDetail(v);buildTree();};
        sub.appendChild(it);
      }
      body.appendChild(sub);
    }
    g.appendChild(body); tree.appendChild(g);
  }
  $("#searchcount").textContent=Q?(shown+" match"+(shown==1?"":"es")):"";
  if(shown===0) tree.appendChild(el("div","empty",CHANGED?"No unsaved changes":(Q?"No settings match":"")));
}

function ctrl(v){
  const val=curVal(v);
  if(v.is_template){
    return '<div class="editor" data-tpl><div class="etabs">'
      +'<div class="etab on" data-t="edit">Edit</div><div class="etab" data-t="prev">Preview</div>'
      +'<div class="etab'+(isDirty(v.name)?'':' dis')+'" data-t="diff">Diff</div>'
      +'<div class="etool">Markdown · {placeholder} + {{#if}} blocks</div></div>'
      +'<div class="epane" data-p="edit"><textarea class="tpl" spellcheck="false">'+esc(val)+'</textarea></div>'
      +'<div class="epane" data-p="prev" style="display:none"><div class="preview help">'+md(val)+'</div></div>'
      +'<div class="epane" data-p="diff" style="display:none"></div></div>';
  }
  if(v.widget.startsWith("toggle")){
    const on=(val===true||val==="True"||val==="true");
    return '<label class="switch"><input type="checkbox" '+(on?"checked":"")+'><span class="track"></span><span class="swval">'+(on?"Enabled":"Disabled")+'</span></label>';
  }
  if(v.widget.startsWith("dropdown")&&v.enum)
    return '<select>'+v.enum.map(o=>'<option '+(String(o)===String(val)?"selected":"")+'>'+esc(o)+'</option>').join("")+'</select>';
  if(v.input&&v.input.type==="select"&&v.input.options)
    return '<select>'+v.input.options.map(o=>{const ov=(o&&typeof o==="object")?o.value:o;const ol=(o&&typeof o==="object")?o.label:o;return '<option value="'+esc(ov)+'"'+(String(ov)===String(val)?" selected":"")+'>'+esc(ol)+'</option>';}).join("")+'</select>';
  if(v.input&&v.input.type==="color")
    return '<div class="colorrow" style="display:flex;align-items:center;gap:10px"><input type="color" value="'+esc(val||"#000000")+'" style="width:44px;height:38px;padding:2px;border:1px solid var(--border2);border-radius:8px;cursor:pointer"><input class="txt" type="text" value="'+esc(val)+'" style="min-width:160px"></div>';
  if(v.widget==="array"||Array.isArray(v.default))
    return '<textarea class="tpl" style="min-height:80px" spellcheck="false" placeholder="comma, separated, values">'+esc(Array.isArray(val)?val.join(", "):(val||""))+'</textarea><div class="hint">Comma-separated list — OWUI stores arrays as comma-joined text.</div>';
  if(v.widget.startsWith("number")){
    const b=v.bounds||{}, flt=v.widget.includes("float");
    return '<input class="num" type="number" value="'+esc(val)+'"'+(b.ge!=null?' min="'+b.ge+'"':"")+(b.le!=null?' max="'+b.le+'"':"")+' step="'+(flt?"0.1":"1")+'"><div class="err-slot"></div>'
      +'<div class="hint">'+(b.ge!=null||b.le!=null?("Range "+(b.ge??"−∞")+" to "+(b.le??"no limit")):"No fixed range")+'</div>';
  }
  if(v.secret||v.widget.startsWith("masked")){
    const ph=v.secret_set?"●●●●  configured — type to replace":"●●●●  not set — type to set";
    const hint=v.secret_set?"Sensitive · a value is configured · type to replace · encrypted at rest":"Sensitive · encrypted at rest · write-only";
    return '<div class="secretrow"><input class="txt" type="password" value="" placeholder="'+ph+'" style="min-width:320px"><button class="eye" type="button" title="Show while typing">👁</button></div><div class="hint">'+hint+'</div>';
  }
  const long=(norm(val).length>48)||/MODELS|HOSTS|ALLOWLIST|ICON_SET|VARIANT/.test(v.name);
  if(long) return '<textarea class="tpl" style="min-height:90px" spellcheck="false">'+esc(val)+'</textarea>';
  return '<input class="txt" type="text" value="'+esc(val)+'" style="min-width:340px">';
}

function pendingBlock(v){
  if(!isDirty(v.name)) return "";
  let inner;
  if(v.secret) inner='<span class="pk">Pending:</span> a new value will be set';
  else if(v.is_template){ const c=diffCount(baseline[v.name],edits[v.name]); inner='<span class="pk">Pending:</span> template modified <span class="newv">(+'+c.add+' −'+c.del+')</span> — see the Diff tab'; }
  else inner='<span class="pk">Pending:</span> <span class="oldv">'+esc(fmtVal(v,baseline[v.name]))+'</span> <span class="arrow">→</span> <span class="newv">'+esc(fmtVal(v,edits[v.name]))+'</span>';
  return '<div class="pending">'+inner+'<button class="revert" type="button">Revert</button></div>';
}

function renderDetail(v){
  const d=$("#detail");
  const badges=[];
  if(v.secret)badges.push('<span class="badge sec">Secret</span>');
  if(v.per_user)badges.push('<span class="badge usr">Per-user override</span>');
  const b=v.bounds||{};
  const meta=[['Default',v.secret?(v.secret_set?"«configured»":"«unset»"):(norm(v.default)===""?"(empty)":esc(String(v.default).slice(0,80)))],['Type',esc(v.widget)],
    (b.ge!=null||b.le!=null)?['Range',(b.ge??"−∞")+" – "+(b.le??"no limit")]:null,
    v.enum?['Options',v.enum.map(esc).join(", ")]:null].filter(Boolean);
  d.innerHTML='<div class="crumb">'+esc(v.top)+' <span style="opacity:.5">/</span> <b>'+esc(v.sub)+'</b></div>'
    +'<div class="dtitle">'+esc(v.title)+'</div><span class="dname">'+esc(v.name)+'</span>'
    +'<div class="badges">'+badges.join("")+'</div>'
    +pendingBlock(v)
    +(v.enriched?"":'<div class="callout warning" style="margin:0 0 16px"><strong>Not documented yet.</strong> This setting has no curated help \u2014 showing its built-in description below.</div>')
    +'<div class="card"><div class="clabel">'+(v.is_template?'Template editor':'Value')+'</div>'+ctrl(v)+'</div>'
    +'<div class="secth">What this does</div><div class="help">'+md(v.detail)+'</div>'
    +'<div class="secth">Details</div><div class="meta">'+meta.map(m=>'<div class="mcell"><div class="mk">'+m[0]+'</div><div class="mv">'+m[1]+'</div></div>').join("")+'</div>';
  wireControl(d,v);
}

function wireControl(d,v){
  const rev=d.querySelector(".revert"); if(rev)rev.onclick=()=>{delete edits[v.name];invalid.delete(v.name);updateBar();renderDetail(v);buildTree();};
  if(v.is_template){
    const ta=d.querySelector('.epane[data-p="edit"] textarea');
    ta.oninput=()=>{setEdit(v,ta.value);
      d.querySelector('.etab[data-t="diff"]').classList.toggle("dis",!isDirty(v.name));
      renderPendingInline(d,v);
    };
    d.querySelectorAll(".etab").forEach(tab=>tab.onclick=()=>{
      if(tab.classList.contains("dis"))return;
      d.querySelectorAll(".etab").forEach(t=>t.classList.remove("on")); tab.classList.add("on");
      const w=tab.dataset.t;
      if(w==="prev")d.querySelector('.epane[data-p="prev"] .preview').innerHTML=md(ta.value);
      if(w==="diff")d.querySelector('.epane[data-p="diff"]').innerHTML=isDirty(v.name)?diffHtml(baseline[v.name],ta.value):'<div style="padding:18px;color:var(--faint)">No changes yet.</div>';
      ["edit","prev","diff"].forEach(p=>d.querySelector('.epane[data-p="'+p+'"]').style.display=(p===w?"":"none"));
    });
    return;
  }
  const sw=d.querySelector(".switch input");
  if(sw){ sw.onchange=()=>{d.querySelector(".swval").textContent=sw.checked?"Enabled":"Disabled";setEdit(v,sw.checked);renderPendingInline(d,v);}; return; }
  const sel=d.querySelector("select"); if(sel){ sel.onchange=()=>{setEdit(v,sel.value);renderPendingInline(d,v);}; return; }
  const num=d.querySelector("input.num");
  if(num){ validateNum(d,v,num); num.oninput=()=>{ validateNum(d,v,num); setEdit(v,num.value); renderPendingInline(d,v); }; return; }
  const eye=d.querySelector(".eye"); const sec=d.querySelector(".secretrow input");
  if(sec){ eye.onclick=()=>{sec.type=sec.type==="password"?"text":"password";}; sec.oninput=()=>{setEdit(v,sec.value);renderPendingInline(d,v);}; return; }
  const col=d.querySelector('input[type="color"]');
  if(col){ const tf=d.querySelector(".colorrow .txt"); col.oninput=()=>{const cv=col.value.toUpperCase(); if(tf)tf.value=cv; setEdit(v,cv); renderPendingInline(d,v);}; if(tf)tf.oninput=()=>{try{col.value=tf.value;}catch(e){} setEdit(v,tf.value); renderPendingInline(d,v);}; return; }
  const tx=d.querySelector(".txt")||d.querySelector("textarea.tpl");
  if(tx){ tx.oninput=()=>{setEdit(v,tx.value);renderPendingInline(d,v);}; }
}
function renderPendingInline(d,v){
  let p=d.querySelector(".pending");
  const html=pendingBlock(v);
  if(p){ if(html){ const t=document.createElement("div"); t.innerHTML=html; p.replaceWith(t.firstChild); wireRevert(d,v);} else p.remove(); }
  else if(html){ d.querySelector(".badges").insertAdjacentHTML("afterend",html); wireRevert(d,v); }
}
function wireRevert(d,v){ const r=d.querySelector(".revert"); if(r)r.onclick=()=>{delete edits[v.name];invalid.delete(v.name);updateBar();renderDetail(v);buildTree();}; }
function validateNum(d,v,num){
  const b=v.bounds||{}; const raw=num.value.trim(); let bad=null;
  if(raw===""){ if(v.default!=null) bad="Required"; }
  else { const x=Number(raw);
    if(Number.isNaN(x))bad="Must be a number";
    else if(b.ge!=null&&x<b.ge)bad="Must be ≥ "+b.ge;
    else if(b.le!=null&&x>b.le)bad="Must be ≤ "+b.le;
  }
  num.classList.toggle("err",!!bad);
  let slot=d.querySelector(".err-slot"); slot.innerHTML=bad?'<div class="errmsg">'+bad+'</div>':"";
  if(bad)invalid.add(v.name); else invalid.delete(v.name);
  updateBar();
}

function openReview(){
  const names=Object.keys(edits); if(!names.length)return;
  const rows=names.map(n=>{ const v=byName[n]; let body;
    if(v.secret) body='<div class="mrv"><span class="chip">secret</span> new value will be set</div>';
    else if(v.is_template){ const c=diffCount(baseline[n],edits[n]);
      body='<div class="mrv"><span class="chip">template</span> modified (+'+c.add+' −'+c.del+') <span class="expand" data-x="'+esc(n)+'">view diff</span><div class="dx" data-dx="'+esc(n)+'" style="display:none;margin-top:8px">'+diffHtml(baseline[n],edits[n])+'</div></div>';}
    else body='<div class="mrv"><span class="oldv">'+esc(fmtVal(v,baseline[n]))+'</span> <span class="arrow">→</span> <span class="newv">'+esc(fmtVal(v,edits[n]))+'</span></div>';
    return '<div class="mrow"><div class="mrl">'+esc(v.title)+' <span class="mrn">'+esc(n)+'</span></div>'+body+'</div>';
  }).join("");
  const m=$("#modal");
  m.innerHTML='<div class="modalcard"><div class="mhead">Review changes<small>'+names.length+' setting'+(names.length>1?"s":"")+' will be saved to Open WebUI and applied on the next request.</small></div>'
    +'<div class="mbody">'+rows+'</div>'
    +'<div class="mfoot"><span class="sp">Nothing is saved until you confirm.</span><button class="btn" id="mCancel">Cancel</button><button class="btn primary" id="mSave">Save '+names.length+'</button></div></div>';
  m.classList.add("show");
  m.querySelectorAll(".expand").forEach(x=>x.onclick=()=>{const dx=m.querySelector('.dx[data-dx="'+CSS.escape(x.dataset.x)+'"]');const s=dx.style.display==="none";dx.style.display=s?"":"none";x.textContent=s?"hide diff":"view diff";});
  $("#mCancel").onclick=()=>m.classList.remove("show");
  $("#mSave").onclick=()=>commitSave();
  m.onclick=e=>{if(e.target===m)m.classList.remove("show");};
}
function commitSave(){
  const names=Object.keys(edits); if(!names.length)return;
  inflightSave=true;
  const payload={}; names.forEach(n=>payload[n]=edits[n]);
  const btn=$("#mSave"); if(btn){btn.disabled=true;btn.textContent="Saving\u2026";}
  callAction("config_set",{edits:payload,rev:REV}).then(resp=>{
    const r=resp&&resp.result;
    if(!resp||resp.error||!r){ inflightSave=false; if(btn){btn.disabled=false;btn.textContent="Save "+names.length;} toast("Save failed"+(resp&&resp.error?": "+resp.error:"")); return; }
    if(r.conflict){ inflightSave=false; $("#modal").classList.remove("show"); if(btn){btn.disabled=false;btn.textContent="Save "+names.length;} showConflict(); return; }
    names.forEach(n=>{ const v=byName[n]; if(v&&v.secret){v.secret_set=true;} else if(v){baseline[n]=edits[n];} delete edits[n]; });
    if(r.rev!=null){REV=r.rev;lastSeenRev=r.rev;}
    inflightSave=false;
    invalid.clear(); $("#modal").classList.remove("show"); updateBar();
    if(SEL&&byName[SEL])renderDetail(byName[SEL]); buildTree();
    toast("Saved "+names.length+" setting"+(names.length>1?"s":"")); reportHeight();
  }).catch(()=>{ inflightSave=false; if(btn){btn.disabled=false;btn.textContent="Save "+names.length;} toast("Save failed"); });
}
function toast(msg){ const t=$("#toast"); t.textContent=msg; t.classList.add("show"); clearTimeout(t._h); t._h=setTimeout(()=>t.classList.remove("show"),2600); }

function showConflict(){
  const c=$("#conflict"); if(!c)return;
  c.style.display="flex";
  c.innerHTML='<span>Another administrator changed the configuration while you were editing.</span><button type="button" id="cfgReload">Reload latest</button>';
  const rb=$("#cfgReload"); if(rb)rb.onclick=()=>{
    if(!Object.keys(edits).length){ loadConfig(); return; }
    if(rb.dataset.cfgArmed==="1"){
      delete rb.dataset.cfgArmed; rb.textContent=rb.dataset.cfgLabel||"Reload"; loadConfig(); return;
    }
    rb.dataset.cfgArmed="1"; rb.dataset.cfgLabel=rb.textContent; rb.textContent="Discard changes?";
    setTimeout(()=>{ if(rb.dataset.cfgArmed==="1"){ delete rb.dataset.cfgArmed; rb.textContent=rb.dataset.cfgLabel||"Reload"; } },5000);
  };
  reportHeight();
}
function hideConflict(){ const c=$("#conflict"); if(c){c.style.display="none";c.innerHTML="";} }
function applySnapshot(r){
  VALVES=r.valves||[]; if(r.rev!=null)REV=r.rev;
  byName={}; baseline={};
  VALVES.forEach(v=>{ byName[v.name]=v; baseline[v.name]=v.value; });
  Object.keys(edits).forEach(n=>delete edits[n]); invalid.clear();
  const dn=$("#driftnote"); const u=(r.drift&&r.drift.unenriched)?r.drift.unenriched.length:0;
  if(dn)dn.textContent=u?(u+" not documented"):"";
  hideConflict(); updateBar();
  if(SEL&&byName[SEL])renderDetail(byName[SEL]); else { SEL=null; const d=$("#detail"); if(d)d.innerHTML='<div class="empty">Select a setting to view and edit it.</div>'; }
  buildTree(); reportHeight();
}
function loadConfig(){
  const tree=$("#tree"); if(tree)tree.innerHTML='<div class="empty" style="margin-top:60px">Loading configuration\u2026</div>';
  callAction("config_get",{}).then(resp=>{
    const r=resp&&resp.result;
    if(!resp||resp.error||!r){ if(tree)tree.innerHTML='<div class="empty" style="margin-top:60px">Could not load configuration'+(resp&&resp.error?": "+esc(resp.error):"")+'</div>'; return; }
    applySnapshot(r);
  }).catch(()=>{ if(tree)tree.innerHTML='<div class="empty" style="margin-top:60px">Could not load configuration.</div>'; });
}
function quietReload(){
  const tree=$("#tree"), det=$("#detail");
  const ts=tree?tree.scrollTop:0, ds=det?det.scrollTop:0;
  callAction("config_get",{}).then(resp=>{
    const r=resp&&resp.result; if(!resp||resp.error||!r)return;
    applySnapshot(r);
    const t2=$("#tree"), d2=$("#detail"); if(t2)t2.scrollTop=ts; if(d2)d2.scrollTop=ds;
  }).catch(()=>{});
}


$("#search").addEventListener("input",e=>{Q=e.target.value.trim().toLowerCase();buildTree();});
$("#search").addEventListener("keydown",e=>{if(e.key==="Escape"){e.target.value="";Q="";buildTree();}});
$("#chgToggle").addEventListener("change",e=>{CHANGED=e.target.checked;buildTree();});
$("#expandAll").onclick=()=>{[...new Set(VALVES.map(v=>v.top))].forEach(t=>openG[t]=true);buildTree();};
$("#collapseAll").onclick=()=>{[...new Set(VALVES.map(v=>v.top))].forEach(t=>openG[t]=false);buildTree();};
$("#save").onclick=openReview;
$("#discard").onclick=()=>{ if(!Object.keys(edits).length)return; Object.keys(edits).forEach(n=>delete edits[n]); invalid.clear(); updateBar(); if(SEL)renderDetail(byName[SEL]); buildTree(); toast("Discarded all changes"); };
updateBar();
cfgFetch=loadConfig;
cfgOnEvent=function(rev){
  if(inflightSave||rev==null||REV==null||rev<=REV||rev<=lastSeenRev)return;
  lastSeenRev=rev;
  if(Object.keys(edits).length===0)quietReload();
  else showConflict();
};

})();
"""
