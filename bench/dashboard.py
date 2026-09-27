#!/usr/bin/env python3
"""dashboard.py - render every stored run into one self-contained dashboard.html.

The table is the easy half. The half that matters is the diff: pick two runs and see the
serve-flag, env and hotfix deltas next to the metric delta. On 2026-08-14 answering "is
this regression real or config drift?" took reflog archaeology and two production boots;
it should be a click.

The summary row is the median at c=TARGET_C; expanding a row shows every concurrency
level the run actually measured, because a config that wins at c=10 can lose at c=1.

Self-contained by design - no CDN, no server. It has to open from a checkout, and from
the head node, offline.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import agg
from bench import load_runs

HERE = Path(__file__).resolve().parent
OUT = HERE / "dashboard.html"
TARGET_C = agg.TARGET_C


def _med(vals):
    """Median of the values that exist. -1 is agg's 'server did not report' sentinel."""
    vals = [v for v in vals if v is not None and v == v and v != -1]
    return statistics.median(vals) if vals else None


def metrics_for(rundir: Path) -> dict:
    """Median score/ttft/tps across the run's counted reps, via agg's scoring."""
    rows = agg.arm(str(rundir))
    if not rows:
        return {}
    med = lambda k: statistics.median([r[k] for r in rows])  # noqa: E731
    return {
        "n": len(rows),
        "score": med("score"),
        "ttft_p95": med("ttft_p95"),
        "tps_req": med("tps_req"),
        "tps_agg": med("tps_agg"),
        "spread": agg.spread([r["score"] for r in rows]),
        "preempt": max(r["preempt"] for r in rows),
        "cache": med("cache"),
        "queue": med("queue"),
        "prefill": med("prefill"),
    }


def levels_for(rundir: Path) -> list[dict]:
    """Median of every metric at every concurrency level the run measured.

    agg.arm() only ever looks at TARGET_C; this is the same reduction applied per level
    so the expanded row can show the whole curve.
    """
    acc: dict[int, list[dict]] = {}
    for f in sorted(rundir.glob("loadgen-*.json")):
        try:
            j = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for lv in j.get("summary") or []:
            if lv.get("ttft_p95") is None or not lv.get("tps_per_req"):
                continue  # pre-2026-08-12 loadgen stamped null timings; unscoreable
            acc.setdefault(lv["level"], []).append(lv)

    out = []
    for c, xs in sorted(acc.items()):
        srv = [x.get("server") or {} for x in xs]
        g = lambda k: _med([x.get(k) for x in xs])  # noqa: E731
        s = lambda k: _med([x.get(k) for x in srv])  # noqa: E731
        ttft, tps = g("ttft_p95"), g("tps_per_req")
        out.append({
            "c": c,
            "n": len(xs),
            "score": (agg.STEPS * ttft + agg.TASK_OUT_TOKENS / tps) if ttft and tps else None,
            "ttft_p50": g("ttft_p50"),
            "ttft_p95": ttft,
            "total_p95": g("total_p95"),
            "tps_req": tps,
            "tps_agg": g("tps_aggregate"),
            "tpot": g("tpot_mean"),
            "out": g("out_tokens_mean"),
            "prompt": g("prompt_tokens_mean"),
            "queue": s("queue_mean"),
            "prefill": s("prefill_mean"),
            "cache": s("cache_hit_pct"),
            "preempt": max([x.get("preemptions") for x in srv if x.get("preemptions") is not None]
                           or [None]),
            "ok": f"{int(sum(x['succeeded'] for x in xs))}/{int(sum(x['launched'] for x in xs))}",
        })
    return out


def collect() -> list[dict]:
    out = []
    for m in load_runs():
        d = m["_dir"]
        m["metrics"] = metrics_for(d)
        m["levels"] = levels_for(d)
        # compose runs store .env.dspark, sparkrun runs store the whole recipe; the
        # manifest's config_file says which, and legacy rows predate the field.
        cfg = d / m.get("config_file", "env.dspark")
        m["env_text"] = cfg.read_text() if cfg.exists() else ""
        m["dir"] = d.name
        m.pop("_dir")
        out.append(m)
    out.sort(key=lambda r: r["dir"], reverse=True)
    return out


CSS = """
:root{
  --bg:#f6f7f9; --surface:#fff; --surface2:#f9fafb; --fg:#16181d; --mut:#6b7280;
  --faint:#9aa1ab; --line:#e5e7eb; --line2:#eef0f3;
  --ok:#15803d; --bad:#b91c1c; --warn:#b45309; --accent:#1d4ed8;
  --accent-soft:rgba(29,78,216,.10); --shadow:0 1px 2px rgba(16,24,40,.05),0 8px 24px -12px rgba(16,24,40,.18);
}
@media (prefers-color-scheme:dark){:root{
  --bg:#0d0f13; --surface:#14171d; --surface2:#191d24; --fg:#e6e8eb; --mut:#9aa1ab;
  --faint:#6b7280; --line:#262b33; --line2:#1e232a;
  --ok:#4ade80; --bad:#f87171; --warn:#fbbf24; --accent:#93b4ff;
  --accent-soft:rgba(147,180,255,.14); --shadow:0 1px 2px rgba(0,0,0,.3),0 12px 32px -16px rgba(0,0,0,.8);
}}
*{box-sizing:border-box}
body{margin:0;padding:26px 22px 60px;background:var(--bg);color:var(--fg);
  font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased}
.page{max-width:1700px;margin:0 auto}
/* the titles borrow the code stack rather than load a webfont - this file has to open offline */
code,.num,h1,.panel h2{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
  font-variant-numeric:tabular-nums}
/* plain text, but keeps the .unver badge's warn ink so the two titles still read as one voice */
h1{font-size:21px;margin:0 0 14px;letter-spacing:.06em;font-weight:600;line-height:1.25;
  color:var(--warn);text-transform:uppercase}
/* header facts reuse the detail panel's .chip, one fact per chip */
.chips.head{margin:0 0 16px;gap:9px}
.chips.head .chip{padding:8px 15px}
.chips.head b{font-size:13.5px}

/* toolbar */
.bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:0 0 12px}
.bar input[type=search]{flex:0 1 320px;padding:7px 11px;border:1px solid var(--line);
  border-radius:8px;background:var(--surface);color:var(--fg);font:13px/1.4 inherit}
.bar input[type=search]:focus{outline:2px solid var(--accent-soft);border-color:var(--accent)}
.tog{display:inline-flex;align-items:center;gap:6px;color:var(--mut);font-size:13px;
  padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--surface);
  cursor:pointer;user-select:none}
.tog:hover{border-color:var(--accent)}
.tog input{accent-color:var(--accent);margin:0}
button.tog{font-family:inherit}
.spacer{flex:1}
.count{color:var(--faint);font-size:12.5px}

/* table */
.wrap{background:var(--surface);border:1px solid var(--line);border-radius:12px;
  box-shadow:var(--shadow);overflow:auto;max-height:min(74vh,900px)}
table{border-collapse:separate;border-spacing:0;width:100%;min-width:1080px}
th,td{padding:9px 9px;text-align:right;white-space:nowrap;border-bottom:1px solid var(--line2)}
td{font-variant-numeric:tabular-nums}
td code{font-size:12.5px}
thead th{position:sticky;top:0;z-index:3;background:var(--surface2);color:var(--mut);
  font-size:10.5px;font-weight:600;text-transform:uppercase;letter-spacing:.04em;
  border-bottom:1px solid var(--line);cursor:pointer;user-select:none}
thead th:hover{color:var(--fg)}
thead th.nos{cursor:help;text-align:center;font-size:13px;color:var(--faint)}
thead th.nos:hover{color:var(--accent)}
/* CSS tooltip, not title= - the browser's own delay before showing title is ~1s and
   is not settable. Sits inside the th's sticky stacking context, so it paints over
   the rows and is not clipped by .wrap's scroll. */
thead th.nos::after{content:attr(data-tip);position:absolute;left:2px;top:calc(100% + 5px);
  z-index:10;white-space:nowrap;padding:5px 9px;border-radius:7px;pointer-events:none;
  background:var(--fg);color:var(--bg);box-shadow:var(--shadow);
  font-size:11.5px;font-weight:500;text-transform:none;letter-spacing:0;
  opacity:0;transition:opacity .1s}
thead th.nos:hover::after{opacity:1}
th .ar{opacity:.25;font-size:9px;margin-left:4px}
th.on{color:var(--accent)} th.on .ar{opacity:1}
td.l,th.l{text-align:left}
tbody tr.run{cursor:pointer}
tbody tr.run:hover td{background:var(--surface2)}
tbody tr.run.open td{background:var(--accent-soft)}
tbody tr.run.sel td:first-child{box-shadow:inset 3px 0 0 var(--accent)}
td.pick{width:34px;text-align:center;padding-left:12px}
/* keep the run's identity on screen when the metrics scroll sideways. The sticky
   cells need an opaque background or the scrolled columns show through. */
td.pick,td.runcell{position:sticky;left:0;z-index:2;background:var(--surface)}
td.runcell{left:34px}
thead th:nth-child(-n+2){position:sticky;left:0;z-index:5;background:var(--surface2)}
thead th:nth-child(2){left:34px}
tbody tr.run:hover td.pick,tbody tr.run:hover td.runcell{background:var(--surface2)}
tbody tr.run.open td.pick,tbody tr.run.open td.runcell{
  background:color-mix(in srgb,var(--accent) 11%,var(--surface))}
td.pick input{accent-color:var(--accent);cursor:pointer;margin:0}
.caret{display:inline-block;width:12px;color:var(--faint);transition:transform .12s;
  font-size:10px}
tr.run.open .caret{transform:rotate(90deg);color:var(--accent)}
.ref{font-weight:600;font-size:13px}
.when{color:var(--faint);font-size:11.5px;font-weight:400;margin-left:0}

/* badges */
.badge{font-size:10.5px;padding:2px 8px;border-radius:99px;border:1px solid var(--line);
  letter-spacing:.02em;text-transform:uppercase;font-weight:600}
.full{color:var(--ok);border-color:color-mix(in srgb,var(--ok) 45%,transparent);
  background:color-mix(in srgb,var(--ok) 10%,transparent)}
.unver{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 45%,transparent);
  background:color-mix(in srgb,var(--warn) 10%,transparent)}
.gates{display:inline-flex;gap:3px}
.g{width:16px;height:16px;line-height:16px;border-radius:4px;font-size:10px;
  text-align:center;font-weight:700;font-family:ui-monospace,monospace}
.gP{color:var(--ok);background:color-mix(in srgb,var(--ok) 16%,transparent)}
.gF{color:var(--bad);background:color-mix(in srgb,var(--bad) 16%,transparent)}
.gn{color:var(--faint);background:var(--surface2)}
.dash{color:var(--faint)}

/* in-cell magnitude bar: one hue, anchored under the right-aligned number */
.bar-cell{position:relative}
.bar-cell .fill{position:absolute;right:11px;bottom:4px;height:3px;border-radius:2px;
  background:var(--accent);opacity:.5}
td.score{font-weight:600}

/* detail */
tr.det>td{padding:0;background:var(--surface2);border-bottom:1px solid var(--line);
  text-align:left}
.det-in{padding:16px 18px 18px}
/* stat tile: muted label over its value, so the two never compete on one line */
.chips{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}
.chip{display:flex;flex-direction:column;gap:3px;line-height:1.3;
  background:var(--surface);border:1px solid var(--line);border-radius:9px;padding:6px 12px}
.chip .k{font-size:9.5px;font-weight:600;text-transform:uppercase;letter-spacing:.06em;
  color:var(--mut)}
.chip b{color:var(--fg);font-weight:600;font-family:ui-monospace,monospace;font-size:12px}
.det h3{margin:0 0 8px;font-size:12px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--mut);font-weight:600}
table.lv{min-width:0;width:auto;background:var(--surface);border:1px solid var(--line);
  border-radius:10px;overflow:hidden}
table.lv th{position:static;background:var(--surface2);cursor:default;font-size:10.5px}
table.lv td,table.lv th{padding:7px 12px}
table.lv tbody tr:last-child td{border-bottom:none}
table.lv tbody tr.tc td{background:var(--accent-soft)}
table.lv td.c{font-weight:700}
/* a header cell naming a run: the ref reads as a name, not as a column label */
table.lv th.who{text-transform:none;letter-spacing:0;font-size:11.5px;color:var(--fg)}
table.lv th.grp{color:var(--mut);border-bottom:1px solid var(--line2);text-align:center}
.ab{color:var(--accent);font-weight:700}
table.lv .sep{border-left:1px solid var(--line)}
table.lv tbody tr:nth-child(even) td{background:color-mix(in srgb,var(--fg) 3%,transparent)}
table.lv tbody tr.tc td{background:var(--accent-soft)}
.note{margin-top:12px;color:var(--mut);font-size:12.5px;max-width:90ch;font-style:italic}

/* compare popup - native <dialog>, so backdrop/Esc/focus-trap come for free */
dialog.panel{margin:auto;width:min(1100px,94vw);max-height:88vh;overflow:auto;
  border:1px solid var(--line);border-radius:12px;padding:18px;
  background:var(--surface);color:var(--fg);box-shadow:var(--shadow)}
dialog.panel::backdrop{background:rgba(0,0,0,.45)}
/* the title and the ✕ stay put while a long diff scrolls under them */
.mhead{display:flex;align-items:center;gap:10px;position:sticky;top:-18px;z-index:1;
  background:var(--surface);padding:2px 0 9px;margin:-2px 0 0}
.mhead form{margin-left:auto}
.panel h2{font-size:16px;margin:0;font-weight:600;letter-spacing:.06em;line-height:1.25;
  color:var(--warn);text-transform:uppercase}
.panel h3{margin:16px 0 7px;font-size:12px;text-transform:uppercase;letter-spacing:.05em;
  color:var(--mut);font-weight:600}
.diff{display:grid;grid-template-columns:minmax(210px,auto) 1fr 1fr;
  border:1px solid var(--line);border-radius:9px;overflow:hidden}
.diff>div{padding:6px 11px;border-bottom:1px solid var(--line2);overflow-x:auto;
  font-size:12.5px;min-width:0}
.diff>div:nth-child(3n+2),.diff>div:nth-child(3n){border-left:1px solid var(--line)}
.diff>div:nth-child(6n+4),.diff>div:nth-child(6n+5),.diff>div:nth-child(6n+6){
  background:color-mix(in srgb,var(--fg) 3%,transparent)}
.diff .h{background:var(--surface2)!important;font-weight:600;font-size:11.5px;
  color:var(--fg);border-bottom:1px solid var(--line)}
.diff .h.lbl{color:var(--mut);text-transform:uppercase;letter-spacing:.04em;font-size:11px}
.diff code{white-space:pre-wrap;word-break:break-all}
.diff .h .when,th.who .when{display:block;font-weight:400;margin-top:1px}
.add{color:var(--ok)}.del{color:var(--bad)}
.good{color:var(--ok);font-weight:600}.worse{color:var(--bad);font-weight:600}
.hint{color:var(--mut);font-size:13px;margin:0}
.warnbox{margin-top:10px;padding:8px 11px;border-radius:8px;font-size:12.5px;color:var(--warn);
  background:color-mix(in srgb,var(--warn) 9%,transparent);
  border:1px solid color-mix(in srgb,var(--warn) 35%,transparent)}
.empty{padding:26px;text-align:center;color:var(--faint)}
"""

JS = r"""
const RUNS = __DATA__, TC = __TC__;

const f = (v,d)=> (v==null||v!==v) ? '<span class="dash">—</span>' : (+v).toFixed(d);
const esc = s => String(s==null?'':s).replace(/[&<>"]/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// dir is "2026-08-14T08-27Z__ref" - split it so the table can show both halves.
function parts(dir){
  const i = dir.indexOf('__');
  return i<0 ? {when:'', ref:dir} : {when:dir.slice(0,i).replace('T',' ').replace(/-(\d\d)Z$/,':$1Z'),
                                     ref:dir.slice(i+2)};
}
const m = (r,k) => (r.metrics||{})[k];

const COLS = [
  {k:'dir',   t:'run', l:1, get:r=>r.dir, cell:r=>{const p=parts(r.dir);
     return '<span class="caret">&#9656;</span> <span class="ref">'+esc(p.ref)+
            '</span><div class="when num">'+esc(p.when)+'</div>';}},
  {k:'provenance', t:'prov', get:r=>r.provenance==='full'?1:0, cell:r=>{
     const pv=r.provenance||'unverified';
     return '<span class="badge '+(pv==='full'?'full':'unver')+'">'+esc(pv)+'</span>';}},
  {k:'gates', t:'gates', get:r=>GK.filter(k=>(r.gates||{})[k]).length, cell:r=>{
     const g=r.gates||{};
     return '<span class="gates" title="'+GK.join(' / ')+'">'+GK.map(k=>{
       const v=g[k], c=v===undefined||v===null?'gn':v?'gP':'gF';
       return '<span class="g '+c+'" title="'+k+'">'+(v===undefined||v===null?'·':v?'P':'F')+'</span>';
     }).join('')+'</span>';}},
  {k:'dspark_sha', t:'dspark', get:r=>r.dspark_sha||'', cell:r=>'<code>'+
     esc((r.dspark_sha||'—').slice(0,8))+'</code>'},
  {k:'served', t:'served as', l:1, get:r=>r.served_model_name||'',
     cell:r=>'<code>'+esc(r.served_model_name||'—')+'</code>'},
  {k:'n',        t:'n',       get:r=>m(r,'n'),        cell:r=>m(r,'n')??'<span class="dash">—</span>'},
  {k:'score',    t:'score',   get:r=>m(r,'score'),    cell:r=>f(m(r,'score'),1), lower:1, bar:1},
  {k:'ttft_p95', t:'ttft p95',get:r=>m(r,'ttft_p95'), cell:r=>f(m(r,'ttft_p95'),2), lower:1},
  {k:'tps_req',  t:'tok/s req',get:r=>m(r,'tps_req'), cell:r=>f(m(r,'tps_req'),2)},
  {k:'tps_agg',  t:'tok/s agg',get:r=>m(r,'tps_agg'), cell:r=>f(m(r,'tps_agg'),1)},
  {k:'queue',    t:'queue s', get:r=>m(r,'queue'),    cell:r=>f(m(r,'queue'),2), lower:1},
  {k:'prefill',  t:'prefill s',get:r=>m(r,'prefill'), cell:r=>f(m(r,'prefill'),2), lower:1},
  {k:'cache',    t:'cache %', get:r=>m(r,'cache'),    cell:r=>f(m(r,'cache'),1)},
  {k:'preempt',  t:'preempt', get:r=>m(r,'preempt'),  cell:r=>f(m(r,'preempt'),0), lower:1},
];
const GK = ['smoke','toolprobe','loadgen','decode_shapes'];
// The bar under score is inverted: score is lower-is-better, so the best run gets the full
// width and a long bar keeps meaning "good", the way it did under tok/s agg.
const BEST = Math.min(...RUNS.map(r=>m(r,'score')).filter(v=>v>0), Infinity);
const barw = (v,best)=> (v>0 && isFinite(best) ? best/v*46 : 0).toFixed(1);

let sort={k:'dir',dir:-1}, sel=[], open=new Set(), q='', verOnly=false;

function visible(){
  let rows = RUNS.map((r,i)=>({r,i}));
  if(verOnly) rows = rows.filter(x=>x.r.provenance==='full');
  if(q){const s=q.toLowerCase();
    rows = rows.filter(x=>(x.r.dir+' '+(x.r.served_model_name||'')+' '+(x.r.note||'')+' '+
                           (x.r.dspark_sha||'')).toLowerCase().includes(s));}
  const col = COLS.find(c=>c.k===sort.k) || COLS[0];
  rows.sort((a,b)=>{
    const x=col.get(a.r), y=col.get(b.r);
    const nx=x==null||x!==x, ny=y==null||y!==y;
    if(nx||ny) return nx&&ny ? 0 : nx ? 1 : -1;        // missing always last
    return (typeof x==='string' ? x.localeCompare(y) : x-y) * sort.dir;
  });
  return rows;
}

function levelTable(r){
  const L = r.levels||[];
  if(!L.length) return '<p class="hint">No scoreable loadgen samples in this run.</p>';
  const best = Math.min(...L.map(l=>l.score).filter(v=>v>0), Infinity);
  const cols = [
    ['c','c',0],['n','reps',0],['score','score',1],['ttft_p50','ttft p50',2],
    ['ttft_p95','ttft p95',2],['total_p95','total p95',2],['tps_req','tok/s req',2],
    ['tps_agg','tok/s agg',1],['tpot','tpot ms',1],['queue','queue s',3],
    ['prefill','prefill s',2],['cache','cache %',1],['preempt','preempt',0],
    ['out','out tok',0],['ok','ok',0]];
  let h='<table class="lv"><thead><tr>'+cols.map(c=>'<th>'+c[1]+'</th>').join('')+
        '</tr></thead><tbody>';
  L.forEach(l=>{
    h+='<tr'+(l.c===TC?' class="tc" title="the level the summary row and score use"':'')+'>';
    cols.forEach(([k,,d])=>{
      if(k==='ok') h+='<td class="num">'+esc(l.ok)+'</td>';
      else if(k==='c') h+='<td class="num c">'+l.c+'</td>';
      else if(k==='tpot') h+='<td class="num">'+f(l.tpot==null?null:l.tpot*1000,d)+'</td>';
      else if(k==='score') h+='<td class="num bar-cell score">'+f(l.score,d)+
        '<span class="fill" style="width:'+barw(l.score,best)+'px"></span></td>';
      else h+='<td class="num">'+f(l[k],d)+'</td>';
    });
    h+='</tr>';
  });
  return h+'</tbody></table>';
}

function detail(r){
  const w=r.workload||{}, g=r.gates||{};
  const chip=(k,v)=> v==null||v===''?'':'<span class="chip"><span class="k">'+k+
    '</span><b>'+esc(v)+'</b></span>';
  const sp=(r.metrics||{}).spread;
  let h='<div class="det-in"><div class="chips">'+
    chip('captured', r.captured_utc)+chip('model', r.model)+
    chip('vllm', r.vllm_version)+
    chip('score spread', sp==null?null:sp.toFixed(1)+'%')+chip('prompt', w.prompt_tokens)+
    chip('unique', w.unique_tokens)+chip('max_tokens', w.max_tokens)+
    chip('levels', (w.levels||[]).join(', '))+
    chip('gates', GK.map(k=>k+'='+(g[k]===undefined||g[k]===null?'·':g[k]?'pass':'FAIL')).join(' '))+
    '</div><h3>Metrics per concurrency level</h3>'+levelTable(r);
  if(r.note) h+='<p class="note">'+esc(r.note)+'</p>';
  return h+'</div>';
}

function draw(){
  const rows=visible();
  let h='<table><thead><tr><th class="nos" aria-label="Select two runs to compare '+
    'their results" data-tip="Select two runs to compare their results">&#9432;</th>'+
    COLS.map(c=>
    '<th class="'+(c.l?'l ':'')+(sort.k===c.k?'on':'')+'" data-k="'+c.k+
    '" title="sort by '+c.t+'">'+c.t+
    '<span class="ar">'+(sort.k===c.k?(sort.dir<0?'▼':'▲'):'↕')+'</span></th>'
  ).join('')+'</tr></thead><tbody>';
  rows.forEach(({r,i})=>{
    const cls='run'+(open.has(i)?' open':'')+(sel.includes(i)?' sel':'');
    h+='<tr class="'+cls+'" data-i="'+i+'"><td class="pick"><input type="checkbox" '+
       (sel.includes(i)?'checked ':'')+'title="select two runs to compare"></td>'+
       COLS.map(c=>{
         let inner=c.cell(r);
         if(c.bar) inner+='<span class="fill" style="width:'+barw(m(r,c.k),BEST)+'px"></span>';
         return '<td class="'+(c.l?'l':'num')+(c.bar?' bar-cell':'')+
                (c.k==='score'?' score':'')+(c.k==='dir'?' runcell':'')+
                '">'+inner+'</td>';
       }).join('')+'</tr>';
    if(open.has(i))
      h+='<tr class="det"><td colspan="'+(COLS.length+1)+'">'+detail(r)+'</td></tr>';
  });
  h+='</tbody></table>';
  if(!rows.length) h='<div class="empty">No run matches this filter.</div>';
  document.getElementById('tbl').innerHTML=h;
  document.getElementById('count').textContent=
    rows.length+' of '+RUNS.length+' runs'+(sel.length?' · '+sel.length+' selected':'');
    compare();
}

/* ---- compare popup ---- */
function envMap(t){const o={};(t||'').split('\n').forEach(l=>{l=l.trim();
  if(!l||l[0]==='#'||!l.includes('='))return;const i=l.indexOf('=');
  o[l.slice(0,i).trim()]=l.slice(i+1).trim();});return o;}
function diffKeys(a,b){
  const ks=new Set([...Object.keys(a||{}),...Object.keys(b||{})]);
  return [...ks].sort().filter(k=>String((a||{})[k])!==String((b||{})[k]));
}
/* Render only - draw() calls this on every filter/sort/expand, so it must never open the
   dialog. Opening is the second-tick transition, or the toolbar button. */
function compare(){
  const el=document.getElementById('panel');
  if(sel.length!==2){el.close();return;}   // no-op if already closed
  const [A,B]=sel.map(i=>RUNS[i]);
  // Column headers name the run by its ref; the timestamp is the disambiguator underneath.
  const who=(r,ab)=>{const p=parts(r.dir);
    return '<span class="ab">'+ab+'</span> <span class="ref">'+esc(p.ref)+
           '</span><span class="when num">'+esc(p.when)+'</span>';};
  let h='<div class="mhead"><h2>Compare</h2>'+
        '<form method="dialog"><button class="tog" title="close (Esc)">&#10005;</button>'+
        '</form></div>'+
        '<p class="hint"><span class="ab">A</span> <b>'+
        esc(parts(A.dir).ref)+'</b> &nbsp;&rarr;&nbsp; <span class="ab">B</span> <b>'+
        esc(parts(B.dir).ref)+'</b> - &Delta; is B against A.</p>';
  if(A.provenance!=='full'||B.provenance!=='full')
    h+='<div class="warnbox">One side has unverified provenance - a metric delta here may '+
       'be config drift, not a real change.</div>';

  // pct change of y against x, colored by whether that direction is an improvement
  const pct=(x,y,lower)=>{
    if(x==null||y==null||!x) return '<span class="dash">—</span>';
    const p=(y-x)/x*100, cls=Math.abs(p)<0.5?'':((p<0)===!!lower?'good':'worse');
    return '<span class="'+cls+'">'+(p>=0?'+':'')+p.toFixed(1)+'%</span>';};

  const ma=A.metrics||{}, mb=B.metrics||{};
  const MET=[['score',1,1],['ttft_p95',2,1],['tps_req',2,0],['tps_agg',1,0],['queue',2,1],
             ['prefill',2,1],['cache',1,0]];
  h+='<h3>metric @ c='+TC+'</h3><table class="lv"><thead><tr><th class="l">metric</th>'+
     '<th class="who">'+who(A,'A')+'</th><th class="who">'+who(B,'B')+
     '</th><th>&Delta;</th></tr>'+
     '</thead><tbody>'+MET.map(([k,d,lo])=>
       '<tr><td class="l"><code>'+k+'</code></td><td class="num">'+f(ma[k],d)+
       '</td><td class="num">'+f(mb[k],d)+'</td><td class="num">'+
       pct(ma[k],mb[k],lo)+'</td></tr>').join('')+'</tbody></table>';

  // per-level, union of levels: A / B / delta grouped per metric so the eye compares down
  const la={},lb={};
  (A.levels||[]).forEach(l=>la[l.c]=l); (B.levels||[]).forEach(l=>lb[l.c]=l);
  const cs=[...new Set([...Object.keys(la),...Object.keys(lb)])].map(Number).sort((x,y)=>x-y);
  if(cs.length){
    const GRP=[['score','score',1,1],['tps_agg','tok/s agg',1,0],['ttft_p95','ttft p95',2,1]];
    h+='<h3>per concurrency level</h3><table class="lv"><thead><tr><th class="l" rowspan="2">c'+
       '</th>'+GRP.map(g=>'<th class="grp sep" colspan="3">'+g[1]+'</th>').join('')+'</tr><tr>'+
       GRP.map(()=>'<th class="who sep"><span class="ab">A</span></th>'+
         '<th class="who"><span class="ab">B</span></th><th>&Delta;</th>')
         .join('')+'</tr></thead><tbody>';
    cs.forEach(c=>{
      const x=la[c],y=lb[c];
      h+='<tr'+(c===TC?' class="tc"':'')+'><td class="num c">'+c+'</td>'+
         GRP.map(([k,,d,lo])=>'<td class="num sep">'+(x?f(x[k],d):
           '<span class="dash">not run</span>')+'</td><td class="num">'+(y?f(y[k],d):
           '<span class="dash">not run</span>')+'</td><td class="num">'+
           pct(x&&x[k],y&&y[k],lo)+'</td>').join('')+'</tr>';});
    h+='</tbody></table>';
  }
  const sec=(title,a,b)=>{
    let s='<h3>'+title+'</h3><div class="diff"><div class="h lbl">key</div><div class="h">'+
          who(A,'A')+'</div><div class="h">'+who(B,'B')+'</div>';
    // One side with nothing captured would otherwise render every key as a "difference".
    const na=!Object.keys(a||{}).length, nb=!Object.keys(b||{}).length;
    if(na||nb) return s+'<div class="hint" style="grid-column:1/4">not captured for '+
      esc(na&&nb?'either run':parts(na?A.dir:B.dir).ref)+' - nothing to diff</div></div>';
    const ks=diffKeys(a,b);
    if(!ks.length) s+='<div class="hint" style="grid-column:1/4">identical</div>';
    ks.forEach(k=>{s+='<div><code>'+esc(k)+'</code></div><div class="del"><code>'+
      ((a||{})[k]===undefined?'-':esc((a||{})[k]))+'</code></div><div class="add"><code>'+
      ((b||{})[k]===undefined?'-':esc((b||{})[k]))+'</code></div>';});
    return s+'</div>';};
  h+=sec('serve flag',A.serve_flags,B.serve_flags);
  // Label the section by whichever config file each side actually stored
  // (env.dspark for DeepSeek, env.qwen38fn for the Qwen/vLLM recipe, recipe.yaml
  // for sparkrun) -- labelling a Qwen .env "env.dspark" is how a diff gets misread.
  const ca=A.config_file||'env.dspark', cb=B.config_file||'env.dspark';
  const cfgLabel=ca===cb?ca:ca+' vs '+cb;
  h+=sec(cfgLabel,envMap(A.env_text),envMap(B.env_text));
  const hf=r=>{const o={};Object.entries(r.hotfixes||{}).forEach(([k,v])=>
    o[k]=(v||[]).join(' ')||'(none)');return o;};
  h+=sec('hotfixes applied',hf(A),hf(B));
  el.innerHTML=h;
}

/* ---- events (delegated: the table is re-rendered on every change) ---- */
document.getElementById('tbl').addEventListener('click',e=>{
  const th=e.target.closest('th[data-k]');
  if(th){const k=th.dataset.k;
    sort = sort.k===k ? {k,dir:-sort.dir} : {k,dir:(k==='dir'?-1:1)};
    return draw();}
  const tr=e.target.closest('tr.run'); if(!tr) return;
  const i=+tr.dataset.i, before=sel.length;
  if(e.target.matches('input[type=checkbox]')){
    const at=sel.indexOf(i);
    if(at>=0) sel.splice(at,1); else {if(sel.length===2) sel.shift(); sel.push(i);}
  } else {
    open.has(i) ? open.delete(i) : open.add(i);
  }
  draw();
  if(before<2&&sel.length===2) document.getElementById('panel').showModal();
});
// With no reopen button, closing the popup releases the pair - otherwise two runs stay
// ticked with no way back to their diff. Guarded so the close() that fires when a pair is
// broken by unticking does not also clear the run still selected.
document.getElementById('panel').addEventListener('close',()=>{
  if(sel.length===2){sel=[];draw();}});
document.getElementById('q').addEventListener('input',e=>{q=e.target.value;draw();});
document.getElementById('ver').addEventListener('change',e=>{verOnly=e.target.checked;draw();});
document.getElementById('expand').addEventListener('change',e=>{
  if(e.target.checked) visible().forEach(x=>open.add(x.i)); else open.clear(); draw();});
draw();
"""


def build() -> int:
    runs = collect()
    slim = [
        {k: r.get(k) for k in (
            "dir", "ref", "provenance", "captured_utc", "served_model_name", "model",
            "dspark_sha", "vllm_version", "serve_flags", "env_text", "hotfixes",
            "deployment", "config_file",
            "workload", "gates", "metrics", "levels", "note")}
        for r in runs
    ]
    data = json.dumps(slim).replace("</", "<\\/")  # never close the <script> early
    full = sum(1 for r in runs if r.get("provenance") == "full")
    html = (
        "<title>Spark bench</title>"
        f"<style>{CSS}</style>"
        '<div class="page">'
        "<h1>DGX Spark Bench</h1>"
        '<div class="chips head">'
        f'<span class="chip"><span class="k">runs</span><b>{len(runs)}</b></span>'
        f'<span class="chip"><span class="k">full provenance</span><b>{full}</b></span>'
        f'<span class="chip"><span class="k">summary row</span><b>median at c={TARGET_C}</b>'
        "</span>"
        '<span class="chip"><span class="k">score - lower is better</span>'
        f'<b>{agg.STEPS}·ttft_p95 + {agg.TASK_OUT_TOKENS}/tok-s-per-req</b></span>'
        '<span class="chip"><span class="k">gates</span>'
        "<b>smoke / toolprobe / loadgen / decode-shapes</b></span>"
        "</div>"
        '<div class="bar">'
        '<input type="search" id="q" placeholder="filter by ref, model, sha or note…">'
        '<label class="tog"><input type="checkbox" id="ver"> full provenance only</label>'
        '<label class="tog"><input type="checkbox" id="expand"> expand all</label>'
        '<span class="spacer"></span><span class="count" id="count"></span>'
        "</div>"
        '<div class="wrap" id="tbl"></div>'
        '<dialog class="panel" id="panel"></dialog>'
        "</div>"
        f"<script>{JS.replace('__DATA__', data).replace('__TC__', str(TARGET_C))}</script>"
    )
    OUT.write_text(html)
    print(f"wrote {OUT}  ({len(runs)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(build())
