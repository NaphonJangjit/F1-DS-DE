"""Generate silverstone_report.html — per lap x sector MAE heatmap (cond-anchor model)."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features
from _cond_anchor import cond_anchors

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)

HTML = """<meta name="claude:viewport-fit-sides" content="cover">
<title>Silverstone Lap-Sector Error</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root{
    --bg:#f6f7f9; --surface:#ffffff; --fg:#1c1f24; --muted:#5c626d;
    --border:#e4e7ec; --accent:#e26b1a; --wet:#3f6fb0; --drying:#e08a1e; --dry:#b8bfc9;
    --cell-empty:#eceff3;
    font-family:'Inter',system-ui,-apple-system,'Segoe UI',sans-serif;
  }
  @media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
    --bg:#131519; --surface:#1a1d22; --fg:#e8eaee; --muted:#9aa2ad;
    --border:#2a2e36; --accent:#f0843a; --wet:#5f8fd0; --drying:#eda13d; --dry:#6a7280;
    --cell-empty:#23272e; color-scheme:dark;
  }}
  :root[data-theme="dark"]{
    --bg:#131519; --surface:#1a1d22; --fg:#e8eaee; --muted:#9aa2ad;
    --border:#2a2e36; --accent:#f0843a; --wet:#5f8fd0; --drying:#eda13d; --dry:#6a7280;
    --cell-empty:#23272e; color-scheme:dark;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);
       padding-left:max(16px,env(safe-area-inset-left,0px));
       padding-right:max(16px,env(safe-area-inset-right,0px));
       padding-block:32px 48px;}
  .wrap{max-width:1080px;margin:0 auto;display:grid;gap:20px}
  h1{font-size:1.45rem;font-weight:700;margin:0;letter-spacing:-.01em;text-wrap:balance}
  .sub{color:var(--muted);font-size:.95rem;margin:2px 0 0}
  .stats{display:flex;flex-wrap:wrap;gap:10px}
  .stat{background:var(--surface);border:1px solid var(--border);border-radius:10px;
        padding:10px 14px;min-width:120px}
  .stat b{display:block;font-size:1.25rem;font-weight:700;font-variant-numeric:tabular-nums}
  .stat span{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
  .legend{display:flex;flex-wrap:wrap;gap:14px;align-items:center;font-size:.82rem;color:var(--muted)}
  .legend .sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:5px;vertical-align:-1px}
  .grad{height:11px;width:130px;border-radius:3px;background:linear-gradient(90deg,hsl(18,85%,93%),hsl(18,85%,33%))}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px}
  .scroll{overflow-x:auto}
  .grid{display:grid;grid-template-columns:26px repeat(var(--nl),16px);gap:2px;align-items:center}
  .grid .ylab{font-size:.7rem;color:var(--muted);text-align:right;padding-right:6px;font-weight:600}
  .grid .cell{width:16px;height:22px;border-radius:2px}
  .condgrid{display:grid;grid-template-columns:26px repeat(var(--nl),16px);gap:2px;align-items:center;margin-bottom:6px}
  .condgrid .cell{width:16px;height:6px;border-radius:2px}
  .xlab{grid-column:2 / -1;display:grid;grid-template-columns:repeat(var(--nl),16px);gap:2px;margin-top:4px}
  .xlab span{font-size:.62rem;color:var(--muted);text-align:center;transform:translateX(-1px)}
  .tooltip{position:fixed;background:var(--surface);border:1px solid var(--border);border-radius:8px;
           padding:8px 10px;font-size:.78rem;pointer-events:none;z-index:5;box-shadow:0 4px 16px rgba(0,0,0,.12);display:none}
  .phases{display:grid;gap:6px;font-size:.85rem}
  .phases .row{display:grid;grid-template-columns:76px 1fr auto;gap:12px;padding:6px 0;border-bottom:1px solid var(--border)}
  .phases .row:last-child{border-bottom:0}
  .phases .range{font-weight:600;font-variant-numeric:tabular-nums}
  .phases .desc{color:var(--muted)}
  .phases .mae{font-weight:600;font-variant-numeric:tabular-nums}
</style>

<div class="wrap">
  <header>
    <h1>Silverstone 2025 — per lap &times; sector error</h1>
    <p class="sub">Condition-aware-anchor model, frozen RF (2023+24 &rarr; 2025). Each cell is the mean absolute error (seconds) of one lap-sector across all drivers.</p>
  </header>

  <div class="stats">
    <div class="stat"><b>1.319 s</b><span>overall MAE</span></div>
    <div class="stat"><b id="st-worst">—</b><span>worst cell</span></div>
    <div class="stat"><b id="st-best">—</b><span>best cell</span></div>
    <div class="stat"><b>51</b><span>laps</span></div>
    <div class="stat"><b>3</b><span>sectors</span></div>
  </div>

  <div class="legend">
    <span><i class="sw" style="background:var(--wet)"></i>Wet (rain falling)</span>
    <span><i class="sw" style="background:var(--drying)"></i>Drying (dry flag, still on inters)</span>
    <span><i class="sw" style="background:var(--dry)"></i>Dry</span>
    <span style="margin-left:6px">MAE scale:</span>
    <span class="grad"></span>
    <span>0 s</span><span style="opacity:.7">&rarr; 10 s</span>
  </div>

  <div class="card">
    <div class="scroll">
      <div class="condgrid" id="condgrid"></div>
      <div class="grid" id="heatmap"></div>
      <div class="xlab" id="xlabs"></div>
    </div>
  </div>

  <div class="card">
    <div style="font-weight:600;margin-bottom:6px">Race phases</div>
    <div class="phases">
      <div class="row"><span class="range">L2&ndash;10</span><span class="desc">Wet start, mixed tyres, field spread out</span><span class="mae">0.5&ndash;9.0 s</span></div>
      <div class="row"><span class="range">L11&ndash;15</span><span class="desc">Rain falling, everyone on inters</span><span class="mae">1.0&ndash;5.9 s</span></div>
      <div class="row"><span class="range">L18&ndash;24</span><span class="desc">Drying track &mdash; the hardest phase</span><span class="mae">0.2&ndash;9.8 s</span></div>
      <div class="row"><span class="range">L25&ndash;37</span><span class="desc">Stable inters &mdash; model nails it</span><span class="mae">0.09&ndash;1.2 s</span></div>
      <div class="row"><span class="range">L38&ndash;45</span><span class="desc">Inter &rarr; dry tyre switch</span><span class="mae">0.1&ndash;7.8 s</span></div>
      <div class="row"><span class="range">L46&ndash;52</span><span class="desc">Full dry &mdash; model nails it</span><span class="mae">0.13&ndash;0.65 s</span></div>
    </div>
  </div>
</div>

<div class="tooltip" id="tip"></div>

<script>
const DATA = __DATA__;
const LAPS = DATA.laps, SECTORS = DATA.sectors;
document.documentElement.style.setProperty('--nl', LAPS.length);

const COND = {WET:'var(--wet)', DRYING:'var(--drying)', DRY:'var(--dry)'};
function maeColor(mae){
  if(mae==null) return 'var(--cell-empty)';
  const t = Math.min(1, mae/10);
  return `hsl(18,85%,${93 - t*58}%)`;
}

// condition strip (per lap)
const cg = document.getElementById('condgrid');
cg.innerHTML = '<span class="ylab">cond</span>' + LAPS.map(l=>{
  const c = DATA.lapCond[l] || 'DRY';
  return `<div class="cell" title="Lap ${l}: ${c}" style="background:${COND[c]}"></div>`;
}).join('');

// heatmap
const hm = document.getElementById('heatmap');
let worst = {v:0, key:''}, best = {v:Infinity, key:''};
hm.innerHTML = SECTORS.map(s=>{
  const cells = LAPS.map(l=>{
    const k = l+','+s;
    const c = DATA.cells[k];
    const mae = c ? c.mae : null;
    if(mae!=null){ if(mae>worst.v) worst={v:mae,key:k}; if(mae<best.v) best={v:mae,key:k}; }
    const tip = c ? `Lap ${l} &middot; Sector ${s} &middot; ${c.cond} &middot; MAE ${c.mae.toFixed(3)} s (n=${c.n})` : `Lap ${l} &middot; Sector ${s} &middot; no data`;
    return `<div class="cell" data-k="${k}" data-tip="${tip}" style="background:${maeColor(mae)}"></div>`;
  }).join('');
  return `<span class="ylab">S${s}</span>` + cells;
}).join('');

// x labels (every 5 laps)
const xl = document.getElementById('xlabs');
xl.innerHTML = LAPS.map(l=>`<span>${l%5===0?l:''}</span>`).join('');

document.getElementById('st-worst').textContent = worst.v.toFixed(3)+' s';
document.getElementById('st-best').textContent = best.v.toFixed(3)+' s';

// hover tooltip
const tip = document.getElementById('tip');
hm.addEventListener('mouseover', e=>{
  const c = e.target.closest('.cell'); if(!c) return;
  const t = c.dataset.tip; if(!t) return;
  tip.innerHTML = t; tip.style.display='block';
});
hm.addEventListener('mousemove', e=>{
  tip.style.left = (e.clientX+12)+'px'; tip.style.top = (e.clientY+12)+'px';
});
hm.addEventListener('mouseout', e=>{ if(e.target.closest('.cell')) tip.style.display='none'; });
</script>
"""


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ca, tr_feats, _, _ = _prepare_lab(cond_anchors(tr), [])
    te_ca, te_feats, _, _ = _prepare_lab(cond_anchors(te), [])
    feats = [c for c in _features(tr_feats, tr_ca) if c in te_ca.columns]

    pre = build_preprocessor(feats).fit(tr_ca[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ca[feats]),
                                                    tr_ca[TARGET].to_numpy(float))
    sil = te_ca[te_ca["Session_Key"].to_numpy() == 9947].copy()
    p = rf.predict(pre.transform(sil[feats]))
    sil["err"] = np.abs(np.exp(sil[TARGET].to_numpy(float)) - np.exp(p))
    sil["wet"] = (sil["Rainfall"].to_numpy() == 1)
    sil["inter"] = (sil["Tyre"].to_numpy() == "INTERMEDIATE")

    laps = sorted(int(x) for x in sil["Lap"].unique())
    lap_cond = {}
    for lap in laps:
        g = sil[sil["Lap"] == lap]
        wet = g["wet"].mean() > 0.5
        inter = g["inter"].mean() > 0.5
        lap_cond[str(lap)] = "WET" if wet else ("DRYING" if inter else "DRY")

    cells = {}
    for (lap, sec), g in sil.groupby(["Lap", "Sector"]):
        wet = g["wet"].mean() > 0.5
        inter = g["inter"].mean() > 0.5
        cond = "WET" if wet else ("DRYING" if inter else "DRY")
        cells[f"{int(lap)},{int(sec)}"] = {"mae": float(g["err"].mean()), "cond": cond, "n": int(len(g))}

    data = {"laps": laps, "sectors": [1, 2, 3], "lapCond": lap_cond, "cells": cells}
    html = HTML.replace("__DATA__", json.dumps(data))
    out = ROOT / "silverstone_report.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out.name} ({len(laps)} laps, {len(cells)} cells, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
