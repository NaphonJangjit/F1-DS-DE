"""Generate silverstone_pred_actual.html — predicted vs actual sector time for two drivers."""

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
SK = 9947
DRIVERS = [(27, "Nico Hulkenberg"), (1, "Max Verstappen")]
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}


def drop_sc_vsc(df):
    """Remove SC/VSC sectors + the lap immediately following any SC/VSC lap."""
    d = df.copy()
    d["_is_sc"] = d["Track_Status"].isin(SC_SET)
    laps = (d[["Session_Key", "Driver", "Lap", "_is_sc"]]
            .groupby(["Session_Key", "Driver", "Lap"], as_index=False).max())
    laps = laps.sort_values(["Session_Key", "Driver", "Lap"])
    laps["_prev_sc"] = laps.groupby(["Session_Key", "Driver"])["_is_sc"].shift(1).fillna(False)
    d = d.merge(laps[["Session_Key", "Driver", "Lap", "_prev_sc"]],
                on=["Session_Key", "Driver", "Lap"], how="left")
    drop = d["_is_sc"] | d["_prev_sc"].fillna(False)
    return d[~drop].drop(columns=["_is_sc", "_prev_sc"])

HTML = """<meta name="claude:viewport-fit-sides" content="cover">
<title>Silverstone Predicted vs Actual</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.32.0/plotly.min.js"></script>
<style>
  :root{--bg:#f6f7f9;--surface:#fff;--fg:#1c1f24;--muted:#5c626d;--border:#e4e7ec;--accent:#e26b1a;--pred:#4c78a8}
  @media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){--bg:#131519;--surface:#1a1d22;--fg:#e8eaee;--muted:#9aa2ad;--border:#2a2e36;--accent:#f0843a;--pred:#5f8fd0;color-scheme:dark}}
  :root[data-theme="dark"]{--bg:#131519;--surface:#1a1d22;--fg:#e8eaee;--muted:#9aa2ad;--border:#2a2e36;--accent:#f0843a;--pred:#5f8fd0;color-scheme:dark}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font-family:'Inter',system-ui,sans-serif;
       padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:32px 48px}
  .wrap{max-width:1080px;margin:0 auto;display:grid;gap:24px}
  h1{font-size:1.4rem;margin:0;font-weight:700;letter-spacing:-.01em}
  .sub{color:var(--muted);font-size:.93rem;margin:2px 0 0}
  .legend{font-size:.82rem;color:var(--muted)}
  .legend .l{display:inline-block;width:22px;height:2px;vertical-align:3px;margin:0 6px 0 14px}
  .legend .a{background:var(--accent)} .legend .p{background:var(--pred);opacity:.7}
  .driver{border:1px solid var(--border);border-radius:14px;background:var(--surface);padding:18px}
  .driver h2{margin:0 0 4px;font-size:1.1rem;font-weight:700}
  .driver .stat{color:var(--muted);font-size:.82rem;margin-bottom:12px}
  .driver .stat b{color:var(--fg);font-variant-numeric:tabular-nums}
  .charts{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
  .chart{min-width:0;height:230px}
  @media (max-width:720px){.charts{grid-template-columns:1fr}}
</style>

<div class="wrap">
  <header>
    <h1>Silverstone 2025 — predicted vs actual sector time</h1>
    <p class="sub">Condition-aware anchor model, green-only (SC/VSC excluded). Solid = actual, dashed = predicted.</p>
    <div class="legend"><span class="l a"></span>actual<span class="l p"></span>predicted</div>
  </header>
  <div id="drivers"></div>
</div>

<script>
const DATA = __DATA__;
const SURFACE = getComputedStyle(document.body).getPropertyValue('--surface').trim();
const FG = getComputedStyle(document.body).getPropertyValue('--fg').trim();
const MUTED = getComputedStyle(document.body).getPropertyValue('--muted').trim();
const ACCENT = getComputedStyle(document.body).getPropertyValue('--accent').trim();
const PRED = getComputedStyle(document.body).getPropertyValue('--pred').trim();
const GRID = 'rgba(128,138,150,0.25)';

const host = document.getElementById('drivers');
DATA.forEach((d, di) => {
  const card = document.createElement('div');
  card.className = 'driver';
  card.innerHTML = `<h2>${d.name} <span style="color:var(--muted);font-weight:500">#${d.num}</span></h2>
    <div class="stat">driver MAE <b>${d.mae.toFixed(3)} s</b> &middot; ${d.n} sectors</div>
    <div class="charts">${[1,2,3].map(s=>`<div class="chart" id="c${di}_${s}"></div>`).join('')}</div>`;
  host.appendChild(card);

  [1,2,3].forEach(s => {
    const t = d.sectors[s];
    const traces = [
      {x:t.lap, y:t.actual, mode:'lines', name:'actual', line:{color:ACCENT, width:2},
       hovertemplate:'Lap %{x}<br>actual %{y:.2f}s<extra></extra>'},
      {x:t.lap, y:t.pred, mode:'lines', name:'predicted', line:{color:PRED, width:1.6, dash:'dash'},
       hovertemplate:'Lap %{x}<br>predicted %{y:.2f}s<extra></extra>'}
    ];
    Plotly.newPlot(`c${di}_${s}`, traces, {
      title:{text:`Sector ${s}`, font:{size:12, color:FG}},
      margin:{l:44, r:10, t:30, b:34},
      paper_bgcolor:SURFACE, plot_bgcolor:SURFACE,
      font:{family:'Inter', size:11, color:MUTED},
      xaxis:{title:{text:'lap'}, gridcolor:GRID, zeroline:false},
      yaxis:{title:{text:'seconds'}, gridcolor:GRID, zeroline:false},
      legend:{orientation:'h', y:1.12, font:{size:10, color:MUTED}},
      hovermode:'x unified'
    }, {displayModeBar:false, responsive:true});
  });
});
</script>
"""


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = drop_sc_vsc(pd.read_parquet(ROOT / "data_2025.parquet"))
    tr = cond_anchors(tr)
    te = cond_anchors(te)

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats]),
                                                    tr_ml[TARGET].to_numpy(float))
    p_all = rf.predict(pre.transform(te_ml[feats]))
    overall = float(np.abs(np.exp(te_ml[TARGET].to_numpy(float)) - np.exp(p_all)).mean())
    print(f"green-only 2025 overall MAE = {overall:.4f}s")
    sil = te_ml[te_ml["Session_Key"].to_numpy() == SK].copy()
    p = rf.predict(pre.transform(sil[feats]))
    sil["actual"] = np.exp(sil[TARGET].to_numpy(float))
    sil["pred"] = np.exp(p)

    drivers = []
    for num, name in DRIVERS:
        d = sil[sil["Driver"].to_numpy() == num].sort_values("Lap")
        err = np.abs(d["actual"] - d["pred"])
        sectors = {}
        for s in (1, 2, 3):
            g = d[d["Sector"].to_numpy() == s]
            sectors[str(s)] = {"lap": [int(x) for x in g["Lap"]],
                               "actual": [round(float(x), 3) for x in g["actual"]],
                               "pred": [round(float(x), 3) for x in g["pred"]]}
        drivers.append({"name": name, "num": num, "mae": float(err.mean()),
                        "n": int(len(d)), "sectors": sectors})

    html = HTML.replace("__DATA__", json.dumps(drivers))
    out = ROOT / "silverstone_pred_actual.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out.name} ({time.time()-t0:.0f}s)")
    for d in drivers:
        print(f"  {d['name']:16s} MAE={d['mae']:.3f}s  n={d['n']}")


if __name__ == "__main__":
    main()
