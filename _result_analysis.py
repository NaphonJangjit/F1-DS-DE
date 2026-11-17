"""Computes the numbers for result_report.md (old baseline RF, no FE).

1) MAE by race (full / dry / intermediate)
2) Train-vs-test feature distribution shift (KS) per condition (DRY / INTER)
3) Residual analysis per condition (dispersion + directional bias)
4) Within-race comparison (Australia 2025 = dry + intermediate in the same race)
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
DRY = ["SOFT", "MEDIUM", "HARD"]
AU_SK = 9693


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats]),
                                                    tr_ml[TARGET].to_numpy(float))
    p_log = rf.predict(pre.transform(te_ml[feats]))
    y_log = te_ml[TARGET].to_numpy(float)
    y_s = np.exp(y_log)
    p_s = np.exp(p_log)
    res_s = y_s - p_s  # >0 = under-predicted (actual slower), <0 = over-predicted

    tyre_tr = tr_ml["Tyre"].to_numpy()
    tyre_te = te_ml["Tyre"].to_numpy()
    dry_te = np.isin(tyre_te, DRY)
    inter_te = tyre_te == "INTERMEDIATE"
    dry_tr = np.isin(tyre_tr, DRY)
    inter_tr = tyre_tr == "INTERMEDIATE"

    # ---- 1. MAE by race ----
    country = te.groupby("Session_Key")["Country"].first().to_dict()
    races = []
    for sk in np.unique(te_ml["Session_Key"].to_numpy()):
        m = te_ml["Session_Key"].to_numpy() == sk
        r = {"country": str(country.get(int(sk), sk)), "n": int(m.sum())}
        r["full_mae"] = float(mean_absolute_error(y_s[m], p_s[m]))
        dm = m & dry_te
        im = m & inter_te
        r["dry_mae"] = float(mean_absolute_error(y_s[dm], p_s[dm])) if dm.any() else None
        r["inter_mae"] = float(mean_absolute_error(y_s[im], p_s[im])) if im.any() else None
        r["n_dry"] = int(dm.sum()); r["n_inter"] = int(im.sum())
        races.append(r)
    races.sort(key=lambda r: -r["full_mae"])

    # ---- 2. train vs test distribution shift (KS) per feature per condition ----
    shifts = []
    for f in feats:
        if not pd.api.types.is_numeric_dtype(tr_ml[f]):
            continue
        trv = pd.to_numeric(tr_ml[f], errors="coerce")
        tev = pd.to_numeric(te_ml[f], errors="coerce")
        for cond, trmask, temask in [("DRY", dry_tr, dry_te), ("INTER", inter_tr, inter_te)]:
            a = trv[trmask].dropna().to_numpy()
            b = tev[temask].dropna().to_numpy()
            if len(a) < 100 or len(b) < 100:
                continue
            ks = float(stats.ks_2samp(a, b).statistic)
            shifts.append({"feature": f, "cond": cond, "ks": ks,
                           "train_mean": float(np.mean(a)), "test_mean": float(np.mean(b)),
                           "train_std": float(np.std(a)), "test_std": float(np.std(b))})
    shifts = pd.DataFrame(shifts)
    top = {c: shifts[shifts.cond == c].sort_values("ks", ascending=False).head(12)
           .to_dict("records") for c in ["DRY", "INTER"]}

    # ---- 3. residual analysis per condition ----
    resid = {}
    for c, mask in [("DRY", dry_te), ("INTER", inter_te)]:
        r = res_s[mask]
        resid[c] = {"n": int(mask.sum()), "mean_bias": float(r.mean()), "std": float(r.std()),
                    "median": float(np.median(r)), "p5": float(np.percentile(r, 5)),
                    "p95": float(np.percentile(r, 95)),
                    "frac_underpredicted": float((r > 0).mean()),
                    "frac_overpredicted": float((r < 0).mean()),
                    "mae": float(mean_absolute_error(y_s[mask], p_s[mask])),
                    "rmse": float(np.sqrt(mean_squared_error(y_s[mask], p_s[mask])))}

    # ---- 4. within-race (Australia): dry vs inter in the same race ----
    au = te_ml["Session_Key"].to_numpy() == AU_SK
    au_dry = au & dry_te
    au_inter = au & inter_te
    within = {
        "race": "Australia",
        "dry": {"n": int(au_dry.sum()),
                "mae": float(mean_absolute_error(y_s[au_dry], p_s[au_dry])),
                "mean_bias": float(res_s[au_dry].mean())},
        "inter": {"n": int(au_inter.sum()),
                  "mae": float(mean_absolute_error(y_s[au_inter], p_s[au_inter])),
                  "mean_bias": float(res_s[au_inter].mean())},
    }

    out = {"races": races, "shift_top": top, "residual": resid, "within_race": within}
    with open(ROOT / "result_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"=== MAE BY RACE (top 12) ===")
    for r in races[:12]:
        print(f"  {r['country']:18s} full={r['full_mae']:.3f} dry={r['dry_mae'] or float('nan'):.3f} "
              f"inter={r['inter_mae'] or float('nan'):.3f} (n={r['n']}, dry={r['n_dry']}, inter={r['n_inter']})")
    print(f"\n=== RESIDUAL (bias + dispersion) ===")
    for c, d in resid.items():
        print(f"  {c:5s} n={d['n']:,} MAE={d['mae']:.3f} bias_mean={d['mean_bias']:+.3f} std={d['std']:.3f} "
              f"under={d['frac_underpredicted']:.2%} over={d['frac_overpredicted']:.2%}")
    print(f"\n=== WITHIN-RACE (Australia) ===")
    print(f"  dry   n={within['dry']['n']} MAE={within['dry']['mae']:.3f} bias={within['dry']['mean_bias']:+.3f}")
    print(f"  inter n={within['inter']['n']} MAE={within['inter']['mae']:.3f} bias={within['inter']['mean_bias']:+.3f}")
    print(f"\n=== TOP SHIFTED FEATURES (DRY) ===")
    for x in top["DRY"][:6]:
        print(f"  {x['feature']:36s} ks={x['ks']:.3f} train={x['train_mean']:.2f}->test={x['test_mean']:.2f}")
    print(f"\n=== TOP SHIFTED FEATURES (INTER) ===")
    for x in top["INTER"][:6]:
        print(f"  {x['feature']:36s} ks={x['ks']:.3f} train={x['train_mean']:.2f}->test={x['test_mean']:.2f}")
    print(f"\n({time.time()-t0:.0f}s)  saved -> result_analysis.json")


if __name__ == "__main__":
    main()
