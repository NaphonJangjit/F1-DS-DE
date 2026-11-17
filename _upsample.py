"""Upsample rare regimes (wet / SC / VSC) to correct data bias, then retrain
RF + linear models. SCRATCH — does not modify main code.

- Training split = 2023+2024, test = 2025 (temporal), same as _rf_drift.py.
- Upsampling is implemented as inverse-frequency sample_weight (mathematically
  equivalent to duplicating the rare rows, without physically copying them).
  Each rare regime (Rainfall==1, Track_Status in {SC,SC_ENDING},
  {VSC,VSC_ENDING}) is up-weighted to ~TARGET_SHARE of total training weight,
  combined multiplicatively (rare+rare is rarer) and capped at CAP.
- Baseline (no weights) vs Upsample for RandomForest / LinearRegression / Ridge.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent

TARGET_SHARE = 0.10      # each rare regime -> ~10% of total training weight
CAP = 20.0               # max multiplicative weight per row
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)


def sec_metrics(y_log, pred_log):
    a = np.exp(np.asarray(y_log, float))
    p = np.exp(np.asarray(pred_log, float))
    return (float(mean_absolute_error(a, p)),
            float(np.sqrt(mean_squared_error(a, p))))


def make_rf():
    return RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                 min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                 random_state=42)


def compute_weights(df):
    """Return (weights, per-condition multipliers, dict of masks)."""
    wet = (df["Rainfall"] == 1).to_numpy()
    sc = df["Track_Status"].isin(["SC", "SC_ENDING"]).to_numpy()
    vsc = df["Track_Status"].isin(["VSC", "VSC_ENDING"]).to_numpy()

    def mult(mask):
        s = float(mask.mean())
        return 1.0 if s >= TARGET_SHARE or s <= 0 else TARGET_SHARE / s

    m_wet, m_sc, m_vsc = mult(wet), mult(sc), mult(vsc)
    w = np.ones(len(df), dtype=float)
    w = np.where(wet, w * m_wet, w)
    w = np.where(sc, w * m_sc, w)
    w = np.where(vsc, w * m_vsc, w)
    w = np.minimum(w, CAP)
    return w, {"wet": m_wet, "sc": m_sc, "vsc": m_vsc}, {"wet": wet, "sc": sc, "vsc": vsc}


def cond_mae(yte, pred, mask):
    return sec_metrics(yte[mask], pred[mask])[0] if mask.any() else float("nan")


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = _features(tr_feats, tr_ml)
    feats = [c for c in feats if c in te_ml.columns]

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    Xtr = pre.transform(tr_ml[feats])
    Xte = pre.transform(te_ml[feats])
    ytr = tr_ml[TARGET].to_numpy(float)
    yte = te_ml[TARGET].to_numpy(float)

    w, mults, tr_masks = compute_weights(tr_ml)
    print(f"cores={CORES}  features={len(feats)}  train_rows={len(tr_ml):,}  test_rows={len(te_ml):,}")
    print(f"target_share={TARGET_SHARE}  cap={CAP}")
    print("train condition shares: " + ", ".join(
        f"{k}={tr_masks[k].mean()*100:.2f}%" for k in tr_masks))
    print("multipliers: " + ", ".join(f"{k}={v:.2f}" for k, v in mults.items()))
    eff = {k: float((w * tr_masks[k]).sum() / w.sum() * 100) for k in tr_masks}
    print("weighted (effective) shares: " + ", ".join(f"{k}={v:.2f}%" for k, v in eff.items()))
    print(f"weight range [{w.min():.2f}, {w.max():.2f}]  mean={w.mean():.3f}")

    # condition masks on test
    te_wet = (te_ml["Rainfall"] == 1).to_numpy()
    te_sc = te_ml["Track_Status"].isin(["SC", "SC_ENDING"]).to_numpy()
    te_vsc = te_ml["Track_Status"].isin(["VSC", "VSC_ENDING"]).to_numpy()
    te_clean = ~(te_wet | te_sc | te_vsc)
    cond = {"wet": te_wet, "sc": te_sc, "vsc": te_vsc, "clean": te_clean}
    print(f"\ntest condition sizes: " + ", ".join(f"{k}={int(cond[k].sum()):,}" for k in cond))

    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)

    models = [
        ("RandomForest", make_rf(), Xtr, Xte),
        ("LinearRegression", LinearRegression(), Xtr_s, Xte_s),
        ("Ridge", Ridge(alpha=1.0), Xtr_s, Xte_s),
    ]

    header = f"\n{'model':16s} {'variant':9s} {'overall':>9s} {'wet':>8s} {'sc':>8s} {'vsc':>8s} {'clean':>8s}"
    print(header)
    print("-" * len(header))

    summary = []
    preds = {}
    for name, est, Xa, Xb in models:
        for variant, sw in [("base", None), ("upsample", w)]:
            t = time.time()
            est.fit(Xa, ytr, sample_weight=sw)
            p = est.predict(Xb)
            fit_s = time.time() - t
            o_mae, o_rmse = sec_metrics(yte, p)
            line = (f"{name:16s} {variant:9s} {o_mae:9.4f} "
                    + " ".join(f"{cond_mae(yte, p, cond[k]):8.4f}" for k in ["wet", "sc", "vsc", "clean"]))
            print(line + f"   ({fit_s:.0f}s)")
            preds[(name, variant)] = p
            summary.append({"model": name, "variant": variant, "mae": o_mae, "rmse": o_rmse,
                            **{f"mae_{k}": cond_mae(yte, p, cond[k]) for k in cond}})

    # per-race RF comparison (base vs upsample) for the hardest races
    country_of = te.groupby("Session_Key")["Country"].first().to_dict()
    rf_base, rf_up = preds[("RandomForest", "base")], preds[("RandomForest", "upsample")]
    race_rows = []
    for sk, g in te_ml.groupby("Session_Key", sort=False):
        idx = g.index.to_numpy()
        b = sec_metrics(yte[idx], rf_base[idx])[0]
        u = sec_metrics(yte[idx], rf_up[idx])[0]
        race_rows.append((str(country_of.get(sk, "?")), len(idx), b, u))
    race_rows.sort(key=lambda r: -r[2])
    print("\nRF per-race (hardest 8, base -> upsample):")
    print(f"{'race':22s} {'rows':>6s} {'base':>8s} {'upsample':>8s} {'delta':>8s}")
    for name, n, b, u in race_rows[:8]:
        print(f"{name:22s} {n:>6d} {b:>8.4f} {u:>8.4f} {u-b:>+8.4f}")

    print(f"\ntotal={time.time()-t0:.0f}s")
    with open(ROOT / "upsample_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary,
                   "multipliers": mults, "eff_shares": eff,
                   "target_share": TARGET_SHARE, "cap": CAP,
                   "race_compare": [{"country": r[0], "rows": r[1], "base": r[2], "upsample": r[3]}
                                    for r in race_rows]}, f, indent=2)
    print("saved -> upsample_results.json")
    print("DONE")


if __name__ == "__main__":
    main()
