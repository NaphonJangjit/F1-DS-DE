"""RF concept-drift adaptation (SCRATCH — does not modify main code).

Protocol:
  - Warm: RandomForest on 2023+2024 (temporal).
  - Prequential: iterate 2025 race-by-race (chronological), predict-then-update.
  - Drift detection: DDM on per-race MAE_s (warning @2sigma, drift @3sigma).
  - On drift: "unlearn" stale data (keep only the most recent UNLEARN_KEEP sectors)
    and retrain RF, then reset the DDM state.
  - Baseline: frozen warm RF (no adaptation) for comparison.

Cores: half the logical cores, min 2 (n_jobs) — "multiple but not all".
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
UNLEARN_KEEP = 20000          # sectors retained after unlearning
DDM_WARN = 2.0
DDM_DRIFT = 3.0
MIN_OBS = 3                   # races before DDM may fire (let std stabilise)


def sec_metrics(y_log, pred_log):
    a = np.exp(np.asarray(y_log, float))
    p = np.exp(np.asarray(pred_log, float))
    return (float(mean_absolute_error(a, p)),
            float(np.sqrt(mean_squared_error(a, p))))


def make_rf():
    return RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                 min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                 random_state=42)


def update_ddm(d, e):
    d["n"] += 1
    d["s"] += e
    d["s2"] += e * e
    mean = d["s"] / d["n"]
    var = max(d["s2"] / d["n"] - mean * mean, 0.0)
    std = var ** 0.5
    if mean + std < d["pmin"] + d["smin"]:
        d["pmin"] = mean
        d["smin"] = std
    drift = (d["n"] >= MIN_OBS and d["smin"] > 1e-9
             and mean + std > d["pmin"] + DDM_DRIFT * d["smin"])
    return drift, mean, std


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
    ywarm = tr_ml[TARGET].to_numpy(float)

    print(f"cores={CORES}  n_estimators={N_EST}  max_features={MAX_FEATURES}  "
          f"features={len(feats)}  warm_rows={len(tr_ml):,}  test_rows={len(te_ml):,}")

    model = make_rf().fit(pre.transform(tr_ml[feats]), ywarm)

    # ---- chronological order of 2025 ----
    te_ml = te_ml.sort_values("Time_Start").reset_index(drop=True)
    Xall = pre.transform(te_ml[feats])
    yall = te_ml[TARGET].to_numpy(float)

    static_pred = model.predict(Xall)
    s_mae, s_rmse = sec_metrics(yall, static_pred)
    print(f"[STATIC frozen 23+24]  MAE_s={s_mae:.4f}  RMSE_s={s_rmse:.4f}")

    country_of = te.groupby("Session_Key")["Country"].first().to_dict()

    groups = list(te_ml.groupby("Session_Key", sort=False))

    ddm = {"n": 0, "s": 0.0, "s2": 0.0, "pmin": np.inf, "smin": np.inf}
    buf = tr_ml.copy()
    adaptive_pred = np.full(len(te_ml), np.nan)
    retrains = 0
    race_rows = []

    print(f"\n{'#':>2s} {'race':20s} {'rows':>6s} {'static':>8s} {'adaptive':>8s} "
          f"{'ddm_mean':>8s} {'ddm_std':>8s} {'drift?':>7s}")

    for i, (sk, g) in enumerate(groups):
        idx = g.index.to_numpy()
        y_r = yall[idx]
        p_r = model.predict(Xall[idx])
        adaptive_pred[idx] = p_r

        mae_r, _ = sec_metrics(y_r, p_r)
        s_r, _ = sec_metrics(y_r, static_pred[idx])
        drift, d_mean, d_std = update_ddm(ddm, mae_r)

        label = str(country_of.get(sk, "?"))
        print(f"{i:>2d} {label:20s} {len(idx):>6d} {s_r:>8.4f} {mae_r:>8.4f} "
              f"{d_mean:>8.4f} {d_std:>8.4f} {'DRIFT' if drift else '':>7s}")
        race_rows.append({"i": i, "session_key": int(sk), "country": label, "rows": len(idx),
                          "static_mae": s_r, "adaptive_mae": mae_r, "drift": drift})

        # add this race's data AFTER predicting (point-in-time)
        buf = pd.concat([buf, g], ignore_index=True)

        if drift:
            retrains += 1
            buf = buf.tail(UNLEARN_KEEP).reset_index(drop=True)
            t = time.time()
            model = make_rf().fit(pre.transform(buf[feats]), buf[TARGET].to_numpy(float))
            print(f"      -> UNLEARN (keep {len(buf):,} sectors) + retrain RF "
                  f"({time.time() - t:.1f}s)")
            ddm = {"n": 0, "s": 0.0, "s2": 0.0, "pmin": np.inf, "smin": np.inf}

    a_mae, a_rmse = sec_metrics(yall, adaptive_pred)
    print(f"\n[ADAPTIVE unlearn+retrain on drift]  MAE_s={a_mae:.4f}  RMSE_s={a_rmse:.4f}")
    print(f"retrains={retrains}  total={time.time() - t0:.0f}s")
    print(f"delta vs static: MAE_s={a_mae - s_mae:+.4f}  RMSE_s={a_rmse - s_rmse:+.4f}")

    out = {"cores": CORES, "n_estimators": N_EST, "max_features": MAX_FEATURES,
           "unlearn_keep": UNLEARN_KEEP, "features": len(feats),
           "warm_rows": len(tr_ml), "test_rows": len(te_ml),
           "static_mae": s_mae, "static_rmse": s_rmse,
           "adaptive_mae": a_mae, "adaptive_rmse": a_rmse,
           "retrains": retrains, "races": race_rows}
    with open(ROOT / "rf_drift_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved -> rf_drift_results.json")
    print("DONE")


if __name__ == "__main__":
    main()
