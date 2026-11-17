"""KNN on the no-anchor feature set (64 features), with feature selection.

Anchors (pace history) removed -> geometry/telemetry/weather remain. KNN is weak in
high dims, so select top-N by RF importance (of the no-anchor RF) and use k=10
distance-weighted (the best KNN config found earlier).

Compare vs RF-no-anchor (0.756) and KNN-with-anchors (0.317).
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}
ANCHORS = ["Best_Duration_So_Far", "Clean_Pace", "Prev_Duration",
           "Prev_Duration_roll5_mean", "Prev_Duration_roll5_slope",
           "Prev_Sector_Duration", "Prev_Sector_Duration_roll5_mean",
           "Prev_Sector_Duration_roll5_slope"]
K_MAX = 50


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def green_mask(df):
    d = df[["Session_Key", "Driver", "Lap", "Track_Status"]].copy()
    is_sc = d["Track_Status"].isin(SC_SET).to_numpy()
    laps = (d.assign(_is_sc=is_sc)
            .groupby(["Session_Key", "Driver", "Lap"], sort=False)["_is_sc"].max()
            .reset_index().sort_values(["Session_Key", "Driver", "Lap"]))
    laps["_prev_sc"] = laps.groupby(["Session_Key", "Driver"])["_is_sc"].shift(1).fillna(False)
    merged = d[["Session_Key", "Driver", "Lap"]].merge(
        laps[["Session_Key", "Driver", "Lap", "_prev_sc"]],
        on=["Session_Key", "Driver", "Lap"], how="left")
    return ~(is_sc | merged["_prev_sc"].to_numpy().astype(bool))


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    feats_na = [c for c in feats if c not in ANCHORS]

    pre = build_preprocessor(feats_na).fit(tr_ml[feats_na])
    Xtr = pre.transform(tr_ml[feats_na])
    Xte = pre.transform(te_ml[feats_na])
    ytr = tr_ml[TARGET].to_numpy(float)
    yte = te_ml[TARGET].to_numpy(float)
    te_g = green_mask(te_ml)
    te_wet = te_ml["Rainfall"].to_numpy() == 1

    # RF no-anchor for importance + reference
    log("RF no-anchor ...")
    t = time.time()
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(Xtr, ytr)
    p_rf = rf.predict(Xte)
    order = np.argsort(rf.feature_importances_)[::-1]
    rf_m = {"full": sec_mae(yte, p_rf), "green": sec_mae(yte[te_g], p_rf[te_g]),
            "wet": sec_mae(yte[te_wet], p_rf[te_wet]), "dry": sec_mae(yte[~te_wet], p_rf[~te_wet])}
    log(f"  RF no-anchor full={rf_m['full']:.4f} wet={rf_m['wet']:.4f} ({time.time()-t:.0f}s)")

    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xte_s = sc.transform(Xte)

    print("\n=== KNN on no-anchor (k=10 distance-weighted) ===")
    print(f"{'top-N':6s} {'full':>8s} {'green':>8s} {'wet':>8s} {'dry':>8s}")
    for n_feats in [5, 10, 15, 20, 30]:
        cols = order[:n_feats]
        nn = NearestNeighbors(n_neighbors=K_MAX, algorithm="brute", n_jobs=CORES).fit(Xtr_s[:, cols])
        dist, idx = nn.kneighbors(Xte_s[:, cols])
        d = dist[:, :10]
        yn = ytr[idx[:, :10]]
        w = 1.0 / (d + 1e-8)
        p = (yn * w).sum(axis=1) / w.sum(axis=1)
        print(f"top-{n_feats:<3d} {sec_mae(yte, p):8.4f} {sec_mae(yte[te_g], p[te_g]):8.4f} "
              f"{sec_mae(yte[te_wet], p[te_wet]):8.4f} {sec_mae(yte[~te_wet], p[~te_wet]):8.4f}")

    print(f"\nRF no-anchor      full={rf_m['full']:.4f} green={rf_m['green']:.4f} wet={rf_m['wet']:.4f} dry={rf_m['dry']:.4f}")
    print("(reference: RF with-anchors full=0.2649 / KNN top-5 with-anchors=0.317)")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
