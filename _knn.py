"""k-Nearest-Neighbours regression — "similar conditions -> similar duration".

Same features + split as the frozen RF (2023+24 -> 2025, Log_Duration). Features are
StandardScaler'd (KNN needs it). Neighbours are found ONCE at k=50 with brute force,
then k=5/10/20/50 and uniform-vs-distance weighting are read off the same search.

Also reports the naive mean baseline (predict the global mean) and the RF reference.

SCRATCH — does not modify main code.
"""

import json
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
K_MAX = 50
KS = [5, 10, 20, 50]


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
    log("loading + preparing ...")
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    log(f"features={len(feats)}  train={len(tr_ml):,}  test={len(te_ml):,}")

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    Xtr = pre.transform(tr_ml[feats])
    Xte = pre.transform(te_ml[feats])
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xte_s = sc.transform(Xte)
    ytr = tr_ml[TARGET].to_numpy(float)
    yte = te_ml[TARGET].to_numpy(float)
    te_g = green_mask(te_ml)

    # naive mean baseline
    naive_full = sec_mae(yte, np.full(len(yte), np.mean(ytr)))

    # RF reference
    log("RF reference ...")
    t = time.time()
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(Xtr, ytr)
    p_rf = rf.predict(Xte)
    rf_full = sec_mae(yte, p_rf)
    rf_green = sec_mae(yte[te_g], p_rf[te_g])
    log(f"  RF full={rf_full:.4f} green={rf_green:.4f} ({time.time()-t:.0f}s)")

    # RF feature importances (for supervised feature selection on KNN)
    feat_names = list(pre.get_feature_names_out())
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    log("top-15 features by RF importance:")
    for i in order[:15]:
        log(f"    {feat_names[i]:44s} {imp[i]:.4f}")

    # one neighbour search at k_max
    log(f"KNN neighbour search (brute, k={K_MAX}, n_jobs={CORES}) ...")
    t = time.time()
    nn = NearestNeighbors(n_neighbors=K_MAX, algorithm="brute", n_jobs=CORES).fit(Xtr_s)
    dist, idx = nn.kneighbors(Xte_s)
    y_neigh = ytr[idx]  # (test, k_max)
    log(f"  search done ({time.time()-t:.0f}s)")

    results = []
    for k in KS:
        d = dist[:, :k]
        yn = y_neigh[:, :k]
        p_uni = yn.mean(axis=1)
        w = 1.0 / (d + 1e-8)
        p_dist = (yn * w).sum(axis=1) / w.sum(axis=1)
        for name, p in [("uniform", p_uni), ("distance", p_dist)]:
            full = sec_mae(yte, p)
            green = sec_mae(yte[te_g], p[te_g])
            results.append({"k": k, "weights": name, "full": full, "green": green})
            log(f"  k={k:2d} {name:8s} full={full:.4f} green={green:.4f}")

    # ---- feature selection -> KNN (KNN is weak in high dims) ----
    log("feature-selected KNN (top-N by RF importance, k=10 distance-weighted) ...")
    fs_results = []
    for n_feats in [5, 10, 15, 20, 30]:
        cols = order[:n_feats]
        nn2 = NearestNeighbors(n_neighbors=K_MAX, algorithm="brute", n_jobs=CORES).fit(Xtr_s[:, cols])
        d2, i2 = nn2.kneighbors(Xte_s[:, cols])
        d10 = d2[:, :10]
        yn10 = ytr[i2[:, :10]]
        w10 = 1.0 / (d10 + 1e-8)
        p = (yn10 * w10).sum(axis=1) / w10.sum(axis=1)
        full = sec_mae(yte, p)
        green = sec_mae(yte[te_g], p[te_g])
        fs_results.append({"n_features": n_feats, "full": full, "green": green})
        log(f"  top-{n_feats:2d} features: full={full:.4f} green={green:.4f}")

    print("\n" + "=" * 60)
    print(f"{'model':26s} {'full':>8s} {'green':>8s}")
    print("-" * 60)
    print(f"{'naive mean':26s} {naive_full:8.4f} {'--':>8s}")
    print(f"{'RF baseline':26s} {rf_full:8.4f} {rf_green:8.4f}")
    for r in results:
        label = f"KNN k={r['k']} {r['weights']}"
        print(f"{label:26s} {r['full']:8.4f} {r['green']:8.4f}")
    print("-" * 60)
    for r in fs_results:
        print(f"{'KNN top-'+str(r['n_features'])+' feat':26s} {r['full']:8.4f} {r['green']:8.4f}")
    print("=" * 60)

    out = {"naive_mean_full": naive_full, "rf": {"full": rf_full, "green": rf_green},
           "knn": results, "knn_feature_selected": fs_results, "features": len(feats)}
    with open(ROOT / "knn_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"saved -> knn_results.json  total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
