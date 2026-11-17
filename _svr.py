"""Kernel SVM (SVR) on the sector table.

Exact RBF-SVR is O(n^2)..O(n^3), so a full 138k-row fit is infeasible. Two honest
approaches:
  1. exact RBF SVR on a 6k-row random subsample (LIBSVM), several feature sets
  2. Nystroem RBF approximation + LinearSVR on the FULL data (linear-time kernel)

Feature sets (reusing the RF-importance ordering from the KNN experiment):
  top-5  = pace anchors + length   (where the signal is)
  top-15, full-72                  (the curse-of-dimensionality tail)

Same split/target as always (2023+24 -> 2025, Log_Duration). SCRATCH.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}
N_SUB = 6000
SEED = 42


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
    ytr = tr_ml[TARGET].to_numpy(float)
    yte = te_ml[TARGET].to_numpy(float)
    te_g = green_mask(te_ml)

    # RF reference + importance ordering
    log("RF reference ...")
    t = time.time()
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(Xtr, ytr)
    p_rf = rf.predict(Xte)
    rf_full = sec_mae(yte, p_rf)
    rf_green = sec_mae(yte[te_g], p_rf[te_g])
    order = np.argsort(rf.feature_importances_)[::-1]
    log(f"  RF full={rf_full:.4f} green={rf_green:.4f} ({time.time()-t:.0f}s)")

    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xte_s = sc.transform(Xte)

    feat_sets = {"top5": order[:5], "top15": order[:15], "full": np.arange(Xtr.shape[1])}

    # exact RBF SVR on subsample
    rng = np.random.RandomState(SEED)
    sub = rng.choice(len(ytr), size=N_SUB, replace=False)
    log(f"exact RBF SVR on subsample n={N_SUB:,} ...")

    results = [{"model": "RF baseline", "full": rf_full, "green": rf_green}]
    for fs_name, cols in feat_sets.items():
        Xsub = Xtr_s[sub][:, cols]
        ysub = ytr[sub]
        combos = [(1.0, 0.1)] if fs_name != "top5" else [(1.0, 0.1), (1.0, 0.02), (10.0, 0.1)]
        for C, eps in combos:
            t = time.time()
            svr = SVR(kernel="rbf", C=C, gamma="scale", epsilon=eps, cache_size=1000).fit(Xsub, ysub)
            p = svr.predict(Xte_s[:, cols])
            full = sec_mae(yte, p)
            green = sec_mae(yte[te_g], p[te_g])
            results.append({"model": f"SVR(rbf) {fs_name} C={C} eps={eps}", "full": full, "green": green,
                            "n_sv": int(svr.n_support_.sum())})
            log(f"  SVR(rbf) {fs_name:6s} C={C:4.1f} eps={eps:.2f} full={full:.4f} green={green:.4f} "
                f"n_sv={svr.n_support_.sum()} ({time.time()-t:.0f}s)")

    # Nystroem approximation on full data
    log("Nystroem RBF approx + LinearSVR (full data) ...")
    for fs_name in ["top5", "full"]:
        cols = feat_sets[fs_name]
        t = time.time()
        ny = Nystroem(kernel="rbf", gamma=None, n_components=200, random_state=42)
        Xtr_n = ny.fit_transform(Xtr_s[:, cols])
        Xte_n = ny.transform(Xte_s[:, cols])
        lsvr = LinearSVR(C=1.0, epsilon=0.1, random_state=42, max_iter=5000, dual="auto").fit(Xtr_n, ytr)
        p = lsvr.predict(Xte_n)
        full = sec_mae(yte, p)
        green = sec_mae(yte[te_g], p[te_g])
        results.append({"model": f"LinearSVR(nystroem) {fs_name}", "full": full, "green": green})
        log(f"  LinearSVR(nystroem) {fs_name:6s} full={full:.4f} green={green:.4f} ({time.time()-t:.0f}s)")

    print("\n" + "=" * 66)
    print(f"{'model':32s} {'full':>8s} {'green':>8s}")
    print("-" * 66)
    for r in results:
        print(f"{r['model']:32s} {r['full']:8.4f} {r['green']:8.4f}")
    print("=" * 66)
    print("(KNN top-5 for reference: full 0.3170 / green 0.2939)")

    with open(ROOT / "svr_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "n_sub": N_SUB, "features": len(feats)}, f, indent=2)
    log(f"saved -> svr_results.json  total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
