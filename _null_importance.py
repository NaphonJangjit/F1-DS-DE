"""Null-importance feature selection + RF feature importance.

Null importance: a feature is "real" only if its importance from a model trained on
the TRUE target exceeds the distribution of importances from models trained on
SHUFFLED targets (destroying any signal). Uses HistGradientBoosting (fast) for both
real and null runs (N=20 shuffles).

Also prints the full RF feature-importance ranking.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
N_NULL = 20


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    Xtr = pre.transform(tr_ml[feats])
    ytr = tr_ml[TARGET].to_numpy(float)
    names = list(pre.get_feature_names_out())
    log(f"features={len(names)}  warm={len(ytr):,}")

    # ---- RF importance (print) ----
    log("RF feature importance ...")
    t = time.time()
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(Xtr, ytr)
    rf_imp = rf.feature_importances_
    log(f"  ({time.time()-t:.0f}s)")

    # ---- null importance (fast model) ----
    log(f"null importance ({N_NULL} shuffles, ExtraTrees) ...")
    t = time.time()
    et = ExtraTreesRegressor(n_estimators=100, max_features=0.5, min_samples_leaf=2,
                             n_jobs=CORES, random_state=42)
    real = et.fit(Xtr, ytr).feature_importances_
    rng = np.random.RandomState(42)
    null = np.zeros((N_NULL, len(names)))
    for i in range(N_NULL):
        y_shuf = rng.permutation(ytr)
        null[i] = et.fit(Xtr, y_shuf).feature_importances_
        if (i + 1) % 5 == 0:
            log(f"  shuffle {i+1}/{N_NULL} ({time.time()-t:.0f}s)")
    null_mean = null.mean(axis=0)
    null_p95 = np.percentile(null, 95, axis=0)
    null_max = null.max(axis=0)

    # comparison
    rows = []
    for i, n in enumerate(names):
        rows.append({"feature": n, "rf_imp": rf_imp[i], "real_imp": real[i],
                     "null_mean": null_mean[i], "null_p95": null_p95[i],
                     "pass_p95": real[i] > null_p95[i], "pass_max": real[i] > null_max[i]})
    df = pd.DataFrame(rows).sort_values("rf_imp", ascending=False).reset_index(drop=True)

    print("\n=== RF feature importance (full) ===")
    for i, r in df.iterrows():
        print(f"  {i+1:3d} {r['feature']:40s} {r['rf_imp']:.5f}")

    print(f"\n=== null-importance selection (real vs null, HGB) ===")
    print(f"{'feature':40s} {'real':>8s} {'null_mean':>9s} {'null_p95':>8s}  pass")
    print("-" * 80)
    for _, r in df.iterrows():
        mark = "KEEP" if r["pass_p95"] else "drop"
        print(f"{r['feature']:40s} {r['real_imp']:8.5f} {r['null_mean']:9.5f} "
              f"{r['null_p95']:8.5f}  {mark}")

    n_keep = int(df["pass_p95"].sum())
    print(f"\nfeatures passing null-importance (real > null p95): {n_keep}/{len(df)}")
    print("kept:", ", ".join(df[df["pass_p95"]]["feature"].tolist()))
    print(f"total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
