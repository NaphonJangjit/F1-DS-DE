"""Bring the outlier laps BACK + SMOTE the rare regimes (wet / SC / VSC).

Two changes vs the default pipeline:
  1. Disable `remove_duration_outliers` (monkeypatch -> no-op) so SC/VSC/wet slow
     laps are NOT thrown away as outliers.
  2. Oversample those rare regimes with SYNTHETIC samples (SMOTE-for-regression):
     for each wet/SC/VSC row, interpolate numeric features + target with a random
     k-NN neighbour *in the same regime*; the one-hot Track_Status is copied from
     the seed row. (imblearn is not installed, so SMOTE is implemented here.)

Baseline (no outlier filter, no SMOTE) vs SMOTE, for RF + linear models.
Test = 2025, also with the outlier filter disabled (fair comparison).
SCRATCH — does not modify main code.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

import f1pit.wind_rain_lab as wrl
from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

# ---- keep the outlier (slow SC/VSC/wet) laps -------------------------------
wrl.remove_duration_outliers = lambda df, k=4.0, mad_floor=0.5: (df, 0)

ROOT = Path(__file__).resolve().parent
TARGET_RARE_SHARE = 0.20   # after SMOTE, rare regimes ~20% of training weight
SMOTE_K = 5                # neighbours considered per seed
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
REGIMES = ("wet", "sc", "vsc")


def sec_metrics(y_log, pred_log):
    a = np.exp(np.asarray(y_log, float)); p = np.exp(np.asarray(pred_log, float))
    return float(mean_absolute_error(a, p)), float(np.sqrt(mean_squared_error(a, p)))


def make_rf():
    return RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                 min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                 random_state=42)


def regime_labels(df):
    wet = (df["Rainfall"] == 1).to_numpy()
    sc = df["Track_Status"].isin(["SC", "SC_ENDING"]).to_numpy()
    vsc = df["Track_Status"].isin(["VSC", "VSC_ENDING"]).to_numpy()
    return np.where(wet, "wet", np.where(sc, "sc", np.where(vsc, "vsc", "clean")))


def smote_regression(X, y, regime, num_idx, n_synth_per_seed, k=SMOTE_K, seed=42):
    """Interpolate numeric features (+target) between same-regime neighbours;
    one-hot (non-num) columns copied from the seed row."""
    rng = np.random.RandomState(seed)
    Xnum = X[:, num_idx].astype(float)
    sc = StandardScaler().fit(Xnum)
    Xs = sc.transform(Xnum)
    SX, Sy = [], []
    for reg in REGIMES:
        idx = np.where(regime == reg)[0]
        if len(idx) < 2:
            continue
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(idx))).fit(Xs[idx])
        neigh = nn.kneighbors(Xs[idx], return_distance=False)
        if neigh.shape[1] > 1:
            neigh = neigh[:, 1:]          # drop self
        for _ in range(int(n_synth_per_seed * len(idx))):
            i = rng.randint(len(idx))
            j = rng.choice(neigh[i])
            a = rng.uniform(0.0, 1.0)
            row = X[idx[i]].astype(float).copy()
            row[num_idx] = X[idx[i], num_idx] + a * (X[idx[j], num_idx] - X[idx[i], num_idx])
            SX.append(row)
            Sy.append(y[idx[i]] + a * (y[idx[j]] - y[idx[i]]))
    return np.asarray(SX), np.asarray(Sy)


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

    names = list(pre.get_feature_names_out())
    cat_idx = [i for i, nm in enumerate(names) if nm.startswith("cat__")]
    num_idx = [i for i, nm in enumerate(names) if nm.startswith("num__")]

    reg_tr = regime_labels(tr_ml)
    reg_te = regime_labels(te_ml)
    te_masks = {r: (reg_te == r) for r in REGIMES}
    te_masks["clean"] = reg_te == "clean"

    rare_n = int((reg_tr != "clean").sum())
    n = len(ytr)
    share_now = rare_n / n
    n_synth = int((TARGET_RARE_SHARE * n - rare_n) / (1 - TARGET_RARE_SHARE)) if TARGET_RARE_SHARE > share_now else 0
    per_seed = max(1, round(n_synth / rare_n)) if n_synth > 0 else 0

    print(f"cores={CORES}  features={len(feats)}  train_rows={n:,}  test_rows={len(yte):,}")
    print(f"rare (wet|sc|vsc) before SMOTE: {rare_n:,} ({share_now*100:.1f}%)  -> target {TARGET_RARE_SHARE*100:.0f}%")
    print(f"SMOTE: {n_synth:,} synthetic samples  ({per_seed} per rare seed, k={SMOTE_K})")

    # ---- SMOTE augment -----------------------------------------------------
    SX, Sy = smote_regression(Xtr, ytr, reg_tr, num_idx, per_seed)
    Xaug = np.vstack([Xtr, SX])
    yaug = np.hstack([ytr, Sy])
    print(f"augmented train: {len(yaug):,} rows  (rare share now {n_synth + rare_n:.0f}/{len(yaug):,} = {(rare_n+n_synth)/len(yaug)*100:.1f}%)")

    models = [("RandomForest", make_rf(), False),
              ("LinearRegression", LinearRegression(), True),
              ("Ridge", Ridge(alpha=1.0), True)]

    header = f"\n{'model':16s} {'variant':9s} {'overall':>9s} {'wet':>8s} {'sc':>8s} {'vsc':>8s} {'clean':>8s}"
    print(header); print("-" * len(header))
    preds = {}
    for name, est, needs_scale in models:
        for variant, Xa, ya in [("base", Xtr, ytr), ("SMOTE", Xaug, yaug)]:
            t = time.time()
            if needs_scale:
                sc = StandardScaler().fit(Xa)
                est.fit(sc.transform(Xa), ya)
                p = est.predict(sc.transform(Xte))
            else:
                est.fit(Xa, ya)
                p = est.predict(Xte)
            dt = time.time() - t
            o_mae, o_rmse = sec_metrics(yte, p)
            print(f"{name:16s} {variant:9s} {o_mae:9.4f} "
                  + " ".join(f"{cond_mae(yte, p, te_masks[r]):8.4f}" for r in ["wet", "sc", "vsc", "clean"])
                  + f"   ({dt:.0f}s)")
            preds[(name, variant)] = p

    # per-race RF comparison
    country_of = te.groupby("Session_Key")["Country"].first().to_dict()
    rf_b = preds[("RandomForest", "base")]; rf_s = preds[("RandomForest", "SMOTE")]
    races = []
    for sk, g in te_ml.groupby("Session_Key", sort=False):
        idx = g.index.to_numpy()
        races.append((str(country_of.get(sk, "?")), len(idx),
                      sec_metrics(yte[idx], rf_b[idx])[0], sec_metrics(yte[idx], rf_s[idx])[0]))
    races.sort(key=lambda r: -r[2])
    print("\nRF per-race (hardest 10, base -> SMOTE):")
    print(f"{'race':22s} {'rows':>6s} {'base':>8s} {'SMOTE':>8s} {'delta':>8s}")
    for name, n, b, s in races[:10]:
        print(f"{name:22s} {n:>6d} {b:>8.4f} {s:>8.4f} {s-b:>+8.4f}")

    print(f"\ntotal={time.time()-t0:.0f}s  DONE")


if __name__ == "__main__":
    main()
