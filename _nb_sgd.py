"""Naive Bayes + Spline SGDRegressor, online (same protocol as ARF), with SMOTE.

- Naive Bayes : river GaussianNB on a DISCRETIZED Log_Duration (10 quantile bins).
                Predicted bin -> its median duration. SMOTE oversamples rare bins.
- Spline SGD  : sklearn SplineTransformer (piecewise-linear) + SGDRegressor
                (partial_fit = online). SMOTE oversamples rare wet/SC/VSC regimes.

Warm = 2023+24 (learn_one/partial_fit in time order), stream 2025 chronologically.
Static RF reported as reference. Runs independently of _arf.py.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from river.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import SplineTransformer, StandardScaler

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}
NB_BINS = 10
SMOTE_SHARE = 0.10


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def smote_regime(X, y, regime, target_share=SMOTE_SHARE, seed=42):
    """SMOTE-for-regression: oversample wet/SC/VSC regimes by k-NN interpolation."""
    rng = np.random.RandomState(seed)
    rare_mask = (regime != "clean")
    rare_n = int(rare_mask.sum())
    n = len(y)
    if rare_n == 0 or rare_n / n >= target_share:
        return X, y
    n_synth = int((target_share * n - rare_n) / (1 - target_share))
    idx = np.where(rare_mask)[0]
    Xr = X[idx].astype(float)
    sc = StandardScaler().fit(Xr)
    Xs = sc.transform(Xr)
    nn = NearestNeighbors(n_neighbors=min(6, len(idx))).fit(Xs)
    neigh = nn.kneighbors(Xs, return_distance=False)
    SX, Sy = [], []
    for _ in range(n_synth):
        i = rng.randint(len(idx))
        j = rng.choice(neigh[i])
        a = rng.uniform(0.0, 1.0)
        SX.append(X[idx[i]] + a * (X[idx[j]] - X[idx[i]]))
        Sy.append(y[idx[i]] + a * (y[idx[j]] - y[idx[i]]))
    return np.vstack([X, np.asarray(SX)]), np.concatenate([y, np.asarray(Sy)])


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns and c != "Track_Status"]
    log(f"features={len(feats)}  warm={len(tr_ml):,}  test={len(te_ml):,}")

    medians = {f: float(tr_ml[f].median()) for f in feats}

    def Xy(df):
        X = df[feats].to_numpy(float)
        X = np.where(np.isfinite(X), X, np.array([medians[f] for f in feats]))
        return X, df[TARGET].to_numpy(float)

    Xw, yw = Xy(tr_ml)
    Xte, yte = Xy(te_ml)

    # chronological order
    w_sort = np.argsort(tr_ml["Time_Start"].to_numpy())
    t_sort = np.argsort(te_ml["Time_Start"].to_numpy())
    tr_ml = tr_ml.iloc[w_sort].reset_index(drop=True)
    te_ml = te_ml.iloc[t_sort].reset_index(drop=True)
    Xw = Xw[w_sort]; yw = yw[w_sort]
    Xte = Xte[t_sort]; yte = yte[t_sort]
    country = te.groupby("Session_Key")["Country"].first().to_dict()
    te_sk = te_ml["Session_Key"].to_numpy()

    # static RF reference
    log("static RF ...")
    t = time.time()
    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats]), yw)
    p_static = rf.predict(pre.transform(te_ml[feats]))
    log(f"  static RF overall={sec_mae(yte, p_static):.4f} ({time.time()-t:.0f}s)")

    # ---- Naive Bayes (discretized) ----
    log("Naive Bayes (discretized) ...")
    t = time.time()
    bins = pd.qcut(yw, NB_BINS, labels=False, duplicates="drop")
    bin_median = {b: float(np.median(yw[bins == b])) for b in np.unique(bins)}
    # SMOTE: oversample rare bins by duplication
    _, counts = np.unique(bins, return_counts=True)
    rare = np.where(counts < np.median(counts) * 0.5)[0]
    Xw_nb, yw_nb = Xw, bins.astype(float)
    if len(rare):
        # duplicate rare bins to balance
        extra_x, extra_y = [], []
        for b in rare:
            m = bins == b
            k = int(np.median(counts) - counts[b])
            for _ in range(min(k, m.sum())):
                extra_x.append(Xw[m]); extra_y.append(np.full(m.sum(), b))
        if extra_x:
            Xw_nb = np.vstack([Xw] + extra_x)
            yw_nb = np.concatenate([bins.astype(float)] + [e.ravel() for e in extra_y])
    nb = GaussianNB()
    for i in range(len(Xw_nb)):
        nb.learn_one(dict(zip(feats, Xw_nb[i])), yw_nb[i])
    p_nb = np.empty(len(yte))
    for i in range(len(yte)):
        pred = nb.predict_one(dict(zip(feats, Xte[i])))
        p_nb[i] = bin_median.get(pred, np.median(yw))
    nb_mae = sec_mae(yte, p_nb)
    log(f"  NB overall={nb_mae:.4f} ({time.time()-t:.0f}s)")

    # ---- Spline SGDRegressor ----
    log("Spline SGDRegressor ...")
    t = time.time()
    spline = SplineTransformer(n_knots=3, degree=1, include_bias=False).fit(Xw)
    Xw_s = spline.transform(Xw)
    Xte_s = spline.transform(Xte)
    regime = np.where(tr_ml["Rainfall"].to_numpy() == 1, "wet",
                      np.where(tr_ml["Track_Status"].isin(SC_SET).to_numpy(), "sc", "clean"))
    Xw_s, yw = smote_regime(Xw_s, yw, regime)
    sgd = SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-3,
                       learning_rate="constant", eta0=1e-3, random_state=42)
    # warm (online partial_fit, chunked)
    for i in range(0, len(Xw_s), 500):
        sgd.partial_fit(Xw_s[i:i + 500], yw[i:i + 500])
    p_sgd = sgd.predict(Xte_s)
    sgd_mae = sec_mae(yte, p_sgd)
    log(f"  Spline SGD overall={sgd_mae:.4f} ({time.time()-t:.0f}s)")

    # per-race
    race_order = list(te_ml.groupby("Session_Key")["Time_Start"].min().sort_values().index)
    print("\n=== per-race MAE (chronological) ===")
    print(f"{'race':18s} {'n':>6s} {'static':>9s} {'NB':>9s} {'SGD':>9s}")
    print("-" * 58)
    for sk in race_order:
        m = te_sk == sk
        s = sec_mae(yte[m], p_static[m])
        nb = sec_mae(yte[m], p_nb[m])
        sg = sec_mae(yte[m], p_sgd[m])
        print(f"{str(country.get(int(sk), sk)):18s} {int(m.sum()):>6d} {s:>9.4f} {nb:>9.4f} {sg:>9.4f}")
    print("-" * 58)
    print(f"{'OVERALL':18s} {'':>6s} {sec_mae(yte, p_static):>9.4f} {nb_mae:>9.4f} {sgd_mae:>9.4f}")
    print(f"total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
