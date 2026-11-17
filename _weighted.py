"""Class weighting (sample weights) instead of SMOTE, on Spline SGD + NB + ARF.

Weight the rare regimes (wet / SC-VSC) inversely by frequency (capped), so the
model pays more attention to the extreme-lap-time tracks. Same online protocol:
warm 2023+24 (learn with weights), stream 2025 chronologically (predict -> learn).
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from river.naive_bayes import GaussianNB
from river.forest.adaptive_random_forest import ARFRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import SplineTransformer

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
CAP = 20.0
TARGET_SHARE = 0.10


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def regime_of(df_ml):
    wet = df_ml["Rainfall"].to_numpy() == 1
    sc = df_ml["Track_Status"].isin(SC_SET).to_numpy()
    return np.where(wet, "wet", np.where(sc, "sc", "clean"))


def weights_of(regime, cap=CAP, target=0.10):
    n = len(regime)
    w = np.ones(n)
    for r in ["wet", "sc"]:
        m = regime == r
        s = float(m.mean())
        if 0 < s < target:
            w[m] = min(target / s, cap)
    return w / w.mean()


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

    w_sort = np.argsort(tr_ml["Time_Start"].to_numpy())
    t_sort = np.argsort(te_ml["Time_Start"].to_numpy())
    tr_ml = tr_ml.iloc[w_sort].reset_index(drop=True)
    te_ml = te_ml.iloc[t_sort].reset_index(drop=True)

    Xw, yw = Xy(tr_ml)
    Xte, yte = Xy(te_ml)
    reg_w = regime_of(tr_ml)
    reg_t = regime_of(te_ml)
    ww = weights_of(reg_w)
    wt = weights_of(reg_t)
    log(f"warm regime shares: wet={float((reg_w=='wet').mean())*100:.1f}%  "
        f"sc={(reg_w=='sc').mean()*100:.1f}%  clean={(reg_w=='clean').mean()*100:.1f}%")
    log(f"weight range warm [{ww.min():.2f}, {ww.max():.2f}]")

    country = te.groupby("Session_Key")["Country"].first().to_dict()
    te_sk = te_ml["Session_Key"].to_numpy()
    race_order = list(te_ml.groupby("Session_Key")["Time_Start"].min().sort_values().index)

    # static RF reference (unweighted)
    log("static RF ...")
    t = time.time()
    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats]), yw)
    p_static = rf.predict(pre.transform(te_ml[feats]))
    log(f"  static RF={sec_mae(yte, p_static):.4f} ({time.time()-t:.0f}s)")

    preds = {}

    # ---- Spline SGD (weighted) ----
    log("Spline SGD (weighted) ...")
    t = time.time()
    spline = SplineTransformer(n_knots=3, degree=1, include_bias=False).fit(Xw)
    Xw_s, Xte_s = spline.transform(Xw), spline.transform(Xte)
    sgd = SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-3,
                       learning_rate="constant", eta0=1e-3, random_state=42)
    for i in range(0, len(Xw_s), 1000):
        sgd.partial_fit(Xw_s[i:i + 1000], yw[i:i + 1000], sample_weight=ww[i:i + 1000])
    preds["SGD"] = sgd.predict(Xte_s)
    log(f"  Spline SGD={sec_mae(yte, preds['SGD']):.4f} ({time.time()-t:.0f}s)")

    # ---- Naive Bayes (weighted, discretized) ----
    log("Naive Bayes (weighted) ...")
    t = time.time()
    bins = pd.qcut(yw, NB_BINS, labels=False, duplicates="drop")
    bin_median = {b: float(np.median(yw[bins == b])) for b in np.unique(bins)}
    nb = GaussianNB()
    rec_w = tr_ml[feats].to_dict("records")
    for i, rec in enumerate(rec_w):
        d = {f: (medians[f] if pd.isna(v) else float(v)) for f, v in rec.items()}
        for _ in range(max(1, int(round(float(ww[i]))))):  # GaussianNB has no w= arg
            nb.learn_one(d, float(bins[i]))
    rec_t = te_ml[feats].to_dict("records")
    p_nb = np.empty(len(yte))
    for i, rec in enumerate(rec_t):
        d = {f: (medians[f] if pd.isna(v) else float(v)) for f, v in rec.items()}
        p_nb[i] = bin_median.get(nb.predict_one(d), np.median(yw))
    preds["NB"] = p_nb
    log(f"  NB={sec_mae(yte, p_nb):.4f} ({time.time()-t:.0f}s)")

    # ---- ARF (weighted) ----
    log("ARF (weighted) ...")
    t = time.time()
    arf = ARFRegressor(seed=42)
    for i, rec in enumerate(rec_w):
        d = {f: (medians[f] if pd.isna(v) else float(v)) for f, v in rec.items()}
        arf.learn_one(d, yw[i], w=float(ww[i]))
        if (i + 1) % 50000 == 0:
            log(f"  warm {i+1:,}/{len(rec_w):,} ({time.time()-t:.0f}s)")
    p_arf = np.empty(len(yte))
    for i, rec in enumerate(rec_t):
        d = {f: (medians[f] if pd.isna(v) else float(v)) for f, v in rec.items()}
        p = arf.predict_one(d)
        p_arf[i] = p if p is not None else np.nan
        arf.learn_one(d, yte[i], w=float(wt[i]))
    preds["ARF"] = p_arf
    log(f"  ARF={sec_mae(yte[pd.notna(p_arf)], p_arf[pd.notna(p_arf)]):.4f} ({time.time()-t:.0f}s)")

    # report
    print("\n=== per-race MAE (chronological) ===")
    print(f"{'race':18s} {'n':>6s} {'static':>9s} {'SGD':>9s} {'NB':>9s} {'ARF':>9s}")
    print("-" * 60)
    for sk in race_order:
        m = te_sk == sk
        s = sec_mae(yte[m], p_static[m])
        sg = sec_mae(yte[m], preds["SGD"][m])
        nb = sec_mae(yte[m], preds["NB"][m])
        ar = sec_mae(yte[m], p_arf[m][pd.notna(p_arf[m])]) if pd.notna(p_arf[m]).any() else np.nan
        print(f"{str(country.get(int(sk), sk)):18s} {int(m.sum()):>6d} {s:>9.4f} {sg:>9.4f} {nb:>9.4f} {ar:>9.4f}")
    ok = pd.notna(p_arf)
    print("-" * 60)
    print(f"{'OVERALL':18s} {'':>6s} {sec_mae(yte, p_static):>9.4f} {sec_mae(yte, preds['SGD']):>9.4f} "
          f"{sec_mae(yte, preds['NB']):>9.4f} {sec_mae(yte[ok], p_arf[ok]):>9.4f}")
    print(f"total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
