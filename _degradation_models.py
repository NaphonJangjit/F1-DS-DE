"""Train RF(300), ARF(10), Spline SGD on the degradation features (Deg_So_Far etc.).

- RF 300 trees: full feature set + degradation features -> MAE + importance
- Spline SGD: RF-importance-selected top features only -> MAE
- ARF 10 trees: numeric features + degradation features (online) -> MAE
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from river.forest.adaptive_random_forest import ARFRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import SplineTransformer, StandardScaler

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features
from _tyre_compound import median_stint_len_per_compound, add_features

ROOT = Path(__file__).resolve().parent
CORES = max(2, (os.cpu_count() or 8) // 2)
TOP_N = 15


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def main():
    t0 = time.time()
    tr_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                        pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te_raw = pd.read_parquet(ROOT / "data_2025.parquet")

    med_len = median_stint_len_per_compound(tr_raw)
    tr = add_features(tr_raw, med_len)
    te = add_features(te_raw, med_len)
    extra = ["Deg_So_Far", "Deg_Rate", "Compound_Life_Fraction"]
    tr_ml, tr_feats, _, _ = _prepare_lab(tr, extra)
    te_ml, te_feats, _, _ = _prepare_lab(te, extra)
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    log(f"features={len(feats)}  warm={len(tr_ml):,}  test={len(te_ml):,}")

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    Xtr = pre.transform(tr_ml[feats])
    Xte = pre.transform(te_ml[feats])
    ytr = tr_ml[TARGET].to_numpy(float)
    yte = te_ml[TARGET].to_numpy(float)
    names = list(pre.get_feature_names_out())

    # ---- RF 300 ----
    log("RF 300 trees ...")
    t = time.time()
    rf = RandomForestRegressor(n_estimators=300, max_features=0.3, min_samples_leaf=2,
                               n_jobs=CORES, random_state=42).fit(Xtr, ytr)
    p_rf = rf.predict(Xte)
    rf_mae = sec_mae(yte, p_rf)
    log(f"  RF300 MAE={rf_mae:.4f} ({time.time()-t:.0f}s)")

    # top features by RF importance
    imp = rf.feature_importances_
    top_idx = np.argsort(imp)[::-1][:TOP_N]
    top_names = [names[i] for i in top_idx]
    log(f"top-{TOP_N} by RF importance: {top_names}")

    # ---- Spline SGD on top features ----
    log("Spline SGD (top features) ...")
    t = time.time()
    Xtr_t = Xtr[:, top_idx]
    Xte_t = Xte[:, top_idx]
    spline = SplineTransformer(n_knots=3, degree=1, include_bias=False).fit(Xtr_t)
    sgd = SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-3, random_state=42)
    sgd.fit(spline.transform(Xtr_t), ytr)
    p_sgd = sgd.predict(spline.transform(Xte_t))
    sgd_mae = sec_mae(yte, p_sgd)
    log(f"  SplineSGD(top-{TOP_N}) MAE={sgd_mae:.4f} ({time.time()-t:.0f}s)")

    # ---- ARF 10 (numeric + degradation) ----
    log("ARF 10 trees ...")
    t = time.time()
    num_feats = [c for c in feats if c != "Track_Status"]
    medians = {f: float(tr_ml[f].median()) for f in num_feats}
    tr_ml_s = tr_ml.sort_values("Time_Start").reset_index(drop=True)
    te_ml_s = te_ml.sort_values("Time_Start").reset_index(drop=True)

    def dicts(df):
        recs = df[num_feats].to_dict("records")
        y = df[TARGET].to_numpy(float)
        return [({f: (medians[f] if pd.isna(v) else float(v)) for f, v in rec.items()}, yv)
                for rec, yv in zip(recs, y)]

    warm = dicts(tr_ml_s)
    test = dicts(te_ml_s)
    arf = ARFRegressor(n_models=10, seed=42)
    for i, (x, y) in enumerate(warm):
        arf.learn_one(x, y)
        if (i + 1) % 50000 == 0:
            log(f"  warm {i+1:,}/{len(warm):,} ({time.time()-t:.0f}s)")
    p_arf = np.empty(len(test))
    for i, (x, y) in enumerate(test):
        pr = arf.predict_one(x)
        p_arf[i] = pr if pr is not None else np.nan
        arf.learn_one(x, y)
    ok = np.isfinite(p_arf)
    arf_mae = sec_mae(yte[ok], p_arf[ok])
    log(f"  ARF10 MAE={arf_mae:.4f} ({time.time()-t:.0f}s)")

    print("\n" + "=" * 50)
    print(f"{'model':22s} {'MAE':>9s}")
    print("-" * 50)
    print(f"{'RF 300 (deg feat)':22s} {rf_mae:>9.4f}")
    print(f"{'Spline SGD (top-15)':22s} {sgd_mae:>9.4f}")
    print(f"{'ARF 10 (deg feat)':22s} {arf_mae:>9.4f}")
    print("=" * 50)
    print("references: baseline RF150=0.2649, RF150+Deg_So_Far=0.2576, ARF10(no deg)=0.4106, SGD(full)=1.32")
    print(f"total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
