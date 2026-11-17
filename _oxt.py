"""Online Extra Trees (OXT) — faster online tree ensemble, same protocol as ARF.

Warm 2023+24, stream 2025 chronologically (predict -> learn). Sweep n_models = 10, 50.
Numeric features only. Static RF reported as reference. Runs independently.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from river.forest.online_extra_trees import OXTRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


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
    tr_ml = tr_ml.sort_values("Time_Start").reset_index(drop=True)
    te_ml = te_ml.sort_values("Time_Start").reset_index(drop=True)
    country = te.groupby("Session_Key")["Country"].first().to_dict()
    te_sk = te_ml["Session_Key"].to_numpy()

    def dicts(df):
        recs = df[feats].to_dict("records")
        y = df[TARGET].to_numpy(float)
        return [({f: (medians[f] if pd.isna(v) else float(v)) for f, v in rec.items()}, yv)
                for rec, yv in zip(recs, y)]

    warm = dicts(tr_ml)
    test = dicts(te_ml)
    yte = te_ml[TARGET].to_numpy(float)

    # static RF reference
    log("static RF ...")
    t = time.time()
    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats]),
                                                    tr_ml[TARGET].to_numpy(float))
    p_static = rf.predict(pre.transform(te_ml[feats]))
    log(f"  static RF={sec_mae(yte, p_static):.4f} ({time.time()-t:.0f}s)")

    race_order = list(te_ml.groupby("Session_Key")["Time_Start"].min().sort_values().index)
    all_preds = {}

    for n_models in [10, 50]:
        log(f"OXT n_models={n_models} ...")
        t = time.time()
        oxt = OXTRegressor(n_models=n_models, seed=42)
        for i, (x, y) in enumerate(warm):
            oxt.learn_one(x, y)
            if (i + 1) % 50000 == 0:
                log(f"  warm {i+1:,}/{len(warm):,} ({time.time()-t:.0f}s)")
        p = np.empty(len(test))
        for i, (x, y) in enumerate(test):
            pred = oxt.predict_one(x)
            p[i] = pred if pred is not None else np.nan
            oxt.learn_one(x, y)
        all_preds[n_models] = p
        ok = np.isfinite(p)
        log(f"  OXT{n_models}={sec_mae(yte[ok], p[ok]):.4f} ({time.time()-t:.0f}s)")

    print(f"\n{'race':18s} {'static':>9s} {'OXT10':>9s} {'OXT50':>9s}")
    print("-" * 50)
    for sk in race_order:
        m = te_sk == sk
        s = sec_mae(yte[m], p_static[m])
        a = sec_mae(yte[m], all_preds[10][m][np.isfinite(all_preds[10][m])]) if np.isfinite(all_preds[10][m]).any() else np.nan
        b = sec_mae(yte[m], all_preds[50][m][np.isfinite(all_preds[50][m])]) if np.isfinite(all_preds[50][m]).any() else np.nan
        print(f"{str(country.get(int(sk), sk)):18s} {s:>9.4f} {a:>9.4f} {b:>9.4f}")
    ok10 = np.isfinite(all_preds[10]); ok50 = np.isfinite(all_preds[50])
    print("-" * 50)
    print(f"{'OVERALL':18s} {sec_mae(yte, p_static):>9.4f} "
          f"{sec_mae(yte[ok10], all_preds[10][ok10]):>9.4f} {sec_mae(yte[ok50], all_preds[50][ok50]):>9.4f}")
    print(f"total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
