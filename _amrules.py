"""AMRules (Adaptive Model Rules) — online, same protocol as ARF, using a disk cache.

The prepared data (warm/test parquets + static-RF predictions + metadata) is built
ONCE and cached to _online_cache/, so each online model reuses it instead of re-running
_prepare_lab + to_dict every time.

Warm 2023+24 -> stream 2025 chronologically (predict -> learn). Numeric features only.
"""

import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from river.rules import AMRules
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "_online_cache"
WARM_PQ = CACHE / "warm.parquet"
TEST_PQ = CACHE / "test.parquet"
META_PKL = CACHE / "meta.pkl"
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def build_cache():
    log("building cache (prepare + sort + static RF) ...")
    t = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns and c != "Track_Status"]

    w_sort = np.argsort(tr_ml["Time_Start"].to_numpy())
    t_sort = np.argsort(te_ml["Time_Start"].to_numpy())
    tr_ml = tr_ml.iloc[w_sort].reset_index(drop=True)
    te_ml = te_ml.iloc[t_sort].reset_index(drop=True)

    medians = {f: float(tr_ml[f].median()) for f in feats}

    # static RF predictions (so downstream models don't re-fit RF)
    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats]),
                                                    tr_ml[TARGET].to_numpy(float))
    p_static = rf.predict(pre.transform(te_ml[feats]))

    country = te.groupby("Session_Key")["Country"].first().to_dict()
    race_order = list(te_ml.groupby("Session_Key")["Time_Start"].min().sort_values().index)

    CACHE.mkdir(parents=True, exist_ok=True)
    tr_ml.to_parquet(WARM_PQ, index=False)
    te_ml.to_parquet(TEST_PQ, index=False)
    with open(META_PKL, "wb") as f:
        pickle.dump({"feats": feats, "medians": medians, "p_static": p_static,
                     "country": country, "race_order": race_order}, f)
    log(f"  cached ({time.time()-t:.0f}s)")


def load_cache():
    tr_ml = pd.read_parquet(WARM_PQ)
    te_ml = pd.read_parquet(TEST_PQ)
    with open(META_PKL, "rb") as f:
        meta = pickle.load(f)
    return tr_ml, te_ml, meta


def dicts_of(df, feats, medians):
    recs = df[feats].to_dict("records")
    y = df[TARGET].to_numpy(float)
    return [({f: (medians[f] if pd.isna(v) else float(v)) for f, v in rec.items()}, yv)
            for rec, yv in zip(recs, y)]


def main():
    t0 = time.time()
    if not META_PKL.exists():
        build_cache()
    tr_ml, te_ml, meta = load_cache()
    feats, medians = meta["feats"], meta["medians"]
    p_static = meta["p_static"]
    country, race_order = meta["country"], meta["race_order"]
    yte = te_ml[TARGET].to_numpy(float)
    te_sk = te_ml["Session_Key"].to_numpy()
    log(f"loaded cache: warm={len(tr_ml):,} test={len(te_ml):,} features={len(feats)}")

    warm = dicts_of(tr_ml, feats, medians)
    test = dicts_of(te_ml, feats, medians)

    log("AMRules ...")
    t = time.time()
    model = AMRules()
    for i, (x, y) in enumerate(warm):
        model.learn_one(x, y)
        if (i + 1) % 50000 == 0:
            log(f"  warm {i+1:,}/{len(warm):,} ({time.time()-t:.0f}s)")
    p = np.empty(len(test))
    for i, (x, y) in enumerate(test):
        pred = model.predict_one(x)
        p[i] = pred if pred is not None else np.nan
        model.learn_one(x, y)
    ok = np.isfinite(p)
    log(f"  AMRules overall={sec_mae(yte[ok], p[ok]):.4f} ({time.time()-t:.0f}s)")

    print(f"\n{'race':18s} {'static':>9s} {'AMRules':>9s}")
    print("-" * 40)
    for sk in race_order:
        m = te_sk == sk
        s = sec_mae(yte[m], p_static[m])
        a = sec_mae(yte[m], p[m][np.isfinite(p[m])]) if np.isfinite(p[m]).any() else np.nan
        print(f"{str(country.get(int(sk), sk)):18s} {s:>9.4f} {a:>9.4f}")
    print("-" * 40)
    print(f"{'OVERALL':18s} {sec_mae(yte, p_static):>9.4f} {sec_mae(yte[ok], p[ok]):>9.4f}")
    print(f"total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
