"""ARF with 200 trees (vs the default 10) — same online protocol.

Warm 2023+24 (learn_one in time order), stream 2025 (predict -> learn). Numeric
features only (ARF's leaf model is a linear regressor, no strings). Static RF ref.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from river.forest.adaptive_random_forest import ARFRegressor
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
N_MODELS = 200


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
    log(f"features={len(feats)}  warm={len(tr_ml):,}  test={len(te_ml):,}  n_models={N_MODELS}")

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

    # static RF reference
    log("static RF ...")
    t = time.time()
    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats]),
                                                    tr_ml[TARGET].to_numpy(float))
    p_static = rf.predict(pre.transform(te_ml[feats]))
    yte = te_ml[TARGET].to_numpy(float)
    log(f"  static RF={sec_mae(yte, p_static):.4f} ({time.time()-t:.0f}s)")

    log("ARF warm-up (2023+24) ...")
    t = time.time()
    arf = ARFRegressor(n_models=N_MODELS, seed=42)
    for i, (x, y) in enumerate(warm):
        arf.learn_one(x, y)
        if (i + 1) % 20000 == 0:
            log(f"  warm {i+1:,}/{len(warm):,} ({time.time()-t:.0f}s)")
    log(f"  warm done ({time.time()-t:.0f}s)")

    log("ARF online 2025 (predict -> learn) ...")
    t = time.time()
    p_arf = np.empty(len(test))
    for i, (x, y) in enumerate(test):
        p = arf.predict_one(x)
        p_arf[i] = p if p is not None else np.nan
        arf.learn_one(x, y)
        if (i + 1) % 20000 == 0:
            log(f"  test {i+1:,}/{len(test):,} ({time.time()-t:.0f}s)")

    ok = np.isfinite(p_arf)
    arf_mae = sec_mae(yte[ok], p_arf[ok])
    print(f"\nOVERALL  static={sec_mae(yte, p_static):.4f}  ARF200={arf_mae:.4f}  (valid {ok.sum():,}/{len(yte):,})")
    race_order = list(te_ml.groupby("Session_Key")["Time_Start"].min().sort_values().index)
    print(f"{'race':18s} {'static':>9s} {'ARF200':>9s}")
    for sk in race_order:
        m = te_sk == sk
        a = sec_mae(yte[m], p_arf[m][np.isfinite(p_arf[m])]) if np.isfinite(p_arf[m]).any() else np.nan
        print(f"{str(country.get(int(sk), sk)):18s} {sec_mae(yte[m], p_static[m]):>9.4f} {a:>9.4f}")
    print(f"total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
