"""Adaptive Random Forest (river ARFRegressor) — online, race-by-race, chronological.

Warm = 2023+2024 (learn_one in time order). Then stream 2025 in time order:
predict_one -> record -> learn_one. Compare to the static batch RF (warm-only).

ARF adapts via drift detectors on its Hoeffding trees, so no batch retraining.
Target = Log_Duration; error back to seconds.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from river.forest.adaptive_random_forest import ARFRegressor
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
    a = np.exp(np.asarray(y_log, float)); p = np.exp(np.asarray(p_log, float))
    return float(mean_absolute_error(a, p))


def to_dicts(df_ml, feats, medians):
    """-> list of (feature_dict, y_log) for river (numeric features only)."""
    records = df_ml[feats].to_dict("records")
    y = df_ml[TARGET].to_numpy(float)
    out = []
    for rec, yv in zip(records, y):
        d = {f: (medians[f] if pd.isna(v) else float(v)) for f, v in rec.items()}
        out.append((d, yv))
    return out


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns and c != "Track_Status"]
    log(f"features={len(feats)}  warm={len(tr_ml):,}  test={len(te_ml):,}")

    # median imputation (fit on warm)
    medians = {f: float(tr_ml[f].median()) for f in feats}

    # sort chronologically
    tr_ml = tr_ml.sort_values("Time_Start").reset_index(drop=True)
    te_ml = te_ml.sort_values("Time_Start").reset_index(drop=True)
    country = te.groupby("Session_Key")["Country"].first().to_dict()

    log("building dicts ...")
    warm = to_dicts(tr_ml, feats, medians)
    test = to_dicts(te_ml, feats, medians)

    # ---- static batch RF reference ----
    log("static RF reference ...")
    t = time.time()
    from sklearn.ensemble import RandomForestRegressor
    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats]),
                                                    tr_ml[TARGET].to_numpy(float))
    p_static = rf.predict(pre.transform(te_ml[feats]))
    log(f"  static RF overall={sec_mae(te_ml[TARGET].to_numpy(float), p_static):.4f} ({time.time()-t:.0f}s)")

    # ---- ARF ----
    log("ARF warm-up (2023+24) ...")
    t = time.time()
    model = ARFRegressor(seed=42)
    for i, (x, y) in enumerate(warm):
        model.learn_one(x, y)
        if (i + 1) % 50000 == 0:
            log(f"  warm {i+1:,}/{len(warm):,} ({time.time()-t:.0f}s)")
    log(f"  warm done ({time.time()-t:.0f}s)")

    log("ARF online 2025 (predict -> learn) ...")
    t = time.time()
    preds = []
    sk_arr = te_ml["Session_Key"].to_numpy()
    race_preds = {}  # sk -> list of (y, p)
    for i, (x, y) in enumerate(test):
        p = model.predict_one(x)
        if p is None:
            p = np.nan
        preds.append(p)
        sk = int(sk_arr[i])
        race_preds.setdefault(sk, []).append((y, p))
        model.learn_one(x, y)
        if (i + 1) % 50000 == 0:
            log(f"  test {i+1:,}/{len(test):,} ({time.time()-t:.0f}s)")

    yte = te_ml[TARGET].to_numpy(float)
    preds = np.asarray(preds, float)
    ok = np.isfinite(preds)
    arf_mae = sec_mae(yte[ok], preds[ok])
    log(f"ARF overall={arf_mae:.4f} (valid preds {ok.sum():,}/{len(yte):,})")

    # per-race
    race_order = list(te_ml.groupby("Session_Key")["Time_Start"].min().sort_values().index)
    print("\n=== per-race MAE (chronological) ===")
    print(f"{'race':18s} {'n':>6s} {'static':>9s} {'ARF':>9s}")
    print("-" * 48)
    for sk in race_order:
        rows = race_preds.get(int(sk), [])
        y = np.array([r[0] for r in rows]); p = np.array([r[1] for r in rows])
        m = np.isfinite(p)
        s = sec_mae(y, p_static[te_ml["Session_Key"].to_numpy() == sk]) if len(y) else np.nan
        a = sec_mae(y[m], p[m]) if m.any() else np.nan
        print(f"{str(country.get(int(sk), sk)):18s} {len(y):>6d} {s:>9.4f} {a:>9.4f}")
    print(f"\nOVERALL  static={sec_mae(yte, p_static):.4f}  ARF={arf_mae:.4f}")
    print(f"total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
