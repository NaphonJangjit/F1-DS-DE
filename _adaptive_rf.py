"""Adaptive RF — iterate over Grands Prix in calendar order (point-in-time correct).

Warm = 2023+2024. Then process 2025 races one at a time, chronologically. Within each
race, process laps in 15-lap chunks (test-then-train): predict the chunk with the
current model, then retrain on warm + every race/chunk seen so far.

Compare static (warm only) vs adaptive.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
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
CHUNK = 15


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def make_rf():
    return RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                 min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                 random_state=42)


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    Xw = pre.transform(tr_ml[feats])
    yw = tr_ml[TARGET].to_numpy(float)
    Xte = pre.transform(te_ml[feats])
    yte = te_ml[TARGET].to_numpy(float)

    # chronological race order
    race_order = list(te_ml.groupby("Session_Key")["Time_Start"].min().sort_values().index)
    country = te.groupby("Session_Key")["Country"].first().to_dict()
    log(f"warm={len(yw):,}  test={len(yte):,}  races={len(race_order)}")

    # static
    log("static (warm only) ...")
    t = time.time()
    static = make_rf().fit(Xw, yw)
    p_static = static.predict(Xte)
    log(f"  static overall={sec_mae(yte, p_static):.4f} ({time.time()-t:.0f}s)")

    # adaptive: iterate races chronologically, 15-lap chunks within each race
    log("adaptive (race-by-race, 15-lap chunks) ...")
    Xpool, ypool = Xw, yw
    model = make_rf().fit(Xpool, ypool)
    p_adapt = np.zeros(len(yte))
    sk_arr = te_ml["Session_Key"].to_numpy()
    lap_arr = te_ml["Lap"].to_numpy()
    race_rows = []
    n_chunks = 0

    for sk in race_order:
        idx = np.where(sk_arr == sk)[0]
        # sort within race by lap
        idx = idx[np.argsort(lap_arr[idx])]
        r_static, r_adapt = [], []
        for start in range(0, len(idx), CHUNK):
            c = idx[start:start + CHUNK]
            p = model.predict(Xte[c])
            p_adapt[c] = p
            r_static.append(sec_mae(yte[c], p_static[c]))
            r_adapt.append(sec_mae(yte[c], p))
            Xpool = np.vstack([Xpool, Xte[c]])
            ypool = np.concatenate([ypool, yte[c]])
            model = make_rf().fit(Xpool, ypool)
            n_chunks += 1
        race_rows.append((str(country.get(int(sk), sk)), float(np.mean(r_static)),
                          float(np.mean(r_adapt)), int(len(idx))))
        if len(race_rows) % 6 == 0:
            log(f"  processed {len(race_rows)}/{len(race_order)} races, pool={len(ypool):,}")

    print("\n=== per-race MAE (chronological order) ===")
    print(f"{'race':18s} {'n':>6s} {'static':>9s} {'adaptive':>9s}")
    print("-" * 48)
    for name, s, a, n in race_rows:
        print(f"{name:18s} {n:>6d} {s:>9.4f} {a:>9.4f}")
    print("-" * 48)
    print(f"{'OVERALL':18s} {'':>6s} {sec_mae(yte, p_static):>9.4f} {sec_mae(yte, p_adapt):>9.4f}")
    print(f"chunks={n_chunks}  total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
