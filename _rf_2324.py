"""Quick check: frozen RF trained on 2023 only, tested on 2024.

Same pipeline/config as the main baseline (2023+24 -> 2025), just a different split.
SCRATCH.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}


def sec(y_log, p_log):
    a = np.exp(np.asarray(y_log, float)); p = np.exp(np.asarray(p_log, float))
    return float(mean_absolute_error(a, p)), float(np.sqrt(mean_squared_error(a, p)))


def green_mask(df):
    d = df[["Session_Key", "Driver", "Lap", "Track_Status"]].copy()
    is_sc = d["Track_Status"].isin(SC_SET).to_numpy()
    laps = (d.assign(_is_sc=is_sc)
            .groupby(["Session_Key", "Driver", "Lap"], sort=False)["_is_sc"].max()
            .reset_index().sort_values(["Session_Key", "Driver", "Lap"]))
    laps["_prev_sc"] = laps.groupby(["Session_Key", "Driver"])["_is_sc"].shift(1).fillna(False)
    merged = d[["Session_Key", "Driver", "Lap"]].merge(
        laps[["Session_Key", "Driver", "Lap", "_prev_sc"]],
        on=["Session_Key", "Driver", "Lap"], how="left")
    return ~(is_sc | merged["_prev_sc"].to_numpy().astype(bool))


def main():
    t0 = time.time()
    tr = pd.read_parquet(ROOT / "data_2023.parquet")
    te = pd.read_parquet(ROOT / "data_2024.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    Xtr = pre.transform(tr_ml[feats])
    Xte = pre.transform(te_ml[feats])
    ytr = tr_ml[TARGET].to_numpy(float)
    yte = te_ml[TARGET].to_numpy(float)

    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(Xtr, ytr)
    p = rf.predict(Xte)

    full_mae, full_rmse = sec(yte, p)
    g = green_mask(te_ml)
    green_mae, green_rmse = sec(yte[g], p[g])

    print(f"train(2023) rows={len(tr_ml):,}  test(2024) rows={len(te_ml):,}  features={len(feats)}")
    print(f"FULL   : MAE={full_mae:.4f}s  RMSE={full_rmse:.4f}s")
    print(f"GREEN  : MAE={green_mae:.4f}s  RMSE={green_rmse:.4f}s")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
