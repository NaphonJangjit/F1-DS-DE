"""MAE / RMSE / R^2 per tyre condition for the OLD frozen RF (no FE), 2023+24 -> 2025.

DRY = SOFT/MEDIUM/HARD, INTER = INTERMEDIATE, WET = WET (likely empty in 2025).
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)

CONDS = {
    "DRY": ["SOFT", "MEDIUM", "HARD"],
    "INTER": ["INTERMEDIATE"],
    "WET": ["WET"],
}


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats]),
                                                    tr_ml[TARGET].to_numpy(float))
    p_log = rf.predict(pre.transform(te_ml[feats]))
    y_log = te_ml[TARGET].to_numpy(float)

    y_s = np.exp(y_log)
    p_s = np.exp(p_log)
    tyre = te_ml["Tyre"].to_numpy()

    print(f"\n{'cond':6s} {'n':>8s} {'MAE_s':>8s} {'RMSE_s':>8s} {'R2_log':>8s} {'R2_s':>8s}")
    print("-" * 52)
    rows = []
    for name, compounds in CONDS.items():
        m = np.isin(tyre, compounds)
        if m.sum() == 0:
            print(f"{name:6s} {0:>8d} {'--':>8s} {'--':>8s} {'--':>8s} {'--':>8s}")
            rows.append({"cond": name, "n": 0})
            continue
        mae = mean_absolute_error(y_s[m], p_s[m])
        rmse = np.sqrt(mean_squared_error(y_s[m], p_s[m]))
        r2l = r2_score(y_log[m], p_log[m])
        r2s = r2_score(y_s[m], p_s[m])
        print(f"{name:6s} {int(m.sum()):>8d} {mae:>8.4f} {rmse:>8.4f} {r2l:>8.4f} {r2s:>8.4f}")
        rows.append({"cond": name, "n": int(m.sum()), "mae": mae, "rmse": rmse,
                     "r2_log": r2l, "r2_s": r2s})

    # overall for reference
    print("-" * 52)
    print(f"{'ALL':6s} {len(y_s):>8d} {mean_absolute_error(y_s, p_s):>8.4f} "
          f"{np.sqrt(mean_squared_error(y_s, p_s)):>8.4f} {r2_score(y_log, p_log):>8.4f} "
          f"{r2_score(y_s, p_s):>8.4f}")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
