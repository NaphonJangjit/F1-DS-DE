"""Per-row error ranking for the frozen RF (2023+24 -> 2025).

Absolute error in seconds |exp(actual_log) - exp(pred_log)|, every 2025 sector,
sorted highest first. Saves the full ranking to top_errors_2025.csv and prints
the top rows with context (wet / status / tyre / pace) to see why they fail.

SCRATCH.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)

CTX = ["Country", "Driver", "Lap", "Sector", "Rainfall", "Track_Status", "Tyre",
       "Prev_Duration", "Clean_Pace", "Best_Duration_So_Far", "Total_Sector_Length"]


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
    pred_log = rf.predict(pre.transform(te_ml[feats]))
    y_log = te_ml[TARGET].to_numpy(float)

    actual = np.exp(y_log)
    pred = np.exp(pred_log)
    err = np.abs(actual - pred)

    country = te.groupby("Session_Key")["Country"].first().to_dict()
    out = pd.DataFrame({
        "error_s": err,
        "Duration": actual,
        "Predicted": pred,
        "Country": te_ml["Session_Key"].map(lambda sk: country.get(int(sk), "?")).to_numpy(),
        "Session_Key": te_ml["Session_Key"].to_numpy(),
        "Time_Start": te_ml["Time_Start"].to_numpy(),
    })
    for c in ["Driver", "Lap", "Sector", "Rainfall", "Track_Status", "Tyre",
              "Prev_Duration", "Clean_Pace", "Best_Duration_So_Far", "Total_Sector_Length"]:
        if c in te_ml.columns:
            out[c] = te_ml[c].to_numpy()

    out = out.sort_values("error_s", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", out.index + 1)

    out.to_csv(ROOT / "top_errors_2025.csv", index=False)
    print(f"saved {len(out):,} rows -> top_errors_2025.csv  ({time.time()-t0:.0f}s)\n")

    show = out.head(40)
    cols = ["rank", "error_s", "Country", "Driver", "Lap", "Sector", "Duration", "Predicted",
            "Rainfall", "Track_Status", "Tyre", "Prev_Duration", "Clean_Pace"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    print(show[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
