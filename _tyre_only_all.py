"""Remove ALL pace/behaviour features that mask the tyre, keep tyre + static + env + pit.

Remove: Best_Duration_So_Far, Clean_Pace, all Prev_* pace/telemetry, and all _roll5_ trends.
Keep: Tyre_Age_<compound>, Tyre_Age_Total, geometry, distance, weather, traffic, interval,
pit flags, Lap, Track_Status. Then retrain RF and see if tyre age surfaces.
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
KEEP_PIT = {"Prev_Lap_Pitted", "Prev_Lap_Box_Duration"}


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    feats_kept = [c for c in feats
                  if c not in {"Best_Duration_So_Far", "Clean_Pace"}
                  and "_roll5_" not in c
                  and not (c.startswith("Prev_") and c not in KEEP_PIT)]
    print(f"removed {len(feats) - len(feats_kept)} pace/behaviour features; kept {len(feats_kept)}:")
    print("  " + ", ".join(feats_kept))

    pre = build_preprocessor(feats_kept).fit(tr_ml[feats_kept])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats_kept]),
                                                    tr_ml[TARGET].to_numpy(float))
    p = rf.predict(pre.transform(te_ml[feats_kept]))
    yte = te_ml[TARGET].to_numpy(float)
    mae = float(mean_absolute_error(np.exp(yte), np.exp(p)))
    print(f"overall MAE = {mae:.4f} ({time.time()-t0:.0f}s)")

    names = list(pre.get_feature_names_out())
    imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
    print("\n=== feature importance (all kept) ===")
    for i, (n, v) in enumerate(imp.items(), 1):
        print(f"  {i:3d} {n:40s} {v:.5f}")


if __name__ == "__main__":
    main()
