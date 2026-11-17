"""Degradation from TYRE + AGE only — no pace, no length scaling.

Add raw Tyre_Age (laps) and the compound as one-hot, so the RF can learn the
compound-specific degradation curve directly from the tyre.
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


def main():
    t0 = time.time()
    tr_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                        pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te_raw = pd.read_parquet(ROOT / "data_2025.parquet")

    # raw tyre + age (+ compound one-hot)
    for df in (tr_raw, te_raw):
        df["Tyre_Age_raw"] = pd.to_numeric(df["Tyre_Age"], errors="coerce")
        df["Tyre_Age_raw_2"] = df["Tyre_Age_raw"] ** 2
        for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
            df[f"Is_{c}"] = (df["Tyre"].astype(str).str.upper() == c).astype(float)

    extra = ["Tyre_Age_raw", "Tyre_Age_raw_2", "Is_SOFT", "Is_MEDIUM", "Is_HARD",
             "Is_INTERMEDIATE", "Is_WET"]
    tr_ml, tr_feats, _, _ = _prepare_lab(tr_raw, extra)
    te_ml, te_feats, _, _ = _prepare_lab(te_raw, extra)
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats]),
                                                    tr_ml[TARGET].to_numpy(float))
    p = rf.predict(pre.transform(te_ml[feats]))
    mae = float(mean_absolute_error(np.exp(te_ml[TARGET].to_numpy(float)), np.exp(p)))
    print(f"overall MAE = {mae:.4f} ({time.time()-t0:.0f}s)")

    names = list(pre.get_feature_names_out())
    imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
    print("\n=== raw tyre features (rank / importance) ===")
    for n, v in imp.items():
        if "Tyre_Age_raw" in n or n.startswith("num__Is_") or "Tyre_Age_" in n:
            print(f"  rank {imp.index.get_loc(n)+1:3d}  {n:36s} {v:.5f}")
    print("\n=== top 10 ===")
    for i, (n, v) in enumerate(imp.head(10).items(), 1):
        print(f"  {i:3d} {n:40s} {v:.5f}")


if __name__ == "__main__":
    main()
