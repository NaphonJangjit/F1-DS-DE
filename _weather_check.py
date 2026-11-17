"""Check whether weather features are still important, now with degradation features added."""

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
from _tyre_compound import median_stint_len_per_compound, add_features

ROOT = Path(__file__).resolve().parent
CORES = max(2, (os.cpu_count() or 8) // 2)
WEATHER = ["Track_Temperature", "Rainfall", "Pressure", "Humidity", "Air_Temperature"]


def main():
    t0 = time.time()
    tr_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                        pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te_raw = pd.read_parquet(ROOT / "data_2025.parquet")
    med = median_stint_len_per_compound(tr_raw)
    tr = add_features(tr_raw, med)
    te = add_features(te_raw, med)
    tr_ml, tr_feats, _, _ = _prepare_lab(tr, ["Deg_So_Far", "Deg_Rate", "Compound_Life_Fraction"])
    te_ml, te_feats, _, _ = _prepare_lab(te, ["Deg_So_Far", "Deg_Rate", "Compound_Life_Fraction"])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=150, max_features=0.3, min_samples_leaf=2,
                               n_jobs=CORES, random_state=42).fit(pre.transform(tr_ml[feats]),
                                                                  tr_ml[TARGET].to_numpy(float))
    names = list(pre.get_feature_names_out())
    imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)

    print("weather feature importance (with degradation features):")
    for n, v in imp.items():
        base = n.replace("num__", "").replace("cat__", "")
        if base in WEATHER or any(w in n for w in ["Rain", "Humid", "Track_Temp", "Air_Temp", "Pressure"]):
            print(f"  rank {imp.index.get_loc(n)+1:3d}  {n:40s} {v:.5f}")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
