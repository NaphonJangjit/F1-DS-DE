"""Compound-aware tyre degradation.

Each compound has its own life / degradation curve (SOFT ~15 laps, HARD ~30+). So on
top of Deg_So_Far, add:
  Compound_Life_Fraction = Tyre_Age / (median stint length of THAT compound)  [used-up %]
  Deg_Rate              = Deg_So_Far / Tyre_Age                                [pace lost per lap]
Median stint length per compound is computed on TRAIN only (point-in-time safe).
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


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def stint_info(df):
    df = df.sort_values(["Session_Key", "Driver", "Lap"]).reset_index(drop=True)
    tta = pd.to_numeric(df["Tyre_Age_Total"], errors="coerce").to_numpy(float)
    new_stint = (tta < np.r_[np.inf, tta[:-1]]).astype(int)
    df["_ns"] = new_stint
    df["_stint"] = df.groupby(["Session_Key", "Driver"], sort=False)["_ns"].cumsum()
    return df


def median_stint_len_per_compound(df):
    d = stint_info(df)
    g = d.groupby(["Session_Key", "Driver", "_stint"], sort=False)
    length = g["Tyre_Age_Total"].max()
    comp = g["Tyre"].first()
    return (length.groupby(comp).median()).to_dict()


def add_features(df, med_len):
    df = stint_info(df)
    tta = pd.to_numeric(df["Tyre_Age_Total"], errors="coerce").to_numpy(float)
    age = pd.to_numeric(df["Tyre_Age"], errors="coerce").to_numpy(float)
    cp = pd.to_numeric(df["Clean_Pace"], errors="coerce").to_numpy(float)

    # stint start clean pace (min clean pace in stint so far) -> Deg_So_Far
    df["_cp"] = cp
    stint_min = df.groupby(["Session_Key", "Driver", "_stint"], sort=False)["_cp"].transform(
        lambda s: s.cummin()).to_numpy()
    df["Deg_So_Far"] = cp - stint_min
    df["Deg_Rate"] = df["Deg_So_Far"] / age

    # compound of the stint -> life fraction
    stint_comp = df.groupby(["Session_Key", "Driver", "_stint"], sort=False)["Tyre"].transform("first")
    df["Compound_Life_Fraction"] = age / stint_comp.map(med_len).to_numpy()

    return df.drop(columns=["_ns", "_stint", "_cp"])


def main():
    t0 = time.time()
    tr_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                        pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te_raw = pd.read_parquet(ROOT / "data_2025.parquet")

    med_len = median_stint_len_per_compound(tr_raw)
    print("median stint length (laps) per compound:")
    for c, v in sorted(med_len.items()):
        print(f"  {c:12s} {v:.1f}")

    tr = add_features(tr_raw, med_len)
    te = add_features(te_raw, med_len)
    extra = ["Deg_So_Far", "Deg_Rate", "Compound_Life_Fraction"]
    tr_ml, tr_feats, _, _ = _prepare_lab(tr, extra)
    te_ml, te_feats, _, _ = _prepare_lab(te, extra)
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats]),
                                                    tr_ml[TARGET].to_numpy(float))
    p = rf.predict(pre.transform(te_ml[feats]))
    log(f"overall MAE = {sec_mae(te_ml[TARGET].to_numpy(float), p):.4f} ({time.time()-t0:.0f}s)")

    names = list(pre.get_feature_names_out())
    imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
    print("\n=== degradation features (rank / importance) ===")
    for n, v in imp.items():
        if any(k in n for k in ["Deg_So_Far", "Deg_Rate", "Compound_Life", "Tyre_Age"]):
            print(f"  rank {imp.index.get_loc(n)+1:3d}  {n:40s} {v:.5f}")
    print("\n=== top 12 overall ===")
    for i, (n, v) in enumerate(imp.head(12).items(), 1):
        print(f"  {i:3d} {n:40s} {v:.5f}")


if __name__ == "__main__":
    main()
