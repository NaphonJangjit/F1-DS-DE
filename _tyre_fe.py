"""Make tyre age matter: encode degradation trajectory, not raw laps.

Hypothesis: raw Tyre_Age is redundant with the pace anchors (which already reflect
"the car is slow now"). What the anchors miss is the CUMULATIVE degradation since the
stint started. New features (point-in-time):
  Deg_So_Far   = Clean_Pace - (min Clean_Pace in this stint so far)  [lost pace]
  Stint_Progress = Tyre_Age / (track median stint length)            [where on the curve]
  Tyre_Age     = raw laps on tyre (reinstate, unscaled)
  Tyre_Age_2   = Tyre_Age^2 (non-linear late-stint cliff)

Train RF and report each feature's importance + overall MAE change.
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


def add_tyre_features(raw):
    df = raw.copy()
    df = df.sort_values(["Session_Key", "Driver", "Lap"]).reset_index(drop=True)
    tta = pd.to_numeric(df["Tyre_Age_Total"], errors="coerce").to_numpy(float)
    cp = pd.to_numeric(df["Clean_Pace"], errors="coerce").to_numpy(float)
    age = pd.to_numeric(df["Tyre_Age"], errors="coerce").to_numpy(float)

    # stint id: Tyre_Age_Total reset -> new stint
    new_stint = (tta < np.r_[np.inf, tta[:-1]]).astype(int)
    df["_ns"] = new_stint
    df["_stint"] = df.groupby(["Session_Key", "Driver"], sort=False)["_ns"].cumsum()

    # Deg_So_Far = Clean_Pace - min(Clean_Pace in stint so far)
    df["_cp"] = cp
    stint_min = df.groupby(["Session_Key", "Driver", "_stint"], sort=False)["_cp"].transform(
        lambda s: s.cummin()).to_numpy()
    df["Deg_So_Far"] = cp - stint_min

    # Stint_Progress = Tyre_Age / track median stint length
    med_len = df.groupby("Session_Key")["Tyre_Age_Total"].transform("max")
    df["Stint_Progress"] = tta / med_len.replace(0, np.nan)

    # raw Tyre_Age (unscaled) + squared
    df["Tyre_Age_raw"] = age
    df["Tyre_Age_raw_2"] = age ** 2
    return df.drop(columns=["_ns", "_stint", "_cp"])


def main():
    t0 = time.time()
    tr = add_tyre_features(pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                                      pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True))
    te = add_tyre_features(pd.read_parquet(ROOT / "data_2025.parquet"))

    extra = ["Deg_So_Far", "Stint_Progress", "Tyre_Age_raw", "Tyre_Age_raw_2"]
    tr_ml, tr_feats, _, _ = _prepare_lab(tr, extra)
    te_ml, te_feats, _, _ = _prepare_lab(te, extra)
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    log(f"features={len(feats)}")

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    Xtr = pre.transform(tr_ml[feats])
    Xte = pre.transform(te_ml[feats])
    ytr = tr_ml[TARGET].to_numpy(float)
    yte = te_ml[TARGET].to_numpy(float)

    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(Xtr, ytr)
    p = rf.predict(Xte)
    log(f"overall MAE (with tyre features) = {sec_mae(yte, p):.4f} ({time.time()-t0:.0f}s)")

    names = list(pre.get_feature_names_out())
    imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
    print("\n=== tyre-age-related feature importance (rank / value) ===")
    for n, v in imp.items():
        if any(k in n for k in ["Tyre_Age", "Deg_So_Far", "Stint_Progress"]):
            print(f"  rank {imp.index.get_loc(n)+1:3d}  {n:40s} {v:.5f}")

    print("\n=== top 15 overall (for context) ===")
    for i, (n, v) in enumerate(imp.head(15).items(), 1):
        print(f"  {i:3d} {n:40s} {v:.5f}")


if __name__ == "__main__":
    main()
