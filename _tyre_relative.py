"""Isolate tyre degradation from fuel/track evolution.

Deg_So_Far = Clean_Pace - stint_start Clean_Pace is confounded: fuel burn + track
rubbering-in make EVERY car faster, so it's (fuel gain) - (tyre loss), not tyre alone.

Fix (difference-in-differences): remove the FIELD's pace change over the same window.
  Field_Pace     = median Clean_Pace across drivers at (session, lap, sector)
  Field_Change   = Field_Pace - Field_Pace at stint start   (common fuel/track effect)
  Tyre_Rel_Deg   = Deg_So_Far - Field_Change                (driver-specific = tyre)
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
from _tyre_compound import median_stint_len_per_compound

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


def add_relative_deg(df):
    df = df.sort_values(["Session_Key", "Driver", "Lap"]).reset_index(drop=True)
    tta = pd.to_numeric(df["Tyre_Age_Total"], errors="coerce").to_numpy(float)
    cp = pd.to_numeric(df["Clean_Pace"], errors="coerce").to_numpy(float)

    # stint id
    new_stint = (tta < np.r_[np.inf, tta[:-1]]).astype(int)
    df["_ns"] = new_stint
    df["_stint"] = df.groupby(["Session_Key", "Driver"], sort=False)["_ns"].cumsum()

    # Deg_So_Far (pace change since stint start)
    df["_cp"] = cp
    stint_min = df.groupby(["Session_Key", "Driver", "_stint"], sort=False)["_cp"].transform(
        lambda s: s.cummin()).to_numpy()
    df["Deg_So_Far"] = cp - stint_min

    # Field_Pace = median Clean_Pace across drivers at (session, lap, sector)
    fp = df.groupby(["Session_Key", "Lap", "Sector"])["_cp"].median().reset_index()
    fp.columns = ["Session_Key", "Lap", "Sector", "Field_Pace"]
    df = df.merge(fp, on=["Session_Key", "Lap", "Sector"], how="left")

    # Field_Pace at stint start
    df["_start_lap"] = df.groupby(["Session_Key", "Driver", "_stint"], sort=False)["Lap"].transform("min")
    fp2 = fp.rename(columns={"Lap": "_start_lap", "Field_Pace": "Field_Pace_start"})
    df = df.merge(fp2, on=["Session_Key", "_start_lap", "Sector"], how="left")

    df["Tyre_Rel_Deg"] = df["Deg_So_Far"] - (df["Field_Pace"] - df["Field_Pace_start"])
    return df.drop(columns=["_ns", "_stint", "_cp", "_start_lap"])


def main():
    t0 = time.time()
    tr_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                        pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te_raw = pd.read_parquet(ROOT / "data_2025.parquet")
    tr = add_relative_deg(tr_raw)
    te = add_relative_deg(te_raw)

    # sanity: correlation between Deg_So_Far and Tyre_Rel_Deg
    print(f"corr(Deg_So_Far, Tyre_Rel_Deg) = {tr[['Deg_So_Far','Tyre_Rel_Deg']].corr().iloc[0,1]:.3f}")
    print(f"Deg_So_Far mean={tr['Deg_So_Far'].mean():+.3f}  Tyre_Rel_Deg mean={tr['Tyre_Rel_Deg'].mean():+.3f}")

    extra = ["Deg_So_Far", "Tyre_Rel_Deg"]
    tr_ml, tr_feats, _, _ = _prepare_lab(tr, extra)
    te_ml, te_feats, _, _ = _prepare_lab(te, extra)
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats]),
                                                    tr_ml[TARGET].to_numpy(float))
    p = rf.predict(pre.transform(te_ml[feats]))
    log(f"overall MAE (Deg_So_Far + Tyre_Rel_Deg) = {sec_mae(te_ml[TARGET].to_numpy(float), p):.4f} ({time.time()-t0:.0f}s)")

    names = list(pre.get_feature_names_out())
    imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
    print("\n=== degradation features ===")
    for n, v in imp.items():
        if "Deg_So_Far" in n or "Tyre_Rel_Deg" in n or "Field_Pace" in n:
            print(f"  rank {imp.index.get_loc(n)+1:3d}  {n:40s} {v:.5f}")
    print("\n=== top 10 ===")
    for i, (n, v) in enumerate(imp.head(10).items(), 1):
        print(f"  {i:3d} {n:40s} {v:.5f}")


if __name__ == "__main__":
    main()
