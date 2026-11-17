"""Compute tyre degradation DIRECTLY from the tyre (compound x age), not from pace.

Each compound has its own degradation rate r(compound) in s/lap. Estimated from
training stints: r = median over stints of (max Clean_Pace - min Clean_Pace) / stint_len.

  Tyre_Deg_Expected = r(compound) * Tyre_Age      (predicted pace loss from tyre alone)
  Tyre_Deg_Fraction = Tyre_Age / expected_life(compound)   (how used-up the tyre is)

Point-in-time safe: rates are estimated on TRAIN only.
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


def estimate_rates(df):
    """per-compound degradation rate (s/lap) + expected life (laps), from train."""
    df = df.sort_values(["Session_Key", "Driver", "Lap"]).reset_index(drop=True)
    tta = pd.to_numeric(df["Tyre_Age_Total"], errors="coerce").to_numpy(float)
    cp = pd.to_numeric(df["Clean_Pace"], errors="coerce").to_numpy(float)
    new_stint = (tta < np.r_[np.inf, tta[:-1]]).astype(int)
    df["_ns"] = new_stint
    df["_stint"] = df.groupby(["Session_Key", "Driver"], sort=False)["_ns"].cumsum()
    df["_cp"] = cp
    g = df.groupby(["Session_Key", "Driver", "_stint"], sort=False)
    stint_deg = g["_cp"].max() - g["_cp"].min()
    stint_len = g["Tyre_Age_Total"].max()
    stint_comp = g["Tyre"].first()
    deg_per_lap = (stint_deg / stint_len.replace(0, np.nan)).reset_index(drop=True)
    tbl = pd.DataFrame({"deg_per_lap": deg_per_lap, "len": stint_len.reset_index(drop=True),
                        "comp": stint_comp.reset_index(drop=True)})
    rates = tbl.groupby("comp")["deg_per_lap"].median().to_dict()
    life = tbl.groupby("comp")["len"].median().to_dict()
    return rates, life


def add_direct(df, rates, life):
    df = df.copy()
    age = pd.to_numeric(df["Tyre_Age"], errors="coerce").to_numpy(float)
    comp = df["Tyre"].astype(str).str.upper().to_numpy()
    df["Tyre_Deg_Expected"] = age * np.array([rates.get(c, np.nan) for c in comp])
    df["Tyre_Deg_Fraction"] = age / np.array([life.get(c, np.nan) for c in comp])
    return df


def main():
    t0 = time.time()
    tr_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                        pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te_raw = pd.read_parquet(ROOT / "data_2025.parquet")
    rates, life = estimate_rates(tr_raw)
    print("per-compound degradation rate (s/lap) and expected life (laps):")
    for c in sorted(rates):
        print(f"  {c:12s} rate={rates[c]:+.3f} s/lap   life={life[c]:.0f} laps")

    tr = add_direct(tr_raw, rates, life)
    te = add_direct(te_raw, rates, life)
    extra = ["Tyre_Deg_Expected", "Tyre_Deg_Fraction"]
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
    print("\n=== tyre-direct features ===")
    for n, v in imp.items():
        if "Tyre_Deg" in n or "Tyre_Age" in n:
            print(f"  rank {imp.index.get_loc(n)+1:3d}  {n:40s} {v:.5f}")
    print("\n=== top 10 ===")
    for i, (n, v) in enumerate(imp.head(10).items(), 1):
        print(f"  {i:3d} {n:40s} {v:.5f}")


if __name__ == "__main__":
    main()
