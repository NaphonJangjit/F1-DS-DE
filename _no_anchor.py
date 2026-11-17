"""Remove the duration pace-anchors so weak features can surface.

Anchors = Best_Duration_So_Far, Clean_Pace, Prev_Duration, Prev_Sector_Duration
(+ their roll5 mean/slope trend derivatives). These carry ~93% of RF importance and
may hide the weather/tyre/traffic signal for dry<->wet transitions.

Trains RF (2023+24 -> 2025) WITH vs WITHOUT anchors and reports:
  - MAE (full / green / wet / dry) degradation
  - feature importance WITHOUT anchors (what surfaces?)

SCRATCH.
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
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}

ANCHORS = ["Best_Duration_So_Far", "Clean_Pace", "Prev_Duration",
           "Prev_Duration_roll5_mean", "Prev_Duration_roll5_slope",
           "Prev_Sector_Duration", "Prev_Sector_Duration_roll5_mean",
           "Prev_Sector_Duration_roll5_slope"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


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
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    feats_no_anchor = [c for c in feats if c not in ANCHORS]
    log(f"features full={len(feats)}  no-anchor={len(feats_no_anchor)} (removed {len(ANCHORS)})")

    te_g = green_mask(te_ml)
    te_wet = te_ml["Rainfall"].to_numpy() == 1

    pre_full = build_preprocessor(feats).fit(tr_ml[feats])
    pre_na = build_preprocessor(feats_no_anchor).fit(tr_ml[feats_no_anchor])

    results = {}
    for name, cols, pre in [("full", feats, pre_full), ("no_anchor", feats_no_anchor, pre_na)]:
        log(f"training {name} ({len(cols)}) ...")
        t = time.time()
        rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                   min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                   random_state=42).fit(pre.transform(tr_ml[cols]),
                                                        tr_ml[TARGET].to_numpy(float))
        p = rf.predict(pre.transform(te_ml[cols]))
        yte = te_ml[TARGET].to_numpy(float)
        results[name] = {"model": rf, "pre": pre,
                         "full": sec_mae(yte, p),
                         "green": sec_mae(yte[te_g], p[te_g]),
                         "wet": sec_mae(yte[te_wet], p[te_wet]),
                         "dry": sec_mae(yte[~te_wet], p[~te_wet])}
        m = results[name]
        log(f"  {name:10s} full={m['full']:.4f} green={m['green']:.4f} wet={m['wet']:.4f} "
            f"dry={m['dry']:.4f} ({time.time()-t:.0f}s)")

    # feature importance without anchors
    imp = pd.Series(results["no_anchor"]["model"].feature_importances_,
                    index=results["no_anchor"]["pre"].get_feature_names_out()).sort_values(ascending=False)
    print("\n=== TOP 30 IMPORTANCE (no anchor) ===")
    for i, (n, v) in enumerate(imp.head(30).items(), 1):
        print(f"  {i:3d} {n:40s} {v:.5f}")

    print("\n=== KEY NON-ANCHOR features (weather / tyre / traffic / status) ===")
    for n, v in imp.items():
        if any(k in n for k in ["Rain", "Tyre_Age", "Humid", "Track_Temp", "Air_Temp",
                                "Pressure", "Traffic", "Interval", "Track_Status", "Lap"]):
            print(f"  rank {imp.index.get_loc(n)+1:3d}  {n:40s} {v:.5f}")

    print("\n=== MAE comparison ===")
    for name, m in results.items():
        print(f"  {name:10s} full={m['full']:.4f} green={m['green']:.4f} wet={m['wet']:.4f} dry={m['dry']:.4f}")


if __name__ == "__main__":
    main()
