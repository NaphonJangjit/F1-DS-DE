"""Feature-engineering v5 — Rain_mmh + Laps_Since_Rain + Pace_Residual on v3-best.

Adds, on top of the winning v3 feature set:
  Rain_mmh        : Open-Meteo hourly precipitation (as-of join)
  Laps_Since_Rain : laps since the last lap with Rain_mmh > 0.05  (track wetness decay)
  Pace_Residual   : prev-lap field median duration - course green median duration
                    (positive = field slower than the circuit's normal pace -> env. slowdown)

Compares v3-best vs v3 + these three. SCRATCH.
"""

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features
from _fe_v2 import NEW_COLS, green_mask, sec_mae, eval_all
from _fe_v3 import augment_v3, V3_COLS
from _rain_mmh import session_locations, fetch_precip, add_rain_mmh

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
CACHE_PATH = ROOT / "rain_mmh_cache.pkl"
RAIN_THRESH = 0.05

V5_COLS = ["Rain_mmh", "Laps_Since_Rain", "Pace_Residual"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_precip(loc):
    if CACHE_PATH.exists():
        return pickle.load(open(CACHE_PATH, "rb"))
    cache = {}
    for (lat, lon) in sorted(set(loc.values())):
        cache[(float(lat), float(lon))] = fetch_precip(lat, lon, "2023-01-01", "2025-12-31")
        time.sleep(0.25)
    pickle.dump(cache, open(CACHE_PATH, "wb"))
    return cache


def augment_v5(raw, loc, precip_cache):
    df = add_rain_mmh(raw, loc, precip_cache)   # Rain_mmh
    df = augment_v3(df)                          # v2 + v3

    # Laps_Since_Rain
    rain_flag = (pd.to_numeric(df["Rain_mmh"], errors="coerce") > RAIN_THRESH).astype(int)
    df["_rf"] = rain_flag.to_numpy()
    lap_rain = (df.groupby(["Session_Key", "Lap"], sort=False)["_rf"].max().reset_index()
                .sort_values(["Session_Key", "Lap"]))
    lap_rain["_rain_lap"] = lap_rain["Lap"].where(lap_rain["_rf"] > 0)
    lap_rain["_last"] = lap_rain.groupby("Session_Key")["_rain_lap"].ffill()
    lap_rain["Laps_Since_Rain"] = lap_rain["Lap"] - lap_rain["_last"]
    df = df.merge(lap_rain[["Session_Key", "Lap", "Laps_Since_Rain"]],
                  on=["Session_Key", "Lap"], how="left")
    df = df.drop(columns=["_rf"])

    # Pace_Residual = prev-lap field median duration - course green median (per session/sector)
    dur = pd.to_numeric(df["Duration"], errors="coerce").to_numpy(float)
    green = df["Track_Status"].isin(["GREEN", "UNKNOWN"]).to_numpy()
    df["_dur"] = dur
    df["_gd"] = np.where(green, dur, np.nan)
    df["_course_ref"] = df.groupby(["Session_Key", "Sector"], sort=False)["_gd"].transform("median").to_numpy()
    field_med = (df.groupby(["Session_Key", "Sector", "Lap"], sort=False)["_dur"].median()
                 .reset_index().sort_values(["Session_Key", "Sector", "Lap"]))
    field_med["_prev"] = field_med.groupby(["Session_Key", "Sector"])["_dur"].shift(1)
    df = df.merge(field_med[["Session_Key", "Sector", "Lap", "_prev"]],
                  on=["Session_Key", "Sector", "Lap"], how="left")
    df["Pace_Residual"] = df["_prev"] - df["_course_ref"]
    df = df.drop(columns=["_dur", "_gd", "_course_ref", "_prev"])

    return df.reset_index(drop=True)


def main():
    t0 = time.time()
    log("mapping + loading precip ...")
    loc = session_locations()
    precip_cache = load_precip(loc)
    log(f"precip non-empty={sum(1 for v in precip_cache.values() if not v.empty)}/{len(precip_cache)}")

    log("loading + preparing v5 frames ...")
    tr_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                        pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te_raw = pd.read_parquet(ROOT / "data_2025.parquet")

    extra = ["Tyre_Age", *NEW_COLS, *V3_COLS, *V5_COLS]
    tr_ml, tr_feats, _, _ = _prepare_lab(augment_v5(tr_raw, loc, precip_cache), extra)
    te_ml, te_feats, _, _ = _prepare_lab(augment_v5(te_raw, loc, precip_cache), extra)
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    log(f"features v5={len(feats)}")

    v3_best = [c for c in feats if c not in V5_COLS and c != "Will_Box"]
    v5 = [c for c in feats if c != "Will_Box"]
    log(f"v3_best={len(v3_best)}  v5={len(v5)}")

    results = {}
    for name, cols in [("v3_best", v3_best), ("v3+rain_mmh+laps+PaceRes", v5)]:
        log(f"training {name} ({len(cols)}) ...")
        t = time.time()
        pre = build_preprocessor(cols).fit(tr_ml[cols])
        rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                   min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                   random_state=42).fit(pre.transform(tr_ml[cols]),
                                                        tr_ml[TARGET].to_numpy(float))
        p = rf.predict(pre.transform(te_ml[cols]))
        m = eval_all(te_ml[TARGET].to_numpy(float), p, te_ml)
        results[name] = m
        log(f"  {name:26s} full={m['full']:.4f} green={m['green']:.4f} cdg={m['cdg']:.4f} "
            f"wet={m['wet']:.4f} sc={m['sc']:.4f} vsc={m['vsc']:.4f} "
            f"jeddah4={m['jeddah4']:.4f} silverstone={m['silverstone']:.4f} ({time.time()-t:.0f}s)")

    print("\n" + "=" * 90)
    for name, m in results.items():
        print(f"{name:26s} full={m['full']:.4f} green={m['green']:.4f} cdg={m['cdg']:.4f} "
              f"wet={m['wet']:.4f} sc={m['sc']:.4f} vsc={m['vsc']:.4f} "
              f"jeddah4={m['jeddah4']:.4f} silverstone={m['silverstone']:.4f}")
    print("=" * 90)

    with open(ROOT / "fe_v5_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"saved -> fe_v5_results.json  total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
