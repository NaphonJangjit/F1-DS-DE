"""Feature-engineering v4 — Rain_mmh + Phase 5 + weather deltas (on top of v3).

Adds:
  Phase 5.2  : Rain_mmh (Open-Meteo hourly precipitation, as-of join)
  Phase 5    : Prev_Sector_Residual, Is_Leader, Dirty_Air_Laps
  Phase 4    : Track_Temp_Delta_3, Humidity_Delta_3

Builds on _fe_v3.augment_v3. Precip is disk-cached. SCRATCH.
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
from _fe_v2 import augment, NEW_COLS, green_mask, sec_mae, eval_all
from _fe_v3 import augment_v3, V3_COLS
from _rain_mmh import session_locations, fetch_precip, add_rain_mmh

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
CACHE_PATH = ROOT / "rain_mmh_cache.pkl"

V4_COLS = ["Rain_mmh", "Prev_Sector_Residual", "Is_Leader", "Dirty_Air_Laps",
           "Track_Temperature_Delta_3", "Humidity_Delta_3"]


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


def augment_v4(raw, loc, precip_cache):
    df = add_rain_mmh(raw, loc, precip_cache)   # Rain_mmh
    df = augment_v3(df)                          # v2 + v3

    df["Prev_Sector_Residual"] = (pd.to_numeric(df["Prev_Sector_Duration"], errors="coerce")
                                  - pd.to_numeric(df["Clean_Pace"], errors="coerce"))
    df["Is_Leader"] = (pd.to_numeric(df["Interval_Ahead"], errors="coerce") >= 59.9).astype(float)

    dirty = (pd.to_numeric(df["Interval_Ahead"], errors="coerce") < 1.5).astype(int)
    df["_dirty"] = dirty.to_numpy()
    lap_dirty = (df.groupby(["Session_Key", "Driver", "Lap"], sort=False)["_dirty"].max()
                 .reset_index().sort_values(["Session_Key", "Driver", "Lap"]))
    lap_dirty["_dsum"] = (lap_dirty.groupby(["Session_Key", "Driver"])["_dirty"]
                          .transform(lambda s: s.rolling(3, min_periods=1).sum().shift(1)))
    df = df.merge(lap_dirty[["Session_Key", "Driver", "Lap", "_dsum"]],
                  on=["Session_Key", "Driver", "Lap"], how="left")
    df["Dirty_Air_Laps"] = df["_dsum"].astype(float)
    df = df.drop(columns=["_dirty", "_dsum"])

    for c in ["Track_Temperature", "Humidity"]:
        lap_w = (df.groupby(["Session_Key", "Lap"], sort=False)[c].first().reset_index()
                 .sort_values(["Session_Key", "Lap"]))
        lap_w["_lag"] = lap_w.groupby("Session_Key")[c].shift(3)
        df = df.merge(lap_w[["Session_Key", "Lap", "_lag"]], on=["Session_Key", "Lap"], how="left")
        df[f"{c}_Delta_3"] = pd.to_numeric(df[c], errors="coerce") - df["_lag"]
        df = df.drop(columns=["_lag"])

    return df.reset_index(drop=True)


def main():
    t0 = time.time()
    log("mapping sessions -> coords ...")
    loc = session_locations()
    log("loading precip cache ...")
    precip_cache = load_precip(loc)
    log(f"precip locations={len(precip_cache)} non-empty={sum(1 for v in precip_cache.values() if not v.empty)}")

    log("loading + preparing v4 frames ...")
    tr_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                        pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te_raw = pd.read_parquet(ROOT / "data_2025.parquet")

    extra = ["Tyre_Age", *NEW_COLS, *V3_COLS, *V4_COLS]
    tr_ml, tr_feats, _, _ = _prepare_lab(augment_v4(tr_raw, loc, precip_cache), extra)
    te_ml, te_feats, _, _ = _prepare_lab(augment_v4(te_raw, loc, precip_cache), extra)
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    log(f"features v4={len(feats)}")
    log(f"Rain_mmh>0 test share={(te_ml['Rain_mmh']>0).mean()*100:.2f}%  "
        f"corr with Rainfall={te_ml[['Rain_mmh','Rainfall']].corr().iloc[0,1]:.3f}")

    v3_full = [c for c in feats if c not in V4_COLS and c != "Will_Box" and c != "Tyre_Age"]
    v4_full = [c for c in feats if c != "Will_Box"]

    results = {}
    for name, cols in [("v3_full", v3_full), ("v4_full", v4_full)]:
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
        log(f"  {name:10s} full={m['full']:.4f} green={m['green']:.4f} cdg={m['cdg']:.4f} "
            f"wet={m['wet']:.4f} sc={m['sc']:.4f} vsc={m['vsc']:.4f} "
            f"jeddah4={m['jeddah4']:.4f} silverstone={m['silverstone']:.4f} ({time.time()-t:.0f}s)")

    print("\n" + "=" * 90)
    for name, m in results.items():
        print(f"{name:10s} full={m['full']:.4f} green={m['green']:.4f} cdg={m['cdg']:.4f} "
              f"wet={m['wet']:.4f} sc={m['sc']:.4f} vsc={m['vsc']:.4f} "
              f"jeddah4={m['jeddah4']:.4f} silverstone={m['silverstone']:.4f}")
    print("=" * 90)

    with open(ROOT / "fe_v4_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"saved -> fe_v4_results.json  total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
