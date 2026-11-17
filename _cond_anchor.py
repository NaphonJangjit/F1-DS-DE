"""Condition-aware pace anchors: dry anchors from dry laps, wet anchors from wet laps.

Problem: Prev_Duration / Best_Duration_So_Far / Clean_Pace are computed over ALL prior
laps regardless of wet/dry. On a wet lap the anchor is often a fast DRY lap -> the
model over-predicts (predicts too fast), which is exactly the wet-regime bias we saw.

Fix: for each sector, walk back to the most recent prior lap with the SAME condition
(Rainfall 0/1) AND green (skip SC/VSC). Fall back to the original anchor when no
same-condition lap exists yet.

Scratch: recompute anchors in the raw table, then _prepare_lab re-derives the trend
features from the corrected anchors.
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


def cond_anchors(df):
    # pit in/out laps (lap-level): out = Tyre_Age==1 (first lap on fresh tyre),
    # in = the lap immediately before an out-lap.
    lap_info = (df[["Session_Key", "Driver", "Lap", "Tyre_Age"]]
                .drop_duplicates(["Session_Key", "Driver", "Lap"])
                .sort_values(["Session_Key", "Driver", "Lap"]))
    lap_info["_out"] = (pd.to_numeric(lap_info["Tyre_Age"], errors="coerce") == 1).astype(int)
    lap_info["_in"] = lap_info.groupby(["Session_Key", "Driver"])["_out"].shift(-1).fillna(0).astype(int)
    df = df.merge(lap_info[["Session_Key", "Driver", "Lap", "_out", "_in"]],
                  on=["Session_Key", "Driver", "Lap"], how="left")

    df = df.sort_values(["Session_Key", "Driver", "Sector", "Lap"]).reset_index(drop=True)
    dur = pd.to_numeric(df["Duration"], errors="coerce").to_numpy(float)
    green = df["Track_Status"].isin(["GREEN", "UNKNOWN"]).to_numpy()
    wet = (df["Rainfall"].to_numpy() == 1)
    valid = green & (np.nan_to_num(df["_in"].to_numpy(float)) == 0) & (np.nan_to_num(df["_out"].to_numpy(float)) == 0)

    df["_vdry"] = np.where(valid & ~wet, dur, np.nan)
    df["_vwet"] = np.where(valid & wet, dur, np.nan)
    key = ["Session_Key", "Driver", "Sector"]
    g = df.groupby(key, sort=False)

    dry_last = g["_vdry"].transform(lambda s: s.ffill().shift(1))
    wet_last = g["_vwet"].transform(lambda s: s.ffill().shift(1))
    dry_best = g["_vdry"].transform(lambda s: s.rolling(5, min_periods=1).min().shift(1))
    wet_best = g["_vwet"].transform(lambda s: s.rolling(5, min_periods=1).min().shift(1))
    dry_cp = g["_vdry"].transform(lambda s: s.rolling(3, min_periods=1).median().shift(1))
    wet_cp = g["_vwet"].transform(lambda s: s.rolling(3, min_periods=1).median().shift(1))

    raw_prev = pd.to_numeric(df["Prev_Duration"], errors="coerce")
    raw_best = pd.to_numeric(df["Best_Duration_So_Far"], errors="coerce")
    raw_cp = pd.to_numeric(df["Clean_Pace"], errors="coerce")

    def pick(dry, wetv, raw):
        x = np.where(wet, wetv.to_numpy(), dry.to_numpy())
        o = np.where(wet, dry.to_numpy(), wetv.to_numpy())  # first lap of a condition -> other condition
        x = np.where(np.isfinite(x), x, o)
        return np.where(np.isfinite(x), x, raw.to_numpy())

    df["Prev_Duration"] = pick(dry_last, wet_last, raw_prev)
    df["Best_Duration_So_Far"] = pick(dry_best, wet_best, raw_best)
    df["Clean_Pace"] = pick(dry_cp, wet_cp, raw_cp)

    return df.drop(columns=["_vdry", "_vwet", "_out", "_in"])


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    log("preparing baseline + cond-anchor frames ...")
    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    tr_ca, tr_feats_ca, _, _ = _prepare_lab(cond_anchors(tr), [])
    te_ca, te_feats_ca, _, _ = _prepare_lab(cond_anchors(te), [])

    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    feats_ca = [c for c in _features(tr_feats_ca, tr_ca) if c in te_ca.columns]
    log(f"features base={len(feats)} cond-anchor={len(feats_ca)}")

    te_g = green_mask(te_ml)
    te_wet = te_ml["Rainfall"].to_numpy() == 1
    te_g_ca = green_mask(te_ca)
    te_wet_ca = te_ca["Rainfall"].to_numpy() == 1

    results = {}
    for name, trm, tem, cols in [("baseline", tr_ml, te_ml, feats),
                                 ("cond_anchor", tr_ca, te_ca, feats_ca)]:
        log(f"training {name} ...")
        t = time.time()
        pre = build_preprocessor(cols).fit(trm[cols])
        rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                   min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                   random_state=42).fit(pre.transform(trm[cols]),
                                                        trm[TARGET].to_numpy(float))
        p = rf.predict(pre.transform(tem[cols]))
        yte = tem[TARGET].to_numpy(float)
        gmask = te_g if name == "baseline" else te_g_ca
        wmask = te_wet if name == "baseline" else te_wet_ca
        m = {"full": sec_mae(yte, p), "green": sec_mae(yte[gmask], p[gmask]),
             "wet": sec_mae(yte[wmask], p[wmask]), "dry": sec_mae(yte[~wmask], p[~wmask])}
        results[name] = m
        log(f"  {name:12s} full={m['full']:.4f} green={m['green']:.4f} wet={m['wet']:.4f} "
            f"dry={m['dry']:.4f} ({time.time()-t:.0f}s)")

    print("\n" + "=" * 56)
    print(f"{'config':12s} {'full':>8s} {'green':>8s} {'wet':>8s} {'dry':>8s}")
    print("-" * 56)
    for name, m in results.items():
        print(f"{name:12s} {m['full']:8.4f} {m['green']:8.4f} {m['wet']:8.4f} {m['dry']:8.4f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
