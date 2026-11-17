"""Rank every 2025 track/race by frozen-RF error, high -> low.

Two error columns per track:
  full   : model scored on ALL laps (SC/VSC/wet included)
  green  : model scored on green-only laps (SC/VSC + transition removed)

Plus wet%/SC%/VSC% context so the ranking is explainable. SCRATCH — read-only.
"""

import json
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


def sec_mae(y_log, pred_log):
    a = np.exp(np.asarray(y_log, float)); p = np.exp(np.asarray(pred_log, float))
    return float(mean_absolute_error(a, p))


def make_rf():
    return RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                 min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                 random_state=42)


def drop_sc_vsc(df):
    d = df.copy()
    d["_is_sc"] = d["Track_Status"].isin(SC_SET)
    laps = (d[["Session_Key", "Driver", "Lap", "_is_sc"]]
            .groupby(["Session_Key", "Driver", "Lap"], as_index=False).max())
    laps = laps.sort_values(["Session_Key", "Driver", "Lap"])
    laps["_prev_sc"] = laps.groupby(["Session_Key", "Driver"])["_is_sc"].shift(1).fillna(False)
    d = d.merge(laps[["Session_Key", "Driver", "Lap", "_prev_sc"]],
                on=["Session_Key", "Driver", "Lap"], how="left")
    drop = d["_is_sc"] | d["_prev_sc"].fillna(False)
    return d[~drop].drop(columns=["_is_sc", "_prev_sc"])


def per_track(ml, y, p):
    out = {}
    for sk, g in ml.groupby("Session_Key", sort=False):
        out[int(sk)] = sec_mae(y[g.index.to_numpy()], p[g.index.to_numpy()])
    return out


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")
    te_g = drop_sc_vsc(te)

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    te_g_ml, te_g_feats, _, _ = _prepare_lab(te_g, [])

    feats = _features(tr_feats, tr_ml)
    feats = [c for c in feats if c in te_ml.columns]

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = make_rf().fit(pre.transform(tr_ml[feats]), tr_ml[TARGET].to_numpy(float))

    yte = te_ml[TARGET].to_numpy(float)
    p_full = rf.predict(pre.transform(te_ml[feats]))
    yte_g = te_g_ml[TARGET].to_numpy(float)
    p_green = rf.predict(pre.transform(te_g_ml[feats]))

    full_map = per_track(te_ml, yte, p_full)
    green_map = per_track(te_g_ml, yte_g, p_green)

    country = te.groupby("Session_Key")["Country"].first().to_dict()
    rows = []
    for sk, g in te.groupby("Session_Key", sort=False):
        sk = int(sk)
        rows.append({
            "track": str(country.get(sk, "?")),
            "full_mae": full_map.get(sk, np.nan),
            "green_mae": green_map.get(sk, np.nan),
            "wet_pct": float(g["Rainfall"].mean() * 100),
            "sc_pct": float(g["Track_Status"].isin(["SC", "SC_ENDING"]).mean() * 100),
            "vsc_pct": float(g["Track_Status"].isin(["VSC", "VSC_ENDING"]).mean() * 100),
        })
    t = pd.DataFrame(rows).sort_values("full_mae", ascending=False).reset_index(drop=True)
    t.insert(0, "rank", t.index + 1)

    pd.set_option("display.width", 140)
    print("=== 2025 tracks by FULL error (all laps), high -> low ===")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    g = t.sort_values("green_mae", ascending=False).reset_index(drop=True)
    print("\n=== same, ranked by GREEN-only error (predictable laps), high -> low ===")
    print(g[["track", "green_mae", "wet_pct", "sc_pct"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))

    t.to_json(ROOT / "track_rank.json", orient="records", indent=2)
    print(f"\ntotal={time.time()-t0:.0f}s  saved -> track_rank.json")


if __name__ == "__main__":
    main()
