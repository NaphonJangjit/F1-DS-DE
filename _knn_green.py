"""KNN on the green-only (SC/VSC + red-flag removed) test set.

Feature selection via f_regression (model-agnostic, no RF). StandardScaler + k=10
distance-weighted KNN (the best config found earlier). Compare to RF green-only = 0.2143.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_regression
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from f1pit.api import fetch
from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
CORES = max(2, (os.cpu_count() or 8) // 2)
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}
K_MAX = 50


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


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


def red_flag_laps(sessions):
    out = {}
    for sk in sessions:
        rc = fetch("race_control", session_key=int(sk))
        laps = set()
        if not rc.empty:
            for _, r in rc.iterrows():
                if str(r.get("flag", "")).upper() == "RED":
                    lap = pd.to_numeric(r.get("lap_number"), errors="coerce")
                    if pd.notna(lap):
                        laps.add(int(lap)); laps.add(int(lap) + 1)
        out[int(sk)] = laps
    return out


def drop_red(df, rl):
    keep = np.ones(len(df), bool)
    sk = df["Session_Key"].to_numpy(); lp = df["Lap"].to_numpy()
    for i in range(len(df)):
        if int(lp[i]) in rl.get(int(sk[i]), set()):
            keep[i] = False
    return df[keep]


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")
    rl = red_flag_laps(te["Session_Key"].unique())
    te = drop_red(drop_sc_vsc(te), rl)
    log(f"test rows after green+red filter = {len(te):,}")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    Xtr = pre.transform(tr_ml[feats])
    Xte = pre.transform(te_ml[feats])
    ytr = tr_ml[TARGET].to_numpy(float)
    yte = te_ml[TARGET].to_numpy(float)
    te_wet = te_ml["Rainfall"].to_numpy() == 1

    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xte_s = sc.transform(Xte)

    # feature selection (f_regression, no RF)
    fscore, _ = f_regression(Xtr_s, ytr)
    order = np.argsort(fscore)[::-1]

    print(f"\n=== KNN (k=10 distance-weighted) on green-only, by top-N f_regression features ===")
    print(f"{'top-N':6s} {'overall':>9s} {'dry':>8s} {'wet':>8s}")
    for n_feats in [5, 10, 15, 20, 30, 50]:
        cols = order[:n_feats]
        nn = NearestNeighbors(n_neighbors=K_MAX, algorithm="brute", n_jobs=CORES).fit(Xtr_s[:, cols])
        dist, idx = nn.kneighbors(Xte_s[:, cols])
        d = dist[:, :10]
        yn = ytr[idx[:, :10]]
        w = 1.0 / (d + 1e-8)
        p = (yn * w).sum(axis=1) / w.sum(axis=1)
        print(f"top-{n_feats:<3d} {sec_mae(yte, p):9.4f} {sec_mae(yte[~te_wet], p[~te_wet]):8.4f} "
              f"{sec_mae(yte[te_wet], p[te_wet]):8.4f}")

    print("\n(reference: RF green-only overall=0.2143 / dry=0.2086 / wet=0.7753)")
    print(f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
