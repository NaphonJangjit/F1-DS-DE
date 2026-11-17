"""Exclude SC/VSC (unpredictable regimes) explicitly and retrain.

Rule: drop every sector whose Track_Status is SC/VSC/SC_ENDING/VSC_ENDING, and
drop the immediately-following lap of each driver (so Prev_* features don't leak
the removed slow lap). Wet is KEPT as a feature.

Variants (all scored on the SAME green-only 2025 test set, for fairness):
  baseline    : train on full data (outlier filter ON, SC/VSC kept)  [= 0.265 ref]
  green       : train on green-only data (outlier filter ON)
  green_noout : train on green-only data (outlier filter OFF) — tests whether the
                deterministic SC/VSC rule can replace the duration-based heuristic.

SCRATCH — does not modify main code.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

import f1pit.wind_rain_lab as wrl
from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}


def sec_metrics(y_log, pred_log):
    a = np.exp(np.asarray(y_log, float)); p = np.exp(np.asarray(pred_log, float))
    return float(mean_absolute_error(a, p)), float(np.sqrt(mean_squared_error(a, p)))


def make_rf():
    return RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                 min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                 random_state=42)


def drop_sc_vsc(df):
    """Remove SC/VSC sectors + the lap immediately following any SC/VSC lap."""
    d = df.copy()
    d["_is_sc"] = d["Track_Status"].isin(SC_SET)
    laps = (d[["Session_Key", "Driver", "Lap", "_is_sc"]]
            .groupby(["Session_Key", "Driver", "Lap"], as_index=False).max())
    laps = laps.sort_values(["Session_Key", "Driver", "Lap"])
    laps["_prev_sc"] = laps.groupby(["Session_Key", "Driver"])["_is_sc"].shift(1).fillna(False)
    d = d.merge(laps[["Session_Key", "Driver", "Lap", "_prev_sc"]],
                on=["Session_Key", "Driver", "Lap"], how="left")
    drop = d["_is_sc"] | d["_prev_sc"].fillna(False)
    return d[~drop].drop(columns=["_is_sc", "_prev_sc"]), int(drop.sum())


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_g, n_drop_tr = drop_sc_vsc(tr)
    te_g, n_drop_te = drop_sc_vsc(te)
    print(f"SC/VSC+transition sectors removed: train {n_drop_tr:,} ({n_drop_tr/len(tr)*100:.1f}%)  "
          f"test {n_drop_te:,} ({n_drop_te/len(te)*100:.1f}%)")

    # --- prepare (outlier filter ON) ---
    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    trg_ml, trg_feats, _, _ = _prepare_lab(tr_g, [])
    teg_ml, teg_feats, _, _ = _prepare_lab(te_g, [])

    # --- green-only, outlier filter OFF ---
    orig_rod = wrl.remove_duration_outliers
    wrl.remove_duration_outliers = lambda df, k=4.0, mad_floor=0.5: (df, 0)
    trg_ml_no, trg_feats_no, _, _ = _prepare_lab(tr_g, [])
    wrl.remove_duration_outliers = orig_rod

    feats = _features(tr_feats, tr_ml)
    feats = [c for c in feats if c in te_ml.columns and c in teg_ml.columns and c in trg_ml_no.columns]
    print(f"features={len(feats)}  train(base)={len(tr_ml):,}  train(green)={len(trg_ml):,}  "
          f"train(green_noout)={len(trg_ml_no):,}  test(green)={len(teg_ml):,}")

    # --- models (each with its own preprocessor fit on its own train) ---
    configs = [
        ("baseline", tr_ml, te_ml),          # full data + full test
        ("green", trg_ml, teg_ml),
        ("green_noout", trg_ml_no, teg_ml),
    ]
    # common green-only eval set for the fair comparison
    teg_ml_eval = teg_ml
    yte_g = teg_ml_eval[TARGET].to_numpy(float)
    te_wet = (teg_ml_eval["Rainfall"] == 1).to_numpy()
    te_dry = ~te_wet

    preds = {}
    for name, trm, _ in configs:
        pre = build_preprocessor(feats).fit(trm[feats])
        Xtr = pre.transform(trm[feats])
        Xte_g = pre.transform(teg_ml_eval[feats])
        ytr = trm[TARGET].to_numpy(float)

        for mname, est, needs_scale in [("RF", make_rf(), False),
                                        ("Lin", LinearRegression(), True),
                                        ("Ridge", Ridge(alpha=1.0), True)]:
            t = time.time()
            if needs_scale:
                sc = StandardScaler().fit(Xtr)
                est.fit(sc.transform(Xtr), ytr)
                p = est.predict(sc.transform(Xte_g))
            else:
                est.fit(Xtr, ytr)
                p = est.predict(Xte_g)
            preds[(name, mname)] = p
            print(f"  fit {name}/{mname} ({time.time()-t:.0f}s)")

    # --- report: all on the green-only eval set ---
    print(f"\nEvaluation on green-only 2025 test ({len(yte_g):,} rows; wet={int(te_wet.sum()):,}, dry={int(te_dry.sum()):,})")
    print(f"{'variant':14s} {'model':6s} {'overall':>9s} {'wet':>8s} {'dry':>8s}")
    print("-" * 48)
    for name in ["baseline", "green", "green_noout"]:
        for mname in ["RF", "Lin", "Ridge"]:
            p = preds[(name, mname)]
            o = sec_metrics(yte_g, p)[0]
            w = sec_metrics(yte_g[te_wet], p[te_wet])[0] if te_wet.any() else float("nan")
            d = sec_metrics(yte_g[te_dry], p[te_dry])[0]
            print(f"{name:14s} {mname:6s} {o:9.4f} {w:8.4f} {d:8.4f}")

    # baseline reference on FULL test (with SC/VSC) for continuity
    pre_full = build_preprocessor(feats).fit(tr_ml[feats])
    Xte_full = pre_full.transform(te_ml[feats])
    yte_full = te_ml[TARGET].to_numpy(float)
    rf = make_rf().fit(pre_full.transform(tr_ml[feats]), tr_ml[TARGET].to_numpy(float))
    full_mae = sec_metrics(yte_full, rf.predict(Xte_full))[0]
    print(f"\n[reference] baseline RF on FULL test (SC/VSC included) = {full_mae:.4f}")

    # per-race RF on green-only test
    country_of = te.groupby("Session_Key")["Country"].first().to_dict()
    races = []
    for sk, g in teg_ml_eval.groupby("Session_Key", sort=False):
        idx = g.index.to_numpy()
        b = sec_metrics(yte_g[idx], preds[("baseline", "RF")][idx])[0]
        gr = sec_metrics(yte_g[idx], preds[("green", "RF")][idx])[0]
        races.append((str(country_of.get(sk, "?")), len(idx), b, gr))
    races.sort(key=lambda r: -r[2])
    print("\nRF per-race on green-only test (hardest 8, baseline -> green):")
    print(f"{'race':22s} {'rows':>6s} {'baseline':>8s} {'green':>8s} {'delta':>8s}")
    for name, n, b, gr in races[:8]:
        print(f"{name:22s} {n:>6d} {b:>8.4f} {gr:>8.4f} {gr-b:>+8.4f}")

    print(f"\ntotal={time.time()-t0:.0f}s  DONE")


if __name__ == "__main__":
    main()
