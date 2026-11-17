"""Drift detection + machine-unlearning for the frozen RF.

Warm start : 2023 (all) + the FIRST 18 GPs of 2024 (last 6 races of 2024 dropped).
Test       : 2025, streamed chronologically race-by-race (prequential).

Per-race MAE_s feeds a DDM (Drift Detection Method). On a drift alarm the RF is
retrained, and "unlearning old data" is realised by dropping the oldest YEAR of the
training pool:
    pool = 2023 + first18(2024)          (warm)
    drift #1 -> drop 2023                pool = first18(2024) + observed 2025
    drift #2 -> drop first18(2024)       pool = observed 2025 only

Three variants (all point-in-time: retrain uses only races observed so far):
  static   : warm model frozen, never retrained
  unlearn  : retrain + drop oldest year on drift
  keepall  : retrain + KEEP everything (warm + observed 2025) on drift
             -> isolates whether *unlearning* specifically helps vs just retraining.

SCRATCH — does not modify main code.
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

N_WARM_GP = 18            # first N GPs of 2024 (plus all of 2023)
DDM_MIN_INSTANCES = 5     # start drift-checking after this many races
DDM_WARN = 2.0
DDM_DRIFT = 3.0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def make_rf():
    return RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                 min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                 random_state=42)


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


class DDM:
    """Drift Detection Method on a univariate error stream (per-race MAE_s)."""

    def __init__(self, warn=DDM_WARN, drift=DDM_DRIFT, min_instances=DDM_MIN_INSTANCES):
        self.warn = warn
        self.drift = drift
        self.min_instances = min_instances
        self.reset()

    def reset(self):
        self.hist = []
        self.p_min = float("inf")
        self.s_min = float("inf")

    def add(self, x):
        self.hist.append(x)
        n = len(self.hist)
        if n < self.min_instances:
            return False
        p = float(np.mean(self.hist))
        s = float(np.std(self.hist, ddof=1)) if n >= 2 else 0.0
        self.p_min = min(self.p_min, p)
        self.s_min = min(self.s_min, s)
        if p + s >= self.p_min + self.drift * self.s_min:
            self.reset()
            return True
        return False


def run_stream(Xw, yw, warm23, Xte, yte, te_sk, race_order, mode):
    """Prequential 2025; returns (preds, drift_races, per_race_mae)."""
    n = len(yte)
    preds = np.empty(n, dtype=float)
    model = make_rf().fit(Xw, yw)
    ddm = DDM()
    dropped23 = dropped24 = False
    obs = []
    drift_races = []
    per_race = []

    for sk in race_order:
        idx = np.where(te_sk == sk)[0]
        p = model.predict(Xte[idx])
        preds[idx] = p
        mae = sec_mae(yte[idx], p)
        per_race.append((int(sk), len(idx), mae))
        obs.extend(idx.tolist())

        if mode != "static" and ddm.add(mae):
            drift_races.append(int(sk))
            if mode == "unlearn":
                if not dropped23:
                    dropped23 = True
                elif not dropped24:
                    dropped24 = True
                warm_mask = np.zeros(len(warm23), dtype=bool) if dropped24 else (~warm23 if dropped23 else np.ones(len(warm23), dtype=bool))
            else:  # keepall
                warm_mask = np.ones(len(warm23), dtype=bool)

            Xp = np.vstack([Xw[warm_mask], Xte[obs]])
            yp = np.concatenate([yw[warm_mask], yte[obs]])
            model = make_rf().fit(Xp, yp)

    return preds, drift_races, per_race


def main():
    t0 = time.time()
    log("loading data ...")
    raw23 = pd.read_parquet(ROOT / "data_2023.parquet")
    raw24 = pd.read_parquet(ROOT / "data_2024.parquet")
    raw25 = pd.read_parquet(ROOT / "data_2025.parquet")

    sess24 = list(raw24.groupby("Session_Key")["Time_Start"].min().sort_values().index)
    first18 = set(sess24[:N_WARM_GP])
    warm_raw = pd.concat([raw23, raw24[raw24["Session_Key"].isin(first18)]], ignore_index=True)
    log(f"warm pool: 2023 ({raw23['Session_Key'].nunique()} GPs) + first "
        f"{len(first18)} GPs of 2024 (dropped {raw24['Session_Key'].nunique() - len(first18)} races of 2024)")

    log("preparing (outlier filter ON) ...")
    warm_ml, warm_feats, _, _ = _prepare_lab(warm_raw, ["Year"])
    te_ml, te_feats, _, _ = _prepare_lab(raw25, ["Year"])
    feats = [c for c in _features(warm_feats, warm_ml) if c in te_ml.columns]
    log(f"features={len(feats)}  warm_rows={len(warm_ml):,}  test_rows={len(te_ml):,}")

    pre = build_preprocessor(feats).fit(warm_ml[feats])
    Xw = pre.transform(warm_ml[feats])
    yw = warm_ml[TARGET].to_numpy(float)
    warm23 = (warm_ml["Year"].to_numpy() == 2023)
    Xte = pre.transform(te_ml[feats])
    yte = te_ml[TARGET].to_numpy(float)
    te_sk = te_ml["Session_Key"].to_numpy()

    race_order = list(te_ml.groupby("Session_Key")["Time_Start"].min().sort_values().index)
    country_of = raw25.groupby("Session_Key")["Country"].first().to_dict()
    te_g = green_mask(te_ml)

    summary = {}
    per_race_by_mode = {}
    for mode in ["static", "unlearn", "keepall"]:
        log(f"running {mode} ...")
        t = time.time()
        preds, drift_races, per_race = run_stream(Xw, yw, warm23, Xte, yte, te_sk, race_order, mode)
        full = sec_mae(yte, preds)
        green = sec_mae(yte[te_g], preds[te_g])
        summary[mode] = {"full_mae": full, "green_mae": green, "n_drifts": len(drift_races),
                         "drift_races": [str(country_of.get(sk, sk)) for sk in drift_races]}
        per_race_by_mode[mode] = [(str(country_of.get(sk, sk)), int(sk), int(n), mae)
                                  for (sk, n, mae) in per_race]
        log(f"  {mode:8s} full={full:.4f} green={green:.4f} drifts={len(drift_races)} "
            f"{[country_of.get(sk) for sk in drift_races]} ({time.time()-t:.0f}s)")

    # per-race trajectory
    print("\n=== per-race MAE_s (static | unlearn | keepall) ===")
    print(f"{'race':20s} {'static':>8s} {'unlearn':>8s} {'keepall':>8s}")
    for sk in race_order:
        name = str(country_of.get(sk, sk))
        s = next(x[3] for x in per_race_by_mode["static"] if x[1] == int(sk))
        u = next(x[3] for x in per_race_by_mode["unlearn"] if x[1] == int(sk))
        k = next(x[3] for x in per_race_by_mode["keepall"] if x[1] == int(sk))
        print(f"{name:20s} {s:8.4f} {u:8.4f} {k:8.4f}")

    print("\n=== SUMMARY ===")
    for mode, d in summary.items():
        print(f"  {mode:8s} full={d['full_mae']:.4f} green={d['green_mae']:.4f} "
              f"drifts={d['n_drifts']} {d['drift_races']}")

    with open(ROOT / "drift_unlearn_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_race": {m: [[r[0], r[2], r[3]] for r in rows]
                                                    for m, rows in per_race_by_mode.items()},
                   "n_warm_gp": N_WARM_GP,
                   "ddm": {"min_instances": DDM_MIN_INSTANCES, "warn": DDM_WARN, "drift": DDM_DRIFT},
                   "features": len(feats),
                   "warm_rows": int(len(warm_ml)), "test_rows": int(len(te_ml))}, f, indent=2)
    log(f"saved -> drift_unlearn_results.json  total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
