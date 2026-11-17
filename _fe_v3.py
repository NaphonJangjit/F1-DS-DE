"""Feature-engineering v3 — the remaining feasible phases (on top of _fe_v2).

Adds, point-in-time, on top of the v2 features:
  Phase 6 tyre/fuel : bare Tyre_Age (reinstated), Tyre_Age_x_Track_Temp,
                      Stint_Number, Is_First_Stint
  Phase 4 (5.3)     : Field_Pace_Residual (field median of Duration-Clean_Pace, prev lap)
  Phase 2.4         : Green_Deg_Slope (slope of last 3 green durations)
  Phase 7           : residual target  ln(Duration) - ln(Clean_Pace)  (separate config)

SCRATCH. Builds on _fe_v2.augment.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features
from _fe_v2 import augment, NEW_COLS, green_mask, sec_mae, eval_all

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)

V3_COLS = ["Tyre_Age_x_Track_Temp", "Stint_Number", "Is_First_Stint",
           "Field_Pace_Residual", "Green_Deg_Slope"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def augment_v3(raw):
    df = augment(raw)  # v2 (anchors + rain + restart)

    # bare Tyre_Age is already a column; we re-add it to features via extra_cols.

    # Tyre_Age x Track_Temp interaction
    df["Tyre_Age_x_Track_Temp"] = (pd.to_numeric(df["Tyre_Age_Total"], errors="coerce")
                                   * pd.to_numeric(df["Track_Temperature"], errors="coerce"))

    # Stint_Number: new stint when Tyre_Age_Total resets (drops vs previous lap)
    df = df.sort_values(["Session_Key", "Driver", "Lap"]).reset_index(drop=True)
    tta = pd.to_numeric(df["Tyre_Age_Total"], errors="coerce").to_numpy(float)
    new_stint = (tta < np.r_[np.inf, tta[:-1]]).astype(int)
    df["_ns"] = new_stint
    df["Stint_Number"] = df.groupby(["Session_Key", "Driver"], sort=False)["_ns"].cumsum() + 1
    df["Is_First_Stint"] = (df["Stint_Number"] == 1).astype(float)
    df = df.drop(columns=["_ns"])

    # Field_Pace_Residual: per (session, lap, sector) median over drivers of
    # (Duration - Clean_Pace), shifted to the PREVIOUS lap (point-in-time).
    dur = pd.to_numeric(df["Duration"], errors="coerce").to_numpy(float)
    cp = pd.to_numeric(df["Clean_Pace"], errors="coerce").to_numpy(float)
    df["_res"] = dur - cp
    fr = (df.groupby(["Session_Key", "Lap", "Sector"], sort=False)["_res"].median()
          .reset_index().sort_values(["Session_Key", "Sector", "Lap"]))
    fr["_prev"] = fr.groupby(["Session_Key", "Sector"])["_res"].shift(1)
    df = df.merge(fr[["Session_Key", "Lap", "Sector", "_prev"]],
                  on=["Session_Key", "Lap", "Sector"], how="left")
    df["Field_Pace_Residual"] = df["_prev"].astype(float)
    df = df.drop(columns=["_res", "_prev"])

    # Green_Deg_Slope: slope of last 3 green durations (per session/driver/sector).
    key = ["Session_Key", "Driver", "Sector"]
    green = df["Track_Status"].isin(["GREEN", "UNKNOWN"]).to_numpy()
    df["_gd"] = np.where(green, dur, np.nan)
    df = df.sort_values(key + ["Lap"]).reset_index(drop=True)
    df["_gd"] = np.where(df["Track_Status"].isin(["GREEN", "UNKNOWN"]).to_numpy(),
                         pd.to_numeric(df["Duration"], errors="coerce").to_numpy(float), np.nan)
    df["Green_Deg_Slope"] = df.groupby(key, sort=False)["_gd"].transform(
        lambda s: s.rolling(3, min_periods=2).apply(
            lambda w: (w[-1] - w[0]) / max(1, len(w) - 1) if len(w) >= 2 else np.nan,
            raw=True).shift(1))
    df = df.drop(columns=["_gd"])

    return df.reset_index(drop=True)


def main():
    t0 = time.time()
    log("loading ...")
    tr_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                        pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te_raw = pd.read_parquet(ROOT / "data_2025.parquet")

    log("preparing v3 frames ...")
    extra = ["Tyre_Age", *NEW_COLS, *V3_COLS]
    tr_ml, tr_feats, _, _ = _prepare_lab(augment_v3(tr_raw), extra)
    te_ml, te_feats, _, _ = _prepare_lab(augment_v3(te_raw), extra)
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    log(f"features v3={len(feats)}")

    v2_best = [c for c in feats if c not in V3_COLS and c != "Tyre_Age"]
    v2_best = [c for c in v2_best if c != "Will_Box"]
    v3_full = [c for c in feats if c != "Will_Box"]
    log(f"v2_best={len(v2_best)}  v3_full={len(v3_full)}")

    results = {}
    for name, cols in [("v2_best", v2_best), ("v2+v3_full", v3_full)]:
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
        log(f"  {name:12s} full={m['full']:.4f} green={m['green']:.4f} cdg={m['cdg']:.4f} "
            f"wet={m['wet']:.4f} jeddah4={m['jeddah4']:.4f} silverstone={m['silverstone']:.4f} "
            f"({time.time()-t:.0f}s)")

    # ---- Phase 7: residual target ----
    log("Phase 7 residual target (ln Duration - ln Clean_Pace) ...")
    cp = pd.to_numeric(te_ml["Clean_Pace"], errors="coerce").to_numpy(float)
    cp_tr = pd.to_numeric(tr_ml["Clean_Pace"], errors="coerce").to_numpy(float)
    ok_tr = np.isfinite(cp_tr) & (cp_tr > 0)
    ok_te = np.isfinite(cp) & (cp > 0)
    ytr = tr_ml[TARGET].to_numpy(float) - np.log(cp_tr)
    yte = te_ml[TARGET].to_numpy(float) - np.log(cp)
    cols = v3_full
    pre = build_preprocessor(cols).fit(tr_ml[cols])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[cols])[ok_tr], ytr[ok_tr])
    p_res = rf.predict(pre.transform(te_ml[cols]))
    pred_dur = cp * np.exp(p_res)
    actual = np.exp(te_ml[TARGET].to_numpy(float))
    err = np.abs(actual - pred_dur)
    m_res = {"full": float(err[ok_te].mean()), "n": int(ok_te.sum())}
    results["residual_target"] = {"full": m_res["full"]}
    log(f"  residual_target full={m_res['full']:.4f} (n={m_res['n']:,})")

    print("\n" + "=" * 80)
    for name, m in results.items():
        if "green" in m:
            print(f"{name:14s} full={m['full']:.4f} green={m['green']:.4f} cdg={m['cdg']:.4f} "
                  f"wet={m['wet']:.4f} sc={m['sc']:.4f} vsc={m['vsc']:.4f} "
                  f"jeddah4={m['jeddah4']:.4f} silverstone={m['silverstone']:.4f}")
        else:
            print(f"{name:14s} full={m['full']:.4f}")
    print("=" * 80)

    with open(ROOT / "fe_v3_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"saved -> fe_v3_results.json  total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
