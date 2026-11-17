"""Feature-engineering experiment harness (Phase 0) + first ablations.

Frozen eval sets reported every run: full / green / clean-dry-green / wet / SC /
VSC MAE, plus the two named rows (Jeddah lap 4, Silverstone) from top_errors_2025.csv.

Configs (one change per experiment, behind flags):
  baseline          : current pipeline (frozen RF, 72 features) -> reproduce 0.2649/0.2439
  -Will_Box         : drop the future-leaking Will_Box feature (plan 1.1)
  filter_train_only : outlier filter on TRAIN only, full unfiltered test (plan 1.2)
  green_anchors     : Prev_Duration / Best_Duration_So_Far walk back to last GREEN lap
                      (plan 2.2, the Jeddah lap-4 mechanism) — recomputed from the parquet.

SCRATCH. Flags live here; promotion to features.py/train.py comes after the ablation.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

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
JEDDAH_SK = 10022      # Saudi Arabia
SILVERSTONE_SK = 9947  # United Kingdom


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(np.mean(np.abs(np.exp(np.asarray(y_log, float)) - np.exp(np.asarray(p_log, float)))))


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


def green_anchors(df):
    """Recompute Prev_Duration / Best_Duration_So_Far from GREEN laps only.

    For each (Session_Key, Driver, Sector), sorted by Lap:
      Prev_Duration        = Duration of the most recent PRIOR green lap (walk back)
      Best_Duration_So_Far = min Duration over prior green laps
    Point-in-time safe: only prior laps, never current/future.
    """
    df = df.sort_values(["Session_Key", "Driver", "Sector", "Lap"]).reset_index(drop=True)
    green = df["Track_Status"].isin(["GREEN", "UNKNOWN"]).to_numpy()
    dur = pd.to_numeric(df["Duration"], errors="coerce").to_numpy()
    df["_gd"] = np.where(green, dur, np.nan)
    df["_ginf"] = np.where(green, dur, np.inf)
    key = ["Session_Key", "Driver", "Sector"]
    df["Prev_Duration"] = (df.groupby(key, sort=False)["_gd"]
                           .transform(lambda s: s.ffill().shift(1)))
    df["Best_Duration_So_Far"] = (df.groupby(key, sort=False)["_ginf"]
                                  .transform(lambda s: s.cummin().shift(1)).replace(np.inf, np.nan))
    return df.drop(columns=["_gd", "_ginf"])


def eval_all(yte, pred, te_ml):
    green = green_mask(te_ml)
    wet = te_ml["Rainfall"].to_numpy() == 1
    sc = te_ml["Track_Status"].isin(["SC", "SC_ENDING"]).to_numpy()
    vsc = te_ml["Track_Status"].isin(["VSC", "VSC_ENDING"]).to_numpy()
    dry = ~wet
    cdg = green & dry
    sk = te_ml["Session_Key"].to_numpy()
    lap = te_ml["Lap"].to_numpy()
    jl4 = (sk == JEDDAH_SK) & (lap == 4)
    sil = sk == SILVERSTONE_SK
    sil_wet = sil & (wet | te_ml["Tyre"].isin(["INTERMEDIATE", "WET"]).to_numpy())
    err = np.abs(np.exp(np.asarray(yte, float)) - np.exp(np.asarray(pred, float)))

    def m(mask):
        return float(err[mask].mean()) if mask.any() else float("nan")

    return {"full": m(np.ones(len(err), bool)), "green": m(green),
            "clean_dry_green": m(cdg), "wet": m(wet), "sc": m(sc), "vsc": m(vsc),
            "jeddah_lap4": m(jl4), "silverstone": m(sil), "silverstone_wet": m(sil_wet)}


def main():
    t0 = time.time()
    log("loading ...")
    tr_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                        pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te_raw = pd.read_parquet(ROOT / "data_2025.parquet")

    log("preparing base frames ...")
    tr_ml, tr_feats, removed_tr, _ = _prepare_lab(tr_raw, [])
    te_ml_f, te_feats_f, removed_te, _ = _prepare_lab(te_raw, [])

    orig = wrl.remove_duration_outliers
    wrl.remove_duration_outliers = lambda df, k=4.0, mad_floor=0.5: (df, 0)
    te_ml_unf, te_feats_unf, _, _ = _prepare_lab(te_raw, [])
    wrl.remove_duration_outliers = orig

    log("preparing green-anchor frames ...")
    tr_ml_ga, tr_feats_ga, _, _ = _prepare_lab(green_anchors(tr_raw), [])
    te_ml_ga_f, te_feats_ga_f, _, _ = _prepare_lab(green_anchors(te_raw), [])

    feats = [c for c in _features(tr_feats, tr_ml)
             if c in te_ml_f.columns and c in te_ml_unf.columns and c in te_ml_ga_f.columns]
    feats_nowb = [c for c in feats if c != "Will_Box"]
    log(f"features={len(feats)}  -Will_Box={len(feats_nowb)}  "
        f"train_rows={len(tr_ml):,}  test_filtered={len(te_ml_f):,}  test_unfiltered={len(te_ml_unf):,}")
    log(f"outlier filter removed: train={removed_tr:,}  test={removed_te:,}")

    configs = [
        ("baseline", tr_ml, te_ml_f, feats),
        ("-Will_Box", tr_ml, te_ml_f, feats_nowb),
        ("filter_train_only", tr_ml, te_ml_unf, feats),
        ("green_anchors", tr_ml_ga, te_ml_ga_f, feats),
    ]

    results = {}
    for name, trm, tem, cols in configs:
        log(f"training {name} ...")
        t = time.time()
        pre = build_preprocessor(cols).fit(trm[cols])
        rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                   min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                   random_state=42).fit(pre.transform(trm[cols]),
                                                        trm[TARGET].to_numpy(float))
        p = rf.predict(pre.transform(tem[cols]))
        m = eval_all(tem[TARGET].to_numpy(float), p, tem)
        m["n_test"] = int(len(tem))
        results[name] = m
        log(f"  {name:18s} full={m['full']:.4f} green={m['green']:.4f} "
            f"cdg={m['clean_dry_green']:.4f} wet={m['wet']:.4f} "
            f"jeddah4={m['jeddah_lap4']:.4f} silverstone={m['silverstone']:.4f} "
            f"({time.time()-t:.0f}s)")

    cols_show = ["full", "green", "clean_dry_green", "wet", "sc", "vsc",
                 "jeddah_lap4", "silverstone", "silverstone_wet"]
    print("\n" + "=" * 96)
    print(f"{'config':18s}" + "".join(f"{c:>10s}" for c in cols_show))
    print("-" * 96)
    for name, m in results.items():
        print(f"{name:18s}" + "".join(f"{m[c]:10.4f}" for c in cols_show))
    print("=" * 96)
    print("baseline reference: full 0.2649 / green 0.2439")

    with open(ROOT / "fe_exp_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"saved -> fe_exp_results.json  total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
