"""Feature-engineering v2 — full pass, one report at the end.

Implements the feasible, high-value items from reportv2.md as post-processing on the
existing parquets (point-in-time safe), then ablades against the frozen RF baseline:

  baseline : 72 features (reproduce 0.2649 / 0.2439)
  +anchors : Prev_Duration / Best_Duration_So_Far walk back to the last
             "representative" lap (green + duration-plausible), plus Anchor_Age and
             Prev_Lap_NonGreen.
  +rain    : Rainfall_PrevLap, Rainfall_Delta, Frac_Field_On_Wet_PrevLap.
  +restart : Laps_Since_SC_VSC, Is_Restart_Lap.
  full     : all of the above, minus the future-leaking Will_Box.

Representative lap = GREEN/UNKNOWN AND duration <= 1.12 * (rolling median of the last
5 green durations, backward). This excludes wet and mid-lap-incident laps from being
used as pace anchors, while staying strictly backward-looking.

SCRATCH.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab, _rain_features
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}
JEDDAH_SK = 10022
SILVERSTONE_SK = 9947
TOL = 1.12

NEW_COLS = ["Anchor_Age", "Prev_Lap_NonGreen", "Rainfall_PrevLap", "Rainfall_Delta",
            "Frac_Field_On_Wet_PrevLap", "Laps_Since_SC_VSC", "Is_Restart_Lap"]


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


def augment(raw):
    """Add v2 features; override Prev_Duration / Best_Duration_So_Far with
    representative-lap walk-back (point-in-time)."""
    df = raw.copy()
    key = ["Session_Key", "Driver", "Sector"]
    df = df.sort_values(key + ["Lap"]).reset_index(drop=True)

    dur = pd.to_numeric(df["Duration"], errors="coerce").to_numpy(float)
    green = df["Track_Status"].isin(["GREEN", "UNKNOWN"]).to_numpy()
    lap = df["Lap"].to_numpy(float)

    df["_gd"] = np.where(green, dur, np.nan)
    df["_ginf"] = np.where(green, dur, np.inf)
    df["_lap"] = lap
    g = df.groupby(key, sort=False)

    ref = g["_gd"].transform(lambda s: s.rolling(5, min_periods=1).median().shift(1)).to_numpy()
    ref_safe = np.where(np.isfinite(ref), ref, np.inf)
    rep = green & (dur <= TOL * ref_safe)

    df["_rep_dur"] = np.where(rep, dur, np.nan)
    df["_rep_inf"] = np.where(rep, dur, np.inf)
    df["_rep_lap"] = np.where(rep, lap, np.nan)

    raw_prev = pd.to_numeric(df["Prev_Duration"], errors="coerce")
    raw_best = pd.to_numeric(df["Best_Duration_So_Far"], errors="coerce")
    df["Prev_Duration"] = g["_rep_dur"].transform(lambda s: s.ffill().shift(1)).fillna(raw_prev).to_numpy()
    df["Best_Duration_So_Far"] = (g["_rep_inf"].transform(lambda s: s.cummin().shift(1))
                                  .replace(np.inf, np.nan).fillna(raw_best).to_numpy())
    df["Anchor_Age"] = (lap - g["_rep_lap"].transform(lambda s: s.ffill().shift(1)).to_numpy()).astype(float)
    df = df.drop(columns=["_gd", "_ginf", "_lap", "_rep_dur", "_rep_inf", "_rep_lap"])

    # --- lap-level features (via merge) ---
    ng = (~green).astype(int)
    df["_ng"] = ng
    lap_ng = (df.groupby(["Session_Key", "Driver", "Lap"], sort=False)["_ng"].max().reset_index()
              .sort_values(["Session_Key", "Driver", "Lap"]))
    lap_ng["_prev_ng"] = lap_ng.groupby(["Session_Key", "Driver"])["_ng"].shift(1).fillna(0)
    df = df.merge(lap_ng[["Session_Key", "Driver", "Lap", "_prev_ng"]],
                  on=["Session_Key", "Driver", "Lap"], how="left")
    df["Prev_Lap_NonGreen"] = df["_prev_ng"].astype(float)
    df = df.drop(columns=["_ng", "_prev_ng"])

    # rainfall prev-lap / delta
    r = _rain_features(df)
    df["Rainfall_PrevLap"] = r["Rainfall_PrevLap"].to_numpy(float)
    df["Rainfall_Delta"] = r["Rainfall_Delta"].to_numpy(float)

    # field wet-tyre share, previous lap
    wet_tyre = df["Tyre"].isin(["INTERMEDIATE", "WET"]).astype(float).to_numpy()
    df["_wt"] = wet_tyre
    lap_wet = (df.groupby(["Session_Key", "Lap"], sort=False)["_wt"].mean().reset_index()
               .sort_values(["Session_Key", "Lap"]))
    lap_wet["_prev_wet"] = lap_wet.groupby("Session_Key")["_wt"].shift(1)
    df = df.merge(lap_wet[["Session_Key", "Lap", "_prev_wet"]], on=["Session_Key", "Lap"], how="left")
    df["Frac_Field_On_Wet_PrevLap"] = df["_prev_wet"].astype(float)
    df = df.drop(columns=["_wt", "_prev_wet"])

    # laps since SC/VSC + restart
    sc_flag = df["Track_Status"].isin(["SC", "VSC", "SC_ENDING", "VSC_ENDING"]).astype(int).to_numpy()
    df["_sc"] = sc_flag
    lap_sc = (df.groupby(["Session_Key", "Lap"], sort=False)["_sc"].max().reset_index()
              .sort_values(["Session_Key", "Lap"]))
    lap_sc["_sc_lap"] = lap_sc["Lap"].where(lap_sc["_sc"] > 0)
    lap_sc["_last_sc"] = lap_sc.groupby("Session_Key")["_sc_lap"].ffill()
    lap_sc["Laps_Since_SC_VSC"] = lap_sc["Lap"] - lap_sc["_last_sc"]
    df = df.merge(lap_sc[["Session_Key", "Lap", "Laps_Since_SC_VSC"]],
                  on=["Session_Key", "Lap"], how="left")
    df["Is_Restart_Lap"] = (df["Laps_Since_SC_VSC"] == 1).astype(float)
    df = df.drop(columns=["_sc"])

    return df.reset_index(drop=True)


def eval_all(yte, pred, te_ml):
    green = green_mask(te_ml)
    wet = te_ml["Rainfall"].to_numpy() == 1
    sc = te_ml["Track_Status"].isin(["SC", "SC_ENDING"]).to_numpy()
    vsc = te_ml["Track_Status"].isin(["VSC", "VSC_ENDING"]).to_numpy()
    cdg = green & ~wet
    sk = te_ml["Session_Key"].to_numpy()
    lap = te_ml["Lap"].to_numpy()
    jl4 = (sk == JEDDAH_SK) & (lap == 4)
    sil = sk == SILVERSTONE_SK
    sil_wet = sil & (wet | te_ml["Tyre"].isin(["INTERMEDIATE", "WET"]).to_numpy())
    err = np.abs(np.exp(np.asarray(yte, float)) - np.exp(np.asarray(pred, float)))

    def m(mask):
        return float(err[mask].mean()) if mask.any() else float("nan")

    return {"full": m(np.ones(len(err), bool)), "green": m(green), "cdg": m(cdg),
            "wet": m(wet), "sc": m(sc), "vsc": m(vsc),
            "jeddah4": m(jl4), "silverstone": m(sil), "silverstone_wet": m(sil_wet),
            "n": int(len(err))}


def main():
    t0 = time.time()
    log("loading ...")
    tr_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                        pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te_raw = pd.read_parquet(ROOT / "data_2025.parquet")

    log("preparing baseline frames ...")
    tr_ml, tr_feats, _, _ = _prepare_lab(tr_raw, [])
    te_ml, te_feats, _, _ = _prepare_lab(te_raw, [])
    feats_base = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]

    log("augmenting + preparing v2 frames ...")
    tr_ml_aug, tr_feats_aug, _, _ = _prepare_lab(augment(tr_raw), NEW_COLS)
    te_ml_aug, te_feats_aug, _, _ = _prepare_lab(augment(te_raw), NEW_COLS)
    feats_aug = [c for c in _features(tr_feats_aug, tr_ml_aug) if c in te_ml_aug.columns]
    log(f"features base={len(feats_base)}  aug={len(feats_aug)}")

    anchors = ["Anchor_Age", "Prev_Lap_NonGreen"]
    rain = ["Rainfall_PrevLap", "Rainfall_Delta", "Frac_Field_On_Wet_PrevLap"]
    restart = ["Laps_Since_SC_VSC", "Is_Restart_Lap"]

    configs = [
        ("baseline", tr_ml, te_ml, feats_base),
        ("+anchors", tr_ml_aug, te_ml_aug, feats_base + anchors),
        ("+anchors+rain", tr_ml_aug, te_ml_aug, feats_base + anchors + rain),
        ("+anchors+rain+restart", tr_ml_aug, te_ml_aug, feats_base + anchors + rain + restart),
        ("full(-Will_Box)", tr_ml_aug, te_ml_aug,
         [c for c in feats_base + anchors + rain + restart if c != "Will_Box"]),
    ]

    results = {}
    for name, trm, tem, cols in configs:
        log(f"training {name} ({len(cols)} feats) ...")
        t = time.time()
        pre = build_preprocessor(cols).fit(trm[cols])
        rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                   min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                   random_state=42).fit(pre.transform(trm[cols]),
                                                        trm[TARGET].to_numpy(float))
        p = rf.predict(pre.transform(tem[cols]))
        m = eval_all(tem[TARGET].to_numpy(float), p, tem)
        results[name] = m
        log(f"  {name:24s} full={m['full']:.4f} green={m['green']:.4f} cdg={m['cdg']:.4f} "
            f"wet={m['wet']:.4f} sc={m['sc']:.4f} vsc={m['vsc']:.4f} "
            f"jeddah4={m['jeddah4']:.4f} silverstone={m['silverstone']:.4f} ({time.time()-t:.0f}s)")

    cols_show = ["full", "green", "cdg", "wet", "sc", "vsc", "jeddah4", "silverstone", "silverstone_wet"]
    print("\n" + "=" * 100)
    print(f"{'config':26s}" + "".join(f"{c:>11s}" for c in cols_show))
    print("-" * 100)
    for name, m in results.items():
        print(f"{name:26s}" + "".join(f"{m[c]:11.4f}" for c in cols_show))
    print("=" * 100)

    with open(ROOT / "fe_v2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log(f"saved -> fe_v2_results.json  total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
