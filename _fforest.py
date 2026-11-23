"""Forest-of-forests v3: tyre-aware telemetry (stage 1) -> sector Duration (stage 2).

Pipeline
--------
Data prep — add_pace_context (f1pit/pace_context.py): point-in-time lap context built from earlier laps
only: representative-lap anchors (Rep_*: last GREEN lap not slower than REP_TOL x recent green median),
previous-lap SC/VSC, laps since SC, rain history, stint context. Track_Status must come from the
red-flag-aware parser (rebuild, or run _repair_status.py on existing parquet).

Stage 1 — 8 telemetry forests. Each channel predicts its current value from its previous value +
tyre (age x compound interactions) + weather + traffic + Lap +
Track_Status + year-invariant geometry (sector/straight lengths; the per-year Num_Corner_* cluster ids
are NOT comparable across years and are not used). The variant (level / delta / delta_slog / log) is
chosen on 2023 -> held-out 2024 and gated against persistence; the winner is then 5-fold cross-fitted
(GroupKFold by session) on 2023+2024 -> out-of-fold Pred_* for the stage-2 training rows, and a full
refit on 2023+2024 predicts 2025.

Stage 2 — main forest, trained on 2023+2024, tested on 2025. Target = delta vs the representative
anchor: Log_Duration - log(Rep_Prev_Duration); Duration = Rep_Prev_Duration x exp(delta). Telemetry
enters ONLY as the tyre-aware Pred_* (raw Prev_* telemetry and the Prev_I1/I2/St speed traps are
excluded, so the tyre effect on lap time flows through the sub-forests and stays explainable).
Duration anchors are fed as log-ratios to the anchor (Rel_*, scale-free across circuits).

Run:  python _fforest.py
"""

import json
import os
import time
from pathlib import Path

import joblib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from f1pit.train import TARGET, TYRE_AGE_INTERACTION, TYRE_AGE_SQUARED, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.features import TELEMETRY_HISTORY, repair_track_status
from f1pit.pace_context import (CONTEXT_COLUMNS, PACE_CONTEXT_COLUMNS, RAIN_COLUMNS, REP_COLUMNS,
                                REP_TOL, add_pace_context, add_relative_pace)

ROOT = Path(__file__).resolve().parent
N_EST = 100
MAX_FEATURES = 0.5
MIN_SAMPLES_LEAF = 10
CORES = max(2, (os.cpu_count() or 8) // 2)
VERSION = "v3"
N_FOLDS = 5
MODEL_DIR = ROOT / "model" / VERSION

CHANNELS = list(TELEMETRY_HISTORY.items())   # 8 (prev, out) pairs, no Duration

TYRE = ["Tyre_Age_Total", *TYRE_AGE_INTERACTION, *TYRE_AGE_SQUARED]
WEATHER = ["Track_Temperature", "Air_Temperature", "Rainfall", "Pressure", "Humidity"]
TRAFFIC = ["Traffic_Within_1s", "Traffic_Within_2s", "Traffic_Within_5s", "Interval_Ahead"]

# stage-1 extra tyre data. Tested ["Tyre", "Distance_On_Tyre", "Total_Distance_Driven", "Stint_Number",
# "Tyre_Age_x_Track_Temp"]: no gain (OOF telemetry MAE slightly worse on 8/8 channels, stage-2 MAE equal),
# so stage 1 keeps the compound-interacted tyre age (TYRE) only.
S1_TYRE_EXTRA = []

# per-channel geometry (physical driver of each channel), year-invariant only
_SECTOR = ["Total_Sector_Length", "Straight_Ratio", "Log_Total_Sector_Length"]
GEOMETRY = {
    "Outcome_Avg_Speed_Corners": [*_SECTOR],
    "Outcome_Avg_Speed_Straight": ["Total_Straight_Length", "Straight_Ratio", "Log_Total_Straight_Length",
                                   "Total_Sector_Length", "Log_Total_Sector_Length"],
    "Outcome_Full_Throttle_Pct": ["Total_Straight_Length", *_SECTOR],
    "Outcome_Brake_Pct": [*_SECTOR],
    "Outcome_DRS_Pct": ["Total_Straight_Length", *_SECTOR],
    "Outcome_Max_Speed": ["Total_Straight_Length", "Log_Total_Straight_Length", *_SECTOR],
    "Outcome_Min_Speed": [*_SECTOR],
    "Outcome_Avg_Gear": ["Total_Straight_Length", *_SECTOR],
}

# stage-2 target anchor + non-telemetry features (raw names; duration anchors -> Rel_* at fit time)
ANCHOR = "Rep_Prev_Duration"
BASE2 = ["Prev_Duration", "Prev_Sector_Duration", "Clean_Pace", "Best_Duration_So_Far",
         "Tyre", *WEATHER, "Track_Status", "Track_Status_Inferred", *TRAFFIC,
         "Total_Sector_Length", "Total_Straight_Length", "Total_Distance_Driven", "Distance_On_Tyre",
         "Prev_Duration_roll5_mean", "Prev_Duration_roll5_slope",
         *REP_COLUMNS, *CONTEXT_COLUMNS, *RAIN_COLUMNS]

# channels whose target is positive-skewed -> log-target variant is meaningful
LOGGABLE = {"Outcome_Avg_Speed_Corners", "Outcome_Avg_Speed_Straight",
            "Outcome_Max_Speed", "Outcome_Min_Speed"}


def _mae(y, p):
    return float(mean_absolute_error(y, p))


def _rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def _r2(y, p):
    return float(r2_score(y, p))


def _mae_s(y_log, p_log):
    return _mae(np.exp(y_log), np.exp(p_log))


def _eval_breakdown(y_log, p_log, df):
    """6-condition MAE (seconds) breakdown on a test frame.

    conditions: overall / clean (no SC·VSC·event = GREEN only) / dry / wet /
    dry+clean / wet+clean.
    """
    y_sec = np.exp(np.asarray(y_log, float))
    p_sec = np.exp(np.asarray(p_log, float))
    status = df["Track_Status"].astype(str).str.upper()
    rain = pd.to_numeric(df["Rainfall"], errors="coerce").fillna(0).to_numpy(float)
    clean = (status == "GREEN").to_numpy()
    wet = rain > 0.0
    dry = ~wet
    masks = {
        "overall": np.ones(len(y_sec), dtype=bool),
        "clean_no_sc_vsc": clean,
        "dry": dry,
        "wet": wet,
        "dry_clean": dry & clean,
        "wet_clean": wet & clean,
    }
    out = {}
    for name, m in masks.items():
        n = int(m.sum())
        out[name] = {"mae_s": round(float(mean_absolute_error(y_sec[m], p_sec[m])), 4) if n else None,
                     "n": n}
    return out


def _slog(x):
    """signed log: sign(x) * log1p(|x|) — compresses heavy-tailed signed deltas."""
    return np.sign(x) * np.log1p(np.abs(x))


def _slog_inv(y):
    return np.sign(y) * np.expm1(np.abs(y))


def _target(name, y, prev):
    """variant name -> target array."""
    if name == "level":
        return y
    if name == "delta":
        return y - prev
    if name == "delta_slog":
        return _slog(y - prev)
    return np.log(np.clip(y, 1e-6, None))  # log


def _to_level(name, pred, prev):
    """variant-space prediction -> level prediction."""
    if name == "level":
        return pred
    if name == "delta":
        return prev + pred
    if name == "delta_slog":
        # clip before expm1: a linear (SGD) extrapolation past ~±10 would explode in exp space
        return prev + _slog_inv(np.clip(pred, -10.0, 10.0))
    return np.exp(pred)  # log


def _engineer(df):
    """Add engineered geometry features (Straight_Ratio, log lengths)."""
    df = df.copy()
    sl = pd.to_numeric(df.get("Total_Sector_Length"), errors="coerce")
    st = pd.to_numeric(df.get("Total_Straight_Length"), errors="coerce")
    df["Straight_Ratio"] = st / sl.replace(0, np.nan)
    for c in ["Total_Straight_Length", "Total_Sector_Length"]:
        if c in df.columns:
            df[f"Log_{c}"] = np.log(pd.to_numeric(df[c], errors="coerce").clip(lower=1.0))
    return df


def _fit_rf(Xtr, ytr, Xva, Xte):
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(Xtr, ytr)
    return rf, rf.predict(Xva), rf.predict(Xte)


def _fit_sgd(Xtr, ytr, Xva, Xte):
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(np.asarray(ytr, float).reshape(-1, 1))
    sgd = SGDRegressor(loss="huber", penalty="l2", alpha=1e-4,
                       learning_rate="invscaling", eta0=0.01, max_iter=1000, tol=1e-3,
                       random_state=42).fit(xs.transform(Xtr),
                                            ys.transform(np.asarray(ytr, float).reshape(-1, 1)).ravel())

    def _inv(p):
        return ys.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()
    return sgd, _inv(sgd.predict(xs.transform(Xva))), _inv(sgd.predict(xs.transform(Xte)))


def _fit_knn(Xtr, ytr, Xva, Xte):
    xs = StandardScaler().fit(Xtr)
    knn = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=CORES).fit(xs.transform(Xtr), ytr)
    return knn, knn.predict(xs.transform(Xva)), knn.predict(xs.transform(Xte))


def _fit_xgb(Xtr, ytr, Xva, Xte):
    import xgboost as xgb
    m = xgb.XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.03, min_child_weight=3,
                         subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=3.0,
                         objective="reg:squarederror", tree_method="hist",
                         random_state=42, n_jobs=CORES, verbosity=0)
    m.fit(Xtr, ytr)
    return m, m.predict(Xva), m.predict(Xte)


def _fit_cat(Xtr, ytr, Xva, Xte):
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(iterations=1000, depth=7, learning_rate=0.035, loss_function="RMSE",
                          l2_leaf_reg=5, random_seed=42, thread_count=-1, verbose=False,
                          allow_writing_files=False)
    m.fit(Xtr, ytr)
    return m, m.predict(Xva), m.predict(Xte)


def _clip_mad(t, k=6.0):
    """MAD-based robust clip of a (train) target; returns (clipped finite subset, finite mask)."""
    m = np.isfinite(t)
    med = np.median(t[m])
    mad = np.median(np.abs(t[m] - med))
    return (np.clip(t[m], med - k * mad, med + k * mad) if mad > 0 else t[m]), m


def _fit_sub(tr, prev, out, variant, feats):
    """fit one stage-1 RF (variant target, MAD clip) on `tr` -> (pre, rf)."""
    t = _target(variant, tr[out].to_numpy(float), tr[prev].to_numpy(float))
    t_fit, m = _clip_mad(t)
    pre = build_preprocessor(feats).fit(tr[feats][m])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr[feats][m]), t_fit)
    return pre, rf


def _sub_predict(pre, rf, df, prev, variant, feats):
    return _to_level(variant, rf.predict(pre.transform(df[feats])), df[prev].to_numpy(float))


def _reconstruct_outcomes(raw):
    """Outcome_X(t) = Prev_X(t+1); missing when the next lap only carries telemetry forward."""
    raw = raw.sort_values(["Session_Key", "Driver", "Sector", "Lap"]).reset_index(drop=True)
    g = raw.groupby(["Session_Key", "Driver", "Sector"], sort=False)
    ok = (g["Lap"].shift(-1) == raw["Lap"] + 1).to_numpy()
    nxt = {p: g[p].shift(-1) for p, _ in CHANNELS}
    carried = np.ones(len(raw), bool)
    for p, _ in CHANNELS:
        a, b = nxt[p].to_numpy(float), raw[p].to_numpy(float)
        carried &= (a == b) | (np.isnan(a) & np.isnan(b))
    for p, o in CHANNELS:
        raw[o] = nxt[p].where(ok & ~carried)
    return raw


def load_raw_year(year):
    """data_<year>_out.parquet if present; otherwise data_<year>.parquet with Outcome_* rebuilt from the
    next lap's Prev_* (validated in documentation.md §12.1: baseline RF 0.3200 vs 0.3224 on the real file)
    and Track_Status re-parsed with the red-flag-aware parser (1 cached race_control request/session)."""
    out_path = ROOT / f"data_{year}_out.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    raw = _reconstruct_outcomes(pd.read_parquet(ROOT / f"data_{year}.parquet"))
    raw, n_fix = repair_track_status(raw)
    print(f"  {year}: {out_path.name} not found -> Outcome_* rebuilt from data_{year}.parquet; "
          f"Track_Status repaired on {n_fix:,} rows", flush=True)
    return raw


def _load_year(year, extra):
    raw = add_pace_context(load_raw_year(year))
    df, _, _, _ = _prepare_lab(raw, [*extra, *PACE_CONTEXT_COLUMNS])
    return _engineer(df)


def main():
    t0 = time.time()
    extra_out = [out for _, out in CHANNELS]
    EXTRA = ["Duration", *extra_out]

    print("Loading rebuilt parquet (with Outcome_*) + pace context ...", flush=True)
    tr23, tr24, te25 = _load_year(2023, EXTRA), _load_year(2024, EXTRA), _load_year(2025, EXTRA)
    print(f"  rows: 2023={len(tr23):,} 2024={len(tr24):,} 2025={len(te25):,}", flush=True)
    for name, d in (("2023", tr23), ("2024", tr24), ("2025", te25)):
        sc = d["Track_Status"].astype(str).isin(["SC", "VSC"]).mean()
        if sc > 0.05:
            print(f"  ! {name}: {sc:.1%} of rows are SC/VSC - run _repair_status.py (red-flag parser fix)",
                  flush=True)

    common = set(tr23.columns) & set(tr24.columns) & set(te25.columns)

    summary = {"stage1": {}, "stage2": {}}
    sub_models = {}  # out -> full-refit model (pre, rf, feats, variant, prev) for saving
    both = pd.concat([tr23, tr24], ignore_index=True)   # stage-1 cross-fit pool (= stage-2 train rows)
    n23 = len(tr23)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(both, groups=both["Session_Key"]))

    print(f"\n{'=' * 80}\nSTAGE 1: 8 telemetry forests (variant select 2023->2024, gate, "
          f"{N_FOLDS}-fold cross-fit 2023+2024)\n{'=' * 80}", flush=True)

    for prev, out in CHANNELS:
        if out not in common or prev not in common:
            print(f"  ! skip {out}: missing column", flush=True)
            continue

        geom = [c for c in GEOMETRY[out] if c in common]
        trend = [c for c in [f"{prev}_roll5_mean", f"{prev}_roll5_slope"] if c in common]
        cols_level = [c for c in dict.fromkeys([prev, *TYRE, *S1_TYRE_EXTRA, *WEATHER, "Track_Status",
                                                *TRAFFIC, "Lap", *geom, *trend])
                      if c in common and c != out]
        cols_delta = [c for c in cols_level if c != prev]

        y24 = tr24[out].to_numpy(float); p24 = tr24[prev].to_numpy(float)
        pmask = np.isfinite(y24) & np.isfinite(p24)
        pers = _mae(y24[pmask], p24[pmask])

        variants = [("level", cols_level), ("delta", cols_delta), ("delta_slog", cols_delta)]
        if out in LOGGABLE:
            variants.append(("log", cols_level))

        # ---- variant selection + gate: fit 2023, score held-out 2024 ----
        best = None
        for name, feats in variants:
            if not np.isfinite(_target(name, tr23[out].to_numpy(float), tr23[prev].to_numpy(float))).any():
                continue
            pre, rf = _fit_sub(tr23, prev, out, name, feats)
            p24_lvl = _sub_predict(pre, rf, tr24, prev, name, feats)
            m24 = np.isfinite(_target(name, y24, p24))
            mae24 = _mae(y24[m24], p24_lvl[m24])
            if best is None or mae24 < best["mae"]:
                best = {"name": name, "mae": mae24, "p24": p24_lvl, "rf": rf, "pre": pre, "feats": feats,
                        "names": list(pre.get_feature_names_out()), "m24": m24}

        if best is None:
            print(f"  ! {out}: no usable variant", flush=True)
            continue
        name, feats, m24 = best["name"], best["feats"], best["m24"]
        beat = best["mae"] < pers

        if beat:
            # ---- leakage-safe Pred_*: out-of-fold for 2023+2024 (stage-2 train), full refit -> 2025 ----
            oof = np.full(len(both), np.nan)
            for tr_i, va_i in folds:
                pre_f, rf_f = _fit_sub(both.iloc[tr_i], prev, out, name, feats)
                oof[va_i] = _sub_predict(pre_f, rf_f, both.iloc[va_i], prev, name, feats)
            pre_full, rf_full = _fit_sub(both, prev, out, name, feats)
            tr23[f"Pred_{out}"] = oof[:n23]
            tr24[f"Pred_{out}"] = oof[n23:]
            te25[f"Pred_{out}"] = _sub_predict(pre_full, rf_full, te25, prev, name, feats)
            sub_models[out] = {"pre": pre_full, "rf": rf_full, "feats": feats, "variant": name, "prev": prev}
            yb = both[out].to_numpy(float)
            mo = np.isfinite(yb) & np.isfinite(oof)
            oof_mae = _mae(yb[mo], oof[mo])
        else:
            oof_mae = None   # channel dropped from stage 2 (no raw Prev_* fallback)

        # SGD on the winning variant (same feats/target, 2023 -> 2024) for comparison
        t23_fit, m23 = _clip_mad(_target(name, tr23[out].to_numpy(float), tr23[prev].to_numpy(float)))
        pre = build_preprocessor(feats).fit(tr23[feats][m23])
        _, ps24, _ = _fit_sgd(pre.transform(tr23[feats][m23]), t23_fit,
                              pre.transform(tr24[feats]), pre.transform(tr24[feats])[:1])
        ps24_lvl = _to_level(name, ps24, p24)

        imp = pd.Series(best["rf"].feature_importances_, index=best["names"]).sort_values(ascending=False)
        tyre_imp = {n: round(float(v), 5) for n, v in imp.items()
                    if n.startswith(("num__Tyre_Age_", "cat__Tyre_", "num__Distance_On_Tyre", "num__Stint_",
                                     "num__Is_First_Stint"))}
        summary["stage1"][out] = {
            "persistence_mae": round(pers, 4),
            "best_variant": name,
            "beat_persistence": bool(beat),
            "rf": {"mae": round(best["mae"], 4),
                   "rmse": round(_rmse(y24[m24], best["p24"][m24]), 4),
                   "r2": round(_r2(y24[m24], best["p24"][m24]), 4)},
            "rf_oof_2023_2024_mae": round(oof_mae, 4) if oof_mae is not None else None,
            "sgd": {"mae": round(_mae(y24[m24], ps24_lvl[m24]), 4),
                    "rmse": round(_rmse(y24[m24], ps24_lvl[m24]), 4),
                    "r2": round(_r2(y24[m24], ps24_lvl[m24]), 4)},
            "tyre_importance_sum": round(sum(tyre_imp.values()), 5),
            "importance": [[str(n), round(float(v), 6)] for n, v in imp.items()],
        }
        print(f"\n[{out}]  prev={prev}  variant={name}  beat_persistence={beat}", flush=True)
        print(f"  persistence={pers:.4f}  RF={best['mae']:.4f} (R2={_r2(y24[m24], best['p24'][m24]):.4f})"
              f"  SGD={_mae(y24[m24], ps24_lvl[m24]):.4f}"
              + (f"  OOF(2023+2024)={oof_mae:.4f}" if oof_mae is not None else ""), flush=True)
        print(f"  tyre importance={sum(tyre_imp.values()):.4f}  top6:", flush=True)
        for n, v in imp.head(6).items():
            print(f"     {n:42s} {v:.5f}", flush=True)

    # ============================================================
    # Stage 2 - main forest: delta vs the representative-lap anchor
    #   target = Log_Duration - log(Rep_Prev_Duration);  Duration = Rep_Prev_Duration * exp(delta)
    #   telemetry = tyre-aware Pred_* only; duration anchors as log-ratio to the anchor (Rel_*)
    # ============================================================
    print(f"\n{'=' * 80}\nSTAGE 2: main forest -> delta vs {ANCHOR} (Pred_* telemetry, RF + SGD + KNN"
          f" [+ XGB + CatBoost if installed])\n{'=' * 80}", flush=True)

    tr2 = pd.concat([tr23, tr24], ignore_index=True)
    tr2, rel_map = add_relative_pace(tr2, ANCHOR)
    te25, _ = add_relative_pace(te25, ANCHOR)

    telemetry_pred = [f"Pred_{out}" for _, out in CHANNELS if f"Pred_{out}" in tr2.columns]
    dropped = [out for _, out in CHANNELS if f"Pred_{out}" not in tr2.columns]
    if dropped:
        print(f"  ! channels without Pred_* (did not beat persistence) -> dropped: {dropped}", flush=True)
    raw_feats = [*telemetry_pred, *BASE2]
    feats = [c for c in dict.fromkeys(rel_map.get(c, c) for c in raw_feats)
             if c in tr2.columns and c in te25.columns]

    log_a_tr = np.log(tr2[ANCHOR].to_numpy(float))
    log_a_te = np.log(te25[ANCHOR].to_numpy(float))
    ytr = tr2[TARGET].to_numpy(float) - log_a_tr
    ok = np.isfinite(ytr)
    tr2, ytr = tr2[ok].reset_index(drop=True), ytr[ok]
    yte_log = te25[TARGET].to_numpy(float)
    actual_sec = np.exp(yte_log)
    print(f"  train rows (2023+2024) = {len(tr2):,}   test rows (2025) = {len(te25):,}   features = {len(feats)}",
          flush=True)

    stage2 = {}
    pre2 = build_preprocessor(feats).fit(tr2[feats])
    Xtr = pre2.transform(tr2[feats]); Xte = pre2.transform(te25[feats])
    fits = {"rf": _fit_rf, "sgd": _fit_sgd, "knn": _fit_knn, "xgb": _fit_xgb, "cat": _fit_cat}
    preds, rf = {}, None
    for key, fit in fits.items():
        try:
            mdl, _, p = fit(Xtr, ytr, Xte[:1], Xte)
        except ImportError as exc:
            print(f"  ({key} skipped: {exc})", flush=True)
            continue
        preds[key] = p
        if key == "rf":
            rf = mdl
    res = {}
    for key, p in preds.items():
        sec = np.exp(log_a_te + p)
        res[f"{key}_mae_s"] = round(_mae(actual_sec, sec), 4)
        res[f"{key}_r2"] = round(_r2(actual_sec, sec), 4)
    imp = pd.Series(rf.feature_importances_, index=list(pre2.get_feature_names_out())).sort_values(ascending=False)
    res["importance"] = [[str(n), round(float(v), 6)] for n, v in imp.items()]
    res["pred_importance_sum"] = round(float(imp[[n for n in imp.index if "Pred_" in n]].sum()), 4)
    res["breakdown"] = _eval_breakdown(yte_log, log_a_te + preds["rf"], te25)
    res["anchor"] = ANCHOR
    stage2["pred"] = res
    print("  " + "  ".join(f"{k.upper()}={res[f'{k}_mae_s']:.4f} (R2={res[f'{k}_r2']:.4f})" for k in preds),
          flush=True)
    print(f"  Pred_* importance share (RF) = {res['pred_importance_sum']:.3f}", flush=True)
    main_model = {"pre": pre2, "rf": rf, "feats": feats}

    summary["stage2"] = stage2
    print(f"\n  === 6-condition MAE (pred variant, RF) ===", flush=True)
    for name, v in stage2["pred"]["breakdown"].items():
        mae = v["mae_s"] if v["mae_s"] is not None else "-"
        print(f"    {name:18s} n={v['n']:>7,}  MAE_s={mae}", flush=True)

    # ---- degradation curve (delta vs Tyre_Age_Total, from the main forest) ----
    # NOTE: the main forest has no Tyre_Age_Total feature (tyre enters via stage-1 Pred_*), so this
    # finds no column and is skipped; a curve would need to vary tyre age through stage 1.
    print(f"\n{'=' * 80}\nDEGRADATION CURVE (delta vs Tyre_Age_Total)\n{'=' * 80}", flush=True)
    try:
        mm = main_model
        names = list(mm["pre"].get_feature_names_out())
        idx = [i for i, n in enumerate(names) if n == "num__Tyre_Age_Total"]
        if idx:
            Xs = mm["pre"].transform(tr2[mm["feats"]])[:5000]
            pd_res = partial_dependence(mm["rf"], Xs, features=idx, kind="average", grid_resolution=20)
            grid = [float(x) for x in pd_res["grid_values"][0]]
            curve = [float(x) for x in pd_res["average"][0]]
            summary["degradation"] = {"feature": "Tyre_Age_Total", "grid": grid, "delta_log": curve}
            print("  Tyre_Age_Total -> mean predicted delta (log ratio):", flush=True)
            for g, c in zip(grid, curve):
                print(f"     age={g:6.1f}  delta={c:.4f}", flush=True)
        else:
            print("  skipped: Tyre_Age_Total is not a main-forest feature", flush=True)
    except Exception as exc:
        print(f"  degradation curve skipped: {exc}", flush=True)

    # ---- save models for reuse (no retrain) ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for out, b in sub_models.items():
        joblib.dump(b, MODEL_DIR / f"sub_{out}.pkl", compress=3)
    joblib.dump(main_model, MODEL_DIR / "main.pkl", compress=3)
    meta = {
        "version": VERSION,
        "channels": [[p, o] for p, o in CHANNELS if o in sub_models],
        "main_feats": main_model["feats"],
        "anchor": ANCHOR,
        "rep_tol": REP_TOL,
        "extra": ["Duration", *[o for _, o in CHANNELS], *PACE_CONTEXT_COLUMNS],
    }
    with open(MODEL_DIR / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"Saved {len(sub_models)} sub-forests + main forest -> {MODEL_DIR}", flush=True)

    # ---- save ----
    out_path = ROOT / "fforest_results.json"
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nSaved -> {out_path.name}  ({time.time() - t0:.0f}s)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
