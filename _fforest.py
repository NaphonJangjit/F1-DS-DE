"""Forest-of-forests v2: per-channel feature engineering, sub-forest optimization, then Duration.

Pipeline
--------
Stage 1 — 8 telemetry forests. Each channel predicts its current value from its previous value +
tyre + weather + traffic + Lap + Track_Status + the geometry that physically drives that channel
(corner profile Num_Corner_* for corner channels, straight length for straight channels). Feature
engineering: level vs delta vs log target, Straight_Ratio, log lengths, roll5 trends. The winning
variant per channel is chosen by held-out-2024 MAE, and its RF prediction is used as Pred_<X> only
if it beats the persistence baseline (predict current = previous).

Stage 2 — main forest whose only target is Duration (Log_Duration): baseline (raw Prev_*) vs
stacked (Pred_* for the channels that beat persistence, raw Prev_* otherwise).

Leakage-safe: stage-1 trained on 2023 only -> predicts 2024 (stage-2 train) and 2025 (stage-2 test).

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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from f1pit.train import TARGET, TYRE_AGE_INTERACTION, TYRE_AGE_SQUARED, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.features import TELEMETRY_HISTORY

ROOT = Path(__file__).resolve().parent
N_EST = 100
MAX_FEATURES = 0.5
MIN_SAMPLES_LEAF = 10
CORES = max(2, (os.cpu_count() or 8) // 2)
VERSION = "v2"
MODEL_DIR = ROOT / "model" / VERSION

CHANNELS = list(TELEMETRY_HISTORY.items())   # 8 (prev, out) pairs, no Duration

TYRE = ["Tyre_Age_Total", *TYRE_AGE_INTERACTION, *TYRE_AGE_SQUARED]
WEATHER = ["Track_Temperature", "Air_Temperature", "Rainfall", "Pressure", "Humidity"]
TRAFFIC = ["Traffic_Within_1s", "Traffic_Within_2s", "Traffic_Within_5s", "Interval_Ahead"]

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


def main():
    t0 = time.time()
    extra_out = [out for _, out in CHANNELS]
    EXTRA = ["Duration", *extra_out]

    print("Loading rebuilt parquet (with Outcome_*) ...", flush=True)
    raw23 = pd.read_parquet(ROOT / "data_2023_out.parquet")
    raw24 = pd.read_parquet(ROOT / "data_2024_out.parquet")
    raw25 = pd.read_parquet(ROOT / "data_2025_out.parquet")

    print("Preparing + engineering features ...", flush=True)
    tr23, _, _, _ = _prepare_lab(raw23, EXTRA)
    tr24, _, _, _ = _prepare_lab(raw24, EXTRA)
    te25, _, _, _ = _prepare_lab(raw25, EXTRA)
    tr23, tr24, te25 = _engineer(tr23), _engineer(tr24), _engineer(te25)
    print(f"  rows: 2023={len(tr23):,} 2024={len(tr24):,} 2025={len(te25):,}", flush=True)

    common = set(tr23.columns) & set(tr24.columns) & set(te25.columns)
    num_corner = sorted([c for c in common if c.startswith("Num_Corner_")])
    print(f"  Num_Corner_* columns = {len(num_corner)}", flush=True)

    # per-channel geometry (the physical driver of each channel)
    GEOMETRY = {
        "Outcome_Avg_Speed_Corners": [*num_corner],
        "Outcome_Avg_Speed_Straight": ["Total_Straight_Length", "Straight_Ratio", "Log_Total_Straight_Length"],
        "Outcome_Full_Throttle_Pct": ["Total_Straight_Length", "Total_Sector_Length", "Straight_Ratio", *num_corner],
        "Outcome_Brake_Pct": [*num_corner],
        "Outcome_DRS_Pct": ["Total_Straight_Length", "Straight_Ratio"],
        "Outcome_Max_Speed": ["Total_Straight_Length", "Log_Total_Straight_Length"],
        "Outcome_Min_Speed": [*num_corner],
        "Outcome_Avg_Gear": ["Total_Straight_Length", "Straight_Ratio", *num_corner],
    }

    summary = {"stage1": {}, "stage2": {}}
    pred_cols = {}   # prev -> "Pred_<out>" if it beats persistence, else prev (raw)
    sub_models = {}  # out -> winning model (pre, rf, feats, variant, prev) for saving

    print(f"\n{'=' * 80}\nSTAGE 1: 8 telemetry forests (feature engineering + gate)\n{'=' * 80}", flush=True)

    for prev, out in CHANNELS:
        if out not in common or prev not in common:
            print(f"  ! skip {out}: missing column", flush=True)
            continue

        geom = [c for c in GEOMETRY[out] if c in common]
        trend = [c for c in [f"{prev}_roll5_mean", f"{prev}_roll5_slope"] if c in common]
        cols_level = [c for c in [prev, *TYRE, *WEATHER, "Track_Status", *TRAFFIC, "Lap", *geom, *trend]
                      if c in common and c != out]
        cols_delta = [c for c in cols_level if c != prev]

        y23 = tr23[out].to_numpy(float); y24 = tr24[out].to_numpy(float); yte = te25[out].to_numpy(float)
        p23 = tr23[prev].to_numpy(float); p24 = tr24[prev].to_numpy(float); pte = te25[prev].to_numpy(float)

        pmask = np.isfinite(y24) & np.isfinite(p24)
        pers = _mae(y24[pmask], p24[pmask])

        variants = [
            ("level", cols_level),
            ("delta", cols_delta),
            ("delta_slog", cols_delta),
        ]
        if out in LOGGABLE:
            variants.append(("log", cols_level))

        best = None
        for name, feats in variants:
            t23 = _target(name, y23, p23)
            t24 = _target(name, y24, p24)
            m23 = np.isfinite(t23)
            m24 = np.isfinite(t24)
            if not m23.any() or not m24.any():
                continue
            # MAD-based robust clip on the train target (telemetry artifacts / heavy tails)
            med = np.median(t23[m23])
            mad = np.median(np.abs(t23[m23] - med))
            if mad > 0:
                lo, hi = med - 6.0 * mad, med + 6.0 * mad
                t23_fit = np.clip(t23[m23], lo, hi)
            else:
                t23_fit = t23[m23]

            pre = build_preprocessor(feats).fit(tr23[feats][m23])
            X23 = pre.transform(tr23[feats][m23])
            X24 = pre.transform(tr24[feats])
            Xte = pre.transform(te25[feats])
            rf, pr24, prte = _fit_rf(X23, t23_fit, X24, Xte)

            p24_lvl = _to_level(name, pr24, p24)
            pte_lvl = _to_level(name, prte, pte)
            mae24 = _mae(y24[m24], p24_lvl[m24])
            if best is None or mae24 < best["mae"]:
                best = {"name": name, "mae": mae24, "p24": p24_lvl, "pte": pte_lvl,
                        "rf": rf, "names": list(pre.get_feature_names_out()), "feats": feats,
                        "pre": pre, "prev": prev}

        if best is None:
            print(f"  ! {out}: no usable variant", flush=True)
            continue
        sub_models[out] = best

        beat = best["mae"] < pers
        if beat:
            tr24[f"Pred_{out}"] = best["p24"]
            te25[f"Pred_{out}"] = best["pte"]
            pred_cols[prev] = f"Pred_{out}"
        else:
            pred_cols[prev] = prev   # keep raw prev: Pred_* would be worse than persistence

        # SGD on the winning variant (same feats/target) for comparison
        name = best["name"]; feats = best["feats"]
        t23 = _target(name, y23, p23); t24 = _target(name, y24, p24)
        m23 = np.isfinite(t23); m24 = np.isfinite(t24)
        med = np.median(t23[m23]); mad = np.median(np.abs(t23[m23] - med))
        if mad > 0:
            t23 = np.clip(t23, med - 6.0 * mad, med + 6.0 * mad)
        pre = build_preprocessor(feats).fit(tr23[feats][m23])
        X23 = pre.transform(tr23[feats][m23]); X24 = pre.transform(tr24[feats]); Xte = pre.transform(te25[feats])
        _, ps24, _ = _fit_sgd(X23, t23[m23], X24, Xte)
        ps24_lvl = _to_level(name, ps24, p24)

        imp = pd.Series(best["rf"].feature_importances_, index=best["names"]).sort_values(ascending=False)
        tyre_imp = {n: round(float(v), 5) for n, v in imp.items()
                    if n.startswith("num__Tyre_Age_") or n == "num__Tyre_Age_Total"}
        summary["stage1"][out] = {
            "persistence_mae": round(pers, 4),
            "best_variant": name,
            "beat_persistence": bool(beat),
            "rf": {"mae": round(best["mae"], 4),
                   "rmse": round(_rmse(y24[m24], best["p24"][m24]), 4),
                   "r2": round(_r2(y24[m24], best["p24"][m24]), 4)},
            "sgd": {"mae": round(_mae(y24[m24], ps24_lvl[m24]), 4),
                    "rmse": round(_rmse(y24[m24], ps24_lvl[m24]), 4),
                    "r2": round(_r2(y24[m24], ps24_lvl[m24]), 4)},
            "tyre_importance_sum": round(sum(tyre_imp.values()), 5),
            "importance": [[str(n), round(float(v), 6)] for n, v in imp.items()],
        }
        print(f"\n[{out}]  prev={prev}  variant={name}  beat_persistence={beat}", flush=True)
        print(f"  persistence={pers:.4f}  RF={best['mae']:.4f} (R2={_r2(y24[m24], best['p24'][m24]):.4f})"
              f"  SGD={_mae(y24[m24], ps24_lvl[m24]):.4f}", flush=True)
        print(f"  tyre importance={sum(tyre_imp.values()):.4f}  top6:", flush=True)
        for n, v in imp.head(6).items():
            print(f"     {n:42s} {v:.5f}", flush=True)

    # ============================================================
    # Stage 2 — main forest: Duration = f(Pred_* telemetry + duration pace anchors + static)
    #           telemetry Prev_* -> Pred_* (tyre-aware); duration anchors
    #           (Prev_Duration, Clean_Pace, Best_Duration_So_Far, ...) KEPT as-is.
    # ============================================================
    print(f"\n{'=' * 80}\nSTAGE 2: main forest -> delta (Prev_* vs Pred_*, RF + SGD + KNN)\n{'=' * 80}", flush=True)

    telemetry_prev = [prev for prev, _ in CHANNELS]
    telemetry_pred = [f"Pred_{out}" for _, out in CHANNELS]
    pace = ["Prev_Duration", "Prev_Sector_Duration", "Clean_Pace", "Best_Duration_So_Far",
            "Prev_I1_Speed", "Prev_I2_Speed", "Prev_St_Speed"]
    roll5 = [c for c in ["Prev_Duration_roll5_mean", "Prev_Duration_roll5_slope"]
             if c in tr24.columns and c in te25.columns]
    base2 = [*pace, "Tyre", *WEATHER, "Track_Status", "Track_Status_Inferred", *TRAFFIC,
             "Total_Sector_Length", "Total_Straight_Length", "Total_Distance_Driven", "Distance_On_Tyre",
             *num_corner, *roll5]

    yte_log = te25[TARGET].to_numpy(float)
    log_prev_tr = np.log(tr24["Prev_Duration"].to_numpy(float))
    log_prev_te = np.log(te25["Prev_Duration"].to_numpy(float))
    ytr = tr24[TARGET].to_numpy(float) - log_prev_tr
    yte = yte_log - log_prev_te
    actual_sec = np.exp(yte_log)

    variants = [
        ("prev", [c for c in [*telemetry_prev, *base2] if c in tr24.columns and c in te25.columns]),
        ("pred", [c for c in [*telemetry_pred, *base2] if c in tr24.columns and c in te25.columns]),
    ]

    stage2 = {}
    main_model = None  # save the "prev" variant for reuse
    for label, feats in variants:
        pre2 = build_preprocessor(feats).fit(tr24[feats])
        Xtr = pre2.transform(tr24[feats]); Xte = pre2.transform(te25[feats])
        rf, _, prf = _fit_rf(Xtr, ytr, Xte, Xte)
        _, _, psgd = _fit_sgd(Xtr, ytr, Xte, Xte)
        _, _, pknn = _fit_knn(Xtr, ytr, Xte, Xte)
        _, _, pxgb = _fit_xgb(Xtr, ytr, Xte, Xte)
        _, _, pcat = _fit_cat(Xtr, ytr, Xte, Xte)
        imp = pd.Series(rf.feature_importances_, index=list(pre2.get_feature_names_out())).sort_values(ascending=False)
        pred_rf_sec = np.exp(log_prev_te + prf)
        pred_sgd_sec = np.exp(log_prev_te + psgd)
        pred_knn_sec = np.exp(log_prev_te + pknn)
        pred_xgb_sec = np.exp(log_prev_te + pxgb)
        pred_cat_sec = np.exp(log_prev_te + pcat)
        m_rf = _mae(actual_sec, pred_rf_sec)
        m_sgd = _mae(actual_sec, pred_sgd_sec)
        m_knn = _mae(actual_sec, pred_knn_sec)
        m_xgb = _mae(actual_sec, pred_xgb_sec)
        m_cat = _mae(actual_sec, pred_cat_sec)
        r2_rf = _r2(actual_sec, pred_rf_sec)
        r2_sgd = _r2(actual_sec, pred_sgd_sec)
        r2_knn = _r2(actual_sec, pred_knn_sec)
        r2_xgb = _r2(actual_sec, pred_xgb_sec)
        r2_cat = _r2(actual_sec, pred_cat_sec)
        print(f"  [{label}] delta  RF={m_rf:.4f} (R2={r2_rf:.4f})  SGD={m_sgd:.4f}  KNN={m_knn:.4f}  "
              f"XGB={m_xgb:.4f}  CatBoost={m_cat:.4f}  (features={len(feats)})", flush=True)
        stage2[label] = {
            "rf_mae_s": round(m_rf, 4), "rf_r2": round(r2_rf, 4),
            "sgd_mae_s": round(m_sgd, 4), "sgd_r2": round(r2_sgd, 4),
            "knn_mae_s": round(m_knn, 4), "knn_r2": round(r2_knn, 4),
            "xgb_mae_s": round(m_xgb, 4), "xgb_r2": round(r2_xgb, 4),
            "cat_mae_s": round(m_cat, 4), "cat_r2": round(r2_cat, 4),
            "importance": [[str(n), round(float(v), 6)] for n, v in imp.items()],
            "breakdown": _eval_breakdown(yte_log, log_prev_te + prf, te25),
        }
        if label == "prev":
            main_model = {"pre": pre2, "rf": rf, "feats": feats}

    summary["stage2"] = stage2
    print(f"\n  === 6-condition MAE (prev variant, RF) ===", flush=True)
    for name, v in stage2["prev"]["breakdown"].items():
        mae = v["mae_s"] if v["mae_s"] is not None else "—"
        print(f"    {name:18s} n={v['n']:>7,}  MAE_s={mae}", flush=True)

    # ---- degradation curve (delta vs Tyre_Age_Total, from prev-variant main forest) ----
    print(f"\n{'=' * 80}\nDEGRADATION CURVE (delta vs Tyre_Age_Total)\n{'=' * 80}", flush=True)
    try:
        mm = main_model
        names = list(mm["pre"].get_feature_names_out())
        idx = [i for i, n in enumerate(names) if n == "num__Tyre_Age_Total"]
        if idx:
            Xs = mm["pre"].transform(tr24[mm["feats"]])[:5000]
            pd_res = partial_dependence(mm["rf"], Xs, features=idx, kind="average", grid_resolution=20)
            grid = [float(x) for x in pd_res["grid_values"][0]]
            curve = [float(x) for x in pd_res["average"][0]]
            summary["degradation"] = {"feature": "Tyre_Age_Total", "grid": grid, "delta_log": curve}
            print("  Tyre_Age_Total -> mean predicted delta (log ratio):", flush=True)
            for g, c in zip(grid, curve):
                print(f"     age={g:6.1f}  delta={c:.4f}", flush=True)
    except Exception as exc:
        print(f"  degradation curve skipped: {exc}", flush=True)

    # ---- save models for reuse (no retrain) ----
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for out, b in sub_models.items():
        joblib.dump({"pre": b["pre"], "rf": b["rf"], "feats": b["feats"],
                     "variant": b["name"], "prev": b["prev"]},
                    MODEL_DIR / f"sub_{out}.pkl", compress=3)
    joblib.dump({"pre": main_model["pre"], "rf": main_model["rf"], "feats": main_model["feats"]},
                MODEL_DIR / "main.pkl", compress=3)
    meta = {
        "version": VERSION,
        "channels": [[p, o] for p, o in CHANNELS],
        "main_feats": main_model["feats"],
        "extra": ["Duration", *[o for _, o in CHANNELS]],
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
