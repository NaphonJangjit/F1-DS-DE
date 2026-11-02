"""Temporal split: train 2023+2024 -> test 2025 (leak-free ข้ามฤดูกาล)

Ridge (log-duration) + ExtraTrees บน (a) pace-only และ (b) all year-stable features
(ตัด Num_Corner_* ออก เพราะ cluster index เปลี่ยนตามปี — k ถูก auto-select ต่อปี)

Run:  python -m f1pit.eval_lab_temporal
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from f1pit import config
from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab

ROOT = Path(__file__).resolve().parent.parent
PACE = ["Best_Duration_So_Far", "Clean_Pace", "Prev_Duration"]
DURATION_LIKE = ["Prev_Duration", "Prev_Sector_Duration", "Clean_Pace", "Best_Duration_So_Far"]


def _metrics(y_log_true, y_log_pred):
    a = np.exp(np.asarray(y_log_true, float))
    p = np.exp(np.asarray(y_log_pred, float))
    return (mean_absolute_error(a, p), float(np.sqrt(mean_squared_error(a, p))),
            r2_score(y_log_true, y_log_pred))


def _log_dur(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = np.log(pd.to_numeric(df[c], errors="coerce").clip(lower=1e-6))
    return df


def _ensure_cols(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan


def eval_ridge(train_ml, test_ml, use, log_dur=True, alpha=1.0):
    tr = train_ml[use].copy()
    te = test_ml[use].copy()
    _ensure_cols(te, use)
    if log_dur:
        dl = [c for c in DURATION_LIKE if c in use]
        tr = _log_dur(tr, dl)
        te = _log_dur(te, dl)
    pipe = make_pipeline(build_preprocessor(use), StandardScaler(), Ridge(alpha=alpha))
    pipe.fit(tr, train_ml[TARGET])
    return _metrics(test_ml[TARGET].to_numpy(float), np.asarray(pipe.predict(te), float))


def eval_et(train_ml, test_ml, use):
    tr = train_ml[use].copy()
    te = test_ml[use].copy()
    _ensure_cols(te, use)
    est = ExtraTreesRegressor(n_estimators=300, max_features=0.9, min_samples_leaf=2,
                              n_jobs=-1, random_state=config.RANDOM_STATE)
    pipe = make_pipeline(build_preprocessor(use), est)
    pipe.fit(tr, train_ml[TARGET])
    return _metrics(test_ml[TARGET].to_numpy(float), np.asarray(pipe.predict(te), float))


def main():
    train_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                           pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    test_raw = pd.read_parquet(ROOT / "data_2025.parquet")
    print(f"train rows={len(train_raw):,} (2023+2024)  test rows={len(test_raw):,} (2025)")

    train_ml, feats_tr, _, _ = _prepare_lab(train_raw, [])
    test_ml, feats_te, _, _ = _prepare_lab(test_raw, [])
    print(f"train features={len(feats_tr)}  test features={len(feats_te)}")
    print(f"train rows={len(train_ml):,}  test rows={len(test_ml):,}\n")

    stable = [c for c in feats_tr if c in feats_te and not c.startswith("Num_Corner_")]
    print(f"year-stable features (excl Num_Corner_*) = {len(stable)}\n")

    for label, use, fn in [
        ("Ridge pace-only (3, log)", PACE, lambda t, e: eval_ridge(t, e, PACE)),
        ("Ridge all-stable (log)", stable, lambda t, e: eval_ridge(t, e, stable)),
        ("Ridge all-stable (raw)", stable, lambda t, e: eval_ridge(t, e, stable, log_dur=False)),
        ("ExtraTrees pace-only (3)", PACE, lambda t, e: eval_et(t, e, PACE)),
        ("ExtraTrees all-stable", stable, lambda t, e: eval_et(t, e, stable)),
    ]:
        mae, rmse, r2 = fn(train_ml, test_ml)
        print(f"  {label:28s} MAE_s={mae:.4f}  RMSE_s={rmse:.4f}  R2(log)={r2:.5f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
