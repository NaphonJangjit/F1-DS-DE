"""Ridge-only on the temporal split: train 2023+2024 -> test 2025.

Explore feature sets + alpha tuning (log-duration alignment สำหรับ linear model).
Ridge เท่านั้น — ไม่ใช้ tree.

Run:  python -m f1pit.eval_lab_ridge_temporal
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab

ROOT = Path(__file__).resolve().parent.parent
PACE = ["Best_Duration_So_Far", "Clean_Pace", "Prev_Duration"]
DURATION_LIKE = ["Prev_Duration", "Prev_Sector_Duration", "Clean_Pace", "Best_Duration_So_Far"]

WEATHER = ["Track_Temperature", "Rainfall", "Pressure", "Humidity", "Air_Temperature"]
GEOM = ["Total_Sector_Length", "Total_Straight_Length", "Total_Distance_Driven", "Distance_On_Tyre"]
SPEED = ["Prev_Avg_Speed_Corners", "Prev_Avg_Speed_Straight", "Prev_Max_Speed", "Prev_Min_Speed",
         "Prev_Avg_Gear", "Prev_Full_Throttle_Pct", "Prev_Brake_Pct", "Prev_DRS_Pct",
         "Prev_I1_Speed", "Prev_I2_Speed", "Prev_St_Speed"]


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


def _ensure(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan


def fit_ridge(train_ml, test_ml, use, alpha=1.0):
    tr = train_ml[use].copy()
    te = test_ml[use].copy()
    _ensure(te, use)
    dl = [c for c in DURATION_LIKE if c in use]
    tr = _log_dur(tr, dl)
    te = _log_dur(te, dl)
    pipe = make_pipeline(build_preprocessor(use), StandardScaler(), Ridge(alpha=alpha))
    pipe.fit(tr, train_ml[TARGET])
    return _metrics(test_ml[TARGET].to_numpy(float), np.asarray(pipe.predict(te), float))


def main():
    train_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                           pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    test_raw = pd.read_parquet(ROOT / "data_2025.parquet")
    train_ml, feats_tr, _, _ = _prepare_lab(train_raw, [])
    test_ml, feats_te, _, _ = _prepare_lab(test_raw, [])
    stable = [c for c in feats_tr if c in feats_te and not c.startswith("Num_Corner_")]
    print(f"train rows={len(train_ml):,}  test rows={len(test_ml):,}  "
          f"year-stable features={len(stable)}")

    def show(label, use, alpha=1.0):
        mae, rmse, r2 = fit_ridge(train_ml, test_ml, use, alpha=alpha)
        print(f"  {label:34s} MAE_s={mae:.4f}  RMSE_s={rmse:.4f}  R2(log)={r2:.5f}")

    print("\n-- feature sets (Ridge, log-duration, alpha=1) --")
    show("pace only (3)", PACE)
    show("pace + geometry", PACE + GEOM)
    show("pace + weather", PACE + WEATHER)
    show("pace + speed", PACE + SPEED)
    show("pace + geom + weather", PACE + GEOM + WEATHER)
    show("all stable (40)", stable)

    print("\n-- alpha sweep on all-stable --")
    for a in (0.1, 1.0, 10.0, 100.0, 1000.0):
        show(f"all stable, alpha={a:<6}", stable, alpha=a)

    print("\n-- alpha sweep on pace + geometry --")
    for a in (0.1, 1.0, 10.0, 100.0):
        show(f"pace+geom, alpha={a:<6}", PACE + GEOM, alpha=a)

    print("\nDONE")


if __name__ == "__main__":
    main()
