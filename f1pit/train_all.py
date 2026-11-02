"""Full model comparison on the honest temporal split (train 2023+2024 -> test 2025).

- Features: ทุก feature + pace anchor (Prev_Duration / Prev_Sector_Duration /
  Clean_Pace / Best_Duration_So_Far) — ตัดเฉพาะ Num_Corner_* ที่ index เปลี่ยนตามปี
- Models: LinearRegression, Ridge, Lasso, ElasticNet (linear, standardized),
          ExtraTrees, RandomForest, GradientBoosting, HistGradientBoosting, Stacking
- ใช้ build_preprocessor ของ train.py (impute + one-hot Track_Status)
- Output: ตาราง MAE_s / RMSE_s / R2(log) จัดอันดับ; บันทึก results -> train_all_results.json

Run:  python -m f1pit.train_all
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from f1pit.train import TARGET, build_estimator, build_pipeline, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab

ROOT = Path(__file__).resolve().parent.parent


def _features(all_features, df_ml):
    # pace anchor ใส่ไว้; ตัดเฉพาะ Num_Corner_* (cluster index เปลี่ยนตามปี)
    feats = [c for c in all_features if not c.startswith("Num_Corner_")]
    if "Lap" in df_ml.columns and "Lap" not in feats:
        feats.append("Lap")
    return feats


def _models(feats):
    pre = build_preprocessor(feats)
    lin = lambda est: make_pipeline(pre, StandardScaler(), est)
    tree = lambda est: make_pipeline(pre, est)
    return [
        ("LinearRegression", lin(LinearRegression())),
        ("Ridge", lin(Ridge(alpha=1.0))),
        ("Lasso", lin(Lasso(alpha=0.0005, max_iter=10000))),
        ("ElasticNet", lin(ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=10000))),
        ("ExtraTrees", tree(build_estimator("ExtraTrees"))),
        ("RandomForest", tree(build_estimator("RandomForest"))),
        ("GradientBoosting", tree(build_estimator("GradientBoosting"))),
        ("HistGradientBoosting", tree(build_estimator("HistGradientBoosting"))),
        ("Stacking Ensemble", build_pipeline("Stacking Ensemble", feats)),
    ]


def _sec_metrics(y_log, pred_log):
    a, p = np.exp(y_log), np.exp(pred_log)
    return {
        "MAE_s": float(mean_absolute_error(a, p)),
        "RMSE_s": float(np.sqrt(mean_squared_error(a, p))),
        "R2_log": float(r2_score(y_log, pred_log)),
    }


def main():
    train_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                           pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    test_raw = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(train_raw, [])
    te_ml, _, _, _ = _prepare_lab(test_raw, [])
    feats = _features(tr_feats, tr_ml)
    # ใช้คอลัมน์ feature ที่มีร่วมกันทั้ง train/test (stable)
    feats = [c for c in feats if c in te_ml.columns]
    ytr = tr_ml[TARGET].to_numpy(float)
    yte = te_ml[TARGET].to_numpy(float)

    print(f"train rows={len(tr_ml):,}  test rows={len(te_ml):,}  features={len(feats)}")

    rows = []
    print(f"\n{'model':22s} {'train MAE_s':>11s} {'test MAE_s':>11s} {'test RMSE_s':>11s} "
          f"{'test R2':>8s} {'sec':>6s}")
    for name, pipe in _models(feats):
        t0 = time.time()
        pipe.fit(tr_ml[feats], ytr)
        fit_s = time.time() - t0
        tr_metrics = _sec_metrics(ytr, np.asarray(pipe.predict(tr_ml[feats]), float))
        te_metrics = _sec_metrics(yte, np.asarray(pipe.predict(te_ml[feats]), float))
        rows.append({"model": name, **te_metrics, "train_MAE_s": tr_metrics["MAE_s"],
                     "train_R2_log": tr_metrics["R2_log"], "fit_sec": round(fit_s, 1)})
        print(f"{name:22s} {tr_metrics['MAE_s']:11.4f} {te_metrics['MAE_s']:11.4f} "
              f"{te_metrics['RMSE_s']:11.4f} {te_metrics['R2_log']:8.4f} {fit_s:6.1f}")

    ranked = sorted(rows, key=lambda r: r["MAE_s"])
    print("\n=== RANKED (by test MAE_s) ===")
    print(f"{'#':>2s} {'model':22s} {'test MAE_s':>11s} {'test RMSE_s':>11s} "
          f"{'test R2':>8s} {'train MAE_s':>11s} {'train R2':>8s}")
    for i, r in enumerate(ranked, 1):
        print(f"{i:>2d} {r['model']:22s} {r['MAE_s']:11.4f} {r['RMSE_s']:11.4f} "
              f"{r['R2_log']:8.4f} {r['train_MAE_s']:11.4f} {r['train_R2_log']:8.4f}")

    out_path = ROOT / "train_all_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"features": feats, "results": ranked,
                   "train_rows": len(tr_ml), "test_rows": len(te_ml)}, f, indent=2)
    print(f"\nSaved -> {out_path.name}")
    print("DONE")


if __name__ == "__main__":
    main()
