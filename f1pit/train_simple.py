"""Train a simple, explainable linear model: Ridge on Log_Duration.

- Features: ทุก feature + pace anchor (Prev_Duration / Prev_Sector_Duration /
  Clean_Pace / Best_Duration_So_Far) — ตัดเฉพาะ Num_Corner_* ที่ index เปลี่ยนตามปี
- Train: 2023 + 2024   Test: 2025  (honest, ไม่ leak)
- Output: (1) accuracy, (2) ตาราง coefficient ทั้งหมด (explainability),
          (3) save model -> model_simple_2025.pkl

Run:  python -m f1pit.train_simple
"""

import pickle
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


def _features(all_features, df_ml):
    # ใส่ pace anchor กลับ; ตัดเฉพาะ Num_Corner_* (cluster index เปลี่ยนตามปี)
    feats = [c for c in all_features if not c.startswith("Num_Corner_")]
    if "Lap" in df_ml.columns and "Lap" not in feats:
        feats.append("Lap")
    return feats


def _build_X(df_ml, features):
    num = [c for c in features if c in df_ml.columns and c != "Track_Status"]
    X = df_ml[num].apply(pd.to_numeric, errors="coerce").copy()
    X = X.fillna(X.median())
    if "Track_Status" in df_ml.columns:
        d = pd.get_dummies(df_ml["Track_Status"].astype("string").fillna("UNKNOWN"))
        X = pd.concat([X.reset_index(drop=True), d.reset_index(drop=True)], axis=1)
    return X, list(X.columns)


def main():
    train_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                           pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    test_raw = pd.read_parquet(ROOT / "data_2025.parquet")

    train_ml, tr_feats, _, _ = _prepare_lab(train_raw, [])
    test_ml, _, _, _ = _prepare_lab(test_raw, [])
    feats = _features(tr_feats, train_ml)

    Xtr, ntr = _build_X(train_ml, feats)
    Xte, _ = _build_X(test_ml, feats)
    Xte = Xte.reindex(columns=ntr).fillna(Xtr.median())
    ytr = train_ml[TARGET].to_numpy(float)
    yte = test_ml[TARGET].to_numpy(float)
    mean_dur = float(np.exp(np.r_[ytr, yte]).mean())

    # ---- fit: standardized Ridge (stable coefficients) ----
    pipe = make_pipeline(build_preprocessor(feats), StandardScaler(), Ridge(alpha=1.0))
    pipe.fit(train_ml[feats], ytr)

    # raw-unit coefficients for interpretation (fit on raw X, no pipeline)
    ridge_raw = Ridge(alpha=1.0).fit(Xtr.to_numpy(float), ytr)
    raw = pd.Series(ridge_raw.coef_, index=ntr)

    # standardized coefficients for ranking
    Xs = StandardScaler().fit(Xtr).transform(Xtr)
    ridge_std = Ridge(alpha=1.0).fit(Xs, ytr)
    std = pd.Series(ridge_std.coef_, index=ntr)

    # ---- accuracy ----
    pred = np.asarray(pipe.predict(test_ml[feats]), float)
    mae = mean_absolute_error(np.exp(yte), np.exp(pred))
    rmse = float(np.sqrt(mean_squared_error(np.exp(yte), np.exp(pred))))
    r2 = r2_score(yte, pred)
    r2_in = r2_score(ytr, np.asarray(pipe.predict(train_ml[feats]), float))

    print(f"train rows={len(train_ml):,}  test rows={len(test_ml):,}  features={len(ntr)}  mean sector={mean_dur:.1f}s")
    print("\n=== ACCURACY (test = 2025) ===")
    print(f"  MAE_s  = {mae:.4f}")
    print(f"  RMSE_s = {rmse:.4f}")
    print(f"  R2(log) test = {r2:.4f}   (in-sample {r2_in:.4f})")

    # ---- explainability: coefficient table ----
    print(f"\n=== COEFFICIENTS (sorted by |standardized|; + = slower, - = faster) ===")
    print(f"  {'feature':34s} {'std.coef':>9s} {'sec/unit':>9s}")
    order = std.reindex(std.abs().sort_values(ascending=False).index).index
    for name in order:
        sc = std[name]
        ru = raw[name] * mean_dur
        print(f"  {name:34s} {sc:>+9.4f} {ru:>+9.4f}")

    # ---- save model ----
    model = {"pipeline": pipe, "features": feats, "mean_dur": mean_dur,
             "test_metrics": {"MAE_s": mae, "RMSE_s": rmse, "R2_log": r2}}
    with open(ROOT / "model_simple_2025.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved model -> model_simple_2025.pkl")
    print("DONE")


if __name__ == "__main__":
    main()
