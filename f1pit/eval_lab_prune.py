"""LOTO with pruned (core) feature sets — validate that dropping noisy features helps cross-track.

จาก eval_lab: importance บน train รวมกัน 4 feature (Best/Clean/Prev + Total_Sector_Length) = 91%
และบน Abu Dhabi การ prune เหลือ 4-5 feature ลด MAE 1.89 -> 0.89s
ที่นี่ validate ว่าเป็นจริงทุกสนามไหม ด้วย LOTO ครบ 22 สนาม

Run:  python -m f1pit.eval_lab_prune
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline

from f1pit import config
from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab

DATA_PATH = Path(__file__).resolve().parent.parent / "data_2023.parquet"

PACE = ["Best_Duration_So_Far", "Clean_Pace", "Prev_Duration"]
SETS = {
    "pace only (3)": PACE,
    "pace + length (4)": PACE + ["Total_Sector_Length"],
    "pace + length + corner8 (5)": PACE + ["Total_Sector_Length", "Num_Corner_8"],
    "pace + length + top8 (8)": PACE + ["Total_Sector_Length",
                                          "Num_Corner_8", "Num_Corner_0", "Num_Corner_1"],
}


def _metrics(y_log_true, y_log_pred):
    a = np.exp(np.asarray(y_log_true, float))
    p = np.exp(np.asarray(y_log_pred, float))
    return mean_absolute_error(a, p), float(np.sqrt(mean_squared_error(a, p)))


def loto(feature_set, n_est=150):
    raw = pd.read_parquet(DATA_PATH)
    sessions = sorted(raw["Session_Key"].unique())
    light = ExtraTreesRegressor(n_estimators=n_est, max_features=0.9, min_samples_leaf=2,
                                n_jobs=-1, random_state=config.RANDOM_STATE)
    maes, rmses = [], []
    for s in sessions:
        tr_r = raw[raw["Session_Key"] != s].copy()
        te_r = raw[raw["Session_Key"] == s].copy()
        tr_ml, feats, _, _ = _prepare_lab(tr_r, [])
        te_ml, _, _, _ = _prepare_lab(te_r, [])
        use = [c for c in feature_set if c in feats]
        for c in use:
            if c not in te_ml.columns:
                te_ml[c] = np.nan
        pipe = make_pipeline(build_preprocessor(use), light)
        pipe.fit(tr_ml[use], tr_ml[TARGET])
        pred = np.asarray(pipe.predict(te_ml[use]), float)
        mae, rmse = _metrics(te_ml[TARGET].to_numpy(float), pred)
        maes.append(mae); rmses.append(rmse)
    return float(np.mean(maes)), float(np.mean(rmses)), maes, rmses


def main():
    print(f"reference: LOTO ALL 49 features = MAE_s=1.7242  RMSE_s=2.3216\n")
    for name, fset in SETS.items():
        m, r, maes, rmses = loto(fset)
        print(f"LOTO {name:30s}  MAE_s={m:.4f}  RMSE_s={r:.4f}  "
              f"(min={min(maes):.4f}, max={max(maes):.4f})")
    print("\nDONE")


if __name__ == "__main__":
    main()
