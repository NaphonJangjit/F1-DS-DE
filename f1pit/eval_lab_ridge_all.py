"""Ridge Regression on ALL features (LOTO, leak-free) — เปรียบเทียบกับ pace-only.

Log-align เฉพาะ feature ที่เป็นวินาที (ให้ตรงกับ target Log_Duration) แล้ว
ColumnTransformer (impute+one-hot fit บน train) -> StandardScaler -> Ridge

Run:  python -m f1pit.eval_lab_ridge_all
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab

DATA_PATH = Path(__file__).resolve().parent.parent / "data_2023.parquet"
DURATION_LIKE = ["Prev_Duration", "Prev_Sector_Duration", "Clean_Pace", "Best_Duration_So_Far"]
PACE = ["Best_Duration_So_Far", "Clean_Pace", "Prev_Duration"]


def _metrics(y_log_true, y_log_pred):
    a = np.exp(np.asarray(y_log_true, float))
    p = np.exp(np.asarray(y_log_pred, float))
    return mean_absolute_error(a, p), float(np.sqrt(mean_squared_error(a, p)))


def _log_dur(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = np.log(pd.to_numeric(df[c], errors="coerce").clip(lower=1e-6))
    return df


def loto_ridge(feature_set, log_dur=True, alpha=1.0):
    raw = pd.read_parquet(DATA_PATH)
    sessions = sorted(raw["Session_Key"].unique())
    maes, rmses = [], []
    for s in sessions:
        tr_ml, feats, _, _ = _prepare_lab(raw[raw["Session_Key"] != s].copy(), [])
        te_ml, _, _, _ = _prepare_lab(raw[raw["Session_Key"] == s].copy(), [])
        use = [c for c in feature_set if c in feats]
        tr = tr_ml[use].copy()
        te = te_ml[use].copy()
        for c in use:
            if c not in te.columns:
                te[c] = np.nan
        if log_dur:
            dl = [c for c in DURATION_LIKE if c in use]
            tr = _log_dur(tr, dl)
            te = _log_dur(te, dl)
        pipe = make_pipeline(build_preprocessor(use), StandardScaler(), Ridge(alpha=alpha))
        pipe.fit(tr, tr_ml[TARGET])
        pred = np.asarray(pipe.predict(te), float)
        m, r = _metrics(te_ml[TARGET].to_numpy(float), pred)
        maes.append(m); rmses.append(r)
    return float(np.mean(maes)), float(np.mean(rmses))


def main():
    raw = pd.read_parquet(DATA_PATH)
    _, all_feats, _, _ = _prepare_lab(raw, [])
    print(f"features (all) = {len(all_feats)}\n")

    m, r = loto_ridge(PACE, log_dur=True)
    print(f"Ridge pace-only (3, log)          MAE_s={m:.4f}  RMSE_s={r:.4f}   [reference]")

    m, r = loto_ridge(all_feats, log_dur=True)
    print(f"Ridge ALL features (log-dur)      MAE_s={m:.4f}  RMSE_s={r:.4f}")

    m, r = loto_ridge(all_feats, log_dur=False)
    print(f"Ridge ALL features (raw)          MAE_s={m:.4f}  RMSE_s={r:.4f}")

    for a in (10.0, 100.0):
        m, r = loto_ridge(all_feats, log_dur=True, alpha=a)
        print(f"Ridge ALL features (log, alpha={a:<5})  MAE_s={m:.4f}  RMSE_s={r:.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
