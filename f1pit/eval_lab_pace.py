"""Ceiling for the pace-only nowcast: persistence floor + linear (Ridge) vs ExtraTrees.

ExtraTrees pace-only LOTO = 0.5609s (จาก eval_lab_prune).
ที่นี่วัดว่า (1) floor ของ persistence (ทำนาย = Prev_Duration) คือเท่าไร และ
(2) Ridge (linear) บน pace 3 ตัวทำได้ดีกว่า ExtraTrees ไหม (ความสัมพันธ์เกือบ linear)

Run:  python -m f1pit.eval_lab_pace
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from f1pit.train import TARGET
from f1pit.wind_rain_lab import _prepare_lab

DATA_PATH = Path(__file__).resolve().parent.parent / "data_2023.parquet"
PACE = ["Best_Duration_So_Far", "Clean_Pace", "Prev_Duration"]


def _metrics(y_log_true, y_log_pred):
    a = np.exp(np.asarray(y_log_true, float))
    p = np.exp(np.asarray(y_log_pred, float))
    return mean_absolute_error(a, p), float(np.sqrt(mean_squared_error(a, p)))


def main():
    raw = pd.read_parquet(DATA_PATH)
    sessions = sorted(raw["Session_Key"].unique())

    # 1. persistence floor: predict = Prev_Duration
    mae_p, rmse_p = [], []
    for s in sessions:
        te_ml, _, _, _ = _prepare_lab(raw[raw["Session_Key"] == s].copy(), [])
        pred = np.log(te_ml["Prev_Duration"].to_numpy(float))
        m, r = _metrics(te_ml[TARGET].to_numpy(float), pred)
        mae_p.append(m); rmse_p.append(r)
    print(f"LOTO persistence (Prev_Duration only)  MAE_s={np.mean(mae_p):.4f}  "
          f"RMSE_s={np.mean(rmse_p):.4f}  (min={min(mae_p):.4f}, max={max(mae_p):.4f})")

    # 2. Ridge on pace
    for label, log_dur in [("Ridge pace (raw sec)", False), ("Ridge pace (log sec)", True)]:
        mae_r, rmse_r = [], []
        for s in sessions:
            tr_ml, _, _, _ = _prepare_lab(raw[raw["Session_Key"] != s].copy(), [])
            te_ml, _, _, _ = _prepare_lab(raw[raw["Session_Key"] == s].copy(), [])
            Xtr = tr_ml[PACE].apply(pd.to_numeric, errors="coerce")
            Xte = te_ml[PACE].apply(pd.to_numeric, errors="coerce")
            if log_dur:
                Xtr = np.log(Xtr.clip(lower=1e-6))
                Xte = np.log(Xte.clip(lower=1e-6))
            Xtr = Xtr.fillna(Xtr.median())
            Xte = Xte.fillna(Xtr.median())
            scaler = StandardScaler().fit(Xtr)
            ridge = Ridge(alpha=1.0).fit(scaler.transform(Xtr), tr_ml[TARGET].to_numpy(float))
            pred = ridge.predict(scaler.transform(Xte))
            m, r = _metrics(te_ml[TARGET].to_numpy(float), pred)
            mae_r.append(m); rmse_r.append(r)
        print(f"LOTO {label:28s}  MAE_s={np.mean(mae_r):.4f}  RMSE_s={np.mean(rmse_r):.4f}  "
              f"(min={min(mae_r):.4f}, max={max(mae_r):.4f})")

    print("\nDONE")


if __name__ == "__main__":
    main()
