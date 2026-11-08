"""Online / prequential evaluation ระดับ race-lap — ตรวจ concept drift ในฤดูกาล 2025

Warm start: 2023 + 2024 ทั้งหมด (online linear regression = SGDRegressor, feature ชุดเดียวกับ train_all)
Online loop (2025 เรียงตามเวลา, ทีละ race-lap = (Session_Key, Lap)):
    ทำนาย lap t ด้วยโมเดลปัจจุบัน -> บันทึก error -> partial_fit ด้วย lap t -> ทำนาย lap t+1
Static baseline: โมเดลเดียวกัน ฝึกครั้งเดียวบน 23+24 แล้ว "แช่แข็ง" ทำนายทั้ง 2025

Model: SGDRegressor (linear model ที่เรียน incremental ได้ผ่าน partial_fit)
ถ้ามี concept drift: error ของ static จะโตขึ้นตามฤดูกาล ขณะที่ online ตามทัน (หรือตามไม่ทัน)

Run:  python -m f1pit.online_lab
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from f1pit import plotting
from f1pit.train import TARGET
from f1pit.wind_rain_lab import _prepare_lab

ROOT = Path(__file__).resolve().parent.parent

ROLLING_WINDOW = 15          # race-laps: smoothing สำหรับกราฟ drift
STATIC_COLOR = "#8b92a5"     # แช่แข็ง (muted gray)
ONLINE_COLOR = "#F58518"     # อัปเดตออนไลน์ (accent)


def _features(all_features, df_ml):
    # ตัด Num_Corner_* (cluster index เปลี่ยนตามปี) แล้วเติม Lap เหมือน train_all
    feats = [c for c in all_features if not c.startswith("Num_Corner_")]
    if "Lap" in df_ml.columns and "Lap" not in feats:
        feats.append("Lap")
    return feats


def _build_matrix(df_ml, features, onehot_cols):
    """numeric features + one-hot Track_Status -> DataFrame (คอลัมน์คงที่)"""
    num = [c for c in features if c in df_ml.columns and c != "Track_Status"]
    X = df_ml[num].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    if "Track_Status" in df_ml.columns:
        d = pd.get_dummies(df_ml["Track_Status"].astype("string").fillna("UNKNOWN"), dtype=float)
        d = d.reindex(columns=onehot_cols, fill_value=0.0).reset_index(drop=True)
        X = pd.concat([X, d], axis=1)
    return X


def _sec_metrics(y_log, pred_log):
    a, p = np.exp(np.asarray(y_log, float)), np.exp(np.asarray(pred_log, float))
    return (float(mean_absolute_error(a, p)),
            float(np.sqrt(mean_squared_error(a, p))))


def _make_model():
    # Online linear regression: SGD + least-squares objective + L2 (≈ online Ridge).
    # NOTE: squared_error loss is sensitive to extreme outliers — the squared
    # tyre-age terms (Tyre_Age_*_2) are the ones to watch. If predictions blow up
    # to inf during partial_fit, either remove/winsorize those features or switch
    # to loss="huber" (robust linear regression, same model class).
    return SGDRegressor(loss="squared_error", penalty="l2", alpha=1e-2,
                        learning_rate="constant", eta0=1e-4,
                        max_iter=1000, tol=1e-4, random_state=42)


def main():
    train_raw = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                           pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    test_raw = pd.read_parquet(ROOT / "data_2025.parquet")

    train_ml, tr_feats, _, _ = _prepare_lab(train_raw, [])
    test_ml, te_feats, _, _ = _prepare_lab(test_raw, [])

    feats = _features(tr_feats, train_ml)
    feats = [c for c in feats if c in test_ml.columns]   # stable ข้ามปี

    onehot_cols = sorted(pd.concat([train_ml["Track_Status"], test_ml["Track_Status"]])
                         .astype("string").fillna("UNKNOWN").unique())

    # ---- feature matrix + fixed preprocessing (fit บน train เท่านั้น) ----
    Xtr = _build_matrix(train_ml, feats, onehot_cols)
    med = Xtr.median()
    Xtr = Xtr.fillna(med)
    y_train = train_ml[TARGET].to_numpy(float)
    scaler = StandardScaler().fit(Xtr)
    Xs_train = scaler.transform(Xtr.to_numpy(float))

    # ---- sort 2025 chronologically ----
    test_ml = test_ml.sort_values("Time_Start").reset_index(drop=True)
    Xte = _build_matrix(test_ml, feats, onehot_cols)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0.0).fillna(med)
    y_test = test_ml[TARGET].to_numpy(float)
    Xs_test = scaler.transform(Xte.to_numpy(float))

    # ---- static: ฝึกครั้งเดียว แช่แข็ง ----
    static = _make_model().fit(Xs_train, y_train)
    static_pred = static.predict(Xs_test)

    # ---- online: เริ่มจากโมเดลเดิม แล้ว partial_fit ทีละ lap ----
    online = copy.deepcopy(static)
    online_pred = np.empty_like(y_test)

    country_of = test_raw.groupby("Session_Key")["Country"].first().to_dict()

    grp = test_ml.groupby(["Session_Key", "Lap"], sort=False)
    group_order = grp["Time_Start"].min().sort_values().index   # (Session_Key, Lap) ตามเวลา

    rows = []
    for step, (s_key, lap) in enumerate(group_order):
        idx = grp.get_group((s_key, lap)).index.to_numpy()
        Xs = Xs_test[idx]
        y = y_test[idx]

        pred = online.predict(Xs)          # ทำนาย lap นี้ด้วยโมเดลปัจจุบัน
        online_pred[idx] = pred

        s_mae, _ = _sec_metrics(y, static_pred[idx])
        o_mae, _ = _sec_metrics(y, pred)
        rows.append({"step": step, "Session_Key": int(s_key), "Lap": int(lap),
                     "Country": country_of.get(s_key, "?"),
                     "Time_Start": test_ml.loc[idx[0], "Time_Start"],
                     "n": len(idx), "static_mae": s_mae, "online_mae": o_mae})

        online.partial_fit(Xs, y)          # แล้ว "เรียนรู้" lap นี้

    history = pd.DataFrame(rows)

    # ---- overall ----
    s_mae, s_rmse = _sec_metrics(y_test, static_pred)
    o_mae, o_rmse = _sec_metrics(y_test, online_pred)

    print(f"train rows={len(train_ml):,}  test rows={len(test_ml):,}  "
          f"features={len(feats)}  race-laps={len(history)}")
    print(f"\n=== OVERALL (2025, MAE/RMSE หน่วยวินาที) ===")
    print(f"  Static (23+24, frozen) : MAE_s={s_mae:.4f}  RMSE_s={s_rmse:.4f}")
    print(f"  Online (lap-by-lap)    : MAE_s={o_mae:.4f}  RMSE_s={o_rmse:.4f}")
    print(f"  Δ (online - static)    : MAE_s={o_mae - s_mae:+.4f}  RMSE_s={o_rmse - s_rmse:+.4f}")

    # ---- per-race (ฤดูกาล -> เห็น drift เป็นขั้น ๆ) ----
    print(f"\n=== PER-RACE (เรียงตามเวลา; MAE_s) ===")
    race_order = history.groupby("Session_Key")["Time_Start"].min().sort_values().index
    print(f"  {'race':22s} {'laps':>4s} {'static':>8s} {'online':>8s} {'Δ':>8s}")
    for rk in race_order:
        h = history[history["Session_Key"] == rk]
        print(f"  {str(h['Country'].iloc[0]):22s} {len(h):>4d} "
              f"{h['static_mae'].mean():>8.4f} {h['online_mae'].mean():>8.4f} "
              f"{(h['online_mae'] - h['static_mae']).mean():>+8.4f}")

    # ---- rolling -> chart ----
    history["static_roll"] = history["static_mae"].rolling(ROLLING_WINDOW, min_periods=1).mean()
    history["online_roll"] = history["online_mae"].rolling(ROLLING_WINDOW, min_periods=1).mean()

    save_path = ROOT / "online_drift_2025.png"
    _plot(history, save_path)
    print(f"\nSaved drift chart -> {save_path.name}")
    print("DONE")


def _plot(history, save_path):
    import matplotlib.pyplot as plt

    plotting.apply_theme()
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    x = history["step"].to_numpy()

    ax.plot(x, history["static_roll"], color=STATIC_COLOR, linewidth=2,
            label="Static (23+24, frozen)")
    ax.plot(x, history["online_roll"], color=ONLINE_COLOR, linewidth=2,
            label="Online (updates each lap)")

    ax.set_xlabel("Chronological race-lap index (2025)")
    ax.set_ylabel(f"Rolling MAE (s, window {ROLLING_WINDOW} laps)")
    ax.set_title("Concept drift — frozen linear model vs lap-by-lap online linear model")
    ax.legend()
    plotting.show(fig, save_path=save_path)


if __name__ == "__main__":
    main()
