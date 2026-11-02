"""Leak-free evaluation lab (does NOT modify core logic).

ปัญหาเดิม: train.py ใช้ train_test_split แบบสุ่ม 20% -> แถวจาก session/สนามเดียวกัน
ไปอยู่ทั้ง train และ test -> model "ท่องจำ" track profile / session pace -> MAE ต่ำเกินจริง

ที่นี่ split ตาม Session_Key (สนาม):
  * Abu Dhabi (Yas Marina) = test, ที่เหลือ 21 สนาม = train (ตามที่ขอ)
  * + leave-one-track-out (LOTO) ครบ 22 สนาม เพื่อให้ได้ตัวเลข leak-free ที่ robust
ใช้ build_pipeline ของ train.py (impute median / one-hot fit บน TRAIN เท่านั้น -> ไม่ leak)
แล้ว "ถอด" feature ที่ importance ต่ำ (noise) แล้ววัด accuracy ใหม่

Run:  python -m f1pit.eval_lab
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from f1pit import config
from f1pit.train import TARGET, TEST_SIZE, build_pipeline, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.feature_lab import build_X

DATA_PATH = Path(__file__).resolve().parent.parent / "data_2023.parquet"
ABU_DHABI_COUNTRY = "United Arab Emirates"

NOISE_IMPORTANCE_CUT = 0.005   # 0.5% — feature ที่รวม importance ต่ำกว่านี้ถือเป็น noise


def _metrics(y_log_true, y_log_pred):
    a = np.exp(np.asarray(y_log_true, float))
    p = np.exp(np.asarray(y_log_pred, float))
    return {
        "MAE_s": mean_absolute_error(a, p),
        "RMSE_s": float(np.sqrt(mean_squared_error(a, p))),
        "R2_log": r2_score(y_log_true, y_log_pred),
    }


def _eval_pipeline(features, train_ml, test_ml, algo="ExtraTrees", est=None, label=""):
    """fit บน train (impute/one-hot จาก train) -> predict test -> metrics เป็นวินาที"""
    for c in features:
        if c not in test_ml.columns:
            test_ml[c] = np.nan
    if est is None:
        pipe = build_pipeline(algo, features)
    else:
        pipe = make_pipeline(build_preprocessor(features), est)
    pipe.fit(train_ml[features], train_ml[TARGET])
    pred = np.asarray(pipe.predict(test_ml[features]), float)
    m = _metrics(test_ml[TARGET].to_numpy(float), pred)
    print(f"  {label:46s} MAE_s={m['MAE_s']:.4f}  RMSE_s={m['RMSE_s']:.4f}  R2(log)={m['R2_log']:.5f}")
    return m


def _importance(features, train_ml):
    """importance ต่อ feature (รวม one-hot กลับเข้า parent) — คำนวณบน TRAIN เท่านั้น"""
    X, cols = build_X(train_ml, features, log_duration=False)
    y = train_ml[TARGET].to_numpy(float)
    est = ExtraTreesRegressor(n_estimators=400, max_features=0.9, min_samples_leaf=2,
                              n_jobs=-1, random_state=config.RANDOM_STATE)
    est.fit(X, y)
    imp = pd.Series(est.feature_importances_, index=cols)
    # รวม one-hot Track_Status_* กลับเป็น "Track_Status" (แต่ "Track_Status_Inferred"
    # เป็น numeric feature จริง ไม่ใช่ one-hot -> อย่าไปรวม)
    parent = pd.Series(
        ["Track_Status" if c.startswith("Track_Status_") and c != "Track_Status_Inferred" else c
         for c in imp.index],
        index=imp.index,
    )
    agg = imp.groupby(parent).sum()
    return agg.sort_values(ascending=False)


def _abudhabi_session(raw):
    sess = raw.loc[raw["Country"] == ABU_DHABI_COUNTRY, "Session_Key"].unique()
    return int(sess[0])


def main():
    print("Loading data ...")
    raw = pd.read_parquet(DATA_PATH)
    sessions = sorted(raw["Session_Key"].unique())
    abu = _abudhabi_session(raw)
    print(f"  sessions={len(sessions)}  Abu Dhabi Session_Key={abu}")

    # ------------------------------------------------------------------
    # 1. เลขเก่า (leaky) สำหรับเทียบ — random 20% บนข้อมูลรวมทุกสนาม
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("1) LEAKY reference (random 20% split, old style)")
    full_ml, full_features, _, _ = _prepare_lab(raw, [])
    tr, te = train_test_split(np.arange(len(full_ml)), test_size=TEST_SIZE,
                              random_state=config.RANDOM_STATE)
    train_ml = full_ml.iloc[tr].copy(); test_ml = full_ml.iloc[te].copy()
    _eval_pipeline(full_features, train_ml, test_ml, label="ExtraTrees (random split)  [old number]")

    # ------------------------------------------------------------------
    # 2. Leak-free: Abu Dhabi = test, rest = train
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("2) LEAK-FREE: Abu Dhabi = test, 21 races = train")
    train_raw = raw[raw["Session_Key"] != abu].copy()
    test_raw = raw[raw["Session_Key"] == abu].copy()
    train_ml, features, _, _ = _prepare_lab(train_raw, [])
    test_ml, _, _, _ = _prepare_lab(test_raw, [])
    print(f"  train rows={len(train_ml):,}  test rows={len(test_ml):,}  features={len(features)}")

    base_m = _eval_pipeline(features, train_ml, test_ml, label="ExtraTrees (all features)")

    # ------------------------------------------------------------------
    # 3. importance (บน train) -> ถอด noise
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("3) FEATURE IMPORTANCE (fit on train only) + noise removal")
    imp = _importance(features, train_ml)
    for name, v in imp.head(20).items():
        print(f"  {name:34s} {v:.4f}")

    for cut in (0.0, NOISE_IMPORTANCE_CUT, 0.01, 0.02):
        keep = [c for c in features if c in imp.index and imp[c] > cut]
        # Track_Status เก็บไว้เสมอ (categorical สำคัญ SC/flag)
        if "Track_Status" not in keep and "Track_Status" in features:
            keep.append("Track_Status")
        dropped = [c for c in features if c not in keep]
        _eval_pipeline(keep, train_ml, test_ml,
                       label=f"ExtraTrees (drop {len(dropped)} feats, imp<{cut})")
        if cut == NOISE_IMPORTANCE_CUT:
            print(f"     dropped (noise): {dropped}")

    # ------------------------------------------------------------------
    # 4. LOTO: leave-one-track-out ครบ 22 สนาม (light ExtraTrees)
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("4) LEAVE-ONE-TRACK-OUT (all 22 races, ExtraTrees 150 trees)")
    light = ExtraTreesRegressor(n_estimators=150, max_features=0.9, min_samples_leaf=2,
                                n_jobs=-1, random_state=config.RANDOM_STATE)
    maes, rmses = [], []
    for s in sessions:
        tr_r = raw[raw["Session_Key"] != s].copy()
        te_r = raw[raw["Session_Key"] == s].copy()
        tr_ml, feats, _, _ = _prepare_lab(tr_r, [])
        te_ml, _, _, _ = _prepare_lab(te_r, [])
        for c in feats:
            if c not in te_ml.columns:
                te_ml[c] = np.nan
        pipe = make_pipeline(build_preprocessor(feats), light)
        pipe.fit(tr_ml[feats], tr_ml[TARGET])
        pred = np.asarray(pipe.predict(te_ml[feats]), float)
        m = _metrics(te_ml[TARGET].to_numpy(float), pred)
        maes.append(m["MAE_s"]); rmses.append(m["RMSE_s"])
        print(f"  {s:6d}  MAE_s={m['MAE_s']:.4f}  RMSE_s={m['RMSE_s']:.4f}")
    print(f"  LOTO mean:  MAE_s={np.mean(maes):.4f}  RMSE_s={np.mean(rmses):.4f}  "
          f"(min MAE={min(maes):.4f}, max={max(maes):.4f})")

    print("\nDONE")


if __name__ == "__main__":
    main()
