"""Factor analysis — อะไรทำให้ lap/sector time ขึ้นหรือลง (ตัด pace anchor ออก)

ปัญหาเดิม: Prev_Duration/Clean_Pace/Best_Duration_So_Far = "ค่าเฉลี่ยรอบก่อนๆ"
ทำนายเก่งแต่ไม่บอกสาเหตุ (circular). ที่นี่เอาออกทั้งหมด แล้ววัดว่า factor จริง
(fuel, ยาง, อากาศ, traffic, track geometry, การขับ) อธิบาย lap time ได้แค่ไหน และตัวไหนสำคัญสุด

Model: Ridge บน Log_Duration (linear -> อ่าน coefficient เป็น % และวินาทีได้โดยตรง)
Data : 2023+2024+2025 รวมกัน (ตัด Num_Corner_* ออก — cluster index ไม่เท่ากันข้ามปี)

Run:  python -m f1pit.factor_analysis
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from f1pit.train import TARGET
from f1pit.wind_rain_lab import _prepare_lab

ROOT = Path(__file__).resolve().parent.parent
PACE_ANCHORS = ["Prev_Duration", "Prev_Sector_Duration", "Clean_Pace", "Best_Duration_So_Far"]

GROUPS = {
    "Fuel / Tyre deg": ["Lap", "Total_Distance_Driven", "Distance_On_Tyre", "Tyre_Age_Total",
                        "Pit_Stops_Done", "Tyre_Age_SOFT", "Tyre_Age_MEDIUM", "Tyre_Age_HARD",
                        "Tyre_Age_INTERMEDIATE", "Tyre_Age_WET",
                        "Tyre_Age_SOFT_2", "Tyre_Age_MEDIUM_2", "Tyre_Age_HARD_2",
                        "Tyre_Age_INTERMEDIATE_2", "Tyre_Age_WET_2"],
    "Weather": ["Track_Temperature", "Air_Temperature", "Humidity", "Pressure", "Rainfall"],
    "Traffic / Pit": ["Interval_Ahead", "Traffic_Within_1s", "Traffic_Within_1.5s",
                      "Traffic_Within_2s", "Traffic_Within_5s",
                      "Will_Box", "Prev_Lap_Pitted", "Prev_Lap_Box_Duration"],
    "Track geometry": ["Total_Sector_Length", "Total_Straight_Length"],
    "Driving (speed/gear/DRS)": ["Prev_Avg_Speed_Corners", "Prev_Avg_Speed_Straight", "Prev_Max_Speed",
                                 "Prev_Min_Speed", "Prev_Avg_Gear", "Prev_Full_Throttle_Pct",
                                 "Prev_Brake_Pct", "Prev_DRS_Pct", "Prev_I1_Speed", "Prev_I2_Speed",
                                 "Prev_St_Speed", "Track_Status_Inferred"],
}


def _group_of(name):
    if name.startswith("Track_Status_"):
        return "Track status (flag)"
    for g, cols in GROUPS.items():
        if name in cols:
            return g
    return "other"


def _build_X(df_ml, features):
    num = [c for c in features if c in df_ml.columns and c != "Track_Status"]
    X = df_ml[num].apply(pd.to_numeric, errors="coerce").copy()
    X = X.fillna(X.median())
    if "Track_Status" in df_ml.columns:
        d = pd.get_dummies(df_ml["Track_Status"].astype("string").fillna("UNKNOWN"))
        X = pd.concat([X.reset_index(drop=True), d.reset_index(drop=True)], axis=1)
    return X, list(X.columns)


def _analysis_features(all_features, df_ml):
    feats = [c for c in all_features
             if c not in PACE_ANCHORS and not c.startswith("Num_Corner_")]
    if "Lap" in df_ml.columns and "Lap" not in feats:
        feats.append("Lap")
    return feats


def main():
    years = [pd.read_parquet(ROOT / f"data_{y}.parquet") for y in (2023, 2024, 2025)]
    raw = pd.concat(years, ignore_index=True)
    df_ml, all_features, _, _ = _prepare_lab(raw, [])
    feats = _analysis_features(all_features, df_ml)
    print(f"rows={len(df_ml):,}  analysis features={len(feats)} "
          f"(pace anchors + Num_Corner removed)")

    X, names = _build_X(df_ml, feats)
    y = df_ml[TARGET].to_numpy(float)

    # ---- standardized Ridge: rank factors ----
    Xs = StandardScaler().fit(X).transform(X)
    ridge = Ridge(alpha=1.0).fit(Xs, y)
    coef = pd.Series(ridge.coef_, index=names)
    r2_in = r2_score(y, ridge.predict(Xs))
    print(f"\nIn-sample R2(log) = {r2_in:.4f}  (no pace anchors)")

    # ---- raw-unit Ridge: interpretable seconds ----
    ridge_raw = Ridge(alpha=1.0).fit(X.to_numpy(float), y)
    raw_coef = pd.Series(ridge_raw.coef_, index=names)
    mean_dur = float(np.exp(y).mean())

    print(f"\nTop-25 factors by |standardized coefficient|  (mean sector = {mean_dur:.1f}s):")
    rank = coef.reindex(coef.abs().sort_values(ascending=False).index)
    for name, c in rank.head(25).items():
        sec = c * mean_dur          # 1 std -> seconds
        print(f"  {name:32s} coef={c:+.4f}  (1 std -> {sec:+.2f}s)")

    print(f"\nInterpretable effects (raw units -> seconds, at {mean_dur:.1f}s mean sector):")
    for name in ["Lap", "Tyre_Age_Total", "Track_Temperature", "Air_Temperature",
                 "Pressure", "Rainfall", "Traffic_Within_1s", "Traffic_Within_1.5s",
                 "Traffic_Within_2s", "Traffic_Within_5s", "Total_Sector_Length", "Total_Straight_Length",
                 "Prev_DRS_Pct", "Prev_Avg_Speed_Corners", "Prev_Avg_Speed_Straight"]:
        if name in raw_coef.index:
            c = raw_coef[name]
            print(f"  {name:32s} {c:+.5f} /unit  -> {c * mean_dur:+.4f} s per unit")

    print(f"\nGroup importance (sum |standardized coef|):")
    gsum = coef.abs().groupby([_group_of(n) for n in coef.index]).sum().sort_values(ascending=False)
    for g, v in gsum.items():
        print(f"  {g:28s} {v:.4f}")

    # ---- temporal: train 23+24 -> test 25 (honest generalization of factors) ----
    print("\nTemporal hold-out (train 23+24 -> test 25), Ridge on factors only:")
    tr_raw = pd.concat(years[:2], ignore_index=True)
    te_raw = years[2]
    tr_ml, tr_feats, _, _ = _prepare_lab(tr_raw, [])
    te_ml, _, _, _ = _prepare_lab(te_raw, [])
    trf = _analysis_features(tr_feats, tr_ml)
    Xtr, ntr = _build_X(tr_ml, trf)
    Xte, nte = _build_X(te_ml, trf)
    Xte = Xte.reindex(columns=ntr).fillna(Xtr.median())
    sc = StandardScaler().fit(Xtr)
    r = Ridge(alpha=1.0).fit(sc.transform(Xtr), tr_ml[TARGET].to_numpy(float))
    pred = r.predict(sc.transform(Xte))
    a, p = np.exp(te_ml[TARGET].to_numpy(float)), np.exp(pred)
    print(f"  MAE_s={mean_absolute_error(a, p):.4f}  RMSE_s={float(np.sqrt(mean_squared_error(a, p))):.4f}  "
          f"R2(log)={r2_score(te_ml[TARGET].to_numpy(float), pred):.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
