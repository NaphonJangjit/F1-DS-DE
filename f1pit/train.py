"""Train + เปรียบเทียบหลายโมเดลบน Log_Duration

>>> out = f1pit.train_models(data)                          # ทุกโมเดล + HTML report
>>> out = f1pit.train_models(data, algorithms=["LightGBM", "ExtraTrees"])
>>> out.results                 # ตาราง MAE / RMSE / R2
>>> out.best_model.predict(X)   # pipeline ที่ fit แล้ว
>>> out.seconds_metrics()       # MAE/RMSE หน่วยวินาที
"""

import importlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config
from .features import TRAFFIC_COLUMNS, TYRE_COMPOUNDS, WEATHER_MAP

# ==================================================
# Config
# ==================================================
TARGET = "Log_Duration"
TEST_SIZE = 0.2

CATEGORICAL_FEATURES = ["Tyre", "Track_Status"]

# Prev_* trend features: rolling mean + slope (derivative) ย้อนหลัง PREV_TREND_WINDOW lap
# (ใช้เฉพาะค่าจากแถวก่อนหน้า -> point-in-time ถูกต้อง)
PREV_TREND_WINDOW = 5

# Prev_* ที่เป็น "previous" ตาม sector เดียวกัน (ค่าจาก lap ก่อนหน้าใน sector นั้น)
PREV_TREND_BASE = [
    "Prev_Avg_Speed_Corners", "Prev_Avg_Speed_Straight", "Prev_Duration",
    "Prev_Full_Throttle_Pct", "Prev_Brake_Pct", "Prev_DRS_Pct",
    "Prev_Max_Speed", "Prev_Min_Speed", "Prev_Avg_Gear",
    "Prev_I1_Speed", "Prev_I2_Speed", "Prev_St_Speed",
]
# Prev_Sector_Duration เป็นลำดับตามเวลา (ต่อเนื่องข้าม sector) -> group แยก
PREV_SECTOR_SEQ = "Prev_Sector_Duration"


def _trend_cols(base):
    return [f"{base}_roll{PREV_TREND_WINDOW}_mean", f"{base}_roll{PREV_TREND_WINDOW}_slope"]


PREV_TREND_COLUMNS = [c for b in [*PREV_TREND_BASE, PREV_SECTOR_SEQ] for c in _trend_cols(b)]

NUMERIC_FEATURES = [
    "Tyre_Age", "Tyre_Age_Total", "Track_Status_Inferred",
    "Prev_Avg_Speed_Corners", "Prev_Avg_Speed_Straight", "Prev_Duration", "Prev_Sector_Duration",
    "Clean_Pace", "Best_Duration_So_Far",
    "Prev_Full_Throttle_Pct", "Prev_Brake_Pct", "Prev_DRS_Pct",
    "Prev_Max_Speed", "Prev_Min_Speed", "Prev_Avg_Gear",
    "Prev_I1_Speed", "Prev_I2_Speed", "Prev_St_Speed",
    "Interval_Ahead",
    *TRAFFIC_COLUMNS,
    "Will_Box", "Prev_Lap_Pitted", "Prev_Lap_Box_Duration", "Pit_Stops_Done",
    *WEATHER_MAP.values(),
    # Track profile (Num_Corner_* ถูกตรวจจับอัตโนมัติ)
    "Total_Sector_Length", "Total_Straight_Length",
    "Total_Distance_Driven", "Distance_On_Tyre",
    *PREV_TREND_COLUMNS,
]

REQUIRED_COLS = ["Lap", "Sector", "Tyre", "Tyre_Age", "Prev_Avg_Speed_Corners", "Prev_Avg_Speed_Straight",
                 "Prev_Duration", TARGET, "Track_Status"]

# interaction compound × age: แทน one-hot ของ Tyre + ค่า Tyre_Age เดี่ยว ๆ
TYRE_AGE_INTERACTION = [f"Tyre_Age_{c}" for c in TYRE_COMPOUNDS]

# พจน์ quadratic ต่อ compound (จับยางเสื่อม non-linear) — คู่กับ Tyre_Age_<compound> เดิม
TYRE_AGE_SQUARED = [f"Tyre_Age_{c}_2" for c in TYRE_COMPOUNDS]

# Outlier removal: ตัด sector ที่ Duration สูงผิดปกติ (อุบัติเหตุ / collision / SC / ธงเหลือง)
OUTLIER_K = 4.0          # Duration > median + OUTLIER_K * MAD -> outlier (ด้านช้า)
OUTLIER_MAD_FLOOR = 0.5  # วินาที: MAD ขั้นต่ำ ป้องกัน MAD≈0 ทำให้ตัดทุกอย่าง

ALGO_COLORS = {
    "CatBoost": "#4C78A8",
    "XGBoost": "#F58518",
    "LightGBM": "#54A24B",
    "HistGradientBoosting": "#E45756",
    "ExtraTrees": "#B279A2",
    "RandomForest": "#72B7B2",
    "GradientBoosting": "#FF9DA6",
    "Stacking Ensemble": "#9D755D",
    "LinearRegression": "#9aa5b1",
}

OPTIONAL_LIBS = {"CatBoost": ("catboost", "CatBoostRegressor"),
                 "XGBoost": ("xgboost", "XGBRegressor"),
                 "LightGBM": ("lightgbm", "LGBMRegressor")}

ALL_ALGORITHMS = ["LinearRegression", "CatBoost", "XGBoost", "LightGBM", "HistGradientBoosting",
                  "ExtraTrees", "RandomForest", "GradientBoosting", "Stacking Ensemble"]


def optional_model(name):
    """คืน class ของโมเดล optional (catboost/xgboost/lightgbm) หรือ raise ImportError"""
    module, cls = OPTIONAL_LIBS[name]
    return getattr(importlib.import_module(module), cls)


def available_algorithms():
    """(รายชื่อโมเดลที่ใช้ได้, ข้อความ error ของโมเดลที่ import ไม่ได้)"""
    algos, errors = [], []
    for name in ALL_ALGORITHMS:
        if name in OPTIONAL_LIBS:
            try:
                optional_model(name)
            except Exception as exc:
                errors.append(f"{name} unavailable: {exc}")
                continue
        algos.append(name)
    return algos, errors


# ==================================================
# Data preparation
# ==================================================
def _to_float(series):
    series = series.map(lambda v: float(v) if isinstance(v, (bool, np.bool_)) else v)
    return pd.to_numeric(series, errors="coerce").astype("float64")


def corner_feature_columns(df):
    cols = [c for c in df.columns if c.startswith("Num_Corner_") and c.rsplit("_", 1)[-1].isdigit()]
    return sorted(cols, key=lambda c: int(c.rsplit("_", 1)[-1]))


def add_tyre_age_interaction(df):
    """เติม Tyre_Age_<compound> (linear) และ Tyre_Age_<compound>_2 (quadratic) ต่อ compound

    interaction compound × age × Circuit_Total_Length: ยางเสื่อมคิดเป็น "ระยะทาง" ต่อ lap
    (length × age) เพื่อเทียบข้ามสนามที่มีความยาว lap ต่างกันได้
    - Tyre_Age_<compound>   = 1[compound] × Tyre_Age × Circuit_Total_Length
    - Tyre_Age_<compound>_2 = 1[compound] × Tyre_Age² × Circuit_Total_Length
    (แทน one-hot ของ Tyre + ค่า Tyre_Age เดี่ยว ๆ; ไม่มี Circuit_Total_Length -> length=1.0)
    """
    out = df.copy()
    if "Tyre" not in out.columns or "Tyre_Age" not in out.columns:
        for c in TYRE_COMPOUNDS:
            out[f"Tyre_Age_{c}"] = np.nan
            out[f"Tyre_Age_{c}_2"] = np.nan
        return out
    tyre = out["Tyre"].fillna("").astype(str).str.upper().str.strip()
    age = pd.to_numeric(out["Tyre_Age"], errors="coerce")
    age_arr = age.to_numpy(dtype=float)
    ok_age = age.notna().to_numpy()

    # circuit length (เมตร); ถ้าไม่มี / NaN / <=0 -> 1.0 (ไม่ scale)
    if "Circuit_Total_Length" in out.columns:
        length = pd.to_numeric(out["Circuit_Total_Length"], errors="coerce").to_numpy(dtype=float)
    else:
        length = np.ones(len(out), dtype=float)
    length = np.where(np.isfinite(length) & (length > 0), length, 1.0)

    scaled = age_arr * length            # age × length
    scaled2 = age_arr ** 2 * length      # age² × length

    for c in TYRE_COMPOUNDS:
        m = (tyre.to_numpy() == c) & ok_age
        out[f"Tyre_Age_{c}"] = np.where(m, scaled, 0.0)
        out[f"Tyre_Age_{c}_2"] = np.where(m, scaled2, 0.0)
    return out


def _roll_slope(s, window):
    """Slope (OLS) ของ window ล่าสุด ต่อแถว — vectorized ผ่าน shift ด้วย weight คงที่

    x = 0..window-1 (เก่า -> ใหม่); slope = Σ (x - x̄) / Σ(x - x̄)² · y
    แถวที่ยังมีข้อมูลใน window ไม่ครบ (shift เกินหัว) -> NaN
    """
    x = np.arange(window, dtype=float)
    w = (x - x.mean()) / float(((x - x.mean()) ** 2).sum())
    out = pd.Series(0.0, index=s.index, dtype=float)
    for k in range(window):
        out = out + w[k] * s.shift(window - 1 - k)
    return out


def add_prev_trend_features(df):
    """เติม rolling mean + slope (แนวโน้ม / derivative) ของ Prev_* features ย้อนหลัง PREV_TREND_WINDOW lap

    - PREV_TREND_BASE : group (Session_Key, Driver, Sector) เรียงตาม Lap (ค่าใน sector เดียวกัน)
    - Prev_Sector_Duration : ลำดับตามเวลา (Lap, Sector) group (Session_Key, Driver)
    ใช้เฉพาะค่าจากแถวก่อนหน้า (shift/rolling backward) -> point-in-time ถูกต้อง

    คืน (df ใหม่, รายชื่อคอลัมน์ใหม่ที่เพิ่มจริง)
    """
    out = df.copy()
    if not {"Session_Key", "Driver", "Lap", "Sector"}.issubset(out.columns):
        return out, []
    window = PREV_TREND_WINDOW
    new_cols = []

    def _apply(base, sorted_df, group_cols):
        mean_col, slope_col = _trend_cols(base)
        mean = pd.Series(np.nan, index=sorted_df.index, dtype=float)
        slope = pd.Series(np.nan, index=sorted_df.index, dtype=float)
        for _, g in sorted_df.groupby(group_cols, sort=False):
            v = pd.to_numeric(g[base], errors="coerce")
            mean.loc[g.index] = v.rolling(window, min_periods=1).mean().to_numpy()
            slope.loc[g.index] = _roll_slope(v, window).to_numpy()
        return mean.reindex(out.index), slope.reindex(out.index)

    sector_sorted = out.sort_values(["Session_Key", "Driver", "Sector", "Lap"])
    for base in PREV_TREND_BASE:
        if base not in out.columns:
            continue
        mean_col, slope_col = _trend_cols(base)
        out[mean_col], out[slope_col] = _apply(base, sector_sorted, ["Session_Key", "Driver", "Sector"])
        new_cols += [mean_col, slope_col]

    if PREV_SECTOR_SEQ in out.columns:
        seq_sorted = out.sort_values(["Session_Key", "Driver", "Lap", "Sector"])
        mean_col, slope_col = _trend_cols(PREV_SECTOR_SEQ)
        out[mean_col], out[slope_col] = _apply(PREV_SECTOR_SEQ, seq_sorted, ["Session_Key", "Driver"])
        new_cols += [mean_col, slope_col]

    return out, new_cols


def remove_duration_outliers(df, k=OUTLIER_K, mad_floor=OUTLIER_MAD_FLOOR):
    """ตัดแถวที่ Duration สูงผิดปกติ (ด้านช้า) ภายในกลุ่ม (session, sector)

    อุบัติเหตุ / collision / SC / ธงเหลือง ทำให้ sector ช้ามาก -> outlier ด้านสูง
    ใช้ median + k*MAD (robust: ค่า outlier เองไม่ bias ค่ากลาง) — คืน (df, จำนวนที่ตัด)
    """
    if len(df) == 0:
        return df, 0
    raw_col = next((c for c in ("Duration", "Outcome_Duration") if c in df.columns), None)
    if raw_col is None:
        dur = np.exp(pd.to_numeric(df[TARGET], errors="coerce"))
    else:
        dur = pd.to_numeric(df[raw_col], errors="coerce")

    group_cols = []
    sid = next((c for c in ("Session_Key", "Session", "Country") if c in df.columns), None)
    if sid:
        group_cols.append(sid)
    if "Sector" in df.columns:
        group_cols.append("Sector")

    keys = [df[c] for c in group_cols] if group_cols else [pd.Series(0, index=df.index)]
    med = dur.groupby(keys, dropna=False).transform("median")
    mad = (dur - med).abs().groupby(keys, dropna=False).transform("median")
    thresh = med + k * mad.clip(lower=mad_floor)
    outlier = (dur > thresh).fillna(False).astype(bool)
    removed = int(outlier.sum())
    return df[~outlier].reset_index(drop=True), removed


def prepare_data(df, outlier_k=OUTLIER_K):
    """-> (df_ml, features, notes)

    outlier_k : float | None
        ตัด outlier ด้านช้าของ Duration (median + outlier_k * MAD ต่อ session/sector);
        None = ไม่ตัด
    """
    notes = []
    df = df.copy()

    # trend features (rolling mean/slope ของ Prev_*) — คำนวณบน raw history ก่อนตัด outlier
    df, _ = add_prev_trend_features(df)

    # สร้าง Log_Duration จาก Duration ถ้ายังไม่มี (ไม่ log ซ้ำ)
    if TARGET not in df.columns:
        raw_col = next((c for c in ("Duration", "Outcome_Duration") if c in df.columns), None)
        if raw_col is not None:
            raw = pd.to_numeric(df[raw_col], errors="coerce")
            df[TARGET] = np.log(raw.where(raw > 0))
            notes.append(f"Created {TARGET} = ln({raw_col}).")

    # ตัด outlier ด้านช้า (อุบัติเหตุ / collision / SC) ก่อน subset คอลัมน์
    if outlier_k is not None:
        df, removed = remove_duration_outliers(df, k=float(outlier_k))
        if removed:
            notes.append(f"Removed {removed} rows as high-side Duration outliers (k={outlier_k}).")

    df = df.rename(columns={low: cap for low, cap in WEATHER_MAP.items()
                            if cap not in df.columns and low in df.columns})

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}\nAvailable: {df.columns.tolist()}")

    corner_cols = corner_feature_columns(df)
    all_features = list(dict.fromkeys([*CATEGORICAL_FEATURES, *NUMERIC_FEATURES, *corner_cols]))
    present = [c for c in all_features if c in df.columns]
    absent = [c for c in all_features if c not in df.columns]
    if absent:
        notes.append("Features not found and excluded: " + ", ".join(absent))

    # Circuit_Total_Length เก็บไว้ใช้กับ interaction เท่านั้น (ไม่ใช่ feature เอง)
    keep_extra = [c for c in ("Circuit_Total_Length",) if c in df.columns]
    df_ml = df[list(dict.fromkeys(["Lap", "Sector", TARGET, *keep_extra, *present]))].copy()

    for col in CATEGORICAL_FEATURES:
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].astype("string").str.upper().str.strip()
    df_ml["Tyre"] = df_ml["Tyre"].replace("", pd.NA)
    df_ml["Track_Status"] = df_ml["Track_Status"].fillna("UNKNOWN").replace("", "UNKNOWN")

    for col in ["Lap", "Sector", TARGET, *[c for c in present if c not in CATEGORICAL_FEATURES]]:
        df_ml[col] = _to_float(df_ml[col])

    df_ml = df_ml.dropna(subset=["Tyre", "Tyre_Age", "Prev_Avg_Speed_Corners", "Prev_Duration", TARGET, "Lap", "Sector"])
    df_ml = df_ml[
        np.isfinite(df_ml[TARGET].to_numpy(dtype=float))
        & (df_ml["Tyre_Age"] > 0)
        & (df_ml["Prev_Avg_Speed_Corners"] > 0)
        & (df_ml["Prev_Duration"] > 0)
        & df_ml["Sector"].isin([1, 2, 3])
    ].reset_index(drop=True)
    if len(df_ml) < 10:
        raise ValueError("ข้อมูลเหลือน้อยเกินไปหลังจากทำความสะอาดข้อมูล")

    all_nan = [c for c in present if df_ml[c].isna().all()]
    if all_nan:
        notes.append("Features with no usable values excluded: " + ", ".join(all_nan))
    features = [c for c in present if c not in all_nan]

    # แทน Tyre (one-hot) + Tyre_Age (เดี่ยว) ด้วย interaction Tyre_Age_<compound> (+ quadratic)
    df_ml = add_tyre_age_interaction(df_ml)
    features = ([c for c in features if c not in ("Tyre", "Tyre_Age")]
                + TYRE_AGE_INTERACTION + TYRE_AGE_SQUARED)
    return df_ml, features, notes


# ==================================================
# Models
# ==================================================
def build_preprocessor(features):
    cat = [c for c in CATEGORICAL_FEATURES if c in features]
    num = [c for c in features if c not in cat]
    transformers = []
    if cat:
        transformers.append(("cat", make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            OneHotEncoder(handle_unknown="ignore", sparse_output=False)), cat))
    if num:
        transformers.append(("num", SimpleImputer(strategy="median", add_indicator=True), num))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_estimator(algo):
    rs = config.RANDOM_STATE
    if algo == "CatBoost":
        return optional_model("CatBoost")(
            iterations=1000, depth=7, learning_rate=0.035, loss_function="RMSE", l2_leaf_reg=5,
            random_seed=rs, thread_count=-1, verbose=False, allow_writing_files=False)
    if algo == "XGBoost":
        return optional_model("XGBoost")(
            n_estimators=1000, max_depth=6, learning_rate=0.03, min_child_weight=3,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=3.0,
            objective="reg:squarederror", eval_metric="mae", tree_method="hist",
            random_state=rs, n_jobs=-1, verbosity=0)
    if algo == "LightGBM":
        return optional_model("LightGBM")(
            n_estimators=1000, learning_rate=0.03, num_leaves=31, max_depth=-1,
            min_child_samples=20, subsample=0.85, subsample_freq=1, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=3.0, random_state=rs, n_jobs=-1, verbosity=-1)
    if algo == "HistGradientBoosting":
        return HistGradientBoostingRegressor(
            max_iter=600, learning_rate=0.04, max_leaf_nodes=31, min_samples_leaf=15,
            l2_regularization=2.0, early_stopping=True, random_state=rs)
    if algo == "ExtraTrees":
        return ExtraTreesRegressor(n_estimators=800, max_features=0.9, min_samples_leaf=2,
                                   n_jobs=-1, random_state=rs)
    if algo == "RandomForest":
        return RandomForestRegressor(n_estimators=700, max_features=0.9, min_samples_leaf=2,
                                     n_jobs=-1, random_state=rs)
    if algo == "GradientBoosting":
        return GradientBoostingRegressor(n_estimators=500, learning_rate=0.03, max_depth=3,
                                         min_samples_leaf=5, loss="huber", random_state=rs)
    if algo == "LinearRegression":
        return LinearRegression()
    raise ValueError(f"Unknown algorithm: {algo}")


def build_pipeline(algo, features):
    if algo != "Stacking Ensemble":
        return make_pipeline(build_preprocessor(features), build_estimator(algo))

    rs = config.RANDOM_STATE
    base = [
        ("extra_trees", ExtraTreesRegressor(n_estimators=400, max_features=0.9, min_samples_leaf=2,
                                            n_jobs=1, random_state=rs)),
        ("hist_gradient", HistGradientBoostingRegressor(max_iter=350, learning_rate=0.04, max_leaf_nodes=31,
                                                        min_samples_leaf=15, l2_regularization=2.0,
                                                        random_state=rs)),
        ("gradient_boost", GradientBoostingRegressor(n_estimators=300, learning_rate=0.03, max_depth=3,
                                                     min_samples_leaf=5, loss="huber", random_state=rs)),
    ]
    return StackingRegressor(
        estimators=[(name, make_pipeline(build_preprocessor(features), est)) for name, est in base],
        final_estimator=Ridge(alpha=2.0), cv=5, n_jobs=-1, passthrough=False)


# ==================================================
# Train / evaluate
# ==================================================
@dataclass
class TrainResult:
    results: pd.DataFrame
    predictions: dict
    features: list
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    models: dict
    notes: list = field(default_factory=list)
    report_path: str = None

    @property
    def best_name(self):
        return str(self.results.iloc[0]["Algorithm"])

    @property
    def best_model(self):
        return self.models[self.best_name]

    def seconds_metrics(self, algo=None):
        """MAE / RMSE ในหน่วยวินาที (exp กลับจาก log)"""
        algo = algo or self.best_name
        actual = np.exp(self.test_df[TARGET].to_numpy())
        pred = np.exp(self.predictions[algo])
        return {"Algorithm": algo,
                "MAE_s": mean_absolute_error(actual, pred),
                "RMSE_s": float(np.sqrt(mean_squared_error(actual, pred)))}

    def predict(self, df, algo=None):
        """ทำนาย Duration (วินาที) จาก DataFrame ดิบ (เช่นผลจาก build_sector_table)"""
        X = df.copy()
        for col in CATEGORICAL_FEATURES:
            if col in X.columns:
                X[col] = X[col].astype("string").str.upper().str.strip()
        X, _ = add_prev_trend_features(X)
        X = add_tyre_age_interaction(X)
        for col in self.features:
            if col not in X.columns:
                X[col] = np.nan
            elif col not in CATEGORICAL_FEATURES:
                X[col] = _to_float(X[col])
        return np.exp(self.models[algo or self.best_name].predict(X[self.features]))


def run_experiments(df_ml, features, algorithms=None, verbose=True):
    """-> (results, preds, train_df, test_df, models, errors)"""
    train_idx, test_idx = train_test_split(np.arange(len(df_ml)), test_size=TEST_SIZE,
                                           random_state=config.RANDOM_STATE)
    train_df, test_df = df_ml.iloc[train_idx].copy(), df_ml.iloc[test_idx].copy()
    y_train, y_test = train_df[TARGET], test_df[TARGET]

    rows, preds, models, errors = [], {}, {}, []
    for algo in algorithms:
        verbose and print(f"Training: {algo}")
        try:
            model = build_pipeline(algo, features)
            model.fit(train_df[features], y_train)
            pred = np.asarray(model.predict(test_df[features]), dtype=float)
        except Exception as exc:
            errors.append(f"{algo}: {exc}")
            print(f"Skipped {algo}: {exc}")
            continue
        preds[algo], models[algo] = pred, model
        rows.append({"Algorithm": algo,
                     "MAE": mean_absolute_error(y_test, pred),
                     "RMSE": float(np.sqrt(mean_squared_error(y_test, pred))),
                     "R2": r2_score(y_test, pred)})

    if not rows:
        raise RuntimeError("ไม่มีโมเดลใด train สำเร็จ:\n" + "\n".join(errors))
    results = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    return results, preds, train_df, test_df, models, errors


def train_models(df, algorithms=None, report_path="model_report.html", open_report=False, verbose=True,
                 outlier_k=OUTLIER_K):
    """prepare_data -> train ทุกโมเดล -> (optional) HTML report -> TrainResult

    algorithms  : list ของชื่อโมเดล (None = ทุกตัวที่ติดตั้งไว้)
    report_path : None = ไม่สร้าง report
    open_report : True = เปิด report ใน browser
    outlier_k   : ตัด outlier ด้านช้าของ Duration (None = ไม่ตัด)
    """
    df_ml, features, notes = prepare_data(df, outlier_k=outlier_k)
    available, import_errors = available_algorithms()
    if algorithms is None:
        algorithms = available
        notes += import_errors
    else:
        unknown = [a for a in algorithms if a not in ALL_ALGORITHMS]
        if unknown:
            raise ValueError(f"Unknown algorithms {unknown}; choose from {ALL_ALGORITHMS}")

    results, preds, train_df, test_df, models, errors = run_experiments(df_ml, features, algorithms, verbose)
    notes += [f"Model training failed: {e}" for e in errors]
    out = TrainResult(results, preds, features, train_df, test_df, models, notes)

    if report_path:
        from .report import write_report
        out.report_path = write_report(out, df_ml, report_path, open_browser=open_report)

    if verbose:
        print(results.to_string(index=False))
        sec = out.seconds_metrics()
        print(f"\nBest model: {out.best_name} | MAE {sec['MAE_s']:.4f} s | RMSE {sec['RMSE_s']:.4f} s")
        if out.report_path:
            print(f"Report: {out.report_path}")
    return out
