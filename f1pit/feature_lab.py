"""Feature-engineering lab (does NOT modify core logic).

โหลด dataset ที่ build แล้ว -> ใช้ prepare_data เดิม -> ทดลอง/วิเคราะห์ feature:
  1. persistence baseline (เพดานล่างของ MAE)
  2. log-align duration features (ให้ตรงกับ target Log_Duration)
  3. เพิ่ม Lap (fuel/race-phase) + Sector
  4. all-x-all crossing (PolynomialFeatures) + feature selection (Lasso)
ใช้แค่โมเดลง่าย (LinearRegression / Ridge / Lasso)

Run:  python -m f1pit.feature_lab
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from f1pit import config
from f1pit.train import TARGET, TEST_SIZE, prepare_data

DATA_PATH = Path(__file__).resolve().parent.parent / "data_2023.parquet"

# feature ที่เป็น "วินาที" -> log ให้ align กับ target Log_Duration (= ln Duration)
DURATION_LIKE = ["Prev_Duration", "Prev_Sector_Duration", "Clean_Pace", "Best_Duration_So_Far"]


def _mae_s(y_log, p_log):
    return mean_absolute_error(np.exp(y_log), np.exp(p_log))


def _rmse_s(y_log, p_log):
    return float(np.sqrt(mean_squared_error(np.exp(y_log), np.exp(p_log))))


def build_X(df, features, *, log_duration=True, extra_num=(), extra_cat=(), poly=None):
    """df_ml + feature list -> (X, names) พร้อม impute + one-hot categorical"""
    num = [c for c in features if c in df.columns and c != "Track_Status"]
    for c in extra_num:
        if c in df.columns and c not in num:
            num.append(c)
    X = df[num].apply(pd.to_numeric, errors="coerce").copy()
    if log_duration:
        for c in DURATION_LIKE:
            if c in X.columns:
                X[c] = np.log(X[c].clip(lower=1e-6))
    X = X.fillna(X.median())

    cats = [c for c in (["Track_Status", *extra_cat]) if c in df.columns]
    if cats:
        d = pd.get_dummies(df[cats].astype("string").fillna("UNKNOWN"))
        X = pd.concat([X.reset_index(drop=True), d.reset_index(drop=True)], axis=1)
    names = list(X.columns)
    X = X.to_numpy(dtype=float)
    if poly is not None:
        pf = PolynomialFeatures(degree=poly, include_bias=False)
        X = pf.fit_transform(X)
        names = pf.get_feature_names_out(names)
    return X, names


def _fit_eval(X, ytr, yte, tr_idx, te_idx, model, label):
    scaler = StandardScaler().fit(X[tr_idx])   # scale บน train เท่านั้น (กัน leakage)
    Xs = scaler.transform(X)
    model.fit(Xs[tr_idx], ytr)
    p = model.predict(Xs[te_idx])
    print(f"  {label:44s} MAE_s={_mae_s(yte, p):.4f}  RMSE_s={_rmse_s(yte, p):.4f}  R2(log)={r2_score(yte, p):.5f}")
    return p


def main():
    df = pd.read_parquet(DATA_PATH)
    df_ml, features, notes = prepare_data(df)
    print("=" * 78)
    print(f"rows={len(df_ml):,}  features={len(features)}  target={TARGET}")
    print("prepare notes:", notes)

    y = df_ml[TARGET].to_numpy(dtype=float)
    tr_idx, te_idx = train_test_split(np.arange(len(df_ml)), test_size=TEST_SIZE,
                                      random_state=config.RANDOM_STATE)
    ytr, yte = y[tr_idx], y[te_idx]

    print("=" * 78)
    print("BASELINES (holdout 20%)")
    # 0. persistence: ทำนาย = sector เดียวกันของ lap ก่อนหน้า
    prev = np.log(df_ml["Prev_Duration"].to_numpy(dtype=float))
    print(f"  {'persistence (prev same-sector)':44s} MAE_s={_mae_s(yte, prev[te_idx]):.4f}  "
          f"RMSE_s={_rmse_s(yte, prev[te_idx]):.4f}")

    # 1. linear บน feature เดิม (raw seconds)
    Xb, _ = build_X(df_ml, features, log_duration=False)
    _fit_eval(Xb, ytr, yte, tr_idx, te_idx, LinearRegression(), "Linear (base, raw durations)")

    # 2. linear บน feature เดิม + log durations
    Xl, nl = build_X(df_ml, features, log_duration=True)
    _fit_eval(Xl, ytr, yte, tr_idx, te_idx, LinearRegression(), "Linear (base, log-durations)")

    print("=" * 78)
    print("FEATURE IMPORTANCE (Ridge standardized |coef|, base+log)")
    scaler = StandardScaler().fit(Xl[tr_idx])
    Xs = scaler.transform(Xl)
    ridge = Ridge(alpha=1.0).fit(Xs[tr_idx], ytr)
    imp = pd.Series(np.abs(ridge.coef_), index=nl).sort_values(ascending=False)
    for name, v in imp.head(22).items():
        print(f"  {name:42s} {v:.4f}")

    print("=" * 78)
    print("ADD Lap (fuel/race-phase) + Sector")
    Xe, ne = build_X(df_ml, features, log_duration=True, extra_num=["Lap"], extra_cat=["Sector"])
    _fit_eval(Xe, ytr, yte, tr_idx, te_idx, Ridge(alpha=1.0), "Ridge (log + Lap + Sector)")

    print("=" * 78)
    print("ALL-X-ALL CROSSING (PolyFeatures degree=2) + Lasso selection")
    Xp, pnames = build_X(df_ml, features, log_duration=True, extra_num=["Lap"], extra_cat=["Sector"], poly=2)
    print(f"  polynomial feature count: {len(pnames)}")
    scaler_p = StandardScaler().fit(Xp[tr_idx])
    Xp_s = scaler_p.transform(Xp)

    for alpha in (1e-3, 1e-2):
        lasso = Lasso(alpha=alpha, max_iter=20000, random_state=0).fit(Xp_s[tr_idx], ytr)
        sel = lasso.coef_ != 0
        k = int(sel.sum())
        print(f"  Lasso alpha={alpha}: selected {k}/{len(pnames)}")
        if k == 0:
            continue
        Xsel = Xp_s[:, sel]
        r = Ridge(alpha=1.0).fit(Xsel[tr_idx], ytr)
        p = r.predict(Xsel[te_idx])
        print(f"  {'-> Ridge on selected':44s} MAE_s={_mae_s(yte, p):.4f}  RMSE_s={_rmse_s(yte, p):.4f}  "
              f"R2(log)={r2_score(yte, p):.5f}")
        order = np.argsort(-np.abs(lasso.coef_[sel]))
        print("   top selected (by |lasso coef|):")
        for i in order[:16]:
            print(f"      {str(pnames[sel][i]):54s} {lasso.coef_[sel][i]:+.4f}")

    print("=" * 78)
    print("TREE MODELS (simple sklearn): does feature engineering help?")
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

    def tree_eval(X, label, est=None):
        est = est or ExtraTreesRegressor(n_estimators=300, max_features=0.9, min_samples_leaf=2,
                                         n_jobs=-1, random_state=config.RANDOM_STATE)
        est.fit(X[tr_idx], ytr)
        p = est.predict(X[te_idx])
        print(f"  {label:46s} MAE_s={_mae_s(yte, p):.4f}  RMSE_s={_rmse_s(yte, p):.4f}")

    # ratio features: "วิ่งช้ากว่า/เร็วกว่า pace ของตัวเองแค่ไหน" (relative pace)
    df_ml["Rel_Prev_vs_Clean"] = df_ml["Prev_Duration"] / df_ml["Clean_Pace"].clip(lower=1e-6)
    df_ml["Rel_Prev_vs_Best"] = df_ml["Prev_Duration"] / df_ml["Best_Duration_So_Far"].clip(lower=1e-6)
    extra = ["Lap", "Rel_Prev_vs_Clean", "Rel_Prev_vs_Best"]

    Xt0, _ = build_X(df_ml, features, log_duration=False)
    tree_eval(Xt0, "ExtraTrees (base, raw)  [reference]")
    Xt1, _ = build_X(df_ml, features, log_duration=True, extra_num=["Lap"], extra_cat=["Sector"])
    tree_eval(Xt1, "ExtraTrees (log + Lap + Sector)")
    Xt2, _ = build_X(df_ml, features, log_duration=True, extra_num=extra, extra_cat=["Sector"])
    tree_eval(Xt2, "ExtraTrees (log + Lap + ratios + Sector)")
    Xt3, _ = build_X(df_ml, features, log_duration=True, extra_num=extra, extra_cat=["Sector"])
    tree_eval(Xt3, "HistGradBoost (log + Lap + ratios + Sector)",
              HistGradientBoostingRegressor(max_iter=600, learning_rate=0.04, max_leaf_nodes=31,
                                            min_samples_leaf=15, l2_regularization=2.0,
                                            early_stopping=True, random_state=config.RANDOM_STATE))

    # feature selection สำหรับ tree: ตัด feature ที่ importance ต่ำ
    print("  -- ExtraTrees feature selection (drop low-importance) --")
    Xsel, cols = build_X(df_ml, features, log_duration=True, extra_num=extra, extra_cat=["Sector"])
    et = ExtraTreesRegressor(n_estimators=300, max_features=0.9, min_samples_leaf=2,
                             n_jobs=-1, random_state=config.RANDOM_STATE).fit(Xsel[tr_idx], ytr)
    imp2 = pd.Series(et.feature_importances_, index=cols).sort_values(ascending=False)
    print("   top-15 features by ExtraTrees importance:")
    for name, v in imp2.head(15).items():
        print(f"     {name:46s} {v:.4f}")
    for keep_k in (10, 25, 50):
        keep = imp2.head(keep_k).index.tolist()
        mask = np.isin(cols, keep)
        et2 = ExtraTreesRegressor(n_estimators=300, max_features=0.9, min_samples_leaf=2,
                                  n_jobs=-1, random_state=config.RANDOM_STATE).fit(Xsel[:, mask][tr_idx], ytr)
        p = et2.predict(Xsel[:, mask][te_idx])
        print(f"   ExtraTrees top-{keep_k} features: MAE_s={_mae_s(yte, p):.4f}")

    print("=" * 78)
    print("DONE")


if __name__ == "__main__":
    main()
