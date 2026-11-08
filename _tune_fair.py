"""Fair, tuned comparison: batch vs online tree/linear models.

Protocol (matches how the batch RF was evaluated):
  - Tune each online model's hyperparameters on a TEMPORAL validation
    (warm=2023 -> val=2024, frozen prediction).
  - Re-warm on full 23+24 and evaluate FROZEN on 2025 (same setting as batch RF).
  - Also report the online-prequential (predict-then-learn each 2025 lap) number.
All tuning and final runs use the FULL data (no subsampling).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from river import ensemble, forest, tree
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from f1pit.train import TARGET
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _build_matrix, _features, _to_dicts

ROOT = Path(__file__).resolve().parent


def sec_mae(y_log, pred_log):
    pred_log = np.asarray(pred_log, float)
    if not np.isfinite(pred_log).all():
        return np.inf          # diverged config -> reject it, don't crash the grid
    p = np.exp(pred_log)
    if not np.isfinite(p).all():
        return np.inf
    a = np.exp(np.asarray(y_log, float))
    return float(mean_absolute_error(a, p))


def prep(raw):
    raw = raw.sort_values("Time_Start").reset_index(drop=True)
    ml, feats, _, _ = _prepare_lab(raw, [])
    feats = [c for c in _features(feats, ml)]
    return ml, feats


def build_matrices():
    m23, f23 = prep(pd.read_parquet(ROOT / "data_2023.parquet"))
    m24, f24 = prep(pd.read_parquet(ROOT / "data_2024.parquet"))
    m25, f25 = prep(pd.read_parquet(ROOT / "data_2025.parquet"))

    feats = [c for c in f23 if c in f24 and c in f25]
    oh = sorted(pd.concat([m23["Track_Status"], m24["Track_Status"], m25["Track_Status"]])
                .astype("string").fillna("UNKNOWN").unique())

    X23 = _build_matrix(m23, feats, oh)
    X24 = _build_matrix(m24, feats, oh)
    X25 = _build_matrix(m25, feats, oh)
    cols = X23.columns
    med = X23.median()
    for X in (X23, X24, X25):
        X.fillna(med, inplace=True)
        X.columns = cols
    y23 = m23[TARGET].to_numpy(float)
    y24 = m24[TARGET].to_numpy(float)
    y25 = m25[TARGET].to_numpy(float)

    # scalers: tuning (fit 2023), final (fit 23+24)
    sc_tune = StandardScaler().fit(X23)
    s23 = pd.DataFrame(sc_tune.transform(X23.to_numpy(float)), columns=cols)
    s24 = pd.DataFrame(sc_tune.transform(X24.to_numpy(float)), columns=cols)
    warm_full = pd.concat([X23, X24], ignore_index=True)
    sc_full = StandardScaler().fit(warm_full)
    s_warm = pd.DataFrame(sc_full.transform(warm_full.to_numpy(float)), columns=cols)
    s25 = pd.DataFrame(sc_full.transform(X25.to_numpy(float)), columns=cols)
    y_warm = np.concatenate([y23, y24])

    return (s23, y23, s24, y24, s_warm, y_warm, s25, y25, m25, feats)


def frozen_stream_mae(make_model, warm_df, warm_y, test_df, test_y):
    m = make_model()
    for x, y in _to_dicts(warm_df, warm_y):
        m.learn_one(x, y)
    preds = np.array([m.predict_one(x) for x, _ in _to_dicts(test_df, test_y)])
    return sec_mae(test_y, preds)


def prequential_mae(make_model, warm_df, warm_y, test_df, test_y, group_order, grp):
    m = make_model()
    for x, y in _to_dicts(warm_df, warm_y):
        m.learn_one(x, y)
    stream = list(_to_dicts(test_df, test_y))
    preds = np.empty(len(stream))
    for (sk, lp) in group_order:
        idx = grp.get_group((sk, lp)).index.to_numpy()
        for j in idx:
            preds[j] = m.predict_one(stream[j][0])
        for j in idx:
            m.learn_one(*stream[j])
    return sec_mae(test_y, preds)


def prequential_sgd_mae(make_model, warm_X, warm_y, test_X, test_y, group_order, grp):
    m = make_model().fit(warm_X.to_numpy(float), warm_y)
    Xn = test_X.to_numpy(float)
    preds = np.empty(len(test_y))
    for (sk, lp) in group_order:
        idx = grp.get_group((sk, lp)).index.to_numpy()
        preds[idx] = m.predict(Xn[idx])
        m.partial_fit(Xn[idx], test_y[idx])
    return sec_mae(test_y, preds)


def srp(**kw):
    return ensemble.SRPRegressor(model=tree.HoeffdingTreeRegressor(leaf_prediction="mean"), **kw)


def main():
    s23, y23, s24, y24, s_warm, y_warm, s25, y25, m25, feats = build_matrices()
    print(f"features={len(feats)}  warm23={len(s23)} val24={len(s24)} full_warm={len(s_warm)} test25={len(s25)}")

    # full data — no subsampling: tune on all of 2023 -> validate on all of 2024
    warm_df = s23; warm_y = y23
    val_df = s24;   val_y = y24

    sgd_grid = [dict(loss="huber", epsilon=0.5, penalty="l2", alpha=a,
                     learning_rate="constant", eta0=e, max_iter=1000, tol=1e-4, random_state=42)
                for a in (1e-3, 1e-2) for e in (1e-4, 1e-3)]
    ht_grid = [dict(leaf_prediction=l, grace_period=g)
               for l in ("mean", "adaptive") for g in (50, 200)]
    arf_grid = [dict(n_models=10, max_features=mf, leaf_prediction=l, aggregation_method="mean", seed=42)
                for mf in ("sqrt", 1.0) for l in ("mean", "adaptive")]
    srp_grid = [dict(n_models=10, subspace_size=ss, training_method=tm, seed=42)
                for ss in (0.6, 1.0) for tm in ("patches", "subspaces")]

    results = {}

    # ---- SGD (sklearn, frozen fit on warm) ----
    best = (None, np.inf)
    for kw in sgd_grid:
        try:
            m = SGDRegressor(**kw).fit(warm_df.to_numpy(float), warm_y)
            e = sec_mae(val_y, m.predict(val_df.to_numpy(float)))
        except Exception as ex:
            e = np.inf
            print(f"  SGD {kw} ERROR {ex}")
        if e < best[1]:
            best = (kw, e)
    results["SGD"] = best
    print(f"[SGD] best val={best[1]:.4f}  {best[0]}")

    # ---- Hoeffding tree ----
    best = (None, np.inf)
    for kw in ht_grid:
        try:
            e = frozen_stream_mae(lambda k=kw: tree.HoeffdingTreeRegressor(**k), warm_df, warm_y, val_df, val_y)
            if e < best[1]:
                best = (kw, e)
        except Exception as ex:
            print(f"  HT {kw} ERROR {ex}")
    results["HoeffdingTree"] = best
    print(f"[HoeffdingTree] best val={best[1]:.4f}  {best[0]}")

    # ---- ARF ----
    best = (None, np.inf)
    for kw in arf_grid:
        try:
            e = frozen_stream_mae(lambda k=kw: forest.ARFRegressor(**k), warm_df, warm_y, val_df, val_y)
            print(f"  ARF {kw} -> {e:.4f}")
            if e < best[1]:
                best = (kw, e)
        except Exception as ex:
            print(f"  ARF {kw} ERROR {ex}")
    results["ARF"] = best
    print(f"[ARF] best val={best[1]:.4f}  {best[0]}")

    # ---- SRP ----
    best = (None, np.inf)
    for kw in srp_grid:
        try:
            e = frozen_stream_mae(lambda k=kw: srp(**k), warm_df, warm_y, val_df, val_y)
            print(f"  SRP {kw} -> {e:.4f}")
            if e < best[1]:
                best = (kw, e)
        except Exception as ex:
            print(f"  SRP {kw} ERROR {ex}")
    results["SRP"] = best
    print(f"[SRP] best val={best[1]:.4f}  {best[0]}")

    # ---- final full frozen eval (23+24 -> 2025) ----
    print("\n=== FINAL FROZEN (warm=23+24, test=2025) ===")
    m = SGDRegressor(**results["SGD"][0]).fit(s_warm.to_numpy(float), y_warm)
    print(f"  SGD           : {sec_mae(y25, m.predict(s25.to_numpy(float))):.4f}")
    print(f"  HoeffdingTree : {frozen_stream_mae(lambda k=results['HoeffdingTree'][0]: tree.HoeffdingTreeRegressor(**k), s_warm, y_warm, s25, y25):.4f}")
    print(f"  ARF           : {frozen_stream_mae(lambda k=results['ARF'][0]: forest.ARFRegressor(**k), s_warm, y_warm, s25, y25):.4f}")
    print(f"  SRP           : {frozen_stream_mae(lambda k=results['SRP'][0]: srp(**k), s_warm, y_warm, s25, y25):.4f}")

    # ---- online prequential (drift task) ----
    grp = m25.groupby(["Session_Key", "Lap"], sort=False)
    group_order = grp["Time_Start"].min().sort_values().index
    print("\n=== ONLINE PREQUENTIAL (adapt each 2025 lap) ===")
    print(f"  SGD           : {prequential_sgd_mae(lambda k=results['SGD'][0]: SGDRegressor(**k), s_warm, y_warm, s25, y25, group_order, grp):.4f}")
    print(f"  HoeffdingTree : {prequential_mae(lambda k=results['HoeffdingTree'][0]: tree.HoeffdingTreeRegressor(**k), s_warm, y_warm, s25, y25, group_order, grp):.4f}")
    print(f"  ARF           : {prequential_mae(lambda k=results['ARF'][0]: forest.ARFRegressor(**k), s_warm, y_warm, s25, y25, group_order, grp):.4f}")
    print(f"  SRP           : {prequential_mae(lambda k=results['SRP'][0]: srp(**k), s_warm, y_warm, s25, y25, group_order, grp):.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
