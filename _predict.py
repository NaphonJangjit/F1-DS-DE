"""Load saved forest-of-forests models and predict Duration WITHOUT retraining.

Usage:
    python _predict.py                          # predict data_2025_out.parquet -> predictions_2025.parquet
    from _predict import load_models, predict   # use as a library

Requires the models saved by _fforest.py under model/v2/ (sub_<channel>.pkl + main.pkl + meta.json).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from f1pit.wind_rain_lab import _prepare_lab
import _fforest as ff

ROOT = Path(__file__).resolve().parent
VERSION = "v2"
MODEL_DIR = ROOT / "model" / VERSION


def load_models(version=VERSION):
    """-> (meta, subs, main). subs: {outcome_col: {pre, rf, feats, variant, prev}}."""
    d = ROOT / "model" / version
    meta = json.loads((d / "meta.json").read_text())
    subs = {}
    for prev, out in meta["channels"]:
        subs[out] = joblib.load(d / f"sub_{out}.pkl")
    main = joblib.load(d / "main.pkl")
    return meta, subs, main


def predict(raw, meta, subs, main):
    """raw: DataFrame from data_*_out.parquet -> same frame + Pred_<channel> + Pred_Duration_s (sec).

    `raw` must contain the point-in-time features (Prev_*, Tyre_Age_*, weather, traffic, Lap,
    Track_Status, geometry) and Duration (for Log_Duration + outlier cleaning).
    """
    df, _, _, _ = _prepare_lab(raw, meta["extra"])
    df = ff._engineer(df)

    for prev, out in meta["channels"]:
        m = subs[out]
        p = m["rf"].predict(m["pre"].transform(df[m["feats"]]))
        if m["variant"] == "delta":
            p = df[prev].to_numpy(float) + p
        elif m["variant"] == "delta_slog":
            p = df[prev].to_numpy(float) + ff._slog_inv(p)
        elif m["variant"] == "log":
            p = np.exp(p)
        df[f"Pred_{out}"] = p

    df["Pred_Duration_s"] = df["Prev_Duration"].to_numpy(float) * np.exp(
        main["rf"].predict(main["pre"].transform(df[main["feats"]])))
    return df


if __name__ == "__main__":
    meta, subs, main = load_models()
    raw = pd.read_parquet(ROOT / "data_2025_out.parquet")
    out = predict(raw, meta, subs, main)

    cols = ["Driver", "Session_Key", "Lap", "Sector", "Duration", "Pred_Duration_s"]
    print(out[cols].head(10))
    err = (out["Pred_Duration_s"] - out["Duration"]).abs()
    print(f"\nrows={len(out):,}  Pred_Duration_s mean={out['Pred_Duration_s'].mean():.3f}s  "
          f"MAE={err.mean():.4f}s  RMSE={np.sqrt((err ** 2).mean()):.4f}s")

    print(f"\n=== 6-condition MAE (seconds) ===")
    bd = ff._eval_breakdown(np.log(out["Duration"].to_numpy(float)),
                            np.log(out["Pred_Duration_s"].to_numpy(float)), out)
    for name, v in bd.items():
        mae = v["mae_s"] if v["mae_s"] is not None else "—"
        print(f"  {name:18s} n={v['n']:>7,}  MAE_s={mae}")

    out_path = ROOT / "predictions_2025.parquet"
    out.to_parquet(out_path)
    print(f"Saved -> {out_path.name}")
