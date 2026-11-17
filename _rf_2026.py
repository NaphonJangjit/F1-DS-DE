"""Train RF on 2023+2024+2025, test on 2026 (finished sessions only).

Step 1 & 2 build the 2026 sector table through the package pipeline (slow, resumable):
  ctx_2026.pkl  <- build_track_context(2026)
  data_2026.parquet <- build_dataset(2026, ctx=ctx)
Future GPs (no laps yet) are skipped automatically.

Step 3 trains the frozen RF on 2023-25 and evaluates on 2026.

SCRATCH.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

import f1pit
from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec(y_log, p_log):
    a = np.exp(np.asarray(y_log, float)); p = np.exp(np.asarray(p_log, float))
    return float(mean_absolute_error(a, p)), float(np.sqrt(mean_squared_error(a, p)))


def green_mask(df):
    d = df[["Session_Key", "Driver", "Lap", "Track_Status"]].copy()
    is_sc = d["Track_Status"].isin(SC_SET).to_numpy()
    laps = (d.assign(_is_sc=is_sc)
            .groupby(["Session_Key", "Driver", "Lap"], sort=False)["_is_sc"].max()
            .reset_index().sort_values(["Session_Key", "Driver", "Lap"]))
    laps["_prev_sc"] = laps.groupby(["Session_Key", "Driver"])["_is_sc"].shift(1).fillna(False)
    merged = d[["Session_Key", "Driver", "Lap"]].merge(
        laps[["Session_Key", "Driver", "Lap", "_prev_sc"]],
        on=["Session_Key", "Driver", "Lap"], how="left")
    return ~(is_sc | merged["_prev_sc"].to_numpy().astype(bool))


def build_2026():
    ctx_path = ROOT / "ctx_2026.pkl"
    if ctx_path.exists():
        log("loading cached ctx_2026.pkl")
        ctx = f1pit.TrackContext.load(ctx_path)
    else:
        log("build_track_context(2026) — slow (fetch + track reconstruction + clustering) ...")
        t = time.time()
        ctx = f1pit.build_track_context(2026, verbose=True, plot=False)
        ctx.save(ctx_path)
        log(f"saved ctx_2026.pkl ({time.time()-t:.0f}s)")

    data_path = ROOT / "data_2026.parquet"
    if data_path.exists():
        log("loading cached data_2026.parquet")
        data = pd.read_parquet(data_path)
    else:
        log("build_dataset(2026) — slow (loops session x driver) ...")
        t = time.time()
        data = f1pit.build_dataset(2026, ctx=ctx, save_path=str(data_path), verbose=False)
        log(f"data_2026 rows={len(data):,} ({time.time()-t:.0f}s)")

    return data


def main():
    t0 = time.time()
    data26 = build_2026()
    log(f"2026: rows={len(data26):,}  sessions={data26['Session_Key'].nunique()}  "
        f"countries={sorted(data26['Country'].dropna().unique())}")
    log(f"2026 date range: {data26['Time_Start'].min()} .. {data26['Time_Start'].max()}")

    log("loading train (2023+24+25) ...")
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet"),
                    pd.read_parquet(ROOT / "data_2025.parquet")], ignore_index=True)

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(data26, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    log(f"train rows={len(tr_ml):,}  test rows={len(te_ml):,}  features={len(feats)}")

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    Xtr = pre.transform(tr_ml[feats])
    Xte = pre.transform(te_ml[feats])
    ytr = tr_ml[TARGET].to_numpy(float)
    yte = te_ml[TARGET].to_numpy(float)

    log("training RF (frozen config) ...")
    t = time.time()
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(Xtr, ytr)
    p = rf.predict(Xte)
    log(f"fit ({time.time()-t:.0f}s)")

    full_mae, full_rmse = sec(yte, p)
    g = green_mask(te_ml)
    green_mae, green_rmse = sec(yte[g], p[g])
    print(f"\n=== RF 2023-25 -> 2026 ===\nFULL  : MAE={full_mae:.4f}s  RMSE={full_rmse:.4f}s\n"
          f"GREEN : MAE={green_mae:.4f}s  RMSE={green_rmse:.4f}s")

    # per-race
    country_of = data26.groupby("Session_Key")["Country"].first().to_dict()
    print("\nper-race MAE_s:")
    rows = []
    for sk, gdf in te_ml.groupby("Session_Key", sort=False):
        idx = gdf.index.to_numpy()
        m = sec(yte[idx], p[idx])[0]
        rows.append((str(country_of.get(int(sk), sk)), len(idx), m))
    rows.sort(key=lambda r: -r[2])
    print(f"{'race':22s} {'rows':>6s} {'MAE':>8s}")
    for name, n, m in rows:
        print(f"{name:22s} {n:>6d} {m:8.4f}")

    print(f"\ntotal={time.time()-t0:.0f}s  DONE")


if __name__ == "__main__":
    main()
