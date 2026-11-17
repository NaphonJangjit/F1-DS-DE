"""How much of the 2025 error is rain?

Frozen RF (2023+24 -> 2025). Same model, same predictions; we only change which
test sectors we score:
  full        : all sectors
  dry         : Rainfall != 1 (rain removed, SC/VSC kept)
  wet         : Rainfall == 1
  dry+green   : rain AND SC/VSC removed (cleanest subset)
Then rain's contribution = full - dry.

SCRATCH.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

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


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


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


def main():
    t0 = time.time()
    log("loading + preparing ...")
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ml[feats]),
                                                    tr_ml[TARGET].to_numpy(float))
    p = rf.predict(pre.transform(te_ml[feats]))
    yte = te_ml[TARGET].to_numpy(float)
    log(f"fitted. features={len(feats)} test={len(yte):,}")

    wet = (te_ml["Rainfall"].to_numpy() == 1)
    dry = ~wet
    green = green_mask(te_ml)          # not SC/VSC, not transition lap
    dry_green = dry & green

    full = sec_mae(yte, p)
    d = sec_mae(yte[dry], p[dry])
    w = sec_mae(yte[wet], p[wet])
    dg = sec_mae(yte[dry_green], p[dry_green])

    print("\n" + "=" * 66)
    print(f"{'subset':28s} {'n':>9s} {'MAE(s)':>9s} {'share':>8s}")
    print("-" * 66)
    print(f"{'full (all sectors)':28s} {len(yte):>9,} {full:>9.4f} {'100%':>8s}")
    print(f"{'dry (rain removed)':28s} {int(dry.sum()):>9,} {d:>9.4f} {dry.mean()*100:>7.1f}%")
    print(f"{'wet (rain only)':28s} {int(wet.sum()):>9,} {w:>9.4f} {wet.mean()*100:>7.1f}%")
    print(f"{'dry + green (no rain, no SC/VSC)':28s} {int(dry_green.sum()):>9,} {dg:>9.4f} {dry_green.mean()*100:>7.1f}%")
    print("=" * 66)
    print(f"rain contribution to overall MAE = full - dry = {full - d:.4f}s  "
          f"({(full - d)/full*100:.1f}% of error)")
    print(f"wet vs dry error ratio = {w/d:.2f}x")
    print(f"({time.time()-t0:.0f}s)")

    out = {"full": full, "dry": d, "wet": w, "dry_green": dg,
           "n": {"full": int(len(yte)), "dry": int(dry.sum()),
                 "wet": int(wet.sum()), "dry_green": int(dry_green.sum())},
           "rain_contrib_abs": full - d, "rain_contrib_pct": (full - d) / full * 100}
    pd.Series(out).to_json(ROOT / "rain_effect.json")
    log("saved -> rain_effect.json")


if __name__ == "__main__":
    main()
