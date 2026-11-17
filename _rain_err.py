"""Rain-session error breakdown for the frozen RF (2023+24 -> 2025).

For each 2025 session flagged wet (Rainfall==1 anywhere):
  - how many laps it rained
  - MAE overall, and MAE on rain sectors only
  - MAE on rain sectors, split by tyre compound (INTERMEDIATE vs dry tyres)

Note: OpenF1 calls the wet tyre "INTERMEDIATE"; there is no "WET" compound in 2025
(no full-wet running). SCRATCH — read-only.
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


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def main():
    t0 = time.time()
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
    pred = rf.predict(pre.transform(te_ml[feats]))
    yte = te_ml[TARGET].to_numpy(float)
    print(f"RF frozen trained ({time.time()-t0:.0f}s). features={len(feats)}", flush=True)

    country = te.groupby("Session_Key")["Country"].first().to_dict()
    rain_sess = [sk for sk, r in te.groupby("Session_Key")["Rainfall"].max().items() if r == 1]

    rows = []
    for sk in rain_sess:
        sk = int(sk)
        gm = te_ml["Session_Key"].to_numpy() == sk
        wet = gm & (te_ml["Rainfall"].to_numpy() == 1)
        total_laps = int(te_ml.loc[gm, "Lap"].nunique())
        rain_laps = int(te_ml.loc[wet, "Lap"].nunique())
        mae_all = sec_mae(yte[gm], pred[gm])
        mae_wet = sec_mae(yte[wet], pred[wet])

        print(f"\n=== {country.get(sk, sk)} (Session_Key {sk}) ===")
        print(f"  laps: total={total_laps}  rained={rain_laps}  rain_rows={int(wet.sum()):,}")
        print(f"  MAE: whole session={mae_all:.3f}s   rain sectors only={mae_wet:.3f}s")
        print(f"  {'tyre':14s} {'rain_rows':>10s} {'rain_laps':>10s} {'MAE(s)':>8s}")
        for tyre in sorted(te_ml.loc[wet, "Tyre"].dropna().unique()):
            tm = wet & (te_ml["Tyre"].to_numpy() == tyre)
            m = sec_mae(yte[tm], pred[tm])
            laps = int(te_ml.loc[tm, "Lap"].nunique())
            print(f"  {tyre:14s} {int(tm.sum()):>10d} {laps:>10d} {m:>8.3f}")
            rows.append({"country": str(country.get(sk, sk)), "session_key": sk,
                         "total_laps": total_laps, "rain_laps": rain_laps,
                         "tyre": str(tyre), "rain_rows": int(tm.sum()),
                         "rain_laps_tyre": laps, "mae": m})
        rows.append({"country": str(country.get(sk, sk)), "session_key": sk,
                     "total_laps": total_laps, "rain_laps": rain_laps,
                     "tyre": "ALL(rain)", "rain_rows": int(wet.sum()),
                     "rain_laps_tyre": rain_laps, "mae": mae_wet})

    pd.DataFrame(rows).to_json(ROOT / "rain_err.json", orient="records", indent=2)
    print(f"\nsaved -> rain_err.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
