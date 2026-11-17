"""Rank 2025 races by frozen-RF MAE_s and cross-tab against weather / track status.

Reads per-race MAE from rf_drift_results.json (produced by _rf_drift.py) and joins
with data_2025.parquet to explain *why* some races are hard (wet, SC/VSC, cold...).
SCRATCH — read-only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
df = pd.read_parquet(ROOT / "data_2025.parquet")
res = json.loads((ROOT / "rf_drift_results.json").read_text())
mae_by_sk = {int(r["session_key"]): r["static_mae"] for r in res["races"]}

rows = []
for sk, g in df.groupby("Session_Key", sort=False):
    rows.append({
        "session_key": int(sk),
        "Country": g["Country"].iloc[0],
        "rows": len(g),
        "MAE_s": mae_by_sk[int(sk)],
        "wet_pct": float(g["Rainfall"].mean() * 100),
        "sc_pct": float(g["Track_Status"].isin(["SC", "SC_ENDING"]).mean() * 100),
        "vsc_pct": float(g["Track_Status"].isin(["VSC", "VSC_ENDING"]).mean() * 100),
        "track_temp": float(g["Track_Temperature"].mean()),
        "humidity": float(g["Humidity"].mean()),
        "air_temp": float(g["Air_Temperature"].mean()),
    })

t = pd.DataFrame(rows).sort_values("MAE_s", ascending=False).reset_index(drop=True)
t.insert(0, "Rank", t.index + 1)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# correlations of race-level MAE with conditions
num = t[["MAE_s", "wet_pct", "sc_pct", "vsc_pct", "track_temp", "humidity", "air_temp"]].corr()["MAE_s"].drop("MAE_s")
print("\nrace-level |corr| with MAE_s:")
print(num.to_string(float_format=lambda x: f"{x:+.3f}"))

# wet vs dry aggregate error (sector-weighted)
wet = df[df["Rainfall"] == 1]
dry = df[df["Rainfall"] == 0]
print(f"\nwet sectors={len(wet):,}  dry sectors={len(dry):,}")
for name, sub in [("WET", wet), ("DRY", dry)]:
    sk_mae = [mae_by_sk[int(k)] for k in sub["Session_Key"]]
    print(f"  {name}: races involved={sub['Session_Key'].nunique()}  "
          f"mean of their race-MAE={np.mean(sk_mae):.3f}")
print("DONE")
