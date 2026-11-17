"""Why are Baku / Spa / Monaco hard? Welch t-tests on candidate variables.

Compares green-only sectors of the HARD tracks (high green-MAE) vs EASY tracks
(low green-MAE), per feature, and ranks by |Cohen's d| (effect size) since the
sample is large (p ~ 0 for everything). SC/VSC + transition laps are removed.
SCRATCH — read-only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}

# session keys (from data_2025.parquet; identified in the ranking step)
HARD = {9904: "Baku", 9939: "Spa", 9979: "Monaco"}
EASY = {9912: "Imola", 10033: "Las Vegas", 10022: "Jeddah", 9955: "Spielberg"}

VARS = [
    "Interval_Ahead", "Traffic_Within_1s", "Traffic_Within_2s", "Traffic_Within_5s",
    "Total_Sector_Length", "Total_Straight_Length",
    "Prev_Avg_Speed_Corners", "Prev_Avg_Speed_Straight", "Prev_Max_Speed", "Prev_Min_Speed",
    "Prev_Brake_Pct", "Prev_Full_Throttle_Pct", "Prev_Avg_Gear",
    "Humidity", "Track_Temperature", "Air_Temperature", "Pressure",
    "Duration",
]


def drop_sc_vsc(df):
    d = df.copy()
    d["_is_sc"] = d["Track_Status"].isin(SC_SET)
    laps = (d[["Session_Key", "Driver", "Lap", "_is_sc"]]
            .groupby(["Session_Key", "Driver", "Lap"], as_index=False).max())
    laps = laps.sort_values(["Session_Key", "Driver", "Lap"])
    laps["_prev_sc"] = laps.groupby(["Session_Key", "Driver"])["_is_sc"].shift(1).fillna(False)
    d = d.merge(laps[["Session_Key", "Driver", "Lap", "_prev_sc"]],
                on=["Session_Key", "Driver", "Lap"], how="left")
    drop = d["_is_sc"] | d["_prev_sc"].fillna(False)
    return d[~drop].drop(columns=["_is_sc", "_prev_sc"])


def main():
    te = drop_sc_vsc(pd.read_parquet(ROOT / "data_2025.parquet"))
    hard = te[te["Session_Key"].isin(HARD.keys())]
    easy = te[te["Session_Key"].isin(EASY.keys())]
    print(f"HARD tracks={sorted(HARD.values())}  n={len(hard):,}")
    print(f"EASY tracks={sorted(EASY.values())}  n={len(easy):,}\n")

    rows = []
    for v in VARS:
        a = pd.to_numeric(hard[v], errors="coerce").to_numpy(float)
        b = pd.to_numeric(easy[v], errors="coerce").to_numpy(float)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if len(a) < 10 or len(b) < 10:
            continue
        t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
        ma, mb = a.mean(), b.mean()
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d = (ma - mb) / pooled if pooled > 0 else 0.0
        rows.append({"var": v, "mean_hard": ma, "mean_easy": mb,
                     "diff": ma - mb, "cohens_d": d, "t": t, "p": p})

    t = pd.DataFrame(rows).sort_values("cohens_d", key=np.abs, ascending=False)
    print(f"{'variable':24s} {'hard':>10s} {'easy':>10s} {'diff':>9s} {'cohen_d':>8s} {'t':>8s} {'p':>8s}")
    print("-" * 88)
    for _, r in t.iterrows():
        star = " *" if r["p"] < 0.05 else ""
        print(f"{r['var']:24s} {r['mean_hard']:10.3f} {r['mean_easy']:10.3f} "
              f"{r['diff']:9.3f} {r['cohens_d']:8.2f} {r['t']:8.1f} {r['p']:8.1e}{star}")

    # circuit lengths + duration spread, descriptively
    print("\nCircuit length (km) & within-group spread:")
    for label, keys in [("HARD", HARD), ("EASY", EASY)]:
        sub = te[te["Session_Key"].isin(keys)]
        for sk in sorted(keys):
            g = sub[sub["Session_Key"] == sk]
            print(f"  {label} {keys[sk]:10s} len={g['Circuit_Total_Length'].iloc[0]/1000:.2f}km  "
                  f"dur_mean={g['Duration'].mean():.3f}s  dur_std={g['Duration'].std():.3f}s")

    print("\n* p<0.05 (two-sided Welch). Large n => p tiny; interpret |cohen_d| (>=0.5 large).")
    print("DONE")


if __name__ == "__main__":
    main()
