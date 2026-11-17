"""Hypothesis test: George Russell (Hard, no stop) vs Kimi Antonelli (pitted lap 28 -> Medium).

Monza 2026, from lap 28 to the end. Uses only the laps endpoint (driver mapping via drivers).
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind

import f1pit as f

START_LAP = 28


def main():
    sessions = f.fetch("sessions", year=2026, session_name="Race")
    monza = sessions[sessions["circuit_short_name"].str.lower() == "monza"]
    if monza.empty:
        monza = sessions[sessions["country_name"].str.lower() == "italy"]
    if monza.empty:
        print("Monza 2026 race not found")
        return
    sk = int(monza.iloc[0]["session_key"])
    print(f"Monza 2026 race  session_key={sk}  circuit={monza.iloc[0]['circuit_short_name']}")

    laps = f.fetch("laps", session_key=sk)
    drivers = f.fetch("drivers", session_key=sk)

    def num_for(name):
        m = drivers[drivers.get("full_name", pd.Series(dtype=str)).str.contains(name, case=False, na=False)]
        if m.empty:
            m = drivers[drivers.get("name_acronym", pd.Series(dtype=str)).str.contains(name.upper(), case=False, na=False)]
        if m.empty:
            m = drivers[drivers.get("broadcast_name", pd.Series(dtype=str)).str.contains(name, case=False, na=False)]
        return int(m.iloc[0]["driver_number"]) if not m.empty else None

    rn, an = num_for("Russell"), num_for("Antonelli")
    print(f"Russell driver_number={rn}   Antonelli driver_number={an}")

    if "lap_duration" in laps.columns:
        laps["lap_time"] = pd.to_numeric(laps["lap_duration"], errors="coerce")
    else:
        laps["lap_time"] = sum(
            pd.to_numeric(laps.get(c), errors="coerce")
            for c in ["duration_sector_1", "duration_sector_2", "duration_sector_3"])

    r = laps[(laps["driver_number"] == rn) & (laps["lap_number"] >= START_LAP)]["lap_time"].dropna().to_numpy(float)
    a = laps[(laps["driver_number"] == an) & (laps["lap_number"] >= START_LAP)]["lap_time"].dropna().to_numpy(float)

    print(f"\nRussell (Hard, no stop)  lap {START_LAP}+ : n={len(r)}  mean={r.mean():.3f}s  std={r.std():.3f}s  min={r.min():.3f}  max={r.max():.3f}")
    print(f"Antonelli (Medium)        lap {START_LAP}+ : n={len(a)}  mean={a.mean():.3f}s  std={a.std():.3f}s  min={a.min():.3f}  max={a.max():.3f}")

    t, p = ttest_ind(r, a, equal_var=False)
    u, pu = mannwhitneyu(r, a, alternative="two-sided")
    print(f"\nWelch t-test      : t={t:.3f}   p={p:.5f}")
    print(f"Mann-Whitney U    : U={u:.1f}   p={pu:.5f}")
    print(f"mean diff (Russell - Antonelli) = {r.mean() - a.mean():+.3f}s")


if __name__ == "__main__":
    main()
