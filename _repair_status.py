"""Repair Track_Status / Track_Status_Inferred in already-built parquet (red-flag parser fix).

Bug (fixed in f1pit/features.py): SC/VSC deployed -> RED FLAG -> standing restart has no
"SAFETY CAR IN THIS LAP" message, so the old parser kept the status at SC until the end of the race
(2023: Brazil 9205, Mexico 9181, 7787, 9149 -> ~6,000 rows / 8.5% of 2023; 2024: 9636 -> 298 rows).

This re-derives the status for every row from race_control (1 OpenF1 request per session, cached),
using only events before each sector's Time_Start (point-in-time, same as the build). It does NOT
recompute build-time features that used the status (Clean_Pace's "clean lap" pool) — a full rebuild
with _rebuild_outcomes.py fixes those too.

Run from the project root (overwrites the files in place):
    python _repair_status.py                 # data_<y>.parquet and data_<y>_out.parquet, 2023-2025
"""

import sys
from pathlib import Path

import pandas as pd

from f1pit.features import repair_track_status

ROOT = Path(__file__).resolve().parent
YEARS = (2023, 2024, 2025)


def main():
    for year in YEARS:
        for name in (f"data_{year}.parquet", f"data_{year}_out.parquet"):
            path = ROOT / name
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            fixed, n = repair_track_status(df)
            print(f"{name}: {n:,} rows changed ({n / max(len(df), 1):.2%})", flush=True)
            if n:
                fixed.to_parquet(path)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
