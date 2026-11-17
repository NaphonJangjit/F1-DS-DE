"""Rebuild data_<year>.parquet WITH Outcome_* telemetry columns.

เหตุผล: forest-of-forests ต้องการ "current telemetry" เป็น target (train current = f(prev, tyre))
แต่ parquet เดิม export เฉพาะ Prev_* + Duration. ต้อง rebuild ด้วย include_outcome_columns=True
(ใช้ ctx_<year>.pkl เดิม + cache .cache/openf1 -> ไม่ต้อง re-cluster / re-fetch)

Run จาก project root:  python _rebuild_outcomes.py
(ช้า ~ชั่วโมง ต่อปี — รันเป็น background:  python _rebuild_outcomes.py > rebuild_outcomes_run.log 2>&1)
"""

import sys
import time

import f1pit as f

f.config.TRACKS_JSON = "tracks.json"   # relative to CWD (project root)

YEARS = (2023, 2024, 2025)


def main():
    for year in YEARS:
        t0 = time.time()
        print(f"\n{'=' * 70}\n== {year}: load TrackContext ==\n{'=' * 70}", flush=True)
        ctx = f.TrackContext.load(f"ctx_{year}.pkl")
        print(f"  loaded ctx_{year}.pkl (n_clusters={ctx.n_clusters})", flush=True)

        print(f"\n== {year}: build_dataset (include_outcome_columns=True) ==\n", flush=True)
        data = f.build_dataset(year, ctx=ctx, include_outcome_columns=True,
                               save_path=f"data_{year}_out.parquet")
        outcome_cols = [c for c in data.columns if c.startswith("Outcome_")]
        print(f"== {year}: {len(data):,} rows, {len(data.columns)} cols, "
              f"{len(outcome_cols)} outcome cols ({time.time() - t0:.0f}s) ==", flush=True)
        print(f"   outcome cols: {outcome_cols}", flush=True)

    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
