"""Build one season end to end: OpenF1 cache pre-warm -> TrackContext -> lap-sector dataset.

Run:  py -3.14 _build_year.py 2026
Out:  ctx_<year>.pkl, data_<year>.parquet  (races whose circuit is missing from tracks.json are skipped
      with a warning, e.g. 2026 Madring and Kuala Lumpur). Resumable: the OpenF1 disk cache is reused.
"""

import sys
import time

import f1pit as f
import _prewarm_cache as pw

f.config.TRACKS_JSON = "tracks.json"


def main(year):
    t0 = time.time()
    pw.YEARS = (year,)
    print(f"[{year}] pre-warm OpenF1 cache ...", flush=True)
    pw.main()
    print(f"[{year}] track context ... ({time.time() - t0:.0f}s)", flush=True)
    ctx = f.build_track_context(year, verbose=True)
    ctx.save(f"ctx_{year}.pkl")
    print(f"[{year}] dataset ... ({time.time() - t0:.0f}s)", flush=True)
    # require_track_profile: races whose circuit has no tracks.json layout (2026 Madring, Kuala Lumpur)
    # are left out instead of being kept with empty geometry
    df = f.build_dataset(year, ctx=ctx, save_path=f"data_{year}.parquet", verbose=True,
                         require_track_profile=True)
    print(f"[{year}] done: {len(df):,} rows, {df['Session_Key'].nunique() if len(df) else 0} races "
          f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2026)
