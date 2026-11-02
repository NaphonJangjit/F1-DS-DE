"""Build data_2024.parquet + data_2025.parquet (สำหรับ temporal split train 23+24 -> test 25)

Run จาก project root:  python -m f1pit.build_years
(ช้า: fetch OpenF1 ทุก session/driver — รันเป็น background)
"""

import f1pit as f

f.config.TRACKS_JSON = "tracks.json"   # relative to CWD (project root)

for year in (2024, 2025):
    print(f"\n{'=' * 70}\n== {year}: build_track_context ==\n{'=' * 70}")
    ctx = f.build_track_context(year, verbose=True, plot=False)
    ctx.save(f"ctx_{year}.pkl")

    print(f"\n{'=' * 70}\n== {year}: build_dataset ==\n{'=' * 70}")
    data = f.build_dataset(year, ctx=ctx, save_path=f"data_{year}.parquet")
    print(f"== {year}: {len(data)} rows, {len(data.columns)} columns ==")

print("\nALL DONE")
