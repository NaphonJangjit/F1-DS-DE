"""Add quantitative rain (precipitation mm/h) from Open-Meteo archive API.

OpenF1 weather has only a binary `rainfall` flag. This pulls hourly precipitation
(ERA5 reanalysis, mm) from Open-Meteo for each circuit's lat/lon and merges it
as-of each sector's Time_Start (backward -> point-in-time correct).

Then retrains the frozen RF (2023+24 -> 2025) with vs without the new `Rain_mmh`
feature and reports the difference (overall + wet-regime MAE).

SCRATCH. Requires network to archive-api.open-meteo.com.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from f1pit.api import fetch
from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}
OM_URL = "https://archive-api.open-meteo.com/v1/archive"

# OpenF1 circuit_short_name -> tracks.json slug (only the mismatches)
COORD_ALIAS = {
    "Austin": "circuit-of-the-americas",
    "Catalunya": "barcelona",
    "Lusail": "losail",
    "Monte Carlo": "monaco",
    "Sakhir": "bahrain",
    "Spielberg": "red-bull-ring",
    "Yas Marina Circuit": "yas-marina",
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def session_locations():
    """session_key -> (lat, lon)."""
    tracks = json.load(open(ROOT / "tracks.json"))
    by_slug = {t["slug"]: (t["location.lat"], t["location.lon"]) for t in tracks}

    loc = {}
    for year in [2023, 2024, 2025]:
        s = pd.read_parquet(ROOT / f"data_{year}.parquet")[["Session_Key"]].drop_duplicates()
        sessions = fetch("sessions", year=year, session_name="Race")
        name_of = dict(zip(sessions["session_key"], sessions["circuit_short_name"]))
        for sk in s["Session_Key"]:
            cs = name_of.get(int(sk))
            if cs is None:
                continue
            slug = COORD_ALIAS.get(cs, cs.lower().replace(" ", "-"))
            coord = by_slug.get(slug)
            if coord is None:
                log(f"  !! no coord for {cs} (slug={slug})")
                continue
            loc[int(sk)] = coord
    return loc


def fetch_precip(lat, lon, start, end):
    empty = pd.DataFrame({"time": pd.Series(dtype="datetime64[ns]"),
                          "precip": pd.Series(dtype=float)})
    for attempt in range(3):
        try:
            r = requests.get(OM_URL, params={"latitude": lat, "longitude": lon,
                                             "start_date": start, "end_date": end,
                                             "hourly": "precipitation", "timezone": "UTC"},
                             timeout=120)
            if r.status_code != 200:
                log(f"  status {r.status_code} for ({lat},{lon}) attempt {attempt}")
                time.sleep(2 * (attempt + 1))
                continue
            d = r.json()
            if "hourly" not in d:
                return empty
            t = pd.to_datetime(d["hourly"]["time"], utc=True).tz_localize(None)
            p = pd.to_numeric(d["hourly"]["precipitation"], errors="coerce").astype(float)
            return pd.DataFrame({"time": t, "precip": p})
        except Exception as exc:
            log(f"  retry {attempt} for ({lat},{lon}): {exc}")
            time.sleep(2 * (attempt + 1))
    return empty


def add_rain_mmh(raw, loc, precip_cache):
    raw = raw.copy()
    lats = raw["Session_Key"].map(lambda sk: loc.get(int(sk), (np.nan, np.nan))[0])
    lons = raw["Session_Key"].map(lambda sk: loc.get(int(sk), (np.nan, np.nan))[1])
    raw["Rain_mmh"] = np.nan
    ts = pd.to_datetime(raw["Time_Start"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
    for (lat, lon) in sorted(set(zip(lats, lons))):
        if pd.isna(lat):
            continue
        mask = (lats == lat) & (lons == lon)
        key = (float(lat), float(lon))
        w = precip_cache.get(key)
        if w is None or w.empty:
            continue
        w = w.copy()
        w["time"] = pd.to_datetime(w["time"], utc=True).dt.tz_localize(None).astype("datetime64[ns]")
        w = w.sort_values("time").drop_duplicates("time", keep="last")
        t = pd.DataFrame({"Time_Start": ts[mask]})
        t["_ord"] = np.arange(len(t))
        t = t.sort_values("Time_Start")
        m = pd.merge_asof(t, w, left_on="Time_Start", right_on="time", direction="backward")
        m = m.sort_values("_ord")
        raw.loc[mask, "Rain_mmh"] = m["precip"].to_numpy(float)
    return raw


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
    log("mapping sessions -> circuit coordinates ...")
    loc = session_locations()
    log(f"locations={len(set(loc.values()))}")

    log("fetching Open-Meteo hourly precipitation per location ...")
    precip_cache = {}
    for (lat, lon) in sorted(set(loc.values())):
        precip_cache[(float(lat), float(lon))] = fetch_precip(lat, lon, "2023-01-01", "2025-12-31")
        time.sleep(0.3)
    log(f"fetched {len(precip_cache)} locations, non-empty={sum(1 for v in precip_cache.values() if not v.empty)}")

    log("loading + merging Rain_mmh into 2023/24/25 ...")
    tr = add_rain_mmh(pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                                 pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True),
                      loc, precip_cache)
    te = add_rain_mmh(pd.read_parquet(ROOT / "data_2025.parquet"), loc, precip_cache)
    log(f"Rain_mmh coverage: train non-null={tr['Rain_mmh'].notna().mean()*100:.1f}%  "
        f"test={te['Rain_mmh'].notna().mean()*100:.1f}%")
    log(f"Rain_mmh>0 share: train={(tr['Rain_mmh']>0).mean()*100:.2f}%  "
        f"test={(te['Rain_mmh']>0).mean()*100:.2f}%")
    corr = te[["Rain_mmh", "Rainfall"]].corr().iloc[0, 1]
    log(f"corr(Rain_mmh, Rainfall binary) on test = {corr:.3f}")

    log("preparing ML frames (extra_cols=[Rain_mmh]) ...")
    tr_ml, tr_feats, _, _ = _prepare_lab(tr, ["Rain_mmh"])
    te_ml, te_feats, _, _ = _prepare_lab(te, ["Rain_mmh"])
    feats_all = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    feats_base = [c for c in feats_all if c != "Rain_mmh"]
    log(f"features base={len(feats_base)}  +Rain_mmh={len(feats_all)}")

    pre = build_preprocessor(feats_all).fit(tr_ml[feats_all])
    Xtr = pre.transform(tr_ml[feats_all])
    ytr = tr_ml[TARGET].to_numpy(float)
    te_g = green_mask(te_ml)

    results = {}
    for label, cols in [("baseline", feats_base), ("+Rain_mmh", feats_all)]:
        log(f"training RF {label} ...")
        t = time.time()
        pre_ = build_preprocessor(cols).fit(tr_ml[cols])
        rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                   min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                   random_state=42).fit(pre_.transform(tr_ml[cols]), ytr)
        p = rf.predict(pre_.transform(te_ml[cols]))
        yte = te_ml[TARGET].to_numpy(float)
        full = sec_mae(yte, p)
        green = sec_mae(yte[te_g], p[te_g])
        wet = (te_ml["Rainfall"].to_numpy() == 1)
        w = sec_mae(yte[wet], p[wet]) if wet.any() else float("nan")
        results[label] = {"full": full, "green": green, "wet": w}
        log(f"  {label:10s} full={full:.4f} green={green:.4f} wet={w:.4f} ({time.time()-t:.0f}s)")

    print("\n" + "=" * 56)
    print(f"{'model':12s} {'full':>8s} {'green':>8s} {'wet':>8s}")
    print("-" * 56)
    for label in ["baseline", "+Rain_mmh"]:
        r = results[label]
        print(f"{label:12s} {r['full']:8.4f} {r['green']:8.4f} {r['wet']:8.4f}")
    print("=" * 56)
    d = results["baseline"]["full"] - results["+Rain_mmh"]["full"]
    print(f"improvement (full): {d:+.4f}s  (green: "
          f"{results['baseline']['green']-results['+Rain_mmh']['green']:+.4f}s)")

    with open(ROOT / "rain_mmh_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "corr_rainmmh_rainfall": corr}, f, indent=2)
    log(f"saved -> rain_mmh_results.json  total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
