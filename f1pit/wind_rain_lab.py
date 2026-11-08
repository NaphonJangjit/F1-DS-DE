"""Wind + rainfall feature lab (does NOT modify core logic).

โหลด data_2023.parquet แล้วทดลองเพิ่ม feature:
  1. Wind_Speed / Wind_Direction (weather endpoint มีอยู่แล้ว แต่ core ยังไม่ extract)
  2. Wind alignment = ทิศทาง track (ต่อ sector) · ทิศทางลม  -> เฉลี่ยใน sector
       * ทิศทางลมเป็น compass bearing (องศาเหนือ) แต่พิกัด XY ของ OpenF1 location
         เป็น Cartesian unit ที่ไม่ระบุ scale/origin (เอกสารบอกชัดว่า arbitrary)
         และ racing-line กับ centerline marker ของ tracks.json ใช้ parametrization
         คนละแบบ ทำให้ georeference จาก location ไม่ตรง (residual ~344 m)
       * จึงใช้ lat/lon ของ corner apex + start/finish จาก tracks.json (เป็น compass
         อยู่แล้ว) สร้าง polyline แล้วหาทิศทางเฉลี่ยต่อ sector — static, point-in-time ถูกต้อง
       * dot product (unit vectors) ∈ [-1, 1]; แบ่ง |dot| เป็น bin 0-0.15, 0.15-0.3, ...
  3. Rainfall_Delta = rainfall รอบนี้ - rainfall รอบก่อน (track ยังเปียกหลังฝนตก)

แล้ว rerun feature importance โดย "ถอด" pace anchor (Best_Duration_So_Far, Clean_Pace,
Prev_Duration, Prev_Sector_Duration) เพื่อให้ feature อื่น (traffic/weather/wind/rain) โผล่

Run:  python -m f1pit.wind_rain_lab
"""

import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

from f1pit import config
from f1pit.api import fetch, to_utc
from f1pit.tracks import get_track_by_name, gp_layout, range_layer, sector_bounds
from f1pit.train import (
    TARGET, TEST_SIZE, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TYRE_AGE_INTERACTION, TYRE_AGE_SQUARED,
    REQUIRED_COLS, WEATHER_MAP, OUTLIER_K, _to_float,
    add_tyre_age_interaction, add_prev_trend_features, corner_feature_columns, remove_duration_outliers,
)
from f1pit.feature_lab import _mae_s, _rmse_s, build_X

DATA_PATH = Path(__file__).resolve().parent.parent / "data_2023.parquet"
CTX_PATH = Path(__file__).resolve().parent.parent / "ctx_2023.pkl"
YEAR = 2023

# bin ของ |car_dir · wind_dir| ∈ [0, 1] (0 = crosswind, 1 = head/tail wind)
ALIGN_BINS = [(0.0, 0.15), (0.15, 0.3), (0.3, 0.45), (0.45, 0.6),
              (0.6, 0.75), (0.75, 0.9), (0.9, 1.0)]

WIND_ALIGN_BIN_COLS = [f"Wind_Align_{int(lo * 100):02d}_{int(hi * 100):02d}" for lo, hi in ALIGN_BINS]
WIND_COLS = ["Wind_Speed", "Wind_Dir_Sin", "Wind_Dir_Cos", "Wind_Along", "Wind_Align", *WIND_ALIGN_BIN_COLS]
RAIN_COLS = ["Rainfall_PrevLap", "Rainfall_Delta"]
NEW_COLS = [*WIND_COLS, *RAIN_COLS]

# v2: แยกเฉพาะส่วนตรง (straight) + interaction กับความเร็วลม / ความยาวตรง
WIND_STRAIGHT_COLS = ["Wind_Along_Straight", "Wind_Align_Straight"]
WIND_INTERACT_COLS = ["Wind_Along_Speed", "Wind_Along_Straight_Speed",
                      "Wind_Along_Straight_Speed_x_StraightLen"]

# pace anchor ที่จะ "ถอด" เพื่อให้ feature เล็ก ๆ โผล่
ANCHORS = ["Best_Duration_So_Far", "Clean_Pace", "Prev_Duration", "Prev_Sector_Duration"]

_M_PER_DEG_LAT = 110540.0
_M_PER_DEG_LON = 111320.0   # * cos(lat) ที่ runtime


# ---------------------------------------------------------------------
# Wind (speed / direction) as-of Time_Start
# ---------------------------------------------------------------------
def _wind_asof(weather, times):
    """weather (ดิบจาก fetch) -> (wind_speed, wind_direction) จับคู่ backward ตาม `times`"""
    w = weather[["date", "wind_speed", "wind_direction"]].copy()
    w["date"] = to_utc(w["date"])
    w["wind_speed"] = pd.to_numeric(w["wind_speed"], errors="coerce")
    w["wind_direction"] = pd.to_numeric(w["wind_direction"], errors="coerce")
    w = w.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")

    t = pd.DataFrame({"Time_Start": pd.DatetimeIndex(times)})
    t["_ord"] = np.arange(len(t))
    t = t.sort_values("Time_Start")
    m = pd.merge_asof(t, w, left_on="Time_Start", right_on="date", direction="backward")
    m = m.sort_values("_ord")
    return m["wind_speed"].to_numpy(dtype=float), m["wind_direction"].to_numpy(dtype=float)


# ---------------------------------------------------------------------
# Track direction per sector (จาก tracks.json geometry — compass-correct)
# ---------------------------------------------------------------------
def _project(lons, lats):
    lat0 = float(lats.mean()); lon0 = float(lons.mean())
    east = (lons - lon0) * math.cos(math.radians(lat0)) * _M_PER_DEG_LON
    north = (lats - lat0) * _M_PER_DEG_LAT
    return east, north


def _track_points(track):
    """start/finish + corner apexes -> [(marker, [lon, lat])]"""
    gp = gp_layout(track)
    if gp is None:
        return []
    pts = []
    for layer in gp.get("point_layers", []):
        lid = layer.get("id")
        if lid == "layout_points":
            for it in layer.get("items", []):
                if it.get("id") == "start_finish" and "marker" in it and "location" in it:
                    pts.append((float(it["marker"]), it["location"]))
        elif lid == "corners":
            for it in layer.get("items", []):
                if "marker" in it and "location" in it:
                    pts.append((float(it["marker"]), it["location"]))
    return pts


def _sector_segments(track):
    """-> {sector: (list[unit (east,north)], list[length_m])} ของ polyline segment ใน sector"""
    pts = sorted(_track_points(track), key=lambda p: p[0])
    if len(pts) < 3:
        return None
    markers = np.array([p[0] for p in pts], float)
    lons = np.array([p[1][0] for p in pts], float)
    lats = np.array([p[1][1] for p in pts], float)
    east, north = _project(lons, lats)
    # ปิดรอบ: จุดแรกซ้ำที่ marker+1
    markers = np.append(markers, markers[0] + 1.0)
    east = np.append(east, east[0])
    north = np.append(north, north[0])

    bounds = sector_bounds(track)["sectors"]
    out = {}
    for si, (start, end) in enumerate(bounds, start=1):
        uu, ll = [], []
        for i in range(len(markers) - 1):
            m0, m1 = markers[i], markers[i + 1]
            lo, hi = max(m0, start), min(m1, end)
            if hi <= lo or m1 <= m0:
                continue
            de, dn = east[i + 1] - east[i], north[i + 1] - north[i]
            L = math.hypot(de, dn)
            if L < 1e-9:
                continue
            uu.append(np.array([de / L, dn / L]))
            ll.append(L * ((hi - lo) / (m1 - m0)))
        out[si] = (uu, ll)
    return out


def _corner_ranges(track):
    """[(start, end)] ของ corner_ranges (marker space)"""
    gp = gp_layout(track)
    layer = range_layer(gp, "corner_ranges")
    if layer is None:
        return []
    return sorted((float(it["start"]), float(it["end"]))
                  for it in layer.get("items", []) if "start" in it and "end" in it)


def _len_minus_corners(lo, hi, corners):
    """ความยาว (marker) ของ [lo, hi] ที่ไม่อยู่ใน corner ใด ๆ"""
    covered = 0.0
    for c0, c1 in corners:
        a, b = max(c0, lo), min(c1, hi)
        if b > a:
            covered += b - a
    return (hi - lo) - covered


def _straight_segments(track):
    """เหมือน _sector_segments แต่เก็บเฉพาะส่วน "ตรง" (ไม่นับโค้งตาม corner_ranges)"""
    pts = sorted(_track_points(track), key=lambda p: p[0])
    if len(pts) < 3:
        return None
    markers = np.array([p[0] for p in pts], float)
    lons = np.array([p[1][0] for p in pts], float)
    lats = np.array([p[1][1] for p in pts], float)
    east, north = _project(lons, lats)
    markers = np.append(markers, markers[0] + 1.0)
    east = np.append(east, east[0])
    north = np.append(north, north[0])

    corners = _corner_ranges(track)
    bounds = sector_bounds(track)["sectors"]
    out = {}
    for si, (start, end) in enumerate(bounds, start=1):
        uu, ll = [], []
        for i in range(len(markers) - 1):
            m0, m1 = markers[i], markers[i + 1]
            if m1 <= m0:
                continue
            lo, hi = max(m0, start), min(m1, end)
            if hi <= lo:
                continue
            de, dn = east[i + 1] - east[i], north[i + 1] - north[i]
            L = math.hypot(de, dn)
            if L < 1e-9:
                continue
            straight_len = _len_minus_corners(lo, hi, corners)
            if straight_len <= 0:
                continue
            uu.append(np.array([de / L, dn / L]))
            ll.append(L * straight_len / (m1 - m0))
        out[si] = (uu, ll)
    return out


def _sector_align(U, L, wind_dir):
    """U:(k,2) unit tangents, L:(k,) lengths, wind_dir:(n,) deg -> dict of (n,) alignment"""
    n = len(wind_dir)
    nan = np.full(n, np.nan)
    out = {c: nan.copy() for c in ["Wind_Along", "Wind_Align", *WIND_ALIGN_BIN_COLS]}
    if U is None or len(U) == 0:
        return out
    U = np.asarray(U, float); L = np.asarray(L, float)
    Lsum = L.sum()
    if Lsum <= 0:
        return out
    phi = np.radians(wind_dir)
    W = np.vstack([-np.sin(phi), -np.cos(phi)])   # wind "toward" unit (east, north)
    dots = U @ W                                   # (k, n)
    adots = np.abs(dots)
    out["Wind_Along"] = (L @ dots) / Lsum
    out["Wind_Align"] = (L @ adots) / Lsum
    for k, (lo, hi) in enumerate(ALIGN_BINS):
        mask = (adots >= lo) & (adots < hi) if k < len(ALIGN_BINS) - 1 else (adots >= lo)
        out[WIND_ALIGN_BIN_COLS[k]] = (L @ mask) / Lsum
    return out


# ---------------------------------------------------------------------
# Rainfall lap-level features
# ---------------------------------------------------------------------
def _rain_features(df):
    """-> DataFrame [Rainfall_PrevLap, Rainfall_Delta] align กับ df"""
    d = df[["Session_Key", "Driver", "Lap", "Sector", "Rainfall"]].copy()
    # rainfall ระดับ lap = ค่าของ sector แรก (row เรียงตาม Sector แล้ว)
    lap_rain = d.groupby(["Session_Key", "Driver", "Lap"], sort=False)["Rainfall"].first()
    prev = lap_rain.groupby(["Session_Key", "Driver"], sort=False).shift(1)
    delta = lap_rain - prev
    merged = d[["Session_Key", "Driver", "Lap"]].merge(
        pd.DataFrame({"Rainfall_PrevLap": prev, "Rainfall_Delta": delta}).reset_index(),
        on=["Session_Key", "Driver", "Lap"], how="left")
    return pd.DataFrame({"Rainfall_PrevLap": merged["Rainfall_PrevLap"].to_numpy(),
                         "Rainfall_Delta": merged["Rainfall_Delta"].to_numpy()}, index=d.index)


# ---------------------------------------------------------------------
# Build augmented raw df with wind + rain columns
# ---------------------------------------------------------------------
def _add_wind_rain(raw, corners):
    raw = raw.copy()
    wind_speed = np.full(len(raw), np.nan)
    wind_dir = np.full(len(raw), np.nan)
    align_cols = ["Wind_Along", "Wind_Align", *WIND_ALIGN_BIN_COLS,
                  "Wind_Along_Straight", "Wind_Align_Straight"]
    wind_align = {c: np.full(len(raw), np.nan) for c in align_cols}

    meeting_segs = {}       # full-sector
    meeting_straight = {}   # straight-only
    n_geo_ok = 0

    for s_key, g in raw.groupby("Session_Key", sort=False):
        idx = g.index.to_numpy()
        w = fetch("weather", session_key=int(s_key))
        if w.empty:
            continue
        ws, wd = _wind_asof(w, g["Time_Start"])
        wind_speed[idx] = ws
        wind_dir[idx] = wd

        meeting_key = int(w["meeting_key"].iloc[0])
        if meeting_key not in meeting_segs:
            slug = corners.loc[corners["Meeting_Key"] == meeting_key, "Track_Slug"]
            slug = slug.iloc[0] if len(slug) else None
            track = get_track_by_name(slug) if slug else None
            if track is None:
                meeting_segs[meeting_key] = None
                meeting_straight[meeting_key] = None
            else:
                meeting_segs[meeting_key] = _sector_segments(track)
                meeting_straight[meeting_key] = _straight_segments(track)
        segs = meeting_segs[meeting_key]
        strs = meeting_straight[meeting_key]
        if segs is None:
            continue
        n_geo_ok += 1

        sectors = g["Sector"].to_numpy(dtype=int)
        for sector in (1, 2, 3):
            m = sectors == sector
            if not m.any():
                continue
            u, l = segs.get(sector, (None, None))
            al = _sector_align(u, l, wind_dir[idx[m]])
            for c in ["Wind_Along", "Wind_Align", *WIND_ALIGN_BIN_COLS]:
                wind_align[c][idx[m]] = al[c]

            us, ls = (strs or {}).get(sector, (None, None))
            als = _sector_align(us, ls, wind_dir[idx[m]])
            wind_align["Wind_Along_Straight"][idx[m]] = als["Wind_Along"]
            wind_align["Wind_Align_Straight"][idx[m]] = als["Wind_Align"]

    raw["Wind_Speed"] = wind_speed
    raw["Wind_Direction_raw"] = wind_dir
    raw["Wind_Dir_Sin"] = np.sin(np.radians(wind_dir))
    raw["Wind_Dir_Cos"] = np.cos(np.radians(wind_dir))
    for c, v in wind_align.items():
        raw[c] = v

    rain = _rain_features(raw)
    raw["Rainfall_PrevLap"] = rain["Rainfall_PrevLap"].to_numpy()
    raw["Rainfall_Delta"] = rain["Rainfall_Delta"].to_numpy()

    return raw, n_geo_ok


# ---------------------------------------------------------------------
# prepare_data เดียวกับ train.py แต่เก็บ keys + extra columns ไว้
# ---------------------------------------------------------------------
def _prepare_lab(raw, extra_cols):
    df = raw.copy()
    df, _ = add_prev_trend_features(df)
    if TARGET not in df.columns:
        rawc = pd.to_numeric(df["Duration"], errors="coerce")
        df[TARGET] = np.log(rawc.where(rawc > 0))
    df, removed = remove_duration_outliers(df, k=float(OUTLIER_K))
    df = df.rename(columns={low: cap for low, cap in WEATHER_MAP.items()
                            if cap not in df.columns and low in df.columns})
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    corner_cols = corner_feature_columns(df)
    all_features = list(dict.fromkeys([*CATEGORICAL_FEATURES, *NUMERIC_FEATURES, *corner_cols]))
    present = [c for c in all_features if c in df.columns]

    keep = list(dict.fromkeys(["Session_Key", "Driver", "Lap", "Sector", "Time_Start",
                               TARGET, "Circuit_Total_Length", *present, *extra_cols]))
    keep = [c for c in keep if c in df.columns]
    df_ml = df[keep].copy()

    for col in CATEGORICAL_FEATURES:
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].astype("string").str.upper().str.strip()
    df_ml["Tyre"] = df_ml["Tyre"].replace("", pd.NA)
    df_ml["Track_Status"] = df_ml["Track_Status"].fillna("UNKNOWN").replace("", "UNKNOWN")

    for col in ["Lap", "Sector", TARGET,
                *[c for c in present if c not in CATEGORICAL_FEATURES], *extra_cols]:
        df_ml[col] = _to_float(df_ml[col])

    df_ml = df_ml.dropna(subset=["Tyre", "Tyre_Age", "Prev_Avg_Speed_Corners",
                                 "Prev_Duration", TARGET, "Lap", "Sector"])
    df_ml = df_ml[np.isfinite(df_ml[TARGET].to_numpy(dtype=float))
                  & (df_ml["Tyre_Age"] > 0)
                  & (df_ml["Prev_Avg_Speed_Corners"] > 0)
                  & (df_ml["Prev_Duration"] > 0)
                  & df_ml["Sector"].isin([1, 2, 3])].reset_index(drop=True)
    if len(df_ml) < 10:
        raise ValueError("ข้อมูลเหลือน้อยเกินไป")

    all_nan = [c for c in present if df_ml[c].isna().all()]
    features = [c for c in present if c not in all_nan]
    df_ml = add_tyre_age_interaction(df_ml)
    features = ([c for c in features if c not in ("Tyre", "Tyre_Age")]
                + TYRE_AGE_INTERACTION + TYRE_AGE_SQUARED)

    extra_ok = [c for c in extra_cols if c in df_ml.columns and not df_ml[c].isna().all()]
    features = features + extra_ok
    return df_ml, features, removed, extra_ok


# ---------------------------------------------------------------------
# importance / eval helpers
# ---------------------------------------------------------------------
def _tree(y, X, tr_idx, n_est=350):
    return ExtraTreesRegressor(n_estimators=n_est, max_features=0.9, min_samples_leaf=2,
                               n_jobs=-1, random_state=config.RANDOM_STATE).fit(X[tr_idx], y[tr_idx])


def _report_mae(y, X, tr_idx, te_idx, label):
    est = _tree(y, X, tr_idx)
    p = est.predict(X[te_idx])
    print(f"  {label:52s} MAE_s={_mae_s(y[te_idx], p):.4f}  RMSE_s={_rmse_s(y[te_idx], p):.4f}")


def _report_importance(y, X, cols, tr_idx, title, top=24):
    est = _tree(y, X, tr_idx)
    imp = pd.Series(est.feature_importances_, index=cols).sort_values(ascending=False)
    print(f"\n{title}")
    for name, v in imp.head(top).items():
        print(f"  {name:42s} {v:.4f}")
    return imp


# ---------------------------------------------------------------------
def main():
    print("Loading data + TrackContext ...")
    df = pd.read_parquet(DATA_PATH)
    ctx = pickle.load(open(CTX_PATH, "rb"))
    corners = ctx.corners

    print("Building wind + rainfall features (track direction from tracks.json geometry) ...")
    df_aug, n_geo_ok = _add_wind_rain(df, corners)
    print(f"  sessions with track geometry: {n_geo_ok} / {df['Session_Key'].nunique()}")

    # v2 interactions: wind × speed / straight length
    df_aug["Wind_Along_Speed"] = df_aug["Wind_Speed"] * df_aug["Wind_Along"]
    df_aug["Wind_Along_Straight_Speed"] = df_aug["Wind_Speed"] * df_aug["Wind_Along_Straight"]
    df_aug["Wind_Along_Straight_Speed_x_StraightLen"] = (
        df_aug["Wind_Along_Straight_Speed"] * df_aug["Total_Straight_Length"])

    V1 = [*WIND_COLS, *RAIN_COLS]
    V2 = [*WIND_STRAIGHT_COLS, *WIND_INTERACT_COLS]
    all_new = [*V1, *V2]

    print("\nNew feature coverage (non-null) on raw data:")
    for c in all_new:
        nn = df_aug[c].notna().sum()
        print(f"  {c:38s} {nn:>7,} / {len(df_aug):,}")
    print(f"  Wind_Speed range: {df_aug['Wind_Speed'].min():.2f} .. {df_aug['Wind_Speed'].max():.2f} m/s")
    print(f"  Wind_Along (full, +tail/-head):   mean={df_aug['Wind_Along'].mean():+.3f}  sd={df_aug['Wind_Along'].std():.3f}")
    print(f"  Wind_Along_Straight (+tail/-head): mean={df_aug['Wind_Along_Straight'].mean():+.3f}  sd={df_aug['Wind_Along_Straight'].std():.3f}")

    print("\nPreparing ML frame (mirror prepare_data + keep keys/extras) ...")
    df_ml, features, removed, extra_ok = _prepare_lab(df_aug, all_new)
    print(f"  rows={len(df_ml):,}  outlier-removed={removed:,}  base+new features={len(features)}")
    print(f"  new features kept: {len(extra_ok)}")

    y = df_ml[TARGET].to_numpy(dtype=float)
    tr_idx, te_idx = train_test_split(np.arange(len(df_ml)), test_size=TEST_SIZE,
                                      random_state=config.RANDOM_STATE)

    def _run(feat_list, label):
        X, _ = build_X(df_ml, feat_list, log_duration=True, extra_num=["Lap"], extra_cat=["Sector"])
        _report_mae(y, X, tr_idx, te_idx, label)

    base_feats = [c for c in features if c not in all_new]

    print("\n" + "=" * 80)
    print("MAE (ExtraTrees) — anchors KEPT (does wind help at all?)")
    _run(features, "base + all wind/rain (anchors kept)")
    _run(base_feats, "base only (anchors kept)  [reference]")

    print("\n" + "=" * 80)
    print("MAE (ExtraTrees) — anchors REMOVED (wind visible)")
    no_anchor_base = [c for c in base_feats if c not in ANCHORS]
    _run(no_anchor_base, "base only  [reference]")
    _run([c for c in features if c not in [*V2, *ANCHORS]], "base + v1 wind/rain (full-sector)")
    _run([c for c in features if c not in [*V1, *ANCHORS]], "base + v2 straight/interact")
    _run([c for c in features if c not in ANCHORS], "base + all wind/rain")

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE (ExtraTrees) — anchors removed, all wind/rain")
    no_anchor = [c for c in features if c not in ANCHORS]
    X_na, na_cols = build_X(df_ml, no_anchor, log_duration=True, extra_num=["Lap"], extra_cat=["Sector"])
    imp = _report_importance(y, X_na, na_cols, tr_idx, "== anchors REMOVED (top 30) ==", top=30)

    wind_feats = [c for c in na_cols if c.startswith("Wind_") or c.startswith("Rainfall_")]
    w_imp = imp[imp.index.isin(wind_feats)].sum()
    print(f"\n  -> total importance of ALL wind/rain features (no anchors): {w_imp:.4f}")
    for name, v in imp[imp.index.isin(wind_feats)].head(10).items():
        print(f"     {name:38s} {v:.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
