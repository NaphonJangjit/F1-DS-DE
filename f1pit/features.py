"""Lap/sector dataset builder (point-in-time correct) — เดิม create_a_fucking_table / create_fucking_table

หลักการ: ทุก feature ต้อง "รู้ได้ ณ Time_Start ของ sector นั้น" เท่านั้น
ไม่มีข้อมูลจากภายใน sector ปัจจุบันหรืออนาคตปนเข้ามา

>>> df = f1pit.build_sector_table(1, "Bahrain", 2024, "Race", ctx=ctx)
>>> df = f1pit.build_sector_table(1, session_key=9472, ctx=ctx)   # ระบุ session ตรง ๆ

Target / Input
--------------
Target   : Duration (เวลาของ sector ปัจจุบัน)
Will_Box : INPUT — pit บันทึกที่ lap N -> Will_Box = 1 ที่ Sector 1 ของ lap N+1
Prev_Lap_Box_Duration = pit_duration ที่ Sector 1 ของ lap N+2 ; Prev_Lap_Pitted = 1 ที่ lap N+2

Features
--------
Prev_*               : ค่าจาก lap ล่าสุดก่อนหน้าที่มีค่า ใน sector เดียวกัน
Prev_Sector_Duration : sector ที่เพิ่งจบก่อนหน้า (ตามลำดับเวลา)
Clean_Pace           : median Duration ของ lap สะอาด `clean_window` lap ล่าสุด (fallback: ทุก lap)
Best_Duration_So_Far : Duration ดีที่สุดใน sector เดียวกันของ lap ก่อนหน้า
Prev_I1/I2/St_Speed  : ความเร็วจุดวัดล่าสุดที่รู้ ณ จบ lap ก่อนหน้า
Interval_Ahead       : gap ถึงรถคันหน้า ณ Time_Start (ผู้นำ/ถูกแซง lap/ไม่มีข้อมูล -> interval_fill)
Traffic_Within_Ns    : จำนวนรถคันอื่นที่ gap (จาก gap_to_leader) ต่างกันไม่เกิน N วินาที ณ Time_Start
Weather              : observation ล่าสุด ณ Time_Start (as-of, tolerance)
"""

import bisect

import numpy as np
import pandas as pd

from .api import fetch, to_utc
from .context import add_track_profile

# ---------------------------------------------------------------------
# Column groups
# ---------------------------------------------------------------------
TARGET_COLUMN = "Duration"

META_COLUMNS = ["Year", "Country", "Session", "Session_Key", "Driver",
                "Lap", "Sector", "Time_Start"]

WEATHER_MAP = {
    "track_temperature": "Track_Temperature",
    "rainfall": "Rainfall",
    "pressure": "Pressure",
    "humidity": "Humidity",
    "air_temperature": "Air_Temperature",
}

# สารประกอบยางทั้งหมด (ใช้สร้าง interaction Tyre_Age_<compound>)
TYRE_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

# นับรถคันอื่นที่อยู่ในระยะ gap ไม่เกิน threshold นี้ (วินาที)
TRAFFIC_THRESHOLDS = (1.0, 1.5, 2.0, 5.0)


def _traffic_col(s):
    return f"Traffic_Within_{s:g}s"


TRAFFIC_COLUMNS = [_traffic_col(s) for s in TRAFFIC_THRESHOLDS]

FEATURE_COLUMNS = [
    "Tyre", "Tyre_Age", "Tyre_Age_Total",
    "Track_Status", "Track_Status_Inferred",
    "Prev_Avg_Speed_Corners", "Prev_Avg_Speed_Straight", "Prev_Duration", "Prev_Sector_Duration",
    "Clean_Pace", "Best_Duration_So_Far",
    "Prev_Full_Throttle_Pct", "Prev_Brake_Pct", "Prev_DRS_Pct",
    "Prev_Max_Speed", "Prev_Avg_Gear", "Prev_Min_Speed",
    "Prev_I1_Speed", "Prev_I2_Speed", "Prev_St_Speed",
    "Interval_Ahead",
    *TRAFFIC_COLUMNS,
    "Will_Box", "Prev_Lap_Pitted", "Prev_Lap_Box_Duration", "Pit_Stops_Done",
    *WEATHER_MAP.values(),
]

# ผลลัพธ์ของ sector ปัจจุบัน -> ห้ามใช้เป็น feature (export ไว้วิเคราะห์ได้)
# ครบชุด telemetry ที่มี Prev_* counterpart (ดู TELEMETRY_HISTORY) — ใช้เป็น target
# ของ forest-of-forests (predict current telemetry จาก Prev_* + tyre)
OUTCOME_COLUMNS = ["Outcome_Time_End", "Outcome_Avg_Speed", "Outcome_Avg_Speed_Corners", "Outcome_Avg_Speed_Straight",
                   "Outcome_Full_Throttle_Pct", "Outcome_Brake_Pct", "Outcome_DRS_Pct",
                   "Outcome_Max_Speed", "Outcome_Min_Speed", "Outcome_Avg_Gear"]

INTERVAL_FAR_DEFAULT = 60.0   # วินาที: "ไม่มีรถคันหน้าใกล้ ๆ"

# Prev_* feature <- outcome column ของ lap ก่อนหน้า
TELEMETRY_HISTORY = {
    "Prev_Avg_Speed_Corners": "Outcome_Avg_Speed_Corners",
    "Prev_Avg_Speed_Straight": "Outcome_Avg_Speed_Straight",
    "Prev_Full_Throttle_Pct": "Outcome_Full_Throttle_Pct",
    "Prev_Brake_Pct": "Outcome_Brake_Pct",
    "Prev_DRS_Pct": "Outcome_DRS_Pct",
    "Prev_Max_Speed": "Outcome_Max_Speed",
    "Prev_Min_Speed": "Outcome_Min_Speed",
    "Prev_Avg_Gear": "Outcome_Avg_Gear",
}


# ---------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------
def _safe_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        return ""
    return str(value).strip()


def _normalize_code(value):
    """'**Track,**' -> 'TRACK'"""
    return "".join(ch for ch in _safe_text(value).upper() if ch.isalnum())


def _empty(df):
    return df is None or df.empty


def _numeric_cols(df, cols, default=np.nan):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else default
    return df


def _lower_columns(df, name):
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise KeyError(f"'{name}' has no 'date'. Columns: {df.columns.tolist()}")
    df["date"] = to_utc(df["date"])
    return df


def _last_valid(history, key):
    """ค่าล่าสุดที่ไม่ใช่ NaN จาก history (เก่า -> ใหม่)"""
    for h in reversed(history):
        v = h[key]
        if v is not None and not pd.isna(v):
            return v
    return np.nan


# ---------------------------------------------------------------------
# Race control -> track status
# ---------------------------------------------------------------------
def _classify_message(message, flag, scope):
    # ตรวจ VSC ก่อน SC
    if "VIRTUAL SAFETY CAR DEPLOYED" in message:
        return "VSC_DEPLOYED"
    if "VIRTUAL SAFETY CAR ENDING" in message:
        return "VSC_ENDING"
    if "SAFETY CAR DEPLOYED" in message:
        return "SC_DEPLOYED"
    if "SAFETY CAR IN THIS LAP" in message:
        return "SC_ENDING"
    # RED FLAG หยุดการแข่ง แล้วเริ่มใหม่ (standing start) โดยไม่มีข้อความ "SAFETY CAR IN THIS LAP"
    # ถ้าไม่ reset ตรงนี้ สถานะ SC จะค้างไปจนจบเรซ (เช่น Brazil/Mexico 2023)
    if flag == "RED" and scope == "TRACK":
        return "RED_FLAG"
    explicit_green = message in {"GREEN FLAG", "GREEN FLAG SHOWN", "GREEN FLAG AT START"}
    green_track = flag == "GREEN" and scope == "TRACK" and "PIT EXIT" not in message
    return "GREEN" if (explicit_green or green_track) else None


def extract_control_events(rc, lap_starts):
    """race_control -> list ของ event SC/VSC/GREEN เรียงตามเวลา

    lap_starts: DataFrame[lap_number, date_start] ใช้เติม lap ของ event ที่ไม่มี lap_number
    """
    rc = _lower_columns(rc, "race_control").dropna(subset=["date"]).sort_values("date")
    lap_times = lap_starts["date_start"].reset_index(drop=True)
    lap_nums = lap_starts["lap_number"].reset_index(drop=True)

    events = []
    for _, row in rc.iterrows():
        message = " ".join(_safe_text(row.get("message", "")).upper().split())
        etype = _classify_message(message, _normalize_code(row.get("flag")), _normalize_code(row.get("scope")))
        if etype is None:
            continue

        lap = pd.to_numeric(row.get("lap_number"), errors="coerce")
        if pd.notna(lap):
            lap = int(lap)
        else:
            idx = lap_times.searchsorted(row["date"], side="right") - 1
            lap = int(lap_nums.iloc[idx]) if idx >= 0 else None
        events.append({"date": row["date"], "event_type": etype, "lap_number": lap, "message": message})
    return events


def track_status_at(events, available, sector_start, sector_lap):
    """สถานะแทร็ก ณ sector_start -> (status, inferred)"""
    if not available:
        return "UNKNOWN", False

    status, inferred = "GREEN", False
    sc_ending_lap = vsc_ending_time = None
    for ev in events:
        if ev["date"] > sector_start:
            break
        t, inferred = ev["event_type"], False
        sc_ending_lap = ev["lap_number"] if t == "SC_ENDING" else None
        vsc_ending_time = ev["date"] if t == "VSC_ENDING" else None
        status = {"SC_DEPLOYED": "SC", "SC_ENDING": "SC_ENDING", "VSC_DEPLOYED": "VSC",
                  "VSC_ENDING": "VSC_ENDING", "GREEN": "GREEN", "RED_FLAG": "GREEN"}[t]
        if t == "RED_FLAG":
            inferred = True   # SC/VSC ก่อน red flag สิ้นสุดลงเมื่อเรซเริ่มใหม่

    # SC_ENDING ไม่ควรค้างข้าม lap
    if (status == "SC_ENDING" and sc_ending_lap is not None
            and pd.notna(sector_lap) and sector_lap > sc_ending_lap):
        status, inferred = "GREEN", True
    # VSC_ENDING -> GREEN หลัง 15 วินาที ถ้าไม่มีสัญญาณเขียว
    if (status == "VSC_ENDING" and vsc_ending_time is not None
            and sector_start >= vsc_ending_time + pd.Timedelta(seconds=15)):
        status, inferred = "GREEN", True
    return status, inferred


def repair_track_status(df, fetch_fn=fetch):
    """คำนวณ Track_Status / Track_Status_Inferred ใหม่ให้ตารางที่ build ไว้แล้ว (ไม่ต้อง rebuild ทั้งปี)

    ใช้ parser ปัจจุบัน (เช่น fix RED FLAG) กับ race_control ของแต่ละ session (1 request/session)
    lap_starts ของแต่ละ driver = Time_Start ของ sector 1 ในตารางเอง (= laps.date_start ตอน build)
    ใช้เฉพาะ event ก่อน Time_Start -> point-in-time เหมือนตอน build

    หมายเหตุ: feature ที่ derive จาก Track_Status ตอน build (Clean_Pace ใช้ "lap สะอาด") ไม่ถูกคำนวณใหม่
    คืน (df ใหม่, จำนวนแถวที่สถานะเปลี่ยน)
    """
    out = df.copy()
    status = out["Track_Status"].astype("string").to_numpy(dtype=object).copy()
    inferred = pd.to_numeric(out["Track_Status_Inferred"], errors="coerce").to_numpy(float).copy()
    t_start = to_utc(out["Time_Start"])
    for sk, g in out.groupby("Session_Key", sort=False):
        rc = fetch_fn("race_control", session_key=int(sk))
        if _empty(rc):
            continue
        for _, gd in g.groupby("Driver", sort=False):
            s1 = gd[gd["Sector"] == 1]
            lap_starts = (pd.DataFrame({"lap_number": s1["Lap"].astype(int).to_numpy(),
                                        "date_start": t_start.loc[s1.index].to_numpy()})
                          .sort_values("date_start"))
            if lap_starts.empty:
                continue
            events = extract_control_events(rc, lap_starts)
            for idx in gd.index:
                st, inf = track_status_at(events, True, t_start.loc[idx], gd.at[idx, "Lap"])
                status[out.index.get_loc(idx)] = st
                inferred[out.index.get_loc(idx)] = float(inf)
    changed = int((pd.Series(status, index=out.index).astype("string")
                   != out["Track_Status"].astype("string")).fillna(True).sum())
    out["Track_Status"] = status
    out["Track_Status_Inferred"] = inferred
    return out, changed


# ---------------------------------------------------------------------
# Loaders (แต่ละตัวคืนข้อมูลที่พร้อมใช้ หรือ None)
# ---------------------------------------------------------------------
def resolve_session(year=None, country=None, session_name="Race", session_key=None, fetch_fn=fetch):
    """คืนแถว session (Series) — ระบุ session_key ตรง ๆ ได้ (แนะนำสำหรับประเทศที่มีหลายสนาม เช่น USA)"""
    if session_key is not None:
        sessions = fetch_fn("sessions", session_key=int(session_key))
    else:
        sessions = fetch_fn("sessions", year=year, country_name=country, session_name=session_name)
    if _empty(sessions):
        return None
    if session_key is None and len(sessions) > 1:
        names = sessions.get("circuit_short_name", pd.Series(dtype=str)).tolist()
        print(f"WARNING: {country} {year} {session_name} มี {len(sessions)} sessions {names} "
              f"— ใช้อันแรก; ระบุ session_key=... เพื่อเลือก")
    return sessions.iloc[0]


def _load_laps(laps):
    laps = laps.copy()
    for col in ["date_start", "lap_number"]:
        if col not in laps.columns:
            raise KeyError(f"'{col}' not found in laps. Columns: {laps.columns.tolist()}")
    laps["date_start"] = to_utc(laps["date_start"])
    _numeric_cols(laps, ["lap_number", "duration_sector_1", "duration_sector_2", "duration_sector_3",
                         "i1_speed", "i2_speed", "st_speed"])
    if "is_pit_out_lap" not in laps.columns:
        laps["is_pit_out_lap"] = False
    laps["is_pit_out_lap"] = laps["is_pit_out_lap"].map(lambda v: bool(v) if pd.notna(v) else False)
    return laps.sort_values("lap_number").reset_index(drop=True)


def _trap_speed_lookup(laps):
    """ฟังก์ชัน lap -> (i1, i2, st) ของ lap ล่าสุดที่ <= lap-1 (ffill ย้อนหลังเท่านั้น)"""
    cols = ["i1_speed", "i2_speed", "st_speed"]
    ff = laps[["lap_number"] + cols].dropna(subset=["lap_number"]).sort_values("lap_number").copy()
    ff[cols] = ff[cols].ffill()
    table = {int(r.lap_number): (r.i1_speed, r.i2_speed, r.st_speed) for r in ff.itertuples(index=False)}
    keys = sorted(table)

    def before(lap):
        idx = bisect.bisect_right(keys, lap - 1) - 1
        return table[keys[idx]] if idx >= 0 else (np.nan, np.nan, np.nan)

    return before


def _load_stints(stints):
    stints = stints.copy()
    stints["lap_start"] = pd.to_numeric(stints["lap_start"], errors="coerce")
    # stint สุดท้าย/ที่ยังไม่จบอาจเป็น NaN -> ถือว่าไปจนจบ
    stints["lap_end"] = pd.to_numeric(stints["lap_end"], errors="coerce").fillna(np.inf)
    stints["tyre_age_at_start"] = pd.to_numeric(
        stints.get("tyre_age_at_start", 0), errors="coerce").fillna(0)
    return stints


class CarTelemetry:
    """ค่าเฉลี่ย telemetry ในช่วงเวลาใด ๆ แบบ O(1) ด้วย cumulative sums"""

    def __init__(self, car):
        car = car.copy()
        if "date" not in car.columns:
            raise KeyError(f"'date' not found in car_data. Columns: {car.columns.tolist()}")
        car["date"] = to_utc(car["date"])
        car = car.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        _numeric_cols(car, ["speed", "throttle", "brake", "drs", "n_gear"])

        self.dates = car["date"]
        self.speed = car["speed"].to_numpy(dtype=float)
        throttle, brake, drs = (car[c].to_numpy(dtype=float) for c in ("throttle", "brake", "drs"))

        def indicator(mask, source):   # 1/0 ; NaN ถ้าไม่มีค่าต้นทาง
            return np.where(np.isnan(source), np.nan, mask.astype(float))

        series = {
            "avg_speed": self.speed,
            "full_throttle_pct": indicator(throttle >= 99, throttle),
            "brake_pct": indicator(brake > 0, brake),
            "drs_pct": indicator(np.isin(drs, [10, 12, 14]), drs),
            "avg_gear": car["n_gear"].to_numpy(dtype=float),
        }
        self.csums = {}
        for name, arr in series.items():
            ok = ~np.isnan(arr)
            self.csums[name] = (np.r_[0.0, np.cumsum(np.where(ok, arr, 0.0))], np.r_[0, np.cumsum(ok)])

        # conditional speed: corner (ไม่ full throttle) vs straight (full throttle)
        thr_ok = ~np.isnan(throttle)
        speed_ok = ~np.isnan(self.speed)
        self._cond_speed = {}
        for name, mask in (
            ("corner", thr_ok & (throttle < 99) & speed_ok),
            ("straight", thr_ok & (throttle >= 99) & speed_ok),
        ):
            self._cond_speed[name] = (
                np.r_[0.0, np.cumsum(np.where(mask, self.speed, 0.0))],
                np.r_[0, np.cumsum(mask)],
            )

    def window(self, start, end):
        i0 = self.dates.searchsorted(start, side="left")
        i1 = self.dates.searchsorted(end, side="left")
        out = {}
        for name, (csum, count) in self.csums.items():
            n = count[i1] - count[i0]
            mean = (csum[i1] - csum[i0]) / n if n > 0 else np.nan
            out[name] = mean * 100 if name.endswith("_pct") else mean
        for name, (csum, count) in self._cond_speed.items():
            n = count[i1] - count[i0]
            out[f"{name}_speed"] = (csum[i1] - csum[i0]) / n if n > 0 else np.nan
        seg = self.speed[i0:i1]
        has = np.isfinite(seg).any()
        out["max_speed"] = np.nanmax(seg) if has else np.nan
        out["min_speed"] = np.nanmin(seg) if has else np.nan
        return out


def _load_pits(pits):
    """{lap_number: pit_duration}"""
    if _empty(pits):
        return {}
    pits = pits.copy()
    if "lap_number" not in pits.columns:
        raise KeyError(f"'lap_number' not in pit data. Columns: {pits.columns.tolist()}")
    _numeric_cols(pits, ["lap_number", "pit_duration"])
    pits = pits.dropna(subset=["lap_number"]).drop_duplicates("lap_number", keep="first")
    return {int(r.lap_number): r.pit_duration for r in pits.itertuples(index=False)}


def _load_weather(weather):
    if _empty(weather):
        return None
    weather = _lower_columns(weather, "weather")
    _numeric_cols(weather, list(WEATHER_MAP))
    return (weather.dropna(subset=["date"]).sort_values("date")
            .drop_duplicates("date", keep="last")[["date", *WEATHER_MAP]].reset_index(drop=True))


def _load_intervals(intervals, far_value):
    if _empty(intervals):
        return None
    intervals = _lower_columns(intervals, "intervals")
    raw = intervals["interval"] if "interval" in intervals.columns else pd.Series(np.nan, index=intervals.index)
    # ผู้นำ (None) และรถที่ถูกแซง lap ("+1 LAP") -> ค่าไกล
    is_far = raw.isna() | raw.astype(str).str.upper().str.contains("LAP")
    intervals["Interval_Ahead"] = pd.to_numeric(raw, errors="coerce").mask(is_far, far_value).clip(upper=far_value)
    return (intervals.dropna(subset=["date"]).sort_values("date")
            .drop_duplicates("date", keep="last")[["date", "Interval_Ahead"]].reset_index(drop=True))


class TrafficLookup:
    """นับรถคันอื่นที่อยู่ในระยะ gap_to_leader ± threshold ณ เวลาที่กำหนด (as-of backward)

    gap ระหว่างคัน X กับ Y ประมาณ |gap_to_leader_X - gap_to_leader_Y|
    (leader: gap_to_leader ≈ 0; รถที่ถูกแซง lap มักเป็น null -> ข้ามไป)
    """

    def __init__(self, by_driver):
        self.by_driver = by_driver        # {driver_number: (DatetimeIndex, gap_to_leader np.array)}

    def _gap_at(self, driver, t):
        arr = self.by_driver.get(int(driver))
        if arr is None:
            return None
        dates, gaps = arr
        i = dates.searchsorted(t, side="right") - 1
        return float(gaps[i]) if i >= 0 else None

    def counts(self, driver, t, thresholds=TRAFFIC_THRESHOLDS):
        """{Traffic_Within_<N>s: int} — นับรถคันอื่น (ไม่รวมตัวเอง) ที่ |Δgap| <= threshold"""
        self_gap = self._gap_at(driver, t)
        if self_gap is None or not np.isfinite(self_gap):
            return {_traffic_col(s): np.nan for s in thresholds}

        gaps = []
        for d, (dates, gs) in self.by_driver.items():
            if int(d) == int(driver):
                continue
            i = dates.searchsorted(t, side="right") - 1
            if i >= 0 and np.isfinite(gs[i]):
                gaps.append(gs[i])
        if not gaps:
            return {_traffic_col(s): 0 for s in thresholds}

        diff = np.abs(np.asarray(gaps, dtype=float) - self_gap)
        return {_traffic_col(s): int(np.count_nonzero(diff <= s)) for s in thresholds}


def _load_traffic(intervals_all):
    """intervals ทั้ง session (ทุกคัน) -> TrafficLookup (gap_to_leader as-of ต่อคัน)"""
    if _empty(intervals_all):
        return None
    df = _lower_columns(intervals_all, "intervals")
    if "gap_to_leader" not in df.columns or "driver_number" not in df.columns:
        return None
    df["gap_to_leader"] = pd.to_numeric(df["gap_to_leader"], errors="coerce")
    df["driver_number"] = pd.to_numeric(df["driver_number"], errors="coerce")
    df = df.dropna(subset=["date", "driver_number", "gap_to_leader"]).sort_values("date")
    if df.empty:
        return None
    by_driver = {}
    for d, g in df.groupby("driver_number", sort=False):
        g = g.sort_values("date")
        by_driver[int(d)] = (pd.DatetimeIndex(g["date"]), g["gap_to_leader"].to_numpy(dtype=float))
    return TrafficLookup(by_driver)


def _asof_join(result, right, tolerance=None):
    right = right.copy()
    right["date"] = right["date"].astype("datetime64[ns, UTC]")
    return pd.merge_asof(result.sort_values("Time_Start"), right,
                         left_on="Time_Start", right_on="date",
                         direction="backward",          # ห้ามมองอนาคต
                         tolerance=tolerance).drop(columns="date")


# ---------------------------------------------------------------------
# Core steps
# ---------------------------------------------------------------------
def _sector_records(laps, stints, telemetry, events, rc_available):
    records = []
    for row in laps.itertuples(index=False):
        lap_start, lap_num = row.date_start, row.lap_number
        if pd.isna(lap_start) or pd.isna(lap_num):
            continue
        lap_num = int(lap_num)

        match = stints[(stints["lap_start"] <= lap_num) & (stints["lap_end"] >= lap_num)]
        if match.empty:
            continue
        stint = match.iloc[0]
        tyre_age = int(lap_num - stint["lap_start"] + 1)

        current = lap_start
        durations = [row.duration_sector_1, row.duration_sector_2, row.duration_sector_3]
        for sector, duration in enumerate(durations, start=1):
            if pd.isna(duration) or duration <= 0:
                break
            s_start, s_end = current, current + pd.Timedelta(seconds=float(duration))
            status, inferred = track_status_at(events, rc_available, s_start, lap_num)
            tele = telemetry.window(s_start, s_end)
            records.append({
                "Lap": lap_num, "Sector": sector, "Time_Start": s_start,
                "Tyre": stint["compound"],
                "Tyre_Age": tyre_age,
                "Tyre_Age_Total": tyre_age + int(stint["tyre_age_at_start"]),
                "Track_Status": status, "Track_Status_Inferred": inferred,
                # --- outcome ของ sector ปัจจุบัน (ไม่ใช่ feature) ---
                "Outcome_Duration": float(duration),
                "Outcome_Time_End": s_end,
                "Outcome_Avg_Speed": tele["avg_speed"],
                "Outcome_Avg_Speed_Corners": tele["corner_speed"],
                "Outcome_Avg_Speed_Straight": tele["straight_speed"],
                "Outcome_Full_Throttle_Pct": tele["full_throttle_pct"],
                "Outcome_Brake_Pct": tele["brake_pct"],
                "Outcome_DRS_Pct": tele["drs_pct"],
                "Outcome_Max_Speed": tele["max_speed"],
                "Outcome_Min_Speed": tele["min_speed"],
                "Outcome_Avg_Gear": tele["avg_gear"],
            })
            current = s_end
    return records


def _add_history_features(records, pit_info, out_laps, speeds_before, clean_window):
    """เติม Prev_* / Clean_Pace / pit features จากข้อมูลย้อนหลังเท่านั้น (แก้ records in-place)"""
    pit_laps = set(pit_info)
    by_key = {(r["Lap"], r["Sector"]): r for r in records}
    sec_laps = {s: sorted(l for (l, ss) in by_key if ss == s) for s in (1, 2, 3)}

    # lap สะอาด = ไม่ใช่ lap 1, ไม่ใช่ in/out-lap, สถานะ GREEN (หรือ UNKNOWN ถ้าไม่มี race control)
    for r in records:
        lap = r["Lap"]
        r["_clean"] = bool(lap > 1 and lap not in pit_laps and (lap - 1) not in pit_laps
                           and lap not in out_laps and r["Track_Status"] in {"GREEN", "UNKNOWN"})

    for i, r in enumerate(records):
        lap, sec = r["Lap"], r["Sector"]
        past = sec_laps[sec][: bisect.bisect_left(sec_laps[sec], lap)]
        history = [by_key[(l, sec)] for l in past]
        durations = [h["Outcome_Duration"] for h in history]

        r["Prev_Duration"] = durations[-1] if durations else np.nan
        for out_col, src_col in TELEMETRY_HISTORY.items():
            r[out_col] = _last_valid(history, src_col)
        r["Prev_Sector_Duration"] = records[i - 1]["Outcome_Duration"] if i > 0 else np.nan
        r["Prev_I1_Speed"], r["Prev_I2_Speed"], r["Prev_St_Speed"] = speeds_before(lap)

        clean = [h["Outcome_Duration"] for h in history if h["_clean"]]
        pool = clean or durations          # fallback: ยังไม่มี lap สะอาด
        r["Clean_Pace"] = float(np.median(pool[-clean_window:])) if pool else np.nan
        r["Best_Duration_So_Far"] = min(durations) if durations else np.nan

        # Pit: ใช้เฉพาะสิ่งที่เกิดและจบก่อน Time_Start
        r["Prev_Lap_Pitted"] = int((lap - 2) in pit_laps)
        r["Pit_Stops_Done"] = sum(1 for p in pit_laps if p < lap)
        if sec == 1 and (lap - 2) in pit_info:
            dur = pit_info[lap - 2]
            r["Prev_Lap_Box_Duration"] = float(dur) if pd.notna(dur) else np.nan
        else:
            r["Prev_Lap_Box_Duration"] = 0.0
        r["Will_Box"] = int(sec == 1 and (lap - 1) in pit_laps)


# ---------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------
def build_sector_table(
    driver_number,
    country=None,
    year=None,
    session_name="Race",
    *,
    session_key=None,
    ctx=None,
    fetch_fn=fetch,
    include_outcome_columns=False,
    weather_tolerance=pd.Timedelta(minutes=2),
    clean_window=3,
    interval_fill=INTERVAL_FAR_DEFAULT,
    drop_no_history=True,
    require_track_profile=False,
    verbose=True,
):
    """สร้าง dataset ระดับ lap-sector ของนักแข่ง 1 คนใน 1 session -> DataFrame หรือ None

    ctx : TrackContext | None
        ถ้าส่งมา จะเติม Total_Sector_Length / Total_Straight_Length / Num_Corner_*
    interval_fill : float | None
        ค่าแทน Interval_Ahead เมื่อไม่มีรถคันหน้า/ไม่มีข้อมูล; None = ปล่อย NaN
    drop_no_history : ตัด lap แรกของ driver (Prev_* ไม่มีความหมาย)
    require_track_profile : True และหา profile ไม่ครบ -> คืน None
    include_outcome_columns : export Outcome_* ไว้วิเคราะห์ (ห้ามใช้เป็น feature)
    """
    log = print if verbose else (lambda *a, **k: None)
    far_value = INTERVAL_FAR_DEFAULT if interval_fill is None else float(interval_fill)

    # 1. Session
    session = resolve_session(year, country, session_name, session_key, fetch_fn)
    if session is None:
        log(f"No session found for {year} {country} {session_name}")
        return None
    s_key = int(session["session_key"])
    year = year if year is not None else session.get("year")
    country = country if country is not None else session.get("country_name")
    session_name = session.get("session_name", session_name)
    label = f"{year} {country} {session_name} #{driver_number}"

    # 2-3. Laps + stints
    raw_laps = fetch_fn("laps", session_key=s_key, driver_number=driver_number)
    if _empty(raw_laps):
        log(f"{label}: No laps data found.")
        return None
    laps = _load_laps(raw_laps)
    lap_starts = laps[["lap_number", "date_start"]].dropna().sort_values("date_start")
    out_laps = set(int(n) for n in laps.loc[laps["is_pit_out_lap"], "lap_number"].dropna())

    raw_stints = fetch_fn("stints", session_key=s_key, driver_number=driver_number)
    if _empty(raw_stints):
        log(f"{label}: No stints data found.")
        return None
    stints = _load_stints(raw_stints)

    # 4. Car telemetry
    car = fetch_fn("car_data", session_key=s_key, driver_number=driver_number)
    if _empty(car):
        log(f"{label}: No car data found.")
        return None
    telemetry = CarTelemetry(car)

    # 5. Race control
    rc = fetch_fn("race_control", session_key=s_key)
    rc_available = not _empty(rc)
    events = extract_control_events(rc, lap_starts) if rc_available else []
    log(f"{label}: SC/VSC/Green events = {len(events)}" if rc_available
        else f"{label}: Warning — no race_control; Track_Status = UNKNOWN")

    # 6-7. Pits, weather, intervals
    pit_info = _load_pits(fetch_fn("pit", session_key=s_key, driver_number=driver_number))
    weather = _load_weather(fetch_fn("weather", session_key=s_key))
    if weather is None:
        log(f"{label}: Warning — no weather data")
    intervals_all = fetch_fn("intervals", session_key=s_key)
    if not _empty(intervals_all):
        self_mask = pd.to_numeric(intervals_all.get("driver_number"), errors="coerce").eq(driver_number)
        intervals_self = intervals_all[self_mask]
    else:
        intervals_self = intervals_all
    intervals = _load_intervals(intervals_self, far_value)
    if intervals is None:
        log(f"{label}: Warning — no intervals data (race only)")
    traffic = _load_traffic(intervals_all) if not _empty(intervals_all) else None

    # 8. Sector records + 9. history features
    records = _sector_records(laps, stints, telemetry, events, rc_available)
    if not records:
        log(f"{label}: No sector records could be created.")
        return None
    _add_history_features(records, pit_info, out_laps, _trap_speed_lookup(laps), clean_window)

    result = pd.DataFrame(records)
    result["Time_Start"] = result["Time_Start"].astype("datetime64[ns, UTC]")

    if drop_no_history:
        before = len(result)
        result = result[result["Prev_Duration"].notna()].copy()
        log(f"{label}: dropped {before - len(result)} rows without history")
        if result.empty:
            return None

    # 10. As-of joins
    if weather is not None and not weather.empty:
        result = _asof_join(result, weather, tolerance=weather_tolerance)
    else:
        for col in WEATHER_MAP:
            result[col] = np.nan
    result = result.rename(columns=WEATHER_MAP)

    if intervals is not None and not intervals.empty:
        result = _asof_join(result, intervals)
    else:
        result["Interval_Ahead"] = np.nan
    if interval_fill is not None:
        result["Interval_Ahead"] = result["Interval_Ahead"].fillna(far_value)

    # 11. Meta + column order
    result["Year"], result["Country"], result["Session"] = year, country, session_name
    result["Session_Key"], result["Driver"] = s_key, driver_number
    result[TARGET_COLUMN] = result["Outcome_Duration"]

    columns = META_COLUMNS + FEATURE_COLUMNS
    result = result.sort_values(["Lap", "Sector"]).reset_index(drop=True)

    # 11.5 Traffic: นับรถคันอื่นที่อยู่ในระยะ gap ณ Time_Start (as-of backward)
    if traffic is not None:
        rows = [traffic.counts(driver_number, ts) for ts in result["Time_Start"]]
        for col in TRAFFIC_COLUMNS:
            result[col] = [r.get(col, np.nan) for r in rows]
    else:
        for col in TRAFFIC_COLUMNS:
            result[col] = np.nan

    # 12. Track profile features
    profile_cols = []
    if ctx is not None:
        result = add_track_profile(result, ctx, session.get("circuit_short_name"),
                                   require=require_track_profile, verbose=verbose, label=label)
        if result is None:
            return None
        profile_cols = ctx.profile_columns

    # 13. Distance features: circuit length × laps (รวม / บนยางชุดนี้)
    if "Circuit_Total_Length" in result.columns:
        length = result["Circuit_Total_Length"]
        result["Total_Distance_Driven"] = length * result["Lap"]
        result["Distance_On_Tyre"] = length * result["Tyre_Age_Total"]
    else:
        result["Total_Distance_Driven"] = np.nan
        result["Distance_On_Tyre"] = np.nan

    columns += profile_cols + ["Total_Distance_Driven", "Distance_On_Tyre"] + [TARGET_COLUMN]
    if include_outcome_columns:
        columns += OUTCOME_COLUMNS
    return result[columns]

