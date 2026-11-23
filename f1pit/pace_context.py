"""Pace-context features (point-in-time) + relative-pace transform สำหรับ forest-of-forests

add_pace_context(raw)   : เติม feature บริบทของ lap จากข้อมูลย้อนหลังเท่านั้น
    - Rep_* anchors   : เวลา sector ของ "lap ตัวแทน" ล่าสุด (GREEN/UNKNOWN และไม่ช้ากว่า
                        REP_TOL x median ของ 5 lap เขียวก่อนหน้า) -> ไม่ใช้ lap ที่โดน SC / pit /
                        อุบัติเหตุเป็นจุดอ้างอิง (Prev_Duration ดิบเพี้ยนใน lap เหล่านั้น)
    - Prev_Lap_NonGreen, Laps_Since_SC : สถานะ SC/VSC ของ lap ก่อน ๆ
    - Rainfall_PrevLap / Rainfall_Delta / Field_Wet_PrevLap : ประวัติฝน + สัดส่วนรถที่ใช้ยางฝน lap ก่อน
    - Stint_Number / Is_First_Stint / Tyre_Age_x_Track_Temp : บริบทยาง (ใช้ใน stage-1)
add_relative_pace(df, anchor) : แปลงเวลาอ้างอิงเป็น log-ratio เทียบ anchor (ไม่ขึ้นกับความยาวสนาม)

ทุก feature ใช้เฉพาะ lap ก่อนหน้า (หรือ sector ก่อนหน้า) -> point-in-time ถูกต้อง
ต้องเรียกกับตารางทั้ง session (ต้องมี lap ก่อนหน้าครบ) ก่อน _prepare_lab / remove outliers
"""

import numpy as np
import pandas as pd

REP_TOL = 1.12          # lap ตัวแทน: Duration <= REP_TOL x median(5 lap เขียวก่อนหน้า)
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}
KEY = ["Session_Key", "Driver", "Sector"]

REP_COLUMNS = ["Rep_Prev_Duration", "Rep_Best_Duration", "Rep_Median3", "Anchor_Age"]
CONTEXT_COLUMNS = ["Prev_Lap_NonGreen", "Laps_Since_SC", "Prev_Lap_Pitted", "Pit_Stops_Done", "Lap"]
RAIN_COLUMNS = ["Rainfall_PrevLap", "Rainfall_Delta", "Field_Wet_PrevLap"]
TYRE_CONTEXT_COLUMNS = ["Stint_Number", "Is_First_Stint", "Tyre_Age_x_Track_Temp"]
PACE_CONTEXT_COLUMNS = [*REP_COLUMNS, *[c for c in CONTEXT_COLUMNS if c != "Lap"], *RAIN_COLUMNS,
                        *TYRE_CONTEXT_COLUMNS]

# เวลาอ้างอิง (sector เดียวกัน) ที่แปลงเป็น log(x / anchor)
RELATIVE_SOURCES = ["Prev_Duration", "Clean_Pace", "Best_Duration_So_Far", "Prev_Duration_roll5_mean",
                    "Rep_Prev_Duration", "Rep_Best_Duration", "Rep_Median3"]


def _num(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def _green(df):
    return df["Track_Status"].astype(str).str.upper().isin(["GREEN", "UNKNOWN"]).to_numpy()


def add_pace_context(raw, tol=REP_TOL):
    """raw (ตารางจาก build_dataset) -> DataFrame เดิม + PACE_CONTEXT_COLUMNS (เรียงใหม่ตาม KEY + Lap)"""
    df = raw.copy()
    df = df.sort_values(KEY + ["Lap"]).reset_index(drop=True)
    dur = _num(df, "Duration").to_numpy(float)
    green = _green(df)
    lap = _num(df, "Lap").to_numpy(float)

    # ---- representative-lap anchors ----
    df["_gd"] = np.where(green, dur, np.nan)
    g = df.groupby(KEY, sort=False)
    ref = g["_gd"].transform(lambda s: s.rolling(5, min_periods=1).median().shift(1)).to_numpy()
    rep = green & (dur <= tol * np.where(np.isfinite(ref), ref, np.inf))
    df["_rd"] = np.where(rep, dur, np.nan)
    df["_ri"] = np.where(rep, dur, np.inf)
    df["_rl"] = np.where(rep, lap, np.nan)
    g = df.groupby(KEY, sort=False)
    df["Rep_Prev_Duration"] = (g["_rd"].transform(lambda s: s.ffill().shift(1))
                               .fillna(_num(df, "Prev_Duration")).to_numpy())
    df["Rep_Best_Duration"] = (g["_ri"].transform(lambda s: s.cummin().shift(1)).replace(np.inf, np.nan)
                               .fillna(_num(df, "Best_Duration_So_Far")).to_numpy())
    df["Anchor_Age"] = lap - g["_rl"].transform(lambda s: s.ffill().shift(1)).to_numpy()
    m3 = g["_rd"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).median())
    df["Rep_Median3"] = m3.fillna(df["Rep_Prev_Duration"]).to_numpy()
    df = df.drop(columns=["_gd", "_rd", "_ri", "_rl"])

    # ---- lap ก่อนหน้าของ driver นี้ไม่เขียว (sector ใดก็ได้) ----
    df["_ng"] = (~green).astype(int)
    lap_ng = (df.groupby(["Session_Key", "Driver", "Lap"])["_ng"].max().reset_index()
              .sort_values(["Session_Key", "Driver", "Lap"]))
    lap_ng["Prev_Lap_NonGreen"] = lap_ng.groupby(["Session_Key", "Driver"])["_ng"].shift(1).fillna(0)
    df = df.merge(lap_ng[["Session_Key", "Driver", "Lap", "Prev_Lap_NonGreen"]],
                  on=["Session_Key", "Driver", "Lap"], how="left").drop(columns=["_ng"])

    # ---- จำนวน lap นับจาก SC/VSC ล่าสุดของ session (เฉพาะ lap ก่อนหน้า) ----
    df["_sc"] = df["Track_Status"].astype(str).str.upper().isin(SC_SET).astype(int)
    lap_sc = df.groupby(["Session_Key", "Lap"])["_sc"].max().reset_index().sort_values(["Session_Key", "Lap"])
    lap_sc["_scl"] = lap_sc["Lap"].where(lap_sc["_sc"] > 0)
    lap_sc["_last"] = lap_sc.groupby("Session_Key")["_scl"].transform(lambda s: s.ffill().shift(1))
    lap_sc["Laps_Since_SC"] = (lap_sc["Lap"] - lap_sc["_last"]).fillna(99.0).clip(upper=99.0)
    df = df.merge(lap_sc[["Session_Key", "Lap", "Laps_Since_SC"]], on=["Session_Key", "Lap"],
                  how="left").drop(columns=["_sc"])

    # ---- ฝน: lap ก่อน (sector เดียวกัน) + สัดส่วนรถทั้งสนามที่ใช้ยางฝนใน lap ก่อน ----
    df = df.sort_values(KEY + ["Lap"]).reset_index(drop=True)
    rain = _num(df, "Rainfall" if "Rainfall" in df.columns else "rainfall")
    df["Rainfall_PrevLap"] = rain.groupby([df[k] for k in KEY]).shift(1)
    df["Rainfall_Delta"] = rain - df["Rainfall_PrevLap"]
    df["_wt"] = df["Tyre"].astype(str).str.upper().isin(["INTERMEDIATE", "WET"]).astype(float)
    lw = df.groupby(["Session_Key", "Lap"])["_wt"].mean().reset_index().sort_values(["Session_Key", "Lap"])
    lw["Field_Wet_PrevLap"] = lw.groupby("Session_Key")["_wt"].shift(1)
    df = df.merge(lw[["Session_Key", "Lap", "Field_Wet_PrevLap"]], on=["Session_Key", "Lap"],
                  how="left").drop(columns=["_wt"])

    # ---- บริบทยาง ----
    stops = _num(df, "Pit_Stops_Done")
    df["Stint_Number"] = stops + 1
    df["Is_First_Stint"] = (stops == 0).astype(float)
    temp = _num(df, "Track_Temperature" if "Track_Temperature" in df.columns else "track_temperature")
    df["Tyre_Age_x_Track_Temp"] = _num(df, "Tyre_Age_Total") * temp
    return df.sort_values(KEY + ["Lap"]).reset_index(drop=True)


def add_relative_pace(df, anchor="Rep_Prev_Duration"):
    """เติม Rel_<col> = log(col / anchor) และ Rel_roll5_slope = slope / anchor

    คืน (df ใหม่, mapping {คอลัมน์เดิม: คอลัมน์ Rel_}) ไว้แทนชื่อใน feature list
    """
    out = df.copy()
    a = _num(out, anchor)
    mapping = {}
    for c in RELATIVE_SOURCES:
        if c in out.columns and c != anchor:
            out[f"Rel_{c}"] = np.log(_num(out, c) / a)
            mapping[c] = f"Rel_{c}"
    if "Prev_Duration_roll5_slope" in out.columns:
        out["Rel_roll5_slope"] = _num(out, "Prev_Duration_roll5_slope") / a
        mapping["Prev_Duration_roll5_slope"] = "Rel_roll5_slope"
    return out, mapping
