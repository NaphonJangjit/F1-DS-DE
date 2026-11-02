"""ข้อมูลสนาม: tracks.json lookup + เส้นทางสนามจาก telemetry + เรขาคณิตโค้ง"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .api import fetch, to_utc


# ---------------------------------------------------------------------
# tracks.json
# ---------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_tracks_cached(path_str):
    with open(path_str, "r", encoding="utf-8") as f:
        return json.load(f)


def load_tracks(path=None):
    """โหลด tracks.json (cache ไว้ใน memory; เปลี่ยน path ผ่าน config.TRACKS_JSON)"""
    path = Path(path or config.TRACKS_JSON)
    if not path.exists():
        raise FileNotFoundError(
            f"ไม่พบ {path} — ตั้งค่า f1pit.config.TRACKS_JSON = 'path/to/tracks.json'"
        )
    return _load_tracks_cached(str(path.resolve()))


def _aliases(track):
    names = [track.get("name"), track.get("slug"), track.get("location.locality"),
             *track.get("aka", [])]
    return {str(n).lower() for n in names if n}


def get_track_by_name(name):
    """หา track จากชื่อ / aka / slug / locality (ไม่สนตัวพิมพ์); ไม่เจอ -> None"""
    if not name:
        return None
    key = str(name).lower()
    return next((t for t in load_tracks() if key in _aliases(t)), None)


def _find_by_id(items, item_id):
    return next((x for x in items or [] if x.get("id") == item_id), None)


def gp_layout(track):
    """layout id='gp' ของสนาม; ไม่มี -> None"""
    return _find_by_id((track or {}).get("layouts"), "gp")


def range_layer(layout, layer_id):
    """range layer เช่น 'timing_sectors', 'corner_ranges'; ไม่มี -> None"""
    return _find_by_id((layout or {}).get("range_layers"), layer_id)


def sector_bounds(track):
    """{'length_m', 'direction', 'sectors': [(start, end) x3]} จาก GP layout; ไม่ครบ -> None

    start/end เป็นสัดส่วนของรอบ (0..1)
    """
    gp = gp_layout(track)
    timing = range_layer(gp, "timing_sectors")
    if gp is None or timing is None or len(timing.get("items", [])) < 3:
        return None
    return {
        "length_m": gp.get("length_m"),
        "direction": gp.get("direction"),
        "sectors": [(it["start"], it["end"]) for it in timing["items"][:3]],
    }


# ---------------------------------------------------------------------
# Track path จาก location telemetry
# ---------------------------------------------------------------------
def prepare_track_path(location):
    """สร้าง XY path ที่ index ด้วยระยะทางสะสม จาก telemetry 1 รอบ"""
    loc = location.copy()
    loc["date"] = to_utc(loc["date"])
    loc["x"] = pd.to_numeric(loc["x"], errors="coerce")
    loc["y"] = pd.to_numeric(loc["y"], errors="coerce")
    loc = (
        loc.dropna(subset=["date", "x", "y"])
        .sort_values("date")
        .drop_duplicates(subset=["date"])
        .reset_index(drop=True)
    )
    if len(loc) < 30:
        raise ValueError("Not enough location samples to reconstruct a lap.")

    # ตัดจุดที่รถไม่ขยับ (พิกัดซ้ำ)
    xy = loc[["x", "y"]].to_numpy(dtype=float)
    keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > 1e-6]
    loc = loc.loc[keep].reset_index(drop=True)

    xy = loc[["x", "y"]].to_numpy(dtype=float)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if distance[-1] <= 0:
        raise ValueError("Could not reconstruct track distance.")

    return {
        "data": loc,
        "x": xy[:, 0],
        "y": xy[:, 1],
        "distance": distance,
        "length_m": float(distance[-1]),
    }


def get_best_race_lap_track(meeting_key, verbose=True):
    """XY path ของ fastest valid lap ใน GP Race ของ meeting นี้

    Returns (track_path, lap_info) หรือ (None, None)
    """
    sessions = fetch("sessions", meeting_key=int(meeting_key))
    if sessions.empty:
        return None, None
    races = sessions[sessions["session_name"].eq("Race")]
    if races.empty:
        verbose and print(f"No Race session for meeting {meeting_key}")
        return None, None
    session_key = int(races.sort_values("date_start").iloc[-1]["session_key"])

    laps = fetch("laps", session_key=session_key)
    if laps.empty:
        return None, None
    laps["lap_duration"] = pd.to_numeric(laps["lap_duration"], errors="coerce")
    valid = laps.dropna(subset=["date_start", "lap_duration", "driver_number"])
    valid = valid[valid["lap_duration"] > 0]
    if "is_pit_out_lap" in valid.columns:
        valid = valid[~valid["is_pit_out_lap"].fillna(False).astype(bool)]
    if valid.empty:
        verbose and print(f"No valid timed laps in session {session_key}")
        return None, None

    best = valid.loc[valid["lap_duration"].idxmin()]
    lap_start = pd.to_datetime(best["date_start"], utc=True)
    lap_end = lap_start + pd.to_timedelta(float(best["lap_duration"]), unit="s")
    driver_number = int(best["driver_number"])

    location = fetch("location", session_key=session_key, driver_number=driver_number)
    if location.empty:
        return None, None
    location["date"] = to_utc(location["date"])
    lap_location = location[location["date"].between(lap_start, lap_end)]

    try:
        track_path = prepare_track_path(lap_location)
    except ValueError as exc:
        verbose and print(f"Could not reconstruct lap: {exc}")
        return None, None

    lap_info = {
        "meeting_key": int(meeting_key),
        "session_key": session_key,
        "driver_number": driver_number,
        "lap_number": int(best["lap_number"]),
        "lap_duration": float(best["lap_duration"]),
        "lap_start": lap_start,
    }
    if verbose:
        print(f"Meeting {meeting_key}: driver {driver_number}, "
              f"lap {lap_info['lap_number']}, {lap_info['lap_duration']:.3f} s")
    return track_path, lap_info


# ---------------------------------------------------------------------
# Corner geometry
# ---------------------------------------------------------------------
def point_at_fraction(track_path, fraction):
    """พิกัด XY ที่สัดส่วน `fraction` ของรอบ (interpolate)"""
    distance = (float(fraction) % 1.0) * track_path["length_m"]
    s = track_path["distance"]
    return np.array([np.interp(distance, s, track_path["x"]),
                     np.interp(distance, s, track_path["y"])])


def calculate_corner_angle(track_path, corner_start, corner_end, lookahead_fraction=0.003):
    """มุมเลี้ยวแบบมีเครื่องหมาย (องศา) จากทิศทางขาเข้า/ขาออกของโค้ง; คำนวณไม่ได้ -> None"""
    if track_path is None:
        return None
    start, end = float(corner_start) % 1.0, float(corner_end) % 1.0

    v_in = point_at_fraction(track_path, start) - point_at_fraction(track_path, start - lookahead_fraction)
    v_out = point_at_fraction(track_path, end + lookahead_fraction) - point_at_fraction(track_path, end)
    if np.linalg.norm(v_in) < 1e-9 or np.linalg.norm(v_out) < 1e-9:
        return None

    dot = np.dot(v_in, v_out)
    cross = v_in[0] * v_out[1] - v_in[1] * v_out[0]
    angle = float(np.degrees(np.arctan2(cross, dot)))
    return {
        "Angle": angle,
        "Angle_Magnitude": abs(angle),
        "Incoming_Heading": float(np.degrees(np.arctan2(v_in[1], v_in[0]))),
        "Outgoing_Heading": float(np.degrees(np.arctan2(v_out[1], v_out[0]))),
    }
