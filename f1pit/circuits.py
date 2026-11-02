"""สร้างฐานข้อมูลสนาม + โค้ง ของทั้งฤดูกาล (เดิม create_circuit_db)

>>> circuits, corners = f1pit.create_circuit_db(2024)
"""

import numpy as np
import pandas as pd
import requests

from . import config
from .api import fetch
from .tracks import (
    calculate_corner_angle,
    get_best_race_lap_track,
    get_track_by_name,
    gp_layout,
    point_at_fraction,
    range_layer,
    sector_bounds,
)

_NAN_GEOMETRY = dict.fromkeys(["Angle", "Angle_Magnitude", "Incoming_Heading", "Outgoing_Heading"], np.nan)


def fetch_circuit_info(url):
    """JSON ข้อมูลโค้งจาก MultiViewer (circuit_info_url ของ meeting)"""
    r = requests.get(url, headers=config.BROWSER_HEADERS, timeout=config.REQUEST_TIMEOUT)
    r.raise_for_status()
    return r.json()


def _circuit_row(meeting_key, track, bounds):
    length = bounds["length_m"]
    row = {
        "Meeting_Key": meeting_key,
        "Track_Slug": track.get("slug"),
        "Circuit_Name": track["name"],
        "Circuit_Total_Length": length,
        "Circuit_Direction": bounds["direction"],
    }
    for i, (start, end) in enumerate(bounds["sectors"], start=1):
        row[f"S{i}_Length"] = (end - start) * length
    for i, (start, end) in enumerate(bounds["sectors"], start=1):
        row[f"S{i}_Start"], row[f"S{i}_End"] = start, end
    return row


def _corner_rows(meeting_key, track, gp, track_path, info, verbose):
    corner_layer = range_layer(gp, "corner_ranges")
    if corner_layer is None:
        return []
    ranges = {item["id"]: item for item in corner_layer["items"]}

    rows = []
    for corner in info.get("corners", []):
        number = corner["number"]
        if corner.get("letter") is not None:
            verbose and print(f"SKIP: {info.get('circuitName')} corner {number}{corner['letter']}")
            continue
        rng = ranges.get(f"t{number}")
        if rng is None:
            verbose and print(f"SKIP: {info.get('circuitName')} corner {number} ไม่มีใน corner_ranges")
            continue

        start, end = rng["start"], rng["end"]
        geometry = calculate_corner_angle(track_path, start, end) or _NAN_GEOMETRY
        p_start = point_at_fraction(track_path, start)
        p_end = point_at_fraction(track_path, end)

        rows.append({
            "Meeting_Key": meeting_key,
            "Track_Slug": track.get("slug"),
            "Circuit_Name": info.get("circuitName", track["name"]),
            "Corner": number,
            **geometry,                              # มุมจาก XY telemetry
            "Marker_Angle": corner.get("angle"),     # มุม marker เดิมของ MultiViewer
            "Corner_Start": start,
            "Corner_End": end,
            "Corner_Length": ((end - start) % 1.0) * gp["length_m"],
            "Corner_Start_X": p_start[0],
            "Corner_Start_Y": p_start[1],
            "Corner_End_X": p_end[0],
            "Corner_End_Y": p_end[1],
        })
    return rows


def create_circuit_db(year, verbose=True):
    """Returns (circuits_df, corners_df) ของทุก meeting ในปีนั้น (ไม่รวม Pre-Season Testing)

    meeting ที่ข้อมูลไม่ครบจะถูกข้ามพร้อมแจ้งเตือน (ไม่หยุดทั้งปี)
    """
    meetings = fetch("meetings", year=year, is_cancelled=False)
    meetings = meetings[~meetings["meeting_name"].str.contains("Pre-Season Testing", na=False)]

    circuits, corners = [], []
    for item in meetings.itertuples(index=False):
        meeting_key = int(item.meeting_key)
        short_name = item.circuit_short_name

        track = get_track_by_name(short_name)
        bounds = sector_bounds(track)
        if track is None or bounds is None:
            print(f"WARNING: ไม่พบ track / GP layout / timing_sectors ของ '{short_name}' — ข้าม")
            continue

        track_path, _ = get_best_race_lap_track(meeting_key, verbose=verbose)
        if track_path is None:
            continue

        circuits.append(_circuit_row(meeting_key, track, bounds))

        try:
            info = fetch_circuit_info(item.circuit_info_url)
        except Exception as exc:
            print(f"WARNING: โหลด circuit info ของ '{short_name}' ไม่ได้: {exc}")
            continue
        corners += _corner_rows(meeting_key, track, gp_layout(track), track_path, info, verbose)

    return pd.DataFrame(circuits), pd.DataFrame(corners)
