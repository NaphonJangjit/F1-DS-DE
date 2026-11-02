"""TrackContext: รวม circuits + corners ที่ cluster แล้ว ไว้ในวัตถุเดียว
(แทน global `circuits`, `corners_clustered_df`, `KMODE_CLUSTERS` ใน Colab)

>>> ctx = f1pit.build_track_context(2024)       # โหลด + cluster
>>> ctx.save("ctx_2024.pkl")
>>> ctx = f1pit.TrackContext.load("ctx_2024.pkl")
>>> ctx.sector_profiles("Sakhir")               # {1: {...}, 2: {...}, 3: {...}}
"""

import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .tracks import get_track_by_name, sector_bounds


def _norm(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return "".join(ch for ch in str(value).lower() if ch.isalnum())


@dataclass
class TrackContext:
    circuits: pd.DataFrame
    corners: pd.DataFrame                 # ต้องมีคอลัมน์ cluster_col
    cluster_col: str = "KModes_Cluster"
    n_clusters: int = None

    def __post_init__(self):
        if self.n_clusters is None:
            labels = pd.to_numeric(self.corners.get(self.cluster_col), errors="coerce").dropna()
            self.n_clusters = int(labels.max()) + 1 if len(labels) else 0

    # ---------------- persistence ----------------
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Saved TrackContext -> {path}")

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)

    # ---------------- lookup ----------------
    @property
    def profile_columns(self):
        return ["Circuit_Total_Length", "Total_Sector_Length", "Total_Straight_Length",
                *[f"Num_Corner_{i}" for i in range(self.n_clusters)]]

    def corners_for(self, track):
        """โค้งของสนามนี้ (match ด้วย Track_Slug ก่อน แล้ว fallback เป็นชื่อ)"""
        c = self.corners
        if "Track_Slug" in c.columns and track.get("slug"):
            rows = c[c["Track_Slug"] == track["slug"]]
            if not rows.empty:
                return rows

        names = {_norm(n) for n in [track.get("name"), track.get("slug"), *track.get("aka", [])]} - {""}
        norm_col = c["Circuit_Name"].map(_norm)
        rows = c[norm_col.isin(names)]
        if rows.empty:  # ชื่อย่อ vs ชื่อเต็ม: ยอมรับถ้า match ได้สนามเดียว
            partial = norm_col.map(lambda v: bool(v) and any(n in v or v in n for n in names))
            if c.loc[partial, "Circuit_Name"].nunique() == 1:
                rows = c[partial]
        return rows

    def track_profile(self, corners, total_length, sector_start, sector_end):
        """feature ของ 1 sector: ความยาว sector, ความยาวทางตรง, จำนวนโค้งแต่ละ cluster

        นับเฉพาะโค้งที่อยู่ใน sector ทั้งโค้ง (Corner_Start >= start และ Corner_End <= end)
        """
        inside = corners[(corners["Corner_Start"] >= sector_start) & (corners["Corner_End"] <= sector_end)]
        sector_len = (sector_end - sector_start) * total_length
        profile = {
            "Circuit_Total_Length": total_length,
            "Total_Sector_Length": sector_len,
            "Total_Straight_Length": sector_len - inside["Corner_Length"].sum(),
        }
        counts = inside[self.cluster_col].value_counts()
        for i in range(self.n_clusters):
            profile[f"Num_Corner_{i}"] = int(counts.get(i, 0))
        return profile

    def sector_profiles(self, circuit_short_name):
        """{sector: profile dict} — profile ว่าง {} ถ้าหาไม่เจอ;
        ถ้ามี timing_sectors แต่ไม่มีข้อมูลโค้ง -> มีแค่ Total_Sector_Length"""
        out = {1: {}, 2: {}, 3: {}}
        track = get_track_by_name(circuit_short_name)
        bounds = sector_bounds(track)
        if bounds is None or pd.isna(pd.to_numeric(bounds["length_m"], errors="coerce")):
            return out, False

        total = float(bounds["length_m"])
        corners = self.corners_for(track)
        complete = not corners.empty
        for sec, (start, end) in enumerate(bounds["sectors"], start=1):
            if complete:
                out[sec] = self.track_profile(corners, total, start, end)
            else:
                out[sec] = {"Total_Sector_Length": float((end - start) * total)}
        return out, complete


def build_track_context(year, verbose=True, plot=False, cluster_col="KModes_Cluster"):
    """create_circuit_db(year) + cluster_corners() -> TrackContext"""
    from .circuits import create_circuit_db
    from .clustering import cluster_corners

    circuits, corners = create_circuit_db(year, verbose=verbose)
    clustered = cluster_corners(corners, verbose=verbose, plot=plot)
    return TrackContext(circuits=circuits, corners=clustered.data, cluster_col=cluster_col)


def add_track_profile(table, ctx, circuit_short_name, require=False, verbose=True, label=""):
    """เติมคอลัมน์ track profile ให้ทุกแถวตาม Sector; require=True และหาไม่ครบ -> None"""
    profiles, complete = ctx.sector_profiles(circuit_short_name)
    if not complete:
        verbose and print(f"Track profile MISSING: {label} | short_name={circuit_short_name}")
        if require:
            return None

    table = table.copy()
    for col in ctx.profile_columns:
        table[col] = table["Sector"].map({s: profiles[s].get(col, np.nan) for s in (1, 2, 3)})
    return table
