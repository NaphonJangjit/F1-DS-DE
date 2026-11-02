"""รวม sector table หลาย session / หลายนักแข่ง เป็น dataset เดียว + save/load

>>> data = f1pit.build_dataset(2024, ctx=ctx, countries=["Bahrain", "Japan"])
>>> f1pit.save_df(data, "data_2024.parquet")
>>> data = f1pit.load_df("data_2024.parquet")
"""

from pathlib import Path

import pandas as pd

from .api import fetch
from .features import build_sector_table


def list_sessions(year, session_name="Race", countries=None, fetch_fn=fetch):
    """ตาราง session ของปีนั้น (กรองประเทศได้)"""
    sessions = fetch_fn("sessions", year=year, session_name=session_name)
    if sessions.empty:
        return sessions
    if countries:
        wanted = {c.lower() for c in countries}
        sessions = sessions[sessions["country_name"].str.lower().isin(wanted)]
    cols = [c for c in ["session_key", "meeting_key", "country_name", "circuit_short_name",
                        "session_name", "date_start"] if c in sessions.columns]
    return sessions.sort_values("date_start")[cols].reset_index(drop=True)


def list_drivers(session_key, fetch_fn=fetch):
    drivers = fetch_fn("drivers", session_key=int(session_key))
    if drivers.empty:
        return []
    return sorted(pd.to_numeric(drivers["driver_number"], errors="coerce").dropna().astype(int).unique())


def build_dataset(year, session_name="Race", *, countries=None, drivers=None, ctx=None,
                  fetch_fn=fetch, save_path=None, verbose=False, **table_kwargs):
    """build_sector_table() ทุก session × ทุกนักแข่ง แล้ว concat

    drivers   : list ของ driver_number (None = ทุกคนใน session)
    save_path : ถ้าระบุ จะ save ผลลัพธ์ (.parquet / .csv / .pkl)
    table_kwargs : ส่งต่อให้ build_sector_table (เช่น require_track_profile=True)
    """
    sessions = list_sessions(year, session_name, countries, fetch_fn)
    if sessions.empty:
        print(f"No sessions for {year} {session_name}")
        return pd.DataFrame()

    tables, failures = [], []
    for s in sessions.itertuples(index=False):
        driver_list = drivers or list_drivers(s.session_key, fetch_fn)
        print(f"[{s.country_name} / {s.circuit_short_name}] session {s.session_key}: {len(driver_list)} drivers")
        for d in driver_list:
            try:
                t = build_sector_table(d, session_key=s.session_key, ctx=ctx, fetch_fn=fetch_fn,
                                       verbose=verbose, **table_kwargs)
            except Exception as exc:          # session เสียไม่ควรทำให้ทั้งปีพัง
                failures.append((s.session_key, d, repr(exc)))
                print(f"  ! driver {d}: {exc}")
                continue
            if t is not None and not t.empty:
                tables.append(t)

    data = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    print(f"Done: {len(data):,} rows from {len(tables)} driver-sessions ({len(failures)} failed)")
    data.attrs["failures"] = failures
    if save_path and not data.empty:
        save_df(data, save_path)
    return data


def save_df(df, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        df.to_pickle(path)
    print(f"Saved {len(df):,} rows -> {path}")


def load_df(path):
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, parse_dates=["Time_Start"])
    return pd.read_pickle(path)
