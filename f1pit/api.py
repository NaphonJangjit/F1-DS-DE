"""OpenF1 client: retry + rate-limit backoff + cache (memory และ disk)

>>> from f1pit import fetch
>>> laps = fetch("laps", session_key=9158, driver_number=1)
>>> fetch.clear_cache()        # หรือ f1pit.api.clear_cache()
"""

import hashlib
import json
import pickle
import time

import pandas as pd
import requests

from . import config

_http = requests.Session()
_memory_cache: dict = {}


def to_utc(series):
    """แปลงเป็น datetime UTC (รองรับ ISO ที่มี/ไม่มี fractional seconds); แปลงไม่ได้ -> NaT"""
    series = pd.Series(series)
    try:
        out = pd.to_datetime(series, utc=True, errors="coerce", format="ISO8601")
    except (ValueError, TypeError):
        out = pd.to_datetime(series, utc=True, errors="coerce")
    return out.astype("datetime64[ns, UTC]")


# ---------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------
def _cache_key(endpoint, params):
    raw = json.dumps([endpoint, sorted(params.items())], default=str)
    return f"{endpoint}_{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


def _disk_path(key):
    if config.CACHE_DIR is None:
        return None
    return config.CACHE_DIR / f"{key}.pkl"


def _cache_get(key):
    if key in _memory_cache:
        return _memory_cache[key]
    path = _disk_path(key)
    if path is not None and path.exists():
        try:
            with open(path, "rb") as f:
                df = pickle.load(f)
            _memory_cache[key] = df
            return df
        except Exception:
            path.unlink(missing_ok=True)   # ไฟล์เสีย -> โหลดใหม่
    return None


def _cache_put(key, df):
    _memory_cache[key] = df
    path = _disk_path(key)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(df, f)


def clear_cache(disk=True):
    """ล้าง cache ใน memory (และบน disk ถ้า disk=True)"""
    _memory_cache.clear()
    if disk and config.CACHE_DIR is not None and config.CACHE_DIR.exists():
        for p in config.CACHE_DIR.glob("*.pkl"):
            p.unlink()


# ---------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------
def fetch(endpoint, *, retries=6, use_cache=True, **params):
    """GET {OPENF1_URL}/{endpoint}?params -> DataFrame

    - 404 / ไม่มีข้อมูล -> DataFrame ว่าง (ไม่ cache)
    - 429 / 5xx / network error -> รอแบบ exponential backoff แล้วลองใหม่
    """
    key = _cache_key(endpoint, params)
    if use_cache:
        hit = _cache_get(key)
        if hit is not None:
            return hit.copy()

    url = f"{config.OPENF1_URL}/{endpoint}"
    last_error = None
    for attempt in range(retries):
        wait = min(60.0, 1.5 * 2 ** attempt)
        try:
            r = _http.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(wait)
            continue

        if r.status_code == 404:
            return pd.DataFrame()
        if r.status_code == 429 or r.status_code >= 500:
            last_error = f"HTTP {r.status_code}"
            time.sleep(wait)
            continue
        r.raise_for_status()

        data = r.json()
        df = pd.DataFrame([data] if isinstance(data, dict) else data)
        time.sleep(config.REQUEST_DELAY)
        if use_cache and not df.empty:
            _cache_put(key, df)
        return df.copy()

    raise RuntimeError(f"โหลดไม่สำเร็จหลังลอง {retries} ครั้ง: {url} {params} ({last_error})")


fetch.clear_cache = clear_cache
