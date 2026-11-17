"""Pre-warm the OpenF1 disk cache concurrently so the rebuild becomes CPU-bound.

ปัญหา: rebuild เดิม fetch ทีละ request (sequential) บน API ที่ช้า (~6s/req) -> ~9-11 ชม.
แก้: fetch ทุก endpoint ที่ build_dataset ต้องการ พร้อมกัน N threads ลง cache เดิม
     (key/format เดียวกับ f1pit.api.fetch) -> จากนั้น _rebuild_outcomes.py จะ cache-hit ทั้งหมด

Endpoints ที่ต้อง pre-warm (ต่อ session x driver):
  per-session  : sessions(session_key), race_control, weather, intervals
  per-driver   : laps, stints, car_data, pit

Run:  python _prewarm_cache.py
(รัน background ได้; resume ได้เอง — key ที่ cache แล้วจะถูกข้าม)
"""

import pickle
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

import f1pit as f
from f1pit.api import _cache_key, _disk_path

f.config.TRACKS_JSON = "tracks.json"

YEARS = (2023, 2024, 2025)
N_WORKERS = 8

_tls = threading.local()
_lock = threading.Lock()
_stats = {"done": 0, "hit": 0, "fetched": 0, "empty": 0, "fail": 0}


def _session():
    if not hasattr(_tls, "s"):
        _tls.s = requests.Session()
    return _tls.s


def prefetch(endpoint, params):
    key = _cache_key(endpoint, params)
    path = _disk_path(key)
    if path is not None and path.exists():
        with _lock:
            _stats["hit"] += 1
        return "hit"

    s = _session()
    url = f"{f.config.OPENF1_URL}/{endpoint}"
    for attempt in range(6):
        wait = min(60.0, 1.5 * 2 ** attempt)
        try:
            r = s.get(url, params=params, timeout=f.config.REQUEST_TIMEOUT)
        except requests.RequestException:
            time.sleep(wait)
            continue
        if r.status_code == 404:
            with _lock:
                _stats["empty"] += 1
            return "404"
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(wait)
            continue
        try:
            r.raise_for_status()
        except requests.HTTPError:
            time.sleep(wait)
            continue
        data = r.json()
        df = pd.DataFrame([data] if isinstance(data, dict) else data)
        if not df.empty and path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as fh:
                pickle.dump(df, fh)
        with _lock:
            _stats["fetched"] += 1
        return "ok"
    with _lock:
        _stats["fail"] += 1
    return "fail"


def build_tasks():
    tasks = []
    for year in YEARS:
        sessions = f.list_sessions(year)
        print(f"{year}: {len(sessions)} sessions", flush=True)
        for s in sessions.itertuples(index=False):
            sk = int(s.session_key)
            tasks.append(("sessions", {"session_key": sk}))
            tasks.append(("race_control", {"session_key": sk}))
            tasks.append(("weather", {"session_key": sk}))
            tasks.append(("intervals", {"session_key": sk}))
            drivers = f.list_drivers(sk)
            for d in drivers:
                tasks.append(("laps", {"session_key": sk, "driver_number": d}))
                tasks.append(("stints", {"session_key": sk, "driver_number": d}))
                tasks.append(("car_data", {"session_key": sk, "driver_number": d}))
                tasks.append(("pit", {"session_key": sk, "driver_number": d}))
    # dedupe ตาม key (กัน session ซ้ำ)
    seen, uniq = set(), []
    for endpoint, params in tasks:
        k = _cache_key(endpoint, params)
        if k not in seen:
            seen.add(k)
            uniq.append((endpoint, params))
    return uniq


def main():
    t0 = time.time()
    tasks = build_tasks()
    total = len(tasks)
    print(f"total unique fetches = {total}", flush=True)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(prefetch, e, p): (e, p) for e, p in tasks}
        for fut in as_completed(futs):
            with _lock:
                _stats["done"] += 1
                done = _stats["done"]
            if done % 200 == 0 or done == total:
                s = dict(_stats)
                el = time.time() - t0
                print(f"  {done}/{total}  hit={s['hit']} fetched={s['fetched']} "
                      f"empty={s['empty']} fail={s['fail']}  ({el:.0f}s)", flush=True)

    el = time.time() - t0
    print(f"\nDONE in {el:.0f}s: {_stats}", flush=True)
    print("Cache is warm. Now run:  python _rebuild_outcomes.py", flush=True)


if __name__ == "__main__":
    main()
