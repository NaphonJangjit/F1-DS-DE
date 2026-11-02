"""ค่าตั้งค่ากลางของ package — แก้ได้ตอน runtime เช่น

>>> import f1pit
>>> f1pit.config.TRACKS_JSON = "data/tracks.json"
>>> f1pit.config.CACHE_DIR = None          # ปิด disk cache

ทุกโมดูลอ่านค่าจาก `config.X` ตอนเรียกใช้ ไม่ได้ copy ค่าไว้ตอน import
"""

import os
from pathlib import Path

# ไฟล์ข้อมูลสนาม (layouts / timing_sectors / corner_ranges)
TRACKS_JSON = Path(os.environ.get("F1PIT_TRACKS_JSON", "tracks.json"))

# OpenF1
OPENF1_URL = "https://api.openf1.org/v1"
REQUEST_DELAY = 0.4          # วินาที หลังทุก request (กัน rate limit)
REQUEST_TIMEOUT = 90

# Disk cache ของผล fetch (None = ปิด)
CACHE_DIR = Path(os.environ.get("F1PIT_CACHE", ".cache/openf1"))

# MultiViewer circuit-info API ต้องการ browser-like headers
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

RANDOM_STATE = 42
