# f1pit

โค้ดจาก Colab ที่จัดใหม่เป็น package สำหรับเรียกจาก `python` interactive (REPL)
ไม่มี global variable ข้ามไฟล์แล้ว ทุกอย่างส่งผ่าน argument หรือ `TrackContext`

## ติดตั้ง

```bash
pip install -r requirements.txt
# optional: pip install lightgbm xgboost catboost pyarrow
```

วาง `tracks.json` ไว้ในโฟลเดอร์ที่รัน `python` (หรือตั้ง `f.config.TRACKS_JSON`)

## ใช้งานใน python interactive

```python
>>> import f1pit as f
>>> f.config.TRACKS_JSON = "tracks.json"

# 1) สนาม + โค้ง + clustering  (ช้า: ทำครั้งเดียวแล้ว save)
>>> ctx = f.build_track_context(2024)
>>> ctx.save("ctx_2024.pkl")
>>> ctx = f.TrackContext.load("ctx_2024.pkl")      # ครั้งต่อไป

# 2) ตาราง sector ของนักแข่ง 1 คน
>>> df = f.build_sector_table(1, "Bahrain", 2024, "Race", ctx=ctx)
>>> f.list_sessions(2024)                           # ดู session_key
>>> df = f.build_sector_table(1, session_key=9472, ctx=ctx)

# 3) dataset ทั้งฤดูกาล (ทุก session x ทุกนักแข่ง)
>>> data = f.build_dataset(2024, ctx=ctx, save_path="data_2024.parquet")
>>> data = f.load_df("data_2024.parquet")

# 4) train + report
>>> out = f.train_models(data, open_report=True)
>>> out = f.train_models(data, algorithms=["LightGBM", "ExtraTrees"])   # เลือกบางโมเดล ทดลองเร็ว
>>> out.results
>>> out.best_name, out.seconds_metrics()
>>> out.predict(df)                                 # ทำนาย Duration (วินาที)

# แก้โค้ดแล้วโหลดใหม่ไม่ต้องออกจาก python
>>> f.reload()
```

### ทำทีละขั้นเอง (แทน `build_track_context`)

```python
>>> circuits, corners = f.create_circuit_db(2024)
>>> res = f.cluster_corners(corners)
>>> res.summary(); res.plot()                       # หรือ res.plot("k.png")
>>> ctx = f.TrackContext(circuits, res.data)
```

### Cache

ผลจาก OpenF1 ถูก cache ไว้ที่ `.cache/openf1/` ทำให้รันซ้ำ / เปิด python ใหม่ไม่ต้องโหลดใหม่

```python
>>> f.config.CACHE_DIR = None    # ปิด cache
>>> f.clear_cache()              # ล้าง cache
```

## โครงสร้าง

| ไฟล์ | หน้าที่ | เดิมอยู่ที่ |
|---|---|---|
| `config.py` | path, URL, header, seed | กระจายหลายที่ |
| `api.py` | `fetch()` + retry/backoff + cache, `to_utc()` | `utils.py`, `preprocessing/utils.py` |
| `tracks.py` | tracks.json lookup, track path, มุมโค้ง | `preprocessing/tracks.py`, `preprocessing/utils.py` |
| `circuits.py` | `create_circuit_db()` | `preprocessing/tracks.py` |
| `clustering.py` | `cluster_corners()` (K-Means + K-Modes) | `ml/corner_clustering.py` (script) |
| `context.py` | `TrackContext`, track profile features | global `corners_clustered_df`, `KMODE_CLUSTERS`, `circuits` |
| `features.py` | `build_sector_table()` | `create_a_fucking_table` + `create_fucking_table` |
| `dataset.py` | `build_dataset()`, `save_df/load_df` | (ใหม่ — แทน loop ที่สร้าง `dfngongo`) |
| `train.py` | `train_models()`, `TrainResult` | `ml/train.py` |
| `report.py` | HTML report | `ml/train.py` |
| `plotting.py` | dark theme, `fmt_lap`, `cap_axis` | `utils.py` |

## ชื่อเดิม → ชื่อใหม่

| เดิม | ใหม่ |
|---|---|
| `create_fucking_table(d, c, y, s)` | `build_sector_table(d, c, y, s, ctx=ctx)` |
| `create_a_fucking_table(...)` | `build_sector_table(...)` ไม่ส่ง `ctx` |
| `create_track_profile(...)` | `ctx.track_profile(...)` / `ctx.sector_profiles(name)` |
| `corners_clustered_df` | `ctx.corners` |
| `KMODE_CLUSTERS` | `ctx.n_clusters` |
| `build_report(dfngongo)` | `train_models(data)` |
| `out["fitted_models"][name]` | `out.models[name]` / `out.best_model` |

## สิ่งที่แก้ไปด้วย

- **pandas 3:** `create_a_fucking_table` เดิม crash (`Cannot losslessly convert units`) เพราะ resolution ของ datetime ไม่ตรงกัน ตอนนี้ `to_utc()` แปลงเป็น `ns` เสมอ
- **`create_circuit_db`:** เดิมถ้าหาสนามใดใน tracks.json ไม่เจอจะ `return [], []` ทิ้งข้อมูลทั้งปี ตอนนี้ข้ามเฉพาะสนามนั้นพร้อมแจ้งเตือน ถ้าโค้งไม่มีใน `corner_ranges` ก็ข้ามแทน KeyError
- **ประเทศที่มีหลายสนาม** (USA, Italy ฯลฯ): เดิมเลือก session แรกเงียบ ๆ ตอนนี้มี warning และส่ง `session_key=` ได้ ส่วน `build_dataset` ใช้ session_key อยู่แล้ว
- **match สนามกับโค้ง:** เพิ่มคอลัมน์ `Track_Slug` ใน circuits/corners ทำให้ไม่ต้องเดาชื่อแบบ fuzzy (ยังมี fallback สำหรับข้อมูลเก่า)
- **rate limit:** retry แบบ exponential backoff ใน `fetch` ตัวเดียว แทน `except: sleep(60)`
- เพิ่มคอลัมน์ `Session_Key` ใน META และลบ argument `interval_tolerance` ที่ไม่ได้ใช้
- import package ไม่เปลี่ยน matplotlib theme เอง แล้ว (เรียก `f.plotting.apply_theme()` ถ้าต้องการ)

ตรวจแล้วว่า output ของ `build_sector_table` และ track profile ตรงกับโค้ดเดิมทุกค่า (ทดสอบด้วยข้อมูลจำลอง)
