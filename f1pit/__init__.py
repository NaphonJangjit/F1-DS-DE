"""f1pit — F1 sector-duration dataset builder + models (OpenF1)

Quick start (python interactive)
--------------------------------
>>> import f1pit as f
>>> f.config.TRACKS_JSON = "tracks.json"
>>> ctx = f.build_track_context(2024)           # สนาม + โค้ง + clustering
>>> ctx.save("ctx_2024.pkl")                     # ครั้งหน้า: ctx = f.TrackContext.load("ctx_2024.pkl")
>>> df  = f.build_sector_table(1, "Bahrain", 2024, "Race", ctx=ctx)
>>> data = f.build_dataset(2024, ctx=ctx, save_path="data_2024.parquet")
>>> out = f.train_models(data, open_report=True)
>>> out.results, out.best_name, out.seconds_metrics()

แก้โค้ดแล้วอยากโหลดใหม่โดยไม่ต้องออกจาก python:
>>> f.reload()
"""

from . import config
from .api import clear_cache, fetch, to_utc
from .circuits import create_circuit_db
from .clustering import ClusterResult, cluster_corners
from .context import TrackContext, add_track_profile, build_track_context
from .dataset import build_dataset, list_drivers, list_sessions, load_df, save_df
from .features import build_sector_table
from .tracks import get_best_race_lap_track, get_track_by_name, load_tracks
from .train import TrainResult, prepare_data, train_models

__all__ = [
    "config", "fetch", "clear_cache", "to_utc",
    "load_tracks", "get_track_by_name", "get_best_race_lap_track",
    "create_circuit_db", "cluster_corners", "ClusterResult",
    "TrackContext", "build_track_context", "add_track_profile",
    "build_sector_table", "build_dataset", "list_sessions", "list_drivers", "save_df", "load_df",
    "prepare_data", "train_models", "TrainResult", "write_report", "reload",
]


def write_report(out, df_ml, path="model_report.html", open_browser=False):
    from .report import write_report as _write
    return _write(out, df_ml, path, open_browser)


def reload():
    """reload ทุก submodule (หลังแก้โค้ด) — ใช้ต่อด้วย `f = f1pit.reload()` หรือแค่ `f.reload()`

    (config ไม่ถูก reload เพื่อคงค่าที่ตั้งไว้)
    หมายเหตุ: object ที่สร้างไว้ก่อน reload (เช่น ctx) ยังใช้ class เดิมอยู่
    ถ้าแก้ class เหล่านั้นให้สร้างใหม่ / load ใหม่
    """
    import importlib
    import sys

    order = ["api", "tracks", "circuits", "clustering", "context",
             "features", "dataset", "train", "report", "plotting"]
    for name in order:
        mod = sys.modules.get(f"{__name__}.{name}")
        if mod is not None:
            importlib.reload(mod)
    pkg = importlib.reload(sys.modules[__name__])
    print("f1pit reloaded")
    return pkg
