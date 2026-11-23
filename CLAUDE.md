# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`f1pit` is a Python package that builds an F1 lap-sector dataset from the [OpenF1](https://api.openf1.org/v1) API and trains regression models to predict sector duration (`Duration`, in seconds). It was refactored out of a Colab notebook into an importable package meant to be used interactively in a Python REPL.

The README and all docstrings/comments are in Thai. The code itself (identifiers, config, columns) is in English.

**There is no build system, linter, test suite, or CLI entry point.** There is no `pyproject.toml`/`setup.py`, and the directory is not a git repository. "Running" the project means using it from `python` interactively.

## Install

```bash
pip install -r requirements.txt
# optional models / parquet support:
pip install lightgbm xgboost catboost pyarrow
```

Place `tracks.json` in the directory where you run `python`, or set `f1pit.config.TRACKS_JSON` (or the `F1PIT_TRACKS_JSON` env var) to its path.

## The pipeline (read these in order to understand data flow)

1. **`api.py` → `tracks.py` → `circuits.py`** — fetch OpenF1 `meetings`/`sessions`/`laps`/`location` per meeting, look up each circuit in `tracks.json`, reconstruct the track path from fastest-race-lap telemetry, and compute per-corner geometry (`Angle`, `Corner_Length`, XY positions).
2. **`clustering.py`** — cluster corners with K-Means (continuous features) and K-Modes (categorical features), choosing `k` via elbow + silhouette.
3. **`context.py`** — `TrackContext` bundles circuits + clustered corners into one object; it produces per-sector track-profile features (`Total_Sector_Length`, `Total_Straight_Length`, `Num_Corner_<cluster>`).
4. **`features.py`** — `build_sector_table()` builds the per-driver lap-sector table. This is the core, most complex module.
5. **`dataset.py`** — `build_dataset()` loops `build_sector_table` over every session × driver and concatenates.
6. **`train.py`** — `train_models()` fits many regressors on `Log_Duration` (= `ln(Duration)`) and returns a `TrainResult`.
7. **`report.py`** — writes a self-contained HTML report (charts embedded as base64).

Typical REPL flow:

```python
import f1pit as f
ctx = f.build_track_context(2024)      # slow; cache it
ctx.save("ctx_2024.pkl")               # then ctx = f.TrackContext.load(...) next time
df   = f.build_sector_table(1, "Bahrain", 2024, "Race", ctx=ctx)
data = f.build_dataset(2024, ctx=ctx, save_path="data_2024.parquet")
out  = f.train_models(data, open_report=True)
out.best_name, out.seconds_metrics()
```

## Design invariants (important to preserve)

These are the non-obvious rules that shape the code and are easy to break:

- **Point-in-time correctness (`features.py`).** Every feature must be knowable *as of* the sector's `Time_Start` — no information from within the current sector or the future may leak in. `Prev_*` features come only from earlier laps; as-of joins (`pd.merge_asof`, `direction="backward"`) are used for weather/interval; `Track_Status` is derived only from race-control events that occurred before `Time_Start`. `Outcome_*` columns describe the current sector and are explicitly excluded from features (only exported with `include_outcome_columns=True`).
- **No cross-file mutable global state.** The Colab originals used module-level globals (`corners_clustered_df`, `KMODE_CLUSTERS`, `circuits`). These are now consolidated into `TrackContext` and passed explicitly as arguments. Don't reintroduce shared globals.
- **Config is read at call time, not import time.** Modules read `config.X` when used, so `f1pit.config.TRACKS_JSON = ...` or `f1pit.config.CACHE_DIR = None` at runtime takes effect. `reload()` deliberately does not reload `config` so runtime settings persist.
- **`fetch()` is the only OpenF1 entry point** (`api.py`): single shared `requests.Session`, exponential backoff on 429/5xx, and a two-level cache (memory + `.cache/openf1/*.pkl`). 404 returns an empty DataFrame. Use the `fetch_fn=` parameter (already threaded through the builders) for testing rather than bypassing `fetch`.

## Gotchas

- **pandas 3 datetime resolution.** Mixing datetimes with differing resolutions crashes `pd.to_datetime` ("Cannot losslessly convert units"). `to_utc()` in `api.py` always coerces to `datetime64[ns, UTC]` — route all datetime parsing through it.
- **Countries with multiple circuits** (USA, Italy, etc.): name-based session lookup may silently pick the wrong one. `resolve_session()` warns and returns the first; callers should pass `session_key=` to disambiguate.
- **Track↔corner matching** uses the `Track_Slug` column (added during the refactor) first, with name/`aka` fallback for legacy data — not fuzzy matching.
- **`create_circuit_db()` skips incomplete meetings** with a warning instead of aborting the whole year (a past bug returned `[], []` and dropped all data).
- **`Num_Corner_*` is not comparable across years.** Corner clusters are fitted per year (9/11/8 clusters for 2023/2024/2025), so `Num_Corner_k` means a different corner type each year — don't use it in models trained on one year and tested on another.
- **Red flags and Track_Status.** `SC/VSC DEPLOYED → RED FLAG → standing restart` sends no SC-ending message; the parser treats `RED FLAG` as a reset. Parquet built before that fix has SC stuck to the end of the race (2023 Brazil/Mexico…); repair with `_repair_status.py` / `repair_track_status()`.
- **`ClusterResult`/`TrackContext` pickles are class-bound**: after changing a class and calling `f.reload()`, previously-created objects still use the old class. Rebuild or reload them.

## Interactive development

`f1pit.reload()` re-imports all submodules (in dependency order) so you can edit code without exiting Python. Call it as `f = f1pit.reload()` or `f.reload()`. It skips `config` to preserve runtime settings.
