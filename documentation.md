# Developer Documentation — f1pit + forest-of-forests

This project builds an F1 lap-sector dataset from the [OpenF1](https://api.openf1.org/v1) API and
trains a **forest-of-forests** to predict sector `Duration` (lap time). The modelling pipeline is a
two-stage design:

1. **Stage 1 — 8 sub-forests.** Each predicts a *current* telemetry channel (corner/straight speed,
   throttle, brake, DRS, max/min speed, gear) from its *previous* value + tyre + geometry + weather +
   traffic. Outputs are the tyre-aware `Pred_<channel>` estimates.
2. **Stage 2 — main forest.** Predicts the **delta** `Log_Duration − log(Prev_Duration)` from
   `Pred_*` (or raw `Prev_*`) + pace anchors + weather + traffic + distance + corner profile. The
   result is reconstructed as `Duration = Prev_Duration × exp(delta)`.

This document covers how to install, prepare data, train, predict, evaluate and run the auxiliary
analyses.

---

## 1. Prerequisites

```bash
pip install -r requirements.txt
# optional tree/gradient-boosting models + parquet:
pip install xgboost catboost pyarrow
```

`scipy` is also required by the analysis scripts (`ttest`, `skew`/`kurtosis`).

Place `tracks.json` in the project root (where you run `python`), or point at it with
`f1pit.config.TRACKS_JSON` (or env `F1PIT_TRACKS_JSON`).

## 2. Data files (inputs / outputs)

| path | what it is |
|---|---|
| `tracks.json` | circuit layouts / timing sectors / corner ranges (static geometry) |
| `ctx_2023/2024/2025.pkl` | `TrackContext` — circuits + clustered corners per year |
| `data_2023/2024/2025.parquet` | built lap-sector table (features only, no current telemetry) |
| `data_2023/2024/2025_out.parquet` | same + `Outcome_*` current telemetry (used by the forest) |
| `.cache/openf1/*.pkl` | OpenF1 raw API cache (the `fetch()` disk cache) |
| `model/v2/` | saved trained models (`sub_*.pkl`, `main.pkl`, `meta.json`) |
| `fforest_results.json` | full results (MAE, importance, breakdown, degradation) |
| `fif_report_v2.md` | generated report |

## 3. Data preparation (one-time, ~hours)

The forest needs the **current telemetry** (`Outcome_*`) columns, which the default parquet does not
export. Rebuild with them:

```bash
python _rebuild_outcomes.py
```

This re-runs `build_dataset(..., include_outcome_columns=True)` for 2023/2024/2025 using the existing
`ctx_*.pkl` and the `.cache/openf1` disk cache, and writes `data_<year>_out.parquet`.

**Why it can be slow:** `build_dataset` fetches `laps`/`stints`/`car_data`/`race_control`/`pit`/
`weather`/`intervals` once per driver × session (~5,800 requests). If the disk cache is cold, the
OpenF1 API (~5–6 s/request sequential) dominates. Pre-warm it concurrently first:

```bash
python _prewarm_cache.py
```

This fetches every endpoint the rebuild needs, in parallel (resumable — already-cached keys are
skipped), so the rebuild then becomes cache-hit / CPU-bound.

## 4. Training the forest-of-forests

```bash
python _fforest.py
```

What it does:

1. Loads `data_<year>_out.parquet`, runs `_prepare_lab` (log-target, outlier removal, `_roll5_`
   trend features, tyre interaction) + `_engineer` (`Straight_Ratio`, log lengths).
2. **Stage 1** — for each of 8 telemetry channels, tries variants `level` / `delta` / `delta_slog`
   (+ `log` for the speed channels), with a MAD-based clip on the train target, and picks the variant
   with lowest held-out-2024 MAE. Saves each sub-forest as `Pred_<channel>` (only if it beats the
   persistence baseline "predict current = previous").
3. **Stage 2** — main forest on the **delta** target, for two feature variants (`prev` = raw `Prev_*`
   telemetry, `pred` = `Pred_*`), each trained with RF / SGD / KNN / XGBoost / CatBoost. Reports MAE
   (seconds) + the 6-condition breakdown + degradation curve.
4. Saves models to `model/v2/` and results to `fforest_results.json`.

**Leakage-safe split:** stage-1 is trained on **2023 only** and predicts 2024 + 2025; stage-2 is
trained on **2024** and tested on **2025**.

**Hyperparameters** (in `_fforest.py`): RF `n_estimators=100, max_features=0.5, min_samples_leaf=10`;
SGD `loss="huber", learning_rate="invscaling", eta0=0.01`; KNN `k=15, weights="distance"`;
XGBoost / CatBoost use the `train.py` configs (1000 rounds).

## 5. Predicting without retraining

```bash
python _predict.py
```

Loads `model/v2/`, runs stage-1 → stage-2 on `data_2025_out.parquet`, writes
`predictions_2025.parquet` (with `Pred_<channel>` and `Pred_Duration_s`), and prints MAE + the
6-condition breakdown. Or use it as a library:

```python
from _predict import load_models, predict
meta, subs, main = load_models()
out = predict(raw_df, meta, subs, main)   # raw_df: a data_*_out.parquet frame
```

## 6. Report

```bash
python _write_report.py
```

Reads `fforest_results.json` and writes `fif_report_v2.md` (stage-1 table, full per-forest feature
importance, degradation curve, stage-2 5-model comparison, 6-condition breakdown).

## 7. Evaluation details

**6-condition MAE breakdown** (on the 2025 test set):

| condition | definition |
|---|---|
| overall | all rows |
| clean_no_sc_vsc | `Track_Status == "GREEN"` |
| dry | `Rainfall == 0` |
| wet | `Rainfall > 0` |
| dry_clean | dry AND green |
| wet_clean | wet AND green |

**Degradation curve** — partial dependence of the delta (`Log_Duration − log Prev_Duration`) on
`Tyre_Age_Total`, holding everything else at its median. The slope is the degradation rate.

## 8. Hypothesis test (Monza 2026 example)

```bash
python _monza_ttest.py
```

Paired t-test (Welch) between two drivers over the same laps, plus Mann-Whitney U as a robust
check. It fetches only `sessions` + `laps` (no `stints`/`car_data`).

## 9. Distribution analysis (sub-forest diagnostics)

```bash
python _dist_analysis.py
```

Computes skew + excess kurtosis of the sub-forest **target**, **delta** (`current − prev`) and
**residual**, and writes `dist_analysis.csv` + `dist_target.png` / `dist_residual.png`.

---

## 10. Current feature engineering (main forest)

The main forest currently uses **41 raw features** (before one-hot):

| group | features |
|---|---|
| `Prev_*` telemetry (8) | Avg_Speed_Corners, Avg_Speed_Straight, Full_Throttle, Brake, DRS, Max_Speed, Min_Speed, Avg_Gear |
| duration pace anchors (7) | Prev_Duration, Prev_Sector_Duration, Clean_Pace, Best_Duration_So_Far, Prev_I1/I2/St_Speed |
| compound | `Tyre` (one-hot) |
| weather (5) | Track_Temperature, Air_Temperature, Rainfall, Pressure, Humidity |
| track status | `Track_Status` (one-hot) + `Track_Status_Inferred` |
| traffic (4) | Traffic_Within_1s/2s/5s, Interval_Ahead |
| geometry (2) | Total_Sector_Length, Total_Straight_Length |
| distance (2) | Total_Distance_Driven (fuel), Distance_On_Tyre (tyre distance) |
| corner profile (8) | Num_Corner_0 … Num_Corner_7 |
| duration trend (2) | Prev_Duration_roll5_mean, Prev_Duration_roll5_slope |

The **target** is `delta = Log_Duration − log(Prev_Duration)` (the lap-over-lap log ratio). `Duration`
is reconstructed as `Prev_Duration × exp(delta)`.

## 11. Design notes / gotchas

- **Point-in-time correctness** — every feature is knowable *as of* the sector's `Time_Start`; no
  within-sector or future info leaks in (see `f1pit/features.py`).
- **`fetch()` is the only OpenF1 entry point** — shared session, exponential backoff, two-level cache
  (memory + `.cache/openf1`). Route datetime parsing through `api.to_utc`.
- **Leakage-safe stacking** — stage-1 models never see the stage-2 test rows; stage-2 train rows get
  out-of-sample `Pred_*` (stage-1 was trained on an earlier year).
- **SGD diverges on `squared_error`** for these targets — use `huber` + `invscaling` (see
  `_fit_sgd`). The signed-log delta (`delta_slog`) must have its prediction clipped before `expm1`
  or a linear model explodes.
- **Heavy-tailed telemetry deltas** — the delta (`current − prev`) has kurtosis 20–58 (occasional
  huge telemetry spikes). The MAD clip + signed-log target reduce, but do not remove, this.
- **Wet is under-sampled** (~1.2% of rows) — wet MAE is ~3–4× the dry MAE; a single forest cannot
  learn wet physics from that sample.
- **Tyre signal is tiny** — after controlling for pace (`Clean_Pace`/`Best_Duration_So_Far`) and
  duration trend, every tyre feature (`Tyre_Age_*`, compound one-hot) sits near zero importance; the
  only meaningful tyre-ish term is `Distance_On_Tyre`.

## 12. Latest results (delta target, 41 features, 2025 test)

**Stage 2 — main forest (MAE seconds / R² on reconstructed Duration):**

| variant | RF | SGD | KNN | XGB | CatBoost |
|---|---|---|---|---|---|
| raw Prev_* | 0.2992 / 0.9865 | 0.5198 / 0.9300 | 0.4464 / 0.9654 | 0.3457 / 0.9873 | 0.3523 / 0.9887 |
| Pred_* | 0.3224 / 0.9849 | 0.5205 / 0.9307 | 0.4792 / 0.9578 | 0.4383 / 0.9810 | 0.3997 / 0.9862 |

Each cell = MAE (s) / R². **RF is best by MAE**; XGBoost/CatBoost edge RF on R² (slightly fewer
extreme errors). Raw `Prev_*` beats `Pred_*` across every model.

**6-condition MAE (raw Prev_*, RF):** overall 0.2992 · clean 0.2815 · dry 0.2890 · wet 1.1535 ·
dry+clean 0.2729 · wet+clean 1.0196.

**Stage 1 (sub-forests, held-out 2024):** all 8 beat persistence; RF R² 0.57–0.96 (best variants:
`delta_slog` for 7 channels, `delta` for Brake).

**Degradation curve:** currently skipped — `Tyre_Age_Total` is no longer a feature (replaced by
`Distance_On_Tyre`); re-point the partial-dependence in `_fforest.py` at `Distance_On_Tyre` to restore it.

## 13. File reference

| file | purpose |
|---|---|
| `f1pit/api.py` | OpenF1 client (`fetch`, cache, backoff, `to_utc`) |
| `f1pit/tracks.py`, `circuits.py` | circuit geometry + corner database |
| `f1pit/clustering.py`, `context.py` | corner clustering + `TrackContext` |
| `f1pit/features.py` | `build_sector_table` (the point-in-time feature builder) |
| `f1pit/dataset.py` | `build_dataset` / `save_df` / `load_df` |
| `f1pit/train.py` | `prepare_data`, `train_models`, `build_preprocessor`, tyre interaction, `_roll5_` trends |
| `f1pit/wind_rain_lab.py` | `_prepare_lab` (ML frame prep reused by the forest) |
| `_rebuild_outcomes.py` | rebuild parquet with `Outcome_*` columns |
| `_prewarm_cache.py` | concurrent OpenF1 cache pre-warm |
| `_fforest.py` | train the forest-of-forests (stage 1 + stage 2) + save models |
| `_predict.py` | load saved models + predict (no retrain) |
| `_write_report.py` | generate `fif_report_v2.md` |
| `_dist_analysis.py` | skew/kurtosis + histogram diagnostics |
| `_monza_ttest.py` | paired t-test between two drivers |
