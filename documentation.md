# Developer Documentation — f1pit + forest-of-forests

This project builds an F1 lap-sector dataset from the [OpenF1](https://api.openf1.org/v1) API and
trains a **forest-of-forests** to predict sector `Duration`. The modelling pipeline is a two-stage
design (v3):

1. **Stage 1 — 8 sub-forests.** Each predicts a *current* telemetry channel (corner/straight speed,
   throttle, brake, DRS, max/min speed, gear) from its *previous* value + tyre + year-invariant
   geometry + weather + traffic. Outputs are the tyre-aware `Pred_<channel>` estimates.
2. **Stage 2 — main forest.** Predicts the **delta vs the last representative lap**
   `Log_Duration − log(Rep_Prev_Duration)` from `Pred_*` + relative pace anchors + lap context
   (SC/VSC history, pit, rain history) + weather + traffic + distance. The result is reconstructed as
   `Duration = Rep_Prev_Duration × exp(delta)`. Raw `Prev_*` telemetry (and the `Prev_I1/I2/St`
   speed traps) is **not** a stage-2 feature: the goal is to explain lap time *through tyres*, and
   tyre features put directly into the main forest carried no signal, so telemetry enters only via
   the tyre-aware `Pred_*`.

This document covers how to install, prepare data, train, predict, evaluate and run the auxiliary
analyses.

---

## 1. Prerequisites

```bash
pip install -r requirements.txt
# optional tree/gradient-boosting models + parquet:
pip install xgboost catboost pyarrow
```

`scipy` is also required by the analysis scripts (`ttest`, `skew`/`kurtosis`). XGBoost / CatBoost are
optional: `_fforest.py` skips them when they are not installed (RF / SGD / KNN always run).

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
| `model/v3/` | saved trained models (`sub_*.pkl`, `main.pkl`, `meta.json`) |
| `fforest_results.json` | full results (MAE, importance, breakdown) |
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

### 3.1 Track_Status repair (parquet built before the red-flag fix)

Parquet built with the old race-control parser has **SC stuck until the end of the race** after
`SC/VSC DEPLOYED → RED FLAG → standing restart` (no "SAFETY CAR IN THIS LAP" message is ever sent).
2023: Brazil (9205), Mexico (9181), 7787, 9149 → 6,046 rows (8.5% of 2023) labelled SC/VSC while the
field was at full race pace; 2024: 9636 → 298 rows; 2025: none. The parser is fixed in
`f1pit/features.py` (a `RED FLAG` event resets the status), so a rebuild is clean. To repair existing
files without rebuilding (1 OpenF1 request per session, overwrites in place):

```bash
python _repair_status.py
```

`_fforest.py` prints a warning when a year has > 5% SC/VSC rows (a sign the repair was not applied).

## 4. Training the forest-of-forests

```bash
python _fforest.py
```

What it does:

1. Loads `data_<year>_out.parquet`, adds the **pace context** (`f1pit.pace_context.add_pace_context`,
   §10.2), then runs `_prepare_lab` (log-target, outlier removal, `_roll5_` trend features, tyre
   interaction) + `_engineer` (`Straight_Ratio`, log lengths).
2. **Stage 1** — for each of 8 telemetry channels, tries variants `level` / `delta` / `delta_slog`
   (+ `log` for the speed channels), with a MAD-based clip on the train target, fitting on 2023 and
   picking the variant with lowest held-out-2024 MAE. A channel is used only if it beats the
   persistence baseline ("predict current = previous"); otherwise it is dropped (never replaced by raw
   `Prev_*`). The winning variant is then **5-fold cross-fitted** (`GroupKFold` by session) on
   2023+2024 → out-of-fold `Pred_*` for every stage-2 training row, and a **full refit on 2023+2024**
   (saved as `sub_<channel>.pkl`) predicts 2025.
3. **Stage 2** — main forest on the **delta vs `Rep_Prev_Duration`** target, trained on **2023+2024**,
   tested on **2025**, with RF / SGD / KNN (+ XGBoost / CatBoost if installed). Reports MAE (seconds),
   R², the `Pred_*` importance share and the 6-condition breakdown.
4. Saves models to `model/v3/` (`main.pkl` = the RF model; `meta.json` holds the anchor, `rep_tol`,
   the channel list and the exact feature list) and results to `fforest_results.json`.

**Leakage-safe split:** stage-2 training rows only ever see out-of-fold `Pred_*` (the stage-1 model that
produced them never saw that session); the 2025 test season is never used for fitting or for any
selection inside `_fforest.py`.

**Hyperparameters** (in `_fforest.py`, unchanged since v2): RF `n_estimators=100, max_features=0.5,
min_samples_leaf=10`; SGD `loss="huber", learning_rate="invscaling", eta0=0.01`; KNN `k=15,
weights="distance"`; XGBoost / CatBoost use the `train.py` configs (1000 rounds).

**Runtime:** stage 1 now fits 3–4 variants + 5 folds + 1 refit per channel (~30–40 min on 6 cores).

## 5. Predicting without retraining

```bash
python _predict.py
```

Loads `model/v3/`, runs pace context → stage 1 → stage 2 on `data_2025_out.parquet`, writes
`predictions_2025.parquet` (with `Pred_<channel>` and `Pred_Duration_s`), and prints MAE + the
6-condition breakdown. Or use it as a library:

```python
from _predict import load_models, predict
meta, subs, main = load_models()
out = predict(raw_df, meta, subs, main)   # raw_df: a data_*_out.parquet frame (whole sessions)
```

`raw_df` must contain whole sessions: the pace context (`Rep_*` anchors, previous-lap SC, rain
history) is built from earlier laps of the same driver/sector.

## 6. Report

```bash
python _write_report.py
```

Reads `fforest_results.json` and writes `fif_report_v2.md` (stage-1 table, full per-forest feature
importance, stage-2 model comparison, `Pred_*` importance share, 6-condition breakdown).

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

**Degradation curve** — skipped in v3: the main forest has no `Tyre_Age_Total` feature (tyre enters
through stage-1 `Pred_*`), so a curve has to vary tyre age at the stage-1 input and propagate through
`Pred_*` into stage 2.

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

## 9a. Online learning (Adaptive Random Forest vs SGD vs kNN)

```bash
pip install river
python _online.py run            # ARF, SGD, kNN + frozen RF reference (~55 min; kNN ~37, ARF ~12, SGD <1)
python _online.py run SGD,RF     # any subset
python _online.py plot           # -> online_results.json + figures/online_results.png
```

Stage 2 retrained as a *stream* (river): warm-up on all of 2023, then 2024 → 2025 in calendar order,
each race cut into 10-lap chunks — predict the chunk, then learn it (prequential), so the model is
updated every 10 laps of the race that is running. Same v3 target (delta vs `Rep_Prev_Duration`) and
the same 42 stage-2 features; stage 1 is fitted on 2023 only (out-of-fold for the warm-up rows) so
nothing from 2024/2025 reaches the stream. Every online model is also scored as a frozen copy taken
after the warm-up, plus a batch RF trained once on 2023. Results: §12.4. Per-row predictions and the
data/stage-1 caches go to `online/` (git-ignored; delete `online/*.pkl` after changing the data).

## 9b. Web: fully incremental pipeline + drift lab (one model lineage)

The web page (`index.html` of the F1 web repo) uses **only online learners, in both stages**: no batch
model anywhere (no Random Forest in stage 1). Its *AI Predict* panel, *Race replay* and *ML Drift Lab*
section read the same files, so they always show the same models and numbers.

```bash
py -3.14 _build_year.py 2026                                   # OpenF1 -> ctx_2026.pkl + data_2026.parquet
set F1_ONLINE_YEARS=2023,2024,2025,2026
py -3.14 _online.py run SGD                                    # only to cache online/stream_2023_2026/frames.pkl
py -3.14 _online_s1.py all                                     # incremental stage 1 + promote + stage 2 (~4 min)
py -3.14 _drift_lab.py --web "D:\Works\F1 web"               # drift_lab.json + model/prod/
py -3.14 _web_api.py                                           # http://127.0.0.1:8000
```

**Engines (`_stream_fast.py`, one chunk interface `predict_many` / `learn_many`):** SGD = river
(Huber, inverse-scaling 0.01, scaled X / y); ARF = CapyMOA `AdaptiveRandomForestRegressor` (MOA, Java;
10 trees, sqrt features, ADWIN) with online target scaling; kNN = numpy (k = 15, distance-weighted,
last 2,000 rows, online scaler; identical to river's kNN, max difference 1e-15). `pip install capymoa`
needs a JDK; the JVM heap is capped at 1 GB per process (`CAPYMOA_JVM_ARGS`), since one JVM runs per
worker. All jobs run in parallel (24 stage-1 jobs on 11 processes, 3 stage-2 jobs).

- **Stage 1 (`_online_s1.py`):** 8 telemetry channels × SGD / ARF / kNN. Every learner starts empty at
  the first race of 2023 and streams to the end of 2026 in 10-lap chunks: predict the chunk, then learn its
  real telemetry (known once the sector is done). 2023 `Pred_*` come from this stream too. Target variant /
  6 × MAD label clip / features as v3; clip bounds, imputer medians and one-hot categories are set from
  2023 rows (data preparation, not model training).
- **Stage-1 champion / challenger:** same rules as the drift lab, race by race from 2023. v001 = SGD.
  Score = mean over channels of MAE / MAE of "same as last lap" on 2023 (1.0 = no better than repeating
  the previous lap). Trigger = PSI ≥ 0.25 or champion score ≥ 1.25 × its own average so far; cooldown 2
  races; promote if the score is ≥ 0.02 lower and the green-lap score is not worse by > 0.01. Stage 2
  receives the champion's `Pred_*` race by race (`online/fast/s1_promote/`).
- **Stage 2:** SGD / ARF / kNN online on the served `Pred_*` (warm-up 2023, stream 2024 → 2026), plus each
  model frozen after the warm-up (`online/fast/s2_incremental/`).
- **Filter for training and analysis only, never for inference:** the stream holds every sector row
  (`frames_full.pkl`, no outlier filter: pit / Safety Car / incident laps included, 217,079 rows 2024-2026).
  Models learn and every reported MAE / decision uses only the clean rows (`clean` = row survives the
  training filter, 200,072 rows; numbers identical to the filtered run), but every row gets a prediction,
  so the web predicts every lap that exists in the data (only lap 1 is absent: no previous lap for the
  inputs). A non-clean row with no "last clean lap" anchor falls back to its previous lap.
- **Drift lab (`_drift_lab.py`, reads `F1_STREAM_DIR`, default `online/fast/s2_incremental`):** one race
  = one step over the last 3 GPs of 2025 + 2026. v001 = SGD serves; ARF and kNN learn as shadow challengers.
  Trigger = PSI ≥ 0.25 on a telemetry feature (vs the same circuit in earlier seasons) or champion MAE ≥
  1.25 × its 2025 MAE (cooldown 2 races); a challenger is promoted only if it beats the champion by ≥ 0.010 s
  with green-lap MAE not worse by > 0.005 s. The benchmark table is computed from the same stream.
- **Web API (`_web_api.py`):** `POST /predict {driver_number, lap, session_key}` returns the prediction
  of the model that was serving at that race (plus SGD / ARF / kNN side by side); `GET /sessions` lists
  the covered races; `GET /race/{session_key}` returns every predicted sector of one race for the
  *Race replay*, which shows the served model's prediction before each sector starts, scores it when the
  sector ends (error per car in the timing tower, running MAE for the followed car and the field; when a
  sector has no prediction the card says why - lap 1, Safety Car / wet laps for most of the field, or this
  car's pit / incident - and when the next prediction comes) and moves on to the next sector. 2023 is
  warm-up only; pit / SC / incident laps are not in the stream. The server reloads `model/prod/` when it
  changes. `_online_s1.py web` writes `stage1_lab.json/.js` for the page's stage-1 section.
- 2026 races whose circuit has no `tracks.json` layout (Madring, Kuala Lumpur) are excluded.

**Result (66 races 2024 → 2026, 200,072 sectors; 2026 = 18 races):**

| | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| stage-1 score SGD / ARF / kNN | 36.6 / 1.18 / 1.11 | 1.67 / 1.08 / 1.02 | **0.99** / 1.09 / 1.04 | **1.49** / 1.63 / 1.56 |
| stage-1 served (champion) | 1.16 | 1.03 | 0.99 | 1.49 |
| "same as last lap" | 1.00 | 0.94 | 1.00 | 1.54 |

Stage 1: 8 promotions, mostly in 2023–24 while the empty models learn (SGD diverges at first); SGD serves
from 2024 Spa to the end. Stage 2, 2026 MAE (learning vs frozen after 2023): SGD 0.338 vs 0.414 s
(−18%), kNN 0.359 vs 0.369 s, ARF 0.376 vs 0.367 s; 2024–2025: SGD 0.300 vs 0.305 s. The final error equals
the earlier batch-RF stage 1 (SGD 2026 0.339 s), so going fully incremental costs nothing.
Drift lab: the last 3 GPs of 2025 are stable (0.216 vs 0.217 s frozen); every 2026 race shows major
feature drift (DRS gone, PSI 0.85–2.98); over the 18 races of 2026 the served model averages 0.340 s vs
0.422 s for SGD frozen (−19%); Monza 2026: 1.84 s frozen vs 0.47 s online. 10 decisions, 2 promotions
(kNN at Catalunya → v002, SGD back at Silverstone → v003). Served MAE over the stream 0.312 s.
Data notes: Bahrain (Sakhir) and Saudi Arabia 2026 have no laps in OpenF1; Madring and Kuala Lumpur are
excluded (no `tracks.json` layout; `_build_year.py` requires the track profile); Monaco 2026 location
telemetry stops a few laps into the race, so its track path comes from 2025 (`f1pit/circuits.py`).

---

## 10. Current feature engineering

### 10.1 Stage 1 (per channel)

`Prev_<channel>` (level/log variants) + `Tyre_Age_Total` + `Tyre_Age_<compound>` (+ squared) + weather
(5) + `Track_Status` + traffic (4) + `Lap` + `<channel>_roll5_mean/slope` + geometry:

| channel | geometry |
|---|---|
| Avg_Speed_Corners, Brake, Min_Speed | Total_Sector_Length, Straight_Ratio, Log_Total_Sector_Length |
| Avg_Speed_Straight, Max_Speed | + Total_Straight_Length, Log_Total_Straight_Length |
| Full_Throttle, DRS, Avg_Gear | + Total_Straight_Length |

`Num_Corner_*` is **not** used: the corner clusters are fitted per year (9 / 11 / 8 clusters in
2023 / 2024 / 2025), so `Num_Corner_k` means a different corner type each year.

### 10.2 Pace context (`f1pit/pace_context.py`, point-in-time)

| feature | definition (earlier laps only) |
|---|---|
| `Rep_Prev_Duration` | duration of the last *representative* lap in this sector: GREEN/UNKNOWN and ≤ `REP_TOL` (1.12) × median of the previous 5 green laps; falls back to `Prev_Duration` |
| `Rep_Best_Duration`, `Rep_Median3` | best / median-of-3 over representative laps |
| `Anchor_Age` | laps since that representative lap (NaN = none yet → anchor is the raw previous lap) |
| `Prev_Lap_NonGreen` | any sector of this driver's previous lap not GREEN |
| `Laps_Since_SC` | laps since the last SC/VSC lap of the session (99 = none) |
| `Rainfall_PrevLap`, `Rainfall_Delta`, `Field_Wet_PrevLap` | previous-lap rain, its change, share of the field on INTER/WET last lap |
| `Stint_Number`, `Is_First_Stint`, `Tyre_Age_x_Track_Temp` | stint context (not used by the current models) |

Verified point-in-time: deleting everything after (lap L, sector 1) leaves every value at lap L
unchanged (648 session/lap/feature checks, 0 mismatches).

### 10.3 Stage 2 (42 raw features before one-hot)

| group | features |
|---|---|
| `Pred_*` telemetry (8, stage-1 OOF output) | Avg_Speed_Corners, Avg_Speed_Straight, Full_Throttle, Brake, DRS, Max_Speed, Min_Speed, Avg_Gear |
| anchor (absolute scale) | Rep_Prev_Duration |
| relative pace (log-ratio to the anchor) | Rel_Prev_Duration, Rel_Clean_Pace, Rel_Best_Duration_So_Far, Rel_Rep_Best_Duration, Rel_Rep_Median3, Rel_Prev_Duration_roll5_mean, Rel_roll5_slope (= slope / anchor) |
| other pace | Prev_Sector_Duration |
| lap context | Anchor_Age, Prev_Lap_NonGreen, Laps_Since_SC, Prev_Lap_Pitted, Pit_Stops_Done, Lap |
| rain history | Rainfall_PrevLap, Rainfall_Delta, Field_Wet_PrevLap |
| compound | `Tyre` (one-hot) |
| weather (5) | Track_Temperature, Air_Temperature, Rainfall, Pressure, Humidity |
| track status | `Track_Status` (one-hot) + `Track_Status_Inferred` |
| traffic (4) | Traffic_Within_1s/2s/5s, Interval_Ahead |
| geometry / distance (4) | Total_Sector_Length, Total_Straight_Length, Total_Distance_Driven (fuel), Distance_On_Tyre |

## 11. Design notes / gotchas

- **Point-in-time correctness** — every feature is knowable *as of* the sector's `Time_Start`; no
  within-sector or future info leaks in (see `f1pit/features.py`, `f1pit/pace_context.py`).
- **`fetch()` is the only OpenF1 entry point** — shared session, exponential backoff, two-level cache
  (memory + `.cache/openf1`). Route datetime parsing through `api.to_utc`.
- **Why the representative anchor** — with `Prev_Duration` as the anchor, the delta is dominated by
  mean reversion: when the previous lap was abnormal (SC, pit out-lap, mistake) the target becomes
  "undo last lap's anomaly". 5.7% of rows (|error| > 1 s) carried 49% of the total error, concentrated
  after SC/VSC (`Laps_Since_SC` ≤ 1: 3% of rows, 28% of error) and on laps whose previous lap was
  non-green (2.7% of rows, 26% of error). Anchoring on the last representative lap removes that
  structure, so the telemetry path (`Pred_*`) no longer has to carry the previous-lap anomaly.
- **Relative pace** — durations differ by circuit/sector (20–40 s); expressing anchors as log-ratios
  to the target anchor makes the features scale-free (largest single gain for RF after the anchor).
- **Leakage-safe stacking** — stage-2 train rows get out-of-fold `Pred_*` (GroupKFold by session);
  the 2025 test season gets the full 2023+2024 refit.
- **SGD diverges on `squared_error`** for these targets — use `huber` + `invscaling` (see
  `_fit_sgd`). The signed-log delta (`delta_slog`) must have its prediction clipped before `expm1`
  or a linear model explodes.
- **Do not clip the stage-2 target.** A 6×MAD winsorisation of the delta destroys the SC/VSC rows
  (MAE 0.32 → 0.61).
- **Heavy-tailed telemetry deltas** — the delta (`current − prev`) has kurtosis 20–58 (occasional
  huge telemetry spikes). The MAD clip + signed-log target reduce, but do not remove, this.
- **Wet is under-sampled** (~1.2% of 2025 rows) — wet MAE is still ~4× the dry MAE.
- **Tyre signal is small** — extra tyre data in stage 1 (compound one-hot, `Distance_On_Tyre`,
  fuel proxy, stint, age × track temperature) did not improve any telemetry channel, so stage 1 uses
  only `Tyre_Age_*` × compound.

## 12. Latest results (v3, 2025 test, 70,410 rows)

### 12.1 Data-centric optimisation (models and hyperparameters fixed)

Run on a lab copy of the data with `Outcome_*` reconstructed from the next lap's `Prev_*`
(validated against the original run: baseline RF 0.3200 vs 0.3224). Each step is added on top of the
previous one; RF noise across seeds ≈ ±0.001–0.003 s.

| step | RF | KNN | SGD |
|---|---|---|---|
| v2 baseline (`Pred_*`, anchor `Prev_Duration`, train 2024) | 0.3200 | 0.4787 | 0.5230 |
| + Track_Status repair (red flag) | 0.3165 | 0.4793 | 0.5190 |
| + representative anchor (`Rep_Prev_Duration`) | 0.2701 | 0.3147 | 0.3203 |
| + lap context (prev-lap SC, laps since SC, pit, anchor age) | 0.2673 | 0.3002 | 0.3206 |
| − `Num_Corner_*` (stage 1 + 2) | 0.2643 | 0.2838 | 0.3165 |
| + rain history | 0.2643 | 0.2853 | 0.3120 |
| + relative pace (log-ratio to anchor) | 0.2449 | 0.2532 | 0.2910 |
| − speed traps (`Prev_I1/I2/St_Speed`) | 0.2447 | 0.2533 | 0.2904 |
| + stage-1 cross-fit + stage 2 trained on 2023+2024 (**v3**) | **0.2282** | **0.2470** | **0.2760** |

v3 vs v2 (session-level paired bootstrap, 24 races): RF −28.9% (95% CI −0.116 … −0.071 s), KNN −48%,
SGD −47%; **all 24 races improve for all three models**. Biggest gains are on the rows the old anchor
broke: after a non-green lap 3.11 → 1.36 s, SC/VSC 3.70 → 2.51 s, wet 1.17 → 0.84 s, green 0.300 →
0.214 s. The same data changes also hold on a different season (§12.3).

Tried and rejected: level target (no anchor), 6×MAD target clip, `log1p(Interval_Ahead)`, anchor =
`Clean_Pace` / `Rep_Median3`, `REP_TOL` 1.05 / 1.08 / 1.20 (1.08 within noise; kept 1.12 to avoid
tuning on the test year), stage-1 extra tyre data, stage-1 cross-fit without the 2023+2024 stage-2
training.

### 12.2 v3 end-to-end run (`_fforest.py` as committed, same lab data)

| model | MAE (s) | R² |
|---|---|---|
| RF | **0.2305** | 0.9927 |
| KNN | 0.2473 | 0.9909 |
| SGD | 0.2749 | 0.9847 |

**6-condition MAE (RF):** overall 0.2305 · clean 0.2168 · dry 0.2230 · wet 0.8552 · dry+clean 0.2102 ·
wet+clean 0.7885. `_predict.py` on the saved `model/v3/` reproduces the RF MAE exactly.

**Stage 1:** all 8 channels beat persistence. With `Num_Corner_*` removed the variant selection now
picks `log` (Avg_Speed_Corners), `level` (Full_Throttle, Brake) and `delta_slog` (the other 5). Out-of-fold
MAE on 2023+2024 ≈ the held-out-2024 MAE (e.g. corner speed 5.23 vs persistence 6.05).

**`Pred_*` share of RF importance:** 0.092 in this run (v2: 0.13). The §12.1 lab run kept the v2
variants (`delta_slog` / `delta`) and had a share of 0.14 at the same MAE (0.2282), so the drop comes
from the stage-1 variant re-selection, not from the data changes. Top features: `Rel_Best_Duration_So_Far`,
`Anchor_Age` missing-indicator (no representative lap yet), `Rep_Prev_Duration`,
`Rel_Prev_Duration_roll5_mean`, `Pred_Avg_Speed_Corners`.

`fforest_results.json` / `model/v3/` in the repo are produced by `_fforest.py` on the real
`data_*_out.parquet`; re-run it (after §3.1) to refresh them.

### 12.3 Temporal replication

To check the choices were not tuned to 2025, the same feature/data design was re-run as
train 2023 → test 2024 (both seasons with out-of-fold `Pred_*`; the "train on 2 seasons" lever cannot
be replicated here):

| features | RF | KNN | SGD |
|---|---|---|---|
| v2 stage-2 features | 0.3575 | 0.4085 | 0.4806 |
| v3 stage-2 features | 0.3060 (−14%) | 0.2715 (−34%) | 0.3211 (−33%) |

### 12.4 Online learning (`_online.py`, §9a)

Warm-up 2023 (65,200 rows) → stream 2024 + 2025 (143,427 rows, 48 races, 308 ten-lap chunks), predict
then learn. MAE in seconds per sector (`online_results.json`):

| model | all | 2024 | 2025 | 2025 wet |
|---|---|---|---|---|
| **SGD online** (best online) | **0.2998** | 0.3025 | 0.2971 | 0.921 |
| ARF online (Adaptive Random Forest, 10 Hoeffding trees + ADWIN) | 0.3017 | 0.3045 | 0.2988 | **0.900** |
| kNN online (k = 15, last 2,000 rows) | 0.3322 | 0.3300 | 0.3346 | 0.925 |
| ARF frozen after warm-up | 0.3161 | 0.3153 | 0.3170 | 0.906 |
| SGD frozen after warm-up | 0.3046 | 0.3058 | 0.3034 | 0.900 |
| RF batch, trained once on 2023 | 0.2955 | 0.3160 | 0.2742 | 1.736 |

Session-level paired bootstrap (95% CI): SGD − ARF online −0.0019 s [−0.0083, +0.0039] → **tie**;
ARF online − frozen −0.0144 s [−0.0207, −0.0086] and SGD online − frozen −0.0047 s [−0.0068, −0.0026]
→ **updating helps**; kNN online − frozen +0.0006 s [−0.008, +0.011] → no gain (a sliding window
forgets as fast as it learns). SGD is picked as the online model: same accuracy as ARF, but warm-up +
stream take seconds (13–55 s) instead of ~15 min for ARF. The frozen batch RF is still the most accurate on dry races (it has the full
2023 forest) but has no way to adapt to rain: wet MAE 1.74 s vs ~0.90 s for every online model.

Note: stage 1 here is fitted on 2023 only, so these numbers are not comparable with §12.2 (stage 1
cross-fitted on 2023+2024, stage 2 trained on two seasons). Re-running `_online.py` reproduces the
stream exactly; SGD gives 0.2999 vs 0.2998 and RF 0.2951 vs 0.2955 — stage-1 DRS / Avg_Gear differ
only through the RF's column-order randomness.

## 13. File reference

| file | purpose |
|---|---|
| `f1pit/api.py` | OpenF1 client (`fetch`, cache, backoff, `to_utc`) |
| `f1pit/tracks.py`, `circuits.py` | circuit geometry + corner database |
| `f1pit/clustering.py`, `context.py` | corner clustering + `TrackContext` |
| `f1pit/features.py` | `build_sector_table` (point-in-time feature builder), race-control parser, `repair_track_status` |
| `f1pit/pace_context.py` | `add_pace_context` (Rep anchors, lap context, rain history), `add_relative_pace` |
| `f1pit/dataset.py` | `build_dataset` / `save_df` / `load_df` |
| `f1pit/train.py` | `prepare_data`, `train_models`, `build_preprocessor`, tyre interaction, `_roll5_` trends |
| `f1pit/wind_rain_lab.py` | `_prepare_lab` (ML frame prep reused by the forest) |
| `_rebuild_outcomes.py` | rebuild parquet with `Outcome_*` columns |
| `_repair_status.py` | repair Track_Status in existing parquet (red-flag fix) |
| `_prewarm_cache.py` | concurrent OpenF1 cache pre-warm |
| `_fforest.py` | train the forest-of-forests (stage 1 + stage 2) + save models |
| `_predict.py` | load saved models + predict (no retrain) |
| `_write_report.py` | generate `fif_report_v2.md` |
| `_dist_analysis.py` | skew/kurtosis + histogram diagnostics |
| `_monza_ttest.py` | paired t-test between two drivers |
| `_online.py` | online stage 2 (river ARF / SGD / kNN, 10-lap prequential stream) → `online_results.json` |
| `_build_year.py` | build one season (OpenF1 pre-warm → TrackContext → dataset), e.g. 2026 |
| `_online_s1.py` | fully incremental pipeline: online stage 1 (SGD / ARF / kNN) + its champion / challenger + online stage 2 → `online/fast/`, `online_s1_*.json`, `stage1_lab.json` (§9b) |
| `_stream_fast.py` | fast online learners with one chunk interface: SGD (river), ARF (CapyMOA), kNN (numpy) (§9b) |
| `_drift_lab.py` | race-by-race champion / challenger over the incremental stream → `drift_lab.json`, `model/prod/` (§9b) |
| `_deck_drift_fig.py`, `_deck_s1_fig.py` | slide figures for the 2026 drift and the incremental stage 1 → `figures/` |
| `_web_api.py` | FastAPI server for the F1 web page: online-lineage predictions per race / driver / lap (§9b) |
| `_data_quality.py` | 5 data-quality checks → `data_quality_report.md` / `data_quality.json` |
| `_anova_chi2.py` | one-way ANOVA (tyre degradation) + Chi-square (pit vs SC) → `hypothesis_extra.json` |
| `_eda_figures.py` | EDA figures for the slides → `figures/eda_*.png` |
