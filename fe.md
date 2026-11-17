# Feature Engineering — Current Process

This documents the feature-engineering pipeline **exactly as it is implemented
today** (no proposed changes). Source of truth: `f1pit/features.py`,
`f1pit/dataset.py`, `f1pit/train.py`, `f1pit/online_lab.py`.

## The one invariant everything obeys

**Point-in-time correctness.** Every feature must be knowable as of that sector's
`Time_Start`. Nothing from inside the current sector or from the future may leak
in. `Prev_*` features come only from *earlier* laps; weather/interval are joined
`merge_asof(..., direction="backward")`; `Track_Status` is derived only from
race-control events that happened before `Time_Start`. The current sector's
telemetry (`Outcome_*`) is computed but is explicitly *not* used as its own
feature — only later sectors/laps see it (as `Prev_*`).

There are four stages:

- **A.** `build_sector_table()` — raw API data → one sector-row per driver-session, plus the base features.
- **B.** `build_dataset()` — loop A over all sessions × drivers and concatenate.
- **C.** `prepare_data()` — turn the concatenated table into the final ML frame.
- **D.** feature selection + preprocessing at fit time.

---

## Stage A — `build_sector_table()` (`f1pit/features.py`)

Runs once per (driver, session). Inputs are raw OpenF1 endpoints; output is a
lap-sector DataFrame.

### A1. Resolve the session
`resolve_session()` picks the session row (by `session_key`, or by year/country/
session-name).
**Why:** identify the exact race. `session_key` disambiguates countries with
multiple circuits (USA, Italy).

### A2. Load laps
`_load_laps()`: fetch `laps`, coerce `date_start` to UTC, convert numeric cols
(`lap_number`, `duration_sector_1/2/3`, `i1/i2/st_speed`), keep an
`is_pit_out_lap` flag, sort by lap number.
**Why:** laps are the skeleton — one row per lap with the three sector durations
(which later become the target) and the speed-trap values.

### A3. Load stints
`_load_stints()`: fetch `stints`, numeric `lap_start`/`lap_end` (open stint →
`inf`), `tyre_age_at_start`.
**Why:** to know which tyre compound was fitted on each lap and to compute tyre
age.

### A4. Load car telemetry (cumulative-sum index)
`CarTelemetry`: fetch `car_data` (speed/throttle/brake/drs/n_gear at ~4 Hz),
sort by time, and precompute cumulative sums so any `[start,end]` window's mean
is O(1). It also precomputes *conditional* speed (corner = throttle<99, straight
= throttle≥99).
**Why:** the sector's telemetry outcome must be aggregated over the sector's time
window; cumulative sums make this fast and exact.

### A5. Load race control → control events
`extract_control_events()`: fetch `race_control`, classify each message into
`SC_DEPLOYED` / `SC_ENDING` / `VSC_DEPLOYED` / `VSC_ENDING` / `GREEN`, and attach
a lap number (inferred from lap start times when missing).
**Why:** these events are the only source of the track's SC/VSC state; everything
else (status flags) is derived from this timeline.

### A6. Load pits
`_load_pits()`: fetch `pit` → `{lap_number: pit_duration}`.
**Why:** pit stops change pace; the features must know when a stop happened and
how long it took.

### A7. Load weather
`_load_weather()`: fetch `weather`, numeric `track_temperature/rainfall/pressure/
humidity/air_temperature`, sort + de-duplicate by `date`.
**Why:** environmental conditions (esp. the binary `rainfall`) are features, to be
joined as-of.

### A8. Load intervals (this driver)
`_load_intervals()`: fetch `intervals` for the driver, compute `Interval_Ahead`
(gap to the car ahead). The leader and lapped cars are mapped to a "far" value.
**Why:** distance to the car ahead is a pace/clean-air feature.

### A9. Load intervals (all drivers) → traffic
`_load_traffic()`: fetch `intervals` for *everyone*, build a per-driver
`gap_to_leader` time series.
**Why:** to later count how many other cars are within N seconds (traffic density),
which needs every car's gap, not just the current driver's.

### A10. Build sector records
`_sector_records()`: for each lap, walk sectors 1→3 accumulating the sector
`Time_Start`/`Time_End` from the durations; compute per sector:
- tyre compound + `Tyre_Age` (= `lap - stint.lap_start + 1`) and
  `Tyre_Age_Total` (= age + `tyre_age_at_start`),
- `Track_Status` (from the event timeline at `Time_Start`) + `Track_Status_Inferred`,
- telemetry outcomes over the sector window (`Outcome_*`: avg speed, corner/
  straight speed, full-throttle %, brake %, DRS %, max/min speed, avg gear,
  `Outcome_Duration`).
**Why:** this converts lap-level data into the sector-level rows that are the
actual training examples, and computes the current sector's *outcomes* (which are
the target and the raw material for future `Prev_*` features).

### A11. Add history (backward-looking) features
`_add_history_features()` — the heart of point-in-time feature engineering. For
each sector, using **only earlier laps in the same sector** (plus the immediately
preceding sector in time):

- Mark `_clean` = GREEN/UNKNOWN, not lap 1, not an in-lap, not an out-lap, not a
  pit lap. **Why:** "clean" laps are the reference pace; dirty laps (SC/VSC/pits)
  would corrupt that reference.
- `Prev_Duration` = previous lap's `Outcome_Duration` in this sector.
  **Why:** the driver's most recent time through this sector is the strongest pace
  predictor.
- `Prev_*` telemetry (avg speed corners/straight, full-throttle %, brake %, DRS %,
  max/min speed, avg gear) = last valid prior-lap outcome of each.
  **Why:** recent driving-behaviour features (how the driver drove here last time).
- `Prev_Sector_Duration` = duration of the immediately preceding sector in time.
  **Why:** sequential context — the sector just finished.
- `Prev_I1/I2/St_Speed` = speed-trap values from the last lap that has them
  (backward ffill). **Why:** speed-trap readings are sparse; carry the latest
  known forward.
- `Clean_Pace` = median of the last `clean_window=3` clean laps' durations in this
  sector (fallback: all durations if none are clean yet).
  **Why:** a robust "reference pace" — median ignores one-off slow laps.
- `Best_Duration_So_Far` = minimum prior duration in this sector.
  **Why:** best-so-far reference (how fast the driver has been).
- Pit features: `Will_Box` (pitting next lap), `Prev_Lap_Pitted`, `Pit_Stops_Done`,
  `Prev_Lap_Box_Duration` (pit time two laps ago). **Why:** pit-cycle context.

**Why this whole step:** all of these are the "memory" of the model, and they are
built strictly from the past, so nothing leaks.

### A12. Drop rows without history
If `drop_no_history=True`, drop rows where `Prev_Duration` is NaN (the first lap).
**Why:** on lap 1 the `Prev_*`/pace features are undefined.

### A13. As-of join weather
`_asof_join(..., direction="backward", tolerance=2 min)` merges the latest weather
observation at or before `Time_Start`.
**Why:** weather must reflect conditions *now* without looking ahead; the 2-min
tolerance tolerates gaps between weather samples.

### A14. As-of join interval
Same backward join for `Interval_Ahead`; fill NaN with `interval_fill=60`.
**Why:** gap-to-car-ahead as-of `Time_Start`; "60 s" encodes "no car nearby".

### A15. Meta columns
Attach `Year`, `Country`, `Session`, `Session_Key`, `Driver`.
**Why:** identifiers for grouping/later analysis.

### A16. Traffic counts
For each sector, count other cars whose `|Δ gap_to_leader|` is within each
threshold in `(1.0, 1.5, 2.0, 5.0)` s → `Traffic_Within_1s/1.5s/2s/5s`.
**Why:** traffic density (how many cars you are fighting) affects lap time.

### A17. Track-profile features
`add_track_profile()`: from the `TrackContext`, attach `Total_Sector_Length`,
`Total_Straight_Length`, and `Num_Corner_*` (corner-cluster counts) per sector.
**Why:** static circuit geometry — how long the sector is and how many/what kind
of corners it has.

### A18. Distance features
`Total_Distance_Driven = Circuit_Total_Length × Lap`;
`Distance_On_Tyre = Circuit_Total_Length × Tyre_Age_Total`.
**Why:** cumulative-distance proxies for fuel load and tyre wear.

### A19. Set the target
`Duration = Outcome_Duration` (the current sector's time).
**Why:** this is what the model predicts.

---

## Stage B — `build_dataset()` (`f1pit/dataset.py`)

Loops over every (session, driver), calls `build_sector_table`, and concatenates
all tables into one DataFrame (failures are caught per driver so one bad session
doesn't kill the whole year).
**Why:** the model needs a single multi-year training table and a separate test
year, all in one consistent schema.

---

## Stage C — `prepare_data()` (`f1pit/train.py`)

Turns the raw sector table into the final ML frame (features + target).

### C1. Prev_* trend features
`add_prev_trend_features()`: for each `Prev_*` base column (and
`Prev_Sector_Duration`), compute a **rolling mean** and **rolling slope** over the
last `PREV_TREND_WINDOW=5` laps (backward, within driver).
**Why:** recent *trend* (is the driver getting faster or slower) — a derivative
signal on top of the raw previous value.

### C2. Log target
`Log_Duration = ln(Duration)`.
**Why:** durations are positive and skewed; the log makes errors multiplicative
and roughly symmetric, which fits tree/regression objectives better.

### C3. Remove slow-lap outliers
`remove_duration_outliers(k=4.0, mad_floor=0.5)`: within each (session, sector),
drop rows where `Duration > median + 4 × MAD` (MAD floored at 0.5 s).
**Why:** strips the slow tail (accidents / collisions / SC / yellow-flag laps) so
the model isn't trained on unforecastable chaos.

### C4. Rename weather columns
`track_temperature→Track_Temperature`, `rainfall→Rainfall`, … (via `WEATHER_MAP`).
**Why:** consistent capitalized naming across stages.

### C5. Subset to the feature list
Keep only `CATEGORICAL_FEATURES` (`Tyre`, `Track_Status`) + `NUMERIC_FEATURES` +
corner columns + `Circuit_Total_Length` (kept only for the tyre-age interaction,
not a feature itself).
**Why:** drop meta/outcome columns; keep only what the model may use.

### C6. Normalise categoricals
Uppercase + strip `Tyre`/`Track_Status`; fill missing `Tyre`/`Track_Status` with
NA / "UNKNOWN".
**Why:** clean, consistent category labels for one-hot encoding.

### C7. Coerce numerics
Convert every non-categorical feature to float via `_to_float`.
**Why:** uniform numeric dtype for the model.

### C8. Drop unusable rows
`dropna` on required cols; keep only `Tyre_Age > 0`, `Prev_Avg_Speed_Corners > 0`,
`Prev_Duration > 0`, and `Sector ∈ {1,2,3}`.
**Why:** remove first-lap rows, sensor-garbage rows, and any non-race sectors.

### C9. Tyre-age interaction features
`add_tyre_age_interaction()`: **replace** the `Tyre` one-hot and bare `Tyre_Age`
with:
- `Tyre_Age_<compound> = 1[compound] × Tyre_Age × Circuit_Total_Length`
- `Tyre_Age_<compound>_2 = 1[compound] × Tyre_Age² × Circuit_Total_Length`

for each compound in `{SOFT, MEDIUM, HARD, INTERMEDIATE, WET}`.
**Why:** tyre degradation depends on *distance*, not just lap count, so age is
scaled by lap length; the `×length` scaling lets different circuits be compared.
The quadratic term captures non-linear degradation.

---

## Stage D — feature selection + preprocessing at fit time

### D1. Feature selection (`online_lab._features`)
Drop `Num_Corner_*` columns and append `Lap`.
**Why:** corner-cluster *ids* are recomputed per year (a different k/assignment
each year), so they don't transfer across years; `Lap` is added as a feature.

### D2. Preprocessing (`train.build_preprocessor`)
A `ColumnTransformer`:
- categorical (`Track_Status`): most-frequent impute → one-hot (unknown ignored),
- numeric: median impute + missing indicator.
**Why:** tree models can't take NaN; median is a robust numeric fill and the
missing-indicator lets the model know a value was absent. One-hot makes
`Track_Status` usable as numeric splits.

---

## Feature inventory (what the model actually sees)

**Categorical:** `Track_Status` (one-hot).
**Numeric:** tyre-age interactions (`Tyre_Age_<compound>`, `…_2`), `Track_Status_Inferred`,
all `Prev_*` pace/telemetry columns, `Prev_Sector_Duration`, `Clean_Pace`,
`Best_Duration_So_Far`, `Interval_Ahead`, `Traffic_Within_1s/1.5s/2s/5s`, pit
features (`Will_Box`, `Prev_Lap_Pitted`, `Prev_Lap_Box_Duration`, `Pit_Stops_Done`),
weather (`Track_Temperature`, `Rainfall`, `Pressure`, `Humidity`, `Air_Temperature`),
track profile (`Total_Sector_Length`, `Total_Straight_Length`), distance features,
`Lap`, and all the `Prev_*_roll5_mean/_slope` trend columns.
**Target:** `Log_Duration = ln(Duration)`.

*(Note: `f1pit/wind_rain_lab.py` contains a separate experimental feature set —
`Wind_*`, `Rainfall_PrevLap`, `Rainfall_Delta` — but it is **not** part of the
default pipeline above; it is only used in the wind/rain lab.)*
