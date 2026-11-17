# f1pit — Report v2: Current Problems & Next Feature-Engineering Steps

> Follow-up to `report.md`. This document states the problems we are facing now
> (as established by every experiment this session) and records the next
> feature-engineering plan.

## 1. The problems we face now

### 1.1 Random Forest is the ceiling, and it is stuck at ~0.26 s

| model | full MAE (s) | green MAE (s) |
|---|---|---|
| **Random Forest** | **0.2649** | **0.2439** |
| KNN (top-5 features) | 0.317 | 0.294 |
| SVR rbf (top-5) | 0.345 | 0.323 |
| EBM | 0.579 | 0.565 |
| LinearRegression / Ridge | 0.918 | ~0.90 |
| DNN (best) | 1.258 | 1.227 |

Every non-tree model is 2–10× worse. More capacity, interpretability, or kernel
tricks do not help. The remaining error is **not** a modeling problem — it is a
**feature/condition problem**.

### 1.2 The error is concentrated in unforecastable events *and* stale features

- **Wet** laps: 4.6× the dry error (1.18 vs 0.25 s), but only 4.1% of aggregate
  error because they are only 1.2% of sectors.
- **SC/VSC** laps: 2.0–3.9 s error, ~7.4% of aggregate error.
- **The single biggest individual errors (~10 s)** come from **stale pace anchors
  during transitions**. Two concrete cases from `top_errors_2025.csv`:
  - **Jeddah lap 4**: a field-wide ~10 s **over-prediction**. `Prev_Duration` was
    ~60–69 s (from an SC/incident lap), so the model predicted slow, but lap 4 was
    a clean 30 s.
  - **Silverstone (wet/drying)**: ~10 s **under-prediction**. `Rainfall=0` (binary
    flag says "dry") but the cars were on INTERMEDIATE tyres doing ~50 s sectors vs
    a ~38 s prediction — the track was drying and the model couldn't see it.

### 1.3 What we have already ruled out

- **Concept drift**: none (0 drift events in both warm-window configs). Static RF
  generalizes flatly across 2025.
- **Data bias fixes**: upsampling, sample weights, and SMOTE all fail or hurt.
  SC/VSC/wet are rare because they are *unpredictable*, not under-represented.
- **Bringing outliers back**: triples the error (0.843 s). The outlier filter is
  load-bearing.

## 2. Feature engineering v2 (suggestions)

### 2.1 Make pace anchors SC/VSC-aware (green-only anchoring)

**Problem.** `Prev_Duration`, `Clean_Pace`, and `Best_Duration_So_Far` carry pace
information from the immediately-preceding lap even when that lap was under
SC/VSC. That stale anchor is the direct cause of the Jeddah-lap-4 class of errors.

**Rule.** When building these features, *skip non-green laps*:

- If `Prev_Duration` (or any `Prev_*`) comes from an SC/VSC/SC_ENDING/VSC_ENDING
  lap, walk back to the **previous-of-previous**, then further back, **until a
  GREEN lap is found**, and use that value.
- Same for `Clean_Pace` / `Best_Duration_So_Far`: never let a non-green event
  contribute to the "clean"/"best" reference.

**Correctness.** This is point-in-time safe: it only ever looks at *past* laps
(never the current or future sector), so no information leaks.

**Expected effect.** Removes the largest ~10 s errors caused by stale anchors
(SC/VSC start/end, lap-1 incidents). SC/VSC still *happens*, but it stops
poisoning the features the model reads on subsequent green laps.

### 2.2 Add a wet/drying signal (last-lap rainfall delta)

**Problem.** `Rainfall` is binary and only tells us "raining now". It cannot
detect a *drying* track (Silverstone) or *worsening* conditions.

**Rule.** Add
```
Rainfall_Delta = Rainfall(current lap) - Rainfall(previous lap)
```
(and the companion `Rainfall_PrevLap`) so the model can see whether the track is
getting wetter or drying. A negative delta = drying; positive = worsening.

**Already prototyped.** `f1pit/wind_rain_lab.py` already computes
`Rainfall_PrevLap` and `Rainfall_Delta` (`_rain_features()`), but they are **not
wired into the default feature set** — the baseline trains on 72 features without
them. Next step is to promote them into the production pipeline.

**Richer option.** The Open-Meteo `Rain_mmh` (continuous precipitation mm/h,
merged as-of `Time_Start`) is the quantitative version of the same idea — it gives
intensity, not just a flag. (See `_rain_mmh.py`.)

### 2.3 Re-run feature engineering and evaluate

1. Recompute the pace anchors with green-only walk-back (2.1).
2. Add `Rainfall_Delta` / `Rainfall_PrevLap` (and optionally `Rain_mmh`) to the
   feature set (2.2).
3. Retrain the frozen RF (2023+24 → 2025) and compare:
   - overall full / green MAE,
   - per-regime MAE (dry / wet / SC / VSC),
   - and specifically re-check the Jeddah-lap-4 and Silverstone rows in
     `top_errors_2025.csv` to confirm the ~10 s errors shrink.

## 3. Priority

1. **2.1 (green-only pace anchors)** — highest value: it fixes the largest errors
   and is cheap (a feature-computation change, no new data source).
2. **2.2 (rainfall delta)** — fixes the wet/drying blindness; wire in the existing
   `_rain_features()` first, then consider `Rain_mmh`.
3. Re-evaluate against `report.md` baselines before touching the model itself.

**North-star metric:** clean dry-green MAE = **0.234 s** (no rain, no SC/VSC) is
the current honest floor. Feature work should push the *overall* number toward it
by reducing the transition errors, not by changing the model.
