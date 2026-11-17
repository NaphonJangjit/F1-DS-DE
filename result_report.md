# result_report.md — Baseline Model Diagnosis (old RF, no feature engineering)

Setup: frozen `RandomForestRegressor(n_estimators=150, max_features=0.3, min_samples_leaf=2)`,
train 2023+2024 → test 2025, target `Log_Duration`, error in seconds.
Tyre conditions: **DRY** = SOFT/MEDIUM/HARD, **INTER** = INTERMEDIATE (there is no WET tyre in 2025).

---

## 1. MAE by Race

| Race | full MAE | dry MAE | inter MAE | n | dry | inter |
|---|---|---|---|---|---|---|
| United Kingdom | 1.307 | 0.907 | 1.477 | 1904 | 568 | 1336 |
| Australia | 0.967 | 1.294 | 0.881 | 1886 | 393 | 1493 |
| Azerbaijan | 0.494 | 0.494 | — | 2602 | 2602 | 0 |
| Monaco | 0.323 | 0.323 | — | 3952 | 3952 | 0 |
| Belgium (Spa) | 0.303 | 0.303 | — | 1892 | 1892 | 0 |
| Brazil | 0.274 | 0.274 | — | 3279 | 3279 | 0 |
| United States | 0.242 | 0.242 | — | 2416 | 2416 | 0 |
| Saudi Arabia | 0.231 | 0.231 | — | 2480 | 2480 | 0 |
| Canada | 0.223 | 0.223 | — | 3724 | 3724 | 0 |
| China | 0.217 | 0.217 | — | 3073 | 3073 | 0 |
| Japan | 0.206 | 0.206 | — | 3049 | 3049 | 0 |
| Netherlands | 0.205 | 0.205 | — | 3297 | 3297 | 0 |
| Singapore | 0.203 | 0.203 | — | 3537 | 3537 | 0 |
| Spain | 0.196 | 0.196 | — | 3204 | 3204 | 0 |
| Austria | 0.178 | 0.178 | — | 3149 | 3149 | 0 |
| Hungary | 0.175 | 0.175 | — | 3976 | 3976 | 0 |
| United States | 0.175 | 0.175 | — | 3015 | 3015 | 0 |
| Italy | 0.172 | 0.172 | — | 2991 | 2991 | 0 |
| Bahrain | 0.172 | 0.172 | — | 2973 | 2973 | 0 |
| United States | 0.170 | 0.170 | — | 1412 | 1412 | 0 |
| Mexico | 0.162 | 0.162 | — | 3594 | 3594 | 0 |
| United Arab Emirates | 0.162 | 0.162 | — | 3354 | 3354 | 0 |
| Qatar | 0.152 | 0.152 | — | 2867 | 2867 | 0 |
| Italy | 0.138 | 0.138 | — | 2784 | 2784 | 0 |

**Only two races contain INTERMEDIATE sectors: United Kingdom (1336) and Australia (1493).**
Every other race is dry-only and sits at 0.14–0.49 s. In the UK the inter laps are *worse* than
dry (1.48 vs 0.91); in Australia the inter laps are *better* than dry (0.88 vs 1.29) — see §4.

---

## 2. Train vs Test feature distribution (KS) per condition

Distribution shift is measured with the two-sample Kolmogorov–Smirnov statistic (0 = identical,
1 = disjoint). Large n makes p-values meaningless, so read the KS effect size.

### DRY — train ≈ test (negligible shift)

| feature | KS | train mean | test mean |
|---|---|---|---|
| Humidity | 0.170 | 51.53 | 51.19 |
| Pressure | 0.123 | 986.60 | 985.85 |
| Track_Temperature | 0.122 | 36.04 | 35.73 |
| Prev_St_Speed | 0.114 | 301.75 | 304.16 |
| Prev_St_Speed_roll5_mean | 0.102 | 299.71 | 301.93 |
| Tyre_Age_HARD | 0.100 | 46602.8 | 39791.9 |
| Tyre_Age_HARD_2 | 0.100 | 1.13e6 | 0.98e6 |
| Air_Temperature | 0.098 | 24.27 | 24.09 |
| Track_Status_Inferred | 0.095 | 0.35 | 0.45 |
| Pit_Stops_Done | 0.093 | 1.02 | 0.89 |
| Prev_I2_Speed | 0.085 | 250.11 | 255.73 |
| Prev_I2_Speed_roll5_mean | 0.079 | 248.53 | 253.95 |

All remaining DRY features have KS < 0.08. **The dry population is stable across years** —
which is exactly why the model predicts dry laps at R² ≈ 0.996.

### INTER — train ≉ test (large shift)

| feature | KS | train mean | test mean |
|---|---|---|---|
| Track_Status_Inferred | 0.624 | 0.33 | 0.96 |
| Total_Sector_Length | 0.614 | 1455.3 | 1856.6 |
| Track_Temperature | 0.604 | 24.14 | 21.09 |
| Prev_Avg_Speed_Corners | 0.601 | 137.50 | 159.18 |
| Prev_Avg_Speed_Corners_roll5_mean | 0.563 | 136.24 | 155.87 |
| Total_Straight_Length | 0.528 | 741.4 | 1019.7 |
| Prev_Sector_Duration_roll5_mean | 0.504 | 29.85 | 34.30 |
| Air_Temperature | 0.492 | 19.14 | 16.88 |
| Pressure | 0.490 | 965.9 | 1000.1 |
| Prev_I2_Speed | 0.488 | 238.71 | 264.38 |
| Pit_Stops_Done | 0.452 | 0.97 | 2.23 |
| Prev_I2_Speed_roll5_mean | 0.370 | 233.78 | 254.63 |

**The INTERMEDIATE condition shifts dramatically (KS 0.5–0.6) while DRY barely moves (KS ≤ 0.17).**
The 2025 inter laps (UK + Australia) are colder, on longer/faster sectors, with higher corner
speed and far more "inferred" track status than the inter laps the model saw in 2023–24.
This is a **coverage + shift problem**: the wet-race conditions in 2025 were under-represented
and *different* from training.

---

## 3. Residual analysis (dispersion + directional bias)

Residual = actual − predicted (seconds). Positive = under-predicted (model too fast), negative = over-predicted.

| condition | n | MAE | RMSE | mean bias | std | under-pred | over-pred |
|---|---|---|---|---|---|---|---|
| DRY | 67,581 | 0.227 | 0.489 | **+0.013** | 0.489 | 46.98% | 53.02% |
| INTER | 2,829 | 1.162 | 2.178 | **+0.951** | 1.960 | **75.89%** | 24.11% |

**INTERMEDIATE shows both high dispersion and a strong directional bias:**
- **Dispersion:** std = 1.96 s vs 0.49 s for dry — **4× more variable** errors.
- **Bias:** mean residual = **+0.95 s**, and **76% of inter sectors are under-predicted** — the
  model systematically predicts intermediate laps too *fast*.
- DRY is essentially unbiased (+0.013 s) and symmetric (47%/53%).

So the intermediate failure is **not** random scatter — it is a systematic under-prediction
(the model thinks wet laps should be near dry pace).

---

## 4. Compare within the same race (Australia 2025)

Australia is the cleanest "same race, both conditions" case (393 dry-tyre + 1493 inter sectors,
controlling for circuit/event).

| condition (Australia only) | n | MAE | mean bias |
|---|---|---|---|
| DRY tyres | 393 | **1.294** | +0.775 |
| INTERMEDIATE tyres | 1493 | **0.881** | +0.708 |

**Controlling for the circuit reverses the picture:** inside Australia, the *dry-tyre* laps are
*worse* (MAE 1.29 vs 0.88) than the inter laps. Both are under-predicted (positive bias), but the
dry laps on a wet track — drivers caught on dries during rain/SC — are the harder case, not the
inter laps. This shows **tyre compound is a proxy for track condition, not the causal driver**:
"dry tyre on a wet track" is worse than "inter tyre on a wet track".

---

## 5. Summary

1. **Error is concentrated in two wet races (UK, Australia)** — the only races with inter tyres.
2. **DRY has no train→test shift (KS ≤ 0.17)** → predicted near-perfectly (R² 0.996).
3. **INTER has a huge train→test shift (KS 0.5–0.6)** → the 2025 wet conditions were under-seen
   and different from training.
4. **INTER residuals are both high-dispersion (4×) and systematically biased +0.95 s** (76% under-predicted).
5. **Within the same race (Australia), dry-tyre-on-wet-track is worse than inter** — tyre label is
   a proxy for the real variable (track wetness), which the model can't see from a binary rainfall flag.

The actionable conclusion is the same as throughout: the fix is a **real track-wetness signal**
(continuous rain intensity / field pace residual), not more model capacity.
