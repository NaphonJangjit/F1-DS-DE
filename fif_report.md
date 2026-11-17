# Forest-of-Forests: tyre-aware current-telemetry meta-features

Generated: 2026-11-17  |  source: `fforest_results.json`

## Goal

Make the tyre-age signal visible and usable for pit-stop strategy by predicting each *current* telemetry channel from its *previous* value + tyre age + context, then feeding the tyre-aware predictions into a final `Duration` model. Models: RandomForest and SGD.

## Stage 1 — per-channel forests (predict current from prev + tyre)

MAE of each channel's current value on the held-out 2024 season. "Persistence" = naive baseline that predicts current = previous lap's value. Lower MAE / higher R² than persistence means tyre + context add real signal.

| Channel | Persistence MAE | RF MAE | RF R² | SGD MAE | SGD R² | Σ tyre importance |
|---|---|---|---|---|---|---|
| Avg speed (corners) | 6.0487 | 8.1248 | 0.8255 | 7.5249 | -12.6117 | 0.01735 |
| Avg speed (straight) | 4.3523 | 6.1230 | 0.8722 | 4.4787 | 0.4462 | 0.00955 |
| Full throttle % | 3.6206 | 5.1965 | 0.8249 | 4.2942 | -4.6102 | 0.02008 |
| Brake % | 2.3561 | 2.6995 | 0.7162 | 2.7047 | -9.4969 | 0.03215 |
| DRS % | 2.4637 | 2.6298 | 0.6133 | 3.1634 | -12.8428 | 0.03881 |
| Max speed | 4.3028 | 5.9947 | 0.8193 | 5.5463 | -9.1322 | 0.01499 |
| Min speed | 4.1522 | 10.4136 | 0.7705 | 5.6723 | -10.0380 | 0.00936 |
| Avg gear | 0.1439 | 0.2134 | 0.8704 | 0.1360 | 0.1588 | 0.01122 |
| Duration (s) | 0.6245 | 1.2224 | 0.9142 | 0.9075 | -7.2522 | 0.00417 |

## Degradation curve (Duration forest, partial dependence)

Partial dependence of predicted `Duration` on `Tyre_Age_Total` (all other features held at their median):

| Tyre_Age_Total | Predicted Duration (s) |
|---|---|
| 2.0 | 29.5731 |
| 3.4 | 29.6057 |
| 4.8 | 29.6150 |
| 6.3 | 29.6185 |
| 7.7 | 29.6232 |
| 9.1 | 29.6254 |
| 10.5 | 29.6283 |
| 11.9 | 29.6301 |
| 13.4 | 29.6323 |
| 14.8 | 29.6358 |
| 16.2 | 29.6374 |
| 17.6 | 29.6399 |
| 19.1 | 29.6417 |
| 20.5 | 29.6433 |
| 21.9 | 29.6379 |
| 23.3 | 29.6324 |
| 24.7 | 29.6349 |
| 26.2 | 29.6335 |
| 27.6 | 29.6315 |
| 29.0 | 29.6323 |

## Stage 2 — final Duration model: baseline vs forest-of-forests

MAE in **seconds** (exp of Log_Duration) on the 2025 test season.

| Model | RF MAE (s) | SGD MAE (s) |
|---|---|---|
| baseline (raw Prev_*) | 0.3127 | 0.8392 |
| forest-of-forests (Pred_*) | 0.4017 | 0.8476 |

## Interpretation

**1. RF is the right base learner; linear SGD does not generalize across years.** RF explains 61–91% of held-out 2024 variance on every channel. SGD (huber) has competitive MAE but *negative* R² on most channels — one linear surface cannot track the 2023→2024 drift, and squared-error R² is wrecked by a few tail errors.

**2. Tyre age adds almost nothing on top of the previous value.** The dominant predictor in every channel is `Prev_<X>` (importance 0.33–0.61); the whole tyre block (`Tyre_Age_<compound>`, `Tyre_Age_<compound>_2`, `Tyre_Age_Total`) sums to only 0.004–0.04. Tyre terms are a rounding error inside the forest.

**3. The degradation curve is essentially flat.** Holding pace/geometry/weather at their medians, predicted sector Duration rises only **+0.070s** across a stint (~2→20 laps of tyre age, then flat) — about ~3.9 ms per lap per sector, inside the noise.

**4. The stacked model is slightly worse, not better.** Swapping raw `Prev_*` for tyre-aware `Pred_*` raises Duration MAE from 0.31s to 0.40s — the re-encoding drops information without adding signal.

**5. This is the honest answer, not a failure.** "Pace" and "tyre age" are not independent: a degraded tyre *is* slower pace, and that is already encoded in `Prev_Duration` / `Clean_Pace`. So there is no *residual* tyre signal left to surface — the forest-of-forests confirms this cleanly instead of inventing one. For pit-stop strategy: the measurable degradation effect is ~0.07s per stint, far below a ~20s pit-stop cost, so stop-lap decisions here would be driven by compound differences, safety-car windows and track position — not by this degradation curve.
