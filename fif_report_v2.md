# Forest-of-forests v2 — tyre-aware telemetry → Duration

Generated: 2026-11-17  |  source: `fforest_results.json`

## Stage 1 — 8 telemetry forests (per-channel feature engineering)

Each channel predicts its current value from its previous value + tyre + weather + traffic + Lap + Track_Status + channel-specific geometry. Winning variant (level / delta / log) chosen by held-out-2024 MAE. "Beat persistence" = the forest predicts current better than the naive `current = previous` baseline.

| Channel | variant | beat persist. | persist. MAE | RF MAE | RF R² | SGD MAE | Σ tyre imp. |
|---|---|---|---|---|---|---|---|
| Avg speed (corners) | delta_slog | ✓ | 6.0487 | 5.2051 | 0.887 | 13.5608 | 0.18011 |
| Avg speed (straight) | delta_slog | ✓ | 4.3523 | 3.7313 | 0.9274 | 17.6184 | 0.14081 |
| Full throttle % | delta_slog | ✓ | 3.6206 | 3.1917 | 0.911 | 8.2173 | 0.14128 |
| Brake % | delta | ✓ | 2.3561 | 2.1483 | 0.7815 | 2.2789 | 0.14733 |
| DRS % | delta_slog | ✓ | 2.4637 | 2.1495 | 0.5674 | 15.1886 | 0.05314 |
| Max speed | delta_slog | ✓ | 4.3028 | 3.9217 | 0.8571 | 14.4773 | 0.1286 |
| Min speed | delta_slog | ✓ | 4.1522 | 2.8062 | 0.9585 | 14.2133 | 0.19302 |
| Avg gear | delta_slog | ✓ | 0.1439 | 0.1263 | 0.878 | 0.2643 | 0.15765 |

Channels beating persistence: **8/8**.

## Stage 1 — full feature importance per forest

### Avg speed (corners) — variant `delta_slog`

| rank | feature | importance |
|---|---|---|
| 1 | `Prev_Avg_Speed_Corners_roll5_slope` | 0.35232 |
| 2 | `Tyre_Age_Total` | 0.09031 |
| 3 | `Prev_Avg_Speed_Corners_roll5_mean` | 0.08373 |
| 4 | `Interval_Ahead` | 0.06531 |
| 5 | `Traffic_Within_5s` | 0.05255 |
| 6 | `Lap` | 0.04527 |
| 7 | `Track_Temperature` | 0.04183 |
| 8 | `Air_Temperature` | 0.03519 |
| 9 | `Pressure` | 0.02848 |
| 10 | `Humidity` | 0.02683 |
| 11 | `Tyre_Age_HARD_2` | 0.02285 |
| 12 | `Tyre_Age_HARD` | 0.02228 |
| 13 | `Traffic_Within_2s` | 0.01903 |
| 14 | `Tyre_Age_MEDIUM_2` | 0.01579 |
| 15 | `Tyre_Age_MEDIUM` | 0.01578 |
| 16 | `Num_Corner_6` | 0.01069 |
| 17 | `Num_Corner_1` | 0.00941 |
| 18 | `Traffic_Within_1s` | 0.00850 |
| 19 | `Num_Corner_0` | 0.00832 |
| 20 | `Num_Corner_2` | 0.00760 |
| 21 | `Num_Corner_4` | 0.00709 |
| 22 | `Num_Corner_7` | 0.00599 |
| 23 | `Tyre_Age_SOFT_2` | 0.00579 |
| 24 | `Tyre_Age_SOFT` | 0.00575 |
| 25 | `Num_Corner_3` | 0.00530 |
| 26 | `missingindicator_Prev_Avg_Speed_Corners_roll5_slope` | 0.00261 |
| 27 | `Num_Corner_5` | 0.00246 |
| 28 | `Tyre_Age_INTERMEDIATE` | 0.00079 |
| 29 | `Tyre_Age_INTERMEDIATE_2` | 0.00077 |
| 30 | `Track_Status_GREEN` | 0.00061 |
| 31 | `Track_Status_SC` | 0.00038 |
| 32 | `Rainfall` | 0.00036 |
| 33 | `Track_Status_VSC` | 0.00000 |
| 34 | `Track_Status_VSC_ENDING` | 0.00000 |
| 35 | `Track_Status_SC_ENDING` | 0.00000 |
| 36 | `Tyre_Age_WET_2` | 0.00000 |
| 37 | `Tyre_Age_WET` | 0.00000 |

### Avg speed (straight) — variant `delta_slog`

| rank | feature | importance |
|---|---|---|
| 1 | `Prev_Avg_Speed_Straight_roll5_slope` | 0.38399 |
| 2 | `Prev_Avg_Speed_Straight_roll5_mean` | 0.09277 |
| 3 | `Interval_Ahead` | 0.06950 |
| 4 | `Tyre_Age_Total` | 0.06219 |
| 5 | `Lap` | 0.04368 |
| 6 | `Traffic_Within_5s` | 0.04256 |
| 7 | `Track_Temperature` | 0.03963 |
| 8 | `Air_Temperature` | 0.03230 |
| 9 | `Straight_Ratio` | 0.02779 |
| 10 | `Pressure` | 0.02659 |
| 11 | `Humidity` | 0.02599 |
| 12 | `Log_Total_Straight_Length` | 0.02224 |
| 13 | `Total_Straight_Length` | 0.02161 |
| 14 | `Tyre_Age_HARD_2` | 0.02064 |
| 15 | `Tyre_Age_HARD` | 0.02031 |
| 16 | `Traffic_Within_2s` | 0.01868 |
| 17 | `Tyre_Age_MEDIUM_2` | 0.01378 |
| 18 | `Tyre_Age_MEDIUM` | 0.01360 |
| 19 | `Traffic_Within_1s` | 0.00885 |
| 20 | `Tyre_Age_SOFT` | 0.00435 |
| 21 | `Tyre_Age_SOFT_2` | 0.00430 |
| 22 | `missingindicator_Prev_Avg_Speed_Straight_roll5_slope` | 0.00145 |
| 23 | `Tyre_Age_INTERMEDIATE` | 0.00083 |
| 24 | `Tyre_Age_INTERMEDIATE_2` | 0.00081 |
| 25 | `Track_Status_GREEN` | 0.00062 |
| 26 | `Track_Status_SC` | 0.00054 |
| 27 | `Rainfall` | 0.00039 |
| 28 | `Track_Status_VSC_ENDING` | 0.00002 |
| 29 | `Track_Status_VSC` | 0.00001 |
| 30 | `Tyre_Age_WET_2` | 0.00000 |
| 31 | `Track_Status_SC_ENDING` | 0.00000 |
| 32 | `Tyre_Age_WET` | 0.00000 |

### Full throttle % — variant `delta_slog`

| rank | feature | importance |
|---|---|---|
| 1 | `Prev_Full_Throttle_Pct_roll5_slope` | 0.37626 |
| 2 | `Prev_Full_Throttle_Pct_roll5_mean` | 0.08109 |
| 3 | `Interval_Ahead` | 0.06427 |
| 4 | `Tyre_Age_Total` | 0.06128 |
| 5 | `Lap` | 0.04652 |
| 6 | `Traffic_Within_5s` | 0.04480 |
| 7 | `Track_Temperature` | 0.03934 |
| 8 | `Air_Temperature` | 0.03231 |
| 9 | `Straight_Ratio` | 0.02398 |
| 10 | `Humidity` | 0.02323 |
| 11 | `Pressure` | 0.02259 |
| 12 | `Tyre_Age_HARD` | 0.02099 |
| 13 | `Tyre_Age_HARD_2` | 0.02041 |
| 14 | `Total_Sector_Length` | 0.01998 |
| 15 | `Total_Straight_Length` | 0.01948 |
| 16 | `Traffic_Within_2s` | 0.01758 |
| 17 | `Tyre_Age_MEDIUM_2` | 0.01372 |
| 18 | `Tyre_Age_MEDIUM` | 0.01274 |
| 19 | `Traffic_Within_1s` | 0.00840 |
| 20 | `Num_Corner_1` | 0.00617 |
| 21 | `Tyre_Age_SOFT_2` | 0.00524 |
| 22 | `Num_Corner_2` | 0.00509 |
| 23 | `Tyre_Age_SOFT` | 0.00504 |
| 24 | `Num_Corner_6` | 0.00504 |
| 25 | `Num_Corner_0` | 0.00496 |
| 26 | `Num_Corner_4` | 0.00376 |
| 27 | `Num_Corner_7` | 0.00359 |
| 28 | `missingindicator_Prev_Full_Throttle_Pct_roll5_slope` | 0.00354 |
| 29 | `Num_Corner_3` | 0.00353 |
| 30 | `Num_Corner_5` | 0.00168 |
| 31 | `Tyre_Age_INTERMEDIATE_2` | 0.00094 |
| 32 | `Tyre_Age_INTERMEDIATE` | 0.00093 |
| 33 | `Track_Status_GREEN` | 0.00054 |
| 34 | `Track_Status_SC` | 0.00051 |
| 35 | `Rainfall` | 0.00042 |
| 36 | `Track_Status_VSC` | 0.00006 |
| 37 | `Track_Status_VSC_ENDING` | 0.00002 |
| 38 | `Track_Status_SC_ENDING` | 0.00000 |
| 39 | `Tyre_Age_WET` | 0.00000 |
| 40 | `Tyre_Age_WET_2` | 0.00000 |

### Brake % — variant `delta`

| rank | feature | importance |
|---|---|---|
| 1 | `Prev_Brake_Pct_roll5_slope` | 0.40853 |
| 2 | `Prev_Brake_Pct_roll5_mean` | 0.08399 |
| 3 | `Tyre_Age_Total` | 0.06877 |
| 4 | `Interval_Ahead` | 0.06632 |
| 5 | `Lap` | 0.04413 |
| 6 | `Track_Temperature` | 0.04243 |
| 7 | `Air_Temperature` | 0.03565 |
| 8 | `Traffic_Within_5s` | 0.03019 |
| 9 | `Pressure` | 0.02763 |
| 10 | `Humidity` | 0.02647 |
| 11 | `Tyre_Age_HARD_2` | 0.02038 |
| 12 | `Tyre_Age_HARD` | 0.01939 |
| 13 | `Traffic_Within_2s` | 0.01630 |
| 14 | `Tyre_Age_MEDIUM_2` | 0.01341 |
| 15 | `Tyre_Age_MEDIUM` | 0.01319 |
| 16 | `Num_Corner_0` | 0.01026 |
| 17 | `Num_Corner_1` | 0.01007 |
| 18 | `Traffic_Within_1s` | 0.00892 |
| 19 | `Num_Corner_6` | 0.00885 |
| 20 | `Num_Corner_2` | 0.00835 |
| 21 | `Num_Corner_7` | 0.00699 |
| 22 | `Tyre_Age_SOFT_2` | 0.00599 |
| 23 | `Num_Corner_4` | 0.00579 |
| 24 | `Num_Corner_3` | 0.00512 |
| 25 | `Tyre_Age_SOFT` | 0.00503 |
| 26 | `Num_Corner_5` | 0.00244 |
| 27 | `missingindicator_Prev_Brake_Pct_roll5_slope` | 0.00235 |
| 28 | `Track_Status_GREEN` | 0.00071 |
| 29 | `Track_Status_SC` | 0.00069 |
| 30 | `Tyre_Age_INTERMEDIATE` | 0.00059 |
| 31 | `Tyre_Age_INTERMEDIATE_2` | 0.00057 |
| 32 | `Rainfall` | 0.00041 |
| 33 | `Track_Status_VSC` | 0.00007 |
| 34 | `Track_Status_VSC_ENDING` | 0.00000 |
| 35 | `Track_Status_SC_ENDING` | 0.00000 |
| 36 | `Tyre_Age_WET_2` | 0.00000 |
| 37 | `Tyre_Age_WET` | 0.00000 |

### DRS % — variant `delta_slog`

| rank | feature | importance |
|---|---|---|
| 1 | `Prev_DRS_Pct_roll5_slope` | 0.30499 |
| 2 | `Interval_Ahead` | 0.21244 |
| 3 | `Prev_DRS_Pct_roll5_mean` | 0.12198 |
| 4 | `Straight_Ratio` | 0.05321 |
| 5 | `Lap` | 0.04449 |
| 6 | `Total_Straight_Length` | 0.03543 |
| 7 | `Traffic_Within_1s` | 0.02891 |
| 8 | `Pressure` | 0.02654 |
| 9 | `Track_Temperature` | 0.02539 |
| 10 | `Air_Temperature` | 0.02455 |
| 11 | `Traffic_Within_5s` | 0.02369 |
| 12 | `Humidity` | 0.02202 |
| 13 | `Traffic_Within_2s` | 0.01823 |
| 14 | `Tyre_Age_Total` | 0.01710 |
| 15 | `Tyre_Age_HARD` | 0.00799 |
| 16 | `Tyre_Age_HARD_2` | 0.00784 |
| 17 | `Tyre_Age_MEDIUM` | 0.00690 |
| 18 | `Tyre_Age_MEDIUM_2` | 0.00646 |
| 19 | `missingindicator_Prev_DRS_Pct_roll5_slope` | 0.00327 |
| 20 | `Tyre_Age_SOFT_2` | 0.00267 |
| 21 | `Tyre_Age_SOFT` | 0.00262 |
| 22 | `Tyre_Age_INTERMEDIATE_2` | 0.00092 |
| 23 | `Track_Status_SC` | 0.00066 |
| 24 | `Track_Status_GREEN` | 0.00064 |
| 25 | `Tyre_Age_INTERMEDIATE` | 0.00064 |
| 26 | `Rainfall` | 0.00020 |
| 27 | `Track_Status_VSC` | 0.00019 |
| 28 | `Tyre_Age_WET` | 0.00000 |
| 29 | `Track_Status_VSC_ENDING` | 0.00000 |
| 30 | `Track_Status_SC_ENDING` | 0.00000 |
| 31 | `Tyre_Age_WET_2` | 0.00000 |

### Max speed — variant `delta_slog`

| rank | feature | importance |
|---|---|---|
| 1 | `Prev_Max_Speed_roll5_slope` | 0.28082 |
| 2 | `Prev_Max_Speed_roll5_mean` | 0.12757 |
| 3 | `Interval_Ahead` | 0.12034 |
| 4 | `Tyre_Age_Total` | 0.05152 |
| 5 | `Lap` | 0.05085 |
| 6 | `Track_Temperature` | 0.05012 |
| 7 | `Air_Temperature` | 0.04090 |
| 8 | `Traffic_Within_5s` | 0.03619 |
| 9 | `Log_Total_Straight_Length` | 0.03476 |
| 10 | `Total_Straight_Length` | 0.03432 |
| 11 | `Pressure` | 0.03145 |
| 12 | `Humidity` | 0.03118 |
| 13 | `Tyre_Age_HARD_2` | 0.01943 |
| 14 | `Tyre_Age_HARD` | 0.01896 |
| 15 | `Traffic_Within_2s` | 0.01895 |
| 16 | `Tyre_Age_MEDIUM_2` | 0.01391 |
| 17 | `Tyre_Age_MEDIUM` | 0.01355 |
| 18 | `Traffic_Within_1s` | 0.01079 |
| 19 | `Tyre_Age_SOFT_2` | 0.00527 |
| 20 | `Tyre_Age_SOFT` | 0.00515 |
| 21 | `Track_Status_SC` | 0.00101 |
| 22 | `Track_Status_GREEN` | 0.00088 |
| 23 | `missingindicator_Prev_Max_Speed_roll5_slope` | 0.00078 |
| 24 | `Tyre_Age_INTERMEDIATE_2` | 0.00047 |
| 25 | `Tyre_Age_INTERMEDIATE` | 0.00034 |
| 26 | `Rainfall` | 0.00032 |
| 27 | `Track_Status_VSC_ENDING` | 0.00013 |
| 28 | `Track_Status_VSC` | 0.00002 |
| 29 | `Track_Status_SC_ENDING` | 0.00000 |
| 30 | `Tyre_Age_WET` | 0.00000 |
| 31 | `Tyre_Age_WET_2` | 0.00000 |

### Min speed — variant `delta_slog`

| rank | feature | importance |
|---|---|---|
| 1 | `Prev_Min_Speed_roll5_slope` | 0.35459 |
| 2 | `Prev_Min_Speed_roll5_mean` | 0.12147 |
| 3 | `Tyre_Age_Total` | 0.11663 |
| 4 | `Interval_Ahead` | 0.06264 |
| 5 | `Traffic_Within_5s` | 0.05119 |
| 6 | `Lap` | 0.04094 |
| 7 | `Track_Temperature` | 0.03292 |
| 8 | `Air_Temperature` | 0.03091 |
| 9 | `Humidity` | 0.02338 |
| 10 | `Pressure` | 0.02108 |
| 11 | `Tyre_Age_HARD` | 0.01854 |
| 12 | `Tyre_Age_HARD_2` | 0.01840 |
| 13 | `Traffic_Within_2s` | 0.01551 |
| 14 | `Tyre_Age_MEDIUM_2` | 0.01423 |
| 15 | `Tyre_Age_MEDIUM` | 0.01260 |
| 16 | `Traffic_Within_1s` | 0.00805 |
| 17 | `Num_Corner_1` | 0.00751 |
| 18 | `Num_Corner_2` | 0.00652 |
| 19 | `Num_Corner_0` | 0.00621 |
| 20 | `Num_Corner_6` | 0.00601 |
| 21 | `Tyre_Age_SOFT_2` | 0.00547 |
| 22 | `Tyre_Age_SOFT` | 0.00524 |
| 23 | `Num_Corner_4` | 0.00511 |
| 24 | `Num_Corner_7` | 0.00466 |
| 25 | `Num_Corner_3` | 0.00376 |
| 26 | `Num_Corner_5` | 0.00202 |
| 27 | `missingindicator_Prev_Min_Speed_roll5_slope` | 0.00143 |
| 28 | `Tyre_Age_INTERMEDIATE` | 0.00103 |
| 29 | `Tyre_Age_INTERMEDIATE_2` | 0.00086 |
| 30 | `Track_Status_SC` | 0.00044 |
| 31 | `Track_Status_GREEN` | 0.00039 |
| 32 | `Rainfall` | 0.00021 |
| 33 | `Track_Status_VSC` | 0.00001 |
| 34 | `Track_Status_VSC_ENDING` | 0.00000 |
| 35 | `Track_Status_SC_ENDING` | 0.00000 |
| 36 | `Tyre_Age_WET_2` | 0.00000 |
| 37 | `Tyre_Age_WET` | 0.00000 |

### Avg gear — variant `delta_slog`

| rank | feature | importance |
|---|---|---|
| 1 | `Prev_Avg_Gear_roll5_slope` | 0.42885 |
| 2 | `Tyre_Age_Total` | 0.09557 |
| 3 | `Prev_Avg_Gear_roll5_mean` | 0.07133 |
| 4 | `Traffic_Within_5s` | 0.05478 |
| 5 | `Lap` | 0.05082 |
| 6 | `Interval_Ahead` | 0.03633 |
| 7 | `Track_Temperature` | 0.02929 |
| 8 | `Straight_Ratio` | 0.02662 |
| 9 | `Air_Temperature` | 0.02596 |
| 10 | `Total_Straight_Length` | 0.02234 |
| 11 | `Pressure` | 0.01830 |
| 12 | `Traffic_Within_2s` | 0.01702 |
| 13 | `Humidity` | 0.01622 |
| 14 | `Tyre_Age_HARD_2` | 0.01536 |
| 15 | `Tyre_Age_HARD` | 0.01373 |
| 16 | `Tyre_Age_MEDIUM` | 0.01075 |
| 17 | `Tyre_Age_MEDIUM_2` | 0.01066 |
| 18 | `missingindicator_Prev_Avg_Gear_roll5_slope` | 0.00697 |
| 19 | `Traffic_Within_1s` | 0.00649 |
| 20 | `Num_Corner_1` | 0.00524 |
| 21 | `Tyre_Age_SOFT_2` | 0.00493 |
| 22 | `Num_Corner_0` | 0.00493 |
| 23 | `Tyre_Age_SOFT` | 0.00462 |
| 24 | `Num_Corner_6` | 0.00421 |
| 25 | `Num_Corner_2` | 0.00390 |
| 26 | `Num_Corner_4` | 0.00356 |
| 27 | `Num_Corner_7` | 0.00348 |
| 28 | `Num_Corner_3` | 0.00264 |
| 29 | `Num_Corner_5` | 0.00158 |
| 30 | `Tyre_Age_INTERMEDIATE` | 0.00111 |
| 31 | `Tyre_Age_INTERMEDIATE_2` | 0.00092 |
| 32 | `Track_Status_SC` | 0.00059 |
| 33 | `Track_Status_GREEN` | 0.00051 |
| 34 | `Rainfall` | 0.00038 |
| 35 | `Track_Status_VSC` | 0.00001 |
| 36 | `Track_Status_VSC_ENDING` | 0.00001 |
| 37 | `Track_Status_SC_ENDING` | 0.00000 |
| 38 | `Tyre_Age_WET_2` | 0.00000 |
| 39 | `Tyre_Age_WET` | 0.00000 |

## Degradation curve (delta vs Tyre_Age_Total)

(no degradation curve)

## Stage 2 — main forest (target = delta)

The main forest predicts the **delta** `Log_Duration − log(Prev_Duration)` (the lap-over-lap log ratio), then reconstructs `Duration = Prev_Duration × exp(delta)`. Two telemetry feature sets are compared — raw `Prev_*` vs tyre-aware `Pred_*` — each with RF, SGD, KNN, XGBoost and CatBoost.

MAE in **seconds** on the 2025 test season.

| variant | RF | SGD | KNN | XGB | CatBoost |
|---|---|---|---|---|---|
| raw Prev_* | 0.2992 / R²=0.9865 | 0.5198 / R²=0.93 | 0.4464 / R²=0.9654 | 0.3457 / R²=0.9873 | 0.3523 / R²=0.9887 |
| Pred_* | 0.3224 / R²=0.9849 | 0.5205 / R²=0.9307 | 0.4792 / R²=0.9578 | 0.4383 / R²=0.981 | 0.3997 / R²=0.9862 |

Each cell = MAE (s) / R² (on reconstructed Duration, 2025 test).
### 6-condition MAE breakdown (raw Prev_*, RF)

| condition | n | MAE (s) |
|---|---|---|
| overall | 70410 | 0.2992 |
| clean_no_sc_vsc | 70004 | 0.2815 |
| dry | 69575 | 0.289 |
| wet | 835 | 1.1535 |
| dry_clean | 69200 | 0.2729 |
| wet_clean | 804 | 1.0196 |

### Stage 2 feature importance — raw Prev_*

| rank | feature | importance |
|---|---|---|
| 1 | `Prev_Duration_roll5_slope` | 0.34775 |
| 2 | `Prev_Avg_Speed_Corners` | 0.22334 |
| 3 | `Prev_Min_Speed` | 0.13158 |
| 4 | `Prev_Duration` | 0.06022 |
| 5 | `Prev_Avg_Gear` | 0.05806 |
| 6 | `Prev_St_Speed` | 0.03330 |
| 7 | `Prev_Max_Speed` | 0.01491 |
| 8 | `Best_Duration_So_Far` | 0.01265 |
| 9 | `Traffic_Within_5s` | 0.01104 |
| 10 | `Total_Distance_Driven` | 0.01060 |
| 11 | `Clean_Pace` | 0.01052 |
| 12 | `Prev_Duration_roll5_mean` | 0.01025 |
| 13 | `Distance_On_Tyre` | 0.00977 |
| 14 | `Total_Sector_Length` | 0.00839 |
| 15 | `Prev_Full_Throttle_Pct` | 0.00692 |
| 16 | `Total_Straight_Length` | 0.00587 |
| 17 | `Track_Status_Inferred` | 0.00523 |
| 18 | `Prev_Avg_Speed_Straight` | 0.00505 |
| 19 | `Prev_I1_Speed` | 0.00421 |
| 20 | `Prev_I2_Speed` | 0.00356 |
| 21 | `Pressure` | 0.00353 |
| 22 | `Prev_Sector_Duration` | 0.00340 |
| 23 | `Humidity` | 0.00315 |
| 24 | `Track_Temperature` | 0.00289 |
| 25 | `Interval_Ahead` | 0.00203 |
| 26 | `Air_Temperature` | 0.00186 |
| 27 | `Track_Status_GREEN` | 0.00165 |
| 28 | `Track_Status_SC` | 0.00119 |
| 29 | `Prev_Brake_Pct` | 0.00117 |
| 30 | `Num_Corner_0` | 0.00080 |

### Stage 2 feature importance — Pred_*

| rank | feature | importance |
|---|---|---|
| 1 | `Prev_Duration_roll5_slope` | 0.47229 |
| 2 | `Prev_Duration` | 0.17702 |
| 3 | `Pred_Outcome_Avg_Gear` | 0.06583 |
| 4 | `Pred_Outcome_Avg_Speed_Corners` | 0.03770 |
| 5 | `Prev_St_Speed` | 0.03058 |
| 6 | `Distance_On_Tyre` | 0.02704 |
| 7 | `Best_Duration_So_Far` | 0.02525 |
| 8 | `Traffic_Within_5s` | 0.01881 |
| 9 | `Clean_Pace` | 0.01876 |
| 10 | `Total_Sector_Length` | 0.01416 |
| 11 | `Total_Distance_Driven` | 0.01334 |
| 12 | `Pred_Outcome_Max_Speed` | 0.01236 |
| 13 | `Prev_Duration_roll5_mean` | 0.01040 |
| 14 | `Pred_Outcome_Min_Speed` | 0.00778 |
| 15 | `Prev_I1_Speed` | 0.00592 |
| 16 | `Prev_I2_Speed` | 0.00526 |
| 17 | `Pred_Outcome_Avg_Speed_Straight` | 0.00519 |
| 18 | `Total_Straight_Length` | 0.00479 |
| 19 | `Pressure` | 0.00455 |
| 20 | `Prev_Sector_Duration` | 0.00423 |
| 21 | `Track_Temperature` | 0.00403 |
| 22 | `Track_Status_Inferred` | 0.00381 |
| 23 | `Track_Status_GREEN` | 0.00315 |
| 24 | `Humidity` | 0.00297 |
| 25 | `Pred_Outcome_Full_Throttle_Pct` | 0.00297 |
| 26 | `Air_Temperature` | 0.00270 |
| 27 | `Track_Status_SC` | 0.00240 |
| 28 | `Num_Corner_4` | 0.00203 |
| 29 | `Pred_Outcome_Brake_Pct` | 0.00168 |
| 30 | `Traffic_Within_2s` | 0.00166 |

## Interpretation

- **Gate result:** 8/8 sub-forests beat the persistence baseline. Channels that did not beat persistence keep their raw `Prev_*` in the main forest (feeding a worse `Pred_*` would only hurt).
- **Prev_* vs Pred_* (delta target):** raw `Prev_*` telemetry vs tyre-aware `Pred_*`, each with RF/SGD/KNN. If `Pred_*` ≈ `Prev_*`, the sub-forests add no signal for the delta; if KNN ≈ RF, the signal is local/linear.
- **Degradation:** the delta curve's slope vs `Tyre_Age_Total` is the degradation rate a pit-stop optimizer would use.