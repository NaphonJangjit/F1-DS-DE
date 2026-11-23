# Data quality report

Source: `data_<year>.parquet` (built lap-sector tables).

| check | 2023 | 2024 | 2025 |
|---|---|---|---|
| rows | 71,395 | 77,918 | 76,952 |
| sessions / drivers | 22 / 22 | 24 / 25 | 24 / 21 |
| 1. columns with missing values | 2 | 1 | 4 |
|    worst column (% missing) | ('Prev_Avg_Speed_Straight', 0.39) | ('Prev_Avg_Speed_Straight', 0.38) | ('Prev_I2_Speed', 3.38) |
| 2. duplicate keys / rows | 0 / 0 | 0 / 0 | 0 / 0 |
| 3. out-of-range values | 9 | 4 | 1 |
|    SC/VSC rows (median Duration/Clean_Pace) | 8,403 (1.003) | 2,699 (1.299) | 4,122 (1.351) |
|    sessions labelled SC at full race pace | [7787, 9181, 9205] | - | - |
| 4. outliers IQR high / low | 5,351 / 108 | 5,003 / 109 | 5,437 / 125 |
|    removed by MAD rule (median + 4 MAD) | 6,195 | 4,900 | 6,511 |
| 5. Num_Corner_* columns | 9 | 11 | 8 |

Corner-cluster consistency: of 63 circuit-sectors that appear in 2+ years, only 0 keep the same `Num_Corner_*` profile -> cluster ids are re-fitted per year and are not comparable across years.

## 2023

- missing (%): {'Prev_Avg_Speed_Straight': 0.39, 'Prev_St_Speed': 0.0}
- out of range: {'Prev_Avg_Gear': 9}
- Tyre values: ['HARD', 'INTERMEDIATE', 'MEDIUM', 'SOFT', 'WET']
- Track_Status values: ['GREEN', 'SC', 'SC_ENDING', 'VSC', 'VSC_ENDING']

## 2024

- missing (%): {'Prev_Avg_Speed_Straight': 0.38}
- out of range: {'Prev_Avg_Gear': 4}
- Tyre values: ['HARD', 'INTERMEDIATE', 'MEDIUM', 'SOFT', 'WET']
- Track_Status values: ['GREEN', 'SC', 'SC_ENDING', 'VSC', 'VSC_ENDING']

## 2025

- missing (%): {'Prev_I2_Speed': 3.38, 'Prev_Avg_Speed_Straight': 0.08, 'Prev_St_Speed': 0.02, 'Prev_I1_Speed': 0.02}
- out of range: {'Prev_Avg_Gear': 1}
- Tyre values: ['HARD', 'INTERMEDIATE', 'MEDIUM', 'SOFT']
- Track_Status values: ['GREEN', 'SC', 'SC_ENDING', 'VSC', 'VSC_ENDING']
