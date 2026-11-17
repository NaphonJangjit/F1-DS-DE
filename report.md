# f1pit — Experiment Results Report

> Consolidated results across every experiment this session. Updated 2026-11-13.

## 0. Setup (held fixed everywhere)

- **Target:** `Log_Duration = ln(Duration)`; error reported in **seconds** (`exp` back).
- **Split:** train = 2023 + 2024, test = 2025 (temporal, no leakage).
- **Frozen RF config (reference):** `RandomForestRegressor(n_estimators=150, max_features=0.3, min_samples_leaf=2, n_jobs=6, random_state=42)`.
- **Preprocessing:** one-hot `Track_Status`, median-impute numerics, tyre-age×length interactions, `Prev_*` trend features; **outlier filter ON** (`median + 4·MAD`, removes slow SC/VSC/accident laps) unless stated.
- **Feature count:** 72 (base) — 73–74 with rainfall-transition extras.
- **Data sizes:** train ≈ 138k rows, test ≈ 70k rows (2025).

---

## 1. Headline leaderboard — 2025 test, MAE (seconds)

| Model | full MAE | green MAE* | vs RF |
|---|---|---|---|
| **Random Forest (frozen)** | **0.2649** | **0.2439** | — (best) |
| RF + inverse-freq upsampling | 0.2649 | — | ±0 |
| RF, outlier filter **OFF** | 0.843 | — | 3.2× worse |
| RF, no filter + SMOTE | 0.833 | — | 3.1× worse |
| RF, 18-GP-2024 warm pool | 0.2852 | 0.2641 | +0.02 (less data) |
| EBM (single, main effects) | 0.5787 | 0.5648 | 2.2× worse |
| LinearRegression | 0.9185 | ≈0.904 | 3.5× worse |
| Ridge (α=1) | 0.9185 | ≈0.904 | 3.5× worse |
| LinearRegression + upsampling | 0.9486 | — | worse than base |
| DNN deep `[512,256,128]` | 1.2581 | 1.2267 | 4.8× worse |
| DNN deeper `[512,256,128,64]` | 2.5206 | 2.5057 | 9.5× worse |

*\* "green" = SC/VSC + transition laps masked from the prepared test (0.2439). A cleaner
green-only pipeline that drops SC/VSC **before** the outlier filter gives RF **0.2144**.*

**Takeaway:** every alternative model family — linear, EBM, deep net — is **2× to 10× worse**
than the frozen Random Forest. More capacity or "interpretability" does not help.

---

## 2. Regime breakdown — where the error actually lives (RF, full 2025 test)

| Regime | MAE (s) | share of test |
|---|---|---|
| Clean / dry | 0.242 | ~94% |
| Wet | 1.179 | ~1.2% |
| Transition (drying/worsening) | 0.399 | ~0.3% |
| SC | 1.998 | small |
| VSC | 3.867 | small |

The overall 0.265 is ~94% clean laps at 0.24 s. The error is concentrated in **wet (1.18 s,
~5× dry) and SC/VSC (2–4 s)** — exactly the regimes that are *unforecastable*, not *under-trained*.

---

## 3. Experiment-by-experiment

### 3.1 Concept drift + machine unlearning → **no drift; unlearning would only hurt**

- Full warm (2023+24): static MAE = adaptive MAE = **0.2649**, **0 retrains**.
- Shorter warm (2023 + first 18 GPs of 2024): static = unlearn = keep-all = **0.2852**, **0 drift events**.
- Per-race error is flat (~0.17–0.27 s) across the whole season; the only spikes are isolated
  wet/street races (Australia 0.97, Britain 1.31, Azerbaijan 0.49) — **point events, not a level shift**.
- Dropping data already costs: 0.2852 (18-GP warm) vs 0.2649 (full warm). "Unlearning old data"
  removes 22 races of signal for no benefit.

**Conclusion:** no concept drift in 2025; the static 2023–24 model generalizes flatly. The
drift/unlearn machinery solves a problem the data doesn't have.

### 3.2 Data-bias corrections (upsampling / SMOTE / bring-outliers-back) → **all fail**

| Experiment | RF MAE | verdict |
|---|---|---|
| Baseline (outlier filter ON) | 0.2649 | reference |
| Inverse-frequency `sample_weight` upsampling (wet/SC/VSC → 10%) | 0.2649 | no change; linear got *worse* (0.9185→0.9486) |
| Outlier filter OFF ("bring outliers back") | 0.843 | 3.2× worse |
| Outlier OFF + SMOTE (synthetic wet/SC/VSC) | 0.833 | tiny gain on top of a bad base |

**Conclusion:** SC/VSC/wet are rare because they're *unpredictable*, not because the model
hasn't seen enough of them. The outlier filter is load-bearing (removes ~93% SC / ~84% VSC)
and cannot be replaced by upsampling or synthetic data.

### 3.3 Explicit SC/VSC removal (green-only training) → **redundant**

- Baseline on green-only test: **0.2144**.
- Train on green-only data: 0.2178 (no improvement).
- Green-only with outlier filter OFF: 0.3311 (worse).

**Conclusion:** the duration-based outlier filter already removes the SC/VSC laps; removing
them explicitly from training changes nothing. SC/VSC belong in the *evaluation target*
(green-only), not in a separate training scheme.

### 3.4 Deep neural network → **5–10× worse; data is not "deep-complex"**

| MLP architecture | full | green | val (train yrs) |
|---|---|---|---|
| `[128]` | 2.633 | 2.613 | 0.516 |
| `[256,128]` | 1.986 | 1.966 | 0.549 |
| `[512,256,128]` | **1.258** | 1.227 | 0.486 |
| `[512,256,128,64]` | 2.521 | 2.506 | 0.491 |

- Best DNN is 4.8× worse than RF, and even worse than linear regression.
- **Depth is non-monotonic** (deeper ≠ better) → no smooth high-order structure to exploit.
- DNN underfits even in-sample: best val MAE (0.49) ≈ 2× RF's *test* MAE.
- Huge train→test gap (val 0.49 → test 1.26) → it fits 2023–24 noise that doesn't transfer.

**Conclusion:** the residual after RF is noise/variance (wet, traffic, street chaos), not
untapped non-linear structure.

### 3.5 EBM / interpretable model → **2.2× worse, and very slow**

- Single EBM (main-effects only, `max_bins=64`, early stopping): full **0.5787**, green 0.5648.
- Fit time ≈ 86 min vs RF ≈ 1 min — impractical at this data size.

### 3.6 t-test — why Baku / Spa / Monaco are hard (green-only, Welch + Cohen's d)

Hard = {Baku, Spa, Monaco}; Easy = {Imola, Las Vegas, Jeddah, Spielberg}. n≈9k/group → read
effect size, not p-value.

| Variable | hard | easy | Cohen's d | |
|---|---|---|---|---|
| Air_Temperature | 20.5 °C | 28.9 °C | −4.15 | much colder |
| Humidity | 62.4% | 44.8% | +1.45 | more humid |
| Prev_Avg_Speed_Corners | 137 km/h | 176 km/h | −1.32 | slower corners |
| Prev_Avg_Gear | 4.76 | 5.89 | −1.14 | tighter/lower gear |
| Prev_Brake_Pct | 23.2% | 16.7% | +0.86 | heavier braking |
| Prev_Full_Throttle_Pct | 44.9% | 58.1% | −0.69 | less flat-out |

And the smoking gun — **target variance**:

| Track | Duration std (s) |
|---|---|
| Spa | 11.07 |
| Monaco | 7.75 |
| Baku | 7.58 |
| … | … |
| Imola | 1.95 |

**Conclusion:** Baku/Monaco = slow tight street circuits (heavy braking, traffic); Spa = long
lap + micro-climate the binary `Rainfall` flag can't see; all three = cold/humid. Their common
trait is **high intrinsic lap-time variance**, which no model can predict away.

---

## 4. Track error ranking (2025, RF full-warm, MAE seconds)

| # | Race | MAE | cause |
|---|---|---|---|
| 1 | United Kingdom | 1.307 | wet |
| 2 | Australia | 0.967 | wet |
| 3 | Azerbaijan | 0.494 | street / chaos |
| 4 | Monaco | 0.323 | street / traffic |
| 5 | Belgium (Spa) | 0.303 | weather variance |
| 6 | Brazil | 0.274 | mixed |
| 7 | United States | 0.242 | |
| 8 | Saudi Arabia | 0.231 | |

Dry, permanent circuits sit at 0.14–0.23 s; only wet/street races exceed 0.30.

---

## 5. Conclusions

1. **Random Forest is the ceiling here** — 0.265 full / 0.214–0.244 green. Linear, EBM, and
   DNN are all 2–10× worse. The data is noisy tabular data with sharp local structure that
   trees capture and smooth/global models average away.
2. **No concept drift** (both warm-window configs, 0 events). Drop the unlearn/retrain machinery.
3. **No data-bias fix works** — upsampling, SMOTE, and bringing outliers back all fail; the
   outlier filter is essential.
4. **SC/VSC must be excluded from the evaluation target** (honest metric = green-only 0.2144),
   not "modeled".
5. **The one real, addressable gap is the wet/drying signal.** Wet laps run ~5× the dry error
   (1.18 vs 0.24 s) and the binary `Rainfall` flag cannot see drying/half-wet tracks (Spa's 11 s
   spread is mostly weather).

### Next step (single highest-value)

Replace/augment the binary `Rainfall` with a **richer track-condition signal** — drying state,
per-lap wetness proxy from lap-time residuals, or the weather endpoint's trends — to flag
transition laps. That targets the only error source a model can actually influence.
