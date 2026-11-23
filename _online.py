"""Online (streaming) stage 2: Adaptive Random Forest vs SGD vs kNN (river), v3 data design.

Question: if the lap-time model keeps learning while the season runs, which online model is best,
and does updating beat a model that is trained once and frozen?

Protocol (prequential = predict first, then learn, like a live pipeline)
------------------------------------------------------------------------
warm-up : learn every 2023 row in time order (sessions by start time, rows by Time_Start).
stream  : 2024 then 2025, sessions in calendar order; each session is cut into chunks of 10 laps
          (laps 1-10, 11-20, ...). For each chunk: predict all its rows with the current model, then
          learn them in Time_Start order. The model is updated every 10 laps of the race that is
          running, in the order the data would really arrive (not "laps 1-10 of every race" at once).
frozen  : a deep copy of each model right after the warm-up, never updated (same features).
RF batch: sklearn Random Forest (the _fforest.py hyper-parameters) fitted once on 2023, never updated.

Leakage rules
-------------
* Stage 1 (Pred_*) is fitted on 2023 only: 5-fold GroupKFold(session) out-of-fold predictions for the
  2023 warm-up rows, one full-2023 fit predicts 2024 + 2025. The stage-1 target variant per channel is
  fixed to the v3 choice (S1_VARIANT below) so no stage-1 refit sees 2024/2025.
* Imputer / one-hot are fitted on 2023 rows only; scaling is online (river StandardScaler).
* Target = delta vs Rep_Prev_Duration (v3); Duration = Rep_Prev_Duration * exp(delta).
  Features = the _fforest.py stage-2 set (Pred_* + BASE2, duration anchors as Rel_* log-ratios).

Data
----
data_<y>_out.parquet if present (real Outcome_* columns). Otherwise data_<y>.parquet with
Outcome_X(t) reconstructed as Prev_X(t+1) (same session / driver / sector, next lap; a next lap whose
8 telemetry channels are all carried forward unchanged counts as missing). Track_Status is re-derived
with the red-flag-aware parser (repair_track_status, race_control from the .cache/openf1 disk cache).

Run (needs `pip install river`)
-------------------------------
    py -3.14 _online.py run                 # all models: ARF, SGD, kNN, RF batch  (~55 min)
    py -3.14 _online.py run SGD,RF          # a subset (SGD < 1 min, ARF ~12 min, kNN ~37 min)
    py -3.14 _online.py plot                # summary + bootstrap -> online_results.json,
                                            #   figures/online_results.png
Outputs: online/ (per-row predictions, stream index, caches), online_results.json, figure.
"""

import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

import _fforest as ff
from f1pit.features import repair_track_status
from f1pit.pace_context import PACE_CONTEXT_COLUMNS, add_pace_context, add_relative_pace
from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab

ROOT = Path(__file__).resolve().parent
# F1_ONLINE_YEARS="2023,2024,2025,2026" extends the stream (warm-up stays 2023) into its own folder,
# so the published 2023-2025 results in online/ are left untouched
YEARS = tuple(int(y) for y in os.environ.get("F1_ONLINE_YEARS", "2023,2024,2025").split(","))
OUT = ROOT / "online" / ("" if YEARS == (2023, 2024, 2025) else f"stream_{YEARS[0]}_{YEARS[-1]}")
FIG = ROOT / "figures"
CHUNK_LAPS = 10
KNN_WINDOW = 2000
MODELS = ("ARF", "SGD", "kNN", "RF")

# stage-1 target variant per channel = the v3 _fforest.py choice (2023 -> held-out 2024 selection)
S1_VARIANT = {
    "Outcome_Avg_Speed_Corners": "log",
    "Outcome_Avg_Speed_Straight": "delta_slog",
    "Outcome_Full_Throttle_Pct": "level",
    "Outcome_Brake_Pct": "level",
    "Outcome_DRS_Pct": "delta_slog",
    "Outcome_Max_Speed": "delta_slog",
    "Outcome_Min_Speed": "delta_slog",
    "Outcome_Avg_Gear": "delta_slog",
}
PRED = [f"Pred_{o}" for _, o in ff.CHANNELS]
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ data
_reconstruct_outcomes = ff._reconstruct_outcomes   # Outcome_X(t) = Prev_X(t+1), shared with _fforest.py


def load_frames():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "frames.pkl"
    if p.exists():
        return pd.read_pickle(p)
    frames = {}
    for y in YEARS:
        out_path = ROOT / f"data_{y}_out.parquet"
        if out_path.exists():
            raw, src = pd.read_parquet(out_path), out_path.name
        else:
            raw, src = _reconstruct_outcomes(pd.read_parquet(ROOT / f"data_{y}.parquet")), \
                f"data_{y}.parquet (Outcome_* reconstructed)"
        raw, n_fix = repair_track_status(raw)
        raw = add_pace_context(raw)
        df, _, _, _ = _prepare_lab(raw, ["Duration", *[o for _, o in ff.CHANNELS], *PACE_CONTEXT_COLUMNS])
        df = ff._engineer(df)
        df["Year"] = y
        frames[y] = df
        log(f"{y}: {len(df):,} rows from {src}; Track_Status repaired on {n_fix:,} rows")
    pd.to_pickle(frames, p)
    return frames


def _s1_cols(prev, out, variant, common):
    trend = [f"{prev}_roll5_mean", f"{prev}_roll5_slope"]
    cols = [prev, *ff.TYRE, *ff.S1_TYRE_EXTRA, *ff.WEATHER, "Track_Status", *ff.TRAFFIC, "Lap",
            *ff.GEOMETRY[out], *trend]
    cols = [c for c in dict.fromkeys(cols) if c in common and c != out]
    return [c for c in cols if c != prev] if variant in ("delta", "delta_slog") else cols


def stage1_2023(FR):
    """Pred_* from 2023 only: OOF for 2023, full-2023 fit for 2024 / 2025."""
    p = OUT / "stage1_2023.pkl"
    if p.exists():
        return pd.read_pickle(p)
    common = set.intersection(*(set(FR[y].columns) for y in YEARS))
    d23 = FR[2023]
    folds = list(GroupKFold(n_splits=ff.N_FOLDS).split(d23, groups=d23["Session_Key"]))
    out = {y: {} for y in YEARS}
    for prev, o in ff.CHANNELS:
        v = S1_VARIANT[o]
        cols = _s1_cols(prev, o, v, common)
        oof = np.full(len(d23), np.nan)
        for tr_i, va_i in folds:
            pre, rf = ff._fit_sub(d23.iloc[tr_i], prev, o, v, cols)
            oof[va_i] = ff._sub_predict(pre, rf, d23.iloc[va_i], prev, v, cols)
        pre, rf = ff._fit_sub(d23, prev, o, v, cols)
        out[2023][f"Pred_{o}"] = oof
        for y in YEARS[1:]:                       # every later season gets the full-2023 fit
            out[y][f"Pred_{o}"] = ff._sub_predict(pre, rf, FR[y], prev, v, cols)
        log(f"stage 1 {o} ({v}) done")
    pd.to_pickle(out, p)
    return out


def build_stream():
    """-> (rows in stream order, X, feature names); 2023 rows first (warm-up), then 2024-2025."""
    FR = load_frames()
    P = stage1_2023(FR)
    parts = []
    for y in YEARS:
        d = FR[y].copy()
        for c in PRED:
            d[c] = np.asarray(P[y][c])
        parts.append(d)
    allf = pd.concat(parts, ignore_index=True)
    allf, rel_map = add_relative_pace(allf, ff.ANCHOR)
    feats = [c for c in dict.fromkeys(rel_map.get(c, c) for c in [*PRED, *ff.BASE2]) if c in allf.columns]
    if ff.ANCHOR not in feats:
        feats.append(ff.ANCHOR)
    allf["y"] = allf[TARGET] - np.log(allf[ff.ANCHOR])
    allf["t"] = pd.to_datetime(allf["Time_Start"], utc=True)
    allf["s_start"] = allf.groupby("Session_Key")["t"].transform("min")
    allf["chunk"] = ((allf["Lap"] - 1) // CHUNK_LAPS).astype(int)
    allf = allf.sort_values(["s_start", "Session_Key", "chunk", "t"]).reset_index(drop=True)
    pre = build_preprocessor(feats).fit(allf.loc[allf.Year == 2023, feats])
    X = pre.transform(allf[feats])
    names = [n.replace("num__", "").replace("cat__", "") for n in pre.get_feature_names_out()]
    return allf, X, names


# ------------------------------------------------------------------ models (river)
def make(name, knn_window=KNN_WINDOW):
    from river import forest, linear_model, neighbors, optim, preprocessing
    if name == "ARF":
        core = forest.ARFRegressor(n_models=10, seed=42)
        return preprocessing.StandardScaler() | preprocessing.TargetStandardScaler(regressor=core)
    if name == "SGD":   # same recipe as _fforest._fit_sgd: huber loss, inverse-scaling lr 0.01, scaled X/y
        core = linear_model.LinearRegression(optimizer=optim.SGD(optim.schedulers.InverseScaling(0.01)),
                                             loss=optim.losses.Huber(), l2=1e-4)
        return preprocessing.StandardScaler() | preprocessing.TargetStandardScaler(regressor=core)
    if name == "kNN":   # k=15, distance-weighted (as _fforest._fit_knn), sliding window of recent rows
        core = neighbors.KNNRegressor(n_neighbors=15, engine=neighbors.LazySearch(window_size=knn_window),
                                      aggregation_method="weighted_mean")
        return preprocessing.StandardScaler() | core
    raise ValueError(name)


def run(models, knn_window=KNN_WINDOW):
    OUT.mkdir(parents=True, exist_ok=True)
    allf, X, names = build_stream()
    rows = [dict(zip(names, r)) for r in X.tolist()]
    y = allf["y"].to_numpy()
    year = allf["Year"].to_numpy()
    warm, stream = np.where(year == 2023)[0], np.where(year > 2023)[0]
    keys = allf.loc[stream, ["Session_Key", "chunk"]].to_numpy()
    bounds = np.r_[0, np.where((keys[1:] != keys[:-1]).any(axis=1))[0] + 1, len(stream)]
    log(f"warm-up rows {len(warm):,}; stream rows {len(stream):,} in {len(bounds) - 1:,} chunks "
        f"({allf.loc[stream, 'Session_Key'].nunique()} sessions); features {len(names)}")

    s = allf.loc[stream].reset_index(drop=True)
    s[["Session_Key", "Driver", "Year", "chunk", "Lap", "Sector", "Time_Start", "s_start", "Track_Status",
       "Rainfall"]].assign(
        actual=np.exp(s[TARGET].to_numpy(float)), anchor=s[ff.ANCHOR].to_numpy(float)
    ).to_parquet(OUT / "stream_index.parquet")

    for name in models:
        if name == "RF":   # batch reference: trained once on 2023, never updated
            rf = RandomForestRegressor(n_estimators=ff.N_EST, max_features=ff.MAX_FEATURES,
                                       min_samples_leaf=ff.MIN_SAMPLES_LEAF, n_jobs=ff.CORES,
                                       random_state=42).fit(X[warm], y[warm])
            np.savez(OUT / "preds_RF.npz", static=rf.predict(X[stream]))
            log("RF batch done")
            continue
        m = make(name, knn_window)
        for i in warm:
            m.learn_one(rows[i], y[i])
        frozen = copy.deepcopy(m)
        log(f"{name}: warm-up done")
        p_on, p_st = np.empty(len(stream)), np.empty(len(stream))
        for b in range(len(bounds) - 1):
            idx = stream[bounds[b]:bounds[b + 1]]
            for j, i in enumerate(idx):                  # 1) predict the whole 10-lap chunk
                p_on[bounds[b] + j] = m.predict_one(rows[i])
                p_st[bounds[b] + j] = frozen.predict_one(rows[i])
            for i in idx:                                # 2) then learn it
                m.learn_one(rows[i], y[i])
            if b % 100 == 0:
                log(f"  {name}: chunk {b}/{len(bounds) - 1}")
        np.savez(OUT / f"preds_{name}.npz", online=p_on, static=p_st)
        log(f"{name}: stream done")


# ------------------------------------------------------------------ summary + figure
def _boot_diff(e1, e2, sess, n=5000, seed=0):
    """session-level paired bootstrap of mean(e1 - e2) -> (mean, ci_lo, ci_hi)."""
    d = pd.DataFrame({"s": sess, "d": e1 - e2}).groupby("s")["d"].agg(["sum", "count"])
    idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
    bs = d["sum"].to_numpy()[idx].sum(1) / d["count"].to_numpy()[idx].sum(1)
    return float((e1 - e2).mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def summarize():
    s = pd.read_parquet(OUT / "stream_index.parquet")
    act, anc = s["actual"].to_numpy(), s["anchor"].to_numpy()
    series = {}
    for name in ("ARF", "SGD", "kNN"):
        p = OUT / f"preds_{name}.npz"
        if p.exists():
            z = np.load(p)
            series[f"{name} online"] = np.abs(act - anc * np.exp(z["online"]))
            series[f"{name} frozen"] = np.abs(act - anc * np.exp(z["static"]))
    if (OUT / "preds_RF.npz").exists():
        series["RF batch (frozen)"] = np.abs(act - anc * np.exp(np.load(OUT / "preds_RF.npz")["static"]))
    m25 = (s["Year"] == 2025).to_numpy()
    res = {}
    for k, e in series.items():
        res[k] = {"all": float(e.mean()), "2024": float(e[s.Year == 2024].mean()),
                  "2025": float(e[m25].mean())}
    # 2025 breakdown (overall / clean / dry / wet ...) from the saved predictions
    for k in res:
        name = k.split()[0]
        if k == "RF batch (frozen)":
            p = np.load(OUT / "preds_RF.npz")["static"]
        else:
            p = np.load(OUT / f"preds_{name}.npz")["online" if k.endswith("online") else "static"]
        bd = ff._eval_breakdown(np.log(act[m25]), np.log(anc[m25] * np.exp(p[m25])), s[m25])
        res[k]["2025_breakdown"] = {kk: v["mae_s"] for kk, v in bd.items()}
    online = [k for k in res if k.endswith("online")]
    best = min(online, key=lambda k: res[k]["all"])
    tests = {}
    sess = s["Session_Key"].to_numpy()
    if len(online) > 1:
        second = sorted(online, key=lambda k: res[k]["all"])[1]
        tests[f"{best} - {second}"] = _boot_diff(series[best], series[second], sess)
    for name in ("ARF", "SGD", "kNN"):
        if f"{name} online" in series:
            tests[f"{name} online - {name} frozen"] = _boot_diff(series[f"{name} online"],
                                                                 series[f"{name} frozen"], sess)
    out = {"protocol": {"warm_up": "2023", "stream": "2024-2025, calendar order",
                        "chunk_laps": CHUNK_LAPS, "knn_window": KNN_WINDOW,
                        "stream_rows": int(len(s)), "sessions": int(s["Session_Key"].nunique())},
           "mae_s": res, "best_online": best,
           "paired_bootstrap_session": {k: {"diff": round(v[0], 4), "ci95": [round(v[1], 4), round(v[2], 4)]}
                                        for k, v in tests.items()}}
    (ROOT / "online_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, r in res.items():
        print(f"{k:20s} all={r['all']:.4f}  2024={r['2024']:.4f}  2025={r['2025']:.4f}  "
              f"wet2025={r['2025_breakdown']['wet']}")
    print("best online:", best)
    for k, v in out["paired_bootstrap_session"].items():
        print(f"  {k:28s} diff={v['diff']:+.4f}  95% CI {v['ci95']}")
    return s, series, res, best


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s, series, res, best = summarize()
    RED, DARK, GREY, BLUE = "#C8282D", "#151515", "#8C8C8C", "#2F6DB5"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 17,
                         "axes.titlelocation": "left"})
    df = s.assign(**series)
    order = df.groupby("Session_Key")["s_start"].first().sort_values().index
    per_race = df.groupby("Session_Key")[list(series)].mean().loc[order].reset_index(drop=True)
    years = df.groupby("Session_Key")["Year"].first().loc[order].to_numpy()

    style = {"ARF online": (RED, "-", 3), "SGD online": (BLUE, "-", 2), "kNN online": (DARK, "-", 2),
             "ARF frozen": (RED, ":", 2), "RF batch (frozen)": (GREY, "--", 2)}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.2), gridspec_kw={"width_ratios": [1.7, 1]})
    x = np.arange(1, len(per_race) + 1)
    for k, (c, ls, lw) in style.items():
        if k in per_race:
            a1.plot(x, per_race[k].rolling(3, min_periods=1).mean(), color=c, ls=ls, lw=lw,
                    label=f"{k}  ({res[k]['all']:.3f} s)")
    cut = int((years == 2024).sum())
    top = a1.get_ylim()[1] * 1.12
    a1.set_ylim(top=top)
    a1.axvline(cut + 0.5, color=GREY, lw=1)
    for xpos, lab in ((cut * 0.5, "2024"), (cut + (len(x) - cut) * 0.5, "2025")):
        a1.text(xpos, top * 0.99, lab, ha="center", va="top", color=GREY, fontsize=16, fontweight="bold")
    a1.set_xlabel("Race number in the stream (calendar order)")
    a1.set_ylabel("MAE per sector (s), 3-race average")
    a1.set_title("Learning while racing: error race by race")
    a1.legend(frameon=False, fontsize=12.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)

    names = [k for k in ("ARF online", "SGD online", "kNN online", "RF batch (frozen)") if k in res]
    vals = [res[k]["2025"] for k in names]
    bars = a2.barh(range(len(names))[::-1], vals, color=[RED if k == best else GREY for k in names], height=0.6)
    for b_, v, k in zip(bars, vals, names):
        tag = "  best online" if k == best else ("  no updates" if "frozen" in k else "")
        a2.text(v + 0.003, b_.get_y() + b_.get_height() / 2, f"{v:.3f} s{tag}", va="center", fontsize=15,
                fontweight="bold", color=RED if k == best else DARK)
    a2.set_yticks(range(len(names))[::-1], names, fontsize=14)
    a2.set_xlim(0, max(vals) * 1.55)
    a2.set_xlabel("MAE on 2025 (s) - lower is better")
    a2.set_title("2025 season")
    fig.tight_layout()
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "online_results.png", dpi=200, facecolor="white", bbox_inches="tight")
    print("wrote figures/online_results.png")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run(sys.argv[2].split(",") if len(sys.argv) > 2 else list(MODELS))
    elif cmd == "plot":
        plot()
    else:
        raise SystemExit(__doc__)
