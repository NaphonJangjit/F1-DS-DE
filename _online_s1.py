"""Fully incremental two-stage pipeline: no batch model anywhere, every learner predicts first, then learns.

Stage 1 (8 telemetry channels x 3 online learners: SGD, ARF, kNN; engines in _stream_fast.py)
* every learner starts empty at the first race of 2023 and streams 2023 -> 2026 in calendar order, in
  10-lap chunks: predict the chunk (Pred_* for those rows), then learn its real telemetry. A sector's
  telemetry is known once the sector is done, so nothing from the future is used. 2023 Pred_* come from
  this stream too (no Random Forest, no out-of-fold batch fit).
* target variant per channel / 6 x MAD label clip / features as the v3 design (clip bounds, imputer
  medians and one-hot categories are set from 2023 rows).

Stage-1 champion / challenger (same rules as the web drift lab, _drift_lab.py), race by race from 2023
* v001 = SGD (as in stage 2); challengers ARF and kNN keep learning in the background.
* score = mean over channels of MAE / MAE of "same as last lap" on 2023 (unit-free; 1.0 = no better
  than repeating the previous lap, lower is better).
* after each race: trigger if PSI >= 0.25 (telemetry vs the same circuit earlier) or the champion's
  score >= 1.25 x its own average so far; cooldown 2 races; promote the best challenger if its score is
  >= 0.02 lower and its green-lap score is not worse by > 0.01. Stage 2 gets the champion's Pred_*.

Stage 2: SGD / ARF / kNN online on the served stage-1 Pred_* (warm-up 2023, stream 2024-2026), plus
each model frozen after the 2023 warm-up as the "no learning" reference.

Run:
    py -3.14 _online_s1.py all        # stage1 -> promote -> stage2 -> compare -> web   (~5 min)
    py -3.14 _drift_lab.py --web "D:\\Works\\F1 web"   # stage-2 champion / challenger + web API store
Outputs: online/fast/ (s1_<model>/, s1_promote/, s2_incremental/), online_s1_*.json, stage1_lab.json
"""

import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("F1_ONLINE_YEARS", "2023,2024,2025,2026")

import numpy as np
import pandas as pd

import _fforest as ff
import _online as on
import _stream_fast as sf
from f1pit.train import TARGET, build_preprocessor

BASE = on.OUT                                     # online/stream_2023_2026: cached frames
FAST = on.ROOT / "online" / "fast"
SLIM = FAST / "s1_slim_all.pkl"
LEARNERS = ("SGD", "ARF", "kNN")
CHAMPION0 = "SGD"
S2 = FAST / "s2_incremental"
PSI_TRIGGER, SCORE_TRIGGER, COOLDOWN = 0.25, 1.25, 2
PROMOTE_MARGIN, GUARDRAIL = 0.02, 0.01
WORKERS = max(1, (os.cpu_count() or 4) - 1)
WEB = r"D:\Works\F1 web"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.0f}s]", *a, flush=True)


def s1_dir(m):
    return FAST / f"s1_{m}"


_FR = None
KEYS = ["Session_Key", "Driver", "Lap", "Sector"]


def frames():
    """every sector row (no outlier filter: pit / SC / incident laps included) with `_clean` = the row
    is in the filtered training frames. Models LEARN and are SCORED on clean rows only; they PREDICT
    every row, so inference has a prediction for every lap that exists in the data."""
    global _FR
    if _FR is not None:
        return _FR
    p = FAST / "frames_full.pkl"
    if p.exists():
        _FR = pd.read_pickle(p)
        return _FR
    import math
    from f1pit.features import repair_track_status
    from f1pit.pace_context import PACE_CONTEXT_COLUMNS, add_pace_context
    from f1pit.wind_rain_lab import _prepare_lab
    clean = pd.read_pickle(BASE / "frames.pkl")
    FR = {}
    for y in on.YEARS:
        out_path = on.ROOT / f"data_{y}_out.parquet"
        raw = (pd.read_parquet(out_path) if out_path.exists()
               else ff._reconstruct_outcomes(pd.read_parquet(on.ROOT / f"data_{y}.parquet")))
        raw, _ = repair_track_status(raw)
        raw = add_pace_context(raw)
        df, _, _, _ = _prepare_lab(raw, ["Duration", *[o for _, o in ff.CHANNELS], *PACE_CONTEXT_COLUMNS],
                                   outlier_k=math.inf)
        df = ff._engineer(df)
        df["Year"] = y
        ck = clean[y][KEYS].astype("int64").assign(_c=True)
        df = df.merge(ck, on=KEYS, how="left")
        df["_clean"] = df.pop("_c").fillna(False).astype(bool)
        # an inference-only row with no "last clean lap" anchor falls back to its previous lap
        miss = ~df["_clean"] & df[ff.ANCHOR].isna()
        df.loc[miss, ff.ANCHOR] = df.loc[miss, "Prev_Duration"]
        FR[y] = df
        log(f"{y}: {len(df):,} rows ({int(df['_clean'].sum()):,} clean = training / scoring rows)")
    FAST.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(FR, p)
    _FR = FR
    return FR


def _order(df):
    """stream order: sessions by start, 10-lap chunks, rows by Time_Start."""
    t = pd.to_datetime(df["Time_Start"], utc=True)
    s_start = t.groupby(df["Session_Key"]).transform("min")
    chunk = ((df["Lap"] - 1) // on.CHUNK_LAPS).astype(int)
    return df.assign(_t=t, _s=s_start, _chunk=chunk).sort_values(["_s", "Session_Key", "_chunk", "_t"])


def _bounds(keys):
    return np.r_[0, np.where((keys[1:] != keys[:-1]).any(axis=1))[0] + 1, len(keys)]


# ------------------------------------------------------------------ stage 1
def prepare():
    FR = frames()
    common = set.intersection(*(set(FR[y].columns) for y in on.YEARS))
    need = {"Session_Key", "Lap", "Time_Start", "_clean"}
    spec = {}
    for prev, o in ff.CHANNELS:
        v = on.S1_VARIANT[o]
        cols = on._s1_cols(prev, o, v, common)
        spec[o] = (prev, v, cols)
        need |= {prev, o, *cols}
    need = [c for c in need if c in common]
    parts = [FR[y][need].assign(_year=y, _pos=np.arange(len(FR[y]))) for y in on.YEARS]
    stream = _order(pd.concat(parts, ignore_index=True)).reset_index(drop=True)
    FAST.mkdir(parents=True, exist_ok=True)
    pd.to_pickle({"stream": stream, "spec": spec}, SLIM)
    log(f"slim data: {len(stream):,} rows 2023-{on.YEARS[-1]}")


def _s1_job(task):
    learner, o = task
    t0 = time.time()
    D = pd.read_pickle(SLIM)
    S = D["stream"]
    prev, v, cols = D["spec"][o]
    clean = S["_clean"].to_numpy(bool)
    y23 = (S["_year"] == 2023).to_numpy() & clean
    t = ff._target(v, S[o].to_numpy(float), S[prev].to_numpy(float))
    t[~clean] = np.nan                       # learn from clean rows only; every row is still predicted
    t23 = t[y23 & np.isfinite(t)]
    med = np.median(t23)
    mad = np.median(np.abs(t23 - med))
    lo, hi = (med - 6 * mad, med + 6 * mad) if mad > 0 else (-np.inf, np.inf)
    pre = build_preprocessor(cols).fit(S.loc[y23, cols])
    X = np.asarray(pre.transform(S[cols]), float)
    bounds = _bounds(S[["Session_Key", "_chunk"]].to_numpy())
    empty = np.empty((0, X.shape[1]))
    p, _ = sf.prequential(learner, empty, np.empty(0), X, np.clip(t, lo, hi), bounds)
    lvl = ff._to_level(v, p, S[prev].to_numpy(float))
    print(f"[stage 1] {learner:4s} {o:28s} {time.time() - t0:6.0f}s", flush=True)
    return learner, o, lvl


def stage1():
    if not SLIM.exists():
        prepare()
    tasks = [(m, o) for m in LEARNERS for _, o in ff.CHANNELS]
    log(f"stage 1: {len(tasks)} jobs on {WORKERS} processes, every learner from an empty model")
    with Pool(WORKERS) as pool:
        res = pool.map(_s1_job, tasks, chunksize=1)
    S = pd.read_pickle(SLIM)["stream"]
    FR = frames()
    for m in LEARNERS:
        P = {y: {} for y in on.YEARS}
        for learner, o, lvl in res:
            if learner != m:
                continue
            for y in on.YEARS:
                sel = (S["_year"] == y).to_numpy()
                arr = np.full(len(FR[y]), np.nan)
                arr[S.loc[sel, "_pos"].to_numpy()] = lvl[sel]
                P[y][f"Pred_{o}"] = arr
        s1_dir(m).mkdir(parents=True, exist_ok=True)
        pd.to_pickle(P, s1_dir(m) / "stage1.pkl")
        acc = {}
        for prev, o in ff.CHANNELS:
            acc[o] = {}
            for y in on.YEARS:
                act, last = FR[y][o].to_numpy(float), FR[y][prev].to_numpy(float)
                f1 = np.asarray(P[y][f"Pred_{o}"], float)
                k = np.isfinite(act) & np.isfinite(f1) & np.isfinite(last) & FR[y]["_clean"].to_numpy(bool)
                acc[o][str(y)] = {"online": round(float(np.abs(act[k] - f1[k]).mean()), 4) if k.any() else None,
                                  "same_as_last_lap": round(float(np.abs(act[k] - last[k]).mean()), 4) if k.any() else None}
        (s1_dir(m) / "stage1_accuracy.json").write_text(json.dumps(acc, indent=2), encoding="utf-8")
        log(f"stage 1 {m} 2025: " + "  ".join(
            f"{o.replace('Outcome_', '')} {a['2025']['online']} (last lap {a['2025']['same_as_last_lap']})"
            for o, a in acc.items()))


# ------------------------------------------------------------------ stage-1 champion / challenger
def promote():
    import _drift_lab as dl
    FR = frames()
    cands = {f"{m} online": pd.read_pickle(s1_dir(m) / "stage1.pkl") for m in LEARNERS}
    champ0 = f"{CHAMPION0} online"
    chans = [o for _, o in ff.CHANNELS]
    prevs = dict((o, p) for p, o in ff.CHANNELS)
    d23 = FR[2023][FR[2023]["_clean"]]                 # scoring uses clean rows only (as the analysis)
    scale = {o: float(np.nanmean(np.abs(d23[o].to_numpy(float) - d23[prevs[o]].to_numpy(float)))) for o in chans}

    meta = pd.concat([pd.read_parquet(on.ROOT / f"data_{y}.parquet",
                                      columns=["Session_Key", "Country", "Circuit_Total_Length", "Time_Start"])
                      for y in on.YEARS], ignore_index=True)
    races = meta.groupby("Session_Key").agg(country=("Country", "first"), length=("Circuit_Total_Length", "first"),
                                            t=("Time_Start", "min")).reset_index()
    races["date"] = pd.to_datetime(races["t"], utc=True).dt.strftime("%Y-%m-%d")
    races["circuit"] = (races["country"] + " "
                        + (pd.to_numeric(races["length"], errors="coerce") / 1000).round(1).astype(str) + " km")
    races = dl._name_circuits(races)
    feats = pd.concat([d[d["_clean"]].assign(Year=y)[["Session_Key", "Year", *dl.DRIFT_FEATURES]]
                       for y, d in FR.items()], ignore_index=True)
    feats["Prev_DRS_Pct"] = feats["Prev_DRS_Pct"].fillna(0.0)
    feats = feats.merge(races[["Session_Key", "circuit", "date"]], on="Session_Key", how="left")

    err, parts = {}, []
    for y in on.YEARS:
        d = FR[y]
        parts.append(pd.DataFrame({"Session_Key": d["Session_Key"].to_numpy(), "Year": y,
                                   "green": d["Track_Status"].astype(str).str.upper().eq("GREEN").to_numpy()}))
        for c, P in cands.items():
            e = np.column_stack([np.abs(d[o].to_numpy(float) - np.asarray(P[y][f"Pred_{o}"], float)) / scale[o]
                                 for o in chans])
            err.setdefault(c, []).append(e)
        e = np.column_stack([np.abs(d[o].to_numpy(float) - d[prevs[o]].to_numpy(float)) / scale[o] for o in chans])
        err.setdefault("same as last lap", []).append(e)
    rows = pd.concat(parts, ignore_index=True)
    err = {c: np.vstack(v) for c, v in err.items()}
    notclean = ~np.concatenate([FR[y]["_clean"].to_numpy(bool) for y in on.YEARS])
    for c in err:
        err[c][notclean] = np.nan                       # inference-only rows are not scored
    order = races[races["Session_Key"].isin(rows["Session_Key"].unique())].sort_values("date").reset_index(drop=True)

    def score(c, m):
        if not m.any():
            return np.nan
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(np.nanmean(np.nanmean(err[c][m], axis=0)))

    champion, version = champ0, "v001"
    lineage = [{"version": "v001", "model": champion, "since": "2023 race 1",
                "reason": "same starting champion as stage 2 (SGD); every learner starts empty"}]
    served, steps, last, hist = {}, [], -99, []
    for i, r in order.iterrows():
        sk = r["Session_Key"]
        served[int(sk)] = champion
        m = (rows["Session_Key"] == sk).to_numpy()
        g = m & rows["green"].to_numpy()
        sc = {c: score(c, m) for c in [*cands, "same as last lap"]}
        gs = {c: score(c, g) for c in cands}
        fcur = feats[feats["Session_Key"] == sk]
        ref = feats[(feats["circuit"] == r["circuit"]) & (feats["date"] < r["date"])]
        if len(ref) < 500:
            ref = feats[feats["date"] < r["date"]].tail(20000)
        psis = {f: dl.psi(ref[f].to_numpy(float), fcur[f].to_numpy(float)) for f in dl.DRIFT_FEATURES}
        psi_max = max((v for v in psis.values() if v is not None), default=0.0)
        base = np.mean(hist) if len(hist) >= 3 else None
        reason = ("feature drift" if psi_max >= PSI_TRIGGER
                  else "performance" if base is not None and sc[champion] >= SCORE_TRIGGER * base else None)
        decision = "NO TRIGGER" if reason is None else "COOLDOWN"
        before = f"{version} {champion}"
        if reason and i - last >= COOLDOWN:
            last = i
            best = min((c for c in cands if c != champion), key=lambda c: sc[c])
            d_ = sc[best] - sc[champion]
            ok = gs[best] - gs[champion] <= GUARDRAIL if np.isfinite(gs[best]) and np.isfinite(gs[champion]) else True
            if d_ <= -PROMOTE_MARGIN and ok:
                version = f"v{len(lineage) + 1:03d}"
                champion = best
                lineage.append({"version": version, "model": best, "since": f"after {r['date'][:4]} {r['circuit']}",
                                "reason": f"{reason}: score {d_:+.3f}"})
                decision = "PROMOTE"
                hist = []
            else:
                decision = "KEEP"
        if np.isfinite(sc[before.split(' ', 1)[1]]):
            hist.append(sc[before.split(' ', 1)[1]])
        steps.append({"session_key": int(sk), "year": int(r["date"][:4]), "date": r["date"], "circuit": r["circuit"],
                      "champion": before, "scores": {c: round(v, 4) for c, v in sc.items()}, "psi_max": psi_max,
                      "trigger": reason, "decision": decision, "champion_after": f"{version} {champion}"})
        log(f"{i + 1:2d} {r['date']} {r['circuit']:<22s} {before:<16s} " +
            " ".join(f"{c.split()[0]} {sc[c]:.3f}" for c in cands) + f" last {sc['same as last lap']:.3f}"
            f" | PSI {psi_max:.2f} {decision}")

    P = {y: {} for y in on.YEARS}
    for y in on.YEARS:
        who = np.array([served[int(k)] for k in FR[y]["Session_Key"].to_numpy()])
        for o in chans:
            arr = np.full(len(who), np.nan)
            for c, Pc in cands.items():
                use = who == c
                arr[use] = np.asarray(Pc[y][f"Pred_{o}"], float)[use]
            P[y][f"Pred_{o}"] = arr
    s1_dir("promote").mkdir(parents=True, exist_ok=True)
    pd.to_pickle(P, s1_dir("promote") / "stage1.pkl")
    yrs = rows["Year"].to_numpy()
    served_rows = np.array([served[int(k)] for k in rows["Session_Key"]])
    by_year = {}
    for y in on.YEARS:
        my = yrs == y
        by_year[str(y)] = {c: score(c, my) for c in [*cands, "same as last lap"]}
        e = np.vstack([err[c][my & (served_rows == c)] for c in cands])
        by_year[str(y)]["served"] = float(np.nanmean(np.nanmean(e, axis=0)))
    res = {"score": "mean over channels of MAE / MAE of 'same as last lap' on 2023 (1.0 = no better than "
                    "repeating the previous lap, lower is better)",
           "rules": {"psi_trigger": PSI_TRIGGER, "score_trigger": SCORE_TRIGGER, "cooldown": COOLDOWN,
                     "promote_margin": PROMOTE_MARGIN, "guardrail": GUARDRAIL},
           "lineage": lineage, "score_by_year": by_year, "steps": steps}
    (on.ROOT / "online_s1_promote.json").write_text(json.dumps(dl._clean(res), indent=1), encoding="utf-8")
    log(f"lineage: {[l['version'] + ' ' + l['model'] + ' (' + l['since'] + ')' for l in lineage]}")
    for y, v in by_year.items():
        log(y + " " + "  ".join(f"{c}={x:.3f}" for c, x in v.items()))


# ------------------------------------------------------------------ stage 2
def _build_stream(P):
    """= _online.build_stream on every row; preprocessing fitted on clean 2023 rows, y only on clean rows."""
    from f1pit.pace_context import add_relative_pace
    FR = frames()
    parts = []
    for y in on.YEARS:
        d = FR[y].copy()
        for c in on.PRED:
            d[c] = np.asarray(P[y][c])
        parts.append(d)
    allf = pd.concat(parts, ignore_index=True)
    allf, rel_map = add_relative_pace(allf, ff.ANCHOR)
    feats = [c for c in dict.fromkeys(rel_map.get(c, c) for c in [*on.PRED, *ff.BASE2]) if c in allf.columns]
    if ff.ANCHOR not in feats:
        feats.append(ff.ANCHOR)
    allf["y"] = allf[TARGET] - np.log(allf[ff.ANCHOR])
    allf.loc[~allf["_clean"], "y"] = np.nan              # inference-only rows: predicted, never learned
    allf["t"] = pd.to_datetime(allf["Time_Start"], utc=True)
    allf["s_start"] = allf.groupby("Session_Key")["t"].transform("min")
    allf["chunk"] = ((allf["Lap"] - 1) // on.CHUNK_LAPS).astype(int)
    allf = allf.sort_values(["s_start", "Session_Key", "chunk", "t"]).reset_index(drop=True)
    pre = build_preprocessor(feats).fit(allf.loc[(allf.Year == 2023) & allf["_clean"], feats])
    X = pre.transform(allf[feats])
    names = [n.replace("num__", "").replace("cat__", "") for n in pre.get_feature_names_out()]
    return allf, X, names


def _stage2_data():
    P = pd.read_pickle(s1_dir("promote") / "stage1.pkl")
    allf, X, names = _build_stream(P)
    year = allf["Year"].to_numpy()
    warm, stream = np.where(year == 2023)[0], np.where(year > 2023)[0]
    bounds = _bounds(allf.loc[stream, ["Session_Key", "chunk"]].to_numpy())
    S2.mkdir(parents=True, exist_ok=True)
    np.savez(S2 / "stream.npz", Xw=np.asarray(X[warm], float), yw=allf["y"].to_numpy()[warm],
             X=np.asarray(X[stream], float), y=allf["y"].to_numpy()[stream], bounds=bounds)
    s = allf.loc[stream].reset_index(drop=True)
    s[["Session_Key", "Driver", "Year", "chunk", "Lap", "Sector", "Time_Start", "s_start", "Track_Status",
       "Rainfall"]].assign(actual=np.exp(s[TARGET].to_numpy(float)), anchor=s[ff.ANCHOR].to_numpy(float),
                           clean=s["_clean"].to_numpy(bool)).to_parquet(S2 / "stream_index.parquet")
    log(f"stage 2 data: warm-up {len(warm):,}, stream {len(stream):,}, {len(bounds) - 1} chunks, {X.shape[1]} features")


def _s2_job(name):
    t0 = time.time()
    z = np.load(S2 / "stream.npz")
    p_on, p_st = sf.prequential(name, z["Xw"], z["yw"], z["X"], z["y"], z["bounds"], frozen=True)
    np.savez(S2 / f"preds_{name}.npz", online=p_on, static=p_st)
    print(f"[stage 2] {name:4s} {time.time() - t0:6.0f}s", flush=True)
    return name


def stage2():
    _stage2_data()
    with Pool(3) as pool:
        pool.map(_s2_job, ["ARF", "kNN", "SGD"], chunksize=1)


# ------------------------------------------------------------------ compare
def compare():
    s = pd.read_parquet(S2 / "stream_index.parquet")
    act, anc = s["actual"].to_numpy(), s["anchor"].to_numpy()
    first = s.groupby("Session_Key")[["s_start", "Year"]].first().sort_values("s_start")
    last3 = first[first["Year"] == 2025].index[-3:]
    periods = {"2024": s.Year == 2024, "2025": s.Year == 2025, "2026": s.Year == 2026,
               "2024-2025": s.Year.isin([2024, 2025]), "2025 last 3 GPs": s.Session_Key.isin(last3),
               "all": s.Year > 0}
    periods = {k: m & s["clean"] for k, m in periods.items()}      # scored on clean rows (as the analysis)
    res = {}
    for name in ("SGD", "ARF", "kNN"):
        z = np.load(S2 / f"preds_{name}.npz")
        for mode, lab in (("online", "online"), ("static", "frozen")):
            e = np.abs(act - anc * np.exp(z[mode]))
            res[f"{name} {lab}"] = {k: round(float(e[m.to_numpy()].mean()), 4) for k, m in periods.items()}
    out = {"pipeline": "fully incremental: stage 1 online (champion / challenger) -> stage 2 online",
           "engines": "SGD river · ARF CapyMOA (MOA) · kNN numpy (identical to river kNN)",
           "mae_s": res,
           "stage1_accuracy": {m: json.loads((s1_dir(m) / "stage1_accuracy.json").read_text(encoding="utf-8"))
                               for m in LEARNERS},
           "stage1_promote": {k: v for k, v in json.loads((on.ROOT / "online_s1_promote.json")
                                                          .read_text(encoding="utf-8")).items() if k != "steps"}}
    (on.ROOT / "online_s1_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for k, v in res.items():
        print(f"{k:12s} " + "  ".join(f"{p}={x:.4f}" for p, x in v.items()))


def web(folder=WEB):
    """stage1_lab.json / .js for the web drift lab."""
    import _drift_lab as dl
    pr = json.loads((on.ROOT / "online_s1_promote.json").read_text(encoding="utf-8"))
    rs = json.loads((on.ROOT / "online_s1_results.json").read_text(encoding="utf-8"))
    out = {"rules": pr["rules"], "score": pr["score"], "lineage": pr["lineage"],
           "score_by_year": pr["score_by_year"], "candidates": [f"{m} online" for m in LEARNERS],
           "races": [{k: s[k] for k in ("year", "circuit", "champion", "scores", "psi_max", "trigger", "decision",
                                        "champion_after")} for s in pr["steps"]],
           "served_by": {c: sum(s["champion"].split(" ", 1)[1] == c for s in pr["steps"])
                         for c in [f"{m} online" for m in LEARNERS]},
           "stage2": rs["mae_s"], "engines": rs["engines"]}
    payload = json.dumps(dl._clean(out), indent=1)
    (on.ROOT / "stage1_lab.json").write_text(payload, encoding="utf-8")
    (on.ROOT / "stage1_lab.js").write_text(f"window.STAGE1_LAB = {payload};\n", encoding="utf-8")
    for n in ("stage1_lab.json", "stage1_lab.js"):
        shutil.copy(on.ROOT / n, Path(folder) / n)
    log(f"stage1_lab.json/.js -> {folder}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    steps = {"prepare": [prepare], "stage1": [stage1], "promote": [promote], "stage2": [stage2],
             "compare": [compare], "web": [web], "all": [stage1, promote, stage2, compare, web]}
    if cmd not in steps:
        raise SystemExit(__doc__)
    for f in steps[cmd]:
        f()
