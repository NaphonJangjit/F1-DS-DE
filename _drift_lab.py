"""F1 drift lab (online family only): race-by-race Champion / Challenger, end of 2025 -> 2026.

Built on the fully incremental stream of _online_s1.py (stage 1 = online telemetry learners with their own
champion / challenger, no batch model; stage 2 warm-up 2023, then 2024 -> 2026 in calendar order, every race
cut into 10-lap chunks: predict the chunk, then learn it). F1_STREAM_DIR picks another stream folder. Three online learners run side by side on that
stream (river): SGD (Huber), ARF (Adaptive Random Forest + ADWIN) and kNN (sliding window). Every sector
of 2024-2026 therefore has a prediction made BEFORE the model saw it.

Production rule (one race = one step):
  * one model serves (the Champion, v001 = SGD); the other two are shadow Challengers that keep learning
  * after each simulated race (last 3 GPs of 2025, then every 2026 race):
      - feature drift: PSI of 5 telemetry features vs the same circuit in earlier seasons
      - trigger = PSI >= 0.25 on any feature, or Champion MAE >= 1.25 x its MAE on 2025 so far
        (cooldown: one decision every 2 races)
      - on a trigger, the best Challenger on that race is promoted only if it beats the Champion by
        >= PROMOTE_MARGIN s and its green-lap MAE is not worse by > GUARDRAIL s
  * "SGD frozen" = SGD that stopped learning after the 2023 warm-up: what no online learning costs

Run (after the stream exists):
    py -3.14 _online_s1.py all
    py -3.14 _drift_lab.py --web "D:\\Works\\F1 web"
Out: drift_lab.json (page data), model/prod/web_predictions.parquet + model/prod/meta.json (_web_api.py).
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("F1_ONLINE_YEARS", "2023,2024,2025,2026")
import _online as on   # noqa: E402  (reads F1_ONLINE_YEARS)

ROOT = Path(__file__).resolve().parent
# stage-2 stream to replay: the fully incremental pipeline of _online_s1.py (no batch model in stage 1)
STREAM = Path(os.environ.get("F1_STREAM_DIR", ROOT / "online" / "fast" / "s2_incremental"))
OUT = ROOT / "drift_lab.json"
PROD = ROOT / "model" / "prod"
MODELS = {"SGD": "Linear SGD, Huber loss (river)", "ARF": "Adaptive Random Forest + ADWIN (CapyMOA / MOA)",
          "kNN": "kNN, k=15, last 2,000 sectors (numpy, = river kNN)"}
CHAMPION0 = "SGD"
N_LAST_2025 = 3
SIM_YEAR = 2026
PSI_TRIGGER = 0.25
MAE_TRIGGER = 1.25
COOLDOWN = 2
PROMOTE_MARGIN = 0.010
GUARDRAIL = 0.005
DRIFT_FEATURES = {
    "Prev_DRS_Pct": "DRS open %",
    "Prev_Max_Speed": "Max speed (km/h)",
    "Prev_Avg_Speed_Straight": "Straight speed (km/h)",
    "Prev_Avg_Gear": "Average gear",
    "Prev_Full_Throttle_Pct": "Full throttle %",
}   # weather is not monitored: a race has only a few readings, so its PSI is noise
KEY = ["Session_Key", "Driver", "Lap", "Sector"]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.0f}s] {msg}", flush=True)


def psi(ref, cur, bins=10):
    ref, cur = ref[np.isfinite(ref)], cur[np.isfinite(cur)]
    if len(ref) < 50 or len(cur) < 50:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        lo, hi = min(ref.min(), cur.min()), max(ref.max(), cur.max())
        edges = np.linspace(lo, hi if hi > lo else lo + 1, bins + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    return round(float(np.sum((c - r) * np.log((c + 1e-4) / (r + 1e-4)))), 3)


def hist3(ref, prev, cur, bins=12):
    vals = np.concatenate([x[np.isfinite(x)] for x in (ref, prev, cur) if x is not None and len(x)])
    lo, hi = np.quantile(vals, 0.01), np.quantile(vals, 0.99)
    hi = hi if hi > lo else lo + 1
    edges = np.linspace(lo, hi, bins + 1)

    def frac(x):
        if x is None or not len(x):
            return None
        x = np.clip(x[np.isfinite(x)], lo, hi)
        return [round(float(v), 4) for v in np.histogram(x, edges)[0] / max(len(x), 1)]
    return {"edges": [round(float(e), 2) for e in edges], "ref": frac(ref), "prev": frac(prev), "cur": frac(cur)}


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def mae(err, green):
    return {"mae": round(float(err.mean()), 4),
            "green_mae": round(float(err[green].mean()), 4) if green.any() else None, "n": int(len(err))}


def _name_circuits(races):
    try:
        from f1pit.api import fetch
        ses = pd.concat([fetch("sessions", year=int(y), session_name="Race") for y in on.YEARS], ignore_index=True)
        names = dict(zip(ses["session_key"].astype(int), ses["circuit_short_name"]))
        key_name = {}
        for sk, key in zip(races["Session_Key"], races["circuit"]):
            if int(sk) in names:
                key_name.setdefault(key, names[int(sk)])
        races["circuit"] = races["circuit"].map(lambda k: key_name.get(k, k))
    except Exception as exc:
        log(f"OpenF1 names unavailable ({exc.__class__.__name__}); circuit = country + lap length")
    return races


def load_stream():
    s = pd.read_parquet(STREAM / "stream_index.parquet")
    for name in MODELS:
        z = np.load(STREAM / f"preds_{name}.npz")
        s[f"pred_{name}"] = s["anchor"] * np.exp(z["online"])
        s[f"frozen_{name}"] = s["anchor"] * np.exp(z["static"])
        if name == CHAMPION0:
            s["pred_frozen"] = s[f"frozen_{name}"]
    fr = pd.concat([d.assign(Year=y) for y, d in on.load_frames().items()], ignore_index=True)
    feats = fr[[*KEY, "Year", *[f for f in DRIFT_FEATURES if f in fr.columns]]].copy()
    if "Prev_DRS_Pct" in feats:      # 2026 has no DRS: OpenF1 sends no value -> the flap is never open (0 %)
        feats["Prev_DRS_Pct"] = feats["Prev_DRS_Pct"].fillna(0.0)
    for c in KEY:
        feats[c] = feats[c].astype("int64")
        s[c] = s[c].astype("int64")
    meta = pd.concat([pd.read_parquet(ROOT / f"data_{y}.parquet",
                                      columns=["Session_Key", "Country", "Circuit_Total_Length", "Time_Start"])
                      for y in on.YEARS], ignore_index=True)
    races = meta.groupby("Session_Key").agg(country=("Country", "first"), length=("Circuit_Total_Length", "first"),
                                            t=("Time_Start", "min")).reset_index()
    races["date"] = pd.to_datetime(races["t"], utc=True).dt.strftime("%Y-%m-%d")
    races["circuit"] = (races["country"] + " "
                        + (pd.to_numeric(races["length"], errors="coerce") / 1000).round(1).astype(str) + " km")
    return s, feats, _name_circuits(races)


def main(web=None):
    s, feats, races = load_stream()
    s = s.merge(races[["Session_Key", "country", "circuit", "date"]], on="Session_Key", how="left")
    feats = feats.merge(races[["Session_Key", "circuit", "date"]], on="Session_Key", how="left")
    order = (races.merge(s.groupby("Session_Key")["Year"].first().reset_index(), on="Session_Key")
             .sort_values("date").reset_index(drop=True))
    sim_keys = [*order[order["Year"] == 2025]["Session_Key"].tail(N_LAST_2025),
                *order[order["Year"] == SIM_YEAR]["Session_Key"]]
    log(f"stream {len(s):,} rows, {s['Session_Key'].nunique()} races; simulating {len(sim_keys)}")

    s["green"] = s["Track_Status"].astype(str).str.upper().eq("GREEN")
    if "clean" not in s.columns:
        s["clean"] = True
    sc = s[s["clean"]]           # every metric / decision uses clean rows; the web store keeps every row
    first_date = order.loc[order["Session_Key"] == sim_keys[0], "date"].iloc[0]
    pre25 = sc[(sc["Year"] == 2025) & (sc["date"] < first_date)]
    base = float(np.abs(pre25[f"pred_{CHAMPION0}"] - pre25["actual"]).mean())
    log(f"champion {CHAMPION0} MAE on 2025 before the simulation: {base:.4f} s -> trigger {MAE_TRIGGER * base:.3f}")

    b2425 = sc[sc["Year"].isin([2024, 2025])]
    bench = {m: (float(np.abs(b2425[f"pred_{m}"] - b2425["actual"]).mean()),
                 float(np.abs(b2425[f"frozen_{m}"] - b2425["actual"]).mean())) for m in MODELS}
    champion, version = CHAMPION0, "v001"
    lineage = [{"version": "v001", "model": CHAMPION0, "algorithm": MODELS[CHAMPION0],
                "since": "2024 (after the 2023 warm-up)",
                "reason": f"best online model on 2024-2025 (MAE {bench[CHAMPION0][0]:.3f} s)"}]
    served = {int(k): (CHAMPION0, "v001") for k in order["Session_Key"] if k not in sim_keys}
    races_out, events, last_event = [], [], -99
    for i, sk in enumerate(sim_keys):
        cur = sc[sc["Session_Key"] == sk]
        info = order[order["Session_Key"] == sk].iloc[0]
        served[int(sk)] = (champion, version)
        green = cur["green"].to_numpy()
        res = {m: mae(np.abs(cur[f"pred_{m}"] - cur["actual"]).to_numpy(), green) for m in MODELS}
        frozen = mae(np.abs(cur["pred_frozen"] - cur["actual"]).to_numpy(), green)
        champ = res[champion]

        fcur = feats[feats["Session_Key"] == sk]
        ref = feats[(feats["circuit"] == info["circuit"]) & (feats["date"] < info["date"])]
        ref_label = f"{info['circuit']} in earlier seasons"
        if len(ref) < 500:
            ref = feats[(feats["Year"] == 2025) & (feats["date"] < info["date"])]
            ref_label = "all earlier 2025 races (no earlier race at this circuit)"
        prev_key = order[order["date"] < info["date"]]["Session_Key"].iloc[-1]
        fprev = feats[feats["Session_Key"] == prev_key]
        psis, hists = {}, {}
        for f in DRIFT_FEATURES:
            if f in fcur.columns:
                psis[f] = psi(ref[f].to_numpy(float), fcur[f].to_numpy(float))
                hists[f] = hist3(ref[f].to_numpy(float), fprev[f].to_numpy(float), fcur[f].to_numpy(float))
        psi_max = max((v for v in psis.values() if v is not None), default=0.0)
        drift = "MAJOR" if psi_max >= PSI_TRIGGER else "MINOR" if psi_max >= 0.1 else "STABLE"
        reason = ("feature drift" if psi_max >= PSI_TRIGGER
                  else "performance" if champ["mae"] >= MAE_TRIGGER * base else None)
        trigger = reason is not None and i - last_event >= COOLDOWN
        decision, before, before_model = ("NO TRIGGER" if reason is None else "COOLDOWN"), version, champion
        if trigger:
            last_event = i
            others = sorted((m for m in MODELS if m != champion), key=lambda m: res[m]["mae"])
            best = others[0]
            d = round(res[best]["mae"] - champ["mae"], 4)
            g_ok = (res[best]["green_mae"] is None or champ["green_mae"] is None
                    or res[best]["green_mae"] - champ["green_mae"] <= GUARDRAIL)
            promote = d <= -PROMOTE_MARGIN and g_ok
            decision = "PROMOTE" if promote else "KEEP"
            new = f"v{len(lineage) + 1:03d}" if promote else None
            events.append({"race": i, "trigger": reason, "champion": f"{version} {champion}", "winner": best,
                           "champion_mae": champ["mae"], "winner_mae": res[best]["mae"], "delta_mae": d,
                           "guardrail_ok": bool(g_ok), "decision": f"PROMOTE -> {new}" if promote else "KEEP",
                           "challengers": [{"model": f"{version} {champion} (champion)", **champ}]
                           + [{"model": m, **res[m]} for m in others]
                           + [{"model": "SGD frozen (no learning)", **frozen}]})
            if promote:
                champion, version = best, new
                lineage.append({"version": new, "model": best, "algorithm": MODELS[best],
                                "since": f"after {info['date'][:4]} {info['circuit']}",
                                "reason": f"{reason}: dMAE {d:+.3f} s, guardrail ok"})
        log(f"{i + 1:2d} {info['date']} {info['circuit']:<20s} {before} {before_model} {champ['mae']:.3f} | "
            f"SGD {res['SGD']['mae']:.3f} ARF {res['ARF']['mae']:.3f} kNN {res['kNN']['mae']:.3f} "
            f"frozen {frozen['mae']:.3f} | PSI {psi_max:.2f} {drift} {decision}")
        races_out.append({"step": i + 1, "session_key": int(sk), "year": int(info["Year"]), "date": info["date"],
                          "country": info["country"], "circuit": info["circuit"], "rows": champ["n"],
                          "champion": before, "champion_model": before_model, "champion_after": version,
                          "champion_after_model": champion, "mae": champ["mae"], "green_mae": champ["green_mae"],
                          "models": {m: res[m]["mae"] for m in MODELS}, "frozen_mae": frozen["mae"],
                          "psi": psis, "psi_max": psi_max, "drift": drift, "reference": ref_label,
                          "trigger": bool(trigger), "decision": decision, "hist": hists})

    # ---- web store: every streamed sector with the prediction of the model serving at that race ----
    s["Model"] = s["Session_Key"].map(lambda k: served[int(k)][0])
    s["Version"] = s["Session_Key"].map(lambda k: served[int(k)][1])
    s["Pred_Duration_s"] = np.select([s["Model"] == m for m in MODELS], [s[f"pred_{m}"] for m in MODELS], np.nan)
    sim = s[s["Session_Key"].isin(sim_keys) & s["clean"]]
    live_mae = float(np.abs(sim["Pred_Duration_s"] - sim["actual"]).mean())
    all_mae = float(np.abs(s.loc[s["clean"], "Pred_Duration_s"] - s.loc[s["clean"], "actual"]).mean())
    PROD.mkdir(parents=True, exist_ok=True)
    keep = [*KEY, "Year", "Time_Start", "Track_Status", "anchor", "actual", "Pred_Duration_s", "Model", "Version",
            *[f"pred_{m}" for m in MODELS], "pred_frozen", "clean"]
    s[keep].to_parquet(PROD / "web_predictions.parquet", index=False)
    sess = (s.groupby("Session_Key").agg(year=("Year", "first"), country=("country", "first"),
                                         circuit=("circuit", "first"), date=("date", "first"),
                                         model=("Model", "first"), version=("Version", "first"))
            .reset_index().rename(columns={"Session_Key": "session_key"}))
    sess["simulated"] = sess["session_key"].isin(sim_keys)
    (PROD / "meta.json").write_text(json.dumps(_clean({
        "family": "online (river): SGD / ARF / kNN, predict-then-learn every 10 laps",
        "warm_up": "2023", "stream": f"2024-{on.YEARS[-1]}", "champion": version, "champion_model": champion,
        "lineage": lineage, "live_mae_sim": round(live_mae, 4), "live_mae_all": round(all_mae, 4),
        "sessions": sess.to_dict("records")}), indent=1, default=str), encoding="utf-8")
    log(f"web store -> {PROD}: {len(s):,} sectors, served MAE {all_mae:.4f} (2024-) / {live_mae:.4f} (simulated)")

    out = {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "family": "online", "models": MODELS, "base_mae": round(base, 4), "live_mae": round(live_mae, 4),
        "live_mae_all": round(all_mae, 4),
        "rules": {"psi_trigger": PSI_TRIGGER, "mae_trigger": round(MAE_TRIGGER * base, 3), "cooldown": COOLDOWN,
                  "promote_margin": PROMOTE_MARGIN, "guardrail": GUARDRAIL, "chunk_laps": on.CHUNK_LAPS},
        "features": DRIFT_FEATURES, "races": races_out, "events": events, "champions": lineage,
        "benchmark": [{"algorithm": f"{m} online", "kind": MODELS[m],
                       "mae": round(bench[m][0], 4), "frozen": round(bench[m][1], 4)} for m in ("SGD", "ARF", "kNN")],
        "pipeline": "fully incremental: stage 1 online (SGD / ARF / kNN champion / challenger, no batch model) "
                    "-> stage 2 online",
        "data": {"races_total": int(s["Session_Key"].nunique()), "rows_total": int(len(s)),
                 "warm_up": "2023", "stream": f"2024-{on.YEARS[-1]}",
                 "skipped_2026": ""},
    }
    payload = json.dumps(_clean(out), indent=1, default=str)
    OUT.write_text(payload, encoding="utf-8")
    # same data as a script, so the page also works when opened from disk (file:// blocks fetch of JSON)
    OUT.with_suffix(".js").write_text(f"window.DRIFT_LAB = {payload};\n", encoding="utf-8")
    log(f"saved {OUT.name} + {OUT.with_suffix('.js').name}: {len(races_out)} races, {len(events)} decisions, "
        f"{len(lineage) - 1} promotions")
    if web:
        for p in (OUT, OUT.with_suffix(".js")):
            shutil.copy(p, Path(web) / p.name)
        log(f"copied to {web}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--web", help="folder of the F1 web page (copies drift_lab.json there)")
    main(ap.parse_args().web)
