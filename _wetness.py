"""Track-wetness features from race_control (DRS + "TRACK SURFACE SLIPPERY") + rain-risk.

The binary Rainfall flag misses "rained before the race" and the drying curve. race_control
gives three better signals, all point-in-time:
  - "TRACK SURFACE SLIPPERY IN TRACK SECTOR N"  -> official "surface is wet/slippery"
  - "DRS DISABLED" / "DRS ENABLED"              -> official wet <-> dry decision (also SC/VSC)
  - "RISK OF RAIN FOR F1 RACE IS X%"            -> pre-race rain probability

Built features:
  Surface_Slippery         1 if a "SLIPPERY" message within the last 10 min
  DRS_Disabled             1 while DRS is disabled (as-of)
  Wet_Official             Surface_Slippery OR (DRS_Disabled AND Track_Status green)
  Rain_Risk                pre-race rain % (constant per session)
  Laps_Since_Slippery      laps since the most recent "SLIPPERY" message
  Field_Laps_Since_Slippery  = Laps_Since_Slippery x n_drivers  (racing-line drying: more cars -> drier line)

SCRATCH.
"""

import re
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from f1pit.api import fetch, to_utc
from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}
SLIP_WINDOW = pd.Timedelta(minutes=10)

WET_COLS = ["Surface_Slippery", "DRS_Disabled", "Wet_Official", "Rain_Risk",
            "Laps_Since_Slippery", "Field_Laps_Since_Slippery"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def green_mask(df):
    d = df[["Session_Key", "Driver", "Lap", "Track_Status"]].copy()
    is_sc = d["Track_Status"].isin(SC_SET).to_numpy()
    laps = (d.assign(_is_sc=is_sc)
            .groupby(["Session_Key", "Driver", "Lap"], sort=False)["_is_sc"].max()
            .reset_index().sort_values(["Session_Key", "Driver", "Lap"]))
    laps["_prev_sc"] = laps.groupby(["Session_Key", "Driver"])["_is_sc"].shift(1).fillna(False)
    merged = d[["Session_Key", "Driver", "Lap"]].merge(
        laps[["Session_Key", "Driver", "Lap", "_prev_sc"]],
        on=["Session_Key", "Driver", "Lap"], how="left")
    return ~(is_sc | merged["_prev_sc"].to_numpy().astype(bool))


def build_rc_cache(session_keys):
    cache = {}
    for sk in sorted(session_keys):
        rc = fetch("race_control", session_key=int(sk))
        if rc.empty:
            cache[int(sk)] = None
            continue
        rc = rc.copy()
        rc["date"] = to_utc(rc["date"]).dt.tz_localize(None)
        rc = rc.sort_values("date")
        drs = []          # (time, disabled 0/1)
        slippery = []     # times
        rain_risk = None
        for _, row in rc.iterrows():
            msg = str(row.get("message", "")).upper()
            ts = row["date"]
            if "DRS DISABLED" in msg and "ZONE" not in msg:
                drs.append((ts, 1))
            elif "DRS ENABLED" in msg and "ZONE" not in msg:
                drs.append((ts, 0))
            if "TRACK SURFACE SLIPPERY" in msg:
                slippery.append(ts)
            m = re.search(r"RISK OF RAIN FOR F1 RACE IS (\d+)", msg)
            if m and rain_risk is None:
                rain_risk = int(m.group(1))
        drs_df = pd.DataFrame(drs, columns=["time", "drs_disabled"]) if drs else pd.DataFrame(columns=["time", "drs_disabled"])
        slip_df = pd.DataFrame({"time": pd.to_datetime(slippery)}) if slippery else pd.DataFrame(columns=["time"])
        cache[int(sk)] = {"drs": drs_df, "slippery": slip_df, "rain_risk": rain_risk}
    return cache


def add_wetness(raw, rc_cache):
    df = raw.copy()
    for col in WET_COLS:
        df[col] = np.nan

    # lap start times per session (min Time_Start per lap) — tz-naive for searchsorted
    lap_time = (df.groupby(["Session_Key", "Lap"])["Time_Start"].min().reset_index()
                .sort_values(["Session_Key", "Lap"]))
    lap_time["Time_Start"] = pd.to_datetime(lap_time["Time_Start"], utc=True).dt.tz_localize(None)

    n_drivers = df.groupby("Session_Key")["Driver"].nunique().to_dict()

    for sk in df["Session_Key"].unique():
        sk = int(sk)
        info = rc_cache.get(sk)
        if info is None:
            continue
        m = df["Session_Key"].to_numpy() == sk
        ts = pd.to_datetime(df.loc[m, "Time_Start"], utc=True).dt.tz_localize(None)

        # Rain_Risk (constant)
        if info["rain_risk"] is not None:
            df.loc[m, "Rain_Risk"] = info["rain_risk"]

        # DRS_Disabled as-of (backward state)
        drs = info["drs"]
        if not drs.empty:
            drs = drs.sort_values("time")
            t = pd.DataFrame({"Time_Start": ts})
            t["_ord"] = np.arange(len(t))
            t = t.sort_values("Time_Start")
            mm = pd.merge_asof(t, drs, left_on="Time_Start", right_on="time", direction="backward")
            mm = mm.sort_values("_ord")
            df.loc[m, "DRS_Disabled"] = mm["drs_disabled"].to_numpy(float)

        # Surface_Slippery: last slippery time within window
        sl = info["slippery"]
        if not sl.empty:
            sl = sl.sort_values("time")
            t = pd.DataFrame({"Time_Start": ts})
            t["_ord"] = np.arange(len(t))
            t = t.sort_values("Time_Start")
            mm = pd.merge_asof(t, sl, left_on="Time_Start", right_on="time", direction="backward")
            mm = mm.sort_values("_ord")
            since = t["Time_Start"].to_numpy() - mm["time"].to_numpy()
            df.loc[m, "Surface_Slippery"] = (pd.to_timedelta(since) <= SLIP_WINDOW).astype(float)

            # laps since last slippery: map last slippery time -> lap via session lap times
            lt = lap_time[lap_time["Session_Key"] == sk]
            if not lt.empty:
                slip_laps = []
                for st in sl["time"]:
                    idx = lt["Time_Start"].searchsorted(st) - 1
                    slip_laps.append(int(lt.iloc[max(0, idx)]["Lap"]))
                last_slip_lap = max(slip_laps)
                df.loc[m, "Laps_Since_Slippery"] = (df.loc[m, "Lap"].to_numpy(float) - last_slip_lap).clip(min=0)
                df.loc[m, "Field_Laps_Since_Slippery"] = df.loc[m, "Laps_Since_Slippery"] * n_drivers.get(sk, 20)

    # Wet_Official = Surface_Slippery OR (DRS_Disabled AND green status)
    green = df["Track_Status"].isin(["GREEN", "UNKNOWN"]).to_numpy()
    df["Wet_Official"] = ((df["Surface_Slippery"].fillna(0) == 1)
                          | ((df["DRS_Disabled"].fillna(0) == 1) & green)).astype(float)
    return df


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    log("building race_control cache ...")
    rc_cache = build_rc_cache(set(tr["Session_Key"]) | set(te["Session_Key"]))
    log(f"cached {len(rc_cache)} sessions")

    log("adding wetness features ...")
    tr = add_wetness(tr, rc_cache)
    te = add_wetness(te, rc_cache)
    # coverage
    for name, d in [("train", tr), ("test", te)]:
        log(f"  {name}: Surface_Slippery=1 {(d['Surface_Slippery']==1).mean()*100:.1f}%  "
            f"DRS_Disabled=1 {(d['DRS_Disabled']==1).mean()*100:.1f}%  "
            f"Wet_Official=1 {(d['Wet_Official']==1).mean()*100:.1f}%  "
            f"Rain_Risk non-null {d['Rain_Risk'].notna().mean()*100:.1f}%")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, WET_COLS)
    te_ml, te_feats, _, _ = _prepare_lab(te, WET_COLS)
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    feats_base = [c for c in feats if c not in WET_COLS]
    log(f"features base={len(feats_base)} +wetness={len(feats)}")

    te_g = green_mask(te_ml)
    te_wet = te_ml["Rainfall"].to_numpy() == 1
    sil = te_ml["Session_Key"].to_numpy() == 9947

    results = {}
    for name, cols in [("baseline", feats_base), ("+wetness", feats)]:
        log(f"training {name} ...")
        t = time.time()
        pre = build_preprocessor(cols).fit(tr_ml[cols])
        rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                   min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                   random_state=42).fit(pre.transform(tr_ml[cols]),
                                                        tr_ml[TARGET].to_numpy(float))
        p = rf.predict(pre.transform(te_ml[cols]))
        yte = te_ml[TARGET].to_numpy(float)
        m = {"full": sec_mae(yte, p), "green": sec_mae(yte[te_g], p[te_g]),
             "wet": sec_mae(yte[te_wet], p[te_wet]), "dry": sec_mae(yte[~te_wet], p[~te_wet]),
             "silverstone": sec_mae(yte[sil], p[sil])}
        results[name] = m
        log(f"  {name:10s} full={m['full']:.4f} green={m['green']:.4f} wet={m['wet']:.4f} "
            f"dry={m['dry']:.4f} silverstone={m['silverstone']:.4f} ({time.time()-t:.0f}s)")

    print("\n" + "=" * 62)
    print(f"{'config':10s} {'full':>8s} {'green':>8s} {'wet':>8s} {'dry':>8s} {'silver':>8s}")
    print("-" * 62)
    for name, m in results.items():
        print(f"{name:10s} {m['full']:8.4f} {m['green']:8.4f} {m['wet']:8.4f} {m['dry']:8.4f} {m['silverstone']:8.4f}")
    print("=" * 62)


if __name__ == "__main__":
    main()
