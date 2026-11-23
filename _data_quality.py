"""Data quality report (CSS 342 Ch2 "Data Quality Checks" + Ch3 "Data Preparation").

Checks the built lap-sector tables data_<year>.parquet on the 5 checks taught in class:
  1. Missing values          — % missing per column
  2. Duplicate records       — duplicate (Session_Key, Driver, Lap, Sector) keys + full-row duplicates
  3. Accuracy & validity     — value ranges / allowed categories, and a pace check of SC/VSC labels
                               (a lap labelled SC should be clearly slower than the driver's clean pace)
  4. Outlier detection       — IQR rule (Ch3) and the MAD rule used by the pipeline, per session x sector
  5. Consistency & uniques   — category values, corner-cluster columns per year, same circuit across years

Run from the project root:
    python _data_quality.py          -> data_quality_report.md + data_quality.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
YEARS = (2023, 2024, 2025)
KEY = ["Session_Key", "Driver", "Lap", "Sector"]
COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
STATUSES = {"GREEN", "SC", "VSC", "SC_ENDING", "VSC_ENDING", "UNKNOWN"}
RANGES = {   # column: (min, max) physically plausible
    "Duration": (5, 200), "Lap": (1, 90), "Sector": (1, 3), "Tyre_Age": (1, 80), "Tyre_Age_Total": (1, 90),
    "Track_Temperature": (0, 70), "Air_Temperature": (0, 50), "Humidity": (0, 100), "Pressure": (700, 1100),
    "Rainfall": (0, 1), "Prev_Avg_Speed_Corners": (0, 380), "Prev_Avg_Speed_Straight": (0, 380),
    "Prev_Max_Speed": (0, 380), "Prev_Min_Speed": (0, 380), "Prev_Full_Throttle_Pct": (0, 100),
    "Prev_Brake_Pct": (0, 100), "Prev_DRS_Pct": (0, 100), "Prev_Avg_Gear": (0, 8), "Interval_Ahead": (0, 200),
}


def check_year(df):
    r = {"rows": int(len(df)), "columns": int(df.shape[1]), "sessions": int(df["Session_Key"].nunique()),
         "drivers": int(df["Driver"].nunique())}

    # 1. missing
    miss = df.isna().mean().sort_values(ascending=False)
    r["missing_pct"] = {c: round(float(v) * 100, 2) for c, v in miss.items() if v > 0}

    # 2. duplicates
    r["duplicate_keys"] = int(df.duplicated(KEY).sum())
    r["duplicate_rows"] = int(df.duplicated().sum())

    # 3. validity
    bad = {}
    for c, (lo, hi) in RANGES.items():
        if c in df.columns:
            v = pd.to_numeric(df[c], errors="coerce")
            n = int(((v < lo) | (v > hi)).sum())
            if n:
                bad[c] = n
    bad["Tyre_not_in_compounds"] = int((~df["Tyre"].astype(str).str.upper().isin(COMPOUNDS)).sum())
    bad["Track_Status_unknown_value"] = int((~df["Track_Status"].astype(str).str.upper().isin(STATUSES)).sum())
    r["out_of_range"] = {k: v for k, v in bad.items() if v}

    # SC/VSC label vs pace: SC laps are ~30-40% slower; a label on a full-pace lap is suspicious
    st = df["Track_Status"].astype(str).str.upper()
    ratio = pd.to_numeric(df["Duration"], errors="coerce") / pd.to_numeric(df["Clean_Pace"], errors="coerce")
    sc = st.isin(["SC", "VSC"])
    share = sc.groupby(df["Session_Key"]).mean()
    pace = ratio[sc].groupby(df.loc[sc, "Session_Key"]).median()
    suspicious = [int(s) for s in share.index if share[s] > 0.25 and pace.get(s, 9) < 1.03]
    r["sc_vsc_rows"] = int(sc.sum())
    r["sc_vsc_median_pace_ratio"] = round(float(ratio[sc].median()), 3) if sc.any() else None
    r["sessions_sc_at_full_pace"] = suspicious
    r["rows_in_those_sessions_labelled_sc"] = int((sc & df["Session_Key"].isin(suspicious)).sum())

    # 4. outliers (Duration within session x sector)
    g = df.groupby(["Session_Key", "Sector"])["Duration"]
    q1, q3 = g.transform(lambda s: s.quantile(0.25)), g.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    r["outliers_iqr_high"] = int((df["Duration"] > q3 + 1.5 * iqr).sum())
    r["outliers_iqr_low"] = int((df["Duration"] < q1 - 1.5 * iqr).sum())
    med = g.transform("median")
    mad = (df["Duration"] - med).abs().groupby([df["Session_Key"], df["Sector"]]).transform("median")
    r["outliers_mad_rule_removed"] = int((df["Duration"] > med + 4 * mad.clip(lower=0.5)).sum())

    # 5. consistency / unique values
    r["unique_tyre"] = sorted(df["Tyre"].astype(str).unique().tolist())
    r["unique_track_status"] = sorted(df["Track_Status"].astype(str).unique().tolist())
    r["num_corner_columns"] = int(sum(c.startswith("Num_Corner_") for c in df.columns))
    return r


def corner_consistency(frames):
    """Same circuit + sector, different years -> are the Num_Corner_k counts the same?"""
    rows = []
    for y, d in frames.items():
        nc = [c for c in d.columns if c.startswith("Num_Corner_")]
        first = d.groupby(["Country", "Sector"])[nc].first()
        for (country, sector), v in first.iterrows():
            rows.append({"year": y, "country": country, "sector": int(sector),
                         "profile": tuple(int(x) for x in v.fillna(0).to_numpy())})
    t = pd.DataFrame(rows)
    both = t.groupby(["country", "sector"])["year"].nunique()
    multi = both[both > 1].index
    same = sum(t.set_index(["country", "sector"]).loc[[k]]["profile"].nunique() == 1 for k in multi)
    return {"circuit_sectors_in_2plus_years": int(len(multi)), "identical_corner_profile": int(same)}


def to_markdown(rep):
    L = ["# Data quality report", "", "Source: `data_<year>.parquet` (built lap-sector tables).", ""]
    L += ["| check | " + " | ".join(str(y) for y in YEARS) + " |", "|---|" + "---|" * len(YEARS)]

    def row(label, f):
        L.append(f"| {label} | " + " | ".join(str(f(rep[str(y)])) for y in YEARS) + " |")
    row("rows", lambda r: f"{r['rows']:,}")
    row("sessions / drivers", lambda r: f"{r['sessions']} / {r['drivers']}")
    row("1. columns with missing values", lambda r: len(r["missing_pct"]))
    row("   worst column (% missing)", lambda r: next(iter(r["missing_pct"].items()), "-"))
    row("2. duplicate keys / rows", lambda r: f"{r['duplicate_keys']} / {r['duplicate_rows']}")
    row("3. out-of-range values", lambda r: sum(r["out_of_range"].values()))
    row("   SC/VSC rows (median Duration/Clean_Pace)", lambda r: f"{r['sc_vsc_rows']:,} ({r['sc_vsc_median_pace_ratio']})")
    row("   sessions labelled SC at full race pace", lambda r: r["sessions_sc_at_full_pace"] or "-")
    row("4. outliers IQR high / low", lambda r: f"{r['outliers_iqr_high']:,} / {r['outliers_iqr_low']:,}")
    row("   removed by MAD rule (median + 4 MAD)", lambda r: f"{r['outliers_mad_rule_removed']:,}")
    row("5. Num_Corner_* columns", lambda r: r["num_corner_columns"])
    c = rep["corner_consistency"]
    L += ["", f"Corner-cluster consistency: of {c['circuit_sectors_in_2plus_years']} circuit-sectors that appear in "
          f"2+ years, only {c['identical_corner_profile']} keep the same `Num_Corner_*` profile -> cluster ids are "
          "re-fitted per year and are not comparable across years.", ""]
    for y in YEARS:
        r = rep[str(y)]
        L += [f"## {y}", "", f"- missing (%): {r['missing_pct'] or 'none'}",
              f"- out of range: {r['out_of_range'] or 'none'}",
              f"- Tyre values: {r['unique_tyre']}", f"- Track_Status values: {r['unique_track_status']}", ""]
    return "\n".join(L)


def main():
    frames = {y: pd.read_parquet(ROOT / f"data_{y}.parquet") for y in YEARS}
    rep = {str(y): check_year(d) for y, d in frames.items()}
    rep["corner_consistency"] = corner_consistency(frames)
    (ROOT / "data_quality.json").write_text(json.dumps(rep, indent=2, default=str), encoding="utf-8")
    (ROOT / "data_quality_report.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep))


if __name__ == "__main__":
    main()
