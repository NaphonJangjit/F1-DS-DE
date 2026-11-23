"""Hypothesis tests beyond the paired t-test (CSS 342 Ch8): one-way ANOVA + Chi-square independence.

Data: 2024 + 2025 races (data_<year>.parquet), Track_Status re-derived with the red-flag-aware parser
(f1pit.features.repair_track_status, race_control from the .cache/openf1 disk cache).

1) One-way ANOVA — "Do SOFT, MEDIUM and HARD tyres degrade at the same rate?"
   Unit = one stint (session, driver, stint) on a dry, green-flag run of >= 8 clean laps.
   Degradation = OLS slope of lap time vs tyre age, as % of the stint's median lap per lap
   (scale-free across circuits; includes the ~same fuel-burn effect for every compound).
   H0: mean degradation is equal for the 3 compounds.  H1: at least one differs.
   + Levene (equal variances), Welch-type Alexander-Govern, Kruskal-Wallis (robust), Tukey HSD.

2) Chi-square test of independence — "Do drivers pit more often under a Safety Car / VSC?"
   Unit = one driver-lap. Pitted = this lap is the in-lap (next lap starts from the pit lane).
   Track = lap had SC/VSC in any sector vs fully green.
   H0: pitting is independent of track status.  H1: they are associated.  Effect size: Cramer's V.

Run:  python _anova_chi2.py   -> hypothesis_extra.json + figures/hyp_anova.png + figures/hyp_chi2.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from f1pit.features import repair_track_status

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
YEARS = (2024, 2025)
SC = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}
RED, DARK, GREY = "#C8282D", "#151515", "#8C8C8C"
TYRE_COLORS = {"SOFT": "#DA291C", "MEDIUM": "#F2C200", "HARD": "#9E9E9E"}


def load():
    parts = []
    for y in YEARS:
        df, n = repair_track_status(pd.read_parquet(ROOT / f"data_{y}.parquet"))
        print(f"{y}: {len(df):,} rows, Track_Status repaired on {n:,} rows", flush=True)
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def lap_table(df):
    """sector rows -> one row per (session, driver, lap) with lap time + flags."""
    st = df["Track_Status"].astype(str).str.upper()
    d = df.assign(_sc=st.isin(SC).astype(int), _green=(st == "GREEN").astype(int))
    g = d.groupby(["Session_Key", "Driver", "Lap"])
    lap = g.agg(n_sec=("Sector", "nunique"), lap_time=("Duration", "sum"), sc=("_sc", "max"),
                green=("_green", "min"), rain=("Rainfall", "max"), tyre=("Tyre", "first"),
                tyre_age=("Tyre_Age", "first"), stops=("Pit_Stops_Done", "first"),
                prev_pitted=("Prev_Lap_Pitted", "max")).reset_index()
    s1 = d[d["Sector"] == 1][["Session_Key", "Driver", "Lap", "Will_Box"]]
    s1 = s1.assign(Lap=s1["Lap"] - 1).rename(columns={"Will_Box": "in_lap"})   # next lap starts in pit lane
    lap = lap.merge(s1, on=["Session_Key", "Driver", "Lap"], how="left")
    lap["in_lap"] = lap["in_lap"].fillna(0).astype(int)
    return lap


def degradation(lap):
    ok = ((lap.n_sec == 3) & (lap.green == 1) & (lap.rain == 0) & (lap.in_lap == 0) & (lap.prev_pitted == 0)
          & lap.tyre.isin(TYRE_COLORS))
    clean = lap[ok].copy()
    rows = []
    for (sk, drv, stops), s in clean.groupby(["Session_Key", "Driver", "stops"]):
        if s["tyre"].nunique() != 1:
            continue
        med = s["lap_time"].median()
        s = s[s["lap_time"] <= 1.03 * med]            # drop traffic / mistake laps inside the stint
        if len(s) < 8 or s["tyre_age"].nunique() < 8:
            continue
        slope = np.polyfit(s["tyre_age"], s["lap_time"], 1)[0]
        rows.append({"session": sk, "driver": drv, "tyre": s["tyre"].iloc[0], "laps": len(s),
                     "deg_pct_per_lap": slope / med * 100, "deg_s_per_lap": slope})
    return pd.DataFrame(rows)


def anova(stints):
    groups = {t: stints.loc[stints.tyre == t, "deg_pct_per_lap"].to_numpy() for t in TYRE_COLORS}
    f, p = stats.f_oneway(*groups.values())
    lev = stats.levene(*groups.values())
    ag = stats.alexandergovern(*groups.values())
    kw = stats.kruskal(*groups.values())
    allv = np.concatenate(list(groups.values()))
    ss_between = sum(len(v) * (v.mean() - allv.mean()) ** 2 for v in groups.values())
    eta2 = ss_between / ((allv - allv.mean()) ** 2).sum()
    tk = stats.tukey_hsd(*groups.values())
    names = list(groups)
    pairs = {f"{names[i]}-{names[j]}": {"diff": round(float(groups[names[i]].mean() - groups[names[j]].mean()), 4),
                                        "p": float(tk.pvalue[i, j])}
             for i in range(3) for j in range(i + 1, 3)}
    desc = {t: {"n": int(len(v)), "mean": round(float(v.mean()), 4), "sd": round(float(v.std(ddof=1)), 4),
                "ci95": [round(float(x), 4) for x in stats.t.interval(0.95, len(v) - 1, v.mean(), stats.sem(v))]}
            for t, v in groups.items()}
    return {"question": "Do SOFT / MEDIUM / HARD degrade at the same rate (% lap time per lap)?",
            "groups": desc, "F": float(f), "p": float(p), "eta_squared": float(eta2),
            "levene_p": float(lev.pvalue), "alexander_govern_p": float(ag.pvalue), "kruskal_p": float(kw.pvalue),
            "tukey": pairs, "decision_alpha_0.05": "reject H0" if p < 0.05 else "fail to reject H0"}, groups


def chi_square(lap):
    d = lap[(lap.n_sec == 3)].copy()
    tab = pd.crosstab(np.where(d.sc == 1, "SC/VSC lap", "Green lap"), np.where(d.in_lap == 1, "pitted", "no pit"))
    chi2, p, dof, exp = stats.chi2_contingency(tab)
    n = tab.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(tab.shape) - 1)))
    rate = (tab["pitted"] / tab.sum(axis=1) * 100).round(2).to_dict()
    a, b = tab.loc["SC/VSC lap", "pitted"], tab.loc["SC/VSC lap", "no pit"]
    c, dd = tab.loc["Green lap", "pitted"], tab.loc["Green lap", "no pit"]
    return {"question": "Is pitting independent of SC/VSC?", "table": tab.to_dict(),
            "expected": pd.DataFrame(exp, index=tab.index, columns=tab.columns).round(1).to_dict(),
            "chi2": float(chi2), "dof": int(dof), "p": float(p), "cramers_v": float(v),
            "pit_rate_pct": rate, "odds_ratio": float((a * dd) / (b * c)),
            "decision_alpha_0.05": "reject H0" if p < 0.05 else "fail to reject H0"}, tab


def plots(groups, an, tab, ch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False,
                         "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    data = [groups[t] for t in TYRE_COLORS]
    bp = ax.boxplot(data, tick_labels=list(TYRE_COLORS), widths=0.55, showfliers=False, patch_artist=True,
                    medianprops={"color": DARK, "linewidth": 2})
    for patch, t in zip(bp["boxes"], TYRE_COLORS):
        patch.set_facecolor(TYRE_COLORS[t]); patch.set_alpha(0.85); patch.set_edgecolor(DARK)
    top = max(w.get_ydata().max() for w in bp["whiskers"])
    for i, t in enumerate(TYRE_COLORS, 1):
        m = groups[t].mean()
        ax.scatter([i], [m], color="white", edgecolor=DARK, zorder=3, s=70)
        ax.text(i, top * 1.08, f"mean {m:+.3f}%", ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_ylim(top=top * 1.25)
    ax.axhline(0, color=GREY, lw=1, ls="--")
    ax.set_ylabel("Lap time change per lap of tyre age (%)")
    ax.set_title(f"One-way ANOVA: F = {an['F']:.1f}, p = {an['p']:.1e}  ->  {an['decision_alpha_0.05']}",
                 fontsize=15, fontweight="bold", color=RED, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "hyp_anova.png", transparent=False, facecolor="white"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    labels = ["Green lap", "SC/VSC lap"]
    vals = [ch["pit_rate_pct"][k] for k in labels]
    bars = ax.bar(labels, vals, color=[GREY, RED], width=0.55)
    for b_, v_, k in zip(bars, vals, labels):
        n_pit = int(tab.loc[k, "pitted"]); n_all = int(tab.loc[k].sum())
        ax.text(b_.get_x() + b_.get_width() / 2, v_, f"{v_:.1f}%\n({n_pit:,} / {n_all:,} laps)",
                ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_ylabel("Laps that end with a pit stop (%)")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title(f"Chi-square: chi2 = {ch['chi2']:.0f}, p = {ch['p']:.1e}, Cramer's V = {ch['cramers_v']:.2f}",
                 fontsize=15, fontweight="bold", color=RED, loc="left")
    fig.tight_layout(); fig.savefig(FIG / "hyp_chi2.png", facecolor="white"); plt.close(fig)


def main():
    df = load()
    lap = lap_table(df)
    stints = degradation(lap)
    an, groups = anova(stints)
    ch, tab = chi_square(lap)
    out = {"years": list(YEARS), "anova": an, "chi_square": ch}
    (ROOT / "hypothesis_extra.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    plots(groups, an, tab, ch)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
