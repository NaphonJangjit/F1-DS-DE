"""EDA figures for the slides (CSS 342 Ch4 EDA + Ch6 Data Visualization), in the deck theme.

Data: 2024 + 2025 races (data_<year>.parquet), Track_Status re-derived with the red-flag-aware parser,
pace context from f1pit.pace_context (representative-lap anchor).

  figures/eda_race.png          Data visualization — one race at a glance (line chart: lap time, pits, SC)
  figures/eda_univariate.png    Univariate — lap-to-lap change: histogram + skewness/kurtosis, box plot by track status
  figures/eda_bivariate.png     Bivariate — tyre age vs pace by compound; last clean lap vs this lap (r)
  figures/eda_multivariate.png  Multivariate — Spearman correlation heatmap of the main drivers of lap time

Run:  python _eda_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from f1pit.features import repair_track_status
from f1pit.pace_context import add_pace_context

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
YEARS = (2024, 2025)
RED, DARK, GREY, LIGHT = "#C8282D", "#151515", "#8C8C8C", "#F3D3D4"
TYRE_COLORS = {"SOFT": "#DA291C", "MEDIUM": "#E6B800", "HARD": "#7A7A7A", "INTERMEDIATE": "#43B02A", "WET": "#0067AD"}
SC = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 17,
                     "axes.titlelocation": "left", "axes.titlecolor": DARK})


def load():
    parts = []
    for y in YEARS:
        df, _ = repair_track_status(pd.read_parquet(ROOT / f"data_{y}.parquet"))
        df["Year"] = y
        parts.append(add_pace_context(df))
    d = pd.concat(parts, ignore_index=True)
    d["status"] = d["Track_Status"].astype(str).str.upper()
    d["change_pct"] = (d["Duration"] / d["Prev_Duration"] - 1) * 100       # vs same sector, previous lap
    d["vs_rep_pct"] = (d["Duration"] / d["Rep_Prev_Duration"] - 1) * 100   # vs last representative lap
    return d


def save(fig, name):
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / name, dpi=200, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("wrote", FIG / name, flush=True)


def race_at_a_glance(d):
    lap = (d.groupby(["Session_Key", "Country", "Year", "Driver", "Lap"])
           .agg(t=("Duration", "sum"), n=("Sector", "nunique"), sc=("status", lambda s: s.isin(SC).any()),
                rain=("Rainfall", "max"), tyre=("Tyre", "first"), stops=("Pit_Stops_Done", "first"))
           .reset_index())
    lap = lap[lap.n == 3]
    # a dry 2025 race with a short SC period
    cand = (lap[lap.Year == 2025].groupby(["Session_Key", "Country"])
            .agg(sc_laps=("sc", lambda s: lap.loc[s.index[s], "Lap"].nunique()), rain=("rain", "max"))
            .reset_index())
    cand = cand[(cand.rain == 0) & cand.sc_laps.between(3, 8)]
    sk, country = cand.iloc[0][["Session_Key", "Country"]]
    r = lap[lap.Session_Key == sk]
    drivers = r.groupby("Driver")["Lap"].count().sort_values(ascending=False).index[:3]
    fig, ax = plt.subplots(figsize=(14, 6))
    sc_laps = sorted(r.loc[r.sc, "Lap"].unique())
    for L in sc_laps:
        ax.axvspan(L - 0.5, L + 0.5, color=LIGHT, lw=0)
    top = r["t"].max() * 1.04
    pit_laps = {}
    for drv, ls in zip(drivers, ["-", "--", ":"]):
        x = r[r.Driver == drv].sort_values("Lap")
        ax.plot(x.Lap, x.t, color=DARK, ls=ls, lw=2, label=f"#{int(drv)}")
        for _, s in x.groupby("stops"):
            ax.scatter(s.Lap, s.t, s=20, color=TYRE_COLORS.get(s.tyre.iloc[0], GREY), zorder=3)
        for L in x.loc[x.stops.diff() > 0, "Lap"]:          # first lap on new tyres -> in-lap is L-1
            t_in = x.loc[x.Lap == L - 1, "t"]
            if len(t_in):
                pit_laps.setdefault(int(L - 1), []).append(float(t_in.iloc[0]))
    for L, ts in pit_laps.items():                           # one label per lap, above the highest in-lap
        label = "PIT" if len(ts) == 1 else f"PIT x{len(ts)}"
        ax.annotate(label, (L, max(ts)), xytext=(0, 8), textcoords="offset points",
                    ha="center", color=RED, fontsize=12, fontweight="bold")
    if sc_laps:
        ax.text(np.mean(sc_laps), top, "SAFETY CAR", ha="center", va="top", color=RED, fontweight="bold")
    ax.set_ylim(r["t"].min() - 1.5, top + 1)
    ax.set_xlabel("Lap"); ax.set_ylabel("Lap time (s)")
    ax.set_title(f"One race at a glance - {country} 2025")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=t.title(), markersize=9)
               for t, c in TYRE_COLORS.items() if t in set(r.tyre)]
    leg1 = ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, title="Driver")
    ax.add_artist(leg1)
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 0.55), frameon=False, title="Tyre")
    save(fig, "eda_race.png")


def univariate(d):
    x = d["change_pct"].dropna()
    x = x[np.isfinite(x)]
    sk, ku = stats.skew(x), stats.kurtosis(x)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1.35, 1]})
    inside = x[(x >= -15) & (x <= 30)]
    a1.hist(inside, bins=120, color=RED, alpha=0.9)
    a1.set_yscale("log")
    a1.set_xlabel("Sector time vs same sector last lap (%)"); a1.set_ylabel("Number of sectors (log scale)")
    a1.set_title("Distribution: most laps repeat, a few explode")
    a1.text(0.03, 0.95, f"{(1 - len(inside) / len(x)) * 100:.1f}% of sectors\nare beyond -15% / +30%",
            transform=a1.transAxes, ha="left", va="top", fontsize=12, color=GREY)
    a1.text(0.97, 0.95, f"median {x.median():+.2f}%\nskewness {sk:+.1f}\nkurtosis {ku:.0f}",
            transform=a1.transAxes, ha="right", va="top", fontsize=15, fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": RED})
    groups = {"Green": d.loc[d.status == "GREEN", "change_pct"],
              "SC / VSC": d.loc[d.status.isin(SC), "change_pct"],
              "Lap after\npit stop": d.loc[d.Prev_Lap_Pitted == 1, "change_pct"]}
    bp = a2.boxplot([g.dropna() for g in groups.values()], tick_labels=list(groups), showfliers=False,
                    patch_artist=True, widths=0.55, medianprops={"color": DARK, "linewidth": 2})
    for p_, c in zip(bp["boxes"], [GREY, RED, DARK]):
        p_.set_facecolor(c); p_.set_alpha(0.8)
    a2.axhline(0, color=GREY, lw=1, ls="--")
    a2.set_ylabel("Change vs last lap (%)")
    a2.set_title("Box plot by situation")
    fig.tight_layout()
    save(fig, "eda_univariate.png")


def bivariate(d):
    g = d[(d.status == "GREEN") & (d.Rainfall == 0) & (d.Prev_Lap_Pitted == 0)
          & d.Tyre.isin(["SOFT", "MEDIUM", "HARD"])].copy()
    lap = (g.groupby(["Session_Key", "Driver", "Lap"])
           .agg(t=("Duration", "sum"), n=("Sector", "nunique"), tyre=("Tyre", "first"),
                age=("Tyre_Age", "first"), stint=("Pit_Stops_Done", "first")).reset_index())
    lap = lap[lap.n == 3]
    base = lap[lap.age.between(3, 5)].groupby(["Session_Key", "Driver", "stint"])["t"].median().rename("t0")
    lap = lap.join(base, on=["Session_Key", "Driver", "stint"]).dropna(subset=["t0"])
    lap["rel"] = (lap.t / lap.t0 - 1) * 100
    lap = lap[lap.rel.abs() < 3]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.8))
    a1.text(0.02, 0.03, "Medium / Hard get faster: fuel burns off faster than the tyre wears",
            transform=a1.transAxes, fontsize=11.5, color=GREY)
    for t in ["SOFT", "MEDIUM", "HARD"]:
        s = lap[(lap.tyre == t) & lap.age.between(3, 35)].groupby("age")["rel"].agg(["mean", "sem", "count"])
        s = s[s["count"] >= 30]
        a1.plot(s.index, s["mean"], color=TYRE_COLORS[t], lw=3, label=t.title())
        a1.fill_between(s.index, s["mean"] - 1.96 * s["sem"], s["mean"] + 1.96 * s["sem"],
                        color=TYRE_COLORS[t], alpha=0.2, lw=0)
    a1.axhline(0, color=GREY, lw=1, ls="--")
    a1.set_xlabel("Tyre age (laps)"); a1.set_ylabel("Lap time vs laps 3-5 of the stint (%)")
    a1.set_title("Tyre age vs pace (dry, green)")
    a1.legend(frameon=False)
    s2 = d[(d.status == "GREEN") & d.Rep_Prev_Duration.notna()].sample(6000, random_state=1)
    r = stats.pearsonr(s2.Rep_Prev_Duration, s2.Duration)[0]
    a2.scatter(s2.Rep_Prev_Duration, s2.Duration, s=6, alpha=0.35, color=RED)
    lim = [s2.Duration.min(), s2.Duration.quantile(0.995)]
    a2.plot(lim, lim, color=DARK, lw=1.5, ls="--")
    a2.set_xlim(lim); a2.set_ylim(lim)
    a2.set_xlabel("Last clean lap, same sector (s)"); a2.set_ylabel("This lap (s)")
    a2.set_title(f"Last clean lap vs this lap: r = {r:.3f}")
    fig.tight_layout()
    save(fig, "eda_bivariate.png")


def multivariate(d):
    cols = {"vs_rep_pct": "Lap time vs last clean lap", "Tyre_Age": "Tyre age", "Distance_On_Tyre": "Distance on tyre",
            "Total_Distance_Driven": "Race distance (fuel)", "Track_Temperature": "Track temp",
            "Humidity": "Humidity", "Rainfall": "Rain", "Traffic_Within_1s": "Cars within 1 s",
            "Interval_Ahead": "Gap to car ahead", "Laps_Since_SC": "Laps since SC", "Prev_Lap_NonGreen": "Last lap not green",
            "Anchor_Age": "Laps since clean lap"}
    m = d[list(cols)].apply(pd.to_numeric, errors="coerce").rename(columns=cols).corr(method="spearman")
    cmap = LinearSegmentedColormap.from_list("rw", ["#2B2B2B", "white", RED])
    fig, ax = plt.subplots(figsize=(12, 9.5))
    im = ax.imshow(m, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(m)), m.columns, rotation=45, ha="right", fontsize=13)
    ax.set_yticks(range(len(m)), m.columns, fontsize=13)
    for i in range(len(m)):
        for j in range(len(m)):
            v = m.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                    color="white" if abs(v) > 0.6 else DARK)
    ax.spines[:].set_visible(False)
    ax.set_title("Spearman correlation of the main lap-time drivers")
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    save(fig, "eda_multivariate.png")


def main():
    d = load()
    race_at_a_glance(d)
    univariate(d)
    bivariate(d)
    multivariate(d)


if __name__ == "__main__":
    main()
