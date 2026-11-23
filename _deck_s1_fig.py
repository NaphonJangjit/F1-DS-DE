"""Slide figure: a fully incremental pipeline (no batch model in stage 1) -> figures/deck_stage1_online.png
Reads online_s1_results.json + online_s1_promote.json (py -3.14 _online_s1.py all)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
R = json.loads((ROOT / "online_s1_results.json").read_text(encoding="utf-8"))
P = json.loads((ROOT / "online_s1_promote.json").read_text(encoding="utf-8"))
RED, DARK, GREY = "#C8282D", "#151515", "#8C8C8C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 17,
                     "axes.titlelocation": "left"})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 6.3), gridspec_kw={"width_ratios": [1.25, 1]})

# left: stage-1 telemetry score per season (1.0 = "same as last lap")
years = ["2023", "2024", "2025", "2026"]
sc = P["score_by_year"]
cands = [("SGD online", "SGD", "#2F6DB5"), ("kNN online", "kNN", "#8E7CC3"), ("ARF online", "ARF", "#E0A030"),
         ("served", "champion in use", RED)]
w = 0.2
x = np.arange(len(years))
CAP = 2.0
for j, (k, lab, c) in enumerate(cands):
    v = [min(sc[y][k], CAP) for y in years]
    bars = a1.bar(x + (j - 1.5) * w, v, w, color=c, label=lab)
    for b_, y in zip(bars, years):
        raw = sc[y][k]
        a1.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.03,
                f"{raw:.2f}" if raw < 10 else f"{raw:.0f}", ha="center", fontsize=11,
                color=RED if k == "served" else "#444444", fontweight="bold" if k == "served" else "normal")
last = [sc[y]["same as last lap"] for y in years]
for i, v in enumerate(last):
    a1.plot([i - 2 * w, i + 2 * w], [v, v], color=DARK, ls="--", lw=1.6, zorder=0,
            label="same as last lap" if i == 0 else None)
a1.set_xticks(x, years)
a1.set_ylim(0, CAP + 0.35)
a1.set_ylabel("Telemetry error (1.0 = last lap, 2023)")
a1.set_title("Stage 1: online telemetry models, from an empty start")
a1.legend(frameon=False, fontsize=12, ncol=3, loc="upper left")
lin = P["lineage"]
a1.text(0.0, -0.17, f"Champion / challenger: {len(lin) - 1} promotions, mostly in 2023-24 while the models learn; "
        f"{lin[-1]['model'].split()[0]} serves from {lin[-1]['since'].replace('after ', '')} on.",
        transform=a1.transAxes, fontsize=12.5, color=DARK)

# right: stage-2 sector error on 2026, learning vs frozen
mae = R["mae_s"]
models = ["SGD", "kNN", "ARF"]
on_ = [mae[f"{m} online"]["2026"] for m in models]
fr_ = [mae[f"{m} frozen"]["2026"] for m in models]
y2 = np.arange(len(models))
a2.barh(y2 - 0.18, on_, 0.34, color=RED, label="learns every 10 laps")
a2.barh(y2 + 0.18, fr_, 0.34, color=GREY, label="frozen after 2023")
for i in range(len(models)):
    a2.text(on_[i] + 0.004, y2[i] - 0.18, f"{on_[i]:.3f} s", va="center", fontsize=13, color=RED, fontweight="bold")
    a2.text(fr_[i] + 0.004, y2[i] + 0.18, f"{fr_[i]:.3f} s", va="center", fontsize=13)
a2.set_yticks(y2, [f"{m}" for m in models])
a2.invert_yaxis()
a2.set_xlim(0, max(on_ + fr_) * 1.35)
a2.set_xlabel("Sector error on 2026 (MAE, s)")
a2.set_title("Stage 2: sector error, 2026")
a2.legend(frameon=False, fontsize=12.5, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=2)
fig.tight_layout()
out = ROOT / "figures" / "deck_stage1_online.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print(out)
