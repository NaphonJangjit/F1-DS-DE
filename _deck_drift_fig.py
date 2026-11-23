"""Slide figure: does the online model survive the 2026 rule change?  -> figures/deck_drift_2026.png
Reads drift_lab.json (py -3.14 _drift_lab.py)."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT / "drift_lab.json").read_text(encoding="utf-8"))
R = D["races"]
RED, DARK, GREY, BLUE = "#C8282D", "#151515", "#8C8C8C", "#2F6DB5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "axes.titlesize": 17,
                     "axes.titlelocation": "left"})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 6.3), gridspec_kw={"width_ratios": [2.1, 1]})
x = np.arange(1, len(R) + 1)
served = np.array([r["mae"] for r in R])
frozen = np.array([r["frozen_mae"] for r in R])
a1.plot(x, frozen, color=GREY, ls="--", lw=2.2, marker="o", ms=4, label="SGD frozen after 2023 (no learning)")
a1.plot(x, served, color=RED, lw=3, marker="o", ms=5, label="Online model in use (learns every 10 laps)")
n25 = sum(r["year"] == 2025 for r in R)
a1.axvline(n25 + 0.5, color=DARK, lw=1)
top = max(frozen.max(), served.max()) * 1.12
a1.set_ylim(0, top)
a1.text(n25 / 2 + 0.5, top * 0.97, "end of 2025", ha="center", va="top", color=GREY, fontsize=14, fontweight="bold")
a1.text(n25 + (len(R) - n25) / 2 + 0.5, top * 0.97, "2026: new engines, no DRS", ha="center", va="top",
        color=RED, fontsize=14, fontweight="bold")
k = 0
for i, r in enumerate(R):
    if r["decision"] == "PROMOTE":
        a1.scatter([i + 1], [served[i]], s=170, marker="D", color=DARK, zorder=5)
        a1.annotate(f"{r['circuit']}: {r['champion_after_model']} promoted ({r['champion_after']})", (i + 1, served[i]),
                    xytext=(i - 2.5, served[i] + top * (0.20 + 0.09 * k)), fontsize=13,
                    arrowprops={"arrowstyle": "-", "color": DARK})
        k += 1
mz = int(np.argmax(frozen))
a1.annotate(f"{R[mz]['circuit']}: {frozen[mz]:.2f} s frozen\nvs {served[mz]:.2f} s online", (mz + 1, frozen[mz]),
            xytext=(mz - 3.6, frozen[mz] * 0.93), fontsize=13, color=DARK,
            arrowprops={"arrowstyle": "-", "color": GREY})
a1.set_xticks(x)
a1.set_xticklabels([r["circuit"].replace("Yas Marina Circuit", "Yas Marina").replace("Spa-Francorchamps", "Spa")
                    for r in R], rotation=55, ha="right", fontsize=11.5)
a1.set_ylabel("MAE per sector (s)")
a1.set_title("Race by race: the model keeps learning through the rule change")
a1.legend(frameon=False, fontsize=13, loc="upper left", bbox_to_anchor=(0.0, 0.9))

r25 = [r for r in R if r["year"] == 2025]
r26 = [r for r in R if r["year"] == 2026]
vals = [(np.mean([r["frozen_mae"] for r in r25]), np.mean([r["mae"] for r in r25])),
        (np.mean([r["frozen_mae"] for r in r26]), np.mean([r["mae"] for r in r26]))]
w = 0.36
for j, (f, s) in enumerate(vals):
    a2.bar(j - w / 2, f, w, color=GREY)
    a2.bar(j + w / 2, s, w, color=RED)
    a2.text(j - w / 2, f + 0.006, f"{f:.3f}", ha="center", fontsize=13)
    a2.text(j + w / 2, s + 0.006, f"{s:.3f}", ha="center", fontsize=13, fontweight="bold", color=RED)
gain = (vals[1][0] - vals[1][1]) / vals[1][0] * 100
a2.text(1, max(vals[1]) * 1.13, f"{gain:.0f}% less error", ha="center", fontsize=15, fontweight="bold", color=RED)
a2.set_xticks([0, 1])
a2.set_xticklabels([f"Last {len(r25)} GPs of 2025", f"{len(r26)} GPs of 2026"])
a2.set_ylim(0, max(max(v) for v in vals) * 1.3)
a2.set_ylabel("Average MAE per sector (s)")
a2.set_title("Average error")
fig.tight_layout()
out = ROOT / "figures" / "deck_drift_2026.png"
fig.savefig(out, dpi=110)
print(out, f"2025 {vals[0]}, 2026 {vals[1]}, gain {gain:.1f}%")
