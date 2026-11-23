"""Deck figure: how the sector table is joined and transformed (f1pit/features.py).

    py -3.14 _deck_join_fig.py   ->  figures/deck_joins.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

RED, GREY, DARK, MUTED = "#C8282E", "#F3F3F3", "#181818", "#9A9A9A"
OUT = Path(__file__).resolve().parent / "figures" / "deck_joins.png"

HEAD = ("Table", "Join", "Rule", "Adds to the sector row")
ROWS = [
    ("laps", "split", "lap start + earlier sector times = Time_Start", "1 row per driver × lap × sector"),
    ("stints", "range join", "lap_start ≤ lap ≤ lap_end", "compound, tyre age"),
    ("car_data", "window + lag", "average inside the sector, use the previous lap", "8 telemetry values"),
    ("race_control", "last event", "last SC / VSC / green message before Time_Start", "track status"),
    ("weather", "as-of join", "latest reading before Time_Start, max 2 min old", "temperature, humidity, rain"),
    ("intervals", "as-of join", "latest gap before Time_Start", "gap ahead, cars within 1–5 s"),
    ("pit", "lag", "pit on lap N is known from lap N+1", "pit flags, stops done"),
    ("tracks.json", "key join", "circuit + sector number", "sector length, straights, corners"),
]
COLS = (0.0, 0.12, 0.24, 0.66, 1.0)


def table(ax):
    n = len(ROWS) + 1
    ax.set_xlim(0, 1); ax.set_ylim(n, 0); ax.axis("off")
    for i in range(n):
        if i == 0:
            ax.add_patch(plt.Rectangle((0, i), 1, 1, color=RED, lw=0))
        elif i % 2 == 1:
            ax.add_patch(plt.Rectangle((0, i), 1, 1, color=GREY, lw=0))
        cells = HEAD if i == 0 else ROWS[i - 1]
        for j, txt in enumerate(cells):
            if i == 0:
                ax.text((COLS[j] + COLS[j + 1]) / 2, i + 0.5, txt, ha="center", va="center",
                        color="white", fontsize=26, fontweight="bold")
            else:
                ax.text(COLS[j] + 0.01, i + 0.5, txt, ha="left", va="center", fontsize=24,
                        color=RED if j == 1 else "black", fontweight="bold" if j < 2 else "normal",
                        family="monospace" if j == 0 else None)
    for x in COLS[1:-1]:
        ax.plot([x, x], [0, n], color="white", lw=3)


def asof(ax):
    """as-of example: weather every minute, sector starts at 14:02:40 -> takes 14:02"""
    ax.set_xlim(-0.3, 4.3); ax.set_ylim(-1.2, 1.5); ax.axis("off")
    ax.set_title("as-of join example (weather)", loc="left", fontsize=26, fontweight="bold", pad=6)
    ax.plot([0, 4], [0, 0], color=DARK, lw=3)
    for k in range(5):
        future = k > 2
        ax.plot(k, 0, "o", ms=20, color=MUTED if future else DARK)
        ax.text(k, -0.55, f"14:0{k}", ha="center", fontsize=22, color=MUTED if future else DARK)
    t = 2.67
    ax.plot([t, t], [-0.25, 1.0], color=RED, lw=4)
    ax.text(t, 1.12, "sector starts 14:02:40", ha="center", fontsize=22, color=RED, fontweight="bold")
    ax.annotate("", xy=(2.05, 0.18), xytext=(t - 0.05, 0.55),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=3, mutation_scale=30))
    ax.text(2.0, 0.62, "uses 14:02", ha="right", fontsize=22, color=RED)
    ax.text(3.5, 0.35, "14:03 = future, never used", ha="center", fontsize=20, color=MUTED)


def steps(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("after the joins", loc="left", fontsize=26, fontweight="bold", pad=6)
    boxes = [("anchor", "last clean lap,\nsame sector"),
             ("target", "ln(time / anchor)"),
             ("concat", "all drivers × races\n→ 1 Parquet / season")]
    w, gap = 0.29, 0.065
    for k, (head, body) in enumerate(boxes):
        x = k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.08), w, 0.78, boxstyle="round,pad=0,rounding_size=0.05",
                                    color=RED if k == 1 else DARK, lw=0, mutation_aspect=0.4))
        ax.text(x + w / 2, 0.66, head, ha="center", va="center", color="white", fontsize=26,
                fontweight="bold", style="italic")
        ax.text(x + w / 2, 0.34, body, ha="center", va="center", color="white", fontsize=21)
        if k < 2:
            ax.annotate("", xy=(x + w + gap - 0.008, 0.47), xytext=(x + w + 0.008, 0.47),
                        arrowprops=dict(arrowstyle="-|>", color=RED, lw=3, mutation_scale=26))


def main():
    fig = plt.figure(figsize=(28.5, 10.6), dpi=100)
    table(fig.add_axes([0.01, 0.33, 0.98, 0.66]))
    asof(fig.add_axes([0.01, 0.02, 0.46, 0.27]))
    steps(fig.add_axes([0.52, 0.02, 0.47, 0.27]))
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
