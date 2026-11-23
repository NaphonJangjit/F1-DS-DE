"""Deck figure: Data Engineering table (same style as figures/deck_sources.png).

    py -3.14 _deck_de_fig.py   ->  figures/deck_de_table.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RED, GREY = "#C8282E", "#F3F3F3"
OUT = Path(__file__).resolve().parent / "figures" / "deck_de_table.png"

HEAD = ("Stage", "What we built", "Data Engineering idea")
ROWS = [
    ("Ingest", "one fetch(): retry x6, exponential back-off on 429 / 5xx", "reliable, rate-limit-safe batch pull"),
    ("Cache", "memory + disk: 2,006 API responses (1.7 GB raw)", "idempotent: re-runs never re-download"),
    ("Transform", "6 steps, UTC timestamps, as-of joins (backward)", "point-in-time correct, no leakage"),
    ("Quality", "5 checks, red-flag repair fixes data in place", "validate before storing"),
    ("Store", "1 Parquet per season: 288,510 rows, 17 MB", "columnar, partitioned by season"),
    ("Stream", "replay races in order, micro-batch of 10 laps", "predict first, then learn"),
    ("Serve", "FastAPI /predict /race: 217,079 saved predictions", "serving layer, reloads on new data"),
]
COLS = (0.0, 0.14, 0.6, 1.0)           # column edges (fractions of width)


def main():
    n = len(ROWS) + 1
    fig = plt.figure(figsize=(28.5, 0.62 * n + 0.3), dpi=100)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_xlim(0, 1); ax.set_ylim(n, 0); ax.axis("off")
    for i in range(n):
        y0 = i
        if i == 0:
            ax.add_patch(plt.Rectangle((0, y0), 1, 1, color=RED, lw=0))
        elif i % 2 == 1:
            ax.add_patch(plt.Rectangle((0, y0), 1, 1, color=GREY, lw=0))
        cells = HEAD if i == 0 else ROWS[i - 1]
        for j, txt in enumerate(cells):
            if i == 0:
                ax.text((COLS[j] + COLS[j + 1]) / 2, y0 + 0.5, txt, ha="center", va="center",
                        color="white", fontsize=27, fontweight="bold")
            else:
                ax.text(COLS[j] + 0.012, y0 + 0.5, txt, ha="left", va="center", color="black",
                        fontsize=26, fontweight="bold" if j == 0 else "normal")
    for x in COLS[1:-1]:
        ax.plot([x, x], [0, n], color="white", lw=3)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print(OUT)


if __name__ == "__main__":
    main()
