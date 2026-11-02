"""Theme + helper สำหรับกราฟ (dark theme เดิมจาก utils.py)

ไม่เปลี่ยน rcParams ตอน import แล้ว — เรียก apply_theme() เองเมื่อต้องการ

>>> from f1pit import plotting
>>> plotting.apply_theme()
>>> plotting.fmt_lap(92.345)     # '1:32.345'
"""

import pandas as pd
from matplotlib.ticker import FuncFormatter

BG, PANEL, FG, GRID, MUTED = "#0f1117", "#171a23", "#e8eaf0", "#2a2f3d", "#8b92a5"

COMPOUND_COLORS = {
    "SOFT": "#ff3b3b", "MEDIUM": "#ffd60a", "HARD": "#f2f2f2",
    "INTERMEDIATE": "#39d353", "WET": "#3b9dff", "UNKNOWN": "#8b92a5",
}

THEME = {
    "figure.facecolor": BG, "savefig.facecolor": BG, "figure.dpi": 110,
    "axes.facecolor": PANEL, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
    "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": FG, "axes.titlepad": 14, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.alpha": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": FG,
    "font.size": 11, "legend.frameon": False, "legend.labelcolor": FG,
    "lines.solid_capstyle": "round",
}

# None = ไม่ zoom; เช่น 1.07 = ตัดแกนที่ 107% ของค่าอ้างอิง
OUTLIER_FACTOR = None


def apply_theme():
    import matplotlib.pyplot as plt
    plt.rcParams.update(THEME)


def fmt_lap(seconds, nd=3):
    """92.345 -> '1:32.345'"""
    if seconds is None or pd.isna(seconds):
        return "–"
    m, sec = divmod(float(seconds), 60)
    return f"{int(m)}:{sec:0{nd + 3}.{nd}f}"


lap_axis = FuncFormatter(lambda v, _: fmt_lap(v, 1))


def cap_axis(ax, values, ref, axis="y", factor=None):
    """ตัดแกนที่ ref * factor และเขียนบอกว่ามีกี่ค่าที่หลุดกรอบ"""
    factor = factor if factor is not None else OUTLIER_FACTOR
    if factor is None:
        return
    cap = ref * factor
    n = int((pd.Series(values) > cap).sum())
    ax.set_ylim(top=cap) if axis == "y" else ax.set_xlim(right=cap)
    if n:
        ax.text(0.005, 0.97, f"zoomed: {n} slower values off-scale", transform=ax.transAxes,
                fontsize=9, color=MUTED, va="top")


def show(fig, tight=True, save_path=None):
    """tight_layout แล้ว show (หรือ save ถ้าระบุ save_path)"""
    import matplotlib.pyplot as plt
    if tight:
        fig.tight_layout()
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()
