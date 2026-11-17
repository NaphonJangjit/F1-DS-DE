"""Per-track MAE (cond-anchor model, 2025) — horizontal bar chart, ranked high->low."""

import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features
from _cond_anchor import cond_anchors

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
WET_RACES = {"United Kingdom", "Australia"}
DRY_C = "#4C78A8"
WET_C = "#F58518"


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def main():
    t0 = time.time()
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ca, tr_feats, _, _ = _prepare_lab(cond_anchors(tr), [])
    te_ca, te_feats, _, _ = _prepare_lab(cond_anchors(te), [])
    feats = [c for c in _features(tr_feats, tr_ca) if c in te_ca.columns]

    pre = build_preprocessor(feats).fit(tr_ca[feats])
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(pre.transform(tr_ca[feats]),
                                                    tr_ca[TARGET].to_numpy(float))
    p = rf.predict(pre.transform(te_ca[feats]))
    yte = te_ca[TARGET].to_numpy(float)
    print(f"trained cond-anchor ({time.time()-t0:.0f}s)")

    country = te.groupby("Session_Key")["Country"].first().to_dict()
    per = []
    for sk in np.unique(te_ca["Session_Key"].to_numpy()):
        m = te_ca["Session_Key"].to_numpy() == sk
        per.append((str(country.get(int(sk), sk)), sec_mae(yte[m], p[m])))
    per.sort(key=lambda r: -r[1])  # descending

    names = [r[0] for r in per][::-1]
    maes = [r[1] for r in per][::-1]
    colors = [WET_C if n in WET_RACES else DRY_C for n in names]

    fig, ax = plt.subplots(figsize=(9.5, 9))
    ax.barh(names, maes, color=colors, height=0.66)
    ax.set_xlabel("MAE (seconds)")
    ax.set_title("Per-track MAE — condition-aware-anchor model (2025)", loc="left", fontsize=13, pad=12)
    ax.xaxis.grid(True, color="#e3e6ea", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_xlim(0, max(maes) * 1.08)

    for i, m in enumerate(maes):
        ax.text(m + 0.012, i, f"{m:.3f}", va="center", fontsize=8, color="#333")

    # legend via annotation (two races are wet)
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=WET_C, label="wet race"),
                       mpatches.Patch(color=DRY_C, label="dry race")],
              loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    out = ROOT / "track_mae_cond_anchor.png"
    fig.savefig(out, dpi=150)
    print(f"saved -> {out.name} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
