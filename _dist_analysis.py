"""Distribution analysis of the sub-forests: skew + kurtosis + histograms (2025 test).

Uses the saved models (model/v2) — no retrain.
Outputs: dist_analysis.csv, dist_residual.png, dist_target.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

import matplotlib.pyplot as plt

from f1pit.wind_rain_lab import _prepare_lab
from f1pit import plotting
import _predict as P
import _fforest as ff

ROOT = Path(__file__).resolve().parent

FRIENDLY = {
    "Outcome_Avg_Speed_Corners": "Avg speed (corners)",
    "Outcome_Avg_Speed_Straight": "Avg speed (straight)",
    "Outcome_Full_Throttle_Pct": "Full throttle %",
    "Outcome_Brake_Pct": "Brake %",
    "Outcome_DRS_Pct": "DRS %",
    "Outcome_Max_Speed": "Max speed",
    "Outcome_Min_Speed": "Min speed",
    "Outcome_Avg_Gear": "Avg gear",
}
ACCENT = "#F58518"
MUTED = "#8b92a5"


def _stats(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(skew(x)), float(kurtosis(x, fisher=True))


def main():
    meta, subs, main = P.load_models()
    raw = pd.read_parquet(ROOT / "data_2025_out.parquet")
    df, _, _, _ = _prepare_lab(raw, meta["extra"])
    df = ff._engineer(df)

    data = {}
    for prev, out in meta["channels"]:
        m = subs[out]
        pred = m["rf"].predict(m["pre"].transform(df[m["feats"]]))
        if m["variant"] == "delta":
            pred = df[prev].to_numpy(float) + pred
        elif m["variant"] == "delta_slog":
            pred = df[prev].to_numpy(float) + ff._slog_inv(pred)
        elif m["variant"] == "log":
            pred = np.exp(pred)
        y = df[out].to_numpy(float)
        pv = df[prev].to_numpy(float)
        mask = np.isfinite(y) & np.isfinite(pv) & np.isfinite(pred)
        y, pv, pred = y[mask], pv[mask], pred[mask]
        data[out] = {"y": y, "prev": pv, "pred": pred, "delta": y - pv, "resid": y - pred}

    # ---- skew / kurtosis table ----
    rows = []
    for prev, out in meta["channels"]:
        d = data[out]
        ts, tk = _stats(d["y"])
        ds, dk = _stats(d["delta"])
        rs, rk = _stats(d["resid"])
        rows.append({
            "channel": FRIENDLY.get(out, out),
            "target_skew": round(ts, 3), "target_kurt": round(tk, 3),
            "delta_skew": round(ds, 3), "delta_kurt": round(dk, 3),
            "resid_skew": round(rs, 3), "resid_kurt": round(rk, 3),
            "mae": round(float(np.abs(d["resid"]).mean()), 4),
        })
    table = pd.DataFrame(rows)
    print("Skew / kurtosis per sub-forest (2025). Kurtosis = excess (0 = normal).")
    print(table.to_string(index=False))
    table.to_csv(ROOT / "dist_analysis.csv", index=False)

    # ---- plots ----
    plotting.apply_theme()
    channels = [o for _, o in meta["channels"]]

    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
    axes = axes.ravel()
    for ax, out in zip(axes, channels):
        d = data[out]
        x = d["resid"]
        ax.hist(x, bins=60, density=True, alpha=0.65, color=ACCENT)
        mu, sd = x.mean(), x.std()
        xs = np.linspace(x.min(), x.max(), 300)
        ax.plot(xs, norm.pdf(xs, mu, sd), color=MUTED, lw=1.6)
        rs, rk = _stats(x)
        ax.set_title(f"{FRIENDLY.get(out, out)}  |  skew={rs:.2f}  kurt={rk:.2f}", fontsize=10)
    fig.suptitle("Sub-forest residual distribution (actual − predicted), 2025 test", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(ROOT / "dist_residual.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(13, 15))
    axes = axes.ravel()
    for ax, out in zip(axes, channels):
        d = data[out]
        x = d["y"]
        ax.hist(x, bins=60, density=True, alpha=0.65, color=ACCENT)
        ts, tk = _stats(x)
        ax.set_title(f"{FRIENDLY.get(out, out)}  |  skew={ts:.2f}  kurt={tk:.2f}", fontsize=10)
    fig.suptitle("Sub-forest target distribution (current telemetry), 2025 test", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(ROOT / "dist_target.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved -> dist_residual.png, dist_target.png, dist_analysis.csv")


if __name__ == "__main__":
    main()
