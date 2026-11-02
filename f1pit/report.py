"""HTML report ของผล train (self-contained, รูปฝังเป็น base64)

>>> f1pit.write_report(out, df_ml, "report.html", open_browser=True)
"""

import base64
import html
import io
import webbrowser
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from . import config
from .train import ALGO_COLORS, CATEGORICAL_FEATURES, TARGET

CSS = """
*{box-sizing:border-box}
body{
    margin:0;padding:24px;background:#f1f4f8;color:#202938;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,
    "Noto Sans Thai",sans-serif;line-height:1.55
}
.rpt{
    max-width:1400px;margin:0 auto;padding:32px;background:#fff;
    border-radius:14px;box-shadow:0 4px 24px rgba(22,34,51,.06)
}
.rpt h1{font-size:28px;line-height:1.3;margin:0 0 8px}
.rpt h2{
    font-size:19px;margin:34px 0 14px;padding-bottom:9px;
    border-bottom:1px solid #e4e9f0
}
.rpt h3{font-size:14px;margin:0 0 12px;color:#526174}
.rpt .sub{color:#758195;font-size:13px;margin-bottom:18px}
.rpt .kpis{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
    gap:14px;margin:22px 0
}
.rpt .kpi{
    border:1px solid #e5eaf1;border-radius:11px;padding:17px;
    background:#fafbfd;min-width:0
}
.rpt .kpi .l{font-size:12px;color:#758195}
.rpt .kpi .v{font-size:25px;font-weight:750;margin:5px 0;overflow-wrap:anywhere}
.rpt .kpi .s{font-size:12px;color:#758195;overflow-wrap:anywhere}
.rpt .grid3{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));
    gap:18px;align-items:start
}
.rpt .grid2{
    display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));
    gap:18px;align-items:start
}
.rpt .panel{
    min-width:0;padding:18px;border:1px solid #e5eaf1;
    border-radius:11px;margin:12px 0;background:#fff
}
.rpt .tw{width:100%;overflow-x:auto;border:1px solid #e8edf3;border-radius:9px}
.rpt table{border-collapse:collapse;width:100%;font-size:12px;white-space:nowrap}
.rpt th{
    background:#f4f6fa;text-align:left;padding:10px 12px;font-weight:650;
    color:#465267;border-bottom:1px solid #e0e6ee
}
.rpt td{padding:9px 12px;border-bottom:1px solid #edf0f5}
.rpt tbody tr:last-child td{border-bottom:0}
.rpt td.num,.rpt th.num{text-align:right;font-variant-numeric:tabular-nums}
.rpt tbody tr:hover td{background:#f8fafc}
.rpt tr.best td{background:#eaf6ee}
.rpt .tag{
    display:inline-block;font-size:11px;font-weight:650;
    padding:3px 9px;border-radius:6px;color:#fff
}
.rpt .rank{
    display:inline-block;width:23px;height:23px;line-height:23px;
    text-align:center;border-radius:50%;background:#edf1f6;
    margin-right:6px;font-size:11px
}
.rpt .mae-cell{min-width:155px}
.rpt .bar{
    display:inline-block;width:65px;height:6px;background:#e8edf3;
    border-radius:6px;margin-right:8px;vertical-align:middle;overflow:hidden
}
.rpt .bar span{display:block;height:100%;border-radius:6px}
.rpt .note{
    background:#fff8e8;border:1px solid #f1dfb5;padding:11px 14px;
    border-radius:8px;margin:9px 0;font-size:12px;overflow-wrap:anywhere
}
.rpt img{display:block;max-width:100%;height:auto}
.rpt .chart{min-width:0;overflow:hidden}
.rpt .muted{color:#758195;font-size:12px}
@media(max-width:700px){
    body{padding:8px}
    .rpt{padding:16px;border-radius:8px}
    .rpt h1{font-size:23px}
    .rpt h2{font-size:17px}
    .rpt .grid2,.rpt .grid3{grid-template-columns:1fr}
}
"""


# ==================================================
# HTML helpers
# ==================================================
def esc(x):
    return html.escape(str(x))


def fmt(x, d=4):
    return "–" if pd.isna(x) else f"{x:.{d}f}"


def algo_tag(algo):
    return f'<span class="tag" style="background:{ALGO_COLORS.get(algo, "#52606d")}">{esc(algo)}</span>'


def html_table(headers, rows, row_classes=None, num_from=99):
    def cls(j):
        return "num" if j >= num_from else ""

    head = "".join(f'<th class="{cls(j)}">{h}</th>' for j, h in enumerate(headers))
    body = []
    for i, row in enumerate(rows):
        rc = row_classes[i] if row_classes and i < len(row_classes) and row_classes[i] else ""
        attr = f' class="{rc}"' if rc else ""
        body.append(f"<tr{attr}>" + "".join(f'<td class="{cls(j)}">{c}</td>' for j, c in enumerate(row)) + "</tr>")
    return f'<div class="tw"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def series_table(s, title_col, value_col="Rows"):
    return html_table([title_col, value_col], [[esc(k), f"{int(v):,}"] for k, v in s.items()], num_from=1)


def fig_to_img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" alt="chart">'


def clean_axes(ax, grid_axis="both"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#dce3ec")
    ax.grid(axis=grid_axis, color="#e7ecf2", linewidth=0.7)
    ax.set_axisbelow(True)


# ==================================================
# Charts (ใช้ matplotlib.figure.Figure -> ไม่เปิดหน้าต่าง/ไม่ต้องมี display)
# ==================================================
def chart_model_mae(results):
    ordered = results.sort_values("MAE").reset_index(drop=True)
    names = ordered["Algorithm"].tolist()[::-1]
    values = ordered["MAE"].to_numpy()[::-1]

    fig = Figure(figsize=(8.5, max(3.2, 0.48 * len(names) + 1.0)))
    ax = fig.subplots()
    y = np.arange(len(names))
    bars = ax.barh(y, values, color=[ALGO_COLORS.get(n, "#52606d") for n in names], height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean Absolute Error (log-duration units)", fontsize=9)
    ax.set_title("Model accuracy comparison — lower is better", fontsize=11, pad=14)
    vmax = max(float(np.max(values)), 0.001)
    ax.set_xlim(0, vmax * 1.25)
    for bar, v in zip(bars, values):
        ax.text(v + vmax * 0.02, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    clean_axes(ax, grid_axis="x")
    fig.subplots_adjust(left=0.29, right=0.94, top=0.88, bottom=0.13)
    return fig_to_img(fig)


def chart_actual_vs_pred(actual, pred, algo):
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    ok = np.isfinite(actual) & np.isfinite(pred)
    actual, pred = actual[ok], pred[ok]

    fig = Figure(figsize=(7, 5.4))
    ax = fig.subplots()
    ax.scatter(actual, pred, s=30, alpha=0.72, color=ALGO_COLORS.get(algo, "#4C78A8"),
               edgecolor="white", linewidth=0.45)
    lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
    if np.isclose(lo, hi):
        lo, hi = lo - 0.5, hi + 0.5
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#D9534F", linewidth=1.4, label="Perfect prediction")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Actual Log_Duration", ylabel="Predicted Log_Duration")
    ax.set_title(f"Actual vs Predicted — {algo}", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    clean_axes(ax)
    fig.subplots_adjust(left=0.14, right=0.97, top=0.9, bottom=0.14)
    return fig_to_img(fig)


def chart_sector_mae(sector_mae):
    names = sorted(sector_mae, key=lambda n: np.nanmean([sector_mae[n].get(s, np.nan) for s in (1, 2, 3)]))
    fig = Figure(figsize=(10, max(3.5, 0.48 * len(names) + 1.4)))
    ax = fig.subplots()
    y, h = np.arange(len(names)), 0.23
    colors = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B"}
    for sec, offset in [(1, h), (2, 0), (3, -h)]:
        ax.barh(y + offset, [sector_mae[n].get(sec, np.nan) for n in names],
                height=h * 0.9, color=colors[sec], label=f"Sector {sec}")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("MAE (log-duration units)")
    ax.set_title("Error by sector — lower is better", fontsize=11)
    ax.legend(frameon=False, ncol=3, loc="lower right", fontsize=8)
    clean_axes(ax, grid_axis="x")
    fig.subplots_adjust(left=0.27, right=0.97, top=0.92, bottom=0.12)
    return fig_to_img(fig)


# ==================================================
# Sections
# ==================================================
def section_overview(df_ml, out, best):
    cards = [
        ("Rows used", f"{len(df_ml):,}", "หลังทำความสะอาดข้อมูล"),
        ("Train / Test", f"{len(out.train_df):,} / {len(out.test_df):,}", f"Random split · seed {config.RANDOM_STATE}"),
        ("Features", f"{len(out.features)}", "Features ที่ใช้ฝึกโมเดล"),
        ("Best MAE", f"{best['MAE']:.4f} log units", str(best["Algorithm"])),
    ]
    kpis = "".join(f'<div class="kpi"><div class="l">{esc(l)}</div><div class="v">{esc(v)}</div>'
                   f'<div class="s">{esc(s)}</div></div>' for l, v, s in cards)
    feat_rows = [[esc(c), "Categorical" if c in CATEGORICAL_FEATURES else "Numeric",
                  f"{df_ml[c].notna().mean() * 100:.1f}%"] for c in out.features]
    return f"""
    <div class="kpis">{kpis}</div>
    <div class="grid2">
      <div class="panel"><h3>Rows by tyre compound</h3>{series_table(df_ml["Tyre"].value_counts(dropna=False), "Tyre")}</div>
      <div class="panel"><h3>Rows by track status</h3>{series_table(df_ml["Track_Status"].value_counts(dropna=False), "Track Status")}</div>
    </div>
    <div class="panel"><h3>Features included in the models</h3>
      <div style="max-height:380px;overflow:auto">{html_table(["Feature", "Type", "Non-missing"], feat_rows, num_from=2)}</div>
    </div>"""


def section_comparison(results, actual, preds):
    results = results.sort_values("MAE").reset_index(drop=True)
    max_mae = max(float(results["MAE"].max()), 1e-12)
    rows, classes = [], []
    for rank, row in enumerate(results.itertuples(index=False), start=1):
        color = ALGO_COLORS.get(row.Algorithm, "#52606d")
        mae_cell = (f'<div class="mae-cell"><span class="bar"><span style="width:{row.MAE / max_mae * 100:.1f}%;'
                    f'background:{color}"></span></span><strong>{row.MAE:.4f}</strong></div>')
        rows.append([f'<span class="rank">{rank}</span>' + algo_tag(row.Algorithm),
                     mae_cell, f"{row.RMSE:.4f}", f"{row.R2:.4f}"])
        classes.append("best" if rank == 1 else "")
    table = html_table(["Rank / Algorithm", "MAE (log units)", "RMSE (log units)", "R²"],
                       rows, row_classes=classes, num_from=2)
    best = str(results.iloc[0]["Algorithm"])
    return f"""
    <div class="panel"><h3>Performance ranking</h3>
      <div class="muted" style="margin-bottom:12px">เรียงตาม MAE จากน้อยไปมาก · ค่า MAE และ RMSE อยู่บนสเกล log-duration</div>
      {table}
    </div>
    <div class="grid2">
      <div class="panel chart"><h3>MAE comparison</h3>{chart_model_mae(results)}</div>
      <div class="panel chart"><h3>Prediction quality — best model</h3>
        <div class="muted" style="margin-bottom:10px">Best model: {esc(best)}</div>
        {chart_actual_vs_pred(actual, preds[best], best)}
      </div>
    </div>"""


def compute_sector_mae(test_df, preds):
    y = test_df[TARGET].to_numpy()
    return {algo: pd.DataFrame({"Sector": test_df["Sector"].to_numpy(), "AE": np.abs(y - p)})
            .groupby("Sector")["AE"].mean() for algo, p in preds.items()}


def section_sector(test_df, sector_mae):
    counts = test_df.groupby("Sector").size()
    names = list(sector_mae)
    rows = [[f"Sector {s}", f"{int(counts.get(s, 0)):,}", *[fmt(sector_mae[a].get(s, np.nan)) for a in names]]
            for s in (1, 2, 3)]
    table = html_table(["Sector", "Test rows", *[esc(n) for n in names]], rows, num_from=1)
    return f"""
    <div class="panel"><h3>MAE by sector</h3>{table}</div>
    <div class="panel chart">{chart_sector_mae(sector_mae)}</div>"""


def section_status(test_df, preds):
    y = test_df[TARGET].to_numpy()
    frame = pd.DataFrame({"Track_Status": test_df["Track_Status"].to_numpy()})
    for algo, p in preds.items():
        frame[f"{algo}_err"] = p - y
    rows = []
    for status, g in frame.groupby("Track_Status", dropna=False):
        row = [esc(status), f"{len(g):,}"]
        for algo in preds:
            e = g[f"{algo}_err"]
            row += [fmt(e.abs().mean()), f"{e.mean():+.4f}"]
        rows.append(row)
    headers = ["Track Status", "Rows", *[h for a in preds for h in (f"{a} MAE", f"{a} Bias")]]
    return f"""
    <div class="panel">
      <div class="muted" style="margin-bottom:12px">Bias = prediction − actual. ค่าบวกหมายถึงโมเดลทำนายสูงกว่าค่าจริง</div>
      {html_table(headers, rows, num_from=1)}
    </div>"""


def section_worst(best_name, test_df, pred, n=15):
    comp = test_df[["Lap", "Sector", "Tyre", "Tyre_Age", "Track_Status", TARGET]].copy()
    comp["Predicted"] = np.asarray(pred)
    comp["Error"] = comp["Predicted"] - comp[TARGET]
    worst = comp.reindex(comp["Error"].abs().sort_values(ascending=False).index).head(n)
    rows = [[str(int(r.Lap)), str(int(r.Sector)), esc(r.Tyre), str(int(r.Tyre_Age)), esc(r.Track_Status),
             f"{getattr(r, TARGET):.3f}", f"{r.Predicted:.3f}", f"{r.Error:+.3f}"]
            for r in worst.itertuples(index=False)]
    headers = ["Lap", "Sector", "Tyre", "Tyre age", "Track status",
               "Actual Log_Duration", "Predicted Log_Duration", "Error (log units)"]
    return f"""
    <div class="panel"><h3>Largest errors — {esc(best_name)}</h3>
      <div class="muted" style="margin-bottom:12px">{n} ตัวอย่างที่มี absolute error สูงที่สุด</div>
      {html_table(headers, rows, num_from=3)}
    </div>"""


# ==================================================
# Assemble
# ==================================================
def build_html(out, df_ml):
    results, preds, test_df = out.results, out.predictions, out.test_df
    best = results.iloc[0]
    notes_html = "".join(f'<div class="note">{esc(n)}</div>' for n in out.notes)
    title = "F1 Sector Log-Duration — Model Evaluation"
    body = f"""
    <div class="rpt">
      <h1>{title}</h1>
      <div class="sub">Target: {esc(TARGET)} · {len(results)} successful models · {len(out.features)} features ·
        Generated {datetime.now():%Y-%m-%d %H:%M}</div>
      {notes_html}
      <h2>1. Dataset overview</h2>{section_overview(df_ml, out, best)}
      <h2>2. Model performance</h2>{section_comparison(results, test_df[TARGET].to_numpy(), preds)}
      <h2>3. Prediction error by sector</h2>{section_sector(test_df, compute_sector_mae(test_df, preds))}
      <h2>4. Prediction error by track status</h2>{section_status(test_df, preds)}
      <h2>5. Largest prediction errors</h2>{section_worst(out.best_name, test_df, preds[out.best_name])}
      <div class="sub" style="margin-top:28px">Metrics are evaluated on the test split using Log_Duration.
        MAE and RMSE are in log-duration units; lower is better. R² measures explained variance on the log scale.</div>
    </div>"""
    return (f'<!DOCTYPE html><html lang="th"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">'
            f"<title>{title}</title><style>{CSS}</style></head><body>{body}</body></html>")


def write_report(out, df_ml, path="model_report.html", open_browser=False):
    """เขียน HTML report -> คืน path (absolute)"""
    path = Path(path).resolve()
    path.write_text(build_html(out, df_ml), encoding="utf-8")
    if open_browser:
        webbrowser.open(path.as_uri())
    return str(path)
