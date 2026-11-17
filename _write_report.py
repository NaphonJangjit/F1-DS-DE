"""Write fif_report_v2.md from fforest_results.json (forest-of-forests v2)."""

import json
from datetime import date
from pathlib import Path

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


def _strip(n):
    return n.replace("num__", "").replace("cat__", "")


def _imp_table(imp, top=None):
    lines = ["| rank | feature | importance |", "|---|---|---|"]
    for i, (n, v) in enumerate(imp[:top] if top else imp, 1):
        lines.append(f"| {i} | `{_strip(n)}` | {v:.5f} |")
    return "\n".join(lines)


def main():
    res = ROOT / "fforest_results.json"
    if not res.exists():
        (ROOT / "fif_report_v2.md").write_text(
            "# Forest-of-forests v2 report\n\nNo results (`fforest_results.json` missing).\n",
            encoding="utf-8")
        print("No fforest_results.json; wrote stub", flush=True)
        return

    data = json.loads(res.read_text())
    stage1 = data.get("stage1", {})
    stage2 = data.get("stage2", {})
    deg = data.get("degradation", {})

    L = []
    L.append("# Forest-of-forests v2 — tyre-aware telemetry → Duration")
    L.append("")
    L.append(f"Generated: {date.today().isoformat()}  |  source: `fforest_results.json`")
    L.append("")

    # ---- stage 1 summary ----
    L.append("## Stage 1 — 8 telemetry forests (per-channel feature engineering)")
    L.append("")
    L.append("Each channel predicts its current value from its previous value + tyre + weather + "
             "traffic + Lap + Track_Status + channel-specific geometry. Winning variant (level / "
             "delta / log) chosen by held-out-2024 MAE. \"Beat persistence\" = the forest predicts "
             "current better than the naive `current = previous` baseline.")
    L.append("")
    L.append("| Channel | variant | beat persist. | persist. MAE | RF MAE | RF R² | SGD MAE | Σ tyre imp. |")
    L.append("|---|---|---|---|---|---|---|---|")
    n_beat = 0
    for out, m in stage1.items():
        name = FRIENDLY.get(out, out)
        rf = m.get("rf", {}) or {}
        sgd = m.get("sgd", {}) or {}
        beat = m.get("beat_persistence", False)
        n_beat += int(beat)
        L.append(f"| {name} | {m.get('best_variant', '—')} | {'✓' if beat else '✗'} | "
                 f"{m.get('persistence_mae', '—')} | {rf.get('mae', '—')} | {rf.get('r2', '—')} | "
                 f"{sgd.get('mae', '—')} | {m.get('tyre_importance_sum', '—')} |")
    L.append("")
    L.append(f"Channels beating persistence: **{n_beat}/8**.")
    L.append("")

    # ---- stage 1 full importance ----
    L.append("## Stage 1 — full feature importance per forest")
    L.append("")
    for out, m in stage1.items():
        name = FRIENDLY.get(out, out)
        imp = m.get("importance", [])
        L.append(f"### {name} — variant `{m.get('best_variant', '—')}`")
        L.append("")
        if imp:
            L.append(_imp_table(imp))
        else:
            L.append("(no importance)")
        L.append("")

    # ---- degradation ----
    L.append("## Degradation curve (delta vs Tyre_Age_Total)")
    L.append("")
    if deg:
        curve = deg.get("delta_log", [])
        grid = deg.get("grid", [])
        if curve:
            lo, hi = min(curve), max(curve)
            L.append(f"Partial dependence of predicted **delta** (`Log_Duration − log Prev_Duration`) on "
                     f"`Tyre_Age_Total` (main forest, all else at median). Range {lo:.4f} → {hi:.4f} "
                     f"(Δ ≈ {(hi - lo):.4f} in log-ratio space).")
            L.append("")
            L.append("| Tyre_Age_Total | delta (log ratio) |")
            L.append("|---|---|")
            for g, c in zip(grid, curve):
                L.append(f"| {g:.1f} | {c:.4f} |")
            L.append("")
    else:
        L.append("(no degradation curve)")
        L.append("")

    # ---- stage 2 ----
    L.append("## Stage 2 — main forest (target = delta)")
    L.append("")
    L.append("The main forest predicts the **delta** `Log_Duration − log(Prev_Duration)` (the lap-over-lap "
             "log ratio), then reconstructs `Duration = Prev_Duration × exp(delta)`. Two telemetry feature "
             "sets are compared — raw `Prev_*` vs tyre-aware `Pred_*` — each with RF, SGD, KNN, XGBoost "
             "and CatBoost.")
    L.append("")
    L.append("MAE in **seconds** on the 2025 test season.")
    L.append("")
    L.append("| variant | RF | SGD | KNN | XGB | CatBoost |")
    L.append("|---|---|---|---|---|---|")
    for label in ["prev", "pred"]:
        v = stage2.get(label, {}) or {}
        name = "raw Prev_*" if label == "prev" else "Pred_*"
        cells = []
        for m in ["rf", "sgd", "knn", "xgb", "cat"]:
            mae = v.get(f"{m}_mae_s", "—")
            r2 = v.get(f"{m}_r2", "—")
            cells.append(f"{mae} / R²={r2}")
        L.append(f"| {name} | {' | '.join(cells)} |")
    L.append("")
    L.append("Each cell = MAE (s) / R² (on reconstructed Duration, 2025 test).")

    bd = (stage2.get("prev") or {}).get("breakdown", {})
    if bd:
        L.append("### 6-condition MAE breakdown (raw Prev_*, RF)")
        L.append("")
        L.append("| condition | n | MAE (s) |")
        L.append("|---|---|---|")
        for k in ["overall", "clean_no_sc_vsc", "dry", "wet", "dry_clean", "wet_clean"]:
            v = bd.get(k, {})
            mae = v.get("mae_s") if v.get("mae_s") is not None else "—"
            L.append(f"| {k} | {v.get('n', '—')} | {mae} |")
        L.append("")

    for label in ["prev", "pred"]:
        v = stage2.get(label, {}) or {}
        name = "raw Prev_*" if label == "prev" else "Pred_*"
        L.append(f"### Stage 2 feature importance — {name}")
        L.append("")
        L.append(_imp_table(v.get("importance", []), top=30))
        L.append("")

    # ---- interpretation ----
    L.append("## Interpretation")
    L.append("")
    L.append(f"- **Gate result:** {n_beat}/8 sub-forests beat the persistence baseline. Channels that "
             "did not beat persistence keep their raw `Prev_*` in the main forest (feeding a worse "
             "`Pred_*` would only hurt).")
    L.append("- **Prev_* vs Pred_* (delta target):** raw `Prev_*` telemetry vs tyre-aware `Pred_*`, each "
             "with RF/SGD/KNN. If `Pred_*` ≈ `Prev_*`, the sub-forests add no signal for the delta; if "
             "KNN ≈ RF, the signal is local/linear.")
    L.append("- **Degradation:** the delta curve's slope vs `Tyre_Age_Total` is the degradation rate a "
             "pit-stop optimizer would use.")

    out_path = ROOT / "fif_report_v2.md"
    out_path.write_text("\n".join(L), encoding="utf-8")
    print(f"Wrote {out_path.name}", flush=True)


if __name__ == "__main__":
    main()
