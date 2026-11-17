"""Write research_takeaway.xlsx — full experiment log (past + future)."""

import pandas as pd

COLS = ["ID", "Status", "Category", "Experiment", "Hypothesis", "Key Result", "Verdict", "Takeaway"]

rows = [
    # ---------------- past: models ----------------
    ["M1", "DONE", "Model", "RF baseline (2023+24 -> 2025)",
     "Tree ensemble is the best model", "full 0.2649 / green 0.2439", "PASS",
     "RF is the reference; every other model is worse."],
    ["M2", "DONE", "Model", "LinearRegression / Ridge",
     "Linear model suffices", "0.9185 s (3.5x worse)", "FAIL",
     "Global linear is too smooth for the heavy-tailed target."],
    ["M3", "DONE", "Model", "EBM (interpretable boosting)",
     "Interpretable non-linear beats RF", "0.5787 s (2.2x worse), 86 min", "FAIL",
     "EBM is smoother + far slower; not competitive."],
    ["M4", "DONE", "Model", "Deep MLP (PyTorch)",
     "Data has untapped non-linear structure", "1.26-2.63 s (5-10x worse)", "FAIL",
     "Deep nets overfit noise; depth is non-monotonic."],
    ["M5", "DONE", "Model", "KNN (72 features)",
     "Similar conditions -> similar duration", "1.28 s", "FAIL",
     "Curse of dimensionality kills KNN in 72 dims."],
    ["M6", "DONE", "Model", "KNN (top-5 features)",
     "Feature selection fixes KNN", "0.317 s", "NEUTRAL",
     "~5 dims carry 93% of signal; still loses to RF's regime splits."],
    ["M7", "DONE", "Model", "Kernel SVR (rbf, top-5)",
     "Kernel handles non-linearity", "0.345 s", "FAIL",
     "Smooth kernel can't represent regime discontinuities."],
    # ---------------- past: data bias ----------------
    ["D1", "DONE", "Data", "Upsampling (inverse-freq sample_weight)",
     "Rare wet/SC/VSC are under-represented", "0.2649 -> 0.2649 (no change)", "FAIL",
     "Imbalance is not the problem."],
    ["D2", "DONE", "Data", "SMOTE (synthetic wet/SC/VSC)",
     "Synthetic data teaches rare regimes", "0.833 s", "FAIL",
     "Rare regimes are unpredictable, not under-sampled."],
    ["D3", "DONE", "Data", "Outlier filter OFF (bring slow laps back)",
     "Keep SC/VSC/wet laps to learn them", "0.843 s (3x worse)", "FAIL",
     "Outlier filter is load-bearing; slow laps are noise."],
    ["D4", "DONE", "Data", "Green-only training (remove SC/VSC)",
     "SC/VSC are unpredictable, remove from training", "0.2178 vs 0.2144", "FAIL",
     "Redundant: outlier filter already removes ~93% SC."],
    ["D5", "DONE", "Data", "Concept drift + machine unlearning",
     "2025 drifts; unlearn old data to adapt", "0 retrains, static==adaptive", "FAIL",
     "No drift; unlearning (dropping data) only hurts."],
    # ---------------- past: feature engineering ----------------
    ["F1", "DONE", "Feature", "FE v2 anchors (green walk-back Prev/Best + Anchor_Age)",
     "Stale SC/VSC pace anchors cause big errors", "full 0.2649->0.2591; Jeddah L4 4.44->3.46", "PASS",
     "MAIN WIN: walking back to last green lap fixes lap-after-SC errors."],
    ["F2", "DONE", "Feature", "FE v3 (tyre-age x temp, stint#, field pace residual)",
     "Fuel/tyre/field context helps", "full 0.2591->0.2538; green 0.2365->0.2305", "PASS",
     "Small consistent gain; field pace residual nudges Silverstone."],
    ["F3", "DONE", "Feature", "FE v4 (Rain_mmh + within-lap + Is_Leader + dirty-air)",
     "Continuous rain intensity + Phase-5 features help", "full 0.2538->0.2604 (worse)", "FAIL",
     "Wet-specific features add noise to 98% dry; wet barely improves."],
    ["F4", "DONE", "Feature", "Rain_mmh (Open-Meteo mm/h)",
     "Quantitative rain beats binary flag", "corr 0.43 with Rainfall; didn't help aggregate", "FAIL",
     "More granular rain, but RF can't use it without hurting dry."],
    ["F5", "DONE", "Feature", "Residual target (ln Duration - ln Clean_Pace)",
     "Predicting residual removes track base time", "0.2542 vs 0.2538 (wash)", "FAIL",
     "No gain; log target is fine."],
    ["F6", "DONE", "Feature", "Feature selection (RF importance ranking)",
     "Find the informative features", "top-5 = pace anchors + length (93% importance)", "PASS",
     "Data is effectively ~5-dimensional; tail is regime/condition."],
    # ---------------- past: analysis ----------------
    ["A1", "DONE", "Analysis", "Rain effect quantification",
     "How much error is rain?", "rain = 4.1% of error; wet 4.6x dry", "PASS",
     "Wet is worst per-lap but rare (1.2%), so small aggregate effect."],
    ["A2", "DONE", "Analysis", "Tyre-condition metrics",
     "How does error split by tyre?", "DRY MAE 0.23/R2 .996; INTER 1.16/R2 .955", "PASS",
     "Everything bad is on INTERMEDIATE (4% of test)."],
    ["A3", "DONE", "Analysis", "Train-vs-test distribution shift",
     "Does the test set shift?", "DRY KS<=0.17; INTER KS 0.5-0.6", "PASS",
     "INTER has huge coverage+shift; DRY is stable."],
    ["A4", "DONE", "Analysis", "Residual bias/dispersion per condition",
     "Is INTER error biased?", "INTER bias +0.95s (76% under-pred), std 4x", "PASS",
     "INTER is systematically under-predicted AND high-variance."],
    ["A5", "DONE", "Analysis", "Within-race (Australia) dry vs inter",
     "Control circuit to isolate tyre effect", "dry 1.29 vs inter 0.88", "PASS",
     "Tyre is a proxy; dry-on-wet-track is the real failure."],
    # ---------------- future / current ----------------
    ["F7", "DONE", "Feature", "Rain_mmh (re-applied, integrated with v3-best)",
     "Continuous rain intensity, combined with winning features", "wet 1.168->1.100; full 0.2538->0.2545 (flat)", "PARTIAL",
     "Helps wet (biggest wet gain yet); doesn't move aggregate (wet is 1.2%)."],
    ["F8", "DONE", "Feature", "Laps_Since_Last_Rain (track wetness decay)",
     "Time since rain reflects drying track", "contributes to wet -0.068; Silverstone still worse", "PARTIAL",
     "Captures steady wet, not rapid lap-to-lap drying."],
    ["F9", "DONE", "Feature", "Pace_Residual (current pace - course reference pace)",
     "Field slowdown vs reference reveals environmental change", "Jeddah L4 3.13->2.39 (-0.73)", "PASS",
     "Strong SC/slowdown signal; also helped wet."],
    ["F10", "PLANNED", "Feature", "Race-control 5.1 (yellow/red/DRS flags)",
     "Yellow/red/DRS events add condition signal", "pending", "PLANNED",
     "Requires fetching race_control per session."],
    ["F11", "PLANNED", "Feature", "Full Prev_* telemetry green walk-back",
     "Walk ALL telemetry anchors back to green", "pending", "BLOCKED",
     "Needs Outcome_* columns -> dataset rebuild."],
    ["F12", "PLANNED", "Analysis", "Multi-seed stability (3-5 seeds)",
     "Confirm gains exceed seed noise", "pending", "PLANNED",
     "0.2602 vs 0.2591 = ~0.001 noise; validate wins."],
]

df = pd.DataFrame(rows, columns=COLS)
path = "research_takeaway.xlsx"
with pd.ExcelWriter(path, engine="openpyxl") as w:
    df.to_excel(w, index=False, sheet_name="Takeaways")
    ws = w.sheets["Takeaways"]
    widths = {"A": 5, "B": 9, "C": 10, "D": 42, "E": 40, "F": 38, "G": 9, "H": 46}
    for col, wdt in widths.items():
        ws.column_dimensions[col].width = wdt
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
print(f"wrote {path}: {len(df)} rows ({len(df[df.Status=='DONE'])} done, {len(df[df.Status=='PLANNED'])} planned)")
