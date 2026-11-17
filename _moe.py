"""Explainable regime-aware Mixture-of-Experts ablation (E0..E7).

Realises the proposed architecture as far as the data permits, point-in-time safe:

  E0  RF (baseline, frozen config)                    [reference]
  E1  single EBM
  E2  single EBM + rainfall-transition features        (Rainfall_PrevLap / Rainfall_Delta)
  E3  single EBM + condition-estimator probabilities   (P_wet, P_transition as features)
  E4  rainfall *intensity* estimation                  -> BLOCKED: OpenF1 weather has only
      binary rainfall, no mm/h ground truth. (E2 is the feasible proxy.)
  E5  Dry / Wet experts (hard gate)
  E6  Dry / Wet / Transition experts (hard gate)
  E7  E6 experts + learned interpretable gate (soft MoE) + condition-estimator eval

Condition labels are derived as-of Time_Start only (current Rainfall + previous-lap
Rainfall_PrevLap), so no future information leaks into the gate or experts.

Test = 2025 (temporal). Metrics in seconds (exp back from Log_Duration).
SCRATCH — does not modify main code.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, confusion_matrix, mean_absolute_error, precision_recall_fscore_support,
)

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab, _rain_features
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}
RAIN_EXTRA = ["Rainfall_PrevLap", "Rainfall_Delta"]
RAIN_ALL = ["Rainfall", *RAIN_EXTRA]
MIN_EXPERT = 100

EBM_KW = dict(interactions=0, max_bins=64, early_stopping_rounds=100,
              n_jobs=CORES, random_state=42)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def add_rain_transitions(raw):
    raw = raw.copy()
    r = _rain_features(raw)
    raw["Rainfall_PrevLap"] = r["Rainfall_PrevLap"].to_numpy()
    raw["Rainfall_Delta"] = r["Rainfall_Delta"].to_numpy()
    return raw


def add_regimes(df):
    """as-of condition labels from Rainfall (now) + Rainfall_PrevLap (previous lap)."""
    df = df.copy()
    wet = (df["Rainfall"].to_numpy() == 1)
    prev = df["Rainfall_PrevLap"].to_numpy()
    prev_wet = (prev == 1)
    prev_known = ~np.isnan(prev)
    transition = prev_known & (wet != prev_wet)
    df["regime2"] = np.where(wet, "wet", "dry")
    df["regime3"] = np.where(wet, "wet", np.where(transition, "transition", "dry"))
    return df


def green_mask(df):
    """SC/VSC + immediately-following lap -> False (evaluation target only)."""
    d = df[["Session_Key", "Driver", "Lap", "Track_Status"]].copy()
    is_sc = d["Track_Status"].isin(SC_SET).to_numpy()
    laps = (d.assign(_is_sc=is_sc)
            .groupby(["Session_Key", "Driver", "Lap"], sort=False)["_is_sc"].max()
            .reset_index().sort_values(["Session_Key", "Driver", "Lap"]))
    laps["_prev_sc"] = laps.groupby(["Session_Key", "Driver"])["_is_sc"].shift(1).fillna(False)
    merged = d[["Session_Key", "Driver", "Lap"]].merge(
        laps[["Session_Key", "Driver", "Lap", "_prev_sc"]],
        on=["Session_Key", "Driver", "Lap"], how="left")
    return ~(is_sc | merged["_prev_sc"].to_numpy().astype(bool))


def make_rf():
    return RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                                 min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                                 random_state=42)


def main():
    t0 = time.time()
    log("loading data ...")
    tr = add_rain_transitions(pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                                         pd.read_parquet(ROOT / "data_2024.parquet")],
                                        ignore_index=True))
    te = add_rain_transitions(pd.read_parquet(ROOT / "data_2025.parquet"))

    log("preparing ML frames (outlier filter ON, + rainfall-transition features) ...")
    tr_ml, tr_feats, _, _ = _prepare_lab(tr, RAIN_EXTRA)
    te_ml, te_feats, _, _ = _prepare_lab(te, RAIN_EXTRA)
    tr_ml = add_regimes(tr_ml)
    te_ml = add_regimes(te_ml)

    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    feats_base = [c for c in feats if c not in RAIN_EXTRA]
    feats_gate = [c for c in feats if c not in RAIN_ALL]
    log(f"rows train={len(tr_ml):,} test={len(te_ml):,}  "
        f"features base={len(feats_base)} ext={len(feats)} gate={len(feats_gate)}")
    for r in ["dry", "wet", "transition"]:
        log(f"  test regime3 '{r}': {(te_ml['regime3'] == r).sum():,}  "
            f"train: {(tr_ml['regime3'] == r).sum():,}")

    log("building preprocessors + matrices ...")
    pre_base = build_preprocessor(feats_base).fit(tr_ml)
    pre_ext = build_preprocessor(feats).fit(tr_ml)
    pre_gate = build_preprocessor(feats_gate).fit(tr_ml)
    Xtr_base, Xte_base = pre_base.transform(tr_ml[feats_base]), pre_base.transform(te_ml[feats_base])
    Xtr_ext, Xte_ext = pre_ext.transform(tr_ml[feats]), pre_ext.transform(te_ml[feats])
    Xtr_gate, Xte_gate = pre_gate.transform(tr_ml[feats_gate]), pre_gate.transform(te_ml[feats_gate])
    ytr = tr_ml[TARGET].to_numpy(float)
    yte = te_ml[TARGET].to_numpy(float)

    te_g = green_mask(te_ml)
    te_r2 = te_ml["regime2"].to_numpy()
    te_r3 = te_ml["regime3"].to_numpy()

    def report(name, p, extra=""):
        full = sec_mae(yte, p)
        green = sec_mae(yte[te_g], p[te_g])
        dry = sec_mae(yte[te_r3 == "dry"], p[te_r3 == "dry"]) if (te_r3 == "dry").any() else np.nan
        wet = sec_mae(yte[te_r3 == "wet"], p[te_r3 == "wet"]) if (te_r3 == "wet").any() else np.nan
        trn = sec_mae(yte[te_r3 == "transition"], p[te_r3 == "transition"]) if (te_r3 == "transition").any() else np.nan
        row = {"exp": name, "full": full, "green": green, "dry": dry, "wet": wet, "transition": trn,
               "note": extra}
        results.append(row)
        log(f"  {name:34s} full={full:.4f} green={green:.4f} dry={dry:.4f} "
            f"wet={wet:.4f} trans={trn:.4f}  {extra}")

    results = []

    # ---- E0: RF baseline ----
    log("E0 RF ...")
    t = time.time()
    rf = make_rf().fit(Xtr_base, ytr)
    report("E0 RF", rf.predict(Xte_base), f"{time.time()-t:.0f}s")

    # ---- E1: single EBM ----
    log("E1 single EBM ...")
    t = time.time()
    e1 = ExplainableBoostingRegressor(**EBM_KW).fit(Xtr_base, ytr)
    report("E1 EBM", e1.predict(Xte_base), f"{time.time()-t:.0f}s")

    # ---- E2: single EBM + rainfall-transition features ----
    log("E2 EBM + rain-transition ...")
    t = time.time()
    e2 = ExplainableBoostingRegressor(**EBM_KW).fit(Xtr_ext, ytr)
    report("E2 EBM+rain_trans", e2.predict(Xte_ext), f"{time.time()-t:.0f}s")

    # ---- condition estimator (gate) on behaviour + weather-minus-rainfall ----
    log("condition estimator (EBM classifier, gate features, labels=regime3) ...")
    t = time.time()
    gate = ExplainableBoostingClassifier(**EBM_KW).fit(Xtr_gate, tr_ml["regime3"])
    g_tr = gate.predict_proba(Xtr_gate)
    g_te = gate.predict_proba(Xte_gate)
    cls = list(gate.classes_)
    gi = {c: i for i, c in enumerate(cls)}
    y_hat3 = gate.predict(Xte_gate)
    log(f"  gate classes={cls}  fit {time.time()-t:.0f}s")

    # E3: single EBM + P(wet), P(transition) features
    log("E3 EBM + condition probs ...")
    t = time.time()
    Ptr = np.column_stack([g_tr[:, gi["wet"]], g_tr[:, gi["transition"]]])
    Pte = np.column_stack([g_te[:, gi["wet"]], g_te[:, gi["transition"]]])
    e3 = ExplainableBoostingRegressor(**EBM_KW).fit(np.hstack([Xtr_base, Ptr]), ytr)
    report("E3 EBM+cond_probs", e3.predict(np.hstack([Xte_base, Pte])), f"{time.time()-t:.0f}s")

    # ---- E5 / E6 / E7: experts ----
    def fit_experts(Xtr, ytr, regime, regimes):
        experts = {}
        for r in regimes:
            m = regime == r
            if m.sum() >= MIN_EXPERT:
                experts[r] = ExplainableBoostingRegressor(**EBM_KW).fit(Xtr[m], ytr[m])
            else:
                log(f"    (regime '{r}' has {m.sum():,} train rows -> fallback)")
                experts[r] = None
        return experts

    global_ebm = e2  # fallback for any regime too small to train

    log("E5 Dry/Wet experts ...")
    t = time.time()
    e5 = fit_experts(Xtr_ext, ytr, tr_ml["regime2"].to_numpy(), ["dry", "wet"])
    p_dry = (e5["dry"] or global_ebm).predict(Xte_ext)
    p_wet = (e5["wet"] or global_ebm).predict(Xte_ext)
    report("E5 dry/wet experts", np.where(te_r2 == "wet", p_wet, p_dry), f"{time.time()-t:.0f}s")

    log("E6 Dry/Wet/Transition experts ...")
    t = time.time()
    e6 = fit_experts(Xtr_ext, ytr, tr_ml["regime3"].to_numpy(), ["dry", "wet", "transition"])
    p6 = np.empty_like(yte)
    for r in ["dry", "wet", "transition"]:
        pr = (e6[r] or global_ebm).predict(Xte_ext)
        p6[te_r3 == r] = pr[te_r3 == r]
    report("E6 dry/wet/trans experts", p6, f"{time.time()-t:.0f}s")

    # E7: E6 experts + soft learned gate
    log("E7 soft MoE (E6 experts + learned gate) ...")
    p7 = np.zeros_like(yte)
    for r in cls:
        pr = (e6.get(r) or global_ebm).predict(Xte_ext)
        p7 += g_te[:, gi[r]] * pr
    report("E7 soft MoE (learned gate)", p7, "")

    # ---- condition estimator evaluation (§8.2) ----
    log("condition estimator evaluation ...")
    acc = accuracy_score(te_ml["regime3"], y_hat3)
    p_, r_, f1, _ = precision_recall_fscore_support(te_ml["regime3"], y_hat3,
                                                    labels=cls, average=None, zero_division=0)
    cm = confusion_matrix(te_ml["regime3"], y_hat3, labels=cls)
    cond_eval = {"accuracy": float(acc), "classes": cls,
                 "precision": {c: float(v) for c, v in zip(cls, p_)},
                 "recall": {c: float(v) for c, v in zip(cls, r_)},
                 "f1": {c: float(v) for c, v in zip(cls, f1)},
                 "confusion": cm.tolist(), "confusion_labels": cls}
    log(f"  acc={acc:.4f}")
    for c, prec, rec, f in zip(cls, p_, r_, f1):
        log(f"    {c:12s} precision={prec:.3f} recall={rec:.3f} f1={f:.3f}")
    log("  confusion (rows=true, cols=pred):")
    log("        " + "".join(f"{c:>12s}" for c in cls))
    for i, c in enumerate(cls):
        log(f"    {c:12s}" + "".join(f"{cm[i, j]:>12d}" for j in range(len(cls))))

    # ---- summary table ----
    print("\n" + "=" * 90)
    print(f"{'experiment':34s} {'full':>8s} {'green':>8s} {'dry':>8s} {'wet':>8s} {'trans':>8s}")
    print("-" * 90)
    for r in results:
        print(f"{r['exp']:34s} {r['full']:8.4f} {r['green']:8.4f} {r['dry']:8.4f} "
              f"{r['wet']:8.4f} {r['transition']:8.4f}")
    print("=" * 90)
    print(f"green-only reference (RF, prior session) = 0.2144 s")
    print(f"total={time.time()-t0:.0f}s")

    out = {"results": results, "condition_estimator": cond_eval,
           "ebm_kw": {k: v for k, v in EBM_KW.items()},
           "features": {"base": len(feats_base), "ext": len(feats), "gate": len(feats_gate)},
           "rows": {"train": int(len(tr_ml)), "test": int(len(te_ml))},
           "regimes_test": {r: int((te_ml["regime3"] == r).sum()) for r in ["dry", "wet", "transition"]}}
    with open(ROOT / "moe_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log("saved -> moe_results.json")


if __name__ == "__main__":
    main()
