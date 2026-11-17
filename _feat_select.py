"""Correlation + feature-selection proposal (SCRATCH — does not modify main code).

Analyzes the current engineered feature set (tyre-age x length interaction + Prev_*
trend features included) on the temporal split, then proposes a reduced set via
correlation-based pruning and validates with a quick Ridge (linear).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from f1pit.train import TARGET
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _build_matrix, _features

ROOT = Path(__file__).resolve().parent

THRESH = 0.85  # pairwise |corr| above this -> redundant


def load_prep():
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")
    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = _features(tr_feats, tr_ml)
    feats = [c for c in feats if c in te_ml.columns]
    return tr_ml, te_ml, feats


def numeric_cols(df_ml, feats):
    num = [c for c in feats if c != "Track_Status" and c in df_ml.columns]
    return df_ml[num].apply(pd.to_numeric, errors="coerce")


def eval_ridge(Xtr_full, Xte_full, ytr, yte, num_feats):
    cols = [c for c in Xtr_full.columns if c in num_feats or c.startswith("Track_Status_")]
    Xtr = Xtr_full[cols].to_numpy(float)
    Xte = Xte_full[cols].to_numpy(float)
    sc = StandardScaler().fit(Xtr)
    m = Ridge(alpha=1.0).fit(sc.transform(Xtr), ytr)
    p = m.predict(sc.transform(Xte))
    a, pp = np.exp(yte), np.exp(p)
    return float(mean_absolute_error(a, pp)), float(np.sqrt(mean_squared_error(a, pp))), len(cols)


def main():
    tr_ml, te_ml, feats = load_prep()
    num = numeric_cols(tr_ml, feats)
    y = tr_ml[TARGET].to_numpy(float)
    all_num = list(num.columns)
    print(f"numeric features={len(all_num)}  (+Track_Status categorical)  train rows={len(tr_ml):,}")

    # 1) target correlation
    tc = num.corrwith(pd.Series(y, index=num.index)).abs().fillna(0.0).sort_values(ascending=False)
    print("\n=== |corr| with Log_Duration (top 40 of %d) ===" % len(all_num))
    for f, v in tc.head(40).items():
        print(f"  {f:44s} {v:.3f}")

    # 2) pairwise redundancy
    C = num.corr().abs().fillna(0.0)
    pairs = []
    cols = list(num.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            v = C.iloc[i, j]
            if v >= THRESH:
                pairs.append((cols[i], cols[j], float(v)))
    pairs.sort(key=lambda t: -t[2])
    print(f"\n=== pairwise |corr| >= {THRESH} ({len(pairs)} pairs, top 50) ===")
    for a, b, v in pairs[:50]:
        print(f"  {a:40s} <-> {b:40s} {v:.3f}")

    # 3) greedy selection: keep highest |target-corr|, drop collinear
    order = tc.sort_values(ascending=False).index.tolist()
    kept, dropped = [], {}
    for f in order:
        if f not in cols:
            continue
        hit = None
        for k in kept:
            v = abs(C.loc[f, k])
            if v >= THRESH:
                hit = (k, v)
                break
        if hit:
            dropped.setdefault(hit[0], []).append((f, hit[1]))
        else:
            kept.append(f)

    print(f"\n=== proposed selection (greedy, thresh {THRESH}) ===")
    print(f"KEPT numeric: {len(kept)} / {len(all_num)}")
    for f in kept:
        print(f"  KEEP  {f}  (|target corr|={tc[f]:.3f})")
    print(f"\nDROPPED numeric: {len(all_num) - len(kept)}  (grouped by the KEPT feature they're redundant with)")
    for k, lst in sorted(dropped.items(), key=lambda kv: -len(kv[1])):
        for d, v in lst:
            print(f"  DROP  {d:44s} -> {k}  (|corr|={v:.3f})")

    # 4) quick Ridge validation (temporal)
    oh = sorted(pd.concat([tr_ml["Track_Status"], te_ml["Track_Status"]])
                .astype("string").fillna("UNKNOWN").unique())
    Xtr = _build_matrix(tr_ml, feats, oh)
    Xte = _build_matrix(te_ml, feats, oh).reindex(columns=Xtr.columns, fill_value=0.0)
    med = Xtr.median()
    Xtr = Xtr.fillna(med)
    Xte = Xte.fillna(med)
    ytr = tr_ml[TARGET].to_numpy(float)
    yte = te_ml[TARGET].to_numpy(float)

    full = eval_ridge(Xtr, Xte, ytr, yte, all_num)
    sel = eval_ridge(Xtr, Xte, ytr, yte, kept)
    print("\n=== Ridge (linear) temporal validation: full vs selected ===")
    print(f"  FULL ({full[2]} feats)      MAE_s={full[0]:.4f}  RMSE_s={full[1]:.4f}")
    print(f"  SELECTED ({sel[2]} feats)   MAE_s={sel[0]:.4f}  RMSE_s={sel[1]:.4f}")
    print("DONE")


if __name__ == "__main__":
    main()
