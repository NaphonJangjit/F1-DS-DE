"""Plain deep neural network on the sector table — how complex is the data?

No experts, no regimes, no transition labels. Same feature set as the frozen RF
baseline (2023+2024 -> 2025, target Log_Duration). We sweep MLP depth/width to
probe how much non-tree structure exists:

  - if deep nets barely beat RF: the residual is noise/variance, not complexity
  - if a big train/val gap: the data is noisy -> models overfit the noise

Metrics in seconds (exp back). SCRATCH — does not modify main code.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from f1pit.train import TARGET, build_preprocessor
from f1pit.wind_rain_lab import _prepare_lab
from f1pit.online_lab import _features

ROOT = Path(__file__).resolve().parent
N_EST = 150
MAX_FEATURES = 0.3
MIN_SAMPLES_LEAF = 2
CORES = max(2, (os.cpu_count() or 8) // 2)
SC_SET = {"SC", "VSC", "SC_ENDING", "VSC_ENDING"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4096
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 300
PATIENCE = 25
VAL_FRAC = 0.1

CONFIGS = {
    "shallow [128]": [128],
    "medium [256,128]": [256, 128],
    "deep [512,256,128]": [512, 256, 128],
    "deeper [512,256,128,64]": [512, 256, 128, 64],
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sec_mae(y_log, p_log):
    return float(mean_absolute_error(np.exp(np.asarray(y_log, float)),
                                     np.exp(np.asarray(p_log, float))))


def green_mask(df):
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


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, dropout=0.0, batchnorm=True):
        super().__init__()
        dims = [in_dim, *hidden, 1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if batchnorm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(nn.ReLU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(Xtr, ytr, hidden, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(Xtr)
    idx = np.random.permutation(n)
    n_val = int(n * VAL_FRAC)
    tr_i, va_i = idx[n_val:], idx[:n_val]

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)

    model = MLP(Xtr.shape[1], hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    tr_loader = DataLoader(TensorDataset(Xtr_t[tr_i], ytr_t[tr_i]),
                           batch_size=BATCH, shuffle=True, drop_last=True)
    va_loader = DataLoader(TensorDataset(Xtr_t[va_i], ytr_t[va_i]),
                           batch_size=BATCH, shuffle=False)

    best_va = float("inf")
    best_state = None
    bad = 0
    hist = {"val_mae": []}
    for ep in range(MAX_EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            loss = loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            opt.step()
        # validation (seconds MAE)
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for xb, yb in va_loader:
                vp.append(model(xb.to(DEVICE)).cpu())
                vt.append(yb)
        vp = torch.cat(vp).numpy()
        vt = torch.cat(vt).numpy()
        va_mae = sec_mae(vt, vp)
        hist["val_mae"].append(va_mae)
        if va_mae < best_va - 1e-5:
            best_va = va_mae
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return model, best_va, ep + 1, hist


def predict_mlp(model, X, bs=BATCH):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(Xt), bs):
            out.append(model(Xt[i:i + bs].to(DEVICE)).cpu())
    return torch.cat(out).numpy()


def main():
    t0 = time.time()
    log(f"device={DEVICE}")
    log("loading + preparing (same features as RF baseline) ...")
    tr = pd.concat([pd.read_parquet(ROOT / "data_2023.parquet"),
                    pd.read_parquet(ROOT / "data_2024.parquet")], ignore_index=True)
    te = pd.read_parquet(ROOT / "data_2025.parquet")

    tr_ml, tr_feats, _, _ = _prepare_lab(tr, [])
    te_ml, te_feats, _, _ = _prepare_lab(te, [])
    feats = [c for c in _features(tr_feats, tr_ml) if c in te_ml.columns]
    log(f"features={len(feats)}  train={len(tr_ml):,}  test={len(te_ml):,}")

    pre = build_preprocessor(feats).fit(tr_ml[feats])
    Xtr = pre.transform(tr_ml[feats])
    Xte = pre.transform(te_ml[feats])
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    Xte_s = sc.transform(Xte)
    ytr = tr_ml[TARGET].to_numpy(float)
    yte = te_ml[TARGET].to_numpy(float)
    te_g = green_mask(te_ml)

    results = []

    # RF reference (same data, frozen config)
    log("RF reference ...")
    t = time.time()
    rf = RandomForestRegressor(n_estimators=N_EST, max_features=MAX_FEATURES,
                               min_samples_leaf=MIN_SAMPLES_LEAF, n_jobs=CORES,
                               random_state=42).fit(Xtr, ytr)
    p_rf = rf.predict(Xte)
    results.append({"model": "RF baseline", "full": sec_mae(yte, p_rf),
                    "green": sec_mae(yte[te_g], p_rf[te_g]), "best_val_mae": None})
    log(f"  RF full={sec_mae(yte, p_rf):.4f} green={sec_mae(yte[te_g], p_rf[te_g]):.4f} "
        f"({time.time()-t:.0f}s)")

    for name, hidden in CONFIGS.items():
        log(f"training {name} ...")
        t = time.time()
        model, best_va, epochs, hist = train_mlp(Xtr_s, ytr, hidden)
        p = predict_mlp(model, Xte_s)
        full = sec_mae(yte, p)
        green = sec_mae(yte[te_g], p[te_g])
        n_params = sum(v.numel() for v in model.parameters())
        results.append({"model": name, "full": full, "green": green,
                        "best_val_mae": best_va, "epochs": epochs, "params": n_params})
        log(f"  {name:24s} full={full:.4f} green={green:.4f} "
            f"best_val={best_va:.4f} epochs={epochs} params={n_params:,} ({time.time()-t:.0f}s)")

    print("\n" + "=" * 72)
    print(f"{'model':26s} {'full':>8s} {'green':>8s} {'val_mae':>8s}")
    print("-" * 72)
    for r in results:
        v = f"{r['best_val_mae']:.4f}" if r["best_val_mae"] is not None else "  --"
        print(f"{r['model']:26s} {r['full']:8.4f} {r['green']:8.4f} {v:>8s}")
    print("=" * 72)
    print(f"green-only RF reference (prior session) = 0.2144 s")
    print(f"total={time.time()-t0:.0f}s")

    with open(ROOT / "dnn_results.json", "w", encoding="utf-8") as f:
        json.dump({"results": results, "device": DEVICE, "features": len(feats),
                   "lr": LR, "weight_decay": WEIGHT_DECAY, "batch": BATCH}, f, indent=2)
    log("saved -> dnn_results.json")


if __name__ == "__main__":
    main()
