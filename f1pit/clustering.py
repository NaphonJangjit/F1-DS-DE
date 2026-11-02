"""จัดกลุ่มโค้ง: K-Means (ตัวเลข) + K-Modes (หมวดหมู่) เลือก k ด้วย elbow + silhouette

>>> res = f1pit.cluster_corners(corners_df)
>>> res.summary()
>>> res.plot()                       # หรือ res.plot("k_selection.png")
>>> res.data                         # corners + KMeans_Cluster / KModes_Cluster
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config

TURN_BINS = [-0.001, 15, 45, 90, 135, 180.001]
TURN_LABELS = ["Very_Low", "Gentle", "Moderate", "Sharp", "Very_Sharp"]
LENGTH_LABELS = ["Short", "Medium", "Long"]
KMEANS_FEATURES = ["Angle_Magnitude", "Log_Corner_Length"]
KMODES_FEATURES = ["Turn_Class", "Length_Class"]


# ---------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------
def prepare_corners(corners_df):
    """ทำความสะอาด + เพิ่ม Log_Corner_Length, Turn_Class, Length_Class"""
    data = corners_df.copy()
    data["Corner_Length"] = pd.to_numeric(data["Corner_Length"], errors="coerce")

    angle_abs = pd.to_numeric(data.get("Angle"), errors="coerce").abs() if "Angle" in data else np.nan
    if "Angle_Magnitude" in data.columns:
        data["Angle_Magnitude"] = pd.to_numeric(data["Angle_Magnitude"], errors="coerce").fillna(angle_abs)
    else:
        data["Angle_Magnitude"] = angle_abs

    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Angle_Magnitude", "Corner_Length"])
    data = data[data["Corner_Length"].gt(0) & data["Angle_Magnitude"].between(0, 180)].copy()
    if len(data) < 4:
        raise ValueError("Not enough valid corners for clustering.")

    data["Log_Corner_Length"] = np.log1p(data["Corner_Length"])  # ลด skew
    data["Turn_Class"] = pd.cut(data["Angle_Magnitude"], bins=TURN_BINS,
                                labels=TURN_LABELS, include_lowest=True).astype(str)
    # แบ่งความยาวเป็น 3 กลุ่มขนาดใกล้กัน
    data["Length_Class"] = pd.qcut(data["Corner_Length"].rank(method="first"),
                                   q=3, labels=LENGTH_LABELS).astype(str)
    return data.reset_index(drop=True)


# ---------------------------------------------------------------------
# k selection
# ---------------------------------------------------------------------
def _elbow(k_values, costs, method_name):
    from kneed import KneeLocator

    elbow = KneeLocator(k_values, costs, curve="convex", direction="decreasing").elbow
    if elbow is not None:
        return int(elbow), "KneeLocator"

    # fallback: จุดที่ไกลจากเส้นตรงเชื่อมปลายทั้งสองมากที่สุด
    if len(k_values) < 3:
        raise ValueError(f"{method_name}: insufficient k values to detect an elbow.")
    cost_range = costs.max() - costs.min()
    if cost_range <= 1e-12:
        raise ValueError(f"{method_name}: cost is effectively constant; no elbow.")
    x = np.linspace(0, 1, len(costs))
    y = (costs - costs.min()) / cost_range
    idx = int(np.argmax(np.abs((1 - x) - y)[1:-1])) + 1
    return int(k_values[idx]), "maximum-distance fallback"


def select_k(results, cost_col, method_name, window=2, verbose=True):
    """หา elbow แล้วเลือก k ที่ silhouette สูงสุดในช่วง elbow ± window -> (best_k, elbow_k)"""
    k_values = results["k"].to_numpy()
    elbow_k, how = _elbow(k_values, results[cost_col].to_numpy(dtype=float), method_name)

    candidates = results[results["k"].between(elbow_k - window, elbow_k + window)]
    candidates = candidates.dropna(subset=["Silhouette_Score"])
    if candidates.empty:
        raise ValueError(f"{method_name}: no valid silhouette scores near elbow k={elbow_k}.")
    best = candidates.loc[candidates["Silhouette_Score"].idxmax()]
    best_k = int(best["k"])

    if verbose:
        print(f"\n{'=' * 55}\n{method_name}: K SELECTION\n{'=' * 55}")
        print(f"Elbow k: {elbow_k} ({how})")
        print(candidates.round(4).to_string(index=False))
        print(f"SELECTED K ({method_name}): {best_k}  "
              f"(silhouette {float(best['Silhouette_Score']):.4f})")
    return best_k, elbow_k


def _silhouette(X, labels, n_rows, **kw):
    n_labels = len(np.unique(labels))
    return silhouette_score(X, labels, **kw) if 1 < n_labels < n_rows else np.nan


def _make_kmeans(k):
    return KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=20)


def _make_kmodes(k):
    from kmodes.kmodes import KModes
    return KModes(n_clusters=k, init="Huang", n_init=10, max_iter=100,
                  random_state=config.RANDOM_STATE, verbose=0)


def _scan(make_model, X, X_score, k_values, cost_attr, cost_col, n_rows, **score_kw):
    rows = []
    for k in k_values:
        model = make_model(k)
        labels = model.fit_predict(X)
        rows.append({"k": k, cost_col: getattr(model, cost_attr),
                     "Silhouette_Score": _silhouette(X_score, labels, n_rows, **score_kw)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------
@dataclass
class ClusterResult:
    data: pd.DataFrame
    kmeans_scan: pd.DataFrame
    kmodes_scan: pd.DataFrame
    best_kmeans: int
    elbow_kmeans: int
    best_kmodes: int
    elbow_kmodes: int
    models: dict = field(default_factory=dict)

    def summary(self):
        d = self.data
        print(f"K-Means k={self.best_kmeans} | K-Modes k={self.best_kmodes}")
        print("\nK-Means cluster profiles:")
        print(d.groupby("KMeans_Cluster").agg(
            Count=("Angle_Magnitude", "size"),
            Median_Angle=("Angle_Magnitude", "median"),
            Median_Length=("Corner_Length", "median"),
        ).round(2))
        print("\nK-Modes category profiles:")
        print(d.groupby("KModes_Cluster").agg(
            Count=("Turn_Class", "size"),
            Typical_Turn=("Turn_Class", lambda s: s.mode().iloc[0]),
            Typical_Length=("Length_Class", lambda s: s.mode().iloc[0]),
        ))

    def plot(self, save_path=None):
        """กราฟ elbow + silhouette (save_path=None -> plt.show())"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        panels = [
            (axes[0, 0], self.kmeans_scan, "Inertia", "K-Means — Elbow", self.elbow_kmeans, self.best_kmeans),
            (axes[0, 1], self.kmeans_scan, "Silhouette_Score", "K-Means — Silhouette", self.elbow_kmeans, self.best_kmeans),
            (axes[1, 0], self.kmodes_scan, "Cost", "K-Modes — Elbow", self.elbow_kmodes, self.best_kmodes),
            (axes[1, 1], self.kmodes_scan, "Silhouette_Score", "K-Modes — Silhouette (Hamming)", self.elbow_kmodes, self.best_kmodes),
        ]
        for ax, scan, col, title, elbow, best in panels:
            ax.plot(scan["k"], scan[col], marker="o", markersize=4)
            ax.axvline(elbow, color="red", linestyle="--", label=f"Elbow k={elbow}")
            ax.axvline(best, color="green", linestyle=":", label=f"Selected k={best}")
            ax.set(title=title, ylabel=col, xlabel="Number of clusters (k)")
            ax.legend()
            ax.grid(alpha=0.3)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120)
            plt.close(fig)
            print(f"Saved {save_path}")
        else:
            plt.show()
        return fig


def cluster_corners(corners_df, max_k=30, verbose=True, plot=False):
    """จัดกลุ่มโค้ง -> ClusterResult (ดู .data สำหรับ label)"""
    data = prepare_corners(corners_df)
    n = len(data)
    verbose and print("Valid corners:", n)

    # ---- K-Means บน feature ต่อเนื่อง ----
    scaler = StandardScaler()
    X_num = scaler.fit_transform(data[KMEANS_FEATURES].to_numpy())
    max_k_num = min(max_k, n - 1, len(np.unique(X_num, axis=0)) - 1)
    if max_k_num < 3:
        raise ValueError("Not enough distinct numerical feature combinations to evaluate an elbow.")
    kmeans_scan = _scan(_make_kmeans, X_num, X_num, range(2, max_k_num + 1),
                        "inertia_", "Inertia", n)
    best_kmeans, elbow_kmeans = select_k(kmeans_scan, "Inertia", "K-Means", verbose=verbose)

    # ---- K-Modes บน feature หมวดหมู่ (silhouette ด้วย Hamming) ----
    X_cat = data[KMODES_FEATURES].astype(str)
    X_cat_arr = X_cat.to_numpy(dtype=object)
    X_cat_codes = np.column_stack([pd.Categorical(X_cat[c]).codes for c in KMODES_FEATURES])
    max_k_cat = min(max_k, n - 1, len(X_cat.drop_duplicates()) - 1)
    if max_k_cat < 3:
        raise ValueError("Too few distinct categorical combinations for K-Modes elbow.")
    kmodes_scan = _scan(_make_kmodes, X_cat_arr, X_cat_codes, range(2, max_k_cat + 1),
                        "cost_", "Cost", n, metric="hamming")
    best_kmodes, elbow_kmodes = select_k(kmodes_scan, "Cost", "K-Modes", verbose=verbose)

    # ---- Final models ----
    km = _make_kmeans(best_kmeans)
    data["KMeans_Cluster"] = km.fit_predict(X_num)
    kmo = _make_kmodes(best_kmodes)
    data["KModes_Cluster"] = kmo.fit_predict(X_cat_arr)

    result = ClusterResult(
        data=data,
        kmeans_scan=kmeans_scan, kmodes_scan=kmodes_scan,
        best_kmeans=best_kmeans, elbow_kmeans=elbow_kmeans,
        best_kmodes=best_kmodes, elbow_kmodes=elbow_kmodes,
        models={"scaler": scaler, "kmeans": km, "kmodes": kmo},
    )
    if verbose:
        result.summary()
    if plot:
        result.plot()
    return result
