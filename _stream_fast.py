"""Fast online learners with one chunk interface: predict_many(X) -> y, learn_many(X, y) (row by row inside).

Same recipes as _online.make (river), faster engines:
  SGD : river LinearRegression, SGD inverse-scaling 0.01, Huber, l2 1e-4, StandardScaler + TargetStandardScaler
        (already fast: ~20 s for a whole stream, so it stays on river)
  ARF : CapyMOA AdaptiveRandomForestRegressor (MOA, Java): 10 Hoeffding trees, sqrt features, ADWIN drift
        detection + background trees; online target standardisation as river's TargetStandardScaler
  kNN : numpy, k = 15, distance-weighted mean, sliding window of the last 2,000 learned rows, online
        StandardScaler (each stored row scaled with the statistics at the time it was learned, like river).
        Inside a 10-lap chunk the model does not change, so the whole chunk is scored at once.
"""

import copy
import os

import numpy as np

# CapyMOA's default JVM heap is -Xmx8g per process; with one JVM per worker that exhausts RAM
os.environ.setdefault("CAPYMOA_JVM_ARGS", "-Xmx1g -Xss10M")

KNN_K, KNN_WINDOW = 15, 2000
ARF_TREES = 10


class _RunningScaler:
    """running mean / population variance; batch update that also returns the stats after each row."""

    def __init__(self, d):
        self.n, self.mean, self.m2 = 0, np.zeros(d), np.zeros(d)

    def std(self, var):
        s = np.sqrt(np.maximum(var, 0.0))
        return np.where(s > 0, s, 1.0)

    def current(self):
        var = self.m2 / self.n if self.n else np.zeros_like(self.mean)
        return self.mean, self.std(var)

    def update_many(self, X):
        """-> (means, stds) after each row, shape (m, d); updates the state."""
        m = len(X)
        n_i = self.n + np.arange(1, m + 1)[:, None]
        s1 = self.n * self.mean + np.cumsum(X, axis=0)
        s2 = (self.m2 + self.n * self.mean ** 2) + np.cumsum(X ** 2, axis=0)
        means = s1 / n_i
        var = s2 / n_i - means ** 2
        self.n += m
        self.mean = means[-1].copy()
        self.m2 = var[-1] * self.n
        return means, self.std(var)


class KNNNumpy:
    def __init__(self, d, k=KNN_K, window=KNN_WINDOW):
        self.k, self.w = k, window
        self.sc = _RunningScaler(d)
        self.B = np.empty((0, d))
        self.yb = np.empty(0)

    def learn_many(self, X, y):
        means, stds = self.sc.update_many(X)
        self.B = np.vstack([self.B, (X - means) / stds])[-self.w:]
        self.yb = np.concatenate([self.yb, y])[-self.w:]

    def predict_many(self, X):
        if not len(self.yb):
            return np.zeros(len(X))
        mean, std = self.sc.current()
        Q = (X - mean) / std
        d2 = (Q ** 2).sum(1)[:, None] + (self.B ** 2).sum(1)[None, :] - 2 * Q @ self.B.T
        d = np.sqrt(np.maximum(d2, 0.0))
        k = min(self.k, len(self.yb))
        idx = np.argpartition(d, k - 1, axis=1)[:, :k]
        dk = np.take_along_axis(d, idx, 1)
        yk = self.yb[idx]
        exact = dk <= 1e-12
        w = 1.0 / np.where(exact, 1.0, dk)
        p = (w * yk).sum(1) / w.sum(1)
        hit = exact.any(1)
        if hit.any():                                   # a stored twin: use its target (as river does)
            p[hit] = np.array([yk[i][exact[i]].mean() for i in np.where(hit)[0]])
        return p

    def clone(self):
        return copy.deepcopy(self)


class ARFCapy:
    def __init__(self, d, seed=42, trees=ARF_TREES):
        from capymoa.core import RegressionInstance
        from capymoa.regressor import AdaptiveRandomForestRegressor
        from capymoa.stream import Schema
        self._inst = RegressionInstance
        self.sch = Schema.from_custom(features=[f"f{i}" for i in range(d)] + ["y"], target="y")
        self.m = AdaptiveRandomForestRegressor(schema=self.sch, ensemble_size=trees, max_features="sqrt",
                                               random_seed=seed)
        self.tn, self.tmean, self.tm2 = 0, 0.0, 0.0

    def _tstd(self):
        if self.tn < 2:
            return 1.0
        s = (self.tm2 / self.tn) ** 0.5
        return s if s > 0 else 1.0

    def learn_many(self, X, y):
        for x, t in zip(X, y):
            self.tn += 1                                 # TargetStandardScaler: update, then scale, then learn
            dlt = t - self.tmean
            self.tmean += dlt / self.tn
            self.tm2 += dlt * (t - self.tmean)
            ys = (t - self.tmean) / self._tstd()
            self.m.train(self._inst.from_array(self.sch, x, float(ys)))

    def predict_many(self, X):
        out = np.empty(len(X))
        for i, x in enumerate(X):
            p = self.m.predict(self._inst.from_array(self.sch, x, 0.0))
            out[i] = 0.0 if p is None or not np.isfinite(p) else p
        return out * self._tstd() + self.tmean


class SGDRiver:
    def __init__(self, d):
        import _online as on
        self.m = on.make("SGD")

    @staticmethod
    def _rows(X):
        return [dict(enumerate(r)) for r in X.tolist()]

    def learn_many(self, X, y):
        for r, t in zip(self._rows(X), y):
            self.m.learn_one(r, float(t))

    def predict_many(self, X):
        return np.array([self.m.predict_one(r) for r in self._rows(X)], float)

    def clone(self):
        return copy.deepcopy(self)


def make(name, d, seed=42):
    return {"SGD": SGDRiver, "kNN": KNNNumpy, "ARF": lambda d: ARFCapy(d, seed)}[name](d)


def prequential(name, Xw, yw, X, y, bounds, frozen=False, log=None):
    """warm-up on (Xw, yw), then for each chunk [a, b): predict all its rows, then learn them.
    Rows with a non-finite target are predicted but not learned.
    -> online predictions (and the frozen-after-warm-up predictions if frozen=True)."""
    d = X.shape[1]
    m = make(name, d)
    ok = np.isfinite(yw)
    if ok.any():
        m.learn_many(Xw[ok], yw[ok])
    if frozen:
        if name == "ARF":                    # JVM objects cannot be deep-copied: same seed, same warm-up
            f = make(name, d)
            if ok.any():
                f.learn_many(Xw[ok], yw[ok])
        else:
            f = m.clone()
    p_on = np.empty(len(X))
    p_st = np.empty(len(X)) if frozen else None
    for b in range(len(bounds) - 1):
        a, z = bounds[b], bounds[b + 1]
        p_on[a:z] = m.predict_many(X[a:z])
        if frozen:
            p_st[a:z] = f.predict_many(X[a:z])
        k = np.isfinite(y[a:z])
        if k.any():                          # e.g. DRS in 2026: no telemetry value, nothing to learn
            m.learn_many(X[a:z][k], y[a:z][k])
        if log and b % 100 == 0:
            log(f"  {name}: chunk {b}/{len(bounds) - 1}")
    return p_on, p_st
