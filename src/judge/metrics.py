"""Agreement metrics, bootstrap confidence intervals and selective-automation curves.

Binary labels throughout: 1 = monetary relief, 0 = no monetary relief.
Undefined quantities (e.g. precision with no predicted positives) are NaN, never 0.
"""
from __future__ import annotations

import math

import numpy as np

from . import config as C

METRICS = ("accuracy", "kappa", "precision_1", "recall_1", "precision_0", "recall_0")


def _rates(Y: np.ndarray, P: np.ndarray) -> dict:
    """Row-wise metrics for 2-D arrays (one bootstrap replicate per row)."""
    Y = Y.astype(bool)
    P = P.astype(bool)
    n = Y.shape[1]
    tp = (Y & P).sum(1)
    tn = (~Y & ~P).sum(1)
    pos_pred = P.sum(1)
    pos_true = Y.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        po = (tp + tn) / n
        py, pp = pos_true / n, pos_pred / n
        pe = py * pp + (1 - py) * (1 - pp)
        kappa = np.where(pe < 1, (po - pe) / (1 - pe), np.nan)
        out = {
            "accuracy": po,
            "kappa": kappa,
            "precision_1": np.where(pos_pred > 0, tp / pos_pred, np.nan),
            "recall_1": np.where(pos_true > 0, tp / pos_true, np.nan),
            "precision_0": np.where(n - pos_pred > 0, tn / (n - pos_pred), np.nan),
            "recall_0": np.where(n - pos_true > 0, tn / (n - pos_true), np.nan),
        }
    return out


def point(y, p) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    r = _rates(y[None, :], p[None, :])
    return {k: float(v[0]) for k, v in r.items()}


def accuracy(y, p) -> float:
    return point(y, p)["accuracy"]


def cohen_kappa(y, p) -> float:
    return point(y, p)["kappa"]


def _indices(n: int, B: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(B, n))


def bootstrap(y, p, B: int = C.BOOTSTRAP_B, seed: int = C.BOOTSTRAP_SEED) -> dict:
    """Percentile bootstrap (resampling complaints) for every metric in METRICS."""
    y, p = np.asarray(y), np.asarray(p)
    est = point(y, p)
    idx = _indices(len(y), B, seed)
    reps = _rates(y[idx], p[idx])
    out = {}
    for k in METRICS:
        v = reps[k][~np.isnan(reps[k])]
        lo, hi = (np.percentile(v, [2.5, 97.5]) if len(v) else (np.nan, np.nan))
        out[k] = {"value": est[k], "lo": float(lo), "hi": float(hi)}
    out["n"] = int(len(y))
    yb, pb = y.astype(bool), p.astype(bool)
    out["confusion"] = {"relief_relief": int((yb & pb).sum()), "relief_none": int((yb & ~pb).sum()),
                        "none_relief": int((~yb & pb).sum()), "none_none": int((~yb & ~pb).sum())}
    out["base_rate"] = float(np.mean(y))
    out["predicted_positive_rate"] = float(np.mean(p))
    return out


def paired_difference(y, p_a, p_b, B: int = C.BOOTSTRAP_B, seed: int = C.BOOTSTRAP_SEED) -> dict:
    """Bootstrap CI of metric(b) - metric(a) on the same resampled complaints."""
    y, p_a, p_b = np.asarray(y), np.asarray(p_a), np.asarray(p_b)
    idx = _indices(len(y), B, seed)
    ra, rb = _rates(y[idx], p_a[idx]), _rates(y[idx], p_b[idx])
    ea, eb = point(y, p_a), point(y, p_b)
    out = {}
    for k in ("accuracy", "kappa"):
        d = rb[k] - ra[k]
        d = d[~np.isnan(d)]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out[k] = {"value": eb[k] - ea[k], "lo": float(lo), "hi": float(hi)}
    return out


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (good at 0/n)."""
    if n == 0:
        return (math.nan, math.nan)
    ph = k / n
    den = 1 + z * z / n
    centre = (ph + z * z / (2 * n)) / den
    half = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


# ------------------------------------------------------------------ selective automation
def confidence(pred, p_relief):
    """Confidence in the decision actually taken: p if pred==1 else 1-p.
    A probability that contradicts the boolean decision yields confidence < 0.5."""
    pred, p = np.asarray(pred), np.asarray(p_relief, dtype=float)
    return np.where(pred == 1, p, 1 - p)


def selective_curve(y, pred, conf) -> list[dict]:
    """One point per distinct confidence threshold t: automate items with conf >= t.
    NaN confidence (invalid output) is never automated."""
    y, pred, conf = np.asarray(y), np.asarray(pred), np.asarray(conf, dtype=float)
    ok = ~np.isnan(conf)
    pts = []
    for t in sorted(set(conf[ok].tolist()), reverse=True):
        m = ok & (conf >= t)
        k = int(m.sum())
        pts.append({"threshold": float(t), "n_auto": k, "coverage": k / len(y),
                    "agreement": float(np.mean(y[m] == pred[m])),
                    "auto_relief_share": float(np.mean(pred[m]))})
    return pts


def choose_threshold(y, pred, conf, target: float = C.TARGET_AGREEMENT,
                     min_auto: int = C.MIN_AUTOMATED):
    """Pre-registered rule: the lowest threshold (largest coverage) whose
    automated subset reaches `target` agreement with at least `min_auto` items.
    Returns None when no threshold qualifies."""
    best = None
    for pt in selective_curve(y, pred, conf):
        if pt["n_auto"] >= min_auto and pt["agreement"] >= target:
            best = pt["threshold"]
    return best


def at_threshold(y, pred, conf, t, B: int = C.BOOTSTRAP_B, seed: int = C.BOOTSTRAP_SEED) -> dict:
    """Coverage and agreement when automating conf >= t, with bootstrap CIs."""
    y, pred, conf = np.asarray(y), np.asarray(pred), np.asarray(conf, dtype=float)
    auto = ~np.isnan(conf) & (conf >= t)
    n, k = len(y), int(auto.sum())
    agree = (y == pred)
    idx = _indices(n, B, seed)
    A, G = auto[idx], agree[idx]
    cov = A.mean(1)
    with np.errstate(invalid="ignore"):
        acc = np.where(A.sum(1) > 0, (A & G).sum(1) / A.sum(1), np.nan)
    acc = acc[~np.isnan(acc)]
    return {
        "threshold": float(t), "n": n, "n_auto": k,
        "coverage": {"value": k / n, "lo": float(np.percentile(cov, 2.5)),
                     "hi": float(np.percentile(cov, 97.5))},
        "agreement": {"value": float(agree[auto].mean()) if k else math.nan,
                      "lo": float(np.percentile(acc, 2.5)) if len(acc) else math.nan,
                      "hi": float(np.percentile(acc, 97.5)) if len(acc) else math.nan},
        "auto_relief": int(pred[auto].sum()), "auto_no_relief": int(k - pred[auto].sum()),
        # the two kinds of error among automated decisions, kept apart
        "wrong_no_relief": int((auto & (pred == 0) & (y == 1)).sum()),
        "wrong_relief": int((auto & (pred == 1) & (y == 0)).sum()),
        "majority_agreement_on_auto": float(max(y[auto].mean(), 1 - y[auto].mean())) if k else math.nan,
    }
