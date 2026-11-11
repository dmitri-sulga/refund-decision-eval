"""Power of the plan's rule A (paired 95% percentile bootstrap of kappa(judge) - kappa(lookup), n=200,
'differs' iff lo > 0) and rule B (payout share minus record share, 'holds' iff lo > 0).
Population = the 200 DEV (record, issuer lookup) pairs, resampled to make each simulated study;
synthetic judges of known sensitivity/specificity against the record. No model call."""
import json, sys
import numpy as np

D = "."   # the reviewer's absolute working directory, replaced by "." when committed; run from this directory
y0, L0, v1, v2 = np.load(D + "/dev_arrays.npy")
N, PI = 200, y0.mean()
rng = np.random.default_rng(12345)
B, SIMS = 2000, 500

def kap(Y, P):
    po = (Y == P).mean(-1); py = Y.mean(-1); pp = P.mean(-1)
    pe = py * pp + (1 - py) * (1 - pp)
    return (po - pe) / (1 - pe)

def sens_spec(k, q, pi=PI):
    pe = pi * q + (1 - pi) * (1 - q)
    po = pe + k * (1 - pe)
    tp = (po - 1 + pi + q) / 2
    se, sp = tp / pi, 1 - (q - tp) / (1 - pi)
    return (se, sp) if 0 <= se <= 1 and 0 <= sp <= 1 else None

def judge(y, L, se, sp, r, rng):
    """r = share of cases where the judge copies the issuer lookup (dependence on the issuer)."""
    own = np.where(y == 1, rng.random(y.shape) < se, rng.random(y.shape) < 1 - sp).astype(int)
    copy = rng.random(y.shape) < r
    return np.where(copy, L, own)

def true_values(se, sp, r):
    big = rng.integers(0, N, size=2_000_000)
    y, L = y0[big], L0[big]
    J = judge(y, L, se, sp, r, rng)
    return float(kap(y[None], J[None])[0] - kap(y[None], L[None])[0]), float(kap(y[None], J[None])[0]), float(J.mean())

def study(se, sp, r):
    s = rng.integers(0, N, size=N)
    y, L = y0[s], L0[s]
    J = judge(y, L, se, sp, r, rng)
    idx = rng.integers(0, N, size=(B, N))
    d = kap(y[idx], J[idx]) - kap(y[idx], L[idx])
    d = d[~np.isnan(d)]
    lo, hi = np.percentile(d, [2.5, 97.5])
    lob, hib = np.percentile(d, [0.3125, 99.6875])
    pb = J[idx].mean(1) - y[idx].mean(1)
    plo, phi = np.percentile(pb, [2.5, 97.5])
    return lo, hi, lob, hib, plo, phi

res = {"A": [], "B": []}
for r in (0.0, 0.3):
    for q in (0.30, 0.46, 0.685):
        for k in (0.318, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
            ss = sens_spec(k / (1 - r) if False else k, q)
            if ss is None:
                continue
            # with copying (r>0) the nominal k is the own-judgement kappa; the true kappa is computed
            tdiff, tk, tq = true_values(*ss, r)
            R = np.array([study(*ss, r) for _ in range(SIMS)])
            res["A"].append({"copy_lookup_share": r, "pays_share_nominal": q, "own_kappa": k,
                             "true_kappa": round(tk, 3), "true_advantage": round(tdiff, 3),
                             "P_differs_95": round(float((R[:, 0] > 0).mean()), 3),
                             "P_differs_99375": round(float((R[:, 2] > 0).mean()), 3),
                             "P_below_95": round(float((R[:, 1] < 0).mean()), 3),
                             "median_width": round(float(np.median(R[:, 1] - R[:, 0])), 3)})
            print(json.dumps(res["A"][-1]), flush=True)

# B: judges of modest quality (kappa 0.12, like v1 on DEV) paying a share q
for q in (0.30, 0.33, 0.35, 0.38, 0.40, 0.43, 0.45, 0.50):
    ss = sens_spec(0.12, q)
    if ss is None:
        continue
    R = np.array([study(*ss, 0.0) for _ in range(SIMS)])
    res["B"].append({"pays_share": q, "pays_of_200": round(q * 200), "P_holds_95": round(float((R[:, 4] > 0).mean()), 3),
                     "median_width": round(float(np.median(R[:, 5] - R[:, 4])), 3)})
    print(json.dumps(res["B"][-1]), flush=True)
json.dump(res, open(D + "/power.json", "w"), indent=1)
