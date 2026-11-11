"""Read-only recomputation of the DEV figures quoted in docs/analysis_plan_models.md,
from the frozen copy's caches. No model call. Run with PYTHONPATH=<copy>/src, cwd = copy."""
import json
import numpy as np
from judge.data import load_records, load_splits
from judge.llm import Cache
from judge.judgments import parse
from judge.baselines import issuer_rate
from judge import metrics as M
from judge import label as L

D = "."   # the reviewer's absolute working directory, replaced by "." when committed; run from this directory
R, S = load_records(), load_splits()
cache = Cache()
dev = S["dev"]
y = np.array([R[i]["label"] for i in dev])
rate_fn, base, _ = issuer_rate(S["train"], R)
look = np.array([int(rate_fn(R[i]) > base) for i in dev])

def kap(Y, P):
    po = (Y == P).mean(1); py = Y.mean(1); pp = P.mean(1)
    pe = py * pp + (1 - py) * (1 - pp)
    return (po - pe) / (1 - pe)

out = {"dev_n": len(dev), "record_pays_dev": int(y.sum()), "lookup_pays_dev": int(look.sum()),
       "lookup_kappa": M.cohen_kappa(y, look)}
preds = {}
rng = np.random.default_rng(0)
idx = rng.integers(0, len(y), size=(10000, len(y)))
for v in ("v1", "v2"):
    parsed = [parse(cache.records[f"{v}|{i}"]) for i in dev]
    p = np.array([x["pred"] for x in parsed])
    preds[v] = p
    pd = M.paired_difference(y, look, p)
    d = kap(y[idx], p[idx]) - kap(y[idx], look[idx])
    db = p[idx].mean(1) - y[idx].mean(1)
    out[v] = {"invalid": int(sum(not x["valid"] for x in parsed)),
              "done_reason_not_stop": int(sum(cache.records[f"{v}|{i}"]["done_reason"] != "stop" for i in dev)),
              "pays": int(p.sum()), "kappa": M.cohen_kappa(y, p),
              "minus_lookup_metrics_py": pd["kappa"],
              "minus_lookup_independent": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
              "minus_lookup_99375": [float(np.percentile(d, 0.3125)), float(np.percentile(d, 99.6875))],
              "half_width_95": (pd["kappa"]["hi"] - pd["kappa"]["lo"]) / 2,
              "boot_sd": float(d.std()),
              "discordant_with_lookup": int((p != look).sum()),
              "payout_minus_record": {"value": float(p.mean() - y.mean()),
                                      "ci95": [float(np.percentile(db, 2.5)), float(np.percentile(db, 97.5))],
                                      "ci9875": [float(np.percentile(db, 0.625)), float(np.percentile(db, 99.375))]}}

pe = sorted(((r["prompt_eval_count"] or 0), r["key"]) for r in cache.records.values())
out["max_prompt_tokens_any_cached_call"] = pe[-1]
lab = L.load_sample()["ids"]
nar = sorted((len(R[i]["narrative"]), i) for i in list(dev) + list(lab))
out["longest_narratives_dev_labelled_chars"] = nar[-3:]
out["n_truncated_dev_labelled"] = sum(n > 6000 for n, _ in nar)
out["max_eval_count"] = max(r["eval_count"] or 0 for r in cache.records.values())
out["done_reason_values"] = sorted({str(r["done_reason"]) for r in cache.records.values()})

test = S["test"]
yt = np.array([R[i]["label"] for i in test])
pt = np.array([parse(cache.records[f"v1|{i}"])["pred"] for i in test])
out["heldout_v1_pays"], out["heldout_record_pays"] = int(pt.sum()), int(yt.sum())
out["labelled_record_pays"] = int(sum(R[i]["label"] for i in lab))
lv = {v: int(sum(parse(cache.records[f"{v}|{i}"])["pred"] for i in lab)) for v in ("v1", "v2")}
out["labelled_qwen_pays"] = lv
print(json.dumps(out, indent=1, default=float))
np.save(D + "/dev_arrays.npy", np.vstack([y, look, preds["v1"], preds["v2"]]))
