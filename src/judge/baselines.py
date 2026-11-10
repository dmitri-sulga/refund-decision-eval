"""Baselines: majority class, a keyword rule, a company-prior rule, and TF-IDF + logistic regression.

All baselines see the same text the LLM judge sees (case header + visible
narrative); only the company rule uses the company name, on purpose: it shows
how much of the recorded outcome is explained by who answered, not by the case.

python -m judge.baselines --benchmark   # measure CPU cost once -> results/baseline_cost.json
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from . import config as C
from .data import load_records, load_splits
from .prompts import _case_header, visible_narrative

COST_FILE = C.RESULTS / "baseline_cost.json"
GRID = [{"C": c, "class_weight": w} for c in (0.25, 1.0, 4.0, 16.0, 64.0)
        for w in (None, "balanced")]


def classic_text(rec: dict) -> str:
    return _case_header(rec) + visible_narrative(rec["narrative"])


def make_lr(params: dict) -> Pipeline:
    return Pipeline([
        # double precision and a tight tolerance: with the defaults the fitted model
        # depended on the order of the training rows
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                                  sublinear_tf=True, dtype=np.float64)),
        ("lr", LogisticRegression(C=params["C"], class_weight=params["class_weight"],
                                  tol=1e-8, max_iter=20000)),
    ])


def choose_params(texts, y) -> tuple[dict, list]:
    """5-fold CV on TRAIN only, scored by Cohen's kappa at threshold 0.5."""
    from sklearn.metrics import cohen_kappa_score, make_scorer

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = []
    for params in GRID:
        s = cross_val_score(make_lr(params), texts, y, cv=cv, n_jobs=5,
                            scoring=make_scorer(cohen_kappa_score))
        scores.append({**params, "cv_kappa": float(np.mean(s))})
    best = max(scores, key=lambda r: r["cv_kappa"])
    return {"C": best["C"], "class_weight": best["class_weight"]}, scores


def fit_lr(train_ids, records, params: dict | None = None):
    texts = [classic_text(records[i]) for i in train_ids]
    y = np.array([records[i]["label"] for i in train_ids])
    cv_scores = None
    if params is None:
        params, cv_scores = choose_params(texts, y)
    model = make_lr(params).fit(texts, y)
    return model, params, cv_scores


def lr_predict(model, ids, records):
    p = model.predict_proba([classic_text(records[i]) for i in ids])[:, 1]
    return (p >= 0.5).astype(int), p


def company_rule(train_ids, records):
    """Predict relief when the company's TRAIN relief rate exceeds 0.5
    (companies with fewer than 20 TRAIN complaints use the overall rate)."""
    by_co: dict[str, list[int]] = {}
    for i in train_ids:
        by_co.setdefault(records[i]["company"], []).append(records[i]["label"])
    overall = np.mean([records[i]["label"] for i in train_ids])
    rate = {co: (np.mean(v) if len(v) >= 20 else overall) for co, v in by_co.items()}
    return lambda rec: int(rate.get(rec["company"], overall) > 0.5), rate


def keyword_rule(train_ids, records):
    """A rule a person could write: predict relief when the narrative matches a
    set of keyword cues (cues.py). Candidates are every single cue, the union of
    the cues the v2 prompt calls 'more relief', and the union of all cues whose
    TRAIN relief rate is above the TRAIN base rate; the candidate with the best
    Cohen's kappa on TRAIN is used. Nothing but TRAIN is read."""
    from .cues import CUES
    from .metrics import cohen_kappa

    texts = [visible_narrative(records[i]["narrative"]) for i in train_ids]
    y = np.array([records[i]["label"] for i in train_ids])
    hits = {name: np.array([proxy(t) for t in texts]) for name, _, proxy in CUES}
    proxies = {name: proxy for name, _, proxy in CUES}
    base = y.mean()
    candidates = {name: [name] for name in hits}
    candidates["any cue the v2 prompt calls 'more relief'"] = [n for n, stated, _ in CUES if stated == "more relief"]
    candidates["any cue with above-average relief on TRAIN"] = [n for n, h in hits.items()
                                                                if h.any() and y[h].mean() > base]
    scores = []
    for label, members in candidates.items():
        pred = np.any([hits[m] for m in members], axis=0).astype(int)
        scores.append({"rule": label, "cues": members, "train_kappa": float(cohen_kappa(y, pred)),
                       "train_predicted_relief": float(pred.mean())})
    best = max(scores, key=lambda r: r["train_kappa"])
    chosen = [proxies[m] for m in best["cues"]]
    return (lambda rec: int(any(p(visible_narrative(rec["narrative"])) for p in chosen))), best, scores


def learning_curve(train_ids, dev_ids, records, params: dict, sizes=(250, 500, 1000, 2000, 4000)):
    """LR kappa on DEV as a function of the number of TRAIN labels (nested,
    stratified-by-shuffle subsets)."""
    from .metrics import point

    rng = np.random.default_rng(0)
    order = list(rng.permutation(sorted(train_ids, key=int)))
    y_dev = [records[i]["label"] for i in dev_ids]
    out = []
    for n in [s for s in sizes if s < len(order)] + [len(order)]:
        model = make_lr(params).fit([classic_text(records[i]) for i in order[:n]],
                               [records[i]["label"] for i in order[:n]])
        pred, _ = lr_predict(model, dev_ids, records)
        m = point(y_dev, pred)
        out.append({"n_train": n, "kappa": m["kappa"], "accuracy": m["accuracy"]})
    return out


def benchmark() -> dict:
    """CPU cost of the classic baseline, measured once (single process)."""
    records, splits = load_records(), load_splits()
    train, dev = splits["train"], splits["dev"]
    t0 = time.perf_counter()
    model, params, _ = fit_lr(train, records)
    t_select_and_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    model = make_lr(params).fit([classic_text(records[i]) for i in train],
                           [records[i]["label"] for i in train])
    t_fit = time.perf_counter() - t0
    texts = [classic_text(records[i]) for i in dev]
    reps = []
    for _ in range(5):
        t0 = time.perf_counter()
        model.predict_proba(texts)
        reps.append(time.perf_counter() - t0)
    per_item_ms = 1000 * float(np.median(reps)) / len(texts)
    t0 = time.perf_counter()
    for t in texts[:50]:
        model.predict_proba([t])
    single_ms = 1000 * (time.perf_counter() - t0) / 50
    rule, _, _ = keyword_rule(train, records)
    t0 = time.perf_counter()
    for _ in range(20):
        for i in dev:
            rule(records[i])
    rule_ms = 1000 * (time.perf_counter() - t0) / (20 * len(dev))
    out = {"n_train": len(train), "chosen": params, "fit_seconds": round(t_fit, 2),
           "keyword_rule_ms_per_item": round(rule_ms, 4),
           "cv_select_plus_fit_seconds": round(t_select_and_fit, 1),
           "predict_ms_per_item_batched": round(per_item_ms, 4),
           "predict_ms_per_item_one_by_one": round(single_ms, 3)}
    COST_FILE.parent.mkdir(parents=True, exist_ok=True)
    COST_FILE.write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    if ap.parse_args().benchmark:
        print(json.dumps(benchmark(), indent=1))
