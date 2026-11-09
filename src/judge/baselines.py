"""Baselines: majority class, a company-prior rule, and TF-IDF + logistic regression.

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
C_GRID = (0.25, 1.0, 4.0, 16.0)


def classic_text(rec: dict) -> str:
    return _case_header(rec) + visible_narrative(rec["narrative"])


def make_lr(c: float) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                                  sublinear_tf=True, dtype=np.float32)),
        ("lr", LogisticRegression(C=c, max_iter=5000)),
    ])


def choose_c(texts, y) -> tuple[float, dict]:
    """5-fold CV on TRAIN only, scored by Cohen's kappa at threshold 0.5."""
    from sklearn.metrics import cohen_kappa_score, make_scorer

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    scores = {c: float(np.mean(cross_val_score(make_lr(c), texts, y, cv=cv,
                                               scoring=make_scorer(cohen_kappa_score))))
              for c in C_GRID}
    return max(scores, key=scores.get), scores


def fit_lr(train_ids, records, c: float | None = None):
    texts = [classic_text(records[i]) for i in train_ids]
    y = np.array([records[i]["label"] for i in train_ids])
    cv_scores = None
    if c is None:
        c, cv_scores = choose_c(texts, y)
    model = make_lr(c).fit(texts, y)
    return model, c, cv_scores


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


def learning_curve(train_ids, dev_ids, records, c: float, sizes=(250, 500, 1000, 2000, 4000, 8000)):
    """LR kappa on DEV as a function of the number of TRAIN labels (nested,
    stratified-by-shuffle subsets)."""
    from .metrics import point

    rng = np.random.default_rng(0)
    order = list(rng.permutation(sorted(train_ids, key=int)))
    y_dev = [records[i]["label"] for i in dev_ids]
    out = []
    for n in [s for s in sizes if s < len(order)] + [len(order)]:
        model = make_lr(c).fit([classic_text(records[i]) for i in order[:n]],
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
    model, c, _ = fit_lr(train, records, c=None)
    t_select_and_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    model = make_lr(c).fit([classic_text(records[i]) for i in train],
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
    out = {"n_train": len(train), "chosen_C": c, "fit_seconds": round(t_fit, 2),
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
