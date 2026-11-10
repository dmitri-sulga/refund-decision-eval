"""Defence for the gaming test: two layers that do not trust the narrative.

1. Input filter (programmatic, no model): finds text addressed to an automated
   reviewer or trying to set the output, removes the enclosing sentence or
   bracketed span, and flags the complaint. Flagged complaints are routed to a
   human instead of being auto-decided. The patterns were written against the
   'injection' rewrite only; 'fake_case_note' was written afterwards to get past
   them, to show what a pattern filter misses.
2. Fact model: the decision is computed from the facts the LLM extracted
   (plus the structured complaint fields), by a small logistic regression fit
   on DEV labels, instead of by the LLM's own holistic verdict. Pressure in the
   text can then only act through the extracted facts.
"""
from __future__ import annotations

import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

PATTERNS = [
    # addressed to an automated reader
    r"\b(?:note|message|instructions?|attention|memo)\s+(?:to|for)\s+(?:the\s+|any\s+|all\s+)?"
    r"(?:ai|a\.i\.|llm|language\s+model|model|assistant|bot|chatbot|reviewer|evaluator|grader|judge|"
    r"system|automated\s+\w+)\b",
    # override attempts
    r"\b(?:ignore|disregard|override|forget)\s+(?:(?:all|any|the|your|previous|prior|above|earlier|these)\s+)*"
    r"(?:instructions?|rubric|rules?|guidelines?|prompts?|polic(?:y|ies)|criteria)\b",
    # trying to set output fields or the verdict directly
    r"\b(?:monetary_relief|p_relief)\b",
    r"\b(?:return|output|respond\s+with|answer\s+with)\s+(?:true|false|json|yes|no)\b",
    # fake role markers
    r"\[\s*(?:system|assistant|admin|developer)\b",
    r"(?m)^\s*(?:system|assistant)\s*:",
    r"<\s*/?\s*(?:system|instructions?)\s*>",
]
_RX = [re.compile(p, re.I) for p in PATTERNS]


def _span(text: str, start: int, end: int) -> tuple[int, int]:
    """Enclosing [...] span if the match is inside brackets, else the sentence."""
    lb, rb = text.rfind("[", 0, start), text.find("]", end)
    if lb != -1 and rb != -1 and text.find("]", lb, start) == -1:
        return lb, rb + 1
    s = max(text.rfind(c, 0, start) for c in ".!?\n") + 1
    ends = [i for i in (text.find(c, end) for c in ".!?\n") if i != -1]
    return s, (min(ends) + 1 if ends else len(text))


def sanitize(text: str) -> tuple[str, bool]:
    """Return (text without flagged spans, flagged?). Unflagged text is returned
    unchanged, byte for byte."""
    spans = []
    for rx in _RX:
        for m in rx.finditer(text):
            spans.append(_span(text, m.start(), m.end()))
    if not spans:
        return text, False
    keep, pos = [], 0
    for s, e in sorted(spans):
        if s > pos:
            keep.append(text[pos:s])
        pos = max(pos, e)
    keep.append(text[pos:])
    return "".join(keep).rstrip(), True


def filtered_item_id(item_id: str) -> str:
    """Cache item id of the filtered version of a flagged narrative."""
    return f"{item_id}~filtered"


# ------------------------------------------------------------------ fact model
def encode(facts_list: list[dict], recs: list[dict], vocab: list[str] | None = None):
    """One-hot encode extracted facts (strings and booleans) plus the structured
    sub-product and sub-issue. Decision and probability fields are excluded."""
    rows = []
    for facts, rec in zip(facts_list, recs):
        toks = {f"sub_product={rec['sub_product']}", f"sub_issue={rec['sub_issue']}"}
        for k, v in facts.items():
            if k in ("monetary_relief", "p_relief", "reason"):
                continue
            toks.add(f"{k}={v}")
        rows.append(toks)
    if vocab is None:
        vocab = sorted(set().union(*rows))
    index = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(rows), len(vocab)))
    for r, toks in enumerate(rows):
        for t in toks:
            if t in index:
                X[r, index[t]] = 1.0
    return X, vocab


class FactModel:
    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, facts_list, recs, y):
        X, self.vocab = encode(facts_list, recs)
        self.lr = LogisticRegression(C=self.C, max_iter=2000).fit(X, y)
        return self

    def predict_proba(self, facts_list, recs):
        X, _ = encode(facts_list, recs, self.vocab)
        return self.lr.predict_proba(X)[:, 1]

    def weights(self) -> list[tuple[str, float]]:
        return sorted(zip(self.vocab, self.lr.coef_[0].tolist()), key=lambda t: -abs(t[1]))


def cross_fitted(facts_list, recs, y, n_splits: int = 10, seed: int = 0) -> np.ndarray:
    """Out-of-fold probabilities on DEV (each complaint scored by a model that
    did not see its label)."""
    y = np.asarray(y)
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        m = FactModel().fit([facts_list[i] for i in tr], [recs[i] for i in tr], y[tr])
        p[te] = m.predict_proba([facts_list[i] for i in te], [recs[i] for i in te])
    return p
