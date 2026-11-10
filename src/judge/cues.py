"""Keyword-proxy check, on TRAIN only, of the cues that the v2 prompt states.

Each cue is a crude regular expression or length rule applied to the visible
narrative; the table gives the share of TRAIN complaints closed with monetary
relief with and without the cue. It documents how much (or how little) each
statement in the v2 prompt is backed by recorded outcomes. DEV and TEST are
never read here.
"""
from __future__ import annotations

import re

from . import metrics as M
from .prompts import visible_narrative

AMOUNT = re.compile(r"\{\$([\d,]+(?:\.\d+)?)\}")


def amounts(text: str) -> list[float]:
    return [float(a.replace(",", "")) for a in AMOUNT.findall(text)]


def _rx(pattern: str):
    rx = re.compile(pattern, re.I)
    return lambda t: bool(rx.search(t))


# name, statement in the v2 prompt (direction), proxy
CUES = [
    ("largest amount under 100 dollars", "more relief",
     lambda t: bool(amounts(t)) and max(amounts(t)) < 100),
    ("largest amount 1000 dollars or more", "(contrast for the row above)",
     lambda t: bool(amounts(t)) and max(amounts(t)) >= 1000),
    ("no dollar amount in the narrative", "less relief", lambda t: not amounts(t)),
    ("narrative shorter than 300 characters", "less relief", lambda t: len(t) < 300),
    ("fees or interest mentioned", "more relief",
     _rx(r"\b(late fees?|finance charges?|interest (charges?|was|were)|charged (me )?interest)\b")),
    ("card replaced or reissued", "more relief",
     _rx(r"\b(new|replacement) card\b|\bcard (was |has been |got )?(cancel|replac|reissu)")),
    ("issuer had denied the dispute", "no rule (recorded as a fact)",
     _rx(r"\b(denied|declined|rejected)\b|in favou?r of the merchant|"
         r"\bclosed (the|my) (dispute|claim|case)\b")),
    ("dispute described as pending", "no rule (recorded as a fact)",
     _rx(r"\b(still (pending|under investigation|investigating|waiting)|under investigation|"
         r"is pending|have not (heard|received a response))\b")),
    ("service quality words", "less relief",
     _rx(r"\b(poor|bad|terrible|horrible|unprofessional) (service|quality|work|workmanship)\b")),
]


def cue_table(train_ids, records) -> dict:
    texts = [visible_narrative(records[i]["narrative"]) for i in train_ids]
    y = [records[i]["label"] for i in train_ids]
    n, k = len(y), sum(y)
    rows = []
    for name, stated, proxy in CUES:
        hit = [proxy(t) for t in texts]
        n1 = sum(hit)
        k1 = sum(yy for yy, h in zip(y, hit) if h)
        n0, k0 = n - n1, k - k1
        rows.append({"cue": name, "stated": stated, "n_with": n1,
                     "relief_with": k1 / n1 if n1 else float("nan"),
                     "ci_with": list(M.wilson(k1, n1)),
                     "n_without": n0, "relief_without": k0 / n0 if n0 else float("nan"),
                     "difference": (k1 / n1 - k0 / n0) if n1 and n0 else float("nan")})
    return {"n_train": n, "relief_rate": k / n, "rows": rows}
