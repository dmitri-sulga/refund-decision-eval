"""Figures and labels of the local-models extension (docs/analysis_plan_models.md, Amendment 1).

Written before any of the models was called, from the rules in the amendment; reads the extension's cache, the
study's cache (for the reference model qwen3.8:27b) and the committed data. Calls no model.

python -m judge.models_analysis            # writes results/models/analysis.json and prints the section
"""
from __future__ import annotations

import json
import statistics

import numpy as np

from . import config as C
from . import metrics as M
from .judgments import parse
from . import models as X
from .models import CELLS, MAX_PROMPT_TOKENS, MODELS, REFERENCE, _jsonl, item_sets, key

LEVEL = 0.95
N_A = 2 * len(MODELS)          # comparisons with the issuer lookup (models x prompts)
N_B = len(MODELS)              # payout comparisons (models)
FORMAT_LIMIT = {"dev": 20, "labelled": 10}
A_WORDS = ("above the issuer lookup", "below the issuer lookup", "not distinguishable from the issuer lookup")
B_WORDS = ("pays more often than the companies did", "pays less often than the companies did",
           "not distinguishable from the companies' rate")
READINGS = ("as no money back", "as money back", "left out")


# ------------------------------------------------------------------ answers
def malformed(rec: dict, parsed: dict) -> bool:
    return (not parsed["valid"] or rec.get("done_reason") != "stop"
            or (rec.get("prompt_eval_count") or 0) > MAX_PROMPT_TOKENS)


def cell_answers(model: str, version: str, ids: list, cache: dict, study_cache: dict) -> dict:
    """{id: (pred, malformed, record)} for the answered complaints of one cell."""
    out = {}
    for i in ids:
        rec = study_cache.get(f"{version}|{i}") if model == REFERENCE[0] else cache.get(key(model, version, i))
        if rec is not None:
            p = parse(rec)
            out[i] = (p["pred"], malformed(rec, p), rec)
    return out


# ------------------------------------------------------------------ intervals
def _alpha(level: float) -> tuple[float, float]:
    return 100 * (1 - level) / 2, 100 * (1 + level) / 2


def paired(y, a, b, stat: str, levels=(LEVEL,)) -> dict:
    """stat(b) - stat(a) on the same resampled complaints (10,000 resamples, seed 0, as the study).
    stat 'kappa': kappa against y; stat 'share': payout share of b minus that of a (a is then the record)."""
    y, a, b = (np.asarray(x, dtype=int) for x in (y, a, b))
    idx = M._indices(len(y), C.BOOTSTRAP_B, C.BOOTSTRAP_SEED)
    if stat == "kappa":
        d = M._rates(y[idx], b[idx])["kappa"] - M._rates(y[idx], a[idx])["kappa"]
        est = M.cohen_kappa(y, b) - M.cohen_kappa(y, a)
    else:
        d = b[idx].mean(1) - a[idx].mean(1)
        est = float(b.mean() - a.mean())
    d = d[~np.isnan(d)]
    out = {"value": float(est)}
    for lv in levels:
        lo, hi = np.percentile(d, _alpha(lv))
        out[f"{lv:.6g}"] = [float(lo), float(hi)]
    return out


def label(interval, words) -> str:
    lo, hi = interval
    return words[0] if lo > 0 else words[1] if hi < 0 else words[2]


# ------------------------------------------------------------------ one model's DEV cells
def dev_cell(y, lookup, preds, bad, words, stat, n_comp) -> dict:
    """Labels under the three readings of malformed answers; the label printed is the first reading's."""
    corrected = 1 - (1 - LEVEL) / n_comp
    out = {"readings": {}}
    for reading in READINGS:
        keep = np.ones(len(y), bool) if reading != "left out" else ~bad
        p = preds.copy()
        if reading == "as money back":
            p[bad] = 1
        a = lookup[keep] if stat == "kappa" else y[keep]
        iv = paired(y[keep], a, p[keep], stat, (LEVEL, corrected))
        out["readings"][reading] = {**iv, "label": label(iv[f"{LEVEL:.6g}"], words),
                                    "corrected_label": label(iv[f"{corrected:.6g}"], words)}
    first = out["readings"][READINGS[0]]
    out["label"] = first["label"]
    out["corrected_level"] = corrected
    out["also_at_corrected_level"] = first["corrected_label"] == first["label"] if first["label"] != words[2] else None
    out["format_sensitive"] = bad.any() and len({r["label"] for r in out["readings"].values()}) > 1
    return out


def analyse(cache_path=None, study_cache_path=None, records=None) -> dict:
    from .baselines import issuer_rate
    from .data import load_records, load_splits
    from .label import labels_path, load_labels

    records = records if records is not None else load_records()
    splits = load_splits()
    sets, _ = item_sets()
    cache = {r["key"]: r for r in _jsonl(cache_path or X.CACHE_FILE)}
    study_cache = {r["key"]: r for r in _jsonl(study_cache_path or C.CACHE_FILE)}
    events = _jsonl(X.RUN_LOG)
    closed = {e["model"]: e["event"] for e in events if e.get("event") in ("model_dropped", "closed_incomplete",
                                                                          "model_finished")}
    f, base, _ = issuer_rate(splits["train"], records)
    dev = sets["dev"]
    y = np.array([records[i]["label"] for i in dev])
    lookup = np.array([int(f(records[i]) > base) for i in dev])
    person = load_labels(labels_path("owner"))
    lab = sets["labelled"]
    out = {"order": [REFERENCE[0], *(t for t, _ in MODELS)], "record_pays_dev": int(y.sum()), "n_dev": len(dev),
           "lookup_pays_dev": int(lookup.sum()), "models": {}}
    ref_preds = {}
    for model in out["order"]:
        started = model == REFERENCE[0] or any(k.startswith(model + "|") for k in cache)
        m = {"status": closed.get(model) or ("reference" if model == REFERENCE[0] else
                                             "not closed" if started else "not run"), "cells": {}}
        for version, split in CELLS:
            ids = dev if split == "dev" else lab
            ans = cell_answers(model, version, ids, cache, study_cache)
            c = {"n": len(ids), "answered": len(ans), "malformed": sum(v[1] for v in ans.values())}
            walls = [v[2]["wall_s"] for v in ans.values() if v[2].get("wall_s") is not None]
            if walls:
                c["seconds_per_call"] = {"median": statistics.median(walls), "mean": statistics.fmean(walls)}
            if len(ans) < len(ids):
                c["status"] = f"incomplete ({len(ans)} of {len(ids)} answered)"
                c["pays_answered"] = sum(v[0] for v in ans.values())
                m["cells"][f"{version}|{split}"] = c
                continue
            preds = np.array([ans[i][0] for i in ids])
            bad = np.array([ans[i][1] for i in ids])
            c["pays"] = int(preds.sum())
            if model == REFERENCE[0]:
                ref_preds[(version, split)] = preds
            elif (version, split) in ref_preds:
                c["kappa_with_reference"] = M.cohen_kappa(ref_preds[(version, split)], preds)
            if c["malformed"] > FORMAT_LIMIT[split]:
                c["status"] = "not interpretable (format)"
            else:
                c["status"] = "complete"
            yy = y if split == "dev" else np.array([records[i]["label"] for i in ids])
            c["kappa_vs_record"] = M.bootstrap(yy, preds)["kappa"]
            if split == "dev" and c["status"] == "complete":
                c["A"] = dev_cell(y, lookup, preds, bad, A_WORDS, "kappa", N_A)
                if version == "v1":
                    c["B"] = dev_cell(y, lookup, preds, bad, B_WORDS, "share", N_B)
            if split == "labelled":
                c["vs_person"] = person_figures(ids, preds, person)
            m["cells"][f"{version}|{split}"] = c
        out["models"][model] = m
    out["summary"] = summary(out)
    return out


def person_figures(ids, preds, person) -> dict:
    """Descriptive only: the person's 56 decided labels, and 'cannot tell' counted as no money back over all 100."""
    lab = {i: person[i]["label"] for i in ids if i in person}
    dec = [k for k, i in enumerate(ids) if lab.get(i) in ("relief", "no_relief")]
    yd = np.array([int(lab[ids[k]] == "relief") for k in dec])
    pd = preds[dec]
    ya = np.array([int(lab.get(i) == "relief") for i in ids])
    decided = M.bootstrap(yd, pd)["kappa"]
    as_no = M.bootstrap(ya, preds)["kappa"]
    side = lambda iv: "above" if iv["lo"] > 0 else "below" if iv["hi"] < 0 else "includes"
    sign = lambda iv: np.sign(iv["value"])
    agree = side(decided) == side(as_no) and sign(decided) == sign(as_no)
    return {"decided_n": len(dec), "kappa_decided": decided, "kappa_cannot_tell_as_no": as_no,
            "readings_agree": bool(agree)}


# ------------------------------------------------------------------ the one summary sentence (section 3)
def summary(out: dict) -> str:
    counted, excluded = [], []
    for model in out["order"][1:]:
        m = out["models"][model]
        cells = m["cells"]
        why = None
        if m["status"] in ("not run", "model_dropped"):
            why = "not run" if m["status"] == "not run" else "dropped after refusing its first request"
        elif any(c.get("status") != "complete" for c in cells.values()):
            why = "; ".join(f"{k}: {c['status']}" for k, c in cells.items() if c.get("status") != "complete")
        elif any(c.get(x, {}).get("format_sensitive") for c in cells.values() for x in ("A", "B")):
            why = "format-sensitive"
        (excluded.append(f"{model} ({why})") if why else counted.append(model))
    m_ = len(counted)
    k = sum(out["models"][t]["cells"]["v1|dev"]["B"]["label"] == B_WORDS[0] for t in counted)
    a_labels = [out["models"][t]["cells"][f"{v}|dev"]["A"]["label"] for t in counted for v in ("v1", "v2")]
    j, a = a_labels.count(A_WORDS[1]), a_labels.count(A_WORDS[0])
    s = (f"On DEV, judge v1 paid more often than the companies did for {k} of {m_} models; across the {m_} models "
         f"and both prompts, {j} of {2 * m_} comparisons with the issuer lookup were below it, {a} were above it and "
         f"the rest were not distinguishable ({', '.join(counted) or 'none'}).")
    if a:
        s += (" This is one run of one build; it was not compared with qwen3.8:27b; the model may have seen these "
              "complaints and their outcomes in training (see section 10).")
    s += (" The design compares each model with the issuer lookup and with the record, not models with one another; "
          "it does not show that the results are independent of the model.")
    if excluded:
        s += " Not counted: " + "; ".join(excluded) + "."
    return s


def main() -> None:
    out = analyse()
    (X.DIR / "analysis.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(out["summary"])


if __name__ == "__main__":
    main()
