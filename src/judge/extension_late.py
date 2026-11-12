"""Registered outputs of the model extension that the registered analysis code does not compute.

WRITTEN AFTER THE RESULTS WERE SEEN (eleventh post-run change, after two reviews of the first version of that
change). The definitions are the registered plan's, unchanged (docs/analysis_plan_models.md, lines 81-95, "What is
computed", which Amendment 2, section 10 applies to the hosted arm):

- item 1: accuracy against the record, with the study's bootstrap (metrics.bootstrap: 10,000 resamples, seed 0);
- item 2: on DEV, judge minus TF-IDF (metrics.paired_difference, same settings), against the study's own TF-IDF
  model with its frozen settings (the DEV decisions behind its frozen DEV kappa);
- item 3: for every cell, the paired bootstrap interval (10,000 resamples, seed 0, complaints resampled) of the
  payout share minus the record's, computed by the registered function models_analysis.paired;
- item 4: accuracy against the person on his decided labels, and on all of the sample with 'cannot tell' counted as
  no money back (the registered sensitivity line), with the same bootstrap.

Nothing here gives a label, a verdict or a corrected level: the plan attaches none to these outputs. Item 5's tokens
per second are not computed: the plan does not fix how per-call rates are aggregated, and the hosted answers carry
no server timings. Calls no model.
"""
from __future__ import annotations

import numpy as np

from . import metrics as M

LATE = "registered, computed after the results were seen"


def tfidf_dev_decisions(records, splits) -> np.ndarray:
    """The study's TF-IDF model on DEV, fitted on TRAIN with the frozen settings (as src/judge/sql.py does)."""
    from . import freeze
    from .analysis import Systems
    from .llm import Cache
    S = Systems(records, splits, Cache(), lr_params=freeze.load()["tfidf_lr"])
    return np.asarray(S.predict("tfidf_lr", "dev")["pred"], dtype=int)


def cell_figures(y, preds, split: str, tfidf=None, person=None, ids=None) -> dict:
    """Items 1-4 for one cell, from its decisions (malformed answers already scored as no money back)."""
    from .models_analysis import paired
    y, preds = np.asarray(y, dtype=int), np.asarray(preds, dtype=int)
    out = {"accuracy_vs_record": M.bootstrap(y, preds)["accuracy"],
           "payout_minus_record": paired(y, y, preds, "share"),
           "record_pays": int(y.sum()), "pays": int(preds.sum()), "n": int(len(y))}
    if split == "dev" and tfidf is not None:
        out["minus_tfidf"] = M.paired_difference(y, np.asarray(tfidf, dtype=int), preds)
    if split == "labelled" and person is not None:
        lab = {i: person[i]["label"] for i in ids if i in person}
        dec = [k for k, i in enumerate(ids) if lab.get(i) in ("relief", "no_relief")]
        yd = np.array([int(lab[ids[k]] == "relief") for k in dec])
        ya = np.array([int(lab.get(i) == "relief") for i in ids])
        out["accuracy_vs_person_decided"] = {**M.bootstrap(yd, preds[dec])["accuracy"], "n": len(dec)}
        out["accuracy_vs_person_cannot_tell_as_no"] = {**M.bootstrap(ya, preds)["accuracy"], "n": len(ids)}
    return out


def compute(records, local: dict, hosted: dict) -> dict:
    """Items 1-4 for every complete cell of both arms, reference rows included."""
    from . import hosted as H
    from . import models as X
    from . import config as C
    from .data import load_splits
    from .label import labels_path, load_labels
    from .models import CELLS, _jsonl, item_sets
    from .models_analysis import cell_answers

    splits = load_splits()
    sets, _ = item_sets()
    tfidf = tfidf_dev_decisions(records, splits)
    person = load_labels(labels_path("owner"))
    study = {r["key"]: r for r in _jsonl(C.CACHE_FILE)}
    caches = {"local": {r["key"]: r for r in _jsonl(X.CACHE_FILE)},
              "hosted": {r["key"]: r for r in _jsonl(H.CACHE_FILE)}}
    out = {"what": LATE, "tfidf_dev_kappa": M.bootstrap([records[i]["label"] for i in sets["dev"]], tfidf)["kappa"],
           "arms": {}}
    for arm, an in (("local", local), ("hosted", hosted)):
        out["arms"][arm] = {}
        for model in an["order"]:
            cells = {}
            for version, split in CELLS:
                ids = sets[split]
                ans = cell_answers(model, version, ids, caches[arm], study)
                if len(ans) < len(ids):
                    continue                      # an incomplete cell gets no figure (none here)
                y = [records[i]["label"] for i in ids]
                preds = [ans[i][0] for i in ids]
                cells[f"{version}|{split}"] = cell_figures(y, preds, split, tfidf, person, ids)
            out["arms"][arm][model] = cells
    return out
