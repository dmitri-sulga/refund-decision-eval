"""The human-label output once labels lock, tested with synthetic labels in a temporary directory.
Nothing is written under results/human_labels/."""
import copy
import json

import numpy as np
import pytest

from judge import analysis as A
from judge import config as C
from judge import label as L
from judge import posthoc as P
from judge import report as Rp
from judge import review as V
from judge.data import load_records, load_splits
from judge.llm import Cache

METRICS = json.loads((C.RESULTS / "metrics.json").read_text())
REAL_DIR = C.RESULTS / "human_labels"


def _names():
    return sorted(p.name for p in REAL_DIR.glob("*")) if REAL_DIR.exists() else []


@pytest.fixture(scope="module")
def locked(tmp_path_factory):
    before = _names()
    tmp = tmp_path_factory.mktemp("labels")
    mp = pytest.MonkeyPatch()
    mp.setattr(L, "LABELS_DIR", tmp)
    ids = L.load_sample()["ids"]
    rng = np.random.default_rng(3)
    for i, c in zip(ids, rng.choice(["relief", "no_relief", "cannot_tell"], size=len(ids), p=[.35, .55, .10])):
        L._append(L.labels_path("synthetic"), {
            "complaint_id": i, "label": str(c), "note": "", "annotator": "synthetic",
            "seconds": float(rng.integers(20, 200)), "ui": "test", "translation_shown": True,
            "translation_problems": [], "labelled_at": "2026-11-10T12:00:00+00:00"})
    L._lock("synthetic")
    first = V.review_rows()[0]
    other = next(g for g in ("model_wrong", "record_questionable", "undecidable") if g != first["group"])
    L._append(V.review_path("synthetic"), {"complaint_id": first["complaint_id"], "group": other,
                                           "drafted_group": first["group"], "confirmed": False, "note": "synthetic",
                                           "annotator": "synthetic", "reviewed_at": "2026-11-10T13:00:00+00:00"})
    orig = A.Systems.predict

    def predict(self, name, split):          # the judges "answer" the labelled sample
        if split == "label" and name in ("v1", "v2"):
            n = len(self.splits["label"])
            r = np.random.default_rng(1 if name == "v1" else 2)
            return {"pred": r.integers(0, 2, n), "p": np.full(n, np.nan), "invalid": 0}
        return orig(self, name, split)

    mp.setattr(A.Systems, "predict", predict)
    S = A.Systems(load_records(), load_splits(), Cache(), lr_params=METRICS["tfidf_lr"]["chosen"])
    m = copy.deepcopy(METRICS)
    m["human"] = A.human_labels(S)
    m["posthoc"]["human"] = P.human_extras(m)
    yield m, ids
    mp.undo()
    assert _names() == before                # the real label directory was not touched


def _table_rows(text, header_word):
    """Rows of the first markdown table whose header contains header_word."""
    lines = text.splitlines()
    start = next(k for k, ln in enumerate(lines) if ln.startswith("|") and header_word in ln)
    rows = []
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        rows.append(ln)
    return rows


def test_the_person_never_sits_in_a_dev_or_held_out_column(locked):
    m, _ = locked
    for text in (Rp.b_headline(m), Rp.b_summary(m)):
        rows = _table_rows(text, "DEV kappa")
        assert rows and not any("Person" in r for r in rows)
        assert "same " in text and "decided cases of the 100-complaint TRAIN labelling sample" in text
        person = _table_rows(text, "Kappa against the record")
        assert person[0].startswith("| Person") and any(r.startswith("| Issuer relief rate") for r in person)


def test_human_section_has_the_required_parts(locked):
    m, ids = locked
    text, detail = Rp.BLOCKS["human"](m), Rp.BLOCKS["human_detail"](m)
    against = _table_rows(text, "System against the person")
    assert any(r.startswith("| LLM judge v1") and "same question as the rubric" in r for r in against)
    assert any(r.startswith("| LLM judge v2") for r in against) and any(r.startswith("| TF-IDF + logistic") for r in against)
    assert "person would pay, system would not" in text and "not in the registered plan" in text
    assert "rules the judge v1 prompt was given" in text and "not ground truth" in text
    assert "median" in text and "quartiles" in text and "No v3 prompt exists" in text
    assert "minus TF-IDF + logistic regression (registered)" in text
    assert "minus LLM judge v1 (plain rubric) (not in the registered plan" in text
    assert "Registered check of the two readings" in text and "Accuracy [95% CI], decided cases" in text
    assert "Person against the recorded company decision (registered)" in detail
    assert "'cannot tell' as no money back" in text and "Time per label (not in the registered plan" in text
    for t in (Rp.b_headline(m), Rp.b_summary(m)):
        assert "Issuer relief rate (added later) (not in the registered plan" in t
    review = _table_rows(detail, "Author's reading")
    assert len(review) == 1 and "synthetic" in review[0]
    assert "Calibration half" in detail and "Evaluation half" in detail
    for t in (text, detail, Rp.b_headline(m), Rp.b_summary(m)):
        assert not any(i in t for i in ids)        # no labelled complaint is named
