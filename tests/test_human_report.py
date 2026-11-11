"""The human-label output once labels lock, tested with synthetic labels in a temporary directory.
Nothing is written under results/human_labels/.

The synthetic labels are locked now, while the real cache holds the judges' answers for the labelled sample
(requested after the real lock). The fixture therefore reads a view of the cache without any answer for a
labelled complaint (the judges' answers are synthetic, see predict below), so the production guard in
analysis.human_labels is left as it is; test_the_guard_refuses_an_answer_that_predates_the_lock checks that it
still refuses."""
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


def cache_without(ids, extra=()):
    """The response cache as a view without any answer for the given complaints, plus extra records."""
    c = Cache()
    drop = set(ids)
    c.records = {k: r for k, r in c.records.items() if r["item_id"].split("~")[0] not in drop}
    c.records.update({r["key"]: r for r in extra})
    return c


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
    S = A.Systems(load_records(), load_splits(), cache_without(ids), lr_params=METRICS["tfidf_lr"]["chosen"])
    m = copy.deepcopy(METRICS)
    m["human"] = A.human_labels(S)
    m["posthoc"]["human"] = P.human_extras(m)
    yield m, ids, S
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


def test_the_guard_refuses_an_answer_that_predates_the_lock(locked):
    _, ids, S = locked
    early = {"key": f"v1|{ids[0]}", "item_id": ids[0], "prompt_version": "v1",
             "requested_at": "2026-11-10T00:00:00+00:00"}        # before the synthetic lock, which is now
    S2 = A.Systems(S.records, S.splits, cache_without(ids, [early]), lr_params=METRICS["tfidf_lr"]["chosen"])
    with pytest.raises(RuntimeError, match="predates the lock"):
        A.human_labels(S2)


def test_the_person_never_sits_in_a_dev_or_held_out_column(locked):
    m, _, _ = locked
    head, summary, human = Rp.b_headline(m), Rp.b_summary(m), Rp.BLOCKS["human"](m)
    for text in (head, summary):
        rows = _table_rows(text, "DEV kappa")
        assert rows and not any("Person" in r for r in rows)
    assert "**Blind human labels**" in head and "| Person" not in head     # the full tables print once, below
    assert "(registered)" in head and "registered sensitivity line" in head
    assert "| Person" not in summary and "prints once, in the human-label section" in summary
    assert "same " in human and "decided cases of the 100-complaint TRAIN labelling sample" in human
    person = _table_rows(human, "Kappa against the record")
    assert person[0].startswith("| Person") and any(r.startswith("| Issuer relief rate") for r in person)


def test_human_section_has_the_required_parts(locked):
    m, ids, _ = locked
    text, detail = Rp.BLOCKS["human"](m), Rp.BLOCKS["human_detail"](m)
    against = _table_rows(text, "System against the person")
    assert any(r.startswith("| LLM judge v1") and "same question as the rubric" in r for r in against)
    assert any(r.startswith("| LLM judge v2") for r in against) and any(r.startswith("| TF-IDF + logistic") for r in against)
    assert "person would pay, system would not" in text and "not in the registered plan" in text
    assert "rules the judge v1 prompt was given" in text and "not ground truth" in text
    assert "median" in text and "quartiles" in text and "No v3 prompt" in text and "not carried out" in text
    assert "minutes in all" not in text and "include the annotator's breaks" in text
    assert "'Cannot tell' in" in text and "not recorded as a field" in text
    assert "minus TF-IDF + logistic regression (registered)" in text
    assert "minus LLM judge v1 (plain rubric) (not in the registered plan" in text
    assert "Registered check of the two readings" in text and "Accuracy [95% CI], decided cases" in text
    assert "Person against the recorded company decision (registered)" in detail
    assert "'cannot tell' as no money back" in text and "Time per label (not in the registered plan" in text
    assert "Issuer relief rate (added later) (not in the registered plan" in text
    assert "they disagree in sign" in text or "they agree in sign" in text
    assert "His share of 'cannot tell' by the time spent on a label" in text and "This annotator" in text
    assert "he went through 1 of the 24" in detail and "The plan registers these figures for each half" in detail
    assert "reads these only through the v3 rule" not in detail
    review = _table_rows(detail, "Author's reading")
    assert len(review) == 1 and "synthetic" in review[0]
    assert "Calibration half" in detail and "Evaluation half" in detail
    for t in (text, detail, Rp.b_headline(m), Rp.b_summary(m)):
        assert not any(i in t for i in ids)        # no labelled complaint is named
