"""The registered outputs of the model extension that the registered analysis code leaves out (extension_late.py).

That module was written after the results were seen (eleventh post-run change, after two reviews); its definitions
are the plan's (docs/analysis_plan_models.md, lines 81-95). These tests check that it applies the study's own
functions, that it reproduces registered figures wherever the registered code or the study already computed the
same thing, and that the committed report carries what it computes."""
import json

import numpy as np

from judge import config as C
from judge import extension_late as L
from judge import metrics as M
from judge import models_analysis as A

METRICS = json.loads((C.RESULTS / "metrics.json").read_text())
LOCAL = json.loads((C.RESULTS / "models" / "analysis.json").read_text())
HOSTED = json.loads((C.RESULTS / "models" / "hosted" / "analysis.json").read_text())
LATE = METRICS["extension"]["late"]


def test_cell_figures_apply_the_studys_bootstrap_and_the_registered_paired_function():
    rng = np.random.default_rng(1)
    y, p, t = (rng.integers(0, 2, 200) for _ in range(3))
    f = L.cell_figures(y, p, "dev", tfidf=t)
    assert f["accuracy_vs_record"] == M.bootstrap(y, p)["accuracy"]
    assert f["payout_minus_record"] == A.paired(y, y, p, "share")
    assert f["minus_tfidf"] == M.paired_difference(y, t, p)
    assert np.isclose(f["payout_minus_record"]["value"], p.mean() - y.mean())
    assert (f["pays"], f["record_pays"], f["n"]) == (int(p.sum()), int(y.sum()), 200)


def test_person_accuracy_uses_the_decided_labels_and_the_sensitivity_reading():
    ids = [str(i) for i in range(6)]
    person = {"0": {"label": "relief"}, "1": {"label": "no_relief"}, "2": {"label": "cannot_tell"},
              "3": {"label": "relief"}, "4": {"label": "no_relief"}}            # "5" unlabelled
    preds = np.array([1, 1, 1, 0, 0, 1])
    f = L.cell_figures([0] * 6, preds, "labelled", person=person, ids=ids)
    assert f["accuracy_vs_person_decided"]["n"] == 4 and np.isclose(f["accuracy_vs_person_decided"]["value"], 2 / 4)
    assert f["accuracy_vs_person_cannot_tell_as_no"]["n"] == 6
    assert np.isclose(f["accuracy_vs_person_cannot_tell_as_no"]["value"], np.mean(preds == [1, 0, 0, 1, 0, 0]))
    assert "minus_tfidf" not in f


def _arms():
    return (("local", LOCAL), ("hosted", HOSTED))


def test_every_complete_cell_of_both_arms_has_the_late_figures():
    for arm, an in _arms():
        for t in an["order"]:
            assert set(LATE["arms"][arm][t]) == set(an["models"][t]["cells"]), (arm, t)
            for k, f in LATE["arms"][arm][t].items():
                assert f["pays"] == an["models"][t]["cells"][k]["pays"], (arm, t, k)
                assert ("minus_tfidf" in f) == k.endswith("dev")
                assert ("accuracy_vs_person_decided" in f) == k.endswith("labelled")


def test_the_payout_interval_equals_the_registered_one_where_the_registered_code_computes_it():
    for arm, an in _arms():
        for t in an["order"]:
            c = an["models"][t]["cells"]["v1|dev"]
            reg = c["B"]["readings"]["as no money back"] if "B" in c else c["B_descriptive"]
            late = LATE["arms"][arm][t]["v1|dev"]["payout_minus_record"]
            assert np.isclose(late["value"], reg["value"]) and np.allclose(late["0.95"], reg["0.95"], atol=1e-6)


def test_judge_minus_tfidf_reproduces_the_studys_own_registered_figure_for_the_study_model():
    for v in ("v1", "v2"):
        late = LATE["arms"]["local"]["qwen3.8:27b"][f"{v}|dev"]["minus_tfidf"]["kappa"]
        study = METRICS["dev"][f"{v}_minus_tfidf_lr"]["kappa"]
        assert all(np.isclose(late[x], study[x], atol=1e-6) for x in ("value", "lo", "hi")), v
        acc = LATE["arms"]["local"]["qwen3.8:27b"][f"{v}|dev"]["accuracy_vs_record"]
        assert all(np.isclose(acc[x], METRICS["dev"][v]["accuracy"][x], atol=1e-6) for x in ("value", "lo", "hi"))


def test_the_tfidf_decisions_are_the_frozen_models_dev_decisions():
    from judge.data import load_records, load_splits
    records, splits = load_records(), load_splits()
    pred = L.tfidf_dev_decisions(records, splits)
    y = [records[i]["label"] for i in splits["dev"]]
    assert np.isclose(M.cohen_kappa(y, pred), METRICS["dev"]["tfidf_lr"]["kappa"]["value"], atol=1e-6)
    assert np.isclose(LATE["tfidf_dev_kappa"]["value"], METRICS["dev"]["tfidf_lr"]["kappa"]["value"], atol=1e-6)
