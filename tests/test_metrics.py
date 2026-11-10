import math

import numpy as np
import pytest
from scipy.stats import binomtest
from sklearn.metrics import cohen_kappa_score

from judge import metrics as M


def test_kappa_matches_sklearn_on_random_data():
    rng = np.random.default_rng(1)
    for _ in range(50):
        n = int(rng.integers(5, 300))
        y = rng.integers(0, 2, n)
        p = np.where(rng.random(n) < 0.7, y, 1 - y)
        if len(set(y)) == 1 and len(set(p)) == 1:
            continue
        assert M.cohen_kappa(y, p) == pytest.approx(cohen_kappa_score(y, p), abs=1e-12)


def test_kappa_edge_cases():
    assert M.cohen_kappa([0, 1, 1, 0], [0, 1, 1, 0]) == pytest.approx(1.0)
    # constant prediction carries no information beyond chance
    assert M.cohen_kappa([0, 1, 1, 0], [0, 0, 0, 0]) == pytest.approx(0.0)
    # expected agreement of 1: kappa undefined -> NaN, never a fake 0 or 1
    assert math.isnan(M.cohen_kappa([1, 1, 1], [1, 1, 1]))


def test_precision_recall_by_hand():
    y = [1, 1, 0, 0, 1]
    p = [1, 0, 0, 1, 1]          # tp=2 fn=1 fp=1 tn=1
    m = M.point(y, p)
    assert m["accuracy"] == pytest.approx(3 / 5)
    assert m["precision_1"] == pytest.approx(2 / 3)
    assert m["recall_1"] == pytest.approx(2 / 3)
    assert m["precision_0"] == pytest.approx(1 / 2)
    assert m["recall_0"] == pytest.approx(1 / 2)


def test_undefined_precision_is_nan():
    m = M.point([1, 0, 1], [0, 0, 0])
    assert math.isnan(m["precision_1"])
    assert m["recall_1"] == 0.0


def test_bootstrap_is_deterministic_and_brackets_estimate():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    p = np.where(rng.random(200) < 0.8, y, 1 - y)
    a = M.bootstrap(y, p, B=2000, seed=5)
    b = M.bootstrap(y, p, B=2000, seed=5)
    assert a == b
    for k in ("accuracy", "kappa"):
        assert a[k]["lo"] <= a[k]["value"] <= a[k]["hi"]
    assert a["n"] == 200


def test_bootstrap_of_perfect_predictions_is_degenerate():
    y = [0, 1] * 20
    out = M.bootstrap(y, y, B=500, seed=0)
    assert out["accuracy"]["lo"] == out["accuracy"]["hi"] == 1.0


def test_paired_difference_of_identical_predictions_is_zero():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 100)
    p = rng.integers(0, 2, 100)
    d = M.paired_difference(y, p, p, B=500, seed=0)
    assert d["accuracy"] == {"value": 0.0, "lo": 0.0, "hi": 0.0}


@pytest.mark.parametrize("k,n", [(0, 10), (5, 10), (3, 30), (30, 30), (17, 250)])
def test_wilson_matches_scipy(k, n):
    ref = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    lo, hi = M.wilson(k, n)
    assert lo == pytest.approx(ref.low, abs=1e-9)
    assert hi == pytest.approx(ref.high, abs=1e-9)


def test_confidence_flags_contradictory_probability():
    c = M.confidence([1, 1, 0, 0], [0.9, 0.3, 0.2, 0.6])
    assert list(np.round(c, 6)) == [0.9, 0.3, 0.8, 0.4]


def test_selective_curve_and_threshold_rule():
    # confidences descending; the first three decisions agree with the label
    y = np.array([1, 0, 0, 1, 1, 0])
    pred = np.array([1, 0, 0, 0, 0, 1])
    conf = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6])
    curve = M.selective_curve(y, pred, conf)
    assert [p["n_auto"] for p in curve] == [1, 2, 3, 4, 5, 6]
    assert curve[2]["agreement"] == 1.0
    assert curve[3]["agreement"] == pytest.approx(0.75)
    assert M.choose_threshold(y, pred, conf, target=0.9, min_auto=2) == 0.9
    assert M.choose_threshold(y, pred, conf, target=0.9, min_auto=4) is None
    # ties at the threshold are automated together
    tied = M.selective_curve(y, pred, np.array([0.9] * 6))
    assert len(tied) == 1 and tied[0]["n_auto"] == 6


def test_invalid_outputs_are_never_automated():
    y = np.array([1, 0, 1])
    pred = np.array([1, 0, 0])
    conf = np.array([0.9, np.nan, 0.8])
    r = M.at_threshold(y, pred, conf, 0.5, B=200, seed=0)
    assert r["n_auto"] == 2
    assert r["coverage"]["value"] == pytest.approx(2 / 3)
    assert r["agreement"]["value"] == pytest.approx(0.5)


def test_the_two_kinds_of_automated_error_are_counted_apart():
    y = np.array([1, 0, 0, 1, 0])
    pred = np.array([0, 0, 1, 1, 1])
    conf = np.array([0.9, 0.9, 0.9, 0.9, 0.1])
    r = M.at_threshold(y, pred, conf, 0.5, B=200, seed=0)
    assert (r["n_auto"], r["wrong_no_relief"], r["wrong_relief"]) == (4, 1, 1)
    assert M.bootstrap(y, pred, B=50)["confusion"] == {"relief_relief": 1, "relief_none": 1,
                                                       "none_relief": 2, "none_none": 1}


def test_auc_and_within_group_auc_by_hand():
    y = [1, 0, 1, 0]
    assert M.auc(y, [0.9, 0.1, 0.8, 0.2]) == 1.0
    assert M.auc(y, [0.5, 0.5, 0.5, 0.5]) == 0.5
    assert math.isnan(M.auc([1, 1], [0.1, 0.2]))
    # inside each group the order is reversed; across groups one pair is in order
    y, s, g = [1, 0, 1, 0], [0.9, 0.95, 0.1, 0.2], ["a", "a", "b", "b"]
    assert M.within_group_auc(y, s, g) == 0.0 and M.auc(y, s) == 0.25


def test_threshold_tolerates_last_digit_noise():
    y, pred = np.array([0, 0, 1]), np.array([0, 0, 0])
    conf = np.array([0.7316803363204014, 0.9, 0.2])
    r = M.at_threshold(y, pred, conf, 0.7316803363204015, B=50, seed=0)
    assert r["n_auto"] == 2 and r["agreement_wilson"][0] < 1.0
