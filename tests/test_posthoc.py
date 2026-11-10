"""Exploratory analyses added after the held-out run (posthoc.py)."""
import json

import numpy as np

from judge import config as C
from judge import metrics as M
from judge import posthoc as P
from judge import report as Rp

METRICS = json.loads((C.RESULTS / "metrics.json").read_text())


def test_paired_matches_the_frozen_paired_difference_at_95_percent():
    rng = np.random.default_rng(0)
    y, a, b = (rng.integers(0, 2, 120) for _ in range(3))
    x, ref = P.paired(y, a, b), M.paired_difference(y, a, b)["kappa"]
    assert np.isclose(x["value"], ref["value"]) and np.isclose(x["lo"], ref["lo"]) and np.isclose(x["hi"], ref["hi"])
    wide = P.paired(y, a, b, level=1 - 0.05 / 6)
    assert wide["lo"] <= x["lo"] and wide["hi"] >= x["hi"]


def test_issuer_sentences_use_names_from_the_published_top_features():
    top = METRICS["issuer"]["top_features"]
    for key, side in (("names_towards_relief", "towards_relief"), ("names_towards_no_relief", "towards_no_relief")):
        names = P.TFIDF_SENTENCES[key].split("with ", 1)[1].rstrip(".").replace(" and ", ", ").split(", ")
        assert len(names) == 4
        assert all(any(w in top[side] for w in n.lower().split()) for n in names), names


def test_tfidf_gaming_is_reproducible_on_dev():
    from judge.analysis import Systems
    from judge.data import load_records, load_splits
    from judge.llm import Cache

    S = Systems(load_records(), load_splits(), Cache(), lr_params=METRICS["tfidf_lr"]["chosen"])
    got = P.tfidf_gaming(S)
    want = METRICS["posthoc"]["tfidf_gaming"]
    for name, s in want["sets"].items():
        assert got["sets"][name]["base_denials"] == s["base_denials"]
        for a, r in s["rows"].items():
            assert (got["sets"][name]["rows"][a]["to_relief"], got["sets"][name]["rows"][a]["to_no_relief"]) == \
                (r["to_relief"], r["to_no_relief"]), a
    assert set(got["sets"]["gaming_subset"]["rows"]) >= {"neutral", "names_towards_relief", "names_towards_no_relief"}


def test_integrity_fails_on_a_changed_answer_file_or_gist_hash(monkeypatch):
    m = {"frozen": {"sha256": P.GIST["revisions"][-1]["frozen_sha256"]},
         "test": {"commit": P.GIST["revisions"][-1]["commit"]}}
    assert P.integrity(m)["problems"] == []
    f = "results/cache/sent.jsonl"
    monkeypatch.setitem(P.RUN_FILES, f, (P.RUN_FILES[f][0], "0" * 64))
    assert any("sent.jsonl" in p for p in P.integrity(m)["problems"])
    monkeypatch.setitem(P.RUN_FILES, f, (10 ** 9, P.RUN_FILES[f][1]))
    assert P.integrity(m)["problems"]
    m["frozen"]["sha256"] = "1" * 64
    assert any("gist" in p for p in P.integrity(m)["problems"])


def test_a_file_added_after_the_freeze_is_accepted_only_when_recorded(tmp_path, monkeypatch):
    from judge import freeze

    new = "src/judge/posthoc.py"
    sha = freeze.sha256_path(C.ROOT / new)
    problems = [f"{new} changed since the freeze", "src/judge/cues.py changed since the freeze"]

    def world(records):
        f = tmp_path / "post_run_changes.json"
        f.write_text(json.dumps(records))
        monkeypatch.setattr(Rp, "POST_RUN_FILE", f)
        return {"test": {"n": 1}, "frozen": {"sha256": "x", "content": {"files": {"src/judge/cues.py": "c" * 64}},
                                             "problems": list(problems)}}

    rec = {"file": new, "frozen_sha256": None, "sha256": sha, "what": "added after the run"}
    assert Rp.accept_post_run_changes(world([rec]))["frozen"]["problems"] == problems[1:]
    assert Rp.accept_post_run_changes(world([]))["frozen"]["problems"] == problems
    assert Rp.accept_post_run_changes(world([{**rec, "sha256": "d" * 64}]))["frozen"]["problems"] == problems
    cues = {"file": "src/judge/cues.py", "frozen_sha256": "c" * 64,
            "sha256": freeze.sha256_path(C.ROOT / "src/judge/cues.py"), "what": "x"}
    assert Rp.accept_post_run_changes(world([rec, cues]))["frozen"]["problems"] == problems[1:]   # not presentation


def test_first_screen_fits_in_the_first_twelve_lines_of_the_readme():
    lines = (C.ROOT / "README.md").read_text().splitlines()
    assert lines[2] == "<!-- BEGIN first_screen -->" and "<!-- END first_screen -->" in lines[:12]
    head = "\n".join(lines[:12])
    for phrase in ("Always answering", "would pay", "follows which issuer answered", "goodwill", "regulator",
                   "not a model that grades other models"):
        assert phrase in head, phrase
    for phrase in ("the best system", "+3.6", "quiet", "coin toss."):    # post-hoc pick and unmeasured causes
        assert phrase not in head, phrase
    # The cap was 170 while the human-label block was a 5-line placeholder. The locked labels add the registered
    # output once (the person against the record, the systems against the person, the paired differences, the
    # check of the two readings: 35 lines), a 4-line summary in the held-out section and the markers of the
    # 'who' block; nothing else grew. The cap is the length that needs, so any repeat of a table fails it.
    assert len(lines) <= 200
    assert sum(ln.startswith("| Person (blind") for ln in lines) == 1      # the person's table prints once


def test_the_registered_check_of_the_two_readings_flags_a_sign_change_or_a_crossing_of_zero():
    from judge import frontpage as F

    same = {"value": 0.2, "lo": 0.05, "hi": 0.35}
    assert "agree in sign" in F._sign_check([("a", same, {"value": 0.15, "lo": 0.01, "hi": 0.3})])
    out = F._sign_check([("a", same, {"value": 0.1, "lo": -0.02, "hi": 0.25}),
                         ("b", {"value": 0.007, "lo": -0.1, "hi": 0.1}, {"value": -0.034, "lo": -0.15, "hi": 0.08})])
    assert "disagree" in out and "a; b" in out and "No conclusion is drawn" in out


def test_within_issuer_by_issuer_resamples_issuers_and_matches_the_frozen_point():
    ph = json.loads((C.RESULTS / "metrics.json").read_text())["posthoc"]
    wb = ph["within_issuer_by_issuer"]
    sy = json.loads((C.RESULTS / "metrics.json").read_text())["test"]["ranking"]["systems"]
    assert wb["issuers_with_pairs"] < 20 and wb["B"] == C.BOOTSTRAP_B
    for k, x in wb["systems"].items():
        assert abs(x["value"] - sy[k]["within_issuer_auc"]["value"]) < 1e-6 and x["lo"] <= x["value"] <= x["hi"]
