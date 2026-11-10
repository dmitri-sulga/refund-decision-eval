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


def test_first_screen_fits_in_the_first_ten_lines_of_the_readme():
    lines = (C.ROOT / "README.md").read_text().splitlines()
    assert lines[2] == "<!-- BEGIN first_screen -->" and "<!-- END first_screen -->" in lines[:10]
    head = "\n".join(lines[:10])
    for phrase in ("Always answering", "would pay", "follows which issuer answered", "goodwill", "regulator"):
        assert phrase in head, phrase
    assert len(lines) <= 150
