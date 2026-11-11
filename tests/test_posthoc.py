"""Exploratory analyses added after the held-out run (posthoc.py)."""
import json

import numpy as np
import pytest

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


def test_integrity_covers_the_labelled_answers_once_labels_are_locked(monkeypatch):
    m = {"frozen": {"sha256": P.GIST["revisions"][-1]["frozen_sha256"]},
         "test": {"commit": P.GIST["revisions"][-1]["commit"]}, "human": {"annotators": {"owner": {}}}}
    assert P.integrity(m)["problems"] == []
    f = "results/cache/responses.jsonl"
    monkeypatch.setitem(P.LABELLED_RUN_FILES, f, (P.LABELLED_RUN_FILES[f][0], "0" * 64))
    assert any("labelled-run commit" in p for p in P.integrity(m)["problems"])
    m["human"]["annotators"] = {}                       # before the lock the labelled answers did not exist
    assert P.integrity(m)["problems"] == []


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
    # Sixth change (after the held-out run and the labels): a section pointing to the SQL in sql/ adds 6 lines (a
    # blank line, the heading, a blank line, the block's two markers and its one paragraph), so the cap is 206. The
    # paragraph's figures come from metrics.json, so it has to be a generated block, and a block needs its markers.
    # Also in the sixth change: the deviations paragraph became a generated block ('deviations_in_short'), so that its
    # count of post-run changes comes from the records; that adds its two markers, so the cap is 208.
    assert len(lines) <= 208
    assert sum(ln.startswith("| Person (blind") for ln in lines) == 1      # the person's table prints once


def test_the_count_of_post_run_changes_comes_from_the_records_and_matches_the_deviations_list():
    """The README once said 'five times after the held-out run' after a sixth change, because the count was a
    hand-written word that the check does not see. It is generated now: the highest 'Nth change' the post-run records
    name, which must equal the number of post-run entries in docs/deviations.md and the changes the summary lists."""
    import re

    from judge import frontpage as F
    from judge import report as Rp

    m = json.loads((C.RESULTS / "metrics.json").read_text())
    n = F.post_run_change_count(m)
    entries = re.findall(r"^- \*\*After the held-out run, (?:presentation only \()?(\w+) change",
                         (C.ROOT / "docs" / "deviations.md").read_text(), re.M)
    assert entries == list(F.ORDINALS[:n]) and n >= 6
    assert set(F.CHANGES_IN_SHORT) == set(range(1, n + 1))
    readme = (C.ROOT / "README.md").read_text()
    block = readme.split("<!-- BEGIN deviations_in_short -->")[1].split("<!-- END deviations_in_short -->")[0]
    assert f"{F.TIMES[n]} after the held-out run" in block
    outside = Rp.BLOCK_RE.sub("", readme)
    assert not re.search(r"\b(?:" + "|".join(F.TIMES[2:]) + r"|once|twice) after the held-out run", outside)
    with pytest.raises(RuntimeError):            # a recorded change the summary does not describe stops the report
        F.b_deviations_in_short({"frozen": {"post_run_changes": [{"what": f"{F.ORDINALS[n].capitalize()} change: x"}]}})


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


def test_the_locked_labels_are_anchored_even_when_the_lock_is_resealed(tmp_path, monkeypatch):
    """Fifth change: a changed label file fails the check although a resealed lock matches it again."""
    import hashlib
    import shutil

    for f in P.LABEL_FILES:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(C.ROOT / f, tmp_path / f)
    assert P.label_file_problems(tmp_path) == []
    labels = tmp_path / "results/human_labels/owner.jsonl"
    lock = tmp_path / "results/human_labels/owner.lock.json"

    def reseal():
        x = json.loads(lock.read_text())
        x["labels_sha256"] = hashlib.sha256(labels.read_bytes()).hexdigest()
        lock.write_text(json.dumps(x, indent=1) + "\n")

    lines = labels.read_text(encoding="utf-8").splitlines(True)
    noted = next(k for k, ln in enumerate(lines) if json.loads(ln).get("note"))
    r = json.loads(lines[noted])
    r["note"] = r["note"] + "."                                        # the one free-text note, one byte longer
    lines[noted] = json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
    labels.write_text("".join(lines), encoding="utf-8")
    reseal()
    assert json.loads(lock.read_text())["labels_sha256"] == hashlib.sha256(labels.read_bytes()).hexdigest()
    got = P.label_file_problems(tmp_path)
    assert any("owner.jsonl" in p for p in got) and any("owner.lock.json" in p for p in got)

    shutil.copy(C.ROOT / "results/human_labels/owner.jsonl", labels)    # the history of the undo dropped instead
    kept = [ln for ln in labels.read_text(encoding="utf-8").splitlines(True) if not json.loads(ln).get("undo")]
    labels.write_text("".join(kept), encoding="utf-8")
    reseal()
    assert any("owner.jsonl" in p for p in P.label_file_problems(tmp_path))

    review = tmp_path / "results/human_labels/owner_error_review.jsonl"
    review.write_bytes(review.read_bytes().replace(b'"note": ""', b'"note": "x"', 1))
    assert any("owner_error_review.jsonl" in p for p in P.label_file_problems(tmp_path))

    m = {"frozen": {"sha256": P.GIST["revisions"][-1]["frozen_sha256"]},
         "test": {"commit": P.GIST["revisions"][-1]["commit"]}}
    f = "results/human_labels/owner.jsonl"
    monkeypatch.setitem(P.LABEL_FILES, f, (P.LABEL_FILES[f][0], "0" * 64))
    assert any("locked labels committed in" in p for p in P.integrity(m)["problems"])


def test_a_figure_whose_side_of_zero_moves_with_the_seed_is_left_out_of_the_check():
    from judge import frontpage as F

    clear = {"value": 0.2, "lo": 0.05, "hi": 0.35}
    edge = {"value": 0.216, "lo": 0.0, "hi": 0.435}
    wide = {"value": 0.19, "lo": -0.02, "hi": 0.4}
    out = F._sign_check([("a", clear, clear), ("v2", edge, wide)], unsettled={"v2"}, registered={"a", "v2"})
    assert "they agree in sign" in out and "For v2, a bound of the interval sits at zero" in out
    assert "disagree" not in out and "No conclusion is drawn" in out
    out = F._sign_check([("a", clear, wide), ("b", clear, wide)], registered={"a"})
    assert "interval for a." in out and "(descriptive): they disagree for b." in out


def test_seed_sides_name_only_figures_that_flip_and_reproduce_the_reported_interval():
    h = METRICS["posthoc"]["human"]["owner"]
    sides = h["seed_sides"]
    assert sides["seeds"] == list(P.SEEDS) and C.BOOTSTRAP_SEED in sides["seeds"]
    rep = METRICS["human"]["annotators"]["owner"]["all"]["systems_vs_person"]["v2"]["kappa"]
    x = sides["scopes"]["all"]["v2_vs_person:kappa"]["main"]
    assert x["lo_min"] - 1e-6 <= rep["lo"] <= x["lo_max"] + 1e-6 and x["hi_min"] - 1e-6 <= rep["hi"] <= x["hi_max"] + 1e-6
    for scope in sides["scopes"].values():
        for k, v in scope.items():
            assert v["settled"] == (len(v["main"]["sides"]) == 1 and len(v["sens"]["sides"]) == 1), k
