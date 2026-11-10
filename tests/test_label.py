import json

import pytest

from judge import label as L
from judge.data import load_splits


def test_sample_file_is_the_seeded_draw_and_stays_inside_dev():
    sample = L.load_sample()
    assert sample == L.make_sample(sample["split"], sample["n"], sample["seed"])
    assert sample["split"] == "dev" and len(set(sample["ids"])) == sample["n"] == 100
    assert set(sample["ids"]) <= set(load_splits()["dev"])


def test_held_out_complaints_cannot_be_sampled_before_the_held_out_run(monkeypatch):
    monkeypatch.setattr(L, "_heldout_finished", lambda: False)
    with pytest.raises(SystemExit):
        L.make_sample("test")


def test_session_records_resumes_and_undoes(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "LABELS_DIR", tmp_path)
    ids = L.load_sample()["ids"]
    shown = []

    def run(answers):
        it = iter(answers)
        return L.session("t", ask=lambda _: next(it), show=shown.append, clock=lambda: 0.0)

    assert run(["1", "", "x", "3", "unclear", "q"]) == 2          # 'x' is not an answer and is asked again
    path = tmp_path / "t.jsonl"
    labels = L.load_labels(path)
    assert [labels[i]["label"] for i in ids[:2]] == ["relief", "cannot_tell"]
    assert labels[ids[1]]["note"] == "unclear"
    assert "Case 3 of 100" in shown[-2]                           # third case was shown, then quit
    assert run(["2", "", "u", "1", "", "q"]) == 2                 # resume at case 3, undo it, answer again
    labels = L.load_labels(path)
    assert labels[ids[2]]["label"] == "relief" and len(labels) == 3
    assert all(set(json.loads(line)) <= {"complaint_id", "label", "note", "annotator", "seconds",
                                         "labelled_at", "undo"} for line in path.read_text().splitlines())


def test_screen_shows_no_id_and_no_outcome():
    from judge.data import load_records
    cid = L.load_sample()["ids"][0]
    rec = load_records()[cid]
    screen = L.render(rec, 1, 100)
    assert cid not in screen and rec["company_response"] not in screen and rec["date_received"] not in screen
