"""The real blind labels, committed and locked: the analysis guard passes on them. Read only."""
import hashlib
import json

from judge import config as C
from judge.llm import Cache

DIR = C.RESULTS / "human_labels"


def _labels():
    out = {}
    for line in (DIR / "owner.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out.pop(r["complaint_id"], None)
            if not r.get("undo"):
                out[r["complaint_id"]] = r
    return out


def test_the_real_lock_matches_its_file_and_the_registered_sample():
    lock = json.loads((DIR / "owner.lock.json").read_text())
    assert hashlib.sha256((DIR / "owner.jsonl").read_bytes()).hexdigest() == lock["labels_sha256"]
    ids = json.loads((C.DATA / "label_sample.json").read_text())["ids"]
    labels = _labels()
    assert set(labels) == set(ids) and len(ids) == lock["n"] == 100
    assert all(r["labelled_at"] <= lock["locked_at"] for r in labels.values())


def test_every_model_answer_for_a_labelled_complaint_follows_the_real_lock():
    lock = json.loads((DIR / "owner.lock.json").read_text())
    ids = set(json.loads((C.DATA / "label_sample.json").read_text())["ids"])
    answers = [r for r in Cache().records.values() if r["item_id"].split("~")[0] in ids]
    assert {(r["prompt_version"], r["item_id"]) for r in answers} == {(v, i) for v in ("v1", "v2") for i in ids}
    assert all(r["requested_at"] >= lock["locked_at"] for r in answers)
