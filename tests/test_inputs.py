"""What the systems are shown, and what must never reach them."""
import json

from judge import config as C
from judge.baselines import classic_text
from judge.data import load_records, load_splits
from judge.label import render
from judge.llm import Cache
from judge.prompts import VERSIONS, build_messages, visible_narrative
from judge.replay import REPLAY_FILE

SECRET = {"complaint_id": "ID-SENTINEL", "date_received": "DATE-SENTINEL", "company": "COMPANY-SENTINEL",
          "company_response": "RESPONSE-SENTINEL", "label": "LABEL-SENTINEL",
          "product": "Credit card", "sub_product": "Store credit card", "issue": "An issue",
          "sub_issue": "A sub-issue", "narrative": "I was charged twice."}
HIDDEN = ("ID-SENTINEL", "DATE-SENTINEL", "COMPANY-SENTINEL", "RESPONSE-SENTINEL", "LABEL-SENTINEL")


def test_frozen_records_hold_no_field_written_after_the_decision_except_the_label():
    rec = next(iter(load_records().values()))
    assert set(rec) == {"complaint_id", "date_received", "product", "sub_product", "issue", "sub_issue",
                        "company", "company_response", "narrative", "label"}


def test_prompts_baseline_text_and_labelling_screen_show_only_header_and_narrative():
    shown = [classic_text(SECRET), render(SECRET, 1, 1)]
    for v in VERSIONS:
        messages, _ = build_messages(v, SECRET, visible_narrative(SECRET["narrative"]))
        shown.append(json.dumps(messages))
    for text in shown:
        assert "I was charged twice." in text and "Store credit card" in text
        assert not any(h in text for h in HIDDEN)


def test_llm_and_classic_baseline_get_the_same_text():
    messages, _ = build_messages("v1", SECRET, visible_narrative(SECRET["narrative"]))
    user = messages[1]["content"]
    header, narrative = classic_text(SECRET).split("\n")[:2], SECRET["narrative"]
    assert all(line in user for line in header) and narrative in user


def test_splits_are_disjoint_and_labels_follow_the_mapping():
    records, splits = load_records(), load_splits()
    seen = set()
    for name, ids in splits.items():
        assert len(ids) == len(set(ids)) and not seen & set(ids), name
        seen |= set(ids)
    assert seen == set(records)
    assert all(r["label"] == C.LABEL_MAP[r["company_response"]] for r in records.values())
    texts = {}
    for name, ids in splits.items():
        for i in ids:
            texts.setdefault(" ".join(records[i]["narrative"].lower().split()), set()).add(name)
    assert all(len(v) == 1 for v in texts.values()), "the same narrative appears in two splits"


def test_no_held_out_complaint_reached_the_model_before_the_held_out_run():
    marker = C.RESULTS / "test_run.json"
    if marker.exists():
        return
    test_ids = set(load_splits()["test"])
    for path in (C.CACHE_FILE, REPLAY_FILE):
        if path.exists():
            for r in Cache(path).records.values():
                assert r["item_id"].split("~")[0] not in test_ids
    bench = C.RESULTS / "cache" / "concurrency_replays.jsonl"
    if bench.exists():
        for line in bench.read_text().splitlines():
            assert json.loads(line)["item_id"] not in test_ids
