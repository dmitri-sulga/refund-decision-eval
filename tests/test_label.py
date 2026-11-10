import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from judge import label as L
from judge import review as V
from judge.data import load_records, load_splits
from judge.translate import problems

SHOWN_KEYS = {"header_en", "header_ru", "narrative_en", "narrative_ru", "translation_problems"}


def test_sample_file_is_the_registered_hash_draw_from_train():
    sample = L.load_sample()
    assert sample == L.make_sample()
    assert sample["split"] == "train" and len(set(sample["ids"])) == sample["n"] == 100
    assert set(sample["ids"]) <= set(load_splits()["train"])


def test_no_sampled_complaint_appears_in_an_annotation_or_a_report():
    import csv

    from judge import config as C
    ids = set(L.load_sample()["ids"])
    with open(C.RESULTS / "error_analysis" / "v1_dev.csv", newline="") as f:
        assert not ids & {r["complaint_id"] for r in csv.DictReader(f)}
    for name in ("README.md", "results/report.md"):
        text = (C.ROOT / name).read_text()
        assert not any(i in text for i in ids), name


def test_the_judge_cannot_have_answered_the_sample_before_the_labels_are_locked():
    from judge.llm import Cache

    if any(L.LABELS_DIR.glob("*.lock.json")):
        return
    ids = set(L.load_sample()["ids"])
    assert not any(r["item_id"].split("~")[0] in ids for r in Cache().records.values())


def test_the_view_holds_only_what_the_judge_sees():
    cid = L.load_sample()["ids"][0]
    rec = load_records()[cid]
    view = L.case_view(rec, None)
    assert set(view) == SHOWN_KEYS
    blob = json.dumps(view, ensure_ascii=False)
    assert cid not in blob and rec["company_response"] not in blob and rec["date_received"] not in blob
    screen = L.render(rec, 1, 100)
    assert cid not in screen and rec["company_response"] not in screen


def test_terminal_session_records_resumes_and_undoes(tmp_path, monkeypatch):
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
    assert labels[ids[1]]["note"] == "unclear" and labels[ids[1]]["ui"] == "terminal"
    assert "Case 3 of 100" in shown[-2]                           # third case was shown, then quit
    assert run(["2", "", "q"]) == 1                               # resumes at case 3
    assert run(["u", "1", "", "q"]) == 1                          # undo case 3 and answer it again
    labels = L.load_labels(path)
    assert labels[ids[2]]["label"] == "relief" and len(labels) == 3
    assert not L.is_locked("t")


def _label_all(annotator, n):
    records = load_records()
    for _ in range(n):
        state = L.next_case(annotator, records, None)
        L.record_answer(annotator, state["position"], "no_relief", "", 1.0, "test", state)


def test_answers_lock_after_the_last_case_and_cannot_change(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "LABELS_DIR", tmp_path)
    _label_all("t", 99)
    assert not L.is_locked("t") and L.undo_last("t")              # undo works until the lock
    with pytest.raises(ValueError, match="not the next"):
        L.record_answer("t", 5, "relief", "", 1.0, "test", L.next_case("t", load_records(), None))
    _label_all("t", 2)
    assert L.is_locked("t") and "position" not in L.next_case("t", load_records(), None)
    with pytest.raises(ValueError, match="locked"):
        L.undo_last("t")
    with open(tmp_path / "t.jsonl", "a") as f:                    # tampering after the lock is detected
        f.write("\n")
    with pytest.raises(SystemExit, match="changed after"):
        L.is_locked("t")


def test_second_screen_opens_only_after_the_lock(tmp_path, monkeypatch):
    from judge.llm import Cache

    monkeypatch.setattr(L, "LABELS_DIR", tmp_path)
    records, cache = load_records(), Cache()
    with pytest.raises(ValueError, match="not locked"):
        V.next_review("t", records, cache, None)
    _label_all("t", 100)
    state = V.next_review("t", records, cache, None)
    assert state["total"] == 24 and state["position"] == 1 and state["recorded"].startswith("Closed")
    V.record_review("t", 1, "undecidable", "my note")
    saved = V.load_reviews(V.review_path("t"))
    row = next(iter(saved.values()))
    assert row["group"] == "undecidable" and row["confirmed"] == (row["drafted_group"] == "undecidable")
    assert V.next_review("t", records, cache, None)["position"] == 2
    assert L.is_locked("t")                                       # the review does not touch the blind labels


def test_labelling_page_serves_blind_cases_and_saves_answers(tmp_path, monkeypatch):
    from judge.labelserver import make_handler

    monkeypatch.setattr(L, "LABELS_DIR", tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler("t"))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read()) if "json" in r.headers["Content-Type"] else r.read().decode()
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    try:
        status, page = call("/")
        assert status == 200 and "Вернуть деньги" in page
        assert "Правила слепой разметки" in call("/rubric")[1]
        status, state = call("/api/next")
        assert set(state) == SHOWN_KEYS | {"total", "done", "locked", "can_undo", "position"}
        assert call("/api/answer", {"position": 1, "label": "relief", "note": "чек", "seconds": 12.3})[0] == 200
        assert call("/api/answer", {"position": 1, "label": "relief"})[0] == 400      # not the next case
        assert call("/api/review/next")[0] == 400                                       # blind pass not locked
        saved = L.load_labels(tmp_path / "t.jsonl")
        row = saved[L.load_sample()["ids"][0]]
        assert row["label"] == "relief" and row["note"] == "чек" and row["seconds"] == 12.3 and row["ui"] == "web"
        assert call("/api/undo", {})[0] == 200 and not L.load_labels(tmp_path / "t.jsonl")
    finally:
        server.shutdown()


def test_translation_checks_flag_incomplete_translations():
    src = "I was charged {$45.00} twice by XXXX and the bank refused to help me with this problem."
    good = {"content": "С меня дважды списали {$45.00} в XXXX, и банк отказался помочь мне с этой проблемой."}
    assert problems(src, good) == []
    assert "dollar amounts differ from the original" in problems(src, {**good, "content": good["content"].replace("45", "54")})
    assert "length differs a lot from the original" in problems(src, {"content": good["content"][:30]})
    assert "not mostly Russian" in problems(src, {"content": src})
