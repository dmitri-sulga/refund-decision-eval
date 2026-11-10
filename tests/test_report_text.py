"""Generated text that must stay true after the held-out run."""
import copy
import json

from judge import analysis as A
from judge import config as C
from judge import report as Rp

METRICS = json.loads((C.RESULTS / "metrics.json").read_text())


def test_lead_says_whether_the_held_out_split_has_been_run():
    m = copy.deepcopy(METRICS)
    m["test"] = None
    before = Rp.b_lead(m)
    assert "the held-out split has not been run" in before
    m["test"] = {"results": {}}
    after = Rp.b_lead(m)
    assert "has not been run" not in after and "the held-out results are below" in after


def test_lead_states_the_masked_comparison_as_the_intervals_allow():
    m = copy.deepcopy(METRICS)
    lead = Rp.b_lead(m)
    assert "mostly" not in lead
    above = all(m["dev"][f"tfidf_masked_minus_{v}"]["kappa"]["lo"] > 0 for v in ("v1", "v2"))
    assert ("still above both judges" in lead) == above
    m["dev"]["tfidf_masked_minus_v2"]["kappa"]["lo"] = -0.01
    assert "its lead over the judges is not demonstrated" in Rp.b_lead(m)


def test_cost_table_files_noise_pass_calls_under_the_held_out_split(tmp_path, monkeypatch):
    def rec(version, item, at):
        return {"prompt_version": version, "item_id": item, "wall_s": 2.0, "requested_at": at,
                "done_reason": "stop", "eval_count": 5, "prompt_eval_count": 10}

    class Fake:
        def __init__(self, recs):
            self.records = {f"{r['prompt_version']}|{r['item_id']}": r for r in recs}

    splits = {"dev": ["1"], "test": ["2"], "train": [], "debug": []}
    cache = Fake([rec("v1", "1", "2026-01-01T00:00:00"), rec("v1", "2", "2026-01-02T00:00:00")])
    replays = Fake([rec("v1@replay", "1", "2026-01-03T00:00:00"), rec("v1@replay", "2", "2026-01-04T00:00:00")])
    log = tmp_path / "run_log.jsonl"
    log.write_text(json.dumps({"event": "translation", "new_calls": 3}) + "\n")
    monkeypatch.setattr(A, "_replays", lambda: replays)
    monkeypatch.setattr(A, "COST_FILE", tmp_path / "missing.json")
    monkeypatch.setattr(A, "compare_concurrency", lambda r, c: [])
    monkeypatch.setattr(A, "load_concurrency_replays", lambda: [])
    monkeypatch.setattr(C, "RUN_LOG", log)
    c = A.cost(cache, splits)
    sets = {(r["version"], r["set"]): r["calls"] for r in c["by_run"]}
    assert sets == {("v1", "dev"): 1, ("v1", "test"): 1, ("v1@replay", "dev (replay file)"): 1,
                    ("v1@replay", "test noise pass (replay file)"): 1}
    assert c["all_calls"]["held_out_calls"] == 2 and c["translation_calls_not_counted"] == 3


def test_headline_states_each_frozen_claim_as_its_rule_decides():
    m = copy.deepcopy(METRICS)
    head = Rp.b_headline(m)
    assert head.startswith("**Held-out run**") and "Result so far" not in head
    assert "*v2 replaces v1*" in head and "**no improvement of v2 over v1 was demonstrated**" in head
    assert "**the frozen TF-IDF operating point: not shown to meet the target**" in head
    assert "outcome words were chosen after the run" not in head
    assert "| Person" not in head                      # the person's full tables print once, in the human section
    if m["human"]["annotators"]:
        f = m["human"]["annotators"]["owner"]["all"]
        assert "**Blind human labels**" in head and "Blind human labels are being collected" not in head
        assert f"could not decide {f['cannot_tell']} of them from the text (registered)" in head
        assert "agrees with the record about as weakly as the systems do" in head
        assert "no difference between the person and any system was tested" in head
        assert "no system is shown to agree with the person" in head and "Registered check of the two readings" in head
        assert "not a lawyer" in head and "machine translation" in head
        assert len([ln for ln in head.split("**Blind human labels**")[1].splitlines() if ln.strip()]) <= 4
    pending = copy.deepcopy(m)
    pending["human"]["annotators"] = {}
    assert "Blind human labels are being collected" in Rp.b_headline(pending)   # before the lock
    for c in m["test"]["claims"]:
        if c["id"] == "v2_vs_v1":
            c["verdict"] = "v2 replaces v1"
        if c["id"] == "tfidf_operating_point":
            c["verdict"] = "the frozen TF-IDF operating point: misses the target"
    head = Rp.b_headline(m)
    assert "*v2 replaces v1* (only if the paired kappa difference, v2 minus v1, lies entirely above zero): " \
           "**v2 replaces v1**" in head
    assert "**the frozen TF-IDF operating point: misses the target**" in head
    m["test"] = None
    assert Rp.b_headline(m) == Rp.b_lead(m)


def test_practice_says_what_is_not_measured_while_labels_are_pending():
    m = copy.deepcopy(METRICS)
    text = Rp.b_practice(m)
    if not m["human"]["annotators"]:
        assert "a person: not measured until the blind labels are locked" in text
    else:
        assert "the median time per label" in text and "times 1,000, which leaves out breaks" in text
    assert text.startswith("Only the claims above were frozen before the run. Everything in this section is exploratory")
    assert "Only the prompt that asks the model to predict the record is moved" not in text
    assert "at least as well" not in text and "0 of 10 other wordings" in text
    assert "quiet" not in text and "issuer habit shows in the wording" not in text and "the same rates" not in text
    m["test"] = None
    assert Rp.b_practice(m) == Rp.PENDING


def test_post_run_change_is_accepted_only_for_a_recorded_presentation_file(tmp_path, monkeypatch):
    from judge import freeze

    report_sha = freeze.sha256_path(C.ROOT / "src/judge/report.py")
    frozen_files = {"src/judge/report.py": "a" * 64, "src/judge/analysis.py": "b" * 64}
    problems = ["src/judge/report.py changed since the freeze", "src/judge/analysis.py changed since the freeze"]

    def world(records, test=True):
        f = tmp_path / "post_run_changes.json"
        f.write_text(json.dumps(records))
        monkeypatch.setattr(Rp, "POST_RUN_FILE", f)
        return {"test": {"n": 1} if test else None,
                "frozen": {"sha256": "x", "content": {"files": frozen_files}, "problems": list(problems)}}

    rec = {"file": "src/judge/report.py", "frozen_sha256": "a" * 64, "sha256": report_sha, "what": "presentation"}
    out = Rp.accept_post_run_changes(world([rec]))
    assert out["frozen"]["problems"] == problems[1:] and [x["file"] for x in out["frozen"]["post_run_changes"]] == \
        ["src/judge/report.py"]
    assert Rp.accept_post_run_changes(world([rec], test=False))["frozen"]["problems"] == problems   # before the run
    assert Rp.accept_post_run_changes(world([{**rec, "sha256": "c" * 64}]))["frozen"]["problems"] == problems
    assert Rp.accept_post_run_changes(world([{**rec, "frozen_sha256": "c" * 64}]))["frozen"]["problems"] == problems
    other = {**rec, "file": "src/judge/analysis.py", "frozen_sha256": "b" * 64,
             "sha256": freeze.sha256_path(C.ROOT / "src/judge/analysis.py")}
    assert Rp.accept_post_run_changes(world([other]))["frozen"]["problems"] == problems   # not a presentation file
