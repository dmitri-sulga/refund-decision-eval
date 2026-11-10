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
