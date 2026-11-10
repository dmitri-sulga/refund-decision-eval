import json

import pytest

from judge import freeze, llm
from judge import run as R

FROZEN = {"final_versions": ["v1", "v2"], "guarded_facts_version": "v2", "workers": 1,
          "expected_s_per_call": 6.0, "environment": {"ollama_version": "x", "model_digest": "d"}}


@pytest.fixture
def world(tmp_path, monkeypatch):
    import judge.replay as P

    calls = []
    frozen_file = tmp_path / "frozen.json"
    frozen_file.write_text(json.dumps(FROZEN))
    monkeypatch.setattr(freeze, "FROZEN_FILE", frozen_file)
    monkeypatch.setattr(freeze, "verify_method", lambda f: [])
    monkeypatch.setattr(llm, "environment", lambda: {"ollama_version": "x", "model_digest": "d"})

    class EmptyCache(llm.Cache):
        def __init__(self, *a, **k):
            self.records = {}

    monkeypatch.setattr(llm, "Cache", EmptyCache)
    monkeypatch.setattr(R, "TEST_MARKER", tmp_path / "test_run.json")
    monkeypatch.setattr(R, "git_state", lambda: ("abc123", []))
    monkeypatch.setattr(R, "log_event", lambda e: calls.append(("log", e["event"])))
    monkeypatch.setattr(R, "run", lambda v, items, **k: calls.append(("run", v, len(list(items)))) or
                        {"version": v, "new_calls": 0})
    monkeypatch.setattr(P, "replay", lambda v, ids, records, label="", **k: calls.append(("noise", v, len(ids))))
    monkeypatch.setattr(R, "SENT_FILE", tmp_path / "sent.jsonl")
    return calls, tmp_path


def test_dry_run_checks_everything_and_calls_nothing(world, capsys):
    calls, tmp = world
    R.final_test_run(dry_run=True)
    assert calls == [] and not (tmp / "test_run.json").exists()
    plan = json.loads(capsys.readouterr().out.split("dry run")[0])
    assert plan["held_out_complaints"] == 250 and plan["noise_pass_calls"] == 500 and plan["commit"] == "abc123"


def test_refuses_without_a_frozen_method(world):
    _, tmp = world
    (tmp / "frozen.json").unlink()
    with pytest.raises(SystemExit, match="frozen"):
        R.final_test_run()


def test_refuses_when_a_frozen_file_or_the_model_build_changed(world, monkeypatch):
    calls, _ = world
    monkeypatch.setattr(freeze, "verify_method", lambda f: ["src/judge/prompts.py changed since the freeze"])
    with pytest.raises(SystemExit, match="prompts.py"):
        R.final_test_run()
    monkeypatch.setattr(freeze, "verify_method", lambda f: [])
    monkeypatch.setattr(llm, "environment", lambda: {"ollama_version": "y", "model_digest": "d"})
    with pytest.raises(SystemExit, match="ollama_version"):
        R.final_test_run()
    assert calls == []


def test_runs_once_scored_pass_then_noise_pass_and_then_refuses(world):
    calls, tmp = world
    R.final_test_run()
    assert [c for c in calls if c[0] != "log"] == [("run", "v1", 250), ("run", "v2", 250),
                                                    ("noise", "v1", 250), ("noise", "v2", 250)]
    state = json.loads((tmp / "test_run.json").read_text())
    assert state["finished_at"] and state["noise_finished_at"] and state["commit"] == "abc123"
    with pytest.raises(SystemExit, match="already run"):
        R.final_test_run()


def test_refuses_to_start_from_uncommitted_changes(world, monkeypatch):
    calls, _ = world
    monkeypatch.setattr(R, "git_state", lambda: ("abc123", ["src/judge/prompts.py"]))
    with pytest.raises(SystemExit, match="uncommitted"):
        R.final_test_run()
    assert calls == []


def test_verify_method_notices_changed_files_caches_and_rules():
    import copy

    frozen = {"files": freeze.file_hashes(), "caches": freeze.cache_fingerprints(),
              "final_versions": list(freeze.FINAL_VERSIONS), "claims": copy.deepcopy(freeze.CLAIMS),
              "request_templates": freeze.template_hashes(freeze.FINAL_VERSIONS),
              "heldout": dict(zip(("n", "ids_sha256"), freeze.heldout_ids_sha256()))}
    assert freeze.verify_method(frozen) == []
    assert {"src/judge/analysis.py", "src/judge/report.py", "src/judge/cues.py", "src/judge/label.py",
            "requirements.txt", "data/label_sample.json", "docs/analysis_plan.md"} <= set(frozen["files"])
    frozen["files"]["src/judge/analysis.py"] = "0" * 64
    frozen["heldout"]["ids_sha256"] = "0" * 64
    frozen["caches"]["results/cache/responses.jsonl"]["sha256"] = "0" * 64
    frozen["claims"][0]["rule"] += " (edited)"
    assert len(freeze.verify_method(frozen)) == 4
