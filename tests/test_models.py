"""The local-models extension (docs/analysis_plan_models.md, Amendment 1). Every test uses a fake Ollama client or
no client at all; nothing here contacts Ollama."""
import hashlib
import io
import json
import subprocess
import urllib.error

import numpy as np
import pytest

from judge import config as C
from judge import freeze, llm
from judge import models as X
from judge.prompts import build_messages


def _body_without_extension(version):
    messages, schema = build_messages(version, freeze._PROBE, "narrative")
    return llm.request_body(messages, schema)


# ------------------------------------------------------------------ the registered text and the request
def test_registered_lines_equal_the_registered_commit():
    blob = subprocess.run(["git", "-C", str(C.ROOT), "show", f"{X.REGISTERED_COMMIT}:docs/analysis_plan_models.md"],
                          capture_output=True, check=True).stdout
    assert blob.count(b"\n") == X.REGISTERED_LINES
    assert hashlib.sha256(blob).hexdigest() == X.REGISTERED_SHA256 == X.registered_part_sha256()
    assert X.PLAN_FILE.read_bytes().startswith(blob)


def test_only_the_model_and_three_neutral_penalties_differ_from_the_frozen_request():
    frozen = freeze.load()["request_templates"]
    for v in X.VERSIONS:
        b = X.build_body("phi4:14b", v, freeze._PROBE, "narrative")
        # put the study model back and take the penalties out: the frozen template hash
        back = {**b, "model": C.MODEL, "options": {k: x for k, x in b["options"].items() if k not in X.NEUTRAL_PENALTIES}}
        assert X.request_hash(back) == llm.request_hash(back) == frozen[v]
        assert b["options"] == {**C.OPTIONS, "presence_penalty": 0, "frequency_penalty": 0, "repeat_penalty": 1}
        assert {k for k in b if b[k] != _body_without_extension(v)[k]} == {"model", "options"}
        assert C.OPTIONS == {"temperature": 0, "seed": 42, "num_ctx": 8192, "num_predict": 512}  # not mutated


def test_dropping_think_or_logprobs_removes_only_those_fields():
    b = X.build_body("phi4:14b", "v1", freeze._PROBE, "narrative", drop=["think", "logprobs"])
    assert "think" not in b and "logprobs" not in b and "top_logprobs" not in b and "messages" in b


def test_the_model_table_matches_the_environment_dump_and_the_order_is_fixed():
    env = json.loads(X.ENV_FILE.read_text())
    assert [t for t, _ in X.MODELS] == ["qwen3.5:9b", "gemma4:12b", "phi4:14b", "mistral-small3.2:24b",
                                       "granite4.1:30b", "glm-4.7-flash:latest", "gemma4:31b"]
    assert env["ollama_version"] == X.OLLAMA_VERSION
    for t, d in (*X.MODELS, X.REFERENCE):
        assert env["models"][t]["digest"] == d
    assert X.verify() == []          # the real repository, before any request


# ------------------------------------------------------------------ a sandbox with a fake client
class FakeClient:
    def __init__(self, fail_at=None, error=None, version="0.40.15", digests=None):
        self.calls, self.fail_at, self.error, self._version = [], fail_at, error, version
        self.digests = digests or {**dict(X.MODELS), X.REFERENCE[0]: X.REFERENCE[1]}

    def version(self):
        return self._version

    def tags(self):
        return dict(self.digests)

    def show(self, tag):
        return json.loads(X.ENV_FILE.read_text())["models"][tag]["show"]

    def chat(self, body):
        self.calls.append(body)
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise self.error
        return {"model": body["model"], "message": {"content": json.dumps({"monetary_relief": len(self.calls) % 3 == 0,
                                                                          "p_relief": 0.5, "reason": "x"})},
                "done_reason": "stop", "prompt_eval_count": 700, "eval_count": 40}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    for name in ("CACHE_FILE", "JOURNAL", "RUN_LOG", "REGISTRATION", "LOG_EXTRACT"):
        monkeypatch.setattr(X, name, tmp_path / getattr(X, name).name)
    monkeypatch.setattr(X, "OLLAMA_LOG", str(tmp_path / "server.log"))
    (tmp_path / "server.log").write_text('[GIN] 2026/11/11 | 200 | 1ms | 127.0.0.1 | GET "/api/tags"\n')
    sets = {"dev": ["d1", "d2", "d3"], "labelled": ["l1", "l2"]}
    test = {"h1"}
    records = {i: {"product": "P", "sub_product": "S", "issue": "I", "sub_issue": "J", "narrative": f"text {i}",
                   "label": 0} for i in ["d1", "d2", "d3", "l1", "l2", "h1"]}
    pre = {"sets": sets, "test": test, "ollama_version": "0.40.15"}
    monkeypatch.setattr(X, "preflight", lambda tag, client, published_head=None: {**pre, "state": X.state()})
    return {"tmp": tmp_path, "records": records, "sets": sets}


def test_a_full_model_journals_every_request_before_its_answer_and_verifies(sandbox, monkeypatch):
    c = FakeClient()
    out = X.run_model("qwen3.5:9b", c, records=sandbox["records"])
    assert out == {"model": "qwen3.5:9b", "new_answers": 10, "missing": 0} and len(c.calls) == 10
    journal = X._jsonl(X.JOURNAL)
    cache = X._jsonl(X.CACHE_FILE)
    assert [e["key"] for e in journal] == [r["key"] for r in cache]
    assert [X.split_key(r["key"])[1:] for r in cache][:3] == [("v1", "d1"), ("v1", "d2"), ("v1", "d3")]
    assert all(e["at"] <= r["requested_at"] for e, r in zip(journal, cache))
    assert all(r["model"] == "qwen3.5:9b" and r["model_digest"] == X.DIGESTS["qwen3.5:9b"] for r in cache)
    assert all(b["options"]["presence_penalty"] == 0 and b["model"] == "qwen3.5:9b" for b in c.calls)
    assert X.LOG_EXTRACT.exists()
    assert X.state()["status"] == {"qwen3.5:9b": "model_finished"}
    with pytest.raises(X.Refusal):              # a finished model is never run again (real preflight)
        X.close_incomplete("qwen3.5:9b")


def test_an_injected_held_out_id_sends_and_journals_nothing(sandbox, monkeypatch):
    real = X.work_list

    def with_held_out(tag, sets, records):
        items = real(tag, sets, records)
        return [("v1", "dev", "h1", sandbox["records"]["h1"], "text h1"), *items]
    monkeypatch.setattr(X, "work_list", with_held_out)
    c = FakeClient()
    with pytest.raises(X.Refusal, match="h1"):
        X.run_model("qwen3.5:9b", c, records=sandbox["records"])
    assert c.calls == [] and X._jsonl(X.JOURNAL) == [] and X._jsonl(X.CACHE_FILE) == []


def _http(code, message):
    return urllib.error.HTTPError("http://x/api/chat", code, "err", {}, io.BytesIO(message.encode()))


def test_refusals_interruptions_and_the_one_resend(sandbox):
    # an interruption (500) on the 4th request: stop, nothing dropped
    c = FakeClient(fail_at=4, error=_http(500, "runner crashed"))
    with pytest.raises(X.Refusal, match="interrupted"):
        X.run_model("qwen3.5:9b", c, records=sandbox["records"])
    assert len(X._jsonl(X.CACHE_FILE)) == 3 and len(X._jsonl(X.JOURNAL)) == 4
    with pytest.raises(X.Refusal, match="resend-unanswered"):        # no silent resend
        X.run_model("qwen3.5:9b", FakeClient(), records=sandbox["records"])
    # the resend, failing again on its first request: that key is now exhausted
    c2 = FakeClient(fail_at=1, error=TimeoutError("timed out"))
    with pytest.raises(X.Refusal):
        X.run_model("qwen3.5:9b", c2, resend_unanswered=True, records=sandbox["records"])
    c3 = FakeClient()
    out = X.run_model("qwen3.5:9b", c3, resend_unanswered=True, records=sandbox["records"])
    assert out["missing"] == 1 and len(c3.calls) == 6
    kinds = [e["kind"] for e in X._jsonl(X.JOURNAL) if e["key"] == "qwen3.5:9b|v1|l1"]
    assert kinds == ["send", "resend"]
    X.close_incomplete("qwen3.5:9b")
    assert X.state()["status"]["qwen3.5:9b"] == "closed_incomplete"


def test_a_think_refusal_on_the_first_request_drops_the_field_not_the_model(sandbox):
    c = FakeClient(fail_at=1, error=_http(400, '"phi4:14b" does not support think'))
    out = X.run_model("phi4:14b", c, records=sandbox["records"])
    assert out["missing"] == 0 and "think" in c.calls[0] and all("think" not in b for b in c.calls[1:])
    assert [e["field"] for e in X._jsonl(X.RUN_LOG) if e["event"] == "field_dropped"] == ["think"]


def test_other_refusals_and_late_errors():
    assert X.classify(_http(400, "invalid value for logprobs"), True)[0] == "drop_field:logprobs"
    assert X.classify(_http(400, "format: invalid schema"), True)[0] == "drop_model"
    assert X.classify(_http(400, "does not support think"), False)[0] == "interruption"   # not the first request
    assert X.classify(_http(404, "not found"), True)[0] == "interruption"              # names no field
    assert X.classify(_http(503, "think"), True)[0] == "interruption"
    assert X.classify(ConnectionResetError(), True)[0] == "interruption"


# ------------------------------------------------------------------ preflight refusals (no network, no Ollama)
@pytest.fixture
def pre_env(tmp_path, monkeypatch):
    for name in ("CACHE_FILE", "JOURNAL", "RUN_LOG", "REGISTRATION"):
        monkeypatch.setattr(X, name, tmp_path / getattr(X, name).name)
    X.REGISTRATION.write_text(json.dumps({"repository": "o/r", "code_commit": "c" * 40,
                                          "code_published_at": "2026-11-11T00:00:00Z", "gist_id": "g",
                                          "gist_url": "https://gist.github.com/g", "gist_created_at": "t"}))
    answers = {"diff": "", "status": "", "rev-parse": "h" * 40}
    monkeypatch.setattr(X, "git", lambda *a: answers[a[0]] if not (a[0] == "diff" and a[2] == "c" * 40 and
                                                                   answers.get("changed")) else answers["changed"])
    return answers


def test_preflight_passes_with_everything_in_order(pre_env):
    out = X.preflight("qwen3.5:9b", FakeClient(), published_head=lambda: "h" * 40)
    assert out["ollama_version"] == "0.40.15" and "h1" not in out["sets"]["dev"]


@pytest.mark.parametrize("what, match", [
    ("version", "Ollama is 0.40.16"), ("digest", "digest"), ("unpublished", "not the commit GitHub shows"),
    ("changed_plan", "differ from the published code commit"), ("dirty", "uncommitted changes outside"),
    ("order", "comes before"), ("unknown", "not a model of the plan"), ("registration", "is missing"),
    ("registered_text", "lines 1-149"),
])
def test_preflight_refuses(pre_env, monkeypatch, tmp_path, what, match):
    client, head, tag = FakeClient(), (lambda: "h" * 40), "qwen3.5:9b"
    if what == "version":
        client = FakeClient(version="0.40.16")
    elif what == "digest":
        client = FakeClient(digests={**dict(X.MODELS), "qwen3.5:9b": "0" * 64, X.REFERENCE[0]: X.REFERENCE[1]})
    elif what == "unpublished":
        head = lambda: "x" * 40  # noqa: E731
    elif what == "changed_plan":
        pre_env["changed"] = "docs/analysis_plan_models.md\n"
    elif what == "dirty":
        pre_env["status"] = " M src/judge/models.py\n"
    elif what == "order":
        tag = "gemma4:12b"
    elif what == "unknown":
        tag = "qwen3.8:27b"
    elif what == "registration":
        X.REGISTRATION.unlink()
    elif what == "registered_text":
        p = tmp_path / "plan.md"
        p.write_bytes(X.PLAN_FILE.read_bytes().replace(b"(planned, not run)", b"(planned)", 1))
        monkeypatch.setattr(X, "PLAN_FILE", p)
    with pytest.raises(X.Refusal, match=match):
        X.preflight(tag, client, published_head=head)


# ------------------------------------------------------------------ verify
def test_verify_catches_a_rerun_and_a_held_out_answer(sandbox, monkeypatch):
    monkeypatch.setattr(X, "item_sets", lambda: (sandbox["sets"], {"h1"}))
    X.REGISTRATION.write_text(json.dumps({"repository": "o/r", "code_commit": "c", "code_published_at": "2000",
                                          "gist_id": "g", "gist_url": "u", "gist_created_at": "t"}))
    X.run_model("qwen3.5:9b", FakeClient(), records=sandbox["records"])
    assert X.verify() == []
    j = X._jsonl(X.JOURNAL)
    X._append(X.JOURNAL, {**j[0], "at": X.now()})               # the same request sent a second time
    X._append(X.JOURNAL, {**j[0], "key": "qwen3.5:9b|v1|h1"})
    problems = X.verify()
    assert any("more often" in p for p in problems) and any("not a DEV" in p for p in problems)


# ------------------------------------------------------------------ the analysis
def test_the_analysis_reproduces_the_plans_reference_figures_and_labels_a_copy_alike(tmp_path, monkeypatch):
    from judge import models_analysis as A
    from judge.data import load_splits

    study = {r["key"]: r for r in X._jsonl(C.CACHE_FILE)}
    sets, _ = X.item_sets()
    # a fake model that answers exactly as qwen3.8:27b did, one malformed answer on v2 DEV
    rows = []
    for v, s in X.CELLS:
        for i in sets[s]:
            r = dict(study[f"{v}|{i}"])
            r["key"] = X.key("qwen3.5:9b", v, i)
            if (v, s) == ("v2", "dev") and i == sets["dev"][0]:
                r["content"] = "not json"
            rows.append(r)
    cache = tmp_path / "responses.jsonl"
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(X, "RUN_LOG", tmp_path / "run_log.jsonl")
    out = A.analyse(cache_path=cache)
    ref = out["models"]["qwen3.8:27b"]["cells"]
    a1 = ref["v1|dev"]["A"]["readings"]["as no money back"]
    assert round(a1["value"], 3) == -0.198 and [round(x, 3) for x in a1["0.95"]] == [-0.359, -0.036]
    assert ref["v1|dev"]["pays"] == 137 and out["record_pays_dev"] == 60 and out["lookup_pays_dev"] == 92
    assert ref["v1|dev"]["A"]["also_at_corrected_level"] is False      # 14 comparisons: zero inside
    new = out["models"]["qwen3.5:9b"]["cells"]
    assert new["v1|dev"]["A"]["label"] == ref["v1|dev"]["A"]["label"] == "below the issuer lookup"
    assert new["v1|dev"]["B"]["label"] == "pays more often than the companies did"
    assert new["v2|dev"]["malformed"] == 1 and new["v1|dev"]["kappa_with_reference"] == 1.0
    assert new["v1|labelled"]["vs_person"]["decided_n"] == 56
    s = out["summary"]
    assert s.startswith("On DEV, judge v1 paid more often than the companies did for 1 of 1 models;")
    assert "it does not show that the results are independent of the model" in s
    assert "gemma4:12b (not run)" in s and "depend on the model" not in s.replace("independent of the model", "")
    assert load_splits()  # data present


def test_incomplete_and_format_rules(tmp_path, monkeypatch):
    from judge import models_analysis as A

    study = {r["key"]: r for r in X._jsonl(C.CACHE_FILE)}
    sets, _ = X.item_sets()
    rows = []
    for v, s in X.CELLS:
        for n, i in enumerate(sets[s]):
            r = dict(study[f"{v}|{i}"], key=X.key("gemma4:12b", v, i))
            if (v, s) == ("v1", "dev") and n < 21:
                r["done_reason"] = "length"                 # 21 malformed of 200: not interpretable
            if (v, s) == ("v2", "labelled") and n == 0:
                continue                                    # one answer missing: incomplete
            rows.append(r)
    cache = tmp_path / "responses.jsonl"
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(X, "RUN_LOG", tmp_path / "run_log.jsonl")
    out = A.analyse(cache_path=cache)
    cells = out["models"]["gemma4:12b"]["cells"]
    assert cells["v1|dev"]["status"] == "not interpretable (format)" and "A" not in cells["v1|dev"]
    assert cells["v2|labelled"]["status"] == "incomplete (99 of 100 answered)"
    assert "gemma4:12b (v1|dev: not interpretable (format); v2|labelled: incomplete (99 of 100 answered))" in out["summary"]
    assert np.isclose(cells["v2|dev"]["kappa_with_reference"], 1.0)
