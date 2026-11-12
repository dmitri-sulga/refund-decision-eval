"""The cloud branch (docs/analysis_plan_models.md, Amendment 3). Every test uses a fake transport or a mock server on
127.0.0.1; nothing here leaves the machine, and no test reads the real key file (each points OPENROUTER_KEY_FILE at a
temporary file with a fake key). The tests pass whether the branch's registration is still empty or already filled."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from judge import cloud as K
from judge import cloud_select as S
from judge import config as C
from judge import freeze, llm
from judge import hosted as H
from judge import models as X
from judge.judgments import parse
from judge.prompts import build_messages

FAKE_KEY = "sk-" + "or-v1-" + "FAKEKEYFORTESTS" * 4
FILLED = {"repository": "o/r", "code_commit": "c" * 40, "code_published_at": "2000-01-01T00:00:00Z",
          "code_published_source": "GitHub", "gist_id": "g", "gist_url": "https://gist.github.com/x/g",
          "gist_created_at": "2000-01-01T00:00:01Z", "gist_file_sha256": {"x": "0" * 64},
          "key_reading": {"limit": 15.0, "limit_reset": None, "free_model_daily_requests": {"limit": 1000}}}
FREE1, PAID1 = K.FREE_QUEUE[0], K.PAID_QUEUE[0]


@pytest.fixture(autouse=True)
def fake_key(tmp_path_factory, monkeypatch):
    p = tmp_path_factory.mktemp("key") / "openrouter_key"
    p.write_text(FAKE_KEY + "\n")
    os.chmod(p, 0o600)
    monkeypatch.setenv(K.KEY_FILE_ENV, str(p))
    return p


# ------------------------------------------------------------------ the request
def test_the_request_carries_the_frozen_messages_and_schema_and_only_the_registered_settings():
    frozen = freeze.load()["request_templates"]
    for tag in K.MODELS:
        ep = K.ENDPOINT[tag]
        for v in X.VERSIONS:
            b = K.build_body(tag, v, freeze._PROBE, "narrative")
            messages, schema = build_messages(v, freeze._PROBE, "narrative")
            assert b["messages"] == messages
            back = llm.request_body(b["messages"], schema)
            assert llm.request_hash(back) == frozen[v]
            if ep["format"] == "json_schema":
                sent = b["response_format"]["json_schema"]["schema"]
                assert {k: x for k, x in sent.items() if k != "additionalProperties"} == schema
                assert sent["additionalProperties"] is False and b["response_format"]["json_schema"]["strict"] is True
            elif ep["format"] == "json_object":
                assert b["response_format"] == {"type": "json_object"}
            else:
                assert "response_format" not in b
            assert b["temperature"] == 0 and b["max_tokens"] == 512 and b["reasoning"] == {"effort": "none"}
            assert b["provider"] == {"order": [ep["tag"]], "allow_fallbacks": False, "require_parameters": True}
            assert ("seed" in b) == ep["seed"] and b.get("seed", 42) == 42 and b["stream"] is False
            assert set(b) <= {"model", "messages", "temperature", "max_tokens", "reasoning", "stream", "provider",
                              "seed", "response_format"}
    dropped = K.build_body(PAID1, "v1", freeze._PROBE, "n", drop=("reasoning", "seed"))
    assert "reasoning" not in dropped and "seed" not in dropped


def test_the_registered_list_is_what_the_rules_select_from_the_committed_snapshot():
    sel = S.resolve(*S.load_snapshot())
    assert [r["id"] for r in sel["free"]] == list(K.FREE_QUEUE[:3])
    assert [r["id"] for r in sel["reference"]] == list(K.REFERENCE_ONLY)
    assert [r["id"] for r in sel["paid"]] == list(K.PAID_QUEUE)
    for r in sel["free"] + sel["reference"] + sel["paid"]:
        ep, reg = r["endpoint"], K.ENDPOINT[r["id"]]
        assert (ep["tag"], ep["provider_name"]) == (reg["tag"], reg["provider"])
        assert (ep["prompt_usd_per_token"], ep["completion_usd_per_token"]) == pytest.approx(
            (reg["prompt"], reg["completion"]), abs=1e-15)
        assert ep["supports"]["seed"] == reg["seed"]
        fmt = "json_schema" if ep["supports"]["structured_outputs"] else "json_object" if \
            ep["supports"]["response_format"] else None
        assert fmt == reg["format"]
    assert all(r["parameters"] > S.MIN_PARAMETERS for r in sel["free"] + sel["paid"])
    assert K.COMPARED == K.FREE_QUEUE[:3] + K.PAID_QUEUE


def test_the_selection_rules_reject_what_the_plan_says_they_reject():
    base = {"id": "dev/model-x", "hugging_face_id": "dev/Model-X", "description": "A model.", "created": 1,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "pricing": {"prompt": "0.0000001", "completion": "0.0000002"}, "reasoning": {"mandatory": False}}
    hf = {"dev/Model-X": {"total": 100 * 10 ** 9}}
    assert S.why_not(base, hf) is None
    assert "mandatory" in S.why_not({**base, "reasoning": {"mandatory": True}}, hf)
    assert "abliterated" in S.why_not({**base, "id": "dev/model-x-abliterated"}, hf)
    assert "uncensor" in S.why_not({**base, "id": "dev/model-uncensored"}, hf)
    assert "derivative" in S.why_not({**base, "description": "Built on [Other](https://x) with tuning."}, hf)
    assert S.why_not({**base, "description": "Built on a hybrid Transformer-Mamba architecture."}, hf) is None
    assert "Anthropic" in S.why_not({**base, "id": "anthropic/x"}, hf)
    assert "open weights" in S.why_not({**base, "hugging_face_id": ""}, hf)
    assert "parameter count" in S.why_not(base, {})
    ok = {"status": 0, "quantization": "fp8", "supported_parameters": ["structured_outputs", "temperature",
                                                                         "max_tokens"],
          "pricing": {"prompt": "0.0000005", "completion": "0.000002"}, "tag": "p/fp8"}
    assert S.paid_endpoint_ok(ok)
    assert not S.paid_endpoint_ok({**ok, "quantization": "fp4"})
    assert not S.paid_endpoint_ok({**ok, "quantization": "unknown"})
    assert not S.paid_endpoint_ok({**ok, "pricing": {"prompt": "0.0000011", "completion": "0.000002"}})
    assert not S.paid_endpoint_ok({**ok, "pricing": {"prompt": "0.0000005", "completion": "0.0000041"}})
    assert not S.paid_endpoint_ok({**ok, "status": -2})


def test_the_estimate_fits_the_caps_with_the_worst_case():
    from judge.data import load_records
    sets, _ = X.item_sets()
    records = load_records()
    rows = [K.estimate(t, sets, records) for t in K.MODELS]
    assert all(r["requests"] == 600 for r in rows)
    assert all(r["worst_case_usd"] == 0 for r in rows[:4])
    assert all(r["worst_case_usd"] < K.SUB_CAP_USD for r in rows)
    assert sum(r["worst_case_usd"] for r in rows) < K.CAP_USD
    # Correction, M1: with room kept for one overrun answer, the registered worst case still fits the branch cap
    assert sum(r["worst_case_usd"] for r in rows) + max(K.overrun_headroom(t) for t in K.MODELS) < K.CAP_USD


# ------------------------------------------------------------------ registration
def test_the_registration_refuses_while_empty_and_is_read_once_filled(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "REGISTRATION", tmp_path / "registration.json")
    with pytest.raises(X.Refusal, match="missing"):
        K.load_registration()
    K.REGISTRATION.write_text(json.dumps({**FILLED, "gist_created_at": None}))
    with pytest.raises(X.Refusal, match="gist_created_at"):
        K.load_registration()
    for f in ("gist_file_sha256", "code_published_source", "key_reading"):        # Correction, m8
        K.REGISTRATION.write_text(json.dumps({**FILLED, f: None}))
        with pytest.raises(X.Refusal, match=f):
            K.load_registration()
    K.REGISTRATION.write_text(json.dumps(FILLED))
    assert K.load_registration()["code_commit"] == "c" * 40


def test_the_committed_registration_is_either_waiting_or_complete():
    reg = json.loads((C.RESULTS / "models" / "cloud" / "registration.json").read_text())
    fields = ("code_commit", "code_published_at", "code_published_source", "gist_id", "gist_url", "gist_created_at",
              "gist_file_sha256", "key_reading")
    assert all(reg.get(f) is None for f in fields) or all(reg.get(f) for f in fields)
    assert reg["models"] == list(K.MODELS) and reg["compared"] == list(K.COMPARED)
    assert reg["cap_usd"] == K.CAP_USD and reg["sub_cap_usd"] == K.SUB_CAP_USD
    assert reg["snapshot_sha256"] == S.snapshot_sha256()


# ------------------------------------------------------------------ a sandbox with a fake transport
def _resp(content, tag, usage=None, model=None, provider=None, finish="stop", reasoning=None):
    u = usage or {"prompt_tokens": 700, "completion_tokens": 40}
    if "cost" not in u:
        u = {**u, "cost": K.computed_cost(tag, u)}
    msg = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning"] = reasoning
    return {"id": "gen-1", "model": model or tag, "provider": provider or K.ENDPOINT[tag]["provider"],
            "choices": [{"message": msg, "finish_reason": finish, "native_finish_reason": finish}], "usage": u}


class FakeTransport:
    """The API as documented on 2026-11-12: GET /api/v1/key gives the key's limit, limit_remaining, limit_reset, usage
    and the free-model daily count; /api/v1/credits needs a management key and is never called."""

    def __init__(self, fail_at=(), error=None, usage=None, model=None, provider="registered", remaining=1000,
                 limit=15.0, limit_remaining=15.0, limit_reset=None, reasoning=None, content=None, omit=()):
        self.calls, self.error, self.usage, self.model, self.provider = [], error, usage, model, provider
        self.fail_at = {fail_at} if isinstance(fail_at, int) else set(fail_at)
        self.remaining, self.reasoning, self.gets, self.content = remaining, reasoning, [], content
        self.limit, self.limit_remaining, self.limit_reset, self.omit = limit, limit_remaining, limit_reset, omit

    def send(self, body):
        self.calls.append(body)
        if len(self.calls) in self.fail_at:
            raise self.error
        content = self.content or json.dumps({"monetary_relief": len(self.calls) % 3 == 0, "p_relief": 0.5,
                                              "reason": "x"})
        r = _resp(content, body["model"], self.usage, self.model,
                  None if self.provider == "registered" else self.provider, reasoning=self.reasoning)
        if self.provider is None:
            r.pop("provider")
        return r, {"generation-id": f"gen-{len(self.calls)}", "date": "Thu, 12 Nov 2026 12:00:00 GMT"}

    def get(self, url):
        self.gets.append(url)
        assert url == K.KEY_URL
        data = {"label": "sk-or-v1-abc...xyz", "limit": self.limit, "limit_remaining": self.limit_remaining,
                "limit_reset": self.limit_reset, "usage": 0.0, "usage_daily": 0.0,
                "include_byok_in_limit": False, "is_free_tier": False,
                "free_model_daily_requests": {"used": 1000 - self.remaining, "limit": 1000,
                                              "remaining": self.remaining}}
        return {"data": {f: v for f, v in data.items() if f not in self.omit}}


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(K, "DIR", tmp_path)
    for name in ("CACHE_FILE", "JOURNAL", "RUN_LOG", "SPEND", "REGISTRATION", "LOCK", "QUARANTINE"):
        monkeypatch.setattr(K, name, tmp_path / getattr(K, name).name)
    sets = {"dev": ["d1", "d2", "d3"], "labelled": ["l1", "l2"]}
    records = {i: {"product": "P", "sub_product": "S", "issue": "I", "sub_issue": "J", "narrative": f"text {i}",
                   "label": 0} for i in ["d1", "d2", "d3", "l1", "l2", "h1"]}
    monkeypatch.setattr(K, "preflight", lambda tag, published_head=None, getter=None: {
        "sets": sets, "test": {"h1"}, "registration": FILLED})
    monkeypatch.setattr(X, "item_sets", lambda: (sets, {"h1"}))
    K.REGISTRATION.write_text(json.dumps(FILLED))
    return {"tmp": tmp_path, "records": records, "sets": sets}


def run(tag, sandbox, transport, **kw):
    return K.run_model(tag, transport, records=sandbox["records"], sleep=lambda s: None, **kw)


def events():
    return [e["event"] for e in X._jsonl(K.RUN_LOG)]


def test_a_full_paid_model_journals_records_spend_reads_credits_and_verifies(sandbox):
    t = FakeTransport()
    out = run(PAID1, sandbox, t)
    assert out["missing"] == 0 and out["new_answers"] == 10 and len(t.calls) == 10
    journal, cache = X._jsonl(K.JOURNAL), X._jsonl(K.CACHE_FILE)
    assert [e["key"] for e in journal] == [r["key"] for r in cache]
    spend = X._jsonl(K.SPEND)
    answers = [e for e in spend if e["event"] == "answer"]
    per = K.ENDPOINT[PAID1]["prompt"] * 700 + K.ENDPOINT[PAID1]["completion"] * 40
    assert len(answers) == 10 and np.isclose(answers[-1]["running_total_usd"], 10 * per)
    assert events()[0] == "account_before" and events()[-1] == "account_after"
    first = X._jsonl(K.RUN_LOG)[0]
    assert first["limit_remaining"] == 15.0 and first["limit"] == 15.0 and set(t.gets) == {K.KEY_URL}
    assert "label" not in first                       # the key's label shows part of the key: never kept
    assert K.verify(records=sandbox["records"]) == [] and not K.LOCK.exists()
    with open(K.SPEND, "a") as f:
        f.write(json.dumps({**answers[0], "cost_usd": 1.0}) + "\n")
    assert any("running total" in p for p in K.verify(records=sandbox["records"]))


def test_a_cloud_answer_parses_exactly_as_a_local_answer_with_the_same_text():
    from judge import models_analysis as A
    for content in ('{"monetary_relief": true, "p_relief": 0.9, "reason": "x"}', '```json\n{"monetary_relief": true}'
                    '\n```', "", "not json"):
        r = K.answer_record(PAID1, "v1", "d1", {}, "h", _resp(content, PAID1), {}, "t", 1.0, [])
        assert parse(r) == {**parse({"content": content}), "wall_s": 1.0, "eval_count": 40, "prompt_eval_count": 700}
    cut = K.answer_record(PAID1, "v1", "d1", {}, "h", _resp('{"monetary_relief": true}', PAID1, finish="length"),
                          {}, "t", 1.0, [])
    assert A.malformed(cut, parse(cut))                  # a truncated answer is malformed, scored no money back


def test_an_injected_held_out_id_sends_journals_and_spends_nothing(sandbox, monkeypatch):
    real = K.work_list
    monkeypatch.setattr(K, "work_list", lambda tag, sets, records: [
        ("v1", "dev", "h1", sandbox["records"]["h1"], "text h1"), *real(tag, sets, records)])
    t = FakeTransport()
    with pytest.raises(X.Refusal, match="h1"):
        run(PAID1, sandbox, t)
    assert t.calls == [] and X._jsonl(K.JOURNAL) == [] and X._jsonl(K.CACHE_FILE) == [] and K.spent() == 0


def test_the_model_sub_cap_stops_before_a_request_whose_worst_case_would_exceed_it(sandbox, monkeypatch):
    body = K.build_body(PAID1, "v1", sandbox["records"]["d1"], "text d1")
    per = K.computed_cost(PAID1, {"prompt_tokens": 700, "completion_tokens": 40})
    reserve = K.worst_case(PAID1, body)
    monkeypatch.setattr(K, "SUB_CAP_USD", 3 * per + reserve * 0.999)
    monkeypatch.setattr(K, "projected", lambda tag, b: 0.0)
    t = FakeTransport()
    with pytest.raises(X.Refusal, match="budget"):
        run(PAID1, sandbox, t)
    assert len(t.calls) == 3 and np.isclose(K.spent(), 3 * per)
    stop = [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "budget_stop"][0]
    assert stop["scope"] == "model" and "closed_incomplete" in events()


def test_the_branch_cap_stops_and_no_paid_model_starts_after_it(sandbox, monkeypatch):
    monkeypatch.setattr(K, "CAP_USD", 0.0001)
    t = FakeTransport()
    with pytest.raises(X.Refusal, match="cap"):
        run(PAID1, sandbox, t)
    assert t.calls == [] and X._jsonl(K.JOURNAL) == []
    assert [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "budget_stop"][0]["scope"] == "branch"


def test_a_paid_model_needs_enough_of_the_keys_limit_before_it_starts(sandbox):
    t = FakeTransport(limit_remaining=0.5)
    with pytest.raises(X.Refusal, match="of its limit left"):
        run(PAID1, sandbox, t)
    assert t.calls == []


def test_a_paid_model_refuses_a_key_without_a_limit_and_a_free_model_does_not(sandbox):
    """Correction, B1, and Correction 2, C15: the paid arm needs a key limited in OpenRouter to exactly 15 USD, never
    resetting, and a reading that reports every field it relies on (the review's m-a: a missing limit_reset is not
    "never resets"); a free run on another key is recorded as a deviation (m-j)."""
    for kw, why in (({"limit": None, "limit_remaining": None}, "no credit limit"),
                    ({"limit": 50.0, "limit_remaining": 50.0}, "not exactly the branch cap"),
                    ({"limit": 10.0, "limit_remaining": 10.0}, "not exactly the branch cap"),
                    ({"limit_reset": "monthly"}, "resets"),
                    ({"omit": ("limit_reset",)}, "did not report limit_reset"),
                    ({"omit": ("limit", "limit_remaining")}, "did not report limit, limit_remaining")):
        t = FakeTransport(**kw)
        with pytest.raises(X.Refusal, match=why):
            run(PAID1, sandbox, t)
        assert t.calls == [] and X._jsonl(K.JOURNAL) == []
    before = [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "account_before"]
    assert before[-1]["fields_missing"] == ["limit", "limit_remaining", "budgets", "effective_budget"]
    assert "budgets" in before[-1] and "effective_budget" in before[-1]           # m-i: recorded (null if absent)
    assert not [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "deviation"]   # the registered key: no deviation
    run(FREE1, sandbox, FakeTransport(remaining=K.FREE_DAILY_MARGIN + 1))
    assert not [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "deviation"]
    t = FakeTransport(limit=None, limit_remaining=None)
    assert run(FREE1, sandbox, t)["missing"] == 0 and len(t.calls) == 9
    dev = [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "deviation"]
    assert len(dev) == 1 and dev[0]["kind"] == "key_differs" and dev[0]["fields"] == ["limit"]
    assert K.verify(records=sandbox["records"]) == []


def test_a_failed_key_reading_stops_a_paid_model(sandbox):
    class Failing(FakeTransport):
        def get(self, url):
            raise K.CloudError(401, "No auth")
    t = Failing()
    with pytest.raises(X.Refusal, match="GET /api/v1/key failed"):
        run(PAID1, sandbox, t)
    assert t.calls == []


def test_a_free_model_costs_nothing_and_stops_at_the_days_allowance(sandbox):
    t = FakeTransport(remaining=K.FREE_DAILY_MARGIN + 4)
    out = run(FREE1, sandbox, t)
    assert out.get("paused") and len(t.calls) == 4 and K.spent() == 0
    assert "daily_pause" in events()
    t2 = FakeTransport(remaining=1000)
    out = run(FREE1, sandbox, t2)
    assert out["missing"] == 0 and len(t2.calls) == 6 and K.verify(records=sandbox["records"]) == []


def test_a_free_answer_that_reports_a_cost_is_charged(sandbox):
    t = FakeTransport(usage={"prompt_tokens": 700, "completion_tokens": 40, "cost": 0.01})
    run(FREE1, sandbox, t)
    assert np.isclose(K.spend_state()["computed"], 0.1)


def test_the_daily_limit_429_stops_without_counting_or_closing(sandbox):
    err = K.CloudError(429, "Rate limit exceeded: free-models-per-day", remaining="0")
    t = FakeTransport(fail_at=3, error=err)
    with pytest.raises(X.Refusal, match="daily request limit"):
        run(FREE1, sandbox, t)
    assert K.error_count(X._jsonl(K.JOURNAL), FREE1) == 0 and not K.state()["status"]
    assert [e["kind"] for e in X._jsonl(K.JOURNAL)][-1] == "quota"


def test_rate_limits_and_server_errors_are_waited_out_not_charged_and_do_not_use_the_resend(sandbox):
    t = FakeTransport(fail_at=(2, 3), error=K.CloudError(503, "No provider available"))
    out = run(PAID1, sandbox, t)
    assert out["missing"] == 0 and len(t.calls) == 12
    kinds = [e["kind"] for e in X._jsonl(K.JOURNAL)]
    assert kinds.count("transient") == 2 and kinds.count("retry") == 2 and "resend" not in kinds
    assert not [e for e in X._jsonl(K.SPEND) if e["event"] == "charged"]
    assert K.verify(records=sandbox["records"]) == []


def test_an_error_inside_a_200_is_classified_by_its_code():
    assert K.classify(K.CloudError(502, "upstream", in_200=True), PAID1)[0] == "transient"
    assert K.classify(K.CloudError(400, "bad", in_200=True), PAID1)[0] == "rejected"
    assert K.classify(K.CloudError(402, "Insufficient credits"), PAID1)[0] == "account"
    assert K.classify(K.CloudError(401, "No auth"), PAID1)[0] == "account"
    assert K.classify(K.CloudError(404, "No endpoints found"), PAID1)[0] == "rejected"
    assert K.classify(K.CloudError(404, "No endpoints found matching your data policy (Free model publication)"),
                      FREE1)[0] == "account"
    assert K.classify(K.CloudError(429, "slow down", remaining="3"), FREE1)[0] == "transient"
    assert K.classify(K.CloudError(429, "slow down", remaining="0"), PAID1)[0] == "transient"
    assert K.classify(K.CloudError(403, "flagged", moderation={"reasons": ["x"]}), PAID1)[0] == "moderated"
    assert K.classify(K.CloudError(None, "timeout", maybe_billed=True), PAID1)[0] == "lost"
    assert K.classify(TimeoutError("t"), PAID1)[0] == "lost"


def test_a_moderation_flag_is_a_malformed_answer_and_the_run_goes_on(sandbox):
    from judge import models_analysis as A
    err = K.CloudError(403, "Input was flagged", moderation={"reasons": ["violence"], "provider_name": "X"})
    t = FakeTransport(fail_at=2, error=err)
    out = run(PAID1, sandbox, t)
    assert out["missing"] == 0 and len(t.calls) == 10
    flagged = [r for r in X._jsonl(K.CACHE_FILE) if r["moderated"]]
    assert len(flagged) == 1 and A.malformed(flagged[0], parse(flagged[0])) and parse(flagged[0])["pred"] == 0
    assert K.verify(records=sandbox["records"]) == []


def test_a_rejection_closes_a_model_only_when_repeated_24_hours_apart(sandbox):
    def rejecting(date):
        return K.CloudError(404, "No endpoints found for nvidia/x", etype=None, date=date)
    t = FakeTransport(fail_at=1, error=rejecting("Thu, 12 Nov 2026 12:00:00 GMT"))
    with pytest.raises(X.Refusal, match="rejected"):
        run(FREE1, sandbox, t)
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(FREE1)
    t = FakeTransport(fail_at=1, error=rejecting("Thu, 12 Nov 2026 20:00:00 GMT"))
    with pytest.raises(X.Refusal):
        run(FREE1, sandbox, t)
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(FREE1)
    t = FakeTransport(fail_at=1, error=rejecting("Fri, 13 Nov 2026 12:00:01 GMT"))
    with pytest.raises(X.Refusal):
        run(FREE1, sandbox, t)
    assert K.close_incomplete(FREE1).startswith("not run: the request was rejected")
    with pytest.raises(X.Refusal, match="closed"):
        run(FREE1, sandbox, FakeTransport())


def test_an_account_condition_never_closes_a_model(sandbox):
    for date in ("Thu, 12 Nov 2026 12:00:00 GMT", "Sat, 14 Nov 2026 12:00:00 GMT"):
        t = FakeTransport(fail_at=1, error=K.CloudError(402, "Insufficient credits", date=date))
        with pytest.raises(X.Refusal, match="account"):
            run(PAID1, sandbox, t)
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(PAID1)


def test_a_first_request_refusal_naming_reasoning_drops_that_field_for_this_model_only(sandbox):
    t = FakeTransport(fail_at=1, error=K.CloudError(400, "Unsupported parameter: reasoning"))
    out = run(PAID1, sandbox, t)
    assert out["missing"] == 0 and "reasoning" in t.calls[0] and all("reasoning" not in b for b in t.calls[1:])
    assert [e["field"] for e in X._jsonl(K.RUN_LOG) if e["event"] == "field_dropped"] == ["reasoning"]
    assert all(r["fields_dropped"] == ["reasoning"] for r in X._jsonl(K.CACHE_FILE))
    assert K.verify(records=sandbox["records"]) == []


def test_a_later_refusal_naming_a_field_drops_nothing(sandbox):
    t = FakeTransport(fail_at=3, error=K.CloudError(400, "Unsupported parameter: seed"))
    with pytest.raises(X.Refusal, match="rejected"):
        run(PAID1, sandbox, t)
    assert "field_dropped" not in events() and all("seed" in b for b in t.calls)


def test_an_answer_from_another_provider_is_quarantined_and_a_second_one_closes_the_model(sandbox):
    """Correction, M2: charged, kept in quarantine, not cached; the run stops; only a second one closes the model."""
    t = FakeTransport(provider="SomeoneElse")
    with pytest.raises(X.Refusal, match="quarantine"):
        run(PAID1, sandbox, t)
    assert X._jsonl(K.CACHE_FILE) == [] and not K.state()["status"] and len(X._jsonl(K.QUARANTINE)) == 1
    assert [e for e in X._jsonl(K.SPEND) if e["event"] == "charged"]
    out = run(PAID1, sandbox, FakeTransport(fail_at=4, error=K.CloudError(503, "x")))   # resumes; same item sent again
    assert out["missing"] == 0 and K.verify(records=sandbox["records"]) == []
    kinds = [e["kind"] for e in X._jsonl(K.JOURNAL)]
    assert kinds[:3] == ["send", "mismatch", "send"] and "resend" not in kinds


def test_a_second_answer_from_another_model_closes_the_model(sandbox):
    for _ in range(2):
        with pytest.raises(X.Refusal):
            run(PAID1, sandbox, FakeTransport(model="deepseek/deepseek-v4-flash"))
    assert K.state()["status"][PAID1].startswith("not run: served by deepseek/deepseek-v4-flash")
    assert len(X._jsonl(K.QUARANTINE)) == 2 and X._jsonl(K.CACHE_FILE) == []


def test_the_provider_check_ignores_case_and_punctuation_and_accepts_a_missing_field(sandbox):
    """Correction, M2 (the reviewer's PROBE3 and PROBE4): 'NVIDIA' for 'Nvidia' and 'atlas-cloud' for 'AtlasCloud' are
    the registered provider; a missing provider field is accepted by an explicit rule and counted."""
    assert K.provider_matches(FREE1, "NVIDIA") and K.provider_matches(FREE1, " nvidia ")
    assert K.provider_matches("dots-studio/dots-3-note-preview:free", "atlas-cloud")
    assert K.provider_matches("google/gemma-4-31b-it:free", "google ai studio")
    assert K.provider_matches(FREE1, None) and not K.provider_matches(FREE1, "Novita")
    out = run(FREE1, sandbox, FakeTransport(provider="NVIDIA"))
    assert out["missing"] == 0 and K.verify(records=sandbox["records"]) == []


def test_an_interruption_is_charged_and_resent_once(sandbox):
    t = FakeTransport(fail_at=2, error=K.CloudError(None, "no answer: timeout", maybe_billed=True))
    with pytest.raises(X.Refusal, match="resend-unanswered"):
        run(PAID1, sandbox, t)
    with pytest.raises(X.Refusal, match="resend-unanswered"):
        run(PAID1, sandbox, FakeTransport())
    out = run(PAID1, sandbox, FakeTransport(), resend_unanswered=True)
    assert out["missing"] == 0
    kinds = [e["kind"] for e in X._jsonl(K.JOURNAL)]
    assert kinds.count("resend") == 1
    charged = [e for e in X._jsonl(K.SPEND) if e["event"] == "charged"]
    assert len(charged) == 1 and charged[0]["note"].startswith("worst case")
    assert K.verify(records=sandbox["records"]) == []


def test_a_crash_leaves_its_reservation_charged_until_the_next_run(sandbox):
    body = K.build_body(PAID1, "v1", sandbox["records"]["d1"], "text d1")
    X._append(K.SPEND, {"at": X.now(), "event": "reserved", "id": "r1", "model": PAID1, "key": f"{PAID1}|v1|d1",
                        "cost_usd": K.worst_case(PAID1, body)})
    assert np.isclose(K.spent(), K.worst_case(PAID1, body))
    K.reconcile()
    assert K.spend_state()["open"] == [] and np.isclose(K.spend_state()["computed"], K.worst_case(PAID1, body))


def test_the_queue_order_and_the_lock(sandbox, monkeypatch):
    K.acquire_lock()
    with pytest.raises(X.Refusal, match="run.lock"):
        run(PAID1, sandbox, FakeTransport())
    K.release_lock()
    assert K.break_stale_lock() == "no run.lock"


def test_the_key_is_sent_only_in_its_header_and_never_written(tmp_path, monkeypatch, capsys):
    seen = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["auth"] = self.headers.get("Authorization")
            seen["body"] = self.rfile.read(int(self.headers["Content-Length"])).decode()
            msg = json.dumps({"error": {"code": 400, "message": f"bad key {FAKE_KEY} echoed"}}).encode()
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(msg)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(K, "API_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    try:
        with pytest.raises(K.CloudError) as e:
            K.OpenRouterTransport().send({"model": "x"})
    finally:
        srv.shutdown()
    assert seen["auth"] == "Bearer " + FAKE_KEY and FAKE_KEY not in seen["body"]
    assert FAKE_KEY not in str(e.value) and "[redacted]" in str(e.value)
    assert FAKE_KEY not in capsys.readouterr().out


def test_the_key_file_must_be_private_and_nothing_is_sent_without_it(fake_key, monkeypatch):
    os.chmod(fake_key, 0o644)
    assert "group or others" in K.key_file_ok()
    with pytest.raises(K.CloudError, match="nothing was sent"):
        K.OpenRouterTransport().send({"model": "x"})
    monkeypatch.setenv(K.KEY_FILE_ENV, str(fake_key) + ".missing")
    assert "does not exist" in K.key_file_ok()
    assert K.classify(K.CloudError(None, "x"), PAID1)[0] == "account"


def test_the_makefile_and_the_code_name_no_key():
    text = (C.ROOT / "Makefile").read_text() + (C.ROOT / "src" / "judge" / "cloud.py").read_text()
    assert not K.KEY_PATTERN.search(text) and "openrouter_key" not in (C.ROOT / "Makefile").read_text()


def test_the_real_cloud_directory_holds_no_key_and_verifies():
    assert K.verify() == []


def _getter(models, eps, dates=None):
    dates = iter(dates) if dates else None

    def g(url):
        d = next(dates) if dates else "Thu, 12 Nov 2026 12:00:00 GMT"
        if url == K.MODELS_URL:
            return {"data": models}, d
        mid = url.split("/models/", 1)[1].rsplit("/endpoints", 1)[0]
        return {"data": {"endpoints": eps.get(mid, [])}}, d
    return g


def _hourly(start_h, n, day=12):
    from datetime import datetime, timedelta, timezone
    t0 = datetime(2026, 11, day, tzinfo=timezone.utc) + timedelta(hours=start_h)
    return [(t0 + timedelta(hours=k // 2)).strftime("%a, %d %b %Y %H:%M:%S GMT") for k in range(2 * n)]


def test_the_endpoint_status_is_recorded_with_every_flip_but_is_not_a_gate(sandbox):
    """Correction, B2: crusoe/bf16 went from status 0 to -2 within an hour of the snapshot."""
    models, _, eps = S.load_snapshot()
    tag = "moonshotai/kimi-k2.6"
    flip = {**eps, tag: [{**e, "status": -2} if e["tag"] == "crusoe/bf16" else e for e in eps[tag]]}
    why, _, status = K.listing_check(tag, _getter(models, flip))
    assert why is None and status == -2
    for e in (flip, eps, eps):
        K.record_listing(tag, *K.listing_check(tag, _getter(models, e)), "test")
    flips = [(e["from"], e["to"]) for e in X._jsonl(K.RUN_LOG) if e["event"] == "endpoint_status_changed"]
    assert flips == [(0, -2), (-2, 0)]
    assert all(e.get("server_date") and e.get("at") for e in X._jsonl(K.RUN_LOG))


def test_an_unavailable_model_closes_only_after_24_hours_of_hourly_failures_with_no_success_between(sandbox):
    """Correction, B2 (the reviewer's PROBE5): two failing checks 24 hours apart with a run that answered between
    them, or with a gap of more than two hours between checks, do not close the model; 25 hourly failures do."""
    models, _, eps = S.load_snapshot()
    gone = {**eps, FREE1: []}
    real_pre = K.preflight

    def stop_before(dates):
        g = _getter(models, gone, dates)
        why, date, status = K.listing_check(FREE1, g)
        K.record_listing(FREE1, why, date, status, "run")
        X._append(K.RUN_LOG, {"at": X.now(), "event": "run_stopped", "model": FREE1, "verdict": "unavailable",
                              "reason": why})
    stop_before(_hourly(0, 1))                                       # 12.11 00:00
    run(FREE1, sandbox, FakeTransport(remaining=K.FREE_DAILY_MARGIN + 2))   # a run in between answers twice
    stop_before(_hourly(24, 1))                                      # 13.11 00:00, 24 hours later
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(FREE1)
    stop_before(_hourly(30, 1))                                      # six hours later: a gap of over two hours
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(FREE1)
    msg = K.watch_listing(FREE1, _getter(models, gone, _hourly(31, 30)), sleep=lambda s: None)
    assert "may close it" in msg
    assert K.close_incomplete(FREE1).startswith("incomplete (2 of 10 answered): the model or its pinned endpoint")
    assert real_pre is K.preflight


def test_a_passing_check_ends_a_failing_streak(sandbox):
    models, _, eps = S.load_snapshot()
    gone = {**eps, FREE1: []}
    msg = K.watch_listing(FREE1, _getter(models, gone, _hourly(0, 3)), sleep=lambda s: None, max_checks=3)
    assert "still failing" in msg
    assert "passes" in K.watch_listing(FREE1, _getter(models, eps, _hourly(3, 1)), sleep=lambda s: None)
    K.watch_listing(FREE1, _getter(models, gone, _hourly(4, 24)), sleep=lambda s: None, max_checks=23)
    assert not K.streak_closes(K.unavailable_streak(FREE1, X._jsonl(K.RUN_LOG)))   # 22 hours since the pass


def test_the_listing_check_refuses_a_missing_model_a_changed_endpoint_or_a_higher_price():
    snap_models, _, snap_eps = S.load_snapshot()

    def getter(models=snap_models, eps=snap_eps):
        def g(url):
            if url == K.MODELS_URL:
                return {"data": models}, "Thu, 12 Nov 2026 12:00:00 GMT"
            mid = url.split("/models/", 1)[1].rsplit("/endpoints", 1)[0]
            return {"data": {"endpoints": eps.get(mid, [])}}, "Thu, 12 Nov 2026 12:00:00 GMT"
        return g
    for tag in K.MODELS:
        assert K.listing_check(tag, getter())[0] is None
    assert "no longer listed" in K.listing_check(PAID1, getter(models=[m for m in snap_models
                                                                       if m["id"] != PAID1]))[0]
    dearer = {**snap_eps, PAID1: [{**e, "pricing": {**e["pricing"], "prompt": "0.00001"}} for e in snap_eps[PAID1]]}
    assert "costs more" in K.listing_check(PAID1, getter(eps=dearer))[0]
    gone = {**snap_eps, FREE1: []}
    assert "endpoint" in K.listing_check(FREE1, getter(eps=gone))[0]


# ------------------------------------------------------------------ the analysis
def test_the_analysis_counts_twelve_and_six_comparisons_and_moves_a_reasoning_model(tmp_path, monkeypatch):
    from judge import cloud_analysis as CA
    from judge.data import load_records
    sets, _ = X.item_sets()
    records = load_records()
    cache, log = tmp_path / "responses.jsonl", tmp_path / "run_log.jsonl"
    with open(cache, "w") as f:
        for t in K.MODELS:
            for v, s in X.CELLS:
                for i in sets[s]:
                    r = K.answer_record(t, v, i, {}, "h", _resp(json.dumps({"monetary_relief": records[i]["label"] == 1,
                                                                             "p_relief": 0.5, "reason": "r"}), t,
                                                                 reasoning="thinking..." if t == PAID1 else None),
                                        {}, "t", 1.0, [])
                    f.write(json.dumps(r) + "\n")
    with open(log, "w") as f:
        for t in K.MODELS:
            f.write(json.dumps({"at": "x", "event": "model_finished", "model": t, "answers": 600}) + "\n")
    out = CA.analyse(cache_path=cache, run_log=log, spend_path=tmp_path / "none.jsonl")
    assert out["comparisons"] == {"A": 12, "B": 6}
    # Correction, M3: the registered outputs 1-4 for every cell, by the functions extension_late.py uses
    from judge import extension_late as EL
    from judge.data import load_splits
    from judge.label import labels_path, load_labels
    tf = EL.tfidf_dev_decisions(records, load_splits())
    person = load_labels(labels_path("owner"))
    for t in K.MODELS:
        for v, s in X.CELLS:
            ro = out["models"][t]["cells"][f"{v}|{s}"]["registered_outputs"]
            y = [records[i]["label"] for i in sets[s]]
            want = EL.cell_figures(y, y, s, tf if s == "dev" else None, person, sets[s])
            assert json.dumps(ro, sort_keys=True, default=float) == json.dumps(want, sort_keys=True, default=float)
            assert "accuracy_vs_record" in ro and "payout_minus_record" in ro and ("minus_tfidf" in ro) == (s == "dev")
            assert ("accuracy_vs_person_decided" in ro) == (s == "labelled")
    assert "also at the corrected level of 99.583%" in out["summary"]      # Correction, M6
    assert not out["models"][PAID1]["compared"] and out["models"][PAID1]["row_note"] == CA.MOVED_ROW
    assert out["models"][K.REFERENCE_ONLY[0]]["row_note"] == CA.SERVING_ROW
    sc = out["models"][K.REFERENCE_ONLY[0]]["cells"]["v1|dev"]["serving_contrast"]
    assert sc["local_model"] == "gemma4:31b" and sc["n"] == 200
    a = out["models"][FREE1]["cells"]["v1|dev"]["A"]
    assert np.isclose(a["corrected_level"], 1 - 0.05 / 12)
    assert np.isclose(out["models"][FREE1]["cells"]["v1|dev"]["B"]["corrected_level"], 1 - 0.05 / 6)
    assert "for 0 of 5 models" in out["summary"] and "Reference rows" in out["summary"]
    assert "not ranked" in out["summary"]


def test_the_served_model_must_be_the_registered_one():
    assert K.model_matches(FREE1, FREE1) and K.model_matches(FREE1, FREE1.removesuffix(":free"))
    assert K.model_matches(PAID1, PAID1 + "-20261013") and K.model_matches("moonshotai/kimi-k2.6", "moonshotai/kimi-k2.6")
    assert not K.model_matches(PAID1, "deepseek/deepseek-v4-flash") and not K.model_matches(PAID1, None)
    assert not K.model_matches("moonshotai/kimi-k2.6", "moonshotai/kimi-k2.6-turbo")


# ------------------------------------------------------------------ corrections after the independent review
def test_an_output_overrun_on_a_paid_model_is_journaled_closes_it_and_check_still_passes(sandbox):
    """Correction, M1 (the reviewer's PROBE1): a provider that bills output beyond max_tokens."""
    t = FakeTransport(usage={"prompt_tokens": 700, "completion_tokens": 2000})
    with pytest.raises(X.Refusal, match="output above max_tokens"):
        run(PAID1, sandbox, t)
    assert len(t.calls) == 1 and len(X._jsonl(K.CACHE_FILE)) == 1             # kept as an answer, scored as it is
    ov = [e for e in X._jsonl(K.JOURNAL) if e["kind"] == "overrun"]
    assert len(ov) == 1 and ov[0]["completion_tokens"] == 2000 and ov[0]["max_tokens"] == 512
    assert K.state()["status"][PAID1].startswith("incomplete (1 of 10 answered)")
    assert [e for e in X._jsonl(K.SPEND) if e["event"] == "answer"][0]["over_reservation"] is True
    assert K.verify(records=sandbox["records"]) == []
    lines = [ln for ln in K.JOURNAL.read_text().splitlines() if '"overrun"' not in ln]   # an overrun not journaled
    K.JOURNAL.write_text("\n".join(lines) + "\n")
    problems = K.verify(records=sandbox["records"])
    assert any("max_tokens and no overrun" in p for p in problems)
    assert any("cost more than its reservation" in p for p in problems)


def test_a_cost_above_the_reservation_closes_a_paid_model(sandbox):
    t = FakeTransport(usage={"prompt_tokens": 700, "completion_tokens": 40, "cost": 1.0})
    with pytest.raises(X.Refusal, match="cost above the reservation"):
        run(PAID1, sandbox, t)
    assert K.verify(records=sandbox["records"]) == []


def test_an_output_overrun_on_a_free_model_is_journaled_and_the_run_goes_on(sandbox):
    t = FakeTransport(usage={"prompt_tokens": 700, "completion_tokens": 900})
    assert run(FREE1, sandbox, t)["missing"] == 0
    assert sum(e["kind"] == "overrun" for e in X._jsonl(K.JOURNAL)) == 10
    assert K.verify(records=sandbox["records"]) == []


def test_the_branch_cap_keeps_room_for_one_overrun_answer(sandbox, monkeypatch):
    body = K.build_body(PAID1, "v1", sandbox["records"]["d1"], "text d1")
    room = K.overrun_headroom(PAID1)
    assert np.isclose(room, 2.97e-6 * (393216 - 512))
    monkeypatch.setattr(K, "CAP_USD", K.worst_case(PAID1, body) + room * 0.999)
    monkeypatch.setattr(K, "projected", lambda tag, b: 0.0)
    t = FakeTransport(limit=K.CAP_USD, limit_remaining=K.CAP_USD)
    with pytest.raises(X.Refusal, match="budget"):
        run(PAID1, sandbox, t)
    assert t.calls == [] and [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "budget_stop"][0]["scope"] == "branch"


def test_reasoning_written_into_the_text_is_reported_reasoning(sandbox):
    """Correction, M4 (the reviewer's PROBE2)."""
    from judge import cloud_analysis as CA
    content = '<think>the consumer asks for a refund</think>{"monetary_relief": true, "p_relief": 0.9, "reason": "x"}'
    r = K.answer_record(FREE1, "v1", "d1", {}, "h", _resp(content, FREE1), {}, "t", 1.0, [])
    assert r["reasoning_in_content_chars"] > 0 and r["reasoning_chars"] > 0 and r["reasoning_tokens"] == 0
    assert not CA.reasoning_off(FREE1, [r])
    open_only = K.answer_record(FREE1, "v1", "d1", {}, "h", _resp("<think>", FREE1), {}, "t", 1.0, [])
    assert open_only["reasoning_in_content_chars"] > 0
    plain = K.answer_record(FREE1, "v1", "d1", {}, "h", _resp('{"monetary_relief": true}', FREE1), {}, "t", 1.0, [])
    assert plain["reasoning_chars"] == 0 and CA.reasoning_off(FREE1, [plain])


def test_scrub_never_reads_the_key_file(fake_key, monkeypatch):
    """Correction, m1: no request, no read of the key file, also on an error path."""
    import builtins
    opened, real_open = [], builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1])
    monkeypatch.setattr(K, "_KEYS_READ", set())
    K.scrub("x " + FAKE_KEY)
    K.CloudError(400, "y")
    K.classify(RuntimeError("z"), PAID1)
    assert str(fake_key) not in opened
    assert K.scrub(FAKE_KEY) == "[redacted]"


def test_a_per_minute_429_is_not_the_daily_limit():
    """Correction, m2."""
    assert K.classify(K.CloudError(429, "Rate limit exceeded", remaining="0", limit="20"), FREE1)[0] == "transient"
    assert K.classify(K.CloudError(429, "Rate limit exceeded: free-models-per-min", remaining="0"), FREE1)[0] == \
        "transient"
    assert K.classify(K.CloudError(429, "Rate limit exceeded: free-models-per-day", remaining="0", limit="1000"),
                      FREE1)[0] == "quota"


def test_records_written_before_any_request_do_not_block_the_next_start(sandbox, monkeypatch):
    """Correction, m10, and Correction 2, C17 (the review's m-d, PROBE_G): listing checks, the key's readings, the
    estimate and a refusal before the first send need no commit; a journaled request does."""
    monkeypatch.setattr(C, "ROOT", sandbox["tmp"])
    rel, srel = K.RUN_LOG.relative_to(C.ROOT).as_posix(), K.SPEND.relative_to(C.ROOT).as_posix()
    X._append(K.RUN_LOG, {"at": X.now(), "event": "run_started", "model": PAID1})
    committed = {rel: K.RUN_LOG.read_text(), srel: ""}
    monkeypatch.setattr(X, "git", lambda *a: committed[a[1].split(":", 1)[1]] if a[0] == "show" else "")
    X._append(K.RUN_LOG, {"at": X.now(), "event": "listing", "model": FREE1, "ok": False})
    X._append(K.RUN_LOG, {"at": X.now(), "event": "run_stopped", "model": FREE1, "verdict": "unavailable"})
    assert K.only_pre_request_appends([rel])
    assert not K.only_pre_request_appends([rel, "sent.jsonl"])
    with pytest.raises(X.Refusal, match="did not report limit_reset"):     # a paid model refused before any send
        run(PAID1, sandbox, FakeTransport(omit=("limit_reset",)))
    run(FREE1, sandbox, FakeTransport(remaining=K.FREE_DAILY_MARGIN))     # a free run with no allowance left today
    assert X._jsonl(K.JOURNAL) == [] and K.only_pre_request_appends([rel, srel])
    X._append(K.RUN_LOG, {"at": X.now(), "event": "closed_incomplete", "model": FREE1, "reason": "x"})
    assert not K.only_pre_request_appends([rel, srel])


# ------------------------------------------------------------------ preflight (Correction, m7)
@pytest.fixture
def pre_env(sandbox, monkeypatch):
    monkeypatch.setattr(K, "preflight", PREFLIGHT)
    monkeypatch.setattr(X, "state", lambda: {"status": {t: "complete" for t, _ in X.MODELS}})
    monkeypatch.setattr(H, "state", lambda: {"status": {t: "complete" for t in H.MODELS}})
    env = {"porcelain": "", "head": "h" * 40, "changed": ""}

    def git(*a):
        if a[0] == "rev-parse":
            return env["head"] + "\n"
        if a[0] == "status":
            return env["porcelain"]
        if a[0] == "diff":
            return env["changed"]
        return ""
    monkeypatch.setattr(X, "git", git)
    import judge.label as L
    env["locked"] = True
    monkeypatch.setattr(L, "is_locked", lambda who: env["locked"])
    models, _, eps = S.load_snapshot()
    env["getter"] = _getter(models, eps)
    env["published"] = lambda: "h" * 40
    return env


PREFLIGHT = K.preflight


def test_preflight_refuses_each_registered_condition(pre_env, monkeypatch):
    def pre(tag=PAID1, **kw):
        return K.preflight(tag, published_head=kw.get("published", pre_env["published"]), getter=pre_env["getter"])
    assert pre()["sets"]
    pre_env["porcelain"] = " M src/judge/cloud.py\n"
    with pytest.raises(X.Refusal, match="outside results/models/cloud/"):
        pre()
    pre_env["porcelain"] = " M results/models/cloud/sent.jsonl\n"
    with pytest.raises(X.Refusal, match="commit and publish the previous model"):
        pre()
    pre_env["porcelain"] = ""
    with pytest.raises(X.Refusal, match="not the commit GitHub shows"):
        pre(published=lambda: "x" * 40)
    pre_env["changed"] = "src/judge/cloud.py\n"
    with pytest.raises(X.Refusal, match="differ from the branch's published code commit"):
        pre()
    pre_env["changed"] = ""
    with pytest.raises(X.Refusal, match="comes before"):
        pre("moonshotai/kimi-k2.6")
    X._append(K.RUN_LOG, {"at": X.now(), "event": "closed_incomplete", "model": PAID1, "reason": "x"})
    with pytest.raises(X.Refusal, match="is closed"):
        pre()
    X._append(K.RUN_LOG, {"at": X.now(), "event": "budget_stop", "model": PAID1, "scope": "branch"})
    with pytest.raises(X.Refusal, match="15 USD cap"):
        pre("moonshotai/kimi-k2.6")
    assert pre(FREE1)["sets"]                         # the free queue is not stopped by the branch cap
    pre_env["locked"] = False
    with pytest.raises(X.Refusal, match="not locked"):
        pre(FREE1)
    pre_env["locked"] = True
    monkeypatch.setattr(X, "state", lambda: {"status": {}})
    with pytest.raises(X.Refusal, match="Amendment 1 or 2 is not closed"):
        pre(FREE1)
    monkeypatch.setattr(X, "state", lambda: {"status": {t: "complete" for t, _ in X.MODELS}})
    X._append(K.RUN_LOG, {"at": X.now(), "event": "key_limit_stop", "model": FREE1, "arm": "free"})  # Correction 2
    with pytest.raises(X.Refusal, match="key's limit in OpenRouter was spent during the free arm"):
        pre(FREE1)


# ------------------------------------------------------------------ Correction 2, after the second independent review
SPENT = "Budget limit exceeded (key limit). $15.00 of $15.00 used; resets never."
GEMMA = K.REFERENCE_ONLY[0]


def test_403_and_402_are_sorted_by_what_they_report():
    """Correction 2, C11 (the review's MAJOR-1): a 403 is never a rejection; as corrected by Correction 3, C18-C19:
    only the structured fields sort it into a key-limit stop or a content-filter refusal."""
    c = K.classify
    assert c(K.CloudError(403, SPENT, limit_source="openrouter_key_limit"), PAID1)[0] == "key_limit"
    assert c(K.CloudError(403, SPENT), FREE1)[0] == "forbidden"                # the message alone: never a stop (C19)
    assert c(K.CloudError(403, "x", limit_source="openrouter_guardrail_budget"), PAID1)[0] == "account"
    assert c(K.CloudError(403, "Provider returned error (SAFETY)", etype="content_policy_violation"), GEMMA)[0] \
        == "filtered"
    assert c(K.CloudError(403, "refused", etype="refusal"), PAID1)[0] == "filtered"
    assert c(K.CloudError(403, "Forbidden"), PAID1)[0] == "forbidden"
    assert c(K.CloudError(403, "flagged", moderation={"reasons": ["x"]}), PAID1)[0] == "moderated"
    assert c(K.CloudError(403, "No endpoints found matching your data policy"), FREE1)[0] == "account"
    assert c(K.CloudError(402, "Insufficient credits in flight", limit_source=K.IN_FLIGHT_SOURCE), PAID1)[0] \
        == "transient"
    assert c(K.CloudError(402, "Insufficient credits"), PAID1)[0] == "account"
    assert c(K.CloudError(404, "No endpoints found"), PAID1)[0] == "rejected"
    # the error type is read from metadata.error_type, as documented; `reasons` only from OpenRouter's moderation
    e = K._error_from_json({"code": 403, "message": "Provider returned error", "metadata": {
        "error_type": "content_policy_violation", "provider_name": "Google AI Studio", "raw": "SAFETY"}}, 403, {})
    assert e.etype == "content_policy_violation" and e.moderation is None and c(e, GEMMA)[0] == "filtered"
    e = K._error_from_json({"code": 403, "message": SPENT, "metadata": {"limit_source": "openrouter_key_limit"}},
                           403, {})
    assert e.limit_source == "openrouter_key_limit" and c(e, PAID1)[0] == "key_limit"
    assert K._error_from_json({"code": 400, "message": "m", "type": "bad"}, 400, {}).etype == "bad"


def test_a_spent_key_limit_stops_the_arm_and_never_closes_a_model_as_rejected(sandbox):
    """Correction 2, C13 (the review's PROBE_A: a paid model closed as 'rejected' on the key's spent limit)."""
    for i, date in enumerate(("Thu, 12 Nov 2026 12:00:00 GMT", "Fri, 13 Nov 2026 12:00:01 GMT")):
        err = K.CloudError(403, SPENT, limit_source="openrouter_key_limit", date=date)
        with pytest.raises(X.Refusal, match="no model is closed by it"):
            run(PAID1, sandbox, FakeTransport(fail_at=2, error=err))
    assert not K.state()["status"]                                       # no model closed by the 403
    assert K.repeated(PAID1, X._jsonl(K.RUN_LOG), X._jsonl(K.JOURNAL)) is None
    assert [e["arm"] for e in X._jsonl(K.RUN_LOG) if e["event"] == "key_limit_stop"] == ["paid", "paid"]
    assert K.arm_stopped_by_key(K.PAID_QUEUE[2], X._jsonl(K.RUN_LOG)) and not K.arm_stopped_by_key(
        FREE1, X._jsonl(K.RUN_LOG))
    assert all(e["kind"] != "rejected" for e in X._jsonl(K.JOURNAL))
    why = K.close_incomplete(PAID1)                                      # only by the operator, as a budget stop
    assert why.startswith("incomplete (2 of 10 answered): stopped by the budget") and "rejected" not in why
    assert K.close_incomplete(K.PAID_QUEUE[2]).startswith("not run: stopped by the budget")
    assert K.verify(records=sandbox["records"]) == []


def test_a_content_filter_403_fails_one_request_and_the_run_goes_on(sandbox):
    """Correction 2, C12 (the review's PROBE_B: Gemma closed at 2 of 10 answers by Google AI Studio's filter)."""
    def safety(date):
        return K.CloudError(403, "Provider returned error (SAFETY)", etype="content_policy_violation", date=date)
    out = run(GEMMA, sandbox, FakeTransport(fail_at=3, error=safety("Thu, 12 Nov 2026 12:00:00 GMT")))
    assert out["new_answers"] == 9 and out["filtered_to_send_again"] == 1 and not K.state()["status"]
    kinds = [e["kind"] for e in X._jsonl(K.JOURNAL)]
    assert kinds.count("filtered") == 1 and "rejected" not in kinds
    assert [e["event"] for e in X._jsonl(K.RUN_LOG)].count("filtered") == 1
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(GEMMA)
    t = FakeTransport(fail_at=1, error=safety("Thu, 12 Nov 2026 13:00:00 GMT"))
    run(GEMMA, sandbox, t)                                  # refused again one hour later: not final (Correction 3, C20)
    assert len(t.calls) == 1 and not K.state()["status"]
    t = FakeTransport(fail_at=1, error=safety("Fri, 13 Nov 2026 12:00:00 GMT"))
    run(GEMMA, sandbox, t)                                  # refused again 24 hours after the first: final
    assert len(t.calls) == 1
    assert K.state()["status"][GEMMA] == ("incomplete (9 of 10 answered): 1 request(s) refused by the provider's "
                                          "content filter on 2 runs at least 24 hours apart")
    assert K.verify(records=sandbox["records"]) == []
    from judge import cloud_analysis as CA
    d = CA.deviations(GEMMA, X._jsonl(K.JOURNAL), X._jsonl(K.RUN_LOG), [], X._jsonl(K.CACHE_FILE))
    assert d["filtered"] == 3 and d["filtered_final"] == 1
    k = [e["key"] for e in X._jsonl(K.JOURNAL) if e["kind"] == "filtered"][0]
    X._append(K.JOURNAL, {"at": X.now(), "key": k, "kind": "send", "request_sha256": "x", "fields_dropped": []})
    assert any("after its content-filter refusal" in p for p in K.verify(records=sandbox["records"]))


def test_a_filtered_request_answered_on_its_second_run_is_an_answer(sandbox):
    err = K.CloudError(403, "Provider returned error", etype="refusal")
    run(PAID1, sandbox, FakeTransport(fail_at=5, error=err))
    out = run(PAID1, sandbox, FakeTransport())
    assert out["missing"] == 0 and K.state()["status"][PAID1] == "complete"
    assert K.verify(records=sandbox["records"]) == []


def test_make_check_accepts_a_model_over_its_cap_only_by_its_journaled_overrun(sandbox, monkeypatch):
    """Correction 2, C14 (the review's MAJOR-2, PROBE_D: 0.8917 USD spent against a sub-cap of 0.5)."""
    monkeypatch.setattr(K, "SUB_CAP_USD", 0.5)
    with pytest.raises(X.Refusal, match="output above max_tokens"):
        run(PAID1, sandbox, FakeTransport(usage={"prompt_tokens": 700, "completion_tokens": 300000}))
    assert np.isclose(K.committed_for(PAID1), 0.8917, atol=1e-4)
    assert K.state()["status"][PAID1].startswith("incomplete (1 of 10 answered)")
    assert K.verify(records=sandbox["records"]) == []
    good = K.SPEND.read_text()
    extra = 0.6                                                         # spend the overrun does not account for
    X._append(K.SPEND, {"at": X.now(), "event": "charged", "model": PAID1, "key": f"{PAID1}|v1|d2",
                        "cost_usd": extra, "running_total_usd": round(K.spend_state()["computed"] + extra, 6)})
    assert any("by more than its journaled overrun" in p for p in K.verify(records=sandbox["records"]))
    K.SPEND.write_text(good)
    K.JOURNAL.write_text("".join(ln + "\n" for ln in K.JOURNAL.read_text().splitlines() if '"overrun"' not in ln))
    assert any("no journaled overrun accounts for it" in p for p in K.verify(records=sandbox["records"]))


def test_a_quarantined_answer_above_its_reservation_is_an_overrun_and_closes_a_paid_model(sandbox):
    """Correction 2, C14 (the review's m-c, PROBE_F)."""
    t = FakeTransport(provider="SomeoneElse", usage={"prompt_tokens": 700, "completion_tokens": 40, "cost": 0.89})
    with pytest.raises(X.Refusal, match="quarantined answer cost"):
        run(PAID1, sandbox, t)
    ov = [e for e in X._jsonl(K.JOURNAL) if e["kind"] == "overrun"]
    assert len(ov) == 1 and ov[0]["quarantined"] and K.state()["status"][PAID1].startswith("not run:")
    assert K.verify(records=sandbox["records"]) == []


def test_an_error_inside_a_200_without_usage_is_charged_on_a_paid_model_only(sandbox):
    """Correction 2, C17 (the review's m-e, PROBE_H)."""
    err = K.CloudError(502, "upstream error", in_200=True)
    assert run(PAID1, sandbox, FakeTransport(fail_at=2, error=err))["missing"] == 0
    charged = [e for e in X._jsonl(K.SPEND) if e["event"] == "charged"]
    assert len(charged) == 1 and charged[0]["note"].startswith("worst case")
    assert run(FREE1, sandbox, FakeTransport(fail_at=2, error=err))["missing"] == 0
    assert len([e for e in X._jsonl(K.SPEND) if e["event"] == "charged"]) == 1
    assert K.verify(records=sandbox["records"]) == []


def test_an_undated_listing_check_does_not_block_a_later_full_day_of_failures():
    """Correction 2, C17 (the review's m-b, PROBE_E)."""
    undated = [{"at": "2026-11-12T00:00:00Z", "event": "listing", "model": FREE1, "ok": False, "server_date": None}]
    hourly = [{"at": "2026-11-12T01:00:00Z", "event": "listing", "model": FREE1, "ok": False, "server_date": d}
              for d in _hourly(1, 40)[::2]]
    assert K.streak_closes(K.unavailable_streak(FREE1, undated + hourly, cache={}))
    assert not K.streak_closes(K.unavailable_streak(FREE1, hourly + undated, cache={}))   # the latest is undated
    assert not K.streak_closes(K.unavailable_streak(FREE1, undated + hourly[:20], cache={}))


def test_the_listing_check_compares_max_completion_tokens_on_paid_models():
    """Correction 2, C17 (the review's m-f): the overrun room of C3 rests on the registered value."""
    models, _, eps = S.load_snapshot()
    for tag, extra, bad in ((PAID1, 1, True), (PAID1, None, True), (PAID1, -1000, False), (FREE1, 1, False)):
        reg = K.ENDPOINT[tag]["max_completion_tokens"]
        changed = {**eps, tag: [{**e, "max_completion_tokens": None if extra is None else reg + extra}
                                if e["tag"] == K.ENDPOINT[tag]["tag"] else e for e in eps[tag]]}
        why = K.listing_check(tag, _getter(models, changed))[0]
        assert (why is not None and "max_completion_tokens" in why) if bad else why is None


def test_a_keys_label_is_shaped_like_a_key():
    """Correction 2, C17 (the review's m-h, PROBE_I)."""
    assert K.scrub("label sk-or-v1-au7...890 here") == "label [redacted] here"
    assert K.KEY_PATTERN.search("sk-or-v1-abc...xyz")


def test_the_key_readings_reported_are_those_of_the_paid_runs():
    """Correction 2, C17 (the review's m-m)."""
    from judge import cloud_analysis as CA
    log = [{"at": "a", "event": "account_before", "model": FREE1, "limit": 15.0, "limit_remaining": 15.0},
           {"at": "b", "event": "account_before", "model": PAID1, "limit": 15.0, "limit_remaining": 14.5},
           {"at": "c", "event": "account_after", "model": PAID1, "limit": 15.0, "limit_remaining": 12.0},
           {"at": "d", "event": "account_after", "model": FREE1, "limit": 15.0, "limit_remaining": 12.0}]
    acc = CA.account(log)
    assert acc["first"]["at"] == "b" and acc["last"]["at"] == "c"
    assert CA.account(log[:1] + log[3:]) == {}


def test_the_report_lists_the_recorded_deviations(sandbox):
    """Correction 2, C17 (the review's m-g): overruns, quarantined answers, resends, content-filter refusals, fields
    left out, answers without a provider field and free runs on another key are counted and printed."""
    from judge import cloud_analysis as CA
    journal = [{"key": f"{PAID1}|v1|d1", "kind": "overrun"}, {"key": f"{PAID1}|v1|d2", "kind": "resend"},
               {"key": f"{FREE1}|v1|d1", "kind": "filtered", "server_date": "Thu, 12 Nov 2026 12:00:00 GMT"},
               {"key": f"{FREE1}|v1|d1", "kind": "filtered", "server_date": "Fri, 13 Nov 2026 12:00:00 GMT"}]
    log = [{"event": "field_dropped", "model": PAID1, "field": "seed"},
           {"event": "deviation", "kind": "key_differs", "model": FREE1}]
    out = {"order": [FREE1, PAID1], "models": {
        t: {"deviations": CA.deviations(t, journal, log, [{"model": PAID1}], [{"provider": None}])}
        for t in (FREE1, PAID1)}}
    s = CA.deviations_sentence(out)
    assert f"{PAID1}: 1 overrun, 1 answer kept in quarantine, 1 resend, 1 answer without a provider field, left out " \
           "seed" in s
    assert f"{FREE1}: 2 content-filter refusals, 1 request refused on two runs at least 24 hours apart" in s \
        and "another key" in s
    empty = {"order": [FREE1], "models": {FREE1: {"deviations": CA.deviations(FREE1, [], [], [], [])}}}
    assert CA.deviations_sentence(empty).startswith("Recorded deviations: none")


# ------------------------------------------------------------------ Correction 3, after the third independent review
def _http(code, body, headers=None):
    """A CloudError built exactly as OpenRouterTransport builds one from an HTTP error response."""
    import io
    import urllib.error
    hdrs = {"date": "Thu, 12 Nov 2026 12:00:00 GMT", **(headers or {})}
    return K._http_failure(urllib.error.HTTPError("https://x", code, "Forbidden", hdrs, io.BytesIO(body.encode())))


def _err(code, message, **meta):
    return json.dumps({"error": {"code": code, "message": message, "metadata": meta}})


CF_HTML = "<html><title>Attention Required! | Cloudflare</title>Sorry, you have been blocked</html>"


def test_documented_403_and_402_shapes_are_sorted_by_their_structured_fields_only():
    """Correction 3, C18-C19 (the review's R1, MAJOR-1 to MAJOR-3 and m-4), on the shapes of api-reference/errors and
    limits read on 2026-11-12: a 403 is a content-filter refusal only with error_type content_policy_violation or
    refusal; a budget stop only with limit_source openrouter_key_limit, on a 403 or a 402; message text, and the
    provider's raw payload above all, never sorts an error into either."""
    cases = (
        ("guardrail block (the documentation's example)", 403,
         _err(403, "Request blocked: prompt injection patterns detected", patterns=["ignore all previous"]), "forbidden"),
        ("permission_denied", 403, _err(403, "Forbidden", error_type="permission_denied"), "forbidden"),
        ("a CDN's HTML page", 403, CF_HTML, "forbidden"),
        ("budget text without limit_source", 403, _err(403, SPENT), "forbidden"),
        ("filter whose raw text names a limit", 403, _err(403, "Provider returned error", error_type=
         "content_policy_violation", provider_name="Google AI Studio", raw="safety limit exceeded"), "filtered"),
        ("refusal", 403, _err(403, "Provider returned error", error_type="refusal", provider_name="X"), "filtered"),
        ("permission_denied whose raw text names a rate limit", 403, _err(403, "Provider returned error",
         error_type="permission_denied", raw="Rate limit exceeded for this project"), "forbidden"),
        ("402 openrouter_key_limit (Handling 402 errors)", 402, _err(402, "Key limit exceeded",
         limit_source="openrouter_key_limit"), "key_limit"),
        ("403 openrouter_key_limit (Budget errors)", 403, _err(403, SPENT, limit_source="openrouter_key_limit"),
         "key_limit"),
        ("403 guardrail budget that resets", 403, _err(403, "Budget limit exceeded (weekly limit).", limit_source=
         "openrouter_guardrail_budget", resets_at="2026-10-12T00:00:00.000Z"), "account"),
        ("403 workspace budget", 403, _err(403, "x", limit_source="openrouter_workspace_budget"), "account"),
        ("402 openrouter_credits", 402, _err(402, "x", limit_source="openrouter_credits"), "account"),
        ("402 in-flight", 402, _err(402, "in flight", limit_source=K.IN_FLIGHT_SOURCE), "transient"),
        ("400 token_limit_exceeded (m-4)", 400, _err(400, "token budget", error_type="token_limit_exceeded"),
         "account"),
        ("400 whose provider raw text names the data policy", 400, _err(400, "Provider returned error",
         raw="violates our data policy"), "rejected"),
        ("404 whose own message names the data policy", 404, _err(404, "No endpoints found matching your data "
         "policy"), "account"),
    )
    for name, code, body, want in cases:
        for tag in (FREE1, PAID1):
            assert K.classify(_http(code, body), tag)[0] == want, name
    e = _http(403, cases[4][2])
    assert "safety limit exceeded" in e.message and "limit" not in e.own_message.replace("Provider", "")
    # an error inside a 200 carries the code in the body: the same sorting applies
    e = K._error_from_json({"code": 403, "message": "m", "metadata": {"error_type": "content_policy_violation"}}, 200,
                           {}, in_200=True)
    assert K.classify(e, GEMMA)[0] == "filtered"
    e = K._error_from_json({"code": 403, "message": "m", "metadata": {"error_type": "permission_denied"}}, 200, {},
                           in_200=True)
    assert K.classify(e, GEMMA)[0] == "forbidden"


def test_a_storm_of_other_403s_stops_each_run_and_never_closes_a_model(sandbox):
    """Correction 3, C18 (the review's R2): a 403 that is not a content filter's (here a CDN's page) stops the run as
    an infrastructure or permission condition, journaled as such; quick runs one after another close nothing."""
    for date in ("Thu, 12 Nov 2026 12:00:00 GMT", "Thu, 12 Nov 2026 12:03:00 GMT", "Sat, 14 Nov 2026 12:00:00 GMT"):
        t = FakeTransport(fail_at=range(1, 100), error=_http(403, CF_HTML, {"date": date}))
        with pytest.raises(X.Refusal, match="infrastructure or permission condition"):
            run(FREE1, sandbox, t)
        assert len(t.calls) == 1                           # the run stops at the first: one request of the day's quota
    kinds = [e["kind"] for e in X._jsonl(K.JOURNAL)]
    assert kinds.count("forbidden") == 3 and "filtered" not in kinds and "rejected" not in kinds
    assert not K.state()["status"] and K.error_count(X._jsonl(K.JOURNAL), FREE1) == 0
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(FREE1)
    assert K.verify(records=sandbox["records"]) == []
    out = run(FREE1, sandbox, FakeTransport())                 # the condition is gone: the same requests are answered
    assert out["missing"] == 0 and K.state()["status"][FREE1] == "complete"
    assert K.verify(records=sandbox["records"]) == []


def test_content_filter_refusals_minutes_apart_do_not_close_a_model(sandbox):
    """Correction 3, C20 (the review's R2): a second refusal counts only at least 24 hours after the first by the
    server's dates, as the rejection rule; two runs minutes apart close nothing, a third a day later closes the model."""
    def filt(date):
        return _http(403, _err(403, "Provider returned error", error_type="content_policy_violation",
                               provider_name="Nvidia", raw="SAFETY"), {"date": date})
    out = run(FREE1, sandbox, FakeTransport(fail_at=range(1, 100), error=filt("Thu, 12 Nov 2026 12:00:00 GMT")))
    assert out["filtered_to_send_again"] == 10 and not K.state()["status"]
    out = run(FREE1, sandbox, FakeTransport(fail_at=range(1, 100), error=filt("Thu, 12 Nov 2026 12:03:00 GMT")))
    assert out["filtered_to_send_again"] == 10 and not K.state()["status"]
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(FREE1)
    t = FakeTransport(fail_at=range(1, 100), error=filt("Fri, 13 Nov 2026 11:59:59 GMT"))
    out = run(FREE1, sandbox, t)                                # 23:59:59 after the first: still not final
    assert out["filtered_to_send_again"] == 10 and not K.state()["status"] and len(t.calls) == 10
    t = FakeTransport(fail_at=range(1, 100), error=filt("Fri, 13 Nov 2026 12:00:00 GMT"))
    run(FREE1, sandbox, t)
    assert K.state()["status"][FREE1] == ("not run: 10 request(s) refused by the provider's content filter on 2 runs "
                                          "at least 24 hours apart")
    assert K.verify(records=sandbox["records"]) == []
    k = f"{FREE1}|v1|d1"
    X._append(K.JOURNAL, {"at": X.now(), "key": k, "kind": "send", "request_sha256": "x", "fields_dropped": []})
    assert any("after its content-filter refusal became final" in p for p in K.verify(records=sandbox["records"]))


def test_a_filter_refusal_recorded_without_a_filters_error_type_fails_the_check(sandbox):
    """Correction 3, C18: `make check` fails on a content-filter refusal journaled for an error that is not one."""
    run(FREE1, sandbox, FakeTransport(fail_at=2, error=K.CloudError(403, "x", etype="refusal")))
    assert K.verify(records=sandbox["records"]) == []
    k = [e["key"] for e in X._jsonl(K.JOURNAL) if e["kind"] == "filtered"][0]
    X._append(K.JOURNAL, {"at": X.now(), "key": k, "kind": "filtered", "status": 403, "error_type": None})
    assert any("are not a content filter's" in p for p in K.verify(records=sandbox["records"]))


def test_a_content_filter_whose_raw_text_names_a_limit_stops_no_arm(sandbox):
    """Correction 3, C19 (the review's R3): message text never stops an arm, and `--close-incomplete` closes no model
    that never ran on a text match."""
    body = _err(403, "Provider returned error", error_type="content_policy_violation", provider_name="X",
                raw="Request blocked: safety limit exceeded")
    out = run(FREE1, sandbox, FakeTransport(fail_at=2, error=_http(403, body)))
    assert out["new_answers"] == 9 and out["filtered_to_send_again"] == 1
    with pytest.raises(X.Refusal, match="infrastructure or permission"):
        run(FREE1, sandbox, FakeTransport(fail_at=1, error=_http(403, _err(
            403, "Provider returned error", error_type="permission_denied", raw="Rate limit exceeded"))))
    log = X._jsonl(K.RUN_LOG)
    assert "key_limit_stop" not in events() and not any(K.arm_stopped_by_key(t, log) for t in K.MODELS)
    with pytest.raises(X.Refusal, match="has not started"):
        K.close_incomplete(K.FREE_QUEUE[1])
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(FREE1)
    assert K.verify(records=sandbox["records"]) == []
    X._append(K.RUN_LOG, {"at": X.now(), "event": "key_limit_stop", "model": FREE1, "arm": "free"})
    assert any("key_limit_stop of the free arm without limit_source" in p for p in K.verify(records=sandbox["records"]))


def test_a_key_limit_spent_with_a_402_stops_the_arm_and_its_models_can_be_closed(sandbox):
    """Correction 3, C19 (the review's R4, first half): 'Handling 402 errors' documents the spent key limit as a 402
    with limit_source openrouter_key_limit; it is the same budget stop of the arm as the 403."""
    err = _http(402, _err(402, "Key limit exceeded", limit_source="openrouter_key_limit"))
    with pytest.raises(X.Refusal, match="no model is closed by it"):
        run(PAID1, sandbox, FakeTransport(fail_at=3, error=err))
    stop = [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "key_limit_stop"]
    assert len(stop) == 1 and stop[0]["arm"] == "paid" and stop[0]["status"] == 402
    assert [e["limit_source"] for e in X._jsonl(K.JOURNAL) if e["kind"] == "key_limit"] == ["openrouter_key_limit"]
    assert not K.state()["status"]
    assert K.close_incomplete(PAID1).startswith("incomplete (2 of 10 answered): stopped by the budget")
    assert K.close_incomplete(K.PAID_QUEUE[1]).startswith("not run: stopped by the budget")
    assert K.close_incomplete(K.PAID_QUEUE[2]).startswith("not run: stopped by the budget")
    assert not K.arm_stopped_by_key(FREE1, X._jsonl(K.RUN_LOG))
    assert K.verify(records=sandbox["records"]) == []


def test_a_short_remaining_key_limit_closes_the_paid_model_so_the_queue_goes_on(sandbox):
    """Correction 3, C19 (the review's R4, second half): the key's remaining limit cannot cover a paid model's remaining
    worst case, and the limit cannot be raised (exactly 15 USD, never reset): the model is closed by the budget and
    the next model of the queue may run; a key that is configured otherwise is a resumable refusal and closes
    nothing."""
    with pytest.raises(X.Refusal, match="account condition"):
        run(PAID1, sandbox, FakeTransport(fail_at=3, error=K.CloudError(401, "No auth")))
    for kw in ({"limit": 10.0, "limit_remaining": 10.0}, {"limit_reset": "monthly"}, {"omit": ("limit_remaining",)}):
        with pytest.raises(X.Refusal, match="nothing was sent.$"):
            run(PAID1, sandbox, FakeTransport(**kw))
        assert not K.state()["status"]
    t = FakeTransport(limit_remaining=0.01)
    with pytest.raises(X.Refusal, match="it is closed"):
        run(PAID1, sandbox, t)
    assert t.calls == []
    assert K.state()["status"][PAID1].startswith("incomplete (2 of 10 answered): stopped by the budget: the key's "
                                                 "remaining limit in OpenRouter (0.01 USD)")
    stop = [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "budget_stop"]
    assert len(stop) == 1 and stop[0]["scope"] == "key" and stop[0]["model"] == PAID1
    with pytest.raises(X.Refusal, match="it is closed"):
        run(K.PAID_QUEUE[1], sandbox, FakeTransport(limit_remaining=0.01))
    assert K.state()["status"][K.PAID_QUEUE[1]].startswith("not run: stopped by the budget")
    out = run(K.PAID_QUEUE[2], sandbox, FakeTransport(limit_remaining=14.0))
    assert out["missing"] == 0 and K.state()["status"][K.PAID_QUEUE[2]] == "complete"
    assert K.verify(records=sandbox["records"]) == []


def test_a_paid_model_refuses_a_key_under_a_guardrail_or_workspace_budget(sandbox):
    """Correction 3, C19: the paid arm runs only under the key's own limit; another budget listed by GET /api/v1/key is
    a resumable refusal (the owner removes it), never a budget stop."""
    class Budgeted(FakeTransport):
        def __init__(self, budgets, **kw):
            super().__init__(**kw)
            self.budgets = budgets

        def get(self, url):
            d = super().get(url)
            d["data"]["budgets"] = self.budgets
            d["data"]["effective_budget"] = self.budgets[0] if self.budgets else None
            return d
    t = Budgeted([{"scope": "member", "limit_usd": 50, "remaining_usd": 10}])
    with pytest.raises(X.Refusal, match=r"budget other than the key's own limit \(member\)"):
        run(PAID1, sandbox, t)
    assert t.calls == [] and not K.state()["status"]
    out = run(PAID1, sandbox, Budgeted([{"scope": "key_limit", "limit_usd": 15, "remaining_usd": 15}]))
    assert out["missing"] == 0


def test_an_overrun_line_written_twice_does_not_hide_overspend(sandbox, monkeypatch):
    """Correction 3, C21 (the review's m-1, R5): each spend entry accounts for one overrun at most, and a paid model
    has at most one overrun."""
    monkeypatch.setattr(K, "SUB_CAP_USD", 0.5)
    with pytest.raises(X.Refusal):
        run(PAID1, sandbox, FakeTransport(usage={"prompt_tokens": 700, "completion_tokens": 300000}))
    assert K.verify(records=sandbox["records"]) == []
    X._append(K.SPEND, {"at": X.now(), "event": "charged", "model": PAID1, "key": f"{PAID1}|v1|d2", "cost_usd": 0.6,
                        "running_total_usd": round(K.spend_state()["computed"] + 0.6, 6)})
    ov = json.loads([ln for ln in K.JOURNAL.read_text().splitlines() if '"overrun"' in ln][0])
    X._append(K.JOURNAL, {**ov, "at": X.now()})
    problems = K.verify(records=sandbox["records"])
    assert any("by more than its journaled overrun" in p for p in problems)
    assert any("more than one overrun entry" in p for p in problems)
    reserved = {e["id"]: e["cost_usd"] for e in X._jsonl(K.SPEND) if e["event"] == "reserved"}
    once = K.overrun_excess(PAID1, [ov], X._jsonl(K.SPEND), reserved)
    assert once > 0 and K.overrun_excess(PAID1, [ov, ov], X._jsonl(K.SPEND), reserved) == once


def test_nan_and_infinity_are_not_key_readings():
    """Correction 3, C21 (the review's m-2, R6): json.loads accepts NaN and Infinity; neither passes "exactly 15" nor
    counts as a remaining limit."""
    base = {"limit": 15.0, "limit_reset": None, "limit_remaining": 15.0, "fields_missing": []}
    assert K.paid_key_refusal(PAID1, base, 4.8) is None
    for f, v in (("limit", float("nan")), ("limit", float("inf")), ("limit_remaining", float("nan")),
                 ("limit_remaining", float("inf")), ("limit", "15"), ("limit", True)):
        assert K.paid_key_refusal(PAID1, {**base, f: v}, 4.8) is not None, (f, v)
        assert not K.key_short({**base, f: v}, 4.8)
    parsed = json.loads('{"limit": NaN, "limit_remaining": Infinity, "limit_reset": null}')
    assert K.paid_key_refusal(PAID1, {**parsed, "fields_missing": []}, 4.8) is not None
    assert not K._num(float("nan")) and not K._num(float("-inf")) and K._num(15) and K._num(0.1 * 150)


def test_a_token_limit_exceeded_400_is_an_account_condition_not_a_rejection(sandbox):
    """Correction 3, C18 (the review's m-4): "a token budget enforced by OpenRouter (e.g. credit-based cap)"."""
    for date in ("Thu, 12 Nov 2026 12:00:00 GMT", "Sat, 14 Nov 2026 12:00:00 GMT"):
        err = _http(400, _err(400, "token budget", error_type="token_limit_exceeded"), {"date": date})
        with pytest.raises(X.Refusal, match="account condition"):
            run(PAID1, sandbox, FakeTransport(fail_at=1, error=err))
    assert [e["kind"] for e in X._jsonl(K.JOURNAL) if e["kind"] not in ("send",)] == ["account", "account"]
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(PAID1)


def test_the_in_flight_402_is_not_counted_among_the_error_answers(sandbox, monkeypatch):
    """Correction 3, C21 (the review's m-5): the in-flight budget concerns the account's settling spend, not the model;
    it is waited out with Retry-After and does not count toward the 150 error answers."""
    monkeypatch.setattr(K, "MAX_ERRORS_PER_MODEL", 2)
    flight = _http(402, _err(402, "in flight", limit_source=K.IN_FLIGHT_SOURCE,
                             reason="in_flight_budget_exhausted"), {"retry-after": "7"})
    out = run(PAID1, sandbox, FakeTransport(fail_at=(2, 3, 4), error=flight))
    assert out["missing"] == 0 and K.error_count(X._jsonl(K.JOURNAL), PAID1) == 0
    assert [e["seconds"] for e in X._jsonl(K.RUN_LOG) if e["event"] == "transient_wait"] == [30, 60, 120]
    with pytest.raises(X.Refusal, match="rate limit, an overload or a server error"):
        run(K.PAID_QUEUE[1], sandbox, FakeTransport(fail_at=(2, 3, 4), error=K.CloudError(503, "overloaded")))
    assert K.error_count(X._jsonl(K.JOURNAL), K.PAID_QUEUE[1]) == 2
    assert K.verify(records=sandbox["records"]) == []


def test_an_http_error_that_reports_usage_is_charged_by_it(sandbox):
    """Correction 3, C21 (the review's m-6): C12 releases a failed request's reservation unless the error reports
    usage; an HTTP error body with usage is now read, as an error inside a 200 already was."""
    body = json.dumps({"error": {"code": 400, "message": "Invalid request: bad schema"},
                       "usage": {"prompt_tokens": 700, "completion_tokens": 0}})
    with pytest.raises(X.Refusal, match="rejected"):
        run(PAID1, sandbox, FakeTransport(fail_at=1, error=_http(400, body)))
    charged = [e for e in X._jsonl(K.SPEND) if e["event"] == "charged"]
    assert len(charged) == 1 and np.isclose(charged[0]["cost_usd"], K.ENDPOINT[PAID1]["prompt"] * 700)
    with pytest.raises(X.Refusal, match="infrastructure or permission"):
        run(PAID1, sandbox, FakeTransport(fail_at=1, error=_http(403, CF_HTML)))
    assert len([e for e in X._jsonl(K.SPEND) if e["event"] == "charged"]) == 1
    assert X._jsonl(K.SPEND)[-1]["event"] == "released"
    assert K.verify(records=sandbox["records"]) == []
