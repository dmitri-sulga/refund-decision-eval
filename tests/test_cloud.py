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
                "include_byok_in_limit": False, "is_free_tier": False, "budgets": [], "effective_budget": None,
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


class Clock:
    """This machine's clock as the runner reads it (K.now), set by the tests to the server dates they simulate; each
    reading moves on by a millisecond, so the journals' times stay in order."""

    def __init__(self):
        self.t, self.n = H._server_seconds("Thu, 12 Nov 2026 12:00:00 GMT"), 0

    def set(self, date: str, plus_s: float = 0.0):
        self.t = H._server_seconds(date) + plus_s

    def __call__(self) -> str:
        from datetime import datetime, timezone
        self.n += 1
        return datetime.fromtimestamp(self.t + self.n / 1000, timezone.utc).isoformat(timespec="microseconds")


@pytest.fixture
def clock(monkeypatch):
    c = Clock()
    monkeypatch.setattr(K, "now", c)
    return c


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
    assert before[-1]["fields_missing"] == ["limit", "limit_remaining"]
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
    assert pre(FREE1)["sets"]
    with pytest.raises(X.Refusal, match="the free queue is not closed"):       # Correction 4, C24
        pre()
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
    X._append(K.RUN_LOG, {"at": X.now(), "event": "key_limit_stop", "model": FREE1, "arm": "free"})
    assert pre(FREE1)["sets"]              # Correction 4, C25 (m-5): a stop without limit_source stops nothing
    X._append(K.RUN_LOG, {"at": X.now(), "event": "key_limit_stop", "model": FREE1, "arm": "free",
                          "limit_source": K.KEY_LIMIT_SOURCE})                       # Correction 2
    with pytest.raises(X.Refusal, match="key's limit in OpenRouter was spent during the free arm"):
        pre(FREE1)


def test_the_paid_arm_starts_only_after_every_free_model_is_closed(pre_env):
    """Correction 4, C24 (the fourth review's m-6): one key serves both arms, and the documentation says a negative
    credit balance can refuse free models too; the free queue therefore runs to its end before the first paid
    request, whichever way each free model was closed."""
    def pre(tag):
        return K.preflight(tag, published_head=pre_env["published"], getter=pre_env["getter"])
    for n, t in enumerate(K.FREE_QUEUE):
        with pytest.raises(X.Refusal, match=r"the free queue is not closed \(" + ", ".join(K.FREE_QUEUE[n:])
                           .replace(".", r"\.")):
            pre(PAID1)
        X._append(K.RUN_LOG, {"at": X.now(), "event": "model_finished", "model": t, "answers": 600} if n % 2 else
                  {"at": X.now(), "event": "closed_incomplete", "model": t, "reason": "not run: x"})
    assert pre(PAID1)["sets"]


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


def test_a_content_filter_403_fails_one_request_and_the_run_goes_on(sandbox, clock):
    """Correction 2, C12 (the review's PROBE_B: Gemma closed at 2 of 10 answers by Google AI Studio's filter), as
    corrected by Correction 4, C22: the refused request is sent to the filter exactly once more, 24 hours later."""
    def safety(date):
        return K.CloudError(403, "Provider returned error (SAFETY)", etype="content_policy_violation", date=date)
    clock.set("Thu, 12 Nov 2026 12:00:00 GMT")
    out = run(GEMMA, sandbox, FakeTransport(fail_at=3, error=safety("Thu, 12 Nov 2026 12:00:00 GMT")))
    assert out["new_answers"] == 9 and out["filtered_to_send_again"] == 1 and not K.state()["status"]
    kinds = [e["kind"] for e in X._jsonl(K.JOURNAL)]
    assert kinds.count("filtered") == 1 and "rejected" not in kinds
    assert [e["event"] for e in X._jsonl(K.RUN_LOG)].count("filtered") == 1
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(GEMMA)
    clock.set("Thu, 12 Nov 2026 13:00:00 GMT")
    t = FakeTransport(fail_at=1, error=safety("Thu, 12 Nov 2026 13:00:00 GMT"))
    out = run(GEMMA, sandbox, t)                            # one hour later: the request is not due, nothing is sent
    assert t.calls == [] and out["filtered_to_send_again"] == 1 and not K.state()["status"]
    clock.set("Fri, 13 Nov 2026 12:10:00 GMT")
    t = FakeTransport(fail_at=1, error=safety("Fri, 13 Nov 2026 12:10:00 GMT"))
    run(GEMMA, sandbox, t)                                  # its second send, 24 hours 10 minutes later: final
    assert len(t.calls) == 1
    assert K.state()["status"][GEMMA] == ("incomplete (9 of 10 answered): 1 request(s) refused by the provider's "
                                          "content filter on both of their 2 sends, at least 24 hours apart")
    assert K.verify(records=sandbox["records"]) == []
    from judge import cloud_analysis as CA
    d = CA.deviations(GEMMA, X._jsonl(K.JOURNAL), X._jsonl(K.RUN_LOG), [], X._jsonl(K.CACHE_FILE))
    assert d["filtered"] == 2 and d["filtered_final"] == 1 and d["filtered_final_local"] == 0
    k = [e["key"] for e in X._jsonl(K.JOURNAL) if e["kind"] == "filtered"][0]
    X._append(K.JOURNAL, {"at": K.now(), "key": k, "kind": "send", "request_sha256": "x", "fields_dropped": []})
    assert any("after its content-filter refusal" in p for p in K.verify(records=sandbox["records"]))


def test_a_filtered_request_answered_on_its_second_run_is_an_answer(sandbox, clock):
    err = K.CloudError(403, "Provider returned error", etype="refusal")              # no server date: local times
    run(PAID1, sandbox, FakeTransport(fail_at=5, error=err))
    t = FakeTransport()
    out = run(PAID1, sandbox, t)                                 # the same minute: not due (Correction 4, C22)
    assert t.calls == [] and out["filtered_to_send_again"] == 1
    clock.set("Fri, 13 Nov 2026 12:10:01 GMT")
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
    assert f"{FREE1}: 2 content-filter refusals, 1 request refused on both of its 2 sends at least 24 hours apart" \
        in s \
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


def test_content_filter_refusals_minutes_apart_do_not_close_a_model(sandbox, clock):
    """Correction 3, C20 (the review's R2), as corrected by Correction 4, C22: a refused request is sent to the filter
    once more only 24 hours and 10 minutes after its first refusal; runs before then send nothing and close nothing."""
    def filt(date):
        return _http(403, _err(403, "Provider returned error", error_type="content_policy_violation",
                               provider_name="Nvidia", raw="SAFETY"), {"date": date})
    out = run(FREE1, sandbox, FakeTransport(fail_at=range(1, 100), error=filt("Thu, 12 Nov 2026 12:00:00 GMT")))
    assert out["filtered_to_send_again"] == 10 and not K.state()["status"]
    clock.set("Thu, 12 Nov 2026 12:03:00 GMT")
    t = FakeTransport(fail_at=range(1, 100), error=filt("Thu, 12 Nov 2026 12:03:00 GMT"))
    out = run(FREE1, sandbox, t)
    assert out["filtered_to_send_again"] == 10 and not K.state()["status"] and t.calls == []
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(FREE1)
    clock.set("Fri, 13 Nov 2026 12:09:59 GMT")
    t = FakeTransport(fail_at=range(1, 100), error=filt("Fri, 13 Nov 2026 12:09:59 GMT"))
    out = run(FREE1, sandbox, t)                                # 24 h 9 min 59 s after the first: not due yet
    assert out["filtered_to_send_again"] == 10 and not K.state()["status"] and t.calls == []
    clock.set("Fri, 13 Nov 2026 12:10:01 GMT")
    t = FakeTransport(fail_at=range(1, 100), error=filt("Fri, 13 Nov 2026 12:10:01 GMT"))
    run(FREE1, sandbox, t)
    assert len(t.calls) == 10
    assert K.state()["status"][FREE1] == ("not run: 10 request(s) refused by the provider's content filter on both of "
                                          "their 2 sends, at least 24 hours apart")
    assert K.verify(records=sandbox["records"]) == []
    k = f"{FREE1}|v1|d1"
    X._append(K.JOURNAL, {"at": K.now(), "key": k, "kind": "send", "request_sha256": "x", "fields_dropped": []})
    assert any("after its content-filter refusal became final" in p for p in K.verify(records=sandbox["records"]))


def test_a_filter_refusal_recorded_without_a_filters_error_type_fails_the_check(sandbox):
    """Correction 3, C18: `make check` fails on a content-filter refusal journaled for an error that is not one."""
    run(FREE1, sandbox, FakeTransport(fail_at=2, error=K.CloudError(403, "x", etype="refusal")))
    assert K.verify(records=sandbox["records"]) == []
    k = [e["key"] for e in X._jsonl(K.JOURNAL) if e["kind"] == "filtered"][0]
    X._append(K.JOURNAL, {"at": X.now(), "key": k, "kind": "filtered", "status": 403, "error_type": None})
    assert any("are not a content filter's" in p for p in K.verify(records=sandbox["records"]))


def test_a_content_filter_whose_raw_text_names_a_limit_stops_no_arm(sandbox, clock):
    """Correction 3, C19 (the review's R3): message text never stops an arm, and `--close-incomplete` closes no model
    that never ran on a text match."""
    body = _err(403, "Provider returned error", error_type="content_policy_violation", provider_name="X",
                raw="Request blocked: safety limit exceeded")
    out = run(FREE1, sandbox, FakeTransport(fail_at=2, error=_http(403, body)))
    assert out["new_answers"] == 9 and out["filtered_to_send_again"] == 1
    clock.set("Fri, 13 Nov 2026 12:30:00 GMT")                 # its second send is due (Correction 4, C22)
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
    X._append(K.RUN_LOG, {"at": K.now(), "event": "key_limit_stop", "model": FREE1, "arm": "free"})
    assert any("key_limit_stop of the free arm without limit_source" in p for p in K.verify(records=sandbox["records"]))
    # Correction 4, C25 (the fourth review's m-5): such a record stops no arm and lets no model of it be closed
    assert not K.arm_stopped_by_key(K.FREE_QUEUE[1], X._jsonl(K.RUN_LOG))
    with pytest.raises(X.Refusal, match="has not started"):
        K.close_incomplete(K.FREE_QUEUE[1])


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


def test_a_short_remaining_key_limit_closes_the_paid_model_so_the_queue_goes_on(sandbox, monkeypatch):
    """Correction 3, C19 (the review's R4, second half): the key's remaining limit cannot cover a paid model's remaining
    worst case, and the limit cannot be raised (exactly 15 USD, never reset): the model is closed by the budget and
    the next model of the queue may run; a key that is configured otherwise is a resumable refusal and closes
    nothing. As corrected by Correction 4, C25 (the fourth review's m-1, N3): this closing happens on its own only when
    the key's use agrees with the branch's spend journal; key use outside the branch is a recorded deviation, stops the
    run without closing the model, and the operator may then close it with that deviation in its reason."""
    with pytest.raises(X.Refusal, match="account condition"):
        run(PAID1, sandbox, FakeTransport(fail_at=3, error=K.CloudError(401, "No auth")))
    for kw in ({"limit": 10.0, "limit_remaining": 10.0}, {"limit_reset": "monthly"}, {"omit": ("limit_remaining",)}):
        with pytest.raises(X.Refusal, match="nothing was sent.$"):
            run(PAID1, sandbox, FakeTransport(**kw))
        assert not K.state()["status"]
    journaled = K.spend_state()["computed"]
    assert 0 < journaled < 0.01
    # the legitimate path: OpenRouter's use equals the journal, and the model's worst case plus the margin does not fit
    monkeypatch.setattr(K, "CREDIT_MARGIN_USD", 14.999)
    t = FakeTransport(limit_remaining=15.0 - journaled)
    with pytest.raises(X.Refusal, match="it is closed"):
        run(PAID1, sandbox, t)
    assert t.calls == []
    assert K.state()["status"][PAID1].startswith("incomplete (2 of 10 answered): stopped by the budget: the key's "
                                                 "remaining limit in OpenRouter (14.99")
    stop = [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "budget_stop"]
    assert len(stop) == 1 and stop[0]["scope"] == "key" and stop[0]["model"] == PAID1
    assert np.isclose(stop[0]["key_used_usd"], journaled, atol=1e-6) and abs(stop[0]["beyond_journal_usd"]) < 1e-6
    assert not [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "deviation"]
    # N3: 14.99 USD of the key used outside the branch: recorded, the run stops, the model is not closed by itself
    monkeypatch.setattr(K, "CREDIT_MARGIN_USD", 1.0)
    t = FakeTransport(limit_remaining=0.01)
    with pytest.raises(X.Refusal, match="not closed by the budget on its own"):
        run(K.PAID_QUEUE[1], sandbox, t)
    assert t.calls == [] and not K.state()["status"].get(K.PAID_QUEUE[1])
    dev = [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "deviation"]
    assert len(dev) == 1 and dev[0]["kind"] == "key_use_exceeds_journal" and dev[0]["stopped"] is True
    assert np.isclose(dev[0]["beyond_journal_usd"], 14.99 - journaled, atol=1e-6)
    why = K.close_incomplete(K.PAID_QUEUE[1])
    assert why.startswith("not run: stopped by the budget: the key's remaining limit in OpenRouter (0.01 USD)")
    assert "outside the branch's spend journal (a recorded deviation)" in why
    # the key's use agrees with the journal again: the last model runs; a small difference is not a deviation
    out = run(K.PAID_QUEUE[2], sandbox, FakeTransport(limit_remaining=15.0 - journaled - 0.04))
    assert out["missing"] == 0 and K.state()["status"][K.PAID_QUEUE[2]] == "complete"
    assert len([e for e in X._jsonl(K.RUN_LOG) if e["event"] == "deviation"]) == 1
    from judge import cloud_analysis as CA
    d = CA.deviations(K.PAID_QUEUE[1], X._jsonl(K.JOURNAL), X._jsonl(K.RUN_LOG), [], [])
    assert d["key_use_exceeds_journal"] == 1
    assert "1 key reading whose use exceeded the branch's spend journal" in CA.deviations_sentence(
        {"order": [K.PAID_QUEUE[1]], "models": {K.PAID_QUEUE[1]: {"deviations": d}}})
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
    base = {"limit": 15.0, "limit_reset": None, "limit_remaining": 15.0, "fields_missing": [], "budgets": [],
            "effective_budget": None}
    assert K.paid_key_refusal(PAID1, base, 4.8) is None
    for f, v in (("limit", float("nan")), ("limit", float("inf")), ("limit_remaining", float("nan")),
                 ("limit_remaining", float("inf")), ("limit", "15"), ("limit", True)):
        assert K.paid_key_refusal(PAID1, {**base, f: v}, 4.8) is not None, (f, v)
        assert not K.key_short({**base, f: v}, 4.8)
    parsed = json.loads('{"limit": NaN, "limit_remaining": Infinity, "limit_reset": null, "budgets": []}')
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


# ------------------------------------------------------------------ Correction 4, after the fourth independent review
class Flaky(FakeTransport):
    """A provider's filter that refuses each request on its first `refuse` sends and answers after that (the fourth
    review's N2: a filter that is not deterministic, modelled deterministically). `seen` is shared between runs."""

    def __init__(self, seen, refuse, date, **kw):
        super().__init__(**kw)
        self.seen, self.refuse, self.date = seen, refuse, date

    def send(self, body):
        h = json.dumps(body["messages"], sort_keys=True)
        self.seen[h] = self.seen.get(h, 0) + 1
        if self.seen[h] <= self.refuse:
            self.calls.append(body)
            raise _http(403, _err(403, "Provider returned error", error_type="content_policy_violation",
                                  provider_name="Nvidia"), {"date": self.date})
        return super().send(body)


def _schedule(sandbox, clock, dates, refuse):
    seen, calls, out = {}, [], None
    for d in dates:
        clock.set(d)
        t = Flaky(seen, refuse, d)
        out = run(FREE1, sandbox, t)
        calls.append(len(t.calls))
    return K.state()["status"].get(FREE1), calls, out


D0 = "Thu, 12 Nov 2026 12:00:00 GMT"
LATER = "Fri, 13 Nov 2026 12:15:00 GMT"             # 24 hours 15 minutes after D0


@pytest.mark.parametrize("refuse", (1, 2))
def test_a_filtered_request_is_sent_exactly_twice_whatever_the_number_of_runs(tmp_path, monkeypatch, refuse):
    """Correction 4, C22 (the fourth review's MAJOR-1, N2a against N2b): whether a compared model is counted must not
    depend on how many runs happen in the first 24 hours. The same filter gives the same status to an operator who
    runs twice, 24 hours apart, and to one who runs three times within two hours and then once after 24 hours: the
    request is sent exactly twice, the second time after the wait, and runs before then send nothing."""
    results = []
    for schedule in ((D0, LATER), (D0, "Thu, 12 Nov 2026 13:00:00 GMT", "Thu, 12 Nov 2026 14:00:00 GMT", LATER)):
        with monkeypatch.context() as m:
            sub = tmp_path / str(len(schedule))
            sub.mkdir()
            m.setattr(K, "DIR", sub)
            for name in ("CACHE_FILE", "JOURNAL", "RUN_LOG", "SPEND", "REGISTRATION", "LOCK", "QUARANTINE"):
                m.setattr(K, name, sub / getattr(K, name).name)
            sets = {"dev": ["d1", "d2", "d3"], "labelled": ["l1", "l2"]}
            records = {i: {"product": "P", "sub_product": "S", "issue": "I", "sub_issue": "J",
                           "narrative": f"text {i}", "label": 0} for i in ["d1", "d2", "d3", "l1", "l2", "h1"]}
            m.setattr(K, "preflight", lambda tag, published_head=None, getter=None: {
                "sets": sets, "test": {"h1"}, "registration": FILLED})
            m.setattr(X, "item_sets", lambda: (sets, {"h1"}))
            K.REGISTRATION.write_text(json.dumps(FILLED))
            c = Clock()
            m.setattr(K, "now", c)
            status, calls, out = _schedule({"records": records}, c, schedule, refuse)
            sends = [e for e in X._jsonl(K.JOURNAL) if e["kind"] in ("send", "resend", "retry")]
            assert K.verify(records=records) == []
            results.append((status, calls, len(sends)))
    (status_a, calls_a, sends_a), (status_b, calls_b, sends_b) = results
    assert status_a == status_b and sends_a == sends_b == 20
    assert calls_a == [10, 10] and calls_b == [10, 0, 0, 10]
    if refuse == 2:
        assert status_a == ("not run: 10 request(s) refused by the provider's content filter on both of their 2 sends, "
                            "at least 24 hours apart")
    else:
        assert status_a == "complete"


def test_the_fourth_reviews_n2_schedules_without_a_later_run_close_nothing_alike(sandbox, clock):
    """The fourth review's N2 as it was written (two runs exactly 24 hours apart; three runs within two hours): with
    the wait of 24 hours and 10 minutes, neither sends the request a second time and neither closes the model."""
    status, calls, out = _schedule(sandbox, clock, (D0, "Thu, 12 Nov 2026 13:00:00 GMT",
                                                    "Thu, 12 Nov 2026 14:00:00 GMT"), 2)
    assert status is None and calls == [10, 0, 0] and out["filtered_to_send_again"] == 10


def test_a_filter_refusal_without_a_server_date_becomes_final_by_the_local_clock(sandbox, clock):
    """Correction 4, C22 (the fourth review's m-4, N8): a refusal without a server date is final at the second send by
    the journal's local times, and that is recorded as a deviation; it can no longer stay open for ever."""
    e = _http(403, _err(403, "e", error_type="content_policy_violation"), {"date": ""})
    assert e.date == ""
    for when, calls in ((D0, 10), ("Thu, 12 Nov 2026 18:00:00 GMT", 0), (LATER, 1)):
        clock.set(when)
        t = FakeTransport(fail_at=1, error=e)
        run(FREE1, sandbox, t)
        assert len(t.calls) == calls
    assert K.state()["status"][FREE1] == ("incomplete (9 of 10 answered): 1 request(s) refused by the provider's "
                                          "content filter on both of their 2 sends, at least 24 hours apart")
    dev = [x for x in X._jsonl(K.RUN_LOG) if x["event"] == "deviation"]
    assert [x["kind"] for x in dev] == ["filter_final_local_clock"] and dev[0]["server_gap_hours"] is None
    from judge import cloud_analysis as CA
    d = CA.deviations(FREE1, X._jsonl(K.JOURNAL), X._jsonl(K.RUN_LOG), [], X._jsonl(K.CACHE_FILE))
    assert d["filtered_final"] == 1 and d["filtered_final_local"] == 1
    assert K.verify(records=sandbox["records"]) == []
    # the server's dates less than 24 hours apart although the wait was kept on this machine: final by local times
    fs = K.filter_state([{"kind": "filtered", "at": "2026-11-12T12:00:00+00:00", "server_date": D0},
                         {"kind": "filtered", "at": "2026-11-13T12:15:00+00:00",
                          "server_date": "Fri, 13 Nov 2026 11:00:00 GMT"}])
    assert fs["final"] and fs["basis"] == "local clock" and fs["server_gap_s"] == 23 * 3600


def test_make_check_fails_on_a_send_before_the_filter_wait_and_on_a_third_refusal(sandbox, clock):
    """Correction 4, C22: `make check` fails if a request is sent again before the wait after its first content-filter
    refusal, or is refused by a filter more than twice."""
    err = _http(403, _err(403, "e", error_type="refusal"), {"date": D0})
    run(FREE1, sandbox, FakeTransport(fail_at=1, error=err))
    assert K.verify(records=sandbox["records"]) == []
    k = [e["key"] for e in X._jsonl(K.JOURNAL) if e["kind"] == "filtered"][0]
    good_journal = K.JOURNAL.read_text()
    clock.set("Fri, 13 Nov 2026 12:00:00 GMT")                              # 24 hours: not yet 24 h 10 min
    X._append(K.JOURNAL, {"at": K.now(), "key": k, "kind": "send", "request_sha256": "x", "fields_dropped": []})
    assert any("less than 24 hours and 10 minutes after its first" in p for p in K.verify(records=sandbox["records"]))
    K.JOURNAL.write_text(good_journal)
    clock.set(LATER)
    for d in (LATER, "Sat, 14 Nov 2026 13:00:00 GMT"):
        X._append(K.JOURNAL, {"at": K.now(), "key": k, "kind": "send", "request_sha256": "x", "fields_dropped": []})
        X._append(K.JOURNAL, {"at": K.now(), "key": k, "kind": "filtered", "status": 403, "error_type": "refusal",
                              "server_date": d})
    problems = K.verify(records=sandbox["records"])
    assert any("refused by a content filter 3 times" in p for p in problems)
    assert any("after its content-filter refusal became final" in p for p in problems)


D1, D2, D3 = "Fri, 13 Nov 2026 12:00:00 GMT", "Sat, 14 Nov 2026 12:00:00 GMT", "Sun, 15 Nov 2026 12:00:00 GMT"


def _denied(date):
    return _http(403, _err(403, "Provider returned error", error_type="permission_denied", provider_name="Nvidia",
                           raw="User location is not supported"), {"date": date})


def test_a_persistent_forbidden_closes_the_model_only_after_3_runs_over_72_hours(sandbox, clock):
    """Correction 4, C23 (the fourth review's MAJOR-2, N1): the same 403 that is neither a content filter nor a budget,
    on the same request, on 3 runs whose server dates span at least 72 hours with no answer in between, lets
    `--close-incomplete` close the model, with the operator's record of what was checked and a recorded deviation;
    48 hours are not enough; `make check` accepts the closing and the analysis gives its count."""
    for d in (D0, D1, D2):
        clock.set(d)
        with pytest.raises(X.Refusal, match="infrastructure or permission"):
            run(FREE1, sandbox, FakeTransport(fail_at=range(1, 100), error=_denied(d)))
    with pytest.raises(X.Refusal, match="cannot be closed"):                 # three runs over 48 hours
        K.close_incomplete(FREE1, checked="key permissions, guardrails, network")
    clock.set(D3)
    with pytest.raises(X.Refusal, match="infrastructure or permission"):
        run(FREE1, sandbox, FakeTransport(fail_at=range(1, 100), error=_denied(D3)))
    with pytest.raises(X.Refusal, match="first record what was checked"):
        K.close_incomplete(FREE1)
    assert not K.state()["status"]
    why = K.close_incomplete(FREE1, checked="key permissions, no guardrail, network reaches openrouter.ai")
    assert why == ("not run: forbidden on 4 runs over 72 hours by the server's dates "
                   f"({FREE1}|v1|d1: HTTP 403 permission_denied): " + K.VERDICT_WORDS["forbidden"])
    dev = [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "deviation"]
    assert len(dev) == 1 and dev[0]["kind"] == "forbidden_closed" and dev[0]["checked"].startswith("key permissions")
    assert K.verify(records=sandbox["records"]) == []
    # the analysis: the model is closed, so a count is given, and it is listed as not counted with its reason
    from judge import cloud_analysis as CA
    closed = {e["model"]: e for e in X._jsonl(K.RUN_LOG) if e["event"] == "closed_incomplete"}
    status, legit = CA.model_status(FREE1, closed, 0, 600, X._jsonl(K.RUN_LOG))
    assert legit and status == why
    d = CA.deviations(FREE1, X._jsonl(K.JOURNAL), X._jsonl(K.RUN_LOG), [], [])
    out = {"order": [FREE1], "models": {FREE1: {"status": status, "closed": True, "compared": True, "cells": {},
                                                "row_note": None, "deviations": d}}}
    s = CA.summary(out)
    assert "not finished" not in s and f"Not counted: {FREE1} ({why})" in s
    assert "1 closing after the same 403, neither a content filter nor a budget, on 3 runs over at least 72 hours" in s
    # a closing on a persistent 403 that the files do not back fails `make check`
    log = X._jsonl(K.RUN_LOG)
    K.RUN_LOG.write_text("".join(json.dumps(e) + "\n" for e in log if e["event"] != "deviation"))
    assert any("closed on a persistent 403" in p for p in K.verify(records=sandbox["records"]))


def test_a_persistent_forbidden_series_needs_the_span_and_no_answer_between(sandbox, clock):
    """Correction 4, C23: three runs minutes apart, or an answer of the model in between, close nothing; exactly three
    runs over 72 hours suffice."""
    for d in (D0, "Thu, 12 Nov 2026 12:03:00 GMT", "Thu, 12 Nov 2026 12:06:00 GMT"):
        clock.set(d)
        with pytest.raises(X.Refusal, match="infrastructure or permission"):
            run(FREE1, sandbox, FakeTransport(fail_at=range(1, 100), error=_http(403, CF_HTML, {"date": d})))
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(FREE1, checked="x")
    clock.set(D1)
    with pytest.raises(X.Refusal, match="infrastructure or permission"):          # one answer, then the 403 again
        run(FREE1, sandbox, FakeTransport(fail_at=range(2, 100), error=_denied(D1)))
    clock.set(D3)
    with pytest.raises(X.Refusal, match="infrastructure or permission"):
        run(FREE1, sandbox, FakeTransport(fail_at=range(1, 100), error=_denied(D3)))
    with pytest.raises(X.Refusal, match="cannot be closed"):                       # 2 runs, 48 hours, since the answer
        K.close_incomplete(FREE1, checked="x")
    clock.set("Mon, 16 Nov 2026 12:00:00 GMT")
    with pytest.raises(X.Refusal, match="infrastructure or permission"):
        run(FREE1, sandbox, FakeTransport(fail_at=range(1, 100), error=_denied("Mon, 16 Nov 2026 12:00:00 GMT")))
    assert K.close_incomplete(FREE1, checked="x").startswith(
        "incomplete (1 of 10 answered): forbidden on 3 runs over 72 hours by the server's dates")
    assert K.verify(records=sandbox["records"]) == []


def test_huge_integers_are_not_key_readings():
    """Correction 4, C25 (the fourth review's m-2, N4): json.loads keeps a 400-digit number as an int, on which
    math.isfinite raises OverflowError; it is no reading, and nothing raises."""
    huge = json.loads("1" + "0" * 400)
    assert isinstance(huge, int) and not K._num(huge) and not K._num(-huge)
    base = {"limit": 15.0, "limit_reset": None, "limit_remaining": 15.0, "fields_missing": [], "budgets": [],
            "effective_budget": None}
    for f in ("limit", "limit_remaining"):
        assert K.paid_key_refusal(PAID1, {**base, f: huge}, 4.8) is not None
        assert not K.key_short({**base, f: huge}, 4.8)
    models, _, eps = S.load_snapshot()
    changed = {**eps, PAID1: [{**e, "max_completion_tokens": huge} if e["tag"] == K.ENDPOINT[PAID1]["tag"] else e
                              for e in eps[PAID1]]}
    assert "max_completion_tokens" in K.listing_check(PAID1, _getter(models, changed))[0]


def test_a_paid_model_refuses_a_reading_without_budgets(sandbox):
    """Correction 4, C25 (the fourth review's m-3, N5): without `budgets` (or with it not a list) a guardrail or
    workspace budget cannot be ruled out, so the paid arm refuses; an empty list or only `key_limit` passes."""
    base = {"limit": 15.0, "limit_reset": None, "limit_remaining": 15.0, "fields_missing": [], "budgets": [],
            "effective_budget": None}
    assert K.paid_key_refusal(PAID1, base, 4.8) is None
    assert K.paid_key_refusal(PAID1, {**base, "budgets": [{"scope": "key_limit"}],
                                      "effective_budget": {"scope": "key_limit"}}, 4.8) is None
    for bad in ({"fields_missing": ["budgets", "effective_budget"], "budgets": None},
                {"budgets": {"scope": "member"}}, {"effective_budget": [{"scope": "key_limit"}]}):
        why = K.paid_key_refusal(PAID1, {**base, **bad}, 4.8)
        assert why is not None and "budgets" in why, bad
        assert not K.key_short({**base, **bad}, 4.8)
    t = FakeTransport(omit=("budgets",))
    with pytest.raises(X.Refusal, match="did not report budgets"):
        run(PAID1, sandbox, t)
    assert t.calls == [] and not K.state()["status"]


# ------------------------------------------------------------------ Correction 5: the owner's waiver (C26) and C27
WAIVER_WORDS = {"quote_ru": "это лимит никакой, там и так 25 долларов, просто работаю аккуратно и все нормально делают",
                "quote_en": "this limit is pointless; there are only 25 dollars there anyway; I just work carefully, "
                            "and everyone does it normally",
                "date": "2026-11-12", "given_by": "the author"}
READ_OK = {"balance_usd": 25.0, "read_at": "2026-11-12T11:00:00Z", "read_by": "operator", "source": "dashboard"}
NO_LIMIT = {"limit": None, "limit_remaining": None}


def _waiver(monkeypatch, sandbox, readings=(READ_OK,), **over):
    """registration.json with the owner's waiver (Correction 5, C26), as the runner and `make check` read it."""
    reg = {**FILLED, "key_reading": {"limit": None, "limit_reset": None, "free_model_daily_requests": {"limit": 1000}},
           "key_limit_waived": True, "key_limit_waiver": dict(WAIVER_WORDS),
           "account_balance_readings": [dict(r) for r in readings], **over}
    reg = {k: v for k, v in reg.items() if v is not ...}
    monkeypatch.setattr(K, "preflight", lambda tag, published_head=None, getter=None: {
        "sets": sandbox["sets"], "test": {"h1"}, "registration": reg})
    K.REGISTRATION.write_text(json.dumps(reg))
    return reg


def _prefill(**usd):
    """Charges already in the spend journal, by model (a probe of the caps; no request)."""
    for tag, amount in usd.items():
        X._append(K.SPEND, {"at": K.now(), "event": "charged", "model": tag, "key": f"{tag}|v1|d1",
                            "cost_usd": amount, "running_total_usd": round(K.spend_state()["computed"] + amount, 6)})


P2, P3 = K.PAID_QUEUE[1], K.PAID_QUEUE[2]


def test_the_owners_waiver_lets_a_paid_model_run_on_a_key_without_a_limit_and_is_recorded(sandbox, clock, monkeypatch):
    """Correction 5, C26: with the waiver recorded in registration.json, a key whose reading reports `limit` null may
    run the paid arm, on a fresh balance reading; each such run is a recorded deviation that names the balance, the
    analysis counts and prints it, and the key-readings sentence says that nothing outside the code stopped the arm."""
    _waiver(monkeypatch, sandbox)
    t = FakeTransport(**NO_LIMIT)
    out = run(PAID1, sandbox, t)
    assert out["missing"] == 0 and len(t.calls) == 10 and K.state()["status"][PAID1] == "complete"
    log = X._jsonl(K.RUN_LOG)
    dev = [e for e in log if e["event"] == "deviation"]
    assert [d["kind"] for d in dev] == ["key_limit_waived"]
    assert dev[0]["balance_usd"] == 25.0 and dev[0]["balance_read_at"] == READ_OK["read_at"]
    assert dev[0]["required_balance_usd"] == 17.0 and dev[0]["waiver_date"] == "2026-11-12"
    assert [e["event"] for e in log].index("deviation") < [e["event"] for e in log].index("run_started")
    assert K.verify(records=sandbox["records"]) == []
    from judge import cloud_analysis as CA
    d = CA.deviations(PAID1, X._jsonl(K.JOURNAL), log, [], X._jsonl(K.CACHE_FILE))
    assert d["key_limit_waived"] == 1
    s = CA.deviations_sentence({"order": [PAID1], "models": {PAID1: {"deviations": d}}})
    assert "1 paid run on a key without a credit limit, under the owner's waiver" in s and "account's balance" in s
    acc = CA.account(log)
    assert acc["first"]["limit"] is None and acc["waiver"]["runs"] == 1 and acc["waiver"]["first_balance_usd"] == 25
    out = {"order": [PAID1], "account": acc, "models": {PAID1: {
        "status": "not run: x", "closed": True, "compared": True, "cells": {}, "row_note": None, "deviations": d}}}
    s = CA.summary(out)
    assert "had no credit limit in OpenRouter, by the owner's waiver of 2026-11-12" in s
    assert "nothing outside the code stopped the paid arm at 15 USD" in s and "only outer bound" in s


def test_without_a_complete_waiver_a_key_without_a_limit_is_still_refused(sandbox, clock, monkeypatch):
    """Correction 5, C26: no waiver, a waiver not exactly true, or one without the owner's words or a date is no
    waiver; the paid arm then refuses a key without a limit, sends nothing and closes nothing."""
    w = dict(WAIVER_WORDS)
    for over, why in (({"key_limit_waived": ...}, "no credit limit"), ({"key_limit_waived": False}, "no credit limit"),
                      ({"key_limit_waived": "true"}, "no credit limit"),
                      ({"key_limit_waiver": None}, "records no key_limit_waiver"),
                      ({"key_limit_waiver": {**w, "quote_ru": "  "}}, "lacks quote_ru"),
                      ({"key_limit_waiver": {k: v for k, v in w.items() if k != "quote_en"}}, "lacks quote_en"),
                      ({"key_limit_waiver": {**w, "date": "12.11.2026"}}, r"date \(YYYY-MM-DD\)")):
        reg = _waiver(monkeypatch, sandbox, **over)
        assert K.key_limit_waiver(reg)[0] is False
        t = FakeTransport(**NO_LIMIT)
        with pytest.raises(X.Refusal, match=why):
            run(PAID1, sandbox, t)
        assert t.calls == [] and X._jsonl(K.JOURNAL) == [] and not K.state()["status"]
    assert K.key_limit_waiver({**FILLED, "key_limit_waived": True, "key_limit_waiver": w}) == (True, None)
    assert K.key_limit_waiver(FILLED) == (False, None)
    assert not [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "deviation"]


def test_the_waiver_never_admits_a_key_whose_limit_is_not_exactly_15_or_resets(sandbox, clock, monkeypatch):
    """Correction 5, C26: the waiver covers only a key with no limit. A key that has a limit must still have exactly
    15 USD that never resets (10 USD with the waiver is refused); a missing field, a reset or another budget is still
    refused; a key limited to exactly 15 USD runs as before, with no balance reading and no waiver deviation."""
    _waiver(monkeypatch, sandbox)
    for kw, why in (({"limit": 10.0, "limit_remaining": 10.0}, "not exactly the branch cap"),
                    ({"limit": 50.0, "limit_remaining": 50.0}, "not exactly the branch cap"),
                    ({"limit": float("nan"), "limit_remaining": 15.0}, "not exactly the branch cap"),
                    ({"limit_reset": "monthly"}, "resets"),
                    ({**NO_LIMIT, "limit_reset": "monthly"}, "resets"),
                    ({**NO_LIMIT, "omit": ("limit",)}, "did not report limit"),
                    ({**NO_LIMIT, "omit": ("budgets",)}, "did not report budgets")):
        t = FakeTransport(**kw)
        with pytest.raises(X.Refusal, match=why):
            run(PAID1, sandbox, t)
        assert t.calls == [] and X._jsonl(K.JOURNAL) == [] and not K.state()["status"]
    before = {**NO_LIMIT, "limit_reset": None, "fields_missing": [], "budgets": [{"scope": "member"}],
              "effective_budget": None}
    assert "other than the key's own limit" in K.paid_key_refusal(PAID1, before, 4.8, waived=True)
    assert not [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "deviation"]
    _waiver(monkeypatch, sandbox, readings=())               # exactly 15 USD: the registered guard, no reading needed
    t = FakeTransport()
    assert run(PAID1, sandbox, t)["missing"] == 0 and len(t.calls) == 10
    assert not [e for e in X._jsonl(K.RUN_LOG) if e["event"] == "deviation"]
    assert K.verify(records=sandbox["records"]) == []


def test_under_the_waiver_each_paid_run_needs_a_fresh_balance_reading_that_covers_the_cap(sandbox, clock,
                                                                                          monkeypatch):
    """Correction 5, C26: the code cannot read the account's balance (GET /api/v1/credits needs a management key), so
    under the waiver the paid run refuses unless registration.json holds a reading taken by hand: the field present
    and a finite number, an ISO time with its offset, not in the future, at most 24 hours old, later than the last
    paid send, and at least the branch's remaining room under 15 USD plus 2 USD. A refusal sends and closes nothing."""
    cases = (((), "no reading of the account's credit balance"),
             (({"read_at": "2026-11-12T11:00:00Z"},), "no balance_usd as a finite number"),
             (({**READ_OK, "balance_usd": "25"},), "no balance_usd as a finite number"),
             (({**READ_OK, "balance_usd": float("inf")},), "no balance_usd as a finite number"),
             (({**READ_OK, "read_at": "2026-11-12T11:00:00"},), "with its offset"),
             (({**READ_OK, "read_at": "2026-11-12T12:30:00Z"},), "later than this machine's clock"),
             (({**READ_OK, "read_at": "2026-11-11T11:59:00Z"},), "more than 24 hours old"),
             (({**READ_OK, "balance_usd": 16.99},), "less than the branch's remaining room"))
    for readings, why in cases:
        _waiver(monkeypatch, sandbox, readings=readings)
        t = FakeTransport(**NO_LIMIT)
        with pytest.raises(X.Refusal, match=why):
            run(PAID1, sandbox, t)
        assert t.calls == [] and X._jsonl(K.JOURNAL) == [] and not K.state()["status"]
    _waiver(monkeypatch, sandbox, account_balance_readings=...)          # the field itself missing
    with pytest.raises(X.Refusal, match="no reading of the account's credit balance"):
        run(PAID1, sandbox, FakeTransport(**NO_LIMIT))
    _waiver(monkeypatch, sandbox, readings=({**READ_OK, "balance_usd": 17.0},))  # exactly 15 + 2: enough
    assert run(PAID1, sandbox, FakeTransport(**NO_LIMIT))["missing"] == 0
    t = FakeTransport(**NO_LIMIT)
    with pytest.raises(X.Refusal, match="not later than the branch's last paid send"):
        run(P2, sandbox, t)                               # the same reading again, after a paid run: refused
    assert t.calls == [] and P2 not in K.state()["status"]
    clock.set("Thu, 12 Nov 2026 13:00:00 GMT")
    _waiver(monkeypatch, sandbox, readings=(READ_OK, {**READ_OK, "read_at": "2026-11-12T12:59:00Z"}))
    t = FakeTransport(**NO_LIMIT)
    assert run(P2, sandbox, t)["missing"] == 0 and len(t.calls) == 10
    assert K.verify(records=sandbox["records"]) == []


def test_under_the_waiver_the_in_code_caps_still_stop_a_probe_beyond_15_usd(sandbox, clock, monkeypatch):
    """Correction 5, C26: without a key limit the caps in code are the bound. A paid model whose next request's worst
    case and the overrun room do not fit under the 15 USD branch cap is stopped before sending; one answer billed far
    above its reservation (which only the provider can do) closes the model, and no paid request is sent after the
    branch is over its cap: the whole excess is that one answer, and only the account's balance bounds it."""
    _waiver(monkeypatch, sandbox)
    _prefill(**{P2: 5.0, P3: 5.0, PAID1: 3.83})          # 13.83: the room left (1.17) is less than reserve + overrun room
    t = FakeTransport(**NO_LIMIT)
    with pytest.raises(X.Refusal, match="15 USD|cap"):
        run(PAID1, sandbox, t)
    assert t.calls == [] and K.state()["status"][PAID1].startswith("not run: ")
    assert K.spend_state()["committed"] <= K.CAP_USD + 1e-9

    for name in ("SPEND", "RUN_LOG", "JOURNAL", "CACHE_FILE"):
        getattr(K, name).unlink(missing_ok=True)
    _prefill(**{P2: 5.0, P3: 4.5})                       # 9.5 spent by the others
    monkeypatch.setattr(K, "projected", lambda tag, b: 0.0)
    t = FakeTransport(**NO_LIMIT, usage={"prompt_tokens": 700, "completion_tokens": 40, "cost": 6.0})
    with pytest.raises(X.Refusal, match="cost above the reservation"):    # a provider bills 6 USD for one answer
        run(PAID1, sandbox, t)
    assert len(t.calls) == 1 and K.state()["status"][PAID1].startswith("incomplete (1 of 10 answered)")
    assert K.spend_state()["computed"] == pytest.approx(15.5)              # over the cap by that one answer only
    clock.set("Thu, 12 Nov 2026 13:00:00 GMT")
    _waiver(monkeypatch, sandbox, readings=(READ_OK, {**READ_OK, "read_at": "2026-11-12T12:59:00Z"}))
    monkeypatch.setattr(K, "projected", lambda tag, b: 0.0001)
    for tag in (P2, P3):                                  # nothing more is sent: the branch is over its cap
        t = FakeTransport(**NO_LIMIT)
        with pytest.raises(X.Refusal):
            run(tag, sandbox, t)
        assert t.calls == [] and K.state()["status"][tag].startswith("not run: ")
    assert K.spend_state()["computed"] == pytest.approx(15.5)


def test_make_check_fails_on_a_paid_run_without_the_limit_or_the_waivers_record(sandbox, clock, monkeypatch):
    """Correction 5, C26: `make check` accepts a paid run on a key whose limit is not 15 USD only with the waiver's
    deviation after its key reading, the waiver complete in registration.json and the named balance reading there."""
    reg = _waiver(monkeypatch, sandbox)
    run(PAID1, sandbox, FakeTransport(**NO_LIMIT))
    assert K.verify(records=sandbox["records"]) == []
    K.REGISTRATION.write_text(json.dumps({**reg, "account_balance_readings": []}))
    assert any("balance reading" in p and "does not hold" in p for p in K.verify(records=sandbox["records"]))
    K.REGISTRATION.write_text(json.dumps({**reg, "key_limit_waived": False}))
    assert any("does not record completely" in p for p in K.verify(records=sandbox["records"]))
    K.REGISTRATION.write_text(json.dumps(reg))
    log = X._jsonl(K.RUN_LOG)
    K.RUN_LOG.write_text("".join(json.dumps(e) + "\n" for e in log if e["event"] != "deviation"))
    assert any("without the owner's waiver recorded for it" in p for p in K.verify(records=sandbox["records"]))


def test_under_the_waiver_key_use_is_read_from_usage_and_stops_nothing(sandbox, clock, monkeypatch):
    """Correction 5, C26 with C25: a key without a limit has no limit_remaining; its all-time `usage` beyond the
    branch's journal is a recorded deviation that stops nothing (a key that is not dedicated shows other use here)."""
    _waiver(monkeypatch, sandbox)

    class Used(FakeTransport):
        def get(self, url):
            r = super().get(url)
            r["data"]["usage"] = 3.0
            return r
    t = Used(**NO_LIMIT)
    assert run(PAID1, sandbox, t)["missing"] == 0 and len(t.calls) == 10
    dev = {e["kind"]: e for e in X._jsonl(K.RUN_LOG) if e["event"] == "deviation"}
    assert set(dev) == {"key_use_exceeds_journal", "key_limit_waived"}
    assert dev["key_use_exceeds_journal"]["key_used_usd"] == 3.0 and dev["key_use_exceeds_journal"]["stopped"] is False
    assert K.key_use({"limit": None, "usage": 3.0}, 0.0) == {"key_used_usd": None, "journaled_usd": 0.0,
                                                             "beyond_journal_usd": None}   # no waiver: as before


def test_the_committed_registration_records_the_owners_waiver_with_no_balance_yet():
    """Correction 5, C26: the owner's decision of 2026-11-12 is in the published record word for word; the balance
    readings are left to the operator (one before each paid run)."""
    reg = json.loads((C.RESULTS / "models" / "cloud" / "registration.json").read_text())
    assert K.key_limit_waiver(reg) == (True, None)
    assert reg["key_limit_waiver"]["quote_ru"] == WAIVER_WORDS["quote_ru"] and reg["key_limit_waiver"]["date"] == \
        "2026-11-12"
    assert isinstance(reg["account_balance_readings"], list)


def test_one_undated_refusal_does_not_move_a_dated_forbidden_series_to_the_local_clock(sandbox, clock):
    """Correction 5, C27 (the fifth review's m-1, B6): three dated runs over 48 hours and a fourth without a server
    date 80 hours after the first by the local clock do not close the model; with fewer than three dated refusals
    the local times decide, and the record says so."""
    for d in (D0, D1, D2):
        clock.set(d)
        with pytest.raises(X.Refusal, match="infrastructure or permission"):
            run(FREE1, sandbox, FakeTransport(fail_at=range(1, 100), error=_denied(d)))
    clock.set(D2, plus_s=32 * 3600)
    undated = K.CloudError(403, "Provider returned error", etype="permission_denied")
    with pytest.raises(X.Refusal, match="infrastructure or permission"):
        run(FREE1, sandbox, FakeTransport(fail_at=range(1, 100), error=undated))
    with pytest.raises(X.Refusal, match="cannot be closed"):
        K.close_incomplete(FREE1, checked="x")
    fb = K.forbidden_series(FREE1, X._jsonl(K.RUN_LOG), [
        {**e, "server_date": None} if n else e for n, e in enumerate(X._jsonl(K.JOURNAL))
        if e.get("kind") == "forbidden"])
    assert fb is not None and fb["date_basis"] == "local clock" and fb["span_hours"] == 80


def test_make_check_reads_what_a_forbidden_closing_says_it_checked(sandbox, clock):
    """Correction 5, C27 (the fifth review's m-2, B5): a `forbidden_closed` deviation without `checked`, or naming
    another request, status or error type than the series, fails `make check`."""
    for d in (D0, D1, D3):
        clock.set(d)
        with pytest.raises(X.Refusal, match="infrastructure or permission"):
            run(FREE1, sandbox, FakeTransport(fail_at=range(1, 100), error=_denied(d)))
    K.close_incomplete(FREE1, checked="key permissions, guardrails, network")
    assert K.verify(records=sandbox["records"]) == []
    good = X._jsonl(K.RUN_LOG)
    for change in ({"checked": "   "}, {"checked": None}, {"key": f"{FREE1}|v2|d1"}, {"status": 401},
                   {"error_type": "x"}):
        K.RUN_LOG.write_text("".join(json.dumps({**e, **change} if e.get("kind") == "forbidden_closed" else e) + "\n"
                                     for e in good))
        assert any("closed on a persistent 403" in p for p in K.verify(records=sandbox["records"])), change


def test_the_report_gives_the_server_gap_and_both_causes_of_key_use(sandbox):
    """Correction 5, C27 (the fifth review's m-3 and m-6): a refusal made final by the local times is printed with the
    server's gap (23.99 hours and 1 hour no longer read alike), and the key-use deviation names both causes."""
    from judge import cloud_analysis as CA
    k1, k2, k3 = f"{FREE1}|v1|d1", f"{FREE1}|v1|d2", f"{FREE1}|v1|d3"
    journal = [{"at": "2026-11-12T12:00:00Z", "key": k1, "kind": "filtered", "server_date": D0},
               {"at": "2026-11-13T12:10:00Z", "key": k1, "kind": "filtered",
                "server_date": "Fri, 13 Nov 2026 11:59:24 GMT"},
               {"at": "2026-11-12T12:00:00Z", "key": k2, "kind": "filtered", "server_date": D0},
               {"at": "2026-11-13T12:10:00Z", "key": k2, "kind": "filtered", "server_date": "Thu, 12 Nov 2026 13:00:00 GMT"},
               {"at": "2026-11-12T12:00:00Z", "key": k3, "kind": "filtered"},
               {"at": "2026-11-13T12:10:00Z", "key": k3, "kind": "filtered"}]
    log = [{"event": "deviation", "kind": "key_use_exceeds_journal", "model": FREE1}]
    d = CA.deviations(FREE1, journal, log, [], [])
    assert d["filtered_final_local"] == 3 and d["filtered_final_local_server_gaps_h"] == [1.0, 23.99]
    s = CA.deviations_sentence({"order": [FREE1], "models": {FREE1: {"deviations": d}}})
    assert "(the server's dates 1, 23.99 hours apart; 1 without a server date)" in s
    assert "(spend outside the branch, or charges the journal does not hold)" in s
