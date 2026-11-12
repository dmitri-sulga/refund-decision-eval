"""The hosted arm (docs/analysis_plan_models.md, Amendment 2). Every test uses a fake transport or a mock server on
127.0.0.1; nothing here leaves the machine, and no test reads a real API key. The tests pass whether the arm's
registration is still empty or already filled."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from judge import config as C
from judge import freeze, llm
from judge import hosted as H
from judge import models as X
from judge.judgments import parse
from judge.prompts import build_messages

FAKE_KEY = "sk-" + "ant-api03-" + "FAKEKEYFORTESTS" * 4 + "-q9Z"
OTHER_KEY = "sk-" + "ant-api03-" + "OTHERKEY" * 3
FILLED = {"repository": "o/r", "code_commit": "c" * 40, "code_published_at": "2000-01-01T00:00:00Z"}


# ------------------------------------------------------------------ the request
def test_the_request_carries_the_frozen_messages_and_schema_and_only_the_registered_settings():
    frozen = freeze.load()["request_templates"]
    for tag in H.MODELS:
        for v in X.VERSIONS:
            b = H.build_body(tag, v, freeze._PROBE, "narrative")
            messages, schema = build_messages(v, freeze._PROBE, "narrative")
            assert b["system"] == messages[0]["content"] and b["messages"] == [messages[1]]
            sent = b["output_config"]["format"]["schema"]
            assert {k: x for k, x in sent.items() if k != "additionalProperties"} == schema
            assert sent["additionalProperties"] is False and "additionalProperties" not in schema
            back = llm.request_body([{"role": "system", "content": b["system"]}, *b["messages"]],
                                    {k: x for k, x in sent.items() if k != "additionalProperties"})
            assert llm.request_hash(back) == frozen[v]
            assert set(b) == {"model", "max_tokens", "system", "messages", "output_config", "inference_geo"} | (
                {"thinking"} if H.THINKING[tag] else set())
            assert b["output_config"]["effort"] == "low" and b["model"] == tag and b["inference_geo"] == "global"
            assert not {"temperature", "top_p", "top_k", "fallbacks", "speed", "service_tier", "stop_sequences",
                        "tools", "stream"} & set(b)
    probe = lambda t: H.build_body(t, "v1", freeze._PROBE, "n")   # noqa: E731
    assert probe("claude-haiku-5-5")["thinking"] == {"type": "disabled"}
    assert probe("claude-sonnet-5-5")["thinking"] == {"type": "between_tools"}
    assert "thinking" not in probe("claude-opus-5-5")
    assert [probe(t)["max_tokens"] for t in H.MODELS] == [512, 512, 4096]
    import inspect
    assert list(inspect.signature(H.build_body).parameters) == ["tag", "version", "rec", "narrative"]
    assert H.API_VERSION == "2026-10-21"


def test_the_model_list_comparison_set_prices_and_cap_are_the_registered_ones():
    assert H.MODELS == ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")
    assert H.COMPARED == ("claude-haiku-5-5", "claude-sonnet-5-5") and H.REFERENCE_ONLY == ("claude-opus-5-5",)
    assert {t: (p["input"], p["output"]) for t, p in H.PRICES.items()} == {
        "claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}
    assert H.CAP_USD == 15.0 and H.RETRY_DELAYS_S == (30, 60, 120) and H.MAX_ERRORS_PER_MODEL == 60
    for gone in ("DROPPABLE", "REQUEST_FIELDS", "ENVIRONMENT_FIELDS", "_names"):     # no word matching, no drop
        assert not hasattr(H, gone)
    plan = X.PLAN_FILE.read_text()
    a2 = plan.split("## Amendment 2")[1]
    assert plan.index("## Amendment 1") < plan.index("## Amendment 2")
    for t in H.MODELS:
        assert f"`{t}`" in a2
    assert "Models whose reasoning cannot be switched off" in a2 and "its reasoning cannot be switched off" in a2
    assert "4 comparisons" in a2 and "98.75%" in a2 and "97.5%" in a2
    assert "15 − 10.84 = 4.16 USD" in a2 and "not run: projection exceeded remaining budget" in a2
    assert "This project does not rely on a spend limit set in the provider's Console." in a2
    assert X.registered_part_sha256() == X.REGISTERED_SHA256        # lines 1-149 untouched


# ------------------------------------------------------------------ the registration, in both states
def test_the_registration_refuses_while_empty_and_is_read_once_filled(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "REGISTRATION", tmp_path / "registration.json")
    with pytest.raises(X.Refusal, match="missing"):
        H.load_registration()
    H.REGISTRATION.write_text(json.dumps({**FILLED, "code_commit": None, "code_published_at": None}))
    with pytest.raises(X.Refusal, match="lacks"):
        H.load_registration()
    H.REGISTRATION.write_text(json.dumps(FILLED))
    assert H.load_registration()["code_commit"] == "c" * 40


def test_the_committed_registration_is_either_waiting_or_complete():
    reg = json.loads((C.RESULTS / "models" / "hosted" / "registration.json").read_text())
    assert reg["requests_before_this_file"] == 0 and reg["models"] == list(H.MODELS)
    fields = ("code_commit", "code_published_at")
    if all(reg[k] is None for k in fields):
        with pytest.raises(X.Refusal, match="lacks"):
            H.load_registration()
    else:
        got = H.load_registration()
        assert re.fullmatch(r"[0-9a-f]{40}", got["code_commit"]) and got["code_published_at"]


# ------------------------------------------------------------------ parsing parity with the local path
def _local_like(content, done_reason="stop", prompt_eval_count=700):
    return {"content": content, "done_reason": done_reason, "prompt_eval_count": prompt_eval_count,
            "eval_count": 40, "wall_s": 1.0, "logprobs": None}


def _resp(content, stop="end_turn", usage=None, thinking=False, model="claude-haiku-5-5", geo="global"):
    blocks = ([{"type": "thinking", "thinking": "", "signature": "s"}] if thinking else []) + \
             ([{"type": "text", "text": content}] if content is not None else [])
    return {"id": "msg_x", "model": model, "content": blocks, "stop_reason": stop, "stop_details": None,
            "usage": usage or {"input_tokens": 700, "output_tokens": 40, "cache_creation_input_tokens": 0,
                               "cache_read_input_tokens": 0, "inference_geo": geo}}


def _hosted_record(resp, tag="claude-haiku-5-5"):
    body = H.build_body(tag, "v1", freeze._PROBE, "n")
    return H.answer_record(tag, "v1", "d1", body, H.request_hash(body), resp, {"request-id": "req_x"}, X.now(), 1.0)


@pytest.mark.parametrize("content", [
    '{"monetary_relief": true, "p_relief": 0.8, "reason": "x"}',
    '{"monetary_relief": false, "p_relief": 0.1, "reason": "x"}',
    '{"monetary_relief": "yes", "p_relief": 0.1, "reason": "x"}',
    'not json', '', '[true]',
])
def test_a_hosted_answer_parses_exactly_as_a_local_answer_with_the_same_text(content):
    from judge import models_analysis as A
    h = _hosted_record(_resp(content, thinking=True))
    loc = _local_like(content)
    ph, pl = parse(h), parse(loc)
    assert {k: ph[k] for k in ("valid", "pred", "fields")} == {k: pl[k] for k in ("valid", "pred", "fields")}
    assert ph["p_verbal"] == pl["p_verbal"] or (np.isnan(ph["p_verbal"]) and np.isnan(pl["p_verbal"]))
    assert A.malformed(h, ph) == A.malformed(loc, pl)
    assert h["content"] == content and h["done_reason"] == "stop" and h["prompt_eval_count"] == 700


def test_refusals_truncations_and_other_stops_are_malformed_and_scored_as_no_money_back():
    from judge import models_analysis as A
    good = '{"monetary_relief": true, "p_relief": 0.8, "reason": "x"}'
    for stop, done in (("refusal", "refusal"), ("max_tokens", "length"),
                       ("model_context_window_exceeded", "model_context_window_exceeded")):
        h = _hosted_record(_resp(good, stop=stop))
        assert h["done_reason"] == done and A.malformed(h, parse(h))
    empty = _hosted_record(_resp(None, stop="refusal"))
    assert parse(empty)["pred"] == 0 and not parse(empty)["valid"] and empty["content"] == ""
    usage = {"input_tokens": 7000, "output_tokens": 40, "cache_read_input_tokens": 700}
    assert A.malformed(_hosted_record(_resp(good, usage=usage)), parse({"content": good}))   # 7,700 > 7,680


# ------------------------------------------------------------------ a sandbox with a fake transport
class FakeTransport:
    def __init__(self, fail_at=(), error=None, usage=None, model=None, geo="global"):
        self.calls, self.error, self.usage, self.model, self.geo = [], error, usage, model, geo
        self.fail_at = {fail_at} if isinstance(fail_at, int) else set(fail_at)

    def send(self, body):
        self.calls.append(body)
        if len(self.calls) in self.fail_at:
            raise self.error
        content = json.dumps({"monetary_relief": len(self.calls) % 3 == 0, "p_relief": 0.5, "reason": "x"})
        return (_resp(content, usage=self.usage, model=self.model or body["model"], geo=self.geo),
                {"request-id": f"req_{len(self.calls)}"})


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "DIR", tmp_path)
    for name in ("CACHE_FILE", "JOURNAL", "RUN_LOG", "SPEND", "REGISTRATION", "LOCK"):
        monkeypatch.setattr(H, name, tmp_path / getattr(H, name).name)
    sets = {"dev": ["d1", "d2", "d3"], "labelled": ["l1", "l2"]}
    records = {i: {"product": "P", "sub_product": "S", "issue": "I", "sub_issue": "J", "narrative": f"text {i}",
                   "label": 0} for i in ["d1", "d2", "d3", "l1", "l2", "h1"]}
    monkeypatch.setattr(H, "preflight", lambda tag, published_head=None: {
        "sets": sets, "test": {"h1"}, "registration": {}})
    real_item_sets = X.item_sets
    monkeypatch.setattr(X, "item_sets", lambda: (sets, {"h1"}))
    H.REGISTRATION.write_text(json.dumps(FILLED))
    return {"tmp": tmp_path, "records": records, "sets": sets, "real_item_sets": real_item_sets}


def run(tag, sandbox, transport, **kw):
    return H.run_model(tag, transport, records=sandbox["records"], sleep=lambda s: None, **kw)


def events():
    return [e["event"] for e in X._jsonl(H.RUN_LOG)]


def test_a_full_model_journals_records_spend_and_verifies(sandbox):
    t = FakeTransport()
    out = run("claude-haiku-5-5", sandbox, t)
    assert out["missing"] == 0 and out["new_answers"] == 10 and len(t.calls) == 10
    journal, cache = X._jsonl(H.JOURNAL), X._jsonl(H.CACHE_FILE)
    assert [e["key"] for e in journal] == [r["key"] for r in cache]
    spend = X._jsonl(H.SPEND)
    answers = [e for e in spend if e["event"] == "answer"]
    assert len(answers) == 10 and all(e["input_tokens"] == 700 and e["inference_geo"] == "global" for e in answers)
    assert np.isclose(answers[-1]["running_total_usd"], 10 * (700 * 0.10 + 40 * 0.50) / 1e6)
    assert spend[0]["event"] == "estimate" and spend[-1]["event"] == "model_total" and spend[-1]["answers"] == 10
    assert H.verify(records=sandbox["records"]) == [] and not H.LOCK.exists()
    with open(H.SPEND, "a") as f:                     # a spend line that does not match its usage
        f.write(json.dumps({**answers[0], "cost_usd": 1.0}) + "\n")
    assert any("running total" in p for p in H.verify(records=sandbox["records"]))


# ------------------------------------------------------------------ held-out complaints can never be sent
def test_an_injected_held_out_id_sends_journals_and_spends_nothing(sandbox, monkeypatch):
    real = H.work_list

    def with_held_out(tag, sets, records):
        return [("v1", "dev", "h1", sandbox["records"]["h1"], "text h1"), *real(tag, sets, records)]
    monkeypatch.setattr(H, "work_list", with_held_out)
    t = FakeTransport()
    with pytest.raises(X.Refusal, match="h1"):
        run("claude-haiku-5-5", sandbox, t)
    assert t.calls == [] and X._jsonl(H.JOURNAL) == [] and X._jsonl(H.CACHE_FILE) == [] and H.spent() == 0


def test_the_allowed_sets_come_only_from_dev_and_the_labelled_sample():
    from judge.data import load_splits
    sets, test = X.item_sets()
    assert set(sets) == {"dev", "labelled"} and test == set(load_splits()["test"]) and len(test) == 250
    assert not test & (set(sets["dev"]) | set(sets["labelled"]))
    assert {s for _, s in X.CELLS} == {"dev", "labelled"}


def test_verify_catches_a_held_out_journal_entry(sandbox):
    run("claude-haiku-5-5", sandbox, FakeTransport())
    X._append(H.JOURNAL, {"at": X.now(), "key": "claude-haiku-5-5|v1|h1", "kind": "send", "request_sha256": "x"})
    assert any("not a DEV" in p for p in H.verify(records=sandbox["records"]))


# ------------------------------------------------------------------ the budget cap
def test_the_cap_stops_before_a_request_that_could_take_the_total_over_it(sandbox, monkeypatch):
    body = H.build_body("claude-opus-5-5", "v1", sandbox["records"]["d1"], "text d1")
    per_call = H.cost("claude-opus-5-5", {"input_tokens": 700, "output_tokens": 300})
    reserve = H.worst_case("claude-opus-5-5", body)
    assert reserve > per_call
    cap = 3 * per_call + reserve * 0.999             # three calls fit; the fourth's worst case does not
    monkeypatch.setattr(H, "CAP_USD", cap)
    monkeypatch.setattr(H, "projected", lambda tag, b: 0.0)       # let the projection pass, test the guard
    t = FakeTransport(usage={"input_tokens": 700, "output_tokens": 300}, model="claude-opus-5-5")
    with pytest.raises(X.Refusal, match="budget"):
        run("claude-opus-5-5", sandbox, t)
    assert len(t.calls) == 3 and H.spent() <= cap and np.isclose(H.spent(), 3 * per_call)
    assert events()[-2:] == ["budget_stop", "closed_incomplete"]
    assert X._jsonl(H.SPEND)[-1]["event"] == "model_total" and X._jsonl(H.SPEND)[-1]["answers"] == 3


def test_the_projection_refuses_a_model_that_cannot_fit_and_sends_nothing(sandbox, monkeypatch):
    monkeypatch.setattr(H, "CAP_USD", 0.001)
    t = FakeTransport()
    with pytest.raises(X.Refusal, match="exceeds the cap"):
        run("claude-opus-5-5", sandbox, t)
    assert t.calls == [] and X._jsonl(H.JOURNAL) == [] and "budget_stop" in events()


def test_an_answer_whose_input_exceeds_the_bound_closes_the_model(sandbox):
    t = FakeTransport(usage={"input_tokens": 10 ** 6, "output_tokens": 1})
    with pytest.raises(X.Refusal, match="above the bound"):
        run("claude-haiku-5-5", sandbox, t)
    assert len(t.calls) == 1 and H.state()["status"]["claude-haiku-5-5"].startswith("incomplete (1 of 10 answered): ")


# ------------------------------------------------------------------ interruptions (Amendment 2, section 9)
E = H.HostedError


@pytest.mark.parametrize("err, verdict", [
    (E(400, "invalid_request_error: thinking.type: disabled is not supported", etype="invalid_request_error"),
     "rejected"),
    (E(400, "invalid_request_error: output_config.format.schema: bad", etype="invalid_request_error"), "rejected"),
    (E(400, "invalid_request_error: Inference geo 'global' is not allowed for this model", etype="invalid_request_error"),
     "rejected"),
    (E(400, "You have reached your specified API usage limits", etype="invalid_request_error"), "account"),
    (E(400, "Your credit balance is too low to access the Anthropic API.", etype="invalid_request_error"), "account"),
    (E(401, "invalid x-api-key", etype="authentication_error"), "account"),
    (E(402, "payment", etype="billing_error"), "account"),
    (E(403, "permission_error: Your API key does not have access to model: claude-haiku-5-5",
       etype="permission_error"), "rejected"),
    (E(404, "not_found_error: model: claude-haiku-5-5", etype="not_found_error"), "rejected"),
    (E(302, "redirect"), "rejected"),
    (E(429, "rate_limit_error: model", etype="rate_limit_error"), "transient"),
    (E(429, "rate_limit_error: spend cap", etype="rate_limit_error", code="enforced_spend_limit_reached"), "account"),
    (E(529, "overloaded_error", etype="overloaded_error"), "transient"),
    (E(500, "api_error: model", etype="api_error"), "transient"),
    (E(None, "no answer: TimeoutError", maybe_billed=True), "lost"),
    (E(200, "unreadable answer", maybe_billed=True), "lost"),
    (E(None, "ANTHROPIC_API_KEY is not set"), "account"),
])
def test_classify(err, verdict):
    assert H.classify(err)[0] == verdict


@pytest.mark.parametrize("status, etype, message", [
    (400, "invalid_request_error", "messages: request is temporarily blocked for this model"),
    (400, "invalid_request_error", "thinking.type: disabled is not supported"),
    (403, "permission_error", "Your API key does not have access to model: claude-haiku-5-5"),
    (404, "not_found_error", "model: claude-haiku-5-5"),
    (401, "authentication_error", "invalid x-api-key"),
    (400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API."),
])
def test_any_4xx_on_any_request_stops_the_run_drops_nothing_and_changes_no_setting(sandbox, status, etype, message):
    t = FakeTransport(fail_at=1, error=E(status, f"{etype}: {message}", etype=etype))
    with pytest.raises(X.Refusal, match="not served"):
        run("claude-haiku-5-5", sandbox, t)
    assert events() == ["run_started", "run_stopped"] and "claude-haiku-5-5" not in H.state()["status"]
    rej = [e for e in X._jsonl(H.JOURNAL) if e["kind"] in ("rejected", "account")]
    assert rej[0]["status"] == status and rej[0]["error_type"] == etype and H.spent() == 0
    t2 = FakeTransport()                                         # resumed without --resend-unanswered, same request
    assert run("claude-haiku-5-5", sandbox, t2)["missing"] == 0 and t2.calls[0] == t.calls[0]


class Clock:
    """H.now, moved by hand, and the server's `date` header at the same moment (the 24-hour rule of section 9 reads
    the server's dates)."""
    def __init__(self):
        from datetime import datetime, timezone
        self.t = datetime(2026, 12, 1, tzinfo=timezone.utc)

    def __call__(self):
        from datetime import timedelta
        self.t += timedelta(microseconds=1)
        return self.t.isoformat(timespec="microseconds")

    def hours(self, h):
        from datetime import timedelta
        self.t += timedelta(hours=h)

    def http(self):
        from email.utils import format_datetime
        return format_datetime(self.t, usegmt=True)


def test_a_rejection_closes_a_model_only_when_repeated_on_runs_24_hours_apart(sandbox, monkeypatch):
    from judge import hosted_analysis as HA
    clock = Clock()
    monkeypatch.setattr(H, "now", clock)
    def rejection():
        return E(400, "thinking: not supported", etype="invalid_request_error", date=clock.http())

    def stop(error):
        with pytest.raises(X.Refusal, match="the request was rejected"):
            run("claude-haiku-5-5", sandbox, FakeTransport(fail_at=1, error=error))
    stop(rejection())
    with pytest.raises(X.Refusal, match="not with the same status and error type on runs at least 24 hours apart"):
        H.close_incomplete("claude-haiku-5-5")                     # the operator's choice: refused
    clock.hours(23)
    stop(rejection())
    with pytest.raises(X.Refusal, match="24 hours apart"):
        H.close_incomplete("claude-haiku-5-5")                     # 23 hours: still refused
    clock.hours(2)
    stop(E(404, "model: claude-haiku-5-5", etype="not_found_error", date=clock.http()))
    with pytest.raises(X.Refusal, match="24 hours apart"):
        H.close_incomplete("claude-haiku-5-5")                     # another status: not the same rejection
    stop(rejection())                                              # 25 hours after the first: closed
    reason = H.close_incomplete("claude-haiku-5-5")
    assert reason == ("not run: the same request was rejected on runs at least 24 hours apart (claude-haiku-5-5|v1|d1: "
                      "HTTP 400 invalid_request_error: thinking: not supported)")
    assert "invalid_request_error: invalid_request_error" not in reason and not H.LOCK.exists()
    monkeypatch.setattr(X, "item_sets", sandbox["real_item_sets"])
    out = HA.analyse(cache_path=H.CACHE_FILE, run_log=H.RUN_LOG, spend_path=H.SPEND)
    assert out["models"]["claude-haiku-5-5"]["status"] == reason
    assert out["summary"].startswith("Hosted arm, on DEV: not finished, so no count is given; not closed: "
                                     "claude-sonnet-5-5 (not started); claude-opus-5-5 (not started).")
    assert f"Not counted: claude-haiku-5-5 ({reason})." in out["summary"] and " of 1 models" not in out["summary"]


def test_the_24_hour_rule_reads_the_server_dates_and_check_sees_a_clock_moved_back(sandbox, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(H, "now", clock)
    server = "Tue, 01 Dec 2026 00:00:00 GMT"                      # the server's clock does not move
    for _ in range(2):
        with pytest.raises(X.Refusal, match="the request was rejected"):
            run("claude-haiku-5-5", sandbox, FakeTransport(fail_at=1, error=E(
                400, "thinking: not supported", etype="invalid_request_error", date=server)))
        clock.hours(25)                                            # this machine's clock moved 25 hours
    assert [e["server_date"] for e in X._jsonl(H.JOURNAL) if e["kind"] == "rejected"] == [server, server]
    with pytest.raises(X.Refusal, match="24 hours apart"):
        H.close_incomplete("claude-haiku-5-5")                     # local times 25 h apart do not count
    without_date = E(400, "thinking: not supported", etype="invalid_request_error")
    with pytest.raises(X.Refusal):
        run("claude-haiku-5-5", sandbox, FakeTransport(fail_at=1, error=without_date))
    with pytest.raises(X.Refusal, match="24 hours apart"):
        H.close_incomplete("claude-haiku-5-5")                     # no server date: never closes
    assert H.verify(records=sandbox["records"]) == []
    clock.hours(-100)                                              # the clock set back, then another run
    with pytest.raises(X.Refusal):
        run("claude-haiku-5-5", sandbox, FakeTransport(fail_at=1, error=without_date))
    problems = H.verify(records=sandbox["records"])
    assert any(p.startswith("sent.jsonl: its times are not in order") for p in problems)
    assert any(p.startswith("run_log.jsonl: its times are not in order") for p in problems)


def test_verify_catches_an_unclosed_over_bound_answer_and_an_unclosed_mismatch(sandbox):
    run("claude-haiku-5-5", sandbox, FakeTransport())
    assert H.verify(records=sandbox["records"]) == []
    rows = X._jsonl(H.CACHE_FILE)
    rows[0]["prompt_eval_count"] = 10 ** 6                         # what a crash before the close would leave
    rows[1]["eval_count"] = 513
    H.CACHE_FILE.write_text("".join(json.dumps(r) + "\n" for r in rows))
    X._append(H.SPEND, {"at": X.now(), "event": "charged", "model": "claude-haiku-5-5", "key": rows[2]["key"],
                        "cost_usd": 0.0, "running_total_usd": round(H.spend_state()["computed"], 6),
                        "note": "answer served by claude-sonnet-5-5"})
    problems = H.verify(records=sandbox["records"])
    assert any("billed more input than the bound" in p for p in problems)
    assert any("more output tokens than max_tokens" in p for p in problems)
    assert any("answered by another model, and claude-haiku-5-5 is not closed for it" in p for p in problems)


def test_sixty_rejections_in_a_row_do_not_reach_the_error_bound_and_cannot_close_a_model(sandbox):
    rejection = E(400, "thinking: not supported", etype="invalid_request_error")
    for _ in range(H.MAX_ERRORS_PER_MODEL):                       # 60 immediate re-runs of a rejected request
        with pytest.raises(X.Refusal, match="the request was rejected"):
            run("claude-haiku-5-5", sandbox, FakeTransport(fail_at=1, error=rejection))
    journal = X._jsonl(H.JOURNAL)
    assert sum(e["kind"] == "rejected" for e in journal) == 60 and H.error_count(journal, "claude-haiku-5-5") == 0
    assert H.COUNTED_ERRORS == ("transient",)
    with pytest.raises(X.Refusal, match="24 hours apart"):
        H.close_incomplete("claude-haiku-5-5")                     # only the 24-hour rule can close it
    assert "claude-haiku-5-5" not in H.state()["status"]


@pytest.mark.parametrize("status, etype, message, code", [
    (400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API.", None),
    (400, "invalid_request_error", "You have reached your specified API usage limits.", None),
    (401, "authentication_error", "invalid x-api-key", None),
    (429, "rate_limit_error", "monthly spend limit", "enforced_spend_limit_reached"),
])
def test_an_account_condition_never_closes_a_model_even_24_hours_later(sandbox, monkeypatch, status, etype, message,
                                                                        code):
    from judge import hosted_analysis as HA
    clock = Clock()
    monkeypatch.setattr(H, "now", clock)
    for _ in range(2):
        t = FakeTransport(fail_at=1, error=E(status, message, etype=etype, code=code))
        with pytest.raises(X.Refusal, match=r"an account condition \(the key, the credit balance or a spend limit\)"):
            run("claude-haiku-5-5", sandbox, t)
        with pytest.raises(X.Refusal, match="never justifies a close"):
            H.close_incomplete("claude-haiku-5-5")
        clock.hours(25)
    assert "claude-haiku-5-5" not in H.state()["status"] and H.error_count(X._jsonl(H.JOURNAL), "claude-haiku-5-5") == 0
    monkeypatch.setattr(X, "item_sets", sandbox["real_item_sets"])
    s = HA.analyse(cache_path=H.CACHE_FILE, run_log=H.RUN_LOG, spend_path=H.SPEND)["summary"]
    assert s.startswith("Hosted arm, on DEV: not finished, so no count is given; not closed: claude-haiku-5-5 (0 of "
                        "600 answered; its last run stopped: an account condition (the key, the credit balance or a "
                        "spend limit)); claude-sonnet-5-5 (not started)")
    assert " of 1 models" not in s and " of 0 models" not in s and "request rejected" not in s


def test_mock_a_credit_balance_400_does_not_close_haiku(mock):
    """The third review's case, through the real transport."""
    Mock.plan[:] = [err(400, "invalid_request_error", "Your credit balance is too low to access the Anthropic API. "
                                                      "Please go to Plans & Billing to upgrade or purchase credits.")]
    with pytest.raises(X.Refusal, match="account condition"):
        run_real(mock)
    with pytest.raises(X.Refusal, match="never justifies a close"):
        H.close_incomplete("claude-haiku-5-5")
    assert "claude-haiku-5-5" not in H.state()["status"]
    err_entry = [e for e in X._jsonl(H.JOURNAL) if e["kind"] == "account"][0]
    assert H._server_seconds(err_entry["server_date"]) is not None          # the server's date header, recorded
    Mock.plan[:] = [ok() for _ in range(10)]
    assert run_real(mock)["missing"] == 0
    assert all(H._server_seconds(r["server_date"]) for r in X._jsonl(H.CACHE_FILE))


def test_a_budget_stop_that_a_crash_left_unclosed_can_be_closed(sandbox):
    k = "claude-haiku-5-5|v1|d1"
    X._append(H.JOURNAL, {"at": X.now(), "key": k, "kind": "send", "request_sha256": "x"})
    X._append(H.RUN_LOG, {"event": "budget_stop", "model": "claude-haiku-5-5", "key": k})
    with pytest.raises(X.Refusal, match="resend-unanswered"):
        run("claude-haiku-5-5", sandbox, FakeTransport())        # nothing settled the send: the run asks for a resend
    assert H.close_incomplete("claude-haiku-5-5") == "not run: stopped by the 15 USD cap"


def test_close_incomplete_takes_the_run_lock(sandbox, monkeypatch):
    monkeypatch.setattr(H, "MAX_ERRORS_PER_MODEL", 1)
    with pytest.raises(X.Refusal):
        run("claude-haiku-5-5", sandbox, FakeTransport(fail_at=1, error=E(529, "overloaded", etype="overloaded_error")))
    H.acquire_lock()
    with pytest.raises(X.Refusal, match="run.lock"):
        H.close_incomplete("claude-haiku-5-5")
    assert "claude-haiku-5-5" not in H.state()["status"]
    H.release_lock()
    assert H.close_incomplete("claude-haiku-5-5") == "not run: error-answer limit reached"


def test_an_interruption_and_the_one_resend(sandbox):
    t = FakeTransport(fail_at=4, error=E(None, "no answer: TimeoutError", maybe_billed=True))
    with pytest.raises(X.Refusal, match="interrupted"):
        run("claude-haiku-5-5", sandbox, t)
    assert [e["note"] for e in X._jsonl(H.SPEND) if e["event"] == "charged"] == [
        "worst case of a request that may have been billed"]
    with pytest.raises(X.Refusal, match="resend-unanswered"):
        run("claude-haiku-5-5", sandbox, FakeTransport())
    with pytest.raises(X.Refusal):
        run("claude-haiku-5-5", sandbox, FakeTransport(fail_at=1, error=E(None, "no answer", maybe_billed=True)),
            resend_unanswered=True)
    t3 = FakeTransport()
    out = run("claude-haiku-5-5", sandbox, t3, resend_unanswered=True)
    assert out["missing"] == 1 and len(t3.calls) == 6                # closed by the code at the end of its work list
    assert H.state()["status"]["claude-haiku-5-5"] == ("incomplete (9 of 10 answered): 1 request(s) unanswered after "
                                                       "their one resend")
    with pytest.raises(X.Refusal, match="already closed"):
        H.close_incomplete("claude-haiku-5-5")
    assert H.verify(records=sandbox["records"]) == []


def test_a_request_interrupted_twice_is_printed_with_its_resend_and_causes(sandbox, monkeypatch):
    from judge import hosted_analysis as HA
    interrupt = E(None, "no answer: KeyboardInterrupt: ", maybe_billed=True)
    with pytest.raises(X.Refusal, match="interrupted"):
        run("claude-haiku-5-5", sandbox, FakeTransport(fail_at=4, error=interrupt))
    with pytest.raises(X.Refusal, match="interrupted"):
        run("claude-haiku-5-5", sandbox, FakeTransport(fail_at=1, error=interrupt), resend_unanswered=True)
    assert run("claude-haiku-5-5", sandbox, FakeTransport(), resend_unanswered=True)["missing"] == 1
    monkeypatch.setattr(X, "item_sets", sandbox["real_item_sets"])
    out = HA.analyse(cache_path=H.CACHE_FILE, run_log=H.RUN_LOG, spend_path=H.SPEND, journal_path=H.JOURNAL)
    k = "claude-haiku-5-5|v1|l1"
    assert (f"Resends and losses: claude-haiku-5-5: 1 resent ({k}), 2 charged without an answer ({k}: no answer: "
            f"KeyboardInterrupt; {k}: no answer: KeyboardInterrupt).") in out["summary"]
    assert ("claude-haiku-5-5 (incomplete (9 of 10 answered): 1 request(s) unanswered after their one resend)"
            in out["summary"])
    assert "claude-haiku-5-5 0.00 USD (9 answers, 2 charged without an answer; 6300 input, 360 output and 0 " \
           "thinking tokens)" in out["summary"]


def test_rate_limits_and_overloads_are_waited_out_not_charged_and_do_not_use_the_resend(sandbox):
    waits = []
    err = E(429, "rate_limit_error: per-minute rate limit", etype="rate_limit_error", retry_after=90)
    t = FakeTransport(fail_at=(1, 2, 3, 4), error=err)          # three retries, then a fourth 429: the run stops
    with pytest.raises(X.Refusal, match="not served"):
        H.run_model("claude-haiku-5-5", t, records=sandbox["records"], sleep=waits.append)
    assert [w for w in waits if w >= 30] == [90, 90, 120]       # max(fixed delay, retry-after), at most 300
    t2 = FakeTransport(fail_at=(1, 2, 3, 4), error=E(529, "overloaded_error", etype="overloaded_error"))
    with pytest.raises(X.Refusal):                               # no --resend-unanswered needed: nothing served
        run("claude-haiku-5-5", sandbox, t2)
    out = run("claude-haiku-5-5", sandbox, FakeTransport())
    assert out["missing"] == 0
    kinds = [e["kind"] for e in X._jsonl(H.JOURNAL) if e["key"] == "claude-haiku-5-5|v1|d1"]
    assert kinds.count("transient") == 8 and kinds[-1] == "send" and "resend" not in kinds
    assert not [e for e in X._jsonl(H.SPEND) if e["event"] == "charged"]          # error answers are not charged
    assert H.verify(records=sandbox["records"]) == []


def test_a_5xx_is_retried_and_not_charged(sandbox):
    t = FakeTransport(fail_at=1, error=E(500, "api_error: internal", etype="api_error"))
    out = run("claude-haiku-5-5", sandbox, t)
    assert out["missing"] == 0 and len(t.calls) == 11
    assert [e["event"] for e in X._jsonl(H.SPEND)].count("charged") == 0
    assert np.isclose(H.spent(), 10 * H.cost("claude-haiku-5-5", {"input_tokens": 700, "output_tokens": 40}))


def test_a_spend_cap_429_stops_at_once_and_is_not_counted(sandbox):
    waits = []
    t = FakeTransport(fail_at=1, error=E(429, "rate_limit_error: cap", etype="rate_limit_error",
                                         code="enforced_spend_limit_reached"))
    with pytest.raises(X.Refusal, match="account condition"):
        H.run_model("claude-haiku-5-5", t, records=sandbox["records"], sleep=waits.append)
    assert len(t.calls) == 1 and not [w for w in waits if w >= 30]
    assert H.error_count(X._jsonl(H.JOURNAL), "claude-haiku-5-5") == 0


def test_the_error_answers_of_a_model_are_bounded(sandbox, monkeypatch):
    monkeypatch.setattr(H, "MAX_ERRORS_PER_MODEL", 2)
    t = FakeTransport(fail_at=(1, 2, 3), error=E(529, "overloaded_error", etype="overloaded_error"))
    with pytest.raises(X.Refusal):
        run("claude-haiku-5-5", sandbox, t)
    assert len(t.calls) == 2
    with pytest.raises(X.Refusal, match="registered maximum"):
        run("claude-haiku-5-5", sandbox, FakeTransport())
    H.close_incomplete("claude-haiku-5-5")
    assert [e for e in X._jsonl(H.RUN_LOG) if e["event"] == "closed_incomplete"][0]["reason"] == \
        "not run: error-answer limit reached"


def test_verify_catches_too_many_retries_of_one_send(sandbox):
    run("claude-haiku-5-5", sandbox, FakeTransport())
    k = "claude-haiku-5-5|v1|d1"
    for n in range(4):                       # four retries after one send, with error answers between them
        X._append(H.JOURNAL, {"at": X.now(), "key": k, "kind": "transient", "status": 529})
        X._append(H.JOURNAL, {"at": X.now(), "key": k, "kind": "retry", "retry": n + 1, "request_sha256": "x"})
    assert any("retried more often" in p for p in H.verify(records=sandbox["records"]))


def test_a_different_model_is_charged_not_cached_and_closes_the_model(sandbox):
    with pytest.raises(X.Refusal, match="closed as incomplete"):
        run("claude-haiku-5-5", sandbox, FakeTransport(model="claude-sonnet-5-5"))
    assert H.state()["status"]["claude-haiku-5-5"] == "not run: served by claude-sonnet-5-5" and \
        X._jsonl(H.CACHE_FILE) == []
    assert len([e for e in X._jsonl(H.SPEND) if e["event"] == "charged"]) == 1


def test_an_answer_reporting_another_routing_is_kept_priced_at_1_1x_and_counted(sandbox):
    out = run("claude-haiku-5-5", sandbox, FakeTransport(geo="us"))
    assert out["missing"] == 0 and len(X._jsonl(H.CACHE_FILE)) == 10
    answers = [e for e in X._jsonl(H.SPEND) if e["event"] == "answer"]
    assert np.isclose(answers[0]["cost_usd"], 1.1 * H.cost("claude-haiku-5-5", {"input_tokens": 700,
                                                                                 "output_tokens": 40}))
    assert X._jsonl(H.SPEND)[-1]["routed_elsewhere"] == 10 and "claude-haiku-5-5" in H.state()["status"]
    assert H.verify(records=sandbox["records"]) == []


# ------------------------------------------------------------------ the write-ahead reservation and the lock
def test_a_hard_crash_leaves_its_reservation_charged_until_the_next_run_charges_it(sandbox):
    run("claude-haiku-5-5", sandbox, FakeTransport())
    k = "claude-sonnet-5-5|v1|d1"
    body = H.build_body("claude-sonnet-5-5", "v1", sandbox["records"]["d1"], "text d1")
    before = H.spent()
    at = X.now()                                  # what a run leaves behind when it is killed after sending
    X._append(H.SPEND, {"at": at, "event": "reserved", "id": f"{at}|{k}", "model": "claude-sonnet-5-5", "key": k,
                        "cost_usd": H.worst_case("claude-sonnet-5-5", body)})
    X._append(H.JOURNAL, {"at": at, "key": k, "kind": "send", "request_sha256": H.request_hash(body)})
    assert np.isclose(H.spent(), before + H.worst_case("claude-sonnet-5-5", body))     # counted against the cap
    with pytest.raises(X.Refusal, match="resend-unanswered"):    # the next run charges it, then asks for the resend
        run("claude-sonnet-5-5", sandbox, FakeTransport())
    charged = [e for e in X._jsonl(H.SPEND) if e["event"] == "charged"]
    assert len(charged) == 1 and charged[0]["note"].endswith("(crash)") and H.spend_state()["open"] == []
    out = run("claude-sonnet-5-5", sandbox, FakeTransport(), resend_unanswered=True)
    assert out["missing"] == 0 and H.verify(records=sandbox["records"]) == []


class Crash(BaseException):
    pass


def test_a_crash_after_an_answer_is_billed_charges_nothing_twice_and_leaves_check_passing(sandbox, monkeypatch):
    from judge import hosted_analysis as HA
    real = H._append

    def crash_at_the_cache_line(path, record):
        if path == H.CACHE_FILE and record["key"].endswith("|v1|d2"):
            raise Crash()
        real(path, record)
    monkeypatch.setattr(H, "_append", crash_at_the_cache_line)
    with pytest.raises(Crash):
        run("claude-haiku-5-5", sandbox, FakeTransport())
    monkeypatch.setattr(H, "_append", real)
    H.release_lock()                                              # what --break-stale-lock does after a real crash
    assert H.spend_state()["open"] == [] and H.verify(records=sandbox["records"]) == []
    one = H.cost("claude-haiku-5-5", {"input_tokens": 700, "output_tokens": 40})
    assert np.isclose(H.spent(), 2 * one)                         # the lost answer was billed, and only once
    with pytest.raises(X.Refusal, match="resend-unanswered"):
        run("claude-haiku-5-5", sandbox, FakeTransport())
    assert [e["event"] for e in X._jsonl(H.SPEND)].count("charged") == 0
    assert run("claude-haiku-5-5", sandbox, FakeTransport(), resend_unanswered=True)["missing"] == 0
    assert H.verify(records=sandbox["records"]) == []
    c = HA.costs(X._jsonl(H.SPEND))["claude-haiku-5-5"]
    assert c["answers"] == 10 and c["charged_without_answer"] == 1 and np.isclose(c["usd"], 11 * one, atol=1e-4)


def test_a_reservation_left_by_a_crash_is_charged_by_a_close_and_counted_by_the_report(sandbox):
    from judge import hosted_analysis as HA
    k = "claude-haiku-5-5|v1|d1"
    body = H.build_body("claude-haiku-5-5", "v1", sandbox["records"]["d1"], "text d1")
    at = X.now()
    X._append(H.SPEND, {"at": at, "event": "reserved", "id": f"{at}|{k}", "model": "claude-haiku-5-5", "key": k,
                        "cost_usd": H.worst_case("claude-haiku-5-5", body)})
    X._append(H.JOURNAL, {"at": at, "key": k, "kind": "send", "request_sha256": H.request_hash(body)})
    c = HA.costs(X._jsonl(H.SPEND))
    assert c["claude-haiku-5-5"]["open_reservations"] == 1 and c["total_usd"] > 0     # counted before any close
    with pytest.raises(X.Refusal, match="cannot be closed"):
        H.close_incomplete("claude-haiku-5-5")
    assert H.spend_state()["open"] == [] and [e["event"] for e in X._jsonl(H.SPEND)].count("charged") == 1
    assert np.isclose(HA.costs(X._jsonl(H.SPEND))["total_usd"], c["total_usd"])


def test_every_send_is_reserved_before_it_is_journaled_and_released_after(sandbox):
    run("claude-haiku-5-5", sandbox, FakeTransport())
    spend = X._jsonl(H.SPEND)
    reserved = [e for e in spend if e["event"] == "reserved"]
    assert len(reserved) == 10 and H.spend_state()["open"] == []
    journal = X._jsonl(H.JOURNAL)
    assert all(r["at"] == j["at"] for r, j in zip(reserved, journal))


def test_the_lock_is_taken_before_any_state_is_read(sandbox, monkeypatch):
    H.acquire_lock()
    monkeypatch.setattr(H, "preflight", lambda *a, **k: pytest.fail("state was read while another run held the lock"))
    with pytest.raises(X.Refusal, match="run.lock"):
        run("claude-haiku-5-5", sandbox, FakeTransport())
    H.release_lock()


def test_a_second_run_started_during_the_first_sends_nothing_twice(sandbox):
    gate, started = threading.Event(), threading.Event()

    class Slow(FakeTransport):
        def send(self, body):
            started.set()
            gate.wait(5)
            return super().send(body)
    first = Slow()
    th = threading.Thread(target=lambda: run("claude-haiku-5-5", sandbox, first))
    th.start()
    started.wait(5)
    second = FakeTransport()
    with pytest.raises(X.Refusal, match="run.lock"):            # the reviewer's second run: kept out
        run("claude-haiku-5-5", sandbox, second)
    gate.set()
    th.join(10)
    out = run("claude-haiku-5-5", sandbox, second)               # afterwards it reads fresh state: nothing to send
    assert out["new_answers"] == 0 and second.calls == [] and len(first.calls) == 10
    keys = [r["key"] for r in X._jsonl(H.CACHE_FILE)]
    assert len(keys) == len(set(keys)) == 10 and H.verify(records=sandbox["records"]) == []


def test_a_stale_lock_is_removed_only_when_its_process_is_gone(sandbox):
    import subprocess
    import sys
    H.acquire_lock()                                             # held by this (living) process
    with pytest.raises(X.Refusal, match="still running"):
        H.break_stale_lock()
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    H.LOCK.write_text(json.dumps({"pid": dead.pid, "at": X.now()}))
    assert "removed" in H.break_stale_lock() and not H.LOCK.exists()
    assert events()[-1] == "stale_lock_removed" and H.break_stale_lock() == "no run.lock"


# ------------------------------------------------------------------ the real transport against a mock server
class Mock(BaseHTTPRequestHandler):
    plan, seen = [], []

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length") or 0)
        body = self.rfile.read(n) if n else b""
        Mock.seen.append((self.server.server_port, self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        status, headers, payload = Mock.plan.pop(0)
        self.send_response(status)
        for k, v in headers.items():
            self.send_header(k, v)
        if "content-length" not in {k.lower() for k in headers}:
            self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)
    do_GET = do_POST


def ok(model="claude-haiku-5-5", usage=None):
    return (200, {"request-id": "req_ok", "content-type": "application/json"}, json.dumps(_resp(
        '{"monetary_relief": true, "p_relief": 0.5, "reason": "x"}', model=model, usage=usage)).encode())


def err(status, etype, msg, headers=None):
    return (status, {"request-id": "req_err", "content-type": "application/json", **(headers or {})},
            json.dumps({"type": "error", "error": {"type": etype, "message": msg}, "request_id": "req_err"}).encode())


@pytest.fixture
def mock(sandbox, monkeypatch):
    servers = [ThreadingHTTPServer(("127.0.0.1", 0), Mock) for _ in range(2)]
    for s in servers:
        threading.Thread(target=s.serve_forever, daemon=True).start()
    Mock.plan, Mock.seen = [], []
    monkeypatch.setattr(H, "API_URL", f"http://127.0.0.1:{servers[0].server_port}/v1/messages")
    monkeypatch.setenv(H.KEY_ENV, FAKE_KEY)
    yield {**sandbox, "api": servers[0].server_port, "other": servers[1].server_port}
    for s in servers:
        s.shutdown()
        s.server_close()


def run_real(mock):
    return H.run_model("claude-haiku-5-5", H.AnthropicTransport(), records=mock["records"], sleep=lambda s: None)


def test_mock_403_and_404_naming_the_model_stop_and_drop_nothing(mock):
    for status, etype, msg in ((403, "permission_error", "Your API key does not have access to model: "
                                                         "claude-haiku-5-5"),
                               (404, "not_found_error", "model: claude-haiku-5-5")):
        Mock.plan[:] = [err(status, etype, msg)]
        with pytest.raises(X.Refusal, match="not served"):
            run_real(mock)
        assert "model_dropped" not in events() and "closed_incomplete" not in events()


@pytest.mark.parametrize("answer", [
    "truncated", "not json", "odd shape",
])
def test_mock_a_200_that_cannot_be_read_is_charged_and_stops_the_run(mock, answer):
    full = ok()[2]
    if answer == "truncated":
        Mock.plan[:] = [(200, {"request-id": "req_cut", "content-length": str(len(full))}, full[: len(full) // 2])]
    elif answer == "not json":
        Mock.plan[:] = [(200, {"request-id": "req_html"}, b"<html>gateway</html>")]
    else:
        Mock.plan[:] = [(200, {"request-id": "req_odd"}, json.dumps({
            "model": "claude-haiku-5-5", "content": ["x"], "stop_reason": "end_turn",
            "usage": {"input_tokens": 700, "output_tokens": 40}}).encode())]
    with pytest.raises(X.Refusal, match="interrupted"):
        run_real(mock)
    charged = [e for e in X._jsonl(H.SPEND) if e["event"] == "charged"]
    assert len(charged) == 1 and charged[0]["cost_usd"] > 0 and events()[-1] == "run_stopped"
    if answer == "odd shape":                                    # the usage was readable: charged at its price
        assert np.isclose(charged[0]["cost_usd"], H.cost("claude-haiku-5-5", {"input_tokens": 700,
                                                                               "output_tokens": 40}))
    assert X._jsonl(H.CACHE_FILE) == []
    with pytest.raises(X.Refusal, match="resend-unanswered"):    # processed: it uses the one resend
        run_real(mock)


def test_mock_a_redirect_is_not_followed_and_the_key_goes_nowhere_else(mock):
    Mock.plan[:] = [(302, {"location": f"http://127.0.0.1:{mock['other']}/elsewhere"}, b""), ok()]
    with pytest.raises(X.Refusal, match="not served"):
        run_real(mock)
    assert [port for port, *_ in Mock.seen] == [mock["api"]]     # the other host never saw a request
    assert Mock.seen[0][2]["x-api-key"] == FAKE_KEY and Mock.seen[0][2]["anthropic-version"] == "2026-10-21"


def test_mock_an_interrupt_while_waiting_is_charged(sandbox):
    class Interrupting:
        def send(self, body):
            raise KeyboardInterrupt()
    with pytest.raises(X.Refusal, match="interrupted"):
        run("claude-haiku-5-5", sandbox, Interrupting())
    assert [e["event"] for e in X._jsonl(H.SPEND)].count("charged") == 1 and events()[-1] == "run_stopped"


def test_mock_routing_us_is_kept_and_priced_at_1_1x(mock):
    Mock.plan[:] = [ok(usage={"input_tokens": 1000, "output_tokens": 0, "inference_geo": "us"})] + \
        [ok() for _ in range(9)]
    assert run_real(mock)["missing"] == 0
    answers = [e for e in X._jsonl(H.SPEND) if e["event"] == "answer"]
    assert np.isclose(answers[0]["cost_usd"], 1000 * 0.10 * 1.1 / 1e6) and answers[0]["inference_geo"] == "us"
    assert json.loads(Mock.seen[0][3])["inference_geo"] == "global"


def test_mock_rate_limit_then_answers(mock):
    Mock.plan[:] = [err(429, "rate_limit_error", "per-minute", {"retry-after": "5"})] + [ok() for _ in range(10)]
    out = run_real(mock)
    assert out["missing"] == 0 and H.spent() > 0 and H.verify(records=mock["records"]) == []


# ------------------------------------------------------------------ the key never reaches a file or a message
def test_the_key_is_sent_only_in_its_header_and_never_written(mock, capsys):
    Mock.plan[:] = [ok(), ok(), ok(), err(400, "invalid_request_error", f"x-api-key {FAKE_KEY} rejected; also "
                                                                         f"{OTHER_KEY}", {"x-api-key": FAKE_KEY})]
    with pytest.raises(X.Refusal) as exc:
        run_real(mock)
    assert len(Mock.seen) == 4 and all(h["x-api-key"] == FAKE_KEY for _, _, h, _ in Mock.seen)
    assert all(FAKE_KEY.encode() not in body for *_, body in Mock.seen)
    for e in (TimeoutError("timed out"), ConnectionResetError(f"reset {FAKE_KEY}")):
        def boom(req, timeout, e=e):
            raise e
        H_open = H._open
        H._open = boom
        try:
            with pytest.raises(H.HostedError) as he:
                H.AnthropicTransport().send(H.build_body("claude-haiku-5-5", "v1", mock["records"]["d1"], "x"))
        finally:
            H._open = H_open
        assert FAKE_KEY not in str(he.value) and he.value.__context__ is None and he.value.__cause__ is None
    out = capsys.readouterr()
    texts = [str(exc.value), out.out, out.err] + [p.read_text() for p in mock["tmp"].rglob("*") if p.is_file()]
    assert any("req_err" in t for t in texts)                      # the error was recorded...
    for t in texts:                                                 # ...without the key or anything like it
        assert FAKE_KEY not in t and "sk-" + "ant-" not in t and "FAKEKEYFORTESTS" not in t
    assert all("API key" not in p for p in H.verify(records=mock["records"]))


def test_no_key_means_nothing_is_sent(monkeypatch):
    monkeypatch.delenv(H.KEY_ENV, raising=False)
    monkeypatch.setattr(H, "_open", lambda *a, **k: pytest.fail("a request was sent"))
    with pytest.raises(H.HostedError, match="not set"):
        H.AnthropicTransport().send({"model": "claude-haiku-5-5"})


def test_main_prints_no_traceback(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["judge.hosted", "--run", "claude-haiku-5-5"])
    monkeypatch.setattr(H, "run_model", lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"boom {FAKE_KEY}")))
    with pytest.raises(SystemExit) as e:
        H.main()
    assert "hosted arm stopped: RuntimeError" in str(e.value.code) and FAKE_KEY not in str(e.value.code)


def test_the_makefile_and_the_code_name_no_key_and_no_key_file():
    make = (C.ROOT / "Makefile").read_text()
    target = make.split("models-hosted:")[1].split("\n\n")[0]
    assert "judge.hosted --run $(MODEL)" in target
    code = (C.ROOT / "src/judge/hosted.py").read_text()
    for text in (make, code):
        assert "sk-" + "ant-" not in text.replace('r"sk-ant-[A-Za-z0-9_\\-]{8,}"', "")
        assert not re.search(r"""['"/]\.env\b|\.netrc|keychain|dotenv""", text)     # no key file is read
    assert code.count("os.environ.get(KEY_ENV)") == 3 and "os.environ[" not in code   # send, scrub, preflight
    assert "$(ANTHROPIC_API_KEY)" not in make and "ANTHROPIC_API_KEY=" not in make
    assert "add_unredirected_header(\"x-api-key\"" in code


def test_the_real_hosted_directory_holds_no_key_and_verifies():
    assert H.verify() == []


# ------------------------------------------------------------------ preflight
@pytest.fixture
def pre_env(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "RUN_LOG", tmp_path / "local_run_log.jsonl")
    for name in ("CACHE_FILE", "JOURNAL", "RUN_LOG", "SPEND", "REGISTRATION", "LOCK"):
        monkeypatch.setattr(H, name, tmp_path / getattr(H, name).name)
    monkeypatch.setattr(X, "git", lambda *a: "h" * 40 if a[0] == "rev-parse" else "")
    monkeypatch.setenv(H.KEY_ENV, FAKE_KEY)
    return tmp_path


def test_preflight_order_of_refusals(pre_env, monkeypatch):
    head = lambda: "h" * 40   # noqa: E731
    with pytest.raises(X.Refusal, match="not a model of Amendment 2"):
        H.preflight("claude-fable-5-2", published_head=head)
    with pytest.raises(X.Refusal, match="local models are not all closed"):
        H.preflight("claude-haiku-5-5", published_head=head)
    for t, _ in X.MODELS:
        X._append(X.RUN_LOG, {"event": "model_finished", "model": t})
    with pytest.raises(X.Refusal, match="missing"):
        H.preflight("claude-haiku-5-5", published_head=head)
    H.REGISTRATION.write_text(json.dumps(FILLED))
    assert H.preflight("claude-haiku-5-5", published_head=head)["sets"]
    with pytest.raises(X.Refusal, match="comes before"):
        H.preflight("claude-sonnet-5-5", published_head=head)
    with pytest.raises(X.Refusal, match="not the commit GitHub shows"):
        H.preflight("claude-haiku-5-5", published_head=lambda: "x" * 40)
    monkeypatch.delenv(H.KEY_ENV)
    with pytest.raises(X.Refusal, match="not set"):
        H.preflight("claude-haiku-5-5", published_head=head)
    monkeypatch.setenv(H.KEY_ENV, FAKE_KEY)
    X._append(H.RUN_LOG, {"event": "budget_stop", "model": "claude-haiku-5-5"})
    X._append(H.RUN_LOG, {"event": "closed_incomplete", "model": "claude-haiku-5-5",
                          "reason": "incomplete (3 of 600 answered): stopped by the 15 USD cap"})
    with pytest.raises(X.Refusal, match="no request of this arm is sent after a budget stop"):
        H.preflight("claude-sonnet-5-5", published_head=head)


# ------------------------------------------------------------------ the analysis
def _copy_reference(tag, sets, study, **extra):
    rows = []
    for v, s in X.CELLS:
        for i in sets[s]:
            rows.append(dict(study[f"{v}|{i}"], key=X.key(tag, v, i), model=tag, stop_reason="end_turn", **extra))
    return rows


def test_the_analysis_compares_haiku_and_sonnet_only_and_shows_opus_as_a_labelled_reference_row(tmp_path):
    from judge import hosted_analysis as HA
    from judge import models_analysis as A

    study = {r["key"]: r for r in X._jsonl(C.CACHE_FILE)}
    sets, _ = X.item_sets()
    rows = (_copy_reference("claude-haiku-5-5", sets, study)
            + _copy_reference("claude-opus-5-5", sets, study, thinking_tokens=120, block_types=["thinking", "text"]))
    cache = tmp_path / "responses.jsonl"
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    spend = tmp_path / "spend.jsonl"
    spend.write_text(json.dumps({"event": "answer", "model": "claude-haiku-5-5", "key": "k", "cost_usd": 0.5,
                                 "input_tokens": 10, "output_tokens": 1}) + "\n")
    log = tmp_path / "run_log.jsonl"
    log.write_text("".join(json.dumps(e) + "\n" for e in (
        {"event": "model_finished", "model": "claude-haiku-5-5"},
        {"event": "closed_incomplete", "model": "claude-sonnet-5-5", "reason": "not run: error-answer limit reached"},
        {"event": "model_finished", "model": "claude-opus-5-5"})))
    out = HA.analyse(cache_path=cache, run_log=log, spend_path=spend)
    local = A.analyse(cache_path=tmp_path / "empty.jsonl")
    ref = local["models"]["qwen3.8:27b"]["cells"]
    haiku = out["models"]["claude-haiku-5-5"]
    assert haiku["compared"] and haiku["cells"]["v1|dev"]["A"]["label"] == ref["v1|dev"]["A"]["label"]
    assert haiku["cells"]["v1|dev"]["pays"] == 137 and haiku["cells"]["v1|dev"]["kappa_with_reference"] == 1.0
    assert np.isclose(haiku["cells"]["v1|dev"]["A"]["corrected_level"], 1 - 0.05 / 4)     # 98.75%
    assert np.isclose(haiku["cells"]["v1|dev"]["B"]["corrected_level"], 1 - 0.05 / 2)     # 97.5%
    opus = out["models"]["claude-opus-5-5"]
    assert not opus["compared"] and opus["row_note"] == HA.REFERENCE_ROW
    assert "A" not in opus["cells"]["v1|dev"] and "A_descriptive" in opus["cells"]["v1|dev"]
    assert opus["cells"]["v1|dev"]["thinking_tokens"]["median"] == 120
    s = out["summary"]
    assert s.startswith("Hosted arm, on DEV: judge v1 paid more often than the companies did for 1 of 1 models;")
    assert "Not counted: claude-sonnet-5-5 (not run: error-answer limit reached)." in s
    assert "claude-opus-5-5" not in s.split("Not counted:")[1].split(".")[0]
    note = HA.opus_note(out)
    assert note.startswith(HA.OPUS_SENTENCE) and note.endswith("claude-haiku-5-5 ran without up-front thinking and "
                                                               "max_tokens 512, as the local models did.")
    assert f"claude-opus-5-5 ({HA.REFERENCE_ROW}; complete)" in s and s.count(HA.DISCLOSURE) == 1 and note in s
    import inspect
    src = inspect.getsource(HA.summary)
    assert "Amendment 2, section 14)" in src and "section 13" not in src          # training data is section 14
    plan = X.PLAN_FILE.read_text()
    assert HA.DISCLOSURE.startswith("Same vendor. The three models under test are Anthropic's.")
    assert " ".join(HA.DISCLOSURE.split()[2:12]) in " ".join(plan.split())               # read from the plan
    for sentence in (HA.OPUS_SENTENCE, HA.PERSON_NOTE, "Hosted arm, on DEV: not finished, so no count is given; not "
                     "closed:", "not show that the results are independent of the model."):
        assert sentence in " ".join(plan.split())                                     # as registered
    for fact in ("claude-opus-5-5`, one of the three models evaluated here, and so did every independent review of "
                 "this amendment",
                 "`claude-fable-5-2`", "only after all seven local models are closed"):
        assert fact in HA.DISCLOSURE
    assert "not ranked" in s and "kappa 0.202" in s and "`claude-opus-5-5`, one of the three models" in s
    assert "0.50 USD" in s and out["cost"]["total_usd"] == 0.5
    assert s.endswith(" No request was resent or charged without an answer.")
    t = out["table"]
    assert f"`claude-opus-5-5` ({HA.REFERENCE_ROW})" in t and "descriptive only" in t
    assert HA.DISCLOSURE not in t and HA.OPUS_SENTENCE not in t                  # printed once, in the summary
    opus_v1 = [ln for ln in t.splitlines() if ln.startswith("| `claude-opus-5-5`") and "| v1|dev |" in ln][0]
    assert opus_v1.endswith("| 120 / 120 | 0 | - |")                              # thinking tokens, routing, stops
    assert "Thinking tokens (median / max) | Routing not global | Other stops |" in t


def test_a_compared_model_that_reports_thinking_becomes_a_reference_row(tmp_path):
    from judge import hosted_analysis as HA

    study = {r["key"]: r for r in X._jsonl(C.CACHE_FILE)}
    sets, _ = X.item_sets()
    rows = _copy_reference("claude-sonnet-5-5", sets, study, inference_geo="us")
    rows[5]["block_types"] = ["redacted_thinking", "text"]           # the documented redacted block also counts
    cache = tmp_path / "responses.jsonl"
    cache.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = HA.analyse(cache_path=cache, run_log=tmp_path / "none.jsonl", spend_path=tmp_path / "none2.jsonl")
    m = out["models"]["claude-sonnet-5-5"]
    assert not m["compared"] and m["row_note"] == HA.MOVED_ROW and "A" not in m["cells"]["v1|dev"]
    assert sum(c["routed_elsewhere"] for c in m["cells"].values()) == 600      # kept, and counted
    rows = [ln for ln in out["table"].splitlines() if ln.startswith("| `claude-sonnet")]
    assert len(rows) == 4 and all(ln.split(" | ")[2].split(" of ")[0] == ln.split(" | ")[-2] for ln in rows)
    assert "claude-opus-5-5 ran" not in out["summary"]                        # Opus has no answers: no sentence

