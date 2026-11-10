"""The held-out runner against a fake local model server (no real model is called).

Each test interrupts a run, resumes it and counts, on the server side, how often
each distinct request arrived: once in the scored pass and once in the noise pass.
"""
import hashlib
import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from judge import config as C
from judge import freeze, llm
from judge import replay as P
from judge import run as R

N = 12
IDS = [str(1000 + i) for i in range(N)]
RECORDS = {i: {"product": "Credit card", "sub_product": "General-purpose credit card or charge card",
               "issue": "Problem with a purchase shown on your statement",
               "sub_issue": "Credit card company isn't resolving a dispute about a purchase on your statement",
               "narrative": f"I disputed a charge of $40 for an order that never arrived. Reference {i}.",
               "label": 0} for i in IDS}
SPLITS = {"test": IDS}
FROZEN = {"final_versions": ["v1", "v2"], "guarded_facts_version": "v2", "workers": 1,
          "expected_s_per_call": 0.01, "environment": {"ollama_version": "fake", "model_digest": "d"}}


class FakeModel:
    """Answers /api/chat after an optional delay; counts every chat request by the hash of its messages."""

    def __init__(self):
        self.requests, self.lock = [], threading.Lock()
        self.on_call = {}           # call number -> action ("sigint", "sigterm", "slow", "error")
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, obj, code=200):
                data = json.dumps(obj).encode()
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path == "/api/version":
                    return self._send({"version": "fake"})
                if self.path == "/api/tags":
                    return self._send({"models": [{"name": C.MODEL, "digest": "d"}]})
                self._send({}, 404)

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/show":
                    return self._send({"details": {}})
                with outer.lock:
                    outer.requests.append(hashlib.sha256(json.dumps(body["messages"]).encode()).hexdigest())
                    n = len(outer.requests)
                action = outer.on_call.get(n)
                if action in ("sigint", "sigterm"):
                    os.kill(os.getpid(), signal.SIGINT if action == "sigint" else signal.SIGTERM)
                    time.sleep(0.3)                 # the answer arrives after the signal
                elif action == "slow":
                    time.sleep(1.0)                 # longer than the client's timeout
                elif action == "error":
                    return self._send({"error": "boom"}, 500)
                self._send({"message": {"content": '{"relief": false}'}, "done_reason": "stop",
                            "prompt_eval_count": 10, "prompt_eval_cached_count": 0, "eval_count": 5,
                            "total_duration": 1, "load_duration": 0, "prompt_eval_duration": 1, "eval_duration": 1})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def counts(self):
        out = {}
        for h in self.requests:
            out[h] = out.get(h, 0) + 1
        return out


@pytest.fixture
def world(tmp_path, monkeypatch):
    fake = FakeModel()
    frozen_file = tmp_path / "frozen.json"
    frozen_file.write_text(json.dumps(FROZEN))
    monkeypatch.setattr(C, "OLLAMA_URL", fake.url)
    monkeypatch.setattr(C, "RUN_LOG", tmp_path / "run_log.jsonl")
    monkeypatch.setattr(llm.Cache.__init__, "__defaults__", (tmp_path / "responses.jsonl",))
    monkeypatch.setattr(P, "REPLAY_FILE", tmp_path / "replays.jsonl")
    monkeypatch.setattr(R, "TEST_MARKER", tmp_path / "test_run.json")
    monkeypatch.setattr(R, "SENT_FILE", tmp_path / "sent.jsonl")
    monkeypatch.setattr(freeze, "FROZEN_FILE", frozen_file)
    monkeypatch.setattr(freeze, "verify_method", lambda f: [])
    monkeypatch.setattr(R, "git_state", lambda: ("abc123", []))
    monkeypatch.setattr(R, "load_records", lambda: RECORDS)
    monkeypatch.setattr(R, "load_splits", lambda: SPLITS)
    yield fake, tmp_path
    fake.server.shutdown()


def answers(tmp):
    n = 0
    for f in ("responses.jsonl", "replays.jsonl"):
        p = tmp / f
        n += sum(1 for line in p.read_text().splitlines() if line.strip()) if p.exists() else 0
    return n


def journal(tmp):
    return [(e["pass"], e["key"]) for e in R.read_journal(tmp / "sent.jsonl")]


def resume_after_stop(fake, tmp, **kw):
    with pytest.raises(SystemExit, match="stopped on request"):
        R.final_test_run(**kw)


def test_stop_signals_mid_run_then_resume_send_each_request_once_per_pass(world):
    fake, tmp = world
    fake.on_call = {5: "sigint", 2 * N + 4: "sigterm"}       # Ctrl-C in the scored pass, SIGTERM in the noise pass
    resume_after_stop(fake, tmp)
    assert len(fake.requests) == 5 == answers(tmp)            # nothing sent after the signal; the answer on its way kept
    assert max(fake.counts().values()) == 1
    resume_after_stop(fake, tmp)                              # resume: scored pass completes, noise pass stops
    state = json.loads((tmp / "test_run.json").read_text())
    assert state["finished_at"] and not state.get("noise_finished_at")
    assert len(fake.requests) == 2 * N + 4 == answers(tmp)
    R.final_test_run()                                        # resume: noise pass completes
    counts = fake.counts()
    assert len(counts) == 2 * N and set(counts.values()) == {2}   # each request: once scored, once in the noise pass
    assert len(fake.requests) == 4 * N == answers(tmp)
    j = journal(tmp)
    assert len(j) == len(set(j)) == 4 * N                     # every journaled send is distinct
    assert sum(p == "scored" for p, _ in j) == 2 * N and sum(p == "noise" for p, _ in j) == 2 * N
    assert all("@replay|" in k for p, k in j if p == "noise") and all("@" not in k for p, k in j if p == "scored")
    with pytest.raises(SystemExit, match="already run"):
        R.final_test_run()
    assert len(fake.requests) == 4 * N


@pytest.mark.parametrize("failure", ["slow", "error"])
def test_timeout_or_error_stops_at_once_and_resume_does_not_resend_silently(world, monkeypatch, failure):
    fake, tmp = world
    monkeypatch.setattr(C, "REQUEST_TIMEOUT_S", 0.5)
    fake.on_call = {7: failure}
    with pytest.raises(Exception) as e:
        R.final_test_run()
    assert not isinstance(e.value, SystemExit) or "stopped" not in str(e.value)
    time.sleep(1.2)                                           # let the slow answer finish on the server side
    assert len(fake.requests) == 7 and answers(tmp) == 6       # stopped at the failing call; nothing queued
    with pytest.raises(SystemExit, match="sent without an answer"):
        R.final_test_run()                                    # a plain resume refuses to send it again
    assert len(fake.requests) == 7
    R.final_test_run(resend_unanswered=True)
    counts = fake.counts()
    assert len(counts) == 2 * N and len(fake.requests) == 4 * N + 1 == answers(tmp) + 1
    assert sorted(counts.values()) == [2] * (2 * N - 1) + [3]  # only the unanswered one went twice in the scored pass
    j = journal(tmp)
    assert len(j) == 4 * N + 1 and len(set(j)) == 4 * N


def test_threads_never_queue_more_than_the_free_slots(world):
    fake, tmp = world
    fake.on_call = {3: "sigint"}
    items = [(i, RECORDS[i], RECORDS[i]["narrative"]) for i in IDS]
    with pytest.raises(SystemExit, match="stopped on request"):
        R.run("v1", items, label="dev", workers=3)
    assert len(fake.requests) <= 3 + 2 and len(fake.requests) == answers(tmp)   # every sent request answered and kept
    R.run("v1", items, label="dev", workers=3)
    assert len(fake.requests) == N == answers(tmp) and set(fake.counts().values()) == {1}


def test_dry_run_reports_unanswered_requests_and_sends_nothing(world, capsys):
    fake, tmp = world
    R.final_test_run(dry_run=True)
    plan = json.loads(capsys.readouterr().out.split("dry run")[0])
    assert plan["sent_without_answer"] == 0 and fake.requests == []
