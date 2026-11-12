"""Judge v1 and v2 on three hosted Anthropic models: docs/analysis_plan_models.md, Amendment 2.

Added after the held-out run, after the blind labels were locked and while the local models of Amendment 1 were being
run. Sends only DEV and labelled-sample complaints and refuses any held-out ID (the local runner's item sets and
guard). Every request is built from the frozen messages and JSON schema of judge v1 or v2 (prompts.py), mapped to the
Messages API as Amendment 2 fixes it. Its cache, journal, run log and spend journal live under results/models/hosted/;
nothing is written to results/cache/, results/run_log.jsonl, results/human_labels/ or the local extension's files.

The API key is read from the environment variable ANTHROPIC_API_KEY when a request is sent, and only then. It is
sent in one header that is never forwarded on a redirect (redirects are not followed at all), and never written to a
file, a log line or an error message: errors are rebuilt from the response body with the key and anything shaped like
a key removed, and the original exception (which holds the request) is dropped.

python -m judge.hosted --verify                          # the checks `make check` runs; no network
python -m judge.hosted --estimate                        # projected cost of every model from the inputs; no network
python -m judge.hosted --run MODEL                       # one model's four cells, in the registered order
python -m judge.hosted --run MODEL --resend-unanswered   # after an interruption: one resend of each served request
python -m judge.hosted --close-incomplete MODEL          # only on a condition of section 9, checked in the journal
python -m judge.hosted --break-stale-lock                # after a crash: removes run.lock if its process is gone

With the key unset, `--run MODEL` is a dry run: every check of section 12 runs, then it stops with nothing sent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

from . import config as C
from . import models as X
from .models import Refusal, _append, _jsonl, key, now, split_key
from .prompts import build_messages, visible_narrative

DIR = C.RESULTS / "models" / "hosted"
CACHE_FILE = DIR / "responses.jsonl"
JOURNAL = DIR / "sent.jsonl"
RUN_LOG = DIR / "run_log.jsonl"
SPEND = DIR / "spend.jsonl"
REGISTRATION = DIR / "registration.json"
LOCK = DIR / "run.lock"

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2026-10-21"          # the anthropic-version header; temperature, top_p and top_k are not sent
KEY_ENV = "ANTHROPIC_API_KEY"
INFERENCE_GEO = "global"            # sent with every request: the routing the registered prices are for
TIMEOUT_S = 600
MIN_INTERVAL_S = 1.0                # pacing between requests; affects no answer

# run order = the only order of every table of the arm (Amendment 2, section 2)
MODELS = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")
COMPARED = ("claude-haiku-5-5", "claude-sonnet-5-5")    # reasoning off: the comparison set (Amendment 2, section 3)
REFERENCE_ONLY = ("claude-opus-5-5",)                   # reasoning cannot be switched off: descriptive only
EFFORT = "low"
# the lowest thinking setting each model accepts at effort low (Amendment 2, section 5); None = field not sent
THINKING = {"claude-haiku-5-5": {"type": "disabled"}, "claude-sonnet-5-5": {"type": "between_tools"},
            "claude-opus-5-5": None}
MAX_TOKENS = {"claude-haiku-5-5": 512, "claude-sonnet-5-5": 512, "claude-opus-5-5": 4096}
# USD per million tokens, Claude API, inference_geo "global", prompts up to 100,000 tokens (Amendment 2, section 2)
PRICES = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.10},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
}
HAIKU_PRICE_LIMIT = 100_000          # above this many prompt tokens Haiku 5.5 is priced higher; refused here
CAP_USD = 15.00
FORMAT_ALLOWANCE = 2_000             # worst case: tokens for the format instructions the API adds to each request
FORMAT_PROJECTED = 500               # projection: the same, as an assumption (not documented, not measured)
PROJECTED_BYTES_PER_TOKEN = 3        # projection: input tokens per request
PROJECTED_OUTPUT = 512               # projection: output tokens per request

# Amendment 2, section 9
RETRY_DELAYS_S = (30, 60, 120)                      # within a run: at most three retries of one send
RETRY_AFTER_MAX_S = 300
MAX_ERRORS_PER_MODEL = 60                           # waited-out error answers only; then the model may be closed
SPEND_CAP_CODE = "enforced_spend_limit_reached"     # a 429 that will not pass by waiting: an account condition
ACCOUNT_STATUSES = (401, 402)                       # the key, billing: account conditions, never a reason to close
ACCOUNT_PHRASES = ("credit balance", "api usage limits")   # a 400 about the account (credit, Console limit)
REPEAT_GAP_S = 24 * 3600                            # a rejection closes a model only if repeated this far apart
NOT_SERVED = ("transient", "rejected", "account")   # journal kinds of an error answer: the send is not used up
COUNTED_ERRORS = ("transient",)                     # the bound counts 429 (not the spend cap), 529 and 5xx only
VERDICT_WORDS = {"rejected": "the request was rejected",
                 "account": "an account condition (the key, the credit balance or a spend limit)",
                 "transient": "a rate limit, an overload or a server error, after its retries"}
DONE_REASON = {"end_turn": "stop", "max_tokens": "length"}     # every other stop_reason is kept as it is
KEY_PATTERN = re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}")
# files that must equal the published code commit of this arm when a run starts
PROTECTED = ("docs/analysis_plan_models.md", "src/judge/hosted.py", "src/judge/hosted_analysis.py",
             "src/judge/models.py", "src/judge/models_analysis.py", "src/judge/prompts.py", "src/judge/llm.py",
             "src/judge/judgments.py", "src/judge/config.py", "src/judge/metrics.py", "src/judge/data.py",
             "src/judge/label.py", "src/judge/baselines.py", "src/judge/freeze.py", "Makefile",
             "tests/test_hosted.py")


def log_event(event: dict) -> None:
    _append(RUN_LOG, {"at": now(), **event})


# ------------------------------------------------------------------ the key
def scrub(text) -> str:
    """Text with the key in the environment, and anything shaped like a key, removed."""
    s = str(text)
    k = os.environ.get(KEY_ENV)
    if k:
        s = s.replace(k, "[redacted]")
    return KEY_PATTERN.sub("[redacted]", s)


class HostedError(Exception):
    """A failed request, rebuilt without the request (no headers, no key)."""

    def __init__(self, status: int | None, message: str, request_id: str | None = None, etype: str | None = None,
                 maybe_billed: bool = False, retry_after: float | None = None, usage: dict | None = None,
                 code: str | None = None, date: str | None = None):
        self.status, self.message, self.request_id, self.etype = status, scrub(message)[:500], request_id, etype
        self.maybe_billed, self.retry_after, self.usage, self.code = maybe_billed, retry_after, usage, code
        self.date = date                         # the server's `date` header: the clock the 24-hour rule uses
        super().__init__(f"HTTP {status}: {self.message}" if status else self.message)


# ------------------------------------------------------------------ requests
def schema_for_api(schema: dict) -> dict:
    """The frozen schema with "additionalProperties": false on the object, which the Messages API requires for
    structured outputs; nothing else changes."""
    return {**schema, "additionalProperties": False}


def build_body(tag: str, version: str, rec: dict, narrative: str) -> dict:
    messages, schema = build_messages(version, rec, narrative)
    assert [m["role"] for m in messages] == ["system", "user"]
    body = {"model": tag, "max_tokens": MAX_TOKENS[tag], "system": messages[0]["content"],
            "messages": [{"role": "user", "content": messages[1]["content"]}],
            "output_config": {"effort": EFFORT, "format": {"type": "json_schema", "schema": schema_for_api(schema)}},
            "inference_geo": INFERENCE_GEO}
    if THINKING[tag] is not None:
        body["thinking"] = THINKING[tag]
    return body


def request_hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def body_bytes(body: dict) -> int:
    return len(json.dumps(body, ensure_ascii=False).encode("utf-8"))


def input_upper(body: dict) -> int:
    """An upper bound on the input tokens billed for this body: no token covers less than one byte, plus an
    allowance for the format instructions the API adds. Checked against every answer's usage."""
    return body_bytes(body) + FORMAT_ALLOWANCE


def geo_factor(usage: dict) -> float:
    return 1.1 if (usage or {}).get("inference_geo") == "us" else 1.0


def cost(tag: str, usage: dict) -> float:
    p = PRICES[tag]
    return geo_factor(usage) * (p["input"] * (usage.get("input_tokens") or 0)
                                + p["output"] * (usage.get("output_tokens") or 0)
                                + p["cache_write"] * (usage.get("cache_creation_input_tokens") or 0)
                                + p["cache_read"] * (usage.get("cache_read_input_tokens") or 0)) / 1e6


def worst_case(tag: str, body: dict) -> float:
    """The most one request can cost, at the US price (1.1x), so that an answer reporting US routing fits too."""
    p = PRICES[tag]
    return 1.1 * (p["input"] * input_upper(body) + p["output"] * body["max_tokens"]) / 1e6


def projected(tag: str, body: dict) -> float:
    p = PRICES[tag]
    tokens = math.ceil(body_bytes(body) / PROJECTED_BYTES_PER_TOKEN) + FORMAT_PROJECTED
    return (p["input"] * tokens + p["output"] * PROJECTED_OUTPUT) / 1e6


# ------------------------------------------------------------------ transport
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None                               # a 3xx becomes an HTTPError: an interruption, nothing forwarded


def _open(req, timeout):
    return urllib.request.build_opener(_NoRedirect).open(req, timeout=timeout)


def _valid_usage(u) -> dict | None:
    if isinstance(u, dict) and all(isinstance(u.get(k), int) for k in ("input_tokens", "output_tokens")):
        return u
    return None


def check_shape(resp) -> None:
    """Raise unless the answer has the fields this runner reads, with the types it expects."""
    ok = (isinstance(resp, dict) and isinstance(resp.get("content"), list)
          and all(isinstance(b, dict) and isinstance(b.get("type"), str) for b in resp["content"])
          and all(isinstance(b.get("text"), str) for b in resp["content"] if b["type"] == "text")
          and isinstance(resp.get("stop_reason"), str) and isinstance(resp.get("model"), str)
          and _valid_usage(resp.get("usage")) is not None
          and (resp.get("stop_details") is None or isinstance(resp.get("stop_details"), dict)))
    if not ok:
        raise ValueError("unexpected answer shape")


class AnthropicTransport:
    """POST /v1/messages with urllib (no new dependency, no retry hidden in a client, no redirect followed). Tests
    replace it, or replace _open underneath it."""

    def send(self, body: dict) -> tuple[dict, dict]:
        api_key = os.environ.get(KEY_ENV)
        if not api_key:
            raise HostedError(None, f"{KEY_ENV} is not set; nothing was sent.")
        req = urllib.request.Request(API_URL, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"anthropic-version": API_VERSION, "content-type": "application/json"})
        req.add_unredirected_header("x-api-key", api_key)
        del api_key
        failure, resp = None, None
        try:
            r = _open(req, TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            failure = _http_failure(exc)
        except BaseException as exc:             # noqa: BLE001 - timeout, connection, interrupt: may have been billed
            failure = HostedError(None, f"no answer: {type(exc).__name__}: {exc}", maybe_billed=True)
        else:
            rid = r.headers.get("request-id") if r.headers else None
            try:
                with r:
                    resp = json.loads(r.read().decode("utf-8"))
                check_shape(resp)
                return resp, {"request-id": rid, "date": r.headers.get("date") if r.headers else None}
            except BaseException as exc:         # noqa: BLE001 - an answer arrived and could not be read: billed
                usage = _valid_usage(resp.get("usage")) if isinstance(resp, dict) else None
                failure = HostedError(200, f"unreadable answer: {type(exc).__name__}", rid, maybe_billed=True,
                                      usage=usage)
        finally:
            del req
        raise failure                            # raised outside the except blocks: no chained original


def _http_failure(exc: urllib.error.HTTPError) -> HostedError:
    """An error answer. It is not charged: the docs do not say that error answers are billed (section 7)."""
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:                            # noqa: BLE001
        raw = ""
    code = None
    try:
        err = json.loads(raw)
        etype, message, rid = err["error"]["type"], err["error"]["message"], err.get("request_id")
        code = ((err["error"].get("details") or {}).get("error_code"))
    except Exception:                            # noqa: BLE001
        etype, message, rid = None, raw[:300] or str(exc.reason), None
    hdrs = exc.headers
    rid = rid or (hdrs.get("request-id") if hdrs else None)
    try:
        retry_after = float(hdrs.get("retry-after")) if hdrs and hdrs.get("retry-after") else None
    except ValueError:
        retry_after = None
    return HostedError(exc.code, str(message), rid, etype, retry_after=retry_after, code=code,
                       date=hdrs.get("date") if hdrs else None)


def is_account(status, message: str, code=None) -> bool:
    """Section 9: a condition of the account, not of the request: 401, 402, the 429 of a monthly spend cap, or a 400
    that says the credit balance is too low or the API usage limits were reached. It never justifies a close."""
    return (status in ACCOUNT_STATUSES or (status == 429 and code == SPEND_CAP_CODE)
            or (status == 400 and any(p in (message or "").lower() for p in ACCOUNT_PHRASES)))


def classify(exc: BaseException) -> tuple[str, str]:
    """Amendment 2, section 9. 'transient': 429 (except the spend cap) and every 5xx, 529 included, waited out.
    'account': an account condition (is_account), or nothing sent at all (no key). 'rejected': any other 3xx or 4xx.
    Each of these stops the run when it is not waited out, and none removes anything: a model is never dropped.
    'lost': anything else (a timeout, a lost connection, an interrupt, an answer that cannot be read)."""
    if not isinstance(exc, HostedError):
        return "lost", scrub(f"{type(exc).__name__}: {exc}")[:500]
    status = exc.status
    if is_account(status, exc.message, exc.code):
        return "account", exc.message
    if status is not None and (status == 429 or status >= 500):
        return "transient", exc.message
    if status is not None and 300 <= status < 500:
        return "rejected", exc.message
    if status is None and not exc.maybe_billed:
        return "account", exc.message             # nothing was sent (no key)
    return "lost", exc.message


def answer_record(tag: str, version: str, item_id: str, body: dict, rhash: str, resp: dict, headers: dict,
                  requested_at: str, wall: float) -> dict:
    """The cache record, in the fields models_analysis reads (content, done_reason, prompt_eval_count, eval_count,
    wall_s), plus what the API returned."""
    usage = resp["usage"]
    blocks = resp["content"]
    stop = resp["stop_reason"]
    details = usage.get("output_tokens_details") or {}
    return {
        "key": key(tag, version, item_id), "model": tag, "prompt_version": version, "item_id": item_id,
        "request_sha256": rhash, "api_version": API_VERSION, "response_model": resp["model"],
        "request_id": headers.get("request-id"), "message_id": resp.get("id"),
        "max_tokens": body["max_tokens"], "effort": body["output_config"]["effort"],
        "thinking_sent": body.get("thinking"), "inference_geo_sent": body.get("inference_geo"),
        "inference_geo": usage.get("inference_geo"), "requested_at": requested_at, "wall_s": round(wall, 3),
        "server_date": headers.get("date"),
        "content": "".join(b["text"] for b in blocks if b["type"] == "text"), "stop_reason": stop,
        "stop_details": resp.get("stop_details"), "done_reason": DONE_REASON.get(stop, stop),
        "block_types": [b["type"] for b in blocks], "usage": usage,
        "prompt_eval_count": sum(usage.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens",
                                                             "cache_read_input_tokens")),
        "eval_count": usage["output_tokens"], "thinking_tokens": details.get("thinking_tokens") or 0,
        "logprobs": None, "thinking": None,
    }


# ------------------------------------------------------------------ state and spend
def spend_state(spend: list[dict] | None = None) -> dict:
    """Computed spend (usage-based cost of answers plus charges) and the reservations not yet settled. A reservation
    is settled by the answer or charge that names it, or by a release (an error answer, not charged); so a crash
    after any of these is never charged a second time."""
    spend = _jsonl(SPEND) if spend is None else spend
    settled = {e["id"] for e in spend if e.get("event") == "released"} | {
        e["reservation"] for e in spend if e.get("event") in ("answer", "charged") and e.get("reservation")}
    open_ = [e for e in spend if e.get("event") == "reserved" and e["id"] not in settled]
    computed = sum(e.get("cost_usd") or 0.0 for e in spend if e.get("event") in ("answer", "charged"))
    return {"computed": computed, "open": open_, "committed": computed + sum(e["cost_usd"] for e in open_)}


def spent() -> float:
    """What the cap is checked against: computed spend plus open reservations."""
    return spend_state()["committed"]


def state() -> dict:
    cache = {r["key"]: r for r in _jsonl(CACHE_FILE)}
    journal = _jsonl(JOURNAL)
    status: dict[str, str] = {}
    for e in _jsonl(RUN_LOG):
        if e.get("event") == "model_finished":
            status[e["model"]] = "complete"
        elif e.get("event") == "closed_incomplete":
            status[e["model"]] = e["reason"]
    return {"cache": cache, "journal": journal, "status": status, "spent": spent()}


def expected_keys(tag: str, sets: dict) -> set:
    return {key(tag, v, i) for v, s in X.CELLS for i in sets[s]}


def _seconds(at: str) -> float:
    from datetime import datetime
    return datetime.fromisoformat(at.replace("Z", "+00:00")).timestamp()


def _server_seconds(date: str | None) -> float | None:
    """The server's `date` header (RFC 9110) in seconds, or None if it is missing or cannot be read."""
    from email.utils import parsedate_to_datetime
    try:
        return parsedate_to_datetime(date).timestamp() if date else None
    except (TypeError, ValueError):
        return None


def repeated_rejection(tag: str, journal: list[dict], run_log: list[dict]) -> dict | None:
    """Section 9, condition (c): the model's last run stopped on a rejection (not an account condition), and the same
    request received the same HTTP status and error type at least 24 hours apart by the server's `date` headers (not
    this machine's clock). Returns that last stop, or None."""
    runs = [e for e in run_log if e.get("model") == tag and e.get("event") in ("run_started", "run_stopped")]
    last = runs[-1] if runs and runs[-1]["event"] == "run_stopped" else None
    if not last or last.get("verdict") != "rejected":
        return None
    same = [e for e in journal if e.get("key") == last["key"] and e.get("kind") == "rejected"
            and e.get("status") == last.get("status") and e.get("error_type") == last.get("error_type")]
    times = [t for t in (_server_seconds(e.get("server_date")) for e in same) if t is not None]
    if len(times) >= 2 and max(times) - min(times) >= REPEAT_GAP_S:
        return last
    return None


def attempts(entries: list[dict]) -> list[dict]:
    """The sends of one request (send, resend, retry), each marked served unless the next entry for that request is
    an error answer (transient, rejected or account). Amendment 2, section 9."""
    out, base = [], "send"
    for n, e in enumerate(entries):
        if e.get("kind") in ("send", "resend", "retry"):
            base = e["kind"] if e["kind"] != "retry" else base       # a retry belongs to the send it retries
            nxt = entries[n + 1] if n + 1 < len(entries) else {}
            out.append({**e, "base": base, "served": nxt.get("kind") not in NOT_SERVED})
    return out


def resend_served(entries: list[dict]) -> bool:
    """The request's one resend was served (section 9): it is never sent again."""
    return any(a["served"] and a["base"] == "resend" for a in attempts(entries))


def by_key(journal: list[dict]) -> dict:
    out: dict[str, list[dict]] = {}
    for e in journal:
        out.setdefault(e["key"], []).append(e)
    return out


def error_count(journal: list[dict], tag: str) -> int:
    """Error answers that the per-model bound counts: the waited-out ones only. Rejections and account conditions
    never count, so re-running a rejected request cannot reach a close before the 24-hour rule (section 9)."""
    return sum(1 for e in journal if e.get("kind") in COUNTED_ERRORS and split_key(e["key"])[0] == tag)


def load_registration() -> dict:
    if not REGISTRATION.exists():
        raise Refusal(f"results/models/hosted/{REGISTRATION.name} is missing; refusing.")
    reg = json.loads(REGISTRATION.read_text())
    need = ("repository", "code_commit", "code_published_at")
    missing = [k for k in need if not reg.get(k)]
    if missing:
        raise Refusal(f"results/models/hosted/{REGISTRATION.name} lacks {', '.join(missing)}: the arm's code "
                      "commit and the time GitHub records for its publication must be recorded before the first "
                      "request.")
    return reg


def github_main_head() -> str:
    """The commit GitHub shows as main, for the repository named in this arm's registration."""
    slug = load_registration()["repository"]
    out = subprocess.run(["gh", "api", f"repos/{slug}/git/refs/heads/main", "--jq", ".object.sha"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


# ------------------------------------------------------------------ before a model (called with the lock held)
def preflight(tag: str, published_head=None) -> dict:
    if tag not in MODELS:
        raise Refusal(f"{tag} is not a model of Amendment 2; no model is ever added or replaced.")
    if X.registered_part_sha256() != X.REGISTERED_SHA256:
        raise Refusal("lines 1-149 of docs/analysis_plan_models.md differ from the registered text; refusing.")
    local = X.state()["status"]
    open_local = [t for t, _ in X.MODELS if not local.get(t)]
    if open_local:
        raise Refusal("the local models are not all closed (" + ", ".join(open_local) + "); the hosted arm starts "
                      "only after the last of them.")
    reg = load_registration()
    try:
        changed = X.git("diff", "--name-only", reg["code_commit"], "--", *PROTECTED).split()
    except Exception:                            # noqa: BLE001
        raise Refusal(f"the published code commit {reg['code_commit']} is not in this repository; refusing.") from None
    if changed:
        raise Refusal("these files differ from the arm's published code commit: " + ", ".join(changed))
    dirty = [ln[3:] for ln in X.git("status", "--porcelain").splitlines()]
    outside = [p for p in dirty if not p.startswith("results/models/hosted/")]
    if outside:
        raise Refusal("uncommitted changes outside results/models/hosted/: " + ", ".join(outside))
    st = state()
    started = any(split_key(e["key"])[0] == tag for e in st["journal"])
    if dirty and not started:
        raise Refusal("results/models/hosted/ has uncommitted changes: commit and publish the previous model's "
                      "results before the next model starts.")
    head = X.git("rev-parse", "HEAD").strip()
    if (published_head or github_main_head)() != head:
        raise Refusal(f"HEAD {head[:12]} is not the commit GitHub shows as main: publish before sending.")
    if st["status"].get(tag):
        raise Refusal(f"{tag} is closed ({st['status'][tag]}); a closed model is never run again.")
    for earlier in MODELS[: MODELS.index(tag)]:
        if not st["status"].get(earlier):
            raise Refusal(f"{earlier} comes before {tag} in the registered order and is not closed.")
    if any(e.get("event") == "budget_stop" for e in _jsonl(RUN_LOG)):
        raise Refusal("the arm stopped at its cap; no request of this arm is sent after a budget stop.")
    from .label import is_locked
    if not is_locked("owner"):
        raise Refusal("the blind labels are not locked; refusing.")
    if not os.environ.get(KEY_ENV):
        raise Refusal(f"{KEY_ENV} is not set in the environment; nothing was sent.")
    sets, test = X.item_sets()
    return {"sets": sets, "test": test, "registration": reg}


def work_list(tag: str, sets: dict, records: dict) -> list[tuple]:
    return X.work_list(tag, sets, records)


def estimate(tag: str, sets: dict, records: dict, cache: dict | None = None) -> dict:
    """Projected and worst-case cost of the requests of `tag` that have no answer yet. No network."""
    cache = cache or {}
    todo = [(v, s, i, r, n) for v, s, i, r, n in work_list(tag, sets, records) if key(tag, v, i) not in cache]
    bodies = [build_body(tag, v, r, n) for v, s, i, r, n in todo]
    return {"model": tag, "requests": len(bodies),
            "input_tokens_projected": sum(math.ceil(body_bytes(b) / PROJECTED_BYTES_PER_TOKEN) + FORMAT_PROJECTED
                                          for b in bodies),
            "input_tokens_upper": sum(input_upper(b) for b in bodies),
            "projected_usd": round(sum(projected(tag, b) for b in bodies), 6),
            "worst_case_usd": round(sum(worst_case(tag, b) for b in bodies), 6)}


# ------------------------------------------------------------------ the lock
def acquire_lock() -> None:
    DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise Refusal(f"another run holds {LOCK.name}. If no run is going on (a crash), run "
                      "`python -m judge.hosted --break-stale-lock`.") from None
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"pid": os.getpid(), "at": now()}) + "\n")


def release_lock() -> None:
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def break_stale_lock() -> str:
    """Section 12: removes run.lock only when the process that wrote it no longer exists, and logs it. The open
    reservation of a crashed run stays in the spend journal and is charged at the next run."""
    if not LOCK.exists():
        return "no run.lock"
    try:
        held = json.loads(LOCK.read_text())
        pid = int(held["pid"])
    except Exception:                            # noqa: BLE001
        raise Refusal(f"{LOCK.name} cannot be read; remove it by hand only after checking that no run is going on.") \
            from None
    if _alive(pid):
        raise Refusal(f"{LOCK.name} belongs to process {pid}, which is still running; nothing was removed. If that "
                      "process is another program (its number was reused), remove run.lock by hand only after "
                      "checking that no run is going on.")
    LOCK.unlink()
    log_event({"event": "stale_lock_removed", "lock": held})
    return f"removed {LOCK.name} of process {pid}, which no longer runs"


# ------------------------------------------------------------------ closing
def close_model(tag: str, reason: str) -> str:
    """Close a model as incomplete with its reason, and write its spend totals. The reason is in plain words: "not
    run: ..." when the model has no answer, otherwise "incomplete (k of 600 answered): ..."."""
    st = state()
    answered = sum(1 for k in st["cache"] if split_key(k)[0] == tag)
    n = len(expected_keys(tag, X.item_sets()[0]))
    if not reason.startswith("not run"):
        reason = f"not run: {reason}" if answered == 0 else f"incomplete ({answered} of {n} answered): {reason}"
    log_event({"event": "closed_incomplete", "model": tag, "answers": answered, "reason": reason})
    write_model_total(tag)
    return reason


def write_model_total(tag: str) -> None:
    spend = _jsonl(SPEND)
    mine = [e for e in spend if e.get("model") == tag and e.get("event") in ("answer", "charged")]
    answers = list({e["key"]: e for e in mine if e["event"] == "answer"}.values())      # the last per request
    _append(SPEND, {"at": now(), "event": "model_total", "model": tag, "answers": len(answers),
                    "charged_without_answer": len(mine) - len(answers),
                    "routed_elsewhere": sum(1 for e in answers if e.get("inference_geo") not in (INFERENCE_GEO, None)),
                    **{f: sum(e.get(f) or 0 for e in answers) for f in ("input_tokens", "output_tokens",
                                                                       "thinking_tokens")},
                    "cost_usd": round(sum(e["cost_usd"] for e in mine), 6),
                    "computed_total_usd": round(spend_state()["computed"], 6)})


def reconcile() -> None:
    """Charge every reservation that was never settled (a crash after the request was journaled). Called with the
    lock held, before a run and before a close. The charge names its reservation, which settles it."""
    for e in spend_state()["open"]:
        _append(SPEND, {"at": now(), "event": "charged", "model": e["model"], "key": e["key"], "request_id": None,
                        "cost_usd": e["cost_usd"], "running_total_usd": round(spend_state()["computed"]
                                                                              + e["cost_usd"], 6),
                        "note": "reservation of a request whose outcome was lost (crash)", "reservation": e["id"]})
        log_event({"event": "reservation_charged", "model": e["model"], "key": e["key"], "usd": e["cost_usd"]})


# ------------------------------------------------------------------ the run
def run_model(tag: str, transport=None, resend_unanswered: bool = False, published_head=None, records=None,
              sleep=time.sleep) -> dict:
    acquire_lock()                               # before any state is read (section 12)
    try:
        pre = preflight(tag, published_head)
        return _run(tag, transport or AnthropicTransport(), resend_unanswered, records, pre, sleep)
    finally:
        release_lock()


def _run(tag, transport, resend_unanswered, records, pre, sleep) -> dict:
    from .data import load_records

    reconcile()
    st = state()
    sets, test = pre["sets"], pre["test"]
    allowed = set(sets["dev"]) | set(sets["labelled"])
    records = records if records is not None else load_records()
    journal = by_key(st["journal"])
    served_unanswered = [k for k, es in journal.items() if split_key(k)[0] == tag and k not in st["cache"]
                         and any(a["served"] for a in attempts(es))]
    if served_unanswered and not resend_unanswered:
        raise Refusal(f"{len(served_unanswered)} request(s) of {tag} may have been served without an answer "
                      "arriving: " + ", ".join(sorted(served_unanswered)[:10]) + ". Resume with --resend-unanswered "
                      "(each is sent once more, and the report lists it).")
    errors = error_count(st["journal"], tag)
    if errors >= MAX_ERRORS_PER_MODEL:
        raise Refusal(f"{tag} has had {errors} error answers, the registered maximum; close it with "
                      f"--close-incomplete {tag}.")
    committed = st["spent"]
    est = estimate(tag, sets, records, st["cache"])
    _append(SPEND, {"at": now(), "event": "estimate", **est, "spent_before_usd": round(committed, 6),
                    "cap_usd": CAP_USD})
    if committed + est["projected_usd"] > CAP_USD:
        log_event({"event": "budget_stop", "model": tag, "spent_usd": round(committed, 6),
                   "when": "before the model", **est})
        close_model(tag, "not run: projection exceeded remaining budget" if not any(
            split_key(k)[0] == tag for k in st["cache"]) else
            f"stopped by the {CAP_USD:.0f} USD cap: the projection of its remaining requests did not fit")
        raise Refusal(f"{tag}: spent {committed:.4f} USD plus the projected {est['projected_usd']:.4f} USD exceeds the "
                      f"cap of {CAP_USD:.2f} USD; nothing was sent, and the model is closed.")
    log_event({"event": "run_started", "model": tag, "resend_unanswered": resend_unanswered})
    done, last = 0, [0.0]

    def record_spend(entry: dict) -> None:
        _append(SPEND, {"at": now(), "model": tag, **entry,
                        "running_total_usd": round(spend_state()["computed"] + (entry.get("cost_usd") or 0), 6)})

    for version, split, item_id, rec, narrative in work_list(tag, sets, records):
        k = key(tag, version, item_id)
        if k in st["cache"]:
            continue
        es = journal.get(k, [])
        served = [a for a in attempts(es) if a["served"]]
        if served:
            if resend_served(es):
                continue                      # its one resend was served: stays unanswered (incomplete)
            kind = "resend"
        else:
            kind = "send"
        retries = 0
        while True:
            body = build_body(tag, version, rec, narrative)
            rhash = request_hash(body)
            if tag == "claude-haiku-5-5" and input_upper(body) > HAIKU_PRICE_LIMIT:
                raise Refusal(f"{k}: the prompt could exceed {HAIKU_PRICE_LIMIT} tokens, where other prices apply.")
            reserve = worst_case(tag, body)
            committed = spent()
            if committed + reserve > CAP_USD:
                log_event({"event": "budget_stop", "model": tag, "key": k, "spent_usd": round(committed, 6),
                           "next_worst_case_usd": round(reserve, 6)})
                close_model(tag, f"stopped by the {CAP_USD:.0f} USD cap: the next request's worst case did not fit")
                raise Refusal(f"budget: spent {committed:.4f} USD; the next request could cost up to {reserve:.4f} "
                              f"USD, over the cap of {CAP_USD:.2f} USD. Nothing more of this arm is sent; {tag} is "
                              "closed as incomplete.")
            guard_item(item_id, allowed, test)
            wait = MIN_INTERVAL_S - (time.time() - last[0])
            if wait > 0:
                sleep(wait)
            entry = {"at": now(), "key": k, "kind": kind if retries == 0 else "retry", "request_sha256": rhash,
                     **({"retry": retries} if retries else {})}
            rid_ = f"{entry['at']}|{k}"
            _append(SPEND, {"at": entry["at"], "event": "reserved", "id": rid_, "model": tag, "key": k,
                            "cost_usd": round(reserve, 8)})       # write-ahead: charged unless settled below
            _append(JOURNAL, entry)
            es.append(entry)
            journal[k] = es
            requested_at = now()
            t0 = last[0] = time.time()
            try:
                resp, headers = transport.send(body)
            except BaseException as exc:          # noqa: BLE001 - every failure is classified and recorded
                verdict, message = classify(exc)
                he = exc if isinstance(exc, HostedError) else None
                rid = he.request_id if he else None
                if verdict == "lost":             # may have been served and billed: charged
                    amount = cost(tag, he.usage) if he and he.usage else reserve
                    record_spend({"event": "charged", "key": k, "request_id": rid, "cost_usd": round(amount, 8),
                                  "note": "usage of an answer that could not be used" if he and he.usage
                                  else "worst case of a request that may have been billed", "reservation": rid_})
                else:                             # an error answer: not charged, the reservation is released
                    _append(SPEND, {"at": now(), "event": "released", "id": rid_})
                    err = {"at": now(), "key": k, "kind": verdict, "status": he.status, "error_type": he.etype,
                           "error_code": he.code, "message": message, "request_id": rid, "server_date": he.date}
                    _append(JOURNAL, err)
                    es.append(err)
                if verdict == "transient":
                    errors += 1
                    if retries < len(RETRY_DELAYS_S) and errors < MAX_ERRORS_PER_MODEL:
                        delay = min(max(RETRY_DELAYS_S[retries], he.retry_after or 0), RETRY_AFTER_MAX_S)
                        retries += 1
                        log_event({"event": "transient_wait", "model": tag, "key": k, "status": he.status,
                                   "seconds": delay})
                        sleep(delay)
                        continue
                log_event({"event": "run_stopped", "model": tag, "key": k, "verdict": verdict,
                           "status": he.status if he else None, "error_type": he.etype if he else None,
                           "reason": message, "request_id": rid})
                if verdict == "lost":
                    raise Refusal(f"interrupted at {k} ({message[:200]}); it may have been served and is charged. "
                                  f"Nothing is dropped; resume with --run {tag} --resend-unanswered.") from None
                detail = f"HTTP {he.status} {he.etype}: {message[:200]}" if he and he.status else message[:200]
                raise Refusal(f"stopped at {k}: {VERDICT_WORDS[verdict]} ({detail}). The request was not served and "
                              f"keeps its sends; nothing is dropped. Resume with --run {tag} once the cause is "
                              "fixed.") from None
            r = answer_record(tag, version, item_id, body, rhash, resp, headers, requested_at, time.time() - t0)
            c = cost(tag, r["usage"])
            if not r["response_model"].startswith(tag):   # not an answer of this model: charged, not cached
                record_spend({"event": "charged", "key": k, "request_id": r["request_id"], "cost_usd": round(c, 8),
                              "note": f"answer served by {r['response_model']}", "reservation": rid_})
                _append(JOURNAL, {"at": now(), "key": k, "kind": "serving_mismatch", "model_answered":
                                  r["response_model"], "request_id": r["request_id"]})
                close_model(tag, f"served by {r['response_model']}")
                raise Refusal(f"the API answered {k} with model {r['response_model']}; {tag} is closed as "
                              "incomplete.")
            # the spend entry first: it settles the reservation, so a crash after it is never charged again, and a
            # crash before the cache line leaves a request that is resent, never an answer without a spend entry
            record_spend({"event": "answer", "key": k, "request_id": r["request_id"],
                          **{f: r["usage"].get(f) for f in ("input_tokens", "output_tokens",
                                                            "cache_creation_input_tokens", "cache_read_input_tokens",
                                                            "inference_geo")},
                          "thinking_tokens": r["thinking_tokens"], "cost_usd": round(c, 8), "reservation": rid_})
            _append(CACHE_FILE, r)
            st["cache"][k] = r
            done += 1
            print(f"[{tag} {version} {split}] {item_id} {r['wall_s']:.1f}s stop={r['stop_reason']} "
                  f"in={r['prompt_eval_count']} out={r['eval_count']} geo={r['inference_geo']} "
                  f"total={spend_state()['computed']:.4f} USD", flush=True)
            if r["prompt_eval_count"] > input_upper(body):
                close_model(tag, f"{k}: billed input above the bound the cap relies on")
                raise Refusal(f"{k}: {r['prompt_eval_count']} input tokens billed, above the bound {input_upper(body)} "
                              f"the cap relies on; {tag} is closed as incomplete.")
            break
    expected = expected_keys(tag, sets)
    missing = expected - set(st["cache"])
    if not missing:
        log_event({"event": "model_finished", "model": tag, "answers": len(expected)})
        write_model_total(tag)
        print(f"{tag}: all {len(expected)} answers are in ({spend_state()['computed']:.4f} USD so far). Commit and "
              "publish results/models/hosted/ before the next model.")
    else:                                         # the end of the work list: each missing request has had its one
        if not all(resend_served(journal.get(k, [])) for k in missing):                      # resend (section 9)
            raise Refusal(f"{tag}: a missing request has not had its one resend; nothing was closed.")
        close_model(tag, f"{len(missing)} request(s) unanswered after their one resend")
        print(f"{tag}: {len(missing)} answer(s) missing after their one resend; the model is closed as incomplete. "
              "Commit and publish results/models/hosted/ before the next model.")
    return {"model": tag, "new_answers": done, "missing": len(missing),
            "spent_usd": round(spend_state()["computed"], 6)}


def guard_item(item_id: str, allowed: set, test: set) -> None:
    """Called just before a request is journaled; the local runner's guard."""
    X.guard(item_id, allowed, test)


def close_incomplete(tag: str) -> str:
    """Section 9. A close is never the operator's choice: it is accepted only when the journal shows (a) a budget
    stop of the model, (b) the bound of error answers, or (c) the same rejection of the same request on runs at
    least 24 hours apart. An account condition never justifies a close. Takes the run lock, and charges any
    reservation a crash left open before it closes."""
    acquire_lock()
    try:
        reconcile()
        st = state()
        if st["status"].get(tag):
            raise Refusal(f"{tag} is already closed ({st['status'][tag]}).")
        run_log = _jsonl(RUN_LOG)
        if not any(split_key(e["key"])[0] == tag for e in st["journal"]):
            raise Refusal(f"{tag} has not started; there is nothing to close.")
        if any(e.get("event") == "budget_stop" and e.get("model") == tag for e in run_log):
            reason = f"stopped by the {CAP_USD:.0f} USD cap"
        elif error_count(st["journal"], tag) >= MAX_ERRORS_PER_MODEL:
            reason = "error-answer limit reached"
        else:
            last = repeated_rejection(tag, st["journal"], run_log)
            if last is None:
                raise Refusal(_why_not_closed(tag, run_log))
            reason = (f"the same request was rejected on runs at least 24 hours apart ({last['key']}: HTTP "
                      f"{last.get('status')} {last.get('error_type')}: {last.get('reason')})")
        return close_model(tag, reason)
    finally:
        release_lock()


def _why_not_closed(tag: str, run_log: list[dict]) -> str:
    runs = [e for e in run_log if e.get("model") == tag and e.get("event") in ("run_started", "run_stopped")]
    last = runs[-1] if runs and runs[-1]["event"] == "run_stopped" else None
    head = f"{tag} cannot be closed: "
    if last and last.get("verdict") == "account":
        return (head + f"its last run stopped on {VERDICT_WORDS['account']} (HTTP {last.get('status')} "
                f"{last.get('error_type')}: {last.get('reason')}), which never justifies a close. Fix it and resume "
                f"with --run {tag}.")
    if last and last.get("verdict") == "rejected":
        return (head + f"{last['key']} was rejected (HTTP {last.get('status')} {last.get('error_type')}), but not "
                "with the same status and error type on runs at least 24 hours apart. Resume with "
                f"--run {tag} once the cause is fixed, or after 24 hours.")
    return (head + "none of the conditions of section 9 holds (a budget stop, the error-answer limit, the same "
            f"rejection on runs at least 24 hours apart). Resume with --run {tag}.")


# ------------------------------------------------------------------ the checks of `make check`
def verify(records=None) -> list[str]:
    """Amendment 2, section 13. Reads files only."""
    problems = []
    if list(MODELS) != ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"] or \
            COMPARED + REFERENCE_ONLY != MODELS:
        problems.append("the hosted model list, its order or its comparison set differs from Amendment 2")
    if set(PRICES) != set(MODELS) or set(THINKING) != set(MODELS) or set(MAX_TOKENS) != set(MODELS):
        problems.append("prices, thinking settings or max_tokens do not cover exactly the registered models")
    if any(t in X.DIGESTS or t == X.REFERENCE[0] for t in MODELS):
        problems.append("a hosted model is also a local model")
    for p in sorted(DIR.glob("*")) if DIR.exists() else []:
        if p.is_file() and KEY_PATTERN.search(p.read_text(encoding="utf-8", errors="replace")):
            problems.append(f"{p.name} contains something shaped like an API key")
    journal, cache, spend = _jsonl(JOURNAL), _jsonl(CACHE_FILE), _jsonl(SPEND)
    run_log = _jsonl(RUN_LOG)
    for name, entries in (("sent.jsonl", journal), ("run_log.jsonl", run_log), ("spend.jsonl", spend)):
        times = [_seconds(e["at"]) for e in entries if e.get("at")]
        if any(b < a for a, b in zip(times, times[1:])):
            problems.append(f"{name}: its times are not in order (a clock moved back, or an edit)")
    if not journal and not cache:
        return problems
    from .data import load_records
    records = records if records is not None else load_records()
    sets, test = X.item_sets()
    allowed = set(sets["dev"]) | set(sets["labelled"])
    try:
        reg = load_registration()
        if min(_seconds(e["at"]) for e in journal) <= _seconds(reg["code_published_at"]):
            problems.append("the first request was journaled before the arm's code was published")
    except Refusal as e:
        problems.append(str(e))
    sent = by_key(journal)
    for k, entries in sent.items():
        tag, version, item = split_key(k)
        if tag not in MODELS or version not in X.VERSIONS:
            problems.append(f"journal: {k} is not a registered model and prompt")
        if item in test or item not in allowed:
            problems.append(f"journal: {k} is not a DEV or labelled-sample complaint")
        served = [a for a in attempts(entries) if a["served"]]
        if sum(a["base"] == "resend" for a in served) > 1 or sum(a["base"] == "send" for a in served) > 1:
            problems.append(f"journal: {k} was served more often than the rules allow")
        streak = 0
        for e in entries:
            if e.get("kind") in ("send", "resend"):
                streak = 0                         # a new send starts a new chain of retries
            elif e.get("kind") == "retry":
                streak += 1
                if streak > len(RETRY_DELAYS_S):
                    problems.append(f"journal: {k} was retried more often than the rules allow")
                    break
    for tag in MODELS:
        if error_count(journal, tag) > MAX_ERRORS_PER_MODEL:
            problems.append(f"journal: {tag} has more error answers than the rules allow")
    answers = {e["key"]: e for e in spend if e.get("event") == "answer"}
    closed = {e["model"]: e.get("reason") or "" for e in run_log if e.get("event") == "closed_incomplete"}
    for e in spend:                                # an answer from another model closes its model (section 9)
        if e.get("event") == "charged" and str(e.get("note") or "").startswith("answer served by") and \
                "served by" not in closed.get(e.get("model"), ""):
            problems.append(f"spend: {e.get('key')} was answered by another model, and {e.get('model')} is not "
                            "closed for it")
    seen = set()
    for r in cache:
        k = r["key"]
        tag, version, item = split_key(k)
        if k in seen:
            problems.append(f"cache: {k} appears twice")
        seen.add(k)
        if r.get("model") != tag or tag not in MODELS:
            problems.append(f"cache: {k} records model {r.get('model')}")
            continue
        if not str(r.get("response_model") or "").startswith(tag):
            problems.append(f"cache: {k} was answered by {r.get('response_model')}")
        if item in test or item not in records:
            problems.append(f"cache: {k} concerns a held-out or unknown complaint")
            continue
        body = build_body(tag, version, records[item], visible_narrative(records[item]["narrative"]))
        if (r.get("request_sha256") != request_hash(body) or r.get("api_version") != API_VERSION
                or r.get("effort") != EFFORT or r.get("max_tokens") != body["max_tokens"]
                or r.get("thinking_sent") != body.get("thinking") or r.get("inference_geo_sent") != INFERENCE_GEO):
            problems.append(f"cache: {k} was not sent as registered")
        if (r.get("prompt_eval_count") or 0) > input_upper(body) and \
                "billed input above the bound" not in closed.get(tag, ""):
            problems.append(f"cache: {k} billed more input than the bound the cap relies on, and {tag} is not "
                            "closed for it")
        if (r.get("eval_count") or 0) > body["max_tokens"]:
            problems.append(f"cache: {k} reports more output tokens than max_tokens")
        before = [e for e in sent.get(k, []) if e.get("kind") in ("send", "resend", "retry")
                  and e["at"] <= r["requested_at"] and e.get("request_sha256") == r["request_sha256"]]
        if not before:
            problems.append(f"cache: {k} has no journal entry before its request")
        s = answers.get(k)
        if s is None:
            problems.append(f"spend: {k} has no spend entry")
        elif abs(s["cost_usd"] - cost(tag, r["usage"])) > 1e-7:
            problems.append(f"spend: {k} records {s['cost_usd']} USD, the usage gives {cost(tag, r['usage'])}")
    running = 0.0
    for e in spend:
        if e.get("event") in ("answer", "charged"):
            running += e["cost_usd"]
            if abs(e["running_total_usd"] - running) > 1e-5:
                problems.append(f"spend: the running total at {e.get('key')} is {e['running_total_usd']}, the "
                                f"entries give {running:.6f}")
    committed = spend_state(spend)["committed"]
    if committed > CAP_USD + 1e-9:
        problems.append(f"spend: {committed:.4f} USD (computed spend and open reservations) is over the cap of "
                        f"{CAP_USD:.2f} USD")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true")
    g.add_argument("--estimate", action="store_true")
    g.add_argument("--run", metavar="MODEL")
    g.add_argument("--close-incomplete", metavar="MODEL")
    g.add_argument("--break-stale-lock", action="store_true")
    ap.add_argument("--resend-unanswered", action="store_true")
    a = ap.parse_args()
    try:
        if a.verify:
            problems = verify()
            if problems:
                sys.exit("hosted arm check failed:\n- " + "\n- ".join(problems))
            print("hosted arm check passed")
        elif a.estimate:
            from .data import load_records
            sets, _ = X.item_sets()
            records = load_records()
            st = state()
            rows = [estimate(t, sets, records, st["cache"]) for t in MODELS]
            print(json.dumps({"cap_usd": CAP_USD, "models": rows,
                              "projected_usd": round(sum(r["projected_usd"] for r in rows), 4),
                              "worst_case_usd": round(sum(r["worst_case_usd"] for r in rows), 4)}, indent=1))
        elif a.break_stale_lock:
            print(break_stale_lock())
        elif a.run:
            print(json.dumps(run_model(a.run, resend_unanswered=a.resend_unanswered)))
        else:
            print(f"{a.close_incomplete} closed: {close_incomplete(a.close_incomplete)}")
    except SystemExit:
        raise
    except BaseException as exc:                 # noqa: BLE001 - no raw traceback, nothing that could hold the key
        sys.exit(f"hosted arm stopped: {scrub(f'{type(exc).__name__}: {exc}')[:300]}")


if __name__ == "__main__":
    main()
