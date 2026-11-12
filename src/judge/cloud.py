"""Judge v1 and v2 on larger open-weights models through OpenRouter: docs/analysis_plan_models.md, Amendment 3.

Added after the held-out run, after the blind labels were locked and after both arms of Amendments 1 and 2 had run and
been reported. Sends only DEV and labelled-sample complaints and refuses any held-out ID (the local runner's item sets
and guard). Every request carries the frozen messages of judge v1 or v2 (prompts.py) and, where the pinned endpoint
supports it, the frozen JSON schema. Its cache, journal, run log and spend journal live under results/models/cloud/;
nothing is written anywhere else.

The API key is read from a file (OPENROUTER_KEY_FILE, default ~/.openrouter_key, which must not be readable by group
or others) when a request is sent, and only then. It is sent in one header that is never forwarded on a redirect
(redirects are not followed), and it is never written to a file, a log line or an error message: errors are rebuilt
from the response body with the key, and anything shaped like a key, removed.

python -m judge.cloud --verify                          # the checks `make check` runs; no network
python -m judge.cloud --estimate                        # projected and worst-case cost; no network
python -m judge.cloud --status                          # credits and the free-model daily count (key; no model)
python -m judge.cloud --run MODEL                       # one model's four cells, in the registered order
python -m judge.cloud --run MODEL --resend-unanswered   # after an interruption: one resend of each served request
python -m judge.cloud --close-incomplete MODEL          # only on a condition of section 9, checked in the files
python -m judge.cloud --break-stale-lock                # after a crash: removes run.lock if its process is gone
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import stat
import sys
import time
import urllib.error
import urllib.request

from . import config as C
from . import hosted as H
from . import models as X
from .models import Refusal, _append, _jsonl, key, now, split_key
from .prompts import build_messages, visible_narrative

DIR = C.RESULTS / "models" / "cloud"
CACHE_FILE = DIR / "responses.jsonl"
JOURNAL = DIR / "sent.jsonl"
RUN_LOG = DIR / "run_log.jsonl"
SPEND = DIR / "spend.jsonl"
REGISTRATION = DIR / "registration.json"
LOCK = DIR / "run.lock"

API_URL = "https://openrouter.ai/api/v1/chat/completions"
CREDITS_URL = "https://openrouter.ai/api/v1/credits"
KEY_URL = "https://openrouter.ai/api/v1/key"
MODELS_URL = "https://openrouter.ai/api/v1/models"
ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{id}/endpoints"
KEY_FILE_ENV = "OPENROUTER_KEY_FILE"
DEFAULT_KEY_FILE = "~/.openrouter_key"
TIMEOUT_S = 600

# ------------------------------------------------------------------ the registered models (Amendment 3, section 2)
# Resolved on 2026-11-12 by src/judge/cloud_select.py from the committed snapshot; `--check` there re-derives this.
# Order = run order within each queue = the only order of every table of this branch.
FREE_QUEUE = ("nvidia/nemotron-3-ultra-550b-a55b:free", "dots-studio/dots-3-note-preview:free",
              "nvidia/nemotron-3.5-super-vl:free", "google/gemma-4-31b-it:free")
PAID_QUEUE = ("deepseek/deepseek-v4-pro-0813", "moonshotai/kimi-k2.6", "xiaomi/mimo-v2.6-pro")
MODELS = FREE_QUEUE + PAID_QUEUE
REFERENCE_ONLY = ("google/gemma-4-31b-it:free",)          # the same weights as Amendment 1's gemma4:31b
COMPARED = tuple(t for t in MODELS if t not in REFERENCE_ONLY)
SAME_WEIGHTS_AS = {"google/gemma-4-31b-it:free": "gemma4:31b"}
FREE = set(FREE_QUEUE)
# the pinned endpoint of each model: provider tag sent in provider.order, provider name the answer must report,
# USD per token, and what the endpoint supports (which decides seed and the format field)
ENDPOINT = {
    "nvidia/nemotron-3-ultra-550b-a55b:free": {"tag": "nvidia", "provider": "Nvidia", "quantization": "unknown",
                                               "prompt": 0.0, "completion": 0.0, "seed": True, "format": None},
    "dots-studio/dots-3-note-preview:free": {"tag": "atlas-cloud/fp8", "provider": "AtlasCloud", "quantization": "fp8",
                                             "prompt": 0.0, "completion": 0.0, "seed": False, "format": "json_schema"},
    "nvidia/nemotron-3.5-super-vl:free": {"tag": "nvidia", "provider": "Nvidia", "quantization": "unknown",
                                          "prompt": 0.0, "completion": 0.0, "seed": True, "format": "json_schema"},
    "google/gemma-4-31b-it:free": {"tag": "google-ai-studio", "provider": "Google AI Studio", "quantization": "unknown",
                                   "prompt": 0.0, "completion": 0.0, "seed": True, "format": "json_object"},
    "deepseek/deepseek-v4-pro-0813": {"tag": "novita/fp8", "provider": "Novita", "quantization": "fp8",
                                      "prompt": 0.99e-6, "completion": 2.97e-6, "seed": True, "format": "json_schema"},
    "moonshotai/kimi-k2.6": {"tag": "crusoe/bf16", "provider": "Crusoe", "quantization": "bf16",
                             "prompt": 0.70e-6, "completion": 3.50e-6, "seed": True, "format": "json_schema"},
    "xiaomi/mimo-v2.6-pro": {"tag": "gmicloud/bf16", "provider": "GMICloud", "quantization": "bf16",
                             "prompt": 0.32625e-6, "completion": 0.6525e-6, "seed": True, "format": "json_schema"},
}
TEMPERATURE, SEED, MAX_TOKENS = 0, 42, 512          # the local runs' temperature, seed and num_predict
REASONING_OFF = {"effort": "none"}                    # OpenRouter's documented switch; checked on every answer
DROPPABLE = ("reasoning", "seed")                     # left out for a model only on a first-request refusal (s. 9)

# ------------------------------------------------------------------ money (Amendment 3, section 7)
CAP_USD = 15.00                                       # the whole branch
SUB_CAP_USD = 5.00                                    # each model
FORMAT_ALLOWANCE = 2_000                              # worst case: tokens a provider may add for the format
PROJECTED_BYTES_PER_TOKEN = 3
FORMAT_PROJECTED = 500
CREDIT_MARGIN_USD = 1.00                              # a paid model starts only with this much credit to spare

# ------------------------------------------------------------------ pace and errors (Amendment 3, section 9)
MIN_INTERVAL_S = {"free": 4.0, "paid": 1.0}           # 15 per minute on free models, under their 20
FREE_DAILY_MARGIN = 25                                # free sends left unused of the day's remaining count
FREE_DAILY_FALLBACK = 950                             # if /api/v1/key does not report the count: journal-based cap
RETRY_DELAYS_S = (30, 60, 120, 240)
RETRY_AFTER_MAX_S = 300
MAX_ERRORS_PER_MODEL = 150                            # waited-out error answers only
REPEAT_GAP_S = 24 * 3600
TRANSIENT_STATUSES = (408, 429, 500, 502, 503, 504, 520, 522, 524, 529)
ACCOUNT_STATUSES = (401, 402)
# a refusal caused by the account's own settings, not by the request: privacy / data policy, guardrails
ACCOUNT_PHRASES = ("data policy", "guardrail", "privacy setting")
NOT_SERVED = ("transient", "rejected", "account", "quota", "refused")
KEY_PATTERN = re.compile(r"sk-or-[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{20,}")
DONE_REASON = {"stop": "stop", "length": "length"}    # every other finish_reason is kept as it is (malformed)
VERDICT_WORDS = {"rejected": "the request was rejected",
                 "account": "an account condition (the key, the credit balance or the account's privacy settings)",
                 "quota": "the free models' daily request limit",
                 "transient": "a rate limit, an overload or a server error, after its retries",
                 "unavailable": "the model or its pinned endpoint is no longer listed as registered"}
PROTECTED = ("docs/analysis_plan_models.md", "src/judge/cloud.py", "src/judge/cloud_select.py",
             "src/judge/cloud_analysis.py", "src/judge/hosted.py", "src/judge/models.py",
             "src/judge/models_analysis.py", "src/judge/prompts.py", "src/judge/llm.py", "src/judge/judgments.py",
             "src/judge/config.py", "src/judge/metrics.py", "src/judge/data.py", "src/judge/label.py",
             "src/judge/baselines.py", "src/judge/freeze.py", "Makefile", "tests/test_cloud.py")


def arm(tag: str) -> str:
    return "free" if tag in FREE else "paid"


def log_event(event: dict) -> None:
    _append(RUN_LOG, {"at": now(), **event})


# ------------------------------------------------------------------ the key
def key_path() -> str:
    return os.path.expanduser(os.environ.get(KEY_FILE_ENV) or DEFAULT_KEY_FILE)


def key_file_ok() -> str | None:
    """None if the key file exists and only its owner can read it; otherwise why not. Reads no content."""
    p = key_path()
    try:
        st = os.stat(p)
    except FileNotFoundError:
        return f"the key file {KEY_FILE_ENV} points to (or {DEFAULT_KEY_FILE}) does not exist"
    if st.st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        return "the key file can be read by group or others (chmod 600 it)"
    return None


def read_key() -> str:
    """Called only when a request is sent."""
    why = key_file_ok()
    if why:
        raise CloudError(None, why + "; nothing was sent.")
    with open(key_path(), encoding="utf-8") as f:
        k = f.read().strip()
    if not k:
        raise CloudError(None, "the key file is empty; nothing was sent.")
    return k


def scrub(text) -> str:
    s = str(text)
    try:
        if key_file_ok() is None:
            with open(key_path(), encoding="utf-8") as f:
                k = f.read().strip()
            if k:
                s = s.replace(k, "[redacted]")
    except OSError:
        pass
    return KEY_PATTERN.sub("[redacted]", s)


class CloudError(Exception):
    """A failed request, rebuilt without the request (no headers, no key)."""

    def __init__(self, status, message, request_id=None, etype=None, maybe_billed=False, retry_after=None,
                 usage=None, date=None, remaining=None, moderation=None, in_200=False):
        self.status, self.message, self.request_id, self.etype = status, scrub(message)[:500], request_id, etype
        self.maybe_billed, self.retry_after, self.usage, self.date = maybe_billed, retry_after, usage, date
        self.remaining, self.moderation, self.in_200 = remaining, moderation, in_200
        super().__init__(f"HTTP {status}: {self.message}" if status else self.message)


# ------------------------------------------------------------------ requests
def schema_for_api(schema: dict) -> dict:
    return {**schema, "additionalProperties": False}


def build_body(tag: str, version: str, rec: dict, narrative: str, drop=()) -> dict:
    messages, schema = build_messages(version, rec, narrative)
    assert [m["role"] for m in messages] == ["system", "user"]
    ep = ENDPOINT[tag]
    body = {"model": tag, "messages": messages, "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS,
            "reasoning": dict(REASONING_OFF), "stream": False,
            "provider": {"order": [ep["tag"]], "allow_fallbacks": False, "require_parameters": True}}
    if ep["seed"]:
        body["seed"] = SEED
    if ep["format"] == "json_schema":
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": f"judge_{version}", "strict": True, "schema": schema_for_api(schema)}}
    elif ep["format"] == "json_object":
        body["response_format"] = {"type": "json_object"}
    for f in drop:
        body.pop(f, None)
    return body


def request_hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def body_bytes(body: dict) -> int:
    return len(json.dumps(body, ensure_ascii=False).encode("utf-8"))


def input_upper(body: dict) -> int:
    return body_bytes(body) + FORMAT_ALLOWANCE


def computed_cost(tag: str, usage: dict) -> float:
    ep = ENDPOINT[tag]
    return (ep["prompt"] * (usage.get("prompt_tokens") or 0)
            + ep["completion"] * (usage.get("completion_tokens") or 0))


def cost(tag: str, usage: dict) -> float:
    """What an answer is charged against the cap: the larger of the cost the API reports and the registered prices
    applied to the reported tokens."""
    reported = usage.get("cost")
    reported = float(reported) if isinstance(reported, (int, float)) and not isinstance(reported, bool) else 0.0
    return max(reported, computed_cost(tag, usage))


def worst_case(tag: str, body: dict) -> float:
    ep = ENDPOINT[tag]
    return ep["prompt"] * input_upper(body) + ep["completion"] * body["max_tokens"]


def projected(tag: str, body: dict) -> float:
    ep = ENDPOINT[tag]
    tokens = math.ceil(body_bytes(body) / PROJECTED_BYTES_PER_TOKEN) + FORMAT_PROJECTED
    return ep["prompt"] * tokens + ep["completion"] * MAX_TOKENS


# ------------------------------------------------------------------ transport
def _open(req, timeout):
    return urllib.request.build_opener(H._NoRedirect).open(req, timeout=timeout)


def _valid_usage(u) -> dict | None:
    if isinstance(u, dict) and all(isinstance(u.get(k), int) for k in ("prompt_tokens", "completion_tokens")):
        return u
    return None


def check_shape(resp) -> None:
    ok = (isinstance(resp, dict) and isinstance(resp.get("choices"), list) and len(resp["choices"]) >= 1
          and isinstance(resp["choices"][0], dict) and isinstance(resp["choices"][0].get("message"), dict)
          and isinstance(resp["choices"][0]["message"].get("content"), (str, type(None)))
          and isinstance(resp["choices"][0].get("finish_reason"), (str, type(None)))
          and isinstance(resp.get("model"), str) and _valid_usage(resp.get("usage")) is not None)
    if not ok:
        raise ValueError("unexpected answer shape")


def _error_from_json(err: dict, status, hdrs, in_200=False, usage=None) -> CloudError:
    meta = err.get("metadata") or {}
    moderation = meta if isinstance(meta, dict) and "reasons" in meta else None
    code = err.get("code") if isinstance(err.get("code"), int) else status
    raw = meta.get("raw") if isinstance(meta, dict) else None
    message = str(err.get("message") or "") + (f" ({str(raw)[:200]})" if raw else "")
    try:
        retry_after = float(hdrs.get("retry-after")) if hdrs and hdrs.get("retry-after") else None
    except ValueError:
        retry_after = None
    return CloudError(code, message, hdrs.get("x-generation-id") if hdrs else None, err.get("type"),
                      retry_after=retry_after, usage=usage, date=hdrs.get("date") if hdrs else None,
                      remaining=hdrs.get("x-ratelimit-remaining") if hdrs else None, moderation=moderation,
                      in_200=in_200)


class OpenRouterTransport:
    """POST /api/v1/chat/completions and the two GETs of the account, with urllib. Tests replace it."""

    def _request(self, url: str, body: dict | None = None):
        api_key = read_key()
        req = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode("utf-8"),
                                     method="GET" if body is None else "POST",
                                     headers={"content-type": "application/json"})
        req.add_unredirected_header("Authorization", "Bearer " + api_key)
        del api_key
        return req

    def send(self, body: dict) -> tuple[dict, dict]:
        req = self._request(API_URL, body)
        failure, resp = None, None
        try:
            r = _open(req, TIMEOUT_S)
        except urllib.error.HTTPError as exc:
            failure = _http_failure(exc)
        except BaseException as exc:             # noqa: BLE001 - timeout, connection, interrupt: may have been billed
            failure = CloudError(None, f"no answer: {type(exc).__name__}: {exc}", maybe_billed=True)
        else:
            hdrs = r.headers
            try:
                with r:
                    resp = json.loads(r.read().decode("utf-8"))
                if isinstance(resp, dict) and isinstance(resp.get("error"), dict) and not resp.get("choices"):
                    failure = _error_from_json(resp["error"], 200, hdrs, in_200=True,
                                               usage=_valid_usage(resp.get("usage")))
                else:
                    check_shape(resp)
                    return resp, {"generation-id": resp.get("id"), "date": hdrs.get("date") if hdrs else None}
            except BaseException as exc:         # noqa: BLE001 - an answer arrived and could not be read: billed
                usage = _valid_usage(resp.get("usage")) if isinstance(resp, dict) else None
                failure = CloudError(200, f"unreadable answer: {type(exc).__name__}", maybe_billed=True, usage=usage,
                                     date=hdrs.get("date") if hdrs else None)
        finally:
            del req
        raise failure

    def get(self, url: str) -> dict:
        req = self._request(url)
        try:
            with _open(req, 60) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            failure = _http_failure(exc)
        except BaseException as exc:             # noqa: BLE001
            failure = CloudError(None, f"no answer: {type(exc).__name__}: {exc}")
        finally:
            del req
        raise failure


def public_get(url: str) -> tuple[dict, str | None]:
    """A request without the key (the public listings). Returns the body and the server's date."""
    req = urllib.request.Request(url, headers={"User-Agent": "refund-decision-eval"})
    with _open(req, 60) as r:
        return json.loads(r.read().decode("utf-8")), r.headers.get("date")


def _http_failure(exc: urllib.error.HTTPError) -> CloudError:
    try:
        raw = exc.read().decode("utf-8", "replace")
    except Exception:                            # noqa: BLE001
        raw = ""
    try:
        err = json.loads(raw)["error"]
        assert isinstance(err, dict)
    except Exception:                            # noqa: BLE001
        err = {"message": raw[:300] or str(exc.reason)}
    e = _error_from_json(err, exc.code, exc.headers)
    e.status = exc.code                          # the HTTP status decides; error.code may repeat it
    return e


def classify(exc: BaseException, tag: str) -> tuple[str, str]:
    """Amendment 3, section 9. 'moderated': a moderation flag (an answer, malformed, scored no money back).
    'quota': the free models' daily limit (stops; resumes after the reset). 'account': 401, 402, a 4xx naming the
    account's data policy, privacy setting or a guardrail, or nothing sent.
    'transient': 408, 429 and 5xx, waited out. 'rejected': any other 3xx or 4xx. 'lost': anything else."""
    if not isinstance(exc, CloudError):
        return "lost", scrub(f"{type(exc).__name__}: {exc}")[:500]
    s = exc.status
    if exc.moderation is not None:
        return "moderated", exc.message
    if s == 429 and tag in FREE and str(exc.remaining) == "0":
        return "quota", exc.message
    if s in ACCOUNT_STATUSES or (s is not None and 400 <= s < 500
                                 and any(p in exc.message.lower() for p in ACCOUNT_PHRASES)):
        return "account", exc.message
    if s is not None and (s in TRANSIENT_STATUSES or 500 <= s < 600):
        return "transient", exc.message
    if s is not None and 300 <= s < 500:
        return "rejected", exc.message
    if s is None and not exc.maybe_billed:
        return "account", exc.message             # nothing was sent (key file)
    return "lost", exc.message


def first_request_drop(exc: CloudError, tag: str, journal: list[dict]) -> str | None:
    """Section 9: a 400 to a model's very first request (no answer of the model yet, no other request of it
    journaled) whose message names a droppable field leaves that field out for this model only, once per field.
    Returns the field, or None."""
    if exc.status != 400 or any(split_key(k)[0] == tag for k in _cache_keys()):
        return None
    if len({e["key"] for e in journal if split_key(e["key"])[0] == tag}) > 1:
        return None
    for f in DROPPABLE:
        if re.search(r"(?<![A-Za-z_])" + f + r"(?![A-Za-z_])", exc.message, re.I):
            return f
    return None


def _cache_keys() -> set:
    return {r["key"] for r in _jsonl(CACHE_FILE)}


@functools.lru_cache(maxsize=None)
def accepted_names(tag: str) -> frozenset:
    """The names an answer of `tag` may report: the id, the id without ':free', and the canonical slug of the
    snapshot (with or without ':free')."""
    base = tag.removesuffix(":free")
    from . import cloud_select as S
    slug = next((m.get("canonical_slug") for m in S.load_snapshot()[0] if m["id"] == tag), None)
    return frozenset({tag, base} | ({slug, slug + ":free"} if slug else set()))


def model_matches(tag: str, name) -> bool:
    """Served by the registered model: one of accepted_names, or the id without ':free' followed by a date suffix
    (-YYYYMMDD or -MMDD), which is how a dated snapshot of the same id is reported."""
    if not isinstance(name, str):
        return False
    base = tag.removesuffix(":free")
    return name in accepted_names(tag) or re.fullmatch(re.escape(base) + r"-(\d{8}|\d{4})(:free)?", name) is not None


def reasoning_seen(resp: dict) -> tuple[int, int]:
    """(reasoning tokens reported, characters of reasoning text returned)."""
    usage = resp.get("usage") or {}
    rt = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
    msg = resp["choices"][0]["message"]
    text = msg.get("reasoning") or ""
    details = msg.get("reasoning_details") or []
    chars = len(text) if isinstance(text, str) else 0
    chars += sum(len(str(d.get("text") or d.get("summary") or "")) for d in details if isinstance(d, dict))
    return int(rt) if isinstance(rt, int) else 0, chars


def answer_record(tag, version, item_id, body, rhash, resp, headers, requested_at, wall, drop) -> dict:
    usage = resp["usage"]
    choice = resp["choices"][0]
    finish = choice.get("finish_reason")
    rt, rchars = reasoning_seen(resp)
    return {
        "key": key(tag, version, item_id), "model": tag, "prompt_version": version, "item_id": item_id,
        "request_sha256": rhash, "response_model": resp["model"], "provider": resp.get("provider"),
        "generation_id": headers.get("generation-id"), "server_date": headers.get("date"),
        "requested_at": requested_at, "wall_s": round(wall, 3), "fields_dropped": sorted(drop),
        "endpoint_tag": ENDPOINT[tag]["tag"], "content": choice["message"].get("content") or "",
        "finish_reason": finish, "native_finish_reason": choice.get("native_finish_reason"),
        "done_reason": DONE_REASON.get(finish, finish), "reasoning_tokens": rt, "reasoning_chars": rchars,
        "usage": usage, "prompt_eval_count": usage["prompt_tokens"], "eval_count": usage["completion_tokens"],
        "logprobs": None, "thinking": None, "moderated": None,
    }


def moderated_record(tag, version, item_id, rhash, exc: CloudError, requested_at, wall, drop) -> dict:
    """A moderation flag is an answer: no text, malformed, scored no money back (as the frozen rule scores a
    refusal); its reasons are kept."""
    return {"key": key(tag, version, item_id), "model": tag, "prompt_version": version, "item_id": item_id,
            "request_sha256": rhash, "response_model": None, "provider": (exc.moderation or {}).get("provider_name"),
            "generation_id": exc.request_id, "server_date": exc.date, "requested_at": requested_at,
            "wall_s": round(wall, 3), "fields_dropped": sorted(drop), "endpoint_tag": ENDPOINT[tag]["tag"],
            "content": "", "finish_reason": "content_filter", "native_finish_reason": None,
            "done_reason": "content_filter", "reasoning_tokens": 0, "reasoning_chars": 0,
            "usage": exc.usage or {"prompt_tokens": 0, "completion_tokens": 0}, "prompt_eval_count": 0,
            "eval_count": 0, "logprobs": None, "thinking": None,
            "moderated": {"status": exc.status, "reasons": (exc.moderation or {}).get("reasons")}}


# ------------------------------------------------------------------ state and spend
def spend_state(spend=None) -> dict:
    return H.spend_state(_jsonl(SPEND) if spend is None else spend)


def committed_for(tag: str, spend=None) -> float:
    spend = _jsonl(SPEND) if spend is None else spend
    return spend_state([e for e in spend if e.get("model") == tag or e.get("event") == "released"])["committed"]


def state() -> dict:
    cache = {r["key"]: r for r in _jsonl(CACHE_FILE)}
    status: dict[str, str] = {}
    for e in _jsonl(RUN_LOG):
        if e.get("event") == "model_finished":
            status[e["model"]] = "complete"
        elif e.get("event") == "closed_incomplete":
            status[e["model"]] = e["reason"]
    return {"cache": cache, "journal": _jsonl(JOURNAL), "status": status, "spent": spend_state()["committed"]}


def expected_keys(tag: str, sets: dict) -> set:
    return {key(tag, v, i) for v, s in X.CELLS for i in sets[s]}


def dropped_fields(tag: str) -> list[str]:
    return [e["field"] for e in _jsonl(RUN_LOG) if e.get("event") == "field_dropped" and e.get("model") == tag]


def attempts(entries):
    out, base = [], "send"
    for n, e in enumerate(entries):
        if e.get("kind") in ("send", "resend", "retry"):
            base = e["kind"] if e["kind"] != "retry" else base
            nxt = entries[n + 1] if n + 1 < len(entries) else {}
            out.append({**e, "base": base, "served": nxt.get("kind") not in NOT_SERVED})
    return out


def resend_served(entries) -> bool:
    return any(a["served"] and a["base"] == "resend" for a in attempts(entries))


def error_count(journal, tag) -> int:
    return sum(1 for e in journal if e.get("kind") == "transient" and split_key(e["key"])[0] == tag)


def free_sends_today(journal) -> int:
    day = now()[:10]
    return sum(1 for e in journal if e.get("kind") in ("send", "resend", "retry") and e["at"][:10] == day
               and split_key(e["key"])[0] in FREE)


def repeated(tag: str, run_log: list[dict], journal: list[dict]) -> dict | None:
    """Section 9, condition (c): the model's last run stopped on a rejection or on 'unavailable', and the same
    condition (same request, status and error type; or the model or endpoint missing from the listing) was observed
    at least 24 hours apart by the server's date headers. Returns that last stop, or None."""
    runs = [e for e in run_log if e.get("model") == tag and e.get("event") in ("run_started", "run_stopped")]
    last = runs[-1] if runs and runs[-1]["event"] == "run_stopped" else None
    if not last or last.get("verdict") not in ("rejected", "unavailable"):
        return None
    if last["verdict"] == "unavailable":
        same = [e for e in run_log if e.get("event") == "unavailable" and e.get("model") == tag]
    else:
        same = [e for e in journal if e.get("key") == last.get("key") and e.get("kind") == "rejected"
                and e.get("status") == last.get("status") and e.get("error_type") == last.get("error_type")]
    times = [t for t in (H._server_seconds(e.get("server_date")) for e in same) if t is not None]
    return last if len(times) >= 2 and max(times) - min(times) >= REPEAT_GAP_S else None


def load_registration() -> dict:
    if not REGISTRATION.exists():
        raise Refusal(f"results/models/cloud/{REGISTRATION.name} is missing; refusing.")
    reg = json.loads(REGISTRATION.read_text())
    need = ("repository", "code_commit", "code_published_at", "gist_id", "gist_url", "gist_created_at")
    missing = [k for k in need if not reg.get(k)]
    if missing:
        raise Refusal(f"results/models/cloud/{REGISTRATION.name} lacks {', '.join(missing)}: the branch's code "
                      "commit, the time GitHub records for its publication and the secret gist must be recorded "
                      "before the first request.")
    return reg


def github_main_head() -> str:
    import subprocess
    slug = load_registration()["repository"]
    out = subprocess.run(["gh", "api", f"repos/{slug}/git/refs/heads/main", "--jq", ".object.sha"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


# ------------------------------------------------------------------ live listings (no key)
def listing_check(tag: str, getter=public_get) -> tuple[str | None, str | None]:
    """(None or why the model is not available as registered, the server's date). The model must still be listed
    under its id with its Hugging Face id, the pinned endpoint must be listed with status 0, the same provider and
    the same or wider precision, and a price no higher than the registered one (zero for a free model)."""
    from . import cloud_select as S
    snap = {m["id"]: m for m in S.load_snapshot()[0]}
    models, date = getter(MODELS_URL)
    live = {m["id"]: m for m in models.get("data") or []}
    if tag not in live:
        return f"{tag} is no longer listed", date
    if live[tag].get("hugging_face_id") != snap[tag].get("hugging_face_id"):
        return f"{tag} is now listed with other weights ({live[tag].get('hugging_face_id')})", date
    eps, date = getter(ENDPOINTS_URL.format(id=tag))
    ep = ENDPOINT[tag]
    match = [e for e in (eps.get("data") or {}).get("endpoints") or [] if e.get("tag") == ep["tag"]]
    if not match:
        return f"the pinned endpoint {ep['tag']} of {tag} is no longer listed", date
    e = match[0]
    if e.get("status") != 0:
        return f"the pinned endpoint {ep['tag']} reports status {e.get('status')}", date
    if e.get("provider_name") != ep["provider"]:
        return f"the pinned endpoint {ep['tag']} is now served by {e.get('provider_name')}", date
    if (e.get("quantization") or "unknown") != ep["quantization"]:
        return f"the pinned endpoint {ep['tag']} now reports quantization {e.get('quantization')}", date
    p = e.get("pricing") or {}
    if float(p.get("prompt") or 0) > ep["prompt"] + 1e-15 or float(p.get("completion") or 0) > ep["completion"] + 1e-15 \
            or float(p.get("request") or 0) > 0:
        return f"the pinned endpoint {ep['tag']} of {tag} now costs more than registered", date
    return None, date


# ------------------------------------------------------------------ before a model (lock held)
def preflight(tag: str, published_head=None, getter=public_get, transport=None) -> dict:
    if tag not in MODELS:
        raise Refusal(f"{tag} is not a model of Amendment 3; no model is ever added or replaced.")
    if X.registered_part_sha256() != X.REGISTERED_SHA256:
        raise Refusal("lines 1-149 of docs/analysis_plan_models.md differ from the registered text; refusing.")
    local, hosted = X.state()["status"], H.state()["status"]
    open_ = [t for t, _ in X.MODELS if not local.get(t)] + [t for t in H.MODELS if not hosted.get(t)]
    if open_:
        raise Refusal("a model of Amendment 1 or 2 is not closed (" + ", ".join(open_) + "); refusing.")
    reg = load_registration()
    try:
        changed = X.git("diff", "--name-only", reg["code_commit"], "--", *PROTECTED).split()
    except Exception:                            # noqa: BLE001
        raise Refusal(f"the published code commit {reg['code_commit']} is not in this repository; refusing.") from None
    if changed:
        raise Refusal("these files differ from the branch's published code commit: " + ", ".join(changed))
    dirty = [ln[3:] for ln in X.git("status", "--porcelain").splitlines()]
    outside = [p for p in dirty if not p.startswith("results/models/cloud/")]
    if outside:
        raise Refusal("uncommitted changes outside results/models/cloud/: " + ", ".join(outside))
    st = state()
    started = any(split_key(e["key"])[0] == tag for e in st["journal"])
    if dirty and not started:
        raise Refusal("results/models/cloud/ has uncommitted changes: commit and publish the previous model's results "
                      "before the next model starts.")
    head = X.git("rev-parse", "HEAD").strip()
    if (published_head or github_main_head)() != head:
        raise Refusal(f"HEAD {head[:12]} is not the commit GitHub shows as main: publish before sending.")
    if st["status"].get(tag):
        raise Refusal(f"{tag} is closed ({st['status'][tag]}); a closed model is never run again.")
    queue = FREE_QUEUE if tag in FREE else PAID_QUEUE
    for earlier in queue[: queue.index(tag)]:
        if not st["status"].get(earlier):
            raise Refusal(f"{earlier} comes before {tag} in the registered order and is not closed.")
    if tag not in FREE and any(e.get("event") == "budget_stop" and e.get("scope") == "branch" for e in _jsonl(RUN_LOG)):
        raise Refusal("the branch stopped at its 15 USD cap; no paid request is sent after that.")
    from .label import is_locked
    if not is_locked("owner"):
        raise Refusal("the blind labels are not locked; refusing.")
    why = key_file_ok()
    if why:
        raise Refusal(why + "; nothing was sent.")
    why, date = listing_check(tag, getter)
    if why:
        log_event({"event": "unavailable", "model": tag, "reason": why, "server_date": date})
        log_event({"event": "run_stopped", "model": tag, "verdict": "unavailable", "reason": why})
        raise Refusal(f"stopped before {tag}: {why}. Nothing is substituted; run again later, and if the same holds "
                      "24 hours later the model can be closed with --close-incomplete.")
    sets, test = X.item_sets()
    return {"sets": sets, "test": test, "registration": reg}


def account_readings(transport) -> dict:
    """Credits and the free-model daily count, as the API reports them (key; no model is called)."""
    out = {}
    try:
        c = (transport.get(CREDITS_URL) or {}).get("data") or {}
        out["total_credits"], out["total_usage"] = c.get("total_credits"), c.get("total_usage")
        if isinstance(c.get("total_credits"), (int, float)) and isinstance(c.get("total_usage"), (int, float)):
            out["remaining_credits"] = round(c["total_credits"] - c["total_usage"], 6)
    except Exception as exc:                     # noqa: BLE001
        out["credits_error"] = scrub(f"{type(exc).__name__}: {exc}")[:200]
    try:
        k = (transport.get(KEY_URL) or {}).get("data") or {}
        fr = k.get("free_model_daily_requests")
        out["free_model_daily_requests"] = fr if isinstance(fr, dict) else None
        out["is_free_tier"] = k.get("is_free_tier")
    except Exception as exc:                     # noqa: BLE001
        out["key_error"] = scrub(f"{type(exc).__name__}: {exc}")[:200]
    return out


def work_list(tag, sets, records):
    return X.work_list(tag, sets, records)


def estimate(tag, sets, records, cache=None) -> dict:
    cache = cache or {}
    todo = [(v, s, i, r, n) for v, s, i, r, n in work_list(tag, sets, records) if key(tag, v, i) not in cache]
    bodies = [build_body(tag, v, r, n) for v, s, i, r, n in todo]
    from . import cloud_select as S
    ep = ENDPOINT[tag]
    return {"model": tag, "requests": len(bodies),
            "projected_usd": round(sum(projected(tag, b) for b in bodies), 6),
            "worst_case_usd": round(sum(worst_case(tag, b) for b in bodies), 6),
            "basis_usd": round(S.projected_cost(ep["prompt"], ep["completion"]) * len(bodies) / 600, 6)}


# ------------------------------------------------------------------ the lock
def acquire_lock():
    DIR.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        raise Refusal(f"another run holds {LOCK.name}. If no run is going on (a crash), run "
                      "`python -m judge.cloud --break-stale-lock`.") from None
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps({"pid": os.getpid(), "at": now()}) + "\n")


def release_lock():
    try:
        LOCK.unlink()
    except FileNotFoundError:
        pass


def break_stale_lock() -> str:
    if not LOCK.exists():
        return "no run.lock"
    try:
        held = json.loads(LOCK.read_text())
        pid = int(held["pid"])
    except Exception:                            # noqa: BLE001
        raise Refusal(f"{LOCK.name} cannot be read; remove it by hand only after checking that no run is going on.") \
            from None
    if H._alive(pid):
        raise Refusal(f"{LOCK.name} belongs to process {pid}, which is still running; nothing was removed.")
    LOCK.unlink()
    log_event({"event": "stale_lock_removed", "lock": held})
    return f"removed {LOCK.name} of process {pid}, which no longer runs"


# ------------------------------------------------------------------ closing
def close_model(tag: str, reason: str) -> str:
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
    answers = list({e["key"]: e for e in mine if e["event"] == "answer"}.values())
    _append(SPEND, {"at": now(), "event": "model_total", "model": tag, "answers": len(answers),
                    "charged_without_answer": len(mine) - len(answers),
                    **{f: sum(e.get(f) or 0 for e in answers) for f in ("prompt_tokens", "completion_tokens",
                                                                       "reasoning_tokens")},
                    "cost_usd": round(sum(e["cost_usd"] for e in mine), 6),
                    "reported_cost_usd": round(sum(e.get("reported_cost_usd") or 0 for e in answers), 6),
                    "computed_total_usd": round(spend_state()["computed"], 6)})


def reconcile() -> None:
    for e in spend_state()["open"]:
        _append(SPEND, {"at": now(), "event": "charged", "model": e["model"], "key": e["key"], "cost_usd": e["cost_usd"],
                        "running_total_usd": round(spend_state()["computed"] + e["cost_usd"], 6),
                        "note": "reservation of a request whose outcome was lost (crash)", "reservation": e["id"]})
        log_event({"event": "reservation_charged", "model": e["model"], "key": e["key"], "usd": e["cost_usd"]})


# ------------------------------------------------------------------ the run
def run_model(tag, transport=None, resend_unanswered=False, published_head=None, records=None, sleep=time.sleep,
              getter=public_get) -> dict:
    acquire_lock()
    try:
        transport = transport or OpenRouterTransport()
        pre = preflight(tag, published_head, getter)
        before = account_readings(transport)
        log_event({"event": "account_before", "model": tag, **before})
        try:
            return _run(tag, transport, resend_unanswered, records, pre, sleep, before)
        finally:
            log_event({"event": "account_after", "model": tag, **account_readings(transport)})
    finally:
        release_lock()


def _free_allowance(tag, before, journal) -> int | None:
    if tag not in FREE:
        return None
    fr = before.get("free_model_daily_requests") or {}
    if isinstance(fr.get("remaining"), int):
        return max(0, fr["remaining"] - FREE_DAILY_MARGIN)
    return max(0, FREE_DAILY_FALLBACK - free_sends_today(journal))


def _run(tag, transport, resend_unanswered, records, pre, sleep, before) -> dict:
    from .data import load_records

    reconcile()
    st = state()
    if st["status"].get(tag):                    # also checked in preflight; checked again under the lock
        raise Refusal(f"{tag} is closed ({st['status'][tag]}); a closed model is never run again.")
    sets, test = pre["sets"], pre["test"]
    allowed = set(sets["dev"]) | set(sets["labelled"])
    records = records if records is not None else load_records()
    journal = H.by_key(st["journal"])
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
    est = estimate(tag, sets, records, st["cache"])
    committed, mine = st["spent"], committed_for(tag)
    _append(SPEND, {"at": now(), "event": "estimate", **est, "spent_before_usd": round(committed, 6),
                    "cap_usd": CAP_USD, "sub_cap_usd": SUB_CAP_USD})
    if tag not in FREE:
        credits = before.get("remaining_credits")
        if not isinstance(credits, (int, float)) or credits < est["worst_case_usd"] + CREDIT_MARGIN_USD:
            raise Refusal(f"{tag}: the account reports {credits} USD of credit, less than the worst case of the "
                          f"remaining requests ({est['worst_case_usd']:.4f} USD) plus {CREDIT_MARGIN_USD:.2f}; "
                          "nothing was sent.")
        if committed + est["projected_usd"] > CAP_USD or mine + est["projected_usd"] > SUB_CAP_USD:
            scope = "branch" if committed + est["projected_usd"] > CAP_USD else "model"
            log_event({"event": "budget_stop", "model": tag, "scope": scope, "when": "before the model",
                       "spent_usd": round(committed, 6), **est})
            close_model(tag, "not run: projection exceeded remaining budget" if not any(
                split_key(k)[0] == tag for k in st["cache"]) else "stopped by the cap: the projection did not fit")
            raise Refusal(f"{tag}: the projection {est['projected_usd']:.4f} USD does not fit under the "
                          f"{'branch' if scope == 'branch' else 'model'} cap; nothing was sent, and it is closed.")
    allowance = _free_allowance(tag, before, st["journal"])
    log_event({"event": "run_started", "model": tag, "resend_unanswered": resend_unanswered,
               "free_sends_allowed": allowance})
    drop = dropped_fields(tag)
    done, sends, last = 0, 0, [0.0]

    def record_spend(entry):
        _append(SPEND, {"at": now(), "model": tag, **entry,
                        "running_total_usd": round(spend_state()["computed"] + (entry.get("cost_usd") or 0), 6)})

    def stop(k, verdict, message, he):
        log_event({"event": "run_stopped", "model": tag, "key": k, "verdict": verdict,
                   "status": he.status if he else None, "error_type": he.etype if he else None,
                   "reason": message, "request_id": he.request_id if he else None})

    for version, split, item_id, rec, narrative in work_list(tag, sets, records):
        k = key(tag, version, item_id)
        if k in st["cache"]:
            continue
        es = journal.get(k, [])
        served = [a for a in attempts(es) if a["served"]]
        if served:
            if resend_served(es):
                continue
            kind = "resend"
        else:
            kind = "send"
        retries = 0
        while True:
            if allowance is not None and sends >= allowance:
                log_event({"event": "daily_pause", "model": tag, "sends": sends})
                print(f"{tag}: {sends} free requests sent in this run, the day's allowance; resume after the daily "
                      f"reset with --run {tag}.", flush=True)
                return {"model": tag, "new_answers": done, "paused": True}
            body = build_body(tag, version, rec, narrative, drop)
            rhash = request_hash(body)
            reserve = worst_case(tag, body)
            committed, mine = spent(), committed_for(tag)
            if committed + reserve > CAP_USD or mine + reserve > SUB_CAP_USD:
                scope = "branch" if committed + reserve > CAP_USD else "model"
                log_event({"event": "budget_stop", "model": tag, "key": k, "scope": scope,
                           "spent_usd": round(committed, 6), "model_spent_usd": round(mine, 6),
                           "next_worst_case_usd": round(reserve, 6)})
                close_model(tag, f"stopped by the {'15 USD branch' if scope == 'branch' else '5 USD model'} cap: "
                                 "the next request's worst case did not fit")
                raise Refusal(f"budget: the next request of {tag} could cost up to {reserve:.4f} USD, over the "
                              f"{scope} cap; {tag} is closed as incomplete.")
            guard_item(item_id, allowed, test)
            wait = MIN_INTERVAL_S[arm(tag)] - (time.time() - last[0])
            if wait > 0:
                sleep(wait)
            entry = {"at": now(), "key": k, "kind": kind if retries == 0 else "retry", "request_sha256": rhash,
                     "fields_dropped": sorted(drop), **({"retry": retries} if retries else {})}
            rid_ = f"{entry['at']}|{k}"
            _append(SPEND, {"at": entry["at"], "event": "reserved", "id": rid_, "model": tag, "key": k,
                            "cost_usd": round(reserve, 10)})
            _append(JOURNAL, entry)
            es.append(entry)
            journal[k] = es
            sends += 1
            requested_at = now()
            t0 = last[0] = time.time()
            try:
                resp, headers = transport.send(body)
            except BaseException as exc:          # noqa: BLE001 - every failure is classified and recorded
                verdict, message = classify(exc, tag)
                he = exc if isinstance(exc, CloudError) else None
                if verdict == "moderated":
                    r = moderated_record(tag, version, item_id, rhash, he, requested_at, time.time() - t0, drop)
                    c = cost(tag, r["usage"])
                    record_spend({"event": "answer", "key": k, "request_id": he.request_id, "prompt_tokens": 0,
                                  "completion_tokens": 0, "reasoning_tokens": 0, "reported_cost_usd": None,
                                  "cost_usd": round(c, 10), "reservation": rid_, "note": "moderation flag"})
                    _append(CACHE_FILE, r)
                    st["cache"][k] = r
                    done += 1
                    break
                if verdict == "lost" or (he and he.in_200 and he.usage):
                    amount = cost(tag, he.usage) if he and he.usage else reserve
                    record_spend({"event": "charged", "key": k, "request_id": he.request_id if he else None,
                                  "cost_usd": round(amount, 10), "reservation": rid_,
                                  "note": "usage of an answer that could not be used" if he and he.usage
                                  else "worst case of a request that may have been billed"})
                else:
                    _append(SPEND, {"at": now(), "event": "released", "id": rid_})
                if verdict != "lost":
                    err = {"at": now(), "key": k, "kind": verdict, "status": he.status if he else None,
                           "error_type": he.etype if he else None, "message": message,
                           "request_id": he.request_id if he else None, "server_date": he.date if he else None,
                           "in_200": bool(he and he.in_200)}
                    _append(JOURNAL, err)
                    es.append(err)
                if verdict == "rejected" and he is not None:
                    field = first_request_drop(he, tag, [x for v in journal.values() for x in v])
                    if field and field not in drop:
                        log_event({"event": "field_dropped", "model": tag, "field": field, "message": message[:300]})
                        drop.append(field)
                        kind, retries = "send", 0
                        continue
                if verdict == "transient":
                    errors += 1
                    if retries < len(RETRY_DELAYS_S) and errors < MAX_ERRORS_PER_MODEL:
                        delay = min(max(RETRY_DELAYS_S[retries], he.retry_after or 0), RETRY_AFTER_MAX_S)
                        retries += 1
                        log_event({"event": "transient_wait", "model": tag, "key": k, "status": he.status,
                                   "seconds": delay})
                        sleep(delay)
                        continue
                stop(k, verdict, message, he)
                if verdict == "quota":
                    raise Refusal(f"{tag}: the free models' daily request limit was reached at {k}; nothing is "
                                  f"dropped. Resume after the daily reset with --run {tag}.") from None
                if verdict == "lost":
                    raise Refusal(f"interrupted at {k} ({message[:200]}); it may have been served and is charged. "
                                  f"Nothing is dropped; resume with --run {tag} --resend-unanswered.") from None
                detail = f"HTTP {he.status} {he.etype}: {message[:200]}" if he and he.status else message[:200]
                raise Refusal(f"stopped at {k}: {VERDICT_WORDS[verdict]} ({detail}). The request was not served and "
                              f"keeps its sends; nothing is dropped. Resume with --run {tag} once the cause is "
                              "fixed.") from None
            r = answer_record(tag, version, item_id, body, rhash, resp, headers, requested_at, time.time() - t0, drop)
            c = cost(tag, r["usage"])
            provider_ok = r["provider"] in (None, ENDPOINT[tag]["provider"])
            if not model_matches(tag, r["response_model"]) or not provider_ok:
                record_spend({"event": "charged", "key": k, "request_id": r["generation_id"], "cost_usd": round(c, 10),
                              "note": f"answer served by {r['response_model']} at {r['provider']}",
                              "reservation": rid_})
                _append(JOURNAL, {"at": now(), "key": k, "kind": "serving_mismatch",
                                  "model_answered": r["response_model"], "provider": r["provider"]})
                close_model(tag, f"served by {r['response_model']} at {r['provider']}")
                raise Refusal(f"the API answered {k} with {r['response_model']} at {r['provider']}; {tag} is closed "
                              "as incomplete.")
            record_spend({"event": "answer", "key": k, "request_id": r["generation_id"],
                          "prompt_tokens": r["usage"]["prompt_tokens"],
                          "completion_tokens": r["usage"]["completion_tokens"],
                          "reasoning_tokens": r["reasoning_tokens"], "reported_cost_usd": r["usage"].get("cost"),
                          "cost_usd": round(c, 10), "reservation": rid_})
            _append(CACHE_FILE, r)
            st["cache"][k] = r
            done += 1
            print(f"[{tag} {version} {split}] {item_id} {r['wall_s']:.1f}s finish={r['finish_reason']} "
                  f"in={r['prompt_eval_count']} out={r['eval_count']} reasoning={r['reasoning_tokens']} "
                  f"total={spend_state()['computed']:.4f} USD", flush=True)
            if r["prompt_eval_count"] > input_upper(body) and tag not in FREE:
                close_model(tag, f"{k}: billed input above the bound the cap relies on")
                raise Refusal(f"{k}: {r['prompt_eval_count']} input tokens billed, above the bound "
                              f"{input_upper(body)} the cap relies on; {tag} is closed as incomplete.")
            break
    expected = expected_keys(tag, sets)
    missing = expected - set(st["cache"])
    if not missing:
        log_event({"event": "model_finished", "model": tag, "answers": len(expected)})
        write_model_total(tag)
        print(f"{tag}: all {len(expected)} answers are in ({spend_state()['computed']:.4f} USD so far). Commit and "
              "publish results/models/cloud/ before the next model.")
    else:
        if not all(resend_served(journal.get(k, [])) for k in missing):
            raise Refusal(f"{tag}: a missing request has not had its one resend; nothing was closed.")
        close_model(tag, f"{len(missing)} request(s) unanswered after their one resend")
    return {"model": tag, "new_answers": done, "missing": len(missing),
            "spent_usd": round(spend_state()["computed"], 6)}


def spent() -> float:
    return spend_state()["committed"]


def guard_item(item_id, allowed, test) -> None:
    X.guard(item_id, allowed, test)


def close_incomplete(tag: str) -> str:
    acquire_lock()
    try:
        reconcile()
        st = state()
        if st["status"].get(tag):
            raise Refusal(f"{tag} is already closed ({st['status'][tag]}).")
        run_log = _jsonl(RUN_LOG)
        if not any(e.get("model") == tag and e.get("event") in ("run_started", "run_stopped") for e in run_log):
            raise Refusal(f"{tag} has not started; there is nothing to close.")
        if any(e.get("event") == "budget_stop" and e.get("model") == tag for e in run_log):
            reason = "stopped by the cap"
        elif error_count(st["journal"], tag) >= MAX_ERRORS_PER_MODEL:
            reason = "error-answer limit reached"
        else:
            last = repeated(tag, run_log, st["journal"])
            if last is None:
                raise Refusal(f"{tag} cannot be closed: none of the conditions of section 9 holds (a budget stop, the "
                              "error-answer limit, the same rejection or unavailability at least 24 hours apart by "
                              f"the server's dates). Resume with --run {tag}.")
            reason = (f"{VERDICT_WORDS[last['verdict']]}, observed at least 24 hours apart "
                      f"({last.get('key') or tag}: {last.get('reason')})")
        return close_model(tag, reason)
    finally:
        release_lock()


# ------------------------------------------------------------------ the checks of `make check`
def verify(records=None) -> list[str]:
    """Amendment 3, section 13. Reads files only."""
    problems = []
    if MODELS != FREE_QUEUE + PAID_QUEUE or set(REFERENCE_ONLY) - set(MODELS) or set(ENDPOINT) != set(MODELS):
        problems.append("the model list, its queues, reference rows or endpoints differ from Amendment 3")
    if any(t in X.DIGESTS or t in H.MODELS for t in MODELS):
        problems.append("a model of this branch is also a model of Amendment 1 or 2")
    for p in sorted(DIR.rglob("*")) if DIR.exists() else []:
        if p.is_file() and KEY_PATTERN.search(p.read_text(encoding="utf-8", errors="replace")):
            problems.append(f"{p.relative_to(DIR)} contains something shaped like an API key")
    journal, cache, spend, run_log = _jsonl(JOURNAL), _jsonl(CACHE_FILE), _jsonl(SPEND), _jsonl(RUN_LOG)
    for name, entries in (("sent.jsonl", journal), ("run_log.jsonl", run_log), ("spend.jsonl", spend)):
        times = [H._seconds(e["at"]) for e in entries if e.get("at")]
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
        if min(H._seconds(e["at"]) for e in journal) <= H._seconds(reg["code_published_at"]):
            problems.append("the first request was journaled before the branch's code was published")
    except Refusal as e:
        problems.append(str(e))
    sent = H.by_key(journal)
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
                streak = 0
            elif e.get("kind") == "retry":
                streak += 1
                if streak > len(RETRY_DELAYS_S):
                    problems.append(f"journal: {k} was retried more often than the rules allow")
                    break
    for tag in MODELS:
        if error_count(journal, tag) > MAX_ERRORS_PER_MODEL:
            problems.append(f"journal: {tag} has more error answers than the rules allow")
    answers = {e["key"]: e for e in spend if e.get("event") == "answer"}
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
        if r.get("moderated") is None and (not model_matches(tag, r.get("response_model"))
                                           or r.get("provider") not in (None, ENDPOINT[tag]["provider"])):
            problems.append(f"cache: {k} was answered by {r.get('response_model')} at {r.get('provider')}")
        if item in test or item not in records:
            problems.append(f"cache: {k} concerns a held-out or unknown complaint")
            continue
        body = build_body(tag, version, records[item], visible_narrative(records[item]["narrative"]),
                          r.get("fields_dropped") or ())
        if r.get("request_sha256") != request_hash(body) or r.get("endpoint_tag") != ENDPOINT[tag]["tag"]:
            problems.append(f"cache: {k} was not sent as registered")
        if set(r.get("fields_dropped") or ()) - set(DROPPABLE):
            problems.append(f"cache: {k} left out a field the rules do not allow to leave out")
        if (r.get("eval_count") or 0) > MAX_TOKENS:
            problems.append(f"cache: {k} reports more output tokens than max_tokens")
        before = [e for e in sent.get(k, []) if e.get("kind") in ("send", "resend", "retry")
                  and e["at"] <= r["requested_at"] and e.get("request_sha256") == r["request_sha256"]]
        if not before:
            problems.append(f"cache: {k} has no journal entry before its request")
        s = answers.get(k)
        if s is None:
            problems.append(f"spend: {k} has no spend entry")
        elif abs(s["cost_usd"] - cost(tag, r["usage"])) > 1e-9:
            problems.append(f"spend: {k} records {s['cost_usd']} USD, the usage gives {cost(tag, r['usage'])}")
    running = 0.0
    for e in spend:
        if e.get("event") in ("answer", "charged"):
            running += e["cost_usd"]
            if abs(e["running_total_usd"] - running) > 1e-5:
                problems.append(f"spend: the running total at {e.get('key')} is {e['running_total_usd']}, the "
                                f"entries give {running:.6f}")
    if spend_state(spend)["committed"] > CAP_USD + 1e-9:
        problems.append("spend: the computed spend and open reservations are over the 15 USD cap")
    for tag in MODELS:
        if committed_for(tag, spend) > SUB_CAP_USD + 1e-9:
            problems.append(f"spend: {tag} is over its 5 USD cap")
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true")
    g.add_argument("--estimate", action="store_true")
    g.add_argument("--status", action="store_true")
    g.add_argument("--run", metavar="MODEL")
    g.add_argument("--close-incomplete", metavar="MODEL")
    g.add_argument("--break-stale-lock", action="store_true")
    ap.add_argument("--resend-unanswered", action="store_true")
    a = ap.parse_args()
    try:
        if a.verify:
            problems = verify()
            if problems:
                sys.exit("cloud branch check failed:\n- " + "\n- ".join(problems))
            print("cloud branch check passed")
        elif a.estimate:
            from .data import load_records
            sets, _ = X.item_sets()
            records, st = load_records(), state()
            rows = [estimate(t, sets, records, st["cache"]) for t in MODELS]
            print(json.dumps({"cap_usd": CAP_USD, "sub_cap_usd": SUB_CAP_USD, "models": rows,
                              **{f: round(sum(r[f] for r in rows), 4) for f in ("projected_usd", "worst_case_usd",
                                                                                "basis_usd")}}, indent=1))
        elif a.status:
            print(json.dumps(account_readings(OpenRouterTransport()), indent=1))
        elif a.break_stale_lock:
            print(break_stale_lock())
        elif a.run:
            print(json.dumps(run_model(a.run, resend_unanswered=a.resend_unanswered)))
        else:
            print(f"{a.close_incomplete} closed: {close_incomplete(a.close_incomplete)}")
    except SystemExit:
        raise
    except BaseException as exc:                 # noqa: BLE001 - no raw traceback, nothing that could hold the key
        sys.exit(f"cloud branch stopped: {scrub(f'{type(exc).__name__}: {exc}')[:300]}")


if __name__ == "__main__":
    main()
